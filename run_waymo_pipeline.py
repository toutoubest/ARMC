# Loads Waymo scenarios via waymo_pb_lite, builds SDC residuals, and
# fits/scores ARMC v3.1 on the held-out episodes.
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/claude/robust_cusum_v2")
sys.path.insert(0, "/home/claude/waymo_proto")

from core_v3 import (
    standardize_batch,
    location_cusum_batch,
    location_profile_moments,
    onesided_profile_moments,
    build_lambda_grid,
    huberize,
)
from methods_v3 import (
    HIER_Q_WITHIN,
    HIER_Q_LOC_OUTER,
    HIER_Q_DISP_OUTER,
    get_detection_times_batch,
)
from methods_v3_1 import (
    DEFAULT_HP_V31,
    robust_scale_feature,
    dependence_features_batch_v31,
    _raw_components_v31,
    _hier_family_scores_v31,
    combined_scores_v31_hier,
)
from waymo_pb_lite import iter_tfrecords, parse_scenario

print("ARMC v3.1 loaded successfully.")
print("Frozen c_disp =", DEFAULT_HP_V31["c_disp"])

WAYMO_FILE = (
    "/mnt/user-data/uploads/Downloads/13th waymo Paper/"
    "uncompressed_scenario_validation_validation.tfrecord-00000-of-00150"
)


def scenario_to_sdc_residuals(scenario):
    idx = scenario["sdc_track_index"]
    tracks = scenario["tracks"]
    if idx is None or idx < 0 or idx >= len(tracks):
        return None

    states = tracks[idx]
    if len(states) < 10:
        return None

    ts = np.asarray(scenario["timestamps_seconds"], dtype=float)
    if len(ts) != len(states):
        return None

    x = np.array([s["center_x"] for s in states], dtype=float)
    y = np.array([s["center_y"] for s in states], dtype=float)
    vx = np.array([s["velocity_x"] for s in states], dtype=float)
    vy = np.array([s["velocity_y"] for s in states], dtype=float)
    valid = np.array([bool(s["valid"]) for s in states], dtype=bool)

    rows = []
    for t in range(2, len(states)):
        if not (valid[t] and valid[t - 1]):
            return None
        dt = ts[t] - ts[t - 1]
        if not np.isfinite(dt) or dt <= 0:
            return None
        pred_x = x[t - 1] + vx[t - 1] * dt
        pred_y = y[t - 1] + vy[t - 1] * dt
        rx = x[t] - pred_x
        ry = y[t] - pred_y
        rvx = vx[t] - vx[t - 1]
        rvy = vy[t] - vy[t - 1]
        row = [rx, ry, rvx, rvy]
        if not np.all(np.isfinite(row)):
            return None
        rows.append(row)

    R = np.asarray(rows, dtype=float)
    if R.ndim != 2 or R.shape[1] != 4:
        return None
    return R


def load_waymo_residual_episodes(tfrecord_path, max_episodes=300):
    episodes = []
    scenario_ids = []
    seen = 0
    for raw in iter_tfrecords(tfrecord_path):
        scenario = parse_scenario(raw)
        seen += 1
        R = scenario_to_sdc_residuals(scenario)
        if R is None:
            continue
        episodes.append(R)
        scenario_ids.append(scenario["scenario_id"])
        if len(episodes) >= max_episodes:
            break
    if len(episodes) == 0:
        raise RuntimeError("No usable residual episodes extracted.")
    common_T = min(r.shape[0] for r in episodes)
    episodes = np.stack([r[:common_T] for r in episodes], axis=0)
    return episodes, scenario_ids, seen


t0 = time.time()
X, scenario_ids, n_seen = load_waymo_residual_episodes(WAYMO_FILE, max_episodes=300)
print(f"\n[load took {time.time()-t0:.1f}s]")
print("Raw scenarios inspected:", n_seen)
print("Usable residual episodes:", X.shape[0])
print("Residual sequence length T:", X.shape[1])
print("Residual dimension p:", X.shape[2])
print("\nX shape =", X.shape)
print("\nFirst 5 residual rows from first scenario:")
print(X[0, :5, :])

np.savez_compressed(
    "/home/claude/waymo_proto/waymo_residuals_shard00000.npz",
    X=X,
    scenario_ids=np.asarray(scenario_ids),
)
print("\nSaved residual data to waymo_residuals_shard00000.npz")


def fit_real_v31(baseline, calibration, n_train=20, alpha=0.05):
    hp = dict(DEFAULT_HP_V31)

    Xb = np.asarray(baseline, dtype=float)
    Xc = np.asarray(calibration, dtype=float)

    if Xb.ndim != 3:
        raise ValueError("baseline must have shape (B,T,p)")
    if Xc.ndim != 3:
        raise ValueError("calibration must have shape (B,T,p)")
    if Xb.shape[1] != Xc.shape[1]:
        raise ValueError("baseline and calibration must have same T")
    if Xb.shape[2] != Xc.shape[2]:
        raise ValueError("baseline and calibration must have same p")
    if Xb.shape[0] < 20 or Xc.shape[0] < 20:
        raise ValueError("Too few baseline/calibration episodes.")

    B, T, p = Xb.shape
    if n_train + hp["burn_in"] >= T - 5:
        raise ValueError(f"T={T} too short for n_train={n_train} and burn_in={hp['burn_in']}.")

    c_disp = hp["c_disp"]

    hp_within = dict(hp)
    hp_within["q_dense_offset"] = HIER_Q_WITHIN
    hp_within["q_sparse_offset"] = HIER_Q_WITHIN
    hp_within["q_scale_offset"] = HIER_Q_WITHIN
    hp_within["q_dep_offset"] = HIER_Q_WITHIN

    stat_start = n_train + hp["burn_in"]

    Z = standardize_batch(Xb, n_train, robust=True)
    Zh = huberize(Z, hp["huber_c"])

    prof = {
        "_channels": ("dense", "sparse", "scale", "dependence"),
        "_hp_within": hp_within,
        "_c_disp": c_disp,
    }

    probe_end = min(T, stat_start + 60)

    # location channel
    C_probe = location_cusum_batch(Zh[:, :probe_end, :], hp["kappa_loc"], n_train)
    sigma_loc = np.std(C_probe[:, stat_start:, :])
    if not np.isfinite(sigma_loc) or sigma_loc <= 0:
        sigma_loc = 1.0
    lam_loc = build_lambda_grid(sigma_loc, hp["n_lambda_loc"], hp["lambda_max_mult_loc"])
    mu_loc, sd_loc, valid_loc = location_profile_moments(
        Zh, hp["kappa_loc"], lam_loc, n_train, stat_start, hp["min_nonzero"]
    )
    prof["_lam_loc"] = lam_loc
    prof["_mu_loc"] = mu_loc
    prof["_sd_loc"] = sd_loc
    prof["_valid_loc"] = valid_loc

    # scale channel
    V = robust_scale_feature(Z, c_disp)
    S_pos = np.zeros((B, p))
    levels = []
    for t in range(n_train, probe_end):
        S_pos = np.maximum(0.0, S_pos + V[:, t, :] - hp["kappa_scale"])
        if t >= stat_start:
            levels.append(S_pos.copy())
    sigma_scale = np.std(np.stack(levels)) if levels else 1.0
    if not np.isfinite(sigma_scale) or sigma_scale <= 0:
        sigma_scale = 1.0
    lam_scale = build_lambda_grid(sigma_scale, hp["n_lambda_scale"], hp["lambda_max_mult_scale"])
    mu_scale, sd_scale, valid_scale = onesided_profile_moments(
        V, hp["kappa_scale"], lam_scale, n_train, stat_start, hp["min_nonzero"]
    )
    prof["_lam_scale"] = lam_scale
    prof["_mu_scale"] = mu_scale
    prof["_sd_scale"] = sd_scale
    prof["_valid_scale"] = valid_scale

    # dependence channel
    Wc = dependence_features_batch_v31(Z, n_train, c_disp)
    m = Wc.shape[2]
    C_pos = np.zeros((B, m))
    C_neg = np.zeros((B, m))
    levels = []
    for t in range(n_train, probe_end):
        zz = Wc[:, t, :]
        C_pos = np.maximum(0.0, C_pos + zz - hp["kappa_dep"])
        C_neg = np.minimum(0.0, C_neg + zz + hp["kappa_dep"])
        if t >= stat_start:
            levels.append(np.maximum(C_pos, -C_neg).copy())
    sigma_dep = np.std(np.stack(levels)) if levels else 1.0
    if not np.isfinite(sigma_dep) or sigma_dep <= 0:
        sigma_dep = 1.0
    lam_dep = build_lambda_grid(sigma_dep, hp["n_lambda_dep"], hp["lambda_max_mult_dep"])
    mu_dep, sd_dep, valid_dep = location_profile_moments(
        Wc, hp["kappa_dep"], lam_dep, n_train, stat_start, hp["min_nonzero"]
    )
    prof["_lam_dep"] = lam_dep
    prof["_mu_dep"] = mu_dep
    prof["_sd_dep"] = sd_dep
    prof["_valid_dep"] = valid_dep

    # within-family offsets
    channels = prof["_channels"]
    comp = _raw_components_v31(Xb, n_train, prof, hp_within, channels, c_disp)
    offsets = {}
    for name, vals in comp.items():
        vals = vals[np.isfinite(vals)]
        q = hp_within.get(f"q_{name}_offset", HIER_Q_WITHIN)
        offsets[name] = float(np.quantile(vals, q)) if vals.size else 0.0
    prof["_offsets"] = offsets

    # outer family offsets
    loc, disp = _hier_family_scores_v31(Xb, n_train, prof, hp_within, channels)
    loc_vals = loc[:, n_train:].ravel()
    loc_vals = loc_vals[np.isfinite(loc_vals)]
    disp_vals = disp[:, n_train:].ravel()
    disp_vals = disp_vals[np.isfinite(disp_vals)]
    if loc_vals.size == 0:
        raise RuntimeError("No finite location scores.")
    if disp_vals.size == 0:
        raise RuntimeError("No finite dispersion scores.")
    prof["_a_loc"] = float(np.quantile(loc_vals, HIER_Q_LOC_OUTER))
    prof["_a_disp"] = float(np.quantile(disp_vals, HIER_Q_DISP_OUTER))
    prof["_has_loc"] = True
    prof["_has_disp"] = True

    # threshold calibration
    cal_scores, _ = combined_scores_v31_hier(Xc, n_train, prof, hp)
    maxima = np.max(cal_scores[:, n_train:], axis=1)
    if not np.all(np.isfinite(maxima)):
        raise RuntimeError("Non-finite calibration maxima encountered.")
    threshold = float(np.quantile(maxima, 1.0 - alpha))

    return prof, threshold


print("\nReal-data v3.1 fitter defined.")

B_total = X.shape[0]
if B_total < 150:
    raise ValueError(f"Need at least ~150 episodes. Only got {B_total}.")

n_base = int(B_total * 0.40)
n_cal = int(B_total * 0.40)

baseline = X[:n_base]
calibration = X[n_base : n_base + n_cal]
observed = X[n_base + n_cal :]

print("Total episodes:", B_total)
print("baseline:", baseline.shape)
print("calibration:", calibration.shape)
print("observed:", observed.shape)

N_TRAIN_REAL = 20
prof_real, threshold_real = fit_real_v31(baseline, calibration, n_train=N_TRAIN_REAL, alpha=0.05)

print("\n" + "=" * 40)
print("REAL WAYMO ARMC v3.1")
print("=" * 40)
print("threshold =", threshold_real)
print("\nValid lambda cells:")
print("location   =", prof_real["_valid_loc"])
print("scale      =", prof_real["_valid_scale"])
print("dependence =", prof_real["_valid_dep"])

scores_real, components_real = combined_scores_v31_hier(observed, N_TRAIN_REAL, prof_real, DEFAULT_HP_V31)
alarm_times = get_detection_times_batch(scores_real, threshold_real, N_TRAIN_REAL)
alarm_rate = np.mean(alarm_times >= 0)

print("\nHeld-out observed episodes:", observed.shape[0])
print("Number of alarms:", np.sum(alarm_times >= 0))
print("Alarm proportion:", round(alarm_rate, 4))
print("\nFirst 20 alarm times:")
print(alarm_times[:20])

obs_start = n_base + n_cal
obs_end = obs_start + observed.shape[0]
summary = pd.DataFrame(
    {
        "episode": np.arange(observed.shape[0]),
        "scenario_id": scenario_ids[obs_start:obs_end],
        "alarm_time": alarm_times,
        "max_global_score": np.max(scores_real[:, N_TRAIN_REAL:], axis=1),
    }
)
summary["alarm"] = summary["alarm_time"] >= 0

pd.set_option("display.width", 120)
print("\nSummary (head 20):")
print(summary.head(20))
print("\nSummary stats:")
print(summary[["alarm", "max_global_score"]].describe())

summary.to_csv("/home/claude/waymo_proto/waymo_holdout_summary_shard00000.csv", index=False)
np.savez_compressed(
    "/home/claude/waymo_proto/waymo_scores_shard00000.npz",
    scores_real=scores_real,
    **{f"comp_{k}": v for k, v in components_real.items()},
    threshold=threshold_real,
    n_train=N_TRAIN_REAL,
)
print("\nSaved summary CSV and component scores.")
