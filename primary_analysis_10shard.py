# 10-fold leave-one-shard-out Waymo analysis for ARMC v3.1. In fold i,
# shard i is held out; the other 9 shards split by a fixed rotation into
# a 5-shard baseline pool and a 4-shard calibration pool.
import sys
import time
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "/home/claude/robust_cusum_v2")
sys.path.insert(0, "/home/claude/waymo_proto")

from load_with_speed import load_shard_with_speed
from run_waymo_pipeline import fit_real_v31
from methods_v3_1 import combined_scores_v31_hier, component_scores_v31, DEFAULT_HP_V31
from methods_v3 import get_detection_times_batch

N_TRAIN_REAL = 20
ALPHA = 0.05
N_SHARDS = 10
N_BASE_SHARDS = 5
N_CAL_SHARDS = 4

def make_rotations(n_shards, n_base, n_cal):
    rotations = []
    for h in range(n_shards):
        others = [(h + 1 + k) % n_shards for k in range(n_shards - 1)]
        base = others[:n_base]
        cal = others[n_base:n_base + n_cal]
        assert len(base) == n_base and len(cal) == n_cal
        rotations.append((base, cal, [h]))
    return rotations

ROTATIONS = make_rotations(N_SHARDS, N_BASE_SHARDS, N_CAL_SHARDS)

t0 = time.time()
shard_eps, shard_speeds, shard_sids = {}, {}, {}
for s in range(N_SHARDS):
    eps, speeds, sids = load_shard_with_speed(s)
    shard_eps[s] = eps
    shard_speeds[s] = speeds
    shard_sids[s] = sids
    print(f"shard {s}: {len(eps)} usable episodes  [{time.time()-t0:.1f}s]")

common_T = min(r.shape[0] for eps in shard_eps.values() for r in eps)
print("common_T:", common_T)


def stack_shards(shard_list):
    eps, speeds, sids = [], [], []
    for s in shard_list:
        eps.extend(shard_eps[s])
        speeds.extend(shard_speeds[s])
        sids.extend(shard_sids[s])
    X = np.stack([r[:common_T] for r in eps], axis=0)
    Sp = np.stack([sp[:common_T] for sp in speeds], axis=0)
    return X, Sp, sids


# speed-regime cutoffs
def regime_of(median_speed):
    if median_speed < 1.0:
        return "near_stationary (<1 m/s)"
    elif median_speed < 5.0:
        return "low_speed (1-5 m/s)"
    else:
        return "normal_higher_speed (>=5 m/s)"


all_rows = []
fold_summaries = []

for fold_i, (base_sh, cal_sh, held_sh) in enumerate(ROTATIONS):
    Xb, _, _ = stack_shards(base_sh)
    Xc, _, _ = stack_shards(cal_sh)
    Xo, Spo, sids_o = stack_shards(held_sh)

    prof, thr = fit_real_v31(Xb, Xc, n_train=N_TRAIN_REAL, alpha=ALPHA)
    scores, _ = combined_scores_v31_hier(Xo, N_TRAIN_REAL, prof, prof["_hp_within"])
    alarm_times = get_detection_times_batch(scores, thr, N_TRAIN_REAL)
    comp = component_scores_v31(Xo, N_TRAIN_REAL, prof, prof["_hp_within"])
    offsets = prof["_offsets"]

    n_o = Xo.shape[0]
    med_speed = np.median(Spo[:, N_TRAIN_REAL:], axis=1)

    for i in range(n_o):
        at = int(alarm_times[i])
        row = dict(
            fold=fold_i, baseline_shards=base_sh, cal_shards=cal_sh, held_out_shard=held_sh[0],
            scenario_id=sids_o[i], alarm_time=at, alarm=at >= 0,
            max_global_score=float(np.max(scores[i, N_TRAIN_REAL:])),
            threshold=thr, median_monitored_speed_mps=med_speed[i],
            speed_regime=regime_of(med_speed[i]),
        )
        if at >= 0:
            dense_c = comp["dense"][i, at] - offsets["dense"]
            sparse_c = comp["sparse"][i, at] - offsets["sparse"]
            scale_c = comp["scale"][i, at] - offsets["scale"]
            dep_c = comp["dependence"][i, at] - offsets["dependence"]
            within = dict(dense=dense_c, sparse=sparse_c, scale=scale_c, dependence=dep_c)
            driver = max(within, key=within.get)
            row.update(dense=dense_c, sparse=sparse_c, scale=scale_c, dependence=dep_c,
                       driving_component=driver)
        else:
            row.update(dense=np.nan, sparse=np.nan, scale=np.nan, dependence=np.nan,
                       driving_component=None)
        all_rows.append(row)

    n_alarm = int((alarm_times >= 0).sum())
    fold_summaries.append(dict(fold=fold_i, baseline_shards=base_sh, cal_shards=cal_sh,
                                held_out_shard=held_sh[0], n_baseline=Xb.shape[0], n_calibration=Xc.shape[0],
                                n_held_out=n_o, threshold=thr, n_alarms=n_alarm, alarm_rate=n_alarm / n_o))
    print(f"fold {fold_i}: base={base_sh} cal={cal_sh} held={held_sh} "
          f"threshold={thr:.3f} alarms={n_alarm}/{n_o} ({n_alarm/n_o*100:.2f}%)")

full = pd.DataFrame(all_rows)
folds_df = pd.DataFrame(fold_summaries)
full.to_csv("/home/claude/waymo_proto/primary10_full_holdout_table.csv", index=False)
folds_df.to_csv("/home/claude/waymo_proto/primary10_fold_summary.csv", index=False)

print(f"\nTotal held-out scenarios across all {N_SHARDS} folds (each scenario held out exactly once): {len(full)}")


def wilson_ci(k, n, z=1.959963985):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (center - half, center + half)


# overall pooled alarm rate
k_total = int(full["alarm"].sum())
n_total = len(full)
lo, hi = wilson_ci(k_total, n_total)
print(f"\n=== OVERALL POOLED ALARM RATE (10-fold leave-one-shard-out, N={n_total}) ===")
print(f"alarms = {k_total}/{n_total} = {k_total/n_total*100:.2f}%  "
      f"Wilson 95% CI [{lo*100:.2f}%, {hi*100:.2f}%]  (nominal target alpha={ALPHA*100:.0f}%)")

pval_two = stats.binomtest(k_total, n_total, ALPHA, alternative="two-sided").pvalue
print(f"Exact binomial test vs nominal alpha={ALPHA}: two-sided p = {pval_two:.4f}")

# per-regime breakdown
print("\n=== ALARM RATE BY SPEED REGIME ===")
regime_rows = []
for regime, g in full.groupby("speed_regime"):
    k = int(g["alarm"].sum())
    n = len(g)
    lo, hi = wilson_ci(k, n)
    regime_rows.append(dict(speed_regime=regime, n=n, n_alarms=k, alarm_rate=k / n,
                             ci_lo=lo, ci_hi=hi))
    print(f"  {regime:28s} n={n:5d}  alarms={k:3d}  rate={k/n*100:5.2f}%  "
          f"95% CI [{lo*100:.2f}%, {hi*100:.2f}%]")
regime_df = pd.DataFrame(regime_rows)
regime_df.to_csv("/home/claude/waymo_proto/primary10_regime_breakdown.csv", index=False)

ct = pd.crosstab(full["speed_regime"], full["alarm"])
print("\nContingency table (speed_regime x alarm):")
print(ct)
chi2, pchi, dof, _ = stats.chi2_contingency(ct)
print(f"\nChi-square test of independence: chi2={chi2:.3f}, dof={dof}, p={pchi:.4f}")

ns_mask = full["speed_regime"] == "near_stationary (<1 m/s)"
a = int((ns_mask & full["alarm"]).sum())
b = int((ns_mask & ~full["alarm"]).sum())
c = int((~ns_mask & full["alarm"]).sum())
d = int((~ns_mask & ~full["alarm"]).sum())
odds, pfisher = stats.fisher_exact([[a, b], [c, d]])
print(f"\nNear-stationary vs rest 2x2: [[{a},{b}],[{c},{d}]]  "
      f"odds ratio={odds:.2f}  Fisher exact p={pfisher:.4f}")
print(f"near-stationary alarm rate: {a/(a+b)*100:.2f}%  vs rest: {c/(c+d)*100:.2f}%")

# channel attribution among alarmed episodes
alarmed = full[full["alarm"]]
print(f"\n=== CHANNEL ATTRIBUTION among {len(alarmed)} alarmed episodes (all {N_SHARDS} folds) ===")
print(alarmed["driving_component"].value_counts())
print("\nby speed regime:")
ct2 = pd.crosstab(alarmed["speed_regime"], alarmed["driving_component"])
print(ct2)
ct2.to_csv("/home/claude/waymo_proto/primary10_channel_attribution_by_regime.csv")

attrib_summary = alarmed["driving_component"].value_counts().rename_axis("driving_component").reset_index(name="n_alarms")
attrib_summary.to_csv("/home/claude/waymo_proto/primary10_channel_attribution.csv", index=False)

print(f"\nthreshold across folds: mean={folds_df.threshold.mean():.3f} sd={folds_df.threshold.std():.3f} "
      f"min={folds_df.threshold.min():.3f} max={folds_df.threshold.max():.3f}")
print(f"alarm_rate across folds: mean={folds_df.alarm_rate.mean()*100:.2f}% sd={folds_df.alarm_rate.std()*100:.2f}pp")

print(f"\n[total wall time: {time.time()-t0:.1f}s]")
