# v3 method definitions, built on the batched engine (core_v3.py).
#
# Channels: dense location (lambda=0), sparse location (max over
# lambda>0), dispersion/variance (one-sided CUSUM on huberized Z^2),
# and dependence (Phase-I-centered, huberized lag-1 cross products,
# two-sided CUSUM).
#
# Combination: power-enhancement-style per-channel offsets, estimated
# on a held-out profiling batch disjoint from calibration/evaluation.

import numpy as np
from core_v3 import (
    huberize, standardize_batch, location_cusum_batch, location_cusum_Q_batch,
    onesided_cusum_Q_batch, location_profile_moments, onesided_profile_moments,
    build_lambda_grid, generate_batch,
)

DEFAULT_HP_V3 = dict(
    kappa_loc=0.12,
    huber_c=2.0,
    kappa_scale=1.15,
    c_scale=9.0,
    kappa_dep=0.12,
    c_dep=9.0,
    n_lambda_loc=7,
    n_lambda_scale=7,
    n_lambda_dep=5,
    lambda_max_mult_loc=4.0,
    lambda_max_mult_scale=4.0,
    lambda_max_mult_dep=4.0,
    q_dense_offset=0.30,
    q_sparse_offset=0.99,
    q_scale_offset=0.99,
    q_dep_offset=0.99,
    burn_in=30,
    min_nonzero=2000,
)


def dependence_features_batch(Z, n_train, c_dep):
    """Phase-I-centered, huberized lag-1 cross products. Z: (B,T,p) -> (B,T,p-1)."""
    W = Z[:, :, :-1] * Z[:, :, 1:]
    baseline = np.median(W[:, :n_train, :], axis=1)
    Wc = huberize(W - baseline[:, None, :], c_dep)
    return Wc


def standard_scores_batch(X, n_train, kappa=0.15):
    Z = standardize_batch(X, n_train, robust=False)
    C = location_cusum_batch(Z, kappa, n_train)
    return np.linalg.norm(C, axis=2)


def huber_scores_batch(X, n_train, kappa=0.12, huber_c=2.0):
    Z = standardize_batch(X, n_train, robust=True)
    Z = huberize(Z, huber_c)
    C = location_cusum_batch(Z, kappa, n_train)
    return np.linalg.norm(C, axis=2)


def armc_v1_scores_batch(X, n_train, kappa=0.12, huber_c=2.0, top_frac=0.20):
    """v1 ablation baseline: naive max(dense, sparse) with NO null calibration."""
    Z = standardize_batch(X, n_train, robust=True)
    Z = huberize(Z, huber_c)
    C = location_cusum_batch(Z, kappa, n_train)
    B, T, p = Z.shape
    m = max(1, int(np.ceil(p * top_frac)))
    dense = np.linalg.norm(C, axis=2) / np.sqrt(p)
    part = np.partition(C, -m, axis=2)[:, :, -m:]
    sparse = np.sqrt(np.mean(part ** 2, axis=2))
    return np.maximum(dense, sparse)


class NullProfileV3:
    def __init__(self, channels):
        self.channels = channels   # dict: name -> dict(lam, mu, sd, valid, offset)


def _profile_channel(Q_like_stat_fn, X, n_train, hp_kappa, hp_c, lam_max_mult, n_lambda,
                      stat_start, min_nonzero, kind):
    """kind: 'location' (two-sided, dense+sparse) or 'onesided' (scale/dependence)."""
    raise NotImplementedError


def fit_profile_v3(scenario, B_profile, T, p, tau, n_train, rho, seed, hp=DEFAULT_HP_V3,
                    channels=("dense", "sparse", "scale", "dependence"),
                    **scen_kwargs):
    """Null profiling: lambda grids, pooled null moments, and
    power-enhancement offsets for the requested channels, from one
    null batch."""
    rng = np.random.default_rng(seed)
    X, _ = generate_batch(B_profile, T, p, tau, scenario, rng, rho=rho,
                           no_change=True, **scen_kwargs)
    Z = standardize_batch(X, n_train, robust=True)
    Zh = huberize(Z, hp["huber_c"])

    stat_start = min(n_train + hp["burn_in"], T - 1)
    prof = {}

    need_loc = ("dense" in channels) or ("sparse" in channels)
    if need_loc:
        C_probe = location_cusum_batch(Zh[:, :min(T, stat_start + 60), :], hp["kappa_loc"], n_train)
        sigma_loc = np.std(C_probe[:, stat_start:, :])
        lam_loc = build_lambda_grid(sigma_loc, hp["n_lambda_loc"], hp["lambda_max_mult_loc"])
        mu_loc, sd_loc, valid_loc = location_profile_moments(
            Zh, hp["kappa_loc"], lam_loc, n_train, stat_start, hp["min_nonzero"])
        prof["_lam_loc"] = lam_loc
        prof["_mu_loc"] = mu_loc
        prof["_sd_loc"] = sd_loc
        prof["_valid_loc"] = valid_loc

    if "scale" in channels:
        V = np.minimum(Z ** 2, hp["c_scale"])
        S_probe = onesided_cusum_Q_batch(V[:, :min(T, stat_start + 60), :],
                                          hp["kappa_scale"], np.array([0.0]), n_train)
        Bc, Tc, mC = V.shape
        S_pos = np.zeros((Bc, mC))
        levels = []
        for t in range(n_train, min(T, stat_start + 60)):
            S_pos = np.maximum(0.0, S_pos + V[:, t, :] - hp["kappa_scale"])
            if t >= stat_start:
                levels.append(S_pos.copy())
        sigma_scale = np.std(np.stack(levels)) if levels else 1.0
        lam_scale = build_lambda_grid(sigma_scale, hp["n_lambda_scale"], hp["lambda_max_mult_scale"])
        mu_scale, sd_scale, valid_scale = onesided_profile_moments(
            V, hp["kappa_scale"], lam_scale, n_train, stat_start, hp["min_nonzero"])
        prof["_lam_scale"] = lam_scale
        prof["_mu_scale"] = mu_scale
        prof["_sd_scale"] = sd_scale
        prof["_valid_scale"] = valid_scale

    if "dependence" in channels:
        Wc = dependence_features_batch(Z, n_train, hp["c_dep"])
        Bc, Tc, mC = Wc.shape
        C_pos = np.zeros((Bc, mC)); C_neg = np.zeros((Bc, mC))
        levels = []
        for t in range(n_train, min(T, stat_start + 60)):
            zz = Wc[:, t, :]
            C_pos = np.maximum(0.0, C_pos + zz - hp["kappa_dep"])
            C_neg = np.minimum(0.0, C_neg + zz + hp["kappa_dep"])
            if t >= stat_start:
                levels.append(np.maximum(C_pos, -C_neg).copy())
        sigma_dep = np.std(np.stack(levels)) if levels else 1.0
        lam_dep = build_lambda_grid(sigma_dep, hp["n_lambda_dep"], hp["lambda_max_mult_dep"])
        mu_dep, sd_dep, valid_dep = location_profile_moments(
            Wc, hp["kappa_dep"], lam_dep, n_train, stat_start, hp["min_nonzero"])
        prof["_lam_dep"] = lam_dep
        prof["_mu_dep"] = mu_dep
        prof["_sd_dep"] = sd_dep
        prof["_valid_dep"] = valid_dep

    # component scores on the profiling batch, for offsets
    comp = _raw_components(X, n_train, prof, hp, channels)
    offsets = {}
    for name, vals in comp.items():
        finite = vals[np.isfinite(vals)]
        q = hp.get(f"q_{name}_offset", 0.99)
        offsets[name] = float(np.quantile(finite, q)) if finite.size else 0.0
    prof["_offsets"] = offsets
    prof["_channels"] = channels
    return prof


def _raw_components(X, n_train, prof, hp, channels):
    """Returns dict name -> flattened standardized component values over
    t >= n_train, for offset fitting."""
    out = {}
    need_loc = ("dense" in channels) or ("sparse" in channels)
    if need_loc:
        Z = standardize_batch(X, n_train, robust=True)
        Zh = huberize(Z, hp["huber_c"])
        Q_loc = location_cusum_Q_batch(Zh, hp["kappa_loc"], prof["_lam_loc"], n_train)
        R_loc = (Q_loc - prof["_mu_loc"]) / prof["_sd_loc"]
        R_loc[:, :, ~prof["_valid_loc"]] = -np.inf
        if "dense" in channels:
            out["dense"] = R_loc[:, n_train:, 0].ravel()
        if "sparse" in channels and R_loc.shape[2] > 1:
            out["sparse"] = R_loc[:, n_train:, 1:].max(axis=2).ravel()
    if "scale" in channels:
        Z = standardize_batch(X, n_train, robust=True)
        V = np.minimum(Z ** 2, hp["c_scale"])
        Q_scale = onesided_cusum_Q_batch(V, hp["kappa_scale"], prof["_lam_scale"], n_train)
        R_scale = (Q_scale - prof["_mu_scale"]) / prof["_sd_scale"]
        R_scale[:, :, ~prof["_valid_scale"]] = -np.inf
        out["scale"] = R_scale[:, n_train:, :].max(axis=2).ravel()
    if "dependence" in channels:
        Z = standardize_batch(X, n_train, robust=True)
        Wc = dependence_features_batch(Z, n_train, hp["c_dep"])
        Q_dep = location_cusum_Q_batch(Wc, hp["kappa_dep"], prof["_lam_dep"], n_train)
        R_dep = (Q_dep - prof["_mu_dep"]) / prof["_sd_dep"]
        R_dep[:, :, ~prof["_valid_dep"]] = -np.inf
        out["dependence"] = R_dep[:, n_train:, :].max(axis=2).ravel()
    return out


def component_scores(X, n_train, prof, hp):
    """Per-timestep component scores, shape (B,T) each, t<n_train set
    to -inf. Returns dict name -> (B,T) array."""
    channels = prof["_channels"]
    out = {}
    Z = standardize_batch(X, n_train, robust=True)
    if ("dense" in channels) or ("sparse" in channels):
        Zh = huberize(Z, hp["huber_c"])
        Q_loc = location_cusum_Q_batch(Zh, hp["kappa_loc"], prof["_lam_loc"], n_train)
        R_loc = (Q_loc - prof["_mu_loc"]) / prof["_sd_loc"]
        R_loc[:, :, ~prof["_valid_loc"]] = -np.inf
        if "dense" in channels:
            d = R_loc[:, :, 0].copy()
            d[:, :n_train] = -np.inf
            out["dense"] = d
        if "sparse" in channels and R_loc.shape[2] > 1:
            s = R_loc[:, :, 1:].max(axis=2)
            s[:, :n_train] = -np.inf
            out["sparse"] = s
    if "scale" in channels:
        V = np.minimum(Z ** 2, hp["c_scale"])
        Q_scale = onesided_cusum_Q_batch(V, hp["kappa_scale"], prof["_lam_scale"], n_train)
        R_scale = (Q_scale - prof["_mu_scale"]) / prof["_sd_scale"]
        R_scale[:, :, ~prof["_valid_scale"]] = -np.inf
        sc = R_scale.max(axis=2)
        sc[:, :n_train] = -np.inf
        out["scale"] = sc
    if "dependence" in channels:
        Wc = dependence_features_batch(Z, n_train, hp["c_dep"])
        Q_dep = location_cusum_Q_batch(Wc, hp["kappa_dep"], prof["_lam_dep"], n_train)
        R_dep = (Q_dep - prof["_mu_dep"]) / prof["_sd_dep"]
        R_dep[:, :, ~prof["_valid_dep"]] = -np.inf
        dep = R_dep.max(axis=2)
        dep[:, :n_train] = -np.inf
        out["dependence"] = dep
    return out


def combined_scores(X, n_train, prof, hp, use_channels=None):
    """Power-enhancement combination: max over (component - its own
    offset), restricted to use_channels if given, else all channels."""
    comps = component_scores(X, n_train, prof, hp)
    channels = use_channels if use_channels is not None else list(comps.keys())
    stacked = None
    for name in channels:
        if name not in comps:
            continue
        val = comps[name] - prof["_offsets"].get(name, 0.0)
        stacked = val[None] if stacked is None else np.concatenate([stacked, val[None]], axis=0)
    combined = stacked.max(axis=0)
    return combined, comps


def get_detection_times_batch(scores, threshold, n_train):
    """scores: (B,T). Returns array of first-exceedance index or -1 if none, shape (B,)."""
    B, T = scores.shape
    hit = scores[:, n_train:] > threshold
    any_hit = hit.any(axis=1)
    first = np.where(any_hit, hit.argmax(axis=1) + n_train, -1)
    return first


def summarize_run(detect_idx, tau, no_change):
    """Returns false_alarm_rate/power, mean/median delay (conditional
    on genuine post-tau detections), early_false_alarm rate, and
    n_detections."""
    B = detect_idx.shape[0]
    if no_change:
        alarmed = detect_idx >= 0
        return dict(false_alarm_rate=float(alarmed.mean()), power=None,
                    mean_delay=np.nan, median_delay=np.nan,
                    early_false_alarm=np.nan, n_detections=0, n=B)
    early = (detect_idx >= 0) & (detect_idx < tau)
    genuine = detect_idx >= tau
    delays = detect_idx[genuine] - tau
    power = float(genuine.mean())
    return dict(false_alarm_rate=None, power=power,
                mean_delay=float(delays.mean()) if delays.size else np.nan,
                median_delay=float(np.median(delays)) if delays.size else np.nan,
                early_false_alarm=float(early.mean()),
                n_detections=int(genuine.sum()), n=B)


def se_prop(phat, n):
    return float(np.sqrt(max(phat * (1 - phat), 0.0) / n)) if n > 0 else np.nan


# Alternative combination rule: puts every (channel, lambda) cell on a
# common empirical-null p-value scale and combines via max of
# -log10(p), with no per-channel offsets.
def _build_pvalue_grid(pooled_vals, n_quantiles=4000):
    qs = np.linspace(0.0, 1.0, n_quantiles)
    return np.quantile(pooled_vals, qs)


def _pvalue_from_grid(observed, grid_vals):
    n = len(grid_vals)
    idx = np.searchsorted(grid_vals, observed, side="right")
    frac_le = idx / n
    p_upper = np.clip(1.0 - frac_le + 1.0 / n, 1.0 / n, 1.0)
    return p_upper


def fit_profile_v3_pvalue(scenario, B_profile, T, p, tau, n_train, rho, seed, hp=DEFAULT_HP_V3,
                           channels=("dense", "sparse", "scale", "dependence"),
                           n_quantiles=4000, **scen_kwargs):
    """Profiling for the p-value combination rule: same lambda grids as
    fit_profile_v3, but stores a pooled quantile grid of the null Q
    distribution per lambda instead of mean/std."""
    rng = np.random.default_rng(seed)
    X, _ = generate_batch(B_profile, T, p, tau, scenario, rng, rho=rho,
                           no_change=True, **scen_kwargs)
    Z = standardize_batch(X, n_train, robust=True)
    Zh = huberize(Z, hp["huber_c"])
    stat_start = min(n_train + hp["burn_in"], T - 1)

    prof = {"_channels": channels, "_pvalue": True}

    if ("dense" in channels) or ("sparse" in channels):
        C_probe = location_cusum_batch(Zh[:, :min(T, stat_start + 60), :], hp["kappa_loc"], n_train)
        sigma_loc = np.std(C_probe[:, stat_start:, :])
        lam_loc = build_lambda_grid(sigma_loc, hp["n_lambda_loc"], hp["lambda_max_mult_loc"])
        Q_loc = location_cusum_Q_batch(Zh, hp["kappa_loc"], lam_loc, n_train)
        pooled = Q_loc[:, stat_start:, :].reshape(-1, len(lam_loc))
        grids_loc = [_build_pvalue_grid(pooled[:, k], n_quantiles) for k in range(len(lam_loc))]
        prof["_lam_loc"] = lam_loc
        prof["_grids_loc"] = grids_loc

    if "scale" in channels:
        V = np.minimum(Z ** 2, hp["c_scale"])
        Bc, Tc, mC = V.shape
        S_pos = np.zeros((Bc, mC)); levels = []
        for t in range(n_train, min(T, stat_start + 60)):
            S_pos = np.maximum(0.0, S_pos + V[:, t, :] - hp["kappa_scale"])
            if t >= stat_start:
                levels.append(S_pos.copy())
        sigma_scale = np.std(np.stack(levels)) if levels else 1.0
        lam_scale = build_lambda_grid(sigma_scale, hp["n_lambda_scale"], hp["lambda_max_mult_scale"])
        Q_scale = onesided_cusum_Q_batch(V, hp["kappa_scale"], lam_scale, n_train)
        pooled = Q_scale[:, stat_start:, :].reshape(-1, len(lam_scale))
        grids_scale = [_build_pvalue_grid(pooled[:, k], n_quantiles) for k in range(len(lam_scale))]
        prof["_lam_scale"] = lam_scale
        prof["_grids_scale"] = grids_scale

    if "dependence" in channels:
        Wc = dependence_features_batch(Z, n_train, hp["c_dep"])
        Bc, Tc, mC = Wc.shape
        C_pos = np.zeros((Bc, mC)); C_neg = np.zeros((Bc, mC)); levels = []
        for t in range(n_train, min(T, stat_start + 60)):
            zz = Wc[:, t, :]
            C_pos = np.maximum(0.0, C_pos + zz - hp["kappa_dep"])
            C_neg = np.minimum(0.0, C_neg + zz + hp["kappa_dep"])
            if t >= stat_start:
                levels.append(np.maximum(C_pos, -C_neg).copy())
        sigma_dep = np.std(np.stack(levels)) if levels else 1.0
        lam_dep = build_lambda_grid(sigma_dep, hp["n_lambda_dep"], hp["lambda_max_mult_dep"])
        Q_dep = location_cusum_Q_batch(Wc, hp["kappa_dep"], lam_dep, n_train)
        pooled = Q_dep[:, stat_start:, :].reshape(-1, len(lam_dep))
        grids_dep = [_build_pvalue_grid(pooled[:, k], n_quantiles) for k in range(len(lam_dep))]
        prof["_lam_dep"] = lam_dep
        prof["_grids_dep"] = grids_dep

    return prof


def component_pscores(X, n_train, prof, hp):
    """Same as component_scores, but each channel's value is -log10(p)
    from the empirical null p-value transform."""
    channels = prof["_channels"]
    out = {}
    Z = standardize_batch(X, n_train, robust=True)
    if ("dense" in channels) or ("sparse" in channels):
        Zh = huberize(Z, hp["huber_c"])
        Q_loc = location_cusum_Q_batch(Zh, hp["kappa_loc"], prof["_lam_loc"], n_train)
        B, T, K = Q_loc.shape
        NL = -np.log10(np.stack(
            [_pvalue_from_grid(Q_loc[:, :, k], prof["_grids_loc"][k]) for k in range(K)], axis=2))
        NL[:, :n_train, :] = -np.inf
        if "dense" in channels:
            out["dense"] = NL[:, :, 0]
        if "sparse" in channels and K > 1:
            out["sparse"] = NL[:, :, 1:].max(axis=2)
    if "scale" in channels:
        V = np.minimum(Z ** 2, hp["c_scale"])
        Q_scale = onesided_cusum_Q_batch(V, hp["kappa_scale"], prof["_lam_scale"], n_train)
        B, T, K = Q_scale.shape
        NL = -np.log10(np.stack(
            [_pvalue_from_grid(Q_scale[:, :, k], prof["_grids_scale"][k]) for k in range(K)], axis=2))
        NL[:, :n_train, :] = -np.inf
        out["scale"] = NL.max(axis=2)
    if "dependence" in channels:
        Wc = dependence_features_batch(Z, n_train, hp["c_dep"])
        Q_dep = location_cusum_Q_batch(Wc, hp["kappa_dep"], prof["_lam_dep"], n_train)
        B, T, K = Q_dep.shape
        NL = -np.log10(np.stack(
            [_pvalue_from_grid(Q_dep[:, :, k], prof["_grids_dep"][k]) for k in range(K)], axis=2))
        NL[:, :n_train, :] = -np.inf
        out["dependence"] = NL.max(axis=2)
    return out


def combined_pscores(X, n_train, prof, hp, use_channels=None):
    comps = component_pscores(X, n_train, prof, hp)
    channels = use_channels if use_channels is not None else list(comps.keys())
    stacked = None
    for name in channels:
        if name not in comps:
            continue
        val = comps[name]
        stacked = val[None] if stacked is None else np.concatenate([stacked, val[None]], axis=0)
    combined = stacked.max(axis=0)
    return combined, comps


# Hierarchical combination: dense and sparse are combined symmetrically
# as the location family; scale and dependence are combined
# symmetrically as the dispersion family. The two family scores are
# then combined with an asymmetric offset (location generous, dispersion
# strict).
HIER_Q_WITHIN = 0.90      # within-family offset (dense vs sparse; scale vs dependence)
HIER_Q_LOC_OUTER = 0.30   # location family outer offset
HIER_Q_DISP_OUTER = 0.99  # dispersion family outer offset


def fit_profile_v3_hier(scenario, B_profile, T, p, tau, n_train, rho, seed, hp=DEFAULT_HP_V3,
                         channels=("dense", "sparse", "scale", "dependence"), **scen_kwargs):
    """Profiling for the hierarchical combination: within-family channel
    profile and offsets, then family scores and outer offsets, all from
    one profiling batch."""
    hp_within = dict(hp)
    hp_within["q_dense_offset"] = HIER_Q_WITHIN
    hp_within["q_sparse_offset"] = HIER_Q_WITHIN
    hp_within["q_scale_offset"] = HIER_Q_WITHIN
    hp_within["q_dep_offset"] = HIER_Q_WITHIN

    prof = fit_profile_v3(scenario, B_profile, T, p, tau, n_train, rho, seed,
                           hp=hp_within, channels=channels, **scen_kwargs)
    prof["_hp_within"] = hp_within

    has_loc = ("dense" in channels) or ("sparse" in channels)
    has_disp = ("scale" in channels) or ("dependence" in channels)

    rng = np.random.default_rng(seed)
    X, _ = generate_batch(B_profile, T, p, tau, scenario, rng, rho=rho,
                           no_change=True, **scen_kwargs)
    loc_p, disp_p = _hier_family_scores(X, n_train, prof, hp_within, channels)

    a_loc = 0.0
    if has_loc:
        v = loc_p[:, n_train:].ravel()
        v = v[np.isfinite(v)]
        a_loc = float(np.quantile(v, HIER_Q_LOC_OUTER)) if v.size else 0.0
    a_disp = 0.0
    if has_disp:
        v = disp_p[:, n_train:].ravel()
        v = v[np.isfinite(v)]
        a_disp = float(np.quantile(v, HIER_Q_DISP_OUTER)) if v.size else 0.0

    prof["_a_loc"] = a_loc
    prof["_a_disp"] = a_disp
    prof["_has_loc"] = has_loc
    prof["_has_disp"] = has_disp
    return prof


def _hier_family_scores(X, n_train, prof, hp_within, channels):
    """Returns (loc_score, disp_score), each (B,T); -inf where the
    family has no channels requested."""
    comps = component_scores(X, n_train, prof, hp_within)
    off = prof["_offsets"]
    B, T = X.shape[0], X.shape[1]
    loc_parts = []
    if "dense" in comps:
        loc_parts.append(comps["dense"] - off.get("dense", 0.0))
    if "sparse" in comps:
        loc_parts.append(comps["sparse"] - off.get("sparse", 0.0))
    loc = np.maximum.reduce(loc_parts) if loc_parts else np.full((B, T), -np.inf)

    disp_parts = []
    if "scale" in comps:
        disp_parts.append(comps["scale"] - off.get("scale", 0.0))
    if "dependence" in comps:
        disp_parts.append(comps["dependence"] - off.get("dependence", 0.0))
    disp = np.maximum.reduce(disp_parts) if disp_parts else np.full((B, T), -np.inf)
    return loc, disp


def combined_scores_hier(X, n_train, prof, hp):
    """Hierarchical v3 scoring. Returns (final_score, comps_dict), with
    comps_dict holding the raw per-channel scores plus '_loc'/'_disp'
    family scores."""
    channels = prof["_channels"]
    hp_within = prof.get("_hp_within", hp)
    comps = component_scores(X, n_train, prof, hp_within)
    loc, disp = _hier_family_scores(X, n_train, prof, hp_within, channels)
    final = np.maximum(loc - prof["_a_loc"], disp - prof["_a_disp"])
    comps = dict(comps)
    comps["_loc"] = loc
    comps["_disp"] = disp
    return final, comps
