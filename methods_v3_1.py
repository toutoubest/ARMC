# ARMC v3.1: v3 with a smaller Huber clip radius (c_disp=1.345 instead
# of the effective c=3.0) on the scale and dependence channels, applied
# by clipping before squaring/multiplying rather than clipping the
# squared value or raw product. Location channel and the hierarchical
# combination are unchanged from methods_v3.py.
import numpy as np
from core_v3 import (
    standardize_batch, location_cusum_batch, location_cusum_Q_batch,
    onesided_cusum_Q_batch, location_profile_moments, onesided_profile_moments,
    build_lambda_grid, generate_batch, huberize,
)
from methods_v3 import (
    DEFAULT_HP_V3, HIER_Q_WITHIN, HIER_Q_LOC_OUTER, HIER_Q_DISP_OUTER,
)

C_DISP_DEFAULT = 1.345  # Huber 95%-efficiency constant

DEFAULT_HP_V31 = dict(DEFAULT_HP_V3)
DEFAULT_HP_V31["c_disp"] = C_DISP_DEFAULT


def robust_scale_feature(Z, c_disp):
    """Scale channel feature: clip then square."""
    return np.clip(Z, -c_disp, c_disp) ** 2


def dependence_features_batch_v31(Z, n_train, c_disp):
    """Dependence channel feature: clip each factor, then multiply,
    Phase-I median-centered."""
    U = np.clip(Z, -c_disp, c_disp)
    W = U[:, :, :-1] * U[:, :, 1:]
    baseline = np.median(W[:, :n_train, :], axis=1)
    return W - baseline[:, None, :]


def fit_profile_v31_hier(scenario, B_profile, T, p, tau, n_train, rho, seed, hp=DEFAULT_HP_V31,
                          channels=("dense", "sparse", "scale", "dependence"), **scen_kwargs):
    """Same structure as methods_v3.fit_profile_v3_hier, with the v3.1
    scale/dependence features; dense/sparse channel is unchanged."""
    c_disp = hp.get("c_disp", C_DISP_DEFAULT)
    hp_within = dict(hp)
    hp_within["q_dense_offset"] = HIER_Q_WITHIN
    hp_within["q_sparse_offset"] = HIER_Q_WITHIN
    hp_within["q_scale_offset"] = HIER_Q_WITHIN
    hp_within["q_dep_offset"] = HIER_Q_WITHIN

    rng = np.random.default_rng(seed)
    X, _ = generate_batch(B_profile, T, p, tau, scenario, rng, rho=rho,
                           no_change=True, **scen_kwargs)
    Z = standardize_batch(X, n_train, robust=True)
    Zh = huberize(Z, hp["huber_c"])
    stat_start = min(n_train + hp["burn_in"], T - 1)
    prof = {"_channels": channels, "_hp_within": hp_within, "_c_disp": c_disp}

    need_loc = ("dense" in channels) or ("sparse" in channels)
    if need_loc:
        C_probe = location_cusum_batch(Zh[:, :min(T, stat_start + 60), :], hp["kappa_loc"], n_train)
        sigma_loc = np.std(C_probe[:, stat_start:, :])
        lam_loc = build_lambda_grid(sigma_loc, hp["n_lambda_loc"], hp["lambda_max_mult_loc"])
        mu_loc, sd_loc, valid_loc = location_profile_moments(
            Zh, hp["kappa_loc"], lam_loc, n_train, stat_start, hp["min_nonzero"])
        prof["_lam_loc"] = lam_loc; prof["_mu_loc"] = mu_loc
        prof["_sd_loc"] = sd_loc; prof["_valid_loc"] = valid_loc

    if "scale" in channels:
        V = robust_scale_feature(Z, c_disp)
        Bc, Tc, mC = V.shape
        S_pos = np.zeros((Bc, mC)); levels = []
        for t in range(n_train, min(T, stat_start + 60)):
            S_pos = np.maximum(0.0, S_pos + V[:, t, :] - hp["kappa_scale"])
            if t >= stat_start:
                levels.append(S_pos.copy())
        sigma_scale = np.std(np.stack(levels)) if levels else 1.0
        lam_scale = build_lambda_grid(sigma_scale, hp["n_lambda_scale"], hp["lambda_max_mult_scale"])
        mu_scale, sd_scale, valid_scale = onesided_profile_moments(
            V, hp["kappa_scale"], lam_scale, n_train, stat_start, hp["min_nonzero"])
        prof["_lam_scale"] = lam_scale; prof["_mu_scale"] = mu_scale
        prof["_sd_scale"] = sd_scale; prof["_valid_scale"] = valid_scale

    if "dependence" in channels:
        Wc = dependence_features_batch_v31(Z, n_train, c_disp)
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
        mu_dep, sd_dep, valid_dep = location_profile_moments(
            Wc, hp["kappa_dep"], lam_dep, n_train, stat_start, hp["min_nonzero"])
        prof["_lam_dep"] = lam_dep; prof["_mu_dep"] = mu_dep
        prof["_sd_dep"] = sd_dep; prof["_valid_dep"] = valid_dep

    # within-family offsets from component scores on the profiling batch
    comp = _raw_components_v31(X, n_train, prof, hp_within, channels, c_disp)
    offsets = {}
    for name, vals in comp.items():
        finite = vals[np.isfinite(vals)]
        q = hp_within.get(f"q_{name}_offset", 0.90)
        offsets[name] = float(np.quantile(finite, q)) if finite.size else 0.0
    prof["_offsets"] = offsets

    # outer family offsets (location vs dispersion)
    has_loc = ("dense" in channels) or ("sparse" in channels)
    has_disp = ("scale" in channels) or ("dependence" in channels)
    loc_p, disp_p = _hier_family_scores_v31(X, n_train, prof, hp_within, channels)
    a_loc = 0.0
    if has_loc:
        v = loc_p[:, n_train:].ravel(); v = v[np.isfinite(v)]
        a_loc = float(np.quantile(v, HIER_Q_LOC_OUTER)) if v.size else 0.0
    a_disp = 0.0
    if has_disp:
        v = disp_p[:, n_train:].ravel(); v = v[np.isfinite(v)]
        a_disp = float(np.quantile(v, HIER_Q_DISP_OUTER)) if v.size else 0.0
    prof["_a_loc"] = a_loc; prof["_a_disp"] = a_disp
    prof["_has_loc"] = has_loc; prof["_has_disp"] = has_disp
    return prof


def _raw_components_v31(X, n_train, prof, hp, channels, c_disp):
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
        V = robust_scale_feature(Z, c_disp)
        Q_scale = onesided_cusum_Q_batch(V, hp["kappa_scale"], prof["_lam_scale"], n_train)
        R_scale = (Q_scale - prof["_mu_scale"]) / prof["_sd_scale"]
        R_scale[:, :, ~prof["_valid_scale"]] = -np.inf
        out["scale"] = R_scale[:, n_train:, :].max(axis=2).ravel()
    if "dependence" in channels:
        Z = standardize_batch(X, n_train, robust=True)
        Wc = dependence_features_batch_v31(Z, n_train, c_disp)
        Q_dep = location_cusum_Q_batch(Wc, hp["kappa_dep"], prof["_lam_dep"], n_train)
        R_dep = (Q_dep - prof["_mu_dep"]) / prof["_sd_dep"]
        R_dep[:, :, ~prof["_valid_dep"]] = -np.inf
        out["dependence"] = R_dep[:, n_train:, :].max(axis=2).ravel()
    return out


def component_scores_v31(X, n_train, prof, hp):
    channels = prof["_channels"]
    c_disp = prof.get("_c_disp", C_DISP_DEFAULT)
    out = {}
    Z = standardize_batch(X, n_train, robust=True)
    if ("dense" in channels) or ("sparse" in channels):
        Zh = huberize(Z, hp["huber_c"])
        Q_loc = location_cusum_Q_batch(Zh, hp["kappa_loc"], prof["_lam_loc"], n_train)
        R_loc = (Q_loc - prof["_mu_loc"]) / prof["_sd_loc"]
        R_loc[:, :, ~prof["_valid_loc"]] = -np.inf
        if "dense" in channels:
            d = R_loc[:, :, 0].copy(); d[:, :n_train] = -np.inf
            out["dense"] = d
        if "sparse" in channels and R_loc.shape[2] > 1:
            s = R_loc[:, :, 1:].max(axis=2); s[:, :n_train] = -np.inf
            out["sparse"] = s
    if "scale" in channels:
        V = robust_scale_feature(Z, c_disp)
        Q_scale = onesided_cusum_Q_batch(V, hp["kappa_scale"], prof["_lam_scale"], n_train)
        R_scale = (Q_scale - prof["_mu_scale"]) / prof["_sd_scale"]
        R_scale[:, :, ~prof["_valid_scale"]] = -np.inf
        sc = R_scale.max(axis=2); sc[:, :n_train] = -np.inf
        out["scale"] = sc
    if "dependence" in channels:
        Wc = dependence_features_batch_v31(Z, n_train, c_disp)
        Q_dep = location_cusum_Q_batch(Wc, hp["kappa_dep"], prof["_lam_dep"], n_train)
        R_dep = (Q_dep - prof["_mu_dep"]) / prof["_sd_dep"]
        R_dep[:, :, ~prof["_valid_dep"]] = -np.inf
        dep = R_dep.max(axis=2); dep[:, :n_train] = -np.inf
        out["dependence"] = dep
    return out


def _hier_family_scores_v31(X, n_train, prof, hp_within, channels):
    comps = component_scores_v31(X, n_train, prof, hp_within)
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


def combined_scores_v31_hier(X, n_train, prof, hp):
    """Combine channel scores via the v3 hierarchical rule, using the
    v3.1 scale/dependence features."""
    channels = prof["_channels"]
    hp_within = prof.get("_hp_within", hp)
    comps = component_scores_v31(X, n_train, prof, hp_within)
    loc, disp = _hier_family_scores_v31(X, n_train, prof, hp_within, channels)
    final = np.maximum(loc - prof["_a_loc"], disp - prof["_a_disp"])
    comps = dict(comps)
    comps["_loc"] = loc
    comps["_disp"] = disp
    return final, comps
