# ARMC v3.1: Adaptive Robust Multiscale CUSUM

from dataclasses import dataclass, field
import numpy as np

FROZEN_HP = dict(
    kappa_loc=0.12,        # location channel CUSUM allowance
    huber_c=2.0,           # location channel huberization clip
    kappa_scale=1.15,      # scale channel CUSUM allowance
    kappa_dep=0.12,        # dependence channel CUSUM allowance
    c_disp=1.345,          # shared clip radius for scale and dependence channels
    n_lambda_loc=7,        # location lambda-grid size
    n_lambda_scale=7,      # scale lambda-grid size
    n_lambda_dep=5,        # dependence lambda-grid size
    lambda_max_mult=4.0,   # lambda grid spans [0, 4*sigma]
    q_within=0.90,         # within-family offset quantile
    q_loc_outer=0.30,      # location-family outer offset quantile
    q_disp_outer=0.99,     # dispersion-family outer offset quantile
    burn_in=30,            # warm-up steps skipped before pooling null moments
    min_nonzero=2000,      # minimum pooled nonzero-excess count for a valid lambda cell
    alpha=0.05,            # default target false-alarm rate
)


def _huberize(x, c):
    return np.clip(x, -c, c)


def _to_batch(X, name):
    X = np.asarray(X, dtype=float)
    if X.ndim == 2:
        X = X[None, :, :]
    elif X.ndim != 3:
        raise ValueError(f"{name} must have shape (T,p) or (B,T,p), got shape {X.shape}")
    return X


def _standardize(X, n_train):
    """Per-replicate robust (median/MAD) standardization using each
    replicate's OWN first n_train steps as the in-control reference
    window. X: (B,T,p) -> Z: (B,T,p)."""
    X_train = X[:, :n_train, :]
    center = np.median(X_train, axis=1)
    mad = np.median(np.abs(X_train - center[:, None, :]), axis=1)
    scale = np.maximum(mad / 0.67448975, 1e-8)
    return (X - center[:, None, :]) / scale[:, None, :]


def _build_lambda_grid(sigma, n_lambda, max_mult):
    return sigma * np.linspace(0.0, max_mult, n_lambda)


def _location_cusum_Q(Z, kappa, lambdas, n_train):
    """Two-sided coordinate CUSUM + multi-lambda soft-threshold
    reduction, fused. Z: (B,T,p) -> Q: (B,T,K)."""
    B, T, p = Z.shape
    K = len(lambdas)
    C_pos = np.zeros((B, p)); C_neg = np.zeros((B, p))
    Q = np.zeros((B, T, K))
    for t in range(n_train, T):
        z = Z[:, t, :]
        C_pos = np.maximum(0.0, C_pos + z - kappa)
        C_neg = np.minimum(0.0, C_neg + z + kappa)
        c = np.maximum(C_pos, -C_neg)
        for k, lam in enumerate(lambdas):
            excess = np.clip(c - lam, 0.0, None)
            excess *= excess
            Q[:, t, k] = excess.sum(axis=1)
    return Q


def _onesided_cusum_Q(V, kappa, lambdas, n_train):
    """One-sided CUSUM + multi-lambda reduction, fused. Used for both
    the scale and dependence channels. V: (B,T,m) -> Q: (B,T,K)."""
    B, T, m = V.shape
    K = len(lambdas)
    S_pos = np.zeros((B, m))
    Q = np.zeros((B, T, K))
    for t in range(n_train, T):
        v = V[:, t, :]
        S_pos = np.maximum(0.0, S_pos + v - kappa)
        for k, lam in enumerate(lambdas):
            excess = np.clip(S_pos - lam, 0.0, None)
            excess *= excess
            Q[:, t, k] = excess.sum(axis=1)
    return Q


def _location_profile_moments(Z, kappa, lambdas, n_train, stat_start, min_nonzero):
    """Pooled null mean/std/validity of the location channel's Q_t(lambda),
    pooled over (replicate x coordinate x post-burn-in time window)."""
    B, T, p = Z.shape
    K = len(lambdas)
    C_pos = np.zeros((B, p)); C_neg = np.zeros((B, p))
    s1 = np.zeros(K); s2 = np.zeros(K); nz = np.zeros(K); n = 0
    for t in range(n_train, T):
        z = Z[:, t, :]
        C_pos = np.maximum(0.0, C_pos + z - kappa)
        C_neg = np.minimum(0.0, C_neg + z + kappa)
        c = np.maximum(C_pos, -C_neg)
        if t >= stat_start:
            n += B * p
            for k, lam in enumerate(lambdas):
                excess = np.clip(c - lam, 0.0, None)
                excess *= excess
                s1[k] += excess.sum(); s2[k] += (excess * excess).sum()
                nz[k] += int((excess > 0).sum())
    mean_j = s1 / n
    var_j = np.maximum(s2 / n - mean_j ** 2, 0.0)
    mu = p * mean_j
    sd = np.sqrt(np.maximum(p * var_j, 1e-10))
    return mu, sd, (nz >= min_nonzero)


def _onesided_profile_moments(V, kappa, lambdas, n_train, stat_start, min_nonzero):
    B, T, m = V.shape
    K = len(lambdas)
    S_pos = np.zeros((B, m))
    s1 = np.zeros(K); s2 = np.zeros(K); nz = np.zeros(K); n = 0
    for t in range(n_train, T):
        v = V[:, t, :]
        S_pos = np.maximum(0.0, S_pos + v - kappa)
        if t >= stat_start:
            n += B * m
            for k, lam in enumerate(lambdas):
                excess = np.clip(S_pos - lam, 0.0, None)
                excess *= excess
                s1[k] += excess.sum(); s2[k] += (excess * excess).sum()
                nz[k] += int((excess > 0).sum())
    mean_j = s1 / n
    var_j = np.maximum(s2 / n - mean_j ** 2, 0.0)
    mu = m * mean_j
    sd = np.sqrt(np.maximum(m * var_j, 1e-10))
    return mu, sd, (nz >= min_nonzero)


def _robust_scale_feature(Z, c_disp):
    """v3.1 scale feature: clip THEN square."""
    return np.clip(Z, -c_disp, c_disp) ** 2


def _dependence_feature(Z, n_train, c_disp):
    """Dependence channel feature: clip each factor, multiply, then
    Phase-I median-center."""
    U = np.clip(Z, -c_disp, c_disp)
    W = U[:, :, :-1] * U[:, :, 1:]
    baseline = np.median(W[:, :n_train, :], axis=1)
    return W - baseline[:, None, :]


@dataclass
class ARMCv31Calibration:
    """Per-channel lambda grids, null mean/std/validity, within- and
    outer-family offsets, and the alarm threshold, as fit from the
    baseline/calibration data. Pass to `score()`."""
    p: int
    n_train: int
    hp: dict
    lam_loc: np.ndarray
    mu_loc: np.ndarray
    sd_loc: np.ndarray
    valid_loc: np.ndarray
    lam_scale: np.ndarray
    mu_scale: np.ndarray
    sd_scale: np.ndarray
    valid_scale: np.ndarray
    lam_dep: np.ndarray
    mu_dep: np.ndarray
    sd_dep: np.ndarray
    valid_dep: np.ndarray
    offset_dense: float
    offset_sparse: float
    offset_scale: float
    offset_dependence: float
    a_loc: float
    a_disp: float
    threshold: float
    alpha: float


def _raw_component_scores(X, n_train, cal):
    """Per-(replicate, time) null-standardized channel scores; -inf
    before n_train or where a lambda cell is invalid. Returns dict of
    (B,T) arrays."""
    hp = cal.hp
    Z = _standardize(X, n_train)
    Zh = _huberize(Z, hp["huber_c"])

    Q_loc = _location_cusum_Q(Zh, hp["kappa_loc"], cal.lam_loc, n_train)
    R_loc = (Q_loc - cal.mu_loc) / cal.sd_loc
    R_loc[:, :, ~cal.valid_loc] = -np.inf
    dense = R_loc[:, :, 0].copy(); dense[:, :n_train] = -np.inf
    sparse = (R_loc[:, :, 1:].max(axis=2) if R_loc.shape[2] > 1
              else np.full(R_loc.shape[:2], -np.inf))
    sparse[:, :n_train] = -np.inf

    V = _robust_scale_feature(Z, hp["c_disp"])
    Q_scale = _onesided_cusum_Q(V, hp["kappa_scale"], cal.lam_scale, n_train)
    R_scale = (Q_scale - cal.mu_scale) / cal.sd_scale
    R_scale[:, :, ~cal.valid_scale] = -np.inf
    scale = R_scale.max(axis=2); scale[:, :n_train] = -np.inf

    Wc = _dependence_feature(Z, n_train, hp["c_disp"])
    Q_dep = _location_cusum_Q(Wc, hp["kappa_dep"], cal.lam_dep, n_train)
    R_dep = (Q_dep - cal.mu_dep) / cal.sd_dep
    R_dep[:, :, ~cal.valid_dep] = -np.inf
    dependence = R_dep.max(axis=2); dependence[:, :n_train] = -np.inf

    return dict(dense=dense, sparse=sparse, scale=scale, dependence=dependence)


def fit(baseline_residuals, n_train, calibration_residuals=None, alpha=0.05, hp=None):
    """Fit the ARMC v3.1 null profile and alarm threshold from a
    baseline/calibration residual set.

    Parameters
    ----------
    baseline_residuals : array-like, shape (T_cal, p) or (B_cal, T_cal, p)
        In-control residuals used to estimate the null moments and offsets.
    n_train : int
        Length of the leading in-control window used for each
        replicate's own robust standardization.
    calibration_residuals : array-like, optional
        Independent in-control data used to calibrate the alarm
        threshold; defaults to baseline_residuals if omitted.
    alpha : float, default 0.05
        Target false-alarm rate for the threshold.
    hp : dict, optional
        Override of FROZEN_HP.

    Returns
    -------
    ARMCv31Calibration
    """
    hp = dict(FROZEN_HP if hp is None else hp)
    X = _to_batch(baseline_residuals, "baseline_residuals")
    B, T, p = X.shape
    if T <= n_train + hp["burn_in"] + 10:
        raise ValueError(
            f"baseline sequence length T={T} is too short for n_train={n_train} "
            f"(+ burn_in={hp['burn_in']}); need at least ~10 more post-burn-in steps."
        )
    stat_start = n_train + hp["burn_in"]

    Z = _standardize(X, n_train)
    Zh = _huberize(Z, hp["huber_c"])

    # location channel: lambda grid + pooled null moments
    probe_end = min(T, stat_start + 60)
    C_probe = np.zeros((B, p)); C_probe_stack = []
    C_pos = np.zeros((B, p)); C_neg = np.zeros((B, p))
    for t in range(n_train, probe_end):
        z = Zh[:, t, :]
        C_pos = np.maximum(0.0, C_pos + z - hp["kappa_loc"])
        C_neg = np.minimum(0.0, C_neg + z + hp["kappa_loc"])
        if t >= stat_start:
            C_probe_stack.append(np.maximum(C_pos, -C_neg).copy())
    sigma_loc = np.std(np.stack(C_probe_stack)) if C_probe_stack else 1.0
    lam_loc = _build_lambda_grid(sigma_loc, hp["n_lambda_loc"], hp["lambda_max_mult"])
    mu_loc, sd_loc, valid_loc = _location_profile_moments(
        Zh, hp["kappa_loc"], lam_loc, n_train, stat_start, hp["min_nonzero"])

    # scale channel
    V = _robust_scale_feature(Z, hp["c_disp"])
    S_pos = np.zeros((B, p)); levels = []
    for t in range(n_train, probe_end):
        S_pos = np.maximum(0.0, S_pos + V[:, t, :] - hp["kappa_scale"])
        if t >= stat_start:
            levels.append(S_pos.copy())
    sigma_scale = np.std(np.stack(levels)) if levels else 1.0
    lam_scale = _build_lambda_grid(sigma_scale, hp["n_lambda_scale"], hp["lambda_max_mult"])
    mu_scale, sd_scale, valid_scale = _onesided_profile_moments(
        V, hp["kappa_scale"], lam_scale, n_train, stat_start, hp["min_nonzero"])

    # dependence channel
    Wc = _dependence_feature(Z, n_train, hp["c_disp"])
    C_pos = np.zeros((B, p - 1)); C_neg = np.zeros((B, p - 1)); levels = []
    for t in range(n_train, probe_end):
        w = Wc[:, t, :]
        C_pos = np.maximum(0.0, C_pos + w - hp["kappa_dep"])
        C_neg = np.minimum(0.0, C_neg + w + hp["kappa_dep"])
        if t >= stat_start:
            levels.append(np.maximum(C_pos, -C_neg).copy())
    sigma_dep = np.std(np.stack(levels)) if levels else 1.0
    lam_dep = _build_lambda_grid(sigma_dep, hp["n_lambda_dep"], hp["lambda_max_mult"])
    mu_dep, sd_dep, valid_dep = _location_profile_moments(
        Wc, hp["kappa_dep"], lam_dep, n_train, stat_start, hp["min_nonzero"])

    cal = ARMCv31Calibration(
        p=p, n_train=n_train, hp=hp,
        lam_loc=lam_loc, mu_loc=mu_loc, sd_loc=sd_loc, valid_loc=valid_loc,
        lam_scale=lam_scale, mu_scale=mu_scale, sd_scale=sd_scale, valid_scale=valid_scale,
        lam_dep=lam_dep, mu_dep=mu_dep, sd_dep=sd_dep, valid_dep=valid_dep,
        offset_dense=0.0, offset_sparse=0.0, offset_scale=0.0, offset_dependence=0.0,
        a_loc=0.0, a_disp=0.0, threshold=0.0, alpha=alpha,
    )

    # within-family offsets
    comp = _raw_component_scores(X, n_train, cal)
    offsets = {}
    for name, arr in comp.items():
        vals = arr[:, n_train:].ravel()
        vals = vals[np.isfinite(vals)]
        offsets[name] = float(np.quantile(vals, hp["q_within"])) if vals.size else 0.0
    cal.offset_dense = offsets["dense"]
    cal.offset_sparse = offsets["sparse"]
    cal.offset_scale = offsets["scale"]
    cal.offset_dependence = offsets["dependence"]

    # outer family offsets
    loc_fam = np.maximum(comp["dense"] - cal.offset_dense, comp["sparse"] - cal.offset_sparse)
    disp_fam = np.maximum(comp["scale"] - cal.offset_scale, comp["dependence"] - cal.offset_dependence)
    loc_vals = loc_fam[:, n_train:].ravel(); loc_vals = loc_vals[np.isfinite(loc_vals)]
    disp_vals = disp_fam[:, n_train:].ravel(); disp_vals = disp_vals[np.isfinite(disp_vals)]
    cal.a_loc = float(np.quantile(loc_vals, hp["q_loc_outer"])) if loc_vals.size else 0.0
    cal.a_disp = float(np.quantile(disp_vals, hp["q_disp_outer"])) if disp_vals.size else 0.0

    # threshold calibration
    Xc = _to_batch(calibration_residuals, "calibration_residuals") if calibration_residuals is not None else X
    if Xc.shape[2] != p:
        raise ValueError("calibration_residuals must have the same number of coordinates p as baseline_residuals")
    comp_c = _raw_component_scores(Xc, n_train, cal)
    loc_c = np.maximum(comp_c["dense"] - cal.offset_dense, comp_c["sparse"] - cal.offset_sparse)
    disp_c = np.maximum(comp_c["scale"] - cal.offset_scale, comp_c["dependence"] - cal.offset_dependence)
    global_c = np.maximum(loc_c - cal.a_loc, disp_c - cal.a_disp)
    per_replicate_max = np.nanmax(np.where(np.isfinite(global_c), global_c, -np.inf)[:, n_train:], axis=1)
    cal.threshold = float(np.quantile(per_replicate_max, 1.0 - alpha))
    return cal


@dataclass
class ARMCv31ScoreResult:
    """Output of `score()`. *_score arrays have shape (T,) or (B,T);
    entries before n_train are NaN. `alarm_time` is the first index
    where global_score exceeds `threshold`, or None/-1 if no alarm."""
    global_score: np.ndarray
    location_score: np.ndarray
    scale_score: np.ndarray
    dependence_score: np.ndarray
    dense_score: np.ndarray
    sparse_score: np.ndarray
    dispersion_score: np.ndarray
    threshold: float
    alarm_time: object  # int/None for a single sequence, np.ndarray[int] for a batch


def score(observed_residuals, calibration: ARMCv31Calibration):
    """Score an observed residual sequence against a fitted
    ARMCv31Calibration.

    Parameters
    ----------
    observed_residuals : array-like, shape (T,p) or (B,T,p)
        Sequence(s) to monitor. The first `calibration.n_train` steps
        are used for this sequence's own robust standardization.
    calibration : ARMCv31Calibration
        Output of `fit()`.

    Returns
    -------
    ARMCv31ScoreResult
    """
    X = _to_batch(observed_residuals, "observed_residuals")
    was_single = (np.asarray(observed_residuals).ndim == 2)
    B, T, p = X.shape
    if p != calibration.p:
        raise ValueError(f"observed_residuals has p={p} coordinates but calibration was fit with p={calibration.p}")
    n_train = calibration.n_train

    comp = _raw_component_scores(X, n_train, calibration)
    loc = np.maximum(comp["dense"] - calibration.offset_dense, comp["sparse"] - calibration.offset_sparse)
    disp = np.maximum(comp["scale"] - calibration.offset_scale, comp["dependence"] - calibration.offset_dependence)
    global_score = np.maximum(loc - calibration.a_loc, disp - calibration.a_disp)

    def _nan_before_train(arr):
        arr = arr.copy()
        arr[:, :n_train] = np.nan
        return arr

    global_score_out = _nan_before_train(global_score)
    loc_out = _nan_before_train(loc)
    scale_out = _nan_before_train(comp["scale"])
    dep_out = _nan_before_train(comp["dependence"])
    dense_out = _nan_before_train(comp["dense"])
    sparse_out = _nan_before_train(comp["sparse"])
    disp_out = _nan_before_train(disp)

    hit = global_score >= calibration.threshold
    hit[:, :n_train] = False
    any_hit = hit.any(axis=1)
    first = np.where(any_hit, hit.argmax(axis=1), -1)

    if was_single:
        alarm_time = int(first[0]) if first[0] >= 0 else None
        return ARMCv31ScoreResult(
            global_score=global_score_out[0], location_score=loc_out[0],
            scale_score=scale_out[0], dependence_score=dep_out[0],
            dense_score=dense_out[0], sparse_score=sparse_out[0], dispersion_score=disp_out[0],
            threshold=calibration.threshold, alarm_time=alarm_time,
        )
    return ARMCv31ScoreResult(
        global_score=global_score_out, location_score=loc_out,
        scale_score=scale_out, dependence_score=dep_out,
        dense_score=dense_out, sparse_score=sparse_out, dispersion_score=disp_out,
        threshold=calibration.threshold, alarm_time=first,
    )
