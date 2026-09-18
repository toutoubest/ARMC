# Batched (B,T,p) data generation and CUSUM channels for ARMC v3.
# Channels: dense location, sparse location, dispersion/variance, and
# dependence (lag-1 cross products, for correlation shifts that leave
# the marginal variance unchanged).

import numpy as np


# basic utilities
def huberize(x, c):
    return np.clip(x, -c, c)


def make_covariance(p, rho):
    idx = np.arange(p)
    return rho ** np.abs(np.subtract.outer(idx, idx))


SCENARIOS = [
    "gaussian_dense",
    "gaussian_sparse",
    "t3_dense",
    "contaminated_dense",
    "variance_shift",
    "correlation_shift",
]

# scenarios whose H1 alters the covariance structure rather than the mean
COV_SCENARIOS = {"variance_shift", "correlation_shift"}


# batched data generation
def generate_batch(
    B, T, p, tau, scenario, rng,
    delta=0.7, rho=0.3,
    contamination_prob=0.05, contamination_scale=8.0,
    sparse_s=3,
    var_gamma=1.6,        # variance_shift: Sigma1 = gamma^2 * Sigma0 (same rho)
    corr_rho_after=0.6,   # correlation_shift: Sigma1 = AR1(corr_rho_after), diag still 1
    ramp_len=0,           # >0: linear ramp-in of the mean shift over this many steps (drift stress test)
    no_change=False,
):
    """Generate X of shape (B, T, p) for one scenario.

    Mean-shift scenarios: covariance is AR1(rho) throughout; mean is 0
    before tau, mu1 after. variance_shift: covariance scales by
    gamma^2 after tau, rho unchanged. correlation_shift: covariance
    goes from AR1(rho) to AR1(corr_rho_after), diagonal unchanged."""
    Sigma0 = make_covariance(p, rho)
    L0 = np.linalg.cholesky(Sigma0)

    mu1 = np.zeros(p)
    if scenario in ("gaussian_dense", "t3_dense", "contaminated_dense"):
        mu1[:] = delta / np.sqrt(p)
    elif scenario == "gaussian_sparse":
        s = min(sparse_s, p)
        mu1[:s] = delta / np.sqrt(s)
    elif scenario == "variance_shift":
        Sigma1 = (var_gamma ** 2) * make_covariance(p, rho)
        L1 = np.linalg.cholesky(Sigma1)
    elif scenario == "correlation_shift":
        Sigma1 = make_covariance(p, corr_rho_after)
        L1 = np.linalg.cholesky(Sigma1)
    elif scenario == "combo_location_dependence":
        # simultaneous dense location shift + dependence shift
        mu1[:] = delta / np.sqrt(p)
        Sigma1 = make_covariance(p, corr_rho_after)
        L1 = np.linalg.cholesky(Sigma1)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    # pre-change noise
    Zraw = rng.standard_normal((B, T, p))
    noise = Zraw @ L0.T

    if scenario == "t3_dense":
        u = rng.chisquare(df=3, size=(B, T))
        noise = noise / np.sqrt(u / 3.0)[:, :, None]
        noise = noise / np.sqrt(3.0)

    if scenario == "contaminated_dense":
        mask = rng.random((B, T)) < contamination_prob
        outlier = contamination_scale * rng.standard_normal((B, T, p))
        noise = noise + mask[:, :, None] * outlier

    X = noise

    if not no_change:
        if scenario == "combo_location_dependence":
            Zraw2 = rng.standard_normal((B, T, p))
            noise_post = Zraw2 @ L1.T + mu1[None, None, :]
            tmask = (np.arange(T) >= tau)
            X = np.where(tmask[None, :, None], noise_post, X)
        elif scenario in COV_SCENARIOS:
            Zraw2 = rng.standard_normal((B, T, p))
            noise_post = Zraw2 @ L1.T
            tmask = (np.arange(T) >= tau)
            X = np.where(tmask[None, :, None], noise_post, X)
        else:
            if ramp_len and ramp_len > 0:
                tt = np.arange(T)
                frac = np.clip((tt - tau) / float(ramp_len), 0.0, 1.0)
                add = frac[None, :, None] * mu1[None, None, :]
            else:
                tmask = (np.arange(T) >= tau).astype(float)
                add = tmask[None, :, None] * mu1[None, None, :]
            X = X + add

    return X, mu1


# Phase-I standardization
def standardize_batch(X, n_train, robust):
    X_train = X[:, :n_train, :]
    if robust:
        center = np.median(X_train, axis=1)
        mad = np.median(np.abs(X_train - center[:, None, :]), axis=1)
        scale = np.maximum(mad / 0.67448975, 1e-8)
    else:
        center = np.mean(X_train, axis=1)
        scale = np.maximum(np.std(X_train, axis=1, ddof=1), 1e-8)
    return (X - center[:, None, :]) / scale[:, None, :]


# batched coordinate-wise CUSUM channels
def location_cusum_batch(Z, kappa, n_train):
    """Two-sided coordinate CUSUM. Z: (B,T,p). Returns C: (B,T,p) >= 0."""
    B, T, p = Z.shape
    C_pos = np.zeros((B, p))
    C_neg = np.zeros((B, p))
    C = np.zeros((B, T, p))
    for t in range(n_train, T):
        z = Z[:, t, :]
        C_pos = np.maximum(0.0, C_pos + z - kappa)
        C_neg = np.minimum(0.0, C_neg + z + kappa)
        C[:, t, :] = np.maximum(C_pos, -C_neg)
    return C


def scale_cusum_batch(Z, c_scale, kappa_scale, n_train):
    """One-sided CUSUM on capped squared deviations (variance channel)."""
    B, T, p = Z.shape
    V = np.minimum(Z ** 2, c_scale)
    S_pos = np.zeros((B, p))
    S = np.zeros((B, T, p))
    for t in range(n_train, T):
        v = V[:, t, :]
        S_pos = np.maximum(0.0, S_pos + v - kappa_scale)
        S[:, t, :] = S_pos
    return S


def dependence_cusum_batch(Z, c_dep, kappa_dep, n_train):
    """One-sided CUSUM on huberized lag-1 cross products
    W_tj = Z_tj * Z_t,j+1. Returns D: (B,T,p-1) >= 0."""
    B, T, p = Z.shape
    W = huberize(Z[:, :, :-1] * Z[:, :, 1:], c_dep)
    D_pos = np.zeros((B, p - 1))
    D = np.zeros((B, T, p - 1))
    for t in range(n_train, T):
        w = W[:, t, :]
        D_pos = np.maximum(0.0, D_pos + w - kappa_dep)
        D[:, t, :] = D_pos
    return D


# multi-lambda soft-thresholded aggregation
def location_cusum_Q_batch(Z, kappa, lambdas, n_train):
    """Fuses location_cusum_batch with the multi-lambda reduction:
    Q_t(lambda) = sum_j (C_tj - lambda)_+^2, without materializing the
    full (B,T,p) C matrix. Returns Q: (B,T,K)."""
    B, T, p = Z.shape
    K = len(lambdas)
    C_pos = np.zeros((B, p))
    C_neg = np.zeros((B, p))
    Q = np.zeros((B, T, K))
    for t in range(n_train, T):
        z = Z[:, t, :]
        C_pos = np.maximum(0.0, C_pos + z - kappa)
        C_neg = np.minimum(0.0, C_neg + z + kappa)
        c = np.maximum(C_pos, -C_neg)               # (B,p)
        for k, lam in enumerate(lambdas):
            excess = np.clip(c - lam, 0.0, None)
            excess *= excess
            Q[:, t, k] = excess.sum(axis=1)
    return Q


def onesided_cusum_Q_batch(V, kappa, lambdas, n_train):
    """Fused one-sided CUSUM + multi-lambda reduction, used for both
    the scale channel (V=huberized Z^2) and the dependence channel
    (V=huberized lag-1 products). Returns Q: (B,T,K)."""
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


def location_profile_moments(Z, kappa, lambdas, n_train, stat_start, min_nonzero=2000):
    """Null-profiling companion to location_cusum_Q_batch: runs the same
    recursion but accumulates pooled null mean/std sufficient statistics
    per lambda cell instead of storing Q per (replicate, t).
    Returns mu, sd, valid, each shape (K,)."""
    B, T, p = Z.shape
    K = len(lambdas)
    C_pos = np.zeros((B, p))
    C_neg = np.zeros((B, p))
    s1 = np.zeros(K)   # sum of excess
    s2 = np.zeros(K)   # sum of excess^2
    nz = np.zeros(K)   # count of nonzero excess
    n = 0
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
                s1[k] += excess.sum()
                s2[k] += (excess * excess).sum()
                nz[k] += int((excess > 0).sum())
    mean_j = s1 / n
    var_j = np.maximum(s2 / n - mean_j ** 2, 0.0)
    mu = p * mean_j
    sd = np.sqrt(np.maximum(p * var_j, 1e-10))
    valid = nz >= min_nonzero
    return mu, sd, valid


def onesided_profile_moments(V, kappa, lambdas, n_train, stat_start, min_nonzero=2000):
    """Null-profiling companion to onesided_cusum_Q_batch (see location_profile_moments)."""
    B, T, m = V.shape
    K = len(lambdas)
    S_pos = np.zeros((B, m))
    s1 = np.zeros(K)
    s2 = np.zeros(K)
    nz = np.zeros(K)
    n = 0
    for t in range(n_train, T):
        v = V[:, t, :]
        S_pos = np.maximum(0.0, S_pos + v - kappa)
        if t >= stat_start:
            n += B * m
            for k, lam in enumerate(lambdas):
                excess = np.clip(S_pos - lam, 0.0, None)
                excess *= excess
                s1[k] += excess.sum()
                s2[k] += (excess * excess).sum()
                nz[k] += int((excess > 0).sum())
    mean_j = s1 / n
    var_j = np.maximum(s2 / n - mean_j ** 2, 0.0)
    mu = m * mean_j
    sd = np.sqrt(np.maximum(m * var_j, 1e-10))
    valid = nz >= min_nonzero
    return mu, sd, valid


def multi_lambda_Q_batch(C, lambdas):
    """C: (B,T,p) nonnegative coordinate statistics.
    Returns Q: (B,T,K), Q[b,t,k] = sum_j (C[b,t,j]-lambdas[k])_+^2."""
    B, T, p = C.shape
    K = len(lambdas)
    Q = np.empty((B, T, K))
    for k, lam in enumerate(lambdas):
        excess = np.clip(C - lam, 0.0, None)
        excess *= excess
        Q[:, :, k] = excess.sum(axis=2)
    return Q


def build_lambda_grid(sigma_C, n_lambda, max_mult):
    return sigma_C * np.linspace(0.0, max_mult, n_lambda)
