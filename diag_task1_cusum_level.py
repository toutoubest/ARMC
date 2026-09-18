# CUSUM level and max-aggregated statistic under heavy tails/contamination.
import numpy as np
from core_v3 import generate_batch, standardize_batch
from core_v3 import onesided_cusum_Q_batch, location_cusum_Q_batch, build_lambda_grid

B, T, p, TAU, N_TRAIN, RHO = 3000, 400, 20, 200, 100, 0.3
KAPPA_SCALE = 1.15
KAPPA_DEP = 0.12
N_LAMBDA = 7

for c_label, c in [("current_v3 (c=3.0)", 3.0), ("candidate (c=1.345)", 1.345)]:
    print(f"\n=== dispersion channel, {c_label} ===")
    level_by_scen = {}
    for scenario in ["gaussian_dense", "t3_dense", "contaminated_dense"]:
        rng = np.random.default_rng(750000 + hash(scenario) % 1000)
        X, _ = generate_batch(B, T, p, TAU, scenario, rng, rho=RHO, no_change=True)
        Z = standardize_batch(X, N_TRAIN, robust=True)
        V = np.clip(Z, -c, c) ** 2
        stat_start = N_TRAIN + 30
        # sigma probe for the lambda grid, same convention as fit_profile_v3
        Bc, Tc, mC = V.shape
        S_pos = np.zeros((Bc, mC)); levels = []
        for t in range(N_TRAIN, min(T, stat_start + 60)):
            S_pos = np.maximum(0.0, S_pos + V[:, t, :] - KAPPA_SCALE)
            if t >= stat_start:
                levels.append(S_pos.copy())
        sigma = np.std(np.stack(levels)) if levels else 1.0
        lam = build_lambda_grid(sigma, N_LAMBDA, 4.0)
        Q = onesided_cusum_Q_batch(V, KAPPA_SCALE, lam, N_TRAIN)
        pooled = Q[:, stat_start:, :].reshape(-1, len(lam))
        # lambda=0 is the dense-equivalent cell
        mean_lam0 = float(pooled[:, 0].mean())
        max_over_lam = pooled.max(axis=1)
        mean_max = float(max_over_lam.mean())
        q99_max = float(np.quantile(max_over_lam, 0.99))
        level_by_scen[scenario] = dict(mean_lam0=mean_lam0, mean_max=mean_max, q99_max=q99_max)
        print(f"  {scenario:20s} CUSUM-level lambda=0 mean={mean_lam0:8.3f}  "
              f"max-over-lambda mean={mean_max:8.3f}  q99={q99_max:8.3f}")
    g = level_by_scen["gaussian_dense"]
    for scen in ["t3_dense", "contaminated_dense"]:
        r = level_by_scen[scen]
        print(f"  ratio {scen} vs gaussian: lambda=0 mean ratio={r['mean_lam0']/g['mean_lam0']:.3f}  "
              f"max-over-lambda mean ratio={r['mean_max']/g['mean_max']:.3f}  "
              f"q99 ratio={r['q99_max']/g['q99_max']:.3f}")
