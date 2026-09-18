# Decompose the dispersion and dependence channels into stages
# (raw residual, squared/cross-product, clipped score) and compare
# Gaussian vs t3 vs contaminated Gaussian at each stage.
import numpy as np
import json
from core_v3 import generate_batch, standardize_batch, huberize

B, T, p, TAU, N_TRAIN, RHO = 4000, 300, 20, 150, 100, 0.3
rng_seed_base = 700000

results = {}

for scenario in ["gaussian_dense", "t3_dense", "contaminated_dense"]:
    rng = np.random.default_rng(rng_seed_base + hash(scenario) % 1000)
    X, _ = generate_batch(B, T, p, TAU, scenario, rng, rho=RHO, no_change=True)
    Z = standardize_batch(X, N_TRAIN, robust=True)
    Zpost = Z[:, N_TRAIN + 30:, :]

    flat = Zpost.ravel()
    stage0_q = {q: float(np.quantile(np.abs(flat), q)) for q in [0.5, 0.9, 0.95, 0.99, 0.999]}
    stage0_max = float(np.max(np.abs(flat)))

    raw_sq = flat ** 2
    stage1 = dict(mean=float(raw_sq.mean()),
                  q=[float(np.quantile(raw_sq, q)) for q in [0.9, 0.95, 0.99, 0.999]],
                  max=float(raw_sq.max()))

    def clip_sq_mean(c):
        u = np.clip(flat, -c, c)
        return float((u ** 2).mean()), float((u ** 2).std())

    stage2_c9 = clip_sq_mean(3.0)
    candidates = {}
    for c in [1.345, 1.5, 2.0, 2.5, 3.0]:
        m, s = clip_sq_mean(c)
        candidates[c] = dict(mean=m, std=s)

    W = Zpost[:, :, :-1] * Zpost[:, :, 1:]
    Wflat = W.ravel()
    dep_raw = dict(mean=float(Wflat.mean()), std=float(Wflat.std()),
                   q=[float(np.quantile(np.abs(Wflat), q)) for q in [0.9, 0.95, 0.99, 0.999]],
                   max=float(np.max(np.abs(Wflat))))

    def dep_clip_then_mult(c):
        u = np.clip(Zpost, -c, c)
        prod = (u[:, :, :-1] * u[:, :, 1:]).ravel()
        return dict(mean=float(prod.mean()), std=float(prod.std()))

    dep_candidates = {c: dep_clip_then_mult(c) for c in [1.345, 1.5, 2.0, 2.5, 3.0]}
    dep_direct_clip9 = dict(mean=float(np.clip(Wflat, -9, 9).mean()),
                             std=float(np.clip(Wflat, -9, 9).std()))

    results[scenario] = dict(
        n_cells=int(flat.size),
        stage0_abs_quantiles=stage0_q, stage0_abs_max=stage0_max,
        stage1_raw_Zsq=stage1,
        stage2_current_v3_clip_c3=dict(mean=stage2_c9[0], std=stage2_c9[1]),
        stage2b_candidate_clips=candidates,
        dependence_raw=dep_raw,
        dependence_current_v3_direct_clip9=dep_direct_clip9,
        dependence_candidate_clip_then_mult=dep_candidates,
    )
    print(f" {scenario} ")
    print(f"  |Z| quantiles (0.9/0.95/0.99/0.999): {[round(stage0_q[q],3) for q in [0.9,0.95,0.99,0.999]]}  max={stage0_max:.2f}")
    print(f"  raw Z^2 mean={stage1['mean']:.3f}  q99={stage1['q'][2]:.2f}  q999={stage1['q'][3]:.2f}  max={stage1['max']:.1f}")
    print(f"  current v3 (clip z at 3, i.e. c_scale=9) mean={stage2_c9[0]:.4f} std={stage2_c9[1]:.4f}")
    for c in [1.345, 1.5, 2.0, 2.5, 3.0]:
        print(f"    candidate c={c}: mean={candidates[c]['mean']:.4f} std={candidates[c]['std']:.4f}")
    print(f"  dependence raw W: mean={dep_raw['mean']:.4f} std={dep_raw['std']:.4f} q99={dep_raw['q'][2]:.2f} max={dep_raw['max']:.1f}")
    print(f"  dependence current v3 (direct clip at 9): mean={dep_direct_clip9['mean']:.4f} std={dep_direct_clip9['std']:.4f}")
    for c in [1.345, 1.5, 2.0, 2.5, 3.0]:
        print(f"    dep candidate (clip-then-mult) c={c}: mean={dep_candidates[c]['mean']:.4f} std={dep_candidates[c]['std']:.4f}")

print("\n RELATIVE INFLATION (t3 or contam mean / gaussian mean) ")
g = results["gaussian_dense"]
for scen in ["t3_dense", "contaminated_dense"]:
    r = results[scen]
    print(f"\n {scen} vs gaussian_dense ")
    print(f"  stage1 raw Z^2 mean ratio: {r['stage1_raw_Zsq']['mean']/g['stage1_raw_Zsq']['mean']:.3f}")
    print(f"  current v3 clip(c=3) mean ratio: {r['stage2_current_v3_clip_c3']['mean']/g['stage2_current_v3_clip_c3']['mean']:.3f}")
    for c in [1.345, 1.5, 2.0, 2.5, 3.0]:
        ratio = r['stage2b_candidate_clips'][c]['mean'] / g['stage2b_candidate_clips'][c]['mean']
        print(f"    candidate c={c} mean ratio: {ratio:.3f}")
    print(f"  dependence current v3 direct-clip mean/std ratio: mean={r['dependence_current_v3_direct_clip9']['mean']:.4f} std ratio={r['dependence_current_v3_direct_clip9']['std']/g['dependence_current_v3_direct_clip9']['std']:.3f}")
    for c in [1.345, 1.5, 2.0, 2.5, 3.0]:
        rs = r['dependence_candidate_clip_then_mult'][c]['std']
        gs = g['dependence_candidate_clip_then_mult'][c]['std']
        print(f"    dep candidate c={c} std ratio: {rs/gs:.3f}")

with open("diag_task1_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nwrote diag_task1_results.json")
