# (a) marginal mean/variance sanity check for correlation_shift; (b) a
# correlation-shift power curve near rho0, for v3 and v3.1.
import numpy as np, csv, time
from core_v3 import generate_batch
from production_common import HP, T, TAU, N_TRAIN, RHO, SCENARIOS
from methods_v3 import (
    DEFAULT_HP_V3, fit_profile_v3_hier, combined_scores_hier,
    get_detection_times_batch, summarize_run, se_prop,
)
from methods_v3_1 import DEFAULT_HP_V31, fit_profile_v31_hier, combined_scores_v31_hier

# ---- (a) sanity check: large-sample marginal mean/var pre vs post tau ----
p = 20
rng = np.random.default_rng(800001)
B_CHECK = 8000
X, _ = generate_batch(B_CHECK, T, p, TAU, "correlation_shift", rng, rho=0.2, no_change=False,
                       corr_rho_after=0.6)
pre = X[:, :TAU, :]
post = X[:, TAU + 50:, :]  # a bit after tau to avoid the exact boundary sample
pre_mean = pre.reshape(-1, p).mean(axis=0)
post_mean = post.reshape(-1, p).mean(axis=0)
pre_var = pre.reshape(-1, p).var(axis=0)
post_var = post.reshape(-1, p).var(axis=0)
# coordinate-averaged lag-1 autocorrelation
def lag1_corr(block):
    a = block[:, :-1, :].reshape(-1, p)
    b = block[:, 1:, :].reshape(-1, p)
    num = ((a - a.mean(0)) * (b - b.mean(0))).mean(0)
    den = a.std(0) * b.std(0)
    return (num / den).mean()

pre_lag1 = lag1_corr(pre)
post_lag1 = lag1_corr(post)

print("=== Task 8(a): correlation_shift marginal sanity check (B=8000) ===")
print(f"pre-tau  mean: min={pre_mean.min():.4f} max={pre_mean.max():.4f} mean-of-means={pre_mean.mean():.4f}")
print(f"post-tau mean: min={post_mean.min():.4f} max={post_mean.max():.4f} mean-of-means={post_mean.mean():.4f}")
print(f"pre-tau  var:  min={pre_var.min():.4f} max={pre_var.max():.4f} mean={pre_var.mean():.4f}")
print(f"post-tau var:  min={post_var.min():.4f} max={post_var.max():.4f} mean={post_var.mean():.4f}")
print(f"pre-tau  avg lag-1 autocorr (coordinate-avg): {pre_lag1:.4f}  (nominal rho0=0.2)")
print(f"post-tau avg lag-1 autocorr (coordinate-avg): {post_lag1:.4f}  (nominal rho1=0.6)")

with open("task8_correlation_sanity.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["coord", "pre_mean", "post_mean", "pre_var", "post_var"])
    for j in range(p):
        w.writerow([j, pre_mean[j], post_mean[j], pre_var[j], post_var[j]])
print("wrote task8_correlation_sanity.csv")

# ---- (b) weaker correlation-shift power curve, v3 vs v3.1 ----
print("\n=== Task 8(b): weaker correlation-shift power curve ===")
B_PROFILE, B_CAL, B_EVAL = 2000, 2000, 1500
SEED_PROFILE, SEED_CAL = 210000, 220000
SEED_EVAL_BASE = 230000
rho1_grid = [0.22, 0.25, 0.28, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]

rows = []
for label, fit_fn, score_fn, hp in [
    ("v3", fit_profile_v3_hier, combined_scores_hier, DEFAULT_HP_V3),
    ("v3.1", fit_profile_v31_hier, combined_scores_v31_hier, DEFAULT_HP_V31),
]:
    t0 = time.time()
    prof = fit_fn("correlation_shift", B_PROFILE, T, p, TAU, N_TRAIN, 0.2, seed=SEED_PROFILE,
                   hp=hp, channels=("dense", "sparse", "scale", "dependence"))
    rngc = np.random.default_rng(SEED_CAL)
    Xc, _ = generate_batch(B_CAL, T, p, TAU, "correlation_shift", rngc, rho=0.2, no_change=True)
    scc, _ = score_fn(Xc, N_TRAIN, prof, hp)
    thr = float(np.quantile(scc[:, N_TRAIN:].max(axis=1), 0.95))

    for j, rho1 in enumerate(rho1_grid):
        seed = SEED_EVAL_BASE + 11 * j
        rng = np.random.default_rng(seed)
        X1, _ = generate_batch(B_EVAL, T, p, TAU, "correlation_shift", rng, rho=0.2, no_change=False,
                                corr_rho_after=rho1)
        s1, _ = score_fn(X1, N_TRAIN, prof, hp)
        pw = summarize_run(get_detection_times_batch(s1, thr, N_TRAIN), TAU, no_change=False)
        p_ = pw["power"]
        print(f"{label:6s} rho1={rho1:.2f} power={p_:.3f} (SE={se_prop(p_,B_EVAL):.3f}) delay={pw['mean_delay']:.1f}")
        rows.append([label, rho1, p_, se_prop(p_, B_EVAL), pw["mean_delay"], pw["median_delay"], B_EVAL])
    print(f"  [{label} done in {time.time()-t0:.1f}s]")

with open("correlation_power_curve_weak.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "rho1", "power", "SE_power", "mean_delay", "median_delay", "n_eval"])
    w.writerows(rows)
print("wrote correlation_power_curve_weak.csv")
