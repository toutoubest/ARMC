# Sparsity grid (p, s): dense-only vs dense+sparse location channels,
# matched-FAR, delta=1.0.
import csv, time
import numpy as np
from core_v3 import generate_batch
from methods_v3 import DEFAULT_HP_V3, fit_profile_v3_hier, combined_scores_hier, get_detection_times_batch, summarize_run, se_prop

T, TAU, N_TRAIN, RHO = 500, 250, 100, 0.3
DELTA = 1.0
hp = DEFAULT_HP_V3
configs = [(60, 3), (60, 5), (100, 3), (100, 5)]

rows = []
for i, (p, s) in enumerate(configs):
    # smaller batch for p=100 to limit memory use
    B_PROFILE, B_CAL, B_EVAL = (900, 900, 700) if p >= 100 else (2000, 2000, 1500)
    kwargs = dict(sparse_s=s)
    seed_profile = 510000 + 19 * i
    seed_cal = 520000 + 19 * i
    seed_fa = 530000 + 19 * i
    seed_pw = 540000 + 19 * i
    results = {}
    for chset_name, chset in [("dense_only", ("dense",)), ("dense+sparse", ("dense", "sparse"))]:
        t0 = time.time()
        prof = fit_profile_v3_hier("gaussian_sparse", B_PROFILE, T, p, TAU, N_TRAIN, RHO,
                                    seed=seed_profile, hp=hp, channels=chset, **kwargs)
        rngc = np.random.default_rng(seed_cal)
        Xc, _ = generate_batch(B_CAL, T, p, TAU, "gaussian_sparse", rngc, rho=RHO, no_change=True, **kwargs)
        scc, _ = combined_scores_hier(Xc, N_TRAIN, prof, hp)
        thr = float(np.quantile(scc[:, N_TRAIN:].max(axis=1), 0.95))

        rng0 = np.random.default_rng(seed_fa)
        X0, _ = generate_batch(B_EVAL, T, p, TAU, "gaussian_sparse", rng0, rho=RHO, no_change=True, **kwargs)
        s0, _ = combined_scores_hier(X0, N_TRAIN, prof, hp)
        fa = summarize_run(get_detection_times_batch(s0, thr, N_TRAIN), TAU, no_change=True)

        rng1 = np.random.default_rng(seed_pw)
        X1, _ = generate_batch(B_EVAL, T, p, TAU, "gaussian_sparse", rng1, rho=RHO, delta=DELTA, no_change=False, **kwargs)
        s1, _ = combined_scores_hier(X1, N_TRAIN, prof, hp)
        pw = summarize_run(get_detection_times_batch(s1, thr, N_TRAIN), TAU, no_change=False)
        results[chset_name] = (fa, pw)
        print(f"p={p} s={s} {chset_name:14s} FA={fa['false_alarm_rate']:.3f} power={pw['power']:.3f} delay={pw['mean_delay']:.1f}  [{time.time()-t0:.1f}s]")

    fa1, pw1 = results["dense_only"]; fa2, pw2 = results["dense+sparse"]
    gain = pw2["power"] - pw1["power"]
    rows.append([p, s, fa1["false_alarm_rate"], pw1["power"], se_prop(pw1["power"], B_EVAL), pw1["mean_delay"],
                 fa2["false_alarm_rate"], pw2["power"], se_prop(pw2["power"], B_EVAL), pw2["mean_delay"], gain])
    print(f"  => gain = {gain:+.3f}")

with open("sparsity_grid_final.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["p", "s", "dense_only_FA", "dense_only_power", "dense_only_SE", "dense_only_delay",
                "dense_sparse_FA", "dense_sparse_power", "dense_sparse_SE", "dense_sparse_delay", "power_gain"])
    w.writerows(rows)
print("wrote sparsity_grid_final.csv")
