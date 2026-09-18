# ARMC-v3.1 ablations: location-only, dispersion-only, and full model.
import numpy as np, csv, time
from core_v3 import generate_batch
from production_common import HP, T, TAU, N_TRAIN, RHO, SCENARIOS, SCEN_ORDER, scen_kwargs
from methods_v3_1 import DEFAULT_HP_V31, fit_profile_v31_hier, combined_scores_v31_hier
from methods_v3 import get_detection_times_batch, summarize_run, se_prop

B_PROFILE, B_CAL, B_EVAL = 2000, 2000, 1500
ABLATIONS = {
    "dense_only":      ("dense",),
    "location_only":   ("dense", "sparse"),
    "dispersion_only": ("scale", "dependence"),
}

rows = []
for i, scenario in enumerate(SCEN_ORDER):
    cfg = SCENARIOS[scenario]
    p = cfg["p"]; rho = cfg.get("rho", RHO)
    B_p, B_c, B_e = (1200, 1200, 900) if scenario == "gaussian_sparse" else (B_PROFILE, B_CAL, B_EVAL)
    seed_profile = 310000 + 17 * i
    seed_cal = 320000 + 17 * i
    seed_fa = 330000 + 17 * i
    seed_pw = 340000 + 17 * i
    for abl_name, channels in ABLATIONS.items():
        t0 = time.time()
        prof = fit_profile_v31_hier(scenario, B_p, T, p, TAU, N_TRAIN, rho, seed=seed_profile,
                                     hp=DEFAULT_HP_V31, channels=channels, **scen_kwargs(cfg))
        rngc = np.random.default_rng(seed_cal)
        Xc, _ = generate_batch(B_c, T, p, TAU, scenario, rngc, rho=rho, no_change=True, **scen_kwargs(cfg))
        scc, _ = combined_scores_v31_hier(Xc, N_TRAIN, prof, DEFAULT_HP_V31)
        thr = float(np.quantile(scc[:, N_TRAIN:].max(axis=1), 0.95))

        rng0 = np.random.default_rng(seed_fa)
        X0, _ = generate_batch(B_e, T, p, TAU, scenario, rng0, rho=rho, no_change=True, **scen_kwargs(cfg))
        s0, _ = combined_scores_v31_hier(X0, N_TRAIN, prof, DEFAULT_HP_V31)
        fa = summarize_run(get_detection_times_batch(s0, thr, N_TRAIN), TAU, no_change=True)

        rng1 = np.random.default_rng(seed_pw)
        X1, _ = generate_batch(B_e, T, p, TAU, scenario, rng1, rho=rho, no_change=False, delta=1.0, **scen_kwargs(cfg))
        s1, _ = combined_scores_v31_hier(X1, N_TRAIN, prof, DEFAULT_HP_V31)
        pw = summarize_run(get_detection_times_batch(s1, thr, N_TRAIN), TAU, no_change=False)

        far = fa["false_alarm_rate"]; p_ = pw["power"]
        print(f"{scenario:20s} {abl_name:16s} FA={far:.3f} power={p_:.3f}(SE={se_prop(p_,B_e):.3f}) "
              f"delay={pw['mean_delay']:.1f}  [{time.time()-t0:.1f}s]")
        rows.append([scenario, abl_name, far, se_prop(far, B_e), p_, se_prop(p_, B_e),
                      pw["mean_delay"], pw["median_delay"], B_p, B_c, B_e])

with open("ablations_v31.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["scenario", "ablation", "FA", "SE_FA", "power", "SE_power", "mean_delay", "median_delay",
                "B_profile", "B_cal", "B_eval"])
    w.writerows(rows)
print("wrote ablations_v31.csv")
