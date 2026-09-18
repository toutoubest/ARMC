# Experiment A (matched-FAR) for ARMC-v3.1, all scenarios, using the
# frozen final-eval seeds from production_common.seeds_for.
import sys, time, csv, os
import numpy as np
from core_v3 import generate_batch
from production_common import HP, T, TAU, N_TRAIN, RHO, SCENARIOS, SCEN_ORDER, scen_kwargs, seeds_for
from methods_v3_1 import DEFAULT_HP_V31, fit_profile_v31_hier, combined_scores_v31_hier
from methods_v3 import get_detection_times_batch, summarize_run, se_prop

OUT = "expA_main_results.csv"
hp = DEFAULT_HP_V31

for scenario in SCEN_ORDER:
    cfg = SCENARIOS[scenario]
    p = cfg["p"]; rho = cfg.get("rho", RHO)
    if scenario == "gaussian_sparse":
        B_PROFILE, B_CAL, B_EVAL = 1500, 1500, 1200
    else:
        B_PROFILE, B_CAL, B_EVAL = 3000, 3000, 2000
    sd = seeds_for(scenario)
    t0 = time.time()
    prof = fit_profile_v31_hier(scenario, B_PROFILE, T, p, TAU, N_TRAIN, rho, seed=sd["profile"],
                                 hp=hp, channels=("dense", "sparse", "scale", "dependence"),
                                 **scen_kwargs(cfg))
    rngc = np.random.default_rng(sd["cal"])
    Xc, _ = generate_batch(B_CAL, T, p, TAU, scenario, rngc, rho=rho, no_change=True, **scen_kwargs(cfg))
    scc, _ = combined_scores_v31_hier(Xc, N_TRAIN, prof, hp)
    thr = float(np.quantile(scc[:, N_TRAIN:].max(axis=1), 0.95))

    rng0 = np.random.default_rng(sd["fa_eval"])
    X0, _ = generate_batch(B_EVAL, T, p, TAU, scenario, rng0, rho=rho, no_change=True, **scen_kwargs(cfg))
    s0, _ = combined_scores_v31_hier(X0, N_TRAIN, prof, hp)
    fa = summarize_run(get_detection_times_batch(s0, thr, N_TRAIN), TAU, no_change=True)

    rng1 = np.random.default_rng(sd["power_eval"])
    X1, _ = generate_batch(B_EVAL, T, p, TAU, scenario, rng1, rho=rho, no_change=False, delta=1.0, **scen_kwargs(cfg))
    s1, _ = combined_scores_v31_hier(X1, N_TRAIN, prof, hp)
    pw = summarize_run(get_detection_times_batch(s1, thr, N_TRAIN), TAU, no_change=False)

    far = fa["false_alarm_rate"]; power = pw["power"]
    print(f"{scenario:20s} armc_v3_1  FA={far:.3f} power={power:.3f} delay={pw['mean_delay']:.1f}  [{time.time()-t0:.1f}s]")

    write_header = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["scenario", "method", "threshold", "FA", "SE_FA", "power", "SE_power",
                        "mean_delay", "median_delay", "n_detections", "n_eval",
                        "B_profile", "B_cal", "B_eval", "T", "tau", "n_train", "delta"])
        w.writerow([scenario, "armc_v3_1", thr, far, se_prop(far, B_EVAL), power, se_prop(power, B_EVAL),
                    pw["mean_delay"], pw["median_delay"], pw["n_detections"], B_EVAL,
                    B_PROFILE, B_CAL, B_EVAL, T, TAU, N_TRAIN, 1.0])
