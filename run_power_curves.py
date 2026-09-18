# Power curves (matched-FAR) for correlation_shift, variance_shift, and
# the gaussian scenarios, across all methods.
import numpy as np, csv, time
from core_v3 import generate_batch
from production_common import HP, T, TAU, N_TRAIN, RHO, SCENARIOS, scen_kwargs, METHODS
from methods_v3 import get_detection_times_batch, summarize_run, se_prop

B_PROFILE, B_CAL, B_EVAL = 2000, 2000, 1500
CURVE_SEED_PROFILE, CURVE_SEED_CAL = 210000, 220000
CURVE_SEED_EVAL_BASE = 230000

rows = []

# correlation_shift power curve over corr_rho_after
scenario = "correlation_shift"
cfg = SCENARIOS[scenario]
rho0 = cfg.get("rho", RHO)
rho1_grid = [0.4, 0.5, 0.6, 0.7]

for name, (fit_fn, score_fn) in METHODS.items():
    t0 = time.time()
    prof = fit_fn(scenario, cfg, B_PROFILE, CURVE_SEED_PROFILE)
    rngc = np.random.default_rng(CURVE_SEED_CAL)
    Xc, _ = generate_batch(B_CAL, T, cfg["p"], TAU, scenario, rngc, rho=rho0, no_change=True,
                            **{k: v for k, v in scen_kwargs(cfg).items() if k != "corr_rho_after"})
    scc = score_fn(Xc, prof)
    thr = float(np.quantile(scc[:, N_TRAIN:].max(axis=1), 0.95))

    for j, rho1 in enumerate(rho1_grid):
        seed = CURVE_SEED_EVAL_BASE + 11 * j
        rng = np.random.default_rng(seed)
        X1, _ = generate_batch(B_EVAL, T, cfg["p"], TAU, scenario, rng, rho=rho0, no_change=False,
                                corr_rho_after=rho1)
        s1 = score_fn(X1, prof)
        pw = summarize_run(get_detection_times_batch(s1, thr, N_TRAIN), TAU, no_change=False)
        p_ = pw["power"]
        print(f"correlation_shift {name:10s} rho1={rho1} power={p_:.3f} (SE={se_prop(p_,B_EVAL):.3f}) delay={pw['mean_delay']:.1f}")
        rows.append(["correlation_shift", name, "corr_rho_after", rho1, p_, se_prop(p_, B_EVAL),
                      pw["mean_delay"], pw["median_delay"], B_EVAL])
    print(f"  [{name} done in {time.time()-t0:.1f}s]")

# variance_shift power curve over var_gamma
scenario = "variance_shift"
cfg = SCENARIOS[scenario]
gamma_grid = [1.1, 1.3, 1.5, 1.7, 2.0]
for name, (fit_fn, score_fn) in METHODS.items():
    t0 = time.time()
    prof = fit_fn(scenario, cfg, B_PROFILE, CURVE_SEED_PROFILE)
    rngc = np.random.default_rng(CURVE_SEED_CAL)
    Xc, _ = generate_batch(B_CAL, T, cfg["p"], TAU, scenario, rngc, rho=RHO, no_change=True,
                            **{k: v for k, v in scen_kwargs(cfg).items() if k != "var_gamma"})
    scc = score_fn(Xc, prof)
    thr = float(np.quantile(scc[:, N_TRAIN:].max(axis=1), 0.95))
    for j, gamma in enumerate(gamma_grid):
        seed = CURVE_SEED_EVAL_BASE + 1000 + 11 * j
        rng = np.random.default_rng(seed)
        X1, _ = generate_batch(B_EVAL, T, cfg["p"], TAU, scenario, rng, rho=RHO, no_change=False,
                                var_gamma=gamma)
        s1 = score_fn(X1, prof)
        pw = summarize_run(get_detection_times_batch(s1, thr, N_TRAIN), TAU, no_change=False)
        p_ = pw["power"]
        print(f"variance_shift {name:10s} gamma={gamma} power={p_:.3f} (SE={se_prop(p_,B_EVAL):.3f}) delay={pw['mean_delay']:.1f}")
        rows.append(["variance_shift", name, "var_gamma", gamma, p_, se_prop(p_, B_EVAL),
                      pw["mean_delay"], pw["median_delay"], B_EVAL])
    print(f"  [{name} done in {time.time()-t0:.1f}s]")

# gaussian_dense / gaussian_sparse power curve over delta
for scenario in ["gaussian_dense", "gaussian_sparse"]:
    cfg = SCENARIOS[scenario]
    delta_grid = [0.4, 0.6, 0.8, 1.0, 1.2]
    B_p, B_c, B_e = (1200, 1200, 900) if scenario == "gaussian_sparse" else (B_PROFILE, B_CAL, B_EVAL)
    for name, (fit_fn, score_fn) in METHODS.items():
        t0 = time.time()
        prof = fit_fn(scenario, cfg, B_p, CURVE_SEED_PROFILE)
        rngc = np.random.default_rng(CURVE_SEED_CAL)
        Xc, _ = generate_batch(B_c, T, cfg["p"], TAU, scenario, rngc, rho=RHO, no_change=True,
                                **scen_kwargs(cfg))
        scc = score_fn(Xc, prof)
        thr = float(np.quantile(scc[:, N_TRAIN:].max(axis=1), 0.95))
        for j, delta in enumerate(delta_grid):
            seed = CURVE_SEED_EVAL_BASE + 2000 + 11 * j
            rng = np.random.default_rng(seed)
            X1, _ = generate_batch(B_e, T, cfg["p"], TAU, scenario, rng, rho=RHO, no_change=False,
                                    delta=delta, **scen_kwargs(cfg))
            s1 = score_fn(X1, prof)
            pw = summarize_run(get_detection_times_batch(s1, thr, N_TRAIN), TAU, no_change=False)
            p_ = pw["power"]
            print(f"{scenario} {name:10s} delta={delta} power={p_:.3f} (SE={se_prop(p_,B_e):.3f}) delay={pw['mean_delay']:.1f}")
            rows.append([scenario, name, "delta", delta, p_, se_prop(p_, B_e),
                          pw["mean_delay"], pw["median_delay"], B_e])
        print(f"  [{scenario}/{name} done in {time.time()-t0:.1f}s]")

with open("power_curves.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["scenario", "method", "x_var", "x_value", "power", "SE_power", "mean_delay", "median_delay", "n_eval"])
    w.writerows(rows)
print("wrote power_curves.csv")
