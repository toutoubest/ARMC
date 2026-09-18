# Stress tests G (gradual drift), H (rarer/larger contamination), and
# I (combined small location + dependence shift), each reusing the
# frozen Experiment A profile+threshold for its base scenario.
import numpy as np, csv, time
from core_v3 import generate_batch
from production_common import HP, T, TAU, N_TRAIN, RHO, SCENARIOS, METHODS, seeds_for, scen_kwargs
from methods_v3 import get_detection_times_batch, summarize_run, se_prop

B_PROFILE, B_CAL, B_EVAL = 3000, 3000, 2000
STRESS_SEED_BASE = 410000
rows = []


def refit_frozen(scenario, cfg):
    """Recompute Experiment A's profile and threshold for a scenario
    from its frozen seeds (profiles are not persisted to disk)."""
    sd = seeds_for(scenario)
    frozen = {}
    for name, (fit_fn, score_fn) in METHODS.items():
        prof = fit_fn(scenario, cfg, B_PROFILE, sd["profile"])
        rngc = np.random.default_rng(sd["cal"])
        Xc, _ = generate_batch(B_CAL, T, cfg["p"], TAU, scenario, rngc, rho=cfg.get("rho", RHO),
                                no_change=True, **scen_kwargs(cfg))
        sc = score_fn(Xc, prof)
        thr = float(np.quantile(sc[:, N_TRAIN:].max(axis=1), 0.95))
        frozen[name] = (prof, thr, score_fn)
    return frozen


# G: gradual drift
print("=== Stress G: gradual/drift shift (ramp_len=100), gaussian_dense frozen fit ===")
cfg = SCENARIOS["gaussian_dense"]
frozen = refit_frozen("gaussian_dense", cfg)
for name, (prof, thr, score_fn) in frozen.items():
    rng = np.random.default_rng(STRESS_SEED_BASE + 1)
    X1, _ = generate_batch(B_EVAL, T, cfg["p"], TAU, "gaussian_dense", rng, rho=RHO, no_change=False,
                            delta=1.0, ramp_len=100)
    s1 = score_fn(X1, prof)
    pw = summarize_run(get_detection_times_batch(s1, thr, N_TRAIN), TAU, no_change=False)
    p_ = pw["power"]
    print(f"  {name:10s} power={p_:.3f}(SE={se_prop(p_,B_EVAL):.3f}) delay={pw['mean_delay']:.1f}")
    rows.append(["G_gradual_drift", name, None, None, p_, se_prop(p_, B_EVAL), pw["mean_delay"], pw["median_delay"]])

# H: rarer/larger contamination
print("=== Stress H: rarer/larger contamination (prob=0.01, scale=15), contaminated_dense frozen fit ===")
cfg = SCENARIOS["contaminated_dense"]
frozen = refit_frozen("contaminated_dense", cfg)
for name, (prof, thr, score_fn) in frozen.items():
    rng0 = np.random.default_rng(STRESS_SEED_BASE + 2)
    X0, _ = generate_batch(B_EVAL, T, cfg["p"], TAU, "contaminated_dense", rng0, rho=RHO, no_change=True,
                            contamination_prob=0.01, contamination_scale=15.0)
    s0 = score_fn(X0, prof)
    fa = summarize_run(get_detection_times_batch(s0, thr, N_TRAIN), TAU, no_change=True)
    rng1 = np.random.default_rng(STRESS_SEED_BASE + 3)
    X1, _ = generate_batch(B_EVAL, T, cfg["p"], TAU, "contaminated_dense", rng1, rho=RHO, no_change=False,
                            delta=1.0, contamination_prob=0.01, contamination_scale=15.0)
    s1 = score_fn(X1, prof)
    pw = summarize_run(get_detection_times_batch(s1, thr, N_TRAIN), TAU, no_change=False)
    far = fa["false_alarm_rate"]; p_ = pw["power"]
    print(f"  {name:10s} FA={far:.3f}(SE={se_prop(far,B_EVAL):.3f}) power={p_:.3f}(SE={se_prop(p_,B_EVAL):.3f}) delay={pw['mean_delay']:.1f}")
    rows.append(["H_rare_large_contam", name, far, se_prop(far, B_EVAL), p_, se_prop(p_, B_EVAL), pw["mean_delay"], pw["median_delay"]])

# I: combined small location + dependence shift
print("=== Stress I: simultaneous small location (delta=0.5) + dependence shift (rho 0.3->0.55) ===")
cfg = SCENARIOS["gaussian_dense"]
frozen = refit_frozen("gaussian_dense", cfg)
for name, (prof, thr, score_fn) in frozen.items():
    rng = np.random.default_rng(STRESS_SEED_BASE + 4)
    X1, _ = generate_batch(B_EVAL, T, cfg["p"], TAU, "combo_location_dependence", rng, rho=RHO,
                            no_change=False, delta=0.5, corr_rho_after=0.55)
    s1 = score_fn(X1, prof)
    pw = summarize_run(get_detection_times_batch(s1, thr, N_TRAIN), TAU, no_change=False)
    p_ = pw["power"]
    print(f"  {name:10s} power={p_:.3f}(SE={se_prop(p_,B_EVAL):.3f}) delay={pw['mean_delay']:.1f}")
    rows.append(["I_combo_small_loc_dep", name, None, None, p_, se_prop(p_, B_EVAL), pw["mean_delay"], pw["median_delay"]])

with open("stress_ghi.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["stress_test", "method", "FA", "SE_FA", "power", "SE_power", "mean_delay", "median_delay"])
    w.writerows(rows)
print("wrote stress_ghi.csv")
