# Experiment B for ARMC-v3.1: calibrate once on gaussian_dense, apply the
# frozen threshold unchanged to the other scenarios.
import numpy as np, csv, time
from core_v3 import generate_batch
from production_common import HP, T, TAU, N_TRAIN, RHO, SCENARIOS, scen_kwargs
from methods_v3_1 import DEFAULT_HP_V31, fit_profile_v31_hier, combined_scores_v31_hier
from methods_v3 import get_detection_times_batch, summarize_run, se_prop

B_PROFILE, B_CAL, B_EVAL = 3000, 3000, 2000
REF_SCENARIO = "gaussian_dense"
REF_SEED_PROFILE, REF_SEED_CAL = 10000, 20000
TARGET_SCENARIOS = ["gaussian_dense", "t3_dense", "contaminated_dense", "variance_shift", "correlation_shift"]
EVAL_SEED_BASE = 190000

hp = DEFAULT_HP_V31
ref_cfg = SCENARIOS[REF_SCENARIO]
t0 = time.time()
prof = fit_profile_v31_hier(REF_SCENARIO, B_PROFILE, T, ref_cfg["p"], TAU, N_TRAIN, RHO,
                             seed=REF_SEED_PROFILE, hp=hp, channels=("dense", "sparse", "scale", "dependence"))
rngc = np.random.default_rng(REF_SEED_CAL)
Xc, _ = generate_batch(B_CAL, T, ref_cfg["p"], TAU, REF_SCENARIO, rngc, rho=RHO, no_change=True)
scc, _ = combined_scores_v31_hier(Xc, N_TRAIN, prof, hp)
thr = float(np.quantile(scc[:, N_TRAIN:].max(axis=1), 0.95))

rows = []
for j, target in enumerate(TARGET_SCENARIOS):
    cfg = SCENARIOS[target]
    rho = cfg.get("rho", RHO)
    seed = EVAL_SEED_BASE + 7 * j
    rng = np.random.default_rng(seed)
    X0, _ = generate_batch(B_EVAL, T, cfg["p"], TAU, target, rng, rho=rho, no_change=True, **scen_kwargs(cfg))
    s0, _ = combined_scores_v31_hier(X0, N_TRAIN, prof, hp)
    fa = summarize_run(get_detection_times_batch(s0, thr, N_TRAIN), TAU, no_change=True)
    far = fa["false_alarm_rate"]
    print(f"armc_v3_1  target={target:20s} FA={far:.3f} (SE={se_prop(far,B_EVAL):.3f})")
    rows.append(["armc_v3_1", target, thr, far, se_prop(far, B_EVAL), B_EVAL])
print(f"[done in {time.time()-t0:.1f}s]")

with open("expB_threshold_robustness.csv", "a", newline="") as f:
    w = csv.writer(f)
    w.writerows(rows)
print("appended to expB_threshold_robustness.csv")
