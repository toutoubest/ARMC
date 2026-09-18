
import os
import pandas as pd
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(_HERE, "..", "data")
WAY = os.path.join(_HERE, "..", "data")

pd.set_option("display.width", 200)

METHOD_ORDER = ["standard", "huber", "armc_v1", "armc_v2", "armc_v3", "armc_v3_1"]
METHOD_TEX = {
    "standard": "Standard", "huber": "Huber", "armc_v1": "ARMC-v1",
    "armc_v2": "ARMC-v2", "armc_v3": "ARMC-v3", "armc_v3_1": "ARMC-v3.1",
}
SCEN_TEX = {
    "gaussian_dense": "Gaussian dense", "gaussian_sparse": "Gaussian sparse ($p{=}60,s{=}3$)",
    "t3_dense": "$t_3$ dense", "contaminated_dense": "Contaminated dense",
    "variance_shift": "Variance shift", "correlation_shift": "Correlation shift",
}
SCEN_ORDER = ["gaussian_dense", "gaussian_sparse", "t3_dense", "contaminated_dense",
              "variance_shift", "correlation_shift"]

print("=" * 70, "\nTABLE: Experiment A power only (main text)\n", "=" * 70)
expA = pd.read_csv(f"{SIM}/expA_main_results.csv")
for scen in SCEN_ORDER:
    row = [SCEN_TEX[scen]]
    for m in METHOD_ORDER:
        r = expA[(expA.scenario == scen) & (expA.method == m)]
        row.append(f"{float(r.power.iloc[0]):.3f}" if len(r) else "--")
    print(" & ".join(row) + r" \\")

print("\n", "=" * 70, "\nTABLE: Experiment A full (FA(SE) / power(SE))  appendix\n", "=" * 70)
for scen in SCEN_ORDER:
    for m in METHOD_ORDER:
        r = expA[(expA.scenario == scen) & (expA.method == m)]
        if not len(r):
            continue
        r = r.iloc[0]
        print(f"{SCEN_TEX[scen]} & {METHOD_TEX[m]} & "
              f"{r.FA:.3f} ({r.SE_FA:.3f}) & {r.power:.3f} ({r.SE_power:.3f}) & "
              f"{r.mean_delay:.1f} & {int(r.median_delay)} \\\\")

print("\n", "=" * 70, "\nTABLE: Experiment B threshold transportability (main text)\n", "=" * 70)
expB = pd.read_csv(f"{SIM}/expB_threshold_robustness.csv")
targets = ["gaussian_dense", "t3_dense", "contaminated_dense", "variance_shift", "correlation_shift"]
for t in targets:
    row = [SCEN_TEX[t]]
    for m in METHOD_ORDER:
        r = expB[(expB.method == m) & (expB.target_scenario == t)]
        row.append(f"{float(r.FA.iloc[0]):.3f}" if len(r) else "--")
    print(" & ".join(row) + r" \\")

print("\n", "=" * 70, "\nTABLE: Ablation study v3.1 (main text)\n", "=" * 70)
ab = pd.read_csv(f"{SIM}/ablations_v31.csv")
FULL_POWER = {scen: float(expA[(expA.scenario == scen) & (expA.method == "armc_v3_1")].power.iloc[0])
              for scen in SCEN_ORDER}
for scen in SCEN_ORDER:
    row = [SCEN_TEX[scen]]
    for abln in ["dense_only", "location_only", "dispersion_only"]:
        r = ab[(ab.scenario == scen) & (ab.ablation == abln)]
        row.append(f"{float(r.power.iloc[0]):.3f}" if len(r) else "--")
    row.append(f"{FULL_POWER[scen]:.3f}")
    print(" & ".join(row) + r" \\")

print("\n", "=" * 70, "\nTABLE: sparsity grid (appendix)\n", "=" * 70)
sg = pd.read_csv(f"{SIM}/sparsity_grid_final.csv")
for r in sg.itertuples():
    print(f"{int(r.p)} & {int(r.s)} & {r.dense_only_power:.3f} & {r.dense_sparse_power:.3f} & "
          f"{r.power_gain:+.3f} \\\\")

print("\n", "=" * 70, "\nTABLE: stress tests G/H/I (appendix)\n", "=" * 70)
sg2 = pd.read_csv(f"{SIM}/stress_ghi.csv")
for test, label in [("G_gradual_drift", "G: gradual drift (power)"),
                     ("H_rare_large_contam", "H: rare large contamination (FA)"),
                     ("H_rare_large_contam", "H: rare large contamination (power)"),
                     ("I_combo_small_loc_dep", "I: compound location+dependence (power)")]:
    pass
for m in METHOD_ORDER[:5]:  # v3.1 not run on stress tests
    rG = sg2[(sg2.stress_test == "G_gradual_drift") & (sg2.method == m)]
    rH = sg2[(sg2.stress_test == "H_rare_large_contam") & (sg2.method == m)]
    rI = sg2[(sg2.stress_test == "I_combo_small_loc_dep") & (sg2.method == m)]
    g = f"{float(rG.power.iloc[0]):.3f}" if len(rG) else "--"
    hfa = f"{float(rH.FA.iloc[0]):.3f}" if len(rH) else "--"
    hpow = f"{float(rH.power.iloc[0]):.3f}" if len(rH) else "--"
    ipow = f"{float(rI.power.iloc[0]):.3f}" if len(rI) else "--"
    print(f"{METHOD_TEX[m]} & {g} & {hfa} & {hpow} & {ipow} \\\\")

print("\n", "=" * 70, "\nTABLE: correlation-shift scenario validity check (appendix)\n", "=" * 70)
t8 = pd.read_csv(f"{SIM}/task8_correlation_sanity.csv")
print(f"Coordinatewise mean, pre-change & [{t8.pre_mean.min():.4f}, {t8.pre_mean.max():.4f}] \\\\")
print(f"Coordinatewise mean, post-change & [{t8.post_mean.min():.4f}, {t8.post_mean.max():.4f}] \\\\")
print(f"Coordinatewise variance, pre-change & [{t8.pre_var.min():.4f}, {t8.pre_var.max():.4f}] \\\\")
print(f"Coordinatewise variance, post-change & [{t8.post_var.min():.4f}, {t8.post_var.max():.4f}] \\\\")

print("\n", "=" * 70, "\nTABLE: Waymo 10-fold summary (main text)\n", "=" * 70)
f10 = pd.read_csv(f"{WAY}/primary10_fold_summary.csv")
for r in f10.itertuples():
    print(f"{int(r.fold)} & {int(r.held_out_shard)} & {int(r.n_baseline)} & {int(r.n_calibration)} & "
          f"{int(r.n_held_out)} & {r.threshold:.3f} & {int(r.n_alarms)} & {r.alarm_rate*100:.2f}\\% \\\\")

print("\n", "=" * 70, "\nTABLE: Waymo 5-fold summary (appendix)\n", "=" * 70)
f5 = pd.read_csv(f"{WAY}/primary5_fold_summary.csv")
for r in f5.itertuples():
    print(f"{int(r.fold)} & {int(r.held_out_shard)} & {int(r.n_held_out)} & "
          f"{r.threshold:.3f} & {int(r.n_alarms)} & {r.alarm_rate*100:.2f}\\% \\\\")

print("\n", "=" * 70, "\nTABLE: Waymo behavioral case studies (appendix, 7 pilot alarms)\n", "=" * 70)
beh = pd.read_csv(f"{WAY}/alarm_behavior_characterization.csv")
for r in beh.itertuples():
    print(f"{r.episode} & {r.scenario_id} & {r.driving_component} & {r.speed_at_alarm_mps:.2f} & "
          f"{r.speed_mean_pre20_mps:.2f} & {r.max_abs_yaw_rate_radps:.4f} \\\\")
