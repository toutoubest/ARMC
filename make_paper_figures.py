# Generates every figure 

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(_HERE, "..", "data")
WAY = os.path.join(_HERE, "..", "data")
OUT = os.path.join(_HERE, "..", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "pdf.fonttype": 42,
})

METHOD_LABELS = {
    "standard": "Standard",
    "huber": "Huber",
    "armc_v1": "ARMC-v1",
    "armc_v2": "ARMC-v2",
    "armc_v3": "ARMC-v3",
    "armc_v3_1": "ARMC-v3.1",
}
METHOD_ORDER = ["standard", "huber", "armc_v1", "armc_v2", "armc_v3", "armc_v3_1"]
METHOD_COLORS = {
    "standard": "#9e9e9e", "huber": "#6baed6", "armc_v1": "#c994c7",
    "armc_v2": "#fdae6b", "armc_v3": "#74c476", "armc_v3_1": "#2c5f8a",
}

SCEN_LABELS = {
    "gaussian_dense": "Gaussian\ndense",
    "t3_dense": "$t_3$\ndense",
    "contaminated_dense": "Contaminated\ndense",
    "variance_shift": "Variance\nshift",
    "correlation_shift": "Correlation\nshift",
}

# Figure 1: threshold-transportability (Experiment B)
expB = pd.read_csv(f"{SIM}/expB_threshold_robustness.csv")
targets = ["gaussian_dense", "t3_dense", "contaminated_dense", "variance_shift", "correlation_shift"]
methods = METHOD_ORDER

fig, ax = plt.subplots(figsize=(6.8, 4.2))
n_t, n_m = len(targets), len(methods)
width = 0.8 / n_m
x = np.arange(n_t)
for j, m in enumerate(methods):
    vals = []
    for t in targets:
        row = expB[(expB.method == m) & (expB.target_scenario == t)]
        vals.append(float(row["FA"].iloc[0]) if len(row) else np.nan)
    ax.bar(x + (j - n_m / 2 + 0.5) * width, vals, width * 0.95,
           label=METHOD_LABELS[m], color=METHOD_COLORS[m], edgecolor="black", linewidth=0.4)
ax.axhline(0.05, color="firebrick", linestyle="--", linewidth=1.1, zorder=0, label="Nominal $\\alpha=0.05$")
ax.set_xticks(x)
ax.set_xticklabels([SCEN_LABELS[t] for t in targets])
ax.set_ylabel("False-alarm rate under a threshold\ncalibrated only on Gaussian-dense data")
ax.set_yscale("log")
ax.set_ylim(0.008, 1.3)
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.30), frameon=False, columnspacing=1.2, handlelength=1.4)
plt.tight_layout()
plt.savefig(f"{OUT}/fig_threshold_transport.pdf")
plt.close()
print("saved fig_threshold_transport.pdf")

# Figure 2: weak-signal correlation-shift power curve, v3 vs v3.1
wk = pd.read_csv(f"{SIM}/correlation_power_curve_weak.csv")
fig, ax = plt.subplots(figsize=(5.6, 4.0))
for meth, color, marker, label in [("v3", "#74c476", "o", "ARMC-v3"), ("v3.1", "#2c5f8a", "s", "ARMC-v3.1")]:
    sub = wk[wk.method == meth].sort_values("rho1")
    ax.errorbar(sub.rho1, sub.power, yerr=1.96 * sub.SE_power, marker=marker, color=color,
                label=label, capsize=2.5, linewidth=1.3, markersize=4.5)
ax.axvline(0.2, color="gray", linestyle=":", linewidth=1.0, label="Null value ($\\rho_0=0.2$)")
ax.set_xlabel(r"Post-change lag-1 cross-coordinate correlation $\rho_1$")
ax.set_ylabel("Power (matched nominal FA $=0.05$)")
ax.set_ylim(-0.03, 1.05)
ax.legend(loc="lower right", frameon=False)
plt.tight_layout()
plt.savefig(f"{OUT}/fig_weak_correlation_power.pdf")
plt.close()
print("saved fig_weak_correlation_power.pdf")

# Figure 3: Waymo 10-shard alarm rate by speed regime
full10 = pd.read_csv(f"{WAY}/primary10_full_holdout_table.csv")
regime10 = pd.read_csv(f"{WAY}/primary10_regime_breakdown.csv")


def wilson_ci(k, n, z=1.959963985):
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


order = ["near_stationary (<1 m/s)", "low_speed (1-5 m/s)", "normal_higher_speed (>=5 m/s)"]
regime10 = regime10.set_index("speed_regime").loc[order].reset_index()
k10, n10 = int(full10["alarm"].sum()), len(full10)
lo10, hi10 = wilson_ci(k10, n10)

fig, ax = plt.subplots(figsize=(6.4, 4.2))
labels = [f"Overall\n(N={n10})"] + ["Near-stationary\n(<1 m/s)", "Low-speed\n(1--5 m/s)", "Normal/higher\n($\\geq$5 m/s)"]
rates = [k10 / n10] + list(regime10["alarm_rate"])
los = [lo10] + list(regime10["ci_lo"])
his = [hi10] + list(regime10["ci_hi"])
ns = [n10] + list(regime10["n"])
ks = [k10] + list(regime10["n_alarms"])
x = np.arange(len(labels))
yerr = np.array([[r - lo for r, lo in zip(rates, los)], [hi - r for r, hi in zip(rates, his)]]) * 100
colors = ["#2c5f8a", "#8fb8d8", "#8fb8d8", "#8fb8d8"]
ax.bar(x, np.array(rates) * 100, yerr=yerr, capsize=5, color=colors, edgecolor="black", linewidth=0.6, zorder=3)
ax.axhline(5.0, color="firebrick", linestyle="--", linewidth=1.2, zorder=2, label="Nominal target ($\\alpha=5\\%$)")
for xi, r, k, n in zip(x, rates, ks, ns):
    ax.text(xi, r * 100 + 0.9, f"{k}/{n}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Held-out alarm rate (\\%)")
ax.set_ylim(0, max(his) * 100 + 3.5)
ax.legend(loc="upper right", frameon=False)
plt.tight_layout()
plt.savefig(f"{OUT}/fig_waymo_alarmrate_regime.pdf")
plt.close()
print("saved fig_waymo_alarmrate_regime.pdf")

# Figure 4: 5-shard vs 10-shard convergence
full5 = pd.read_csv(f"{WAY}/primary5_full_holdout_table.csv")
regime5 = pd.read_csv(f"{WAY}/primary5_regime_breakdown.csv").set_index("speed_regime").loc[order].reset_index()
k5, n5 = int(full5["alarm"].sum()), len(full5)
lo5, hi5 = wilson_ci(k5, n5)

fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.0))

ax = axes[0]
xs = [0, 1]
rates2 = [k5 / n5 * 100, k10 / n10 * 100]
los2 = [lo5 * 100, lo10 * 100]
his2 = [hi5 * 100, hi10 * 100]
yerr2 = np.array([[r - lo for r, lo in zip(rates2, los2)], [hi - r for r, hi in zip(rates2, his2)]])
ax.bar(xs, rates2, yerr=yerr2, capsize=6, color=["#8fb8d8", "#2c5f8a"], edgecolor="black", linewidth=0.6)
ax.axhline(5.0, color="firebrick", linestyle="--", linewidth=1.1, label="Nominal $\\alpha=5\\%$")
for xi, r, k, n in zip(xs, rates2, [k5, k10], [n5, n10]):
    ax.text(xi, r + 0.6, f"{k}/{n}", ha="center", fontsize=8.5)
ax.set_xticks(xs)
ax.set_xticklabels([f"5 shards\n(N={n5})", f"10 shards\n(N={n10})"])
ax.set_ylabel("Overall held-out alarm rate (\\%)")
ax.legend(fontsize=8, loc="upper right", frameon=False)
ax.set_ylim(0, max(his2) + 3)

ax = axes[1]
width = 0.35
xpos = np.arange(3)
r5 = regime5["alarm_rate"].values * 100
r10 = regime10["alarm_rate"].values * 100
lo5r = (regime5["alarm_rate"].values - regime5["ci_lo"].values) * 100
hi5r = (regime5["ci_hi"].values - regime5["alarm_rate"].values) * 100
lo10r = (regime10["alarm_rate"].values - regime10["ci_lo"].values) * 100
hi10r = (regime10["ci_hi"].values - regime10["alarm_rate"].values) * 100
ax.bar(xpos - width / 2, r5, width, yerr=[lo5r, hi5r], capsize=4, label="5 shards", color="#8fb8d8",
       edgecolor="black", linewidth=0.6)
ax.bar(xpos + width / 2, r10, width, yerr=[lo10r, hi10r], capsize=4, label="10 shards", color="#2c5f8a",
       edgecolor="black", linewidth=0.6)
ax.axhline(5.0, color="firebrick", linestyle="--", linewidth=1.1)
ax.set_xticks(xpos)
ax.set_xticklabels(["Near-\nstationary", "Low-\nspeed", "Normal/\nhigher"], fontsize=8.5)
ax.set_ylabel("Alarm rate (\\%)")
ax.legend(fontsize=8, frameon=False)

plt.tight_layout()
plt.savefig(f"{OUT}/fig_waymo_convergence.pdf")
plt.close()
print("saved fig_waymo_convergence.pdf")

# Figure 5: channel attribution (10-shard)
alarmed = full10[full10["alarm"]]
ct = pd.crosstab(alarmed["speed_regime"], alarmed["driving_component"]).reindex(order).fillna(0)
overall_counts = alarmed["driving_component"].value_counts()
components = ["sparse", "dependence"]

fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.0), gridspec_kw={"width_ratios": [1, 1.5]})
ax = axes[0]
counts = [overall_counts.get(c, 0) for c in components]
ax.bar(components, counts, color=["#c76f3c", "#4c8c6b"], edgecolor="black", linewidth=0.6)
for xi, c in zip(range(len(components)), counts):
    ax.text(xi, c + 1.5, str(int(c)), ha="center", fontsize=9)
ax.set_ylabel("Number of alarmed episodes")
ax.set_title(f"All {len(alarmed)} alarms (10 folds)", fontsize=9.5)
ax.set_ylim(0, max(counts) * 1.25)

ax = axes[1]
bottom = np.zeros(3)
colors2 = {"sparse": "#c76f3c", "dependence": "#4c8c6b"}
for c in components:
    vals = ct[c].values if c in ct.columns else np.zeros(3)
    ax.bar(range(3), vals, bottom=bottom, label=c.capitalize(), color=colors2[c], edgecolor="black", linewidth=0.6)
    bottom += vals
ax.set_xticks(range(3))
ax.set_xticklabels(["Near-\nstationary", "Low-\nspeed", "Normal/\nhigher"], fontsize=8.5)
ax.set_ylabel("Number of alarmed episodes")
ax.set_title("By speed regime", fontsize=9.5)
ax.legend(fontsize=8.5, frameon=False)

plt.tight_layout()
plt.savefig(f"{OUT}/fig_waymo_channel_attribution.pdf")
plt.close()
print("saved fig_waymo_channel_attribution.pdf")

# Figure A1: variance-shift and correlation-shift power curves
pc = pd.read_csv(f"{SIM}/power_curves.csv")
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))

ax = axes[0]
for m in ["standard", "huber", "armc_v1", "armc_v2", "armc_v3"]:
    sub = pc[(pc.scenario == "variance_shift") & (pc.method == m)].sort_values("x_value")
    ax.errorbar(sub.x_value, sub.power, yerr=1.96 * sub.SE_power, marker="o", markersize=3.5,
                color=METHOD_COLORS[m], label=METHOD_LABELS[m], linewidth=1.2, capsize=2)
ax.set_xlabel(r"Variance-inflation factor $\gamma$")
ax.set_ylabel("Power (matched nominal FA $=0.05$)")
ax.set_title("Variance-shift scenario", fontsize=9.5)
ax.set_ylim(-0.03, 1.05)

ax = axes[1]
for m in ["standard", "huber", "armc_v1", "armc_v2", "armc_v3"]:
    sub = pc[(pc.scenario == "correlation_shift") & (pc.method == m)].sort_values("x_value")
    ax.errorbar(sub.x_value, sub.power, yerr=1.96 * sub.SE_power, marker="o", markersize=3.5,
                color=METHOD_COLORS[m], label=METHOD_LABELS[m], linewidth=1.2, capsize=2)
ax.set_xlabel(r"Post-change correlation $\rho_1$ ($\rho_0=0.2$)")
ax.set_title("Correlation-shift scenario", fontsize=9.5)
ax.set_ylim(-0.03, 1.05)
ax.legend(loc="center right", fontsize=8, frameon=False)

plt.tight_layout()
plt.savefig(f"{OUT}/fig_A_power_curves.pdf")
plt.close()
print("saved fig_A_power_curves.pdf")

# Figure A2: sparsity-adaptivity grid
sg = pd.read_csv(f"{SIM}/sparsity_grid_final.csv")
fig, ax = plt.subplots(figsize=(5.6, 3.8))
labels = [f"$p={int(r.p)}$\n$s={int(r.s)}$" for r in sg.itertuples()]
x = np.arange(len(sg))
width = 0.35
ax.bar(x - width / 2, sg.dense_only_power, width, label="Dense-only", color="#9e9e9e", edgecolor="black", linewidth=0.6)
ax.bar(x + width / 2, sg.dense_sparse_power, width, label="Dense $+$ sparse", color="#2c5f8a", edgecolor="black", linewidth=0.6)
for xi, gain in zip(x, sg.power_gain):
    ax.text(xi, max(sg.dense_only_power[xi], sg.dense_sparse_power[xi]) + 0.02,
            f"{gain:+.3f}", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Power ($\\delta=1.0$)")
ax.set_ylim(0, 1.0)
ax.legend(loc="lower left", frameon=False, fontsize=8.5)
plt.tight_layout()
plt.savefig(f"{OUT}/fig_A_sparsity_grid.pdf")
plt.close()
print("saved fig_A_sparsity_grid.pdf")

# Figure A3: single-shard pilot split-sensitivity vs pooled design
ss = pd.read_csv(f"{WAY}/split_sensitivity_50seeds.csv")
rot = pd.read_csv(f"{WAY}/pooled_rotation_results.csv")

fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
ax = axes[0]
ax.hist(ss.alarm_rate * 100, bins=12, color="#c76f3c", edgecolor="black", linewidth=0.5, alpha=0.85)
ax.axvline(7 / 58 * 100, color="black", linestyle="--", linewidth=1.3, label="Originally reported (7/58)")
ax.axvline(5.0, color="gray", linestyle=":", linewidth=1.2, label="Nominal target (5\\%)")
ax.set_xlabel("Held-out alarm rate (\\%)")
ax.set_ylabel("Count (of 50 reshuffled splits)")
ax.set_title("Single-shard pilot: 40/40/20 split\nunder 50 random reshuffles", fontsize=9)
ax.legend(fontsize=7.5, frameon=False)

ax = axes[1]
ax.bar(range(1, 6), rot.rate * 100, color="#2c5f8a", edgecolor="black", linewidth=0.6)
ax.axhline(5.0, color="gray", linestyle=":", linewidth=1.2, label="Nominal target (5\\%)")
ax.axhline(rot.rate.mean() * 100, color="firebrick", linestyle="-", linewidth=1.2,
           label=f"Mean across rotations ({rot.rate.mean()*100:.2f}\\%)")
ax.set_xlabel("Rotation")
ax.set_ylabel("Held-out alarm rate (\\%)")
ax.set_title("Pooled 5-shard design:\nshard-role rotation", fontsize=9)
ax.legend(fontsize=7.5, frameon=False, loc="upper left")

plt.tight_layout()
plt.savefig(f"{OUT}/fig_A_split_sensitivity.pdf")
plt.close()
print("saved fig_A_split_sensitivity.pdf")

print("\nAll figures written to", OUT)
