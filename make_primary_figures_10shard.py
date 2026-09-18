import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

full = pd.read_csv("primary10_full_holdout_table.csv")
folds = pd.read_csv("primary10_fold_summary.csv")
regime = pd.read_csv("primary10_regime_breakdown.csv")

full5 = pd.read_csv("primary_full_holdout_table.csv")
regime5 = pd.read_csv("primary_regime_breakdown.csv")


def wilson_ci(k, n, z=1.959963985):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (center - half, center + half)


k10, n10 = int(full["alarm"].sum()), len(full)
lo10, hi10 = wilson_ci(k10, n10)
rate10 = k10 / n10

k5, n5 = int(full5["alarm"].sum()), len(full5)
lo5, hi5 = wilson_ci(k5, n5)
rate5 = k5 / n5

order = ["near_stationary (<1 m/s)", "low_speed (1-5 m/s)", "normal_higher_speed (>=5 m/s)"]
regime = regime.set_index("speed_regime").loc[order].reset_index()
regime5 = regime5.set_index("speed_regime").loc[order].reset_index()

# Figure 1: alarm rate with CI, overall and by speed regime
fig, ax = plt.subplots(figsize=(7.5, 5))
labels = ["Overall\n(pooled, N=%d)" % n10] + [r.replace(" (", "\n(") for r in regime["speed_regime"]]
rates = [rate10] + list(regime["alarm_rate"])
los = [lo10] + list(regime["ci_lo"])
his = [hi10] + list(regime["ci_hi"])
ns = [n10] + list(regime["n"])
ks = [k10] + list(regime["n_alarms"])

x = np.arange(len(labels))
yerr = np.array([[r - lo for r, lo in zip(rates, los)], [hi - r for r, hi in zip(rates, his)]]) * 100
colors = ["#2c5f8a", "#7fa8c9", "#7fa8c9", "#7fa8c9"]

ax.bar(x, np.array(rates) * 100, yerr=yerr, capsize=5, color=colors,
       edgecolor="black", linewidth=0.8, zorder=3)
ax.axhline(5.0, color="firebrick", linestyle="--", linewidth=1.3, zorder=2,
           label="Nominal target ($\\alpha$ = 5%)")

for xi, r, k, n in zip(x, rates, ks, ns):
    ax.text(xi, r * 100 + 1.0, f"{k}/{n}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Held-out alarm rate (%)")
ax.set_title("ARMC v3.1 held-out alarm rate on Waymo validation data (FINAL, 10 shards)\n"
              "10-fold leave-one-shard-out pooled calibration, with Wilson 95% CIs")
ax.set_ylim(0, max(his) * 100 + 5)
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="y", alpha=0.3, zorder=0)
plt.tight_layout()
plt.savefig("fig1_primary_alarm_rate_by_regime_10shard.png", dpi=150)
plt.close()
print("saved fig1_primary_alarm_rate_by_regime_10shard.png")

# Figure 2: channel attribution
alarmed = full[full["alarm"]]
ct = pd.crosstab(alarmed["speed_regime"], alarmed["driving_component"]).reindex(order).fillna(0)
overall_counts = alarmed["driving_component"].value_counts()

components = ["sparse", "dependence", "dense", "scale"]
components = [c for c in components if c in overall_counts.index or c in ct.columns]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1, 1.6]})

ax = axes[0]
counts = [overall_counts.get(c, 0) for c in components]
ax.bar(components, counts, color=["#c76f3c", "#4c8c6b", "#999999", "#999999"][:len(components)],
       edgecolor="black", linewidth=0.8)
for xi, c in zip(range(len(components)), counts):
    ax.text(xi, c + 1.2, str(int(c)), ha="center", fontsize=9)
ax.set_ylabel("Number of alarmed episodes")
ax.set_title(f"Driving ARMC channel\n(all {len(alarmed)} alarms, 10 folds pooled)")
ax.set_ylim(0, max(counts) * 1.25)

ax = axes[1]
bottom = np.zeros(len(order))
comp_colors = {"sparse": "#c76f3c", "dependence": "#4c8c6b", "dense": "#999999", "scale": "#666666"}
for c in components:
    vals = ct[c].values if c in ct.columns else np.zeros(len(order))
    ax.bar(range(len(order)), vals, bottom=bottom, label=c, color=comp_colors.get(c, "gray"),
           edgecolor="black", linewidth=0.7)
    bottom += vals
ax.set_xticks(range(len(order)))
ax.set_xticklabels([o.replace(" (", "\n(") for o in order], fontsize=9)
ax.set_ylabel("Number of alarmed episodes")
ax.set_title("Driving channel by speed regime")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("fig2_channel_attribution_10shard.png", dpi=150)
plt.close()
print("saved fig2_channel_attribution_10shard.png")

# Figure 3: 5-shard vs 10-shard convergence comparison
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

ax = axes[0]
xs = [0, 1]
rates2 = [rate5 * 100, rate10 * 100]
los2 = [lo5 * 100, lo10 * 100]
his2 = [hi5 * 100, hi10 * 100]
yerr2 = np.array([[r - lo for r, lo in zip(rates2, los2)], [hi - r for r, hi in zip(rates2, his2)]])
ax.bar(xs, rates2, yerr=yerr2, capsize=6, color=["#7fa8c9", "#2c5f8a"], edgecolor="black", linewidth=0.8)
ax.axhline(5.0, color="firebrick", linestyle="--", linewidth=1.2, label="Nominal target (5%)")
for xi, r, k, n in zip(xs, rates2, [k5, k10], [n5, n10]):
    ax.text(xi, r + 0.6, f"{k}/{n}", ha="center", fontsize=9)
ax.set_xticks(xs)
ax.set_xticklabels([f"5 shards\n(N={n5})", f"10 shards (FINAL)\n(N={n10})"])
ax.set_ylabel("Overall held-out alarm rate (%)")
ax.set_title("Overall pooled alarm rate:\n5-shard vs 10-shard")
ax.legend(fontsize=8, loc="upper right")
ax.set_ylim(0, max(his2) + 3)

ax = axes[1]
width = 0.35
xpos = np.arange(len(order))
r5 = regime5["alarm_rate"].values * 100
r10 = regime["alarm_rate"].values * 100
lo5r = (regime5["alarm_rate"].values - regime5["ci_lo"].values) * 100
hi5r = (regime5["ci_hi"].values - regime5["alarm_rate"].values) * 100
lo10r = (regime["alarm_rate"].values - regime["ci_lo"].values) * 100
hi10r = (regime["ci_hi"].values - regime["alarm_rate"].values) * 100

ax.bar(xpos - width/2, r5, width, yerr=[lo5r, hi5r], capsize=4, label="5 shards", color="#7fa8c9",
       edgecolor="black", linewidth=0.7)
ax.bar(xpos + width/2, r10, width, yerr=[lo10r, hi10r], capsize=4, label="10 shards (FINAL)", color="#2c5f8a",
       edgecolor="black", linewidth=0.7)
ax.axhline(5.0, color="firebrick", linestyle="--", linewidth=1.2)
ax.set_xticks(xpos)
ax.set_xticklabels([o.replace(" (", "\n(") for o in order], fontsize=8)
ax.set_ylabel("Alarm rate (%)")
ax.set_title("By speed regime: 5-shard vs 10-shard")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("fig3_5shard_vs_10shard_comparison.png", dpi=150)
plt.close()
print("saved fig3_5shard_vs_10shard_comparison.png")

print(f"\n5-shard:  {k5}/{n5} = {rate5*100:.2f}%  CI [{lo5*100:.2f}%, {hi5*100:.2f}%]  width={{(hi5-lo5)*100:.2f}}pp")
print(f"10-shard: {k10}/{n10} = {rate10*100:.2f}%  CI [{lo10*100:.2f}%, {hi10*100:.2f}%]  width={{(hi10-lo10)*100:.2f}}pp")
print(f"CI width shrank by {(1 - (hi10-lo10)/(hi5-lo5))*100:.1f}%")
