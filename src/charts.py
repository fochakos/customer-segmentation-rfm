"""Draw the README charts from the outputs of segment.py.

Usage (from the project root, after build.py and segment.py):
    python src/charts.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import REPORTS  # noqa: E402

FIG = REPORTS / "figures"
RESULTS = json.loads((REPORTS / "results.json").read_text())

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
CLUSTER_COLORS = {"VIP regulars": BLUE, "Steady mid-value": ORANGE,
                  "Recent light buyers": AQUA, "Lapsed one-timers": YELLOW}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 24,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 160, "savefig.bbox": "tight", "savefig.facecolor": SURFACE,
})


def subtitle(ax, text):
    ax.text(0, 1.02, text, transform=ax.transAxes, color=INK_2, fontsize=9.5, va="bottom")


def suptitles(fig, title, sub):
    """Figure title + subtitle above the panel titles, with fixed spacing in inches."""
    h = fig.get_figheight()
    fig.subplots_adjust(top=1 - 0.55 / h)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK, y=1 + 0.05 / h,
                 va="bottom")
    fig.text(0.01, 1 - 0.12 / h, sub, color=INK_2, fontsize=9.5, va="bottom")


def customers_vs_spend():
    seg = pd.read_csv(REPORTS / "segments_current.csv").iloc[::-1]
    y = np.arange(len(seg))
    h = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.barh(y + h / 2 + 0.01, seg["share_customers"], height=h, color=MUTED, label="Share of customers")
    ax.barh(y - h / 2 - 0.01, seg["share_spend"], height=h, color=BLUE, label="Share of spend")
    for yi, c, s in zip(y, seg["share_customers"], seg["share_spend"]):
        ax.text(c + 0.006, yi + h / 2, f"{c:.0%}", va="center", color=INK_2, fontsize=8.5)
        ax.text(s + 0.006, yi - h / 2, f"{s:.0%}", va="center", color=INK, fontsize=8.5)
    ax.set_yticks(y, seg["segment"])
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="lower right")
    top = seg.set_index("segment").loc[["Champions", "Loyal"]].sum()
    ax.set_title(f"Champions and Loyal: {top['share_customers']:.0%} of customers, "
                 f"{top['share_spend']:.0%} of spend")
    subtitle(ax, f"RFM segments for the 12 months to 9 Dec 2011 "
                 f"({RESULTS['current']['customers']:,} customers, spend net of returns)")
    fig.savefig(FIG / "01_segments_customers_vs_spend.png")
    plt.close(fig)


def backtest():
    bt = pd.read_csv(REPORTS / "backtest_by_segment.csv").sort_values("bought_again")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    axes[0].barh(bt["segment"], bt["bought_again"], height=0.6, color=BLUE)
    for i, v in enumerate(bt["bought_again"]):
        axes[0].text(v + 0.01, i, f"{v:.0%}", va="center", color=INK, fontsize=9)
    axes[0].xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[0].set_xlim(0, 1.08)
    axes[0].set_title("Bought again in the next 12 months", fontsize=11, pad=8)
    axes[1].barh(bt["segment"], bt["next_year_revenue_per_customer"], height=0.6, color=BLUE)
    for i, v in enumerate(bt["next_year_revenue_per_customer"]):
        axes[1].text(v + 80, i, f"£{v:,.0f}", va="center", color=INK, fontsize=9)
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"£{v:,.0f}"))
    axes[1].set_xlim(0, bt["next_year_revenue_per_customer"].max() * 1.18)
    axes[1].set_title("Next-year revenue per customer", fontsize=11, pad=8)
    for ax in axes:
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    suptitles(fig, "Backtest: the segments predicted next year's behaviour",
              f"Customers segmented on 10 Dec 2010, then followed for the next 365 days "
              f"({RESULTS['backtest']['customers']:,} customers)")
    fig.savefig(FIG / "02_backtest.png")
    plt.close(fig)


def k_selection():
    k = pd.read_csv(REPORTS / "k_selection.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, col, title in [(axes[0], "inertia", "Within-cluster spread (elbow)"),
                           (axes[1], "silhouette", "Silhouette score (higher = better separated)")]:
        ax.plot(k["k"], k[col], color=BLUE, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2)
        chosen = k.loc[k["k"] == 4, col].iloc[0]
        ax.scatter([4], [chosen], s=110, facecolor="none", edgecolor=ORANGE, linewidth=2, zorder=3)
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_xlabel("Number of clusters (k)")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[1].annotate("k = 4 chosen", (4, k.loc[k["k"] == 4, "silhouette"].iloc[0]), xytext=(10, 12),
                     textcoords="offset points", color=INK, fontsize=9.5)
    suptitles(fig, "Choosing k: k = 2 scores best but only splits active from inactive",
              "k-means on log-scaled, standardised recency, frequency and spend; k = 3 and 4 are nearly tied")
    fig.savefig(FIG / "03_choosing_k.png")
    plt.close(fig)


def cluster_profiles():
    clu = pd.read_csv(REPORTS / "clusters_current.csv")
    names = clu["cluster_name"].tolist()
    colors = [CLUSTER_COLORS[n] for n in names]
    panels = [("median_recency_days", "Days since last order", lambda v: f"{v:.0f}"),
              ("median_orders", "Orders in 12 months", lambda v: f"{v:.0f}"),
              ("median_spend", "Spend in 12 months", lambda v: f"£{v:,.0f}"),
              ("share_spend", "Share of total spend", lambda v: f"{v:.0%}")]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6), sharey=True)
    y = np.arange(len(names))[::-1]
    for ax, (col, title, fmt) in zip(axes, panels):
        vals = clu[col].to_numpy()
        ax.barh(y, vals, height=0.6, color=colors)
        for yi, v in zip(y, vals):
            ax.text(v + vals.max() * 0.03, yi, fmt(v), va="center", color=INK, fontsize=9)
        ax.set_xlim(0, vals.max() * 1.35)
        ax.set_xticks([])
        ax.set_title(title, fontsize=10.5, pad=8)
        for s in ["left", "bottom"]:
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, [f"{n} ({c:,})" for n, c in zip(names, clu["customers"])])
    suptitles(fig, "Four k-means clusters, from VIPs to lapsed one-time buyers",
              "Medians per cluster, 12 months to 9 Dec 2011 (number of customers in brackets)")
    fig.savefig(FIG / "04_cluster_profiles.png")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    customers_vs_spend()
    backtest()
    k_selection()
    cluster_profiles()
    print(f"Saved charts to {FIG}")


if __name__ == "__main__":
    main()
