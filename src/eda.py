"""Exploratory analysis: produces the figures used in the report and the Loom.

Run:  python src/eda.py
Writes PNGs to report/figures/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "report" / "figures"

# Categorical palette (fixed order) + neutral ink colours
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1f2328", "#6b6f76", "#e3e5e8"
EQUIP_COLORS = {"Dry Van": BLUE, "Reefer": ORANGE, "Flatbed": AQUA}

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.titlelocation": "left", "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA / "train_test.csv", parse_dates=["date"])
    valid = pd.read_csv(DATA / "validation.csv", parse_dates=["date"])
    train["rpm"] = train["posted_rate"] / train["distance"]  # rate per mile
    return train, valid


def is_typical(df: pd.DataFrame) -> pd.Series:
    """Rows whose rate per mile is in a plausible band (excludes injected outliers)."""
    return df["rpm"].between(1.2, 4.0)


def fig_target(train: pd.DataFrame) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
    a.hist(train["posted_rate"], bins=120, color=BLUE)
    a.set(title="Posted rate is right-skewed", xlabel="Posted rate ($)", ylabel="Loads")
    b.hist(np.log(train["posted_rate"]), bins=120, color=BLUE)
    b.set(title="Log scale is near-symmetric", xlabel="log(posted rate)")
    fig.savefig(FIG / "01_target_distribution.png"); plt.close(fig)


def fig_distance(train: pd.DataFrame) -> None:
    t = train[is_typical(train)].sample(6000, random_state=0)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for eq, col in EQUIP_COLORS.items():
        s = t[t["equipment"] == eq]
        ax.scatter(s["distance"], s["rpm"], s=8, alpha=0.35, color=col, label=eq, linewidths=0)
    ax.set(title="Rate per mile falls with distance; Reefer > Flatbed > Dry Van",
           xlabel="Distance (miles)", ylabel="Rate per mile ($)")
    ax.legend(frameon=False, markerscale=2.5)
    fig.savefig(FIG / "02_rate_per_mile_vs_distance.png"); plt.close(fig)


def fig_quote_regimes(train: pd.DataFrame, valid: pd.DataFrame) -> None:
    t = train[is_typical(train)]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax in axes[1:3]:
        ax.sharey(axes[0])
    panels = [("Copy (Jan–Mar, Jun, Sep)", [1, 2, 3, 6, 9], BLUE),
              ("Mirror (Apr, May, Jul, Oct)", [4, 5, 7, 10], ORANGE),
              ("Noise (Aug)", [8], MUTED)]
    for ax, (title, months, col) in zip(axes[:3], panels):
        s = t[t["date"].dt.month.isin(months)].sample(2500, random_state=0)
        ax.scatter(s["rpm"], s["quote_signal"], s=5, alpha=0.3, color=col, linewidths=0)
        ax.set(title=title, xlabel="Actual rate per mile ($)")
    axes[0].set_ylabel("quote_signal")

    # Label-free fingerprint: corr(quote_signal, log distance) per month
    ax = axes[3]
    both = pd.concat([train.assign(src="train"), valid.assign(src="valid")])
    fp = both.groupby(both["date"].dt.month).apply(
        lambda g: np.corrcoef(g["quote_signal"], np.log(g["distance"]))[0, 1],
        include_groups=False)
    colors = [BLUE if v < -0.5 else ORANGE if v > 0.5 else MUTED for v in fp]
    ax.bar(fp.index, fp.values, color=colors, width=0.7)
    ax.axvspan(10.5, 12.5, color=GRID, zorder=0)
    ax.text(11.5, 0.9, "validation", ha="center", color=MUTED, fontsize=9)
    ax.set(title="Fingerprint (no labels needed)", xlabel="Month",
           ylabel="corr(quote, log distance)", ylim=(-1, 1), xticks=range(1, 13))
    ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("quote_signal switches regime by month — Nov/Dec match the noise regime",
                 x=0.01, ha="left", fontweight="bold", y=1.04)
    fig.savefig(FIG / "03_quote_signal_regimes.png"); plt.close(fig)


def fig_time(train: pd.DataFrame, valid: pd.DataFrame) -> None:
    both = pd.concat([train, valid])
    mi = both.groupby("date")["market_index"].mean().rolling(7, center=True).mean()
    t = train[is_typical(train)]
    rpm = t.groupby("date")["rpm"].median().rolling(7, center=True).mean()

    fig, (a, b) = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True)
    for ax in (a, b):
        ax.axvspan(pd.Timestamp("2025-11-01"), pd.Timestamp("2025-12-31"), color=GRID, zorder=0)
    a.plot(mi.index, mi.values, color=BLUE)
    a.set(title="market_index (7-day avg) — known for validation dates too", ylabel="Index")
    a.text(pd.Timestamp("2025-12-01"), mi.max() * 0.98, "validation", ha="center", color=MUTED)
    b.plot(rpm.index, rpm.values, color=ORANGE)
    b.set(title="Median rate per mile (7-day avg) — labels end Oct 31", ylabel="$ per mile")
    fig.savefig(FIG / "04_market_index_and_rate_over_time.png"); plt.close(fig)


def fig_outliers(train: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.hist(np.log10(train["rpm"]), bins=150, color=BLUE)
    for x in (np.log10(1.2), np.log10(4.0)):
        ax.axvline(x, color=ORANGE, linewidth=1.5, linestyle="--")
    ticks = [0.2, 0.5, 1, 2, 4, 10, 20]
    ax.set_xticks(np.log10(ticks), [f"${v:g}" for v in ticks])
    ax.set_yscale("log")
    ax.set(title="~1.4% of loads have implausible rate per mile (outside dashed band)",
           xlabel="Rate per mile (log scale)", ylabel="Loads (log scale)")
    fig.savefig(FIG / "05_rate_per_mile_outliers.png"); plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    train, valid = load()
    fig_target(train)
    fig_distance(train)
    fig_quote_regimes(train, valid)
    fig_time(train, valid)
    fig_outliers(train)
    print(f"Figures written to {FIG}")


if __name__ == "__main__":
    main()
