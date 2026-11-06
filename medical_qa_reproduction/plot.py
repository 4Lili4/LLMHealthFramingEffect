#!/usr/bin/env python3
"""Plot the saved results from reproduce.py without rerunning the analysis."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FormatStrFormatter
import pandas as pd

from reproduce import OUTPUT_DIR, MODELS, CONDITIONS, METRICS, TABLE_DECIMALS, PNG_DPI


def save_plot(fig, name):
    for extension in ["png", "svg"]:
        fig.savefig(OUTPUT_DIR / f"{name}.{extension}", dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)


def make_figures(figure2b, figure3, figure5, table):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#cccccc", "axes.linewidth": 1.2,
                         "svg.fonttype": "none"})
    colors = ["#f1ad43", "#008399", "#d8475e", "#5547a3", "#9bc733",
              "#487e9e", "#303447", "#bd6a7b", "#ee2020"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "o"]
    styles = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-"]
    # Figure 2(b): one Baseline-to-Framed line per model, plus the pooled average.
    fig, ax = plt.subplots(figsize=(10, 5.7))
    for i, model in enumerate([*MODELS.values(), "Average"]):
        y = [r["agreement_percent"] for r in figure2b if r["model"] == model]
        label = model.replace("Claude Sonnet", "Claude\nSonnet").replace("Maverick", "\nMaverick")
        ax.plot([0, 1], y, color=colors[i], marker=markers[i], linestyle=styles[i],
                linewidth=2.6, markersize=9, label=label)
    ax.set(xlim=(-.25, 1.3), ylim=(69.5, 78.5), ylabel="Agreement (%)",
           title="(b) Single-Turn Only", xticks=[0, 1], xticklabels=CONDITIONS)
    ax.grid(axis="y", linestyle="--", alpha=.35)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.15), ncol=3,
              frameon=False, fontsize=10)
    fig.subplots_adjust(bottom=.30)
    save_plot(fig, "figure2b")

    # Figure 3: odds ratios and confidence intervals; one means no framing effect.
    fig, ax = plt.subplots(figsize=(10, 6.3))
    for i, row in enumerate(figure3):
        mid, lo, hi = [row[k] for k in ["odds_ratio", "ci_low", "ci_high"]]
        ax.errorbar(mid, i, xerr=[[mid-lo], [hi-mid]], fmt="D", color="#3d3d3d",
                    ecolor="#909090", capsize=4, markersize=9, linewidth=2)
    ax.set_xscale("log")
    ticks = [.5, .6, .7, .8, .9, 1., 1.1, 1.2]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set(xlim=(.5, 1.2), yticks=range(len(figure3)),
           yticklabels=[r["question_type"] for r in figure3],
           xlabel="Odds Ratio (Framed vs. Baseline)")
    ax.invert_yaxis()
    ax.axvline(1, color="#e53935", linestyle="--", linewidth=2)
    ax.grid(axis="y", linestyle="--", alpha=.35)
    fig.tight_layout()
    save_plot(fig, "figure3")

    # Figure 5: compare language styles using their two shared question types.
    fig, ax = plt.subplots(figsize=(7.3, 4.6))
    for style, color, marker, line in [("Plain Language", "#d8475e", "o", "-"),
                                        ("Technical", "#008399", "s", "--")]:
        y = [r["agreement_percent"] for r in figure5
             if r["group"] == style and r["model"] == "Average"]
        ax.plot([0, 1], y, label=style, color=color, marker=marker,
                linestyle=line, markersize=9, linewidth=2.6)
    ax.set(xlim=(-.5, 1.5), ylim=(70, 80), ylabel="Agreement (%)",
           xticks=[0, 1], xticklabels=CONDITIONS)
    ax.grid(axis="y", linestyle="--", alpha=.35)
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_plot(fig, "figure5")

    # Table 3: render the same rounded differences written to CSV.
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.axis("off")
    cell_text = [[r["model"], *[f"{r[k]:.{TABLE_DECIMALS}f}" for k in METRICS]] for r in table]
    display = ax.table(cellText=cell_text,
                       colLabels=["Model", "Cosine similarity", "Entity overlap",
                                  "Citation overlap", "Numerical overlap"],
                       colWidths=[.30, .19, .17, .17, .17], cellLoc="center", loc="center")
    display.auto_set_font_size(False)
    display.set_fontsize(10)
    display.scale(1, 1.65)
    for (row, col), cell in display.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_facecolor("#e9edf1" if row % 2 == 0 else "white")
        if row == 0:
            cell.set_text_props(weight="bold")
        if col == 0:
            cell.set_text_props(ha="left")
    ax.set_title("Table 3: Framed minus Baseline", fontweight="bold", pad=12)
    fig.tight_layout()
    save_plot(fig, "table3")


def main():
    # Read finished result tables; no model inference or regressions run here.
    figure2b = pd.read_csv(OUTPUT_DIR / "figure2b.csv").to_dict("records")
    figure3 = pd.read_csv(OUTPUT_DIR / "figure3.csv").to_dict("records")
    figure5 = pd.read_csv(OUTPUT_DIR / "figure5.csv").to_dict("records")
    table3 = pd.read_csv(OUTPUT_DIR / "table3.csv").to_dict("records")
    make_figures(figure2b, figure3, figure5, table3)
    print(f"Plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
