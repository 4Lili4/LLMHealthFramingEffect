#!/usr/bin/env python3
"""Reproduce Table 3 and Figures 2(b), 3 and 5 from the saved evaluation data."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FormatStrFormatter
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR.parent / "code" / "outputs"
OUTPUT_DIR = PROJECT_DIR / "outputs"

MODELS = {
    "claude_4.5_sonnet": "Claude Sonnet 4.5",
    "gpt-5.1": "GPT-5.1",
    "huatuo-7B": "HuatuoGPT-o1 7B",
    "huatuo-8B": "HuatuoGPT-o1 8B",
    "api-llama3.3": "Llama 3.3 70B",
    "api-llama4": "Llama 4 Maverick 17B",
    "qwen3-4B": "Qwen3 4B",
    "qwen3-30B": "Qwen3 30B",
}
TYPES = {
    "effectiveness": "Effectiveness", "doctor": "Second Opinion",
    "studies": "Scientific Evidence", "friend": "Friend Anecdote",
    "safety": "Safety", "ai": "AI", "testimonials": "Testimonials",
    "timepressure": "Time Pressure", "family": "Family Anecdote",
    "journals": "Online Journals", "efficacy": "Efficacy", "cost": "Cost",
}
METRICS = ("cosine", "entity", "citation", "numerical")
CONDITIONS = ("Baseline", "Framed")
TWO_TYPES = {"effectiveness", "efficacy"}


def jaccard(first, second):
    """Shared distinct tokens / all distinct tokens; empty pairs score zero."""
    a, b = set(first), set(second)
    return len(a & b) / len(a | b) if a | b else 0.0


def load_data():
    """Turn the supplied response pairs into one DataFrame."""
    rows = []
    for model, label in MODELS.items():
        for style, suffix in [("Technical", ""), ("Plain Language", "_simplified")]:
            for condition, folder in [("Baseline", "baseline_evaluation"),
                                      ("Framed", "evaluation")]:
                path = DATA_DIR / folder / f"{model}_eval_results{suffix}.json"
                for key, pair in json.loads(path.read_text()).items():
                    review, template = key.split("_", 1)
                    first = pair["first_response_metrics"]
                    second = pair["second_response_metrics"]
                    # Saved cosine/entity scores are reused; overlaps are recalculated.
                    rows.append({
                        "model": label, "style": style, "condition": condition,
                        "review": review, "template": template,
                        "framed": int(condition == "Framed"),
                        "plain": int(style == "Plain Language"),
                        # Equal 'uncertain' judgments also count as agreement.
                        "agreement": int(first["evidence_direction"] == second["evidence_direction"]),
                        "cosine": pair["comparison"]["cosine_similarity"],
                        "entity": pair["comparison"]["entity_jaccard_similarity"],
                        "citation": jaccard(first["unique_references"], second["unique_references"]),
                        "numerical": jaccard(first["numerical_instances"], second["numerical_instances"]),
                        "empty_citation": not (first["unique_references"] or second["unique_references"]),
                        "empty_numerical": not (first["numerical_instances"] or second["numerical_instances"]),
                    })
    return pd.DataFrame(rows)


def differences(means):
    """Framed minus Baseline for each model's mean scores."""
    return means.xs("Framed", level="condition") - means.xs("Baseline", level="condition")


def logistic_regression(data, formula, term):
    """Let statsmodels calculate the coefficient, uncertainty and significance."""
    fit = smf.logit(formula, data=data).fit(disp=False)
    low, high = fit.conf_int().loc[term]
    return {
        "beta": fit.params[term], "se": fit.bse[term],
        "z": fit.tvalues[term], "p": fit.pvalues[term],
        "odds_ratio": np.exp(fit.params[term]),
        "ci_low": np.exp(low), "ci_high": np.exp(high),
    }


def framing_effect(data, predictor="framed"):
    """Fit the two-group comparison and include its agreement counts."""
    counts = data.groupby(predictor)["agreement"].agg(["sum", "size"])
    return {
        "baseline_agree": int(counts.loc[0, "sum"]),
        "baseline_total": int(counts.loc[0, "size"]),
        "framed_agree": int(counts.loc[1, "sum"]),
        "framed_total": int(counts.loc[1, "size"]),
        **logistic_regression(data, f"agreement ~ {predictor}", predictor),
    }


def interaction(technical, plain):
    # The interaction tests whether the framing effect differs by language style.
    return logistic_regression(pd.concat([technical, plain]),
                               "agreement ~ framed * plain", "framed:plain")


def rate_rows(data, group):
    """Agreement percentages by model, followed by the pooled average."""
    models = data.groupby(["model", "condition"], sort=False)["agreement"].agg(
        agreements="sum", pairs="size").reset_index()
    average = data.groupby("condition", sort=False)["agreement"].agg(
        agreements="sum", pairs="size").reset_index().assign(model="Average")
    rates = pd.concat([models, average], ignore_index=True)
    rates["group"] = group
    rates["agreement_percent"] = 100 * rates["agreements"] / rates["pairs"]
    return rates[["group", "model", "condition", "agreements", "pairs", "agreement_percent"]].to_dict("records")


def save_plot(fig, name):
    for extension in ["png", "svg"]:
        fig.savefig(OUTPUT_DIR / f"{name}.{extension}", dpi=200, bbox_inches="tight")
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

    # Figure 3: odds ratios with 95% intervals; one means no framing effect.
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
    cell_text = [[r["model"], *[f"{r[k]:.2f}" for k in METRICS]] for r in table]
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
    OUTPUT_DIR.mkdir(exist_ok=True)
    data = load_data()
    technical = data[data["style"] == "Technical"]
    single = technical[technical["template"].isin(TYPES)]
    technical_two = technical[technical["template"].isin(TWO_TYPES)]
    plain = data[data["style"] == "Plain Language"]

    # Table 3 includes all 16 technical templates, including conversation turns.
    grouped = technical.groupby(["model", "condition"], sort=False)
    means = grouped[list(METRICS)].mean()
    full = differences(means)
    # The published table rounds each mean BEFORE subtracting the conditions.
    table = differences(means.round(2)).round(2)
    condition_means = means.join(grouped.size().rename("pairs")).reset_index()

    # Report how the overlap convention and single-turn restriction change Table 3.
    sensitivities = []
    for scope, population in [("all technical", technical), ("single-turn technical", single)]:
        for empty in [0, 1]:
            adjusted = population.copy()
            for metric in ["citation", "numerical"]:
                adjusted[metric] += empty * adjusted[f"empty_{metric}"]
            avg = adjusted.groupby(["model", "condition"], sort=False)[list(METRICS)].mean()
            sensitivities.append(differences(avg).reset_index().assign(scope=scope, empty_union=empty))

    figure2b = rate_rows(single, "Technical single-turn")
    figure3 = [dict(template=template, question_type=label,
                    **framing_effect(single[single["template"] == template]))
               for template, label in TYPES.items()]
    figure3.sort(key=lambda row: row["odds_ratio"])
    # Figure 5 compares only Effectiveness and Efficacy, the two shared types.
    figure5 = rate_rows(plain, "Plain Language") + rate_rows(technical_two, "Technical")
    matched = technical_two[technical_two["review"].isin(plain["review"])]
    shared_types = pd.concat([technical_two, plain])

    # Keep both RQ3 populations to explain the discrepancy in the paper's statistics.
    regressions = {
        "rq1_all_technical": framing_effect(technical),
        "rq1_single_turn_technical": framing_effect(single),
        "rq3_plain_only": framing_effect(plain),
        "rq3_technical_two_types": framing_effect(technical_two),
        "rq3_pooled_two_types": framing_effect(shared_types),
        "rq3_language_main_effect_pooled_conditions": framing_effect(shared_types, "plain"),
        "rq3_interaction_two_types": interaction(technical_two, plain),
        "rq3_interaction_all_technical": interaction(technical, plain),
        "rq3_interaction_same_reviews_two_types": interaction(matched, plain),
    }
    tables = {
        "table3": table.reset_index(),
        "table3_full_precision": full.reset_index(),
        "table3_condition_means": condition_means[["model", "condition", "pairs", *METRICS]],
        "table3_sensitivity": pd.concat(sensitivities)[["model", "scope", "empty_union", *METRICS]],
        "figure2b": pd.DataFrame(figure2b),
        "figure3": pd.DataFrame(figure3),
        "figure5": pd.DataFrame(figure5),
        "figure5_matched_reviews": pd.DataFrame(rate_rows(matched, "Technical, same 363 reviews")),
    }
    for name, frame in tables.items():
        frame.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
    (OUTPUT_DIR / "regressions.json").write_text(json.dumps(regressions, indent=2) + "\n")
    make_figures(figure2b, figure3, figure5, table.reset_index().to_dict("records"))
    print(f"Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
