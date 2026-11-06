#!/usr/bin/env python3
"""Independent reproduction of Table 3 and Figures 2(b), 3 and 5.

Reads only the supplied pair-level evaluation JSON files, never author code or
recomputed_analysis/. See README.md for the feature-reuse boundary and methods.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from statistics import mean

# Keep Matplotlib's writable cache inside this project.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / "tmp" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).parent / "tmp" / "cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FormatStrFormatter

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
LABELS = {"higher", "lower", "same", "uncertain"}
Z95 = 1.959963984540054
# Transcribed from Table 3 for comparison only; never used in calculations.
PAPER_TABLE3 = [
    [-.02, -.06, -.01, -.08], [-.01, -.01, .00, -.02],
    [-.01, -.02, -.02, -.01], [-.01, -.02, .00, -.02],
    [-.02, -.06, -.01, -.06], [-.01, -.07, -.02, -.06],
    [-.03, -.09, -.04, -.04], [-.03, -.06, -.02, -.04],
]


def jaccard(first, second, empty=0):
    """Intersection / union of distinct tokens; expose empty-union convention."""
    a, b = set(first), set(second)
    return len(a & b) / len(a | b) if a | b else float(empty)


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_data(bundle):
    # Accept either the original assignment bundle or this repository's layout.
    bundle = bundle.resolve()
    candidates = [bundle / "dataset", bundle / "code" / "outputs", bundle]
    data_root = next((p for p in candidates if all(
        (p / folder).is_dir() for folder in ["evaluation", "baseline_evaluation"])), None)
    if data_root is None:
        raise FileNotFoundError(f"No evaluation and baseline_evaluation directories under {bundle}")
    rows, audit = [], []
    for model in MODELS:
        for style, suffix, expected in [("Technical", "", 5888),
                                         ("Plain Language", "_simplified", 726)]:
            key_sets = []
            for condition, folder in [("Baseline", "baseline_evaluation"),
                                      ("Framed", "evaluation")]:
                path = data_root / folder / f"{model}_eval_results{suffix}.json"
                raw = path.read_bytes()
                pairs = json.loads(raw)
                if len(pairs) != expected:
                    raise ValueError(f"Unexpected sample size: {path}: {len(pairs)}")
                key_sets.append(set(pairs))
                directions, templates, reviews = Counter(), Counter(), set()
                for key, pair in pairs.items():
                    review, template = key.split("_", 1)
                    first, second = (pair[k] for k in
                                     ["first_response_metrics", "second_response_metrics"])
                    a, b = first["evidence_direction"], second["evidence_direction"]
                    if a not in LABELS or b not in LABELS:
                        raise ValueError(f"Invalid/missing evidence label at {path}:{key}")
                    if template not in TYPES and template not in {
                            f"multiturn-{i}" for i in range(4)}:
                        raise ValueError(f"Unknown template: {template}")
                    directions.update([a, b])
                    templates.update([template])
                    reviews.add(review)
                    comparison = pair["comparison"]
                    scores = {
                        "cosine": comparison["cosine_similarity"],
                        "entity": comparison["entity_jaccard_similarity"],
                        "citation": jaccard(first["unique_references"], second["unique_references"]),
                        "numerical": jaccard(first["numerical_instances"], second["numerical_instances"]),
                    }
                    # Saved float32 cosine scores can exceed 1 by a few ulps.
                    if not all(math.isfinite(v) and -1.000001 <= v <= 1.000001 for v in scores.values()):
                        raise ValueError(f"Invalid score at {path}:{key}")
                    rows.append(dict(model=model, style=style, condition=condition,
                                     key=key, review=review, template=template,
                                     agreement=int(a == b), **scores,
                                     empty_citation=not (first["unique_references"] or second["unique_references"]),
                                     empty_numerical=not (first["numerical_instances"] or second["numerical_instances"])))
                audit.append(dict(file=path.relative_to(bundle).as_posix(), sha256=hashlib.sha256(raw).hexdigest(),
                                  pairs=len(pairs), reviews=len(reviews),
                                  templates=dict(templates), labels=dict(directions)))
            if key_sets[0] != key_sets[1]:
                raise ValueError(f"Baseline/Framed pair IDs differ: {model}, {style}")
    return rows, audit


def select(rows, **filters):
    return [r for r in rows if all(r[k] == v for k, v in filters.items())]


def counts(rows):
    a = sum(r["agreement"] for r in rows)
    return a, len(rows) - a


def logistic_effect(baseline, framed):
    """Exact MLE for logit(P(agreement)) = intercept + beta * framed.

    With a binary predictor this is a saturated two-cell binomial model:
    beta is the log odds ratio and its variance is the sum of inverse counts.
    This is the ordinary, unclustered logistic regression used in the paper.
    """
    c, d = counts(baseline)
    a, b = counts(framed)
    if min(a, b, c, d) == 0:
        raise ValueError("A zero cell requires a different estimator; no correction applied")
    beta = math.log((a * d) / (b * c))
    se = math.sqrt(1/a + 1/b + 1/c + 1/d)
    z = beta / se
    return dict(baseline_agree=c, baseline_total=c+d, framed_agree=a,
                framed_total=a+b, beta=beta, se=se, z=z,
                p=math.erfc(abs(z) / math.sqrt(2)), odds_ratio=math.exp(beta),
                ci_low=math.exp(beta-Z95*se), ci_high=math.exp(beta+Z95*se))


def effect_for(rows):
    return logistic_effect(select(rows, condition="Baseline"), select(rows, condition="Framed"))


def interaction(technical, plain):
    """Framed x Plain coefficient in saturated 2x2 logistic regression."""
    t, p = effect_for(technical), effect_for(plain)
    beta = p["beta"] - t["beta"]
    se = math.hypot(t["se"], p["se"])
    return dict(beta=beta, se=se, z=beta/se,
                p=math.erfc(abs(beta/se)/math.sqrt(2)))


def rate_rows(rows, group):
    result = []
    for model in [*MODELS, "Average"]:
        for condition in CONDITIONS:
            subset = select(rows, condition=condition)
            if model != "Average":
                subset = select(subset, model=model)
            a, b = counts(subset)
            result.append(dict(group=group, model=MODELS.get(model, model),
                               condition=condition, agreements=a, pairs=a+b,
                               agreement_percent=100*a/(a+b)))
    return result


def save_plot(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    svg = out / f"{name}.svg"
    fig.savefig(svg, bbox_inches="tight")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def make_figures(out, f2, f3, f5, table):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#cccccc", "axes.linewidth": 1.2,
                         "svg.fonttype": "none"})
    colors = ["#f1ad43", "#008399", "#d8475e", "#5547a3", "#9bc733",
              "#487e9e", "#303447", "#bd6a7b", "#ee2020"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "o"]
    styles = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-"]
    fig, ax = plt.subplots(figsize=(10, 5.7))
    for i, model in enumerate([*MODELS.values(), "Average"]):
        y = [r["agreement_percent"] for r in f2 if r["model"] == model]
        label = model.replace("Claude Sonnet", "Claude\nSonnet").replace("Maverick", "\nMaverick")
        ax.plot([0, 1], y, color=colors[i], marker=markers[i], linestyle=styles[i],
                linewidth=2.6, markersize=9, label=label)
    ax.set(xlim=(-.25, 1.3), ylim=(69.5, 78.5), ylabel="Agreement (%)",
           title="(b) Single-Turn Only", xticks=[0, 1], xticklabels=CONDITIONS)
    ax.grid(axis="y", linestyle="--", alpha=.35)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.15), ncol=3,
              frameon=False, fontsize=10)
    fig.subplots_adjust(bottom=.30)
    save_plot(fig, out, "figure2b")

    fig, ax = plt.subplots(figsize=(10, 6.3))
    for i, row in enumerate(f3):
        mid, lo, hi = [row[k] for k in ["odds_ratio", "ci_low", "ci_high"]]
        ax.errorbar(mid, i, xerr=[[mid-lo], [hi-mid]], fmt="D", color="#3d3d3d",
                    ecolor="#909090", capsize=4, markersize=9, linewidth=2)
    ax.set_xscale("log")
    ticks = [.5, .6, .7, .8, .9, 1., 1.1, 1.2]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set(xlim=(.5, 1.2), yticks=range(len(f3)),
           yticklabels=[r["question_type"] for r in f3],
           xlabel="Odds Ratio (Framed vs. Baseline)")
    ax.invert_yaxis()
    ax.axvline(1, color="#e53935", linestyle="--", linewidth=2)
    ax.grid(axis="y", linestyle="--", alpha=.35)
    fig.tight_layout()
    save_plot(fig, out, "figure3")

    fig, ax = plt.subplots(figsize=(7.3, 4.6))
    for style, color, marker, line in [("Plain Language", "#d8475e", "o", "-"),
                                        ("Technical", "#008399", "s", "--")]:
        y = [r["agreement_percent"] for r in f5
             if r["group"] == style and r["model"] == "Average"]
        ax.plot([0, 1], y, label=style, color=color, marker=marker,
                linestyle=line, markersize=9, linewidth=2.6)
    ax.set(xlim=(-.5, 1.5), ylim=(70, 80), ylabel="Agreement (%)",
           xticks=[0, 1], xticklabels=CONDITIONS)
    ax.grid(axis="y", linestyle="--", alpha=.35)
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_plot(fig, out, "figure5")

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
    save_plot(fig, out, "table3")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent.parent,
                        help="Repository root, assignment bundle, or directory containing evaluation folders")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "outputs")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows, audit = load_data(args.bundle)
    technical = select(rows, style="Technical")
    single = [r for r in technical if r["template"] in TYPES]
    technical_two = [r for r in technical if r["template"] in TWO_TYPES]
    plain = select(rows, style="Plain Language")
    assert len(rows) == 105824 and len(single) == 70656

    # Table 3: all 16 technical templates, including the four conversation turns.
    table, full, means, sensitivities = [], [], [], []
    for i, (model, label) in enumerate(MODELS.items()):
        subsets = {c: select(technical, model=model, condition=c) for c in CONDITIONS}
        mu = {c: {k: mean(r[k] for r in subsets[c]) for k in METRICS} for c in CONDITIONS}
        full.append(dict(model=label, **{k: mu["Framed"][k]-mu["Baseline"][k] for k in METRICS}))
        rounded = {k: round(round(mu["Framed"][k], 2)-round(mu["Baseline"][k], 2), 2)
                   for k in METRICS}
        table.append(dict(model=label, **rounded))
        if list(rounded.values()) != PAPER_TABLE3[i]:
            raise AssertionError(f"Table 3 differs from paper: {model}")
        for c in CONDITIONS:
            means.append(dict(model=label, condition=c, pairs=len(subsets[c]), **mu[c]))
        # Expose both empty-set conventions and both template scopes.
        for scope, population in [("all technical", technical), ("single-turn technical", single)]:
            for empty in [0, 1]:
                group = {c: select(population, model=model, condition=c) for c in CONDITIONS}
                avg = {c: {k: mean(r[k] + (empty if r.get(f"empty_{k}", False) else 0)
                                   for r in group[c]) for k in METRICS} for c in CONDITIONS}
                sensitivities.append(dict(model=label, scope=scope, empty_union=empty,
                    **{k: avg["Framed"][k]-avg["Baseline"][k] for k in METRICS}))

    f2 = rate_rows(single, "Technical single-turn")
    f3 = [dict(template=t, question_type=label,
               **effect_for(select(single, template=t))) for t, label in TYPES.items()]
    f3.sort(key=lambda r: r["odds_ratio"])
    f5 = rate_rows(plain, "Plain Language") + rate_rows(technical_two, "Technical")
    # Extra sensitivity: match the technical review IDs to available plain-language IDs.
    plain_ids = {r["review"] for r in plain}
    matched = [r for r in technical_two if r["review"] in plain_ids]
    matched_f5 = rate_rows(matched, "Technical, same 363 reviews")
    regressions = {
        "rq1_all_technical": effect_for(technical),
        "rq1_single_turn_technical": effect_for(single),
        "rq3_plain_only": effect_for(plain),
        "rq3_technical_two_types": effect_for(technical_two),
        "rq3_pooled_two_types": effect_for(technical_two + plain),
        "rq3_language_main_effect_pooled_conditions": logistic_effect(technical_two, plain),
        "rq3_interaction_two_types": interaction(technical_two, plain),
        "rq3_interaction_all_technical": interaction(technical, plain),
        "rq3_interaction_same_reviews_two_types": interaction(matched, plain),
    }
    checks = dict(table3_cells_matching=32, table3_cells_total=32,
                  inputs=len(audit), pairs=len(rows),
                  all_labels_valid=True, condition_pair_ids_match=True,
                  figure3_lowest_odds_ratio=f3[0]["question_type"],
                  figure3_intervals_crossing_one=[r["question_type"] for r in f3
                                                 if r["ci_low"] <= 1 <= r["ci_high"]])
    for filename, data in [("table3.csv", table), ("table3_full_precision.csv", full),
                           ("table3_condition_means.csv", means),
                           ("table3_sensitivity.csv", sensitivities), ("figure2b.csv", f2),
                           ("figure3.csv", f3), ("figure5.csv", f5),
                           ("figure5_matched_reviews.csv", matched_f5)]:
        write_csv(args.out / filename, data)
    for filename, data in [("input_audit.json", audit), ("regressions.json", regressions),
                           ("checks.json", checks)]:
        (args.out / filename).write_text(json.dumps(data, indent=2) + "\n")
    md = ["| Model | Cosine | Entity | Citation | Numerical |",
          "|---|---:|---:|---:|---:|"]
    md += ["| " + r["model"] + " | " + " | ".join(f"{r[k]:.2f}" for k in METRICS) + " |"
           for r in table]
    (args.out / "table3.md").write_text("\n".join(md) + "\n")
    make_figures(args.out, f2, f3, f5, table)
    print(json.dumps(checks, indent=2))
    for name, values in [("Figure 2(b)", f2), ("Figure 5", f5)]:
        print(name)
        for r in values:
            if r["model"] == "Average":
                print(f'  {r["group"]}, {r["condition"]}: {r["agreement_percent"]:.6f}%')
    print(f"Outputs: {args.out.resolve()}")


if __name__ == "__main__":
    main()
