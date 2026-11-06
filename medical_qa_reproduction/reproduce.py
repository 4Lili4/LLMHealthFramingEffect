#!/usr/bin/env python3
"""Reproduce Table 3 and Figures 2(b), 3 and 5 from the saved evaluation data."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data_pre"
PROCESSED_DATA_DIR = PROJECT_DIR / "data_post"
OUTPUT_DIR = PROJECT_DIR / "outputs"
METRICS_FILE = PROCESSED_DATA_DIR / "response_metrics.csv"

# Text models and inference settings (also imported by compute_metrics.py).
EMBEDDING_MODEL = "NeuML/pubmedbert-base-embeddings"
MODEL_REVISION = "b79526d6ef3645e0df4530322e266f24c829f5ef"
EMBEDDING_MAX_TOKENS = 512
EMBEDDING_BATCH_SIZE = 64
EMBEDDING_CHUNK_SIZE = 1024
SPACY_MODEL = "en_core_web_sm"
SPACY_DISABLED_COMPONENTS = ["tagger", "parser", "lemmatizer", "attribute_ruler"]
NER_BATCH_SIZE = 64
NER_PROCESSES = 4

# Extraction rules: retain the answer section, lowercase entities, remove citations
# before counting numbers, and assign zero overlap to two empty sets.
USE_FULL_ANSWER = True
LOWERCASE_ENTITIES = True
EMPTY_SET_OVERLAP = 0.0
EMPTY_SET_SENSITIVITY = (0, 1)
FULL_ANSWER_PATTERN = re.compile(r"^\s*(?:-\s*)?\*{0,2}Full Answer\s*:?\*{0,2}\s*:?\s*", re.I | re.M)
CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*(?:,|[-–])\s*\d+)*)\]")
NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")

# Analysis settings. Use consistent populations and round only final differences.
CONFIDENCE_LEVEL = 0.95
FRAMING_FORMULA = "agreement ~ framed"
LANGUAGE_FORMULA = "agreement ~ plain"
INTERACTION_FORMULA = "agreement ~ framed * plain"
LOGIT_METHOD = "newton"
LOGIT_MAX_ITERATIONS = 100
TABLE_DECIMALS = 2
ROUND_CONDITION_MEANS = False
MATCH_LANGUAGE_REVIEWS = True
CLUSTER_BY_REVIEW = True
PNG_DPI = 200

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


def load_data():
    """Read only evidence-direction judgments and the IDs needed to group them."""
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
                    rows.append({
                        "model": label, "style": style, "condition": condition,
                        "review": review, "template": template,
                        "framed": int(condition == "Framed"),
                        "plain": int(style == "Plain Language"),
                        # Equal 'uncertain' judgments also count as agreement.
                        "agreement": int(first["evidence_direction"] == second["evidence_direction"]),
                    })
    return pd.DataFrame(rows)


def differences(means):
    """Framed minus Baseline for each model's mean scores."""
    return means.xs("Framed", level="condition") - means.xs("Baseline", level="condition")


def logistic_regression(data, formula, term):
    """Let statsmodels calculate the coefficient, uncertainty and significance."""
    # Reviews contribute repeated observations; account for that in uncertainty.
    covariance = {"cov_type": "cluster", "cov_kwds": {"groups": data["review"]}} if CLUSTER_BY_REVIEW else {}
    fit = smf.logit(formula, data=data).fit(method=LOGIT_METHOD,
                                         maxiter=LOGIT_MAX_ITERATIONS, disp=False, **covariance)
    low, high = fit.conf_int(alpha=1 - CONFIDENCE_LEVEL).loc[term]
    return {
        "beta": fit.params[term], "se": fit.bse[term],
        "z": fit.tvalues[term], "p": fit.pvalues[term],
        "odds_ratio": np.exp(fit.params[term]),
        "ci_low": np.exp(low), "ci_high": np.exp(high),
    }


def framing_effect(data):
    """Fit the two-group comparison and include its agreement counts."""
    counts = data.groupby("framed")["agreement"].agg(["sum", "size"])
    return {
        "baseline_agree": int(counts.loc[0, "sum"]),
        "baseline_total": int(counts.loc[0, "size"]),
        "framed_agree": int(counts.loc[1, "sum"]),
        "framed_total": int(counts.loc[1, "size"]),
        **logistic_regression(data, FRAMING_FORMULA, "framed"),
    }


def interaction(technical, plain):
    # The interaction tests whether the framing effect differs by language style.
    return logistic_regression(pd.concat([technical, plain]),
                               INTERACTION_FORMULA, "framed:plain")


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



def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    data = load_data()
    technical = data[data["style"] == "Technical"]
    # These measurements are generated by our compute_metrics.py, not the authors.
    metrics = pd.read_csv(METRICS_FILE)
    technical = technical.merge(metrics, on=["model", "condition", "review", "template"])
    single = technical[technical["template"].isin(TYPES)]
    technical_two = technical[technical["template"].isin(TWO_TYPES)]
    plain = data[data["style"] == "Plain Language"]
    if MATCH_LANGUAGE_REVIEWS:
        # Use the same review IDs, not 368 technical reviews vs 363 plain reviews.
        common_reviews = set(technical_two["review"]) & set(plain["review"])
        technical_two = technical_two[technical_two["review"].isin(common_reviews)]
        plain = plain[plain["review"].isin(common_reviews)]

    # Table 3 includes all 16 technical templates, including conversation turns.
    grouped = technical.groupby(["model", "condition"], sort=False)
    means = grouped[list(METRICS)].mean()
    full = differences(means)
    # Default: subtract full-precision means, then round for display.
    display_means = means.round(TABLE_DECIMALS) if ROUND_CONDITION_MEANS else means
    table = differences(display_means).round(TABLE_DECIMALS)
    condition_means = means.join(grouped.size().rename("pairs")).reset_index()

    # Report how the overlap convention and single-turn restriction change Table 3.
    sensitivities = []
    for scope, population in [("all technical", technical), ("single-turn technical", single)]:
        for empty in EMPTY_SET_SENSITIVITY:
            adjusted = population.copy()
            for metric in ["citation", "numerical"]:
                adjusted[metric] = adjusted[metric].where(~adjusted[f"empty_{metric}"], float(empty))
            avg = adjusted.groupby(["model", "condition"], sort=False)[list(METRICS)].mean()
            sensitivities.append(differences(avg).reset_index().assign(scope=scope, empty_union=empty))

    figure2b = rate_rows(single, "Technical single-turn")
    figure3 = [dict(template=template, question_type=label,
                    **framing_effect(single[single["template"] == template]))
               for template, label in TYPES.items()]
    figure3.sort(key=lambda row: row["odds_ratio"])
    # Figure 5 compares only Effectiveness and Efficacy, the two shared types.
    figure5 = rate_rows(plain, "Plain Language") + rate_rows(technical_two, "Technical")
    shared_types = pd.concat([technical_two, plain])

    # Every RQ3 comparison now uses the same question types and review population.
    regressions = {
        "rq1_all_technical": framing_effect(technical),
        "rq1_single_turn_technical": framing_effect(single),
        "rq3_plain_only": framing_effect(plain),
        "rq3_technical_two_types": framing_effect(technical_two),
        "rq3_pooled_two_types": framing_effect(shared_types),
        "rq3_language_main_effect_pooled_conditions": logistic_regression(shared_types, LANGUAGE_FORMULA, "plain"),
        "rq3_interaction_two_types": interaction(technical_two, plain),
    }
    tables = {
        "table3": table.reset_index(),
        # Published values are comparison targets only, never inputs to our metrics.
        "table3_comparison": (table - pd.read_csv(DATA_DIR / "paper_table3.csv")
                              .set_index("model")).round(TABLE_DECIMALS).reset_index(),
        "table3_full_precision": full.reset_index(),
        "table3_condition_means": condition_means[["model", "condition", "pairs", *METRICS]],
        "table3_sensitivity": pd.concat(sensitivities)[["model", "scope", "empty_union", *METRICS]],
        "figure2b": pd.DataFrame(figure2b),
        "figure3": pd.DataFrame(figure3),
        "figure5": pd.DataFrame(figure5),
    }
    for name, frame in tables.items():
        frame.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
    (OUTPUT_DIR / "regressions.json").write_text(json.dumps(regressions, indent=2) + "\n")
    print(f"Tables and regressions saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
