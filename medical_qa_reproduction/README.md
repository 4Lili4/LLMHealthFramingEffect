# Independent medical QA reproduction

This folder reproduces Table 3 and Figures 2(b), 3 and 5 from *This Treatment
Works, Right? Evaluating LLM Sensitivity to Patient Question Framing in Medical
QA* (Yun et al., arXiv:2604.05051v1), using the paper and supplied evaluation data.
All analysis code was written independently from the methods in the paper.
No author scripts or notebooks were read, imported or executed, and nothing in
`dataset/recomputed_analysis/` is used.

## Run

From this folder, create an environment, install the plotting dependency, and run.
The default input is the repository's `code/outputs/` directory:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python reproduce.py
.venv/bin/python verify.py
```

The generated results are already included in `outputs/`; opening them does not
require installation. The script also supports the original assignment bundle:
`python reproduce.py --bundle /path/to/reproduction_bundle --out outputs`.
`--bundle` accepts a repository root, an assignment bundle containing `dataset/`,
or a directory containing the two evaluation folders directly.
The only direct third-party dependency is Matplotlib; the calculations use
Python's standard library. The numerical verification uses NumPy, which is
installed as a Matplotlib dependency.

## What is independently calculated

- All evidence agreement indicators, percentages, aggregations and plots.
- Citation and numerical Jaccard similarities, calculated from the supplied
  per-response `unique_references` and `numerical_instances` lists as sets.
- Table 3 condition means and differences.
- Logistic regression coefficients, standard errors, odds ratios, normal Wald
  confidence intervals, two-sided p-values and language interactions.

**Feature reuse boundary:** Table 3 uses the supplied pair-level
`cosine_similarity` and `entity_jaccard_similarity` measurements. It does not
regenerate PubMedBERT embeddings or run spaCy entity extraction. It also uses
the supplied citation/number token lists rather than re-extracting them from
response text. The paper names PubMedBERT and spaCy but does not fully specify
the embedding/extraction pipeline, including the spaCy model/version and the
empty-set convention. This is an independent reproduction of the analyses from
the supplied evaluation measurements, not a reproduction of all feature extraction.
The figures use only supplied evidence-direction labels, as the assignment allows.

## Samples and agreement

The script validates 32 input JSON files containing 105,824 evaluated response
pairs across eight models, two language styles and two framing conditions.
The two conditions have identical pair IDs within each model/style.

Technical data contain 368 reviews x 16 templates = 5,888 pairs per model and
condition. Twelve templates are single-turn; the other four are successive
turns in the multi-turn conversation. Plain-language data contain 363 reviews x
two templates = 726 pairs per model and condition. No labels are missing or
outside the four permitted categories.

A pair agrees if its labels are exactly equal, including `uncertain` paired
with `uncertain`. Uncertain judgments are not dropped. Baseline compares two
positive responses; Framed compares positive and negative responses. Different
labels indicate disagreement, which need not be a direct higher/lower contradiction.
All models have the same sample size in each plotted population, so the pooled
rate equals the unweighted mean of the eight model rates.

## Table 3

All **32 of 32 published cells match**, using all 16 technical templates.
Two otherwise unspecified conventions are needed to reproduce the display:

1. If both citation sets or number sets are empty, their Jaccard score is zero.
2. Round each condition mean to two decimals, then subtract Framed minus Baseline.

These conventions were inferred by comparing the independently calculated
results with the published table; they are not explicitly stated in the paper.
For example, GPT-5.1 citation overlap has a full-precision difference of about
-0.00744. Rounding that difference gives -0.01. The condition means instead
both round to 0.93, giving the published 0.00.

`table3.csv`, `table3.md` and `table3.png` reproduce the published display.
`table3_full_precision.csv` preserves differences without premature rounding;
`table3_condition_means.csv` preserves each mean. `table3_sensitivity.csv`
also reports the alternative empty-set convention (one) and single-turn scope.
The 32 paper reference cells in the script are comparison targets only and never
supply the computed results.

## Figure 2(b)

This uses only the 12 single-turn technical question types: 4,416 pairs per
model/condition, or 35,328 per condition across all eight models.

| Condition | Agreements / pairs | Average agreement |
|---|---:|---:|
| Baseline | 26,651 / 35,328 | 75.438745% |
| Framed | 25,567 / 35,328 | 72.370358% |

The 3.068387 percentage point decrease and all eight model lines agree with the
supplied figure. The single-turn logistic coefficient is -0.159243, SE 0.017156,
z -9.282035, matching the paper's reported -0.159 and -9.28.
Across all technical templates, the coefficient is -0.219429, SE 0.014906,
z -14.721261, also matching the RQ1 text.

## Figure 3

For each question type, pool the eight models and calculate the framing effect
on binary agreement, independently of the other types. Each condition has
2,944 pairs per question type.

For Framed agreement/disagreement counts a,b and Baseline counts c,d:

```
beta = log((a*d)/(b*c))
SE = sqrt(1/a + 1/b + 1/c + 1/d)
odds ratio = exp(beta)
95% CI = exp(beta +/- 1.959964*SE)
```

These are the exact maximum-likelihood coefficient and ordinary standard error
for logistic regression with an intercept and one binary predictor; numerical
optimization is unnecessary. The plot uses a logarithmic odds-ratio axis and
sorts types by their calculated odds ratio, reproducing the supplied ordering
and plotted point/interval locations.

Effectiveness has the smallest odds ratio: 0.673053, 95% CI approximately
[0.599414, 0.755738]. All 12 point estimates are below one. However, the intervals
for Family Anecdote, Online Journals, Efficacy and Cost include one: the data do
not support a separate statistically significant effect for every type at 5%.
The original figure also shows these intervals crossing one.

## Figure 5 and the RQ3 scope discrepancy

The technical series must be restricted to Effectiveness and Efficacy, the only
two types present in plain language. Keep all 368 technical reviews and 363
plain-language reviews to match the figure.

| Language | Baseline | Framed | Change, percentage points |
|---|---:|---:|---:|
| Plain language | 75.533747% | 71.883609% | -3.650138 |
| Technical | 76.171875% | 71.450408% | -4.721467 |

These match the supplied Figure 5. The paper's text gives approximately 76.1%
for technical Baseline; the underlying value here would round to 76.2% at one
decimal, a small textual discrepancy.

The surrounding regression results appear to use different populations:

| Specification | beta | SE | p |
|---|---:|---:|---:|
| Plain language only, framing effect | -0.188589 | 0.042232 | 0.00000799 |
| Both language styles, two types pooled, framing effect | -0.216815 | 0.029797 | 3.43e-13 |
| Language main effect, two types, conditions pooled | -0.005294 | 0.029724 | 0.858642 |
| Framing x language, two types in both styles | 0.056176 | 0.059598 | 0.345896 |
| Framing x language, all 16 technical types vs two plain types | 0.030840 | 0.044786 | 0.491066 |

The paper labels approximately -0.216, SE 0.030, z -7.28 as the framing effect
within plain language. Those values instead closely match pooling the two
language styles for the two shared types (z = -7.276356); they do not match the
plain-only calculation. The reported interaction p = 0.491 is reproduced by
including all technical types, whereas Figure 5 uses only the two shared types.
This is evidence of inconsistent analysis populations/reporting, inferred from
the numerical matches; the intended regression population is not fully specified.

Either interaction specification gives no statistically significant interaction,
so the paper's main RQ3 conclusion is unchanged. Restricting technical data to
the same 363 reviews as plain language also gives a nonsignificant interaction
(p = 0.380922); the additional counts are in `figure5_matched_reviews.csv`.

The regression calculations reproduce the paper's ordinary unclustered
uncertainty estimates. Repeated observations from the same review/model are
not independent; these are reproductions of that analysis, not new estimates
using clustered standard errors.

## Outputs and validation

`outputs/` contains PNG and SVG versions of all four requested artifacts, CSVs
of their underlying values, `regressions.json`, `checks.json`, and
`input_audit.json` with sample counts, label counts and SHA-256 hashes of all
32 input files. Input paths are relative to the selected repository/bundle root.
All four rendered artifacts were visually inspected.
`verify.py` independently checks the closed-form regression results against
numerical binomial logistic maximum likelihood and inverse Fisher information.
Figures are replotted from data; supplied picture files are used only as visual
references and are not read by the script.

Paper source: [arXiv:2604.05051v1](https://arxiv.org/abs/2604.05051v1), pp. 5-8.
Input data source: `code/outputs/evaluation/` and
`code/outputs/baseline_evaluation/` in this repository, corresponding to
`dataset/evaluation/` and `dataset/baseline_evaluation/` in the assignment bundle.
