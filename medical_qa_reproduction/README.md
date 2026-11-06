# Independent reproduction and reanalysis

This project analyzes RQ1 and RQ3 from *This Treatment Works, Right? Evaluating
LLM Sensitivity to Patient Question Framing in Medical QA*
([paper, v1](https://arxiv.org/abs/2604.05051v1)). It calculates Table 3 and
Figures 2(b), 3 and 5 independently, while deliberately changing the questionable
analysis choices listed in [not_replicated.txt](not_replicated.txt).

## Current run status

The latest change only reorganized files; no analysis was rerun. The active
text-metric calculation was stopped during the reorganization. Six of eight models
were completed. Their metric CSVs are preserved separately in
`data_post/partial_metrics/`; there is no completed `response_metrics.csv` yet.
The existing tables and figures in `outputs/` are from the earlier analysis and have not been
regenerated for the new independent metrics or methodological settings. See
`data_post/run_status.txt`.

## Run

From this folder:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python compute_metrics.py
.venv/bin/python reproduce.py
```

`compute_metrics.py` runs pretrained models locally over 188,416 technical
answers to calculate 94,208 response-pair measurements. This is the slow step.
Sentence Transformers downloads its model on the first run and selects an
available GPU or CPU. Model weights are cached in the ignored `tmp/` folder.

`reproduce.py` performs the aggregation, regressions and plotting. Rerun only
this script when changing analysis settings. Rerun `compute_metrics.py` first
when changing the text models or extraction settings.

## Settings

Model names, model revision, sequence limit, inference batch sizes, spaCy
settings, extraction regexes, overlap convention, regression formulas,
confidence level and analysis switches are at the beginning of `reproduce.py`.
`compute_metrics.py` imports these settings instead of maintaining a second copy.

The default analysis differs from the paper in three explicit ways:

```python
ROUND_CONDITION_MEANS = False  # subtract full-precision means before rounding
MATCH_LANGUAGE_REVIEWS = True # same reviews and question types in both styles
CLUSTER_BY_REVIEW = True      # account for repeated observations per review
```

These switches allow controlled comparisons without altering the response text
measurements. Their defaults produce a reanalysis, not an exact replication of
the paper's statistical choices.

## Data and reuse boundary

The directory structure is:

- `data_pre/`: supplied input copies.
- `data_post/`: intermediate data we calculate, including response-pair metrics.
- `outputs/`: final tables, figures, their CSV values and regression summaries.

All required input files are copied into this project's `data_pre/` directory.
The originals under the repository's `code/outputs/` remain unchanged.
Both scripts read the copies:

- `data_pre/responses/`: eight technical response files, with positive/negative pairs.
- `data_pre/baseline_responses/`: eight files containing two positive responses per pair.
- `data_pre/evaluation/` and `data_pre/baseline_evaluation/`: 32 files containing the
  supplied technical/plain-language evidence-direction judgments.
- `data_pre/paper_table3.csv`: the published table, transcribed for comparison only.

Only generated answer text, review/question IDs and evidence-direction labels
are reused. Saved cosine scores, entity scores, citation lists and number lists
are not used. No author analysis scripts or notebooks are imported or executed.

## Independently calculated text measurements

`compute_metrics.py` analyzes the `Full Answer` section where present. Answers
without this marker are used as supplied.

- Cosine similarity uses [NeuML/pubmedbert-base-embeddings](https://huggingface.co/NeuML/pubmedbert-base-embeddings),
  identified through the paper's Mezzetti (2023) reference. The revision is pinned;
  inference uses float32, mean pooling and a 512-token limit. Cosine is the dot
  product of normalized embeddings.
- Entity overlap uses [spaCy's en_core_web_sm 3.8.0](https://spacy.io/models/en#en_core_web_sm).
  All entity types are included and their surface text is lowercased.
- Citation overlap uses bracketed article IDs, splitting comma lists and expanding ranges.
- Numerical overlap uses integer/decimal strings after removing bracketed citations.
- Each overlap is the set intersection divided by the set union. Two empty
  sets score zero by default; the sensitivity output also shows the alternative
  of scoring them as one.

The paper does not specify all these extraction details, including its spaCy
model. These are stated reproduction choices, not claims to have recovered an
undocumented implementation. They are not tuned to force agreement with the paper.

## Analyses

Table 3 uses all 16 technical templates: 12 single-turn templates plus the four
conversation turns. Condition means are subtracted at full precision, then the
final difference is rounded for display. `table3_comparison.csv` reports our
rounded difference minus the published value for each cell. Published values
are never substituted into the calculations.

Figure 2(b) uses only single-turn technical questions. Agreement means the two
evidence-direction labels are equal, including `uncertain` paired with `uncertain`.
Each condition contains 35,328 pairs across the eight models.

Figure 3 fits a framing logistic regression separately for each single-turn
question type. Its odds ratios come from statsmodels; confidence intervals and
p-values use standard errors clustered by review, with 95% confidence by default.

Figure 5 and every RQ3 regression use the same 363 reviews and the same two
question types (Effectiveness and Efficacy) in both language styles. Each
language/condition cell contains 5,808 pairs. This avoids mixing the paper's
368-review technical population with its 363-review plain-language population,
or mixing all 16 technical types with only two plain-language types.

The RQ3 interaction uses `agreement ~ framed * plain`. Plain-language framing
is also fitted separately; a coefficient pooled across language styles is not
reported as a plain-only effect.

## Results and outputs

`data_post/` holds the calculated pair measurements in `response_metrics.csv`
once the metric run is complete. Its existing partial files remain preserved.
`outputs/` holds PNG/SVG versions of the table and three figures, their CSV
values, the published-table comparison, sensitivity results and `regressions.json`.
`not_replicated.txt` recaps the omitted procedures and methodological departures.
