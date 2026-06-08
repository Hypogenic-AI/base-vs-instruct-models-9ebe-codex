# Code Walkthrough

## Overview

The experiment code lives in `src/run_divergence_experiment.py`. It is a single runnable pipeline:

```text
local datasets -> normalized text traces -> shared tokenizer -> base/instruct logits
-> per-token KL metrics -> grouped train/test split -> KL predictors
-> residual/error analysis -> figures and report-ready CSVs
```

## Key Functions

### `build_examples(cfg)`

Loads Alpaca, JUST-EVAL-INSTRUCT, BeaverTails, WikiText-2, and Chatbot Arena from `datasets/`, then converts them into common text traces with `prompt_end` markers. Unsafe BeaverTails snippets are later redacted for qualitative outputs.

### `encode_examples(tokenizer, examples, max_length)`

Tokenizes each trace with offsets so token positions can be labeled as `prompt`, `response`, or `generation_start`.

### `score_kl(cfg, tokenizer, encoded)`

Loads the real base and instruct models, runs batched teacher-forced forward passes, and saves token rows with:

- forward KL, reverse KL, and Jensen-Shannon divergence
- base entropy, top probability, and margin
- observed-token base logprob and rank
- decoded base/instruct top-1 tokens
- source, segment, token class, and sanitized context snippet

### `train_predictors(df, cfg)`

Splits rows by `example_id`, trains the mean baseline, position/source baseline, and rich predictor on `log1p(KL)`, then saves held-out predictions and residuals.

### `residual_analysis(predictions, cfg)`

Finds the largest positive, negative, and absolute rich-predictor residuals. It also computes enrichment of sources, segments, and token classes among the top 2% absolute residuals.

### `make_figures(df, predictions, cfg)`

Creates the report figures in `figures/`:

- `kl_by_source.png`
- `kl_position_trend.png`
- `predictor_scatter.png`
- `residual_token_class_enrichment.png`

## How to Run

```bash
source .venv/bin/activate
python src/run_divergence_experiment.py \
  --examples-per-source 60 \
  --max-length 224 \
  --batch-size 8 \
  --bootstrap-iterations 500
```

For a quick smoke test:

```bash
source .venv/bin/activate
python src/run_divergence_experiment.py \
  --examples-per-source 1 \
  --max-length 64 \
  --batch-size 1 \
  --bootstrap-iterations 20
```

## Resource Requirements

The final run used one NVIDIA RTX A6000 on `cuda:0`. Batch size 8 was chosen because the script computes full-vocabulary KL for two 1.5B models at every token position.

## Validation

The full run was repeated with the same seed and configuration. The row count, source stats, predictor metrics, bootstrap summaries, and residual summaries matched exactly.
