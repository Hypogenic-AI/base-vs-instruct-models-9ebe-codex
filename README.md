# Base vs Instruct Token Divergence

This workspace tests whether token-level distribution divergence between a base model and its instruction-tuned counterpart can be predicted, and whether predictor failures reveal interesting differences. The experiment uses real logits from `Qwen/Qwen2.5-1.5B` and `Qwen/Qwen2.5-1.5B-Instruct`.

Full findings are in [REPORT.md](REPORT.md).

## Key Findings

- The rich KL predictor improved held-out MAE from 0.0601 to 0.0460 versus a position/source baseline.
- Generation-start positions were highly divergent, especially in BeaverTails safety traces.
- The largest positive predictor misses were unexpectedly concentrated in WikiText narrative continuations, not only assistant-style or refusal tokens.
- Large negative misses often occurred at literal `Assistant:` boundaries where both base and instruct models agreed.
- The full experiment was rerun with identical seed/config; key summaries matched exactly.

## Reproduce

```bash
uv sync
source .venv/bin/activate
python src/run_divergence_experiment.py \
  --examples-per-source 60 \
  --max-length 224 \
  --batch-size 8 \
  --bootstrap-iterations 500
```

The run downloads/cache-loads the two Qwen checkpoints if needed. Final runtime in this environment was 66.4 seconds after model cache availability on `cuda:0` with an NVIDIA RTX A6000.

## File Structure

```text
planning.md                         Motivation, novelty, and experimental plan
REPORT.md                           Final research report with actual results
src/run_divergence_experiment.py     Scoring, predictor training, analysis, plots
results/token_kl_metrics.csv.gz      Per-token KL and auxiliary metrics
results/heldout_predictions.csv      Held-out predictions and residuals
results/top_abs_residuals.csv        Largest rich-predictor misses
results/summary.json                 Main machine-readable summary
figures/*.png                        Report figures
datasets/README.md                   Local dataset loading/reproduction notes
resources.md                         Gathered resources plus execution notes
```

## Environment

Dependencies are managed by `uv` in this workspace only. The final run used Python 3.12.8, PyTorch 2.12.0+cu130, Transformers 5.10.2, scikit-learn 1.9.0, and CUDA 13.0.
