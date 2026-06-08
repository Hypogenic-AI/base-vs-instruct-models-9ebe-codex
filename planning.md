# Research Plan: Token-Level Base vs Instruct Divergence

## Motivation & Novelty Assessment

### Why This Research Matters

Instruction tuning changes not only answer quality but the full next-token distribution of a model. If we can predict where a base and instruct model should diverge, then the failures of that predictor become a compact way to find model changes that are not just ordinary assistant style, refusal boilerplate, or position effects.

This matters for alignment auditing, model diffing, and dataset diagnosis: researchers can focus human inspection and mechanistic follow-up on a small set of surprising token locations instead of scanning millions of logits.

### Gap in Existing Work

The gathered literature shows strong aggregate evidence that instruction tuning concentrates shift in response starts, safety phrases, formatting, and discourse markers. However, prior work mostly reports average token-shift plots or known categories. It rarely trains a predictor of token-level divergence and then studies residual failures as the main discovery signal.

### Our Novel Contribution

This experiment treats per-token KL divergence as a supervised target. We compute real token distributions from a same-tokenizer base/instruct pair, train a lightweight predictor using base-only and surface features, and manually/quantitatively analyze positions where actual KL is much higher or lower than predicted.

### Experiment Justification

- Experiment 1: Compute teacher-forced per-position `KL(p_instruct || p_base)` on multiple corpora. This establishes the empirical divergence landscape and avoids sampling confounds.
- Experiment 2: Train a residual predictor from source, position, token class, and base-model uncertainty features. This tests whether predictable style/position effects explain most divergence.
- Experiment 3: Inspect the largest residual misses. This directly tests the submitter's hypothesis that predictor failures identify interesting differences.
- Experiment 4: Compare residual composition across instruction, safety/chat, and natural-text sources. This checks whether surprises are domain-specific or just generic language-model uncertainty.

## Research Question

Where do a real base model and its instruction-tuned counterpart differ token by token, and do failures of a learned KL-divergence predictor reveal interesting or surprising categories of difference?

## Background and Motivation

The literature review highlights URIAL/Unlocking Spell, limitations of instruction tuning, shallow safety alignment, token-level DPO, and refusal-direction work. Together these papers suggest that many base-vs-instruct differences are sparse and interpretable at the token level, especially around response openings, refusal phrases, discourse markers, and formatting.

The gap is residual discovery. Instead of only confirming known high-KL categories, this project asks what remains unexplained after a model learns the obvious regularities.

## Hypothesis Decomposition

- H1: Token-level KL between base and instruct models is predictable above simple baselines from source, position, segment, token class, and base-model uncertainty.
- H2: The largest positive residuals are enriched for semantically meaningful shifts, such as safety/refusal context, assistant persona markers, formatting boundaries, code/math, or factual/entity tokens.
- H3: The largest negative residuals correspond to places where a simple predictor expects instruction-tuning shift but the models are actually distributionally similar.
- H4: Natural text should have lower and less structured divergence than instruction/safety/chat corpora.

Independent variables: dataset/source, prompt vs response segment, token position, token/surface class, base-model entropy/logprob/rank, and safety labels where available.

Dependent variables: forward KL, reverse KL, Jensen-Shannon divergence, top-1 agreement, predictor error, and residual category enrichment.

## Proposed Methodology

### Approach

Use `Qwen/Qwen2.5-1.5B` and `Qwen/Qwen2.5-1.5B-Instruct` because they are real open-weight same-family models with a shared tokenizer and are small enough to run full-vocabulary token KL on the available A6000 GPU. Teacher-forced scoring keeps both models on identical contexts.

### Experimental Steps

1. Load local dataset samples from Alpaca, JUST-EVAL-INSTRUCT, BeaverTails, Chatbot Arena, and WikiText-2.
2. Convert each example into a common text trace with a prompt-like prefix and, where available, a response segment.
3. Tokenize each trace with the shared Qwen tokenizer and cap traces at 256 tokens for a broad but feasible first pass.
4. Run both models in eval mode with mixed precision on GPU and compute per-position forward KL, reverse KL, JS divergence, base entropy, observed-token base logprob, observed-token rank, and top-1 agreement.
5. Split by example id to avoid token leakage between train and test.
6. Train a simple positional/source baseline and a richer divergence predictor on `log1p(KL)`.
7. Evaluate MAE, RMSE, R2, Spearman correlation, and residual distributions.
8. Analyze top absolute, positive, and negative residuals using token/context snippets and heuristic token categories.

### Baselines

- Mean-only baseline: predicts the training mean `log1p(KL)`.
- Position/source baseline: predicts from source, segment, and token position only.
- Rich predictor: adds token surface class, base uncertainty, observed-token rank/logprob, and local context features.

### Evaluation Metrics

- `KL(p_instruct || p_base)`: primary divergence target.
- Reverse KL and JS divergence: stability/symmetry checks.
- MAE/RMSE on `log1p(KL)`: regression accuracy under a skewed target.
- R2 and Spearman correlation: explained variance and rank ordering.
- Top residual enrichment: whether high-error tokens are overrepresented in specific categories.

### Statistical Analysis Plan

Use bootstrap confidence intervals over held-out examples for model MAE and paired bootstrap differences between predictors. Use Mann-Whitney U tests for source-level KL differences when distributions are non-normal, with Holm correction for multiple comparisons. Report effect sizes using Cliff's delta for source comparisons and paired mean error differences for predictor comparisons.

## Expected Outcomes

Support for the hypothesis would look like: the rich predictor improves over baselines, but its largest residuals are not random; they cluster around interpretable categories such as safety refusals, assistant identity, formatting boundaries, code/math, and factual/entity completions. Refutation would look like residuals dominated by noise, tokenization artifacts, or only the same categories already captured by source/position effects.

## Timeline and Milestones

- Setup and validation: 10-20 minutes.
- Data loading and scoring pipeline: 45-75 minutes.
- KL extraction on a medium sample: 30-90 minutes depending on model download and GPU speed.
- Predictor training and statistical analysis: 20-40 minutes.
- Report, README, and reproducibility validation: 20-30 minutes.

## Potential Challenges

- Model download or CUDA incompatibility: fall back to the smaller `Qwen/Qwen2.5-0.5B` pair while preserving the method.
- Long scoring time: reduce examples per source or max sequence length, and document the computational scale.
- KL numerical instability: compute in float32 from model logits and use JS divergence as a bounded check.
- Residual interpretation bias: report both quantitative category enrichment and representative snippets, including negative results.
- Unsafe safety prompts: avoid reproducing harmful instructions in detail; use redacted snippets where needed.

## Success Criteria

- Real base and instruct model logits are used; no simulated model behavior.
- Per-token KL and related metrics are saved in `results/`.
- At least one learned predictor is trained and evaluated against baselines.
- Residual outliers are analyzed quantitatively and qualitatively.
- `REPORT.md`, `README.md`, code, figures, and reproducibility instructions document actual results.
