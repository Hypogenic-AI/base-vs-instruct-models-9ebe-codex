# Literature Review: Base vs Instruct Model Divergence

## Review Scope

### Research Question

Where do base and instruction-tuned language models differ token by token, and can failures of a KL-divergence predictor reveal surprising behavioral or mechanistic differences?

### Inclusion Criteria

- Papers comparing base, chat, instruction-tuned, safety-aligned, or reasoning-tuned variants.
- Papers using token-level distribution shift, KL divergence, logit differences, or per-position preference/safety signals.
- Papers with datasets, code, or methods that can be adapted for divergence-prediction experiments.
- Foundational instruction tuning/RLHF papers needed to define baselines.

### Exclusion Criteria

- Papers focused only on unrelated downstream benchmark gains without analysis of distributional or behavioral shifts.
- Closed-only methods without enough methodological detail to reproduce.
- Very large dataset papers unless they provide useful prompt corpora or evaluation protocols.

### Search Log

| Date | Query / Source | Results | Notes |
|---|---|---:|---|
| 2026-06-08 | paper-finder: "base versus instruction tuned language models token-level KL divergence behavioral differences" | 142 | Downloaded all 17 relevance >= 2 papers and selected 9 foundational relevance-1 papers. |
| 2026-06-08 | Manual web/arXiv/Semantic Scholar follow-up | 26 PDFs | Filled PDF links, code repositories, and dataset sources. |

## Research Area Overview

The most relevant literature supports a strong working premise: instruction tuning often changes a model unevenly across token positions. Several studies find that base and instruct/chat variants agree over many ordinary content tokens but diverge at response starts, discourse markers, safety/refusal phrases, hedges, formatting, and other style or policy tokens. This makes per-token KL a plausible measurement target, but also means a divergence predictor must separate "expected style shift" from genuinely surprising changes in knowledge, refusal, reasoning, or representation.

## Key Papers

### The Unlocking Spell on Base LLMs

Lin et al. compare base and aligned token distributions for model pairs such as Llama-2 vs Llama-2-chat and argue that alignment tuning often leaves top token choices unchanged at most positions. The largest shifts concentrate in stylistic tokens such as discourse markers and safety disclaimers. They introduce URIAL, a tuning-free in-context alignment prompt using a system prompt and a few restyled examples, and JUST-EVAL-INSTRUCT for multi-aspect evaluation. This is the closest prior work to the proposed hypothesis and should be treated as the primary baseline.

### A Closer Look at the Limitations of Instruction Tuning

Ghosh et al. analyze instruction tuning across LoRA and full-parameter fine-tuning. Their key finding is that LoRA mainly learns response initiation and style tokens, while full fine-tuning can degrade knowledge and increase hallucination by borrowing tokens from instruction data. They explicitly inspect shifted token positions and find many shifted/marginal tokens trace back to the tuning dataset. This motivates recording whether high-KL positions are generic style tokens, dataset artifacts, or content-bearing tokens.

### Is In-Context Learning Sufficient for Instruction Following?

Zhao et al. test URIAL and many-shot ICL against instruction fine-tuning on MT-Bench and AlpacaEval 2.0. URIAL helps, but usually does not fully match instruction tuning for stronger base models. Their appendix reports that IFT can move a model farther from the base distribution than ICL, even with very small fine-tuning sets. This suggests a useful control: compare base, instruct, and base+URIAL. If divergence predictor errors vanish under URIAL, the surprising shift may be mostly prompt/style; if not, it may reflect weight-level post-training changes.

### Safety Alignment Should Be Made More Than Just a Few Tokens Deep

Qi et al. show safety alignment can take a shallow shortcut where the aligned model's distribution differs mostly in the first few output tokens. They compute per-token KL between fine-tuned and initially aligned models and use positional plots to explain jailbreak susceptibility. This is directly relevant for safety subsets: refusal prompts should produce high early KL, but surprising predictor failures may appear where refusal persists or collapses later in the sequence.

### Crosscoders for Chat-Tuning

Minder et al. study model diffing between Gemma 2 base and chat models. They warn that L1 crosscoders can falsely identify shared concepts as chat-only because of sparsity artifacts, and they recommend BatchTopK plus latent scaling. Although this is activation-level rather than token-level, it provides a caution: naive "difference" detectors can hallucinate model-specific features. If the KL predictor finds a surprising token region, activation-level follow-up with crosscoders or patching may help validate it.

### Related Foundations

InstructGPT, FLAN, and T0 define standard SFT/RLHF and instruction-tuning baselines. Token-level DPO provides a preference-learning objective with explicit per-token KL constraints. Proxy tuning and Nudging show that tuned-vs-untuned logit differences can be applied at inference time, often affecting only a sparse set of tokens. Refusal-direction work gives a mechanistic baseline for safety behavior shifts.

## Common Methodologies

- Token distribution comparison: run base and instruct variants on the same context, compare top-k agreement, token rank changes, KL, JS divergence, or log-prob deltas.
- Position-wise analysis: average divergence by generated token index, with special attention to first response tokens and safety prefixes.
- Teacher-forced traces: compute logits along reference completions from Alpaca, BeaverTails, or chat transcripts to avoid sampling confounds.
- Generated-trace analysis: decode from one model and score the same sequence under both models to inspect actual behavioral divergence.
- Prompt controls: compare raw base, base+URIAL/system prompt, and instruct model to isolate prompt-induced style from weight-induced post-training.
- Mechanistic follow-up: use refusal directions, activation patching, or crosscoders for high-surprise spans found by the KL predictor.

## Baselines and Metrics

Recommended baselines:

- Constant positional baseline: average KL by token index and dataset type.
- Token-frequency/style baseline: predict high divergence for common assistant openers, refusal phrases, bullets, hedges, and formatting tokens.
- Shallow model baseline: n-gram or small transformer predictor using local context only.
- Model-family control: compare same-family base/instruct pairs first, such as Gemma base vs Gemma-it or Llama base vs chat.
- Prompt control: base vs base+URIAL, and base+URIAL vs instruct.

Recommended metrics:

- Per-token KL: `KL(p_instruct || p_base)` and optionally reverse KL.
- JS divergence or symmetric KL for stability.
- Top-1/top-k agreement and rank displacement of the observed token.
- Calibration of the divergence predictor: MSE/MAE plus residual outlier analysis.
- Residual enrichment: whether high-error tokens are overrepresented in refusal, style, factual entities, code/math, or prompt-boundary positions.

## Datasets in the Literature

JUST-EVAL-INSTRUCT and MT-Bench are used for instruction-following evaluation and are directly aligned with the URIAL papers. Alpaca is a canonical SFT dataset. BeaverTails provides safety/refusal contexts that should reveal shallow alignment effects. Chatbot Arena conversations provide realistic chat prompts and comparisons. WikiText-2 gives a natural-text baseline where instruction style should be less dominant.

## Gaps and Opportunities

- Existing papers often show aggregate token-shift plots but do not train a predictor and analyze its residual failures.
- Prior work focuses on known shift categories, especially style and refusal. A residual-based approach may surface less obvious categories such as factual entities, tool-use conventions, code formatting, uncertainty phrases, or hidden prompt-boundary artifacts.
- Token-level KL can be dominated by high-probability modes. Tail-aware or top-k-separated divergences may reveal differences that vanilla KL hides.
- Activation-level validation is still fragile; crosscoder artifacts show that "model-specific" findings require controls.

## Recommendations for Our Experiment

Use a staged design:

1. Compute teacher-forced per-token divergence on Alpaca, JUST-EVAL-INSTRUCT generated responses, BeaverTails, Chatbot Arena, and WikiText-2.
2. Train a lightweight divergence predictor using token identity, position, local context embedding, prompt/completion segment, and dataset/source features.
3. Analyze largest positive and negative residuals by token class and context type.
4. Compare same-family base/instruct pairs first to avoid tokenizer/model-family confounds.
5. Add base+URIAL as a control. Residuals unique to instruct weights are more interesting than residuals reproduced by a prompt.
6. Validate surprising safety residuals against refusal-direction or shallow-alignment methods, but avoid relying on unsafe datasets unless access and review protocols are in place.
