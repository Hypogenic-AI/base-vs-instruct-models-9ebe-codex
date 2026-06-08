# Cloned Repositories

All repositories were cloned shallowly into `code/` for inspection and adaptation. Heavy model training/evaluation stacks were not installed.

| Repository | URL | Location | Purpose | Practical notes |
|---|---|---|---|---|
| URIAL | https://github.com/Re-Align/URIAL | `code/urial/` | Tuning-free alignment prompts and inference scripts from Unlocking Spell. | Useful prompts in `urial_prompts/`; inference entry point `src/unified_infer.py`; expects vLLM/GPU for full runs. |
| ICL Alignment | https://github.com/tml-epfl/icl-alignment | `code/icl-alignment/` | Follow-up experiments comparing URIAL/ICL and instruction fine-tuning. | Contains MT-Bench scripts and greedy/many-shot prompt sets under `urial_prompts/`. |
| Just-Eval | https://github.com/Re-Align/just-eval | `code/just-eval/` | Multi-aspect LLM response evaluation used by Re-Align. | Requires OpenAI API for GPT judge modes; local examples show expected JSON format. |
| Shallow vs Deep Alignment | https://github.com/Unispac/shallow-vs-deep-alignment | `code/shallow-vs-deep-alignment/` | Safety alignment experiments with per-token KL and constrained fine-tuning. | Requires gated/unsafe safety data, Llama/Gemma checkpoints, 4 A100/H100-class GPUs, and API-based safety judging. |
| Refusal Direction | https://github.com/andyrdt/refusal_direction | `code/refusal_direction/` | Mechanistic extraction of refusal direction in chat models. | Includes artifacts for several small model families; setup asks for HF and Together API tokens. |
| Token-level DPO | https://github.com/Vance0124/Token-level-Direct-Preference-Optimization | `code/token-level-dpo/` | Reference implementation of token-level DPO with per-token KL terms. | Useful objective implementation in `trainers.py`; training example uses Anthropic HH and multi-GPU FSDP. |
| Crosscode | https://github.com/oli-clive-griffin/crosscode | `code/crosscode/` | Crosscoder/SAE/transcoder training library with TopK/BatchTopK support. | Useful if experiment moves from logit divergence to activation/model diffing. |
| TransformerLens | https://github.com/TransformerLensOrg/TransformerLens | `code/transformer-lens/` | Mechanistic interpretability and activation-caching library. | Useful for activation patching/cache workflows; gated models need `HF_TOKEN`. |

## Suggested Reuse

1. Start with Hugging Face `transformers` for token-level logits/KL between a base and instruct pair with shared tokenizer.
2. Reuse URIAL prompt templates to create "base plus prompt alignment" controls.
3. Use JUST-EVAL and MT-Bench prompt formats for evaluation prompts, but compute divergence without requiring GPT judging.
4. Use TDPO and shallow-alignment code as references for per-token KL computation and positional plots.
5. Use TransformerLens or Crosscode only if the project expands into activation-level diagnostics.
