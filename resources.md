# Resources Catalog

## Summary

Resources gathered for "What's surprisingly different between base and instruct models?"

- Papers downloaded: 26 PDFs
- Datasets downloaded: 6 local dataset directories
- Code repositories cloned: 8 repositories
- Paper search logs: `paper_search_results/`
- PDF chunks for key papers: `papers/pages/`

## Papers

See `papers/README.md` for full descriptions.

| Title | Year | File | Key info |
|---|---:|---|---|
| Unlocking Spell / URIAL | 2023 | `papers/2312.01552_the_unlocking_spell_on_base_llms_rethinking_alignment_via_in_context_learn.pdf` | Direct token distribution shift analysis between base and aligned models. |
| Limitations of Instruction Tuning | 2024 | `papers/2402.05119_a_closer_look_at_the_limitations_of_instruction_tuning.pdf` | Shifted tokens, style/response-initiation tokens, dataset borrowing. |
| Crosscoders for Chat-Tuning | 2025 | `papers/2504.02922_overcoming_sparsity_artifacts_in_crosscoders_to_interpret_chat_tuning.pdf` | Activation-level model diffing and BatchTopK controls. |
| ICL Sufficient for Instruction Following? | 2024 | `papers/2405.19874_is_in_context_learning_sufficient_for_instruction_following_in_llms.pdf` | URIAL/ICL vs IFT and KL comparison. |
| Shallow Safety Alignment | 2024 | `papers/2406.05946_safety_alignment_should_be_made_more_than_just_a_few_tokens_deep.pdf` | Per-token KL concentrated in early safety tokens. |
| Token-level DPO | 2024 | `papers/2404.11999_token_level_direct_preference_optimization.pdf` | Per-token KL preference objective. |
| Refusal Direction | 2024 | `papers/2406.11717_refusal_in_language_models_is_mediated_by_a_single_direction.pdf` | Mechanistic refusal baseline. |
| Foundational instruction tuning/RLHF | 2021-2023 | `papers/2109.01652_*`, `papers/2203.02155_*`, `papers/2210.11416_*`, `papers/2301.13688_*` | Baseline alignment literature. |

The remaining PDFs cover safety/disalignment, proxy tuning, nudging, SFT knowledge effects, reasoning-model differences, and recent cross-patching/transcoder diagnostics.

## Datasets

See `datasets/README.md` for download and loading instructions.

| Name | Source | Size | Task | Location | Notes |
|---|---|---:|---|---|---|
| JUST-EVAL-INSTRUCT | Hugging Face `re-align/just-eval-instruct` | 1,000 examples | Instruction prompt evaluation | `datasets/just_eval_instruct/` | Used by URIAL/Unlocking Spell. |
| MT-Bench Prompts | Hugging Face `HuggingFaceH4/mt_bench_prompts` | 80 prompts | Multi-turn instruction evaluation | `datasets/mt_bench_prompts/` | Good compact prompt suite. |
| Alpaca 52k | Hugging Face `tatsu-lab/alpaca` | 52,002 examples | Instruction-output SFT | `datasets/alpaca_52k/` | Good teacher-forced response traces. |
| WikiText-2 Raw | Hugging Face `Salesforce/wikitext` | 44,836 rows across splits | Natural text | `datasets/wikitext_2_raw/` | Non-instruction baseline. |
| BeaverTails 30k | Hugging Face `PKU-Alignment/BeaverTails` | 30,207 examples | Safety/refusal data | `datasets/beavertails_30k/` | Useful for shallow safety alignment. |
| Chatbot Arena mirror | Hugging Face `dim/lmsys_chatbot_arena_conversations` | 33,000 rows | Real chat comparisons | `datasets/lmsys_chatbot_arena_33k/` | Substitute for gated LMSYS-Chat-1M. |

## Code Repositories

See `code/README.md` for details.

| Name | URL | Purpose | Location | Notes |
|---|---|---|---|---|
| URIAL | https://github.com/Re-Align/URIAL | Prompt-based base-model alignment | `code/urial/` | Reuse prompt templates and inference format. |
| ICL Alignment | https://github.com/tml-epfl/icl-alignment | URIAL follow-up experiments | `code/icl-alignment/` | Many-shot/greedy prompt controls. |
| Just-Eval | https://github.com/Re-Align/just-eval | Multi-aspect evaluation | `code/just-eval/` | Optional GPT-judge evaluation. |
| Shallow vs Deep Alignment | https://github.com/Unispac/shallow-vs-deep-alignment | Per-token safety KL experiments | `code/shallow-vs-deep-alignment/` | Heavy GPU/gated safety data. |
| Refusal Direction | https://github.com/andyrdt/refusal_direction | Refusal feature extraction | `code/refusal_direction/` | Includes artifacts for smaller model families. |
| Token-level DPO | https://github.com/Vance0124/Token-level-Direct-Preference-Optimization | Per-token KL preference training | `code/token-level-dpo/` | Objective reference. |
| Crosscode | https://github.com/oli-clive-griffin/crosscode | Crosscoder / BatchTopK model diffing | `code/crosscode/` | Activation-level extension path. |
| TransformerLens | https://github.com/TransformerLensOrg/TransformerLens | Activation cache and patching tools | `code/transformer-lens/` | Useful for mechanistic follow-up. |

## Search Strategy

The primary paper-finder query targeted base-vs-instruct model divergence, token-level KL, and behavioral differences. Manual follow-up searched arXiv, Semantic Scholar, Hugging Face datasets, Papers with Code, and GitHub for code/data attached to the most relevant papers.

Selection prioritized:

- Direct base vs instruct/chat comparisons.
- Token-level or per-position KL/logit analysis.
- Methods for alignment without full fine-tuning.
- Safety/refusal divergence and shallow alignment.
- Reproducible datasets and code.

## Challenges and Workarounds

- `lmsys/lmsys-chat-1m` is gated. Workaround: downloaded `dim/lmsys_chatbot_arena_conversations`.
- Several safety resources require gated models, unsafe-content agreements, or API-based judges. Workaround: downloaded BeaverTails 30k and documented heavier repos as optional follow-up.
- Some recent 2026 papers have sparse citation/code metadata. Workaround: downloaded open arXiv PDFs and treated them as secondary references.
- Full crosscoder reproduction is expensive. Workaround: cloned Crosscode/TransformerLens for future mechanistic validation, but recommended starting with logit/KL experiments.

## Recommendations for Experiment Design

Primary datasets:

- Start with Alpaca 52k, JUST-EVAL-INSTRUCT, MT-Bench, BeaverTails, Chatbot Arena mirror, and WikiText-2.

Primary model pairs:

- Same-tokenizer same-family base/instruct pairs where possible: Gemma base vs Gemma-it, Llama base vs chat/instruct, Mistral base vs instruct.

Baseline methods:

- Position-only KL predictor.
- Token/style-class predictor.
- Lightweight contextual predictor.
- Base+URIAL prompt control.

Evaluation:

- Per-token KL/JS divergence.
- Top-k agreement and rank displacement.
- Residual outlier categories by token position, dataset, and semantic class.
- Manual review of high-error spans, with special labels for assistant openers, refusal phrases, factual entities, code/math, lists, and prompt boundaries.

Code to adapt:

- Use URIAL prompts for prompt controls.
- Use TDPO and shallow-alignment code as per-token KL references.
- Use TransformerLens or Crosscode only after residual analysis identifies promising mechanistic targets.

## Execution Notes from This Research Session

The completed experiment used the local datasets listed above and a same-family open model pair:

- Base model: `Qwen/Qwen2.5-1.5B`
- Instruct model: `Qwen/Qwen2.5-1.5B-Instruct`
- Dataset sample: 60 examples each from Alpaca, JUST-EVAL-INSTRUCT, BeaverTails, Chatbot Arena mirror, and WikiText-2, with one WikiText row dropped after token filtering.
- Final scale: 299 examples and 24,049 token-level KL rows.
- Main script: `src/run_divergence_experiment.py`
- Primary report: `REPORT.md`

Key generated outputs:

- `results/token_kl_metrics.csv.gz`: full token-level divergence table.
- `results/heldout_predictions.csv`: held-out KL predictor predictions and residuals.
- `results/top_abs_residuals.csv`: largest rich-predictor misses.
- `results/residual_enrichment.csv`: residual category enrichment.
- `figures/kl_by_source.png`, `figures/kl_position_trend.png`, `figures/predictor_scatter.png`, `figures/residual_token_class_enrichment.png`.

The rich predictor improved held-out MAE on `log1p(KL)` from 0.0601 for the position/source baseline to 0.0460. The largest positive residuals were unexpectedly concentrated in WikiText narrative spans, while negative residuals often appeared at explicit `Assistant:` boundaries where both models agreed more than predicted.
