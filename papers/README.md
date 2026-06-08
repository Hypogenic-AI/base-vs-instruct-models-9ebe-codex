# Downloaded Papers

This directory contains 26 validated PDFs selected for the project "What's surprisingly different between base and instruct models?" All 17 papers with paper-finder relevance >= 2 were downloaded, plus 9 foundational or method papers that are directly useful for experimental design.

Core papers were chunked with the PDF chunker under `papers/pages/` for deeper reading:

- `2312.01552` Unlocking Spell / URIAL
- `2402.05119` Limitations of Instruction Tuning
- `2405.19874` ICL vs instruction fine-tuning
- `2406.05946` Shallow safety alignment
- `2504.02922` Crosscoders for chat-tuning

## Papers

| Title | Year | File | Why relevant |
|---|---:|---|---|
| The Unlocking Spell on Base LLMs: Rethinking Alignment via In-Context Learning | 2023 | `2312.01552_the_unlocking_spell_on_base_llms_rethinking_alignment_via_in_context_learn.pdf` | Directly compares base/aligned token distributions; introduces URIAL and JUST-EVAL-INSTRUCT. |
| A Closer Look at the Limitations of Instruction Tuning | 2024 | `2402.05119_a_closer_look_at_the_limitations_of_instruction_tuning.pdf` | Finds LoRA/full SFT shifts mostly response starts, style tokens, and dataset-borrowed tokens. |
| Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning | 2025 | `2504.02922_overcoming_sparsity_artifacts_in_crosscoders_to_interpret_chat_tuning.pdf` | Model-diffing method for base vs chat representations; BatchTopK reduces false chat-only latents. |
| Is In-Context Learning Sufficient for Instruction Following in LLMs? | 2024 | `2405.19874_is_in_context_learning_sufficient_for_instruction_following_in_llms.pdf` | Systematic URIAL follow-up; compares ICL and instruction fine-tuning with KL analysis. |
| Analyzing the Effects of Supervised Fine-Tuning on Model Knowledge from Token and Parameter Levels | 2025 | `2509.16596_analyzing_the_effects_of_supervised_fine_tuning_on_model_knowledge_from_to.pdf` | Token/parameter analysis of SFT knowledge changes and degradation. |
| Safety Alignment Should Be Made More Than Just a Few Tokens Deep | 2024 | `2406.05946_safety_alignment_should_be_made_more_than_just_a_few_tokens_deep.pdf` | Per-token KL evidence that safety alignment can concentrate in initial output tokens. |
| Transcoder Adapters for Reasoning-Model Diffing | 2026 | `2602.20904_transcoder_adapters_for_reasoning_model_diffing.pdf` | Recent method for interpretable approximations of post-training computation differences. |
| P-Aligner: Enabling Pre-Alignment of Language Models via Principled Instruction Synthesis | 2026 | `2508.04626_p_aligner_enabling_pre_alignment_of_language_models.pdf` | Instruction rewriting/pre-alignment; useful contrast to model-weight or token-distribution changes. |
| PromptKD: Distilling Student-Friendly Knowledge for Generative Language Models via Prompt Tuning | 2024 | `2402.12842_promptkd_distilling_student_friendly_knowledge_for_generative_language_mod.pdf` | Distillation/prompt tuning around token distributions and exposure bias. |
| Shallow Preference Signals: Large Language Model Aligns Even Better with Truncated Data? | 2025 | `2505.17122_shallow_preference_signals_large_language_model_aligns_even_better_with_tr.pdf` | Preference signal concentration in early tokens; supports positional divergence experiments. |
| Rethinking Deep Alignment Through The Lens Of Incomplete Learning | 2025 | `2511.12155_rethinking_deep_alignment_through_the_lens_of_incomplete_learning.pdf` | Uses base-favored tokens as indicators of incomplete safety learning. |
| What makes Reasoning Models Different? Follow the Reasoning Leader for Efficient Decoding | 2025 | `2506.06998_what_makes_reasoning_models_different_follow_the_reasoning_leader_for_effi.pdf` | Token-level misalignment analysis between reasoning and non-reasoning models. |
| Nudging: Inference-time Alignment of LLMs via Guided Decoding | 2024 | `2410.09300_nudging_inference_time_alignment_of_llms_via_guided_decoding.pdf` | Exploits sparse alignment-related token positions for inference-time steering. |
| The Synergy Dilemma of Long-CoT SFT and RL | 2025 | `2507.07562_the_synergy_dilemma_of_long_cot_sft_and_rl_investigating_post_training_tec.pdf` | Post-training comparison for reasoning VLMs; secondary relevance. |
| RL, but don't do anything I wouldn't do | 2024 | `2410.06213_rl_but_don_t_do_anything_i_wouldn_t_do.pdf` | KL-constrained alignment framing; useful for divergence metrics. |
| Emulated Disalignment: Safety Alignment for Large Language Models May Backfire! | 2024 | `2402.12343_emulated_disalignment_safety_alignment_for_large_language_models_may_backf.pdf` | Safety alignment failure mode; useful for surprising divergence regions. |
| Instruction Tuning Changes How Upstream State Conditions Late Readout | 2026 | `2605.07284_instruction_tuning_changes_how_upstream_state_conditions_late_readout_a_cr.pdf` | Cross-patching diagnostic for how instruction tuning changes late readout. |
| Training language models to follow instructions with human feedback | 2022 | `2203.02155_training_language_models_to_follow_instructions_with_human_feedback.pdf` | Foundational RLHF/instruct baseline. |
| Token-level Direct Preference Optimization | 2024 | `2404.11999_token_level_direct_preference_optimization.pdf` | Token-level preference objective with per-token KL constraints. |
| Language Models Resist Alignment: Evidence From Data Compression | 2024 | `2406.06144_language_models_resist_alignment_evidence_from_data_compression.pdf` | Evidence that alignment may not deeply overwrite base behavior. |
| Tuning Language Models by Proxy | 2024 | `2401.08565_tuning_language_models_by_proxy.pdf` | Applies small-model tuned-vs-untuned logit differences to a larger base model. |
| The Flan Collection | 2023 | `2301.13688_the_flan_collection_designing_data_and_methods_for_effective_instruction_t.pdf` | Foundational instruction tuning data/methods. |
| Scaling Instruction-Finetuned Language Models | 2022 | `2210.11416_scaling_instruction_finetuned_language_models.pdf` | Foundational FLAN scaling baseline. |
| Finetuned Language Models Are Zero-Shot Learners | 2021 | `2109.01652_finetuned_language_models_are_zero_shot_learners.pdf` | Early instruction tuning foundation. |
| Refusal in Language Models Is Mediated by a Single Direction | 2024 | `2406.11717_refusal_in_language_models_is_mediated_by_a_single_direction.pdf` | Mechanistic refusal feature baseline for safety-related divergence. |
| From Language Modeling to Instruction Following | 2023 | `2310.00492_from_language_modeling_to_instruction_following_understanding_the_behavior.pdf` | Behavioral shift analysis after instruction tuning. |

Metadata for all downloaded papers is saved in `paper_search_results/selected_papers_metadata.json`.
