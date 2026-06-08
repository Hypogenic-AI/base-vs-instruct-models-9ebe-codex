# Downloaded Datasets

Data files in this directory are excluded from git by `datasets/.gitignore`. The experiment runner can use the local copies directly, and the instructions below make the downloads reproducible.

## Summary

| Dataset | Source | Local location | Splits / size | Task |
|---|---|---|---|---|
| JUST-EVAL-INSTRUCT | `re-align/just-eval-instruct` | `datasets/just_eval_instruct/` | test: 1,000 | Instruction prompt evaluation |
| MT-Bench Prompts | `HuggingFaceH4/mt_bench_prompts` | `datasets/mt_bench_prompts/` | train: 80 | Multi-turn instruction prompts |
| Alpaca 52k | `tatsu-lab/alpaca` | `datasets/alpaca_52k/` | train: 52,002 | Instruction-output SFT data |
| WikiText-2 Raw | `Salesforce/wikitext`, config `wikitext-2-raw-v1` | `datasets/wikitext_2_raw/` | train: 36,718; validation: 3,760; test: 4,358 | Natural text baseline |
| BeaverTails 30k | `PKU-Alignment/BeaverTails` | `datasets/beavertails_30k/` | train: 27,186; test: 3,021 | Safety/refusal prompt-response data |
| Chatbot Arena Conversations mirror | `dim/lmsys_chatbot_arena_conversations` | `datasets/lmsys_chatbot_arena_33k/` | train: 33,000 | Real chat comparisons |

Small samples are under each dataset's `samples/` directory. A generated machine-readable summary is in `datasets/dataset_summary.json`.

## Loading Local Copies

```python
from datasets import load_from_disk

just_eval = load_from_disk("datasets/just_eval_instruct")
mt_bench = load_from_disk("datasets/mt_bench_prompts")
alpaca = load_from_disk("datasets/alpaca_52k")
wikitext = load_from_disk("datasets/wikitext_2_raw")
beavertails = load_from_disk("datasets/beavertails_30k")
arena = load_from_disk("datasets/lmsys_chatbot_arena_33k")
```

## Download Instructions

```python
from datasets import DatasetDict, load_dataset

load_dataset("re-align/just-eval-instruct", split="test").save_to_disk(
    "datasets/just_eval_instruct"
)

load_dataset("HuggingFaceH4/mt_bench_prompts", split="train").save_to_disk(
    "datasets/mt_bench_prompts"
)

load_dataset("tatsu-lab/alpaca", split="train").save_to_disk(
    "datasets/alpaca_52k"
)

load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1").save_to_disk(
    "datasets/wikitext_2_raw"
)

DatasetDict({
    "train": load_dataset("PKU-Alignment/BeaverTails", split="30k_train"),
    "test": load_dataset("PKU-Alignment/BeaverTails", split="30k_test"),
}).save_to_disk("datasets/beavertails_30k")

load_dataset("dim/lmsys_chatbot_arena_conversations", split="train").save_to_disk(
    "datasets/lmsys_chatbot_arena_33k"
)
```

## Notes

- `lmsys/lmsys-chat-1m` was identified as highly relevant, but it is gated on Hugging Face without an available token in this environment. The open `dim/lmsys_chatbot_arena_conversations` mirror was downloaded as a smaller substitute.
- JUST-EVAL-INSTRUCT and MT-Bench are best for prompt-only divergence scans.
- Alpaca and BeaverTails include target responses, which are useful for teacher-forced token-by-token KL along complete answer traces.
- WikiText-2 gives a non-instruction baseline to test whether a divergence predictor overfits to chat/instruction style.
