"""Token-level KL divergence experiment for base vs instruct model pairs.

This script:
1. Loads real base and instruct causal LMs with a shared tokenizer.
2. Computes teacher-forced token-level distribution divergences.
3. Trains baselines and a richer KL predictor on held-out examples.
4. Saves residual analyses and figures for reporting.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, load_from_disk
from scipy import stats
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OrdinalEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer


WORKSPACE = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    seed: int = 42
    base_model: str = "Qwen/Qwen2.5-1.5B"
    instruct_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    examples_per_source: int = 60
    max_length: int = 224
    batch_size: int = 8
    device: str = "cuda:0"
    output_dir: str = "results"
    figure_dir: str = "figures"
    log_dir: str = "logs"
    top_residual_count: int = 150
    bootstrap_iterations: int = 500


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs(cfg: Config) -> None:
    for dirname in [cfg.output_dir, cfg.figure_dir, cfg.log_dir]:
        (WORKSPACE / dirname).mkdir(parents=True, exist_ok=True)


def load_any(path: str) -> Dataset | DatasetDict:
    return load_from_disk(str(WORKSPACE / path))


def dataset_part(ds: Dataset | DatasetDict, preferred: str = "train") -> Dataset:
    if isinstance(ds, DatasetDict):
        if preferred in ds:
            return ds[preferred]
        return ds[next(iter(ds.keys()))]
    return ds


def compact(text: Any, limit: int = 1600) -> str:
    if text is None:
        return ""
    value = str(text)
    value = re.sub(r"\s+", " ", value).strip()
    return value[:limit]


def pick_evenly(items: list[Any], n: int, seed: int) -> list[Any]:
    if len(items) <= n:
        return items
    rng = random.Random(seed)
    indices = list(range(len(items)))
    rng.shuffle(indices)
    return [items[i] for i in sorted(indices[:n])]


def build_examples(cfg: Config) -> list[dict[str, Any]]:
    """Load local datasets and normalize them into text traces."""

    examples: list[dict[str, Any]] = []

    def add_example(
        source: str,
        local_id: str,
        prompt: str,
        response: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        prompt = compact(prompt)
        response = compact(response)
        if not prompt:
            return
        if response:
            text = f"{prompt}{response}"
            prompt_end = len(prompt)
        else:
            text = prompt
            prompt_end = len(prompt)
        examples.append(
            {
                "example_id": f"{source}:{local_id}",
                "source": source,
                "text": text,
                "prompt_end": prompt_end,
                "has_response": bool(response),
                "metadata": metadata or {},
            }
        )

    # Alpaca instruction-output traces.
    alpaca = dataset_part(load_any("datasets/alpaca_52k"))
    rows = pick_evenly(list(range(len(alpaca))), cfg.examples_per_source, cfg.seed + 1)
    for idx in rows:
        ex = alpaca[int(idx)]
        instruction = compact(ex.get("instruction", ""))
        inp = compact(ex.get("input", ""))
        prompt = "### Instruction:\n" + instruction + "\n"
        if inp:
            prompt += "\n### Input:\n" + inp + "\n"
        prompt += "\n### Response:\n"
        add_example(
            "alpaca",
            str(idx),
            prompt,
            compact(ex.get("output", "")),
            {"task": "instruction_sft"},
        )

    # JUST-EVAL prompts usually lack reference output, so score the generation boundary.
    just_eval = dataset_part(load_any("datasets/just_eval_instruct"), preferred="test")
    rows = pick_evenly(list(range(len(just_eval))), cfg.examples_per_source, cfg.seed + 2)
    for idx in rows:
        ex = just_eval[int(idx)]
        prompt = "### Instruction:\n" + compact(ex.get("instruction", "")) + "\n\n### Response:\n"
        add_example(
            "just_eval",
            str(ex.get("id", idx)),
            prompt,
            "",
            {
                "category": str(ex.get("category", "")),
                "difficulty": str(ex.get("difficulty", "")),
                "task": ",".join(ex.get("task", []) or []),
            },
        )

    # BeaverTails safety traces.
    beaver = dataset_part(load_any("datasets/beavertails_30k"))
    safe_indices = [i for i, row in enumerate(beaver) if bool(row.get("is_safe", False))]
    unsafe_indices = [i for i, row in enumerate(beaver) if not bool(row.get("is_safe", False))]
    half = max(1, cfg.examples_per_source // 2)
    rows = pick_evenly(safe_indices, half, cfg.seed + 3) + pick_evenly(
        unsafe_indices, cfg.examples_per_source - half, cfg.seed + 4
    )
    for idx in rows:
        ex = beaver[int(idx)]
        prompt = "User:\n" + compact(ex.get("prompt", "")) + "\n\nAssistant:\n"
        add_example(
            "beavertails",
            str(idx),
            prompt,
            compact(ex.get("response", "")),
            {
                "is_safe": bool(ex.get("is_safe", False)),
                "category": ",".join([k for k, v in (ex.get("category") or {}).items() if v]),
            },
        )

    # WikiText natural-language baseline. Use non-empty rows only.
    wiki = dataset_part(load_any("datasets/wikitext_2_raw"), preferred="test")
    non_empty = [i for i, row in enumerate(wiki) if compact(row.get("text", ""))]
    rows = pick_evenly(non_empty, cfg.examples_per_source, cfg.seed + 5)
    for idx in rows:
        text = compact(wiki[int(idx)].get("text", ""))
        add_example("wikitext", str(idx), text, "", {"task": "natural_text"})

    # Chatbot Arena: use first user/assistant pair from one conversation.
    arena = dataset_part(load_any("datasets/lmsys_chatbot_arena_33k"))
    valid_arena: list[int] = []
    for i, row in enumerate(arena):
        conv = row.get("conversation_b") or row.get("conversation_a") or []
        if len(conv) >= 2 and conv[0].get("role") == "user" and conv[1].get("role") == "assistant":
            if compact(conv[0].get("content", "")) and compact(conv[1].get("content", "")):
                valid_arena.append(i)
        if len(valid_arena) >= cfg.examples_per_source * 10:
            break
    rows = pick_evenly(valid_arena, cfg.examples_per_source, cfg.seed + 6)
    for idx in rows:
        ex = arena[int(idx)]
        conv = ex.get("conversation_b") or ex.get("conversation_a")
        prompt = "User:\n" + compact(conv[0].get("content", "")) + "\n\nAssistant:\n"
        add_example(
            "arena",
            str(idx),
            prompt,
            compact(conv[1].get("content", "")),
            {
                "model_a": str(ex.get("model_a", "")),
                "model_b": str(ex.get("model_b", "")),
                "winner": str(ex.get("winner", "")),
            },
        )

    return examples


def token_class(token: str) -> str:
    stripped = token.strip()
    if token == "":
        return "missing"
    if "\n" in token:
        return "newline"
    if stripped == "":
        return "space"
    low = stripped.lower()
    refusal_words = {
        "sorry",
        "apologize",
        "cannot",
        "can't",
        "unable",
        "assist",
        "help",
        "harmful",
        "safe",
        "illegal",
        "policy",
    }
    assistant_words = {"assistant", "user", "response", "instruction", "answer"}
    discourse_words = {"however", "therefore", "because", "first", "second", "finally", "also"}
    if low in refusal_words:
        return "refusal_style"
    if low in assistant_words:
        return "assistant_marker"
    if low in discourse_words:
        return "discourse"
    if any(ch in token for ch in "`{}[]<>="):
        return "code_math"
    if re.fullmatch(r"[\W_]+", stripped):
        return "punctuation"
    if any(ch.isdigit() for ch in stripped):
        return "digit_or_entity"
    if stripped[:1].isupper() and len(stripped) > 1:
        return "capitalized"
    if stripped.isalpha():
        return "word"
    return "mixed"


def safe_token_text(token: str) -> str:
    display = token.replace("\n", "\\n").replace("\t", "\\t")
    if len(display) > 80:
        display = display[:77] + "..."
    return display


def sanitize_snippet(row: pd.Series | dict[str, Any], snippet: str) -> str:
    source = row.get("source", "")
    is_safe = row.get("is_safe", True)
    snippet = re.sub(r"\s+", " ", snippet).strip()
    if source == "beavertails" and not bool(is_safe):
        return "[unsafe BeaverTails context redacted]"
    return snippet[:260]


def load_model(model_name: str, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    kwargs = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": False,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, **kwargs)
    model.to(device)
    model.eval()
    return model


def compute_batch_metrics(
    base_model: torch.nn.Module,
    instruct_model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    next_token_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute divergences for every context position in a padded batch."""

    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=input_ids.is_cuda):
            base_logits = base_model(input_ids=input_ids, attention_mask=attention_mask).logits
            instruct_logits = instruct_model(input_ids=input_ids, attention_mask=attention_mask).logits

        base_logits = base_logits.float()
        instruct_logits = instruct_logits.float()

        logp_base = F.log_softmax(base_logits, dim=-1)
        logp_inst = F.log_softmax(instruct_logits, dim=-1)
        p_base = torch.exp(logp_base)
        p_inst = torch.exp(logp_inst)

        fwd_kl = (p_inst * (logp_inst - logp_base)).sum(dim=-1)
        rev_kl = (p_base * (logp_base - logp_inst)).sum(dim=-1)
        log_mix = torch.logaddexp(logp_inst, logp_base) - math.log(2.0)
        js = 0.5 * (p_inst * (logp_inst - log_mix)).sum(dim=-1)
        js = js + 0.5 * (p_base * (logp_base - log_mix)).sum(dim=-1)

        entropy_base = -(p_base * logp_base).sum(dim=-1)
        base_top_prob, base_top1 = p_base.max(dim=-1)
        inst_top1 = instruct_logits.argmax(dim=-1)
        top1_agree = base_top1.eq(inst_top1)
        base_sorted_top2 = torch.topk(p_base, k=2, dim=-1).values
        base_margin = base_sorted_top2[..., 0] - base_sorted_top2[..., 1]

        safe_next = next_token_ids.clamp(min=0)
        observed_mask = next_token_ids.ge(0)
        obs_logprob = torch.gather(logp_base, dim=-1, index=safe_next.unsqueeze(-1)).squeeze(-1)
        obs_logit = torch.gather(base_logits, dim=-1, index=safe_next.unsqueeze(-1)).squeeze(-1)
        obs_rank = (base_logits > obs_logit.unsqueeze(-1)).sum(dim=-1) + 1
        obs_logprob = obs_logprob.masked_fill(~observed_mask, float("nan"))
        obs_rank = obs_rank.masked_fill(~observed_mask, 0)

    return {
        "fwd_kl": fwd_kl.cpu(),
        "rev_kl": rev_kl.cpu(),
        "js": js.cpu(),
        "entropy_base": entropy_base.cpu(),
        "base_top_prob": base_top_prob.cpu(),
        "base_margin": base_margin.cpu(),
        "base_top1": base_top1.cpu(),
        "inst_top1": inst_top1.cpu(),
        "top1_agree": top1_agree.cpu(),
        "obs_logprob": obs_logprob.cpu(),
        "obs_rank": obs_rank.cpu(),
    }


def encode_examples(
    tokenizer: Any,
    examples: list[dict[str, Any]],
    max_length: int,
) -> list[dict[str, Any]]:
    encoded: list[dict[str, Any]] = []
    for ex in examples:
        enc = tokenizer(
            ex["text"],
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        if len(input_ids) < 4:
            continue
        prompt_end = min(ex["prompt_end"], len(ex["text"]))
        prompt_token_count = 0
        for start, end in offsets:
            if start < prompt_end:
                prompt_token_count += 1
        encoded.append(
            {
                **ex,
                "input_ids": input_ids,
                "offsets": offsets,
                "prompt_token_count": prompt_token_count,
                "truncated": len(input_ids) >= max_length,
            }
        )
    return encoded


def make_rows_for_batch(
    tokenizer: Any,
    batch_examples: list[dict[str, Any]],
    metrics: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bi, ex in enumerate(batch_examples):
        ids = ex["input_ids"]
        offsets = ex["offsets"]
        length = len(ids)
        # Rows correspond to the distribution after context token c, usually predicting token c+1.
        context_positions = list(range(length - 1))
        if not ex["has_response"]:
            # Also include the generation-start distribution after a prompt-only trace.
            context_positions.append(length - 1)

        for cpos in context_positions:
            has_next = cpos + 1 < length
            next_idx = cpos + 1 if has_next else -1
            next_id = ids[next_idx] if has_next else -1
            token_text = tokenizer.decode([next_id]) if has_next else ""
            prompt_token_count = ex["prompt_token_count"]

            if has_next:
                start_char = offsets[next_idx][0]
                segment = "response" if start_char >= ex["prompt_end"] else "prompt"
                response_token_index = next_idx - prompt_token_count
            else:
                start_char = offsets[cpos][1]
                segment = "generation_start"
                response_token_index = 0

            is_generation_start = int(
                segment == "generation_start" or (segment == "response" and response_token_index == 0)
            )
            prev_start = max(0, offsets[cpos][1] - 100)
            next_end = min(len(ex["text"]), (offsets[next_idx][1] if has_next else offsets[cpos][1]) + 100)
            context_snippet = ex["text"][prev_start:next_end]

            obs_logprob = np.nan
            obs_rank = np.nan
            if has_next:
                obs_logprob = float(metrics["obs_logprob"][bi, cpos])
                obs_rank = int(metrics["obs_rank"][bi, cpos])

            metadata = ex.get("metadata", {})
            base_top1_id = int(metrics["base_top1"][bi, cpos])
            inst_top1_id = int(metrics["inst_top1"][bi, cpos])
            rows.append(
                {
                    "example_id": ex["example_id"],
                    "source": ex["source"],
                    "segment": segment,
                    "context_position": cpos,
                    "next_token_index": next_idx,
                    "response_token_index": response_token_index,
                    "norm_position": cpos / max(1, length - 1),
                    "is_generation_start": is_generation_start,
                    "has_observed_token": int(has_next),
                    "token_id": int(next_id),
                    "token_text": safe_token_text(token_text),
                    "token_class": token_class(token_text),
                    "token_len": len(token_text),
                    "token_has_space": int(token_text.startswith(" ")),
                    "token_has_newline": int("\n" in token_text),
                    "token_has_digit": int(any(ch.isdigit() for ch in token_text)),
                    "token_is_capitalized": int(token_text.strip()[:1].isupper()),
                    "fwd_kl": float(metrics["fwd_kl"][bi, cpos]),
                    "rev_kl": float(metrics["rev_kl"][bi, cpos]),
                    "js": float(metrics["js"][bi, cpos]),
                    "base_entropy": float(metrics["entropy_base"][bi, cpos]),
                    "base_top_prob": float(metrics["base_top_prob"][bi, cpos]),
                    "base_margin": float(metrics["base_margin"][bi, cpos]),
                    "base_logprob_observed": obs_logprob,
                    "base_rank_observed": obs_rank,
                    "base_top1_token_id": base_top1_id,
                    "inst_top1_token_id": inst_top1_id,
                    "base_top1_text": safe_token_text(tokenizer.decode([base_top1_id])),
                    "inst_top1_text": safe_token_text(tokenizer.decode([inst_top1_id])),
                    "top1_agree": int(metrics["top1_agree"][bi, cpos]),
                    "is_safe": metadata.get("is_safe", True),
                    "metadata_category": metadata.get("category", ""),
                    "truncated": int(ex.get("truncated", False)),
                    "context_snippet": context_snippet,
                }
            )
    return rows


def score_kl(cfg: Config, tokenizer: Any, encoded: list[dict[str, Any]]) -> pd.DataFrame:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"Loading models on {device} with dtype={dtype}...", flush=True)
    base_model = load_model(cfg.base_model, device, dtype)
    instruct_model = load_model(cfg.instruct_model, device, dtype)

    all_rows: list[dict[str, Any]] = []
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    start_time = time.time()
    for start in range(0, len(encoded), cfg.batch_size):
        batch_examples = encoded[start : start + cfg.batch_size]
        max_len = max(len(ex["input_ids"]) for ex in batch_examples)
        ids = torch.full((len(batch_examples), max_len), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch_examples), max_len), dtype=torch.long)
        next_ids = torch.full((len(batch_examples), max_len), -1, dtype=torch.long)
        for bi, ex in enumerate(batch_examples):
            seq = torch.tensor(ex["input_ids"], dtype=torch.long)
            ids[bi, : len(seq)] = seq
            mask[bi, : len(seq)] = 1
            if len(seq) > 1:
                next_ids[bi, : len(seq) - 1] = seq[1:]
        ids = ids.to(device)
        mask = mask.to(device)
        next_ids = next_ids.to(device)
        metrics = compute_batch_metrics(base_model, instruct_model, ids, mask, next_ids)
        all_rows.extend(make_rows_for_batch(tokenizer, batch_examples, metrics))
        del ids, mask, next_ids, metrics
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        done = min(start + cfg.batch_size, len(encoded))
        elapsed = time.time() - start_time
        print(f"Scored {done}/{len(encoded)} examples in {elapsed:.1f}s", flush=True)

    del base_model, instruct_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    df = pd.DataFrame(all_rows)
    df["context_snippet"] = df.apply(lambda r: sanitize_snippet(r, r["context_snippet"]), axis=1)
    df["log1p_fwd_kl"] = np.log1p(df["fwd_kl"].clip(lower=0))
    df["log_base_rank_observed"] = np.log1p(df["base_rank_observed"].fillna(df["base_rank_observed"].max()))
    return df


def source_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, part in df.groupby("source"):
        rows.append(
            {
                "source": source,
                "tokens": len(part),
                "examples": part["example_id"].nunique(),
                "mean_fwd_kl": part["fwd_kl"].mean(),
                "median_fwd_kl": part["fwd_kl"].median(),
                "p90_fwd_kl": part["fwd_kl"].quantile(0.90),
                "p99_fwd_kl": part["fwd_kl"].quantile(0.99),
                "mean_js": part["js"].mean(),
                "top1_disagree_rate": 1.0 - part["top1_agree"].mean(),
                "mean_base_entropy": part["base_entropy"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_fwd_kl", ascending=False)


def encode_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, OrdinalEncoder, list[int]]:
    train_num = train[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    test_num = test[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    train_cat = train[categorical_cols].fillna("__missing__").astype(str)
    test_cat = test[categorical_cols].fillna("__missing__").astype(str)
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
        min_frequency=5,
    )
    train_cat_arr = encoder.fit_transform(train_cat)
    test_cat_arr = encoder.transform(test_cat)
    # HistGradientBoosting expects non-negative categorical codes. Unknown -1 becomes 0.
    train_cat_arr = train_cat_arr + 1
    test_cat_arr = test_cat_arr + 1

    X_train = np.hstack([train_num.to_numpy(), train_cat_arr])
    X_test = np.hstack([test_num.to_numpy(), test_cat_arr])
    categorical_indices = list(range(len(numeric_cols), len(numeric_cols) + len(categorical_cols)))
    return X_train, X_test, encoder, categorical_indices


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    spearman = stats.spearmanr(y_true, y_pred, nan_policy="omit")
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman_r": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def train_predictors(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=cfg.seed)
    idx_train, idx_test = next(splitter.split(df, groups=df["example_id"]))
    train = df.iloc[idx_train].copy()
    test = df.iloc[idx_test].copy()
    y_train = train["log1p_fwd_kl"].to_numpy()
    y_test = test["log1p_fwd_kl"].to_numpy()

    numeric_base = [
        "context_position",
        "norm_position",
        "response_token_index",
        "is_generation_start",
    ]
    categorical_base = ["source", "segment"]

    numeric_rich = numeric_base + [
        "token_len",
        "token_has_space",
        "token_has_newline",
        "token_has_digit",
        "token_is_capitalized",
        "has_observed_token",
        "base_entropy",
        "base_top_prob",
        "base_margin",
        "base_logprob_observed",
        "log_base_rank_observed",
    ]
    # Keep token identity only for moderately frequent tokens to reduce overfitting.
    token_counts = Counter(train["token_text"].astype(str))
    top_tokens = {tok for tok, _ in token_counts.most_common(120)}
    category_counts = Counter(train["metadata_category"].fillna("").astype(str))
    top_categories = {cat for cat, _ in category_counts.most_common(120)}
    for frame in [train, test]:
        frame["token_text_frequent"] = [
            tok if str(tok) in top_tokens and token_counts.get(str(tok), 0) >= 8 else "__rare__"
            for tok in frame["token_text"].astype(str)
        ]
        frame["metadata_category_frequent"] = [
            cat if str(cat) in top_categories else "__rare__"
            for cat in frame["metadata_category"].fillna("").astype(str)
        ]
        frame["safety_label"] = frame["is_safe"].astype(str)

    categorical_rich = [
        "source",
        "segment",
        "token_class",
        "token_text_frequent",
        "safety_label",
        "metadata_category_frequent",
    ]

    results: dict[str, Any] = {
        "split": {
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_examples": int(train["example_id"].nunique()),
            "test_examples": int(test["example_id"].nunique()),
        },
        "models": {},
    }

    predictions = test[
        [
            "example_id",
            "source",
            "segment",
            "context_position",
            "response_token_index",
            "token_text",
            "token_class",
            "fwd_kl",
            "log1p_fwd_kl",
            "js",
            "top1_agree",
            "base_top1_text",
            "inst_top1_text",
            "base_entropy",
            "base_logprob_observed",
            "base_rank_observed",
            "is_safe",
            "metadata_category",
            "context_snippet",
        ]
    ].copy()

    dummy = DummyRegressor(strategy="mean")
    dummy.fit(np.zeros((len(train), 1)), y_train)
    pred_dummy = dummy.predict(np.zeros((len(test), 1)))
    predictions["pred_mean"] = pred_dummy
    results["models"]["mean_baseline"] = evaluate_predictions(y_test, pred_dummy)

    X_train_base, X_test_base, _, cat_idx_base = encode_features(
        train, test, numeric_base, categorical_base
    )
    pos_model = HistGradientBoostingRegressor(
        max_iter=180,
        learning_rate=0.06,
        max_leaf_nodes=15,
        l2_regularization=0.05,
        categorical_features=cat_idx_base,
        random_state=cfg.seed,
    )
    pos_model.fit(X_train_base, y_train)
    pred_pos = pos_model.predict(X_test_base)
    predictions["pred_position_source"] = pred_pos
    results["models"]["position_source"] = evaluate_predictions(y_test, pred_pos)

    X_train_rich, X_test_rich, _, cat_idx_rich = encode_features(
        train, test, numeric_rich, categorical_rich
    )
    rich_model = HistGradientBoostingRegressor(
        max_iter=280,
        learning_rate=0.045,
        max_leaf_nodes=31,
        min_samples_leaf=25,
        l2_regularization=0.08,
        categorical_features=cat_idx_rich,
        random_state=cfg.seed,
    )
    rich_model.fit(X_train_rich, y_train)
    pred_rich = rich_model.predict(X_test_rich)
    predictions["pred_rich"] = pred_rich
    results["models"]["rich_predictor"] = evaluate_predictions(y_test, pred_rich)

    for col in ["pred_mean", "pred_position_source", "pred_rich"]:
        predictions[f"residual_{col}"] = predictions["log1p_fwd_kl"] - predictions[col]
        predictions[f"abs_residual_{col}"] = predictions[f"residual_{col}"].abs()

    results["bootstrap"] = bootstrap_predictor_comparison(predictions, cfg)
    results["feature_sets"] = {
        "position_source_numeric": numeric_base,
        "position_source_categorical": categorical_base,
        "rich_numeric": numeric_rich,
        "rich_categorical": categorical_rich,
    }
    return predictions, results


def bootstrap_predictor_comparison(predictions: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    rng = np.random.default_rng(cfg.seed)
    examples = predictions["example_id"].unique()
    records = []
    for _ in range(cfg.bootstrap_iterations):
        sampled = rng.choice(examples, size=len(examples), replace=True)
        parts = [predictions[predictions["example_id"] == ex] for ex in sampled]
        boot = pd.concat(parts, ignore_index=True)
        y = boot["log1p_fwd_kl"].to_numpy()
        row = {}
        for name, col in [
            ("mean", "pred_mean"),
            ("position_source", "pred_position_source"),
            ("rich", "pred_rich"),
        ]:
            row[f"{name}_mae"] = mean_absolute_error(y, boot[col].to_numpy())
        row["rich_minus_position_mae"] = row["rich_mae"] - row["position_source_mae"]
        row["rich_minus_mean_mae"] = row["rich_mae"] - row["mean_mae"]
        records.append(row)
    boot_df = pd.DataFrame(records)
    summary: dict[str, Any] = {}
    for col in boot_df.columns:
        vals = boot_df[col].to_numpy()
        summary[col] = {
            "mean": float(np.mean(vals)),
            "ci_low": float(np.quantile(vals, 0.025)),
            "ci_high": float(np.quantile(vals, 0.975)),
        }
    (WORKSPACE / cfg.output_dir / "bootstrap_predictor_metrics.csv").write_text(
        boot_df.to_csv(index=False)
    )
    return summary


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    # Efficient enough for source-level example means.
    x = np.asarray(x)
    y = np.asarray(y)
    greater = 0
    less = 0
    for value in x:
        greater += int(np.sum(value > y))
        less += int(np.sum(value < y))
    return float((greater - less) / max(1, len(x) * len(y)))


def source_tests(df: pd.DataFrame) -> pd.DataFrame:
    ex_means = df.groupby(["example_id", "source"], as_index=False)["fwd_kl"].mean()
    sources = sorted(ex_means["source"].unique())
    rows = []
    for i, a in enumerate(sources):
        for b in sources[i + 1 :]:
            xa = ex_means.loc[ex_means["source"] == a, "fwd_kl"].to_numpy()
            xb = ex_means.loc[ex_means["source"] == b, "fwd_kl"].to_numpy()
            test = stats.mannwhitneyu(xa, xb, alternative="two-sided")
            rows.append(
                {
                    "source_a": a,
                    "source_b": b,
                    "n_a": len(xa),
                    "n_b": len(xb),
                    "mean_a": float(np.mean(xa)),
                    "mean_b": float(np.mean(xb)),
                    "median_a": float(np.median(xa)),
                    "median_b": float(np.median(xb)),
                    "mann_whitney_u": float(test.statistic),
                    "p_raw": float(test.pvalue),
                    "cliffs_delta_a_vs_b": cliffs_delta(xa, xb),
                }
            )
    tests = pd.DataFrame(rows).sort_values("p_raw").reset_index(drop=True)
    if len(tests):
        m = len(tests)
        adjusted = np.empty(m)
        prev = 0.0
        for rank, idx in enumerate(tests.index, start=1):
            value = min(1.0, (m - rank + 1) * tests.loc[idx, "p_raw"])
            prev = max(prev, value)
            adjusted[idx] = prev
        tests["p_holm"] = adjusted
    return tests


def residual_analysis(predictions: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    pred = predictions.copy()
    residual_col = "residual_pred_rich"
    abs_col = "abs_residual_pred_rich"
    pred["residual_direction"] = np.where(pred[residual_col] >= 0, "underpredicted", "overpredicted")
    top_abs = pred.nlargest(cfg.top_residual_count, abs_col).copy()
    top_pos = pred.nlargest(cfg.top_residual_count // 2, residual_col).copy()
    top_neg = pred.nsmallest(cfg.top_residual_count // 2, residual_col).copy()

    out_cols = [
        "example_id",
        "source",
        "segment",
        "context_position",
        "response_token_index",
        "token_text",
        "token_class",
        "fwd_kl",
        "log1p_fwd_kl",
        "pred_rich",
        residual_col,
        abs_col,
        "js",
        "top1_agree",
        "base_top1_text",
        "inst_top1_text",
        "base_entropy",
        "base_logprob_observed",
        "base_rank_observed",
        "is_safe",
        "metadata_category",
        "context_snippet",
    ]
    top_abs[out_cols].to_csv(WORKSPACE / cfg.output_dir / "top_abs_residuals.csv", index=False)
    top_pos[out_cols].to_csv(WORKSPACE / cfg.output_dir / "top_positive_residuals.csv", index=False)
    top_neg[out_cols].to_csv(WORKSPACE / cfg.output_dir / "top_negative_residuals.csv", index=False)

    threshold = pred[abs_col].quantile(0.98)
    pred["high_abs_residual"] = pred[abs_col] >= threshold
    enrichment_rows = []
    for col in ["source", "segment", "token_class"]:
        bg = pred[col].value_counts(normalize=True)
        hi = pred.loc[pred["high_abs_residual"], col].value_counts(normalize=True)
        for key in sorted(set(bg.index).union(hi.index)):
            enrichment_rows.append(
                {
                    "feature": col,
                    "value": str(key),
                    "background_frac": float(bg.get(key, 0.0)),
                    "high_residual_frac": float(hi.get(key, 0.0)),
                    "enrichment": float((hi.get(key, 0.0) + 1e-9) / (bg.get(key, 0.0) + 1e-9)),
                }
            )
    enrichment = pd.DataFrame(enrichment_rows).sort_values("enrichment", ascending=False)
    enrichment.to_csv(WORKSPACE / cfg.output_dir / "residual_enrichment.csv", index=False)

    summary = {
        "residual_threshold_abs_98pct": float(threshold),
        "mean_abs_residual": float(pred[abs_col].mean()),
        "median_abs_residual": float(pred[abs_col].median()),
        "top_abs_by_source": top_abs["source"].value_counts().to_dict(),
        "top_abs_by_token_class": top_abs["token_class"].value_counts().to_dict(),
        "top_positive_by_source": top_pos["source"].value_counts().to_dict(),
        "top_negative_by_source": top_neg["source"].value_counts().to_dict(),
    }
    return summary


def make_figures(df: pd.DataFrame, predictions: pd.DataFrame, cfg: Config) -> None:
    sns.set_theme(style="whitegrid")
    fig_dir = WORKSPACE / cfg.figure_dir

    plt.figure(figsize=(10, 5))
    order = df.groupby("source")["fwd_kl"].median().sort_values(ascending=False).index
    plot_df = df.copy()
    plot_df["fwd_kl_plot"] = plot_df["fwd_kl"] + 1e-8
    sns.boxplot(data=plot_df, x="source", y="fwd_kl_plot", order=order, showfliers=False)
    plt.yscale("log")
    plt.xlabel("Source")
    plt.ylabel("Forward KL: instruct || base (log scale)")
    plt.title("Token-Level Divergence by Source")
    plt.tight_layout()
    plt.savefig(fig_dir / "kl_by_source.png", dpi=180)
    plt.close()

    pos_df = df.copy()
    pos_df["position_bin"] = pd.cut(
        pos_df["norm_position"], bins=np.linspace(0, 1, 11), include_lowest=True
    )
    trend = (
        pos_df.groupby(["source", "position_bin"], observed=True)["fwd_kl"]
        .median()
        .reset_index()
    )
    trend["fwd_kl"] = trend["fwd_kl"] + 1e-8
    trend["position_mid"] = trend["position_bin"].apply(lambda x: x.mid).astype(float)
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=trend, x="position_mid", y="fwd_kl", hue="source", marker="o")
    plt.yscale("log")
    plt.xlabel("Normalized token position")
    plt.ylabel("Median forward KL (log scale)")
    plt.title("Divergence Across Trace Position")
    plt.tight_layout()
    plt.savefig(fig_dir / "kl_position_trend.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 6))
    sample = predictions.sample(min(5000, len(predictions)), random_state=cfg.seed)
    sns.scatterplot(
        data=sample,
        x="pred_rich",
        y="log1p_fwd_kl",
        hue="source",
        alpha=0.35,
        s=12,
        linewidth=0,
    )
    lim_max = max(sample["pred_rich"].max(), sample["log1p_fwd_kl"].max())
    plt.plot([0, lim_max], [0, lim_max], color="black", linewidth=1)
    plt.xlabel("Predicted log1p(KL)")
    plt.ylabel("Actual log1p(KL)")
    plt.title("Rich Predictor Calibration on Held-Out Examples")
    plt.tight_layout()
    plt.savefig(fig_dir / "predictor_scatter.png", dpi=180)
    plt.close()

    enrich_path = WORKSPACE / cfg.output_dir / "residual_enrichment.csv"
    if enrich_path.exists():
        enrichment = pd.read_csv(enrich_path)
        plot_enrich = enrichment[
            (enrichment["feature"] == "token_class") & (enrichment["background_frac"] > 0.005)
        ].nlargest(10, "enrichment")
        if len(plot_enrich):
            plt.figure(figsize=(9, 5))
            sns.barplot(data=plot_enrich, x="enrichment", y="value", color="#4677a9")
            plt.axvline(1.0, color="black", linewidth=1)
            plt.xlabel("High-residual enrichment vs background")
            plt.ylabel("Token class")
            plt.title("Token Classes Overrepresented Among Rich Predictor Misses")
            plt.tight_layout()
            plt.savefig(fig_dir / "residual_token_class_enrichment.png", dpi=180)
            plt.close()


def write_environment(cfg: Config, elapsed_seconds: float, n_rows: int) -> dict[str, Any]:
    env = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpus": [],
        "config": asdict(cfg),
        "elapsed_seconds": elapsed_seconds,
        "scored_rows": n_rows,
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            env["gpus"].append(
                {
                    "index": i,
                    "name": props.name,
                    "total_memory_mib": int(props.total_memory / (1024**2)),
                }
            )
    with open(WORKSPACE / cfg.output_dir / "environment.json", "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)
    return env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples-per-source", type=int, default=Config.examples_per_source)
    parser.add_argument("--max-length", type=int, default=Config.max_length)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--base-model", default=Config.base_model)
    parser.add_argument("--instruct-model", default=Config.instruct_model)
    parser.add_argument("--device", default=Config.device)
    parser.add_argument("--bootstrap-iterations", type=int, default=Config.bootstrap_iterations)
    args = parser.parse_args()

    cfg = Config(
        examples_per_source=args.examples_per_source,
        max_length=args.max_length,
        batch_size=args.batch_size,
        base_model=args.base_model,
        instruct_model=args.instruct_model,
        device=args.device,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    set_seed(cfg.seed)
    ensure_dirs(cfg)
    start = time.time()
    print("Config:", json.dumps(asdict(cfg), indent=2), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.instruct_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = build_examples(cfg)
    encoded = encode_examples(tokenizer, examples, cfg.max_length)
    with open(WORKSPACE / cfg.output_dir / "sampled_examples.json", "w", encoding="utf-8") as f:
        safe_examples = []
        for ex in encoded:
            safe_examples.append(
                {
                    "example_id": ex["example_id"],
                    "source": ex["source"],
                    "prompt_end": ex["prompt_end"],
                    "has_response": ex["has_response"],
                    "tokens": len(ex["input_ids"]),
                    "metadata": ex["metadata"],
                    "text_preview": sanitize_snippet(
                        {"source": ex["source"], "is_safe": ex["metadata"].get("is_safe", True)},
                        ex["text"][:260],
                    ),
                }
            )
        json.dump(safe_examples, f, indent=2)
    print(f"Loaded {len(encoded)} encoded examples", flush=True)

    df = score_kl(cfg, tokenizer, encoded)
    df.to_csv(WORKSPACE / cfg.output_dir / "token_kl_metrics.csv.gz", index=False, compression="gzip")
    stats_df = source_stats(df)
    stats_df.to_csv(WORKSPACE / cfg.output_dir / "source_stats.csv", index=False)
    tests_df = source_tests(df)
    tests_df.to_csv(WORKSPACE / cfg.output_dir / "source_pair_tests.csv", index=False)

    predictions, predictor_results = train_predictors(df, cfg)
    predictions.to_csv(WORKSPACE / cfg.output_dir / "heldout_predictions.csv", index=False)

    residual_summary = residual_analysis(predictions, cfg)
    make_figures(df, predictions, cfg)

    elapsed = time.time() - start
    env = write_environment(cfg, elapsed, len(df))
    summary = {
        "environment": env,
        "source_stats": stats_df.to_dict(orient="records"),
        "source_tests": tests_df.to_dict(orient="records"),
        "predictor_results": predictor_results,
        "residual_summary": residual_summary,
    }
    with open(WORKSPACE / cfg.output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["predictor_results"]["models"], indent=2), flush=True)
    print(f"Finished in {elapsed:.1f}s with {len(df)} token rows", flush=True)


if __name__ == "__main__":
    main()
