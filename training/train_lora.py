#!/usr/bin/env python3
"""Native bf16 LoRA fine-tuning script for Talksify Qwen/Qwen3.5-0.8B.

Supports:
- Native bf16 CUDA training on RTX PRO 6000 (96 GB VRAM)
- Apple Silicon MPS / CPU execution (with MLX recipe documented in README.md)
- Exact chat-template alignment with ``src-tauri/core/src/llm_backend.rs``
- Strict completions-only loss masking (system + user + ``<think>\\n\\n</think>\\n\\n`` masked to -100)
- Per-epoch validation loss logging across ``overall``, ``en``, ``tier1``, and ``tier2`` splits
- Fast ``--smoke-test`` mode verifying schema, chat template, and loss masking invariants

Usage:
    # Full bf16 LoRA training on queue GPU
    uv run python train_lora.py \\
        --data-dir .artifacts/dataset \\
        --output-dir .artifacts/run_qwen35_08b

    # Fast smoke test (schema + chat-template + completions-only loss masking)
    uv run python train_lora.py --smoke-test --data-dir .artifacts/smoke_dataset
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from prompts import (
    BASE_MODEL_ID,
    PARAKEET_V3_LANGUAGES,
    format_user_message,
    load_clean_prompt,
    render_chat_pair,
    tokenize_for_sft,
)

logger = logging.getLogger("talksify.train_lora")

# All linear projections across Qwen3.5-0.8B's full-attention, Gated DeltaNet linear-attention,
# and MLP blocks (verified against Qwen3_5ForCausalLM named_modules).
QWEN35_LORA_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def validate_dataset_rows(rows: list[dict[str, Any]], expected_system_prompt: str) -> None:
    """Verify every dataset row matches Talksify's exact production prompt & framing schema."""
    if not rows:
        raise ValueError("Dataset split is empty")
    for idx, row in enumerate(rows):
        for field in ("raw", "clean", "lang", "tier", "messages"):
            if field not in row:
                raise ValueError(f"Row {idx} missing required field {field!r}")
        if row["lang"] not in PARAKEET_V3_LANGUAGES:
            raise ValueError(f"Row {idx} has unsupported Parakeet v3 language {row['lang']!r}")
        msgs = row["messages"]
        if not isinstance(msgs, list) or len(msgs) != 3:
            raise ValueError(f"Row {idx} must have exactly 3 messages (system, user, assistant)")
        if msgs[0]["role"] != "system" or msgs[0]["content"] != expected_system_prompt:
            raise ValueError(f"Row {idx} system prompt does not match production clean_prompt")
        expected_user = format_user_message(row["raw"])
        if msgs[1]["role"] != "user" or msgs[1]["content"] != expected_user:
            raise ValueError(f"Row {idx} user message does not match <transcript> framing")
        if msgs[2]["role"] != "assistant" or msgs[2]["content"] != row["clean"]:
            raise ValueError(f"Row {idx} assistant message does not match clean target")


class SftDataset(Dataset[dict[str, Any]]):
    """Pre-tokenized completions-only SFT dataset."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        system_prompt: str,
        max_seq_len: int = 1536,
    ) -> None:
        self.samples: list[dict[str, Any]] = []
        self.skipped_too_long = 0
        for row in rows:
            tok = tokenize_for_sft(
                tokenizer,
                raw_transcript=row["raw"],
                clean_transcript=row["clean"],
                system_prompt=system_prompt,
                max_seq_len=max_seq_len,
            )
            if tok is None:
                self.skipped_too_long += 1
                continue
            tok["lang"] = row["lang"]
            tok["tier"] = row["tier"]
            self.samples.append(tok)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Mega-batch length-grouped batch sampler reducing padding overhead while preserving randomness."""

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        mega_batch_mult: int = 32,
        seed: int = 407,
    ) -> None:
        self.lengths = lengths
        self.batch_size = batch_size
        self.mega_batch_mult = mega_batch_mult
        self.seed = seed
        self._epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        indices = list(range(len(self.lengths)))
        rng.shuffle(indices)
        mega_size = self.batch_size * self.mega_batch_mult
        batches: list[list[int]] = []
        for start in range(0, len(indices), mega_size):
            chunk = indices[start : start + mega_size]
            chunk.sort(key=lambda idx: self.lengths[idx])
            for b_start in range(0, len(chunk), self.batch_size):
                batches.append(chunk[b_start : b_start + self.batch_size])
        rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return math.ceil(len(self.lengths) / self.batch_size)


@dataclass
class CompletionsCollator:
    pad_token_id: int

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(len(item["input_ids"]) for item in batch)
        # Pad to multiple of 8 for Tensor Core efficiency.
        max_len = ((max_len + 7) // 8) * 8
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        tiers: list[str] = []
        for item in batch:
            pad_n = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * pad_n)
            attention_mask.append(item["attention_mask"] + [0] * pad_n)
            labels.append(item["labels"] + [-100] * pad_n)
            tiers.append(item["tier"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "tiers": tiers,
        }


def resolve_device_and_dtype(device_arg: str) -> tuple[torch.device, torch.dtype]:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda"), torch.bfloat16
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps"), torch.bfloat16
        return torch.device("cpu"), torch.bfloat16
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda"), torch.bfloat16
    if device_arg == "mps":
        return torch.device("mps"), torch.bfloat16
    return torch.device("cpu"), torch.bfloat16


def verify_completions_only_masking(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    system_prompt: str,
    max_seq_len: int,
    sample_limit: int = 32,
) -> dict[str, Any]:
    """Assert strict template alignment and completions-only loss masking across sample rows."""
    checked = 0
    prompt_lens: list[int] = []
    comp_lens: list[int] = []
    for row in rows[:sample_limit]:
        prefix, full = render_chat_pair(
            tokenizer, row["raw"], row["clean"], system_prompt=system_prompt
        )
        assert "<think>\n\n</think>\n\n" in prefix, "Expected enable_thinking=False empty think block"
        assert full.startswith(prefix), "Full chat render must start with generation prompt prefix"
        tok = tokenize_for_sft(
            tokenizer,
            row["raw"],
            row["clean"],
            system_prompt=system_prompt,
            max_seq_len=max_seq_len,
        )
        if tok is None:
            continue
        p_len = tok["prompt_len"]
        assert all(lbl == -100 for lbl in tok["labels"][:p_len]), "Prompt tokens must be masked"
        assert all(
            lbl == tid for lbl, tid in zip(tok["labels"][p_len:], tok["input_ids"][p_len:])
        ), "Completion tokens must be unmasked"
        decoded_unmasked = tokenizer.decode(
            [tid for tid, lbl in zip(tok["input_ids"], tok["labels"]) if lbl != -100]
        )
        assert decoded_unmasked.startswith(row["clean"]), (
            f"Unmasked completion mismatch: {decoded_unmasked!r} vs {row['clean']!r}"
        )
        assert "<think>" not in decoded_unmasked, "Empty think block leaked into loss tokens"
        prompt_lens.append(p_len)
        comp_lens.append(tok["completion_len"])
        checked += 1
    if checked == 0:
        raise ValueError("No rows fit within max_seq_len during smoke masking check")
    return {
        "checked_rows": checked,
        "avg_prompt_tokens": round(sum(prompt_lens) / checked, 1),
        "avg_completion_tokens": round(sum(comp_lens) / checked, 1),
        "max_total_tokens": max(p + c for p, c in zip(prompt_lens, comp_lens)),
    }


@torch.inference_mode()
def evaluate_by_tier(
    model: Any,
    val_loader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> dict[str, float]:
    """Compute token-weighted cross-entropy loss overall and per tier (en, tier1, tier2)."""
    model.eval()
    loss_sum: dict[str, float] = {"overall": 0.0, "en": 0.0, "tier1": 0.0, "tier2": 0.0}
    tok_sum: dict[str, int] = {"overall": 0, "en": 0, "tier1": 0, "tier2": 0}
    ce = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        tiers = batch["tiers"]

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        per_tok = ce(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        per_tok = per_tok.view(shift_labels.shape)

        for idx, tier in enumerate(tiers):
            mask = shift_labels[idx] != -100
            n_tok = int(mask.sum().item())
            if n_tok == 0:
                continue
            row_loss = float(per_tok[idx][mask].sum().item())
            loss_sum["overall"] += row_loss
            tok_sum["overall"] += n_tok
            if tier in loss_sum:
                loss_sum[tier] += row_loss
                tok_sum[tier] += n_tok

    model.train()
    return {
        key: round(loss_sum[key] / tok_sum[key], 4)
        for key in ("overall", "en", "tier1", "tier2")
        if tok_sum[key] > 0
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen/Qwen3.5-0.8B with bf16 LoRA for Talksify dictation cleanup."
    )
    parser.add_argument("--base-model", type=str, default=BASE_MODEL_ID)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / ".artifacts" / "dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / ".artifacts" / "lora_qwen35_08b",
    )
    parser.add_argument("--max-seq-len", type=int, default=1536)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=407)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Validate dataset schema, exact chat template alignment, and completions-only loss masking.",
    )
    parser.add_argument(
        "--smoke-train-steps",
        type=int,
        default=0,
        help="When > 0 with --smoke-test, run N real LoRA forward+backward steps and save adapter.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    system_prompt = load_clean_prompt()
    train_rows = load_jsonl(args.data_dir / "train.jsonl")
    val_rows = load_jsonl(args.data_dir / "val.jsonl")
    validate_dataset_rows(train_rows, system_prompt)
    validate_dataset_rows(val_rows, system_prompt)

    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    mask_report = verify_completions_only_masking(
        tokenizer, train_rows, system_prompt, args.max_seq_len
    )
    logger.info("Chat template & completions-only masking verified: %s", mask_report)

    if args.smoke_test and args.smoke_train_steps <= 0:
        logger.info("Smoke test passed cleanly.")
        return 0

    device, dtype = resolve_device_and_dtype(args.device)
    if device.type == "cpu" and not args.smoke_test:
        raise RuntimeError(
            "Full LoRA training requires CUDA (or Apple Silicon MPS), but torch.cuda.is_available() "
            "is False. Pass --smoke-test --smoke-train-steps 2 for a CPU smoke training check, or "
            "run on the queue RTX PRO 6000 GPU."
        )

    logger.info("Loading %s on %s (%s)...", args.base_model, device, dtype)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    from peft import LoraConfig, get_peft_model

    present_linears = {
        name.split(".")[-1]
        for name, mod in model.named_modules()
        if isinstance(mod, torch.nn.Linear)
    }
    target_modules = [m for m in QWEN35_LORA_TARGET_MODULES if m in present_linears]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.to(device)
    model.print_trainable_parameters()

    if args.smoke_test:
        train_rows = train_rows[: max(4, args.smoke_train_steps * 2)]
        val_rows = val_rows[:4]

    train_ds = SftDataset(train_rows, tokenizer, system_prompt, max_seq_len=args.max_seq_len)
    val_ds = SftDataset(val_rows, tokenizer, system_prompt, max_seq_len=args.max_seq_len)
    logger.info(
        "Tokenized train=%d (skipped_long=%d), val=%d (skipped_long=%d)",
        len(train_ds),
        train_ds.skipped_too_long,
        len(val_ds),
        val_ds.skipped_too_long,
    )

    collator = CompletionsCollator(pad_token_id=tokenizer.pad_token_id)
    bs = 1 if args.smoke_test else args.batch_size
    grad_accum = 1 if args.smoke_test else args.grad_accum
    pin_mem = device.type == "cuda"
    if args.smoke_test:
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collator)
    else:
        lengths = [len(s["input_ids"]) for s in train_ds.samples]
        batch_sampler = LengthBucketBatchSampler(lengths, batch_size=bs, seed=args.seed)
        train_loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            collate_fn=collator,
            pin_memory=pin_mem,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        collate_fn=collator,
        pin_memory=pin_mem,
    )

    epochs = 1 if args.smoke_test else args.epochs
    steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum))
    total_steps = args.smoke_train_steps if args.smoke_test else steps_per_epoch * epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio)) if not args.smoke_test else 0

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {
        "base_model": args.base_model,
        "device": str(device),
        "dtype": str(dtype),
        "smoke_test": args.smoke_test,
        "hyperparameters": {
            "max_seq_len": args.max_seq_len,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": target_modules,
            "epochs": epochs,
            "batch_size": bs,
            "grad_accum": grad_accum,
            "gradient_checkpointing": args.gradient_checkpointing,
            "lr": args.lr,
            "warmup_steps": warmup_steps,
            "total_steps": total_steps,
        },
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "mask_report": mask_report,
        "train_log": [],
        "val_epochs": [],
    }

    baseline_val = evaluate_by_tier(model, val_loader, device)
    logger.info("Epoch 0 (pre-train baseline) val loss: %s", baseline_val)
    metrics["val_epochs"].append({"epoch": 0, "val_loss": baseline_val})

    best_val_loss = float("inf")
    global_step = 0
    t_start = time.time()
    model.train()

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        micro_in_accum = 0
        optimizer.zero_grad(set_to_none=True)

        for step_idx, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss / grad_accum
            loss.backward()
            running_loss += float(out.loss.item())
            micro_in_accum += 1

            is_boundary = micro_in_accum == grad_accum or step_idx == len(train_loader)
            if not is_boundary:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            avg_step_loss = running_loss / micro_in_accum
            running_loss = 0.0
            micro_in_accum = 0

            if global_step == 1 or global_step % args.log_every == 0 or global_step == total_steps:
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.time() - t_start
                entry = {
                    "step": global_step,
                    "epoch": round(epoch - 1 + step_idx / len(train_loader), 3),
                    "loss": round(avg_step_loss, 4),
                    "lr": round(lr_now, 7),
                    "elapsed_sec": round(elapsed, 1),
                }
                metrics["train_log"].append(entry)
                logger.info(
                    "step=%d/%d epoch=%.2f loss=%.4f lr=%.2e elapsed=%.1fs",
                    global_step,
                    total_steps,
                    entry["epoch"],
                    entry["loss"],
                    lr_now,
                    elapsed,
                )

            if args.smoke_test and global_step >= args.smoke_train_steps:
                break

        val_loss = evaluate_by_tier(model, val_loader, device)
        logger.info("Epoch %d val loss: %s", epoch, val_loss)
        metrics["val_epochs"].append({"epoch": epoch, "val_loss": val_loss})

        ckpt_dir = args.output_dir / f"checkpoint-epoch-{epoch}"
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)

        overall_val = val_loss.get("overall", float("inf"))
        if overall_val < best_val_loss:
            best_val_loss = overall_val
            best_dir = args.output_dir / "best_adapter"
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)

        if args.smoke_test and global_step >= args.smoke_train_steps:
            break

    final_dir = args.output_dir / "final_adapter"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    metrics["total_elapsed_sec"] = round(time.time() - t_start, 1)

    (args.output_dir / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Saved final LoRA adapter to %s in %.1fs", final_dir, metrics["total_elapsed_sec"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
