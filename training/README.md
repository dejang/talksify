# Talksify Multilingual `Qwen/Qwen3.5-0.8B` Fine-Tuning Pipeline

Reproducible LoRA fine-tuning and GGUF export pipeline turning raw Parakeet TDT 0.6B v3 speech transcripts into polished written prose across **all 25 European languages**, engineered directly around the 5 failure modes found in `F-406`:

1. **Scoped self-corrections & quoted-negation traps** — resolve local retractions to the *new* replacement without wiping earlier sentences, and never delete text when negation/retraction phrases are quoted (`saying "no, that's definitely not it"`, `she said "scratch that"`).
2. **Cut-off word & contraction false starts** — cleanly collapse truncated stems (`in the dir in the direction`, `the arch the architecture`) and contraction/verb restarts (`It's it runs`, `is it are they`).
3. **Parakeet + Silero VAD mid-sentence pause breaks** — join false `. Capital` splits and lowercase stray mid-sentence capitals (`on the. Build server`, `can Work towards`).
4. **25-language multilingual coverage with zero English-translation drift** — cover Tier 1 (`de`, `fr`, `es`, `it`, `ro` — ~60% of multilingual slice) and Tier 2 (`bg`, `hr`, `cs`, `da`, `nl`, `et`, `fi`, `el`, `hu`, `lv`, `lt`, `mt`, `pl`, `pt`, `ru`, `sk`, `sl`, `sv`, `uk` — ~40%) using Talksify's exact English `clean_prompt`, removing idiomatic local fillers and self-correction cues without relocating cues (e.g., never moving `"Scuze,"` to the start).
5. **Formatting restraint** — zero assistant chatter, zero unwanted blank-line paragraph injections, zero silent deletions, and zero clause duplications; reconciled with `src-tauri/core/src/conventions.rs`.

## Exact Production Prompt Alignment

Every training and validation example is formatted byte-for-byte identically to production inference:

- **System message**: extracted directly from `clean_prompt` in `src-tauri/core/src/llm_postprocessor.rs`.
- **User message**: `<transcript>\n{raw}\n</transcript>` (matching `src-tauri/core/src/llm_backend.rs`).
- **Chat template**: `enable_thinking=False` (`<|im_start|>assistant\n<think>\n\n</think>\n\n{clean}<|im_end|>\n`).
- **Completions-only loss**: all system, user, and `<think>\n\n</think>\n\n` tokens are masked to `-100` so gradients apply strictly to the cleaned assistant output and closing `<|im_end|>`.

## Anti-Contamination Policy

- `/home/agent/recordings` and `src-tauri/core/tests/cleanup_quality_eval/cases.rs` are **strictly held-out evaluation sets**.
- Never inspect, sample, paraphrase, or derive training rows or teacher prompts from either location.
- Never commit datasets, LoRA checkpoints, or `.gguf` weights to git (all outputs under `training/.artifacts/` and `*.gguf` are gitignored).
- Published production GGUFs are hosted on Hugging Face and downloaded to `src-tauri/models/qwen-0.6b/model.gguf` via `scripts/download-models.sh` (keep candidate exports isolated under `training/.artifacts/` while benchmarking).

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.12 (3.12–3.13):

```bash
cd training
uv sync --python 3.12 --extra dev
```

On NVIDIA CUDA GPUs (e.g., RTX PRO 6000 96 GB VRAM), PyTorch with CUDA wheels is installed automatically by `uv sync`.

## Fast Verification (Unit Tests + Smoke Pipeline)

```bash
cd training

# 1. Run unit tests (prompt parity with Rust, conventions, QA filters, masking, quotas)
uv run pytest -v

# 2. Fast dataset smoke test (live HF license verification + small sample + splits)
uv run python prepare_data.py --smoke-test --output-dir .artifacts/smoke_dataset

# 3. Validate dataset schema, exact chat template alignment, and completions-only loss masking
uv run python train_lora.py --smoke-test --data-dir .artifacts/smoke_dataset

# 4. Optional: 2-step smoke train + full LoRA merge -> GGUF (--no-mtp) -> Q6_K/Q8_0 -> talksify-cli check
uv run python train_lora.py \
  --smoke-test --smoke-train-steps 2 \
  --data-dir .artifacts/smoke_dataset \
  --output-dir .artifacts/smoke_lora

uv run python export_gguf.py \
  --adapter-dir .artifacts/smoke_lora/final_adapter \
  --output-dir .artifacts/smoke_export \
  --quants Q6_K Q8_0
```

## Full Training & Export Workflow (CUDA GPU — RTX PRO 6000 96 GB)

### 1. Prepare Dataset (~67% English, ~33% Multilingual across 25 Languages)

`prepare_data.py` audits Hugging Face dataset licenses in `manifest.json` and ingests all 5 configured datasets unconditionally (4 English datasets + native German `nyralabs/disfluency_speech_german`), then runs multi-round GPU teacher adaptation to fill all 24 non-English Parakeet v3 language quotas.

```bash
cd training
uv run python prepare_data.py \
  --output-dir .artifacts/dataset \
  --max-en-rows 28000 \
  --en-gapfill-rows 2800 \
  --multilingual-rows 10000 \
  --teacher-model Qwen/Qwen3-4B-Instruct-2507-FP8 \
  --teacher-batch-size 192
```

Teacher outputs are cached incrementally to `.artifacts/dataset/teacher_multilingual_cache.jsonl` so interrupted runs resume cleanly. `qa_filters.py` enforces zero `EMPTY` sentinels, strict `raw == clean` on passthrough categories (`clean_passthrough`, `passthrough`, `no_edit`), a single `is_retraction_category` rule across loaders and local repair, strict `TokenAlignment` checks (zero invented alphabetic words outside homophone/stem fixes, bounded moved/reordered words, `>= 0.90` LCS retention on `>= 10`-word targets, zero line-start or inline `1. ... 2. ...` list restructuring, and bounded contiguous deleted spans), and caps `google-research-datasets/disfl_qa` to `<= 8%` of English rows. To apply filter updates locally to an existing `.artifacts/dataset` (regenerating `synthetic_en_gapfill` locally and dropping stale teacher `thinking_aloud` rows without full HF/teacher re-extraction):

```bash
cd training
uv run python prepare_data.py \
  --repair-existing \
  --output-dir .artifacts/dataset \
  --en-gapfill-rows 2800 \
  --regen-passthroughs 0 \
  --skip-teacher
```

### 2. Native `bf16` LoRA Fine-Tuning

```bash
cd training
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python train_lora.py \
  --data-dir .artifacts/dataset \
  --output-dir .artifacts/lora_qwen35_08b \
  --max-seq-len 1536 \
  --lora-r 32 \
  --lora-alpha 64 \
  --lr 1.5e-4 \
  --epochs 2 \
  --batch-size 8 \
  --grad-accum 4
```

Validation cross-entropy loss is logged at epoch 0 (pre-train baseline) and after every epoch broken down by `overall`, `en`, `tier1` (`de/fr/es/it/ro`), and `tier2` (19 languages), and saved to `.artifacts/lora_qwen35_08b/training_metrics.json`.

### 3. Merge LoRA & Export Published `< 1 GB` GGUFs (`--no-mtp`)

Both **Epoch 2** (`best_adapter` / `checkpoint-epoch-2`, lowest validation loss, default shipped via `scripts/download-models.sh`) and **Epoch 1** (`checkpoint-epoch-1`, published conservative reference checkpoint) are exported in **`Q6_K`** quantization and hosted on Hugging Face at [`garret2k2/talksify-qwen35-0.8b-gguf`](https://huggingface.co/garret2k2/talksify-qwen35-0.8b-gguf):

```bash
cd training

# Export Epoch 2 Q6_K (best_adapter / checkpoint-epoch-2, default shipped model fetched by scripts/download-models.sh)
uv run python export_gguf.py \
  --adapter-dir .artifacts/lora_qwen35_08b/best_adapter \
  --output-dir .artifacts/export_qwen35_08b \
  --quants Q6_K

# Export Epoch 1 Q6_K (published conservative reference checkpoint)
uv run python export_gguf.py \
  --adapter-dir .artifacts/lora_qwen35_08b/checkpoint-epoch-1 \
  --output-dir .artifacts/export_epoch1 \
  --quants Q6_K
```

| Published HF File (`garret2k2/talksify-qwen35-0.8b-gguf`) | Checkpoint / Local Artifact | Size & Layout | SHA-256 |
| --- | --- | --- | --- |
| [`talksify-qwen35-0.8b-epoch2-Q6_K.gguf`](https://huggingface.co/garret2k2/talksify-qwen35-0.8b-gguf/resolve/main/talksify-qwen35-0.8b-epoch2-Q6_K.gguf) *(default in `scripts/download-models.sh`)* | `best_adapter` (`checkpoint-epoch-2`) -> `.artifacts/export_qwen35_08b/talksify-qwen35-0.8b-Q6_K.gguf` | `629,743,712` bytes (`600.57 MB`, `< 1 GB`), `320` unfused tensors (`--no-mtp`) | `6b6beff4b254565030aeddb525ce61a3999887838e671d1fa0b6d8ebef5094ae` |
| [`talksify-qwen35-0.8b-Q6_K.gguf`](https://huggingface.co/garret2k2/talksify-qwen35-0.8b-gguf/resolve/main/talksify-qwen35-0.8b-Q6_K.gguf) *(conservative reference checkpoint)* | `checkpoint-epoch-1` -> `.artifacts/export_epoch1/talksify-qwen35-0.8b-Q6_K.gguf` | `629,743,712` bytes (`600.57 MB`, `< 1 GB`), `320` unfused tensors (`--no-mtp`) | `df21112849d9651b7b8bf9fc9d3c4fdd2e5bdf3b8bc09b1cd2ae312cb0e78d9b` |

`export_gguf.py` merges the LoRA weights into `Qwen/Qwen3.5-0.8B` in `bf16`, checks out pinned `llama.cpp` commit `19bc41dbb41e858eebc269b49ccdec1f0d28880c` and builds `llama-quantize` from that pinned checkout if needed, converts via `convert_hf_to_gguf.py --no-mtp` (disabling `Qwen3_5TextModel._SHARED_INPUT_MERGES` in-process so `mistralrs` receives the exact unfused 320-tensor `attn_qkv`/`attn_gate`/`ssm_beta`/`ssm_alpha` layout), verifies `< 1 GB` file sizes and zero MTP/vision tensors, computes SHA-256 checksums, and runs an isolated `talksify-cli --replay` smoke verification. Pass `--quants Q6_K Q8_0` to also export `Q8_0` checkpoints for comparison.

### 4. Evaluate Against Talksify Harness & Eval Suites

Stage each candidate GGUF in an isolated model directory while benchmarking:

```bash
# Dump raw STT once from held-out recordings (uses stock models)
cargo run --manifest-path src-tauri/Cargo.toml -p talksify-cli -- \
  /home/agent/recordings --dump-raw training/.artifacts/raw_recordings.json

# Stage candidate GGUF (Epoch 2 best_adapter default)
mkdir -p training/.artifacts/eval_q6k/qwen-0.6b
ln -sf "$(realpath training/.artifacts/export_qwen35_08b/talksify-qwen35-0.8b-Q6_K.gguf)" \
  training/.artifacts/eval_q6k/qwen-0.6b/model.gguf

# 1. Fast replay across all 20 recordings
cargo run --manifest-path src-tauri/Cargo.toml -p talksify-cli -- \
  --replay training/.artifacts/raw_recordings.json \
  --models training/.artifacts/eval_q6k

# 2. Prompt eval suite (10 cases)
LLM_EVAL_MODELS="$(realpath training/.artifacts/eval_q6k)" \
  cargo test --manifest-path src-tauri/Cargo.toml -p talksify-core -- clean_prompt_eval --ignored --nocapture

# 3. Full cleanup quality eval suite (8 categories / 29 cases incl. DE/FR/ES/IT/RO)
LLM_EVAL_MODELS="$(realpath training/.artifacts/eval_q6k)" \
  cargo test --manifest-path src-tauri/Cargo.toml -p talksify-core --test cleanup_quality_eval -- --ignored --nocapture

# 4. Tier 2 multilingual spot-check (nl, pl, pt, cs, sv, el, uk)
cd training && uv run python eval_multilingual.py \
  --gguf .artifacts/export_qwen35_08b/talksify-qwen35-0.8b-Q6_K.gguf
```

## Apple Silicon Local Runs (MPS & MLX)

- **PyTorch MPS** (uses the exact same script, completions-only loss masking, and LoRA targets):
  ```bash
  uv run python train_lora.py \
    --device mps --gradient-checkpointing \
    --batch-size 2 --grad-accum 16 \
    --data-dir .artifacts/dataset \
    --output-dir .artifacts/lora_mps
  ```
- **MLX (`mlx-lm`)**: install `uv sync --extra mlx` on macOS arm64; `train.jsonl` and `val.jsonl` already store standard chat `"messages"` lists compatible with `mlx_lm.lora --mask-prompt --model Qwen/Qwen3.5-0.8B --data .artifacts/dataset --train`.
