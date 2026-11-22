#!/usr/bin/env bash
# Download ML models for Talksify.
# Run once after cloning the repo.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="$ROOT_DIR/src-tauri/models"

mkdir -p "$MODELS_DIR/stt" "$MODELS_DIR/qwen-0.6b"

if command -v sha256sum >/dev/null 2>&1; then
  sha256_file() { sha256sum "$1" | awk '{print $1}'; }
elif command -v shasum >/dev/null 2>&1; then
  sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }
else
  echo "Error: neither sha256sum nor shasum is installed; cannot verify model checksums." >&2
  exit 1
fi

LLM_TMP=""
cleanup_llm_tmp() {
  if [ -n "$LLM_TMP" ] && [ -f "$LLM_TMP" ]; then
    rm -f "$LLM_TMP"
  fi
}
trap cleanup_llm_tmp EXIT
trap 'cleanup_llm_tmp; exit 130' INT
trap 'cleanup_llm_tmp; exit 143' TERM

# Nvidia Parakeet TDT 0.6B v3, 25 European languages (int8, ~490MB). The decoder size check
# replaces an older v2 model.
STT_NAME="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
STT_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$STT_NAME.tar.bz2"
STT_DECODER="$MODELS_DIR/stt/decoder.int8.onnx"
STT_DECODER_SIZE=11845275
if [ ! -f "$STT_DECODER" ] || [ "$(wc -c < "$STT_DECODER" | tr -d ' ')" != "$STT_DECODER_SIZE" ]; then
  echo "Downloading Parakeet STT model..."
  TMP=$(mktemp -d)
  curl -L -o "$TMP/stt.tar.bz2" "$STT_URL"
  tar xjf "$TMP/stt.tar.bz2" -C "$TMP"
  SRC="$TMP/$STT_NAME"
  cp "$SRC/encoder.int8.onnx" "$MODELS_DIR/stt/encoder.int8.onnx"
  cp "$SRC/decoder.int8.onnx" "$MODELS_DIR/stt/decoder.int8.onnx"
  cp "$SRC/joiner.int8.onnx" "$MODELS_DIR/stt/joiner.int8.onnx"
  cp "$SRC/tokens.txt" "$MODELS_DIR/stt/tokens.txt"
  rm -rf "$TMP"
  echo "Parakeet model ready."
else
  echo "STT model already present."
fi

# Talksify fine-tuned Qwen3.5-0.8B GGUF (Epoch 2 (best_adapter), Q6_K, ~601 MB, text only, 25 European languages).
# Verify by SHA-256 because Epoch 1 and Epoch 2 Q6_K have the exact same byte size.
LLM_URL="https://huggingface.co/garret2k2/talksify-qwen35-0.8b-gguf/resolve/main/talksify-qwen35-0.8b-epoch2-Q6_K.gguf"
LLM_FILE="$MODELS_DIR/qwen-0.6b/model.gguf"
LLM_SHA256="6b6beff4b254565030aeddb525ce61a3999887838e671d1fa0b6d8ebef5094ae"
LLM_CURRENT_SHA256=""
if [ -f "$LLM_FILE" ]; then
  LLM_CURRENT_SHA256="$(sha256_file "$LLM_FILE")"
fi
if [ "$LLM_CURRENT_SHA256" != "$LLM_SHA256" ]; then
  echo "Downloading LLM model..."
  LLM_TMP="$(mktemp "$MODELS_DIR/qwen-0.6b/model.gguf.tmp.XXXXXX")"
  curl -fL -o "$LLM_TMP" "$LLM_URL"
  DOWNLOADED_SHA256="$(sha256_file "$LLM_TMP")"
  if [ "$DOWNLOADED_SHA256" != "$LLM_SHA256" ]; then
    echo "Error: SHA-256 mismatch for downloaded LLM model." >&2
    echo "  expected: $LLM_SHA256" >&2
    echo "  actual:   $DOWNLOADED_SHA256" >&2
    exit 1
  fi
  chmod 644 "$LLM_TMP"
  mv -f "$LLM_TMP" "$LLM_FILE"
  LLM_TMP=""
  echo "LLM model ready."
else
  echo "LLM model already present."
fi

# Silero VAD (~630KB), used to cut audio into chunks at pauses
VAD_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
mkdir -p "$MODELS_DIR/vad"
if [ ! -f "$MODELS_DIR/vad/silero_vad.onnx" ]; then
  echo "Downloading VAD model..."
  curl -L -o "$MODELS_DIR/vad/silero_vad.onnx" "$VAD_URL"
  echo "VAD model ready."
else
  echo "VAD model already present."
fi

echo "All models in place."
