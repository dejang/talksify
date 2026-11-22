#!/usr/bin/env bash
# Download ML models for Talksify.
# Run once after cloning the repo.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="$ROOT_DIR/src-tauri/models"

mkdir -p "$MODELS_DIR/stt" "$MODELS_DIR/qwen-0.6b"

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

# Talksify fine-tuned Qwen3.5-0.8B GGUF (Epoch 1, Q6_K, ~601MB, text only, 25 European languages).
# The size check replaces an older model.gguf.
LLM_URL="https://huggingface.co/garret2k2/talksify-qwen35-0.8b-gguf/resolve/main/talksify-qwen35-0.8b-Q6_K.gguf"
LLM_FILE="$MODELS_DIR/qwen-0.6b/model.gguf"
LLM_SIZE=629743712
if [ ! -f "$LLM_FILE" ] || [ "$(wc -c < "$LLM_FILE" | tr -d ' ')" != "$LLM_SIZE" ]; then
  echo "Downloading LLM model..."
  curl -L -o "$LLM_FILE" "$LLM_URL"
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
