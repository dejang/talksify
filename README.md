# Talksify

Local voice dictation for macOS. Press a shortcut, talk, press it again, and cleaned-up text gets typed into whatever window you have focused. Everything runs on your machine, nothing leaves your computer.

## How it works

Audio is captured from your mic and passed through a three-stage local pipeline:

1. **Voice activity detection (Silero VAD)** splits speech at natural pauses so long dictations are processed in chunks.
2. **Speech-to-text (Nvidia Parakeet TDT 0.6B v3)** transcribes each chunk via `sherpa-onnx` (int8, ~490 MB).
3. **Cleanup (Qwen3-0.6B + rules)** strips filler words (`um`, `uh`), formats spoken lists, fixes punctuation and capitalization, and resolves self-corrections ("no wait, make that Tuesday") via `mistral.rs` on Metal (Q8_0 GGUF, ~610 MB).

Once processed, the text is typed into the active app (or pasted via the clipboard if the app doesn't accept synthetic keystrokes). If the LLM isn't available, dictation falls back to basic rule-based cleanup.

## Supported languages

Parakeet TDT 0.6B v3 detects and transcribes **25 European languages** automatically without needing to switch settings:

Bulgarian, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, German, Greek, Hungarian, Italian, Latvian, Lithuanian, Maltese, Polish, Portuguese, Romanian, Russian, Slovak, Slovenian, Spanish, Swedish, and Ukrainian.

## Getting started

Requires **macOS on Apple Silicon**, Rust, Node.js, and `pnpm`.

```bash
pnpm install
./scripts/download-models.sh   # downloads ~1.2 GB of models into src-tauri/models/
pnpm tauri dev
```

On first launch, grant **Microphone** access when prompted, and add the app under **System Settings -> Privacy & Security -> Accessibility** so it can type into other windows.

The default shortcut is **`Ctrl+Alt+Space`** (press once to start recording, press again to finish).

To build a release app bundle:

```bash
pnpm tauri build
```

## Configuration

You can optionally create `~/Library/Application Support/talksify/config.toml` to change the shortcut or pin a specific microphone:

```toml
hotkey = "Ctrl+Alt+Space"
# audio_device = "MacBook Pro Microphone"
```
