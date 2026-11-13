# Talksify

Local voice dictation: speech is transcribed and cleaned up on your machine, never in the cloud.

## Truth Model

**`docs/design.json`** is the architecture board (understudy format, rev 229).
It is the single source of truth for:
- Part responsibilities, operations, state, and failures
- Connections between parts
- Boundaries (what runs where)
- Tests (happy path)

**`docs/PLAN.md`** is the implementation plan.
It references board part IDs. Phases are ordered by dependency.

When implementing, always check the board for:
- What a part owns (responsibility)
- What it exposes (operations with signatures)
- What state it holds
- What can go wrong (failures + recovery)
- How it connects to other parts

## Quick Start

```bash
pnpm install
pnpm tauri dev          # dev mode with hot reload
pnpm lint               # oxlint + ast-grep
cd src-tauri
cargo clippy --workspace --all-targets
cargo run -p talksify-cli -- <file.wav>
cargo run -p talksify-cli -- <folder>
```

## Layout

```
src/                    React overlay (HUD)
src-tauri/src/          Tauri app
  lib.rs                App entry, plugin wiring
  event_handler.rs      Hotkey → Transcriber → Inject glue
  inject/               CGEvent / xdotool (board: inject)
  config.rs             Settings store (board: config)
  updater.rs            App updates
src-tauri/core/src/     Tauri-free transcription pipeline
  lib.rs                Public core API
  audio_source.rs       AudioSource contract (board: audio-source)
  transcriber.rs        Orchestrator (board: module-1)
  recording.rs          WAV writer for saved recordings (board: module-1)
  audio_device.rs       cpal capture (board: module-2)
  stt.rs                STT trait (board: contract-1)
  parakeet_stt.rs       Parakeet via sherpa-onnx (board: module-3)
  raw_transcript.rs     Transcript accumulator (board: module-4)
  llm_postprocessor.rs  mistral.rs cleanup (board: module-5)
  text_format.rs        Rule-based filler and list formatting around the LLM (board: module-5)
src-tauri/cli/src/      WAV transcription CLI
  main.rs               Argument parsing and pipeline lifecycle (board: cli)
  file_source.rs        Real-time WAV source (board: file-source)
  harness.rs            Folder transcript comparison (board: cli)
  score.rs              Tokenization and token-level edit distance (board: cli)
src-tauri/models/       Bundled ML models
docs/design.json        Architecture board (truth model)
docs/PLAN.md            Implementation plan
.ast-grep/              Lint rules
```

## Rules

- `pnpm lint` and `cargo clippy --workspace --all-targets` must pass before any commit
- No `unwrap()`/`expect()` outside test modules
- No ad-hoc `thread::spawn`. Threading goes through Transcriber. One exception: `FileSource` in the CLI owns the thread that plays the file, as cpal's callback thread does for the mic.
- Models are bundled, never downloaded at runtime
