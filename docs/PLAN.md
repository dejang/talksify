# Implementation Plan — Talksify

**Truth model:** `docs/design.json` (understudy board, rev 216).
Every task below references part IDs from that file. When in doubt, the board wins.

## Decisions (locked)

| Choice | Value |
|--------|-------|
| Platform | macOS + Linux |
| Language | Rust + Tauri 2 |
| STT | Nvidia Parakeet via sherpa-onnx |
| LLM | Qwen 0.6B via mistral.rs |
| Text injection | CGEvent (macOS) / xdotool (Linux) |
| Feedback | Floating overlay HUD (Tauri window) |
| Auto-start | Yes (launch agent / systemd user service) |
| Cloud | None — all local |
| Models | Bundled with app (`src-tauri/models/`) |

## Architecture (from board)

### Tauri process boundary (`tauri-app`)

| Part | Name | Role |
|------|------|------|
| `hotkey` | Hotkey Listener | Global hotkey via tauri-plugin-global-shortcut |
| `module-6` | Event Handler | Thin Tauri glue: receives hotkey, calls Transcriber + Inject |
| `inject` | Text Injector | CGEvent / xdotool text injection |
| `config` | Configuration | User settings store |

### Transcriber code boundary (`boundary-1`)

| Part | Name | Role |
|------|------|------|
| `module-1` | Transcriber | Orchestrator. Owns 3 worker threads. Black-box API. |
| `module-2` | AudioDevice | cpal mic capture, lock-free HeapProd/HeapCons queue |
| `contract-1` | STT (trait) | `push_audio(&[f32])`, `subscribe(on_transcribe)` |
| `module-3` | ParakeetSTT | Implements STT trait, warm thread, sherpa-onnx |
| `module-4` | RawTranscript | `sentences: Vec<String>`, `push_words(String)` |
| `module-5` | LLM PostProcessor | mistral.rs, warm thread, `clean_transcript(RawTranscript)` |

### Connections

```
hotkey → Event Handler → Transcriber → AudioDevice
                         Transcriber → STT trait → ParakeetSTT
                         Transcriber → RawTranscript
                         Transcriber → LLM PostProcessor
         Event Handler → Inject
```

### Flow

1. App start: `Transcriber::new(config)` spawns warm ParakeetSTT + LLM threads
2. Hotkey press → Event Handler → `Transcriber.start_transcribing()`
3. Transcriber spawns AudioDevice thread, pulls `HeapCons<f32>`, calls `push_audio` on STT
4. ParakeetSTT transcribes, calls `on_transcribe` callback → writes to `Mutex<RawTranscript>`
5. Hotkey press → Event Handler → `Transcriber.stop_transcribing() -> Result<Transcript, TranscriberError>`
6. Transcriber sets `is_recording=false`, joins AudioDevice thread, runs LLM cleanup, returns Transcript
7. Event Handler calls `inject.type_text(transcript)`

### Failure model

- `is_recording: AtomicBool` is the central kill switch
- Worker threads check it in their loop, bail when false
- Listener thread aggregates errors from workers, flips the switch on failure
- Fatal at init: STT/LLM model load failure → app cannot start
- Inject: ask permission, log if blocked
- Double-start guarded by `is_transcribing` atomic bool

## Implementation Phases

### Phase 1: Skeleton + Config (`config`, `module-6`) ✅

**Goal:** App starts, shows overlay window, hotkey fires, Event Handler logs the event.

- [x] `src-tauri/src/config.rs` — Config struct (hotkey binding, model paths, audio device)
- [x] `src-tauri/src/event_handler.rs` — Tauri command handlers, registers global shortcut
- [x] `src-tauri/src/lib.rs` — Wire plugins, create overlay window, start Event Handler
- [x] `src/` — Minimal React overlay (shows "idle" / "recording" / "processing" states)
- [x] Verify: `pnpm tauri dev` shows overlay, hotkey logs to console

### Phase 2: AudioDevice (`module-2`) ✅

**Goal:** Mic capture produces f32 samples via lock-free queue.

- [x] `src-tauri/src/audio_device.rs` — cpal stream, `HeapProd<f32>` producer
- [x] `start_capture() -> Result<HeapCons<f32>, AudioError>`
- [x] `stop_capture() -> Result<(), AudioError>`
- [x] Failure: cannot acquire device → return error
- [ ] Unit test: start/stop produces samples (needs real hardware)

### Phase 3: STT trait + ParakeetSTT (`contract-1`, `module-3`) ✅

**Goal:** Warm thread accepts audio, produces transcripts via callback.

- [x] `src-tauri/src/stt.rs` — STT trait definition
- [x] `src-tauri/src/parakeet_stt.rs` — sherpa-onnx OnlineRecognizer (NeMo CTC), streaming decode, endpoint detection
- [x] `push_audio(&mut self, audio: &[f32])` — accumulates in `scratch_buffer`
- [x] `subscribe(on_transcribe)` — registers callback
- [x] Wire real sherpa-onnx model
- [ ] Model bundled at `src-tauri/models/parakeet/` (download model.onnx + tokens.txt)
- [ ] Integration test: feed known audio, get expected transcript

### Phase 4: RawTranscript + LLM PostProcessor (`module-4`, `module-5`) ✅

**Goal:** Transcript accumulator + LLM cleanup pipeline.

- [x] `src-tauri/src/raw_transcript.rs` — `sentences: Vec<String>`, `push_words`, `clear`, `to_joined_string`
- [x] `src-tauri/src/llm_postprocessor.rs` — GgufModelBuilder + BlockingModel, real inference via model.chat()
- [x] Wire real mistralrs model
- [ ] Model bundled at `src-tauri/models/qwen-0.6b/` (download Qwen2.5-0.5B GGUF)
- [x] Prompt template for filler removal + punctuation + formatting
- [x] Unit tests: fallback join, empty transcript, warm_up (5/5 pass)

### Phase 5: Transcriber orchestrator (`module-1`) ✅

**Goal:** Black-box API wiring all internals together.

- [x] `src-tauri/src/transcriber.rs` — `new(config)`, `start_transcribing()`, `stop_transcribing()`, `is_transcribing()`
- [x] Spawns AudioDevice thread per session (AudioDevice created inside thread for Send safety)
- [x] Pulls `HeapCons<f32>`, sends to STT thread via channel
- [x] `on_transcribe` callback writes to `Mutex<RawTranscript>`
- [x] `stop_transcribing` sets `is_recording=false`, joins threads, runs LLM, returns cleaned text
- [x] STT warm thread with channel-based command loop
- [x] Drop impl: shutdown signal + join
- [ ] Integration test: full start → speak → stop → transcript (needs real hardware)

### Phase 6: Text Injector (`inject`) ✅

**Goal:** Clean text appears in the focused application.

- [x] `src-tauri/src/inject.rs` — enigo-based text injection (CGEvent on macOS, X11 on Linux)
- [x] `type_text(&self, text: &str) -> Result<(), InjectError>`
- [x] `check_accessibility()` stub for macOS permission check
- [x] Wired in event_handler: stop → inject.type_text(transcript)
- [ ] Manual test: dictation appears in TextEdit / VS Code

### Phase 7: Overlay HUD (`module-6` feedback) ✅

**Goal:** User sees recording state and partial text.

- [x] React overlay: states (idle→hidden, recording→red pulse, processing→spinner)
- [ ] Show partial transcript while recording (needs streaming partial results from STT)
- [x] Tauri events: `dictation-state` emitted from event_handler
- [x] Auto-hide after injection completes (state → idle → null render)

### Phase 8: Auto-start + Polish ✅

**Goal:** App starts on login, production-ready.

- [x] macOS: LaunchAgent plist (docs/AUTOSTART.md)
- [x] Linux: desktop entry in ~/.config/autostart/ (docs/AUTOSTART.md)
- [x] Tray icon with "Toggle Dictation" + "Quit"
- [x] Config file (`~/.config/talksify/config.toml`)
- [x] Window show/hide on state (visible when recording/processing, hidden when idle)
- [ ] Error notifications (toast or overlay message)

## Crate Dependencies (to add in Phase 2-5)

```toml
[dependencies]
cpal = "0.15"                    # audio capture
rtrb = "0.3"                     # lock-free ring buffer (HeapProd/HeapCons)
sherpa-onnx = "1.x"              # Parakeet STT (check crate name)
mistralrs = "0.x"                # LLM inference
enigo = "0.2"                    # cross-platform input simulation (or core-graphics + xdotool)
```

## Conventions

- All code follows `.ast-grep/rules/` and clippy lints in `Cargo.toml`
- No `unwrap()` / `expect()` outside `#[cfg(test)]`
- No ad-hoc `thread::spawn` — all threading goes through Transcriber
- Comments: max 3 lines, no TODO/FIXME
- Files: max 600 lines
- Run `pnpm lint` and `cargo clippy` before committing

## Model Bundling

Models are downloaded once during development and committed/bundled:

```
src-tauri/models/
├── parakeet/          # Nvidia Parakeet ONNX model (sherpa-onnx format)
│   ├── model.onnx
│   └── tokens.txt
└── qwen-0.6b/         # Qwen 0.6B GGUF (mistral.rs format)
    └── model.gguf
```

`tauri.conf.json` bundles `models/*` as app resources.
At runtime, resolve via `tauri::Manager::path().resource_dir()`.
