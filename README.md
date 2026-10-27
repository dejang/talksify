# Talksify

Local voice dictation. Press a hotkey, talk, press it again — clean, correctly punctuated
text is typed into whatever app has focus. Transcription and cleanup both run on your
machine; nothing is sent to the cloud.

```
talk ──► mic (cpal) ──► Silero VAD ──► Parakeet TDT 0.6B (sherpa-onnx)
                                              │  raw transcript
                                              ▼
     focused app ◄── keystrokes / paste ◄── Qwen3-0.6B (mistral.rs) + rule-based formatter
```

## What it does

- **Push-to-toggle dictation** — one global hotkey starts and stops a recording.
- **Speech-to-text** — Nvidia Parakeet TDT 0.6B v3 (int8) via sherpa-onnx, with Silero VAD
  cutting audio at pauses. Each VAD segment is back-dated by a 600 ms pre-roll so the first
  syllable after a pause is not clipped.
- **LLM cleanup** — Qwen3-0.6B (GGUF, Metal) re-joins sentences split at pauses, fixes
  punctuation and capitalisation, and rewrites self-corrections ("no, wait, send it Tuesday")
  keeping only the last version.
- **Rule-based formatting** around the LLM — filler removal (`um`, `uh`, `erm`) before it,
  numbered and bulleted list layout after it, so the LLM never has to do mechanical work.
- **Text injection** — the result lands in the focused app either as synthetic keystrokes
  (CGEvent, one key pair per character) or as a clipboard paste, whichever the target accepts.
  See [Delivery](#delivery-how-text-gets-into-your-app).
- **Overlay HUD** — a small transparent pill at the bottom of the screen: a live mic-level
  meter while recording, a spinner while processing, hidden when idle.
- **Tray icon** — enable/disable the hotkey, toggle start-at-login, quit.

If the LLM fails to load, dictation still works: the transcript falls back to basic
rule-based cleanup instead of failing.

## Requirements

- **macOS** on Apple Silicon. `sherpa-onnx` (STT/VAD) and `mistralrs` with the `metal`
  feature (LLM) are declared as macOS-only dependencies; the Linux injection path exists in
  `inject.rs`, but the model layers are not ported yet.
- Rust (edition 2024 toolchain), Node.js, pnpm.
- ~1.2 GB free disk for the models.
- macOS permissions: **Microphone** and **Accessibility** (needed to post keystrokes).

## Quick start

```bash
pnpm install
./scripts/download-models.sh   # one-time, ~1.2 GB
pnpm tauri dev                 # run from the repo root
```

On first run macOS asks for microphone access. Text injection additionally needs the app in
**System Settings → Privacy & Security → Accessibility** — until it is trusted, injection is
refused and logged (`process is not trusted; ...`), because macOS drops synthetic events from
untrusted processes silently.

Production build:

```bash
pnpm tauri build
```

`tauri.conf.json` pins a `signingIdentity` for this author's machine; change or remove it to
build elsewhere.

## Usage

1. Press the hotkey (default **`Ctrl+Alt+Space`**) — the overlay appears with a level meter.
2. Speak. Pauses end a segment; the meter shows the mic is hearing you.
3. Press the hotkey again — the overlay shows *Processing…*, then the cleaned text is typed
   into the focused app and the overlay disappears.

An empty transcript is skipped rather than injected. Pressing the hotkey twice in quick
succession is safe: a second start while recording is rejected with `AlreadyRunning`.

## Delivery: how text gets into your app

Apps accept synthetic input very differently, and a dropped keystroke is never reported —
`CGEventPost` succeeds whether or not the app used the event. So the injector decides per call
and then checks its own work, instead of keeping a list of apps:

1. Read the focused element's text through Accessibility, from the frontmost app by pid.
2. **Readable, and no newlines in the text** → type it as one key-down/key-up pair per
   character, then read the element again. Changed → done. Byte-identical → the app ignored the
   keystrokes → paste instead.
3. **Not readable, or the text contains newlines** → paste: write the clipboard, send `Cmd+V`,
   wait for the target to read it, then put the old clipboard contents back.

Per character matters. A whole-string payload on virtual keycode 0 is not a keystroke, so apps
that parse key events themselves (Zed, terminals) discard it, while a single character is
handled like real typing.

Newlines are always pasted: typed, they send the message in chat apps and run the command in a
terminal. Paste also survives apps that disable synthetic keys.

## Troubleshooting

Run the read-back probe with the problem app frontmost to see what the injector can see:

```bash
cd src-tauri && cargo test --lib inject::target -- --ignored --nocapture
```

| Output | Meaning |
|---|---|
| `accessibility trusted: false` | Grants are missing: **System Settings → Privacy & Security → Accessibility** |
| `unreadable` with `-25211` | Same: this process is not trusted for Accessibility |
| `unreadable` with `-25212` | The app exposes no Accessibility tree (Chromium and Firefox build one only when asked). Dictation pastes there, so text still arrives |
| `unreadable` with `-25204` | No focused element to read; the app has nothing text-shaped selected |
| `unreadable` with `-25205` | The focused element supports no value at all — a button or list row, not a text field |
| A value is printed | Typing is used and verified in that app |

Other cases worth knowing:

- **Nothing appears in any app**: either the permission was revoked, or secure input mode is on
  — a password field, or a password manager holding it, makes the window server discard every
  synthetic event for every app. There is no way to detect it from here.
- **The clipboard briefly holds the transcript.** The paste route rewrites it and restores what
  was there, but clipboard managers will still record the entry.
- **Paste is not identical to typing** in editors that auto-indent or reformat on paste.

## Configuration

Config file (created by you, not by the app):

| Platform | Path |
|---|---|
| macOS | `~/Library/Application Support/talksify/config.toml` |
| Linux | `~/.config/talksify/config.toml` |

```toml
# Global shortcut. Omit for the default: "Ctrl+Alt+Space".
hotkey = "Ctrl+Alt+Space"

# Optional: name of the input device, as reported by the audio host.
# Omit to use the system default input.
# audio_device = "MacBook Pro Microphone"
```

A missing or invalid file falls back to defaults. `model_dir` is always resolved from the
app's bundled resources at runtime, so setting it in the file has no effect.

Auto-start on login is covered in [`docs/AUTOSTART.md`](docs/AUTOSTART.md) (LaunchAgent on
macOS, autostart desktop entry on Linux); the tray's **Start at Login** item toggles it too.

## Models

Bundled as app resources, never downloaded at runtime. `scripts/download-models.sh` fetches
them once during development and skips files already present at the expected size.

| Path | Model | Size |
|---|---|---|
| `src-tauri/models/stt/` | Parakeet TDT 0.6B v3 int8 (encoder/decoder/joiner + `tokens.txt`), 25 European languages | ~490 MB |
| `src-tauri/models/vad/` | Silero VAD | ~630 KB |
| `src-tauri/models/qwen-0.6b/` | Qwen3-0.6B GGUF `Q8_0` | ~610 MB |

Model binaries are gitignored; clone the repo and re-run the script.

## Architecture

The board in [`docs/design.json`](docs/design.json) is the source of truth for
responsibilities, operations, state and failure modes; [`docs/PLAN.md`](docs/PLAN.md) is the
implementation plan keyed to its part IDs.

| File | Board part | Role |
|---|---|---|
| `src-tauri/src/lib.rs` | `tauri-app` | Entry point: plugins, tray, hotkey registration, overlay window |
| `src-tauri/src/event_handler.rs` | `module-6` | Thin glue: hotkey → Transcriber → Inject, overlay state |
| `src-tauri/src/transcriber.rs` | `module-1` | Orchestrator; owns the audio and STT worker threads |
| `src-tauri/src/audio_device.rs` | `module-2` | cpal capture, resample to 16 kHz, level meter, lock-free ring buffer |
| `src-tauri/src/stt.rs` | `contract-1` | `Stt` trait: `push_audio`, `subscribe`, `flush` |
| `src-tauri/src/parakeet_stt.rs` | `module-3` | Silero VAD + sherpa-onnx transducer recognizer |
| `src-tauri/src/raw_transcript.rs` | `module-4` | Transcript accumulator (`Vec<String>` of sentences) |
| `src-tauri/src/llm_postprocessor.rs` | `module-5` | mistral.rs `BlockingModel`, clean/correction prompts, warm-up |
| `src-tauri/src/text_format.rs` | `module-5` | Rule-based fillers, counting, bullets |
| `src-tauri/src/inject/` | `inject` | Delivery: per-character CGEvents, clipboard paste, Accessibility read-back |
| `src-tauri/src/config.rs` | `config` | TOML settings store |
| `src/` | HUD | React overlay: state pill + mic-level meter |

**Threading.** `Transcriber::new` starts one long-lived warm STT thread and warms the LLM;
each recording spawns an audio thread that pushes samples into a 10-second `rtrb` ring buffer
(the STT thread can fall behind without blocking capture). `is_recording: AtomicBool` is the
kill switch all loops check.

**Stop sequence:** flip the switch → join the audio thread (it drains the ring buffer) →
send `Flush` and block on its reply channel → clone and clear the raw transcript → run LLM
cleanup → return the text.

## Development

```bash
pnpm lint                 # oxlint + ast-grep rules in .ast-grep/rules
cd src-tauri && cargo clippy
cd src-tauri && cargo test            # unit tests, no models needed
cd src-tauri && cargo test --ignored -- --nocapture   # end-to-end, needs models + macOS `say`
```

The `#[ignore]`d tests speak sentences with `say`, run them through the real STT and LLM, and
assert on the result — including a multi-language check (de/fr/es/it) and an over-cut check
that fails if cleanup drops words the speaker actually said.

Conventions, enforced by clippy and `.ast-grep/rules/`:

- No `unwrap()` / `expect()` outside `#[cfg(test)]`
- No ad-hoc `thread::spawn` — threading goes through `Transcriber`
- No debug prints — use the `log` macros or propagate the error
- No `#[allow]`/`#[expect]` clippy suppressions on modules, impls or functions, and never on
  `dead_code`/`unused` (delete the code instead)
- No lint-disable directives in the frontend either
- Comments max 3 lines, no `TODO`/`FIXME` narrative
- Files max 600 lines

Both `pnpm lint` and `cargo clippy` must pass before a commit.

## Status & limitations

- macOS only in practice (see Requirements).
- Partial transcript is not shown in the overlay while recording — only the level meter.
- Errors are logged, not surfaced in the UI yet.
- The overlay hides itself when idle; there is no toast/notification channel.

## License

Not specified yet.
