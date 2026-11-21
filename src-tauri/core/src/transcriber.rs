use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex};
use std::thread::{Builder, JoinHandle};

use crate::audio_device::{
    AudioDevice, AudioProcessor, HeapProd, InputFormat, LevelMeter, ring_buffer,
};
use crate::audio_source::AudioSource;
use crate::llm_postprocessor::LlmPostProcessor;
use crate::parakeet_stt::ParakeetStt;
use crate::raw_transcript::{RawTranscript, Transcript};
use crate::recording;
use crate::stt::Stt;
use crate::stt_worker::{SttCommand, SttWorker};
use crate::text_cleaner::TextCleaner;

#[derive(Debug, thiserror::Error)]
pub enum TranscriberError {
    #[error("init failed: {0}")]
    Init(String),
    #[error("already transcribing")]
    AlreadyRunning,
    #[error("not transcribing")]
    NotRunning,
    #[error("stop failed: {0}")]
    Stop(String),
}

struct WavRecording {
    tx: HeapProd<f32>,
    writer: JoinHandle<()>,
    dropped: usize,
}

impl WavRecording {
    fn start(path: PathBuf, fmt: InputFormat) -> Option<Self> {
        let cap = fmt.sample_rate as usize * usize::from(fmt.channels) * 10;
        let (tx, rx) = ring_buffer(cap);
        Builder::new()
            .name("audio-writer".into())
            .spawn(move || recording::write_wav(path, fmt, rx))
            .inspect_err(|e| log::error!("[recording] writer thread: {e}"))
            .ok()
            .map(|writer| Self {
                tx,
                writer,
                dropped: 0,
            })
    }

    fn write(&mut self, audio: &[f32]) {
        self.dropped += self.tx.push_slice(audio);
    }

    fn finish(self) {
        drop(self.tx);
        let _ = self.writer.join();
        if self.dropped > 0 {
            log::warn!("[recording] dropped {} samples", self.dropped);
        }
    }
}

pub struct Transcriber<A: AudioSource = AudioDevice> {
    is_recording: Arc<AtomicBool>,
    audio_handle: Option<JoinHandle<()>>,
    stt_handle: Option<JoinHandle<()>>,
    stt_tx: std::sync::mpsc::Sender<SttCommand>,
    raw_transcript: Arc<Mutex<RawTranscript>>,
    llm: LlmPostProcessor,
    make_source: Arc<dyn Fn() -> A + Send + Sync>,
    stt_rate: u32,
    level_meter: LevelMeter,
    on_level: Arc<dyn Fn(f32) + Send + Sync>,
}

impl Transcriber<AudioDevice> {
    /// Builds a transcriber using the selected microphone.
    /// # Errors
    /// Returns an initialization error if model loading or the STT thread fails.
    pub fn with_default(
        model_dir: &Path,
        audio_device: Option<String>,
        on_level: Arc<dyn Fn(f32) + Send + Sync>,
    ) -> Result<Self, TranscriberError> {
        Self::new(
            model_dir,
            move || AudioDevice::new(audio_device.clone()),
            on_level,
        )
    }
}

impl<A: AudioSource + 'static> Transcriber<A> {
    /// `on_level` receives loudness (0..=1) about 20 times a second while recording.
    /// # Errors
    /// Returns an initialization error if model loading or the STT thread fails.
    pub fn new(
        model_dir: &Path,
        make_source: impl Fn() -> A + Send + Sync + 'static,
        on_level: Arc<dyn Fn(f32) + Send + Sync>,
    ) -> Result<Self, TranscriberError> {
        let cleaner = TextCleaner::new(model_dir);
        Self::with_cleaner(model_dir, cleaner, make_source, on_level)
    }

    /// Builds a transcriber using a pre-initialized [`TextCleaner`].
    /// # Errors
    /// Returns an initialization error if STT model loading or the STT thread fails.
    pub fn with_cleaner(
        model_dir: &Path,
        cleaner: TextCleaner,
        make_source: impl Fn() -> A + Send + Sync + 'static,
        on_level: Arc<dyn Fn(f32) + Send + Sync>,
    ) -> Result<Self, TranscriberError> {
        let mut stt = ParakeetStt::new(model_dir)
            .ok_or_else(|| TranscriberError::Init("failed to load STT model".into()))?;
        let llm = cleaner.into_llm();
        let stt_rate = stt.sample_rate();
        let raw_transcript = Arc::new(Mutex::new(RawTranscript::default()));

        stt.subscribe(Box::new({
            let transcript = Arc::clone(&raw_transcript);
            move |text: String| {
                if let Ok(mut t) = transcript.lock() {
                    t.push_words(text);
                }
            }
        }));

        let (stt_tx, stt_rx) = channel::<SttCommand>();
        let shutdown = Arc::new(AtomicBool::new(false));
        let worker = SttWorker::new(stt, stt_rx, Arc::clone(&shutdown));

        let stt_handle = Builder::new()
            .name("stt-warm".into())
            .spawn(move || worker.run())
            .map_err(|e| TranscriberError::Init(format!("stt thread: {e}")))?;

        Ok(Self {
            is_recording: Arc::new(AtomicBool::new(false)),
            audio_handle: None,
            stt_handle: Some(stt_handle),
            stt_tx,
            raw_transcript,
            llm,
            make_source: Arc::new(make_source),
            stt_rate,
            level_meter: LevelMeter::new(stt_rate as usize / 20, -55.0, 45.0),
            on_level,
        })
    }

    /// # Errors
    /// Returns an error if already recording or a worker cannot start.
    pub fn start_transcribing(
        &mut self,
        save_audio_to: Option<PathBuf>,
    ) -> Result<(), TranscriberError> {
        if self.is_recording.load(Ordering::Relaxed) {
            return Err(TranscriberError::AlreadyRunning);
        }

        log::info!("[transcriber] starting");
        // 10 s of audio, so the STT thread can fall behind during a long transcription.
        let (mut audio_tx, audio_rx) = ring_buffer(self.stt_rate as usize * 10);
        self.stt_tx
            .send(SttCommand::Audio(audio_rx))
            .map_err(|e| TranscriberError::Init(format!("stt send: {e}")))?;
        self.is_recording.store(true, Ordering::Relaxed);

        let is_recording = Arc::clone(&self.is_recording);
        let make_source = Arc::clone(&self.make_source);
        let stt_rate = self.stt_rate;
        let mut meter = self.level_meter.clone();
        let on_level = Arc::clone(&self.on_level);

        let handle = Builder::new()
            .name("audio-consumer".into())
            .spawn(move || {
                let mut device = make_source();
                let Ok(mut consumer) = device.start_capture() else {
                    log::error!("failed to start audio capture");
                    is_recording.store(false, Ordering::Relaxed);
                    return;
                };
                let mut buf = vec![0f32; 480];
                let format = device.input_format();
                let Some((format, mut processor)) =
                    format.and_then(|f| Some((f, AudioProcessor::new(f, stt_rate, buf.len())?)))
                else {
                    log::error!("cannot process {format:?} into {stt_rate} Hz mono");
                    is_recording.store(false, Ordering::Relaxed);
                    let _ = device.stop_capture();
                    return;
                };
                let mut rec = save_audio_to.and_then(|path| WavRecording::start(path, format));

                let mut total_samples: u64 = 0;
                let mut dropped = 0;
                let mut last_log = std::time::Instant::now();

                while is_recording.load(Ordering::Relaxed) {
                    match consumer.pop() {
                        Ok(sample) => {
                            buf[0] = sample;
                            let mut count = 1;
                            while count < buf.len() {
                                match consumer.pop() {
                                    Ok(s) => {
                                        buf[count] = s;
                                        count += 1;
                                    }
                                    Err(_) => break,
                                }
                            }
                            total_samples += count as u64;
                            if let Some(recording) = &mut rec {
                                recording.write(&buf[..count]);
                            }
                            let audio = processor.process(&buf[..count]);
                            if let Some(level) = meter.push(audio) {
                                on_level(level);
                            }
                            if last_log.elapsed().as_secs() >= 2 {
                                log::info!("[audio] {total_samples} samples captured");
                                last_log = std::time::Instant::now();
                            }
                            dropped += audio_tx.push_slice(audio);
                            if audio_tx.is_abandoned() {
                                break;
                            }
                        }
                        Err(_) => {
                            std::thread::sleep(std::time::Duration::from_millis(1));
                        }
                    }
                }

                // Drain remaining samples from ring buffer
                let mut remaining = Vec::new();
                while let Ok(sample) = consumer.pop() {
                    remaining.push(sample);
                }
                log::info!("[audio] draining {} remaining samples", remaining.len());
                if let Some(recording) = &mut rec {
                    recording.write(&remaining);
                }
                for block in remaining.chunks(buf.len()) {
                    dropped += audio_tx.push_slice(processor.process(block));
                }
                dropped += audio_tx.push_slice(processor.finish());
                if let Some(recording) = rec {
                    recording.finish();
                }
                if dropped > 0 {
                    log::warn!("[audio] STT fell behind, dropped {dropped} samples");
                }
                log::info!("[audio] stopped, {total_samples} total samples");
                let _ = device.stop_capture();
            })
            .map_err(|e| TranscriberError::Init(format!("audio thread: {e}")))?;

        self.audio_handle = Some(handle);
        log::info!("[transcriber] audio thread spawned");
        Ok(())
    }

    fn finish_recording(&mut self, strict: bool) -> Result<Transcript, TranscriberError> {
        if !self.is_recording.load(Ordering::Relaxed) {
            return Err(TranscriberError::NotRunning);
        }

        log::info!("[transcriber] stopping");
        self.is_recording.store(false, Ordering::Relaxed);

        if let Some(handle) = self.audio_handle.take() {
            let _ = handle.join();
        }

        // STT drains the completed ring buffer, flushes, then signals done.
        let (done_tx, done_rx) = channel::<()>();
        self.stt_tx
            .send(SttCommand::Flush(done_tx))
            .map_err(|e| TranscriberError::Stop(format!("stt send: {e}")))?;
        let _ = done_rx.recv();

        let raw = {
            let mut transcript = self
                .raw_transcript
                .lock()
                .map_err(|e| TranscriberError::Stop(format!("lock: {e}")))?;
            let cloned = transcript.clone();
            transcript.clear();
            cloned
        };

        log::info!(
            "[transcriber] raw transcript ({} sentences): {:?}",
            raw.sentences.len(),
            raw.sentences
        );
        let cleaned = if strict {
            self.llm
                .try_clean_transcript(&raw)
                .map_err(TranscriberError::Stop)?
        } else {
            self.llm.clean_transcript(&raw)
        };
        log::info!("[transcriber] cleaned: {cleaned:?}");
        Ok(Transcript {
            raw: raw.to_joined_string(),
            cleaned,
        })
    }

    /// # Errors
    /// Returns an error if idle or flushing the transcript fails.
    pub fn stop_transcribing(&mut self) -> Result<Transcript, TranscriberError> {
        self.finish_recording(false)
    }

    /// Stops recording and propagates local LLM load or inference errors instead of falling back.
    /// # Errors
    /// Returns an error if idle, flushing fails, or LLM cleanup fails.
    pub fn stop_transcribing_strict(&mut self) -> Result<Transcript, TranscriberError> {
        self.finish_recording(true)
    }

    #[must_use]
    pub fn is_transcribing(&self) -> bool {
        self.is_recording.load(Ordering::Relaxed)
    }
}

impl<A: AudioSource> Drop for Transcriber<A> {
    fn drop(&mut self) {
        self.is_recording.store(false, Ordering::Relaxed);
        if let Some(handle) = self.audio_handle.take() {
            let _ = handle.join();
        }
        let _ = self.stt_tx.send(SttCommand::Shutdown);
        if let Some(handle) = self.stt_handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use cpal::Sample;

    use super::*;
    use crate::parakeet_stt::tests::{say, synthesize, synthesize_in};
    use crate::{AudioError, HeapCons};

    #[derive(Clone)]
    struct TestSource {
        samples: Arc<[f32]>,
        format: InputFormat,
        started: Arc<AtomicBool>,
    }

    impl AudioSource for TestSource {
        fn start_capture(&mut self) -> Result<HeapCons<f32>, AudioError> {
            let (mut producer, consumer) = rtrb::RingBuffer::new(self.samples.len());
            for &sample in self.samples.iter() {
                producer.push(sample).unwrap();
            }
            self.started.store(true, Ordering::Release);
            Ok(consumer)
        }

        fn stop_capture(&mut self) -> Result<(), AudioError> {
            Ok(())
        }

        fn input_format(&self) -> Option<InputFormat> {
            Some(self.format)
        }
    }

    struct TranscriberHarness {
        transcriber: Transcriber<TestSource>,
        input: Arc<Mutex<TestSource>>,
    }

    impl TranscriberHarness {
        fn new() -> Self {
            let models = Path::new(env!("CARGO_MANIFEST_DIR")).join("../models");
            let input = Arc::new(Mutex::new(TestSource {
                samples: Arc::from([]),
                format: InputFormat {
                    sample_rate: 16_000,
                    channels: 1,
                },
                started: Arc::new(AtomicBool::new(false)),
            }));
            let source = Arc::clone(&input);
            let transcriber = Transcriber::new(
                &models,
                move || source.lock().unwrap().clone(),
                Arc::new(|_| {}),
            )
            .unwrap();
            Self { transcriber, input }
        }

        fn transcribe(&mut self, samples: impl Into<Arc<[f32]>>, format: InputFormat) -> String {
            let ready = {
                let mut source = self.input.lock().unwrap();
                source.samples = samples.into();
                source.format = format;
                source.started.store(false, Ordering::Release);
                Arc::clone(&source.started)
            };
            assert!(self.transcriber.stop_transcribing().is_err());
            self.transcriber.start_transcribing(None).unwrap();
            let deadline = Instant::now() + Duration::from_secs(5);
            while !ready.load(Ordering::Acquire) {
                assert!(Instant::now() < deadline, "audio source did not start");
                std::thread::sleep(Duration::from_millis(1));
            }
            assert!(self.transcriber.is_transcribing());
            assert!(self.transcriber.start_transcribing(None).is_err());
            let text = self.transcriber.stop_transcribing().unwrap();
            assert!(!self.transcriber.is_transcribing());
            text.cleaned.to_lowercase()
        }
    }

    #[test]
    #[ignore = "loads the bundled STT and LLM models and uses macOS `say`"]
    fn transcribes_other_languages() {
        let mut harness = TranscriberHarness::new();
        let format = InputFormat {
            sample_rate: 16_000,
            channels: 1,
        };
        let cases: &[(&str, &str, &[&str])] = &[
            (
                "Anna",
                "Morgen treffen wir uns um zehn Uhr im Büro.",
                &["morgen", "uns um", "büro"],
            ),
            (
                "Jacques",
                "Demain, nous avons une réunion avec toute l'équipe.",
                &["demain", "équipe"],
            ),
            (
                "Mónica",
                "Mañana enviaré el informe al cliente.",
                &["mañana", "informe"],
            ),
            (
                "Alice",
                "Domani mattina finiamo il progetto.",
                &["domani", "progetto"],
            ),
        ];
        for (voice, speech, words) in cases {
            let out = harness.transcribe(synthesize_in(Some(voice), speech), format);
            for word in *words {
                assert!(out.contains(word), "{voice}: {out:?}, missing {word:?}");
            }
        }
    }

    #[test]
    #[ignore = "loads the bundled STT and LLM models and uses macOS `say`"]
    fn transcribes_common_device_formats() {
        let mut harness = TranscriberHarness::new();
        let mut mono = Vec::new();
        let mut samples = Vec::new();
        for (data_format, sample_rate, channels) in [
            ("LEI16@24000", 24_000, 1),
            ("LEF32@44100", 44_100, 2),
            ("LEI16@48000", 48_000, 2),
        ] {
            let bytes = say(
                "The tests pass locally, but on the build server they time out.",
                data_format,
                None,
            );
            mono.clear();
            if data_format.starts_with("LEI16") {
                mono.extend(
                    bytes
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|b| f32::from_sample(i16::from_le_bytes(*b))),
                );
            } else {
                mono.extend(
                    bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|b| f32::from_le_bytes(*b)),
                );
            }
            samples.clear();
            samples.extend(
                mono.iter()
                    .flat_map(|&s| [s, 0.5 * s].into_iter().take(usize::from(channels))),
            );
            let out = harness.transcribe(
                samples.as_slice(),
                InputFormat {
                    sample_rate,
                    channels,
                },
            );
            assert!(out.contains("build server"), "{data_format}: {out:?}");
        }
    }

    // Input speech and required/forbidden output phrases, including a pause inside a negation.
    #[test]
    #[ignore = "loads the bundled STT, VAD and LLM models and uses macOS `say`"]
    fn speech_to_clean_text() {
        let mut harness = TranscriberHarness::new();
        let cases: &[(&str, &[&str], &[&str])] = &[
            (
                "So the plan is. One, we refactor the parser. Two, we add more tests. \
                 Three, we ship it on Friday.",
                &["1. ", "\n2. ", "\n3. "],
                &[],
            ),
            (
                "There are a few things we need to buy. Milk, eggs, bread, and coffee.",
                &["- milk", "\n- eggs", "\n- bread", "\n- coffee"],
                &[],
            ),
            (
                "So these are small text improvements that are rewrites, but they're not \
                 complete rewrites, if that makes any sense.",
                &["not complete rewrites"],
                &["would be"],
            ),
            (
                "So these are small text improvements that are rewrites, but they're not \
                 [[slnc 800]] complete rewrites, if that makes any sense.",
                &["not complete rewrites"],
                &["would be"],
            ),
            (
                "Um, I think we should, uh, push the release to Monday.",
                &["i think we should push the release to monday"],
                &[],
            ),
            (
                "The problem is this. The build takes way too long on CI.",
                &["this: the build"],
                &[],
            ),
        ];

        for (speech, must, must_not) in cases {
            let out = harness.transcribe(
                synthesize(speech),
                InputFormat {
                    sample_rate: 16_000,
                    channels: 1,
                },
            );
            for phrase in *must {
                assert!(
                    out.contains(phrase),
                    "{speech:?} -> {out:?}, missing {phrase:?}"
                );
            }
            for phrase in *must_not {
                assert!(
                    !out.contains(phrase),
                    "{speech:?} -> {out:?}, unwanted {phrase:?}"
                );
            }
        }
    }
}
