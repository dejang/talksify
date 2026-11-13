use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{RecvTimeoutError, channel};
use std::sync::{Arc, Mutex};
use std::thread::Builder;
use std::thread::JoinHandle;

use crate::audio_device::{AudioDevice, AudioProcessor, HeapCons, HeapProd, InputFormat, LevelMeter};
use crate::audio_source::AudioSource;
use crate::llm_postprocessor::LlmPostProcessor;
use crate::parakeet_stt::ParakeetStt;
use crate::raw_transcript::{RawTranscript, Transcript};
use crate::recording;
use crate::stt::Stt;

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

enum SttCommand {
    /// Audio for the recording that is starting, replacing the previous recording's buffer.
    Audio(HeapCons<f32>),
    Flush(std::sync::mpsc::Sender<()>),
    Shutdown,
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
        let mut stt = ParakeetStt::new(model_dir)
            .ok_or_else(|| TranscriberError::Init("failed to load STT model".into()))?;
        let stt_rate = stt.sample_rate();

        let mut llm = LlmPostProcessor::new(model_dir);
        llm.warm_up();
        if !llm.is_ready() {
            log::warn!("[transcriber] LLM unavailable; transcripts get basic cleanup only");
        }

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
        let shutdown_for_thread = Arc::clone(&shutdown);

        let stt_handle = Builder::new()
            .name("stt-warm".into())
            .spawn(move || {
                stt_warm_loop(stt, &stt_rx, &shutdown_for_thread);
            })
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
    pub fn start_transcribing(&mut self, save_audio_to: Option<PathBuf>) -> Result<(), TranscriberError> {
        if self.is_recording.load(Ordering::Relaxed) {
            return Err(TranscriberError::AlreadyRunning);
        }

        log::info!("[transcriber] starting");
        // 10 s of audio, so the STT thread can fall behind during a long transcription.
        let (mut audio_tx, audio_rx) = rtrb::RingBuffer::new(self.stt_rate as usize * 10);
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
                let mut rec = save_audio_to.and_then(|path| spawn_writer(path, format));

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
                            if let Some((tx, _, rec_dropped)) = &mut rec {
                                *rec_dropped += send_audio(tx, &buf[..count]);
                            }
                            let audio = processor.process(&buf[..count]);
                            if let Some(level) = meter.push(audio) {
                                on_level(level);
                            }
                            if last_log.elapsed().as_secs() >= 2 {
                                log::info!("[audio] {total_samples} samples captured");
                                last_log = std::time::Instant::now();
                            }
                            dropped += send_audio(&mut audio_tx, audio);
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
                if let Some((tx, _, rec_dropped)) = &mut rec {
                    *rec_dropped += send_audio(tx, &remaining);
                }
                for block in remaining.chunks(buf.len()) {
                    dropped += send_audio(&mut audio_tx, processor.process(block));
                }
                dropped += send_audio(&mut audio_tx, processor.finish());
                if let Some((tx, writer, n)) = rec {
                    finish_writer(tx, writer, n);
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

    /// # Errors
    /// Returns an error if idle or flushing the transcript fails.
    pub fn stop_transcribing(&mut self) -> Result<Transcript, TranscriberError> {
        if !self.is_recording.load(Ordering::Relaxed) {
            return Err(TranscriberError::NotRunning);
        }

        log::info!("[transcriber] stopping");
        self.is_recording.store(false, Ordering::Relaxed);

        // 1. Join audio thread (drains ring buffer, sends last samples to STT)
        if let Some(handle) = self.audio_handle.take() {
            let _ = handle.join();
        }

        // 2. Send Flush with reply channel — STT drains the audio ring buffer (complete, since
        //    the audio thread has exited), flushes, then signals done. Blocking recv, no timeout.
        let (done_tx, done_rx) = channel::<()>();
        self.stt_tx
            .send(SttCommand::Flush(done_tx))
            .map_err(|e| TranscriberError::Stop(format!("stt send: {e}")))?;
        let _ = done_rx.recv();

        // 3. All STT work is done. Read transcript.
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
        let cleaned = self.llm.clean_transcript(&raw);
        log::info!("[transcriber] cleaned: {cleaned:?}");
        Ok(Transcript {
            raw: raw.to_joined_string(),
            cleaned,
        })
    }

    #[must_use]
    pub fn is_transcribing(&self) -> bool {
        self.is_recording.load(Ordering::Relaxed)
    }
}

impl<A: AudioSource> Drop for Transcriber<A> {
    fn drop(&mut self) {
        let _ = self.stt_tx.send(SttCommand::Shutdown);
        if let Some(handle) = self.stt_handle.take() {
            let _ = handle.join();
        }
    }
}

/// Pushes `audio` into the STT thread's ring buffer; returns how many samples did not fit.
fn send_audio(audio_tx: &mut HeapProd<f32>, audio: &[f32]) -> usize {
    let fits = audio_tx.slots().min(audio.len());
    if let Ok(chunk) = audio_tx.write_chunk_uninit(fits) {
        chunk.fill_from_iter(audio.iter().copied());
    }
    audio.len() - fits
}

fn spawn_writer(path: PathBuf, fmt: InputFormat) -> Option<(HeapProd<f32>, JoinHandle<()>, usize)> {
    let cap = fmt.sample_rate as usize * usize::from(fmt.channels) * 10;
    let (tx, rx) = rtrb::RingBuffer::new(cap);
    Builder::new()
        .name("audio-writer".into())
        .spawn(move || recording::write_wav(path, fmt, rx))
        .inspect_err(|e| log::error!("[recording] writer thread: {e}"))
        .ok()
        .map(|writer| (tx, writer, 0))
}

fn finish_writer(tx: HeapProd<f32>, writer: JoinHandle<()>, n: usize) {
    drop(tx);
    let _ = writer.join();
    if n > 0 {
        log::warn!("[recording] dropped {n} samples");
    }
}

fn stt_warm_loop(
    mut stt: ParakeetStt,
    rx: &std::sync::mpsc::Receiver<SttCommand>,
    shutdown: &AtomicBool,
) {
    let mut audio: Option<HeapCons<f32>> = None;
    while !shutdown.load(Ordering::Relaxed) {
        // While recording, wake every 10 ms to pick up new audio; otherwise wait for a command.
        let command = if audio.is_some() {
            rx.recv_timeout(std::time::Duration::from_millis(10))
        } else {
            rx.recv().map_err(RecvTimeoutError::from)
        };
        if let Some(audio_rx) = &mut audio
            && let Ok(chunk) = audio_rx.read_chunk(audio_rx.slots())
        {
            let (first, second) = chunk.as_slices();
            stt.push_audio(first);
            stt.push_audio(second);
            chunk.commit_all();
        }
        match command {
            Ok(SttCommand::Audio(audio_rx)) => {
                audio = Some(audio_rx);
            }
            Ok(SttCommand::Flush(done_tx)) => {
                stt.flush();
                audio = None;
                let _ = done_tx.send(());
            }
            Err(RecvTimeoutError::Timeout) => {}
            Ok(SttCommand::Shutdown) | Err(RecvTimeoutError::Disconnected) => {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::parakeet_stt::tests::{say, synthesize, synthesize_in};
    use crate::{AudioError, AudioSource, HeapCons, InputFormat, Transcriber};
    use cpal::Sample;
    use std::path::Path;
    use std::sync::{Arc, Mutex};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};

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

    // Only supply audio and observe the public lifecycle and final transcript.
    fn test_transcriber() -> (Transcriber<TestSource>, Arc<Mutex<TestSource>>) {
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
        (transcriber, input)
    }

    fn transcribe(
        transcriber: &mut Transcriber<TestSource>,
        input: &Mutex<TestSource>,
        samples: impl Into<Arc<[f32]>>,
        format: InputFormat,
    ) -> String {
        let ready = {
            let mut source = input.lock().unwrap();
            source.samples = samples.into();
            source.format = format;
            source.started.store(false, Ordering::Release);
            Arc::clone(&source.started)
        };
        assert!(transcriber.stop_transcribing().is_err());
        transcriber.start_transcribing(None).unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while !ready.load(Ordering::Acquire) {
            assert!(Instant::now() < deadline, "audio source did not start");
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(transcriber.is_transcribing());
        assert!(transcriber.start_transcribing(None).is_err());
        let text = transcriber.stop_transcribing().unwrap();
        assert!(!transcriber.is_transcribing());
        text.cleaned.to_lowercase()
    }

    #[test]
    #[ignore = "loads the bundled STT and LLM models and uses macOS `say`"]
    fn transcribes_other_languages() {
        let (mut transcriber, input) = test_transcriber();
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
            let out = transcribe(
                &mut transcriber,
                &input,
                synthesize_in(Some(voice), speech),
                InputFormat {
                    sample_rate: 16_000,
                    channels: 1,
                },
            );
            for word in *words {
                assert!(out.contains(word), "{voice}: {out:?}, missing {word:?}");
            }
        }
    }

    #[test]
    #[ignore = "loads the bundled STT and LLM models and uses macOS `say`"]
    fn transcribes_common_device_formats() {
        let (mut transcriber, input) = test_transcriber();
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
            samples.extend(mono.iter().flat_map(|&s| {
                [s, 0.5 * s].into_iter().take(usize::from(channels))
            }));
            let out = transcribe(
                &mut transcriber,
                &input,
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
        let (mut transcriber, input) = test_transcriber();
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
            ("The problem is this. The build takes way too long on CI.", &["this: the build"], &[]),
        ];

        for (speech, must, must_not) in cases {
            let out = transcribe(
                &mut transcriber,
                &input,
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
