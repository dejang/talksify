use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{RecvTimeoutError, channel};
use std::sync::{Arc, Mutex};
use std::thread::Builder;
use std::thread::JoinHandle;

use crate::audio_device::{AudioDevice, AudioProcessor, HeapCons, HeapProd, LevelMeter};
use crate::config::Config;
use crate::llm_postprocessor::LlmPostProcessor;
use crate::parakeet_stt::ParakeetStt;
use crate::raw_transcript::RawTranscript;
use crate::stt::Stt;

#[derive(Debug, thiserror::Error)]
pub(crate) enum TranscriberError {
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

pub(crate) struct Transcriber {
    is_recording: Arc<AtomicBool>,
    audio_handle: Option<JoinHandle<()>>,
    stt_handle: Option<JoinHandle<()>>,
    stt_tx: std::sync::mpsc::Sender<SttCommand>,
    raw_transcript: Arc<Mutex<RawTranscript>>,
    llm: LlmPostProcessor,
    audio_device_name: Option<String>,
    stt_rate: u32,
    level_meter: LevelMeter,
    on_level: Arc<dyn Fn(f32) + Send + Sync>,
}

impl Transcriber {
    /// `on_level` receives the microphone loudness (0..=1) about 20 times a second while recording.
    pub(crate) fn new(
        config: &Config,
        on_level: Arc<dyn Fn(f32) + Send + Sync>,
    ) -> Result<Self, TranscriberError> {
        let mut stt = ParakeetStt::new(&config.model_dir)
            .ok_or_else(|| TranscriberError::Init("failed to load STT model".into()))?;
        let stt_rate = stt.sample_rate();

        let mut llm = LlmPostProcessor::new(&config.model_dir);
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
            audio_device_name: config.audio_device.clone(),
            stt_rate,
            level_meter: LevelMeter::new(stt_rate as usize / 20, -55.0, 45.0),
            on_level,
        })
    }

    pub(crate) fn start_transcribing(&mut self) -> Result<(), TranscriberError> {
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
        let device_name = self.audio_device_name.clone();
        let stt_rate = self.stt_rate;
        let mut meter = self.level_meter.clone();
        let on_level = Arc::clone(&self.on_level);

        let handle = Builder::new()
            .name("audio-consumer".into())
            .spawn(move || {
                let mut device = AudioDevice::new(device_name);
                let Ok(mut consumer) = device.start_capture() else {
                    log::error!("failed to start audio capture");
                    is_recording.store(false, Ordering::Relaxed);
                    return;
                };
                let mut buf = vec![0f32; 480];
                let format = device.input_format();
                let Some(mut processor) =
                    format.and_then(|f| AudioProcessor::new(f, stt_rate, buf.len()))
                else {
                    log::error!("cannot process {format:?} into {stt_rate} Hz mono");
                    is_recording.store(false, Ordering::Relaxed);
                    let _ = device.stop_capture();
                    return;
                };

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
                for block in remaining.chunks(buf.len()) {
                    dropped += send_audio(&mut audio_tx, processor.process(block));
                }
                dropped += send_audio(&mut audio_tx, processor.finish());
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

    pub(crate) fn stop_transcribing(&mut self) -> Result<String, TranscriberError> {
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
            "[transcriber] raw transcript: {} sentences",
            raw.sentences.len()
        );
        let cleaned = self.llm.clean_transcript(&raw);
        log::trace!("[transcriber] cleaned: {cleaned:?}");
        Ok(cleaned)
    }

    pub(crate) fn is_transcribing(&self) -> bool {
        self.is_recording.load(Ordering::Relaxed)
    }
}

impl Drop for Transcriber {
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
    use super::*;
    use crate::audio_device::InputFormat;
    use crate::llm_postprocessor::tests::WordCheck;
    use crate::parakeet_stt::tests::{say, synthesize, synthesize_in};
    use cpal::Sample;
    use std::path::Path;

    /// Transcribes speech captured like a device in `data_format` (`LEI16@…` or `LEF32@…`),
    /// scaled as the audio callback does, with a quieter copy in the right channel when stereo.
    fn transcribe_as_device(stt: &mut ParakeetStt, data_format: &str, format: InputFormat) -> String {
        let bytes = say(
            "The tests pass locally, but on the build server they time out.",
            data_format,
            None,
        );
        let mono: Vec<f32> = if data_format.starts_with("LEI16") {
            let (samples, _) = bytes.as_chunks::<2>();
            samples.iter().map(|b| f32::from_sample(i16::from_le_bytes(*b))).collect()
        } else {
            let (samples, _) = bytes.as_chunks::<4>();
            samples.iter().map(|b| f32::from_le_bytes(*b)).collect()
        };
        let interleaved: Vec<f32> = if format.channels == 2 {
            mono.iter().flat_map(|&s| [s, 0.5 * s]).collect()
        } else {
            mono
        };

        let text = Arc::new(Mutex::new(Vec::new()));
        stt.subscribe(Box::new({
            let text = Arc::clone(&text);
            move |chunk: String| text.lock().unwrap().push(chunk)
        }));
        let block = 512 * usize::from(format.channels);
        let mut processor = AudioProcessor::new(format, stt.sample_rate(), block).unwrap();
        let mut processed = Vec::new();
        for callback in interleaved.chunks(block) {
            processed.extend_from_slice(processor.process(callback));
        }
        processed.extend_from_slice(processor.finish());
        stt.push_audio(&processed);
        stt.flush();
        let text = text.lock().unwrap().join(" ");
        println!("{data_format} {format:?} -> {text:?}");
        text.to_lowercase()
    }

    #[test]
    fn send_audio_reports_samples_that_do_not_fit() {
        let (mut audio_tx, mut audio_rx) = rtrb::RingBuffer::new(4);
        let (first, second) = ([0.1, 0.2, 0.3], [0.4, 0.5, 0.6]);
        assert_eq!(send_audio(&mut audio_tx, &first), 0);
        assert_eq!(send_audio(&mut audio_tx, &second), 2);
        assert_eq!(audio_rx.slots(), 4);
        assert!(audio_rx.pop().is_ok_and(|s| (s - 0.1).abs() < 1e-6));
    }

    #[test]
    #[ignore = "loads the bundled STT models and uses macOS `say`"]
    fn stt_thread_transcribes_ring_buffer_audio() {
        let models = Path::new(env!("CARGO_MANIFEST_DIR")).join("models");
        let mut stt = ParakeetStt::new(&models).unwrap();
        let text = Arc::new(Mutex::new(Vec::new()));
        stt.subscribe(Box::new({
            let text = Arc::clone(&text);
            move |chunk: String| text.lock().unwrap().push(chunk)
        }));
        let (stt_tx, stt_rx) = channel::<SttCommand>();
        let shutdown = Arc::new(AtomicBool::new(false));
        let handle = Builder::new()
            .spawn({
                let shutdown = Arc::clone(&shutdown);
                move || stt_warm_loop(stt, &stt_rx, &shutdown)
            })
            .unwrap();

        let audio = synthesize("The tests pass locally, but on the build server they time out.");
        let (mut audio_tx, audio_rx) = rtrb::RingBuffer::new(audio.len());
        stt_tx.send(SttCommand::Audio(audio_rx)).unwrap();
        for block in audio.chunks(160) {
            assert_eq!(send_audio(&mut audio_tx, block), 0);
        }
        let (done_tx, done_rx) = channel::<()>();
        stt_tx.send(SttCommand::Flush(done_tx)).unwrap();
        done_rx.recv().unwrap();
        stt_tx.send(SttCommand::Shutdown).unwrap();
        handle.join().unwrap();

        let text = text.lock().unwrap().join(" ");
        assert!(text.to_lowercase().contains("build server"), "{text:?}");
    }

    // Speaks `text` in a `say` voice, transcribes and cleans it, and describes what went wrong:
    // key `words` missing from the STT or the cleaned text, or words the LLM dropped.
    fn check_language(
        stt: &mut ParakeetStt,
        raw: &Mutex<RawTranscript>,
        llm: &LlmPostProcessor,
        (voice, text, words): (&str, &str, &[&str]),
    ) -> Option<String> {
        let audio = synthesize_in(Some(voice), text);
        for block in audio.chunks(160) {
            stt.push_audio(block);
        }
        stt.flush();
        let transcript = std::mem::take(&mut *raw.lock().unwrap());
        let heard = transcript.to_joined_string();
        let out = llm.clean_transcript(&transcript);
        println!("{voice}: SAID {text:?}\n  STT {heard:?}\n  OUT {out:?}\n");
        let (heard_lower, out_lower) = (heard.to_lowercase(), out.to_lowercase());
        let missing: Vec<&&str> = words.iter().filter(|w| !heard_lower.contains(**w)).collect();
        let lost: Vec<&&str> = words.iter().filter(|w| !out_lower.contains(**w)).collect();
        let mut check = WordCheck::default();
        let over_cut = check.over_cut(&heard, &out);
        (!missing.is_empty() || !lost.is_empty() || over_cut).then(|| {
            format!(
                "{voice}: STT missing {missing:?}, cleanup lost {lost:?}, LLM dropped {:?}",
                check.dropped
            )
        })
    }

    #[test]
    #[ignore = "loads the bundled STT and LLM models and uses macOS `say`"]
    fn transcribes_other_languages() {
        let models = Path::new(env!("CARGO_MANIFEST_DIR")).join("models");
        let mut stt = ParakeetStt::new(&models).unwrap();
        let raw = Arc::new(Mutex::new(RawTranscript::default()));
        stt.subscribe(Box::new({
            let raw = Arc::clone(&raw);
            move |text: String| raw.lock().unwrap().push_words(text)
        }));
        let mut llm = LlmPostProcessor::new(&models);
        llm.warm_up();

        let failures: Vec<String> = [
            check_language(&mut stt, &raw, &llm, (
                "Anna",
                "Morgen treffen wir uns um zehn Uhr im Büro.",
                &["morgen", "uns um", "büro"],
            )),
            check_language(&mut stt, &raw, &llm, (
                "Jacques",
                "Demain, nous avons une réunion avec toute l'équipe.",
                &["demain", "équipe"],
            )),
            check_language(&mut stt, &raw, &llm, (
                "Mónica",
                "Mañana enviaré el informe al cliente.",
                &["mañana", "informe"],
            )),
            check_language(&mut stt, &raw, &llm, (
                "Alice",
                "Domani mattina finiamo il progetto.",
                &["domani", "progetto"],
            )),
        ]
        .into_iter()
        .flatten()
        .collect();
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    #[test]
    #[ignore = "loads the bundled STT models and uses macOS `say`"]
    fn transcribes_common_device_formats() {
        let models = Path::new(env!("CARGO_MANIFEST_DIR")).join("models");
        let mut stt = ParakeetStt::new(&models).unwrap();
        let mono_24k = InputFormat { sample_rate: 24_000, channels: 1 };
        let stereo_44k = InputFormat { sample_rate: 44_100, channels: 2 };
        let stereo_48k = InputFormat { sample_rate: 48_000, channels: 2 };
        assert!(transcribe_as_device(&mut stt, "LEI16@24000", mono_24k).contains("build server"));
        assert!(transcribe_as_device(&mut stt, "LEF32@44100", stereo_44k).contains("build server"));
        assert!(transcribe_as_device(&mut stt, "LEI16@48000", stereo_48k).contains("build server"));
    }

    // Each case: text spoken by `say` ([[slnc N]] = N ms of silence), phrases the output must and
    // must not contain, case-insensitive. Without a take-back, at most 10% of words may be dropped.
    #[test]
    #[ignore = "loads the bundled STT, VAD and LLM models and uses macOS `say`"]
    fn speech_to_clean_text() {
        let models = Path::new(env!("CARGO_MANIFEST_DIR")).join("models");
        let mut stt = ParakeetStt::new(&models).unwrap();
        let raw = Arc::new(Mutex::new(RawTranscript::default()));
        stt.subscribe(Box::new({
            let raw = Arc::clone(&raw);
            move |text: String| raw.lock().unwrap().push_words(text)
        }));
        let mut llm = LlmPostProcessor::new(&models);
        llm.warm_up();
        assert!(llm.is_ready(), "LLM failed to load");

        let cases: &[(&str, &[&str], &[&str])] = &[
            (
                "Let's have dinner tomorrow. Oh no, no, no, wait, I have something else. \
                 Let's have Thursday dinner.",
                &["thursday"],
                &["tomorrow", "wait"],
            ),
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
            ("Call me at five. Actually, make it six.", &["six"], &["five"]),
        ];

        let mut failures = Vec::new();
        let mut missing: Vec<&&str> = Vec::new();
        let mut present: Vec<&&str> = Vec::new();
        let mut check = WordCheck::default();
        for (speech, must, must_not) in cases {
            let audio = synthesize(speech);
            for block in audio.chunks(480) {
                stt.push_audio(block);
            }
            let start = std::time::Instant::now();
            stt.flush();
            let transcript = std::mem::take(&mut *raw.lock().unwrap());
            let out = llm.clean_transcript(&transcript);
            let elapsed = start.elapsed();
            let over_cut = check.over_cut(&transcript.to_joined_string(), &out);
            println!(
                "SAID: {speech}\nSTT:  {:?}\nOUT:  {out}\n({elapsed:?} after speech ended, dropped {:?})\n",
                transcript.sentences, check.dropped
            );
            let lower = out.to_lowercase();
            missing.clear();
            missing.extend(must.iter().filter(|m| !lower.contains(**m)));
            present.clear();
            present.extend(must_not.iter().filter(|m| lower.contains(**m)));
            if !missing.is_empty() || !present.is_empty() || over_cut {
                failures.push(format!(
                    "{out:?}\n  missing {missing:?}\n  unwanted {present:?}\n  dropped {}/{} {:?}",
                    check.dropped.len(),
                    check.checked,
                    check.dropped
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "{} of {} cases failed:\n{}",
            failures.len(),
            cases.len(),
            failures.join("\n")
        );
    }
}
