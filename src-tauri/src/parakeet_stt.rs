use std::sync::Arc;

use sherpa_onnx::{
    OfflineModelConfig, OfflineRecognizer, OfflineRecognizerConfig,
    OfflineTransducerModelConfig, SileroVadModelConfig, VadModelConfig, VoiceActivityDetector,
};

use crate::stt::Stt;

pub(crate) struct ParakeetStt {
    recognizer: Arc<OfflineRecognizer>,
    vad: VoiceActivityDetector,
    scratch_buffer: Vec<f32>,
    vad_window: usize,
    stt_rate: i32,
    // 16 kHz audio fed to the VAD; `history[0]` is sample `history_start` since the last reset.
    history: Vec<f32>,
    history_start: usize,
    fed: usize,
    last_end: usize,
    pre_roll: usize,
    idle_keep: usize,
    callback: Option<Box<dyn Fn(String) + Send>>,
}

impl ParakeetStt {
    pub(crate) fn new(model_dir: &std::path::Path) -> Option<Self> {
        let stt_dir = model_dir.join("stt");
        let encoder = stt_dir.join("encoder.int8.onnx");
        let decoder = stt_dir.join("decoder.int8.onnx");
        let joiner = stt_dir.join("joiner.int8.onnx");
        let tokens = stt_dir.join("tokens.txt");
        let vad_model = model_dir.join("vad").join("silero_vad.onnx");

        if !encoder.exists() || !tokens.exists() {
            log::error!("parakeet model not found in {}", stt_dir.display());
            return None;
        }
        if !vad_model.exists() {
            log::error!("VAD model not found at {}", vad_model.display());
            return None;
        }

        let start = std::time::Instant::now();
        let model_config = OfflineModelConfig {
            transducer: OfflineTransducerModelConfig {
                encoder: Some(encoder.to_string_lossy().to_string()),
                decoder: Some(decoder.to_string_lossy().to_string()),
                joiner: Some(joiner.to_string_lossy().to_string()),
            },
            tokens: Some(tokens.to_string_lossy().to_string()),
            num_threads: 4,
            provider: Some("cpu".to_string()),
            ..Default::default()
        };

        let config = OfflineRecognizerConfig {
            model_config,
            decoding_method: Some("greedy_search".to_string()),
            ..Default::default()
        };

        let recognizer = OfflineRecognizer::create(&config)?;
        log::info!(
            "parakeet loaded from {} in {:?}",
            stt_dir.display(),
            start.elapsed()
        );

        let stt_rate = 16_000;
        let vad_window = 512;
        let vad_config = VadModelConfig {
            silero_vad: SileroVadModelConfig {
                model: Some(vad_model.to_string_lossy().to_string()),
                threshold: 0.5,
                min_silence_duration: 1.0,
                min_speech_duration: 0.25,
                window_size: vad_window,
                max_speech_duration: 20.0,
            },
            sample_rate: stt_rate,
            num_threads: 1,
            provider: Some("cpu".to_string()),
            ..Default::default()
        };
        let vad = VoiceActivityDetector::create(&vad_config, 30.0)?;
        log::info!("VAD loaded from {}", vad_model.display());

        let samples_per_second = usize::try_from(stt_rate).ok()?;
        Some(Self {
            recognizer: Arc::new(recognizer),
            vad,
            scratch_buffer: Vec::with_capacity(samples_per_second),
            vad_window: usize::try_from(vad_window).ok()?,
            stt_rate,
            history: Vec::new(),
            history_start: 0,
            fed: 0,
            last_end: 0,
            pre_roll: samples_per_second * 6 / 10,
            idle_keep: samples_per_second * 2,
            callback: None,
        })
    }

    /// Rate `push_audio` expects: mono f32 samples at this many Hz.
    pub(crate) fn sample_rate(&self) -> u32 {
        self.stt_rate.unsigned_abs()
    }

    fn transcribe_chunk(&self, chunk: &[f32]) {
        let audio_ms = chunk.len() * 1000 / self.sample_rate() as usize;
        let start = std::time::Instant::now();

        let stream = self.recognizer.create_stream();
        stream.accept_waveform(self.stt_rate, chunk);
        self.recognizer.decode(&stream);

        let elapsed = start.elapsed();
        if let Some(result) = stream.get_result() {
            if result.text.is_empty() {
                log::debug!("[stt] {audio_ms} ms -> silence ({elapsed:?})");
            } else {
                log::info!("[stt] {audio_ms} ms segment transcribed in {elapsed:?}");
                log::trace!("[stt] segment text: {:?}", result.text);
                if let Some(ref cb) = self.callback {
                    cb(result.text);
                }
            }
        }
    }

    fn feed_vad(&mut self) {
        let full = self.scratch_buffer.len() / self.vad_window * self.vad_window;
        for window in self.scratch_buffer[..full].chunks_exact(self.vad_window) {
            self.vad.accept_waveform(window);
        }
        self.history.extend_from_slice(&self.scratch_buffer[..full]);
        self.fed += full;
        self.scratch_buffer.drain(..full);
    }

    /// Transcribes queued VAD segments, each extended by `pre_roll` samples of earlier audio
    /// so word onsets are not clipped.
    fn transcribe_segments(&mut self) {
        while let Some(segment) = self.vad.front() {
            let samples = segment.samples();
            let start = usize::try_from(segment.start()).unwrap_or(0);
            let end = start + samples.len();
            let in_history = start >= self.history_start
                && end <= self.fed
                && self.history[start - self.history_start..end - self.history_start] == *samples;
            if in_history {
                let from = start
                    .saturating_sub(self.pre_roll)
                    .max(self.last_end)
                    .max(self.history_start);
                self.transcribe_chunk(
                    &self.history[from - self.history_start..end - self.history_start],
                );
                self.history.drain(..end - self.history_start);
                self.history_start = end;
                self.last_end = end;
            } else {
                log::warn!("[stt] VAD segment at {start} not in history, skipping pre-roll");
                self.transcribe_chunk(samples);
            }
            self.vad.pop();
        }
        if !self.vad.detected() && self.history.len() > self.idle_keep {
            let excess = self.history.len() - self.idle_keep;
            self.history.drain(..excess);
            self.history_start += excess;
        }
    }
}

impl Stt for ParakeetStt {
    fn push_audio(&mut self, audio: &[f32]) {
        self.scratch_buffer.extend_from_slice(audio);
        self.feed_vad();
        self.transcribe_segments();
    }

    fn subscribe(&mut self, on_transcribe: Box<dyn Fn(String) + Send>) {
        self.callback = Some(on_transcribe);
    }

    fn flush(&mut self) {
        if !self.scratch_buffer.is_empty() {
            self.scratch_buffer.resize(self.vad_window, 0.0);
            self.feed_vad();
        }
        self.vad.flush();
        self.transcribe_segments();
        self.vad.reset();
        self.history.clear();
        self.history_start = 0;
        self.fed = 0;
        self.last_end = 0;
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::path::Path;
    use std::sync::Mutex;

    /// Speaks `text` with macOS `say` (in `voice`, if given) and returns the raw sample bytes,
    /// e.g. `LEI16@24000`.
    pub(crate) fn say(text: &str, data_format: &str, voice: Option<&str>) -> Vec<u8> {
        let thread = std::thread::current().id();
        let path = std::env::temp_dir().join(format!("talksify-stt-{}-{thread:?}.wav", std::process::id()));
        let mut command = std::process::Command::new("say");
        if let Some(voice) = voice {
            command.args(["-v", voice]);
        }
        let status = command
            .args(["--file-format=WAVE", &format!("--data-format={data_format}"), "-o"])
            .arg(&path)
            .arg(text)
            .status()
            .unwrap();
        assert!(status.success(), "say failed");
        let bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let data = bytes.windows(4).position(|w| w == b"data").unwrap() + 8;
        bytes[data..].to_vec()
    }

    /// Speaks `text` as 16 kHz mono f32, the format `push_audio` expects.
    pub(crate) fn synthesize(text: &str) -> Vec<f32> {
        synthesize_in(None, text)
    }

    /// Like `synthesize`, in a `say` voice such as "Anna" (German).
    pub(crate) fn synthesize_in(voice: Option<&str>, text: &str) -> Vec<f32> {
        let bytes = say(text, "LEF32@16000", voice);
        let (samples, _) = bytes.as_chunks::<4>();
        samples.iter().map(|b| f32::from_le_bytes(*b)).collect()
    }

    #[test]
    #[ignore = "loads the bundled STT models and uses macOS `say`"]
    fn chunks_follow_pauses() {
        let models = Path::new(env!("CARGO_MANIFEST_DIR")).join("models");
        let mut stt = ParakeetStt::new(&models).unwrap();
        let chunks = Arc::new(Mutex::new(Vec::new()));
        stt.subscribe(Box::new({
            let chunks = Arc::clone(&chunks);
            move |text: String| chunks.lock().unwrap().push(text)
        }));

        let audio = synthesize(
            "So these are small text improvements that are rewrites, but they're not complete \
             rewrites, if that makes any sense. [[slnc 1500]] The tests pass locally, but on the \
             build server they time out.",
        );
        for block in audio.chunks(160) {
            stt.push_audio(block);
        }
        stt.flush();

        let chunks = chunks.lock().unwrap().clone();
        println!("{} s of audio -> {chunks:#?}", audio.len() / 16_000);
        assert_eq!(chunks.len(), 2, "expected one chunk per sentence");
        assert!(chunks[0].to_lowercase().contains("not complete rewrites"));
        assert!(chunks[1].to_lowercase().contains("build server"));
    }
}
