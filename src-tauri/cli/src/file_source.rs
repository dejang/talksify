use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use talksify_core::{AudioError, AudioSource, HeapCons, InputFormat, Transcriber};

/// Validates the WAV header and returns its native stream format without decoding samples.
pub(crate) fn inspect_wav_format(path: &Path) -> Result<InputFormat, String> {
    hound::WavReader::open(path)
        .and_then(|reader| format_from_wav_spec(reader.spec()))
        .map_err(|error| wav_read_error(path, &error))
}

fn wav_read_error(path: &Path, error: &hound::Error) -> String {
    format!("cannot read {}: {error}", path.display())
}

fn format_from_wav_spec(spec: hound::WavSpec) -> Result<InputFormat, hound::Error> {
    if spec.sample_rate == 0 || spec.channels == 0 {
        return Err(hound::Error::FormatError(
            "invalid WAV sample rate or channels",
        ));
    }
    Ok(InputFormat {
        sample_rate: spec.sample_rate,
        channels: spec.channels,
    })
}

/// Decoded interleaved f32 WAV samples and their native stream format.
pub(crate) struct WavAudio {
    samples: Vec<f32>,
    format: InputFormat,
}

impl WavAudio {
    /// Reads a whole WAV file into interleaved f32 samples in -1..=1.
    fn read(path: &Path) -> Result<Self, String> {
        Self::decode(path).map_err(|error| wav_read_error(path, &error))
    }

    fn decode(path: &Path) -> Result<Self, hound::Error> {
        let mut reader = hound::WavReader::open(path)?;
        let spec = reader.spec();
        let format = format_from_wav_spec(spec)?;
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => {
                let scale = 2.0_f32.powi(i32::from(spec.bits_per_sample) - 1);
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "WAV integers are normalized to f32"
                )]
                reader
                    .samples::<i32>()
                    .map(|s| s.map(|s| s as f32 / scale))
                    .collect::<Result<_, _>>()?
            }
            hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        };
        Ok(Self { samples, format })
    }

    /// Computes total playback duration in seconds for this audio clip.
    fn duration_secs(&self) -> f64 {
        self.format.duration_secs(self.samples.len())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum PlaybackOutcome {
    Pending,
    Finished(f64),
    Failed(String),
}

/// Shared completion state between [`FileSource`] playback and [`PlaybackState::play`].
pub(crate) struct PlaybackState {
    outcome: Mutex<PlaybackOutcome>,
}

impl PlaybackState {
    pub(crate) fn new() -> Self {
        Self {
            outcome: Mutex::new(PlaybackOutcome::Pending),
        }
    }

    pub(crate) fn reset(&self) {
        *self.outcome.lock().unwrap_or_else(PoisonError::into_inner) = PlaybackOutcome::Pending;
    }

    fn finish(&self, duration_secs: f64) {
        *self.outcome.lock().unwrap_or_else(PoisonError::into_inner) =
            PlaybackOutcome::Finished(duration_secs);
    }

    fn fail(&self, error: String) {
        *self.outcome.lock().unwrap_or_else(PoisonError::into_inner) =
            PlaybackOutcome::Failed(error);
    }

    pub(crate) fn outcome(&self) -> PlaybackOutcome {
        self.outcome
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Starts capture and waits until playback finishes, returning the clip's duration in seconds.
    pub(crate) fn play(&self, transcriber: &mut Transcriber<FileSource>) -> Result<f64, String> {
        transcriber
            .start_transcribing(None)
            .map_err(|e| e.to_string())?;
        loop {
            match self.outcome() {
                PlaybackOutcome::Finished(duration_secs) => return Ok(duration_secs),
                PlaybackOutcome::Failed(error) => {
                    if transcriber.is_transcribing() {
                        let _ = transcriber.stop_transcribing();
                    }
                    return Err(error);
                }
                PlaybackOutcome::Pending => {
                    if !transcriber.is_transcribing() {
                        return match self.outcome() {
                            PlaybackOutcome::Finished(duration_secs) => Ok(duration_secs),
                            PlaybackOutcome::Failed(error) => Err(error),
                            PlaybackOutcome::Pending => {
                                Err("audio capture stopped before the end of the file".into())
                            }
                        };
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
        }
    }
}

pub(crate) struct FileSource {
    path: PathBuf,
    format: Option<InputFormat>,
    playback: Arc<PlaybackState>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl FileSource {
    pub(crate) fn new(path: PathBuf, playback: Arc<PlaybackState>) -> Self {
        Self {
            path,
            format: None,
            playback,
            stop: Arc::new(AtomicBool::new(false)),
            thread: None,
        }
    }
}

impl AudioSource for FileSource {
    fn start_capture(&mut self) -> Result<HeapCons<f32>, AudioError> {
        let format = inspect_wav_format(&self.path).map_err(|message| {
            self.playback.fail(message.clone());
            AudioError::StreamError(message)
        })?;
        self.format = Some(format);

        let channels = usize::from(format.channels);
        let (mut producer, consumer) =
            rtrb::RingBuffer::new(format.sample_rate as usize * channels);
        let path = self.path.clone();
        let playback = Arc::clone(&self.playback);
        let stop = Arc::clone(&self.stop);
        self.thread = Some(
            std::thread::Builder::new()
                .name("file-source".into())
                .spawn(move || {
                    let audio = match WavAudio::read(&path) {
                        Ok(audio) => audio,
                        Err(message) => {
                            playback.fail(message);
                            return;
                        }
                    };
                    let start = Instant::now();
                    let block_samples = (audio.format.sample_rate as usize / 100).max(1)
                        * usize::from(audio.format.channels);
                    let mut end_sample = 0;
                    let mut dropped = 0;
                    'playback: for block in audio.samples.chunks(block_samples) {
                        end_sample += block.len();
                        let deadline =
                            start + Duration::from_secs_f64(audio.format.duration_secs(end_sample));
                        loop {
                            if stop.load(Ordering::Acquire) {
                                break 'playback;
                            }
                            let now = Instant::now();
                            if now >= deadline {
                                break;
                            }
                            std::thread::park_timeout(deadline - now);
                        }
                        for &sample in block {
                            if producer.push(sample).is_err() {
                                dropped += 1;
                            }
                        }
                    }
                    if dropped > 0 {
                        log::warn!("[file-source] dropped {dropped} samples");
                    }
                    if !stop.load(Ordering::Acquire) {
                        playback.finish(audio.duration_secs());
                    }
                })
                .map_err(|e| AudioError::StreamError(e.to_string()))?,
        );
        Ok(consumer)
    }

    fn stop_capture(&mut self) -> Result<(), AudioError> {
        self.stop.store(true, Ordering::Release);
        let Some(thread) = self.thread.take() else {
            return Ok(());
        };
        thread.thread().unpark();
        thread
            .join()
            .map_err(|_| AudioError::StreamError("file-source playback thread panicked".into()))
    }

    fn input_format(&self) -> Option<InputFormat> {
        self.format
    }
}

impl Drop for FileSource {
    fn drop(&mut self) {
        let _ = self.stop_capture();
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    fn write_wav<S: hound::Sample + Copy>(path: &Path, spec: hound::WavSpec, samples: &[S]) {
        let mut writer = hound::WavWriter::create(path, spec).unwrap();
        for &sample in samples {
            writer.write_sample(sample).unwrap();
        }
        writer.finalize().unwrap();
    }

    fn drain_until_finished(
        source: &mut FileSource,
        playback: &PlaybackState,
    ) -> (Vec<f32>, Duration, f64) {
        let start = Instant::now();
        let mut consumer = source.start_capture().unwrap();
        let mut received = Vec::new();
        let duration_secs = loop {
            while let Ok(sample) = consumer.pop() {
                received.push(sample);
            }
            match playback.outcome() {
                PlaybackOutcome::Finished(secs) => break secs,
                PlaybackOutcome::Failed(err) => panic!("unexpected playback failure: {err}"),
                PlaybackOutcome::Pending => {}
            }
            assert!(start.elapsed() < Duration::from_secs(2), "playback stalled");
            std::thread::sleep(Duration::from_millis(1));
        };
        let elapsed = start.elapsed();
        while let Ok(sample) = consumer.pop() {
            received.push(sample);
        }
        source.stop_capture().unwrap();
        (received, elapsed, duration_secs)
    }

    #[test]
    fn plays_integer_and_float_wavs_in_order_at_real_time_speed() {
        let dir = TempDir::new().unwrap();

        // Integer stereo WAV: verifies int-to-f32 normalization, ordering, and reported format.
        let int_path = dir.path().join("int_stereo.wav");
        let int_input = [i16::MIN, i16::MAX, 0, -8192, 16384, 8192];
        write_wav(
            &int_path,
            hound::WavSpec {
                channels: 2,
                sample_rate: 22_050,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
            &int_input,
        );
        let playback = Arc::new(PlaybackState::new());
        let mut source = FileSource::new(int_path, Arc::clone(&playback));
        let (received, _, _) = drain_until_finished(&mut source, &playback);
        assert_eq!(
            source.input_format(),
            Some(InputFormat {
                sample_rate: 22_050,
                channels: 2,
            })
        );
        assert_eq!(received.len(), int_input.len());
        assert!(received.iter().zip(int_input).all(|(&actual, expected)| {
            (actual - f32::from(expected) / 32768.0).abs() < f32::EPSILON
        }));
        assert!((received[0] + 1.0).abs() < f32::EPSILON);

        // Odd stereo frame count at 22,050 Hz (0.5 s): exercises fractional & final short blocks.
        let float_path = dir.path().join("float_stereo.wav");
        let expected: Vec<f32> = (0..11_025 * 2)
            .map(|i| f32::from(u16::try_from(i % 1000).unwrap()) / 1000.0)
            .collect();
        write_wav(
            &float_path,
            hound::WavSpec {
                channels: 2,
                sample_rate: 22_050,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
            &expected,
        );
        let playback = Arc::new(PlaybackState::new());
        let mut source = FileSource::new(float_path, Arc::clone(&playback));
        let (received, elapsed, duration_secs) = drain_until_finished(&mut source, &playback);
        assert_eq!(received, expected);
        assert!((duration_secs - 0.5).abs() < 1e-9);
        assert!(
            elapsed >= Duration::from_millis(500) && elapsed < Duration::from_millis(700),
            "unexpected real-time duration: {elapsed:?}"
        );
    }

    #[test]
    fn stopping_is_prompt_and_errors_are_reported() {
        let dir = TempDir::new().unwrap();
        let wav_path = dir.path().join("one_sec.wav");
        write_wav(
            &wav_path,
            hound::WavSpec {
                channels: 1,
                sample_rate: 22_050,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
            &vec![0.0_f32; 22_050],
        );

        let playback = Arc::new(PlaybackState::new());
        let mut source = FileSource::new(wav_path, Arc::clone(&playback));
        let _consumer = source.start_capture().unwrap();
        std::thread::sleep(Duration::from_millis(50));
        let start = Instant::now();
        source.stop_capture().unwrap();
        assert!(start.elapsed() < Duration::from_millis(100));
        assert_eq!(playback.outcome(), PlaybackOutcome::Pending);

        // Missing WAV fails capture and records the failure outcome.
        let missing_playback = Arc::new(PlaybackState::new());
        let mut missing = FileSource::new(
            dir.path().join("missing.wav"),
            Arc::clone(&missing_playback),
        );
        assert!(missing.start_capture().is_err());
        assert!(matches!(
            missing_playback.outcome(),
            PlaybackOutcome::Failed(_)
        ));
    }
}
