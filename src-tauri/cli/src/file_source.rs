use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use talksify_core::{AudioError, AudioSource, HeapCons, InputFormat};

/// Reads a whole WAV file into interleaved f32 samples in -1..=1.
pub(crate) fn read_wav(path: &Path) -> Result<(Arc<[f32]>, InputFormat), hound::Error> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.sample_rate == 0 || spec.channels == 0 {
        return Err(hound::Error::FormatError(
            "invalid WAV sample rate or channels",
        ));
    }
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
    Ok((
        samples.into(),
        InputFormat {
            sample_rate: spec.sample_rate,
            channels: spec.channels,
        },
    ))
}

pub(crate) struct FileSource {
    samples: Arc<[f32]>,
    format: InputFormat,
    finished: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl FileSource {
    pub(crate) fn new(samples: Arc<[f32]>, format: InputFormat, finished: Arc<AtomicBool>) -> Self {
        Self {
            samples,
            format,
            finished,
            stop: Arc::new(AtomicBool::new(false)),
            thread: None,
        }
    }
}

impl AudioSource for FileSource {
    fn start_capture(&mut self) -> Result<HeapCons<f32>, AudioError> {
        let format = self.format;
        let channels = usize::from(format.channels);
        let (mut producer, consumer) =
            rtrb::RingBuffer::new(format.sample_rate as usize * channels);
        let samples = Arc::clone(&self.samples);
        let finished = Arc::clone(&self.finished);
        let stop = Arc::clone(&self.stop);
        let start = Instant::now();
        self.thread = Some(
            std::thread::Builder::new()
                .name("file-source".into())
                .spawn(move || {
                    let block_samples = (format.sample_rate as usize / 100).max(1) * channels;
                    let mut end_sample = 0;
                    let mut dropped = 0;
                    'playback: for block in samples.chunks(block_samples) {
                        end_sample += block.len();
                        let end_frame = end_sample.div_ceil(channels);
                        #[expect(
                            clippy::cast_precision_loss,
                            reason = "frame counts fit f64 for practical WAV files"
                        )]
                        let deadline = start
                            + Duration::from_secs_f64(
                                end_frame as f64 / f64::from(format.sample_rate),
                            );
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
                        finished.store(true, Ordering::Release);
                    }
                })
                .map_err(|e| AudioError::StreamError(e.to_string()))?,
        );
        Ok(consumer)
    }

    fn stop_capture(&mut self) -> Result<(), AudioError> {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            thread.thread().unpark();
            let _ = thread.join();
        }
        Ok(())
    }

    fn input_format(&self) -> Option<InputFormat> {
        Some(self.format)
    }
}

impl Drop for FileSource {
    fn drop(&mut self) {
        let _ = self.stop_capture();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_integer_stereo_in_order() {
        let path = std::env::temp_dir().join(format!("talksify-wav-{}.wav", std::process::id()));
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 22_050,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let input = [i16::MIN, i16::MAX, 0, -8192, 16384, 8192];
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        for sample in input {
            writer.write_sample(sample).unwrap();
        }
        writer.finalize().unwrap();
        let (samples, format) = read_wav(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(
            format,
            InputFormat {
                sample_rate: 22_050,
                channels: 2
            }
        );
        assert_eq!(samples.len(), input.len());
        assert!(samples.iter().zip(input).all(|(&actual, expected)| {
            (actual - f32::from(expected) / 32768.0).abs() < f32::EPSILON
        }));
        assert!((samples[0] + 1.0).abs() < f32::EPSILON);
    }

    fn source(frames: usize, channels: u16) -> (FileSource, Arc<AtomicBool>, Arc<[f32]>) {
        let samples: Arc<[f32]> = (0..frames * usize::from(channels))
            .map(|i| f32::from(u16::try_from(i % 1000).unwrap()) / 1000.0)
            .collect();
        let finished = Arc::new(AtomicBool::new(false));
        (
            FileSource::new(
                Arc::clone(&samples),
                InputFormat {
                    sample_rate: 22_050,
                    channels,
                },
                Arc::clone(&finished),
            ),
            finished,
            samples,
        )
    }

    #[test]
    fn delivers_all_samples_then_finishes_at_real_time_speed() {
        // An odd frame count at 22,050 Hz exercises fractional blocks and the final short block.
        let (mut source, finished, samples) = source(11_025, 2);
        let start = Instant::now();
        let mut consumer = source.start_capture().unwrap();
        let mut received = Vec::new();
        loop {
            while let Ok(sample) = consumer.pop() {
                received.push(sample);
            }
            if finished.load(Ordering::Acquire) {
                break;
            }
            assert!(start.elapsed() < Duration::from_secs(2), "playback stalled");
            std::thread::sleep(Duration::from_millis(1));
        }
        let elapsed = start.elapsed();
        while let Ok(sample) = consumer.pop() {
            received.push(sample);
        }
        source.stop_capture().unwrap();
        assert_eq!(received, &*samples);
        assert!(
            elapsed >= Duration::from_millis(500),
            "finished early: {elapsed:?}"
        );
        assert!(
            elapsed < Duration::from_millis(700),
            "finished late: {elapsed:?}"
        );
    }

    #[test]
    fn stopping_is_prompt_and_does_not_signal_finished() {
        let (mut source, finished, _) = source(22_050, 1);
        let _consumer = source.start_capture().unwrap();
        std::thread::sleep(Duration::from_millis(50));
        let start = Instant::now();
        source.stop_capture().unwrap();
        assert!(start.elapsed() < Duration::from_millis(100));
        assert!(!finished.load(Ordering::Acquire));
    }
}
