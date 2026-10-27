use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{
    FromSample, Host, Sample, SampleFormat, SizedSample, Stream, StreamConfig, StreamError,
};
use rubato::{FftFixedIn, Resampler};

pub(crate) type HeapProd<T> = rtrb::Producer<T>;
pub(crate) type HeapCons<T> = rtrb::Consumer<T>;

#[derive(Debug, thiserror::Error)]
pub(crate) enum AudioError {
    #[error("audio device unavailable")]
    DeviceUnavailable,
    #[error("stream error: {0}")]
    StreamError(String),
}

/// Native layout of the captured samples: interleaved `channels` at `sample_rate` Hz.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct InputFormat {
    pub(crate) sample_rate: u32,
    pub(crate) channels: u16,
}

pub(crate) struct AudioDevice {
    device_name: Option<String>,
    stream: Option<Stream>,
    format: Option<InputFormat>,
    ring_seconds: usize,
}

impl AudioDevice {
    pub(crate) fn new(device_name: Option<String>) -> Self {
        Self {
            device_name,
            stream: None,
            format: None,
            ring_seconds: 1,
        }
    }

    /// Layout of the samples `start_capture` delivers; `None` before capture starts.
    pub(crate) fn input_format(&self) -> Option<InputFormat> {
        self.format
    }

    /// Delivers the device's interleaved samples, scaled to f32 in -1..=1 and otherwise
    /// untouched, so the audio callback stays cheap. See [`AudioProcessor`].
    pub(crate) fn start_capture(&mut self) -> Result<HeapCons<f32>, AudioError> {
        let host = cpal::default_host();
        let device = self.find_device(&host)?;

        let config = device
            .default_input_config()
            .map_err(|e| AudioError::StreamError(e.to_string()))?;
        let stream_config: StreamConfig = config.config();
        let format = InputFormat {
            sample_rate: stream_config.sample_rate.0,
            channels: stream_config.channels,
        };
        log::info!("[audio] capturing {:?} {format:?}", config.sample_format());

        let capacity =
            format.sample_rate as usize * usize::from(format.channels) * self.ring_seconds;
        let (producer, consumer) = rtrb::RingBuffer::new(capacity);
        let stream = match config.sample_format() {
            SampleFormat::F32 => build_stream::<f32>(&device, &stream_config, producer),
            SampleFormat::F64 => build_stream::<f64>(&device, &stream_config, producer),
            SampleFormat::I8 => build_stream::<i8>(&device, &stream_config, producer),
            SampleFormat::I16 => build_stream::<i16>(&device, &stream_config, producer),
            SampleFormat::I32 => build_stream::<i32>(&device, &stream_config, producer),
            SampleFormat::I64 => build_stream::<i64>(&device, &stream_config, producer),
            SampleFormat::U8 => build_stream::<u8>(&device, &stream_config, producer),
            SampleFormat::U16 => build_stream::<u16>(&device, &stream_config, producer),
            SampleFormat::U32 => build_stream::<u32>(&device, &stream_config, producer),
            SampleFormat::U64 => build_stream::<u64>(&device, &stream_config, producer),
            format => {
                return Err(AudioError::StreamError(format!(
                    "unsupported sample format: {format:?}"
                )));
            }
        }?;

        stream
            .play()
            .map_err(|e| AudioError::StreamError(e.to_string()))?;

        self.stream = Some(stream);
        self.format = Some(format);
        Ok(consumer)
    }

    pub(crate) fn stop_capture(&mut self) -> Result<(), AudioError> {
        if let Some(stream) = self.stream.take() {
            stream
                .pause()
                .map_err(|e| AudioError::StreamError(e.to_string()))?;
            drop(stream);
        }
        Ok(())
    }

    fn find_device(&self, host: &Host) -> Result<cpal::Device, AudioError> {
        if let Some(name) = &self.device_name {
            host.input_devices()
                .map_err(|e| AudioError::StreamError(e.to_string()))?
                .find(|d| d.name().is_ok_and(|n| n == *name))
                .ok_or(AudioError::DeviceUnavailable)
        } else {
            host.default_input_device()
                .ok_or(AudioError::DeviceUnavailable)
        }
    }
}

/// Opens an input stream whose real-time callback only scales samples into the ring buffer.
fn build_stream<T>(
    device: &cpal::Device,
    config: &StreamConfig,
    mut producer: HeapProd<f32>,
) -> Result<Stream, AudioError>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    device
        .build_input_stream(
            config,
            move |data: &[T], _| {
                for sample in data {
                    let _ = producer.push(f32::from_sample(*sample));
                }
            },
            |err: StreamError| log::error!("audio callback error: {err}"),
            None,
        )
        .map_err(|e| AudioError::StreamError(e.to_string()))
}

/// Downmixes the device's interleaved samples to mono and resamples them to the model's rate,
/// using only buffers allocated in `new`.
pub(crate) struct AudioProcessor {
    channels: u16,
    resampler: FftFixedIn<f32>,
    /// Zero chunks that push the resampler's delayed samples out in `finish`.
    flush_chunks: usize,
    /// Samples of a frame cut off at the end of the previous block.
    partial_frame: Vec<f32>,
    /// Mono samples waiting for a full resampler chunk.
    pending: Vec<f32>,
    /// One resampled chunk.
    chunk_out: Vec<f32>,
    /// Resampled audio returned by the latest `process` or `finish`.
    output: Vec<f32>,
}

impl AudioProcessor {
    /// `max_block`: the most interleaved samples passed to one `process` call.
    pub(crate) fn new(format: InputFormat, output_rate: u32, max_block: usize) -> Option<Self> {
        let channels = usize::from(format.channels);
        let input_rate = format.sample_rate as usize;
        let resampler =
            FftFixedIn::new(input_rate, output_rate as usize, input_rate / 100, 1, 1).ok()?;
        let chunk_in = resampler.input_frames_next();
        let chunk_out = resampler.output_frames_max();
        let flush_chunks = resampler.output_delay().div_ceil(chunk_out) + 1;
        let max_chunks = (max_block / channels + chunk_in).div_ceil(chunk_in) + flush_chunks;
        Some(Self {
            channels: format.channels,
            resampler,
            flush_chunks,
            partial_frame: Vec::with_capacity(channels),
            pending: Vec::with_capacity(max_chunks * chunk_in),
            chunk_out: vec![0.0; chunk_out],
            output: Vec::with_capacity(max_chunks * chunk_out),
        })
    }

    /// Takes up to `max_block` interleaved samples and returns the resampled audio now ready.
    pub(crate) fn process(&mut self, interleaved: &[f32]) -> &[f32] {
        for &sample in interleaved {
            self.partial_frame.push(sample);
            if self.partial_frame.len() == usize::from(self.channels) {
                let sum: f32 = self.partial_frame.iter().sum();
                self.pending.push(sum / f32::from(self.channels));
                self.partial_frame.clear();
            }
        }
        self.resample_pending()
    }

    /// Returns the audio still held for resampling and resets for the next recording.
    pub(crate) fn finish(&mut self) -> &[f32] {
        let chunk_in = self.resampler.input_frames_next();
        let padded = (self.pending.len().div_ceil(chunk_in) + self.flush_chunks) * chunk_in;
        self.pending.resize(padded, 0.0);
        self.partial_frame.clear();
        self.resample_pending();
        self.resampler.reset();
        &self.output
    }

    fn resample_pending(&mut self) -> &[f32] {
        self.output.clear();
        let chunk_in = self.resampler.input_frames_next();
        let mut start = 0;
        while self.pending.len() - start >= chunk_in {
            let input = [&self.pending[start..start + chunk_in]];
            if let Ok((_, written)) =
                self.resampler
                    .process_into_buffer(&input, &mut [&mut self.chunk_out], None)
            {
                self.output.extend_from_slice(&self.chunk_out[..written]);
            }
            start += chunk_in;
        }
        self.pending.drain(..start);
        &self.output
    }
}

/// Turns captured samples into loudness levels for the overlay meter.
#[derive(Clone)]
pub(crate) struct LevelMeter {
    window: usize,
    floor_db: f32,
    range_db: f32,
    sum_squares: f32,
    count: usize,
}

impl LevelMeter {
    /// One level per `window` samples; `floor_db`..`floor_db + range_db` (dBFS) maps to 0..1.
    pub(crate) fn new(window: usize, floor_db: f32, range_db: f32) -> Self {
        Self {
            window,
            floor_db,
            range_db,
            sum_squares: 0.0,
            count: 0,
        }
    }

    /// Returns a level in 0..=1 each time a full window has been pushed.
    pub(crate) fn push(&mut self, samples: &[f32]) -> Option<f32> {
        self.sum_squares += samples.iter().map(|s| s * s).sum::<f32>();
        self.count += samples.len();
        if self.count < self.window {
            return None;
        }
        #[expect(clippy::cast_precision_loss, reason = "a window is far below 2^24 samples")]
        let count = self.count as f32;
        let rms = (self.sum_squares / count).sqrt();
        self.sum_squares = 0.0;
        self.count = 0;
        let db = 20.0 * rms.max(1e-6).log10();
        Some(((db - self.floor_db) / self.range_db).clamp(0.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn process_all(format: InputFormat, samples: &[f32]) -> Vec<f32> {
        let mut processor = AudioProcessor::new(format, 16_000, 479).unwrap();
        let mut out = Vec::new();
        for block in samples.chunks(479) {
            out.extend_from_slice(processor.process(block));
        }
        out.extend_from_slice(processor.finish());
        out
    }

    // One second of a 0.5-amplitude sine at `hz`, sampled at `rate`.
    fn tone(hz: f32, rate: u16) -> Vec<f32> {
        (0..rate)
            .map(|i| 0.5 * (2.0 * PI * hz * f32::from(i) / f32::from(rate)).sin())
            .collect()
    }

    fn rms(samples: &[f32]) -> f64 {
        let sum: f64 = samples.iter().map(|&s| f64::from(s).powi(2)).sum();
        (sum / f64::from(u32::try_from(samples.len()).unwrap())).sqrt()
    }

    #[test]
    fn integer_samples_scale_to_unit_range() {
        assert!((f32::from_sample(i16::MIN) + 1.0).abs() < 1e-6);
        assert!((f32::from_sample(i16::MAX) - 1.0).abs() < 1e-4);
        assert!(f32::from_sample(0.25f64).abs() > 0.2);
    }

    #[test]
    fn processor_averages_channels_across_reads() {
        let stereo: Vec<f32> = [0.2, 0.4].repeat(48_000);
        let out = process_all(
            InputFormat {
                sample_rate: 48_000,
                channels: 2,
            },
            &stereo,
        );
        assert!(out[1_000..15_000].iter().all(|s| (s - 0.3).abs() < 1e-3));
    }

    // A second of 440 Hz tone at 0.5 amplitude must come out as a second at 16 kHz, same level,
    // give or take the resampler's delay.
    fn assert_resamples(rate: u16) {
        let out = process_all(
            InputFormat {
                sample_rate: u32::from(rate),
                channels: 1,
            },
            &tone(440.0, rate),
        );
        assert!(
            (16_000..17_000).contains(&out.len()),
            "{rate} Hz gave {} samples",
            out.len()
        );
        let level = rms(&out[1_000..15_000]);
        assert!((level - 0.354).abs() < 0.02, "{rate} Hz tone RMS {level}");
    }

    #[test]
    fn processor_resamples_to_16k() {
        assert_resamples(48_000);
        assert_resamples(44_100);
        assert_resamples(24_000);
        assert_resamples(16_000);
    }

    #[test]
    fn processor_never_grows_its_buffers() {
        let format = InputFormat {
            sample_rate: 44_100,
            channels: 2,
        };
        let mut processor = AudioProcessor::new(format, 16_000, 479).unwrap();
        let capacities = |p: &AudioProcessor| {
            (
                p.partial_frame.capacity(),
                p.pending.capacity(),
                p.output.capacity(),
            )
        };
        let before = capacities(&processor);
        let stereo = tone(440.0, 44_100).repeat(2);
        for block in stereo.chunks(479) {
            let _ = processor.process(block);
        }
        let _ = processor.finish();
        assert_eq!(capacities(&processor), before);
    }

    #[test]
    fn processor_filters_out_content_above_8k() {
        let out = process_all(
            InputFormat {
                sample_rate: 48_000,
                channels: 1,
            },
            &tone(12_000.0, 48_000),
        );
        assert!(rms(&out[1_000..15_000]) < 0.02);
    }

    #[test]
    fn level_meter_reports_once_per_window() {
        let mut meter = LevelMeter::new(800, -55.0, 45.0);
        let levels: Vec<f32> = vec![0.0f32; 1_600]
            .chunks(160)
            .filter_map(|c| meter.push(c))
            .collect();
        assert_eq!(levels.len(), 2);
    }

    #[test]
    fn level_meter_maps_silence_and_speech() {
        let mut meter = LevelMeter::new(480, -55.0, 45.0);
        assert!(meter.push(&[0.0; 480]).is_some_and(|l| l < 0.01));
        let tone: Vec<f32> = (0..480u16).map(|i| 0.1 * (f32::from(i) * 0.3).sin()).collect();
        // A 0.1-amplitude sine is about -23 dBFS, so roughly 70% of the meter.
        assert!(meter.push(&tone).is_some_and(|l| (0.6..0.8).contains(&l)));
        assert!(meter.push(&[1.0; 480]).is_some_and(|l| l > 0.99));
    }
}
