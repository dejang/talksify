use std::path::{Path, PathBuf};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::audio_device::{HeapCons, HeapProd, InputFormat};
use crate::transcriber::send_audio;

pub(crate) struct Recorder {
    tx: HeapProd<f32>,
    writer: JoinHandle<()>,
    dropped: usize,
}

impl Recorder {
    pub(crate) fn new(tx: HeapProd<f32>, writer: JoinHandle<()>) -> Self {
        Self {
            tx,
            writer,
            dropped: 0,
        }
    }

    /// Pushes what fits into the writer's ring buffer and counts the rest.
    pub(crate) fn push(&mut self, samples: &[f32]) {
        self.dropped += send_audio(&mut self.tx, samples);
    }

    /// Drops the write end, joins the writer, and logs dropped samples, if any.
    pub(crate) fn finish(self) {
        drop(self.tx);
        let _ = self.writer.join();
        if self.dropped > 0 {
            log::warn!("[recording] dropped {} samples", self.dropped);
        }
    }
}

/// Writes everything from `audio` to a 32-bit float WAV at `path`, until the
/// producer is dropped.
pub(crate) fn write_wav(path: PathBuf, format: InputFormat, mut audio: HeapCons<f32>) {
    match try_write_wav(&path, format, &mut audio) {
        Ok(()) => log::info!("[recording] saved {}", path.display()),
        Err(e) => {
            log::error!("[recording] {}: {e}", path.display());
            drain_until_abandoned(&mut audio);
        }
    }
    drop(path);
}

fn try_write_wav(
    path: &Path,
    format: InputFormat,
    audio: &mut HeapCons<f32>,
) -> hound::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let spec = hound::WavSpec {
        channels: format.channels,
        sample_rate: format.sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;

    loop {
        let abandoned = audio.is_abandoned();
        let mut wrote_any = false;
        while let Ok(sample) = audio.pop() {
            writer.write_sample(sample)?;
            wrote_any = true;
        }
        if abandoned {
            break;
        }
        if !wrote_any {
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    writer.finalize()
}

fn drain_until_abandoned(audio: &mut HeapCons<f32>) {
    loop {
        let abandoned = audio.is_abandoned();
        if let Ok(chunk) = audio.read_chunk(audio.slots()) {
            chunk.commit_all();
        }
        if abandoned {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "talksify-recording-{}-{name}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn writes_float_wav_in_missing_subfolder() {
        let dir = test_dir("valid");
        let wav_path = dir.join("nested").join("sub").join("dictation.wav");
        let format = InputFormat {
            sample_rate: 48_000,
            channels: 2,
        };
        let samples = [-0.75f32, -0.25, 0.0, 0.25, 0.5, 0.875];
        let (mut tx, rx) = rtrb::RingBuffer::new(samples.len());
        for &s in &samples {
            tx.push(s).unwrap();
        }
        drop(tx);

        write_wav(wav_path.clone(), format, rx);

        let mut reader = hound::WavReader::open(&wav_path).unwrap();
        assert_eq!(
            reader.spec(),
            hound::WavSpec {
                channels: 2,
                sample_rate: 48_000,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            }
        );
        let read_back: Vec<f32> = reader.samples::<f32>().map(Result::unwrap).collect();
        assert_eq!(read_back, samples);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn returns_cleanly_when_path_cannot_be_created() {
        let dir = test_dir("invalid");
        let blocker = dir.join("not_a_directory");
        std::fs::write(&blocker, b"file").unwrap();
        let invalid_wav = blocker.join("sub").join("dictation.wav");
        let format = InputFormat {
            sample_rate: 16_000,
            channels: 1,
        };
        let (mut tx, rx) = rtrb::RingBuffer::new(16);
        for s in [0.1f32, 0.2, 0.3, 0.4] {
            tx.push(s).unwrap();
        }
        drop(tx);

        write_wav(invalid_wav, format, rx);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
