use std::borrow::Cow;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use talksify_core::{TextCleaner, Transcriber, Transcript};

use crate::dump::RecordingName;
use crate::file_source::{FileSource, PlaybackState};
use crate::harness::{SampleItem, SampleRunner};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct WavPair {
    pub(crate) wav: PathBuf,
    pub(crate) text: PathBuf,
}

impl SampleItem for WavPair {
    fn sample_name(&self) -> Cow<'_, RecordingName> {
        Cow::Owned(RecordingName::from_path(&self.wav))
    }
}

/// Discovered `.wav`/`.txt` pairs in a recording folder, plus `.wav` files missing a `.txt`.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct RecordingFolder {
    pub(crate) paired: Vec<WavPair>,
    pub(crate) skipped: Vec<PathBuf>,
}

impl RecordingFolder {
    pub(crate) fn scan(folder: &Path) -> std::io::Result<Self> {
        let mut paired = Vec::new();
        let mut skipped = Vec::new();
        for entry in fs::read_dir(folder)? {
            let path = entry?.path();
            if path.extension().is_some_and(|ext| ext == "wav") && path.is_file() {
                let text = path.with_extension("txt");
                if text.is_file() {
                    paired.push(WavPair { wav: path, text });
                } else {
                    skipped.push(path);
                }
            }
        }
        paired.sort();
        skipped.sort();
        Ok(Self { paired, skipped })
    }

    pub(crate) fn skipped_names(&self) -> Vec<RecordingName> {
        self.skipped
            .iter()
            .map(|path| RecordingName::from_path(path))
            .collect()
    }
}

/// Result of transcribing one paired `.wav`/`.txt` recording.
pub(crate) struct TranscribedSample {
    pub(crate) duration_secs: f64,
    pub(crate) expected: String,
    pub(crate) transcript: Transcript,
    pub(crate) wait: Duration,
}

/// Real-time WAV transcription runner backed by [`Transcriber<FileSource>`].
pub(crate) struct WavRunner {
    transcriber: Transcriber<FileSource>,
    current_wav: Arc<Mutex<PathBuf>>,
    playback: Arc<PlaybackState>,
}

impl WavRunner {
    pub(crate) fn new(models: &Path, cleaner: TextCleaner) -> Result<Self, String> {
        let current_wav = Arc::new(Mutex::new(PathBuf::new()));
        let playback = Arc::new(PlaybackState::new());
        let source_wav = Arc::clone(&current_wav);
        let source_playback = Arc::clone(&playback);
        let make_source = move || {
            let path = source_wav
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clone();
            FileSource::new(path, Arc::clone(&source_playback))
        };
        let transcriber = Transcriber::with_cleaner(models, cleaner, make_source, Arc::new(|_| {}))
            .map_err(|e| e.to_string())?;
        Ok(Self {
            transcriber,
            current_wav,
            playback,
        })
    }
}

impl SampleRunner<WavPair> for WavRunner {
    fn run_sample(&mut self, pair: &WavPair) -> Result<TranscribedSample, String> {
        let expected = fs::read_to_string(&pair.text)
            .map_err(|e| format!("cannot read {}: {e}", pair.text.display()))?;
        self.current_wav
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone_from(&pair.wav);
        self.playback.reset();
        let duration_secs = FileSource::play_into(&mut self.transcriber, &self.playback)?;
        let stopped = Instant::now();
        let transcript = self
            .transcriber
            .stop_transcribing_strict()
            .map_err(|e| e.to_string())?;
        let wait = stopped.elapsed();
        Ok(TranscribedSample {
            duration_secs,
            expected,
            transcript,
            wait,
        })
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    #[test]
    fn scans_recording_folders() {
        let root = TempDir::new().unwrap();

        let populated = root.path().join("populated");
        fs::create_dir_all(populated.join("sub")).unwrap();
        fs::create_dir(populated.join("directory.wav")).unwrap();
        for name in [
            "b.wav",
            "b.txt",
            "a.wav",
            "a.txt",
            "c.wav",
            "d.txt",
            "sub/e.wav",
            "sub/e.txt",
            "upper.WAV",
            "upper.txt",
        ] {
            fs::write(populated.join(name), "").unwrap();
        }
        let scanned = RecordingFolder::scan(&populated).unwrap();
        assert_eq!(
            scanned.paired,
            vec![
                WavPair {
                    wav: populated.join("a.wav"),
                    text: populated.join("a.txt"),
                },
                WavPair {
                    wav: populated.join("b.wav"),
                    text: populated.join("b.txt"),
                },
            ]
        );
        assert_eq!(scanned.skipped, vec![populated.join("c.wav")]);

        let empty = root.path().join("empty");
        fs::create_dir(&empty).unwrap();
        assert_eq!(
            RecordingFolder::scan(&empty).unwrap(),
            RecordingFolder {
                paired: Vec::new(),
                skipped: Vec::new(),
            }
        );
        assert!(RecordingFolder::scan(&root.path().join("missing")).is_err());
    }
}
