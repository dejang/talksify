use std::borrow::Cow;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use talksify_core::{TextCleaner, Transcript};

use crate::corpus::TranscribedSample;
use crate::dump::{RawDump, RawEntry, RecordingName};
use crate::harness::{Harness, SampleItem, SampleRunner};

impl SampleItem for RawEntry {
    fn sample_name(&self) -> Cow<'_, RecordingName> {
        Cow::Borrowed(&self.file)
    }
}

impl SampleRunner<RawEntry> for TextCleaner {
    fn run_sample(&mut self, entry: &RawEntry) -> Result<TranscribedSample, String> {
        let started = Instant::now();
        let cleaned = self.try_clean(&entry.raw)?;
        Ok(TranscribedSample {
            duration_secs: entry.duration_secs,
            expected: entry.expected.clone(),
            transcript: Transcript {
                raw: entry.raw.clone(),
                cleaned,
            },
            wait: started.elapsed(),
        })
    }
}

pub(crate) fn run_replay(dump_path: &Path, models: &Path) -> Result<ExitCode, String> {
    let dump = RawDump::read(dump_path)?;
    if dump.is_empty() {
        return Err(format!("no entries in {}", dump_path.display()));
    }
    let mut cleaner = TextCleaner::try_new(models)?;
    Ok(ReplayHarness::new(dump).execute(&mut cleaner))
}

/// Text-only replay runner over a loaded [`RawDump`] using the shared [`Harness`].
pub(crate) struct ReplayHarness {
    dump: RawDump,
    harness: Harness,
}

impl ReplayHarness {
    pub(crate) fn new(dump: RawDump) -> Self {
        Self {
            dump,
            harness: Harness::default(),
        }
    }

    pub(crate) fn execute(mut self, runner: &mut impl SampleRunner<RawEntry>) -> ExitCode {
        self.harness.evaluate(self.dump.entries(), runner, None);
        self.harness.finish()
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::Duration;

    use tempfile::TempDir;

    use super::*;
    use crate::dump::StagedDump;

    struct StubCleaner {
        fail_on: String,
    }

    impl SampleRunner<RawEntry> for StubCleaner {
        fn run_sample(&mut self, entry: &RawEntry) -> Result<TranscribedSample, String> {
            if entry.raw == self.fail_on {
                Err("LLM inference failed: runtime error".into())
            } else {
                Ok(TranscribedSample {
                    duration_secs: entry.duration_secs,
                    expected: entry.expected.clone(),
                    transcript: Transcript {
                        raw: entry.raw.clone(),
                        cleaned: "Hello world.".into(),
                    },
                    wait: Duration::from_millis(10),
                })
            }
        }
    }

    #[test]
    fn validates_dump_paths_and_runs_fallback_without_models() {
        let dir = TempDir::new().unwrap();

        let empty_path = dir.path().join("empty.json");
        StagedDump::prepare(&empty_path).unwrap().publish().unwrap();

        let bad_duration = dir.path().join("neg.json");
        fs::write(
            &bad_duration,
            r#"[{"file":"bad.wav","duration_secs":-90.0,"expected":"Hello.","raw":"hello."}]"#,
        )
        .unwrap();

        let valid_path = dir.path().join("valid.json");
        let mut staged = StagedDump::prepare(&valid_path).unwrap();
        staged.push(RawEntry {
            file: RecordingName::from_path(Path::new("short.wav")),
            duration_secs: 12.5,
            expected: "Hello world.".into(),
            raw: "um hello world.".into(),
        });
        staged.publish().unwrap();

        for (label, dump_path, models_dir) in [
            (
                "missing dump",
                dir.path().join("missing.json"),
                dir.path().to_path_buf(),
            ),
            ("empty dump", empty_path, dir.path().to_path_buf()),
            (
                "negative duration",
                bad_duration,
                dir.path().join("missing-models"),
            ),
        ] {
            assert!(run_replay(&dump_path, &models_dir).is_err(), "{label}");
        }
        assert_eq!(run_replay(&valid_path, dir.path()), Ok(ExitCode::SUCCESS));
    }

    #[test]
    fn fails_nonzero_when_any_replayed_entry_errors() {
        let good = RawEntry {
            file: RecordingName::from_path(Path::new("good.wav")),
            duration_secs: 10.0,
            expected: "Hello world.".into(),
            raw: "hello world.".into(),
        };
        let failed = RawEntry {
            file: RecordingName::from_path(Path::new("failed.wav")),
            duration_secs: 75.0,
            expected: "This sample should not be scored.".into(),
            raw: "trigger-inference-error".into(),
        };

        let mut cleaner = StubCleaner {
            fail_on: "trigger-inference-error".into(),
        };
        for (label, entries, expected) in [
            ("all succeed", vec![good.clone()], ExitCode::SUCCESS),
            (
                "mixed failure",
                vec![good, failed.clone()],
                ExitCode::FAILURE,
            ),
            ("all fail", vec![failed], ExitCode::FAILURE),
        ] {
            let mut dump = RawDump::new();
            for entry in entries {
                dump.push(entry);
            }
            assert_eq!(
                ReplayHarness::new(dump).execute(&mut cleaner),
                expected,
                "{label}"
            );
        }
    }
}
