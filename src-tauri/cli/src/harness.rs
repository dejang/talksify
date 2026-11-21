use std::borrow::Cow;
use std::path::Path;
use std::process::ExitCode;

use talksify_core::TextCleaner;

use crate::corpus::{RecordingFolder, TranscribedSample, WavPair, WavRunner};
use crate::dump::{RawEntry, RecordingName, StagedDump};
use crate::report::{FileSample, HarnessReport};

/// Corpus item evaluated by [`Harness`] with an unambiguous report/dump name.
pub(crate) trait SampleItem {
    fn sample_name(&self) -> Cow<'_, RecordingName>;
}

/// Evaluates one corpus item (a WAV/TXT pair or dumped raw entry) into a scored sample.
pub(crate) trait SampleRunner<T> {
    fn run_sample(&mut self, item: &T) -> Result<TranscribedSample, String>;
}

struct FolderPlan {
    corpus: RecordingFolder,
    staged_dump: Option<StagedDump>,
}

impl FolderPlan {
    fn prepare(folder: &Path, dump_raw: Option<&Path>) -> Result<Self, String> {
        let corpus = RecordingFolder::scan(folder)
            .map_err(|error| format!("cannot read {}: {error}", folder.display()))?;
        if corpus.paired.is_empty() {
            return Err(format!(
                "no .wav files with a matching .txt in {}",
                folder.display()
            ));
        }
        let staged_dump = dump_raw.map(StagedDump::prepare).transpose()?;
        Ok(Self {
            corpus,
            staged_dump,
        })
    }

    fn execute(self, runner: &mut impl SampleRunner<WavPair>) -> ExitCode {
        let mut harness = Harness::default();
        harness.record_skipped(self.corpus.skipped_names());
        harness.evaluate(&self.corpus.paired, runner, self.staged_dump);
        harness.finish()
    }
}

/// Shared harness state and folder WAV/TXT evaluation runner.
#[derive(Default)]
pub(crate) struct Harness {
    report: HarnessReport,
    failed: bool,
}

impl Harness {
    pub(crate) fn run_folder(
        folder: &Path,
        models: &Path,
        dump_raw: Option<&Path>,
    ) -> Result<ExitCode, String> {
        let plan = FolderPlan::prepare(folder, dump_raw)?;
        let cleaner = TextCleaner::try_new(models)?;
        let mut runner = WavRunner::new(models, cleaner)?;
        Ok(plan.execute(&mut runner))
    }

    pub(crate) fn record_skipped(&mut self, files: impl IntoIterator<Item = RecordingName>) {
        self.report.record_skipped(files);
    }

    pub(crate) fn evaluate<T: SampleItem>(
        &mut self,
        items: &[T],
        runner: &mut impl SampleRunner<T>,
        mut staged_dump: Option<StagedDump>,
    ) {
        for item in items {
            let name = item.sample_name();
            match runner.run_sample(item) {
                Ok(sample) => {
                    self.record_sample(&name, &sample);
                    if let Some(staged) = staged_dump.as_mut() {
                        staged.push(RawEntry {
                            file: name.into_owned(),
                            duration_secs: sample.duration_secs,
                            expected: sample.expected.trim().to_string(),
                            raw: sample.transcript.raw,
                        });
                    }
                }
                Err(error) => self.record_failure(&name, &error),
            }
        }
        if let Some(staged) = staged_dump {
            self.publish_dump(staged);
        }
    }

    pub(crate) fn finish(self) -> ExitCode {
        for line in self.report.to_string().lines() {
            log::info!("{line}");
        }
        if self.failed {
            ExitCode::FAILURE
        } else {
            ExitCode::SUCCESS
        }
    }

    fn record_sample(&mut self, name: &RecordingName, sample: &TranscribedSample) {
        let entry = self.report.record(&FileSample {
            name,
            duration_secs: sample.duration_secs,
            expected: &sample.expected,
            raw: &sample.transcript.raw,
            cleaned: &sample.transcript.cleaned,
            wait: sample.wait,
        });
        for line in entry.lines() {
            log::info!("{line}");
        }
    }

    fn record_failure(&mut self, name: &RecordingName, error: &str) {
        log::error!("{}  failed: {error}", name.quoted());
        self.failed = true;
    }

    fn publish_dump(&mut self, staged: StagedDump) {
        let destination = staged.destination().to_path_buf();
        if self.failed {
            log::error!(
                "skipping raw dump to {}: not all paired recordings succeeded",
                destination.display()
            );
        } else {
            match staged.publish() {
                Ok(count) => log::info!(
                    "wrote {count} raw transcripts to {}",
                    destination.display()
                ),
                Err(error) => {
                    log::error!("error: {error}");
                    self.failed = true;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::Duration;

    use talksify_core::Transcript;
    use tempfile::TempDir;

    use super::*;
    use crate::dump::RawDump;
    use crate::replay::ReplayHarness;

    struct FakeCorpus {
        dir: TempDir,
    }

    impl FakeCorpus {
        fn new() -> Self {
            Self {
                dir: TempDir::new().unwrap(),
            }
        }

        fn path(&self) -> &Path {
            self.dir.path()
        }

        fn add_pair(&self, stem: &str, reference: &str) -> PathBuf {
            let wav = self.dir.path().join(format!("{stem}.wav"));
            let txt = self.dir.path().join(format!("{stem}.txt"));
            fs::write(&wav, b"").unwrap();
            fs::write(&txt, reference).unwrap();
            txt
        }

        fn sorted_names(dir: &Path) -> Vec<RecordingName> {
            let mut names: Vec<_> = fs::read_dir(dir)
                .unwrap()
                .map(|entry| RecordingName::from_path(&entry.unwrap().path()))
                .collect();
            names.sort();
            names
        }
    }

    struct StubRunner {
        fail_file: Option<String>,
        fail_error: String,
        race_create: Option<PathBuf>,
    }

    impl Default for StubRunner {
        fn default() -> Self {
            Self {
                fail_file: None,
                fail_error: "corrupt wav".into(),
                race_create: None,
            }
        }
    }

    impl SampleRunner<WavPair> for StubRunner {
        fn run_sample(&mut self, pair: &WavPair) -> Result<TranscribedSample, String> {
            if let Some(race) = &self.race_create {
                fs::write(race, "created concurrently").unwrap();
            }
            if let Some(fail) = &self.fail_file
                && pair.wav.ends_with(fail)
            {
                return Err(self.fail_error.clone());
            }
            let expected = fs::read_to_string(&pair.text).unwrap();
            let raw = expected.clone();
            let cleaned = expected.clone();
            Ok(TranscribedSample {
                duration_secs: 5.0,
                expected,
                transcript: Transcript { raw, cleaned },
                wait: Duration::from_millis(25),
            })
        }
    }

    #[test]
    fn folder_dump_publishes_only_when_complete_and_aborts_on_errors() {
        // Empty folder fails.
        let empty = TempDir::new().unwrap();
        assert!(Harness::run_folder(empty.path(), empty.path(), None).is_err());

        let corpus = FakeCorpus::new();
        let original = "Original reference transcript.\n";
        let reference = corpus.add_pair("good", original);
        corpus.add_pair("bad", "Second.");
        let out_dir = TempDir::new().unwrap();

        // Startup failure cleans up staging file and never publishes.
        let bad_models = TempDir::new().unwrap();
        let llm_dir = bad_models.path().join("qwen-0.6b");
        fs::create_dir(&llm_dir).unwrap();
        fs::write(llm_dir.join("model.gguf"), b"invalid-gguf").unwrap();
        let startup_dump = out_dir.path().join("startup.json");
        assert!(
            Harness::run_folder(corpus.path(), bad_models.path(), Some(&startup_dump)).is_err()
        );
        assert!(!startup_dump.exists());

        // Invalid destination paths fail before startup without modifying existing files.
        let missing_parent = corpus.path().join("missing-dir/raw.json");
        let file_as_parent = reference.join("raw.json");
        let trailing_slash = PathBuf::from(format!("{}/", corpus.path().join("raw.json").display()));
        let trailing_dot = PathBuf::from(format!("{}/.", corpus.path().join("raw.json").display()));
        for bad_dest in [
            &missing_parent,
            &file_as_parent,
            &trailing_slash,
            &trailing_dot,
            &reference,
        ] {
            assert!(
                FolderPlan::prepare(corpus.path(), Some(bad_dest)).is_err(),
                "{}",
                bad_dest.display()
            );
            assert!(
                Harness::run_folder(corpus.path(), bad_models.path(), Some(bad_dest)).is_err(),
                "{}",
                bad_dest.display()
            );
        }
        assert_eq!(fs::read_to_string(&reference).unwrap(), original);

        // Pair failure or cleanup error aborts publication and leaves no staging files behind.
        for (file, fail_error) in [
            ("incomplete.json", "corrupt wav"),
            (
                "strict.json",
                "stop failed: LLM inference failed: runtime error",
            ),
        ] {
            let dest = out_dir.path().join(file);
            let plan = FolderPlan::prepare(corpus.path(), Some(&dest)).unwrap();
            let mut runner = StubRunner {
                fail_file: Some("bad.wav".into()),
                fail_error: fail_error.into(),
                ..StubRunner::default()
            };
            assert_eq!(plan.execute(&mut runner), ExitCode::FAILURE);
            assert!(!dest.exists());
        }
        assert_eq!(
            FakeCorpus::sorted_names(out_dir.path()),
            Vec::<RecordingName>::new()
        );

        // Concurrent destination creation fails publish without clobbering.
        let raced = out_dir.path().join("raced.json");
        let plan = FolderPlan::prepare(corpus.path(), Some(&raced)).unwrap();
        let mut race_runner = StubRunner {
            race_create: Some(raced.clone()),
            ..StubRunner::default()
        };
        assert_eq!(plan.execute(&mut race_runner), ExitCode::FAILURE);
        assert_eq!(fs::read_to_string(&raced).unwrap(), "created concurrently");

        // All pairs succeeding publishes the complete sorted dump.
        let complete_dump = out_dir.path().join("complete.json");
        let plan = FolderPlan::prepare(corpus.path(), Some(&complete_dump)).unwrap();
        let mut ok_runner = StubRunner::default();
        assert_eq!(plan.execute(&mut ok_runner), ExitCode::SUCCESS);
        let dump = RawDump::read(&complete_dump).unwrap();
        let names: Vec<_> = dump.entries().iter().map(|e| e.file.as_str()).collect();
        assert_eq!(names, vec!["bad.wav", "good.wav"]);
    }

    #[cfg(unix)]
    #[test]
    fn comma_and_newline_filenames_round_trip_through_dump_and_replay() {
        let corpus = FakeCorpus::new();
        let out_dir = TempDir::new().unwrap();
        let dump_path = out_dir.path().join("raw.json");

        corpus.add_pair("a.wav, b", "Comma.");
        corpus.add_pair("line1\nline2", "Multiline.");
        fs::write(corpus.path().join("skip\nme, too.wav"), b"").unwrap();

        let plan = FolderPlan::prepare(corpus.path(), Some(&dump_path)).unwrap();
        let mut runner = StubRunner::default();
        assert_eq!(plan.execute(&mut runner), ExitCode::SUCCESS);

        let dump = RawDump::read(&dump_path).unwrap();
        let names: Vec<_> = dump.entries().iter().map(|e| e.file.as_str()).collect();
        assert_eq!(names, vec!["a.wav, b.wav", "line1\nline2.wav"]);
        assert_eq!(
            ReplayHarness::run(&dump_path, out_dir.path()),
            Ok(ExitCode::SUCCESS)
        );
    }
}
