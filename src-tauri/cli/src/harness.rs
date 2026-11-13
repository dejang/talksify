use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use talksify_core::{InputFormat, Transcriber};

use crate::file_source::{FileSource, read_wav};
use crate::{play_file, score};

type WavPair = (PathBuf, PathBuf);

/// The WAV/TXT pairs in this folder, sorted by name, and WAVs without a TXT.
pub(crate) fn pairs(folder: &Path) -> std::io::Result<(Vec<WavPair>, Vec<PathBuf>)> {
    let mut paired = Vec::new();
    let mut skipped = Vec::new();
    for entry in fs::read_dir(folder)? {
        let path = entry?.path();
        if path.extension().is_some_and(|ext| ext == "wav") && path.is_file() {
            let text = path.with_extension("txt");
            if text.is_file() {
                paired.push((path, text));
            } else {
                skipped.push(path);
            }
        }
    }
    paired.sort();
    skipped.sort();
    Ok((paired, skipped))
}

pub(crate) fn run(folder: &Path, models: &Path) -> ExitCode {
    let (paired, skipped) = match pairs(folder) {
        Ok(pairs) => pairs,
        Err(error) => {
            log::error!("error: cannot read {}: {error}", folder.display());
            return ExitCode::FAILURE;
        }
    };
    if paired.is_empty() {
        log::error!(
            "error: no .wav files with a matching .txt in {}",
            folder.display()
        );
        return ExitCode::FAILURE;
    }

    let audio = Arc::new(Mutex::new((
        Arc::<[f32]>::from([]),
        InputFormat {
            sample_rate: 16_000,
            channels: 1,
        },
    )));
    let finished = Arc::new(AtomicBool::new(false));
    let source_audio = Arc::clone(&audio);
    let source_finished = Arc::clone(&finished);
    let mut transcriber = match Transcriber::new(
        models,
        move || {
            let (samples, format) = &*source_audio.lock().unwrap_or_else(PoisonError::into_inner);
            FileSource::new(Arc::clone(samples), *format, Arc::clone(&source_finished))
        },
        Arc::new(|_| {}),
    ) {
        Ok(transcriber) => transcriber,
        Err(error) => {
            log::error!("error: {error}");
            return ExitCode::FAILURE;
        }
    };

    let mut raw_errors = score::Errors::default();
    let mut cleaned_errors = score::Errors::default();
    let mut total_expected = 0;
    let mut waits = Vec::new();
    let mut worse = Vec::new();
    let mut failed = false;
    for (wav, text) in paired {
        let name = wav.file_name().unwrap_or_default().to_string_lossy();
        let result = (|| {
            let data = read_wav(&wav).map_err(|e| format!("cannot read {}: {e}", wav.display()))?;
            let expected = fs::read_to_string(&text)
                .map_err(|e| format!("cannot read {}: {e}", text.display()))?;
            *audio.lock().unwrap_or_else(PoisonError::into_inner) = data;
            finished.store(false, Ordering::Release);
            play_file(&mut transcriber, &finished)?;
            let stopped = Instant::now();
            let output = transcriber.stop_transcribing().map_err(|e| e.to_string())?;
            let wait = stopped.elapsed();
            Ok::<_, String>((expected, output, wait))
        })();
        match result {
            Ok((expected, output, wait)) => {
                let expected = expected.trim();
                let raw = output.raw.trim();
                let cleaned = output.cleaned.trim();
                let expected_tokens = score::tokens(expected);
                let raw_score = score::errors(&expected_tokens, &score::tokens(raw));
                let cleaned_score = score::errors(&expected_tokens, &score::tokens(cleaned));
                let tokens = expected_tokens.len();
                log::info!("{name}  wait {:.1} s", wait.as_secs_f64());
                report("  raw    ", &raw_score, tokens);
                report("  cleaned", &cleaned_score, tokens);
                log::info!("  expected: {expected:?}");
                log::info!("  raw:      {raw:?}");
                log::info!("  got:      {cleaned:?}");
                if cleaned_score.total() > raw_score.total() {
                    worse.push(name.to_string());
                }
                raw_errors.accumulate(&raw_score);
                cleaned_errors.accumulate(&cleaned_score);
                total_expected += tokens;
                waits.push((wait, name.to_string()));
            }
            Err(error) => {
                log::error!("{name}  failed: {error}");
                failed = true;
            }
        }
    }
    report_overall(&raw_errors, &cleaned_errors, total_expected, &waits, &worse);
    if !skipped.is_empty() {
        let names: Vec<_> = skipped
            .iter()
            .map(|path| path.file_name().unwrap_or_default().to_string_lossy())
            .collect();
        log::info!("skipped, no .txt: {}", names.join(", "));
    }
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

fn report_overall(
    raw: &score::Errors,
    cleaned: &score::Errors,
    tokens: usize,
    waits: &[(Duration, String)],
    worse: &[String],
) {
    report("overall raw    ", raw, tokens);
    report("overall cleaned", cleaned, tokens);
    if let Some((longest, name)) = waits.iter().max_by_key(|(wait, _)| *wait) {
        let sum: Duration = waits.iter().map(|(wait, _)| *wait).sum();
        #[expect(clippy::cast_precision_loss, reason = "practical file counts fit f64")]
        let average = sum.as_secs_f64() / waits.len() as f64;
        log::info!(
            "wait             average {average:.1} s, longest {:.1} s ({name})",
            longest.as_secs_f64()
        );
    }
    if !worse.is_empty() {
        log::info!("cleanup made these worse: {}", worse.join(", "));
    }
}

fn report(label: &str, errors: &score::Errors, tokens: usize) {
    log::info!(
        "{label}  {:.1}%  ({}/{tokens})  case {}  punctuation {}  line breaks {}  words {}",
        score::rate(errors.total(), tokens),
        errors.total(),
        errors.case,
        errors.punctuation,
        errors.line_breaks,
        errors.words
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    struct TempFolder(PathBuf);

    impl TempFolder {
        fn new() -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let folder = std::env::temp_dir().join(format!(
                "talksify-harness-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&folder).unwrap();
            Self(folder)
        }
    }

    impl Drop for TempFolder {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn pairs_only_matching_files_in_name_order_without_recursing() {
        let folder = TempFolder::new();
        fs::create_dir(folder.0.join("sub")).unwrap();
        fs::create_dir(folder.0.join("directory.wav")).unwrap();
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
            fs::write(folder.0.join(name), "").unwrap();
        }
        let (paired, skipped) = pairs(&folder.0).unwrap();
        assert_eq!(
            paired,
            vec![
                (folder.0.join("a.wav"), folder.0.join("a.txt")),
                (folder.0.join("b.wav"), folder.0.join("b.txt")),
            ]
        );
        assert_eq!(skipped, vec![folder.0.join("c.wav")]);
    }

    #[test]
    fn empty_folder_and_unreadable_folder() {
        let folder = TempFolder::new();
        assert_eq!(pairs(&folder.0).unwrap(), (vec![], vec![]));
        assert!(pairs(&folder.0.join("missing")).is_err());
        assert_eq!(run(&folder.0, &folder.0), ExitCode::FAILURE);
    }
}
