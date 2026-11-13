use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use talksify_core::{InputFormat, Transcriber};

use crate::file_source::{FileSource, read_wav};
use crate::{score, transcribe};

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

    let mut total_errors = 0;
    let mut total_expected = 0;
    let mut count = 0;
    let mut failed = false;
    for (wav, text) in paired {
        let name = wav.file_name().unwrap_or_default().to_string_lossy();
        let result = (|| {
            let data = read_wav(&wav).map_err(|e| format!("cannot read {}: {e}", wav.display()))?;
            let expected = fs::read_to_string(&text)
                .map_err(|e| format!("cannot read {}: {e}", text.display()))?;
            *audio.lock().unwrap_or_else(PoisonError::into_inner) = data;
            finished.store(false, Ordering::Release);
            let output = transcribe(&mut transcriber, &finished)?;
            Ok::<_, String>((expected, output))
        })();
        match result {
            Ok((expected, output)) => {
                let expected = expected.trim();
                let output = output.trim();
                let expected_tokens = score::tokens(expected);
                let errors = score::errors(&expected_tokens, &score::tokens(output));
                let tokens = expected_tokens.len();
                log::info!(
                    "{name}  {:.1}%  ({errors}/{tokens})",
                    score::rate(errors, tokens)
                );
                log::info!("  expected: {expected:?}");
                log::info!("  got:      {output:?}");
                total_errors += errors;
                total_expected += tokens;
                count += 1;
            }
            Err(error) => {
                log::error!("{name}  failed: {error}");
                failed = true;
            }
        }
    }
    log::info!(
        "overall: {:.1}%  ({total_errors}/{total_expected}) over {count} files",
        score::rate(total_errors, total_expected)
    );
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
