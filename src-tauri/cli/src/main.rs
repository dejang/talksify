mod file_source;

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use file_source::{FileSource, read_wav};
use talksify_core::Transcriber;

fn arguments() -> Option<(PathBuf, PathBuf)> {
    let mut args = std::env::args().skip(1);
    let mut file = None;
    let mut models = None;
    while let Some(arg) = args.next() {
        if arg == "--models" && models.is_none() {
            let dir = args.next()?;
            if dir.starts_with('-') {
                return None;
            }
            models = Some(PathBuf::from(dir));
        } else if !arg.starts_with('-') && file.is_none() {
            file = Some(PathBuf::from(arg));
        } else {
            return None;
        }
    }
    Some((
        file?,
        models.unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../models")),
    ))
}

fn run(file: &Path, models: &Path) -> Result<String, String> {
    let (samples, format) =
        read_wav(file).map_err(|e| format!("cannot read {}: {e}", file.display()))?;
    let finished = Arc::new(AtomicBool::new(false));
    let source_finished = Arc::clone(&finished);
    let mut transcriber = Transcriber::new(
        models,
        move || FileSource::new(Arc::clone(&samples), format, Arc::clone(&source_finished)),
        Arc::new(|_| {}),
    )
    .map_err(|e| e.to_string())?;
    transcriber
        .start_transcribing(None)
        .map_err(|e| e.to_string())?;
    while !finished.load(Ordering::Acquire) {
        if !transcriber.is_transcribing() {
            return Err("audio capture stopped before the end of the file".into());
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    transcriber.stop_transcribing().map_err(|e| e.to_string())
}

fn main() -> ExitCode {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .target(env_logger::Target::Stderr)
        .init();
    let Some((file, models)) = arguments() else {
        eprintln!("talksify-cli <file.wav> [--models <dir>]");
        return ExitCode::FAILURE;
    };
    match run(&file, &models) {
        Ok(transcript) => {
            println!("{transcript}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}
