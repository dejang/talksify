mod content_score;
mod corpus;
mod dump;
mod file_source;
mod harness;
mod replay;
mod report;
mod score;

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use file_source::{FileSource, PlaybackState, WavAudio};
use harness::Harness;
use replay::ReplayHarness;
use talksify_core::Transcriber;

#[derive(Debug, PartialEq, Eq)]
enum Mode {
    Path {
        target: PathBuf,
        dump_raw: Option<PathBuf>,
    },
    Replay(PathBuf),
}

#[derive(Debug, PartialEq, Eq)]
struct Args {
    mode: Mode,
    models: PathBuf,
}

impl Args {
    fn parse(args: impl IntoIterator<Item = OsString>) -> Option<Self> {
        let mut args = args.into_iter();
        let mut target = None;
        let mut models = None;
        let mut dump_raw = None;
        let mut replay = None;
        while let Some(arg) = args.next() {
            if arg == "--models" && models.is_none() {
                models = Some(Self::next_flag_value(&mut args)?);
            } else if arg == "--dump-raw" && dump_raw.is_none() {
                dump_raw = Some(Self::next_flag_value(&mut args)?);
            } else if arg == "--replay" && replay.is_none() {
                replay = Some(Self::next_flag_value(&mut args)?);
            } else if !Self::starts_with_dash(&arg) && target.is_none() {
                target = Some(PathBuf::from(arg));
            } else {
                return None;
            }
        }
        let mode = match (target, dump_raw, replay) {
            (None, None, Some(path)) => Mode::Replay(path),
            (Some(target), dump_raw, None) => Mode::Path { target, dump_raw },
            _ => return None,
        };
        Some(Self {
            mode,
            models: models.unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../models")),
        })
    }

    fn execute(self) -> ExitCode {
        let result = match &self.mode {
            Mode::Replay(path) => ReplayHarness::run(path, &self.models),
            Mode::Path { target, dump_raw } if target.is_dir() => {
                Harness::run_folder(target, &self.models, dump_raw.as_deref())
            }
            Mode::Path {
                dump_raw: Some(_), ..
            } => Err("--dump-raw requires a folder of .wav/.txt pairs".into()),
            Mode::Path { target, .. } => self.transcribe_file(target).map(|transcript| {
                println!("{transcript}");
                ExitCode::SUCCESS
            }),
        };
        match result {
            Ok(code) => code,
            Err(error) => {
                eprintln!("error: {error}");
                ExitCode::FAILURE
            }
        }
    }

    fn transcribe_file(&self, file: &Path) -> Result<String, String> {
        WavAudio::inspect_format(file)?;
        let path = file.to_path_buf();
        let playback = Arc::new(PlaybackState::new());
        let source_playback = Arc::clone(&playback);
        let mut transcriber = Transcriber::new(
            &self.models,
            move || FileSource::new(path.clone(), Arc::clone(&source_playback)),
            Arc::new(|_| {}),
        )
        .map_err(|e| e.to_string())?;
        FileSource::play_into(&mut transcriber, &playback)?;
        transcriber
            .stop_transcribing()
            .map(|transcript| transcript.cleaned)
            .map_err(|e| e.to_string())
    }

    fn starts_with_dash(arg: &OsStr) -> bool {
        arg.as_encoded_bytes().first() == Some(&b'-')
    }

    fn next_flag_value(args: &mut impl Iterator<Item = OsString>) -> Option<PathBuf> {
        let value = args.next()?;
        if Self::starts_with_dash(&value) {
            return None;
        }
        Some(PathBuf::from(value))
    }
}

fn main() -> ExitCode {
    // Show harness reports without enabling verbose model/pipeline logs.
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,talksify_cli::harness=info"),
    )
    .target(env_logger::Target::Stderr)
    .init();

    let Some(args) = Args::parse(std::env::args_os().skip(1)) else {
        eprintln!(
            "usage:\n  talksify-cli <file.wav | folder> [--models <dir>] [--dump-raw <path.json>]\n  \
             talksify-cli --replay <path.json> [--models <dir>]"
        );
        return ExitCode::FAILURE;
    };
    args.execute()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn os_args(raw: &[&str]) -> Vec<OsString> {
        raw.iter().map(OsString::from).collect()
    }

    #[test]
    fn parses_valid_cli_arguments() {
        let default_models = Path::new(env!("CARGO_MANIFEST_DIR")).join("../models");
        let mut cases: Vec<(Vec<OsString>, Args)> = vec![
            (
                os_args(&["clip.wav"]),
                Args {
                    mode: Mode::Path {
                        target: PathBuf::from("clip.wav"),
                        dump_raw: None,
                    },
                    models: default_models.clone(),
                },
            ),
            (
                os_args(&["--dump-raw", "out.json", "recordings", "--models", "/m"]),
                Args {
                    mode: Mode::Path {
                        target: PathBuf::from("recordings"),
                        dump_raw: Some(PathBuf::from("out.json")),
                    },
                    models: PathBuf::from("/m"),
                },
            ),
            (
                os_args(&["--replay", "out.json"]),
                Args {
                    mode: Mode::Replay(PathBuf::from("out.json")),
                    models: default_models,
                },
            ),
            (
                os_args(&["--models", "/m", "--replay", "out.json"]),
                Args {
                    mode: Mode::Replay(PathBuf::from("out.json")),
                    models: PathBuf::from("/m"),
                },
            ),
        ];

        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStringExt;

            let models_dir = PathBuf::from(OsString::from_vec(b"/tmp/models-\xff".to_vec()));
            let replay_path = PathBuf::from(OsString::from_vec(b"/tmp/raw-\xff.json".to_vec()));
            let target_path = PathBuf::from(OsString::from_vec(b"/tmp/rec-\xff".to_vec()));
            let dump_path = PathBuf::from(OsString::from_vec(b"/tmp/dump-\xff.json".to_vec()));
            cases.push((
                vec![
                    OsString::from("--replay"),
                    replay_path.clone().into_os_string(),
                    OsString::from("--models"),
                    models_dir.clone().into_os_string(),
                ],
                Args {
                    mode: Mode::Replay(replay_path),
                    models: models_dir.clone(),
                },
            ));
            cases.push((
                vec![
                    target_path.clone().into_os_string(),
                    OsString::from("--dump-raw"),
                    dump_path.clone().into_os_string(),
                    OsString::from("--models"),
                    models_dir.clone().into_os_string(),
                ],
                Args {
                    mode: Mode::Path {
                        target: target_path,
                        dump_raw: Some(dump_path),
                    },
                    models: models_dir,
                },
            ));
        }

        for (raw, expected) in cases {
            assert_eq!(Args::parse(raw.clone()), Some(expected), "{raw:?}");
        }
    }

    #[test]
    fn rejects_invalid_cli_arguments() {
        let mut cases: Vec<Vec<OsString>> = [
            &[][..],
            &["--replay"],
            &["--replay", "--models", "/m"],
            &["--dump-raw", "out.json"],
            &["recordings", "--replay", "out.json"],
            &["--replay", "in.json", "--dump-raw", "out.json"],
            &["a.wav", "b.wav"],
            &["--models", "/a", "--models", "/b", "a.wav"],
            &["--unknown", "a.wav"],
        ]
        .into_iter()
        .map(os_args)
        .collect();

        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStringExt;
            cases.push(vec![
                OsString::from("--replay"),
                OsString::from_vec(b"-\xff".to_vec()),
            ]);
        }

        for bad in cases {
            assert_eq!(Args::parse(bad.clone()), None, "{bad:?}");
        }
    }
}
