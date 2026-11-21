#[cfg(test)]
mod cli_integration {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::process::{Command, Output};

    use tempfile::TempDir;

    struct CliEnv {
        dir: TempDir,
        recordings: PathBuf,
        replay_path: PathBuf,
    }

    impl CliEnv {
        fn new() -> Self {
            let dir = TempDir::new().unwrap();
            let recordings = dir.path().join("recordings");
            fs::create_dir(&recordings).unwrap();
            fs::write(recordings.join("sample.wav"), b"not-played").unwrap();
            fs::write(recordings.join("sample.txt"), "Hello world.").unwrap();

            let replay_path = dir.path().join("existing.json");
            fs::write(
                &replay_path,
                r#"[{"file":"sample.wav","duration_secs":3.0,"expected":"Hello world.","raw":"hello world."}]"#,
            )
            .unwrap();

            Self {
                dir,
                recordings,
                replay_path,
            }
        }

        fn command() -> Command {
            let mut cmd = Command::new(env!("CARGO_BIN_EXE_talksify-cli"));
            cmd.env_remove("RUST_LOG");
            cmd.env("RUST_LOG_STYLE", "never");
            cmd
        }

        #[cfg(unix)]
        fn symlink_bundled_stt_and_vad(models_dir: &Path) {
            let bundled = Path::new(env!("CARGO_MANIFEST_DIR")).join("../models");
            for sub in ["stt", "vad"] {
                let src = bundled.join(sub);
                if src.is_dir() {
                    std::os::unix::fs::symlink(&src, models_dir.join(sub)).unwrap();
                }
            }
        }

        fn root(&self) -> &Path {
            self.dir.path()
        }

        fn run_replay_file(replay_path: &Path, models_dir: &Path) -> Output {
            Self::command()
                .arg("--replay")
                .arg(replay_path)
                .arg("--models")
                .arg(models_dir)
                .output()
                .unwrap()
        }

        fn assert_single_error(
            case_name: &str,
            mode: &str,
            output: &Output,
            expected_error: &str,
        ) {
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                !output.status.success(),
                "[{case_name}] expected {mode} failure, got {:?}\nstderr:\n{stderr}",
                output.status
            );
            assert!(
                !stderr.contains("failed to load STT model"),
                "[{case_name}] expected LLM startup error before STT load in {mode} mode, got:\n{stderr}"
            );
            assert_eq!(
                stderr.matches(expected_error).count(),
                1,
                "[{case_name}] expected {mode} error containing {expected_error:?} once, got:\n{stderr}"
            );
            assert!(
                !stderr.contains(" ERROR "),
                "[{case_name}] expected no duplicate log::error! output in {mode} mode, got:\n{stderr}"
            );
        }

        fn assert_strict_startup_fails(
            &self,
            case_name: &str,
            models_dir: &Path,
            expected_error: &str,
        ) {
            let folder_output = Self::command()
                .arg(&self.recordings)
                .arg("--models")
                .arg(models_dir)
                .output()
                .unwrap();
            Self::assert_single_error(case_name, "folder", &folder_output, expected_error);

            let out_dir = TempDir::new_in(self.dir.path()).unwrap();
            let dump_path = out_dir.path().join("raw.json");
            let dump_output = Self::command()
                .arg(&self.recordings)
                .arg("--dump-raw")
                .arg(&dump_path)
                .arg("--models")
                .arg(models_dir)
                .output()
                .unwrap();
            Self::assert_single_error(case_name, "dump", &dump_output, expected_error);
            assert!(
                !dump_path.exists(),
                "[{case_name}] failed dump must not publish output"
            );
            assert_eq!(
                fs::read_dir(out_dir.path()).unwrap().count(),
                0,
                "[{case_name}] failed dump must not leave staging files behind"
            );

            let replay_output = Self::run_replay_file(&self.replay_path, models_dir);
            Self::assert_single_error(case_name, "replay", &replay_output, expected_error);
        }

        fn assert_replay_succeeds(case_name: &str, replay_path: &Path, models_dir: &Path) {
            let output = Self::run_replay_file(replay_path, models_dir);
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                output.status.success(),
                "[{case_name}] expected exit 0, got {:?}\nstderr:\n{stderr}",
                output.status
            );
            assert_eq!(
                stderr.matches("basic cleanup only").count(),
                1,
                "[{case_name}] expected single LLM-unavailable warning, got:\n{stderr}"
            );
        }
    }

    #[test]
    fn cli_binary_enforces_strict_model_startup_and_fallback_baselines() {
        let env = CliEnv::new();

        // Failure matrix: invalid GGUF files and non-UTF-8 model paths.
        let invalid_models = env.root().join("invalid-models");
        let invalid_llm = invalid_models.join("qwen-0.6b");
        fs::create_dir_all(&invalid_llm).unwrap();
        #[cfg(unix)]
        CliEnv::symlink_bundled_stt_and_vad(&invalid_models);
        fs::write(invalid_llm.join("model.gguf"), b"not-a-valid-gguf").unwrap();

        let mut failure_cases: Vec<(&str, PathBuf, &str)> = vec![(
            "invalid-gguf",
            invalid_models,
            "invalid GGUF magic",
        )];

        // Success matrix: intentional no-GGUF baselines and non-UTF-8 replay path.
        let empty_qwen = env.root().join("empty-qwen");
        fs::create_dir_all(empty_qwen.join("qwen-0.6b")).unwrap();
        let no_qwen = env.root().join("no-qwen");
        fs::create_dir(&no_qwen).unwrap();
        let mut success_cases: Vec<(&str, PathBuf, PathBuf)> = vec![
            ("empty-qwen", env.replay_path.clone(), empty_qwen),
            ("no-qwen", env.replay_path.clone(), no_qwen.clone()),
            (
                "absent-nested",
                env.replay_path.clone(),
                env.root().join("absent/models"),
            ),
        ];

        #[cfg(unix)]
        {
            use std::ffi::OsString;
            use std::os::unix::ffi::OsStringExt;

            let bad_dir_models = env
                .root()
                .join(OsString::from_vec(b"models-\xff".to_vec()));
            let bad_dir_llm = bad_dir_models.join("qwen-0.6b");
            fs::create_dir_all(&bad_dir_llm).unwrap();
            fs::write(bad_dir_llm.join("model.gguf"), b"gguf").unwrap();
            failure_cases.push((
                "non-utf8-dir",
                bad_dir_models,
                "is not valid UTF-8 and cannot be loaded",
            ));

            let bad_file_models = env.root().join("utf8-models");
            let bad_file_llm = bad_file_models.join("qwen-0.6b");
            fs::create_dir_all(&bad_file_llm).unwrap();
            fs::write(
                bad_file_llm.join(OsString::from_vec(b"model-\xff.gguf".to_vec())),
                b"gguf",
            )
            .unwrap();
            failure_cases.push((
                "non-utf8-file",
                bad_file_models,
                "is not valid UTF-8 and cannot be loaded",
            ));

            let non_utf8_replay = env
                .root()
                .join(OsString::from_vec(b"raw-\xff.json".to_vec()));
            fs::copy(&env.replay_path, &non_utf8_replay).unwrap();
            success_cases.push(("non-utf8-replay-path", non_utf8_replay, no_qwen));
        }

        for (case_name, models_dir, expected_error) in failure_cases {
            env.assert_strict_startup_fails(case_name, &models_dir, expected_error);
        }
        for (case_name, replay_path, models_dir) in success_cases {
            CliEnv::assert_replay_succeeds(case_name, &replay_path, &models_dir);
        }
    }

    #[test]
    fn cli_binary_quotes_commas_and_escapes_newlines_in_replay_report() {
        let dir = TempDir::new().unwrap();
        let models_dir = dir.path().join("empty-models");
        fs::create_dir(&models_dir).unwrap();

        let two_files_json = dir.path().join("two.json");
        fs::write(
            &two_files_json,
            r#"[
              {"file":"a.wav","duration_secs":3.0,"expected":"hello world","raw":"hello world"},
              {"file":"b.wav","duration_secs":4.0,"expected":"hello world","raw":"hello world"}
            ]"#,
        )
        .unwrap();

        let comma_and_newline_json = dir.path().join("tricky.json");
        fs::write(
            &comma_and_newline_json,
            r#"[
              {"file":"a.wav, b.wav","duration_secs":3.0,"expected":"hello world","raw":"hello world"},
              {"file":"line1\nline2.wav","duration_secs":4.0,"expected":"hello world","raw":"hello world"}
            ]"#,
        )
        .unwrap();

        let two_out = CliEnv::command()
            .arg("--replay")
            .arg(&two_files_json)
            .arg("--models")
            .arg(&models_dir)
            .output()
            .unwrap();
        assert!(two_out.status.success());
        let two_stderr = String::from_utf8_lossy(&two_out.stderr);
        assert!(
            two_stderr.contains(r#"exact worse      "a.wav", "b.wav""#),
            "unexpected stderr:\n{two_stderr}"
        );

        let tricky_out = CliEnv::command()
            .arg("--replay")
            .arg(&comma_and_newline_json)
            .arg("--models")
            .arg(&models_dir)
            .output()
            .unwrap();
        assert!(tricky_out.status.success());
        let tricky_stderr = String::from_utf8_lossy(&tricky_out.stderr);
        assert!(
            tricky_stderr.contains(r#""line1\nline2.wav"  (4.0 s)"#),
            "per-file header must stay single-line and quoted:\n{tricky_stderr}"
        );
        assert!(
            tricky_stderr.contains(r#"exact worse      "a.wav, b.wav", "line1\nline2.wav""#),
            "exact worse must distinguish comma filename and escape newline:\n{tricky_stderr}"
        );
        assert!(
            !tricky_stderr.lines().any(|line| line.starts_with("line2.wav")),
            "newline filename split across stderr lines:\n{tricky_stderr}"
        );
    }
}
