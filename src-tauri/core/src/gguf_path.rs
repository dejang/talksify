use std::path::Path;

/// A validated local GGUF model directory and filename safe to pass to `mistralrs`.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct LocalGguf {
    dir: String,
    file: String,
}

impl LocalGguf {
    /// Selects and resolves the local GGUF model under `model_dir/qwen-0.6b`.
    pub(crate) fn locate(model_dir: &Path) -> Result<Option<Self>, String> {
        let llm_dir = model_dir.join("qwen-0.6b");
        match std::fs::metadata(&llm_dir) {
            Ok(meta) if meta.is_dir() => {}
            Ok(_) => {
                return Err(format!(
                    "model directory {} is not a directory",
                    llm_dir.display()
                ));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(format!(
                    "cannot inspect model directory {}: {error}",
                    llm_dir.display()
                ));
            }
        }

        let entries = std::fs::read_dir(&llm_dir).map_err(|error| {
            format!("cannot read model directory {}: {error}", llm_dir.display())
        })?;
        let mut candidates = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|error| {
                format!(
                    "cannot read entry in model directory {}: {error}",
                    llm_dir.display()
                )
            })?;
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "gguf") && !path.is_dir() {
                candidates.push(path);
            }
        }
        candidates.sort();

        let Some(chosen) = candidates.first() else {
            return Ok(None);
        };
        if candidates.len() > 1 {
            log::warn!(
                "multiple .gguf models in {}; selecting {}",
                llm_dir.display(),
                chosen.display()
            );
        }
        Self::resolve(chosen).map(Some)
    }

    pub(crate) fn dir(&self) -> &str {
        &self.dir
    }

    pub(crate) fn file(&self) -> &str {
        &self.file
    }

    /// Resolves a local GGUF path without lossy UTF-8 conversion or remote lookup fallback.
    fn resolve(gguf_path: &Path) -> Result<Self, String> {
        let resolved = std::fs::canonicalize(gguf_path)
            .map_err(|e| format!("cannot resolve local GGUF {}: {e}", gguf_path.display()))?;
        if !resolved.is_file() {
            return Err(format!(
                "resolved GGUF path {} is not a file",
                resolved.display()
            ));
        }
        let parent = resolved.parent().ok_or_else(|| {
            format!(
                "local GGUF path {} has no parent directory",
                resolved.display()
            )
        })?;
        let name = resolved
            .file_name()
            .ok_or_else(|| format!("local GGUF path {} has no file name", resolved.display()))?;
        let dir = parent.to_str().ok_or_else(|| {
            format!(
                "local GGUF directory {} is not valid UTF-8 and cannot be loaded",
                parent.display()
            )
        })?;
        let file = name.to_str().ok_or_else(|| {
            format!(
                "local GGUF filename {} is not valid UTF-8 and cannot be loaded",
                name.display()
            )
        })?;
        Ok(Self {
            dir: dir.to_owned(),
            file: file.to_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use tempfile::TempDir;

    use super::*;

    #[test]
    fn locate_succeeds_for_valid_and_absent_model_dirs() {
        let root = TempDir::new().unwrap();

        let empty_models = root.path().join("empty-models");
        fs::create_dir(&empty_models).unwrap();

        let empty_llm = root.path().join("empty-llm");
        fs::create_dir_all(empty_llm.join("qwen-0.6b/00-dir.gguf")).unwrap();

        let multi_models = root.path().join("multi-models");
        let multi_llm = multi_models.join("qwen-0.6b");
        fs::create_dir_all(&multi_llm).unwrap();
        fs::write(multi_llm.join("a-model.gguf"), b"gguf").unwrap();
        let multi_canonical = fs::canonicalize(&multi_llm)
            .unwrap()
            .to_str()
            .unwrap()
            .to_owned();
        fs::create_dir_all(multi_models.join("qwen-0.6b/00-dir.gguf")).unwrap();
        fs::write(multi_models.join("qwen-0.6b/z-model.gguf"), b"gguf").unwrap();
        fs::write(multi_models.join("qwen-0.6b/notes.txt"), b"notes").unwrap();

        let cases = [
            ("absent root", root.path().join("absent"), None),
            ("absent nested", root.path().join("absent/models"), None),
            ("empty models dir", empty_models, None),
            ("empty qwen-0.6b with .gguf dir", empty_llm, None),
            (
                "deterministic first .gguf",
                multi_models,
                Some((multi_canonical, "a-model.gguf".to_owned())),
            ),
        ];

        for (label, path, expected) in cases {
            let actual = LocalGguf::locate(&path)
                .map(|found| found.map(|gguf| (gguf.dir().to_owned(), gguf.file().to_owned())));
            assert_eq!(actual, Ok(expected), "{label}");
        }
    }

    #[test]
    fn locate_fails_for_invalid_model_paths() {
        let root = TempDir::new().unwrap();

        let file_as_llm = root.path().join("file-as-llm");
        fs::create_dir(&file_as_llm).unwrap();
        fs::write(file_as_llm.join("qwen-0.6b"), b"data").unwrap();

        let mut cases: Vec<(&str, PathBuf, &str)> =
            vec![("qwen-0.6b is a file", file_as_llm, "is not a directory")];

        #[cfg(unix)]
        {
            use std::ffi::OsString;
            use std::os::unix::ffi::OsStringExt;

            let bad_dir_models = root.path().join(OsString::from_vec(b"real-\xff".to_vec()));
            let bad_dir_llm = bad_dir_models.join("qwen-0.6b");
            fs::create_dir_all(&bad_dir_llm).unwrap();
            fs::write(bad_dir_llm.join("model.gguf"), b"gguf").unwrap();
            cases.push(("non-utf8 real dir", bad_dir_models, "UTF-8"));

            let bad_file_models = root.path().join("utf8-models");
            let bad_file_llm = bad_file_models.join("qwen-0.6b");
            fs::create_dir_all(&bad_file_llm).unwrap();
            fs::write(
                bad_file_llm.join(OsString::from_vec(b"model-\xff.gguf".to_vec())),
                b"gguf",
            )
            .unwrap();
            cases.push(("non-utf8 real file", bad_file_models, "UTF-8"));
        }

        for (label, path, expected_err) in cases {
            let err = LocalGguf::locate(&path).expect_err(&format!(
                "{label}: expected error containing {expected_err:?}"
            ));
            assert!(
                err.contains(expected_err),
                "{label}: expected {expected_err:?}, got {err:?}"
            );
        }
    }
}
