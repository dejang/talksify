use std::path::Path;

use crate::llm_postprocessor::LlmPostProcessor;
use crate::raw_transcript::RawTranscript;

/// Cleans raw speech-to-text transcripts using the local LLM and formatting rules.
pub struct TextCleaner {
    llm: LlmPostProcessor,
}

impl TextCleaner {
    /// Loads and warms the LLM from `model_dir`, falling back to rule-based cleanup if missing.
    #[must_use]
    pub fn new(model_dir: &Path) -> Self {
        let mut llm = LlmPostProcessor::new(model_dir);
        match llm.warm_up() {
            Ok(()) => Self::from_warmed(llm),
            Err(error) => {
                log::error!("{error}");
                Self { llm }
            }
        }
    }

    /// Loads and warms the LLM from `model_dir`, failing if a local model cannot be loaded.
    /// # Errors
    /// Returns an error if resolving, loading, or warming a local LLM failed.
    pub fn try_new(model_dir: &Path) -> Result<Self, String> {
        let mut llm = LlmPostProcessor::new(model_dir);
        llm.warm_up()?;
        Ok(Self::from_warmed(llm))
    }

    fn from_warmed(llm: LlmPostProcessor) -> Self {
        if !llm.is_ready() {
            log::warn!("[text-cleaner] LLM unavailable; transcripts get basic cleanup only");
        }
        Self { llm }
    }

    /// Whether the local LLM loaded and warmed successfully.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.llm.is_ready()
    }

    /// Cleans a raw transcript string through the same pipeline used by [`crate::Transcriber`].
    #[must_use]
    pub fn clean(&self, raw: &str) -> String {
        let transcript = RawTranscript::from_text(raw);
        self.llm.clean_transcript(&transcript)
    }

    /// Cleans a raw transcript string, propagating local LLM load or inference errors.
    /// # Errors
    /// Returns an error if the local LLM failed to load or failed during inference.
    pub fn try_clean(&self, raw: &str) -> Result<String, String> {
        let transcript = RawTranscript::from_text(raw);
        self.llm.try_clean_transcript(&transcript)
    }

    pub(crate) fn into_llm(self) -> LlmPostProcessor {
        self.llm
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_cleans_text_without_model() {
        let cleaner = TextCleaner::try_new(Path::new("/nonexistent-talksify-models"))
            .unwrap_or_else(|error| panic!("unexpected load error: {error}"));
        assert!(!cleaner.is_ready());
        for (input, expected) in [
            ("", ""),
            ("   ", ""),
            (
                "um hello world. twelve files in dot wav.",
                "Hello world. 12 files in .wav.",
            ),
        ] {
            assert_eq!(cleaner.clean(input), expected, "{input:?}");
            assert_eq!(cleaner.try_clean(input), Ok(expected.into()), "{input:?}");
        }
    }

    #[test]
    fn reports_load_errors_while_preserving_fallback_clean() {
        let root = tempfile::TempDir::new().unwrap();

        let invalid_models = root.path().join("invalid-gguf");
        let invalid_llm = invalid_models.join("qwen-0.6b");
        std::fs::create_dir_all(&invalid_llm).unwrap();
        std::fs::write(invalid_llm.join("model.gguf"), b"not-a-valid-gguf").unwrap();

        let mut cases = vec![("invalid gguf", invalid_models, "invalid GGUF magic")];

        #[cfg(unix)]
        {
            use std::ffi::OsString;
            use std::os::unix::ffi::OsStringExt;

            let non_utf8_models = root
                .path()
                .join(OsString::from_vec(b"models-\xff".to_vec()));
            let non_utf8_llm = non_utf8_models.join("qwen-0.6b");
            std::fs::create_dir_all(&non_utf8_llm).unwrap();
            std::fs::write(non_utf8_llm.join("model.gguf"), b"gguf").unwrap();
            cases.push(("non-utf8 path", non_utf8_models, "UTF-8"));
        }

        for (label, models_path, expected_err) in cases {
            let cleaner = TextCleaner::new(&models_path);
            assert!(!cleaner.is_ready(), "{label}");
            let Err(try_new_err) = TextCleaner::try_new(&models_path) else {
                panic!("{label}: expected try_new error");
            };
            assert!(
                try_new_err.contains(expected_err),
                "{label}: expected {expected_err:?}, got {try_new_err:?}"
            );
            assert_eq!(cleaner.clean("um hello world."), "Hello world.", "{label}");
            let try_err = cleaner.try_clean("um hello world.").unwrap_err();
            assert!(
                try_err.contains(expected_err),
                "{label}: expected {expected_err:?}, got {try_err:?}"
            );
        }
    }
}
