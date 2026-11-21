/// Raw and cleaned text from one recording.
#[derive(Debug, Default, Clone)]
pub struct Transcript {
    /// The STT segments joined with spaces, before any cleanup.
    pub raw: String,
    /// The final text after cleanup.
    pub cleaned: String,
}

#[derive(Debug, Default, Clone)]
pub(crate) struct RawTranscript {
    pub(crate) sentences: Vec<String>,
}

impl RawTranscript {
    pub(crate) fn from_text(text: &str) -> Self {
        let mut transcript = Self::default();
        if !text.is_empty() {
            transcript.push_words(text.into());
        }
        transcript
    }

    pub(crate) fn push_words(&mut self, words: String) {
        self.sentences.push(words);
    }

    pub(crate) fn to_joined_string(&self) -> String {
        self.sentences.join(" ")
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.sentences.is_empty()
    }

    pub(crate) fn clear(&mut self) {
        self.sentences.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulates_joins_and_clears_transcripts() {
        assert!(RawTranscript::from_text("").is_empty());
        let from_str = RawTranscript::from_text("hello world");
        assert!(!from_str.is_empty());
        assert_eq!(from_str.to_joined_string(), "hello world");

        let mut rt = RawTranscript::default();
        rt.push_words("hello".into());
        rt.push_words("world".into());
        assert_eq!(rt.to_joined_string(), "hello world");
        rt.clear();
        assert!(rt.is_empty());
    }
}
