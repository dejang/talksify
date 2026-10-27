#[derive(Debug, Default, Clone)]
pub(crate) struct RawTranscript {
    pub(crate) sentences: Vec<String>,
}

impl RawTranscript {
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
    fn push_and_join() {
        let mut rt = RawTranscript::default();
        rt.push_words("hello".into());
        rt.push_words("world".into());
        assert_eq!(rt.to_joined_string(), "hello world");
    }

    #[test]
    fn clear_resets() {
        let mut rt = RawTranscript::default();
        rt.push_words("something".into());
        assert!(!rt.is_empty());
        rt.clear();
        assert!(rt.is_empty());
    }
}
