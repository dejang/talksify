use std::fmt;

use crate::score::{Edit, ErrorRate};

/// Normalized word-only edit distance classified into substitutions, deletions, and insertions.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct ContentErrors {
    pub(crate) substitutions: usize,
    pub(crate) deletions: usize,
    pub(crate) insertions: usize,
}

impl ContentErrors {
    /// Splits lowercased text into content words.
    /// Drops punctuation and line breaks, strips intra-word apostrophes, and splits hyphens.
    pub(crate) fn tokenize(text: &str) -> Vec<String> {
        let lower = text.to_lowercase();
        let mut words = Vec::new();
        let mut current = String::new();
        let mut chars = lower.chars().peekable();
        while let Some(ch) = chars.next() {
            if Self::is_content_char(ch) {
                current.push(ch);
            } else if Self::is_apostrophe(ch)
                && !current.is_empty()
                && chars.peek().copied().is_some_and(Self::is_content_char)
            {
                // Omit intra-word apostrophe variants so contractions compare equal.
            } else if !current.is_empty() {
                words.push(std::mem::take(&mut current));
            }
        }
        if !current.is_empty() {
            words.push(current);
        }
        words
    }

    /// Word-level edit distance classified into substitutions, deletions, and insertions.
    pub(crate) fn between(expected: &[String], actual: &[String]) -> Self {
        let mut errors = Self::default();
        for edit in Edit::trace(expected, actual) {
            match edit {
                Edit::Substitute(_, _) => errors.substitutions += 1,
                Edit::Delete(_) => errors.deletions += 1,
                Edit::Insert(_) => errors.insertions += 1,
            }
        }
        errors
    }

    pub(crate) fn total(&self) -> usize {
        self.substitutions + self.deletions + self.insertions
    }

    pub(crate) fn rate(&self, expected: usize) -> ErrorRate {
        ErrorRate::new(self.total(), expected)
    }

    pub(crate) fn accumulate(&mut self, other: &Self) {
        self.substitutions += other.substitutions;
        self.deletions += other.deletions;
        self.insertions += other.insertions;
    }

    fn is_apostrophe(ch: char) -> bool {
        matches!(ch, '\'' | '’' | 'ʼ')
    }

    fn is_content_char(ch: char) -> bool {
        ch.is_alphanumeric() && !Self::is_apostrophe(ch)
    }
}

impl fmt::Display for ContentErrors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "sub {}  del {}  ins {}",
            self.substitutions, self.deletions, self.insertions
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::score::Errors;

    #[test]
    fn normalizes_content_tokens_without_changing_exact_scoring() {
        let token_cases: &[(&str, &[&str])] = &[
            (
                "Hello, WORLD!\n- Don’t stop: follow-up, École, 42%.",
                &[
                    "hello", "world", "dont", "stop", "follow", "up", "école", "42",
                ],
            ),
            ("l\u{02bc}été", &["lété"]),
            ("Don\u{02bc}t wait.", &["dont", "wait"]),
            ("\u{02bc}word\u{02bc} \u{02bc}", &["word"]),
        ];
        for &(input, expected) in token_cases {
            assert_eq!(ContentErrors::tokenize(input), expected, "{input:?}");
        }

        let equivalent_pairs = [
            ("Hello world", "hello, world!"),
            ("One, two.\nThree.", "one two three"),
            ("List:\n- Alpha\n- Beta", "list alpha beta"),
            ("École café", "école, CAFÉ!"),
            ("Don’t wait.", "Don't wait."),
            ("Don’t wait.", "Dont wait."),
            ("Don't wait.", "Dont wait."),
            ("Don’t wait.", "Don\u{02bc}t wait."),
            ("Don't wait.", "Don\u{02bc}t wait."),
            ("Dont wait.", "Don\u{02bc}t wait."),
            ("l’été", "l'été"),
            ("l’été", "lété"),
            ("l’été", "l\u{02bc}été"),
            ("l'été", "l\u{02bc}été"),
            ("lété", "l\u{02bc}été"),
            ("Don't follow-up.", "dont, follow up?"),
            ("real-time speech-to-text", "real time speech to text"),
        ];
        for (expected, actual) in equivalent_pairs {
            let result = ContentErrors::between(
                &ContentErrors::tokenize(expected),
                &ContentErrors::tokenize(actual),
            );
            assert_eq!(
                result,
                ContentErrors::default(),
                "{expected:?} vs {actual:?}"
            );
            assert_eq!(result.total(), 0);
        }

        // Exact scoring stays untouched and still flags apostrophe and case diffs.
        for (expected, actual, want) in [
            (
                "Don’t wait.",
                "Dont wait.",
                Errors {
                    words: 1,
                    ..Errors::default()
                },
            ),
            (
                "l’été",
                "l\u{02bc}été",
                Errors {
                    words: 1,
                    ..Errors::default()
                },
            ),
            (
                "Hello world",
                "hello world",
                Errors {
                    case: 1,
                    ..Errors::default()
                },
            ),
        ] {
            assert_eq!(
                Errors::between(&Errors::tokenize(expected), &Errors::tokenize(actual)),
                want,
                "{expected:?} vs {actual:?}"
            );
        }
    }

    #[test]
    fn classifies_edits_handles_empty_ties_and_computes_rates() {
        let edit_cases = [
            ("send notes today", "send letters today", (1, 0, 0)),
            ("send the blue notes", "send notes", (0, 2, 0)),
            ("send notes", "um send uh notes please", (0, 0, 3)),
            (
                "Please send notes, then bring\nblue folders.",
                "please send notes then bring red folders today.",
                (1, 0, 1),
            ),
            ("", "", (0, 0, 0)),
            (".,!?\n", "one two", (0, 0, 2)),
            ("one two", "...", (0, 2, 0)),
            // Ties prefer diagonal (substitution) over deletion/insertion.
            ("alpha beta", "gamma", (1, 1, 0)),
            // Ties between deletion and insertion prefer deletion first.
            ("a b a", "b x a b", (0, 1, 2)),
        ];
        for (expected, actual, (substitutions, deletions, insertions)) in edit_cases {
            let want = ContentErrors {
                substitutions,
                deletions,
                insertions,
            };
            let got = ContentErrors::between(
                &ContentErrors::tokenize(expected),
                &ContentErrors::tokenize(actual),
            );
            assert_eq!(got, want, "{expected:?} vs {actual:?}");
            assert_eq!(got.total(), substitutions + deletions + insertions);
        }

        assert!(
            (ContentErrors::between(&[], &[]).rate(0).percentage() - 0.0).abs() < f64::EPSILON
        );
        assert!(
            (ContentErrors::between(&[], &ContentErrors::tokenize("extra"))
                .rate(0)
                .percentage()
                - 100.0)
                .abs()
                < f64::EPSILON
        );
        assert_eq!(
            ContentErrors {
                substitutions: 1,
                deletions: 2,
                insertions: 3,
            }
            .to_string(),
            "sub 1  del 2  ins 3"
        );
    }
}
