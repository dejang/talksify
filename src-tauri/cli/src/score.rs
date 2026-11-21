use std::fmt;

pub(crate) enum Edit<'a, T> {
    Substitute(&'a T, &'a T),
    Delete(&'a T),
    Insert(&'a T),
}

impl<'a, T: PartialEq> Edit<'a, T> {
    /// Computes one deterministic cheapest edit script (ties: diagonal, deletion, insertion).
    pub(crate) fn trace(expected: &'a [T], actual: &'a [T]) -> Vec<Self> {
        let width = actual.len() + 1;
        let mut matrix = vec![0; (expected.len() + 1) * width];
        for (j, cell) in matrix[..width].iter_mut().enumerate() {
            *cell = j;
        }
        for (i, a) in expected.iter().enumerate() {
            matrix[(i + 1) * width] = i + 1;
            for (j, b) in actual.iter().enumerate() {
                matrix[(i + 1) * width + j + 1] = (matrix[i * width + j] + usize::from(a != b))
                    .min(matrix[i * width + j + 1] + 1)
                    .min(matrix[(i + 1) * width + j] + 1);
            }
        }

        let mut edits = Vec::new();
        let (mut i, mut j) = (expected.len(), actual.len());
        while i > 0 || j > 0 {
            let distance = matrix[i * width + j];
            // Ties prefer diagonal (match/substitution), then deletion, then insertion.
            if i > 0
                && j > 0
                && distance
                    == matrix[(i - 1) * width + j - 1]
                        + usize::from(expected[i - 1] != actual[j - 1])
            {
                if expected[i - 1] != actual[j - 1] {
                    edits.push(Self::Substitute(&expected[i - 1], &actual[j - 1]));
                }
                i -= 1;
                j -= 1;
            } else if i > 0 && distance == matrix[(i - 1) * width + j] + 1 {
                edits.push(Self::Delete(&expected[i - 1]));
                i -= 1;
            } else {
                edits.push(Self::Insert(&actual[j - 1]));
                j -= 1;
            }
        }
        edits
    }
}

/// Percentage of expected tokens or words that need editing, including insertions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ErrorRate {
    errors: usize,
    expected: usize,
}

impl ErrorRate {
    pub(crate) fn new(errors: usize, expected: usize) -> Self {
        Self { errors, expected }
    }

    pub(crate) fn percentage(self) -> f64 {
        if self.expected == 0 {
            if self.errors == 0 { 0.0 } else { 100.0 }
        } else {
            #[expect(clippy::cast_precision_loss, reason = "practical token counts fit f64")]
            let percentage = self.errors as f64 / self.expected as f64 * 100.0;
            percentage
        }
    }
}

impl fmt::Display for ErrorRate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:.1}%  ({}/{})",
            self.percentage(),
            self.errors,
            self.expected
        )
    }
}

/// Splits text into words, punctuation marks and line breaks.
pub(crate) fn tokenize(text: &str) -> Vec<&str> {
    let mut tokens = Vec::new();
    let mut chars = text.char_indices().peekable();
    while let Some((start, ch)) = chars.next() {
        if ch == '\n' {
            tokens.push(&text[start..=start]);
        } else if ch.is_alphanumeric() {
            let mut end = start + ch.len_utf8();
            while let Some(&(index, next)) = chars.peek() {
                let alphanumeric = next.is_alphanumeric();
                let connector = matches!(next, '\'' | '’' | '-')
                    && chars
                        .clone()
                        .nth(1)
                        .is_some_and(|(_, c)| c.is_alphanumeric());
                if !alphanumeric && !connector {
                    break;
                }
                chars.next();
                end = index + next.len_utf8();
            }
            tokens.push(&text[start..end]);
        } else if !ch.is_whitespace() {
            tokens.push(&text[start..start + ch.len_utf8()]);
        }
    }
    tokens
}

#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Errors {
    pub(crate) case: usize,
    pub(crate) punctuation: usize,
    pub(crate) line_breaks: usize,
    pub(crate) words: usize,
}

impl Errors {
    /// Token-level edit distance, classified along one deterministic cheapest path.
    pub(crate) fn between(expected: &[&str], actual: &[&str]) -> Self {
        let mut errors = Self::default();
        for edit in Edit::trace(expected, actual) {
            match edit {
                Edit::Substitute(exp, act) => errors.record(Some(*exp), Some(*act)),
                Edit::Delete(exp) => errors.record(Some(*exp), None),
                Edit::Insert(act) => errors.record(None, Some(*act)),
            }
        }
        errors
    }

    pub(crate) fn total(&self) -> usize {
        self.case + self.punctuation + self.line_breaks + self.words
    }

    pub(crate) fn rate(&self, expected: usize) -> ErrorRate {
        ErrorRate::new(self.total(), expected)
    }

    pub(crate) fn accumulate(&mut self, other: &Self) {
        self.case += other.case;
        self.punctuation += other.punctuation;
        self.line_breaks += other.line_breaks;
        self.words += other.words;
    }

    fn record(&mut self, expected: Option<&str>, actual: Option<&str>) {
        let tokens = [expected, actual];
        if tokens.contains(&Some("\n")) {
            self.line_breaks += 1;
        } else if tokens
            .into_iter()
            .flatten()
            .any(|token| token.chars().next().is_some_and(|ch| !ch.is_alphanumeric()))
        {
            self.punctuation += 1;
        } else if expected
            .zip(actual)
            .is_some_and(|(a, b)| a.to_lowercase() == b.to_lowercase())
        {
            self.case += 1;
        } else {
            self.words += 1;
        }
    }
}

impl fmt::Display for Errors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "case {}  punctuation {}  line breaks {}  words {}",
            self.case, self.punctuation, self.line_breaks, self.words
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizes_exact_words_punctuation_connectors_and_line_breaks() {
        let cases: &[(&str, &[&str])] = &[
            (
                "Hello, world.\n- Don't",
                &["Hello", ",", "world", ".", "\n", "-", "Don't"],
            ),
            (
                "  follow-up\t don't\r l’été  \r\n",
                &["follow-up", "don't", "l’été", "\n"],
            ),
            (" \t\r\u{2003}", &[]),
            (
                "'word' a--b a- -b ’ - a’b 42 café",
                &[
                    "'", "word", "'", "a", "-", "-", "b", "a", "-", "-", "b", "’", "-", "a’b",
                    "42", "café",
                ],
            ),
        ];
        for &(input, expected) in cases {
            assert_eq!(tokenize(input), expected, "{input:?}");
        }
    }

    #[test]
    fn scores_and_classifies_exact_edits_with_tie_breaking() {
        for (expected, actual) in [
            ("report", "Report"),
            ("Hello, world", "Hello world"),
            ("one word", "one extra word"),
            ("one word", "one replacement"),
            ("List:", "List."),
            ("one\ntwo", "one two"),
        ] {
            let expected = tokenize(expected);
            let actual = tokenize(actual);
            assert_eq!(Errors::between(&expected, &actual).total(), 1);
            assert_eq!(Errors::between(&actual, &expected).total(), 1);
            assert_eq!(Errors::between(&expected, &expected).total(), 0);
        }
        assert_eq!(Errors::between(&[], &[]).total(), 0);
        assert_eq!(Errors::between(&[], &["one", "two"]).total(), 2);
        assert_eq!(
            Errors::between(&["a", "b", "c"], &["x", "b", "y", "z"]).total(),
            3
        );

        let classification_cases = [
            ("report", "Report", [1, 0, 0, 0]),
            ("ready, now", "ready now", [0, 1, 0, 0]),
            (",", "and", [0, 1, 0, 0]),
            ("a\nb", "a b", [0, 0, 1, 0]),
            ("send notes", "send the notes", [0, 0, 0, 1]),
            ("send notes", "send letters", [0, 0, 0, 1]),
            ("École", "école", [1, 0, 0, 0]),
            ("\n", ",", [0, 0, 1, 0]),
            (
                "Please send notes, then bring\nblue folders.",
                "please send notes then bring red folders today.",
                [1, 1, 1, 2],
            ),
            // Diagonal tie-breaking: substitutes B -> b before deleting a.
            ("a B", "b", [1, 0, 0, 1]),
            ("a ,", ", a", [0, 2, 0, 0]),
            // Deletion-over-insertion tie-breaking.
            ("a A a", "A , a A", [0, 1, 0, 2]),
        ];
        for (expected, actual, counts) in classification_cases {
            let errors = Errors::between(&tokenize(expected), &tokenize(actual));
            assert_eq!(
                [
                    errors.case,
                    errors.punctuation,
                    errors.line_breaks,
                    errors.words
                ],
                counts,
                "{expected:?} vs {actual:?}"
            );
            assert_eq!(errors.total(), counts.iter().sum::<usize>());
        }
    }

    #[test]
    fn computes_and_formats_error_rates() {
        for (actual, expected) in [
            (Errors::between(&[], &[]).rate(0).percentage(), 0.0),
            (Errors::between(&[], &["word"]).rate(0).percentage(), 100.0),
            (ErrorRate::new(1, 4).percentage(), 25.0),
            (ErrorRate::new(3, 2).percentage(), 150.0),
        ] {
            assert!((actual - expected).abs() < f64::EPSILON);
        }
        assert_eq!(ErrorRate::new(1, 4).to_string(), "25.0%  (1/4)");
        assert_eq!(
            Errors {
                case: 1,
                punctuation: 2,
                line_breaks: 3,
                words: 4,
            }
            .to_string(),
            "case 1  punctuation 2  line breaks 3  words 4"
        );
    }
}
