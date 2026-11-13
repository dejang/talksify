/// Splits text into words, punctuation marks and line breaks.
pub(crate) fn tokens(text: &str) -> Vec<&str> {
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
    pub(crate) fn total(&self) -> usize {
        self.case + self.punctuation + self.line_breaks + self.words
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

/// Token-level edit distance, classified along one deterministic cheapest path.
pub(crate) fn errors(expected: &[&str], actual: &[&str]) -> Errors {
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

    let mut errors = Errors::default();
    let (mut i, mut j) = (expected.len(), actual.len());
    while i > 0 || j > 0 {
        let distance = matrix[i * width + j];
        // Ties prefer diagonal (match/substitution), then deletion, then insertion.
        if i > 0
            && j > 0
            && distance
                == matrix[(i - 1) * width + j - 1] + usize::from(expected[i - 1] != actual[j - 1])
        {
            if expected[i - 1] != actual[j - 1] {
                errors.record(Some(expected[i - 1]), Some(actual[j - 1]));
            }
            i -= 1;
            j -= 1;
        } else if i > 0 && distance == matrix[(i - 1) * width + j] + 1 {
            errors.record(Some(expected[i - 1]), None);
            i -= 1;
        } else {
            errors.record(None, Some(actual[j - 1]));
            j -= 1;
        }
    }
    errors
}

/// Percentage of expected tokens that need editing, including insertions.
pub(crate) fn rate(errors: usize, expected: usize) -> f64 {
    if expected == 0 {
        if errors == 0 { 0.0 } else { 100.0 }
    } else {
        #[expect(clippy::cast_precision_loss, reason = "practical token counts fit f64")]
        let percentage = errors as f64 / expected as f64 * 100.0;
        percentage
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_punctuation_and_line_breaks() {
        assert_eq!(
            tokens("Hello, world.\n- Don't"),
            ["Hello", ",", "world", ".", "\n", "-", "Don't"]
        );
        assert_eq!(
            tokens("  follow-up\t don't\r l’été  \r\n"),
            ["follow-up", "don't", "l’été", "\n"]
        );
        assert_eq!(tokens(" \t\r\u{2003}"), Vec::<&str>::new());
    }

    #[test]
    fn connectors_must_have_alphanumeric_neighbors() {
        assert_eq!(
            tokens("'word' a--b a- -b ’ - a’b 42 café"),
            [
                "'", "word", "'", "a", "-", "-", "b", "a", "-", "-", "b", "’", "-", "a’b", "42",
                "café"
            ]
        );
    }

    #[test]
    fn exact_token_edit_distance() {
        for (expected, actual) in [
            ("report", "Report"),
            ("Hello, world", "Hello world"),
            ("one word", "one extra word"),
            ("one word", "one replacement"),
            ("List:", "List."),
            ("one\ntwo", "one two"),
        ] {
            let expected = tokens(expected);
            let actual = tokens(actual);
            assert_eq!(errors(&expected, &actual).total(), 1);
            assert_eq!(errors(&actual, &expected).total(), 1);
            assert_eq!(errors(&expected, &expected).total(), 0);
        }
        assert_eq!(errors(&[], &[]).total(), 0);
        assert_eq!(errors(&[], &["one", "two"]).total(), 2);
        assert_eq!(errors(&["a", "b", "c"], &["x", "b", "y", "z"]).total(), 3);
    }

    #[test]
    fn edits_are_classified_by_priority() {
        for (expected, actual, counts) in [
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
        ] {
            let errors = errors(&tokens(expected), &tokens(actual));
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
    fn traceback_ties_prefer_diagonal() {
        // Diagonal substitutes B -> b before deleting a; deletion-first gives two word errors.
        assert_eq!(
            errors(&tokens("a B"), &tokens("b")),
            Errors {
                case: 1,
                words: 1,
                ..Errors::default()
            }
        );
        assert_eq!(
            errors(&tokens("a ,"), &tokens(", a")),
            Errors {
                punctuation: 2,
                ..Errors::default()
            }
        );
    }

    #[test]
    fn traceback_ties_prefer_deletion_over_insertion() {
        assert_eq!(
            errors(&["a", "A", "a"], &["A", ",", "a", "A"]),
            Errors {
                punctuation: 1,
                words: 2,
                ..Errors::default()
            }
        );
    }

    #[test]
    fn empty_reference_rates_and_insertions() {
        let cases = [
            (rate(errors(&[], &[]).total(), 0), 0.0),
            (rate(errors(&[], &["word"]).total(), 0), 100.0),
            (rate(1, 4), 25.0),
            (rate(3, 2), 150.0),
        ];
        for (actual, expected) in cases {
            assert!((actual - expected).abs() < f64::EPSILON);
        }
    }
}
