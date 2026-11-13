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

/// Token-level edit distance: substitutions + deletions + insertions.
pub(crate) fn errors(expected: &[&str], actual: &[&str]) -> usize {
    // Keep the rows as short as possible; edit distance is symmetric.
    let (rows, columns) = if expected.len() >= actual.len() {
        (expected, actual)
    } else {
        (actual, expected)
    };
    let mut previous: Vec<usize> = (0..=columns.len()).collect();
    let mut current = vec![0; columns.len() + 1];
    for (i, row) in rows.iter().enumerate() {
        current[0] = i + 1;
        for (j, column) in columns.iter().enumerate() {
            current[j + 1] = (previous[j] + usize::from(row != column))
                .min(previous[j + 1] + 1)
                .min(current[j] + 1);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[columns.len()]
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
            assert_eq!(errors(&expected, &actual), 1);
            assert_eq!(errors(&actual, &expected), 1);
            assert_eq!(errors(&expected, &expected), 0);
        }
        assert_eq!(errors(&[], &[]), 0);
        assert_eq!(errors(&[], &["one", "two"]), 2);
        assert_eq!(errors(&["a", "b", "c"], &["x", "b", "y", "z"]), 3);
    }

    #[test]
    fn empty_reference_rates_and_insertions() {
        let cases = [
            (rate(errors(&[], &[]), 0), 0.0),
            (rate(errors(&[], &["word"]), 0), 100.0),
            (rate(1, 4), 25.0),
            (rate(3, 2), 150.0),
        ];
        for (actual, expected) in cases {
            assert!((actual - expected).abs() < f64::EPSILON);
        }
    }
}
