#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum MatchMode {
    NormalizedProse,
    StrictProse,
    ExactTrimmed,
}

pub(crate) struct QualityCase {
    pub(crate) category: &'static str,
    pub(crate) label: &'static str,
    pub(crate) chunks: &'static [&'static str],
    pub(crate) expected: &'static [&'static str],
    pub(crate) match_mode: MatchMode,
}

impl QualityCase {
    pub(crate) fn new(
        category: &'static str,
        label: &'static str,
        chunks: &'static [&'static str],
        expected: &'static [&'static str],
    ) -> Self {
        Self {
            category,
            label,
            chunks,
            expected,
            match_mode: MatchMode::NormalizedProse,
        }
    }

    pub(crate) fn strict_prose(mut self) -> Self {
        self.match_mode = MatchMode::StrictProse;
        self
    }

    pub(crate) fn exact(mut self) -> Self {
        self.match_mode = MatchMode::ExactTrimmed;
        self
    }

    pub(crate) fn input_text(&self) -> String {
        self.chunks.join(" ")
    }
}

/// Evaluates cleaned transcripts against expected variants using reusable token buffers.
#[derive(Default)]
pub(crate) struct ProseMatcher {
    out_tokens: Vec<String>,
    exp_tokens: Vec<String>,
}

impl ProseMatcher {
    pub(crate) fn check(&mut self, case: &QualityCase, output: &str) -> Option<String> {
        let matched = match case.match_mode {
            MatchMode::ExactTrimmed => case
                .expected
                .iter()
                .any(|target| output.trim() == target.trim()),
            MatchMode::NormalizedProse | MatchMode::StrictProse => {
                let strict = case.match_mode == MatchMode::StrictProse;
                Self::tokenize_into(output, strict, &mut self.out_tokens);
                let mut found = false;
                for target in case.expected {
                    Self::tokenize_into(target, strict, &mut self.exp_tokens);
                    if self.out_tokens == self.exp_tokens {
                        found = true;
                        break;
                    }
                }
                found
            }
        };

        if matched {
            return None;
        }
        Some(format!(
            "[{} / {}] {output:?}\n  expected one of {:?}",
            case.category, case.label, case.expected
        ))
    }

    /// Normalizes prose tokens while keeping meaning-bearing punctuation.
    /// In `strict` mode, casing and commas are preserved verbatim.
    fn tokenize_into(text: &str, strict: bool, out: &mut Vec<String>) {
        out.clear();
        let mut current = String::new();
        let mut prev_alnum = false;
        let mut prev_digit = false;
        let mut chars = text.trim().chars().peekable();
        while let Some(ch) = chars.next() {
            let is_apos = matches!(ch, '\'' | '’' | 'ʼ');
            let intra_apostrophe = is_apos
                && prev_alnum
                && chars
                    .peek()
                    .is_some_and(|next| next.is_alphanumeric() && !matches!(next, 'ʼ'));
            if intra_apostrophe {
                current.push('\'');
                prev_alnum = false;
                prev_digit = false;
            } else if !is_apos && (ch.is_alphanumeric() || ch == '_') {
                if strict {
                    current.push(ch);
                } else {
                    current.extend(ch.to_lowercase());
                }
                prev_alnum = true;
                prev_digit = ch.is_numeric();
            } else {
                if !current.is_empty() {
                    out.push(std::mem::take(&mut current));
                }
                let optional_comma = !strict
                    && ch == ','
                    && prev_alnum
                    && !prev_digit
                    && chars
                        .peek()
                        .is_some_and(|next| next.is_whitespace() && !matches!(next, '\n' | '\r'));
                if ch == '\n' {
                    out.push("\n".into());
                } else if ch == '\r' {
                    if chars.peek() == Some(&'\n') {
                        chars.next();
                    }
                    out.push("\n".into());
                } else if !ch.is_whitespace() && !optional_comma {
                    let mut token = String::new();
                    token.push(if ch == '−' { '-' } else { ch });
                    out.push(token);
                }
                prev_alnum = false;
                prev_digit = false;
            }
        }
        if !current.is_empty() {
            out.push(current);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_apostrophes_and_nonbreaking_space() {
        let mut matcher = ProseMatcher::default();
        let verb = QualityCase::new(
            "cutoff-false-start",
            "verb-repair",
            &["What we're passing in is are two separate files."],
            &["What we're passing in are 2 separate files."],
        );
        for variant in [
            "What we’re passing in are 2 separate files.",
            "What we\u{02bc}re passing in are 2 separate files.",
        ] {
            assert_eq!(matcher.check(&verb, variant), None);
        }

        let strict = QualityCase::new(
            "forced-pause-break",
            "metrics-endpoint",
            &["After restarting the daemon, the metrics.", "Endpoint started."],
            &["After restarting the daemon, the metrics endpoint started."],
        )
        .strict_prose();
        assert_eq!(
            matcher.check(
                &strict,
                "After restarting the daemon, the metrics\u{00a0}endpoint started."
            ),
            None
        );
        assert!(
            matcher
                .check(
                    &strict,
                    "After restarting the daemon, the metrics, endpoint started."
                )
                .is_some()
        );
    }
}
