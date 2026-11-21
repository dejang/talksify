use std::fmt;
use std::time::Duration;

use crate::content_score::ContentErrors;
use crate::dump::RecordingName;
use crate::score::Errors;

pub(crate) struct FileSample<'a> {
    pub(crate) name: &'a RecordingName,
    pub(crate) duration_secs: f64,
    pub(crate) expected: &'a str,
    pub(crate) raw: &'a str,
    pub(crate) cleaned: &'a str,
    pub(crate) wait: Duration,
}

#[derive(Default)]
struct ScoreTotals {
    exact_raw: Errors,
    exact_cleaned: Errors,
    exact_tokens: usize,
    content_raw: ContentErrors,
    content_cleaned: ContentErrors,
    content_words: usize,
}

impl ScoreTotals {
    fn from_texts(expected: &str, raw: &str, cleaned: &str) -> Self {
        let expected_tokens = Errors::tokenize(expected);
        let expected_words = ContentErrors::tokenize(expected);
        Self {
            exact_raw: Errors::between(&expected_tokens, &Errors::tokenize(raw)),
            exact_cleaned: Errors::between(&expected_tokens, &Errors::tokenize(cleaned)),
            exact_tokens: expected_tokens.len(),
            content_raw: ContentErrors::between(&expected_words, &ContentErrors::tokenize(raw)),
            content_cleaned: ContentErrors::between(
                &expected_words,
                &ContentErrors::tokenize(cleaned),
            ),
            content_words: expected_words.len(),
        }
    }

    fn accumulate(&mut self, other: &Self) {
        self.exact_raw.accumulate(&other.exact_raw);
        self.exact_cleaned.accumulate(&other.exact_cleaned);
        self.exact_tokens += other.exact_tokens;
        self.content_raw.accumulate(&other.content_raw);
        self.content_cleaned.accumulate(&other.content_cleaned);
        self.content_words += other.content_words;
    }

    fn render_lines(&self, prefix: &str) -> [String; 4] {
        [
            format!(
                "{prefix}exact raw        {}  {}",
                self.exact_raw.rate(self.exact_tokens),
                self.exact_raw
            ),
            format!(
                "{prefix}exact cleaned    {}  {}",
                self.exact_cleaned.rate(self.exact_tokens),
                self.exact_cleaned
            ),
            format!(
                "{prefix}content raw      {}  {}",
                self.content_raw.rate(self.content_words),
                self.content_raw
            ),
            format!(
                "{prefix}content cleaned  {}  {}",
                self.content_cleaned.rate(self.content_words),
                self.content_cleaned
            ),
        ]
    }
}

#[derive(Default)]
struct Aggregate {
    files: usize,
    scores: ScoreTotals,
}

impl Aggregate {
    fn accumulate(&mut self, scores: &ScoreTotals) {
        self.files += 1;
        self.scores.accumulate(scores);
    }
}

#[derive(Default)]
struct RecordingList(Vec<RecordingName>);

impl RecordingList {
    fn push(&mut self, name: RecordingName) {
        self.0.push(name);
    }

    fn extend(&mut self, names: impl IntoIterator<Item = RecordingName>) {
        self.0.extend(names);
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Display for RecordingList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            f.write_str("none")
        } else {
            let joined = self
                .0
                .iter()
                .map(RecordingName::quoted)
                .collect::<Vec<_>>()
                .join(", ");
            f.write_str(&joined)
        }
    }
}

#[derive(Default)]
pub(crate) struct HarnessReport {
    overall: Aggregate,
    over_60s: Aggregate,
    waits: Vec<(Duration, RecordingName)>,
    exact_worse: RecordingList,
    content_worse: RecordingList,
    skipped: RecordingList,
}

impl HarnessReport {
    pub(crate) fn record(&mut self, sample: &FileSample<'_>) -> String {
        let expected = sample.expected.trim();
        let raw = sample.raw.trim();
        let cleaned = sample.cleaned.trim();
        let scores = ScoreTotals::from_texts(expected, raw, cleaned);

        if scores.exact_cleaned.total() > scores.exact_raw.total() {
            self.exact_worse.push(sample.name.clone());
        }
        if scores.content_cleaned.total() > scores.content_raw.total() {
            self.content_worse.push(sample.name.clone());
        }
        self.overall.accumulate(&scores);
        if sample.duration_secs > 60.0 {
            self.over_60s.accumulate(&scores);
        }
        self.waits.push((sample.wait, sample.name.clone()));

        let mut lines = Vec::with_capacity(8);
        lines.push(format!(
            "{}  ({:.1} s)  wait {:.1} s",
            sample.name.quoted(),
            sample.duration_secs,
            sample.wait.as_secs_f64()
        ));
        lines.extend(scores.render_lines("  "));
        lines.push(format!("  expected: {expected:?}"));
        lines.push(format!("  raw:      {raw:?}"));
        lines.push(format!("  got:      {cleaned:?}"));
        lines.join("\n")
    }

    pub(crate) fn record_skipped(&mut self, files: impl IntoIterator<Item = RecordingName>) {
        self.skipped.extend(files);
    }
}

impl fmt::Display for HarnessReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut lines = Vec::with_capacity(12);
        if self.overall.files > 0 {
            lines.extend(self.overall.scores.render_lines("overall "));
            lines.extend(
                self.over_60s
                    .scores
                    .render_lines(&format!(">60 s ({} files) ", self.over_60s.files)),
            );
            if let Some((longest, name)) = self.waits.iter().max_by_key(|(wait, _)| *wait) {
                let sum: Duration = self.waits.iter().map(|(wait, _)| *wait).sum();
                #[expect(clippy::cast_precision_loss, reason = "practical file counts fit f64")]
                let average = sum.as_secs_f64() / self.waits.len() as f64;
                lines.push(format!(
                    "wait             average {average:.1} s, longest {:.1} s ({})",
                    longest.as_secs_f64(),
                    name.quoted()
                ));
            }
            lines.push(format!("exact worse      {}", self.exact_worse));
            lines.push(format!("content worse    {}", self.content_worse));
        }
        if !self.skipped.is_empty() {
            lines.push(format!("skipped, no .txt: {}", self.skipped));
        }
        f.write_str(&lines.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    fn find_line<'a>(rendered: &'a str, prefix: &str) -> &'a str {
        rendered
            .lines()
            .find(|line| line.starts_with(prefix))
            .unwrap_or_else(|| panic!("missing summary row starting with {prefix:?}:\n{rendered}"))
    }

    #[test]
    fn reports_overall_over_60s_and_separate_worse_lists() {
        assert_eq!(HarnessReport::default().to_string(), "");

        // Boundary: exactly 60.0 s must be excluded from the strictly >60 s aggregate.
        let mut boundary_report = HarnessReport::default();
        let boundary = RecordingName::from_path(Path::new("boundary.wav"));
        boundary_report.record(&FileSample {
            name: &boundary,
            duration_secs: 60.0,
            expected: "Exactly sixty seconds long",
            raw: "um exactly sixty seconds long",
            cleaned: "Exactly sixty seconds long.",
            wait: Duration::from_millis(120),
        });
        let boundary_rendered = boundary_report.to_string();
        for prefix in [
            ">60 s (0 files) exact raw",
            ">60 s (0 files) exact cleaned",
            ">60 s (0 files) content raw",
            ">60 s (0 files) content cleaned",
        ] {
            assert!(
                find_line(&boundary_rendered, prefix).contains("0.0%  (0/0)"),
                "unexpected row for {prefix}:\n{boundary_rendered}"
            );
        }

        // Mixed durations: comma-only diff worsens exact only; dropped word >60 s worsens both.
        let mut report = HarnessReport::default();
        let samples = [
            (
                "short.wav",
                45.0,
                "We ship it on Friday",
                "um we ship it on Friday",
                "We, ship, it, on, Friday.",
                100,
            ),
            (
                "exact_60.wav",
                60.0,
                "Sixty seconds exact",
                "Sixty seconds exact",
                "Sixty seconds exact",
                150,
            ),
            (
                "long.wav",
                61.5,
                "Keep every single word here",
                "Keep every single word here",
                "Keep every word here",
                250,
            ),
        ];
        for (name, duration_secs, expected, raw, cleaned, wait_ms) in samples {
            let recording = RecordingName::from_path(Path::new(name));
            report.record(&FileSample {
                name: &recording,
                duration_secs,
                expected,
                raw,
                cleaned,
                wait: Duration::from_millis(wait_ms),
            });
        }
        let rendered = report.to_string();
        assert!(
            find_line(&rendered, ">60 s (1 files) content raw")
                .contains("0.0%  (0/5)  sub 0  del 0  ins 0")
        );
        assert!(
            find_line(&rendered, ">60 s (1 files) content cleaned")
                .contains("20.0%  (1/5)  sub 0  del 1  ins 0")
        );
        assert_eq!(
            find_line(&rendered, "exact worse"),
            r#"exact worse      "short.wav", "long.wav""#
        );
        assert_eq!(
            find_line(&rendered, "content worse"),
            r#"content worse    "long.wav""#
        );
    }

    #[test]
    fn quotes_special_filenames_unambiguously_on_single_lines() {
        let matrix: &[(&[&str], &str)] = &[
            (&["a.wav", "b.wav"], r#"exact worse      "a.wav", "b.wav""#),
            (&["a.wav, b.wav"], r#"exact worse      "a.wav, b.wav""#),
            (
                &["a.wav\", \"b.wav"],
                r#"exact worse      "a.wav\", \"b.wav""#,
            ),
        ];
        for &(names, expected_worse) in matrix {
            let mut report = HarnessReport::default();
            for &name in names {
                let recording = RecordingName::from_path(Path::new(name));
                report.record(&FileSample {
                    name: &recording,
                    duration_secs: 5.0,
                    expected: "Hello world.",
                    raw: "Hello world.",
                    cleaned: "Wrong output.",
                    wait: Duration::from_millis(100),
                });
            }
            assert_eq!(find_line(&report.to_string(), "exact worse"), expected_worse);
        }

        let tricky =
            RecordingName::from_path(Path::new("line1\nline2\r\u{85}\u{2028}\u{2029}, \"tail\".wav"));
        let expected_token = tricky.quoted();
        let mut report = HarnessReport::default();
        let entry = report.record(&FileSample {
            name: &tricky,
            duration_secs: 12.3,
            expected: "Hello world.",
            raw: "Hello world.",
            cleaned: "Wrong output.",
            wait: Duration::from_millis(300),
        });
        report.record_skipped([
            tricky.clone(),
            RecordingName::from_path(Path::new("plain.wav")),
        ]);

        let summary = report.to_string();
        assert_eq!(entry.lines().count(), 8);
        assert_eq!(summary.lines().count(), 12);
        assert_eq!(
            entry.lines().next().unwrap(),
            format!("{expected_token}  (12.3 s)  wait 0.3 s")
        );
        assert!(find_line(&summary, "wait ").ends_with(&format!("({expected_token})")));
        assert_eq!(
            find_line(&summary, "exact worse"),
            format!("exact worse      {expected_token}")
        );
        assert_eq!(
            find_line(&summary, "content worse"),
            format!("content worse    {expected_token}")
        );
        assert_eq!(
            find_line(&summary, "skipped, no .txt:"),
            format!("skipped, no .txt: {expected_token}, \"plain.wav\"")
        );
        for rendered in [entry, summary] {
            assert!(
                !rendered
                    .chars()
                    .any(|ch| matches!(ch, '\r' | '\u{85}' | '\u{2028}' | '\u{2029}')),
                "report contains raw line break:\n{rendered}"
            );
        }
    }
}
