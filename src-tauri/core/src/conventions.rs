/// Writes numbers as digits, percentages as `N%` and spoken file extensions as `.ext`.
pub(crate) fn apply(text: &str) -> String {
    let numbered = TokenizedText::new(text).replace_numbers();
    TokenizedText::new(&numbered).replace_extensions()
}

struct Word<'a> {
    text: &'a str,
    start: usize,
    end: usize,
}

struct TokenizedText<'a> {
    text: &'a str,
    words: Vec<Word<'a>>,
}

impl<'a> TokenizedText<'a> {
    fn new(text: &'a str) -> Self {
        let words = text
            .match_indices(|c: char| c.is_alphanumeric() || matches!(c, '_' | '\''))
            .fold(Vec::<Word<'a>>::new(), |mut words, (start, ch)| {
                let end = start + ch.len();
                if let Some(last) = words.last_mut().filter(|last| last.end == start) {
                    last.end = end;
                    last.text = &text[last.start..end];
                } else {
                    words.push(Word {
                        text: &text[start..end],
                        start,
                        end,
                    });
                }
                words
            });
        Self { text, words }
    }

    // The separator immediately before a word (or the leading text for word zero).
    fn gap_before(&self, index: usize) -> &'a str {
        let start = index.checked_sub(1).map_or(0, |i| self.words[i].end);
        &self.text[start..self.words[index].start]
    }

    fn percent_end(&self, index: usize) -> Option<usize> {
        let next = self.words.get(index)?;
        if !space(self.gap_before(index)) {
            return None;
        }
        if next.text.eq_ignore_ascii_case("percent") {
            return Some(index + 1);
        }
        let cent = self.words.get(index + 1)?;
        (next.text.eq_ignore_ascii_case("per")
            && cent.text.eq_ignore_ascii_case("cent")
            && space(self.gap_before(index + 1)))
        .then_some(index + 2)
    }

    // Keep the whole number in a mixed fraction, not just its fractional tail.
    fn is_mixed_fraction(&self, parts: &[&str], index: usize) -> bool {
        if parts.get(index) != Some(&"and") || !space(self.gap_before(index)) {
            return false;
        }
        let mut j = index + 1;
        while j < parts.len() && number_gap(self.gap_before(j)) {
            if ordinal_or_fraction(parts[j]) && parts[j] != "first" {
                return true;
            }
            if !(is_number(parts[j]) || (j == index + 1 && parts[j] == "a")) {
                return false;
            }
            j += 1;
        }
        false
    }

    fn replace_numbers(&self) -> String {
        let lower: Vec<_> = self
            .words
            .iter()
            .map(|word| word.text.to_ascii_lowercase())
            .collect();
        let parts: Vec<_> = lower.iter().map(String::as_str).collect();
        let mut output = String::with_capacity(self.text.len());
        let mut cursor = 0;
        let mut i = 0;
        while i < self.words.len() {
            let start = i;
            let digits = lower[i].bytes().all(|b| b.is_ascii_digit());
            if !digits && !is_number(&lower[i]) {
                i += 1;
                continue;
            }
            i += 1;
            if !digits {
                while i < self.words.len() && number_gap(self.gap_before(i)) {
                    if is_number(&lower[i]) {
                        i += 1;
                    } else if lower[i] == "and"
                        && number_word(&lower[i - 1]).is_some_and(|v| v >= 100)
                        && i + 1 < self.words.len()
                        && number_gap(self.gap_before(i + 1))
                    {
                        if ordinal_or_fraction(&lower[i + 1]) || lower[i + 1] == "second" {
                            i += 2;
                            break;
                        }
                        if !is_number(&lower[i + 1]) {
                            break;
                        }
                        let mut end = i + 2;
                        while end < self.words.len()
                            && is_number(&lower[end])
                            && number_gap(self.gap_before(end))
                        {
                            end += 1;
                        }
                        if cardinal(&parts[start..i]).is_some()
                            && cardinal(&parts[start..end]).is_none()
                        {
                            break;
                        }
                        i = end;
                    } else {
                        break;
                    }
                }
            }
            let percent = self.percent_end(i);
            let before = start
                .checked_sub(1)
                .filter(|_| space(self.gap_before(start)));
            let after_point =
                lower.get(i).is_some_and(|w| w == "point") && space(self.gap_before(i));
            let after_ordinal = lower.get(i).is_some_and(|word| {
                let gap = self.gap_before(i);
                (ordinal_or_fraction(word) && number_gap(gap)) || (word == "second" && gap == "-")
            });
            let excluded = before.is_some_and(|j| lower[j] == "point")
                || after_point
                || after_ordinal
                || self.is_mixed_fraction(&parts, i);
            let value = if digits {
                Some(self.words[start].text.to_owned())
            } else {
                cardinal(&parts[start..i]).map(|n| n.to_string())
            };
            let lone_one = i == start + 1 && lower[start] == "one" && percent.is_none();
            if !excluded
                && !lone_one
                && let Some(value) = value
            {
                let end = percent.unwrap_or(i);
                output.push_str(&self.text[cursor..self.words[start].start]);
                output.push_str(&value);
                if percent.is_some() {
                    output.push('%');
                }
                cursor = self.words[end - 1].end;
                i = end;
            }
        }
        output.push_str(&self.text[cursor..]);
        output
    }

    fn replace_extensions(&self) -> String {
        let mut output = String::with_capacity(self.text.len());
        let mut cursor = 0;
        for (i, pair) in self.words.windows(2).enumerate() {
            let [dot, ext] = pair else { continue };
            if !dot.text.eq_ignore_ascii_case("dot") || !space(self.gap_before(i + 1)) {
                continue;
            }
            let lower = ext.text.to_ascii_lowercase();
            let Some(ext_name) = (match lower.as_str() {
                "wav" | "wave" => Some("wav"),
                "txt" | "text" => Some("txt"),
                "json" | "jason" => Some("json"),
                "md" | "pdf" | "csv" | "rs" | "ts" | "tsx" | "js" | "py" | "toml" | "yaml"
                | "yml" | "html" | "css" | "png" | "jpg" | "mp3" => Some(lower.as_str()),
                _ => None,
            }) else {
                continue;
            };
            let mut start = dot.start;
            if i > 0 {
                let previous = &self.words[i - 1];
                let gap = self.gap_before(i);
                let standalone = matches!(
                    previous.text.to_ascii_lowercase().as_str(),
                    "a" | "an" | "the" | "in" | "as" | "of" | "to" | "or" | "and" | "with" | "from"
                );
                if !standalone && space(gap) && !gap.contains(['\n', '\r']) {
                    start = previous.end;
                }
            }
            output.push_str(&self.text[cursor..start]);
            output.push('.');
            output.push_str(ext_name);
            cursor = ext.end;
        }
        output.push_str(&self.text[cursor..]);
        output
    }
}

fn number_word(word: &str) -> Option<u64> {
    match word {
        "zero" => Some(0),
        "one" => Some(1),
        "two" => Some(2),
        "three" => Some(3),
        "four" => Some(4),
        "five" => Some(5),
        "six" => Some(6),
        "seven" => Some(7),
        "eight" => Some(8),
        "nine" => Some(9),
        "ten" => Some(10),
        "eleven" => Some(11),
        "twelve" => Some(12),
        "thirteen" => Some(13),
        "fourteen" => Some(14),
        "fifteen" => Some(15),
        "sixteen" => Some(16),
        "seventeen" => Some(17),
        "eighteen" => Some(18),
        "nineteen" => Some(19),
        "twenty" => Some(20),
        "thirty" => Some(30),
        "forty" => Some(40),
        "fifty" => Some(50),
        "sixty" => Some(60),
        "seventy" => Some(70),
        "eighty" => Some(80),
        "ninety" => Some(90),
        "hundred" => Some(100),
        "thousand" => Some(1_000),
        "million" => Some(1_000_000),
        _ => None,
    }
}

fn is_number(word: &str) -> bool {
    number_word(word).is_some()
}

fn under_hundred(parts: &[&str]) -> Option<u64> {
    match parts {
        [word] => number_word(word).filter(|&v| v < 100),
        [tens, units] => {
            let tens = number_word(tens).filter(|&v| v < 100)?;
            let units = number_word(units).filter(|&v| v < 100)?;
            (tens >= 20 && tens % 10 == 0 && (1..10).contains(&units)).then_some(tens + units)
        }
        _ => None,
    }
}

fn under_thousand(parts: &[&str]) -> Option<u64> {
    if let [unit, "hundred", rest @ ..] = parts {
        let unit = number_word(unit).filter(|&v| (1..10).contains(&v))?;
        if rest.is_empty() {
            return Some(unit * 100);
        }
        let rest = rest.strip_prefix(&["and"]).unwrap_or(rest);
        let tail = under_hundred(rest)?;
        return (tail > 0).then_some(unit * 100 + tail);
    }
    under_hundred(parts)
}

// Descending scales and strict groups reject a whole malformed run, not just its prefix.
fn cardinal(parts: &[&str]) -> Option<u64> {
    let mut start = 0;
    let mut total = 0;
    let mut previous_scale = u64::MAX;
    for (index, part) in parts.iter().enumerate() {
        let Some(scale) = number_word(part).filter(|&v| v > 100) else {
            continue;
        };
        if scale >= previous_scale {
            return None;
        }
        let group = under_thousand(&parts[start..index])?;
        if group == 0 {
            return None;
        }
        total += group * scale;
        previous_scale = scale;
        start = index + 1;
    }
    let mut rest = &parts[start..];
    if rest.is_empty() {
        return (total > 0).then_some(total);
    }
    if total > 0 {
        rest = rest.strip_prefix(&["and"]).unwrap_or(rest);
    }
    let tail = under_thousand(rest)?;
    (total == 0 || tail > 0).then_some(total + tail)
}

fn space(gap: &str) -> bool {
    !gap.is_empty() && gap.chars().all(char::is_whitespace)
}

fn number_gap(gap: &str) -> bool {
    space(gap) || gap == "-"
}

// Cardinal prefixes of ordinals and fractions must stay written as words too.
fn ordinal_or_fraction(word: &str) -> bool {
    let singular = word.strip_suffix('s').unwrap_or(word);
    matches!(
        singular,
        "first"
            | "third"
            | "fourth"
            | "fifth"
            | "sixth"
            | "seventh"
            | "eighth"
            | "ninth"
            | "tenth"
            | "eleventh"
            | "twelfth"
            | "thirteenth"
            | "fourteenth"
            | "fifteenth"
            | "sixteenth"
            | "seventeenth"
            | "eighteenth"
            | "nineteenth"
            | "twentieth"
            | "thirtieth"
            | "fortieth"
            | "fiftieth"
            | "sixtieth"
            | "seventieth"
            | "eightieth"
            | "ninetieth"
            | "hundredth"
            | "thousandth"
            | "millionth"
            | "half"
            | "halve"
            | "quarter"
    )
}

#[cfg(test)]
mod tests {
    use super::apply;
    use crate::text_format::TextFormatter;

    #[test]
    fn converts_numbers_percentages_and_respects_boundaries() {
        let converted = [
            // Cardinals
            ("forty two files", "42 files"),
            ("twenty-five", "25"),
            ("one hundred and twenty", "120"),
            ("two thousand", "2000"),
            ("Twelve people", "12 people"),
            ("twenty one", "21"),
            ("zero, nineteen, ninety", "0, 19, 90"),
            ("two million three hundred thousand and six", "2300006"),
            (
                "nine hundred ninety-nine thousand eight hundred seventy-six",
                "999876",
            ),
            // Percentages
            ("ten percent", "10%"),
            ("10 percent", "10%"),
            ("one percent", "1%"),
            ("twenty five per cent", "25%"),
            ("ONE PER CENT!", "1%!"),
            ("0 percent", "0%"),
            // Boundaries
            ("wait thirty seconds", "wait 30 seconds"),
            ("five seconds", "5 seconds"),
            ("a ten second pause", "a 10 second pause"),
            ("the twenty-second", "the twenty-second"),
            ("between one hundred and two hundred", "between 100 and 200"),
            ("one thousand and two thousand", "1000 and 2000"),
            ("between five and ten", "between 5 and 10"),
            ("between ten and twenty percent", "between 10 and 20%"),
            ("one and two", "one and 2"),
            ("three hundred and zero", "300 and 0"),
            ("twenty and five", "20 and 5"),
            ("a ten percent discount", "a 10% discount"),
            ("an eighty percent chance", "an 80% chance"),
            ("a twenty-minute walk", "a 20-minute walk"),
            ("a million times", "a million times"),
            ("the twenty-first century", "the twenty-first century"),
            ("two thirds", "two thirds"),
            (
                "one hundred and twenty-first",
                "one hundred and twenty-first",
            ),
            ("three quarters", "three quarters"),
            ("two halves", "two halves"),
            ("twenty sixths", "twenty sixths"),
            ("two hundred and five", "205"),
            ("two thousand and ten", "2010"),
        ];
        for (input, expected) in converted {
            assert_eq!(apply(input), expected, "{input}");
        }

        let unchanged = [
            "no one came",
            "one of them",
            "the one I want",
            "twenty twenty six",
            "the first step",
            "half of it",
            "a third",
            "about two and a half hours",
            "three and a half percent",
            "two and three quarters",
            "one and a half",
            "two hundred and a half",
            "two and twenty three hundredths",
            "TWO AND A THIRD",
            "three point five",
            "point five",
            "a hundred times",
            "a thousand",
            "about a hundred and fifty users",
            "a hundred and twenty percent",
            "a thousand and ten",
            "a million and two",
            "two hundred and first",
            "two thousand and first",
            "two hundred and second",
            "two thousand and thirds",
            "one one",
            "two hundred hundred",
            "one thousand million",
            "zero hundred",
            "ten-zero",
        ];
        for input in unchanged {
            assert_eq!(apply(input), input, "{input}");
        }
    }

    #[test]
    fn spoken_extensions() {
        for (input, expected) in [
            ("ends in dot wav", "ends in .wav"),
            ("a dot text file", "a .txt file"),
            ("open main dot rs", "open main.rs"),
            ("Dot JSON", ".json"),
            ("Use dot JASON", "Use.json"),
            ("dot wave", ".wav"),
            ("a red dot on the map", "a red dot on the map"),
            ("connect the dots", "connect the dots"),
            ("dot com", "dot com"),
            ("main dot rs and test dot TSX", "main.rs and test.tsx"),
        ] {
            assert_eq!(apply(input), expected, "{input}");
        }
        for ext in [
            "md", "pdf", "csv", "rs", "ts", "tsx", "js", "py", "toml", "yaml", "yml", "html",
            "css", "png", "jpg", "mp3",
        ] {
            assert_eq!(
                apply(&format!("dot {}", ext.to_uppercase())),
                format!(".{ext}")
            );
        }
        for prefix in [
            "a", "an", "the", "in", "as", "of", "to", "or", "and", "with", "from",
        ] {
            assert_eq!(
                apply(&format!("{prefix} dot txt")),
                format!("{prefix} .txt")
            );
        }
    }

    #[test]
    fn preserves_punctuation_and_list_order() {
        assert_eq!(apply("“Twelve” files;\nforty-two!"), "“12” files;\n42!");
        let listed = TextFormatter::new().format_lists("One, we plan. Two, we build.");
        assert_eq!(apply(&listed), "1. We plan.\n2. We build.");
        assert_eq!(apply(""), "");
    }
}
