/// Writes numbers as digits, percentages as `N%` and spoken file extensions as `.ext`.
pub(crate) fn apply(text: &str) -> String {
    extensions(&numbers(text))
}

struct Word<'a> {
    text: &'a str,
    start: usize,
    end: usize,
}

fn words(text: &str) -> Vec<Word<'_>> {
    text.match_indices(|c: char| c.is_alphanumeric() || matches!(c, '_' | '\''))
        .fold(Vec::<Word<'_>>::new(), |mut words, (start, ch)| {
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
        })
}

fn small(word: &str) -> Option<u64> {
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
        _ => None,
    }
}

fn scale(word: &str) -> Option<u64> {
    match word {
        "hundred" => Some(100),
        "thousand" => Some(1_000),
        "million" => Some(1_000_000),
        _ => None,
    }
}

fn is_number(word: &str) -> bool {
    small(word).is_some() || scale(word).is_some()
}

fn under_hundred(parts: &[&str]) -> Option<u64> {
    match parts {
        [word] => small(word),
        [tens, units] => {
            let tens = small(tens)?;
            let units = small(units)?;
            (tens >= 20 && tens % 10 == 0 && (1..10).contains(&units)).then_some(tens + units)
        }
        _ => None,
    }
}

fn under_thousand(parts: &[&str]) -> Option<u64> {
    if let [unit, "hundred", rest @ ..] = parts {
        let unit = small(unit)?;
        if !(1..10).contains(&unit) {
            return None;
        }
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
        let Some(scale) = scale(part).filter(|scale| *scale > 100) else {
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

// The separator immediately before a word (or the leading text for word zero).
fn gap_before<'a>(text: &'a str, words: &[Word<'_>], index: usize) -> &'a str {
    let start = index.checked_sub(1).map_or(0, |i| words[i].end);
    &text[start..words[index].start]
}

fn space(gap: &str) -> bool {
    !gap.is_empty() && gap.chars().all(char::is_whitespace)
}

fn number_gap(gap: &str) -> bool {
    space(gap) || gap == "-"
}

fn percent_end(text: &str, words: &[Word<'_>], index: usize) -> Option<usize> {
    let next = words.get(index)?;
    if !space(gap_before(text, words, index)) {
        return None;
    }
    if next.text.eq_ignore_ascii_case("percent") {
        return Some(index + 1);
    }
    let cent = words.get(index + 1)?;
    (next.text.eq_ignore_ascii_case("per")
        && cent.text.eq_ignore_ascii_case("cent")
        && space(gap_before(text, words, index + 1)))
    .then_some(index + 2)
}

// Cardinal prefixes of ordinals and fractions must stay written as words too.
fn ordinal_or_fraction(word: &str) -> bool {
    let singular = word.strip_suffix('s').unwrap_or(word);
    matches!(
        singular,
        "first" | "third" | "fourth" | "fifth" | "sixth" | "seventh"
            | "eighth" | "ninth" | "tenth" | "eleventh" | "twelfth" | "thirteenth"
            | "fourteenth" | "fifteenth" | "sixteenth" | "seventeenth" | "eighteenth"
            | "nineteenth" | "twentieth" | "thirtieth" | "fortieth" | "fiftieth"
            | "sixtieth" | "seventieth" | "eightieth" | "ninetieth" | "hundredth"
            | "thousandth" | "millionth" | "half" | "halve" | "quarter"
    )
}

// Keep the whole number in a mixed fraction, not just its fractional tail.
fn mixed_fraction(text: &str, words: &[Word<'_>], parts: &[&str], index: usize) -> bool {
    if parts.get(index) != Some(&"and") || !space(gap_before(text, words, index)) {
        return false;
    }
    let mut i = index + 1;
    while i < parts.len() && number_gap(gap_before(text, words, i)) {
        let word = parts[i];
        if ordinal_or_fraction(word) && word != "first" {
            return true;
        }
        if !(is_number(word) || (i == index + 1 && word == "a")) {
            break;
        }
        i += 1;
    }
    false
}

fn numbers(text: &str) -> String {
    let words = words(text);
    let lower: Vec<_> = words
        .iter()
        .map(|word| word.text.to_ascii_lowercase())
        .collect();
    let parts: Vec<_> = lower.iter().map(String::as_str).collect();
    let mut output = String::with_capacity(text.len());
    let mut cursor = 0;
    let mut i = 0;
    while i < words.len() {
        let start = i;
        let digits = lower[i].bytes().all(|b| b.is_ascii_digit());
        if !digits && !is_number(&lower[i]) {
            i += 1;
            continue;
        }
        i += 1;
        if !digits {
            while i < words.len() && number_gap(gap_before(text, &words, i)) {
                if is_number(&lower[i]) {
                    i += 1;
                } else if lower[i] == "and"
                    && scale(&lower[i - 1]).is_some()
                    && i + 1 < words.len()
                    && number_gap(gap_before(text, &words, i + 1))
                {
                    // Keep ordinal/fraction phrases intact across the conjunction.
                    if ordinal_or_fraction(&lower[i + 1]) || lower[i + 1] == "second" {
                        i += 2;
                        break;
                    }
                    if !is_number(&lower[i + 1]) {
                        break;
                    }
                    let mut end = i + 2;
                    while end < words.len()
                        && is_number(&lower[end])
                        && number_gap(gap_before(text, &words, end))
                    {
                        end += 1;
                    }
                    // Split independent valid cardinals, but never split an invalid
                    // prefix such as "a hundred" from its "and fifty" tail.
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
        let percent = percent_end(text, &words, i);
        let before = start
            .checked_sub(1)
            .filter(|_| space(gap_before(text, &words, start)));
        let after_point = lower.get(i).is_some_and(|word| word == "point")
            && space(gap_before(text, &words, i));
        // "second" with a space can be a duration; hyphenated "twenty-second"
        // is unambiguously ordinal. Plural "seconds" is always a duration.
        let after_ordinal_or_fraction = lower.get(i).is_some_and(|word| {
            let gap = gap_before(text, &words, i);
            (ordinal_or_fraction(word) && number_gap(gap)) || (word == "second" && gap == "-")
        });
        let excluded = before.is_some_and(|j| lower[j] == "point")
            || after_point
            || after_ordinal_or_fraction
            || mixed_fraction(text, &words, &parts, i);
        let value = if digits {
            Some(words[start].text.to_owned())
        } else {
            cardinal(&parts[start..i]).map(|n| n.to_string())
        };
        let lone_one = i == start + 1 && lower[start] == "one" && percent.is_none();
        if !excluded
            && !lone_one
            && let Some(value) = value
        {
            let end = percent.unwrap_or(i);
            output.push_str(&text[cursor..words[start].start]);
            output.push_str(&value);
            if percent.is_some() {
                output.push('%');
            }
            cursor = words[end - 1].end;
            i = end;
        }
    }
    output.push_str(&text[cursor..]);
    output
}

fn extension(word: &str) -> Option<&str> {
    match word {
        "wav" | "wave" => Some("wav"),
        "txt" | "text" => Some("txt"),
        "json" | "jason" => Some("json"),
        "md" | "pdf" | "csv" | "rs" | "ts" | "tsx" | "js" | "py" | "toml" | "yaml" | "yml"
        | "html" | "css" | "png" | "jpg" | "mp3" => Some(word),
        _ => None,
    }
}

fn extensions(text: &str) -> String {
    let words = words(text);
    let mut output = String::with_capacity(text.len());
    let mut cursor = 0;
    for (i, pair) in words.windows(2).enumerate() {
        let [dot, ext] = pair else { continue };
        if !dot.text.eq_ignore_ascii_case("dot") || !space(gap_before(text, &words, i + 1)) {
            continue;
        }
        let lower = ext.text.to_ascii_lowercase();
        let Some(ext_name) = extension(&lower) else {
            continue;
        };
        let mut start = dot.start;
        if i > 0 {
            let previous = &words[i - 1];
            let gap = gap_before(text, &words, i);
            let standalone = matches!(
                previous.text.to_ascii_lowercase().as_str(),
                "a" | "an" | "the" | "in" | "as" | "of" | "to" | "or" | "and" | "with" | "from"
            );
            if !standalone && space(gap) && !gap.contains(['\n', '\r']) {
                start = previous.end;
            }
        }
        output.push_str(&text[cursor..start]);
        output.push('.');
        output.push_str(ext_name);
        cursor = ext.end;
    }
    output.push_str(&text[cursor..]);
    output
}

#[cfg(test)]
mod tests {
    use super::apply;
    use crate::text_format::TextFormatter;

    #[test]
    fn cardinal_numbers() {
        for (input, expected) in [
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
        ] {
            assert_eq!(apply(input), expected, "{input}");
        }
    }

    #[test]
    fn percentages() {
        for (input, expected) in [
            ("ten percent", "10%"),
            ("10 percent", "10%"),
            ("one percent", "1%"),
            ("twenty five per cent", "25%"),
            ("ONE PER CENT!", "1%!"),
            ("0 percent", "0%"),
        ] {
            assert_eq!(apply(input), expected, "{input}");
        }
    }

    #[test]
    fn ambiguous_and_out_of_scope_numbers_stay_unchanged() {
        for input in [
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
        ] {
            assert_eq!(apply(input), input);
        }
    }

    #[test]
    fn number_boundaries() {
        for (input, expected) in [
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
            ("one hundred and twenty-first", "one hundred and twenty-first"),
            ("three quarters", "three quarters"),
            ("two halves", "two halves"),
            ("twenty sixths", "twenty sixths"),
            ("two hundred and five", "205"),
            ("two thousand and ten", "2010"),
        ] {
            assert_eq!(apply(input), expected, "{input}");
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
