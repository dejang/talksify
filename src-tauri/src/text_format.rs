use std::collections::{HashMap, HashSet};
use std::fmt::Write;

/// Rule-based cleanup around the LLM: filler removal before it, list layout after it.
pub(crate) struct TextFormatter {
    fillers: HashSet<&'static str>,
    ordinals: HashMap<&'static str, usize>,
    min_counted_items: usize,
    min_bullet_items: usize,
    max_bullet_words: usize,
}

impl TextFormatter {
    pub(crate) fn new() -> Self {
        let counted = [
            ["1", "one", "first"],
            ["2", "two", "second"],
            ["3", "three", "third"],
            ["4", "four", "fourth"],
            ["5", "five", "fifth"],
            ["6", "six", "sixth"],
            ["7", "seven", "seventh"],
            ["8", "eight", "eighth"],
            ["9", "nine", "ninth"],
            ["10", "ten", "tenth"],
        ];
        Self {
            fillers: [
                "uh", "uhh", "uhm", "um", "umm", "erm", "er", "hmm", "hm", "mm", "mhm",
            ]
            .into_iter()
            .collect(),
            ordinals: counted
                .iter()
                .enumerate()
                .flat_map(|(index, words)| words.iter().map(move |word| (*word, index)))
                .collect(),
            min_counted_items: 2,
            min_bullet_items: 3,
            max_bullet_words: 5,
        }
    }

    /// Removes hesitation sounds ("uh", "um", ...) and punctuation that only surrounded them.
    pub(crate) fn strip_fillers(&self, text: &str) -> String {
        let has_german_uhr = text
            .split(|c: char| !c.is_alphabetic())
            .any(|w| w.eq_ignore_ascii_case("uhr"));
        let mut words: Vec<String> = Vec::new();
        let mut capitalize_next = false;
        for token in text.split_whitespace() {
            let core = token.trim_end_matches([',', '.', ';', ':', '!', '?']);
            let lower = core.to_lowercase();
            let set_off = core.len() < token.len();
            // Bare "er" and German "um <time> Uhr" are real words; other fillers strip anywhere.
            let keep_word = (lower == "er" && !set_off)
                || (lower == "um" && !set_off && core == "um" && has_german_uhr);
            if keep_word || !self.fillers.contains(lower.as_str()) {
                let mut word = String::with_capacity(token.len());
                if capitalize_next {
                    push_capitalized(&mut word, token);
                } else {
                    word.push_str(token);
                }
                words.push(word);
                capitalize_next = false;
                continue;
            }
            let punct = &token[core.len()..];
            let terminal = punct.chars().find(|c| matches!(c, '.' | '!' | '?'));
            match words.last_mut() {
                None => capitalize_next = true,
                Some(prev) if prev.ends_with(['.', '!', '?']) => capitalize_next = true,
                Some(prev) => {
                    if let Some(end) = terminal {
                        if prev.ends_with([',', ';', ':']) {
                            prev.pop();
                        }
                        prev.push(end);
                        capitalize_next = true;
                    } else if punct.contains(',') && prev.ends_with(',') {
                        prev.pop();
                    }
                }
            }
        }
        words.join(" ")
    }

    /// Lays out spoken counting as a numbered list and short colon lists as bullets.
    pub(crate) fn format_lists(&self, text: &str) -> String {
        self.bulletize(&self.number_counted(text))
    }

    fn number_counted(&self, text: &str) -> String {
        let tokens: Vec<(usize, &str)> = text
            .split_whitespace()
            .map(|token| (token.as_ptr() as usize - text.as_ptr() as usize, token))
            .collect();

        // Each marker spans "[and] two," as (start byte, end byte).
        let mut markers: Vec<(usize, usize)> = Vec::new();
        for (i, &(start, token)) in tokens.iter().enumerate() {
            let Some(word) = token.strip_suffix([',', '.', ':']) else {
                continue;
            };
            let Some(&index) = self.ordinals.get(word.to_lowercase().as_str()) else {
                continue;
            };
            let prev = i.checked_sub(1).map(|j| tokens[j]);
            let (marker_start, before) = match prev {
                Some((and_start, and)) if and.eq_ignore_ascii_case("and") => {
                    (and_start, i.checked_sub(2).map(|j| tokens[j].1))
                }
                _ => (start, prev.map(|(_, t)| t)),
            };
            let clause_start = before.is_none_or(|t| t.ends_with([',', '.', ';', ':', '!', '?']));
            if (index == 0 && token.ends_with('.')) || (index > 0 && !clause_start) {
                continue;
            }
            if index == markers.len() {
                markers.push((marker_start, start + token.len()));
            } else if markers.len() >= self.min_counted_items {
                break;
            } else if index == 0 {
                markers.clear();
                markers.push((marker_start, start + token.len()));
            } else {
                markers.clear();
            }
        }
        if markers.len() < self.min_counted_items {
            return text.to_string();
        }

        let (last_item, trailing) = split_sentence(&text[markers[markers.len() - 1].1..]);
        let mut items: Vec<&str> = markers
            .windows(2)
            .map(|pair| &text[pair[0].1..pair[1].0])
            .collect();
        items.push(last_item);

        let mut out = String::new();
        let lead_in = text[..markers[0].0]
            .trim()
            .trim_end_matches(['.', ',', ';', ':']);
        if !lead_in.is_empty() {
            out.push_str(lead_in);
            out.push_str(":\n");
        }
        for (n, item) in items.iter().enumerate() {
            let item = item.trim().trim_end_matches([',', ';', '.']);
            if item.is_empty() {
                return text.to_string();
            }
            if n > 0 {
                out.push('\n');
            }
            let _ = write!(out, "{}. ", n + 1);
            push_capitalized(&mut out, item);
            if !item.ends_with(['?', '!']) {
                out.push('.');
            }
        }
        let trailing = trailing.trim();
        if !trailing.is_empty() {
            out.push('\n');
            out.push_str(&self.number_counted(trailing));
        }
        out
    }

    fn bulletize(&self, text: &str) -> String {
        text.lines()
            .map(|line| self.bulletize_line(line))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn bulletize_line(&self, line: &str) -> String {
        let Some((lead_in, after)) = line.split_once(": ") else {
            return line.to_string();
        };
        if lead_in.trim().is_empty() || lead_in.trim_start().starts_with("- ") {
            return line.to_string();
        }
        let (list, trailing) = split_sentence(after);
        let items = split_items(list.trim().trim_end_matches('.'));
        let announced = lead_in.trim_end().to_lowercase().ends_with("following");
        if !self.is_bullet_list(&items, announced) {
            return line.to_string();
        }

        let mut out = format!("{}:", lead_in.trim_end());
        for item in &items {
            out.push_str("\n- ");
            push_capitalized(&mut out, item);
        }
        let trailing = trailing.trim();
        if !trailing.is_empty() {
            out.push('\n');
            out.push_str(&self.bulletize_line(trailing));
        }
        out
    }

    /// `announced`: the lead-in ends with "the following", so items of any length qualify.
    fn is_bullet_list(&self, items: &[&str], announced: bool) -> bool {
        let min_items = if announced { 2 } else { self.min_bullet_items };
        if items.len() < min_items || items.iter().any(|i| i.contains(['.', ';', ':', '!', '?'])) {
            return false;
        }
        if announced {
            return true;
        }
        let words: Vec<usize> = items.iter().map(|i| i.split_whitespace().count()).collect();
        let longest_rest = words[1..].iter().copied().max().unwrap_or(0);
        words.iter().all(|&w| (1..=self.max_bullet_words).contains(&w))
            && words[0] <= longest_rest + 2
    }
}

impl Default for TextFormatter {
    fn default() -> Self {
        Self::new()
    }
}

/// Splits after the first sentence end (`.`, `!` or `?` followed by whitespace or the end).
fn split_sentence(text: &str) -> (&str, &str) {
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        let at_boundary = chars.peek().is_none_or(|(_, next)| next.is_whitespace());
        if matches!(c, '.' | '!' | '?') && at_boundary {
            return text.split_at(i + c.len_utf8());
        }
    }
    (text, "")
}

/// Splits "a, b, and c" or "a, b and c" into items; empty unless joined by "and" or "or".
fn split_items(list: &str) -> Vec<&str> {
    let mut items: Vec<&str> = list.split(", ").map(str::trim).collect();
    let Some(final_part) = items.pop() else {
        return items;
    };
    if let Some(tail) = final_part.strip_prefix("and ").or_else(|| final_part.strip_prefix("or ")) {
        items.push(tail);
    } else if let Some((a, b)) =
        final_part.rsplit_once(" and ").or_else(|| final_part.rsplit_once(" or "))
    {
        items.push(a);
        items.push(b);
    } else {
        items.clear();
    }
    items
}

/// Appends `text` to `out` with its first letter upper-cased.
fn push_capitalized(out: &mut String, text: &str) {
    let mut chars = text.chars();
    if let Some(first) = chars.next() {
        out.extend(first.to_uppercase());
        out.push_str(chars.as_str());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_fillers_and_their_commas() {
        let f = TextFormatter::new();
        assert_eq!(
            f.strip_fillers("Um, I think we should, uh, push the release to Monday."),
            "I think we should push the release to Monday."
        );
        assert_eq!(
            f.strip_fillers("Uh so the instructions must be clear."),
            "So the instructions must be clear."
        );
        assert_eq!(
            f.strip_fillers("Interesting. Uh maybe we can fix it."),
            "Interesting. Maybe we can fix it."
        );
        assert_eq!(
            f.strip_fillers("For instance, um it needs to remove uh uh things that uh it outputs."),
            "For instance, it needs to remove things that it outputs."
        );
        assert_eq!(f.strip_fillers("So, um, I guess we could go."), "So I guess we could go.");
        assert_eq!(f.strip_fillers("We'll see, um. Then we go."), "We'll see. Then we go.");
        assert_eq!(f.strip_fillers("Umbrella and uh-huh stay."), "Umbrella and uh-huh stay.");
        assert_eq!(f.strip_fillers("Uh."), "");
    }

    #[test]
    fn keeps_filler_spellings_that_are_words_in_other_languages() {
        let f = TextFormatter::new();
        for text in [
            "Morgen treffen wir uns um 10 Uhr im Büro.",
            "Er kommt später, er hat noch ein Meeting.",
        ] {
            assert_eq!(f.strip_fillers(text), text);
        }
    }

    #[test]
    fn numbers_spoken_counting() {
        let f = TextFormatter::new();
        assert_eq!(
            f.format_lists("We're saying one, we do that, two, we do that, three, we do that."),
            "We're saying:\n1. We do that.\n2. We do that.\n3. We do that."
        );
        assert_eq!(
            f.format_lists(
                "So the plan is. One, we refactor the parser. Two, we add more tests. \
                 Three, we ship it on Friday. Then we rest."
            ),
            "So the plan is:\n1. We refactor the parser.\n2. We add more tests.\n\
             3. We ship it on Friday.\nThen we rest."
        );
        assert_eq!(
            f.format_lists("First, open the settings. Second, turn on sync. And third, restart the app."),
            "1. Open the settings.\n2. Turn on sync.\n3. Restart the app."
        );
        assert_eq!(
            f.format_lists(
                "So the plan is: 1, we refactor the parser. 2, we add more tests. 3. We ship it on Friday."
            ),
            "So the plan is:\n1. We refactor the parser.\n2. We add more tests.\n3. We ship it on Friday."
        );
    }

    #[test]
    fn leaves_numbers_that_are_not_counting() {
        let f = TextFormatter::new();
        for text in [
            "I have one, maybe two, cats.",
            "One, I don't like it.",
            "The plan is:\n1. Refactor the parser.\n2. Add more tests.",
        ] {
            assert_eq!(f.format_lists(text), text);
        }
    }

    #[test]
    fn bullets_short_colon_lists() {
        let f = TextFormatter::new();
        assert_eq!(
            f.format_lists("There are a few things we need to buy: milk, eggs, bread, and coffee."),
            "There are a few things we need to buy:\n- Milk\n- Eggs\n- Bread\n- Coffee"
        );
        assert_eq!(
            f.format_lists("For the party we still need: chairs, plates, a speaker and some ice. Then we start."),
            "For the party we still need:\n- Chairs\n- Plates\n- A speaker\n- Some ice\nThen we start."
        );
        assert_eq!(
            f.format_lists(
                "Interesting. Maybe we can now also fix the following: move the overlay to the \
                 bottom of the screen, make the overlay show a volumeter instead of the listening \
                 blinking thing, and hopefully the overlay doesn't really interact with the rest of \
                 the page."
            ),
            "Interesting. Maybe we can now also fix the following:\n\
             - Move the overlay to the bottom of the screen\n\
             - Make the overlay show a volumeter instead of the listening blinking thing\n\
             - Hopefully the overlay doesn't really interact with the rest of the page"
        );
    }

    #[test]
    fn leaves_colon_sentences_that_are_not_lists() {
        let f = TextFormatter::new();
        for text in [
            "The problem is this: the build is slow, flaky, and expensive.",
            "Here's what I think: we should wait until the data comes in.",
            "We need: milk and eggs.",
            "For the party we still need:\n- Chairs\n- Plates\n- Some ice",
        ] {
            assert_eq!(f.format_lists(text), text);
        }
    }
}
