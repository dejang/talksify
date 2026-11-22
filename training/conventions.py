"""Conventions reconciliation mirroring ``src-tauri/core/src/conventions.rs``.

Applies Talksify's exact English number, percentage, and file-extension conventions so
fine-tuning targets match production post-processing cleanly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SMALL_NUMBERS: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}

_SCALES: dict[str, int] = {"hundred": 100, "thousand": 1_000, "million": 1_000_000}

_ORDINAL_OR_FRACTION: frozenset[str] = frozenset(
    {
        "first", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
        "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
        "eighteenth", "nineteenth", "twentieth", "thirtieth", "fortieth", "fiftieth", "sixtieth",
        "seventieth", "eightieth", "ninetieth", "hundredth", "thousandth", "millionth", "half",
        "halve", "quarter",
    }
)

_EXTENSIONS: dict[str, str] = {
    "wav": "wav",
    "wave": "wav",
    "txt": "txt",
    "text": "txt",
    "json": "json",
    "jason": "json",
    "md": "md",
    "pdf": "pdf",
    "csv": "csv",
    "rs": "rs",
    "ts": "ts",
    "tsx": "tsx",
    "js": "js",
    "py": "py",
    "toml": "toml",
    "yaml": "yaml",
    "yml": "yml",
    "html": "html",
    "css": "css",
    "png": "png",
    "jpg": "jpg",
    "mp3": "mp3",
}

_STANDALONE_EXT_PREFIXES: frozenset[str] = frozenset(
    {"a", "an", "the", "in", "as", "of", "to", "or", "and", "with", "from"}
)

_WORD_RE = re.compile(r"[0-9A-Za-z_']+")


@dataclass(frozen=True)
class _Word:
    text: str
    start: int
    end: int


def _extract_words(text: str) -> list[_Word]:
    return [_Word(m.group(0), m.start(), m.end()) for m in _WORD_RE.finditer(text)]


def _is_number(word: str) -> bool:
    return word in _SMALL_NUMBERS or word in _SCALES


def _under_hundred(parts: list[str]) -> int | None:
    if len(parts) == 1:
        return _SMALL_NUMBERS.get(parts[0])
    if len(parts) == 2:
        tens = _SMALL_NUMBERS.get(parts[0])
        units = _SMALL_NUMBERS.get(parts[1])
        if tens is not None and units is not None and tens >= 20 and tens % 10 == 0 and 1 <= units < 10:
            return tens + units
    return None


def _under_thousand(parts: list[str]) -> int | None:
    if len(parts) >= 2 and parts[1] == "hundred":
        unit = _SMALL_NUMBERS.get(parts[0])
        if unit is None or not (1 <= unit < 10):
            return None
        rest = parts[2:]
        if not rest:
            return unit * 100
        if rest[0] == "and":
            rest = rest[1:]
        tail = _under_hundred(rest)
        return unit * 100 + tail if tail is not None and tail > 0 else None
    return _under_hundred(parts)


def _cardinal(parts: list[str]) -> int | None:
    start = 0
    total = 0
    previous_scale = 10**18
    for idx, part in enumerate(parts):
        scale = _SCALES.get(part)
        if scale is None or scale <= 100:
            continue
        if scale >= previous_scale:
            return None
        group = _under_thousand(parts[start:idx])
        if group is None or group == 0:
            return None
        total += group * scale
        previous_scale = scale
        start = idx + 1
    rest = parts[start:]
    if not rest:
        return total if total > 0 else None
    if total > 0 and rest[0] == "and":
        rest = rest[1:]
    tail = _under_thousand(rest)
    if tail is None:
        return None
    if total == 0 or tail > 0:
        return total + tail
    return None


def _gap_before(text: str, words: list[_Word], index: int) -> str:
    start = 0 if index == 0 else words[index - 1].end
    return text[start : words[index].start]


def _is_space(gap: str) -> bool:
    return bool(gap) and gap.isspace()


def _is_number_gap(gap: str) -> bool:
    return _is_space(gap) or gap == "-"


def _is_ordinal_or_fraction(word: str) -> bool:
    singular = word[:-1] if word.endswith("s") else word
    return singular in _ORDINAL_OR_FRACTION


def _percent_end(text: str, words: list[_Word], index: int) -> int | None:
    if index >= len(words) or not _is_space(_gap_before(text, words, index)):
        return None
    nxt = words[index].text.lower()
    if nxt == "percent":
        return index + 1
    if index + 1 < len(words):
        cent = words[index + 1].text.lower()
        if nxt == "per" and cent == "cent" and _is_space(_gap_before(text, words, index + 1)):
            return index + 2
    return None


def _is_mixed_fraction(text: str, words: list[_Word], parts: list[str], index: int) -> bool:
    if index >= len(parts) or parts[index] != "and" or not _is_space(_gap_before(text, words, index)):
        return False
    i = index + 1
    while i < len(parts) and _is_number_gap(_gap_before(text, words, i)):
        w = parts[i]
        if _is_ordinal_or_fraction(w) and w != "first":
            return True
        if not (_is_number(w) or (i == index + 1 and w == "a")):
            break
        i += 1
    return False


def _apply_numbers(text: str) -> str:
    words = _extract_words(text)
    lower = [w.text.lower() for w in words]
    out: list[str] = []
    cursor = 0
    i = 0
    while i < len(words):
        start = i
        digits = lower[i].isascii() and lower[i].isdigit()
        if not digits and not _is_number(lower[i]):
            i += 1
            continue
        i += 1
        if not digits:
            while i < len(words) and _is_number_gap(_gap_before(text, words, i)):
                if _is_number(lower[i]):
                    i += 1
                elif (
                    lower[i] == "and"
                    and lower[i - 1] in _SCALES
                    and i + 1 < len(words)
                    and _is_number_gap(_gap_before(text, words, i + 1))
                ):
                    if _is_ordinal_or_fraction(lower[i + 1]) or lower[i + 1] == "second":
                        i += 2
                        break
                    if not _is_number(lower[i + 1]):
                        break
                    end = i + 2
                    while (
                        end < len(words)
                        and _is_number(lower[end])
                        and _is_number_gap(_gap_before(text, words, end))
                    ):
                        end += 1
                    if _cardinal(lower[start:i]) is not None and _cardinal(lower[start:end]) is None:
                        break
                    i = end
                else:
                    break

        percent = _percent_end(text, words, i)
        before = start - 1 if start > 0 and _is_space(_gap_before(text, words, start)) else None
        after_point = i < len(lower) and lower[i] == "point" and _is_space(_gap_before(text, words, i))
        after_ord_or_frac = i < len(lower) and (
            (_is_ordinal_or_fraction(lower[i]) and _is_number_gap(_gap_before(text, words, i)))
            or (lower[i] == "second" and _gap_before(text, words, i) == "-")
        )
        excluded = (
            (before is not None and lower[before] == "point")
            or after_point
            or after_ord_or_frac
            or _is_mixed_fraction(text, words, lower, i)
        )
        value = words[start].text if digits else (
            str(n) if (n := _cardinal(lower[start:i])) is not None else None
        )
        lone_one = i == start + 1 and lower[start] == "one" and percent is None
        if not excluded and not lone_one and value is not None:
            end = percent if percent is not None else i
            out.append(text[cursor : words[start].start])
            out.append(value)
            if percent is not None:
                out.append("%")
            cursor = words[end - 1].end
            i = end
    out.append(text[cursor:])
    return "".join(out)


def _apply_extensions(text: str) -> str:
    words = _extract_words(text)
    out: list[str] = []
    cursor = 0
    for i in range(len(words) - 1):
        dot, ext = words[i], words[i + 1]
        if dot.text.lower() != "dot" or not _is_space(_gap_before(text, words, i + 1)):
            continue
        ext_name = _EXTENSIONS.get(ext.text.lower())
        if ext_name is None:
            continue
        start = dot.start
        if i > 0:
            prev = words[i - 1]
            gap = _gap_before(text, words, i)
            if prev.text.lower() not in _STANDALONE_EXT_PREFIXES and _is_space(gap) and "\n" not in gap and "\r" not in gap:
                start = prev.end
        out.append(text[cursor:start])
        out.append(f".{ext_name}")
        cursor = ext.end
    out.append(text[cursor:])
    return "".join(out)


def apply_talksify_conventions(text: str) -> str:
    """Apply Talksify's exact English number, percentage, and file-extension conventions."""
    return _apply_extensions(_apply_numbers(text))
