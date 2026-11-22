"""Dataset quality assurance filters and language-drift validation for Talksify fine-tuning.

Guards training data against all 5 ``F-406`` failure modes and round-2 dataset defects:
1. Empty sentinel targets (``EMPTY`` / ``category == "empty"``)
2. Passthrough invariant enforcement (``raw == clean`` for passthrough categories)
3. Restructured bullet/numbered lists, introduced newlines, and unwanted blank-line breaks
4. Non-subsequence rephrasing/reordering, over-deletion, and dropped opening clauses
5. Assistant chatter, duplicated clauses, leftover fillers, relocated cues, and English drift
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any

from conventions import apply_talksify_conventions
from prompts import PARAKEET_V3_LANGUAGES

PASSTHROUGH_CATEGORIES: frozenset[str] = frozenset(
    {"clean_passthrough", "passthrough", "no_edit"}
)

RETRACTION_CATEGORIES: frozenset[str] = frozenset(
    {
        "scoped_self_correction",
        "self_correction",
        "self_corrections_heavy",
        "false_start_collapse",
        "backtrack_phrase",
        "mixed_l2",
        "thinking_aloud",
        "native_de_disfluency",
    }
)

_HOMOPHONE_CATEGORIES: frozenset[str] = frozenset({"homophone_fix", "mixed_l1"})


def is_retraction_category(category: str, source: str = "") -> bool:
    """Single source of truth deciding whether a row is a retraction across loaders and repair."""
    cat_lower = category.strip().lower()
    if (
        cat_lower in PASSTHROUGH_CATEGORIES
        or cat_lower == "disfl_qa_question"
        or source == "google-research-datasets/disfl_qa"
    ):
        return False
    return cat_lower in RETRACTION_CATEGORIES

_CHATTER_PREFIXES: tuple[str, ...] = (
    "here is the cleaned",
    "here's the cleaned",
    "cleaned transcript:",
    "cleaned text:",
    "sure, here",
    "certainly,",
    "of course,",
    "as an ai",
    "i cannot ",
    "i can't answer",
    "output:",
    "result:",
)

_FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "<transcript>",
    "</transcript>",
    "<think>",
    "</think>",
    "```",
    "<|im_start|>",
    "<|im_end|>",
)

# Unambiguous hesitation fillers per language (avoiding real words like German "um"/"er",
# Spanish demonstrative "este", Croatian "ovaj", Hungarian "izé", Romanian "păi").
_UNAMBIGUOUS_FILLERS_BY_LANG: dict[str, frozenset[str]] = {
    "en": frozenset({"uh", "uhh", "uhm", "um", "umm", "erm", "hmm", "hm", "mhm"}),
    "de": frozenset({"äh", "ähm", "ähh", "öhm", "öh", "hmm"}),
    "fr": frozenset({"euh", "euhh", "heu", "hmm"}),
    "es": frozenset({"eh", "ehh", "ehm", "mmm"}),
    "it": frozenset({"ehm", "ehh", "mmm", "uhm"}),
    "ro": frozenset({"ăă", "ăăă", "îî", "îîî", "ehm", "mmm"}),
    "bg": frozenset({"ъъ", "ъъъ", "хм"}),
    "hr": frozenset({"aaa", "eee", "hmm"}),
    "cs": frozenset({"eee", "ehm", "hmm"}),
    "da": frozenset({"øh", "øhm", "øhh", "hmm"}),
    "nl": frozenset({"uh", "uhm", "ehm", "hmm"}),
    "et": frozenset({"eee", "öö", "ehm", "hmm"}),
    "fi": frozenset({"öö", "ööö", "öhm", "ehm", "hmm"}),
    "el": frozenset({"εε", "εεε", "εμ", "χμ", "χμμ"}),
    "hu": frozenset({"öö", "ööö", "őő", "hmm"}),
    "lv": frozenset({"ēē", "ēēē", "ehm", "hmm"}),
    "lt": frozenset({"eee", "ehm", "aaa", "hmm"}),
    "mt": frozenset({"emm", "ehh", "ehm", "hmm"}),
    "pl": frozenset({"yyy", "yy", "eee", "ehm", "hmm"}),
    "pt": frozenset({"hã", "éé", "ehm", "hmm"}),
    "ru": frozenset({"ээ", "эээ", "мм", "ммм", "хм"}),
    "sk": frozenset({"eee", "ehm", "hmm"}),
    "sl": frozenset({"eee", "emm", "ehm", "hmm"}),
    "sv": frozenset({"öh", "öhm", "ehm", "hmm"}),
    "uk": frozenset({"ее", "еее", "ем", "ммм", "хм"}),
}

_EMAIL_SIGNATURE_RE = re.compile(
    r"(?:^|\n)\s*(?:best regards|kind regards|warm regards|sincerely|cheers|subject:|dear\s+[A-Z])",
    re.IGNORECASE,
)

_LIST_LINE_RE = re.compile(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+")
_INLINE_NUMBERED_LIST_RE = re.compile(r"(?:^|\s)1[.)]\s+.+\s2[.)]\s+")
_UNICODE_WORD_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)


def normalize_whitespace(text: str) -> str:
    """Normalize trailing spaces per line and outer whitespace while preserving single newlines."""
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.strip().splitlines()]
    return "\n".join(lines).strip()


def is_empty_sentinel(clean: str, category: str = "") -> bool:
    """Return True if a row uses the synthetic ``EMPTY`` sentinel or ``empty`` category."""
    if category.strip().lower() == "empty":
        return True
    return clean.strip(" .!?,;:\"'()[]").upper() == "EMPTY"


def has_assistant_chatter(text: str) -> bool:
    """Return True if ``text`` contains assistant framing, tags, or markdown fences."""
    lower = text.strip().lower()
    if any(tok in lower for tok in _FORBIDDEN_SUBSTRINGS):
        return True
    return any(lower.startswith(prefix) for prefix in _CHATTER_PREFIXES)


def has_email_block(text: str) -> bool:
    """Return True if ``text`` looks like an email block/signature excluded from dictation."""
    return bool(_EMAIL_SIGNATURE_RE.search(text))


def has_unwanted_paragraph_breaks(raw: str, clean: str) -> bool:
    """Reject spontaneous blank-line paragraph splits injected on single-topic prose."""
    return "\n\n" in clean and "\n\n" not in raw


def has_introduced_list_or_newlines(raw: str, clean: str) -> bool:
    """Reject targets that restructure prose into bullet/numbered lists or inject new line breaks."""
    if _LIST_LINE_RE.search(clean) and not _LIST_LINE_RE.search(raw):
        return True
    if _INLINE_NUMBERED_LIST_RE.search(clean) and not _INLINE_NUMBERED_LIST_RE.search(raw):
        return True
    return "\n" in clean and "\n" not in raw


def has_clause_duplication(text: str, min_words: int = 5) -> bool:
    """Detect hallucinated consecutive duplicated multi-word clauses (e.g. F-406 13-35-46)."""
    tokens = [t.lower().strip(".,;:!?\"'()[]—–-") for t in text.split()]
    tokens = [t for t in tokens if t]
    n = len(tokens)
    if n < min_words * 2:
        return False
    for span in range(min_words, min(20, n // 2) + 1):
        for i in range(n - 2 * span + 1):
            if tokens[i : i + span] == tokens[i + span : i + 2 * span]:
                return True
    return False


def has_leftover_fillers(clean: str, lang: str) -> bool:
    """Return True if unambiguous hesitation fillers remain in the cleaned target."""
    fillers = _UNAMBIGUOUS_FILLERS_BY_LANG.get(lang, _UNAMBIGUOUS_FILLERS_BY_LANG["en"])
    for raw_tok in clean.split():
        core = raw_tok.strip(".,;:!?\"'()[]—–-").lower()
        if core in fillers:
            return True
    return False


def has_relocated_cue_prefix(raw: str, clean: str, lang: str) -> bool:
    """Reject outputs that relocate a mid-sentence self-correction cue to the start of the output."""
    spec = PARAKEET_V3_LANGUAGES.get(lang)
    if spec is None or not spec.relocated_cue_prefixes:
        return False
    raw_lower = raw.strip().lower()
    clean_lower = clean.strip().lower()
    for prefix in spec.relocated_cue_prefixes:
        if clean_lower.startswith(prefix) and not raw_lower.startswith(prefix):
            return True
    return False


def _strip_diacritics(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))


def _normalized_word_tokens(text: str, lang: str) -> list[str]:
    if lang == "en":
        text = apply_talksify_conventions(text)
    return [_strip_diacritics(tok) for tok in _UNICODE_WORD_RE.findall(text.lower())]


def _lcs_matched_raw_indices(seq_a: list[str], seq_b: list[str]) -> list[int]:
    """Return sorted indices in ``seq_a`` belonging to a longest common subsequence with ``seq_b``."""
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return []
    if n * m > 400_000:
        idx = 0
        matched: list[int] = []
        for tok in seq_b:
            while idx < n and seq_a[idx] != tok:
                idx += 1
            if idx < n:
                matched.append(idx)
                idx += 1
        return matched

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        tok_a = seq_a[i]
        row_i = dp[i]
        row_next = dp[i + 1]
        for j in range(m - 1, -1, -1):
            if tok_a == seq_b[j]:
                row_i[j] = row_next[j + 1] + 1
            else:
                down = row_next[j]
                right = row_i[j + 1]
                row_i[j] = down if down >= right else right

    matched_raw: list[int] = []
    i = j = 0
    while i < n and j < m:
        if seq_a[i] == seq_b[j]:
            matched_raw.append(i)
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return matched_raw


@dataclass(frozen=True)
class TokenAlignment:
    """Order-preserving token alignment metrics between raw and cleaned transcripts."""

    clean_len: int
    lcs: int
    added_words: tuple[str, ...]
    truly_added_words: tuple[str, ...]
    moved_words: int
    max_deleted_span: int

    @property
    def ratio(self) -> float:
        return self.lcs / self.clean_len if self.clean_len else 0.0

    def is_valid(self, *, is_retraction: bool = False, category: str = "") -> tuple[bool, str]:
        """Verify target neither invents words, reorders clauses, nor deletes over-scoped spans."""
        if self.clean_len == 0:
            return False, "empty token sequence"
        cat_lower = category.strip().lower()
        max_truly_added = 1 if cat_lower in _HOMOPHONE_CATEGORIES else 0
        if len(self.truly_added_words) > max_truly_added or len(self.added_words) > 1:
            return False, f"added words not in input ({', '.join(self.added_words[:4])})"
        max_moved = 2 if is_retraction else 1
        if self.moved_words > max_moved:
            return False, f"reordered or moved words ({self.moved_words} > {max_moved})"
        if self.clean_len < 10:
            if (self.clean_len - self.lcs) > 1:
                return False, f"short target not near-subsequence ({self.lcs}/{self.clean_len})"
        elif self.ratio < 0.90:
            return False, f"target not near-subsequence ({self.ratio:.2f} < 0.90)"
        max_del_limit = 16 if is_retraction else 10
        if self.max_deleted_span > max_del_limit:
            return False, f"over-scoped deleted span ({self.max_deleted_span} > {max_del_limit})"
        return True, "ok"


def analyze_token_alignment(raw: str, clean: str, lang: str) -> TokenAlignment:
    """Compute exact LCS alignment, added words, moved words, and max deleted non-filler span."""
    raw_tokens = _normalized_word_tokens(raw, lang)
    clean_tokens = _normalized_word_tokens(clean, lang)
    n, m = len(raw_tokens), len(clean_tokens)
    if m == 0:
        return TokenAlignment(
            clean_len=0,
            lcs=0,
            added_words=(),
            truly_added_words=(),
            moved_words=0,
            max_deleted_span=n,
        )

    matched_raw = _lcs_matched_raw_indices(raw_tokens, clean_tokens)
    lcs = len(matched_raw)

    rem = Counter(raw_tokens)
    added_all: list[str] = []
    for tok in clean_tokens:
        if rem[tok] > 0:
            rem[tok] -= 1
        else:
            added_all.append(tok)

    added_words = [w for w in added_all if any(ch.isalpha() for ch in w)]
    moved_words = (m - len(added_all)) - lcs

    matched_set = set(matched_raw)
    deleted_raw = [raw_tokens[i] for i in range(n) if i not in matched_set]
    truly_added: list[str] = []
    for word in added_words:
        is_stem_variant = len(word) >= 4 and any(
            len(d) >= 4
            and (d.startswith(word[:4]) or word.startswith(d[:4]))
            and abs(len(d) - len(word)) <= 3
            for d in deleted_raw
        )
        if not is_stem_variant:
            truly_added.append(word)

    fillers_folded = {
        _strip_diacritics(f)
        for f in _UNAMBIGUOUS_FILLERS_BY_LANG.get(lang, _UNAMBIGUOUS_FILLERS_BY_LANG["en"])
    }
    if matched_raw:
        gaps = [sum(1 for t in raw_tokens[: matched_raw[0]] if t not in fillers_folded)]
        for k in range(1, len(matched_raw)):
            gaps.append(
                sum(
                    1
                    for t in raw_tokens[matched_raw[k - 1] + 1 : matched_raw[k]]
                    if t not in fillers_folded
                )
            )
        gaps.append(sum(1 for t in raw_tokens[matched_raw[-1] + 1 :] if t not in fillers_folded))
        max_deleted_span = max(gaps)
    else:
        max_deleted_span = sum(1 for t in raw_tokens if t not in fillers_folded)

    return TokenAlignment(
        clean_len=m,
        lcs=lcs,
        added_words=tuple(added_words),
        truly_added_words=tuple(truly_added),
        moved_words=moved_words,
        max_deleted_span=max_deleted_span,
    )


def drops_opening_clause(raw: str, clean: str, lang: str) -> bool:
    """Detect non-retraction targets that silently delete a multi-word spoken opening clause."""
    fillers = {
        _strip_diacritics(f)
        for f in _UNAMBIGUOUS_FILLERS_BY_LANG.get(lang, _UNAMBIGUOUS_FILLERS_BY_LANG["en"])
    }
    raw_tokens = [t for t in _normalized_word_tokens(raw, lang) if t not in fillers]
    clean_tokens = _normalized_word_tokens(clean, lang)
    if len(raw_tokens) < 10 or len(clean_tokens) < 4:
        return False
    first, second = clean_tokens[0], clean_tokens[1]
    for idx in range(len(raw_tokens) - 1):
        if raw_tokens[idx] == first and raw_tokens[idx + 1] == second:
            return idx >= 5
    return False


def script_matches(text: str, expected_script: str) -> bool:
    """Verify the dominant alphabetic script matches ``latin``, ``cyrillic``, or ``greek``."""
    counts = {"latin": 0, "cyrillic": 0, "greek": 0}
    for ch in text:
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        if "CYRILLIC" in name:
            counts["cyrillic"] += 1
        elif "GREEK" in name:
            counts["greek"] += 1
        elif "LATIN" in name:
            counts["latin"] += 1
    total = sum(counts.values())
    if total == 0:
        return False
    return counts.get(expected_script, 0) / total >= 0.65


class LanguageValidator:
    """Detects English-translation drift and verifies target language across all 25 Parakeet languages."""

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self._detector: Any = None
        self._lang_enum_by_code: dict[str, Any] = {}
        if enabled:
            self._init_detector()

    def _init_detector(self) -> None:
        try:
            from lingua import IsoCode639_1, Language, LanguageDetectorBuilder
        except ImportError:
            self._detector = None
            return

        langs: list[Any] = []
        for code in PARAKEET_V3_LANGUAGES:
            iso = getattr(IsoCode639_1, code.upper(), None)
            if iso is None:
                continue
            try:
                lang_obj = Language.from_iso_code_639_1(iso)
            except Exception:
                continue
            if lang_obj is not None:
                self._lang_enum_by_code[code] = lang_obj
                langs.append(lang_obj)

        if langs:
            self._detector = LanguageDetectorBuilder.from_languages(*langs).build()

    def check_language(self, text: str, expected_code: str) -> tuple[bool, str]:
        """Verify ``text`` stays strictly in ``expected_code`` with zero English-translation drift."""
        spec = PARAKEET_V3_LANGUAGES.get(expected_code)
        if spec is None:
            return False, f"unknown language code {expected_code!r}"
        if not script_matches(text, spec.script):
            return False, f"script mismatch for {expected_code} (expected {spec.script})"
        if not self._enabled or self._detector is None or expected_code == "en":
            return True, "ok"

        alpha_words = [w for w in text.split() if any(ch.isalpha() for ch in w)]
        if len(alpha_words) < 5:
            return True, "ok"

        en_enum = self._lang_enum_by_code.get("en")
        target_enum = self._lang_enum_by_code.get(expected_code)
        confidences = self._detector.compute_language_confidence_values(text)
        conf_map = {cv.language: cv.value for cv in confidences}

        en_conf = conf_map.get(en_enum, 0.0) if en_enum is not None else 0.0
        if target_enum is None:
            if en_conf >= 0.55:
                return False, f"english drift detected for {expected_code} (en_conf={en_conf:.2f})"
            return True, "ok"

        target_conf = conf_map.get(target_enum, 0.0)
        if en_conf > max(0.35, target_conf * 1.15):
            return (
                False,
                f"english drift detected for {expected_code} "
                f"(en_conf={en_conf:.2f}, target_conf={target_conf:.2f})",
            )
        top_lang = confidences[0].language if confidences else None
        if top_lang != target_enum and target_conf < 0.08:
            return (
                False,
                f"language mismatch for {expected_code}: top={top_lang}, "
                f"target_conf={target_conf:.2f}",
            )
        return True, "ok"


def validate_pair(
    raw: str,
    clean: str,
    lang: str,
    *,
    is_retraction: bool = False,
    is_passthrough: bool = False,
    category: str = "",
    allow_email: bool = False,
    lang_validator: LanguageValidator | None = None,
) -> tuple[bool, str]:
    """Validate one ``(raw, clean)`` training pair against Talksify quality invariants."""
    raw_s = normalize_whitespace(raw)
    clean_s = normalize_whitespace(clean)
    cat_lower = category.strip().lower()
    if not raw_s or not clean_s:
        return False, "empty input or output"
    if is_empty_sentinel(clean_s, cat_lower):
        return False, "empty sentinel target"
    if cat_lower == "shopping_lists":
        return False, "restructured shopping list"
    if is_passthrough or cat_lower in PASSTHROUGH_CATEGORIES:
        if raw_s != clean_s:
            return False, "passthrough raw != clean"
    if has_assistant_chatter(clean_s):
        return False, "assistant chatter or framing tags"
    if not allow_email and has_email_block(clean_s):
        return False, "email block or signature"
    if has_unwanted_paragraph_breaks(raw_s, clean_s):
        return False, "unwanted paragraph break injection"
    if has_introduced_list_or_newlines(raw_s, clean_s):
        return False, "introduced list or newline formatting"
    if has_clause_duplication(clean_s):
        return False, "duplicated clause hallucination"
    if has_leftover_fillers(clean_s, lang):
        return False, f"leftover hesitation filler for {lang}"
    if has_relocated_cue_prefix(raw_s, clean_s, lang):
        return False, f"relocated self-correction cue prefix for {lang}"

    raw_words = raw_s.split()
    clean_words = clean_s.split()
    if len(raw_words) == 0 or len(clean_words) == 0:
        return False, "zero words"

    ratio = len(clean_words) / len(raw_words)
    if is_retraction:
        min_ratio = 0.50 if len(raw_words) >= 18 else 0.42
    else:
        min_ratio = 0.65 if len(raw_words) >= 12 else 0.60
    max_ratio = 1.25 if len(raw_words) >= 8 else 1.50
    if ratio < min_ratio:
        return False, f"over-deletion (word ratio {ratio:.2f} < {min_ratio:.2f})"
    if ratio > max_ratio:
        return False, f"hallucinated expansion (word ratio {ratio:.2f} > {max_ratio:.2f})"

    if raw_s != clean_s:
        align_ok, align_reason = analyze_token_alignment(raw_s, clean_s, lang).is_valid(
            is_retraction=is_retraction, category=cat_lower
        )
        if not align_ok:
            return False, align_reason
        if (
            not is_retraction
            and cat_lower != "disfl_qa_question"
            and drops_opening_clause(raw_s, clean_s, lang)
        ):
            return False, "dropped opening clause"

    if lang_validator is not None:
        ok, reason = lang_validator.check_language(clean_s, lang)
        if not ok:
            return False, reason

    return True, "ok"
