"""Prompt, chat-template, and Parakeet v3 25-language definitions for Talksify fine-tuning.

This module is the single source of truth aligning training inputs with Talksify's production
inference pipeline in ``src-tauri/core/src/llm_postprocessor.rs`` and
``src-tauri/core/src/llm_backend.rs``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
LLM_POSTPROCESSOR_RS = REPO_ROOT / "src-tauri" / "core" / "src" / "llm_postprocessor.rs"
BASE_MODEL_ID = "Qwen/Qwen3.5-0.8B"
EXPECTED_GGUF_TENSORS_NO_MTP = 320
MAX_GGUF_BYTES = 1_000_000_000


def parse_rust_string_literal(source: str, open_quote_idx: int) -> str:
    """Parse a Rust normal string literal starting at ``open_quote_idx`` (the opening ``\"``)."""
    if source[open_quote_idx] != '"':
        raise ValueError(f"Expected opening quote at index {open_quote_idx}")
    out: list[str] = []
    i = open_quote_idx + 1
    n = len(source)
    while i < n:
        ch = source[i]
        if ch == '"':
            return "".join(out)
        if ch == "\\":
            i += 1
            if i >= n:
                raise ValueError("Unterminated escape in Rust string literal")
            esc = source[i]
            if esc == "n":
                out.append("\n")
                i += 1
            elif esc == "r":
                out.append("\r")
                i += 1
            elif esc == "t":
                out.append("\t")
                i += 1
            elif esc == '"':
                out.append('"')
                i += 1
            elif esc == "\\":
                out.append("\\")
                i += 1
            elif esc == "\n" or (esc == "\r" and i + 1 < n and source[i + 1] == "\n"):
                # Rust string line continuation: consume newline and following ASCII whitespace.
                if esc == "\r":
                    i += 2
                else:
                    i += 1
                while i < n and source[i] in (" ", "\t", "\n", "\r"):
                    i += 1
            else:
                raise ValueError(f"Unsupported Rust string escape \\{esc!r}")
        else:
            out.append(ch)
            i += 1
    raise ValueError("Unterminated Rust string literal")


def extract_clean_prompt_from_rust(rust_path: Path = LLM_POSTPROCESSOR_RS) -> str:
    """Extract the exact ``clean_prompt`` string literal from ``llm_postprocessor.rs``."""
    text = rust_path.read_text(encoding="utf-8")
    marker = "clean_prompt:"
    idx = text.find(marker)
    if idx == -1:
        raise ValueError(f"Could not find {marker!r} in {rust_path}")
    quote_idx = text.find('"', idx + len(marker))
    if quote_idx == -1:
        raise ValueError(f"Could not find opening quote after {marker!r} in {rust_path}")
    return parse_rust_string_literal(text, quote_idx)


def load_clean_prompt(rust_path: Path = LLM_POSTPROCESSOR_RS) -> str:
    """Return Talksify's exact production ``clean_prompt`` directly from Rust source."""
    return extract_clean_prompt_from_rust(rust_path)


def format_user_message(raw_transcript: str) -> str:
    """Wrap raw transcript identically to ``MistralBackend::complete`` in ``llm_backend.rs``."""
    return f"<transcript>\n{raw_transcript}\n</transcript>"


def build_messages(
    raw_transcript: str,
    clean_transcript: str | None = None,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    """Build the chat message list matching ``src-tauri/core/src/llm_backend.rs``."""
    prompt = system_prompt if system_prompt is not None else load_clean_prompt()
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": format_user_message(raw_transcript)},
    ]
    if clean_transcript is not None:
        messages.append({"role": "assistant", "content": clean_transcript})
    return messages


def render_chat_pair(
    tokenizer: Any,
    raw_transcript: str,
    clean_transcript: str,
    system_prompt: str | None = None,
) -> tuple[str, str]:
    """Render ``(prompt_prefix, full_text)`` with ``enable_thinking=False``.

    Guarantees that ``full_text`` starts with the exact ``prompt_prefix`` generated at inference
    time (including ``<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n`` for Qwen3.5).
    """
    prompt_msgs = build_messages(raw_transcript, clean_transcript=None, system_prompt=system_prompt)
    full_msgs = build_messages(
        raw_transcript, clean_transcript=clean_transcript, system_prompt=system_prompt
    )
    prompt_prefix = tokenizer.apply_chat_template(
        prompt_msgs,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full_text = tokenizer.apply_chat_template(
        full_msgs,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if not full_text.startswith(prompt_prefix):
        raise ValueError(
            "Chat template mismatch between add_generation_prompt=True and full assistant turn. "
            f"Prefix tail: {prompt_prefix[-80:]!r} vs full slice: "
            f"{full_text[max(0, len(prompt_prefix) - 80):len(prompt_prefix) + 40]!r}"
        )
    return prompt_prefix, full_text


def tokenize_for_sft(
    tokenizer: Any,
    raw_transcript: str,
    clean_transcript: str,
    system_prompt: str | None = None,
    max_seq_len: int = 1536,
) -> dict[str, Any] | None:
    """Tokenize one example with strict completions-only loss masking.

    System prompt, user ``<transcript>`` message, and the assistant empty-think prefix
    (``<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n``) are all masked to ``-100`` so loss
    is computed strictly on the cleaned transcript tokens and closing ``<|im_end|>``.

    Returns ``None`` if the tokenized sequence exceeds ``max_seq_len`` (never truncates mid-output).
    """
    prompt_prefix, full_text = render_chat_pair(
        tokenizer, raw_transcript, clean_transcript, system_prompt=system_prompt
    )
    completion_text = full_text[len(prompt_prefix) :]
    if not completion_text:
        raise ValueError("Empty completion text after prompt prefix")

    prompt_ids: list[int] = tokenizer.encode(prompt_prefix, add_special_tokens=False)
    completion_ids: list[int] = tokenizer.encode(completion_text, add_special_tokens=False)
    input_ids = prompt_ids + completion_ids
    if len(input_ids) > max_seq_len:
        return None

    labels = [-100] * len(prompt_ids) + list(completion_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "prompt_len": len(prompt_ids),
        "completion_len": len(completion_ids),
    }


@dataclass(frozen=True)
class LanguageSpec:
    """Metadata for one of the 25 European languages supported by Parakeet TDT 0.6B v3."""

    code: str
    name: str
    tier: str  # "en", "tier1", or "tier2"
    script: str  # "latin", "cyrillic", or "greek"
    fillers: tuple[str, ...]
    correction_cues: tuple[str, ...]
    # Leading words that must never be relocated to the start of a cleaned sentence
    # when a self-correction cue is stripped (e.g. Romanian "nu, scuze," -> never "Scuze, ...").
    relocated_cue_prefixes: tuple[str, ...] = ()


PARAKEET_V3_LANGUAGES: dict[str, LanguageSpec] = {
    "en": LanguageSpec(
        code="en",
        name="English",
        tier="en",
        script="latin",
        fillers=("uh", "uhh", "uhm", "um", "umm", "erm", "hmm", "hm", "mhm"),
        correction_cues=(
            "no, sorry",
            "wait, no",
            "actually no",
            "scratch that",
            "I mean",
            "wait, I meant",
            "never mind",
        ),
        relocated_cue_prefixes=("sorry,", "scratch that,"),
    ),
    # Tier 1 (~60% of multilingual slice)
    "de": LanguageSpec(
        code="de",
        name="German",
        tier="tier1",
        script="latin",
        fillers=("äh", "ähm", "ähh", "öhm", "öh", "hm", "hmm"),
        correction_cues=(
            "nein, sorry",
            "also nein",
            "ach nein",
            "warte, nein",
            "ich meine",
            "nein, warte",
        ),
        relocated_cue_prefixes=("sorry,", "entschuldigung,"),
    ),
    "fr": LanguageSpec(
        code="fr",
        name="French",
        tier="tier1",
        script="latin",
        fillers=("euh", "euhh", "heu", "hmm", "bah"),
        correction_cues=(
            "enfin non",
            "non, pardon",
            "pardon",
            "attends, non",
            "je veux dire",
            "plutôt",
        ),
        relocated_cue_prefixes=("pardon,", "désolé,"),
    ),
    "es": LanguageSpec(
        code="es",
        name="Spanish",
        tier="tier1",
        script="latin",
        fillers=("eh", "ehh", "ehm", "mmm", "este"),
        correction_cues=(
            "no, perdón",
            "perdón",
            "o sea, no",
            "espera, no",
            "mejor dicho",
            "digo",
        ),
        relocated_cue_prefixes=("perdón,", "disculpa,"),
    ),
    "it": LanguageSpec(
        code="it",
        name="Italian",
        tier="tier1",
        script="latin",
        fillers=("ehm", "eh", "ehh", "mmm", "uhm"),
        correction_cues=(
            "anzi no",
            "no, scusa",
            "aspetta, no",
            "volevo dire",
            "cioè no",
        ),
        relocated_cue_prefixes=("scusa,", "scusami,"),
    ),
    "ro": LanguageSpec(
        code="ro",
        name="Romanian",
        tier="tier1",
        script="latin",
        fillers=("ăă", "ăăă", "ă", "îî", "îîî", "păi", "ehm", "mmm"),
        correction_cues=(
            "nu, scuze",
            "stai, nu",
            "adică nu",
            "pardon",
            "vreau să spun",
            "ba nu",
        ),
        relocated_cue_prefixes=("scuze,", "pardon,"),
    ),
    # Tier 2 (~40% of multilingual slice — remaining 19 Parakeet TDT 0.6B v3 languages)
    "bg": LanguageSpec(
        code="bg",
        name="Bulgarian",
        tier="tier2",
        script="cyrillic",
        fillers=("ъъ", "ъъъ", "ъ", "ами", "хм"),
        correction_cues=("не, извинявай", "чакай, не", "всъщност не", "искам да кажа"),
        relocated_cue_prefixes=("извинявай,",),
    ),
    "hr": LanguageSpec(
        code="hr",
        name="Croatian",
        tier="tier2",
        script="latin",
        fillers=("ovaj", "aaa", "eee", "hm", "hmm"),
        correction_cues=("ne, oprosti", "čekaj, ne", "zapravo ne", "mislim"),
        relocated_cue_prefixes=("oprosti,",),
    ),
    "cs": LanguageSpec(
        code="cs",
        name="Czech",
        tier="tier2",
        script="latin",
        fillers=("eee", "ehm", "eh", "hmm", "jakože"),
        correction_cues=("ne, promiň", "počkej, ne", "vlastně ne", "tedy"),
        relocated_cue_prefixes=("promiň,",),
    ),
    "da": LanguageSpec(
        code="da",
        name="Danish",
        tier="tier2",
        script="latin",
        fillers=("øh", "øhm", "øhh", "hm", "hmm"),
        correction_cues=("nej, undskyld", "vent, nej", "altså nej", "jeg mener"),
        relocated_cue_prefixes=("undskyld,",),
    ),
    "nl": LanguageSpec(
        code="nl",
        name="Dutch",
        tier="tier2",
        script="latin",
        fillers=("eh", "uh", "uhm", "ehm", "hmm"),
        correction_cues=("nee, sorry", "wacht, nee", "of nee", "ik bedoel"),
        relocated_cue_prefixes=("sorry,",),
    ),
    "et": LanguageSpec(
        code="et",
        name="Estonian",
        tier="tier2",
        script="latin",
        fillers=("ee", "eee", "öö", "ehm", "hmm"),
        correction_cues=("ei, vabandust", "oota, ei", "tähendab ei", "õigemini"),
        relocated_cue_prefixes=("vabandust,",),
    ),
    "fi": LanguageSpec(
        code="fi",
        name="Finnish",
        tier="tier2",
        script="latin",
        fillers=("öö", "ööö", "öhm", "ehm", "hmm"),
        correction_cues=("ei, anteeksi", "hetkinen, ei", "eiku", "tai siis"),
        relocated_cue_prefixes=("anteeksi,",),
    ),
    "el": LanguageSpec(
        code="el",
        name="Greek",
        tier="tier2",
        script="greek",
        fillers=("εε", "εεε", "εμ", "χμ", "χμμ"),
        correction_cues=("όχι, συγγνώμη", "περίμενε, όχι", "βασικά όχι", "δηλαδή"),
        relocated_cue_prefixes=("συγγνώμη,",),
    ),
    "hu": LanguageSpec(
        code="hu",
        name="Hungarian",
        tier="tier2",
        script="latin",
        fillers=("öö", "ööö", "őő", "hmm", "izé"),
        correction_cues=("nem, bocs", "várj, nem", "vagyis nem", "úgy értem"),
        relocated_cue_prefixes=("bocs,", "bocsánat,"),
    ),
    "lv": LanguageSpec(
        code="lv",
        name="Latvian",
        tier="tier2",
        script="latin",
        fillers=("ēē", "ēēē", "ehm", "hmm"),
        correction_cues=("nē, atvaino", "paga, nē", "precīzāk", "tas ir"),
        relocated_cue_prefixes=("atvaino,",),
    ),
    "lt": LanguageSpec(
        code="lt",
        name="Lithuanian",
        tier="tier2",
        script="latin",
        fillers=("eee", "ehm", "aaa", "hmm"),
        correction_cues=("ne, atsiprašau", "palauk, ne", "tiksliau", "tai yra"),
        relocated_cue_prefixes=("atsiprašau,",),
    ),
    "mt": LanguageSpec(
        code="mt",
        name="Maltese",
        tier="tier2",
        script="latin",
        fillers=("emm", "ehh", "ehm", "hmm"),
        correction_cues=("le, skużani", "stenni, le", "jiġifieri le", "aħjar ngħid"),
        relocated_cue_prefixes=("skużani,",),
    ),
    "pl": LanguageSpec(
        code="pl",
        name="Polish",
        tier="tier2",
        script="latin",
        fillers=("yyy", "yy", "eee", "ehm", "hmm"),
        correction_cues=("nie, przepraszam", "czekaj, nie", "to znaczy nie", "właściwie"),
        relocated_cue_prefixes=("przepraszam,",),
    ),
    "pt": LanguageSpec(
        code="pt",
        name="Portuguese",
        tier="tier2",
        script="latin",
        fillers=("hã", "éé", "ehm", "hum", "hmm"),
        correction_cues=("não, desculpa", "espera, não", "quer dizer", "aliás"),
        relocated_cue_prefixes=("desculpa,",),
    ),
    "ru": LanguageSpec(
        code="ru",
        name="Russian",
        tier="tier2",
        script="cyrillic",
        fillers=("ээ", "эээ", "мм", "ммм", "хм"),
        correction_cues=("нет, извини", "подожди, нет", "точнее", "то есть нет"),
        relocated_cue_prefixes=("извини,", "прости,"),
    ),
    "sk": LanguageSpec(
        code="sk",
        name="Slovak",
        tier="tier2",
        script="latin",
        fillers=("eee", "ehm", "eh", "hmm"),
        correction_cues=("nie, prepáč", "počkaj, nie", "vlastne nie", "teda"),
        relocated_cue_prefixes=("prepáč,",),
    ),
    "sl": LanguageSpec(
        code="sl",
        name="Slovenian",
        tier="tier2",
        script="latin",
        fillers=("eee", "emm", "ehm", "hmm"),
        correction_cues=("ne, oprosti", "čakaj, ne", "pravzaprav ne", "mislim"),
        relocated_cue_prefixes=("oprosti,",),
    ),
    "sv": LanguageSpec(
        code="sv",
        name="Swedish",
        tier="tier2",
        script="latin",
        fillers=("eh", "öh", "öhm", "ehm", "hmm"),
        correction_cues=("nej, förlåt", "vänta, nej", "eller nej", "jag menar"),
        relocated_cue_prefixes=("förlåt,",),
    ),
    "uk": LanguageSpec(
        code="uk",
        name="Ukrainian",
        tier="tier2",
        script="cyrillic",
        fillers=("ее", "еее", "ем", "ммм", "хм"),
        correction_cues=("ні, вибач", "почекай, ні", "точніше", "тобто ні"),
        relocated_cue_prefixes=("вибач,", "перепрошую,"),
    ),
}

TIER1_CODES: tuple[str, ...] = tuple(
    code for code, spec in PARAKEET_V3_LANGUAGES.items() if spec.tier == "tier1"
)
TIER2_CODES: tuple[str, ...] = tuple(
    code for code, spec in PARAKEET_V3_LANGUAGES.items() if spec.tier == "tier2"
)
MULTILINGUAL_CODES: tuple[str, ...] = TIER1_CODES + TIER2_CODES
