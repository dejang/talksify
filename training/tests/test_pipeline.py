"""Unit and integration tests for the Talksify multilingual fine-tuning pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from conventions import apply_talksify_conventions
from eval_multilingual import TIER2_SPOT_CASES, _contains_term
from export_gguf import LLAMA_CPP_COMMIT, sha256_file
from prepare_data import (
    DEFAULT_SOURCE_MAX_SHARE,
    balance_english_and_multilingual,
    build_arg_parser as build_prepare_arg_parser,
    filter_and_deduplicate,
    normalize_german_disfluency_transcript,
    repair_existing_dataset,
    stratified_split,
    write_jsonl_with_messages,
)
from prompts import (
    LLM_POSTPROCESSOR_RS,
    MULTILINGUAL_CODES,
    PARAKEET_V3_LANGUAGES,
    TIER1_CODES,
    TIER2_CODES,
    build_messages,
    extract_clean_prompt_from_rust,
    format_user_message,
    load_clean_prompt,
)
from qa_filters import (
    PASSTHROUGH_CATEGORIES,
    LanguageValidator,
    analyze_token_alignment,
    drops_opening_clause,
    has_assistant_chatter,
    has_clause_duplication,
    has_introduced_list_or_newlines,
    has_leftover_fillers,
    has_relocated_cue_prefix,
    has_unwanted_paragraph_breaks,
    is_empty_sentinel,
    is_retraction_category,
    validate_pair,
)
from synthetic_en import generate_english_gap_fill
from synthetic_multilingual import (
    BUILTIN_MULTILINGUAL_SEEDS,
    _build_source_buckets,
    allocate_language_quotas,
    build_teacher_adaptation_prompt,
    parse_teacher_json,
)
from talksify_replay import parse_llm_outputs
from train_lora import (
    CompletionsCollator,
    load_jsonl,
    validate_dataset_rows,
    verify_completions_only_masking,
)


def test_clean_prompt_matches_rust_source_and_fails_loudly_when_missing(tmp_path: Path) -> None:
    assert LLM_POSTPROCESSOR_RS.is_file()
    extracted = extract_clean_prompt_from_rust(LLM_POSTPROCESSOR_RS)
    assert load_clean_prompt() == extracted
    assert "<transcript>...</transcript>" in extracted
    assert 'Use a colon (":")' in extracted
    assert "written text.Each user message" in extracted
    with pytest.raises(FileNotFoundError):
        load_clean_prompt(tmp_path / "missing_llm_postprocessor.rs")


def test_user_message_matches_llm_backend_framing() -> None:
    raw = "Um, hello world."
    assert format_user_message(raw) == "<transcript>\nUm, hello world.\n</transcript>"
    msgs = build_messages(raw, "Hello world.")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[0]["content"] == load_clean_prompt()
    assert msgs[1]["content"] == "<transcript>\nUm, hello world.\n</transcript>"
    assert msgs[2]["content"] == "Hello world."


def test_parakeet_v3_25_language_coverage_and_quotas() -> None:
    assert len(PARAKEET_V3_LANGUAGES) == 25
    assert set(TIER1_CODES) == {"de", "fr", "es", "it", "ro"}
    assert len(TIER2_CODES) == 19
    assert len(MULTILINGUAL_CODES) == 24

    quotas = allocate_language_quotas(10_000, tier1_ratio=0.60)
    assert sum(quotas.values()) == 10_000
    assert sum(quotas[c] for c in TIER1_CODES) == 6_000
    assert sum(quotas[c] for c in TIER2_CODES) == 4_000
    assert all(quotas[c] > 0 for c in MULTILINGUAL_CODES)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("forty two files", "42 files"),
        ("twenty-five", "25"),
        ("one hundred and twenty", "120"),
        ("ten percent", "10%"),
        ("one percent", "1%"),
        ("twenty five per cent", "25%"),
        ("ends in dot wav", "ends in .wav"),
        ("open main dot rs", "open main.rs"),
        ("a dot text file", "a .txt file"),
        ("Dot JSON", ".json"),
        # Ambiguous / out-of-scope forms must remain unchanged (mirroring conventions.rs).
        ("no one came", "no one came"),
        ("the first step", "the first step"),
        ("about two and a half hours", "about two and a half hours"),
        ("three point five", "three point five"),
        ("a hundred times", "a hundred times"),
        ("the twenty-second", "the twenty-second"),
        ("wait thirty seconds", "wait 30 seconds"),
        ("a red dot on the map", "a red dot on the map"),
    ],
)
def test_conventions_parity_with_rust(raw: str, expected: str) -> None:
    assert apply_talksify_conventions(raw) == expected


def test_quality_filters_catch_all_f406_failure_modes() -> None:
    assert has_assistant_chatter("Here is the cleaned transcript: Hello.")
    assert has_assistant_chatter("<transcript>\nHello.\n</transcript>")
    assert has_assistant_chatter("<think></think>Hello.")
    assert not has_assistant_chatter("Hello, this is a normal sentence.")

    assert has_unwanted_paragraph_breaks("One sentence. Two sentence.", "One sentence.\n\nTwo sentence.")
    assert not has_unwanted_paragraph_breaks("One sentence. Two sentence.", "One sentence. Two sentence.")

    dup = (
        "We should deploy the new worker before Friday afternoon "
        "we should deploy the new worker before Friday afternoon."
    )
    assert has_clause_duplication(dup)
    assert not has_clause_duplication("We should deploy the new worker before Friday afternoon.")

    # Fillers caught, while German real words "um" / "er" are preserved.
    assert has_leftover_fillers("Äh, morgen treffen wir uns.", "de")
    assert not has_leftover_fillers("Morgen treffen wir uns um 10 Uhr im Büro.", "de")
    assert not has_leftover_fillers("Er kommt später zum Treffen.", "de")
    assert has_leftover_fillers("Ăă, mergem mâine la birou.", "ro")
    assert not has_leftover_fillers("Mergem mâine la birou.", "ro")

    # Relocated self-correction cue prefix (e.g. Romanian "nu, scuze," -> "Scuze, ...").
    assert has_relocated_cue_prefix(
        "Ăă, mergem marți, nu, scuze, joi la birou.",
        "Scuze, mergem joi la birou.",
        "ro",
    )
    assert not has_relocated_cue_prefix(
        "Ăă, mergem marți, nu, scuze, joi la birou.",
        "Mergem joi la birou.",
        "ro",
    )

    # Catastrophic over-deletion rejected on non-retraction row.
    ok, reason = validate_pair(
        "I showed the prototype to Elena and she looked at the screen saying no that is definitely not it so we revised the layout.",
        "We revised.",
        "en",
        is_retraction=False,
    )
    assert not ok and "over-deletion" in reason


def test_language_validator_catches_english_translation_drift() -> None:
    validator = LanguageValidator(enabled=True)
    # Genuine native sentences pass.
    for lang, text in [
        ("de", "Wir prüfen das neue Mikrofonkabel morgen früh direkt im Aufnahmeraum."),
        ("fr", "Nous devons vérifier la configuration du serveur avant de publier la mise à jour."),
        ("es", "Vamos a guardar las capturas de audio en la carpeta principal del proyecto."),
        ("it", "Abbiamo ordinato sei monitor nuovi per il laboratorio di registrazione audio."),
        ("ro", "Am rezervat sala de conferințe la etajul patru pentru ședința echipei."),
        ("el", "Θέλω να ελέγξω την ακουστική του δωματίου πριν αρχίσουμε την ηχογράφηση."),
        ("uk", "Скільки вільного місця залишилося на зовнішньому диску для резервних копій?"),
    ]:
        ok, reason = validator.check_language(text, lang)
        assert ok, f"Expected {lang} to pass, got: {reason}"

    # English translations of non-English targets MUST fail!
    english_drift = "We need to move the product launch from Tuesday to Thursday afternoon."
    for lang in ("de", "fr", "es", "it", "ro", "nl", "pl", "mt", "el", "uk"):
        ok, reason = validator.check_language(english_drift, lang)
        assert not ok, f"Expected English drift for {lang} to be rejected, got: {reason}"


def test_english_gap_fill_generator_quality_and_diversity() -> None:
    rows = generate_english_gap_fill(seed=407, target_count=300)
    assert len(rows) >= 250
    cats = {r.category for r in rows}
    assert {
        "scoped_self_correction",
        "quoted_negation_trap",
        "cutoff_false_start",
        "vad_pause_break",
        "thinking_aloud",
        "clean_passthrough",
    }.issubset(cats)

    validator = LanguageValidator(enabled=True)
    for row in rows:
        is_pass = row.category in PASSTHROUGH_CATEGORIES
        if is_pass:
            assert row.raw == row.clean
        ok, reason = validate_pair(
            row.raw,
            row.clean,
            "en",
            is_retraction=row.is_retraction,
            is_passthrough=is_pass,
            category=row.category,
            lang_validator=validator,
        )
        assert ok, f"Gap-fill row failed QA ({row.category}): {reason} | raw={row.raw!r} clean={row.clean!r}"


def test_builtin_multilingual_seeds_cover_all_24_languages_and_pass_qa() -> None:
    langs = {r.lang for r in BUILTIN_MULTILINGUAL_SEEDS}
    assert langs == set(MULTILINGUAL_CODES)
    validator = LanguageValidator(enabled=True)
    for row in BUILTIN_MULTILINGUAL_SEEDS:
        is_pass = row.category in PASSTHROUGH_CATEGORIES
        ok, reason = validate_pair(
            row.raw,
            row.clean,
            row.lang,
            is_retraction=row.is_retraction,
            is_passthrough=is_pass,
            category=row.category,
            lang_validator=validator,
        )
        assert ok, f"Seed {row.lang} failed QA: {reason} | clean={row.clean!r}"


def test_parse_teacher_json_handles_thinking_and_markdown() -> None:
    raw_out = '<think>plan</think>\n```json\n{"raw": "Äh, hallo Welt.", "clean": "Hallo Welt."}\n```'
    assert parse_teacher_json(raw_out) == ("Äh, hallo Welt.", "Hallo Welt.")
    assert parse_teacher_json("not json") is None
    assert parse_teacher_json('{"raw": "", "clean": "ok"}') is None


def test_filtering_balancing_splitting_and_collation(tmp_path: Path) -> None:
    validator = LanguageValidator(enabled=True)
    en_rows = [
        {
            "raw": r.raw,
            "clean": r.clean,
            "lang": "en",
            "tier": "en",
            "source": r.source,
            "category": r.category,
            "is_retraction": r.is_retraction,
        }
        for r in generate_english_gap_fill(seed=1, target_count=100)
    ]
    multi_rows = [
        {
            "raw": r.raw,
            "clean": r.clean,
            "lang": r.lang,
            "tier": r.tier,
            "source": r.source,
            "category": r.category,
            "is_retraction": r.is_retraction,
        }
        for r in BUILTIN_MULTILINGUAL_SEEDS
    ]
    en_clean, _ = filter_and_deduplicate(en_rows, lang_validator=validator)
    multi_clean, _ = filter_and_deduplicate(multi_rows, lang_validator=validator)
    sel_en, sel_multi = balance_english_and_multilingual(
        en_clean, multi_clean, target_en_ratio=0.67, max_total_en=200
    )
    train, val = stratified_split(sel_en + sel_multi, val_ratio=0.1, seed=407)
    assert len(train) > len(val) > 0

    prompt = load_clean_prompt()
    out_file = tmp_path / "train.jsonl"
    write_jsonl_with_messages(train, out_file, prompt)
    loaded = load_jsonl(out_file)
    validate_dataset_rows(loaded, prompt)
    assert len(loaded) == len(train)

    # Collator pads to multiple of 8 and masks padding labels to -100.
    collator = CompletionsCollator(pad_token_id=0)
    batch = collator(
        [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, -100, 3], "tier": "en"},
            {"input_ids": [4, 5, 6, 7, 8], "attention_mask": [1, 1, 1, 1, 1], "labels": [-100, 5, 6, 7, 8], "tier": "tier1"},
        ]
    )
    assert batch["input_ids"].shape == (2, 8)
    assert torch.equal(batch["labels"][0, 3:], torch.full((5,), -100, dtype=torch.long))
    assert batch["tiers"] == ["en", "tier1"]


class _Qwen35MockTokenizer:
    """Offline byte-level mock tokenizer implementing Qwen3.5's exact enable_thinking=False template."""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        enable_thinking: bool = False,
    ) -> str:
        assert not tokenize
        assert not enable_thinking
        parts: list[str] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "assistant":
                parts.append(f"<|im_start|>assistant\n<think>\n\n</think>\n\n{content}<|im_end|>\n")
            else:
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        return "".join(parts)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(ids).decode("utf-8")


def test_completions_only_loss_masking_invariants_and_sha256(tmp_path: Path) -> None:
    tok = _Qwen35MockTokenizer()
    prompt = load_clean_prompt()
    rows = [
        {"raw": "Um, hello world.", "clean": "Hello world.", "lang": "en", "tier": "en"},
        {
            "raw": "Ăă, mergem marți, nu, scuze, joi.",
            "clean": "Mergem joi.",
            "lang": "ro",
            "tier": "tier1",
        },
    ]
    report = verify_completions_only_masking(tok, rows, prompt, max_seq_len=4096)
    assert report["checked_rows"] == 2

    sample = tmp_path / "weights.bin"
    sample.write_bytes(b"talksify-gguf-test")
    assert len(sha256_file(sample)) == 64


def test_german_disfluency_normalization() -> None:
    raw = (
        "[UH] ich bin mir nicht sicher, ob wir [UM] nächstes Mal "
        "[lipsmack], bru* [UH] Bruder, [throatclearing] v* verschieben sollten. [UH] Genau."
    )
    normalized = normalize_german_disfluency_transcript(raw)
    assert normalized == (
        "Äh ich bin mir nicht sicher, ob wir ähm nächstes Mal, bru äh Bruder, "
        "v verschieben sollten. Äh Genau."
    )
    assert "[" not in normalized and "*" not in normalized


def test_pinned_llama_cpp_replay_parser_and_tier2_probe_independence() -> None:
    assert len(LLAMA_CPP_COMMIT) == 40
    assert all(ch in "0123456789abcdef" for ch in LLAMA_CPP_COMMIT)

    # No --strict-licenses flag in prepare_data CLI (all 5 datasets included unconditionally).
    parser = build_prepare_arg_parser()
    option_strings = {opt for action in parser._actions for opt in action.option_strings}
    assert "--strict-licenses" not in option_strings

    # Shared replay log parser extracts quoted JSON outputs in order.
    sample_stderr = (
        '2026-11-21 INFO talksify_core::llm_postprocessor: [llm] output (0.4s): "Hello world."\n'
        '2026-11-21 INFO talksify_core::llm_postprocessor: [llm] output (0.5s): "A \\"quoted\\" note."\n'
    )
    assert parse_llm_outputs(sample_stderr) == ["Hello world.", 'A "quoted" note.']

    # Tier 2 spot-check probes must have distinct categories and texts from BUILTIN_MULTILINGUAL_SEEDS,
    # and word-boundary matching must avoid substring false positives (e.g. "restaurant" in "restaurante").
    assert not _contains_term("escolher o restaurante perto", "restaurant")
    assert _contains_term("nog, eh, stille zitplaatsen", "eh")
    seed_by_lang = {r.lang: r for r in BUILTIN_MULTILINGUAL_SEEDS}
    validator = LanguageValidator(enabled=True)
    for case in TIER2_SPOT_CASES:
        seed = seed_by_lang[case.lang]
        assert case.raw.lower() != seed.raw.lower()
        assert case.category != seed.category
        ok, reason = validator.check_language(case.expected, case.lang)
        assert ok, f"Tier 2 probe expected text failed LID for {case.lang}: {reason}"


def test_round2_dataset_qa_guards_empty_passthrough_subsequence_lists_and_caps(
    tmp_path: Path,
) -> None:
    # 1. EMPTY sentinel rejection.
    assert is_empty_sentinel("EMPTY")
    assert is_empty_sentinel("EMPTY.")
    assert is_empty_sentinel("LEER", "empty")
    assert not is_empty_sentinel("The room is empty.", "prose")
    ok, reason = validate_pair("ăăă uhh.", "EMPTY", "ro", category="empty")
    assert not ok and "empty sentinel" in reason

    # 2. Passthrough dedicated teacher prompt & strict raw == clean invariant.
    pass_msgs = build_teacher_adaptation_prompt(
        "How many licenses do we need?",
        "How many licenses do we need?",
        "it",
        "clean_passthrough",
    )
    assert "Do NOT inject ANY hesitation fillers" in pass_msgs[1]["content"]
    assert '"raw" == "clean"' in pass_msgs[1]["content"]

    ok, reason = validate_pair(
        "Ehm, puoi controllare il file?",
        "Puoi controllare il file?",
        "it",
        category="clean_passthrough",
    )
    assert not ok and "passthrough" in reason
    ok, _ = validate_pair(
        "Puoi controllare il file?",
        "Puoi controllare il file?",
        "it",
        category="clean_passthrough",
    )
    assert ok

    # 3. Near-subsequence catches SQuAD rephrasing/reordering while accepting clean edits.
    ok, _ = analyze_token_alignment(
        "What key term is typically defined in connection I mean is distinct from constitutional impasse?",
        "Constitutional impasse is distinct from what key term?",
        "en",
    ).is_valid()
    assert not ok
    ok, _ = analyze_token_alignment(
        "What institution no um publication printed that forty two people attended?",
        "What publication printed that 42 people attended?",
        "en",
    ).is_valid()
    assert ok

    # 4. List restructuring, shopping_lists, and dropped spoken openers rejected.
    assert has_introduced_list_or_newlines(
        "camping trip tent sleeping bags and cooler",
        "Camping packing list:\n- Tent\n- Sleeping bags\n- Cooler",
    )
    assert drops_opening_clause(
        "streaming rights clause here we go um the Licensor grants to the Licensee the exclusive right",
        "The Licensor grants to the Licensee the exclusive right",
        "en",
    )
    ok, reason = validate_pair(
        "streaming rights clause here we go um the Licensor grants to the Licensee the exclusive right to stream and distribute the licensed content across North America for 5 years",
        "The Licensor grants to the Licensee the exclusive right to stream and distribute the licensed content across North America for 5 years",
        "en",
        is_retraction=False,
        category="legal_contract",
    )
    assert not ok and "opening clause" in reason

    # 5. Source bucket builder drops empty/shopping_lists and guards tiny-category cloning.
    pool = [
        {"raw": f"Sample sentence number {i} for prose.", "clean": f"Sample sentence number {i} for prose.", "category": "prose"}
        for i in range(60)
    ] + [
        {"raw": "uhh", "clean": "EMPTY", "category": "empty"},
        {"raw": "milk eggs", "clean": "Milk, eggs.", "category": "shopping_lists"},
        {"raw": "tiny one", "clean": "Tiny one.", "category": "tiny_rare_cat"},
    ]
    import random

    by_cat, _ = _build_source_buckets(pool, random.Random(407))
    assert "empty" not in by_cat
    assert "shopping_lists" not in by_cat
    assert "tiny_rare_cat" not in by_cat
    assert "prose" in by_cat

    # 6. disfl_qa share cap in balance_english_and_multilingual.
    en_rows = [
        {
            "raw": f"Prose dictation line {i}.",
            "clean": f"Prose dictation line {i}.",
            "lang": "en",
            "tier": "en",
            "source": "category271/whispernow-cleanup",
            "category": "prose",
            "is_retraction": False,
        }
        for i in range(400)
    ] + [
        {
            "raw": f"What year um what decade did event {i} happen?",
            "clean": f"What decade did event {i} happen?",
            "lang": "en",
            "tier": "en",
            "source": "google-research-datasets/disfl_qa",
            "category": "disfl_qa_question",
            "is_retraction": False,
        }
        for i in range(400)
    ]
    sel_en, _ = balance_english_and_multilingual(en_rows, [], max_total_en=500)
    dq_count = sum(1 for r in sel_en if r["source"] == "google-research-datasets/disfl_qa")
    assert dq_count / len(sel_en) <= DEFAULT_SOURCE_MAX_SHARE["google-research-datasets/disfl_qa"] + 0.01

    # 7. Local repair regenerates synthetic_en_gapfill, drops stale teacher thinking_aloud, and filters locally.
    prompt = load_clean_prompt()
    dirty_rows = [
        {"raw": "uhh.", "clean": "EMPTY", "lang": "en", "tier": "en", "source": "category271/whispernow-cleanup", "category": "empty"},
        {"raw": "Ehm, ciao mondo.", "clean": "Ciao mondo.", "lang": "it", "tier": "tier1", "source": "teacher:mock", "category": "clean_passthrough"},
        {"raw": "Ähm, eigentlich denke ich nicht, dass wir die Freigabe blockieren sollen. Ich glaube, wir sollten den Kernfix jetzt shippen.", "clean": "Ich glaube, wir sollten den Kernfix jetzt shippen.", "lang": "de", "tier": "tier1", "source": "teacher:mock", "category": "thinking_aloud"},
        {"raw": "Actually I don't think we should block the release, I think we should ship the core fix now.", "clean": "Actually, I think we should ship the core fix now instead of blocking the release.", "lang": "en", "tier": "en", "source": "synthetic_en_gapfill", "category": "thinking_aloud"},
        {"raw": "What key term is typically defined in connection I mean is distinct from constitutional impasse?", "clean": "Constitutional impasse is distinct from what key term?", "lang": "en", "tier": "en", "source": "google-research-datasets/disfl_qa", "category": "disfl_qa_question"},
        {"raw": "camping trip tent and cooler", "clean": "Camping:\n- Tent\n- Cooler", "lang": "en", "tier": "en", "source": "shantanugoel/aawaaz-transcript-cleanup-dataset", "category": "shopping_lists"},
        {"raw": "Before the sprint starts, make sure everyone has the checklist.", "clean": "Before the sprint starts, make sure everyone has the checklist.", "lang": "en", "tier": "en", "source": "category271/whispernow-cleanup", "category": "prose"},
    ]
    write_jsonl_with_messages(dirty_rows, tmp_path / "train.jsonl", prompt)
    write_jsonl_with_messages([], tmp_path / "val.jsonl", prompt)
    manifest = repair_existing_dataset(
        tmp_path,
        system_prompt=prompt,
        lang_validator=LanguageValidator(enabled=True),
        en_gapfill_rows=40,
        regen_passthroughs=0,
        skip_teacher=True,
    )
    assert manifest["by_category"].get("empty", 0) == 0
    assert manifest["by_category"].get("shopping_lists", 0) == 0
    repaired = load_jsonl(tmp_path / "train.jsonl") + load_jsonl(tmp_path / "val.jsonl")
    assert all(r["clean"].strip().upper() != "EMPTY" for r in repaired)
    assert all(r["raw"] == r["clean"] for r in repaired if r["category"] in PASSTHROUGH_CATEGORIES)
    assert not any("instead of blocking the release" in r["clean"].lower() for r in repaired)
    assert not any(r["source"].startswith("teacher:") and r["category"] == "thinking_aloud" for r in repaired)


def test_round3_strict_alignment_retraction_parity_and_template_fixes() -> None:
    # Single is_retraction_category rule handles case sensitivity (mixed_L2) and source overrides.
    assert is_retraction_category("mixed_L2", "Akash-Sakala/transcript-formatter-curriculum")
    assert is_retraction_category("self_correction", "category271/whispernow-cleanup")
    assert is_retraction_category("self_corrections_heavy", "shantanugoel/aawaaz-transcript-cleanup-dataset")
    assert not is_retraction_category("legal_contract", "shantanugoel/aawaaz-transcript-cleanup-dataset")
    assert not is_retraction_category("disfl_qa_question", "google-research-datasets/disfl_qa")

    # Regenerated English gap-fill keeps both clauses of "block the release" and has 0 old reordered templates.
    rows = generate_english_gap_fill(seed=407, target_count=2800)
    for bad in ("instead of blocking the release", "less intrusive than a modal", "with serde instead of writing"):
        assert not any(bad in r.clean.lower() for r in rows)
    release_rows = [r for r in rows if "block the release" in r.raw.lower()]
    assert release_rows and all("block the release" in r.clean.lower() for r in release_rows)

    # Rejects invented headings, inline numbered lists, added/paraphrased words, moved clauses, and over-scoped deletes.
    assert not validate_pair(
        "espera não quer dizer que o período de resolução de conflitos seja igual então qualquer disputa será resolvida por arbitragem",
        "Seção 21 — Resolução de Conflitos. Qualquer disputa será resolvida por arbitragem.",
        "pt", category="legal_contract",
    )[0]
    assert not validate_pair(
        "Les commandes pour l'outil de montage. Couper les 30 secondes. Ajouter des chapitres. Exporter le fichier.",
        "Les commandes pour l'outil de montage : 1. Couper les 30 secondes. 2. Ajouter des chapitres. 3. Exporter le fichier.",
        "fr", category="list",
    )[0]
    assert not validate_pair(
        "su un weekend buono posso prendere sei cento no sette cinquanta dopo le spese",
        "Su un weekend buono posso guadagnare circa 750 dollari dopo le spese.",
        "it", is_retraction=True, category="self_corrections_heavy",
    )[0]
    assert not validate_pair(
        "Es kommen immer verschiedene Konkurrenten-Technologien vor und das ist genau die Daten die wir für das Projekt brauchen",
        "Es kommen immer verschiedene konkurrierende Technologien vor die genau die Daten sind die wir für das Projekt brauchen.",
        "de", category="sent_start_cap",
    )[0]
    assert not validate_pair(
        "Section eleven point one Arbitration Panel Selection the parties shall select the arbitration panel as follows colon each party shall no scratch that within fifteen business days each party shall designate one arbitrator",
        "Section 11.1, Arbitration Panel Selection. Within 15 business days each party shall designate one arbitrator.",
        "en", category="legal_contract",
    )[0]

    # Accepts legitimate homophone fixes, diacritic normalization, and Slavic stem inflection.
    assert validate_pair("The proposal past the review committee.", "The proposal passed the review committee.", "en", category="homophone_fix")[0]
    assert validate_pair("Rezultatul sectiunii congelate pe nodul limfatic.", "Rezultatul secțiunii congelate pe nodul limfatic.", "ro", category="medical_clinical")[0]
    assert analyze_token_alignment("Przypisałem bilety do zespołu analizy, albo raczej, do epika.", "Przypisałem epik do zespołu analizy.", "pl").is_valid(is_retraction=True, category="scoped_self_correction")[0]
