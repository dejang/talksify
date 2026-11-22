#!/usr/bin/env python3
"""Tier 2 Parakeet v3 multilingual spot-check runner via Talksify's exact production pipeline.

Evaluates representative Tier 2 European languages (``nl``, ``pl``, ``pt``, ``cs``, ``sv``,
``el``, ``uk``) through ``talksify-cli --replay`` using an exported GGUF staged in an isolated
temporary directory (never modifying ``src-tauri/models/qwen-0.6b/model.gguf``).

Every probe uses a domain and disfluency structure distinct from ``BUILTIN_MULTILINGUAL_SEEDS``
so results measure genuine generalization rather than seed-template recall.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qa_filters import LanguageValidator, has_leftover_fillers, has_relocated_cue_prefix
from talksify_replay import ReplayCase, run_talksify_replay

logger = logging.getLogger("talksify.eval_multilingual")


@dataclass(frozen=True)
class Tier2SpotCase:
    lang: str
    category: str
    raw: str
    expected: str
    must_contain: tuple[str, ...]
    must_not_contain: tuple[str, ...]


# Held-out Tier 2 evaluation probes whose domains and structures do not appear in
# BUILTIN_MULTILINGUAL_SEEDS (synthetic_multilingual.py).
TIER2_SPOT_CASES: tuple[Tier2SpotCase, ...] = (
    Tier2SpotCase(
        lang="nl",
        category="question_stutter",
        raw="Heeft de de trein naar Utrecht om kwart over acht nog, eh, stille zitplaatsen vrij?",
        expected="Heeft de trein naar Utrecht om kwart over acht nog stille zitplaatsen vrij?",
        must_contain=("trein", "utrecht", "zitplaatsen"),
        must_not_contain=("de de", "eh", "train", "seats"),
    ),
    Tier2SpotCase(
        lang="pl",
        category="multi_sentence_scoped_retraction",
        raw="Zamówienie z magazynu zostało już spakowane. Kurier ma je dostarczyć do Krakowa, to znaczy do Gdańska.",
        expected="Zamówienie z magazynu zostało już spakowane. Kurier ma je dostarczyć do Gdańska.",
        must_contain=("zamówienie", "spakowane", "gdańska"),
        must_not_contain=("krakowa", "to znaczy", "courier", "warehouse"),
    ),
    Tier2SpotCase(
        lang="pt",
        category="cutoff_false_start",
        raw="Éé, precisamos escolher o rest o restaurante perto do mercado central para o jantar de equipa.",
        expected="Precisamos escolher o restaurante perto do mercado central para o jantar de equipa.",
        must_contain=("restaurante", "mercado central"),
        must_not_contain=("éé", "o rest ", "restaurant"),
    ),
    Tier2SpotCase(
        lang="cs",
        category="imperative_with_filler",
        raw="Zjisti prosím, kolik, ehm, vstupenek zbývá na sobotní koncert v Rudolfinu.",
        expected="Zjisti prosím, kolik vstupenek zbývá na sobotní koncert v Rudolfinu.",
        must_contain=("vstupenek", "koncert", "rudolfinu"),
        must_not_contain=("ehm", "tickets", "concert"),
    ),
    Tier2SpotCase(
        lang="sv",
        category="transport_self_correction",
        raw="Vi bokar biljetter med färjan, vänta, nej, med direkttåget in till centrum.",
        expected="Vi bokar biljetter med direkttåget in till centrum.",
        must_contain=("biljetter", "direkttåget", "centrum"),
        must_not_contain=("färjan", "vänta", "train", "ferry"),
    ),
    Tier2SpotCase(
        lang="el",
        category="vad_pause_break",
        raw="Αν βρέξει αύριο το πρωί θα. Πάρουμε το μετρό μέχρι το λιμάνι.",
        expected="Αν βρέξει αύριο το πρωί, θα πάρουμε το μετρό μέχρι το λιμάνι.",
        must_contain=("μετρό", "λιμάνι", "θα πάρουμε"),
        must_not_contain=("θα. πάρουμε", "metro", "harbor", "port"),
    ),
    Tier2SpotCase(
        lang="uk",
        category="multi_sentence_quantity_retraction",
        raw="Стіл для студії вже привезли зранку. Залишилося замовити чотири крісла, точніше шість крісел для гостей.",
        expected="Стіл для студії вже привезли зранку. Залишилося замовити шість крісел для гостей.",
        must_contain=("стіл", "привезли", "шість"),
        must_not_contain=("чотири", "точніше", "chairs", "studio"),
    ),
)


def _contains_term(text_lower: str, term: str) -> bool:
    t = term.lower()
    if t.strip() == t and all(ch.isalnum() or ch in ("_", "-") for ch in t):
        return re.search(rf"(?<!\w){re.escape(t)}(?!\w)", text_lower) is not None
    return t in text_lower


def run_spot_check(gguf_path: Path, cargo_manifest: Path | None = None) -> dict[str, Any]:
    """Run all Tier 2 spot cases through ``talksify-cli --replay`` with verbose output logging."""
    lang_validator = LanguageValidator(enabled=True)
    replay_cases = [
        ReplayCase(
            file=f"tier2_{case.lang}.wav",
            duration_secs=4.0,
            raw=case.raw,
            expected=case.expected,
        )
        for case in TIER2_SPOT_CASES
    ]
    replay = run_talksify_replay(
        gguf_path,
        replay_cases,
        cargo_manifest=cargo_manifest,
        rust_log="warn,talksify_cli::harness=info,talksify_core::llm_postprocessor=info",
        tmp_prefix="talksify_tier2_eval_",
    )

    results: list[dict[str, Any]] = []
    passed = 0
    for idx, case in enumerate(TIER2_SPOT_CASES):
        out = replay.llm_outputs[idx] if idx < len(replay.llm_outputs) else ""
        lower = out.lower()
        lang_ok, lang_reason = lang_validator.check_language(out, case.lang)
        missing = [m for m in case.must_contain if not _contains_term(lower, m)]
        unwanted = [u for u in case.must_not_contain if _contains_term(lower, u)]
        filler_left = has_leftover_fillers(out, case.lang)
        relocated = has_relocated_cue_prefix(case.raw, out, case.lang)
        ok = bool(out) and lang_ok and not missing and not unwanted and not filler_left and not relocated
        if ok:
            passed += 1
        results.append(
            {
                "lang": case.lang,
                "category": case.category,
                "passed": ok,
                "raw": case.raw,
                "expected": case.expected,
                "output": out,
                "lang_check": lang_reason,
                "missing": missing,
                "unwanted": unwanted,
                "filler_left": filler_left,
                "relocated_cue": relocated,
            }
        )
    return {
        "gguf": str(gguf_path.resolve()),
        "passed": passed,
        "total": len(TIER2_SPOT_CASES),
        "cases": results,
        "harness_summary": replay.stdout.strip(),
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run Tier 2 multilingual spot-check on a GGUF.")
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args(argv)

    report = run_spot_check(args.gguf)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
