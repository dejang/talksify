"""25-language synthetic adaptation pipeline for Talksify Qwen3.5-0.8B fine-tuning.

Covers all 24 non-English European languages supported by Parakeet TDT 0.6B v3:
- Tier 1 (~60% weight): de, fr, es, it, ro
- Tier 2 (~40% weight): bg, hr, cs, da, nl, et, fi, el, hu, lv, lt, mt, pl, pt, ru, sk, sl, sv, uk

On GPU (RTX PRO 6000 96 GB), adapts a stratified subset of English ``(raw, clean)`` pairs via a
local open-weight multilingual teacher model (default ``Qwen/Qwen3-4B-Instruct-2507-FP8``,
configurable via ``--teacher-model``), using a dedicated no-disfluency instruction for passthrough
categories and strict automated QA (language ID, filler-free, near-subsequence, and restraint).
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from multilingual_seeds import BUILTIN_MULTILINGUAL_SEEDS, MultilingualRow
from prompts import MULTILINGUAL_CODES, PARAKEET_V3_LANGUAGES, TIER1_CODES, TIER2_CODES
from qa_filters import (
    PASSTHROUGH_CATEGORIES,
    LanguageValidator,
    is_empty_sentinel,
    is_retraction_category,
    normalize_whitespace,
    validate_pair,
)

logger = logging.getLogger(__name__)

DEFAULT_TEACHER_MODEL = "Qwen/Qwen3-4B-Instruct-2507-FP8"
_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")


def allocate_language_quotas(
    total_multilingual: int,
    tier1_ratio: float = 0.60,
) -> dict[str, int]:
    """Allocate exact per-language targets across Tier 1 (~60%) and Tier 2 (~40%)."""
    if total_multilingual <= 0:
        return {code: 0 for code in MULTILINGUAL_CODES}

    tier1_total = round(total_multilingual * tier1_ratio)
    tier2_total = total_multilingual - tier1_total

    quotas: dict[str, int] = {}
    t1_base, t1_rem = divmod(tier1_total, len(TIER1_CODES))
    for i, code in enumerate(TIER1_CODES):
        quotas[code] = t1_base + (1 if i < t1_rem else 0)

    t2_base, t2_rem = divmod(tier2_total, len(TIER2_CODES))
    for i, code in enumerate(TIER2_CODES):
        quotas[code] = t2_base + (1 if i < t2_rem else 0)

    return quotas


def build_teacher_adaptation_prompt(
    en_raw: str,
    en_clean: str,
    target_lang: str,
    category: str,
) -> list[dict[str, str]]:
    """Build chat messages asking a local multilingual teacher to idiomatically adapt a pair.

    Passthrough categories (``clean_passthrough``, ``passthrough``, ``no_edit``) receive a
    dedicated instruction requiring clean idiomatic translation with zero disfluencies injected
    and ``raw == clean`` character-for-character.
    """
    spec = PARAKEET_V3_LANGUAGES[target_lang]
    system = (
        "You create realistic speech-to-text dictation cleanup training pairs. "
        "Always reply with a single valid JSON object containing exactly two string keys: "
        '"raw" and "clean". No markdown fences, no commentary, no extra keys.'
    )

    if category.strip().lower() in PASSTHROUGH_CATEGORIES:
        user = (
            f"Translate the following clean English dictation into natural, well-punctuated "
            f"{spec.name} ({spec.code}, {spec.script} script) as a CLEAN PASSTHROUGH example.\n"
            f"Category: {category}\n"
            f"Rules:\n"
            f'1. Both "raw" and "clean" MUST be written strictly in {spec.name} using {spec.script} '
            f"script (zero English words).\n"
            f'2. Do NOT inject ANY hesitation fillers, stutters, false starts, pause breaks, or '
            f'self-corrections. Both "raw" and "clean" MUST be completely clean and IDENTICAL '
            f'character-for-character ("raw" == "clean").\n'
            f"3. Preserve the exact sentence type, facts, and meaning verbatim: keep questions as "
            f"questions, imperatives as imperatives, and negations intact. Never drop clauses, "
            f"never answer questions, and never add line breaks or bullet lists.\n\n"
            f"Source English text:\n{en_clean}"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    fillers_str = ", ".join(spec.fillers[:5])
    cues_str = ", ".join(spec.correction_cues[:4])
    reloc_hint = spec.relocated_cue_prefixes[0] if spec.relocated_cue_prefixes else ""
    user = (
        f"Adapt the following English speech dictation cleanup pair into natural, idiomatic "
        f"{spec.name} ({spec.code}, {spec.script} script).\n"
        f"Category: {category}\n"
        f"Rules:\n"
        f'1. Both "raw" and "clean" MUST be written strictly in {spec.name} using {spec.script} '
        f"script (zero English words or English translation).\n"
        f'2. In "raw", use natural spoken {spec.name} disfluencies matching the source pattern: '
        f"local hesitation fillers ({fillers_str}), stutters/cut-off word false starts, mid-sentence "
        f"pause breaks ('. Capital'), or local self-correction cues ({cues_str}).\n"
        f'3. In "clean", produce smooth written {spec.name} prose with ALL hesitation fillers and '
        f"self-correction cues completely removed. Keep the exact word order and facts of the kept "
        f"clauses without rephrasing. For any self-correction, discard ONLY the old retracted phrase "
        f"and keep the new replacement (never relocate apology cues like '{reloc_hint}' to the start).\n"
        f"4. Keep the speaker's real facts, opening clauses, questions, or imperatives intact. "
        f"Never answer questions, never restructure prose into bullet lists, and never add line breaks.\n\n"
        f"Source English raw:\n{en_raw}\n\n"
        f"Source English clean:\n{en_clean}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_teacher_json(text: str) -> tuple[str, str] | None:
    """Extract ``(raw, clean)`` strings from teacher model JSON output."""
    cleaned = text.strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
    match = _JSON_OBJ_RE.search(cleaned)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    raw = payload.get("raw")
    clean = payload.get("clean")
    if not isinstance(raw, str) or not isinstance(clean, str):
        return None
    raw_s = normalize_whitespace(raw)
    clean_s = normalize_whitespace(clean)
    if not raw_s or not clean_s:
        return None
    return raw_s, clean_s


def _build_source_buckets(
    english_pool: Sequence[dict[str, Any]],
    rng: random.Random,
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    priority_cats = {
        "scoped_self_correction",
        "quoted_negation_trap",
        "cutoff_false_start",
        "vad_pause_break",
        "thinking_aloud",
        "clean_passthrough",
        "self_corrections_heavy",
        "false_start_collapse",
        "backtrack_phrase",
        "stutter_removal",
        "filler_removal",
        "passthrough",
    }
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for item in english_pool:
        cat = str(item.get("category", "general"))
        if is_empty_sentinel(str(item.get("clean", "")), cat) or cat.lower() == "shopping_lists":
            continue
        if len(item["raw"]) > 550 or len(item["clean"]) > 550:
            continue
        by_cat.setdefault(cat, []).append(item)

    # Guard against tiny-category cloning (e.g., 3 rows cycled 60x across all languages).
    min_bucket = 15 if len(english_pool) >= 50 else 1
    by_cat = {cat: rows for cat, rows in by_cat.items() if len(rows) >= min_bucket}
    if not by_cat:
        for item in english_pool:
            cat = str(item.get("category", "general"))
            if not is_empty_sentinel(str(item.get("clean", "")), cat):
                by_cat.setdefault(cat, []).append(item)

    weighted_cats: list[str] = []
    for cat in sorted(by_cat):
        rng.shuffle(by_cat[cat])
        weight = 3 if cat in priority_cats else 1
        weighted_cats.extend([cat] * weight)
    return by_cat, weighted_cats


def _plan_round_tasks(
    rows_by_lang: dict[str, list[MultilingualRow]],
    quotas: dict[str, int],
    by_cat: dict[str, list[dict[str, Any]]],
    weighted_cats: list[str],
    cat_cursors: dict[str, int],
    used_src_by_lang: dict[str, set[str]],
    *,
    oversample: float,
    batch_size: int,
    rng: random.Random,
) -> list[tuple[str, dict[str, Any]]]:
    tasks: list[tuple[str, dict[str, Any]]] = []
    for lang in MULTILINGUAL_CODES:
        need = max(0, quotas[lang] - len(rows_by_lang[lang]))
        if need == 0:
            continue
        to_request = max(need + 6, int(need * oversample) + 6)
        used = used_src_by_lang[lang]
        attempts = 0
        added = 0
        max_attempts = to_request * 4
        while added < to_request and attempts < max_attempts:
            cat = weighted_cats[attempts % len(weighted_cats)]
            attempts += 1
            pool = by_cat[cat]
            c_idx = cat_cursors[cat]
            cat_cursors[cat] = c_idx + 1
            src = pool[c_idx % len(pool)]
            src_key = str(src["raw"]).strip().lower()
            if src_key in used and len(used) < sum(len(p) for p in by_cat.values()):
                continue
            used.add(src_key)
            tasks.append((lang, src))
            added += 1

    rng.shuffle(tasks)
    mega_size = max(batch_size * 12, 128)
    bucketed: list[tuple[str, dict[str, Any]]] = []
    for m_start in range(0, len(tasks), mega_size):
        chunk = tasks[m_start : m_start + mega_size]
        chunk.sort(key=lambda pair: len(pair[1]["raw"]) + len(pair[1]["clean"]))
        bucketed.extend(chunk)
    return bucketed


class MultilingualTeacher:
    """Owns the local CUDA teacher model lifecycle and validated batch adaptation."""

    def __init__(
        self,
        teacher_model: str = DEFAULT_TEACHER_MODEL,
        *,
        batch_size: int = 192,
        max_new_tokens: int = 384,
        lang_validator: LanguageValidator | None = None,
    ) -> None:
        self.teacher_model = teacher_model
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.validator = lang_validator or LanguageValidator(enabled=True)
        self._tokenizer: Any = None
        self._model: Any = None

    def __enter__(self) -> MultilingualTeacher:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Local multilingual teacher generation requires a CUDA GPU (torch.cuda.is_available() "
                "is False). Run on the RTX PRO 6000 queue GPU or use --skip-teacher / --smoke-test."
            )
        logger.info("Loading local multilingual teacher %s on CUDA (batch_size=%d)...", self.teacher_model, self.batch_size)
        torch.set_float32_matmul_precision("high")
        self._tokenizer = AutoTokenizer.from_pretrained(self.teacher_model, padding_side="left")
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        self._model = AutoModelForCausalLM.from_pretrained(
            self.teacher_model, dtype="auto", attn_implementation="sdpa", device_map="auto"
        )
        self._model.eval()
        return self

    def __exit__(self, *_exc: object) -> None:
        import torch

        self._model = None
        self._tokenizer = None
        torch.cuda.empty_cache()

    def generate_batch(
        self,
        batch: list[tuple[str, dict[str, Any]]],
        *,
        temperature: float,
    ) -> list[str]:
        import torch

        max_pair_chars = max(len(src["raw"]) + len(src["clean"]) for _, src in batch)
        batch_max_new = min(self.max_new_tokens, max(96, int(max_pair_chars * 0.65) + 64))
        prompts = [
            self._tokenizer.apply_chat_template(
                build_teacher_adaptation_prompt(
                    src["raw"], src["clean"], lang, str(src.get("category", "general"))
                ),
                tokenize=False,
                add_generation_prompt=True,
            )
            for lang, src in batch
        ]
        enc = self._tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1536)
        enc = {k: v.to(self._model.device) for k, v in enc.items()}
        with torch.inference_mode():
            gen = self._model.generate(
                **enc,
                max_new_tokens=batch_max_new,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        prompt_len = enc["input_ids"].shape[1]
        return self._tokenizer.batch_decode(gen[:, prompt_len:], skip_special_tokens=True)

    def run_generation_loop(
        self,
        *,
        rows_by_lang: dict[str, list[MultilingualRow]],
        seen_raw_by_lang: dict[str, set[str]],
        quotas: dict[str, int],
        by_cat: dict[str, list[dict[str, Any]]],
        weighted_cats: list[str],
        max_rounds: int,
        rng: random.Random,
        stats: dict[str, Any],
        cache_path: Path | None = None,
    ) -> None:
        cat_cursors: dict[str, int] = {cat: 0 for cat in by_cat}
        used_src_by_lang: dict[str, set[str]] = {code: set() for code in MULTILINGUAL_CODES}
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_fh = cache_path.open("a", encoding="utf-8") if cache_path is not None else None
        t_start = time.time()
        total_target = sum(quotas.values())

        try:
            for round_idx in range(1, max_rounds + 1):
                oversample = 1.35 if round_idx == 1 else (2.0 if round_idx == 2 else 3.0)
                round_tasks = _plan_round_tasks(
                    rows_by_lang, quotas, by_cat, weighted_cats, cat_cursors, used_src_by_lang,
                    oversample=oversample, batch_size=self.batch_size, rng=rng,
                )
                if not round_tasks:
                    break
                stats["rounds_run"] = round_idx
                underfilled = {c: quotas[c] - len(rows_by_lang[c]) for c in MULTILINGUAL_CODES if len(rows_by_lang[c]) < quotas[c]}
                logger.info(
                    "Teacher round %d/%d: %d tasks across %d underfilled languages (deficit=%d)",
                    round_idx, max_rounds, len(round_tasks), len(underfilled), sum(underfilled.values()),
                )

                batches_done = 0
                total_batches = (len(round_tasks) + self.batch_size - 1) // self.batch_size
                for start in range(0, len(round_tasks), self.batch_size):
                    batch = [
                        (lang, src)
                        for lang, src in round_tasks[start : start + self.batch_size]
                        if len(rows_by_lang[lang]) < quotas[lang]
                    ]
                    if not batch:
                        continue
                    decoded = self.generate_batch(batch, temperature=0.5 if round_idx == 1 else 0.65)
                    for (lang, src), out_text in zip(batch, decoded):
                        if len(rows_by_lang[lang]) >= quotas[lang]:
                            continue
                        row, rej_key = self._validate_teacher_output(lang, src, out_text, seen_raw_by_lang[lang])
                        if row is None:
                            stats["rejected"] += 1
                            stats["rejection_reasons"][rej_key] = stats["rejection_reasons"].get(rej_key, 0) + 1
                            continue
                        seen_raw_by_lang[lang].add(row.raw.lower())
                        rows_by_lang[lang].append(row)
                        if cache_fh is not None:
                            cache_fh.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
                            cache_fh.flush()

                    batches_done += 1
                    if batches_done == 1 or batches_done % 5 == 0 or start + self.batch_size >= len(round_tasks):
                        accepted_now = sum(len(v) for v in rows_by_lang.values())
                        elapsed = max(0.1, time.time() - t_start)
                        logger.info(
                            "Teacher round %d batch %d/%d | accepted=%d/%d | rejected=%d | %.1f rows/s | elapsed=%.0fs",
                            round_idx, batches_done, total_batches, accepted_now, total_target,
                            stats["rejected"], accepted_now / elapsed, elapsed,
                        )
        finally:
            if cache_fh is not None:
                cache_fh.close()

    def _validate_teacher_output(
        self,
        lang: str,
        src: dict[str, Any],
        out_text: str,
        seen_raw: set[str],
    ) -> tuple[MultilingualRow | None, str]:
        parsed = parse_teacher_json(out_text)
        if parsed is None:
            return None, "invalid_json"
        raw_t, clean_t = parsed
        raw_key = raw_t.lower()
        if raw_key in seen_raw:
            return None, "duplicate_raw"

        cat = str(src.get("category", "general"))
        source_name = f"teacher:{self.teacher_model}"
        is_pass = cat.strip().lower() in PASSTHROUGH_CATEGORIES
        is_ret = is_retraction_category(cat, source_name)
        ok_raw, raw_reason = self.validator.check_language(raw_t, lang)
        if not ok_raw:
            return None, f"raw:{raw_reason.split('(')[0].strip()}"
        ok, reason = validate_pair(
            raw_t, clean_t, lang,
            is_retraction=is_ret,
            is_passthrough=is_pass,
            category=cat,
            lang_validator=self.validator,
        )
        if not ok:
            return None, reason.split("(")[0].strip()

        return (
            MultilingualRow(
                raw=raw_t,
                clean=clean_t,
                lang=lang,
                tier=PARAKEET_V3_LANGUAGES[lang].tier,
                source=source_name,
                category=cat,
                is_retraction=is_ret,
            ),
            "ok",
        )


def generate_multilingual_with_teacher(
    english_pool: Sequence[dict[str, Any]],
    total_multilingual: int,
    *,
    teacher_model: str = DEFAULT_TEACHER_MODEL,
    cache_path: Path | None = None,
    batch_size: int = 192,
    max_new_tokens: int = 384,
    max_rounds: int = 4,
    seed: int = 407,
    lang_validator: LanguageValidator | None = None,
) -> tuple[list[MultilingualRow], dict[str, Any]]:
    """Generate and validate the 25-language synthetic slice using a local open-weight teacher."""
    rng = random.Random(seed)
    validator = lang_validator or LanguageValidator(enabled=True)
    quotas = allocate_language_quotas(total_multilingual)
    rows_by_lang: dict[str, list[MultilingualRow]] = {code: [] for code in MULTILINGUAL_CODES}
    seen_raw_by_lang: dict[str, set[str]] = {code: set() for code in MULTILINGUAL_CODES}
    stats: dict[str, Any] = {
        "teacher_model": teacher_model,
        "requested": total_multilingual,
        "accepted": 0,
        "rejected": 0,
        "rounds_run": 0,
        "rejection_reasons": {},
    }

    if cache_path is not None and cache_path.is_file():
        with cache_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                lang = rec.get("lang")
                if lang not in rows_by_lang or len(rows_by_lang[lang]) >= quotas[lang]:
                    continue
                cat = str(rec.get("category", "general"))
                raw_s = normalize_whitespace(str(rec.get("raw", "")))
                clean_s = normalize_whitespace(str(rec.get("clean", "")))
                raw_key = raw_s.lower()
                if raw_key in seen_raw_by_lang[lang]:
                    continue
                is_ret = is_retraction_category(cat, str(rec.get("source", "")))
                ok, _ = validate_pair(
                    raw_s, clean_s, lang,
                    is_retraction=is_ret,
                    is_passthrough=(cat.lower() in PASSTHROUGH_CATEGORIES),
                    category=cat,
                    lang_validator=validator,
                )
                if not ok:
                    continue
                seen_raw_by_lang[lang].add(raw_key)
                rec["is_retraction"] = is_ret
                rows_by_lang[lang].append(MultilingualRow(**rec))
        logger.info("Loaded %d valid cached multilingual rows from %s", sum(len(v) for v in rows_by_lang.values()), cache_path)

    for seed_row in BUILTIN_MULTILINGUAL_SEEDS:
        raw_key = seed_row.raw.strip().lower()
        if raw_key not in seen_raw_by_lang[seed_row.lang] and len(rows_by_lang[seed_row.lang]) < quotas[seed_row.lang]:
            seen_raw_by_lang[seed_row.lang].add(raw_key)
            rows_by_lang[seed_row.lang].append(seed_row)

    if all(len(rows_by_lang[code]) >= quotas[code] for code in MULTILINGUAL_CODES):
        all_rows = [r for code in MULTILINGUAL_CODES for r in rows_by_lang[code][: quotas[code]]]
        stats["accepted"] = len(all_rows)
        return all_rows, stats

    by_cat, weighted_cats = _build_source_buckets(english_pool, rng)
    with MultilingualTeacher(
        teacher_model, batch_size=batch_size, max_new_tokens=max_new_tokens, lang_validator=validator
    ) as teacher:
        teacher.run_generation_loop(
            rows_by_lang=rows_by_lang,
            seen_raw_by_lang=seen_raw_by_lang,
            quotas=quotas,
            by_cat=by_cat,
            weighted_cats=weighted_cats,
            max_rounds=max_rounds,
            rng=rng,
            stats=stats,
            cache_path=cache_path,
        )

    all_rows = [r for code in MULTILINGUAL_CODES for r in rows_by_lang[code][: quotas[code]]]
    stats["accepted"] = len(all_rows)
    return all_rows, stats


def generate_multilingual_passthroughs(
    english_passthrough_pool: Sequence[dict[str, Any]],
    total_passthroughs: int,
    *,
    existing_raw_by_lang: dict[str, set[str]] | None = None,
    teacher_model: str = DEFAULT_TEACHER_MODEL,
    batch_size: int = 192,
    max_new_tokens: int = 384,
    max_rounds: int = 4,
    seed: int = 407,
    lang_validator: LanguageValidator | None = None,
) -> tuple[list[MultilingualRow], dict[str, Any]]:
    """Regenerate ONLY multilingual passthrough rows (strict ``raw == clean``) across all 24 languages."""
    rng = random.Random(seed)
    validator = lang_validator or LanguageValidator(enabled=True)
    quotas = allocate_language_quotas(total_passthroughs)
    rows_by_lang: dict[str, list[MultilingualRow]] = {code: [] for code in MULTILINGUAL_CODES}
    seen_raw_by_lang: dict[str, set[str]] = {
        code: set(existing_raw_by_lang.get(code, set())) if existing_raw_by_lang else set()
        for code in MULTILINGUAL_CODES
    }
    stats: dict[str, Any] = {
        "teacher_model": teacher_model,
        "mode": "passthrough_only_regeneration",
        "requested": total_passthroughs,
        "accepted": 0,
        "rejected": 0,
        "rounds_run": 0,
        "rejection_reasons": {},
    }

    clean_pool = [
        {
            "raw": normalize_whitespace(str(r["clean"])),
            "clean": normalize_whitespace(str(r["clean"])),
            "category": "clean_passthrough",
            "is_retraction": False,
        }
        for r in english_passthrough_pool
        if normalize_whitespace(str(r.get("raw", ""))) == normalize_whitespace(str(r.get("clean", "")))
        and 10 <= len(str(r.get("clean", ""))) <= 450
    ]
    if not clean_pool:
        raise ValueError("english_passthrough_pool has no valid raw == clean rows")

    rng.shuffle(clean_pool)
    by_cat = {"clean_passthrough": clean_pool}
    weighted_cats = ["clean_passthrough"]

    with MultilingualTeacher(
        teacher_model, batch_size=batch_size, max_new_tokens=max_new_tokens, lang_validator=validator
    ) as teacher:
        teacher.run_generation_loop(
            rows_by_lang=rows_by_lang,
            seen_raw_by_lang=seen_raw_by_lang,
            quotas=quotas,
            by_cat=by_cat,
            weighted_cats=weighted_cats,
            max_rounds=max_rounds,
            rng=rng,
            stats=stats,
            cache_path=None,
        )

    all_rows = [r for code in MULTILINGUAL_CODES for r in rows_by_lang[code][: quotas[code]]]
    stats["accepted"] = len(all_rows)
    return all_rows, stats
