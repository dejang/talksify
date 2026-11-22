#!/usr/bin/env python3
"""Reproducible multilingual dataset preparation pipeline for Talksify Qwen3.5-0.8B fine-tuning.

Ingests all 5 configured Hugging Face datasets (recording live license audit metadata in
``manifest.json``), adds targeted English synthetic gap-fill addressing all 5 F-406 failure
modes, generates a 25-language synthetic adaptation slice across all Parakeet TDT 0.6B v3
European languages (Tier 1 ~60%, Tier 2 ~40%), enforces strict QA & anti-English-drift filtering,
and writes stratified train/val JSONL splits aligned with Talksify's exact production
``clean_prompt`` and ``<transcript>`` format.

Usage:
    # Full GPU pipeline (25-language local teacher + ~67% EN / ~33% multilingual mix)
    uv run python prepare_data.py --output-dir .artifacts/dataset

    # Local repair of an existing dataset directory (no full re-extraction; regenerates only passthroughs)
    uv run python prepare_data.py --repair-existing --output-dir .artifacts/dataset

    # Fast smoke test (verifies live HF licenses, schema, filters, and splits in seconds)
    uv run python prepare_data.py --smoke-test --output-dir .artifacts/smoke_dataset
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from hf_datasets import (
    audit_hf_licenses,
    load_all_hf_datasets,
    normalize_german_disfluency_transcript,
)
from prompts import (
    MULTILINGUAL_CODES,
    PARAKEET_V3_LANGUAGES,
    TIER1_CODES,
    TIER2_CODES,
    build_messages,
    load_clean_prompt,
)
from qa_filters import (
    PASSTHROUGH_CATEGORIES,
    LanguageValidator,
    is_retraction_category,
    normalize_whitespace,
    validate_pair,
)
from synthetic_en import generate_english_gap_fill
from synthetic_multilingual import (
    BUILTIN_MULTILINGUAL_SEEDS,
    DEFAULT_TEACHER_MODEL,
    generate_multilingual_passthroughs,
    generate_multilingual_with_teacher,
)

__all__ = [
    "DEFAULT_SOURCE_MAX_SHARE",
    "balance_english_and_multilingual",
    "filter_and_deduplicate",
    "normalize_german_disfluency_transcript",
    "repair_existing_dataset",
    "sample_stratified_by_source",
    "stratified_split",
    "write_jsonl_with_messages",
]

logger = logging.getLogger("talksify.prepare_data")

# Cap narrow single-style question datasets so they cannot dominate general dictation prose.
DEFAULT_SOURCE_MAX_SHARE: dict[str, float] = {
    "google-research-datasets/disfl_qa": 0.08,
}


def filter_and_deduplicate(
    rows: list[dict[str, Any]],
    *,
    lang_validator: LanguageValidator,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Apply quality filters and deduplicate by normalized ``(lang, raw)`` key."""
    kept: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()

    for row in rows:
        raw = normalize_whitespace(str(row["raw"]))
        clean = normalize_whitespace(str(row["clean"]))
        lang = str(row["lang"])
        category = str(row.get("category", "general"))
        cat_lower = category.strip().lower()
        key = (lang, raw.lower())
        if key in seen:
            rejected["duplicate_raw"] += 1
            continue

        source = str(row["source"])
        is_passthrough = cat_lower in PASSTHROUGH_CATEGORIES
        is_retraction = is_retraction_category(category, source)
        ok, reason = validate_pair(
            raw,
            clean,
            lang,
            is_retraction=is_retraction,
            is_passthrough=is_passthrough,
            category=category,
            lang_validator=lang_validator,
        )
        if not ok:
            rejected[reason.split("(")[0].strip()] += 1
            continue

        seen.add(key)
        kept.append(
            {
                "raw": raw,
                "clean": clean,
                "lang": lang,
                "tier": PARAKEET_V3_LANGUAGES[lang].tier,
                "source": str(row["source"]),
                "category": category,
                "is_retraction": is_retraction,
            }
        )
    return kept, rejected


def stratified_split(
    rows: list[dict[str, Any]],
    *,
    val_ratio: float = 0.05,
    seed: int = 407,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split rows into train and validation stratified by language code."""
    rng = random.Random(seed)
    by_lang: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_lang[row["lang"]].append(row)

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for lang in sorted(by_lang):
        group = list(by_lang[lang])
        rng.shuffle(group)
        if len(group) == 1:
            train.extend(group)
            continue
        n_val = max(1, round(len(group) * val_ratio))
        n_val = min(n_val, len(group) - 1)
        val.extend(group[:n_val])
        train.extend(group[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def write_jsonl_with_messages(
    rows: list[dict[str, Any]],
    out_path: Path,
    system_prompt: str,
) -> None:
    """Serialize dataset rows paired with Talksify's exact English ``clean_prompt``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            record = {
                "raw": row["raw"],
                "clean": row["clean"],
                "lang": row["lang"],
                "tier": row["tier"],
                "source": row["source"],
                "category": row["category"],
                "messages": build_messages(
                    row["raw"], clean_transcript=row["clean"], system_prompt=system_prompt
                ),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def sample_stratified_by_source(
    rows: list[dict[str, Any]],
    limit: int,
    seed: int = 407,
    source_caps: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Sample up to ``limit`` rows across sources while honoring optional per-source caps."""
    rng = random.Random(seed)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
    sources = sorted(by_source)
    caps = source_caps or {}
    for src in sources:
        rng.shuffle(by_source[src])
        if src in caps:
            by_source[src] = by_source[src][: max(0, caps[src])]

    total_available = sum(len(v) for v in by_source.values())
    if total_available <= limit:
        selected = [row for src in sources for row in by_source[src]]
        rng.shuffle(selected)
        return selected

    selected: list[dict[str, Any]] = []
    idx = 0
    while len(selected) < limit:
        added = False
        for src in sources:
            bucket = by_source[src]
            if idx < len(bucket) and len(selected) < limit:
                selected.append(bucket[idx])
                added = True
        if not added:
            break
        idx += 1
    rng.shuffle(selected)
    return selected


def balance_english_and_multilingual(
    en_rows: list[dict[str, Any]],
    multi_rows: list[dict[str, Any]],
    *,
    target_en_ratio: float = 0.67,
    max_total_en: int = 28000,
    seed: int = 407,
    source_max_share: dict[str, float] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Balance English & multilingual pools toward ~65-70% EN / ~30-35% multilingual."""
    rng = random.Random(seed)
    shares = DEFAULT_SOURCE_MAX_SHARE if source_max_share is None else source_max_share

    gapfill = [r for r in en_rows if r["source"] == "synthetic_en_gapfill"]
    other_en = [r for r in en_rows if r["source"] != "synthetic_en_gapfill"]

    if multi_rows:
        ratio_cap = round(len(multi_rows) * (target_en_ratio / (1.0 - target_en_ratio)))
        en_budget = min(max_total_en, max(len(gapfill), ratio_cap))
    else:
        en_budget = min(max_total_en, len(en_rows))

    # Compute source caps against actual realizable English size so capped sources obey share even
    # when other pools are smaller than en_budget.
    uncapped_total = len(gapfill) + sum(1 for r in other_en if r["source"] not in shares)
    source_caps: dict[str, int] = {}
    for src, max_share in shares.items():
        if max_share <= 0.0:
            source_caps[src] = 0
        else:
            budget_cap = round(en_budget * max_share)
            realizable_cap = round(uncapped_total * (max_share / max(1e-6, 1.0 - max_share)))
            source_caps[src] = max(1, min(budget_cap, realizable_cap))

    keep_other = max(0, en_budget - len(gapfill))
    selected_en = gapfill + sample_stratified_by_source(
        other_en, keep_other, seed=seed, source_caps=source_caps
    )
    rng.shuffle(selected_en)
    return selected_en, list(multi_rows)


def _write_manifest(
    *,
    output_dir: Path,
    combined: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    system_prompt: str,
    smoke_test: bool,
    seed: int,
    license_audit: dict[str, Any],
    en_rejected: Counter[str],
    multi_rejected: Counter[str],
    teacher_stats: dict[str, Any],
) -> dict[str, Any]:
    total = len(combined)
    en_count = sum(1 for r in combined if r["lang"] == "en")
    tier1_count = sum(1 for r in combined if r["lang"] in TIER1_CODES)
    tier2_count = sum(1 for r in combined if r["lang"] in TIER2_CODES)
    multi_count = tier1_count + tier2_count

    manifest = {
        "smoke_test": smoke_test,
        "seed": seed,
        "system_prompt_len": len(system_prompt),
        "license_audit": license_audit,
        "totals": {
            "combined": total,
            "train": len(train_rows),
            "val": len(val_rows),
            "en": en_count,
            "en_pct": round(100.0 * en_count / max(1, total), 2),
            "multilingual": multi_count,
            "multilingual_pct": round(100.0 * multi_count / max(1, total), 2),
            "tier1": tier1_count,
            "tier1_pct_of_multi": round(100.0 * tier1_count / max(1, multi_count), 2),
            "tier2": tier2_count,
            "tier2_pct_of_multi": round(100.0 * tier2_count / max(1, multi_count), 2),
        },
        "languages_present": sorted({r["lang"] for r in combined}),
        "languages_missing": sorted(set(MULTILINGUAL_CODES) - {r["lang"] for r in combined}),
        "by_source": dict(Counter(r["source"] for r in combined)),
        "by_lang": dict(Counter(r["lang"] for r in combined)),
        "by_category": dict(Counter(r["category"] for r in combined)),
        "en_rejected": dict(en_rejected),
        "multi_rejected": dict(multi_rejected),
        "teacher_stats": teacher_stats,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def _load_existing_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def repair_existing_dataset(
    output_dir: Path,
    *,
    system_prompt: str,
    lang_validator: LanguageValidator,
    en_gapfill_rows: int = 2800,
    regen_passthroughs: int = 960,
    skip_teacher: bool = False,
    teacher_model: str = DEFAULT_TEACHER_MODEL,
    teacher_batch_size: int = 192,
    val_ratio: float = 0.05,
    seed: int = 407,
) -> dict[str, Any]:
    """Correct an existing ``.artifacts/dataset`` locally without full HF or teacher re-extraction."""
    prior_manifest_path = output_dir / "manifest.json"
    prior_manifest = json.loads(prior_manifest_path.read_text(encoding="utf-8")) if prior_manifest_path.is_file() else {}
    existing = _load_existing_jsonl(output_dir / "train.jsonl") + _load_existing_jsonl(output_dir / "val.jsonl")
    if not existing:
        raise FileNotFoundError(f"No existing train.jsonl/val.jsonl rows found in {output_dir}")

    raw_en: list[dict[str, Any]] = []
    raw_multi: list[dict[str, Any]] = []
    for row in existing:
        rec = {
            "raw": str(row["raw"]),
            "clean": str(row["clean"]),
            "lang": str(row["lang"]),
            "tier": str(row.get("tier", PARAKEET_V3_LANGUAGES[str(row["lang"])].tier)),
            "source": str(row["source"]),
            "category": str(row.get("category", "general")),
        }
        # Replace synthetic_en_gapfill and builtin_multilingual_seed from current templates, and
        # drop stale teacher thinking_aloud rows adapted from pre-fix English templates.
        if rec["source"] in ("builtin_multilingual_seed", "synthetic_en_gapfill"):
            continue
        if rec["source"].startswith("teacher:") and rec["category"].strip().lower() == "thinking_aloud":
            continue
        if rec["source"] == "category271/whispernow-cleanup" and rec["category"] == "no_edit":
            if normalize_whitespace(rec["raw"]) != normalize_whitespace(rec["clean"]):
                rec["category"] = "prose"
        rec["is_retraction"] = is_retraction_category(rec["category"], rec["source"])
        if rec["lang"] == "en":
            raw_en.append(rec)
        else:
            raw_multi.append(rec)

    for synth_row in generate_english_gap_fill(seed=seed, target_count=en_gapfill_rows):
        raw_en.append(
            {
                "raw": synth_row.raw,
                "clean": synth_row.clean,
                "lang": "en",
                "tier": "en",
                "source": synth_row.source,
                "category": synth_row.category,
                "is_retraction": synth_row.is_retraction,
            }
        )
    for seed_row in BUILTIN_MULTILINGUAL_SEEDS:
        raw_multi.append(asdict(seed_row))

    en_clean, en_rejected = filter_and_deduplicate(raw_en, lang_validator=lang_validator)
    multi_clean, multi_rejected = filter_and_deduplicate(raw_multi, lang_validator=lang_validator)

    teacher_stats: dict[str, Any] = {"mode": "local_repair_no_regen"}
    if regen_passthroughs > 0 and not skip_teacher:
        en_pass_pool = [
            r for r in en_clean
            if r["category"].strip().lower() in PASSTHROUGH_CATEGORIES and r["raw"] == r["clean"]
        ]
        existing_raw_by_lang: dict[str, set[str]] = defaultdict(set)
        for r in multi_clean:
            existing_raw_by_lang[r["lang"]].add(r["raw"].lower())
        new_pass_rows, teacher_stats = generate_multilingual_passthroughs(
            en_pass_pool,
            total_passthroughs=regen_passthroughs,
            existing_raw_by_lang=existing_raw_by_lang,
            teacher_model=teacher_model,
            batch_size=teacher_batch_size,
            seed=seed,
            lang_validator=lang_validator,
        )
        for row in new_pass_rows:
            multi_clean.append(asdict(row))
        multi_clean, regen_rej = filter_and_deduplicate(multi_clean, lang_validator=lang_validator)
        multi_rejected.update(regen_rej)

    # Cap disfl_qa share without re-extracting other sources.
    en_selected, multi_selected = balance_english_and_multilingual(
        en_clean, multi_clean, target_en_ratio=0.67, max_total_en=len(en_clean), seed=seed
    )

    # Rewrite teacher_multilingual_cache.jsonl so stale invalid rows cannot leak into future runs.
    cache_path = output_dir / "teacher_multilingual_cache.jsonl"
    teacher_rows = [r for r in multi_selected if r["source"].startswith("teacher:")]
    with cache_path.open("w", encoding="utf-8") as fh:
        for r in teacher_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    combined = en_selected + multi_selected
    train_rows, val_rows = stratified_split(combined, val_ratio=val_ratio, seed=seed)
    write_jsonl_with_messages(train_rows, output_dir / "train.jsonl", system_prompt)
    write_jsonl_with_messages(val_rows, output_dir / "val.jsonl", system_prompt)

    return _write_manifest(
        output_dir=output_dir,
        combined=combined,
        train_rows=train_rows,
        val_rows=val_rows,
        system_prompt=system_prompt,
        smoke_test=False,
        seed=seed,
        license_audit=prior_manifest.get("license_audit", {}),
        en_rejected=en_rejected,
        multi_rejected=multi_rejected,
        teacher_stats=teacher_stats,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Prepare Talksify multilingual Qwen3.5-0.8B fine-tuning dataset.")
    parser.add_argument("--output-dir", type=Path, default=root / ".artifacts" / "dataset")
    parser.add_argument("--cache-dir", type=Path, default=root / ".cache" / "hf_datasets")
    parser.add_argument("--max-en-rows", type=int, default=28000)
    parser.add_argument("--en-gapfill-rows", type=int, default=2800)
    parser.add_argument("--multilingual-rows", type=int, default=10000)
    parser.add_argument("--teacher-model", type=str, default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--teacher-batch-size", type=int, default=192)
    parser.add_argument("--skip-teacher", action="store_true")
    parser.add_argument("--skip-hf-download", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--repair-existing", action="store_true")
    parser.add_argument("--regen-passthroughs", type=int, default=960)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=407)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)

    resolved_out = str(args.output_dir.resolve())
    if "/home/agent/recordings" in resolved_out or "cleanup_quality_eval" in resolved_out:
        raise ValueError(f"Refusing contaminated output path: {resolved_out}")

    system_prompt = load_clean_prompt()
    lang_validator = LanguageValidator(enabled=True)

    if args.repair_existing:
        manifest = repair_existing_dataset(
            args.output_dir,
            system_prompt=system_prompt,
            lang_validator=lang_validator,
            en_gapfill_rows=args.en_gapfill_rows,
            regen_passthroughs=args.regen_passthroughs,
            skip_teacher=args.skip_teacher,
            teacher_model=args.teacher_model,
            teacher_batch_size=args.teacher_batch_size,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        logger.info(
            "Repaired dataset in %s (train=%d, val=%d, EN=%.1f%%, multi=%.1f%%, langs=%d/25)",
            args.output_dir,
            manifest["totals"]["train"],
            manifest["totals"]["val"],
            manifest["totals"]["en_pct"],
            manifest["totals"]["multilingual_pct"],
            len(manifest["languages_present"]),
        )
        return 0

    license_audit: dict[str, dict[str, Any]] = {}
    per_source_cap = 200 if args.smoke_test else None
    raw_en_rows: list[dict[str, Any]] = []
    raw_de_native_rows: list[dict[str, Any]] = []
    if not args.skip_hf_download:
        license_audit = audit_hf_licenses()
        raw_en_rows, raw_de_native_rows = load_all_hf_datasets(
            args.cache_dir, per_source_cap=per_source_cap
        )

    gapfill_count = 120 if args.smoke_test else args.en_gapfill_rows
    for row in generate_english_gap_fill(seed=args.seed, target_count=gapfill_count):
        raw_en_rows.append(
            {
                "raw": row.raw,
                "clean": row.clean,
                "lang": "en",
                "tier": "en",
                "source": row.source,
                "category": row.category,
                "is_retraction": row.is_retraction,
            }
        )

    en_clean, en_rejected = filter_and_deduplicate(raw_en_rows, lang_validator=lang_validator)
    logger.info("Filtered English rows: %d kept, %d rejected", len(en_clean), sum(en_rejected.values()))

    teacher_stats: dict[str, Any] = {"mode": "builtin_seeds_only"}
    raw_multi_rows: list[dict[str, Any]] = list(raw_de_native_rows)
    if args.smoke_test or args.skip_teacher:
        raw_multi_rows.extend(asdict(r) for r in BUILTIN_MULTILINGUAL_SEEDS)
    else:
        teacher_cache = args.output_dir / "teacher_multilingual_cache.jsonl"
        gen_rows, teacher_stats = generate_multilingual_with_teacher(
            en_clean,
            total_multilingual=args.multilingual_rows,
            teacher_model=args.teacher_model,
            cache_path=teacher_cache,
            batch_size=args.teacher_batch_size,
            seed=args.seed,
            lang_validator=lang_validator,
        )
        raw_multi_rows.extend(asdict(r) for r in gen_rows)

    multi_clean, multi_rejected = filter_and_deduplicate(raw_multi_rows, lang_validator=lang_validator)
    logger.info("Filtered multilingual rows: %d kept, %d rejected", len(multi_clean), sum(multi_rejected.values()))

    if args.smoke_test or args.skip_teacher:
        cap = 200 if args.smoke_test else args.max_en_rows
        selected_en = sample_stratified_by_source(en_clean, cap, seed=args.seed)
        selected_multi = multi_clean
    else:
        selected_en, selected_multi = balance_english_and_multilingual(
            en_clean,
            multi_clean,
            target_en_ratio=0.67,
            max_total_en=args.max_en_rows,
            seed=args.seed,
        )

    combined = selected_en + selected_multi
    train_rows, val_rows = stratified_split(combined, val_ratio=args.val_ratio, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_with_messages(train_rows, args.output_dir / "train.jsonl", system_prompt)
    write_jsonl_with_messages(val_rows, args.output_dir / "val.jsonl", system_prompt)

    manifest = _write_manifest(
        output_dir=args.output_dir,
        combined=combined,
        train_rows=train_rows,
        val_rows=val_rows,
        system_prompt=system_prompt,
        smoke_test=args.smoke_test,
        seed=args.seed,
        license_audit=license_audit,
        en_rejected=en_rejected,
        multi_rejected=multi_rejected,
        teacher_stats=teacher_stats,
    )
    logger.info(
        "Wrote dataset to %s (train=%d, val=%d, EN=%.1f%%, multi=%.1f%%, langs=%d/25)",
        args.output_dir,
        len(train_rows),
        len(val_rows),
        manifest["totals"]["en_pct"],
        manifest["totals"]["multilingual_pct"],
        len(manifest["languages_present"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
