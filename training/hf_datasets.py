"""Hugging Face dataset license auditing and ingestion for Talksify fine-tuning.

Ingests all 5 configured datasets unconditionally (4 English base datasets + native German
``nyralabs/disfluency_speech_german``) while recording each dataset's live HF license metadata
in the build manifest.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from conventions import apply_talksify_conventions
from qa_filters import (
    has_email_block,
    is_empty_sentinel,
    is_retraction_category,
    normalize_whitespace,
)

logger = logging.getLogger("talksify.hf_datasets")

ALLOWED_OPEN_LICENSES: frozenset[str] = frozenset(
    {"mit", "apache-2.0", "cc-by-4.0", "bsd-3-clause", "bsd-2-clause", "cc0-1.0"}
)

HF_DATASETS_SPEC: tuple[dict[str, str], ...] = (
    {"id": "category271/whispernow-cleanup", "role": "en_base"},
    {"id": "shantanugoel/aawaaz-transcript-cleanup-dataset", "role": "en_base"},
    {"id": "Akash-Sakala/transcript-formatter-curriculum", "role": "en_curriculum"},
    {"id": "google-research-datasets/disfl_qa", "role": "en_disfl_qa"},
    {"id": "nyralabs/disfluency_speech_german", "role": "de_native"},
)

# Curriculum categories kept (explicitly excluding email_block and email signatures).
CURRICULUM_KEEP_CATEGORIES: frozenset[str] = frozenset(
    {
        "passthrough",
        "stutter_removal",
        "filler_removal",
        "false_start_collapse",
        "backtrack_phrase",
        "list_content",
        "sent_end_punct",
        "sent_start_cap",
        "mixed_punct_cap",
        "comma_insertion",
        "pronoun_I_cap",
        "day_month_cap",
        "homophone_fix",
        "mixed_L1",
        "mixed_L2",
    }
)

AAWAAZ_KEEP_FILES: tuple[str, ...] = (
    "self_corrections_heavy.jsonl",
    "casual_conversation.jsonl",
    "meeting_notes.jsonl",
    "technical_code.jsonl",
    "academic_research.jsonl",
    "financial_business.jsonl",
    "medical_clinical.jsonl",
    "creative_writing.jsonl",
    "recipe_cooking.jsonl",
    "legal_contract.jsonl",
)


@dataclass(frozen=True)
class LicenseCheck:
    repo_id: str
    licenses: tuple[str, ...]
    compatible: bool
    status: str


def _fetch_json(url: str, timeout: int = 20) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "talksify-f407/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def verify_hf_dataset_license(repo_id: str) -> LicenseCheck:
    """Query the live Hugging Face dataset card API and record permissive license status."""
    url = f"https://huggingface.co/api/datasets/{repo_id}"
    try:
        meta = _fetch_json(url)
    except Exception as exc:
        return LicenseCheck(repo_id=repo_id, licenses=(), compatible=False, status=f"api_error:{exc}")

    raw_lic = meta.get("cardData", {}).get("license")
    licenses: list[str] = []
    if isinstance(raw_lic, str):
        licenses = [raw_lic.strip().lower()]
    elif isinstance(raw_lic, list):
        licenses = [str(x).strip().lower() for x in raw_lic if x]
    if not licenses:
        for tag in meta.get("tags", []):
            if isinstance(tag, str) and tag.startswith("license:"):
                licenses.append(tag.split(":", 1)[1].strip().lower())

    unique_lics = tuple(sorted(set(licenses)))
    if not unique_lics or any(lic in ("unknown", "other") for lic in unique_lics):
        return LicenseCheck(
            repo_id=repo_id,
            licenses=unique_lics,
            compatible=False,
            status="flagged_unknown_or_non_osi_license",
        )
    compatible = all(lic in ALLOWED_OPEN_LICENSES for lic in unique_lics)
    return LicenseCheck(
        repo_id=repo_id,
        licenses=unique_lics,
        compatible=compatible,
        status="verified_open" if compatible else "flagged_non_osi_license",
    )


def audit_hf_licenses() -> dict[str, dict[str, Any]]:
    """Audit live Hugging Face licenses for all 5 datasets and return a manifest dict."""
    audit: dict[str, dict[str, Any]] = {}
    for spec in HF_DATASETS_SPEC:
        check = verify_hf_dataset_license(spec["id"])
        audit[spec["id"]] = {
            "licenses": list(check.licenses),
            "compatible": check.compatible,
            "included": True,
            "status": check.status,
            "role": spec["role"],
        }
        logger.info(
            "License audit %-52s -> %s (%s)",
            spec["id"],
            check.status,
            ",".join(check.licenses) or "none",
        )
    return audit


def _download_hf_file(repo_id: str, filename: str, cache_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        cache_dir=str(cache_dir),
    )
    return Path(local)


def load_whispernow_cleanup(cache_dir: Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    """Ingest ``category271/whispernow-cleanup`` train & val parquet splits."""
    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = []
    files = ("data/train-00000-of-00001.parquet", "data/val-00000-of-00001.parquet")
    if max_rows is not None and max_rows <= 500:
        files = ("data/val-00000-of-00001.parquet",)

    for rel in files:
        path = _download_hf_file("category271/whispernow-cleanup", rel, cache_dir)
        table = pq.read_table(path)
        inputs = table["input"].to_pylist()
        outputs = table["output"].to_pylist()
        cats = table["category"].to_pylist() if "category" in table.column_names else ["general"] * len(inputs)
        layouts = table["layout"].to_pylist() if "layout" in table.column_names else ["plain"] * len(inputs)
        for raw, clean, cat, layout in zip(inputs, outputs, cats, layouts):
            if not isinstance(raw, str) or not isinstance(clean, str):
                continue
            cat_str = str(cat or "general")
            if str(layout).lower() == "email" or "email" in cat_str.lower():
                continue
            if is_empty_sentinel(clean, cat_str):
                continue
            clean_conv = apply_talksify_conventions(clean)
            if cat_str.lower() == "no_edit" and normalize_whitespace(raw) != normalize_whitespace(clean_conv):
                cat_str = "prose"
            rows.append(
                {
                    "raw": raw,
                    "clean": clean_conv,
                    "lang": "en",
                    "tier": "en",
                    "source": "category271/whispernow-cleanup",
                    "category": cat_str,
                    "is_retraction": is_retraction_category(cat_str, "category271/whispernow-cleanup"),
                }
            )
            if max_rows is not None and len(rows) >= max_rows:
                return rows
    return rows


def load_aawaaz_dataset(cache_dir: Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    """Ingest non-email domains from ``shantanugoel/aawaaz-transcript-cleanup-dataset``."""
    rows: list[dict[str, Any]] = []
    files = AAWAAZ_KEEP_FILES if (max_rows is None or max_rows > 500) else AAWAAZ_KEEP_FILES[:2]
    for filename in files:
        path = _download_hf_file("shantanugoel/aawaaz-transcript-cleanup-dataset", filename, cache_dir)
        domain = filename.removesuffix(".jsonl")
        is_retraction = is_retraction_category(domain, "shantanugoel/aawaaz-transcript-cleanup-dataset")
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                raw = rec.get("input") or rec.get("raw")
                clean = rec.get("output") or rec.get("clean")
                if not isinstance(raw, str) or not isinstance(clean, str):
                    continue
                if has_email_block(clean):
                    continue
                rows.append(
                    {
                        "raw": raw,
                        "clean": apply_talksify_conventions(clean),
                        "lang": "en",
                        "tier": "en",
                        "source": "shantanugoel/aawaaz-transcript-cleanup-dataset",
                        "category": domain,
                        "is_retraction": is_retraction,
                    }
                )
                if max_rows is not None and len(rows) >= max_rows:
                    return rows
    return rows


def load_curriculum_dataset(cache_dir: Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    """Ingest stutters, fillers, punctuation, lists, and passthrough traps from curriculum CSVs."""
    rows: list[dict[str, Any]] = []
    files = (
        "dataset_staging/sft_l0/L0.csv",
        "dataset_staging/sft_l1/L1_train.csv",
        "dataset_staging/sft_l2/L2_train.csv",
        "dataset_staging/sft_l5/L5_train.csv",
        "dataset_staging/sft_l5/L5_val.csv",
    )
    if max_rows is not None and max_rows <= 500:
        files = ("dataset_staging/sft_l5/L5_val.csv",)

    for rel in files:
        path = _download_hf_file("Akash-Sakala/transcript-formatter-curriculum", rel, cache_dir)
        text = path.read_text(encoding="utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        for rec in reader:
            cat = (rec.get("category") or "").strip()
            if cat not in CURRICULUM_KEEP_CATEGORIES:
                continue
            raw = rec.get("input") or ""
            clean = rec.get("output") or ""
            if not raw or not clean or has_email_block(clean):
                continue
            is_retraction = is_retraction_category(cat, "Akash-Sakala/transcript-formatter-curriculum")
            clean_norm = clean if cat == "passthrough" else apply_talksify_conventions(clean)
            rows.append(
                {
                    "raw": raw,
                    "clean": clean_norm,
                    "lang": "en",
                    "tier": "en",
                    "source": "Akash-Sakala/transcript-formatter-curriculum",
                    "category": cat,
                    "is_retraction": is_retraction,
                }
            )
            if max_rows is not None and len(rows) >= max_rows:
                return rows
    return rows


def load_disfl_qa_dataset(cache_dir: Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    """Ingest human-written restarts & self-corrections on questions from ``disfl_qa``."""
    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = []
    files = ("data/train-00000-of-00001.parquet", "data/validation-00000-of-00001.parquet")
    if max_rows is not None and max_rows <= 500:
        files = ("data/validation-00000-of-00001.parquet",)

    for rel in files:
        path = _download_hf_file("google-research-datasets/disfl_qa", rel, cache_dir)
        table = pq.read_table(path)
        disf = table["disfluent question"].to_pylist()
        orig = table["original question"].to_pylist()
        for raw, clean in zip(disf, orig):
            if not isinstance(raw, str) or not isinstance(clean, str):
                continue
            raw_s = normalize_whitespace(raw)
            clean_s = apply_talksify_conventions(normalize_whitespace(clean))
            if not raw_s or not clean_s or raw_s.lower() == "#n/a":
                continue
            rows.append(
                {
                    "raw": raw_s,
                    "clean": clean_s,
                    "lang": "en",
                    "tier": "en",
                    "source": "google-research-datasets/disfl_qa",
                    "category": "disfl_qa_question",
                    "is_retraction": is_retraction_category(
                        "disfl_qa_question", "google-research-datasets/disfl_qa"
                    ),
                }
            )
            if max_rows is not None and len(rows) >= max_rows:
                return rows
    return rows


_NON_SPEECH_TAG_RE = re.compile(r"\[[^\]]+\]")
_STAR_STEM_RE = re.compile(r"(\w)\*")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")


def normalize_german_disfluency_transcript(text: str) -> str:
    """Convert ``nyralabs/disfluency_speech_german`` markup to realistic Parakeet ASR text.

    Replaces ``[UH]``/``[UM]`` with idiomatic German ``äh``/``ähm`` (capitalized at utterance or
    sentence starts), strips non-speech event tags (e.g. ``[throatclearing]``, ``[lipsmack]``),
    and removes ``*`` suffixes on truncated cut-off stems (e.g. ``bru*`` -> ``bru``).
    """
    out = re.sub(r"(?:^|(?<=[.!?]\s))\[UH\]", "Äh", text.strip())
    out = re.sub(r"(?:^|(?<=[.!?]\s))\[UM\]", "Ähm", out)
    out = out.replace("[UH]", "äh").replace("[UM]", "ähm")
    out = _NON_SPEECH_TAG_RE.sub("", out)
    out = _STAR_STEM_RE.sub(r"\1", out)
    out = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", out)
    out = normalize_whitespace(out)
    out = re.sub(r"^[,;:]+\s*", "", out)
    return out


def load_german_disfluency_dataset(
    cache_dir: Path, max_rows: int | None = None
) -> list[dict[str, Any]]:
    """Ingest native German spoken disfluencies from ``nyralabs/disfluency_speech_german``."""
    import pyarrow.parquet as pq

    path = _download_hf_file(
        "nyralabs/disfluency_speech_german",
        "data/test-00000-of-00001.parquet",
        cache_dir,
    )
    table = pq.read_table(path, columns=["verbatim_transcript", "intended_transcript"])
    verbatims = table["verbatim_transcript"].to_pylist()
    intended = table["intended_transcript"].to_pylist()
    rows: list[dict[str, Any]] = []
    for raw, clean in zip(verbatims, intended):
        if not isinstance(raw, str) or not isinstance(clean, str):
            continue
        raw_s = normalize_german_disfluency_transcript(raw)
        clean_s = normalize_whitespace(clean)
        if not raw_s or not clean_s:
            continue
        rows.append(
            {
                "raw": raw_s,
                "clean": clean_s,
                "lang": "de",
                "tier": "tier1",
                "source": "nyralabs/disfluency_speech_german",
                "category": "native_de_disfluency",
                "is_retraction": is_retraction_category(
                    "native_de_disfluency", "nyralabs/disfluency_speech_german"
                ),
            }
        )
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


def load_all_hf_datasets(
    cache_dir: Path,
    *,
    per_source_cap: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load all 4 English base datasets and the native German dataset unconditionally."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_en_rows: list[dict[str, Any]] = []
    en_loaders = (
        ("category271/whispernow-cleanup", load_whispernow_cleanup),
        ("shantanugoel/aawaaz-transcript-cleanup-dataset", load_aawaaz_dataset),
        ("Akash-Sakala/transcript-formatter-curriculum", load_curriculum_dataset),
        ("google-research-datasets/disfl_qa", load_disfl_qa_dataset),
    )
    for repo_id, loader in en_loaders:
        loaded = loader(cache_dir, max_rows=per_source_cap)
        logger.info("Loaded %d raw rows from %s", len(loaded), repo_id)
        raw_en_rows.extend(loaded)

    raw_de_rows = load_german_disfluency_dataset(cache_dir, max_rows=per_source_cap)
    logger.info("Loaded %d native German rows from nyralabs/disfluency_speech_german", len(raw_de_rows))
    return raw_en_rows, raw_de_rows
