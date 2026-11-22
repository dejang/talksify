#!/usr/bin/env python3
"""Merge LoRA adapter into Qwen/Qwen3.5-0.8B, export GGUF (--no-mtp), quantize, and verify.

Outputs ``Q6_K`` (~640 MB) and ``Q8_0`` (~815-835 MB) GGUFs strictly under 1 GB matching the
320-tensor text-only layout of stock ``Qwen3.5-0.8B``, and verifies each GGUF loads and runs
cleanly through ``talksify-core``'s ``TextCleaner`` / ``MistralBackend``.

Usage:
    uv run python export_gguf.py \\
        --adapter-dir .artifacts/lora_qwen35_08b/checkpoint-epoch-1 \\
        --output-dir .artifacts/export_epoch1 \\
        --quants Q6_K
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from prompts import (
    BASE_MODEL_ID,
    EXPECTED_GGUF_TENSORS_NO_MTP,
    MAX_GGUF_BYTES,
)
from talksify_replay import ReplayCase, run_talksify_replay

logger = logging.getLogger("talksify.export_gguf")

LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp.git"
LLAMA_CPP_COMMIT = "19bc41dbb41e858eebc269b49ccdec1f0d28880c"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_gguf(path: Path) -> dict[str, Any]:
    """Validate GGUF file size (< 1 GB), tensor layout (no MTP / vision), and chat template."""
    from gguf import GGUFReader

    size_bytes = path.stat().st_size
    reader = GGUFReader(str(path), mode="r")
    tensor_names = [t.name for t in reader.tensors]
    mtp_tensors = [name for name in tensor_names if "mtp" in name.lower()]
    vision_tensors = [name for name in tensor_names if "visual" in name.lower() or "vision" in name.lower()]
    has_chat_template = "tokenizer.chat_template" in reader.fields

    arch_field = reader.fields.get("general.architecture")
    arch = None
    if arch_field is not None and arch_field.parts:
        raw_val = bytes(arch_field.parts[arch_field.data[0]])
        arch = raw_val.decode("utf-8", errors="replace").rstrip("\x00")

    return {
        "path": str(path.resolve()),
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 2),
        "under_1gb": size_bytes < MAX_GGUF_BYTES,
        "sha256": sha256_file(path),
        "architecture": arch,
        "tensor_count": len(tensor_names),
        "matches_stock_320_tensors": len(tensor_names) == EXPECTED_GGUF_TENSORS_NO_MTP,
        "mtp_tensors": mtp_tensors,
        "vision_tensors": vision_tensors,
        "has_chat_template": has_chat_template,
    }


def merge_lora_adapter(
    base_model: str,
    adapter_dir: Path,
    merged_dir: Path,
) -> Path:
    """Merge PEFT LoRA weights cleanly into ``Qwen/Qwen3.5-0.8B`` in bf16."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading base model %s in bfloat16...", base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16)
    logger.info("Loading LoRA adapter from %s...", adapter_dir)
    peft_model = PeftModel.from_pretrained(model, str(adapter_dir))
    merged = peft_model.merge_and_unload()

    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_dir) if (adapter_dir / "tokenizer_config.json").is_file() else base_model
    )
    tokenizer.save_pretrained(merged_dir)
    logger.info("Saved merged bf16 model to %s", merged_dir)
    return merged_dir


def _git_head_commit(repo_dir: Path) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def ensure_llama_cpp(
    llama_cpp_dir: Path,
    commit: str = LLAMA_CPP_COMMIT,
) -> tuple[Path, Path]:
    """Locate or bootstrap pinned ``llama.cpp`` converter script and ``llama-quantize`` binary."""
    converter = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not converter.is_file():
        logger.info("Fetching pinned llama.cpp (%s) into %s...", commit[:12], llama_cpp_dir)
        llama_cpp_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "-C", str(llama_cpp_dir), "init"], check=True)
        subprocess.run(
            ["git", "-C", str(llama_cpp_dir), "remote", "add", "origin", LLAMA_CPP_REPO],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(llama_cpp_dir), "fetch", "--depth", "1", "origin", commit],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(llama_cpp_dir), "checkout", "--detach", "FETCH_HEAD"],
            check=True,
        )
    else:
        head = _git_head_commit(llama_cpp_dir)
        if head != commit:
            logger.info(
                "Updating llama.cpp in %s from %s to pinned commit %s...",
                llama_cpp_dir,
                (head or "unknown")[:12],
                commit[:12],
            )
            subprocess.run(
                ["git", "-C", str(llama_cpp_dir), "fetch", "--depth", "1", "origin", commit],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(llama_cpp_dir), "checkout", "--detach", "FETCH_HEAD"],
                check=True,
            )

    verified_head = _git_head_commit(llama_cpp_dir)
    if verified_head != commit:
        raise RuntimeError(
            f"Expected llama.cpp at pinned commit {commit}, found {verified_head!r} in {llama_cpp_dir}"
        )

    quantize_candidates = [
        llama_cpp_dir / "build" / "bin" / "llama-quantize",
        llama_cpp_dir / "build" / "llama-quantize",
    ]
    quantize_bin = next((p for p in quantize_candidates if p.is_file()), None)
    if quantize_bin is None:
        build_dir = llama_cpp_dir / "build"
        logger.info("Building llama-quantize in %s...", build_dir)
        subprocess.run(
            [
                "cmake",
                "-S",
                str(llama_cpp_dir),
                "-B",
                str(build_dir),
                "-DCMAKE_BUILD_TYPE=Release",
                "-DLLAMA_BUILD_TESTS=OFF",
                "-DLLAMA_BUILD_EXAMPLES=OFF",
                "-DLLAMA_BUILD_SERVER=OFF",
                "-DLLAMA_BUILD_COMMON=ON",
                "-DGGML_CURL=OFF",
                "-DLLAMA_CURL=OFF",
            ],
            check=True,
        )
        jobs = str(min(16, os.cpu_count() or 4))
        subprocess.run(
            ["cmake", "--build", str(build_dir), "--target", "llama-quantize", "-j", jobs],
            check=True,
        )
        quantize_bin = next((p for p in quantize_candidates if p.is_file()), None)
        if quantize_bin is None:
            raise FileNotFoundError(f"llama-quantize not found after build in {build_dir}")

    return converter, quantize_bin


def convert_to_unquantized_gguf(
    converter: Path,
    merged_dir: Path,
    out_gguf: Path,
    outtype: str = "bf16",
) -> Path:
    """Convert merged HF checkpoint to unquantized GGUF with ``--no-mtp``.

    Modern ``llama.cpp`` fuses Qwen3.5 ``(attn_qkv, attn_gate) -> attn_qkv_gate`` and
    ``(ssm_beta, ssm_alpha) -> ssm_beta_alpha`` (284 tensors), whereas Talksify's ``mistralrs``
    backend requires the unfused 320-tensor layout (``attn_qkv``, ``attn_gate``, ``ssm_beta``,
    ``ssm_alpha``). We disable ``Qwen3_5TextModel._SHARED_INPUT_MERGES`` in-process without
    mutating the pinned ``llama.cpp`` clone.
    """
    out_gguf.parent.mkdir(parents=True, exist_ok=True)
    converter_args = [
        str(merged_dir),
        "--outfile",
        str(out_gguf),
        "--outtype",
        outtype,
        "--no-mtp",
    ]
    shim = (
        "import runpy, sys\n"
        f"sys.path.insert(0, {str(converter.parent)!r})\n"
        "from conversion import qwen as _qwen_conv\n"
        "if not hasattr(_qwen_conv, 'Qwen3_5TextModel') or not hasattr(_qwen_conv.Qwen3_5TextModel, '_SHARED_INPUT_MERGES'):\n"
        "    raise RuntimeError('Pinned llama.cpp missing Qwen3_5TextModel._SHARED_INPUT_MERGES')\n"
        "_qwen_conv.Qwen3_5TextModel._SHARED_INPUT_MERGES = ()\n"
        f"sys.argv = [{str(converter)!r}] + sys.argv[1:]\n"
        f"runpy.run_path({str(converter)!r}, run_name='__main__')\n"
    )
    cmd = [sys.executable, "-c", shim, *converter_args]

    logger.info("Running GGUF conversion: %s", " ".join(converter_args))
    subprocess.run(cmd, check=True)
    return out_gguf


def quantize_gguf(quantize_bin: Path, src_gguf: Path, dst_gguf: Path, quant: str) -> Path:
    """Quantize an F16/BF16 GGUF to target quantization (e.g. ``Q6_K`` or ``Q8_0``)."""
    dst_gguf.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(quantize_bin), str(src_gguf), str(dst_gguf), quant]
    logger.info("Quantizing %s -> %s (%s)...", src_gguf.name, dst_gguf.name, quant)
    subprocess.run(cmd, check=True)
    return dst_gguf


def verify_gguf_with_talksify(gguf_path: Path, cargo_manifest: Path | None = None) -> dict[str, Any]:
    """Load and run the exported GGUF end-to-end through ``talksify-cli --replay``.

    Stages the GGUF inside an isolated temporary model directory so bundled
    ``src-tauri/models/qwen-0.6b/model.gguf`` is never touched.
    """
    smoke_cases = [
        ReplayCase(
            file="smoke_en.wav",
            duration_secs=3.5,
            raw="Um, I think we should, uh, review the changelog before tagging.",
            expected="I think we should review the changelog before tagging.",
        ),
        ReplayCase(
            file="smoke_de.wav",
            duration_secs=3.2,
            raw="Äh, wir prüfen die Protokolle vor dem nächsten Neustart.",
            expected="Wir prüfen die Protokolle vor dem nächsten Neustart.",
        ),
    ]
    result = run_talksify_replay(
        gguf_path,
        smoke_cases,
        cargo_manifest=cargo_manifest,
        tmp_prefix="talksify_gguf_verify_",
    )
    return {
        "verified": True,
        "stdout_tail": "\n".join(result.stdout.strip().splitlines()[-12:]),
        "stderr_tail": "\n".join(result.stderr.strip().splitlines()[-12:]),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge Qwen3.5-0.8B LoRA adapter and export <1GB GGUFs (--no-mtp)."
    )
    parser.add_argument("--base-model", type=str, default=BASE_MODEL_ID)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / ".artifacts" / "export",
    )
    parser.add_argument(
        "--llama-cpp-dir",
        type=Path,
        default=Path(__file__).resolve().parent / ".artifacts" / "llama.cpp",
    )
    parser.add_argument("--outtype", choices=("f16", "bf16"), default="bf16")
    parser.add_argument("--quants", nargs="+", default=["Q6_K", "Q8_0"])
    parser.add_argument("--name-prefix", type=str, default="talksify-qwen35-0.8b")
    parser.add_argument("--keep-unquantized", action="store_true")
    parser.add_argument(
        "--skip-talksify-verify",
        action="store_true",
        help="Skip running talksify-cli --replay smoke inference on exported GGUFs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged_dir = args.output_dir / "merged_hf"
    merge_lora_adapter(args.base_model, args.adapter_dir, merged_dir)

    converter, quantize_bin = ensure_llama_cpp(args.llama_cpp_dir)
    unquant_gguf = args.output_dir / f"{args.name_prefix}-{args.outtype.upper()}.gguf"
    convert_to_unquantized_gguf(converter, merged_dir, unquant_gguf, outtype=args.outtype)

    unquant_info = inspect_gguf(unquant_gguf)
    logger.info(
        "Converted unquantized GGUF: %d tensors, arch=%s, chat_template=%s, mtp=%d",
        unquant_info["tensor_count"],
        unquant_info["architecture"],
        unquant_info["has_chat_template"],
        len(unquant_info["mtp_tensors"]),
    )
    if unquant_info["mtp_tensors"]:
        raise RuntimeError(f"Unexpected MTP tensors present: {unquant_info['mtp_tensors'][:5]}")
    if unquant_info["vision_tensors"]:
        raise RuntimeError(
            f"Unexpected vision tensors present: {unquant_info['vision_tensors'][:5]}"
        )
    if not unquant_info["has_chat_template"]:
        raise RuntimeError("Exported GGUF is missing tokenizer.chat_template metadata")
    if not unquant_info["matches_stock_320_tensors"]:
        raise RuntimeError(
            f"Expected stock 320-tensor unfused Qwen3.5 layout for mistralrs compatibility, "
            f"got {unquant_info['tensor_count']} tensors"
        )

    exports: dict[str, dict[str, Any]] = {}
    for quant in args.quants:
        dst = args.output_dir / f"{args.name_prefix}-{quant}.gguf"
        quantize_gguf(quantize_bin, unquant_gguf, dst, quant)
        info = inspect_gguf(dst)
        if not info["under_1gb"]:
            raise RuntimeError(
                f"Exported {dst.name} exceeds 1 GB strict constraint: {info['size_bytes']} bytes"
            )
        if not args.skip_talksify_verify:
            info["talksify_verify"] = verify_gguf_with_talksify(dst)
        exports[quant] = info
        logger.info(
            "Exported %s: %d bytes (%.1f MB), %d tensors, sha256=%s",
            dst.name,
            info["size_bytes"],
            info["size_mb"],
            info["tensor_count"],
            info["sha256"],
        )

    if not args.keep_unquantized and unquant_gguf.is_file():
        unquant_gguf.unlink()

    manifest = {
        "base_model": args.base_model,
        "adapter_dir": str(args.adapter_dir.resolve()),
        "llama_cpp_commit": LLAMA_CPP_COMMIT,
        "unquantized_check": {
            "architecture": unquant_info["architecture"],
            "tensor_count": unquant_info["tensor_count"],
            "matches_stock_320_tensors": unquant_info["matches_stock_320_tensors"],
            "has_chat_template": unquant_info["has_chat_template"],
        },
        "exports": exports,
    }
    (args.output_dir / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
