"""Shared helper for running candidate GGUFs through ``talksify-cli --replay``.

Stages a candidate GGUF inside an isolated temporary ``models/qwen-0.6b/model.gguf`` slot
so the bundled ``src-tauri/models/qwen-0.6b/model.gguf`` is never touched.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from prompts import REPO_ROOT

logger = logging.getLogger("talksify.replay")


@dataclass(frozen=True)
class ReplayCase:
    file: str
    duration_secs: float
    raw: str
    expected: str


@dataclass(frozen=True)
class ReplayResult:
    stdout: str
    stderr: str
    llm_outputs: list[str]


def parse_llm_outputs(stderr: str) -> list[str]:
    """Extract ``[llm] output (...): "..."`` strings logged by ``LlmPostProcessor`` in order."""
    outputs: list[str] = []
    marker = "[llm] output ("
    for line in stderr.splitlines():
        idx = line.find(marker)
        if idx == -1:
            continue
        colon = line.find("): ", idx)
        if colon == -1:
            continue
        raw_repr = line[colon + 3 :].strip()
        try:
            parsed = json.loads(raw_repr)
            outputs.append(parsed if isinstance(parsed, str) else str(parsed))
        except json.JSONDecodeError:
            outputs.append(raw_repr.strip('"'))
    return outputs


def run_talksify_replay(
    gguf_path: Path,
    cases: Sequence[ReplayCase],
    *,
    cargo_manifest: Path | None = None,
    rust_log: str | None = None,
    tmp_prefix: str = "talksify_replay_",
) -> ReplayResult:
    """Stage ``gguf_path`` in a temp model dir and execute ``talksify-cli --replay``."""
    manifest = cargo_manifest or (REPO_ROOT / "src-tauri" / "Cargo.toml")
    with tempfile.TemporaryDirectory(prefix=tmp_prefix) as tmp:
        tmp_path = Path(tmp)
        model_slot = tmp_path / "models" / "qwen-0.6b"
        model_slot.mkdir(parents=True, exist_ok=True)
        (model_slot / "model.gguf").symlink_to(gguf_path.resolve())

        replay_json = tmp_path / "replay.json"
        payload = [asdict(case) for case in cases]
        replay_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        cmd = [
            "cargo",
            "run",
            "--manifest-path",
            str(manifest),
            "-p",
            "talksify-cli",
            "--",
            "--replay",
            str(replay_json),
            "--models",
            str(tmp_path / "models"),
        ]
        env = dict(os.environ)
        if rust_log is not None:
            env["RUST_LOG"] = rust_log

        logger.info("Running talksify-cli --replay (%d cases) with %s...", len(cases), gguf_path.name)
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"talksify-cli --replay failed for {gguf_path} (exit {proc.returncode}):\n"
                f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        return ReplayResult(
            stdout=proc.stdout,
            stderr=proc.stderr,
            llm_outputs=parse_llm_outputs(proc.stderr),
        )
