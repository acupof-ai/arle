#!/usr/bin/env python3
"""W&B sidecar for the Rust train metrics sink.

Reads JSONL messages on stdin:
- {"type":"metric","step":N,"phase":"train","fields":{...}}
- {"type":"event","kind":"run_start",...}
- {"type":"finish"}

The Rust foreground stays on a Rust-only hot path; this helper is a best-effort
background adapter around the official W&B SDK. Default mode is offline so the
run can be synced later with `wandb sync`.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

try:
    import wandb
except ImportError as exc:  # pragma: no cover - exercised by runtime use, not unit tests
    sys.stderr.write(
        "[train.metrics] wandb helper requires the `wandb` package. "
        "Install with `pip install -e .[observe]` or disable TRAIN_WANDB_PROJECT.\n"
    )
    raise SystemExit(3) from exc


def getenv(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def truthy(name: str, default: bool) -> bool:
    value = getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def parse_tags() -> list[str] | None:
    raw = getenv("TRAIN_WANDB_TAGS")
    if raw is None:
        return None
    tags = [part.strip() for part in raw.split(",") if part.strip()]
    return tags or None


def init_run() -> Any:
    kwargs: dict[str, Any] = {
        "project": os.environ["WANDB_PROJECT"],
        "mode": getenv("WANDB_MODE") or "offline",
        "entity": getenv("WANDB_ENTITY"),
        "name": getenv("WANDB_NAME"),
        "notes": getenv("WANDB_NOTES"),
        "group": getenv("WANDB_RUN_GROUP"),
        "job_type": getenv("WANDB_JOB_TYPE"),
        "id": getenv("WANDB_RUN_ID"),
        "resume": getenv("WANDB_RESUME"),
        "dir": getenv("WANDB_DIR"),
        "tags": parse_tags(),
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    return wandb.init(**kwargs)


def flatten_event(
    kind: str,
    strings: dict[str, str],
    scalars: dict[str, float],
    bools: dict[str, bool],
) -> dict[str, Any]:
    payload: dict[str, Any] = {f"event/{kind}": 1}
    for key, value in strings.items():
        payload[f"event/{kind}/{key}"] = value
    for key, value in scalars.items():
        payload[f"event/{kind}/{key}"] = value
    for key, value in bools.items():
        payload[f"event/{kind}/{key}"] = value
    return payload


def log_checkpoint_artifact(run: Any, step: int | None, strings: dict[str, str]) -> None:
    if not truthy("TRAIN_WANDB_LOG_CHECKPOINTS", True):
        return
    path = strings.get("path")
    if not path:
        return
    root = Path(path)
    if not root.exists():
        return
    artifact_name = f"{run.id}-checkpoint"
    if step is not None:
        artifact_name += f"-step-{step:06d}"
    metadata = {"checkpoint_path": str(root)}
    artifact = wandb.Artifact(artifact_name, type="model", metadata=metadata)
    added = False
    for key, filename in strings.items():
        if not key.startswith("artifact_"):
            continue
        candidate = root / filename
        if candidate.is_file():
            artifact.add_file(str(candidate), name=filename)
            added = True
    if not added and root.is_dir():
        for candidate in root.iterdir():
            if candidate.is_file():
                artifact.add_file(str(candidate), name=candidate.name)
                added = True
    if not added:
        return
    aliases = ["latest"]
    if step is not None:
        aliases.append(f"step-{step:06d}")
    run.log_artifact(artifact, aliases=aliases)


def handle_metric(run: Any, message: dict[str, Any]) -> None:
    step = message.get("step")
    phase = message.get("phase") or "train"
    fields = message.get("fields") or {}
    payload = {f"{phase}/{key}": value for key, value in fields.items()}
    payload["global_step"] = step
    run.log(payload, step=step)


def handle_event(run: Any, message: dict[str, Any]) -> bool:
    kind = message["kind"]
    step = message.get("step")
    strings = dict(message.get("strings") or {})
    scalars = dict(message.get("scalars") or {})
    bools = dict(message.get("bools") or {})

    if kind == "run_start":
        config_payload: dict[str, Any] = {}
        for key, value in strings.items():
            config_payload[key] = value
        for key, value in scalars.items():
            config_payload[key] = value
        for key, value in bools.items():
            config_payload[key] = value
        if config_payload:
            run.config.update(config_payload, allow_val_change=True)

    run.log(flatten_event(kind, strings, scalars, bools), step=step)

    if kind == "checkpoint":
        log_checkpoint_artifact(run, step, strings)

    if kind == "run_end":
        for key, value in strings.items():
            run.summary[key] = value
        for key, value in scalars.items():
            run.summary[key] = value
        for key, value in bools.items():
            run.summary[key] = value
        return False

    return True


def main() -> int:
    run = init_run()
    keep_running = True
    try:
        for raw in sys.stdin:
            raw = raw.strip()
            if not raw:
                continue
            message = json.loads(raw)
            msg_type = message.get("type")
            if msg_type == "metric":
                handle_metric(run, message)
            elif msg_type == "event":
                keep_running = handle_event(run, message)
                if not keep_running:
                    break
            elif msg_type == "finish":
                break
    finally:
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
