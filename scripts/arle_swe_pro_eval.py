#!/usr/bin/env python3
"""ARLE SWE-bench Pro runner scaffold.

This script keeps the result boundary explicit:

1. ARLE inference produces exactly one candidate artifact: a unified diff patch.
2. The harness may extract that diff from the raw text, but may not repair it.
3. The final verdict comes from the official SWE-bench Pro evaluator.

Typical flow:

    python scripts/arle_swe_pro_eval.py prepare --output bench-output/swepro-smoke --limit 3

    python scripts/arle_swe_pro_eval.py generate \
      --output bench-output/swepro-smoke \
      --base-url http://localhost:8123 \
      --model-id <served-model> \
      --limit 3

    python scripts/arle_swe_pro_eval.py evaluate \
      --output bench-output/swepro-smoke \
      --eval-repo /path/to/SWE-bench_Pro-os \
      --use-local-docker
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DATASET_NAME = "ScaleAI/SWE-bench_Pro"
DEFAULT_DATASET_REVISION = "main"
DEFAULT_SPLIT = "test"
SCHEMA_VERSION = "arle-swe-pro-eval-v1"

FORBIDDEN_MODEL_FIELDS = {"patch", "test_patch", "fail_to_pass", "pass_to_pass"}
MODEL_VISIBLE_FIELDS = (
    "repo",
    "instance_id",
    "base_commit",
    "problem_statement",
    "requirements",
    "interface",
    "repo_language",
)

GENERATION_CONTRACT = {
    "candidate_source": "ARLE inference output",
    "candidate_artifact": "unified diff patch",
    "forbidden_model_fields": sorted(FORBIDDEN_MODEL_FIELDS),
    "harness_may_repair_patch": False,
    "scoring_policy": "official SWE-bench Pro Docker evaluator",
}


class ArleChatClient:
    """Small OpenAI-compatible chat client for a local ARLE serve."""

    def __init__(
        self,
        base_url: str,
        model_id: str,
        api_key: str | None = None,
        timeout: float = 600.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.api_key = api_key
        self.timeout = timeout

    def chat(self, messages: list[dict[str, str]], max_tokens: int, temperature: float) -> str:
        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            _chat_completions_url(self.base_url),
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read())
        return body["choices"][0]["message"].get("content") or ""


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _fingerprint_rows(rows: list[dict[str, Any]]) -> str:
    payload = [
        {
            "repo": row.get("repo"),
            "instance_id": row.get("instance_id"),
            "base_commit": row.get("base_commit"),
            "problem_statement": row.get("problem_statement"),
        }
        for row in rows
    ]
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _safe_prefix(value: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return prefix or "arle"


def load_dataset_rows(split: str, revision: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("SWE Pro eval requires `datasets`; install with `pip install datasets`.") from exc
    ds = load_dataset(DATASET_NAME, split=split, revision=revision)
    return [dict(row) for row in ds]


def select_rows(
    rows: list[dict[str, Any]],
    *,
    limit: int | None = None,
    instance_ids: list[str] | None = None,
    repo: str | None = None,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    selected = list(rows)
    if instance_ids:
        wanted = set(instance_ids)
        selected = [row for row in selected if row.get("instance_id") in wanted]
        missing = sorted(wanted - {str(row.get("instance_id")) for row in selected})
        if missing:
            raise ValueError(f"unknown instance ids: {missing}")
    if repo:
        selected = [row for row in selected if row.get("repo") == repo]
    if seed is not None:
        random.Random(f"swe-pro-{seed}").shuffle(selected)
    if limit is not None:
        selected = selected[:limit]
    return selected


def sanitize_instance(row: dict[str, Any]) -> dict[str, Any]:
    sanitized = {field: row.get(field) for field in MODEL_VISIBLE_FIELDS if row.get(field) not in (None, "")}
    leaked = FORBIDDEN_MODEL_FIELDS.intersection(sanitized)
    if leaked:
        raise ValueError(f"sanitized instance leaked forbidden fields: {sorted(leaked)}")
    return sanitized


def build_patch_prompt(row: dict[str, Any]) -> list[dict[str, str]]:
    inst = sanitize_instance(row)
    sections = [
        f"Repository: {inst.get('repo', '')}",
        f"Base commit: {inst.get('base_commit', '')}",
        f"Language: {inst.get('repo_language', '')}",
        "",
        "Problem statement:",
        str(inst.get("problem_statement", "")).strip(),
    ]
    if inst.get("requirements"):
        sections.extend(["", "Project requirements:", str(inst["requirements"]).strip()])
    if inst.get("interface"):
        sections.extend(["", "Relevant interface notes:", str(inst["interface"]).strip()])

    user = "\n".join(sections).strip()
    return [
        {
            "role": "system",
            "content": (
                "You are a software engineering agent. Produce a minimal unified diff patch "
                "against the repository at the specified base commit. Return only the patch. "
                "Do not include markdown fences, explanations, tests, or analysis."
            ),
        },
        {"role": "user", "content": user},
    ]


def extract_unified_diff(text: str) -> str:
    stripped = text.strip()
    fenced = _extract_fenced_diff(stripped)
    if fenced is not None:
        stripped = fenced.strip()
    marker_positions = [pos for pos in (stripped.find("diff --git "), stripped.find("--- ")) if pos >= 0]
    if marker_positions:
        stripped = stripped[min(marker_positions):].strip()
    return stripped


def _extract_fenced_diff(text: str) -> str | None:
    fence_start = text.find("```")
    if fence_start < 0:
        return None
    content_start = text.find("\n", fence_start)
    if content_start < 0:
        return None
    fence_end = text.find("```", content_start + 1)
    if fence_end < 0:
        return None
    return text[content_start + 1:fence_end]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def write_raw_samples_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key)) for key in fieldnames})


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def prepare_output_dir(output: Path, rows: list[dict[str, Any]], *, split: str, revision: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    sanitized = [sanitize_instance(row) for row in rows]
    write_jsonl(output / "instances.jsonl", sanitized)
    write_raw_samples_csv(output / "raw_samples.csv", rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": {
            "name": DATASET_NAME,
            "split": split,
            "revision": revision,
            "count": len(rows),
            "sample_fingerprint": _fingerprint_rows(rows),
        },
        "generation_contract": GENERATION_CONTRACT,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def generate_patches(
    rows: list[dict[str, Any]],
    client: ArleChatClient,
    *,
    max_tokens: int,
    temperature: float,
    output: Path,
) -> None:
    raw_records: list[dict[str, Any]] = []
    patches: list[dict[str, str]] = []
    prefix = _safe_prefix(client.model_id)
    for index, row in enumerate(rows):
        instance_id = str(row["instance_id"])
        print(f"[swe-pro] generate {index + 1}/{len(rows)} {instance_id}", flush=True)
        messages = build_patch_prompt(row)
        try:
            raw_text = client.chat(messages, max_tokens=max_tokens, temperature=temperature)
            error = None
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, KeyError, IndexError) as exc:
            raw_text = ""
            error = f"{type(exc).__name__}: {exc}"
        patch = extract_unified_diff(raw_text) if raw_text else ""
        raw_records.append({
            "instance_id": instance_id,
            "repo": row.get("repo"),
            "model_id": client.model_id,
            "prefix": prefix,
            "messages": messages,
            "raw_output": raw_text,
            "extracted_patch": patch,
            "error": error,
        })
        patches.append({"instance_id": instance_id, "patch": patch, "prefix": prefix})

    write_jsonl(output / "raw_outputs.jsonl", raw_records)
    (output / "patches.json").write_text(json.dumps(patches, indent=2), encoding="utf-8")


def build_official_eval_command(
    *,
    python: str,
    eval_repo: Path,
    output: Path,
    num_workers: int,
    dockerhub_username: str,
    use_local_docker: bool,
    docker_platform: str | None = None,
    block_network: bool = False,
    redo: bool = False,
) -> list[str]:
    eval_repo = eval_repo.resolve()
    output = output.resolve()
    cmd = [
        python,
        str(eval_repo / "swe_bench_pro_eval.py"),
        f"--raw_sample_path={output / 'raw_samples.csv'}",
        f"--patch_path={output / 'patches.json'}",
        f"--output_dir={output / 'official_eval'}",
        f"--scripts_dir={eval_repo / 'run_scripts'}",
        f"--num_workers={num_workers}",
        f"--dockerhub_username={dockerhub_username}",
    ]
    if use_local_docker:
        cmd.append("--use_local_docker")
    if docker_platform:
        cmd.append(f"--docker_platform={docker_platform}")
    if block_network:
        cmd.append("--block_network")
    if redo:
        cmd.append("--redo")
    return cmd


def _load_selected_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = load_dataset_rows(args.split, args.dataset_revision)
    return select_rows(
        rows,
        limit=args.limit,
        instance_ids=args.instance_id,
        repo=args.repo,
        seed=args.seed,
    )


def _add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--dataset-revision", default=os.environ.get("ARLE_SWE_PRO_REVISION", DEFAULT_DATASET_REVISION))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--instance-id", action="append", default=None)
    parser.add_argument("--repo", default=None)
    parser.add_argument("--seed", type=int, default=None)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="materialize selected SWE Pro instances and manifest")
    _add_dataset_args(prepare)

    generate = sub.add_parser("generate", help="generate ARLE patches for selected instances")
    _add_dataset_args(generate)
    generate.add_argument("--base-url", default=os.environ.get("ARLE_BASE_URL", "http://localhost:8123"))
    generate.add_argument("--model-id", required=True)
    generate.add_argument("--api-key", default=os.environ.get("ARLE_API_KEY"))
    generate.add_argument("--max-tokens", type=int, default=4096)
    generate.add_argument("--temperature", type=float, default=0.0)
    generate.add_argument("--timeout", type=float, default=600.0)

    evaluate = sub.add_parser("evaluate", help="run the official SWE Pro evaluator on patches.json")
    evaluate.add_argument("--output", type=Path, required=True)
    eval_repo_default = Path(os.environ["SWE_PRO_EVAL_REPO"]) if os.environ.get("SWE_PRO_EVAL_REPO") else None
    evaluate.add_argument("--eval-repo", type=Path, default=eval_repo_default)
    evaluate.add_argument("--python", default=sys.executable)
    evaluate.add_argument("--num-workers", type=int, default=4)
    evaluate.add_argument("--dockerhub-username", default=os.environ.get("SWE_PRO_DOCKERHUB_USERNAME", "jefzda"))
    evaluate.add_argument("--use-local-docker", action="store_true")
    evaluate.add_argument("--docker-platform", default=None,
                          help="official evaluator platform override, e.g. linux/amd64")
    evaluate.add_argument("--block-network", action="store_true")
    evaluate.add_argument("--redo", action="store_true")
    evaluate.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "prepare":
        rows = _load_selected_rows(args)
        prepare_output_dir(args.output, rows, split=args.split, revision=args.dataset_revision)
        print(f"[swe-pro] prepared {len(rows)} instances under {args.output}", flush=True)
        return 0

    if args.command == "generate":
        rows = _load_selected_rows(args)
        prepare_output_dir(args.output, rows, split=args.split, revision=args.dataset_revision)
        client = ArleChatClient(
            base_url=args.base_url,
            model_id=args.model_id,
            api_key=args.api_key,
            timeout=args.timeout,
        )
        generate_patches(rows, client, max_tokens=args.max_tokens, temperature=args.temperature, output=args.output)
        print(f"[swe-pro] wrote {args.output / 'patches.json'}", flush=True)
        return 0

    if args.command == "evaluate":
        if args.eval_repo is None:
            print("--eval-repo or SWE_PRO_EVAL_REPO is required", file=sys.stderr)
            return 2
        if not (args.output / "raw_samples.csv").exists():
            print(f"missing {args.output / 'raw_samples.csv'}; run prepare or generate first", file=sys.stderr)
            return 2
        if not (args.output / "patches.json").exists():
            print(f"missing {args.output / 'patches.json'}; run generate first", file=sys.stderr)
            return 2
        cmd = build_official_eval_command(
            python=args.python,
            eval_repo=args.eval_repo,
            output=args.output,
            num_workers=args.num_workers,
            dockerhub_username=args.dockerhub_username,
            use_local_docker=args.use_local_docker,
            docker_platform=args.docker_platform,
            block_network=args.block_network,
            redo=args.redo,
        )
        print("[swe-pro] " + " ".join(cmd), flush=True)
        if args.dry_run:
            return 0
        completed = subprocess.run(cmd, check=False, cwd=args.eval_repo)
        return completed.returncode

    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
