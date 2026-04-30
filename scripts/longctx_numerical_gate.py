#!/usr/bin/env python3
"""HTTP numerical gate for long-context Phase 1 S3.

The script intentionally does not launch servers. Start the ARLE/SGLang
targets with the desired KV dtype first, then compare their greedy completion
trajectories through the OpenAI-compatible /v1/completions API.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
from tokenizers import Tokenizer


DEFAULT_PROMPTS = [
    "Explain why prefix caches help long-context inference in practical serving systems.",
    "Write a concise design review for a CUDA scheduler that mixes prefill and decode.",
    "Summarize the tradeoffs between fp8 KV cache and bf16 KV cache for Qwen models.",
    "Derive a checklist for validating numerical parity after changing an attention kernel.",
    "Describe how a production inference server should expose queue and KV utilization metrics.",
    "Give a short example of a failure caused by reusing stale KV cache entries.",
    "Compare recomputation-based preemption with swap-based preemption for LLM serving.",
    "Write a test plan for a benchmark harness that compares ARLE and SGLang.",
    "Explain why long-context prefill can dominate TTFT even when decode is optimized.",
    "Draft a postmortem for a regression where fp8 outputs diverge at generated token zero.",
    "List the invariants a paged KV pool must preserve during mixed batch execution.",
    "Explain how deterministic greedy decoding helps isolate numerical drift.",
    "Write a concise operator note for running a 32k prompt throughput benchmark.",
    "Describe a staged rollout plan for making fp8 KV the default in a runtime.",
    "Explain why commit-pinned baselines matter for public performance claims.",
    "Write a compact summary of MagicDec and why long context changes speculation economics.",
    "Explain how to interpret a divergence_p50 metric in token trajectory comparisons.",
    "Describe why Tensor Core prefill kernels matter for long-context throughput.",
    "Write a short plan for validating StepPlan::Mixed on Qwen3-4B at 32k context.",
    "Explain how synthetic prompt token clamps avoid benchmark distribution drift.",
    "Create a risk register for a mission to lead open-source long-context throughput.",
    "Summarize how service-side traces complement client-side GuideLLM metrics.",
    "Explain why a benchmark that emits empty outputs should not create a wins entry.",
    "Describe how to compare two greedy model outputs when exact text differs.",
    "Write an incident rule for avoiding misleading reused-server baseline artifacts.",
    "Explain why first-token-only smoke tests miss durable KV readback bugs.",
    "Draft a concise project update about finishing S1 and S2 of a long-context plan.",
    "Describe the minimum evidence needed before claiming world-class throughput.",
    "Explain how a single shared scratch buffer can be reused across attention layers.",
    "Write a short note on why sparse attention should stay feature-gated by default.",
    "Compare long-decode and max-throughput workloads in serving benchmarks.",
    "Explain how a remote pending benchmark entry should record missing hardware access.",
]


@dataclass
class Completion:
    text: str
    token_ids: list[int] | None
    prompt_tokens: int | None
    completion_tokens: int | None
    elapsed_ms: float


@dataclass
class CaseResult:
    index: int
    prompt_tokens: int
    left_prompt_tokens_reported: int | None
    right_prompt_tokens_reported: int | None
    left_text: str
    right_text: str | None
    left_tokens: int
    right_tokens: int | None
    exact: bool | None
    common_prefix_tokens: int | None
    common_token_match_rate: float | None
    earliest_divergence: int | None
    left_elapsed_ms: float
    right_elapsed_ms: float | None
    ok: bool
    error: str | None = None


def resolve_tokenizer(path: str) -> Tokenizer:
    tokenizer_path = Path(path)
    if tokenizer_path.is_dir():
        tokenizer_path = tokenizer_path / "tokenizer.json"
    if not tokenizer_path.exists():
        raise SystemExit(f"tokenizer not found: {tokenizer_path}")
    return Tokenizer.from_file(str(tokenizer_path))


def encode(tokenizer: Tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False).ids


def stable_repeat_piece(tokenizer: Tokenizer, prompt_tokens: int) -> str | None:
    sample_count = min(8, max(1, prompt_tokens))
    candidates = [
        " token",
        " alpha",
        " context",
        " inference",
        " 0",
        "\n",
        ".",
    ]
    try:
        vocab = sorted(tokenizer.get_vocab().items(), key=lambda item: item[1])
    except Exception:  # noqa: BLE001 - tokenizer implementations vary.
        vocab = []
    for token, token_id in vocab[:2048]:
        if not token or token.startswith("["):
            continue
        try:
            decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        except Exception:  # noqa: BLE001 - skip malformed/special vocab entries.
            continue
        if decoded and decoded.strip():
            candidates.extend([decoded, f" {decoded}", f"{decoded} ", f"{decoded}\n"])

    seen = set()
    for piece in candidates:
        if piece in seen or not piece.strip():
            continue
        seen.add(piece)
        if len(encode(tokenizer, piece)) != 1:
            continue
        if len(encode(tokenizer, piece * sample_count)) == sample_count:
            return piece
    return None


def build_prompts(tokenizer: Tokenizer, count: int, prompt_tokens: int | None) -> list[str]:
    if count < 1:
        raise SystemExit("--prompt-count must be >= 1")
    seeds = [DEFAULT_PROMPTS[i % len(DEFAULT_PROMPTS)] for i in range(count)]
    if prompt_tokens is None:
        return seeds

    prompts = []
    repeat_piece = stable_repeat_piece(tokenizer, prompt_tokens)
    for idx, seed in enumerate(seeds):
        if repeat_piece is not None:
            text = repeat_piece * prompt_tokens
        else:
            scaffold = (
                f"{seed}\n\n"
                "Use this repeated technical context as stable synthetic input. "
                "The answer should remain concise and deterministic.\n"
            )
            seed_ids = encode(tokenizer, scaffold)
            if not seed_ids:
                raise SystemExit("tokenizer produced no ids for synthetic prompt seed")
            repeated = [seed_ids[i % len(seed_ids)] for i in range(prompt_tokens)]
            text = tokenizer.decode(repeated, skip_special_tokens=False)
        actual_tokens = len(encode(tokenizer, text))
        if actual_tokens != prompt_tokens:
            raise SystemExit(
                f"synthetic prompt {idx} re-tokenized to {actual_tokens} tokens, "
                f"expected --prompt-tokens={prompt_tokens}"
            )
        prompts.append(text)
    return prompts


def parse_token_ids(value: Any, field: str) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise RuntimeError(f"{field} was present but not a list")
    try:
        return [int(tok) for tok in value]
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{field} contained a non-integer token id") from exc


def parse_sglang_logprob_token_ids(value: Any, field: str) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise RuntimeError(f"{field} was present but not a list")
    token_ids: list[int] = []
    for idx, item in enumerate(value):
        token_id: Any = None
        if isinstance(item, dict):
            token_id = item.get("token_id", item.get("id"))
        elif isinstance(item, list):
            if len(item) > 1 and isinstance(item[1], int) and not isinstance(item[1], bool):
                token_id = item[1]
            elif item and isinstance(item[0], int) and not isinstance(item[0], bool):
                token_id = item[0]
        elif isinstance(item, int) and not isinstance(item, bool):
            token_id = item

        if token_id is None:
            raise RuntimeError(f"{field}[{idx}] did not contain a token id")
        token_ids.append(int(token_id))
    return token_ids


def extract_response_token_ids(
    data: dict[str, Any],
    choice: dict[str, Any],
) -> list[int] | None:
    for field, value in [
        ("choices[0].token_ids", choice.get("token_ids")),
        ("choices[0].response_token_ids", choice.get("response_token_ids")),
        ("response_token_ids", data.get("response_token_ids")),
        ("output_token_ids", data.get("output_token_ids")),
    ]:
        token_ids = parse_token_ids(value, field)
        if token_ids is not None:
            return token_ids
    for owner, meta_info in [
        ("choices[0].meta_info", choice.get("meta_info")),
        ("meta_info", data.get("meta_info")),
    ]:
        if not isinstance(meta_info, dict):
            continue
        token_ids = parse_token_ids(meta_info.get("output_token_ids"), f"{owner}.output_token_ids")
        if token_ids is not None:
            return token_ids
        token_ids = parse_sglang_logprob_token_ids(
            meta_info.get("output_token_logprobs"),
            f"{owner}.output_token_logprobs",
        )
        if token_ids is not None:
            return token_ids
    return None


RESERVED_PAYLOAD_KEYS = {
    "model",
    "prompt",
    "max_tokens",
    "temperature",
    "seed",
    "stream",
    "ignore_eos",
}


def parse_extra_json(raw: str, flag: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{flag} must be a JSON object: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{flag} must be a JSON object")
    reserved = sorted(RESERVED_PAYLOAD_KEYS.intersection(value))
    if reserved:
        raise SystemExit(f"{flag} may not override core request fields: {', '.join(reserved)}")
    return value


def complete(
    client: httpx.Client,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    seed: int,
    ignore_eos: bool,
    require_token_ids: bool,
    extra_json: dict[str, Any],
) -> Completion:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": seed,
        "stream": False,
    }
    if ignore_eos:
        payload["ignore_eos"] = True
    payload.update(extra_json)

    start = time.perf_counter()
    resp = client.post(f"{url.rstrip('/')}/v1/completions", json=payload)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("response had no choices")
    choice = choices[0]
    text = choice.get("text") or ""
    usage = data.get("usage") or {}
    token_ids = extract_response_token_ids(data, choice)
    completion_tokens = usage.get("completion_tokens")
    if completion_tokens is not None:
        completion_tokens = int(completion_tokens)
    if require_token_ids:
        if token_ids is None:
            raise RuntimeError(
                "response did not expose generated token ids; expected one of "
                "choices[0].token_ids, choices[0].response_token_ids, "
                "response_token_ids, output_token_ids, or SGLang meta_info token IDs"
            )
        if completion_tokens is None:
            raise RuntimeError(
                "response exposed generated token ids but did not report "
                "usage.completion_tokens; trajectory completeness is unverifiable"
            )
        if len(token_ids) != completion_tokens:
            raise RuntimeError(
                f"response exposed {len(token_ids)} generated token ids while "
                f"usage.completion_tokens was {completion_tokens}; incomplete "
                "token trajectories are unavailable"
            )
    return Completion(
        text=text,
        token_ids=token_ids,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=completion_tokens,
        elapsed_ms=elapsed_ms,
    )


def common_prefix(left: list[int], right: list[int]) -> int:
    total = min(len(left), len(right))
    for idx in range(total):
        if left[idx] != right[idx]:
            return idx
    return total


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def classify(
    avg_match: float,
    divergence_p50: float | None,
    pass_threshold: float,
    degraded_threshold: float,
    min_divergence_p50: int,
) -> str:
    if (
        avg_match >= pass_threshold
        and divergence_p50 is not None
        and divergence_p50 >= min_divergence_p50
    ):
        return "pass"
    if avg_match >= degraded_threshold:
        return "degraded"
    return "stop"


def token_count(completion: Completion) -> int:
    if completion.token_ids is not None:
        return len(completion.token_ids)
    if completion.completion_tokens is not None:
        return int(completion.completion_tokens)
    return 0


def prompt_usage_error(
    name: str,
    reported: int | None,
    expected: int,
    tolerance: int,
) -> str | None:
    if reported is None:
        return f"{name} response did not report usage.prompt_tokens"
    delta = abs(int(reported) - expected)
    if delta > tolerance:
        return (
            f"{name} usage.prompt_tokens={reported} did not match local prompt_tokens={expected} "
            f"within tolerance={tolerance}"
        )
    return None


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    summary = payload["summary"]
    lines = [
        f"# Longctx Numerical Gate - {payload['config']['label']}",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in [
        "mode",
        "status",
        "pairs",
        "ok_pairs",
        "failed_pairs",
        "exact_pairs",
        "exact_pair_rate",
        "avg_common_token_match",
        "earliest_divergence_min",
        "divergence_p50",
        "nonempty_rate",
    ]:
        if key in summary:
            lines.append(f"| {key} | {summary[key]} |")
    lines.extend(
        [
            "",
            "## Config",
            "",
            "```json",
            json.dumps(payload["config"], indent=2, sort_keys=True),
            "```",
            "",
            "## Cases",
            "",
            "| idx | prompt tok | left reported | right reported | exact | match | divergence | left tok | right tok | ok |",
            "|---:|---:|---:|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    for case in payload["cases"]:
        lines.append(
            "| {index} | {prompt_tokens} | {left_reported} | {right_reported} | {exact} | {match} | {div} | {left} | {right} | {ok} |".format(
                index=case["index"],
                prompt_tokens=case["prompt_tokens"],
                left_reported=(
                    "n/a"
                    if case["left_prompt_tokens_reported"] is None
                    else case["left_prompt_tokens_reported"]
                ),
                right_reported=(
                    "n/a"
                    if case["right_prompt_tokens_reported"] is None
                    else case["right_prompt_tokens_reported"]
                ),
                exact=case["exact"],
                match="n/a" if case["common_token_match_rate"] is None else f"{case['common_token_match_rate']:.4f}",
                div="n/a" if case["earliest_divergence"] is None else case["earliest_divergence"],
                left=case["left_tokens"],
                right="n/a" if case["right_tokens"] is None else case["right_tokens"],
                ok=case["ok"],
            )
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--left-url", required=True)
    parser.add_argument("--right-url")
    parser.add_argument("--left-name", default="left")
    parser.add_argument("--right-name", default="right")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--tokenizer", default="infer/models/Qwen3-4B")
    parser.add_argument("--prompt-count", type=int, default=16)
    parser.add_argument("--prompt-tokens", type=int)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--out-dir", default="bench-output/longctx-numerical-gate")
    parser.add_argument("--pass-threshold", type=float, default=0.70)
    parser.add_argument("--degraded-threshold", type=float, default=0.60)
    parser.add_argument("--min-divergence-p50", type=int, default=30)
    parser.add_argument("--prompt-token-tolerance", type=int, default=0)
    parser.add_argument("--left-extra-json", default="{}")
    parser.add_argument("--right-extra-json", default="{}")
    parser.add_argument("--fail-on", choices=["stop", "degraded"], default="stop")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tokenizer = resolve_tokenizer(args.tokenizer)
    prompts = build_prompts(tokenizer, args.prompt_count, args.prompt_tokens)
    left_extra_json = parse_extra_json(args.left_extra_json, "--left-extra-json")
    right_extra_json = parse_extra_json(args.right_extra_json, "--right-extra-json")
    if args.right_url and not args.ignore_eos:
        raise SystemExit(
            "compare mode requires --ignore-eos so fixed max-token trajectories "
            "are scored instead of early-EOS lengths"
        )
    timeout = httpx.Timeout(args.timeout, connect=30.0)
    cases: list[CaseResult] = []

    with httpx.Client(timeout=timeout) as client:
        for idx, prompt in enumerate(prompts):
            prompt_len = len(encode(tokenizer, prompt))
            try:
                left = complete(
                    client,
                    args.left_url,
                    args.model,
                        prompt,
                        args.max_tokens,
                        args.seed,
                        args.ignore_eos,
                        require_token_ids=bool(args.right_url),
                        extra_json=left_extra_json,
                    )
                if args.right_url:
                    right = complete(
                        client,
                        args.right_url,
                        args.model,
                        prompt,
                        args.max_tokens,
                        args.seed,
                        args.ignore_eos,
                        require_token_ids=True,
                        extra_json=right_extra_json,
                    )
                    if left.token_ids is None or right.token_ids is None:
                        raise RuntimeError("internal error: compare mode reached without token ids")
                    prefix = common_prefix(left.token_ids, right.token_ids)
                    denom = max(1, args.max_tokens, len(left.token_ids), len(right.token_ids))
                    exact = left.token_ids == right.token_ids
                    errors = [
                            err
                            for err in [
                                prompt_usage_error(
                                    args.left_name,
                                    left.prompt_tokens,
                                    prompt_len,
                                    args.prompt_token_tolerance,
                                ),
                                prompt_usage_error(
                                    args.right_name,
                                    right.prompt_tokens,
                                    prompt_len,
                                    args.prompt_token_tolerance,
                                ),
                            ]
                            if err is not None
                        ]
                    pair_ok = not errors
                    cases.append(
                        CaseResult(
                            index=idx,
                            prompt_tokens=prompt_len,
                            left_prompt_tokens_reported=left.prompt_tokens,
                            right_prompt_tokens_reported=right.prompt_tokens,
                            left_text=left.text,
                            right_text=right.text,
                            left_tokens=len(left.token_ids),
                            right_tokens=len(right.token_ids),
                            exact=exact,
                            common_prefix_tokens=prefix,
                            common_token_match_rate=prefix / denom,
                            earliest_divergence=prefix,
                            left_elapsed_ms=left.elapsed_ms,
                            right_elapsed_ms=right.elapsed_ms,
                            ok=pair_ok,
                            error="; ".join(errors) if errors else None,
                        )
                    )
                else:
                    first5 = left.text[:5]
                    non_degenerate = bool(left.text) and not (len(first5) == 5 and len(set(first5)) == 1)
                    prompt_error = prompt_usage_error(
                        args.left_name,
                        left.prompt_tokens,
                        prompt_len,
                        args.prompt_token_tolerance,
                    )
                    ok = non_degenerate and prompt_error is None
                    errors = []
                    if not non_degenerate:
                        errors.append("empty or degenerate output")
                    if prompt_error is not None:
                        errors.append(prompt_error)
                    cases.append(
                        CaseResult(
                            index=idx,
                            prompt_tokens=prompt_len,
                            left_prompt_tokens_reported=left.prompt_tokens,
                            right_prompt_tokens_reported=None,
                            left_text=left.text,
                            right_text=None,
                            left_tokens=token_count(left),
                            right_tokens=None,
                            exact=None,
                            common_prefix_tokens=None,
                            common_token_match_rate=None,
                            earliest_divergence=None,
                            left_elapsed_ms=left.elapsed_ms,
                            right_elapsed_ms=None,
                            ok=ok,
                            error="; ".join(errors) if errors else None,
                        )
                    )
            except Exception as exc:  # noqa: BLE001 - report all remote failures in JSON.
                cases.append(
                    CaseResult(
                        index=idx,
                        prompt_tokens=prompt_len,
                        left_prompt_tokens_reported=None,
                        right_prompt_tokens_reported=None,
                        left_text="",
                        right_text=None,
                        left_tokens=0,
                        right_tokens=None,
                        exact=None,
                        common_prefix_tokens=None,
                        common_token_match_rate=None,
                        earliest_divergence=None,
                        left_elapsed_ms=0.0,
                        right_elapsed_ms=None,
                        ok=False,
                        error=str(exc),
                    )
                )

    case_dicts = [asdict(case) for case in cases]
    ok_cases = [case for case in cases if case.ok]
    failed_cases = len(cases) - len(ok_cases)
    mode = "compare" if args.right_url else "single"
    if args.right_url:
        match_rates = [
            case.common_token_match_rate if case.ok and case.common_token_match_rate is not None else 0.0
            for case in cases
        ]
        divergences = [
            float(case.earliest_divergence) if case.ok and case.earliest_divergence is not None else 0.0
            for case in cases
        ]
        avg_match = statistics.fmean(match_rates) if match_rates else 0.0
        divergence_p50 = percentile(divergences, 0.50)
        status = (
            "stop"
            if failed_cases
            else classify(
                avg_match,
                divergence_p50,
                args.pass_threshold,
                args.degraded_threshold,
                args.min_divergence_p50,
            )
        )
        summary: dict[str, Any] = {
            "mode": mode,
            "status": status,
            "pairs": len(cases),
            "ok_pairs": len(ok_cases),
            "failed_pairs": failed_cases,
            "exact_pairs": sum(1 for case in ok_cases if case.exact),
            "exact_pair_rate": round(sum(1 for case in ok_cases if case.exact) / max(1, len(cases)), 4),
            "avg_common_token_match": round(avg_match, 4),
            "earliest_divergence_min": min(divergences) if divergences else None,
            "divergence_p50": divergence_p50,
        }
    else:
        nonempty_rate = sum(1 for case in ok_cases if case.left_tokens > 0) / max(1, len(cases))
        status = "pass" if len(ok_cases) == len(cases) else "stop"
        summary = {
            "mode": mode,
            "status": status,
            "pairs": len(cases),
            "ok_pairs": len(ok_cases),
            "failed_pairs": failed_cases,
            "nonempty_rate": round(nonempty_rate, 4),
        }

    payload = {
        "config": {
            "label": args.label,
            "left_name": args.left_name,
            "left_url": args.left_url,
            "right_name": args.right_name if args.right_url else None,
            "right_url": args.right_url,
            "model": args.model,
            "prompt_count": args.prompt_count,
            "prompt_tokens": args.prompt_tokens,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "ignore_eos": args.ignore_eos,
            "pass_threshold": args.pass_threshold,
            "degraded_threshold": args.degraded_threshold,
            "min_divergence_p50": args.min_divergence_p50,
            "prompt_token_tolerance": args.prompt_token_tolerance,
            "left_extra_json": left_extra_json,
            "right_extra_json": right_extra_json,
        },
        "summary": summary,
        "cases": case_dicts,
    }
    out_dir = Path(args.out_dir)
    write_outputs(out_dir, payload)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {out_dir / 'summary.json'}")
    print(f"wrote {out_dir / 'summary.md'}")

    if status == "stop":
        return 1
    if status == "degraded" and args.fail_on == "degraded":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
