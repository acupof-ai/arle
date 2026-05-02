#!/usr/bin/env python3
"""Multi-turn agent trace replayer and cross-session prefix-hit benchmark.

Replays a JSONL trace of agent sessions against an OpenAI-compatible
`/v1/chat/completions` endpoint, threading `session_id` through every
request so the server can route turns of the same conversation to the
same slot / radix subtree. Measures per-turn TTFT, inter-token latency,
wall time, and tokens generated. Multiple sessions run concurrently.

This is the P1-and-beyond scoreboard for the Tiered KV Cache project:
the exit gate for P1 is ">= 70% cross-session prefix hit rate" and this
script provides the observable trace that measures it. See
`docs/plans/tiered-kv-cache-tasks.md` §2 for context.

NOTE: the existing `scripts/bench_agent.py` drives the Rust *binary* via
stdin and greps logs — a different workload shape. This replayer is
HTTP + async + trace-driven and lives under a different name to avoid
overwriting that script.

Usage:
  # Against the infer HTTP server, default trace, default label:
  python3 scripts/bench_agent_trace.py --server http://localhost:8000

  # Against an sglang server with a custom label and JSON snapshot:
  python3 scripts/bench_agent_trace.py \\
      --server http://localhost:30000 \\
      --label sglang-main \\
      --out docs/experience/wins/2026-04-13-bench-agent-trace.json

  # Higher concurrency (drives more cross-session contention):
  python3 scripts/bench_agent_trace.py --num-concurrent 8

Trace format (JSONL, one session per line):
  {"session_id": "agent-001",
   "system_prompt": "You are ...",
   "turns": [
     {"role": "user", "content": "..."},
     {"role": "assistant", "content": "..."},
     {"role": "user", "content": "..."}
   ]}

Each session's final turn MUST have role=user — the replayer stops
generation after that user message and records the model's completion.
Turns before the final user message are fed as conversation history.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import httpx
except ImportError:
    sys.exit("pip install httpx>=0.27 (see pyproject.toml [bench] extra)")


DEFAULT_TRACE = Path(__file__).parent / "data" / "agent_trace_default.jsonl"
WORKLOAD_TRACE = "trace"
WORKLOAD_W3 = "agent-w3-short-multiturn"
WORKLOAD_W4 = "agent-w4-tool-resume"
W3_SEED = 20260502
W3_WARM_SESSIONS = 64
W3_COLD_SESSIONS = 64
W3_SCORED_WARM_TURNS_PER_SESSION = 4
W3_CONCURRENCY = 16
W3_MAX_TOKENS = 64
W3_BASE_PROMPT_TOKENS = 1024
W3_BASE_PROMPT_JITTER = 32
W3_USER_TAIL_TOKENS = 64
W3_USER_TAIL_JITTER = 8
W4_SEED = 20260502
W4_SESSIONS = 128
W4_CONCURRENCY = 8
W4_WARMUP_MAX_TOKENS = 64
W4_RESUME_MAX_TOKENS = 256
W4_BASE_PROMPT_TOKENS = 8192
W4_BASE_PROMPT_JITTER = 64
W4_TOOL_OUTPUT_TOKENS = 256
W4_TOOL_OUTPUT_JITTER = 16

_TOKENISH_WORDS = (
    "agent",
    "plan",
    "trace",
    "state",
    "cache",
    "slot",
    "query",
    "reply",
    "task",
    "step",
    "tool",
    "result",
    "memory",
    "record",
    "window",
    "prefix",
)


# ─────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────


@dataclass
class Session:
    session_id: str
    system_prompt: str
    turns: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> "Session":
        session_id = str(obj["session_id"])
        system_prompt = str(obj.get("system_prompt", ""))
        turns = list(obj["turns"])
        if not turns:
            raise ValueError(f"session {session_id!r} has no turns")
        for i, turn in enumerate(turns):
            if turn.get("role") not in ("user", "assistant", "tool"):
                raise ValueError(
                    f"session {session_id!r} turn {i} has invalid role: {turn.get('role')!r}"
                )
            if not isinstance(turn.get("content"), str):
                raise ValueError(
                    f"session {session_id!r} turn {i} content must be a string"
                )
        metadata = obj.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError(f"session {session_id!r} metadata must be an object")
        return cls(
            session_id=session_id,
            system_prompt=system_prompt,
            turns=turns,
            metadata=metadata,
        )


@dataclass
class TurnResult:
    session_id: str
    turn_idx: int
    prompt_messages: int
    wall_ms: float
    ttft_ms: float | None
    itl_ms: float | None
    tokens_out: int
    finish_reason: str | None
    turn_kind: str = "trace"
    scored: bool = True
    expected_prompt_tokens: int | None = None
    request_max_tokens: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_idx": self.turn_idx,
            "turn_kind": self.turn_kind,
            "scored": self.scored,
            "prompt_messages": self.prompt_messages,
            "expected_prompt_tokens": self.expected_prompt_tokens,
            "request_max_tokens": self.request_max_tokens,
            "wall_ms": round(self.wall_ms, 2),
            "ttft_ms": round(self.ttft_ms, 2) if self.ttft_ms is not None else None,
            "itl_ms": round(self.itl_ms, 2) if self.itl_ms is not None else None,
            "tokens_out": self.tokens_out,
            "finish_reason": self.finish_reason,
            "error": self.error,
        }


@dataclass
class ServerStats:
    """Parsed snapshot of the server's `/v1/stats` plain-text response.

    The infer server (see `infer/src/http_server.rs:393-409`) emits a
    single-line key=value blob like::

        requests=0 active=0 waiting=0 tokens_out=0 kv_util=0.0% \
            ttft_p50=— ttft_p99=— tpot_p50=—

    All fields are optional on the client side because different
    servers may expose different subsets. Unknown keys are dropped.
    """

    raw: str
    fields: dict[str, str] = field(default_factory=dict)

    @classmethod
    def parse(cls, text: str) -> "ServerStats":
        fields: dict[str, str] = {}
        for tok in text.split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                fields[k] = v
        return cls(raw=text.strip(), fields=fields)

    def int_field(self, name: str) -> int | None:
        v = self.fields.get(name)
        if v is None:
            return None
        try:
            return int(v)
        except ValueError:
            return None

    def to_dict(self) -> dict[str, Any]:
        return {"raw": self.raw, "fields": self.fields}


@dataclass
class RunStats:
    label: str
    server: str
    trace_path: str
    num_concurrent: int
    workload: str
    timestamp: str
    turns: list[TurnResult] = field(default_factory=list)
    # Optional server-side probes taken before/after the run. `None`
    # means the server did not expose `/v1/stats` or the probe failed.
    stats_before: ServerStats | None = None
    stats_after: ServerStats | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "server": self.server,
            "trace": self.trace_path,
            "num_concurrent": self.num_concurrent,
            "workload": self.workload,
            "timestamp": self.timestamp,
            "turns": [t.to_dict() for t in self.turns],
            "stats_before": self.stats_before.to_dict()
            if self.stats_before is not None
            else None,
            "stats_after": self.stats_after.to_dict()
            if self.stats_after is not None
            else None,
        }


# ─────────────────────────────────────────────────────────────────────
# Trace loading
# ─────────────────────────────────────────────────────────────────────


async def fetch_server_stats(
    client: httpx.AsyncClient, server: str
) -> ServerStats | None:
    """Fetch `/v1/stats` as a plain-text blob and parse it.

    Returns `None` if the endpoint is missing, unreachable, or returns
    a non-200 — the benchmark still runs, we just do not annotate the
    report with server-side deltas.

    TODO (P3+ server-side): the infer server's `/v1/stats` today
    exposes `requests`, `active`, `waiting`, `tokens_out`, `kv_util`,
    `ttft_p50/p99`, `tpot_p50`. It does NOT expose `prefix_hits` or
    `prefix_hit_tokens` yet — the scheduler has no counters wired in.
    Until those land, cross-session prefix hit rate is NOT directly
    observable from this probe; we report the delta of `tokens_out`
    and the final `kv_util` / `ttft_*` gauges as the best proxy. The
    add-the-counters change is ~30–50 LOC in `infer/src/metrics.rs`
    plus a hook in the scheduler's prefix-cache lookup site. See the
    I1 research report in
    `docs/plans/tiered-kv-cache-tasks.md` §N.Addendum (or scroll to
    the bottom of the file for the consolidated remote-validation
    checklist).
    """
    try:
        resp = await client.get(
            f"{server.rstrip('/')}/v1/stats",
            timeout=httpx.Timeout(5.0, connect=5.0),
        )
    except httpx.HTTPError:
        return None
    if resp.status_code != 200:
        return None
    return ServerStats.parse(resp.text)


def load_trace(path: Path) -> list[Session]:
    sessions: list[Session] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
            sessions.append(Session.from_json(obj))
    if not sessions:
        raise ValueError(f"trace {path} is empty")
    return sessions


def _tokenish_text(label: str, count: int, salt: int) -> str:
    words: list[str] = []
    if count >= 4:
        words.extend(["session", label, "unique", str(salt)])
    while len(words) < count:
        idx = (len(words) + salt) % len(_TOKENISH_WORDS)
        words.append(_TOKENISH_WORDS[idx])
    return " ".join(words[:count])


def _assistant_stub(session_idx: int, turn: int) -> str:
    return (
        f"Recorded session {session_idx:03d} turn {turn}. "
        "I will preserve the prior context and continue concisely."
    )


def _w3_prompt_len(rng: random.Random, center: int, jitter: int) -> int:
    return center + rng.randint(-jitter, jitter)


def generate_w3_short_multiturn_trace() -> list[Session]:
    """Build the W3 canonical trace from the agent-load bench spec.

    The generated trace has 64 warm sessions with one unscored base turn plus
    four scored same-session warm turns, and 64 scored cold distractor sessions.
    Scored turns therefore split 256 warm / 64 cold = 80% warm.
    """
    rng = random.Random(W3_SEED)
    warm_sessions: list[Session] = []
    cold_sessions: list[Session] = []
    system_prompt = (
        "You are a concise agent runtime benchmark assistant. "
        "Keep answers short and preserve session context exactly."
    )

    for session_idx in range(W3_WARM_SESSIONS):
        turns: list[dict[str, Any]] = []
        base_tokens = _w3_prompt_len(rng, W3_BASE_PROMPT_TOKENS, W3_BASE_PROMPT_JITTER)
        turns.append(
            {
                "role": "user",
                "content": _tokenish_text(
                    f"warm-{session_idx:03d}-base", base_tokens, session_idx
                ),
                "bench_kind": "warmup",
                "scored": False,
                "expected_prompt_tokens": base_tokens,
            }
        )
        turns.append(
            {
                "role": "assistant",
                "content": _assistant_stub(session_idx, 0),
                "bench_kind": "history",
                "scored": False,
            }
        )

        expected_prompt_tokens = base_tokens
        for scored_turn in range(1, W3_SCORED_WARM_TURNS_PER_SESSION + 1):
            tail_tokens = _w3_prompt_len(
                rng, W3_USER_TAIL_TOKENS, W3_USER_TAIL_JITTER
            )
            expected_prompt_tokens += tail_tokens
            turns.append(
                {
                    "role": "user",
                    "content": _tokenish_text(
                        f"warm-{session_idx:03d}-tail-{scored_turn}",
                        tail_tokens,
                        session_idx + scored_turn * 97,
                    ),
                    "bench_kind": "warm",
                    "scored": True,
                    "expected_prompt_tokens": expected_prompt_tokens,
                    "shared_prefix_min_ratio": 0.8,
                }
            )
            if scored_turn != W3_SCORED_WARM_TURNS_PER_SESSION:
                turns.append(
                    {
                        "role": "assistant",
                        "content": _assistant_stub(session_idx, scored_turn),
                        "bench_kind": "history",
                        "scored": False,
                    }
                )

        warm_sessions.append(
            Session(
                session_id=f"w3-warm-{session_idx:03d}",
                system_prompt=system_prompt,
                turns=turns,
                metadata={
                    "workload": WORKLOAD_W3,
                    "session_kind": "warm",
                    "seed": W3_SEED,
                    "scored_warm_turns": W3_SCORED_WARM_TURNS_PER_SESSION,
                },
            )
        )

    for session_idx in range(W3_COLD_SESSIONS):
        cold_tokens = _w3_prompt_len(
            rng, W3_BASE_PROMPT_TOKENS, W3_BASE_PROMPT_JITTER
        )
        cold_sessions.append(
            Session(
                session_id=f"w3-cold-{session_idx:03d}",
                system_prompt=system_prompt,
                turns=[
                    {
                        "role": "user",
                        "content": _tokenish_text(
                            f"cold-{session_idx:03d}",
                            cold_tokens,
                            10_000 + session_idx,
                        ),
                        "bench_kind": "cold",
                        "scored": True,
                        "expected_prompt_tokens": cold_tokens,
                    }
                ],
                metadata={
                    "workload": WORKLOAD_W3,
                    "session_kind": "cold",
                    "seed": W3_SEED,
                },
            )
        )

    sessions = []
    for warm_session, cold_session in zip(warm_sessions, cold_sessions):
        sessions.append(warm_session)
        sessions.append(cold_session)

    _validate_w3_shape(sessions)
    return sessions


def generate_w4_tool_resume_trace() -> list[Session]:
    """Build the W4 canonical tool-resume trace from the agent-load spec."""
    rng = random.Random(W4_SEED)
    sessions: list[Session] = []
    system_prompt = (
        "You are an agent runtime benchmark assistant. "
        "Use tool results as authoritative context and resume directly."
    )

    for session_idx in range(W4_SESSIONS):
        base_tokens = _w3_prompt_len(
            rng, W4_BASE_PROMPT_TOKENS, W4_BASE_PROMPT_JITTER
        )
        tool_tokens = _w3_prompt_len(
            rng, W4_TOOL_OUTPUT_TOKENS, W4_TOOL_OUTPUT_JITTER
        )
        tool_call_id = f"call_w4_{session_idx:03d}"
        turns: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": _tokenish_text(
                    f"w4-base-{session_idx:03d}", base_tokens, session_idx
                ),
                "bench_kind": "warmup",
                "scored": False,
                "expected_prompt_tokens": base_tokens,
                "max_tokens": W4_WARMUP_MAX_TOKENS,
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": "retrieve_context",
                            "arguments": "{\"query\":\"session-context\"}",
                        },
                    }
                ],
                "bench_kind": "history",
                "scored": False,
            },
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": _tokenish_text(
                    f"w4-tool-{session_idx:03d}",
                    tool_tokens,
                    20_000 + session_idx,
                ),
                "bench_kind": "resume",
                "scored": True,
                "request_after": True,
                "expected_prompt_tokens": base_tokens + tool_tokens,
                "expected_tool_output_tokens": tool_tokens,
                "max_tokens": W4_RESUME_MAX_TOKENS,
            },
        ]
        sessions.append(
            Session(
                session_id=f"w4-session-{session_idx:03d}",
                system_prompt=system_prompt,
                turns=turns,
                metadata={
                    "workload": WORKLOAD_W4,
                    "session_kind": "tool-resume",
                    "seed": W4_SEED,
                },
            )
        )

    _validate_w4_shape(sessions)
    return sessions


def _validate_w3_shape(sessions: list[Session]) -> None:
    warm_sessions = [s for s in sessions if s.metadata.get("session_kind") == "warm"]
    cold_sessions = [s for s in sessions if s.metadata.get("session_kind") == "cold"]
    warm_scored = sum(
        1
        for session in warm_sessions
        for turn in session.turns
        if turn.get("role") == "user"
        and turn.get("bench_kind") == "warm"
        and bool(turn.get("scored", True))
    )
    cold_scored = sum(
        1
        for session in cold_sessions
        for turn in session.turns
        if turn.get("role") == "user"
        and turn.get("bench_kind") == "cold"
        and bool(turn.get("scored", True))
    )
    if len(warm_sessions) != W3_WARM_SESSIONS:
        raise RuntimeError(f"W3 warm session count mismatch: {len(warm_sessions)}")
    if len(cold_sessions) != W3_COLD_SESSIONS:
        raise RuntimeError(f"W3 cold session count mismatch: {len(cold_sessions)}")
    if warm_scored != W3_WARM_SESSIONS * W3_SCORED_WARM_TURNS_PER_SESSION:
        raise RuntimeError(f"W3 scored warm turn count mismatch: {warm_scored}")
    if cold_scored != W3_COLD_SESSIONS:
        raise RuntimeError(f"W3 scored cold turn count mismatch: {cold_scored}")
    scored_total = warm_scored + cold_scored
    if scored_total == 0 or warm_scored / scored_total < 0.8:
        raise RuntimeError("W3 warm scored ratio fell below 80%")


def _validate_w4_shape(sessions: list[Session]) -> None:
    warmup = 0
    resume = 0
    for session in sessions:
        for turn in session.turns:
            if turn.get("bench_kind") == "warmup":
                warmup += 1
                if turn.get("max_tokens") != W4_WARMUP_MAX_TOKENS:
                    raise RuntimeError("W4 warmup max_tokens mismatch")
            if turn.get("bench_kind") == "resume":
                resume += 1
                if not turn.get("request_after"):
                    raise RuntimeError("W4 resume turn must set request_after")
                if not turn.get("scored", False):
                    raise RuntimeError("W4 resume turn must be scored")
                if turn.get("max_tokens") != W4_RESUME_MAX_TOKENS:
                    raise RuntimeError("W4 resume max_tokens mismatch")
    if len(sessions) != W4_SESSIONS:
        raise RuntimeError(f"W4 session count mismatch: {len(sessions)}")
    if warmup != W4_SESSIONS:
        raise RuntimeError(f"W4 warmup count mismatch: {warmup}")
    if resume != W4_SESSIONS:
        raise RuntimeError(f"W4 resume count mismatch: {resume}")


def _session_to_json_obj(session: Session) -> dict[str, Any]:
    return {
        "session_id": session.session_id,
        "system_prompt": session.system_prompt,
        "turns": session.turns,
        "metadata": session.metadata,
    }


def write_trace_jsonl(sessions: list[Session], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for session in sessions:
            json.dump(_session_to_json_obj(session), f, ensure_ascii=False)
            f.write("\n")


def _load_or_generate_sessions(args: argparse.Namespace) -> tuple[list[Session], str]:
    if args.workload == WORKLOAD_W3:
        sessions = generate_w3_short_multiturn_trace()
        trace_ref = f"generated:{WORKLOAD_W3}"
        if args.trace_out is not None:
            trace_path = Path(args.trace_out).expanduser().resolve()
            write_trace_jsonl(sessions, trace_path)
            trace_ref = str(trace_path)
        return sessions, trace_ref
    if args.workload == WORKLOAD_W4:
        sessions = generate_w4_tool_resume_trace()
        trace_ref = f"generated:{WORKLOAD_W4}"
        if args.trace_out is not None:
            trace_path = Path(args.trace_out).expanduser().resolve()
            write_trace_jsonl(sessions, trace_path)
            trace_ref = str(trace_path)
        return sessions, trace_ref

    trace_path = Path(args.trace).expanduser().resolve()
    if not trace_path.is_file():
        raise SystemExit(f"trace file not found: {trace_path}")
    return load_trace(trace_path), str(trace_path)


def print_trace_generation_summary(sessions: list[Session], trace_ref: str) -> None:
    warm_scored = sum(
        1
        for session in sessions
        for turn in session.turns
        if turn.get("bench_kind") == "warm" and bool(turn.get("scored", True))
    )
    cold_scored = sum(
        1
        for session in sessions
        for turn in session.turns
        if turn.get("bench_kind") == "cold" and bool(turn.get("scored", True))
    )
    warmup = sum(
        1
        for session in sessions
        for turn in session.turns
        if turn.get("bench_kind") == "warmup"
    )
    resume = sum(
        1
        for session in sessions
        for turn in session.turns
        if turn.get("bench_kind") == "resume" and bool(turn.get("scored", True))
    )
    scored_total = warm_scored + cold_scored
    warm_ratio = f"{warm_scored / scored_total:.3f}" if scored_total else "n/a"
    print(f"trace: {trace_ref}")
    print(f"sessions: {len(sessions)}")
    print(f"warmup turns: {warmup}")
    print(f"scored warm turns: {warm_scored}")
    print(f"scored cold turns: {cold_scored}")
    print(f"scored resume turns: {resume}")
    print(f"scored warm ratio: {warm_ratio}")
    if sessions:
        print("sample session:")
        print(json.dumps(_session_to_json_obj(sessions[0]), ensure_ascii=False)[:1200])


# ─────────────────────────────────────────────────────────────────────
# Replayer core
# ─────────────────────────────────────────────────────────────────────


def _build_messages(session: Session, up_to_turn: int) -> list[dict[str, Any]]:
    """Build the `messages` array for the `up_to_turn`-th request.

    The request includes: system prompt + all turns 0..up_to_turn (inclusive
    of the final user turn). Assistant and tool turns in between are part
    of the growing history.
    """
    messages: list[dict[str, Any]] = []
    if session.system_prompt:
        messages.append({"role": "system", "content": session.system_prompt})
    for turn in session.turns[: up_to_turn + 1]:
        message = {"role": turn["role"], "content": turn["content"]}
        for key in ("tool_call_id", "tool_calls", "name"):
            if key in turn:
                message[key] = turn[key]
        messages.append(message)
    return messages


def _turn_requests_model(turn: dict[str, Any]) -> bool:
    return turn.get("role") == "user" or bool(turn.get("request_after", False))


def _turn_kind(session: Session, turn_idx: int) -> str:
    return str(session.turns[turn_idx].get("bench_kind", "trace"))


def _turn_scored(session: Session, turn_idx: int) -> bool:
    return bool(session.turns[turn_idx].get("scored", True))


def _turn_expected_prompt_tokens(session: Session, turn_idx: int) -> int | None:
    value = session.turns[turn_idx].get("expected_prompt_tokens")
    return value if isinstance(value, int) else None


def _turn_max_tokens(session: Session, turn_idx: int, default_max_tokens: int) -> int:
    value = session.turns[turn_idx].get("max_tokens")
    return value if isinstance(value, int) else default_max_tokens


async def _stream_one_turn(
    client: httpx.AsyncClient,
    server: str,
    session: Session,
    turn_idx: int,
    max_tokens: int,
) -> TurnResult:
    """Send one turn's request and measure TTFT / ITL / wall."""
    messages = _build_messages(session, turn_idx)
    turn_kind = _turn_kind(session, turn_idx)
    scored = _turn_scored(session, turn_idx)
    expected_prompt_tokens = _turn_expected_prompt_tokens(session, turn_idx)
    request_max_tokens = _turn_max_tokens(session, turn_idx, max_tokens)
    body = {
        "model": "default",
        "messages": messages,
        "session_id": session.session_id,
        "stream": True,
        "max_tokens": request_max_tokens,
        "temperature": 0.0,
    }

    start = time.perf_counter()
    first_token_at: float | None = None
    token_deltas: list[float] = []
    last_token_at: float | None = None
    tokens_out = 0
    finish_reason: str | None = None

    try:
        async with client.stream(
            "POST",
            f"{server.rstrip('/')}/v1/chat/completions",
            json=body,
            timeout=httpx.Timeout(600.0, connect=30.0),
        ) as resp:
            if resp.status_code != 200:
                body_bytes = await resp.aread()
                return TurnResult(
                    session_id=session.session_id,
                    turn_idx=turn_idx,
                    prompt_messages=len(messages),
                    wall_ms=(time.perf_counter() - start) * 1000,
                    ttft_ms=None,
                    itl_ms=None,
                    tokens_out=0,
                    finish_reason=None,
                    turn_kind=turn_kind,
                    scored=scored,
                    expected_prompt_tokens=expected_prompt_tokens,
                    request_max_tokens=request_max_tokens,
                    error=f"HTTP {resp.status_code}: {body_bytes.decode(errors='replace')[:200]}",
                )
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content:
                    now = time.perf_counter()
                    if first_token_at is None:
                        first_token_at = now
                    else:
                        assert last_token_at is not None
                        token_deltas.append((now - last_token_at) * 1000)
                    last_token_at = now
                    tokens_out += 1
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
    except httpx.HTTPError as e:
        return TurnResult(
            session_id=session.session_id,
            turn_idx=turn_idx,
            prompt_messages=len(messages),
            wall_ms=(time.perf_counter() - start) * 1000,
            ttft_ms=None,
            itl_ms=None,
            tokens_out=tokens_out,
            finish_reason=None,
            turn_kind=turn_kind,
            scored=scored,
            expected_prompt_tokens=expected_prompt_tokens,
            request_max_tokens=request_max_tokens,
            error=f"{type(e).__name__}: {e}",
        )

    wall = (time.perf_counter() - start) * 1000
    ttft = (first_token_at - start) * 1000 if first_token_at is not None else None
    itl = statistics.median(token_deltas) if token_deltas else None

    return TurnResult(
        session_id=session.session_id,
        turn_idx=turn_idx,
        prompt_messages=len(messages),
        wall_ms=wall,
        ttft_ms=ttft,
        itl_ms=itl,
        tokens_out=tokens_out,
        finish_reason=finish_reason,
        turn_kind=turn_kind,
        scored=scored,
        expected_prompt_tokens=expected_prompt_tokens,
        request_max_tokens=request_max_tokens,
    )


async def _drive_session(
    client: httpx.AsyncClient,
    server: str,
    session: Session,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> list[TurnResult]:
    """Drive one session through every user turn.

    We replay **user** turns only — the assistant turns already present
    in the trace are part of the history we send along. Each request
    stops generation on the trace's final user turn and we collect
    metrics per request.

    The semaphore caps concurrent in-flight turns across all sessions to
    --num-concurrent. Turns within a session are always sequential (a
    growing conversation).
    """
    results: list[TurnResult] = []
    for turn_idx, turn in enumerate(session.turns):
        if not _turn_requests_model(turn):
            continue
        async with semaphore:
            result = await _stream_one_turn(
                client, server, session, turn_idx, max_tokens
            )
        results.append(result)
        if result.error is not None:
            # Stop this session on first error to avoid cascading noise;
            # other sessions keep going.
            break
    return results


async def run_benchmark(args: argparse.Namespace) -> RunStats:
    sessions, trace_ref = _load_or_generate_sessions(args)
    print(
        f"[bench_agent_trace] loaded {len(sessions)} sessions from {trace_ref}",
        file=sys.stderr,
    )

    stats = RunStats(
        label=args.label,
        server=args.server,
        trace_path=trace_ref,
        num_concurrent=args.num_concurrent,
        workload=args.workload,
        timestamp=_dt.datetime.now(_dt.timezone.utc).isoformat(),
    )

    semaphore = asyncio.Semaphore(args.num_concurrent)
    async with httpx.AsyncClient(http2=False) as client:
        # Probe server-side /v1/stats before the run so the reporter can
        # compute deltas afterwards. Silent on failure — the client-side
        # metrics are always the primary view.
        if args.probe_stats:
            stats.stats_before = await fetch_server_stats(client, args.server)
            if stats.stats_before is None:
                print(
                    "[bench_agent_trace] /v1/stats probe before run returned "
                    "no data (endpoint missing or unreachable)",
                    file=sys.stderr,
                )

        session_tasks = [
            _drive_session(client, args.server, sess, args.max_tokens, semaphore)
            for sess in sessions
        ]
        # as_completed would interleave prints; gather is cleaner for a
        # small benchmark. For large traces add a progress bar later.
        all_results = await asyncio.gather(*session_tasks, return_exceptions=False)

        if args.probe_stats:
            stats.stats_after = await fetch_server_stats(client, args.server)

    for session_results in all_results:
        stats.turns.extend(session_results)
    return stats


# ─────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────


def _fmt_ms(v: float | None) -> str:
    return f"{v:.1f}" if v is not None else "—"


def print_report(stats: RunStats) -> None:
    header = (
        "session          kind      turn  msgs   wall(ms)  ttft(ms)   itl(ms)  tokens  finish"
    )
    sep = "─" * len(header)
    print()
    print(f"## {stats.label}  @  {stats.server}")
    print(f"trace: {stats.trace_path}")
    print(f"timestamp: {stats.timestamp}")
    print(f"num_concurrent: {stats.num_concurrent}")
    print()
    print(header)
    print(sep)
    for t in stats.turns:
        tag = (
            f"{t.session_id:15s}  "
            f"{t.turn_kind[:8]:8s}  "
            f"{t.turn_idx:4d}  "
            f"{t.prompt_messages:4d}  "
            f"{_fmt_ms(t.wall_ms):>9s}  "
            f"{_fmt_ms(t.ttft_ms):>8s}  "
            f"{_fmt_ms(t.itl_ms):>8s}  "
            f"{t.tokens_out:6d}  "
            f"{(t.finish_reason or '-'):s}"
        )
        if t.error:
            tag += f"  ERROR: {t.error[:80]}"
        print(tag)
    print(sep)
    _print_aggregate(stats)


def _print_aggregate(stats: RunStats) -> None:
    ok_turns = [t for t in stats.turns if t.error is None and t.tokens_out > 0]
    if not ok_turns:
        print("\nno successful turns — check server logs")
        _print_server_probe(stats)
        return
    scored_turns = [t for t in ok_turns if t.scored]
    aggregate_turns = scored_turns or ok_turns
    ttfts = [t.ttft_ms for t in aggregate_turns if t.ttft_ms is not None]
    itls = [t.itl_ms for t in aggregate_turns if t.itl_ms is not None]
    tokens_total = sum(t.tokens_out for t in aggregate_turns)
    wall_total = sum(t.wall_ms for t in aggregate_turns)

    def pct(xs: list[float], p: float) -> float:
        if not xs:
            return 0.0
        xs_sorted = sorted(xs)
        k = int(round((p / 100) * (len(xs_sorted) - 1)))
        return xs_sorted[k]

    print()
    print(f"turns OK:        {len(ok_turns)} / {len(stats.turns)}")
    if scored_turns:
        print(f"scored turns OK: {len(scored_turns)}")
    print(f"tokens total:    {tokens_total}")
    print(f"wall total (s):  {wall_total / 1000:.2f}")
    if ttfts:
        print(f"TTFT p50/p99:    {pct(ttfts, 50):.1f} / {pct(ttfts, 99):.1f} ms")
    if itls:
        print(f"ITL  p50/p99:    {pct(itls, 50):.1f} / {pct(itls, 99):.1f} ms")
    _print_w3_aggregate(aggregate_turns, pct)
    _print_w4_aggregate(aggregate_turns, pct)

    _print_server_probe(stats)


def _print_w3_aggregate(
    aggregate_turns: list[TurnResult],
    pct: Any,
) -> None:
    warm = [t for t in aggregate_turns if t.turn_kind == "warm"]
    cold = [t for t in aggregate_turns if t.turn_kind == "cold"]
    if not warm and not cold:
        return

    def ttft_pair(turns: list[TurnResult]) -> tuple[str, str]:
        ttfts = [t.ttft_ms for t in turns if t.ttft_ms is not None]
        if not ttfts:
            return "—", "—"
        return f"{pct(ttfts, 50):.1f}", f"{pct(ttfts, 99):.1f}"

    warm_p50, warm_p99 = ttft_pair(warm)
    cold_p50, cold_p99 = ttft_pair(cold)
    print()
    print("W3 scored split:")
    print(f"  warm turns: {len(warm)} TTFT p50/p99={warm_p50}/{warm_p99} ms")
    print(f"  cold turns: {len(cold)} TTFT p50/p99={cold_p50}/{cold_p99} ms")


def _print_w4_aggregate(
    aggregate_turns: list[TurnResult],
    pct: Any,
) -> None:
    resume = [t for t in aggregate_turns if t.turn_kind == "resume"]
    if not resume:
        return
    ttfts = [t.ttft_ms for t in resume if t.ttft_ms is not None]
    walls = [t.wall_ms for t in resume]
    ttft_p50 = f"{pct(ttfts, 50):.1f}" if ttfts else "—"
    ttft_p99 = f"{pct(ttfts, 99):.1f}" if ttfts else "—"
    wall_p50 = f"{pct(walls, 50):.1f}" if walls else "—"
    wall_p99 = f"{pct(walls, 99):.1f}" if walls else "—"
    print()
    print("W4 scored resume:")
    print(f"  resume turns: {len(resume)} TTFT p50/p99={ttft_p50}/{ttft_p99} ms")
    print(f"  resume E2E p50/p99={wall_p50}/{wall_p99} ms")


def _print_server_probe(stats: RunStats) -> None:
    """Report the delta between before/after snapshots of /v1/stats.

    Monotonic counters (`requests`, `tokens_out`) get a computed delta;
    gauges (`active`, `waiting`, `kv_util`, `ttft_p*`, `tpot_p50`) are
    shown as their post-run value because delta doesn't make sense for
    them. If either snapshot is missing, this is a no-op.
    """
    before, after = stats.stats_before, stats.stats_after
    if before is None and after is None:
        return
    print()
    print("server /v1/stats:")
    if before is None:
        print("  before: (unavailable)")
    else:
        print(f"  before: {before.raw}")
    if after is None:
        print("  after:  (unavailable)")
        return
    print(f"  after:  {after.raw}")

    # Delta for cumulative counters.
    if before is not None:
        counter_deltas: list[tuple[str, int]] = []
        for name in ("requests", "tokens_out"):
            b, a = before.int_field(name), after.int_field(name)
            if b is not None and a is not None:
                counter_deltas.append((name, a - b))
        if counter_deltas:
            pretty = " ".join(f"{name}=+{delta}" for name, delta in counter_deltas)
            print(f"  delta:  {pretty}")

    # Inform the operator that prefix-hit rate is not yet in /v1/stats.
    # The I1 research explicitly flagged this: scheduler has no prefix
    # hit counters today, so cross-session prefix hit rate cannot be
    # derived from this probe. The P1 exit gate will need a server-side
    # addition. See TODO in fetch_server_stats().
    if "prefix_hit_rate" not in after.fields and "prefix_hits" not in after.fields:
        print(
            "  note:   prefix_hit_rate not exposed by /v1/stats yet; "
            "server-side counter addition pending (see I1 research in "
            "docs/plans/tiered-kv-cache-tasks.md)"
        )


def save_json(stats: RunStats, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(stats.to_dict(), f, indent=2)
    print(f"\nwrote {out}", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-turn agent trace replayer for Tiered KV Cache scoreboard",
    )
    p.add_argument(
        "--workload",
        choices=(WORKLOAD_TRACE, WORKLOAD_W3, WORKLOAD_W4),
        default=WORKLOAD_TRACE,
        help=(
            "Trace source. 'trace' replays --trace; "
            f"'{WORKLOAD_W3}' and '{WORKLOAD_W4}' generate canonical workloads."
        ),
    )
    p.add_argument(
        "--server",
        default="http://localhost:8000",
        help="OpenAI-compatible server URL (default: %(default)s)",
    )
    p.add_argument(
        "--trace",
        default=str(DEFAULT_TRACE),
        help="JSONL trace file path (default: scripts/data/agent_trace_default.jsonl)",
    )
    p.add_argument(
        "--label",
        default="infer",
        help="Label tag for the run (default: %(default)s)",
    )
    p.add_argument(
        "--num-concurrent",
        type=int,
        default=None,
        help=(
            "Max concurrent in-flight turns across all sessions. "
            "Default: 4 for --workload trace, 16 for W3, 8 for W4."
        ),
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=(
            "Per-turn generation cap. Default: 256 for --workload trace, "
            "64 for W3, 256 for W4 resume. W4 warmup uses 64."
        ),
    )
    p.add_argument(
        "--out",
        default=None,
        help="Optional JSON snapshot output path",
    )
    p.add_argument(
        "--trace-out",
        default=None,
        help="Optional JSONL output path for generated workload traces.",
    )
    p.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate the selected workload trace and exit without HTTP requests.",
    )
    p.add_argument(
        "--probe-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Poll GET /v1/stats before and after the run; report deltas "
            "in the aggregate section. Turn off with --no-probe-stats "
            "when running against servers that do not expose /v1/stats."
        ),
    )
    args = p.parse_args(argv)
    _resolve_workload_defaults(args)
    return args


def _resolve_workload_defaults(args: argparse.Namespace) -> None:
    if args.workload == WORKLOAD_W3:
        if args.num_concurrent is not None and args.num_concurrent != W3_CONCURRENCY:
            raise SystemExit(
                f"{WORKLOAD_W3} requires --num-concurrent {W3_CONCURRENCY}"
            )
        if args.max_tokens is not None and args.max_tokens != W3_MAX_TOKENS:
            raise SystemExit(f"{WORKLOAD_W3} requires --max-tokens {W3_MAX_TOKENS}")
        args.num_concurrent = W3_CONCURRENCY
        args.max_tokens = W3_MAX_TOKENS
        return
    if args.workload == WORKLOAD_W4:
        if args.num_concurrent is not None and args.num_concurrent != W4_CONCURRENCY:
            raise SystemExit(
                f"{WORKLOAD_W4} requires --num-concurrent {W4_CONCURRENCY}"
            )
        if args.max_tokens is not None and args.max_tokens != W4_RESUME_MAX_TOKENS:
            raise SystemExit(
                f"{WORKLOAD_W4} requires --max-tokens {W4_RESUME_MAX_TOKENS}"
            )
        args.num_concurrent = W4_CONCURRENCY
        args.max_tokens = W4_RESUME_MAX_TOKENS
        return

    if args.num_concurrent is None:
        args.num_concurrent = 4
    if args.max_tokens is None:
        args.max_tokens = 256


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.generate_only:
        sessions, trace_ref = _load_or_generate_sessions(args)
        print_trace_generation_summary(sessions, trace_ref)
        return 0
    try:
        stats = asyncio.run(run_benchmark(args))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    print_report(stats)
    if args.out is not None:
        save_json(stats, Path(args.out).expanduser())
    return 0


if __name__ == "__main__":
    sys.exit(main())
