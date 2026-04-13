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


# ─────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────


@dataclass
class Session:
    session_id: str
    system_prompt: str
    turns: list[dict[str, str]]

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
        return cls(session_id=session_id, system_prompt=system_prompt, turns=turns)


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
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_idx": self.turn_idx,
            "prompt_messages": self.prompt_messages,
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


# ─────────────────────────────────────────────────────────────────────
# Replayer core
# ─────────────────────────────────────────────────────────────────────


def _build_messages(session: Session, up_to_turn: int) -> list[dict[str, str]]:
    """Build the `messages` array for the `up_to_turn`-th request.

    The request includes: system prompt + all turns 0..up_to_turn (inclusive
    of the final user turn). Assistant and tool turns in between are part
    of the growing history.
    """
    messages: list[dict[str, str]] = []
    if session.system_prompt:
        messages.append({"role": "system", "content": session.system_prompt})
    for turn in session.turns[: up_to_turn + 1]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    return messages


async def _stream_one_turn(
    client: httpx.AsyncClient,
    server: str,
    session: Session,
    turn_idx: int,
    max_tokens: int,
) -> TurnResult:
    """Send one turn's request and measure TTFT / ITL / wall."""
    messages = _build_messages(session, turn_idx)
    body = {
        "model": "default",
        "messages": messages,
        "session_id": session.session_id,
        "stream": True,
        "max_tokens": max_tokens,
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
        if turn["role"] != "user":
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
    trace_path = Path(args.trace).expanduser().resolve()
    if not trace_path.is_file():
        raise SystemExit(f"trace file not found: {trace_path}")
    sessions = load_trace(trace_path)
    print(
        f"[bench_agent_trace] loaded {len(sessions)} sessions from {trace_path}",
        file=sys.stderr,
    )

    stats = RunStats(
        label=args.label,
        server=args.server,
        trace_path=str(trace_path),
        num_concurrent=args.num_concurrent,
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
        "session          turn  msgs   wall(ms)  ttft(ms)   itl(ms)  tokens  finish"
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
    ttfts = [t.ttft_ms for t in ok_turns if t.ttft_ms is not None]
    itls = [t.itl_ms for t in ok_turns if t.itl_ms is not None]
    tokens_total = sum(t.tokens_out for t in ok_turns)
    wall_total = sum(t.wall_ms for t in ok_turns)

    def pct(xs: list[float], p: float) -> float:
        if not xs:
            return 0.0
        xs_sorted = sorted(xs)
        k = int(round((p / 100) * (len(xs_sorted) - 1)))
        return xs_sorted[k]

    print()
    print(f"turns OK:        {len(ok_turns)} / {len(stats.turns)}")
    print(f"tokens total:    {tokens_total}")
    print(f"wall total (s):  {wall_total / 1000:.2f}")
    if ttfts:
        print(f"TTFT p50/p99:    {pct(ttfts, 50):.1f} / {pct(ttfts, 99):.1f} ms")
    if itls:
        print(f"ITL  p50/p99:    {pct(itls, 50):.1f} / {pct(itls, 99):.1f} ms")

    _print_server_probe(stats)


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
        default=4,
        help="Max concurrent in-flight turns across all sessions (default: %(default)s)",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Per-turn generation cap (default: %(default)s)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Optional JSON snapshot output path",
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
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
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
