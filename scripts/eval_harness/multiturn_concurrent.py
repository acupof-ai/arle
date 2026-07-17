"""Multi-turn concurrent agent-workload harness — TTFT/TPOT/throughput with
decode-region KV reuse actually firing.

Independent synthetic prompts have no shared growing prefix, so they
cannot exercise the DSv4 decode-region reuse. Here C conversations run T turns
each; turn k replays the EXACT prior-turn token ids (prompt_ids ++ generated_ids)
as a token-id prompt, so the radix matches the full history and reuse fires INTO
the prior turn's generated region — the campaign's missing measurement.

Method — per turn, phased so concurrency stays exactly C and /v1/stats deltas
attribute cleanly:
  probe phase — C concurrent non-streaming max_tokens=1 return_token_ids POSTs.
    dt ≈ TTFT (prefill of the reused prefix + 1 token, under load C). Publishes
    the turn's full prompt to the radix.
  full  phase — C concurrent non-streaming max_tokens=GEN return_token_ids POSTs,
    the SAME prompt (reuses the probe's just-published prefix ⇒ ~no prefill), so
    dt/len(GT) ≈ TPOT (decode-only). Yields the EXACT GT for the next turn.
The server's SSE stream carries decoded text only (no token ids —
coordinator.rs), so a streamed turn cannot preserve exact ids for feed-back;
re-tokenizing the text drifts the boundary and truncates reuse. The probe+full
split gives real TTFT/TPOT AND token-exact history at ~one generation of decode
per turn (probe adds only a prefill+1).

Reuse proof: the turn-2 probe phase is bracketed by /v1/stats; its
prefix_cache_hit_tokens delta must exceed the prompt-only floor (Σ_c
floor(|PT1_c|)), proving reuse extended past the prompt boundary into the prior
turn's generated region. A flag-OFF serve collapses the delta to ≤ that floor.

Emits one JSON block per concurrency level:
  {concurrency: {c: {ttft_p50, ttft_p99, tpot_p50, tpot_p99, agg_tok_s,
                     reuse_hit_tokens_turn2, prompt_floor, reuse_ok}}}
"""
from __future__ import annotations

import concurrent.futures
import json
import math
import random
import time
import urllib.request

from . import BASE, MODEL, PREFIX_HIT_TOKENS, Gate, Verdict, build_doc, get_stats, stat_delta

NEEDLE = "738291"
NEW_TURN_TEXT = "\n\nUser: Continue and elaborate further.\n\nAssistant:"


def post(
    prompt: str | list[int],
    max_tokens: int,
    return_token_ids: bool = True,
    ignore_eos: bool = False,
    timeout: int = 600,
) -> dict:
    """Non-streaming /v1/completions with a string OR token-id-array prompt.
    `ignore_eos` forces a fixed GEN length so TPOT samples are comparable."""
    body: dict = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "return_token_ids": return_token_ids,
    }
    if ignore_eos:
        body["ignore_eos"] = True
    req = urllib.request.Request(
        BASE + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    resp = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    dt = time.time() - t0
    choice = resp["choices"][0]
    return {
        "token_ids": choice.get("token_ids") or [],
        "prompt_token_ids": choice.get("prompt_token_ids") or [],
        "dt": dt,
    }


def pct(xs: list[float], p: float) -> float | None:
    """Nearest-rank percentile (robust for the small n a sweep produces)."""
    if not xs:
        return None
    s = sorted(xs)
    k = max(0, min(len(s) - 1, math.ceil(p / 100 * len(s)) - 1))
    return round(s[k], 6)


class MultiTurnConcurrentGate(Gate):
    name = "multiturn_concurrent"

    def __init__(
        self,
        concurrency: list[int] | None = None,
        turns: int = 4,
        prompt_tokens: int = 500,
        gen_tokens: int = 128,
        page: int = 16,
        needle: str = NEEDLE,
    ):
        super().__init__(runs=1)
        self.concurrency = concurrency or [1, 4, 8, 16]
        self.turns = turns
        self.prompt_tokens = prompt_tokens
        self.gen_tokens = gen_tokens
        self.page = page
        self.needle = needle
        self._new_turn_ids: list[int] = []

    def _floor(self, n: int) -> int:
        return (n // self.page) * self.page

    def _turn1_prompt(self, salt: str, c: int) -> str:
        # Unique session prefix ⇒ no cross-conversation reuse (each conversation
        # reuses only its OWN prior turns).
        doc, _ = build_doc(self.prompt_tokens, f"{self.needle}{c:03d}", needle_pos=0.5)
        return f"[sess {salt}-{c}] " + doc

    def _run_level(self, c: int) -> dict:
        salt = f"{random.randint(0, 1 << 30):x}"
        histories: list[list[int]] = [[] for _ in range(c)]
        pt1_lens = [0] * c
        ttft: list[float] = []
        tpot: list[float] = []
        total_gen = 0
        reuse_turn2 = 0

        t_level = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
            for turn in range(1, self.turns + 1):
                if turn == 1:
                    prompts: list = [self._turn1_prompt(salt, i) for i in range(c)]
                else:
                    prompts = [histories[i] + self._new_turn_ids for i in range(c)]

                s0 = get_stats() if turn == 2 else None
                probes = list(pool.map(lambda p: post(p, max_tokens=1), prompts))
                if turn == 2:
                    time.sleep(0.3)
                    reuse_turn2 = stat_delta(s0, get_stats(), PREFIX_HIT_TOKENS)
                ttft.extend(r["dt"] for r in probes)

                # Full phase replays the probe's EXACT prompt ids (reuses probe).
                full_prompts = [r["prompt_token_ids"] for r in probes]
                fulls = list(
                    pool.map(
                        lambda p: post(p, max_tokens=self.gen_tokens, ignore_eos=True),
                        full_prompts,
                    )
                )
                for i, (pr, fr) in enumerate(zip(probes, fulls)):
                    gt = fr["token_ids"]
                    if gt:
                        tpot.append(fr["dt"] / len(gt))
                    total_gen += len(gt)
                    histories[i] = list(pr["prompt_token_ids"]) + list(gt)
                    if turn == 1:
                        pt1_lens[i] = len(pr["prompt_token_ids"])
        wall = time.time() - t_level

        prompt_floor = sum(self._floor(n) for n in pt1_lens)
        reuse_ok = reuse_turn2 > prompt_floor
        return {
            "ttft_p50": pct(ttft, 50),
            "ttft_p99": pct(ttft, 99),
            "tpot_p50": pct(tpot, 50),
            "tpot_p99": pct(tpot, 99),
            "agg_tok_s": round(total_gen / wall, 2) if wall > 0 else None,
            "reuse_hit_tokens_turn2": reuse_turn2,
            "prompt_floor": prompt_floor,
            "reuse_ok": reuse_ok,
        }

    def run(self) -> Verdict:
        # Exact ids of the fixed inter-turn user message (round-tripped once).
        self._new_turn_ids = post(NEW_TURN_TEXT, max_tokens=1)["prompt_token_ids"]
        levels: dict[str, dict] = {}
        for c in self.concurrency:
            r = self._run_level(c)
            levels[str(c)] = r
            print(
                f"  C={c:>3} ttft_p50={r['ttft_p50']} ttft_p99={r['ttft_p99']} "
                f"tpot_p50={r['tpot_p50']} tpot_p99={r['tpot_p99']} "
                f"agg_tok_s={r['agg_tok_s']} reuse_turn2={r['reuse_hit_tokens_turn2']} "
                f"floor={r['prompt_floor']} reuse_ok={'Y' if r['reuse_ok'] else 'N'}"
            )
        all_ok = all(r["reuse_ok"] for r in levels.values())
        v = Verdict(self.name, all_ok, {"concurrency": levels})
        if not all_ok:
            bad = [c for c, r in levels.items() if not r["reuse_ok"]]
            v.reason = (
                f"turn-2 reuse delta did not exceed the prompt floor at C={bad} — "
                f"decode-region reuse inactive (reuse-OFF serve, or reuse not firing)"
            )
        return v
