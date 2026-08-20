"""L2 — Resident-sequence prefix reuse (position-locked backends).

For backends whose device KV is one flat, absolute-position buffer with no page
indirection — the Vulkan lane — reuse cannot be expressed as "attach these radix
pages at these positions". Such a lane holds exactly ONE resident sequence and
can only serve a request that CONTINUES it: `tokens` must extend the resident
token list exactly. `token_reuse` cannot measure this, because it fetches its
suffix ids between turn 1 and turn 2 and that third request evicts the lane.

Protocol (temp=0), suffix ids fetched FIRST so nothing intervenes:
  turn 1 — prompt=P (string, salted doc, needle mid-prompt), return_token_ids
           → capture PT (prompt ids) + GT (generated ids).
  ON     — prompt = PT + GT + suffix, issued with NOTHING in between. The lane
           still holds PT+GT, so the hit must reach |PT|+|GT| exactly (not
           |PT|+|GT|-1: that last sampled token is never fed back, so the
           backend owes a finish write-through to close the gap).
  CONTROL— evict with an unrelated short request, then replay the SAME turn-2
           prompt. The hit must collapse to 0, proving the ON hit is the
           feature and not a page-route match.

This gate measures reuse EXTENT and attribution only. Output correctness under
reuse is `prefix_reuse`'s job (needle retrieval after a restore).
"""
from __future__ import annotations

import time

from . import PREFIX_HIT_TOKENS, Gate, Verdict, build_doc, get_stats, stat_delta
from .token_reuse import SUFFIX_TEXT, post_completion

NEEDLE = "738291"


class ResidentReuseGate(Gate):
    name = "resident_reuse"

    def __init__(
        self,
        prompt_tokens: int = 500,
        gen_tokens: int = 128,
        runs: int = 1,
        needle: str = NEEDLE,
    ):
        super().__init__(runs)
        self.prompt_tokens = prompt_tokens
        self.gen_tokens = gen_tokens
        self.needle = needle

    def _timed(self, prompt_ids: list[int]) -> tuple[int, dict]:
        s0 = get_stats()
        r = post_completion(prompt_ids, max_tokens=8, return_token_ids=True)
        time.sleep(0.3)
        return stat_delta(s0, get_stats(), PREFIX_HIT_TOKENS), r

    def run(self) -> Verdict:
        # Round-trip the suffix through the tokenizer BEFORE turn 1: on a
        # single-resident lane this request is itself an eviction.
        suffix = list(
            post_completion(SUFFIX_TEXT, max_tokens=1, return_token_ids=True)["prompt_token_ids"]
        )

        doc, _ = build_doc(self.prompt_tokens, self.needle, needle_pos=0.5)
        turn1 = post_completion(doc, max_tokens=self.gen_tokens, return_token_ids=True)
        pt, gt = list(turn1["prompt_token_ids"]), list(turn1["token_ids"])
        turn2 = pt + gt + suffix
        ceiling = len(pt) + len(gt)

        on_hit, on_r = self._timed(turn2)
        # Evict, then replay the identical prompt for the no-reuse baseline.
        post_completion("Hello", max_tokens=1)
        off_hit, off_r = self._timed(turn2)

        summary = {
            "prompt_len": len(pt),
            "gen_len": len(gt),
            "turn2_len": len(turn2),
            "ceiling": ceiling,
            "on_hit_tokens": on_hit,
            "off_hit_tokens": off_hit,
            "on_dt": round(on_r["dt"], 2),
            "off_dt": round(off_r["dt"], 2),
            "speedup": round(off_r["dt"] / on_r["dt"], 2) if on_r["dt"] else None,
            "on_text": on_r["text"][:80],
        }
        print(
            f"  |PT|={len(pt)} |GT|={len(gt)} ceiling={ceiling} "
            f"on_hit={on_hit} off_hit={off_hit} "
            f"on={on_r['dt']:.2f}s off={off_r['dt']:.2f}s"
        )

        if on_hit < ceiling:
            gap = ceiling - on_hit
            return Verdict(
                self.name,
                False,
                summary,
                reason=(
                    f"on_hit={on_hit} short of ceiling={ceiling} by {gap}"
                    + (
                        " — the finish write-through never ran, so the last sampled "
                        "token is missing from the resident image"
                        if gap == 1
                        else " — resident reuse did not cover the prior turn"
                    )
                ),
            )
        if off_hit != 0:
            return Verdict(
                self.name,
                False,
                summary,
                reason=(
                    f"off_hit={off_hit} after eviction — the control arm reused too, "
                    "so the ON hit is not attributable to the resident image"
                ),
            )
        return Verdict(self.name, True, summary)
