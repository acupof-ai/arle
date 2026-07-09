"""L2 — Prefix-reuse correctness gate.

Tests the exact path: radix cache hit → sidecar restore → suffix prefill.
Catches silent KV corruption when FlashMLA device page tables go stale.

Protocol per run:
  1. WARM   — doc only (max_tokens=1) → seeds cache
  2. REUSE  — doc + question → hits prefix, prefills question (suffix)
  3. CTRL   — mutated(doc) + question → full recompute baseline

The needle sits on the last prefix page. If suffix pack writes corrupt that
page (stale device table), reuse fails needle retrieval while ctrl succeeds.
"""
from __future__ import annotations

import time

from . import (
    Gate,
    Verdict,
    build_doc,
    get_stats,
    mutate_prefix,
    query,
    stat_delta,
)

QUESTION = "\n\nQuestion: What is the secret access code stated in the document?\nAnswer:"
NEEDLE = "738291"


class PrefixReuseGate(Gate):
    name = "prefix_reuse"

    def __init__(self, target_tokens: int = 2000, runs: int = 3, needle: str = NEEDLE):
        super().__init__(runs)
        self.target = target_tokens
        self.needle = needle

    def _warm(self, doc: str) -> int:
        s0 = get_stats()
        query(doc, max_tokens=1)
        time.sleep(0.3)
        s1 = get_stats()
        return stat_delta(s0, s1, "prefix_hit_tokens")

    def _reuse(self, doc: str) -> tuple[dict, int, int]:
        s0 = get_stats()
        r = query(doc + QUESTION, max_tokens=48)
        time.sleep(0.3)
        s1 = get_stats()
        return r, stat_delta(s0, s1, "prefix_hit_tokens"), stat_delta(s0, s1, "prefix_hit_pages")

    def _control(self, doc: str) -> dict:
        return query(mutate_prefix(doc) + QUESTION, max_tokens=48)

    def run(self) -> Verdict:
        results = []
        for i in range(self.runs):
            doc, approx = build_doc(self.target, self.needle)
            warm_hit = self._warm(doc)
            reuse, hit_tok, hit_pg = self._reuse(doc)
            ctrl = self._control(doc)

            r_needle = self.needle in reuse["text"]
            c_needle = self.needle in ctrl["text"]
            actual = hit_tok > 0

            flag = "!" if actual and not r_needle else " "
            print(
                f"  run {i}: tok≈{approx} "
                f"reuse_hit={hit_tok}t/{hit_pg}p "
                f"warm_hit={warm_hit}t "
                f"reuse={'Y' if r_needle else 'N'}{flag} "
                f"ctrl={'Y' if c_needle else 'N'} "
                f"actual={'Y' if actual else 'N'}",
            )
            results.append({
                "run": i,
                "approx_tokens": approx,
                "warm_hit_tokens": warm_hit,
                "reuse_hit_tokens": hit_tok,
                "reuse_hit_pages": hit_pg,
                "reuse_actual": actual,
                "reuse_needle": r_needle,
                "ctrl_needle": c_needle,
                "reuse_text": reuse["text"][:80],
                "ctrl_text": ctrl["text"][:80],
                "reuse_dt": reuse["dt"],
                "ctrl_dt": ctrl["dt"],
            })

        actual_runs = [r for r in results if r["reuse_actual"]]
        if not actual_runs:
            return Verdict(
                self.name,
                False,
                {"results": results},
                "prefix_hit_tokens never increased — cache not seeded",
            )

        reuse_ok = sum(1 for r in actual_runs if r["reuse_needle"])
        ctrl_ok = sum(1 for r in results if r["ctrl_needle"])
        n_a = len(actual_runs)
        n_t = len(results)

        reuse_rate = reuse_ok / n_a
        ctrl_rate = ctrl_ok / n_t
        fail = reuse_ok < ctrl_ok and ctrl_rate > 0 and reuse_rate + 0.34 < ctrl_rate

        avg_hit_tok = sum(r["reuse_hit_tokens"] for r in actual_runs) / n_a
        avg_reuse_dt = sum(r["reuse_dt"] for r in actual_runs) / n_a
        avg_ctrl_dt = sum(r["ctrl_dt"] for r in results) / n_t
        speedup = avg_ctrl_dt / max(0.001, avg_reuse_dt)

        v = Verdict(
            self.name,
            not fail,
            {
                "target_tokens": self.target,
                "runs": n_t,
                "reuse_runs_actual": n_a,
                "reuse_needle": f"{reuse_ok}/{n_a}",
                "ctrl_needle": f"{ctrl_ok}/{n_t}",
                "avg_reuse_hit_tokens": round(avg_hit_tok, 1),
                "avg_reuse_dt_s": round(avg_reuse_dt, 2),
                "avg_ctrl_dt_s": round(avg_ctrl_dt, 2),
                "reuse_speedup": round(speedup, 2),
                "results": results,
            },
        )
        if fail:
            v.reason = f"reuse {reuse_ok}/{n_a} < ctrl {ctrl_ok}/{n_t} — KV corruption"
        return v
