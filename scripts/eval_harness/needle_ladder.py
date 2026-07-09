"""L1 — Needle ladder: same prompt × N repeats, check retrieval consistency.

Catches general correctness regressions (not prefix-reuse-specific).
Spans the 241-token boundary where many attention implementations diverge.
"""
from __future__ import annotations

from . import Gate, Verdict, build_doc, query

QUESTION = "\n\nQuestion: What does the treasure chest contain?\nAnswer:"
NEEDLE = "427 gold coins"


class NeedleLadderGate(Gate):
    name = "needle_ladder"

    def __init__(
        self,
        lengths: list[int] | None = None,
        runs: int = 3,
        needle: str = NEEDLE,
    ):
        super().__init__(runs)
        self.lengths = lengths or [115, 180, 241, 300, 446, 1000, 2000, 4000]
        self.needle = needle

    def run(self) -> Verdict:
        results = {}
        total_hits = 0
        total = 0

        for length in self.lengths:
            hits = 0
            texts = []
            for _ in range(self.runs):
                doc, approx = build_doc(length, self.needle, needle_pos=0.5)
                r = query(doc + QUESTION, max_tokens=32)
                found = self.needle in r["text"]
                hits += int(found)
                texts.append(r["text"][:60])
                total += 1

            total_hits += hits
            det = len(set(texts)) == 1
            results[str(length)] = {
                "approx_tokens": approx,
                "hits": f"{hits}/{self.runs}",
                "deterministic": det,
                "samples": texts,
            }
            print(f"  {length}tok: {hits}/{self.runs} det={'Y' if det else 'N'}")

        passed = total_hits == total
        return Verdict(
            self.name,
            passed,
            {
                "lengths": self.lengths,
                "runs_per_length": self.runs,
                "total": f"{total_hits}/{total}",
                "per_length": results,
            },
            "" if passed else f"{total - total_hits} misses across {total} queries",
        )
