"""L2 — Token-preserving multi-turn prefix-reuse delta.

Measures whether turn-2 reuse extends INTO the prior turn's GENERATED region —
the DSv4 finish-write-through decode-region reuse. The text-reconstruction driver
re-tokenizes the fed-back completion, so the boundary tokens shift and the radix
match truncates at the prompt boundary; the decode-region delta is then
unmeasurable. Here turn 2 replays EXACT token ids (prompt ids + generated ids),
so a match can only truncate where the cache truly diverges.

Protocol (one live serve, temp=0):
  turn 1 — POST prompt=P (string, salted doc, needle mid-prompt),
           return_token_ids → capture PT (prompt ids) + GT (generated ids).
  ON     — POST prompt = PT + GT + suffix (TOKEN-ID array) → prefix_hit_tokens
           delta should reach floor((|PT|+|GT|)/page)·page (reuse into GT).
  CONTROL— POST prompt = PT + suffix + GT (shares only PT: diverges right after
           the prompt) → prefix_hit_tokens delta hits floor(|PT|/page)·page.
           A flag-OFF serve collapses ON onto this same prompt-boundary floor,
           so delta≈0 proves the ON delta is the feature.

Reports {off_hit_tokens, on_hit_tokens, prompt_floor, expected_finish_floor,
delta}. PASS = on_hit_tokens reaches the finish floor (delta > 0).
"""
from __future__ import annotations

import json
import time
import urllib.request

from . import BASE, MODEL, Gate, Verdict, build_doc, get_stats, stat_delta

NEEDLE = "738291"
SUFFIX_TEXT = "\nQ:"


def post_completion(
    prompt: str | list[int],
    max_tokens: int,
    return_token_ids: bool = False,
    timeout: int = 600,
) -> dict:
    """POST /v1/completions with a string OR token-id-array prompt. `prompt` and
    `token_ids` come back only when `return_token_ids` is set."""
    body: dict = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    if return_token_ids:
        body["return_token_ids"] = True
    req = urllib.request.Request(
        BASE + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    resp = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    choice = resp["choices"][0]
    return {
        "text": choice["text"],
        "token_ids": choice.get("token_ids"),
        "prompt_token_ids": choice.get("prompt_token_ids"),
        "dt": time.time() - t0,
    }


def _hit_delta(prompt: list[int], max_tokens: int = 8) -> tuple[int, dict]:
    s0 = get_stats()
    r = post_completion(prompt, max_tokens=max_tokens, return_token_ids=True)
    time.sleep(0.3)
    s1 = get_stats()
    return stat_delta(s0, s1, "prefix_hit_tokens"), r


class TokenReuseGate(Gate):
    name = "token_reuse"

    def __init__(
        self,
        prompt_tokens: int = 500,
        gen_tokens: int = 128,
        page: int = 16,
        runs: int = 1,
        needle: str = NEEDLE,
    ):
        super().__init__(runs)
        self.prompt_tokens = prompt_tokens
        self.gen_tokens = gen_tokens
        self.page = page
        self.needle = needle

    def _suffix_ids(self) -> list[int]:
        # Exact ids of a short fixed continuation — round-tripped through the
        # tokenizer via the prompt-id echo (max_tokens=1; 0 is rejected).
        r = post_completion(SUFFIX_TEXT, max_tokens=1, return_token_ids=True)
        return list(r["prompt_token_ids"])

    def _floor(self, n: int) -> int:
        return (n // self.page) * self.page

    def run(self) -> Verdict:
        doc, _ = build_doc(self.prompt_tokens, self.needle, needle_pos=0.5)
        turn1 = post_completion(doc, max_tokens=self.gen_tokens, return_token_ids=True)
        pt = list(turn1["prompt_token_ids"])
        gt = list(turn1["token_ids"])
        suffix = self._suffix_ids()

        prompt_floor = self._floor(len(pt))
        finish_floor = self._floor(len(pt) + len(gt))

        # ON: exact PT + GT + suffix — reuse can extend into the generated region.
        on_hit, on_r = _hit_delta(pt + gt + suffix)
        # CONTROL: PT + suffix + GT — shares only PT (diverges right after the
        # prompt), so it can only reach the prompt-boundary floor.
        off_hit, _ = _hit_delta(pt + suffix + gt)

        delta = on_hit - off_hit
        into_gen = on_hit >= finish_floor and on_hit > prompt_floor

        print(
            f"  |PT|={len(pt)} |GT|={len(gt)} page={self.page} "
            f"prompt_floor={prompt_floor} finish_floor={finish_floor} "
            f"on_hit={on_hit} off_hit={off_hit} delta={delta} "
            f"into_gen={'Y' if into_gen else 'N'}"
        )

        summary = {
            "off_hit_tokens": off_hit,
            "on_hit_tokens": on_hit,
            "prompt_floor": prompt_floor,
            "expected_finish_floor": finish_floor,
            "delta": delta,
            "prompt_len": len(pt),
            "gen_len": len(gt),
            "page": self.page,
            "on_text": on_r["text"][:60],
        }
        v = Verdict(self.name, into_gen, summary)
        if not into_gen:
            v.reason = (
                f"on_hit={on_hit} did not reach finish_floor={finish_floor} "
                f"(prompt_floor={prompt_floor}) — reuse stopped at the prompt "
                f"boundary; decode-region reuse inactive"
            )
        return v
