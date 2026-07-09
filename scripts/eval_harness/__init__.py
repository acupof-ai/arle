"""Engine correctness harness — base classes and shared utilities.

Each gate is a `Gate` subclass that implements `run()` → verdict dict.
The runner orchestrates gates and produces a unified pass/fail report.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

BASE = "http://127.0.0.1:" + os.environ.get("PORT", "18189")
MODEL = os.environ.get("MODEL", "x")
TEMPLATE = os.environ.get("TEMPLATE", "raw")

FILLER_SENTS = [
    "The river flowed gently past the old stone bridge.",
    "Mountains rose sharply against the pale morning sky.",
    "She opened the wooden door and stepped into the hall.",
    "The market was full of fruit, spices, and fresh bread.",
    "A long train crossed the wide green valley at dawn.",
    "Children played near the fountain in the city square.",
    "The library held thousands of dusty leather books.",
    "Rain fell softly on the roof throughout the night.",
    "An old cat napped in the patch of warm sunlight.",
    "The ship sailed toward the distant horizon at dusk.",
    "Birds gathered on the telephone wire before migration.",
    "The kitchen smelled of cinnamon and freshly baked bread.",
    "A narrow path wound through the dense pine forest.",
    "The clock tower chimed once across the quiet village.",
    "Leaves scattered across the cobblestone street in autumn.",
    "The fisherman mended his nets by the lantern light.",
    "A stone wall separated the orchard from the meadow.",
    "The telescope pointed toward the brightest evening star.",
    "Steam rose from the cup on the cold windowsill.",
    "The hiker reached the summit before the clouds rolled in.",
]


def api_chat(prompt: str, max_tokens: int = 32, timeout: int = 600) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    resp = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    return {
        "text": resp["choices"][0]["message"]["content"],
        "prompt_tokens": resp.get("usage", {}).get("prompt_tokens"),
        "dt": time.time() - t0,
    }


def api_completions(prompt: str, max_tokens: int = 32, timeout: int = 600) -> dict:
    if TEMPLATE == "dsv4":
        wrapped = (
            "<｜begin▁of▁sentence｜>You are a helpful assistant."
            "<｜User｜>" + prompt + "<｜Assistant｜></think>"
        )
    elif TEMPLATE == "qwen3_nonthink":
        wrapped = (
            "<|im_start|>user\n" + prompt + "<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
    else:
        wrapped = prompt
    body = {"model": MODEL, "prompt": wrapped, "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(
        BASE + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    resp = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    return {
        "text": resp["choices"][0]["text"],
        "prompt_tokens": resp.get("usage", {}).get("prompt_tokens"),
        "dt": time.time() - t0,
    }


def query(prompt: str, max_tokens: int = 32) -> dict:
    if TEMPLATE == "raw":
        return api_chat(prompt, max_tokens)
    return api_completions(prompt, max_tokens)


def get_stats() -> dict:
    return json.loads(urllib.request.urlopen(BASE + "/v1/stats", timeout=30).read())


def stat_delta(before: dict, after: dict, key: str) -> int:
    return after.get(key, 0) - before.get(key, 0)


def build_doc(target_tokens: int, needle: str, needle_pos: float = 0.88) -> tuple[str, int]:
    """Document with a needle embedded at needle_pos fraction.

    Returns (doc, approx_token_count).
    """
    toks_per_sent = 16
    n = max(8, target_tokens // toks_per_sent)
    k = max(1, int(n * needle_pos))

    pre = " ".join(f"Note {i+1}: {FILLER_SENTS[i % len(FILLER_SENTS)]}" for i in range(k))
    block = f"\n\nImportant: the secret access code is {needle}. Memorize it.\n\n"
    post = " ".join(
        f"Note {i+1}: {FILLER_SENTS[(i + k) % len(FILLER_SENTS)]}" for i in range(n - k)
    )
    doc = pre + block + post
    approx = len(doc.split()) * 4 // 3
    return doc, approx


def mutate_prefix(doc: str) -> str:
    return doc.replace("Note 1:", "Memo 1:", 1)


@dataclass
class Verdict:
    name: str
    passed: bool
    summary: dict = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict:
        d = {"name": self.name, "pass": self.passed}
        if self.reason:
            d["reason"] = self.reason
        d.update(self.summary)
        return d


class Gate(ABC):
    """Base correctness gate. Subclass and implement run()."""

    name: str = "base"

    def __init__(self, runs: int = 3):
        self.runs = runs

    @abstractmethod
    def run(self) -> Verdict:
        ...

    def check_server(self) -> None:
        try:
            get_stats()
        except Exception as e:
            raise RuntimeError(f"server unreachable at {BASE}: {e}") from e


class GateRunner:
    """Orchestrates multiple gates, prints unified report."""

    def __init__(self, gates: list[Gate]):
        self.gates = gates

    def run_all(self) -> list[Verdict]:
        verdicts = []
        for gate in self.gates:
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"GATE: {gate.name}", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)
            try:
                gate.check_server()
                v = gate.run()
            except Exception as e:
                v = Verdict(gate.name, False, reason=str(e))
            verdicts.append(v)
            status = "PASS" if v.passed else "FAIL"
            print(f"\n  [{status}] {v.name}", file=sys.stderr)
            if v.reason:
                print(f"         {v.reason}", file=sys.stderr)
        return verdicts

    def report(self, verdicts: list[Verdict]) -> dict:
        passed = sum(1 for v in verdicts if v.passed)
        total = len(verdicts)
        return {
            "pass": passed == total,
            "gates_passed": f"{passed}/{total}",
            "verdicts": [v.to_dict() for v in verdicts],
        }
