#!/usr/bin/env python3
"""Generate the canonical long-agent bench dataset (bench_throughput.py --prompts-jsonl).

Usage: gen_bench_prompts.py [out.jsonl] [count] [target_tokens] [output_tokens]

One prompt = one agent context at `target_tokens`: system + tool schemas, then
repeated (ask -> tool_call -> tool output -> summary) rounds, then the live
question. Every doc gets a unique header and per-round indices, so no two
prompts share a long prefix — prefix-cache reuse cannot mask prefill cost.
`count` must be >= the highest benched concurrency.

Tokens are estimated at CHARS_PER_TOKEN; the run confirms the real p50 against
`usage.prompt_tokens` (bench spec §3.3).
"""
import json
import sys

# Measured on this agent shape (Qwen3.6 tokenizer): code, JSON and log lines
# tokenize far denser than prose, which runs ~3.6.
CHARS_PER_TOKEN = 2.80

out = sys.argv[1] if len(sys.argv) > 1 else "bench-agent-32k-64.jsonl"
count = int(sys.argv[2]) if len(sys.argv) > 2 else 64
target_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 32768
output_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 256

TOOLS = """Available tools:
- read_file(path: str, start: int, end: int) -> str
- grep(pattern: str, glob: str) -> list[str]
- run(cmd: str, timeout_s: int) -> {stdout: str, stderr: str, code: int}
- edit(path: str, old: str, new: str) -> {applied: bool, hunks: int}
"""

ASKS = [
    "trace where the {sym} counter is incremented and whether it can go negative",
    "the {sym} path allocates per call — find the reuse point and confirm it is hot",
    "check whether {sym} is still referenced after the module split, or is it dead",
    "{sym} reads a stale value under concurrency; find the write that is not fenced",
    "the {sym} fallback fires more than expected — locate the condition selecting it",
]
SYMS = ["queue_depth", "slot_epoch", "page_table", "accept_ratio", "carry_state",
        "retry_budget", "flush_cursor", "lease_token", "chunk_offset", "drain_mark"]

CODE = """    fn {sym}_{i}(&mut self, budget: usize) -> Result<usize> {{
        let mut used = 0;
        for entry in self.entries.iter_mut().take(budget) {{
            if entry.epoch != self.epoch {{
                entry.reset(self.epoch);
                continue;
            }}
            used += entry.drain(self.cursor)?;
        }}
        self.cursor = self.cursor.wrapping_add(used as u64);
        Ok(used)
    }}
"""

LOG = ("[{i:05d}] worker={w} phase={p} depth={d} lat_ms={lat}.{frac} "
       "hits={h} misses={m} evicted={e} residency={r}%\n")


def round_block(d: int, i: int) -> str:
    sym = SYMS[(d + i) % len(SYMS)]
    path = f"crates/core/src/{sym}/stage_{d}_{i}.rs"
    logs = "".join(
        LOG.format(i=i * 16 + k, w=(i + k) % 8, p=("scan", "fold", "emit")[k % 3],
                   d=(i * 7 + k) % 64, lat=(i + k) % 90, frac=(i * 3 + k) % 10,
                   h=i * 13 + k, m=(i + k) % 31, e=(i * 5 + k) % 17,
                   r=40 + (i + k) % 55)
        for k in range(6)
    )
    return (
        f"\n[round {i} / doc {d}]\n"
        f"user: {ASKS[i % len(ASKS)].format(sym=sym)}\n"
        f'assistant: {{"tool": "grep", "pattern": "{sym}", "glob": "crates/**/*.rs"}}\n'
        f"tool: {path}:{i * 4 + 11}, {path}:{i * 4 + 57}\n"
        f'assistant: {{"tool": "read_file", "path": "{path}", '
        f'"start": {i * 4}, "end": {i * 4 + 40}}}\n'
        f"tool:\n{CODE.format(sym=sym, i=i)}{logs}"
        f"assistant: {sym} is written at {path}:{i * 4 + 11} under the epoch guard "
        f"and read at :{i * 4 + 57} without one — round {i} of doc {d} narrows it "
        f"to the drain cursor, not the entry epoch.\n"
    )


target_chars = int(target_tokens * CHARS_PER_TOKEN)
with open(out, "w") as f:
    for d in range(count):
        text = (
            f"Session {d}: agent run on repository shard {d * 7 % 100}, "
            f"transcript volume {d + 1}.\n"
            "You are a coding agent working in a large Rust repository. Use the "
            "tools, cite file:line, and do not guess.\n" + TOOLS
        )
        i = 0
        while len(text) < target_chars:
            i += 1
            text += round_block(d, i)
        # Ends on the assistant cue, not on punctuation: the runner strips the
        # prompt, and a raw completion that ends mid-sentence answers EOS.
        text += (
            f"\nuser: given every round above, name the single write in session {d} "
            "that must take the epoch guard, and say why the read path is still "
            "correct without it.\nassistant:"
        )
        f.write(json.dumps({"text": text, "output_tokens": output_tokens}) + "\n")

print(f"wrote {count} agent contexts x ~{target_tokens} tok "
      f"(~{target_chars} chars @ {CHARS_PER_TOKEN} c/tok) -> {out}")
