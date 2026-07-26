#!/usr/bin/env python3
"""Generate the canonical long-agent bench dataset (bench_throughput.py --prompts-jsonl).

Usage: gen_bench_prompts.py [out.jsonl] [sessions] [target_tokens] [output_tokens] [turns]

One session = a `turns`-long conversation whose transcript starts at
`target_tokens` and grows by one round per turn: system + tool schemas, then
repeated (ask -> tool_call -> tool output -> summary) rounds, then the live
question. Turn k's text is a strict prefix of turn k+1's, so a served session
reuses its own KV exactly as a real agent does; only turn 0 pays the full
prefill. Sessions are mutually unique (header + per-round indices), and prompts
are emitted turn-major, so the in-flight set at concurrency C is C distinct
sessions and every reused prefix belongs to a turn that already finished.

`sessions` must be >= the highest benched concurrency, and
`--requests-per-concurrency` a multiple of `sessions`, or the tail turns never
run and the point measures cold prefill only.

Tokens are estimated at CHARS_PER_TOKEN; the run confirms the real p50 against
`usage.prompt_tokens` (bench spec §3.3).
"""
import json
import sys

# Measured on this agent shape (Qwen3.6 tokenizer): code, JSON and log lines
# tokenize far denser than prose, which runs ~3.6.
CHARS_PER_TOKEN = 2.80

# Defaults are the coding-agent trace medians (TraceLab arXiv:2606.30560,
# 4,265 Claude Code / Codex sessions): 119K prefix tokens, 875 append tokens and
# 214 output tokens per step, 8.8 steps per request. Append lands via the
# per-turn round block (~650 tok) plus the question and answer.
out = sys.argv[1] if len(sys.argv) > 1 else "bench-agent-119k-16x8.jsonl"
sessions = int(sys.argv[2]) if len(sys.argv) > 2 else 16
target_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 119000
output_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 214
turns = int(sys.argv[5]) if len(sys.argv) > 5 else 8

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


def question(d: int, k: int) -> str:
    # Ends on the assistant cue, not on punctuation: the runner strips the
    # prompt, and a raw completion that ends mid-sentence answers EOS.
    return (
        f"\nuser: turn {k} — given every round above, name the single write in "
        f"session {d} that must take the epoch guard, and say why the read path "
        "is still correct without it.\nassistant:"
    )


target_chars = int(target_tokens * CHARS_PER_TOKEN)
prompts = [[] for _ in range(turns)]
for d in range(sessions):
    transcript = (
        f"Session {d}: agent run on repository shard {d * 7 % 100}, "
        f"transcript volume {d + 1}.\n"
        "You are a coding agent working in a large Rust repository. Use the "
        "tools, cite file:line, and do not guess.\n" + TOOLS
    )
    i = 0
    while len(transcript) < target_chars:
        i += 1
        transcript += round_block(d, i)
    for k in range(turns):
        prompts[k].append(transcript + question(d, k))
        # The next turn keeps this turn's question and answer in the transcript,
        # so turn k's text stays a strict prefix of turn k+1's.
        i += 1
        transcript += question(d, k) + (
            f" the guard is on the write at :{i * 4 + 11}; the read is a single "
            "aligned load and cannot tear.\n"
        ) + round_block(d, i)

with open(out, "w") as f:
    for turn_prompts in prompts:  # turn-major: consecutive prompts = distinct sessions
        for text in turn_prompts:
            f.write(json.dumps({"text": text, "output_tokens": output_tokens}) + "\n")

print(f"wrote {sessions} sessions x {turns} turns = {sessions * turns} prompts, "
      f"turn 0 ~{target_tokens} tok (~{target_chars} chars @ {CHARS_PER_TOKEN} "
      f"c/tok), +1 round per turn -> {out}")
