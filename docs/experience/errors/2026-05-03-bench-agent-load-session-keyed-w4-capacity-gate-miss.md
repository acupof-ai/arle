# Session-Keyed W4 Canonical Gate Miss

## Context

A3/A6/B left canonical W4 resume at ~0.35-0.38% avoided-prefill because
lookup still stopped at the chat-template token-32 divergence. Commit
`d0a6f989` added a scheduler-local `session_id -> SessionSlot` side index so
same-session admission can bypass token-walk lookup and find committed KV
blocks by session key.

Validation used the local L4 box:

- GPU: NVIDIA L4 24GB
- Model: `models/default` -> `Qwen3-4B`
- Server:
  `./target/release/infer --model-path models/default --port 8000 --num-slots 8 --max-seq-len 12288 --kv-cache-dtype fp8 --mem-fraction-static 0.85 --t1-host-pinned-high-water 0.98 --t1-host-pinned-low-water 0.95 --t1-host-pinned-capacity-mb 32768 --t1-host-pinned-min-prompt-tokens 4096`
- Trace:
  `python3 scripts/bench_agent_trace.py --workload agent-w4-tool-resume --server http://127.0.0.1:8000 --label a-session-keyed-w4-canonical --out bench-results/a-session-keyed-w4-canonical/results.json --trace-out bench-results/a-session-keyed-w4-canonical/trace.jsonl`

## Root Cause

The side index fixed the lookup-addressing flaw but exposed an unsolved
capacity/lifecycle problem. SessionSlot membership protected every warmup
session's sealed blocks, so canonical W4 filled T1 before the resume phase.
With 128 warmup sessions at ~8k tokens each and 16-token blocks, the protected
working set is far larger than the L4 T0+T1 budget.

The run reached only warmup request 94/128. The server repeatedly logged:

```text
failed to demote block ... to host tier: host pinned tier has no leaf eviction headroom
```

and then fell back to dropping GPU blocks for immediate T0 headroom. Request 94
finished with 0 generated tokens. The benchmark client never produced
`results.json`, so W4 resume avoided-prefill was not measurable.

## Fix

No follow-up implementation was made in this tick. The committed side index is
a correct substrate for bypassing token-walk, but it needs a bounded
session-slot eviction policy before it can pass canonical W4:

- keep active session holds non-evictable;
- allow inactive SessionSlots to be evicted under T1 pressure by session LRU or
  admission priority;
- preserve the side-index read path for retained sessions;
- record evicted-slot accounting so W4 can distinguish lookup failure from
  capacity eviction.

## Rule

Session-keyed lookup cannot mean unbounded session retention. Any SessionSlot
side index must ship with an explicit inactive-slot pressure policy; otherwise
it turns the W4 token-walk miss into a T1 capacity deadlock before resume.
