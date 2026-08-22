# --max-running-requests caps the executor slot budget

> Status: Shipped — pod A/B accepted 2026-07-17.

## Context

`EngineLoadConfig::hot_workspace_slots` returned `num_slots.max(cap)`, so a set
`--max-running-requests` never shrank the requested slot count. DSv4's
demand-paged joint solve maximizes slot count first, leaving the shared comp
pool the remainder: measured 59 slots x 338MB = 19.9GB reserved vs a 640MB pool
(83968 tokens), while a c32 workload needs ~115k tokens — chronic
oversubscription. Slots above the scheduler `running_cap` are unusable by
construction (`min(max_running_requests, num_slots)`, infer-core/src/lib.rs).

## What Worked

Commit 77e0d1d5d: a set cap IS the requested executor slot budget
(`max_running_requests.unwrap_or(num_slots).max(1)`); unset keeps the
`num_slots` auto-ceiling. Predicted at cap=32: ~9.1GB freed -> pool ~1.28M
tokens (~15x). Bench numbers pending the pod A/B.

## Rule

Post-#154-3b DSv4 slots TRADE against comp-pool tokens — "the VRAM budget
always binds first" no longer licenses over-provisioning the slot request.

## Results (pod A/B, 4xH20 TP=4/EP=4 eager, ce5d0b833 build, native runner 256-out)

Slot lines: default = `num_slots 59, shared comp capacity 83968 tokens`;
`--max-running-requests 32` = `num_slots 32, shared comp capacity 1048576
tokens (16384 engine pages)` — **12.5x comp capacity at the same 20582 MB
budget** (hits the 32 x max_seq cap).

c=32, 300 s, bench-prompts-64.jsonl:

| arm | complete | TTFT p50/p99 | ITL p50/p99 |
| --- | --- | --- | --- |
| default (59 slots, oversubscribed) | 74 | 67.2 / 122.0 s | 85.6 / 1955 ms |
| cap 32 | **121** | 83.5 / 91.9 s | 134.6 / **308 ms (-84% p99)** |

Grid c1/4/8/16 with the cap: TTFT
p50 better at c4 (1194 vs 1532 ms) and c8 (2397 vs 2726 ms). The cap costs
nothing at low concurrency and removes preempt-recompute storms at c32
(default arm logged 192 KV-overflow preemptions; cap arm zero).

## Rule

Post-#154-3b, slots and comp-pool tokens trade against each other in one
budget: slots beyond the offered concurrency are pure token-capacity waste.
Set `--max-running-requests` to the real concurrency target on DSv4 serves.
Raw: pod `bench-output/2026-07-16-ab-*`.
