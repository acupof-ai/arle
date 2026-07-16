# --max-running-requests caps the executor slot budget

> Status: pending-remote — pod A/B queued (DSv4 8xH20, c32 comp-pool sizing).

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
