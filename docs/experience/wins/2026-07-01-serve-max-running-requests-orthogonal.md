# Serve max-running-requests is orthogonal to hot workspace capacity

## Context

`arle serve --max-running-requests` still flowed into `EngineLoadConfig.num_slots`.
For DSv4 and Qwen3.6 this changed executor hot-workspace slots and therefore
slot-state/scratch budgeting. A low running cap used for L2/L3 pressure tests
was not a pure scheduler shape; it also changed the KV budget.

## What Worked

Serve now exposes one concurrency knob: `--max-running-requests`. The old
`--num-slots` alias is removed from the serve CLI and current serve scripts.

The builder derives internal hot-workspace slots as
`max(default/internal num_slots, max_running_requests)`, then the backend budget
clamps capacity if needed. The scheduler still enforces
`max_running_requests` as the active-request cap. `--low-impact` also only sets
`max_running_requests=1`; it no longer shrinks executor capacity.

## Verification

Local:

```bash
CUDARC_CUDA_VERSION=12090 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib
cargo test -p infer-core --release --lib
cargo test -p infer-server --release --lib
cargo test -p cli --release --no-default-features --features cpu,no-cuda --lib
bash -n scripts/*.sh
```

Results: all passed. The cli test suite has 206 passing tests and now asserts
`--num-slots` is rejected by `arle serve`.

## Rule

Public serve knobs must control one layer only. `max_running_requests` is the
admission target; hot-workspace capacity is derived internally from model,
budget, and that target, then clamped by the backend.
