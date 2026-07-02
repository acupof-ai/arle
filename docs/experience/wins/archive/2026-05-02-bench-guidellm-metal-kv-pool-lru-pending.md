# Metal KV Pool LRU Foundation — guidellm pending, metal, 2026-05-02

## Goal

- Track the bench obligation for the Metal `kv_pool` LRU/watermark foundation.

## Hypothesis

- The current tranche is structural and should not move runtime numbers because
  the new LRU selection API is not yet wired into `run_metal_scheduler_runtime`
  spill/readmission.

## Command

```bash
scripts/bench_guidellm.sh metal-kv-pool-lru-foundation \
  --target http://localhost:8000 \
  --model Qwen3.5 \
  --processor <metal-model-path>
```

Invoked via: pending; run after P0-1/P1-2 wires the selector into the Metal
runtime.

## Environment

- **Backend:** metal
- **Model:** pending Qwen3.5 Metal model
- **Hardware:** pending Apple Silicon bench host
- **Commit:** pending
- **Feature set:** `cargo build --release --no-default-features --features metal,no-cuda`
- **Non-default flags / env vars:** pending
- **Server launch:** `scripts/start_metal_serve.sh <model> <port>`

## Canonical params

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`
- Wrapper: `scripts/bench_guidellm.sh metal-kv-pool-lru-foundation`

## Results

Pending. No runtime bench was run for this tranche because the diff only adds
the Metal pool accounting API and unit tests; it does not connect the selector
to scheduler spill/admission.

## Problems

- Bench remains required before shipping a runtime-connected Metal T2 path.

## Learnings

- Keep structural KV-pool accounting isolated until the runtime spill path can
  be benchmarked with repeated-prefix traffic.

## Delta vs baseline

- **Baseline:** pending.
- **Delta table:** pending.

## Artefacts

- Raw: pending.
- CSV: pending.
- HTML: pending.
- Service trace: pending.

## Notes

- Code changed since baseline: `infer/src/backend/metal/kv_pool.rs` adds slot
  access ticks, high/low watermark reclaim sizing, and active-slot-protected
  LRU candidate selection.
- Follow-ups: wire this into Metal T2 disk persistence and replace this pending
  entry with a real before/after guidellm snapshot.
