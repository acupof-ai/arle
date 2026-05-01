# Metal SSD KV Options Foundation — guidellm pending, metal, 2026-05-02

## Goal

- Track the bench obligation for the P0-1A Metal SSD KV persistence options
  foundation.

## Hypothesis

- This tranche should not move default runtime numbers because the new
  `MetalKvDiskOptions` plumbing is inert unless `--kv-disk-dir` is passed, and
  no snapshot payload export/import is wired yet.

## Command

```bash
scripts/bench_guidellm.sh metal-kv-disk-options \
  --target http://localhost:8000 \
  --model Qwen3.5 \
  --processor <metal-model-path>
```

Invoked via: pending; run after P0-1A connects snapshot persistence to the
Metal Qwen3.5 prefix runtime.

## Environment

- **Backend:** metal
- **Model:** pending Qwen3.5 Metal model
- **Hardware:** pending Apple Silicon bench host
- **Commit:** pending
- **Feature set:** `cargo build --release --no-default-features --features metal,no-cuda`
- **Non-default flags / env vars:** pending `--kv-disk-dir <dir>`
- **Server launch:** `scripts/start_metal_serve.sh <model> <port> -- --kv-disk-dir <dir>`

## Canonical params

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`
- Wrapper: `scripts/bench_guidellm.sh metal-kv-disk-options`

## Results

Pending. No runtime bench was run for this tranche because the diff only adds
configuration validation and CLI plumbing for future Metal SSD KV persistence.

## Problems

- Bench remains required before any snapshot import/export path can be claimed
  as a TTFT improvement.

## Learnings

- Keep Metal SSD KV options explicit and disabled by default until the Qwen3.5
  snapshot format and disk index are connected.

## Delta vs baseline

- **Baseline:** pending.
- **Delta table:** pending.

## Artefacts

- Raw: pending.
- CSV: pending.
- HTML: pending.
- Service trace: pending.

## Notes

- Code changed since baseline: `MetalKvDiskOptions` and `--kv-disk-*` flags are
  now available to `metal_serve`, `metal_request`, and `metal_bench`.
- Follow-ups: add MLX array byte export/import, then persist
  `Qwen35PrefixSnapshot` values through `DiskStore`.
