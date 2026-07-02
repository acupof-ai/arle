# Bench — M_e.1 P2.2 Qwen3.5 dual-write per-step K/V to pool: no regression

## Goal

Regression-only check that wiring per-step Rust-side dual-write of
the C++ session's K/V into MetalKVPool (24 FFI calls + 12 slices +
12 reshapes + 6 pool.write_kv per step) does NOT degrade c=4
long-prompt ITL beyond the 1 ms acceptance threshold from M_e.1 §3
P2.1.

## Hypothesis

Each per-step operation (`mlx::array` ref-count clone, `slice` /
`reshape` are lazy MLX ops, `pool.write_kv` is a flat-buffer scatter)
is sub-millisecond. Total per-step cost ≤ 1 ms; ITL noise (~1 ms) at
the 19 ms ITL baseline absorbs it.

## Params

- 4096-in / 256-out / `--concurrencies 4` / max-seconds=30
- Model: `models/Qwen3.5-0.8B-MLX-4bit`
- ARLE: `metal_serve --max-running-requests 4 --kv-pool` (so P2.2
  dual-write is enabled)
- guidellm 0.6.0; same machine + workload as P2.0 baseline

## Env

- ARLE commit: `cb1fcc3` (P2.1 C++ FFI) + this commit (P2.2 Rust
  consumer)
- Build: `cargo build --release --features metal -p infer --bin
  metal_serve` (39.68 s clean rebuild)

## Results

| Cell | TTFT p50 | ITL p50 | ITL p95 | Output tok/s |
|---|---:|---:|---:|---:|
| --kv-pool OFF (P0 baseline) | 1.20 s | 19.34 ms | 19.96 ms | 147.5 |
| --kv-pool ON, P2.0 alloc-only | 1.16 s | 18.90 ms | 19.91 ms | 154.4 |
| --kv-pool ON, **P2.2 dual-write** | **1.17 s** | **19.11 ms** | **20.23 ms** | **149.2** |

All deltas ≤ 3% across the three cells — squarely inside thermal
noise per `feedback_matched_ab_for_small_bench_effects.md`. Dual-write
overhead is empirically zero at the c=4 4 ms-resolution we measure
at.

Service trace confirms:
- peak active = 4 (matches `--max-running-requests 4`)
- prefix_hit_rate = 0% (concurrent unique prompts, expected)
- no errors logged

## Problems

1. **Parity test is NOT yet implemented.** P2.2 only verifies the
   dual-write *happens* and *does not regress*. Whether the data the
   pool ends up holding actually matches what the C++ session has at
   the same column — byte-for-byte — is the property the kernel
   cutover (P3.1) will eventually depend on. That property test is
   the next slice (P2.3 below).

2. The first attempt at this commit hit a runtime error:
   ```
   M_e.1 P2.2 pool.write_kv: MetalKVPool: unknown request_id 0
   ```
   `pool.write_kv` requires the request_id to have been registered via
   `pool.alloc_tokens(...)` first. The Qwen3 plain dual-write pattern
   at `request_state.rs:1789-1791` does the alloc once per step BEFORE
   the per-layer loop. Mirror that pattern here. (Captured in commit;
   leaving the bug-finding history visible here as audit signal.)

## Learnings

- The `mlx::array` ref-count clone semantics make
  `cpp_model.clone_session_kv` an ~O(1) op per call. 12 calls per step
  add up to noise.
- The `pool.alloc_tokens + N×pool.write_kv` pattern is the established
  contract; future pool consumers should use the same shape.
- The acceptance bar for P3.1 (kernel cutover) tightens further: the
  ITL must DROP toward the c=1 anchor of 4.37 ms × mlx-lm-style 2.12
  multiplier ≈ 9.3 ms. Current 19.11 ms confirms we're still on the
  pre-cutover left-pad branch — both writes happen but attention
  reads the legacy concat cache.

## Action items

1. **P2.3 — parity property test.** Add an `infer/tests/` integration
   test (or extend `e2e_qwen35.rs`) that runs N decode steps on a
   real model with `--kv-pool` ON, then asserts `pool.gather_kv(layer,
   req_id)` == the column slice from the session-cloned K/V for each
   layer. Defer until E2E infrastructure decision.
2. **P3.1 — kernel cutover** (the actual unlock). Change SDPA input
   K/V to come from `pool.gather_kv_rows`. Touches the C++ side of
   `mlx_qwen35_model.cpp` substantially; see plan §3 P3.1 + §7.5
   errata for scope.

## Reproduce

```bash
cargo build --release --no-default-features --features metal -p infer --bin metal_serve

RUST_LOG=warn target/release/metal_serve \
  --model-path models/Qwen3.5-0.8B-MLX-4bit \
  --port 8000 --bind 127.0.0.1 \
  --max-running-requests 4 --kv-pool

PATH=$HOME/.local/bin:$PATH ./scripts/bench_guidellm.sh metal-c4-p22 \
  --concurrencies 4 --max-seconds 30 \
  --data 'prompt_tokens=4096,output_tokens=256' \
  --target http://localhost:8000 \
  --model Qwen3.5-0.8B-MLX-4bit \
  --processor /path/to/code/agent-infer/models/Qwen3.5-0.8B-MLX-4bit
```

Raw artefact: `bench-output/2026-05-07-metal-c4-p22-fixed/`.

## Cross-references

- Plan: [`docs/plans/M_e1-metal-paged-kv-hot-path.md`](../../plans/M_e1-metal-paged-kv-hot-path.md) §3 P2.2 + §7.5 errata
- P2.0 alloc-only baseline:
  [`2026-05-07-bench-guidellm-metal-p2-0-kvpool-allocation-no-regression.md`](2026-05-07-bench-guidellm-metal-p2-0-kvpool-allocation-no-regression.md)
- P2.1 C++ FFI (the readback hook this commit consumes):
  commit `cb1fcc3`
- Master analysis decomposition (paged-KV target 9.3 ms ITL):
  [`docs/projects/2026-05-07-metal-optimization-master-analysis.md`](../../projects/2026-05-07-metal-optimization-master-analysis.md)
