# Metal Cooperative Scheduler Seams

## Goal

Make `arle serve --backend metal` smoother on local macOS by keeping the service
and scheduler layers backend-neutral while moving Metal-specific limits below
the executor seam.

## Hypothesis

Metal's current stall/error risks come from backend facts that the shared
scheduler cannot see: one executable row per MLX step, no live SSD recall path,
Metal-owned prefix snapshots outliving host radix eviction, and non-greedy
sampling forcing synchronous host logits reads. Expressing those facts as seam
capabilities and cooperative budgets should prevent unsupported plan shapes and
avoid default per-token D2H stalls without adding Metal types above the seam.

## Params

- Backend: Metal/MLX executor.
- Scheduler: shared `infer_core::Engine`.
- Low-impact config: `num_slots=1`, `total_pages<=1024`,
  `max_prompt_tokens<=8192`, `max_total_tokens<=8192`,
  `chunked_prefill_size<=32`, plus cooperative `StepBudget`.
- Spec decode: CUDA checkpoint-native MTP only; Metal speculative routes fail
  closed at CLI/API validation.
- Model bench: not run in this tranche; Qwen3.6/guidellm remains pending.

## Env

- Host: Apple M4 Pro.
- Date: 2026-06-11 16:01 CST.
- Build profile: `--release`.

## Results

| Check | Result |
| --- | --- |
| `cargo test -p infer-core --release` | PASS, 35 passed / 1 ignored |
| `cargo test -p cli --release --no-default-features --features metal,no-cuda` | PASS, 130 passed |
| `cargo test -p infer-metal --release --features metal` | PASS, 7 passed |
| `cargo test -p infer-api --release --no-default-features --features metal,no-cuda` | PASS, 11 passed |
| `CUDARC_CUDA_VERSION=12060 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib` | PASS |
| `cargo test -p agent-bench --release --no-default-features --features metal,no-cuda` | PASS, 6 passed / 9 ignored |
| `cargo test -p agent-infer --release --no-default-features --features cpu,no-cuda,cli` | PASS, 5 passed |
| `cargo clippy -p infer-core -p infer-metal -p infer-api -p cli -p agent-bench --release --no-default-features --features metal,no-cuda -- -D warnings` | PASS |

## Problems

- No Qwen3.6 real-model `guidellm` run was executed in this tranche, so this is
  a seam/correctness win, not a wall-clock smoothness win.
- SSD KV recall is still not implemented in the rewrite serve path. The correct
  stats surface remains `ssd_recall.available=false`; no fake recall metric was
  introduced.
- `StepBudget.max_micros` is still a soft policy field. The implemented hard
  clamp is token-count based, with cooperative periodic yield for low-impact
  Metal serve.

## Learnings

- Backend row capacity belongs in `BackendExecutor`, not as a late executor
  error. Once Metal reports `max_rows_per_step=1`, the shared scheduler admits
  one active slot and never builds the unsupported multi-row shape.
- Prefix cache eviction must notify backend mirrors. Host radix LRU eviction now
  calls `release_prefix_pages`, and Metal drops both page mirrors and any prefix
  snapshots containing those page ids.
- Non-greedy Metal sampling cannot default to host logits materialization on a
  desktop path. The default is constrained to device greedy argmax; the old
  blocking D2H sampler is opt-in via `INFER_METAL_HOST_SAMPLING=1`.
