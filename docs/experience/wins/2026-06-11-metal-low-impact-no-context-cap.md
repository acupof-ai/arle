# Metal low-impact no longer caps context capacity

## Goal

Make `arle serve --backend metal --low-impact` reduce foreground scheduling
impact without silently shrinking the usable KV/context capacity.

## Hypothesis

The macOS stall fix should come from backend-visible scheduling constraints
(single-flight Metal serve, cooperative step budget, smaller prefill chunks),
not from hard caps on `total_pages`, `max_prompt_tokens`, or
`max_total_tokens`. Those caps make long-context behavior look broken and do
not address the real MLX/SSD/KV execution path.

## Params

- Backend: Metal CLI/config path.
- Low-impact now changes only `low_impact=true`, `num_slots=1`, and
  `chunked_prefill_size<=32`.
- Capacity remains the normal `EngineLoadConfig` default unless explicitly set:
  `total_pages=8192`, `page_size=16`, `max_prompt_tokens=32768`,
  `max_total_tokens=65536`.
- Current Metal KV dtype: full-attention K/V buffers in `kv_flat` are BF16
  MLX arrays. `MetalKvPool` is the backend-neutral `HostPagedKvPool` logical
  page allocator, not an int8 KV store.
- SSD tier: `--kv-ssd-path` is accepted as an explicit high-performance,
  non-preemptive SSD KV request, but serve validates the path and fails closed
  because the rewrite stack still has no active SSD recall implementation.

## Env

- Host: local Apple Silicon Mac.
- Date: 2026-06-11.
- Build profile: `--release`.

## Results

| Check | Result |
| --- | --- |
| `cargo test -p cli --release --no-default-features --features metal,no-cuda serve::tests -- --nocapture` | PASS, 18 passed |
| `cargo test -p cli --release --no-default-features --features metal,no-cuda` | PASS, 131 passed |
| `cargo test -p infer-api --release --no-default-features --features metal,no-cuda kv_ssd -- --nocapture` | PASS, 3 passed |
| `cargo test -p infer-api --release --no-default-features --features metal,no-cuda` | PASS, 10 unit + 4 adapter tests |
| `cargo test -p infer-api --release --no-default-features` | PASS, 10 unit + 4 adapter tests |
| `cargo test -p agent-infer --release --no-default-features --features cpu,no-cuda,cli` | PASS, 5 CLI smoke tests |
| `cargo clippy -p cli -p infer-api --release --no-default-features --features metal,no-cuda -- -D warnings` | PASS |
| `CUDARC_CUDA_VERSION=12060 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib` | PASS |

## Problems

- This change does not implement Metal int8 KV. The current Metal executor still
  allocates BF16 full-attention K/V and BF16/FP32 GDR recurrent state.
- This change does not implement SSD recall. The CLI/service layer now refuses
  to start with a fake SSD tier; the executor/radix layer still needs a real
  disk directory keyed by token-prefix fingerprint plus page-block
  serialization before recall can report `available=true`.

## Learnings

- `--low-impact` must mean lower scheduling impact, not smaller model/context
  semantics. Capacity clamps belong to explicit operator flags.
- SSD KV needs a backend-owned page-store and radix-visible disk directory.
  `kv-native-sys` alone is only the persistence substrate.
