# Metal paged KV read default-on — BF16 + INT8 9B reachability

## Goal

Default the Metal single-token decode path to read the attention prefix from
explicit paged K/V inputs instead of always slicing the session-owned contiguous
cache, while keeping the session cache as the write/ownership source.

## Hypothesis

For scalar Metal decode, the runtime can safely pass the live prefix to C++:

- BF16: per-full-attention-layer K/V prefix tensors.
- INT8: per-full-attention-layer K/V q/scale/bias triples.

If both dtypes produce one paged-read hit per generated token and no fallbacks on
the 9B smoke, the path is safe to default for the currently reachable Metal
single-row executor. This does not claim a production performance win.

## Params

- Model: `mlx-community/Qwen3.5-9B-MLX-4bit`
- Backend: Metal
- Prompt target: 256 tokens (`prompt_tokens=265` after tokenization)
- Decode: `max_tokens=4`, greedy default sampling
- Budget: `--memory-budget-gib 12 --low-impact false`
- Pipeline: default on (`INFER_METAL_PIPELINE=true`)
- Paged read: default on; rollback with `INFER_METAL_PAGED_KV_READ=0`

## Env

Local Apple Silicon host, 48 GiB unified memory.

The first attempted 13 GiB budget was rejected by the resource guard because
the current anti-swap budget was 12 GiB. All accepted 9B runs below use 12 GiB.

## Results

9B reachability smoke:

| mode | dtype | wall_ms | ms/token | paged hits | fallbacks | output prefix |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| default on | INT8 | 818.2359 | 204.5590 | 4 | 0 | ` Metal int8 KV` |
| default on | BF16 | 798.1721 | 199.5430 | 4 | 0 | ` Metal int8 KV` |
| opt-out | INT8 | 813.1649 | 203.2912 | 0 | 0 | ` Metal int8 KV` |

Earlier same-binary BF16 env-flip smoke before defaulting:

| mode | wall_ms | ms/token | paged hits | fallbacks | output prefix |
| --- | ---: | ---: | ---: | ---: | --- |
| `INFER_METAL_PAGED_KV_READ=0` | 830.1845 | 207.5461 | 0 | 0 | ` Metal int8 KV` |
| `INFER_METAL_PAGED_KV_READ=1` | 831.9837 | 207.9959 | 4 | 0 | ` Metal int8 KV` |

Verification:

- `cargo check -p infer-api --release --no-default-features --features metal,no-cuda --example metal_kv_memory_probe`: passed.
- `cargo test -p infer-metal --release --no-default-features --features metal int8_prefix_read_inputs -- --nocapture`: passed.
- `cargo test -p infer-metal --release --no-default-features --features metal -- --nocapture`: passed earlier in this tranche before unrelated T2 WIP entered the worktree; final rerun is blocked by that unrelated WIP.
- `cargo clippy -p infer-metal --release --no-default-features --features metal -- -D warnings`: passed.
- `cargo clippy -p infer-api --release --no-default-features --features metal,no-cuda --example metal_kv_memory_probe -- -D warnings`: passed.
- `cargo fmt --check`: passed.
- `git diff --check`: passed.

## Problems

This is a small 9B reachability smoke, not a Qwen3.6 performance license:

- One run per mode and four decode tokens.
- Qwen3.5-9B is not the canonical Metal production target; Qwen3.6-35B-A3B
  remains required before claiming production throughput or latency impact.
- Wall-clock differences are noise-level and must not be read as a win or loss.
- Prefill, batch, and verify keep their existing cache-owned read source because
  their mask and per-row cursor contracts differ from scalar session decode.
- The current working tree contains unrelated, incomplete Metal T2 changes
  (`Cargo.lock`, `crates/infer-metal/Cargo.toml`, `crates/infer-metal/src/mlx.rs`,
  and unstaged hunks in `executor.rs`). They are not part of this paged-read
  commit and currently block a final full `infer-metal` test rerun.

## Learnings

The default Metal decode dtype is INT8, so BF16-only paged read would have left
the real default path unsupported. Passing q/scale/bias triples closes that
support gap without introducing a custom Metal kernel. The old
`qwen35_compiled_step_batch_paged` stub was removed because accepting paged
arguments while ignoring them was a worse support contract than no batch-paged
symbol at all.

Next gate: canonical Qwen3.6-35B-A3B guideLLM/server benchmark with enough
decode tokens to measure wall-clock impact. A custom INT8 attention kernel still
needs separate component evidence that full-prefix dequantize-before-SDPA is
the dominant bottleneck.
