# Metal Paged KV Read Path + INT8 Decode Default

Date: 2026-06-12
Status: implemented for single-token Metal decode; batch/verify remain cache-owned
read paths.

## Goal

Make Metal single-token decode use the explicit paged-prefix K/V read source by
default for both supported KV dtypes:

- BF16 KV cache: pass per-full-attention-layer K/V prefix tensors.
- INT8 KV cache: pass per-full-attention-layer K/V q/scale/bias triples.

The C++ session still owns all fresh K/V writes. Only the SDPA prefix read source
changes.

## Why Default Is Acceptable

`MetalKvCacheDtype::default()` is INT8, so a BF16-only paged read path would
leave the default runtime on the legacy source. The implementation now covers
both BF16 and INT8 for the actual reachable Metal decode path, and keeps an
operator rollback:

```text
INFER_METAL_PAGED_KV_READ=0
```

Defaulting a non-paged path in parallel is not useful today. The Metal executor
is already backed by the host paged-KV seam and currently exposes
`max_rows_per_step() == 1` and `max_live_requests() == 1`; the active runtime
path is scalar session decode, not multi-row batch decode.

## Implemented Code Path

Rust dispatch:

```text
RealMetalExecutor::submit_decode / prequeue_decode
  -> step_session_decode
    -> MetalSlotState::bf16_prefix_read_inputs
       or MetalSlotState::int8_prefix_read_inputs
    -> CppQwen35Model::step_session_paged_bf16
       or CppQwen35Model::step_session_paged_int8
```

C++ bridge:

```text
qwen35_compiled_step_session_paged
  -> stores per-step paged prefix arrays in current_paged_* state
  -> keeps recent prefix arrays alive across lazy pipelined eval
  -> forward_impl
    -> full_attn_step / full_attn_step_int8
      -> slice_update session cache for fresh token
      -> concatenate explicit prefix + current token K/V for attention
      -> MLX scaled_dot_product_attention
```

Files:

- `crates/infer-metal/src/executor.rs`
- `crates/infer-metal/src/qwen35.rs`
- `crates/infer-metal/src/lib.rs`
- `crates/mlx-sys/src/lib.rs`
- `crates/mlx-sys/src/mlx_qwen35_model.cpp`
- `crates/infer-api/examples/metal_kv_memory_probe.rs`

## INT8 Layout Contract

Rust slot layout remains one sextet per full-attention layer:

```text
K.q, K.scale, K.bias, V.q, V.scale, V.bias
```

`int8_prefix_read_inputs(cache_len)` slices each live prefix to `[0, cache_len)`
and sends two flat triple arrays to C++:

```text
K side: K.q, K.scale, K.bias, repeated per full-attn layer
V side: V.q, V.scale, V.bias, repeated per full-attn layer
```

C++ validates every prefix array before use:

- rank 4
- shape `[B, n_kv_heads, cache_pos, tail_dim]`
- dtype `uint32` for packed q
- dtype `bfloat16` for scale/bias
- `B == 1`, `S == 1` for this path

## Deliberate Deletion

The old `qwen35_compiled_step_batch_paged` FFI stub was deleted. It accepted
paged arguments but ignored them, which made the support surface look broader
than the runtime reality. Batch and verify entrypoints still use their existing
cache-owned read source until a real multi-row Metal lane exists and has its own
mask/cache-position contract.

## Risks

- **Lazy lifetime:** MLX graph execution is lazy, and pipeline prequeue can build
  the next step before the previous step is fully materialized. Mitigation:
  `paged_input_keepalive_history` keeps recent prefix arrays alive across
  pipelined evals and is reset on session begin.
- **INT8 layout drift:** a wrong sextet order would silently feed scale/bias as
  q or vice versa. Mitigation: Rust dtype checks plus C++ dtype/shape checks.
- **Scope overclaim:** prefill, batch, and verify have different mask and
  per-row cursor semantics. They are intentionally not defaulted through this
  scalar session path.
- **Performance overclaim:** 9B smoke proves reachability and default safety,
  not a Qwen3.6 production performance win.

## Verification Gates

Local Apple Silicon, `mlx-community/Qwen3.5-9B-MLX-4bit`, 256-token prompt
target, 4 decode tokens, `--memory-budget-gib 12 --low-impact false`.

| dtype | env | hits | fallbacks | wall_ms | output prefix |
| --- | --- | ---: | ---: | ---: | --- |
| INT8 | default on | 4 | 0 | 818.2359 | ` Metal int8 KV` |
| BF16 | default on | 4 | 0 | 798.1721 | ` Metal int8 KV` |
| INT8 | `INFER_METAL_PAGED_KV_READ=0` | 0 | 0 | 813.1649 | ` Metal int8 KV` |

Current-HEAD risk replay (`31da38c1`), 512-token prompt target, 32 decode
tokens, one warmup run, two measured runs:

| dtype | env | measured hits | fallbacks | avg ms/token | output prefix |
| --- | --- | --- | ---: | ---: | --- |
| INT8 | default on | `32, 32` | 0 | 44.2083 | ` Metal int8 KV...` |
| INT8 | `INFER_METAL_PAGED_KV_READ=0` | `0, 0` | 0 | 44.0500 | ` Metal int8 KV...` |

Qwen3.6-35B-A3B was attempted with the same probe shape but blocked by the
Metal resource guard on the current desktop memory state: 48.0 GiB total,
18.2 GiB available, 441 MiB swap used, 25 GiB fixed requirement. Do not treat
the 35B gate as passed until it runs on a host state with enough headroom.

Build/test gates:

- `cargo check -p infer-api --release --no-default-features --features metal,no-cuda --example metal_kv_memory_probe`
- `cargo test -p infer-metal --release --no-default-features --features metal -- --nocapture`
- `cargo clippy -p infer-metal --release --no-default-features --features metal -- -D warnings`
- `cargo clippy -p infer-api --release --no-default-features --features metal,no-cuda --example metal_kv_memory_probe -- -D warnings`
- `git diff --check`

## Next Work

The next licensing step remains a canonical Qwen3.6-35B-A3B guideLLM/server
benchmark with enough decode tokens to measure wall-clock impact, on a host
state that clears the Metal resource guard. A custom INT8 attention kernel is
not licensed by this change alone; it still needs component evidence that
full-prefix dequantize-before-SDPA is the wall-clock bottleneck.
