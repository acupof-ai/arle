# Qwen3.5 paged prefill OOM on L4 is headroom collapse from oversized per-request scratch

## Context

- Date: `2026-04-22`
- Model: `Qwen/Qwen3.5-4B`
- Host: NVIDIA L4 24GB
- Binary under test: `4848cd1` with local compile fix in `infer/src/metrics.rs`
- Repro bench: `bench-output/2026-04-22-2026-04-22-infer-qwen35-4b-l4-c4-4848cd1-localfix/`
- Server log: `bench-output/2026-04-22-infer-qwen35-4b-l4-c4-4848cd1-localfix-server/infer.log`

The first real long-context `guidellm` leg for Qwen3.5 failed immediately.
The warmup probe request (`2` tokens) succeeded; the first `4097`-token
prefill chunk did not.

Measured startup residency with `nvidia-smi`:

| launch config | resident HBM after startup |
|---|---:|
| `--num-slots 10 --max-seq-len 5120 --mem-fraction-static 0.88` | `19906 MiB` |
| `--num-slots 16 --max-seq-len 4608 --mem-fraction-static 0.94` | `21248 MiB` |

OOM begins on the first long prefill:

- `Request 1: chunked prefill starting (4097 effective tokens, chunk_size=4096)`
- immediately followed by
  `prefill batch failed: Alloc failed: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")`

See `bench-output/2026-04-22-infer-qwen35-4b-l4-c4-4848cd1-localfix-server/infer.log`.

## Root Cause

This is not a scheduler-admission bug and not a KV-cap bug.

The failing path is Qwen3.5's **paged prefill buffer materialization**:

1. `forward_prefill_with_pool()` calls
   `Qwen35State::prepare_paged_prefill()`.
2. `prepare_paged_prefill()` lazily allocates `PagedPrefillBuffers35`.
3. `PagedPrefillBuffers35::new()` allocates a **full-sequence scratch pack**
   for the entire `4097`-token chunk, including:
   - full-attention scratch (`q_full`, `q_prepped`, `attn_out_full`, …)
   - linear-attention scratch (`qkv`, `qkv_conv`, `z`, `gdr_out`, …)
   - MLP intermediates (`gate_out`, `up_out`, `act_out`)
   - GDR chunkwise state (`a_tril`, `chunk_state`, …)
   - FlashInfer HD256 workspace via `BatchPrefillPagedPlan::new_hd256()`

Static size estimate for the `4097`-token shape:

| component | approx size |
|---|---:|
| FlashInfer HD256 float workspace | `640 MiB` |
| GDR `chunk_state` | `130 MiB` |
| MLP `gate_out + up_out + act_out` | `216 MiB` |
| `q_full + qkv + qkv_conv` | `192 MiB` |
| remaining hidden/attention/GDR scratch | `~542 MiB` |
| **total one-request prefill scratch** | **`~1.68 GiB`** |

On the current `16 slot / 4608 seq / 0.94 mem_fraction` launch, startup
already consumes `21248 MiB`. Adding a `~1.68 GiB` one-request prefill pack
pushes the process to roughly `22.9 GiB` before allocator fragmentation,
cuBLAS / FlashInfer transient workspace, and graph-capture overhead. That is
enough to trip `CUDA_ERROR_OUT_OF_MEMORY` on the first real request.

There is a second design problem amplifying the headroom collapse:

- `prefill_forward_paged()` enables a **whole-prefill CUDA graph capture**
  when `--cuda-graph=true`.
- So the first long request is not just allocating the persistent scratch
  pack; it is also running the largest prefill shape through graph capture.

That means the current Qwen3.5 paged-prefill path is paying both:

- a very large persistent per-request scratch allocation, and
- first-use graph-capture pressure on the largest shape.

## Fix

Real fixes, in order:

1. **Shrink the Qwen3.5 paged-prefill scratch model.**
   - Alias / reuse intermediates across phases instead of keeping one full
     buffer per stage.
   - Avoid storing both full-attention and linear-attention large activations
     live at the same time.
   - Split long-prefill scratch from decode-lived state so it dies as soon as
     the prefill chunk finishes.

2. **Do not graph-capture the 4k paged-prefill path by default on L4.**
   - Decode graph capture is already valuable.
   - Prefill capture on the largest Qwen3.5 shape currently spends scarce
     HBM headroom on the worst possible step.

3. **Only after 1+2, retune launch knobs** (`num_slots`, `mem_fraction`,
   `chunked_prefill_size`) if needed.
   - Those knobs change whether the bug is visible.
   - They do not fix the oversized scratch design.

## Rule

For long-context Qwen3.5 BF16 on L4, the primary risk is not KV capacity; it
is **prefill scratch residency**. Before raising slots or mem-fraction, compute
the one-request prefill scratch footprint and verify the startup HBM residency
leaves safe headroom for it.
