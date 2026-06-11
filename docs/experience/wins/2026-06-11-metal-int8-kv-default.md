# Metal INT8 KV default path with long-context memory evidence

## Goal

Make Metal `--kv-cache-dtype auto` use an INT8 KV cache by default while keeping
one unified service/scheduler layer and a BF16 fallback. Verify on a real local
Apple Silicon run that long-context active MLX memory drops by the expected
direction and magnitude.

## Hypothesis

The current Metal stall/capacity issue is dominated by unified-memory pressure
from persistent per-slot full-attention K/V. Qwen3.5/Qwen3.6 only need the
full-attention K/V cache quantized; GDR recurrent/conv sidecar state should stay
in its native FP32/BF16 dtype. MLX affine 8-bit group quantization is a
pragmatic Metal first step: not CUDA KIVI, but enough to halve the persistent
full-attention KV bytes without adding backend types above `infer-api`.

## Params

- Backend: Metal, in-process `arle serve` / `infer-api::LoadedInferenceEngine`.
- Model: `mlx-community/Qwen3.6-35B-A3B-4bit` for long-context memory probes;
  `mlx-community/Qwen3.5-0.8B-MLX-4bit` for quick HTTP smoke.
- CLI/config: `--kv-cache-dtype <auto|bf16|int8>`.
- Default: Metal `auto` resolves to INT8. `bf16` is the explicit fallback.
- INT8 format: per full-attention K/V array, MLX affine 8-bit packed
  `uint32 data + bf16 scale + bf16 bias`; group size is 128/64/32 based on
  `head_dim`.
- Mutated buffers:
  - Full-attention K/V slot cache changes from `[K,V]` BF16 to
    `[Kq,Ks,Kb,Vq,Vs,Vb]` quantized triples.
  - C++ session writes only the newly appended K/V chunk into quantized storage.
  - GDR recurrent state and conv ring remain unchanged.
  - Prefix page-store still slices/concats token axis 2 for every rank-4 array;
    scale/bias arrays use the same token axis, so the page-store stays generic.
- SSD recall: unchanged fail-closed rewrite behavior (`available=false`); no fake
  SSD recall metrics are exported.

## Env

- Host: local Apple Silicon Mac, 51,539,607,552 bytes RAM (`sysctl hw.memsize`).
- Date: 2026-06-11.
- Build: `--release --no-default-features --features metal,no-cuda,cli`.

## Results

### Qwen3.6 long-context MLX allocator memory

Probe command shape:

```bash
cargo run -p infer-api --example metal_kv_memory_probe --release \
  --no-default-features --features metal,no-cuda -- \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --kv-cache-dtype <bf16|int8> \
  --prompt-tokens <8192|16384> \
  --max-tokens 2
```

Allocator bytes after `mlx_metal_clear_cache()`:

| Shape | BF16 active | INT8 active | Delta |
| --- | ---: | ---: | ---: |
| Qwen3.6, 8K prompt | 23,699,294,914 | 23,455,462,082 | -243,832,832 B |
| Qwen3.6, 16K prompt | 24,203,061,696 | 23,691,289,712 | -511,771,984 B |

Load-only active memory stayed equal within noise:

| Shape | BF16 after-load | INT8 after-load | Delta |
| --- | ---: | ---: | ---: |
| 8K run | 20,958,573,694 | 20,958,600,318 | +26,624 B |
| 16K run | 20,977,414,774 | 20,972,540,264 | -4,874,510 B |

This isolates the memory reduction to runtime KV, not model weights. Peak also
dropped on the 16K run: 24,631,662,584 B → 24,344,123,384 B
(-287,539,200 B). The after-clear active delta scales with context length,
matching the expected "persistent full-attention KV roughly halves; weights do
not" behavior.

### HTTP smoke + prefix stats

Small-model default `auto` resolved to INT8:

```text
[infer-metal] kv cache dtype = int8
GET /v1/models -> 200
POST /v1/completions max_tokens=8 -> 200, completion_tokens=8
```

Serial repeated-prefix requests on the same server moved prefix counters:

```json
{
  "lookups": 9,
  "hits": 3,
  "hit_tokens": 240,
  "hit_pages": 15,
  "published_pages": 10,
  "cached_pages": 10,
  "ssd_recall": { "available": false }
}
```

## Verification

| Check | Result |
| --- | --- |
| `cargo fmt --check` | PASS |
| `cargo check -p infer-api --release --no-default-features --features metal,no-cuda --lib` | PASS |
| `cargo test -p infer-metal --release --no-default-features --features metal -- --nocapture` | PASS, 9 passed |
| `cargo test -p cli --release --no-default-features --features metal,no-cuda` | PASS, 133 passed |
| `cargo test -p infer-api --release --no-default-features --features metal,no-cuda` | PASS, 10 unit + 4 adapter tests |
| `cargo clippy -p cli -p infer-api --release --no-default-features --features metal,no-cuda -- -D warnings` | PASS |
| `cargo test -p agent-infer --release --no-default-features --features cpu,no-cuda,cli` | PASS, 5 CLI smoke tests |
| `CUDARC_CUDA_VERSION=12060 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib` | PASS |
| `cargo build --release --no-default-features --features metal,no-cuda,cli` | PASS |
| Small-model Metal HTTP smoke, default `auto` | PASS |
| Qwen3.6 8K BF16 vs INT8 memory probe | PASS, -244 MB after-clear active |
| Qwen3.6 16K BF16 vs INT8 memory probe | PASS, -512 MB after-clear active |

## Problems

- This is MLX affine INT8, not CUDA KIVI. CUDA KIVI remains a separate backend
  implementation and quality story.
- The read path still dequantizes the active prefix to BF16 before MLX SDPA.
  Persistent KV memory drops; attention compute can still pay dequant overhead.
- SSD recall remains not re-ported in the rewrite path. The service layer still
  reports `available=false` instead of exporting fake recall counters.

## Learnings

- A Metal KV dtype switch belongs below `infer-api::EngineLoadConfig`; the
  service and scheduler layers only carry a neutral enum.
- On Apple unified memory, the right memory evidence is MLX allocator
  active/peak/cache, not just process RSS.
- For Qwen3.6, total active memory will not halve because weights dominate.
  The valid expectation is that the full-attention KV component roughly halves
  and the absolute delta grows with context length.
