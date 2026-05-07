# M_b.2 — TileLang HD128 FP8 paged decode (Phase 1 41.6% main suspect)

> Sequel to [`M_b-tilelang-hd128-decode.md`](M_b-tilelang-hd128-decode.md)
> + M_b.1 Phase A/B (`b42da5d` / `45e1d0c` / `dd4a6d5`).
> M_b.1 ported the BF16 KV path; M_b.2 ports the **FP8 KV path** —
> the production hot path Phase 1 nsys trace flagged as **41.6% of all
> GPU time**
> (`decode_attention_varlen_quantized_partial_kernel<128, 1, 0>`,
> avg 4.55 ms × 4 104 calls,
> [`fdb531b`](../experience/wins/2026-05-07-m3.6-phase1-nsys-arle-s48-highconc.md)).

## P0 — what currently runs

`crates/cuda-kernels/csrc/attention/decode_attention_varlen_fp8.cu`
(431 lines) implements FlashDecoding split-KV in two kernels:

| Phase | Job | Grid | Block |
|---|---|---|---|
| **partial** | one (q_token, q_head, split_idx) → partial softmax accumulator | `(total_q_tokens, num_q_heads, num_splits)` | 128 (4 warps) |
| **merge** | reduce m/l/acc across split axis → final BF16 output | `(total_q_tokens, num_q_heads)` | 128 (HEAD_DIM-parallel) |

`num_splits = clamp(ceil(max_kv_len / 4096), 1, 16)`.

Pool layout (NHD durable):
- `K_pool / V_pool: [max_pages, page_size, num_kv_heads, HEAD_DIM]` — FP8 E4M3
- `K_scales / V_scales: [max_pages * page_size, num_kv_heads]` — float32

`load_quantized_value` (line 64-77):

```cuda
float value = static_cast<float>(reinterpret_cast<const __nv_fp8_e4m3*>(data)[offset]);
return scales ? value * scales[scale_offset] : value;
```

`scale_offset = (phys_page * PAGE_SIZE + t) * num_kv_heads + kv_head`.

INT8 path uses the same kernel; only `load_quantized_value`'s branch
differs (`(int8_t) * scale` vs `(fp8) → float → * scale`). `int8_kv`
flag at the C ABI (line 379) selects between them.

Workspace (`decode_attention_varlen_fp8_workspace_bytes`):

```
out_bytes  = num_splits * total_q_heads * HEAD_DIM * sizeof(float)
m_bytes    = num_splits * total_q_heads * sizeof(float)
l_bytes    = num_splits * total_q_heads * sizeof(float)
total      = out_bytes + m_bytes + l_bytes
```

Public C ABI: `decode_attention_varlen_fp8_cuda` at
`crates/cuda-kernels/src/ffi/attention.rs:342` — same shape as the
HD256/HD128 BF16 dispatch but with K/V as `*const u8` (FP8 bytes) plus
`k_scales` / `v_scales` / `int8_kv` / `causal` / `workspace` extras.

Call sites in `infer/src/model/qwen3/batch_decode.rs:1263`, `:1722`, `:2109`.

## P1 — TileLang 0.1.9 FP8 capability

From `.venv/lib/python3.14/site-packages/tilelang/`:

| Capability | Status | Reference |
|---|---|---|
| FP8 dtype in IR | ✅ **5 formats** | `language/dtypes.py:81-136` (`float8_e4m3fn`, `float8_e4m3fnuz`, `float8_e5m2`, `float8_e5m2fnuz`, `float8_e8m0fnu`) |
| `T.cast(fp8_var, "bfloat16")` | ✅ should work | `tir.Cast` with FP8 source |
| Mixed-dtype `T.gemm` | ❌ **A/B must match** | TileLang 0.1.9 assertion |
| `T.copy(fp8_src, bf16_dst)` | ❌ bit-by-bit memcpy, not a cast | `language/copy_op.py:51-120` |
| Reference FP8 attention example | ❌ **none** in upstream | grep `e4m3\|float8\|fp8` returned only dtype machinery, no kernels |

Conclusion: the **GEMM input has to be BF16**. Dequant happens in the
parallel load loop that fills `k_tile` / `v_tile` (shared mem), not
inside `T.gemm` itself.

## Approach — Phasing A (pre-dequant in shared-mem load)

Other approaches considered:

| Path | Why rejected |
|---|---|
| (B) inline `__nv_fp8_e4m3` PTX intrinsic via `T.call_intrinsic` | TileLang 0.1.9 doesn't surface FP8 conversion intrinsics in a stable way; maintenance risk |
| (C) keep hand-CUDA, swap in vLLM/Triton kernel | Out of M_b scope; breaks the "TileLang is the only AOT attention surface" goal |

Phasing A puts dequant in the existing `T.Parallel(BLOCK_N, HEAD_DIM)`
load loop:

```python
for j, d in T.Parallel(BLOCK_N, HEAD_DIM):
    abs_col = col0 + j
    page_local = abs_col // PAGE_SIZE
    in_page = abs_col % PAGE_SIZE
    page_idx = T.if_then_else(abs_col < kv_total_len,
                              KV_indices[kv_page_start + page_local], 0)
    scale_idx = (page_idx * PAGE_SIZE + in_page) * num_kv_heads + kv_head
    k_fp8 = T.if_then_else(abs_col < kv_total_len,
                           K_pool_fp8[page_idx, kv_head, in_page, d],
                           T.cast(0, "float8_e4m3fn"))
    k_scale = T.if_then_else(abs_col < kv_total_len,
                             K_scales[scale_idx], T.cast(0, "float32"))
    k_tile[j, d] = T.cast(T.cast(k_fp8, "float32") * k_scale, dtype)  # → bf16
    # same for v_tile
```

Then `T.gemm(q_tile, k_tile, scores, transpose_B=True, ...)` — both
BF16, identical to M_b.1.

### Phases

| Phase | Scope | Files | Estimate |
|---|---|---|---|
| **A0 smoke** | Single-split FP8 decode kernel; one head config (32, 8); verify TileLang codegen lowers FP8 → BF16 dequant + nvcc compiles to a per-SM cubin | new `tools/tilelang/batch_decode_paged_hd128_fp8.py` + 1 build.rs entry + 1 FFI decl (single config) | **2–3 h** |
| **A1 single-split full** | All 4 head configs (16/32/40/64, kv8); causal mask param (so prefill rows can also use it); BF16 unit-test against hand-CUDA baseline at small KV len | extend A0; full FFI macro; numerical diff test | **6–10 h** |
| **A2 multi-split** | Two TileLang kernels (phase-1 partial + phase-2 merge) mirroring hand-CUDA split-KV; workspace plumbing; supports KV up to 4096 × 16 = 65k tokens | two new `.py` + workspace_bytes Rust helper | **8–12 h** |
| **B dispatch** | Wire `kv_quant::decode_attention_varlen_fp8` call sites to dispatch via TileLang when KVFormat is FP8E4M3 + HD128; keep INT8 + non-HD128 paths on hand-CUDA | `infer/src/model/qwen3/batch_decode.rs` (3 call sites), `infer/src/ops/attention.rs`, `crates/cuda-kernels/src/ffi/attention.rs` | **2–3 h** |
| **Bench + verify** | e2e + greedy_consistency at FP8; canonical `bench_guidellm.sh m_b2-arle-s48-highconc` against `fdb531b` baseline | wins entry | **4–6 h** |

**Optimistic total**: 12–18 h (A0 + A1 + B + bench).
**Pessimistic total**: 22–34 h (full A0 → A2 → B → bench iteration).

### Acceptance per phase

- **A0**: `cargo check --features cuda` passes; cubin lands at
  `target/release/build/cuda-kernels-*/out/tilelang_aot/batch_decode_paged_hd128_fp8_q32_kv8_sm89/`;
  `tilelang_batch_decode_paged_hd128_fp8_q32_kv8_run_cuda` symbol present.
- **A1**: numerical diff vs hand-CUDA `decode_attention_varlen_fp8` at small KV (256 tokens, 1 batch) — **bf16 mantissa-level match**.
- **A2**: numerical diff at long KV (8 192 tokens) — same tolerance; workspace bytes match hand-CUDA's.
- **B**: e2e + greedy_consistency PASS with FP8 KV format; no regression vs `2a534c4` (F4-Small) baseline.
- **Bench**: `decode_attention_*` GPU time share drops from 41.6% to **<25%** in nsys trace; out tok/s improves at high-conc; or, if not faster, document why (TileLang dequant overhead vs hand-CUDA `__nv_fp8_e4m3` intrinsic).

## Risks

1. **`T.cast(fp8_var, "float32")` may fail to lower** if TileLang 0.1.9's TVM-FFI codegen doesn't have a runtime path for FP8 → float in the `tir.Cast` lowering. Mitigation: A0 smoke test surfaces this in the first hour. If broken, fall back to declaring K_pool as `uint8` and using `T.call_intrinsic("tir.fp8_to_float", ...)` if such a name exists, otherwise pivot to approach (B).
2. **Performance**: hand-CUDA does `float * scale` in registers; TileLang version may pay extra shared-mem traffic. Bench may show <10% improvement (vs M_b.1's ~10-15% for BF16) or even regression. Mitigation: A2 phase-2 merge kernel might help by reducing redundant work; if not, document and decide whether to ship a TileLang FP8 kernel (uniform AOT surface) or keep hand-CUDA.
3. **Workspace lifecycle**: hand-CUDA workspace is per-call-allocated by the Rust caller (`infer/src/model/qwen3/batch_decode.rs:166,340`). A2 has to keep that contract — no surprise growth in `decode_attention_varlen_fp8_workspace_bytes`.
4. **INT8 KV path**: A0–B do **not** address INT8. Hand-CUDA INT8 path keeps running on the existing kernel until a separate M_b.3 (out of M_b.2 scope).

## Out of scope

- Causal mask in mixed prefill+decode batches (already handled in M_b.1
  via `tilelang_tc_run_layer` fallback to prefill kernel; A1 inherits
  the same pattern).
- INT8 KV (different dequant; same kernel structure but dtype change).
- Sparse-KV draft view (`LogicalSparseDraftView`, MagicDec P2.B) —
  separate path; not touched.
- Metal port — Apple Silicon FP8 is a different conversation.

## Out-of-band gates per CLAUDE.md

- ≥2 rounds of `bun run codex review --uncommitted -c sandbox.timeouts.exec_seconds=900` before commit (fold P1 + P2 findings).
- All 5 typecheck combos (`cuda`, `cuda,no-cuda`, `metal,no-cuda`, `cuda,metal,no-cuda` if applicable, plain `no-cuda`).
- `cargo clippy --release -p infer --features cuda -- -D warnings` — note: run **solo**, not parallel with other cargo invocations (M_b.1 lesson — parallel cargo race triggers `BUILD_GIT_SHA env!()` ghosts).
- e2e + greedy_consistency at every commit boundary.
- Bench entry per CLAUDE.md §Benchmarks; cite `fdb531b` baseline + delta.

## Suggested first move (when prioritized)

A0 smoke alone (2-3 h) is enough to **license or kill** this whole
plan. If TileLang's FP8 dtype lowering breaks at codegen, M_b.2 pivots
to fallback (B) or the entire kernel-axis stays on hand-CUDA. If A0
lands, A1 is a safe extension and the rest is mechanical.

So: A0 first, decide on A1/A2 scope after seeing the A0 cubin.
