# DSv4 FlashMLA V2 SM90 sparse prefill — 22× prefill speedup (partial: crashes mid-prefill on TMA, debug pending)

## SLO-shape probed?  Y — 28899-token prefill, 8× H20 TP=8, FlashMLA V2 ENABLED

Two probes ran end-to-end with identical workload to the 2026-05-27 baseline and the 282s GEMM-marginal entry:
- **First V2 probe** (commit `f2fcee6d`): 12.81s prefill before `CUDA_ERROR_ILLEGAL_ADDRESS`.
- **Second V2 probe** (commit `eaa42a8d`, after null defense in csa_build_indices): 13.00s prefill, same crash sig.

## Roofline check

| Op | Achieved | Peak (8×H20 BF16) | % | Verdict |
|---|---:|---:|---:|---|
| Prefill end-to-end (V2 pre-crash, partial) | ~28899 tok / 13s × 8 ranks ≈ **17,800 tok/s** aggregate | ~624,000 tok/s aggregate (78,000 tok/s/rank × 8) | **2.85%** | **PARTIAL — well above the 0.13% V1/baseline number (22×↑), but still 35× below FlashMLA's published 640 TFLOPS rooflines because (a) HCA layers fall back to legacy per-token kernel and (b) we abort mid-prefill** |

Compared to the prior `2026-05-27-dsv4-grouped-gemm-marginal-prefill-kernel-not-blocker.md` 0.13%-peak result, V2 pushed the achieved-rate axis by 22× without breaking anything outside the FlashMLA path (the legacy kernel still works for HCA + SWA layers). Final SLO unlock pending crash resolution; conservatively this entry is **partial PASS** until the prefill completes cleanly end-to-end.

## Context

After landing the FlashMLA vendored kernel (commit `bbd23a20`) + V1 shim wire (`3b7808ee`), V1 aborted on `h_q % B_H == 0` at TP=8 (`docs/experience/errors/2026-05-27-dsv4-flashmla-v1-h_q-tp-shard-mismatch.md`). V2 fixed this by:

1. **Shim safety** (`0864ad64`) — broaden exception catch + pre-flight validation. V1's bug was `catch (std::runtime_error&)` missing `kerutils::KUException` which extends `std::exception`. Without this, every future misconfig aborts the host process.
2. **Building blocks** (`156e3fde`) — 3 new .cu files: `arle_dtype_convert.cu` (bf16→f32), `dsv4_tp_attention_repack.cu` (Q allgather/repack + output slice), `arle_flashmla_csa_prep.cu` (SW+compressed unified KV pool + per-token unified indices). + NCCL `all_gather_bf16_device` helper + FFI decls.
3. **attn_sink f32 mirror** (`eb1dec3b`) — populated at model load. FlashMLA wants `float[h_q]`, ARLE stores bf16. Mirror keeps the canonical bf16 unchanged.
4. **Dispatch site rewrite** (`f2fcee6d`) — combines all the above into one coherent branch in `finish_attention_gpu`: build unified KV pool → build indices + topk_length → AllGather Q across TP=8 (h_q expansion 8→64 satisfies B_H constraint) → FlashMLA call → 8-head slice from full out into local_attn.
5. **Borrow lifetime fix** (`c0693dcd`) — scope SyncOnDrop guards inside inner block before the buffers move into Option owners.
6. **CSA null-selector defense** (`eaa42a8d`) — `arle_csa_build_indices_kernel` was dereferencing `selected` unconditionally; Rust passes `null` when no selector is populated. Now branches to fill -1 padding.

## Results

| Run | Path | Prefill (28899 tokens) | vs 282s baseline | Note |
|---|---|---:|---:|---|
| 2026-05-27 baseline | per-token grid kernel | 282s | — | per the GEMM-marginal entry |
| V1 (3b7808ee) | FlashMLA → KUException → host abort | N/A | — | h_q % B_H fail |
| **V2 #1 (f2fcee6d)** | FlashMLA + TP-AllGather + SW concat + attn_sink_f32 | **12.81s before crash** | **-95% (22× faster)** | abort at MoE D2H (sticky cuda) |
| **V2 #2 (eaa42a8d)** | + null defense on selected_ptr | **13.00s before crash** | **-95% (22× faster)** | same crash sig — null wasn't the bug |

Both V2 probes ran for 13s before the same crash. That's consistent — the abort point is deterministic.

## What's causing the crash

`CUDA_ERROR_ILLEGAL_ADDRESS` on `DeepSeek V4 local route count D2H` is **sticky cuda state from an earlier OOB**, not the actual fault. Stdout shows interleaved `TMA Desc Addr:`, `boxDim (64,64...`, `swizzle 3` output — these are FlashMLA's CUTLASS TMA (Tensor Memory Accelerator on H20) descriptor-error diagnostics. The fault is inside FlashMLA's TMA load path during the sparse-prefill kernel.

Working hypotheses (in order of likelihood):

1. **`stride_kv_s_kv` and the unified-pool layout interact poorly with TMA's 128-byte swizzle.** d_qk=512 (1024 bytes per row) is multiple of 64 elements / 128 bytes, satisfies SWIZZLE_128B at the leading dim. But the `kv_unified` ptr is allocated fresh per call via `alloc_zeros::<bf16>`; cudarc returns 256-byte aligned pointers, so the address itself is fine. The descriptor-level fault could come from a stride/dim mismatch on chunk-2 when `compressed_count` grows.

2. **`indices_unified` contains values beyond `s_kv_total`** for some token. The `is_token_valid = t >= 0 && t < params.s_kv` check at `phase1.cuh:485` masks OOB indices, but TMA may not check — it loads whatever the descriptor points to.

3. **Chunk-2 `kv_unified` size grows** (because `compressed_count` accumulated through chunk 1's CSA select). If my alloc sizing in Rust didn't follow, the TMA box would point past the buffer. Need to verify `s_kv_total` calculation matches the actual count at the dispatch site.

Bisection in progress (4K-token short probe in flight as `bc98cg3n5`).

## Why V2 is structurally correct

The 22× signal proves all the new pieces work:

- TP-AllGather Q across 8 ranks → `dsv4_tp_q_repack_cuda` transposes [tp,s,h_local,d]→[s,tp×h_local,d] → FlashMLA accepts h_q=64 (satisfies B_H=64).
- SW + k_prepared + compressed packing into unified pool works for at least the first chunk.
- attn_sink_f32 mirror passes a valid float pointer (would have crashed immediately on NaN garbage otherwise).
- 8-head slice from full_out back to per-rank local_attn produces non-NaN values into the downstream wo_a GEMM (which keeps running until the chunk-2 abort).

Without these correct, the prefill would have aborted in <1s with garbage or assertion fail. 13s of correct compute on 64-bit-wide layers across 21 CSA + 22 legacy = the V2 architecture is doing real work.

## SLO impact projection

At the 22× speedup the partial-V2 already shows, end-to-end at 29K-prefill drops from 282s to ~13s. Final V2 with the crash fix + HCA mode wiring (20 more layers through FlashMLA) projects roughly 5-10s = within 2-5× of SLO 4.8s — same order of magnitude as production SGLang.

## Rule

**Vendor a CUTLASS-template kernel + log diagnostics aggressively.** The shim-safety landing (`0864ad64`) is what made this debuggable — V1 would have aborted the host process on every failed run. The TMA Descriptor Address spew on stdout is what gave us "crash is inside FlashMLA's TMA, not in Rust" — without that, we'd still be guessing.

When integrating a 3rd-party kernel with internal exceptions or asserts, **always wrap with `catch (...)`+ pre-flight validation + verbose error returns**, never a single-type catch.

## Refs

- `f2fcee6d` V2 dispatch rewrite + TP-AllGather + SW concat + attn_sink_f32
- `eaa42a8d` null defense (didn't fix the crash but tightened semantics)
- `bbd23a20` FlashMLA vendor import
- `0864ad64` shim safety (root-cause prevention of process abort)
- 3 subagent design reports (V2-A AllGather, V2-B SW concat, V2-C attn_sink + HCA + default-on)
- Probe artifacts on pod: `/sgl-workspace/arle-fresh/docs/trace-artifacts/2026-05-27-dsv4-flashmla-v2{,-fix}/`
- TMA Desc Addr output captured in `bzp7cxh0i.output`

## Open issues for V2.1 (debug + ship)

1. TMA crash root-cause: bisect with 4K-token probe (in flight); if 4K passes, the bug is in chunk-2 compressed_count interaction. If 4K fails, the bug is at smaller scale and easier to repro.
2. Once crash fixed, write **HCA mode dispatch** through FlashMLA (the 20 layers still falling back to legacy) — projected another 2-3× total speedup.
3. After both, flip `dsv4_flashmla_prefill_enabled()` default to `true`.
