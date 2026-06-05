# DSv4 decode — SGLang-gap review (memory / precision / other deltas)

**Date:** 2026-06-06. Parallel SGLang-referenced audit (3 reviewers + synth) of
steady-state DSv4-Flash decode, B=1, TP=8/EP=8, default config (FlashMLA-decode ON,
GPU-router ON, eager). Commissioned by ckl ("仔细借鉴 sglang … 显存申请/复用 … 各个阶段
不同的精度 … 其他细节都得捕捉到").

**§0 measured anchors:** memset **10.4%**, cuMemcpyDtoD **17.8%** of decode (the only
profiled numbers). Every magnitude below except those two is **source-survey =
hypothesis** — each perf change must be licensed by a paired component A/B or nsys
window under the runtime's own sync framing, wall-clock per-token, not a c=1 smoke.

## Findings (magnitude-sorted)

| # | Finding | Evidence | Fix | Impact | Risk |
|---|---|---|---|---|---|
| 1 | mempool release-threshold=0 → per-step OS re-alloc | `tensor.rs` | **LANDED `aea875dd`** (threshold=MAX). Only A/B + prefill-OOM check remain | High | low-med |
| 2 | TP all-reduce synchronous; shared-expert is dep-free overlappable | `tp.rs:173`, `dsv4.rs:928` (moe AR) before `939` (shared expert reads `normed`) | shared-expert FP8 GEMM on side stream during moe AR; join at `add_batch` | High | med (stream_wait) |
| 3 | decode-graph mutually exclusive w/ FlashMLA-decode | `dsv4.rs:739` gate | capture graph over FlashMLA decode | High | high (arch) |
| 4 | `masked_m` D2D every layer | `moe.rs:2011-2017` | pass `counts` as `masked_m`, delete copy (pure alias) | High (17.8% D2D) | **low** |
| 5 | shared-expert output D2D ×2/layer | `moe.rs:1759-1762`, `2323-2325` | GEMM writes in place into `out.data` | med-high (17.8% D2D) | low-med |
| 6 | `route_out` + `grouped_contig` memsets (2/layer waste on default masked path) | `moe.rs:608-627` | drop grouped_contig memsets on non-contig path; skip route_out zero | med-high (10.4% memset) | med (prove scatter covers all slots) |
| 7 | count/cursor/packed_weight memsets | `moe.rs:609-619` | fold zero-init into count/scan/pack kernels | med (10.4%) | med |
| 8 | attn all-reduce not overlapped | `dsv4.rs:840` | overlap behind HC hc_post / next norm | med | med-high |
| 9 | fused `wqkv_a` gated off + decode-only | `attention.rs:1128/2097` | default-on after A/B; extend to prefill | med | low-med |
| 10 | Q all-gather inline (FlashMLA+TP) | `attention.rs:1837` | overlap allgather w/ metadata build | med | med |
| 11 | eager attn/HC/router allocs not on scratch (router_logits scratch unused at `moe.rs:1115`) | `attention.rs:2075..`, `hc.rs:149`, `moe.rs:457/1115` | route eager through per-slot scratch | med — **mostly absorbed by #1's cached pool** | low |
| 12 | RoPE prep standalone kernel + Q/K HBM round-trip | `attention.rs:2353` | fuse into projection epilogue | low-med | med |
| 13 | double activation-quantize (routed + shared from same `normed`) | `moe.rs:2034` vs `2265` | quantize once, share | low-med | low |
| 14 | router logits bf16 before f32 scoring (only stage below SGLang precision) | `moe.rs:1116→183` | router GEMM accumulate-to-f32 | low (correctness) | low-med — **topk-agreement A/B, don't assume** |
| 15 | non-FlashMLA SW K cache bf16 (2× KV) | `sw_window_cache` | keep FlashMLA-decode on (flag hygiene) | low | low |
| 16 | host-built FlashMLA pack tables + per-step H2D | `attention.rs:1167/1253/1792` | derive block-id/row on device | low — but host-on-critical-path, blocks #3 graph | low-med |
| 17 | standalone RMSNorm launches (~6/layer) | `attention.rs:2325/2345`, `dsv4.rs:812/885` | fuse norm into adjacent GEMM epilogue | low each | med |

**Confirmed NOT findings (SGLang-parity — don't chase):** MoE grouped/shared GEMMs are
true FP8×FP8 DeepGEMM (byte-identical); FP8-KV 584 B/tok identical; bf16-Q + FP8-KV-
dequant-to-bf16 is upstream FlashMLA design (not an ARLE upcast); bf16 all-reduce is
SGLang default; MLA-LoRA A16W8 is weight-bound at B=1 (HBM-parity).

## Action order
1. **#1 verify** (A/B `ARLE_CUDA_MEMPOOL_RETAIN` default-vs-0 + prefill-OOM @512/2048).
2. **#4 + #5** alias/in-place — lowest-risk attack on the measured 17.8% D2D.
3. **#6 (+7)** kill redundant memsets — measured 10.4%; start with grouped_contig (pure waste on default path).
4. **#2** overlap shared-expert behind moe AR — cleanest comm overlap (provably dep-free).
5. **#9** A/B + default-on fused wqkv_a.
6. **#14** router-logits f32 — correctness-gated (topk-agreement A/B), off the perf track.
7. **#3** graph over FlashMLA decode — biggest lever, biggest change, last.
8. tail: #8/#10/#11/#12/#13/#16/#17 fusion/overlap/scratch cleanup.

## Uncertainty flags (settle before licensing)
- `out_compact.data.clone()` `moe.rs:2102` — cudarc clone is a **D2D copy**; if it clones a
  per-layer hidden it's another every-layer D2D (→ above #5). Settle: read the clone site + nsys.
- #6 route_out zero-skip only safe if scatter writes **every** slot combine reads — parity-test.
- #14 may flip near-threshold expert selection — topk-agreement A/B, not assumed.
- **Bench hygiene:** `stage_profile`/`linear_profile` wrappers `event.synchronize()` per stage —
  must be OFF in any A/B or the numbers are meaningless. And confirm `use_gpu_router()=true`.
