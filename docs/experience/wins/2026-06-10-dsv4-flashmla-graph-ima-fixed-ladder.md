# FlashMLA-in-graph: IMA fixed (3 capture hazards), whole-step+FlashMLA runs at 33.41 — masked-MoE-in-graph is the last step to beat 38.99

**Date:** 2026-06-10 (late night). **Commit:** `e95e11b6`. 8×H20, same binary,
env-flips, `dsv4_ab_bench.py` B=1.

## Ladder (all + GPU_ROUTER=1 ⇒ pooled MoE tax)

| config | B=1 p50 | IMA |
|---|---|---|
| per-portion graph + FlashMLA (pre-fix) | dead | 12 hits |
| per-portion graph + FlashMLA (fixed) | 31.28 | **0** |
| **whole-step graph + FlashMLA (fixed)** | **33.41** (+6.8% vs per-portion) | **0** |
| masked eager default (the bar) | 38.99 | — |

Whole-step graph works on the production FlashMLA path, byte-coherent, 8/8.
The remaining −14% vs default is the POOLED MoE tax (graph body still runs the
pooled path; old measurement: pooled 28.4 vs masked 37.6).

## The three capture hazards fixed (`e95e11b6`)

1. **topk_length/sched_meta per-step → state init**: slot constants; the
   per-step `memcpy_htod(&[topk])` baked a DEAD STACK ADDRESS into the capture
   (memcpy nodes record the host pointer, not the data) → replay read garbage
   topk → insane splits → IMA. Bonus: −43 sched-meta calls/token in eager too.
2. **Compressed-delta pack devicified**: wired the existing-but-never-called
   `pack_completed_compressor_row_start_pos` kernel (closed form
   `(pos+1) % ratio` from `start_pos_device`) to run EVERY step in-capture;
   host Vec+H2D bulk remains only for multi-row gaps (request boundaries,
   always eager via warm pass).
3. **Request-boundary rearm**: `CudaGraphState::rearm_warm(n)` + warm takes
   precedence over replay; DSv4 slot reset re-arms one eager step per graph so
   per-request host work (SW ring bootstrap, compressed bulk) executes without
   dropping captures — capture cost once per slot lifetime, not per request.

## Next (specced, line-level)

Masked-MoE-in-graph — audited capture-clean except two items:
- `clone_htod(&vec![-1; …])` route-slot sentinel (moe.rs:1223, also :1646) —
  heap-source memcpy node, same hazard class as #1; replace with
  `memset_d8_async(0xFF)` (-1 i32). Count/scan/pack/DeepGEMM tail is already
  device-driven (no D2H anywhere in the masked decode path).
- Hash-routed layers consume HOST token ids (`memcpy_htod(tokens, …)` in
  `dsv4_route_device`); the graph body already maintains `token_ids_u32` on
  device (the pooled decode-graph fn uses it) → add a device-tokens routing
  entry and a masked forward variant over it; graph body switches
  `dsv4_moe_forward_decode_graph`(pooled) + `shared(Some(scratch))` →
  masked(None) + shared(None); drop the `use_gpu_router` gate conjunct.

Predicted: 38.99 × ~1.07 ≈ **41.7+** — the first config to beat the eager
default, with the comm/lockstep/MTP stack still ahead of it.

## Rules

- CUDA-graph memcpy nodes record the HOST POINTER: any `memcpy_htod` from a
  temporary inside a captured body is a use-after-free on replay. Constants →
  init; per-step values → device-derive or pre-replay update into persistent
  buffers.
- Build the device-driven twin BEFORE it's needed and it rots unwired: the
  compressed-delta kernel existed, unreferenced, while the host path shipped.
  Wiring audits beat writing twice.

## Refs
- Hazard discovery: [`wins/2026-06-10-dsv4-graph-relicense-warm-fix-flashmla-ima.md`](2026-06-10-dsv4-graph-relicense-warm-fix-flashmla-ima.md)
- sglang pattern (metadata outside graph + persistent buffers + pre-replay copy): flashmla_backend.py
