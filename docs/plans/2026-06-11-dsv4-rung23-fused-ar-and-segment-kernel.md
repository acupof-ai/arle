# Rung 2+3: device-side AR fusion + hc-enter segment kernel — line-level plan

Commissioned by ckl (“23都做好”). Constraints inherited from the Rung-1
campaign: dynamic SMEM ≤16KB per kernel (carveout-switch kill,
[errors/2026-06-11-…-smem-carveout.md]); single-block bandwidth needs 1024
threads; microbench is necessary-not-sufficient — matched e2e pairs gate
every land, both directions; md5-verify pod files before builds.

## Rung 3 — CAR as a device function: `AR(+add)+hc_post` fused kernels

**Where it pays:** per layer, 2× [staging memcpy (1.2µs+gap) + AR kernel
boundary + hc_post boundary + 8KB/16KB intermediate round-trips]; the moe
site also absorbs `add_batch`. Est. −0.5…0.7 ms/token; architecturally it
proves CAR-in-kernel for the megakernel path.

**Protocol soundness (the one novel invariant):** the vendored 1-stage
kernel grid-strides the whole array, which is only safe because inputs are
fully staged before launch. The fused kernel instead gives each block an
EXCLUSIVE packed chunk: stage own chunk → `__threadfence_system` →
`multi_gpu_barrier<ngpus,true>` (per-block, cross-rank) → `packed_reduce`
own chunk from 8 peer ptrs → hc_post math in registers → end barrier.
Block b only ever reads chunk b, which block b of every rank staged before
its barrier — no cross-block dependency. Grid is FIXED (16×256) so per-block
flag slots stay consistent across launches (Signal supports ≤36 blocks).

**Pieces:**
1. `csrc/comm/custom_all_reduce.cu`: `template<int ngpus> arle_car_ar_hc_post_kernel`
   (+ optional `shared_add` operand for the moe site) + C entry
   `arle_car_fused_ar_hc_post(h, stream, new_x, shared_or_0, residual, post,
   comb, out, tokens, hidden, hc_mult)` — RankData via `car->buffers_` like
   the AG entry; bitwise-fixed accumulation order preserved.
2. `ffi/comm.rs` decl; `tp.rs` `OneShotComm::fused_ar_hc_post(...)`
   (capacity check: tokens*hidden*2B ≤ registered scratch).
3. `dsv4.rs` stream-impl sites: attn `[all_reduce_sum + hc_post]` and moe
   `[all_reduce_sum + add_batch + hc_post]` → fused when comm present AND
   `ARLE_DSV4_FUSED_AR=1` (A/B arm flag; default off until licensed);
   fallback = existing pair. Graph body unchanged.
4. Gates: cross-rank output correctness (needle prompt + same-config-twice),
   matched e2e pair (auto+fused vs auto), co-tenant checks, bench entry.

## Rung 2 — `hc_enter` segment kernel, multi-block + counter sync

Reshape the killed single-block hc_enter with the two lessons applied:
multi-block (gemv bandwidth) + ≤16KB smem (no carveout switch) + absorb the
mix GEMV (nvjet 5.3µs + splitKreduce 1.75µs + gaps).

**Shape:** persistent-style cooperative kernel, fixed grid ~32×256, global
counter sync (Hazy pattern):
- Phase A (all blocks): partial dot for mixes[24] over column chunks of the
  stream (weights 24×16K bf16) + partial sumsq of the stream row →
  atomicAdd into globals; arrive-counter.
- Phase B (block 0): waits counter==grid; finishes mixes scaling (rsqrt),
  sigmoid/Sinkhorn warp tail → pre/post/comb global + release-counter.
- Phase C (all blocks): spin release-counter; pre-mix + rms over column
  chunks — rms needs a SECOND全局 sumsq of the mixed row → two-pass within
  phase C (partial sumsq → counter → normalize) or single pass recomputing
  mix twice (compute-cheap, read stream chunks from L2 twice).
Est: replaces [nvjet 5.3 + splitK 1.75 + params 8.4 + prologue 4.7 + 4 gaps
≈ 26µs] with one ~14-16µs kernel → −0.8…1.0 ms/token. Risk: hand gemv vs
nvjet efficiency; counter-spin overhead; license by matched pair only.

## Order

R3 first (bounded, independent), R2 second. Each lands behind its own flag
with its own matched pair; kill criteria identical to Rung 1.
