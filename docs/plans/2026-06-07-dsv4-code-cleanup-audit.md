# DSv4 Code-Cleanup Audit (post 2026-06-06/07 perf campaign)

**Date:** 2026-06-07. **Scope:** read-only survey of the DSv4 CUDA code state after the
official-kernel adoption campaign. Produces a prioritized cleanup task list. **No code
edits in this doc** — it is the spec for a follow-up implementation pass.

**Why now:** the campaign left a large opt-in env-flag family, several now-default-off
legacy fallbacks, and a few dead/superseded paths from the hand-rolled→official kernel
swaps. ckl's standing constraints govern the cleanup:
- **Flags → CLI `--`:** RUNTIME knobs move to `--xxx` threaded to the config struct;
  env vars stay only as a test-harness shim ([[feedback_runtime_config_cli_flags_not_env]]).
- **"先不要全部删除":** legacy fallbacks (csa_select, masked/pooled decode) stay as
  fallbacks for now — note them, don't delete yet
  ([`2026-06-06-dsv4-handrolled-kernel-audit.md`](2026-06-06-dsv4-handrolled-kernel-audit.md)).
- **No half-states:** any deletion is a full refactor unit, gated on
  needle + same-config-twice floor, not byte-identity.

**Caveat (SOLID):** the `file:line` cites below are from the current `main` HEAD and
this is a hypothesis-grade source survey, not runtime evidence. `crates/infer-cuda/src/{attention,dsv4,executor,moe}.rs`
are being edited concurrently by the batched-decode work — **line numbers will drift;
re-grep the symbol before touching.** Each cleanup is tagged **safe-now** or
**wait-for-batched-decode** (the unified plan
[`2026-06-07-unified-batched-kvpool-abstraction.md`](2026-06-07-unified-batched-kvpool-abstraction.md)).

---

## A. The `ARLE_DSV4_*` env-flag family → CLI `--flags` (task #35)

30 `ARLE_*` env reads across `infer-cuda/src/{attention,dsv4,moe,deepep,lib}.rs`. Split
by intent: **runtime knobs** (→ CLI `--`), **diagnostics** (stay env, test-harness
shim), and **build-gated** (stay env).

### A.1 Runtime knobs — promote to CLI `--flags` (config struct), **safe-now**

These select a production code path and belong on `arle serve` / the DSv4 launch path,
threaded to a config struct, with env retained only as a test-harness override shim.

| flag | site | current default | what it gates |
|---|---|---|---|
| `ARLE_DSV4_DSA_INDEXER` | `attention.rs:2407` (`dsv4_dsa_official_enabled`) | **ON** (opt-out `=0`) | official DSA indexer vs legacy csa_select |
| `ARLE_DSV4_FLASHMLA_DECODE` | `attention.rs:2372` | **ON** (opt-out `=0`) | FlashMLA sparse decode vs scalar hybrid attn |
| `ARLE_DSV4_FLASHMLA_PREFILL` | `attention.rs:2384` | **ON** (opt-out `=0`) | FlashMLA `sparse_fwd` prefill vs scalar |
| `ARLE_DSV4_FP8_LINEAR_DEEPGEMM` | `attention.rs:2395` | **ON** (opt-out `=0`) | fused `wq_a\|wkv` FP8 DeepGEMM vs scalar GEMV |
| `ARLE_DSV4_FUSED_WQKV_DECODE` | `attention.rs:2481` | **ON** (opt-out `=0`) | fused MLA-LoRA decode projection |
| `ARLE_DSV4_GPU_ROUTER` | `moe.rs:834` | **ON** (opt-out `=0`) | on-device MoE router (no per-layer logits D2H) |
| `ARLE_DSV4_MOE_CONTIG_DECODE` | `moe.rs:839` | **OFF** (opt-in `=1`) | contiguous/pooled decode MoE (slower at B=1) |
| `ARLE_DSV4_MOE_TRANSPORT` / `ARLE_DSV4_MOE_BACKEND` | `dsv4.rs:2027`, `deepep.rs:42/45` | **allreduce** | MoE transport: `allreduce` vs `deepep` |
| `ARLE_DSV4_COMM_OVERLAP` | `dsv4.rs:2049` | **OFF** (opt-in `=1`) | shared-expert on comm_stream behind moe AR |
| `ARLE_DSV4_DECODE_GRAPH` | `dsv4.rs:2042`, `attention.rs:4606` | **OFF** (opt-in `=1`) | decode CUDA graph capture |
| `ARLE_DSV4_SPEC_DECODE` | `dsv4.rs:2056` | **OFF** (opt-in `=1`) | MTP/EAGLE spec decode (parked, see §D) |
| `ARLE_DSV4_MTP_FROZEN_LAYER` | `dsv4.rs:1467` | (usize, mapping) | frozen-KV MTP target-layer index |
| `ARLE_CUDA_MEMPOOL_RETAIN` | `tensor.rs` (cuda-kernels) | **ON** (opt-out `=0`) | caching-allocator release threshold |

**Cleanup:** introduce a `Dsv4RuntimeConfig` (or extend the existing serve/launch config)
with `--dsv4-dsa-indexer`, `--dsv4-flashmla-decode`, `--dsv4-flashmla-prefill`,
`--dsv4-fp8-linear-deepgemm`, `--dsv4-fused-wqkv-decode`, `--dsv4-gpu-router`,
`--dsv4-moe-transport=<allreduce|deepep>`, `--dsv4-comm-overlap`, `--dsv4-decode-graph`,
`--dsv4-spec-decode`, `--dsv4-mtp-frozen-layer`. Keep the env reads as a fallback shim
inside the config builder (so the resident A/B harnesses keep working) but the
production source of truth becomes the CLI flag. `ARLE_CUDA_MEMPOOL_RETAIN` is a
cuda-kernels-level knob → a `--cuda-mempool-retain` or a context-init config field.

**Naming wart to fix while here (the #35-noted one):** `ARLE_DSV4_GPU_ROUTER` has TWO
meanings under one name:
- `moe.rs:834` `use_gpu_router()` → gates the **on-device router** (default-on, opt-out `=0`).
- `dsv4.rs:853` / `dsv4.rs:1044` `use_gpu_router = env::var_os(...).is_some()` → gates the
  **pooled/contiguous decode-scratch path** (default-OFF via `.is_some()`), feeding
  `use_moe_decode_scratch` (`dsv4.rs:1046`) and the decode-graph entry guard
  (`dsv4.rs:855-865`). This is a different feature ([[reference_dsv4_pooled_decode_slower_than_masked_b1]]).
  Same env var name → confusing and arguably a latent bug (setting `ARLE_DSV4_GPU_ROUTER=0`
  in moe.rs keeps the router ON but turns the `.is_some()` check… still true). **Give the
  two paths distinct CLI flags** (`--dsv4-gpu-router` for the router; the pooled-scratch
  gate should be folded into `--dsv4-moe-contig-decode` or removed — it overlaps
  `MOE_CONTIG_DECODE` semantically). **safe-now**, but verify the `.is_some()` vs `!="0"`
  semantics divergence is intentional before collapsing.

### A.2 Diagnostics / dump probes — keep as env (test-harness shim), **safe-now to document only**

These are debug-only and correctly stay env vars per the runtime-config rule (env OK for
test-harness shims). They should be **grouped + documented in `docs/environment.md`** as
diagnostics, not promoted to CLI:

`ARLE_DSV4_ATTN_DUMP`, `ARLE_DSV4_CSA_DUMP`, `ARLE_DSV4_NVTX`, `ARLE_DSV4_STAGE_PROFILE`,
`ARLE_DSV4_LINEAR_PROFILE`, `ARLE_DSV4_FLASHMLA_PROBE`, `ARLE_DSV4_DSA_LOGITS_PROBE`(+`_SMS`/`_LIMIT`),
`ARLE_DSV4_DSA_INDEXER_SMS`, `ARLE_DSV4_DEEPEP_NUM_SMS`, `ARLE_DSV4_FLASHMLA_PREFILL_SYNC`,
`ARLE_DSV4_FP8_GEMV_MMA`, `ARLE_DSV4_DEEP_COPY_KEEPALIVE`, `ARLE_DSV4_MTP_DRAFT_DUMP`,
`ARLE_DSV4_MTP_ROLLBACK_DUMP`(+`_LAYER`), `ARLE_DSV4_MTP_VERIFY_PERTOKEN`,
`ARLE_DSV4_FUSED_WQKV_DECODE_ALLOC`, `ARLE_DSV4_FLASHMLA_DECODE_ALLOC`.

**Cleanup (low-priority):** audit whether the `_ALLOC` sibling gates
(`FUSED_WQKV_DECODE_ALLOC`, `FLASHMLA_DECODE_ALLOC`) are still needed now that the parent
gates default-on and the `_alloc_enabled()` helpers fall through to the parent
(`attention.rs:1110` per the FlashMLA-decode win). If the alloc gate is now always
implied by the parent default, the separate `_ALLOC` env is dead — **safe-now** to remove
once confirmed it has no remaining standalone caller.

---

## B. Legacy fallbacks now default-off — KEEP as fallback, note (per "先不要全部删除")

These are the pre-official paths, retained behind the opt-out env. ckl explicitly wants
them kept for now. **No deletion** — this section just records what they are and the gate
that would license eventual removal (per-operator A/B proving the official kernel ≥ legacy
on ≥2 shapes, then delete + no half-state).

### B.1 Legacy hand-rolled `csa_select` selector — **referenced, KEEP, wait**

- **Status: STILL REFERENCED.** Official DSA is default-on, but `csa_select` (`attention.rs:4408`)
  tries `csa_select_official` first (`attention.rs:4471`) and **falls back** to the legacy
  `ffi::dsv4_csa_select_cuda` / `dsv4_csa_select_start_pos_ptr_cuda` (`attention.rs:4502/4520`)
  when official returns `None` or `ARLE_DSV4_DSA_INDEXER=0`. The kernel
  `dsv4_csa_select_kernel` (`cuda-kernels/csrc/misc/dsv4_attention.cu:1546`) + the two FFI
  shims (`cuda-kernels/src/ffi/misc.rs:538/555`) are live.
- **Verdict:** the official path is default-on and proven faster (124ms→26ms @4096), but the
  legacy kernel is the `=0` fallback. **KEEP per ckl.** Eventual-removal gate: confirm
  `csa_select_official` never returns `None` on any supported shape (TP replicate/shard of the
  indexer, num_heads∈{32,64}), then the fallback branch + the kernel + the 2 FFI shims become
  the deletion unit. **wait** — do not remove until the official path's None-return cases are
  enumerated and closed.

### B.2 Masked / pooled decode MoE paths — **KEEP, wait-for-batched-decode**

- The decode MoE has a **masked** (default, fast at B=1: 37.6 tok/s) and a **pooled/contiguous**
  path (`dsv4_moe_forward_decode_pooled`, `moe.rs:1326`; entry `moe.rs:1024/1163`), gated by
  `use_moe_decode_scratch` (`dsv4.rs:1046`) ← the misnamed `ARLE_DSV4_GPU_ROUTER`.is_some() +
  `ARLE_DSV4_MOE_CONTIG_DECODE`. Pooled is **slower at B=1** (28.4 vs 37.6,
  [[reference_dsv4_pooled_decode_slower_than_masked_b1]]) so it is correctly default-off.
- **Verdict:** the pooled path is the *batched-decode* primitive (it operates on a contiguous
  N-token scratch); the masked path is the B=1-optimal one. **Both stay** — but the batched-decode
  work (unified plan Phase 6, `moe.rs:874/1104` one-token scratch guard widened to N) will
  re-home which one is canonical. **wait-for-batched-decode**: don't delete either until the
  batched MoE lands and the c-sweep decides the canonical decode-MoE path. Note the misnamed
  gate (§A.1) should be untangled *before* this, so the batched path isn't coupled to the
  confusing `GPU_ROUTER`.is_some() check.

### B.3 Scalar hybrid-attention + scalar FP8 GEMV decode/prefill paths — **KEEP, wait**

- `dsv4_hybrid_attention` (scalar CSA/HCA) and `dsv4_fp8_gemv_batch*` are the opt-out
  fallbacks for `FLASHMLA_DECODE=0` / `FLASHMLA_PREFILL=0` / `FP8_LINEAR_DEEPGEMM=0` /
  `FUSED_WQKV_DECODE=0`. All four parents are default-on. **KEEP per ckl** as the `=0`
  fallback; deletion is a per-operator unit gated on the multi-shape A/B already partly done
  (decode token-exact +24%; prefill within-floor). **wait** — the fallbacks de-risk the
  official kernels until the batched path and more shapes are validated.

---

## C. Dead / superseded paths from the hand-rolled→official swaps

### C.1 Decode CUDA graph vs FlashMLA-decode — **mutually exclusive, verify not dead**

- `dsv4_decode_graph_enabled` (`dsv4.rs:2042`) gates a decode-graph path that the
  FlashMLA-decode win noted is **superseded on the wall-clock axis** (+1.5% vs +24%), and the
  `dsv4.rs:855-857` guard already makes them mutually exclusive (graph only when FlashMLA-decode
  is OFF and gpu-router pooled path is on). Since FlashMLA-decode is now default-on, **the decode
  graph never runs under the default config.**
- **Verdict:** not dead (it's the `FLASHMLA_DECODE=0 + DECODE_GRAPH=1` lane), but it is
  default-unreachable and its standalone value was sub-noise. The decode-graph code
  (`forward_tokens_decode_graph`, `attention.rs:4606` graph capture) is **wait-for-batched-decode**:
  the unified plan / SGLang-class roadmap want a *full* decode-graph capture paired with one-shot
  comm — the current piecewise graph may be re-homed or replaced there. Don't delete now; flag it
  as "default-unreachable, pending the full-decode-graph rework" so it isn't mistaken for a live path.

### C.2 mhc-fuse TileLang kernel — **parked/blocked, not wired**

- The `mhc_pre_big_fuse` adoption is **blocked** (TileLang f32-mma unsupported on sm_90a,
  [`../experience/errors/2026-06-06-dsv4-mhc-fuse-tilelang-f32-mma-blocked.md`](../experience/errors/2026-06-06-dsv4-mhc-fuse-tilelang-f32-mma-blocked.md)).
  If any half-wired FFI/build.rs registration for it was committed, it is a **half-state** —
  grep `mhc_pre_big_fuse` / `mhc_pre_norm_fn_fwd_mul` in `tools/tilelang/`, `build.rs`,
  `ffi/misc.rs`, `hc.rs`. **Cleanup:** either complete the adopt (fix-option 1/2 from the error
  doc) or revert the partial wiring so the build isn't carrying a non-compiling AOT entry.
  **safe-now** to audit; the error doc says the wiring was written + typechecked but the AOT
  fails — confirm it is NOT in a build-breaking committed state. The current hand-rolled
  `dsv4_mhc_*` (`csrc/misc/dsv4_mhc.cu`) stays as the live path.

### C.3 Saved-to-`/tmp` experiment diffs — **not in tree, no action**

The killed s_q=K attempts live at `/tmp/tranche2_sqk_broken.diff` and `/tmp/a2_sqk_attempt.diff`
(per the error docs), **not committed**. No tree cleanup needed; noted so a future reader knows
they are intentionally not in the repo.

---

## D. Parked MTP / spec-decode code — default-off, KEEP, document the park

The MTP/EAGLE spec-decode path is substantially landed but **parked at the draft-quality wall**
(39% accept vs SGLang 68%,
[`../experience/errors/2026-06-06-dsv4-mtp-perf-acceptance-workload-blockers.md`](../experience/errors/2026-06-06-dsv4-mtp-perf-acceptance-workload-blockers.md)).
Live code (default-off `ARLE_DSV4_SPEC_DECODE`):

- `executor.rs`: `Dsv4SpecSlotState` (`:589`), spec-slot init/reset (`:871/906/1139/1262`),
  the verify state machine + `forward_tokens_verify` calls (`:929/972/1184/1227/1249`).
- `dsv4.rs`: `forward_tokens_verify` (`:885`), `mtp_forward` + frozen-layer mapping
  (`:1359/1427/1462`), spec-rollback snapshot capture (`:488`), MTP head load (`:683`).
- the rollback snapshot/restore buffers (compressor/indexer running buffers + sw/fp8 ring slots
  + `fp8_kv_comp_packed_rows`) from the rollback-fix win.

**Verdict: KEEP, do not delete.** Spec-decode is a ckl must-have ("投机解码必须默认有且默认好用").
It is correct + wired, just not yet a perf win. **Cleanup is documentation, not deletion:**
1. Add a top-of-section comment in `dsv4.rs`/`executor.rs` marking the spec path **parked at the
   draft-quality wall**, pointing at the MTP-blockers error doc + the frozen-KV redesign
   ([`2026-06-06-dsv4-frozen-kv-mtp-redesign.md`](2026-06-06-dsv4-frozen-kv-mtp-redesign.md)),
   so a future reader doesn't mistake default-off for dead. **safe-now.**
2. The `ARLE_DSV4_SPEC_DECODE` flag → `--dsv4-spec-decode` per §A.1 (the others move; this one
   should too, with a doc note that it's experimental/parked). **safe-now.**
3. **wait-for-batched-decode** for any structural change: the frozen-KV MTP redesign interacts
   with the per-slot→shared-pool KV ownership move (unified plan Phase 3); don't refactor the
   spec rollback buffers until the `Dsv4KvAdapter` ownership lands, or the snapshot/restore
   buffer set has to be re-derived.

---

## E. Inconsistencies / smaller items

1. **`infer-api/loaded.rs:437` bails DSv4 from `arle serve`** ("DSv4 is multi-GPU only;
   launch via scripts/dsv4_multigpu_parity.sh"). So none of the §A.1 CLI flags can be set on
   `arle serve` today — DSv4 runs only via the parity/bench harnesses. **The flags→CLI move
   (§A.1) is only fully meaningful once DSv4 is wired into a real serve/launch path.** Until
   then, the CLI flags should land on the DSv4 launch entry the harnesses use, and the serve
   bail should be revisited as part of the throughput/batched-decode program. **wait-for-batched-decode**
   (serving DSv4 is gated on the `sglang` performance profile which requires the batched-decode
   feature set, [`../experience/wins/2026-06-06-dsv4-first-throughput-sweep-scaling-gap.md`](../experience/wins/2026-06-06-dsv4-first-throughput-sweep-scaling-gap.md)).
2. **`docs/environment.md` already has 21 `ARLE_DSV4` references** — when §A lands, reconcile it:
   move the runtime knobs to a "DSv4 CLI flags" table, keep the diagnostics in an env table, and
   drop any flag that was removed. **safe-now** (doc-only).
3. **The `_SMS` override env vars** (`ARLE_DSV4_DSA_INDEXER_SMS`, `ARLE_DSV4_DSA_LOGITS_PROBE_SMS`,
   `ARLE_DSV4_DEEPEP_NUM_SMS`) are tuning knobs — decide per-knob whether they are diagnostics
   (stay env) or production tunables (→ CLI). The DSA indexer SM count is arguably production
   (occupancy tuning) → CLI; the logits-probe SMS is diagnostic → env. **safe-now** to classify.
4. **`MOE_TRANSPORT` vs `MOE_BACKEND` dual env** (`dsv4.rs:2027-2028`, `deepep.rs:42-45`): two
   env names for one knob, with `MOE_TRANSPORT` taking precedence. Collapse to one CLI flag
   `--dsv4-moe-transport`; keep `MOE_BACKEND` as a deprecated alias in the shim only. **safe-now.**

---

## Sequence

1. **safe-now, doc-side:** group + document the diagnostics in `environment.md` (§A.2, §E.2);
   add the "parked" comments to the spec path (§D.1); confirm the mhc-fuse wiring is not
   build-breaking (§C.2); classify the `_SMS`/`_ALLOC` flags (§A.2, §E.3).
2. **safe-now, code-side (one flag at a time, no half-state):** introduce the `Dsv4RuntimeConfig`
   + CLI flags for the §A.1 runtime knobs, env retained as shim; untangle the `GPU_ROUTER`
   naming wart (§A.1) and collapse the `MOE_TRANSPORT/MOE_BACKEND` dual (§E.4). Each is its own
   commit + bench/A-B per the runtime-change bench rule.
3. **wait-for-batched-decode:** the legacy-fallback removals (§B), the decode-graph rehome (§C.1),
   the spec-rollback buffer refactor (§D.3), and the serve-wiring (§E.1) all sequence behind the
   unified batched-decode plan landing — do not start them on the single-row executor.

**Gate for every code change:** needle + same-config-twice non-determinism floor (NOT
byte-identity), per-flag wall-clock A/B for any default-affecting move, and a wins/ (or
errors/) bench entry per the runtime-change rule. Re-grep every `file:line` before editing —
the four hot files are under concurrent edit.
