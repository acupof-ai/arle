# ARLE device-neutral rewrite — completion & verification report

**Status date:** 2026-06-04 (living document — extended as verification lands)
**Branch:** `arch/ideal-inference-engine` (rewrite; PR #53, draft)
**Goal:** delete legacy `infer/`; elegant/extensible architecture; Metal **and** CUDA
verified; TP + EP verified; DeepEP + DeepGEMM verified; CUDA Graph supported.

This report is the single authoritative status surface for the rewrite. Each row
in the matrix cites evidence (commit / test count / bench) or names the gate it
waits on. "Verified" means a passing test or bench, not a source survey.

---

## 1. Architecture

Crate graph (host-only seam between engine and executors):

```
infer-plan (IR: ForwardPlan, SamplingParams)
  → infer-seam (host-only traits: BackendExecutor, KvPool)
    → infer-core (Engine / continuous-batch scheduler / RadixCache)
      → infer-cuda  (CUDA executor: model.rs forward_tokens, attention, loader, graph, tp, moe)
      → infer-metal (Metal executor: MLX bridge — verified)
    → infer-server (frontend thread + OpenAI HTTP)
infer-topo (TP/PP/EP/DP sharding math — 42 tests)
infer-moe  (CPU MoE routing reference — 17 tests)
infer-models (config/skeleton placeholder — NOT the forward home)
```

**Architecture verdict (binding):** backend-specific forward *per executor*. TP
all-reduce, MoE dispatch, and CUDA-graph capture live **inside** `infer-cuda` +
`cuda-kernels`, consuming `infer-topo` (sharding) and `infer-moe` (routing) as pure
CPU libraries. `infer-models::SkeletonModel` / `NoopCommunicator` and the
`infer-seam` `Communicator` / `ModelArch` / `GraphRunner` traits are speculative and
get zero new callers (`GraphRunner` deleted in `6d4a3254`). Device tensors never
cross the engine↔executor seam. No new crate depends on legacy `infer/` (verified —
no circular blocker for cutover).

---

## 2. Verification matrix

| Subsystem | Backend | Status | Evidence / gate |
|---|---|---|---|
| Engine / scheduler / RadixCache | host | **verified (CPU)** | infer-core 19 tests; prefix-reuse + chunked prefill (#8) |
| Metal forward + parity | Metal | **verified** | tasks #1/#2/#8; PR #53 "Metal verified"; Qwen3.5 on M-series |
| CUDA eager forward (Phase 0) | CUDA | **IN PROGRESS** | root cause locked (§4); TileLang 0.1.9 fix A/B in flight on H20 |
| TP sharding math | host | **verified (CPU)** | infer-topo 42 tests |
| TP wiring (tp.rs, shard, mock-comm) | CUDA | **foundation verified** | `6d4a3254`; infer-cuda 28 tests incl. row-parallel all-reduce parity |
| TP all-reduce in forward + TP=8 | CUDA | **pending** | gated on Phase 0 + 8×H20; insert at model.rs o_proj/down_proj |
| MoE routing math | host | **verified (CPU)** | infer-moe 17 tests |
| MoE wrappers + expert load + config | CUDA | **foundation verified** | `6d4a3254`; cuda,no-cuda typecheck clean |
| MoE forward (Qwen3.6) single-GPU | CUDA | **pending** | gated on Phase 0; kernels already exist (`ffi/moe.rs`) |
| EP all-to-all + DeepEP | CUDA | **pending** | gated; `deepep-sys` + legacy `native_deepep.rs` exist → port |
| DeepGEMM FP8 grouped GEMM | CUDA | **pending** | build-gated FFI exists; runtime wiring greenfield |
| CUDA Graph capture/replay | CUDA | **foundation verified** | `6d4a3254` (graph.rs + Engine::warmup); impl+verify gated on Phase 0 |
| CUDA toolchain build (sm_70) | V100 | **verified (build/CPU)** | V100 node: GPU-free suite 64/0; native CUDA-C compiles sm_70 |
| HTTP OpenAI v1 (non-stream) | both | **partial** | infer-server completions; streaming/`/v1/models`/`/v1/stats` pending |

CPU-test foundation totals (this Mac, 2026-06-04): infer-core 19, infer-cuda 28,
infer-topo 42, infer-moe 17, qwen35-spec 30 — all green.

---

## 3. Backend verification detail

### Metal (verified)
Continuous-batch decode via the MLX bridge with variable-length packed decode;
Qwen3.5 forward + greedy parity landed (#1/#2), prefix reuse + chunked prefill (#8).
Canonical production target Qwen3.6-35B-A3B-4bit per the backend matrix.

### V100 (build + CPU-test node only)
Tesla V100-SXM2-32GB, sm_70, CUDA 11.8. GPU-free workspace suite **64/0 green** on a
real Linux CUDA toolchain (independent of this Mac). All native CUDA-C kernel
families compile clean for sm_70 (no CUDA-11.8 API gap). **Cannot** run the GPU
forward: FlashMLA is sm_80+ hardcoded and TileLang's BF16 paged attention needs
sm_80+ (`cp.async`/WGMMA). Role: second build matrix point + CPU-test verifier.
Follow-up nicety: auto-disable FlashMLA when `sm_targets` is Volta.

### CUDA (Phase 0 — see §4)

---

## 4. Phase 0 root-cause analysis — CUDA eager forward

The clean `infer-cuda` BF16 Qwen3 forward (first real GPU bring-up, Qwen3-0.6B on
H20/sm_90) surfaced a sequence of bugs; the last is the current gate.

| # | Bug | Status |
|---|---|---|
| 1 | `SafetensorLoader` O(N²) re-read | fixed (`3f5f2ece`) |
| 2 | wrong `hidden_size == heads*head_dim` assert | fixed (`fe841c62`) |
| 3 | TileLang paged-attn `num_pages`/`total_pages` arg swap → Xid 43 | fixed (`db85d56e`) |
| 4 | **prefill cubin spins (100% util, no Xid) at every prompt geometry** | **fix in flight** |

**Empirical A/B (single-variable prompt-length sweep), all args confirmed correct:**

| prompt | qlen | grid bx | KV trip | Q-tile | result |
|---|---|---|---|---|---|
| 5  | 5  | 1 | 1 | partial | hang |
| 64 | 64 | 1 | 1 | **full** | hang |
| 70 | 70 | 2 | 2 | mixed | hang |

Falsifies partial-tile, trip-count-1, and single-block hypotheses (qlen=64 is a full
tile with zero padding → zero NaN, yet hangs; qlen=70 has trip count 2 / bx 2, yet
hangs). → a **fundamental prefill-cubin wedge on sm_90**, prompt-shape-independent.

**Probes (read-only, on the H20 build host):**
- TileLang version resolved by the build = **0.1.10** (system `/usr/bin/python3`, no venv).
- No `mbarrier`/`cp.async`/`wait_group` in either prefill or decode device source →
  **not** an async-pipeline/mbarrier wedge.
- Prefill dyn-shmem = 49152 B = `q_tile+k_tile+v_tile` = correctly sized → **not** mis-sized.

**TileLang version — ruled out (0.1.9 A/B, 2026-06-04):** installed 0.1.9 in a venv,
forced AOT regen (confirmed new prefill device-source sha), rebuilt, re-ran
`seq_len=64` → **still hangs**. So this is **not** the 0.1.10 FullRow defect of
`errors/2026-05-27`; that fix verdict does not transfer here.

**Narrowed root cause (probe in flight): sm_90a / HD128-FullRow-WGMMA.** The
`FullRow` gemm emits Hopper WGMMA, which nvcc enables only under the `sm_90a`
target (`gen_tilelang_aot.py:538-563` is supposed to pass
`-gencode=arch=compute_90a,code=sm_90a` for `cuda_arch==90`). The 2026-05-30 H20 win
ran the **HD256** prefill FullRow-WGMMA on sm_90 correctly; the **HD128 q16_kv8**
prefill (Qwen3-0.6B) may have never run on sm_90 before R6. A vs B: (A) build didn't
apply sm_90a → build fix; (B) it did and HD128 FullRow-WGMMA is miscompiled on
sm_90a → kernel-level fix (warp policy / tile). Read-only arch + WGMMA-count probe
decides. Then: guarded-`exp2` partial-tile fix + HF-gold greedy parity close Phase 0.

---

## 5. Cutover readiness (delete legacy `infer/`)

~30% ready. No new crate imports legacy `infer/` (no circular blocker). Consumers of
`infer::server_engine`: `agent`, `cli`, `train` (+ root dev-dep). Blockers:

1. **CRITICAL** — no public `InferenceEngine` trait + `LoadedInferenceEngine` dispatch
   in the new stack (the new stack exposes `Engine`/`ServeHandle` with `Vec<u32>` +
   blocking collect, not the trait + `String` + streaming contract). Needs an adapter
   (new `infer-api` crate or in `infer-server`). **Not GPU-gated** — buildable in parallel.
2. CUDA Phase-0 parity (§4).
3. Qwen3.6 / DSv4 CUDA model ports.
4. `infer-server` HTTP: non-streaming only.

Deletion scope: `infer/` = 213 files ~167k LOC / 7.1 MB; new `infer-*` = ~42 files
~12k LOC.

---

## 6. PRs

- #50 / #51 / #52 dependabot — **merged** to main (squash, 2026-06-04); main CI green.
- #53 rewrite — **draft**; promote to ready + merge once Phase 0 + TP/EP/CUDA-graph verify.

---

## 7. Remaining work & sequencing

Critical path is serial through Phase 0:

1. **Phase 0** (CUDA eager parity) — TileLang 0.1.9 fix A/B in flight. **GATE.**
2. **GPU wiring** (gated on 1, serialized on `model.rs::forward_tokens`): CUDA-graph
   decode refactor → TP all-reduce inserts → MoE forward branch. Foundation already
   committed (`6d4a3254`); these are the model.rs/executor.rs edits + H20 verify.
3. **Verification** (parallel once 2 lands): TP=8 greedy parity; MoE single-GPU then
   EP+DeepEP; DeepGEMM FP8; CUDA-graph eager-vs-replay — all on 8×H20.
4. **Cutover** (§5): InferenceEngine adapter (parallelizable now) → migrate
   agent/cli/train → delete `infer/`.
5. **Report**: finalize this document with bench numbers + parity verdicts.
