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
| CUDA eager forward (Phase 0) | CUDA | **VERIFIED** | exact HF-gold greedy parity (16/16 tok) on H20 via chunk=1 decode-kernel prefill — `wins/2026-06-04-r6-cuda-eager-parity-verified.md` |
| CUDA batched-prefill cubin (perf) | CUDA | **known-issue** | HD128 FullRow-WGMMA TileLang codegen spin (§4); perf-only, decode+chunk=1 is the correct path |
| TP sharding math | host | **verified (CPU)** | infer-topo 42 tests |
| TP wiring (tp.rs, shard, mock-comm) | CUDA | **foundation verified** | `6d4a3254`; infer-cuda 28 tests incl. row-parallel all-reduce parity |
| TP all-reduce in forward + TP=8 | CUDA | **pending** | gated on Phase 0 + 8×H20; insert at model.rs o_proj/down_proj |
| MoE routing math | host | **verified (CPU)** | infer-moe 17 tests |
| MoE wrappers + expert load + config | CUDA | **foundation verified** | `6d4a3254`; cuda,no-cuda typecheck clean |
| MoE forward (Qwen3.6) single-GPU | CUDA | **pending** | gated on Phase 0; kernels already exist (`ffi/moe.rs`) |
| EP all-to-all + DeepEP | CUDA | **pending** | gated; `deepep-sys` + legacy `native_deepep.rs` exist → port |
| DeepGEMM FP8 grouped GEMM | CUDA | **pending** | build-gated FFI exists; runtime wiring greenfield |
| CUDA Graph capture/replay | CUDA | **VERIFIED** | H20 eager==replay==HF gold (16/16); nsys: cuGraphLaunch×16 + capture×2 (impl `20274cdb`, `INFER_CUDA_DECODE_GRAPH=1`) |
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

**Root cause (classified): a hard TileLang codegen bug** in the HD128 multi-row
prefill FullRow-WGMMA lowering on sm_90a — see
`errors/2026-06-04-tilelang-hd128-prefill-wgmma-hang-sm90a.md`. Ruled out, each by a
controlled experiment: TileLang version (0.1.9 forced-regen still hangs), build-arch
(cubin confirmed `sm_90a` + WGMMA), `BLOCK_N` (32 still hangs), warp-policy (`Square`
lowered to the *identical* device-source sha as `FullRow` → the knob is inert), FFI
arg order (matches the generated signature), dyn-shmem, trip-count, partial-tile.

**Decisive positive — the rewrite architecture is sound.** A 1-token prompt routes to
the **decode** kernel and ran cleanly through all 28 layers via the clean R6 launch
path → engine→executor→model→attention→launch + the decode cubin all work; the spin
is specific to the HD128 *batched* prefill cubin (`BLOCK_M=64` multi-row WGMMA). The
2026-05-30 win ran HD256 prefill on sm_90 fine → the defect is HD128-shape-specific.

**Resolution — correctness via `chunk_size=1` (in flight):** process the prompt as
sequential 1-token forwards through the proven decode kernel (causally identical to
batched prefill) → end-to-end greedy parity vs HF gold closes Phase-0 *correctness*
without the broken cubin. The batched HD128 prefill cubin (fast long-prompt prefill)
is a documented **perf-only** follow-up (upstream TileLang fix or FlashInfer-C++
migration). Sibling decode `cache_len != kv_seq_len` error was a stale-pod-binary
artifact, not a code bug (current planner.rs is correct; guards added in `8388fc64`).

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
