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
| BF16 MoE forward (SparseMoeBlock) | CUDA | **wired (Mac-verified)** | `96f65bdc` moe.rs (route→pack→grouped-gemm→silu_mul→scatter→combine→shared); GPU-verify gated on a compatible BF16 ungated/full-attn HD128-kv8 MoE (none cached) |
| DSv4 forward TP=8/EP=8 full-16-token parity | CUDA | **partial — 1/3 prompts** | one prompt full `clean_tokens == oracle` (per-slot state + MoE contract, mirrored `08b74b35`); but 2/3 multi-prompt diverge at decode step 3-4 — likely **bf16-vs-FP8 precision** (rewrite bf16 vs FP8 oracle), isolation in flight (§8). Not yet full DSv4 correctness — `wins/2026-06-04-dsv4-multigpu-fullseq-parity.md` |
| W4 grouped GEMM (Qwen3.6-4bit) | CUDA | **pending** | 2 swap points in moe.rs flagged; Qwen3.6 canonical is 4-bit |
| CUDA Graph capture/replay | CUDA | **VERIFIED** | H20 eager==replay==HF gold (16/16); nsys: cuGraphLaunch×16 + capture×2 (impl `20274cdb`, `INFER_CUDA_DECODE_GRAPH=1`) |
| CUDA toolchain build (sm_70) | V100 | **verified (build/CPU)** | V100 node: GPU-free suite 64/0; native CUDA-C compiles sm_70 |
| HTTP / serving adapter | both | **consumer-ready** | infer-api `InferenceEngine` adapter: real incremental streaming (`ed72defc`) + telemetry from live counters (`f2273d43`) + CUDA model dispatch (`c65cd33e`); `/v1/models`+completions (HTTP SSE `stream=true` deferred). 15 infer-server/api unit tests |

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

~55% ready. No new crate imports legacy `infer/` (no circular blocker). Consumers of
`infer::server_engine`: `agent`, `cli`, `train` (+ root dev-dep).

**Adapter — DONE (this session).** `infer-api` exposes the public `InferenceEngine`
trait + `LoadedInferenceEngine` dispatch, now consumer-ready:
- **Real incremental streaming** (`ed72defc`): infer-core token-observer hook →
  infer-server `submit_streaming` channel → infer-api `complete_stream` with
  incremental detokenize + stop-boundary holdback (was a full-text stub).
- **Real telemetry** (`f2273d43`): live scheduler counters (active/queue/free-pages)
  published each tick → `EngineTelemetry`.
- **CUDA model dispatch** (`c65cd33e`): `load_cuda` classifies Qwen3 / Qwen3.5 / DSv4
  from `config.json` and picks the executor (DSv4 → multi-GPU launcher).
- **HTTP** `/v1/models` added (`c0d78626`); `/v1/completions` + `/v1/chat/completions`
  present (SSE `stream=true` still deferred — in-process consumers use the trait).

**Remaining blockers:**
1. `cli` / `agent` migration off legacy `infer::server_engine` → `infer-api`
   (import-only for cli; **not GPU-gated** — the proof step, now unblocked).
2. **`train`** still needs the CUDA-only OPD control surface (`forward_token_logits`,
   `remerge_student_lora`, `offload/reload_engine_weights`) that lives behind
   `infer-cuda` and isn't on the `InferenceEngine` trait — **this is the binding
   blocker for deleting `infer/` entirely** (cross-agent dependency on the
   infer-cuda/OPD side).
3. DSv4 multi-prompt correctness (§8) + the GPU model ports' full verification.

Deletion scope: `infer/` = 213 files ~167k LOC / 7.1 MB; new `infer-*` = ~41 files
~12k LOC (`infer-models` speculative crate removed this session, `72ebaae4`).

---

## 6. PRs

- #50 / #51 / #52 dependabot — **merged** to main (squash, 2026-06-04); main CI green.
- #53 rewrite — **draft**; promote to ready + merge once Phase 0 + TP/EP/CUDA-graph verify.

---

## 8. DSv4 multi-GPU verification frontier (2026-06-04)

Forward fully ported (`9def46fb`); 8-rank load succeeds on 8×H20 (~19.6 GB/rank).
Infra blockers cleared and **mirrored into the repo** (were pod-only hotpatches):

- NCCL bootstrap → **file rendezvous** (`e91cf0da`): the throwaway `--gen-nccl-id`
  helper minted an id embedding a listener socket that died on exit → every rank
  `Connection refused`. Rank 0 now mints in-process and shares via `INFER_NCCL_ID_FILE`.
- MTP layers → **tolerated** (`7a7bd70d`): production config has
  `num_nextn_predict_layers=1`; the base forward loops only `num_hidden_layers`, so
  the rejection (forcing a base-43 symlink config) is gone.
- Launcher device ordinal → **`INFER_CUDA_DEVICE=0`** (`3889ed5d`): under per-rank
  `CUDA_VISIBLE_DEVICES=$r` the GPU is ordinal 0, not physical `$r` (ranks 1-7 were
  `CUDA_ERROR_INVALID_DEVICE`).

**Apparent parity failure was a confounder, not a forward bug.** The harness
default prompt was Qwen ids `785,6722,315,9625,374`, which decode to garbage
`" ar造成 thATE v"` under the DeepSeek tokenizer (vocab 128000); the rewrite was
scored on a *different* prompt than the legacy oracle and emitted `'.'` (token 16),
a plausible garbage continuation — vs the oracle's `' Paris'` (11111). Fixed
(`a882823b`): default now the correct DeepSeek ids `671,6102,294,8760,344` (verified
— they appear at oracle positions 3-7). Reinforces the distilled lesson
*garbage output = config-suspect first*. **Re-run on the correct prompt PASSES**:
rank-0 prefill argmax = 11111 = oracle, TP=8/EP=8, all layer types (the native
bypass forward was never broken). See `errors/2026-06-04-dsv4-parity-prompt-id-confounder.md`.

**Full 16-token canonical parity now PASSES** (`wins/2026-06-04-dsv4-multigpu-fullseq-parity.md`):
`clean_tokens == oracle` end-to-end after fixing (a) per-slot decode state (SW
ring + compressor buffers retained across `start_pos>0`, no bail) and (b) the
MoE shared-expert contract (shared added **after** the routed all-reduce, not
folded in before where the replicated shared weights got summed 8×). Verified on
the **native** expert backend + **bf16** KV path; mirrored to repo `08b74b35`.

**Multi-prompt breadth corrects the scope:** of 3 distinct prompts, only 1 matches
16/16; "largest planet" diverges at decode step 4, "hash table" at step 3. The
rewrite runs **bf16-KV** but the legacy oracle was captured with `--kv-cache-dtype
fp8`, so the leading hypothesis is a **bf16-vs-FP8 precision** flip on tight-margin
steps, not a forward bug — but a real per-slot decode-state bug would equally
corrupt FP8, so the isolation (logit margin at first divergence + legacy-bf16 vs
rewrite-bf16 same-precision A/B) gates the verdict. **DSv4 full correctness is not
yet established.** Still open: DeepEP *serving* (fail-closed → scalar), the
DeepGEMM production backend (`cuLibraryGetKernelCount` multi-rank), FP8-KV decode
(`alloc_fp8_arena` `bail!`-gated), TP=8 Qwen.

**Separate, still-open infra blocker:** the native DeepGEMM bridge fails
`cuLibraryGetKernelCount → CUDA_ERROR_UNKNOWN` in multi-rank (single-process legacy
works). This is why the rewrite fell back to a native-grouped FP8 bypass. The
bridge's JIT cache already has per-kernel lock + unique-tmp (`deepgemm_native.cu`),
so a naive cache race is unlikely; suspect is the kernel-signature digest or
multi-rank JIT/preflight conditions diverging from legacy's warm cache. Codex is
isolating the expert-backend / JIT-cache variable on the pod.

## 9. Remaining work & sequencing

Phase 0 (CUDA eager parity), CUDA Graph, and Metal are **verified** (§2). Open gates:

1. **DSv4 multi-GPU greedy parity** (§8) — re-run with the corrected prompt
   (`671,6102,294,8760,344`); expect rewrite token1 = 11111. **GATE.**
2. **DeepGEMM native bridge multi-rank** (§8) — fix `cuLibraryGetKernelCount`
   so the rewrite runs the production deepgemm expert backend (not the bypass).
3. **TP=8 Qwen greedy parity** (#9) — launcher now correct (file-rendezvous +
   device-ordinal); needs the q2_kv1 decode config + the run.
4. **Qwen3.5/3.6 common-precision verify** (#15) — forward ported (`c9850ef5`);
   BF16/FP8/4-bit GPU runs pending compatible cached weights.
5. **Cutover** (§5): InferenceEngine adapter (parallelizable now) → migrate
   agent/cli/train → delete `infer/`.
6. **Report**: finalize this document + the Qwen3.5/DSv4 performance report with
   per-op latency, overlap, and parity verdicts.
