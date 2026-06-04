# DeepSeek-V4 on SGLang — code-level study & ARLE borrowable map

**Date:** 2026-06-04. **Type:** research (learning + cross-reference, not a wins/bench entry).
**Mechanism sources (verifiable — "code is truth", not analysis prose):**
- SGLang side: read directly from an upstream checkout at **exactly commit
  `c048ebd10d61fc5904dc342fd0cb63d273b21afc`** ("`[Hicache]: skip flaky test (#26764)`",
  2026-05, upstream main) — the same commit the source analysis anchors to. Every code
  block in the [Appendix](#appendix--code-details-sglang-c048ebd10--arle) is a *real excerpt*
  at that commit (`sglang@c048ebd10 python/sglang/...`), and every `file:line` SGLang anchor
  in the matrix was line-verified against that tree, not taken from the prose. Overlap/scheduling
  items cross-check the in-repo HEAD `8980eb82` survey
  ([`2026-06-04-sglang-operator-selection-dsv4-qwen3moe.md`](2026-06-04-sglang-operator-selection-dsv4-qwen3moe.md)).
- ARLE side: every "ARLE state" cell and code block is grounded in current source
  (`crates/deepseek-spec`, `crates/infer-cuda`, `crates/infer-moe`,
  `crates/cuda-kernels`, legacy `infer/src/model/deepseek/`), not inferred from docs.

**Why this doc exists.** Three DSv4 docs already cover the *path* and the *overlap gaps*
([path research](2026-06-01-dsv4-sglang-path-research.md),
[operator selection](2026-06-04-sglang-operator-selection-dsv4-qwen3moe.md),
[port plan](../projects/2026-06-04-dsv4-port-plan.md)). None gives a single
**mechanism-by-mechanism SGLang↔ARLE cross-reference at code level** with a
have/partial/missing/divergence verdict per mechanism and the borrowable delta
localized to an ARLE file. That is this doc. It is the "what V4 actually does in
code, and where ARLE already does it / still needs it" reference.

**Verification boundary (carried from §0 SOLID + the source analysis).** Mechanism
conclusions (which kernel runs, which bytes are stored, which branch is taken) are
code-grounded on both sides. **Performance magnitudes are structure-derived bounds
only** (FLOPs / bytes / launch counts) — no end-to-end multiplier is asserted without
nsys trace or a matched A/B on the SLO shape. The repo's DSv4-Flash smoke (5 s,
512-in/16-out) is reachability evidence, not a throughput number.

---

## TL;DR verdict

**ARLE already implements the V4 *structure*.** The four headline V4 reorganizations
relative to V3 — layered hybrid-sparse attention (SW/CSA/HCA), learned KV compression,
un-absorbed single-head MLA, and Sinkhorn multi-stream residual (MHC) — are all present
in ARLE today (config authority in `deepseek-spec/src/v4.rs`, kernels in
`infer-cuda` + `cuda-kernels`). The borrowable delta is **not** the architecture; it is
**runtime overlap/scheduling and one routing balancer**:

1. **DeepEP Waterfill** (shared expert as the 9th routed expert, capacity-weighted
   dispatch to the least-loaded rank) — **completely absent in ARLE** (zero source hits).
   Downstream of getting DeepEP dispatch/combine *served* (today fail-closed → scalar fallback).
2. **Decode low-latency DeepEP dispatch**, **SBO** (combine↔down-GEMM + shared-expert↔dispatch
   two-stream), **TBO** (two-micro-batch overlap) — catalogued as the perf-phase gaps; ARLE's
   collectives are eager.
3. **Nested multi-stream attention prep** (hide indexer + compressor + KV-write behind the
   `wq_b` GEMM, ~5 streams, event-level deps) — absent; distinct from SBO/TBO and from the
   *compressor-window* overlap ARLE already has.
4. **HiSparse for compressed pages** — ARLE has the host-KV-tier substrate
   (`infer/src/kv_tier/host_pool.rs`) but not the DSv4 compressed-page adaptation.

**Three "easy to misalign" V4 details — ARLE already handles all three.** Output
inverse-RoPE, the compressor's *learned* `ape` bias (≠ pooling), and nope-FP8/rope-BF16
KV precision. Each is wired in ARLE; this doc records *where*, so a future port/refactor
does not silently regress them.

**One divergence to keep, not fix:** ARLE generalizes the SGLang hard `{0,4,128}`
compress-ratio enum to threshold bands (`0` → SW, `1..=15` → CSA, `≥16` → HCA). This is
strictly more general and config-driven; keep it.

---

## The cross-reference matrix (spine)

Status key: ✅ implemented · 🟡 partial / contract-only · ❌ missing · ⚖️ deliberate divergence.

### A. Attention (§2 of the analysis)

| Mechanism | SGLang locus (`c048ebd1`) | ARLE locus | Status |
|---|---|---|---|
| Per-layer 3-tier `compress_ratio` dispatch (SW⊕CSA⊕HCA, one FlashMLA call) | `models/deepseek_v4.py` `MQALayer` `#L223,256`; backend `deepseek_v4_backend.py#L332` | `deepseek-spec/src/v4.rs` `DeepSeekV4AttentionMode` `:417`, `attention_layer_plan` `:202`; `infer-cuda/src/attention.rs` mode dispatch (`prefill_attention:206`/`decode_attention:278`, comment `:590`) | ✅ |
| Single 512-dim KV head (MQA), **no weight absorption** | `deepseek_v4.py#L76`; no `w_kc/w_vc/absorb` in the file | legacy `infer/src/model/deepseek/mla.rs:5` ("single KV head"); rewrite `attention.rs:584` ("single compressed KV latent"); `dsv4.rs:52` 584 B/tok FlashMLA layout (byte-identical to upstream) | ✅ (matches V4; see Divergences re: the V2/V3 "absorbed" note) |
| Lightning indexer — top-512 select, FP8 paged-MQA logits | `layers/attention/dsv4/indexer.py` `C4Indexer:525`, `compute_q:578`, scoring `:433`, `topk_transform_512:490` | `attention.rs` `csa_select:1155`; `cuda-kernels/csrc/misc/dsv4_attention.cu`; FFI `cuda-kernels/src/ffi/misc.rs` | ✅ |
| Learned KV compressor (online softmax + learned `ape` bias, ≠ pooling) | `compressor.py` `Compressor:325`, `compute_kv_score`; kernel `csrc/deepseek_v4/c4.cuh` | `attention.rs` `compressor_forward:1054`; window-overlap state `Dsv4CompressorState:24` (`coff=2`/`2*head_dim` when `ratio<16`) | ✅ |
| Per-head learned `attn_sink` folded into sparse-MLA softmax | `deepseek_v4.py#L345` (`nn.Parameter`) | `attention.rs` (`attn_sink` in SW/decode), `loader.rs` load, `dsv4.rs`; legacy `weights.rs`/`mla.rs` | ✅ |
| Output inverse-RoPE (strip position before O-proj) | `deepseek_v4.py#L881` `fused_rope_inplace(..., inverse=True)` | `cuda-kernels/csrc/misc/dsv4_attention.cu` (`arle_dsv4_output_inverse_rope`); fused into SW kernel (`attention.rs:864-866`) | ✅ |
| Grouped low-rank output proj `wo_a`/`wo_b` (FP8 DeepGEMM einsum, UE8M0) | `deepseek_v4.py#L902,395` | `deepseek-spec/src/v4.rs` `output_projection_shape:273`; O-LoRA in `attention.rs` | ✅ (shape + path) |
| Compress-layer RoPE base split (`compress_rope_theta` vs `rope_theta`) | `deepseek_v4.py#L271` | `attention.rs:806-811` (Q/SW-K/output = `rope_theta`, no YaRN; compress θ inside `compressor_forward`) | ✅ |
| 584-byte KV (nope-FP8 448B + rope-BF16 128B + scales), 4-pool layout | `deepseek_v4_memory_pool.py` `get_bytes_per_token:93` (`assert ==448+64*2+8`, `:108`) | `dsv4.rs` `DSV4_FLASH_KV_BYTES_PER_TOKEN=584:51`, `Dsv4MlaKvArena::from_config:54` (asserts NoPE=448/RoPE=64) | ✅ layout / ⚖️ FP8 decode gated |
| FP8-KV *decode* path live (FlashMLA reads FP8 cache directly) | default decode reads FP8 cache | `dsv4.rs:83-104` — FP8 arena `alloc_fp8_arena` is `bail!`-gated; correctness path attends the **bf16 SW ring + bf16 compressed pool** (`dsv4_hybrid_attention_cuda`) until bf16 parity-matches the oracle | 🟡 (deliberate: correctness-first) |
| Hadamard-before-FP8 on indexer Q/KV (QuaRot) | `indexer.py` `fused_q_indexer_rope_hadamard_quant:578` | `dsv4_attention.cu` indexer Q prep | ✅ (kernel present) |
| HiSparse host↔device tiering for compressed pages | `hisparse_coordinator.py:42`; `load_cache_to_device_buffer_dsv4_mla` | substrate only: `infer/src/kv_tier/host_pool.rs`, `coordinator/`; **no DSv4 compressed-page path** | 🟡 |

### B. MoE / routing (§3)

| Mechanism | SGLang locus | ARLE locus | Status |
|---|---|---|---|
| `sqrt(softplus)` scoring | `topk.py#L858` | `deepseek-spec/src/v4.rs` `router_scores_from_logits:308`; `infer-moe/src/route.rs`, `config.rs`; `cuda-kernels/csrc/moe/dsv4_route.cu` | ✅ |
| Un-grouped flat top-6/256 (`topk_group==n_group` forces non-grouped) | `model_config.py#L265`, `deepseek_v2.py` | `deepseek-spec/src/v4.rs` `moe_routes_from_scores:319` + `noaux_tc` bias top-k `topk_indices_by_score:556` | ✅ |
| noaux_tc correction-bias top-k (select on `score+bias`, weight on `score`) | `biased_topk_impl` | `v4.rs:378` (select `score+bias`, weight raw `score`); `dsv4_route.cu` fused | ✅ |
| **DeepEP Waterfill** — shared expert = 9th routed, capacity-weighted least-loaded dispatch | `deepep_waterfill.py:14,86,364` (`MIN_BATCH_FOR_BALANCE=64`) | **none** — zero source hits across `infer/`, `infer-cuda`, `infer-moe`, `cuda-kernels` | ❌ |
| DeepEP dispatch/combine (decode low-latency vs normal split) | `deepep.py:191-209,785` | sidecar boots but **dispatch/combine not served** (fail-closed → native-scalar fallback; [port plan](../projects/2026-06-04-dsv4-port-plan.md) "Legacy gap"); no LL/normal split | 🟡/❌ |
| mxfp4 / FP8 expert auto-detect from safetensors dtype | `configs/deepseek_v4.py#L13` `try_detect_fp4_experts` | `cuda-kernels/src/tensor.rs`; `infer-cuda/src/loader.rs`, `dsv4.rs` (F4/F8 detect) | ✅ |
| DeepGEMM grouped FP8 expert GEMM (`f8f8bf16`, 128×128 block scale) | `get_moe_impl_class` | `cuda-kernels` DeepGEMM FFI `ffi/gemm.rs`; custom `dsv4_grouped_gemm.cu` (M-tile=32) — must beat or retire vs native masked | ✅/⚖️ |
| EPLB physical placement (`num_groups=None`, free placement) | `deepseek_v2.py#L1996` | `infer-topo::build_moe_ep_groups`, `ExpertSplit` (EP-aware) | ✅ (placement) |

### C. Multi-stream residual — MHC (§4)

| Mechanism | SGLang locus | ARLE locus | Status |
|---|---|---|---|
| 4-lane residual stream (`[n,4,d]`), single-stream compute path | `deepseek_v4.py#L1343` repeat; decoder `#L1118` | `infer-cuda/src/hc.rs` (`initial_stream_from_embeddings:32`, wide `hidden*hc_mult` stream) | ✅ |
| Per-sub-block hc_pre/hc_post mix (collapse 4→1, broadcast 1→4) | `deepseek_v4.py#L971,1110`; `hc_pre_torch_impl#L994` | `hc.rs` `pre`/`post`/`comb` weights `:22-27`; legacy math `weights.rs` `gen_mhc_params`/`hc_pre_from_stream`/`hc_post_to_stream` | ✅ |
| Sinkhorn doubly-stochastic 4×4 mix (20 iters, eps 1e-6) | TileLang Sinkhorn loop | `cuda-kernels/vendor/tilekernels/tile_kernels/mhc/sinkhorn_kernel.py` + `norm_fn_kernel.py` + `multilayer_recompute_kernel.py`; shared `dsv4_mhc_*` kernels via `hc.rs` | ✅ |
| TF32 split-K prenorm GEMM with fused sqrsum (perf) | `deep_gemm.tf32_hc_prenorm_gemm` (PR #26238) | `cuda-kernels/vendor/deepgemm/tests/test_hyperconnection.py`; ARLE prenorm path in `hc.rs`/`moe.rs` | ✅ (path present) |
| MTP draft consumes the un-collapsed `[n,4d]` state | `deepseek_v4_nextn.py#L150`; `compress_ratio=0` draft `#L47` | `deepseek-spec/src/v4.rs` `DeepSeekV4MtpTensorNames` (`enorm`/`hnorm`/`e_proj`/`h_proj`/`hc_head`); MTP contracts in wins (`2026-06-02-dsv4-mtp-loader-contract`, `…-internal-mtp-draft-mode-contract`) | 🟡 (contract + structure; full 4-lane draft consumption in progress) |

### D. Runtime (§5)

| Mechanism | SGLang locus | ARLE locus | Status |
|---|---|---|---|
| In-graph metadata materialization (`PREP_IN_CUDA_GRAPH`, raw→full lazy upgrade, warmup restore) | `deepseek_v4_backend.py#L406`; model `#L1376`; indexer `#L374` | `infer-cuda/src/executor.rs`, `decode_graph.rs`; device-side meta fills (wins: `…-flashmla-topk-device-fill`, `…-window-update-device-startpos`, `…-decode-body-graph-start-pos-abi`); active (errors `…-body-graph-zero-capture-pc11`, `…-decode-row-metadata-pc2a`) | 🟡 (well along) |
| Two-level nested multi-stream attention prep (~5 streams, event deps) | `deepseek_v4.py#L309,500,557`; indexer fork `indexer.py#L284` | **none** for stream overlap — collectives eager (`forward.rs:715`), shared-experts inline (`mlp.rs:1596-1748`). NB: `attention.rs` `prev_overlap_*` is *compressor-window* overlap, not stream overlap | ❌ |
| SBO (combine↔down-GEMM + shared↔dispatch two-stream) | `single_batch_overlap.py:97-124` | none | ❌ |
| TBO (two-micro-batch overlap) | `batch_overlap/operations_strategy.py:94-150` | none | ❌ |
| Graph buckets (DECODE_OR_IDLE / TARGET_VERIFY / DRAFT_EXTEND) | `_GraphBucket` × bs | `decode_graph_key.rs`, `decode_graph.rs`; EAGLE verify gates (wins `…-spec-verifier-contract-gate`, `…-eagle-acceptance-functional-gate-pc10`) | 🟡 |
| Single-layer NextN draft, `topk∈[0,1]` chain | `deepseek_v4_backend.py#L370`; `deepseek_v4_nextn.py` | EAGLE/MTP backend + `deepseek-spec` MTP names; wins `…-internal-mtp-draft-mode-contract` | 🟡 |
| DSA prefill context-parallel (round-robin token split + all-gather-rerange) | PR #23292/#23269 | TP=8/EP=8 runtime (`dsv4.rs build_dsv4_tp_runtime:584`); no DSA prefill-CP | ❌ (not yet a target) |

---

## Borrowable deltas — code-level, localized to ARLE

Ordered by *structural* leverage on the decode SLO (the analysis is explicit that no
end-to-end multiplier is claimed; each lands behind a matched A/B on the SLO shape per
[bench spec §7](../bench-and-trace-spec.md)).

### 1. DeepEP Waterfill (`deepep_waterfill.py` → `infer-moe` route + `moe.rs`)
**What it is.** Make the shared expert a routable expert (top-k 6→7) and dispatch that
extra slot by capacity-weighted random sampling to the least-loaded rank, instead of
always its home rank. Static mode (default): local route counts, no comm, skip balance
when batch < 64. Per-token: `target_total = ceil((k_eff+tokens)/world)`,
`w = max(target_total - rank_load[r], 0)` (home rank ×11/10), LCG-hash sample.
**Why borrowable.** It is a *router-layer load shaper on top of standard DeepEP* — it
changes only the per-token target-expert set before dispatch; the all-to-all kernels are
unchanged. So it is additive to ARLE's existing DeepGEMM grouped GEMM and (once served)
DeepEP path.
**Precondition (license-or-kill ordering).** Waterfill requires fused shared experts
(`num_fused_shared_experts>0`) and a *served* DeepEP dispatch. ARLE's DeepEP today is
fail-closed (scalar fallback) and the default transport is local-routed + EP all-reduce.
**So Waterfill is downstream of port-plan Piece 4** (DeepEP serving). Landing it before
that has nothing to balance.
**Where it lands.** Routing decision in `infer-moe/src/route.rs` (the target-expert set),
fused-shared gating in `infer-cuda/src/moe.rs`. No kernel change.

### 2. Decode low-latency DeepEP dispatch (gap #1)
LL vs normal dispatch split, chosen by batch size; the largest decode-ITL lever in the
operator survey. ARLE sidecar has no LL/normal split. Lands in the DeepEP communicator
(port-plan Piece 4). Pairs naturally with Waterfill's `MIN_BATCH_FOR_BALANCE=64` (small
decode batch → LL dispatch *and* skip balance — both choose "less comm").

### 3. Nested multi-stream attention prep (§5.1)
Main stream runs `wqkv_a`/`q_a`/`q_norm` then the big `wq_b` GEMM; three alt-streams run
indexer (weights-proj + fused-Q + paged-MQA-logits + top-512), KV-write (fused norm+RoPE+FP8
direct-to-cache), and compressor — all hidden behind `wq_b`. Fine-grained
`record_event`/`wait_event`, not `wait_stream`. **Distinct** from ARLE's existing
compressor-window overlap and from SBO/TBO. Only profitable under graph capture at small
batch (the analysis flags the "SM-saturated above 64/128 → no idle SMs to hide in"
assumption as *unverified by nsys* — measure before landing). Lands in
`infer-cuda/src/attention.rs` + `executor.rs`.

### 4. SBO / TBO (gaps #2, #3)
SBO: combine↔down-GEMM and shared-expert↔dispatch two-stream (DeepGEMM signal +
partitioned SMs). TBO: two-micro-batch overlap. Both are structural overlap gaps in *both*
legacy and rewrite — i.e. the perf phase adds value beyond legacy. `moe.rs` (SBO),
scheduler/`forward` (TBO).

### 5. HiSparse for compressed pages (§2.8)
ARLE has the host-tier substrate (`kv_tier/host_pool.rs`, coordinator). The DSv4 piece is:
admit compressed c4 pages to pinned DRAM after prefill, eager-backup each new compressed
token, swap in only the indexer's top-k pages (CUDA-graph-safe, LRU device buffer,
`num_real_reqs` early-return for padding). It trades host bandwidth for long-range
capacity; no FLOP change. Lower priority than 1–4 (capacity lever, not latency lever).

### 6. Finish in-graph metadata (§5.2)
ARLE is well along (device-side topk/window/start-pos fills are captured). The remaining
SGLang shape is the explicit raw→full lazy upgrade with **warmup restore** so the captured
graph materializes metadata *inside* replay. The warmup-restore ordering is the documented
footgun (capture with warmup-host metadata → wrong meta on replay). Track against the
open `…-body-graph-zero-capture-pc11` error.

---

## The three "easy to misalign" details — and ARLE's handling

These three are in the implementation only (not the paper) and silently corrupt
long-context output if a port drops them. **All three are already wired in ARLE** — recorded
here so a refactor does not regress them:

1. **Output inverse-RoPE** — RoPE is applied at KV-write, then *un-applied* on the output's
   rope columns before O-proj. ARLE: `arle_dsv4_output_inverse_rope` in
   `cuda-kernels/csrc/misc/dsv4_attention.cu`, fused into the SW kernel
   (`attention.rs:864-866`). This is the same kernel from the validated long-context fix
   (`project_dsv4_compressed_attention_longctx_bug` memory).
2. **Compressor `ape` is learned, not pooling** — the window collapse is
   `Σ softmax(score_j + ape_j)·kv_j` with a learned per-relative-position bias, not a
   mean/strided pool. Mistaking it for pooling loses long-range info. ARLE:
   `compressor_forward` (`attention.rs:1054`) + `ape` tensor in
   `deepseek-spec/src/v4.rs` `DeepSeekV4CompressorTensorNames`.
3. **nope-FP8 / rope-BF16 KV** — the rope segment carries position phase (quant-sensitive,
   keep BF16); the nope segment is position-free (FP8 OK). Swapping precision drops
   long-context accuracy. ARLE: 584 B/tok layout `dsv4.rs:52`, asserted as 448 (FP8) +
   64×2 (BF16) + 8 (scales).

---

## Divergences (keep vs reconsider)

- **Compress-ratio enum.** SGLang hard-asserts `compress_ratio ∈ {0,4,128}`. ARLE uses
  threshold *bands*: `0`→SW, `1..=15`→CSA (overlap window, `coff=2`), `≥16`→HCA
  (non-overlap, `coff=1`) — `deepseek-spec/src/v4.rs` `from_compress_ratio:425`,
  `compressor_shape:237`. **Keep** — strictly more general, config-driven, and validated by
  the replica configs (`compress_ratios=[…,16,…]` in tests).
- **Weight absorption.** SGLang V4 is *un-absorbed* (single 512-dim KV head, FlashMLA reads
  the cache directly). ARLE's DSv4 path is **also un-absorbed** (single KV latent, 584 B/tok
  byte-identical layout). The "Absorbed-MLA at parity" line in
  [operator selection](2026-06-04-sglang-operator-selection-dsv4-qwen3moe.md) refers to the
  **generic V2/V3 MLA backend**, not the V4 `MQALayer` — not a contradiction. No action.
- **Static hash routing (`num_hash_layers`).** The analysis flags that SGLang's HEAD may not
  activate hash routing (config key `num_hash_layers` vs dataclass `n_hash_layers`). ARLE
  *does* model hash routing as a first-class branch (`moe_routing_kind:284`, `gate.tid2eid`
  tensor names, `moe_routes_from_scores` Hash arm). ARLE keeps it wired and config-gated;
  confirm against a real checkpoint `config.json` before assuming any layer is hash-routed.

---

## Appendix — code details: SGLang `c048ebd10` ↔ ARLE

Every SGLang block is a real excerpt at commit `c048ebd10`; every ARLE block is current
source. Kernels (`csrc/`) are the durable anchors; rewrite-file (`attention.rs`/`dsv4.rs`)
line numbers drift under concurrent edits so they are paired with symbol names.

### 1. Router scoring — `sqrt(softplus)`
**SGLang** (`python/sglang/srt/layers/moe/topk.py:856-871`):
```python
if scoring_func == "sigmoid":
    scores = gating_output.sigmoid()
elif scoring_func == "sqrtsoftplus":
    scores = torch.nn.functional.softplus(gating_output).sqrt()
...
scores_for_choice = scores.view(num_token, -1) + correction_bias.unsqueeze(0)
_, topk_ids = torch.topk(scores_for_choice, k=topk, dim=-1, ...)  # select on score+bias
topk_weights = scores.gather(1, topk_ids)                          # weight on raw score
```
**ARLE** (`crates/cuda-kernels/csrc/moe/dsv4_route.cu:196-203`):
```cuda
__device__ __forceinline__ float dsv4_route_score(float logit, int scoring_kind) {
  if (scoring_kind == DSV4_ROUTE_SIGMOID) return dsv4_route_sigmoid(logit);
  return sqrtf(dsv4_route_softplus(logit));        // sqrtsoftplus
}
```
ARLE Rust reference mirror: `deepseek-spec/src/v4.rs:305-316` (`router_scores_from_logits`)
and `:378` (select on `score+bias`, weight on raw `score`). **Identical contract.**

### 2. Compressor — learned online-softmax (the `ape` bias, NOT pooling)
**SGLang** (`python/sglang/jit_kernel/csrc/deepseek_v4/c4.cuh:181-208`):
```cuda
for (int32_t j = 0; j < 8; ++j)
  score_fp32[j] = cast<float>(score[j][i]) + cast<float>(bias[j][i]);   // bias == ape [8, head_dim]
... max_value = fmaxf(...);                                             // safe online softmax
for (int32_t j = 0; j < 8; ++j) {
  const auto exp_score = expf(score_fp32[j] - max_value);
  sum_product  += cast<float>(kv[j][i]) * exp_score;
  sum_exp_value += exp_score;
}
result[i] = cast<OutFloat>(sum_product / sum_exp_value);               // Σ softmax(score+ape)·kv
```
**ARLE** (`crates/cuda-kernels/csrc/misc/dsv4_attention.cu`, `:815/:865` ape index, `:919` weighted sum):
```cuda
float bias = dsv4_attn_bf16_to_f32(ape[(abs_pos % ratio) * width + col]); // learned per-rel-pos bias
...
float weight = expf(logit - max_logit);                                  // online softmax, same shape
```
The `ape` is a learned per-relative-position bias indexed `abs_pos % ratio` — a *selective*
summary, not a mean/strided pool. SGLang's `score_bias` is `[8, head_dim]` (= the analysis's `ape`).

### 3. Attention sink — folded into the sparse-MLA softmax denominator
**SGLang** (`python/sglang/srt/models/deepseek_v4.py:345,877`):
```python
self.attn_sink = nn.Parameter(torch.empty(self.n_heads, dtype=torch.float32))
... o = self.attn_mqa(..., attn_sink=self.attn_sink, ...)   # passed straight into FlashMLA
```
**ARLE** (`crates/cuda-kernels/csrc/misc/dsv4_attention.cu:592,604`):
```cuda
float sink = dsv4_attn_bf16_to_f32(attn_sink[sink_offset + head]);
if (threadIdx.x == 0) local_max = fmaxf(local_max, sink);   // sink competes in the max
...
if (threadIdx.x == 0) denom += expf(sink - max_shared);     // and in the denominator
```
A per-head learned escape slot in the softmax denom (StreamingLLM idea). When the selected
512 hold nothing relevant, mass flows to the sink instead of being smeared over noise.

### 4. Output inverse-RoPE — strip position phase before O-proj
**SGLang** (`python/sglang/srt/models/deepseek_v4.py:881-887`):
```python
fused_rope_inplace(o[..., -self.qk_rope_head_dim:], None, self.freqs_cis,
                   positions=positions, inverse=True)   # un-rotate the rope tail
```
**ARLE**: fused into the SW kernel (`dsv4_attention.cu`, "un-rotates the rope tail of" the
output), exposed as `arle_dsv4_output_inverse_rope` and cited at `attention.rs` SW path
("attends ... adds the sink, and un-rotates the rope tail"). Dropping this collapses
long-context output — the documented gotcha.

### 5. MHC — Sinkhorn doubly-stochastic mix + pre/post
**SGLang** Sinkhorn (`python/sglang/srt/layers/mhc.py:71-85`):
```python
T.reduce_sum(comb_frag, row_sum, dim=1)
comb_frag[j, k] = comb_frag[j, k] / row_sum[j] + eps          # FIRST iter: eps AFTER divide
T.reduce_sum(comb_frag, col_sum, dim=0)
comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)
for _ in T.serial(sinkhorn_iters - 1):                        # then n-1 standard iters
    ... comb_frag[j, k] = comb_frag[j, k] / (row_sum[j] + eps)
    ... comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)
```
SGLang post-mix (`deepseek_v4.py:1110-1114`):
```python
return (post.unsqueeze(-1) * x.unsqueeze(1)
        + (comb.unsqueeze(-1) * residual.unsqueeze(2)).sum(dim=1)).type_as(x)
```
**ARLE** Sinkhorn (`crates/cuda-kernels/csrc/misc/dsv4_mhc.cu:39-76`):
```cuda
__device__ void row_softmax_plus_eps(float *raw, int n, float eps) {  // FIRST iter
  ... raw[row*n+col] = raw[row*n+col] / denom + eps;                  // eps AFTER divide — matches
}
__device__ void row_normalize(float *raw, int n, float eps) { sum = eps; ... raw[...] /= sum; }
__device__ void column_normalize(float *raw, int n, float eps) { ... }
```
ARLE orchestration (`crates/infer-cuda/src/hc.rs:72-141` `gen_mhc_params`, `:145` `hc_pre`):
```rust
let mix_dim = (2 + hc_mult) * hc_mult;                 // == 24 for hc_mult=4, == SGLang mix_hc
dsv4_linear(ctx, &hc.mix_fn, stream, &mut mixes)?;      // project wide stream → mix weights
ffi::dsv4_mhc_params_cuda(.., pre_ptr, post_ptr, comb_ptr, .., config.hc_eps,
                          config.hc_sinkhorn_iters, ..) // sinkhorn → pre/post/comb
```
The **first-iteration `value/denom + eps` asymmetry** (eps added *after* the divide on the
very first row pass, standard `/(sum+eps)` thereafter) is present in *both* trees — the
strongest evidence ARLE's MHC is a faithful port, not an approximation.

### 6. Indexer — relu-scored lightning select
**SGLang** scoring (`python/sglang/srt/layers/attention/dsv4/indexer.py:87-92`):
```python
scores = torch.bmm(kv_values, q_float.transpose(1, 2))
scores = F.relu(scores)                 # negatives → 0 (reward only positive correlation)
scores = scores * weight.unsqueeze(1)   # learned per-head weight (weights_proj)
scores = scores.sum(dim=2) * kv_scales  # sum over 64 heads → per-token relevance scalar
```
SGLang fused Q (`indexer.py:584-588`): `fused_q_indexer_rope_hadamard_quant(q, weight, ...)`
(RoPE → Hadamard → FP8 in one kernel). **ARLE**: `attention.rs` `csa_select:1155` →
`dsv4_csa_select_cuda` (top-k block selector over the FP8 indexer pool); indexer Q prep in
`dsv4_attention.cu`. Fixed `k=512` → fixed output buffer → CUDA-graph-safe (same rationale both sides).

### 7. KV layout — 584 B/token, nope-FP8 / rope-BF16
**SGLang** (`python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py:108`):
```python
assert bytes_per_token == 448 + 64 * 2 + 8, (
    "DSV4 KV layout: qk_nope_head_dim FP8 (448) + qk_rope_head_dim BF16 ...")
```
**ARLE** (`crates/infer-cuda/src/dsv4.rs:51,69-78`):
```rust
const DSV4_FLASH_KV_BYTES_PER_TOKEN: usize = 584;
ensure!(nope_dim == 448 && rope_dim == 64,
    "DSv4 MLA KV arena only wires the FlashMLA MODEL1 NoPE=448/RoPE=64 pack (584 B/token) ...");
```
Same 448(FP8)+128(BF16)+8(scale) byte budget. **Divergence**: ARLE gates the FP8 *decode*
(`alloc_fp8_arena` is `bail!`) and runs a bf16 SW-ring + bf16 compressed-pool forward until
that bf16 path parity-matches the oracle (`dsv4.rs:83-104`) — correctness before the FP8 perf flip.

### 8. Waterfill — shared expert as the 9th routed expert (**ARLE: missing**)
**SGLang** weight + LCG sample (`python/sglang/srt/layers/moe/deepep_waterfill.py:181-217`):
```python
w = tl.where(target_total > rank_load_r, target_total - rank_load_r, 0)   # waterline headroom
w_vec = tl.where(src_rank == r, w_vec, (w_vec * 10) // 11)                # remote ranks ×10/11
total_w += tl.where(present, w_vec, 0)
token_seed = token_seed * 1664525 + 1013904223                            # LCG hash
u = token_seed % total_w
... pick = (u >= cum) & (u < cum + w_vec); chosen = tl.where(pick, r, chosen)  # cumulative scan
```
Expert-id remap + fused slot (`:243`, `:44`):
```python
new_id = old_id + (old_id // old_experts_per_rank)   # shift ids to make room for the shared slot
torch.empty(0, topk + 1, ...)                        # topk 8 → 9 (shared = the +1 slot)
```
Skip on small decode batch (`:364,412`): `MIN_BATCH_FOR_BALANCE = 64`.
**ARLE**: none. Lands as a routing-layer load shaper in `infer-moe/src/route.rs` (target-expert
set) + fused-shared gating in `infer-cuda/src/moe.rs`; **downstream of DeepEP dispatch being
served** (port-plan Piece 4) — nothing to balance until then.

### 9. MTP NextN — draft consumes the un-collapsed `[n, 4·d]` stream
**SGLang** (`python/sglang/srt/models/deepseek_v4_nextn.py:147-154`):
```python
hc_flat = forward_batch.spec_info.hidden_states.view(n_tokens * self.hc_mult, d)
h_proj_hidden_states = self.h_proj(self.hnorm(hc_flat)).view(n_tokens, self.hc_mult, d)  # per-lane
e_proj_hidden_states = self.e_proj(self.enorm(hidden_states))                            # new embed
hidden_states = e_proj_hidden_states[:, None, :] + h_proj_hidden_states                  # broadcast → [n,4,d]
```
**ARLE**: structure modeled in `deepseek-spec/src/v4.rs` `DeepSeekV4MtpTensorNames`
(`enorm`/`hnorm`/`e_proj`/`h_proj`/`hc_head`); the 4-lane seed mirrors
`hc.rs:initial_stream_from_embeddings`. Draft layer forces `compress_ratio=0` (cheapest SW-MQA)
both sides. Full 4-lane draft consumption is the in-progress piece (🟡, MTP contract wins).

---

## Cross-links

- Port plan (where these land): [`projects/2026-06-04-dsv4-port-plan.md`](../projects/2026-06-04-dsv4-port-plan.md)
- Overlap/operator gaps (perf phase): [`research/2026-06-04-sglang-operator-selection-dsv4-qwen3moe.md`](2026-06-04-sglang-operator-selection-dsv4-qwen3moe.md)
- Path research: [`research/2026-06-01-dsv4-sglang-path-research.md`](2026-06-01-dsv4-sglang-path-research.md)
- Controlling alignment plan: [`plans/2026-06-01-dsv4-sglang-path-alignment.md`](../plans/2026-06-01-dsv4-sglang-path-alignment.md)
- Config/tensor authority: `crates/deepseek-spec/src/v4.rs`
- Long-context RoPE fix memory: `project_dsv4_compressed_attention_longctx_bug`
- Parity-confounder lesson: [`experience/errors/2026-06-04-dsv4-parity-prompt-id-confounder.md`](../experience/errors/2026-06-04-dsv4-parity-prompt-id-confounder.md)
