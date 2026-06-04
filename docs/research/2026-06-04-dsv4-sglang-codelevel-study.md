# DeepSeek-V4 on SGLang — code-level study & ARLE borrowable map

**Date:** 2026-06-04. **Type:** research (learning + cross-reference, not a wins/bench entry).
**Mechanism sources (verifiable):**
- SGLang side: source-read at commit `c048ebd1` (2026-05, upstream main) for the architecture
  mechanisms, cross-checked against the in-repo HEAD `8980eb82` survey
  ([`2026-06-04-sglang-operator-selection-dsv4-qwen3moe.md`](2026-06-04-sglang-operator-selection-dsv4-qwen3moe.md))
  for the overlap/scheduling items.
- ARLE side: every "ARLE state" cell below is grounded in current source
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
| 584-byte KV (nope-FP8 448B + rope-BF16 128B + scales), 4-pool layout | `deepseek_v4_memory_pool.py` `#L356,93` | `dsv4.rs` `DSV4_FLASH_KV_BYTES_PER_TOKEN=584:52`, `Dsv4MlaKvArena:55` | ✅ |
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

## Cross-links

- Port plan (where these land): [`projects/2026-06-04-dsv4-port-plan.md`](../projects/2026-06-04-dsv4-port-plan.md)
- Overlap/operator gaps (perf phase): [`research/2026-06-04-sglang-operator-selection-dsv4-qwen3moe.md`](2026-06-04-sglang-operator-selection-dsv4-qwen3moe.md)
- Path research: [`research/2026-06-01-dsv4-sglang-path-research.md`](2026-06-01-dsv4-sglang-path-research.md)
- Controlling alignment plan: [`plans/2026-06-01-dsv4-sglang-path-alignment.md`](../plans/2026-06-01-dsv4-sglang-path-alignment.md)
- Config/tensor authority: `crates/deepseek-spec/src/v4.rs`
- Long-context RoPE fix memory: `project_dsv4_compressed_attention_longctx_bug`
- Parity-confounder lesson: [`experience/errors/2026-06-04-dsv4-parity-prompt-id-confounder.md`](../experience/errors/2026-06-04-dsv4-parity-prompt-id-confounder.md)
