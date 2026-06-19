# DSv4-Flash decode performance model — what bounds it, what doesn't, and why

**Date:** 2026-06-19. **Driver (ckl):** "理解怎么样才能高性能，而不是永远验证猜测" —
build the first-principles model so the levers are *derived* from structure, not
found by A/B fishing.

**What this supersedes.** The smoke-shape lever docs
([`../plans/2026-06-06-dsv4-decode-6ms-remaining-levers.md`](../plans/2026-06-06-dsv4-decode-6ms-remaining-levers.md),
`-6ms-dag`, `-residual-gemv-fusion`) chased B=1 kernels measured on an 8-token
shape where the real wall (`csa_select`) is trivial. They were overturned by the
wall-clock trace ([`../plans/2026-06-06-dsv4-pd-systematic-analysis.md`](../plans/2026-06-06-dsv4-pd-systematic-analysis.md)).
This doc consolidates the grounded model and labels every claim `[SOLID]`
(measured / official source) or `[Hypothesis]`.

---

## 0. The one-line answer

DSv4-Flash decode is **not** weight-bandwidth-bound (the dense-LLM intuition).
Its active weight read is tiny (13B active → **~0.25–0.4 ms/GPU** depending on
the expert dtype fork, §1.1 — FP8 ~0.4 ms / FP4 ~0.25 ms).
The decode wall is **per-token compute + DSA sparse-prepare + per-layer
collectives + the ~860-kernel serial chain**, and *which* of those dominates
**depends on context length**:

- **short context** (your ~28-tok prompt, 40 t/s): per-token compute + chain
  latency + collectives. ~22–25 ms/step.
- **long context** (≥4096, the SLO shape): **`dsv4_csa_select` sparse top-k
  selection dominates 74.9%** and runs on **1 SM** at B=1. `[SOLID]`

Throughput "doesn't scale" because almost all of that work is *genuinely
per-token* — only a ~7 ms fixed floor (collectives + launch + skew) amortizes,
which is why batching gives ~1.4×, not near-linear.

---

## 1. The model (config — now fully reconciled) `[SOLID]`

| field | value | source |
|---|---|---|
| total / active params | **284B / 13B per token** | HF `deepseek-ai/DeepSeek-V4-Flash` |
| weight dtype | **mixed — see §1.1** → 149 GB on disk | source `WeightFormat` + byte arithmetic |
| num_hidden_layers | **43** | reconciles below; `calls=43/step` measured `[SOLID]` |
| hidden / hc_mult | 4096 / **4** → stream_dim 16384 | `dsv4_microbench.rs:6,34` |
| MLA | head_dim(latent) 512, heads 64, q_lora 1024, o_lora 1024, o_groups 8, qk_rope 64 | `deepseek-spec/v4.rs` shape derivation |
| DSA indexer | index_n_heads 64, index_head_dim 128, index_topk 512 | `v4.rs::indexer_shape` |
| MoE | 256 experts, top-8, n_shared 1, moe_inter 2048, all-MoE | reconciles below |
| MTP / NextN | **1 head**, chained EAGLE-style | pod `config.json`, `num_nextn_predict_layers=1` `[SOLID]` |
| hardware | 8× H20: 96 GB, **4.0 TB/s HBM**, **148 BF16 / 296 FP8 TFLOPS**, **78 SM**, 900 GB/s NVLink | reference-baseline doc |

**Reconciliation** (closes the "43 vs 61 layers" contradiction; the on-disk
**size is left as an OPEN fork**, §1.1): 43 layers × 256 experts × (3 GEMMs ×
2048×4096) ≈ **275B** routed-expert params; + attention/embed/head ≈ **284B
total** ✓ (matches HF). Active = top-8 routed (8×25M) + shared + attention(43)
+ head ≈ **13B** ✓ (matches HF). **61 layers would give ~18B active —
inconsistent with the official 13B, so 61 (in the 06-06 doc) was an
inherited-from-V3 assumption; 43 is correct.** The active *count* (13B) is solid;
the active *bytes* depend on the expert dtype (§1.1).

## 1.1 Quantization map (the load-bearing detail) — and the OPEN size fork

Getting this wrong throws off the on-disk size and the bandwidth floor. The
runtime picks the format per checkpoint via the `WeightFormat` enum; **both an
FP8 and an FP4 routed-expert path exist** (`Fp8BlockScaled`/`Fp8PerShard` at
loader.rs:4080; `Fp4E2M1Group` at loader.rs:4098). Which one the deployed
DeepSeek-V4-Flash checkpoint uses decides the size — and I **cannot verify it
from local source**:

| branch | routed-expert format | on-disk | active bytes | decode expert GEMM |
|---|---|---|---|---|
| **FP8 (leaning — see below)** | FP8 e4m3 block-scaled, 1 B/param | **~284 GB** | 13B×~1B ≈ 13 GB → **1.6 GB/GPU → ~0.4 ms** | **DeepGEMM** `dsv4_fp8_grouped_swiglu_decode` |
| FP4 | NVFP4 = E2M1 (0.5 B) + per-group e4m3 scale + per-tensor FP32 scale (loader builds *both* `qscale_fp8`+`scale_f32`, loader.rs:4098,4116) | ~149–165 GB | 13B×~0.6B ≈ 8 GB → **1 GB/GPU → ~0.25 ms** | custom `dsv4_fp4_grouped_gemv` (**not** DeepGEMM) |

**`[SOLID]` Leaning FP8 / ~284 GB:** (1) `dsv4.rs:8` states "**256 FP8 experts**
… don't fit one GPU" (FP8 → 284 GB is *why* it needs 8 GPUs); (2) the production
MoE decode lane is **DeepEP-LL dispatch → DeepGEMM masked grouped GEMM**
([`moe-batching-eplb-scope`](2026-06-15-dsv4-moe-batching-eplb-scope.md)), and
**DeepGEMM is FP8-only** — FP4 experts would route to `dsv4_fp4_grouped_gemv`.
The earlier "FP4 / 149 GB" leaned on the web-search reference-baseline doc + one
"149 GB model" bench line, which **conflict** with the code. `[Hypothesis —
resolve on pod]` `ls -la $MODEL_DIR` + `config.json` `quantization_config` is the
deciding evidence; until then the size is **284 GB (FP8, likely) or ~149 GB (FP4)**.

**Other weights (independent of the fork) `[SOLID source-path]`:**
- **shared expert** (n_shared=1, every GPU): **FP8 block-scaled e4m3** via DeepGEMM (`Dsv4Fp8DeepGemmWeightCache`, dsv4.rs:262-264).
- **KV cache / token:** **FP8 e4m3 NoPE** (448 MODEL1 / 512 V32) + **bf16 RoPE** (128) + E8M0 scales → **584 / 656 B/token** (dsv4.rs:33,52); FlashMLA sparse-FP8 reads it.
- **attention proj** (wq/wkv/wo, compressor): FP8 block-scaled or `Dsv4Fp4BlockScaled` (checkpoint-dependent); RoPE/norms/gate-bias bf16.
- **Block-scale formats:** DeepGEMM FP8 uses **E8M0/UE8M0** (weight_block_size 128×128); **GLM-5.2** ships **F32 `weight_scale_inv`** for the **1D2D FP8 GEMM** (commit `5583c77b`).

**Either branch, the floor (~0.25–0.4 ms/GPU) is « the 22 ms wall** → the
"not weight-bandwidth-bound" conclusion (§2) is robust to the fork; the dtype
matters for VRAM fit and *which* expert GEMM kernel is on the hot path (DeepGEMM
FP8 vs custom FP4 gemv — different efficiency), not for the bound.

---

## 2. The three floors — and why none is the wall at B=1 `[SOLID]/[算]`

Per GPU, TP8/EP8, one decode token:

| floor | quantity | H20 peak | ideal | B=1 realized eff. |
|---|---|---|---|---|
| **weight bandwidth** | 13B active, /8 ranks (FP8: ~13 GB→1.6 GB/GPU; FP4: ~8 GB→1 GB/GPU — §1.1) | 4.0 TB/s | **~0.25–0.4 ms** | — |
| **compute** | ~6 GFLOP/GPU (=2×active/8) | 296 FP8 TFLOPS | ~0.02 ms | MFU ~0.1% |
| **collectives** | 86 all-reduce + Q-allgather | NCCL latency-floor ~17 µs (flat 14 KB→459 KB) | **~2.3 ms** | latency-bound |

Sum of floors ≈ **2.6–2.7 ms**. Measured B=1 step ≈ **22–25 ms**. So the gap is
~8–10× *over the collective floor* and ~55–90× *over the weight-bandwidth floor*
(FP8/FP4 fork, §1.1) — **the bytes and FLOPs are not the constraint.** This is the official framing:
"decode is NOT weight-bandwidth-bound; it's comm + sparse-prepare + launch
bound" (reference-baseline doc).

> **Why "raise M to amortize the weight read" (dense-LLM intuition) does not
> apply here:** there is almost no weight read to amortize (0.3 ms). The
> per-token wall is *compute the token does* — and that work is paid per token,
> not per weight-load.

---

## 3. The context-dependent bottleneck (the heart of the model)

### 3a. Short context — per-token compute + chain latency `[SOLID]`

Stage profiler, B=1, real CUDA-event GPU time
([`../experience/wins/2026-06-13-dsv4-concurrency-baseline-serial-capped.md`](../experience/wins/2026-06-13-dsv4-concurrency-baseline-serial-capped.md)):

| stage | % GPU | applied to a 22.5 ms step |
|---|---|---|
| MLA attention (prepare machinery: compressor + indexer + pack + FlashMLA math) | **41%** | ~9.2 ms |
| shared dense FFN | 17% | ~3.8 ms |
| MoE route + all-reduce | 15% | ~3.4 ms |
| attn all-reduce | 4% | ~0.9 ms |
| rest (mHC Sinkhorn ~3 ms, indexer, lm_head, embed, rope, norms) | 23% | ~5.2 ms |

The step is **~92% GPU-active** `[SOLID]` (whole-step CUDA graph A/B is
wall-neutral, 8× confirmed) — so it is **not** host-launch-bound. The GPU is
busy, but busy running ~860 tiny **M=1** kernels each at near-zero occupancy: a
single M=1 FP8 projection reads 0.79 MB (0.24 µs at peak) yet takes **51 µs
(DeepGEMM) / 26 µs (scalar)** — ~130–200× over its byte floor, because M=1 fills
only ~2 SMs of 78. The wall is **low occupancy across a serial chain**, not
bandwidth.

### 3b. Long context (≥4096) — `csa_select` on one SM `[SOLID]`

Wall-clock trace at the 4096 SLO shape
([`../plans/2026-06-06-dsv4-pd-systematic-analysis.md`](../plans/2026-06-06-dsv4-pd-systematic-analysis.md),
exclusive-interval, ncu-licensed):

| decode bucket (4096 steady) | per-token | % |
|---|---|---|
| **attention / sparse-prepare — `dsv4_csa_select` dominant** | ~103 ms | **74.9%** |
| MoE expert path | 12.4 ms | 9.0% |
| dense linear | 6.0 ms | 4.4% |
| NCCL | 5.5 ms | 4.0% |
| HC (mHC) | 3.9 ms | 2.9% |
| FlashMLA math itself | ~5.5 ms | small |

**Root cause (ncu, source-confirmed `dsv4_attention.cu:1546`):** the hand-rolled
selector keys `token = blockIdx.x` → at B=1 decode it launches **grid (1,1,1) →
1 active SM of 78 = 1.28%, 12.5% occupancy, "98.72% estimated speedup from
launch-geometry underutilization."** The scoring (≈1024 candidate blocks × 64
heads × 128 dim) + bitonic sort all run on one SM. `csa_select` **scales with
context** (more candidate blocks), so it is invisible at 28 tok and dominant at
4096. This is the single biggest long-context single-request lever.

**Status:** the official multi-SM replacement (`fp8_paged_mqa_logits`, vendored
at `cuda-kernels/vendor/deepgemm/`, the same kernel SGLang's DSA backend uses)
was wired as `dsv4_dsa_official` and **default-on'd**
([`../experience/wins/2026-06-07-dsv4-official-dsa-default-on.md`](../experience/wins/2026-06-07-dsv4-official-dsa-default-on.md)).
`[Hypothesis — verify in the current binary]` that the official multi-SM path,
not the 1-SM `dsv4_csa_select` fallback, is the active long-context selector;
the hand-rolled kernel still exists in source.

---

## 4. Why aggregate throughput "跑不大" (the scaling model) `[SOLID]`

c-sweep arithmetic: **`T_step(B) ≈ 7 ms floor + 15.7 ms/req`** (B=1/4/8/16 →
22.5/74.4/139.4/258.1 ms). Two terms, two fates:

- **7 ms fixed floor** (collectives + launch + skew) — the **only** part that
  amortizes with B → it is the entire source of the weak ~1.4× aggregate gain
  (the floor is just 2.7% of the step at B=16).
- **15.7 ms/req** — genuine per-token compute (sparse-prepare + attention + MoE
  + mHC). It does **not** amortize, because each token does its own DSA prepare,
  its own attention, and activates its own expert set.

Three structural reasons the per-req term is irreducible by batching alone:

1. **DSA sparse-prepare is per-token.** Each new token must be compressed,
   indexed, top-k-selected, packed. Batching N tokens = N× prepare.
2. **MoE is weight-read-bound on expert *tiles*, and #distinct-experts grows
   with B** `[SOLID]` ([`2026-06-15-dsv4-moe-batching-eplb-scope.md`](2026-06-15-dsv4-moe-batching-eplb-scope.md),
   `moe.rs:190,246`). Per-rank cost ≈ (#hit experts on that rank) × one
   weight-read tile; #hit grows with c → MoE weight-read *grows* with B. EP=8
   spreads it (→1.4×, not flat). On 8 GPUs each rank owns 256/8=32 experts; at
   high B all 32 are hit → the per-rank MoE read saturates. **This is why the
   industry path is scale-EP across *many* GPUs (EP=72) — unavailable at 8.**
3. **The executor batches rows, but attention historically fell off the batched
   kernel.** FlashMLA decode was `seq_len==1`-gated → at c>1 attention dropped to
   the prefill path. Batched-FlashMLA decode (Phase B) closed this; co-enabling
   batched-decode + batched-MTP is the c>1 throughput axis (+58% @c=8, +77% @c=8
   for batched-MTP). MoE was *never* per-row (DeepEP-LL grouped, SGLang parity).

---

## 5. Per-layer decode operator map (B=1, one GPU) `[SOLID]`

`~860 kernels/token/GPU` — ~20 per layer × 43:

```
embed → initial stream
per layer ×43:
  ① mHC: gen_mhc_params (Sinkhorn ×20 on 4×4 + 24-elem mixer) + mhc_pre_rms_norm   stream[16384]→normed[4096]
  ② wq_a [1024×4096]·M=1 → c_q; q_norm
  ③ wq_b [heads·qk ×1024] → q
  ④ wkv  [512×4096] → kv latent[512]; kv_norm; RoPE
  ⑤ compressor (CSA/HCA): wkv/wgate [2·512×4096], APE, ring state-update
  ⑥ indexer: wq_b[8192×1024] + weights_proj[64×4096] + k-compress + scoring + top-k(512)   ← csa_select, the long-ctx wall
  ⑦ pack KV → FP8 pool
  ⑧ FlashMLA sparse decode: Q[64,512] × selected-512 KV → attn[64,512]
  ⑨ wo_a [8192×4096] + wo_b [4096×8192] → attn_out[4096]
  ──[A] all-reduce #1 (attn) [1,4096] NCCL ~17µs──
  ⑩ mHC post + expand
  ⑪ ffn_norm
  ⑫ router gate [256×4096] → sigmoid + noaux_tc → top-8
  ⑬ DeepEP-LL dispatch (EP=8)
  ⑭ routed expert (~1/GPU): w1[2048×4096] w3[2048×4096] SwiGLU w2[4096×2048] — FP8 DeepGEMM masked grouped GEMM (or custom FP4 gemv if FP4 ckpt — §1.1)
  ⑮ shared expert: same 3 GEMMs (every GPU) — FP8 DeepGEMM
  ⑯ DeepEP-LL combine (EP-reduce)
  ──[B] all-reduce #2 (moe) [1,4096] NCCL ~17µs──
  ⑰ residual add
final RMSNorm → lm_head [~129k×4096] → sample
```

Each `M=1` GEMM fills ~2 SMs of 78 → bandwidth-bound at ~3–5% of peak. `[SOLID]`

---

## 6. Levers — ranked, license-or-kill status `[SOLID]`

**Landed (measured):**
- Official DSA selector default-on (the long-ctx `csa_select` 1-SM fix) — flat 26 ms base @4096.
- Official FlashMLA `sparse_fwd` + FP8 DeepGEMM prefill (7.2 s → 3.48 s @4096).
- Batched-FlashMLA decode (c≥4 default), DeepEP-LL grouped MoE (SGLang parity), batched-MTP (+77% @c8), compressor-batch (+58% @c4), decode-proj DeepGEMM (+6%), B=1 comm-overlap (default, +9%), mHC warp-tail (+5.9%).

**Washed / killed (predicted by the model — none changes per-token work or M):**
whole-step & per-layer CUDA graph (−5%/wash), faster all-reduce (3.5× faster →
0 wall change), per-kernel shaving, DP-attention (H20 compute-saturated; same-chip
SGLang #23896 measures DeepEP/DP −3–6%), replicated-attn (−48%), EPLB (decode-wash:
balances rows, but decode imbalance is *which experts hit*).

**Open — the real levers, ranked:**
1. **Fix the MTP chain bug** (biggest B=1 win, *no kernel*). SGLang's identical
   1-NextN head gets **2.44 accepted tok/step, 1.8× decode @bs=1** (LMSYS,
   H200 TP8) on a linear `--num-steps 3 --eagle-topk 1` chain. Ours accepts
   **1/4** (drafts loop `[223,4489,223,4489]`, 2-cycle) — **3.4× worse = a chain
   bug, not a head-capacity wall** ([`2026-06-11-dsv4-mtp-eagle-and-decode-operators.md`](2026-06-11-dsv4-mtp-eagle-and-decode-operators.md)).
   The recent topk2/branch rework regressed it further (current MTP 35 < no-spec
   40, verify ≈2.4× a step — [`../experience/errors/2026-06-18-dsv4-tp4-topk2-branch-regression.md`](../experience/errors/2026-06-18-dsv4-tp4-topk2-branch-regression.md)).
   Target shape = linear chain + **frozen-KV verify** (verify reads frozen target
   KV, never re-runs the compressor/indexer per draft — this is *how* spec
   amortizes the per-token sparse-prepare).
2. **Batched-decode × batched-MTP co-enabled** for c>1 aggregate (the executor
   axis; SGLang batches all rows into one forward → near-linear vs our ~1.4×).
3. **Confirm/keep the official multi-SM DSA selector hot at long context**
   (§3b) — the per-token sparse-prepare is the dominant *per-token* cost, so any
   amortization (spec, batching) and the selector kernel itself compound here.

---

## 7. Honest gaps + corrections to prior framings `[SOLID about the corrections]`

This model **corrects an earlier roofline-M framing** (this session, pre-research)
that imported the dense-LLM "decode is weight-bandwidth-bound, raise M to amortize
the weight read" intuition. Adversarial verification against the official HF spec
+ the wall-clock trace broke it on the central point:

- **DSv4-Flash is not weight-bandwidth-bound** (0.3 ms floor; 13B active, FP4).
  There is no large weight read to amortize. "Raise M" only amortizes the ~7 ms
  fixed floor → ~1.4×; the dominant per-token work (sparse-prepare, MoE tiles,
  attention, mHC) does not amortize.
- The ridge point `M*≈37` (FP8 arithmetic-intensity crossover) is **not** the
  operating target it was framed as: tensor-core MFU needs M≈64–128 to plateau,
  and — more importantly — the binding cost here isn't GEMM MFU at all, it's
  `csa_select` (long-ctx) and per-token sparse-prepare.
- "Only M matters" was too strong: at B=1, overlapping the fixed collective
  (comm-overlap, now default) gave a real +9% without touching M.

**Remaining gaps (declared, not silently passed):**
- `[Hypothesis]` the current binary runs the official multi-SM DSA selector, not
  the 1-SM `dsv4_csa_select` fallback, at long context — confirm by symbol/probe.
- `[Hypothesis]` the short-prompt no-spec regression (historical clean 44 →
  current ~35.6 raw / 39.7 with comm-overlap+compressor-batch on `.62`; a clean
  `3e3e50e0` rebuild now gives 32.7) is environmental (build profile
  `release-fast` vs `release`, DeepGEMM JIT cache warmth, co-tenant) — **not yet
  root-caused.** The historical 44/53 has not reproduced from clean bundle builds.
- vocab_size for DSv4-Flash is taken as ~129k (test fixture); not confirmed
  against the pod `config.json`.

---

## 8. Why the current number is ~40 t/s (decomposed) `[SOLID]`

Current default no-spec B=1 = **39.7 t/s** (`087df440`, profiling-off, .62,
short prompt). Not the M=1 floor alone — three stacked terms:

1. **MTP is off** because it's currently net-negative (35 < 40, verify 2.4×/step,
   accept ~1.9/step) → lose the historical +18–20%. Recoverable: it's the chain
   bug (§6 lever 1), not a wall.
2. **No-spec baseline itself regressed** 44 → ~35.6 raw on current `.62`
   (environmental, §7 gap) → comm-overlap + compressor-batch add back to 39.7.
3. **The ~22 ms floor is real** — short-context per-token compute + chain
   latency at ~3–5% GPU occupancy on H20's weak SMs (§3a). 6 ms is an H100
   number; the H20 no-spec floor is ~20–26 ms, and reaching single-digit ms
   *requires* working spec (§6 lever 1).

**So the path to high performance is derived, not searched:** (a) root-cause the
environmental no-spec regression (cheapest, surest), (b) fix the MTP chain to
SGLang's 2.44 tok/step → ~70–80 t/s B=1 with no kernel work, (c) keep the
official multi-SM selector hot for long context, (d) co-enable batched
decode+MTP for c>1 aggregate. A/B confirms each *derived* prediction; it is not
the search itself.

---

**Sources:** HF `deepseek-ai/DeepSeek-V4-Flash` · SGLang #23896 · LMSYS H20
serving blog · pod `DeepSeek-V4-Flash/config.json` (`num_nextn_predict_layers=1`)
· wins/errors cited inline · `infer-cuda/src/{dsv4,attention,moe}.rs`,
`deepseek-spec/src/v4.rs`, `cuda-kernels/csrc/misc/dsv4_attention.cu`,
`cuda-kernels/vendor/deepgemm/`.
