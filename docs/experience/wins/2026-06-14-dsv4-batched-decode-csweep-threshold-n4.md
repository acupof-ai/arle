# DSv4 batched-decode — c-sweep N≈3 threshold + full mechanism (per-row attn kernel-bound, TP Q-allgather, DP-attn gap)

## Goal
License the "c>1 → batched decode by default" flip ckl asked for, AND research the
mechanism through (per ckl: "不符合预期 = 没做好" — find what's under-optimized vs
what's fundamental). Tree 0b70c78c, 8×H20 TP=8/EP=8, DeepGEMM native.

## A/B grid — c-sweep (decode-isolated aggregate tok/s)
Arm A = current production `--spec-type mtp --mtp-draft-tokens 2` (c>1 → MTP-per-row,
batched lane disabled by `executor.rs:1563 !spec_decode`). Arm B = `INFER_DSV4_BATCHED_DECODE=1`,
no mtp (the batched lane). Head-to-head, ~2300-tok prompt:

| c | A (mtp) | B (batched) | Δ% | | short-96tok prior |
|---|---------|-------------|-----|---|---|
| 2 | 49.61 | 46.99 | **−5.3%** | | −6.6% |
| 3 | 52.03 | 58.13 | **+11.7%** | | (c≥4 win) |
| 8 | 46.38 | 73.65 | **+58.8%** | | +32.7% |

**Crossover at c=2↔3** (production ~2300-tok); short-prompt it's c≥4. Context moves it
left (long KV heavies MTP's per-row forward). Needle exact in both arms.

## VERDICT — policy `c==1,2 → MTP; c≥3 → batched` (N≈3, threshold is FUNDAMENTAL)

## Mechanism (measured — the "研究透")

**M1 — MTP acceptance is a c-INDEPENDENT structural ~2×.** tok/fwd/row = 1.989 (c=2)
/ 1.980 (c=8), flat (Δ<0.5%). MTP's per-row 2× multiplier is the whole reason batched
must out-amortize ~2× to break even.

**M2 — batched forward amortizes SUB-LINEARLY.** ms/token: 23.05 (b1) → 21.40 (b2) →
16.26 (b4) → 13.58 (b8). b=2 only reaches 1.07×/token amortization — can't recover
MTP's 2× → loses. b≥3 overtakes.

**M3/Q1 — per-row attention is KERNEL-bound, not overhead-bound (the load-bearing
result).** The batched lane batches MoE (60.8% of GPU work, grouped) but attention is
per-row: FlashMLA `flash_fwd_splitkv_mla_fp8_sparse_kernel` runs `grid=(1,1,78)
gridX=1`, ONE query row per launch, 74,486 launches. Decompose the per-row attention
wall (b8 GPU device-time): **kernel ~72% · launch/host gap ~25% · per-row memcpy ~3%.**
⇒ The "flat 257k→263k ns/row" is REAL per-row KV/compute, not memcpy. **Phase B
(batched `sparse_decode_fwd(b=N)`) reclaims the ~28% launch+memcpy and improves the
gridX=1 low-occupancy at high b — so it helps HIGH-c, but does NOT lower the N≈3
threshold** (at b=2 the kernel dominates and memcpy removal is only ~3%; the threshold
is set by MTP's 2×, which Phase B does not touch). N≈3 is a permanent MTP/batching
tradeoff, NOT a Phase-A artifact.

## 预期-差距-没做好 (expectation vs measured → under-optimized vs fundamental)

| Point | Not-done-well? | Finding |
|---|---|---|
| **per-row attention un-batched** | **YES (Phase B)** | 23.7% of GPU compute but dominates wall-clock via 74k tiny gridX=1 launches + 25% launch-gap. Kernel-bound → Phase B helps high-c (reclaims launch-gap + occupancy), does NOT lower N≈3. |
| **TP Q-allgather + skew** | **YES** | NCCL collectives **15.2%** of kernel time; `flashmla_q_allgather` (Q all-gather, MLA latent is replicated per rank) = **10.4%** (biggest single collective); AllReduce per-rank skew 4–9× max/avg (lockstep-wait). Above the 10% "minor" bar. |
| **DP-attention missing** | **YES (the real parallelism gap)** | SGLang single-node DeepSeek-V3.2 = `--tp 8 --ep 8 --dp 8 --enable-dp-attention`; ARLE lacks DP-attn. It directly removes the Q-allgather (the 10.4% above) + the skew. |
| c=2 batched loss (−5.3%) | NO — fundamental | MTP's c-independent ~1.98×/row (M1) vs b=2's weak 1.07× amortization (M2). N≈3 permanent. |
| long-ctx decode | NO — O(topk), working | short 23.45 vs 31K 24.10 ms/step = **1.03× flat** (topk_unified=640 const; max_compressed_keys from config cap not seq_len). The 32K c=8 3.4× slowdown is one-time PREFILL (~8.5×) + concurrency, NOT decode. |
| PP absent | NO — correct | SGLang uses PP only multi-node (`--nnodes≥2`); single-node 8-GPU DeepSeek = TP/EP/DP, no PP. ARLE's TP8/EP8 no-PP matches. |

## Revised concurrency lever ranking (evidence-grounded, production ~2300-tok)
1. **Phase B batched FlashMLA decode** — reclaims the 25% per-row launch-gap + gridX=1
   occupancy; raises the +58.8% @c=8 (does NOT lower the c≥3 threshold).
2. **DP-attention** (the missing SGLang config) — removes the 10.4% Q-allgather + the
   lockstep skew; the real parallelism gap (not PP).
3. **CUDA graph** — also reclaims the 25% launch-gap (overlaps with #1's launch lever).
Each re-baselined after the prior. MoE is already grouped (well-amortized, −66%/row).

## Correctness
Needle exact in both arms (512 3/3 DET, 6000 3/3, all retrieve 738291 = MoE non-det).

## Rule
- **A "c>1 → X" default flip must be c-swept** — c=8 +48% did NOT generalize to c=2
  (−5.3%); licensed threshold is c≥3 (prod) / c≥4 (short). Concurrency is a shape axis.
- **"Sub-expectation" splits into not-done-well vs fundamental — measure, don't assume.**
  Here per-row-attn + TP-Q-allgather + missing-DP-attn ARE under-optimized; but c=2-loss
  (MTP's real 2×), long-ctx-decode (real O(topk)), and no-PP (correct single-node) are
  NOT — the data exonerated three of the six suspects.
- **"flat per-row ns" ≠ "real per-row work"** by itself — per-row launch overhead is
  also flat-per-row. The kernel-vs-memcpy-vs-gap GPU-device-time split (72/3/25) is what
  settled it (kernel-bound). [[feedback_measured_floor_is_not_physical_floor]]
- Implement the flip as `rows >= N` (N=3), locked default, CLI-tunable
  ([[feedback_kv_features_default_on]]); Phase B + DP-attention stack on top at c≥3.
