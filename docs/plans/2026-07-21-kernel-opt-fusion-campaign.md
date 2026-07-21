# Kernel optimization & fusion campaign

> Status: Active — 2026-07-21 (self-contained after non-July doc purge)  
> Controlling campaign for what is left to optimize/fuse after production DSv4
> all-on and licensed Qwen FA3-prefill / decode-MoE defaults.

## Verdict

**B=1 hand-kernel era is closed.** Live spine already covers FlashMLA, fused
WQKV DeepGEMM, FA3 prefill, decode-band MoE, DSA, DSpark (substrate).

**Remaining wall-clock:**

```
rank = FA3-decode default policy  >  DSv4 slot/MoE amortization  >  EP ownership
       ≫  graph↔FA3 shim surgery  ≫  micro-fuse
```

DSpark = done substrate (optional free-GPU re-measure only).

| Lever | Default today | Measured | Do |
|---|---|---|---|
| FA3 split decode | **OFF** (`--qwen35-fa3-decode false`) | c=1 ITL **−59.9%** (22.8→9.14 ms @4k) | multi-c re-license → **default ON** |
| Qwen whole-step decode graph | **OFF** | only **+5.5%** tok/s | do not pay for device-`seqlen_k` yet |
| DSv4 decode graph | not default | B=1 WASH/−5% | do not chase |
| Qwen batched decode / MoE / gpu-router / FA3 prefill | **ON** | licensed | leave |
| DSv4 thruput | all-on | c16 ~**21%** parallel eff; amortize **≳43 concurrent** | **slots**, not kernel % |

FA3 shim is **non-varlen host `seqlen_k`** on purpose (`arle_fa3_shim.cu`).
Graph-safe device lengths = real varlen work for a +5% graph → **defer**.

---

## Doc hygiene (2026-07-21)

Corpus policy: **July-dated + undated core**, plus a fixed keep list of high-value
pre-July anchors (master strategy v2, OPD pivot, Phase 1/2 plans, support-matrix
wins). Everything else non-July under `docs/**` (including `plans/M_*`) is deleted;
live docs scrubbed to **0 broken `.md` links** (dead targets → plain text / remap).

This campaign is **self-contained**: numbers retained inline; only July paths
or code anchors below.

---

## DSpark status (not the open kernel problem)

| Lane | Status | July evidence |
|---|---|---|
| Qwen3.6 DSpark/DFlash | P0+P1 **LICENSED** 2.4–3.8×; train sidecar Phase 1 e2e | `wins/2026-07-11-dspark-p1-license-qwen36-27b.md`, `…-07-20-dspark-train-sidecar-e2e-verified.md` |
| DSv4 correctness | accept/geometry/TP4/sliding-window **PASS** | `wins/2026-07-13-*`, `…-07-14-dspark-dsv4-accept-and-correctness.md` |
| DSv4 c=1 | **+64%** vs no-spec | `wins/2026-07-20-dspark-sliding-window-c1-win-c8-regress.md` |
| DSv4 c=8/16 | batched verify **code landed**; high-c wall **pending free GPUs** | `wins/2026-07-21-dspark-batched-verify-c8-c16.md` |
| MTP high-c serial draft | dispatch fix 07-19; residual = amortization | `errors/2026-07-19-dsv4-mtp-dspark-high-concurrency-regression.md` |

---

## Goals / non-goals

### Goals

1. **Default-on FA3 split decode** after multi-c wall license (flag exists).
2. DSv4 **slot / concurrent headroom** (tokens-per-expert amortization).
3. DeepEP **token-owned** path toward multi-shape default eligibility.
4. Optional free-GPU DSpark c-sweep re-measure.
5. Graph↔FA3 / micro-fuse only with new wall tax.

### Non-goals

- DSpark greenfield re-architecture.
- FA3 varlen / device-`seqlen_k` as first step.
- Hand-write replacements for FlashMLA / FA3 / DeepGEMM / DSA.
- B=1 launch-count / whole-step graph as main thruput lever.
- Megakernel unless path parity still leaves ≥2× on the floor.
- Marlin/TQ/GGUF/unwired tails without callers + wall A/B.
- `fused_add_rms_norm` default without multi-shape win.
- SGLang `fused_moe_triton` as Qwen BF16 default (historical −18..−46%).

---

## Binding constraints

| Fact | Anchor |
|---|---|
| DSv4 all-on c16 ≈ 196 out tok/s | `wins/2026-07-19-dsv4-production-all-on-reanchor.md` |
| DSv4 amortize ≳43 concurrent (0.094 tok/expert · B) | `wins/2026-07-07-dsv4-decode-optimization.md` |
| FA3 decode default OFF; −59.9% ITL @c1 | `crates/cli/src/args.rs` `qwen35_fa3_decode`; number retained |
| FA3 ignored when decode-graph ON | `qwen35.rs:785-803` |
| FA3 host `seqlen_k` | `qwen35.rs:5903`, `arle_fa3_shim.cu` |
| Batched decode / MoE / gpu-router / FA3 prefill default ON | `args.rs` |

---

## Tracks

```
B0 FA3-decode multi-c → default ON   (PRIMARY)
A2 slot / concurrent headroom        (DSv4 thruput)
C  DeepEP token ownership
A1' free-GPU DSpark re-measure       (optional)
B1 device-seqlen shim                (deferred)
B2/B3/B4 graph / GDN / micro         (later)
```

July children still live:

- `2026-07-11-dsv4-high-concurrency-throughput-campaign.md`
- `2026-07-11-dspark-dsv4-flash-spec-decode.md`
- `2026-07-09-dspark-dflash-spec-decode-qwen36.md`
- `2026-07-02-deepspec-adoption-map.md`
- `wins/2026-07-07-dsv4-decode-optimization.md`

---

## Phase plan

### B0 — FA3 split decode multi-c → default ON (P0)

**Exit:** same-binary A/B Qwen3.6 HD256, c∈{1,4,8}, `qwen35_fa3_decode` 0/1,
**graph OFF**, batched decode ON. Needle pass. Win → flip default true + CHANGELOG.

| Step | Action |
|---|---|
| B0.1 | c1 regression check (4k-ish, splits=8) |
| B0.2 | c=4,8 thruput + ITL |
| B0.3 | multi-c holds → `args.rs` default true |
| B0.4 | fails → errors/ + case-decode; keep opt-in; **no B1** |
| B0.5 | document graph ON still disables FA3 decode |

Not in B0: varlen, device `seqlen_k`, default-on graph.

### B1 — Graph-safe device lengths (P3 deferred)

Only if whole-step graph re-licenses ≥~15% **and** product needs graph+FA3.
Needs FA3 `seqused_k`/varlen — real shim work.

### B2 — Decode graph default (P2)

Prior +5.5% bar miss; default stays OFF until new multi-c license ≥~15%.

### B3 — GDN structure then kernel (P1 later)

1. Slot-indexed GDR/conv pool  
2. FlashQLA chunked prefill (`--qwen35-gdr-chunked`)  
3. Adopt FLA/vendor only with wall A/B — no default flip on WASH history

### B4 — Optional micro (P2)

| Item | Gate |
|---|---|
| Count-aware silu → DeepGEMM re-race | nsys shows silu pad tax; hand MoE still default |
| Prefill QKV/gate_up pack | multi-len TTFT still binds after FA3 prefill |
| `fused_add_rms_norm` wire | multi-shape ITL only |

Skip resurrecting `fused_mlp` / `split_qkv` without A/B.

### A2 — Slot / concurrent headroom (P0 DSv4)

```text
EP4, 64 local experts/rank, top_k=6
tokens/expert ≈ 0.094 · B
amortize ~4 tok/expert → B ≳ 43
c16 parallel efficiency ~21%
```

Lever = more concurrent rows (slots + `max-running-requests` + KV budget).

| Step | Action |
|---|---|
| A2.1 | Reconcile slot vs max-running vs KV (`wins/2026-07-17-max-running-requests-caps-slot-budget.md` if present) |
| A2.2 | Measure tokens/expert + active rows @ c=8/16/32 |
| A2.3 | Lift headroom only with bit-exact VRAM ledger |
| A2.4 | No MegaMoE default |

Child: `2026-07-11-dsv4-high-concurrency-throughput-campaign.md`.

### A1' — DSpark high-c re-measure (P2 optional)

Substrate done. When 4 free H20s: c∈{1,4,8,16} DSpark vs no-spec; policy
c1-only vs all-c from numbers. Does not block B0/A2/C.

### C — DeepEP token ownership (P1 multi-GPU)

| Rule |
|---|
| Each EP rank owns distinct token rows; no replicated-token DeepEP default |
| Prefill must not host-poll as long-pole |
| Default flip only after multi-shape c-sweep; B=1 budget documented |
| Do not advertise SGLang-parity on replicated-token route |

Code: `infer-cuda/src/deepep.rs` (`deepep_ll`), `moe.rs`.

### A3 — DeepSpec confidence verify length (P2)

`2026-07-02-deepspec-adoption-map.md`. No draft megakernel.

---

## Fusion inventory

### Shipped — do not reimplement

| Fusion | Location |
|---|---|
| Q/K RMSNorm + RoPE + paged KV write | `decode_prep_paged*.cu` |
| GQA fused decode | `fused_attention.cu` |
| Expert gate+up + SwiGLU decode | `moe_grouped_gemm.cu`, `dsv4_fp8_decode_moe.cu` |
| DSv4 WQKV-A DeepGEMM | `attention.rs` `run_fused_wqkv_*` |
| DSA Q indexer rope+Hadamard+quant | `dsv4_dsa_official.cu` |
| DeepGEMM swiglu+requant / paged MQA | `dsv4_deepgemm_ops.cu` |
| MHC pre + rms | `dsv4_mhc.cu` → `hc.rs` |

### Adjacent decisions

| Candidate | Decision |
|---|---|
| residual + RMSNorm | B4 only |
| host `seqlen_k` → device | B1 deferred |
| FA3 decode default OFF | **B0 primary** |
| silu pad → count-aware | B4 |
| AR + hc_post / draft+verify megakernel | KILL / SKIP |

---

## Hard KILLs (inline; historical entries purged)

Prefill CUDA graph default · whole-step graph as B=1 wall win · multi-stream AR
overlap · per-projection DeepGEMM / residual-GEMV-as-#2 smoke · FlashMLA prefill
as #1 prefill lever · fused AR+hc / hc_enter · mhc TileLang f32 MMA · pad-free
grouped-GEMV MoE · `fused_moe_triton` Qwen BF16 default · MegaMoE default ·
cuBLAS autotune as gap closer · OPD fused SwiGLU as infer lever · FP8 KV
pair-quantize fuse · TileLang BN tile sweep without binding proof.

---

## Measurement

`docs/bench-and-trace-spec.md` + `scripts/bench_throughput.py`.

Matched A/B · c∈{1,4,8,16} · wall-clock only · needle on attn/draft ·
case-as-fact on regress · wins/errors same day · default flip ≥2 shapes +
CHANGELOG. No formula → no run.

---

## Two-week sequence

```
Week 1
  Day 1-2  B0 FA3-decode multi-c A/B (graph OFF) + needle
  Day 3    B0 default flip OR errors/
  Day 4-5  A2 slot / max-running / KV reconcile + counters

Week 2
  Day 1-2  A2 headroom lift or document ceiling
  Day 3    C design notes
  Day 4    optional A1' if 4 free H20s
  Day 5    consolidate; keep B1/B2/B4 closed unless numbers open them
```

---

## File touch map

| Phase | Paths |
|---|---|
| B0 | `crates/cli/src/args.rs` default; bench; CHANGELOG |
| A2 | serve flags, `moe.rs` counters, KV/slot admission |
| A1' | serve/bench; code only if attributed bug |
| B1 deferred | `qwen35.rs`, `arle_fa3_shim.cu` |
| C | `deepep.rs`, `moe.rs` |

No CUDA types in `infer-core` / `infer-api`.

---

## Success criteria

1. FA3 split decode **default-on** after multi-c license, or multi-c kill with cases.
2. A2 documents tokens/expert vs concurrent cap; headroom lifted or VRAM ceiling written.
3. DSpark stays licensed substrate; A1' optional.
4. No B1 shim without graph re-license ≥ bar.
5. No new hand kernel without adopt-first A/B; KILL posture stable.

---

## Related (July + core only)

- `docs/reviews/kernel-registry.md`
- `docs/bench-and-trace-spec.md`
- `docs/plans/2026-07-11-dsv4-high-concurrency-throughput-campaign.md`
- `docs/plans/2026-07-11-dspark-dsv4-flash-spec-decode.md`
- `docs/plans/2026-07-09-dspark-dflash-spec-decode-qwen36.md`
- `docs/plans/2026-07-02-deepspec-adoption-map.md`
- `docs/experience/wins/2026-07-07-dsv4-decode-optimization.md`
- `docs/experience/wins/2026-07-19-dsv4-production-all-on-reanchor.md`
- `docs/experience/wins/2026-07-20-dspark-sliding-window-c1-win-c8-regress.md`
- `docs/experience/wins/2026-07-21-dspark-batched-verify-c8-c16.md`
