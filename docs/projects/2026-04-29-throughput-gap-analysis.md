# Throughput gap analysis — why we're 28% behind SGLang at c=16

> Snapshot of all known issues from the 2026-04-29 perf push.
> Cross-references the pipeline map
> (`docs/projects/2026-04-29-scheduler-pipeline-map.md`) for the
> "where" and the bug roundup
> (`docs/projects/2026-04-29-perf-bug-roundup.md`) for the "what".
> This doc tells you the **why**.

## Where we are right now

Best-config tok/s @ c=16/4096-in/256-out/120s on L4 / Qwen3-4B:

| Config | tok/s | TTFT p50 | ITL p50 | Note |
|---|---:|---:|---:|---|
| **FP8 + slots=16 + chunk=512** | **145.30** | 11884 | 71.58 | current best |
| FP8 + slots=auto(31) + chunk=512 | 138.99 | 12103 | 71.53 | extra slots waste contig scratch |
| FP8 + slots=16 + chunk=2048 (canonical default) | 105.22 | 10455 | 86.12 | K2 stall dominates |
| FP8 + slots=auto(31) + chunk=2048 | 118.90 | 12555 | 71.72 | +13% over canonical, K2 still gating |
| BF16 + slots=auto(15) + chunk=2048 | (sweep crashed mid-run) | — | — | mixed plan kicks in but bf16 OOMs at high rate |
| BF16 + slots=16 + frac=0.80 + max_prefill=4096 | 65.43 | 11971 | 130.26 | mixed ON, but 2.22× slower than FP8 |
| **SGLang reference** (historical) | **~201** | ~3357 | ~67 | what we're chasing |

We sit at **72.3% of SGLang's tok/s** and **3.5× SGLang's TTFT**.

## Five problems, ranked by how much tok/s they cost us today

### #1 — K2: FP8/INT8 KV can't enter Mixed plan (~50 tok/s)

**The single biggest gap.** `infer/src/model/qwen3/forward.rs:585`'s
`supports_mixed_batch` rejects everything except `KVFormat::BF16`.
Trace from the canonical FP8 c=16 run:

- Plan distribution: 55 × `Split`, 12 × `Decode` (logged), **0 ×
  `Mixed`**.
- Per-step median in Split: `prefill_us = 500 ms`,
  `decode_us = 4 ms` — decode rows wait 125× longer than they take
  to run.
- Wall: 34% of 120s is pure-prefill blocks where 16 decode rows
  produce zero tokens.
- Effective: 5.4 tok/s/req vs SGLang's 12.6 tok/s/req — 2.33× gap
  attributable here (1.91× from the wall-share, ×1.22 from the
  serialized launch overhead).

Why blocked: no FP8/INT8 varlen attention kernel existed. We have
`decode_attention_fp8_partial_kernel`
(`crates/cuda-kernels/csrc/attention/decode_attention_quantized.cu:309`)
that's hard-coded `qlen=1` — can't accept a packed mixed batch with
prefill rows.

Status: **kernel infra landed in `4e4906f5`** (with NHD layout fix
via codex P1). Wire-up pending in
`infer/src/model/qwen3/batch_decode.rs::decode_batch_with_prefill`.
Once wired + verified, K2 gate lifts and FP8 takes the Mixed path.

Expected after wire-up: tok/s 145 → 180-210, TTFT 11.9s → 3-4s.

### #2 — K9: INT8 dequant overhead (~17% ITL)

INT8 ITL p50 is 17% higher than FP8 at same shape (98ms vs 75ms in
n=3 medians). The INT8 attention reads per-page scale tables and
multiplies them in during the dequant. FP8 is self-describing
(E4M3) so no scale lookups.

Status: filed as K9 in the perf-bug-roundup. Real fix is to
template the new varlen kernel for INT8 once the FP8 path is
verified — pretty much free additional kernel variant.

### #3 — Codex P2 from `c4109b29`: BF16 lost merged-QKV/gate_up fast path

When we deduplicated the merged `qkv_proj` and `gate_up_proj`
weights to free 4.7 GB of VRAM (`c4109b29`), bf16 batched-decode
lost the fused-GEMM-then-split fast path. Now does 3 separate q/k/v
GEMM launches per layer per decode step, plus 2 separate gate/up
launches. At c=16/4096-in the larger pool drowns the launch
overhead — net win because admission throttling drops more than
launch cost rises. **At c=1 (decode-bound)** the launch cost is
likely visible — needs a separate bench. Filed as K1.

### #4 — chunked_prefill_size=2048 is wrong default for FP8/INT8

Default is `2048` (matching SGLang). For BF16-with-Mixed-plan it's
correct: each prefill step packs ≤8 chunks, decode rides alongside.
For FP8/INT8 (no Mixed today) each Split-prefill block is
`chunked_prefill_size × kernel_us` of decode-row stall.
`chunk_size=512` shrinks each block 4× → ~125 ms instead of ~500
ms. This recovered +38% tok/s alone (105 → 145).

The right fix is K2 (after which `chunk=2048` is fine again). The
interim fix is to auto-pick `chunk=512` when KV format is
quantized AND mixed-batch is unavailable. Filed as a follow-up;
not urgent because it's a band-aid on K2.

### #5 — Auto-num-slots overshoots for fixed-c demand (~5% tok/s)

`auto_num_slots` picks 31 for FP8 (kv_budget / per_slot = 12 GB /
377 MB ≈ 31). At c=16 demand the extra 15 slots' contiguous KV
scratch is wasted. Manual `--num-slots 16` gives slightly better
tok/s (145.30 vs 138.99 with chunk=512). Magnitude: ~5%.

Mostly cosmetic. The auto value is correct for max-throughput
benches (where concurrency ≫ 16) and for the wins-entry comparison
to SGLang. Not worth fixing yet.

## What "fixing K2 properly" looks like

The K2 fix is multi-step:

1. **Kernel** — DONE (`4e4906f5`): `decode_attention_varlen_fp8.cu`
   handles varlen Q + optional causal mask + inline FP8 E4M3 →
   float dequant. NHD layout matches the existing FP8 writers
   (codex P1 fix at the same SHA).
2. **Wire-up** — pending: in
   `infer/src/model/qwen3/batch_decode.rs::decode_batch_with_prefill`,
   when `paged_kv_pool.format == FP8E4M3`, call the new kernel
   instead of `flashinfer_tc_run_layer`. Drop the `format !=
   KVFormat::BF16` early return at line 481.
3. **Lift the gate** — pending: in
   `infer/src/model/qwen3/forward.rs:585`, change `matches!(...,
   KVFormat::BF16)` to `matches!(..., KVFormat::BF16 |
   FP8E4M3 | INT8)`.
4. **Test e2e** — pending: 4-token prompt must produce coherent
   text (per §10.1 of `docs/bench-and-trace-spec.md`).
5. **Bench** — pending: c=16 fixed 120s. Expected jump to
   180-210 tok/s.
6. **INT8 variant** — pending: template the same kernel for INT8
   (add per-page K/V scale loads).

## The deeper structural issue

The K2 gate exists because we wrote 5 separate attention paths
across `(KVFormat × phase)`:

| Phase | BF16 | FP8 | INT8 |
|---|---|---|---|
| Prefill (paged) | TileLang HD128 / FlashInfer | same (writes bf16, commits to fp8) | same (commits to int8) |
| Single-token decode | Triton AOT | `decode_attention_fp8` | `decode_attention_int8` |
| Batched decode | `flashinfer_tc_run_layer` | `decode_attention_fp8` | `decode_attention_int8` |
| **Mixed (decode+prefill varlen)** | `flashinfer_tc_run_layer` | **MISSING** | **MISSING** |

Three of the four cells for FP8/INT8 already use a quantization-
aware kernel. Only the Mixed cell was missing. After the K2
wire-up there's a complete 4-cell coverage matrix.

After the wire-up, the next architectural lever is consolidating
the 5 distinct kernel paths into a single `BatchAttention(varlen,
KVFormat)` dispatch. That's a cosmetic refactor for downstream
maintenance, not a perf lever.

## Decision tree for the next perf milestone

If the goal is closing the SGLang gap:

1. Ship K2 wire-up → expected 145 → ~190 tok/s, 95% parity.
2. Profile remaining gap with `ncu` on a single layer step.
   Likely candidates: split-KV not used in our kernel (single-CTA
   per row), tile sizes not tuned, ITL p99 tail at admission
   bursts.
3. Land bench tracing patches (#28) so the next iteration has
   per-step-phase metrics in headline_table.md without log
   parsing.

If the goal is generalizing beyond Qwen3-4B:

1. Bench Qwen3.5-4B (different model architecture: hybrid
   full-attention + linear-attention layers) — currently running.
2. Validate that the FP8 varlen kernel + K2 wire-up still works
   on hybrid models.
3. Repeat for Qwen3-8B (longer hidden dim, same attention shape).

## Cross-references

- Pipeline map (where the data flows):
  `docs/projects/2026-04-29-scheduler-pipeline-map.md`
- Bug roundup (the unsorted list of K1-K10):
  `docs/projects/2026-04-29-perf-bug-roundup.md`
- BF16-shadow dead end (why we needed a real kernel, not a
  buffer reuse):
  `docs/experience/errors/2026-04-29-bf16-shadow-mixed-architectural-dead-end.md`
- KV-quant matrix (the saturation tok/s table):
  `docs/experience/wins/2026-04-29-bench-guidellm-cuda-l4-kv-quant-matrix.md`
- c=16 fixed canonical:
  `docs/experience/wins/2026-04-29-bench-guidellm-c16fixed-fp8.md`
- Bench protocol §10:
  `docs/bench-and-trace-spec.md` §10.1-10.5
- Open work: tasks #28 (tracing patches) + #29 (varlen kernel
  wire-up).
