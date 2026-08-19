# Rolling performance baselines

> Status: Active — **one SOTA row per model, plus its config**. A superseded
> row is deleted, not archived here; verdicts, rejected arms, prior champions
> and analysis live in the linked `experience/` entries.

Screening compares a new run against the SOTA row — no second arm.

1. **Effect > ~10%** (2× the measured drift band): verdict valid, replace the
   SOTA row, archive the binary.
2. **Inside the ±3% drift band**: never kill on ambiguity. Escalate to a
   same-shell A/B against the archived binary (≥3 trials/arm, median + range).
3. **Fingerprint change re-anchors**: model, TP/EP, GPU set, serve flags, slot
   line, dataset, driver/CUDA. Re-measure before comparing.
   - `--kv-cache-dtype fp8` is not just a KV format. It turns OFF the whole-step
     decode graph (`qwen35.rs:2106`, `paged_kv_bf16()` gate — the persistent page
     table is BF16-only) and the batched DSpark draft (`qwen35.rs:2343`). On the
     35B the decode graph alone is worth 2.4x at c=1. A BF16-KV row and an
     FP8-KV row are different fingerprints even with every other flag equal.
4. **Anchor audit** every ~5 accepted updates and before any default flip: one
   A/B against the oldest archived binary bounds accumulated drift.
5. **One workload**: the multi-turn long-agent dataset at the TraceLab medians,
   cold and warm turns reported separately.

**Stated deviation: rows run 32K, not the spec's 119K median.** Dense KV is
64 KB/token, so 119K×c16 needs 122 GB against ~69 GB free after weights.
A 119K row is a new anchor, not a re-measure.

```
python3 scripts/gen_bench_prompts.py bench-agent-32k-16x8.jsonl 16 32000 214 8
```

---

## Qwen3.6-27B-FP8 · 1×H20 · single-GPU · eager — LONG-AGENT ANCHOR

### SOTA — DSpark, runtime `fad8f4d5b`, runner `c98c4e0b2` (2026-08-14)

The Qwen3Next target final norm uses `(1+w)`; the Qwen3 DFlash draft uses
plain-weight RMSNorm. The fixed-output runner sends `ignore_eos=true`, and its
production-length warmup has a disjoint prefix-cache key.

Features on: batched draft · replay · snapshot · capture · markov+confidence
head driving the goodput budget. Serve adds `--spec-type dspark
--mtp-draft-model /host/nvme0/Qwen3.6-27B-DFlash --dspark-block-size 6`; 16
slots, 195 MiB per slot.

Identity:

- Runtime commit `fad8f4d5b715698fcada7d2ce382682f18788e03`
- Runner commit `c98c4e0b2`
- Binary SHA-256 `7ba56981695cbdd759b5d6b96e74a0b9b851549c3c55469c0b62d8701c94e9de`
- Kernel bundle `79d522d1bc4f2d4fd6d706c8d7a5ea2040d44b4aeaeac5fcc96472d3040bdd72`
- GPU `GPU-1769a5e7-852b-74f9-e109-f52dbb2c4859` (H20)
- Dataset SHA-256 `8867f63eaac2f0537bb2b17847a7d0d3c1bb8d504c1ad191e97d673e9ecc4f34`
- Per-stage timing breakdown: [perf-stages-27b.md](perf-stages-27b.md)

One fresh serve, ascending concurrency. ITL mean is the per-output-token latency
for speculative decode; event-level ITL percentiles include burst emission and
must not be converted into per-token throughput.

| c | prefix hits | accept | TTFT p50/p99 | ITL mean/p99 | decode tok/s | output tok/s | total tok/s | req/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 112/128 | 40.64% | 978.4 / 10308.6 ms | 10.89 / 49.53 ms | 91.8 | 46.94 | 7675.73 | 0.219 |
| 2 | 128/128 | 27.35% | 581.3 / 930.8 ms | 17.06 / 94.93 ms | 58.6 | 99.48 | 16268.54 | 0.465 |
| 4 | 128/128 | 27.06% | 584.8 / 1395.3 ms | 29.46 / 505.06 ms | 33.9 | 124.42 | 20347.14 | 0.581 |
| 8 | 128/128 | 27.63% | 625.1 / 3408.2 ms | 48.86 / 544.12 ms | 20.5 | 152.94 | 25011.24 | 0.715 |
| 16 | 128/128 | 27.28% | 878.6 / 7646.0 ms | 89.34 / 723.44 ms | 11.2 | 168.30 | 27522.30 | 0.786 |

`decode tok/s` = `1000 / ITL mean`, per request. It is the column to compare a
decode-path change against; `output tok/s` is end-to-end over the sweep point and
on a 33K prompt is prefill-dominated — at c=1 the two differ by 2.0x. The 35B row
below carries the same column; this row lacked it, and comparing a short-prompt
decode measurement against its `output tok/s` is a mistake that has been made
(see [errors/2026-08-19-fp8-dequant-arm-shadows-decode.md](experience/errors/2026-08-19-fp8-dequant-arm-shadows-decode.md)).

Every point completed 128/128 with zero incomplete, error, empty, or
correctness-failed responses. Completion tokens are exactly 214.

The c=1 point has the expected 112 warm hits: 16 cold turn-0 requests followed
by seven reusable turns per session. Later points are fully warm because the
grid is ascending, so this row is the canonical workload baseline and not a pure
concurrency-scaling experiment.

Correctness: needle ladder 512/4096/16384/32768 ×3 passed 12/12 exact,
deterministic at every length.

This row replaces `9b38ba6c0` (2026-08-10) under rule 1 taken as
latest-is-reference, not because a delta was demonstrated. Output tok/s moves
−1.1 / +0.6 / +1.1 / +0.9 / +3.5% across the grid and acceptance moves under
1 pp at every point, so the two runtimes are indistinguishable on this
fingerprint; c=16 alone sits just outside the ±3% band on a single sweep and is
not a claim of improvement. The GPU differs from the prior row
(`GPU-77551814`), which rule 3 counts as part of the fingerprint.

**Re-run 2026-08-17, runtime `5ea12daaa`, same GPU (`GPU-1769a5e7`):**
47.7 / 95.4 / 118.2 / 139.3 / 154.0 out tok/s at c=1/2/4/8/16 — c=1 matches
(+1.6%), c≥2 regresses −4.1 / −5.0 / −9.0 / −8.5%. Acceptance is within 2 pp
of the SOTA row at every point, so the regression is in chain rate, not
speculation quality. Root cause not yet isolated; 78 commits landed between
the SOTA row and this re-run (2D KV sharding, decode-path collapse, event
pool, observability). The SOTA row stands. Needle 12/12 PASS.
### Step budget — where the time goes (2026-08-01, `nsys`, dense FP8)

The SOTA table says how fast; this says what to fix.

> **STALE — do not rank prefill work off this table.** Measured before chunked
> GDR went default-on (`c2eb5de9e`, 08-02). Its #1 row is a kernel that no
> longer runs at the shipped defaults: the FlashQLA chunked path replaced
> `gated_delta_rule_prefill_recurrent` and measured **1.06 s against its 9.37 s**
> ([`wins/2026-08-02-flashqla-chunked-gdr-h48.md`](experience/wins/2026-08-02-flashqla-chunked-gdr-h48.md)),
> which moves 33% of the prefill budget and reorders everything below it. Two
> more prefill changes landed after (`0ac780495`, `301d0c074`). A re-measure
> needs one `nsys` capture; the two attempts that failed and the delay to use
> are recorded in
> [`errors/2026-08-06-decode-lever-board-rebuilt-gemm-is-not-the-top-lever.md`](experience/errors/2026-08-06-decode-lever-board-rebuilt-gemm-is-not-the-top-lever.md).
> The verify decomposition below is also superseded: measured directly on the
> current binary as **5.69 ms/row verify + 3.04 ms/row draft**, and the per-row
> term is not the GDN lane.

**Prefill, 33K in 28.6 s** (single request, 24.0 s GPU-busy, ~37K launches,
2328 `cuMemcpyDtoH` costing 1.58 s):

| kernel | launches | s | share |
|---|---:|---:|---:|
| `gated_delta_rule_prefill_recurrent` (no longer on the default path) | 1152 | 9.37 | 33% |
| DeepGEMM FP8, all shapes | 7936 | 8.33 | 29% |
| TileLang full attention | 368 | 3.93 | 14% |
| `pack_quantize` bf16→fp8 | 9600 | 1.50 | 5% |
| conv1d / norm / silu | 3840 | 0.55 | 2% |
| GPU idle (includes host tokenization) | — | ≤4.6 | ≤16% |

Efficiency of each part against its own ceiling: DeepGEMM `gate_up` 199 TFLOPS
and `down` 189 TFLOPS = **64–67% of the FP8 peak, healthy**; full attention
54 TFLOPS = **36% of the BF16 peak**; the linear-attention recurrence is a
latency chain, **5.9 µs per token per layer** (~6 dependent `__syncthreads`
each). No free parallel axis is left — the block is already 512 threads
(`val_dim 128 × j_slice 4`) and the token axis is the recurrence. Its
`<<<48, ...>>>` grid starves a **78-SM** GPU only at c=1; varlen launches
`grid(num_value_heads, batch)`. Shortening the chain (chunked matmul form) is
the prefill lever.

Verify decomposes as **22 ms intercept + 2.48 ms/row** (5.18 ms/row at 33K), and
the intercept equals one plain non-spec step: verifying 8 speculative tokens
costs what decoding 1 costs. Spec decode is working; the intercept is the wall.

[decode + graph-flag profile](experience/errors/2026-08-01-decode-graph-flag-is-a-noop-under-paged-kv.md) ·
[prefill profile](experience/wins/2026-08-01-prefill-and-decode-step-budget.md) ·
[FP8 small-M attribution](experience/wins/2026-07-10-qwen-fp8-smallm-deepgemm-crossover.md)

---

## Qwen3.6-35B-A3B-FP8 (MoE) · 1×H20 · single-GPU · eager

### SOTA — decode graph, runtime `02867728d` (2026-08-17)

No spec. Features on: whole-step decode graph (default-on, `cb6b3389d`) ·
batched FA3 (one launch per layer) · host-authoritative KV mirror · GDR
chunked (default-on, `c2eb5de9e`).

Identity:

- Runtime commit `02867728d`
- Binary SHA-256 `9567bbccaacdbac585dabb55de10b0931575c17fb83c1a205b31df9c92093de7`
- Kernel bundle `ee06c0c3aea4429ac51d5c32d784e9948fd6c0e85842bf0a87d66fb6186c3c15`
- Dataset SHA-256 `8867f63eaac2f0537bb2b17847a7d0d3c1bb8d504c1ad191e97d673e9ecc4f34`

Same dataset, params, and seed as the 27B row (128 req/point, max_tokens 214,
greedy, seed 20260416). Throughput bench — TTFT cold is not separately
measured; TTFT warm is the p50 across all 128 requests.

| c | TTFT warm | TPOT | ITL p50 | decode tok/s | output tok/s | total tok/s |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.70 s | 6.70 ms | 6.69 ms | 149.3 | 72.5 | 11860.2 |
| 8 | 0.60 s | 36.12 ms | 29.34 ms | 27.7 | 199.3 | 32589.4 |
| 16 | 0.75 s | 66.41 ms | 47.71 ms | 15.1 | 220.6 | 36075.0 |

vs the prior row (`a956f69b1`, 2026-07-28, pre-decode-graph): TPOT −58.7% /
−18.1% / −9.9% at c=1/8/16; total tok/s +77% / +17% / +7%. The c=1 effect is
the decode graph; c≥8 gains are smaller because the GPU is already saturated.
1–3/128 responses per point tripped the repetition checker (greedy, long
context) — no errors, no incomplete.

Correctness: needle ladder 512/4096/16384/32768 ×3 passed 12/12 exact,
deterministic at every length.

---

## Environment — both 1×H20 FP8 rows

- **Box** 1×H20 (sm_90, 78 SM, 96 GB), TP=1, eager, `--max-running-requests 16`.
- **Models** `bottlecapai/ThinkingCap-Qwen3.6-27B-FP8` (dense, 64 layers, 16
  full-attn, kv_heads 4, head_dim 256, KV 64 KB/token) · `Qwen3.6-35B-A3B-FP8`
  (MoE, 40 layers, 10 full-attn, kv_heads 2, 256 experts, top_k 8).
- **Dataset** `bench-agent-32k-16x8.jsonl`, sha256 `8867f63eaac2f053…`,
  `prompt_tokens` p50 34828.
- **Runner** `bench_throughput.py`, 128 req/point, max_tokens 214, greedy,
  seed 20260416. **Gate** `needle_gate.py 512,4096,16384,32768 3 0.0`; a change
  that only fires at concurrency needs `needle_concurrent.py` as well, since the
  ladder is single-request and would run the untouched path.
- **Metrics** TTFT and decode are separate SLOs, never averaged. Decode =
  token-weighted mean ITL (`Σ itl_s / count`); never `e2e − ttft` (this harness
  carries ~4.7 s teardown, inflating TPOT ~1.85×). Cold = session turn 0,
  warm = turns 1–7. `total tok/s` = prompt+generated over wall: capacity, not
  latency.

**`--qwen35-decode-graph` is DEFAULT-ON and working** (2026-08-03, `cb6b3389d`):
the paged-KV early return that made it a no-op was removed, and the graph now
captures the serving default. 35B c=1 TPOT 16.22 → 6.70 ms (−58.7%); see the
35B SOTA row below. The earlier no-op diagnosis is at
[errors/2026-08-01-decode-graph-flag-is-a-noop-under-paged-kv.md](experience/errors/2026-08-01-decode-graph-flag-is-a-noop-under-paged-kv.md).

**`--qwen35-gdr-chunked` is DEFAULT-ON** (2026-08-02, `c2eb5de9e`): 33K cold
prefill −26%; license = chat GSM8K 100 **95/100 both arms, zero
disagreements** + chat MMLU 80 vs 81 + needle 9/9 ×2 + stub-probe fallback.
Named trade: raw-completion few-shot can flip knife-edge boundary tokens
(the 11/100-vs-46/100 collapse was that, not a kernel bug —
[error](experience/errors/2026-08-02-gdr-chunked-gsm-collapse-was-a-knife-edge-harness.md));
chat/agentic serving is parity. **TTFT-cold champion rows predate this flip
and need a re-anchor sweep.**

---

## Qwen3.6-27B-W8A16 (Marlin) · 1×H20 · single-GPU

Model `iso-tc-huihui-w8a16` (Huihui-Qwen3.6-27B abliterated, W8A16 gs=128,
29 GB), GPU 6, `bench-agent-32k-64.jsonl`, c=1, 16 requests × 256 tokens,
temperature 0, seed 20260416. TTFT is cold — 16 distinct prompts, no prefix
hits. SGLang 0.5.13 row serves the GPTQ v1 twin repacked by
a one-off uint8→int8 GPTQ repack (script since deleted) — identical int8 values, identical kernel.

### SOTA — snapshot stride 8192 (2026-08-06)

Two reps per arm; reps agree to 0.10 s TTFT / 0.02 ms ITL. P/D reported
separately: `prefill tok/s = prompt_tokens / TTFT` (33000 prompt tokens),
`decode tok/s = 1 / ITL`.

| arm | TTFT p50 | prefill tok/s | ITL p50 | decode tok/s | ITL p99 | e2e p50 |
|---|---:|---:|---:|---:|---:|---:|
| ARLE | 23.01 s | 1434 | **16.70** | **59.9** | 20.50 | 27.4 s |
| SGLang, same kernel + same weights | **21.03 s** | **1568** | 17.16 | 58.3 | **19.19** | **25.44 s** |

Decode leads by 2.8%, TTFT is 1.09× behind (was 1.48× then 1.19×), p99 7% behind.
Gate: needle 512/4k/16k/32k ×3, all `exact=3 miss=0 DET`.

Prefill idle split, cold 33K, `--cuda-graph-trace=node`:

| | ARLE | SGLang |
|---|---:|---:|
| GPU busy | 21.9 s | within 0.93 s of ARLE |
| in-span idle, stride 2048 | 1.675 s | 0.19 s |
| D2H, pinned staging (was pageable) | 2.771 GB / **0.062 s** (0.577 s) | ~0 |

Periodic snapshot cost by stride, 33K prefill: 18 snapshots 3.13 s · 4
snapshots 0.85 s · 0 snapshots 0 s. Each retains ~150 MB until publish, so the
cost is the count. Remaining gap against SGLang: 1.98 s.

`--chunked-prefill-size` is not a lever on either stack: 2048 vs 4096 (ARLE)
and 4096 vs 8192 (SGLang) all land inside 0.07 s TTFT.

The row is from a build with no `ARLE_CUDA_*` set — FA3 and FlashQLA now build
from vendored-tree + sm_90 detection alone. Confirmed against the env-set
build: TTFT 24.94/24.95 s vs 24.97/25.05, zero fallback lines in the serve log.

[FlashQLA stub build + prefill ledger](experience/wins/2026-08-05-flashqla-was-never-compiled-into-the-pod-binary.md) ·
[decode budget, both stacks](experience/wins/2026-08-04-w8a16-decode-step-kernel-budget.md) ·
[FA3 splits](experience/wins/2026-08-04-fa3-decode-splits-fill-the-sms.md) ·
[conv1d fusion](experience/wins/2026-08-04-conv1d-decode-fusion.md) ·
[repack method](experience/wins/2026-08-02-w8a16-sglang-matched-ab.md)

---

## Qwen3.6-27B-FP8 vs Qwen3.8-27B-NVFP4 · 1×H20 · c=1 no-spec decode

### Reference point for the NVFP4 kernel work (2026-08-19)

The 27B FP8 SOTA row above runs DSpark and a long-agent workload, so it is not
comparable to a short-prompt no-spec decode. This row establishes that
comparison directly: same box, nothing else resident, same bench invocation,
the only variable is the checkpoint's quantization.

```bash
python3 scripts/bench_throughput.py --concurrency-grid 1   --seconds-per-concurrency 30 --max-tokens 128 --temperature 0 --seed 42
```

Both served with `--kv-cache-dtype fp8`, no spec decode.

**`ARLE_CUDA_PROFILE=1` costs 66-73% of decode throughput** — it brackets every
op with a `cudaEventRecord` pair, 192 per step at 64 layers. Every per-op table
below is therefore measured under profiling and is useful only for attribution
between ops, never as a throughput figure. The headline tok/s rows are measured
with profiling OFF:

| | decode, profiled | decode, clean |
|---|---:|---:|
| Qwen3.6-27B-FP8 | 33.2 tok/s | **57.6 tok/s** |
| Qwen3.8-27B-NVFP4 (`5185ce517`) | 31.5 tok/s | **52.3 tok/s** |

NVFP4 is 9% behind FP8 on the clean measurement. The ratio is close to the
profiled one (0.91 vs 0.95) because the overhead is per-op and both models run
the same op count, so profiling did not distort the comparison — only the
absolute numbers.

> **This row is c=1 only, and c=1 is the one point where NVFP4 wins.** The
> matched grid is in the Qwen3.8-27B-NVFP4 section below. Read that before
> quoting any NVFP4-vs-FP8 ratio.

The per-op tables below use `ARLE_CUDA_PROFILE=1`.
Architecturally identical: hidden 5120, intermediate 17408, 64 layers,
vocab 248320.

| | dense_ffn | forward_hidden | decode | dense_ffn GB/s |
|---|---:|---:|---:|---:|
| Qwen3.6-27B-FP8 | 9.84 ms/step | 29.22 ms/step | 33.2 tok/s | 1740 |
| Qwen3.8-27B-NVFP4 (`cb109750e`) | 23.30 ms/step | 42.26 ms/step | 23.5 tok/s | 413 |
| Qwen3.8-27B-NVFP4 (`5185ce517`) | 11.46 ms/step | 31.21 ms/step | 31.5 tok/s | 840 |

NVFP4 moves 150.4 MB of dense-MLP weights per layer against FP8's 267.5 MB —
56% of the bytes — so it should be faster, not 2.4x slower. Both formats run
the same hand-written warp-per-row scalar GEMV at M=1 (FP8 does NOT use
DeepGEMM there: `qwen_fp8_dense_projection.rs` has an empty policy table and
its fallback routes m<2 to Gemv), so the gap is entirely inside two
structurally identical inner loops.

Progress on the NVFP4 side, all at this fingerprint: the kernel started at
86.19 ms/step dense_ffn and 9.3 tok/s. Replacing the constant-memory decode
table with bit manipulation (`cb109750e`) took it to 23.30 / 23.5, and
replacing that with PRMT byte lookups (`5185ce517`) to 11.46 / 31.5 — 7.5x on
the kernel and 3.39x on decode. NVFP4 is now within 5% of FP8 on decode and
16% behind on dense_ffn alone.

ncu attributed each step: the constant-memory table cost a divergent memory
read per weight, and the bit-manipulation form that replaced it pinned the ALU
pipe at 92.4% (against FP8's 59.4%, whose hardware cvt issues on the FMA pipe).
PRMT moves the work back off the ALU: 73.4% ALU, 40.6% FMA.

The format-independent ops agree closely across the two runs
(linear_attention 9.41 vs 9.50 ms, full_attention 3.27 vs 3.26), which is what
makes the dense_ffn column a clean single-variable comparison.

## Qwen3.8-27B-NVFP4 · 1×H20 · single-GPU · eager

### Current state — runtime `4e6ec4b2a` (2026-08-19). NOT a SOTA row.

Mixed-precision: NVFP4 MLP (W4AFP8, group_size=16) on 56 of 64 layers + FP8
per-channel attention (F8_E4M3 + BF16 `[N,1]` weight_scale) everywhere else.
Same `qwen3_5` hybrid architecture as Qwen3.5/3.6 (48 gated-delta linear-attn +
16 full-attn). 1 BF16 MTP layer. **145 of ~200 quantised GEMMs per forward are
FP8, not NVFP4** — the Marlin work landed on the 56-layer NVFP4 minority.

Identity: model `unsloth/Qwen3.8-27B-NVFP4` (22 GB + 811 MB MTP) · 1×H20
(sm_90, 96 GB), TP=1 · `--kv-cache-dtype fp8` · no spec unless stated ·
`ARLE_CUDA_PROFILE` off · 20.69 GiB peak RSS, 6.5 s load.

**Matched A/B, same binary and same moment**, NVFP4 on GPU0 and
Qwen3.6-27B-FP8 on GPU1, identical flags and harness invocation. Synthetic
prompt (mean 8 tokens), `--seconds-per-concurrency 30 --max-tokens 128`.

| c | NVFP4 ITL ms | NVFP4 decode | NVFP4 agg | (pre-fix agg) | FP8 ITL ms | FP8 agg |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 14.95 | **66.9** | 66.9 | 66.6 | 17.60 | 56.8 |
| 2 | 19.43 | 51.5 | **102.9** | 19.9 | 20.21 | 99.0 |
| 4 | 24.25 | 41.2 | 165.0 | 39.2 | 20.58 | 194.4 |
| 8 | 36.70 | 27.2 | 218.0 | 75.6 | 22.43 | 356.7 |
| 16 | 67.49 | 14.8 | 237.1 | 140.4 | 25.46 | 628.4 |

NVFP4 leads FP8 at c=1 (+17.8%) and c=2 (+3.9%) and trails from c=4. The step
cost against batch is the open item: NVFP4 4.5x over 16x concurrency against
FP8's 1.45x. FP8 is flat because DeepGEMM is a tensor-core GEMM; NVFP4's
per-channel FP8 weights sit on a batched GEMV whose register pressure scales
with the tile (`quantized_gemv.cu`), so Marlin `kFE4M3fn` is the next lever.

Long-agent 32K, `bench-agent-32k-16x8.jsonl`, 32 req/point, max_tokens 214,
no spec, both arms on the same box:

| arm | c | ITL ms | decode | out tok/s | completed |
|---|---:|---:|---:|---:|---:|
| NVFP4 | 1 | 22.88 | **43.7** | 13.59 | **32/32** |
| NVFP4 | 4 | 49.57 | 20.2 | 70.59 | **32/32** |
| FP8 | 1 | 26.68 | **37.5** | 18.93 | 32/32 |
| FP8 | 4 | 47.32 | 21.1 | 75.40 | 32/32 |

NVFP4 leads by 16.5% at c=1 and trails 4.3% at c=4. The gap is far narrower than
on an 8-token prompt because attention dominates at 33K and is format-independent
— the short-prompt grid measures the weight-read path almost alone.

**These rows were taken with `MARLIN_MAX_BLOCKS_PER_SM` pinned to 1, but that is
not established as necessary.** The run completed 64/64 and hit zero partial
prefix restores — the condition present at both crashes — so it does not show
that pinning fixes anything. Pinning costs 29-42% of decode throughput. See
[errors/2026-08-19-blocks-per-sm-search-two-latent-bugs.md](experience/errors/2026-08-19-blocks-per-sm-search-two-latent-bugs.md).

Spec decode, synthetic prompt, c=1:

| arm | ITL ms | decode | vs no-spec |
|---|---:|---:|---:|
| no spec | 15.02 | 66.6 | — |
| MTP d=2 (35.1% accept) | 71.23 | 14.0 | −79% |
| DSpark block 6 | 58.90 | 17.0 | −74% |

**Reading.** The NVFP4 decode kernel is genuinely 12–17% faster than FP8's at
c=1, at both 8-token and 33K context. Everything else is worse, and all of it
is one defect: `try_fp8_dequant_bf16_gemm_batch` fired at `M >= 2` and
re-dequantised all 11.56 G FP8 params per call — 84.35 ms per forward. It cost
5× aggregate throughput at c≥2, inverted both spec-decode paths, and crashed
the server at 34K on its un-budgeted 2.54 GB scratch. Root cause and fix:
[errors/2026-08-19-fp8-dequant-arm-shadows-decode.md](experience/errors/2026-08-19-fp8-dequant-arm-shadows-decode.md).

The concurrency grid is post-fix; the 32K and spec-decode tables above are
still the pre-fix (defective) build and are re-measuring. Every cell is n=1. No
NVFP4 row is eligible for SOTA until the 32K run completes.

Superseded rows (9.3 tok/s initial support `33f4863c7`, FP4 GEMV vectorization
`2a3a2164f`) live in
[wins/2026-08-18-qwen38-27b-nvfp4-inference.md](experience/wins/2026-08-18-qwen38-27b-nvfp4-inference.md);
the kernel ladder 52.3 → 57.9 → 60.2 → 63.9 (out tok/s, c=1) is in
[wins/2026-08-19-nvfp4-marlin-tensorcore.md](experience/wins/2026-08-19-nvfp4-marlin-tensorcore.md).

A8 vs AFP8: on H20 (sm_90) FP8 E4M3 and INT8 tensor-core throughput are
identical (989 TFLOPS/TOPS) and FP8 has the wider dynamic range, so W4A8 (INT8
activations) would need a new GEMM kernel for no throughput gain. The shipped
W4AFP8 path is the right choice.

---

## DSv4-Flash-FP8 · 8×H20 · TP=8/EP=8 · eager

### SOTA — DSpark, runtime `fad8f4d5b`, runner `c98c4e0b2` (2026-08-14)

Serve `--spec-type dspark --mtp-draft-model
/host/nvme0/DeepSeek-V4-Flash-DSpark-draft-fp8 --comm-backend nccl`, no other
flags. DSpark runtime: stages=3 block=5 target_layers=[40,41,42],
`confidence_threshold: None`, sps bias 211.0 ms / row 0.53 ms. Base model +
draft on NVMe (HDD load timed out the engine-ready barrier at 924 s); weights
land at 41737 MB/rank, prefetch 17.46 GB/s.

Identity: binary SHA-256
`08fa6f89c1de04b01d87ab9a19198db9377c6407816d69925eb8985110a36878`, kernel
bundle `cb52441a46a39bfe4e65d4dc2c02d21fb5d799d9a3be324ae5f35635bd7e7286`.

Synthetic prompts (64) over raw `/v1/completions`, **120 s/point**, max_tokens
128, greedy, seed 42. Not the 32K agent fingerprint — re-anchor on the agent
workload before ranking.

| c | complete | out tok/s | total tok/s | TTFT p50/p99 | accept | acc/chain | chains/s |
|---|---:|---:|---:|---|---:|---:|---:|
| 1 | 62 | 65.56 | 69.82 | 169.8 / 266.7 ms | 50.36% | 2.00 | 16.5 |
| 8 | 177 | 182.10 | 193.87 | 443.5 / 627.1 ms | 49.02% | 1.84 | 0.5 |
| 16 | 241 | 244.77 | 260.60 | 870.8 / 971.1 ms | 44.52% | 1.70 | 0.6 |

**Speculation only engages at c=1.** At c=8/16 the run completes ~60 chains in
~125 s against >22000 output tokens, so DSpark contributes under 1% there and
those acceptance figures are a few-dozen-chain sample, not an indicator. The
c=8/16 throughput is plain decode.

Rule 3 re-anchor, not a regression against `868043f5f` (2026-08-10). That row
predates `ef8bcd61e`, which added `ignore_eos=true`: its points average
120.7 / 110.1 / 113.5 completion tokens per request, so its requests stopped at
EOS, while every request here emits exactly 128. Forcing generation past EOS is
where the drafter agrees least, which moves acceptance 58.7% → 50.4%; the
throughput follows from acceptance at an unchanged chain rate — restoring
2.42 acc/chain at the measured 16.5 chains/s gives 65.6 + 0.42 × 16.5 =
72.5 tok/s against the old row's 72.4.

Correctness: needle ladder 512/4096/16384 ×3 passed 9/9 exact (NONDET at
4096/16384 is MoE routing). The c=16 point flags 7/241 responses, all from the
one prompt `Explain bloom filters and their use cases.`, whose continuation
style is concurrency-dependent — see
[`errors/2026-08-14-raw-completion-continuation-flips-with-concurrency.md`](experience/errors/2026-08-14-raw-completion-continuation-flips-with-concurrency.md).
The `logit_bias` relay gate passes at TP=8: a biased request returns 200 with
the biased token dominating, two ordinary requests then answer correctly, and
the serve log carries zero `relay deserialize` lines.

**Re-run 2026-08-17, runtime `5cc681759`** (E8M0 loading fix + event pool +
comm-stream fix): 64.68 / 178.31 / 241.09 out tok/s at c=1/8/16 — within
±2% of the SOTA row, net neutral. Needle 512/4096/16384 ×3 = 9/9 exact.
The E8M0 fix (`5cc681759`) unblocks DSv4 FP8 loading: the W4A16 detection
probe called `quant_view_for()` which rejected DSv4's native E8M0 scales;
the DSv4 path now skips that rejection. See
[`errors/2026-08-17-dsv4-e8m0-scale-rejection-blocks-fp8-loading.md`](experience/errors/2026-08-17-dsv4-e8m0-scale-rejection-blocks-fp8-loading.md).

---

## DSv4-Flash-FP8 · 4×H20 · TP=4/EP=4 · eager

### SOTA — Base, `d0525cb06` (re-anchored 2026-07-25, #180)

> Short-prompt fingerprint, retired 2026-07-26 under rule 5 — the dataset is no
> longer reproducible from the repo. Evidence for what it licensed, not a
> comparison target.

Dataset `bench-prompts-20.jsonl`, sha256 `e095ddf1fcc9325a…`, 60 s/point,
max_tokens 256, seed 20260416. Slot line `59 slots / per_slot 338MB / 84736 tok`.

| c | complete | out tok/s | total tok/s | TTFT p50/p99 ms | ITL p50/p99 ms |
|---|---:|---:|---:|---|---|
| 1 | 10 | 38.66 | 456 | 1085 / 1113 | 21.9 / 41.0 |
| 4 | 20 | 74.67 | 876 | 1447 / 2985 | 43.8 / 89.2 |
| 8 | 40 | 152.82 | 1793 | 1069 / 1204 | 47.5 / 93.2 |
| 16 | 48 | 197.51 | 2319 | 2238 / 2265 | 71.4 / 119.0 |

0 errors / 0 incomplete / 0 correctness_failed at every point. c32 needs
`--max-running-requests 32`; without it host-admission oversubscription degrades
to preemption, not a crash (#164/#162 closed).

**Spec decode is c=1-only on this fingerprint and not a default-flip candidate**
— DSpark +5.0% at c=1, −23/−44/−48% at c=4/8/16; MTP negative everywhere. The
crossover is the compute-bound transition: verify is free only while the GPU has
idle compute.

---

## DSv4-Flash-W4AFP8 · 2×H20 · TP=2 · eager

### Initial support — runtime `fb0b877d2` (2026-08-19)

NVFP4 checkpoint (E2M1 float4 + E8M0 block scales) converted to W4AFP8
(signed INT4 + BF16 interleaved scales) at load time on GPU. 4-bit weights
keep the 167 GB model on 2×96 GB H20 (FP8 path needs 4×). SGLang CUTLASS
mixed-input grouped GEMM for routed experts; shared expert stays FP8.

Identity:

- Runtime commit `fb0b877d2` (32MB workspace right-size)
- Model `/data00/DeepSeek-V4-Flash-0731` (166.9 GB NVFP4)
- GPU: 2×H20 (sm_90, 96 GB), TP=2
- Server flags: `--tensor-parallel-size 2 --port 30000`
- Peak VRAM: 95.7 GB/rank (48 slots × 339 MB KV + weights)
- Load time: ~90s (NVFP4→W4AFP8 per-expert GPU conversion)

`bench_dsv4_trace_http.py`, 5 cases, greedy:

| Case | TTFT (s) | Decode tok/s | Status |
|---|---:|---:|---|
| decode64 | 0.10 | 37.0 | 200 |
| prefill1k | 0.48 | — | 200 |
| prefill4k | 1.10 | — | 200 |
| math | 0.08 | 36.0 | 200 (17×23=391, +19=410 ✓) |
| write_zh | 0.13 | 36.7 | 200 (coherent ✓) |

Prefill throughput: 2109 tok/s (1K), 3647 tok/s (4K).

[Wins entry](experience/wins/2026-08-18-nvfp4-w4afp8-tp2-serve.md)

---

## Qwen3.6-27B-W4A16 · 1×V100 (sm_70) · eager

**`aec71ef16` (2026-07-21)** — V100 kernel opts + KV pool floor fix. Synthetic
prompts 64, 60 s/point, max_tokens 256, seed 20260416. KV pool 16384 tok BF16
(1.1 GB), 86 slots. Serve `--max-total-tokens 16384`.

| c | complete | out tok/s | total tok/s | TTFT p50/p99 ms | ITL p50/p99 ms |
|---|---:|---:|---:|---|---|
| 1 | 11 | 22.8 | 24.4 | 251 / 304 | 40.4 / 41.6 |
| 4 | 12 | 25.5 | 27.4 | 17799 / 25769 | 0.02\* / 270 |
| 8 | 17 | 28.4 | 30.4 | 30818 / 54318 | 0.02\* / 335 |
| 16 | 16 | 30.1 | 32.1 | 72270 / 72933 | 0.02\* / 452 |

\* ITL p50 ≈ 0.02 ms is a streaming-sampling artifact at c≥4; read out tok/s.
Decode-bound at every concurrency (+32% from c=1 to c=16); TTFT grows linearly
with concurrency (queueing).

**DSpark on V100 is KILLED (−91% at c=1, errors at c≥8).** ITL 40 → 499 ms;
c=16 produced 131204 errors in 60 s with `[coordinator] lockstep stalled`. The
TP lockstep proposal path deadlocks at world_size=1 — needs a TP=1 fast path
before this arm is retried.

---

# Training baselines — OPD writeback

> Status: Active. Same screening rules as the inference rows above. Training
> numbers were previously scattered across wins entries and task descriptions;
> this section is the single truth. A row not listed here has not been measured.

**Fingerprint** — re-anchor when any of these move: model, LoRA target set,
`cp_size`/`dp_size`, GPU set, sequence length, commit.

- Model `bottlecapai/ThinkingCap-Qwen3.6-27B-FP8` (64 layers = 16 full-attn +
  48 gated-delta, kv_heads 4, head_dim 256), or `qwen35-08b-clean`
  (24 layers, `full_attention_interval: 4`, dense MLP) for the correctness rows.
- LoRA `attention-qv`. Workload = `--synthetic-writeback-seq N` (one masked-CE
  writeback on a synthetic trajectory; no rollout).
- Box 8×H20 (sm_90, 97.9 GB). Two ranks = one CP group unless stated.

**Before trusting any CP row, verify the binary.** `nm -D <bin> | grep
ncclCommInitRank` and `ldd <bin> | grep libnccl`. A shared build target was
silently overwritten by a `cuda`-only build on 2026-08-05 and the resulting run
failed in a way that reads as a code bug. FA3 additionally needs the vendored
hopper tree and an sm_90 target at build time — without them
`ring_fa3_route`'s real-kernel marker returns 0 and the ring falls back to the
scalar kernels.

## SOTA — 27B, cp=2, seq=32768 · `15caff0d0` (2026-08-05)

| | |
|---|---:|
| forward | 34.2 s |
| fused CE | 0.92 s |
| backward | 190.0 s |
| optimizer | 0.05 s |
| **step** | **225.2 s** |
| checkpoint peak | 61,396 MiB/rank |
| loss | 10.871086 |
| grad_norm | 2.263385 |

Both ranks print identical loss and grad_norm (post-all-reduce). Reproduces the
2026-08-04 FA3 reference (10.871086 / 2.264733 / ~212 s) to 6-decimal loss and
0.06% grad-norm; the +6% on step is shared-box variance.

## SOTA — 27B, cp=2, seq=81920 · FlashQLA default-on `fa742a038` (2026-08-05)

FlashQLA GDN chunkwise backward is the default (`--gdr-chunkwise-prefill=true`).
Same harness (`/host/fqgate.sh perf_on`), same seq, only variable is the flag.

| | rank 0 | rank 1 | recurrent (below) | speedup |
|---|---:|---:|---:|---:|
| forward | 64.124 s | 64.125 s | 81.0 s | 1.26× |
| fused CE | 0.83 s | 0.83 s | 1.91 s | — |
| backward | 312.643 s | 312.648 s | 670.275 s | **2.14×** |
| **step** | **378.723 s** | **378.723 s** | 752.956 s | **1.99×** |

Peak host RSS 55.4 GB, loss 4.537510, grad_norm 7.976866, RUN_EXIT=0. The 71%
`linear_attention_chunked_scan_backward_f32` row is gone.

The recurrent column is `--la-backward-mono` on `e675f031b`: device peak
91,547 MiB/rank (93.5% of the card), loss 4.536131, grad_norm 7.202155.

### Step budget — where the time goes

`nsys cuda_gpu_kern_sum`, one step, both ranks combined, FA3 engaged.

| share | time | instances | kernel |
|---|---:|---:|---|
| 71.0% | 707.345 s | 90 | `linear_attention_chunked_scan_backward_f32` |
| 6.7% | 66.316 s | 238,080 | `gated_delta_rule_prefill_recurrent` |
| 3.9% | 38.365 s | 7,436 | nvjet GEMM 128×256 |
| 3.2% | 32.096 s | 4,194 | nvjet GEMM 320×128 TNT |
| 1.9% | 19.134 s | 2,886 | nvjet GEMM 320×128 NNT |
| 1.5% | 15.271 s | 11,664 | `transpose_axes_swap_f32` |
| 1.5% | 14.635 s | 47 | `FlashAttnBwdSm90` |
| 1.4% | 13.553 s | 25,196 | `slice_f32` |

The two gated-delta rows are 77.7% of the step. Both ride the route the
FlashQLA port (`4846f8046`) replaces.

## Correctness rows — 0.8B dense, seq=2048 · `15caff0d0` (2026-08-05)

> **STALE AND UNRUNNABLE.** The cp=2/cp=4 arms error today —
> `flashqla GDN head geometry H=8/Hg=8 not built` — because FlashQLA went
> default-on at this very commit and has no kernel for the 0.8B's per-rank CP
> geometry. cp=1 still reproduces (3.466840 vs 3.464900). With the CP arms
> unable to produce a number, this table stopped gating CP on 2026-08-05; a
> 6.4× gradient regression got through
> ([errors](experience/errors/2026-08-19-cp-training-gradients-regressed-and-the-gate-is-dead.md)).

Mean of 3 serial reps per cell, post-all-reduce (all ranks identical). Within-cell
spread is 5.2e-5 to 2.1e-4 relative across every cell.

| arm | grad_norm | deviation from cp=1 |
|---|---:|---:|
| cp=1 | 3.464900 | — |
| cp=2 | 3.459982 | −1.419e-3 |
| cp=4 | 3.464276 | −1.80e-4 |

FA3 is inert at cp=1 (no ring exists there): toggling it leaves loss identical at
8.963640 and grad_norm inside the spread. The deviation does not compound with
ring-step count — it collapses into the noise floor at cp=4, while the pre-flip
scalar path's grows (+1.085e-3 at cp=2 to +1.655e-3 at cp=4). See
[the gate entry](experience/wins/2026-08-05-fa3-cp-gate-compounding-not-sign.md).

## SOTA — 27B, cp=4 seq ladder · `9c2c84675` (2026-08-19)

4×H20 (97,508 MiB), GPUs 4-7, FA3 engaged, `--synthetic-writeback-seq N`. All
four ranks bit-identical loss at every passing rung.

> **Resource measurement only.** The CP path produces gradients 6.4× off
> single-card as of this commit
> ([errors](experience/errors/2026-08-19-cp-training-gradients-regressed-and-the-gate-is-dead.md)).
> Walls and peaks are real — same tensors, same kernels — but the loss column
> and the ceiling are not measurements on a numerically correct path.
[Entry](experience/wins/2026-08-19-cp4-seq-ceiling-229376-and-17x-step.md).

| seq | RUN_EXIT | forward | fused CE | backward | writeback | peak/rank | loss |
|---|---|---:|---:|---:|---:|---:|---|
| 131072 | 0 | 56.8 s | 1.69 s | 119.05 s | 177.5 s | 74,095 MiB | 7.631271 |
| 163840 | 0 | 73.5 s | 2.01 s | 155.08 s | 230.7 s | 78,959 MiB | 7.189730 |
| 196608 | 0 | 89.8 s | 2.03 s | 192.09 s | 283.6 s | 86,991 MiB | 6.924870 |
| **229376** | 0 | 107.9 s | 2.76 s | 231.38 s | **342.2 s** | 92,655 MiB (95.0%) | 6.742337 |

**The cp=4 ceiling is 229376.** 131072 here is 17.5× the 2026-08-03 row below
(3100 s → 177.5 s); that row and the cp=2 one under-report the current substrate
by more than an order of magnitude.

## Known walls

| shape | outcome |
|---|---|
| 27B cp=1 seq=81920 | forward completes (3972.216 s), **backward OOMs** on `cuda alloc_zeros failed`. Host RSS 104.5 GB. The failing tensor is not named by the log. |
| 27B cp=4 seq=245760 | forward + CE complete, **backward OOMs** on `cuda alloc_zeros failed (la dqkv)` (2026-08-19, `9c2c84675`) |
| 27B cp=4 seq=262144 | forward 126.5 s + CE 3.14 s complete, **backward OOMs** on `cuda alloc_zeros failed (slice_bwd)` — the linear-attn zigzag reorder's `slice` backward allocates a full-input zero buffer (2026-08-19, `9c2c84675`) |
| any cp, rank-local error | the erroring rank unwinds into `ncclCommDestroy` and blocks behind the peers' in-flight collective, so **the error text never prints**. Presents as N−1 GPUs at 100% util and one at 0%, indefinitely. Kill the spinners to release the unwind and read the real line. |
| 27B cp=2 seq=131072 | fits — backward peak 94,175 MiB (96.6%), ~3.3 GB headroom (2026-08-02, older commit) |
| 27B cp=4 seq=131072 | full step ~3100 s, host RSS 170.4 GiB total / ~44.6 GB per rank (2026-08-03, scalar ring, older commit) |
| 27B cp=8 | unmeasured — only 4 GPUs were free on 2026-08-19 |
| any cp>1, gradients | **cp=2 grad_norm 1.401418e1 against cp=1's 2.197122 at 27B seq=32768** — CP agreed with single-card on 2026-08-05 (2.263385) and does not now. Not LoRA rank, not FlashQLA; cause unknown, bisect pending ([errors](experience/errors/2026-08-19-cp-training-gradients-regressed-and-the-gate-is-dead.md)) |
