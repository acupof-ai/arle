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

| c | prefix hits | accept | TTFT p50/p99 | ITL mean/p99 | decode tok/s | req/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 112/128 | 40.64% | 978.4 / 10308.6 ms | 10.89 / 49.53 ms | 91.8 | 0.219 |
| 2 | 128/128 | 27.35% | 581.3 / 930.8 ms | 17.06 / 94.93 ms | 58.6 | 0.465 |
| 4 | 128/128 | 27.06% | 584.8 / 1395.3 ms | 29.46 / 505.06 ms | 33.9 | 0.581 |
| 8 | 128/128 | 27.63% | 625.1 / 3408.2 ms | 48.86 / 544.12 ms | 20.5 | 0.715 |
| 16 | 128/128 | 27.28% | 878.6 / 7646.0 ms | 89.34 / 723.44 ms | 11.2 | 0.786 |

`decode tok/s` = `1000 / ITL mean`, per request. It is the column to compare a
decode-path change against; prefill is compared on TTFT. Comparing a
short-prompt decode measurement against a 33K-prompt sweep point is a mistake
that has been made
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
latest-is-reference, not because a delta was demonstrated. Acceptance moves
under 1 pp at every point, so the two runtimes are indistinguishable on this
fingerprint. The GPU differs from the prior row
(`GPU-77551814`), which rule 3 counts as part of the fingerprint.

**Re-run 2026-08-17, runtime `5ea12daaa`, same GPU (`GPU-1769a5e7`):**
c=1 matches, c≥2 regresses. Acceptance is within 2 pp
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

| c | TTFT warm | TPOT | ITL p50 | decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.70 s | 6.70 ms | 6.69 ms | 149.3 |
| 8 | 0.60 s | 36.12 ms | 29.34 ms | 27.7 |
| 16 | 0.75 s | 66.41 ms | 47.71 ms | 15.1 |

vs the prior row (`a956f69b1`, 2026-07-28, pre-decode-graph): TPOT −58.7% /
−18.1% / −9.9% at c=1/8/16. The c=1 effect is
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
  warm = turns 1–7.

**The whole-step decode graph is DEFAULT-ON and working** (2026-08-03, `cb6b3389d`):
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

### Reference point for the NVFP4 kernel work (2026-08-19, refreshed 2026-08-23)

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

| | decode, clean (`5185ce517`) | decode, clean (`110c6f7f6`) |
|---|---:|---:|
| Qwen3.6-27B-FP8 | 57.6 tok/s | **60.4 tok/s** |
| Qwen3.8-27B-NVFP4 | 52.3 tok/s | **84.5 tok/s** |

NVFP4 is 40% ahead of FP8 on the clean measurement at `110c6f7f6` (was 9%
behind at `5185ce517`). The gain is from decode-graph and Marlin optimizations
that landed between the two commits; the Marlin bps tiebreaker alone is +2-6%
at the kernel level ([wins entry](experience/wins/2026-08-23-marlin-nvfp4-decode-bps-tiebreaker.md)).

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

## Qwen3.8-27B-NVFP4 · 1×H20 · single-GPU · eager — NVFP4 ANCHOR

### SOTA — runtime `97d28ba2c` (2026-08-22)

Mixed-precision: NVFP4 MLP (group_size=16, E2M1 + E4M3 group scales) on 56 of 64
layers + FP8 per-channel attention (F8_E4M3 + BF16 `[N,1]` weight_scale)
everywhere else, 1 BF16 MTP layer. Same `qwen3_5` hybrid architecture as
Qwen3.5/3.6 (48 gated-delta linear-attn + 16 full-attn). **Only 54% of params are
4-bit** — 7.49 GB U8 against 11.56 GB F8_E4M3 — so the checkpoint is 23.42 GB
against the FP8 model's 30.87 GB, 24% fewer bytes rather than half.

Identity: `unsloth/Qwen3.8-27B-NVFP4` · 1×H20 (sm_90, 96 GB), TP=1 ·
`--kv-cache-dtype fp8 --max-running-requests 16` · no spec · decode graph on ·
`ARLE_CUDA_PROFILE` off.

**Slot count matters and is easy to get wrong.** The executor pre-allocates one
~146 MB recurrent block per slot eagerly, and `hot_workspace_slots()` is
`max_running_requests.unwrap_or(num_slots)` with `num_slots` defaulting to 256.
Omitting `--max-running-requests` therefore reserves 37,584 MB for a workload
that admits 16 — these rows pass it, matching the FP8 anchor above.

#### Long-agent 32K — the anchor row

`bench-agent-32k-16x8.jsonl` (sha 8867f63e, 1,052,018 prompt / 6,848 output
tokens per point), 32 req/point, `--max-tokens 214 --temperature 0 --seed 42`.
Both arms on the same binary. 32/32 complete and `SERVER_ERRORS=0` at every cell.

Same-base control: `Qwen3.8-27B-FP8` (the Qwen3.6 control of the 2026-08-20
row compared two different models). Both arms on GPU 0 back to back, no spec,
decode graph on, p50 of 32 requests. Runtime `97d28ba2c` — the tensor-core
quantized paged attention ([entry](experience/wins/2026-08-22-paged-attention-quantized-tensor-core.md)).

| c | NVFP4 ITL ms | FP8 ITL ms | ITL | NVFP4 TTFT s | FP8 TTFT s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | **13.70** | 17.67 | **+29.0%** | 5.93 | 5.56 |
| 4 | **18.91** | 24.37 | **+28.9%** | 1.21 | 1.26 |
| 8 | **24.40** | 29.82 | **+22.2%** | 1.46 | 1.50 |
| 16 | **37.80** | 39.79 | **+5.3%** | 1.52 | 1.56 |

Against the 2026-08-20 row (`ec5edf987`, NVFP4 ITL 20.46 / 39.40 / 69.81 /
130.11 ms) ITL is −33 / −52 / −65 / −71 %; TTFT is unchanged (prefill still
dequantises the quantized prefix into a bf16 temp for FA3). The ITL lead over
FP8 still decays with concurrency — 29.0 → 28.9 → 22.2 → 5.3 — the
`dense_ffn` residue [the occupancy entry](experience/errors/2026-08-19-marlin-decode-is-not-occupancy-limited.md)
measured. TTFT at c≥4 is below c=1 because the 16×8 prompt set shares
prefixes and the radix cache serves them after the first sweep.

Previous row (`ec5edf987`, 2026-08-20, control Qwen3.6-27B-FP8 on GPU 1):

| c | NVFP4 ITL ms | FP8 ITL ms | ITL |
| ---: | ---: | ---: | ---: |
| 1 | 20.46 | 24.81 | +21.3% |
| 4 | 39.40 | 47.57 | +20.7% |
| 8 | 69.81 | 79.00 | +13.2% |
| 16 | 130.11 | 137.23 | +5.5% |

ITL is unchanged across the two prefill-kernel changes below (20.45 → 20.46
at c=1) because the materialisers only run at M ≥ 512 and no decode batch
reaches that; their whole gain lands in prefill (TTFT) on this 154:1
prefill-to-decode workload.

**A c=16 row measured 2026-08-20 on a contended box is discarded, not superseded:
the FP8 control arm — unchanged code, unchanged config, identical resident bytes
and KV pool — moved 3.5× between runs.** A control that moves
invalidates the run it is in; the sign of the NVFP4 delta flipped with it.

#### Long context — 256 K total

`--max-prompt-tokens 262144 --max-total-tokens 262144 --max-running-requests 2`;
the engine caps prompts at `max_total − max_total/8` = 229,376 and keeps the
rest for decode. Runtime `1df0acf68`, GPU 0, one request.

| prompt tok | TTFT s | decode tok/s | generated | needle |
|---:|---:|---:|---:|---|
| 220,054 | 130.4 | 47.9 | 5,015 (EOS) | found at 50 % depth |
| ≈221 K (`needle_gate.py 200000`) | — | — | — | exact 1/1, DET |

TTFT grows 22× from 32 K (5.9 s) to 220 K for 6.7× the tokens: the O(L²)
full-attention term dominates prefill at this length, where it was 22 % at
32 K. Decode at 225 K context is 47.9 tok/s against 73.0 at 32 K.

#### 8-token decode grid

`--seconds-per-concurrency 30 --max-tokens 128`, synthetic prompt (mean 8
tokens). Measured at `1da4e0422`; the prefill work since does not reach these
shapes (the DeepGEMM arms sit above an M floor no decode batch reaches).
`decode tok/s (agg)` = `c × 1000 / ITL mean`.

| c | NVFP4 ITL ms | NVFP4 decode tok/s (agg) | FP8 decode tok/s (agg) | vs FP8 |
|---:|---:|---:|---:|---:|
| 1 | 11.85 | **84.4** | 61.2 | **+37.9%** |
| 2 | 14.17 | **141.1** | 99.8 | **+41.4%** |
| 4 | 14.65 | **273.1** | 195.7 | **+39.5%** |
| 8 | 16.32 | **490.2** | 358.7 | **+36.7%** |
| 16 | 22.29 | **717.9** | 631.2 | +13.7% |

c=16 is a cliff here, not a decay. A per-op profile (`ARLE_CUDA_PROFILE=1`, both
checkpoints, c=1 vs c=16) puts the whole residue in `dense_ffn`: 9.201 ms against
the FP8 checkpoint's 11.635 for the same 17.11 G weight values, while on the five
ops where both read identical bytes (in_proj / out_proj / qkv / o_proj / lm_head)
Marlin is 22-30% faster. Marlin's NVFP4 arm is within 12% of its own per-channel
FP8 arm **per value** (probe, gate_up M=16: 0.093 vs 0.083 ms) — the byte
advantage does not convert because the kernel is issue-bound, not
bandwidth-bound. Three tuning attempts against that are recorded and rejected in
[errors/2026-08-19-marlin-decode-is-not-occupancy-limited.md](experience/errors/2026-08-19-marlin-decode-is-not-occupancy-limited.md).

#### VRAM

`KV budget: free …MB` from the serve log; `resident = 97,871 − free`. Both arms
same flags, same binary, same moment.

| | file | resident | KV pool | full recomputes |
|---|---:|---:|---:|---:|
| Qwen3.8-27B-NVFP4 | 23.42 GB | **22.36 GB** | **1,779,114 tok** | 0 |
| Qwen3.6-27B-FP8 | 30.87 GB | 29.36 GB | 1,582,506 tok | 0 |

The 4-bit model is 7.0 GB smaller resident and holds 12% more KV. It has not
always been: the states this passed through, and why a repack that keeps its
source stores the model twice, are in
[wins/2026-08-20-nvfp4-widen-to-e4m3-deepgemm-prefill.md](experience/wins/2026-08-20-nvfp4-widen-to-e4m3-deepgemm-prefill.md).

#### Engagement — a VRAM number without these is not evidence

Resident falling to 22.36 GB is what you also see if the prefill arms silently
stop firing. Both must be read from the same process:

```
cuda.fp4.widen_fp8_deepgemm          224   = 2 prefill chunks x 56 NVFP4 MLP layers x 2 GEMMs
cuda.qwen.fp8_per_channel_deepgemm   288
cuda.fp4.marlin_tensorcore           336   decode stays on Marlin
cuda.qwen.fp8_marlin_tensorcore      437
cuda.qwen.fp8_gemv                   ABSENT
```

#### Correctness

Needle ladder, `RAW=1 TEMPLATE=qwen3_nonthink`, 3 runs each: 512 / 4096 / 16384 /
32768 all `exact=3 miss=0 DET`. The prefill path is **not** bit-parity with
Marlin — the E2M1 x E4M3 product needs 4 mantissa bits where E4M3 stores 3, and
`dsv4_deepgemm_fp8_gemm_nt` has no BF16-activation entry so activations are
quantised to E4M3 per 128-K block. Sized as RMS output error at K=5120 the fold
costs 1.94-2.38%, against 2.65% for the activation rounding the FP8 baseline
already carries.

#### Prefill kernel reference

`crates/infer-cuda/examples/marlin_fp4_probe.rs`, gate_up [34816, 5120], M=2048:

| path | ms | TFLOPS | share of that format's H20 peak |
|---|---:|---:|---:|
| Marlin, NVFP4 (widen to BF16) | 8.678 | 84 | 57% of BF16's 148 |
| Marlin, per-channel FP8 | 8.457 | 86 | 58% |
| DeepGEMM, FP8 | 2.664 | **274** | 93% of FP8's 296 |

sm_90 has no FP4 tensor core, so a real GEMM must widen the nibbles first and the
only question is what to widen *to*. Widening to E4M3 instead of BF16 doubles the
ceiling.

Per-op, `ARLE_QWEN35_PROFILE`, three 14K-token prefills, ms/call:

| op | first shipped | tables out | vectorised |
|---|---:|---:|---:|
| `qwen/deepgemm/dense_gemm` | 1.4465 | 1.4426 | 1.4427 |
| `qwen/fp4/dense_widen_fp8` | 0.7559 | **0.1871** | 0.1871 |
| `qwen/fp8/dense_channel_scale` | 0.1169 | 0.1153 | **0.0313** |
| `qwen/deepgemm/dense_pack_quantize` | 0.0424 | 0.0409 | 0.0410 |
| `qwen/fp8/dense_materialize` | 0.0508 | 0.0505 | 0.0505 |

Non-GEMM overhead 838 ms → 303 ms, 24.5% → 10.5% of the profiled total. The
widen was **52% of the GEMM it feeds**, not the 3.4% a roofline predicted —
0.756 ms against a 0.093 ms bound, because it was issue-bound on two
divergently-indexed tables, not bandwidth-bound. The path is now GEMM-dominated
at 89.5% and `dense_gemm` is DeepGEMM at 93% of this card's FP8 peak.

**The earlier "A8 vs AFP8 throughput is identical, so W4A8 needs a new kernel for
no gain" note is withdrawn.** It compared INT8 against FP8 activations and missed
the real gap, which was the BF16 *weight* widening Marlin does — worth 3.15x at
prefill and fixed without a new GEMM kernel.

Superseded rows (9.3 tok/s initial support `33f4863c7`, FP4 GEMV vectorization
`2a3a2164f`) live in
[wins/2026-08-18-qwen38-27b-nvfp4-inference.md](experience/wins/2026-08-18-qwen38-27b-nvfp4-inference.md);
the kernel ladder is in
[wins/2026-08-19-nvfp4-marlin-tensorcore.md](experience/wins/2026-08-19-nvfp4-marlin-tensorcore.md).

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

| c | complete | TTFT p50/p99 | accept | acc/chain | chains/s |
| --- | ---: | --- | ---: | ---: | ---: |
| 1 | 62 | 169.8 / 266.7 ms | 50.36% | 2.00 | 16.5 |
| 8 | 177 | 443.5 / 627.1 ms | 49.02% | 1.84 | 0.5 |
| 16 | 241 | 870.8 / 971.1 ms | 44.52% | 1.70 | 0.6 |

**Speculation only engages at c=1.** At c=8/16 the run completes ~60 chains in
~125 s against >22000 output tokens, so DSpark contributes under 1% there and
those acceptance figures are a few-dozen-chain sample, not an indicator. The
c=8/16 points are plain decode.

Rule 3 re-anchor, not a regression against `868043f5f` (2026-08-10). That row
predates `ef8bcd61e`, which added `ignore_eos=true`: its points average
120.7 / 110.1 / 113.5 completion tokens per request, so its requests stopped at
EOS, while every request here emits exactly 128. Forcing generation past EOS is
where the drafter agrees least, which moves acceptance 58.7% → 50.4% at an
unchanged chain rate (16.5 chains/s).

Correctness: needle ladder 512/4096/16384 ×3 passed 9/9 exact (NONDET at
4096/16384 is MoE routing). The c=16 point flags 7/241 responses, all from the
one prompt `Explain bloom filters and their use cases.`, whose continuation
style is concurrency-dependent — see
[`errors/2026-08-14-raw-completion-continuation-flips-with-concurrency.md`](experience/errors/2026-08-14-raw-completion-continuation-flips-with-concurrency.md).
The `logit_bias` relay gate passes at TP=8: a biased request returns 200 with
the biased token dominating, two ordinary requests then answer correctly, and
the serve log carries zero `relay deserialize` lines.

**Re-run 2026-08-17, runtime `5cc681759`** (E8M0 loading fix + event pool +
comm-stream fix): net neutral against the SOTA row at c=1/8/16. Needle 512/4096/16384 ×3 = 9/9 exact.
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

| c | complete | TTFT p50/p99 ms | ITL p50/p99 ms |
| --- | ---: | --- | --- |
| 1 | 10 | 1085 / 1113 | 21.9 / 41.0 |
| 4 | 20 | 1447 / 2985 | 43.8 / 89.2 |
| 8 | 40 | 1069 / 1204 | 47.5 / 93.2 |
| 16 | 48 | 2238 / 2265 | 71.4 / 119.0 |

0 errors / 0 incomplete / 0 correctness_failed at every point. c32 needs
`--max-running-requests 32`; without it host-admission oversubscription degrades
to preemption, not a crash (#164/#162 closed).

**Spec decode is c=1-only on this fingerprint and not a default-flip candidate**
— DSpark positive at c=1, negative at c=4/8/16; MTP negative everywhere. The
crossover is the compute-bound transition: verify is free only while the GPU has
idle compute.

---

## DSv4-Flash · 4×H20 · TP=4/EP=4 · c=1 decode graph (default)

### Default flip — runtime `1a48d179f` (2026-08-23)

The c=1 decode body is captured into one CUDA graph per slot and replayed.
Armed by default; `ARLE_DSV4_DECODE_GRAPH=0` selects the eager arm. The gate is
c=1-only and disarms under DSpark/MTP, so c>=2 and spec-decode are untouched.

Identity:

- Runtime commit `1c56ca0dd`, build `c1-graph-v26` (headline A/B); the
  c=8/16 and FP8 rows below are from `1a48d179f` / `c1-graph-v24b`
- Models `/data00/DeepSeek-V4-Flash-0731` (NVFP4 experts) and `-FP8`
- GPU: 4×H20 (sm_90), TP=4, 4 slots/rank, BF16 KV, `--comm-backend nccl`
- Workload `bench-agent-32k-16x8.jsonl`, prompt p50 28568 tok, max_tokens 256
  exact (ignore_eos), temperature 0
- Capture audit on v26: 0 alloc / 0 free / 0 host memcpy / 0 host callback
  nodes, positive-control verified

NVFP4 experts, 16 requests per point, same binary both arms:

| c | arm | decode tok/s | ITL p50/p99 ms | TTFT p50 ms |
| ---: | --- | ---: | --- | ---: |
| 1 | eager (v26) | 40.8 | 24.1 / 47.7 | 7905 |
| 1 | **graph (v26)** | **44.2** | **22.2 / 44.1** | 7865 |
| 8 | eager | 22.4 | 40.5 / 99.4 | 11409 |
| 8 | graph | 22.4 | 40.8 / 99.0 | 11507 |
| 16 | eager | 17.8 | 51.4 / 104.0 | 22270 |
| 16 | graph | 17.7 | 51.9 / 103.9 | 22530 |

FP8 experts, 8 requests, c=1:

| arm | decode tok/s | ITL p50/p99 ms |
| --- | ---: | --- |
| eager | 52.4 | 18.5 / 41.9 |
| **graph** | **59.5** | **16.3 / 42.0** |

0 errors at every point. c=8/16 are the no-op control that confirms the gate.
The c=1 rows are a same-binary v26 A/B (+8.3% decode); an earlier v24b reading
of +21.3% is superseded because the eager baseline itself improved 14.6%
through concurrent Marlin and Markov-head work.

Correctness: MMLU 5-shot, 200 samples, greedy — 171/200 in both arms, 0
per-item diffs. DSpark control (`--spec-type dspark`): ITL p50 65.8 (off) vs
66.4 ms (on) with 0 graph captures in either arm.

[Wins entry](experience/wins/2026-08-23-dsv4-c1-decode-graph.md)

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

| c | complete | TTFT p50/p99 ms | ITL p50/p99 ms |
| --- | ---: | --- | --- |
| 1 | 11 | 251 / 304 | 40.4 / 41.6 |
| 4 | 12 | 17799 / 25769 | 0.02\* / 270 |
| 8 | 17 | 30818 / 54318 | 0.02\* / 335 |
| 16 | 16 | 72270 / 72933 | 0.02\* / 452 |

\* ITL p50 ≈ 0.02 ms is a streaming-sampling artifact at c≥4; read ITL p99.
TTFT grows linearly with concurrency (queueing).

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
four ranks bit-identical loss at every passing rung. Re-measured on `ad1192864`
after the CP ring byte-offset fix
([wins](experience/wins/2026-08-19-cp-ring-fa3-byte-offset-fix.md)); the broken
path's walls held to within 1% and its peaks to 2%, its loss column did not.
[Entry](experience/wins/2026-08-19-cp4-seq-ceiling-229376-and-17x-step.md).

| seq | RUN_EXIT | forward | fused CE | backward | writeback | peak/rank | loss |
|---|---|---:|---:|---:|---:|---:|---|
| 131072 | 0 | 58.4 s | 1.70 s | 119.48 s | 179.7 s | 72,751 MiB | 3.034899 |
| 196608 | 0 | 90.2 s | 2.43 s | 193.79 s | 286.5 s | 85,871 MiB | 2.072005 |
| **229376** | 0 | 109.9 s | 2.80 s | 232.74 s | **345.6 s** | 94,351 MiB (96.8%) | 1.780185 |

**The cp=4 ceiling is 229376**, unchanged by the fix — 245760 still dies in
backward. 131072 here is 17.3× the 2026-08-03 row below (3100 s → 179.7 s); that
row and the cp=2 one under-report the current substrate by more than an order of
magnitude. Loss falls monotonically with sequence length, as it should; the
pre-fix column (7.63 / 6.92 / 6.74) was inflated by corrupted hidden states.
163840 was measured pre-fix only (230.7 s, 78,959 MiB).

## SOTA — 27B, cp=2 seq ceiling · `62b4927b8` (2026-08-20)

2×H20, `--synthetic-writeback-seq N`, LoRA r16 α32 attention-qv. cp=2 means
local seq = N/2 and the ceiling is per-rank.
[Entry](experience/wins/2026-08-20-cp2-ceiling-114688-to-131072.md).

| global seq | local | outcome | `ckpt-peak actual` |
|---|---:|---|---:|
| 114,688 | 57,344 | pass | — |
| **131,072** | 65,536 | **pass**, 331 s | 75,875 MiB |
| 163,840 | 81,920 | forward + CE pass, **backward OOM** on `zeros [1,81920,5120]` 1.6 GB | 77,026 MiB |

**The cp=2 ceiling is 131,072**, up from 114,688. Matched at local 81,920 the
pool high-water is 84,789 → 77,026 MiB and the peak model's drift +17,420 →
+9,657.

## Known walls

| shape | outcome |
|---|---|
| 27B cp=1 seq=81920 | forward completes (3972.216 s), **backward OOMs** on `cuda alloc_zeros failed`. Host RSS 104.5 GB. The failing tensor is not named by the log. |
| 27B cp=4 seq=245760 | forward + CE complete, **backward OOMs** on `cuda alloc_zeros failed (la dqkv)` (2026-08-19, `9c2c84675`). Still fails on `ad1192864` with the same deadlock signature; that re-run was killed before the error string flushed, so the allocation is not re-confirmed post-fix |
| 27B cp=4 seq=262144 | forward 126.5 s + CE 3.14 s complete, **backward OOMs** on `cuda alloc_zeros failed (slice_bwd)` — the linear-attn zigzag reorder's `slice` backward allocates a full-input zero buffer (2026-08-19, `9c2c84675`) |
| any cp, rank-local error | the erroring rank unwinds into `ncclCommDestroy` and blocks behind the peers' in-flight collective, so **the error text never prints**. Presents as N−1 GPUs at 100% util and one at 0%, indefinitely. Kill the spinners to release the unwind and read the real line. |
| 27B cp=2 seq=131072 | fits — backward peak 94,175 MiB (96.6%), ~3.3 GB headroom (2026-08-02, older commit) |
| 27B cp=4 seq=131072 | full step ~3100 s, host RSS 170.4 GiB total / ~44.6 GB per rank (2026-08-03, scalar ring, older commit) |
| 27B cp=8 | unmeasured — only 4 GPUs were free on 2026-08-19 |
| 0.8B CP correctness arm | **cannot run** — `flashqla GDN head geometry H=8/Hg=8 not built`. FlashQLA went default-on 2026-08-05 with no kernel for that model's per-rank CP geometry, so the cp≥2 arms error instead of producing a number. This is why the FA3 byte-offset bug survived two days ([errors](experience/errors/2026-08-19-cp-training-gradients-regressed-and-the-gate-is-dead.md)) |
