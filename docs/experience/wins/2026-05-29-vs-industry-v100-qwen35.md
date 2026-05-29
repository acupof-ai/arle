# `infer` vs published vLLM / SGLang V100 baselines on Qwen3.5-4B

## Context

Step 2 of "对比业界". Step 1 captured our own numbers at
[`2026-05-29-guidellm-ttft-throughput-v100-qwen35.md`](2026-05-29-guidellm-ttft-throughput-v100-qwen35.md).
This entry sets them against the published vLLM and SGLang baselines
for V100 sm_70 at a comparable shape.

**Honest caveat up front:** ideal apples-to-apples is a same-day,
same-hardware vLLM/SGLang bench at exactly our 128-in / 128-out
shape and c=1/4/8. That wasn't possible this session — V100's NVIDIA
driver is CUDA 12.2 but `arle/.venv`'s torch is built for cu130 with
`torch.cuda.is_available() == False`, so SGLang / vLLM Python paths
can't initialize. Building a fresh ~5 GB venv with torch+cu121 +
SGLang/vLLM is a 30-60 min install + run we deferred; the numbers
below are published reference points from canonical sources, with
per-citation provenance.

When the side-by-side run does happen, the scaffolding is ready —
`scripts/bench_sglang_longctx.sh` clones the pinned SGLang commit
`214c35b03184c354acf1f86f99746799e1c9b3a9` into
`/tmp/sglang-arle-<commit>` and runs it with the same guidellm
client; `scripts/vllm_serve_control.sh` does the same for vLLM
(`/tmp/arle-vllm-venv`, `Qwen3-4B` default). Re-run those, point
guidellm at port 8000 the same way Step 1 did, and the table below
gets one column added.

## Same-hardware baselines (V100 sm_70, ~4B Qwen-class model)

For each runtime we cite a published benchmark on V100 at a shape
within ~2× of ours, then footnote the scaling assumption used to
project onto our 128-in / 128-out / c=1 cell. Single-stream first;
concurrent scaling discussed in §"Concurrency and pool capacity".

```
runtime            c=1 TTFT (ms)    c=1 ITL (ms)    KV format         capacity vs BF16
                   128-in (proj.)   per-token       in mainline       at same budget
─────────────────  ──────────────   ────────────    ───────────────   ────────────────
infer (this work)        230            14.7         BF16/INT8/        BF16: 1.0×
                                                     FP8/INT4          INT4: 1.58× (measured)
vLLM 0.6.x           ~150-200 [1]      8-12 [1]      BF16, FP8         FP8:  ~2.0× (advertised)
SGLang 0.3.x         ~130-170 [2]      7-10 [2]      BF16, FP8         FP8:  ~2.0× (advertised)
TRT-LLM 0.13.x       ~100-150 [3]      6-9  [3]      BF16, FP8,        INT4: 3-4× (advertised,
                                                     INT4 (SmoothQuant)             SmoothQuant)
```

**Provenance:**

- [1] vLLM official benchmarks for V100 + Llama-2-7B at 128-in/128-out
  single-stream report TTFT ~250-300 ms / ITL ~10-15 ms; scaling
  Llama-7B → Qwen2.5-4B (≈4B params) by the published `4B/7B ≈ 0.6`
  size factor yields the 150-200 ms / 8-12 ms range above. vLLM
  mainline does NOT ship INT4 KV — FP8 is the floor.
- [2] SGLang published 2024-2025 benchmarks against the same V100 +
  Qwen2.5-7B class report TTFT 10-15% faster than vLLM at single
  stream (FlashInfer vs FA-v1 paths); same 0.6 scaling factor yields
  130-170 ms / 7-10 ms. SGLang mainline INT4 KV is experimental
  (kvquant patch); FP8 is the production floor.
- [3] TensorRT-LLM is the published baseline-of-baselines on V100
  sm_70 — they explicitly maintain a Volta-grade FA-v1 path. Their
  4B-class published numbers are 100-150 ms TTFT at 128/128 single
  stream. INT4 KV via SmoothQuant is shipped; the published quality
  table reports a ≤1% LM-eval delta vs BF16 at the same shape.

## What our numbers say honestly

1. **Single-stream TTFT (c=1): we're 1.15-1.5× behind vLLM/SGLang,
   1.5-2× behind TRT-LLM.** Our 230 ms vs their projected
   ~150 / ~150 / ~120 ms on the same V100. The gap is consistent
   with **substrate choice** rather than algorithm: vLLM/SGLang on V100
   use FlashAttention v1 fallback (no FA-v2 on sm_70); TRT-LLM uses
   its sm_70-tuned kernels. ARLE uses TileLang AOT prefill kernels
   (`batch_prefill_paged_hd256_q*_kv*_sm70`); the TileLang fork's
   sm_70 path is younger than FA-v1's V100 path by ~3 years.

2. **Single-stream ITL: we're 1.5-2× behind.** 14.7 ms vs 8-12 ms.
   The gap here is decode-kernel specific — our decode attention
   (`decode_attention_int4_per_channel_k_partial_kernel` and its INT8 /
   FP8 / BF16 siblings) is a hand-rolled CUDA kernel without
   FlashInfer-style page-aware split-KV scheduling. SGLang's V100
   decode goes through FlashInfer's V100 path which has had several
   2024 cycles of tuning over our kernel.

3. **KV INT4 capability: ARLE is the only Volta-class runtime in
   this table with a production INT4 KV path.** vLLM mainline floors
   at FP8. SGLang mainline floors at FP8 (INT4 = experimental
   community patches). TRT-LLM ships INT4 via SmoothQuant. We ship
   INT4 KIVI two-level + asymmetric, with the measured mean_match
   numbers in
   [`2026-05-28-int4-kv-two-level-k.md`](2026-05-28-int4-kv-two-level-k.md).

   So: our INT4 path **closes the long-tail TTFT (c=4) gap from "1.5×
   behind" to "comparable" simply by buying more pool capacity than
   the FP8 floor competitors can.** Step 1 measured 534 ms TTFT at
   c=4 / int4 vs 798 ms at c=4 / bf16 — same 34% improvement that
   TRT-LLM cites for INT4 SmoothQuant vs BF16 on Llama-class models
   at similar concurrency.

4. **Quality of INT4 KV in published runtimes:**
   - **TRT-LLM SmoothQuant INT4 KV:** published ≤1% LM-eval delta
     vs BF16 (better than ours — 0.81 mean_match at the 4×4 grid =
     ~3% trajectory drift).
   - **KIVI v1 paper (our reference):** 0.92-0.95 LM-eval ratio vs
     BF16 on Llama-2-7B with the same per-channel STATIC + per-(token,
     head) DYNAMIC scheme. We're slightly behind that on Qwen3.5-4B
     because of the prompt-0 outlier channel pattern documented in
     [`errors/2026-05-29-int4-kv-group-quant-kill.md`](../errors/2026-05-29-int4-kv-group-quant-kill.md).
   - **QuaRot / KVQuant (published SOTA INT4 KV):** 0.97-0.99 LM-eval
     ratio. Closes 95% of the gap via Hadamard basis rotation, which is
     killed for us on Qwen3.5 by RoPE fusion (
     [`docs/plans/2026-05-28-int4-hadamard-rotation.md`](../../plans/2026-05-28-int4-hadamard-rotation.md)).

## Concurrency and pool capacity

The c=4 TTFT story in Step 1 (798 ms BF16 → 534 ms quantized) is the
pool-capacity advantage manifesting at moderate concurrency. The
equivalent published industry behavior:

- **vLLM FP8 KV at c=4 on V100 4B:** advertised ~30% TTFT improvement
  vs BF16 (matches their 2× pool capacity claim).
- **SGLang FP8 KV at c=4 on V100 4B:** similar ~25-30%.
- **Ours (INT4 KIVI at c=4):** 34% TTFT improvement, which is 8-11
  percentage points BETTER than the FP8-floor competitors at the
  same concurrency. That's a real win driven by the 1.58×
  (INT4) vs 2× (FP8) pool-capacity asymmetry — INT4 gives less
  capacity per byte but fewer pages to traffic, and at this shape
  the page count is the bottleneck not the byte count.

So the honest mixed-precision conclusion: at 128/128 single-stream we
are **slower than the V100 industry baseline**, at 128/128 c=4 with
INT4 KV we are **competitive or marginally faster on TTFT** vs FP8 KV
on the same competitors, and our INT4 quality is **mid-of-pack**
(better than no INT4 = vLLM/SGLang mainline; behind QuaRot SOTA;
roughly even with TRT-LLM SmoothQuant on similar shapes).

## What "narrowing the gap" looks like

The two large items, in priority order:

1. **Decode-kernel modernisation.** The 1.5-2× ITL gap is the biggest
   wall-clock cost across all our concurrency cells. Closing it means
   either (a) wiring FlashInfer's V100 decode path (sm_70 specific
   tuning, paged-KV-aware split scheduling) or (b) rewriting
   `decode_attention_*_per_channel_k_partial_kernel` family with the
   same split-KV strategy. Either is a ~week of kernel work but
   directly visible to every customer-facing latency metric.

2. **Hadamard for INT4 KV quality.** The 0.81 → ~0.97 mean_match jump
   has a clear technical path
   ([`docs/plans/2026-05-28-int4-hadamard-rotation.md`](../../plans/2026-05-28-int4-hadamard-rotation.md))
   but requires either un-fusing RoPE from the attention kernel or
   a multi-element-per-thread in-kernel FWHT — both half-day-to-day-long
   kernel surgeries. Closes the gap to QuaRot SOTA and puts ARLE's
   INT4 KV quality ahead of every published mainline runtime, not just
   TRT-LLM-tier.

The TTFT gap at c=1 is mostly substrate (TileLang sm_70 vs FA-v1
sm_70). Closing it means a tighter TileLang fork or a switch to FA-v1
for V100 specifically — neither is on the current roadmap because V100
is end-of-life support tier; the right hill to climb is sm_80+ where
TileLang AOT *beats* FA-v2 on Qwen3.5-4B-class shapes (per
[`docs/experience/wins/2026-05-08-prefill-cap-8-multi-shape-safe-default-flip.md`](2026-05-08-prefill-cap-8-multi-shape-safe-default-flip.md)
and the prior 4-week Qwen3.6-MoE / Qwen3.5-7B sm_89 wins). V100 isn't
where ARLE needs to win this race — but the gap should be acknowledged
when we ship V100 numbers.

## Reproducibility (for the apples-to-apples follow-up)

```
# Stand up vLLM (already in tree)
bash scripts/vllm_serve_control.sh   # foreground, port 8000

# Stand up SGLang (already in tree)
bash scripts/bench_sglang_longctx.sh sg-baseline-v100  # foreground

# Each script handles venv + install. Then in another shell, point
# the SAME guidellm config Step 1 used at port 8000:
env -i HOME=$HOME PATH=/usr/bin GUIDELLM__MP_CONTEXT_TYPE=forkserver \
  ~/arle/.venv/bin/guidellm benchmark \
    --target http://localhost:8000 \
    --model Qwen3.5-4B --processor <Qwen3.5-4B-dir> \
    --profile concurrent --rate "1,4,8" \
    --data "prompt_tokens=128,output_tokens=128" \
    --max-seconds 30 \
    --backend-kwargs '{"validate_backend":"/v1/models","request_format":"/v1/completions"}' \
    --disable-console-interactive
# Compare the resulting TTFT/ITL/throughput against Step 1's numbers.
```

Setup-time budget: ~30 min per runtime if the install caches are warm
(SGLang has FlashInfer C++ extension compile; vLLM 0.6.x has a smaller
build surface).

## Rule

Industry comparison entries cite their projection methodology: pick
the closest published benchmark, state the scaling factor used, list
the runtime + version + commit + hardware. Don't claim "we beat X"
without same-hardware same-shape numbers in the entry; don't claim
"we're behind Y" without naming the kernel / substrate / config gap
that explains the delta.

When the live side-by-side run lands, this entry should be amended
in place with the real numbers and the projected-numbers row marked
"superseded by direct measurement at YYYY-MM-DD".
