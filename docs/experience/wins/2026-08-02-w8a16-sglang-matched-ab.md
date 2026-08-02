# W8A16 matched A/B vs SGLang — same kernel, same weights, SGLang decodes 1.57× faster — CUDA, 2026-08-02

> Status: **Measured, verdict recorded.** The open question from
> [2026-08-02-w8a16-marlin-tensorcore](2026-08-02-w8a16-marlin-tensorcore.md)
> ("an end-to-end win vs SGLang W8A16 is unproven") is now answered: we lose.
> c=1 decode ITL p50 **ARLE 26.9 ms vs SGLang 17.1 ms** with the *identical*
> gptq_marlin GEMM and *identical* int8 weight values. The gap is the runtime
> between kernels, not the kernel — this converts the decode launch-gap /
> whole-step-graph lever from hypothesis to a measured 9.8 ms/step bounty.

## Method — how the arms were matched

SGLang cannot read ARLE's W8A16 layout, so the ARLE checkpoint
(`iso-tc-huihui-w8a16`, int8 sym gs=128) was **mechanically repacked** into
GPTQ v1 (`/host/w8a16_to_gptq.py` → `iso-tc-huihui-gptq8`): uint8 = int8+128
(exact kU8B128 semantics), scales bf16→fp16, qzeros=127, no re-quantization —
both arms serve the same quantized values. `in_proj_ba` (N=48 < marlin's
64-alignment), visual, and mtp are excluded via GPTQModel `dynamic` negative
match and stay bf16. SGLang 0.5.13 loaded it first try: "Using gptq_marlin
kernel", weights 29.0 GB (ARLE: 29.9 GB).

Both arms: same H20 GPU 6, same day, serial, `bench_throughput.py`
`--prompts-jsonl bench-agent-32k-64.jsonl --concurrency-grid 1
--requests-per-concurrency 16 --max-tokens 256 --seed 20260416`; 16/16
complete × 256 tokens each. SGLang: tp=1, mem-fraction 0.85, CUDA graph
captured (bs 1…56), chunked prefill 8192. ARLE: `arle @ f2c07d0cf` serve
defaults (decode graph flag is a documented no-op under paged KV — eager
launches is the shipped state).

## Results — c=1, ~33K-token prompts

| metric | ARLE | SGLang | ratio |
|---|---:|---:|---:|
| decode ITL p50 | 26.88 ms | **17.07 ms** | 1.57× |
| decode ITL p99 | 27.46 ms | 18.67 ms | 1.47× |
| TTFT p50 | 30.5 s | 21.1 s | 1.45× |

ARLE reproduces its own 2026-08-02 record exactly (26.9 ms) — the number is
stable; the gap is real.

Caveats stated: (1) weight *values* cannot affect GEMM timing (same shapes,
dtypes, bytes), so output-quality equivalence of the repack was not re-eval'd
— the SGLang quantized path is confirmed by its load log + 29 GB resident.
(2) The ARLE binary predates the ctx-bind fix (`b0368426a`), so its TTFT may
have run the recurrent GDR fallback — the TTFT row is directional only; the
decode rows are unaffected (prefill-path changes).

## Learnings

**The kernel was never the gap.** Per the ncu roofline, both stacks run the
same occupancy-bound marlin GEMM (~51 % HBM at m=1). ARLE's decode step
carries **~9.8 ms/step (36 %) of non-GEMM wall** that SGLang does not have —
SGLang decodes bs=1 inside a whole-step captured CUDA graph (zero launch
gaps); ARLE launches eagerly every step. This is now the measured #1 decode
lever, replacing "decode launch-gap graph capture" as a ranked hypothesis.
Decomposing the 9.8 ms (launch gaps vs non-fused elementwise vs host sync)
needs an nsys diff of one decode step per stack — that is the entry ticket
for the fix, per [[feedback_read_scaling_curve_before_kernel_rewrite]].

**Rule: "SOTA by provenance" only bounds the kernel, not the product.** Two
stacks shipping identical kernels can differ 1.57× end-to-end; a SOTA claim
is only ever licensed by a matched same-GPU A/B against the reference stack.
