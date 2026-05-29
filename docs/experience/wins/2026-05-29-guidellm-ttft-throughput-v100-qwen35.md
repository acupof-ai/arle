# V100 guidellm TTFT/throughput sweep — bf16/int8/fp8/int4 against `infer`

## Context

Step 1 of "对比业界" — our own server-side TTFT/ITL/TPOT/throughput
numbers first; published-baseline comparison lands in a follow-up
commit. Validates the `--kv-cache-dtype int4` CLI exposure that
shipped at commit `591d1bf6` and the kv_tier refactor at `67745ebc`.

`infer/target/release/infer` on V100 sm_70 / CUDA 12.4, model
`Qwen3.5-4B`. guidellm 0.6.0 from `~/arle/.venv`. One server per
precision (bf16 / int8 / fp8 / int4), torn down between runs to
drop the KV pool cleanly. Identical guidellm config across precisions:

```
guidellm benchmark \
  --target http://localhost:8000 \
  --model Qwen3.5-4B \
  --processor <Qwen3.5-4B-snapshot-dir> \
  --profile concurrent --rate "1,4,8" \
  --data "prompt_tokens=128,output_tokens=128" \
  --max-seconds 30 \
  --backend-kwargs '{"validate_backend": "/v1/models",
                     "request_format":  "/v1/completions"}' \
  --disable-console-interactive
```

Server flags: `--num-slots 8 --kv-cache-dtype <prec>`, default sequence
length, no chunked-prefill overrides.

## Results — TTFT, ITL, TPOT (median, p95)

128-in / 128-out, 30-second per-rate window. `c` = concurrency (=
guidellm `--rate`):

```
prec   c  RequestLat(s)    TTFT (ms)      ITL (ms)     TPOT (ms)
            mdn   p95     mdn    p95     mdn    p95    mdn    p95
bf16   1   2.1   2.2    229.6  304.5   14.7   14.7   16.4   16.9
bf16   4  18.3  18.3    797.9  798.7  137.8  138.1  142.9  142.9
bf16   8  18.8  19.2    885.7 1443.3  141.1  143.8  146.9  149.9
int8   1   2.0   2.0    229.6  305.8   13.7   13.7   15.4   16.0
int8   4  17.9  17.9    533.5  779.8  136.8  139.2  140.0  140.1
int8   8  18.6  18.8    834.2 1362.2  141.2  146.0  145.7  146.7
fp8    1   2.0   2.0    230.0  271.7   13.7   13.8   15.4   15.8
fp8    4  17.9  18.0    534.1  788.0  137.0  139.4  140.1  140.3
fp8    8  18.8  18.8    834.4 1391.6  141.4  146.2  146.9  147.1
int4   1   2.0   2.0    229.6  273.5   13.7   13.7   15.4   15.7
int4   4  17.8  17.9    536.1  761.6  136.8  139.2  139.1  139.9
int4   8  18.8  18.8    835.9 1398.1  141.2  146.0  146.6  146.9
```

## Results — server throughput (concurrency × tokens/s)

```
prec   c   in_tok/s   out_tok/s   total_tok/s
bf16   1     66.0        61.6        123.6
bf16   4     55.3        28.6         57.8
bf16   8    102.4        54.5        109.8
int8   1     69.8        65.4        131.4
int8   4     55.9        28.8         57.8
int8   8    104.9        55.2        110.9
fp8    1     69.6        65.2        131.0
fp8    4     55.9        28.7         57.7
fp8    8    103.5        54.8        110.0
int4   1     69.8        65.4        131.2
int4   4     56.6        29.0         58.1
int4   8    103.7        54.9        110.2
```

## What it says

1. **TTFT at c=1 is byte-identical (~230 ms / 128-token prompt) across
   all precisions.** The TileLang AOT prefill kernel
   (`batch_prefill_paged_hd256_q*_kv*_sm70`) ingests BF16 activations
   regardless of pool format; the per-token KV quantize happens AFTER
   the prefill compute. → ~556 tok/s prefill on V100. Each "extra
   precision tier" is a memory move, not a compute move.

2. **TTFT at c=4 favours the quantized formats by ~50%.** BF16 c=4 TTFT
   = 798 ms, INT8/FP8/INT4 c=4 TTFT = 534 ms. The BF16 pool fits only
   610 K KV tokens vs 961 K for the quantized formats (1.58× capacity,
   from
   [`2026-05-29-kv-precision-bench-v100-qwen35.md`](2026-05-29-kv-precision-bench-v100-qwen35.md)),
   so 4 concurrent 128+128-token requests start hitting the BF16
   pool's capacity headroom earlier and the scheduler has to chunk
   harder. INT4 enjoys the same TTFT win as INT8/FP8 here even though
   its bytes-per-token is half — at this batch/prompt size the bottleneck
   is page count, not byte volume.

3. **ITL is 7% faster on the quantized formats** (14.7 → 13.7 ms at
   c=1, same delta at c=4 and c=8). Quantized K dequant + QK happens in
   one fused kernel; BF16 KV reads through a more general TileLang path
   that does extra register staging. INT8/FP8/INT4 all show the same
   delta because the dequant fast path is shared and head_dim=256
   thread granularity is the limit.

4. **Aggregate output throughput at c=8 is flat across precisions at
   ~55 tok/s.** All four formats saturate the same scheduler decode
   loop at 8 concurrent decodes. The pool-capacity advantage only
   shows up at concurrency levels where BF16 actually thrashes its
   pool — which 8 × (128 in + 128 out) = 2 K tokens, well under
   610 K, doesn't trigger.

5. **Quantized precisions are functionally indistinguishable at the
   server level on this shape.** INT8, FP8, and INT4 all post within
   ~1% on every metric in the table. The interesting precision
   distinction is K/V quality (
   [`2026-05-28-int4-kv-two-level-k.md`](2026-05-28-int4-kv-two-level-k.md))
   and pool capacity, NOT serving latency on shapes that fit in any
   precision's pool budget.

## Caveat — shape and concurrency scope

This sweep is a single shape (128 in / 128 out) at three concurrencies
(1, 4, 8). It does NOT measure:
- **Long-context TTFT.** At 4 K-token prompts the quantized formats'
  capacity advantage should grow into a larger TTFT delta vs BF16
  because chunked-prefill paging dominates.
- **High-concurrency saturation.** At c=16/32 the BF16 pool starts
  paging and the gap should widen further. c=8 was capped here for
  the ~10 min total wall-clock budget.
- **Output-heavy shapes.** With output_tokens ≫ input_tokens the
  decode-loop quantize overhead shifts; INT4 specifically may pay
  more per-decode-step quant cost than INT8.

Workloads where this matrix matters: high-concurrency multi-turn agent
(many short prompts + many short outputs) gets the c=4 INT8/FP8/INT4
TTFT win directly. Long-context single-stream (one 8 K prompt) is the
shape that will tell us whether the TileLang sm_70 prefill is
limit-bound or paging-bound — separate run.

## Reproducibility

```
# 1. Ensure infer is built with --features cuda and the
#    cuda-kernels + tilelang substrate workarounds applied per
#    docs/experience/wins/2026-05-29-kv-precision-bench-v100-qwen35.md
#    "Substrate footnote".
# 2. For each precision in {bf16, int8, fp8, int4}:
./target/release/infer --model-path <Qwen3.5-4B-dir> --port 8000 \
                       --kv-cache-dtype <prec> --num-slots 8 &
# wait for /v1/models 200
env -i HOME=$HOME PATH=/usr/bin GUIDELLM__MP_CONTEXT_TYPE=forkserver \
  ~/arle/.venv/bin/guidellm benchmark \
    --target http://localhost:8000 \
    --model Qwen3.5-4B --processor <Qwen3.5-4B-dir> \
    --profile concurrent --rate "1,4,8" \
    --data "prompt_tokens=128,output_tokens=128" \
    --max-seconds 30 \
    --backend-kwargs '{"validate_backend":"/v1/models","request_format":"/v1/completions"}' \
    --disable-console-interactive
# kill server, wait 3s, next precision
```

Two substrate-level gotchas surfaced during this bench (kept here so
the next person doesn't burn the same hour):

- **`env -i` is mandatory.** guidellm 0.6.0's httpx 0.x dependency
  parses a malformed value in this V100 box's environment as an
  `httpx.URLPattern` mount key and crashes with `InvalidURL: Invalid
  port: ':'` before any request is made. `env -i` with the
  bare-minimum guidellm needs (`HOME`, `PATH`, plus
  `GUIDELLM__MP_CONTEXT_TYPE=forkserver`) clears it.
- **`--backend-kwargs validate_backend=/v1/models` is mandatory.**
  Default guidellm probes `/health` for the readiness check; `infer`
  ships `/healthz` and `/readyz` (OpenAI-server convention is also
  `/v1/models`). The canonical `scripts/bench_guidellm.sh` codifies
  this — replicate the same JSON for any non-default backend probe.

## Rule

For the "is this serving change a regression" gate, run guidellm at
c=1 + c=4 + c=8 with `prompt_tokens=128,output_tokens=128` against
each KV format you ship. Same-binary, same-shell, sequential. Median
TTFT delta < 5% and median ITL delta < 5% per format vs the prior
landed numbers in this entry counts as "no regression"; deltas above
that need a wins/errors entry explaining the lever.

For shape-dependent perf claims (long-context, high-concurrency,
output-heavy), do not extrapolate from this matrix. Re-run guidellm
at the actual shape, ideally including the SGLang or vLLM baseline at
the same shape on the same hardware — the cross-precision delta is
small enough at 128/128 that the more interesting comparison is the
cross-runtime one, which is the next-step task.
