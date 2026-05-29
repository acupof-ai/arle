# L4 guidellm TTFT/throughput sweep — bf16/int8/fp8/int4 against `infer`

## Context

L4 day-1 deliverable #2 from
[`docs/plans/2026-05-29-l4-day1-bench-handover.md`](../../plans/2026-05-29-l4-day1-bench-handover.md).
Mirror of the V100 sweep
([`2026-05-29-guidellm-ttft-throughput-v100-qwen35.md`](2026-05-29-guidellm-ttft-throughput-v100-qwen35.md))
on **L4 sm_89 / 24 GB / CUDA 12.8**, same client config, same model
`Qwen3.5-4B`, so the cross-hardware delta is the headline. One server
per precision (bf16 / int8 / fp8 / int4), torn down between runs to
drop the KV pool cleanly. Identical guidellm 0.6.0 config across
precisions:

```
guidellm benchmark \
  --target http://localhost:8000 \
  --model Qwen3.5-4B --processor /content/Qwen3.5-4B \
  --profile concurrent --rate "1,4,8" \
  --data "prompt_tokens=128,output_tokens=128" \
  --max-seconds 30 \
  --backend-kwargs '{"validate_backend":"/v1/models","request_format":"/v1/completions"}' \
  --disable-console-interactive
```

Server flags: `--num-slots 16 --kv-cache-dtype <prec>`. Build:
`cargo build --release -p infer --bin infer --features cuda`,
`TORCH_CUDA_ARCH_LIST=8.9`, tilelang 0.1.10, CUDA graph on, mem-fraction
0.85, `effective_max_seq_len=4096` (auto). **Substrate footnote:** the L4
build needed exactly **one** workaround — `ARLE_CUDA_DISABLE_FLASHMLA=1`
(sm_89 still lacks `__nv_fp8_e8m0`) — vs the V100 session's three, confirming
the handover-doc prediction. No `strcasecmp` / NCCL-gate regressions.

KV pool sizing (server log, `paged_kv.rs:350`), all formats at the same
9.1 GB budget / num-slots 16:

| prec | max tokens | data/layer | ratio vs bf16 |
|---|---|---|---|
| bf16     | 278,352 | 1140.1 MB | 1.00× |
| int8     | 439,872 |  900.9 MB | 1.58× |
| fp8 E4M3 | 439,872 |  900.9 MB | 1.58× |
| int4     | 439,872 |  450.4 MB | 1.58× (token cap not byte-bound here) |

The 1.58× capacity ratio matches V100's 610K/961K exactly. **Peak demand
at this shape (128+128 × c=8 ≈ 2K tokens) is <1 % of even the bf16 pool,
so pool capacity is NOT the active lever — it is ruled out below.**

## Results — TTFT, ITL, TPOT (median, p95)

128-in / 128-out, 30-second per-rate window. `c` = concurrency
(guidellm `--rate`). `ok/inc` = completed / incomplete-at-window-close.

```
prec   c  ok/inc  ReqLat(s)      TTFT (ms)       ITL (ms)      TPOT (ms)
                   mdn   p95     mdn    p95     mdn    p95     mdn    p95
bf16   1   7/0    4.57  4.60     69.3  103.9   35.41  35.43   35.68  35.92
bf16   4  21/3    5.15  5.17    237.3  239.5   38.70  39.15   40.25  40.41
bf16   8  41/7    5.59  5.64    302.3  429.9   41.59  42.52   43.65  44.05
int8   1   7/0    4.76  4.78     69.3  100.6   36.85  36.93   37.15  37.35
int8   4  24/0    5.14  5.18    176.9  244.4   39.39  40.22   40.17  40.50
int8   8  48/0    5.64  5.65    279.0  450.7   42.20  43.84   44.04  44.10
fp8    1   7/0    4.76  4.79     68.6   99.7   36.94  37.04   37.19  37.44
fp8    4  24/0    5.18  5.18    176.6  245.7   39.37  40.24   40.44  40.50
fp8    8  48/0    5.62  5.64    280.3  437.8   42.18  43.73   43.89  44.07
int4   1   7/0    4.75  4.79     69.5   98.3   36.88  36.97   37.14  37.45
int4   4  24/0    5.16  5.19    175.3  245.2   39.34  40.28   40.33  40.53
int4   8  48/0    5.63  5.64    280.3  451.4   42.17  43.82   44.02  44.04
```

## Results — server throughput (tokens/s, mean)

```
prec   c   in_tok/s   out_tok/s   total_tok/s
bf16   1     33.0       28.1        56.4
bf16   4    105.3       87.6       175.8
bf16   8    189.4      157.4       316.1
int8   1     31.7       27.0        54.2
int8   4    119.5       99.6       200.1
int8   8    217.0      182.1       365.7
fp8    1     31.6       26.9        54.1
fp8    4    118.9       99.2       199.1
fp8    8    217.5      182.6       366.6
int4   1     31.6       27.0        54.2
int4   4    119.1       99.3       199.4
int4   8    217.3      182.4       366.2
```

## L4 vs V100 (mirrored entry) — same shape, same client

c=1 metrics are single-stream → the cleanest cross-hardware ground truth.
V100 numbers quoted from the mirrored entry.

```
                c=1 TTFT(ms)      c=1 ITL(ms)      c=8 total_tok/s
prec      L4 / V100   ratio   L4 / V100  ratio    L4 / V100   ratio
bf16     69.3 / 229.6  0.30×  35.4 / 14.7  2.41×  316.1/109.8  2.88×
int8     69.3 / 229.6  0.30×  36.9 / 13.7  2.69×  365.7/110.9  3.30×
fp8      68.6 / 230.0  0.30×  36.9 / 13.7  2.69×  366.6/110.0  3.33×
int4     69.5 / 229.6  0.30×  36.9 / 13.7  2.69×  366.2/110.2  3.32×
```

- **Prefill: L4 ≈ 3.3× faster** (TTFT c=1 ~69 ms vs ~230 ms). Prefill is
  compute-bound; sm_89 + the TileLang AOT prefill kernel beat sm_70.
- **Decode: L4 ≈ 2.4–2.7× slower per token** (ITL c=1 ~35 ms vs ~14 ms).
  Decode is memory-bandwidth-bound; L4 GDDR6 (~300 GB/s) vs V100 HBM2
  (~900 GB/s) is a ~3× bandwidth gap — the ITL ratio sits right in that
  band. *(Mechanism is a hypothesis consistent with the bandwidth ratio;
  not ncu-profiled this run.)*
- **c=8 aggregate throughput: caution.** The L4 numbers scale monotonically
  (56→176→316 for bf16). The mirrored V100 throughput table is
  **non-monotonic in c** (bf16 c=1 123.6 → c=4 57.8 → c=8 109.8), so the
  "~3× at c=8" column above is dominated by the V100 row not scaling with
  concurrency, not a clean hardware win. Treat the c=1 TTFT/ITL rows as the
  trustworthy cross-hardware deltas; the c=8 throughput row needs a
  same-protocol V100 re-run before any "L4 batches 3× better" claim.

## What it says

1. **TTFT at c=1 is byte-identical (~69 ms) across all four precisions.**
   Same lesson as V100: the TileLang prefill kernel ingests BF16
   activations regardless of pool format; per-token KV quantize happens
   after prefill compute. Each precision tier is a memory move, not a
   compute move, at the prefill stage.

2. **At c≥4 the quantized formats win output throughput by 12–14 %**
   (out_tok/s c=4: bf16 87.6 vs int8/fp8/int4 ~99; c=8: bf16 157.4 vs
   ~182). This is the **opposite** of V100, where all precisions were flat
   at ~55 out_tok/s c=8. The crossover is sharp: at **c=1 bf16 marginally
   *wins*** (28.1 vs 27.0 out_tok/s — no contention, no dequant overhead),
   then loses as concurrency rises.

3. **bf16 leaves requests incomplete at c≥4; quantized formats leave none**
   (3/24 at c=4, 7/48 at c=8 vs 0/0). With the pool <1 % utilized, this is
   not capacity exhaustion — it is bf16 completing fewer requests inside
   the 30 s window because its batched-decode throughput is lower, so more
   in-flight requests are still running at cutoff.

4. **Hypothesis for #2/#3 (NOT yet ncu-verified): L4 decode-bandwidth
   contention.** Under batched decode each concurrent stream re-reads its
   KV every step; bf16 moves 2× the KV bytes/token of int8/fp8 (and the
   contiguous-prefill→paged migration also moves more). On V100's 900 GB/s
   HBM2 that byte volume had headroom at this shape → flat. On L4's
   ~300 GB/s GDDR6 it saturates once c≥4 → quantized formats, reading fewer
   KV bytes, sustain higher batched throughput. The c=1-wins / c≥4-loses
   crossover and the V100-vs-L4 flip are both consistent with a
   bandwidth-bound decode, but confirming the lever needs an ncu
   `dram__throughput` profile of the batched-decode kernel — flagged as the
   next experiment, not claimed as evidence.

5. **int8 / fp8 / int4 are functionally indistinguishable at the server
   level on this shape** (within ~1 % on every row). Same as V100: the
   precision distinction that matters is K/V quality + pool capacity, not
   serving latency on a shape that fits any precision's pool. int4 buys no
   extra serving headroom here because its token cap is the same 439,872 —
   the byte halving only matters once the pool is the binding constraint
   (long-context / high-c), which this shape is not.

## Caveat — shape and concurrency scope

Single shape (128 in / 128 out), three concurrencies (1, 4, 8), low n at
c=1 (7 requests / 30 s window). Does NOT measure long-context TTFT,
high-concurrency (c≥16) pool saturation, or output-heavy shapes — the
exact gaps flagged in the V100 entry, plus two L4-specific ones:

- **The bandwidth-contention hypothesis (#4) is unproven** — needs an ncu
  profile or a controlled KV-bytes-vs-throughput sweep to license it as
  the root cause of the c≥4 quantized win.
- **The c=8 cross-hardware throughput delta is protocol-suspect** (V100
  table non-monotonic) — needs a same-binary same-protocol V100 re-run.

Where this matrix is directly usable: high-concurrency multi-turn agent
(many short prompts + short outputs) on L4 gets the c≥4 quantized
throughput win — and on bandwidth-starved L4 that win is real (12–14 %),
unlike the wash it was on V100.

## Reproducibility

```bash
# Build (one substrate workaround on sm_89):
export CUDA_HOME=/usr/local/cuda PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST=8.9 INFER_TILELANG_PYTHON=$(which python3)
export ARLE_CUDA_DISABLE_FLASHMLA=1
cargo build --release -p infer --bin infer --features cuda

# Sweep — one server per precision, torn down between:
for prec in bf16 int8 fp8 int4; do
  pkill -9 -x infer; sleep 3   # NB: -x infer (exact comm), NOT -f target/release/infer
  ./target/release/infer --model-path /content/Qwen3.5-4B --port 8000 \
                         --kv-cache-dtype "$prec" --num-slots 16 &
  # wait for /v1/models 200, then:
  env -i HOME=$HOME PATH=/usr/bin:/usr/local/bin GUIDELLM__MP_CONTEXT_TYPE=forkserver \
    /usr/local/bin/guidellm benchmark --target http://localhost:8000 \
      --model Qwen3.5-4B --processor /content/Qwen3.5-4B \
      --profile concurrent --rate "1,4,8" \
      --data "prompt_tokens=128,output_tokens=128" --max-seconds 30 \
      --backend-kwargs '{"validate_backend":"/v1/models","request_format":"/v1/completions"}' \
      --disable-console-interactive --output-path /content/bench/$prec/guidellm.json
done
```

Three L4-box gotchas (kept so the next person doesn't burn the hour I did):

- **`pkill -9 -f target/release/infer` SELF-KILLS the launcher.** The
  cleanup pattern matches the bash process running the sweep (its argv
  contains `./target/release/infer …`), SIGKILLing the shell before the
  server ever starts — symptom is "server exits 1 with zero output and no
  log file". Use `pkill -9 -x infer` (exact process name) instead.
- **`env -i … PATH=/usr/bin` cannot find guidellm** — it lives in
  `/usr/local/bin`. Add `/usr/local/bin` to the stripped PATH or call the
  absolute path. (V100 used a venv path so never hit this.)
- **`env -i` + `validate_backend=/v1/models`** are both still mandatory —
  same httpx-proxy and `/health`-404 reasons as the V100 entry.

## Rule

For the regression gate on L4, run guidellm at c=1/4/8 with
`prompt_tokens=128,output_tokens=128` per KV format, same-binary
same-shell sequential. Median TTFT and ITL delta <5 % per format vs the
numbers in *this* entry = no regression. **Do not reuse the V100 entry's
absolute numbers as the L4 gate** — L4 decode is ~2.4× slower and prefill
~3.3× faster; they are different operating points. Any cross-hardware or
cross-precision throughput claim must cite the specific shape (128/128,
c≤8) and, for the bandwidth-contention story, wait for the ncu profile
before being stated as cause rather than hypothesis.
