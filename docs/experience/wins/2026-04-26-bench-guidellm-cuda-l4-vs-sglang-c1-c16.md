# Head-to-head vs SGLang — guidellm c=1..16, cuda L4, 2026-04-26

## Goal

- Comparison (per `docs/bench-and-trace-spec.md` §goal taxonomy):
  benchmark current `infer` HEAD against SGLang 0.5.10.post1 at
  matched server flags + matched bench params on the same L4 box,
  same session. Companion to the 2026-04-22
  [`infer-vs-sglang-c1-c16` entry](./2026-04-22-bench-guidellm-qwen3-4b-infer-vs-sglang-c1-c16.md)
  — measures how the post-2026-04-22 mixed-batch refactors and the
  Phase 0 TileLang AOT have moved us against the same opponent.

## Hypothesis

- ITL (per-token decode latency): infer should match or beat SGLang.
  Our decode kernel is hand-written CUDA with autotuned cuBLASLt; the
  ITL parity in the 2026-04-22 entry already showed this.
- TTFT and aggregate throughput: SGLang should still hold an edge at
  c≥4. The mixed-batch refactors closed a chunk of the prior gap (see
  [`mixed-batch-vs-f98ca92.md`](./2026-04-26-bench-guidellm-cuda-l4-mixed-batch-vs-f98ca92.md)),
  but SGLang's chunked prefill is still ahead.
- TileLang AOT vs FlashInfer vs SGLang: TileLang on Ada (sm_89) should
  land near FlashInfer parity; the +15 % TTFT regression vs FlashInfer
  observed in the
  [`tilelang-prefill-hd128-floor`](./2026-04-26-bench-guidellm-cuda-l4-tilelang-prefill-hd128-floor.md)
  entry should NOT widen the gap to SGLang appreciably.

## Command

All three runs use the same flags + same guidellm invocation. SGLang's
`--page-size` defaults to 1 (vs our 16); other flags match.

```bash
# infer (FlashInfer, the matched-flags baseline)
/tmp/infer-off --model-path models/Qwen3-4B --port 8000 \
  --num-slots 16 --max-seq-len 4608 --mem-fraction-static 0.94 \
  --chunked-prefill-size 4096 --max-prefill-tokens 16384

# infer (TileLang AOT)
/tmp/infer-on --model-path models/Qwen3-4B --port 8000 \
  --num-slots 16 --max-seq-len 4608 --mem-fraction-static 0.94 \
  --chunked-prefill-size 4096 --max-prefill-tokens 16384

# SGLang
python3 -m sglang.launch_server \
  --model-path models/Qwen3-4B --served-model-name Qwen3-4B --port 8000 \
  --mem-fraction-static 0.94 --dtype bfloat16 \
  --max-running-requests 16 --context-length 4608 \
  --disable-cuda-graph-padding --disable-piecewise-cuda-graph

# Bench (same call for all three)
scripts/bench_guidellm.sh <label> --concurrencies 1,2,4,8,16 \
  --max-seconds 60 --warmup 5 \
  --processor /content/workspace/agent-infer/models/Qwen3-4B
```

## Environment

- **Backend (infer):** cuda
- **Model:** Qwen/Qwen3-4B
- **Hardware:** NVIDIA L4, 24 GB, sm_89, driver 580.82.07, CUDA 12.8.93
- **Commit (infer):** `2a4ff6ce` (HEAD)
- **SGLang:** `0.5.10.post1` (matches the 2026-04-22 baseline version)
- **Toolchain:** rustc 1.95.0, nvcc 12.8.93, zig 0.14.0, tilelang 0.1.9

## Results — three-way concurrency table

| conc | SGLang TTFT ms | infer-FI TTFT ms | infer-TL TTFT ms | SGLang ITL ms | infer-FI ITL ms | infer-TL ITL ms | SGLang tok/s | infer-FI tok/s | infer-TL tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  1 |    686.3 |    719.3 |    835.7 | 35.33 | 35.24 | 35.19 |  26.65 | 26.56 | 26.31 |
|  2 |   1343.7 |   1518.6 |   1750.8 | 40.88 | 38.77 | 38.40 |  46.31 | 45.21 | 45.37 |
|  4 |   2006.2 |   2354.4 |   2476.6 | 48.54 | 41.62 | 44.19 |  74.84 | 53.31 | 52.80 |
|  8 |   2778.8 |   3838.0 |   4390.7 | 61.06 | 52.30 | 51.86 | 109.39 | 66.94 | 71.72 |
| 16 |   5520.6 |  16356.9 |  16681.7 | 93.01 | 49.27 | 49.13 | 139.19 | 65.92 | 67.29 |

(`infer-FI` = `--features cuda` FlashInfer baseline,
`infer-TL` = `--features cuda,tilelang-attn` TileLang AOT.)

## Δ infer vs SGLang (negative numbers = infer faster / better)

### infer-FI vs SGLang

| conc | Δ TTFT | Δ ITL p50 | Δ tok/s |
|---|---:|---:|---:|
|  1 |   +4.8 % |  −0.3 % |  −0.3 % |
|  2 |  +13.0 % |  **−5.2 %** |  −2.4 % |
|  4 |  +17.3 % | **−14.3 %** |  −28.8 % |
|  8 |  +38.1 % | **−14.4 %** |  −38.8 % |
| 16 | +196.3 % | **−47.0 %** |  −52.6 % |

### infer-TL vs SGLang

| conc | Δ TTFT | Δ ITL p50 | Δ tok/s |
|---|---:|---:|---:|
|  1 |  +21.8 % |  −0.4 % |  −1.3 % |
|  2 |  +30.3 % |  **−6.1 %** |  −2.0 % |
|  4 |  +23.4 % |  **−9.0 %** |  −29.4 % |
|  8 |  +58.0 % | **−15.1 %** |  −34.4 % |
| 16 | +202.2 % | **−47.2 %** |  −51.7 % |

### infer-TL vs infer-FI (same backend, just kernel swap)

| conc | Δ TTFT | Δ ITL p50 | Δ tok/s |
|---|---:|---:|---:|
|  1 | +16.2 % |  −0.1 % |  −0.9 % |
|  2 | +15.3 % |  −1.0 % |  +0.4 % |
|  4 |  +5.2 % |  +6.2 % |  −1.0 % |
|  8 | +14.4 % |  −0.8 % |  +7.1 % |
| 16 |  +2.0 % |  −0.3 % |  +2.1 % |

## Where infer wins vs SGLang (record this)

**Per-token decode latency (ITL p50)** — infer is consistently 5–47 %
faster than SGLang at every concurrency where it's measurable. This
is the kernel-level win: our cuBLASLt-autotuned per-batch decode plus
Triton-AOT QKV norm/RoPE/silu-mul fuses eat fewer milliseconds per
token than SGLang's FlashInfer-driven decode pipeline.

| conc | infer-FI ITL p50 | SGLang ITL p50 | infer's lead |
|---|---:|---:|---:|
|  4 | 41.62 ms | 48.54 ms | **−14.3 %** |
|  8 | 52.30 ms | 61.06 ms | **−14.4 %** |
| 16 | 49.27 ms | 93.01 ms | **−47.0 %** |

The c=16 ITL gap is dramatic — SGLang's ITL nearly doubles between
c=8 and c=16 while infer's stays flat. SGLang is paying tail-latency
cost for keeping more requests in-flight; infer's per-step cost is
much more stable.

## Where SGLang still wins

- **TTFT at c≥4**: SGLang's prefill admission is leaner. At c=16
  SGLang serves first-token in 5.5 s; infer takes 16.4 s.
- **Aggregate throughput at c≥4**: SGLang sustains higher tok/s
  because it pushes more requests through prefill per second. At c=16
  SGLang hits 139 tok/s vs our 66; the 2× gap closed from the prior
  3× by the mixed-batch refactors but isn't yet a win.

The pattern: SGLang fills the GPU faster (better admission), infer
runs each request faster once they reach decode (better kernel).
Closing the throughput gap is the prefill-admission story; that's
where future scheduler work needs to land.

## Δ vs prior baseline

Comparison against the 2026-04-22 entry
([`infer-vs-sglang-c1-c16`](./2026-04-22-bench-guidellm-qwen3-4b-infer-vs-sglang-c1-c16.md)):

| metric | 2026-04-22 (`f98ca92`) | 2026-04-26 (`2a4ff6ce`) | Δ |
|---|---:|---:|---|
| infer TTFT p50 @ c=4 | 14556.7 ms | 2354.4 ms | **−83.8 %** |
| infer TTFT p50 @ c=8 | 15403.7 ms | 3838.0 ms | **−75.1 %** |
| infer out tok/s @ c=4 | 36.70 | 53.31 | **+45.3 %** |
| infer out tok/s @ c=16 | 45.08 | 65.92 | **+46.2 %** |
| infer/sglang tok/s ratio @ c=16 | 0.329 | 0.474 | +14.5 pp |

The mixed-batch alignment + Phase 0 work delivered substantial wins
against the same opponent at the same flags. Closing the remaining
~52 % aggregate-throughput gap is the next prefill-admission task.

## Artefacts

- SGLang: `bench-output/2026-04-26-sglang-l4-vs-current/`
- infer-FI: see [`mixed-batch-vs-f98ca92`](./2026-04-26-bench-guidellm-cuda-l4-mixed-batch-vs-f98ca92.md) artefacts.
- infer-TL: `bench-output/2026-04-26-cuda-l4-tilelang-on-vs-sglang/`

(Artefact dirs are gitignored — paths are local to the L4 host.)

## TileLang tunables — what room is left on Ada (sm_89)?

Same-session probe of two alternative tile configs to see whether
TileLang's L4 floor moves under simple parameter changes. Both
recompiled and re-benched with the same matched flags as `infer-TL`
above.

| variant | TTFT p50 ms (c=1 → c=16) | tok/s (c=1 → c=16) | numerics |
|---|---|---|---|
| **baseline** `(BLOCK_M=64, BLOCK_N=64, NUM_STAGES=2, NUM_THREADS=128)` | 836 / 1751 / 2477 / 4391 / 16682 | 26.3 / 45.4 / 52.8 / 71.7 / 67.3 | OK |
| `BLOCK_M=128` (wider Q tile) | server starts, sanity probe fails (`!!!!!!!!`) | n/a | **broken** — kernel boundary handling assumes `BLOCK_M=64` |
| `NUM_STAGES=3` (deeper pipeline) | 835 / 1735 / 2729 / 4157 / 16330 | 26.3 / 45.1 / 52.3 / 67.4 / 66.6 | OK, parity vs baseline |

**`STAGES=3` Δ vs baseline (TileLang):**

| conc | Δ TTFT | Δ tok/s |
|---|---:|---:|
|  1 |  −0.1 % |   flat |
|  2 |  −0.9 % |  −0.6 % |
|  4 | +10.2 % |  −0.9 % |
|  8 |  −5.3 % |  −6.0 % |
| 16 |  −2.1 % |  −1.1 % |

Both alternates land within the ±10 % matched-A/B noise band (c=4 TTFT
+10 % is the only outlier). On Ada there's no obvious low-hanging
tunable — the kernel is roughly as fast as it can be without
architecture-specific intrinsics. Real wins live one of two places:

1. **Hopper:** the §5 H100 spike is where the 15 % TTFT regression vs
   FlashInfer should reverse — TileLang's TMA / WGMMA / warp-spec
   emit isn't activated on sm_89.
2. **Ada-specific micro-optimization** (probably not worth pursuing
   inside Phase 0): explicit `cp.async` prologue / Q preload outside
   the K/V pipeline, or a Triton-based kernel that does pick up Ada
   intrinsics that TileLang's CuTeDSL emit doesn't.

For now the recipe stays at `(64, 64, 2, 128)`. The tunable sweep is
a recorded null result — the room-on-Ada question has been asked and
answered: not much without architectural change.

## Notes

- All three runs (SGLang + infer-FI + infer-TL) were back-to-back in
  the same session on the same GPU with the same matched server flags.
  Comparison is direct.
- SGLang's chunked-prefill default is 2048 (vs our 4096) and page-size
  is 1 (vs our 16). Those are SGLang defaults we did not override —
  it's the comparison "as both teams ship them," not a forced parity.
- The tile-sweep also surfaced one durable codebase improvement: the
  AOT generator's `host_kernel_source` regex was brittle against tile
  variants. Replaced with a slot-pair lookup that's robust to TileLang's
  per-shape stack layout (committed alongside this entry).
- Follow-ups: see the verification report
  [`2026-04-26-l4-scheduler-and-tilelang-verification.md`](../../reviews/2026-04-26-l4-scheduler-and-tilelang-verification.md)
  for the cross-cutting list. The biggest single lever for closing
  the throughput gap is prefill admission backpressure when waiting >
  num_slots — that's where TTFT p50 still spikes at c=16.
