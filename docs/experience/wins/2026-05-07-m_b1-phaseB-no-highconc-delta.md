# 2026-05-07 · M_b.1 Phase B bench — no measurable delta at high-conc

## Goal

- Validate the M_b.1 Phase B dispatch routing (`45e1d0c`) at the
  canonical M3.6 high-conc shape. Phase B activates the dedicated
  TileLang HD128 decode kernel for `max_qlen == 1` paths; mixed
  batches still use the prefill alias.
- Goal type: **regression check** + perf delta. The Phase 1 trace
  motivated M_b on the assumption decode_attention HD128 was a
  measurable cost; F4-Small's win then closed the per-row decode
  gap. Question: does M_b.1 Phase B add anything on top of
  F4-Small at the standard bench shape?

## Hypothesis

The dedicated decode kernel drops the unused causal-mask + Q-tile
sweep that the prefill alias paid for. At `qlen=1`, this should
be ~10-20% faster per attention call. With 4104 attention calls
in the 60s Phase 1 capture, that translates to ~few % throughput
gain at high-conc.

## Command

Same as M3.6 Phase 1 / F4-Small bench:
```bash
target/release/infer --port 8000 --max-seq-len 5120 --num-slots 48
guidellm benchmark run --profile concurrent --rate 64 \
  --data 'prompt_tokens=1024,...,output_tokens=256,...' \
  --max-seconds 45 --warmup 5 --random-seed 20260416
```

Build at HEAD (after `45e1d0c` lands the dispatch routing).
Cargo build took 3m 36s due to TileLang regenerating the new
HD128 decode kernel cubins (`HD128/HD256 prefill, HD128/HD256
decode, Qwen3.5 GDR` from build.rs warning).

## Environment

- Hardware: NVIDIA RTX 4080 SUPER, CUDA 13.2.1, SM 8.9
- Build: HEAD (commit chain F4-Small → M_d.1 → M_b.1 Phase A → Phase B)
- Server: `--num-slots 48 --max-seq-len 5120`
- Bench: 1024 in / 256 out, c=64, 45s + 5s warmup
- Workload: Qwen3-4B, FP8 KV pool

## Results

| Metric | F4-Small (`8f83c80`) | M_b.1 Phase B | Δ |
|---|---:|---:|---:|
| out tok/s mean | 843.4 | **841.0** | **-0.3% (scout noise)** |
| in tok/s mean | 3,846 | 3,875 | +0.7% |
| req/s | 2.925 | 2.9 | flat |
| Concurrency | 64 | 64 | flat |

**No measurable performance delta at high-conc.** The new TileLang
HD128 decode kernel is correctness-preserving (e2e + greedy_consistency
pass per `45e1d0c` commit message) but does not measurably improve
throughput at the M3.6 canonical workload.

## Why no gain?

Two plausible reasons:
1. **High-conc is batch-amortized**: each tick processes ~38
   decode rows in one CUDA-graph replay. The "extra Q-tile sweep"
   work in the prefill alias kernel is per-call overhead that
   gets amortized when batched. At qlen=1 with batch 38, the
   savings per call × 38 may be too small to surface.
2. **F4-Small already moved the bottleneck off attention**.
   Phase 1 trace had decode_attention at 41.6% of GPU; after
   F4-Small removed the 65 ms sync wait, GPU stays busier, and
   the next-tier bottleneck is likely GEMM or memory-bandwidth
   bound — making attention micro-optimizations marginal.

The Phase B kernel is still the right architectural choice (clean
separation between decode and prefill code paths, future-proof
for kernel-specific optimizations). It just doesn't move the
high-conc bench number.

## Where it might matter

- **Long-context decode** (large KV state per row): per-call
  attention cost grows with KV length. At 8k+ context the savings
  per call could exceed the noise floor. Bench at long-ctx 8k/c=4
  was attempted but the bench window timed out before completed
  requests landed; deferred to a follow-up run with longer
  --max-seconds.
- **Single-row decode** (c=1 latency-sensitive): batch=1 means no
  amortization, the savings per call show up directly as ITL.
  Untested in this iteration.

## Bench Status

- Bench artifact: `bench-output/2026-05-07-m_b1-phaseB-s48-highconc/`
  (gitignored).
- Long-ctx bench at `bench-output/2026-05-07-m_b1-phaseB-longctx-8k/`
  exists but contains 0-completion data; ignore until rerun with
  `--max-seconds 120`.

## Next actions

1. **M_b.1 Phase B is shipped but on cold standby for high-conc**.
   Don't revert — correctness preserved, future optimizations may
   exploit the dedicated kernel.
2. **Long-ctx M_b.1 Phase B re-bench** with `--max-seconds 120`
   when GPU is free.
3. **Single-row latency bench** (c=1, prompt 256, output 256) to
   characterize the savings without amortization.
4. **Higher-leverage parallel work — B.1.2 (prefill async sync)**.
   Codex is already on this; expected TTFT improvement -800ms at
   long-ctx based on the M3.7 plan. Bench validation after commit.

## Rule

- **Kernel changes that look good in micro-bench may not surface
  in batch-amortized macro-bench**. Always validate at the actual
  production workload before claiming a gain. M_b.1's "dedicated
  HD128 decode kernel saves Q-tile sweep overhead" is true but
  too small to measure at batch 38.
- **F4-Small was a unicorn** in moving the macro-bench by +82.5%.
  Subsequent M3.6 / M3.7 / M_b.x increments will mostly be
  smaller, shape-dependent gains. Don't expect another F4-Small
  scale of change.
