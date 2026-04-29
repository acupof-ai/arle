# CUDA L4 Headline Summary — post plan-fix, 2026-04-29

## Goal

Headline baseline: summarize every configuration that can complete full
inference without a known blocker after today's warmup, scheduler, emit, hybrid
workspace, and HD256 plan-gate fixes.

## Hypothesis

The full fix chain should make Qwen3.5 BF16 valid again, keep Qwen3/Qwen3.5
fp8 runs complete, and show whether fp8 KV is ready to become the ARLE default.

## Command

ARLE rows were rerun on HEAD with one server restart per row:

```bash
/tmp/arle-target/release/infer \
  --model-path infer/models/<Qwen3-4B|Qwen3.5-4B> \
  --port 8000 \
  --num-slots 16 \
  --max-seq-len 8192 \
  --kv-cache-dtype <fp8|bf16>

scripts/bench_guidellm.sh <label> \
  --target http://localhost:8000 \
  --model <Qwen/Qwen3-4B|Qwen/Qwen3.5-4B> \
  --processor infer/models/<Qwen3-4B|Qwen3.5-4B> \
  --profile concurrent \
  --concurrencies 16 \
  --max-seconds 120
```

SGLang rows are reused from today's existing artefacts; no SGLang server was
rerun in this tranche.

## Environment

- Hardware: NVIDIA L4, 23.66 GB VRAM, driver 580.82.07
- ARLE commit: `024e44e6` (`2a9ef4ad` runtime plus docs)
- ARLE feature set: release `cuda`, default `tilelang-attn` ON,
  `tilelang-decode-hd256` OFF
- Workload: c=16, prompt 4096, output 256, 120s, seed `20260416`
- KV: fp8 means FP8E4M3 paged KV with BF16 contiguous prefill cache; bf16 means
  BF16 paged KV

## Results Matrix

| model | backend | KV dtype | tok/s | TTFT p50 | ITL p50 | GPU SM% | KV pool tokens | status |
|---|---|---|---:|---:|---:|---:|---:|---|
| Qwen3-4B | ARLE | fp8 | 138.17 | 13236.6ms | 72.55ms | 100.0 | 148,256 | valid, rerun on HEAD |
| Qwen3-4B | ARLE | bf16 | 83.97 | 12899.6ms | 102.68ms | 99.5 | 78,160 | valid, rerun on HEAD; KV pressure high |
| Qwen3.5-4B | ARLE | fp8 | 151.76 | 13273.2ms | 57.14ms | 99.0 | 398,624 | valid, rerun on HEAD |
| Qwen3.5-4B | ARLE | bf16 | 150.27 | 12617.4ms | 64.08ms | 99.9 | 249,136 | valid after HD256 plan gate fix |
| Qwen3-4B | SGLang | fp8_e4m3 | 133.74 | 8375.7ms | 86.20ms | 76.2 | n/a | reused from `sglang-align-c16`; server KV log not retained |
| Qwen3-4B | SGLang | bf16 | n/a | n/a | n/a | n/a | n/a | skipped: no reusable artefact found; not rerun by request |
| Qwen3.5-4B | SGLang | fp8_e4m3 | 167.32 | 6572.3ms | 70.40ms | 96.3 | 337,910 | reused from fp8 KV ablation |
| Qwen3.5-4B | SGLang | bf16 | 162.41 | 6328.6ms | 72.93ms | 96.4 | 168,955 | reused from fp8 KV ablation |

Skipped configurations:

- `tilelang-decode-hd256`: sm89 build fails; see
  `docs/experience/errors/2026-04-29-tilelang-decode-hd256-sm89-build.md`.
- `tilelang-attn` OFF: separate TileLang ON/OFF diagnosis, intentionally not
  mixed into this headline summary.
- Any previously blocked Qwen3.5 BF16 row before `2a9ef4ad`: superseded by the
  valid plan-fix run above.

## Fix Chain

| commit | effect |
|---|---|
| `047beccf` | Server readiness waits for scheduler warmup, so benches no longer include cold startup/capture. |
| `047ca156` | Added scheduler step-phase timing/profile data used to decide CPU/GPU overlap work. |
| `39db7e9d` | Moved `loop_total` timing to the true loop tail after metrics updates. |
| `b63712fa` | Reserved Qwen3.5 hybrid workspace and capped concurrent long prefill rows. |
| `c5b4ab7e` | Carried scheduler-side generated tokens through the emit `Finish` path. |
| `95d2be64` | Added diagnostics for prefill stop-before-emit. |
| `2a9ef4ad` | Gated Qwen3.5 BF16 HD256 `plan_hd256` on the actual decode-kernel feature, fixing CUDA error 9. |

## Deltas

ARLE fp8 vs bf16:

| model | tok/s | TTFT p50 | ITL p50 | KV pool tokens | reading |
|---|---:|---:|---:|---:|---|
| Qwen3-4B | +64.5% | +2.6% slower | -29.3% | +89.7% | fp8 is the clear throughput/capacity winner despite slightly slower TTFT |
| Qwen3.5-4B | +1.0% | +5.2% slower | -10.8% | +60.0% | fp8 gives capacity and ITL, but throughput is near-tie |

ARLE vs SGLang fp8:

| model | ARLE tok/s | SGLang tok/s | tok/s Δ | TTFT p50 Δ | ITL p50 Δ |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B | 138.17 | 133.74 | +3.3% | +58.0% slower | -15.8% |
| Qwen3.5-4B | 151.76 | 167.32 | -9.3% | +102.0% slower | -18.8% |

ARLE vs SGLang bf16:

| model | ARLE tok/s | SGLang tok/s | tok/s Δ | TTFT p50 Δ | ITL p50 Δ |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B | 83.97 | n/a | n/a | n/a | n/a |
| Qwen3.5-4B | 150.27 | 162.41 | -7.5% | +99.4% slower | -12.1% |

## Numerical Spot-Check

Both spot-checks used 16 fixed prompts, seed `20260429`, `temperature=0`,
`max_tokens=64`, then tokenized generated text with
`infer/models/Qwen3.5-4B`.

| backend | pair | exact pairs | avg common-token match rate | earliest divergence | conclusion |
|---|---|---:|---:|---:|---|
| ARLE | fp8 vs bf16 | 0 / 16 | 2.79% | generated token 0 | fails fp8 default precision gate |
| SGLang | fp8_e4m3 vs bf16 | 8 / 16 | 77.54% | generated token 2 | not identical, but far less divergent than ARLE |

Spot-check artefacts:

- ARLE fp8: `bench-output/2026-04-29-arle-qwen35-spotcheck-fp8.json`
- ARLE bf16: `bench-output/2026-04-29-arle-qwen35-spotcheck-bf16.json`
- ARLE compare: `bench-output/2026-04-29-arle-qwen35-spotcheck-compare.json`
- SGLang fp8: `bench-output/2026-04-29-sglang-qwen35-spotcheck-fp8.json`
- SGLang bf16: `bench-output/2026-04-29-sglang-qwen35-spotcheck-bf16.json`
- SGLang compare: `bench-output/2026-04-29-sglang-qwen35-spotcheck-compare.json`

## Conclusion

Do not flip ARLE fp8 KV to the default yet.

The performance case is strong for Qwen3 and mixed-but-positive for Qwen3.5
capacity/ITL. The blocker is correctness: ARLE Qwen3.5 fp8 vs bf16 diverges
from the first generated token on some prompts and only matches 2.79% of common
tokens. fp8 should stay opt-in until the quantized KV decode path's numerical
behavior is understood and brought within tolerance.

## Problems

- SGLang Qwen3 BF16 was requested as reusable, but no matching artefact or
  wins row exists in this workspace. It was not rerun because the instruction
  said SGLang rows should be reused.
- Qwen3 BF16 ARLE runs at 99.6% peak KV utilization and has lower completed
  token count in the fixed 120s window; it is valid but capacity-constrained.
- Qwen3.5 TTFT remains roughly 2x SGLang for both fp8 and bf16 even though ARLE
  ITL is better.

## Learnings

- Qwen3.5's previous BF16 blocker was not CUDA graph related; it was a plan
  gate mismatch between `tilelang-attn` and `tilelang-decode-hd256`.
- The combined scheduler fixes now make all ARLE fp8/bf16 rows complete, but
  fp8 defaulting must be gated on numerical comparison, not throughput alone.
- ARLE's current strength is ITL; the remaining SGLang gap is TTFT/prefill
  scheduling.

## Artefacts

- ARLE Qwen3 fp8: `bench-output/2026-04-29-headline-arle-qwen3-fp8/`
- ARLE Qwen3 fp8 server: `bench-output/2026-04-29-headline-arle-qwen3-fp8-server/server.log`
- ARLE Qwen3 fp8 dmon: `bench-output/2026-04-29-headline-arle-qwen3-fp8-dmon/gpu_dmon.csv`
- ARLE Qwen3 bf16: `bench-output/2026-04-29-headline-arle-qwen3-bf16/`
- ARLE Qwen3 bf16 server: `bench-output/2026-04-29-headline-arle-qwen3-bf16-server/server.log`
- ARLE Qwen3 bf16 dmon: `bench-output/2026-04-29-headline-arle-qwen3-bf16-dmon/gpu_dmon.csv`
- ARLE Qwen3.5 fp8: `bench-output/2026-04-29-headline-arle-qwen35-fp8/`
- ARLE Qwen3.5 fp8 server: `bench-output/2026-04-29-headline-arle-qwen35-fp8-server/server.log`
- ARLE Qwen3.5 fp8 dmon: `bench-output/2026-04-29-headline-arle-qwen35-fp8-dmon/gpu_dmon.csv`
- ARLE Qwen3.5 bf16: `bench-output/2026-04-29-arle-qwen35-bf16kv-planfix/`
- ARLE Qwen3.5 bf16 server: `bench-output/2026-04-29-arle-qwen35-bf16kv-planfix-probe-server/server.log`
- ARLE Qwen3.5 bf16 dmon: `bench-output/2026-04-29-arle-qwen35-bf16kv-planfix-dmon/gpu_dmon.csv`
- SGLang Qwen3 fp8: `bench-output/2026-04-29-sglang-cuda-l4-qwen3-c16-sglang-align-run2/`
- SGLang Qwen3.5 fp8: `bench-output/2026-04-29-sglang-qwen35-fp8kv-on/`
- SGLang Qwen3.5 bf16: `bench-output/2026-04-29-sglang-qwen35-bf16kv/`

## Δ vs Baseline

This is the first single headline summary after the full fix chain. Prior
entries remain immutable and are cross-linked above:

- `docs/experience/wins/2026-04-29-bench-guidellm-cuda-l4-fp8kv-ablation.md`
- `docs/experience/wins/2026-04-29-bench-guidellm-sglang-align-c16.md`
- `docs/experience/wins/2026-04-29-bench-guidellm-cuda-l4-tilelang-on-vs-off.md`
