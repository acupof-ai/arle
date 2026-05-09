# P0.0 Phase 1.B — SGLang Reverify Post P1.0/P1.2

## Goal

Recompute the same-machine ARLE vs SGLang baseline after the P1.0 hybrid
W4A8 prefill dispatch and P1.2 W4A8 graph-capture hoist wins. This was an
evidence gate for deciding whether the next world-#1 step should keep pushing
ops-layer work or pivot to a larger architectural axis.

## Hypothesis

P1.0 and P1.2 may have closed enough of the old SGLang gap that remaining
ops-layer work is saturated. The reverify needed wall-clock TTFT and ITL data
on the same GPU, not inferred deltas from stale 2026-05-07 baselines.

## Environment

| Field | Value |
|---|---|
| Host GPU | NVIDIA GeForce RTX 4070 Ti SUPER, 16 GiB |
| Driver / CUDA | 595.71.05 / CUDA 13.2.78 |
| ARLE bench commit | `b1062d7` docs HEAD, runtime stack includes P1.0 `9773904` + P1.2 `ca0673b` |
| ARLE model | `infer/models/Qwen3-4B-W4-hybrid-zpfix` |
| ARLE flags | `INFER_HYBRID_W4A8_PREFILL=1`, `--num-slots 8`, `--max-seq-len 8192`, `--admission-policy prefix-aware` |
| SGLang package | `sglang==0.5.11`, `sglang-kernel==0.4.2.post1+cu130`, `flashinfer-python==0.6.8.post1` |
| SGLang model | `infer/models/Qwen3-4B-AWQ` |
| SGLang flags | `--dtype half --quantization awq_marlin --kv-cache-dtype fp8_e4m3 --max-running-requests 8 --context-length 8192 --mem-fraction-static 0.85 --max-total-tokens 70000` |

Note: the installed SGLang wheel did not expose a git SHA via package metadata.
This run is pinned by package versions plus the local AWQ checkpoint, not by a
source checkout commit.

## Commands

ARLE server:

```bash
env INFER_HYBRID_W4A8_PREFILL=1 \
  CUDA_HOME=/opt/cuda \
  NVCC_CCBIN=/usr/bin/g++-14 \
  INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
  TORCH_CUDA_ARCH_LIST=8.9 \
  RUST_LOG=info \
  ./target/release/infer \
    --model-path infer/models/Qwen3-4B-W4-hybrid-zpfix \
    --port 8000 --num-slots 8 --max-seq-len 8192 \
    --admission-policy prefix-aware
```

SGLang server:

```bash
PATH=/home/ckl/sglang-venv/bin:$PATH \
LD_LIBRARY_PATH=/home/ckl/sglang-venv/lib_extra:$LD_LIBRARY_PATH \
NVCC_CCBIN=/usr/bin/g++-14 TORCH_CUDA_ARCH_LIST=8.9 \
CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14 TMPDIR=/var/tmp \
/home/ckl/sglang-venv/bin/python -m sglang.launch_server \
  --host 127.0.0.1 --port 8001 \
  --model-path /home/ckl/projects/arle/infer/models/Qwen3-4B-AWQ \
  --dtype half --quantization awq_marlin \
  --kv-cache-dtype fp8_e4m3 \
  --max-running-requests 8 --context-length 8192 \
  --mem-fraction-static 0.85 --max-total-tokens 70000
```

GuideLLM commands:

```bash
for r in 1 2 3; do
  scripts/bench_guidellm.sh p00p1b-arle-hybrid-longctx4k-c4-r${r} \
    --target http://localhost:8000 \
    --model Qwen3-4B-W4-hybrid-zpfix \
    --processor infer/models/Qwen3-4B \
    --concurrencies 4 --max-seconds 120 --warmup 10 \
    --data 'prompt_tokens=4096,prompt_tokens_stdev=1,prompt_tokens_min=4096,prompt_tokens_max=4096,output_tokens=256,output_tokens_stdev=1,output_tokens_min=256,output_tokens_max=256'

  scripts/bench_guidellm.sh p00p1b-sglang-awqmarlin-longctx4k-c4-r${r} \
    --target http://localhost:8001 \
    --model /home/ckl/projects/arle/infer/models/Qwen3-4B-AWQ \
    --processor infer/models/Qwen3-4B-AWQ \
    --concurrencies 4 --max-seconds 120 --warmup 10 \
    --data 'prompt_tokens=4096,prompt_tokens_stdev=1,prompt_tokens_min=4096,prompt_tokens_max=4096,output_tokens=256,output_tokens_stdev=1,output_tokens_min=256,output_tokens_max=256'
done
```

```bash
for r in 2 3 4; do
  scripts/bench_guidellm.sh p00p1b-arle-hybrid-decode256-c1c4-r${r} \
    --target http://localhost:8000 \
    --model Qwen3-4B-W4-hybrid-zpfix \
    --processor infer/models/Qwen3-4B \
    --concurrencies 1,4 --max-seconds 60 --warmup 10 \
    --data 'prompt_tokens=256,prompt_tokens_stdev=1,prompt_tokens_min=256,prompt_tokens_max=256,output_tokens=256,output_tokens_stdev=1,output_tokens_min=256,output_tokens_max=256'
done

for r in 1 2 3; do
  scripts/bench_guidellm.sh p00p1b-sglang-awqmarlin-decode256-c1c4-r${r} \
    --target http://localhost:8001 \
    --model /home/ckl/projects/arle/infer/models/Qwen3-4B-AWQ \
    --processor infer/models/Qwen3-4B-AWQ \
    --concurrencies 1,4 --max-seconds 60 --warmup 10 \
    --data 'prompt_tokens=256,prompt_tokens_stdev=1,prompt_tokens_min=256,prompt_tokens_max=256,output_tokens=256,output_tokens_stdev=1,output_tokens_min=256,output_tokens_max=256'
done
```

Burst commands:

```bash
for r in 1 2 3; do
  mkdir -p bench-output/2026-05-09-p00p1b-arle-hybrid-multitenant-burst-r${r}
  python scripts/bench_multitenant_burst.py \
    http://localhost:8000 \
    Qwen3-4B-W4-hybrid-zpfix | tee \
    bench-output/2026-05-09-p00p1b-arle-hybrid-multitenant-burst-r${r}/output.txt

  mkdir -p bench-output/2026-05-09-p00p1b-sglang-awqmarlin-multitenant-burst-r${r}
  python scripts/bench_multitenant_burst.py \
    http://localhost:8001 \
    /home/ckl/projects/arle/infer/models/Qwen3-4B-AWQ | tee \
    bench-output/2026-05-09-p00p1b-sglang-awqmarlin-multitenant-burst-r${r}/output.txt
done
```

Raw `command.txt` examples:

```bash
GUIDELLM__MP_CONTEXT_TYPE=forkserver guidellm benchmark run \
  --target http://localhost:8000 \
  --model Qwen3-4B-W4-hybrid-zpfix \
  --processor infer/models/Qwen3-4B \
  --profile concurrent \
  --data 'prompt_tokens=4096,prompt_tokens_stdev=1,prompt_tokens_min=4096,prompt_tokens_max=4096,output_tokens=256,output_tokens_stdev=1,output_tokens_min=256,output_tokens_max=256'

GUIDELLM__MP_CONTEXT_TYPE=forkserver guidellm benchmark run \
  --target http://localhost:8001 \
  --model /home/ckl/projects/arle/infer/models/Qwen3-4B-AWQ \
  --processor infer/models/Qwen3-4B-AWQ \
  --profile concurrent \
  --data 'prompt_tokens=4096,prompt_tokens_stdev=1,prompt_tokens_min=4096,prompt_tokens_max=4096,output_tokens=256,output_tokens_stdev=1,output_tokens_min=256,output_tokens_max=256'
```

## Results

Stable rows use the median of N=3 warmed runs.

| Workload | Engine | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | out tok/s | req/s actual | Stability |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 4k/256 c=4 | ARLE | 1639.3 ms | 1821.7 ms | 11.47 ms | 11.62 ms | 223.45 | 0.84 | pass, sigma < 1% |
| 4k/256 c=4 | SGLang | 928.4 ms | 1814.2 ms | 9.41 ms | 12.92 ms | 272.67 | 1.05 | pass, sigma < 1% |
| 256/256 c=1 | ARLE | 13.2 ms | 14.4 ms | 6.79 ms | 6.79 ms | 146.67 | 0.58 | pass, sigma < 4% |
| 256/256 c=1 | SGLang | 36.1 ms | 37.2 ms | 5.43 ms | 5.43 ms | 180.32 | 0.72 | pass, sigma < 2% |
| 256/256 c=4 | ARLE | 32.6 ms | 33.8 ms | 7.09 ms | 7.18 ms | 555.43 | 2.16 | pass on p50/ITL; one p99 outlier excluded from median |
| 256/256 c=4 | SGLang | 111.0 ms | 113.3 ms | 5.79 ms | 6.05 ms | 644.92 | 2.48 | pass, sigma < 1% |
| multi-tenant burst | ARLE | 279 ms | n/a | n/a | n/a | n/a | n/a | fail, TTFT p50 sigma > 5% |
| multi-tenant burst | SGLang | 105 ms | n/a | n/a | n/a | n/a | n/a | fail, TTFT p50 sigma > 5% |

## Delta vs Baselines

Delta is ARLE relative to SGLang for the same workload; negative means ARLE is
faster/lower, positive means ARLE is slower/higher.

| Workload | TTFT delta | ITL delta | out tok/s delta | req/s delta | Verdict |
|---|---:|---:|---:|---:|---|
| 4k/256 c=4 | +76.6% | +21.9% | -18.0% | -20.4% | SGLang still leads prefill-dominant row |
| 256/256 c=1 | -63.4% | +25.0% | -18.7% | -19.4% | ARLE lower TTFT; SGLang better decode throughput |
| 256/256 c=4 | -70.6% | +22.5% | -13.9% | -12.9% | ARLE lower TTFT; SGLang better decode throughput |
| multi-tenant burst | +165.7% | n/a | n/a | n/a | Directional only, sigma gate failed |

Historical reference: the stale 2026-05-07 SGLang 4k/c=4 TTFT p50 was
972.9 ms. This run's SGLang 928.4 ms is 4.6% lower, so using the stale value
would understate the current gap.

Multi-tenant wall time was more stable than per-request TTFT:

| Engine | Burst wall median | TTFT p50 runs |
|---|---:|---|
| ARLE | 1135 ms | 302 / 279 / 243 ms |
| SGLang | 643 ms | 165 / 99 / 105 ms |

Raw artifacts:

- `bench-output/2026-05-09-p00p1b-arle-hybrid-longctx4k-c4-r{1,2,3}/`
- `bench-output/2026-05-09-p00p1b-sglang-awqmarlin-longctx4k-c4-r{1,2,3}/`
- `bench-output/2026-05-09-p00p1b-arle-hybrid-decode256-c1c4-r{2,3,4}/`
- `bench-output/2026-05-09-p00p1b-sglang-awqmarlin-decode256-c1c4-r{1,2,3}/`
- `bench-output/2026-05-09-p00p1b-arle-hybrid-multitenant-burst-r{1,2,3}/`
- `bench-output/2026-05-09-p00p1b-sglang-awqmarlin-multitenant-burst-r{1,2,3}/`

## Problems

- SGLang was first launched with plain `awq`; it warned that `awq_marlin` was
  available. The server was restarted with `--quantization awq_marlin` before
  any recorded SGLang bench rows.
- SGLang first startup included several minutes of JIT and CUDA graph capture;
  startup time is excluded from the bench rows.
- SGLang does not expose ARLE's `/v1/stats`, so service trace fields in the
  GuideLLM artifacts are `n/a`.
- The multi-tenant burst row did not satisfy the sigma < 5% gate. Treat that
  row as directional evidence only; it should be rerun with a longer scripted
  warm cache phase before it drives a narrow regression decision.
- The SGLang wheel install does not satisfy the requested source-commit pin.
  For reproducibility, this entry records package versions and exact launch
  flags.
- The first ARLE 256/256 repeat was a cold outlier (`TTFT p50=37.6 ms` at c=1);
  it was not used for the stable N=3 decode row. Runs r2-r4 are the recorded
  warmed set.

## Learnings

P1.0 + P1.2 did not close the current SGLang gap on the prefill-dominant
4k/c=4 shape. SGLang is still 43.4% lower TTFT and 22.0% higher output
throughput on that row, so the "ops layer is fully saturated" conclusion is not
supported by same-machine evidence.

The short 256/256 rows show the opposite TTFT shape: ARLE reaches first token
much faster than SGLang at c=1 and c=4, while SGLang still has better ITL and
output throughput. That points away from decode-only micro-optimizations as the
next highest-leverage move.

Decision tree outcome:

- Prefill-dominant gap remains > 20%, so do not pivot solely on "ops-layer
  saturation."
- Since P1.3/P1.4/P1.6 already killed three small ops hypotheses, the next
  prefill investigation should be architectural within the prefill path:
  piecewise CUDA graph behavior, prefill dispatch scheduling, or SGLang's
  prefill graph/capture strategy, not another trivial dispatch wire.
- Multi-tenant still needs a stable rerun before choosing between additional
  radix-cache work and a broader scheduling change.

## Rule

When an upstream comparison is used to choose the next axis, rerun the current
upstream stack on the same machine after local wins land. Stale baselines are
only planning hints, not strategic evidence.
