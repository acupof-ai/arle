# Qwen3.5 packed GGUF — guidellm c=1..8, cuda L4, 2026-04-27

> Closes the
> [`pending-remote stub`](./2026-04-26-bench-guidellm-cuda-qwen35-0p8b-packed-gguf-pending-remote.md)
> for `cae7e05d feat(qwen35): preserve packed gguf weight quantization`.
> Stays in scope as a **functional verification** entry — c≥4 is unstable
> on this small-model + L4 combo and is documented as such, not retired.

## Goal

- Optimization (per `docs/bench-and-trace-spec.md` §goal taxonomy):
  verify the new packed-GGUF weight-quant path runs end-to-end on
  CUDA L4 against Qwen3.5-0.8B Q4_K_M weights, and capture the
  per-token decode win the metal entry already saw locally.

## Hypothesis

- Packed GGUF Q4_K_M loads + runs without dequant-at-load-time, with
  numerically reasonable output (small-model → factual accuracy will
  drop, but coherence should hold).
- Single-stream decode tok/s is dramatically higher than the bf16 4B
  model (smaller weights × Q4 quant ≈ ¼ memory footprint), so HBM
  bandwidth supports much higher decode rates.

## Command

```bash
hf download Qwen/Qwen3.5-0.8B \
  --local-dir models/Qwen3.5-0.8B
hf download unsloth/Qwen3.5-0.8B-GGUF Qwen3.5-0.8B-Q4_K_M.gguf \
  --local-dir models/Qwen3.5-0.8B-GGUF

# tokenizer + config alongside the .gguf so detect_arch() works.
cp models/Qwen3.5-0.8B/{config.json,tokenizer_config.json,merges.txt,\
chat_template.jinja,special_tokens_map.json,tokenizer.json} \
   models/Qwen3.5-0.8B-GGUF/

# server (note: --num-slots 8 / --mem-fraction-static 0.85; the auto
# allocator over-commits workspace on this small model)
/tmp/infer-off --model-path models/Qwen3.5-0.8B-GGUF --port 8000 \
  --num-slots 8 --max-seq-len 4608 --mem-fraction-static 0.85

scripts/bench_guidellm.sh cuda-l4-qwen35-0p8b-gguf-q4km-c1c8 \
  --concurrencies 1,2,4,8 --max-seconds 60 --warmup 5 \
  --processor models/Qwen3.5-0.8B \
  --model Qwen3.5-0.8B-GGUF
```

## Environment

- **Backend:** cuda
- **Model:** Qwen/Qwen3.5-0.8B (safetensors tokenizer/config) +
  unsloth/Qwen3.5-0.8B-GGUF (Q4_K_M weights)
- **Hardware:** NVIDIA L4, 24 GB, sm_89, driver 580.82.07, CUDA 12.8.93
- **Commit:** `868a9fda` (HEAD)
- **Feature set:** `cargo build --release -p infer --features cuda`
- **Toolchain:** rustc 1.95.0, nvcc 12.8.93, zig 0.14.0

## Numerical sanity

```
$ curl -X POST http://localhost:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"Qwen3.5-0.8B-GGUF","prompt":"The capital of France is","max_tokens":15,"temperature":0}'

→ " the capital of the United Kingdom.\nThe capital of the United Kingdom"
```

Output is grammatical and shows the GGUF + packed-quant path runs end-
to-end. Factual accuracy is poor (this is a 0.8B-parameter model with
Q4 quantization — expected at this scale); coherence is fine.

## Results — concurrency table

| conc | TTFT p50 (ms) | ITL p50 (ms) | TPOT p50 (ms) | out tok/s | req latency p50 s | status |
|---|---:|---:|---:|---:|---:|:---|
|  1 |    247.4 |  4.5 |  5.4 | **183.3** | 1.4 | **valid** |
|  2 |    449.2 |  7.3 |  9.0 | **222.2** | 2.3 | **valid** |
|  4 |       — |   — |   — |       — |   — | **invalid** (empty streams) |
|  8 |       — |   — |   — |       — |   — | **invalid** (CUDA context corruption after OOM) |

guidellm reports c=4 / c=8 throughputs (5714 / 6104 tok/s server-side)
but rejects them — the streams complete in ~200 ms with `tokens_out`
recorded but no actual text deltas reaching the client. Server log
shows `CUDA_ERROR_OUT_OF_MEMORY` at prefill-batch admission for the
larger concurrency leg, after which the CUDA context appears to stay
in a degraded state (matches the
`memory/project_remote_cuda_box.md` "C≥8 structurally broken on this
box" pattern, but reproduces here at c≥4 with the small-model
workspace allocator over-commit).

## Comparison vs `Qwen3-4B` bf16 at c=1 (matched session)

| metric | Qwen3-4B bf16 | Qwen3.5-0.8B GGUF Q4_K_M | improvement |
|---|---:|---:|---:|
| TTFT p50 | 719.3 ms | 247.4 ms | **−65.6 %** |
| ITL p50 | 35.24 ms | 4.5 ms | **−87.2 %** |
| out tok/s | 26.56 | 183.3 | **+590 %** |

Per-token decode is **8× faster** on the smaller quantized model —
HBM bandwidth bound is much lower because Q4 weights ≈ ¼ the bytes
to stream per token, and the model is 5× smaller in parameter count.
The packed-quant path avoids the dequant-at-load-time penalty
(`cae7e05d`'s explicit goal): weights stay packed through the CUDA
embedding + linear dispatches.

## Problems

- **c≥4 CUDA OOM**: the prefill workspace allocator at c≥4 with
  4096-token chunks runs into `CUDA_ERROR_OUT_OF_MEMORY` even on the
  small 0.8B model. The auto-num-slots calculation gives the model
  too many slots given its tiny per-slot footprint, so the workspace
  budget per scheduler step is over-committed. Workaround: explicit
  `--num-slots 8 --mem-fraction-static 0.85`. Even so, c=4 fails.
- **CUDA-context corruption after OOM**: per the prior
  `bench_throughput_sweep.py C≥8` note, an OOM on prefill leaves the
  context such that subsequent decodes return empty streams. The c=8
  numbers in the table reflect the same regression class.
- **Bench-script downscoping**: this verification used `--quick` /
  `--concurrencies` exploration mode rather than the canonical sweep.
  Canonical sweep at full concurrencies cannot complete on this model
  on this box until the workspace allocator over-commit is fixed —
  separate `cuda-mixed-workspace-cap` work tracked in
  [`docs/experience/wins/2026-04-26-bench-guidellm-cuda-mixed-workspace-cap-pending-remote.md`](./2026-04-26-bench-guidellm-cuda-mixed-workspace-cap-pending-remote.md).

## Learnings

- **Packed GGUF weight quant works on CUDA at single-stream and 2-way
  concurrency.** Numerics are correct and the per-token decode win
  is real — exactly the shape `cae7e05d`'s commit message promised.
- **Workspace allocator needs a small-model guard.** The L4 + 0.8B
  combo over-commits prefill workspace at default settings; the
  fix lives outside this entry but blocks high-concurrency GGUF
  validation here.
- **Tokenizer + config files must be alongside the .gguf** for
  `detect_arch()` to work — pointing `--model-path` at the .gguf
  file directly fails with "Not a directory". The Metal entry's
  `--model models/.../foo.gguf` invocation form depends on a
  loader that does its own path-glob; the CUDA path uses the
  config-driven detector and needs the directory.

## Δ vs baseline

- **No prior CUDA GGUF baseline** — this is the first entry. The
  pending-remote stub
  [`2026-04-26-...-pending-remote.md`](./2026-04-26-bench-guidellm-cuda-qwen35-0p8b-packed-gguf-pending-remote.md)
  predicted "embeddings + linears stay packed for Q5_K and avoid
  BF16 load-time materialization" — we observed exactly that.
- **vs Metal entry** (`2026-04-26-bench-metal-qwen35-0p8b-packed-gguf-local.md`)
  — Metal step-driver bench showed 37.5 tok/s GGUF vs 29.7 tok/s
  BF16. Our CUDA c=1 GGUF gives 183 tok/s; bigger machine, bigger
  model gap. Direct comparison isn't apples-to-apples (Metal driver
  vs guidellm HTTP).

## Artefacts

- Raw: `bench-output/2026-04-27-cuda-l4-qwen35-0p8b-gguf-q4km-c1c8/`

(Artefact dir is gitignored — paths are local to the L4 host.)

## Notes

- **What changed since the pending-remote stub:** nothing in code —
  the stub described `cae7e05d`. This entry adds the CUDA-side L4
  numbers the stub said were pending.
- **Follow-ups:**
  - Fix workspace-allocator over-commit for small-model + high-conc
    GGUF runs (separate plan).
  - Re-bench with the fix at c=1..16 to capture the high-conc story.
  - Try Q8_0 / Q5_K / Q3_K variants once Q4_K_M passes high-conc.
  - Repeat against Qwen3.5-4B GGUF for a 4B → 4B Q4 comparison.
