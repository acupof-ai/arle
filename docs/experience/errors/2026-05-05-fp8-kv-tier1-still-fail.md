# Qwen3 FP8 KV Tier 1 Still Fails

## Context

Commit `00def315` applied the Qwen3 FP8 paged-prefill KV finalization fix and
the follow-up review repair that restores quantized prefix rows before paged
prefill attention. The required Tier 1 gate from
`2026-05-02-qwen3-fp8-kv-numerical-tier1-fail.md` was rerun on the same
Qwen3-4B 32-prompt, 256-token greedy trajectory comparison.

The patch improves the trajectory slightly, but it does not clear the gate.

## Command

Build and static gates already completed in the code tranche:

```bash
cargo build --release -p infer --features cuda --bin infer
cargo clippy --release -p infer --features cuda -- -D warnings
cargo test --release -p infer --features cuda flashinfer::
cargo check -p infer --no-default-features --features cuda,no-cuda
codex review -
```

BF16 reference server:

```bash
RUST_LOG=info CUDA_HOME=/usr/local/cuda ./target/release/infer \
  --model-path infer/models/Qwen3-4B \
  --port 8011 \
  --kv-cache-dtype bf16 \
  --num-slots 4 \
  --max-seq-len 4096 \
  --mem-fraction-static 0.85 \
  --max-num-batched-tokens 4096 \
  --max-prefill-tokens 4096 \
  --schedule-policy fcfs
```

FP8 server:

```bash
RUST_LOG=info CUDA_HOME=/usr/local/cuda ./target/release/infer \
  --model-path infer/models/Qwen3-4B \
  --port 8012 \
  --kv-cache-dtype fp8 \
  --num-slots 4 \
  --max-seq-len 4096 \
  --mem-fraction-static 0.85 \
  --max-num-batched-tokens 4096 \
  --max-prefill-tokens 4096 \
  --schedule-policy fcfs
```

Collection used the prompt builder and response parser from
`scripts/longctx_numerical_gate.py`, with `return_token_ids=true`, run
sequentially for BF16 then FP8. The HTTP API does not expose full logits, so
the requested `max_abs_delta <= 1e-3` logits criterion remains unverified; the
token trajectory gate fails before this can be accepted.

## Environment

- GPU: NVIDIA L4, 23,034 MiB
- Driver: 580.82.07
- CUDA target: sm_89
- Model: Qwen3-4B, `infer/models/Qwen3-4B`
- Commit: `00def315`
- Feature set: `cargo build --release -p infer --features cuda`

## Results

| Metric | 2026-05-02 | 2026-05-05 | Status |
| --- | ---: | ---: | --- |
| Prompt pairs | 32 | 32 | recorded |
| Generated tokens per side | 8,192 | 8,192 | recorded |
| Exact pairs | 0 / 32 | 0 / 32 | FAIL |
| Exact pair rate | 0.0% | 0.0% | FAIL |
| Average common-prefix token match | 0.43% | 1.22% | FAIL |
| Divergence p50 | token 1 | token 1.5 | FAIL |
| Earliest divergence | token 1 | token 1 | FAIL |
| Token-1 divergences | 30 / 32 | 16 / 32 | FAIL |
| BF16 average latency / prompt | 8,579.8 ms | 8,578.2 ms | recorded |
| FP8 average latency / prompt | 8,637.5 ms | 8,637.9 ms | recorded |
| 70% gate | FAIL | FAIL | blocker |

Common-prefix distribution:

| Common prefix tokens | Prompt pairs |
| ---: | ---: |
| 1 | 16 |
| 2 | 2 |
| 3 | 7 |
| 4 | 4 |
| 6 | 1 |
| 18 | 1 |
| 19 | 1 |

## Root Cause

The patch fixed a real paged-prefill quantized KV finalization hole and the
subsequent review found that quantized prefix rows also needed restoration
before suffix/chunk paged-prefill attention. The remaining failure shows those
were not the only FP8 KV numerical bug.

The new result is directionally better than the previous failure, but the
first generated token trajectory is still not stable enough: half the prompts
diverge at token index 1, and the average common-prefix match remains far below
the 70% threshold.

## Fix

Do not mark the FP8 KV Tier 1 fix as passed. Keep FP8 KV numerical correctness
blocked until a focused internal diagnostic can compare full logits and K/V
page contents across BF16 and FP8 at the first decode boundary.

Next steps:

1. Add an in-process Qwen3 diagnostic that can compare BF16 vs FP8 full logits
   directly, including the requested max-abs logit delta.
2. Compare durable FP8 generated rows and scales after the first decode write.
3. Compare `decode_attention_varlen_fp8` readback against dequantized BF16 for
   the same page table and positions.
4. Rerun this exact 32x256 gate after the next FP8 KV patch.

## Rule

FP8 KV patches must pass the 32x256 token trajectory gate before any throughput
win is accepted. HTTP token-ID parity is necessary but not sufficient; add a
full-logit diagnostic before claiming the `<= 1e-3` logits criterion.

## Artifacts

- BF16 outputs: `bench-output/2026-05-05-fp8-kv-tier1/arle-qwen3-bf16-32x256.json`
- FP8 outputs: `bench-output/2026-05-05-fp8-kv-tier1/arle-qwen3-fp8-32x256.json`
- Compare JSON: `bench-output/2026-05-05-fp8-kv-tier1/arle-qwen3-fp8-vs-bf16-32x256-compare.json`
- Compare summary: `bench-output/2026-05-05-fp8-kv-tier1/arle-qwen3-fp8-vs-bf16-32x256-summary.md`
- BF16 server log: `bench-output/2026-05-05-fp8-kv-tier1-bf16-server/server.log`
- FP8 server log: `bench-output/2026-05-05-fp8-kv-tier1-fp8-server/server.log`
