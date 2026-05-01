# Qwen3 FP8 KV Tier 1 Numerical Gate Failed

## Context

Phase 1 longctx W1 closed on the FP8 KV path, so the post-P2.B correctness
audit must verify that FP8 performance did not come from a numerically broken
decode path.

The prior FP8 numerical drift entry focused on Qwen3.5 and showed ARLE FP8 vs
BF16 at only `39.08%` common-token match after fixing two durable-KV scale
bugs. This Tier 1 run checks mission-critical Qwen3-4B directly on current
HEAD.

## Hypothesis

ARLE Qwen3-4B FP8 KV should stay close enough to ARLE Qwen3-4B BF16 KV under
greedy decoding:

- 32 prompts
- 256 generated tokens per prompt
- `temperature=0`
- `ignore_eos=true`
- `return_token_ids=true`
- pass threshold: average common-prefix token match `>= 70%`

## Command

Build:

```bash
ZIG=/tmp/zig14/zig CUDA_HOME=/usr/local/cuda \
  cargo build --release -p infer --features cuda
```

BF16 server:

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
`scripts/longctx_numerical_gate.py`, with `return_token_ids=true`. BF16 and FP8
were run sequentially because two Qwen3-4B services do not fit comfortably on
one L4 under the normal KV envelope.

## Environment

- **GPU:** NVIDIA L4, 23,034 MiB
- **Driver:** 580.82.07
- **CUDA target:** sm_89
- **Model:** Qwen3-4B, `infer/models/Qwen3-4B`
- **Commit:** `06ae08f2`
- **BF16 KV pool:** 78,448 tokens, 4,903 pages, page size 16
- **FP8 KV pool:** 144,352 tokens, 9,022 pages, page size 16
- **Feature set:** `cargo build --release -p infer --features cuda`

## Results

| Metric | Value | Status |
| --- | ---: | --- |
| Prompt pairs | 32 | recorded |
| Generated tokens per side | 8,192 | recorded |
| Exact pairs | 0 / 32 | FAIL |
| Exact pair rate | 0.0% | FAIL |
| Average common-prefix token match | 0.43% | FAIL |
| Divergence p50 | token 1 | FAIL |
| Earliest divergence | token 1 | FAIL |
| BF16 average latency / prompt | 8,579.8 ms | recorded |
| FP8 average latency / prompt | 8,637.5 ms | recorded |
| 70% gate | FAIL | blocker |

Case distribution:

- 30 / 32 prompts diverged after exactly one generated token.
- 1 / 32 diverged after two generated tokens.
- 1 / 32 diverged after three generated tokens.

The first generated token is generally stable, but decode immediately drifts
once generated KV participates in attention.

## Root Cause

This run did not isolate the exact kernel bug, but it narrows the failure:

1. Prefill logits are probably close enough to produce the same first token.
2. The first decode step writes generated-token KV into the paged pool.
3. The second generated token diverges for nearly every prompt, so the issue is
   likely in FP8 KV write/read semantics after the first generated token, not
   tokenizer, sampling, or prompt construction.

The next audit should compare BF16 vs FP8 at the first decode boundary:

- RoPE application order before pool write
- QK-norm application timing before quantization
- `quantize_scatter_kv_fp8_range` scale computation and scale offset
- `decode_attention_varlen_fp8.cu` accumulation precision and scale readback
- mixed-batch prefill path for FP8 generated rows

SGLang FP8-vs-BF16 was not rerun in this tranche because no pinned SGLang
service was active on this host. The previous SGLang reference remains
`77.54%` from `2026-04-30-arle-fp8kv-numerical-drift.md`, but Qwen3 needs a
fresh same-host SGLang spot-check before final root-cause closure.

## Fix

Do not expand FP8 default policy based on throughput alone.

Next steps:

1. Add a focused first-decode-step diagnostic: same prompt, force first token
   from BF16, then compare K/V page contents, scales, and second-token logits.
2. Audit Qwen3 FP8 generated-token writeback and readback before running more
   longctx performance sweeps.
3. Re-run this exact 32x256 gate after any FP8 KV patch.
4. Run SGLang FP8-vs-BF16 on the same prompt set when a pinned SGLang service
   is available.

## Rule

For FP8 KV, a non-empty long-prompt smoke test is not a numerical gate. The
minimum gate is token-id trajectory comparison across at least 32 prompts and
256 generated tokens. Divergence at generated token 1 is a hard blocker.

## Artifacts

- BF16 outputs: `bench-output/2026-05-02-fp8-qwen3-numerical-tier1/arle-qwen3-bf16-32x256.json`
- FP8 outputs: `bench-output/2026-05-02-fp8-qwen3-numerical-tier1/arle-qwen3-fp8-32x256.json`
- Compare JSON: `bench-output/2026-05-02-fp8-qwen3-numerical-tier1/arle-qwen3-fp8-vs-bf16-32x256-compare.json`
- Compare summary: `bench-output/2026-05-02-fp8-qwen3-numerical-tier1/arle-qwen3-fp8-vs-bf16-32x256-summary.md`
- BF16 server log: `bench-output/2026-05-02-fp8-qwen3-numerical-tier1-bf16-server/server.log`
- FP8 server log: `bench-output/2026-05-02-fp8-qwen3-numerical-tier1-fp8-server/server.log`

