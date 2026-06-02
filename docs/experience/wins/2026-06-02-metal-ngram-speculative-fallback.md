# Metal ngram speculative fallback prototype

Date: 2026-06-02

## SLO-shape probed?

N. This was a local c=1 `metal_bench` probe on the canonical Metal Qwen3.6
model, not a guidellm sweep and not a production SLO shape. It licenses the
local-repeat prototype only; it does not license a default flip.

## Roofline check

Deferred. The goal here was token-level correctness, acceptance accounting, and
wall-clock A/B for a new draft source. No MLX/Xcode per-op roofline trace was
captured. Default decision remains deferred until natural workload coverage and
per-block verifier cost are traced.

## Goal

Check whether a request-local ngram draft source can recover speculative decode
benefit after the MTP route showed low acceptance on local Qwen3.6.

## Hypothesis

If the prompt/output has repeated token patterns, ngram can get high suffix
acceptance without MTP draft-model cost. If no candidate exists, the request
must automatically fall back to standard decode quickly enough to avoid a large
regression.

## Implementation

- Added env-gated Qwen3.6/Qwen3.5-MoE ngram speculative decode in
  `infer/src/backend/metal/request_state.rs`.
- Candidate generation is a request-local linear suffix scan over committed
  token history.
- Target verification reuses the C++ `verify_block_summary` path on
  `[current_token, ngram_suffix...]`; committed tokens remain target-owned.
- Partial rejection replays GDR to the accepted prefix with
  `qwen35_rollback_to_accepted_varlen`.
- If MTP state exists, ngram verify refreshes the MTP seed hidden from the
  accepted target row.
- `metal_bench` now reports ngram blocks, draft tokens, accepted draft tokens,
  average accepted inputs, and acceptance rate.
- If enabled but no candidate is found for `ARLE_METAL_NGRAM_MAX_MISSES`
  consecutive steps, ngram disables itself for that request and standard decode
  can resume double-buffer prequeue.

## Commands

```bash
cargo test -p infer --no-default-features --features metal,no-cuda ngram_draft -- --nocapture
cargo build -p infer --release --no-default-features --features metal,no-cuda --bin metal_bench

./target/release/metal_bench \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --use-step-driver \
  --prompt "Write a compact Rust function that reverses a string and explain it briefly." \
  --generation-tokens 32 \
  --warmup 1 --runs 3 --ignore-eos --json

ARLE_METAL_NGRAM_SPEC=1 ARLE_METAL_NGRAM_MAX_DRAFT_TOKENS=8 \
./target/release/metal_bench \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --use-step-driver \
  --prompt "Write a compact Rust function that reverses a string and explain it briefly." \
  --generation-tokens 32 \
  --warmup 1 --runs 3 --ignore-eos --json

ARLE_METAL_NGRAM_SPEC=1 ARLE_METAL_NGRAM_MAX_DRAFT_TOKENS=8 \
ARLE_METAL_NGRAM_MIN_MATCH=64 \
./target/release/metal_bench \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --use-step-driver \
  --prompt "Write a compact Rust function that reverses a string and explain it briefly." \
  --generation-tokens 32 \
  --warmup 1 --runs 3 --ignore-eos --json
```

The earlier token-count synthetic prompt smoke was superseded. This entry must
be rerun on real prompts before it can be used as benchmark evidence.

## Environment

- Backend: Metal / MLX step-driver path.
- Model: `mlx-community/Qwen3.6-35B-A3B-4bit`.
- Hardware: local Apple Silicon Mac.
- Feature set: `--release --no-default-features --features metal,no-cuda`.
- Non-default env: listed per command.

## Results

Warm-pair, same binary, same prompt, same generation length:

| route | gen tok/s mean | TTFT mean ms | total mean ms | ngram blocks | accepted draft | acceptance | delta gen tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| standard step-driver | 86.80 | 202.34 | 570.99 | n/a | n/a | n/a | baseline |
| ngram max_draft=8 | 185.20 | 200.23 | 373.03 | 12 | 96/96 | 100% | +113.4% |
| forced no-candidate (`min_match=64`) | 84.14 | 200.97 | 581.38 | 0 | 0 | n/a | -3.2% |

Exploratory cold single-run tuning showed block size matters:

| max_draft | gen tok/s | blocks | acceptance |
|---:|---:|---:|---:|
| 4 | 6.10 | 7 | 100% |
| 8 | 18.81 | 4 | 100% |
| 12 | 14.39 | 3 | 100% |
| 16 | 12.41 | 2 | 100% |

The warm result is the decision anchor. The cold table only shows that too
small or too large a block can lose even at 100% acceptance because verifier
shape cost and prequeue loss dominate.

## Problems

- The earlier positive workload used the removed token-count prompt generator.
  Treat those numbers as invalid for performance evidence; rerun on real prompt
  files before using this entry as a win.
- The current prototype is linear-scan local history, not SGLang's corpus/trie
  NGRAM worker or tree verifier.
- Enabling ngram globally still costs a few standard decode steps on requests
  with no candidate before the miss fallback disables it.

## Learnings

- Yes, a speculative performance win is possible locally, but not from the MTP
  head on this pairing. The measured win comes from cheap high-acceptance ngram
  draft on repeated context.
- Acceptance alone is insufficient. `max_draft=4` had 100% acceptance and still
  lost; wall-clock tokens/sec is the ground truth.
- A production route should be an adaptive router: use ngram only when history
  has a strong candidate, use MTP only when recent MTP suffix acceptance is
  healthy, otherwise standard decode.

## Rule

Do not default speculative decode based on reachability or acceptance counters
alone. For Metal Qwen3.6, route selection must be per request and must include
wall-clock A/B plus a no-candidate fallback measurement.
