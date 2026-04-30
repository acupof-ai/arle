# Longctx Phase 1 S3 numerical gate harness — pending remote

## Goal

- Correctness gate: make Phase 1 S3 repeatable for long-prompt smoke,
  ARLE FP8-vs-BF16 token trajectories, ARLE-BF16 vs SGLang-BF16 trajectories,
  and the 32x256 long-tail sweep.

## Hypothesis

- A target-agnostic HTTP runner can reproduce the documented S3 spot-checks
  without embedding server launch policy, so remote validation can run the
  same script against independently launched ARLE and SGLang services.

## Command

Local validation:

```bash
python3 -m py_compile scripts/longctx_numerical_gate.py
python3 scripts/longctx_numerical_gate.py --help
# fake local OpenAI-compatible server/tokenizer smoke: pass, 2/2 exact pairs
```

Remote S3 sequence:

```bash
# 1. Long-prompt e2e non-degeneracy
python3 scripts/longctx_numerical_gate.py \
  --label arle-fp8-longprompt-e2e \
  --left-name arle-fp8 \
  --left-url "$ARLE_FP8_URL" \
  --tokenizer infer/models/Qwen3-4B \
  --prompt-count 1 \
  --prompt-tokens 32768 \
  --max-tokens 64 \
  --ignore-eos \
  --out-dir bench-output/2026-04-30-longctx-s3-arle-fp8-longprompt-e2e

# 2. ARLE FP8 vs ARLE BF16, 16 prompts x 64 generated tokens
python3 scripts/longctx_numerical_gate.py \
  --label arle-fp8-vs-bf16-16x64 \
  --left-name arle-fp8 \
  --left-url "$ARLE_FP8_URL" \
  --right-name arle-bf16 \
  --right-url "$ARLE_BF16_URL" \
  --tokenizer infer/models/Qwen3-4B \
  --prompt-count 16 \
  --max-tokens 64 \
  --ignore-eos \
  --left-extra-json '{"return_token_ids":true}' \
  --right-extra-json '{"return_token_ids":true}' \
  --out-dir bench-output/2026-04-30-longctx-s3-arle-fp8-vs-bf16-16x64

# 3. ARLE BF16 vs SGLang BF16, 16 prompts x 64 generated tokens
python3 scripts/longctx_numerical_gate.py \
  --label arle-bf16-vs-sglang-bf16-16x64 \
  --left-name arle-bf16 \
  --left-url "$ARLE_BF16_URL" \
  --right-name sglang-bf16 \
  --right-url "$SGLANG_BF16_URL" \
  --tokenizer infer/models/Qwen3-4B \
  --prompt-count 16 \
  --max-tokens 64 \
  --ignore-eos \
  --left-extra-json '{"return_token_ids":true}' \
  --right-extra-json '{"return_logprob":true,"logprob_start_len":-1,"top_logprobs_num":0,"return_text_in_logprobs":false}' \
  --out-dir bench-output/2026-04-30-longctx-s3-arle-bf16-vs-sglang-bf16-16x64

# 4. Long-tail sweep, 32 prompts x 256 generated tokens
python3 scripts/longctx_numerical_gate.py \
  --label arle-fp8-vs-bf16-32x256 \
  --left-name arle-fp8 \
  --left-url "$ARLE_FP8_URL" \
  --right-name arle-bf16 \
  --right-url "$ARLE_BF16_URL" \
  --tokenizer infer/models/Qwen3-4B \
  --prompt-count 32 \
  --max-tokens 256 \
  --ignore-eos \
  --left-extra-json '{"return_token_ids":true}' \
  --right-extra-json '{"return_token_ids":true}' \
  --out-dir bench-output/2026-04-30-longctx-s3-arle-fp8-vs-bf16-32x256
```

## Environment

- **Backend:** CUDA / SGLang
- **Model:** Qwen/Qwen3-4B
- **Hardware:** pending remote NVIDIA L4
- **Commit:** pending; fill after commit lands
- **Feature set:** `CUDA_HOME=/usr/local/cuda cargo build --release`
- **Non-default flags / env vars:** `ARLE_FP8_URL`, `ARLE_BF16_URL`,
  `SGLANG_BF16_URL`, ARLE `return_token_ids`, SGLang `return_logprob`
- **Server launch:** external to the runner; launch each target with the KV
  dtype and baseline pin under test. Comparison targets must expose generated
  token IDs in the completion response; text re-tokenization is intentionally
  rejected as S3 evidence.

## Results

- Status: `pending-remote`

| check | local result |
|---|---|
| `python3 -m py_compile scripts/longctx_numerical_gate.py` | pass |
| `python3 scripts/longctx_numerical_gate.py --help` | pass |
| fake ARLE-extra two-target HTTP/tokenizer smoke | pass, 2/2 exact pairs |
| fake SGLang `meta_info.output_token_logprobs` smoke | pass, 1/1 exact pair |
| fake `--prompt-tokens` exact re-tokenization smoke | pass |
| fake single-target prompt-usage smoke | pass, server prompt count checked |
| fake compare without token IDs | stop, as expected |
| fake compare with `token_ids: []` and nonzero completion count | stop, as expected |
| fake compare with partial token IDs | stop, as expected |
| fake compare without `--ignore-eos` | stop before dispatch, as expected |
| `ZIG=.toolchains/zig/zig-x86_64-linux-0.16.0/zig cargo test -p infer --release http_server::openai_v1::tests::completion_response_exposes_token_ids_when_requested` | pass |
| `ZIG=.toolchains/zig/zig-x86_64-linux-0.16.0/zig cargo test -p infer --release completion_response_includes_token_ids_when_requested` | pass |
| `ZIG=.toolchains/zig/zig-x86_64-linux-0.16.0/zig cargo clippy -p infer --release -- -D warnings` | pass |
| `ZIG=.toolchains/zig/zig-x86_64-linux-0.16.0/zig cargo test -p infer --release` | pass, 452 passed / 9 ignored |

## Problems

- This workspace does not have Qwen3-4B weights/tokenizer under
  `infer/models/Qwen3-4B`, nor CUDA ARLE/SGLang services, so real S3 gates
  remain remote work.
- First local `cargo test -p infer --release ...` attempt failed before tests
  because `zig` was not on `PATH`; resolved by the repo-local
  `scripts/setup_zig_toolchain.sh --print-zig` toolchain and explicit `ZIG=`.

## Learnings

- S3 should compare token trajectories from already launched services; the
  correctness runner should not hide server launch, KV dtype, or commit pin
  policy.
- The gate reports both average common-prefix token match and
  `divergence_p50`, matching the project threshold of >=70% pass,
  60-70% degraded, <60% stop, with p50 divergence at token >=30 for pass.
- Strict prefixes are scored against the requested trajectory length, and any
  failed comparison pair forces `stop`; missing tokens are not dropped from
  the aggregate.
- Trajectory comparisons require real generated token IDs from the server.
  Re-tokenized text is not accepted because it can hide ID-level drift.
- Compare mode requires `--ignore-eos`, so max-token scoring measures the
  intended fixed-length trajectory rather than penalizing exact early-EOS
  matches as short outputs.
- ARLE exposes those IDs through an explicit non-standard
  `return_token_ids=true` completion request extension; default OpenAI response
  shape stays unchanged, and streaming requests reject the extension because
  SSE chunks do not carry token trajectories.
- Synthetic `--prompt-tokens` prompts are decoded and then re-tokenized before
  dispatch; mismatch is a hard failure instead of silently accepting a shorter
  or longer long-context smoke.
- Generated `token_ids` must be present and match `usage.completion_tokens`
  exactly; missing, empty-when-nonzero, partial, or unverifiable trajectories
  force `stop`.
- Single-target long-prompt smoke checks `usage.prompt_tokens` against the
  locally tokenized prompt, so a wrong context window or silent clamp cannot
  pass as 32k evidence.

## Delta vs baseline

- **Baseline:** prior S3 evidence existed only as retrospective docs and raw
  artefact references, not as a reusable checked-in runner.

| metric | baseline | now | delta |
|---|---|---|---|
| reusable S3 runner | absent | `scripts/longctx_numerical_gate.py` | added |
| 16x64 trajectory summary | manual artefacts | `summary.json` and `summary.md` | reproducible |
| 32x256 long-tail command | absent | documented remote command | added |
| ARLE completion token IDs | internal only | `return_token_ids=true` response extension | added |

## Artefacts

- Raw: pending remote
- JSON summaries: pending remote
- Markdown summaries: pending remote

## Notes

- Code since baseline:
  - Added `scripts/longctx_numerical_gate.py`, an HTTP-only numerical gate
    runner for single-target non-degeneracy and two-target trajectory
    comparison.
