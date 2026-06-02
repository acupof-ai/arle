# Metal MTP depth sweep does not license a default flip

## Goal

Type: optimization / regression gate.

Wire the split Qwen3.6 MTP drafter into the Metal backend, then test whether it
should be enabled by default and whether drafting more tokens helps.

## Hypothesis

The split drafter
`mlx-community/Qwen3.6-35B-A3B-MTP-4bit` declares `block_size = 3`, so the
expected stable depth is current token plus 2 draft tokens. Drafting 3 or 4
tokens may reduce target verifier calls, but should only win if suffix
acceptance stays high enough to pay for the longer target verify block.

## Command

Build:

```text
cargo build --release --no-default-features --features cli,metal,no-cuda --bin arle
cargo build --release -p infer --no-default-features --features metal --bin metal_serve
```

Depth sweep, run serially with one resident model at a time:

```text
RUST_LOG=info QWEN35_MTP_PROFILE=1 ./target/release/metal_serve \
  --model-path mlx-community/Qwen3.6-35B-A3B-4bit \
  --port 8141 \
  --warmup 0 \
  [--mtp-draft-model mlx-community/Qwen3.6-35B-A3B-MTP-4bit \
   --mtp-draft-tokens N]
```

Each case used one 4-token warmup, then 3 streamed `/v1/completions` requests
with the same prompt, `temperature=0`, `max_tokens=32`.

## Environment

- Host: local Apple Silicon Mac, Metal backend, MLX bridge.
- Target model:
  `mlx-community/Qwen3.6-35B-A3B-4bit`, snapshot
  `38740b847e4cb78f352aba30aa41c76e08e6eb46`.
- Draft model:
  `mlx-community/Qwen3.6-35B-A3B-MTP-4bit`, snapshot
  `0295b81421bf4d0fccca9a7c0fcfb1418dda3516`.
- Draft config: `model_type=qwen3_5_mtp`, `block_size=3`,
  `mtp_num_hidden_layers=1`, quantization `4bit/g64`.
- Feature set: `--no-default-features --features metal`.
- Raw artifacts:
  `bench-output/2026-06-02-metal-mtp-ab/{baseline,mtp}.log` and
  `bench-output/2026-06-02-metal-mtp-draft-tokens/`.

## Results

First A/B, non-streaming prompt with `prompt_bytes=109`, default draft depth
2, showed no license for a default flip:

| Case | TTFT ms | TPOT ms | Total ms | tok/s | MTP acceptance |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline | 67.4 | 11.6 | 428.2 | 74.734 | n/a |
| MTP draft=2 | 65.9 | 12.9 | 465.6 | 68.725 | 54 blocks, avg accepted inputs 1.81/3, suffix accept 0.407 |
| Delta | -2.2% | +11.2% | +8.7% | -8.0% | regression |

Second A/B, streamed prompt with `prompt_bytes=111`, swept draft depths 2, 3,
and 4:

| Case | Effective block | TTFT ms | TPOT ms | Total ms | tok/s | Delta tok/s | Suffix accept | Avg verify ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 1 | 65.2 | 11.8 | 430.7 | 74.301 | 0.0% | n/a | n/a |
| MTP draft=2 | 3 | 66.6 | 11.0 | 409.1 | 78.250 | +5.3% | 0.607 | 24.18 |
| MTP draft=3 | 4 | 63.2 | 11.8 | 427.6 | 74.829 | +0.7% | 0.462 | 27.81 |
| MTP draft=4 | 5 | 64.8 | 12.3 | 444.9 | 71.927 | -3.2% | 0.455 | 34.24 |

Depth 3 and 4 explicitly exceed the draft checkpoint's declared `block_size=3`;
the runtime logs that as an experiment. They are not default candidates.

## KV Contract

- The MTP state owns no persistent draft KV cache.
- The draft layer reads target full-attention KV pair 0 up to
  `target_cache_len` and appends the current token's projected K/V only inside
  the transient attention graph. Those draft K/V tensors are never committed.
- RoPE uses `target_cache_len - 1` for the frozen-KV draft step.
- Target verify is the only path that mutates target KV/GDR. It verifies the
  full block `[current, draft...]`, computes the matched draft prefix, and
  returns the verifier next token.
- Accepted input count is `matched_prefix + 1`; `target_cache_len` advances
  only by that count.
- Full-attention KV tails beyond `target_cache_len` are ignored and overwritten
  by the next target verify. GDR state cannot be truncated by length alone, so
  the verifier runs in tape mode; partial accept restores the GDR snapshot and
  replays only the accepted prefix.
- The next MTP recurrent seed is the target verifier final hidden row
  `accepted_inputs - 1`, captured post-final-RMSNorm and pre-`lm_head`.

## Problems

The path is functionally real, but the performance result is prompt- and
acceptance-sensitive:

- Default draft=2 regressed one prompt by 8.0% tok/s because suffix acceptance
  was only 0.407.
- Draft=2 won a second prompt by 5.3% tok/s when suffix acceptance reached
  0.607.
- Draft=3 and draft=4 reduced verifier calls, but lower suffix acceptance and
  longer verify blocks erased or reversed the gain.
- MTP currently routes scalar per request and disables the standard Qwen3.6
  packed/double-buffer decode path for that row. That overhead matters when
  acceptance is modest.

## Learnings

- Keep Metal MTP behind explicit `--mtp-draft-model`; do not auto-enable it
  for Qwen3.6 until a multi-prompt GuideLLM run shows a stable win.
- For this split checkpoint, draft=2 is the only reasonable experimental
  default because it matches `block_size=3`. Draft=3/4 are useful probes, not
  production settings.
- MTP performance must report acceptance alongside TTFT/TPOT/tok-s. A green
  smoke or a single prompt win is not enough evidence for a default flip.

## Delta vs baseline

Baseline is the same-binary standard Metal decode in this entry. Best observed
case was draft=2 on the streamed prompt: `+5.3%` e2e tok/s and `-6.8%` TPOT.
Worst matched case was draft=2 on the first prompt: `-8.0%` e2e tok/s. Default
flip remains killed.
