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

Long-output chat follow-up, 5 streamed `/v1/chat/completions` cases
(`essay_zh`, `rust_code`, `debug_code`, `qa_kv`, `plan_ops`),
`temperature=0`, `max_tokens=192`. All cases hit `finish=length`.

Raw artifacts:
`bench-output/2026-06-02-metal-mtp-long-cases/`.

| Case | TTFT ms | TPOT ms | Total ms | chars/s | MTP acceptance |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline avg | 190.2 | 11.8 | 2443.0 | 333.0 | n/a |
| MTP draft=2 avg | 153.8 | 11.3 | 2302.7 | 355.2 | 465 blocks, avg accepted inputs 2.09/3, suffix accept 0.546 |
| MTP draft=3 avg | 158.9 | 13.4 | 2718.0 | 301.4 | 433 blocks, avg accepted inputs 2.25/4, suffix accept 0.416 |
| MTP draft=4 avg | 158.2 | 16.0 | 3221.8 | 254.0 | 428 blocks, avg accepted inputs 2.28/5, suffix accept 0.319 |

Per-case total latency deltas vs baseline:

| Case | MTP draft=2 | MTP draft=3 | MTP draft=4 |
| --- | ---: | ---: | ---: |
| essay_zh | +2.5% | +11.5% | +33.9% |
| rust_code | -11.4% | -2.6% | +14.1% |
| debug_code | -9.0% | +12.3% | +30.3% |
| qa_kv | +1.4% | +33.0% | +39.5% |
| plan_ops | -11.6% | +2.6% | +42.4% |

Long-output quality/parity follow-up, 512-token chat requests. The first pass
used the same prompts and showed Qwen reasoning-mode output consuming the full
budget, so a second `/no_think` pass was run for user-visible output. Raw
artifacts:
`bench-output/2026-06-02-metal-mtp-quality512/` and
`bench-output/2026-06-02-metal-mtp-no-think512/`.

Determinism check on one `debug_code` prompt, `max_tokens=192`:

| Path | Repeat 1 sha | Repeat 2 sha | Same path stable |
| --- | --- | --- | --- |
| baseline | `4fd10a8b0ecf247b` | `4fd10a8b0ecf247b` | yes |
| MTP draft=2 | `a9ebd7ccd4e27298` | `a9ebd7ccd4e27298` | yes |

The two paths were stable independently but not byte-identical to each other.
The diff on this prompt was small, but exact greedy target parity is not
licensed by this evidence.

`/no_think`, 512-token chat pass:

| Case | Baseline total ms / TPOT ms | MTP draft=2 total ms / TPOT ms | Delta total | Delta TPOT |
| --- | ---: | ---: | ---: | ---: |
| essay_zh | 6211.8 / 11.8 | 6427.4 / 12.2 | +3.5% | +3.6% |
| rust_code | 6329.9 / 11.9 | 5147.1 / 9.7 | -18.7% | -18.6% |
| qa_kv | 6294.8 / 11.9 | 6815.2 / 13.0 | +8.3% | +9.3% |

Quality notes: code output was coherent and similar across baseline/MTP2, but
both were still truncated before completing the full test block. Essay and KV
Q&A continued to emit `<think>` planning despite `/no_think`, so those quality
samples are useful for latency and degeneration checks, not final-answer
quality scoring.

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

The serving path is real, but the performance result is prompt- and
acceptance-sensitive, and exact greedy parity is not yet licensed:

- Default draft=2 regressed one prompt by 8.0% tok/s because suffix acceptance
  was only 0.407.
- Draft=2 won a second prompt by 5.3% tok/s when suffix acceptance reached
  0.607.
- In the 5-case 192-token chat sweep, draft=2 improved average total latency by
  5.7%, but it regressed essay and KV Q&A while improving code/ops prompts.
- Draft=3 and draft=4 reduced verifier calls, but lower suffix acceptance and
  longer verify blocks erased or reversed the gain.
- MTP currently routes scalar per request and disables the standard Qwen3.6
  packed/double-buffer decode path for that row. That overhead matters when
  acceptance is modest.
- Greedy output parity needs a token-level gate before any default flip:
  baseline and MTP2 repeated deterministically on the same prompt, but their
  output hashes differed.

## Learnings

- Keep Metal MTP behind explicit `--mtp-draft-model`; do not auto-enable it
  for Qwen3.6 until a multi-prompt GuideLLM run shows a stable win.
- For this split checkpoint, draft=2 is the only reasonable experimental
  default because it matches `block_size=3`. Draft=3/4 are useful probes, not
  production settings.
- MTP performance must report acceptance alongside TTFT/TPOT/tok-s. A green
  smoke or a single prompt win is not enough evidence for a default flip.
- Before optimizing the path further, add a strict greedy parity harness that
  compares baseline target tokens vs MTP target-verified tokens under
  `temperature=0`, then debug any divergence at token position rather than
  judging by text snippets.

## Delta vs baseline

Baseline is the same-binary standard Metal decode in this entry. In the
short-output tok/s sweep, draft=2 ranged from `+5.3%` e2e tok/s and `-6.8%`
TPOT to `-8.0%` e2e tok/s. In long-output chat, draft=2 ranged from `-18.7%`
total latency on a code case to `+8.3%` on a KV Q&A case in the `/no_think`
pass. Default flip remains killed, and exact-parity verification is now a
prerequisite for treating the path as more than an opt-in experiment.
