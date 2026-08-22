# `--spec-type auto` as the default — CUDA, 2026-08-21

> Status: Shipped, `4863971eb`. Qwen3.8-27B-NVFP4, 1xH20, FP8 KV.

## Context

Speculative decode existed but nobody turned it on: `--spec-type` defaulted to
`none`, and `auto` was declared in the enum while bailing at both lowering sites
(`cli/src/serve.rs`, `infer-api/src/serve.rs`). An optimisation behind a flag no
default sets is not an optimisation.

## Result

One binary, three arms, 32K agent chain, 32 requests per point, 0 errors, 32/32
complete everywhere (`/host/arle-runs/specab-20260821/`):

| c | ms per committed token | A none | B mtp d=2 | C mtp d=4 |
|---:|---|---:|---:|---:|
| 1 | | 20.50 | **11.94** | 12.19 |
| 1 | tok/decode-step | 1.00 | 1.89 | 2.03 |
| 1 | rows presented (M) | 1.0 | 2.97 | 4.95 |
| 1 | accept rate | — | 44.7% | 25.9% |

**d=2 is the setting.** d=4 buys 2.03 tok/step against 1.89 while the accept rate
falls 44.7% → 25.9%, and lands the same ms per committed token.

**Above c=1 the default is inert, not merely harmless.** The `/v1/stats` spec
chain-counter delta is exactly 0 for B and C at every c≥4 and `tok/decode-step`
matches the control to four significant digits (4.04 / 8.18 / 16.58 / 32.15), so
the three arms are literally the same code path — the MTP branch is pinned to a
single decode row at `executor/qwen35.rs:2369` because it drafts serially.

`ITL` is unusable for the spec arms: MTP emits several tokens per SSE event, so
C's c=1 `itl_p50` of 0.04 ms is a burst artifact. The decode SLO here is
`ms per committed token`, derived from `decode_forward_busy_micros` /
`decode_forward_steps` / `generated_tokens`.

## Correctness

Needle ladder 512 / 4096 / 16384 / 32768, ×3 same-config, `RAW=1
TEMPLATE=qwen3_nonthink`, FP8 KV, both arms on one binary: **`exact=3 partial=0
miss=0 DET` at every length**.

That alone would not have licensed anything. Speculation is output-preserving
under greedy, so identical text cannot distinguish "ran and was correct" from
"never ran" — and this engine demonstrably does disable speculation silently, as
the c≥4 rows above show. The arm is therefore proven engaged from the same
process: `arle_spec_chains_total` **0 → 42**, drafted 84, accepted 18 across the
ladder.

## Detection

`auto` routes on the checkpoint, not on the model name: `config.json` carrying
`mtp_num_hidden_layers > 0` (Qwen3.5 nests it under `text_config`) or
`num_nextn_predict_layers > 0` (DeepSeek-V4; GLM ships 0). A missing or
unparsable config does not speculate. Six shapes are pinned by a unit test,
because a false here is a silently disabled default.

`auto` lowers to `none` off CUDA, and the CUDA-only guard now fires on explicit
routes only, so the new default is silent on Metal and CPU.

## Rule

For an output-preserving optimisation, the correctness gate cannot also be the
engagement gate. Greedy speculative decode is designed to produce identical
text, so a passing needle ladder is consistent with the feature being off.
Read a counter from the same process, in the same run, before quoting the gate.
