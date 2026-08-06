# DSpark draft trainer — serve alignment and chunk-invariant objective, 2026-08-06

> Status: pending-remote

Scope note: `crates/spec-train/`, `crates/autograd/` and `crates/cli/` sit
outside the bench-gate's file list. The entry exists anyway because the tranche
contains two default flips — `--max-len` 4096 -> 20480 and the training-time
attention geometry — and a default flip is in scope wherever it lives.

## Goal

Draft acceptance rate of a trained DSpark draft under
`arle serve --spec-type dspark` on ThinkingCap-27B-FP8, H20. Secondary: the
training loss must not depend on `--blocks-per-backward`, which is a VRAM knob.

## Hypothesis

Two confirmed defects made the trained draft mismatch the serve.

1. **The context window did not slide per row.** At serve
   (`crates/infer-cuda/src/qwen35/dspark.rs:887,913-921`;
   `crates/infer-cuda/src/executor/qwen35.rs:1893-1897,1948,2183`) the anchor is
   `last_token` at absolute position `start = kv_seq_len`, unforwarded when the
   draft runs — its tap is appended only after the verify — so the ring holds
   `[ctx_base, start)`, strictly below the anchor. Row `t` sits at query RoPE
   `start + t` with low bound `(start + t) - (sliding_window - 1)`: the span
   narrows by one key per row. Training gave every row the same
   `[anchor - W, anchor)`, so row `t` saw `1 + t` keys the serve never supplies.
   The upper bound was already right; a first fix moved it to `anchor + 1` on a
   misread of `last_token`, which put `taps[anchor]` in reach — the residual row
   0's own distillation target is projected from — and was reverted.
2. **`blocks_per_backward` changed the objective.** `loss.rs` recomputed
   `denom` over the current chunk while `trainer.rs` scaled every chunk by
   `1/(batch·chunks)`. A row's effective weight became `w_r/(batch·C·W_c)`
   instead of `w_r/(batch·W)`, so it depended on which chunk it landed in.
   Ragged tails and eval-zeroed blocks make `W_c` unequal at the shipped
   defaults.

Predicted effect of the fixes: higher measured acceptance at fixed training
budget, and a loss independent of `--blocks-per-backward`. Magnitude of the
acceptance change: `pending-remote`.

## Parameters

Training, one arm:

```bash
arle train spec-draft \
  --model-path /host/Qwen3.6-27B-FP8 \
  --draft /host/Qwen3.6-27B-DFlash \
  --data /tmp/spec-smoke/train.jsonl \
  --out /tmp/spec-smoke/draft-trained \
  --steps 200 --batch 8 --max-len 20480 \
  --num-anchors 512 --blocks-per-backward 32 \
  --trunk-mem-fraction 0.45 --seed 42 --log-every 1
```

Acceptance measurement, both arms, same serve flags apart from
`--mtp-draft-model`:

```bash
python3 scripts/gen_bench_prompts.py bench-agent-32k-64.jsonl 64 32768 256
python3 scripts/bench_throughput.py \
  --url http://127.0.0.1:8000 \
  --model thinkingcap-27b-fp8 \
  --prompts-jsonl bench-agent-32k-64.jsonl \
  --concurrency-grid 1,8 \
  --requests-per-concurrency 16 \
  --max-tokens 256 --seed 20260416 --timeout-seconds 900 \
  --output bench-output/<label>/bench
curl -s http://127.0.0.1:8000/v1/stats | python3 -m json.tool
```

- Baseline: `/host/Qwen3.6-27B-DFlash` as shipped, serve defaults on.
- Treatment: `/tmp/spec-smoke/draft-trained` from the command above.
- Prompt tokens: `pending-remote`
- Completion tokens: `pending-remote`
- Trials: `pending-remote`

## Environment

- Commit: `57eedcd75`.
- Host / GPU: 8xH20, `pending-remote` (driver, CUDA).
- Model / dtype: ThinkingCap-27B-FP8 trunk, bf16 draft checkpoint.
- Draft: 5 layers, hidden 5120, 32 heads, head_dim 128, block_size 7,
  `target_layer_ids` from the checkpoint config.
- TP / slots / KV: TP=1, `--trunk-mem-fraction 0.45` for the training job.

## Results

### Local, measured

| Gate | Command | Result |
|---|---|---|
| spec-train unit gates | `cargo test -p spec-train --lib` | 35 passed |
| autograd unit gates | `cargo test -p autograd --lib` | 61 passed |
| Workspace | `cargo test --workspace` | 120 result groups, 0 failures |
| Lints | `cargo clippy -p spec-train -p autograd --all-targets -- -D warnings` | clean |
| Mac CUDA typecheck | `cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib` | clean |

### Pod, measured

| Gate | Command | Result |
|---|---|---|
| CUDA/CPU parity, all autograd device ops | `pod_gpu_tests.sh run parity1 0 -- -p autograd --release --features cuda --test test_cuda_lazy_ops` | 43 passed, 0 failed, 28.6 s |
| Target-logits oracle, 16 positions of a 44-token sample | `arle train spec-draft … --probe` | argmax 1.000, top-64 overlap 0.993, mean abs delta log p(next) 0.0120 |
| Tap distinctness, 5 taps at layers 1/16/31/46/61 | same run | L2 1.20e2 -> 1.92e3 monotone in depth; max off-diagonal cosine 0.899 (layers 1-16), min 0.370 (layers 1-61); 0.000% non-finite |

The three cases the draft forward needed — `cuda_rank3_matmul_and_backward_match_cpu_on_draft_attention`, `cuda_concat_axis0_axis1_and_slice_backward_match_cpu`, `cuda_rank5_broadcast_expand_and_backward_match_cpu` — pass against their CPU references. This is the first CUDA execution of any of them.

Gates that now fail if the defect returns:

| Defect | Gate |
|---|---|
| Row RoPE position and context span | `spec_train::mask::tests::every_row_reaches_exactly_what_the_serve_gives_it` |
| Anchor tap reachable from the context | `spec_train::mask::tests::the_anchor_tap_is_out_of_reach`, `spec_train::backbone::tests::a_block_is_blind_from_its_anchor_onward` |
| Distillation target index | `spec_train::block::tests::only_the_anchor_row_carries_a_real_token` |
| Per-chunk denominator | `spec_train::loss::tests::splitting_the_rows_leaves_the_loss_unchanged`, `spec_train::trainer::tests::chunking_does_not_change_the_gradient` |
| Composed-loss backward | `spec_train::backbone::tests::taped_gradients_match_finite_differences` |
| `max_len` cutting the supervised turn | `spec_train::data::tests::the_loss_mask_decodes_to_exactly_the_generated_turn` |

### Remote, unmeasured

| Quantity | Arm | Value |
|---|---|---|
| `spec_decode.accept_rate`, c=1 | baseline | pending-remote |
| `spec_decode.accept_rate`, c=1 | treatment | pending-remote |
| Output tok/s, c=1 / c=8 | both | pending-remote |
| TTFT p50/p99, ITL p50/p99 | both | pending-remote |
| Training loss at step 0 / step 200 | treatment | pending-remote |
| Peak training VRAM vs `peak_activation_bytes` | treatment | pending-remote |
| `s/step` at the reference recipe | treatment | pending-remote |

Raw artifacts: pending-remote.

## Problems

Four gates remain open, all needing the pod:

1. Serve-side checkpoint load — the round-trip gate uses `spec_train`'s own
   reader, not `load_dspark_head`.
2. Measured acceptance of a trained draft.
3. `peak_activation_bytes` against real VRAM at training shape; the local
   measurement is at hidden 8, vocab 24.
4. The `ctx_base` clamp. The serve floors the per-row window at the ring base
   (`dspark.rs:915`); training floors at 0. Once the ring has wrapped, the serve
   sees fewer keys than training fitted on. Not modeled, not measured.

The probe closed the target-logits oracle and tap sanity at 44 tokens. The
trunk's long-context behaviour — sliding window, ring wrap — is the regime that
breaks, and the probe has not run there. Item 4 is the same regime.

## Learnings

pending-remote. Local correctness gates pass; no acceptance number exists yet,
so the tranche is unlicensed for a default flip on the serve side. The next
wall is the pod run in the Problems list, item 5.

## Update 2026-08-06, evening — norm fix, batch denominator, 400-step run

Commits `cf31adb0c` (final-norm kernel + one-`fc`-backward-per-sample),
`eefe83719`, `b13aa15d1`, `60f24b113`, `75b04180b` (pooled tau/accept@k),
`9add3ea1b` (batch-wide denominator). Build `normfix` at `9e4522122`, H20,
GPU 0/1, ThinkingCap-27B-FP8 trunk, `/tmp/spec-corpus/timing2k.jsonl`
(1913 samples).

`forward_training_taps` was the one remaining `rms_norm_offset` call on
`self.norm` after the `e4629d69a` sweep — every distillation target logit was
computed under the wrong norm kernel (~2.04x per-dim error). The earlier probe
passed because the oracle used the same wrong path: agreement between a
co-evolving oracle and the thing it checks is not correctness.

Probe, post-fix (`probe3`, 369-token sample, 16 positions): argmax 0.938,
top-64 overlap 0.989, mean |delta log p(next)| 0.0155. The earlier
argmax 1.000 / 0.0120 row above predates the fix and is superseded.

Training (`tv400`): 400 steps, batch 8, 512 anchors, max-len 4096,
lr 7.5e-5 cosine (sqrt-scaled from the reference's 6e-4 at batch 512),
per-sample denominator (predates `9add3ea1b`).

| Quantity | step 0 | step 399 |
|---|---|---|
| loss | 3.324 | 2.247 |
| CE | 13.31 | 7.12 |
| TV | 1.133 | 0.942 |
| accept | 0.433 | 0.529 |
| tau | 1.77 | 2.15 |
| confidence_bias | -0.240 | -0.010 |
| confidence_abs_error | 0.270 | 0.098 |
| gnorm | 58.8 | 0.77 |

TV falls monotonically net of batch noise; the confidence head converges to
unbiased. 3200 samples seen vs the reference's ~13.5M — the run licenses the
mechanism, not a serve-side acceptance claim. `s/step` 9.5-29.3 tracks the
per-step chunk count (39-95); ~0.28 s/chunk. Peak activation printed 8.9 GiB:
the estimator's dominant terms are per-chunk scores and logits, so hoisting
`fc` out of the chunk loop moves time, not peak.

Still pending-remote: serve-side accept_rate A/B of a trained draft, and the
full-corpus (57k rows) run on the batch-wide denominator objective.

## Update 2026-08-07 — first serve-side acceptance A/B

Build `b4fix`/`b4fix3` (head `21dec1fa5`/`cfe43a3ce`), H20, same serve flags
both arms: `--spec-type dspark --dspark-confidence-threshold 0` (goodput gate
off, raw acceptance), 4 chat prompts x 256 tokens.

| Arm | chains | drafted | accepted | accept_rate |
|---|---|---|---|---|
| `/host/Qwen3.6-27B-DFlash` (shipped warm-start) | 264 | 3960 | 760 | 19.2% |
| `/tmp/spec-smoke/tv400` (from-scratch, 3200 samples) | 963 | 6741 | 57 | 0.85% |

The pipeline round-trips: the trained checkpoint loads through
`load_dspark_head`, drafts, and lands verified tokens — the serve-alignment
gates closed the geometry gap this entry opened with. The rate itself measures
training scale, not the mechanism: 3200 samples vs the reference's ~13.5M.
The from-scratch arm is not a treatment claim; the decisive run is the 57k
full corpus (batch-wide denominator objective, `9add3ea1b`) against the
DFlash warm start.

`b4s100`, 100 steps on the batch-wide denominator: same loss scale
(2.4-2.6), no NaN, gnorm 0.7-2.2 — the objective change is numerically safe
at training shape.

Side fix while measuring: one pod-side `kernel_artifacts.sh export` had made
every later build/run fail `source changed since receipt` — the tarball lands
in the repo root by design but was not gitignored, and `source_digest` counts
untracked-unignored files (`cfe43a3ce`).
