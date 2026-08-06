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

Six gates remain open, all needing the pod:

1. Target-logits oracle — `last_hidden @ lm_head` against
   `LoadedInferenceEngine::forward_token_logits` for the same ids.
2. Tap sanity — per-tap norm and pairwise distinctness of
   `forward_training_taps`.
3. Serve-side checkpoint load — the round-trip gate uses `spec_train`'s own
   reader, not `load_dspark_head`.
4. Measured acceptance of a trained draft.
5. `peak_activation_bytes` against real VRAM at training shape; the local
   measurement is at hidden 8, vocab 24.
6. The `ctx_base` clamp. The serve floors the per-row window at the ring base
   (`dspark.rs:915`); training floors at 0. Once the ring has wrapped, the serve
   sees fewer keys than training fitted on. Not modeled, not measured.

Items 1 and 2 are built and unrun: `arle train spec-draft --probe`.

## Learnings

pending-remote. Local correctness gates pass; no acceptance number exists yet,
so the tranche is unlicensed for a default flip on the serve side. The next
wall is the pod run in the Problems list, item 5.
