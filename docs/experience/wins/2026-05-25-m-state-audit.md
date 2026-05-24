# T8 M-State Audit

## Context

After the OPD CLI, T4a, T5a, G6, T7, and T11 commits, the working tree still
had old M-state and untracked artifacts. T8 asked for a file-by-file verdict:
ship standalone, merge into a landed feature, revert, or ignore output noise.

## What Worked

The tracked Rust diffs were not feature work. They were import reorders and
line wraps from a formatter pass, with no semantic delta. The tracked
`bench-output/.../serve.log` change was appended serve output. I reverted those
paths explicitly because they did not carry a commit-ready OPD/runtime axis.

The untracked BF16 frozen-base research note was commit-ready: it is referenced
by committed backlog/wins/errors entries, so leaving it untracked made those
links stale. It landed as `9bd23ec`.

`runs/` contains OPD run checkpoints and local output, including active P5-era
artifacts. I did not delete it; this patch only ignores future local run
outputs.

## Verdicts

| Path | Verdict | Reason |
| --- | --- | --- |
| `bench-output/2026-05-22-h3-max-seq-len-4096-08b/serve.log` | revert | tracked output append; `bench-output/` is already ignored for new files |
| `crates/autograd/tests/test_cuda_lazy_ops.rs` | revert | import reorder only |
| `crates/train/examples/opd_step_cuda_convergence_bench.rs` | revert | import reorder and line wrap only |
| `crates/train/examples/opd_step_cuda_realckpt_diag.rs` | revert | import reorder only |
| `crates/train/examples/opd_step_cuda_realckpt_profile.rs` | revert | import reorder and line wrap only |
| `crates/train/src/qwen35_checkpoint.rs` | revert | import reorder and assert formatting only |
| `crates/train/src/teacher_infer.rs` | revert | import reorder and test signature formatting only |
| `infer/examples/gptqmodel_w4_gemv_parity.rs` | revert | import reorder only |
| `infer/examples/qwen35_dense_module_dump.rs` | revert | import reorder only |
| `infer/examples/qwen35_linear_attn_parity.rs` | revert | import reorder only |
| `infer/src/backend/cuda/bootstrap.rs` | revert | import reorder only |
| `infer/src/model/qwen35/lora.rs` | revert | import reorder only |
| `infer/src/model/qwen35/weights.rs` | revert | import reorder only |
| `docs/research/2026-05-24-bf16-frozen-base-impl-path.md` | ship | committed docs referenced it; docs-only research note |
| `runs/` | ignore | local training/checkpoint output; do not remove active or historical run artifacts |
| `examples/opd/sft-anchor-mmlu-gsm8k.jsonl` | hard stop | data source/license is not documented; do not ship or delete without ckl decision |

## Verification

```bash
git diff --check --no-index /dev/null docs/research/2026-05-24-bf16-frozen-base-impl-path.md
git diff --cached --check
git status --short
```

No cargo test was run: this was a revert/ignore/docs audit, with no code
behavior shipped.

## Rule

Do not let formatter-only M-state drift through multiple runtime tasks. Revert
format noise explicitly, commit missing docs when other committed docs link to
them, and treat untracked dataset files as license-sensitive until source and
redistribution rights are recorded.
