# ARLE CUDA OPD Tokenizer JSONL Prompt Loading

## Goal

Replace hand-picked token-id-only OPD convergence runs with a real text prompt
surface:

```text
--prompts-file examples/opd/sample-prompts.jsonl
```

Each JSONL row has the form `{"text":"...","max_tokens":16}` and is tokenized
with the Qwen3 tokenizer adjacent to the ModelScope checkpoint.

Verdict: **licensed**. The JSONL path loads the real tokenizer, splits train vs
held-out prompts, and completes a 500-step real-checkpoint CUDA OPD run without
crash or NaN.

## Command

```bash
OUT=bench-output/2026-05-21-arle-cuda-opd-prompts-file-jsonl
mkdir -p "$OUT"
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.free,utilization.gpu --format=csv \
  > "$OUT/nvidia-smi-before.txt"

NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo run -p train --example opd_step_cuda_realckpt_train --release --features cuda -- \
  --prompts-file examples/opd/sample-prompts.jsonl --lr 1e-7 --steps 500 \
  --eval-steps 0,100,250,500 2>&1 | tee "$OUT/run.txt"

nvidia-smi --query-gpu=timestamp,name,memory.used,memory.free,utilization.gpu --format=csv \
  > "$OUT/nvidia-smi-after.txt"
```

## Environment

- GPU: NVIDIA GeForce RTX 4070 Ti SUPER, 16 GiB
- Feature set: `--features cuda`
- Model: `/home/ckl/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B`
- Tokenizer: `/home/ckl/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B/tokenizer.json`
- Prompt file: `examples/opd/sample-prompts.jsonl`
- Prompt rows: 20 total, 16 train, final 4 held out
- Prompt default max tokens: 16
- Truncated rows: 1
- Teacher: frozen Qwen3-0.6B checkpoint
- Student: same checkpoint, all trainable params perturbed by uniform
  `[-1e-3, 1e-3]`
- Optimizer: AdamW lr=`1e-7`, betas=(0.9, 0.999), eps=1e-8, wd=0
- Rollout: `rollout_len=8`

GPU memory snapshots before and after the process are in the artefact directory.

## Implementation

Added `crates/train/src/prompts.rs` with a reusable JSONL loader:

- resolves `tokenizer.json` next to the selected model directory;
- tokenizes each non-empty JSONL row with `add_special=false`;
- applies per-row `max_tokens`, falling back to `--prompt-max-tokens`;
- rejects empty text, empty tokenized prompts, and prompt files that cannot
  provide at least one train row plus held-out rows;
- splits the final 4 rows as held-out prompts.

The harness now supports:

```text
--prompts-file <jsonl>
--example-prompts-file <jsonl>
--prompt-max-tokens <usize>
```

`--prompt-set 8|32` remains available for matched-control built-in prompt runs.
The JSONL path truncates to the configured maximum but does not add pad tokens:
this harness processes one prompt at a time and has no attention-mask plumbing,
so padding would be semantically visible to the model.

## Results

Training wall-clock:

| Metric | Value |
|---|---:|
| total steps | 500 |
| total loop wall seconds | 126.308124 |
| mean OPD step seconds | 0.208117 |
| median OPD step seconds | 0.208654 |
| first sampled OPD loss | 2.112482e-5 |
| step 250 sampled OPD loss | 2.510164e-5 |
| final sampled OPD loss | 2.362544e-5 |

Eval trajectory:

| Step | Train exact % | Held-out exact % | Train KL | Held-out KL | Train NLL | Held-out NLL | Train top-3 % | Held-out top-3 % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 35.937500 | 65.625000 | 1.663877e-2 | 1.731548e-2 | 1.028643 | 1.179136 | 99.218750 | 98.437500 |
| 100 | 59.765625 | 65.625000 | 1.214844e-2 | 1.394402e-2 | 1.012971 | 1.169701 | 100.000000 | 98.437500 |
| 250 | 65.234375 | 62.500000 | 9.284572e-3 | 1.210662e-2 | 1.003534 | 1.165488 | 100.000000 | 98.437500 |
| 500 | 74.609375 | 70.312500 | 7.206479e-3 | 1.073014e-2 | 0.997864 | 1.163611 | 100.000000 | 98.437500 |

Derived deltas:

| Metric | Step 0 -> 500 |
|---|---:|
| train exact overlap | +38.671875 pp |
| held-out exact overlap | +4.687500 pp |
| train KL | -56.69% |
| held-out KL | -38.03% |
| train teacher NLL | -2.99% |
| held-out teacher NLL | -1.32% |

## Interpretation

The JSONL/tokenizer path is stable and keeps the same useful OPD behavior seen
with built-in token-id prompts: train KL drops strongly and held-out KL also
improves. Exact held-out overlap is noisier because this sample file has only
four held-out prompts, but the continuous held-out KL moves in the right
direction.

This closes the first DX gap from the OPD positioning note: users can now point
the real-checkpoint OPD harness at text prompts instead of manually writing
token-id arrays.

## Problems

- `cargo clippy --workspace --all-targets -- -D warnings` is not green because
  of unrelated existing lints in `infer/` (`deepseek_v4_manifest`,
  `tokenizer_fingerprint_radix_isolation`, Metal config / KV pool, and
  `server_engine`). The narrower train gate is clean.
- Padding is intentionally deferred. Adding pad tokens without an attention
  mask would change prompt semantics; padding should land with a batched prompt
  loader that threads masks through OPD eval/training.

## Verification

- `cargo fmt -p train`: passed.
- `cargo check -p train --example opd_step_cuda_realckpt_train --features cuda`:
  passed.
- `cargo test -p train prompts --release`: passed.
- `cargo test -p train --test test_opd_determinism --release`: passed.
- `cargo check --workspace`: passed.
- `cargo clippy -p train --all-targets -- -D warnings`: passed.
- 500-step CUDA JSONL run above: passed, no crash / no NaN.

## Artefacts

- `bench-output/2026-05-21-arle-cuda-opd-prompts-file-jsonl/run.txt`
- `bench-output/2026-05-21-arle-cuda-opd-prompts-file-jsonl/nvidia-smi-before.txt`
- `bench-output/2026-05-21-arle-cuda-opd-prompts-file-jsonl/nvidia-smi-after.txt`
