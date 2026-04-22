# Bench Stub — Qwen3.6 DFlash Prefix Verify Session Reuse

## Context

The prior local optimization switched single-row Qwen3.6 DFlash from
full-block verify to prefix verify, which lifted the Metal lane from
`29.06 tok/s` to `48.24 tok/s` on the canonical local long-generation bench.

That prefix verifier was still calling `cpp_model.step()` per accepted target
step, so KV/GDR state crossed the Rust/C++ FFI boundary on every verified
token. This tranche reuses the existing compiled-session API:
`begin_session() -> step_session() -> end_session()`.

## What Worked

- `infer/src/backend/metal/dflash.rs::qwen35_dflash_speculative_block` now
  runs single-row prefix verify inside one compiled C++ session instead of
  bouncing KV/GDR state across FFI on each accepted step.
- The batched verifier is unchanged and still uses the sampled packed-block
  verify path.
- Local serial step-driver benchmark on `Apple M4 Pro`, target
  `mlx-community/Qwen3.6-35B-A3B-4bit`, draft
  `z-lab/Qwen3.6-35B-A3B-DFlash`, `prompt_tokens=20`,
  `generation_tokens=1024`, `warmup=1`, `runs=3`:

| mode | verify strategy | gen tok/s | repo e2e tok/s | TTFT ms |
|---|---|---:|---:|---:|
| baseline | n/a | 81.96 | 81.52 | 68.21 |
| DFlash before | prefix verify + per-step FFI | 48.24 | 48.07 | 72.64 |
| DFlash now | prefix verify + session reuse | 52.78 | 52.60 | 65.83 |

- Delta vs the immediately previous prefix-verify result:
  - `generation_tps`: `+9.4%`
  - `repo_e2e_tps`: `+9.4%`
  - `TTFT`: `72.64 ms -> 65.83 ms`
- Delta vs the original full-block verify result on the same workload:
  - `generation_tps`: `+81.6%` (`29.06 -> 52.78`)
  - `repo_e2e_tps`: `+81.4%` (`29.00 -> 52.60`)
- Local smoke exercising the single-row runtime with speculative decode:
  - `cargo run -p infer --release --no-default-features --features metal,no-cuda --bin metal_request -- --model /path/to/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46 --prompt hi --raw-prompt --max-new-tokens 2 --dflash-draft-model z-lab/Qwen3.6-35B-A3B-DFlash`
  - Result: `exit 0`, `Output tokens: 2`, `Gen TPS: 46.4 tok/s`
- Validation command:
  - `cargo test -p infer --release --no-default-features --features metal,no-cuda --bin metal_bench tests::baseline_compare_default_draft_qwen36_a3b -- --exact`
  - Result: `1 passed`

## Rule

Status: `pending-remote`

Once single-row DFlash on Metal switches to prefix verify, session reuse is the
next obvious optimization: keep accepted-step state resident in the compiled C++
model and only materialize KV/GDR back to Rust once per speculative block.
