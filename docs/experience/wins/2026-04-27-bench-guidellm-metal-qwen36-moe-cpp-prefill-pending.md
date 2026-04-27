# Metal Qwen3.6 MoE C++ prefill routing — guidellm pending, 2026-04-27

## Goal

- Regression-check the Qwen3.6 / Qwen3.5-MoE Metal path after wiring MoE MLP
  weights into the C++ Qwen3.5 step model, so prompt prefill no longer drops
  to Rust scalar fallback.

## Hypothesis

- `CppQwen35Model::build` should succeed for `mlx-community/Qwen3.6-35B-A3B-4bit`.
  The scheduler step-driver should report `cpp_batch_prefill` for multi-token
  prompt chunks instead of `rust_scalar_prefill`.

## Command

Canonical guidellm command, pending:

```bash
scripts/bench_guidellm.sh metal-qwen36-moe-cpp-prefill \
  --target http://localhost:8000 \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --processor mlx-community/Qwen3.6-35B-A3B-4bit
```

Local routing smoke run:

```bash
cargo check -p infer --release --no-default-features --features metal,no-cuda
cargo build -p infer --release --no-default-features --features metal,no-cuda --bin metal_bench

AGENT_INFER_METAL_QWEN35_TRACE=1 ./target/release/metal_bench \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --use-step-driver \
  --prompt-tokens 8 \
  --generation-tokens 1 \
  --warmup 0 \
  --runs 1 \
  --json

AGENT_INFER_METAL_QWEN35_TRACE=1 ./target/release/metal_bench \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --dflash-draft-model z-lab/Qwen3.6-35B-A3B-DFlash \
  --prompt-tokens 8 \
  --generation-tokens 1 \
  --warmup 0 \
  --runs 1 \
  --json
```

## Environment

- **Backend:** Metal
- **Model:** `mlx-community/Qwen3.6-35B-A3B-4bit`
- **Draft model:** `z-lab/Qwen3.6-35B-A3B-DFlash`
- **Hardware:** Apple M4 Pro, 48 GB unified memory, Metal 4
- **OS:** macOS 26.3.1 (25D771280a)
- **Commit:** `f69187b` plus local hot-path diff
- **Feature set:** `--release --no-default-features --features metal,no-cuda`
- **Non-default flags / env vars:** `AGENT_INFER_METAL_QWEN35_TRACE=1` for smoke
- **Server launch:** pending for canonical guidellm

## Results

Canonical guidellm sweep: **pending**.

Local routing smoke:

```text
metal_trace[qwen35_prefill_tokens:start]: mode=cpp_batch_prefill tokens=8 terminal=true cache_len=0
metal_trace[qwen35_prefill_tokens:done]: mode=cpp_batch_prefill tokens=8 terminal=true cache_len=0=>8 emitted=true elapsed_ms=5340.6
metal_trace[qwen35_prefill_chunk]: phase=Prefill->Finished budget=512 prompt=0..8 processed=8 emitted=true elapsed_ms=5340.6
```

| shape | mode | TTFT ms | prompt tok/s | total ms | peak RSS MB |
|---|---|---:|---:|---:|---:|
| 8 / 1 | step-driver | 5340.650 | 1.498 | 5340.650 | 11666.9 |
| 8 / 1 | DFlash smoke | 4700.572 | 1.702 | 4700.574 | 10975.4 |

## Problems

- Full `guidellm` 4096-in / 256-out sweep was not run in this turn. The local
  runs are routing diagnostics only and must not be interpreted as Qwen3.6
  DFlash performance evidence.

## Learnings

- The Qwen3.6 fallback was structural: the Rust builder rejected
  `MlpKind::Moe(_)` while the C++ bridge already had MoE storage and forward
  support. Wire capability that already exists before adding fallback
  mitigations.
- GDR separate attention projections are independent of the MLP kind. MoE
  layers still need `set_separate_proj_v2`; only the dense gate/up MLP pair is
  optional.

## Delta vs baseline

- **Baseline:** [`2026-04-27-bench-metal-qwen36-a3b-dflash-quick-check.md`](2026-04-27-bench-metal-qwen36-a3b-dflash-quick-check.md)
- **Routing delta:** previous builder returned `cpp_model = None` for Qwen3.6
  MoE and therefore used Rust fallback prefill. Current step-driver trace shows
  `mode=cpp_batch_prefill`.
- **Performance delta:** pending canonical guidellm run.

## Artefacts

- Raw guidellm JSON/CSV/HTML: pending
- Local smoke output: terminal log above

## Notes

- What changed in code since baseline: `CppQwen35Model::build` now registers
  MoE MLP weights with `qwen35_compiled_set_last_moe_mlp`; C++ layer push
  helpers accept MoE layers without dense MLP placeholders.
- Follow-up: replace this pending entry with a completed guidellm sweep after
  a long-context Metal run.
