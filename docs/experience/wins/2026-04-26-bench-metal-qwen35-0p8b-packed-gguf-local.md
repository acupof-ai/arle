# Metal Qwen3.5 0.8B Packed GGUF A/B

## Goal

- Type: regression.
- Compare the new true packed GGUF Metal path against the local BF16
  safetensors Qwen3.5-0.8B baseline on the same step-driver workload.

## Hypothesis

- Packed GGUF should reduce resident memory substantially and improve decode
  throughput, but prompt/TTFT may trail BF16 until the GGUF path can safely use
  the compiled C++ recurrent path again.

## Command

```bash
./target/release/metal_bench \
  --model models/Qwen3.5-0.8B \
  --prompt-tokens 32 \
  --generation-tokens 32 \
  --warmup 0 \
  --runs 1 \
  --use-step-driver \
  --ignore-eos \
  --json

./target/release/metal_bench \
  --model models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf \
  --prompt-tokens 32 \
  --generation-tokens 32 \
  --warmup 0 \
  --runs 1 \
  --use-step-driver \
  --ignore-eos \
  --json
```

Invoked via: local `metal_bench` helper, not `scripts/bench_guidellm.sh`,
because this verifies the Metal step-driver path directly before HTTP serving.

## Environment

- Backend: Metal step-driver path
- Model: Qwen3.5-0.8B safetensors vs Qwen3.5-0.8B GGUF Q4_K_M
- Hardware: Apple M4 Pro, 20-core GPU, 48 GB unified memory
- OS / Metal: macOS 26.3.1 (25D771280a), Metal 4
- Commit: `ea6b3aa` + dirty working tree for this change
- Feature set: `cargo build --release -p infer --no-default-features --features metal,no-cuda`
- Non-default flags / env vars: none
- Server launch: n/a; local step-driver helper

## Results

Raw JSON:

```json
{"avg_tokens":32,"generation_tokens_requested":32,"generation_tps":{"mean":29.672282381631135,"p50":29.672282381631135,"p99":29.672282381631135},"load_ms":413.8585,"mode":"step-driver","model":"models/Qwen3.5-0.8B","peak_rss_mb":1602.6875,"prompt_tokens":32,"prompt_tokens_requested":32,"prompt_tps":{"mean":299.3910432959212,"p50":299.3910432959212,"p99":299.3910432959212},"quantization":"bf16","repo_e2e_tps":{"mean":26.9966747841337,"p50":26.9966747841337,"p99":26.9966747841337},"timed_runs":1,"total_time_ms":{"mean":1185.3311660000002,"p50":1185.3311660000002,"p99":1185.3311660000002},"ttft_ms":{"mean":106.883625,"p50":106.883625,"p99":106.883625},"warmup_runs":0}
{"avg_tokens":32,"generation_tokens_requested":32,"generation_tps":{"mean":37.50513678264615,"p50":37.50513678264615,"p99":37.50513678264615},"load_ms":4648.919333,"mode":"step-driver","model":"models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf","peak_rss_mb":896.46875,"prompt_tokens":32,"prompt_tokens_requested":32,"prompt_tps":{"mean":36.34881971730457,"p50":36.34881971730457,"p99":36.34881971730457},"quantization":"gguf-q4_k","repo_e2e_tps":{"mean":18.458964855861442,"p50":18.458964855861442,"p99":18.458964855861442},"timed_runs":1,"total_time_ms":{"mean":1733.575,"p50":1733.575,"p99":1733.575},"ttft_ms":{"mean":880.358709,"p50":880.358709,"p99":880.358709},"warmup_runs":0}
```

Headline:

| Metric | BF16 safetensors | Packed GGUF Q4_K_M | Delta |
|---|---:|---:|---:|
| Load time | 413.9 ms | 4648.9 ms | +1023.3% |
| Peak RSS | 1602.7 MB | 896.5 MB | -44.1% |
| Prompt throughput mean | 299.39 tok/s | 36.35 tok/s | -87.9% |
| Decode throughput mean | 29.67 tok/s | 37.51 tok/s | +26.4% |
| Repo E2E throughput mean | 27.00 tok/s | 18.46 tok/s | -31.6% |
| TTFT mean | 106.9 ms | 880.4 ms | +723.7% |
| Total time mean | 1185.3 ms | 1733.6 ms | +46.3% |

Correctness checks paired with this run:

| Check | Result |
|---|---|
| `compare_qwen35_0p8b_first_token_logits_safetensors_vs_gguf` | top-1 matched: token 11 `,` |
| `compare_qwen35_0p8b_gguf_incremental_decode_vs_full_replay` | passed 4 decode steps |

## Problems

- A discarded earlier packed run hit a Metal command buffer hang while the C++
  compiled GGUF path still used the custom GDR recurrent kernel. The shipped
  path keeps GGUF on Rust/MLX recurrent ops and still uses packed weights for
  embedding and linear projections.
- Restoring the non-GGUF C++ path changed the baseline materially: BF16 prompt
  throughput/TTFT is now much faster than the packed GGUF Rust/MLX fallback.
- This is a helper benchmark, not a canonical GuideLLM HTTP sweep.
- Working tree was dirty because this entry records the pre-commit validation
  run for the current change.

## Learnings

- Packing must be paired with correct kernel grid semantics: the first Metal
  packed matmul version launched only one threadgroup per 256 output rows,
  leaving most outputs uninitialized.
- Packed GGUF memory and decode wins are real on Metal once weights remain
  packed through load and forward; for this 32/32 run RSS dropped by 44.1% and
  decode throughput rose 26.4%.
- End-to-end latency still regresses against the restored BF16 C++ path because
  packed GGUF deliberately avoids the currently unsafe compiled recurrent route.
- Stateful custom recurrent kernels need decode-vs-replay tests, not just
  first-token logits. First token can pass while recurrent state is corrupt.

## Delta vs baseline

- Baseline: `models/Qwen3.5-0.8B` BF16 safetensors under the same command.
- Prior GGUF baseline:
  [`2026-04-26-bench-metal-qwen35-0p8b-gguf-vs-safetensors-local.md`](2026-04-26-bench-metal-qwen35-0p8b-gguf-vs-safetensors-local.md)
  recorded the old load-time BF16 GGUF path.

| Metric | BF16 baseline | Packed GGUF | Delta |
|---|---:|---:|---:|
| Decode throughput mean | 29.67 tok/s | 37.51 tok/s | +26.4% |
| Prompt throughput mean | 299.39 tok/s | 36.35 tok/s | -87.9% |
| Peak RSS | 1602.7 MB | 896.5 MB | -44.1% |
| TTFT mean | 106.9 ms | 880.4 ms | +723.7% |

## Artefacts

- Raw: inline JSON above
- CSV / HTML: n/a for local helper benchmark
- Service trace: n/a; no HTTP server

## Notes

- Code change since baseline: Metal GGUF Q8_0/Q4_K/Q5_K/Q6_K packed weight
  loading, packed matmul/embedding kernels, and packed recurrent decode guard.
- Suspected cause of prompt/TTFT regression: packed GGUF cannot use the C++
  compiled recurrent path until the custom GDR state update is fixed.
- Follow-ups: fix the custom GDR Metal kernel and C++ session state evaluation,
  then re-enable it behind the same decode-vs-replay gate.
