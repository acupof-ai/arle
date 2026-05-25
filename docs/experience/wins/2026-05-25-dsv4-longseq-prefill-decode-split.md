# DSv4 long-sequence prefill/decode split

## Goal

- Diagnosis/optimization: fix DSv4 6K-token HTTP serving so long prefill does
  not OOM, and decode does not fall back to full-context recompute.

## Hypothesis

- The correct path is split by phase: batched transformer prefill for prompt
  ingestion, incremental KV only for decode. Applying incremental KV to the
  whole prefill should be rejected because it regresses TTFT.

## Command

Identifiers are intentionally omitted. The run used the DSv4 local model path
inside the 8-GPU Kubernetes container.

```bash
INFER_CUDA_DEVICES=0,1,2,3,4,5,6,7 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
RUST_LOG=info NCCL_DEBUG=WARN \
./target/release/infer \
  --port 18084 \
  --num-slots 1 \
  --max-seq-len 8192
```

Profile capture:

```bash
nsys profile --trace cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --export=sqlite \
  --sample=none \
  --kill=none \
  --force-overwrite=true \
  --output trace \
  ./target/release/infer --port 18084 --num-slots 1 --max-seq-len 8192
```

## Environment

- Backend: CUDA, 8 ranks
- Hardware: 8x NVIDIA H20, 102 GB HBM each
- CUDA: 12.9 build path, Nsight Systems 2026.2.1
- Feature set: `cargo build --release -p infer --features cuda,nccl --bin infer`
- Workload: one cold 6171-token chat request, `max_tokens=8`; nsys profile used
  the same prompt with `max_tokens=1`.

## Results

| run | prompt tokens | max new | TTFT / prefill | total | decode behavior | status |
|---|---:|---:|---:|---:|---|---|
| baseline nsys before fix | 6171 | 64 | 35.78 s TTFT | 43.03 s | ~0.23 s/token after first decode | ok |
| giant KV pool + default incremental | 6171 | 8 | 12.38 s prefill | n/a | n/a | OOM |
| incremental prefill on | 6171 | 8 | 72.27 s TTFT | 74.49 s | ~0.23 s/token | regression |
| incremental off | 6171 | 8 | 31.32 s TTFT | 249.59 s | ~31 s/token | decode regression |
| final split path | 6171 | 8 | 31.42 s TTFT | 32.62 s | no full-context decode recompute | ok |

Final nsys:

| NVTX range | instances | rank-aggregate total | per-rank avg |
|---|---:|---:|---:|
| `step_prefill_kernel_launch` | 8 | 251.64 s | 31.46 s |

Top final prefill kernels:

| kernel | rank-aggregate time | share |
|---|---:|---:|
| `dsv4_fp8_gemv_batch_tiled_kernel` | 73.88 s | 29.6% |
| `dsv4_csa_select_kernel` | 56.22 s | 22.5% |
| `dsv4_hybrid_attention_kernel` | 41.50 s | 16.6% |
| `dsv4_fp4_gemv_batch_tiled_kernel` | 40.84 s | 16.4% |
| NCCL all-reduce | 27.40 s | 11.0% |

CUDA API still reports `cuMemcpyDtoHAsync_v2` as the largest API bucket
(`239.56 s` rank-aggregate), but GPU memory-copy time is tiny. It is a
synchronization attribution point, not the data-movement root cause.

## Problems

- `mem_fraction_static=0.85` originally built a 62.6 GB TokenKVPool per rank
  for a single explicit 8192-token slot. That starved prefill scratch and made
  the incremental path OOM.
- Enabling incremental KV for the entire prefill avoided decode recompute but
  changed the prefill path from ~31 s to ~72 s.
- Disabling incremental KV restored prefill speed but made every decode step
  recompute the full context (~31 s/token).

## Learnings

- DSv4 needs a phase split: batched prefill, incremental decode. The
  incremental path is not a prefill optimization.
- For explicit `max_seq_len` DSv4 single-slot validation, cap the paged KV pool
  to the explicit envelope and leave activation/scratch headroom.
- CUDA API D2H attribution inside prefill must be cross-checked against kernel
  and memory-copy summaries before assigning root cause.

## Delta vs baseline

| metric | baseline | final | delta |
|---|---:|---:|---:|
| `step_prefill_kernel_launch` rank-aggregate | 286.11 s | 251.64 s | -12.0% |
| per-rank prefill avg | 35.76 s | 31.46 s | -12.0% |
| long request TTFT | 35.78 s | 31.42 s | -12.2% |
| decode after first token | ~0.23 s/token | no full-context recompute | fixed |

## Artefacts

- Final HTTP run: `dsv4-final-split-prefill-decode-default-max8-8rank-20260525-164336`
- Final nsys run: `dsv4-final-split-prefill-decode-nsys-max1-8rank-20260525-164647`
