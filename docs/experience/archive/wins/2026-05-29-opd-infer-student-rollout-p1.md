# InferStudent bring-up + zero-LoRA rollout validation (OPD P1)

## Context

Plan [`docs/plans/2026-05-29-opd-student-rollout-via-infer.md`](../../plans/2026-05-29-opd-student-rollout-via-infer.md)
P1: route the OPD student rollout through the infer engine instead of the
train-crate hand-written O(n²) decode (208 s / ~130 tok = 1.6–2.88 s/tok).
P1 is standalone bring-up + measurement of the two gating unknowns at
**step 0 (zero LoRA, student == base)** — no LoRA-sync code (that's P2).

New `crates/train/src/infer_student.rs` mirrors `InferTeacher`:
`InferStudent { engine: Arc<Mutex<LoadedInferenceEngine>>, train_backend,
vocab_size }` with `decode_next_token(input_ids, positions)` →
`engine.forward_token_logits` + **host argmax over the last position**.
Validation harness: `crates/train/tests/test_infer_student_rollout.rs`
(`#[ignore]`, cuda-gated).

## Params / Env

- GPU: RTX 4070 Ti SUPER (sm_89), 16376 MiB total, 1051 MiB used by others before.
- CUDA build: `CUDA_HOME=/opt/cuda NVCC_CCBIN=g++-14 INFER_TILELANG_PYTHON=.venv/bin/python TORCH_CUDA_ARCH_LIST=8.9 ARLE_CUDA_DISABLE_FLASHMLA=1`
  (sm89 box — default build compiles FlashMLA sm90 sparse_fp8 decode which
  fails with `cannot specify max blocks per cluster for this GPU architecture`;
  disable FlashMLA + pin arch to 8.9).
- Model: `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base` (24 hybrid
  layers, head_dim=256, NO LoRA = step-0). Single infer engine,
  `max_slots=1`, `mem_fraction_static=0.05`, `max_seq_len=256`, no CUDA graph.
- Prompt: fixed 16-token id sequence. Greedy rollout 128 tokens; each step
  re-submits the full growing sequence with `positions = 0..len`.

## Results

Per-token latency (ms), single greedy rollout, growing context:

| token t | latency (ms) |
|--------:|-------------:|
| t=1     | 176.06 (cold, first prefill) |
| t=32    | 36.92 |
| t=64    | 63.86 |
| t=128   | 126.07 |
| mean    | 67.53 |

- **Total rollout wall-clock: 8.644 s for 128 tokens.**
- VRAM (MiB): before=1051, after_load=2894, **peak=4512** (~4.4 GB for one
  0.8B engine + train TensorStore).
- Tail tokens in-vocab and non-degenerate (`[4128,369,9934,13,271,760,4128,369]`).

### Key structural finding (evidence, not hypothesis)

`forward_raw_logits` (`infer/src/scheduler/cuda/runtime/fetch.rs:399`):
1. **hard-`ensure!` contiguous positions starting at 0** (lines 414–420) →
   v1 does NOT support incremental single-token decode with absolute positions.
2. **creates a fresh `state` each call** (line 424) → no KV reuse across calls;
   each token is a full re-prefill of the growing sequence.

So the observed near-linear growth (37→64→126 ms) is the inherent
re-prefill cost (O(n)/tok → O(n²) rollout), **not** a per-call Arc<Mutex>/
scheduler pathology. The constant is ~24× smaller than the train-crate path
at this shape (8.6 s vs 208 s).

## License/Kill verdict — LICENSE (with caveat)

Anchor: train-crate 208 s / ~130 tok. Infer: **8.64 s / 128 tok → 24× faster**.

- **KILL conditions NOT met**: per-token is tens of ms (not seconds), no
  pathological per-call overhead, peak VRAM 4.5 GB ≪ 6 GB threshold → leaves
  ~11.8 GB headroom on the 16 GB card for the 4B teacher (~8 GB). Two
  in-process engines fit.
- **Plan's literal "<5 s total" bar NOT met (8.64 s)** — but that bar assumed
  flat per-token latency. The growth is structural (v1 re-prefills; no
  incremental decode path), and 24× over the anchor already clears the P1
  PASS gate "step-time drops ≥2×" by a wide margin.
- **LICENSE P2.** The 8.6 s is the re-prefill ceiling; the next throughput
  axis (if needed) is an incremental-decode entry point in
  `forward_raw_logits` (KV-cache reuse), but that is NOT required to beat the
  anchor 2×. Proceed to P2 (B1.5 in-memory re-merge LoRA sync) on this path.

## Rule

- For tiny in-process infer-engine rollouts, the v1 `forward_token_logits`
  path re-prefills the full sequence every call (fresh state, contiguous-from-0
  positions enforced) — budget O(n²) but with a ~500×-smaller constant than a
  hand-written autograd decode loop. Validate per-token *growth shape*, not
  just t=1, before licensing.
- On an sm89 box the default CUDA build must set `ARLE_CUDA_DISABLE_FLASHMLA=1`
  (+ `TORCH_CUDA_ARCH_LIST=8.9`) or FlashMLA sm90 sparse_fp8 decode kernels
  fail to compile.
