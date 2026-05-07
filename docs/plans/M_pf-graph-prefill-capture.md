# M_pf-graph-prefill-capture - Prefill CUDA Graph Capture

## Priority and ROI

**Priority**: P1 strategic, with Phase 0 as a P0 license-or-kill proof.

R1 and the M_world1 P0 wins entry point at the same gap: SGLang
graph-captures prefill/extend by token bucket, while ARLE graph-captures
decode only.

| Shape | Current ARLE | Current #2 | Gap | 30% lead target |
|---|---:|---:|---:|---:|
| 4k/c=4 TTFT p50 | 1976 ms | SGLang 973 ms | ARLE 2.03x slower | <= 748 ms |
| 8k/c=4 TTFT p50 | 4574 ms | vLLM 2362 ms | ARLE 1.94x slower | <= 1816 ms |

ROI math from R1:

- Qwen3-4B prefill is roughly 36 layers x ~7 launch-heavy ops/layer.
- At chunked prefill 2048, a 4k request is two chunks.
- Launch floor: 36 x 7 x 2 x 7.5 us = 3.8 ms.
- R1 conservative range: 4-7 ms saved per 4k request from launch replay alone.
- Direct floor estimate: 1976 ms -> 1969-1972 ms.
- That floor does not explain the full 1004 ms SGLang gap.
- The license exists because graph capture can also remove host-side dispatch,
  cudarc/cublasLt call overhead, event traffic, and dynamic scheduler work.

License threshold: launch-only 1969-1972 ms keeps this only as Phase 2
substrate; 1850-1950 ms licenses Phase 1; <= 1700 ms promotes immediately.

## Negative Case

- Prefill TTFT is dominated by GEMM/attention math, not launch or host dispatch.
- cuBLASLt prefill algorithms require workspace or lazy setup that rejects
  stream capture.
- `prefill_attention_paged_batch` allocates, records events, or mutates
  host-owned metadata inside the capture body.
- Token-bucket capture forces smaller chunks and hurts 8k, matching SGLang's
  2048-chunk 8k failure mode.
- LoRA, prefix reuse, FP8 paged KV, or mixed decode+prefill make Phase 0 look
  good but Phase 1 unusable.

## Kill Criteria

- Stop immediately if this plan or a follow-up plan diff exceeds 250 net lines.
- Kill Phase 0 if the opt-in graph path exceeds ~200 LOC before first bench.
- Kill Phase 0 if 4k/c=4 TTFT p50 improves by < 10 ms and nsys shows no
  material reduction in CPU launch overhead.
- Kill Phase 0 if correctness differs from eager prefill beyond current JSON
  baseline tolerance.
- Kill Phase 0 if capture requires disabling paged prefill or async prefill.
- Kill Phase 1 if 42 buckets add > 1 GiB steady GPU memory or unacceptable
  startup capture time.
- Kill the M_world1 claim if graph capture plus TileLang/FP8 does not move 4k
  TTFT below 1500 ms.

## Phase 0 - License Or Kill

Scope: one bucket, Qwen3 BF16 paged prefill, CUDA only, opt-in
`INFER_PREFILL_GRAPH=1`, default eager, first bucket exactly 2048 tokens,
target size ~200 LOC. Disable for LoRA, non-CUDA, non-Qwen3, non-paged
prefill, mixed batches, and unsupported KV formats.

- Reuse the Qwen3 decode graph pattern in `batch_decode.rs`.
- Decode cache is batch-size indexed; prefill cache must be token-count indexed.
- Add a small `PrefillGraphCache` next to Qwen3 paged prefill buffers/context.
- Split paged prefill into prepare/body/finish helpers.
- `prepare_prefill_graph_inputs(...)`: CPU packing, H2D metadata, allocation.
- `prefill_graph_body_2048(...)`: pure GPU kernel sequence.
- `finish_prefill_graph_outputs(...)`: logits extraction and owner bookkeeping.
- Only the graph body can be captured.
- Allocate all `CudaSlice`, `HiddenStates`, `PagedPrefillForward`, token-row
  arrays, and output buffers before `begin_capture`.
- Capture the full prefill layer loop: norms, QKV GEMMs, Q/K norm, RoPE, KV
  write, TileLang paged attention, O projection, residuals, MLP, final hidden.
- Do not capture request admission, page allocation, H2D metadata upload, host
  vector construction, event record, or logits readback.

- R1's key design point applies: attention must accept caller-provided output.
- ARLE already passes `bufs.attn_output` into `prefill_attention_paged_batch`.
- Phase 0 must prove lower layers do not allocate inside that call.
- If they do allocate, add a narrow `_into` or graph-safe variant first.
- `PagedPrefillForward` and TileLang workspace must be stable by pointer and
  size for the bucket.

- Route to graph only when total admitted prefill tokens equals 2048.
- Other token counts stay eager.
- A 4k single request should run as 2048 + 2048.
- When `INFER_PREFILL_GRAPH=1` is active, set or clamp `chunked_prefill_size`
  to the captured bucket.
- If admission produces 2047/2049 due to page or prefix boundaries, log fallback
  reason and keep eager.
- Acceptance: `cargo test --release`, CUDA/no-CUDA check when needed, one
  longctx 4k/c=4 `scripts/bench_guidellm.sh`, wins/errors entry, and logs for
  bucket, capture success, replay count, fallback count, and disabled reasons.

## Phase 1 - Piecewise Token Buckets

Mirror SGLang's 42 prefill token sizes:

`2048, 1792, 1536, 1280, 1024, 960, 896, 832, 768, 704, 640, 576, 512, 480,
448, 416, 384, 352, 320, 288, 256, 240, 224, 208, 192, 176, 160, 144, 128,
112, 96, 80, 64, 48, 32, 28, 24, 20, 16, 12, 8, 4`.

Design:

- Store graph cache by `num_prefill_tokens`, not by batch size.
- Prefer exact bucket admission first; padding is a later optimization.
- Generate buckets from a constant table so logs and docs match SGLang.
- Add scheduler helper: nearest captured bucket <= current prefill budget.
- Align `chunked_prefill_size` to the max captured bucket.
- Keep 2048 as the default max on 16 GiB GPUs.
- Consider 4096 only after 8k/c=4 analysis.
- Warm buckets lazily on first use; add startup warmup only after memory and
  capture time are known.
- Keep eager fallback for unsupported shapes.

Acceptance: 4k/c=4 improves beyond Phase 0, high-conc c=64 and 8k/c=4 do not
regress, fallback counters explain all eager prefills, and graph memory/capture
time are reported in the wins entry.

## Phase 2 - Unique ARLE Stack

The world #1 angle is not "copy SGLang"; it is the combination:

- Prefill CUDA graph capture by token bucket.
- TileLang HD128 paged prefill attention.
- FP8 paged KV combine for prefix/paged-pool bandwidth.
- Future M_pf GEMM work where it beats cuBLASLt at captured shapes.

Competitor split: SGLang has piecewise prefill graph capture plus FlashInfer,
but BF16 dense GEMM falls through to PyTorch/cuBLAS; vLLM has mature fused
linear primitives and strong attention kernels, but the evidence here points to
decode-only graph capture by default. ARLE already owns TileLang
`batch_prefill_paged_hd128` and FP8 paged KV plumbing; graph capture is the
missing runtime layer.

Phase 2 acceptance:

- 4k/c=4 TTFT moves below 1500 ms before calling the stack strategic.
- The M_world1 4k target remains <= 748 ms, so graph capture must combine with
  kernel-level prefill work.
- Final A/B table: eager BF16, graph BF16, graph+TileLang, graph+TileLang+FP8.

## Tasks

| # | Task | Owner | LOC | Trigger |
|---|---|---:|---:|---|
| 0 | Confirm SGLang log evidence for 42 buckets | Claude | 0 | before impl |
| 1 | Audit Qwen3 paged prefill for allocation/event/H2D in layer loop | Explore | 0 | Phase 0 |
| 2 | Add opt-in env gate and 2048 bucket constant | general-purpose | 20 | Phase 0 |
| 3 | Split Qwen3 paged prefill into prepare/body/finish helpers | general-purpose | 80 | Phase 0 |
| 4 | Add token-count graph cache and capture/replay | general-purpose | 80 | Phase 0 |
| 5 | Add graph hit/fallback/capture logs and counters | general-purpose | 30 | Phase 0 |
| 6 | Bench 4k/c=4 and write wins/errors entry | Claude | 0 | Phase 0 done |
| 7 | Run `codex review --uncommitted` | Claude | 0 | non-trivial diff |
| 8 | Expand to 42 token buckets | general-purpose | 80 | Phase 0 passes |
| 9 | Add scheduler bucket admission helper | general-purpose | 80 | Phase 1 |
| 10 | Combine with TileLang HD128 and FP8 paged KV A/B | general-purpose | varies | Phase 2 |

## Cross-References

- R1 survey: `docs/research/2026-05-07-sglang-prefill-stack-survey.md`
  at commit `7ef707d`.
- M_world1 full table/key innovation:
  `docs/experience/wins/2026-05-07-m_world1-p0-sglang-baseline-extended.md`
  at commit `4ae3b7b`.
- Roadmap: `docs/plans/M_world1-30-percent-lead-roadmap.md`.
- Decode graph cache/capture: `infer/src/model/qwen3/batch_decode.rs:169`,
  `infer/src/model/qwen3/batch_decode.rs:1691`.
- Decode graph wrapper/warmup: `infer/src/model/cuda_graph.rs`,
  `infer/src/scheduler/cuda/core/warmup.rs:11`.
- Qwen3 paged prefill: `infer/src/model/qwen3/prefill.rs:459`,
  `infer/src/model/qwen3/prefill.rs:565`.
- Graph-safe BF16 GEMM: `infer/src/ops/linear.rs:608`,
  `crates/cuda-kernels/csrc/gemm/gemv.cu`.
- TileLang HD128 prefill: `crates/cuda-kernels/tools/tilelang/batch_prefill_paged_hd128.py`,
  `crates/cuda-kernels/src/ffi/attention.rs`.

## Rules

- Token bucket, not batch size: prefill shape is total packed prefill tokens.
- Graph body is GPU-only: no allocation, H2D, host memcpy, event record, or CPU
  sync inside capture.
- Capture is opt-in until a bench entry proves value.
- Eager fallback must stay complete and visible in counters.
- Do not leak CUDA graph types into backend-neutral scheduler interfaces.
- Do not leave parallel old/new prefill paths without one canonical dispatch.
- No LoRA or mixed decode+prefill graphing in Phase 0.
- The first implementation should be small enough to delete cleanly if killed.
