# Session KV-recall on CUDA dense-Qwen3 (restricted page table) — pending-remote

`pending-remote`: implemented + Mac-cross-compiled (no nvcc/GPU locally). The
8×H20 pod validation below is the bench/correctness gate; the human runs it via
`bin/pod`. This stub lands per the §Benchmarks rule (every runtime change → a
wins/ entry or pending-remote stub).

## Context

CUDA mirror of the Metal session KV-recall ("infinite memory") landed in commit
`898bb979` (`docs/plans/2026-06-23-session-infinite-kv-memory.md`,
`wins/2026-06-23-kv-recall-arle-core-e2e.md`). When a session exceeds the GPU
working set, decode attends a recalled subset (sink ∪ top-k mean-key ∪ local)
instead of the full cache.

**CUDA-natural approach (no gather kernel): restricted page table.** CUDA decode
already runs paged attention over `meta.kv_indices` / `kv_indptr` /
`kv_last_page_len`, and the TileLang decode kernel derives the KV length entirely
from the page table (no per-KV-token position array). So recall writes only the
SELECTED pages to the page table; the kernel attends exactly the working set.
Correct because RoPE is baked into the cached K at write time and the query's
RoPE position is independent of which KV pages are attended.

- **Model / path wired:** dense **Qwen3.5** (`from_qwen3_bf16_safetensors`,
  `QwenCudaExecutor`, `model.rs`) — the only CUDA arm with a paged page table +
  page-granular tier. Qwen3.6 hybrid and DSv4 own per-slot KV internally (no
  paged pool) → `--kv-recall` logs + ignores there.
- **Graph compatibility:** recall is **eager-only**. The captured decode graph
  bakes `num_pages`, and recall needs a host layer-0-query read-back + restricted
  page table between steps; a recall-active slot skips the graph
  (`try_recall_decode` runs before `try_captured_decode`). Default off → the
  graph path + baseline are byte-identical.
- **L3 tier offload:** the **resident variant** (full KV stays in HBM; recall
  restricts attention = a decode-compute saving). The flat-VRAM-vs-history win
  (free the non-selected middle pages to L3, keep only the resident rep) is
  `TODO(kv-recall L3)` — it needs the executor to own mid-decode device-page
  lifecycle, which conflicts with the host-`CudaKvPool`-is-single-allocator
  contract (`SlotProgress`). Same deferral Metal made.

## What's real vs stubbed

| Piece | Status |
|---|---|
| `--kv-recall` flag → `EngineLoadConfig.kv_recall` → `CudaExecutor::set_kv_recall` (single `build_cuda_engine`, covers single-GPU + TP) | real |
| Resident per-block mean-key reps (layer-0 K mean-pool, `CudaRecallState::update_block_reps`) | real |
| `q·rep` scoring (post-RoPE layer-0 query read-back, GQA-mean) → `infer_core::plan_recall` → next-step page plan (stale-Q) | real |
| Restricted page table (`PageMeta::for_recall_decode`) + eager recall forward (`CudaModel::forward_decode_recall`) | real |
| Default-off byte-identical (`try_recall_decode` returns `None` when off / below budget / non-BF16) | real |
| L3 device-page free/promote during decode (flat-VRAM win) | **stubbed** (`TODO(kv-recall L3)`) |
| GPU correctness + perf numbers | **pending-remote** (this doc) |

## Pod test plan (8×H20, run via `bin/pod`)

Use a dense BF16 Qwen3.5 checkpoint (the paged-KV arm). Recall is **BF16-only**;
do NOT pass `--kv-cache-dtype int8/fp8` (recall falls back to full attention,
logged once).

### 1. Serve (recall ON, single GPU — eager decode)

```bash
# Recall is eager-only; disable the decode graph so the recall path is exercised
# (the engine already skips the graph for recall-active slots, but be explicit).
INFER_CUDA_DECODE_GRAPH=0 \
arle serve --backend cuda \
  --kv-recall \
  --kv-cache-dtype bf16 \
  --model-path <Qwen3.5-dense-bf16 checkpoint> \
  --port 8000
```

Baseline control (recall OFF, same binary/shell/model): drop `--kv-recall`.

### 2. Correctness — long-context needle (correct-inference gate, NOT byte-identity)

Recall + restricted attention deliberately deviates from a single full-KV run, so
the gate is **needle retrieval = the full-attention answer**, not token-exact vs
baseline (`feedback_correct_inference_not_baseline_identity`). Plant a passkey at
mid-depth in a prompt LONGER than the working-set budget (sink 32 + local 256 +
8×32 = **544 tokens**); e.g. a ~6K-token context with the passkey at depth 0.5.

```bash
# Working set must be exceeded: prompt + generation > 544 tokens.
# Mirror scripts/kv_recall_quality_eval.py's passkey prompt (the Metal e2e used
# ctx~5684, mid-depth passkey, acc 1.00 at 9.6% KV).
curl -s localhost:8000/v1/completions -d '{
  "model": "<id>",
  "prompt": "<~6K-token filler with PASSKEY 84213 planted at depth 0.5> ... What is the passkey?",
  "max_tokens": 32, "temperature": 0
}' | jq -r '.choices[0].text'
```

Pass = the recalled answer contains the planted passkey (= the full-attention
answer). Run x3 same-config repeats (needle ladder vs the baseline envelope,
`scripts/needle_gate.py` style) to absorb MoE/run-to-run non-determinism. A
streaming control (sink+local, no recall) is expected to MISS at the same budget
— that's the Metal e2e result (recall 1.00 vs stream 0.00 at 9.6% KV) and is what
makes recall load-bearing.

### 3. Perf / VRAM — flat-VRAM-vs-history (the §6 decisive evidence)

Because this is the resident variant, **HBM does NOT yet flatten** (full KV stays
resident; `TODO(kv-recall L3)`). What recall flattens NOW is the **decode KV-read
volume / per-step attention cost**: attention reads a bounded 544-token working
set regardless of session length, not the growing full cache. Measure the decode
ITL / tok-s curve as the session grows:

```bash
# Grow one session turn-by-turn well past 544 tokens (e.g. to 4K, 8K, 16K) and
# record per-step decode latency. Recall ON should hold ~flat once past the
# budget; recall OFF grows with cache_len (until DECODE_GRAPH_MAX_SEQ_LEN).
scripts/bench_guidellm.sh cuda-kv-recall-on \
  --model <Qwen3.5-dense-bf16> --extra-serve-args "--kv-recall --kv-cache-dtype bf16"
# Control:
scripts/bench_guidellm.sh cuda-kv-recall-off-baseline \
  --model <Qwen3.5-dense-bf16>
```

Also snapshot `nvidia-smi --query-gpu=memory.used --format=csv -l 1` across the
growing session for both arms. Expected at THIS stage: VRAM curves overlap
(resident variant), decode ITL diverges (recall flat vs baseline rising). When
`TODO(kv-recall L3)` lands, the VRAM curve is the win.

### 4. Default-off regression (mandatory, byte-identical)

Confirm recall OFF is unchanged: run the §3 baseline arm with the captured decode
graph ON (default), verify tok-s matches the latest dense-Qwen3 CUDA baseline
wins entry within noise (the recall code is behind `if !self.kv_recall` and the
graph path is untouched).

## Files changed

- `crates/infer-cuda/Cargo.toml` — add `infer-core` dep (for `plan_recall`;
  acyclic, mirrors `infer-metal`).
- `crates/infer-cuda/src/recall.rs` — new: `CudaRecallState` (reps + page plan),
  `default_recall_config`, `token_ranges_to_pages`.
- `crates/infer-cuda/src/lib.rs` — register `recall` module; `CudaExecutor::set_kv_recall`.
- `crates/infer-cuda/src/executor.rs` — `kv_recall`/`recall_cfg`/`recall` fields,
  `set_kv_recall`, `try_recall_decode` decode hook, epoch-reset.
- `crates/infer-cuda/src/model.rs` — `CudaModel::forward_decode_recall` (eager
  recall forward + layer-0 query capture).
- `crates/infer-cuda/src/loader.rs` — `PageMeta::for_recall_decode` (restricted
  page table).
- `crates/infer-api/src/loaded.rs` — wire `set_kv_recall(config.kv_recall)` in
  `build_cuda_engine`.
- `crates/cli/src/args.rs` — `--kv-recall` docstring now names CUDA dense-Qwen3.

## Gates (local, Mac — no nvcc)

- Mac CUDA typecheck: `CUDARC_CUDA_VERSION=12080 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib` → clean.
- `cargo test -p infer-core recall` → 6/6 pass; full infer-core 66/66.
- `cargo clippy -p infer-cuda` → no findings in changed files (7 pre-existing
  `attention.rs`/`dsv4.rs` lints from rust-1.95 clippy, untouched files).
- no-cuda build clean (ungated `recall` module compiles without the cuda feature).

## Rule

CUDA recall = restricted page table, not a gather kernel: the paged decode
already keys attention off `kv_indices`, so writing only the selected pages
restricts the working set with zero new kernels. Eager-only (graph bakes
`num_pages`); resident variant first (L3 device-page free deferred, same as
Metal); default-off keeps the Stable backend byte-identical.
