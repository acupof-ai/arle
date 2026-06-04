# Model load + compile optimization — research (evidence-gated)

Parallel research pass (2026-06-04). Both halves are **research → ranked opportunities**;
per §0 each ships with a cheap experiment to license-or-kill (source survey =
hypothesis until a local measurement confirms). Implementation is deferred behind
those gates; this is the roadmap.

## A. Model weight LOADING

**Biggest cost (high confidence from code):** CUDA per-tensor **pageable-synchronous
H2D, fully serialized on the compute stream.** Each weight: `fs::read(shard)` →
`SafeTensors::deserialize` → `view.data().to_vec()` (an extra host copy) →
`clone_htod(&Vec<u8>)` on `ctx.stream`. Pageable-source H2D is implicitly synchronous
(driver stages through an internal pinned buffer + blocks) → hundreds–thousands of
blocking copies, no read↔copy overlap, no pinned bandwidth, and the dedicated
`copy_stream` (`cuda-kernels/src/tensor.rs:181`) sits idle. The `shard_cache`
(`loader.rs:305`) holds **every shard's bytes in host RAM simultaneously** (no
eviction) → ~full model duplicated in RAM. Dominant DSv4 (~19.6 GB/rank) cold term.

**Ranked levers** (CUDA-focused; Metal is evidence-backed *out of scope* — M_e.13:
import ~35 µs / disk ~100 ms, not the bottleneck):
1. **(highest leverage, lowest risk) Pinned-buffer + async H2D on `copy_stream`,
   overlapped with the next tensor's read/deserialize.** cudarc primitives
   (`alloc_pinned`/`PinnedHostSlice`/`memcpy_htod_async`) + the `CudaPipelineFence`
   plumbing (`tensor.rs:216`) already exist. Touches `tensor.rs:{2559,652,1701,1770}`
   (the `clone_htod` calls) + the loop drivers `loader.rs:57-151` / `dsv4.rs:306-334`.
   Precedent: RunAI Model Streamer (47s→7.53s, ~6×).
2. **Parallel multi-thread shard read+deserialize** (the CPU/IO half). `loader.rs:618`
   single-shard `RefCell` cache → needs `Arc<Mutex>`/per-thread. Precedent: SGLang
   `buffered_multi_thread_safetensors_weights_iterator`, vLLM `ThreadPoolExecutor`.
3. **mmap instead of full `fs::read` + drop the `.to_vec()` double-copy + evict the
   shard cache** (the RAM-duplication). Precedent: safetensors `MmapedSafetensors`,
   SGLang `weight_loader_disable_mmap=false`.
4. **DSv4: fuse/skip the device-side FP8 repack** (`tensor.rs:1149-1191`
   `dsv4_fill_fp8_deepgemm_weight_cache` — H2D raw FP8 then a 2nd device repack/expert).

**GATE FIRST:** an env-gated load-phase timer (read-µs / deserialize-µs / h2d-µs per
shard, mirror the `INFER_M_E13_TRACE` probe) — until it splits the phases, the #1-vs-2-vs-3
ranking is hypothesis. The timer tells you H2D-bound (→#1) vs IO/parse-bound (→#2/3)
vs RAM-bound (→#3). One cheap change, turns the whole ranking into evidence.

## B. COMPILATION

### B-build (metric: dev rebuild wall-clock)
**Root cause:** `cuda-kernels/build.rs` re-does, with **no internal source-keyed
cache**, on every build-script rerun: (1) a **serial nvcc loop** (`:1821`, one process
per `.cu`, no parallelism/cache); (2) **TileLang AOT full regen** (`:1000-1389` →
`tools/tilelang/gen_tilelang_aot.py:451` — re-lowers + re-`nvcc -cubin` even when the
`.py`+SM are byte-identical; one HD128-decode family alone ~3m, full set × every SM ≈
30 min). `cargo:rerun-if-changed` IS recursive (`emit_rerun_recursive :1423`, the old
gotcha is fixed) — but it only gates *whether* build.rs runs; once it runs it rebuilds
everything. A RUSTFLAGS/feature fingerprint flip forces the full rerun; the existing
`ARLE_CUDA_KERNELS_PREBUILT_DIR` manifest is **RUSTFLAGS-blind + all-or-nothing**.

Prior art already shipped (`wins/2026-06-02-cuda-kernels-build-fast-path.md`, 4.92s
prebuilt path): `ARLE_CUDA_KERNELS_PREBUILT_DIR`, `ARLE_NVCC_WRAPPER` (sccache),
`ARLE_NVCC_SPLIT_COMPILE`, `ARLE_CUDA_KERNEL_SET=dsv4_flash` stub, `release-fast`.

**Ranked:**
1. **(biggest) Per-(source+SM) content-addressed cubin/object cache** keyed on
   `sha256(kernel.py + generator + SM + tilelang ver + nvcc ver)` / `sha256(.cu +
   includes + flags)`, in a stable dir (not `$OUT_DIR`). Makes a RUSTFLAGS/feature
   flip rebuild only Rust + relink — the 30-min TileLang regen + nvcc loop become
   cache hits. (TileLang upstream has `KernelCache`/`TILELANG_CACHE_DIR` keyed on the
   lowered IR — ARLE's generator bypasses it by always re-lowering.) **Risk:** cache-key
   completeness (stale-cubin class, `errors/...dsv4-prebuilt-symbol-gate`); key on the
   full nvcc arg vector + reuse the `nm`-symbol gate (`build.rs:1537`).
2. **sccache for TileLang's internal `nvcc -cubin`** — `gen_tilelang_aot.py:543` calls
   bare `nvcc`, ignoring `ARLE_NVCC_WRAPPER`. Wire the wrapper there.
3. **Parallelize the serial nvcc loop** (`build.rs:1821`, bounded thread pool, cap RAM).
4. **Default a narrowed `TORCH_CUDA_ARCH_LIST`** for iteration (default T1={80,86,89,90}
   = 4 SMs × every TileLang kernel = 4× cost).
- **Gate:** on H20, `touch build.rs` + time full build; re-time with warm sccache and
  with a RUSTFLAGS flip — isolates how much A1's cubin-cache buys beyond A2/A3.

### B-runtime (metric: per-step inference wall-clock)
- **CUDA decode graph: essentially solved** (`graph.rs`/`decode_graph.rs`, replay = one
  `cuGraphLaunch`, keyed by `num_pages`, retained → no thrash). Minor: **batch the 8
  Stage-1 `memcpy_htod` into 1** (`decode_graph.rs:148`); gate on per-token ITL Δ.
- **Metal: the real lever — the ~95%-of-step encode** (23-25 ms, 600-1000 primitives
  via `mx::async_eval`; `wins/2026-05-07-bench-qwen36-encode-bottleneck.md`). Whole-forward
  `mx::compile` is **upstream-blocked** (GatherQMM lacks `output_shapes` mlx#3485;
  value-dependent MoE `sort` re-traces). **Cheapest win = primitive-count reduction**
  (`switch_glu_forward` `expand_dims` + `_gather_sort`/`_scatter_unsort` collapse,
  `mlx_qwen35_model.cpp`); encode is ~linear in primitive count. Secondary: compile the
  position-independent GDR+MLP sublayers (`mlx_qwen35_model.cpp:1429`). **Do NOT** do
  naive multi-thread encode (falsified, `feedback_mlx_async_eval_is_caller_thread`).
  **Gate:** count primitives/step via `INFER_CPP_PHASE_TIMING`, apply the fusion in a
  scratch build, matched c=1 A/B (`feedback_matched_ab_for_small_bench_effects`).

## Priority read
Highest-leverage, each gated on its cheap experiment: **load = #A1 (pinned async H2D)**,
**build = #B-build-1 (content-addressed cubin cache)**, **runtime = #B-Metal primitive
reduction**. All need a GPU to measure (pod/colab); none are licensed until the timer/
A-B confirms — implementation is the follow-up, not done here.
