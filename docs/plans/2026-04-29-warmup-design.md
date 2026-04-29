# Unified server-startup warmup design

**Status**: research + design (no code change).
**Owner**: ckl. **Reviewers**: codex.
**Crosslinks**: `infer/src/scheduler/cuda/core/warmup.rs`,
`infer/src/backend/cuda/bootstrap.rs`, `infer/src/main.rs`,
`crates/cuda-kernels/src/flashinfer.rs`, `docs/bench-and-trace-spec.md` §10.1,
`docs/plans/2026-04-23-cuda-decode-sglang-alignment.md`.

The runtime currently boots, captures CUDA Graphs for batched decode, and
goes straight to serving. Every kernel/plan/algo cache that was *not*
exercised during graph capture pays a JIT cost on the first real request:
TileLang AOT cubin load (`cuModuleLoadData` from
`crates/cuda-kernels/tools/tilelang/gen_tilelang_aot.py:352`),
FlashInfer plan creation for new (B,qlen) combinations
(`crates/cuda-kernels/src/flashinfer.rs:556` `flashinfer_plan`),
and cuBLAS algorithm autotune for unseen GEMM dimensions. That latency
lands on the first client and contaminates TTFT measurements.

This document maps SGLang's and vLLM's startup warmup, contrasts ours,
and proposes an ordered list of warmup steps to reach steady-state at
boot rather than first-request.

---

## §1. Current state — what `infer` warms up today

`infer/src/scheduler/cuda/core/warmup.rs:26` defines
`Scheduler::warmup_cuda_graphs`, which is the **only** warmup step the
runtime runs. It is invoked from `runtime/scheduler_loop.rs:98` once,
right before the scheduler's `run` loop starts.

What it covers (`warmup.rs:46-155`):

1. Allocates one paged-KV slot per warmup batch position
   (`warmup.rs:66-72`).
2. Lazy-creates the decode context (`warmup.rs:75-87` →
   `model.create_decode_context`). This allocates the
   `FlashInferDecodeMetadata` buffers and the FlashInfer workspace
   defined in `crates/cuda-kernels/src/flashinfer.rs:685`.
3. **Pass 1 (`warmup_graphs_pass`, `warmup.rs:159-235`)**: drives
   `forward_decode_batch` with dummy `token_id=0` for every batch size
   in the schedule. Each iteration calls
   `decode_ctx.update_metadata` (`warmup.rs:186`) and
   `plan_attention` (`warmup.rs:200`), then `forward_decode_batch`
   (`warmup.rs:217`). When `supports_cuda_graph_decode()` is true this
   *also* captures a CUDA Graph per batch size; otherwise it just
   populates the cublasLt heuristic algo cache.
4. **`autotune_all_cached_gemms_cuda`** (`warmup.rs:103-111`):
   benchmarks heuristic GEMM candidates and replaces each shape's algo
   with the measured-fastest one.
5. **Pass 2 (`warmup.rs:113-133`)**: invalidates the heuristic-time
   graphs and re-captures them with the autotuned algorithms. Eager
   decode (LoRA) skips this pass.
6. Frees all warmup slots (`warmup.rs:139-142`).

Batch-size schedule (`cuda_graph_batch_sizes`, `warmup.rs:246-264`):
**dense 1..=64** then sparse step-16 up to `min(num_slots, 256)`.

What it does **not** cover:

- **TileLang prefill HD128 / HD256 cubins**. The cubin is embedded in
  the C wrapper and loaded lazily via `cuModuleLoadData` on first
  invocation (`crates/cuda-kernels/tools/tilelang/gen_tilelang_aot.py:352`).
  First real prefill pays the load.
- **FlashInfer prefill paged plan** (`flashinfer.rs:323` `plan_hd128`,
  `flashinfer.rs:350` `plan_hd256`). Decode plans get touched by graph
  capture; prefill plans never run during warmup.
- **Mixed batch path** (`flashinfer.rs:913`
  `update_mixed_batch`). When K2 wires up FP8/INT8 mixed-batch decode,
  the first mixed step will pay first-launch cost.
- **KV-format codecs**: `crates/cuda-kernels/src/kv_quant.rs` and
  `kv_turboquant.rs` (FP8/INT8/TQ encode/decode) are never invoked
  unless the configured pool format already saw traffic during decode
  warmup — and the dummy slots have all-zero data, so quant paths may
  follow degenerate code branches.
- **Sampling kernels** (top-k, top-p, temperature): the current decode
  warmup forwards through the model but does not invoke the sampling
  kernels. Their cubin/JIT cost lands on the first real sampled token.
- **Correctness verification**. Graph capture succeeds with `[0]`
  garbage tokens; there is no smoke check that the produced logits
  match a reference. The 2026-04 `4e4906f5`/`47bad713` `!!!!!` series
  reached benches because warmup did not gate on correctness — see
  `docs/bench-and-trace-spec.md` §10.1.

Wall-clock today: scheduler logs the line
`"CUDA Graph warmup done in {ms}ms"` (`warmup.rs:149-155`); on L4 with
`num_slots=8` this typically lands at ~3-5 s including the cublasLt
autotune pass.

---

## §2. SGLang warmup

SGLang has **two** warmup phases: an in-process kernel/graph warmup in
the worker (`model_runner.py`) and an out-of-process HTTP smoke
warmup in the launcher (`http_server.py`). Together they cover what
our runtime today covers in just step (1) of phase 1.

### 2.1 Phase 1 — `kernel_warmup` + `init_device_graphs`

In `sglang/srt/model_executor/model_runner.py:651-655`, after the
attention backend is initialised, the worker runs **in this order**:

```python
self.init_cublas()              # 1950: 16x16 matmul to wake cuBLAS
self.init_attention_backend()   # 1959: allocate FA/FlashInfer workspaces
self.kernel_warmup()            # 2052: FlashInfer autotune (Hopper+)
self.init_device_graphs()       # 2418: CUDA Graph capture
```

- `init_cublas` (`model_runner.py:1950-1957`) is a 16x16 fp16 matmul
  whose only purpose is to force cuBLAS handle creation before
  anything else can fail.
- `kernel_warmup` (`model_runner.py:2052-2061`) currently only runs
  `_flashinfer_autotune` (`model_runner.py:2095-2110`) and only on
  SM≥9 with the `flashinfer_trtllm` MoE backend. It calls
  `_dummy_run(batch_size=req_to_token_pool.size, ...)`
  (`model_runner.py:2106-2108`) inside `flashinfer.autotune()` so the
  autotuner can profile every kernel variant once.
- `_dummy_run` (`model_runner.py:2112-2400+`) builds a
  `DecodeInputBuffers` tensor pack and a `ForwardBatch`, then calls
  `self.model.forward(...)` once. It always exercises decode (or
  TARGET_VERIFY for spec); prefill is reached via a different
  `forward_mode` only when the model is non-generation.
- `init_device_graphs` (`model_runner.py:2418-2462`) instantiates
  `CudaGraphRunner`, which iterates
  `get_batch_sizes_to_capture` (`cuda_graph_runner.py:454-488`) and
  for each entry calls `capture_one_batch_size`. The schedule is
  `[1, 2, 4, 8, 12] + range(16,257,8) + range(272,512,16) + range(512,
  cuda_graph_max_bs+1, 32)` (`server_args.py:1354-1359`). Default
  `cuda_graph_max_bs` for ≥90 GiB HBM is 512; for 35-60 GiB (L4 class)
  it is 24/80 (`server_args.py:1183-1185`).

Documentation hint: the user-visible log line is
`"Capture cuda graph begin. This can take up to several minutes."`
(`model_runner.py:2446`). On L4 with default settings the visible
elapsed is ~12-25 s for graph capture alone; the autotune phase adds
~1-3 s when enabled.

### 2.2 Phase 2 — HTTP-level synthetic-prompt warmup

After the server is bound and `/model_info` answers, the launcher
sends one real request (`http_server.py:1754-1910`):

- `text = "The capital city of France is"` for generation models, or
  `input_ids = [10, 11, 12]` if `skip_tokenizer_init`.
- `sampling_params = {"temperature": 0, "max_new_tokens": 8}`.
- Status flips to `ServerStatus.Up` only after this request returns
  200 (`http_server.py:1860-1861`). If the request fails the launcher
  kills the process tree.

This is documented as the "the server is fired up and ready to roll!"
log line (`http_server.py:1929`). It is the closest thing SGLang has
to a correctness gate, but it's a non-strict gate: any non-error
HTTP response counts.

### 2.3 What sticks as a side-effect

- cuBLAS handle (forced by `init_cublas`).
- FlashInfer plan-info host buffers, page-locked workspace, and
  `int_workspace`/`float_workspace` (allocated by
  `init_attention_backend`, populated by graph capture).
- CUDA Graph instances per batch size (held in `CudaGraphRunner.graphs`).
- cuBLAS algorithm cache for every (m,n,k) seen during the dummy
  forward.
- L2 working set of model weights warmed once across all layers.

SGLang does **not** verify correctness during warmup. The HTTP smoke
prompt only checks that 200 came back; it does not assert on the
content.

---

## §3. vLLM warmup

vLLM's V1 worker `compile_or_warm_up_model`
(`vllm/v1/worker/gpu_worker.py:552-700`) is more aggressive than
SGLang's. The full sequence:

1. **Compile-only sizes** (`gpu_worker.py:552-578`): collect any
   batch sizes from `compilation_config.compile_sizes` that are
   *not* in `cudagraph_capture_sizes`, plus the endpoints of every
   `compile_range`. These are sizes we want torch.compile to JIT
   without paying CUDA-Graph capture for.
2. **Dummy run for each compile size** in descending order
   (`gpu_worker.py:579-581`):
   `model_runner._dummy_run(size, skip_eplb=True, remove_lora=False)`.
3. **`kernel_warmup(self)`** (`gpu_worker.py:586`, defined in
   `vllm/model_executor/warmup/kernel_warmup.py:27-77`) does three
   things:
   - **deep_gemm_warmup** if enabled — JITs every (m,n,k) the model
     hits at `max_num_batched_tokens`.
   - **flashinfer_autotune** on SM≥9: runs one `_dummy_run` at
     `max_num_batched_tokens` inside `flashinfer.autotune()` so every
     FlashInfer impl is profiled once.
   - **FlashInfer attention warmup**: a `_dummy_run` with
     `num_tokens=16, force_attention=True, create_mixed_batch=True`
     so both prefill and decode attention kernels actually run with
     real tensors.
4. **`capture_model()`** (`gpu_worker.py:590` →
   `gpu_model_runner.capture_model`): CUDA Graph capture across all
   `cudagraph_capture_sizes`. Default is 67 sizes spaced 1..2..4..8 to
   max\_num\_seqs.
5. **V2 path: `warmup_kernels(model_runner, execute_model,
   sample_tokens)`** (`gpu_worker.py:667`, defined in
   `vllm/v1/worker/gpu/warmup.py:23-130`). Runs **two** end-to-end
   `execute_model + sample_tokens` iterations:
   - prefill with `prompt_len = 2 + num_spec_steps` per request,
   - decode with `decode_len = prompt_len + 1 + num_spec_steps`.

   This is the only step that actually touches the *combined* prefill
   + decode + sampler + KV-write path with real tensors. It also
   exercises the structured-output bitmask kernel via a `GrammarOutput`
   built with `np.full(..., -1)` (`warmup.py:108-115`).
6. **V1 path / non-V2**: `_dummy_run(num_tokens=max_num_reqs,
   cudagraph_runtime_mode=CUDAGraphMode.NONE)` followed by
   `_dummy_sampler_run(hidden_states=last_hidden_states)`
   (`gpu_worker.py:680-689`). Important: the sampler dummy run is
   placed **after** `capture_model` deliberately, "to prevent memory
   buffers from being cleared by `torch.accelerator.empty_cache`"
   (`gpu_worker.py:672-678`).
7. `set_random_seed(seed)` to undo any sampler RNG drift
   (`gpu_worker.py:692`).

Wall-clock: vLLM's logs report the elapsed in `CompilationTimes`. On
L4 with Qwen3-class models and torch.compile off, the kernel warmup +
capture is typically 15-30 s. With `VLLM_USE_DEEP_GEMM=1` it climbs
to 60-180 s because deep_gemm warms every (m,n,k).

vLLM also does **not** verify correctness during warmup. The closest
check is that `kv_cache_memory_bytes_to_gpu_limit` math after warmup
must not go negative.

---

## §4. Industry consensus and divergences

Both frameworks agree on:

- **CUDA Graph capture** for a discrete schedule of decode batch
  sizes, with a sparse upper tail so `max_num_seqs` is always
  captured. SGLang dense-then-sparse, vLLM 1/2/4/8/… exponential.
- **cuBLAS init via a tiny matmul** before any other GEMM. SGLang
  does this explicitly; vLLM relies on torch's first-tensor-op
  triggering it.
- **A FlashInfer autotune pass** on capable hardware (SM≥9). Same
  invocation pattern: enter `flashinfer.autotune()`, run one dummy
  forward at the largest expected token count, exit.
- **Synthetic-token dummy forward(s)** to populate workspace
  allocations and cuBLAS algo cache. Both use deterministic token IDs
  (zeros / `[10,11,12]`) and deterministic sampling
  (`temperature=0`).
- **No correctness gate during warmup**. Both treat "no exception"
  as success.

They diverge on:

- **Prefill coverage**: vLLM's `kernel_warmup` calls
  `_dummy_run(create_mixed_batch=True, force_attention=True)`, so the
  prefill attention kernel actually runs. SGLang's `kernel_warmup`
  only runs decode-mode forwards in the autotune pass.
- **Sampler coverage**: vLLM explicitly invokes a `_dummy_sampler_run`
  with `SamplingParams.for_sampler_warmup()` (exercises every
  sampling feature). SGLang's HTTP-warmup uses `temperature=0`
  greedy; the top-k/top-p paths don't get warmed.
- **HTTP-level smoke**: SGLang does it (one real
  `/generate` call with a fixed prompt); vLLM does not — its warmup
  ends inside the worker.
- **Compile-only sizes**: vLLM warms torch.compile sizes that are
  *not* graph-captured; SGLang either captures or skips — there is
  no "compiled-but-not-captured" path.

---

## §5. Proposed warmup for ARLE — ordered

Each step lists its **target file:line plug-in point**, the kernels /
buffers it covers, and a wall-clock budget.

### 1. [Mandatory] Pre-touch all AOT cubins (TileLang, future flash kernels)

**Why**: `cuModuleLoadData` for an embedded TileLang cubin runs lazily
on first kernel invocation. Today this happens during the first real
prefill (HD128 + HD256 paths) and lands ~30-60 ms on L4 per cubin —
all on the user's first TTFT.

**How**: extend `bootstrap.rs:spawn_scheduler_handle_from_path`
(currently `bootstrap.rs:248-295`) with a new helper called
*before* `Scheduler::with_config`. The helper invokes each AOT
wrapper's exposed `ensure_loaded()` (we don't have one yet — must add
to `gen_tilelang_aot.py`'s C wrapper). For each cubin, call the
loader once with no kernel launch; the wrapper's
static-init path (`tools/tilelang/gen_tilelang_aot.py:352`) will load
the module and resolve the function symbol.

**Budget**: ~50-100 ms per cubin × ~8 cubins (HD128/HD256 prefill +
HD128/HD256 decode + KV quant variants) → 0.4-0.8 s.

### 2. [Mandatory] CUDA Graph capture for batched decode (already exists)

**Why**: today's `warmup_cuda_graphs` covers this; keep it. Validate
the schedule against the SGLang reference (`server_args.py:1354-1359`)
once — our dense-1..=64 then step-16 (`warmup.rs:246-264`) is denser
than SGLang's `[1,2,4,8,12] + range(16,257,8)` and is a deliberate
choice for our smaller `max_slots` regime. Document the rationale
inline.

**Plug-in**: `warmup.rs:26` — already wired. No change needed here
beyond reordering relative to new steps below.

**Budget**: existing 3-5 s on L4 with `num_slots=8`.

### 3. [Mandatory] FlashInfer prefill plan creation for max-batch + max-qlen

**Why**: `BatchPrefillPagedPlan::plan_hd128` / `plan_hd256`
(`flashinfer.rs:323`, `flashinfer.rs:350`) is never run during
warmup. The first real prefill pays plan-info construction +
page-locked H2D copy.

**How**: after step 2 finishes, run one synthetic prefill plan with
`batch_size = chunked_prefill_size / 1024` (one chunk-sized req) and
one with `batch_size = max_prefill_requests` decoupled. The plan-only
call (`flashinfer_batch_prefill_paged_hdNNN_plan`) does not need
real KV data — just synthetic `qo_indptr`/`kv_indptr` arrays.

**Plug-in**: new `Scheduler::warmup_prefill_plans` called after
`warmup_cuda_graphs` from `runtime/scheduler_loop.rs:98`.

**Budget**: ~50-150 ms (two plan calls).

### 4. [Mandatory] Synthetic forward × N (decode + prefill, mixed if available)

**Why**: graph capture exercises decode but not prefill, and runs on
all-zero token tensors. cuBLAS algo cache for the prefill GEMM
shapes (especially the long-K activation × weight gemms at
`chunked_prefill_size=4096`) does not get populated. First real
prefill pays cuBLAS heuristic on every layer.

**How**: 1 synthetic prefill at `tokens=chunked_prefill_size` with a
single fake request, then 1 synthetic mixed batch (1 prefill + 1
decode) once mixed-batch dispatch lands (currently no Mixed plan
fires for FP8/INT8 — K2 follow-up). Run with random token IDs in
`[1, vocab_size)` to avoid degenerate `0`-token paths through
embeddings.

**Plug-in**: new `Scheduler::warmup_prefill_forward` after step 3.

**Budget**: ~150-300 ms (one prefill at 4096 tokens on L4 ≈ 100 ms +
H2D + sync).

### 5. [Mandatory] KV-format codec smoke (FP8 / INT8 / TurboQuant)

**Why**: `crates/cuda-kernels/src/kv_quant.rs` and
`kv_turboquant.rs` cubins are loaded lazily on first quantize-on-write
or dequantize-on-read. The first quantized request pays this; with
`auto`-mode default of `KVFormat::FP8E4M3` (`main.rs:455`) every
request is affected.

**How**: after step 4, run one quantize-then-dequantize round-trip
on a synthetic 1-page block whose elements are a sentinel pattern
(e.g. linspaced bf16). This forces both directions of the codec to
load. If `kv_pool_format == BF16`, skip.

**Plug-in**: new `Scheduler::warmup_kv_codec` after step 4.

**Budget**: ~20-40 ms.

### 6. [Recommended] Correctness smoke — known-prompt → non-degenerate output

**Why**: `bench-and-trace-spec.md` §10.1 mandates a correctness gate
before any perf number is published. Our warmup has no gate today,
which is how the `47bad713` `!!!!!` regression reached benches.

**How**: after step 5, run a 4-token deterministic prompt (e.g.
`"The capital of France is"` after tokenization → ~5 tokens) through
`prefill + 4 decode steps` with `temperature=0`. Assert:
- output token IDs are not all identical,
- first-byte of each decoded output is printable ASCII (not `!`,
  not control chars),
- forward did not return NaN/Inf in `last_hidden_states`.

This is the smoke half of §10.1. The full e2e check stays in
`infer/tests/e2e.rs` for CI.

**Plug-in**: new `Scheduler::warmup_correctness_smoke` as the last
step before `run` enters its loop. On failure: panic with a clear
message including which check failed. (The cost of serving with a
broken model is far higher than refusing to start.)

**Budget**: ~80-150 ms (one short prefill + 4 decode steps).

### 7. [Optional] Sampling kernel touch (top-k, top-p, temperature)

**Why**: greedy-only warmup leaves the top-k/top-p kernel paths
cold. First real request with `temperature>0` pays the kernel-load
cost.

**How**: after step 6, run `_dummy_sampler_run`-style invocation with
`temperature=0.7, top_k=40, top_p=0.9` on the hidden states from
step 6.

**Plug-in**: extend step 6's warmup to optionally re-sample the
last hidden states with non-greedy params.

**Budget**: ~10-30 ms.

---

## §6. Migration order

Each row is "what unlocks the most TTFT-on-first-request reduction".

| Order | Step | TTFT impact (first request) | Risk class |
|------:|------|----------------------------:|------------|
| 1 | §5.1 — pre-touch AOT cubins | ~400-800 ms | safety net |
| 2 | §5.4 — synthetic prefill forward | ~100-200 ms | perf |
| 3 | §5.3 — FlashInfer prefill plans | ~80-150 ms | perf |
| 4 | §5.6 — correctness smoke | guards bench validity | safety net |
| 5 | §5.5 — KV codec smoke | ~20-40 ms | perf |
| 6 | §5.7 — sampling kernel touch | ~10-30 ms | perf |
| — | §5.2 — decode graph capture | already shipped | — |

Recommended ship sequence: 1 → 4 → 2 → 3 → 5 → 6. Front-load the
safety nets (cubins + correctness gate) because they are
architectural, not perf-bound; the perf items can land in any order
once safety is in place.

---

## §7. Open questions

1. **Cubin-load API surface**: `gen_tilelang_aot.py` does not
   currently expose an `ensure_loaded()` symbol. Adding one means
   regenerating every cubin wrapper; need to confirm whether the C
   wrapper template still hits `cuModuleLoadData` in a static
   constructor (so a no-op call into the wrapper triggers it) or
   whether we need an explicit symbol to call. **Action**: read the
   generated wrapper for the current HD128 prefill cubin once.
2. **Prefill warmup's KV state**: a real prefill writes into the
   paged pool. Allocating 1 page × `chunked_prefill_size/page_size`
   pages of dummy state and freeing them on teardown is doable, but
   we should confirm the pool's `free_slot` path actually clears the
   epoch counter so subsequent real requests don't see stale-epoch
   slots (cf. `warmup.rs:139-142` already does this for decode
   warmup).
3. **Mixed-batch warmup gating**: K2 (FP8/INT8 mixed dispatch)
   has not landed. Step 4's mixed-batch arm should be feature-gated
   behind a "mixed dispatch available" check, not unconditional.
4. **SGLang's HTTP-level warmup**: SGLang does an additional
   external HTTP request via the launcher
   (`http_server.py:1754`). We could mirror this from the CLI
   (`arle serve`) wrapper rather than from the scheduler thread —
   leaves the scheduler boot path pure-Rust, lets us reuse the
   tokenizer warmup cost.
5. **Wall-clock budget validation**: the per-step budgets in §5 are
   estimates from L4 microbench history; they need a real measurement
   on the target box (HEAD with the warmup additions vs. HEAD
   without) before §6's ship order can be locked.
6. **Interaction with `cuda_graph=false`**: today eager-decode warmup
   skips Pass 2 (`warmup.rs:113`). The new prefill / codec / sampling
   warmup steps should still run in eager mode. Need to confirm
   `forward_decode_batch`'s eager path is exercised by the existing
   warmup loop; if not, eager mode pays the codec/sampling-load cost
   on first request just as graph mode does.
7. **Multi-GPU**: when single-node-multi-GPU lands
   (`docs/plans/2026-04-28-single-node-multi-gpu.md`), every per-rank
   worker needs to run warmup independently. Need to decide whether
   warmup is gated on a barrier (so all ranks finish before serving
   begins) or per-rank (rank 0 starts serving when its warmup is
   done). SGLang barriers; vLLM also barriers via `torch.distributed`.

---

**End of design.** Implementation lands in a follow-up project entry
under `docs/projects/` once §7.1 (cubin-load API) is resolved.
