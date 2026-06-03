# R6 CUDA Port Plan - Existing cuda-kernels Forward to the New Seam

**Branch:** `arch/ideal-inference-engine`
**Scope:** investigation and implementation plan only. No CUDA forward code is
implemented in this round.
**Goal:** wire the existing tested CUDA forward path behind
`infer_seam::BackendExecutor`, without depending on the legacy `infer` crate and
without re-deriving model numerics.

The Metal track proved the architecture with a real backend. R6 is the CUDA
analog: keep the new `infer-core` scheduler and host-only seam, port the tested
CUDA model/kernel path into the new crates, and verify on the pods that actually
have NVIDIA hardware.

## Existing CUDA Construction Sequence

Legacy CUDA enters through `infer/src/backend/cuda/bootstrap.rs`, then hands a
model to `infer/src/scheduler/cuda/*`.

The construction sequence is:

1. `LoadedInferenceEngine::load` builds a `ServerRuntimeConfig` and calls
   `spawn_scheduler_handle_from_path`.
2. `ResolvedModelSource::resolve` turns the model path or HF id into a concrete
   directory/GGUF source.
3. `model_registry::detect_arch` reads the config and chooses one of
   `Qwen3`, `Qwen35`, `Qwen35Moe`, or `DeepSeekV4`.
4. `load_model_components` constructs the model:
   - Qwen3 calls `Qwen3Model::from_safetensors_with_runtime`.
   - Qwen3.5 dense and Qwen3.5/Qwen3.6 MoE call
     `Qwen35Model::from_safetensors_with_options` /
     `from_safetensors_with_runtime`.
   - DeepSeek V4 builds `DeepseekRuntimeConfig::from_model_dir`, then calls
     `DeepseekModel::from_safetensors`.
5. Each model creates a `cuda_kernels::prelude::DeviceContext` with
   `DeviceContext::new`, loads config, mmaps safetensors through
   `infer/src/weight_loader.rs`, deserializes shards, detects quantization, and
   uploads weights to `DeviceMatrix` / `DeviceVec`.
6. `Scheduler::<M>::new` validates the scheduler/model contract, creates one
   model state per slot, sets contiguous fallback length and KV dtype, computes
   runtime workspace bytes, sizes `PagedKVPool::with_format`, creates radix
   cache state, and lazily creates decode/prefill contexts.
7. `warmup_cuda_graphs` optionally captures decode graph buckets by calling the
   model decode path through the same `ModelForward` surface.

### Qwen3 Loader

`infer/src/model/qwen3/weights.rs` is the Qwen3 CUDA loader. It:

1. Creates `DeviceContext`.
2. Resolves local/HF/GGUF source.
3. For GGUF, extracts config metadata and loads via `from_gguf`; TP GGUF is
   rejected.
4. For safetensors, reads `Config::from_file`, calls
   `common::load_safetensors`, then `common::deserialize_shards`.
5. Builds `QuantLoadConfig` from the model path.
6. If TP is requested, validates `TpLoadContext` and shards Q/K/V/O and MLP
   matrices by column/row.
7. Loads embeddings, optional LM head, per-layer RMS norms, Q/K/V/O attention
   weights, MLP gate/up/down, final norm, and precomputed RoPE cache.
8. Creates a `LayerCommunicator`.

### Qwen3.5 / Qwen3.6 Loader

`infer/src/model/qwen35/weights.rs` is the hybrid Qwen3.5 loader. It:

1. Creates `DeviceContext`.
2. Loads `Config35` from config/GGUF/safetensors.
3. Loads safetensors through `weight_loader`, with quant detection.
4. Supports TP sharding for dense Qwen3.5, but rejects quantized TP and rejects
   Qwen3.6 MoE under TP.
5. Loads full-attention layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`,
   `q_norm`, `k_norm`) and linear/GDR layers (`in_proj_qkv`, `in_proj_z`,
   `in_proj_b`, `in_proj_a`, `conv1d`, `dt_bias`, `A_log`, linear norm,
   `out_proj`).
6. For each MLP, uses dense `MLP::load_with_quant_config` unless
   `config.is_moe_layer(i)`.
7. For MoE layers, calls `qwen35/moe.rs::load_moe_mlp`, which loads router,
   per-expert or stacked+fused expert gate/up/down weights, shared expert
   gate/up/down, and shared-expert router. The CUDA MoE path is currently
   single-GPU and correctness-first.
8. Precomputes Qwen3.5-scaled RoPE cache and creates `LayerCommunicator`.

GGUF Qwen3.5 is dense-only today. Qwen3.6 MoE GGUF is rejected.

### DeepSeek V4 Loader

`infer/src/model/deepseek/config.rs` and `deepseek/weights.rs` build DSv4. The
sequence is:

1. `DeepseekRuntimeConfig::from_model_dir` reads `config.json`, then loads TP,
   EP, and multi-axis rank layout from env.
2. `validate_current_axis_support` and `validate_sglang_path_claim` fail closed
   for unsupported SGLang-best-practice claims.
3. `DeepseekModel::from_safetensors` validates the checkpoint manifest.
4. `DeepseekModel::from_config` creates `DeviceContext` and a
   `LayerCommunicator` wired from TP/EP/axis config.
5. Safetensors are mmaped through `common::load_safetensors`.
6. Top-level embedding, norm, and LM head are loaded with
   `load_dsv4_matrix_raw` / `load_tensor_1d`.
7. When layer loading is enabled, `load_layer_weights` loads the V4 attention,
   hyper-connection, MoE, MTP, block-scaled FP8/FP4, DeepGEMM side tensors, and
   FlashMLA metadata-dependent components.

DSv4 runtime gates matter. FlashMLA prefill and decode default on only when real
FlashMLA symbols are linked. The explicit SGLang-best-practice profile requires
FP8 KV, FlashMLA prefill/decode, shared FP8 KV pool, EAGLE/internal-MTP with
accepted drafts, full decode graph, direct GPU prefix attach, native DeepEP,
DeepGEMM, and token-owned distributed row metadata.

## Existing Step Execution

The legacy scheduler owns the model states, decode/prefill contexts, and
device-side `PagedKVPool`. `infer-core` now owns only the host plan and host
page accounting; the CUDA executor must recreate the legacy execution slice
behind `BackendExecutor`.

### Prefill

Legacy prefill starts in `infer/src/scheduler/cuda/prefill.rs`.

1. Admission chooses a slot, prompt chunk, and `start_pos`.
2. Prefix handling may attach retained GPU prefix pages or fall back to
   recompute depending on model capability.
3. `prepare_prefill_batch` builds `PrefillBatchRequest { slot_idx, tokens,
   start_pos, total_tokens }`.
4. If the model supports paged prefill and the pool is active with
   `page_size == 16`, the model runs the paged path:
   - Qwen3 calls `prepare_paged_prefill_batch`, then
     `run_prefill_paged_batch_sync` or async launch through
     `Qwen3PrefillContext`.
   - Qwen3.5 calls `prefill_forward_paged_batch` with `PagedPrefillBuffers35`.
   - DSv4 calls `prefill_batch_chunks` after preparing pool pages; FlashMLA
     prefill is gated through the DSv4 model code.
5. If the paged path is unavailable, contiguous state is filled first and later
   migrated with `GenerationState::migrate_kv_range_to_paged`.
6. On completion, the scheduler samples the first token from prefill logits and
   moves the request to decode.

The low-level kernels are in `infer/src/ops/attention.rs` and
`crates/cuda-kernels`:

- `prefill_attention_paged_batch` for Qwen3 HD128 paged prefill.
- `prefill_attention_paged_run_hd256` for Qwen3.5 HD256 paged prefill.
- `prefill_attention_paged_prep_cuda`,
  `prefill_attention_paged_prep_hd256_cuda`, and TileLang AOT
  `tilelang_batch_prefill_paged_*_run_cuda` symbols.
- DSv4 FlashMLA sparse prefill through
  `arle_flashmla_sm90_sparse_prefill_fwd` plus CSA/HCA index and packing
  helpers.

### Decode

Legacy decode starts in `infer/src/scheduler/cuda/decode.rs`.

1. The scheduler collects one `last_token` per decoding slot.
2. It allocates one token in `PagedKVPool` for each decode row.
3. It lazily creates the model decode context with
   `model.create_decode_context(max_batch_size, max_seq_len, &paged_kv_pool)`.
4. It calls `model.forward_decode_batch_with_request` with
   `DecodeBatchRequest { tokens, slot_indices, distributed_shards }` for DSv4
   or `forward_decode_batch` for non-distributed cases.
5. Qwen3/Qwen3.5 prepare TileLang metadata through
   `DecodeContextOps::update_metadata` and `plan_attention`, then run
   model-specific batched decode.
6. DSv4 goes through `forward_decode_batch_internal`; on eligible paths it
   carries token-owned distributed row metadata into native DeepEP and body
   graph checks.

The Qwen decode kernels are:

- `decode_prep_paged` / `decode_prep_paged_hd256` for QK norm, RoPE, and
  writing K/V into the page selected by the pool metadata.
- TileLang BF16 paged decode
  `tilelang_batch_decode_paged_hd128_*_run_cuda` and
  `tilelang_batch_decode_paged_hd256_*_run_cuda`.
- Quantized decode attention for INT8/FP8/INT4 through
  `decode_attention_*` wrappers and `PagedKVPool` scale/norm buffers.

The DSv4 decode kernels are:

- V4 incremental attention and MoE code in `deepseek/weights.rs`,
  `deepseek/mlp.rs`, and `deepseek/batch_decode.rs`.
- FlashMLA sparse FP8 decode through
  `arle_flashmla_sm90_sparse_decode_fwd`.
- FlashMLA index and packing helpers:
  `dsv4_flashmla_decode_build_indices_*`,
  `dsv4_flashmla_pack_one_sw_token`,
  `dsv4_flashmla_pack_compressor_rows`, `arle_flashmla_csa_pack_kv`,
  and `arle_flashmla_csa_build_indices`.
- DeepGEMM / native DeepEP MoE routes through the DSv4 MoE layer and
  `LayerCommunicator`.

### Sampling and Async Readback

Sampling is GPU-side in `infer/src/ops/sampling.rs`, backed by
`crates/cuda-kernels/csrc/misc/sampling.cu`.

Greedy decode uses:

1. `argmax_batch_logprob_launch` over `decode_ctx.logits_batch`.
2. `decode_ctx.stage_sampled_tokens_for_next_step` to make the sampled token
   available to the next decode input staging path.
3. `decode_ctx.start_greedy_readback_async`.
4. Later `model.sample_batch_greedy_readback`, which returns `None` until the
   async D2H slot is ready.

Non-greedy sampling uses `gpu_sample_launch` / `gpu_sample_readback`. R6a should
stay greedy only. General sampling needs per-slot sampling parameters to cross
or be stored below the seam.

Legacy overlap is the same order the clean engine now encodes:

1. At the top of a scheduler tick, poll pending decode/prefill readback.
2. If not ready, keep the pending handle and return.
3. If ready, apply sampled tokens on the host.
4. Build and launch the next plan while the previous GPU step already ran.

`CudaInflight` should therefore become a real pending CUDA step containing the
stage kind, slot indices, async sampling slot, prefill context/event, and enough
row metadata to build `StepOutput` after readback.

## New-Seam Impedance

The core constraint still holds: `infer-core` must see only host data. Device
tensors, CUDA streams, logits, NCCL groups, and C++/CUDA pointers remain inside
`infer-cuda` and lower seams.

### Host CudaKvPool vs Device PagedKVPool

`infer-cuda::CudaKvPool` already implements the host `KvPool` seam. It owns
logical page ids, slot page tables, slot lengths, slot epochs, and prefix
retain counts. Existing CUDA numerics, however, read and write
`cuda_kernels::PagedKVPool`, which owns device K/V buffers, scale/norm buffers,
free lists, page tables, slot attach counts, and slot epochs.

For R6a, use a lockstep mirror:

- Construct host `CudaKvPool` and device `PagedKVPool` with the same
  `num_slots`, `page_size`, and page count.
- Both allocators are LIFO and yield page id 0 first, so no-prefix,
  single-slot and simple multi-slot paths can keep page ids identical.
- Before submit, assert for each active slot that host
  `kv.page_indices(slot)` and `kv.seq_len(slot)` match the executor's device
  pool table.

For R6c and later, lockstep allocation is not strong enough. Prefix retention,
eviction, and free-slot happen in `infer-core` after executor poll returns.
The executor does not receive explicit `retain_pages`, `release_pages`, or
`free_slot` callbacks. It can infer active slot page tables from `kv`, but it
cannot directly observe cache-only retained pages. A robust CUDA page bridge
therefore needs one of:

- make the device pool page ids completely host-authoritative, and write into
  host-provided page ids without using an independent device allocator; or
- add a host-only lifecycle hook in the seam so executor and `CudaKvPool` can
  mirror retain/release/free exactly; or
- make `CudaKvPool` own a backend-private bridge handle shared with
  `CudaExecutor`, so host retain/release operations update device-retain state
  without exposing device tensors to `infer-core`.

This is the main hard wrap point. The current host seam is sufficient for R6a
and R6b, but prefix reuse with persistent GPU pages needs an explicit mirroring
strategy before R6c.

### Sequence-Length Semantics

`infer-core` builds `ForwardPlan`, then calls `allocate_for_plan`, then submits
the plan. Therefore:

- `PrefillRow.start_pos` is the materialized cache length before this chunk.
- `DecodeRow.kv_seq_len` is the materialized cache length before appending the
  decode token.
- `kv.seq_len(slot)` observed inside `submit` is already advanced by this
  plan's allocation.

This matches legacy CUDA decode, which allocates the append token before
`prepare_decode_context`; TileLang metadata uses `pool.seq_len(slot) - 1` as the
write/read position. R6a should assert:

- for prefill: `row.start_pos + row.tokens.len() == kv.seq_len(row.slot)` after
  host allocation, unless prefix/COW logic changes the table;
- for decode: `row.kv_seq_len + 1 == kv.seq_len(row.slot)`.

Any mismatch is a seam contract bug, not a CUDA numerics bug.

### ForwardPlan Mapping

Mapping is direct for the basic modes:

- `ForwardMode::Prefill`: convert each `PrefillRow` to
  `PrefillBatchRequest`.
- `ForwardMode::Decode`: collect `DecodeRow.last_token` and `DecodeRow.slot`
  into `DecodeBatchRequest` / `forward_decode_batch`.
- `ForwardMode::Mixed`: Qwen3 and Qwen3.5 can map to `MixedBatchRequest` when
  their model gates allow it; otherwise run decode first then prefill as a
  staged fallback inside one `CudaInflight`.
- `TargetVerify` and `DraftExtend`: defer until the DSv4/EAGLE and Qwen spec
  paths are ported.

DSv4 is not fully represented by today's `ForwardPlan`: its legacy
`DecodeBatchRequest` carries `distributed_shards` and token-owned row metadata.
The lower-seam `Communicator` can own NCCL/DeepEP tensors, but the plan still
needs enough host row ownership data once DSv4 distributed decode is brought
over. R6a/R6b avoid this by staying single-rank Qwen.

### Sampling Params

`ForwardPlan` does not carry per-slot sampling params. R6a can hard-code greedy
because both the test and the objective are greedy parity. General CUDA serving
will need either:

- per-slot sampling params in engine-core active request state and plan rows; or
- executor-private request state established at admission.

This is the same deferred seam gap already seen on Metal.

### Slot Release and Reuse

The engine frees a slot after `StepOutput` is applied. `BackendExecutor` has no
release notification. For R6a, the executor can detect reuse from
`kv.seq_len(slot) == 0` and `slot_epoch` changes on the next submit, then reset
the matching model state. That is sufficient for one-slot greedy.

For real prefix reuse and DSv4 metadata, a release hook or shared host/backend
pool bridge is safer. Otherwise backend-private recurrent state, FlashMLA
metadata, and retained page lifetime can silently diverge from host lifecycle.

### CUDA Graph

Legacy graph capture is stateful:

- Qwen3 graph capture is in `GenerationStateBase` / `CudaGraphState` and
  decode contexts; it depends on stable buffer pointers and batch shape.
- Qwen3 prefill graph is explicitly not a default because shape churn regressed
  c=4/c=8/c=16.
- Qwen3.5 uses piecewise graph capture for consecutive linear-attention groups.
- DSv4 reports only piecewise support today and the latest pod evidence shows
  body graph capture at zero because synthetic warmup does not materialize the
  required compressed/FP8/FlashMLA cache substrate.

R6a should disable graph or use the existing eager path. GraphRunner should be
ported only after eager parity is proven.

### NCCL / TP / EP / DeepEP

`LayerCommunicator` already has TP, EP, request-token-sync, attention-DP,
attention-CP, overlap TP, and native DeepEP fields. The new lower seam
`Communicator { type Tensor; all_reduce; all_to_all; send_recv }` is the right
shape for the executor/model internals, but DSv4 needs more than the three
method names:

- topology/axis metadata;
- request row ownership;
- token-owner groups;
- DeepEP buffer lifetime;
- graph capture safety for NCCL and DeepEP;
- separate compute and communication streams for FlashMLA TP overlap.

Keep these backend-internal, but do not claim DSv4 SGLang-best-practice until
the startup contract and request trace prove the path.

## CudaExecutor Real Shape

`infer-cuda` should grow a CUDA feature and depend on `cuda-kernels` under that
feature. It must not depend on the legacy `infer` crate.

The real executor should own:

- `DeviceContext` through the loaded model.
- A model enum or generic model wrapper for the ported model implementations.
- `Vec<State>` with one state per configured slot.
- One device `PagedKVPool` or host-authoritative device page table bridge.
- Lazy decode context sized by `SchedulerConfig.num_slots` and max sequence.
- Lazy prefill context for async prefill models.
- A greedy sampler path using the existing sampling kernels.
- Optional `CudaGraphRunner` state, initially disabled for R6a.
- Optional `CudaCommunicator` / NCCL groups, initially single-rank no-op.

`submit(&ForwardPlan, &mut dyn KvPool)` should:

1. Reject unsupported modes for the current stage.
2. Reconcile slot epochs and host page table with backend-private state.
3. Validate `start_pos` / `kv_seq_len` against host `kv.seq_len`.
4. For prefill rows, call the ported model's prefill batch path.
5. For decode rows, call the ported model's decode batch path.
6. Launch greedy argmax/logprob on the logits.
7. Return `CudaInflight` with pending event/readback state instead of blocking.

`poll(CudaInflight)` should:

1. Query the async prefill event or greedy D2H readback slot.
2. Return `PollResult::NotReady(inflight)` if CUDA reports not ready.
3. On ready, read token ids/logprobs to host and return
   `StepOutput { tokens: Vec<SlotToken> }`.
4. Never expose logits or device buffers through `StepOutput`.

## Port / Extraction Footprint

The port should reuse code, not re-derive numerics. The needed extraction is:

- model configs and tensor-name contracts from `crates/qwen3-spec`,
  `crates/qwen35-spec`, and `crates/deepseek-spec`;
- `model_source`, safetensors/GGUF helpers, `weight_loader`, quant load config,
  and TP load context utilities;
- model math from `infer/src/model/qwen3`, `qwen35`, and `deepseek`;
- CUDA op wrappers from `infer/src/ops`, backed by `crates/cuda-kernels`;
- `LayerCommunicator` and NCCL/DeepEP wrappers into backend-internal CUDA
  communicator code;
- CUDA graph helpers after eager parity.

Do not pull in `infer` as a dependency. If a utility is shared by both old and
new temporarily, move or copy the minimal tested code into a new crate/module
under the clean tree and keep the old tree serving until the parity gate passes.

## Staged Implementation Order

### R6a: Single-slot Qwen3 greedy on V100

Target: one slot, Qwen3 dense, greedy, no graph, no TP/NCCL, no prefix reuse, no
mixed batch.

Implementation:

1. Add `cuda`/`no-cuda` feature plumbing to `infer-cuda`.
2. Add `cuda-kernels` as a CUDA-only dependency.
3. Port the minimal Qwen3 config/loader/ops/model pieces needed for
   safetensors Qwen3 on V100.
4. Add `CudaExecutor::from_model_path_qwen3_single_slot`.
5. Own one state and a device `PagedKVPool` with page ids lockstep-mirrored to
   `CudaKvPool`.
6. `submit` accepts exactly one prefill row with `start_pos == 0`, or one
   decode row for slot 0 after prefill.
7. Run the existing Qwen3 prefill/decode kernels and greedy argmax.
8. `poll` returns one real token.

Verification:

- ignored CUDA test compares the full generated greedy token sequence against
  the legacy CUDA scheduler on the same V100, model, prompt, and seed;
- no CUDA graph;
- no perf claim beyond "real forward reaches parity".

### R6b: Qwen3 multi-slot batched decode

Add multiple slots and decode rows. Keep prefill simple, but batch decode rows
through the Qwen3 `forward_decode_batch` path and greedy readback. Verify:

- single-slot sequence parity stays green;
- two or more concurrent requests match the legacy scheduler on deterministic
  greedy output;
- host/device page tables remain identical across slot free/reuse and
  preemption.

### R6c: Chunked prefill and prefix reuse

Port chunked prefill and radix-attached prefixes for Qwen3 first.

Required before implementation:

- decide the host/device page lifecycle mirroring strategy;
- ensure cache-only retained pages are not recycled or overwritten in the
  device pool;
- support `PrefillRow.start_pos > 0`;
- map attached host page ids to real device K/V pages before suffix prefill.

For Qwen3 full-attention, K/V pages are sufficient. For Qwen3.5 hybrid and
DSv4, K/V pages alone are not sufficient.

### R6d: Qwen3.5 dense / hybrid

Port Qwen3.5 CUDA forward after Qwen3 works:

- full-attention + linear/GDR layers;
- recurrent state and chunkwise prefill;
- Qwen3.5 paged prefill/decode HD256 paths;
- prefix behavior must preserve the existing hybrid downgrade rules when GDR
  state cannot be restored.

Qwen3.6 CUDA MoE remains separate unless a single-GPU Qwen3.6 CUDA target is
explicitly licensed. Current code has a correctness-first single-GPU MoE path
but it is not the canonical CUDA production target in this R6 brief.

### R6e: DSv4 / FlashMLA on H20

Port DSv4 only after Qwen proves the seam and after the DSv4 pod blockers are
explicitly handled.

Scope:

- `DeepseekRuntimeConfig`, DSv4 loader, layer communicator, DeepGEMM, native
  DeepEP, FlashMLA sparse prefill/decode, shared FP8 KV pool, and internal
  MTP/EAGLE verifier/draft paths;
- host row ownership in or adjacent to `ForwardPlan`;
- direct GPU prefix attach or a DSv4 metadata snapshot/restore path;
- full decode graph capture using real cache substrate, not synthetic warmup.

Verification must report TTFT, TPOT, E2E, output throughput, EAGLE acceptance,
and trace fields proving hot-cache attach when the hot-cache target is claimed.

### R6f: GraphRunner, Communicator, mixed/spec polish

Only after eager parity:

- CUDA graph bucket capture/replay for Qwen decode;
- NCCL/TP all-reduce path for Qwen;
- DSv4 graph and DeepEP/NCCL capture safety;
- mixed decode+prefill where model gates allow it;
- `TargetVerify` and `DraftExtend`.

## Pod Build and Verification Recipe

Local Mac can do source review and no-CUDA type checks only. It cannot build or
run real CUDA because there is no nvcc/GPU.

### Local, after code lands

Use local checks only to catch Rust feature mistakes:

```bash
cargo check -p infer-plan -p infer-seam -p infer-core
CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,no-cuda
```

Do not treat Mac CUDA typecheck as correctness or performance evidence.

### V100 Qwen pod

Use the V100 for R6a/R6b Qwen parity.

```bash
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=70
export CMAKE_CUDA_ARCHITECTURES=70

cargo build --release -p infer-cuda --features cuda
cargo test --release -p infer-cuda --features cuda \
  r6a_qwen3_single_slot_matches_legacy_cuda \
  -- --ignored --nocapture --test-threads=1
```

The test should:

- load the same Qwen model in the legacy CUDA engine and new
  `Engine<CudaExecutor, CudaKvPool>`;
- use the same prompt and greedy sampling;
- compare the full token-id sequence, not just first token;
- print both sequences.

For G4 perf once correctness is green:

```bash
scripts/bench_guidellm.sh r6-qwen-v100-legacy --model <same-qwen-model>
scripts/bench_guidellm.sh r6-qwen-v100-clean  --model <same-qwen-model>
```

Report raw TTFT, ITL/TPOT, output tok/s, request/s, and delta percent against
the pre-rewrite baseline. Use the same binary shape, model, prompt profile, and
concurrency envelope.

### H20 DSv4 pod

Use H20 only for R6e DSv4/FlashMLA.

```bash
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=90
export CMAKE_CUDA_ARCHITECTURES=90
export CUDARC_CUDA_VERSION=12080

./scripts/dsv4_toolchain.sh
cargo build --release --features cuda,nccl
```

Then run a correctness gate before any bench:

- single request, short prompt, greedy, legacy vs clean sequence parity;
- longer prompt with `max_tokens >= 32`, because max-token-1 smoke cannot
  validate decode TPOT or EAGLE behavior;
- request trace confirms the intended DSv4 path, including FlashMLA, DeepEP,
  EAGLE acceptance, body graph, and prefix attach fields when those are
  claimed.

For G4:

```bash
scripts/bench_guidellm.sh r6-dsv4-h20-legacy --model <dsv4-model>
scripts/bench_guidellm.sh r6-dsv4-h20-clean  --model <dsv4-model>
```

Compare against the pre-rewrite DSv4 baseline with the same workload. The DSv4
SGLang-alignment rule applies: TTFT, TPOT, E2E, and output throughput must be
reported together. A raw target-model TPOT or a cold single request is not
enough.

## Smallest First Implementation Step

The smallest useful first step is:

> `CudaExecutor` owns a ported Qwen3 dense CUDA model loaded from a safetensors
> model path, one slot state, and a device `PagedKVPool` lockstep-mirrored to
> `CudaKvPool`; `submit` accepts either one `PrefillRow { slot: 0, start_pos:
> 0 }` or one `DecodeRow { slot: 0 }`, runs the existing Qwen3 CUDA
> prefill/decode kernels, launches greedy argmax, and `poll` returns the real
> sampled token. The ignored V100 test compares a 16-token greedy sequence
> against the legacy CUDA scheduler on the same model and prompt.

This is small enough to debug on V100, but it proves the core seam with real CUDA
tokens.

## Hard-to-Wrap Points to Track

1. **PagedKVPool ownership.** Existing CUDA model code wants to allocate and
   retain pages inside `PagedKVPool`. The new engine already allocated host page
   ids before `submit`. Lockstep allocation works only for no-prefix early
   stages; prefix reuse needs a host-authoritative page bridge or lifecycle
   hook.
2. **Cache-only retained pages.** `BackendExecutor` sees active slot tables but
   not `retain_pages` / `release_pages` events. This is risky for GPU prefix
   cache lifetime unless solved before R6c.
3. **DSv4 prefix attach is not just pages.** The latest DSv4 evidence shows
   radix hits still recompute prefill because DeepSeek lacks direct GPU prefix
   attach and metadata restore. R6e cannot claim hot-cache without fixing that.
4. **CUDA graph capture.** DSv4 body graph currently captures zero graphs under
   synthetic warmup because cache substrate is missing. R6 should not carry this
   blocker silently into the clean engine.
5. **NCCL/DeepEP topology.** The lower seam can hide tensors, but DSv4 still
   needs row ownership, token-owner groups, native DeepEP buffers, and graph
   safety. This likely requires additional host plan metadata for DSv4.
6. **Sampling params.** R6a hard-codes greedy. General serving needs per-slot
   sampling parameters below or through the plan.
7. **No legacy `infer` dependency.** The implementation must port/extract the
   minimum tested code into clean crates. A dependency from `infer-cuda` back to
   `infer` would recreate the old skeleton and block cutover.

## Deferred

- DSv4 hot-cache prefix metadata restore.
- DSv4 full decode CUDA graph capture/replay.
- NCCL/DeepEP graph safety.
- General non-greedy sampling.
- Mixed/speculative modes.
- Tiered KV / disaggregated KV movement.
