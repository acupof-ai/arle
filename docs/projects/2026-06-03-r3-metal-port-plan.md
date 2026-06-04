# R3 Metal Port Plan: Existing MLX Forward Into The New Seam

## Goal

R3 makes the clean engine runnable on Metal by wrapping the existing tested MLX
Qwen3.5/Qwen3.6 forward path behind `infer-seam::BackendExecutor`. This is a
port of the current numerics and kernel call sequence, not a re-derived model
implementation.

This document is investigation and integration plan only. The first
implementation tranche should stay intentionally small: one model, one slot,
greedy sampling, real tokens, and a correctness comparison against the legacy
Metal path before any packed batching or prefix-reuse optimization is enabled.

## Existing Metal Path

The serving path for the AI-PC target is in `infer/src/backend/metal/`. For
Qwen3.5 and Qwen3.6 MoE, the active high-performance route is the Qwen35 C++
compiled model in `infer/src/backend/metal/qwen35.rs` backed by FFI declarations
in `crates/mlx-sys/src/lib.rs` and implementation in
`crates/mlx-sys/src/mlx_qwen35_model.cpp`.

`Qwen3.5` and `Qwen3.6` are represented by `MetalModelArch::Qwen35`. Plain
`Qwen3` has a separate Rust/MLX path and is not the first R3 target.

### Load, config, and tokenizer resolution

`MetalBackend::load` in `infer/src/backend/metal.rs` performs this sequence:

1. Log MLX runtime diagnostics and apply Metal runtime limits.
2. Resolve the user path or Hugging Face id through
   `ResolvedModelSource::resolve`.
   - Local directories are used directly.
   - HF ids are resolved through the HF cache helper.
   - GGUF inputs are detected with `try_open_gguf`.
   - GGUF runtime assets, such as `config.json` and tokenizer JSON, may come
     from a sidecar/runtime-assets directory rather than the GGUF file itself.
3. Load the tokenizer with `ResolvedModelSource::load_tokenizer`.
4. Load model config.
   - Safetensors path: `load_metal_config(resolved_path)` reads `config.json`.
   - GGUF path: `load_gguf_metal_config`, which uses either sidecar
     `config.json` or `load_metal_config_from_gguf`, then applies GGUF metadata
     overrides.
5. Dispatch weight loading by architecture.
   - `MetalModelArch::Qwen3`: `load_qwen3_metal_weights`; GGUF is rejected.
   - `MetalModelArch::Qwen35`: `load_qwen35_metal_weights` for safetensors, or
     `load_qwen35_metal_weights_from_gguf` for GGUF.
6. Store `MetalWeights` on the backend, then optionally load DFlash and MTP
   runtimes.
7. Store tokenizer, config, model root, and resolved source path.

For the first clean-engine tranche, the old backend should remain the reference
oracle. The new `infer-metal` crate should not depend on the old `infer` crate;
the necessary MLX wrappers, config parsing, and Qwen35 loader pieces must be
ported or extracted.

### Safetensors weight loading

Safetensors are loaded by `infer/src/backend/metal/loader.rs`:

1. `load_tensor_map(model_dir)` discovers shards from
   `model.safetensors.index.json` or by globbing `*.safetensors`.
2. Each shard is loaded through the MLX bridge function
   `mlx_load_safetensors`, wrapped by `super::mlx::load_safetensors`.
3. The resulting tensor map is keyed by HF tensor name.
4. Dense linear weights are transposed and materialized with MLX `eval`.
5. Quantized MLX weights are recognized by `.scales` tensors and represented
   as `WeightTensor::Quantized { w, scales, biases, group_size, bits }`.
6. Embeddings may be dense or quantized. Tied `lm_head` can be built by
   transposing the embedding.

`load_qwen35_metal_weights` in `qwen35.rs` then:

1. Detects the tensor prefix among `language_model.model`,
   `model.language_model`, and `model`.
2. Loads embedding, final norm, and `lm_head` or ties `lm_head` to embedding.
3. Iterates layers according to `layer_types` from config.
4. For full-attention layers, loads `q_proj`, `k_proj`, `v_proj`, `o_proj`,
   `q_norm`, and `k_norm`.
5. For linear/GDR layers, loads fused or split `qkvz`, `ba`, `conv1d`,
   `A_log`, `dt_bias`, norm, and output projection weights.
6. Loads dense MLP layers or Qwen3.6 MoE layers, including router weights,
   expert stacks, shared experts, and shared-expert gate.
7. Builds `Qwen35MetalWeights`.
8. Unless `METAL_NO_CPP` is set, attempts `CppQwen35Model::build`.

Qwen3.6 MoE uses the same Qwen35 loader, with the MoE config nested in
`MetalQwen35ArchConfig`. Router and expert quantization can use different bit
widths.

### GGUF weight loading

`load_qwen35_metal_weights_from_gguf` supports dense Qwen3.5 GGUF only today.
It rejects Qwen3.6 MoE GGUF. The GGUF route:

1. Requires a Qwen35 config.
2. Builds a `Qwen35LinearGgufLayout`.
3. Loads embedding and tied output head from GGUF tensors.
4. Loads final norm, full-attention layers, linear/GDR layers, and dense MLPs.
5. Converts supported GGUF formats into MLX affine, native packed, or dense
   `WeightTensor` variants.
6. Attempts `CppQwen35Model::build` with those weights.

The first R3 implementation should use safetensors from
`mlx-community/Qwen3.5-0.8B-MLX-4bit` for fast local iteration. GGUF should be
deferred until the safetensors route is correct.

## Qwen35 C++ Compiled Construction Sequence

`CppQwen35Model::build` is the exact construction sequence to preserve.

1. Allocate the compiled model:
   - `qwen35_compiled_new()`
2. Optionally disable the GDR Metal kernel:
   - `qwen35_compiled_set_gdr_metal_kernel_enabled(model, 0)`
3. Register every weight through a local `add_weight` helper:
   - dense: `qwen35_compiled_add_dense_weight`
   - MLX affine quantized:
     `qwen35_compiled_add_affine_weight`
   - GGUF packed:
     `qwen35_compiled_add_gguf_weight`
   - GGUF input-reordered packed:
     `qwen35_compiled_add_gguf_input_reordered_weight`
4. Set model config:
   - `qwen35_compiled_set_config(model, rope_theta, rms_norm_eps,
     num_attention_heads, num_key_value_heads, head_dim, rotary_dim,
     hidden_size)`
5. Enable Q/K gating:
   - `qwen35_compiled_set_qk_gate(model, 1)`
6. Register `lm_head` with `add_weight`.
7. Configure embedding and final norm:
   - dense embedding:
     `qwen35_compiled_set_embed_v2(model, embed_tokens, norm, lm_head_id)`
   - packed GGUF embedding:
     `qwen35_compiled_set_packed_embed_v2(model, embed_id, norm, lm_head_id)`
8. If `lm_head` is dense and a quantized tied embedding exists, register it and
   call:
   - `qwen35_compiled_set_embed_as_linear_v2(model, embed_id)`
9. For each layer, register layernorm pointers and MLP metadata.
10. For dense MLP layers, require row-merged gate/up projection. Register:
    - merged `gate_up_proj`
    - `down_proj`
11. For full-attention layers, register Q/K/V/O projections and call:
    - `qwen35_compiled_push_full_attn_v2(model, input_ln, post_ln, q_id, k_id,
      v_id, o_id, q_norm, k_norm, gate_up_id, gate_dim, down_id)`
12. For linear/GDR layers, register fused projections when present and call:
    - `qwen35_compiled_push_gdr_v2(model, input_ln, post_ln, qkvz_id,
      qkv_split.0, qkv_split.1, ba_id, ba_num_heads, conv1d_weight,
      conv_kernel, a_log, dt_bias, norm_weight, gdr_rms_eps, out_id,
      num_key_heads, key_dim, num_value_heads, value_dim, gate_up_id,
      gate_dim, down_id)`
13. If fused GDR projection inputs are missing or disabled, register the split
    projections and call:
    - `qwen35_compiled_set_separate_proj_v2(model, qkv_id, z_id, b_id, a_id,
      gate_id, up_id)`
14. If fused GDR projections are used but dense MLP split weights are still
    needed, call:
    - `qwen35_compiled_set_separate_mlp_v2(model, gate_id, up_id)`
15. For Qwen3.6 MoE layers, immediately after pushing the attention/GDR layer,
    call `qwen35_compiled_set_last_moe_mlp` through
    `register_qwen35_moe_layer`.
16. Finalize:
    - `qwen35_compiled_finalize(model)`
17. Own the raw handle in `CppQwen35Model`; `Drop` calls
    `qwen35_compiled_free`.

This sequence is the one to port. The old comment in
`mlx_qwen35_model.cpp` is only a simplified sketch; the current Rust builder
with the v2 calls is authoritative.

## Existing Step Execution

### Request state and KV ownership

The legacy runtime creates per-request `MetalRequestState`, which wraps a
`ResumableRequestState<Qwen35StepDriver>`.

`Qwen35StepDriver::new` allocates:

1. One K cache and one V cache per full-attention layer, each shaped
   `[1, num_kv_heads, kv_capacity, head_dim]`.
2. `kv_capacity` rounded up by `KV_CACHE_CHUNK` from
   `prompt_tokens.len() + max_new_tokens`.
3. GDR recurrent and convolution state arrays for linear-attention layers.
4. `Qwen35StepMode::Cpp` when the compiled C++ model exists and
   `AGENT_INFER_QWEN35_FORCE_RUST` is not set. This mode owns flat arrays:
   - `kv_flat = [k0, v0, k1, v1, ...]`
   - `gdr_flat = [state0, conv0, state1, conv1, ...]`
   - `session_active`, `n_kv`, `n_gdr`
5. `Qwen35StepMode::Rust` as a fallback.
6. Optionally, the old per-request `infer/src/backend/metal/kv_pool.rs`
   `MetalKVPool` for dual-write diagnostics. This old pool owns MLX arrays and
   is not the same as the new `infer-metal::MetalKvPool`.

The C++ route has a session model:

1. `Qwen35CppState::ensure_session_active` calls
   `qwen35_session_begin(model, kv_flat, n_kv, gdr_flat, n_gdr)`.
2. C++ stores the active session caches internally.
3. Step and prefill calls mutate the active session's K/V and GDR state.
4. `ensure_caches_drained` or cleanup calls `qwen35_session_end`, returning
   updated `kv_flat` and `gdr_flat` to Rust.

Only one C++ session can be active per compiled model handle. The legacy
runtime works around this by draining other sessions before scalar decode and
by using packed batch execution for concurrent rows.

### Prefill

For Qwen35 C++ prefill with more than one token:

1. Ensure capacity.
2. Build an int32 MLX token array shaped `[prompt_chunk_len]`.
3. Ensure the C++ session is active.
4. Call:
   - `qwen35_compiled_prefill_session(model, token_ids, prompt_len,
     cache_pos, out_logits)`
5. C++ sets:
   - `current_seq_len = prompt_len`
   - `current_cache_pos = cache_pos`
   - `current_last_logits_only` according to the prefill mode
6. C++ runs one forward over `[tokens] + session_kv_caches + session_gdr_states`.
7. C++ updates the active session caches and returns logits.
8. Rust calls `async_eval(&[&logits])`.
9. If old dual-write pool is enabled, Rust clones session K/V and writes the
   just-prefilled columns to that old pool.
10. Rust increments `cache_len` by `prompt_chunk_len`.
11. If this was the terminal prompt chunk, Rust samples from logits and
    materializes the sampled token with `item_i32`.

Scalar prefill falls back to repeated single-token `run_step` calls.

### Decode

The standard single-token decode path is:

1. Use the last committed token as input.
2. If a previous sampled token was kept as `pending_sampled`, commit its cache
   accounting first and prequeue the next step using the sampled MLX tensor as
   the input token.
3. Otherwise, build an int32 MLX token array shaped `[1]`.
4. Ensure the C++ session is active.
5. Call:
   - `qwen35_compiled_step_session(model, token_id, cache_pos, out_logits)`
6. C++ sets:
   - `current_cache_pos = cache_pos`
   - `current_batch_size = 1`
   - `current_seq_len = 1`
   - `current_last_logits_only = false`
7. C++ runs one forward over `[token] + session_kv_caches + session_gdr_states`,
   updates the active session caches, and returns logits.
8. Rust samples on GPU, calls `async_eval` on the sampled token, optionally
   prequeues the next step, and only then reads `item_i32`.

This double-buffering is important. It means the first clean implementation can
be synchronous for correctness, but a real R3 executor should let `poll`
materialize the token while `submit` keeps MLX work queued.

### Packed decode

The legacy c>=2 route is `execute_qwen35_packed_decode_batch` in
`runtime.rs`. It:

1. Groups compatible Qwen35 decode rows.
2. Builds or reuses `Qwen35PackedDecodeBatch`.
3. Packs row K/V and GDR arrays into batch-major arrays.
4. Uses left padding so rows with different sequence lengths can share one
   batch cache length.
5. Builds an additive attention mask when any row is left padded.
6. Builds per-row RoPE offsets:
   - `rope_offsets[row] = batch_cache_len - left_padding[row]`
7. Calls:
   - `qwen35_compiled_step_batch_packed(model, token_ids, batch_size,
     batch_cache_len, packed_kv_caches, n_kv, packed_gdr_states, n_gdr,
     attn_mask, rope_offsets, out_logits, out_packed_kv_caches,
     out_packed_gdr_states)`
8. Samples either batched or per-row depending on whether sampling parameters
   are uniform.
9. Unpacks updated state back to each row.

`qwen35_compiled_prefill_batch_packed` also exists. Its current invariant is
equal prompt lengths per row; variable-length prefill is a later follow-up.

### Sampling

Existing Metal sampling is in `infer/src/backend/metal/sampling.rs`.

Supported parameters are narrower than the pure `infer-plan::SamplingParams`:

- greedy when `temperature <= 1e-6` or `top_k == 1`
- otherwise categorical sampling from temperature-scaled logits
- `top_p < 1.0`, `min_p > 0.0`, repetition penalty, frequency penalty,
  presence penalty, and seeded sampling are rejected today

Sampling currently happens below the backend and returns a host scalar token.
This matches `StepOutput`, but the clean `BackendExecutor::submit` does not
receive per-slot `SamplingParams`. The first R3 tranche should hard-code greedy
sampling. Before non-greedy serving, the seam needs either:

1. a backend-side slot-parameter registration path, or
2. per-row sampling params in `ForwardPlan`.

This is contract friction, not a blocker for a one-row greedy correctness
tranche.

## Existing Metal KV Pool Versus Clean KV Pool

There are two different pools with similar names.

### Legacy MLX-owning pool

`infer/src/backend/metal/kv_pool.rs` owns MLX arrays. It stores token-level K/V
rows in flat per-layer buffers:

- `k_pool[layer]`: `[max_total_tokens, kv_dim]`
- `v_pool[layer]`: `[max_total_tokens, kv_dim]`

It has a pure Rust ledger for token-slot allocation and sharing, then MLX
methods:

- `write_kv` / `write_kv_slots`: scatter K/V rows into the pool
- `gather_kv` / `gather_kv_rows`: gather rows and reshape to
  `[1, num_kv_heads, seq_len, head_dim]`
- `flush`: force materialization to avoid unbounded lazy `slice_update` chains

The current Qwen35 path only dual-writes into this pool for diagnostics and
future cutover. The C++ `qwen35_compiled_step_session_paged` entry point accepts
pre-gathered K/V arrays, but the C++ implementation currently ignores those
arguments and behaves like `step_session`.

### Clean host-only pool

`crates/infer-metal::MetalKvPool` implements `infer_seam::KvPool`. It is
deliberately host-only:

- page size
- free page ids
- slot to page-id lists
- slot sequence lengths
- retained page reference counts
- `attach_pages`, `retain_pages`, `release_pages`, `truncate_slot`, `free_slot`

It does not own MLX arrays. That is correct for the clean seam. The device
storage must be owned by `MetalExecutor` or by backend-internal model state,
not by engine-core-facing signatures.

## New-Seam Impedance

### What `MetalExecutor` must own

The host-only seam means `ForwardPlan` and `KvPool` are not enough to execute
MLX kernels by themselves. `MetalExecutor` must own all Metal state:

- resolved model metadata and config
- tokenizer only if a new CLI/API route needs text input in this crate
- loaded Qwen35 weights
- `CppQwen35Model` raw handle
- per-slot execution state
- per-slot contiguous session K/V arrays for the first tranche
- per-slot GDR recurrent/conv arrays
- sampled-token/logits MLX arrays while a step is in flight
- eventual page-id to MLX K/V storage mapping

`infer-core` sees only host rows and page ids. `MetalExecutor` translates those
to MLX state internally.

### `ForwardPlan` mapping

`ForwardPlan` gives enough host data for the basic path:

- `PrefillRow { slot, tokens, start_pos, total_tokens }`
- `DecodeRow { slot, last_token, kv_seq_len }`

Mapping rules:

1. `PrefillRow.slot` selects or creates a `MetalSlotState`.
2. `PrefillRow.tokens` is the uncached prompt suffix to run now.
3. `PrefillRow.start_pos` is the already-materialized prefix length. For the
   no-prefix first tranche it must be `0`. For prefix reuse, `MetalExecutor`
   must import/materialize the prefix state before running the suffix.
4. `PrefillRow.total_tokens` is the full prompt length and should match
   `start_pos + tokens.len()` for the first unchunked implementation.
5. `DecodeRow.last_token` becomes the one-element MLX token input.
6. `DecodeRow.kv_seq_len` must match the slot state's materialized
   `cache_len`; otherwise the executor should fail loudly because the scheduler
   and backend state diverged.

`ForwardMode::Mixed` can be implemented after independent prefill and decode
paths exist. The first R3 tranche should accept exactly one row, either prefill
or decode, and return a clear error for unsupported mixed/multi-row plans.

### Page ids to MLX KV buffers

R1c prefix reuse made page ids meaningful above the backend:

- engine-core can attach retained pages to a new slot
- `ForwardPlan.prefill_rows[*].start_pos` can skip a cached prefix
- `kv.page_indices(slot)` returns the host page ids in logical order

For Metal, a page hit is only real if those page ids map to real MLX K/V
storage. The clean R3 design needs a backend-internal store:

```text
MetalExecutor
  MetalPageStore
    page_id -> per-full-layer K page
    page_id -> per-full-layer V page
  MetalSlotState
    slot
    cache_len
    kv_capacity
    session kv_flat/gdr_flat
    logical page ids currently materialized
```

The page size must match `KvPool::page_size()` in tokens. Each full page in
the host pool needs enough Metal storage for one block of K/V rows for every
full-attention layer.

There are two viable implementation strategies:

1. Materialize contiguous session arrays from host page ids before running the
   existing C++ `prefill_session` / `step_session`. This requires no immediate
   C++ read-source change, but it copies/gathers pages at reuse time.
2. Finish the paged C++ cutover so `qwen35_compiled_step_session_paged` and
   the corresponding prefill path actually read the provided gathered K/V
   arrays. This is the right performance destination, but it is not the
   smallest runnable first step.

The first tranche should use strategy 1 only for the trivial no-prefix case:
per-slot contiguous session arrays, no page reuse. The first prefix tranche can
materialize from pages. The later performance tranche can switch C++ attention
to read page-backed/gathered K/V directly.

### GDR state and prefix reuse

Qwen3.5/Qwen3.6 is not just full-attention KV. Linear/GDR layers carry recurrent
state and convolution state. A reused prefix must therefore restore both:

- full-attention K/V up to `matched_len`
- GDR recurrent and conv state at exactly `matched_len`

The old prefix snapshot path already captures `kv_flat`, `gdr_flat`,
`cache_len`, and `kv_capacity`. Host page ids alone are insufficient for a
Qwen35 prefix hit unless the backend has a corresponding GDR state snapshot for
that matched prefix.

This is the hardest part to wrap behind the host-only seam. The seam can remain
host-only, but `MetalExecutor` needs a backend-private side table keyed by
prefix identity or by page block sequence that stores the GDR state snapshot
associated with retained K/V pages. Without that, prefix reuse would silently
reuse attention KV while recomputing or mismatching linear-attention state,
which is not a valid Qwen35 state.

## `mlx-sys` Reuse and Additions

### Reuse

Existing FFI is enough for the first greedy single-slot tranche:

- `mlx_guard`
- `mlx_last_error`
- array constructors used by the old `MlxArray` wrapper
- `mlx_load_safetensors`
- `mlx_argmax` / `mlx_argmax_axis`
- `mlx_random_categorical` for later temperature sampling
- `mlx_async_eval`
- `qwen35_compiled_new`
- `qwen35_compiled_free`
- `qwen35_compiled_add_dense_weight`
- `qwen35_compiled_add_affine_weight`
- `qwen35_compiled_add_gguf_weight`
- `qwen35_compiled_add_gguf_input_reordered_weight`
- `qwen35_compiled_set_config`
- `qwen35_compiled_set_qk_gate`
- `qwen35_compiled_set_embed_v2`
- `qwen35_compiled_set_packed_embed_v2`
- `qwen35_compiled_set_embed_as_linear_v2`
- `qwen35_compiled_push_full_attn_v2`
- `qwen35_compiled_push_gdr_v2`
- `qwen35_compiled_set_separate_proj_v2`
- `qwen35_compiled_set_separate_mlp_v2`
- `qwen35_compiled_set_last_moe_mlp`
- `qwen35_compiled_finalize`
- `qwen35_session_begin`
- `qwen35_session_end`
- `qwen35_compiled_prefill_session`
- `qwen35_compiled_step_session`
- `qwen35_compiled_session_kv_clone`
- `qwen35_compiled_step_batch_packed`
- `qwen35_compiled_prefill_batch_packed`

### Additions or extractions

The likely non-FFI work before `infer-metal` can compile independently:

1. Port or extract the old `infer/src/backend/metal/mlx.rs` safe wrapper into a
   place the new crate can own without depending on `infer`.
2. Port the Qwen35 config structs and config loader, or move them into a shared
   model/Metal support crate.
3. Port the Qwen35 weight structs and safetensors loader.
4. Port `CppQwen35Model::build` exactly.
5. Add `mlx-sys` as an `infer-metal` dependency.

Likely FFI or C++ additions for real prefix-page reuse:

1. Make `qwen35_compiled_step_session_paged` actually consume provided K/V
   arrays, or add a new page-backed step entry point.
2. Add a prefill-session equivalent that can consume imported/gathered prefix
   K/V rather than only the active contiguous session.
3. Add backend-private helpers if page slabs are represented more efficiently
   than one MLX array per page.
4. Add or reuse snapshot export/import support for Qwen35 GDR state so a
   page-aligned prefix hit restores the full model state.

## Staged Implementation Plan

### R3a: single-slot greedy correctness

Objective: `infer-metal::MetalExecutor` runs real MLX Qwen35 forward for one
slot and returns one real sampled token per `ForwardPlan`.

Scope:

- model: `mlx-community/Qwen3.5-0.8B-MLX-4bit`
- weights: safetensors only
- rows: exactly one prefill row or exactly one decode row
- sampling: greedy argmax only
- KV: contiguous per-slot C++ session arrays
- prefix reuse: disabled; require `start_pos == 0` for first prefill
- batching: disabled
- DFlash/MTP/ngram/GGUF/Qwen3.6 MoE: disabled

Implementation shape:

1. Add `mlx-sys` to `infer-metal`.
2. Port the minimal MLX array wrapper needed by Qwen35 loading and stepping.
3. Port config parsing for Qwen35 safetensors models.
4. Port Qwen35 safetensors weight loading.
5. Port `CppQwen35Model::build` exactly.
6. Add `MetalExecutor::from_model_path(path)` or equivalent constructor that
   resolves a local/HF model path, loads config and weights, and builds the C++
   model handle.
7. Add backend-private `MetalSlotState`:
   - slot id
   - `cache_len`
   - `kv_capacity`
   - `kv_flat`
   - `gdr_flat`
   - session active flag
8. Implement `submit` for a one-row `Prefill` plan:
   - validate no active conflicting session
   - allocate slot state if absent
   - create MLX token array from `PrefillRow.tokens`
   - begin C++ session
   - call `prefill_session(tokens, tokens.len(), start_pos)`
   - run greedy argmax
   - submit MLX eval
   - store sampled token in `MetalInflight`
9. Implement `submit` for a one-row `Decode` plan:
   - validate `DecodeRow.kv_seq_len == slot_state.cache_len`
   - create `[last_token]`
   - call `step_session`
   - run greedy argmax
   - submit MLX eval
   - store sampled token in `MetalInflight`
10. Implement `poll` by materializing the sampled token and returning
    `StepOutput`.
11. Add a manual or ignored Metal correctness test that compares token ids with
    legacy `MetalBackend::generate_from_token_ids` on the same model and greedy
    params.

Exit evidence:

- a real prompt-token vector generates at least one matching token versus the
  legacy backend on `Qwen3.5-0.8B-MLX-4bit`
- `cargo check -p infer-plan -p infer-seam -p infer-core -p infer-metal` works
  on Mac without CUDA

### R3b: clean engine end-to-end on one slot

Objective: `Engine<MetalExecutor, infer_metal::MetalKvPool>` drives one
request to completion with real MLX tokens.

Scope:

- still one slot
- still greedy
- still no prefix reuse
- route through `infer-core::Engine::run_to_idle`

Required details:

- Decide how `max_tokens` and stop handling stay in `infer-core` while
  `MetalExecutor` only samples the next token.
- Keep `MetalExecutor` slot state alive across `submit` calls.
- On slot finish, provide an executor-side cleanup path tied to `kv.free_slot`
  or a new explicit backend slot release hook if needed.

Potential seam friction:

- `BackendExecutor` is not notified when engine-core frees a slot. Today the
  executor can observe `kv.seq_len(slot) == 0` on later submits, but there is
  no explicit `free_slot` callback to drain C++ sessions promptly. If this
  becomes awkward, add a host-only lifecycle hook rather than leaking MLX types.

### R3c: prefix-aware materialization

Objective: a prefix hit from `infer-core::RadixCache` produces a real Metal KV
reuse, not just skipped host tokens.

Scope:

- page size equals `KvPool::page_size()`
- `MetalExecutor` owns `MetalPageStore`
- on page publication, copy or alias the just-sealed K/V block into page store
- on prefix attach, materialize contiguous slot K/V arrays from page ids before
  running suffix prefill
- restore Qwen35 GDR state from a backend-private snapshot for the matched
  prefix

Invariant:

- do not release or overwrite Metal page storage while a host page is retained
  by the radix cache or attached to an active slot
- do not treat a host page hit as a hot cache hit until both K/V and GDR state
  are restored for the prefix

### R3d: packed decode and mixed batches

Objective: recover the legacy concurrent Metal serving path.

Scope:

- port `Qwen35PackedDecodeBatch`
- map `ForwardPlan.decode_rows` to packed rows
- preserve left-padding, additive mask, and per-row RoPE offset logic
- support `ForwardMode::Mixed` with decode rows plus one prefill row only after
  scalar prefill/decode parity is stable
- keep `qwen35_compiled_prefill_batch_packed` equal-length invariant until a
  later varlen prefill stage

### R3e: Qwen3.6 MoE canonical model

Objective: run the AI-PC canonical model,
`mlx-community/Qwen3.6-35B-A3B-4bit`, through the clean engine.

Scope:

- enable MoE layer registration via `qwen35_compiled_set_last_moe_mlp`
- preserve router/expert quantization handling
- apply wired-limit behavior equivalent to the legacy backend
- run a correctness smoke first, then a guideLLM or equivalent Metal bench
  against the legacy path

### Deferred

- kv_tier staged-prefix and disk/remote KV movement
- DFlash speculative blocks
- MTP speculative blocks
- ngram speculative decode
- GGUF serving in the clean engine
- Qwen3 non-Qwen35 path
- true paged C++ attention read-source cutover
- non-greedy sampling and full penalty/filter support

## Smallest First Implementation Step

Build a real greedy one-row prefill/decode executor:

> `MetalExecutor` owns a Qwen35 compiled model loaded from
> `mlx-community/Qwen3.5-0.8B-MLX-4bit`; `submit` accepts exactly one
> `PrefillRow` with `start_pos == 0` or one `DecodeRow`, runs the existing
> `qwen35_compiled_prefill_session` or `qwen35_compiled_step_session`, samples
> argmax on MLX logits, and `poll` returns a `StepOutput` with that token.

That is the smallest step that proves the clean host-only seam can drive real
Metal numerics. Everything else should be layered only after this token-level
parity check passes against the legacy `MetalBackend`.

## Hard Parts To Review Before Implementation

1. `BackendExecutor::submit` has no per-slot sampling params. Greedy can be
   hard-coded for R3a, but real sampling needs a host-only contract addition or
   a backend slot-registration path.
2. The old C++ model has one active scalar session per model handle. Multi-slot
   serving must use packed batches or carefully drain sessions.
3. Host page ids do not automatically imply live Metal K/V. `MetalExecutor`
   needs a page-id to MLX-storage map before R1c prefix hits become real Metal
   cache hits.
4. Qwen35 prefix reuse also needs GDR recurrent/conv state snapshots. K/V pages
   alone are not sufficient.
5. `qwen35_compiled_step_session_paged` currently ignores the provided K/V
   arrays. The first runnable tranche can avoid it; the real prefix/perf path
   cannot.
6. The clean `infer-metal` crate cannot depend on the legacy `infer` crate.
   The MLX wrappers, config structs, loader, and Qwen35 builder need to be
   ported or extracted without creating a circular dependency.
7. `infer-core` frees host slots through `KvPool::free_slot`, but the executor
   has no explicit slot-release callback. If prompt churn leaves stale MLX slot
   state, add a host-only lifecycle hook rather than exposing device tensors.
