# Metal Qwen3.5 C++ generate path: top_k=1 dropped, DFlash unsafe with GDR fallback

## Context

Commit `25681a2 fix(metal): qk-gate explicit instead of n_gdr heuristic`
landed alongside concurrent work that wired the GGUF Qwen3.5 weights into
the Metal C++ compiled model with `disable_gdr_metal_kernel = true` (ops
fallback because the custom GDR Metal kernel doesn't decode GGUF correctly
yet). A `codex review --commit 25681a2` pass surfaced two follow-on
regressions that the qk-gate change made reachable:

1. `metal_generate_qwen35` and `metal_generate` (Qwen3 path) routed
   through `cpp_model.generate(...)` whenever `cpp_model` was present, but
   the FFI signature `qwen35_compiled_generate(temperature)` only took
   temperature and used `temperature <= 1e-6f` as the greedy switch. For a
   request with `temperature > 0` *and* `top_k = 1` the Rust sampler
   greedy-picks while the C++ path falls into categorical sampling — same
   prompt + params produced different distributions depending on which
   path Metal happened to pick.
2. The `disable_gdr_metal_kernel = true` build still satisfies
   `cpp_model.is_some()`, so DFlash dispatch took the C++ speculative
   block path. DFlash needs both dense target embeddings *and* the GDR
   tape recorder; the GGUF-fallback model has neither, so the first
   speculative block throws inside `qwen35_compiled_verify_block_summary`
   instead of falling back to plain decode.

## Root Cause

Single-bit collapse — both regressions came from the C++ wrapper exposing
only `cpp_model.is_some()` to upstream callers. Anything that needed a
finer feature flag (greedy vs categorical, GDR-tape capable vs ops-only)
re-derived it from incidental signals (`temperature` threshold, model
struct presence) and got it wrong on at least one path.

## Fix

- `qwen35_compiled_generate` FFI gains an explicit `bool greedy`
  parameter; the C++ prefill + decode samplers use it instead of the
  internal `temperature <= 1e-6f` heuristic. Both Rust call sites
  (`metal_generate_qwen35` and the Qwen3 `metal_generate`) pass
  `params.temperature <= 1e-6 || params.top_k == 1` so `top_k = 1`
  remains greedy regardless of temperature.
- `CppQwen35Model` is no longer a tuple struct; it carries
  `gdr_tape_supported: bool`, set to `!disable_gdr_metal_kernel` at
  build. A new `qwen35_dflash_supported(weights)` helper checks both
  dense embedding and `cpp_model.supports_gdr_tape()`. Two DFlash
  dispatch sites — `metal_generate_qwen35` and `Qwen35StepDriver::new`
  in `request_state.rs` — gate the runtime through this helper, so
  GGUF requests with DFlash on disk fall back to plain decode instead
  of crashing on the first speculative block.

## Rule

When a wrapper exposes one flag (e.g. `is_some()`) but the wrapped object
has multiple capability dimensions (sampling shape, speculative-decode
support, embedding density), surface each capability as its own predicate
on the wrapper. Don't let upstream re-derive it from indirect signals —
that's how routing bugs hide until a new caller hits them.
