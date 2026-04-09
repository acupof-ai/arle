// mlx_engine.h — C API for the MLX C++ inference engine.
//
// Provides a simple opaque handle + function interface for Rust FFI.
// All MLX ops happen in C++ (same path as pybind11 → no C wrapper overhead).

#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to a loaded model.
typedef struct mlx_engine_model* mlx_engine_model_t;

/// Opaque handle to per-request decode state (KV cache + recurrent state).
typedef struct mlx_engine_state* mlx_engine_state_t;

/// Load a model from a directory (safetensors + config.json).
/// Returns NULL on failure. Error message written to `err_buf` if non-NULL.
mlx_engine_model_t mlx_engine_load(
    const char* model_dir,
    char* err_buf, int err_buf_len);

/// Free a loaded model.
void mlx_engine_free_model(mlx_engine_model_t model);

/// Create decode state for a new request.
mlx_engine_state_t mlx_engine_new_state(mlx_engine_model_t model);

/// Free decode state.
void mlx_engine_free_state(mlx_engine_state_t state);

/// Run one decode step: token_id → next_token_id (greedy).
/// Updates KV cache and recurrent state in-place.
/// Returns the sampled token ID.
int32_t mlx_engine_decode_step(
    mlx_engine_model_t model,
    mlx_engine_state_t state,
    int32_t token_id,
    float temperature);

/// Run prefill for a sequence of token IDs.
void mlx_engine_prefill(
    mlx_engine_model_t model,
    mlx_engine_state_t state,
    const int32_t* token_ids,
    int num_tokens);

#ifdef __cplusplus
}
#endif
