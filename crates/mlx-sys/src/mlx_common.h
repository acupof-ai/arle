#pragma once
#include "mlx/mlx.h"
#include "mlx/fast.h"
#include <cstdint>
#include <vector>
#include <optional>
#include <string>

using namespace mlx::core;

// ── Error handling ──────────────────────────────────────────────────────────
// Thread-local last error string. Functions that can fail set this and return
// nullptr. Rust checks for nullptr and calls mlx_last_error() to get the message.
inline thread_local std::string g_mlx_last_error;

static inline void mlx_clear_error() { g_mlx_last_error.clear(); }
static inline void mlx_set_error(const char* msg) { g_mlx_last_error = msg; }
static inline void mlx_set_error(const std::string& msg) { g_mlx_last_error = msg; }

extern "C" const char* mlx_last_error();

// Macro for wrapping functions that return mlx_array* — catches C++ exceptions
// and returns nullptr + sets thread-local error.
#define MLX_TRY_RETURN(...) \
    try { \
        mlx_clear_error(); \
        return (__VA_ARGS__); \
    } catch (const std::exception& e) { \
        mlx_set_error(e.what()); \
        return nullptr; \
    }

// Macro for wrapping functions that return scalars / non-array pointers and
// should fall back to a sentinel value on failure.
#define MLX_TRY_RETURN_VALUE(default_value, ...) \
    try { \
        mlx_clear_error(); \
        return (__VA_ARGS__); \
    } catch (const std::exception& e) { \
        mlx_set_error(e.what()); \
        return (default_value); \
    }

// Macro for wrapping void-returning functions so exceptions never cross the C ABI.
#define MLX_TRY_VOID(...) \
    try { \
        mlx_clear_error(); \
        __VA_ARGS__; \
        return; \
    } catch (const std::exception& e) { \
        mlx_set_error(e.what()); \
        return; \
    }

struct mlx_array;  // opaque

// Qwen3.5/3.6 sparse-MoE helper shared between the compiled model path and
// the Rust fallback. `qwen35_moe_block_forward_cpp()` is the internal
// array-native helper used by the compiled model path; the `extern "C"`
// wrapper remains for Rust callers.
array qwen35_moe_block_forward_cpp(
    const array& hidden,
    const array& router_w,
    const array& router_scales,
    const array& router_biases,
    int32_t router_bits,
    int32_t router_group_size,
    const array& expert_gate_w,
    const array& expert_gate_scales,
    const array& expert_gate_biases,
    const array& expert_up_w,
    const array& expert_up_scales,
    const array& expert_up_biases,
    const array& expert_down_w,
    const array& expert_down_scales,
    const array& expert_down_biases,
    int32_t expert_bits,
    int32_t expert_group_size,
    const array& shared_gate_w,
    const array& shared_gate_scales,
    const array& shared_gate_biases,
    const array& shared_up_w,
    const array& shared_up_scales,
    const array& shared_up_biases,
    const array& shared_down_w,
    const array& shared_down_scales,
    const array& shared_down_biases,
    const array& shared_gate_router_w,
    const array& shared_gate_router_scales,
    const array& shared_gate_router_biases,
    int32_t num_experts,
    int32_t top_k,
    bool norm_topk_prob);

// Fixed-M=16 causal SDPA specialization shared between the bridge FFI and the
// compiled Qwen3.5/Qwen3.6 verify path.
array batched_sdpa_2pass_cpp(
    const array& queries,
    const array& keys,
    const array& values,
    float scale,
    int32_t gqa_factor);

extern "C" mlx_array* qwen35_moe_block_forward(
    mlx_array* hidden,
    mlx_array* router_w,
    mlx_array* router_scales,
    mlx_array* router_biases,
    int32_t router_bits,
    int32_t router_group_size,
    mlx_array* expert_gate_w,
    mlx_array* expert_gate_scales,
    mlx_array* expert_gate_biases,
    mlx_array* expert_up_w,
    mlx_array* expert_up_scales,
    mlx_array* expert_up_biases,
    mlx_array* expert_down_w,
    mlx_array* expert_down_scales,
    mlx_array* expert_down_biases,
    int32_t expert_bits,
    int32_t expert_group_size,
    mlx_array* shared_gate_w,
    mlx_array* shared_gate_scales,
    mlx_array* shared_gate_biases,
    mlx_array* shared_up_w,
    mlx_array* shared_up_scales,
    mlx_array* shared_up_biases,
    mlx_array* shared_down_w,
    mlx_array* shared_down_scales,
    mlx_array* shared_down_biases,
    mlx_array* shared_gate_router_w,
    mlx_array* shared_gate_router_scales,
    mlx_array* shared_gate_router_biases,
    int32_t num_experts,
    int32_t top_k,
    bool norm_topk_prob);
// Shape = mlx::core::Shape (SmallVector<int> in v0.31+)

static inline array* to_arr(mlx_array* h) {
    return reinterpret_cast<array*>(h);
}
static inline const array* to_arr(const mlx_array* h) {
    return reinterpret_cast<const array*>(h);
}
static inline mlx_array* from_arr(array&& a) {
    return reinterpret_cast<mlx_array*>(new array(std::move(a)));
}
static inline Shape make_shape(const int32_t* s, size_t n) {
    Shape sh;
    for (size_t i = 0; i < n; i++) sh.push_back(s[i]);
    return sh;
}

// Dtype enum values — must match mlx::core::Dtype::Val
// bool_=0, uint8=1, uint16=2, uint32=3, uint64=4,
// int8=5, int16=6, int32=7, int64=8,
// float16=9, float32=10, bfloat16=12, complex64=13
static inline Dtype to_dtype(int32_t d) {
    switch (d) {
        case 0:  return bool_;
        case 1:  return uint8;
        case 2:  return uint16;
        case 3:  return uint32;
        case 5:  return int8;
        case 6:  return int16;
        case 7:  return int32;
        case 8:  return int64;
        case 9:  return float16;
        case 10: return float32;
        case 12: return bfloat16;
        case 13: return complex64;
        default: return float32;  // fallback
    }
}

static inline int32_t from_dtype(const Dtype& d) {
    // Use the underlying enum value
    if (d == bool_) return 0;
    if (d == uint8) return 1;
    if (d == uint16) return 2;
    if (d == uint32) return 3;
    if (d == int8) return 5;
    if (d == int16) return 6;
    if (d == int32) return 7;
    if (d == int64) return 8;
    if (d == float16) return 9;
    if (d == float32) return 10;
    if (d == bfloat16) return 12;
    if (d == complex64) return 13;
    return 10;  // fallback
}
