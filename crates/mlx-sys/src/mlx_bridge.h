#pragma once

#include "mlx_common.h"

#ifdef __cplusplus
extern "C" {
#endif

int32_t qwen35_compiled_prefill(
    void* model,
    mlx_array* token_ids,
    int32_t prompt_len,
    int32_t cache_pos,
    mlx_array** kv_caches,
    int32_t n_kv,
    mlx_array** gdr_states,
    int32_t n_gdr,
    mlx_array** out_logits,
    mlx_array** out_kv_caches,
    mlx_array** out_gdr_states
);

int32_t qwen35_compiled_block_verify(
    void* model,
    mlx_array* token_ids,
    int32_t block_size,
    int32_t cache_pos,
    mlx_array** kv_caches,
    int32_t n_kv,
    mlx_array** gdr_states,
    int32_t n_gdr,
    mlx_array** out_logits,
    mlx_array** out_kv_caches,
    mlx_array** out_gdr_states
);

#ifdef __cplusplus
}
#endif
