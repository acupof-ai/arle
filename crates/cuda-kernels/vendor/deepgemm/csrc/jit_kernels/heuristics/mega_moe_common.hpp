#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace deep_gemm {

template <typename T>
constexpr T mega_moe_ceil_div(T x, T y) {
    return (x + y - 1) / y;
}

template <typename T>
constexpr T mega_moe_align(T x, T y) {
    return mega_moe_ceil_div(x, y) * y;
}

inline int get_num_wave_pool_tokens(
    int num_ranks, int num_topk, int num_max_tokens_per_rank,
    int num_experts_per_wave, int block_m) {
    if (num_max_tokens_per_rank % block_m != 0)
        throw std::invalid_argument("MegaMoE max tokens must align to block_m");
    const int all_rank_tokens = num_max_tokens_per_rank * num_ranks;
    if (num_experts_per_wave == 1)
        return all_rank_tokens;
    return std::min(
        all_rank_tokens * num_experts_per_wave,
        mega_moe_align(
            all_rank_tokens * num_topk + num_experts_per_wave * (block_m - 1),
            block_m));
}

inline int get_num_experts_per_wave_for_mega_moe(
    int num_experts_per_rank, int num_tokens, int num_topk,
    int intermediate_hidden, int block_m, int block_n, int num_sms,
    int num_ring_tokens, int num_max_tokens_per_rank, int num_ranks) {
    int max_wave = num_experts_per_rank;
    while (max_wave > 0 &&
           get_num_wave_pool_tokens(
               num_ranks, num_topk, num_max_tokens_per_rank, max_wave, block_m) >
               num_ring_tokens)
        --max_wave;
    if (max_wave <= 0)
        throw std::invalid_argument("MegaMoE buffer is too small");

    constexpr int kImbalanceFactor = 2;
    const float expected =
        static_cast<float>(num_tokens * num_topk) / num_experts_per_rank;
    const int m_blocks = std::max(
        mega_moe_ceil_div(static_cast<int>(std::ceil(expected)), block_m), 1);
    const int blocks_per_expert = m_blocks * (2 * intermediate_hidden / block_n);
    int min_wave = mega_moe_ceil_div(kImbalanceFactor * num_sms, blocks_per_expert);
    if (expected < 1)
        min_wave = num_experts_per_rank;
    if (min_wave >= max_wave)
        return max_wave;
    if (blocks_per_expert >= num_sms)
        return min_wave;

    int best = min_wave;
    float best_tail = -1.0f;
    for (int wave = min_wave; wave <= std::min(max_wave, min_wave * 2); ++wave) {
        const int remainder = num_experts_per_rank % wave;
        const float tail = remainder == 0 ? 1.0f : static_cast<float>(remainder) / wave;
        if (tail > best_tail) {
            best = wave;
            best_tail = tail;
        }
    }
    return best;
}

} // namespace deep_gemm
