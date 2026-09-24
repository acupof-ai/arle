#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

#include <deep_gemm/layout/mega_moe.cuh>

#include "mega_moe_common.hpp"

namespace deep_gemm {

struct MegaMoESM90Config {
    int block_m, block_n, block_k;
    int cluster_size;
    int num_max_pool_tokens;
    int num_padded_sf_pool_tokens;
    int swizzle_acts_mode, swizzle_weights_mode;
    int num_experts_per_wave;
    int num_stages, smem_size;
    int num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads;

    friend std::ostream& operator<<(std::ostream& out, const MegaMoESM90Config& c) {
        return out << "MegaMoESM90Config(block_m=" << c.block_m
                   << ", block_n=" << c.block_n << ", block_k=" << c.block_k
                   << ", cluster_size=" << c.cluster_size
                   << ", num_max_pool_tokens=" << c.num_max_pool_tokens
                   << ", num_padded_sf_pool_tokens=" << c.num_padded_sf_pool_tokens
                   << ", swizzle_acts_mode=" << c.swizzle_acts_mode
                   << ", swizzle_weights_mode=" << c.swizzle_weights_mode
                   << ", num_experts_per_wave=" << c.num_experts_per_wave
                   << ", num_stages=" << c.num_stages << ", smem_size=" << c.smem_size
                   << ", num_dispatch_threads=" << c.num_dispatch_threads
                   << ", num_non_epilogue_threads=" << c.num_non_epilogue_threads
                   << ", num_epilogue_threads=" << c.num_epilogue_threads << ')';
    }
};

inline int get_num_experts_per_wave_for_mega_moe_sm90(
    int num_experts_per_rank, int num_tokens, int num_topk,
    int intermediate_hidden, int block_m, int block_n, int num_sms,
    int num_ring_tokens, int num_max_tokens_per_rank, int num_ranks) {
    const float expected =
        static_cast<float>(num_tokens) * num_topk / num_experts_per_rank;
    if (expected < 1.0f || expected > 4.0f)
        return num_experts_per_rank;
    if (block_m == 64 && intermediate_hidden >= 3072 &&
        num_experts_per_rank * (2 * intermediate_hidden / block_n) >= 4 * num_sms)
        return num_experts_per_rank;
    return get_num_experts_per_wave_for_mega_moe(
        num_experts_per_rank, num_tokens, num_topk, intermediate_hidden,
        block_m, block_n, num_sms, num_ring_tokens,
        num_max_tokens_per_rank, num_ranks);
}

inline bool should_use_swap_ab_for_mega_moe_sm90(
    int num_experts_per_rank, int num_tokens, int num_topk,
    int block_m, int num_epilogue_threads, bool enabled) {
    const float expected =
        static_cast<float>(num_tokens) * num_topk / num_experts_per_rank;
    return enabled && block_m == 64 && num_epilogue_threads == 256 &&
           expected > 0.0f && expected < 30.0f;
}

inline std::pair<int, int> get_pipeline_config_for_mega_moe_sm90(
    int smem_capacity, int num_experts, int hidden,
    int block_m, int block_n, int block_k,
    int num_dispatch_warps, int num_epilogue_warps, bool use_swap_ab) {
    constexpr int kSmemAlignment = 1024;
    const int expert_counts = mega_moe_align(
        num_experts * static_cast<int>(sizeof(uint32_t)), kSmemAlignment);
    const int send_buffers = mega_moe_align(
        hidden * num_dispatch_warps, kSmemAlignment);
    const int cd = mega_moe_align(
        std::max({
            block_m * (block_n / 2),
            block_m * block_n * 2,
            use_swap_ab ? block_m * (block_n / 2) * 5 : 0}),
        kSmemAlignment);
    const int sfa = mega_moe_align(2 * block_m * static_cast<int>(sizeof(float)), 128);
    const int per_stage = block_m * block_k + block_n * block_k + sfa + 16;
    const int fixed = expert_counts + send_buffers + cd +
                      (num_dispatch_warps + 2 * num_epilogue_warps) * 8;
    const int stages = (smem_capacity - fixed) / per_stage;
    if (stages < 2)
        throw std::invalid_argument("SM90 MegaMoE needs at least two stages");
    return {stages, fixed + stages * per_stage};
}

inline MegaMoESM90Config get_mega_moe_config_sm90_core(
    int num_ranks, int num_experts, int num_experts_per_rank,
    int num_max_tokens_per_rank, int num_tokens, int num_topk,
    int hidden, int intermediate_hidden, int num_padded_sf_pool_tokens,
    int num_sms, int smem_capacity, bool swap_ab_enabled) {
    const float expected =
        static_cast<float>(num_tokens) * num_ranks * num_topk / num_experts;
    const int block_m = expected > 64.0f ? 128 : 64;
    const int epilogue_threads = block_m == 128 ? 512 : 256;
    const bool use_swap_ab = should_use_swap_ab_for_mega_moe_sm90(
        num_experts_per_rank, num_tokens, num_topk,
        block_m, epilogue_threads, swap_ab_enabled);
    const bool decode_n256 = block_m == 64 && intermediate_hidden >= 2048 &&
                             expected >= 0.25f &&
                             (2 * intermediate_hidden) % 256 == 0 && hidden % 256 == 0;
    const int block_n = use_swap_ab ? 128 : (block_m == 128 || decode_n256 ? 256 : 128);
    constexpr int block_k = 128;
    const int pool_tokens = layout::get_num_max_pool_tokens(
        num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
    const int experts_per_wave = get_num_experts_per_wave_for_mega_moe_sm90(
        num_experts_per_rank, num_tokens, num_topk, intermediate_hidden,
        block_m, block_n, num_sms, pool_tokens,
        num_max_tokens_per_rank, num_ranks);
    const int dispatch_threads = 64;
    const int non_epilogue_threads = 64;
    const auto [stages, smem] = get_pipeline_config_for_mega_moe_sm90(
        smem_capacity, num_experts, hidden, block_m, block_n, block_k,
        dispatch_threads / 32, epilogue_threads / 32, use_swap_ab);
    return {
        block_m, block_n, block_k, 1, pool_tokens, num_padded_sf_pool_tokens,
        128, 128, experts_per_wave, stages, smem,
        dispatch_threads, non_epilogue_threads, epilogue_threads};
}

struct SM90MegaMoeWorkspaceLayout {
    uint64_t num_bytes;
    uint64_t x, x_sf, topk_idx, topk_weights;
    uint64_t l1_acts, l1_acts_sf, l1_topk_weights;
    uint64_t l2_acts, l2_acts_sf, combine;
    int num_max_pool_tokens;
    int num_padded_sf_pool_tokens;
};

struct SM90MegaMoeTmaConfig {
    int weight_block_n;
    int l1_output_box_n;
    int l1_output_box_m;
};

inline SM90MegaMoeTmaConfig get_sm90_mega_moe_tma_config(
    const MegaMoESM90Config& config) {
    const int epilogue_warpgroups = config.num_epilogue_threads / 128;
    const int warpgroup_block_n = config.block_n / epilogue_warpgroups;
    const bool split_n = config.block_m == 64 && epilogue_warpgroups > 1 &&
                         config.block_n % epilogue_warpgroups == 0 &&
                         (warpgroup_block_n == 64 || warpgroup_block_n == 128);
    const bool split_mn = config.block_m == 128 && config.block_n == 256 &&
                          epilogue_warpgroups == 4;
    const int split_m = split_n ? 1 : (split_mn ? 2 : epilogue_warpgroups);
    const int split_n_count = split_n ? epilogue_warpgroups : (split_mn ? 2 : 1);
    const int warpgroup_l1_output_n = config.block_n / split_n_count / 2;
    const bool shared_sf = split_n && warpgroup_l1_output_n < 64;
    return {
        std::min(config.block_n, 256),
        shared_sf ? config.block_n / 2 : warpgroup_l1_output_n,
        shared_sf ? config.block_m : config.block_m / split_m,
    };
}

inline SM90MegaMoeWorkspaceLayout get_sm90_mega_moe_workspace_layout(
    int num_ranks, int num_experts, int num_max_tokens_per_rank,
    int num_topk, int hidden, int intermediate_hidden) {
    if (num_ranks <= 0 || num_experts % num_ranks != 0 ||
        hidden % 128 != 0 || intermediate_hidden % 128 != 0)
        throw std::invalid_argument("invalid SM90 MegaMoE workspace shape");
    const layout::SM90Workspace workspace(
        nullptr, num_ranks, num_experts, num_max_tokens_per_rank, num_topk);
    int padded_sf = 0;
    constexpr int kSm90CandidateBlockM[] = {64, 128};
    for (int block_m : kSm90CandidateBlockM)
        padded_sf = std::max(
            padded_sf,
            layout::get_num_sf_ring_tokens(
                static_cast<int>(workspace.num_max_pool_tokens), block_m));
    uint64_t offset = workspace.get_num_bytes();
    const auto take = [&offset](uint64_t bytes) {
        const uint64_t result = offset;
        offset += bytes;
        return result;
    };
    const int pool = static_cast<int>(workspace.num_max_pool_tokens);
    SM90MegaMoeWorkspaceLayout result{};
    result.x = take(static_cast<uint64_t>(num_max_tokens_per_rank) * hidden);
    result.x_sf = take(static_cast<uint64_t>(num_max_tokens_per_rank) * hidden / 32);
    result.topk_idx = take(static_cast<uint64_t>(num_max_tokens_per_rank) * num_topk * sizeof(int64_t));
    result.topk_weights = take(static_cast<uint64_t>(num_max_tokens_per_rank) * num_topk * sizeof(float));
    result.l1_acts = take(static_cast<uint64_t>(pool) * hidden);
    result.l1_acts_sf = take(static_cast<uint64_t>(padded_sf) * hidden / 32);
    result.l1_topk_weights = take(static_cast<uint64_t>(pool) * sizeof(float));
    result.l2_acts = take(static_cast<uint64_t>(pool) * intermediate_hidden);
    // SM90 L1 emits one float SF per 64 L2 channels; the generic per-128 layout underallocates.
    result.l2_acts_sf = take(static_cast<uint64_t>(padded_sf) * intermediate_hidden / 16);
    result.combine = take(
        static_cast<uint64_t>(num_topk) * num_max_tokens_per_rank * hidden * 2);
    result.num_bytes = offset;
    result.num_max_pool_tokens = pool;
    result.num_padded_sf_pool_tokens = padded_sf;
    return result;
}

struct SM90FP8MegaMoeKernelSpec {
    int num_max_tokens_per_rank, hidden, intermediate_hidden;
    int num_experts, num_topk, num_ranks, num_tokens, num_sms;
    float activation_clamp;
    bool fast_math, reuse_accum_as_final, l2_arrival_counter;
    bool l2_epilogue_requires_full_sync, split_phase_hot_path, use_swap_ab;
    int epilogue_registers;
    MegaMoESM90Config config;
};

inline SM90FP8MegaMoeKernelSpec get_sm90_fp8_mega_moe_kernel_spec(
    int num_ranks, int num_experts, int num_max_tokens_per_rank,
    int num_tokens, int num_topk, int hidden, int intermediate_hidden,
    int num_padded_sf_pool_tokens, float activation_clamp, bool fast_math,
    int num_sms, int smem_capacity, bool swap_ab_enabled) {
    const auto config = get_mega_moe_config_sm90_core(
        num_ranks, num_experts, num_experts / num_ranks,
        num_max_tokens_per_rank, num_tokens, num_topk,
        hidden, intermediate_hidden, num_padded_sf_pool_tokens,
        num_sms, smem_capacity, swap_ab_enabled);
    const bool split_mn = config.block_m == 128 && config.block_n == 256 &&
                          config.num_epilogue_threads == 512;
    const bool l2_counter = split_mn ||
        (config.block_m == 64 && config.block_n == 256 &&
         config.num_epilogue_threads == 256 && num_tokens >= 4 && num_tokens <= 128);
    return {
        num_max_tokens_per_rank, hidden, intermediate_hidden,
        num_experts, num_topk, num_ranks, num_tokens, num_sms,
        activation_clamp, fast_math, config.block_m == 128, l2_counter,
        !l2_counter, split_mn && hidden >= 7168,
        should_use_swap_ab_for_mega_moe_sm90(
            num_experts / num_ranks, num_tokens, num_topk,
            config.block_m, config.num_epilogue_threads, swap_ab_enabled),
        config.num_epilogue_threads == 512 ? 112 : 0, config};
}

inline std::string sm90_mega_moe_float_literal(float value) {
    if (std::isinf(value))
        return value > 0 ? "cute::numeric_limits<float>::infinity()"
                         : "-cute::numeric_limits<float>::infinity()";
    if (std::isnan(value))
        throw std::invalid_argument("MegaMoE activation clamp cannot be NaN");
    std::ostringstream out;
    out << std::hexfloat << value << 'f';
    return out.str();
}

inline std::string generate_sm90_fp8_mega_moe_source(
    const SM90FP8MegaMoeKernelSpec& s) {
    const auto b = [](bool value) { return value ? "true" : "false"; };
    std::ostringstream out;
    out << "#include <deep_gemm/impls/sm90_fp8_mega_moe.cuh>\n"
           "using namespace deep_gemm;\n"
           "static void __instantiate_kernel() { auto ptr = reinterpret_cast<void*>("
           "&sm90_fp8_mega_moe_impl<"
        << s.num_max_tokens_per_rank << ',' << s.hidden << ',' << s.intermediate_hidden << ','
        << s.num_experts << ',' << s.num_topk << ',' << s.config.num_experts_per_wave << ','
        << s.config.block_m << ',' << s.config.block_n << ',' << s.config.block_k << ','
        << s.config.num_max_pool_tokens << ',' << s.config.num_padded_sf_pool_tokens << ','
        << s.config.num_stages << ',' << s.config.num_dispatch_threads << ','
        << s.config.num_non_epilogue_threads << ',' << s.config.num_epilogue_threads << ','
        << s.num_sms << ',' << s.num_ranks << ','
        << sm90_mega_moe_float_literal(s.activation_clamp) << ',' << b(s.fast_math) << ','
        << s.epilogue_registers << ',' << b(s.reuse_accum_as_final) << ','
        << b(s.l2_arrival_counter) << ',' << b(s.l2_epilogue_requires_full_sync) << ','
        << b(s.split_phase_hot_path) << ',' << b(s.use_swap_ab) << ">); }\n";
    return out.str();
}

} // namespace deep_gemm
