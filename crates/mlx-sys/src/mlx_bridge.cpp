#include "mlx_common.h"
#include "mlx/backend/metal/device.h"
#include "mlx/version.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {

void require_rank(const array& arr, int expected, const char* name) {
    if (arr.ndim() != expected) {
        throw std::invalid_argument(std::string(name) + " must have rank " + std::to_string(expected));
    }
}

void require_dtype(const array& arr, Dtype expected, const char* name) {
    if (arr.dtype() != expected) {
        throw std::invalid_argument(std::string(name) + " has an unexpected dtype");
    }
}

auto& gated_delta_tape_kernel() {
    static auto kernel = fast::metal_kernel(
        "gated_delta_tape",
        {"q", "k", "v", "g", "beta", "state_in", "T"},
        {"y", "state_out", "innovation_tape"},
        R"(
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y, innovation_tape: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;
        auto tape_ = innovation_tape + b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // g: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = static_cast<float>(i_state[s_idx]);
        }

        for (int t = 0; t < T; ++t) {
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                state[i] = state[i] * g_[hv_idx];
                kv_mem += state[i] * k_[s_idx];
            }
            kv_mem = simd_sum(kv_mem);

            auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                state[i] = state[i] + k_[s_idx] * delta;
                out += state[i] * q_[s_idx];
            }
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {
                y[dv_idx] = static_cast<InT>(out);
                tape_[dv_idx] = static_cast<InT>(delta);
            }
            q_ += Hk * Dk;
            k_ += Hk * Dk;
            v_ += Hv * Dv;
            y += Hv * Dv;
            tape_ += Hv * Dv;
            g_ += Hv;
            beta_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            o_state[s_idx] = static_cast<StT>(state[i]);
        }
        )",
        "",
        true,
        false);
    return kernel;
}

auto& tape_replay_kernel() {
    static auto kernel = fast::metal_kernel(
        "tape_replay",
        {"tape", "k", "g", "state_in", "T"},
        {"state_out"},
        R"(
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // tape: [B, T, Hv, Dv]
        auto tape_ = tape + b_idx * T * Hv * Dv + hv_idx * Dv;

        // k: [B, T, Hk, Dk]
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }

        // g: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          auto delta = static_cast<float>(tape_[dv_idx]);
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * g_[hv_idx];
            state[i] = state[i] + k_[s_idx] * delta;
          }
          tape_ += Hv * Dv;
          k_ += Hk * Dk;
          g_ += Hv;
        }

        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }
        )",
        "",
        true,
        false);
    return kernel;
}

auto& prefix_match_len_i32_kernel() {
    static auto kernel = fast::metal_kernel(
        "prefix_match_len_i32",
        {"lhs", "rhs", "T"},
        {"out"},
        R"(
        uint tid = thread_position_in_threadgroup.x;
        threadgroup int flags[256];

        if (tid < T) {
            flags[tid] = (lhs[tid] == rhs[tid]) ? 1 : 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            int count = 0;
            for (int i = 0; i < T; ++i) {
                if (flags[i] != 0) {
                    count += 1;
                } else {
                    break;
                }
            }
            out[0] = count;
        }
        )",
        "",
        true,
        false);
    return kernel;
}

auto& prefix_match_len_i32_batched_kernel() {
    static auto kernel = fast::metal_kernel(
        "prefix_match_len_i32_batched",
        {"lhs", "rhs", "T"},
        {"out"},
        R"(
        uint tid = thread_position_in_threadgroup.x;
        uint b_idx = thread_position_in_grid.z;
        threadgroup int flags[256];

        auto lhs_row = lhs + b_idx * T;
        auto rhs_row = rhs + b_idx * T;

        if (tid < T) {
            flags[tid] = (lhs_row[tid] == rhs_row[tid]) ? 1 : 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            int count = 0;
            for (int i = 0; i < T; ++i) {
                if (flags[i] != 0) {
                    count += 1;
                } else {
                    break;
                }
            }
            out[b_idx] = count;
        }
        )",
        "",
        true,
        false);
    return kernel;
}

auto& gather_axis1_i32_kernel() {
    static auto kernel = fast::metal_kernel(
        "gather_axis1_i32",
        {"values", "indices", "T"},
        {"out"},
        R"(
        uint b_idx = thread_position_in_grid.x;
        int col = indices[b_idx];
        out[b_idx] = values[b_idx * T + col];
        )",
        "",
        true,
        false);
    return kernel;
}

auto& tape_replay_varlen_kernel() {
    static auto kernel = fast::metal_kernel(
        "tape_replay_varlen",
        {"tape", "k", "g", "state_in", "steps", "T", "B"},
        {"state_out"},
        R"(
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        if (b_idx >= B) {
          return;
        }
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        auto T_b = steps[b_idx];
        constexpr int n_per_t = Dk / 32;

        // tape: [B, T, Hv, Dv]
        auto tape_ = tape + b_idx * T * Hv * Dv + hv_idx * Dv;

        // k: [B, T, Hk, Dk]
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }

        // g: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;

        for (int t = 0; t < T_b; ++t) {
          auto delta = static_cast<float>(tape_[dv_idx]);
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * g_[hv_idx];
            state[i] = state[i] + k_[s_idx] * delta;
          }
          tape_ += Hv * Dv;
          k_ += Hk * Dk;
          g_ += Hv;
        }

        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }
        )",
        "",
        true,
        false);
    return kernel;
}

auto& batched_sdpa_2pass_partials_kernel() {
    static auto kernel = fast::metal_kernel(
        "batched_sdpa_2pass_partials",
        {
            "queries",
            "keys",
            "values",
            "gqa_factor",
            "N",
            "k_head_stride",
            "k_seq_stride",
            "v_head_stride",
            "v_seq_stride",
            "scale",
            "blocks",
        },
        {"partials", "sums", "maxs"},
        R"(
        constexpr int BD = 32;
        constexpr int qk_per_thread = D / BD;
        constexpr int v_per_thread = V / BD;

        auto q_head_idx = threadgroup_position_in_grid.x;
        auto b_idx = threadgroup_position_in_grid.y;
        auto block_idx = threadgroup_position_in_grid.z;
        auto q_seq_idx = thread_position_in_threadgroup.z;
        auto simd_lid = thread_index_in_simdgroup;

        auto Hq = threadgroups_per_grid.x;
        auto hk_idx = q_head_idx / gqa_factor;
        auto q_batch_head_idx = b_idx * Hq + q_head_idx;
        auto o_offset = q_batch_head_idx * M_FIXED + q_seq_idx;

        auto q_ = queries + (o_offset * D) + simd_lid * qk_per_thread;
        auto k_ = keys + ((b_idx * Hk + hk_idx) * k_head_stride) + block_idx * k_seq_stride + simd_lid * qk_per_thread;
        auto v_ = values + ((b_idx * Hk + hk_idx) * v_head_stride) + block_idx * v_seq_stride + simd_lid * v_per_thread;

        partials += (o_offset * blocks + block_idx) * V + simd_lid * v_per_thread;
        sums += o_offset * blocks + block_idx;
        maxs += o_offset * blocks + block_idx;

        thread float q[qk_per_thread];
        thread float o[v_per_thread];
        threadgroup InT tg_k[BD * qk_per_thread];
        threadgroup InT tg_v[BD * v_per_thread];

        for (int i = 0; i < qk_per_thread; ++i) {
            q[i] = static_cast<float>(scale) * static_cast<float>(q_[i]);
        }
        for (int i = 0; i < v_per_thread; ++i) {
            o[i] = 0.0f;
        }

        float max_score = Limits<float>::finite_min;
        float sum_exp_score = 0.0f;

        for (int n = block_idx; n < N; n += blocks) {
            if (q_seq_idx == 0) {
                for (int i = 0; i < qk_per_thread; ++i) {
                    tg_k[simd_lid * qk_per_thread + i] = k_[i];
                }
                for (int i = 0; i < v_per_thread; ++i) {
                    tg_v[simd_lid * v_per_thread + i] = v_[i];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            bool use_key = (n <= (N - M_FIXED + q_seq_idx));

            if (use_key) {
                float score = 0.0f;
                for (int i = 0; i < qk_per_thread; ++i) {
                    score += q[i] * static_cast<float>(tg_k[simd_lid * qk_per_thread + i]);
                }
                score = simd_sum(score);

                float new_max = metal::max(max_score, score);
                float factor = fast::exp(max_score - new_max);
                float exp_score = fast::exp(score - new_max);

                max_score = new_max;
                sum_exp_score = sum_exp_score * factor + exp_score;
                for (int i = 0; i < v_per_thread; ++i) {
                    o[i] = o[i] * factor + exp_score * static_cast<float>(tg_v[simd_lid * v_per_thread + i]);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
            k_ += blocks * int(k_seq_stride);
            v_ += blocks * int(v_seq_stride);
        }

        if (simd_lid == 0) {
            sums[0] = sum_exp_score;
            maxs[0] = max_score;
        }
        for (int i = 0; i < v_per_thread; ++i) {
            partials[i] = static_cast<InT>(o[i]);
        }
        )",
        "",
        true,
        false);
    return kernel;
}

auto& batched_sdpa_2pass_reduce_kernel() {
    static auto kernel = fast::metal_kernel(
        "batched_sdpa_2pass_reduce",
        {"partials", "sums", "maxs", "blocks"},
        {"out"},
        R"(
        constexpr int BN = 32;
        constexpr int BD = 32;
        constexpr int elem_per_thread = V / BD;

        auto head_idx = threadgroup_position_in_grid.x;
        auto q_seq_idx = threadgroup_position_in_grid.y;
        auto simd_gid = simdgroup_index_in_threadgroup;
        auto simd_lid = thread_index_in_simdgroup;

        auto q_offset = head_idx * M_FIXED + q_seq_idx;
        partials += (q_offset * blocks + simd_gid) * V + simd_lid * elem_per_thread;
        sums += q_offset * blocks;
        maxs += q_offset * blocks;
        out += q_offset * V + simd_gid * elem_per_thread;

        thread float o[elem_per_thread];
        threadgroup float outputs[BN * BD];

        for (int i = 0; i < elem_per_thread; ++i) {
            o[i] = 0.0f;
        }

        float sum_exp_score = 0.0f;
        float max_score = Limits<float>::finite_min;

        for (int b = 0; b < blocks / BN; ++b) {
            max_score = metal::max(max_score, maxs[simd_lid + BN * b]);
        }
        max_score = simd_max(max_score);

        for (int b = 0; b < blocks / BN; ++b) {
            float factor = fast::exp(maxs[simd_lid + BN * b] - max_score);
            sum_exp_score += factor * sums[simd_lid + BN * b];
        }
        sum_exp_score = simd_sum(sum_exp_score);

        for (int b = 0; b < blocks / BN; ++b) {
            float factor = fast::exp(maxs[simd_gid] - max_score);
            for (int i = 0; i < elem_per_thread; ++i) {
                o[i] += factor * static_cast<float>(partials[i]);
            }
            maxs += BN;
            partials += BN * V;
        }

        for (int i = 0; i < elem_per_thread; ++i) {
            outputs[simd_lid * BD + simd_gid] = o[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
            o[i] = sum_exp_score == 0.0f ? o[i] : (o[i] / sum_exp_score);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (simd_lid == 0) {
            for (int i = 0; i < elem_per_thread; ++i) {
                out[i] = static_cast<InT>(o[i]);
            }
        }
        )",
        "",
        true,
        false);
    return kernel;
}

bool verify_qmm_shape_eligible(const array& x, int32_t group_size, int32_t bits, bool transpose) {
    if (!transpose || bits != 4 || (group_size != 32 && group_size != 64 && group_size != 128)) {
        return false;
    }
    if (x.dtype() != bfloat16 && x.dtype() != float16) {
        return false;
    }
    int64_t m = 1;
    for (int axis = 0; axis < x.ndim() - 1; ++axis) {
        m *= x.shape(axis);
    }
    return m == 16;
}

std::string build_verify_qmm_mma2big_source(int group_size) {
    return R"(
        using namespace metal;
        constexpr int BM = 16;
        constexpr int BN = 32;
        constexpr int BK = 32;
        constexpr int BK_SUB = 8;
        constexpr int GS = )" + std::to_string(group_size) + R"(;

        uint tid   = thread_position_in_threadgroup.x;
        uint sg_id = tid / 32;
        uint tg_n  = threadgroup_position_in_grid.y;

        int K = int(K_size);
        int N = int(N_size);
        int K_by_8  = K / 8;
        int K_by_gs = K / GS;
        int n0 = int(tg_n) * BN;

        threadgroup T B_tile[BK * BN];

        simdgroup_matrix<T, 8, 8> a_top, a_bot, b_L, b_R;
        simdgroup_matrix<float, 8, 8> c_tL = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_tR = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_bL = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_bR = simdgroup_matrix<float, 8, 8>(0.0f);

        int t_a = int(tid);
        int t_b = int(tid) + 64;
        int dq_k_a = t_a / BN, dq_n_a = t_a % BN;
        int dq_k_b = t_b / BN, dq_n_b = t_b % BN;
        int sg_n_off = int(sg_id) * 16;

        for (int k0 = 0; k0 < K; k0 += BK) {
            {
                int n_global = n0 + dq_n_a;
                int k_base = k0 + dq_k_a * 8;
                uint32_t packed = w_q[n_global * K_by_8 + (k_base >> 3)];
                float s = float(scales[n_global * K_by_gs + (k_base / GS)]);
                float b = float(biases[n_global * K_by_gs + (k_base / GS)]);
                for (int ki = 0; ki < 8; ++ki) {
                    uint32_t nib = (packed >> (ki * 4)) & 0xFu;
                    B_tile[(dq_k_a * 8 + ki) * BN + dq_n_a] = T(float(nib) * s + b);
                }
            }
            {
                int n_global = n0 + dq_n_b;
                int k_base = k0 + dq_k_b * 8;
                uint32_t packed = w_q[n_global * K_by_8 + (k_base >> 3)];
                float s = float(scales[n_global * K_by_gs + (k_base / GS)]);
                float b = float(biases[n_global * K_by_gs + (k_base / GS)]);
                for (int ki = 0; ki < 8; ++ki) {
                    uint32_t nib = (packed >> (ki * 4)) & 0xFu;
                    B_tile[(dq_k_b * 8 + ki) * BN + dq_n_b] = T(float(nib) * s + b);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (int ks = 0; ks < BK / BK_SUB; ++ks) {
                simdgroup_load(a_top, x + k0 + ks * BK_SUB,                  K);
                simdgroup_load(a_bot, x + 8 * K + k0 + ks * BK_SUB,          K);
                simdgroup_load(b_L, B_tile + ks * BK_SUB * BN + sg_n_off,         BN);
                simdgroup_load(b_R, B_tile + ks * BK_SUB * BN + sg_n_off + 8,     BN);
                simdgroup_multiply_accumulate(c_tL, a_top, b_L, c_tL);
                simdgroup_multiply_accumulate(c_tR, a_top, b_R, c_tR);
                simdgroup_multiply_accumulate(c_bL, a_bot, b_L, c_bL);
                simdgroup_multiply_accumulate(c_bR, a_bot, b_R, c_bR);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        simdgroup_matrix<T, 8, 8> c_tL_T, c_tR_T, c_bL_T, c_bR_T;
        c_tL_T.thread_elements()[0] = T(c_tL.thread_elements()[0]);
        c_tL_T.thread_elements()[1] = T(c_tL.thread_elements()[1]);
        c_tR_T.thread_elements()[0] = T(c_tR.thread_elements()[0]);
        c_tR_T.thread_elements()[1] = T(c_tR.thread_elements()[1]);
        c_bL_T.thread_elements()[0] = T(c_bL.thread_elements()[0]);
        c_bL_T.thread_elements()[1] = T(c_bL.thread_elements()[1]);
        c_bR_T.thread_elements()[0] = T(c_bR.thread_elements()[0]);
        c_bR_T.thread_elements()[1] = T(c_bR.thread_elements()[1]);
        simdgroup_store(c_tL_T, y + n0 + sg_n_off,                  N);
        simdgroup_store(c_tR_T, y + n0 + sg_n_off + 8,              N);
        simdgroup_store(c_bL_T, y + 8 * N + n0 + sg_n_off,          N);
        simdgroup_store(c_bR_T, y + 8 * N + n0 + sg_n_off + 8,      N);
    )";
}

auto make_verify_qmm_kernel(const std::string& name, const std::string& source) {
    return fast::metal_kernel(
        name,
        {"x", "w_q", "scales", "biases", "M_size", "K_size", "N_size"},
        {"y"},
        source,
        "",
        true,
        false);
}

auto& verify_qmm_mma2big_kernel(int group_size) {
    switch (group_size) {
        case 32: {
            static auto kernel = make_verify_qmm_kernel(
                "verify_qmm_mma2big_gs32",
                build_verify_qmm_mma2big_source(32));
            return kernel;
        }
        case 64: {
            static auto kernel = make_verify_qmm_kernel(
                "verify_qmm_mma2big_gs64",
                build_verify_qmm_mma2big_source(64));
            return kernel;
        }
        case 128: {
            static auto kernel = make_verify_qmm_kernel(
                "verify_qmm_mma2big_gs128",
                build_verify_qmm_mma2big_source(128));
            return kernel;
        }
        default:
            throw std::invalid_argument("verify_qmm_mma2big_kernel requires group_size in {32, 64, 128}");
    }
}

std::string gguf_quant_source() {
    return R"(
        using namespace metal;

        float gguf_f16(const device uint8_t* p) {
            ushort bits = ushort(p[0]) | (ushort(p[1]) << 8);
            return float(as_type<half>(bits));
        }

        int gguf_i8(uint8_t v) {
            int x = int(v);
            return x >= 128 ? x - 256 : x;
        }

        void gguf_q4_scales(const device uint8_t* s, thread uint8_t sc[8], thread uint8_t mn[8]) {
            for (int i = 0; i < 4; ++i) {
                sc[i] = s[i] & 0x3f;
                mn[i] = s[i + 4] & 0x3f;
            }
            for (int i = 0; i < 4; ++i) {
                sc[4 + i] = (s[8 + i] & 0x0f) | ((s[i] >> 6) << 4);
                mn[4 + i] = (s[8 + i] >> 4) | ((s[i + 4] >> 6) << 4);
            }
        }

        float gguf_q8_0_value(const device uint8_t* row, int k) {
            int block = k >> 5;
            int lane = k & 31;
            const device uint8_t* p = row + block * 34;
            return gguf_f16(p) * float(gguf_i8(p[2 + lane]));
        }

        float gguf_q3_k_value(const device uint8_t* row, int k) {
            int sb = k >> 8;
            int local = k & 255;
            const device uint8_t* p = row + sb * 110;
            const device uint8_t* hmask = p;
            const device uint8_t* qs = p + 32;
            const device uint8_t* scales = p + 96;
            float d = gguf_f16(p + 108);
            int sub = local >> 4;
            int low4;
            if (sub < 8) {
                low4 = int(scales[sub] & 0x0f);
            } else {
                low4 = int((scales[sub - 8] >> 4) & 0x0f);
            }
            int high2 = int((scales[8 + (sub & 3)] >> (2 * (sub / 4))) & 0x03);
            int scale = (low4 | (high2 << 4)) - 32;
            int q2 = int((qs[local >> 2] >> ((local & 3) * 2)) & 0x03);
            int hbit = int((hmask[local >> 3] >> (local & 7)) & 0x01);
            int q3 = q2 | (hbit << 2);
            return d * float(scale) * (float(q3) - 4.0f);
        }

        float gguf_q4_k_value(const device uint8_t* row, int k) {
            int sb = k >> 8;
            int local = k & 255;
            const device uint8_t* p = row + sb * 144;
            float d = gguf_f16(p);
            float dmin = gguf_f16(p + 2);
            uint8_t sc[8], mn[8];
            gguf_q4_scales(p + 4, sc, mn);
            int iter = local >> 6;
            int h = (local >> 5) & 1;
            int lane = local & 31;
            int sub = iter * 2 + h;
            uint8_t byte = p[16 + iter * 32 + lane];
            float q = h == 0 ? float(byte & 0x0f) : float(byte >> 4);
            return q * (d * float(sc[sub])) - dmin * float(mn[sub]);
        }

        float gguf_q5_k_value(const device uint8_t* row, int k) {
            int sb = k >> 8;
            int local = k & 255;
            const device uint8_t* p = row + sb * 176;
            float d = gguf_f16(p);
            float dmin = gguf_f16(p + 2);
            uint8_t sc[8], mn[8];
            gguf_q4_scales(p + 4, sc, mn);
            const device uint8_t* qh = p + 16;
            const device uint8_t* qs = p + 48;
            int iter = local >> 6;
            int h = (local >> 5) & 1;
            int lane = local & 31;
            int sub = iter * 2 + h;
            uint8_t byte = qs[iter * 32 + lane];
            int nib = h == 0 ? int(byte & 0x0f) : int(byte >> 4);
            int hi = (int(qh[lane]) >> sub) & 1;
            float q = float(nib | (hi << 4));
            return q * (d * float(sc[sub])) - dmin * float(mn[sub]);
        }

        float gguf_q6_k_value(const device uint8_t* row, int k) {
            int sb = k >> 8;
            int local = k & 255;
            const device uint8_t* p = row + sb * 210;
            const device uint8_t* ql_all = p;
            const device uint8_t* qh_all = p + 128;
            const device uint8_t* scales_all = p + 192;
            float d = gguf_f16(p + 208);
            int h = local >> 7;
            int rem = local & 127;
            int lane = rem & 31;
            const device uint8_t* ql = ql_all + h * 64;
            const device uint8_t* qh = qh_all + h * 32;
            const device uint8_t* sc = scales_all + h * 8;
            int q;
            int scale_idx;
            if (rem < 32) {
                q = int((ql[lane] & 0x0f) | ((qh[lane] & 0x03) << 4)) - 32;
                scale_idx = lane / 16;
            } else if (rem < 64) {
                q = int((ql[lane + 32] & 0x0f) | (((qh[lane] >> 2) & 0x03) << 4)) - 32;
                scale_idx = lane / 16 + 2;
            } else if (rem < 96) {
                q = int((ql[lane] >> 4) | (((qh[lane] >> 4) & 0x03) << 4)) - 32;
                scale_idx = lane / 16 + 4;
            } else {
                q = int((ql[lane + 32] >> 4) | (((qh[lane] >> 6) & 0x03) << 4)) - 32;
                scale_idx = lane / 16 + 6;
            }
            return d * float(gguf_i8(sc[scale_idx])) * float(q);
        }

    )";
}

auto make_gguf_matmul_kernel(const std::string& name) {
    return fast::metal_kernel(
        name,
        {"x", "w", "M_size", "K_size", "N_size"},
        {"y"},
        R"(
        uint tid = thread_position_in_threadgroup.x;
        uint n = threadgroup_position_in_grid.x;
        uint m = threadgroup_position_in_grid.y;
        int M = int(M_size);
        int K = int(K_size);
        int N = int(N_size);
        if (m >= uint(M) || n >= uint(N)) {
            return;
        }

        constexpr int TG = 256;
        threadgroup float partial[TG];
        int block_bytes;
        if constexpr (FORMAT == 8) {
            block_bytes = 34;
        } else if constexpr (FORMAT == 11) {
            block_bytes = 110;
        } else if constexpr (FORMAT == 12) {
            block_bytes = 144;
        } else if constexpr (FORMAT == 13) {
            block_bytes = 176;
        } else {
            block_bytes = 210;
        }
        int block_size = FORMAT == 8 ? 32 : 256;
        int row_bytes = (K / block_size) * block_bytes;
        const device uint8_t* row = w + int(n) * row_bytes;

        float sum = 0.0f;
        for (int k = int(tid); k < K; k += TG) {
            float wv;
            if constexpr (FORMAT == 8) {
                wv = gguf_q8_0_value(row, k);
            } else if constexpr (FORMAT == 11) {
                wv = gguf_q3_k_value(row, k);
            } else if constexpr (FORMAT == 12) {
                wv = gguf_q4_k_value(row, k);
            } else if constexpr (FORMAT == 13) {
                wv = gguf_q5_k_value(row, k);
            } else {
                wv = gguf_q6_k_value(row, k);
            }
            sum += float(x[int(m) * K + k]) * wv;
        }
        partial[tid] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = TG >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partial[tid] += partial[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            y[int(m) * N + int(n)] = T(partial[0]);
        }
        )",
        gguf_quant_source(),
        true,
        false);
}

auto make_gguf_matmul_m4_kernel(const std::string& name) {
    return fast::metal_kernel(
        name,
        {"x", "w", "M_size", "K_size", "N_size"},
        {"y"},
        R"(
        uint tid = thread_position_in_threadgroup.x;
        uint n = threadgroup_position_in_grid.x;
        uint m_tile = threadgroup_position_in_grid.y * 4;
        int M = int(M_size);
        int K = int(K_size);
        int N = int(N_size);
        if (m_tile >= uint(M) || n >= uint(N)) {
            return;
        }

        constexpr int TG = 256;
        threadgroup float partial0[TG];
        threadgroup float partial1[TG];
        threadgroup float partial2[TG];
        threadgroup float partial3[TG];
        int block_bytes;
        if constexpr (FORMAT == 8) {
            block_bytes = 34;
        } else if constexpr (FORMAT == 11) {
            block_bytes = 110;
        } else if constexpr (FORMAT == 12) {
            block_bytes = 144;
        } else if constexpr (FORMAT == 13) {
            block_bytes = 176;
        } else {
            block_bytes = 210;
        }
        int block_size = FORMAT == 8 ? 32 : 256;
        int row_bytes = (K / block_size) * block_bytes;
        const device uint8_t* row = w + int(n) * row_bytes;

        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        bool valid1 = int(m_tile) + 1 < M;
        bool valid2 = int(m_tile) + 2 < M;
        bool valid3 = int(m_tile) + 3 < M;
        for (int k = int(tid); k < K; k += TG) {
            float wv;
            if constexpr (FORMAT == 8) {
                wv = gguf_q8_0_value(row, k);
            } else if constexpr (FORMAT == 11) {
                wv = gguf_q3_k_value(row, k);
            } else if constexpr (FORMAT == 12) {
                wv = gguf_q4_k_value(row, k);
            } else if constexpr (FORMAT == 13) {
                wv = gguf_q5_k_value(row, k);
            } else {
                wv = gguf_q6_k_value(row, k);
            }
            int base = int(m_tile) * K + k;
            sum0 += float(x[base]) * wv;
            if (valid1) {
                sum1 += float(x[base + K]) * wv;
            }
            if (valid2) {
                sum2 += float(x[base + 2 * K]) * wv;
            }
            if (valid3) {
                sum3 += float(x[base + 3 * K]) * wv;
            }
        }
        partial0[tid] = sum0;
        partial1[tid] = sum1;
        partial2[tid] = sum2;
        partial3[tid] = sum3;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = TG >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partial0[tid] += partial0[tid + stride];
                partial1[tid] += partial1[tid + stride];
                partial2[tid] += partial2[tid + stride];
                partial3[tid] += partial3[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            int out = int(m_tile) * N + int(n);
            y[out] = T(partial0[0]);
            if (valid1) {
                y[out + N] = T(partial1[0]);
            }
            if (valid2) {
                y[out + 2 * N] = T(partial2[0]);
            }
            if (valid3) {
                y[out + 3 * N] = T(partial3[0]);
            }
        }
        )",
        gguf_quant_source(),
        true,
        false);
}

auto& gguf_matmul_kernel(int format) {
    switch (format) {
        case 8: {
            static auto kernel = make_gguf_matmul_kernel("gguf_q8_0_matmul");
            return kernel;
        }
        case 11: {
            static auto kernel = make_gguf_matmul_kernel("gguf_q3_k_matmul");
            return kernel;
        }
        case 12: {
            static auto kernel = make_gguf_matmul_kernel("gguf_q4_k_matmul");
            return kernel;
        }
        case 13: {
            static auto kernel = make_gguf_matmul_kernel("gguf_q5_k_matmul");
            return kernel;
        }
        case 14: {
            static auto kernel = make_gguf_matmul_kernel("gguf_q6_k_matmul");
            return kernel;
        }
        default:
            throw std::invalid_argument("GGUF matmul supports Q8_0, Q3_K, Q4_K, Q5_K, and Q6_K only");
    }
}

auto& gguf_matmul_m4_kernel(int format) {
    switch (format) {
        case 8: {
            static auto kernel = make_gguf_matmul_m4_kernel("gguf_q8_0_matmul_m4");
            return kernel;
        }
        case 11: {
            static auto kernel = make_gguf_matmul_m4_kernel("gguf_q3_k_matmul_m4");
            return kernel;
        }
        case 12: {
            static auto kernel = make_gguf_matmul_m4_kernel("gguf_q4_k_matmul_m4");
            return kernel;
        }
        case 13: {
            static auto kernel = make_gguf_matmul_m4_kernel("gguf_q5_k_matmul_m4");
            return kernel;
        }
        case 14: {
            static auto kernel = make_gguf_matmul_m4_kernel("gguf_q6_k_matmul_m4");
            return kernel;
        }
        default:
            throw std::invalid_argument("GGUF matmul supports Q8_0, Q3_K, Q4_K, Q5_K, and Q6_K only");
    }
}

auto& gguf_embedding_kernel() {
    static auto kernel = fast::metal_kernel(
        "gguf_embedding",
        {"ids", "w", "T_size", "K_size", "N_size"},
        {"y"},
        R"(
        uint idx = thread_position_in_grid.x;
        int Tn = int(T_size);
        int K = int(K_size);
        int N = int(N_size);
        if (idx >= uint(Tn * K)) {
            return;
        }
        int token_idx = int(idx) / K;
        int col = int(idx) - token_idx * K;
        int row_id = int(ids[token_idx]);
        if (row_id < 0 || row_id >= N) {
            y[idx] = T(0);
            return;
        }
        int block_bytes;
        if constexpr (FORMAT == 8) {
            block_bytes = 34;
        } else if constexpr (FORMAT == 11) {
            block_bytes = 110;
        } else if constexpr (FORMAT == 12) {
            block_bytes = 144;
        } else if constexpr (FORMAT == 13) {
            block_bytes = 176;
        } else {
            block_bytes = 210;
        }
        int block_size = FORMAT == 8 ? 32 : 256;
        int row_bytes = (K / block_size) * block_bytes;
        const device uint8_t* row = w + row_id * row_bytes;
        float value;
        if constexpr (FORMAT == 8) {
            value = gguf_q8_0_value(row, col);
        } else if constexpr (FORMAT == 11) {
            value = gguf_q3_k_value(row, col);
        } else if constexpr (FORMAT == 12) {
            value = gguf_q4_k_value(row, col);
        } else if constexpr (FORMAT == 13) {
            value = gguf_q5_k_value(row, col);
        } else {
            value = gguf_q6_k_value(row, col);
        }
        y[idx] = T(value);
        )",
        gguf_quant_source(),
        true,
        false);
    return kernel;
}

} // namespace

array batched_sdpa_2pass_cpp(
    const array& queries,
    const array& keys,
    const array& values,
    float scale,
    int32_t gqa_factor) {
    constexpr int blocks = 128;

    auto queries_arr = contiguous(queries);
    auto keys_arr = contiguous(keys);
    auto values_arr = contiguous(values);

    require_rank(queries_arr, 4, "queries");
    require_rank(keys_arr, 4, "keys");
    require_rank(values_arr, 4, "values");
    require_dtype(queries_arr, bfloat16, "queries");
    require_dtype(keys_arr, bfloat16, "keys");
    require_dtype(values_arr, bfloat16, "values");

    int bsz = queries_arr.shape(0);
    int Hq = queries_arr.shape(1);
    int q_len = queries_arr.shape(2);
    int D = queries_arr.shape(3);
    int Hk = keys_arr.shape(1);
    int N = keys_arr.shape(2);
    int V = values_arr.shape(3);

    if (bsz != keys_arr.shape(0) || bsz != values_arr.shape(0)) {
        throw std::invalid_argument("mlx_batched_sdpa_2pass got mismatched batch dimensions");
    }
    if (Hk != values_arr.shape(1) || N != values_arr.shape(2)) {
        throw std::invalid_argument("mlx_batched_sdpa_2pass got mismatched kv shapes");
    }
    if (q_len != 16) {
        throw std::invalid_argument("mlx_batched_sdpa_2pass requires query length 16");
    }
    if ((D != 128 && D != 256) || D != V) {
        throw std::invalid_argument("mlx_batched_sdpa_2pass requires D == V and D in {128, 256}");
    }
    if (Hk <= 0 || gqa_factor <= 0 || Hq != Hk * gqa_factor) {
        throw std::invalid_argument("mlx_batched_sdpa_2pass got an invalid gqa_factor");
    }

    int k_head_stride = keys_arr.shape(2) * keys_arr.shape(3);
    int k_seq_stride = keys_arr.shape(3);
    int v_head_stride = values_arr.shape(2) * values_arr.shape(3);
    int v_seq_stride = values_arr.shape(3);

    std::vector<array> partial_inputs = {
        queries_arr,
        keys_arr,
        values_arr,
        array(gqa_factor),
        array(N),
        array(k_head_stride),
        array(k_seq_stride),
        array(v_head_stride),
        array(v_seq_stride),
        array(scale),
        array(blocks),
    };
    Shape partial_shape{bsz * Hq, q_len, blocks, V};
    Shape stats_shape{bsz * Hq, q_len, blocks};
    std::vector<std::pair<std::string, fast::TemplateArg>> partial_tmpl = {
        {"InT", fast::TemplateArg(bfloat16)},
        {"D", fast::TemplateArg(D)},
        {"V", fast::TemplateArg(V)},
        {"Hk", fast::TemplateArg(Hk)},
        {"M_FIXED", fast::TemplateArg(q_len)},
    };

    auto partials_result = batched_sdpa_2pass_partials_kernel()(
        partial_inputs,
        {partial_shape, stats_shape, stats_shape},
        {bfloat16, float32, float32},
        std::make_tuple(Hq * 32, bsz, blocks * q_len),
        std::make_tuple(32, 1, q_len),
        partial_tmpl,
        std::nullopt,
        false,
        {});

    std::vector<std::pair<std::string, fast::TemplateArg>> reduce_tmpl = {
        {"InT", fast::TemplateArg(bfloat16)},
        {"V", fast::TemplateArg(V)},
        {"M_FIXED", fast::TemplateArg(q_len)},
    };

    auto out_result = batched_sdpa_2pass_reduce_kernel()(
        {partials_result[0], partials_result[1], partials_result[2], array(blocks)},
        {queries_arr.shape()},
        {bfloat16},
        std::make_tuple((bsz * Hq) * 1024, q_len, 1),
        std::make_tuple(1024, 1, 1),
        reduce_tmpl,
        std::nullopt,
        false,
        {});

    return std::move(out_result[0]);
}

array verify_quantized_matmul_cpp(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    int32_t group_size,
    int32_t bits,
    bool transpose) {
    if (!verify_qmm_shape_eligible(x, group_size, bits, transpose)) {
        return quantized_matmul(x, w, scales, biases, transpose, group_size, bits);
    }

    auto original_shape = x.shape();
    auto x_2d = contiguous(reshape(x, {16, x.shape(x.ndim() - 1)}));
    auto w_q = contiguous(w);
    auto scales_q = contiguous(scales);
    auto biases_q = contiguous(biases);

    const int M = 16;
    const int K = x_2d.shape(1);
    const int N = w_q.shape(0);
    if ((N % 32) != 0 || (K % 32) != 0) {
        return quantized_matmul(x, w, scales, biases, transpose, group_size, bits);
    }

    std::vector<array> inputs = {
        x_2d,
        w_q,
        scales_q,
        biases_q,
        array(M),
        array(K),
        array(N),
    };
    std::vector<std::pair<std::string, fast::TemplateArg>> tmpl = {
        {"T", fast::TemplateArg(x_2d.dtype())},
    };

    auto y = verify_qmm_mma2big_kernel(group_size)(
        inputs,
        {Shape{M, N}},
        {x_2d.dtype()},
        std::make_tuple(64, N / 32, 1),
        std::make_tuple(64, 1, 1),
        tmpl,
        std::nullopt,
        false,
        {});

    original_shape.back() = N;
    return reshape(y[0], original_shape);
}

array gguf_quantized_matmul_cpp(
    const array& x,
    const array& w,
    int32_t format,
    int32_t rows,
    int32_t cols) {
    if (w.dtype() != uint8) {
        throw std::invalid_argument("GGUF packed matmul requires uint8 packed weights");
    }
    if (x.dtype() != bfloat16 && x.dtype() != float16 && x.dtype() != float32) {
        throw std::invalid_argument("GGUF packed matmul requires floating activations");
    }
    if (x.shape(x.ndim() - 1) != cols) {
        throw std::invalid_argument("GGUF packed matmul input K does not match weight cols");
    }
    auto original_shape = x.shape();
    int64_t m64 = 1;
    for (int i = 0; i < x.ndim() - 1; ++i) {
        m64 *= x.shape(i);
    }
    int M = static_cast<int>(m64);
    auto x_2d = contiguous(reshape(x, {M, cols}));
    auto w_q = contiguous(w);

    std::vector<array> inputs = {
        x_2d,
        w_q,
        array(M),
        array(cols),
        array(rows),
    };
    std::vector<std::pair<std::string, fast::TemplateArg>> tmpl = {
        {"T", fast::TemplateArg(x_2d.dtype())},
        {"FORMAT", fast::TemplateArg(format)},
    };
    auto y = (M >= 2 ? gguf_matmul_m4_kernel(format) : gguf_matmul_kernel(format))(
        inputs,
        {Shape{M, rows}},
        {x_2d.dtype()},
        M >= 2 ? std::make_tuple(rows * 256, (M + 3) / 4, 1)
               : std::make_tuple(rows * 256, M, 1),
        std::make_tuple(256, 1, 1),
        tmpl,
        std::nullopt,
        false,
        {});
    original_shape.back() = rows;
    return reshape(y[0], original_shape);
}

array gguf_embedding_cpp(
    const array& ids,
    const array& w,
    int32_t format,
    int32_t rows,
    int32_t cols) {
    if (w.dtype() != uint8) {
        throw std::invalid_argument("GGUF packed embedding requires uint8 packed weights");
    }
    auto ids_flat = contiguous(astype(flatten(ids), int32));
    int tokens = ids_flat.size();
    std::vector<array> inputs = {
        ids_flat,
        contiguous(w),
        array(tokens),
        array(cols),
        array(rows),
    };
    std::vector<std::pair<std::string, fast::TemplateArg>> tmpl = {
        {"T", fast::TemplateArg(bfloat16)},
        {"FORMAT", fast::TemplateArg(format)},
    };
    auto y = gguf_embedding_kernel()(
        inputs,
        {Shape{tokens, cols}},
        {bfloat16},
        std::make_tuple(tokens * cols, 1, 1),
        std::make_tuple(256, 1, 1),
        tmpl,
        std::nullopt,
        false,
        {});
    auto out_shape = ids.shape();
    out_shape.push_back(cols);
    return reshape(y[0], out_shape);
}

extern "C" {

// === Error handling ===

const char* mlx_last_error() {
    return g_mlx_last_error.empty() ? nullptr : g_mlx_last_error.c_str();
}

const char* mlx_version_string() {
    MLX_TRY_RETURN_VALUE(nullptr, version());
}

int32_t mlx_metal_nax_available() {
    MLX_TRY_RETURN_VALUE(-1, metal::is_nax_available() ? 1 : 0);
}

// === Array lifecycle ===

mlx_array* mlx_array_new_float32(float val) {
    MLX_TRY_RETURN(from_arr(array(val)));
}

mlx_array* mlx_array_new_int32(int32_t val) {
    MLX_TRY_RETURN(from_arr(array(val)));
}

mlx_array* mlx_array_from_data(const void* data, const int32_t* shape, int32_t ndim, int32_t dtype_val) {
    MLX_TRY_RETURN([&]() {
        auto sh = make_shape(shape, static_cast<size_t>(ndim));
        auto dt = to_dtype(dtype_val);
        // MLX array constructor needs the allocator to copy data
        size_t nbytes = 1;
        for (int i = 0; i < ndim; i++) nbytes *= shape[i];
        nbytes *= size_of(dt);
        auto buf = allocator::malloc(nbytes);
        std::memcpy(buf.raw_ptr(), data, nbytes);
        return reinterpret_cast<mlx_array*>(new array(std::move(buf), sh, dt));
    }());
}

mlx_array* mlx_array_clone(mlx_array* a) {
    // Copy the shared_ptr (increment refcount, same underlying data)
    MLX_TRY_RETURN(reinterpret_cast<mlx_array*>(new array(*to_arr(a))));
}

void mlx_array_free(mlx_array* a) {
    MLX_TRY_VOID(delete to_arr(a));
}

int32_t mlx_array_ndim(mlx_array* a) {
    MLX_TRY_RETURN_VALUE(0, static_cast<int32_t>(to_arr(a)->ndim()));
}

const int32_t* mlx_array_shape(mlx_array* a) {
    // MLX shape() returns std::vector<int>; data() gives stable pointer
    // while the array is alive.
    MLX_TRY_RETURN(to_arr(a)->shape().data());
}

int32_t mlx_array_dtype(mlx_array* a) {
    MLX_TRY_RETURN_VALUE(10 /*float32 fallback*/, from_dtype(to_arr(a)->dtype()));
}

int32_t mlx_array_item_int32(mlx_array* a) {
    MLX_TRY_RETURN_VALUE(0, to_arr(a)->item<int32_t>());
}

float mlx_array_item_float32(mlx_array* a) {
    MLX_TRY_RETURN_VALUE(0.0f, to_arr(a)->item<float>());
}

const float* mlx_array_data_float32(mlx_array* a) {
    MLX_TRY_RETURN(to_arr(a)->data<float>());
}

const int32_t* mlx_array_data_int32(mlx_array* a) {
    MLX_TRY_RETURN(to_arr(a)->data<int32_t>());
}

size_t mlx_array_size(mlx_array* a) {
    MLX_TRY_RETURN_VALUE(0, to_arr(a)->size());
}

size_t mlx_array_nbytes(mlx_array* a) {
    MLX_TRY_RETURN_VALUE(0, to_arr(a)->nbytes());
}

size_t mlx_array_export_bytes(mlx_array* a, void* out, size_t out_len) {
    MLX_TRY_RETURN_VALUE(0, [&]() -> size_t {
        if (a == nullptr) {
            throw std::invalid_argument("mlx_array_export_bytes received null array");
        }
        auto arr = contiguous(*to_arr(a));
        eval(arr);

        size_t nbytes = arr.nbytes();
        if (out_len < nbytes) {
            throw std::invalid_argument("mlx_array_export_bytes output buffer is too small");
        }
        if (nbytes > 0 && out == nullptr) {
            throw std::invalid_argument("mlx_array_export_bytes received null output buffer");
        }
        std::memcpy(out, arr.data<char>(), nbytes);
        return nbytes;
    }());
}

// === Binary ops ===

mlx_array* mlx_add(mlx_array* a, mlx_array* b) {
    MLX_TRY_RETURN(from_arr(add(*to_arr(a), *to_arr(b))));
}

mlx_array* mlx_subtract(mlx_array* a, mlx_array* b) {
    MLX_TRY_RETURN(from_arr(subtract(*to_arr(a), *to_arr(b))));
}

mlx_array* mlx_multiply(mlx_array* a, mlx_array* b) {
    MLX_TRY_RETURN(from_arr(multiply(*to_arr(a), *to_arr(b))));
}

mlx_array* mlx_matmul(mlx_array* a, mlx_array* b) {
    MLX_TRY_RETURN(from_arr(matmul(*to_arr(a), *to_arr(b))));
}

mlx_array* mlx_greater(mlx_array* a, mlx_array* b) {
    MLX_TRY_RETURN(from_arr(greater(*to_arr(a), *to_arr(b))));
}

mlx_array* mlx_prefix_match_len_i32(mlx_array* lhs, mlx_array* rhs) {
    MLX_TRY_RETURN([&]() {
        auto lhs_arr = contiguous(*to_arr(lhs));
        auto rhs_arr = contiguous(*to_arr(rhs));
        require_rank(lhs_arr, 1, "lhs");
        require_rank(rhs_arr, 1, "rhs");
        require_dtype(lhs_arr, int32, "lhs");
        require_dtype(rhs_arr, int32, "rhs");
        if (lhs_arr.shape(0) != rhs_arr.shape(0)) {
            throw std::invalid_argument("mlx_prefix_match_len_i32 requires equal-length inputs");
        }
        int T = lhs_arr.shape(0);
        if (T <= 0) {
            return from_arr(array(int32_t{0}));
        }
        if (T > 256) {
            throw std::invalid_argument("mlx_prefix_match_len_i32 supports at most 256 elements");
        }

        auto result = prefix_match_len_i32_kernel()(
            {lhs_arr, rhs_arr, array(T)},
            {{1}},
            {int32},
            std::make_tuple(256, 1, 1),
            std::make_tuple(256, 1, 1),
            {},
            std::nullopt,
            false,
            {});
        return from_arr(std::move(result[0]));
    }());
}

mlx_array* mlx_prefix_match_len_i32_batched(mlx_array* lhs, mlx_array* rhs) {
    MLX_TRY_RETURN([&]() {
        auto lhs_arr = contiguous(*to_arr(lhs));
        auto rhs_arr = contiguous(*to_arr(rhs));
        require_rank(lhs_arr, 2, "lhs");
        require_rank(rhs_arr, 2, "rhs");
        require_dtype(lhs_arr, int32, "lhs");
        require_dtype(rhs_arr, int32, "rhs");
        if (lhs_arr.shape(0) != rhs_arr.shape(0) || lhs_arr.shape(1) != rhs_arr.shape(1)) {
            throw std::invalid_argument(
                "mlx_prefix_match_len_i32_batched requires equal [B, T] inputs");
        }
        int B = lhs_arr.shape(0);
        int T = lhs_arr.shape(1);
        if (B <= 0 || T <= 0) {
            return from_arr(zeros({std::max(B, 0)}, int32));
        }
        if (T > 256) {
            throw std::invalid_argument(
                "mlx_prefix_match_len_i32_batched supports at most 256 elements per row");
        }

        auto result = prefix_match_len_i32_batched_kernel()(
            {lhs_arr, rhs_arr, array(T)},
            {{B}},
            {int32},
            std::make_tuple(256, 1, B),
            std::make_tuple(256, 1, 1),
            {},
            std::nullopt,
            false,
            {});
        return from_arr(std::move(result[0]));
    }());
}

mlx_array* mlx_gather_axis1_i32(mlx_array* values, mlx_array* indices) {
    MLX_TRY_RETURN([&]() {
        auto values_arr = contiguous(*to_arr(values));
        auto indices_arr = contiguous(*to_arr(indices));
        require_rank(values_arr, 2, "values");
        require_rank(indices_arr, 1, "indices");
        require_dtype(values_arr, int32, "values");
        require_dtype(indices_arr, int32, "indices");
        if (values_arr.shape(0) != indices_arr.shape(0)) {
            throw std::invalid_argument(
                "mlx_gather_axis1_i32 requires values[B, T] and indices[B]");
        }

        int B = values_arr.shape(0);
        int T = values_arr.shape(1);
        if (B <= 0) {
            return from_arr(zeros({std::max(B, 0)}, int32));
        }
        if (T <= 0) {
            throw std::invalid_argument("mlx_gather_axis1_i32 requires T > 0");
        }

        auto result = gather_axis1_i32_kernel()(
            {values_arr, indices_arr, array(T)},
            {{B}},
            {int32},
            std::make_tuple(B, 1, 1),
            std::make_tuple(1, 1, 1),
            {},
            std::nullopt,
            false,
            {});
        return from_arr(std::move(result[0]));
    }());
}

// === Unary ops ===

mlx_array* mlx_exp(mlx_array* a) {
    MLX_TRY_RETURN(from_arr(exp(*to_arr(a))));
}

mlx_array* mlx_log1p(mlx_array* a) {
    MLX_TRY_RETURN(from_arr(log1p(*to_arr(a))));
}

mlx_array* mlx_negative(mlx_array* a) {
    MLX_TRY_RETURN(from_arr(negative(*to_arr(a))));
}

mlx_array* mlx_sqrt(mlx_array* a) {
    MLX_TRY_RETURN(from_arr(sqrt(*to_arr(a))));
}

mlx_array* mlx_reciprocal(mlx_array* a) {
    MLX_TRY_RETURN(from_arr(reciprocal(*to_arr(a))));
}

mlx_array* mlx_sigmoid(mlx_array* a) {
    MLX_TRY_RETURN(from_arr(sigmoid(*to_arr(a))));
}

mlx_array* mlx_tanh(mlx_array* a) {
    MLX_TRY_RETURN(from_arr(tanh(*to_arr(a))));
}

mlx_array* mlx_erf(mlx_array* a) {
    MLX_TRY_RETURN(from_arr(erf(*to_arr(a))));
}

mlx_array* mlx_log(mlx_array* a) {
    MLX_TRY_RETURN(from_arr(log(*to_arr(a))));
}

// === Shape ops ===

mlx_array* mlx_reshape(mlx_array* a, const int32_t* shape, size_t ndim) {
    MLX_TRY_RETURN([&]() {
        auto sh = make_shape(shape, ndim);
        return from_arr(reshape(*to_arr(a), sh));
    }());
}

mlx_array* mlx_transpose(mlx_array* a) {
    MLX_TRY_RETURN(from_arr(transpose(*to_arr(a))));
}

mlx_array* mlx_transpose_axes(mlx_array* a, const int32_t* axes, size_t n) {
    MLX_TRY_RETURN([&]() {
        std::vector<int> ax(axes, axes + n);
        return from_arr(transpose(*to_arr(a), ax));
    }());
}

mlx_array* mlx_astype(mlx_array* a, int32_t dtype) {
    MLX_TRY_RETURN(from_arr(astype(*to_arr(a), to_dtype(dtype))));
}

mlx_array* mlx_broadcast_to(mlx_array* a, const int32_t* shape, size_t ndim) {
    MLX_TRY_RETURN([&]() {
        auto sh = make_shape(shape, ndim);
        return from_arr(broadcast_to(*to_arr(a), sh));
    }());
}

mlx_array* mlx_expand_dims(mlx_array* a, int32_t axis) {
    MLX_TRY_RETURN(from_arr(expand_dims(*to_arr(a), static_cast<int>(axis))));
}

mlx_array* mlx_zeros(const int32_t* shape, size_t ndim, int32_t dtype) {
    MLX_TRY_RETURN([&]() {
        auto sh = make_shape(shape, ndim);
        return from_arr(zeros(sh, to_dtype(dtype)));
    }());
}

// === Indexing ===

mlx_array* mlx_take_axis(mlx_array* a, mlx_array* indices, int32_t axis) {
    MLX_TRY_RETURN(from_arr(take(*to_arr(a), *to_arr(indices), static_cast<int>(axis))));
}

mlx_array* mlx_slice(mlx_array* a, const int32_t* start, const int32_t* stop,
                     const int32_t* strides, size_t ndim) {
    MLX_TRY_RETURN([&]() {
        Shape st; for(size_t i=0;i<ndim;i++) st.push_back(start[i]);
        Shape sp; for(size_t i=0;i<ndim;i++) sp.push_back(stop[i]);
        Shape sr; for(size_t i=0;i<ndim;i++) sr.push_back(strides[i]);
        return from_arr(slice(*to_arr(a), st, sp, sr));
    }());
}

mlx_array* mlx_slice_update(mlx_array* src, mlx_array* update,
                            const int32_t* start, const int32_t* stop,
                            const int32_t* strides, size_t ndim) {
    MLX_TRY_RETURN([&]() {
        Shape st; for(size_t i=0;i<ndim;i++) st.push_back(start[i]);
        Shape sp; for(size_t i=0;i<ndim;i++) sp.push_back(stop[i]);
        Shape sr; for(size_t i=0;i<ndim;i++) sr.push_back(strides[i]);
        return from_arr(slice_update(*to_arr(src), *to_arr(update), st, sp, sr));
    }());
}

mlx_array* mlx_concatenate_axis(mlx_array** arrays, size_t count, int32_t axis) {
    MLX_TRY_RETURN([&]() {
        std::vector<array> arrs;
        arrs.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            arrs.push_back(*to_arr(arrays[i]));
        }
        return from_arr(concatenate(arrs, static_cast<int>(axis)));
    }());
}

mlx_array* mlx_where(mlx_array* cond, mlx_array* a, mlx_array* b) {
    MLX_TRY_RETURN(from_arr(where(*to_arr(cond), *to_arr(a), *to_arr(b))));
}

// Scatter-add `prefix_rows` feature vectors into a zero-initialized
// `[vocab, feature_dim]` output. Caller has already filtered OOB/negative
// indices host-side so `indices_data` is guaranteed in-bounds and
// `n_valid == prefix_rows` from the bridge's perspective.
//
// Implementation: build `zeros({vocab, feature_dim})` as the base, upload
// updates as `[n_valid, feature_dim]` and reshape to `[n_valid, 1, feature_dim]`
// (scatter_add wants `updates.ndim() == indices.ndim() + a.ndim()` = 1+2 = 3),
// upload indices as int32 `[n_valid]`, call scatter_add on axis 0.
mlx_array* mlx_scatter_add_rows_f32(const float* updates_data,
                                    const int32_t* indices_data,
                                    int32_t prefix_rows, int32_t feature_dim,
                                    int32_t vocab) {
    MLX_TRY_RETURN([&]() {
        if (vocab <= 0 || feature_dim <= 0) {
            throw std::invalid_argument("mlx_scatter_add_rows_f32: vocab and feature_dim must be positive");
        }
        Shape out_shape = {vocab, feature_dim};
        // prefix_rows == 0 (or everything filtered out) → just return zeros.
        if (prefix_rows <= 0) {
            return from_arr(zeros(out_shape, float32));
        }
        if (updates_data == nullptr || indices_data == nullptr) {
            throw std::invalid_argument("mlx_scatter_add_rows_f32: null data pointer with prefix_rows > 0");
        }
        Shape updates_shape = {prefix_rows, 1, feature_dim};
        size_t updates_bytes = static_cast<size_t>(prefix_rows) *
                               static_cast<size_t>(feature_dim) * sizeof(float);
        auto updates_buf = allocator::malloc(updates_bytes);
        std::memcpy(updates_buf.raw_ptr(), updates_data, updates_bytes);
        auto updates_arr = array(std::move(updates_buf), updates_shape, float32);

        Shape idx_shape = {prefix_rows};
        size_t idx_bytes = static_cast<size_t>(prefix_rows) * sizeof(int32_t);
        auto idx_buf = allocator::malloc(idx_bytes);
        std::memcpy(idx_buf.raw_ptr(), indices_data, idx_bytes);
        auto idx_arr = array(std::move(idx_buf), idx_shape, int32);

        auto base = zeros(out_shape, float32);
        return from_arr(scatter_add(base, idx_arr, updates_arr, 0));
    }());
}

// === Reduction ===

mlx_array* mlx_sum_axis(mlx_array* a, int32_t axis, bool keepdims) {
    MLX_TRY_RETURN(from_arr(sum(*to_arr(a), static_cast<int>(axis), keepdims)));
}

mlx_array* mlx_mean_axis(mlx_array* a, int32_t axis, bool keepdims) {
    MLX_TRY_RETURN(from_arr(mean(*to_arr(a), static_cast<int>(axis), keepdims)));
}

mlx_array* mlx_max_axis(mlx_array* a, int32_t axis, bool keepdims) {
    MLX_TRY_RETURN(from_arr(max(*to_arr(a), static_cast<int>(axis), keepdims)));
}

mlx_array* mlx_logsumexp_axis(mlx_array* a, int32_t axis, bool keepdims) {
    MLX_TRY_RETURN(from_arr(logsumexp(*to_arr(a), static_cast<int>(axis), keepdims)));
}

mlx_array* mlx_softmax_axis(mlx_array* a, int32_t axis, bool precise) {
    MLX_TRY_RETURN(from_arr(softmax(*to_arr(a), static_cast<int>(axis), precise)));
}

mlx_array* mlx_argmax(mlx_array* a, bool keepdims) {
    MLX_TRY_RETURN(from_arr(argmax(*to_arr(a), keepdims)));
}

mlx_array* mlx_argmax_axis(mlx_array* a, int axis, bool keepdims) {
    MLX_TRY_RETURN(from_arr(argmax(*to_arr(a), axis, keepdims)));
}

// === Quantized ===

mlx_array* mlx_quantized_matmul(mlx_array* x, mlx_array* w, mlx_array* scales,
                                mlx_array* biases, bool transpose,
                                int32_t group_size, int32_t bits) {
    MLX_TRY_RETURN(from_arr(quantized_matmul(
        *to_arr(x), *to_arr(w), *to_arr(scales), *to_arr(biases),
        transpose, group_size, bits)));
}

int32_t mlx_quantize(
    mlx_array* w,
    int32_t group_size,
    int32_t bits,
    mlx_array** out_w,
    mlx_array** out_scales,
    mlx_array** out_biases) {
    MLX_TRY_RETURN_VALUE(-1, [&]() {
        if (out_w == nullptr || out_scales == nullptr || out_biases == nullptr) {
            throw std::invalid_argument("mlx_quantize received null output pointer");
        }
        auto outputs = quantize(*to_arr(w), group_size, bits);
        if (outputs.size() != 3) {
            throw std::runtime_error("mlx_quantize expected three affine outputs");
        }
        *out_w = from_arr(std::move(outputs[0]));
        *out_scales = from_arr(std::move(outputs[1]));
        *out_biases = from_arr(std::move(outputs[2]));
        return 0;
    }());
}

// Fused quantized gated MLP: out = down(silu(gate(x)) * up(x))
// All three projections are 4-bit quantized. The fusion lets MLX's graph
// compiler merge intermediate kernels and avoid DRAM round-trips for the
// [N, intermediate] activation buffer.
mlx_array* mlx_fused_quantized_gated_mlp(
    mlx_array* x,
    mlx_array* gate_w, mlx_array* gate_s, mlx_array* gate_b,
    mlx_array* up_w,   mlx_array* up_s,   mlx_array* up_b,
    mlx_array* down_w, mlx_array* down_s, mlx_array* down_b,
    int32_t group_size, int32_t bits) {
    MLX_TRY_RETURN([&]() {
        auto& xr = *to_arr(x);
        auto gate = quantized_matmul(xr, *to_arr(gate_w), *to_arr(gate_s), *to_arr(gate_b), true, group_size, bits);
        auto up   = quantized_matmul(xr, *to_arr(up_w),   *to_arr(up_s),   *to_arr(up_b),   true, group_size, bits);
        // silu(gate) * up — single fused elementwise
        auto h = multiply(multiply(gate, sigmoid(gate)), up);
        return from_arr(quantized_matmul(h, *to_arr(down_w), *to_arr(down_s), *to_arr(down_b), true, group_size, bits));
    }());
}

mlx_array* mlx_dequantize(mlx_array* w, mlx_array* scales, mlx_array* biases,
                          int32_t group_size, int32_t bits) {
    MLX_TRY_RETURN(from_arr(dequantize(*to_arr(w), *to_arr(scales), *to_arr(biases),
                                       group_size, bits)));
}

mlx_array* mlx_gguf_quantized_matmul(
    mlx_array* x,
    mlx_array* w,
    int32_t format,
    int32_t rows,
    int32_t cols) {
    MLX_TRY_RETURN(from_arr(gguf_quantized_matmul_cpp(
        *to_arr(x), *to_arr(w), format, rows, cols)));
}

mlx_array* mlx_gguf_embedding(
    mlx_array* ids,
    mlx_array* w,
    int32_t format,
    int32_t rows,
    int32_t cols) {
    MLX_TRY_RETURN(from_arr(gguf_embedding_cpp(
        *to_arr(ids), *to_arr(w), format, rows, cols)));
}

// === Fast ops ===

mlx_array* mlx_fast_rms_norm(mlx_array* x, mlx_array* weight, float eps) {
    MLX_TRY_RETURN([&]() {
        if (weight == nullptr) {
            return from_arr(fast::rms_norm(*to_arr(x), std::nullopt, eps));
        }
        return from_arr(fast::rms_norm(*to_arr(x), *to_arr(weight), eps));
    }());
}

mlx_array* mlx_fast_rope(mlx_array* x, int32_t dims, bool traditional,
                         float base, float scale, int32_t offset) {
    MLX_TRY_RETURN(from_arr(fast::rope(
        *to_arr(x), dims, traditional, base, scale, offset)));
}

mlx_array* mlx_fast_rope_dynamic(mlx_array* x, int32_t dims, bool traditional,
                                 float base, float scale, mlx_array* offset) {
    MLX_TRY_RETURN(from_arr(fast::rope(
        *to_arr(x), dims, traditional, base, scale, *to_arr(offset))));
}

mlx_array* mlx_fast_sdpa(mlx_array* q, mlx_array* k, mlx_array* v,
                         float scale, const char* mask_mode) {
    // mask_mode: "" for no mask, "causal" for causal masking.
    // NEVER pass nullptr — std::string(nullptr) is UB.
    MLX_TRY_RETURN([&]() {
        std::string mode(mask_mode);
        return from_arr(fast::scaled_dot_product_attention(
            *to_arr(q), *to_arr(k), *to_arr(v), scale, mode));
    }());
}

mlx_array* mlx_fast_sdpa_masked(mlx_array* q, mlx_array* k, mlx_array* v,
                                float scale, mlx_array* mask_arr) {
    MLX_TRY_RETURN([&]() {
        return from_arr(fast::scaled_dot_product_attention(
            *to_arr(q), *to_arr(k), *to_arr(v), scale, "",
            *to_arr(mask_arr)));
    }());
}

void mlx_gated_delta_with_tape(
    mlx_array* q, mlx_array* k, mlx_array* v, mlx_array* g, mlx_array* beta,
    mlx_array* state_in, int T,
    mlx_array** out_y, mlx_array** out_state, mlx_array** out_tape) {
    try {
        mlx_clear_error();
        if (!out_y || !out_state || !out_tape) {
            throw std::invalid_argument("mlx_gated_delta_with_tape output pointers must be non-null");
        }

        auto q_arr = contiguous(*to_arr(q));
        auto k_arr = contiguous(*to_arr(k));
        auto v_arr = contiguous(*to_arr(v));
        auto g_arr = contiguous(*to_arr(g));
        auto beta_arr = contiguous(*to_arr(beta));
        auto state_arr = contiguous(*to_arr(state_in));

        require_rank(q_arr, 4, "q");
        require_rank(k_arr, 4, "k");
        require_rank(v_arr, 4, "v");
        require_rank(g_arr, 3, "g");
        require_rank(beta_arr, 3, "beta");
        require_rank(state_arr, 4, "state_in");
        require_dtype(q_arr, bfloat16, "q");
        require_dtype(k_arr, bfloat16, "k");
        require_dtype(v_arr, bfloat16, "v");
        require_dtype(g_arr, bfloat16, "g");
        require_dtype(beta_arr, bfloat16, "beta");
        require_dtype(state_arr, float32, "state_in");

        int B = q_arr.shape(0);
        int T_q = q_arr.shape(1);
        int Hk = q_arr.shape(2);
        int Dk = q_arr.shape(3);
        int Hv = v_arr.shape(2);
        int Dv = v_arr.shape(3);

        if (T != T_q || T != k_arr.shape(1) || T != v_arr.shape(1) || T != g_arr.shape(1) || T != beta_arr.shape(1)) {
            throw std::invalid_argument("mlx_gated_delta_with_tape got mismatched sequence lengths");
        }
        if (B != k_arr.shape(0) || B != v_arr.shape(0) || B != g_arr.shape(0) || B != beta_arr.shape(0) || B != state_arr.shape(0)) {
            throw std::invalid_argument("mlx_gated_delta_with_tape got mismatched batch dimensions");
        }
        if (Hk != k_arr.shape(2) || Dk != k_arr.shape(3)) {
            throw std::invalid_argument("mlx_gated_delta_with_tape got mismatched q/k shapes");
        }
        if (Hv != g_arr.shape(2) || Hv != beta_arr.shape(2) || Hv != state_arr.shape(1) || Dv != state_arr.shape(2) || Dk != state_arr.shape(3)) {
            throw std::invalid_argument("mlx_gated_delta_with_tape got mismatched head/state shapes");
        }
        if (Hk <= 0 || Dk < 32 || (Dk % 32) != 0 || (Hv % Hk) != 0) {
            throw std::invalid_argument("mlx_gated_delta_with_tape requires Dk multiple of 32 and Hv divisible by Hk");
        }

        std::vector<array> inputs = {q_arr, k_arr, v_arr, g_arr, beta_arr, state_arr, array(T)};
        std::vector<Shape> out_shapes = {
            Shape{B, T, Hv, Dv},
            state_arr.shape(),
            Shape{B, T, Hv, Dv},
        };
        std::vector<Dtype> out_dtypes = {bfloat16, float32, bfloat16};
        std::vector<std::pair<std::string, fast::TemplateArg>> tmpl = {
            {"Dk", fast::TemplateArg(Dk)},
            {"Dv", fast::TemplateArg(Dv)},
            {"Hk", fast::TemplateArg(Hk)},
            {"Hv", fast::TemplateArg(Hv)},
            {"InT", fast::TemplateArg(bfloat16)},
            {"StT", fast::TemplateArg(float32)},
        };

        auto result = gated_delta_tape_kernel()(
            inputs, out_shapes, out_dtypes,
            std::make_tuple(32, Dv, B * Hv),
            std::make_tuple(32, 4, 1),
            tmpl,
            std::nullopt,
            false,
            {});

        *out_y = from_arr(std::move(result[0]));
        *out_state = from_arr(std::move(result[1]));
        *out_tape = from_arr(std::move(result[2]));
    } catch (const std::exception& e) {
        mlx_set_error(e.what());
        if (out_y) {
            *out_y = nullptr;
        }
        if (out_state) {
            *out_state = nullptr;
        }
        if (out_tape) {
            *out_tape = nullptr;
        }
    }
}

mlx_array* mlx_tape_replay(
    mlx_array* tape, mlx_array* k, mlx_array* g, mlx_array* state_in, int steps) {
    MLX_TRY_RETURN([&]() {
        auto tape_arr = contiguous(*to_arr(tape));
        auto k_arr = contiguous(*to_arr(k));
        auto g_arr = contiguous(*to_arr(g));
        auto state_arr = contiguous(*to_arr(state_in));

        require_rank(tape_arr, 4, "tape");
        require_rank(k_arr, 4, "k");
        require_rank(g_arr, 3, "g");
        require_rank(state_arr, 4, "state_in");
        require_dtype(tape_arr, bfloat16, "tape");
        require_dtype(k_arr, bfloat16, "k");
        require_dtype(g_arr, bfloat16, "g");
        require_dtype(state_arr, float32, "state_in");

        int B = tape_arr.shape(0);
        int T = tape_arr.shape(1);
        int Hv = tape_arr.shape(2);
        int Dv = tape_arr.shape(3);
        int Hk = k_arr.shape(2);
        int Dk = k_arr.shape(3);

        if (steps != T || steps != k_arr.shape(1) || steps != g_arr.shape(1)) {
            throw std::invalid_argument("mlx_tape_replay got mismatched step counts");
        }
        if (B != k_arr.shape(0) || B != g_arr.shape(0) || B != state_arr.shape(0)) {
            throw std::invalid_argument("mlx_tape_replay got mismatched batch dimensions");
        }
        if (Hv != g_arr.shape(2) || Hv != state_arr.shape(1) || Dv != state_arr.shape(2) || Dk != state_arr.shape(3)) {
            throw std::invalid_argument("mlx_tape_replay got mismatched tape/state shapes");
        }
        if (Hk <= 0 || Dk < 32 || (Dk % 32) != 0 || (Hv % Hk) != 0) {
            throw std::invalid_argument("mlx_tape_replay requires Dk multiple of 32 and Hv divisible by Hk");
        }

        std::vector<array> inputs = {tape_arr, k_arr, g_arr, state_arr, array(steps)};
        std::vector<Shape> out_shapes = {state_arr.shape()};
        std::vector<Dtype> out_dtypes = {float32};
        std::vector<std::pair<std::string, fast::TemplateArg>> tmpl = {
            {"Dk", fast::TemplateArg(Dk)},
            {"Dv", fast::TemplateArg(Dv)},
            {"Hk", fast::TemplateArg(Hk)},
            {"Hv", fast::TemplateArg(Hv)},
            {"InT", fast::TemplateArg(bfloat16)},
            {"StT", fast::TemplateArg(float32)},
        };

        auto result = tape_replay_kernel()(
            inputs, out_shapes, out_dtypes,
            std::make_tuple(32, Dv, B * Hv),
            std::make_tuple(32, 4, 1),
            tmpl,
            std::nullopt,
            false,
            {});

        return from_arr(std::move(result[0]));
    }());
}

mlx_array* mlx_tape_replay_varlen(
    mlx_array* tape, mlx_array* k, mlx_array* g, mlx_array* state_in, mlx_array* steps_arr) {
    MLX_TRY_RETURN([&]() {
        auto tape_arr = contiguous(*to_arr(tape));
        auto k_arr = contiguous(*to_arr(k));
        auto g_arr = contiguous(*to_arr(g));
        auto state_arr = contiguous(*to_arr(state_in));
        auto steps = contiguous(*to_arr(steps_arr));

        require_rank(tape_arr, 4, "tape");
        require_rank(k_arr, 4, "k");
        require_rank(g_arr, 3, "g");
        require_rank(state_arr, 4, "state_in");
        require_rank(steps, 1, "steps");
        require_dtype(tape_arr, bfloat16, "tape");
        require_dtype(k_arr, bfloat16, "k");
        require_dtype(g_arr, bfloat16, "g");
        require_dtype(state_arr, float32, "state_in");
        require_dtype(steps, int32, "steps");

        int B = tape_arr.shape(0);
        int T_padded = tape_arr.shape(1);
        int Hv = tape_arr.shape(2);
        int Dv = tape_arr.shape(3);
        int Hk = k_arr.shape(2);
        int Dk = k_arr.shape(3);

        if (T_padded != k_arr.shape(1) || T_padded != g_arr.shape(1)) {
            throw std::invalid_argument("mlx_tape_replay_varlen got mismatched step counts");
        }
        if (B != k_arr.shape(0) || B != g_arr.shape(0) || B != state_arr.shape(0) || B != steps.shape(0)) {
            throw std::invalid_argument("mlx_tape_replay_varlen got mismatched batch dimensions");
        }
        if (Hv != g_arr.shape(2) || Hv != state_arr.shape(1) || Dv != state_arr.shape(2) || Dk != state_arr.shape(3)) {
            throw std::invalid_argument("mlx_tape_replay_varlen got mismatched tape/state shapes");
        }
        if (Hk <= 0 || Dk < 32 || (Dk % 32) != 0 || (Hv % Hk) != 0) {
            throw std::invalid_argument("mlx_tape_replay_varlen requires Dk multiple of 32 and Hv divisible by Hk");
        }

        std::vector<array> inputs = {tape_arr, k_arr, g_arr, state_arr, steps, array(T_padded), array(B)};
        std::vector<Shape> out_shapes = {state_arr.shape()};
        std::vector<Dtype> out_dtypes = {float32};
        std::vector<std::pair<std::string, fast::TemplateArg>> tmpl = {
            {"Dk", fast::TemplateArg(Dk)},
            {"Dv", fast::TemplateArg(Dv)},
            {"Hk", fast::TemplateArg(Hk)},
            {"Hv", fast::TemplateArg(Hv)},
            {"InT", fast::TemplateArg(bfloat16)},
            {"StT", fast::TemplateArg(float32)},
        };

        auto result = tape_replay_varlen_kernel()(
            inputs, out_shapes, out_dtypes,
            std::make_tuple(32, Dv, B * Hv),
            std::make_tuple(32, 4, 1),
            tmpl,
            std::nullopt,
            false,
            {});

        return from_arr(std::move(result[0]));
    }());
}

mlx_array* mlx_batched_sdpa_2pass(
    mlx_array* queries, mlx_array* keys, mlx_array* values,
    float scale, int gqa_factor) {
    MLX_TRY_RETURN(from_arr(batched_sdpa_2pass_cpp(
        *to_arr(queries),
        *to_arr(keys),
        *to_arr(values),
        scale,
        gqa_factor
    )));
}

// === Random ===

mlx_array* mlx_random_categorical(mlx_array* logits, int32_t axis) {
    MLX_TRY_RETURN(from_arr(random::categorical(*to_arr(logits), static_cast<int>(axis))));
}

// === Transforms ===

void mlx_eval(mlx_array** arrays, size_t count) {
    try {
        mlx_clear_error();
        std::vector<array> arrs;
        arrs.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            arrs.push_back(*to_arr(arrays[i]));
        }
        eval(arrs);
    } catch (const std::exception& e) {
        mlx_set_error(e.what());
    }
}

// INFER_CPP_PHASE_TIMING=1 enables stderr per-call timing of the
// MLX FFI hot path (async_eval, eval, forward). Cached env probe
// keeps the prod path at one atomic read after first call.
static bool cpp_phase_timing_enabled() {
    static int flag = -1;
    if (flag == -1) {
        const char* v = std::getenv("INFER_CPP_PHASE_TIMING");
        flag = (v && *v && v[0] != '0' && std::string(v) != "false") ? 1 : 0;
    }
    return flag == 1;
}

void mlx_async_eval(mlx_array** arrays, size_t count) {
    try {
        mlx_clear_error();
        bool tracing = cpp_phase_timing_enabled();
        auto t0 = tracing ? std::chrono::high_resolution_clock::now()
                          : std::chrono::high_resolution_clock::time_point{};
        std::vector<array> arrs;
        arrs.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            arrs.push_back(*to_arr(arrays[i]));
        }
        auto t_setup = tracing ? std::chrono::high_resolution_clock::now() : t0;
        async_eval(arrs);
        if (tracing) {
            auto t_end = std::chrono::high_resolution_clock::now();
            auto setup_us = std::chrono::duration_cast<std::chrono::microseconds>(t_setup - t0).count();
            auto async_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_setup).count();
            std::fprintf(stderr,
                "cpp_phase_timing mlx_async_eval count=%zu setup_us=%lld async_eval_call_us=%lld\n",
                count, (long long)setup_us, (long long)async_us);
        }
    } catch (const std::exception& e) {
        mlx_set_error(e.what());
    }
}

// === IO ===

int32_t mlx_load_safetensors(const char* path,
                             const char*** out_names,
                             mlx_array*** out_arrays) {
    mlx_clear_error();
    std::pair<std::unordered_map<std::string, array>, std::unordered_map<std::string, std::string>> result;
    try {
        result = load_safetensors(std::string(path));
    } catch (const std::exception& e) {
        mlx_set_error(e.what());
        *out_names = nullptr;
        *out_arrays = nullptr;
        return -1;
    }
    auto& data = result.first;
    int32_t count = static_cast<int32_t>(data.size());
    if (count == 0) {
        *out_names = nullptr;
        *out_arrays = nullptr;
        return 0;
    }

    auto** names = new const char*[count];
    auto** arrays = new mlx_array*[count];
    int32_t i = 0;
    for (auto& [key, val] : data) {
        // Duplicate the string so Rust can free it later
        char* name = new char[key.size() + 1];
        std::memcpy(name, key.c_str(), key.size() + 1);
        names[i] = name;
        arrays[i] = from_arr(std::move(val));
        ++i;
    }
    *out_names = names;
    *out_arrays = arrays;
    return count;
}

void mlx_free_loaded_tensors(const char** names, mlx_array** arrays, int32_t count) {
    for (int32_t i = 0; i < count; ++i) {
        delete[] names[i];
        delete to_arr(arrays[i]);
    }
    delete[] names;
    delete[] arrays;
}

// === Contiguous ===

mlx_array* mlx_contiguous(mlx_array* a) {
    MLX_TRY_RETURN(from_arr(contiguous(*to_arr(a))));
}

// === Conv1d ===

mlx_array* mlx_conv1d(mlx_array* input, mlx_array* weight,
                      int32_t stride, int32_t padding,
                      int32_t dilation, int32_t groups) {
    MLX_TRY_RETURN(from_arr(conv1d(*to_arr(input), *to_arr(weight),
                                   stride, padding, dilation, groups)));
}

// === Fused compute_g ===
// g = exp(-exp(A_log) * softplus(alpha + dt_bias))
// Fuses 10 ops into 1 C++ call (no heap allocs for intermediates).

mlx_array* mlx_compute_g(mlx_array* A_log, mlx_array* alpha, mlx_array* dt_bias) {
    MLX_TRY_RETURN([&]() {
        auto a = astype(*to_arr(A_log), float32);
        auto ab = *to_arr(alpha) + *to_arr(dt_bias);
        auto sp = where(greater(ab, array(20.0f)), ab, log1p(exp(ab)));
        return from_arr(exp(negative(exp(a)) * sp));
    }());
}

// Metal kernel struct (used by both GDR forward and standalone kernel API)
struct mlx_metal_kernel {
    std::string name;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::string source;
    std::string header;
};

// === Fused GDR layer forward ===
// Full GDR linear attention decode step in C++ — eliminates ~40 FFI calls per layer.
// Matches mlx_lm's GatedDeltaNet.__call__ for decode (T=1).

void mlx_gdr_layer_forward(
    mlx_array* x_in,           // [1, 1, hidden] bf16 input
    // Projection weights (quantized)
    mlx_array* qkvz_w, mlx_array* qkvz_s, mlx_array* qkvz_b,
    int32_t qkvz_gs, int32_t qkvz_bits,
    int32_t qkv_split, int32_t z_split,
    mlx_array* ba_w, mlx_array* ba_s, mlx_array* ba_b,
    int32_t ba_gs, int32_t ba_bits, int32_t ba_num_heads,
    // Conv1d
    mlx_array* conv1d_w,       // [C, K, 1] bf16
    mlx_array** conv_state,    // [1, K-1, C] bf16 — updated in place
    int32_t conv_kernel,
    // Gate params
    mlx_array* a_log,          // [Hv] f32
    mlx_array* dt_bias,        // [Hv] bf16
    // Norm + output
    mlx_array* norm_w,         // [Dv] bf16
    float rms_eps,
    mlx_array* out_w, mlx_array* out_s, mlx_array* out_b,
    int32_t out_gs, int32_t out_bits,
    // Config
    int32_t num_key_heads, int32_t key_dim,
    int32_t num_value_heads, int32_t value_dim,
    float q_scale, float k_scale,
    // State
    mlx_array** gdr_state,     // [1, Hv, Dv, Dk] f32 — updated in place
    // Metal kernel handle
    void* metal_kernel, int32_t use_metal_kernel,
    // Output
    mlx_array* out_result      // receives the output array
) {
    try {
        mlx_clear_error();
        auto x = *to_arr(x_in);
        int hk = num_key_heads, dk = key_dim;
        int hv = num_value_heads, dv = value_dim;
        int q_dim = hk * dk, k_dim_total = q_dim, v_dim = hv * dv;
        int qkv_dim = q_dim + k_dim_total + v_dim;

        // 1. Projections
        auto qkvz = quantized_matmul(x, *to_arr(qkvz_w), *to_arr(qkvz_s), *to_arr(qkvz_b),
                                     true, qkvz_gs, qkvz_bits);
        auto qkv = slice(qkvz, {0, 0, 0}, {1, 1, qkv_split});
        auto z = slice(qkvz, {0, 0, qkv_split}, {1, 1, qkv_split + z_split});

        auto ba = quantized_matmul(x, *to_arr(ba_w), *to_arr(ba_s), *to_arr(ba_b),
                                   true, ba_gs, ba_bits);
        auto b_raw = slice(ba, {0, 0, 0}, {1, 1, ba_num_heads});
        auto a_raw = slice(ba, {0, 0, ba_num_heads}, {1, 1, ba_num_heads * 2});

        // 2. Conv1d (standard depthwise)
        auto conv_st = *to_arr(*conv_state);
        auto conv_input = concatenate({conv_st, qkv}, 1);
        int n_keep = conv_kernel - 1;
        auto new_conv_state = slice(conv_input, {0, 1, 0}, {1, n_keep + 1, qkv_dim});
        delete to_arr(*conv_state);
        *conv_state = from_arr(std::move(new_conv_state));

        auto conv_out = conv1d(conv_input, *to_arr(conv1d_w), 1, 0, 1, qkv_dim);
        conv_out = conv_out * sigmoid(conv_out);

        // 3. Split QKV + RMS normalize
        auto q_raw = reshape(slice(conv_out, {0, 0, 0}, {1, 1, q_dim}), {1, 1, hk, dk});
        auto k_raw = reshape(slice(conv_out, {0, 0, q_dim}, {1, 1, q_dim + k_dim_total}), {1, 1, hk, dk});
        auto v_raw = reshape(
            slice(conv_out, {0, 0, q_dim + k_dim_total}, {1, 1, q_dim + k_dim_total + v_dim}),
            {1, 1, hv, dv});

        auto q = fast::rms_norm(q_raw, std::nullopt, 1e-6f) * array(q_scale);
        auto k = fast::rms_norm(k_raw, std::nullopt, 1e-6f) * array(k_scale);

        // 4. Gate computation
        auto beta = sigmoid(b_raw);
        auto A = *to_arr(a_log);
        auto ab = a_raw + *to_arr(dt_bias);
        auto sp = where(greater(ab, array(20.0f)), ab, log1p(exp(ab)));
        auto g = exp(negative(exp(astype(A, float32))) * sp);

        // 5. Metal kernel state update
        array y_out({0});
        if (use_metal_kernel && metal_kernel) {
            auto q_bf16 = astype(q, bfloat16);
            auto k_bf16 = astype(k, bfloat16);
            auto v_bf16 = astype(v_raw, bfloat16);
            auto g_3d = reshape(g, {1, 1, hv});
            auto beta_3d = reshape(beta, {1, 1, hv});
            auto state_in = *to_arr(*gdr_state);

            auto* mk = static_cast<mlx_metal_kernel*>(metal_kernel);
            auto kernel_fn = fast::metal_kernel(
                mk->name, mk->input_names, mk->output_names,
                mk->source, mk->header, true, false);

            auto t_arr = array(1);
            std::vector<array> inputs = {q_bf16, k_bf16, v_bf16, g_3d, beta_3d, state_in, t_arr};
            std::vector<Shape> out_shapes = {{1, 1, hv, dv}, state_in.shape()};
            std::vector<Dtype> out_dtypes = {bfloat16, float32};
            std::vector<std::pair<std::string, fast::TemplateArg>> tmpl = {
                {"Dk", fast::TemplateArg(dk)}, {"Dv", fast::TemplateArg(dv)},
                {"Hk", fast::TemplateArg(hk)}, {"Hv", fast::TemplateArg(hv)},
                {"InT", fast::TemplateArg(bfloat16)}, {"StT", fast::TemplateArg(float32)}
            };

            auto result = kernel_fn(
                inputs, out_shapes, out_dtypes,
                std::make_tuple(32, dv, 1 * hv),
                std::make_tuple(32, 4, 1),
                tmpl, std::nullopt, false, {});

            y_out = result[0];
            delete to_arr(*gdr_state);
            *gdr_state = from_arr(std::move(result[1]));
        } else {
            auto g_4d = reshape(g, {1, hv, 1, 1});
            auto s = *to_arr(*gdr_state);
            auto s_decayed = s * g_4d;
            int heads_per_key = hv / hk;
            array k_exp = (heads_per_key > 1)
                ? reshape(broadcast_to(expand_dims(k, 2), {1, 1, hk, heads_per_key, dk}), {1, hv, dk})
                : reshape(k, {1, hv, dk});
            array q_exp = (heads_per_key > 1)
                ? reshape(broadcast_to(expand_dims(q, 2), {1, 1, hk, heads_per_key, dk}), {1, hv, dk})
                : reshape(q, {1, hv, dk});
            auto v_3d = reshape(v_raw, {1, hv, dv});
            auto k_4d = reshape(k_exp, {1, hv, 1, dk});
            auto kv_mem = sum(s_decayed * k_4d, -1, false);
            auto beta_3d = reshape(beta, {1, hv, 1});
            auto delta = (v_3d - kv_mem) * beta_3d;
            auto s_updated = s_decayed + reshape(delta, {1, hv, dv, 1}) * k_4d;
            auto q_4d = reshape(q_exp, {1, hv, 1, dk});
            y_out = reshape(sum(s_updated * q_4d, -1, false), {1, 1, hv, dv});
            delete to_arr(*gdr_state);
            *gdr_state = from_arr(std::move(s_updated));
        }

        // 6. Per-head RMSNorm + output gate
        auto y_heads = reshape(y_out, {hv, dv});
        auto normed = fast::rms_norm(y_heads, *to_arr(norm_w), rms_eps);
        auto z_gated = reshape(z, {hv, dv});
        auto silu_z = z_gated * sigmoid(z_gated);
        auto gated_out = normed * silu_z;

        // 7. Output projection
        auto out_flat = reshape(gated_out, {1, hv * dv});
        auto result = quantized_matmul(out_flat, *to_arr(out_w), *to_arr(out_s), *to_arr(out_b),
                                       true, out_gs, out_bits);

        auto** dst = reinterpret_cast<mlx_array**>(out_result);
        *dst = from_arr(std::move(result));
    } catch (const std::exception& e) {
        mlx_set_error(e.what());
        auto** dst = reinterpret_cast<mlx_array**>(out_result);
        *dst = nullptr;
    }
}

// === Metal kernel ===

void* mlx_metal_kernel_new(const char* name,
                           const char** input_names, size_t n_inputs,
                           const char** output_names, size_t n_outputs,
                           const char* source, const char* header) {
    MLX_TRY_RETURN_VALUE(nullptr, [&]() -> void* {
        auto* k = new mlx_metal_kernel();
        k->name = std::string(name);
        for (size_t i = 0; i < n_inputs; ++i) {
            k->input_names.emplace_back(input_names[i]);
        }
        for (size_t i = 0; i < n_outputs; ++i) {
            k->output_names.emplace_back(output_names[i]);
        }
        k->source = std::string(source);
        k->header = std::string(header ? header : "");
        return k;
    }());
}

void mlx_metal_kernel_free(void* kernel) {
    MLX_TRY_VOID(delete static_cast<mlx_metal_kernel*>(kernel));
}

void mlx_metal_kernel_apply(void* kernel,
                            mlx_array** inputs, size_t n_inputs,
                            mlx_array** outputs, size_t n_outputs,
                            const int32_t* grid, const int32_t* threadgroup,
                            const int32_t* output_shapes,
                            const int32_t* output_shape_dims,
                            const int32_t* output_dtypes,
                            const char** template_names,
                            const int32_t* template_int_vals,
                            const int32_t* template_dtype_vals,
                            size_t n_int_templates,
                            size_t n_dtype_templates) {
    try {
        mlx_clear_error();
        auto* k = static_cast<mlx_metal_kernel*>(kernel);

        std::vector<array> in_arrs;
        in_arrs.reserve(n_inputs);
        for (size_t i = 0; i < n_inputs; ++i) {
            in_arrs.push_back(*to_arr(inputs[i]));
        }

        std::vector<Shape> out_shapes;
        std::vector<Dtype> out_dtypes;
        int shape_offset = 0;
        for (size_t i = 0; i < n_outputs; ++i) {
            int ndim = output_shape_dims[i];
            Shape sh(output_shapes + shape_offset, output_shapes + shape_offset + ndim);
            out_shapes.push_back(sh);
            out_dtypes.push_back(to_dtype(output_dtypes[i]));
            shape_offset += ndim;
        }

        std::vector<std::pair<std::string, fast::TemplateArg>> template_args;
        for (size_t i = 0; i < n_int_templates; ++i) {
            template_args.emplace_back(
                std::string(template_names[i]),
                fast::TemplateArg(template_int_vals[i]));
        }
        for (size_t i = 0; i < n_dtype_templates; ++i) {
            template_args.emplace_back(
                std::string(template_names[n_int_templates + i]),
                fast::TemplateArg(to_dtype(template_dtype_vals[i])));
        }

        auto mk = fast::metal_kernel(
            k->name,
            k->input_names,
            k->output_names,
            k->source,
            k->header,
            true,
            false
        );

        auto grid_tuple = std::make_tuple(grid[0], grid[1], grid[2]);
        auto tg_tuple = std::make_tuple(threadgroup[0], threadgroup[1], threadgroup[2]);

        auto result = mk(
            in_arrs, out_shapes, out_dtypes,
            grid_tuple, tg_tuple,
            template_args,
            std::nullopt,
            false,
            {}
        );

        for (size_t i = 0; i < result.size() && i < n_outputs; ++i) {
            outputs[i] = from_arr(std::move(result[i]));
        }
    } catch (const std::exception& e) {
        mlx_set_error(e.what());
    }
}

// === Memory management ===

/// Current active MLX allocator memory in bytes.
size_t mlx_get_active_memory() {
    MLX_TRY_RETURN_VALUE(0, mlx::core::get_active_memory());
}

/// Peak MLX allocator memory in bytes.
size_t mlx_get_peak_memory() {
    MLX_TRY_RETURN_VALUE(0, mlx::core::get_peak_memory());
}

/// Cached MLX allocator memory in bytes.
size_t mlx_get_cache_memory() {
    MLX_TRY_RETURN_VALUE(0, mlx::core::get_cache_memory());
}

/// Set the MLX allocator memory limit in bytes.
size_t mlx_set_memory_limit(size_t limit) {
    MLX_TRY_RETURN_VALUE(0, mlx::core::set_memory_limit(limit));
}

/// Set the MLX allocator cache limit in bytes.
size_t mlx_set_cache_limit(size_t limit) {
    MLX_TRY_RETURN_VALUE(0, mlx::core::set_cache_limit(limit));
}

/// Set the MLX allocator wired limit in bytes.
size_t mlx_set_wired_limit(size_t limit) {
    MLX_TRY_RETURN_VALUE(0, mlx::core::set_wired_limit(limit));
}

/// Release cached Metal buffers and other allocator caches.
/// Equivalent to `mx.metal.clear_cache()` in Python.
void mlx_metal_clear_cache() {
    MLX_TRY_VOID(mlx::core::clear_cache());
}

} // extern "C"
