#include "mlx_common.h"
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


const char* mlx_last_error() {
    return g_mlx_last_error.empty() ? nullptr : g_mlx_last_error.c_str();
}

mlx_array* mlx_array_new_float32(float val) {
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

mlx_array* mlx_exp(mlx_array* a) {
    MLX_TRY_RETURN(from_arr(exp(*to_arr(a))));
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

mlx_array* mlx_zeros(const int32_t* shape, size_t ndim, int32_t dtype) {
    MLX_TRY_RETURN([&]() {
        auto sh = make_shape(shape, ndim);
        return from_arr(zeros(sh, to_dtype(dtype)));
    }());
}


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


mlx_array* mlx_sum_axis(mlx_array* a, int32_t axis, bool keepdims) {
    MLX_TRY_RETURN(from_arr(sum(*to_arr(a), static_cast<int>(axis), keepdims)));
}

mlx_array* mlx_mean_axis(mlx_array* a, int32_t axis, bool keepdims) {
    MLX_TRY_RETURN(from_arr(mean(*to_arr(a), static_cast<int>(axis), keepdims)));
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


mlx_array* mlx_quantized_matmul(mlx_array* x, mlx_array* w, mlx_array* scales,
                                mlx_array* biases, bool transpose,
                                int32_t group_size, int32_t bits, int32_t mode) {
    MLX_TRY_RETURN(from_arr(quantized_matmul(
        *to_arr(x), *to_arr(w), *to_arr(scales),
        to_arr_opt(biases),
        transpose, group_size, bits, quant_mode_str(mode))));
}

mlx_array* mlx_dequantize(mlx_array* w, mlx_array* scales, mlx_array* biases,
                          int32_t group_size, int32_t bits, int32_t mode) {
    MLX_TRY_RETURN(from_arr(dequantize(
        *to_arr(w), *to_arr(scales),
        to_arr_opt(biases),
        group_size, bits, quant_mode_str(mode))));
}

mlx_array* mlx_fast_rms_norm(mlx_array* x, mlx_array* weight, float eps) {
    MLX_TRY_RETURN([&]() {
        if (weight == nullptr) {
            return from_arr(fast::rms_norm(*to_arr(x), std::nullopt, eps));
        }
        return from_arr(fast::rms_norm(*to_arr(x), *to_arr(weight), eps));
    }());
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


mlx_array* mlx_contiguous(mlx_array* a) {
    MLX_TRY_RETURN(from_arr(contiguous(*to_arr(a))));
}


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
