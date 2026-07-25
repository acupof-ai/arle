//! DeepSeek-OCR (`deepseekocr` / `UnlimitedOCRForCausalLM`) MLX forward model.
//!
//! A vision-language model: a DeepEncoder (SAM-base windowed/global ViT + a 16x
//! conv compressor + a CLIP-large ViT that reuses the SAM patch grid) projects
//! an image into 256 soft tokens, which are spliced into the prompt at the
//! `<image>` placeholder positions and decoded by a DeepSeek-MoE text decoder
//! (plain MHA, layer 0 dense, layers 1-11 MoE with 2 fused shared experts).
//!
//! Decoder/projector weights are MXFP8 (uint8 scales, no biases); SAM/CLIP
//! weights are dense BF16. This mirrors the mlx-vlm `deepseekocr` reference
//! (sam.py / vision.py / language.py / deepseekocr.py).

#include "mlx_common.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

using mlx::core::array;
using DeepseekCancelFn = int32_t (*)(const void*);

bool cancelled(DeepseekCancelFn cancel_fn, const void* cancel_ctx) {
    return cancel_fn != nullptr && cancel_fn(cancel_ctx) != 0;
}

array array_from_i32(const int32_t* data, int32_t len) {
    if (len < 0) {
        throw std::invalid_argument("negative token length");
    }
    if (len > 0 && data == nullptr) {
        throw std::invalid_argument("non-empty token input has null data pointer");
    }
    auto buf = allocator::malloc(static_cast<size_t>(len) * sizeof(int32_t));
    if (len > 0) {
        std::memcpy(buf.raw_ptr(), data, static_cast<size_t>(len) * sizeof(int32_t));
    }
    return array(std::move(buf), Shape{len}, int32);
}

array array_from_f32(const float* data, const Shape& shape) {
    size_t count = 1;
    for (int dim : shape) {
        if (dim < 0) {
            throw std::invalid_argument("negative array shape");
        }
        count *= static_cast<size_t>(dim);
    }
    if (count > 0 && data == nullptr) {
        throw std::invalid_argument("non-empty float input has null data pointer");
    }
    auto buf = allocator::malloc(count * sizeof(float));
    if (count > 0) {
        std::memcpy(buf.raw_ptr(), data, count * sizeof(float));
    }
    return array(std::move(buf), shape, float32);
}

// Env-gated L2-norm trace for numerically bisecting the vision tower.
void vdbg(const char* tag, const array& a) {
    if (std::getenv("INFER_DSOCR_VDEBUG") == nullptr) {
        return;
    }
    auto n = sqrt(sum(astype(a, float32) * astype(a, float32)));
    eval({n});
    std::fprintf(stderr, "[vdbg] %-18s l2=%.4f shape=[", tag, n.item<float>());
    for (int d : a.shape()) std::fprintf(stderr, "%d,", d);
    std::fprintf(stderr, "]\n");
}

// GELU (exact erf form, matching nn.GELU default in the reference).
array gelu(const array& x) {
    return x * (array(0.5f) * (array(1.0f) + erf(x * array(0.7071067811865476f))));
}

// silu(gate) * up
array swiglu(const array& gate, const array& up) {
    return (gate * sigmoid(gate)) * up;
}


// A decoder / projector linear: dense BF16 or MXFP8 (uint8 scales, no biases).
struct QWeight {
    array w = array(0);
    array scales = array(0);
    int group_size = 32;
    int bits = 8;
    bool is_dense = true;

    // y = x @ W.T  (W stored [out, in/pack] for mxfp8, [in, out] for dense).
    array apply(const array& x) const {
        if (is_dense) {
            return matmul(x, w);
        }
        return quantized_matmul(
            x, w, scales, /*biases=*/std::nullopt, /*transpose=*/true,
            group_size, bits, "mxfp8");
    }
};

// A dense weight referenced directly (norms, conv kernels, pos embeds, biases)
// lives in the model's weight registry as a plain `array`; no wrapper needed.

struct DecoderLayer {
    array input_ln = array(0);
    array post_attn_ln = array(0);
    QWeight q_proj;
    QWeight k_proj;
    QWeight v_proj;
    QWeight o_proj;

    bool is_moe = false;
    // dense MLP
    QWeight gate_proj;
    QWeight up_proj;
    QWeight down_proj;
    // MoE
    array router_w = array(0); // dense BF16 [E, H]
    QWeight switch_gate; // stacked [E, Hmoe, H/pack]
    QWeight switch_up;
    QWeight switch_down;
    QWeight shared_gate; // fused [Hshared, H/pack]
    QWeight shared_up;
    QWeight shared_down;
    int num_experts = 0;
    int top_k = 0;
    float routed_scaling = 1.0f;
};

struct LayerCache {
    // Fixed-capacity KV ring, lazily allocated on the first write to the layer's
    // K/V dtype: [1, nkv, cap, hd] for keys, [1, nkv, cap, vhd] for values. Each
    // step writes its slot in place via `slice_update` (O(1) traffic), versus the
    // old `concatenate` that reallocated the whole history every token (O(ctx)).
    array keys = array(0);
    array values = array(0);
    int len = 0; // used length
    int cap = 0; // allocated capacity (0 = not yet allocated)
};


struct SamBlock {
    int window_size = 0; // 0 = global attention
    array norm1_w = array(0);
    array norm1_b = array(0);
    array qkv_w = array(0); // [3*dim, dim]
    array qkv_b = array(0);
    array proj_w = array(0); // [dim, dim]
    array proj_b = array(0);
    array rel_pos_h = array(0);
    array rel_pos_w = array(0);
    array norm2_w = array(0);
    array norm2_b = array(0);
    array lin1_w = array(0); // [mlp, dim]
    array lin1_b = array(0);
    array lin2_w = array(0); // [dim, mlp]
    array lin2_b = array(0);
};

struct ClipLayer {
    array ln1_w = array(0);
    array ln1_b = array(0);
    array qkv_w = array(0); // [3*dim, dim]
    array qkv_b = array(0);
    array out_w = array(0); // [dim, dim]
    array out_b = array(0);
    array ln2_w = array(0);
    array ln2_b = array(0);
    array fc1_w = array(0); // [inter, dim]
    array fc1_b = array(0);
    array fc2_w = array(0); // [dim, inter]
    array fc2_b = array(0);
};

struct VisionWeights {
    bool ready = false;
    int image_token_id = -1;
    // CLIP
    int clip_hidden = 0;
    int clip_inter = 0;
    int clip_layers = 0;
    int clip_heads = 0;
    int clip_patch = 14;
    float clip_eps = 1e-6f;
    // SAM
    int sam_width = 768;
    int sam_layers = 12;
    int sam_heads = 12;
    int sam_patch = 16;
    int sam_window = 14;
    int sam_image = 1024;
    // projector
    int proj_input = 2048;
    int proj_n_embed = 1280;

    // SAM stem
    array sam_patch_embed_w = array(0); // [width, ph, pw, 3]
    array sam_patch_embed_b = array(0);
    array sam_pos_embed = array(0); // [1, gh, gw, width]
    array neck0_w = array(0); // conv [256,1,1,768]
    array neck1_w = array(0); // ln [256]
    array neck1_b = array(0);
    array neck2_w = array(0); // conv [256,3,3,256]
    array neck3_w = array(0); // ln [256]
    array neck3_b = array(0);
    array net2_w = array(0); // conv [512,3,3,256]
    array net3_w = array(0); // conv [1024,3,3,512]
    std::vector<SamBlock> sam_blocks;

    // CLIP stem
    array clip_class_embed = array(0); // [dim]
    array clip_pos_embed = array(0); // [num_pos, dim]
    array clip_pre_ln_w = array(0);
    array clip_pre_ln_b = array(0);
    std::vector<ClipLayer> clip_layers_w;

    // projector + tiling specials
    QWeight projector;
    array projector_bias = array(0);
    array image_newline = array(0);
    array view_separator = array(0);
};

struct DeepseekOcrModel {
    int hidden_size = 0;
    int vocab_size = 0;
    int num_heads = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    int v_head_dim = 0;
    float rms_eps = 1e-6f;
    float rope_theta = 10000.0f;
    int embed_group_size = 32;
    int embed_bits = 8;

    std::vector<array> weights; // weight registry (id -> array)
    array embed_tokens = array(0); // dequantized [vocab, hidden]
    QWeight lm_head;
    array final_norm = array(0);
    std::vector<DecoderLayer> layers;
    VisionWeights vision;

    bool finalized = false;

    // per-request state
    std::vector<LayerCache> layer_caches;
    int context_len = 0;
    int kv_cap = 0; // per-request KV ring capacity (prompt + max_new, 256-rounded)

    array array_by_id(int32_t id) {
        if (id < 0 || id >= static_cast<int32_t>(weights.size())) {
            throw std::invalid_argument("DeepSeek-OCR weight id out of range");
        }
        return weights[static_cast<size_t>(id)];
    }

    array rms(const array& x, const array& weight) const {
        return fast::rms_norm(x, weight, rms_eps);
    }

    array ln(const array& x, const array& weight, const array& bias, float eps) const {
        return fast::layer_norm(x, std::optional<array>(weight), std::optional<array>(bias), eps);
    }

    array token_embeddings(const array& token_ids) const {
        return take(embed_tokens, token_ids, 0);
    }

    array logits_from_hidden(const array& x) const {
        auto y = rms(x, final_norm);
        return lm_head.apply(y);
    }

    array attention(const array& x, const DecoderLayer& layer, LayerCache& cache, int offset) const {
        const int s = x.shape(0);
        auto x3 = reshape(x, {1, s, hidden_size});
        auto q = layer.q_proj.apply(x3);
        auto k = layer.k_proj.apply(x3);
        auto v = layer.v_proj.apply(x3);

        q = transpose(reshape(q, {1, s, num_heads, head_dim}), {0, 2, 1, 3});
        k = transpose(reshape(k, {1, s, num_kv_heads, head_dim}), {0, 2, 1, 3});
        v = transpose(reshape(v, {1, s, num_kv_heads, v_head_dim}), {0, 2, 1, 3});

        q = fast::rope(q, head_dim, /*traditional=*/false, rope_theta, 1.0f, offset);
        k = fast::rope(k, head_dim, /*traditional=*/false, rope_theta, 1.0f, offset);

        // Fixed-capacity KV ring: write this step's K/V in place at the cursor and
        // read back the [0, len+s) prefix. Avoids the per-step full-history
        // `concatenate` (O(ctx) traffic/token) the old path paid; the slot write
        // is donated in place because `layer_caches` is decoded against directly
        // (no aliasing copy), mirroring the canonical Qwen35 cache loop.
        const int cap = (kv_cap > 0) ? kv_cap : (cache.len + s);
        if (cache.cap != cap || cache.len == 0) {
            // (Re)allocate the ring to the request capacity on first write.
            cache.keys = zeros({1, num_kv_heads, cap, head_dim}, k.dtype());
            cache.values = zeros({1, num_kv_heads, cap, v_head_dim}, v.dtype());
            cache.cap = cap;
        }
        const int end = cache.len + s;
        cache.keys =
            slice_update(cache.keys, k, {0, 0, cache.len, 0}, {1, num_kv_heads, end, head_dim});
        cache.values =
            slice_update(cache.values, v, {0, 0, cache.len, 0}, {1, num_kv_heads, end, v_head_dim});
        cache.len = end;
        auto k_full = slice(cache.keys, {0, 0, 0, 0}, {1, num_kv_heads, end, head_dim});
        auto v_full = slice(cache.values, {0, 0, 0, 0}, {1, num_kv_heads, end, v_head_dim});

        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        std::string mask_mode = (s > 1) ? "causal" : "";
        auto attn = fast::scaled_dot_product_attention(q, k_full, v_full, scale, mask_mode);
        auto flat = reshape(transpose(attn, {0, 2, 1, 3}), {1, s, num_heads * v_head_dim});
        auto out = layer.o_proj.apply(flat);
        return reshape(out, {s, hidden_size});
    }

    array dense_mlp(const array& x, const DecoderLayer& layer) const {
        auto gate = layer.gate_proj.apply(x);
        auto up = layer.up_proj.apply(x);
        return layer.down_proj.apply(swiglu(gate, up));
    }

    // SwitchGLU over stacked MXFP8 experts (mlx-lm switch_layers layout).
    array switch_experts(const array& x, const array& inds, const DecoderLayer& layer) const {
        auto x5 = expand_dims(x, std::vector<int>{-2, -3});
        auto gate = gather_qmm(
            x5, layer.switch_gate.w, layer.switch_gate.scales, std::nullopt,
            std::nullopt, inds, /*transpose=*/true,
            layer.switch_gate.group_size, layer.switch_gate.bits, "mxfp8", false);
        auto up = gather_qmm(
            x5, layer.switch_up.w, layer.switch_up.scales, std::nullopt,
            std::nullopt, inds, true,
            layer.switch_up.group_size, layer.switch_up.bits, "mxfp8", false);
        auto h = swiglu(gate, up);
        auto y = gather_qmm(
            h, layer.switch_down.w, layer.switch_down.scales, std::nullopt,
            std::nullopt, inds, true,
            layer.switch_down.group_size, layer.switch_down.bits, "mxfp8", false);
        return squeeze(y, -2);
    }

    array moe(const array& x, const DecoderLayer& layer) const {
        // Dense BF16 router: gates = x @ router_w.T  -> [..., E]
        auto gates = matmul(x, transpose(layer.router_w));
        auto scores = softmax(gates, -1, /*precise=*/true);

        const int kth = layer.num_experts - layer.top_k;
        auto part = argpartition(scores, kth, -1);
        Shape start(part.ndim(), 0);
        Shape stop = part.shape();
        Shape strides(part.ndim(), 1);
        start[part.ndim() - 1] = kth;
        auto inds = slice(part, start, stop, strides);

        auto sel = take_along_axis(scores, inds, -1);
        sel = sel * array(layer.routed_scaling);
        if (sel.dtype() != x.dtype()) {
            sel = astype(sel, x.dtype());
        }

        auto y_switch = switch_experts(x, inds, layer);
        auto y = sum(y_switch * expand_dims(sel, -1), -2, false);

        // Fused shared experts (no gate): silu(gate)*up -> down.
        auto sg = layer.shared_gate.apply(x);
        auto su = layer.shared_up.apply(x);
        auto shared = layer.shared_down.apply(swiglu(sg, su));
        return y + shared;
    }

    array decode(const array& token_ids, const array& input_embeds,
                 std::vector<LayerCache>& caches, int offset) const {
        const int len = token_ids.shape(0);
        if (len <= 0) {
            return array(0);
        }
        if (!finalized) {
            throw std::runtime_error("DeepSeek-OCR model was not finalized");
        }
        if (caches.size() != layers.size()) {
            throw std::runtime_error("DeepSeek-OCR cache size mismatch");
        }
        auto x = input_embeds;
        const bool dbg = std::getenv("INFER_DSOCR_DEBUG") != nullptr && len > 1;
        auto dbg_norm = [&](const char* tag, const array& a) {
            if (!dbg) return;
            auto n = sqrt(sum(astype(a, float32) * astype(a, float32)));
            eval({n});
            std::fprintf(stderr, "[dsocr] %s l2=%.4f shape=[", tag, n.item<float>());
            for (int d : a.shape()) std::fprintf(stderr, "%d,", d);
            std::fprintf(stderr, "]\n");
        };
        dbg_norm("embed", x);
        for (size_t i = 0; i < layers.size(); ++i) {
            const auto& layer = layers[i];
            auto residual = x;
            auto h = rms(x, layer.input_ln);
            h = const_cast<DeepseekOcrModel*>(this)->attention(h, layer, caches[i], offset);
            x = residual + h;

            residual = x;
            h = rms(x, layer.post_attn_ln);
            h = layer.is_moe ? moe(h, layer) : dense_mlp(h, layer);
            x = residual + h;
            if (dbg && (i == 0 || i == 1 || i + 1 == layers.size())) {
                char tag[32];
                std::snprintf(tag, sizeof(tag), "layer%zu", i);
                dbg_norm(tag, x);
            }
        }
        return x;
    }


    // Relative-position bias following sam.py get_rel_pos/add_decomposed_rel_pos
    // with q_size == k_size (no interpolation needed at fixed window/grid).
    array get_rel_pos(int size, const array& rel_pos) const {
        const int max_rel = 2 * size - 1;
        array resized = rel_pos;
        if (rel_pos.shape(0) != max_rel) {
            // Linear interpolation along axis 0.
            const int src = rel_pos.shape(0);
            std::vector<int32_t> idx_floor(max_rel);
            std::vector<float> wts(max_rel);
            const float sc = static_cast<float>(src) / static_cast<float>(max_rel);
            for (int i = 0; i < max_rel; ++i) {
                float pos = static_cast<float>(i) * sc;
                int f = static_cast<int>(std::floor(pos));
                idx_floor[i] = f;
                wts[i] = pos - static_cast<float>(f);
            }
            std::vector<int32_t> idx_ceil(max_rel);
            for (int i = 0; i < max_rel; ++i) {
                idx_ceil[i] = std::min(idx_floor[i] + 1, src - 1);
            }
            auto fl = array_from_i32(idx_floor.data(), static_cast<int32_t>(max_rel));
            auto cl = array_from_i32(idx_ceil.data(), static_cast<int32_t>(max_rel));
            auto w = array_from_f32(wts.data(), Shape{max_rel, 1});
            auto lo = take(astype(rel_pos, float32), fl, 0);
            auto hi = take(astype(rel_pos, float32), cl, 0);
            resized = astype(lo * (array(1.0f) - w) + hi * w, rel_pos.dtype());
        }
        // q_coords - k_coords + (size-1), all with q_size==k_size.
        std::vector<int32_t> rel_idx(static_cast<size_t>(size) * size);
        for (int qi = 0; qi < size; ++qi) {
            for (int ki = 0; ki < size; ++ki) {
                rel_idx[static_cast<size_t>(qi) * size + ki] = (qi - ki) + (size - 1);
            }
        }
        auto idx = array_from_i32(rel_idx.data(), static_cast<int32_t>(size * size));
        auto gathered = take(resized, idx, 0); // [size*size, head_dim]
        return reshape(gathered, {size, size, resized.shape(1)});
    }

    // SAM attention over a [B, H, W, C] window. Returns [B, H, W, C].
    array sam_attention(const array& x, const SamBlock& blk, int h, int w) const {
        const int b = x.shape(0);
        const int dim = x.shape(3);
        const int nh = vision.sam_heads;
        const int hd = dim / nh;
        const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

        auto flat = reshape(x, {b, h * w, dim});
        auto qkv = matmul(flat, transpose(blk.qkv_w)) + blk.qkv_b; // [b, hw, 3*dim]
        qkv = reshape(qkv, {b, h * w, 3, nh, hd});
        qkv = transpose(qkv, {2, 0, 3, 1, 4}); // [3, b, heads, hw, hd]
        auto q = squeeze(slice(qkv, {0, 0, 0, 0, 0}, {1, b, nh, h * w, hd}), 0);
        auto k = squeeze(slice(qkv, {1, 0, 0, 0, 0}, {2, b, nh, h * w, hd}), 0);
        auto v = squeeze(slice(qkv, {2, 0, 0, 0, 0}, {3, b, nh, h * w, hd}), 0);

        std::optional<array> mask = std::nullopt;
        // Decomposed relative position bias.
        auto Rh = get_rel_pos(h, blk.rel_pos_h); // [h, h, hd]
        auto Rw = get_rel_pos(w, blk.rel_pos_w); // [w, w, hd]
        // r_q: [b, heads, h, w, hd]
        auto r_q = reshape(q, {b, nh, h, w, hd});
        // rel_h[b,heads,h,w,hk] = sum_c r_q * Rh[h,hk,c]
        auto rel_h = einsum("bnhwc,hkc->bnhwk", std::vector<array>{r_q, astype(Rh, q.dtype())});
        auto rel_w = einsum("bnhwc,wkc->bnhwk", std::vector<array>{r_q, astype(Rw, q.dtype())});
        // bias[b,heads,(h*w),(h*w)] = rel_h[...,:,None] + rel_w[...,None,:]
        auto bias = reshape(rel_h, {b, nh, h * w, h, 1}) +
                    reshape(rel_w, {b, nh, h * w, 1, w});
        bias = reshape(bias, {b, nh, h * w, h * w});
        mask = bias;

        auto attn = fast::scaled_dot_product_attention(q, k, v, scale, "", mask);
        auto out = reshape(transpose(reshape(attn, {b, nh, h, w, hd}), {0, 2, 3, 1, 4}),
                           {b, h, w, dim});
        out = matmul(reshape(out, {b, h * w, dim}), transpose(blk.proj_w)) + blk.proj_b;
        return reshape(out, {b, h, w, dim});
    }

    // Partition [B,H,W,C] into [B*nw, win, win, C], padding to multiples.
    array window_partition(const array& x, int window, int& hp, int& wp) const {
        const int b = x.shape(0);
        const int h = x.shape(1);
        const int w = x.shape(2);
        const int c = x.shape(3);
        const int pad_h = (window - h % window) % window;
        const int pad_w = (window - w % window) % window;
        array y = x;
        if (pad_h > 0 || pad_w > 0) {
            y = pad(x, std::vector<std::pair<int, int>>{{0, 0}, {0, pad_h}, {0, pad_w}, {0, 0}});
        }
        hp = h + pad_h;
        wp = w + pad_w;
        y = reshape(y, {b, hp / window, window, wp / window, window, c});
        y = transpose(y, {0, 1, 3, 2, 4, 5});
        return reshape(y, {-1, window, window, c});
    }

    array window_unpartition(const array& windows, int window, int hp, int wp, int h, int w) const {
        const int c = windows.shape(3);
        const int b = windows.shape(0) / ((hp / window) * (wp / window));
        auto y = reshape(windows, {b, hp / window, wp / window, window, window, c});
        y = transpose(y, {0, 1, 3, 2, 4, 5});
        y = reshape(y, {b, hp, wp, c});
        if (hp > h || wp > w) {
            y = slice(y, {0, 0, 0, 0}, {b, h, w, c});
        }
        return y;
    }

    array sam_block_forward(const array& x, const SamBlock& blk) const {
        const int h = x.shape(1);
        const int w = x.shape(2);
        auto shortcut = x;
        auto y = ln(x, blk.norm1_w, blk.norm1_b, 1e-6f);
        if (blk.window_size > 0) {
            int hp = 0, wp = 0;
            auto windows = window_partition(y, blk.window_size, hp, wp);
            windows = sam_attention(windows, blk, blk.window_size, blk.window_size);
            y = window_unpartition(windows, blk.window_size, hp, wp, h, w);
        } else {
            y = sam_attention(y, blk, h, w);
        }
        auto x1 = shortcut + y;
        auto m = ln(x1, blk.norm2_w, blk.norm2_b, 1e-6f);
        const int b = m.shape(0);
        m = reshape(m, {b * h * w, m.shape(3)});
        m = matmul(m, transpose(blk.lin1_w)) + blk.lin1_b;
        m = gelu(m);
        m = matmul(m, transpose(blk.lin2_w)) + blk.lin2_b;
        m = reshape(m, {b, h, w, x1.shape(3)});
        return x1 + m;
    }

    // Conv2d with PyTorch-style weight [out, kh, kw, in] (MLX layout) + bias.
    array conv_with_bias(const array& x, const array& w, const std::optional<array>& bias,
                         int stride, int padding) const {
        auto y = conv2d(x, w, {stride, stride}, {padding, padding});
        if (bias.has_value()) {
            y = y + reshape(bias.value(), {1, 1, 1, bias.value().shape(0)});
        }
        return y;
    }

    // SAM encoder: image [1, H, W, 3] -> [1, h16, w16, 1024], flattened later.
    array sam_forward(const array& image) const {
        vdbg("sam.image", image);
        // patch_embed conv (stride patch).
        auto x = conv_with_bias(image, vision.sam_patch_embed_w,
                                std::optional<array>(vision.sam_patch_embed_b),
                                vision.sam_patch, 0); // [1, gh, gw, width]
        vdbg("sam.patch_embed", x);
        x = x + vision.sam_pos_embed;
        vdbg("sam.+pos", x);
        int bi = 0;
        for (const auto& blk : vision.sam_blocks) {
            x = sam_block_forward(x, blk);
            if (bi == 0 || bi == 1 || bi == 2 || bi == 11) {
                char tag[24];
                std::snprintf(tag, sizeof(tag), "sam.block%d", bi);
                vdbg(tag, x);
            }
            ++bi;
        }
        // neck: conv1x1 -> ln -> conv3x3(pad1) -> ln  (channel-last layernorm).
        x = conv_with_bias(x, vision.neck0_w, std::nullopt, 1, 0);
        x = ln(x, vision.neck1_w, vision.neck1_b, 1e-6f);
        x = conv_with_bias(x, vision.neck2_w, std::nullopt, 1, 1);
        x = ln(x, vision.neck3_w, vision.neck3_b, 1e-6f);
        vdbg("sam.neck", x);
        // net_2 / net_3: stride-2 conv compressor 256->512->1024.
        x = conv_with_bias(x, vision.net2_w, std::nullopt, 2, 1);
        x = conv_with_bias(x, vision.net3_w, std::nullopt, 2, 1);
        vdbg("sam_out", x);
        return x; // [1, h', w', 1024]
    }


    array clip_attention(const array& x, const ClipLayer& layer) const {
        const int b = x.shape(0);
        const int s = x.shape(1);
        const int dim = vision.clip_hidden;
        const int hd = dim / vision.clip_heads;
        const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
        auto qkv = matmul(x, transpose(layer.qkv_w)) + layer.qkv_b; // [b, s, 3*dim]
        auto q = slice(qkv, {0, 0, 0}, {b, s, dim});
        auto k = slice(qkv, {0, 0, dim}, {b, s, 2 * dim});
        auto v = slice(qkv, {0, 0, 2 * dim}, {b, s, 3 * dim});
        q = transpose(reshape(q, {b, s, vision.clip_heads, hd}), {0, 2, 1, 3});
        k = transpose(reshape(k, {b, s, vision.clip_heads, hd}), {0, 2, 1, 3});
        v = transpose(reshape(v, {b, s, vision.clip_heads, hd}), {0, 2, 1, 3});
        auto attn = fast::scaled_dot_product_attention(q, k, v, scale, "");
        auto out = reshape(transpose(attn, {0, 2, 1, 3}), {b, s, dim});
        return matmul(out, transpose(layer.out_w)) + layer.out_b;
    }

    array clip_layer_forward(const array& x, const ClipLayer& layer) const {
        auto y = ln(x, layer.ln1_w, layer.ln1_b, vision.clip_eps);
        y = clip_attention(y, layer);
        auto x1 = x + y;
        auto m = ln(x1, layer.ln2_w, layer.ln2_b, vision.clip_eps);
        m = matmul(m, transpose(layer.fc1_w)) + layer.fc1_b;
        m = gelu(m);
        m = matmul(m, transpose(layer.fc2_w)) + layer.fc2_b;
        return x1 + m;
    }

    // CLIP encoder, reusing SAM output as patch embeddings.
    // sam_out: [1, gh, gw, 1024]. Returns [1, 1+gh*gw, 1024].
    array clip_forward(const array& sam_out) const {
        const int gh = sam_out.shape(1);
        const int gw = sam_out.shape(2);
        const int dim = vision.clip_hidden;
        auto patches = reshape(sam_out, {1, gh * gw, dim});
        auto cls = reshape(vision.clip_class_embed, {1, 1, dim});
        auto x = concatenate(std::vector<array>{cls, patches}, 1); // [1, 1+gh*gw, dim]
        const int npos = x.shape(1);
        auto pos = slice(vision.clip_pos_embed, {0, 0}, {npos, dim});
        x = x + reshape(pos, {1, npos, dim});
        x = ln(x, vision.clip_pre_ln_w, vision.clip_pre_ln_b, vision.clip_eps);
        for (const auto& layer : vision.clip_layers_w) {
            x = clip_layer_forward(x, layer);
        }
        vdbg("clip_out", x);
        return x;
    }

    // Full DeepEncoder: image [1,H,W,3] -> projected soft tokens with 2D tiling.
    // Returns embeds [n_soft, hidden] to splice at <image> positions.
    array vision_encode_image(const float* pixels, int height, int width, int /*soft_tokens*/) const {
        if (!vision.ready) {
            throw std::runtime_error("DeepSeek-OCR vision tower not configured");
        }
        // pixels are channel-first [3, H, W]; MLX conv wants [1, H, W, 3].
        auto chw = array_from_f32(pixels, Shape{1, 3, height, width});
        auto image = transpose(chw, {0, 2, 3, 1});

        auto sam_out = sam_forward(image); // [1, gh, gw, 1024]
        const int gh = sam_out.shape(1);
        const int gw = sam_out.shape(2);
        auto clip_out = clip_forward(sam_out); // [1, 1+gh*gw, 1024]
        if (std::getenv("INFER_DSOCR_DEBUG")) {
            auto sn = sqrt(sum(astype(sam_out, float32) * astype(sam_out, float32)));
            auto cn = sqrt(sum(astype(clip_out, float32) * astype(clip_out, float32)));
            eval({sn, cn});
            std::fprintf(stderr, "[dsocr] sam_out gh=%d gw=%d dim=%d l2=%.4f | clip_out seq=%d l2=%.4f\n",
                         gh, gw, sam_out.shape(3), sn.item<float>(), clip_out.shape(1), cn.item<float>());
        }

        // local_features = concat(clip_out[:,1:], sam_out.flatten(1,2)) -> [1, gh*gw, 2048]
        auto clip_patches = slice(clip_out, {0, 1, 0},
                                  {1, clip_out.shape(1), vision.clip_hidden});
        const int sam_dim = sam_out.shape(3);
        auto sam_flat = reshape(sam_out, {1, gh * gw, sam_dim});
        auto feats = concatenate(std::vector<array>{clip_patches, sam_flat}, -1); // [1, gh*gw, 2048]
        vdbg("feats", feats);
        // projector (linear) -> [1, gh*gw, n_embed]
        auto proj = vision.projector.apply(feats) + vision.projector_bias;
        vdbg("proj", proj);
        proj = squeeze(proj, 0); // [gh*gw, n_embed]

        const int hw = proj.shape(0);
        const int n_embed = proj.shape(1);
        const int side = static_cast<int>(std::lround(std::sqrt(static_cast<double>(hw))));
        // reshape to [side, side, n_embed], append image_newline per row, flatten.
        auto grid = reshape(proj, {side, side, n_embed});
        auto newline = reshape(vision.image_newline, {1, 1, n_embed});
        auto newline_col = broadcast_to(newline, Shape{side, 1, n_embed});
        grid = concatenate(std::vector<array>{grid, newline_col}, 1); // [side, side+1, n_embed]
        auto flat = reshape(grid, {side * (side + 1), n_embed});
        auto sep = reshape(vision.view_separator, {1, n_embed});
        return concatenate(std::vector<array>{flat, sep}, 0); // [side*(side+1)+1, n_embed]
    }


    void reset_request_state(int total_tokens) {
        layer_caches.assign(layers.size(), LayerCache{});
        context_len = 0;
        // Round the KV ring capacity up to a 256-token chunk (mirrors the
        // canonical Qwen35 loop) so the ring is allocated once per request.
        constexpr int KV_CACHE_CHUNK = 256;
        const int need = std::max(1, total_tokens);
        kv_cap = ((need + KV_CACHE_CHUNK - 1) / KV_CACHE_CHUNK) * KV_CACHE_CHUNK;
    }

    array commit_tokens_and_logits(const int32_t* tokens, int len) {
        if (len <= 0) {
            throw std::invalid_argument("DeepSeek-OCR empty token commit");
        }
        auto token_ids = array_from_i32(tokens, static_cast<int32_t>(len));
        auto embeds = token_embeddings(token_ids);
        // Decode directly against `layer_caches` (no aliasing copy) so the ring's
        // `slice_update` is donated in place rather than copying the whole buffer.
        auto hidden = decode(token_ids, embeds, layer_caches, context_len);
        context_len += len;
        return logits_from_hidden(hidden);
    }

    array commit_multimodal_and_logits(const int32_t* tokens, int len,
                                       const float* pixels, int height, int width,
                                       int soft_tokens) {
        if (len <= 0) {
            throw std::invalid_argument("DeepSeek-OCR empty multimodal prompt");
        }
        auto token_ids = array_from_i32(tokens, static_cast<int32_t>(len));
        auto x = token_embeddings(token_ids); // [len, hidden]
        auto image_embeds = vision_encode_image(pixels, height, width, soft_tokens);
        const int n_img = image_embeds.shape(0);

        int seen = 0;
        for (int i = 0; i < len; ++i) {
            if (tokens[i] != vision.image_token_id) {
                continue;
            }
            if (seen >= n_img) {
                throw std::invalid_argument("more <image> tokens than vision embeddings");
            }
            auto row = reshape(slice(image_embeds, {seen, 0}, {seen + 1, hidden_size}),
                               {1, hidden_size});
            x = slice_update(x, row, {i, 0}, {i + 1, hidden_size}, {1, 1});
            seen += 1;
        }
        if (seen != n_img) {
            throw std::invalid_argument("<image> placeholder count != vision embedding count");
        }
        auto hidden = decode(token_ids, x, layer_caches, context_len);
        context_len += len;
        return logits_from_hidden(hidden);
    }

    // Lazy greedy argmax of the last logits row -> int32 array [1]. No eval, so
    // the result stays a graph node that the next step can build on.
    array lazy_next_token(const array& logits) const {
        const int rows = logits.shape(0);
        if (rows <= 0) {
            throw std::runtime_error("DeepSeek-OCR logits empty");
        }
        auto last = slice(logits, {rows - 1, 0}, {rows, logits.shape(logits.ndim() - 1)});
        return contiguous(astype(argmax(last, -1, false), int32)); // [1]
    }

    // Commit one lazy token id: embed it, run the decoder (lazy KV-cache update),
    // and return next-position logits. All graph-building, no eval.
    array commit_lazy_token(const array& token_id) {
        auto embeds = token_embeddings(token_id); // [1, hidden]
        auto hidden = decode(token_id, embeds, layer_caches, context_len);
        context_len += 1;
        return logits_from_hidden(hidden);
    }

    bool is_stop(uint32_t token, const uint32_t* stop_ids, int stop_ids_len) const {
        for (int i = 0; i < stop_ids_len; ++i) {
            if (token == stop_ids[i]) {
                return true;
            }
        }
        return false;
    }

    // Software-pipelined greedy decode: build step N+1's graph and async_eval its
    // token BEFORE eval-ing step N's token, so the CPU encodes the next step while
    // the GPU finishes the current one (hides the per-step encode cost that caps
    // the M-series Metal path well below the bandwidth ceiling).
    int generate(array logits, int max_new_tokens, const uint32_t* stop_ids, int stop_ids_len,
                 DeepseekCancelFn cancel_fn, const void* cancel_ctx,
                 uint32_t* out_tokens, int* out_finish) {
        std::vector<int32_t> output;
        output.reserve(static_cast<size_t>(std::max(max_new_tokens, 0)));
        *out_finish = 0;
        if (max_new_tokens <= 0) {
            return 0;
        }

        array y = lazy_next_token(logits); // token 0 (lazy)
        async_eval(y);

        int generated = 0;
        while (generated < max_new_tokens) {
            if (cancelled(cancel_fn, cancel_ctx)) {
                throw std::runtime_error("DeepSeek-OCR generation cancelled");
            }
            const bool last_iter = (generated + 1 >= max_new_tokens);
            // Build the next step's graph speculatively (one wasted step at most if
            // the current token turns out to be a stop token — standard pipelining).
            array next_y = y;
            if (!last_iter) {
                auto next_logits = commit_lazy_token(y);
                next_y = lazy_next_token(next_logits);
                async_eval(next_y);
            }
            // The GPU has been computing `y` while we built the next graph above.
            eval(y);
            const int32_t tok = y.item<int32_t>();
            output.push_back(tok);
            generated++;
            if (is_stop(static_cast<uint32_t>(tok), stop_ids, stop_ids_len)) {
                *out_finish = 1;
                break;
            }
            if (last_iter) {
                break;
            }
            y = next_y;
        }

        for (size_t i = 0; i < output.size(); ++i) {
            out_tokens[i] = static_cast<uint32_t>(output[i]);
        }
        if (std::getenv("INFER_DSOCR_DEBUG")) {
            std::fprintf(stderr, "[dsocr] generated %zu tokens:", output.size());
            for (size_t i = 0; i < output.size() && i < 16; ++i) {
                std::fprintf(stderr, " %d", output[i]);
            }
            std::fprintf(stderr, "\n");
        }
        return static_cast<int>(output.size());
    }
};

DeepseekOcrModel* as_model(void* p) {
    return reinterpret_cast<DeepseekOcrModel*>(p);
}

QWeight make_mxfp8(const array& w, const array& scales, int group_size, int bits) {
    QWeight q;
    q.w = w;
    q.scales = scales;
    q.group_size = group_size;
    q.bits = bits;
    q.is_dense = false;
    return q;
}

} // namespace

extern "C" {

void* deepseek_ocr_new() {
    MLX_TRY_RETURN_VALUE(nullptr, reinterpret_cast<void*>(new DeepseekOcrModel()));
}

void deepseek_ocr_free(void* model) {
    delete as_model(model);
}

int32_t deepseek_ocr_add_dense_weight(void* model, mlx_array* w) {
    MLX_TRY_RETURN_VALUE(-1, [&]() {
        auto* m = as_model(model);
        m->weights.push_back(*to_arr(w));
        return static_cast<int32_t>(m->weights.size() - 1);
    }());
}

int32_t deepseek_ocr_add_mxfp8_weight(void* model, mlx_array* w, mlx_array* scales,
                                      int32_t /*group_size*/, int32_t /*bits*/) {
    // Stored as a plain array in the registry; the consuming push records the
    // quant params. We store the weight only; scales live in the next slot.
    MLX_TRY_RETURN_VALUE(-1, [&]() {
        auto* m = as_model(model);
        m->weights.push_back(*to_arr(w));
        int32_t w_id = static_cast<int32_t>(m->weights.size() - 1);
        m->weights.push_back(*to_arr(scales));
        // Return the weight id; scales id is implicitly w_id + 1.
        return w_id;
    }());
}

void deepseek_ocr_set_config(void* model, int32_t hidden_size, int32_t vocab_size,
                             int32_t num_attention_heads, int32_t num_key_value_heads,
                             int32_t head_dim, int32_t v_head_dim, float rms_norm_eps,
                             float rope_theta) {
    MLX_TRY_VOID([&]() {
        auto* m = as_model(model);
        m->hidden_size = hidden_size;
        m->vocab_size = vocab_size;
        m->num_heads = num_attention_heads;
        m->num_kv_heads = num_key_value_heads;
        m->head_dim = head_dim;
        m->v_head_dim = v_head_dim;
        m->rms_eps = rms_norm_eps;
        m->rope_theta = rope_theta;
    }());
}

void deepseek_ocr_set_embed(void* model, int32_t embed_id, int32_t embed_scales_id,
                            int32_t lm_head_id, int32_t lm_head_scales_id, int32_t final_norm_id,
                            int32_t quant_group_size, int32_t quant_bits) {
    MLX_TRY_VOID([&]() {
        auto* m = as_model(model);
        m->embed_group_size = quant_group_size;
        m->embed_bits = quant_bits;
        // Dequantize embeddings once at load (mxfp8 -> bf16) for the take() path.
        auto w = m->array_by_id(embed_id);
        auto s = m->array_by_id(embed_scales_id);
        m->embed_tokens = dequantize(w, s, std::nullopt, quant_group_size, quant_bits, "mxfp8");
        m->lm_head = make_mxfp8(m->array_by_id(lm_head_id), m->array_by_id(lm_head_scales_id),
                                quant_group_size, quant_bits);
        m->final_norm = m->array_by_id(final_norm_id);
        eval({m->embed_tokens});
    }());
}

int32_t deepseek_ocr_push_layer(void* model, int32_t input_ln_id, int32_t post_attn_ln_id,
                                int32_t q_id, int32_t k_id, int32_t v_id, int32_t o_id,
                                int32_t dense_gate_id, int32_t dense_up_id, int32_t dense_down_id,
                                int32_t router_id, int32_t switch_gate_id, int32_t switch_up_id,
                                int32_t switch_down_id, int32_t shared_gate_id, int32_t shared_up_id,
                                int32_t shared_down_id, int32_t num_experts, int32_t top_k,
                                float routed_scaling_factor) {
    MLX_TRY_RETURN_VALUE(-1, [&]() {
        auto* m = as_model(model);
        const int gs = m->embed_group_size;
        const int bits = m->embed_bits;
        DecoderLayer layer;
        layer.input_ln = m->array_by_id(input_ln_id);
        layer.post_attn_ln = m->array_by_id(post_attn_ln_id);
        layer.q_proj = make_mxfp8(m->array_by_id(q_id), m->array_by_id(q_id + 1), gs, bits);
        layer.k_proj = make_mxfp8(m->array_by_id(k_id), m->array_by_id(k_id + 1), gs, bits);
        layer.v_proj = make_mxfp8(m->array_by_id(v_id), m->array_by_id(v_id + 1), gs, bits);
        layer.o_proj = make_mxfp8(m->array_by_id(o_id), m->array_by_id(o_id + 1), gs, bits);
        if (num_experts > 0) {
            layer.is_moe = true;
            layer.router_w = m->array_by_id(router_id);
            layer.switch_gate = make_mxfp8(m->array_by_id(switch_gate_id),
                                           m->array_by_id(switch_gate_id + 1), gs, bits);
            layer.switch_up = make_mxfp8(m->array_by_id(switch_up_id),
                                         m->array_by_id(switch_up_id + 1), gs, bits);
            layer.switch_down = make_mxfp8(m->array_by_id(switch_down_id),
                                           m->array_by_id(switch_down_id + 1), gs, bits);
            layer.shared_gate = make_mxfp8(m->array_by_id(shared_gate_id),
                                           m->array_by_id(shared_gate_id + 1), gs, bits);
            layer.shared_up = make_mxfp8(m->array_by_id(shared_up_id),
                                         m->array_by_id(shared_up_id + 1), gs, bits);
            layer.shared_down = make_mxfp8(m->array_by_id(shared_down_id),
                                           m->array_by_id(shared_down_id + 1), gs, bits);
            layer.num_experts = num_experts;
            layer.top_k = top_k;
            layer.routed_scaling = routed_scaling_factor;
        } else {
            layer.is_moe = false;
            layer.gate_proj = make_mxfp8(m->array_by_id(dense_gate_id),
                                         m->array_by_id(dense_gate_id + 1), gs, bits);
            layer.up_proj = make_mxfp8(m->array_by_id(dense_up_id),
                                       m->array_by_id(dense_up_id + 1), gs, bits);
            layer.down_proj = make_mxfp8(m->array_by_id(dense_down_id),
                                         m->array_by_id(dense_down_id + 1), gs, bits);
        }
        m->layers.push_back(std::move(layer));
        return 0;
    }());
}

int32_t deepseek_ocr_set_vision_config(void* model, int32_t image_token_id,
                                       int32_t clip_hidden_size, int32_t clip_intermediate_size,
                                       int32_t clip_num_layers, int32_t clip_num_heads,
                                       int32_t clip_patch_size, float clip_layer_norm_eps,
                                       int32_t sam_width, int32_t sam_layers, int32_t sam_heads,
                                       int32_t sam_patch_size, int32_t sam_window_size,
                                       int32_t sam_image_size, int32_t projector_input_dim,
                                       int32_t projector_n_embed) {
    MLX_TRY_RETURN_VALUE(-1, [&]() {
        auto* m = as_model(model);
        auto& v = m->vision;
        v.image_token_id = image_token_id;
        v.clip_hidden = clip_hidden_size;
        v.clip_inter = clip_intermediate_size;
        v.clip_layers = clip_num_layers;
        v.clip_heads = clip_num_heads;
        v.clip_patch = clip_patch_size;
        v.clip_eps = clip_layer_norm_eps;
        v.sam_width = sam_width;
        v.sam_layers = sam_layers;
        v.sam_heads = sam_heads;
        v.sam_patch = sam_patch_size;
        v.sam_window = sam_window_size;
        v.sam_image = sam_image_size;
        v.proj_input = projector_input_dim;
        v.proj_n_embed = projector_n_embed;
        return 0;
    }());
}

int32_t deepseek_ocr_set_sam_stem(void* model, int32_t patch_embed_w_id, int32_t patch_embed_b_id,
                                  int32_t pos_embed_id, int32_t neck0_w_id, int32_t neck1_w_id,
                                  int32_t neck1_b_id, int32_t neck2_w_id, int32_t neck3_w_id,
                                  int32_t neck3_b_id, int32_t net2_w_id, int32_t net3_w_id) {
    MLX_TRY_RETURN_VALUE(-1, [&]() {
        auto* m = as_model(model);
        auto& v = m->vision;
        v.sam_patch_embed_w = m->array_by_id(patch_embed_w_id);
        v.sam_patch_embed_b = m->array_by_id(patch_embed_b_id);
        v.sam_pos_embed = m->array_by_id(pos_embed_id);
        v.neck0_w = m->array_by_id(neck0_w_id);
        v.neck1_w = m->array_by_id(neck1_w_id);
        v.neck1_b = m->array_by_id(neck1_b_id);
        v.neck2_w = m->array_by_id(neck2_w_id);
        v.neck3_w = m->array_by_id(neck3_w_id);
        v.neck3_b = m->array_by_id(neck3_b_id);
        v.net2_w = m->array_by_id(net2_w_id);
        v.net3_w = m->array_by_id(net3_w_id);
        return 0;
    }());
}

int32_t deepseek_ocr_push_sam_block(void* model, int32_t window_size, int32_t norm1_w_id,
                                    int32_t norm1_b_id, int32_t qkv_w_id, int32_t qkv_b_id,
                                    int32_t proj_w_id, int32_t proj_b_id, int32_t rel_pos_h_id,
                                    int32_t rel_pos_w_id, int32_t norm2_w_id, int32_t norm2_b_id,
                                    int32_t lin1_w_id, int32_t lin1_b_id, int32_t lin2_w_id,
                                    int32_t lin2_b_id) {
    MLX_TRY_RETURN_VALUE(-1, [&]() {
        auto* m = as_model(model);
        SamBlock blk;
        blk.window_size = window_size;
        blk.norm1_w = m->array_by_id(norm1_w_id);
        blk.norm1_b = m->array_by_id(norm1_b_id);
        blk.qkv_w = m->array_by_id(qkv_w_id);
        blk.qkv_b = m->array_by_id(qkv_b_id);
        blk.proj_w = m->array_by_id(proj_w_id);
        blk.proj_b = m->array_by_id(proj_b_id);
        blk.rel_pos_h = m->array_by_id(rel_pos_h_id);
        blk.rel_pos_w = m->array_by_id(rel_pos_w_id);
        blk.norm2_w = m->array_by_id(norm2_w_id);
        blk.norm2_b = m->array_by_id(norm2_b_id);
        blk.lin1_w = m->array_by_id(lin1_w_id);
        blk.lin1_b = m->array_by_id(lin1_b_id);
        blk.lin2_w = m->array_by_id(lin2_w_id);
        blk.lin2_b = m->array_by_id(lin2_b_id);
        m->vision.sam_blocks.push_back(std::move(blk));
        return 0;
    }());
}

int32_t deepseek_ocr_set_clip_stem(void* model, int32_t class_embedding_id,
                                   int32_t position_embedding_id, int32_t pre_layernorm_w_id,
                                   int32_t pre_layernorm_b_id) {
    MLX_TRY_RETURN_VALUE(-1, [&]() {
        auto* m = as_model(model);
        auto& v = m->vision;
        v.clip_class_embed = m->array_by_id(class_embedding_id);
        v.clip_pos_embed = m->array_by_id(position_embedding_id);
        v.clip_pre_ln_w = m->array_by_id(pre_layernorm_w_id);
        v.clip_pre_ln_b = m->array_by_id(pre_layernorm_b_id);
        return 0;
    }());
}

int32_t deepseek_ocr_push_clip_layer(void* model, int32_t ln1_w_id, int32_t ln1_b_id,
                                     int32_t qkv_w_id, int32_t qkv_b_id, int32_t out_w_id,
                                     int32_t out_b_id, int32_t ln2_w_id, int32_t ln2_b_id,
                                     int32_t fc1_w_id, int32_t fc1_b_id, int32_t fc2_w_id,
                                     int32_t fc2_b_id) {
    MLX_TRY_RETURN_VALUE(-1, [&]() {
        auto* m = as_model(model);
        ClipLayer layer;
        layer.ln1_w = m->array_by_id(ln1_w_id);
        layer.ln1_b = m->array_by_id(ln1_b_id);
        layer.qkv_w = m->array_by_id(qkv_w_id);
        layer.qkv_b = m->array_by_id(qkv_b_id);
        layer.out_w = m->array_by_id(out_w_id);
        layer.out_b = m->array_by_id(out_b_id);
        layer.ln2_w = m->array_by_id(ln2_w_id);
        layer.ln2_b = m->array_by_id(ln2_b_id);
        layer.fc1_w = m->array_by_id(fc1_w_id);
        layer.fc1_b = m->array_by_id(fc1_b_id);
        layer.fc2_w = m->array_by_id(fc2_w_id);
        layer.fc2_b = m->array_by_id(fc2_b_id);
        m->vision.clip_layers_w.push_back(std::move(layer));
        return 0;
    }());
}

int32_t deepseek_ocr_set_projector(void* model, int32_t projector_w_id, int32_t projector_bias_id,
                                   int32_t image_newline_id, int32_t view_separator_id) {
    MLX_TRY_RETURN_VALUE(-1, [&]() {
        auto* m = as_model(model);
        auto& v = m->vision;
        // projector is MXFP8: w at id, scales at id+1.
        v.projector = make_mxfp8(m->array_by_id(projector_w_id),
                                 m->array_by_id(projector_w_id + 1),
                                 m->embed_group_size, m->embed_bits);
        v.projector_bias = m->array_by_id(projector_bias_id);
        v.image_newline = m->array_by_id(image_newline_id);
        v.view_separator = m->array_by_id(view_separator_id);
        v.ready = true;
        return 0;
    }());
}

int32_t deepseek_ocr_finalize(void* model) {
    MLX_TRY_RETURN_VALUE(-1, [&]() {
        auto* m = as_model(model);
        m->finalized = true;
        return 0;
    }());
}

int32_t deepseek_ocr_generate_causal(void* model, const int32_t* prompt, int32_t prompt_len,
                                     int32_t max_new_tokens, uint64_t seed, const uint32_t* stop_ids,
                                     int32_t stop_ids_len,
                                     int32_t (*cancel_fn)(const void*), const void* cancel_ctx,
                                     uint32_t* out_tokens, int32_t* out_len, int32_t* out_finish) {
    MLX_TRY_RETURN_VALUE(-1, [&]() {
        auto* m = as_model(model);
        if (prompt_len <= 0 || prompt == nullptr) {
            throw std::invalid_argument("DeepSeek-OCR generate requires a non-empty prompt");
        }
        random::seed(seed);
        m->reset_request_state(prompt_len + std::max(max_new_tokens, 0));
        if (cancelled(cancel_fn, cancel_ctx)) {
            throw std::runtime_error("DeepSeek-OCR generation cancelled");
        }
        auto logits = m->commit_tokens_and_logits(prompt, prompt_len);
        int finish = 0;
        int n = m->generate(logits, max_new_tokens, stop_ids, stop_ids_len, cancel_fn, cancel_ctx,
                            out_tokens, &finish);
        *out_len = n;
        *out_finish = finish;
        return 0;
    }());
}

int32_t deepseek_ocr_generate_causal_image(void* model, const int32_t* prompt, int32_t prompt_len,
                                           const float* image_pixels, int32_t image_height,
                                           int32_t image_width, int32_t image_soft_tokens,
                                           int32_t max_new_tokens, uint64_t seed,
                                           const uint32_t* stop_ids, int32_t stop_ids_len,
                                           int32_t (*cancel_fn)(const void*), const void* cancel_ctx,
                                           uint32_t* out_tokens, int32_t* out_len,
                                           int32_t* out_finish) {
    MLX_TRY_RETURN_VALUE(-1, [&]() {
        auto* m = as_model(model);
        if (prompt_len <= 0 || prompt == nullptr) {
            throw std::invalid_argument("DeepSeek-OCR image generate requires a non-empty prompt");
        }
        random::seed(seed);
        m->reset_request_state(prompt_len + std::max(max_new_tokens, 0));
        if (cancelled(cancel_fn, cancel_ctx)) {
            throw std::runtime_error("DeepSeek-OCR generation cancelled");
        }
        auto logits = m->commit_multimodal_and_logits(prompt, prompt_len, image_pixels,
                                                      image_height, image_width, image_soft_tokens);
        int finish = 0;
        int n = m->generate(logits, max_new_tokens, stop_ids, stop_ids_len, cancel_fn, cancel_ctx,
                            out_tokens, &finish);
        *out_len = n;
        *out_finish = finish;
        return 0;
    }());
}

} // extern "C"
