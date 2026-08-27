//! Qwen3.8-Flash-Next (`qwen4_exp`) forward on the Vulkan lane.
//!
//! This module owns the model's OWN forward and arena — deliberately not the
//! shared `forward.rs` machinery, whose `DeviceArena` is sized for a 2560-wide
//! qwen35 residual and whose MoE contract quantizes activations to q8_1. This
//! model needs a 10240-wide hyper residual, PLAIN-F32 activations into the
//! NVFP4 expert GEMVs, and stages (`hyper-connection`, PLE) qwen35 does not
//! have.
//!
//! ## Two implementations of every stage, one semantics
//!
//! Every stage exists twice, on purpose:
//!
//! - a HOST f32 transcription of `modeling_qwen4_exp.py` (matvecs accumulate
//!   in f64, elementwise math in f32/f64), which is both the fallback
//!   execution path and the ORACLE the device is measured against;
//! - a DEVICE path over the proven kernels (`rms_norm` grouped,
//!   `qwen36_router_gemv` as the generic F32 GEMV, `Qwen4HcMix`/`Qwen4HcCombine`,
//!   `Qwen4PleGate`/`Qwen4PleConv`, `GemvIdNvfp4` + `Qwen36RouterTopk` +
//!   `Qwen36MoeWeightedAccum` for the 512-expert MoE, `RopeNeox` + `F16KvPack` +
//!   `FlashAttn` + `SigmoidMul` for full attention, `Qwen35SsmConv` +
//!   `Qwen35GatedDeltaNet` for the linear layers).
//!
//! `tests/qwen4_forward.rs` is the parity harness: it runs both on real
//! weights and reports max relative error per stage.
//!
//! ## Deliberate deviations from a bit-exact bf16 reference
//!
//! The checkpoint is bf16 (+ NVFP4 experts); the reference implementation runs
//! it in bf16. This lane computes in f32, so per-op bf16 rounding differences
//! versus the true bf16 model are inherent and NOT what the parity harness
//! measures — it measures device-vs-host of the same f32 semantics. Three
//! places intentionally match the DEVICE contract rather than pure f32:
//!
//! - the linear-attention depthwise conv rounds its pre-SiLU sum to bf16
//!   (`qwen35_ssm_conv.comp` does; so does torch's bf16 `F.conv1d` output);
//! - full-attention K/V round through f16 at cache-write time (the device KV
//!   cache is f16; error ≤ 2^-11 relative, below bf16's own 2^-8);
//! - the gated-delta recurrence accumulates in f32 in the kernel's serial
//!   order (torch also holds this state in f32). The kernel l2-normalizes q/k
//!   with eps 1e-12 where the reference uses 1e-6; on 128-wide heads with O(1)
//!   post-conv values the difference is O(1e-8) relative and is part of the
//!   reported parity error.
//!
//! ## HF-vs-GGUF head-mapping trap (the V-slot permutation)
//!
//! `qwen35_gated_delta_net.comp` maps value head `v` to key head
//! `v % num_key_heads` (llama.cpp's `ggml_repeat` TILES — correct for GGUF
//! checkpoints, whose converter lays heads out for it). This checkpoint is HF
//! safetensors: `repeat_interleave` wants value head `v` to read key head
//! `v / (nv/nk)`. The fix is a load-time permutation of the VALUE-head axis
//! (see [`v_slot_perm`]): storing original value head `perm(s)` at slot `s`
//! makes the tiled map compute the interleaved one. The permutation touches
//! `in_proj_qkv`'s V rows, `in_proj_z`/`in_proj_a`/`in_proj_b` rows, the conv
//! V channels, `A_log`/`dt_bias`, and `out_proj`'s columns — and nothing else,
//! which `v_slot_perm_is_a_bijection_and_fixes_the_k_head_map` pins.
//!
//! ## The QSA indexer is stubbed, exactly
//!
//! For `<= budget + compress_ratio - 1 = 2051` visible tokens the indexer
//! keeps every token (`Qwen4QsaConfig::dense_below_or_equal`), so full
//! attention is plain causal SDPA. `max_context` (2048) is enforced per
//! forward, making the stub exact rather than approximate. MTP and vision are
//! dropped.

use std::collections::BTreeMap;

use anyhow::{Context, Result, anyhow, ensure};

use infer_gguf::dequant::dequantize_row_nvfp4;
use infer_gguf::safetensors::SafeTensorsDir;

use crate::qwen4_config::{GateActivation, Qwen4ExpConfig, Qwen4LayerType};
use crate::qwen4_hc::{self, GatedResidualWeights, HyperConnectionConfig};
use crate::qwen4_names::{ExpertProj, HcSite, Nvfp4Part};
use crate::qwen4_ple::{NGramContext, NGramHash, PleConfig, PleConvState, PleLayer, PleWeights};
use crate::qwen4_upload::{
    Qwen4DeviceFormat, Qwen4HostTables, Qwen4UploadConfig, Qwen4UploadScope, Qwen4Weights,
    bf16_to_f32, expert_tensor_name, f32_to_f16, layer_tensor_name, plan_qwen4_upload,
    upload_qwen4,
};

use vulkan_kernels::{
    FlashAttentionSpec, Kernel, KernelCache, KernelParams, MAT_VEC_FUSION_SCALE0,
    f16_kv_pack_dispatch, f16_kv_pack_params, flash_attn_dispatch, flash_attn_params,
    gemv_id_dispatch, gemv_id_params_fused, qwen4_hc_combine_dispatch, qwen4_hc_combine_params,
    qwen4_hc_mix_dispatch, qwen4_hc_mix_params, qwen4_ple_conv_dispatch, qwen4_ple_conv_params,
    qwen4_ple_gate_dispatch, qwen4_ple_gate_params, qwen35_gated_delta_net_dispatch,
    qwen35_gated_delta_net_params, qwen35_ssm_conv_dispatch, qwen35_ssm_conv_params,
    qwen36_moe_weighted_accum_dispatch, qwen36_moe_weighted_accum_params,
    qwen36_router_gemv_dispatch, qwen36_router_gemv_params, qwen36_router_topk_dispatch,
    qwen36_router_topk_params, record_dispatch, repack_nvfp4_planes, rms_norm_dispatch_rows,
    rms_norm_params_grouped, rms_norm_params_rows, rope_neox_dispatch, rope_neox_params,
    sigmoid_mul_dispatch, sigmoid_mul_params, swiglu_dispatch, swiglu_params,
};
use vulkan_sys::{CommandRecorder, DescriptorSet, DeviceBuffer, VulkanContext};

// ─────────────────────────────────────────────────────────────────────────────
// Small host math, shared by the transcription and the loaders.
// ─────────────────────────────────────────────────────────────────────────────

fn sigmoid64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid32(x: f32) -> f32 {
    sigmoid64(f64::from(x)) as f32
}

fn silu32(x: f32) -> f32 {
    (f64::from(x) * sigmoid64(f64::from(x))) as f32
}

/// `softplus` with the SAME threshold as `qwen35_gated_delta_net.comp` (and
/// torch's default 20): above it the identity is returned so `exp` cannot
/// overflow.
fn softplus32(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + f64::from(x).exp()).ln() as f32
    }
}

/// Round an f32 to the nearest bf16 (ties to even), returned as f32 — the
/// rounding `qwen35_ssm_conv.comp` applies to the conv sum before SiLU.
fn round_to_bf16(x: f32) -> f32 {
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    f32::from_bits((bits.wrapping_add(0x7fff + lsb)) & 0xffff_0000)
}

/// f32 → f16 → f32 round trip (RNE), the device KV cache's storage rounding.
fn round_to_f16(x: f32) -> f32 {
    infer_gguf::dequant::f16_to_f32(f32_to_f16(x))
}

/// `y[o] = Σ_i w_bf16[o][i] · x[i]` over a row-major bf16 byte slice,
/// accumulating in f64, split across threads by output row. This is the host
/// lane for every dense projection: the mmap'd bf16 bytes are read directly
/// (converting to a cached f32 copy would double the memory traffic).
fn matvec_bf16(w: &[u8], in_dim: usize, out_dim: usize, x: &[f32]) -> Vec<f32> {
    assert_eq!(w.len(), in_dim * out_dim * 2, "bf16 weight byte length");
    assert_eq!(x.len(), in_dim, "activation width");
    let mut y = vec![0.0f32; out_dim];
    let row_bytes = in_dim * 2;
    // Threading pays only when the matrix is large; small projections (a/b,
    // router) stay serial to avoid thread-spawn overhead per token.
    let threads = if in_dim * out_dim >= 1 << 20 {
        std::thread::available_parallelism()
            .map_or(1, std::num::NonZero::get)
            .min(16)
    } else {
        1
    };
    let chunk_rows = out_dim.div_ceil(threads);
    std::thread::scope(|s| {
        for (t, y_chunk) in y.chunks_mut(chunk_rows).enumerate() {
            let w = &w[t * chunk_rows * row_bytes..];
            s.spawn(move || {
                for (r, dst) in y_chunk.iter_mut().enumerate() {
                    let row = &w[r * row_bytes..(r + 1) * row_bytes];
                    let mut acc = 0.0f64;
                    for (pair, &xv) in row.chunks_exact(2).zip(x) {
                        let wv = bf16_to_f32(u16::from_le_bytes([pair[0], pair[1]]));
                        acc += f64::from(wv) * f64::from(xv);
                    }
                    *dst = acc as f32;
                }
            });
        }
    });
    y
}

/// Decode a bf16 tensor into f32.
fn bf16_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|p| bf16_to_f32(u16::from_le_bytes([p[0], p[1]])))
        .collect()
}

/// Device slot `s` of `qwen35_gated_delta_net.comp` reads key head `s % nk`;
/// HF `repeat_interleave` wants value head `v` to read key head `v / (nv/nk)`.
/// Storing original value head `perm(s) = (s % nk) * (nv/nk) + s / nk` at slot
/// `s` makes the kernel's tiled map compute the interleaved one:
/// `perm(s) / (nv/nk) == s % nk` for every slot.
#[must_use]
pub const fn v_slot_perm(nk: usize, nv: usize, slot: usize) -> usize {
    let group = nv / nk;
    (slot % nk) * group + slot / nk
}

// ─────────────────────────────────────────────────────────────────────────────
// Host weights: mmap-borrowed bf16 for the dense tier, f32 for the small tier.
// ─────────────────────────────────────────────────────────────────────────────

/// One dense projection: row-major `[out_dim, in_dim]` bf16 bytes borrowed
/// from the checkpoint mmap.
pub struct HostDense<'st> {
    bytes: &'st [u8],
    /// Contraction width.
    pub in_dim: usize,
    /// Output rows.
    pub out_dim: usize,
}

impl<'st> HostDense<'st> {
    fn load(st: &'st SafeTensorsDir, name: &str, in_dim: usize, out_dim: usize) -> Result<Self> {
        let info = st
            .tensor(name)
            .ok_or_else(|| anyhow!("missing tensor `{name}`"))?;
        ensure!(
            info.dtype == "BF16",
            "`{name}` is {}, expected BF16",
            info.dtype
        );
        // `dims` is GGUF ne order: innermost (in_dim) first.
        ensure!(
            info.dims == [in_dim as u64, out_dim as u64],
            "`{name}` dims {:?} != [{in_dim}, {out_dim}] (ne order)",
            info.dims
        );
        Ok(Self {
            bytes: st.tensor_data(name)?,
            in_dim,
            out_dim,
        })
    }

    /// `W @ x`, f64 accumulation.
    #[must_use]
    pub fn matvec(&self, x: &[f32]) -> Vec<f32> {
        matvec_bf16(self.bytes, self.in_dim, self.out_dim, x)
    }

    /// The full matrix as f32 (row-major `[out, in]`), for device staging.
    #[must_use]
    pub fn to_f32(&self) -> Vec<f32> {
        bf16_vec(self.bytes)
    }
}

/// Load a small bf16 tensor as f32, checking its element count.
fn f32_tensor(st: &SafeTensorsDir, name: &str, expect: usize) -> Result<Vec<f32>> {
    let info = st
        .tensor(name)
        .ok_or_else(|| anyhow!("missing tensor `{name}`"))?;
    ensure!(
        info.dtype == "BF16",
        "`{name}` is {}, expected BF16",
        info.dtype
    );
    ensure!(
        info.element_count() as usize == expect,
        "`{name}` has {} elements, expected {expect}",
        info.element_count()
    );
    Ok(bf16_vec(st.tensor_data(name)?))
}

/// Host weights of one linear-attention (gated-delta) block, HF layout.
pub struct HostLinearAttn<'st> {
    /// `in_proj_qkv` `[10240, 2560]` — `[q(nk·kd) | k(nk·kd) | v(nv·vd)]` rows.
    pub qkv: HostDense<'st>,
    /// `in_proj_z` `[6144, 2560]`.
    pub z: HostDense<'st>,
    /// `in_proj_a` `[48, 2560]`.
    pub a: HostDense<'st>,
    /// `in_proj_b` `[48, 2560]`.
    pub b: HostDense<'st>,
    /// `A_log` `[48]`, raw (the `-exp` is applied in the forward).
    pub a_log: Vec<f32>,
    /// `dt_bias` `[48]`.
    pub dt_bias: Vec<f32>,
    /// `conv1d.weight` `[10240, 1, 4]` flattened to `[channel][tap]`.
    pub conv: Vec<f32>,
    /// `norm.weight` `[128]` — `Qwen4ExpTextRMSNormGated`, PLAIN gain
    /// (ones-initialised; no `1 + w`).
    pub norm: Vec<f32>,
    /// `out_proj` `[2560, 6144]`.
    pub out: HostDense<'st>,
}

/// Host weights of one full-attention block.
pub struct HostFullAttn<'st> {
    /// `q_proj` `[12288, 2560]` — per head `[query(256) | gate(256)]`.
    pub q: HostDense<'st>,
    /// `k_proj` `[512, 2560]`.
    pub k: HostDense<'st>,
    /// `v_proj` `[512, 2560]`.
    pub v: HostDense<'st>,
    /// `o_proj` `[2560, 6144]`.
    pub o: HostDense<'st>,
    /// `q_norm.weight` `[256]`, RAW (host applies `1 + w`).
    pub q_norm: Vec<f32>,
    /// `k_norm.weight` `[256]`, RAW.
    pub k_norm: Vec<f32>,
}

/// Host weights of one MoE block's non-expert tensors (the routed experts stay
/// NVFP4 and are dequantized on demand — see [`host_expert_matvec`]).
pub struct HostMoe<'st> {
    /// `mlp.gate` (the router) `[512, 2560]`.
    pub router: HostDense<'st>,
    /// `mlp.shared_expert_gate` `[1, 2560]`.
    pub shexp_gate: HostDense<'st>,
    /// `mlp.shared_expert.gate_proj` `[640, 2560]`.
    pub sh_gate: HostDense<'st>,
    /// `mlp.shared_expert.up_proj` `[640, 2560]`.
    pub sh_up: HostDense<'st>,
    /// `mlp.shared_expert.down_proj` `[2560, 640]`.
    pub sh_down: HostDense<'st>,
}

/// Everything one decoder layer needs on the host.
pub struct HostLayer<'st> {
    /// Which attention this layer runs.
    pub kind: Qwen4LayerType,
    /// `attn_hyper_connection` weights (f32, RAW `hc_norm` — the oracle
    /// applies `1 + w` itself).
    pub attn_hc: GatedResidualWeights,
    /// `mlp_hyper_connection` weights.
    pub mlp_hc: GatedResidualWeights,
    /// Present iff `kind == LinearAttention`.
    pub linear: Option<HostLinearAttn<'st>>,
    /// Present iff `kind == FullAttention`.
    pub full: Option<HostFullAttn<'st>>,
    /// The MoE's dense tensors.
    pub moe: HostMoe<'st>,
    /// Present iff this is the PLE layer.
    pub ple: Option<PleLayer>,
}

fn load_hc(
    st: &SafeTensorsDir,
    prefix: &str,
    hc: &HyperConnectionConfig,
    mixer: bool,
) -> Result<GatedResidualWeights> {
    let hh = hc.hc_hidden();
    let w = GatedResidualWeights {
        hc_norm: f32_tensor(st, &format!("{prefix}.hc_norm.weight"), hh)?,
        mix_down: f32_tensor(
            st,
            &format!("{prefix}.input_mix_weight_down.weight"),
            hc.hc_lowrank * hh,
        )?,
        mix_up: f32_tensor(
            st,
            &format!("{prefix}.input_mix_weight_up.weight"),
            hh * hc.hc_lowrank,
        )?,
        block_inject: if mixer {
            None
        } else {
            Some(f32_tensor(
                st,
                &format!("{prefix}.block_inject_weight.weight"),
                hc.hc_count * hh,
            )?)
        },
    };
    w.validate(hc)?;
    Ok(w)
}

/// Load one decoder layer's host weights from the checkpoint.
pub fn load_host_layer<'st>(
    st: &'st SafeTensorsDir,
    cfg: &Qwen4ExpConfig,
    layer: usize,
) -> Result<HostLayer<'st>> {
    let h = cfg.hidden_size;
    let hc = hc_config(cfg);
    let kind = cfg.layer_types[layer];
    let name = |suffix: &str| layer_tensor_name(layer, suffix);
    let attn_hc = load_hc(
        st,
        &format!("model.language_model.layers.{layer}.attn_hyper_connection"),
        &hc,
        false,
    )?;
    let mlp_hc = load_hc(
        st,
        &format!("model.language_model.layers.{layer}.mlp_hyper_connection"),
        &hc,
        false,
    )?;

    let (linear, full) = match kind {
        Qwen4LayerType::LinearAttention => {
            let kd = cfg.linear_key_head_dim;
            let vd = cfg.linear_value_head_dim;
            let nk = cfg.linear_num_key_heads;
            let nv = cfg.linear_num_value_heads;
            let conv_dim = 2 * nk * kd + nv * vd;
            let lin = HostLinearAttn {
                qkv: HostDense::load(st, &name("linear_attn.in_proj_qkv.weight"), h, conv_dim)?,
                z: HostDense::load(st, &name("linear_attn.in_proj_z.weight"), h, nv * vd)?,
                a: HostDense::load(st, &name("linear_attn.in_proj_a.weight"), h, nv)?,
                b: HostDense::load(st, &name("linear_attn.in_proj_b.weight"), h, nv)?,
                a_log: f32_tensor(st, &name("linear_attn.A_log"), nv)?,
                dt_bias: f32_tensor(st, &name("linear_attn.dt_bias"), nv)?,
                conv: f32_tensor(
                    st,
                    &name("linear_attn.conv1d.weight"),
                    conv_dim * cfg.linear_conv_kernel_dim,
                )?,
                norm: f32_tensor(st, &name("linear_attn.norm.weight"), vd)?,
                out: HostDense::load(st, &name("linear_attn.out_proj.weight"), nv * vd, h)?,
            };
            (Some(lin), None)
        }
        Qwen4LayerType::FullAttention => {
            let hd = cfg.head_dim;
            let full = HostFullAttn {
                q: HostDense::load(
                    st,
                    &name("self_attn.q_proj.weight"),
                    h,
                    cfg.num_attention_heads * hd * 2,
                )?,
                k: HostDense::load(
                    st,
                    &name("self_attn.k_proj.weight"),
                    h,
                    cfg.num_key_value_heads * hd,
                )?,
                v: HostDense::load(
                    st,
                    &name("self_attn.v_proj.weight"),
                    h,
                    cfg.num_key_value_heads * hd,
                )?,
                o: HostDense::load(
                    st,
                    &name("self_attn.o_proj.weight"),
                    cfg.num_attention_heads * hd,
                    h,
                )?,
                q_norm: f32_tensor(st, &name("self_attn.q_norm.weight"), hd)?,
                k_norm: f32_tensor(st, &name("self_attn.k_norm.weight"), hd)?,
            };
            (None, Some(full))
        }
    };

    let moe = HostMoe {
        router: HostDense::load(st, &name("mlp.gate.weight"), h, cfg.num_experts)?,
        shexp_gate: HostDense::load(st, &name("mlp.shared_expert_gate.weight"), h, 1)?,
        sh_gate: HostDense::load(
            st,
            &name("mlp.shared_expert.gate_proj.weight"),
            h,
            cfg.shared_expert_intermediate_size,
        )?,
        sh_up: HostDense::load(
            st,
            &name("mlp.shared_expert.up_proj.weight"),
            h,
            cfg.shared_expert_intermediate_size,
        )?,
        sh_down: HostDense::load(
            st,
            &name("mlp.shared_expert.down_proj.weight"),
            cfg.shared_expert_intermediate_size,
            h,
        )?,
    };

    let ple = if cfg.is_ple_layer(layer) {
        let pc = ple_config(cfg);
        let hh = pc.hc_hidden();
        let weights = PleWeights {
            key_proj: f32_tensor(st, &name("ple.key_proj.weight"), hh * pc.ple_embed_dim)?,
            value_proj: f32_tensor(
                st,
                &name("ple.value_proj.weight"),
                pc.hidden_size * pc.ple_embed_dim,
            )?,
            norm_key: f32_tensor(st, &name("ple.norm_key.weight"), hh)?,
            norm_query: f32_tensor(st, &name("ple.norm_query.weight"), hh)?,
            norm_conv: f32_tensor(st, &name("ple.norm_conv.weight"), hh)?,
            conv1d: f32_tensor(st, &name("ple.conv1d.weight"), hh * pc.conv_kernel_size)?,
        };
        Some(PleLayer::new(pc, weights)?)
    } else {
        None
    };

    Ok(HostLayer {
        kind,
        attn_hc,
        mlp_hc,
        linear,
        full,
        moe,
        ple,
    })
}

/// The stream-level `hyper_connection_mixer` weights (`use_combine = false`),
/// as f32 — for the parity harness and the host mixer stage.
pub fn load_mixer_weights(
    st: &SafeTensorsDir,
    cfg: &Qwen4ExpConfig,
) -> Result<GatedResidualWeights> {
    load_hc(
        st,
        "model.language_model.hyper_connection_mixer",
        &hc_config(cfg),
        true,
    )
}

/// The [`HyperConnectionConfig`] this checkpoint's config implies.
#[must_use]
pub fn hc_config(cfg: &Qwen4ExpConfig) -> HyperConnectionConfig {
    HyperConnectionConfig {
        hidden_size: cfg.hidden_size,
        hc_count: cfg.hc_count,
        hc_lowrank: cfg.hc_lowrank,
        rms_norm_eps: cfg.rms_norm_eps,
    }
}

/// The [`PleConfig`] this checkpoint's config implies. The conv dilation is
/// `ngram_size`, exactly as `Qwen4ExpTextPLELayer.__init__` sets it.
#[must_use]
pub fn ple_config(cfg: &Qwen4ExpConfig) -> PleConfig {
    PleConfig {
        hidden_size: cfg.hidden_size,
        hc_count: cfg.hc_count,
        ple_embed_dim: cfg.ple_embed_dim,
        conv_kernel_size: cfg.ple_conv_kernel_size,
        conv_dilation: cfg.ngram_size,
        rms_norm_eps: cfg.rms_norm_eps,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-slot state.
// ─────────────────────────────────────────────────────────────────────────────

/// Host K/V cache of one full-attention layer: `[pos][kv_head][head_dim]` f32,
/// K stored post-RoPE. Values are rounded through f16 at write time — the
/// device cache stores f16, and holding the host cache to the same contract
/// keeps the two attentions comparable (the rounding is ≤ 2^-11 relative,
/// below the checkpoint's own bf16 precision).
#[derive(Debug, Clone, Default)]
pub struct HostKv {
    /// Post-RoPE keys.
    pub k: Vec<f32>,
    /// Values.
    pub v: Vec<f32>,
}

/// Per-slot recurrent state for one sequence. `seq_len` is the number of
/// tokens already consumed; a forward at `start_pos` must have
/// `seq_len == start_pos`.
pub struct Qwen4ExpState {
    /// Materialized sequence length.
    pub seq_len: usize,
    /// Gated-delta S matrix per LINEAR layer id: `[nv][kd][vd]` f32 in HF
    /// value-head order.
    pub gdr: BTreeMap<usize, Vec<f32>>,
    /// Depthwise conv history per LINEAR layer id: `[kernel-1][channels]`
    /// time-major, oldest first.
    pub conv: BTreeMap<usize, Vec<f32>>,
    /// Host K/V per FULL layer id.
    pub kv: BTreeMap<usize, HostKv>,
    /// PLE short-conv history per PLE layer id.
    pub ple_conv: BTreeMap<usize, PleConvState>,
    /// N-gram hash context (the previous `ngram_size - 1` token ids).
    pub ngram: NGramContext,
}

impl Qwen4ExpState {
    /// Fresh (sequence-start) state for `cfg`.
    #[must_use]
    pub fn new(cfg: &Qwen4ExpConfig, hash: &NGramHash) -> Self {
        let mut gdr = BTreeMap::new();
        let mut conv = BTreeMap::new();
        let mut kv = BTreeMap::new();
        let mut ple_conv = BTreeMap::new();
        let conv_dim = cfg.linear_conv_dim();
        for (l, kind) in cfg.layer_types.iter().enumerate() {
            match kind {
                Qwen4LayerType::LinearAttention => {
                    gdr.insert(
                        l,
                        vec![
                            0.0;
                            cfg.linear_num_value_heads
                                * cfg.linear_key_head_dim
                                * cfg.linear_value_head_dim
                        ],
                    );
                    conv.insert(l, vec![0.0; conv_dim * (cfg.linear_conv_kernel_dim - 1)]);
                }
                Qwen4LayerType::FullAttention => {
                    kv.insert(l, HostKv::default());
                }
            }
        }
        for &l in &cfg.ple_layer_ids {
            ple_conv.insert(l, PleConvState::zeros(&ple_config(cfg)));
        }
        Self {
            seq_len: 0,
            gdr,
            conv,
            kv,
            ple_conv,
            ngram: NGramContext::new(hash),
        }
    }

    /// Zero everything for a fresh generation.
    pub fn reset(&mut self, cfg: &Qwen4ExpConfig, hash: &NGramHash) {
        *self = Self::new(cfg, hash);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Host transcription: linear attention.
// ─────────────────────────────────────────────────────────────────────────────

/// Intermediates of one linear-attention forward, for the parity harness.
pub struct LinearTaps {
    /// Raw `in_proj_qkv` output `[conv_dim]`.
    pub qkv_raw: Vec<f32>,
    /// Post-conv (+SiLU) activations `[conv_dim]`.
    pub qkv_conv: Vec<f32>,
    /// `in_proj_z` output `[nv*vd]`.
    pub z: Vec<f32>,
    /// Recurrence output `[nv*vd]` (pre norm/gate), HF head order.
    pub core: Vec<f32>,
    /// Gated-norm output `[nv*vd]`.
    pub gated: Vec<f32>,
}

/// One token of `Qwen4ExpTextGatedDeltaNet` (the `seq_len == 1` recurrent
/// rule), advancing `gdr_state` and `conv_ring` in place.
pub fn host_linear_attention(
    cfg: &Qwen4ExpConfig,
    w: &HostLinearAttn<'_>,
    x: &[f32],
    gdr_state: &mut [f32],
    conv_ring: &mut [f32],
) -> (Vec<f32>, LinearTaps) {
    let kd = cfg.linear_key_head_dim;
    let vd = cfg.linear_value_head_dim;
    let nk = cfg.linear_num_key_heads;
    let nv = cfg.linear_num_value_heads;
    let kernel = cfg.linear_conv_kernel_dim;
    let conv_dim = 2 * nk * kd + nv * vd;
    let group = nv / nk;
    assert_eq!(gdr_state.len(), nv * kd * vd, "gdr state length");
    assert_eq!(conv_ring.len(), conv_dim * (kernel - 1), "conv ring length");

    let qkv_raw = w.qkv.matvec(x);
    let z = w.z.matvec(x);
    let a = w.a.matvec(x);
    let b = w.b.matvec(x);

    // Depthwise causal conv over `[ring | qkv_raw]`, bf16-rounded sum, SiLU.
    let state_w = kernel - 1;
    let mut qkv_conv = vec![0.0f32; conv_dim];
    for (c, out) in qkv_conv.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        for k in 0..kernel {
            let v = if k < state_w {
                conv_ring[k * conv_dim + c]
            } else {
                qkv_raw[c]
            };
            sum += v * w.conv[c * kernel + k];
        }
        *out = silu32(round_to_bf16(sum));
    }
    // Roll the ring: drop the oldest row, append this token's RAW qkv.
    conv_ring.copy_within(conv_dim.., 0);
    conv_ring[(state_w - 1) * conv_dim..].copy_from_slice(&qkv_raw);

    // Recurrent gated-delta rule, f32 in the kernel's serial order.
    let q_all = &qkv_conv[..nk * kd];
    let k_all = &qkv_conv[nk * kd..2 * nk * kd];
    let v_all = &qkv_conv[2 * nk * kd..];
    let mut core = vec![0.0f32; nv * vd];
    for vh in 0..nv {
        let kh = vh / group; // HF repeat_interleave: block-broadcast.
        let q = &q_all[kh * kd..(kh + 1) * kd];
        let k = &k_all[kh * kd..(kh + 1) * kd];
        let vvec = &v_all[vh * vd..(vh + 1) * vd];
        // l2norm eps 1e-6 (the reference's FLA-style epsilon).
        let q_sumsq: f32 = q.iter().map(|&v| v * v).sum();
        let k_sumsq: f32 = k.iter().map(|&v| v * v).sum();
        let q_scale = (q_sumsq + 1e-6).sqrt().recip() / (kd as f32).sqrt();
        let k_scale = (k_sumsq + 1e-6).sqrt().recip();
        let exp_g = (-(w.a_log[vh].exp()) * softplus32(a[vh] + w.dt_bias[vh])).exp();
        let beta = sigmoid32(b[vh]);
        let state = &mut gdr_state[vh * kd * vd..(vh + 1) * kd * vd];
        for val in 0..vd {
            let mut kv_mem = 0.0f32;
            for (j, &kj) in k.iter().enumerate() {
                let idx = j * vd + val;
                let decayed = state[idx] * exp_g;
                state[idx] = decayed;
                kv_mem += decayed * (kj * k_scale);
            }
            let delta = (vvec[val] - kv_mem) * beta;
            let mut acc = 0.0f32;
            for (j, (&kj, &qj)) in k.iter().zip(q).enumerate() {
                let idx = j * vd + val;
                let updated = state[idx] + delta * (kj * k_scale);
                state[idx] = updated;
                acc += updated * (qj * q_scale);
            }
            core[vh * vd + val] = acc;
        }
    }

    // Gated RMSNorm per value head: `rms(core) * norm_w * act(z)`, PLAIN gain,
    // gate activation from `output_gate_type` (sigmoid on this checkpoint).
    let mut gated = vec![0.0f32; nv * vd];
    for vh in 0..nv {
        let head = &core[vh * vd..(vh + 1) * vd];
        let mean: f64 = head
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum::<f64>()
            / vd as f64;
        let scale = (mean + f64::from(cfg.rms_norm_eps)).sqrt().recip();
        for d in 0..vd {
            let zv = z[vh * vd + d];
            let gate = match cfg.output_gate {
                GateActivation::Sigmoid => sigmoid32(zv),
                GateActivation::Silu => silu32(zv),
            };
            gated[vh * vd + d] = ((f64::from(head[d]) * scale) as f32) * w.norm[d] * gate;
        }
    }

    let y = w.out.matvec(&gated);
    (
        y,
        LinearTaps {
            qkv_raw,
            qkv_conv,
            z,
            core,
            gated,
        },
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Host transcription: full attention (dense — the QSA stub).
// ─────────────────────────────────────────────────────────────────────────────

/// Intermediates of one full-attention forward.
pub struct FullTaps {
    /// `q_proj` output `[nq * 2 * hd]` ([query|gate] per head).
    pub q_full: Vec<f32>,
    /// Normed + roped queries, packed `[nq * hd]`.
    pub q_roped: Vec<f32>,
    /// Normed + roped keys `[nkv * hd]` (this position's, pre-f16).
    pub k_roped: Vec<f32>,
    /// Raw values `[nkv * hd]`.
    pub v_raw: Vec<f32>,
    /// Gated flash output `[nq * hd]`.
    pub gated: Vec<f32>,
}

/// Rotate the leading `rotary_dim` lanes of one `head_dim`-wide head in place:
/// pairs `(d, d + rotary_dim/2)`, angle `pos · theta^(-2d/rotary_dim)`.
fn rope_partial(head: &mut [f32], rotary_dim: usize, pos: usize, theta: f32) {
    let half = rotary_dim / 2;
    for d in 0..half {
        let freq = f64::from(theta).powf(-2.0 * d as f64 / rotary_dim as f64);
        let angle = pos as f64 * freq;
        let (sin, cos) = angle.sin_cos();
        let a = f64::from(head[d]);
        let b = f64::from(head[d + half]);
        head[d] = (a * cos - b * sin) as f32;
        head[d + half] = (a * sin + b * cos) as f32;
    }
}

/// `x * inv_rms * (1 + w)` over one head (the `Qwen4ExpTextRMSNorm` form).
fn head_rms_norm_bias(head: &mut [f32], w: &[f32], eps: f32) {
    let mean: f64 = head
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum::<f64>()
        / head.len() as f64;
    let scale = (mean + f64::from(eps)).sqrt().recip();
    for (v, &g) in head.iter_mut().zip(w) {
        *v = (f64::from(*v) * scale * (1.0 + f64::from(g))) as f32;
    }
}

/// One token of `Qwen4ExpTextAttention` with the indexer stubbed dense (exact
/// for `pos + 1 <= 2051` visible tokens — enforced by the caller's
/// `max_context`). Appends this position's K/V to `kv` (f16-rounded).
pub fn host_full_attention(
    cfg: &Qwen4ExpConfig,
    w: &HostFullAttn<'_>,
    x: &[f32],
    pos: usize,
    kv: &mut HostKv,
) -> (Vec<f32>, FullTaps) {
    let hd = cfg.head_dim;
    let nq = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;
    let group = nq / nkv;
    let kv_dim = nkv * hd;
    assert_eq!(kv.k.len(), pos * kv_dim, "KV cache length vs position");

    let q_full = w.q.matvec(x);
    let mut k_new = w.k.matvec(x);
    let v_new = w.v.matvec(x);

    // Per-head q/k RMSNorm (1 + w), then partial RoPE.
    let mut q_roped = vec![0.0f32; nq * hd];
    for h in 0..nq {
        q_roped[h * hd..(h + 1) * hd].copy_from_slice(&q_full[h * 2 * hd..h * 2 * hd + hd]);
        let head = &mut q_roped[h * hd..(h + 1) * hd];
        head_rms_norm_bias(head, &w.q_norm, cfg.rms_norm_eps);
        rope_partial(head, cfg.rotary_dim, pos, cfg.rope_theta);
    }
    for h in 0..nkv {
        let head = &mut k_new[h * hd..(h + 1) * hd];
        head_rms_norm_bias(head, &w.k_norm, cfg.rms_norm_eps);
        rope_partial(head, cfg.rotary_dim, pos, cfg.rope_theta);
    }
    let k_roped = k_new.clone();

    // Cache through the f16 contract (see [`HostKv`]).
    kv.k.extend(k_new.iter().map(|&v| round_to_f16(v)));
    kv.v.extend(v_new.iter().map(|&v| round_to_f16(v)));

    // Causal SDPA over the cache, scale 1/sqrt(head_dim), f64 softmax.
    let scale = 1.0 / (hd as f64).sqrt();
    let kv_len = pos + 1;
    let mut attn = vec![0.0f32; nq * hd];
    for h in 0..nq {
        let kvh = h / group;
        let q = &q_roped[h * hd..(h + 1) * hd];
        let mut scores = Vec::with_capacity(kv_len);
        let mut max = f64::NEG_INFINITY;
        for t in 0..kv_len {
            let krow = &kv.k[t * kv_dim + kvh * hd..t * kv_dim + (kvh + 1) * hd];
            let dot: f64 = q
                .iter()
                .zip(krow)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum();
            let s = dot * scale;
            max = max.max(s);
            scores.push(s);
        }
        let mut denom = 0.0f64;
        for s in &mut scores {
            *s = (*s - max).exp();
            denom += *s;
        }
        let out = &mut attn[h * hd..(h + 1) * hd];
        let mut acc = vec![0.0f64; hd];
        for (t, &p) in scores.iter().enumerate() {
            let vrow = &kv.v[t * kv_dim + kvh * hd..t * kv_dim + (kvh + 1) * hd];
            for (a, &v) in acc.iter_mut().zip(vrow) {
                *a += p * f64::from(v);
            }
        }
        for (o, a) in out.iter_mut().zip(acc) {
            *o = (a / denom) as f32;
        }
    }

    // Per-element sigmoid gate from the interleaved q projection.
    let mut gated = vec![0.0f32; nq * hd];
    for h in 0..nq {
        for d in 0..hd {
            let gate = q_full[h * 2 * hd + hd + d];
            gated[h * hd + d] = attn[h * hd + d] * sigmoid32(gate);
        }
    }

    let y = w.o.matvec(&gated);
    (
        y,
        FullTaps {
            q_full,
            q_roped,
            k_roped,
            v_raw: v_new,
            gated,
        },
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Host transcription: MoE (router + NVFP4 experts + shared expert).
// ─────────────────────────────────────────────────────────────────────────────

/// Intermediates of one MoE forward.
pub struct MoeTaps {
    /// Router logits `[num_experts]`.
    pub logits: Vec<f32>,
    /// Selected expert ids, prob-descending (ties → lower id).
    pub ids: Vec<i32>,
    /// Selected (renormalised) routing weights.
    pub weights: Vec<f32>,
    /// Routed-expert accumulator `[hidden]` (before the shared expert).
    pub routed: Vec<f32>,
    /// Shared-expert contribution `[hidden]` (already gate-scaled).
    pub shared: Vec<f32>,
}

/// Softmax → top-k (by prob, ties to the lower id) → optional renorm,
/// mirroring `Qwen4ExpTextTopKRouter` and the `qwen36_router_topk.comp` clamp.
#[must_use]
pub fn host_router_topk(logits: &[f32], top_k: usize, norm_topk: bool) -> (Vec<i32>, Vec<f32>) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let denom: f64 = logits.iter().map(|&l| f64::from(l - max).exp()).sum();
    let mut order: Vec<usize> = (0..logits.len()).collect();
    // Stable descending by logit == descending by prob (softmax is monotonic);
    // stability keeps the lower id on ties, matching the shader's strict `>`.
    order.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let kept = &order[..top_k.min(logits.len())];
    let probs: Vec<f64> = kept
        .iter()
        .map(|&e| f64::from(logits[e] - max).exp() / denom)
        .collect();
    let norm = if norm_topk {
        let sum: f64 = probs.iter().sum();
        sum.max(6.103_516e-5f64) // F16_MIN clamp, as the shader.
    } else {
        1.0
    };
    (
        kept.iter().map(|&e| e as i32).collect(),
        probs.iter().map(|&p| (p / norm) as f32).collect(),
    )
}

/// `weight_scale_2 · (dequant(W_e) @ x)` for one NVFP4 expert projection,
/// straight off the checkpoint planes (repack → ggml-block dequant → f64 dot).
pub fn host_expert_matvec(
    st: &SafeTensorsDir,
    layer: usize,
    expert: u32,
    proj: ExpertProj,
    x: &[f32],
) -> Result<Vec<f32>> {
    let name = |part| expert_tensor_name(layer, expert, proj, part);
    let qs_name = name(Nvfp4Part::Packed);
    let info = st
        .tensor(&qs_name)
        .ok_or_else(|| anyhow!("missing expert plane `{qs_name}`"))?;
    // U8 plane dims (ne order): [ncols/2 bytes, nrows].
    let ncols = info.dims[0] as usize * 2;
    let nrows = info.dims[1] as usize;
    ensure!(
        x.len() == ncols,
        "expert `{qs_name}`: x is {} wide, weight wants {ncols}",
        x.len()
    );
    let qs = st.tensor_data(&qs_name)?;
    let scales = st.tensor_data(&name(Nvfp4Part::BlockScale))?;
    let ws2_bytes = st.tensor_data(&name(Nvfp4Part::GlobalScale))?;
    ensure!(ws2_bytes.len() == 4, "weight_scale_2 must be one f32");
    let ws2 = f32::from_le_bytes([ws2_bytes[0], ws2_bytes[1], ws2_bytes[2], ws2_bytes[3]]);

    let row_bytes = ncols / 64 * 36;
    let mut packed = vec![0u8; nrows * row_bytes];
    repack_nvfp4_planes(qs, scales, nrows, ncols, &mut packed)
        .map_err(|e| anyhow!("repack `{qs_name}`: {e}"))?;
    let mut y = vec![0.0f32; nrows];
    for (r, dst) in y.iter_mut().enumerate() {
        let row = dequantize_row_nvfp4(&packed[r * row_bytes..(r + 1) * row_bytes], ncols)
            .map_err(|e| anyhow!("dequant `{qs_name}` row {r}: {e}"))?;
        let acc: f64 = row
            .iter()
            .zip(x)
            .map(|(&w, &v)| f64::from(w) * f64::from(v))
            .sum();
        *dst = (acc * f64::from(ws2)) as f32;
    }
    Ok(y)
}

/// One token of `Qwen4ExpTextSparseMoeBlock` on the host: router softmax
/// top-k (renormalised), NVFP4 routed experts, sigmoid-gated shared expert.
pub fn host_moe(
    cfg: &Qwen4ExpConfig,
    st: &SafeTensorsDir,
    layer: usize,
    w: &HostMoe<'_>,
    x: &[f32],
) -> Result<(Vec<f32>, MoeTaps)> {
    let h = cfg.hidden_size;
    let logits = w.router.matvec(x);
    let (ids, weights) = host_router_topk(&logits, cfg.num_experts_per_tok, cfg.norm_topk_prob);

    let mut routed = vec![0.0f32; h];
    for (&e, &wt) in ids.iter().zip(&weights) {
        let e = u32::try_from(e).map_err(|_| anyhow!("negative expert id"))?;
        let gate = host_expert_matvec(st, layer, e, ExpertProj::Gate, x)?;
        let up = host_expert_matvec(st, layer, e, ExpertProj::Up, x)?;
        let act: Vec<f32> = gate.iter().zip(&up).map(|(&g, &u)| silu32(g) * u).collect();
        let down = host_expert_matvec(st, layer, e, ExpertProj::Down, &act)?;
        for (r, &d) in routed.iter_mut().zip(&down) {
            *r += wt * d;
        }
    }

    let sh_gate_scalar = sigmoid32(w.shexp_gate.matvec(x)[0]);
    let g = w.sh_gate.matvec(x);
    let u = w.sh_up.matvec(x);
    let act: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| silu32(g) * u).collect();
    let mut shared = w.sh_down.matvec(&act);
    for s in &mut shared {
        *s *= sh_gate_scalar;
    }

    let y: Vec<f32> = routed.iter().zip(&shared).map(|(&r, &s)| r + s).collect();
    Ok((
        y,
        MoeTaps {
            logits,
            ids,
            weights,
            routed,
            shared,
        },
    ))
}

// ─────────────────────────────────────────────────────────────────────────────
// Device runner: one host-cached scratch arena + per-dispatch ranged sets.
//
// This is BRING-UP machinery, not the perf path: stages record a handful of
// dispatches, submit synchronously and read back through the HOST_CACHED
// arena (`alloc_uma`/WC reads are the documented ~0.10 GB/s trap). The
// whole-token single-submit recorder of `forward.rs` is a later step; what
// this buys now is that every stage is independently runnable and therefore
// independently *measurable* against the host oracle.
// ─────────────────────────────────────────────────────────────────────────────

/// Byte offsets (256-aligned) of the named scratch regions in the arena.
#[derive(Debug, Clone, Copy)]
pub struct DevSlots {
    h: u64,
    hn: u64,
    u_raw: u64,
    x: u64,
    y: u64,
    qkv: u64,
    kbuf: u64,
    vbuf: u64,
    q: u64,
    attn: u64,
    zbuf: u64,
    abuf: u64,
    bbuf: u64,
    convo: u64,
    gdro: u64,
    gdr_state: u64,
    conv_ring: u64,
    logits: u64,
    ids: u64,
    wts: u64,
    scale0: u64,
    gate: u64,
    up: u64,
    down: u64,
    acc: u64,
    shg: u64,
    ple_emb: u64,
    ple_k: u64,
    ple_v: u64,
    ple_g: u64,
    ple_gn: u64,
    ple_o: u64,
    ple_ring: u64,
    pos: u64,
    dummy: u64,
    total: u64,
}

impl DevSlots {
    fn layout() -> Self {
        let mut off = 0u64;
        let mut take = |elems: u64| {
            let at = off;
            off += (elems * 4).div_ceil(256) * 256;
            at
        };
        Self {
            h: take(10240),
            hn: take(10240),
            u_raw: take(320),
            x: take(2560),
            y: take(2560),
            qkv: take(12288),
            kbuf: take(512),
            vbuf: take(512),
            q: take(6144),
            attn: take(6144),
            zbuf: take(6144),
            abuf: take(64),
            bbuf: take(64),
            convo: take(10240),
            gdro: take(6144),
            gdr_state: take(48 * 128 * 128),
            conv_ring: take(3 * 10240),
            logits: take(512),
            ids: take(16),
            wts: take(16),
            scale0: take(16),
            gate: take(6400),
            up: take(6400),
            down: take(25600),
            acc: take(2560),
            shg: take(8),
            ple_emb: take(2560),
            ple_k: take(10240),
            ple_v: take(2560),
            ple_g: take(10240),
            ple_gn: take(10240),
            ple_o: take(10240),
            ple_ring: take(9 * 10240),
            pos: take(4),
            dummy: take(8),
            total: off,
        }
    }
}

/// Where a dispatch binding's bytes live.
enum Bind<'b> {
    /// The scratch arena at `(offset, len)`.
    A(u64, u64),
    /// The f16 KV cache at `(offset, len)`.
    Kv(u64, u64),
    /// An external buffer (weight slab, permuted upload) at `(offset, len)`.
    Ext(&'b DeviceBuffer<'b>, u64, u64),
}

/// Device f16 KV cache for the RESIDENT full-attention layers, laid out like
/// `forward.rs`'s `DeviceKvCache`: `[K block | V block]`, each block indexed
/// `[full_idx, kv_head, pos, head_dim]` f16.
struct DevKv<'ctx> {
    buffer: DeviceBuffer<'ctx>,
    v_base: u64,
    plane_bytes: u64,
    layer_bytes: u64,
    head_dim: usize,
}

impl<'ctx> DevKv<'ctx> {
    fn new(
        ctx: &'ctx VulkanContext,
        n_full: usize,
        n_kv: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> Result<Self> {
        let plane_bytes = (max_seq * head_dim * 2) as u64;
        let layer_bytes = plane_bytes * n_kv as u64;
        let block = layer_bytes * n_full.max(1) as u64;
        let buffer = DeviceBuffer::alloc_uma(ctx, usize::try_from(block * 2)?.max(4))
            .map_err(|e| anyhow!("alloc qwen4 device KV cache: {e}"))?;
        Ok(Self {
            buffer,
            v_base: block,
            plane_bytes,
            layer_bytes,
            head_dim,
        })
    }

    fn k_plane(&self, full_idx: usize, kvh: usize) -> u64 {
        full_idx as u64 * self.layer_bytes + kvh as u64 * self.plane_bytes
    }

    fn v_plane(&self, full_idx: usize, kvh: usize) -> u64 {
        self.v_base + self.k_plane(full_idx, kvh)
    }

    fn row(&self, plane: u64, pos: usize) -> u64 {
        plane + (pos * self.head_dim * 2) as u64
    }
}

/// The synchronous per-stage device executor.
pub struct Qwen4Dev<'ctx> {
    ctx: &'ctx VulkanContext,
    cache: KernelCache<'ctx>,
    recorder: CommandRecorder<'ctx>,
    arena: DeviceBuffer<'ctx>,
    slots: DevSlots,
    /// Descriptor sets recorded since the last flush; they must outlive the
    /// submit that references them.
    live: Vec<DescriptorSet<'ctx>>,
    kv: Option<DevKv<'ctx>>,
    /// `layer id -> full_idx` for the KV cache, fixed at construction.
    full_idx: BTreeMap<usize, usize>,
    open: bool,
}

impl<'ctx> Qwen4Dev<'ctx> {
    /// Build the runner. `full_layers` are the full-attention layer ids whose
    /// f16 KV planes should exist (pass the ones that will run on device);
    /// `max_seq` caps the cache (use `cfg.max_context`).
    pub fn new(
        ctx: &'ctx VulkanContext,
        cfg: &Qwen4ExpConfig,
        full_layers: &[usize],
        max_seq: usize,
    ) -> Result<Self> {
        let slots = DevSlots::layout();
        let arena = DeviceBuffer::alloc_host_cached(ctx, usize::try_from(slots.total)?)
            .map_err(|e| anyhow!("alloc qwen4 scratch arena ({} B): {e}", slots.total))?;
        let kv = if full_layers.is_empty() {
            None
        } else {
            Some(DevKv::new(
                ctx,
                full_layers.len(),
                cfg.num_key_value_heads,
                cfg.head_dim,
                max_seq,
            )?)
        };
        let full_idx = full_layers
            .iter()
            .enumerate()
            .map(|(i, &l)| (l, i))
            .collect();
        Ok(Self {
            ctx,
            cache: KernelCache::new(),
            recorder: CommandRecorder::new(ctx).map_err(|e| anyhow!("create recorder: {e}"))?,
            arena,
            slots,
            live: Vec::new(),
            kv,
            full_idx,
            open: false,
        })
    }

    fn write_f32(&mut self, off: u64, data: &[f32]) -> Result<()> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.arena
            .copy_from_host_at(off, &bytes)
            .map_err(|e| anyhow!("arena write at {off}: {e}"))
    }

    fn read_f32(&self, off: u64, n: usize) -> Result<Vec<f32>> {
        let mut bytes = vec![0u8; n * 4];
        self.arena
            .copy_to_host_at(off, &mut bytes)
            .map_err(|e| anyhow!("arena read at {off}: {e}"))?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    fn read_i32(&self, off: u64, n: usize) -> Result<Vec<i32>> {
        let mut bytes = vec![0u8; n * 4];
        self.arena
            .copy_to_host_at(off, &mut bytes)
            .map_err(|e| anyhow!("arena read at {off}: {e}"))?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Record one dispatch into the open command buffer (opening it if needed).
    fn rec(
        &mut self,
        kernel: Kernel,
        spec: &[(u32, u32)],
        push: &[u8],
        binds: &[Bind<'_>],
        groups: [u32; 3],
    ) -> Result<()> {
        if !self.open {
            self.recorder
                .begin()
                .map_err(|e| anyhow!("recorder begin: {e}"))?;
            self.open = true;
        }
        let resolved: Vec<(&DeviceBuffer<'_>, u64, u64)> = binds
            .iter()
            .map(|b| match *b {
                Bind::A(off, len) => (&self.arena, off, len),
                Bind::Kv(off, len) => {
                    let kv = self.kv.as_ref().expect("KV bind without a KV cache");
                    (&kv.buffer, off, len)
                }
                Bind::Ext(buf, off, len) => (buf, off, len),
            })
            .collect();
        let (pipeline, layout) = self
            .cache
            .get(self.ctx, kernel, spec, push.len() as u32, binds.len())
            .map_err(|e| anyhow!("build {kernel:?} pipeline: {e}"))?;
        let set = DescriptorSet::storage_buffers_ranged(self.ctx, layout, &resolved)
            .map_err(|e| anyhow!("bind {kernel:?} set: {e}"))?;
        record_dispatch(&mut self.recorder, pipeline, &set, push, groups);
        self.live.push(set);
        Ok(())
    }

    fn barrier(&mut self) {
        if self.open {
            self.recorder.barrier();
        }
    }

    /// Submit everything recorded since the last flush and wait.
    fn flush(&mut self) -> Result<()> {
        if self.open {
            self.recorder
                .submit_and_wait()
                .map_err(|e| anyhow!("submit: {e}"))?;
            self.open = false;
        }
        self.live.clear();
        Ok(())
    }

    // ── hyper-connection site ────────────────────────────────────────────

    /// The pre half of a gated-residual site: grouped norm → down GEMV →
    /// `Qwen4HcMix`. Uploads `h`, returns the block input `x`, and leaves `h`
    /// (raw) + `hn` (normed) in their slots for [`Self::hc_combine`].
    ///
    /// Pass `(None, HcSite::Mixer)` for the stream mixer — there is no combine
    /// half then, and the returned `x` feeds `lm_head` directly.
    pub fn hc_pre(
        &mut self,
        weights: &Qwen4Weights<'_, '_>,
        hc: &HyperConnectionConfig,
        layer: Option<usize>,
        site: HcSite,
        h: &[f32],
    ) -> Result<Vec<f32>> {
        let b = weights.hyper_connection(layer, site)?;
        let hh = hc.hc_hidden() as u64;
        let (norm_buf, norm_off, norm_len) = weights.binding(b.hc_norm)?;
        let (down_buf, down_off, down_len) = weights.binding(b.mix_down)?;
        let (up_buf, up_off, up_len) = weights.binding(b.mix_up)?;
        self.write_f32(self.slots.h, h)?;
        let s = self.slots;
        let push =
            rms_norm_params_grouped(hc.hidden_size as u32, hc.hc_count as u32, hc.rms_norm_eps)
                .to_le_bytes();
        let d = rms_norm_dispatch_rows(hc.hc_count as u32);
        self.rec(
            Kernel::RmsNorm,
            Kernel::RmsNorm.specialization_u32(),
            &push,
            &[
                Bind::A(s.h, hh * 4),
                Bind::Ext(norm_buf, norm_off, norm_len),
                Bind::A(s.hn, hh * 4),
            ],
            [d.x, d.y, d.z],
        )?;
        self.barrier();
        let push = qwen36_router_gemv_params(hc.hc_lowrank as u32, hh as u32, false).to_le_bytes();
        let d = qwen36_router_gemv_dispatch(hc.hc_lowrank as u32);
        self.rec(
            Kernel::Qwen36RouterGemv,
            Kernel::Qwen36RouterGemv.specialization_u32(),
            &push,
            &[
                Bind::A(s.hn, hh * 4),
                Bind::Ext(down_buf, down_off, down_len),
                Bind::A(s.u_raw, hc.hc_lowrank as u64 * 4),
            ],
            [d.x, d.y, d.z],
        )?;
        self.barrier();
        let push = qwen4_hc_mix_params(
            hc.hidden_size as u32,
            hc.hc_count as u32,
            hc.hc_lowrank as u32,
        )
        .to_le_bytes();
        let d = qwen4_hc_mix_dispatch(hc.hidden_size as u32);
        self.rec(
            Kernel::Qwen4HcMix,
            Kernel::Qwen4HcMix.specialization_u32(),
            &push,
            &[
                Bind::A(s.hn, hh * 4),
                Bind::Ext(up_buf, up_off, up_len),
                Bind::A(s.u_raw, hc.hc_lowrank as u64 * 4),
                Bind::A(s.x, hc.hidden_size as u64 * 4),
            ],
            [d.x, d.y, d.z],
        )?;
        self.flush()?;
        self.read_f32(self.slots.x, hc.hidden_size)
    }

    /// The combine half: inject `y` into the residual left by [`Self::hc_pre`]
    /// (which must have run for the SAME site, with no other stage touching
    /// the `h`/`hn` slots in between) and return the updated residual.
    pub fn hc_combine(
        &mut self,
        weights: &Qwen4Weights<'_, '_>,
        hc: &HyperConnectionConfig,
        layer: Option<usize>,
        site: HcSite,
        y: &[f32],
    ) -> Result<Vec<f32>> {
        let b = weights.hyper_connection(layer, site)?;
        let inject = b
            .block_inject
            .ok_or_else(|| anyhow!("hc_combine on the mixer site (no block_inject)"))?;
        let (inj_buf, inj_off, inj_len) = weights.binding(inject)?;
        let hh = hc.hc_hidden() as u64;
        self.write_f32(self.slots.y, y)?;
        let s = self.slots;
        let push = qwen4_hc_combine_params(hc.hidden_size as u32, hc.hc_count as u32).to_le_bytes();
        let d = qwen4_hc_combine_dispatch(hc.hidden_size as u32);
        self.rec(
            Kernel::Qwen4HcCombine,
            Kernel::Qwen4HcCombine.specialization_u32(),
            &push,
            &[
                Bind::A(s.hn, hh * 4),
                Bind::Ext(inj_buf, inj_off, inj_len),
                Bind::A(s.h, hh * 4),
                Bind::A(s.y, hc.hidden_size as u64 * 4),
            ],
            [d.x, d.y, d.z],
        )?;
        self.flush()?;
        self.read_f32(self.slots.h, hc.hc_hidden())
    }

    // ── MoE ──────────────────────────────────────────────────────────────

    /// Device MoE for one token. Router GEMV + top-k on device, ids read back
    /// once for the slot-ordered `weight_scale_2` gather, then the three NVFP4
    /// fused expert GEMVs (PLAIN f32 activations — no q8_1 quantize on this
    /// path) and the weighted accumulate. The shared expert rides on device
    /// when its dense tier is F32-resident; `taps.shared_on_device` says so.
    pub fn moe(
        &mut self,
        weights: &Qwen4Weights<'_, '_>,
        cfg: &Qwen4ExpConfig,
        layer: usize,
        x: &[f32],
    ) -> Result<(Vec<f32>, DevMoeTaps)> {
        let h = cfg.hidden_size;
        let top_k = cfg.num_experts_per_tok;
        let inter = cfg.moe_intermediate_size;
        let s = self.slots;
        self.write_f32(s.x, x)?;

        // Router logits + top-k, one submit.
        let router = *weights.tensor(&layer_tensor_name(layer, "mlp.gate.weight"))?;
        let (rb, ro, rl) = weights.binding(&router)?;
        let push = qwen36_router_gemv_params(cfg.num_experts as u32, h as u32, false).to_le_bytes();
        let d = qwen36_router_gemv_dispatch(cfg.num_experts as u32);
        self.rec(
            Kernel::Qwen36RouterGemv,
            Kernel::Qwen36RouterGemv.specialization_u32(),
            &push,
            &[
                Bind::A(s.x, (h * 4) as u64),
                Bind::Ext(rb, ro, rl),
                Bind::A(s.logits, (cfg.num_experts * 4) as u64),
            ],
            [d.x, d.y, d.z],
        )?;
        self.barrier();
        let push =
            qwen36_router_topk_params(cfg.num_experts as u32, top_k as u32, cfg.norm_topk_prob)
                .to_le_bytes();
        let d = qwen36_router_topk_dispatch();
        self.rec(
            Kernel::Qwen36RouterTopk,
            Kernel::Qwen36RouterTopk.specialization_u32(),
            &push,
            &[
                Bind::A(s.logits, (cfg.num_experts * 4) as u64),
                Bind::A(s.ids, (top_k * 4) as u64),
                Bind::A(s.wts, (top_k * 4) as u64),
            ],
            [d.x, d.y, d.z],
        )?;
        self.flush()?;
        let logits = self.read_f32(s.logits, cfg.num_experts)?;
        let ids = self.read_i32(s.ids, top_k)?;
        let route_weights = self.read_f32(s.wts, top_k)?;

        // Fused expert GEMVs. `weight_scale_2` rides SCALE0, indexed by SLOT.
        // ne11 = 1 shares the one `x` row across slots for gate/up.
        let mut scale0 = Vec::new();
        for (proj, dst_off) in [(ExpertProj::Gate, s.gate), (ExpertProj::Up, s.up)] {
            self.gemv_id_nvfp4(
                weights,
                layer,
                proj,
                &ids,
                h,
                inter,
                s.x,
                1,
                dst_off,
                &mut scale0,
            )?;
            // scale0 lives in one slot and differs per stack; the host write
            // below must not race the recorded read, so flush per projection.
            self.flush()?;
        }
        // act = silu(gate) * up over [top_k * inter].
        let n_act = (top_k * inter) as u32;
        let push = swiglu_params(n_act).to_le_bytes();
        let d = swiglu_dispatch(n_act);
        self.rec(
            Kernel::SwiGlu,
            Kernel::SwiGlu.specialization_u32(),
            &push,
            &[
                Bind::A(s.gate, u64::from(n_act) * 4),
                Bind::A(s.up, u64::from(n_act) * 4),
                Bind::A(s.gate, u64::from(n_act) * 4),
            ],
            [d.x, d.y, d.z],
        )?;
        self.flush()?;
        // down: each expert slot reads ITS OWN activation row (ne11 = top_k).
        self.gemv_id_nvfp4(
            weights,
            layer,
            ExpertProj::Down,
            &ids,
            inter,
            h,
            s.gate,
            top_k,
            s.down,
            &mut scale0,
        )?;
        self.barrier();
        // acc = Σ_e w_e · down[e].
        let push = qwen36_moe_weighted_accum_params(h as u32, top_k as u32, true).to_le_bytes();
        let d = qwen36_moe_weighted_accum_dispatch(h as u32);
        self.rec(
            Kernel::Qwen36MoeWeightedAccum,
            Kernel::Qwen36MoeWeightedAccum.specialization_u32(),
            &push,
            &[
                Bind::A(s.down, (top_k * h * 4) as u64),
                Bind::A(s.wts, (top_k * 4) as u64),
                Bind::A(s.acc, (h * 4) as u64),
            ],
            [d.x, d.y, d.z],
        )?;
        self.flush()?;
        let routed = self.read_f32(s.acc, h)?;

        // Shared expert on device iff its dense tier is F32-resident.
        let shared_on_device = self
            .try_shared_expert(weights, cfg, layer)
            .context("device shared expert")?;
        let y = if shared_on_device {
            self.read_f32(s.acc, h)?
        } else {
            routed.clone()
        };
        Ok((
            y,
            DevMoeTaps {
                logits,
                ids,
                weights: route_weights,
                routed,
                shared_on_device,
            },
        ))
    }

    /// One fused NVFP4 expert GEMV over the selected `ids`, with the stack's
    /// slot-ordered `weight_scale_2` on SCALE0 and `ne11` activation rows at
    /// `b_off` (1 = shared across slots, `top_k` = one row per slot).
    #[expect(clippy::too_many_arguments, reason = "a dispatch is this wide")]
    fn gemv_id_nvfp4(
        &mut self,
        weights: &Qwen4Weights<'_, '_>,
        layer: usize,
        proj: ExpertProj,
        ids: &[i32],
        ncols: usize,
        nrows: usize,
        b_off: u64,
        ne11: usize,
        dst_off: u64,
        scale0: &mut Vec<f32>,
    ) -> Result<()> {
        let top_k = ids.len();
        let s = self.slots;
        let stack = weights.expert_stack(layer, proj)?;
        stack.scale0_for_route(ids, scale0)?;
        let scale0_now = scale0.clone();
        self.write_f32(s.scale0, &scale0_now)?;
        let (sb, so, sl) = weights.binding(&stack.tensor)?;
        let mut words = gemv_id_params_fused(
            ncols as u32,
            nrows as u32,
            top_k as u32,
            MAT_VEC_FUSION_SCALE0,
        )
        .words()
        .to_vec();
        // Word 9 is ne11: how many activation rows `b_offset = (slot % ne11) *
        // stride_b` can address (`forward.rs::record_gemv_id` documents this).
        words[9] = ne11 as u32;
        let push = KernelParams::from_words(words).to_le_bytes();
        let d = gemv_id_dispatch(nrows as u32, top_k as u32);
        self.rec(
            Kernel::GemvIdNvfp4,
            Kernel::GemvIdNvfp4.specialization_u32(),
            &push,
            &[
                Bind::Ext(sb, so, sl),
                Bind::A(b_off, (ncols * ne11 * 4) as u64),
                Bind::A(dst_off, (top_k * nrows * 4) as u64),
                Bind::A(s.scale0, (top_k * 4) as u64),
                Bind::A(s.dummy, 8),
                Bind::A(s.ids, (top_k * 4) as u64),
            ],
            [d.x, d.y, d.z],
        )
    }

    /// Record + run the shared expert if all four of its dense tensors are
    /// F32-resident, accumulating into the `acc` slot. Returns whether it ran.
    fn try_shared_expert(
        &mut self,
        weights: &Qwen4Weights<'_, '_>,
        cfg: &Qwen4ExpConfig,
        layer: usize,
    ) -> Result<bool> {
        let h = cfg.hidden_size;
        let inter = cfg.shared_expert_intermediate_size;
        let names = [
            layer_tensor_name(layer, "mlp.shared_expert_gate.weight"),
            layer_tensor_name(layer, "mlp.shared_expert.gate_proj.weight"),
            layer_tensor_name(layer, "mlp.shared_expert.up_proj.weight"),
            layer_tensor_name(layer, "mlp.shared_expert.down_proj.weight"),
        ];
        let mut tensors = Vec::new();
        for n in &names {
            match weights.tensor(n) {
                Ok(t) if t.format == Qwen4DeviceFormat::F32 => tensors.push(*t),
                _ => return Ok(false),
            }
        }
        let s = self.slots;

        // shgate = sigmoid(W_sg · x) — one output row.
        let (b0, o0, l0) = weights.binding(&tensors[0])?;
        let push = qwen36_router_gemv_params(1, h as u32, true).to_le_bytes();
        let d = qwen36_router_gemv_dispatch(1);
        self.rec(
            Kernel::Qwen36RouterGemv,
            Kernel::Qwen36RouterGemv.specialization_u32(),
            &push,
            &[
                Bind::A(s.x, (h * 4) as u64),
                Bind::Ext(b0, o0, l0),
                Bind::A(s.shg, 4),
            ],
            [d.x, d.y, d.z],
        )?;
        // gate / up.
        for (t, dst) in [(&tensors[1], s.gate), (&tensors[2], s.up)] {
            let (b, o, l) = weights.binding(t)?;
            let push = qwen36_router_gemv_params(inter as u32, h as u32, false).to_le_bytes();
            let d = qwen36_router_gemv_dispatch(inter as u32);
            self.rec(
                Kernel::Qwen36RouterGemv,
                Kernel::Qwen36RouterGemv.specialization_u32(),
                &push,
                &[
                    Bind::A(s.x, (h * 4) as u64),
                    Bind::Ext(b, o, l),
                    Bind::A(dst, (inter * 4) as u64),
                ],
                [d.x, d.y, d.z],
            )?;
        }
        self.barrier();
        let push = swiglu_params(inter as u32).to_le_bytes();
        let d = swiglu_dispatch(inter as u32);
        self.rec(
            Kernel::SwiGlu,
            Kernel::SwiGlu.specialization_u32(),
            &push,
            &[
                Bind::A(s.gate, (inter * 4) as u64),
                Bind::A(s.up, (inter * 4) as u64),
                Bind::A(s.gate, (inter * 4) as u64),
            ],
            [d.x, d.y, d.z],
        )?;
        self.barrier();
        // down: [inter -> h] into the `down` slot's first row.
        let (b3, o3, l3) = weights.binding(&tensors[3])?;
        let push = qwen36_router_gemv_params(h as u32, inter as u32, false).to_le_bytes();
        let d = qwen36_router_gemv_dispatch(h as u32);
        self.rec(
            Kernel::Qwen36RouterGemv,
            Kernel::Qwen36RouterGemv.specialization_u32(),
            &push,
            &[
                Bind::A(s.gate, (inter * 4) as u64),
                Bind::Ext(b3, o3, l3),
                Bind::A(s.down, (h * 4) as u64),
            ],
            [d.x, d.y, d.z],
        )?;
        self.barrier();
        // acc += shgate · down (count = 1, no init).
        let push = qwen36_moe_weighted_accum_params(h as u32, 1, false).to_le_bytes();
        let d = qwen36_moe_weighted_accum_dispatch(h as u32);
        self.rec(
            Kernel::Qwen36MoeWeightedAccum,
            Kernel::Qwen36MoeWeightedAccum.specialization_u32(),
            &push,
            &[
                Bind::A(s.down, (h * 4) as u64),
                Bind::A(s.shg, 4),
                Bind::A(s.acc, (h * 4) as u64),
            ],
            [d.x, d.y, d.z],
        )?;
        self.flush()?;
        Ok(true)
    }
}

/// What the device MoE hands back for parity.
pub struct DevMoeTaps {
    /// Router logits `[num_experts]`.
    pub logits: Vec<f32>,
    /// Selected expert ids (slot order).
    pub ids: Vec<i32>,
    /// Selected routing weights (renormalised on device).
    pub weights: Vec<f32>,
    /// Routed-expert accumulator (before the shared expert).
    pub routed: Vec<f32>,
    /// Whether the shared expert ran on device (its dense tier was
    /// F32-resident); when `false` the caller adds the host shared expert.
    pub shared_on_device: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// Device stages: PLE, full attention, linear attention.
// ─────────────────────────────────────────────────────────────────────────────

/// What the device PLE hands back for parity.
pub struct DevPleTaps {
    /// `key_proj` output `[hc_hidden]`.
    pub key: Vec<f32>,
    /// `value_proj` output `[hidden]`.
    pub value: Vec<f32>,
    /// The UN-normed gated value (the residual branch).
    pub gated: Vec<f32>,
    /// `norm_conv`'s output (the conv branch's input).
    pub gated_normed: Vec<f32>,
    /// `gated + silu(conv(gated_normed))` — what the decoder adds to `h`.
    pub out: Vec<f32>,
}

/// What the device full attention hands back for parity.
pub struct DevFullTaps {
    /// `q_proj` output `[nq * 2 * hd]`.
    pub q_full: Vec<f32>,
    /// Normed + roped queries `[nq * hd]`.
    pub q_roped: Vec<f32>,
    /// Gated flash output `[nq * hd]`.
    pub gated: Vec<f32>,
}

/// Device linear-attention taps, un-permuted back to HF head order so they
/// compare directly against [`LinearTaps`].
pub struct DevLinearTaps {
    /// Raw `in_proj_qkv` output `[conv_dim]`.
    pub qkv_raw: Vec<f32>,
    /// Post-conv activations `[conv_dim]`.
    pub qkv_conv: Vec<f32>,
    /// `in_proj_z` output `[nv*vd]`.
    pub z: Vec<f32>,
    /// Recurrence output `[nv*vd]`.
    pub core: Vec<f32>,
    /// Gated-norm output `[nv*vd]`.
    pub gated: Vec<f32>,
}

impl<'ctx> Qwen4Dev<'ctx> {
    /// Device PLE gate + dilated conv for one token: `key_proj`/`value_proj`
    /// on device (needs the dense tier F32-resident), then the two fused PLE
    /// kernels. `ring_rows` is the host-canonical conv history
    /// ([`PleConvState::rows`], time-major — slot-major with `ring_pos = 0`);
    /// the device's post-dispatch ring is discarded, the host state stays
    /// canonical.
    pub fn ple(
        &mut self,
        weights: &Qwen4Weights<'_, '_>,
        cfg: &Qwen4ExpConfig,
        layer: usize,
        embeddings: &[f32],
        h: &[f32],
        ring_rows: &[f32],
    ) -> Result<DevPleTaps> {
        let pc = ple_config(cfg);
        let hh = pc.hc_hidden();
        ensure!(embeddings.len() == pc.ple_embed_dim, "ple embeddings width");
        ensure!(h.len() == hh, "ple hidden width");
        ensure!(
            ring_rows.len() == pc.short_conv_state_len() * hh,
            "ple ring rows"
        );
        let name = |suffix: &str| layer_tensor_name(layer, suffix);
        let kp = *weights.tensor(&name("ple.key_proj.weight"))?;
        let vp = *weights.tensor(&name("ple.value_proj.weight"))?;
        ensure!(
            kp.format == Qwen4DeviceFormat::F32 && vp.format == Qwen4DeviceFormat::F32,
            "device PLE needs the dense tier F32-resident (found {:?}/{:?})",
            kp.format,
            vp.format
        );
        let s = self.slots;
        self.write_f32(s.ple_emb, embeddings)?;
        self.write_f32(s.h, h)?;
        self.write_f32(s.ple_ring, ring_rows)?;
        let (kb, ko, kl) = weights.binding(&kp)?;
        let push =
            qwen36_router_gemv_params(hh as u32, pc.ple_embed_dim as u32, false).to_le_bytes();
        let d = qwen36_router_gemv_dispatch(hh as u32);
        self.rec(
            Kernel::Qwen36RouterGemv,
            Kernel::Qwen36RouterGemv.specialization_u32(),
            &push,
            &[
                Bind::A(s.ple_emb, (pc.ple_embed_dim * 4) as u64),
                Bind::Ext(kb, ko, kl),
                Bind::A(s.ple_k, (hh * 4) as u64),
            ],
            [d.x, d.y, d.z],
        )?;
        let (vb, vo, vl) = weights.binding(&vp)?;
        let push = qwen36_router_gemv_params(pc.hidden_size as u32, pc.ple_embed_dim as u32, false)
            .to_le_bytes();
        let d = qwen36_router_gemv_dispatch(pc.hidden_size as u32);
        self.rec(
            Kernel::Qwen36RouterGemv,
            Kernel::Qwen36RouterGemv.specialization_u32(),
            &push,
            &[
                Bind::A(s.ple_emb, (pc.ple_embed_dim * 4) as u64),
                Bind::Ext(vb, vo, vl),
                Bind::A(s.ple_v, (pc.hidden_size * 4) as u64),
            ],
            [d.x, d.y, d.z],
        )?;
        self.barrier();
        // The gate kernel reads the PLE norms RAW (it spells the `1 + w`
        // itself — the loader must NOT have folded these, which
        // `qwen4_upload::folds_norm_bias` guarantees).
        let (nk_b, nk_o, nk_l) = weights.binding_by_name(&name("ple.norm_key.weight"))?;
        let (nq_b, nq_o, nq_l) = weights.binding_by_name(&name("ple.norm_query.weight"))?;
        let (nc_b, nc_o, nc_l) = weights.binding_by_name(&name("ple.norm_conv.weight"))?;
        let push = qwen4_ple_gate_params(
            pc.hidden_size as u32,
            pc.hc_count as u32,
            1,
            pc.rms_norm_eps,
        )
        .to_le_bytes();
        let d = qwen4_ple_gate_dispatch(pc.hc_count as u32, 1);
        self.rec(
            Kernel::Qwen4PleGate,
            Kernel::Qwen4PleGate.specialization_u32(),
            &push,
            &[
                Bind::A(s.ple_k, (hh * 4) as u64),
                Bind::A(s.h, (hh * 4) as u64),
                Bind::A(s.ple_v, (pc.hidden_size * 4) as u64),
                Bind::Ext(nk_b, nk_o, nk_l),
                Bind::Ext(nq_b, nq_o, nq_l),
                Bind::Ext(nc_b, nc_o, nc_l),
                Bind::A(s.ple_g, (hh * 4) as u64),
                Bind::A(s.ple_gn, (hh * 4) as u64),
            ],
            [d.x, d.y, d.z],
        )?;
        self.barrier();
        let (cw_b, cw_o, cw_l) = weights.binding_by_name(&name("ple.conv1d.weight"))?;
        let push = qwen4_ple_conv_params(
            hh as u32,
            1,
            pc.conv_kernel_size as u32,
            pc.conv_dilation as u32,
            0,
        )
        .to_le_bytes();
        let d = qwen4_ple_conv_dispatch(hh as u32);
        self.rec(
            Kernel::Qwen4PleConv,
            Kernel::Qwen4PleConv.specialization_u32(),
            &push,
            &[
                Bind::A(s.ple_gn, (hh * 4) as u64),
                Bind::Ext(cw_b, cw_o, cw_l),
                Bind::A(s.ple_ring, (pc.short_conv_state_len() * hh * 4) as u64),
                Bind::A(s.ple_g, (hh * 4) as u64),
                Bind::A(s.ple_o, (hh * 4) as u64),
            ],
            [d.x, d.y, d.z],
        )?;
        self.flush()?;
        Ok(DevPleTaps {
            key: self.read_f32(s.ple_k, hh)?,
            value: self.read_f32(s.ple_v, pc.hidden_size)?,
            gated: self.read_f32(s.ple_g, hh)?,
            gated_normed: self.read_f32(s.ple_gn, hh)?,
            out: self.read_f32(s.ple_o, hh)?,
        })
    }

    /// Whether [`Self::full_attention`] can run `layer` (a KV plane exists and
    /// the dense projections + norms are F32-resident).
    #[must_use]
    pub fn full_attention_ready(&self, weights: &Qwen4Weights<'_, '_>, layer: usize) -> bool {
        self.full_idx.contains_key(&layer)
            && [
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
            ]
            .iter()
            .all(|sfx| {
                weights
                    .tensor(&layer_tensor_name(layer, sfx))
                    .is_ok_and(|t| t.format == Qwen4DeviceFormat::F32)
            })
            && weights
                .tensor(&layer_tensor_name(layer, "self_attn.q_norm.weight"))
                .is_ok()
            && weights
                .tensor(&layer_tensor_name(layer, "self_attn.k_norm.weight"))
                .is_ok()
    }

    /// Device full attention (dense QSA stub) for one token at `pos`, writing
    /// this position's K/V into the layer's f16 planes. F32 GEMVs → per-head
    /// q/k RMSNorm (pre-folded `1 + w` weights) → partial NeoX RoPE → f16 KV
    /// pack → per-head flash decode → per-element sigmoid gate → o-proj.
    pub fn full_attention(
        &mut self,
        weights: &Qwen4Weights<'_, '_>,
        cfg: &Qwen4ExpConfig,
        layer: usize,
        x: &[f32],
        pos: usize,
    ) -> Result<(Vec<f32>, DevFullTaps)> {
        let hd = cfg.head_dim;
        let nq = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let group = nq / nkv;
        let h = cfg.hidden_size;
        let q_dim = nq * hd;
        let kv_dim = nkv * hd;
        let full_idx = *self
            .full_idx
            .get(&layer)
            .ok_or_else(|| anyhow!("layer {layer} has no device KV plane"))?;
        {
            let kv = self
                .kv
                .as_ref()
                .ok_or_else(|| anyhow!("no device KV cache"))?;
            ensure!(
                ((pos + 1) * hd * 2) as u64 <= kv.plane_bytes,
                "position {pos} exceeds the device KV plane"
            );
        }
        let name = |sfx: &str| layer_tensor_name(layer, sfx);
        let q_w = *weights.tensor(&name("self_attn.q_proj.weight"))?;
        let k_w = *weights.tensor(&name("self_attn.k_proj.weight"))?;
        let v_w = *weights.tensor(&name("self_attn.v_proj.weight"))?;
        let o_w = *weights.tensor(&name("self_attn.o_proj.weight"))?;
        for (t, label) in [(&q_w, "q"), (&k_w, "k"), (&v_w, "v"), (&o_w, "o")] {
            ensure!(
                t.format == Qwen4DeviceFormat::F32,
                "device full attention needs {label}_proj F32-resident, found {:?}",
                t.format
            );
        }
        let s = self.slots;
        self.write_f32(s.x, x)?;
        let pos_bytes = i64::try_from(pos).map(|_| (pos as i32).to_le_bytes())?;
        self.arena
            .copy_from_host_at(s.pos, &pos_bytes)
            .map_err(|e| anyhow!("write rope pos: {e}"))?;

        // q/k/v projections off the one staged activation.
        for (t, n_out, dst) in [
            (&q_w, 2 * q_dim, s.qkv),
            (&k_w, kv_dim, s.kbuf),
            (&v_w, kv_dim, s.vbuf),
        ] {
            let (b, o, l) = weights.binding(t)?;
            let push = qwen36_router_gemv_params(n_out as u32, h as u32, false).to_le_bytes();
            let d = qwen36_router_gemv_dispatch(n_out as u32);
            self.rec(
                Kernel::Qwen36RouterGemv,
                Kernel::Qwen36RouterGemv.specialization_u32(),
                &push,
                &[
                    Bind::A(s.x, (h * 4) as u64),
                    Bind::Ext(b, o, l),
                    Bind::A(dst, (n_out * 4) as u64),
                ],
                [d.x, d.y, d.z],
            )?;
        }
        self.barrier();
        // Per-head q RMSNorm out of the interleaved [query|gate] block (source
        // row stride 2*hd extracts the query halves), packed into `q`.
        let (qn_b, qn_o, qn_l) = weights.binding_by_name(&name("self_attn.q_norm.weight"))?;
        let push = rms_norm_params_rows(hd as u32, nq as u32, (2 * hd) as u32, cfg.rms_norm_eps)
            .to_le_bytes();
        let d = rms_norm_dispatch_rows(nq as u32);
        self.rec(
            Kernel::RmsNorm,
            Kernel::RmsNorm.specialization_u32(),
            &push,
            &[
                Bind::A(s.qkv, (nq * 2 * hd * 4) as u64),
                Bind::Ext(qn_b, qn_o, qn_l),
                Bind::A(s.q, (q_dim * 4) as u64),
            ],
            [d.x, d.y, d.z],
        )?;
        // K norm in place (packed rows).
        let (kn_b, kn_o, kn_l) = weights.binding_by_name(&name("self_attn.k_norm.weight"))?;
        let push =
            rms_norm_params_rows(hd as u32, nkv as u32, hd as u32, cfg.rms_norm_eps).to_le_bytes();
        let d = rms_norm_dispatch_rows(nkv as u32);
        self.rec(
            Kernel::RmsNorm,
            Kernel::RmsNorm.specialization_u32(),
            &push,
            &[
                Bind::A(s.kbuf, (kv_dim * 4) as u64),
                Bind::Ext(kn_b, kn_o, kn_l),
                Bind::A(s.kbuf, (kv_dim * 4) as u64),
            ],
            [d.x, d.y, d.z],
        )?;
        self.barrier();
        // Partial NeoX RoPE, one dispatch per head.
        let rope_push =
            rope_neox_params(hd as u32, cfg.rotary_dim as u32, 1, cfg.rope_theta).to_le_bytes();
        let rd = rope_neox_dispatch(cfg.rotary_dim as u32, 1);
        for hh_i in 0..nq + nkv {
            let off = if hh_i < nq {
                s.q + (hh_i * hd * 4) as u64
            } else {
                s.kbuf + ((hh_i - nq) * hd * 4) as u64
            };
            self.rec(
                Kernel::RopeNeox,
                Kernel::RopeNeox.specialization_u32(),
                &rope_push,
                &[
                    Bind::A(off, (hd * 4) as u64),
                    Bind::A(s.pos, 8),
                    Bind::A(s.dummy, 8),
                    Bind::A(off, (hd * 4) as u64),
                    Bind::A(s.pos, 8),
                ],
                [rd.x, rd.y, rd.z],
            )?;
        }
        self.barrier();
        // f16 KV pack: roped K + raw V → the layer's planes at `pos`.
        let (k_rows, v_rows): (Vec<u64>, Vec<u64>) = {
            let kv = self.kv.as_ref().expect("checked above");
            (0..nkv)
                .map(|kvh| {
                    (
                        kv.row(kv.k_plane(full_idx, kvh), pos),
                        kv.row(kv.v_plane(full_idx, kvh), pos),
                    )
                })
                .unzip()
        };
        let pack_push = f16_kv_pack_params(hd as u32).to_le_bytes();
        let pd = f16_kv_pack_dispatch(hd as u32);
        for kvh in 0..nkv {
            self.rec(
                Kernel::F16KvPack,
                Kernel::F16KvPack.specialization_u32(),
                &pack_push,
                &[
                    Bind::A(s.kbuf + (kvh * hd * 4) as u64, (hd * 4) as u64),
                    Bind::Kv(k_rows[kvh], (hd * 2) as u64),
                ],
                [pd.x, pd.y, pd.z],
            )?;
            self.rec(
                Kernel::F16KvPack,
                Kernel::F16KvPack.specialization_u32(),
                &pack_push,
                &[
                    Bind::A(s.vbuf + (kvh * hd * 4) as u64, (hd * 4) as u64),
                    Bind::Kv(v_rows[kvh], (hd * 2) as u64),
                ],
                [pd.x, pd.y, pd.z],
            )?;
        }
        self.barrier();
        // Flash decode, one dispatch per query head against its kv head's
        // planes (gqa_ratio stays 1; the GQA map is host-side).
        let kv_len = pos + 1;
        let scale = 1.0f32 / (hd as f32).sqrt();
        let spec = FlashAttentionSpec::f32_f16(hd as u32);
        let fa_push = flash_attn_params(hd as u32, hd as u32, kv_len as u32, scale).to_le_bytes();
        let fd = flash_attn_dispatch();
        let (k_planes, v_planes): (Vec<u64>, Vec<u64>) = {
            let kv = self.kv.as_ref().expect("checked above");
            (0..nq)
                .map(|hh_i| {
                    let kvh = hh_i / group;
                    (kv.k_plane(full_idx, kvh), kv.v_plane(full_idx, kvh))
                })
                .unzip()
        };
        let kv_bytes = (kv_len * hd * 2) as u64;
        for hh_i in 0..nq {
            self.rec(
                Kernel::FlashAttn,
                spec.specialization_u32(),
                &fa_push,
                &[
                    Bind::A(s.q + (hh_i * hd * 4) as u64, (hd * 4) as u64),
                    Bind::Kv(k_planes[hh_i], kv_bytes),
                    Bind::Kv(v_planes[hh_i], kv_bytes),
                    Bind::A(s.dummy, 8),
                    Bind::A(s.dummy, 8),
                    Bind::A(s.attn + (hh_i * hd * 4) as u64, (hd * 4) as u64),
                    Bind::A(s.dummy, 8),
                ],
                [fd.x, fd.y, fd.z],
            )?;
        }
        self.barrier();
        // Per-element sigmoid gate from the interleaved gate halves, in place.
        let sg_push = sigmoid_mul_params(hd as u32).to_le_bytes();
        let sg_d = sigmoid_mul_dispatch(hd as u32);
        for hh_i in 0..nq {
            let gate_off = s.qkv + ((hh_i * 2 * hd + hd) * 4) as u64;
            let val_off = s.attn + (hh_i * hd * 4) as u64;
            self.rec(
                Kernel::SigmoidMul,
                Kernel::SigmoidMul.specialization_u32(),
                &sg_push,
                &[
                    Bind::A(gate_off, (hd * 4) as u64),
                    Bind::A(val_off, (hd * 4) as u64),
                    Bind::A(val_off, (hd * 4) as u64),
                ],
                [sg_d.x, sg_d.y, sg_d.z],
            )?;
        }
        self.barrier();
        // o-proj → the y slot.
        let (ob, oo, ol) = weights.binding(&o_w)?;
        let push = qwen36_router_gemv_params(h as u32, q_dim as u32, false).to_le_bytes();
        let d = qwen36_router_gemv_dispatch(h as u32);
        self.rec(
            Kernel::Qwen36RouterGemv,
            Kernel::Qwen36RouterGemv.specialization_u32(),
            &push,
            &[
                Bind::A(s.attn, (q_dim * 4) as u64),
                Bind::Ext(ob, oo, ol),
                Bind::A(s.y, (h * 4) as u64),
            ],
            [d.x, d.y, d.z],
        )?;
        self.flush()?;
        Ok((
            self.read_f32(s.y, h)?,
            DevFullTaps {
                q_full: self.read_f32(s.qkv, nq * 2 * hd)?,
                q_roped: self.read_f32(s.q, q_dim)?,
                gated: self.read_f32(s.attn, q_dim)?,
            },
        ))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Device linear attention: permuted F32 uploads + the two qwen35 shaders.
// ─────────────────────────────────────────────────────────────────────────────

fn upload_f32<'ctx>(ctx: &'ctx VulkanContext, data: &[f32]) -> Result<DeviceBuffer<'ctx>> {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut buf = DeviceBuffer::alloc_uma(ctx, bytes.len().max(4))
        .map_err(|e| anyhow!("alloc {} B upload: {e}", bytes.len()))?;
    buf.copy_from_host(&bytes)
        .map_err(|e| anyhow!("stage upload: {e}"))?;
    Ok(buf)
}

/// Map a device qkv CHANNEL (q|k|v concatenated) to its HF channel: q/k pass
/// through, V head blocks go through [`v_slot_perm`].
fn qkv_channel_to_hf(cfg: &Qwen4ExpConfig, c: usize) -> usize {
    let qk = 2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim;
    if c < qk {
        return c;
    }
    let vd = cfg.linear_value_head_dim;
    let slot = (c - qk) / vd;
    let within = (c - qk) % vd;
    qk + v_slot_perm(cfg.linear_num_key_heads, cfg.linear_num_value_heads, slot) * vd + within
}

/// Un-permute a `[nv * vd]` slot-ordered vector back to HF value-head order.
fn unpermute_v_vec(cfg: &Qwen4ExpConfig, dev: &[f32]) -> Vec<f32> {
    let vd = cfg.linear_value_head_dim;
    let nv = cfg.linear_num_value_heads;
    let nk = cfg.linear_num_key_heads;
    let mut out = vec![0.0f32; dev.len()];
    for slot in 0..nv {
        let orig = v_slot_perm(nk, nv, slot);
        out[orig * vd..(orig + 1) * vd].copy_from_slice(&dev[slot * vd..(slot + 1) * vd]);
    }
    out
}

/// Un-permute a `[conv_dim]` slot-ordered qkv vector back to HF order.
fn unpermute_qkv_vec(cfg: &Qwen4ExpConfig, dev: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; dev.len()];
    for (c, &v) in dev.iter().enumerate() {
        out[qkv_channel_to_hf(cfg, c)] = v;
    }
    out
}

/// The permuted F32 device weights of ONE linear-attention layer.
///
/// `qwen35_gated_delta_net.comp` tiles key heads over value heads
/// (`k_head = v_head % nk`); these uploads permute the VALUE-head axis so the
/// tiled map computes HF's `repeat_interleave` — see the module docs and
/// [`v_slot_perm`]. Built from the host bf16 weights, so it costs
/// ~230 MB of F32 per enabled layer; enable it for bring-up subsets, not for
/// all 36 linear layers at once.
pub struct DevLinearAttn<'ctx> {
    qkv_w: DeviceBuffer<'ctx>,
    z_w: DeviceBuffer<'ctx>,
    a_w: DeviceBuffer<'ctx>,
    b_w: DeviceBuffer<'ctx>,
    conv_w: DeviceBuffer<'ctx>,
    /// `-exp(A_log)` per SLOT — the kernel expects the GGUF pre-applied form.
    alog: DeviceBuffer<'ctx>,
    dt_bias: DeviceBuffer<'ctx>,
    norm_w: DeviceBuffer<'ctx>,
    out_w: DeviceBuffer<'ctx>,
}

impl<'ctx> DevLinearAttn<'ctx> {
    /// Build the permuted uploads from the host weights.
    pub fn new(
        ctx: &'ctx VulkanContext,
        cfg: &Qwen4ExpConfig,
        w: &HostLinearAttn<'_>,
    ) -> Result<Self> {
        let h = cfg.hidden_size;
        let kd = cfg.linear_key_head_dim;
        let vd = cfg.linear_value_head_dim;
        let nk = cfg.linear_num_key_heads;
        let nv = cfg.linear_num_value_heads;
        let kernel = cfg.linear_conv_kernel_dim;
        let conv_dim = 2 * nk * kd + nv * vd;
        let qk = 2 * nk * kd;

        // in_proj_qkv rows: q/k verbatim, V head blocks permuted.
        let src = w.qkv.to_f32();
        let mut qkv = vec![0.0f32; src.len()];
        qkv[..qk * h].copy_from_slice(&src[..qk * h]);
        for slot in 0..nv {
            let orig = v_slot_perm(nk, nv, slot);
            qkv[(qk + slot * vd) * h..(qk + (slot + 1) * vd) * h]
                .copy_from_slice(&src[(qk + orig * vd) * h..(qk + (orig + 1) * vd) * h]);
        }
        // in_proj_z rows: value-head blocks permuted.
        let src = w.z.to_f32();
        let mut z = vec![0.0f32; src.len()];
        for slot in 0..nv {
            let orig = v_slot_perm(nk, nv, slot);
            z[slot * vd * h..(slot + 1) * vd * h]
                .copy_from_slice(&src[orig * vd * h..(orig + 1) * vd * h]);
        }
        // a/b rows, A_log (as -exp), dt_bias: one row / element per value head.
        let permute_rows = |src: &[f32], row: usize| -> Vec<f32> {
            let mut out = vec![0.0f32; src.len()];
            for slot in 0..nv {
                let orig = v_slot_perm(nk, nv, slot);
                out[slot * row..(slot + 1) * row]
                    .copy_from_slice(&src[orig * row..(orig + 1) * row]);
            }
            out
        };
        let a = permute_rows(&w.a.to_f32(), h);
        let b = permute_rows(&w.b.to_f32(), h);
        let alog: Vec<f32> = (0..nv)
            .map(|slot| -w.a_log[v_slot_perm(nk, nv, slot)].exp())
            .collect();
        let dtb: Vec<f32> = (0..nv)
            .map(|slot| w.dt_bias[v_slot_perm(nk, nv, slot)])
            .collect();
        // conv rows (channel-major taps): V channels permuted.
        let mut conv = vec![0.0f32; w.conv.len()];
        for c in 0..conv_dim {
            let hf = qkv_channel_to_hf(cfg, c);
            conv[c * kernel..(c + 1) * kernel]
                .copy_from_slice(&w.conv[hf * kernel..(hf + 1) * kernel]);
        }
        // out_proj columns: value-head column blocks permuted.
        let src = w.out.to_f32();
        let mut out = vec![0.0f32; src.len()];
        for row in 0..h {
            for slot in 0..nv {
                let orig = v_slot_perm(nk, nv, slot);
                out[row * nv * vd + slot * vd..row * nv * vd + (slot + 1) * vd].copy_from_slice(
                    &src[row * nv * vd + orig * vd..row * nv * vd + (orig + 1) * vd],
                );
            }
        }
        Ok(Self {
            qkv_w: upload_f32(ctx, &qkv)?,
            z_w: upload_f32(ctx, &z)?,
            a_w: upload_f32(ctx, &a)?,
            b_w: upload_f32(ctx, &b)?,
            conv_w: upload_f32(ctx, &conv)?,
            alog: upload_f32(ctx, &alog)?,
            dt_bias: upload_f32(ctx, &dtb)?,
            norm_w: upload_f32(ctx, &w.norm)?,
            out_w: upload_f32(ctx, &out)?,
        })
    }

    /// One token on device, advancing the HOST-canonical state in place (it is
    /// uploaded permuted, advanced by the kernels, read back, un-permuted).
    pub fn forward(
        &self,
        dev: &mut Qwen4Dev<'ctx>,
        cfg: &Qwen4ExpConfig,
        x: &[f32],
        gdr_state: &mut [f32],
        conv_ring: &mut [f32],
    ) -> Result<(Vec<f32>, DevLinearTaps)> {
        let h = cfg.hidden_size;
        let kd = cfg.linear_key_head_dim;
        let vd = cfg.linear_value_head_dim;
        let nk = cfg.linear_num_key_heads;
        let nv = cfg.linear_num_value_heads;
        let kernel = cfg.linear_conv_kernel_dim;
        let conv_dim = 2 * nk * kd + nv * vd;
        let state_w = kernel - 1;
        ensure!(gdr_state.len() == nv * kd * vd, "gdr state length");
        ensure!(conv_ring.len() == conv_dim * state_w, "conv ring length");
        let s = dev.slots;

        // Host state → device layout: S blocks by slot, ring channel-major.
        let mut dev_state = vec![0.0f32; gdr_state.len()];
        for slot in 0..nv {
            let orig = v_slot_perm(nk, nv, slot);
            dev_state[slot * kd * vd..(slot + 1) * kd * vd]
                .copy_from_slice(&gdr_state[orig * kd * vd..(orig + 1) * kd * vd]);
        }
        let mut dev_ring = vec![0.0f32; conv_ring.len()];
        for c in 0..conv_dim {
            let hf = qkv_channel_to_hf(cfg, c);
            for t in 0..state_w {
                dev_ring[c * state_w + t] = conv_ring[t * conv_dim + hf];
            }
        }
        dev.write_f32(s.x, x)?;
        dev.write_f32(s.gdr_state, &dev_state)?;
        dev.write_f32(s.conv_ring, &dev_ring)?;

        // Four F32 GEMVs off the one staged activation.
        for (buf, n_out, dst) in [
            (&self.qkv_w, conv_dim, s.qkv),
            (&self.z_w, nv * vd, s.zbuf),
            (&self.a_w, nv, s.abuf),
            (&self.b_w, nv, s.bbuf),
        ] {
            let push = qwen36_router_gemv_params(n_out as u32, h as u32, false).to_le_bytes();
            let d = qwen36_router_gemv_dispatch(n_out as u32);
            dev.rec(
                Kernel::Qwen36RouterGemv,
                Kernel::Qwen36RouterGemv.specialization_u32(),
                &push,
                &[
                    Bind::A(s.x, (h * 4) as u64),
                    Bind::Ext(buf, 0, buf.len() as u64),
                    Bind::A(dst, (n_out * 4) as u64),
                ],
                [d.x, d.y, d.z],
            )?;
        }
        dev.barrier();
        // Depthwise conv + SiLU, advancing the ring.
        let push = qwen35_ssm_conv_params(conv_dim as u32, 1, kernel as u32).to_le_bytes();
        let d = qwen35_ssm_conv_dispatch(conv_dim as u32);
        dev.rec(
            Kernel::Qwen35SsmConv,
            Kernel::Qwen35SsmConv.specialization_u32(),
            &push,
            &[
                Bind::A(s.qkv, (conv_dim * 4) as u64),
                Bind::Ext(&self.conv_w, 0, self.conv_w.len() as u64),
                Bind::A(s.conv_ring, (conv_dim * state_w * 4) as u64),
                Bind::A(s.convo, (conv_dim * 4) as u64),
            ],
            [d.x, d.y, d.z],
        )?;
        dev.barrier();
        // Recurrent gated-delta update.
        let push = qwen35_gated_delta_net_params(nk as u32, nv as u32, kd as u32, vd as u32, 1)
            .to_le_bytes();
        let d = qwen35_gated_delta_net_dispatch(nv as u32);
        dev.rec(
            Kernel::Qwen35GatedDeltaNet,
            Kernel::Qwen35GatedDeltaNet.specialization_u32(),
            &push,
            &[
                Bind::A(s.convo, (conv_dim * 4) as u64),
                Bind::A(s.bbuf, (nv * 4) as u64),
                Bind::A(s.abuf, (nv * 4) as u64),
                Bind::Ext(&self.dt_bias, 0, self.dt_bias.len() as u64),
                Bind::Ext(&self.alog, 0, self.alog.len() as u64),
                Bind::A(s.gdr_state, (nv * kd * vd * 4) as u64),
                Bind::A(s.gdro, (nv * vd * 4) as u64),
            ],
            [d.x, d.y, d.z],
        )?;
        dev.barrier();
        // Gated norm: per-head RMSNorm (PLAIN ones-init weight) × gate(z).
        let push =
            rms_norm_params_rows(vd as u32, nv as u32, vd as u32, cfg.rms_norm_eps).to_le_bytes();
        let d = rms_norm_dispatch_rows(nv as u32);
        dev.rec(
            Kernel::RmsNorm,
            Kernel::RmsNorm.specialization_u32(),
            &push,
            &[
                Bind::A(s.gdro, (nv * vd * 4) as u64),
                Bind::Ext(&self.norm_w, 0, self.norm_w.len() as u64),
                Bind::A(s.attn, (nv * vd * 4) as u64),
            ],
            [d.x, d.y, d.z],
        )?;
        dev.barrier();
        let n_gate = (nv * vd) as u32;
        match cfg.output_gate {
            GateActivation::Sigmoid => {
                let push = sigmoid_mul_params(n_gate).to_le_bytes();
                let d = sigmoid_mul_dispatch(n_gate);
                dev.rec(
                    Kernel::SigmoidMul,
                    Kernel::SigmoidMul.specialization_u32(),
                    &push,
                    &[
                        Bind::A(s.zbuf, u64::from(n_gate) * 4),
                        Bind::A(s.attn, u64::from(n_gate) * 4),
                        Bind::A(s.attn, u64::from(n_gate) * 4),
                    ],
                    [d.x, d.y, d.z],
                )?;
            }
            GateActivation::Silu => {
                let push = swiglu_params(n_gate).to_le_bytes();
                let d = swiglu_dispatch(n_gate);
                dev.rec(
                    Kernel::SwiGlu,
                    Kernel::SwiGlu.specialization_u32(),
                    &push,
                    &[
                        Bind::A(s.zbuf, u64::from(n_gate) * 4),
                        Bind::A(s.attn, u64::from(n_gate) * 4),
                        Bind::A(s.attn, u64::from(n_gate) * 4),
                    ],
                    [d.x, d.y, d.z],
                )?;
            }
        }
        dev.barrier();
        // out-proj.
        let push = qwen36_router_gemv_params(h as u32, (nv * vd) as u32, false).to_le_bytes();
        let d = qwen36_router_gemv_dispatch(h as u32);
        dev.rec(
            Kernel::Qwen36RouterGemv,
            Kernel::Qwen36RouterGemv.specialization_u32(),
            &push,
            &[
                Bind::A(s.attn, (nv * vd * 4) as u64),
                Bind::Ext(&self.out_w, 0, self.out_w.len() as u64),
                Bind::A(s.y, (h * 4) as u64),
            ],
            [d.x, d.y, d.z],
        )?;
        dev.flush()?;

        // State back to HF order.
        let dev_state = dev.read_f32(s.gdr_state, nv * kd * vd)?;
        for slot in 0..nv {
            let orig = v_slot_perm(nk, nv, slot);
            gdr_state[orig * kd * vd..(orig + 1) * kd * vd]
                .copy_from_slice(&dev_state[slot * kd * vd..(slot + 1) * kd * vd]);
        }
        let dev_ring = dev.read_f32(s.conv_ring, conv_dim * state_w)?;
        for c in 0..conv_dim {
            let hf = qkv_channel_to_hf(cfg, c);
            for t in 0..state_w {
                conv_ring[t * conv_dim + hf] = dev_ring[c * state_w + t];
            }
        }

        let y = dev.read_f32(s.y, h)?;
        let taps = DevLinearTaps {
            qkv_raw: unpermute_qkv_vec(cfg, &dev.read_f32(s.qkv, conv_dim)?),
            qkv_conv: unpermute_qkv_vec(cfg, &dev.read_f32(s.convo, conv_dim)?),
            z: unpermute_v_vec(cfg, &dev.read_f32(s.zbuf, nv * vd)?),
            core: unpermute_v_vec(cfg, &dev.read_f32(s.gdro, nv * vd)?),
            gated: unpermute_v_vec(cfg, &dev.read_f32(s.attn, nv * vd)?),
        };
        Ok((y, taps))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The model: host transcription + device offload per stage residency.
// ─────────────────────────────────────────────────────────────────────────────

/// `sigmoid(shared_expert_gate · x) · down(silu(gate·x) ⊙ up·x)` on host.
fn host_shared_expert(moe: &HostMoe<'_>, x: &[f32]) -> Vec<f32> {
    let s = sigmoid32(moe.shexp_gate.matvec(x)[0]);
    let g = moe.sh_gate.matvec(x);
    let u = moe.sh_up.matvec(x);
    let act: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| silu32(g) * u).collect();
    let mut y = moe.sh_down.matvec(&act);
    for v in &mut y {
        *v *= s;
    }
    y
}

/// How much of the model rides the device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Qwen4ExpDeviceMode {
    /// No device at all — the pure host transcription (slow; the oracle lane).
    HostOnly,
    /// Full model: the NVFP4 expert stacks + the F32 small tier resident; the
    /// dense bf16 tier and `lm_head` stay host-side. This is the residency
    /// that FITS the driver's heapBudget — the full plan's F16 dense tier has
    /// no registered GEMV to consume it anyway, and dropping it (6.70 GiB) +
    /// `lm_head` (1.18 GiB) brings the commit under the ~70.7 GiB budget the
    /// 74.4 GiB heap actually grants.
    HybridExperts,
    /// A layer subset with the dense tier F32-resident too (bring-up /
    /// parity): all stages of the named layers run on device, including
    /// attention. Device linear attention builds permuted F32 uploads for the
    /// subset's linear layers.
    SubsetF32(Vec<usize>),
}

/// Qwen3.8-Flash-Next on the Vulkan lane. See the module docs for the
/// execution split; the short version: every stage runs on the device when
/// its weights are resident and falls back to the host transcription when
/// they are not, and the two implementations are diffed per stage by
/// `tests/qwen4_forward.rs`.
pub struct VulkanQwen4ExpModel<'ctx, 'st> {
    /// The parsed checkpoint config (`max_context` enforced per forward).
    pub cfg: Qwen4ExpConfig,
    st: &'st SafeTensorsDir,
    hc: HyperConnectionConfig,
    hash: NGramHash,
    tables: Qwen4HostTables<'st>,
    layers: BTreeMap<usize, HostLayer<'st>>,
    mixer: GatedResidualWeights,
    lm_head: HostDense<'st>,
    weights: Option<Qwen4Weights<'ctx, 'st>>,
    dev: Option<Qwen4Dev<'ctx>>,
    dev_linear: BTreeMap<usize, DevLinearAttn<'ctx>>,
    state: Qwen4ExpState,
    /// EOS + the generation-config extras — what the executor reports.
    pub stop_token_ids: Vec<u32>,
}

impl<'ctx, 'st> VulkanQwen4ExpModel<'ctx, 'st> {
    /// Load the model. `ctx = None` forces [`Qwen4ExpDeviceMode::HostOnly`].
    ///
    /// Host layers load EAGERLY for every layer (the forward needs all of
    /// them, and a lazy map fights the borrow checker for nothing).
    pub fn load(
        ctx: Option<&'ctx VulkanContext>,
        st: &'st SafeTensorsDir,
        cfg: Qwen4ExpConfig,
        mode: &Qwen4ExpDeviceMode,
    ) -> Result<Self> {
        ensure!(
            cfg.hidden_act == "silu",
            "only silu experts are implemented"
        );
        // AUDIT PIN: `norm_topk_prob` is absent from config.json and the HF
        // default is TRUE. A `false` here attenuates every MoE output ~2.5x —
        // finite, coherent-looking, wrong. Refuse to run rather than trust it.
        ensure!(
            cfg.norm_topk_prob,
            "norm_topk_prob parsed as false; the reference default is true — check qwen4_config"
        );
        let hc = hc_config(&cfg);
        let hash = NGramHash::new(cfg.ngram_hash_config(0))?;
        let tables = Qwen4HostTables::build(st)?;
        // The gather must address the real table; a mismatch reads unrelated
        // rows of a 47.68 GiB table and serves finite nonsense.
        ensure!(
            tables.ngram()?.rows() == hash.padded_vocab_size(),
            "n-gram table rows {} != hash padded vocab {}",
            tables.ngram()?.rows(),
            hash.padded_vocab_size()
        );

        let mut layers = BTreeMap::new();
        for l in 0..cfg.num_hidden_layers {
            layers.insert(l, load_host_layer(st, &cfg, l)?);
        }
        let mixer = load_hc(st, "model.language_model.hyper_connection_mixer", &hc, true)?;
        let lm_head = HostDense::load(st, "lm_head.weight", cfg.hidden_size, cfg.vocab_size)?;

        let (weights, dev, dev_linear) = match (ctx, mode) {
            (None, _) | (_, Qwen4ExpDeviceMode::HostOnly) => (None, None, BTreeMap::new()),
            (Some(ctx), Qwen4ExpDeviceMode::HybridExperts) => {
                let scope = Qwen4UploadScope {
                    lm_head: false,
                    ..Qwen4UploadScope::full()
                };
                let ucfg = Qwen4UploadConfig::default();
                let mut plan = plan_qwen4_upload(st, &ucfg, &scope)?;
                // Drop the F16 dense tier: no F16 GEMV exists to read it, and
                // WITH it the commit exceeds the driver's heapBudget (the
                // audit's landmine #1). Keep NVFP4 stacks + the F32 tier.
                plan.items.retain(|it| it.format != Qwen4DeviceFormat::F16);
                plan.device_bytes = plan.items.iter().map(|it| it.bytes).sum();
                ensure_within_driver_budget(ctx, plan.device_bytes, ucfg.reserve_bytes)?;
                let weights = upload_qwen4(ctx, st, &plan, &ucfg)?;
                let dev = Qwen4Dev::new(ctx, &cfg, &[], cfg.max_context)?;
                (Some(weights), Some(dev), BTreeMap::new())
            }
            (Some(ctx), Qwen4ExpDeviceMode::SubsetF32(subset)) => {
                let scope = Qwen4UploadScope {
                    lm_head: false,
                    ..Qwen4UploadScope::layers(subset)
                };
                let ucfg = Qwen4UploadConfig {
                    dense_format: Qwen4DeviceFormat::F32,
                    ..Qwen4UploadConfig::default()
                };
                let plan = plan_qwen4_upload(st, &ucfg, &scope)?;
                ensure_within_driver_budget(ctx, plan.device_bytes, ucfg.reserve_bytes)?;
                let weights = upload_qwen4(ctx, st, &plan, &ucfg)?;
                let full_dev: Vec<usize> = subset
                    .iter()
                    .copied()
                    .filter(|&l| cfg.layer_types[l] == Qwen4LayerType::FullAttention)
                    .collect();
                let dev = Qwen4Dev::new(ctx, &cfg, &full_dev, cfg.max_context)?;
                let mut dev_linear = BTreeMap::new();
                for &l in subset {
                    if cfg.layer_types[l] == Qwen4LayerType::LinearAttention {
                        let host = layers[&l].linear.as_ref().expect("linear layer weights");
                        dev_linear.insert(l, DevLinearAttn::new(ctx, &cfg, host)?);
                    }
                }
                (Some(weights), Some(dev), dev_linear)
            }
        };

        let state = Qwen4ExpState::new(&cfg, &hash);
        let stop_token_ids = cfg.stop_token_ids.clone();
        Ok(Self {
            cfg,
            st,
            hc,
            hash,
            tables,
            layers,
            mixer,
            lm_head,
            weights,
            dev,
            dev_linear,
            state,
            stop_token_ids,
        })
    }

    /// The current per-slot state (single slot).
    #[must_use]
    pub fn state(&self) -> &Qwen4ExpState {
        &self.state
    }

    /// Gather one token's concatenated n-gram embedding `[ple_embed_dim]`
    /// from the FP8 table (host-resident; 16 rows of 160).
    fn gather_ple_embedding(&self, row_ids: &[i64]) -> Result<Vec<f32>> {
        let table = self.tables.ngram()?;
        let mut out = Vec::with_capacity(row_ids.len() * table.head_dim());
        for &id in row_ids {
            ensure!(id >= 0, "negative n-gram row id {id}");
            out.extend(table.row(id as u64)?);
        }
        Ok(out)
    }

    /// One token → next-token logits `[vocab_size]`.
    ///
    /// Single-slot: `slot` must be 0. `start_pos == 0` resets the recurrent
    /// state; otherwise it must equal the materialized `seq_len`.
    pub fn forward_token(
        &mut self,
        slot: usize,
        _epoch: u64,
        token: u32,
        start_pos: usize,
    ) -> Result<Vec<f32>> {
        ensure!(
            slot == 0,
            "qwen4_exp Vulkan lane is single-slot (got slot {slot})"
        );
        ensure!(
            (token as usize) < self.cfg.vocab_size,
            "token {token} outside the vocab"
        );
        ensure!(
            start_pos < self.cfg.max_context,
            "position {start_pos} >= max_context {} (the cap that keeps the QSA dense stub exact)",
            self.cfg.max_context
        );
        if start_pos == 0 && self.state.seq_len != 0 {
            self.state = Qwen4ExpState::new(&self.cfg, &self.hash);
        }
        ensure!(
            start_pos == self.state.seq_len,
            "forward at {start_pos} but the state holds {} tokens (uncached full-prefix lane)",
            self.state.seq_len
        );

        // n-gram rows for THIS token, from the context BEFORE it.
        let ple_emb = if self.cfg.ple_layer_ids.is_empty() {
            Vec::new()
        } else {
            let ids = self.hash.row_ids(&self.state.ngram, &[i64::from(token)])?;
            self.gather_ple_embedding(&ids)?
        };
        self.state.ngram.push(&[i64::from(token)]);

        // Seed the hyper residual: the embedding tiled hc_count times.
        let embed = self.tables.embed_row(token as usize)?;
        let mut h = qwen4_hc::seed_hyper_state(&self.hc, &embed)?;

        let Self {
            cfg,
            st,
            hc,
            layers,
            weights,
            dev,
            dev_linear,
            state,
            ..
        } = self;

        for layer in 0..cfg.num_hidden_layers {
            let hl = layers
                .get(&layer)
                .ok_or_else(|| anyhow!("host layer {layer} not loaded"))?;

            // PLE injection (host lane; the device PLE is parity-covered but
            // the gather + ring already live host-side). UNCONDITIONAL on the
            // PLE layer — omitting it is a wrong forward, not a degraded one.
            if let Some(ple) = &hl.ple {
                let ring = state
                    .ple_conv
                    .get_mut(&layer)
                    .ok_or_else(|| anyhow!("no PLE conv state for layer {layer}"))?;
                let out = ple.forward(&ple_emb, &h, ring, None)?;
                for (hv, &ov) in h.iter_mut().zip(&out) {
                    *hv += ov;
                }
            }

            let hc_dev = weights.as_ref().is_some_and(|w| {
                dev.is_some()
                    && w.hyper_connection(Some(layer), HcSite::Attn).is_ok()
                    && w.hyper_connection(Some(layer), HcSite::Mlp).is_ok()
            });

            // ── attention sub-block ────────────────────────────────────
            let (x, host_gr) = if hc_dev {
                let d = dev.as_mut().expect("hc_dev checked");
                let w = weights.as_ref().expect("hc_dev checked");
                (d.hc_pre(w, hc, Some(layer), HcSite::Attn, &h)?, None)
            } else {
                let gr = qwen4_hc::gated_residual(hc, &hl.attn_hc, &h)?;
                (gr.block_input.clone(), Some(gr))
            };

            let y = match hl.kind {
                Qwen4LayerType::LinearAttention => {
                    let gdr = state
                        .gdr
                        .get_mut(&layer)
                        .ok_or_else(|| anyhow!("no gdr state"))?;
                    let ring = state
                        .conv
                        .get_mut(&layer)
                        .ok_or_else(|| anyhow!("no conv ring"))?;
                    match (dev.as_mut(), dev_linear.get(&layer)) {
                        (Some(d), Some(la)) => la.forward(d, cfg, &x, gdr, ring)?.0,
                        _ => {
                            let w = hl.linear.as_ref().expect("linear weights");
                            host_linear_attention(cfg, w, &x, gdr, ring).0
                        }
                    }
                }
                Qwen4LayerType::FullAttention => {
                    let dev_ready = weights
                        .as_ref()
                        .zip(dev.as_ref())
                        .is_some_and(|(w, d)| d.full_attention_ready(w, layer));
                    if dev_ready {
                        let d = dev.as_mut().expect("dev_ready");
                        let w = weights.as_ref().expect("dev_ready");
                        d.full_attention(w, cfg, layer, &x, start_pos)?.0
                    } else {
                        let w = hl.full.as_ref().expect("full weights");
                        let kv = state.kv.get_mut(&layer).ok_or_else(|| anyhow!("no KV"))?;
                        host_full_attention(cfg, w, &x, start_pos, kv).0
                    }
                }
            };

            if hc_dev {
                let d = dev.as_mut().expect("hc_dev checked");
                let w = weights.as_ref().expect("hc_dev checked");
                h = d.hc_combine(w, hc, Some(layer), HcSite::Attn, &y)?;
            } else {
                let gr = host_gr.expect("host gated residual");
                let inj = gr
                    .injection_weights
                    .as_ref()
                    .expect("layer site has injection");
                qwen4_hc::inject_block_output(hc, &mut h, inj, &y)?;
            }

            // ── MoE sub-block ──────────────────────────────────────────
            let (x, host_gr) = if hc_dev {
                let d = dev.as_mut().expect("hc_dev checked");
                let w = weights.as_ref().expect("hc_dev checked");
                (d.hc_pre(w, hc, Some(layer), HcSite::Mlp, &h)?, None)
            } else {
                let gr = qwen4_hc::gated_residual(hc, &hl.mlp_hc, &h)?;
                (gr.block_input.clone(), Some(gr))
            };

            let moe_dev = weights
                .as_ref()
                .is_some_and(|w| dev.is_some() && w.expert_stack(layer, ExpertProj::Gate).is_ok());
            let y = if moe_dev {
                let d = dev.as_mut().expect("moe_dev checked");
                let w = weights.as_ref().expect("moe_dev checked");
                let (mut y, taps) = d.moe(w, cfg, layer, &x)?;
                if !taps.shared_on_device {
                    let sh = host_shared_expert(&hl.moe, &x);
                    for (yv, &sv) in y.iter_mut().zip(&sh) {
                        *yv += sv;
                    }
                }
                y
            } else {
                host_moe(cfg, st, layer, &hl.moe, &x)?.0
            };

            if hc_dev {
                let d = dev.as_mut().expect("hc_dev checked");
                let w = weights.as_ref().expect("hc_dev checked");
                h = d.hc_combine(w, hc, Some(layer), HcSite::Mlp, &y)?;
            } else {
                let gr = host_gr.expect("host gated residual");
                let inj = gr
                    .injection_weights
                    .as_ref()
                    .expect("layer site has injection");
                qwen4_hc::inject_block_output(hc, &mut h, inj, &y)?;
            }
        }

        // Stream mixer (use_combine = false) collapses 10240 → 2560; there is
        // NO other final norm. Then lm_head (host bf16).
        let mixer_dev = self
            .weights
            .as_ref()
            .is_some_and(|w| self.dev.is_some() && w.hyper_connection(None, HcSite::Mixer).is_ok());
        let x = if mixer_dev {
            let d = self.dev.as_mut().expect("mixer_dev checked");
            let w = self.weights.as_ref().expect("mixer_dev checked");
            d.hc_pre(w, &self.hc, None, HcSite::Mixer, &h)?
        } else {
            qwen4_hc::gated_residual(&self.hc, &self.mixer, &h)?.block_input
        };
        let logits = self.lm_head.matvec(&x);

        self.state.seq_len += 1;
        Ok(logits)
    }
}

/// Refuse a device plan that exceeds the DRIVER'S heap budget, not just the
/// heap size. `vulkaninfo` on the target box: heap 1 is 74.43 GiB but
/// `heapBudget` is ~70.71 GiB — `ensure_fits` against the size passes and the
/// load then either dies with OUT_OF_DEVICE_MEMORY or gets silently demoted
/// to a ~5x-slower fallback tier. Checking the budget up front turns that
/// into a loud, immediate error.
fn ensure_within_driver_budget(ctx: &VulkanContext, plan_bytes: u64, reserve: u64) -> Result<()> {
    let heaps = ctx.memory_heaps();
    let budgets = ctx.memory_budgets();
    let mut limit = heaps
        .iter()
        .filter(|&&(_, local)| local)
        .map(|&(size, _)| size)
        .max()
        .unwrap_or(u64::MAX);
    if let Some(budgets) = budgets {
        // Budgets align with heap indices; take the device-local one.
        for (i, &(size, local)) in heaps.iter().enumerate() {
            if local && size == limit {
                if let Some(&(budget, usage)) = budgets.get(i) {
                    limit = limit.min(budget.saturating_sub(usage));
                }
                break;
            }
        }
    }
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    ensure!(
        plan_bytes.saturating_add(reserve) <= limit,
        "qwen4 device plan needs {:.2} GiB (+{:.2} GiB reserve) but the driver grants only \
         {:.2} GiB of the device-local heap (heapBudget, not heap size). KNOWN ISSUE: the \
         full-residency replan is scheduled work; run the hybrid residency or a subset.",
        gib(plan_bytes),
        gib(reserve),
        gib(limit),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift so failures reproduce.
    struct Rng(u64);
    impl Rng {
        fn next_f32(&mut self) -> f32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            ((x >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0
        }
    }

    /// Encode f32s as bf16 bytes (RNE), for synthetic `HostDense` tensors.
    fn bf16_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter()
            .flat_map(|&v| {
                let bits = v.to_bits();
                let lsb = (bits >> 16) & 1;
                let b16 = ((bits.wrapping_add(0x7fff + lsb)) >> 16) as u16;
                b16.to_le_bytes()
            })
            .collect()
    }

    fn max_rel(got: &[f32], want: &[f32]) -> f32 {
        assert_eq!(got.len(), want.len());
        got.iter()
            .zip(want)
            .map(|(&g, &w)| (g - w).abs() / w.abs().max(1e-4))
            .fold(0.0f32, f32::max)
    }

    /// A tiny config with the real model's SHAPE RELATIONS (nv/nk = 3 would
    /// need nk|nv; use 2) so index arithmetic is exercised without 47 GiB.
    fn mini_cfg() -> Qwen4ExpConfig {
        Qwen4ExpConfig {
            hidden_size: 8,
            num_hidden_layers: 2,
            vocab_size: 32,
            rms_norm_eps: 1e-6,
            hidden_act: "silu".into(),
            tie_word_embeddings: false,
            hc_count: 4,
            hc_lowrank: 4,
            layer_types: vec![
                Qwen4LayerType::LinearAttention,
                Qwen4LayerType::FullAttention,
            ],
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            attention_bias: false,
            rotary_dim: 2,
            partial_rotary_factor: 0.5,
            rope_theta: 1e4,
            max_position_embeddings: 64,
            linear_num_key_heads: 2,
            linear_num_value_heads: 4,
            linear_key_head_dim: 2,
            linear_value_head_dim: 2,
            linear_conv_kernel_dim: 4,
            output_gate: GateActivation::Sigmoid,
            num_experts: 8,
            num_experts_per_tok: 3,
            moe_intermediate_size: 4,
            shared_expert_intermediate_size: 4,
            norm_topk_prob: true,
            ple_layer_ids: vec![],
            ple_embed_dim: 8,
            ple_conv_kernel_size: 4,
            ngram_size: 3,
            heads_per_ngram: 8,
            ngram_vocab_size_base: 20_000_000,
            make_ngram_vocab_size_divisible_by: 128,
            ngram_seed: 1234,
            split_ngram_parts: 128,
            indexer: None,
            bos_token_id: None,
            eos_token_id: 0,
            stop_token_ids: vec![0],
            max_context: 64,
        }
    }

    #[test]
    fn v_slot_perm_is_a_bijection_and_fixes_the_k_head_map() {
        // The real shapes: 48 value heads over 16 key heads.
        let (nk, nv) = (16usize, 48usize);
        let group = nv / nk;
        let mut seen = vec![false; nv];
        for slot in 0..nv {
            let orig = v_slot_perm(nk, nv, slot);
            assert!(orig < nv, "perm({slot}) = {orig} out of range");
            assert!(
                !seen[orig],
                "perm({slot}) = {orig} repeats — not a bijection"
            );
            seen[orig] = true;
            // THE point: the kernel reads key head `slot % nk`; HF wants the
            // stored value head to read key head `orig / group`.
            assert_eq!(
                orig / group,
                slot % nk,
                "perm({slot}) reads the wrong key head"
            );
        }
        // And the mini shape used by the other tests.
        let (nk, nv) = (2usize, 4usize);
        for slot in 0..nv {
            assert_eq!(v_slot_perm(nk, nv, slot) / (nv / nk), slot % nk);
        }
    }

    #[test]
    fn qkv_channel_map_roundtrips_the_v_section() {
        let cfg = mini_cfg();
        let conv_dim = cfg.linear_conv_dim();
        let mut seen = vec![false; conv_dim];
        for c in 0..conv_dim {
            let hf = qkv_channel_to_hf(&cfg, c);
            assert!(!seen[hf], "channel {c} -> {hf} repeats");
            seen[hf] = true;
            if c < 2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim {
                assert_eq!(hf, c, "q/k channels must pass through");
            }
        }
        // unpermute(permute) round-trips a distinctive vector.
        let dev: Vec<f32> = (0..conv_dim).map(|i| i as f32).collect();
        let hf = unpermute_qkv_vec(&cfg, &dev);
        for c in 0..conv_dim {
            assert_eq!(hf[qkv_channel_to_hf(&cfg, c)], dev[c]);
        }
    }

    #[test]
    fn round_to_bf16_rounds_to_nearest_even() {
        // bf16 keeps 7 mantissa bits, so the half-ULP at 1.0 is 2^-8
        // (0x0000_8000 of dropped f32 bits). Neighbours: 1.0 and 1.0078125.
        let next = f32::from_bits(0x3F81_0000); // 1.0078125, representable
        assert_eq!(
            round_to_bf16(next),
            next,
            "representable values pass through"
        );
        assert_eq!(
            round_to_bf16(f32::from_bits(0x3F80_7FFF)),
            1.0,
            "below midpoint"
        );
        assert_eq!(
            round_to_bf16(f32::from_bits(0x3F80_8001)),
            next,
            "above midpoint"
        );
        // The exact tie against an EVEN kept mantissa stays down...
        assert_eq!(round_to_bf16(f32::from_bits(0x3F80_8000)), 1.0, "even tie");
        // ...and against an ODD kept mantissa rounds up to the even neighbour.
        assert_eq!(
            round_to_bf16(f32::from_bits(0x3F81_8000)),
            f32::from_bits(0x3F82_0000),
            "odd tie"
        );
    }

    #[test]
    fn threaded_bf16_matvec_matches_serial_f64() {
        // 1030 x 1100 crosses the 2^20 threading threshold with ragged chunks.
        let (out_dim, in_dim) = (1030usize, 1100usize);
        let mut rng = Rng(7);
        let w: Vec<f32> = (0..out_dim * in_dim).map(|_| rng.next_f32()).collect();
        let x: Vec<f32> = (0..in_dim).map(|_| rng.next_f32()).collect();
        let bytes = bf16_bytes(&w);
        let got = matvec_bf16(&bytes, in_dim, out_dim, &x);
        // Serial reference over the SAME bf16-rounded weights.
        for (r, &g) in got.iter().enumerate() {
            let want: f64 = bytes[r * in_dim * 2..(r + 1) * in_dim * 2]
                .chunks_exact(2)
                .zip(&x)
                .map(|(p, &xv)| {
                    f64::from(bf16_to_f32(u16::from_le_bytes([p[0], p[1]]))) * f64::from(xv)
                })
                .sum();
            assert!(
                (f64::from(g) - want).abs() <= want.abs().max(1e-3) * 1e-6,
                "row {r}: got {g}, want {want}"
            );
        }
    }

    #[test]
    fn host_router_topk_renormalizes_and_breaks_ties_low() {
        let logits = vec![0.5f32, 2.0, 2.0, -1.0, 3.0, 0.0, 2.0, -2.0];
        let (ids, weights) = host_router_topk(&logits, 3, true);
        // 4 (3.0) first, then the 2.0 tie broken to the LOWER ids: 1 then 2.
        assert_eq!(ids, vec![4, 1, 2]);
        let sum: f32 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "renormalized weights sum to {sum}"
        );
        assert!(weights[0] > weights[1] && (weights[1] - weights[2]).abs() < 1e-7);
        // Without renorm the kept weights are the raw softmax probabilities.
        let (_, raw) = host_router_topk(&logits, 3, false);
        let denom: f64 = logits.iter().map(|&l| f64::from(l - 3.0).exp()).sum();
        let want = (f64::from(-1.0f32).exp() / denom) as f32; // logit 2.0 - max 3.0
        assert!(
            (raw[1] - want).abs() < 1e-7,
            "raw prob {} want {want}",
            raw[1]
        );
    }

    #[test]
    fn rope_partial_rotates_the_leading_pair_and_passes_the_rest() {
        // rotary_dim 4 on an 8-wide head: pairs (0,2) and (1,3); dims 4..8 pass.
        let mut head: Vec<f32> = (1..=8).map(|v| v as f32).collect();
        let orig = head.clone();
        let (pos, theta) = (7usize, 100.0f32);
        rope_partial(&mut head, 4, pos, theta);
        for d in 0..2 {
            let freq = f64::from(theta).powf(-2.0 * d as f64 / 4.0);
            let (sin, cos) = (pos as f64 * freq).sin_cos();
            let a = f64::from(orig[d]);
            let b = f64::from(orig[d + 2]);
            assert!((f64::from(head[d]) - (a * cos - b * sin)).abs() < 1e-6);
            assert!((f64::from(head[d + 2]) - (a * sin + b * cos)).abs() < 1e-6);
        }
        assert_eq!(
            &head[4..],
            &orig[4..],
            "dims past rotary_dim must pass through"
        );
        // pos 0 is the identity.
        let mut head0 = orig.clone();
        rope_partial(&mut head0, 4, 0, theta);
        assert_eq!(head0, orig);
    }

    /// Synthetic weights, 3 tokens, against an INDEPENDENT recurrence written
    /// with a transposed state layout and f64 arithmetic. Catches the class of
    /// bug the V-slot permutation exists for: a wrong k-head map here produces
    /// finite, plausible output that only a second implementation can convict.
    #[test]
    fn host_linear_attention_matches_an_independent_recurrence() {
        let cfg = mini_cfg();
        let h = cfg.hidden_size;
        let (nk, nv, kd, vd) = (
            cfg.linear_num_key_heads,
            cfg.linear_num_value_heads,
            cfg.linear_key_head_dim,
            cfg.linear_value_head_dim,
        );
        let conv_dim = cfg.linear_conv_dim();
        let kernel = cfg.linear_conv_kernel_dim;
        let mut rng = Rng(0xDEAD_BEEF);
        // Integer-valued weights/activations keep every pre-conv sum exactly
        // representable in bf16, so the conv's bf16 rounding is a no-op and
        // the two implementations need not share the rounding helper.
        // Weights/activations in {-1, 0, 1}: every matvec / conv sum is an
        // integer <= 32, exact in bf16, so the impl's conv rounding is a no-op
        // and the reference need not replicate it.
        let mut int_vec =
            |n: usize| -> Vec<f32> { (0..n).map(|_| (rng.next_f32() * 1.2).round()).collect() };
        let qkv_w = int_vec(conv_dim * h);
        let z_w = int_vec(nv * vd * h);
        let a_w = int_vec(nv * h);
        let b_w = int_vec(nv * h);
        let conv_w = int_vec(conv_dim * kernel);
        let out_w = int_vec(h * nv * vd);
        let a_log: Vec<f32> = (0..nv).map(|i| -1.0 + 0.1 * i as f32).collect();
        let dt_bias: Vec<f32> = (0..nv).map(|i| 0.5 + 0.05 * i as f32).collect();
        let norm: Vec<f32> = (0..vd).map(|i| 1.0 + 0.01 * i as f32).collect();
        let (qkv_b, z_b, a_b, b_b, out_b) = (
            bf16_bytes(&qkv_w),
            bf16_bytes(&z_w),
            bf16_bytes(&a_w),
            bf16_bytes(&b_w),
            bf16_bytes(&out_w),
        );
        let w = HostLinearAttn {
            qkv: HostDense {
                bytes: &qkv_b,
                in_dim: h,
                out_dim: conv_dim,
            },
            z: HostDense {
                bytes: &z_b,
                in_dim: h,
                out_dim: nv * vd,
            },
            a: HostDense {
                bytes: &a_b,
                in_dim: h,
                out_dim: nv,
            },
            b: HostDense {
                bytes: &b_b,
                in_dim: h,
                out_dim: nv,
            },
            a_log: a_log.clone(),
            dt_bias: dt_bias.clone(),
            conv: conv_w.clone(),
            norm: norm.clone(),
            out: HostDense {
                bytes: &out_b,
                in_dim: nv * vd,
                out_dim: h,
            },
        };

        let tokens: Vec<Vec<f32>> = (0..3).map(|_| int_vec(h)).collect();

        // ── Independent reference: f64, state indexed [val][key], history
        //    kept as an explicit list instead of a rolled ring. ──
        let matref = |wv: &[f32], ind: usize, x: &[f32]| -> Vec<f64> {
            wv.chunks_exact(ind)
                .map(|row| {
                    row.iter()
                        .zip(x)
                        .map(|(&a, &b)| f64::from(a) * f64::from(b))
                        .sum()
                })
                .collect()
        };
        let mut history: Vec<Vec<f64>> = Vec::new(); // raw qkv per past token
        let mut state = vec![vec![vec![0.0f64; kd]; vd]; nv];
        let mut want_y: Vec<Vec<f64>> = Vec::new();
        for x in &tokens {
            let qkv = matref(&qkv_w, h, x);
            let z = matref(&z_w, h, x);
            let a = matref(&a_w, h, x);
            let b = matref(&b_w, h, x);
            history.push(qkv.clone());
            let t = history.len() - 1;
            let mut conv = vec![0.0f64; conv_dim];
            for (c, out) in conv.iter_mut().enumerate() {
                let mut acc = 0.0f64;
                for k in 0..kernel {
                    let src = t as i64 - (kernel as i64 - 1) + k as i64;
                    let v = if src < 0 {
                        0.0
                    } else {
                        history[src as usize][c]
                    };
                    acc += v * f64::from(conv_w[c * kernel + k]);
                }
                *out = acc / (1.0 + (-acc).exp()); // silu; the sum is bf16-exact
            }
            let group = nv / nk;
            let mut gated = vec![0.0f64; nv * vd];
            for vh in 0..nv {
                let kh = vh / group;
                let q = &conv[kh * kd..(kh + 1) * kd];
                let k = &conv[nk * kd + kh * kd..nk * kd + (kh + 1) * kd];
                let vvec = &conv[2 * nk * kd + vh * vd..2 * nk * kd + (vh + 1) * vd];
                let qs = (q.iter().map(|&v| v * v).sum::<f64>() + 1e-6)
                    .sqrt()
                    .recip();
                let ks = (k.iter().map(|&v| v * v).sum::<f64>() + 1e-6)
                    .sqrt()
                    .recip();
                let g = (-f64::from(a_log[vh]).exp())
                    * (1.0 + (f64::from(dt_bias[vh]) + a[vh]).exp()).ln();
                let eg = g.exp();
                let beta = 1.0 / (1.0 + (-b[vh]).exp());
                let mut core = vec![0.0f64; vd];
                for val in 0..vd {
                    for cell in &mut state[vh][val] {
                        *cell *= eg;
                    }
                    let kv_mem: f64 = (0..kd).map(|key| state[vh][val][key] * k[key] * ks).sum();
                    let delta = (vvec[val] - kv_mem) * beta;
                    for key in 0..kd {
                        state[vh][val][key] += k[key] * ks * delta;
                    }
                    core[val] = (0..kd)
                        .map(|key| state[vh][val][key] * q[key] * qs / (kd as f64).sqrt())
                        .sum();
                }
                let mean = core.iter().map(|&v| v * v).sum::<f64>() / vd as f64;
                let scale = (mean + f64::from(cfg.rms_norm_eps)).sqrt().recip();
                for val in 0..vd {
                    let gate = 1.0 / (1.0 + (-z[vh * vd + val]).exp());
                    gated[vh * vd + val] = core[val] * scale * f64::from(norm[val]) * gate;
                }
            }
            let gated_f32: Vec<f32> = gated.iter().map(|&v| v as f32).collect();
            want_y.push(matref(&out_w, nv * vd, &gated_f32));
        }

        // ── The implementation under test. ──
        let mut gdr = vec![0.0f32; nv * kd * vd];
        let mut ring = vec![0.0f32; conv_dim * (kernel - 1)];
        for (t, x) in tokens.iter().enumerate() {
            let (y, _taps) = host_linear_attention(&cfg, &w, x, &mut gdr, &mut ring);
            let want: Vec<f32> = want_y[t].iter().map(|&v| v as f32).collect();
            let rel = max_rel(&y, &want);
            assert!(
                rel < 5e-4,
                "token {t}: max rel err {rel} (f32 vs f64 recurrence)"
            );
        }
    }

    /// At position 0 the softmax has one term, so attention output is exactly
    /// the (f16-rounded) value, gated by the sigmoid of the q-projection's
    /// gate half; with an identity o-proj that IS the block output.
    #[test]
    fn host_full_attention_pos0_reduces_to_gated_values() {
        let cfg = mini_cfg();
        let h = cfg.hidden_size;
        let (nq, nkv, hd) = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
        );
        let q_dim = nq * hd;
        let mut rng = Rng(42);
        let mut int_vec =
            |n: usize| -> Vec<f32> { (0..n).map(|_| (rng.next_f32() * 3.0).round()).collect() };
        let q_w = int_vec(nq * hd * 2 * h);
        let k_w = int_vec(nkv * hd * h);
        let v_w = int_vec(nkv * hd * h);
        let mut o_w = vec![0.0f32; h * q_dim];
        for i in 0..h.min(q_dim) {
            o_w[i * q_dim + i] = 1.0;
        }
        let (q_b, k_b, v_b, o_b) = (
            bf16_bytes(&q_w),
            bf16_bytes(&k_w),
            bf16_bytes(&v_w),
            bf16_bytes(&o_w),
        );
        let w = HostFullAttn {
            q: HostDense {
                bytes: &q_b,
                in_dim: h,
                out_dim: nq * hd * 2,
            },
            k: HostDense {
                bytes: &k_b,
                in_dim: h,
                out_dim: nkv * hd,
            },
            v: HostDense {
                bytes: &v_b,
                in_dim: h,
                out_dim: nkv * hd,
            },
            o: HostDense {
                bytes: &o_b,
                in_dim: q_dim,
                out_dim: h,
            },
            q_norm: vec![0.0; hd],
            k_norm: vec![0.0; hd],
        };
        let x = int_vec(h);
        let mut kv = HostKv::default();
        let (y, taps) = host_full_attention(&cfg, &w, &x, 0, &mut kv);
        let group = nq / nkv;
        for head in 0..nq {
            let kvh = head / group;
            for d in 0..hd {
                let v_val = round_to_f16(taps.v_raw[kvh * hd + d]);
                let gate = taps.q_full[head * 2 * hd + hd + d];
                let want = v_val * sigmoid32(gate);
                let got = y[head * hd + d];
                assert!(
                    (got - want).abs() <= want.abs().max(1e-3) * 1e-4,
                    "head {head} dim {d}: got {got}, want {want}"
                );
            }
        }
        assert_eq!(kv.k.len(), nkv * hd, "one position cached");
    }
}
