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
//! ## Where the host stages get their arithmetic
//!
//! A host stage owns its SEMANTICS but not necessarily its GEMVs. Every
//! `W · x` in the host lane goes through [`DenseGemv`], which runs it on the
//! device when the projection's twin is resident and falls back to
//! [`HostDense::matvec`] when it is not — so `host.linear_attn` still spells
//! the conv ring, the gated-delta recurrence and the gated norm exactly as
//! before, while its 2896.9 MiB/token of `in_proj` bytes stream on the GPU.
//! The measured reason: those matvecs were 6779.61 MiB and 679.22 ms of a
//! 898.80 ms token (75.6%) at 10.47 GB/s, against ~205 GB/s on device.
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
    Qwen4DeviceFormat, Qwen4DeviceTensor, Qwen4HostTables, Qwen4UploadConfig, Qwen4UploadScope,
    Qwen4Weights, bf16_to_f32, expert_tensor_name, f32_to_f16, layer_tensor_name,
    plan_qwen4_upload, upload_qwen4,
};

use vulkan_kernels::{
    FlashAttentionSpec, GemvDenseSpec, Kernel, KernelCache, KernelParams, MAT_VEC_FUSION_SCALE0,
    f16_kv_pack_dispatch, f16_kv_pack_params, flash_attn_dispatch, flash_attn_params,
    gemv_dense_dispatch, gemv_id_dispatch, gemv_id_params_fused, gemv_params_f32_b,
    qwen4_block_perm_dispatch, qwen4_block_perm_params, qwen4_hc_combine_dispatch,
    qwen4_hc_combine_params, qwen4_hc_mix_dispatch, qwen4_hc_mix_params, qwen4_ple_conv_dispatch,
    qwen4_ple_conv_params, qwen4_ple_gate_dispatch, qwen4_ple_gate_params,
    qwen35_gated_delta_net_dispatch, qwen35_gated_delta_net_params, qwen35_ssm_conv_dispatch,
    qwen35_ssm_conv_params, qwen36_moe_weighted_accum_dispatch, qwen36_moe_weighted_accum_params,
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
    /// The checkpoint tensor name. Carried so a call site can ask for this
    /// projection's DEVICE twin ([`DenseGemv`]) without threading a second
    /// string alongside every weight — the residency is keyed by name.
    pub name: String,
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
            name: name.to_string(),
            bytes: st.tensor_data(name)?,
            in_dim,
            out_dim,
        })
    }

    /// `W @ x`, f64 accumulation.
    #[must_use]
    pub fn matvec(&self, x: &[f32]) -> Vec<f32> {
        let _p = prof::span_bytes("matvec", self.bytes.len() as u64);
        matvec_bf16(self.bytes, self.in_dim, self.out_dim, x)
    }

    /// The full matrix as f32 (row-major `[out, in]`), for device staging.
    #[must_use]
    pub fn to_f32(&self) -> Vec<f32> {
        bf16_vec(self.bytes)
    }
}

/// Routes [`HostDense`] projections to [`Qwen4Dev::dense_gemv`] when their
/// device twin is resident, and to the host transcription when it is not.
///
/// The host stages own the SEMANTICS of this model — the conv ring, the
/// gated-delta recurrence, RoPE, the softmax, the head permutations — and they
/// keep owning them. All this moves is the arithmetic of `W · x`, which is
/// 75.6% of a measured token and the only part of those stages that is pure
/// bandwidth. A stage with no resident twin is byte-identical to before.
///
/// Resolution is ALL-OR-NOTHING per batch: if any matrix of a shared-`x` group
/// is missing, the whole group runs on the host. Half a batch on each side
/// would still be correct, but it would make a stage's cost — and its error
/// profile — depend on which weights happened to fit, which is not something
/// a measurement should have to guess at.
pub struct DenseGemv<'a, 'ctx, 'st> {
    dev: &'a mut Qwen4Dev<'ctx>,
    weights: &'a Qwen4Weights<'ctx, 'st>,
}

impl<'a, 'ctx, 'st> DenseGemv<'a, 'ctx, 'st> {
    pub fn new(dev: &'a mut Qwen4Dev<'ctx>, weights: &'a Qwen4Weights<'ctx, 'st>) -> Self {
        Self { dev, weights }
    }

    /// This projection's device twin, if it is resident AND its logical shape
    /// matches. The shape check is not paranoia: `mul_mat_vec.comp` derives the
    /// row stride from `ncols`, so a twin that disagreed with the host tensor
    /// about the shape would read rows at the wrong stride and still run.
    fn twin(&self, d: &HostDense<'_>) -> Option<Qwen4DeviceTensor> {
        let t = *self.weights.tensor(&d.name).ok()?;
        (t.ncols == d.in_dim && t.nrows == d.out_dim && t.format != Qwen4DeviceFormat::Nvfp4)
            .then_some(t)
    }

    /// `y_j = W_j · x` for projections sharing `x` — one submit on device, or
    /// the host transcription for all of them.
    pub fn matvec_many(&mut self, mats: &[&HostDense<'_>], x: &[f32]) -> Result<Vec<Vec<f32>>> {
        let twins: Option<Vec<Qwen4DeviceTensor>> = mats.iter().map(|m| self.twin(m)).collect();
        match twins {
            Some(t) if !t.is_empty() => self.dev.dense_gemv(self.weights, x, &t),
            _ => Ok(mats.iter().map(|m| m.matvec(x)).collect()),
        }
    }

    /// [`Self::matvec_many`] for a single projection.
    pub fn matvec(&mut self, m: &HostDense<'_>, x: &[f32]) -> Result<Vec<f32>> {
        let mut out = self.matvec_many(&[m], x)?;
        out.pop()
            .ok_or_else(|| anyhow!("dense gemv returned nothing"))
    }
}

/// [`DenseGemv::matvec_many`] with an optional router — `None` is the pure
/// host transcription, which is what the parity oracle passes.
fn dense_many(
    gemv: Option<&mut DenseGemv<'_, '_, '_>>,
    mats: &[&HostDense<'_>],
    x: &[f32],
) -> Result<Vec<Vec<f32>>> {
    match gemv {
        Some(g) => g.matvec_many(mats, x),
        None => Ok(mats.iter().map(|m| m.matvec(x)).collect()),
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
///
/// `gemv = None` is the pure host oracle. With `Some`, only the four
/// in-projections and `out_proj` move to the device (see [`DenseGemv`]); the
/// conv, the recurrence and the gated norm stay here, bit for bit.
pub fn host_linear_attention(
    cfg: &Qwen4ExpConfig,
    w: &HostLinearAttn<'_>,
    x: &[f32],
    gdr_state: &mut [f32],
    conv_ring: &mut [f32],
    mut gemv: Option<&mut DenseGemv<'_, '_, '_>>,
) -> Result<(Vec<f32>, LinearTaps)> {
    let kd = cfg.linear_key_head_dim;
    let vd = cfg.linear_value_head_dim;
    let nk = cfg.linear_num_key_heads;
    let nv = cfg.linear_num_value_heads;
    let kernel = cfg.linear_conv_kernel_dim;
    let conv_dim = 2 * nk * kd + nv * vd;
    let group = nv / nk;
    assert_eq!(gdr_state.len(), nv * kd * vd, "gdr state length");
    assert_eq!(conv_ring.len(), conv_dim * (kernel - 1), "conv ring length");

    // All four share `x`, so they are one batch — and one submit. `a`/`b` are
    // F32-resident where `qkv`/`z` are F16; `dense_gemv` mixes the two.
    let mut proj = dense_many(gemv.as_deref_mut(), &[&w.qkv, &w.z, &w.a, &w.b], x)?.into_iter();
    let qkv_raw = proj.next().expect("qkv");
    let z = proj.next().expect("z");
    let a = proj.next().expect("a");
    let b = proj.next().expect("b");

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

    prof::phase("host.linattn.recurrence");
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

    prof::phase("host.linattn.post");
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

    let y = match gemv {
        Some(g) => g.matvec(&w.out, &gated)?,
        None => w.out.matvec(&gated),
    };
    Ok((
        y,
        LinearTaps {
            qkv_raw,
            qkv_conv,
            z,
            core,
            gated,
        },
    ))
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
    mut gemv: Option<&mut DenseGemv<'_, '_, '_>>,
) -> Result<(Vec<f32>, FullTaps)> {
    let hd = cfg.head_dim;
    let nq = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;
    let group = nq / nkv;
    let kv_dim = nkv * hd;
    assert_eq!(kv.k.len(), pos * kv_dim, "KV cache length vs position");

    let mut proj = dense_many(gemv.as_deref_mut(), &[&w.q, &w.k, &w.v], x)?.into_iter();
    let q_full = proj.next().expect("q");
    let mut k_new = proj.next().expect("k");
    let v_new = proj.next().expect("v");

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

    prof::phase("host.fullattn.sdpa");
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

    prof::phase("host.fullattn.post");
    // Per-element sigmoid gate from the interleaved q projection.
    let mut gated = vec![0.0f32; nq * hd];
    for h in 0..nq {
        for d in 0..hd {
            let gate = q_full[h * 2 * hd + hd + d];
            gated[h * hd + d] = attn[h * hd + d] * sigmoid32(gate);
        }
    }

    let y = match gemv {
        Some(g) => g.matvec(&w.o, &gated)?,
        None => w.o.matvec(&gated),
    };
    Ok((
        y,
        FullTaps {
            q_full,
            q_roped,
            k_roped,
            v_raw: v_new,
            gated,
        },
    ))
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
    dense_x: u64,
    dense_y: u64,
    dummy: u64,
    total: u64,
}

/// Activation capacity of [`DevSlots::dense_x`], in f32 elements. The widest
/// dense contraction in this checkpoint is 6144 (`out_proj`/`o_proj`).
const DENSE_X_ELEMS: usize = 8192;

/// Output capacity of [`DevSlots::dense_y`], in f32 elements — one batch's
/// worth. `lm_head`'s 248320 rows set the floor; the widest multi-matrix batch
/// is the linear-attention in-projections at 16512 (256-B padded).
const DENSE_Y_ELEMS: usize = 262_144;

/// Where each matrix of a [`Qwen4Dev::dense_gemv`] batch writes inside
/// `dense_y`, in f32 elements, plus the total the batch needs.
///
/// Every offset is a DESCRIPTOR offset, so it must satisfy the device's
/// `minStorageBufferOffsetAlignment` (16 or 64 bytes on this part). 64
/// elements = 256 B clears both and matches the arena's own slot granularity.
/// A 1-row matrix — the shared-expert scalar gate — therefore still costs a
/// full stride, which is why this is padding and not a running sum.
fn dense_batch_offsets(rows: &[usize]) -> (Vec<usize>, usize) {
    let mut offs = Vec::with_capacity(rows.len());
    let mut cursor = 0usize;
    for &n in rows {
        offs.push(cursor);
        cursor += n.next_multiple_of(64);
    }
    (offs, cursor)
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
            dense_x: take(DENSE_X_ELEMS as u64),
            dense_y: take(DENSE_Y_ELEMS as u64),
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
        let _p = prof::span_bytes("h2d", (data.len() * 4) as u64);
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.arena
            .copy_from_host_at(off, &bytes)
            .map_err(|e| anyhow!("arena write at {off}: {e}"))
    }

    fn read_f32(&self, off: u64, n: usize) -> Result<Vec<f32>> {
        let _p = prof::span_bytes("d2h", (n * 4) as u64);
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
        let _p = prof::span_bytes("d2h", (n * 4) as u64);
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
        let _p = prof::span("record");
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
        // Label the dispatch with the stage that recorded it, so
        // ARLE_GPU_TIMESTAMPS reports GPU-busy time in the SAME buckets as the
        // host table and the two can be subtracted.
        self.recorder.label_next(prof::current());
        record_dispatch(&mut self.recorder, pipeline, &set, push, groups);
        self.live.push(set);
        Ok(())
    }

    fn barrier(&mut self) {
        if self.open {
            self.recorder.barrier();
        }
    }

    /// Drain the recorder's per-dispatch GPU timestamp totals — `(label,
    /// dispatches, milliseconds)`, empty unless `ARLE_GPU_TIMESTAMPS` is set.
    /// The labels are [`prof`] stage names (see `rec`), so this table and the
    /// host table share buckets and their difference is submit + fence cost.
    pub fn take_gpu_profile(&mut self) -> Vec<(&'static str, u64, f64)> {
        self.recorder.take_gpu_profile()
    }

    /// Total `vkQueueSubmit`s over this runner's life.
    #[must_use]
    pub fn submit_count(&self) -> u64 {
        self.recorder.submit_count()
    }

    /// Submit everything recorded since the last flush and wait.
    fn flush(&mut self) -> Result<()> {
        if self.open {
            let _p = prof::span("submit");
            self.recorder
                .submit_and_wait()
                .map_err(|e| anyhow!("submit: {e}"))?;
            self.open = false;
        }
        self.live.clear();
        Ok(())
    }

    // ── dense GEMV: the device twin of `HostDense::matvec` ───────────────

    /// `y_j = W_j · x` for projections that SHARE the activation `x`, in ONE
    /// submit.
    ///
    /// This is where the dense tier's bytes stop crossing the CPU. The same
    /// 7.11 GB/token that streams at ~9.8 GB/s through [`matvec_bf16`] streams
    /// at ~205 GB/s here — but a submit costs ~0.14 ms of fence wall on this
    /// box, more than the GEMVs of an entire layer, so the batching is not a
    /// convenience: one submit per shared activation is what makes the move
    /// pay.
    ///
    /// Two shader families behind one call. F16 weights go to
    /// `mul_mat_vec_f16`, whose B operand is a PLAIN f32 vector rather than
    /// `block_q8_1_x4` (there is no non-quantized arm in `mul_mat_vecq`);
    /// F32 weights go to the generic `qwen36_router_gemv` the rest of this
    /// file already uses. Both write f32, so a batch may mix them — which the
    /// linear-attention in-projections need, their `a`/`b` rows being F32 and
    /// their `qkv`/`z` rows F16.
    /// Record ONE dense GEMV (`dst = W · src`, both arena offsets) without
    /// flushing — the format dispatch [`Self::dense_gemv`] and the resident
    /// linear-attention path share, so F16-vs-F32 routing cannot drift
    /// between them.
    fn record_dense_at(
        &mut self,
        weights: &Qwen4Weights<'_, '_>,
        t: &Qwen4DeviceTensor,
        src_off: u64,
        dst_off: u64,
    ) -> Result<()> {
        let s = self.slots;
        let (wb, wo, wl) = weights.binding(t)?;
        let src = Bind::A(src_off, (t.ncols * 4) as u64);
        let dst = Bind::A(dst_off, (t.nrows * 4) as u64);
        let ncols = u32::try_from(t.ncols)?;
        let nrows = u32::try_from(t.nrows)?;
        match t.format {
            Qwen4DeviceFormat::F16 => {
                // `mul_mat_vec.comp` never reads `p.stride_a` — the row
                // stride IS `p.ncols` — so only an exactly-packed
                // row-major weight can be expressed through this push
                // block. `ncols % 4` is the shader's one shape rule.
                ensure!(
                    ncols.is_multiple_of(4),
                    "dense_gemv: F16 ncols {ncols} is not a multiple of 4"
                );
                let spec = GemvDenseSpec::DEFAULT;
                let d = gemv_dense_dispatch(nrows, &spec);
                self.rec(
                    Kernel::GemvF16,
                    spec.specialization_u32(),
                    &gemv_params_f32_b(ncols, nrows).to_le_bytes(),
                    &[
                        Bind::Ext(wb, wo, wl),
                        src,
                        dst,
                        Bind::A(s.dummy, 8),
                        Bind::A(s.dummy, 8),
                    ],
                    [d.x, d.y, d.z],
                )?;
            }
            Qwen4DeviceFormat::F32 => {
                let d = qwen36_router_gemv_dispatch(nrows);
                self.rec(
                    Kernel::Qwen36RouterGemv,
                    Kernel::Qwen36RouterGemv.specialization_u32(),
                    &qwen36_router_gemv_params(nrows, ncols, false).to_le_bytes(),
                    &[src, Bind::Ext(wb, wo, wl), dst],
                    [d.x, d.y, d.z],
                )?;
            }
            Qwen4DeviceFormat::Nvfp4 => {
                return Err(anyhow!(
                    "dense_gemv on an NVFP4 tensor — the routed-expert path owns those"
                ));
            }
        }
        Ok(())
    }

    pub fn dense_gemv(
        &mut self,
        weights: &Qwen4Weights<'_, '_>,
        x: &[f32],
        mats: &[Qwen4DeviceTensor],
    ) -> Result<Vec<Vec<f32>>> {
        ensure!(!mats.is_empty(), "dense_gemv with no matrices");
        ensure!(
            x.len() <= DENSE_X_ELEMS,
            "dense_gemv activation of {} elements exceeds the {DENSE_X_ELEMS}-element scratch",
            x.len()
        );
        let s = self.slots;
        for t in mats {
            ensure!(
                t.ncols == x.len(),
                "dense_gemv: weight ncols {} != activation width {}",
                t.ncols,
                x.len()
            );
        }
        let rows: Vec<usize> = mats.iter().map(|t| t.nrows).collect();
        let (offs, cursor) = dense_batch_offsets(&rows);
        ensure!(
            cursor <= DENSE_Y_ELEMS,
            "dense_gemv batch wants {cursor} output elements, scratch holds {DENSE_Y_ELEMS}"
        );
        self.write_f32(s.dense_x, x)?;
        for (t, &off) in mats.iter().zip(&offs) {
            self.record_dense_at(weights, t, s.dense_x, s.dense_y + (off * 4) as u64)?;
        }
        self.flush()?;
        mats.iter()
            .zip(&offs)
            .map(|(t, &off)| self.read_f32(s.dense_y + (off * 4) as u64, t.nrows))
            .collect()
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
        let _s = prof::stage(match site {
            HcSite::Attn => "dev.hc.attn.pre",
            HcSite::Mlp => "dev.hc.mlp.pre",
            HcSite::Mixer => "dev.hc.mixer",
        });
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
        let _s = prof::stage(match site {
            HcSite::Attn => "dev.hc.attn.comb",
            _ => "dev.hc.mlp.comb",
        });
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
        let _s = prof::stage("dev.moe");
        prof::phase("dev.moe.router");
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
        prof::phase("dev.moe.gate_up");
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
        prof::phase("dev.moe.swiglu");
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
        prof::phase("dev.moe.down_accum");
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

        prof::phase("dev.moe.shared");
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
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let _s = prof::stage("dev.ple");
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
        let s = self.slots;
        self.write_f32(s.ple_emb, embeddings)?;
        self.write_f32(s.h, h)?;
        self.write_f32(s.ple_ring, ring_rows)?;
        // key/value projections from whichever format the tier holds — the
        // F32-only assert this replaced was the audit-noted blocker that kept
        // the whole PLE on the host once the dense tier went F16.
        self.record_dense_at(weights, &kp, s.ple_emb, s.ple_k)?;
        self.record_dense_at(weights, &vp, s.ple_emb, s.ple_v)?;
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
        // Production reads: the residual add and the advanced ring (the conv
        // kernel wrote it in place; the host `PleConvState` stays canonical
        // by taking it back). The full tap set is [`Self::ple_taps`].
        let out = self.read_f32(s.ple_o, hh)?;
        // The kernel is a RING: with ring_pos = 0 the read side matches the
        // host's oldest-first shift layout exactly, but the WRITE lands the
        // new row in slot 0 (the ex-oldest). One rotation restores
        // time-major oldest-first, so `PleConvState` can take it verbatim.
        let dev_ring = self.read_f32(s.ple_ring, pc.short_conv_state_len() * hh)?;
        let state_len = pc.short_conv_state_len();
        let mut ring = vec![0.0f32; dev_ring.len()];
        for t in 0..state_len {
            let src = (t + 1) % state_len;
            ring[t * hh..(t + 1) * hh].copy_from_slice(&dev_ring[src * hh..(src + 1) * hh]);
        }
        Ok((out, ring))
    }

    /// Post-[`Self::ple`] tap reads, for the parity harness only.
    pub fn ple_taps(&self, cfg: &Qwen4ExpConfig) -> Result<DevPleTaps> {
        let pc = ple_config(cfg);
        let hh = pc.hc_hidden();
        let s = self.slots;
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
        let _s = prof::stage("dev.full_attn");
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

// ─────────────────────────────────────────────────────────────────────────────
// Resident linear attention: device-canonical state, tier-resident weights.
// ─────────────────────────────────────────────────────────────────────────────

/// The linear-attention token on device WITHOUT [`DevLinearAttn`]'s two costs:
/// no 7.8 GiB of re-uploaded permuted weights (projections come from the
/// resident dense tier via the same record path as [`Qwen4Dev::dense_gemv`]),
/// and no 6.2 MB of per-layer state round-trips (the GDN state and conv ring
/// live HERE, on device, in the kernel's GGUF layout, and never cross the bus
/// during decode). The HF<->GGUF head map moves to the ACTIVATIONS —
/// ~24 KB/layer through [`Kernel::Qwen4BlockPerm`] — which is what makes the
/// weight-permuting route unnecessary.
///
/// While this path is active the `Qwen4ExpState::{gdr, conv}` host maps are
/// STALE — the device copy is canonical. [`Self::read_state`] exists for the
/// parity harness, not for the decode loop.
pub struct DevResidentLinAttn<'ctx> {
    /// dst-block -> src-block maps, u32: qkv (80 at [`Self::MAP_QKV`]) |
    /// HF->GGUF value heads (48 at [`Self::MAP_V`]) | its inverse (48 at
    /// [`Self::MAP_V_INV`]).
    maps: DeviceBuffer<'ctx>,
    /// Per linear layer, stride [`Self::aux_stride`]: conv taps in GGUF
    /// channel order, then `-exp(A_log)` and `dt_bias` slot-permuted (the
    /// GGUF pre-applied forms the kernels expect).
    aux: DeviceBuffer<'ctx>,
    /// Per linear layer, stride [`Self::state_stride`]: GDN S `[nv][kd][vd]`
    /// then conv ring `[channel][kernel-1]`, both GGUF layout.
    state: DeviceBuffer<'ctx>,
    /// Linear layer id -> dense index into `aux` / `state`.
    index: BTreeMap<usize, usize>,
}

impl<'ctx> DevResidentLinAttn<'ctx> {
    const MAP_QKV: u64 = 0;
    const MAP_V: u64 = 128 * 4;
    const MAP_V_INV: u64 = 192 * 4;

    fn aux_stride(cfg: &Qwen4ExpConfig) -> usize {
        // conv | alog | dtb, each region 64-f32 aligned so every binding
        // offset stays 256-B aligned regardless of the driver's minimum.
        let conv = cfg.linear_conv_dim() * cfg.linear_conv_kernel_dim;
        conv.next_multiple_of(64) + 64 + 64
    }

    fn state_stride(cfg: &Qwen4ExpConfig) -> usize {
        let nv = cfg.linear_num_value_heads;
        let (kd, vd) = (cfg.linear_key_head_dim, cfg.linear_value_head_dim);
        nv * kd * vd + cfg.linear_conv_dim() * (cfg.linear_conv_kernel_dim - 1)
    }

    /// Whether `layer` has resident state and aux here.
    pub fn covers(&self, layer: usize) -> bool {
        self.index.contains_key(&layer)
    }

    pub fn new<'a, 'st: 'a>(
        ctx: &'ctx VulkanContext,
        cfg: &Qwen4ExpConfig,
        layers: impl IntoIterator<Item = (usize, &'a HostLayer<'st>)>,
    ) -> Result<Self> {
        let nk = cfg.linear_num_key_heads;
        let nv = cfg.linear_num_value_heads;
        let kernel = cfg.linear_conv_kernel_dim;
        let conv_dim = cfg.linear_conv_dim();

        // Maps. qkv blocks: q/k identity, v offset by the head permutation.
        let qk_blocks = 2 * nk * cfg.linear_key_head_dim / cfg.linear_value_head_dim;
        let mut maps = vec![0u32; 240];
        for (b, m) in maps.iter_mut().enumerate().take(qk_blocks + nv) {
            *m = if b < qk_blocks {
                b as u32
            } else {
                (qk_blocks + v_slot_perm(nk, nv, b - qk_blocks)) as u32
            };
        }
        for slot in 0..nv {
            maps[128 + slot] = v_slot_perm(nk, nv, slot) as u32;
            maps[192 + v_slot_perm(nk, nv, slot)] = slot as u32;
        }
        let map_bytes: Vec<u8> = maps.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut maps_buf = DeviceBuffer::alloc_uma(ctx, map_bytes.len())
            .map_err(|e| anyhow!("resident maps alloc: {e}"))?;
        maps_buf
            .copy_from_host(&map_bytes)
            .map_err(|e| anyhow!("resident maps upload: {e}"))?;

        // Per-layer aux: conv taps to GGUF channel order, alog/dtb per slot.
        let stride = Self::aux_stride(cfg);
        let mut index = BTreeMap::new();
        let mut aux = Vec::new();
        for (layer, hl) in layers {
            let Some(w) = hl.linear.as_ref() else {
                continue;
            };
            let dense = index.len();
            index.insert(layer, dense);
            let base = dense * stride;
            aux.resize(base + stride, 0.0f32);
            for c in 0..conv_dim {
                let hf = qkv_channel_to_hf(cfg, c);
                aux[base + c * kernel..base + (c + 1) * kernel]
                    .copy_from_slice(&w.conv[hf * kernel..(hf + 1) * kernel]);
            }
            let alog_at = base + (conv_dim * kernel).next_multiple_of(64);
            for slot in 0..nv {
                let orig = v_slot_perm(nk, nv, slot);
                aux[alog_at + slot] = -w.a_log[orig].exp();
                aux[alog_at + 64 + slot] = w.dt_bias[orig];
            }
        }
        ensure!(
            !index.is_empty(),
            "resident linear attention with no layers"
        );
        let aux_buf = upload_f32(ctx, &aux)?;

        // Device-canonical state, zeroed = sequence start.
        let state_bytes = index.len() * Self::state_stride(cfg) * 4;
        let mut state = DeviceBuffer::alloc_uma(ctx, state_bytes)
            .map_err(|e| anyhow!("resident state alloc ({state_bytes} B): {e}"))?;
        state
            .copy_from_host(&vec![0u8; state_bytes])
            .map_err(|e| anyhow!("resident state zero: {e}"))?;

        Ok(Self {
            maps: maps_buf,
            aux: aux_buf,
            state,
            index,
        })
    }

    /// Sequence start: zero every layer's S and ring.
    pub fn reset(&mut self) -> Result<()> {
        let zero = vec![0u8; self.state.len()];
        self.state
            .copy_from_host(&zero)
            .map_err(|e| anyhow!("resident state reset: {e}"))
    }

    fn record_perm(
        &self,
        dev: &mut Qwen4Dev<'ctx>,
        src: u64,
        map_off: u64,
        dst: u64,
        block: u32,
        nblocks: u32,
    ) -> Result<()> {
        let push = qwen4_block_perm_params(block, nblocks).to_le_bytes();
        let d = qwen4_block_perm_dispatch(nblocks);
        dev.rec(
            Kernel::Qwen4BlockPerm,
            Kernel::Qwen4BlockPerm.specialization_u32(),
            &push,
            &[
                Bind::A(src, u64::from(block) * u64::from(nblocks) * 4),
                Bind::Ext(&self.maps, map_off, u64::from(nblocks) * 4),
                Bind::A(dst, u64::from(block) * u64::from(nblocks) * 4),
            ],
            [d.x, d.y, d.z],
        )
    }

    /// One token: record the whole linear-attention block and flush ONCE.
    /// Returns `y` `[hidden]`.
    pub fn forward(
        &self,
        dev: &mut Qwen4Dev<'ctx>,
        weights: &Qwen4Weights<'_, '_>,
        cfg: &Qwen4ExpConfig,
        layer: usize,
        x: &[f32],
    ) -> Result<Vec<f32>> {
        let _s = prof::stage("dev.linear_attn");
        let h = cfg.hidden_size;
        let kd = cfg.linear_key_head_dim;
        let vd = cfg.linear_value_head_dim;
        let nk = cfg.linear_num_key_heads;
        let nv = cfg.linear_num_value_heads;
        let kernel = cfg.linear_conv_kernel_dim;
        let conv_dim = cfg.linear_conv_dim();
        let state_w = kernel - 1;
        let dense = *self
            .index
            .get(&layer)
            .ok_or_else(|| anyhow!("layer {layer} not resident"))?;
        let aux_base = (dense * Self::aux_stride(cfg) * 4) as u64;
        let conv_w_len = (conv_dim * kernel * 4) as u64;
        let alog_at = aux_base + ((conv_dim * kernel).next_multiple_of(64) * 4) as u64;
        let state_base = (dense * Self::state_stride(cfg) * 4) as u64;
        let ring_at = state_base + (nv * kd * vd * 4) as u64;
        let s = dev.slots;

        // In-projections from the resident tier, into a scratch region of
        // `dense_y` laid out qkv | z | a | b at 64-f32 boundaries.
        let t_qkv = *weights.tensor(&layer_tensor_name(layer, "linear_attn.in_proj_qkv.weight"))?;
        let t_z = *weights.tensor(&layer_tensor_name(layer, "linear_attn.in_proj_z.weight"))?;
        let t_a = *weights.tensor(&layer_tensor_name(layer, "linear_attn.in_proj_a.weight"))?;
        let t_b = *weights.tensor(&layer_tensor_name(layer, "linear_attn.in_proj_b.weight"))?;
        let t_out = *weights.tensor(&layer_tensor_name(layer, "linear_attn.out_proj.weight"))?;
        let t_norm = *weights.tensor(&layer_tensor_name(layer, "linear_attn.norm.weight"))?;
        ensure!(
            t_norm.format == Qwen4DeviceFormat::F32,
            "linear_attn.norm must be F32-resident (RAW, per folds_norm_bias)"
        );
        ensure!(
            t_norm.nrows * t_norm.ncols == vd,
            "linear_attn.norm length {} != vd {vd}",
            t_norm.nrows * t_norm.ncols
        );
        let norm_bind = weights.binding(&t_norm)?;
        dev.write_f32(s.x, x)?;
        let at_qkv = s.dense_y;
        let at_z = at_qkv + (conv_dim.next_multiple_of(64) * 4) as u64;
        let at_a = at_z + ((nv * vd).next_multiple_of(64) * 4) as u64;
        let at_b = at_a + 64 * 4;
        for (t, dst) in [(&t_qkv, at_qkv), (&t_z, at_z), (&t_a, at_a), (&t_b, at_b)] {
            dev.record_dense_at(weights, t, s.x, dst)?;
        }
        dev.barrier();

        // HF -> GGUF: qkv/z as 128-wide head blocks, a/b as single slots.
        let vblk = vd as u32;
        self.record_perm(
            dev,
            at_qkv,
            Self::MAP_QKV,
            s.qkv,
            vblk,
            (conv_dim / vd) as u32,
        )?;
        self.record_perm(dev, at_z, Self::MAP_V, s.zbuf, vblk, nv as u32)?;
        self.record_perm(dev, at_a, Self::MAP_V, s.abuf, 1, nv as u32)?;
        self.record_perm(dev, at_b, Self::MAP_V, s.bbuf, 1, nv as u32)?;
        dev.barrier();

        // Conv + recurrence + gated norm, state advancing in place on device.
        let push = qwen35_ssm_conv_params(conv_dim as u32, 1, kernel as u32).to_le_bytes();
        let d = qwen35_ssm_conv_dispatch(conv_dim as u32);
        dev.rec(
            Kernel::Qwen35SsmConv,
            Kernel::Qwen35SsmConv.specialization_u32(),
            &push,
            &[
                Bind::A(s.qkv, (conv_dim * 4) as u64),
                Bind::Ext(&self.aux, aux_base, conv_w_len),
                Bind::Ext(&self.state, ring_at, (conv_dim * state_w * 4) as u64),
                Bind::A(s.convo, (conv_dim * 4) as u64),
            ],
            [d.x, d.y, d.z],
        )?;
        dev.barrier();
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
                Bind::Ext(&self.aux, alog_at + 64 * 4, (nv * 4) as u64),
                Bind::Ext(&self.aux, alog_at, (nv * 4) as u64),
                Bind::Ext(&self.state, state_base, (nv * kd * vd * 4) as u64),
                Bind::A(s.gdro, (nv * vd * 4) as u64),
            ],
            [d.x, d.y, d.z],
        )?;
        dev.barrier();
        let push =
            rms_norm_params_rows(vd as u32, nv as u32, vd as u32, cfg.rms_norm_eps).to_le_bytes();
        let d = rms_norm_dispatch_rows(nv as u32);
        dev.rec(
            Kernel::RmsNorm,
            Kernel::RmsNorm.specialization_u32(),
            &push,
            &[
                Bind::A(s.gdro, (nv * vd * 4) as u64),
                Bind::Ext(norm_bind.0, norm_bind.1, norm_bind.2),
                Bind::A(s.attn, (nv * vd * 4) as u64),
            ],
            [d.x, d.y, d.z],
        )?;
        dev.barrier();
        let n_gate = (nv * vd) as u32;
        let (kern, push) = match cfg.output_gate {
            GateActivation::Sigmoid => (Kernel::SigmoidMul, sigmoid_mul_params(n_gate)),
            GateActivation::Silu => (Kernel::SwiGlu, swiglu_params(n_gate)),
        };
        let d = sigmoid_mul_dispatch(n_gate);
        dev.rec(
            kern,
            kern.specialization_u32(),
            &push.to_le_bytes(),
            &[
                Bind::A(s.zbuf, u64::from(n_gate) * 4),
                Bind::A(s.attn, u64::from(n_gate) * 4),
                Bind::A(s.attn, u64::from(n_gate) * 4),
            ],
            [d.x, d.y, d.z],
        )?;
        dev.barrier();
        // GGUF -> HF, then out-proj from the resident tier.
        self.record_perm(dev, s.attn, Self::MAP_V_INV, s.dense_x, vblk, nv as u32)?;
        dev.barrier();
        dev.record_dense_at(weights, &t_out, s.dense_x, s.y)?;
        dev.flush()?;
        dev.read_f32(s.y, h)
    }

    /// Harness taps: read + unpermute the slot outputs of the LAST
    /// [`Self::forward`]. Only valid immediately after it.
    pub fn read_taps(&self, dev: &Qwen4Dev<'ctx>, cfg: &Qwen4ExpConfig) -> Result<DevLinearTaps> {
        let nv = cfg.linear_num_value_heads;
        let vd = cfg.linear_value_head_dim;
        let conv_dim = cfg.linear_conv_dim();
        Ok(DevLinearTaps {
            qkv_raw: unpermute_qkv_vec(cfg, &dev.read_f32(dev.slots.qkv, conv_dim)?),
            qkv_conv: unpermute_qkv_vec(cfg, &dev.read_f32(dev.slots.convo, conv_dim)?),
            z: unpermute_v_vec(cfg, &dev.read_f32(dev.slots.zbuf, nv * vd)?),
            core: unpermute_v_vec(cfg, &dev.read_f32(dev.slots.gdro, nv * vd)?),
            gated: unpermute_v_vec(cfg, &dev.read_f32(dev.slots.attn, nv * vd)?),
        })
    }

    /// Harness seeding: overwrite a layer's device state from HF-order host
    /// state. This is what keeps the parity harness a SINGLE-TOKEN error
    /// isolator: without it the device and host trajectories drift apart
    /// legitimately (the bf16 conv quantizer flips near-zero boundary
    /// channels differently on each side, and the state compounds it), and
    /// the per-element table stops being comparable across runs.
    pub fn seed_state(
        &mut self,
        cfg: &Qwen4ExpConfig,
        layer: usize,
        gdr_hf: &[f32],
        ring_hf: &[f32],
    ) -> Result<()> {
        let nk = cfg.linear_num_key_heads;
        let nv = cfg.linear_num_value_heads;
        let (kd, vd) = (cfg.linear_key_head_dim, cfg.linear_value_head_dim);
        let conv_dim = cfg.linear_conv_dim();
        let state_w = cfg.linear_conv_kernel_dim - 1;
        ensure!(gdr_hf.len() == nv * kd * vd, "seed gdr length");
        ensure!(ring_hf.len() == conv_dim * state_w, "seed ring length");
        let dense = *self
            .index
            .get(&layer)
            .ok_or_else(|| anyhow!("layer {layer} not resident"))?;
        let mut vals = vec![0.0f32; Self::state_stride(cfg)];
        let (dev_s, dev_ring) = vals.split_at_mut(nv * kd * vd);
        for slot in 0..nv {
            let orig = v_slot_perm(nk, nv, slot);
            dev_s[slot * kd * vd..(slot + 1) * kd * vd]
                .copy_from_slice(&gdr_hf[orig * kd * vd..(orig + 1) * kd * vd]);
        }
        for c in 0..conv_dim {
            let hf = qkv_channel_to_hf(cfg, c);
            for t in 0..state_w {
                dev_ring[c * state_w + t] = ring_hf[t * conv_dim + hf];
            }
        }
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.state
            .copy_from_host_at((dense * Self::state_stride(cfg) * 4) as u64, &bytes)
            .map_err(|e| anyhow!("resident state seed: {e}"))
    }

    /// Harness state view: download a layer's S and ring, back to HF order.
    pub fn read_state(&self, cfg: &Qwen4ExpConfig, layer: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let nk = cfg.linear_num_key_heads;
        let nv = cfg.linear_num_value_heads;
        let (kd, vd) = (cfg.linear_key_head_dim, cfg.linear_value_head_dim);
        let conv_dim = cfg.linear_conv_dim();
        let state_w = cfg.linear_conv_kernel_dim - 1;
        let dense = *self
            .index
            .get(&layer)
            .ok_or_else(|| anyhow!("layer {layer} not resident"))?;
        let base = dense * Self::state_stride(cfg) * 4;
        let mut bytes = vec![0u8; Self::state_stride(cfg) * 4];
        self.state
            .copy_to_host_at(base as u64, &mut bytes)
            .map_err(|e| anyhow!("resident state read: {e}"))?;
        let vals: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let (dev_s, dev_ring) = vals.split_at(nv * kd * vd);
        let mut gdr = vec![0.0f32; nv * kd * vd];
        for slot in 0..nv {
            let orig = v_slot_perm(nk, nv, slot);
            gdr[orig * kd * vd..(orig + 1) * kd * vd]
                .copy_from_slice(&dev_s[slot * kd * vd..(slot + 1) * kd * vd]);
        }
        let mut ring = vec![0.0f32; conv_dim * state_w];
        for c in 0..conv_dim {
            let hf = qkv_channel_to_hf(cfg, c);
            for t in 0..state_w {
                ring[t * conv_dim + hf] = dev_ring[c * state_w + t];
            }
        }
        Ok((gdr, ring))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The model: host transcription + device offload per stage residency.
// ─────────────────────────────────────────────────────────────────────────────

/// `sigmoid(shared_expert_gate · x) · down(silu(gate·x) ⊙ up·x)` on host.
fn host_shared_expert(
    moe: &HostMoe<'_>,
    x: &[f32],
    mut gemv: Option<&mut DenseGemv<'_, '_, '_>>,
) -> Result<Vec<f32>> {
    // The scalar gate rides in the same batch as gate/up: it is a 1-row F32
    // GEMV, and a submit of its own would cost more than the row does.
    let mut proj = dense_many(
        gemv.as_deref_mut(),
        &[&moe.shexp_gate, &moe.sh_gate, &moe.sh_up],
        x,
    )?
    .into_iter();
    let s = sigmoid32(proj.next().expect("shared gate")[0]);
    let g = proj.next().expect("gate_proj");
    let u = proj.next().expect("up_proj");
    let act: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| silu32(g) * u).collect();
    let mut y = match gemv {
        Some(gv) => gv.matvec(&moe.sh_down, &act)?,
        None => moe.sh_down.matvec(&act),
    };
    for v in &mut y {
        *v *= s;
    }
    Ok(y)
}

/// How much of the model rides the device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Qwen4ExpDeviceMode {
    /// No device at all — the pure host transcription (slow; the oracle lane).
    HostOnly,
    /// Full model, everything resident: the NVFP4 expert stacks, the F32 small
    /// tier, AND the F16 dense tier including `lm_head`.
    ///
    /// The dense tier used to be dropped here for two reasons, both now gone:
    /// no registered GEMV consumed it ([`Kernel::GemvF16`] does), and the plan
    /// did not fit the driver's ~70.7 GiB `heapBudget` (`spill_to_fit` moves
    /// the coldest expert stacks to the host heap, which a same-sitting sweep
    /// priced at 0.5% of read bandwidth). Keeping it resident is what takes the
    /// 6779.61 MiB/token of host bf16 matvec — 75.6% of a measured token — off
    /// the CPU.
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
    /// Device-resident linear attention (state on device, tier weights).
    /// `None` in host-only mode.
    resident_linear: Option<DevResidentLinAttn<'ctx>>,
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

        let (weights, dev, resident_linear) = match (ctx, mode) {
            (None, _) | (_, Qwen4ExpDeviceMode::HostOnly) => (None, None, None),
            (Some(ctx), Qwen4ExpDeviceMode::HybridExperts) => {
                // Everything, dense tier included. `upload_qwen4` owns the
                // budget guard AND the host-heap spill that makes the full
                // plan fit; a second copy of that arithmetic here could only
                // disagree with it.
                let ucfg = Qwen4UploadConfig::default();
                let plan = plan_qwen4_upload(st, &ucfg, &Qwen4UploadScope::full())?;
                let weights = upload_qwen4(ctx, st, &plan, &ucfg)?;
                let dev = Qwen4Dev::new(ctx, &cfg, &[], cfg.max_context)?;
                let resident =
                    DevResidentLinAttn::new(ctx, &cfg, layers.iter().map(|(l, hl)| (*l, hl)))?;
                (Some(weights), Some(dev), Some(resident))
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
                let weights = upload_qwen4(ctx, st, &plan, &ucfg)?;
                let full_dev: Vec<usize> = subset
                    .iter()
                    .copied()
                    .filter(|&l| cfg.layer_types[l] == Qwen4LayerType::FullAttention)
                    .collect();
                let dev = Qwen4Dev::new(ctx, &cfg, &full_dev, cfg.max_context)?;
                // Resident linear attention over whichever subset layers are
                // linear; none is fine (a full-attention-only subset).
                let any_linear = subset
                    .iter()
                    .any(|&l| cfg.layer_types[l] == Qwen4LayerType::LinearAttention);
                let resident = if any_linear {
                    Some(DevResidentLinAttn::new(
                        ctx,
                        &cfg,
                        subset
                            .iter()
                            .filter_map(|l| layers.get_key_value(l))
                            .map(|(l, hl)| (*l, hl)),
                    )?)
                } else {
                    None
                };
                (Some(weights), Some(dev), resident)
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
            resident_linear,
            state,
            stop_token_ids,
        })
    }

    /// The current per-slot state (single slot).
    #[must_use]
    pub fn state(&self) -> &Qwen4ExpState {
        &self.state
    }

    /// The device runner, for the profile harness. `None` in
    /// [`Qwen4ExpDeviceMode::HostOnly`].
    pub fn dev_mut(&mut self) -> Option<&mut Qwen4Dev<'ctx>> {
        self.dev.as_mut()
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
            if let Some(rl) = self.resident_linear.as_mut() {
                rl.reset()?;
            }
        }
        ensure!(
            start_pos == self.state.seq_len,
            "forward at {start_pos} but the state holds {} tokens (uncached full-prefix lane)",
            self.state.seq_len
        );

        let _token = prof::stage("token");
        let ngram_span = prof::stage("host.ngram_gather");
        // n-gram rows for THIS token, from the context BEFORE it.
        let ple_emb = if self.cfg.ple_layer_ids.is_empty() {
            Vec::new()
        } else {
            let ids = self.hash.row_ids(&self.state.ngram, &[i64::from(token)])?;
            self.gather_ple_embedding(&ids)?
        };
        self.state.ngram.push(&[i64::from(token)]);
        drop(ngram_span);

        // Seed the hyper residual: the embedding tiled hc_count times.
        let embed_span = prof::stage("host.embed_seed");
        let embed = self.tables.embed_row(token as usize)?;
        let mut h = qwen4_hc::seed_hyper_state(&self.hc, &embed)?;
        drop(embed_span);

        let Self {
            cfg,
            st,
            hc,
            layers,
            weights,
            dev,
            resident_linear,
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
                let ple_dev = weights.as_ref().zip(dev.as_mut()).filter(|(w, _)| {
                    w.tensor(&layer_tensor_name(layer, "ple.key_proj.weight"))
                        .is_ok()
                });
                let out = if let Some((w, d)) = ple_dev {
                    let (out, ring_rows) = d.ple(w, cfg, layer, &ple_emb, &h, ring.rows())?;
                    ring.rows_mut().copy_from_slice(&ring_rows);
                    out
                } else {
                    let _s = prof::stage("host.ple");
                    ple.forward(&ple_emb, &h, ring, None)?
                };
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
                let _s = prof::stage("host.hc_pre");
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
                    let resident = resident_linear
                        .as_ref()
                        .filter(|rl| rl.covers(layer))
                        .zip(dev.as_mut())
                        .zip(weights.as_ref());
                    if let Some(((rl, d), w)) = resident {
                        rl.forward(d, w, cfg, layer, &x)?
                    } else {
                        let _s = prof::stage("host.linear_attn");
                        let w = hl.linear.as_ref().expect("linear weights");
                        let mut gemv = dev
                            .as_mut()
                            .zip(weights.as_ref())
                            .map(|(d, wt)| DenseGemv::new(d, wt));
                        host_linear_attention(cfg, w, &x, gdr, ring, gemv.as_mut())?.0
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
                        let _s = prof::stage("host.full_attn");
                        let w = hl.full.as_ref().expect("full weights");
                        let kv = state.kv.get_mut(&layer).ok_or_else(|| anyhow!("no KV"))?;
                        let mut gemv = dev
                            .as_mut()
                            .zip(weights.as_ref())
                            .map(|(d, wt)| DenseGemv::new(d, wt));
                        host_full_attention(cfg, w, &x, start_pos, kv, gemv.as_mut())?.0
                    }
                }
            };

            if hc_dev {
                let d = dev.as_mut().expect("hc_dev checked");
                let w = weights.as_ref().expect("hc_dev checked");
                h = d.hc_combine(w, hc, Some(layer), HcSite::Attn, &y)?;
            } else {
                let _s = prof::stage("host.hc_comb");
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
                let _s = prof::stage("host.hc_pre");
                let gr = qwen4_hc::gated_residual(hc, &hl.mlp_hc, &h)?;
                (gr.block_input.clone(), Some(gr))
            };

            let moe_dev = weights
                .as_ref()
                .is_some_and(|w| dev.is_some() && w.expert_stack(layer, ExpertProj::Gate).is_ok());
            let y = if moe_dev {
                let (mut y, taps) = {
                    let d = dev.as_mut().expect("moe_dev checked");
                    let w = weights.as_ref().expect("moe_dev checked");
                    d.moe(w, cfg, layer, &x)?
                };
                if !taps.shared_on_device {
                    let _s = prof::stage("host.shared_expert");
                    let mut gemv = dev
                        .as_mut()
                        .zip(weights.as_ref())
                        .map(|(d, wt)| DenseGemv::new(d, wt));
                    let sh = host_shared_expert(&hl.moe, &x, gemv.as_mut())?;
                    for (yv, &sv) in y.iter_mut().zip(&sh) {
                        *yv += sv;
                    }
                }
                y
            } else {
                let _s = prof::stage("host.moe");
                host_moe(cfg, st, layer, &hl.moe, &x)?.0
            };

            if hc_dev {
                let d = dev.as_mut().expect("hc_dev checked");
                let w = weights.as_ref().expect("hc_dev checked");
                h = d.hc_combine(w, hc, Some(layer), HcSite::Mlp, &y)?;
            } else {
                let _s = prof::stage("host.hc_comb");
                let gr = host_gr.expect("host gated residual");
                let inj = gr
                    .injection_weights
                    .as_ref()
                    .expect("layer site has injection");
                qwen4_hc::inject_block_output(hc, &mut h, inj, &y)?;
            }
        }

        // Stream mixer (use_combine = false) collapses 10240 → 2560; there is
        // NO other final norm. Then lm_head — on device when it is resident,
        // which is worth 52 ms of the 899 ms token measured before this lane
        // existed: 1212.5 MiB in ONE projection.
        let mixer_dev = self
            .weights
            .as_ref()
            .is_some_and(|w| self.dev.is_some() && w.hyper_connection(None, HcSite::Mixer).is_ok());
        let x = if mixer_dev {
            let d = self.dev.as_mut().expect("mixer_dev checked");
            let w = self.weights.as_ref().expect("mixer_dev checked");
            d.hc_pre(w, &self.hc, None, HcSite::Mixer, &h)?
        } else {
            let _s = prof::stage("host.hc_pre");
            qwen4_hc::gated_residual(&self.hc, &self.mixer, &h)?.block_input
        };
        let lm_span = prof::stage("host.lm_head");
        let logits = match self.dev.as_mut().zip(self.weights.as_ref()) {
            Some((d, w)) => DenseGemv::new(d, w).matvec(&self.lm_head, &x)?,
            None => self.lm_head.matvec(&x),
        };
        drop(lm_span);

        self.state.seq_len += 1;
        Ok(logits)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-stage wall-clock profile (additive instrumentation; see `prof`).
// ─────────────────────────────────────────────────────────────────────────────

/// Where a token's wall clock goes, charged so the buckets PARTITION it.
///
/// The forward is a chain of synchronous stages, so a table of independently
/// measured stage timings would double-count (a stage's wall contains its
/// copies and its fence wait) and leave the reader unable to tell a real
/// residual from an overlap. Instead every timed scope adds its wall to a
/// thread-local "already charged" counter, and a [`Stage`] books only
/// `wall - (what its children charged)` under its own `cpu` part. The rows
/// therefore sum to the outermost stage's measured wall exactly, and the
/// unattributed remainder shows up as one honest line rather than as slack
/// smeared over the table.
///
/// Off — and untimed, no `Instant::now` — unless [`set_enabled`] turns it on.
/// The consumer is `tests/qwen4_forward.rs`'s `profile_forward_token`.
pub mod prof {
    use std::cell::Cell;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};

    /// One `(stage, part)` accumulator. `part` is `cpu` for a stage's own host
    /// work and `h2d`/`record`/`submit`/`d2h` for the device leaves.
    #[derive(Clone, Debug)]
    pub struct Row {
        /// The innermost open [`Stage`] (or [`phase`]) when this was charged.
        pub stage: &'static str,
        /// Which leaf of that stage.
        pub part: &'static str,
        /// Total wall charged here.
        pub nanos: u64,
        /// How many times this bucket was charged (dispatches for `record`,
        /// `vkQueueSubmit`s for `submit`, copies for `h2d`/`d2h`).
        pub calls: u64,
        /// Bytes moved, for the copy parts.
        pub bytes: u64,
    }

    static ON: AtomicBool = AtomicBool::new(false);
    static ROWS: Mutex<Vec<Row>> = Mutex::new(Vec::new());

    /// An open [`phase`] inside the current stage.
    #[derive(Clone, Copy)]
    struct Open {
        name: &'static str,
        t0: Instant,
        charged0: u64,
        parent: &'static str,
    }

    thread_local! {
        /// The innermost open stage — what [`Span`]s charge to, and the label
        /// `rec` hands the GPU timestamp profiler.
        static STAGE: Cell<&'static str> = const { Cell::new("(outside)") };
        /// Nanoseconds charged to any bucket so far. A stage's own `cpu` is
        /// its wall minus this counter's delta across its lifetime.
        static CHARGED: Cell<u64> = const { Cell::new(0) };
        /// The open phase, if any. One level; phases never straddle a stage.
        static PHASE: Cell<Option<Open>> = const { Cell::new(None) };
    }

    fn nanos(d: Duration) -> u64 {
        u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)
    }

    /// Turn timing on or off for every thread.
    pub fn set_enabled(on: bool) {
        ON.store(on, Ordering::Relaxed);
    }

    /// Whether scopes are currently timed.
    #[must_use]
    pub fn enabled() -> bool {
        ON.load(Ordering::Relaxed)
    }

    /// The innermost open stage's name.
    #[must_use]
    pub fn current() -> &'static str {
        STAGE.with(Cell::get)
    }

    fn accum(stage: &'static str, part: &'static str, nanos: u64, bytes: u64) {
        let Ok(mut rows) = ROWS.lock() else { return };
        match rows.iter_mut().find(|r| r.stage == stage && r.part == part) {
            Some(r) => {
                r.nanos += nanos;
                r.calls += 1;
                r.bytes += bytes;
            }
            None => rows.push(Row {
                stage,
                part,
                nanos,
                calls: 1,
                bytes,
            }),
        }
    }

    fn charge(n: u64) {
        CHARGED.with(|c| c.set(c.get().saturating_add(n)));
    }

    /// Drain and reset the table.
    #[must_use]
    pub fn take() -> Vec<Row> {
        ROWS.lock()
            .map(|mut r| std::mem::take(&mut *r))
            .unwrap_or_default()
    }

    /// A timed leaf of the current stage.
    pub struct Span {
        part: &'static str,
        bytes: u64,
        t0: Option<Instant>,
    }

    /// Time this scope as `part` of the current stage.
    #[must_use]
    pub fn span(part: &'static str) -> Span {
        span_bytes(part, 0)
    }

    /// Time this scope as `part`, also booking `bytes` moved.
    #[must_use]
    pub fn span_bytes(part: &'static str, bytes: u64) -> Span {
        Span {
            part,
            bytes,
            t0: enabled().then(Instant::now),
        }
    }

    impl Drop for Span {
        fn drop(&mut self) {
            let Some(t0) = self.t0 else { return };
            let dt = nanos(t0.elapsed());
            accum(current(), self.part, dt, self.bytes);
            charge(dt);
        }
    }

    /// A stage. Nests: an inner stage's whole wall is a child charge of the
    /// outer one, so the outermost stage's `cpu` row IS the residual.
    pub struct Stage {
        prev: &'static str,
        name: &'static str,
        t0: Option<Instant>,
        charged0: u64,
    }

    /// Open a stage for this scope.
    #[must_use]
    pub fn stage(name: &'static str) -> Stage {
        if !enabled() {
            return Stage {
                prev: current(),
                name,
                t0: None,
                charged0: 0,
            };
        }
        let prev = current();
        let charged0 = CHARGED.with(Cell::get);
        STAGE.with(|s| s.set(name));
        Stage {
            prev,
            name,
            t0: Some(Instant::now()),
            charged0,
        }
    }

    fn close(name: &'static str, t0: Instant, charged0: u64, parent: &'static str) {
        let wall = nanos(t0.elapsed());
        let children = CHARGED.with(Cell::get).saturating_sub(charged0);
        accum(name, "cpu", wall.saturating_sub(children), 0);
        CHARGED.with(|c| c.set(charged0.saturating_add(wall)));
        STAGE.with(|s| s.set(parent));
    }

    impl Drop for Stage {
        fn drop(&mut self) {
            let Some(t0) = self.t0 else { return };
            end_phase();
            close(self.name, t0, self.charged0, self.prev);
        }
    }

    /// Split a stage into sequential phases WITHOUT restructuring its body:
    /// each call closes the open phase and opens `name`; the enclosing
    /// [`Stage`] closes the last one. Keeps the instrumentation diff on a long
    /// straight-line device stage down to one line per phase boundary.
    pub fn phase(name: &'static str) {
        if !enabled() {
            return;
        }
        end_phase();
        let open = Open {
            name,
            t0: Instant::now(),
            charged0: CHARGED.with(Cell::get),
            parent: current(),
        };
        PHASE.with(|p| p.set(Some(open)));
        STAGE.with(|s| s.set(name));
    }

    /// Close the open phase, if any.
    pub fn end_phase() {
        let Some(o) = PHASE.with(Cell::take) else {
            return;
        };
        close(o.name, o.t0, o.charged0, o.parent);
    }
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
    /// A `HostDense` over raw bf16 bytes for the offline fixtures. The empty
    /// name is deliberate: these tensors have no device twin, and a name that
    /// accidentally matched a resident one would silently route the oracle
    /// through the GPU.
    fn dense(bytes: &[u8], in_dim: usize, out_dim: usize) -> HostDense<'_> {
        HostDense {
            name: String::new(),
            bytes,
            in_dim,
            out_dim,
        }
    }

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
            qkv: dense(&qkv_b, h, conv_dim),
            z: dense(&z_b, h, nv * vd),
            a: dense(&a_b, h, nv),
            b: dense(&b_b, h, nv),
            a_log: a_log.clone(),
            dt_bias: dt_bias.clone(),
            conv: conv_w.clone(),
            norm: norm.clone(),
            out: dense(&out_b, nv * vd, h),
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
            let (y, _taps) =
                host_linear_attention(&cfg, &w, x, &mut gdr, &mut ring, None).expect("host linear");
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
            q: dense(&q_b, h, nq * hd * 2),
            k: dense(&k_b, h, nkv * hd),
            v: dense(&v_b, h, nkv * hd),
            o: dense(&o_b, q_dim, h),
            q_norm: vec![0.0; hd],
            k_norm: vec![0.0; hd],
        };
        let x = int_vec(h);
        let mut kv = HostKv::default();
        let (y, taps) = host_full_attention(&cfg, &w, &x, 0, &mut kv, None).expect("host full");
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

    /// The profile buckets must PARTITION the wall, not overlap it: a leaf's
    /// time is subtracted from its parent's `cpu`, and a nested stage's whole
    /// wall from its parent's. Without that subtraction every row containing a
    /// child double-counts, and the table's "where did the token go" answer is
    /// inflated by exactly the amount that matters most.
    ///
    /// Sleeps are coarse (Windows' timer granularity is ~15.6 ms and rounds
    /// up), so the bands are wide; the defect this catches is a 1.5x sum, not
    /// a 5% one.
    #[test]
    fn prof_buckets_partition_the_wall() {
        use std::time::{Duration, Instant};
        let _ = prof::take();
        prof::set_enabled(true);
        let t0 = Instant::now();
        {
            let _outer = prof::stage("outer");
            {
                let _leaf = prof::span("d2h");
                std::thread::sleep(Duration::from_millis(40));
            }
            {
                let _inner = prof::stage("inner");
                {
                    let _leaf = prof::span("submit");
                    std::thread::sleep(Duration::from_millis(30));
                }
                std::thread::sleep(Duration::from_millis(30));
            }
            std::thread::sleep(Duration::from_millis(40));
        }
        let wall = t0.elapsed().as_nanos() as u64;
        prof::set_enabled(false);
        let rows = prof::take();

        // `prof`'s ROWS table is process-global while its stage stack is
        // thread-local, so a test running in PARALLEL with this one lands its
        // own rows here — `HostDense::matvec` opens a `matvec` span, and the
        // host attention oracles open `host.*` phases. That is not a defect in
        // the partition: the claim under test is that ONE thread's stage tree
        // partitions ITS wall, and a foreign thread's rows are noise from a
        // different tree. Scoping the sum to this test's own stage names is
        // what makes it a statement about nesting rather than about the
        // scheduler. (Without it the suite is red only under `--test-threads`
        // > 1, and only sometimes — the worst kind of red.)
        let mine = |r: &prof::Row| r.stage == "outer" || r.stage == "inner";
        let foreign = rows.iter().filter(|r| !mine(r)).count();
        let sum: u64 = rows.iter().filter(|r| mine(r)).map(|r| r.nanos).sum();
        let ratio = sum as f64 / wall as f64;
        assert!(
            (0.97..=1.03).contains(&ratio),
            "own rows sum to {ratio:.3}x the {:.1} ms wall ({foreign} foreign rows ignored): \
             {rows:?}",
            wall as f64 / 1e6
        );

        let at = |stage: &str, part: &str| {
            rows.iter()
                .find(|r| r.stage == stage && r.part == part)
                .map_or(0.0, |r| r.nanos as f64 / 1e6)
        };
        // `outer` slept 40 ms of its own and CONTAINED ~100 ms more.
        let outer_cpu = at("outer", "cpu");
        assert!(
            (25.0..75.0).contains(&outer_cpu),
            "outer cpu {outer_cpu:.1} ms should be its own ~40 ms sleep, not the ~140 ms it spans"
        );
        let inner_cpu = at("inner", "cpu");
        assert!(
            (20.0..60.0).contains(&inner_cpu),
            "inner cpu {inner_cpu:.1} ms should exclude its own ~30 ms leaf"
        );
        assert!(
            (30.0..75.0).contains(&at("outer", "d2h")),
            "leaf charged to its stage"
        );
        assert!(
            (20.0..60.0).contains(&at("inner", "submit")),
            "leaf charged to its stage"
        );
        assert!(
            rows.iter()
                .all(|r| r.stage != "outer" || r.part != "submit"),
            "the inner stage's leaf must not land on the outer stage"
        );
    }

    /// A batch's output offsets are DESCRIPTOR offsets, so the layout has to
    /// pad rather than pack: a running sum would bind the shared expert's
    /// 1-row scalar gate at byte 4, which no device accepts and which
    /// `storage_buffers_ranged` cannot fix for us.
    ///
    /// The two shapes here are the real ones — the shared expert's
    /// `{gate, gate_proj, up_proj}` and the linear-attention
    /// `{qkv, z, a, b}` — so a regression shows up as the wrong bytes, not as
    /// an abstract off-by-one.
    #[test]
    fn dense_batch_offsets_pad_every_output_to_a_bindable_stride() {
        let (offs, total) = dense_batch_offsets(&[1, 640, 640]);
        assert_eq!(offs, vec![0, 64, 704], "shared-expert batch");
        assert_eq!(total, 1344);

        let rows = [10240usize, 6144, 48, 48];
        let (offs, total) = dense_batch_offsets(&rows);
        assert_eq!(offs, vec![0, 10240, 16384, 16448], "linear in-projections");
        assert_eq!(total, 16512);

        for (i, (&o, &n)) in offs.iter().zip(&rows).enumerate() {
            assert_eq!(
                (o * 4) % 256,
                0,
                "output {i} at element {o} is not 256-B aligned"
            );
            let end = offs.get(i + 1).copied().unwrap_or(total);
            assert!(
                o + n <= end,
                "output {i} ({n} rows at {o}) runs into its neighbour"
            );
        }

        // `lm_head` is the widest projection this checkpoint asks for and it
        // goes alone; if it stops fitting, `dense_gemv` refuses the token.
        assert!(
            dense_batch_offsets(&[248_320]).1 <= DENSE_Y_ELEMS,
            "lm_head rows"
        );
        // `out_proj` / `o_proj` are the widest contraction (6144).
        const { assert!(DENSE_X_ELEMS >= 6_144, "widest dense activation") };
    }
}
