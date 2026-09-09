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
//!   with the reference's eps 1e-6, same as the host oracle (an earlier
//!   1e-12 deviation was retired once llama.cpp's qwen35/qwen4exp graphs
//!   confirmed `ggml_l2_norm(eps_norm=1e-6)` on both lanes).
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

use anyhow::{Context, Result, anyhow, bail, ensure};

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
    CoopmatShape, FlashAttentionSpec, GemvDenseSpec, Kernel, KernelCache, KernelParams,
    MAT_VEC_FUSION_SCALE0, MmSpec, QWEN4_GDN_CHUNK, QWEN4_GDN_STATE_COLS, f16_kv_pack_dispatch,
    f16_kv_pack_dispatch_rows, f16_kv_pack_params, f16_kv_pack_params_rows, flash_attn_dispatch,
    flash_attn_dispatch_batched, flash_attn_params, flash_attn_params_batched, gemv_dense_dispatch,
    gemv_dispatch, gemv_id_dispatch, gemv_id_params_fused, gemv_id_params_grouped,
    gemv_nvfp4_spec_cols, gemv_params_f32_b, gemv_params_f32_b_cols, mm_dispatch, mmq_params,
    qwen4_block_perm_dispatch, qwen4_block_perm_params, qwen4_block_scatter_dispatch,
    qwen4_block_scatter_params, qwen4_gdn_chunk_intra_dispatch, qwen4_gdn_chunk_params,
    qwen4_gdn_chunk_scratch_elems, qwen4_gdn_chunk_state_dispatch, qwen4_hc_combine_dispatch,
    qwen4_hc_combine_params, qwen4_hc_mix_dispatch, qwen4_hc_mix_params,
    qwen4_moe_plan_args_down_at, qwen4_moe_plan_args_gateup_at, qwen4_moe_plan_count_dispatch,
    qwen4_moe_plan_count_params, qwen4_moe_plan_emit_dispatch, qwen4_moe_plan_emit_params,
    qwen4_moe_plan_scan_dispatch, qwen4_moe_plan_scan_params, qwen4_moe_plan_scratch_elems,
    qwen4_ple_conv_dispatch, qwen4_ple_conv_params, qwen4_ple_gate_dispatch, qwen4_ple_gate_params,
    qwen35_gated_delta_net_dispatch, qwen35_gated_delta_net_params, qwen35_ssm_conv_dispatch,
    qwen35_ssm_conv_params, qwen36_moe_weighted_accum_dispatch, qwen36_moe_weighted_accum_params,
    qwen36_router_gemv_dispatch, qwen36_router_gemv_params, qwen36_router_topk_dispatch,
    qwen36_router_topk_params, record_dispatch, record_dispatch_indirect, repack_nvfp4_planes,
    rms_norm_dispatch_rows, rms_norm_params_grouped, rms_norm_params_rows, rope_neox_dispatch,
    rope_neox_dispatch_batched, rope_neox_params, rope_neox_params_batched, sigmoid_mul_dispatch,
    sigmoid_mul_params, sigmoid_mul_params_strided, swiglu_dispatch, swiglu_params,
};
use vulkan_sys::{CommandRecorder, DescriptorSet, DeviceBuffer, VulkanContext};

// ─────────────────────────────────────────────────────────────────────────────
// Small host math, shared by the transcription and the loaders.
// ─────────────────────────────────────────────────────────────────────────────

fn sigmoid64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub(crate) fn sigmoid32(x: f32) -> f32 {
    sigmoid64(f64::from(x)) as f32
}

pub(crate) fn silu32(x: f32) -> f32 {
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
pub(crate) fn round_to_f16(x: f32) -> f32 {
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
    /// A borrowed view over raw BF16 rows that is NOT a whole checkpoint
    /// tensor — the MTP head's per-expert slices of the stacked
    /// `experts.gate_up_proj`/`down_proj`. `name` should be the slice's
    /// device-twin key ([`crate::qwen4_upload::mtp_expert_slice_name`]) so
    /// [`DenseGemv`] can route it; the byte length is checked against the
    /// declared shape.
    pub(crate) fn from_bf16_rows(
        name: String,
        bytes: &'st [u8],
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Self> {
        ensure!(
            bytes.len() == in_dim * out_dim * 2,
            "`{name}`: {} B of BF16 for a [{out_dim}, {in_dim}] view",
            bytes.len()
        );
        Ok(Self {
            name,
            bytes,
            in_dim,
            out_dim,
        })
    }

    pub(crate) fn load(
        st: &'st SafeTensorsDir,
        name: &str,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Self> {
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
pub(crate) fn f32_tensor(st: &SafeTensorsDir, name: &str, expect: usize) -> Result<Vec<f32>> {
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

pub(crate) fn load_hc(
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
pub(crate) fn rope_partial(head: &mut [f32], rotary_dim: usize, pos: usize, theta: f32) {
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
pub(crate) fn head_rms_norm_bias(head: &mut [f32], w: &[f32], eps: f32) {
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
    /// One f32 `1.0` — the weight for `h += 1.0 * ple_out` accumulates.
    one: u64,
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

/// Shape of the resident `weight_scale_2` table in the `scale0` slot.
/// `DevSlots` is shape-static like its neighbors (`logits: take(512)`), so
/// these are literals; the seeder bounds-checks the real stacks against them.
const SCALE0_LAYERS: u64 = 48;
const SCALE0_EXPERTS: u64 = 512;

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
            // The RESIDENT `weight_scale_2` table: 48 layers x 3 projections
            // x 512 experts of f32, seeded once per layer on first touch.
            // The expert GEMV indexes it by expert id straight off the
            // router's device-side ids buffer, so routing never round-trips
            // through the host — deleting 48 per-layer fences per token.
            scale0: take(SCALE0_LAYERS * 3 * SCALE0_EXPERTS),
            one: take(1),
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

/// A device-resident `VkDispatchIndirectCommand`: where
/// [`Qwen4Dev::rec_indirect`] reads its workgroup counts. The offset must be
/// 4-byte aligned and the 12-byte triple fully inside `buffer` (which carries
/// `INDIRECT_BUFFER` usage — every `vulkan-sys` `alloc*` constructor sets it).
struct GroupedIndirect<'b> {
    buffer: &'b DeviceBuffer<'b>,
    args_offset: u64,
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
    /// Bitmask of layers whose `weight_scale_2` rows are already seeded into
    /// the resident scale table (first `moe` touch of the layer seeds them).
    scale0_seeded: u64,
    /// Which PLE layer's conv ring is live in the `ple_ring` slot, if any.
    /// [`Self::ple_record_resident`] advances the ring in place, so staged
    /// decode never round-trips the 9x10240 rows through the host; a
    /// host-canonical consumer syncs out via [`Self::read_ple_ring`].
    ple_ring_layer: Option<usize>,
    /// The resident ring's next write slot (the conv kernel's `ring_pos`).
    ple_ring_pos: u32,
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
            scale0_seeded: 0,
            ple_ring_layer: None,
            ple_ring_pos: 0,
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

    /// [`Self::rec`] with the workgroup counts read on the GPU from a
    /// [`GroupedIndirect`] args triple — same pipeline cache, descriptor
    /// plumbing and profiler labeling, only the dispatch command differs.
    fn rec_indirect(
        &mut self,
        kernel: Kernel,
        spec: &[(u32, u32)],
        push: &[u8],
        binds: &[Bind<'_>],
        args: GroupedIndirect<'_>,
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
        self.recorder.label_next(prof::current());
        record_dispatch_indirect(
            &mut self.recorder,
            pipeline,
            &set,
            push,
            args.buffer,
            args.args_offset,
        );
        self.live.push(set);
        Ok(())
    }

    fn barrier(&mut self) {
        if self.open {
            self.recorder.barrier();
        }
    }

    /// [`Self::barrier`] widened to the `DRAW_INDIRECT` stage — required
    /// between a compute WRITE of indirect args and the
    /// [`Self::rec_indirect`] that consumes them (see
    /// `CommandRecorder::barrier_indirect`).
    fn barrier_indirect(&mut self) {
        if self.open {
            self.recorder.barrier_indirect();
        }
    }

    /// Record one buffer-to-buffer copy into the open command buffer (opening
    /// it if needed) — the speculative-decode state snapshot/restore, which
    /// must ride the SAME submit as the dispatches around it: recorded before
    /// a verify chunk it captures the pre-chunk recurrent state for free, and
    /// as a restore it is ordered before the next chunk's reads by
    /// `copy_buffer`'s own barriers.
    fn record_copy(
        &mut self,
        src: &DeviceBuffer<'_>,
        src_offset: u64,
        dst: &DeviceBuffer<'_>,
        dst_offset: u64,
        len: u64,
    ) -> Result<()> {
        let _p = prof::span("record");
        if !self.open {
            self.recorder
                .begin()
                .map_err(|e| anyhow!("recorder begin: {e}"))?;
            self.open = true;
        }
        self.recorder
            .copy_buffer(src, src_offset, dst, dst_offset, len);
        Ok(())
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

    /// Submit the open batch WITHOUT waiting — the depth-2 pipeline half of
    /// [`Self::flush`]: the GPU chews this batch while the host records the
    /// next. Descriptor sets are NOT cleared (they must outlive the batch);
    /// the next [`Self::flush`] is the drain, and its fence wait covers every
    /// earlier submission on the in-order queue.
    fn flush_async(&mut self) -> Result<()> {
        if self.open {
            let _p = prof::span("submit");
            self.recorder
                .submit_async()
                .map_err(|e| anyhow!("async submit: {e}"))?;
            self.open = false;
        }
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
        let src = Bind::A(src_off, (t.ncols * 4) as u64);
        let dst = Bind::A(dst_off, (t.nrows * 4) as u64);
        self.record_dense_binds(weights, t, src, dst)
    }

    /// [`Self::record_dense_at`] with the activation/output bindings supplied
    /// by the caller — the chunked prefill's rows live in ITS arena, not this
    /// one, and the F16-vs-F32 routing must not fork per buffer.
    fn record_dense_binds(
        &mut self,
        weights: &Qwen4Weights<'_, '_>,
        t: &Qwen4DeviceTensor,
        src: Bind<'_>,
        dst: Bind<'_>,
    ) -> Result<()> {
        let s = self.slots;
        let (wb, wo, wl) = weights.binding(t)?;
        let ncols = u32::try_from(t.ncols)?;
        let nrows = u32::try_from(t.nrows)?;
        match t.format {
            Qwen4DeviceFormat::Bf16 | Qwen4DeviceFormat::F16 => {
                // `mul_mat_vec.comp` never reads `p.stride_a` — the row
                // stride IS `p.ncols` — so only an exactly-packed
                // row-major weight can be expressed through this push
                // block. `ncols % 4` is the shader's one shape rule.
                ensure!(
                    ncols.is_multiple_of(4),
                    "dense_gemv: F16 ncols {ncols} is not a multiple of 4"
                );
                let kernel = if t.format == Qwen4DeviceFormat::Bf16 {
                    Kernel::GemvBf16
                } else {
                    Kernel::GemvF16
                };
                let spec = GemvDenseSpec::DEFAULT;
                let d = gemv_dense_dispatch(nrows, &spec);
                self.rec(
                    kernel,
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
            Qwen4DeviceFormat::Q4K | Qwen4DeviceFormat::Q8_0 => {
                // W4A16 / W8A16: the same plain-f32 B contract as the float
                // arms — no activation quantization anywhere on this model.
                let (kernel, gate) = if t.format == Qwen4DeviceFormat::Q4K {
                    (Kernel::GemvQ4KDense, 256)
                } else {
                    (Kernel::GemvQ8_0Dense, 8)
                };
                ensure!(
                    ncols % gate == 0,
                    "{kernel:?}: ncols {ncols} is not a multiple of {gate}"
                );
                let d = gemv_dispatch(nrows);
                self.rec(
                    kernel,
                    kernel.specialization_u32(),
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
        self.write_f32(self.slots.h, h)?;
        self.hc_pre_record(weights, hc, layer, site)?;
        self.flush()?;
        self.read_f32(self.slots.x, hc.hidden_size)
    }

    /// [`Self::hc_pre`] minus the host round trip: `h` is already in the `h`
    /// slot, the block input lands in the `x` slot, nothing is flushed. The
    /// staged token loop chains these; the wrapper above serves the fallback
    /// path and the harness.
    pub fn hc_pre_record(
        &mut self,
        weights: &Qwen4Weights<'_, '_>,
        hc: &HyperConnectionConfig,
        layer: Option<usize>,
        site: HcSite,
    ) -> Result<()> {
        let _s = prof::stage(match site {
            HcSite::Attn => "dev.hc.attn.pre",
            HcSite::Mlp => "dev.hc.mlp.pre",
            HcSite::Mixer => "dev.hc.mixer",
        });
        let b = weights.hyper_connection(layer, site)?;
        let hh = hc.hc_hidden() as u64;
        let (norm_buf, norm_off, norm_len) = weights.binding(b.hc_norm)?;
        let (up_buf, up_off, up_len) = weights.binding(b.mix_up)?;
        ensure!(
            b.mix_up.format == b.mix_down.format,
            "hyper-connection mix_down/mix_up formats diverge"
        );
        let mix_kernel = match b.mix_up.format {
            Qwen4DeviceFormat::F32 => Kernel::Qwen4HcMix,
            Qwen4DeviceFormat::Bf16 => Kernel::Qwen4HcMixBf16,
            f => bail!("hyper-connection mix weights in unsupported format {f:?}"),
        };
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
        // mix_down GEMV rides the format-dispatching dense path (F32 router
        // gemv or verbatim-BF16), same as every other projection.
        self.record_dense_at(weights, b.mix_down, s.hn, s.u_raw)?;
        self.barrier();
        let push = qwen4_hc_mix_params(
            hc.hidden_size as u32,
            hc.hc_count as u32,
            hc.hc_lowrank as u32,
        )
        .to_le_bytes();
        let d = qwen4_hc_mix_dispatch(hc.hidden_size as u32);
        self.rec(
            mix_kernel,
            mix_kernel.specialization_u32(),
            &push,
            &[
                Bind::A(s.hn, hh * 4),
                Bind::Ext(up_buf, up_off, up_len),
                Bind::A(s.u_raw, hc.hc_lowrank as u64 * 4),
                Bind::A(s.x, hc.hidden_size as u64 * 4),
            ],
            [d.x, d.y, d.z],
        )
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
        self.write_f32(self.slots.y, y)?;
        let y_off = self.slots.y;
        self.hc_combine_record(weights, hc, layer, site, y_off)?;
        self.flush()?;
        self.read_f32(self.slots.h, hc.hc_hidden())
    }

    /// [`Self::hc_combine`] minus the host round trip: the block output is
    /// read from `y_off` (the `y` slot for attention, the MoE accumulator for
    /// the MLP site) and the updated residual stays in the `h` slot.
    pub fn hc_combine_record(
        &mut self,
        weights: &Qwen4Weights<'_, '_>,
        hc: &HyperConnectionConfig,
        layer: Option<usize>,
        site: HcSite,
        y_off: u64,
    ) -> Result<()> {
        let _s = prof::stage(match site {
            HcSite::Attn => "dev.hc.attn.comb",
            _ => "dev.hc.mlp.comb",
        });
        let b = weights.hyper_connection(layer, site)?;
        let inject = b
            .block_inject
            .ok_or_else(|| anyhow!("hc_combine on the mixer site (no block_inject)"))?;
        let (inj_buf, inj_off, inj_len) = weights.binding(inject)?;
        let comb_kernel = match inject.format {
            Qwen4DeviceFormat::F32 => Kernel::Qwen4HcCombine,
            Qwen4DeviceFormat::Bf16 => Kernel::Qwen4HcCombineBf16,
            f => bail!("hyper-connection inject weights in unsupported format {f:?}"),
        };
        let hh = hc.hc_hidden() as u64;
        let s = self.slots;
        let push = qwen4_hc_combine_params(hc.hidden_size as u32, hc.hc_count as u32).to_le_bytes();
        let d = qwen4_hc_combine_dispatch(hc.hidden_size as u32);
        self.rec(
            comb_kernel,
            comb_kernel.specialization_u32(),
            &push,
            &[
                Bind::A(s.hn, hh * 4),
                Bind::Ext(inj_buf, inj_off, inj_len),
                Bind::A(s.h, hh * 4),
                Bind::A(y_off, hc.hidden_size as u64 * 4),
            ],
            [d.x, d.y, d.z],
        )
    }

    // ── MoE ──────────────────────────────────────────────────────────────

    /// Device MoE for one token. Router GEMV + top-k on device, ids read back
    /// once for the slot-ordered `weight_scale_2` gather, then the three NVFP4
    /// fused expert GEMVs (PLAIN f32 activations — no q8_1 quantize on this
    /// path) and the weighted accumulate. The shared expert rides on device
    /// when its dense tier is F32-resident; `taps.shared_on_device` says so.
    /// `collect_taps` inserts the extra flush that captures `routed` (the
    /// accumulator BEFORE the shared expert) for the parity harness; the
    /// decode loop passes `false` and the whole expert tail — gate, up,
    /// swiglu, down, weighted accum, shared expert — is ONE submit.
    pub fn moe(
        &mut self,
        weights: &Qwen4Weights<'_, '_>,
        cfg: &Qwen4ExpConfig,
        layer: usize,
        x: &[f32],
        collect_taps: bool,
    ) -> Result<(Vec<f32>, DevMoeTaps)> {
        self.write_f32(self.slots.x, x)?;
        let taps = self.moe_record(weights, cfg, layer, collect_taps)?;
        self.flush()?;
        let y = self.read_f32(self.slots.acc, cfg.hidden_size)?;
        Ok((y, taps))
    }

    /// [`Self::moe`] minus the trailing flush/read: `x` is already staged and
    /// the result stays in the `acc` slot. Fence-free on the hot path: the
    /// router's ids stay on device and the expert GEMV reads its
    /// `weight_scale_2` from the resident id-indexed table, so nothing here
    /// submits — `collect_taps` alone pays for read-backs.
    pub fn moe_record(
        &mut self,
        weights: &Qwen4Weights<'_, '_>,
        cfg: &Qwen4ExpConfig,
        layer: usize,
        collect_taps: bool,
    ) -> Result<DevMoeTaps> {
        let h = cfg.hidden_size;
        let top_k = cfg.num_experts_per_tok;
        let inter = cfg.moe_intermediate_size;
        let _s = prof::stage("dev.moe");
        prof::phase("dev.moe.router");
        let s = self.slots;

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
        // The per-layer ids fence used to live here: flush + read logits/ids/
        // wts back so the host could gather a slot-ordered `weight_scale_2`
        // list. The scale table is device-resident now and the expert GEMV
        // indexes it by expert id — the very ids buffer the router just
        // wrote — so the hot path records straight through: 48 fences per
        // token became zero. The reads survive only as harness taps.
        let (logits, ids, route_weights) = if collect_taps {
            self.flush()?;
            (
                self.read_f32(s.logits, cfg.num_experts)?,
                self.read_i32(s.ids, top_k)?,
                self.read_f32(s.wts, top_k)?,
            )
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };
        self.ensure_scale0_rows(weights, layer)?;
        // The deleted flush was ALSO the top-k → expert-GEMV sync: without a
        // submission boundary the GEMVs race the router's ids/wts writes, so
        // the ordering must now be said out loud.
        self.barrier();

        prof::phase("dev.moe.gate_up");
        for (proj, dst_off) in [(ExpertProj::Gate, s.gate), (ExpertProj::Up, s.up)] {
            self.gemv_id_nvfp4(weights, layer, proj, top_k, h, inter, s.x, 1, dst_off)?;
        }
        self.barrier();
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
        self.barrier();
        prof::phase("dev.moe.down_accum");
        // down: each expert slot reads ITS OWN activation row (ne11 = top_k).
        self.gemv_id_nvfp4(
            weights,
            layer,
            ExpertProj::Down,
            top_k,
            inter,
            h,
            s.gate,
            top_k,
            s.down,
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
        // `routed` is the accumulator BEFORE the shared expert; capturing it
        // costs a flush, so only the harness pays for it.
        let routed = if collect_taps {
            self.flush()?;
            self.read_f32(s.acc, h)?
        } else {
            Vec::new()
        };

        prof::phase("dev.moe.shared");
        self.barrier();
        let shared_on_device = self
            .record_shared_expert(weights, cfg, layer)
            .context("device shared expert")?;
        Ok(DevMoeTaps {
            logits,
            ids,
            weights: route_weights,
            routed,
            shared_on_device,
        })
    }

    /// Byte offset of `(layer, proj)`'s row block inside the resident
    /// `weight_scale_2` table (the `scale0` slot).
    fn scale0_rows_off(layer: usize, proj: ExpertProj) -> u64 {
        let pi = match proj {
            ExpertProj::Gate => 0u64,
            ExpertProj::Up => 1,
            ExpertProj::Down => 2,
        };
        (layer as u64 * 3 + pi) * SCALE0_EXPERTS * 4
    }

    /// The SCALE0 binding for one expert stack: its id-indexed
    /// `weight_scale_2` rows in the resident table.
    fn scale0_rows_bind(&self, layer: usize, proj: ExpertProj) -> Bind<'static> {
        Bind::A(
            self.slots.scale0 + Self::scale0_rows_off(layer, proj),
            SCALE0_EXPERTS * 4,
        )
    }

    /// Seed `layer`'s three `weight_scale_2` row blocks into the resident
    /// table on first touch. First-touch is what makes the host write
    /// race-free without a fence: no recorded-but-unsubmitted dispatch can be
    /// reading rows that have never been bound.
    fn ensure_scale0_rows(&mut self, weights: &Qwen4Weights<'_, '_>, layer: usize) -> Result<()> {
        ensure!(
            (layer as u64) < SCALE0_LAYERS,
            "layer {layer} outside the {SCALE0_LAYERS}-layer scale0 table"
        );
        if self.scale0_seeded & (1 << layer) != 0 {
            return Ok(());
        }
        let mut rows = vec![0.0f32; (3 * SCALE0_EXPERTS) as usize];
        for (i, proj) in [ExpertProj::Gate, ExpertProj::Up, ExpertProj::Down]
            .into_iter()
            .enumerate()
        {
            let ws2 = &weights.expert_stack(layer, proj)?.weight_scale_2;
            ensure!(
                ws2.len() <= SCALE0_EXPERTS as usize,
                "{} experts overflow the {SCALE0_EXPERTS}-row scale0 table",
                ws2.len()
            );
            rows[i * SCALE0_EXPERTS as usize..][..ws2.len()].copy_from_slice(ws2);
        }
        self.write_f32(
            self.slots.scale0 + Self::scale0_rows_off(layer, ExpertProj::Gate),
            &rows,
        )?;
        self.scale0_seeded |= 1 << layer;
        Ok(())
    }

    /// One fused NVFP4 expert GEMV over the router's device-resident ids,
    /// with the stack's id-indexed `weight_scale_2` table on SCALE0 and
    /// `ne11` activation rows at `b_off` (1 = shared across slots, `top_k` =
    /// one row per slot).
    #[expect(clippy::too_many_arguments, reason = "a dispatch is this wide")]
    fn gemv_id_nvfp4(
        &mut self,
        weights: &Qwen4Weights<'_, '_>,
        layer: usize,
        proj: ExpertProj,
        top_k: usize,
        ncols: usize,
        nrows: usize,
        b_off: u64,
        ne11: usize,
        dst_off: u64,
    ) -> Result<()> {
        let s = self.slots;
        let b = Bind::A(b_off, (ncols * ne11 * 4) as u64);
        let dst = Bind::A(dst_off, (top_k * nrows * 4) as u64);
        let scale0 = self.scale0_rows_bind(layer, proj);
        let ids_b = Bind::A(s.ids, (top_k * 4) as u64);
        self.gemv_id_nvfp4_binds(
            weights, layer, proj, top_k, ncols, nrows, b, ne11, dst, scale0, ids_b,
        )
    }

    /// [`Self::gemv_id_nvfp4`] with caller-supplied bindings, so the chunked
    /// prefill can route per-token activations / ids rows out of its own arena.
    #[expect(clippy::too_many_arguments, reason = "a dispatch is this wide")]
    fn gemv_id_nvfp4_binds(
        &mut self,
        weights: &Qwen4Weights<'_, '_>,
        layer: usize,
        proj: ExpertProj,
        top_k: usize,
        ncols: usize,
        nrows: usize,
        b: Bind<'_>,
        ne11: usize,
        dst: Bind<'_>,
        scale0: Bind<'_>,
        ids: Bind<'_>,
    ) -> Result<()> {
        let s = self.slots;
        let stack = weights.expert_stack(layer, proj)?;
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
                b,
                dst,
                scale0,
                Bind::A(s.dummy, 8),
                ids,
            ],
            [d.x, d.y, d.z],
        )
    }

    /// Record ONE grouped-MoE class: expert blocks of `cols` gathered
    /// activation rows each, through a `NUM_COLS = cols` `GemvIdNvfp4`
    /// pipeline — the y axis walks BLOCKS (expert-major) where the decode
    /// dispatch walks one token's slots. `weight_scale_2` rides the same
    /// `MAT_VEC_FUSION_SCALE0` seam as decode: `scale0` is the resident
    /// id-indexed `weight_scale_2` table, read through each block's expert
    /// id.
    ///
    /// The y extent comes from `indirect` when given (a device-computed
    /// `VkDispatchIndirectCommand` — the fence-free planner's block count) and
    /// from `n_blocks` otherwise. `n_blocks` always feeds the push words, and
    /// an UPPER BOUND is sufficient there: word 9 (`ne11`) only needs
    /// `expert_i0 % ne11 == expert_i0` for every dispatched block index, and
    /// words 8/11 are unread by the shader body (`gemv_id_params_grouped`
    /// documents this).
    #[expect(clippy::too_many_arguments, reason = "a dispatch is this wide")]
    fn gemv_id_nvfp4_grouped(
        &mut self,
        weights: &Qwen4Weights<'_, '_>,
        layer: usize,
        proj: ExpertProj,
        cols: usize,
        n_blocks: usize,
        indirect: Option<GroupedIndirect<'_>>,
        ncols: usize,
        nrows: usize,
        b: Bind<'_>,
        dst: Bind<'_>,
        scale0: Bind<'_>,
        ids: Bind<'_>,
    ) -> Result<()> {
        let s = self.slots;
        let stack = weights.expert_stack(layer, proj)?;
        ensure!(
            stack.tensor.ncols == ncols && stack.tensor.nrows == nrows,
            "grouped {proj:?} shape [{}, {}] vs dispatch [{ncols}, {nrows}]",
            stack.tensor.ncols,
            stack.tensor.nrows
        );
        let (sb, so, sl) = weights.binding(&stack.tensor)?;
        let push = gemv_id_params_grouped(
            ncols as u32,
            nrows as u32,
            cols as u32,
            n_blocks as u32,
            MAT_VEC_FUSION_SCALE0,
        )
        .to_le_bytes();
        let binds = [
            Bind::Ext(sb, so, sl),
            b,
            dst,
            scale0,
            Bind::A(s.dummy, 8),
            ids,
        ];
        match indirect {
            Some(args) => self.rec_indirect(
                Kernel::GemvIdNvfp4,
                &gemv_nvfp4_spec_cols(cols as u32),
                &push,
                &binds,
                args,
            ),
            None => {
                let d = gemv_id_dispatch(nrows as u32, n_blocks as u32);
                self.rec(
                    Kernel::GemvIdNvfp4,
                    &gemv_nvfp4_spec_cols(cols as u32),
                    &push,
                    &binds,
                    [d.x, d.y, d.z],
                )
            }
        }
    }

    /// RECORD the shared expert into the open batch (no flush), accumulating
    /// into the `acc` slot. Returns whether it was recorded. The scalar
    /// `shared_expert_gate` needs the F32 router GEMV (its sigmoid rides the
    /// kernel); the three projections take whatever format the tier holds.
    fn record_shared_expert(
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
        for (i, n) in names.iter().enumerate() {
            match weights.tensor(n) {
                Ok(t) if i > 0 || t.format == Qwen4DeviceFormat::F32 => tensors.push(*t),
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
        // gate / up, whatever format the tier holds.
        for (t, dst) in [(&tensors[1], s.gate), (&tensors[2], s.up)] {
            self.record_dense_at(weights, t, s.x, dst)?;
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
        self.record_dense_at(weights, &tensors[3], s.gate, s.down)?;
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
        Ok(true)
    }
}

/// What the device MoE hands back for parity.
pub struct DevMoeTaps {
    /// Router logits `[num_experts]`. Empty unless `collect_taps` — the hot
    /// path leaves routing entirely on device.
    pub logits: Vec<f32>,
    /// Selected expert ids (slot order). Empty unless `collect_taps`.
    pub ids: Vec<i32>,
    /// Selected routing weights (renormalised on device). Empty unless
    /// `collect_taps`.
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
        {
            let pc = ple_config(cfg);
            ensure!(h.len() == pc.hc_hidden(), "ple hidden width");
        }
        self.write_f32(self.slots.h, h)?;
        self.ple_record(weights, cfg, layer, embeddings, ring_rows, false)?;
        self.finish_ple(cfg)
    }

    /// The recorded half of [`Self::ple`]: `h` is already in the `h` slot.
    /// With `add_into_h` the PLE output is accumulated into the residual ON
    /// DEVICE (`h += 1.0 * out`), which is what lets the staged loop keep the
    /// baton on the GPU. [`Self::finish_ple`] flushes and returns
    /// `(out, advanced ring)` for the host-canonical state.
    pub fn ple_record(
        &mut self,
        weights: &Qwen4Weights<'_, '_>,
        cfg: &Qwen4ExpConfig,
        layer: usize,
        embeddings: &[f32],
        ring_rows: &[f32],
        add_into_h: bool,
    ) -> Result<()> {
        // Upload mode clobbers the ring slot, so any resident ring dies here.
        self.ple_ring_layer = None;
        self.ple_record_inner(
            weights,
            cfg,
            layer,
            embeddings,
            Some(ring_rows),
            0,
            add_into_h,
        )
    }

    /// [`Self::ple_record`] against the RESIDENT ring: no upload, no
    /// read-back — the conv kernel advances the ring in place at the current
    /// `ring_pos`. Requires a prior [`Self::seed_ple_ring`] for `layer`.
    pub fn ple_record_resident(
        &mut self,
        weights: &Qwen4Weights<'_, '_>,
        cfg: &Qwen4ExpConfig,
        layer: usize,
        embeddings: &[f32],
        add_into_h: bool,
    ) -> Result<()> {
        ensure!(
            self.ple_ring_layer == Some(layer),
            "PLE ring not resident for layer {layer}"
        );
        let pos = self.ple_ring_pos;
        self.ple_record_inner(weights, cfg, layer, embeddings, None, pos, add_into_h)?;
        let len = ple_config(cfg).short_conv_state_len() as u32;
        self.ple_ring_pos = (pos + 1) % len;
        Ok(())
    }

    /// Seed the resident ring for `layer` from host oldest-first rows. At
    /// `ring_pos = 0` the kernel's layout matches the host's verbatim (the
    /// newest row, lag 1, sits at slot `state_len - 1`), so the upload is a
    /// straight copy.
    pub fn seed_ple_ring(&mut self, layer: usize, rows: &[f32]) -> Result<()> {
        self.write_f32(self.slots.ple_ring, rows)?;
        self.ple_ring_layer = Some(layer);
        self.ple_ring_pos = 0;
        Ok(())
    }

    /// Sync-out half of the resident-ring lifecycle: flush, read the ring
    /// back in host oldest-first order (oldest = the next write slot), and
    /// drop residency. `None` when nothing is resident. Host-canonical
    /// consumers (the fallback loop, batched prefill) call this before
    /// trusting `PleConvState` again.
    pub fn read_ple_ring(&mut self, cfg: &Qwen4ExpConfig) -> Result<Option<(usize, Vec<f32>)>> {
        let Some(layer) = self.ple_ring_layer.take() else {
            return Ok(None);
        };
        let pc = ple_config(cfg);
        let hh = pc.hc_hidden();
        let state_len = pc.short_conv_state_len();
        self.flush()?;
        let dev_ring = self.read_f32(self.slots.ple_ring, state_len * hh)?;
        let mut rows = vec![0.0f32; dev_ring.len()];
        let p = self.ple_ring_pos as usize;
        for t in 0..state_len {
            let src = (p + t) % state_len;
            rows[t * hh..(t + 1) * hh].copy_from_slice(&dev_ring[src * hh..(src + 1) * hh]);
        }
        Ok(Some((layer, rows)))
    }

    /// Drop ring residency WITHOUT syncing — for sequence resets, where the
    /// host state was just zeroed and the device rows are garbage history.
    pub fn invalidate_ple_ring(&mut self) {
        self.ple_ring_layer = None;
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the two public modes share this body"
    )]
    fn ple_record_inner(
        &mut self,
        weights: &Qwen4Weights<'_, '_>,
        cfg: &Qwen4ExpConfig,
        layer: usize,
        embeddings: &[f32],
        ring_upload: Option<&[f32]>,
        ring_pos: u32,
        add_into_h: bool,
    ) -> Result<()> {
        let _s = prof::stage("dev.ple");
        let pc = ple_config(cfg);
        let hh = pc.hc_hidden();
        ensure!(embeddings.len() == pc.ple_embed_dim, "ple embeddings width");
        let name = |suffix: &str| layer_tensor_name(layer, suffix);
        let kp = *weights.tensor(&name("ple.key_proj.weight"))?;
        let vp = *weights.tensor(&name("ple.value_proj.weight"))?;
        let s = self.slots;
        self.write_f32(s.ple_emb, embeddings)?;
        if let Some(ring_rows) = ring_upload {
            ensure!(
                ring_rows.len() == pc.short_conv_state_len() * hh,
                "ple ring rows"
            );
            self.write_f32(s.ple_ring, ring_rows)?;
        }
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
            ring_pos,
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
        if add_into_h {
            self.barrier();
            self.write_f32(s.one, &[1.0])?;
            let push = qwen36_moe_weighted_accum_params(hh as u32, 1, false).to_le_bytes();
            let d = qwen36_moe_weighted_accum_dispatch(hh as u32);
            self.rec(
                Kernel::Qwen36MoeWeightedAccum,
                Kernel::Qwen36MoeWeightedAccum.specialization_u32(),
                &push,
                &[
                    Bind::A(s.ple_o, (hh * 4) as u64),
                    Bind::A(s.one, 4),
                    Bind::A(s.h, (hh * 4) as u64),
                ],
                [d.x, d.y, d.z],
            )?;
        }
        Ok(())
    }

    /// Flush and hand back `(out, advanced ring)` — the ring rotated to the
    /// host's oldest-first layout (see the RING note inside).
    pub fn finish_ple(&mut self, cfg: &Qwen4ExpConfig) -> Result<(Vec<f32>, Vec<f32>)> {
        let pc = ple_config(cfg);
        let hh = pc.hc_hidden();
        let s = self.slots;
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
                    .is_ok_and(|t| t.format != Qwen4DeviceFormat::Nvfp4)
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
        self.write_f32(self.slots.x, x)?;
        self.write_rope_pos(pos)?;
        self.full_attention_record(weights, cfg, layer, pos)?;
        self.flush()?;
        let (hd, nq, q_dim) = (
            cfg.head_dim,
            cfg.num_attention_heads,
            cfg.num_attention_heads * cfg.head_dim,
        );
        let s = self.slots;
        Ok((
            self.read_f32(s.y, cfg.hidden_size)?,
            DevFullTaps {
                q_full: self.read_f32(s.qkv, nq * 2 * hd)?,
                q_roped: self.read_f32(s.q, q_dim)?,
                gated: self.read_f32(s.attn, q_dim)?,
            },
        ))
    }

    /// This token's RoPE position, staged once per token — every
    /// full-attention layer reads the same slot, so in the staged loop the
    /// write happens BEFORE any recording, never between recorded readers.
    pub fn write_rope_pos(&mut self, pos: usize) -> Result<()> {
        let pos_bytes = i64::try_from(pos).map(|_| (pos as i32).to_le_bytes())?;
        self.arena
            .copy_from_host_at(self.slots.pos, &pos_bytes)
            .map_err(|e| anyhow!("write rope pos: {e}"))
    }

    /// [`Self::full_attention`] minus the host round trip: `x` and the RoPE
    /// position are already staged, `y` stays in the `y` slot.
    pub fn full_attention_record(
        &mut self,
        weights: &Qwen4Weights<'_, '_>,
        cfg: &Qwen4ExpConfig,
        layer: usize,
        pos: usize,
    ) -> Result<()> {
        let _s = prof::stage("dev.full_attn");
        let hd = cfg.head_dim;
        let nq = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let group = nq / nkv;
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
        let s = self.slots;

        // q/k/v projections off the one staged activation, whatever format
        // the tier holds — the F32-only gate this replaced kept all twelve
        // full-attention layers on the HOST in hybrid mode (the tier is F16).
        for (t, dst) in [(&q_w, s.qkv), (&k_w, s.kbuf), (&v_w, s.vbuf)] {
            self.record_dense_at(weights, t, s.x, dst)?;
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
        // o-proj → the y slot, tier format.
        self.record_dense_at(weights, &o_w, s.attn, s.y)
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
    /// Device twin of `state` for the speculative-decode rollback: a verify
    /// chunk snapshots the WHOLE recurrent state into it with one recorded
    /// copy (~117 MB device-to-device, riding the chunk's own submit) and a
    /// rejected speculation copies it back. Device-side on purpose: reading
    /// the state to the host is the write-combined ~0.10 GB/s trap, over a
    /// second per snapshot. `None` until the first speculative cycle.
    snapshot: Option<DeviceBuffer<'ctx>>,
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
            snapshot: None,
            index,
        })
    }

    /// Allocate the rollback snapshot buffer (idempotent). Separate from
    /// construction so a plain decode never pays the second state-sized
    /// allocation.
    pub fn ensure_snapshot(&mut self, ctx: &'ctx VulkanContext) -> Result<()> {
        if self.snapshot.is_none() {
            let buf = DeviceBuffer::alloc_uma(ctx, self.state.len())
                .map_err(|e| anyhow!("resident state snapshot alloc: {e}"))?;
            self.snapshot = Some(buf);
        }
        Ok(())
    }

    /// Record `state -> snapshot` (every layer) into the open command buffer.
    /// Recorded BEFORE a verify chunk's dispatches, it captures the pre-chunk
    /// state in the same submit.
    pub fn record_state_save(&self, dev: &mut Qwen4Dev<'ctx>) -> Result<()> {
        let snap = self
            .snapshot
            .as_ref()
            .ok_or_else(|| anyhow!("state snapshot not allocated (ensure_snapshot)"))?;
        dev.record_copy(&self.state, 0, snap, 0, self.state.len() as u64)
    }

    /// Record `snapshot -> state`: the rollback of a rejected speculation.
    pub fn record_state_restore(&self, dev: &mut Qwen4Dev<'ctx>) -> Result<()> {
        let snap = self
            .snapshot
            .as_ref()
            .ok_or_else(|| anyhow!("state snapshot not allocated (ensure_snapshot)"))?;
        dev.record_copy(snap, 0, &self.state, 0, self.state.len() as u64)
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
        dev.write_f32(dev.slots.x, x)?;
        self.forward_record(dev, weights, cfg, layer)?;
        dev.flush()?;
        dev.read_f32(dev.slots.y, cfg.hidden_size)
    }

    /// [`Self::forward`] minus the host round trip: the block input is
    /// already in the `x` slot, `y` stays in the `y` slot, nothing flushes.
    pub fn forward_record(
        &self,
        dev: &mut Qwen4Dev<'ctx>,
        weights: &Qwen4Weights<'_, '_>,
        cfg: &Qwen4ExpConfig,
        layer: usize,
    ) -> Result<()> {
        let _s = prof::stage("dev.linear_attn");
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
        dev.record_dense_at(weights, &t_out, s.dense_x, s.y)
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
pub(crate) fn host_shared_expert(
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
    /// Prefetch-then-read n-gram gather over an OWNING second view of the
    /// checkpoint (the model itself only borrows `st`). `None` when the
    /// checkpoint was opened from an explicit file list, or the pool failed
    /// to build — the serial per-row fallback below still works, just 6-7x
    /// slower.
    ngram_pool: Option<crate::qwen4_ngram_gather::NgramGather>,
    /// Chunk arena for [`Self::forward_prompt`], built lazily on the first
    /// prompt (decode-only runs never pay its ~240 MB).
    prefill: Option<Qwen4Prefill<'ctx>>,
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
                // Dense-tier format A/B without a recompile: ARLE_QWEN4_DENSE
                // = bf16 (default, verbatim bytes) | q8 (load-time Q8_0 —
                // half the bytes at a measured 6.6-7.4e-3 vector-rel per
                // tensor). The harness's SubsetF32 arm below is unaffected.
                // Dense-tier format without a recompile. q4k = W4A16 Q4_K
                // on every width-qualified family (the 640-wide shared-expert
                // down rides Q8_0); q8 = W8A16 Q8_0 everywhere; default is
                // the verbatim BF16 tier until the quality probe flips it.
                let ucfg = match std::env::var("ARLE_QWEN4_DENSE").as_deref() {
                    Ok("q4k" | "q4") => Qwen4UploadConfig {
                        dense_format: Qwen4DeviceFormat::Q4K,
                        ..Qwen4UploadConfig::default()
                    },
                    Ok("q8" | "q8_0") => Qwen4UploadConfig {
                        dense_format: Qwen4DeviceFormat::Q8_0,
                        ..Qwen4UploadConfig::default()
                    },
                    Ok("bf16") => Qwen4UploadConfig::default(),
                    // Default IS q4k: teacher-forced probe put it 2/32
                    // positions behind near-lossless Q8 (84.4% vs 90.6%
                    // step-agreement, disagreements concentrated at
                    // razor-margin positions) at 55.5 vs 72.7 ms/token.
                    // bf16 stays one env var away.
                    Err(_) => Qwen4UploadConfig {
                        dense_format: Qwen4DeviceFormat::Q4K,
                        ..Qwen4UploadConfig::default()
                    },
                    Ok(other) => bail!("ARLE_QWEN4_DENSE={other}: bf16 | q4k | q8"),
                };
                // The MTP head rides the full residency by default (~2 GiB
                // at the Q4_K tier; the speculative-decode drafter).
                // ARLE_QWEN4_MTP=0 drops it for a plain decode.
                let mut scope = Qwen4UploadScope::full();
                if matches!(std::env::var("ARLE_QWEN4_MTP").as_deref(), Ok("0" | "off")) {
                    scope.mtp = false;
                }
                let plan = plan_qwen4_upload(st, &ucfg, &scope)?;
                let weights = upload_qwen4(ctx, st, &plan, &ucfg)?;
                // KV planes for every full-attention layer (~4 MB each at
                // ctx 2048) — without them full_attention_ready is false and
                // all twelve layers silently fall back to the host.
                let full_layers: Vec<usize> = cfg
                    .layer_types
                    .iter()
                    .enumerate()
                    .filter(|(_, t)| **t == Qwen4LayerType::FullAttention)
                    .map(|(l, _)| l)
                    .collect();
                let dev = Qwen4Dev::new(ctx, &cfg, &full_layers, cfg.max_context)?;
                let resident =
                    DevResidentLinAttn::new(ctx, &cfg, layers.iter().map(|(l, hl)| (*l, hl)))?;
                (Some(weights), Some(dev), Some(resident))
            }
            (Some(ctx), Qwen4ExpDeviceMode::SubsetF32(subset)) => {
                let mut scope = Qwen4UploadScope {
                    lm_head: false,
                    ..Qwen4UploadScope::layers(subset)
                };
                // The speculative harness opts the MTP head into a subset
                // load explicitly (F32/BF16 per ARLE_QWEN4_SUBSET_DENSE).
                if matches!(
                    std::env::var("ARLE_QWEN4_SUBSET_MTP").as_deref(),
                    Ok("1" | "on")
                ) {
                    scope.mtp = true;
                }
                // The dense tier defaults to F32 (the parity harness's
                // zero-slack residency: decode and prefill then record the
                // SAME GEMV dispatches). ARLE_QWEN4_SUBSET_DENSE=bf16 stages
                // it BF16 instead, which is the 20-second repro of the full
                // model's dense tier — the prefill's coopmat GEMM route only
                // exists for BF16/F16, so an F32-only gate cannot see it.
                let ucfg = Qwen4UploadConfig {
                    dense_format: match std::env::var("ARLE_QWEN4_SUBSET_DENSE").as_deref() {
                        Ok("bf16") => Qwen4DeviceFormat::Bf16,
                        _ => Qwen4DeviceFormat::F32,
                    },
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

        // The gather pool wants an owning Arc where this model only borrows;
        // a second open of the same directory shares the page cache. Losing
        // the pool is a slowdown, not an error — say so and fall back.
        let ngram_pool = if cfg.ple_layer_ids.is_empty() {
            None
        } else {
            st.root().and_then(|dir| {
                match crate::qwen4_ngram_gather::NgramGather::open_dir(
                    dir,
                    crate::qwen4_ngram_gather::DEFAULT_WORKERS,
                ) {
                    Ok(pool) => Some(pool),
                    Err(e) => {
                        log::warn!("n-gram gather pool unavailable, serial fallback: {e:#}");
                        None
                    }
                }
            })
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
            ngram_pool,
            prefill: None,
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
    /// from the FP8 table (host-resident; 16 rows of 160). The pool path
    /// prefetches all row ranges in one syscall before reading — the serial
    /// loop below re-faults per row and measured 6-7x slower at the decode
    /// pattern.
    fn gather_ple_embedding(&mut self, row_ids: &[i64]) -> Result<Vec<f32>> {
        if let Some(pool) = self.ngram_pool.as_mut() {
            let width = pool.table().row_width();
            let mut out = vec![0.0f32; row_ids.len() * width];
            pool.gather(row_ids, &mut out)?;
            return Ok(out);
        }
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
            if let Some(d) = self.dev.as_mut() {
                // Fresh sequence: the resident PLE ring is garbage history.
                d.invalidate_ple_ring();
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
            lm_head,
            ..
        } = self;

        // ── the staged loop: the baton (h/x/y) never leaves the device ────
        // Eligible when EVERY stage of EVERY layer runs on device; then the
        // only fences per token are the per-layer MoE ids read-back, the PLE
        // ring hand-back, and the final logits. Everything else — 48 layers of
        // hyper-connections, attention and expert tails — is recorded
        // back-to-back into as few submits as the ids fence allows.
        let staged = match (dev.as_ref(), weights.as_ref(), resident_linear.as_ref()) {
            (Some(d), Some(w), Some(rl)) => {
                (0..cfg.num_hidden_layers).all(|l| {
                    w.hyper_connection(Some(l), HcSite::Attn).is_ok()
                        && w.hyper_connection(Some(l), HcSite::Mlp).is_ok()
                        && w.expert_stack(l, ExpertProj::Gate).is_ok()
                        && match cfg.layer_types[l] {
                            Qwen4LayerType::LinearAttention => rl.covers(l),
                            Qwen4LayerType::FullAttention => d.full_attention_ready(w, l),
                        }
                }) && w.hyper_connection(None, HcSite::Mixer).is_ok()
                    && cfg.ple_layer_ids.iter().all(|&l| {
                        w.tensor(&layer_tensor_name(l, "ple.key_proj.weight"))
                            .is_ok()
                    })
                    && w.tensor(&lm_head.name).is_ok()
            }
            _ => false,
        };
        if staged {
            let d = dev.as_mut().expect("staged");
            let w = weights.as_ref().expect("staged");
            let rl = resident_linear.as_ref().expect("staged");
            d.write_f32(d.slots.h, &h)?;
            d.write_rope_pos(start_pos)?;
            for layer in 0..cfg.num_hidden_layers {
                let hl = layers
                    .get(&layer)
                    .ok_or_else(|| anyhow!("host layer {layer} not loaded"))?;
                if hl.ple.is_some() {
                    if d.ple_ring_layer != Some(layer) {
                        let ring = state
                            .ple_conv
                            .get(&layer)
                            .ok_or_else(|| anyhow!("no PLE conv state for layer {layer}"))?;
                        d.seed_ple_ring(layer, ring.rows())?;
                    }
                    d.ple_record_resident(w, cfg, layer, &ple_emb, true)?;
                    // The retired finish_ple flush doubled as the PLE -> hc_pre
                    // sync (same trap as the MoE ids fence); the ordering is
                    // explicit now. Host `PleConvState` goes stale here — the
                    // fallback loop and batched prefill sync out before use.
                    d.barrier();
                }
                d.hc_pre_record(w, hc, Some(layer), HcSite::Attn)?;
                d.barrier();
                match hl.kind {
                    Qwen4LayerType::LinearAttention => {
                        rl.forward_record(d, w, cfg, layer)?;
                    }
                    Qwen4LayerType::FullAttention => {
                        d.full_attention_record(w, cfg, layer, start_pos)?;
                    }
                }
                d.barrier();
                let y_off = d.slots.y;
                d.hc_combine_record(w, hc, Some(layer), HcSite::Attn, y_off)?;
                d.barrier();
                d.hc_pre_record(w, hc, Some(layer), HcSite::Mlp)?;
                d.barrier();
                d.moe_record(w, cfg, layer, false)?;
                d.barrier();
                let acc_off = d.slots.acc;
                d.hc_combine_record(w, hc, Some(layer), HcSite::Mlp, acc_off)?;
                d.barrier();
                // Depth-2 pipelining: hand the GPU a 12-layer batch while the
                // host records the next. The trailing barrier above orders
                // the next batch's first dispatch against this one on the
                // in-order queue, so the split is timing, not semantics.
                if layer % 12 == 11 {
                    d.flush_async()?;
                }
            }
            d.hc_pre_record(w, hc, None, HcSite::Mixer)?;
            d.barrier();
            let lm_span = prof::stage("host.lm_head");
            let lm = *w.tensor(&lm_head.name)?;
            let (x_off, y_off) = (d.slots.x, d.slots.dense_y);
            d.record_dense_at(w, &lm, x_off, y_off)?;
            d.flush()?;
            let logits = d.read_f32(d.slots.dense_y, cfg.vocab_size)?;
            drop(lm_span);
            state.seq_len += 1;
            return Ok(logits);
        }

        // A staged token may have left the PLE ring device-resident; this
        // lane trusts the host `PleConvState`, so sync it out first.
        if let Some(d) = dev.as_mut() {
            if let Some((l, rows)) = d.read_ple_ring(cfg)? {
                if let Some(ring) = state.ple_conv.get_mut(&l) {
                    ring.rows_mut().copy_from_slice(&rows);
                }
            }
        }
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
                    d.moe(w, cfg, layer, &x, false)?
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

        let logits = self.mixer_lm_head(&h)?;
        self.state.seq_len += 1;
        Ok(logits)
    }

    /// The decode TAIL: stream mixer (use_combine = false) collapses
    /// 10240 → 2560 — there is NO other final norm — then `lm_head`, each
    /// through the device when resident and the host transcription otherwise.
    /// One helper on purpose: decode, the prefill's last-token tail and the
    /// batched verify's per-position fallback all route through it, so the
    /// three cannot pick different (and subtly different-valued) routes for
    /// the same `h`. On-device `lm_head` is worth 52 ms of the 899 ms token
    /// measured before that lane existed: 1212.5 MiB in ONE projection.
    fn mixer_lm_head(&mut self, h: &[f32]) -> Result<Vec<f32>> {
        let mixer_dev = self
            .weights
            .as_ref()
            .is_some_and(|w| self.dev.is_some() && w.hyper_connection(None, HcSite::Mixer).is_ok());
        let x = if mixer_dev {
            let d = self.dev.as_mut().expect("mixer_dev checked");
            let w = self.weights.as_ref().expect("mixer_dev checked");
            d.hc_pre(w, &self.hc, None, HcSite::Mixer, h)?
        } else {
            let _s = prof::stage("host.hc_pre");
            qwen4_hc::gated_residual(&self.hc, &self.mixer, h)?.block_input
        };
        let _lm_span = prof::stage("host.lm_head");
        match self.dev.as_mut().zip(self.weights.as_ref()) {
            Some((d, w)) => DenseGemv::new(d, w).matvec(&self.lm_head, &x),
            None => Ok(self.lm_head.matvec(&x)),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Chunked prefill: `forward_prompt`.
//
// The per-token loop above re-reads every weight once per PROMPT token, so a
// 2048-token prompt costs 2048 decode steps (~3 minutes at 85 ms/token). This
// section processes the prompt in chunks of `T` tokens: dense projections
// become ONE coopmat GEMM per weight (`MmCmBf16` / `MmCmF16` — the same
// weight read amortized over `T` rows), the linear-attention conv + gated-delta
// recurrence and the PLE gate/conv run their existing `seq_len = T` kernel
// modes against the SAME device-resident state decode uses, and full attention
// is one causal-masked flash dispatch per layer. The NVFP4 expert tails are
// regrouped EXPERT-major ON DEVICE ([`MoeGroupPlan`] is the host oracle; the
// `Qwen4MoePlan*` kernels build the same plan without the ids ever reaching
// the host), so an active expert's rows stream once per projection instead of
// once per choosing token. Stages with no batched kernel — the 97
// hyper-connection sites and the MoE router/top-k — still record per token
// WITHIN the chunk.
//
// The fence structure is the point: a whole chunk records into ONE submit
// with NO intra-chunk read-back — the per-(layer,chunk) ids fence that used
// to sit in the MoE block held 7.46 s of the 9.27 s chunk-256 wall (80%)
// before the device planner replaced it. Only the end-of-chunk flush waits.
//
// Equivalence contract: prefill-then-decode must EQUAL the per-token loop —
// same logits, same recurrent state (GDN S, conv rings, PLE ring, KV rows).
// `tests/qwen4_prefill.rs` pins that on real weights; every stage here reuses
// decode's kernels (only the `seq_len`/grid changes), so the two paths cannot
// drift semantically without that test failing.

/// Tokens per prefill chunk. `ARLE_QWEN4_PREFILL_CHUNK` overrides the default
/// (256) so the width can be swept against a real load without a rebuild.
#[must_use]
pub fn qwen4_prefill_chunk_tokens() -> usize {
    std::env::var("ARLE_QWEN4_PREFILL_CHUNK")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&t| (1..=1024).contains(&t))
        .unwrap_or(256)
}

/// `NUM_COLS` cap for the chunked DENSE projections (the batched-verify /
/// prefill weight-read amortization). 8 is the same measured crossover as
/// [`PF_MOE_COLS_CAP`], and the two caps are deliberately the same number for
/// the same part. `ARLE_QWEN4_GEMV_COLS` overrides for a matched A/B; `1`
/// disables the batching (per-token GEMV loop, the pre-cols behavior).
#[must_use]
pub fn qwen4_gemv_cols_cap() -> usize {
    std::env::var("ARLE_QWEN4_GEMV_COLS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&t| (1..=8).contains(&t))
        .unwrap_or(8)
}

/// f16 bit patterns for the causal mask (the flash kernel ADDS the mask to the
/// pre-softmax score, so `0` = attend and `-inf` = blocked).
const PF_F16_ZERO: u16 = 0x0000;
const PF_F16_NEG_INF: u16 = 0xFC00;

/// Padded per-token strides (in elements) for rows that are bound PER TOKEN:
/// a descriptor offset must satisfy `minStorageBufferOffsetAlignment`, and 64
/// f32 = 256 B clears every device. Rows whose natural width is already a
/// 64-multiple (`hidden`, `hc_hidden`, ...) stay packed.
const PF_AB_PAD: usize = 64;
const PF_IDS_PAD: usize = 64;
/// Alignment quantum (in 4-byte elements) for the grouped-MoE class lists:
/// each class's ids/scales list starts on a 64-element = 256-B boundary so the
/// per-class binds clear `minStorageBufferOffsetAlignment` on every device.
const PF_LIST_ALIGN: usize = 64;
/// Rows per grouped-MoE GEMV block = the `NUM_COLS` cap. 8 is the measured
/// batched-GEMV crossover on this part (`vulkan-kernels/tests/`
/// `device_batched_crossover.rs`): weight-read amortization is near-free
/// through 8 columns, and past it the win flattens while registers fill. An
/// expert routed more rows gets `ceil(rows / 8)` blocks.
const PF_MOE_COLS_CAP: usize = 8;

/// FIXED per-class regions for the grouped-MoE layout, a function of
/// `(pairs, n_experts)` ALONE — no router ids involved. This is the load-
/// bearing trick that lets the group planning move onto the GPU without
/// touching the grouped GEMV: descriptor offsets must be known at RECORD
/// time, so instead of packing the classes densely (offsets = functions of
/// the device-only block counts, the old layout), every class `w` (block
/// width, `1..=`[`PF_MOE_COLS_CAP`]) owns a worst-case region and only its
/// BLOCK COUNT stays device-side, consumed as a `vkCmdDispatchIndirect` y
/// extent. Capacity rows no block claims are simply never written or read —
/// the gather is a write-side scatter over the live pairs
/// ([`Kernel::Qwen4BlockScatter`]), not a read-side permutation over the
/// capacity.
///
/// Worst cases per class: the full class (`w = CAP`) holds at most
/// `pairs / CAP` blocks; a remainder class `w < CAP` gets at most ONE block
/// per expert and each block consumes `w` pairs, so
/// `min(n_experts, pairs / w)` blocks. Summed at the default chunk (256
/// tokens x top-10 over 512 experts) the capacity is ~6x the live pairs —
/// ~330 MB more HOST-CACHED arena (not device-local heap), traded for
/// deleting the per-(layer,chunk) ids fence that held 80% of the chunk-256
/// prefill wall.
///
/// Classes are laid WIDEST-FIRST (class `CAP` at base 0), matching the old
/// dense layout's class order; the id-list bases are [`PF_LIST_ALIGN`]-
/// aligned so per-class binds clear `minStorageBufferOffsetAlignment`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoePlanLayout {
    /// Block capacity of class `w` at index `w - 1`.
    cap_blocks: [usize; PF_MOE_COLS_CAP],
    /// First gathered-pair row of class `w`'s region at index `w - 1`.
    pair_base: [usize; PF_MOE_COLS_CAP],
    /// Element offset of class `w`'s expert-id list at index `w - 1`.
    list_base: [usize; PF_MOE_COLS_CAP],
    /// Total gathered-pair rows across all class regions.
    pub pair_capacity: usize,
    /// Total expert-id list elements across all class regions.
    pub list_capacity: usize,
}

impl MoePlanLayout {
    #[must_use]
    pub fn new(pairs: usize, n_experts: usize) -> Self {
        let mut cap_blocks = [0usize; PF_MOE_COLS_CAP];
        let mut pair_base = [0usize; PF_MOE_COLS_CAP];
        let mut list_base = [0usize; PF_MOE_COLS_CAP];
        let mut pair_at = 0usize;
        let mut list_at = 0usize;
        for w in (1..=PF_MOE_COLS_CAP).rev() {
            let cap = if w == PF_MOE_COLS_CAP {
                pairs / PF_MOE_COLS_CAP
            } else {
                n_experts.min(pairs / w)
            };
            cap_blocks[w - 1] = cap;
            pair_base[w - 1] = pair_at;
            list_base[w - 1] = list_at;
            pair_at += w * cap;
            list_at = (list_at + cap).next_multiple_of(PF_LIST_ALIGN);
        }
        Self {
            cap_blocks,
            pair_base,
            list_base,
            pair_capacity: pair_at,
            list_capacity: list_at,
        }
    }

    /// Block capacity of class `w` (`1..=`[`PF_MOE_COLS_CAP`]).
    #[must_use]
    pub fn cap_blocks(&self, w: usize) -> usize {
        self.cap_blocks[w - 1]
    }

    /// First gathered-pair row of class `w`'s region.
    #[must_use]
    pub fn pair_base(&self, w: usize) -> usize {
        self.pair_base[w - 1]
    }

    /// Element offset of class `w`'s expert-id list.
    #[must_use]
    pub fn list_base(&self, w: usize) -> usize {
        self.list_base[w - 1]
    }

    /// The per-class bases as the emit kernel's push words (`u32`, class `w`
    /// at index `w - 1`).
    fn push_bases(&self) -> Result<([u32; 8], [u32; 8])> {
        let mut pair = [0u32; 8];
        let mut list = [0u32; 8];
        for w in 1..=PF_MOE_COLS_CAP {
            pair[w - 1] = u32::try_from(self.pair_base[w - 1])?;
            list[w - 1] = u32::try_from(self.list_base[w - 1])?;
        }
        Ok((pair, list))
    }
}

/// The causal mask for one chunk: `[t][kv_len]` f16, row `r` (absolute
/// position `start_pos + r`) attends keys `0..=start_pos + r`.
fn pf_causal_mask(t: usize, kv_len: usize, start_pos: usize) -> Vec<u8> {
    let mut mask = Vec::with_capacity(t * kv_len * 2);
    for r in 0..t {
        let limit = start_pos + r;
        for c in 0..kv_len {
            let bits = if c <= limit {
                PF_F16_ZERO
            } else {
                PF_F16_NEG_INF
            };
            mask.extend_from_slice(&bits.to_le_bytes());
        }
    }
    mask
}

/// Chunk-wide HF<->GGUF permutation maps: one flat u32 buffer holding four
/// token-major regions, so a chunk's activations permute in ONE
/// `Qwen4BlockPerm` dispatch instead of `4 * T`.
///
/// Returns `(flat, [qkv_at, v_at, ab_at, vinv_at])` — element offsets of:
/// - `qkv`  `[T * conv_dim/vd]`: raw qkv `[T][conv_dim]` (HF) -> GGUF slots;
/// - `v`    `[T * nv]`: a packed `[T][nv*vd]` V-head vector (`z`) -> slots;
/// - `ab`   `[T * nv]`: the [`PF_AB_PAD`]-strided per-token `a`/`b` rows ->
///   PACKED `[T][nv]` slot order (compaction and permutation in one pass — the
///   gated-delta kernel wants token stride `nv`, but a GEMV cannot write a
///   192-byte-aligned row);
/// - `vinv` `[T * nv]`: slot order back to HF for `out_proj`.
fn pf_chunk_maps(cfg: &Qwen4ExpConfig, max_tokens: usize) -> (Vec<u32>, [usize; 4]) {
    let nk = cfg.linear_num_key_heads;
    let nv = cfg.linear_num_value_heads;
    let kd = cfg.linear_key_head_dim;
    let vd = cfg.linear_value_head_dim;
    let qk_blocks = 2 * nk * kd / vd;
    let qkv_blocks = qk_blocks + nv;

    let mut flat = Vec::new();
    let mut offs = [0usize; 4];
    let region = |flat: &mut Vec<u32>, at: &mut usize| {
        // 64-element alignment keeps every region's byte offset 256-B aligned.
        while !flat.len().is_multiple_of(64) {
            flat.push(0);
        }
        *at = flat.len();
    };

    region(&mut flat, &mut offs[0]);
    for t in 0..max_tokens {
        for b in 0..qkv_blocks {
            let src = if b < qk_blocks {
                b
            } else {
                qk_blocks + v_slot_perm(nk, nv, b - qk_blocks)
            };
            flat.push((t * qkv_blocks + src) as u32);
        }
    }
    region(&mut flat, &mut offs[1]);
    for t in 0..max_tokens {
        for slot in 0..nv {
            flat.push((t * nv + v_slot_perm(nk, nv, slot)) as u32);
        }
    }
    region(&mut flat, &mut offs[2]);
    for t in 0..max_tokens {
        for slot in 0..nv {
            flat.push((t * PF_AB_PAD + v_slot_perm(nk, nv, slot)) as u32);
        }
    }
    region(&mut flat, &mut offs[3]);
    for t in 0..max_tokens {
        // dst block at HF head `perm(slot)` reads slot-ordered block `slot`.
        let base = flat.len();
        flat.resize(base + nv, 0);
        for slot in 0..nv {
            flat[base + v_slot_perm(nk, nv, slot)] = (t * nv + slot) as u32;
        }
    }
    (flat, offs)
}

/// One chunk's `t x top_k` routed `(token, slot)` pairs regrouped
/// EXPERT-major over the FIXED class regions of a [`MoePlanLayout`], so each
/// active expert's NVFP4 rows stream ONCE per projection instead of once per
/// token that chose it (the measured ~4.7x expert-byte cut of the grouped
/// round; see the layout's doc for why the regions are fixed).
///
/// This host planner is the ORACLE for the device planner
/// ([`Kernel::Qwen4MoePlanCount`] -> [`Kernel::Qwen4MoePlanScan`] ->
/// [`Kernel::Qwen4MoePlanEmit`]) and the `ARLE_QWEN4_PREFILL_HOST_PLAN=1`
/// fallback: `tests/qwen4_moe_plan.rs` asserts ELEMENT-FOR-ELEMENT equality
/// of the two on randomized and real router ids. Both pin the same
/// deterministic order — experts ascending, and WITHIN an expert the pairs in
/// token-major arrival order (token ascending, slot ascending), so an
/// expert's full blocks come out in chunk order and its remainder block last.
/// Bit-exactness of the VALUES never depended on the grouping (each pair's
/// dot product is the same kernel arithmetic at any `NUM_COLS` lane); the
/// determinism is what makes a failure reproduce.
///
/// DECODE STAYS UNTOUCHED — the single-token loop keeps its fused
/// `gemv_id_nvfp4` path.
#[derive(Debug)]
pub struct MoeGroupPlan {
    /// The fixed region geometry this plan was laid out in.
    pub layout: MoePlanLayout,
    /// Blocks of class `w` at index `w - 1` (the dispatch y extents).
    pub n_blocks: [usize; PF_MOE_COLS_CAP],
    /// `Qwen4BlockScatter` map (block = hidden): pair `p`'s row lands at
    /// gathered row `scatter[p]`; the return permutation reads the same map
    /// through `Qwen4BlockPerm` (`[tok][slot]` row `p` pulls gathered row
    /// `scatter[p]`), restoring token order for the untouched
    /// weighted-accumulate.
    pub scatter: Vec<u32>,
    /// Expert-id lists over the layout's full `list_capacity`; class `w`'s
    /// live entries at `layout.list_base(w) .. + n_blocks[w-1]`, everything
    /// else 0 (never read: no dispatch walks past its class's block count).
    pub ids: Vec<i32>,
}

/// Build the [`MoeGroupPlan`] for one chunk's routed ids (`raw_ids` is the
/// device's `[t][ids_stride]` i32 rows, first `top_k` entries live). Errors
/// on an out-of-range expert id rather than binding garbage.
pub fn plan_moe_groups(
    raw_ids: &[i32],
    t: usize,
    ids_stride: usize,
    top_k: usize,
    n_experts: usize,
) -> Result<MoeGroupPlan> {
    ensure!(
        t >= 1 && top_k >= 1,
        "empty grouping ({t} tokens, top-{top_k})"
    );
    ensure!(
        raw_ids.len() >= t * ids_stride && top_k <= ids_stride,
        "ids rows under-span the chunk"
    );
    let pairs = t * top_k;
    let layout = MoePlanLayout::new(pairs, n_experts);
    let cap = PF_MOE_COLS_CAP;

    // Pass 1 (the device count kernel's twin): per-expert pair counts.
    let mut counts = vec![0usize; n_experts];
    let pair_expert = |p: usize| raw_ids[(p / top_k) * ids_stride + (p % top_k)];
    for p in 0..pairs {
        let id = pair_expert(p);
        ensure!(
            usize::try_from(id).is_ok_and(|i| i < n_experts),
            "routed expert id {id} is outside the {n_experts}-expert stack"
        );
        counts[id as usize] += 1;
    }

    // Pass 2 (the scan kernel's twin): expert-ascending exclusive prefixes of
    // block indices, plus the per-class totals.
    let mut fullb = vec![0usize; n_experts];
    let mut remb = vec![0usize; cap * n_experts];
    let mut run = [0usize; PF_MOE_COLS_CAP + 1];
    for e in 0..n_experts {
        fullb[e] = run[cap];
        for w in 1..cap {
            remb[w * n_experts + e] = run[w];
        }
        run[cap] += counts[e] / cap;
        let rem = counts[e] % cap;
        if rem != 0 {
            run[rem] += 1;
        }
    }
    let mut n_blocks = [0usize; PF_MOE_COLS_CAP];
    for w in 1..=cap {
        n_blocks[w - 1] = run[w];
        ensure!(
            run[w] <= layout.cap_blocks(w),
            "class {w} overflows its fixed region ({} > {})",
            run[w],
            layout.cap_blocks(w)
        );
    }

    // Pass 3 (the emit kernel's twin): closed-form scatter rows in token-major
    // arrival order (rank = pairs seen so far for that expert), and the
    // class-major id lists.
    let mut plan = MoeGroupPlan {
        layout,
        n_blocks,
        scatter: vec![0u32; pairs],
        ids: vec![0i32; layout.list_capacity],
    };
    let mut next_rank = vec![0usize; n_experts];
    for p in 0..pairs {
        let e = pair_expert(p) as usize;
        let r = next_rank[e];
        next_rank[e] += 1;
        let nfull = counts[e] / cap;
        let (w, blk, pos) = if r < nfull * cap {
            (cap, fullb[e] + r / cap, r % cap)
        } else {
            let w = counts[e] % cap;
            (w, remb[w * n_experts + e], r - nfull * cap)
        };
        plan.scatter[p] = u32::try_from(layout.pair_base(w) + blk * w + pos)?;
    }
    for e in 0..n_experts {
        let nfull = counts[e] / cap;
        for k in 0..nfull {
            plan.ids[layout.list_base(cap) + fullb[e] + k] = e as i32;
        }
        let rem = counts[e] % cap;
        if rem != 0 {
            plan.ids[layout.list_base(rem) + remb[rem * n_experts + e]] = e as i32;
        }
    }
    Ok(plan)
}

/// Byte offsets of the chunk arena's regions. Every base is 256-B aligned;
/// rows inside a region are addressed `base + token * stride * 4` with strides
/// chosen so per-token binds stay aligned (see [`PF_AB_PAD`] and friends).
#[derive(Debug, Clone, Copy)]
struct PfSlots {
    /// Hyper residual `[T][hc_hidden]`.
    h: u64,
    /// Grouped-norm output `[T][hc_hidden]`.
    hn: u64,
    /// `mix_down` bottleneck `[T][u_stride]`.
    u: u64,
    /// Block input `[T][hidden]`.
    x: u64,
    /// Attention block output `[T][hidden]`.
    y: u64,
    /// MoE accumulator `[T][hidden]`.
    acc: u64,
    /// `in_proj_a` rows, [`PF_AB_PAD`]-strided.
    a_raw: u64,
    /// `in_proj_b` rows, [`PF_AB_PAD`]-strided.
    b_raw: u64,
    /// Slot-ordered PACKED `a` `[T][nv]`.
    a2: u64,
    /// Slot-ordered PACKED `b` `[T][nv]`.
    b2: u64,
    /// Router top-k ids, [`PF_IDS_PAD`]-strided i32 rows.
    ids: u64,
    /// Router top-k weights, [`PF_IDS_PAD`]-strided.
    wts: u64,
    /// Shared-expert scalar gates, [`PF_IDS_PAD`]-strided.
    shg: u64,
    /// Router logits `[T][num_experts]`.
    lgt: u64,
    /// Routed gate activations, EXPERT-major over the [`MoePlanLayout`] class
    /// regions, `[pair_capacity][inter]` (live rows are the routed pairs;
    /// capacity gaps are never read).
    ge: u64,
    /// Routed up activations, expert-major `[pair_capacity][inter]`.
    gu: u64,
    /// Routed down outputs `[T][top_k * hidden]` — TOKEN order (the scatter's
    /// destination; the weighted-accumulate walks these rows unchanged).
    edown: u64,
    /// Gathered block inputs, expert-major `[pair_capacity][hidden]`
    /// (grouped-MoE GEMV B operand).
    xg: u64,
    /// Routed down outputs, expert-major `[pair_capacity][hidden]`,
    /// pre-scatter.
    edc: u64,
    /// Grouped-MoE scatter map `[T * top_k]` u32 ([`MoeGroupPlan::scatter`]) —
    /// device-written by [`Kernel::Qwen4MoePlanEmit`] (host-uploaded under
    /// `ARLE_QWEN4_PREFILL_HOST_PLAN=1`). Serves BOTH permutations: the
    /// write-side gather and the read-side return scatter.
    smap: u64,
    /// Grouped-MoE class-major expert-id lists ([`MoeGroupPlan::ids`]),
    /// [`MoePlanLayout::list_capacity`] i32s.
    eids: u64,
    /// Device planner scratch ([`qwen4_moe_plan_scratch_elems`] u32s): counts,
    /// block-index prefixes, per-class totals and the
    /// `VkDispatchIndirectCommand` triples the grouped GEMVs consume.
    plan: u64,
    /// Shared-expert gate `[T][sh_inter]`.
    sg: u64,
    /// Shared-expert up `[T][sh_inter]`.
    su: u64,
    /// Shared-expert down `[T][hidden]`.
    sd: u64,
    /// PLE n-gram embeddings `[T][ple_embed_dim]`.
    emb: u64,
    /// Generic wide scratch A (projection outputs), `T * wide` f32.
    wa: u64,
    /// Generic wide scratch B (permuted / normed), `T * wide` f32.
    wb: u64,
    /// Generic wide scratch C (conv / attention outputs), `T * wide` f32.
    wc: u64,
    /// `in_proj_z` raw `[T][nv*vd]`.
    z: u64,
    /// `z` in slot order `[T][nv*vd]`.
    z2: u64,
    /// Gated-delta core output `[T][nv*vd]`.
    gcore: u64,
    /// Gated-norm output `[T][nv*vd]`.
    gnorm: u64,
    /// Full-attention K rows `[T][kv_dim]`.
    kbuf: u64,
    /// Full-attention V rows `[T][kv_dim]`.
    vbuf: u64,
    /// PLE `key_proj` output `[T][hc_hidden]`.
    ple_k: u64,
    /// PLE `value_proj` output `[T][hidden]`.
    ple_v: u64,
    /// PLE gated (un-normed) `[T][hc_hidden]`.
    ple_g: u64,
    /// PLE gated-normed `[T][hc_hidden]`.
    ple_gn: u64,
    /// PLE conv output `[T][hc_hidden]`.
    ple_o: u64,
    /// F16 B-operand staging for the coopmat GEMMs, `T * hc_hidden` halfwords.
    b16: u64,
    /// Causal mask `[T][max_context]` f16.
    mask: u64,
    /// RoPE positions `[T]` i32.
    pos: u64,
    /// Chunked-GDN inter-kernel scratch (`ARLE_QWEN4_PREFILL_CHUNKED_GDN=1`
    /// lane), sized for `T.div_ceil(64)` chunks — layout owned by
    /// [`qwen4_gdn_chunk_params`]. ~7 MiB per 64-token chunk at this model's
    /// head shape; allocated unconditionally like `b16`, the other opt-in
    /// lane's staging slot.
    gdn: u64,
    total: u64,
}

/// The batched-GEMM route for one dense format: `(mm kernel, pack kernel)`.
/// `None` = record per-token GEMVs instead — the DEFAULT, and not for lack of
/// speed: the per-token GEMVs are the same dispatches decode records, so the
/// prefill is bit-exact against the decode loop (measured 0.000e0 at full
/// scale). The coopmat lane (`ARLE_QWEN4_PREFILL_GEMM=1`) is ~2x faster on
/// the dense tier but stages activations to f16 (2^-11 — B as PLAIN F16 rows
/// through [`Kernel::F16KvPack`]; bf16 staging, 2^-8, was measured first and
/// rejected), and at 48 layers ANY sub-f32 staging compounds until it crosses
/// expert-selection boundaries in the 512-expert routers — measured 2.6
/// absolute logit drift and a flipped argmax, essentially unchanged between
/// bf16 and f16 staging because the drift saturates on expert flips, not on
/// the seed rounding. A prefill that diverges from decode is a wrong prefill,
/// so the drifting lane stays opt-in until it can be made exact (candidate:
/// a two-GEMM f16 residual split, x = hi + r, recovering ~2^-22) or the MoE
/// batching round changes the numeric layout anyway. The vendored
/// `TO_FLOAT_TYPE_B` seam is what lets [`Kernel::MmCmBf16`] take f16 B (see
/// `vulkan-kernels/build.rs`).
fn pf_gemm_route(
    ctx: &VulkanContext,
    format: Qwen4DeviceFormat,
) -> Option<(Kernel, Kernel, CoopmatShape)> {
    if std::env::var("ARLE_QWEN4_PREFILL_GEMM").as_deref() != Ok("1") {
        return None;
    }
    if std::env::var_os("ARLE_VK_DISABLE_COOPMAT").is_some() {
        return None;
    }
    let shape = ctx.coopmat()?;
    let warp = ctx.subgroup_size().0;
    MmSpec::choose(shape, warp, 32, ctx.max_compute_shared_memory_size())?;
    match format {
        Qwen4DeviceFormat::Bf16 => Some((Kernel::MmCmBf16, Kernel::F16KvPack, shape)),
        Qwen4DeviceFormat::F16 => Some((Kernel::MmCmF16, Kernel::F16KvPack, shape)),
        _ => None,
    }
}

/// The chunked (WY-form) gated-delta prefill lane
/// (`ARLE_QWEN4_PREFILL_CHUNKED_GDN=1`) — OPT-IN like the coopmat GEMM lane
/// (see [`pf_gemm_route`]), and for the same reason: the default per-token
/// scan records the SAME kernel decode runs, so the prefill=decode gate is
/// bit-exact, and the WY reassociation cannot honor that. Unlike the GEMM
/// lane no sub-f32 staging enters anywhere — the scratch, the state and
/// every accumulation stay f32 — so the drift is pure reassociation: the
/// first linear layer's S agrees with the serial scan to ~4e-7 absolute, and
/// the calibrated envelope + expert-flip receipts live in
/// `tests/qwen4_prefill.rs` (`chunked_gdn_drift_*`, `_CHUNKED_AB`). Read at
/// record time, per layer, like the GEMM gate.
fn pf_chunked_gdn() -> bool {
    std::env::var("ARLE_QWEN4_PREFILL_CHUNKED_GDN").as_deref() == Ok("1")
}

/// `ARLE_QWEN4_PREFILL_HOST_PLAN=1`: plan the grouped MoE on the HOST (the
/// pre-device-planner behavior — one ids fence per (layer, chunk)). Kept as
/// the oracle fallback: [`plan_moe_groups`] and the device planner build the
/// same fixed-layout plan, so the two modes record byte-identical GEMV work
/// and the bit-exact prefill=decode gate covers both. Read at record time,
/// per layer, like the other prefill lane gates.
fn pf_host_plan() -> bool {
    std::env::var("ARLE_QWEN4_PREFILL_HOST_PLAN").as_deref() == Ok("1")
}

/// The chunk arena + chunk permutation maps for [`VulkanQwen4ExpModel::forward_prompt`].
///
/// Deliberately SEPARATE from [`DevSlots`] (the decode arena): scaling the
/// decode slots by `T` would multiply `dense_y`'s vocab-sized region into
/// gigabytes, and keeping the two apart leaves the decode path byte-for-byte
/// untouched. ~570 MB of HOST-CACHED memory at the default 256-token width —
/// the grouped-MoE class regions are sized to [`MoePlanLayout`]'s worst case
/// (~6x the live pairs, the price of record-time offsets with device-side
/// block counts; see the layout's doc) — allocated lazily on the first
/// `forward_prompt`, never for decode-only runs.
pub struct Qwen4Prefill<'ctx> {
    buffer: DeviceBuffer<'ctx>,
    maps: DeviceBuffer<'ctx>,
    /// Element offsets into `maps`: `[qkv, v, ab, vinv]` (see [`pf_chunk_maps`]).
    map_at: [usize; 4],
    slots: PfSlots,
    /// Per-token element stride of the `u` slot (packed when `hc_lowrank` is
    /// bind-alignable, padded otherwise — a padded stride also disables the
    /// GEMM route for `mix_down`, which cannot write strided rows).
    u_stride: usize,
    max_tokens: usize,
    /// Per-position logits of a VERIFY chunk, `[t][vocab]` f32 — allocated on
    /// the first `record_verify_tail` (a plain prompt prefill never needs
    /// per-position logits and never pays the ~16 MB). HOST_CACHED: the whole
    /// point of the buffer is a fast host read-back of every position.
    verify_logits: Option<DeviceBuffer<'ctx>>,
}

/// Cap on a verify chunk's positions: pending (≤ k+2 after a rejection) plus
/// k fresh drafts stays ≤ 2k+2, so 16 covers every draft depth this box will
/// see (the vendor caps at 2, the sweep at 4) with headroom.
pub const QWEN4_VERIFY_MAX_TOKENS: usize = 16;

impl<'ctx> Qwen4Prefill<'ctx> {
    pub fn new(
        ctx: &'ctx VulkanContext,
        cfg: &Qwen4ExpConfig,
        hc: &HyperConnectionConfig,
        max_tokens: usize,
    ) -> Result<Self> {
        ensure!(max_tokens >= 1, "prefill chunk of zero tokens");
        let hh = hc.hc_hidden();
        let h = cfg.hidden_size;
        let conv_dim = cfg.linear_conv_dim();
        let nv = cfg.linear_num_value_heads;
        let vd = cfg.linear_value_head_dim;
        let nvd = nv * vd;
        let q2 = cfg.num_attention_heads * cfg.head_dim * 2;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let wide = q2.max(conv_dim);
        let top_k = cfg.num_experts_per_tok;
        let inter = cfg.moe_intermediate_size;
        let sh_inter = cfg.shared_expert_intermediate_size;
        // Worst-case grouped-MoE class regions for the widest chunk; per-chunk
        // layouts (`MoePlanLayout::new(t * top_k, ..)`) always fit inside
        // (every class capacity is monotone in `pairs`).
        let moe_layout = MoePlanLayout::new(max_tokens * top_k, cfg.num_experts);
        let lowrank = hc.hc_lowrank;
        let u_stride = if (lowrank * 4).is_multiple_of(256) {
            lowrank
        } else {
            lowrank.next_multiple_of(64)
        };
        ensure!(nv <= PF_AB_PAD, "nv {nv} exceeds the a/b row pad");
        ensure!(top_k <= PF_IDS_PAD, "top_k {top_k} exceeds the ids row pad");
        // Per-token (and per-class, for the grouped MoE) binds rely on these
        // being 256-B strides; they hold for the real checkpoint and this
        // refuses rather than mis-binds elsewhere.
        for (name, stride) in [
            ("hc_hidden", hh),
            ("hidden", h),
            ("moe inter", inter),
            ("top_k*inter", top_k * inter),
            ("top_k*hidden", top_k * h),
            ("shared inter", sh_inter),
            ("ple_embed", cfg.ple_embed_dim.max(64)),
        ] {
            ensure!(
                (stride * 4).is_multiple_of(256),
                "prefill row stride `{name}` ({stride} f32) is not 256-B aligned"
            );
        }

        let mut off = 0u64;
        let mut take = |bytes: u64| {
            let at = off;
            off += bytes.div_ceil(256) * 256;
            at
        };
        let t = max_tokens as u64;
        let f32s = |elems: usize| t * elems as u64 * 4;
        let slots = PfSlots {
            h: take(f32s(hh)),
            hn: take(f32s(hh)),
            u: take(f32s(u_stride)),
            x: take(f32s(h)),
            y: take(f32s(h)),
            acc: take(f32s(h)),
            a_raw: take(f32s(PF_AB_PAD)),
            b_raw: take(f32s(PF_AB_PAD)),
            a2: take(f32s(nv)),
            b2: take(f32s(nv)),
            ids: take(f32s(PF_IDS_PAD)),
            wts: take(f32s(PF_IDS_PAD)),
            shg: take(f32s(PF_IDS_PAD)),
            lgt: take(f32s(cfg.num_experts)),
            ge: take((moe_layout.pair_capacity * inter * 4) as u64),
            gu: take((moe_layout.pair_capacity * inter * 4) as u64),
            edown: take(f32s(top_k * h)),
            xg: take((moe_layout.pair_capacity * h * 4) as u64),
            edc: take((moe_layout.pair_capacity * h * 4) as u64),
            smap: take(f32s(top_k)),
            eids: take((moe_layout.list_capacity * 4) as u64),
            plan: take(
                u64::from(qwen4_moe_plan_scratch_elems(u32::try_from(
                    cfg.num_experts,
                )?)) * 4,
            ),
            sg: take(f32s(sh_inter)),
            su: take(f32s(sh_inter)),
            sd: take(f32s(h)),
            emb: take(f32s(cfg.ple_embed_dim.max(1))),
            wa: take(f32s(wide)),
            wb: take(f32s(wide)),
            wc: take(f32s(wide)),
            z: take(f32s(nvd)),
            z2: take(f32s(nvd)),
            gcore: take(f32s(nvd)),
            gnorm: take(f32s(nvd)),
            kbuf: take(f32s(kv_dim)),
            vbuf: take(f32s(kv_dim)),
            ple_k: take(f32s(hh)),
            ple_v: take(f32s(h)),
            ple_g: take(f32s(hh)),
            ple_gn: take(f32s(hh)),
            ple_o: take(f32s(hh)),
            b16: take(t * hh as u64 * 2),
            mask: take(t * cfg.max_context as u64 * 2),
            pos: take(t * 4),
            gdn: take(
                qwen4_gdn_chunk_scratch_elems(
                    (max_tokens.div_ceil(QWEN4_GDN_CHUNK as usize)) as u32,
                    nv as u32,
                    cfg.linear_key_head_dim as u32,
                    vd as u32,
                ) as u64
                    * 4,
            ),
            total: off,
        };

        let buffer = DeviceBuffer::alloc_host_cached(ctx, usize::try_from(slots.total)?)
            .map_err(|e| anyhow!("alloc qwen4 prefill arena ({} B): {e}", slots.total))?;
        let (flat, map_at) = pf_chunk_maps(cfg, max_tokens);
        let map_bytes: Vec<u8> = flat.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut maps = DeviceBuffer::alloc_uma(ctx, map_bytes.len().max(4))
            .map_err(|e| anyhow!("alloc qwen4 prefill maps: {e}"))?;
        maps.copy_from_host(&map_bytes)
            .map_err(|e| anyhow!("upload qwen4 prefill maps: {e}"))?;
        Ok(Self {
            buffer,
            maps,
            map_at,
            slots,
            u_stride,
            max_tokens,
            verify_logits: None,
        })
    }

    /// The chunk width this arena was built for.
    #[must_use]
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    fn write_f32_at(&mut self, off: u64, data: &[f32]) -> Result<()> {
        let _p = prof::span_bytes("h2d", (data.len() * 4) as u64);
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.buffer
            .copy_from_host_at(off, &bytes)
            .map_err(|e| anyhow!("prefill arena write at {off}: {e}"))
    }

    fn write_bytes_at(&mut self, off: u64, bytes: &[u8]) -> Result<()> {
        let _p = prof::span_bytes("h2d", bytes.len() as u64);
        self.buffer
            .copy_from_host_at(off, bytes)
            .map_err(|e| anyhow!("prefill arena write at {off}: {e}"))
    }

    fn read_f32_at(&self, off: u64, n: usize) -> Result<Vec<f32>> {
        let _p = prof::span_bytes("d2h", (n * 4) as u64);
        let mut bytes = vec![0u8; n * 4];
        self.buffer
            .copy_to_host_at(off, &mut bytes)
            .map_err(|e| anyhow!("prefill arena read at {off}: {e}"))?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    fn read_i32_at(&self, off: u64, n: usize) -> Result<Vec<i32>> {
        let _p = prof::span_bytes("d2h", (n * 4) as u64);
        let mut bytes = vec![0u8; n * 4];
        self.buffer
            .copy_to_host_at(off, &mut bytes)
            .map_err(|e| anyhow!("prefill arena read at {off}: {e}"))?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Row `tok` of a region whose per-token stride is `stride` f32 elements.
    fn row(base: u64, tok: usize, stride: usize) -> u64 {
        base + (tok * stride * 4) as u64
    }

    /// Stage a packed `[t][k]` f32 block as the GEMM B operand at `b16`,
    /// through whichever pack kernel `route` names.
    fn record_pack(
        &self,
        dev: &mut Qwen4Dev<'ctx>,
        pack_kernel: Kernel,
        src_off: u64,
        ne: usize,
    ) -> Result<()> {
        let push = f16_kv_pack_params(ne as u32).to_le_bytes();
        let d = f16_kv_pack_dispatch_rows(ne as u32, 1);
        dev.rec(
            pack_kernel,
            pack_kernel.specialization_u32(),
            &push,
            &[
                Bind::Ext(&self.buffer, src_off, (ne * 4) as u64),
                Bind::Ext(&self.buffer, self.slots.b16, (ne * 2) as u64),
            ],
            [d.x, d.y, d.z],
        )
    }

    /// One dense projection over the whole chunk: a coopmat GEMM when the
    /// tier's format has one AND the caller staged `b16` with the matching
    /// pack kernel; otherwise `t` recorded GEMVs through the same
    /// format-dispatch decode uses. `packed` is `Some` only when the B operand
    /// for THIS tensor's format is already staged.
    #[expect(clippy::too_many_arguments, reason = "a chunk projection is this wide")]
    fn record_dense_chunk(
        &self,
        dev: &mut Qwen4Dev<'ctx>,
        weights: &Qwen4Weights<'_, '_>,
        w: &Qwen4DeviceTensor,
        src32: u64,
        src_stride: usize,
        packed: bool,
        dst: u64,
        dst_stride: usize,
        t: usize,
    ) -> Result<()> {
        let (k, m) = (w.ncols, w.nrows);
        ensure!(src_stride >= k, "chunk GEMM: source stride under-spans k");
        let route = pf_gemm_route(dev.ctx, w.format);
        if let Some((mm, _, shape)) = route {
            // A GEMM writes rows packed at stride m; a caller needing padded
            // rows gets the GEMV loop instead.
            if packed && src_stride == k && dst_stride == m {
                let warp = dev.ctx.subgroup_size().0;
                let spec = MmSpec::choose(
                    shape,
                    warp,
                    t as u32,
                    dev.ctx.max_compute_shared_memory_size(),
                )
                .ok_or_else(|| anyhow!("no coopmat warptile for n = {t}"))?;
                let (wb, wo, wl) = weights.binding(w)?;
                let push = mmq_params(m as u32, t as u32, k as u32).to_le_bytes();
                let d = mm_dispatch(m as u32, t as u32, &spec);
                return dev.rec(
                    mm,
                    spec.specialization_u32(),
                    &push,
                    &[
                        Bind::Ext(wb, wo, wl),
                        Bind::Ext(&self.buffer, self.slots.b16, (t * k * 2) as u64),
                        Bind::Ext(&self.buffer, dst, (t * m * 4) as u64),
                    ],
                    [d.x, d.y, d.z],
                );
            }
        }
        // NUM_COLS batching: the vendored shader's own batch axis. One
        // dispatch computes up to [`qwen4_gemv_cols_cap`] activation columns
        // with the weight row read ONCE — for a k-token verify this is what
        // divides the dense bytes by the accepted length, and per-column
        // arithmetic is bit-identical to the per-token loop below (the
        // prefill=decode gate covers it at full scale). Packed rows only:
        // column `j` must sit at `j * ncols` (B) and `j * nrows` (D), which is
        // exactly the `gemv_params_f32_b_cols` stride contract.
        // The width gates mirror `record_dense_binds`' own ensures; a shape
        // that fails them falls to the loop and fails THERE, identically.
        let cap = qwen4_gemv_cols_cap();
        let cols_kernel = match w.format {
            Qwen4DeviceFormat::Bf16 if k.is_multiple_of(4) => Some(Kernel::GemvBf16),
            Qwen4DeviceFormat::F16 if k.is_multiple_of(4) => Some(Kernel::GemvF16),
            Qwen4DeviceFormat::Q4K if k.is_multiple_of(256) => Some(Kernel::GemvQ4KDense),
            Qwen4DeviceFormat::Q8_0 if k.is_multiple_of(8) => Some(Kernel::GemvQ8_0Dense),
            _ => None,
        };
        if let Some(kernel) = cols_kernel
            && cap > 1
            && t > 1
            && src_stride == k
            && dst_stride == m
        {
            let (wb, wo, wl) = weights.binding(w)?;
            let push = gemv_params_f32_b_cols(u32::try_from(k)?, u32::try_from(m)?).to_le_bytes();
            let mut tok = 0usize;
            while tok < t {
                let g = (t - tok).min(cap);
                let spec = kernel
                    .gemv_cols_spec(u32::try_from(g)?)
                    .expect("cols kernels all carry a cols spec");
                let d = gemv_dense_dispatch(u32::try_from(m)?, &spec);
                dev.rec(
                    kernel,
                    spec.specialization_u32(),
                    &push,
                    &[
                        Bind::Ext(wb, wo, wl),
                        Bind::Ext(&self.buffer, Self::row(src32, tok, k), (g * k * 4) as u64),
                        Bind::Ext(&self.buffer, Self::row(dst, tok, m), (g * m * 4) as u64),
                        Bind::A(dev.slots.dummy, 8),
                        Bind::A(dev.slots.dummy, 8),
                    ],
                    [d.x, d.y, d.z],
                )?;
                tok += g;
            }
            return Ok(());
        }
        for tok in 0..t {
            dev.record_dense_binds(
                weights,
                w,
                Bind::Ext(
                    &self.buffer,
                    Self::row(src32, tok, src_stride),
                    (k * 4) as u64,
                ),
                Bind::Ext(
                    &self.buffer,
                    Self::row(dst, tok, dst_stride),
                    (m * 4) as u64,
                ),
            )?;
        }
        Ok(())
    }

    /// One chunk-wide block permutation through the token-major maps.
    #[expect(clippy::too_many_arguments, reason = "src/dst spans differ")]
    fn record_perm_chunk(
        &self,
        dev: &mut Qwen4Dev<'ctx>,
        src: u64,
        src_span_elems: usize,
        map_region: usize,
        block: usize,
        nblocks: usize,
        dst: u64,
        dst_span_elems: usize,
    ) -> Result<()> {
        let push = qwen4_block_perm_params(block as u32, nblocks as u32).to_le_bytes();
        let d = qwen4_block_perm_dispatch(nblocks as u32);
        dev.rec(
            Kernel::Qwen4BlockPerm,
            Kernel::Qwen4BlockPerm.specialization_u32(),
            &push,
            &[
                Bind::Ext(&self.buffer, src, (src_span_elems * 4) as u64),
                Bind::Ext(
                    &self.maps,
                    (self.map_at[map_region] * 4) as u64,
                    (nblocks * 4) as u64,
                ),
                Bind::Ext(&self.buffer, dst, (dst_span_elems * 4) as u64),
            ],
            [d.x, d.y, d.z],
        )
    }

    /// [`Self::record_perm_chunk`] over a DYNAMIC map living in the arena
    /// rather than the static chunk maps: the grouped-MoE scatter map is
    /// written per (layer, chunk) — by the device planner in the recorded
    /// stream (barrier-ordered), or by the host after the fence under
    /// `ARLE_QWEN4_PREFILL_HOST_PLAN=1`.
    #[expect(clippy::too_many_arguments, reason = "src/dst spans differ")]
    fn record_perm_dyn(
        &self,
        dev: &mut Qwen4Dev<'ctx>,
        src: u64,
        src_span_elems: usize,
        map_off: u64,
        block: usize,
        nblocks: usize,
        dst: u64,
        dst_span_elems: usize,
    ) -> Result<()> {
        let push = qwen4_block_perm_params(block as u32, nblocks as u32).to_le_bytes();
        let d = qwen4_block_perm_dispatch(nblocks as u32);
        dev.rec(
            Kernel::Qwen4BlockPerm,
            Kernel::Qwen4BlockPerm.specialization_u32(),
            &push,
            &[
                Bind::Ext(&self.buffer, src, (src_span_elems * 4) as u64),
                Bind::Ext(&self.buffer, map_off, (nblocks * 4) as u64),
                Bind::Ext(&self.buffer, dst, (dst_span_elems * 4) as u64),
            ],
            [d.x, d.y, d.z],
        )
    }

    /// The pre half of a hyper-connection site over the chunk: per-token
    /// grouped norm (the norm's per-group gains cannot batch across tokens),
    /// chunked `mix_down`, per-token mix. Leaves `h`/`hn` rows for the combine
    /// half and the block input in the `x` rows.
    fn record_hc_pre(
        &self,
        dev: &mut Qwen4Dev<'ctx>,
        weights: &Qwen4Weights<'_, '_>,
        hc: &HyperConnectionConfig,
        layer: Option<usize>,
        site: HcSite,
        t: usize,
    ) -> Result<()> {
        let _s = prof::stage("pf.hc.pre");
        let b = weights.hyper_connection(layer, site)?;
        let hh = hc.hc_hidden();
        ensure!(
            b.mix_up.format == b.mix_down.format,
            "hyper-connection mix_down/mix_up formats diverge"
        );
        let mix_kernel = match b.mix_up.format {
            Qwen4DeviceFormat::F32 => Kernel::Qwen4HcMix,
            Qwen4DeviceFormat::Bf16 => Kernel::Qwen4HcMixBf16,
            f => bail!("hyper-connection mix weights in unsupported format {f:?}"),
        };
        let s = self.slots;
        let (nb, no, nl) = weights.binding(b.hc_norm)?;
        let push =
            rms_norm_params_grouped(hc.hidden_size as u32, hc.hc_count as u32, hc.rms_norm_eps)
                .to_le_bytes();
        let d = rms_norm_dispatch_rows(hc.hc_count as u32);
        for tok in 0..t {
            dev.rec(
                Kernel::RmsNorm,
                Kernel::RmsNorm.specialization_u32(),
                &push,
                &[
                    Bind::Ext(&self.buffer, Self::row(s.h, tok, hh), (hh * 4) as u64),
                    Bind::Ext(nb, no, nl),
                    Bind::Ext(&self.buffer, Self::row(s.hn, tok, hh), (hh * 4) as u64),
                ],
                [d.x, d.y, d.z],
            )?;
        }
        dev.barrier();
        let packed = if let Some((_, pack, _)) = pf_gemm_route(dev.ctx, b.mix_down.format) {
            self.record_pack(dev, pack, s.hn, t * hh)?;
            dev.barrier();
            true
        } else {
            false
        };
        self.record_dense_chunk(
            dev,
            weights,
            b.mix_down,
            s.hn,
            hh,
            packed,
            s.u,
            self.u_stride,
            t,
        )?;
        dev.barrier();
        let (up_buf, up_off, up_len) = weights.binding(b.mix_up)?;
        let push = qwen4_hc_mix_params(
            hc.hidden_size as u32,
            hc.hc_count as u32,
            hc.hc_lowrank as u32,
        )
        .to_le_bytes();
        let d = qwen4_hc_mix_dispatch(hc.hidden_size as u32);
        for tok in 0..t {
            dev.rec(
                mix_kernel,
                mix_kernel.specialization_u32(),
                &push,
                &[
                    Bind::Ext(&self.buffer, Self::row(s.hn, tok, hh), (hh * 4) as u64),
                    Bind::Ext(up_buf, up_off, up_len),
                    Bind::Ext(
                        &self.buffer,
                        Self::row(s.u, tok, self.u_stride),
                        (hc.hc_lowrank * 4) as u64,
                    ),
                    Bind::Ext(
                        &self.buffer,
                        Self::row(s.x, tok, hc.hidden_size),
                        (hc.hidden_size * 4) as u64,
                    ),
                ],
                [d.x, d.y, d.z],
            )?;
        }
        Ok(())
    }

    /// The combine half over the chunk: inject each token's block output row
    /// (`y_base` at `y_stride`) into its residual row. `h`/`hn` rows must be
    /// the ones [`Self::record_hc_pre`] left for this site.
    #[expect(clippy::too_many_arguments, reason = "site + output rows")]
    fn record_hc_combine(
        &self,
        dev: &mut Qwen4Dev<'ctx>,
        weights: &Qwen4Weights<'_, '_>,
        hc: &HyperConnectionConfig,
        layer: Option<usize>,
        site: HcSite,
        y_base: u64,
        y_stride: usize,
        t: usize,
    ) -> Result<()> {
        let _s = prof::stage("pf.hc.comb");
        let b = weights.hyper_connection(layer, site)?;
        let inject = b
            .block_inject
            .ok_or_else(|| anyhow!("hc combine on the mixer site (no block_inject)"))?;
        let comb_kernel = match inject.format {
            Qwen4DeviceFormat::F32 => Kernel::Qwen4HcCombine,
            Qwen4DeviceFormat::Bf16 => Kernel::Qwen4HcCombineBf16,
            f => bail!("hyper-connection inject weights in unsupported format {f:?}"),
        };
        let (ib, io, il) = weights.binding(inject)?;
        let hh = hc.hc_hidden();
        let s = self.slots;
        let push = qwen4_hc_combine_params(hc.hidden_size as u32, hc.hc_count as u32).to_le_bytes();
        let d = qwen4_hc_combine_dispatch(hc.hidden_size as u32);
        for tok in 0..t {
            dev.rec(
                comb_kernel,
                comb_kernel.specialization_u32(),
                &push,
                &[
                    Bind::Ext(&self.buffer, Self::row(s.hn, tok, hh), (hh * 4) as u64),
                    Bind::Ext(ib, io, il),
                    Bind::Ext(&self.buffer, Self::row(s.h, tok, hh), (hh * 4) as u64),
                    Bind::Ext(
                        &self.buffer,
                        Self::row(y_base, tok, y_stride),
                        (hc.hidden_size * 4) as u64,
                    ),
                ],
                [d.x, d.y, d.z],
            )?;
        }
        Ok(())
    }

    /// The PLE layer over the chunk: chunked key/value projections, then the
    /// gate and dilated-conv kernels in their `seq_len = t` modes against the
    /// ring staged in the DECODE arena's `ple_ring` slot, then `h += out`.
    /// The caller stages the ring before the chunk and reads it back (rotated)
    /// after the end-of-chunk flush.
    fn record_ple(
        &self,
        dev: &mut Qwen4Dev<'ctx>,
        weights: &Qwen4Weights<'_, '_>,
        cfg: &Qwen4ExpConfig,
        layer: usize,
        t: usize,
    ) -> Result<()> {
        let _s = prof::stage("pf.ple");
        let pc = ple_config(cfg);
        let hh = pc.hc_hidden();
        let name = |suffix: &str| layer_tensor_name(layer, suffix);
        let kp = *weights.tensor(&name("ple.key_proj.weight"))?;
        let vp = *weights.tensor(&name("ple.value_proj.weight"))?;
        let s = self.slots;
        let packed = if let Some((_, pack, _)) = pf_gemm_route(dev.ctx, kp.format) {
            self.record_pack(dev, pack, s.emb, t * pc.ple_embed_dim)?;
            dev.barrier();
            true
        } else {
            false
        };
        self.record_dense_chunk(
            dev,
            weights,
            &kp,
            s.emb,
            pc.ple_embed_dim,
            packed,
            s.ple_k,
            hh,
            t,
        )?;
        self.record_dense_chunk(
            dev,
            weights,
            &vp,
            s.emb,
            pc.ple_embed_dim,
            packed && vp.format == kp.format,
            s.ple_v,
            pc.hidden_size,
            t,
        )?;
        dev.barrier();
        let (nk_b, nk_o, nk_l) = weights.binding_by_name(&name("ple.norm_key.weight"))?;
        let (nq_b, nq_o, nq_l) = weights.binding_by_name(&name("ple.norm_query.weight"))?;
        let (nc_b, nc_o, nc_l) = weights.binding_by_name(&name("ple.norm_conv.weight"))?;
        let push = qwen4_ple_gate_params(
            pc.hidden_size as u32,
            pc.hc_count as u32,
            t as u32,
            pc.rms_norm_eps,
        )
        .to_le_bytes();
        let d = qwen4_ple_gate_dispatch(pc.hc_count as u32, t as u32);
        dev.rec(
            Kernel::Qwen4PleGate,
            Kernel::Qwen4PleGate.specialization_u32(),
            &push,
            &[
                Bind::Ext(&self.buffer, s.ple_k, (t * hh * 4) as u64),
                Bind::Ext(&self.buffer, s.h, (t * hh * 4) as u64),
                Bind::Ext(&self.buffer, s.ple_v, (t * pc.hidden_size * 4) as u64),
                Bind::Ext(nk_b, nk_o, nk_l),
                Bind::Ext(nq_b, nq_o, nq_l),
                Bind::Ext(nc_b, nc_o, nc_l),
                Bind::Ext(&self.buffer, s.ple_g, (t * hh * 4) as u64),
                Bind::Ext(&self.buffer, s.ple_gn, (t * hh * 4) as u64),
            ],
            [d.x, d.y, d.z],
        )?;
        dev.barrier();
        let (cw_b, cw_o, cw_l) = weights.binding_by_name(&name("ple.conv1d.weight"))?;
        let push = qwen4_ple_conv_params(
            hh as u32,
            t as u32,
            pc.conv_kernel_size as u32,
            pc.conv_dilation as u32,
            0,
        )
        .to_le_bytes();
        let d = qwen4_ple_conv_dispatch(hh as u32);
        let ring_len = (pc.short_conv_state_len() * hh * 4) as u64;
        dev.rec(
            Kernel::Qwen4PleConv,
            Kernel::Qwen4PleConv.specialization_u32(),
            &push,
            &[
                Bind::Ext(&self.buffer, s.ple_gn, (t * hh * 4) as u64),
                Bind::Ext(cw_b, cw_o, cw_l),
                Bind::A(dev.slots.ple_ring, ring_len),
                Bind::Ext(&self.buffer, s.ple_g, (t * hh * 4) as u64),
                Bind::Ext(&self.buffer, s.ple_o, (t * hh * 4) as u64),
            ],
            [d.x, d.y, d.z],
        )?;
        dev.barrier();
        // h += 1.0 * out, flat over the whole chunk, through the weighted-
        // accum kernel exactly as decode's `add_into_h` does. NOT `add.comp`:
        // its workgroups cover overlapping index ranges (each thread handles
        // `idx` and `idx + 256`), which is benign into a distinct output but a
        // read-after-write race when `d` aliases `a` — the in-place residual
        // add here. The accum kernel touches each element from exactly one
        // thread. The add is UNCONDITIONAL — omitting the PLE is a wrong
        // forward, not a degraded one.
        let n = (t * hh) as u32;
        let push = qwen36_moe_weighted_accum_params(n, 1, false).to_le_bytes();
        let d = qwen36_moe_weighted_accum_dispatch(n);
        dev.rec(
            Kernel::Qwen36MoeWeightedAccum,
            Kernel::Qwen36MoeWeightedAccum.specialization_u32(),
            &push,
            &[
                Bind::Ext(&self.buffer, s.ple_o, u64::from(n) * 4),
                Bind::A(dev.slots.one, 4),
                Bind::Ext(&self.buffer, s.h, u64::from(n) * 4),
            ],
            [d.x, d.y, d.z],
        )
    }

    /// The linear-attention block over the chunk: chunked in-projections,
    /// chunk-wide HF->GGUF permutation, then the conv and gated-delta kernels
    /// in their `seq_len = t` modes — the SAME resident state buffers decode
    /// advances, stepped `t` tokens in one dispatch each.
    fn record_linear(
        &self,
        dev: &mut Qwen4Dev<'ctx>,
        weights: &Qwen4Weights<'_, '_>,
        rl: &DevResidentLinAttn<'ctx>,
        cfg: &Qwen4ExpConfig,
        layer: usize,
        t: usize,
    ) -> Result<()> {
        let _s = prof::stage("pf.linattn");
        let kd = cfg.linear_key_head_dim;
        let vd = cfg.linear_value_head_dim;
        let nk = cfg.linear_num_key_heads;
        let nv = cfg.linear_num_value_heads;
        let kernel = cfg.linear_conv_kernel_dim;
        let conv_dim = cfg.linear_conv_dim();
        let h = cfg.hidden_size;
        let state_w = kernel - 1;
        let dense = *rl
            .index
            .get(&layer)
            .ok_or_else(|| anyhow!("layer {layer} not resident"))?;
        let aux_base = (dense * DevResidentLinAttn::aux_stride(cfg) * 4) as u64;
        let conv_w_len = (conv_dim * kernel * 4) as u64;
        let alog_at = aux_base + ((conv_dim * kernel).next_multiple_of(64) * 4) as u64;
        let state_base = (dense * DevResidentLinAttn::state_stride(cfg) * 4) as u64;
        let ring_at = state_base + (nv * kd * vd * 4) as u64;
        let s = self.slots;

        let t_qkv = *weights.tensor(&layer_tensor_name(layer, "linear_attn.in_proj_qkv.weight"))?;
        let t_z = *weights.tensor(&layer_tensor_name(layer, "linear_attn.in_proj_z.weight"))?;
        let t_a = *weights.tensor(&layer_tensor_name(layer, "linear_attn.in_proj_a.weight"))?;
        let t_b = *weights.tensor(&layer_tensor_name(layer, "linear_attn.in_proj_b.weight"))?;
        let t_out = *weights.tensor(&layer_tensor_name(layer, "linear_attn.out_proj.weight"))?;
        let t_norm = *weights.tensor(&layer_tensor_name(layer, "linear_attn.norm.weight"))?;
        ensure!(
            t_norm.format == Qwen4DeviceFormat::F32,
            "linear_attn.norm must be F32-resident"
        );
        let norm_bind = weights.binding(&t_norm)?;

        // In-projections off the shared block input.
        let packed = if let Some((_, pack, _)) = pf_gemm_route(dev.ctx, t_qkv.format) {
            self.record_pack(dev, pack, s.x, t * h)?;
            dev.barrier();
            true
        } else {
            false
        };
        self.record_dense_chunk(dev, weights, &t_qkv, s.x, h, packed, s.wa, conv_dim, t)?;
        self.record_dense_chunk(
            dev,
            weights,
            &t_z,
            s.x,
            h,
            packed && t_z.format == t_qkv.format,
            s.z,
            nv * vd,
            t,
        )?;
        // a/b rows land padded so their per-token GEMV outputs stay bindable;
        // the `ab` chunk map compacts them into the packed slot order the
        // gated-delta kernel reads.
        self.record_dense_chunk(dev, weights, &t_a, s.x, h, false, s.a_raw, PF_AB_PAD, t)?;
        self.record_dense_chunk(dev, weights, &t_b, s.x, h, false, s.b_raw, PF_AB_PAD, t)?;
        dev.barrier();

        let qkv_blocks = conv_dim / vd;
        self.record_perm_chunk(
            dev,
            s.wa,
            t * conv_dim,
            0,
            vd,
            t * qkv_blocks,
            s.wb,
            t * conv_dim,
        )?;
        self.record_perm_chunk(dev, s.z, t * nv * vd, 1, vd, t * nv, s.z2, t * nv * vd)?;
        self.record_perm_chunk(dev, s.a_raw, t * PF_AB_PAD, 2, 1, t * nv, s.a2, t * nv)?;
        self.record_perm_chunk(dev, s.b_raw, t * PF_AB_PAD, 2, 1, t * nv, s.b2, t * nv)?;
        dev.barrier();

        let push = qwen35_ssm_conv_params(conv_dim as u32, t as u32, kernel as u32).to_le_bytes();
        let d = qwen35_ssm_conv_dispatch(conv_dim as u32);
        dev.rec(
            Kernel::Qwen35SsmConv,
            Kernel::Qwen35SsmConv.specialization_u32(),
            &push,
            &[
                Bind::Ext(&self.buffer, s.wb, (t * conv_dim * 4) as u64),
                Bind::Ext(&rl.aux, aux_base, conv_w_len),
                Bind::Ext(&rl.state, ring_at, (conv_dim * state_w * 4) as u64),
                Bind::Ext(&self.buffer, s.wc, (t * conv_dim * 4) as u64),
            ],
            [d.x, d.y, d.z],
        )?;
        dev.barrier();
        if pf_chunked_gdn() {
            // Chunked WY-form lane: same inputs (post-conv qkv in slot order,
            // packed a/b, resident dt_bias/ssm_a), same resident f32 state,
            // same output layout — only the recurrence is reassociated. Two
            // dispatches instead of one, against a serial scan that was the
            // largest single slice of the prefill GPU drain (2.8 s of 7.2 s
            // at chunk 256); the intra kernel is parallel over
            // (64-token chunk, head) where the scan had ~nv near-serial
            // workgroups for the whole chunk.
            ensure!(
                vd as u32 % QWEN4_GDN_STATE_COLS == 0,
                "chunked GDN needs val_dim ({vd}) divisible by {QWEN4_GDN_STATE_COLS}"
            );
            let n_chunks = t.div_ceil(QWEN4_GDN_CHUNK as usize) as u32;
            let scratch_len =
                qwen4_gdn_chunk_scratch_elems(n_chunks, nv as u32, kd as u32, vd as u32) as u64 * 4;
            let push = qwen4_gdn_chunk_params(
                nk as u32, nv as u32, kd as u32, vd as u32, t as u32, n_chunks,
            )
            .to_le_bytes();
            let d = qwen4_gdn_chunk_intra_dispatch(n_chunks, nv as u32);
            dev.rec(
                Kernel::Qwen4GdnChunkIntra,
                Kernel::Qwen4GdnChunkIntra.specialization_u32(),
                &push,
                &[
                    Bind::Ext(&self.buffer, s.wc, (t * conv_dim * 4) as u64),
                    Bind::Ext(&self.buffer, s.b2, (t * nv * 4) as u64),
                    Bind::Ext(&self.buffer, s.a2, (t * nv * 4) as u64),
                    Bind::Ext(&rl.aux, alog_at + 64 * 4, (nv * 4) as u64),
                    Bind::Ext(&rl.aux, alog_at, (nv * 4) as u64),
                    Bind::Ext(&self.buffer, s.gdn, scratch_len),
                ],
                [d.x, d.y, d.z],
            )?;
            dev.barrier();
            let d = qwen4_gdn_chunk_state_dispatch(nv as u32, vd as u32);
            dev.rec(
                Kernel::Qwen4GdnChunkState,
                Kernel::Qwen4GdnChunkState.specialization_u32(),
                &push,
                &[
                    Bind::Ext(&self.buffer, s.gdn, scratch_len),
                    Bind::Ext(&rl.state, state_base, (nv * kd * vd * 4) as u64),
                    Bind::Ext(&self.buffer, s.gcore, (t * nv * vd * 4) as u64),
                ],
                [d.x, d.y, d.z],
            )?;
        } else {
            let push =
                qwen35_gated_delta_net_params(nk as u32, nv as u32, kd as u32, vd as u32, t as u32)
                    .to_le_bytes();
            let d = qwen35_gated_delta_net_dispatch(nv as u32);
            dev.rec(
                Kernel::Qwen35GatedDeltaNet,
                Kernel::Qwen35GatedDeltaNet.specialization_u32(),
                &push,
                &[
                    Bind::Ext(&self.buffer, s.wc, (t * conv_dim * 4) as u64),
                    Bind::Ext(&self.buffer, s.b2, (t * nv * 4) as u64),
                    Bind::Ext(&self.buffer, s.a2, (t * nv * 4) as u64),
                    Bind::Ext(&rl.aux, alog_at + 64 * 4, (nv * 4) as u64),
                    Bind::Ext(&rl.aux, alog_at, (nv * 4) as u64),
                    Bind::Ext(&rl.state, state_base, (nv * kd * vd * 4) as u64),
                    Bind::Ext(&self.buffer, s.gcore, (t * nv * vd * 4) as u64),
                ],
                [d.x, d.y, d.z],
            )?;
        }
        dev.barrier();
        let push = rms_norm_params_rows(vd as u32, (t * nv) as u32, vd as u32, cfg.rms_norm_eps)
            .to_le_bytes();
        let d = rms_norm_dispatch_rows((t * nv) as u32);
        dev.rec(
            Kernel::RmsNorm,
            Kernel::RmsNorm.specialization_u32(),
            &push,
            &[
                Bind::Ext(&self.buffer, s.gcore, (t * nv * vd * 4) as u64),
                Bind::Ext(norm_bind.0, norm_bind.1, norm_bind.2),
                Bind::Ext(&self.buffer, s.gnorm, (t * nv * vd * 4) as u64),
            ],
            [d.x, d.y, d.z],
        )?;
        dev.barrier();
        let n_gate = (t * nv * vd) as u32;
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
                Bind::Ext(&self.buffer, s.z2, u64::from(n_gate) * 4),
                Bind::Ext(&self.buffer, s.gnorm, u64::from(n_gate) * 4),
                Bind::Ext(&self.buffer, s.gnorm, u64::from(n_gate) * 4),
            ],
            [d.x, d.y, d.z],
        )?;
        dev.barrier();
        self.record_perm_chunk(dev, s.gnorm, t * nv * vd, 3, vd, t * nv, s.wb, t * nv * vd)?;
        dev.barrier();
        let out_packed = if let Some((_, pack, _)) = pf_gemm_route(dev.ctx, t_out.format) {
            self.record_pack(dev, pack, s.wb, t * nv * vd)?;
            dev.barrier();
            true
        } else {
            false
        };
        self.record_dense_chunk(dev, weights, &t_out, s.wb, nv * vd, out_packed, s.y, h, t)
    }

    /// The full-attention block over the chunk: chunked q/k/v projections,
    /// batched q/k norms + RoPE, strided KV pack into the layer's f16 planes
    /// at `start_pos..start_pos + t`, ONE causal-masked flash dispatch, the
    /// strided sigmoid gate, and the chunked o-projection.
    fn record_full(
        &self,
        dev: &mut Qwen4Dev<'ctx>,
        weights: &Qwen4Weights<'_, '_>,
        cfg: &Qwen4ExpConfig,
        layer: usize,
        t: usize,
        start_pos: usize,
    ) -> Result<()> {
        let _s = prof::stage("pf.fullattn");
        let hd = cfg.head_dim;
        let nq = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let h = cfg.hidden_size;
        let q_dim = nq * hd;
        let kv_dim = nkv * hd;
        let kv_len = start_pos + t;
        let full_idx = *dev
            .full_idx
            .get(&layer)
            .ok_or_else(|| anyhow!("layer {layer} has no device KV plane"))?;
        let (plane_bytes, layer_bytes, k_base, v_base, k_rows, v_rows) = {
            let kv = dev
                .kv
                .as_ref()
                .ok_or_else(|| anyhow!("no device KV cache"))?;
            ensure!(
                (kv_len * hd * 2) as u64 <= kv.plane_bytes,
                "chunk end {kv_len} exceeds the device KV plane"
            );
            let k_rows: Vec<u64> = (0..nkv)
                .map(|kvh| kv.row(kv.k_plane(full_idx, kvh), start_pos))
                .collect();
            let v_rows: Vec<u64> = (0..nkv)
                .map(|kvh| kv.row(kv.v_plane(full_idx, kvh), start_pos))
                .collect();
            (
                kv.plane_bytes,
                kv.layer_bytes,
                kv.k_plane(full_idx, 0),
                kv.v_plane(full_idx, 0),
                k_rows,
                v_rows,
            )
        };
        let name = |sfx: &str| layer_tensor_name(layer, sfx);
        let q_w = *weights.tensor(&name("self_attn.q_proj.weight"))?;
        let k_w = *weights.tensor(&name("self_attn.k_proj.weight"))?;
        let v_w = *weights.tensor(&name("self_attn.v_proj.weight"))?;
        let o_w = *weights.tensor(&name("self_attn.o_proj.weight"))?;
        let s = self.slots;

        let packed = if let Some((_, pack, _)) = pf_gemm_route(dev.ctx, q_w.format) {
            self.record_pack(dev, pack, s.x, t * h)?;
            dev.barrier();
            true
        } else {
            false
        };
        self.record_dense_chunk(dev, weights, &q_w, s.x, h, packed, s.wa, nq * 2 * hd, t)?;
        self.record_dense_chunk(
            dev,
            weights,
            &k_w,
            s.x,
            h,
            packed && k_w.format == q_w.format,
            s.kbuf,
            kv_dim,
            t,
        )?;
        self.record_dense_chunk(
            dev,
            weights,
            &v_w,
            s.x,
            h,
            packed && v_w.format == q_w.format,
            s.vbuf,
            kv_dim,
            t,
        )?;
        dev.barrier();
        // Per-head q norm over all (token, head) rows at once: row `r` reads
        // the QUERY half at `r * 2*hd` and writes packed `[t][nq][hd]`.
        let (qn_b, qn_o, qn_l) = weights.binding_by_name(&name("self_attn.q_norm.weight"))?;
        let push = rms_norm_params_rows(
            hd as u32,
            (t * nq) as u32,
            (2 * hd) as u32,
            cfg.rms_norm_eps,
        )
        .to_le_bytes();
        let d = rms_norm_dispatch_rows((t * nq) as u32);
        dev.rec(
            Kernel::RmsNorm,
            Kernel::RmsNorm.specialization_u32(),
            &push,
            &[
                Bind::Ext(&self.buffer, s.wa, (t * nq * 2 * hd * 4) as u64),
                Bind::Ext(qn_b, qn_o, qn_l),
                Bind::Ext(&self.buffer, s.wb, (t * q_dim * 4) as u64),
            ],
            [d.x, d.y, d.z],
        )?;
        let (kn_b, kn_o, kn_l) = weights.binding_by_name(&name("self_attn.k_norm.weight"))?;
        let push = rms_norm_params_rows(hd as u32, (t * nkv) as u32, hd as u32, cfg.rms_norm_eps)
            .to_le_bytes();
        let d = rms_norm_dispatch_rows((t * nkv) as u32);
        dev.rec(
            Kernel::RmsNorm,
            Kernel::RmsNorm.specialization_u32(),
            &push,
            &[
                Bind::Ext(&self.buffer, s.kbuf, (t * kv_dim * 4) as u64),
                Bind::Ext(kn_b, kn_o, kn_l),
                Bind::Ext(&self.buffer, s.kbuf, (t * kv_dim * 4) as u64),
            ],
            [d.x, d.y, d.z],
        )?;
        dev.barrier();
        // Batched partial NeoX RoPE: every head of token `i` rotates by
        // `pos[i]` from the staged position row.
        for (heads, off) in [(nq, s.wb), (nkv, s.kbuf)] {
            let push = rope_neox_params_batched(
                hd as u32,
                cfg.rotary_dim as u32,
                heads as u32,
                t as u32,
                cfg.rope_theta,
            )
            .to_le_bytes();
            let d = rope_neox_dispatch_batched(cfg.rotary_dim as u32, heads as u32, t as u32);
            let block = (t * heads * hd * 4) as u64;
            dev.rec(
                Kernel::RopeNeox,
                Kernel::RopeNeox.specialization_u32(),
                &push,
                &[
                    Bind::Ext(&self.buffer, off, block),
                    Bind::Ext(&self.buffer, s.pos, (t * 4) as u64),
                    Bind::A(dev.slots.dummy, 8),
                    Bind::Ext(&self.buffer, off, block),
                    Bind::Ext(&self.buffer, s.pos, (t * 4) as u64),
                ],
                [d.x, d.y, d.z],
            )?;
        }
        dev.barrier();
        // f16 KV pack: one kv head's `t` strided rows land contiguously at
        // `[start_pos .. start_pos + t]` in its plane.
        let pack_push =
            f16_kv_pack_params_rows(hd as u32, t as u32, kv_dim as u32, hd as u32).to_le_bytes();
        let pd = f16_kv_pack_dispatch_rows(hd as u32, t as u32);
        for kvh in 0..nkv {
            for (src_base, dst_row) in [(s.kbuf, k_rows[kvh]), (s.vbuf, v_rows[kvh])] {
                dev.rec(
                    Kernel::F16KvPack,
                    Kernel::F16KvPack.specialization_u32(),
                    &pack_push,
                    &[
                        Bind::Ext(
                            &self.buffer,
                            src_base + (kvh * hd * 4) as u64,
                            (((t - 1) * kv_dim + hd) * 4) as u64,
                        ),
                        Bind::Kv(dst_row, (t * hd * 2) as u64),
                    ],
                    [pd.x, pd.y, pd.z],
                )?;
            }
        }
        dev.barrier();
        // ONE masked flash dispatch for the whole chunk: grid (t, nq), K/V
        // bound as the layer's whole plane slabs, GQA via nek2 = nkv.
        let scale = 1.0f32 / (hd as f32).sqrt();
        let spec = FlashAttentionSpec::f32_f16_masked(hd as u32, hd as u32);
        let push = flash_attn_params_batched(
            hd as u32,
            hd as u32,
            t as u32,
            kv_len as u32,
            nq as u32,
            nkv as u32,
            u32::try_from(plane_bytes)?,
            u32::try_from(plane_bytes)?,
            scale,
        )
        .to_le_bytes();
        let fd = flash_attn_dispatch_batched(t as u32, nq as u32);
        dev.rec(
            Kernel::FlashAttn,
            spec.specialization_u32(),
            &push,
            &[
                Bind::Ext(&self.buffer, s.wb, (t * q_dim * 4) as u64),
                Bind::Kv(k_base, layer_bytes),
                Bind::Kv(v_base, layer_bytes),
                Bind::Ext(&self.buffer, s.mask, (t * kv_len * 2) as u64),
                Bind::A(dev.slots.dummy, 8),
                Bind::Ext(&self.buffer, s.wc, (t * q_dim * 4) as u64),
                Bind::A(dev.slots.dummy, 8),
            ],
            [fd.x, fd.y, fd.z],
        )?;
        dev.barrier();
        // Strided sigmoid gate off the interleaved [query|gate] projection.
        let push =
            sigmoid_mul_params_strided((t * q_dim) as u32, hd as u32, (2 * hd) as u32, hd as u32)
                .to_le_bytes();
        let d = sigmoid_mul_dispatch((t * q_dim) as u32);
        dev.rec(
            Kernel::SigmoidMul,
            Kernel::SigmoidMul.specialization_u32(),
            &push,
            &[
                Bind::Ext(&self.buffer, s.wa, (t * nq * 2 * hd * 4) as u64),
                Bind::Ext(&self.buffer, s.wc, (t * q_dim * 4) as u64),
                Bind::Ext(&self.buffer, s.wc, (t * q_dim * 4) as u64),
            ],
            [d.x, d.y, d.z],
        )?;
        dev.barrier();
        let out_packed = if let Some((_, pack, _)) = pf_gemm_route(dev.ctx, o_w.format) {
            self.record_pack(dev, pack, s.wc, t * q_dim)?;
            dev.barrier();
            true
        } else {
            false
        };
        self.record_dense_chunk(dev, weights, &o_w, s.wc, q_dim, out_packed, s.y, h, t)
    }

    /// The MoE block over the chunk, FENCE-FREE by default: per-token routers
    /// + top-k and the (ids-independent) shared expert record first, then the
    /// device planner turns the resident ids into the grouped-MoE plan, and
    /// the expert tails dispatch through `vkCmdDispatchIndirect` against it —
    /// all in the same recorded stream. `ARLE_QWEN4_PREFILL_HOST_PLAN=1` (or
    /// the `moe_ids` capture) restores the fenced host planning.
    fn record_moe(
        &mut self,
        dev: &mut Qwen4Dev<'ctx>,
        weights: &Qwen4Weights<'_, '_>,
        cfg: &Qwen4ExpConfig,
        layer: usize,
        t: usize,
    ) -> Result<()> {
        let _s = prof::stage("pf.moe");
        let h = cfg.hidden_size;
        let top_k = cfg.num_experts_per_tok;
        let inter = cfg.moe_intermediate_size;
        let sh_inter = cfg.shared_expert_intermediate_size;
        let s = self.slots;
        let name = |sfx: &str| layer_tensor_name(layer, sfx);
        let router = *weights.tensor(&name("mlp.gate.weight"))?;
        ensure!(
            router.format == Qwen4DeviceFormat::F32,
            "mlp.gate must be F32-resident"
        );
        let shg_w = *weights.tensor(&name("mlp.shared_expert_gate.weight"))?;
        ensure!(
            shg_w.format == Qwen4DeviceFormat::F32,
            "shared_expert_gate must be F32-resident (its sigmoid rides the kernel)"
        );
        let sh_gate = *weights.tensor(&name("mlp.shared_expert.gate_proj.weight"))?;
        let sh_up = *weights.tensor(&name("mlp.shared_expert.up_proj.weight"))?;
        let sh_down = *weights.tensor(&name("mlp.shared_expert.down_proj.weight"))?;

        prof::phase("pf.moe.router");
        // Router + scalar shared gate per token, plus the B staging for the
        // shared projections — all read only the block input.
        let (rb, ro, rl_) = weights.binding(&router)?;
        let r_push =
            qwen36_router_gemv_params(cfg.num_experts as u32, h as u32, false).to_le_bytes();
        let rd = qwen36_router_gemv_dispatch(cfg.num_experts as u32);
        let (gb, go, gl) = weights.binding(&shg_w)?;
        let g_push = qwen36_router_gemv_params(1, h as u32, true).to_le_bytes();
        let gd = qwen36_router_gemv_dispatch(1);
        for tok in 0..t {
            let x_row = Bind::Ext(&self.buffer, Self::row(s.x, tok, h), (h * 4) as u64);
            dev.rec(
                Kernel::Qwen36RouterGemv,
                Kernel::Qwen36RouterGemv.specialization_u32(),
                &r_push,
                &[
                    Bind::Ext(&self.buffer, Self::row(s.x, tok, h), (h * 4) as u64),
                    Bind::Ext(rb, ro, rl_),
                    Bind::Ext(
                        &self.buffer,
                        Self::row(s.lgt, tok, cfg.num_experts),
                        (cfg.num_experts * 4) as u64,
                    ),
                ],
                [rd.x, rd.y, rd.z],
            )?;
            dev.rec(
                Kernel::Qwen36RouterGemv,
                Kernel::Qwen36RouterGemv.specialization_u32(),
                &g_push,
                &[
                    x_row,
                    Bind::Ext(gb, go, gl),
                    Bind::Ext(&self.buffer, Self::row(s.shg, tok, PF_IDS_PAD), 4),
                ],
                [gd.x, gd.y, gd.z],
            )?;
        }
        let sh_packed = if let Some((_, pack, _)) = pf_gemm_route(dev.ctx, sh_gate.format) {
            self.record_pack(dev, pack, s.x, t * h)?;
            true
        } else {
            false
        };
        dev.barrier();
        prof::phase("pf.moe.topk");
        let tk_push =
            qwen36_router_topk_params(cfg.num_experts as u32, top_k as u32, cfg.norm_topk_prob)
                .to_le_bytes();
        let td = qwen36_router_topk_dispatch();
        for tok in 0..t {
            dev.rec(
                Kernel::Qwen36RouterTopk,
                Kernel::Qwen36RouterTopk.specialization_u32(),
                &tk_push,
                &[
                    Bind::Ext(
                        &self.buffer,
                        Self::row(s.lgt, tok, cfg.num_experts),
                        (cfg.num_experts * 4) as u64,
                    ),
                    Bind::Ext(
                        &self.buffer,
                        Self::row(s.ids, tok, PF_IDS_PAD),
                        (top_k * 4) as u64,
                    ),
                    Bind::Ext(
                        &self.buffer,
                        Self::row(s.wts, tok, PF_IDS_PAD),
                        (top_k * 4) as u64,
                    ),
                ],
                [td.x, td.y, td.z],
            )?;
        }
        prof::phase("pf.moe.shared");
        self.record_dense_chunk(dev, weights, &sh_gate, s.x, h, sh_packed, s.sg, sh_inter, t)?;
        self.record_dense_chunk(
            dev,
            weights,
            &sh_up,
            s.x,
            h,
            sh_packed && sh_up.format == sh_gate.format,
            s.su,
            sh_inter,
            t,
        )?;
        dev.barrier();
        let n_sh = (t * sh_inter) as u32;
        let push = swiglu_params(n_sh).to_le_bytes();
        let d = swiglu_dispatch(n_sh);
        dev.rec(
            Kernel::SwiGlu,
            Kernel::SwiGlu.specialization_u32(),
            &push,
            &[
                Bind::Ext(&self.buffer, s.sg, u64::from(n_sh) * 4),
                Bind::Ext(&self.buffer, s.su, u64::from(n_sh) * 4),
                Bind::Ext(&self.buffer, s.sg, u64::from(n_sh) * 4),
            ],
            [d.x, d.y, d.z],
        )?;
        dev.barrier();
        let down_packed = if let Some((_, pack, _)) = pf_gemm_route(dev.ctx, sh_down.format) {
            self.record_pack(dev, pack, s.sg, t * sh_inter)?;
            dev.barrier();
            true
        } else {
            false
        };
        self.record_dense_chunk(
            dev,
            weights,
            &sh_down,
            s.sg,
            sh_inter,
            down_packed,
            s.sd,
            h,
            t,
        )?;

        // ── Group planning: DEVICE by default — the per-(layer,chunk) ids
        // fence (measured 7.46 s of the 9.27 s chunk-256 prefill wall, 80%)
        // exists only under `ARLE_QWEN4_PREFILL_HOST_PLAN=1` (the oracle
        // fallback) or when the `moe_ids` capture needs the ids on host
        // anyway. Both modes lay the SAME fixed class regions
        // ([`MoePlanLayout`]) so every descriptor offset below is record-time
        // and byte-identical between them; only where the maps/counts come
        // from differs.
        let pairs = t * top_k;
        let layout = MoePlanLayout::new(pairs, cfg.num_experts);
        let e32 = u32::try_from(cfg.num_experts)?;
        let host_plan = pf_host_plan() || moe_ids::enabled();
        let mut n_blocks = [0usize; PF_MOE_COLS_CAP];
        if host_plan {
            // THE chunk ids fence: one read of t * top_k ids after draining
            // everything recorded so far, so the map/list uploads below
            // cannot race the gather that reads them.
            prof::phase("pf.moe.ids_fence");
            dev.flush()?;
            let raw_ids = self.read_i32_at(s.ids, t * PF_IDS_PAD)?;
            if moe_ids::enabled() {
                let mut compact = Vec::with_capacity(t * top_k);
                for row in 0..t {
                    compact.extend_from_slice(&raw_ids[row * PF_IDS_PAD..row * PF_IDS_PAD + top_k]);
                }
                moe_ids::push(layer, compact);
            }
            prof::phase("pf.moe.group");
            let plan = plan_moe_groups(&raw_ids, t, PF_IDS_PAD, top_k, cfg.num_experts)?;
            let map_bytes: Vec<u8> = plan.scatter.iter().flat_map(|x| x.to_le_bytes()).collect();
            self.write_bytes_at(s.smap, &map_bytes)?;
            let id_bytes: Vec<u8> = plan.ids.iter().flat_map(|x| x.to_le_bytes()).collect();
            self.write_bytes_at(s.eids, &id_bytes)?;
            n_blocks = plan.n_blocks;
        } else {
            // Device planner, three dispatches (count -> scan -> emit; the
            // shader headers own the scratch layout): the ids never leave the
            // device, and each class's block count lands as the y extent of a
            // `VkDispatchIndirectCommand` in the `plan` scratch.
            prof::phase("pf.moe.plan");
            let plan_len = u64::from(qwen4_moe_plan_scratch_elems(e32)) * 4;
            let push =
                qwen4_moe_plan_count_params(pairs as u32, top_k as u32, PF_IDS_PAD as u32, e32)
                    .to_le_bytes();
            let d = qwen4_moe_plan_count_dispatch();
            dev.rec(
                Kernel::Qwen4MoePlanCount,
                Kernel::Qwen4MoePlanCount.specialization_u32(),
                &push,
                &[
                    Bind::Ext(&self.buffer, s.ids, (t * PF_IDS_PAD * 4) as u64),
                    Bind::Ext(&self.buffer, s.plan, plan_len),
                ],
                [d.x, d.y, d.z],
            )?;
            dev.barrier();
            // x extents of the indirect triples are the projection out-dims —
            // record-time facts the scan bakes in next to the device counts.
            let push = qwen4_moe_plan_scan_params(e32, inter as u32, h as u32).to_le_bytes();
            let d = qwen4_moe_plan_scan_dispatch();
            dev.rec(
                Kernel::Qwen4MoePlanScan,
                Kernel::Qwen4MoePlanScan.specialization_u32(),
                &push,
                &[Bind::Ext(&self.buffer, s.plan, plan_len)],
                [d.x, d.y, d.z],
            )?;
            dev.barrier();
            let (pair_base, list_base) = layout.push_bases()?;
            let push = qwen4_moe_plan_emit_params(
                pairs as u32,
                top_k as u32,
                PF_IDS_PAD as u32,
                e32,
                pair_base,
                list_base,
            )
            .to_le_bytes();
            let d = qwen4_moe_plan_emit_dispatch(pairs as u32, e32);
            dev.rec(
                Kernel::Qwen4MoePlanEmit,
                Kernel::Qwen4MoePlanEmit.specialization_u32(),
                &push,
                &[
                    Bind::Ext(&self.buffer, s.ids, (t * PF_IDS_PAD * 4) as u64),
                    Bind::Ext(&self.buffer, s.plan, plan_len),
                    Bind::Ext(&self.buffer, s.smap, (pairs * 4) as u64),
                    Bind::Ext(&self.buffer, s.eids, (layout.list_capacity * 4) as u64),
                ],
                [d.x, d.y, d.z],
            )?;
            // Widened barrier: the scan's args must be visible to the grouped
            // GEMVs' DRAW_INDIRECT stage, which a compute-only dst mask does
            // not cover.
            dev.barrier_indirect();
        }
        // `weight_scale_2` comes from the resident id-indexed table (seeded
        // here when decode has not touched this layer yet); the grouped GEMV
        // reads it through the block's expert id. First-touch is what makes
        // this host write race-free with NO fence in front of it: rows that
        // have never been bound cannot be read by any recorded dispatch.
        dev.ensure_scale0_rows(weights, layer)?;

        prof::phase("pf.moe.experts");
        // Gather: write-side scatter of the LIVE pairs into the fixed class
        // regions (`xg[smap[p]] = x[p / top_k]`) — capacity gaps are neither
        // written nor read, so the region sparsity costs no bandwidth.
        let push = qwen4_block_scatter_params(h as u32, pairs as u32, top_k as u32).to_le_bytes();
        let d = qwen4_block_scatter_dispatch(pairs as u32);
        dev.rec(
            Kernel::Qwen4BlockScatter,
            Kernel::Qwen4BlockScatter.specialization_u32(),
            &push,
            &[
                Bind::Ext(&self.buffer, s.x, (t * h * 4) as u64),
                Bind::Ext(&self.buffer, s.smap, (pairs * 4) as u64),
                Bind::Ext(&self.buffer, s.xg, (layout.pair_capacity * h * 4) as u64),
            ],
            [d.x, d.y, d.z],
        )?;
        dev.barrier();
        // The classes worth recording: a zero-CAPACITY class can never have
        // blocks; host mode additionally skips classes it KNOWS are empty,
        // while device mode records them all and lets the indirect y extent
        // (possibly 0) decide on the GPU.
        let classes: Vec<usize> = (1..=PF_MOE_COLS_CAP)
            .rev()
            .filter(|&w| layout.cap_blocks(w) > 0 && (!host_plan || n_blocks[w - 1] > 0))
            .collect();
        // Push-word block bound per class: the exact count when the host
        // knows it, the class CAPACITY otherwise — either satisfies the
        // grouped contract (`expert_i0 % ne11 == expert_i0` needs only
        // bound >= every dispatched block index + 1; words 8/11 are unread).
        let bound = |w: usize| {
            if host_plan {
                n_blocks[w - 1]
            } else {
                layout.cap_blocks(w)
            }
        };
        let drive = |args_at: u32| {
            (!host_plan).then(|| GroupedIndirect {
                buffer: &self.buffer,
                args_offset: s.plan + u64::from(args_at) * 4,
            })
        };
        let list_bind = |w: usize| {
            Bind::Ext(
                &self.buffer,
                s.eids + (layout.list_base(w) * 4) as u64,
                (layout.cap_blocks(w) * 4) as u64,
            )
        };
        // Gate/up over every class: outputs are disjoint expert-major slices,
        // so the whole lot records with no intervening barriers.
        for &w in &classes {
            let rows = w * layout.cap_blocks(w);
            for (proj, dst) in [(ExpertProj::Gate, s.ge), (ExpertProj::Up, s.gu)] {
                let scale0 = dev.scale0_rows_bind(layer, proj);
                dev.gemv_id_nvfp4_grouped(
                    weights,
                    layer,
                    proj,
                    w,
                    bound(w),
                    drive(qwen4_moe_plan_args_gateup_at(e32, w as u32)),
                    h,
                    inter,
                    Bind::Ext(
                        &self.buffer,
                        s.xg + (layout.pair_base(w) * h * 4) as u64,
                        (rows * h * 4) as u64,
                    ),
                    Bind::Ext(
                        &self.buffer,
                        dst + (layout.pair_base(w) * inter * 4) as u64,
                        (rows * inter * 4) as u64,
                    ),
                    scale0,
                    list_bind(w),
                )?;
            }
        }
        dev.barrier();
        // SwiGLU over the whole class-region span — elementwise, so the
        // capacity gaps it also touches hold garbage-in/garbage-out values
        // nothing ever reads (the ~6x span costs ~0.1 ms/layer-chunk of
        // bandwidth, noise next to the expert GEMVs).
        let n_act = u32::try_from(layout.pair_capacity * inter)?;
        let push = swiglu_params(n_act).to_le_bytes();
        let d = swiglu_dispatch(n_act);
        dev.rec(
            Kernel::SwiGlu,
            Kernel::SwiGlu.specialization_u32(),
            &push,
            &[
                Bind::Ext(&self.buffer, s.ge, u64::from(n_act) * 4),
                Bind::Ext(&self.buffer, s.gu, u64::from(n_act) * 4),
                Bind::Ext(&self.buffer, s.ge, u64::from(n_act) * 4),
            ],
            [d.x, d.y, d.z],
        )?;
        dev.barrier();
        // Down over the same classes: B is the expert-major SwiGLU output,
        // already block-contiguous — the chain never leaves expert order.
        for &w in &classes {
            let rows = w * layout.cap_blocks(w);
            let scale0 = dev.scale0_rows_bind(layer, ExpertProj::Down);
            dev.gemv_id_nvfp4_grouped(
                weights,
                layer,
                ExpertProj::Down,
                w,
                bound(w),
                drive(qwen4_moe_plan_args_down_at(e32, w as u32)),
                inter,
                h,
                Bind::Ext(
                    &self.buffer,
                    s.ge + (layout.pair_base(w) * inter * 4) as u64,
                    (rows * inter * 4) as u64,
                ),
                Bind::Ext(
                    &self.buffer,
                    s.edc + (layout.pair_base(w) * h * 4) as u64,
                    (rows * h * 4) as u64,
                ),
                scale0,
                list_bind(w),
            )?;
        }
        dev.barrier();
        // Scatter: expert-major down outputs back to the `[tok][slot]` rows,
        // through the READ-side permutation over the same map (`edown[p] =
        // edc[smap[p]]`). `weight_scale_2` was already fused in the GEMV
        // (SCALE0, as decode fuses it), so this is a pure permutation and the
        // weighted accumulate below runs UNCHANGED — decode's slot-order
        // summation is preserved bit for bit.
        self.record_perm_dyn(
            dev,
            s.edc,
            layout.pair_capacity * h,
            s.smap,
            h,
            pairs,
            s.edown,
            pairs * h,
        )?;
        dev.barrier();
        prof::phase("pf.moe.accum");
        let acc_push = qwen36_moe_weighted_accum_params(h as u32, top_k as u32, true).to_le_bytes();
        let ad = qwen36_moe_weighted_accum_dispatch(h as u32);
        for tok in 0..t {
            dev.rec(
                Kernel::Qwen36MoeWeightedAccum,
                Kernel::Qwen36MoeWeightedAccum.specialization_u32(),
                &acc_push,
                &[
                    Bind::Ext(
                        &self.buffer,
                        Self::row(s.edown, tok, top_k * h),
                        (top_k * h * 4) as u64,
                    ),
                    Bind::Ext(
                        &self.buffer,
                        Self::row(s.wts, tok, PF_IDS_PAD),
                        (top_k * 4) as u64,
                    ),
                    Bind::Ext(&self.buffer, Self::row(s.acc, tok, h), (h * 4) as u64),
                ],
                [ad.x, ad.y, ad.z],
            )?;
        }
        dev.barrier();
        let sh_push = qwen36_moe_weighted_accum_params(h as u32, 1, false).to_le_bytes();
        for tok in 0..t {
            dev.rec(
                Kernel::Qwen36MoeWeightedAccum,
                Kernel::Qwen36MoeWeightedAccum.specialization_u32(),
                &sh_push,
                &[
                    Bind::Ext(&self.buffer, Self::row(s.sd, tok, h), (h * 4) as u64),
                    Bind::Ext(&self.buffer, Self::row(s.shg, tok, PF_IDS_PAD), 4),
                    Bind::Ext(&self.buffer, Self::row(s.acc, tok, h), (h * 4) as u64),
                ],
                [ad.x, ad.y, ad.z],
            )?;
        }
        prof::end_phase();
        Ok(())
    }

    /// One chunk end to end: host staging, the layer walk (ONE recorded
    /// stream — the device-planned MoE needs no intra-chunk fence), the
    /// end-of-chunk flush, and the PLE ring hand-back. Leaves each token's
    /// final hyper residual in the `h` rows.
    #[expect(
        clippy::too_many_arguments,
        reason = "the chunk driver owns the split state"
    )]
    fn run_chunk(
        &mut self,
        dev: &mut Qwen4Dev<'ctx>,
        weights: &Qwen4Weights<'_, '_>,
        rl: &DevResidentLinAttn<'ctx>,
        cfg: &Qwen4ExpConfig,
        hc: &HyperConnectionConfig,
        state: &mut Qwen4ExpState,
        seed: &[f32],
        ple_emb: &[f32],
        t: usize,
        start_pos: usize,
    ) -> Result<()> {
        ensure!(t >= 1 && t <= self.max_tokens, "chunk of {t} tokens");
        ensure!(seed.len() == t * hc.hc_hidden(), "seed length");
        let s = self.slots;
        let kv_len = start_pos + t;

        {
            let _st = prof::stage("pf.stage_host");
            self.write_f32_at(s.h, seed)?;
            let pos_bytes: Vec<u8> = (0..t)
                .flat_map(|i| ((start_pos + i) as i32).to_le_bytes())
                .collect();
            self.write_bytes_at(s.pos, &pos_bytes)?;
            self.write_bytes_at(s.mask, &pf_causal_mask(t, kv_len, start_pos))?;
            if !ple_emb.is_empty() {
                self.write_f32_at(s.emb, ple_emb)?;
            }
            // Host-canonical PLE ring (oldest-first == ring_pos 0) into the
            // decode arena's slot; the seq-mode conv advances it in place.
            // `one` feeds the alias-safe `h += 1.0 * ple_out` accumulate.
            dev.write_f32(dev.slots.one, &[1.0])?;
            for (_, ring) in state.ple_conv.iter() {
                dev.write_f32(dev.slots.ple_ring, ring.rows())?;
            }
        }

        for layer in 0..cfg.num_hidden_layers {
            if cfg.is_ple_layer(layer) {
                ensure!(!ple_emb.is_empty(), "PLE layer {layer} with no embeddings");
                self.record_ple(dev, weights, cfg, layer, t)?;
                dev.barrier();
            }
            self.record_hc_pre(dev, weights, hc, Some(layer), HcSite::Attn, t)?;
            dev.barrier();
            match cfg.layer_types[layer] {
                Qwen4LayerType::LinearAttention => {
                    self.record_linear(dev, weights, rl, cfg, layer, t)?;
                }
                Qwen4LayerType::FullAttention => {
                    self.record_full(dev, weights, cfg, layer, t, start_pos)?;
                }
            }
            dev.barrier();
            self.record_hc_combine(
                dev,
                weights,
                hc,
                Some(layer),
                HcSite::Attn,
                s.y,
                cfg.hidden_size,
                t,
            )?;
            dev.barrier();
            self.record_hc_pre(dev, weights, hc, Some(layer), HcSite::Mlp, t)?;
            dev.barrier();
            self.record_moe(dev, weights, cfg, layer, t)?;
            dev.barrier();
            self.record_hc_combine(
                dev,
                weights,
                hc,
                Some(layer),
                HcSite::Mlp,
                s.acc,
                cfg.hidden_size,
                t,
            )?;
            dev.barrier();
        }
        dev.flush()?;

        // The ring after `t` in-place steps: the next write slot is `t %
        // state_len`, which is also the oldest row — one rotation restores the
        // host's oldest-first layout (the `t = 1` case is `finish_ple`'s).
        for (_, ring) in state.ple_conv.iter_mut() {
            let pc = ple_config(cfg);
            let hh = pc.hc_hidden();
            let state_len = pc.short_conv_state_len();
            let dev_ring = dev.read_f32(dev.slots.ple_ring, state_len * hh)?;
            let rows = ring.rows_mut();
            for i in 0..state_len {
                let src = (i + t) % state_len;
                rows[i * hh..(i + 1) * hh].copy_from_slice(&dev_ring[src * hh..(src + 1) * hh]);
            }
        }
        Ok(())
    }

    /// One post-chunk pre-mixer residual row: `h` at chunk index `tok`,
    /// `[hc_hidden]`. Valid after [`Self::run_chunk`]'s flush; the arena is
    /// HOST_CACHED, so a 40 KB row read is microseconds, not the WC trap.
    fn read_h_row(&self, tok: usize, hh: usize) -> Result<Vec<f32>> {
        self.read_f32_at(Self::row(self.slots.h, tok, hh), hh)
    }

    /// Record the VERIFY tail after a chunk: the stream mixer for every one of
    /// the chunk's `t` positions (decode's own kernels, per token) and then
    /// `lm_head` over all `t` block inputs as NUM_COLS-batched GEMVs — the
    /// 1.2 GiB projection read once per chunk instead of once per position.
    /// Logits land row-per-position in [`Self::read_verify_logits`]'s buffer.
    fn record_verify_tail(
        &mut self,
        dev: &mut Qwen4Dev<'ctx>,
        weights: &Qwen4Weights<'_, '_>,
        hc: &HyperConnectionConfig,
        lm: &Qwen4DeviceTensor,
        t: usize,
    ) -> Result<()> {
        let _s = prof::stage("pf.verify_tail");
        ensure!(
            (1..=QWEN4_VERIFY_MAX_TOKENS).contains(&t),
            "verify tail over {t} positions (cap {QWEN4_VERIFY_MAX_TOKENS})"
        );
        let (k, m) = (lm.ncols, lm.nrows);
        ensure!(k == hc.hidden_size, "lm_head ncols {k} != hidden");
        if self.verify_logits.is_none() {
            let bytes = QWEN4_VERIFY_MAX_TOKENS * m * 4;
            self.verify_logits = Some(
                DeviceBuffer::alloc_host_cached(dev.ctx, bytes)
                    .map_err(|e| anyhow!("alloc verify logits ({bytes} B): {e}"))?,
            );
        }
        // Mixer per position: writes the block input rows at `slots.x`.
        self.record_hc_pre(dev, weights, hc, None, HcSite::Mixer, t)?;
        dev.barrier();
        let logits_buf = self.verify_logits.as_ref().expect("allocated above");
        let cols_kernel = match lm.format {
            Qwen4DeviceFormat::Bf16 => Kernel::GemvBf16,
            Qwen4DeviceFormat::F16 => Kernel::GemvF16,
            Qwen4DeviceFormat::Q4K => Kernel::GemvQ4KDense,
            Qwen4DeviceFormat::Q8_0 => Kernel::GemvQ8_0Dense,
            f => bail!("verify tail: lm_head in unsupported format {f:?}"),
        };
        let (wb, wo, wl) = weights.binding(lm)?;
        let push = gemv_params_f32_b_cols(u32::try_from(k)?, u32::try_from(m)?).to_le_bytes();
        let cap = qwen4_gemv_cols_cap();
        let mut tok = 0usize;
        while tok < t {
            let g = (t - tok).min(cap);
            let spec = cols_kernel
                .gemv_cols_spec(u32::try_from(g)?)
                .expect("dense GEMV kernels carry a cols spec");
            let d = gemv_dense_dispatch(u32::try_from(m)?, &spec);
            dev.rec(
                cols_kernel,
                spec.specialization_u32(),
                &push,
                &[
                    Bind::Ext(wb, wo, wl),
                    Bind::Ext(
                        &self.buffer,
                        Self::row(self.slots.x, tok, k),
                        (g * k * 4) as u64,
                    ),
                    Bind::Ext(logits_buf, (tok * m * 4) as u64, (g * m * 4) as u64),
                    Bind::A(dev.slots.dummy, 8),
                    Bind::A(dev.slots.dummy, 8),
                ],
                [d.x, d.y, d.z],
            )?;
            tok += g;
        }
        Ok(())
    }

    /// The logits row of chunk position `tok` after a
    /// [`Self::record_verify_tail`] flush.
    fn read_verify_logits(&self, tok: usize, vocab: usize) -> Result<Vec<f32>> {
        let buf = self
            .verify_logits
            .as_ref()
            .ok_or_else(|| anyhow!("no verify tail was recorded"))?;
        let _p = prof::span_bytes("d2h", (vocab * 4) as u64);
        let mut bytes = vec![0u8; vocab * 4];
        buf.copy_to_host_at((tok * vocab * 4) as u64, &mut bytes)
            .map_err(|e| anyhow!("verify logits read at {tok}: {e}"))?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }
}

impl<'ctx, 'st> VulkanQwen4ExpModel<'ctx, 'st> {
    /// Why [`Self::forward_prompt`] would refuse, or `None` when it can run.
    /// The batched path requires every stage of every layer device-resident
    /// (the same bar as the staged decode loop, minus `lm_head` and the
    /// mixer, which fall back to their decode routes for the one final token).
    #[must_use]
    pub fn prefill_unsupported_reason(&self) -> Option<String> {
        let (Some(d), Some(w), Some(rl)) = (
            self.dev.as_ref(),
            self.weights.as_ref(),
            self.resident_linear.as_ref(),
        ) else {
            return Some("no device residency (host-only mode)".into());
        };
        if self.cfg.ple_layer_ids.len() > 1 {
            return Some("more than one PLE layer (single ring slot)".into());
        }
        for l in 0..self.cfg.num_hidden_layers {
            for site in [HcSite::Attn, HcSite::Mlp] {
                if let Err(e) = w.hyper_connection(Some(l), site) {
                    return Some(format!("layer {l}: {e}"));
                }
            }
            if let Err(e) = w.expert_stack(l, ExpertProj::Gate) {
                return Some(format!("layer {l}: {e}"));
            }
            for sfx in [
                "mlp.gate.weight",
                "mlp.shared_expert_gate.weight",
                "mlp.shared_expert.gate_proj.weight",
                "mlp.shared_expert.up_proj.weight",
                "mlp.shared_expert.down_proj.weight",
            ] {
                match w.tensor(&layer_tensor_name(l, sfx)) {
                    Err(e) => return Some(format!("layer {l}: {e}")),
                    Ok(t) if t.format == Qwen4DeviceFormat::Nvfp4 => {
                        return Some(format!("layer {l}: `{sfx}` is NVFP4"));
                    }
                    Ok(_) => {}
                }
            }
            match self.cfg.layer_types[l] {
                Qwen4LayerType::LinearAttention => {
                    if !rl.covers(l) {
                        return Some(format!("layer {l}: linear attention not resident"));
                    }
                }
                Qwen4LayerType::FullAttention => {
                    if !d.full_attention_ready(w, l) {
                        return Some(format!("layer {l}: full attention not device-ready"));
                    }
                }
            }
            if self.cfg.is_ple_layer(l) {
                for sfx in ["ple.key_proj.weight", "ple.value_proj.weight"] {
                    if let Err(e) = w.tensor(&layer_tensor_name(l, sfx)) {
                        return Some(format!("layer {l}: {e}"));
                    }
                }
            }
        }
        None
    }

    /// Batched prefill of `tokens` at the default chunk width
    /// ([`qwen4_prefill_chunk_tokens`]). Returns the LAST token's logits —
    /// the value the per-token loop's final [`Self::forward_token`] would
    /// return — and advances every piece of recurrent state (device GDN S +
    /// conv rings, KV planes, PLE ring, n-gram context, `seq_len`) exactly as
    /// `tokens.len()` decode steps would. `tests/qwen4_prefill.rs` pins that
    /// equality; a prefill that diverges from decode is a wrong prefill.
    pub fn forward_prompt(
        &mut self,
        slot: usize,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<Vec<f32>> {
        self.forward_prompt_chunked(slot, tokens, start_pos, qwen4_prefill_chunk_tokens())
    }

    /// The executor's batched entry point: `None` when the batched path is
    /// unavailable (the caller falls back to the per-token loop).
    pub fn forward_tokens(
        &mut self,
        slot: usize,
        _epoch: u64,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<Option<Vec<f32>>> {
        if tokens.len() < 2 {
            return Ok(None);
        }
        if let Some(reason) = self.prefill_unsupported_reason() {
            log::info!("qwen4_exp batched prefill unavailable: {reason}");
            return Ok(None);
        }
        self.forward_prompt(slot, tokens, start_pos).map(Some)
    }

    /// [`Self::forward_prompt`] at an explicit chunk width (the equivalence
    /// test drives uneven multi-chunk splits through this).
    pub fn forward_prompt_chunked(
        &mut self,
        slot: usize,
        tokens: &[u32],
        start_pos: usize,
        chunk_width: usize,
    ) -> Result<Vec<f32>> {
        ensure!(
            slot == 0,
            "qwen4_exp Vulkan lane is single-slot (got slot {slot})"
        );
        ensure!(!tokens.is_empty(), "forward_prompt with no tokens");
        for &tok in tokens {
            ensure!(
                (tok as usize) < self.cfg.vocab_size,
                "token {tok} outside the vocab"
            );
        }
        ensure!(
            start_pos + tokens.len() <= self.cfg.max_context,
            "prompt end {} > max_context {} (the cap that keeps the QSA dense stub exact)",
            start_pos + tokens.len(),
            self.cfg.max_context
        );
        if let Some(reason) = self.prefill_unsupported_reason() {
            bail!("qwen4_exp batched prefill unavailable: {reason}");
        }
        if start_pos == 0 && self.state.seq_len != 0 {
            self.state = Qwen4ExpState::new(&self.cfg, &self.hash);
            if let Some(rl) = self.resident_linear.as_mut() {
                rl.reset()?;
            }
            if let Some(d) = self.dev.as_mut() {
                // Fresh sequence: the resident PLE ring is garbage history.
                d.invalidate_ple_ring();
            }
        }
        ensure!(
            start_pos == self.state.seq_len,
            "forward_prompt at {start_pos} but the state holds {} tokens",
            self.state.seq_len
        );
        // A staged decode may hold the PLE ring on device; batched prefill
        // reads and re-seeds the host rows per chunk, so sync out first.
        if let Some(d) = self.dev.as_mut() {
            if let Some((l, rows)) = d.read_ple_ring(&self.cfg)? {
                if let Some(ring) = self.state.ple_conv.get_mut(&l) {
                    ring.rows_mut().copy_from_slice(&rows);
                }
            }
        }
        let width = chunk_width.clamp(1, 1024);
        if self.prefill.as_ref().is_none_or(|p| p.max_tokens() < width) {
            let ctx = self.dev.as_ref().expect("checked resident").ctx;
            self.prefill = Some(Qwen4Prefill::new(ctx, &self.cfg, &self.hc, width)?);
        }

        let _prompt = prof::stage("prompt");
        let t0 = std::time::Instant::now();
        let mut pos = start_pos;
        for chunk in tokens.chunks(width) {
            let t = chunk.len();
            let toks_i64: Vec<i64> = chunk.iter().map(|&v| i64::from(v)).collect();
            let ple_emb = if self.cfg.ple_layer_ids.is_empty() {
                Vec::new()
            } else {
                let _g = prof::stage("pf.ngram_gather");
                let ids = self.hash.row_ids(&self.state.ngram, &toks_i64)?;
                self.gather_ple_embedding(&ids)?
            };
            self.state.ngram.push(&toks_i64);
            let seed = {
                let _e = prof::stage("pf.embed_seed");
                let mut seed = Vec::with_capacity(t * self.hc.hc_hidden());
                for &tok in chunk {
                    let embed = self.tables.embed_row(tok as usize)?;
                    seed.extend(qwen4_hc::seed_hyper_state(&self.hc, &embed)?);
                }
                seed
            };
            let Self {
                cfg,
                hc,
                dev,
                weights,
                resident_linear,
                state,
                prefill,
                ..
            } = self;
            let d = dev.as_mut().expect("checked resident");
            let w = weights.as_ref().expect("checked resident");
            let rl = resident_linear.as_ref().expect("checked resident");
            prefill
                .as_mut()
                .expect("built above")
                .run_chunk(d, w, rl, cfg, hc, state, &seed, &ple_emb, t, pos)?;
            state.seq_len += t;
            pos += t;
        }
        let secs = t0.elapsed().as_secs_f64();
        log::info!(
            "qwen4_exp batched prefill: {} tok @ {start_pos} in {secs:.3}s ({:.1} tok/s, chunk={width})",
            tokens.len(),
            tokens.len() as f64 / secs.max(f64::MIN_POSITIVE),
        );

        // Tail: mixer + lm_head for the LAST token only, through the decode
        // routes (device when resident, host otherwise — same split as
        // `forward_token`'s tail).
        let _tail = prof::stage("pf.tail");
        let t_last = (tokens.len() - 1) % width; // index within the final chunk
        let hh = self.hc.hc_hidden();
        let last_h = {
            let p = self.prefill.as_ref().expect("built above");
            p.read_f32_at(Qwen4Prefill::row(p.slots.h, t_last, hh), hh)?
        };
        self.mixer_lm_head(&last_h)
    }

    /// The resident linear-attention state, for the equivalence harness.
    #[must_use]
    pub fn resident_linear(&self) -> Option<&DevResidentLinAttn<'ctx>> {
        self.resident_linear.as_ref()
    }

    /// The device runner (immutable), for the equivalence harness.
    #[must_use]
    pub fn dev_ref(&self) -> Option<&Qwen4Dev<'ctx>> {
        self.dev.as_ref()
    }

    /// The device route pair the MTP head's forward takes — for harnesses
    /// that drive [`crate::qwen4_mtp::MtpHead::forward`] directly.
    pub fn dev_and_weights(&mut self) -> Option<(&mut Qwen4Dev<'ctx>, &Qwen4Weights<'ctx, 'st>)> {
        self.dev.as_mut().zip(self.weights.as_ref())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Speculative decode: batched verify + state rollback.
//
// The MTP head (`crate::qwen4_mtp`) DRAFTS k tokens; the target model verifies
// pending ++ drafts in ONE prefill chunk (grouped experts, NUM_COLS-batched
// dense — the dense bytes divide by the chunk length) and greedy acceptance
// keeps the longest prefix whose target argmax equals the draft. Rollback is
// the whole trick on a recurrent model: the chunk ADVANCES the gated-delta S,
// the conv rings, the PLE ring, the KV rows and the n-gram window for every
// chunk position, accepted or not. The contract here:
//
// - KV rows and RoPE positions are POSITIONAL — rows past the kept `seq_len`
//   are never read and the next chunk rewrites them, so KV "rollback" is just
//   not advancing `seq_len`.
// - GDN S + conv rings are device-resident and snapshot/restored by one
//   recorded buffer copy each way (`DevResidentLinAttn::record_state_save` /
//   `record_state_restore`).
// - The PLE ring and the n-gram window are host-side and clone-restored.
// - A rejected suffix is NOT replayed on its own: the accepted-but-rolled-back
//   tokens simply ride the FRONT of the next cycle's chunk (`pending`), whose
//   recomputation is deterministic — one weight sweep per cycle, always.
//
// Greedy speculative output therefore EQUALS plain greedy decode token for
// token BY CONSTRUCTION — the verify chunk runs decode's own kernels (the
// prefill=decode bit-exact gate) and emits only target argmaxes —
// so any mismatch is a rollback bug. `tests/qwen4_speculative.rs` holds that
// gate, plus the fault-injection knob below that proves the gate can fail.
// ─────────────────────────────────────────────────────────────────────────────

/// Per-position outputs of one verify chunk.
pub struct Qwen4VerifyOutput {
    /// Target greedy argmax after each chunk position.
    pub argmax: Vec<u32>,
    /// Pre-mixer residual `[hc_hidden]` at each chunk position — what the MTP
    /// head conditions on.
    pub h_rows: Vec<Vec<f32>>,
}

/// One speculative cycle's outcome.
pub struct Qwen4SpecCycle {
    /// Newly emitted tokens: the accepted drafts plus the bonus token the
    /// verify pass computed after them (always at least 1).
    pub emitted: Vec<u32>,
    /// Accepted draft count `L` (0..=k).
    pub accepted: usize,
    /// Whether the recurrent state was rolled back (`L < k`).
    pub rolled_back: bool,
    /// Wall seconds the rollback itself took (restore submit + host clone
    /// restores); 0 on a full accept.
    pub rollback_s: f64,
    /// The next cycle's `pending`: tokens that are part of the output but not
    /// yet inside the recurrent state.
    pub pending: Vec<u32>,
    /// `(absolute position, pre-mixer h)` for the last accepted region — the
    /// drafter's catch-up inputs; the LAST entry is the `h` the next draft
    /// step 1 conditions on.
    pub h_rows: Vec<(usize, Vec<f32>)>,
}

/// Greedy argmax with `infer_plan::sample_token`'s tie-break (first index
/// wins), so the speculative driver and the engine sampler cannot disagree on
/// a razor-thin tie.
#[must_use]
pub fn greedy_argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

/// Rollback fault injection for the equivalence gate, so the gate is proven
/// able to fail: `ARLE_QWEN4_SPEC_FAULT=skip-gdn|skip-ple|skip-ngram` makes
/// [`VulkanQwen4ExpModel::speculative_verify_cycle`] deliberately SKIP that
/// piece of the state restore. Read per rollback (tests toggle it at runtime);
/// unset = no fault, the only production value.
fn spec_fault() -> Option<String> {
    std::env::var("ARLE_QWEN4_SPEC_FAULT")
        .ok()
        .filter(|v| !v.is_empty())
}

impl<'ctx, 'st> VulkanQwen4ExpModel<'ctx, 'st> {
    /// Run `tokens` (= pending ++ drafts) as ONE verify chunk from
    /// `start_pos`, advancing every piece of recurrent state exactly like
    /// [`Self::forward_prompt`], and return each position's target argmax and
    /// pre-mixer residual. The per-position tail runs the stream mixer with
    /// decode's kernels and `lm_head` as a NUM_COLS-batched GEMV when both are
    /// resident, and falls back to [`Self::mixer_lm_head`] per position
    /// otherwise (the subset harness) — the same value either way.
    pub fn forward_verify(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<Qwen4VerifyOutput> {
        ensure!(!tokens.is_empty(), "verify with no tokens");
        ensure!(
            tokens.len() <= QWEN4_VERIFY_MAX_TOKENS,
            "verify chunk of {} tokens (cap {QWEN4_VERIFY_MAX_TOKENS})",
            tokens.len()
        );
        for &tok in tokens {
            ensure!(
                (tok as usize) < self.cfg.vocab_size,
                "token {tok} outside the vocab"
            );
        }
        ensure!(
            start_pos + tokens.len() <= self.cfg.max_context,
            "verify end {} > max_context {}",
            start_pos + tokens.len(),
            self.cfg.max_context
        );
        if let Some(reason) = self.prefill_unsupported_reason() {
            bail!("qwen4_exp verify unavailable: {reason}");
        }
        ensure!(
            start_pos == self.state.seq_len,
            "verify at {start_pos} but the state holds {} tokens",
            self.state.seq_len
        );
        let t = tokens.len();
        let width = qwen4_prefill_chunk_tokens().max(t);
        if self.prefill.as_ref().is_none_or(|p| p.max_tokens() < t) {
            let ctx = self.dev.as_ref().expect("checked resident").ctx;
            self.prefill = Some(Qwen4Prefill::new(ctx, &self.cfg, &self.hc, width)?);
        }

        let _stage = prof::stage("spec.verify");
        let toks_i64: Vec<i64> = tokens.iter().map(|&v| i64::from(v)).collect();
        let ple_emb = if self.cfg.ple_layer_ids.is_empty() {
            Vec::new()
        } else {
            let ids = self.hash.row_ids(&self.state.ngram, &toks_i64)?;
            self.gather_ple_embedding(&ids)?
        };
        self.state.ngram.push(&toks_i64);
        let seed = {
            let mut seed = Vec::with_capacity(t * self.hc.hc_hidden());
            for &tok in tokens {
                let embed = self.tables.embed_row(tok as usize)?;
                seed.extend(qwen4_hc::seed_hyper_state(&self.hc, &embed)?);
            }
            seed
        };
        {
            let Self {
                cfg,
                hc,
                dev,
                weights,
                resident_linear,
                state,
                prefill,
                ..
            } = self;
            let d = dev.as_mut().expect("checked resident");
            let w = weights.as_ref().expect("checked resident");
            let rl = resident_linear.as_ref().expect("checked resident");
            prefill
                .as_mut()
                .expect("built above")
                .run_chunk(d, w, rl, cfg, hc, state, &seed, &ple_emb, t, start_pos)?;
            state.seq_len += t;
        }

        let hh = self.hc.hc_hidden();
        let mut h_rows = Vec::with_capacity(t);
        for tok in 0..t {
            h_rows.push(
                self.prefill
                    .as_ref()
                    .expect("built above")
                    .read_h_row(tok, hh)?,
            );
        }

        // Batched tail when the mixer and `lm_head` are device-resident (the
        // shipping residency); the per-position decode tail otherwise.
        let lm_resident = self
            .weights
            .as_ref()
            .and_then(|w| w.tensor(&self.lm_head.name).ok().copied())
            .filter(|lm| lm.ncols == self.cfg.hidden_size && lm.nrows == self.cfg.vocab_size);
        let mixer_resident = self
            .weights
            .as_ref()
            .is_some_and(|w| w.hyper_connection(None, HcSite::Mixer).is_ok());
        let mut argmax = Vec::with_capacity(t);
        match lm_resident {
            Some(lm) if mixer_resident => {
                let Self {
                    hc,
                    dev,
                    weights,
                    prefill,
                    ..
                } = self;
                let d = dev.as_mut().expect("checked resident");
                let w = weights.as_ref().expect("checked resident");
                let p = prefill.as_mut().expect("built above");
                p.record_verify_tail(d, w, hc, &lm, t)?;
                d.flush()?;
                for tok in 0..t {
                    let logits = self
                        .prefill
                        .as_ref()
                        .expect("built above")
                        .read_verify_logits(tok, self.cfg.vocab_size)?;
                    argmax.push(greedy_argmax(&logits));
                }
            }
            _ => {
                for row in &h_rows {
                    let row = row.clone();
                    let logits = self.mixer_lm_head(&row)?;
                    argmax.push(greedy_argmax(&logits));
                }
            }
        }
        Ok(Qwen4VerifyOutput { argmax, h_rows })
    }

    /// One speculative cycle: verify `pending ++ drafts` in one chunk, accept
    /// the longest matching draft prefix, and either keep the advanced state
    /// (full accept) or roll every recurrent piece back to the pre-chunk
    /// snapshot — the accepted-but-rolled-back tokens then ride the front of
    /// the next cycle's chunk. See the section docs above for why this is
    /// lossless under greedy acceptance.
    pub fn speculative_verify_cycle(
        &mut self,
        pending: &[u32],
        drafts: &[u32],
    ) -> Result<Qwen4SpecCycle> {
        ensure!(
            !pending.is_empty(),
            "speculative cycle with no pending token"
        );
        ensure!(!drafts.is_empty(), "speculative cycle with no drafts");
        let start_pos = self.state.seq_len;

        // Pre-chunk snapshot: host pieces by clone, device GDN state by a
        // recorded copy that rides the chunk's own submit.
        let snap_ngram = self.state.ngram.clone();
        let snap_ple: BTreeMap<usize, PleConvState> = self.state.ple_conv.clone();
        {
            let _s = prof::stage("spec.save");
            let ctx = self
                .dev
                .as_ref()
                .ok_or_else(|| anyhow!("speculative decode needs device residency"))?
                .ctx;
            let rl = self
                .resident_linear
                .as_mut()
                .ok_or_else(|| anyhow!("speculative decode needs resident linear attention"))?;
            rl.ensure_snapshot(ctx)?;
            let d = self.dev.as_mut().expect("checked above");
            self.resident_linear
                .as_ref()
                .expect("checked above")
                .record_state_save(d)?;
        }

        let tokens: Vec<u32> = pending.iter().chain(drafts.iter()).copied().collect();
        let out = self.forward_verify(&tokens, start_pos)?;

        let p = pending.len();
        let mut accepted = 0usize;
        while accepted < drafts.len() && out.argmax[p - 1 + accepted] == drafts[accepted] {
            accepted += 1;
        }
        let bonus = out.argmax[p - 1 + accepted];
        let mut emitted: Vec<u32> = drafts[..accepted].to_vec();
        emitted.push(bonus);

        let rolled_back = accepted < drafts.len();
        let mut rollback_s = 0.0f64;
        let pending_next = if rolled_back {
            let _s = prof::stage("spec.rollback");
            let t0 = std::time::Instant::now();
            let fault = spec_fault();
            let skip = |what: &str| fault.as_deref() == Some(what);
            if !skip("skip-gdn") {
                let d = self.dev.as_mut().expect("checked above");
                self.resident_linear
                    .as_ref()
                    .expect("checked above")
                    .record_state_restore(d)?;
                d.flush()?;
            }
            if !skip("skip-ngram") {
                self.state.ngram = snap_ngram;
            }
            if !skip("skip-ple") {
                self.state.ple_conv = snap_ple;
            }
            // KV + RoPE are positional: winding `seq_len` back IS their
            // rollback (rows past it are never read and get rewritten).
            self.state.seq_len = start_pos;
            let mut next = pending.to_vec();
            next.extend_from_slice(&emitted);
            rollback_s = t0.elapsed().as_secs_f64();
            next
        } else {
            vec![bonus]
        };

        let h_rows = (0..=accepted)
            .map(|j| (start_pos + p - 1 + j, out.h_rows[p - 1 + j].clone()))
            .collect();
        Ok(Qwen4SpecCycle {
            emitted,
            accepted,
            rolled_back,
            rollback_s,
            pending: pending_next,
            h_rows,
        })
    }

    /// Load the MTP drafter's host weights off this model's checkpoint.
    pub fn mtp_drafter(&self) -> Result<crate::qwen4_mtp::MtpDrafter<'st>> {
        Ok(crate::qwen4_mtp::MtpDrafter::new(
            crate::qwen4_mtp::MtpHead::load(self.st, &self.cfg)?,
        ))
    }

    /// Pre-mixer `h` at index `idx` of the LAST prefill/verify chunk (rows of
    /// earlier chunks are overwritten) — the drafter's conditioning inputs.
    pub fn prefill_h_row(&self, idx: usize) -> Result<Vec<f32>> {
        let p = self
            .prefill
            .as_ref()
            .ok_or_else(|| anyhow!("no prefill chunk has run"))?;
        p.read_h_row(idx, self.hc.hc_hidden())
    }

    /// Draft `k` tokens through the MTP head.
    ///
    /// Order of operations per call: drop the previous call's speculative KV
    /// tail, absorb the queued catch-up positions as canonical KV entries
    /// (KV-only forwards), then draft recurrently — step 1 conditions on
    /// (`h_last`, `last_token`), each later step on the MTP's own `h_out` and
    /// its previous draft. `h_last_pos` is the absolute position `h_last` was
    /// read at; the canon's contiguity is asserted, not assumed.
    pub fn mtp_draft(
        &mut self,
        drafter: &mut crate::qwen4_mtp::MtpDrafter<'st>,
        h_last: &[f32],
        h_last_pos: usize,
        last_token: u32,
        k: usize,
    ) -> Result<Vec<u32>> {
        ensure!(k >= 1, "draft depth 0");
        let _stage = prof::stage("spec.draft");
        let Self {
            cfg,
            hc,
            tables,
            dev,
            weights,
            lm_head,
            ..
        } = self;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let (head, kv, base_pos, canonical, catchup) = drafter.parts();
        kv.k.truncate(*canonical * kv_dim);
        kv.v.truncate(*canonical * kv_dim);

        for (pos, h, tok) in catchup.drain(..) {
            if base_pos.is_none() {
                *base_pos = Some(pos);
            }
            let expect = base_pos.expect("set above") + *canonical;
            // A position BEHIND the canon is a legal re-verification: after a
            // rollback the accepted tokens ride the next chunk again (and an
            // absorb re-verifies all of `pending`), so their rows come back a
            // second time — with identical values, the replay being
            // deterministic — and the canonical entry already built from them
            // stands. Only a gap ahead of the canon is a driver bug.
            if pos < expect {
                continue;
            }
            ensure!(
                pos == expect,
                "MTP catch-up at {pos}, canon expects {expect}"
            );
            let embed = tables.embed_row(tok as usize)?;
            head.forward(
                cfg,
                hc,
                &h,
                &embed,
                pos + 1,
                kv,
                true,
                dev.as_mut().zip(weights.as_ref()),
                None,
            )?;
            *canonical += 1;
        }

        if base_pos.is_none() {
            *base_pos = Some(h_last_pos);
        }
        let expect = base_pos.expect("set above") + *canonical;
        ensure!(
            h_last_pos == expect,
            "MTP draft at {h_last_pos}, canon expects {expect}"
        );
        let mut cur_h = h_last.to_vec();
        let mut cur_tok = last_token;
        let mut drafts = Vec::with_capacity(k);
        for j in 0..k {
            let embed = tables.embed_row(cur_tok as usize)?;
            let rope_pos = base_pos.expect("set above") + kv.k.len() / kv_dim + 1;
            let out = head.forward(
                cfg,
                hc,
                &cur_h,
                &embed,
                rope_pos,
                kv,
                false,
                dev.as_mut().zip(weights.as_ref()),
                Some(lm_head),
            )?;
            let logits = out.logits.expect("lm_head was requested");
            let d = greedy_argmax(&logits);
            drafts.push(d);
            if j == 0 {
                // Draft step 1's entry is CANONICAL: its inputs are the true
                // target h and a real emitted token.
                *canonical += 1;
            }
            cur_h = out.h_out;
            cur_tok = d;
        }
        Ok(drafts)
    }

    /// Greedy speculative generation: prefill the prompt, then loop
    /// draft-k / verify-once / rollback until `n_new` tokens exist. The
    /// output token stream EQUALS plain greedy decode's by construction
    /// (`tests/qwen4_speculative.rs` holds the gate). `warmup_cap` bounds
    /// how many tail-of-prompt positions seed the MTP KV canon.
    /// Greedy speculative generation with any [`Qwen4DraftSource`]: prefill
    /// the prompt, then loop draft-k / verify-once / rollback until `n_new`
    /// tokens exist. The output token stream EQUALS plain greedy decode's by
    /// construction (`tests/qwen4_speculative.rs` holds the gate) — for ANY
    /// draft source, which is exactly what lets the gate drive adversarial
    /// drafters through the same loop the MTP head uses.
    pub fn generate_speculative<D: Qwen4DraftSource<'ctx, 'st>>(
        &mut self,
        drafter: &mut D,
        prompt: &[u32],
        n_new: usize,
        k: usize,
    ) -> Result<(Vec<u32>, Qwen4SpecStats)> {
        ensure!(!prompt.is_empty(), "empty prompt");
        ensure!(n_new >= 1 && k >= 1, "degenerate generation request");
        ensure!(
            k + 2 < QWEN4_VERIFY_MAX_TOKENS,
            "draft depth {k} leaves no room in the verify chunk"
        );
        let mut stats = Qwen4SpecStats::new(k);

        let t_pf = std::time::Instant::now();
        let logits = self.forward_prompt(0, prompt, 0)?;
        stats.prefill_s = t_pf.elapsed().as_secs_f64();
        let t_all = std::time::Instant::now();

        // The prompt's last chunk still sits in the prefill arena: its rows
        // seed the first draft's conditioning (and the MTP KV canon).
        let width = qwen4_prefill_chunk_tokens().clamp(1, 1024);
        let last_len = ((prompt.len() - 1) % width) + 1;
        let chunk_start = prompt.len() - last_len;
        let mut h_last_pos = prompt.len() - 1;
        let mut h_last = self.prefill_h_row(h_last_pos - chunk_start)?;
        drafter.warmup(self, prompt, chunk_start)?;

        let mut pending = vec![greedy_argmax(&logits)];
        let mut out: Vec<u32> = pending.clone();
        while out.len() < n_new {
            // A long rejection streak grows `pending`; absorb it as one plain
            // (non-speculative) chunk before it can overflow the verify cap.
            if pending.len() + k + 1 > QWEN4_VERIFY_MAX_TOKENS {
                let start = self.state.seq_len;
                let v = self.forward_verify(&pending, start)?;
                for j in 0..pending.len() - 1 {
                    drafter.note_accepted(start + j, &v.h_rows[j], pending[j + 1]);
                }
                h_last_pos = start + pending.len() - 1;
                h_last = v.h_rows[pending.len() - 1].clone();
                let bonus = v.argmax[pending.len() - 1];
                out.push(bonus);
                pending = vec![bonus];
                stats.absorbs += 1;
                continue;
            }
            let td = std::time::Instant::now();
            let drafts = drafter.draft(
                self,
                &h_last,
                h_last_pos,
                *pending.last().expect("non-empty"),
                k,
            )?;
            ensure!(
                drafts.len() == k,
                "draft source returned {} of {k}",
                drafts.len()
            );
            stats.draft_s += td.elapsed().as_secs_f64();

            let tv = std::time::Instant::now();
            let cycle = self.speculative_verify_cycle(&pending, &drafts)?;
            stats.verify_s += tv.elapsed().as_secs_f64() - cycle.rollback_s;
            stats.rollback_s += cycle.rollback_s;
            stats.cycles += 1;
            stats.drafted += drafts.len();
            stats.accepted_total += cycle.accepted;
            stats.accept_hist[cycle.accepted] += 1;
            if !cycle.rolled_back {
                stats.full_accepts += 1;
            }

            for (j, (pos, h)) in cycle.h_rows.iter().take(cycle.accepted).enumerate() {
                drafter.note_accepted(*pos, h, cycle.emitted[j]);
            }
            let (lp, lh) = cycle.h_rows.last().expect("at least the bonus row");
            h_last_pos = *lp;
            h_last = lh.clone();
            out.extend_from_slice(&cycle.emitted);
            pending = cycle.pending;
        }
        out.truncate(n_new);
        stats.emitted = out.len();
        stats.wall_s = t_all.elapsed().as_secs_f64();
        Ok((out, stats))
    }
}

/// A supplier of draft tokens for [`VulkanQwen4ExpModel::generate_speculative`].
///
/// The MTP head is the production implementation
/// ([`crate::qwen4_mtp::MtpDrafter`]); the equivalence gate drives synthetic
/// ones (always-right, always-wrong, mixed) through the same loop, because the
/// loop's correctness must not depend on WHAT is proposed.
pub trait Qwen4DraftSource<'ctx, 'st> {
    /// One-time seeding after the prompt prefill: the rows of the prompt's
    /// last chunk (`chunk_start..prompt.len()`) are still readable via
    /// [`VulkanQwen4ExpModel::prefill_h_row`]. Default: nothing.
    fn warmup(
        &mut self,
        model: &VulkanQwen4ExpModel<'ctx, 'st>,
        prompt: &[u32],
        chunk_start: usize,
    ) -> Result<()> {
        let _ = (model, prompt, chunk_start);
        Ok(())
    }

    /// A verified position: the target's pre-mixer `h` at `pos` and the
    /// (now-final) token at `pos + 1`. Default: nothing.
    fn note_accepted(&mut self, pos: usize, h: &[f32], next_token: u32) {
        let _ = (pos, h, next_token);
    }

    /// Produce `k` draft tokens continuing `last_token`, whose conditioning
    /// `h` (the target's pre-mixer residual at `h_last_pos`) is provided.
    fn draft(
        &mut self,
        model: &mut VulkanQwen4ExpModel<'ctx, 'st>,
        h_last: &[f32],
        h_last_pos: usize,
        last_token: u32,
        k: usize,
    ) -> Result<Vec<u32>>;
}

/// Counters of one speculative generation.
#[derive(Debug, Clone)]
pub struct Qwen4SpecStats {
    pub cycles: usize,
    /// Total drafted tokens (`cycles * k`).
    pub drafted: usize,
    /// Total accepted drafts.
    pub accepted_total: usize,
    /// Cycles with every draft accepted (no rollback).
    pub full_accepts: usize,
    /// Histogram over the accepted prefix length `L` (index 0..=k).
    pub accept_hist: Vec<usize>,
    /// Emitted new tokens (`>= cycles`, includes each cycle's bonus).
    pub emitted: usize,
    /// Non-speculative absorb chunks (a rejection streak overflowed the
    /// verify cap and `pending` was flushed at baseline speed).
    pub absorbs: usize,
    pub prefill_s: f64,
    /// Decode-phase wall (drafting + verify + rollback + bookkeeping).
    pub wall_s: f64,
    pub draft_s: f64,
    pub verify_s: f64,
    pub rollback_s: f64,
}

impl Qwen4SpecStats {
    fn new(k: usize) -> Self {
        Self {
            cycles: 0,
            drafted: 0,
            accepted_total: 0,
            full_accepts: 0,
            accept_hist: vec![0; k + 1],
            emitted: 0,
            absorbs: 0,
            prefill_s: 0.0,
            wall_s: 0.0,
            draft_s: 0.0,
            verify_s: 0.0,
            rollback_s: 0.0,
        }
    }

    /// Mean accepted drafts per cycle — the `L` in "dense bytes divide by
    /// `L + 1`".
    #[must_use]
    pub fn mean_accepted(&self) -> f64 {
        if self.cycles == 0 {
            0.0
        } else {
            self.accepted_total as f64 / self.cycles as f64
        }
    }

    /// Per-step draft acceptance rate (accepted / drafted).
    #[must_use]
    pub fn acceptance(&self) -> f64 {
        if self.drafted == 0 {
            0.0
        } else {
            self.accepted_total as f64 / self.drafted as f64
        }
    }
}

impl<'ctx> Qwen4Dev<'ctx> {
    /// One cached K or V row (`head_dim` f16 values, decoded to f32) — the
    /// equivalence harness diffs prefill-written and decode-written KV planes
    /// with this. UMA/WC host reads are the documented ~0.10 GB/s trap, which
    /// a few hundred 512-B rows do not feel.
    pub fn read_kv_row(
        &self,
        layer: usize,
        kv_head: usize,
        pos: usize,
        is_v: bool,
    ) -> Result<Vec<f32>> {
        let kv = self
            .kv
            .as_ref()
            .ok_or_else(|| anyhow!("no device KV cache"))?;
        let full_idx = *self
            .full_idx
            .get(&layer)
            .ok_or_else(|| anyhow!("layer {layer} has no device KV plane"))?;
        let plane = if is_v {
            kv.v_plane(full_idx, kv_head)
        } else {
            kv.k_plane(full_idx, kv_head)
        };
        let hd = kv.head_dim;
        let mut bytes = vec![0u8; hd * 2];
        kv.buffer
            .copy_to_host_at(kv.row(plane, pos), &mut bytes)
            .map_err(|e| anyhow!("read KV row: {e}"))?;
        Ok(bytes
            .chunks_exact(2)
            .map(|c| infer_gguf::dequant::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Additive instrumentation: wall-clock profile (`prof`), expert-id capture
// (`moe_ids`).
// ─────────────────────────────────────────────────────────────────────────────

/// Opt-in capture of the prefill's per-(layer, chunk) MoE expert selections,
/// read at the ids fence the prefill performs anyway — zero extra device
/// traffic. Exists for the chunked-GDN receipt in `tests/qwen4_prefill.rs`:
/// "does the reassociated recurrence flip any router selection over a real
/// prompt?" is a question about these exact ids, and nothing else exposes
/// them. Disabled (the default) it is one relaxed atomic load per chunk.
pub mod moe_ids {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};

    static ON: AtomicBool = AtomicBool::new(false);
    static ROWS: Mutex<Vec<(usize, Vec<i32>)>> = Mutex::new(Vec::new());

    /// Turn capture on or off (it starts off).
    pub fn set_enabled(on: bool) {
        ON.store(on, Ordering::Relaxed);
    }

    #[must_use]
    pub(crate) fn enabled() -> bool {
        ON.load(Ordering::Relaxed)
    }

    /// Record one chunk's compacted `[t * top_k]` ids for `layer`.
    pub(crate) fn push(layer: usize, ids: Vec<i32>) {
        if let Ok(mut rows) = ROWS.lock() {
            rows.push((layer, ids));
        }
    }

    /// Drain everything captured so far, in record order.
    #[must_use]
    pub fn take() -> Vec<(usize, Vec<i32>)> {
        ROWS.lock()
            .map(|mut r| std::mem::take(&mut *r))
            .unwrap_or_default()
    }
}

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

    /// Host simulation of `qwen4_block_perm.comp`: dst block `i` <- src block
    /// `map[i]`.
    fn apply_block_perm(src: &[f32], map: &[u32], block: usize, out_len: usize) -> Vec<f32> {
        let mut dst = vec![f32::NAN; out_len];
        for (i, &m) in map.iter().enumerate() {
            for j in 0..block {
                dst[i * block + j] = src[m as usize * block + j];
            }
        }
        dst
    }

    /// The chunk-wide permutation maps must agree, token for token, with the
    /// per-token maps decode uses (`qkv_channel_to_hf` / `v_slot_perm`) —
    /// including the `ab` region's compact-from-padded-rows contract and the
    /// inverse map. An off-by-one here produces finite activations from the
    /// wrong head, which only the device equivalence test would catch (a full
    /// model load away); this pins it at unit scale.
    #[test]
    fn pf_chunk_maps_match_the_per_token_maps() {
        let cfg = mini_cfg();
        let nv = cfg.linear_num_value_heads;
        let nk = cfg.linear_num_key_heads;
        let vd = cfg.linear_value_head_dim;
        let conv_dim = cfg.linear_conv_dim();
        let qkv_blocks = conv_dim / vd;
        let max_tokens = 3usize;
        let (flat, offs) = pf_chunk_maps(&cfg, max_tokens);
        let [qkv_at, v_at, ab_at, vinv_at] = offs;

        // Every region base is 64-element (256-B) aligned.
        for at in offs {
            assert_eq!(at % 64, 0, "map region at element {at} is misaligned");
        }

        // qkv: chunk perm == per-token HF gather.
        let hf: Vec<f32> = (0..max_tokens * conv_dim).map(|i| i as f32).collect();
        let map = &flat[qkv_at..qkv_at + max_tokens * qkv_blocks];
        let dev = apply_block_perm(&hf, map, vd, max_tokens * conv_dim);
        for t in 0..max_tokens {
            for c in 0..conv_dim {
                assert_eq!(
                    dev[t * conv_dim + c],
                    hf[t * conv_dim + qkv_channel_to_hf(&cfg, c)],
                    "qkv map: token {t} channel {c}"
                );
            }
        }

        // v (z): slot s holds HF head perm(s).
        let hf: Vec<f32> = (0..max_tokens * nv * vd).map(|i| i as f32).collect();
        let map = &flat[v_at..v_at + max_tokens * nv];
        let dev = apply_block_perm(&hf, map, vd, max_tokens * nv * vd);
        for t in 0..max_tokens {
            for slot in 0..nv {
                let orig = v_slot_perm(nk, nv, slot);
                assert_eq!(
                    dev[(t * nv + slot) * vd],
                    hf[(t * nv + orig) * vd],
                    "v map: token {t} slot {slot}"
                );
            }
        }

        // ab: compacts PF_AB_PAD-strided rows into packed slot order.
        let padded: Vec<f32> = (0..max_tokens * PF_AB_PAD).map(|i| i as f32).collect();
        let map = &flat[ab_at..ab_at + max_tokens * nv];
        let dev = apply_block_perm(&padded, map, 1, max_tokens * nv);
        for t in 0..max_tokens {
            for slot in 0..nv {
                let orig = v_slot_perm(nk, nv, slot);
                assert_eq!(
                    dev[t * nv + slot],
                    padded[t * PF_AB_PAD + orig],
                    "ab map: token {t} slot {slot}"
                );
            }
        }

        // vinv: undoes the v map, per token.
        let slotted: Vec<f32> = (0..max_tokens * nv * vd).map(|i| i as f32).collect();
        let map = &flat[vinv_at..vinv_at + max_tokens * nv];
        let dev = apply_block_perm(&slotted, map, vd, max_tokens * nv * vd);
        for t in 0..max_tokens {
            let per_token = unpermute_v_vec(&cfg, &slotted[t * nv * vd..(t + 1) * nv * vd]);
            assert_eq!(
                &dev[t * nv * vd..(t + 1) * nv * vd],
                per_token.as_slice(),
                "vinv map: token {t}"
            );
        }
    }

    /// Mask row `r` (absolute position `start_pos + r`) must allow keys
    /// `0..=start_pos + r` and block the rest — the boundary column is where a
    /// look-ahead leak would live.
    #[test]
    fn pf_causal_mask_allows_exactly_the_visible_prefix() {
        let (t, start_pos) = (3usize, 5usize);
        let kv_len = start_pos + t;
        let mask = pf_causal_mask(t, kv_len, start_pos);
        assert_eq!(mask.len(), t * kv_len * 2);
        for r in 0..t {
            for c in 0..kv_len {
                let at = (r * kv_len + c) * 2;
                let bits = u16::from_le_bytes([mask[at], mask[at + 1]]);
                let want = if c <= start_pos + r {
                    PF_F16_ZERO
                } else {
                    PF_F16_NEG_INF
                };
                assert_eq!(bits, want, "row {r} col {c}");
            }
        }
    }

    /// Synthetic routed ids with the three shapes that exercise every planner
    /// branch: a hot expert past the cap (ceil-split), mid-size experts, and
    /// singletons.
    fn grouping_fixture() -> (Vec<i32>, usize, usize, usize) {
        let (t, top_k, n_experts) = (12usize, 4usize, 32usize);
        let mut raw = vec![-7i32; t * PF_IDS_PAD]; // poison beyond top_k
        for tok in 0..t {
            for slot in 0..top_k {
                // Expert 5 in every token's slot 0 (12 rows > cap of 8);
                // the rest scattered but deterministic, distinct per token.
                let id = if slot == 0 {
                    5
                } else {
                    ((tok * 7 + slot * 11) % (n_experts - 1)) as i32 + 6
                };
                raw[tok * PF_IDS_PAD + slot] = if id == 5 { 5 } else { id % n_experts as i32 };
            }
        }
        // The mix above can repeat an id within a token for some (t, top_k);
        // the fixture must honour the router's distinct-per-token contract.
        for tok in 0..t {
            let row = &mut raw[tok * PF_IDS_PAD..tok * PF_IDS_PAD + top_k];
            for i in 1..top_k {
                while row[..i].contains(&row[i]) {
                    row[i] = (row[i] + 1).rem_euclid(n_experts as i32);
                }
            }
        }
        (raw, t, top_k, n_experts)
    }

    /// The load-bearing invariants of [`plan_moe_groups`] over the FIXED
    /// class regions: the scatter map is injective into the layout, the
    /// classes' live blocks tile exactly the routed pairs, every block's
    /// rows belong to the expert its id-list entry names, and the pinned
    /// token-major arrival order makes an expert's rows CONSECUTIVE within
    /// each of its blocks.
    #[test]
    fn moe_grouping_maps_are_a_bijection_and_blocks_match_their_experts() {
        let (raw, t, top_k, n_experts) = grouping_fixture();
        let plan = plan_moe_groups(&raw, t, PF_IDS_PAD, top_k, n_experts).expect("plan");
        let pairs = t * top_k;
        let lay = plan.layout;
        assert_eq!(lay, MoePlanLayout::new(pairs, n_experts), "layout drifted");

        // scatter: injective into the pair capacity.
        let mut owner: Vec<Option<usize>> = vec![None; lay.pair_capacity];
        for (p, &g) in plan.scatter.iter().enumerate() {
            assert!((g as usize) < lay.pair_capacity, "row {g} out of capacity");
            assert!(
                owner[g as usize].is_none(),
                "gathered row {g} claimed twice"
            );
            owner[g as usize] = Some(p);
        }
        // Live blocks tile exactly the routed pairs.
        let live: usize = (1..=PF_MOE_COLS_CAP)
            .map(|w| w * plan.n_blocks[w - 1])
            .sum();
        assert_eq!(live, pairs, "blocks must cover every routed pair");
        // Every block: within its class capacity, rows all owned by pairs of
        // the id-list expert, in ARRIVAL (pair-index ascending) order.
        for w in 1..=PF_MOE_COLS_CAP {
            assert!(
                plan.n_blocks[w - 1] <= lay.cap_blocks(w),
                "class {w} overflow"
            );
            assert!(
                lay.list_base(w).is_multiple_of(PF_LIST_ALIGN),
                "unaligned list"
            );
            for b in 0..plan.n_blocks[w - 1] {
                let expert = plan.ids[lay.list_base(w) + b];
                let mut prev_pair = None;
                for i in 0..w {
                    let row = lay.pair_base(w) + b * w + i;
                    let p = owner[row]
                        .unwrap_or_else(|| panic!("class {w} block {b} row {i} is a hole"));
                    let (tok, slot) = (p / top_k, p % top_k);
                    assert_eq!(
                        raw[tok * PF_IDS_PAD + slot],
                        expert,
                        "class {w} block {b} row {i}: wrong expert"
                    );
                    if let Some(q) = prev_pair {
                        assert!(p > q, "class {w} block {b}: arrival order broken");
                    }
                    prev_pair = Some(p);
                }
            }
            // Capacity rows past the live blocks stay unclaimed.
            for b in plan.n_blocks[w - 1]..lay.cap_blocks(w) {
                for i in 0..w {
                    assert!(owner[lay.pair_base(w) + b * w + i].is_none());
                }
            }
        }
        // The hot expert (12 rows) must split into a full block and a tail.
        let hot_blocks: usize = (1..=PF_MOE_COLS_CAP)
            .map(|w| {
                (0..plan.n_blocks[w - 1])
                    .filter(|&b| plan.ids[lay.list_base(w) + b] == 5)
                    .count()
            })
            .sum();
        assert_eq!(
            hot_blocks, 2,
            "12 rows at cap 8 is one 8-block + one 4-block"
        );
        assert_eq!(
            plan.n_blocks[PF_MOE_COLS_CAP - 1],
            1,
            "exactly one full block"
        );
    }

    /// The fixed regions must be disjoint, list bases aligned, and every
    /// capacity MONOTONE in `pairs` — the arena is laid out once at
    /// `max_tokens` and every smaller chunk's layout must fit inside it.
    #[test]
    fn moe_layout_regions_are_disjoint_aligned_and_monotone() {
        let cases = [(4usize, 32usize), (40, 32), (240, 512), (2560, 512)];
        for (pairs, n_experts) in cases {
            let lay = MoePlanLayout::new(pairs, n_experts);
            let mut at = 0usize;
            for w in (1..=PF_MOE_COLS_CAP).rev() {
                assert_eq!(lay.pair_base(w), at, "pair regions must be contiguous");
                at += w * lay.cap_blocks(w);
                assert!(lay.list_base(w).is_multiple_of(PF_LIST_ALIGN));
                assert!(lay.list_base(w) + lay.cap_blocks(w) <= lay.list_capacity);
            }
            assert_eq!(at, lay.pair_capacity);
        }
        for (small, big) in [(4usize, 40usize), (40, 2560)] {
            let (a, b) = (MoePlanLayout::new(small, 512), MoePlanLayout::new(big, 512));
            for w in 1..=PF_MOE_COLS_CAP {
                assert!(a.cap_blocks(w) <= b.cap_blocks(w), "capacity not monotone");
            }
            assert!(a.pair_capacity <= b.pair_capacity);
            assert!(a.list_capacity <= b.list_capacity);
        }
    }

    /// Garbage ids (the topk kernel broke, or a wrong region was read) must
    /// refuse loudly, not bind a wrong stack offset.
    #[test]
    fn moe_grouping_rejects_out_of_range_ids() {
        let (mut raw, t, top_k, n_experts) = grouping_fixture();
        raw[2 * PF_IDS_PAD + 1] = n_experts as i32; // one past the stack
        assert!(plan_moe_groups(&raw, t, PF_IDS_PAD, top_k, n_experts).is_err());
        raw[2 * PF_IDS_PAD + 1] = -1;
        assert!(plan_moe_groups(&raw, t, PF_IDS_PAD, top_k, n_experts).is_err());
    }

    /// `t = 1` (a one-token chunk tail) degenerates to top_k singleton
    /// blocks — the decode shape, through the grouped path.
    #[test]
    fn moe_grouping_handles_a_single_token() {
        let top_k = 4usize;
        let mut raw = vec![0i32; PF_IDS_PAD];
        raw[..top_k].copy_from_slice(&[9, 2, 30, 17]);
        let plan = plan_moe_groups(&raw, 1, PF_IDS_PAD, top_k, 32).expect("plan");
        for w in 2..=PF_MOE_COLS_CAP {
            assert_eq!(plan.n_blocks[w - 1], 0, "class {w} must be empty");
        }
        assert_eq!(plan.n_blocks[0], top_k);
        // Blocks are expert-ascending; scatter routes each slot to the row
        // whose id-list entry names its expert.
        let base1 = plan.layout.list_base(1);
        assert_eq!(&plan.ids[base1..base1 + top_k], &[2, 9, 17, 30]);
        for (slot, &id) in [9i32, 2, 30, 17].iter().enumerate() {
            let g = plan.scatter[slot] as usize;
            let blk = g - plan.layout.pair_base(1);
            assert_eq!(plan.ids[base1 + blk], id, "slot {slot}");
        }
    }
}
