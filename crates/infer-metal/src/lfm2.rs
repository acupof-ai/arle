//! LFM2.5 (`lfm2_moe`) Metal weight loading + C++ compiled-model bridge.
//!
//! Checkpoint layout (LiquidAI/LFM2.5-8B-A1B-MLX-4bit):
//!   model.embed_tokens.{weight,scales,biases}      6-bit quantized, tied lm_head
//!   model.embedding_norm.weight                    final RMSNorm (despite the name)
//!   model.layers.{i}.operator_norm / ffn_norm      pre-norms
//!   conv layers:  .conv.{in_proj,conv,out_proj}    conv weight [H, K, 1] bf16
//!   attn layers:  .self_attn.{q,k,v,out}_proj + q_layernorm/k_layernorm [head_dim]
//!   dense FFN:    .feed_forward.{gate,up,down}_proj        (layers < num_dense_layers)
//!   MoE FFN:      .feed_forward.gate (dense [E,H] f32 router), .expert_bias [E],
//!                 .feed_forward.switch_mlp.{gate,up,down}_proj (stacked 4-bit)

use std::path::Path;

use anyhow::{Context, Result};

use crate::config::{MetalLfm2Config, MetalModelConfig, MetalQwen35LayerType};
use crate::executor::CompiledMetalModel;
use crate::loader::{
    TensorMap, load_embed_tokens_from_tensors, load_proj_from_tensors, load_tensor_map, tensor_get,
    tie_lm_head_from_embed_tokens,
};
use crate::mlx::MlxArray;
use crate::weights::{
    StackedQuantized, WeightTensor, concat_weight_rows, load_stacked_quantized,
    merge_quantized_projection_rows,
};

pub(crate) struct Lfm2MoeWeights {
    pub(crate) router: WeightTensor,
    pub(crate) expert_bias: MlxArray,
    pub(crate) switch_gate: StackedQuantized,
    pub(crate) switch_up: StackedQuantized,
    pub(crate) switch_down: StackedQuantized,
    pub(crate) num_experts: i32,
    pub(crate) top_k: i32,
    pub(crate) norm_topk_prob: bool,
    pub(crate) expert_bits: i32,
    pub(crate) expert_group_size: i32,
    /// Dense BF16 stacked expert weights (from `experts.{i}.w1/w2/w3.weight`).
    /// When set, the C++ model bypasses the quantized gather_qmm path.
    pub(crate) dense_gate: Option<MlxArray>, // [E, I, H]
    pub(crate) dense_up: Option<MlxArray>,   // [E, I, H]
    pub(crate) dense_down: Option<MlxArray>, // [E, H, I]
}

pub(crate) enum Lfm2Ffn {
    Dense {
        gate_up: WeightTensor,
        gate_dim: i32,
        down: WeightTensor,
    },
    Moe(Lfm2MoeWeights),
}

pub(crate) struct Lfm2ConvWeights {
    pub(crate) op_norm: MlxArray,
    pub(crate) ffn_norm: MlxArray,
    pub(crate) in_proj: WeightTensor,
    pub(crate) conv_w: MlxArray,
    pub(crate) out_proj: WeightTensor,
    pub(crate) ffn: Lfm2Ffn,
}

pub(crate) struct Lfm2AttnWeights {
    pub(crate) op_norm: MlxArray,
    pub(crate) ffn_norm: MlxArray,
    pub(crate) q_proj: WeightTensor,
    pub(crate) k_proj: WeightTensor,
    pub(crate) v_proj: WeightTensor,
    pub(crate) o_proj: WeightTensor,
    pub(crate) q_norm: MlxArray,
    pub(crate) k_norm: MlxArray,
    pub(crate) ffn: Lfm2Ffn,
}

pub(crate) enum Lfm2Layer {
    Conv(Lfm2ConvWeights),
    Attn(Lfm2AttnWeights),
}

pub(crate) struct Lfm2MetalWeights {
    pub(crate) embedding: MlxArray,
    pub(crate) norm: MlxArray,
    pub(crate) lm_head: WeightTensor,
    pub(crate) embed_quantized: Option<WeightTensor>,
    pub(crate) layers: Vec<Lfm2Layer>,
    pub(crate) cpp_model: Option<CppLfm2Model>,
}

impl Lfm2MetalWeights {
    pub(crate) fn cpp_model(&self) -> Result<&CppLfm2Model> {
        self.cpp_model
            .as_ref()
            .context("LFM2 C++ compiled model unavailable")
    }
}

pub(crate) fn load_lfm2_metal_weights(
    model_dir: &Path,
    config: &MetalModelConfig,
) -> Result<Lfm2MetalWeights> {
    let arch = &config.arch;
    let lfm2 = arch
        .lfm2
        .as_ref()
        .context("load_lfm2_metal_weights requires an lfm2 arch config")?;
    let tensors = load_tensor_map(model_dir)?;

    let prefix = ["model"]
        .into_iter()
        .find(|candidate| {
            tensors.contains_key(&format!("{candidate}.embed_tokens.weight"))
                && tensors.contains_key(&format!("{candidate}.embedding_norm.weight"))
        })
        .context("could not detect LFM2 weight prefix")?;

    let get = |name: &str| tensor_get(&tensors, name);
    let load_proj =
        |base: &str| load_proj_from_tensors(&tensors, base, config.quantization.clone());

    let embed_base = format!("{prefix}.embed_tokens");
    let embed_tokens =
        load_embed_tokens_from_tensors(&tensors, &embed_base, config.quantization.clone())?;
    let embed_quantized = load_proj(&embed_base).ok();
    let norm = get(&format!("{prefix}.embedding_norm.weight"))?;
    let lm_head = tie_lm_head_from_embed_tokens(&embed_tokens);

    let n_conv = arch.num_conv_layers();
    let n_full = arch.num_full_attention_layers();
    log::info!(
        "  {} LFM2.5 layers ({} gated conv, {} full attention, {} MoE)",
        config.num_hidden_layers,
        n_conv,
        n_full,
        (lfm2.num_dense_layers..config.num_hidden_layers).count(),
    );

    let layers = (0..config.num_hidden_layers)
        .map(|i| {
            let lp = format!("{prefix}.layers.{i}");
            let op_norm = get(&format!("{lp}.operator_norm.weight"))?;
            let ffn_norm = get(&format!("{lp}.ffn_norm.weight"))?;
            let ffn = if lfm2.is_moe_layer(i) {
                Lfm2Ffn::Moe(load_moe_ffn(
                    &tensors,
                    &lp,
                    &lfm2.moe,
                    config.quantization.as_ref(),
                )?)
            } else {
                load_dense_ffn(&lp, &load_proj)?
            };
            let layer = match arch.layer_types[i] {
                MetalQwen35LayerType::Conv => Lfm2Layer::Conv(Lfm2ConvWeights {
                    op_norm,
                    ffn_norm,
                    in_proj: load_proj(&format!("{lp}.conv.in_proj"))?,
                    conv_w: load_conv_weight(&get(&format!("{lp}.conv.conv.weight"))?, lfm2)?,
                    out_proj: load_proj(&format!("{lp}.conv.out_proj"))?,
                    ffn,
                }),
                MetalQwen35LayerType::FullAttention => Lfm2Layer::Attn(Lfm2AttnWeights {
                    op_norm,
                    ffn_norm,
                    q_proj: load_proj(&format!("{lp}.self_attn.q_proj"))?,
                    k_proj: load_proj(&format!("{lp}.self_attn.k_proj"))?,
                    v_proj: load_proj(&format!("{lp}.self_attn.v_proj"))?,
                    o_proj: load_proj(&format!("{lp}.self_attn.out_proj"))?,
                    q_norm: get(&format!("{lp}.self_attn.q_layernorm.weight"))?,
                    k_norm: get(&format!("{lp}.self_attn.k_layernorm.weight"))?,
                    ffn,
                }),
                MetalQwen35LayerType::LinearAttention => {
                    anyhow::bail!("LFM2 checkpoint has an unexpected linear-attention layer at {i}")
                }
            };
            Ok(layer)
        })
        .collect::<Result<Vec<_>>>()?;

    let mut weights = Lfm2MetalWeights {
        embedding: embed_tokens,
        norm,
        lm_head,
        embed_quantized,
        layers,
        cpp_model: None,
    };
    weights.cpp_model = CppLfm2Model::build(&weights, config, lfm2);
    anyhow::ensure!(
        weights.cpp_model.is_some(),
        "LFM2 requires the C++ compiled model; build failed (see warnings above)"
    );
    Ok(weights)
}

fn load_dense_ffn(lp: &str, load_proj: &impl Fn(&str) -> Result<WeightTensor>) -> Result<Lfm2Ffn> {
    let base = format!("{lp}.feed_forward");
    // BF16 checkpoints use w1/w2/w3 (w1=gate, w2=down, w3=up);
    // quantized checkpoints use gate_proj/down_proj/up_proj.
    let (gate, up, down) = match load_proj(&format!("{base}.gate_proj")) {
        Ok(gate) => (
            gate,
            load_proj(&format!("{base}.up_proj"))?,
            load_proj(&format!("{base}.down_proj"))?,
        ),
        Err(_) => (
            load_proj(&format!("{base}.w1"))?,
            load_proj(&format!("{base}.w3"))?,
            load_proj(&format!("{base}.w2"))?,
        ),
    };
    let gate_dim = gate.output_dim()?;
    let gate_up = match merge_quantized_projection_rows(&[&gate, &up])? {
        Some(merged) => merged,
        None => concat_weight_rows(&gate, &up)?,
    };
    Ok(Lfm2Ffn::Dense {
        gate_up,
        gate_dim,
        down,
    })
}

fn load_moe_ffn(
    tensors: &TensorMap,
    lp: &str,
    moe: &crate::config::MetalLfm2MoeConfig,
    quantization: Option<&crate::config::QuantConfig>,
) -> Result<Lfm2MoeWeights> {
    let base = format!("{lp}.feed_forward");
    let num_experts =
        i32::try_from(moe.num_experts).context("LFM2 num_experts does not fit i32")?;
    let top_k = i32::try_from(moe.num_experts_per_tok)
        .context("LFM2 num_experts_per_tok does not fit i32")?;
    // Router may be quantized (8-bit) in some checkpoints; dequantize to dense
    // because the C++ MoE block uses matmul, not quantized_matmul, for the router.
    let router =
        match load_proj_from_tensors(tensors, &format!("{base}.gate"), quantization.cloned())? {
            d @ WeightTensor::Dense(_) => d,
            q @ WeightTensor::Quantized { .. } => WeightTensor::Dense(q.to_dense_in_out()),
        };
    let expert_bias = tensor_get(tensors, &format!("{base}.expert_bias"))?;

    // Try stacked quantized path first; fall back to individual BF16 expert weights.
    let (switch_gate, switch_up, switch_down, dense_gate, dense_up, dense_down) =
        match load_stacked_quantized(tensors, &format!("{base}.switch_mlp.gate_proj")) {
            Ok(gate) => {
                let up = load_stacked_quantized(tensors, &format!("{base}.switch_mlp.up_proj"))?;
                let down =
                    load_stacked_quantized(tensors, &format!("{base}.switch_mlp.down_proj"))?;
                (gate, up, down, None, None, None)
            }
            Err(_) => {
                // BF16 dense stacked switch_mlp (no scales/biases)
                let dense_gate =
                    tensor_get(tensors, &format!("{base}.switch_mlp.gate_proj.weight"));
                if let Ok(gate) = dense_gate {
                    let up = tensor_get(tensors, &format!("{base}.switch_mlp.up_proj.weight"))?;
                    let down = tensor_get(tensors, &format!("{base}.switch_mlp.down_proj.weight"))?;
                    let empty = || StackedQuantized {
                        weight: crate::mlx::zeros(&[1], crate::mlx::Dtype::Bfloat16),
                        scales: crate::mlx::zeros(&[1], crate::mlx::Dtype::Bfloat16),
                        biases: crate::mlx::zeros(&[1], crate::mlx::Dtype::Bfloat16),
                    };
                    (empty(), empty(), empty(), Some(gate), Some(up), Some(down))
                } else {
                    // Individual experts.{i}.w1/w2/w3.weight → stack to [E, ...]
                    let e = moe.num_experts;
                    let stack = |w: &str| -> Result<MlxArray> {
                        let arrs: Vec<_> = (0..e)
                            .map(|i| tensor_get(tensors, &format!("{base}.experts.{i}.{w}.weight")))
                            .collect::<Result<_>>()?;
                        let expert_shape = arrs[0].shape();
                        let mut new_shape = vec![e as i32];
                        new_shape.extend_from_slice(expert_shape);
                        let flat = crate::mlx::concatenate_axis(&arrs, 0);
                        Ok(crate::mlx::reshape(&flat, &new_shape))
                    };
                    let gate = stack("w1")?;
                    let up = stack("w3")?;
                    let down = stack("w2")?;
                    let empty = || StackedQuantized {
                        weight: crate::mlx::zeros(&[1], crate::mlx::Dtype::Bfloat16),
                        scales: crate::mlx::zeros(&[1], crate::mlx::Dtype::Bfloat16),
                        biases: crate::mlx::zeros(&[1], crate::mlx::Dtype::Bfloat16),
                    };
                    (empty(), empty(), empty(), Some(gate), Some(up), Some(down))
                }
            }
        };

    Ok(Lfm2MoeWeights {
        router,
        expert_bias,
        switch_gate,
        switch_up,
        switch_down,
        num_experts,
        top_k,
        norm_topk_prob: moe.norm_topk_prob,
        expert_bits: moe.expert_bits,
        expert_group_size: moe.expert_group_size,
        dense_gate,
        dense_up,
        dense_down,
    })
}

/// Depthwise conv1d weight. MLX wants `[C_out, K, C_in/groups]` = `[H, K, 1]`.
/// BF16 checkpoints may ship `[H, 1, K]`; transpose to match.
fn load_conv_weight(weight: &MlxArray, lfm2: &MetalLfm2Config) -> Result<MlxArray> {
    let h = weight.shape().first().copied().unwrap_or(0);
    let k = lfm2.conv_kernel as i32;
    match weight.shape() {
        [_, ks, 1] if *ks == k => Ok(weight.clone()),
        [_, 1, ks] if *ks == k => Ok(crate::mlx::transpose_axes(weight, &[0, 2, 1])),
        [_, ks] if *ks == k => Ok(crate::mlx::reshape(weight, &[h, k, 1])),
        shape => {
            anyhow::bail!("unsupported LFM2 conv weight shape {shape:?}, expected [H, {k}, 1]")
        }
    }
}

pub(crate) struct CppLfm2Model {
    raw: *mut std::ffi::c_void,
}

impl Drop for CppLfm2Model {
    fn drop(&mut self) {
        // SAFETY: FFI over the live model handle; free is idempotent per-session.
        unsafe {
            mlx_sys::lfm2_compiled_free(self.raw);
        }
    }
}

impl CppLfm2Model {
    #[allow(unused_unsafe)] // macro's unsafe is nested at some call sites
    pub(crate) fn build(
        weights: &Lfm2MetalWeights,
        config: &MetalModelConfig,
        lfm2: &MetalLfm2Config,
    ) -> Option<Self> {
        // SAFETY: all calls pass live owned MLX handles; failures surface via rc + mlx_last_error.
        let model = unsafe { mlx_sys::lfm2_compiled_new() };
        if model.is_null() {
            return None;
        }

        let add_weight = |weight: &WeightTensor| -> Option<i32> {
            let id = unsafe {
                match weight {
                    WeightTensor::Dense(w) => {
                        mlx_sys::lfm2_compiled_add_dense_weight(model, w.as_raw())
                    }
                    WeightTensor::Quantized {
                        w,
                        scales,
                        biases,
                        group_size,
                        bits,
                        mode,
                    } => mlx_sys::lfm2_compiled_add_quant_weight(
                        model,
                        w.as_raw(),
                        scales.as_raw(),
                        MlxArray::as_raw_opt(biases.as_ref()),
                        *group_size,
                        *bits,
                        *mode as i32,
                    ),
                }
            };
            if id < 0 {
                let err = crate::mlx::check_mlx_error()
                    .err()
                    .map_or_else(|| "unknown MLX error".to_string(), |err| err.to_string());
                log::warn!("C++ LFM2 weight registration failed: {err}");
                None
            } else {
                Some(id)
            }
        };

        macro_rules! some_or_free {
            ($expr:expr) => {
                match $expr {
                    Some(id) => id,
                    None => {
                        unsafe { mlx_sys::lfm2_compiled_free(model) };
                        return None;
                    }
                }
            };
        }
        macro_rules! add_or_free {
            ($weight:expr) => {
                some_or_free!(add_weight($weight))
            };
        }

        unsafe {
            mlx_sys::lfm2_compiled_set_config(
                model,
                config.rope_theta as f32,
                config.rms_norm_eps as f32,
                config.num_attention_heads as i32,
                config.num_key_value_heads as i32,
                config.head_dim as i32,
                config.hidden_size as i32,
                lfm2.conv_kernel as i32,
            );
        }

        let lm_head_id = add_or_free!(&weights.lm_head);
        unsafe {
            mlx_sys::lfm2_compiled_set_embed(
                model,
                weights.embedding.as_raw(),
                weights.norm.as_raw(),
                lm_head_id,
            );
        }
        // Tied lm_head: run the 6-bit quantized embedding as the output projection.
        if let Some(embed_quantized) = &weights.embed_quantized {
            let embed_id = add_or_free!(embed_quantized);
            unsafe {
                mlx_sys::lfm2_compiled_set_embed_as_linear(model, embed_id);
            }
        }

        for layer in &weights.layers {
            let (op_norm, ffn_norm, ffn) = match layer {
                Lfm2Layer::Conv(c) => (&c.op_norm, &c.ffn_norm, &c.ffn),
                Lfm2Layer::Attn(a) => (&a.op_norm, &a.ffn_norm, &a.ffn),
            };
            let (gate_up_id, gate_dim, down_id) = match ffn {
                Lfm2Ffn::Dense {
                    gate_up,
                    gate_dim,
                    down,
                } => (add_or_free!(gate_up), *gate_dim, add_or_free!(down)),
                Lfm2Ffn::Moe(_) => (-1, 0, -1),
            };

            match layer {
                Lfm2Layer::Conv(c) => unsafe {
                    mlx_sys::lfm2_compiled_push_conv_layer(
                        model,
                        op_norm.as_raw(),
                        ffn_norm.as_raw(),
                        add_or_free!(&c.in_proj),
                        c.conv_w.as_raw(),
                        add_or_free!(&c.out_proj),
                        gate_up_id,
                        gate_dim,
                        down_id,
                    );
                },
                Lfm2Layer::Attn(a) => unsafe {
                    mlx_sys::lfm2_compiled_push_attn_layer(
                        model,
                        op_norm.as_raw(),
                        ffn_norm.as_raw(),
                        add_or_free!(&a.q_proj),
                        add_or_free!(&a.k_proj),
                        add_or_free!(&a.v_proj),
                        add_or_free!(&a.o_proj),
                        a.q_norm.as_raw(),
                        a.k_norm.as_raw(),
                        gate_up_id,
                        gate_dim,
                        down_id,
                    );
                },
            }

            if let Lfm2Ffn::Moe(moe) = ffn {
                let WeightTensor::Dense(router_w) = &moe.router else {
                    log::warn!("C++ LFM2 MoE registration requires a dense router weight");
                    unsafe { mlx_sys::lfm2_compiled_free(model) };
                    return None;
                };
                if let (Some(gate), Some(up), Some(down)) =
                    (&moe.dense_gate, &moe.dense_up, &moe.dense_down)
                {
                    // BF16 dense expert path
                    unsafe {
                        mlx_sys::lfm2_compiled_set_last_moe_dense(
                            model,
                            router_w.as_raw(),
                            moe.expert_bias.as_raw(),
                            gate.as_raw(),
                            up.as_raw(),
                            down.as_raw(),
                            moe.num_experts,
                            moe.top_k,
                            moe.norm_topk_prob,
                        );
                    }
                } else {
                    // Quantized stacked expert path
                    let register_stack = |stacked: &StackedQuantized| -> Option<i32> {
                        let wt = WeightTensor::Quantized {
                            w: stacked.weight.clone(),
                            scales: stacked.scales.clone(),
                            biases: Some(stacked.biases.clone()),
                            group_size: moe.expert_group_size,
                            bits: moe.expert_bits,
                            mode: crate::config::QuantMode::Affine,
                        };
                        add_weight(&wt)
                    };
                    unsafe {
                        mlx_sys::lfm2_compiled_set_last_moe(
                            model,
                            router_w.as_raw(),
                            moe.expert_bias.as_raw(),
                            some_or_free!(register_stack(&moe.switch_gate)),
                            some_or_free!(register_stack(&moe.switch_up)),
                            some_or_free!(register_stack(&moe.switch_down)),
                            moe.expert_group_size,
                            moe.expert_bits,
                            moe.num_experts,
                            moe.top_k,
                            moe.norm_topk_prob,
                        );
                    }
                }
                if let Err(err) = crate::mlx::check_mlx_error() {
                    log::warn!("C++ LFM2 MoE registration failed: {err}");
                    unsafe { mlx_sys::lfm2_compiled_free(model) };
                    return None;
                }
            }
        }

        let rc = unsafe { mlx_sys::lfm2_compiled_finalize(model) };
        if rc != 0 {
            log::warn!("C++ LFM2 model finalize failed");
            unsafe { mlx_sys::lfm2_compiled_free(model) };
            return None;
        }
        log::info!(
            "  C++ LFM2.5 forward model ready ({} layers)",
            weights.layers.len()
        );
        Some(Self { raw: model })
    }

    pub(crate) fn begin_session(
        &self,
        kv_caches: &[MlxArray],
        conv_states: &[MlxArray],
    ) -> Result<()> {
        let mut kv_ptrs: Vec<*mut mlx_sys::mlx_array> =
            kv_caches.iter().map(MlxArray::as_raw).collect();
        let mut conv_ptrs: Vec<*mut mlx_sys::mlx_array> =
            conv_states.iter().map(MlxArray::as_raw).collect();
        // SAFETY: FFI over live session arrays; counts match the C++ model.
        let rc = unsafe {
            mlx_sys::lfm2_session_begin(
                self.raw,
                kv_ptrs.as_mut_ptr(),
                kv_ptrs.len() as i32,
                conv_ptrs.as_mut_ptr(),
                conv_ptrs.len() as i32,
            )
        };
        if rc != 0 {
            return Err(crate::mlx::check_mlx_error().unwrap_err());
        }
        Ok(())
    }

    pub(crate) fn end_session(
        &self,
        n_kv: usize,
        n_conv: usize,
    ) -> Result<(Vec<MlxArray>, Vec<MlxArray>)> {
        let mut out_kv: Vec<*mut mlx_sys::mlx_array> = vec![std::ptr::null_mut(); n_kv];
        let mut out_conv: Vec<*mut mlx_sys::mlx_array> = vec![std::ptr::null_mut(); n_conv];
        // SAFETY: FFI over live session arrays; counts match the C++ model.
        let rc = unsafe {
            mlx_sys::lfm2_session_end(
                self.raw,
                out_kv.as_mut_ptr(),
                n_kv as i32,
                out_conv.as_mut_ptr(),
                n_conv as i32,
            )
        };
        if rc != 0 {
            return Err(crate::mlx::check_mlx_error().unwrap_err());
        }
        let kv = out_kv
            .into_iter()
            // SAFETY: the bridge wrote a valid owned handle on success.
            .map(|ptr| unsafe { MlxArray::from_raw(ptr) })
            .collect();
        let conv = out_conv
            .into_iter()
            // SAFETY: the bridge wrote a valid owned handle on success.
            .map(|ptr| unsafe { MlxArray::from_raw(ptr) })
            .collect();
        Ok((kv, conv))
    }

    pub(crate) fn prefill_session(
        &self,
        tokens: &MlxArray,
        prompt_len: i32,
        cache_pos: i32,
    ) -> Result<MlxArray> {
        let mut out_logits: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        // SAFETY: FFI over live session arrays.
        let rc = unsafe {
            mlx_sys::lfm2_compiled_prefill_session(
                self.raw,
                tokens.as_raw(),
                prompt_len,
                cache_pos,
                &raw mut out_logits,
            )
        };
        if rc != 0 {
            return Err(crate::mlx::check_mlx_error().unwrap_err());
        }
        // SAFETY: the bridge wrote a valid owned handle on success.
        Ok(unsafe { MlxArray::from_raw(out_logits) })
    }

    pub(crate) fn step_session(&self, token: &MlxArray, cache_pos: i32) -> Result<MlxArray> {
        let mut out_logits: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        // SAFETY: FFI over live session arrays.
        let rc = unsafe {
            mlx_sys::lfm2_compiled_step_session(
                self.raw,
                token.as_raw(),
                cache_pos,
                &raw mut out_logits,
            )
        };
        if rc != 0 {
            return Err(crate::mlx::check_mlx_error().unwrap_err());
        }
        // SAFETY: the bridge wrote a valid owned handle on success.
        Ok(unsafe { MlxArray::from_raw(out_logits) })
    }

    /// Eager single-token decode for the DSpark adaptive-skip fallback.
    /// The compiled forward_verify traces with S=5 and bakes slice/reshape
    /// indices that fail on S=1; the eager path reads S at runtime.
    pub(crate) fn eager_step_session(&self, token: &MlxArray, cache_pos: i32) -> Result<MlxArray> {
        let mut out_logits: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        // SAFETY: FFI over live session arrays.
        let rc = unsafe {
            mlx_sys::lfm2_eager_step_session(
                self.raw,
                token.as_raw(),
                cache_pos,
                &raw mut out_logits,
            )
        };
        if rc != 0 {
            return Err(crate::mlx::check_mlx_error().unwrap_err());
        }
        // SAFETY: the bridge wrote a valid owned handle on success.
        Ok(unsafe { MlxArray::from_raw(out_logits) })
    }

    /// DSpark block-verification forward: runs `block_len` tokens in one pass
    /// with FULL logits (all positions) and hidden/conv capture enabled. The
    /// captured tails are read back via `drain_captured_hidden` /
    /// `drain_captured_conv_inputs`; clear with `clear_capture_layers` after.
    pub(crate) fn verify_block_session(
        &self,
        tokens: &MlxArray,
        block_len: i32,
        cache_pos: i32,
        capture_layer_ids: &[usize],
    ) -> Result<MlxArray> {
        let ids: Vec<i32> = capture_layer_ids
            .iter()
            .map(|&id| i32::try_from(id).context("capture layer id does not fit in i32"))
            .collect::<Result<Vec<_>>>()?;
        let mut out_logits: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        // SAFETY: FFI over live session arrays.
        let rc = unsafe {
            mlx_sys::lfm2_compiled_verify_block_session(
                self.raw,
                tokens.as_raw(),
                block_len,
                cache_pos,
                ids.as_ptr(),
                ids.len() as i32,
                &raw mut out_logits,
            )
        };
        if rc != 0 {
            return Err(crate::mlx::check_mlx_error().unwrap_err());
        }
        // SAFETY: the bridge wrote a valid owned handle on success.
        Ok(unsafe { MlxArray::from_raw(out_logits) })
    }

    pub(crate) fn set_capture_layers(&self, layer_ids: &[usize]) -> Result<()> {
        let ids: Vec<i32> = layer_ids
            .iter()
            .map(|&id| i32::try_from(id).context("capture layer id does not fit in i32"))
            .collect::<Result<Vec<_>>>()?;
        // SAFETY: FFI over valid owned handle and live caller buffer.
        unsafe {
            mlx_sys::lfm2_set_capture_layers(self.raw, ids.as_ptr(), ids.len() as i32);
        }
        Ok(())
    }

    pub(crate) fn clear_capture_layers(&self) {
        // SAFETY: FFI over valid owned handle.
        unsafe {
            mlx_sys::lfm2_set_capture_layers(self.raw, std::ptr::null(), 0);
        }
    }

    pub(crate) fn drain_captured_hidden(&self) -> Result<Vec<MlxArray>> {
        Self::drain_captured(
            self.raw,
            mlx_sys::lfm2_get_captured_hidden_count,
            mlx_sys::lfm2_get_captured_hidden,
            "hidden",
        )
    }

    pub(crate) fn drain_captured_conv_inputs(&self) -> Result<Vec<MlxArray>> {
        Self::drain_captured(
            self.raw,
            mlx_sys::lfm2_get_captured_conv_count,
            mlx_sys::lfm2_get_captured_conv_input,
            "conv",
        )
    }

    fn drain_captured(
        raw: *mut std::ffi::c_void,
        count_fn: unsafe extern "C" fn(*mut std::ffi::c_void) -> i32,
        get_fn: unsafe extern "C" fn(
            *mut std::ffi::c_void,
            i32,
            *mut *mut mlx_sys::mlx_array,
        ) -> i32,
        what: &str,
    ) -> Result<Vec<MlxArray>> {
        // SAFETY: FFI over valid owned handle; rc/error checked after.
        let n_cap = unsafe { count_fn(raw) };
        anyhow::ensure!(
            n_cap >= 0,
            "LFM2 captured-{what} count was negative: {n_cap}"
        );
        (0..n_cap)
            .map(|idx| {
                let mut h_ptr: *mut mlx_sys::mlx_array = std::ptr::null_mut();
                // SAFETY: FFI over valid owned handle.
                let rc = unsafe { get_fn(raw, idx, &raw mut h_ptr) };
                if rc != 0 {
                    return Err(crate::mlx::check_mlx_error().unwrap_err());
                }
                anyhow::ensure!(
                    !h_ptr.is_null(),
                    "LFM2 captured-{what} handle #{idx} was null"
                );
                // SAFETY: the bridge wrote a valid owned handle on success.
                Ok(unsafe { MlxArray::from_raw(h_ptr) })
            })
            .collect()
    }

    /// Overwrite the session conv states (used by the DSpark spec loop to
    /// roll the conv window back to the accepted position without a re-run).
    pub(crate) fn set_session_conv_states(&self, conv_states: &[MlxArray]) -> Result<()> {
        let mut ptrs: Vec<*mut mlx_sys::mlx_array> =
            conv_states.iter().map(MlxArray::as_raw).collect();
        // SAFETY: FFI over live session arrays.
        let rc = unsafe {
            mlx_sys::lfm2_session_set_conv_states(self.raw, ptrs.as_mut_ptr(), ptrs.len() as i32)
        };
        if rc != 0 {
            return Err(crate::mlx::check_mlx_error().unwrap_err());
        }
        Ok(())
    }
}

impl CompiledMetalModel for CppLfm2Model {
    fn session_begin(&self, kv: &[MlxArray], recurrent: &[MlxArray]) -> Result<()> {
        self.begin_session(kv, recurrent)
    }
    fn session_end(
        &self,
        n_kv: usize,
        n_recurrent: usize,
    ) -> Result<(Vec<MlxArray>, Vec<MlxArray>)> {
        self.end_session(n_kv, n_recurrent)
    }
    fn session_prefill(
        &self,
        tokens: &MlxArray,
        prompt_len: i32,
        cache_pos: i32,
    ) -> Result<MlxArray> {
        self.prefill_session(tokens, prompt_len, cache_pos)
    }
    fn session_step(&self, token: &MlxArray, cache_pos: i32) -> Result<MlxArray> {
        self.step_session(token, cache_pos)
    }
}
