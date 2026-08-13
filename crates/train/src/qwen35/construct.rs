//! Build a Qwen35Model's layer stack into a TensorStore: one builder plus the init-mode/LoRA/TP entry points.

use super::*;

impl Qwen35Model {
    pub fn new(cfg: &Qwen35Config, store: &mut TensorStore) -> Result<Self> {
        Self::new_internal(
            cfg,
            None,
            LoraTargetSet::AllLinear,
            None,
            false,
            TpContext::single(),
            Qwen35InitMode::ScratchTrain,
            store,
        )
    }

    pub fn config(&self) -> &Qwen35Config {
        &self.config
    }

    pub fn tensor_parallel(&self) -> TpContext {
        self.tp
    }

    pub fn lora_layer_start(&self) -> Option<usize> {
        self.lora_layer_start
    }

    pub fn new_for_eval(cfg: &Qwen35Config, store: &mut TensorStore) -> Result<Self> {
        Self::new_internal(
            cfg,
            None,
            LoraTargetSet::AllLinear,
            None,
            false,
            TpContext::single(),
            Qwen35InitMode::LoraOrFrozen {
                materialize_frozen_base: true,
            },
            store,
        )
    }

    pub(crate) fn new_for_checkpoint_load(
        cfg: &Qwen35Config,
        store: &mut TensorStore,
    ) -> Result<Self> {
        Self::new_internal(
            cfg,
            None,
            LoraTargetSet::AllLinear,
            None,
            false,
            TpContext::single(),
            Qwen35InitMode::LoraOrFrozen {
                materialize_frozen_base: false,
            },
            store,
        )
    }

    pub fn new_with_lora(
        cfg: &Qwen35Config,
        lora: Option<LoraConfig>,
        store: &mut TensorStore,
    ) -> Result<Self> {
        Self::new_internal(
            cfg,
            lora,
            LoraTargetSet::AllLinear,
            None,
            false,
            TpContext::single(),
            Qwen35InitMode::LoraOrFrozen {
                materialize_frozen_base: true,
            },
            store,
        )
    }

    pub fn new_with_lora_targets(
        cfg: &Qwen35Config,
        lora: LoraConfig,
        target_set: LoraTargetSet,
        store: &mut TensorStore,
    ) -> Result<Self> {
        Self::new_with_lora_targets_layer_start(cfg, lora, target_set, None, store)
    }

    pub fn new_with_lora_targets_layer_start(
        cfg: &Qwen35Config,
        lora: LoraConfig,
        target_set: LoraTargetSet,
        lora_layer_start: Option<usize>,
        store: &mut TensorStore,
    ) -> Result<Self> {
        Self::new_internal(
            cfg,
            Some(lora),
            target_set,
            lora_layer_start,
            false,
            TpContext::single(),
            Qwen35InitMode::LoraOrFrozen {
                materialize_frozen_base: true,
            },
            store,
        )
    }

    pub(crate) fn new_with_lora_targets_for_checkpoint_load(
        cfg: &Qwen35Config,
        lora: LoraConfig,
        target_set: LoraTargetSet,
        lora_skip_experts: bool,
        store: &mut TensorStore,
    ) -> Result<Self> {
        Self::new_with_lora_targets_for_checkpoint_load_layer_start(
            cfg,
            lora,
            target_set,
            None,
            lora_skip_experts,
            store,
        )
    }

    pub(crate) fn new_with_lora_targets_for_checkpoint_load_layer_start(
        cfg: &Qwen35Config,
        lora: LoraConfig,
        target_set: LoraTargetSet,
        lora_layer_start: Option<usize>,
        lora_skip_experts: bool,
        store: &mut TensorStore,
    ) -> Result<Self> {
        Self::new_internal(
            cfg,
            Some(lora),
            target_set,
            lora_layer_start,
            lora_skip_experts,
            TpContext::single(),
            Qwen35InitMode::LoraOrFrozen {
                materialize_frozen_base: false,
            },
            store,
        )
    }

    #[doc(hidden)]
    pub fn new_with_lora_targets_and_tp(
        cfg: &Qwen35Config,
        lora: LoraConfig,
        target_set: LoraTargetSet,
        tp: TpContext,
        store: &mut TensorStore,
    ) -> Result<Self> {
        Self::new_with_lora_targets_and_tp_layer_start(cfg, lora, target_set, None, tp, store)
    }

    #[doc(hidden)]
    pub fn new_with_lora_targets_and_tp_layer_start(
        cfg: &Qwen35Config,
        lora: LoraConfig,
        target_set: LoraTargetSet,
        lora_layer_start: Option<usize>,
        tp: TpContext,
        store: &mut TensorStore,
    ) -> Result<Self> {
        Self::new_internal(
            cfg,
            Some(lora),
            target_set,
            lora_layer_start,
            false,
            tp,
            Qwen35InitMode::LoraOrFrozen {
                materialize_frozen_base: true,
            },
            store,
        )
    }

    pub fn new_lora_from_base(
        base: &Qwen35Model,
        lora: LoraConfig,
        target_set: LoraTargetSet,
        store: &mut TensorStore,
    ) -> Result<Self> {
        Self::new_lora_from_base_layer_start(base, lora, target_set, None, store)
    }

    pub fn new_lora_from_base_layer_start(
        base: &Qwen35Model,
        lora: LoraConfig,
        target_set: LoraTargetSet,
        lora_layer_start: Option<usize>,
        store: &mut TensorStore,
    ) -> Result<Self> {
        // The base weights are shared from `base`, so do NOT materialize a
        // second frozen-base copy (for a 27B model that is ~108 GB of f32
        // random data immediately discarded by share_base_parameters_from).
        let mut model = Self::new_internal(
            &base.config,
            Some(lora),
            target_set,
            lora_layer_start,
            false,
            base.tp,
            Qwen35InitMode::LoraOrFrozen {
                materialize_frozen_base: false,
            },
            store,
        )?;
        model.gradient_checkpointing = base.gradient_checkpointing;
        model.share_base_parameters_from(base)?;

        let keep = base
            .all_parameter_ids()
            .into_iter()
            .chain(model.all_parameter_ids())
            .collect::<HashSet<_>>();
        store.retain_ids(&keep);
        Ok(model)
    }

    pub(super) fn new_internal(
        cfg: &Qwen35Config,
        lora: Option<LoraConfig>,
        lora_target_set: LoraTargetSet,
        lora_layer_start: Option<usize>,
        lora_skip_experts: bool,
        tp: TpContext,
        mode: Qwen35InitMode,
        store: &mut TensorStore,
    ) -> Result<Self> {
        match mode {
            Qwen35InitMode::ScratchTrain => cfg.validate_train_scratch_contract()?,
            Qwen35InitMode::LoraOrFrozen { .. } => cfg.validate_train_lora_or_frozen_contract()?,
        }
        tp.validate(cfg)?;
        if let Some(start) = lora_layer_start
            && start >= cfg.num_hidden_layers
        {
            return Err(Qwen35Error::InvalidConfig(
                "lora_layer_start must be less than num_hidden_layers",
            ));
        }
        let mut param_names = HashMap::new();
        let mut adapter_names = HashMap::new();
        let mut param_ids = Vec::new();
        let mut seen = HashSet::new();
        let mut register_named =
            |target: &mut HashMap<&'static str, TensorId>, name: &'static str, id: TensorId| {
                target.insert(name, id);
                if seen.insert(id) {
                    param_ids.push(id);
                }
            };
        let base_requires_grad = matches!(mode, Qwen35InitMode::ScratchTrain) && lora.is_none();
        let materialize_frozen_base = match mode {
            Qwen35InitMode::ScratchTrain => true,
            Qwen35InitMode::LoraOrFrozen {
                materialize_frozen_base,
            } => materialize_frozen_base,
        };

        let embed_tokens_name = cfg.embed_tokens_tensor_name();
        let embed_tokens = normal_or_unmaterialized_parameter(
            embed_tokens_name,
            &[cfg.vocab_size, cfg.hidden_size],
            0.02,
            base_requires_grad,
            materialize_frozen_base,
            store,
        )?;
        register_named(&mut param_names, embed_tokens_name, embed_tokens);

        let lm_head_name = cfg.lm_head_tensor_name();
        let lm_head = if cfg.tie_word_embeddings {
            embed_tokens
        } else {
            normal_or_unmaterialized_parameter(
                lm_head_name,
                &[cfg.vocab_size, cfg.hidden_size],
                0.02,
                base_requires_grad,
                materialize_frozen_base,
                store,
            )?
        };
        register_named(&mut param_names, lm_head_name, lm_head);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer_lora = lora_for_layer(lora, lora_layer_start, layer_idx);
            let names = cfg.layer_tensor_names(layer_idx);
            let input_layernorm_name = leak_name(names.common.input_layernorm.clone());
            let post_attention_layernorm_name =
                leak_name(names.common.post_attention_layernorm.clone());

            let input_layernorm = ones_or_unmaterialized_parameter(
                input_layernorm_name,
                &[cfg.hidden_size],
                base_requires_grad,
                materialize_frozen_base,
                store,
            )?;
            let mlp = if cfg.is_moe_layer(layer_idx) {
                if !cfg.norm_topk_prob {
                    return Err(Qwen35Error::InvalidConfig(
                        "train-side Qwen3.6 MoE currently requires norm_topk_prob=true",
                    ));
                }
                let moe_names = names.common.moe_tensor_names();
                Qwen35Mlp::Sparse(Box::new(new_sparse_mlp(
                    &moe_names,
                    cfg,
                    tp,
                    base_requires_grad,
                    materialize_frozen_base,
                    layer_lora,
                    lora_target_set,
                    lora_skip_experts,
                    store,
                )?))
            } else {
                let gate_proj_name = leak_name(names.common.mlp_gate_proj);
                let up_proj_name = leak_name(names.common.mlp_up_proj);
                let down_proj_name = leak_name(names.common.mlp_down_proj);
                Qwen35Mlp::Dense(Box::new(Qwen35DenseMlp {
                    gate_proj: linear_with_base_init(
                        gate_proj_name,
                        cfg.hidden_size,
                        tp.local_intermediate_size(cfg)?,
                        base_requires_grad,
                        lora_for_name(layer_lora, lora_target_set, gate_proj_name),
                        materialize_frozen_base,
                        store,
                    )?,
                    up_proj: linear_with_base_init(
                        up_proj_name,
                        cfg.hidden_size,
                        tp.local_intermediate_size(cfg)?,
                        base_requires_grad,
                        lora_for_name(layer_lora, lora_target_set, up_proj_name),
                        materialize_frozen_base,
                        store,
                    )?,
                    down_proj: linear_with_base_init(
                        down_proj_name,
                        tp.local_intermediate_size(cfg)?,
                        cfg.hidden_size,
                        base_requires_grad,
                        lora_for_name(layer_lora, lora_target_set, down_proj_name),
                        materialize_frozen_base,
                        store,
                    )?,
                }))
            };
            let post_attention_layernorm = ones_or_unmaterialized_parameter(
                post_attention_layernorm_name,
                &[cfg.hidden_size],
                base_requires_grad,
                materialize_frozen_base,
                store,
            )?;

            register_named(&mut param_names, input_layernorm_name, input_layernorm);
            register_mlp(
                &mut param_names,
                &mut adapter_names,
                &mut register_named,
                &mlp,
            );
            register_named(
                &mut param_names,
                post_attention_layernorm_name,
                post_attention_layernorm,
            );

            let self_attn = match names.attention {
                Qwen35AttentionTensorNames::Full(attn_names) => {
                    let q_proj_name = leak_name(attn_names.q_proj);
                    let k_proj_name = leak_name(attn_names.k_proj);
                    let v_proj_name = leak_name(attn_names.v_proj);
                    let o_proj_name = leak_name(attn_names.o_proj);
                    let q_norm_name = leak_name(attn_names.q_norm);
                    let k_norm_name = leak_name(attn_names.k_norm);

                    let q_proj = linear_with_base_init(
                        q_proj_name,
                        cfg.hidden_size,
                        tp.full_attn_q_proj_dim(cfg)?,
                        base_requires_grad,
                        lora_for_name(layer_lora, lora_target_set, q_proj_name),
                        materialize_frozen_base,
                        store,
                    )?;
                    let k_proj = linear_with_base_init(
                        k_proj_name,
                        cfg.hidden_size,
                        tp.full_attn_kv_dim(cfg)?,
                        base_requires_grad,
                        lora_for_name(layer_lora, lora_target_set, k_proj_name),
                        materialize_frozen_base,
                        store,
                    )?;
                    let v_proj = linear_with_base_init(
                        v_proj_name,
                        cfg.hidden_size,
                        tp.full_attn_kv_dim(cfg)?,
                        base_requires_grad,
                        lora_for_name(layer_lora, lora_target_set, v_proj_name),
                        materialize_frozen_base,
                        store,
                    )?;
                    let o_proj = linear_with_base_init(
                        o_proj_name,
                        tp.full_attn_q_dim(cfg)?,
                        cfg.hidden_size,
                        base_requires_grad,
                        lora_for_name(layer_lora, lora_target_set, o_proj_name),
                        materialize_frozen_base,
                        store,
                    )?;
                    let q_norm = ones_or_unmaterialized_parameter(
                        q_norm_name,
                        &[cfg.head_dim],
                        base_requires_grad,
                        materialize_frozen_base,
                        store,
                    )?;
                    let k_norm = ones_or_unmaterialized_parameter(
                        k_norm_name,
                        &[cfg.head_dim],
                        base_requires_grad,
                        materialize_frozen_base,
                        store,
                    )?;

                    register_linear(
                        &mut param_names,
                        &mut adapter_names,
                        &mut register_named,
                        &q_proj,
                    );
                    register_linear(
                        &mut param_names,
                        &mut adapter_names,
                        &mut register_named,
                        &k_proj,
                    );
                    register_linear(
                        &mut param_names,
                        &mut adapter_names,
                        &mut register_named,
                        &v_proj,
                    );
                    register_linear(
                        &mut param_names,
                        &mut adapter_names,
                        &mut register_named,
                        &o_proj,
                    );
                    register_named(&mut param_names, q_norm_name, q_norm);
                    register_named(&mut param_names, k_norm_name, k_norm);

                    Qwen35Attention::Full(Qwen35FullAttention {
                        q_proj,
                        k_proj,
                        v_proj,
                        o_proj,
                        q_norm,
                        k_norm,
                    })
                }
                Qwen35AttentionTensorNames::Linear(attn_names) => {
                    let in_proj_qkv_name = leak_name(attn_names.in_proj_qkv);
                    let in_proj_z_name = leak_name(attn_names.in_proj_z);
                    let in_proj_b_name = leak_name(attn_names.in_proj_b);
                    let in_proj_a_name = leak_name(attn_names.in_proj_a);
                    let conv1d_weight_name = leak_name(attn_names.conv1d_weight);
                    let dt_bias_name = leak_name(attn_names.dt_bias);
                    let a_log_name = leak_name(attn_names.a_log);
                    let norm_name = leak_name(attn_names.norm);
                    let out_proj_name = leak_name(attn_names.out_proj);

                    let in_proj_qkv = linear_with_base_init(
                        in_proj_qkv_name,
                        cfg.hidden_size,
                        cfg.linear_attn_qkv_dim(),
                        base_requires_grad,
                        lora_for_name(layer_lora, lora_target_set, in_proj_qkv_name),
                        materialize_frozen_base,
                        store,
                    )?;
                    let in_proj_z = linear_with_base_init(
                        in_proj_z_name,
                        cfg.hidden_size,
                        cfg.linear_attn_z_dim(),
                        base_requires_grad,
                        lora_for_name(layer_lora, lora_target_set, in_proj_z_name),
                        materialize_frozen_base,
                        store,
                    )?;
                    let in_proj_b = linear_with_base_init(
                        in_proj_b_name,
                        cfg.hidden_size,
                        cfg.linear_num_value_heads,
                        base_requires_grad,
                        lora_for_name(layer_lora, lora_target_set, in_proj_b_name),
                        materialize_frozen_base,
                        store,
                    )?;
                    let in_proj_a = linear_with_base_init(
                        in_proj_a_name,
                        cfg.hidden_size,
                        cfg.linear_num_value_heads,
                        base_requires_grad,
                        lora_for_name(layer_lora, lora_target_set, in_proj_a_name),
                        materialize_frozen_base,
                        store,
                    )?;
                    let conv1d_weight = normal_or_unmaterialized_parameter(
                        conv1d_weight_name,
                        &[cfg.linear_attn_qkv_dim(), cfg.linear_conv_kernel_dim],
                        0.02,
                        base_requires_grad,
                        materialize_frozen_base,
                        store,
                    )?;
                    let dt_bias = normal_or_unmaterialized_parameter(
                        dt_bias_name,
                        &[cfg.linear_num_value_heads],
                        0.02,
                        base_requires_grad,
                        materialize_frozen_base,
                        store,
                    )?;
                    let a_log = normal_or_unmaterialized_parameter(
                        a_log_name,
                        &[cfg.linear_num_value_heads],
                        0.02,
                        base_requires_grad,
                        materialize_frozen_base,
                        store,
                    )?;
                    let norm = ones_or_unmaterialized_parameter(
                        norm_name,
                        &[cfg.linear_value_head_dim],
                        base_requires_grad,
                        materialize_frozen_base,
                        store,
                    )?;
                    let out_proj = linear_with_base_init(
                        out_proj_name,
                        cfg.linear_attn_z_dim(),
                        cfg.hidden_size,
                        base_requires_grad,
                        lora_for_name(layer_lora, lora_target_set, out_proj_name),
                        materialize_frozen_base,
                        store,
                    )?;

                    register_linear(
                        &mut param_names,
                        &mut adapter_names,
                        &mut register_named,
                        &in_proj_qkv,
                    );
                    register_linear(
                        &mut param_names,
                        &mut adapter_names,
                        &mut register_named,
                        &in_proj_z,
                    );
                    register_linear(
                        &mut param_names,
                        &mut adapter_names,
                        &mut register_named,
                        &in_proj_b,
                    );
                    register_linear(
                        &mut param_names,
                        &mut adapter_names,
                        &mut register_named,
                        &in_proj_a,
                    );
                    register_named(&mut param_names, conv1d_weight_name, conv1d_weight);
                    register_named(&mut param_names, dt_bias_name, dt_bias);
                    register_named(&mut param_names, a_log_name, a_log);
                    register_named(&mut param_names, norm_name, norm);
                    register_linear(
                        &mut param_names,
                        &mut adapter_names,
                        &mut register_named,
                        &out_proj,
                    );

                    Qwen35Attention::Linear(Qwen35LinearAttention {
                        in_proj_qkv,
                        in_proj_z,
                        in_proj_b,
                        in_proj_a,
                        conv1d_weight,
                        dt_bias,
                        a_log,
                        norm,
                        out_proj,
                    })
                }
            };

            layers.push(Qwen35Layer {
                index: layer_idx,
                input_layernorm,
                self_attn,
                post_attention_layernorm,
                mlp,
            });
        }

        let final_norm_name = cfg.norm_tensor_name();
        let final_norm = ones_or_unmaterialized_parameter(
            final_norm_name,
            &[cfg.hidden_size],
            base_requires_grad,
            materialize_frozen_base,
            store,
        )?;
        register_named(&mut param_names, final_norm_name, final_norm);

        let (cos_cache, sin_cache) = build_rope_cache(cfg, store)?;
        if seen.insert(cos_cache) {
            param_ids.push(cos_cache);
        }
        if seen.insert(sin_cache) {
            param_ids.push(sin_cache);
        }

        Ok(Self {
            config: cfg.clone(),
            tp,
            lora,
            lora_target_set,
            lora_layer_start,
            lora_skip_experts,
            layers,
            embed_tokens,
            final_norm,
            lm_head,
            cos_cache,
            sin_cache,
            param_names,
            adapter_names,
            param_ids,
            gradient_checkpointing: false,
        })
    }
}
