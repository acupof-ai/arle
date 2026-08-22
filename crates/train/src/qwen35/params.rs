use super::*;

impl Qwen35Model {
    pub fn all_parameter_ids(&self) -> Vec<TensorId> {
        self.param_ids.clone()
    }

    pub fn lm_head_weight_id(&self) -> TensorId {
        self.lm_head
    }

    pub(super) fn share_base_parameters_from(&mut self, base: &Qwen35Model) -> Result<()> {
        if self.layers.len() != base.layers.len() {
            return Err(Qwen35Error::InvalidConfig(
                "cannot share Qwen3.5 base weights across mismatched layer counts",
            ));
        }

        self.embed_tokens = base.embed_tokens;
        self.final_norm = base.final_norm;
        self.lm_head = base.lm_head;
        self.cos_cache = base.cos_cache;
        self.sin_cache = base.sin_cache;

        for (layer, base_layer) in self.layers.iter_mut().zip(&base.layers) {
            layer.input_layernorm = base_layer.input_layernorm;
            layer.post_attention_layernorm = base_layer.post_attention_layernorm;
            share_base_attention(&mut layer.self_attn, &base_layer.self_attn)?;
            share_base_mlp(&mut layer.mlp, &base_layer.mlp)?;
        }

        self.param_names = base.param_names.clone();
        let adapter_ids = self.adapter_names.values().copied().collect::<HashSet<_>>();
        let mut seen = HashSet::new();
        let param_ids: Vec<_> = base
            .param_ids
            .iter()
            .copied()
            .chain(
                self.param_ids
                    .iter()
                    .copied()
                    .filter(|id| adapter_ids.contains(id)),
            )
            .filter(|id| seen.insert(*id))
            .collect();
        self.param_ids = param_ids;
        Ok(())
    }

    pub fn clone_frozen(&self, store: &mut TensorStore) -> Self {
        let cloned = Self::new_internal(
            &self.config,
            self.lora,
            self.lora_target_set,
            self.lora_layer_start,
            self.lora_skip_experts,
            self.tp,
            Qwen35InitMode::LoraOrFrozen {
                materialize_frozen_base: true,
            },
            store,
        )
        .expect("clone_frozen should preserve config");
        copy_frozen_tensor_map(&self.param_names, &cloned.param_names, store);
        copy_frozen_tensor_map(&self.adapter_names, &cloned.adapter_names, store);
        copy_frozen_tensor(self.cos_cache, cloned.cos_cache, store);
        copy_frozen_tensor(self.sin_cache, cloned.sin_cache, store);

        cloned
    }

    pub fn param_name_map(&self) -> HashMap<&'static str, TensorId> {
        self.param_names.clone()
    }

    pub fn adapter_name_map(&self) -> HashMap<&'static str, TensorId> {
        self.adapter_names.clone()
    }

    pub fn materialized_param_name_map(
        &self,
        store: &mut TensorStore,
    ) -> Result<HashMap<&'static str, TensorId>> {
        if self.lora.is_none() {
            return Ok(self.param_names.clone());
        }
        let mut map = self.param_names.clone();
        for layer in &self.layers {
            match &layer.self_attn {
                Qwen35Attention::Full(attn) => {
                    let merged_q = {
                        let tensor = attn.q_proj.merged_tensor(store)?;
                        store.alloc(tensor)
                    };
                    let merged_k = {
                        let tensor = attn.k_proj.merged_tensor(store)?;
                        store.alloc(tensor)
                    };
                    let merged_v = {
                        let tensor = attn.v_proj.merged_tensor(store)?;
                        store.alloc(tensor)
                    };
                    let merged_o = {
                        let tensor = attn.o_proj.merged_tensor(store)?;
                        store.alloc(tensor)
                    };
                    for (name, _) in attn.q_proj.parameter_name_map() {
                        map.insert(name, merged_q);
                    }
                    for (name, _) in attn.k_proj.parameter_name_map() {
                        map.insert(name, merged_k);
                    }
                    for (name, _) in attn.v_proj.parameter_name_map() {
                        map.insert(name, merged_v);
                    }
                    for (name, _) in attn.o_proj.parameter_name_map() {
                        map.insert(name, merged_o);
                    }
                }
                Qwen35Attention::Linear(attn) => {
                    let merged_qkv = {
                        let tensor = attn.in_proj_qkv.merged_tensor(store)?;
                        store.alloc(tensor)
                    };
                    let merged_z = {
                        let tensor = attn.in_proj_z.merged_tensor(store)?;
                        store.alloc(tensor)
                    };
                    let merged_b = {
                        let tensor = attn.in_proj_b.merged_tensor(store)?;
                        store.alloc(tensor)
                    };
                    let merged_a = {
                        let tensor = attn.in_proj_a.merged_tensor(store)?;
                        store.alloc(tensor)
                    };
                    let merged_out = {
                        let tensor = attn.out_proj.merged_tensor(store)?;
                        store.alloc(tensor)
                    };
                    for (name, _) in attn.in_proj_qkv.parameter_name_map() {
                        map.insert(name, merged_qkv);
                    }
                    for (name, _) in attn.in_proj_z.parameter_name_map() {
                        map.insert(name, merged_z);
                    }
                    for (name, _) in attn.in_proj_b.parameter_name_map() {
                        map.insert(name, merged_b);
                    }
                    for (name, _) in attn.in_proj_a.parameter_name_map() {
                        map.insert(name, merged_a);
                    }
                    for (name, _) in attn.out_proj.parameter_name_map() {
                        map.insert(name, merged_out);
                    }
                }
            }
            insert_materialized_mlp_params(&mut map, &layer.mlp, store)?;
        }
        Ok(map)
    }
}

pub(super) fn register_linear(
    param_names: &mut HashMap<&'static str, TensorId>,
    adapter_names: &mut HashMap<&'static str, TensorId>,
    register_named: &mut impl FnMut(&mut HashMap<&'static str, TensorId>, &'static str, TensorId),
    linear: &LinearWithLora,
) {
    for (name, id) in linear.parameter_name_map() {
        register_named(param_names, name, id);
    }
    for (name, id) in linear.adapter_ordered() {
        register_named(adapter_names, name, id);
    }
}

pub(super) fn linear_with_base_init(
    base_name: &'static str,
    in_features: usize,
    out_features: usize,
    base_requires_grad: bool,
    lora: Option<LoraConfig>,
    materialize_frozen_base: bool,
    store: &mut TensorStore,
) -> Result<LinearWithLora> {
    if materialize_frozen_base || base_requires_grad {
        Ok(LinearWithLora::new(
            base_name,
            in_features,
            out_features,
            base_requires_grad,
            lora,
            store,
        )?)
    } else {
        Ok(LinearWithLora::new_with_unmaterialized_base(
            base_name,
            in_features,
            out_features,
            base_requires_grad,
            lora,
            store,
        )?)
    }
}

pub(super) fn new_sparse_mlp(
    names: &Qwen35MoeTensorNames,
    cfg: &Qwen35Config,
    tp: TpContext,
    base_requires_grad: bool,
    materialize_frozen_base: bool,
    lora: Option<LoraConfig>,
    lora_target_set: LoraTargetSet,
    lora_skip_experts: bool,
    store: &mut TensorStore,
) -> Result<Qwen35SparseMlp> {
    let router_gate_name = leak_name(names.router_gate.clone());
    let shared_gate_proj_name = leak_name(names.shared_expert_gate_proj.clone());
    let shared_up_proj_name = leak_name(names.shared_expert_up_proj.clone());
    let shared_down_proj_name = leak_name(names.shared_expert_down_proj.clone());
    let shared_expert_gate_name = leak_name(names.shared_expert_gate.clone());

    // When --lora-skip-experts is set, routed expert projections are frozen
    // (no LoRA adapters). Only attention + shared expert carry LoRA.
    let expert_lora = if lora_skip_experts { None } else { lora };

    // TP: column-parallel gate/up shard the intermediate (out) dim; row-parallel
    // down shards its input dim. Router stays replicated (full num_experts). The
    // forward all-reduces the summed expert+shared output over the TP group.
    let local_moe_intermediate = tp.local_moe_intermediate_size(cfg)?;
    let local_shared_intermediate = tp.local_shared_expert_intermediate_size(cfg)?;

    let experts = (0..cfg.num_experts)
        .map(|expert_idx| {
            let gate_proj_name = leak_name(names.expert_gate_proj(expert_idx));
            let up_proj_name = leak_name(names.expert_up_proj(expert_idx));
            let down_proj_name = leak_name(names.expert_down_proj(expert_idx));
            Ok(Qwen35SparseExpert {
                gate_proj: linear_with_base_init(
                    gate_proj_name,
                    cfg.hidden_size,
                    local_moe_intermediate,
                    base_requires_grad,
                    lora_for_name(expert_lora, lora_target_set, gate_proj_name),
                    materialize_frozen_base,
                    store,
                )?,
                up_proj: linear_with_base_init(
                    up_proj_name,
                    cfg.hidden_size,
                    local_moe_intermediate,
                    base_requires_grad,
                    lora_for_name(expert_lora, lora_target_set, up_proj_name),
                    materialize_frozen_base,
                    store,
                )?,
                down_proj: linear_with_base_init(
                    down_proj_name,
                    local_moe_intermediate,
                    cfg.hidden_size,
                    base_requires_grad,
                    lora_for_name(expert_lora, lora_target_set, down_proj_name),
                    materialize_frozen_base,
                    store,
                )?,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(Qwen35SparseMlp {
        router_gate: linear_with_base_init(
            router_gate_name,
            cfg.hidden_size,
            cfg.num_experts,
            base_requires_grad,
            lora_for_name(lora, lora_target_set, router_gate_name),
            materialize_frozen_base,
            store,
        )?,
        shared_gate_proj: linear_with_base_init(
            shared_gate_proj_name,
            cfg.hidden_size,
            local_shared_intermediate,
            base_requires_grad,
            lora_for_name(lora, lora_target_set, shared_gate_proj_name),
            materialize_frozen_base,
            store,
        )?,
        shared_up_proj: linear_with_base_init(
            shared_up_proj_name,
            cfg.hidden_size,
            local_shared_intermediate,
            base_requires_grad,
            lora_for_name(lora, lora_target_set, shared_up_proj_name),
            materialize_frozen_base,
            store,
        )?,
        shared_down_proj: linear_with_base_init(
            shared_down_proj_name,
            local_shared_intermediate,
            cfg.hidden_size,
            base_requires_grad,
            lora_for_name(lora, lora_target_set, shared_down_proj_name),
            materialize_frozen_base,
            store,
        )?,
        shared_expert_gate: linear_with_base_init(
            shared_expert_gate_name,
            cfg.hidden_size,
            1,
            base_requires_grad,
            lora_for_name(lora, lora_target_set, shared_expert_gate_name),
            materialize_frozen_base,
            store,
        )?,
        experts,
        top_k: cfg.num_experts_per_tok,
    })
}

pub(super) fn share_base_attention(
    attention: &mut Qwen35Attention,
    base_attention: &Qwen35Attention,
) -> Result<()> {
    match (attention, base_attention) {
        (Qwen35Attention::Full(attn), Qwen35Attention::Full(base_attn)) => {
            attn.q_proj.set_base_weight(base_attn.q_proj.base_weight());
            attn.k_proj.set_base_weight(base_attn.k_proj.base_weight());
            attn.v_proj.set_base_weight(base_attn.v_proj.base_weight());
            attn.o_proj.set_base_weight(base_attn.o_proj.base_weight());
            attn.q_norm = base_attn.q_norm;
            attn.k_norm = base_attn.k_norm;
            Ok(())
        }
        (Qwen35Attention::Linear(attn), Qwen35Attention::Linear(base_attn)) => {
            attn.in_proj_qkv
                .set_base_weight(base_attn.in_proj_qkv.base_weight());
            attn.in_proj_z
                .set_base_weight(base_attn.in_proj_z.base_weight());
            attn.in_proj_b
                .set_base_weight(base_attn.in_proj_b.base_weight());
            attn.in_proj_a
                .set_base_weight(base_attn.in_proj_a.base_weight());
            attn.conv1d_weight = base_attn.conv1d_weight;
            attn.dt_bias = base_attn.dt_bias;
            attn.a_log = base_attn.a_log;
            attn.norm = base_attn.norm;
            attn.out_proj
                .set_base_weight(base_attn.out_proj.base_weight());
            Ok(())
        }
        _ => Err(Qwen35Error::InvalidConfig(
            "cannot share Qwen3.5 base weights across mismatched attention layer types",
        )),
    }
}

pub(super) fn share_base_mlp(mlp: &mut Qwen35Mlp, base_mlp: &Qwen35Mlp) -> Result<()> {
    match (mlp, base_mlp) {
        (Qwen35Mlp::Dense(mlp), Qwen35Mlp::Dense(base_mlp)) => {
            mlp.gate_proj
                .set_base_weight(base_mlp.gate_proj.base_weight());
            mlp.up_proj.set_base_weight(base_mlp.up_proj.base_weight());
            mlp.down_proj
                .set_base_weight(base_mlp.down_proj.base_weight());
            Ok(())
        }
        (Qwen35Mlp::Sparse(mlp), Qwen35Mlp::Sparse(base_mlp)) => {
            mlp.router_gate
                .set_base_weight(base_mlp.router_gate.base_weight());
            mlp.shared_gate_proj
                .set_base_weight(base_mlp.shared_gate_proj.base_weight());
            mlp.shared_up_proj
                .set_base_weight(base_mlp.shared_up_proj.base_weight());
            mlp.shared_down_proj
                .set_base_weight(base_mlp.shared_down_proj.base_weight());
            mlp.shared_expert_gate
                .set_base_weight(base_mlp.shared_expert_gate.base_weight());
            if mlp.experts.len() != base_mlp.experts.len() {
                return Err(Qwen35Error::InvalidConfig(
                    "cannot share Qwen3.6 MoE base weights across mismatched expert counts",
                ));
            }
            for (expert, base_expert) in mlp.experts.iter_mut().zip(&base_mlp.experts) {
                expert
                    .gate_proj
                    .set_base_weight(base_expert.gate_proj.base_weight());
                expert
                    .up_proj
                    .set_base_weight(base_expert.up_proj.base_weight());
                expert
                    .down_proj
                    .set_base_weight(base_expert.down_proj.base_weight());
            }
            Ok(())
        }
        _ => Err(Qwen35Error::InvalidConfig(
            "cannot share Qwen3.5 base weights across mismatched MLP layer types",
        )),
    }
}

pub(super) fn lora_for_layer(
    lora: Option<LoraConfig>,
    lora_layer_start: Option<usize>,
    layer_idx: usize,
) -> Option<LoraConfig> {
    if lora_layer_start.is_some_and(|start| layer_idx < start) {
        None
    } else {
        lora
    }
}

pub(super) fn lora_for_name(
    lora: Option<LoraConfig>,
    target_set: LoraTargetSet,
    base_name: &str,
) -> Option<LoraConfig> {
    lora.filter(|_| target_set.includes(base_name))
}

pub(super) fn collect_linear_ids(linear: &LinearWithLora, ids: &mut Vec<TensorId>) {
    ids.extend(linear.parameter_name_map().values().copied());
    ids.extend(linear.adapter_ordered().into_iter().map(|(_, id)| id));
}

pub(super) fn collect_mlp_ids(mlp: &Qwen35Mlp, skip_experts: bool, ids: &mut Vec<TensorId>) {
    match mlp {
        Qwen35Mlp::Dense(dense) => {
            collect_linear_ids(&dense.gate_proj, ids);
            collect_linear_ids(&dense.up_proj, ids);
            collect_linear_ids(&dense.down_proj, ids);
        }
        Qwen35Mlp::Sparse(sparse) => {
            collect_linear_ids(&sparse.router_gate, ids);
            collect_linear_ids(&sparse.shared_gate_proj, ids);
            collect_linear_ids(&sparse.shared_up_proj, ids);
            collect_linear_ids(&sparse.shared_down_proj, ids);
            collect_linear_ids(&sparse.shared_expert_gate, ids);
            if !skip_experts {
                for expert in &sparse.experts {
                    collect_linear_ids(&expert.gate_proj, ids);
                    collect_linear_ids(&expert.up_proj, ids);
                    collect_linear_ids(&expert.down_proj, ids);
                }
            }
        }
    }
}

pub(super) fn register_mlp(
    param_names: &mut HashMap<&'static str, TensorId>,
    adapter_names: &mut HashMap<&'static str, TensorId>,
    register_named: &mut impl FnMut(&mut HashMap<&'static str, TensorId>, &'static str, TensorId),
    mlp: &Qwen35Mlp,
) {
    match mlp {
        Qwen35Mlp::Dense(dense) => {
            register_linear(param_names, adapter_names, register_named, &dense.gate_proj);
            register_linear(param_names, adapter_names, register_named, &dense.up_proj);
            register_linear(param_names, adapter_names, register_named, &dense.down_proj);
        }
        Qwen35Mlp::Sparse(sparse) => {
            register_linear(
                param_names,
                adapter_names,
                register_named,
                &sparse.router_gate,
            );
            register_linear(
                param_names,
                adapter_names,
                register_named,
                &sparse.shared_gate_proj,
            );
            register_linear(
                param_names,
                adapter_names,
                register_named,
                &sparse.shared_up_proj,
            );
            register_linear(
                param_names,
                adapter_names,
                register_named,
                &sparse.shared_down_proj,
            );
            register_linear(
                param_names,
                adapter_names,
                register_named,
                &sparse.shared_expert_gate,
            );
            for expert in &sparse.experts {
                register_linear(
                    param_names,
                    adapter_names,
                    register_named,
                    &expert.gate_proj,
                );
                register_linear(param_names, adapter_names, register_named, &expert.up_proj);
                register_linear(
                    param_names,
                    adapter_names,
                    register_named,
                    &expert.down_proj,
                );
            }
        }
    }
}

pub(super) fn insert_materialized_linear(
    map: &mut HashMap<&'static str, TensorId>,
    linear: &LinearWithLora,
    store: &mut TensorStore,
) -> Result<()> {
    let tensor = linear.merged_tensor(store)?;
    let merged = store.alloc(tensor);
    for (name, _) in linear.parameter_name_map() {
        map.insert(name, merged);
    }
    Ok(())
}

pub(super) fn insert_materialized_mlp_params(
    map: &mut HashMap<&'static str, TensorId>,
    mlp: &Qwen35Mlp,
    store: &mut TensorStore,
) -> Result<()> {
    match mlp {
        Qwen35Mlp::Dense(dense) => {
            insert_materialized_linear(map, &dense.gate_proj, store)?;
            insert_materialized_linear(map, &dense.up_proj, store)?;
            insert_materialized_linear(map, &dense.down_proj, store)?;
        }
        Qwen35Mlp::Sparse(sparse) => {
            insert_materialized_linear(map, &sparse.router_gate, store)?;
            insert_materialized_linear(map, &sparse.shared_gate_proj, store)?;
            insert_materialized_linear(map, &sparse.shared_up_proj, store)?;
            insert_materialized_linear(map, &sparse.shared_down_proj, store)?;
            insert_materialized_linear(map, &sparse.shared_expert_gate, store)?;
            for expert in &sparse.experts {
                insert_materialized_linear(map, &expert.gate_proj, store)?;
                insert_materialized_linear(map, &expert.up_proj, store)?;
                insert_materialized_linear(map, &expert.down_proj, store)?;
            }
        }
    }
    Ok(())
}

pub(super) fn normal_parameter(
    name: &'static str,
    shape: &[usize],
    std: f32,
    requires_grad: bool,
    store: &mut TensorStore,
) -> Result<TensorId> {
    let mut state = seed_from_name(name);
    let size = shape.iter().product();
    let mut data = Vec::with_capacity(size);
    while data.len() < size {
        let u1 = next_uniform(&mut state).max(f32::MIN_POSITIVE);
        let u2 = next_uniform(&mut state);
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = TAU * u2;
        data.push(std * radius * theta.cos());
        if data.len() < size {
            data.push(std * radius * theta.sin());
        }
    }
    Ok(store.alloc(Tensor::new(data, shape.to_vec(), requires_grad)?))
}

pub(super) fn normal_or_unmaterialized_parameter(
    name: &'static str,
    shape: &[usize],
    std: f32,
    requires_grad: bool,
    materialize_frozen_base: bool,
    store: &mut TensorStore,
) -> Result<TensorId> {
    if requires_grad || materialize_frozen_base {
        normal_parameter(name, shape, std, requires_grad, store)
    } else {
        let _ = name;
        Ok(store.alloc(Tensor::unmaterialized(shape.to_vec(), false)?))
    }
}

pub(super) fn ones_parameter(
    name: &'static str,
    shape: &[usize],
    requires_grad: bool,
    store: &mut TensorStore,
) -> Result<TensorId> {
    let _ = name;
    Ok(store.alloc(Tensor::new(
        vec![1.0; shape.iter().product()],
        shape.to_vec(),
        requires_grad,
    )?))
}

pub(super) fn ones_or_unmaterialized_parameter(
    name: &'static str,
    shape: &[usize],
    requires_grad: bool,
    materialize_frozen_base: bool,
    store: &mut TensorStore,
) -> Result<TensorId> {
    if requires_grad || materialize_frozen_base {
        ones_parameter(name, shape, requires_grad, store)
    } else {
        let _ = name;
        Ok(store.alloc(Tensor::unmaterialized(shape.to_vec(), false)?))
    }
}
