use std::collections::{HashMap, HashSet};
use std::error::Error;
#[cfg(feature = "cuda")]
use std::{env, sync::Arc};

#[cfg(feature = "cuda")]
use autograd::backend_cuda::CudaBackend;
use autograd::{
    BackwardOp, Tape, TensorId, TensorStore,
    ops::{mul, sum},
};

use super::{LayerType, Qwen35Config, Qwen35KvCache, Qwen35Model};
use crate::lora::{LoraConfig, LoraTargetSet};

type TestResult<T = ()> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

fn tiny_qwen35_config(max_seq_len: usize) -> Qwen35Config {
    Qwen35Config {
        hidden_size: 16,
        intermediate_size: 32,
        num_hidden_layers: 2,
        vocab_size: 16,
        rms_norm_eps: 1.0e-6,
        stop_token_ids: vec![15],
        bos_token_id: Some(1),
        eos_token_id: 15,
        tie_word_embeddings: false,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 8,
        linear_num_key_heads: 2,
        linear_key_head_dim: 8,
        linear_num_value_heads: 2,
        linear_value_head_dim: 8,
        linear_conv_kernel_dim: 4,
        rope_theta: 10_000.0,
        rope_scaling: None,
        partial_rotary_factor: 1.0,
        rotary_dim: 8,
        rope_cache_len_hint: Some(max_seq_len),
        layer_types: vec![LayerType::FullAttention; 2],
        num_experts: 0,
        num_experts_per_tok: 0,
        decoder_sparse_step: 1,
        moe_intermediate_size: 0,
        shared_expert_intermediate_size: 0,
        norm_topk_prob: true,
        mlp_only_layers: Vec::new(),
        full_attn_gated: true,
    }
}

fn tiny_qwen36_moe_config(max_seq_len: usize) -> Qwen35Config {
    Qwen35Config {
        hidden_size: 8,
        intermediate_size: 0,
        num_hidden_layers: 1,
        vocab_size: 12,
        rms_norm_eps: 1.0e-6,
        stop_token_ids: vec![11],
        bos_token_id: Some(1),
        eos_token_id: 11,
        tie_word_embeddings: false,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        linear_num_key_heads: 2,
        linear_key_head_dim: 4,
        linear_num_value_heads: 2,
        linear_value_head_dim: 4,
        linear_conv_kernel_dim: 4,
        rope_theta: 10_000.0,
        rope_scaling: None,
        partial_rotary_factor: 1.0,
        rotary_dim: 4,
        rope_cache_len_hint: Some(max_seq_len),
        layer_types: vec![LayerType::FullAttention],
        num_experts: 3,
        num_experts_per_tok: 2,
        decoder_sparse_step: 1,
        moe_intermediate_size: 6,
        shared_expert_intermediate_size: 6,
        norm_topk_prob: true,
        mlp_only_layers: Vec::new(),
        full_attn_gated: true,
    }
}

fn logits_host(store: &mut TensorStore, logits: TensorId) -> TestResult<Vec<f32>> {
    Ok(store.to_host(logits)?)
}

fn squared_logits_loss(
    model: &Qwen35Model,
    store: &mut TensorStore,
    tape: &mut Tape,
    tokens: &[u32],
    positions: &[u32],
) -> TestResult<TensorId> {
    let logits = model.forward(store, tape, tokens, positions)?;
    let sq = mul(logits, logits, store, tape)?;
    Ok(sum(sq, store, tape)?)
}

fn squared_logits_loss_value(
    model: &Qwen35Model,
    store: &mut TensorStore,
    tokens: &[u32],
    positions: &[u32],
) -> TestResult<f32> {
    let mut tape = Tape::new();
    tape.set_enabled(false);
    let logits = model.forward(store, &mut tape, tokens, positions)?;
    let host = store.to_host(logits)?;
    Ok(host.iter().map(|value| value * value).sum())
}

fn set_tensor_element(
    store: &mut TensorStore,
    tensor_id: TensorId,
    index: usize,
    value: f32,
) -> TestResult {
    let tensor = store
        .get_mut(tensor_id)
        .ok_or_else(|| format!("missing tensor {tensor_id}"))?;
    let slot = tensor
        .data
        .get_mut(index)
        .ok_or_else(|| format!("tensor {tensor_id} missing index {index}"))?;
    *slot = value;
    Ok(())
}

fn fill_tensor(store: &mut TensorStore, tensor_id: TensorId, value: f32) -> TestResult {
    let tensor = store
        .get_mut(tensor_id)
        .ok_or_else(|| format!("missing tensor {tensor_id}"))?;
    for slot in &mut tensor.data {
        *slot = value;
    }
    tensor.device_handle = None;
    Ok(())
}

fn set_moe_router_tie(
    store: &mut TensorStore,
    model: &Qwen35Model,
    layer_idx: usize,
) -> TestResult {
    let name = format!("model.language_model.layers.{layer_idx}.mlp.gate.weight");
    let params = model.param_name_map();
    let router = *params
        .get(name.as_str())
        .ok_or_else(|| format!("missing router tensor {name}"))?;
    fill_tensor(store, router, 0.0)
}

fn first_nonzero_grad_index(store: &mut TensorStore, grad_id: TensorId) -> TestResult<usize> {
    let grad = store.to_host(grad_id)?;
    grad.iter()
        .enumerate()
        .filter(|(_, value)| value.abs() > 1.0e-7)
        .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
        .map(|(idx, _)| idx)
        .ok_or_else(|| "gradient has no stable non-zero element".into())
}

fn greedy_next(host: &[f32], seq_len: usize, vocab: usize) -> u32 {
    let row = &host[(seq_len - 1) * vocab..seq_len * vocab];
    row.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx as u32)
        .expect("non-empty vocab")
}

#[test]
fn qwen35_lora_layer_start_limits_adapters_and_tape_prefix() -> TestResult {
    let mut store = TensorStore::default();
    let cfg = tiny_qwen35_config(8);
    let lora = LoraConfig {
        rank: 2,
        alpha: 4.0,
    };
    let model = Qwen35Model::new_with_lora_targets_layer_start(
        &cfg,
        lora,
        LoraTargetSet::AllLinear,
        Some(1),
        &mut store,
    )?;
    assert_eq!(model.lora_layer_start(), Some(1));

    let adapters = model.adapter_name_map();
    assert!(!adapters.is_empty(), "suffix LoRA must allocate adapters");
    for (name, &id) in &adapters {
        assert!(
            !name.contains(".layers.0."),
            "prefix layer adapter should not exist: {name}"
        );
        assert!(
            name.contains(".layers.1."),
            "suffix adapter must live in layer 1 for the tiny 2-layer config: {name}"
        );
        let tensor = store
            .get(id)
            .ok_or_else(|| format!("missing adapter {name}"))?;
        assert!(tensor.requires_grad, "adapter {name} must be trainable");
    }

    let prefix_params = model
        .param_name_map()
        .into_iter()
        .filter(|(name, _)| name.contains(".layers.0."))
        .map(|(name, id)| (id, name))
        .collect::<HashMap<_, _>>();
    let prefix_param_ids = prefix_params.keys().copied().collect::<HashSet<_>>();
    let suffix_adapter_ids = adapters.values().copied().collect::<HashSet<_>>();

    let tokens = [1_u32, 3];
    let positions = [0_u32, 1];
    let mut tape = Tape::new();
    let loss = squared_logits_loss(&model, &mut store, &mut tape, &tokens, &positions)?;

    let mut prefix_refs = Vec::new();
    let mut saw_suffix_adapter_site = false;
    for entry in &tape.entries {
        if let Some(site) = entry.profile_site() {
            assert!(
                !site.contains(".layers.0."),
                "tape must not record prefix layer matmul site {site}"
            );
            saw_suffix_adapter_site |= site.contains(".layers.1.") && site.ends_with(".lora_b");
        }
        for input_id in &entry.input_ids {
            if prefix_param_ids.contains(input_id) {
                let name = prefix_params.get(input_id).copied().unwrap_or("<unknown>");
                prefix_refs.push((entry.op.name(), name));
            }
        }
    }
    assert!(
        prefix_refs.is_empty(),
        "tape must not reference prefix layer params: {prefix_refs:?}"
    );
    assert!(
        saw_suffix_adapter_site,
        "tape must include trainable suffix LoRA adapter work"
    );

    let grads = tape.backward(loss, &mut store)?;
    for (name, &id) in &adapters {
        assert!(grads.contains_key(&id), "missing adapter grad for {name}");
    }
    assert!(
        grads.keys().all(|id| !prefix_param_ids.contains(id)),
        "prefix layer params must not receive gradients"
    );
    assert!(
        grads.keys().any(|id| suffix_adapter_ids.contains(id)),
        "at least one suffix adapter must receive a gradient"
    );
    Ok(())
}

#[test]
fn qwen35_rollout_kv_cache_matches_full_forward_tokens() -> TestResult {
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    tape.set_enabled(false);

    let cfg = tiny_qwen35_config(16);
    let model = Qwen35Model::new_for_eval(&cfg, &mut store)?;
    let vocab = cfg.vocab_size;
    let mut rollout = vec![1_u32, 3, 8];
    let mut cache = Qwen35KvCache::new(&model, rollout.len() + 5);

    for step in 0..5 {
        let full_positions = (0..rollout.len() as u32).collect::<Vec<_>>();
        let full_logits = model.forward(&mut store, &mut tape, &rollout, &full_positions)?;
        let full_host = logits_host(&mut store, full_logits)?;
        let full_next = greedy_next(&full_host, rollout.len(), vocab);

        let (cached_input, cached_positions, cached_seq_len) = if step == 0 {
            (rollout.clone(), full_positions, 1)
        } else {
            let last = *rollout.last().expect("rollout stays non-empty");
            (vec![last], vec![(rollout.len() - 1) as u32], 1)
        };
        let cached_logits = model.forward_rollout_cached(
            &mut store,
            &mut tape,
            &cached_input,
            &cached_positions,
            &mut cache,
        )?;
        let cached_host = logits_host(&mut store, cached_logits)?;
        let cached_next = greedy_next(&cached_host, cached_seq_len, vocab);

        let full_row = &full_host[(rollout.len() - 1) * vocab..rollout.len() * vocab];
        let cached_row = &cached_host[(cached_seq_len - 1) * vocab..cached_seq_len * vocab];
        let max_abs = full_row
            .iter()
            .zip(cached_row.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs <= 1.0e-5,
            "cached rollout logits must match full-forward last row at step {step}; max_abs={max_abs}"
        );
        assert_eq!(
            cached_next, full_next,
            "cached rollout token diverged at step {step}"
        );

        rollout.push(full_next);
    }

    assert_eq!(cache.seq_len, rollout.len() - 1);
    Ok(())
}

#[test]
fn qwen35_logits_window_matches_full_forward_slice() -> TestResult {
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    tape.set_enabled(false);

    let cfg = tiny_qwen35_config(16);
    let model = Qwen35Model::new_for_eval(&cfg, &mut store)?;
    let tokens = vec![1_u32, 3, 8, 4, 2];
    let positions = (0..tokens.len() as u32).collect::<Vec<_>>();
    let full_logits = model.forward(&mut store, &mut tape, &tokens, &positions)?;
    let full_host = logits_host(&mut store, full_logits)?;

    let window = super::SequenceWindow { start: 1, end: 4 };
    let window_logits =
        model.forward_logits_window(&mut store, &mut tape, &tokens, &positions, window)?;
    let window_host = logits_host(&mut store, window_logits)?;

    let expected_len = window.len() * cfg.vocab_size;
    assert_eq!(window_host.len(), expected_len);
    for local_pos in 0..window.len() {
        let full_start = (window.start + local_pos) * cfg.vocab_size;
        let window_start = local_pos * cfg.vocab_size;
        let full_row = &full_host[full_start..full_start + cfg.vocab_size];
        let window_row = &window_host[window_start..window_start + cfg.vocab_size];
        let max_abs = full_row
            .iter()
            .zip(window_row.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs <= 1.0e-5,
            "window row {local_pos} must match full logits slice; max_abs={max_abs}"
        );
    }
    Ok(())
}

#[test]
fn qwen36_moe_lora_param_names_match_checkpoint_contract() -> TestResult {
    let mut store = TensorStore::default();
    let cfg = tiny_qwen36_moe_config(8);
    let lora = LoraConfig {
        rank: 2,
        alpha: 4.0,
    };
    let model =
        Qwen35Model::new_with_lora_targets(&cfg, lora, LoraTargetSet::AllLinear, &mut store)?;
    let params = model.param_name_map();
    let adapters = model.adapter_name_map();
    for name in [
        "model.language_model.layers.0.mlp.gate.weight",
        "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
        "model.language_model.layers.0.mlp.shared_expert.up_proj.weight",
        "model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
        "model.language_model.layers.0.mlp.shared_expert_gate.weight",
        "model.language_model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.language_model.layers.0.mlp.experts.0.up_proj.weight",
        "model.language_model.layers.0.mlp.experts.0.down_proj.weight",
        "lm_head.weight",
    ] {
        assert!(params.contains_key(name), "missing param {name}");
    }
    assert!(
        !params.contains_key("model.language_model.layers.0.mlp.gate_proj.weight"),
        "MoE layers must not expose dense MLP gate_proj names"
    );
    assert!(
        adapters.contains_key("model.language_model.layers.0.mlp.experts.0.up_proj.weight.lora_b"),
        "routed expert LoRA-B must be trainable"
    );
    assert!(
        adapters.contains_key(
            "model.language_model.layers.0.mlp.shared_expert.down_proj.weight.lora_b"
        ),
        "shared expert LoRA-B must be trainable"
    );
    Ok(())
}

#[test]
fn qwen36_checkpoint_load_constructor_keeps_frozen_base_unmaterialized() -> TestResult {
    let mut store = TensorStore::default();
    let cfg = tiny_qwen36_moe_config(8);
    let lora = LoraConfig {
        rank: 2,
        alpha: 4.0,
    };
    let model = Qwen35Model::new_with_lora_targets_for_checkpoint_load(
        &cfg,
        lora,
        LoraTargetSet::AllLinear,
        false,
        &mut store,
    )?;
    let params = model.param_name_map();
    let adapters = model.adapter_name_map();
    let base_name = "model.language_model.layers.0.mlp.experts.0.up_proj.weight";
    let adapter_name = "model.language_model.layers.0.mlp.experts.0.up_proj.weight.lora_b";
    let base = *params
        .get(base_name)
        .ok_or_else(|| format!("missing base tensor {base_name}"))?;
    let adapter = *adapters
        .get(adapter_name)
        .ok_or_else(|| format!("missing adapter tensor {adapter_name}"))?;

    let base_tensor = store
        .get(base)
        .ok_or_else(|| format!("missing tensor {base}"))?;
    assert_eq!(
        base_tensor.shape,
        vec![cfg.moe_intermediate_size, cfg.hidden_size]
    );
    assert_eq!(base_tensor.dirty, autograd::tensor::Dirty::Device);
    assert!(base_tensor.data.is_empty());
    assert!(base_tensor.device_handle.is_none());
    assert!(!base_tensor.requires_grad);

    let adapter_tensor = store
        .get(adapter)
        .ok_or_else(|| format!("missing tensor {adapter}"))?;
    assert!(adapter_tensor.requires_grad);
    assert_eq!(
        adapter_tensor.data.len(),
        cfg.moe_intermediate_size * lora.rank
    );
    assert_eq!(adapter_tensor.dirty, autograd::tensor::Dirty::Host);
    Ok(())
}

#[test]
fn qwen36_moe_lora_gradient_matches_finite_difference() -> TestResult {
    let mut store = TensorStore::default();
    let cfg = tiny_qwen36_moe_config(8);
    let lora = LoraConfig {
        rank: 2,
        alpha: 4.0,
    };
    let model =
        Qwen35Model::new_with_lora_targets(&cfg, lora, LoraTargetSet::AllLinear, &mut store)?;
    set_moe_router_tie(&mut store, &model, 0)?;
    let target_name = "model.language_model.layers.0.mlp.experts.0.up_proj.weight.lora_b";
    let target = *model
        .adapter_name_map()
        .get(target_name)
        .ok_or_else(|| format!("missing adapter {target_name}"))?;
    let tokens = [1_u32, 3];
    let positions = [0_u32, 1];

    let mut tape = Tape::new();
    let loss = squared_logits_loss(&model, &mut store, &mut tape, &tokens, &positions)?;
    let grads = tape.backward(loss, &mut store)?;
    let grad_id = *grads
        .get(&target)
        .ok_or_else(|| format!("missing grad for {target_name}"))?;
    let probe_index = first_nonzero_grad_index(&mut store, grad_id)?;
    let analytic = store.to_host(grad_id)?[probe_index];

    let base = store
        .get(target)
        .and_then(|tensor| tensor.data.get(probe_index).copied())
        .ok_or_else(|| format!("missing target element {probe_index}"))?;
    let eps = 1.0e-3_f32;
    set_tensor_element(&mut store, target, probe_index, base + eps)?;
    let plus = squared_logits_loss_value(&model, &mut store, &tokens, &positions)?;
    set_tensor_element(&mut store, target, probe_index, base - eps)?;
    let minus = squared_logits_loss_value(&model, &mut store, &tokens, &positions)?;
    set_tensor_element(&mut store, target, probe_index, base)?;
    let numeric = (plus - minus) / (2.0 * eps);
    let denom = analytic.abs().max(numeric.abs()).max(1.0e-6);
    let rel_err = (analytic - numeric).abs() / denom;
    eprintln!(
        "qwen36_moe_lora_fd target={target_name}[{probe_index}] analytic={analytic:.9e} numeric={numeric:.9e} rel_err={rel_err:.3e}"
    );
    assert!(
        rel_err <= 1.0e-2,
        "Qwen35 integrated MoE LoRA finite diff failed: analytic={analytic:.9e} numeric={numeric:.9e} rel_err={rel_err:.3e}"
    );
    Ok(())
}

fn run_qwen35_checkpoint_lora_fd_gate(
    store: &mut TensorStore,
) -> TestResult<(&'static str, usize, f32, f32, f32)> {
    let cfg = tiny_qwen35_config(8);
    let lora = LoraConfig {
        rank: 2,
        alpha: 4.0,
    };
    let mut model =
        Qwen35Model::new_with_lora_targets(&cfg, lora, LoraTargetSet::AllLinear, store)?;
    model.set_gradient_checkpointing(true);
    assert!(model.gradient_checkpointing());
    // The auto-gate declines the tiny shape on a real GPU (modeled ≪ free);
    // this test exists to exercise the checkpoint path, so force-engage.
    // SAFETY: forced checkpointing only changes which (equivalent) path
    // concurrent tests take.
    unsafe {
        std::env::set_var("ARLE_FORCE_CHECKPOINT", "1");
    }

    let tokens = [1_u32, 3, 8];
    let positions = [0_u32, 1, 2];
    let mut tape = Tape::new();
    let loss = squared_logits_loss(&model, store, &mut tape, &tokens, &positions)?;
    let checkpoint_entries = tape
        .entries
        .iter()
        .filter(|entry| entry.op == BackwardOp::Checkpoint)
        .count();
    assert_eq!(
        checkpoint_entries, cfg.num_hidden_layers,
        "one checkpoint entry per layer is required"
    );
    let adapter_ids = model
        .adapter_name_map()
        .values()
        .copied()
        .collect::<HashSet<_>>();
    let frozen_param_ids = model
        .param_name_map()
        .values()
        .copied()
        .collect::<HashSet<_>>();
    let mut saw_checkpoint_adapter = false;
    for entry in tape
        .entries
        .iter()
        .filter(|entry| entry.op == BackwardOp::Checkpoint)
    {
        for &input_id in entry.input_ids.iter().skip(1) {
            let tensor = store
                .get(input_id)
                .ok_or_else(|| format!("checkpoint input tensor {input_id} missing"))?;
            assert!(
                tensor.requires_grad,
                "checkpoint input {input_id} must be trainable; frozen base params would force useless large gradient computation"
            );
            assert!(
                !frozen_param_ids.contains(&input_id),
                "checkpoint input {input_id} leaked a frozen base parameter"
            );
            saw_checkpoint_adapter |= adapter_ids.contains(&input_id);
        }
    }
    assert!(
        saw_checkpoint_adapter,
        "checkpoint inputs must include trainable LoRA adapters"
    );
    let grads = tape.backward(loss, store)?;

    let mut best: Option<(&'static str, TensorId, usize, f32)> = None;
    let mut adapters = model.adapter_name_map().into_iter().collect::<Vec<_>>();
    adapters.sort_by_key(|(name, _)| *name);
    for (name, tensor_id) in adapters {
        let Some(&grad_id) = grads.get(&tensor_id) else {
            continue;
        };
        let grad = store.to_host(grad_id)?;
        for (index, &value) in grad.iter().enumerate() {
            if best
                .as_ref()
                .is_none_or(|(_, _, _, current)| value.abs() > current.abs())
            {
                best = Some((name, tensor_id, index, value));
            }
        }
    }

    let (name, tensor_id, index, analytic) =
        best.ok_or("checkpointed LoRA backward produced no adapter gradients")?;
    assert!(
        analytic.abs() > 1.0e-8,
        "finite-diff probe gradient is too small for a relative gate: {name}[{index}]={analytic}"
    );
    let base_value = store
        .to_host(tensor_id)?
        .get(index)
        .copied()
        .ok_or("probe tensor index missing")?;
    let eps = 1.0e-3_f32;
    set_tensor_element(store, tensor_id, index, base_value + eps)?;
    let loss_plus = squared_logits_loss_value(&model, store, &tokens, &positions)?;
    set_tensor_element(store, tensor_id, index, base_value - eps)?;
    let loss_minus = squared_logits_loss_value(&model, store, &tokens, &positions)?;
    set_tensor_element(store, tensor_id, index, base_value)?;

    let numeric = (loss_plus - loss_minus) / (2.0 * eps);
    let rel_err = (analytic - numeric).abs() / numeric.abs().max(analytic.abs()).max(1.0e-12);
    eprintln!(
        "qwen35_checkpoint_fd name={name} index={index} eps={eps:.1e} analytic={analytic:.9e} numeric={numeric:.9e} rel_err={rel_err:.3e}"
    );
    assert!(
        rel_err <= 1.0e-2,
        "checkpointed LoRA finite diff failed for {name}[{index}]: analytic={analytic:.9e} numeric={numeric:.9e} rel_err={rel_err:.3e}"
    );
    Ok((name, index, analytic, numeric, rel_err))
}

#[test]
fn qwen35_gradient_checkpointing_lora_finite_diff_gate() -> TestResult {
    let mut store = TensorStore::default();
    run_qwen35_checkpoint_lora_fd_gate(&mut store)?;
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn qwen35_gradient_checkpointing_lora_cuda_finite_diff_gate() -> TestResult {
    let ordinal = env::var("ARLE_CUDA_TEST_DEVICE")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(0);
    let backend = Arc::new(CudaBackend::new(ordinal)?);
    let mut store = TensorStore::with_backend(backend);
    run_qwen35_checkpoint_lora_fd_gate(&mut store)?;
    Ok(())
}

/// Issue #201 guard: `sync_lora_from_store` skips re-pointing a frozen base at
/// the engine's merged bytes iff the projection is LoRA-targeted, keyed by
/// `{param_name}.lora_a` in the adapter map. If either map's naming drifts, the
/// skip silently stops matching and the trainer double-applies the LoRA delta.
#[test]
fn lora_targeted_projections_are_never_repointed_at_merged_bytes() -> TestResult {
    let mut store = TensorStore::default();
    let cfg = tiny_qwen35_config(8);
    let lora = LoraConfig {
        rank: 2,
        alpha: 4.0,
    };
    let model =
        Qwen35Model::new_with_lora_targets(&cfg, lora, LoraTargetSet::AttentionQv, &mut store)?;
    let params = model.param_name_map();
    let adapters = model.adapter_name_map();

    let mut targeted = 0;
    for &name in params.keys() {
        let merged = crate::infer_student::engine_bytes_are_merged(name, &adapters);
        assert_eq!(
            merged,
            LoraTargetSet::AttentionQv.includes(name),
            "sync_lora_from_store skip predicate disagrees with the LoRA target set for {name}"
        );
        targeted += merged as usize;
    }
    // Both arms must engage: 2 layers x (q_proj + v_proj) targeted, rest not.
    assert_eq!(targeted, 4);
    assert!(params.len() > targeted);
    Ok(())
}
