use std::collections::HashSet;

use autograd::TensorStore;
use qwen35_spec::Qwen35AttentionTensorNames;
use train::{
    causal_lm::{
        build_adapter_registry, build_materialized_registry, build_registry, trainable_params,
    },
    lora::LoraConfig,
    qwen3::Qwen3Config,
    qwen3::Qwen3Model,
    qwen35::{LayerType, Qwen35Config, Qwen35Model},
};

type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

#[test]
fn qwen3_lora_freezes_base_and_materializes_weights() -> TestResult {
    let cfg = Qwen3Config {
        vocab_size: 128,
        hidden_size: 32,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 8,
        intermediate_size: 64,
        max_position_embeddings: 16,
        rms_norm_eps: 1.0e-6,
        rope_theta: 10_000.0,
        tie_word_embeddings: false,
    };
    let lora = Some(LoraConfig {
        rank: 2,
        alpha: 4.0,
    });

    let mut store = TensorStore::default();
    let model = Qwen3Model::new_with_lora(&cfg, lora, &mut store)?;

    let base_registry = build_registry(&model);
    let adapter_registry = build_adapter_registry(&model);
    let param_map = model.param_name_map();
    let adapter_map = model.adapter_name_map();

    assert!(base_registry.get(cfg.embed_tokens_tensor_name()).is_some());
    let q_proj_name = cfg.layer_tensor_names(0).q_proj;
    assert!(base_registry.get(q_proj_name.as_str()).is_some());
    assert!(
        base_registry
            .get(format!("{q_proj_name}.lora_a").as_str())
            .is_none()
    );
    assert!(adapter_registry.get(q_proj_name.as_str()).is_none());
    assert!(
        adapter_registry
            .get(format!("{q_proj_name}.lora_a").as_str())
            .is_some()
    );

    assert!(
        param_map
            .values()
            .all(|&id| { !store.get(id).expect("base tensor exists").requires_grad })
    );
    assert!(
        adapter_map
            .values()
            .all(|&id| { store.get(id).expect("adapter tensor exists").requires_grad })
    );

    let trainable = trainable_params(&model, &store);
    let trainable_set = trainable.into_iter().collect::<HashSet<_>>();
    let adapter_ids = adapter_map.values().copied().collect::<HashSet<_>>();
    assert_eq!(trainable_set, adapter_ids);

    let q_proj_a = *adapter_map
        .get(format!("{q_proj_name}.lora_a").as_str())
        .expect("q_proj adapter A");
    let q_proj_b = *adapter_map
        .get(format!("{q_proj_name}.lora_b").as_str())
        .expect("q_proj adapter B");
    {
        let tensor = store.get_mut(q_proj_a).expect("adapter A exists");
        tensor.data[0] = 1.0;
    }
    {
        let tensor = store.get_mut(q_proj_b).expect("adapter B exists");
        tensor.data[0] = 2.0;
    }

    let mut tape = autograd::Tape::new();
    let materialized = build_materialized_registry(&model, &mut store, &mut tape)?;
    assert!(
        materialized
            .get(format!("{q_proj_name}.lora_a").as_str())
            .is_none()
    );
    let base_q = *param_map.get(q_proj_name.as_str()).expect("base q_proj");
    let merged_q = materialized
        .get(q_proj_name.as_str())
        .expect("materialized q_proj present");
    let base = store.to_host(base_q)?;
    let merged = store.to_host(merged_q)?;
    assert_ne!(base, merged, "merged q_proj should include LoRA delta");

    let mut frozen_store = TensorStore::default();
    let frozen = model.clone_frozen(&mut frozen_store);
    let mut frozen_tape = autograd::Tape::new();
    let frozen_logits =
        frozen.forward_tokens(&[1, 2, 3, 4], &mut frozen_store, &mut frozen_tape)?;
    let frozen_shape = frozen_store
        .get(frozen_logits)
        .expect("frozen logits exist")
        .shape
        .clone();
    assert_eq!(frozen_shape, vec![1, 4, cfg.vocab_size]);

    Ok(())
}

#[test]
fn qwen35_lora_freezes_base_and_materializes_weights() -> TestResult {
    let cfg = Qwen35Config {
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        vocab_size: 128,
        rms_norm_eps: 1.0e-6,
        stop_token_ids: vec![2],
        bos_token_id: Some(1),
        eos_token_id: 2,
        tie_word_embeddings: false,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 8,
        linear_num_key_heads: 4,
        linear_key_head_dim: 8,
        linear_num_value_heads: 4,
        linear_value_head_dim: 8,
        linear_conv_kernel_dim: 4,
        rope_theta: 10_000.0,
        partial_rotary_factor: 1.0,
        rotary_dim: 8,
        rope_cache_len_hint: Some(16),
        layer_types: vec![LayerType::FullAttention; 2],
        num_experts: 0,
        num_experts_per_tok: 0,
        decoder_sparse_step: 1,
        moe_intermediate_size: 0,
        shared_expert_intermediate_size: 0,
        norm_topk_prob: true,
        mlp_only_layers: Vec::new(),
    };
    let lora = Some(LoraConfig {
        rank: 2,
        alpha: 4.0,
    });

    let mut store = TensorStore::default();
    let model = Qwen35Model::new_with_lora(&cfg, lora, &mut store)?;

    let base_registry = build_registry(&model);
    let adapter_registry = build_adapter_registry(&model);
    let param_map = model.param_name_map();
    let adapter_map = model.adapter_name_map();

    let layer_names = cfg.layer_tensor_names(0);
    let Qwen35AttentionTensorNames::Full(attn_names) = layer_names.attention else {
        unreachable!("test config uses full attention");
    };
    assert!(base_registry.get(attn_names.q_proj.as_str()).is_some());
    assert!(
        base_registry
            .get(format!("{}.lora_a", attn_names.q_proj).as_str())
            .is_none()
    );
    assert!(adapter_registry.get(attn_names.q_proj.as_str()).is_none());
    assert!(
        adapter_registry
            .get(format!("{}.lora_a", attn_names.q_proj).as_str())
            .is_some()
    );

    assert!(
        param_map
            .values()
            .all(|&id| { !store.get(id).expect("base tensor exists").requires_grad })
    );
    assert!(
        adapter_map
            .values()
            .all(|&id| { store.get(id).expect("adapter tensor exists").requires_grad })
    );

    let trainable = trainable_params(&model, &store);
    let trainable_set = trainable.into_iter().collect::<HashSet<_>>();
    let adapter_ids = adapter_map.values().copied().collect::<HashSet<_>>();
    assert_eq!(trainable_set, adapter_ids);

    let q_proj_a = *adapter_map
        .get(format!("{}.lora_a", attn_names.q_proj).as_str())
        .expect("q_proj adapter A");
    let q_proj_b = *adapter_map
        .get(format!("{}.lora_b", attn_names.q_proj).as_str())
        .expect("q_proj adapter B");
    {
        let tensor = store.get_mut(q_proj_a).expect("adapter A exists");
        tensor.data[0] = 1.0;
    }
    {
        let tensor = store.get_mut(q_proj_b).expect("adapter B exists");
        tensor.data[0] = 2.0;
    }

    let mut tape = autograd::Tape::new();
    let materialized = build_materialized_registry(&model, &mut store, &mut tape)?;
    assert!(
        materialized
            .get(format!("{}.lora_a", attn_names.q_proj).as_str())
            .is_none()
    );
    let base_q = *param_map
        .get(attn_names.q_proj.as_str())
        .expect("base q_proj");
    let merged_q = materialized
        .get(attn_names.q_proj.as_str())
        .expect("materialized q_proj present");
    let base = store.to_host(base_q)?;
    let merged = store.to_host(merged_q)?;
    assert_ne!(base, merged, "merged q_proj should include LoRA delta");

    let mut frozen_store = TensorStore::default();
    let frozen = model.clone_frozen(&mut frozen_store);
    let mut frozen_tape = autograd::Tape::new();
    let frozen_logits =
        frozen.forward_tokens(&[1, 2, 3, 4], &mut frozen_store, &mut frozen_tape)?;
    let frozen_shape = frozen_store
        .get(frozen_logits)
        .expect("frozen logits exist")
        .shape
        .clone();
    assert_eq!(frozen_shape, vec![1, 4, cfg.vocab_size]);

    Ok(())
}
