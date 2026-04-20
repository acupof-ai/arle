use std::collections::HashSet;

use autograd::{Tape, TensorStore};
use qwen35_spec::Qwen35AttentionTensorNames;
use train::qwen35::{LayerType, Qwen35Config, Qwen35Model};

type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

fn tiny_full_attn_cfg() -> Qwen35Config {
    Qwen35Config {
        hidden_size: 16,
        intermediate_size: 32,
        num_hidden_layers: 2,
        vocab_size: 64,
        rms_norm_eps: 1.0e-6,
        stop_token_ids: vec![2],
        bos_token_id: Some(1),
        eos_token_id: 2,
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
    }
}

#[test]
fn qwen35_batch_forward_matches_repeated_single_forward() -> TestResult {
    let cfg = tiny_full_attn_cfg();
    let mut store = TensorStore::default();
    let model = Qwen35Model::new(&cfg, &mut store)?;

    let param_map = model.param_name_map();
    assert!(param_map.contains_key(cfg.embed_tokens_tensor_name()));
    let layer_names = cfg.layer_tensor_names(0);
    let q_proj_name = match layer_names.attention {
        Qwen35AttentionTensorNames::Full(names) => names.q_proj,
        Qwen35AttentionTensorNames::Linear(_) => unreachable!("full-attn test"),
    };
    assert!(param_map.contains_key(q_proj_name.as_str()));

    let param_ids = model.all_parameter_ids();
    let unique_ids = param_ids.iter().copied().collect::<HashSet<_>>();
    assert_eq!(unique_ids.len(), param_ids.len());

    let frozen = model.clone_frozen(&mut store);
    for tensor_id in frozen.all_parameter_ids() {
        assert!(
            !store
                .get(tensor_id)
                .expect("cloned tensor exists")
                .requires_grad
        );
    }

    let single_ids = vec![1, 2, 3, 4];
    let batch_ids = [single_ids.clone(), single_ids.clone()].concat();

    let mut tape = Tape::new();
    let single = model
        .forward_tokens(&single_ids, &mut store, &mut tape)
        .expect("single forward");
    let single_logits = store.to_host(single)?;

    tape.entries.clear();
    let batched = model
        .forward_batch_tokens(&batch_ids, 2, single_ids.len(), &mut store, &mut tape)
        .expect("batch forward");
    let batched_logits = store.to_host(batched)?;

    assert_eq!(batched_logits.len(), 2 * single_logits.len());
    assert_eq!(&batched_logits[..single_logits.len()], &single_logits[..]);
    assert_eq!(&batched_logits[single_logits.len()..], &single_logits[..]);

    Ok(())
}
