use std::collections::HashSet;

use autograd::{Tape, TensorStore};
use qwen35_spec::Qwen35AttentionTensorNames;
use train::qwen35::Qwen35Model;

type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

mod common;

use common::qwen35_test_support::{
    tiny_hybrid_qwen35_scratch_config_with_vocab, tiny_qwen35_scratch_config_with_vocab,
};

#[test]
fn qwen35_batch_forward_matches_repeated_single_forward() -> TestResult {
    let cfg = tiny_qwen35_scratch_config_with_vocab(16, 64);
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

#[test]
fn qwen35_hybrid_forward_supports_partial_rope_and_linear_layers() -> TestResult {
    let cfg = tiny_hybrid_qwen35_scratch_config_with_vocab(16, 64);
    cfg.validate_train_lora_or_frozen_contract()?;
    let mut store = TensorStore::default();
    let model = Qwen35Model::new_for_eval(&cfg, &mut store)?;

    let param_map = model.param_name_map();
    let layer_names = cfg.layer_tensor_names(1);
    let linear_names = match layer_names.attention {
        Qwen35AttentionTensorNames::Linear(names) => names,
        Qwen35AttentionTensorNames::Full(_) => unreachable!("hybrid test expects linear layer"),
    };
    assert!(param_map.contains_key(linear_names.in_proj_qkv.as_str()));
    assert!(param_map.contains_key(linear_names.out_proj.as_str()));
    assert!(param_map.contains_key(linear_names.conv1d_weight.as_str()));

    let mut tape = Tape::new();
    let logits = model.forward_tokens(&[1, 2, 3, 4], &mut store, &mut tape)?;
    let values = store.to_host(logits)?;
    assert_eq!(values.len(), 4 * cfg.vocab_size);
    assert!(values.iter().all(|value| value.is_finite()));

    Ok(())
}
