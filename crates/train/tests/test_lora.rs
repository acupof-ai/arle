use std::collections::{HashMap, HashSet};

use autograd::{Tape, TensorId, TensorStore, module::Module, optim::AdamW};
use train::{
    dataset::{CopyDataset, Dataset},
    lora::LoraConfig,
    model::{Transformer, TransformerConfig},
    trainer::{clip_grad_norm, cross_entropy_loss},
};

#[test]
fn lora_trains_with_frozen_base() {
    let config = TransformerConfig {
        vocab_size: 16,
        d_model: 16,
        n_layers: 2,
        n_heads: 2,
        d_head: 8,
        d_ff: 32,
        max_seq_len: 8,
        lora: Some(LoraConfig {
            rank: 4,
            alpha: 8.0,
        }),
    };

    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let model = Transformer::new(config, &mut store).expect("build tiny model with lora");
    let params = model.parameters();
    let base_params = model.base_parameter_ids();
    let base_before = snapshot(&base_params, &mut store);
    let trainable_before = snapshot(&params, &mut store);
    let mut optimizer = AdamW::new(1.0e-2, (0.9, 0.999), 1.0e-8, 0.0);
    let mut dataset = CopyDataset::with_vocab(1, 4, 7, 15, 15);
    let mut losses = Vec::with_capacity(5);

    for _ in 0..5 {
        let (inputs, targets) = dataset.sample();
        let (batch, seq_len) = dataset.batch_shape();

        tape.entries.clear();
        tape.set_enabled(true);

        let logits = model
            .forward(&inputs, batch, seq_len, &mut store, &mut tape)
            .expect("forward");
        let loss = cross_entropy_loss(logits, &targets, &mut store, &mut tape).expect("loss");
        losses.push(store.to_host(loss).expect("loss value")[0]);

        optimizer.zero_grad(&params, &mut store);
        tape.backward(loss, &mut store).expect("backward");
        clip_grad_norm(&params, 1.0, &mut store);
        optimizer.step(&params, &mut store);

        tape.entries.clear();
        tape.set_enabled(true);
        let keep = retained_ids(&params, &base_params, &store);
        store.retain_ids(&keep);
    }

    for &param_id in &base_params {
        assert_eq!(
            store
                .to_host(param_id)
                .expect("base parameter remains readable"),
            base_before[&param_id],
            "base parameter {param_id} changed despite being frozen"
        );
    }

    for &param_id in &params {
        assert_ne!(
            store
                .to_host(param_id)
                .expect("trainable parameter remains readable"),
            trainable_before[&param_id],
            "trainable parameter {param_id} did not change"
        );
    }

    assert!(
        losses[4] < losses[0],
        "expected loss to decrease over 5 steps, got {losses:?}"
    );
}

fn snapshot(params: &[TensorId], store: &mut TensorStore) -> HashMap<TensorId, Vec<f32>> {
    params
        .iter()
        .map(|&id| {
            (
                id,
                store
                    .to_host(id)
                    .expect("snapshot tensor should be readable"),
            )
        })
        .collect()
}

fn retained_ids(
    params: &[TensorId],
    base_params: &[TensorId],
    store: &TensorStore,
) -> HashSet<TensorId> {
    let mut keep = HashSet::with_capacity((params.len() + base_params.len()) * 2);
    for &param_id in params.iter().chain(base_params.iter()) {
        keep.insert(param_id);
        if let Some(grad_id) = store.get(param_id).and_then(|tensor| tensor.grad) {
            keep.insert(grad_id);
        }
    }
    keep
}
