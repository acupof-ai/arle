use std::collections::HashSet;

use autograd::{Tape, TensorId, TensorStore, module::Module, optim::AdamW};
use train::{
    dataset::{CopyDataset, Dataset},
    model::{TinyLM, TinyLMConfig},
    trainer::{clip_grad_norm, cross_entropy_loss},
};

#[test]
fn tiny_lm_copy_loss_drops_over_three_steps() {
    let config = TinyLMConfig {
        vocab_size: 16,
        d_model: 16,
        n_layers: 2,
        n_heads: 2,
        d_head: 8,
        d_ff: 32,
        max_seq_len: 4,
        lora: None,
    };

    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let model = TinyLM::new(config, &mut store).expect("build tiny model");
    let params = model.parameters();
    let mut optimizer = AdamW::new(1.0e-2, (0.9, 0.999), 1.0e-8, 0.0);
    let mut losses = Vec::with_capacity(3);

    for _ in 0..3 {
        let mut dataset = CopyDataset::with_vocab(1, 4, 7, 15, 15);
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
        let keep = retained_ids(&params, &store);
        store.retain_ids(&keep);
    }

    assert!(
        losses[2] < losses[0],
        "expected loss to decrease, got {losses:?}"
    );
}

fn retained_ids(params: &[TensorId], store: &TensorStore) -> HashSet<TensorId> {
    let mut keep = HashSet::with_capacity(params.len() * 2);
    for &param_id in params {
        keep.insert(param_id);
        if let Some(grad_id) = store.get(param_id).and_then(|tensor| tensor.grad) {
            keep.insert(grad_id);
        }
    }
    keep
}
