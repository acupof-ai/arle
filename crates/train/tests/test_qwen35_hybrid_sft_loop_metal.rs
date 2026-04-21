#![cfg(feature = "metal")]

use std::sync::Arc;

use autograd::{Backend, Tape, TensorStore, backend_metal::MetalBackend, optim::AdamW};
use train::{
    causal_lm::{live_tensor_ids, retained_ids, trainable_params},
    dataset::LcgRng,
    lora::LoraConfig,
    qwen35::Qwen35Model,
};

mod common;

use common::qwen35_test_support::{
    TEST_LR, TestResult, hybrid_qwen35_config, mean, tiny_sft_examples, train_on_example,
};

#[test]
fn qwen35_hybrid_lora_sft_loop_metal_trains() -> TestResult {
    let cfg = hybrid_qwen35_config();
    let dataset = tiny_sft_examples();
    let lora = Some(LoraConfig {
        rank: 8,
        alpha: 16.0,
    });

    let backend: Arc<dyn Backend> = Arc::new(MetalBackend);
    let mut store = TensorStore::with_backend(backend);
    let model = Qwen35Model::new_with_lora(&cfg, lora, &mut store)?;
    let params = trainable_params(&model, &store);
    let model_ids = live_tensor_ids(&store);
    let mut optimizer = AdamW::new(TEST_LR, (0.9, 0.999), 1.0e-8, 0.0);
    let mut tape = Tape::new();
    let mut rng = LcgRng::seed(0x4859_4252_4944_4D45);
    let mut losses = Vec::with_capacity(16);

    for step in 0..16 {
        optimizer.zero_grad(&params, &mut store);
        let example_index = (rng.next_u64() as usize + step) % dataset.len();
        let loss = train_on_example(&model, &dataset[example_index], &mut store, &mut tape)?;
        losses.push(loss);
        optimizer.step(&params, &mut store);

        tape.entries.clear();
        tape.set_enabled(true);
        let keep = retained_ids(&model_ids, &params, &store);
        store.retain_ids(&keep);
    }

    let first_avg = mean(&losses[..4]);
    let last_avg = mean(&losses[12..16]);
    assert!(
        last_avg < first_avg,
        "metal hybrid qwen35+lora: trailing loss {last_avg} >= leading {first_avg}"
    );

    Ok(())
}
