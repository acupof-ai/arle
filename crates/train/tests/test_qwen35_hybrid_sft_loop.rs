use autograd::{Tape, TensorStore, optim::AdamW};
use tempfile::tempdir;
use train::{
    causal_lm::{
        build_registry, live_tensor_ids, retained_ids, save_materialized_registry, trainable_params,
    },
    dataset::LcgRng,
    lora::LoraConfig,
    qwen35::Qwen35Model,
};

mod common;

use common::qwen35_test_support::{
    TEST_LR, TestResult, hybrid_qwen35_config, mean, tiny_sft_examples, train_on_example,
};

#[test]
fn qwen35_hybrid_lora_sft_loop_trains_and_roundtrips() -> TestResult {
    let cfg = hybrid_qwen35_config();
    let dataset = tiny_sft_examples();
    let lora = Some(LoraConfig {
        rank: 8,
        alpha: 16.0,
    });

    let mut store = TensorStore::default();
    let model = Qwen35Model::new_with_lora(&cfg, lora, &mut store)?;
    let params = trainable_params(&model, &store);
    let model_ids = live_tensor_ids(&store);
    let mut optimizer = AdamW::new(TEST_LR, (0.9, 0.999), 1.0e-8, 0.0);
    let mut tape = Tape::new();
    let mut rng = LcgRng::seed(0x4859_4252_4944_4C4F);
    let mut losses = Vec::with_capacity(20);

    for step in 0..20 {
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
    let last_avg = mean(&losses[16..20]);
    assert!(
        last_avg < first_avg,
        "hybrid qwen35+lora: trailing loss {last_avg} >= leading {first_avg}"
    );

    let dir = tempdir()?;
    let path = dir.path().join("model.safetensors");
    save_materialized_registry(&model, &mut store, &mut tape, &path, true)?;

    let mut loaded_store = TensorStore::default();
    let loaded_model = Qwen35Model::new_for_eval(&cfg, &mut loaded_store)?;
    let mut loaded_registry = build_registry(&loaded_model);
    loaded_registry.load_into(&mut loaded_store, &path)?;

    let input_ids = [11, 12, 13, 14];
    let position_ids = [0, 1, 2, 3];
    let source_logits = model.forward(&mut store, &mut tape, &input_ids, &position_ids)?;
    let source = store.to_host(source_logits)?;
    tape.entries.clear();
    tape.set_enabled(true);

    let mut loaded_tape = Tape::new();
    let loaded_logits = loaded_model.forward(
        &mut loaded_store,
        &mut loaded_tape,
        &input_ids,
        &position_ids,
    )?;
    let loaded = loaded_store.to_host(loaded_logits)?;

    assert_eq!(source.len(), loaded.len());
    for (idx, (&left, &right)) in source.iter().zip(loaded.iter()).enumerate() {
        let abs_err = (left - right).abs();
        let rel_err = abs_err / left.abs().max(1.0e-5);
        assert!(
            abs_err <= 2.5e-3 || rel_err <= 6.0e-2,
            "hybrid merged reload drift at index {idx}: left={left} right={right} abs={abs_err} rel={rel_err}"
        );
    }

    Ok(())
}
