use std::{collections::HashSet, error::Error};

use autograd::{
    Tape, TensorId, TensorStore,
    ops::{gather_last_dim, log_softmax, mul, mul_scalar, sum},
    optim::AdamW,
};
use tempfile::tempdir;
use train::{
    dataset::LcgRng,
    qwen3::{Qwen3Config, Qwen3Model},
    qwen3_support::{build_registry, live_tensor_ids, trainable_params},
    sft_data::TokenizedSft,
};

type TestResult<T = ()> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

const LR: f32 = 5.0e-3;

#[test]
fn sft_loop_smoke_trains_and_roundtrips() -> TestResult {
    let cfg = Qwen3Config {
        vocab_size: 200,
        hidden_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 16,
        intermediate_size: 128,
        max_position_embeddings: 32,
        rms_norm_eps: 1.0e-6,
        rope_theta: 10_000.0,
        tie_word_embeddings: false,
    };
    let dataset = tiny_sft_examples();

    let mut store = TensorStore::default();
    let model = Qwen3Model::new(&cfg, &mut store)?;
    let params = trainable_params(&model, &store);
    let model_ids = live_tensor_ids(&store);
    let mut optimizer = AdamW::new(LR, (0.9, 0.999), 1.0e-8, 0.0);
    let mut tape = Tape::new();
    let mut rng = LcgRng::seed(0x5F54_5F53_4D4F_4B45);
    let mut losses = Vec::with_capacity(30);

    for step in 0..30 {
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

    let first_avg = mean(&losses[..5]);
    let last_avg = mean(&losses[25..30]);
    println!(
        "loss[0..5].mean() = {first_avg:.6}, loss[25..30].mean() = {last_avg:.6}, delta = {:.6}",
        last_avg - first_avg
    );
    assert!(
        last_avg < first_avg,
        "expected trailing loss average {last_avg} to be below leading average {first_avg}"
    );

    let registry = build_registry(&model);
    let dir = tempdir()?;
    let path = dir.path().join("model.safetensors");
    registry.save_from(&mut store, &path)?;

    let mut loaded_store = TensorStore::default();
    let loaded_model = Qwen3Model::new(&cfg, &mut loaded_store)?;
    let mut loaded_registry = build_registry(&loaded_model);
    loaded_registry.load_into(&mut loaded_store, &path)?;

    let param_map = model.param_name_map();
    let loaded_param_map = loaded_model.param_name_map();
    let embed_tokens_name = cfg.embed_tokens_tensor_name();
    let source_id = *param_map
        .get(embed_tokens_name)
        .expect("embed tokens weight present");
    let loaded_id = *loaded_param_map
        .get(embed_tokens_name)
        .expect("loaded embed tokens weight present");
    let source = store.to_host(source_id)?;
    let loaded = loaded_store.to_host(loaded_id)?;
    assert_eq!(source.len(), loaded.len());
    for (left, right) in source.iter().zip(loaded.iter()) {
        assert_eq!(left.to_bits(), right.to_bits());
    }

    Ok(())
}

fn tiny_sft_examples() -> Vec<TokenizedSft> {
    vec![
        TokenizedSft {
            input_ids: vec![11, 12, 13, 14, 21, 22, 23, 24],
            labels: vec![-100, -100, -100, -100, 21, 22, 23, 24],
        },
        TokenizedSft {
            input_ids: vec![31, 32, 33, 34, 41, 42, 43, 44],
            labels: vec![-100, -100, -100, -100, 41, 42, 43, 44],
        },
        TokenizedSft {
            input_ids: vec![51, 52, 53, 54, 61, 62, 63, 64],
            labels: vec![-100, -100, -100, -100, 61, 62, 63, 64],
        },
    ]
}

fn train_on_example(
    model: &Qwen3Model,
    example: &TokenizedSft,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> TestResult<f32> {
    let input_len = example.input_ids.len() - 1;
    let position_ids = (0..input_len).map(|index| index as u32).collect::<Vec<_>>();
    let logits = model.forward(store, tape, &example.input_ids[..input_len], &position_ids)?;
    let loss_id = assistant_masked_loss(logits, &example.labels[1..], store, tape)?;
    let loss = store.to_host(loss_id)?[0];
    tape.backward(loss_id, store)?;
    Ok(loss)
}

fn assistant_masked_loss(
    logits: TensorId,
    labels: &[i32],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> TestResult<TensorId> {
    let logits_shape = store
        .get(logits)
        .expect("logits tensor exists")
        .shape
        .clone();
    let prefix_elems = logits_shape[..logits_shape.len() - 1]
        .iter()
        .product::<usize>();
    assert_eq!(labels.len(), prefix_elems);

    let gather_indices = labels
        .iter()
        .map(|&label| usize::try_from(label).unwrap_or(0))
        .collect::<Vec<_>>();
    let mask_values = labels
        .iter()
        .map(|&label| if label >= 0 { 1.0 } else { 0.0 })
        .collect::<Vec<_>>();
    let valid_count = mask_values.iter().filter(|&&value| value > 0.0).count();
    assert!(valid_count > 0);

    let log_probs = log_softmax(logits, store, tape)?;
    let gathered = gather_last_dim(log_probs, &gather_indices, store, tape)?;
    let gathered_shape = store
        .get(gathered)
        .expect("gathered tensor exists")
        .shape
        .clone();
    let mask = store.from_slice(&mask_values, &gathered_shape)?;
    let masked = mul(gathered, mask, store, tape)?;
    let total = sum(masked, store, tape)?;
    Ok(mul_scalar(total, -1.0 / valid_count as f32, store, tape)?)
}

fn retained_ids(
    model_ids: &HashSet<TensorId>,
    params: &[TensorId],
    store: &TensorStore,
) -> HashSet<TensorId> {
    let mut keep = model_ids.clone();
    for &param_id in params {
        keep.insert(param_id);
        if let Some(grad_id) = store.get(param_id).and_then(|tensor| tensor.grad) {
            keep.insert(grad_id);
        }
    }
    keep
}

fn mean(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len() as f32
}
