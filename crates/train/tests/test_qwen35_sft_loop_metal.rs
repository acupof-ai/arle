//! End-to-end verification that the dense/full-attn Qwen3.5 LoRA SFT path
//! works through the Metal backend, plus a merged bf16 save -> reload
//! roundtrip that mirrors what `train_sft --backend metal --model-family qwen35`
//! emits today.
#![cfg(feature = "metal")]

use std::{error::Error, sync::Arc};

use autograd::{
    Backend, Tape, TensorId, TensorStore,
    backend_metal::MetalBackend,
    ops::{gather_last_dim, log_softmax, mul, mul_scalar, sum},
    optim::AdamW,
};
use tempfile::tempdir;
use train::{
    causal_lm::{
        build_registry, live_tensor_ids, retained_ids, save_materialized_registry, trainable_params,
    },
    dataset::LcgRng,
    lora::LoraConfig,
    qwen35::{LayerType, Qwen35Config, Qwen35Model},
    sft_data::TokenizedSft,
};

type TestResult<T = ()> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

const LR: f32 = 5.0e-3;

#[test]
fn qwen35_lora_sft_loop_metal_trains_and_bf16_roundtrips() -> TestResult {
    let cfg = tiny_qwen35_config();
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
    let mut optimizer = AdamW::new(LR, (0.9, 0.999), 1.0e-8, 0.0);
    let mut tape = Tape::new();
    let mut rng = LcgRng::seed(0x5147_3335_4C4F_5241);
    let mut losses = Vec::with_capacity(24);

    for step in 0..24 {
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
    let last_avg = mean(&losses[20..24]);
    println!(
        "[metal][qwen35+lora] loss[0..4]={first_avg:.6} loss[20..24]={last_avg:.6} delta={:.6}",
        last_avg - first_avg
    );
    assert!(
        last_avg < first_avg,
        "metal qwen35+lora: trailing loss {last_avg} >= leading {first_avg}"
    );

    let dir = tempdir()?;
    let path = dir.path().join("model.safetensors");
    save_materialized_registry(&model, &mut store, &mut tape, &path, true)?;

    let mut loaded_store = TensorStore::default();
    let loaded_model = Qwen35Model::new(&cfg, &mut loaded_store)?;
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
            abs_err <= 2.0e-3 || rel_err <= 6.0e-2,
            "bf16 merged reload drift at index {idx}: left={left} right={right} abs={abs_err} rel={rel_err}"
        );
    }

    Ok(())
}

fn tiny_qwen35_config() -> Qwen35Config {
    Qwen35Config {
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 2,
        vocab_size: 256,
        rms_norm_eps: 1.0e-6,
        stop_token_ids: vec![2],
        bos_token_id: Some(1),
        eos_token_id: 2,
        tie_word_embeddings: false,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 16,
        linear_num_key_heads: 4,
        linear_key_head_dim: 16,
        linear_num_value_heads: 4,
        linear_value_head_dim: 16,
        linear_conv_kernel_dim: 4,
        rope_theta: 10_000.0,
        partial_rotary_factor: 1.0,
        rotary_dim: 16,
        rope_cache_len_hint: Some(32),
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
    model: &Qwen35Model,
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
    let valid_count = mask_values.iter().filter(|&&v| v > 0.0).count();
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

fn mean(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len() as f32
}
