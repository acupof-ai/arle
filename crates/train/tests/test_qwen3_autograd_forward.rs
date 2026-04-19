use autograd::{
    Tape, TensorStore,
    ops::sum,
};
use train::qwen3_autograd::{Qwen3Config, Qwen3Model};

type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

#[test]
fn qwen3_autograd_forward_smoke_tiny_config() -> TestResult {
    let cfg = Qwen3Config {
        vocab_size: 200,
        hidden_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_kv_heads: 2,
        head_dim: 16,
        intermediate_size: 128,
        max_position_embeddings: 32,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        tie_word_embeddings: false,
    };

    let mut store = TensorStore::default();
    let model = Qwen3Model::new(&cfg, &mut store)?;
    let mut tape = Tape::new();

    let input_ids = [1_u32, 2, 3, 4];
    let position_ids = [0_u32, 1, 2, 3];
    let logits = model.forward(&mut store, &mut tape, &input_ids, &position_ids)?;

    let logits_shape = store.get(logits).expect("logits tensor exists").shape.clone();
    assert_eq!(logits_shape, vec![1, 4, cfg.vocab_size]);

    let logits_host = store.to_host(logits)?;
    assert!(logits_host.iter().all(|value| value.is_finite()));

    let loss = sum(logits, &mut store, &mut tape)?;
    let grads = tape.backward(loss, &mut store)?;

    let param_map = model.param_name_map();
    let lm_head = *param_map
        .get("lm_head.weight")
        .expect("lm_head weight registered");
    let grad_id = *grads.get(&lm_head).expect("grad for lm_head");
    let grad = store.to_host(grad_id)?;
    assert!(!grad.is_empty());
    assert!(grad.iter().all(|value| value.is_finite()));

    Ok(())
}
