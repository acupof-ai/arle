use autograd::{TensorStore, module::Module};
use train::checkpoint;
use train::model::{Lm, LmConfig};

fn tiny_config() -> LmConfig {
    LmConfig {
        vocab_size: 16,
        d_model: 16,
        n_layers: 2,
        n_heads: 2,
        d_head: 8,
        d_ff: 32,
        max_seq_len: 32,
        lora: None,
    }
}

fn perturb_params(store: &mut TensorStore, model: &Lm, scale: f32) {
    for (i, id) in model.parameters().iter().enumerate() {
        let tensor = store.get_mut(*id).expect("param");
        for (j, value) in tensor.data.iter_mut().enumerate() {
            *value = ((i * 31 + j) as f32) * scale;
        }
    }
}

fn snapshot(store: &TensorStore, model: &Lm) -> Vec<Vec<f32>> {
    model
        .parameters()
        .iter()
        .map(|id| store.get(*id).expect("param").data.clone())
        .collect()
}

#[test]
fn save_then_load_restores_exact_parameter_values() {
    let tmp = std::env::temp_dir().join(format!(
        "train_checkpoint_roundtrip_{}.bin",
        std::process::id()
    ));

    let config = tiny_config();
    let mut store = TensorStore::default();
    let model = Lm::new(config, &mut store).expect("model");

    // Put known values into the live model and snapshot.
    perturb_params(&mut store, &model, 0.125);
    let saved = snapshot(&store, &model);

    checkpoint::save(&model, &config, &store, &tmp).expect("save");

    // Perturb again — guarantees load must actually write.
    perturb_params(&mut store, &model, -9.5);
    let dirty = snapshot(&store, &model);
    assert_ne!(dirty, saved, "pre-condition: perturbation must diverge");

    checkpoint::load(&model, &mut store, &tmp).expect("load");
    let restored = snapshot(&store, &model);
    assert_eq!(
        restored, saved,
        "loaded params must match the original snapshot bit-for-bit"
    );

    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn config_is_readable_standalone() {
    let tmp = std::env::temp_dir().join(format!(
        "train_checkpoint_config_{}.bin",
        std::process::id()
    ));
    let config = tiny_config();
    let mut store = TensorStore::default();
    let model = Lm::new(config, &mut store).expect("model");
    checkpoint::save(&model, &config, &store, &tmp).expect("save");

    let read_back = checkpoint::read_config(&tmp).expect("read config");
    assert_eq!(read_back.vocab_size, config.vocab_size);
    assert_eq!(read_back.d_model, config.d_model);
    assert_eq!(read_back.n_layers, config.n_layers);
    assert_eq!(read_back.n_heads, config.n_heads);
    assert_eq!(read_back.d_head, config.d_head);
    assert_eq!(read_back.d_ff, config.d_ff);
    assert_eq!(read_back.max_seq_len, config.max_seq_len);
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn bad_magic_is_rejected() {
    let tmp = std::env::temp_dir().join(format!(
        "train_checkpoint_bad_magic_{}.bin",
        std::process::id()
    ));
    std::fs::write(&tmp, b"NOTMAGICxxxxxxxxxxxxxxxxxxxxxxxx").expect("write");

    let mut store = TensorStore::default();
    let model = Lm::new(tiny_config(), &mut store).expect("model");
    let err = checkpoint::load(&model, &mut store, &tmp).unwrap_err();
    assert!(
        matches!(err, checkpoint::CheckpointError::BadMagic { .. }),
        "got {err:?}",
    );

    let _ = std::fs::remove_file(&tmp);
}
