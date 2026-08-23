//! A drafter's RoPE must come from its own config, both base and scaling.
//!
//! `transformers >= 5.12` moved both under `rope_parameters`. Reading only the
//! top level defaults the base and drops the scaling silently, and the symptom
//! is not a load error — it is a drafter whose positions disagree with the
//! target's from layer one. Measured on `RadixArk/Qwen3.8-27B-DSpark`: 13%
//! acceptance at c=1 falling to 0% at c=8, which reads as a weak drafter.

use qwen35_spec::{DsparkConfig, RopeScalingConfig};

fn write_config(dir: &std::path::Path, body: &str) {
    std::fs::create_dir_all(dir).expect("mkdir");
    std::fs::write(dir.join("config.json"), body).expect("write config");
}

/// Everything `DsparkConfig::from_dir` requires, minus the RoPE block.
fn base_fields() -> String {
    r#""hidden_size": 5120, "intermediate_size": 10240, "num_hidden_layers": 5,
       "num_attention_heads": 40, "num_key_value_heads": 8, "head_dim": 128,
       "block_size": 7,
       "dflash_config": {"mask_token_id": 248077, "target_layer_ids": [4, 16, 28, 40, 52]}"#
        .to_owned()
}

#[test]
fn nested_rope_parameters_carry_base_and_yarn() {
    let tmp = tempfile::tempdir().expect("tempdir");
    write_config(
        tmp.path(),
        &format!(
            r#"{{{}, "rope_parameters": {{"rope_theta": 10000000, "rope_type": "yarn",
               "factor": 32.0, "original_max_position_embeddings": 8192,
               "beta_fast": 32.0, "beta_slow": 1.0}}}}"#,
            base_fields()
        ),
    );
    let cfg = DsparkConfig::from_dir(tmp.path()).expect("parse");
    assert_eq!(cfg.rope_theta, 1e7);
    match cfg.rope_scaling {
        Some(RopeScalingConfig::Yarn {
            factor,
            original_max_position_embeddings,
            ..
        }) => {
            assert_eq!(factor, 32.0);
            assert_eq!(original_max_position_embeddings, 8192);
        }
        other => panic!("yarn dropped: {other:?}"),
    }
}

#[test]
fn top_level_rope_theta_still_wins_and_scales_none() {
    let tmp = tempfile::tempdir().expect("tempdir");
    write_config(
        tmp.path(),
        &format!(r#"{{{}, "rope_theta": 1000000}}"#, base_fields()),
    );
    let cfg = DsparkConfig::from_dir(tmp.path()).expect("parse");
    assert_eq!(cfg.rope_theta, 1e6);
    assert!(
        cfg.rope_scaling.is_none(),
        "vanilla config gained a scaling"
    );
}

#[test]
fn unsupported_rope_type_is_loud() {
    let tmp = tempfile::tempdir().expect("tempdir");
    write_config(
        tmp.path(),
        &format!(
            r#"{{{}, "rope_parameters": {{"rope_theta": 10000000, "rope_type": "llama3"}}}}"#,
            base_fields()
        ),
    );
    // Silently ignoring it would make every draft wrong while looking healthy.
    assert!(DsparkConfig::from_dir(tmp.path()).is_err());
}
