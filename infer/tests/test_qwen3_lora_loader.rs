//! Integration test for the PEFT LoRA loader (`model::qwen3::lora`).
//!
//! Writes a synthetic PEFT adapter directory (adapter_config.json + f32
//! adapter_model.safetensors) to a tempdir, runs `load_peft_lora`, and
//! asserts the bundle has the right slots populated with the right
//! shapes. Exercises the f32→bf16 upload path, the scale pre-bake on B,
//! and the PEFT key parser's tolerance for the `base_model.model.model.`
//! double-prefix.

#![cfg(feature = "cuda")]

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fs;

use infer::backend::cuda::tensor::DeviceContext;
use infer::model::qwen3::lora::load_peft_lora;
use safetensors::tensor::{Dtype, View};

struct F32Tensor {
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl F32Tensor {
    fn new(shape: Vec<usize>, values: Vec<f32>) -> Self {
        let n: usize = shape.iter().product();
        assert_eq!(n, values.len());
        let mut data = Vec::with_capacity(n * 4);
        for v in &values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        Self { shape, data }
    }
}

impl View for &F32Tensor {
    fn dtype(&self) -> Dtype {
        Dtype::F32
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.data)
    }
    fn data_len(&self) -> usize {
        self.data.len()
    }
}

#[test]
fn loads_synthetic_peft_adapter() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path();

    // r=8, alpha=16 → scale = 2.0
    let r: usize = 8;
    let alpha: f32 = 16.0;
    let expected_scale = alpha / r as f32;

    // Target q_proj on layer 0, gate_proj on layer 1. This exercises both
    // the attention branch (self_attn.<m>) and the MLP branch (mlp.<m>)
    // in `parse_peft_key`, and leaves several slots empty so the
    // Option<LoRAAdapter>-per-slot invariant is visible.
    let q_in = 64usize;
    let q_out = 64usize;
    let gate_in = 64usize;
    let gate_out = 128usize;

    let q_a = F32Tensor::new(vec![r, q_in], vec![0.125f32; r * q_in]);
    let q_b = F32Tensor::new(vec![q_out, r], vec![0.25f32; q_out * r]);
    let gate_a = F32Tensor::new(vec![r, gate_in], vec![-0.5f32; r * gate_in]);
    let gate_b = F32Tensor::new(vec![gate_out, r], vec![0.75f32; gate_out * r]);

    // BTreeMap so iteration order is stable for the safetensors header.
    // Keys use the `base_model.model.model.` double-prefix convention
    // that matches what a PEFT export of a wrapped causal-LM produces.
    let mut tensors: BTreeMap<String, &F32Tensor> = BTreeMap::new();
    tensors.insert(
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
        &q_a,
    );
    tensors.insert(
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
        &q_b,
    );
    tensors.insert(
        "base_model.model.model.layers.1.mlp.gate_proj.lora_A.weight".to_string(),
        &gate_a,
    );
    tensors.insert(
        "base_model.model.model.layers.1.mlp.gate_proj.lora_B.weight".to_string(),
        &gate_b,
    );

    // Extra key the loader should skip without erroring (covers the
    // lm_head / embed delta case that some peft configs export).
    let embed = F32Tensor::new(vec![4, 4], vec![0.0f32; 16]);
    tensors.insert("base_model.model.lm_head.lora_A.weight".to_string(), &embed);

    safetensors::serialize_to_file(tensors, None, &path.join("adapter_model.safetensors"))
        .expect("safetensors write");

    let cfg = serde_json::json!({
        "r": r,
        "lora_alpha": alpha,
        "target_modules": ["q_proj", "gate_proj"],
    });
    fs::write(
        path.join("adapter_config.json"),
        serde_json::to_string_pretty(&cfg).unwrap(),
    )
    .expect("config write");

    let ctx = DeviceContext::new().expect("DeviceContext::new");
    let num_layers = 4;
    let bundle = load_peft_lora(&ctx, path.to_str().unwrap(), num_layers)
        .expect("load_peft_lora should succeed");

    assert_eq!(bundle.layers.len(), num_layers);

    // Layer 0: q_proj only.
    let l0 = &bundle.layers[0];
    let qp = l0.q_proj.as_ref().expect("layer 0 q_proj should be loaded");
    assert_eq!(qp.a.rows, r);
    assert_eq!(qp.a.cols, q_in);
    assert_eq!(qp.b.rows, q_out);
    assert_eq!(qp.b.cols, r);
    assert!(
        (qp.scale - expected_scale).abs() < 1e-6,
        "scale mismatch: got {}, want {}",
        qp.scale,
        expected_scale,
    );
    assert!(l0.k_proj.is_none(), "k_proj should be absent");
    assert!(l0.v_proj.is_none(), "v_proj should be absent");
    assert!(l0.gate_proj.is_none(), "layer 0 gate_proj should be absent");

    // Layer 1: gate_proj only.
    let l1 = &bundle.layers[1];
    assert!(l1.q_proj.is_none(), "layer 1 q_proj should be absent");
    let gp = l1
        .gate_proj
        .as_ref()
        .expect("layer 1 gate_proj should be loaded");
    assert_eq!(gp.a.rows, r);
    assert_eq!(gp.a.cols, gate_in);
    assert_eq!(gp.b.rows, gate_out);
    assert_eq!(gp.b.cols, r);
    assert!((gp.scale - expected_scale).abs() < 1e-6);

    // Layers 2, 3: fully empty.
    for layer_idx in 2..num_layers {
        let l = &bundle.layers[layer_idx];
        assert!(l.q_proj.is_none());
        assert!(l.k_proj.is_none());
        assert!(l.v_proj.is_none());
        assert!(l.o_proj.is_none());
        assert!(l.gate_proj.is_none());
        assert!(l.up_proj.is_none());
        assert!(l.down_proj.is_none());
    }
}

#[test]
fn rejects_missing_directory() {
    let ctx = DeviceContext::new().expect("DeviceContext::new");
    let err = match load_peft_lora(&ctx, "/nonexistent/lora/path/definitely-not-there", 4) {
        Err(e) => e,
        Ok(_) => panic!("missing directory must be rejected"),
    };
    let msg = format!("{err:#}");
    assert!(
        msg.contains("not a directory"),
        "expected 'not a directory' in error, got: {msg}",
    );
}

#[test]
fn rejects_invalid_r_zero() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path();

    // Empty tensors file is fine — the loader should fail at config
    // validation (r=0) before touching the safetensors.
    let empty: BTreeMap<String, &F32Tensor> = BTreeMap::new();
    safetensors::serialize_to_file(empty, None, &path.join("adapter_model.safetensors"))
        .expect("safetensors write");

    let cfg = serde_json::json!({
        "r": 0,
        "lora_alpha": 16.0,
        "target_modules": ["q_proj"],
    });
    fs::write(
        path.join("adapter_config.json"),
        serde_json::to_string_pretty(&cfg).unwrap(),
    )
    .expect("config write");

    let ctx = DeviceContext::new().expect("DeviceContext::new");
    let err = match load_peft_lora(&ctx, path.to_str().unwrap(), 4) {
        Err(e) => e,
        Ok(_) => panic!("r=0 must be rejected"),
    };
    assert!(format!("{err:#}").contains("r=0"));
}
