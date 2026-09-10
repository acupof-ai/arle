//! End-to-end detection of resident weight-quant formats from safetensors
//! headers. The negative tests pin the shape/divisibility constraints: a
//! constraint that stops being enforced turns its test red.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use infer_quant::{
    QuantFormat, QuantManifest, TensorHeader, decode_f8_e4m3fn, detect_quant_format,
    read_quant_manifest, reject_dsv4_e8m0_scale_abi,
};
use safetensors::tensor::Dtype;

fn header(dtype: Dtype, shape: &[usize]) -> TensorHeader {
    TensorHeader {
        dtype,
        shape: shape.to_vec(),
    }
}

fn tensors(pairs: &[(&str, Dtype, &[usize])]) -> BTreeMap<String, TensorHeader> {
    pairs
        .iter()
        .map(|(name, dtype, shape)| (name.to_string(), header(*dtype, shape)))
        .collect()
}

fn detect(
    name: &str,
    t: &BTreeMap<String, TensorHeader>,
    manifest: Option<&QuantManifest>,
) -> anyhow::Result<Option<infer_quant::QuantTensorView>> {
    detect_quant_format(name, t, manifest)
}

#[test]
fn dense_bf16_without_manifest() {
    let t = tensors(&[("layer.weight", Dtype::BF16, &[64, 128])]);
    let view = detect("layer.weight", &t, None).unwrap().unwrap();
    assert_eq!(view.format, QuantFormat::DenseBf16);
    assert_eq!(view.logical_shape, [64, 128]);
    assert!(view.scale_names.is_empty());
}

#[test]
fn dense_bf16_respects_manifest_ignore() {
    let t = tensors(&[("layer.weight", Dtype::BF16, &[64, 128])]);
    let ignoring = QuantManifest {
        modules_to_not_convert: vec!["layer".to_string()],
        ..Default::default()
    };
    assert!(
        detect("layer.weight", &t, Some(&ignoring))
            .unwrap()
            .is_some()
    );
    let not_ignoring = QuantManifest {
        quant_method: Some("gptq".to_string()),
        ..Default::default()
    };
    // A manifest that does not ignore the tensor means it is not dense:
    // detection falls through rather than mislabeling it.
    assert!(
        detect("layer.weight", &t, Some(&not_ignoring))
            .unwrap()
            .is_none()
    );
}

#[test]
fn gptq_w4a16() {
    let t = tensors(&[
        ("layer.qweight", Dtype::I32, &[3, 8]),
        ("layer.scales", Dtype::BF16, &[3, 8]),
        ("layer.qzeros", Dtype::I32, &[3, 1]),
    ]);
    let view = detect("layer.qweight", &t, None).unwrap().unwrap();
    // k = packed_k * 8 = 24, num_groups = 3, group_size = 8.
    assert_eq!(view.format, QuantFormat::GptqW4A16 { group_size: 8 });
    assert_eq!(view.logical_shape, [8, 24]);
    assert_eq!(view.scale_names, ["layer.scales", "layer.qzeros"]);
}

#[test]
fn gptq_k_not_divisible_by_groups_is_rejected() {
    // k = 24, num_groups = 5: 24 % 5 != 0.
    let t = tensors(&[
        ("layer.qweight", Dtype::I32, &[3, 8]),
        ("layer.scales", Dtype::BF16, &[5, 8]),
        ("layer.qzeros", Dtype::I32, &[5, 1]),
    ]);
    assert!(detect("layer.qweight", &t, None).is_err());
}

#[test]
fn fp8_block_scaled_inv() {
    let t = tensors(&[
        ("layer.weight", Dtype::F8_E4M3, &[128, 128]),
        ("layer.weight_scale_inv", Dtype::BF16, &[1, 1]),
    ]);
    let view = detect("layer.weight", &t, None).unwrap().unwrap();
    assert_eq!(
        view.format,
        QuantFormat::Fp8BlockScaled {
            block_m: 128,
            block_k: 128,
            scale_apply: infer_quant::ScaleApply::Multiply,
        }
    );
}

#[test]
fn fp8_block_scale_shape_is_validated() {
    // 128x128 weight needs a [1, 1] scale; a [2, 2] scale is rejected.
    let t = tensors(&[
        ("layer.weight", Dtype::F8_E4M3, &[128, 128]),
        ("layer.weight_scale_inv", Dtype::BF16, &[2, 2]),
    ]);
    assert!(detect("layer.weight", &t, None).is_err());
}

#[test]
fn fp8_per_shard() {
    let t = tensors(&[
        ("layer.weight", Dtype::F8_E4M3, &[64, 128]),
        ("layer.weight_scale", Dtype::F32, &[]),
        ("layer.input_scale", Dtype::F32, &[]),
    ]);
    let view = detect("layer.weight", &t, None).unwrap().unwrap();
    assert_eq!(
        view.format,
        QuantFormat::Fp8PerShard {
            scale_apply: infer_quant::ScaleApply::Multiply,
        }
    );
}

#[test]
fn fp8_per_channel_weight_scale() {
    let t = tensors(&[
        ("layer.weight", Dtype::F8_E4M3, &[64, 128]),
        ("layer.weight_scale", Dtype::BF16, &[64, 1]),
    ]);
    let view = detect("layer.weight", &t, None).unwrap().unwrap();
    assert_eq!(
        view.format,
        QuantFormat::Fp8BlockScaled {
            block_m: 1,
            block_k: 128,
            scale_apply: infer_quant::ScaleApply::Multiply,
        }
    );
}

#[test]
fn fp4_group_global_scales() {
    // U8 [n, k//2] -> logical [n, k]; group scale [n, k/16].
    let t = tensors(&[
        ("layer.weight", Dtype::U8, &[64, 32]),
        ("layer.weight_scale", Dtype::F8_E4M3, &[64, 4]),
        ("layer.weight_global_scale", Dtype::F32, &[]),
        ("layer.input_global_scale", Dtype::F32, &[]),
    ]);
    let view = detect("layer.weight", &t, None).unwrap().unwrap();
    assert_eq!(
        view.format,
        QuantFormat::Fp4E2M1Group {
            group_size: 16,
            global_scale_apply: infer_quant::ScaleApply::Divide,
        }
    );
    assert_eq!(view.logical_shape, [64, 64]);
}

#[test]
fn fp4_group_scale_2_variant() {
    let t = tensors(&[
        ("layer.weight", Dtype::U8, &[64, 32]),
        ("layer.weight_scale", Dtype::F8_E4M3, &[64, 4]),
        ("layer.weight_scale_2", Dtype::F32, &[]),
        ("layer.input_scale", Dtype::F32, &[]),
    ]);
    let view = detect("layer.weight", &t, None).unwrap().unwrap();
    assert_eq!(
        view.format,
        QuantFormat::Fp4E2M1Group {
            group_size: 16,
            global_scale_apply: infer_quant::ScaleApply::Multiply,
        }
    );
}

#[test]
fn w4a16() {
    // U8 [n, k//2], BF16 per-group scale [n, k/group].
    let t = tensors(&[
        ("layer.weight", Dtype::U8, &[64, 32]),
        ("layer.weight_scale", Dtype::BF16, &[64, 4]),
    ]);
    let view = detect("layer.weight", &t, None).unwrap().unwrap();
    assert_eq!(view.format, QuantFormat::W4A16 { group_size: 16 });
    assert_eq!(view.logical_shape, [64, 64]);
}

#[test]
fn w4a16_unaligned_k_is_rejected() {
    // logical K = 64, scale second dim 3 -> group_size = 21, 64 % 21 != 0.
    let t = tensors(&[
        ("layer.weight", Dtype::U8, &[64, 32]),
        ("layer.weight_scale", Dtype::BF16, &[64, 3]),
    ]);
    assert!(detect("layer.weight", &t, None).is_err());
}

#[test]
fn w4afp8() {
    // I8 [N, K//2] with K//2 a multiple of 256, interleaved scale [K//512, N*4].
    let t = tensors(&[
        ("layer.weight", Dtype::I8, &[64, 512]),
        ("layer.weight_scale", Dtype::BF16, &[2, 256]),
    ]);
    let view = detect("layer.weight", &t, None).unwrap().unwrap();
    assert_eq!(view.format, QuantFormat::W4Afp8);
    assert_eq!(view.logical_shape, [64, 1024]);
}

#[test]
fn w8a16() {
    let t = tensors(&[
        ("layer.weight", Dtype::I8, &[64, 128]),
        ("layer.weight_scale", Dtype::BF16, &[64, 4]),
    ]);
    let view = detect("layer.weight", &t, None).unwrap().unwrap();
    assert_eq!(view.format, QuantFormat::W8A16 { group_size: 32 });
    assert_eq!(view.logical_shape, [64, 128]);
}

#[test]
fn dsv4_e8m0_scale_abi_is_rejected() {
    let t = tensors(&[
        ("layer.weight", Dtype::F8_E4M3, &[64, 128]),
        ("layer.weight_scale", Dtype::F8_E8M0, &[1, 1]),
    ]);
    assert!(reject_dsv4_e8m0_scale_abi("layer.weight", &t).is_err());
    let t = tensors(&[
        ("layer.weight", Dtype::F8_E4M3, &[64, 128]),
        ("layer.weight_scale", Dtype::BF16, &[1, 1]),
    ]);
    assert!(reject_dsv4_e8m0_scale_abi("layer.weight", &t).is_ok());
}

#[test]
fn decode_f8_e4m3fn_known_values() {
    assert_eq!(decode_f8_e4m3fn(0x00), 0.0);
    assert_eq!(decode_f8_e4m3fn(0x38), 1.0);
    assert_eq!(decode_f8_e4m3fn(0xB8), -1.0);
    assert_eq!(decode_f8_e4m3fn(0x7E), 448.0);
}

#[test]
fn read_quant_manifest_parses_config() {
    let dir = unique_tmp_dir();
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join("config.json"),
        r#"{"quantization_config": {"quant_method": "gptq", "modules_to_not_convert": ["lm_head"]}}"#,
    )
    .unwrap();
    let manifest = read_quant_manifest(&dir).unwrap().unwrap();
    assert_eq!(manifest.quant_method.as_deref(), Some("gptq"));
    assert!(manifest.ignored("lm_head.weight"));
    assert!(!manifest.ignored("layer.weight"));
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn read_quant_manifest_without_quantization_config_is_none() {
    let dir = unique_tmp_dir();
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("config.json"), r#"{"model_type": "qwen3"}"#).unwrap();
    assert!(read_quant_manifest(&dir).unwrap().is_none());
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn read_quant_manifest_missing_config_errors() {
    let dir = unique_tmp_dir();
    fs::create_dir_all(&dir).unwrap();
    assert!(read_quant_manifest(&dir).is_err());
    fs::remove_dir_all(&dir).unwrap();
}

fn unique_tmp_dir() -> PathBuf {
    std::env::temp_dir().join(format!(
        "infer-quant-test-{}-{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("t")
    ))
}
