use std::{collections::BTreeMap, fs, path::Path};

use anyhow::{Context, Result, bail, ensure};
use safetensors::tensor::Dtype;
use serde::Deserialize;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ScaleApply {
    Multiply,
    Divide,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum QuantFormat {
    DenseBf16,
    DenseF32,
    Fp8BlockScaled {
        block_m: usize,
        block_k: usize,
        scale_apply: ScaleApply,
    },
    Fp8PerShard {
        scale_apply: ScaleApply,
    },
    Fp4E2M1Group {
        group_size: usize,
        global_scale_apply: ScaleApply,
    },
    W4A16 {
        group_size: usize,
    },
    /// GPTQ/AutoRound W4A16: qweight (U32 [k//8, n]), scales (BF16 [groups, n]),
    /// qzeros (U32 [groups, n//8]). Converted to ARLE W4A16 layout at load time.
    GptqW4A16 {
        group_size: usize,
    },
    W8A16 {
        group_size: usize,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TensorHeader {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct QuantTensorView {
    pub name: String,
    pub logical_shape: Vec<usize>,
    pub storage_dtype: Dtype,
    pub format: QuantFormat,
    pub scale_names: Vec<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq)]
pub(crate) struct QuantManifest {
    pub quant_method: Option<String>,
    pub fmt: Option<String>,
    pub format: Option<String>,
    pub activation_scheme: Option<String>,
    #[serde(default)]
    pub modules_to_not_convert: Vec<String>,
    #[serde(default)]
    pub ignore: Vec<String>,
}

impl QuantManifest {
    pub(crate) fn ignored(&self, tensor_name: &str) -> bool {
        self.modules_to_not_convert
            .iter()
            .chain(self.ignore.iter())
            .any(|prefix| tensor_name.starts_with(prefix))
    }
}

pub(crate) fn read_quant_manifest(model_path: &Path) -> Result<Option<QuantManifest>> {
    let config_path = model_path.join("config.json");
    let content = fs::read_to_string(&config_path)
        .with_context(|| format!("read quant manifest config {}", config_path.display()))?;
    let root: serde_json::Value = serde_json::from_str(&content)
        .with_context(|| format!("parse quant manifest config {}", config_path.display()))?;
    let Some(value) = root
        .get("quantization_config")
        .or_else(|| root.pointer("/text_config/quantization_config"))
    else {
        return Ok(None);
    };
    Ok(Some(
        serde_json::from_value(value.clone()).context("parse quantization_config")?,
    ))
}

pub(crate) fn detect_quant_format(
    name: &str,
    tensors: &BTreeMap<String, TensorHeader>,
    manifest: Option<&QuantManifest>,
) -> Result<Option<QuantTensorView>> {
    let Some(tensor) = tensors.get(name) else {
        return Ok(None);
    };
    if !name.ends_with(".weight")
        && !name.ends_with(".weight_packed")
        && !name.ends_with(".qweight")
    {
        return Ok(None);
    }

    let base = weight_base(name);
    let view = match tensor.dtype {
        Dtype::BF16 if manifest.is_none_or(|m| m.ignored(name)) => QuantTensorView {
            name: name.to_owned(),
            logical_shape: tensor.shape.clone(),
            storage_dtype: tensor.dtype,
            format: QuantFormat::DenseBf16,
            scale_names: Vec::new(),
        },
        Dtype::F32 if manifest.is_none_or(|m| m.ignored(name)) => QuantTensorView {
            name: name.to_owned(),
            logical_shape: tensor.shape.clone(),
            storage_dtype: tensor.dtype,
            format: QuantFormat::DenseF32,
            scale_names: Vec::new(),
        },
        // GPTQ/AutoRound W4A16: qweight (I32 [k//8, n]), scales (BF16 [groups, n]),
        // qzeros (I32 [groups, n//8]). Converted to ARLE W4A16 layout at load time.
        Dtype::I32
            if tensors.contains_key(&format!("{base}.scales"))
                && tensors.contains_key(&format!("{base}.qzeros")) =>
        {
            let scales = tensor_by_name(tensors, &format!("{base}.scales"))?;
            ensure!(
                scales.shape.len() == 2,
                "{name}: GPTQ scales must be rank-2, got {:?}",
                scales.shape
            );
            let packed_k = tensor.shape[0];
            let n = tensor.shape[1];
            let k = packed_k * 8;
            let num_groups = scales.shape[0];
            ensure!(
                k % num_groups == 0,
                "{name}: GPTQ k={k} not divisible by num_groups={num_groups}"
            );
            let group_size = k / num_groups;
            QuantTensorView {
                name: name.to_owned(),
                logical_shape: vec![n, k],
                storage_dtype: tensor.dtype,
                format: QuantFormat::GptqW4A16 { group_size },
                scale_names: vec![format!("{base}.scales"), format!("{base}.qzeros")],
            }
        }
        Dtype::F8_E4M3 if tensors.contains_key(&format!("{base}.weight_scale_inv")) => {
            QuantTensorView {
                name: name.to_owned(),
                logical_shape: tensor.shape.clone(),
                storage_dtype: tensor.dtype,
                format: QuantFormat::Fp8BlockScaled {
                    block_m: 128,
                    block_k: 128,
                    scale_apply: ScaleApply::Multiply,
                },
                scale_names: vec![format!("{base}.weight_scale_inv")],
            }
        }
        Dtype::F8_E4M3
            if tensors.contains_key(&format!("{base}.weight_scale"))
                && tensors.contains_key(&format!("{base}.input_scale")) =>
        {
            QuantTensorView {
                name: name.to_owned(),
                logical_shape: tensor.shape.clone(),
                storage_dtype: tensor.dtype,
                format: QuantFormat::Fp8PerShard {
                    scale_apply: ScaleApply::Multiply,
                },
                scale_names: vec![
                    format!("{base}.weight_scale"),
                    format!("{base}.input_scale"),
                ],
            }
        }
        Dtype::U8
            if tensors.contains_key(&format!("{base}.weight_scale"))
                && tensors.contains_key(&format!("{base}.weight_global_scale"))
                && tensors.contains_key(&format!("{base}.input_global_scale")) =>
        {
            let logical_shape = fp4_logical_shape(name, &tensor.shape)?;
            QuantTensorView {
                name: name.to_owned(),
                logical_shape,
                storage_dtype: tensor.dtype,
                format: QuantFormat::Fp4E2M1Group {
                    group_size: 16,
                    global_scale_apply: ScaleApply::Divide,
                },
                scale_names: vec![
                    format!("{base}.weight_scale"),
                    format!("{base}.weight_global_scale"),
                    format!("{base}.input_global_scale"),
                ],
            }
        }
        Dtype::U8
            if tensors.contains_key(&format!("{base}.weight_scale"))
                && tensors.contains_key(&format!("{base}.weight_scale_2"))
                && tensors.contains_key(&format!("{base}.input_scale")) =>
        {
            let logical_shape = fp4_logical_shape(name, &tensor.shape)?;
            QuantTensorView {
                name: name.to_owned(),
                logical_shape,
                storage_dtype: tensor.dtype,
                format: QuantFormat::Fp4E2M1Group {
                    group_size: 16,
                    global_scale_apply: ScaleApply::Multiply,
                },
                scale_names: vec![
                    format!("{base}.weight_scale"),
                    format!("{base}.weight_scale_2"),
                    format!("{base}.input_scale"),
                ],
            }
        }
        Dtype::U8
            if tensors
                .get(&format!("{base}.weight_scale"))
                .is_some_and(|s| s.dtype == Dtype::BF16)
                && !tensors.contains_key(&format!("{base}.weight_global_scale"))
                && !tensors.contains_key(&format!("{base}.input_global_scale"))
                && !tensors.contains_key(&format!("{base}.weight_scale_2")) =>
        {
            let logical_shape = fp4_logical_shape(name, &tensor.shape)?;
            let scale = tensor_by_name(tensors, &format!("{base}.weight_scale"))?;
            ensure!(
                scale.shape.len() == 2,
                "{name}: W4A16 weight_scale must be rank-2, got {:?}",
                scale.shape
            );
            ensure!(
                scale.shape[1] != 0,
                "{name}: W4A16 weight_scale second dim is zero, got {:?}",
                scale.shape
            );
            let group_size = logical_shape[1] / scale.shape[1];
            ensure!(
                group_size > 0 && logical_shape[1] % group_size == 0,
                "{name}: W4A16 logical K {} not group-aligned to {group_size}",
                logical_shape[1]
            );
            QuantTensorView {
                name: name.to_owned(),
                logical_shape,
                storage_dtype: tensor.dtype,
                format: QuantFormat::W4A16 { group_size },
                scale_names: vec![format!("{base}.weight_scale")],
            }
        }
        // W8A16: signed INT8 weights, NOT packed (1 byte/elem, shape ==
        // logical), per-row per-column-group BF16 scale. Distinct from W4A16
        // by dtype (I8 vs U8), so no ambiguity.
        Dtype::I8
            if tensors
                .get(&format!("{base}.weight_scale"))
                .is_some_and(|s| s.dtype == Dtype::BF16) =>
        {
            let scale = tensor_by_name(tensors, &format!("{base}.weight_scale"))?;
            ensure!(
                scale.shape.len() == 2 && scale.shape[1] != 0,
                "{name}: W8A16 weight_scale must be rank-2 with nonzero K, got {:?}",
                scale.shape
            );
            let group_size = tensor.shape[1] / scale.shape[1];
            ensure!(
                group_size > 0 && tensor.shape[1] % group_size == 0,
                "{name}: W8A16 logical K {} not group-aligned to {group_size}",
                tensor.shape[1]
            );
            QuantTensorView {
                name: name.to_owned(),
                logical_shape: tensor.shape.clone(),
                storage_dtype: tensor.dtype,
                format: QuantFormat::W8A16 { group_size },
                scale_names: vec![format!("{base}.weight_scale")],
            }
        }
        _ => return Ok(None),
    };
    validate_scale_shapes(&view, tensors)?;
    Ok(Some(view))
}

pub(crate) fn validate_scale_shapes(
    view: &QuantTensorView,
    tensors: &BTreeMap<String, TensorHeader>,
) -> Result<()> {
    match view.format {
        QuantFormat::DenseBf16 | QuantFormat::DenseF32 => Ok(()),
        QuantFormat::Fp8BlockScaled {
            block_m, block_k, ..
        } => {
            ensure!(
                view.logical_shape.len() == 2,
                "{}: FP8 block-scaled weights must be rank-2",
                view.name
            );
            let scale = tensor_by_name(tensors, &view.scale_names[0])?;
            ensure!(
                scale.dtype == Dtype::BF16 || scale.dtype == Dtype::F32,
                "{}: FP8 block scale must be BF16/F32, got {:?}",
                view.scale_names[0],
                scale.dtype
            );
            let rows = view.logical_shape[0].div_ceil(block_m);
            let cols = view.logical_shape[1].div_ceil(block_k);
            ensure!(
                scale.shape == [rows, cols],
                "{}: FP8 scale shape {:?} != [{rows}, {cols}]",
                view.scale_names[0],
                scale.shape
            );
            Ok(())
        }
        QuantFormat::Fp8PerShard { .. } => {
            ensure!(
                view.scale_names.len() == 2,
                "{}: per-shard FP8 requires weight_scale and input_scale side tensors",
                view.name
            );
            for name in &view.scale_names {
                let scale = tensor_by_name(tensors, name)?;
                ensure!(
                    scale.dtype == Dtype::F32 || scale.dtype == Dtype::BF16,
                    "{name}: per-shard FP8 scale must be float, got {:?}",
                    scale.dtype
                );
                ensure!(
                    scale.shape.len() <= 1 && scale.shape.iter().all(|dim| *dim > 0),
                    "{name}: per-shard FP8 scale must be scalar or 1D, got {:?}",
                    scale.shape
                );
            }
            Ok(())
        }
        QuantFormat::Fp4E2M1Group { group_size, .. } => {
            ensure!(
                view.logical_shape.len() == 2,
                "{}: FP4 group weights must be rank-2",
                view.name
            );
            ensure!(
                view.logical_shape[1].is_multiple_of(group_size),
                "{}: FP4 logical K {} must be group-aligned to {group_size}",
                view.name,
                view.logical_shape[1]
            );
            let scale = tensor_by_name(tensors, &view.scale_names[0])?;
            ensure!(
                scale.dtype == Dtype::F8_E4M3,
                "{}: FP4 group scale must be F8_E4M3, got {:?}",
                view.scale_names[0],
                scale.dtype
            );
            let expected = [view.logical_shape[0], view.logical_shape[1] / group_size];
            ensure!(
                scale.shape == expected,
                "{}: FP4 group scale shape {:?} != {:?}",
                view.scale_names[0],
                scale.shape,
                expected
            );
            for name in view.scale_names.iter().skip(1) {
                let tensor = tensor_by_name(tensors, name)?;
                ensure!(
                    tensor.dtype == Dtype::F32,
                    "{name}: FP4 global/input scale must be F32, got {:?}",
                    tensor.dtype
                );
                ensure!(
                    tensor.shape.is_empty() || tensor.shape == [1],
                    "{name}: FP4 global/input scale must be scalar or [1], got {:?}",
                    tensor.shape
                );
            }
            Ok(())
        }
        QuantFormat::W4A16 { group_size } => {
            ensure!(
                view.logical_shape.len() == 2,
                "{}: W4A16 weights must be rank-2",
                view.name
            );
            ensure!(
                view.logical_shape[1].is_multiple_of(group_size),
                "{}: W4A16 logical K {} must be group-aligned to {group_size}",
                view.name,
                view.logical_shape[1]
            );
            let scale = tensor_by_name(tensors, &view.scale_names[0])?;
            ensure!(
                scale.dtype == Dtype::BF16,
                "{}: W4A16 weight_scale must be BF16, got {:?}",
                view.scale_names[0],
                scale.dtype
            );
            let expected = [view.logical_shape[0], view.logical_shape[1] / group_size];
            ensure!(
                scale.shape == expected,
                "{}: W4A16 weight_scale shape {:?} != {:?}",
                view.scale_names[0],
                scale.shape,
                expected
            );
            Ok(())
        }
        QuantFormat::GptqW4A16 { group_size } => {
            ensure!(
                view.logical_shape.len() == 2,
                "{}: GPTQ W4A16 weights must be rank-2",
                view.name
            );
            let n = view.logical_shape[0];
            let k = view.logical_shape[1];
            ensure!(
                k.is_multiple_of(group_size),
                "{}: GPTQ W4A16 logical K {} must be group-aligned to {group_size}",
                view.name,
                k
            );
            let num_groups = k / group_size;
            // scales: BF16 or F16 [num_groups, n]
            let scales = tensor_by_name(tensors, &view.scale_names[0])?;
            ensure!(
                scales.dtype == Dtype::BF16 || scales.dtype == Dtype::F16,
                "{}: GPTQ scales must be BF16 or F16, got {:?}",
                view.scale_names[0],
                scales.dtype
            );
            ensure!(
                scales.shape == [num_groups, n],
                "{}: GPTQ scales shape {:?} != [{}, {}]",
                view.scale_names[0],
                scales.shape,
                num_groups,
                n
            );
            // qzeros: I32 [num_groups, n//8]
            let qzeros = tensor_by_name(tensors, &view.scale_names[1])?;
            ensure!(
                qzeros.dtype == Dtype::I32,
                "{}: GPTQ qzeros must be I32, got {:?}",
                view.scale_names[1],
                qzeros.dtype
            );
            ensure!(
                qzeros.shape == [num_groups, n / 8],
                "{}: GPTQ qzeros shape {:?} != [{}, {}]",
                view.scale_names[1],
                qzeros.shape,
                num_groups,
                n / 8
            );
            Ok(())
        }
        QuantFormat::W8A16 { group_size } => {
            ensure!(
                view.logical_shape.len() == 2,
                "{}: W8A16 weights must be rank-2",
                view.name
            );
            ensure!(
                view.logical_shape[1].is_multiple_of(group_size),
                "{}: W8A16 logical K {} must be group-aligned to {group_size}",
                view.name,
                view.logical_shape[1]
            );
            let scale = tensor_by_name(tensors, &view.scale_names[0])?;
            ensure!(
                scale.dtype == Dtype::BF16,
                "{}: W8A16 weight_scale must be BF16, got {:?}",
                view.scale_names[0],
                scale.dtype
            );
            let expected = [view.logical_shape[0], view.logical_shape[1] / group_size];
            ensure!(
                scale.shape == expected,
                "{}: W8A16 weight_scale shape {:?} != {:?}",
                view.scale_names[0],
                scale.shape,
                expected
            );
            Ok(())
        }
    }
}

pub(crate) fn decode_f8_e4m3fn(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = (byte >> 3) & 0x0f;
    let mant = byte & 0x07;
    const BIAS: i32 = 7;
    if exp == 0 {
        return sign * (mant as f32 / 8.0) * 2.0f32.powi(1 - BIAS);
    }
    sign * (1.0 + mant as f32 / 8.0) * 2.0f32.powi(exp as i32 - BIAS)
}

pub(crate) fn decode_fp4_e2m1(nibble: u8) -> f32 {
    let nibble = nibble & 0x0f;
    let sign = if nibble & 0x08 != 0 { -1.0 } else { 1.0 };
    let code = nibble & 0x07;
    let value = match code {
        0 => 0.0,
        1 => 0.5,
        2 => 1.0,
        3 => 1.5,
        4 => 2.0,
        5 => 3.0,
        6 => 4.0,
        7 => 6.0,
        _ => unreachable!(),
    };
    sign * value
}

fn tensor_by_name<'a>(
    tensors: &'a BTreeMap<String, TensorHeader>,
    name: &str,
) -> Result<&'a TensorHeader> {
    tensors
        .get(name)
        .with_context(|| format!("missing quant side tensor {name}"))
}

fn weight_base(name: &str) -> &str {
    name.strip_suffix(".weight_packed")
        .or_else(|| name.strip_suffix(".qweight"))
        .or_else(|| name.strip_suffix(".weight"))
        .unwrap_or(name)
}

fn fp4_logical_shape(name: &str, storage_shape: &[usize]) -> Result<Vec<usize>> {
    ensure!(
        storage_shape.len() == 2,
        "{name}: FP4 packed weights must be rank-2, got {:?}",
        storage_shape
    );
    Ok(vec![storage_shape[0], storage_shape[1] * 2])
}

pub(crate) fn reject_dsv4_e8m0_scale_abi(
    name: &str,
    tensors: &BTreeMap<String, TensorHeader>,
) -> Result<()> {
    let base = weight_base(name);
    for suffix in [".weight_scale", ".scale", ".scales"] {
        let scale_name = format!("{base}{suffix}");
        if tensors
            .get(&scale_name)
            .is_some_and(|tensor| tensor.dtype == Dtype::F8_E8M0)
        {
            bail!("{name}: DSv4 E8M0 scale ABI is not a Qwen resident quant format");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(dtype: Dtype, shape: &[usize]) -> TensorHeader {
        TensorHeader {
            dtype,
            shape: shape.to_vec(),
        }
    }

    #[test]
    fn decode_f8_e4m3fn_known_values() {
        assert_eq!(decode_f8_e4m3fn(0x00), 0.0);
        assert_eq!(decode_f8_e4m3fn(0x38), 1.0);
        assert_eq!(decode_f8_e4m3fn(0xb8), -1.0);
        assert_eq!(decode_f8_e4m3fn(0x40), 2.0);
        assert_eq!(decode_f8_e4m3fn(0x30), 0.5);
    }

    #[test]
    fn decode_fp4_e2m1_known_values() {
        let values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        for (code, value) in values.iter().copied().enumerate() {
            assert_eq!(decode_fp4_e2m1(code as u8), value);
            assert_eq!(decode_fp4_e2m1((code as u8) | 0x08), -value);
        }
    }

    #[test]
    fn fp8_block_scale_shape_validation() {
        let mut tensors = BTreeMap::new();
        let weight = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight";
        tensors.insert(weight.to_owned(), header(Dtype::F8_E4M3, &[512, 2048]));
        tensors.insert(
            "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv".to_owned(),
            header(Dtype::BF16, &[4, 16]),
        );

        let view = detect_quant_format(weight, &tensors, None)
            .unwrap()
            .expect("FP8 view");
        assert_eq!(
            view.format,
            QuantFormat::Fp8BlockScaled {
                block_m: 128,
                block_k: 128,
                scale_apply: ScaleApply::Multiply,
            }
        );
    }

    #[test]
    fn w8a16_int8_detect_and_group_size() {
        let mut tensors = BTreeMap::new();
        let base = "model.language_model.layers.0.mlp.down_proj";
        let weight = format!("{base}.weight");
        // INT8 weight [512, 2048], non-packed; BF16 scale [512, 16] => group 128.
        tensors.insert(weight.clone(), header(Dtype::I8, &[512, 2048]));
        tensors.insert(
            format!("{base}.weight_scale"),
            header(Dtype::BF16, &[512, 16]),
        );

        let view = detect_quant_format(&weight, &tensors, None)
            .unwrap()
            .expect("W8A16 view");
        assert_eq!(view.format, QuantFormat::W8A16 { group_size: 128 });
        assert_eq!(view.logical_shape, vec![512, 2048]);
        validate_scale_shapes(&view, &tensors).expect("W8A16 scale shape valid");

        // Wrong scale K (not group-divisible) must reject at detect.
        let mut bad = tensors.clone();
        bad.insert(
            format!("{base}.weight_scale"),
            header(Dtype::BF16, &[512, 15]),
        );
        assert!(detect_quant_format(&weight, &bad, None).is_err());
    }

    #[test]
    fn fp8_per_shard_validates_input_scale_sidecar() {
        let mut tensors = BTreeMap::new();
        let base = "model.language_model.layers.0.mlp.down_proj";
        let weight = format!("{base}.weight");
        tensors.insert(weight.clone(), header(Dtype::F8_E4M3, &[2048, 512]));
        tensors.insert(format!("{base}.weight_scale"), header(Dtype::F32, &[2048]));
        tensors.insert(format!("{base}.input_scale"), header(Dtype::U8, &[1]));
        assert!(detect_quant_format(&weight, &tensors, None).is_err());

        tensors.insert(format!("{base}.input_scale"), header(Dtype::F32, &[1, 1]));
        assert!(detect_quant_format(&weight, &tensors, None).is_err());

        tensors.insert(format!("{base}.input_scale"), header(Dtype::F32, &[1]));
        let view = detect_quant_format(&weight, &tensors, None)
            .unwrap()
            .expect("per-shard FP8 view");
        assert_eq!(
            view.format,
            QuantFormat::Fp8PerShard {
                scale_apply: ScaleApply::Multiply,
            }
        );
        assert_eq!(
            view.scale_names,
            [
                format!("{base}.weight_scale"),
                format!("{base}.input_scale")
            ]
        );
    }

    #[test]
    fn fp4_e2m1_group16_shape_validation() {
        let mut tensors = BTreeMap::new();
        let base = "model.language_model.layers.0.mlp.experts.0.gate_proj";
        let weight = format!("{base}.weight_packed");
        tensors.insert(weight.clone(), header(Dtype::U8, &[512, 1024]));
        tensors.insert(
            format!("{base}.weight_scale"),
            header(Dtype::F8_E4M3, &[512, 128]),
        );
        tensors.insert(
            format!("{base}.weight_global_scale"),
            header(Dtype::F32, &[1]),
        );
        tensors.insert(
            format!("{base}.input_global_scale"),
            header(Dtype::F32, &[1]),
        );

        let view = detect_quant_format(&weight, &tensors, None)
            .unwrap()
            .expect("FP4 view");
        assert_eq!(view.logical_shape, [512, 2048]);
        assert_eq!(
            view.format,
            QuantFormat::Fp4E2M1Group {
                group_size: 16,
                global_scale_apply: ScaleApply::Divide,
            }
        );
    }

    #[test]
    fn fp4_e2m1_group16_rejects_unaligned_k() {
        let mut tensors = BTreeMap::new();
        let base = "x";
        tensors.insert(format!("{base}.weight_packed"), header(Dtype::U8, &[4, 15]));
        tensors.insert(
            format!("{base}.weight_scale"),
            header(Dtype::F8_E4M3, &[4, 2]),
        );
        tensors.insert(
            format!("{base}.weight_global_scale"),
            header(Dtype::F32, &[1]),
        );
        tensors.insert(
            format!("{base}.input_global_scale"),
            header(Dtype::F32, &[1]),
        );

        assert!(detect_quant_format(&format!("{base}.weight_packed"), &tensors, None).is_err());
    }

    #[test]
    fn fp4_e2m1_requires_all_side_tensors() {
        let mut tensors = BTreeMap::new();
        let base = "model.language_model.layers.0.mlp.experts.0.down_proj";
        let weight = format!("{base}.weight_packed");
        tensors.insert(weight.clone(), header(Dtype::U8, &[2048, 256]));
        tensors.insert(
            format!("{base}.weight_scale"),
            header(Dtype::F8_E4M3, &[2048, 32]),
        );
        tensors.insert(
            format!("{base}.weight_global_scale"),
            header(Dtype::F32, &[1]),
        );

        assert!(
            detect_quant_format(&weight, &tensors, None)
                .unwrap()
                .is_none()
        );

        tensors.insert(
            format!("{base}.input_global_scale"),
            header(Dtype::F32, &[1]),
        );
        assert!(
            detect_quant_format(&weight, &tensors, None)
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn quant_format_rejects_dsv4_e8m0_scale_abi() {
        let mut tensors = BTreeMap::new();
        tensors.insert("w.weight".to_owned(), header(Dtype::F8_E4M3, &[128, 128]));
        tensors.insert("w.weight_scale".to_owned(), header(Dtype::F8_E8M0, &[1, 1]));
        assert!(reject_dsv4_e8m0_scale_abi("w.weight", &tensors).is_err());
    }

    #[test]
    fn manifest_ignore_preserves_dense_bf16() {
        let mut tensors = BTreeMap::new();
        let weight = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight";
        tensors.insert(weight.to_owned(), header(Dtype::BF16, &[8192, 2048]));
        let manifest = QuantManifest {
            ignore: vec!["model.language_model.layers.0.linear_attn".to_owned()],
            ..QuantManifest::default()
        };
        let view = detect_quant_format(weight, &tensors, Some(&manifest))
            .unwrap()
            .expect("dense view");
        assert_eq!(view.format, QuantFormat::DenseBf16);
    }

    #[test]
    fn manifest_ignore_does_not_hide_real_fp8_sidecars() {
        let mut tensors = BTreeMap::new();
        let base = "model.language_model.layers.0.linear_attn.in_proj_qkv";
        let weight = format!("{base}.weight");
        tensors.insert(weight.clone(), header(Dtype::F8_E4M3, &[8192, 2048]));
        tensors.insert(
            format!("{base}.weight_scale_inv"),
            header(Dtype::BF16, &[64, 16]),
        );
        let manifest = QuantManifest {
            ignore: vec!["model.language_model.layers.0.linear_attn".to_owned()],
            ..QuantManifest::default()
        };
        let view = detect_quant_format(&weight, &tensors, Some(&manifest))
            .unwrap()
            .expect("FP8 view");
        assert_eq!(
            view.format,
            QuantFormat::Fp8BlockScaled {
                block_m: 128,
                block_k: 128,
                scale_apply: ScaleApply::Multiply,
            }
        );
    }
}
