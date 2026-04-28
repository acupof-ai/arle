use std::{env, path::PathBuf};

use super::*;
use crate::backend::metal::forward::build_forward_graph;
use crate::backend::metal::sampling::{gpu_sample_token, gpu_sample_token_batched};
use crate::backend::metal::{
    config::load_metal_config,
    gdr::MetalRecurrentState,
    mlx::{
        Dtype, as_dtype, concatenate_axis, eval, gguf_quantized_matmul, reshape, slice,
        slice_update, zeros,
    },
    weights::load_qwen3_metal_weights,
};
use crate::gguf::dequant_to_bf16;
use crate::test_support::metal_test_guard;
use crate::tokenizer::Tokenizer;
use half::f16;

fn slice_row_for_sampling(array: &MlxArray, row: i32) -> MlxArray {
    let mut start = vec![0; array.shape().len()];
    let mut end = array.shape().to_vec();
    let strides = vec![1; array.shape().len()];
    start[0] = row;
    end[0] = row + 1;
    slice(array, &start, &end, &strides)
}

#[test]
fn append_qwen35_captured_hidden_chunk_concatenates_context_rows() {
    let _guard = metal_test_guard();
    let mut accumulated = None;
    append_qwen35_captured_hidden_chunk(
        &mut accumulated,
        Some(MlxArray::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2])),
    );
    append_qwen35_captured_hidden_chunk(
        &mut accumulated,
        Some(MlxArray::from_slice_f32(&[5.0, 6.0], &[1, 2])),
    );

    let combined = accumulated.expect("captured hidden");
    let combined = as_dtype(&combined, Dtype::Float32);
    eval(&[&combined]);
    assert_eq!(combined.shape(), &[3, 2]);
    assert_eq!(combined.as_slice_f32(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

fn put_f16(dst: &mut [u8], offset: usize, value: f32) {
    dst[offset..offset + 2].copy_from_slice(&f16::from_f32(value).to_le_bytes());
}

fn synthetic_gguf_packed(format: GgufPackedFormat, rows: usize, cols: usize) -> Vec<u8> {
    let row_bytes = (cols / format.block_size()) * format.block_bytes();
    let mut raw = vec![0u8; rows * row_bytes];
    for row in 0..rows {
        for block in 0..(cols / format.block_size()) {
            let base = row * row_bytes + block * format.block_bytes();
            for i in 0..format.block_bytes() {
                raw[base + i] = (17 * row + 31 * block + 13 * i + 7) as u8;
            }
            match format {
                GgufPackedFormat::Q8_0 => {
                    put_f16(&mut raw, base, 0.00390625);
                    for i in 0..32 {
                        raw[base + 2 + i] = ((i as i8) - 16) as u8;
                    }
                }
                GgufPackedFormat::Q3_K => {
                    for i in 96..108 {
                        raw[base + i] = 0x11 + (i as u8 & 0x03);
                    }
                    put_f16(&mut raw, base + 108, 0.00390625);
                }
                GgufPackedFormat::Q4_K => {
                    put_f16(&mut raw, base, 0.00390625);
                    put_f16(&mut raw, base + 2, 0.001953125);
                }
                GgufPackedFormat::Q5_K => {
                    put_f16(&mut raw, base, 0.00390625);
                    put_f16(&mut raw, base + 2, 0.001953125);
                }
                GgufPackedFormat::Q6_K => {
                    for i in 192..208 {
                        raw[base + i] = (i as u8).wrapping_sub(200);
                    }
                    put_f16(&mut raw, base + 208, 0.00390625);
                }
            }
        }
    }
    raw
}

fn gguf_dtype_for_format(format: GgufPackedFormat) -> GgmlType {
    match format {
        GgufPackedFormat::Q8_0 => GgmlType::Q8_0,
        GgufPackedFormat::Q3_K => GgmlType::Q3_K,
        GgufPackedFormat::Q4_K => GgmlType::Q4_K,
        GgufPackedFormat::Q5_K => GgmlType::Q5_K,
        GgufPackedFormat::Q6_K => GgmlType::Q6_K,
    }
}

#[test]
fn gguf_packed_format_metadata_covers_supported_metal_kernels() {
    assert_eq!(GgufPackedFormat::Q8_0.block_size(), 32);
    assert_eq!(GgufPackedFormat::Q8_0.block_bytes(), 34);
    assert_eq!(GgufPackedFormat::Q3_K.block_size(), 256);
    assert_eq!(GgufPackedFormat::Q3_K.block_bytes(), 110);
    assert_eq!(GgufPackedFormat::Q4_K.block_bytes(), 144);
    assert_eq!(GgufPackedFormat::Q5_K.block_bytes(), 176);
    assert_eq!(GgufPackedFormat::Q6_K.block_bytes(), 210);
}

#[test]
fn qwen35_grouped_v_reorder_preserves_packed_rows() {
    let rows = 12;
    let row_bytes = 3;
    let src: Vec<u8> = (0..rows * row_bytes).map(|i| i as u8).collect();
    let dst = reorder_gguf_packed_v_rows(&src, rows, row_bytes, 2, 3, 2, "dummy").unwrap();
    let src_rows = src.chunks_exact(row_bytes).collect::<Vec<_>>();
    let dst_rows = dst.chunks_exact(row_bytes).collect::<Vec<_>>();
    assert_eq!(dst_rows[0], src_rows[0]);
    assert_eq!(dst_rows[1], src_rows[1]);
    assert_eq!(dst_rows[2], src_rows[4]);
    assert_eq!(dst_rows[3], src_rows[5]);
    assert_eq!(dst_rows[4], src_rows[8]);
    assert_eq!(dst_rows[5], src_rows[9]);
    assert_eq!(dst_rows[6], src_rows[2]);
    assert_eq!(dst_rows[7], src_rows[3]);
    assert_eq!(dst_rows[8], src_rows[6]);
    assert_eq!(dst_rows[9], src_rows[7]);
    assert_eq!(dst_rows[10], src_rows[10]);
    assert_eq!(dst_rows[11], src_rows[11]);
}

#[test]
fn gguf_quantized_matmul_matches_cpu_reference_for_all_metal_packed_formats() {
    let _guard = metal_test_guard();
    let formats = [
        GgufPackedFormat::Q8_0,
        GgufPackedFormat::Q3_K,
        GgufPackedFormat::Q4_K,
        GgufPackedFormat::Q5_K,
        GgufPackedFormat::Q6_K,
    ];
    for format in formats {
        let rows = 3usize;
        let cols = 256usize;
        let raw = synthetic_gguf_packed(format, rows, cols);
        let deq = dequant_to_bf16(&raw, gguf_dtype_for_format(format), rows * cols)
            .unwrap()
            .into_iter()
            .map(f32::from)
            .collect::<Vec<_>>();

        for m in [1usize, 3] {
            let x_host = (0..m * cols)
                .map(|i| ((i % 13) as f32 - 6.0) * 0.001953125)
                .collect::<Vec<_>>();
            let x = MlxArray::from_slice_f32(&x_host, &[m as i32, cols as i32]);
            let w = MlxArray::from_slice_u8(&raw, &[raw.len() as i32]);
            let y = gguf_quantized_matmul(&x, &w, format.as_i32(), rows as i32, cols as i32);
            let y = as_dtype(&y, Dtype::Float32);
            eval(&[&y]);
            let actual = y.as_slice_f32();

            let mut expected = vec![0.0f32; m * rows];
            for mi in 0..m {
                for row in 0..rows {
                    let mut sum = 0.0f32;
                    for k in 0..cols {
                        sum += x_host[mi * cols + k] * deq[row * cols + k];
                    }
                    expected[mi * rows + row] = sum;
                }
            }
            for (idx, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
                let diff = (got - want).abs();
                assert!(
                    diff < 0.15,
                    "format={format:?} m={m} idx={idx} got={got} want={want} diff={diff}"
                );
            }
        }
    }
}

#[test]
fn gguf_affine_repack_matches_cpu_reference() {
    let _guard = metal_test_guard();
    for format in [
        GgufPackedFormat::Q8_0,
        GgufPackedFormat::Q4_K,
        GgufPackedFormat::Q5_K,
        GgufPackedFormat::Q6_K,
    ] {
        let rows = 3usize;
        let cols = 256usize;
        let raw = synthetic_gguf_packed(format, rows, cols);
        let deq = dequant_to_bf16(&raw, gguf_dtype_for_format(format), rows * cols)
            .unwrap()
            .into_iter()
            .map(f32::from)
            .collect::<Vec<_>>();
        let weight = gguf_affine_weight_from_bytes("dummy.weight", &raw, format, rows, cols)
            .expect("repack GGUF K-quant to MLX affine")
            .expect("expected exact affine repack");

        let WeightTensor::Quantized {
            group_size, bits, ..
        } = &weight
        else {
            panic!("expected affine repack for {format:?}");
        };
        match format {
            GgufPackedFormat::Q8_0 => {
                assert_eq!((*group_size, *bits), (32, 8));
            }
            GgufPackedFormat::Q4_K => {
                assert_eq!((*group_size, *bits), (32, 4));
            }
            GgufPackedFormat::Q5_K => {
                assert_eq!((*group_size, *bits), (32, 5));
            }
            GgufPackedFormat::Q6_K => {
                assert_eq!((*group_size, *bits), (16, 6));
            }
            _ => unreachable!(),
        }

        for m in [1usize, 3] {
            let x_host = (0..m * cols)
                .map(|i| ((i % 13) as f32 - 6.0) * 0.001953125)
                .collect::<Vec<_>>();
            let x = MlxArray::from_slice_f32(&x_host, &[m as i32, cols as i32]);
            let y = linear(&x, &weight);
            let y = as_dtype(&y, Dtype::Float32);
            eval(&[&y]);
            let actual = y.as_slice_f32();

            let mut expected = vec![0.0f32; m * rows];
            for mi in 0..m {
                for row in 0..rows {
                    let mut sum = 0.0f32;
                    for k in 0..cols {
                        sum += x_host[mi * cols + k] * deq[row * cols + k];
                    }
                    expected[mi * rows + row] = sum;
                }
            }
            for (idx, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
                let diff = (got - want).abs();
                assert!(
                    diff < 0.15,
                    "format={format:?} m={m} idx={idx} got={got} want={want} diff={diff}"
                );
            }
        }
    }
}

#[test]
fn gguf_native_q4_requantizes_to_mlx_group64() {
    let _guard = metal_test_guard();
    let rows = 3usize;
    let cols = 256usize;
    let raw = synthetic_gguf_packed(GgufPackedFormat::Q6_K, rows, cols);
    let deq = dequant_to_bf16(&raw, GgmlType::Q6_K, rows * cols).unwrap();
    let weight = native_q4_weight_from_bf16_host(
        "dummy.weight",
        &HostTensor {
            data: deq,
            shape: vec![rows, cols],
        },
    )
    .expect("requantize GGUF weight to MLX native q4");

    let WeightTensor::Quantized {
        group_size,
        bits,
        w,
        scales,
        biases,
    } = &weight
    else {
        panic!("expected native q4 quantized weight");
    };
    assert_eq!((*group_size, *bits), (64, 4));
    assert_eq!(w.shape(), &[rows as i32, (cols / 8) as i32]);
    assert_eq!(scales.shape(), &[rows as i32, (cols / 64) as i32]);
    assert_eq!(biases.shape(), &[rows as i32, (cols / 64) as i32]);

    let x = MlxArray::from_slice_f32(&vec![0.001; cols], &[1, cols as i32]);
    let y = linear(&x, &weight);
    eval(&[&y]);
    assert_eq!(y.shape(), &[1, rows as i32]);
}

fn left_pad_kv_cache_row_for_test(
    array: &MlxArray,
    left_pad: i32,
    cache_len: i32,
    target_kv_capacity: i32,
) -> MlxArray {
    let shape = array.shape();
    assert_eq!(shape.len(), 4);
    assert_eq!(shape[0], 1);

    let n_kv = shape[1];
    let head_dim = shape[3];
    let mut padded = zeros(&[1, n_kv, target_kv_capacity, head_dim], array.dtype());
    if cache_len == 0 {
        return padded;
    }

    let valid = slice(
        array,
        &[0, 0, 0, 0],
        &[1, n_kv, cache_len, head_dim],
        &[1, 1, 1, 1],
    );
    padded = slice_update(
        &mut padded,
        &valid,
        &[0, 0, left_pad, 0],
        &[1, n_kv, left_pad + cache_len, head_dim],
    );
    padded
}

fn qwen3_model_path() -> Option<PathBuf> {
    env::var_os("QWEN3_MODEL_PATH")
        .map(PathBuf::from)
        .or_else(|| {
            let fallback = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models/Qwen3-0.6B");
            fallback.exists().then_some(fallback)
        })
}

fn qwen35_safetensors_model_path() -> Option<PathBuf> {
    env::var_os("QWEN35_MODEL_PATH")
        .map(PathBuf::from)
        .or_else(|| {
            let fallback = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models/Qwen3.5-0.8B");
            fallback.exists().then_some(fallback)
        })
}

fn qwen35_gguf_model_path() -> Option<PathBuf> {
    env::var_os("QWEN35_GGUF_PATH")
        .map(PathBuf::from)
        .or_else(|| {
            let fallback = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf");
            fallback.exists().then_some(fallback)
        })
}

fn dense_weight_f32(weight: &WeightTensor) -> Result<MlxArray> {
    match weight {
        WeightTensor::Dense(tensor) => {
            let tensor = as_dtype(tensor, Dtype::Float32);
            eval(&[&tensor]);
            Ok(tensor)
        }
        WeightTensor::Quantized { .. }
        | WeightTensor::GgufPacked { .. }
        | WeightTensor::GgufPackedInputReordered { .. } => {
            anyhow::bail!("expected dense weight tensor")
        }
    }
}

fn tensor_diff_stats(lhs: &MlxArray, rhs: &MlxArray) -> (f32, usize, f32, f32) {
    let lhs = as_dtype(lhs, Dtype::Float32);
    let rhs = as_dtype(rhs, Dtype::Float32);
    eval(&[&lhs, &rhs]);
    assert_eq!(lhs.shape(), rhs.shape(), "shape mismatch");

    let lhs_slice = lhs.as_slice_f32();
    let rhs_slice = rhs.as_slice_f32();
    let mut max_abs = 0.0_f32;
    let mut max_idx = 0_usize;
    let mut diff_sq = 0.0_f32;
    let mut rhs_sq = 0.0_f32;
    for (idx, (&lhs_value, &rhs_value)) in lhs_slice.iter().zip(rhs_slice.iter()).enumerate() {
        let diff = lhs_value - rhs_value;
        let abs = diff.abs();
        if abs > max_abs {
            max_abs = abs;
            max_idx = idx;
        }
        diff_sq += diff * diff;
        rhs_sq += rhs_value * rhs_value;
    }

    let rms = (diff_sq / lhs_slice.len().max(1) as f32).sqrt();
    let rel_rms = rms / rhs_sq.max(1e-12).sqrt();
    (max_abs, max_idx, rms, rel_rms)
}

fn print_tensor_diff(name: &str, lhs: &MlxArray, rhs: &MlxArray) {
    let lhs = as_dtype(lhs, Dtype::Float32);
    let rhs = as_dtype(rhs, Dtype::Float32);
    eval(&[&lhs, &rhs]);
    assert_eq!(lhs.shape(), rhs.shape(), "{name}: shape mismatch");

    let lhs_slice = lhs.as_slice_f32();
    let rhs_slice = rhs.as_slice_f32();
    let (max_abs, max_idx, rms, rel_rms) = tensor_diff_stats(&lhs, &rhs);
    let head_len = lhs_slice.len().min(8);
    eprintln!(
        "{name}: shape={:?} max_abs={max_abs:.6} @ {max_idx} rms={rms:.6} rel_rms={rel_rms:.6} lhs_head={:?} rhs_head={:?}",
        lhs.shape(),
        &lhs_slice[..head_len],
        &rhs_slice[..head_len],
    );
}

fn print_tensor_diff_with_transpose_hint(name: &str, lhs: &MlxArray, rhs: &MlxArray) {
    print_tensor_diff(name, lhs, rhs);
    if lhs.shape().len() != 2 || rhs.shape().len() != 2 {
        return;
    }
    let rhs_t = transpose_axes(rhs, &[1, 0]);
    eval(&[&rhs_t]);
    if lhs.shape() != rhs_t.shape() {
        eprintln!(
            "{name}: transpose_hint skipped lhs_shape={:?} rhs_t_shape={:?}",
            lhs.shape(),
            rhs_t.shape(),
        );
        return;
    }
    let (max_abs, max_idx, rms, rel_rms) = tensor_diff_stats(lhs, &rhs_t);
    eprintln!(
        "{name}: transpose_hint max_abs={max_abs:.6} @ {max_idx} rms={rms:.6} rel_rms={rel_rms:.6}",
    );
}

fn print_embed_row_diff(
    name: &str,
    lhs: &Qwen35MetalWeights,
    rhs: &Qwen35MetalWeights,
    row: i32,
    width: i32,
) {
    let token = MlxArray::from_slice_i32(&[row], &[1]);
    let lhs_row = slice(
        &qwen35_embed_tokens(lhs, &token),
        &[0, 0],
        &[1, width],
        &[1, 1],
    );
    let rhs_row = slice(
        &qwen35_embed_tokens(rhs, &token),
        &[0, 0],
        &[1, width],
        &[1, 1],
    );
    print_tensor_diff(name, &lhs_row, &rhs_row);
}

fn print_gguf_tensor_info(gguf: &GgufFile, hf_name: &str) -> Result<()> {
    let Ok(gguf_name) = crate::gguf::find_tensor_name(gguf, hf_name) else {
        eprintln!("{hf_name}: gguf lookup missing");
        return Ok(());
    };
    let info = &gguf.tensors[&gguf_name];
    let raw_len = gguf.read_tensor_raw(&gguf_name)?.len();
    eprintln!(
        "{hf_name}: gguf_name={gguf_name} dtype={:?} shape={:?} raw_bytes={} size_bytes={}",
        info.dtype,
        info.shape,
        raw_len,
        info.size_bytes(),
    );
    Ok(())
}

fn topk_logits(logits: &MlxArray, k: usize) -> Vec<(usize, f32)> {
    let logits = as_dtype(logits, Dtype::Float32);
    eval(&[&logits]);
    let shape = logits.shape();
    let slice = logits.as_slice_f32();
    let row = match shape {
        [vocab] => &slice[..*vocab as usize],
        [batch, vocab] => {
            let batch = *batch as usize;
            let vocab = *vocab as usize;
            let start = batch.saturating_sub(1) * vocab;
            &slice[start..start + vocab]
        }
        other => panic!("unexpected logits shape: {other:?}"),
    };

    let mut pairs: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
    pairs.sort_unstable_by(|lhs, rhs| rhs.1.total_cmp(&lhs.1));
    pairs.truncate(k);
    pairs
}

fn clone_linear_attn_weights(weights: &MetalLinearAttnWeights) -> MetalLinearAttnWeights {
    MetalLinearAttnWeights {
        in_proj_qkvz: weights.in_proj_qkvz.clone(),
        in_proj_ba: weights.in_proj_ba.clone(),
        in_proj_qkv: weights.in_proj_qkv.clone(),
        in_proj_z: weights.in_proj_z.clone(),
        in_proj_b: weights.in_proj_b.clone(),
        in_proj_a: weights.in_proj_a.clone(),
        qkvz_split: weights.qkvz_split,
        ba_num_heads: weights.ba_num_heads,
        conv1d_weight: weights.conv1d_weight.clone(),
        dt_bias: weights.dt_bias.clone(),
        a_log: weights.a_log.clone(),
        norm_weight: weights.norm_weight.clone(),
        out_proj: weights.out_proj.clone(),
        q_scale: weights.q_scale.clone(),
        k_scale: weights.k_scale.clone(),
    }
}

fn clone_dense_mlp_weights(weights: &MetalQwen35DenseMlpWeights) -> MetalQwen35DenseMlpWeights {
    MetalQwen35DenseMlpWeights {
        inputs: MlpInputProjection::Split {
            gate_proj: weights.gate_proj.clone(),
            up_proj: weights.up_proj.clone(),
        },
        down_proj: weights.down_proj.clone(),
        gate_proj: weights.gate_proj.clone(),
        up_proj: weights.up_proj.clone(),
    }
}

fn clone_dense_mlp_kind(mlp: &MlpKind) -> Result<MlpKind> {
    match mlp {
        MlpKind::Dense(weights) => Ok(MlpKind::Dense(clone_dense_mlp_weights(weights))),
        MlpKind::Moe(_) => anyhow::bail!("expected dense MLP"),
    }
}

fn clone_linear_block(block: &MetalQwen35BlockWeights) -> Result<MetalQwen35BlockWeights> {
    let attention = match &block.attention {
        MetalQwen35Attention::Linear(attn) => {
            MetalQwen35Attention::Linear(clone_linear_attn_weights(attn))
        }
        MetalQwen35Attention::Full(_) => anyhow::bail!("expected linear-attention block"),
    };
    Ok(MetalQwen35BlockWeights {
        input_layernorm: block.input_layernorm.clone(),
        attention,
        post_attention_layernorm: block.post_attention_layernorm.clone(),
        mlp: clone_dense_mlp_kind(&block.mlp)?,
    })
}

fn forward_linear_block_outputs(
    x: &MlxArray,
    block: &MetalQwen35BlockWeights,
    arch: &MetalQwen35ArchConfig,
    config: &MetalModelConfig,
) -> Result<(MlxArray, MlxArray)> {
    let MetalQwen35Attention::Linear(attn) = &block.attention else {
        anyhow::bail!("expected linear-attention block");
    };
    let mut recurrent = MetalRecurrentState::new(1, &arch.linear);
    let attn_out = fused_gdr_step(
        x,
        &block.input_layernorm,
        attn,
        &mut recurrent,
        0,
        &arch.linear,
        config,
    );
    let after_attn = add(x, &attn_out);
    let xn = rms_norm_last_dim(
        &after_attn,
        &block.post_attention_layernorm,
        config.rms_norm_eps as f32,
        config.norm_weight_mode.uses_offset(),
    );
    let mlp = mlp_forward(&block.mlp, &xn);
    let after_block = add(&after_attn, &mlp);
    Ok((after_attn, after_block))
}

fn print_linear_block_mix_diff(
    name: &str,
    x: &MlxArray,
    block: &MetalQwen35BlockWeights,
    st_after_attn: &MlxArray,
    st_after_block: &MlxArray,
    arch: &MetalQwen35ArchConfig,
    config: &MetalModelConfig,
) -> Result<()> {
    let (mix_after_attn, mix_after_block) = forward_linear_block_outputs(x, block, arch, config)?;
    print_tensor_diff(
        &format!("{name}.after_attn"),
        st_after_attn,
        &mix_after_attn,
    );
    print_tensor_diff(
        &format!("{name}.after_block"),
        st_after_block,
        &mix_after_block,
    );
    Ok(())
}

#[test]
fn qwen3_compiled_prefill_matches_rust_prefill_for_long_prompt() -> Result<()> {
    let Some(model_path) = qwen3_model_path() else {
        eprintln!(
            "QWEN3_MODEL_PATH unset and ../models/Qwen3-0.6B missing; skipping Qwen3 compiled prefill equivalence test"
        );
        return Ok(());
    };
    let _guard = metal_test_guard();

    let config = load_metal_config(&model_path)?;
    let weights = load_qwen3_metal_weights(&model_path, &config)?;
    let Some(cpp_model) = weights.cpp_model.as_ref() else {
        eprintln!(
            "Qwen3 compiled C++ model unavailable for {}; skipping equivalence test",
            model_path.display()
        );
        return Ok(());
    };

    let params = SamplingParams {
        temperature: 0.0,
        ..Default::default()
    };
    let prompt_tokens: Vec<u32> = (1..=64).collect();
    let prompt_tokens_i32: Vec<i32> = prompt_tokens.iter().map(|&token| token as i32).collect();
    let prompt_len = i32::try_from(prompt_tokens.len()).expect("prompt len fits in i32");
    let kv_capacity = prompt_len + KV_CACHE_CHUNK;
    let n_heads = config.num_attention_heads as i32;
    let n_kv_heads = config.num_key_value_heads as i32;
    let head_dim = config.head_dim as i32;
    let cache_shape = [1_i32, n_kv_heads, kv_capacity, head_dim];
    let kv_dtype = weights.layers[0].attention_inputs.kv_dtype();
    let init_caches = || {
        (
            (0..config.num_hidden_layers)
                .map(|_| zeros(&cache_shape, kv_dtype))
                .collect::<Vec<_>>(),
            (0..config.num_hidden_layers)
                .map(|_| zeros(&cache_shape, kv_dtype))
                .collect::<Vec<_>>(),
        )
    };

    let (mut rust_k, mut rust_v) = init_caches();
    let rust_sampled = build_forward_graph(
        &prompt_tokens,
        &weights,
        &mut rust_k,
        &mut rust_v,
        0,
        n_heads,
        n_kv_heads,
        head_dim,
        1.0f32 / (head_dim as f32).sqrt(),
        config.rope_theta as f32,
        config.rms_norm_eps as f32,
        None,
        0,
        &params,
    )?;
    let mut rust_refs: Vec<&MlxArray> = Vec::with_capacity(1 + rust_k.len() + rust_v.len());
    rust_refs.push(&rust_sampled);
    rust_refs.extend(rust_k.iter());
    rust_refs.extend(rust_v.iter());
    eval(&rust_refs);
    let rust_token = rust_sampled.item_i32();

    let (mut cpp_k, mut cpp_v) = init_caches();
    let prompt_arr = MlxArray::from_slice_i32(&prompt_tokens_i32, &[prompt_len]);
    let cpp_logits =
        cpp_model.prefill_full_attention(&prompt_arr, prompt_len, 0, &mut cpp_k, &mut cpp_v)?;
    let cpp_sampled = gpu_sample_token(&cpp_logits, &params);
    let mut cpp_refs: Vec<&MlxArray> = Vec::with_capacity(2 + cpp_k.len() + cpp_v.len());
    cpp_refs.push(&cpp_logits);
    cpp_refs.push(&cpp_sampled);
    cpp_refs.extend(cpp_k.iter());
    cpp_refs.extend(cpp_v.iter());
    eval(&cpp_refs);
    let cpp_token = cpp_sampled.item_i32();

    let (session_k, session_v) = init_caches();
    let mut session_kv_flat: Vec<MlxArray> = session_k
        .iter()
        .zip(session_v.iter())
        .flat_map(|(k, v)| [k.clone(), v.clone()])
        .collect();
    cpp_model.begin_session(&session_kv_flat, &[])?;
    let session_logits = cpp_model.prefill_session(&prompt_arr, prompt_len, 0)?;
    let session_sampled = gpu_sample_token(&session_logits, &params);
    let (session_kv_flat_out, session_gdr_flat_out) =
        cpp_model.end_session(session_kv_flat.len(), 0)?;
    anyhow::ensure!(
        session_gdr_flat_out.is_empty(),
        "Qwen3 full-attention session prefill unexpectedly returned GDR state"
    );
    session_kv_flat = session_kv_flat_out;
    let mut session_k = Vec::with_capacity(config.num_hidden_layers);
    let mut session_v = Vec::with_capacity(config.num_hidden_layers);
    let mut session_iter = session_kv_flat.into_iter();
    for _ in 0..config.num_hidden_layers {
        session_k.push(
            session_iter
                .next()
                .context("session prefill missing Qwen3 K cache")?,
        );
        session_v.push(
            session_iter
                .next()
                .context("session prefill missing Qwen3 V cache")?,
        );
    }
    anyhow::ensure!(
        session_iter.next().is_none(),
        "session prefill returned unexpected extra Qwen3 KV caches"
    );
    let mut session_refs: Vec<&MlxArray> =
        Vec::with_capacity(2 + session_k.len() + session_v.len());
    session_refs.push(&session_logits);
    session_refs.push(&session_sampled);
    session_refs.extend(session_k.iter());
    session_refs.extend(session_v.iter());
    eval(&session_refs);
    let session_token = session_sampled.item_i32();

    assert_eq!(rust_token, cpp_token);
    assert_eq!(cpp_token, session_token);

    for (
        layer_idx,
        (
            ((rust_k_layer, rust_v_layer), (cpp_k_layer, cpp_v_layer)),
            (session_k_layer, session_v_layer),
        ),
    ) in rust_k
        .iter()
        .zip(rust_v.iter())
        .zip(cpp_k.iter().zip(cpp_v.iter()))
        .zip(session_k.iter().zip(session_v.iter()))
        .enumerate()
    {
        let rust_k_valid = slice(
            rust_k_layer,
            &[0, 0, 0, 0],
            &[1, n_kv_heads, prompt_len, head_dim],
            &[1, 1, 1, 1],
        );
        let rust_v_valid = slice(
            rust_v_layer,
            &[0, 0, 0, 0],
            &[1, n_kv_heads, prompt_len, head_dim],
            &[1, 1, 1, 1],
        );
        let cpp_k_valid = slice(
            cpp_k_layer,
            &[0, 0, 0, 0],
            &[1, n_kv_heads, prompt_len, head_dim],
            &[1, 1, 1, 1],
        );
        let cpp_v_valid = slice(
            cpp_v_layer,
            &[0, 0, 0, 0],
            &[1, n_kv_heads, prompt_len, head_dim],
            &[1, 1, 1, 1],
        );
        let session_k_valid = slice(
            session_k_layer,
            &[0, 0, 0, 0],
            &[1, n_kv_heads, prompt_len, head_dim],
            &[1, 1, 1, 1],
        );
        let session_v_valid = slice(
            session_v_layer,
            &[0, 0, 0, 0],
            &[1, n_kv_heads, prompt_len, head_dim],
            &[1, 1, 1, 1],
        );

        let rust_k_f32 = as_dtype(&rust_k_valid, Dtype::Float32);
        let rust_v_f32 = as_dtype(&rust_v_valid, Dtype::Float32);
        let cpp_k_f32 = as_dtype(&cpp_k_valid, Dtype::Float32);
        let cpp_v_f32 = as_dtype(&cpp_v_valid, Dtype::Float32);
        let session_k_f32 = as_dtype(&session_k_valid, Dtype::Float32);
        let session_v_f32 = as_dtype(&session_v_valid, Dtype::Float32);
        eval(&[
            &rust_k_f32,
            &rust_v_f32,
            &cpp_k_f32,
            &cpp_v_f32,
            &session_k_f32,
            &session_v_f32,
        ]);

        for (idx, (lhs, rhs)) in rust_k_f32
            .as_slice_f32()
            .iter()
            .zip(cpp_k_f32.as_slice_f32().iter())
            .enumerate()
        {
            assert!(
                (lhs - rhs).abs() < 1e-3,
                "layer {layer_idx} k[{idx}] mismatch: rust={lhs} cpp={rhs}"
            );
        }
        for (idx, (lhs, rhs)) in cpp_k_f32
            .as_slice_f32()
            .iter()
            .zip(session_k_f32.as_slice_f32().iter())
            .enumerate()
        {
            assert!(
                (lhs - rhs).abs() < 1e-3,
                "layer {layer_idx} k[{idx}] mismatch: cpp={lhs} session={rhs}"
            );
        }
        for (idx, (lhs, rhs)) in rust_v_f32
            .as_slice_f32()
            .iter()
            .zip(cpp_v_f32.as_slice_f32().iter())
            .enumerate()
        {
            assert!(
                (lhs - rhs).abs() < 1e-3,
                "layer {layer_idx} v[{idx}] mismatch: rust={lhs} cpp={rhs}"
            );
        }
        for (idx, (lhs, rhs)) in cpp_v_f32
            .as_slice_f32()
            .iter()
            .zip(session_v_f32.as_slice_f32().iter())
            .enumerate()
        {
            assert!(
                (lhs - rhs).abs() < 1e-3,
                "layer {layer_idx} v[{idx}] mismatch: cpp={lhs} session={rhs}"
            );
        }
    }

    Ok(())
}

#[test]
fn load_conv1d_weight_transposes_pytorch_depthwise_layout() -> Result<()> {
    let _guard = metal_test_guard();

    let cfg = crate::backend::metal::gdr::MetalGdrConfig {
        num_key_heads: 1,
        key_dim: 1,
        num_value_heads: 1,
        value_dim: 2,
        conv_kernel: 4,
        hidden_size: 1,
        rms_norm_eps: 1e-6,
    };

    let raw = MlxArray::from_slice_f32(
        &[
            1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0, 1000.0, 2000.0,
            3000.0, 4000.0,
        ],
        &[4, 1, 4],
    );
    let weight = load_conv1d_weight(&raw, &cfg)?;
    let weight_f32 = as_dtype(&weight, Dtype::Float32);
    eval(&[&weight_f32]);

    assert_eq!(weight_f32.shape(), &[4, 4, 1]);
    assert_eq!(
        weight_f32.as_slice_f32(),
        &[
            1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0, 1000.0, 2000.0,
            3000.0, 4000.0,
        ]
    );

    let reshaped = reshape(&weight_f32, &[4, 4]);
    eval(&[&reshaped]);
    assert_eq!(
        reshaped.as_slice_f32(),
        &[
            1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0, 1000.0, 2000.0,
            3000.0, 4000.0,
        ]
    );

    Ok(())
}

#[test]
#[ignore = "debug helper for local Metal Qwen3.5 layer norm inspection"]
fn debug_qwen35_single_token_hidden_norms() -> Result<()> {
    let Some(model_path) = env::var_os("QWEN35_MODEL_PATH").map(PathBuf::from) else {
        eprintln!("QWEN35_MODEL_PATH unset; skipping debug_qwen35_single_token_hidden_norms");
        return Ok(());
    };
    let _guard = metal_test_guard();

    let config = load_metal_config(&model_path)?;
    let MetalModelArch::Qwen35(arch) = &config.arch else {
        anyhow::bail!("QWEN35_MODEL_PATH must point to a Qwen3.5 model");
    };
    let weights = load_qwen35_metal_weights(&model_path, &config)?;
    let num_full_layers = arch.num_full_attention_layers();
    let cache_shape = [
        1_i32,
        config.num_key_value_heads as i32,
        8,
        config.head_dim as i32,
    ];
    let mut k_caches: Vec<MlxArray> = (0..num_full_layers)
        .map(|_| zeros(&cache_shape, Dtype::Bfloat16))
        .collect();
    let mut v_caches: Vec<MlxArray> = (0..num_full_layers)
        .map(|_| zeros(&cache_shape, Dtype::Bfloat16))
        .collect();
    let mut recurrent = MetalRecurrentState::new(arch.num_linear_attention_layers(), &arch.linear);
    let target_layers: Vec<usize> = (0..config.num_hidden_layers).collect();
    let input_ids: Vec<u32> = env::var("QWEN35_DEBUG_IDS")
        .ok()
        .map(|raw| {
            raw.split(',')
                .filter_map(|part| {
                    let trimmed = part.trim();
                    (!trimmed.is_empty())
                        .then(|| trimmed.parse::<u32>().ok())
                        .flatten()
                })
                .collect()
        })
        .filter(|ids: &Vec<u32>| !ids.is_empty())
        .unwrap_or_else(|| vec![9419]);

    let (logits, hidden) = qwen35_forward_with_hidden_states(
        &input_ids,
        &weights,
        &config,
        arch,
        &mut k_caches,
        &mut v_caches,
        &mut recurrent,
        0,
        &target_layers,
    );
    let logits = as_dtype(&logits, Dtype::Float32);
    let hidden = as_dtype(&hidden, Dtype::Float32);
    eval(&[&logits, &hidden]);

    let sampled = gpu_sample_token(
        &logits,
        &crate::sampler::SamplingParams {
            temperature: 0.0,
            ..Default::default()
        },
    );
    eval(&[&sampled]);
    eprintln!(
        "input_ids={input_ids:?} sampled token={}",
        sampled.item_i32()
    );

    let hidden_slice = hidden.as_slice_f32();
    let row_stride = config.num_hidden_layers * config.hidden_size;
    let last_row_start = (input_ids.len() - 1) * row_stride;
    for layer_idx in 0..config.num_hidden_layers {
        let start = last_row_start + layer_idx * config.hidden_size;
        let end = start + config.hidden_size;
        let norm = hidden_slice[start..end]
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt();
        eprintln!("layer {layer_idx:02} norm={norm:.6}");
    }

    Ok(())
}

#[test]
#[ignore = "debug helper for comparing loaded Qwen3.5 safetensors vs GGUF tensors"]
fn compare_qwen35_0p8b_loaded_tensors_safetensors_vs_gguf() -> Result<()> {
    let Some(model_path) = qwen35_safetensors_model_path() else {
        eprintln!("QWEN35_MODEL_PATH unset and ../models/Qwen3.5-0.8B missing; skipping");
        return Ok(());
    };
    let Some(gguf_path) = qwen35_gguf_model_path() else {
        eprintln!(
            "QWEN35_GGUF_PATH unset and ../models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf missing; skipping"
        );
        return Ok(());
    };
    let _guard = metal_test_guard();

    let config = load_metal_config(&model_path)?;
    let MetalModelArch::Qwen35(arch) = &config.arch else {
        anyhow::bail!("QWEN35_MODEL_PATH must point to a Qwen3.5 model");
    };
    let gguf = GgufFile::open(gguf_path.to_str().context("invalid GGUF path")?)?;
    let safetensors = load_qwen35_metal_weights(&model_path, &config)?;
    let gguf_weights = load_qwen35_metal_weights_from_gguf(&gguf, &config)?;

    print_gguf_tensor_info(&gguf, "model.norm.weight")?;
    print_tensor_diff("final_norm", &safetensors.norm, &gguf_weights.norm);
    print_gguf_tensor_info(&gguf, "model.embed_tokens.weight")?;
    print_embed_row_diff(
        "embed_tokens[row0, :64]",
        &safetensors,
        &gguf_weights,
        0,
        64,
    );
    print_embed_row_diff(
        "embed_tokens[row9419, :64]",
        &safetensors,
        &gguf_weights,
        9419,
        64,
    );

    let full_idx = arch
        .layer_types
        .iter()
        .position(|layer| *layer == MetalQwen35LayerType::FullAttention)
        .context("missing full-attention layer")?;
    let linear_idx = arch
        .layer_types
        .iter()
        .position(|layer| *layer == MetalQwen35LayerType::LinearAttention)
        .context("missing linear-attention layer")?;

    let full_s = &safetensors.layers[full_idx];
    let full_g = &gguf_weights.layers[full_idx];
    print_gguf_tensor_info(
        &gguf,
        &format!("model.layers.{full_idx}.input_layernorm.weight"),
    )?;
    print_tensor_diff(
        &format!("layer{full_idx}.input_layernorm"),
        &full_s.input_layernorm,
        &full_g.input_layernorm,
    );
    if let (MetalQwen35Attention::Full(full_s), MetalQwen35Attention::Full(full_g)) =
        (&full_s.attention, &full_g.attention)
    {
        print_gguf_tensor_info(
            &gguf,
            &format!("model.layers.{full_idx}.self_attn.q_norm.weight"),
        )?;
        print_tensor_diff(
            &format!("layer{full_idx}.self_attn.q_norm"),
            &full_s.q_norm,
            &full_g.q_norm,
        );
        print_gguf_tensor_info(
            &gguf,
            &format!("model.layers.{full_idx}.self_attn.k_norm.weight"),
        )?;
        print_tensor_diff(
            &format!("layer{full_idx}.self_attn.k_norm"),
            &full_s.k_norm,
            &full_g.k_norm,
        );
        print_gguf_tensor_info(
            &gguf,
            &format!("model.layers.{full_idx}.self_attn.q_proj.weight"),
        )?;
        print_tensor_diff_with_transpose_hint(
            &format!("layer{full_idx}.self_attn.q_proj"),
            &dense_weight_f32(&full_s.q_proj)?,
            &dense_weight_f32(&full_g.q_proj)?,
        );
        print_gguf_tensor_info(
            &gguf,
            &format!("model.layers.{full_idx}.self_attn.k_proj.weight"),
        )?;
        print_tensor_diff_with_transpose_hint(
            &format!("layer{full_idx}.self_attn.k_proj"),
            &dense_weight_f32(&full_s.k_proj)?,
            &dense_weight_f32(&full_g.k_proj)?,
        );
        print_gguf_tensor_info(
            &gguf,
            &format!("model.layers.{full_idx}.self_attn.v_proj.weight"),
        )?;
        print_tensor_diff_with_transpose_hint(
            &format!("layer{full_idx}.self_attn.v_proj"),
            &dense_weight_f32(&full_s.v_proj)?,
            &dense_weight_f32(&full_g.v_proj)?,
        );
        print_gguf_tensor_info(
            &gguf,
            &format!("model.layers.{full_idx}.self_attn.o_proj.weight"),
        )?;
        print_tensor_diff_with_transpose_hint(
            &format!("layer{full_idx}.self_attn.o_proj"),
            &dense_weight_f32(&full_s.o_proj)?,
            &dense_weight_f32(&full_g.o_proj)?,
        );
    }

    let linear_s = &safetensors.layers[linear_idx];
    let linear_g = &gguf_weights.layers[linear_idx];
    print_gguf_tensor_info(
        &gguf,
        &format!("model.layers.{linear_idx}.input_layernorm.weight"),
    )?;
    print_tensor_diff(
        &format!("layer{linear_idx}.input_layernorm"),
        &linear_s.input_layernorm,
        &linear_g.input_layernorm,
    );
    if let (MetalQwen35Attention::Linear(linear_s), MetalQwen35Attention::Linear(linear_g)) =
        (&linear_s.attention, &linear_g.attention)
    {
        print_gguf_tensor_info(
            &gguf,
            &format!("model.layers.{linear_idx}.linear_attn.in_proj_qkv.weight"),
        )?;
        print_tensor_diff_with_transpose_hint(
            &format!("layer{linear_idx}.linear_attn.in_proj_qkv"),
            &dense_weight_f32(&linear_s.in_proj_qkv)?,
            &dense_weight_f32(&linear_g.in_proj_qkv)?,
        );
        print_gguf_tensor_info(
            &gguf,
            &format!("model.layers.{linear_idx}.linear_attn.in_proj_z.weight"),
        )?;
        print_tensor_diff_with_transpose_hint(
            &format!("layer{linear_idx}.linear_attn.in_proj_z"),
            &dense_weight_f32(&linear_s.in_proj_z)?,
            &dense_weight_f32(&linear_g.in_proj_z)?,
        );
        print_gguf_tensor_info(
            &gguf,
            &format!("model.layers.{linear_idx}.linear_attn.in_proj_b.weight"),
        )?;
        print_tensor_diff_with_transpose_hint(
            &format!("layer{linear_idx}.linear_attn.in_proj_b"),
            &dense_weight_f32(&linear_s.in_proj_b)?,
            &dense_weight_f32(&linear_g.in_proj_b)?,
        );
        print_gguf_tensor_info(
            &gguf,
            &format!("model.layers.{linear_idx}.linear_attn.in_proj_a.weight"),
        )?;
        print_tensor_diff_with_transpose_hint(
            &format!("layer{linear_idx}.linear_attn.in_proj_a"),
            &dense_weight_f32(&linear_s.in_proj_a)?,
            &dense_weight_f32(&linear_g.in_proj_a)?,
        );
        print_gguf_tensor_info(
            &gguf,
            &format!("model.layers.{linear_idx}.linear_attn.conv1d.weight"),
        )?;
        print_tensor_diff(
            &format!("layer{linear_idx}.linear_attn.conv1d_weight"),
            &linear_s.conv1d_weight,
            &linear_g.conv1d_weight,
        );
        print_gguf_tensor_info(
            &gguf,
            &format!("model.layers.{linear_idx}.linear_attn.dt_bias"),
        )?;
        print_tensor_diff(
            &format!("layer{linear_idx}.linear_attn.dt_bias"),
            &linear_s.dt_bias,
            &linear_g.dt_bias,
        );
        print_gguf_tensor_info(
            &gguf,
            &format!("model.layers.{linear_idx}.linear_attn.a_log"),
        )?;
        print_tensor_diff(
            &format!("layer{linear_idx}.linear_attn.a_log"),
            &linear_s.a_log,
            &linear_g.a_log,
        );
        print_gguf_tensor_info(
            &gguf,
            &format!("model.layers.{linear_idx}.linear_attn.norm.weight"),
        )?;
        print_tensor_diff(
            &format!("layer{linear_idx}.linear_attn.norm_weight"),
            &linear_s.norm_weight,
            &linear_g.norm_weight,
        );
        print_gguf_tensor_info(
            &gguf,
            &format!("model.layers.{linear_idx}.linear_attn.out_proj.weight"),
        )?;
        print_tensor_diff_with_transpose_hint(
            &format!("layer{linear_idx}.linear_attn.out_proj"),
            &dense_weight_f32(&linear_s.out_proj)?,
            &dense_weight_f32(&linear_g.out_proj)?,
        );
    }

    if let (MlpKind::Dense(mlp_s), MlpKind::Dense(mlp_g)) = (&linear_s.mlp, &linear_g.mlp) {
        print_gguf_tensor_info(
            &gguf,
            &format!("model.layers.{linear_idx}.mlp.gate_proj.weight"),
        )?;
        print_tensor_diff_with_transpose_hint(
            &format!("layer{linear_idx}.mlp.gate_proj"),
            &dense_weight_f32(&mlp_s.gate_proj)?,
            &dense_weight_f32(&mlp_g.gate_proj)?,
        );
        print_gguf_tensor_info(
            &gguf,
            &format!("model.layers.{linear_idx}.mlp.up_proj.weight"),
        )?;
        print_tensor_diff_with_transpose_hint(
            &format!("layer{linear_idx}.mlp.up_proj"),
            &dense_weight_f32(&mlp_s.up_proj)?,
            &dense_weight_f32(&mlp_g.up_proj)?,
        );
        print_gguf_tensor_info(
            &gguf,
            &format!("model.layers.{linear_idx}.mlp.down_proj.weight"),
        )?;
        print_tensor_diff_with_transpose_hint(
            &format!("layer{linear_idx}.mlp.down_proj"),
            &dense_weight_f32(&mlp_s.down_proj)?,
            &dense_weight_f32(&mlp_g.down_proj)?,
        );
    }

    Ok(())
}

#[test]
#[ignore = "debug helper for comparing Qwen3.5 safetensors vs GGUF first-token logits"]
fn compare_qwen35_0p8b_first_token_logits_safetensors_vs_gguf() -> Result<()> {
    let Some(model_path) = qwen35_safetensors_model_path() else {
        eprintln!("QWEN35_MODEL_PATH unset and ../models/Qwen3.5-0.8B missing; skipping");
        return Ok(());
    };
    let Some(gguf_path) = qwen35_gguf_model_path() else {
        eprintln!(
            "QWEN35_GGUF_PATH unset and ../models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf missing; skipping"
        );
        return Ok(());
    };
    let _guard = metal_test_guard();

    let config = load_metal_config(&model_path)?;
    let MetalModelArch::Qwen35(arch) = &config.arch else {
        anyhow::bail!("QWEN35_MODEL_PATH must point to a Qwen3.5 model");
    };
    let tokenizer = Tokenizer::from_file(model_path.to_str().context("invalid model path")?)?;
    let input_ids = tokenizer.encode("Hello")?;
    eprintln!("input_ids={input_ids:?}");

    let gguf = GgufFile::open(gguf_path.to_str().context("invalid GGUF path")?)?;
    let safetensors = load_qwen35_metal_weights(&model_path, &config)?;
    let gguf_weights = load_qwen35_metal_weights_from_gguf(&gguf, &config)?;

    let num_full_layers = arch.num_full_attention_layers();
    let cache_shape = [
        1_i32,
        config.num_key_value_heads as i32,
        input_ids.len() as i32 + 8,
        config.head_dim as i32,
    ];
    let init_kv = || {
        (
            (0..num_full_layers)
                .map(|_| zeros(&cache_shape, Dtype::Bfloat16))
                .collect::<Vec<_>>(),
            (0..num_full_layers)
                .map(|_| zeros(&cache_shape, Dtype::Bfloat16))
                .collect::<Vec<_>>(),
        )
    };

    let (mut st_k, mut st_v) = init_kv();
    let (mut gg_k, mut gg_v) = init_kv();
    let mut st_recurrent =
        MetalRecurrentState::new(arch.num_linear_attention_layers(), &arch.linear);
    let mut gg_recurrent =
        MetalRecurrentState::new(arch.num_linear_attention_layers(), &arch.linear);

    let target_layer = config.num_hidden_layers - 1;
    let (st_logits, st_hidden) = qwen35_forward_with_hidden_states(
        &input_ids,
        &safetensors,
        &config,
        arch,
        &mut st_k,
        &mut st_v,
        &mut st_recurrent,
        0,
        &[target_layer],
    );
    let (gg_logits, gg_hidden) = qwen35_forward_with_hidden_states(
        &input_ids,
        &gguf_weights,
        &config,
        arch,
        &mut gg_k,
        &mut gg_v,
        &mut gg_recurrent,
        0,
        &[target_layer],
    );

    print_tensor_diff("final_hidden", &st_hidden, &gg_hidden);

    let st_final_norm = rms_norm_last_dim(
        &st_hidden,
        &safetensors.norm,
        config.rms_norm_eps as f32,
        config.norm_weight_mode.uses_offset(),
    );
    let gg_final_norm = rms_norm_last_dim(
        &gg_hidden,
        &gguf_weights.norm,
        config.rms_norm_eps as f32,
        config.norm_weight_mode.uses_offset(),
    );
    let st_hidden_st_lm = linear(&st_final_norm, &safetensors.lm_head);
    let st_hidden_gg_lm = linear(&st_final_norm, &gguf_weights.lm_head);
    let gg_hidden_st_lm = linear(&gg_final_norm, &safetensors.lm_head);
    let gg_hidden_gg_lm = linear(&gg_final_norm, &gguf_weights.lm_head);

    let st_top = topk_logits(&st_logits, 8);
    let gg_top = topk_logits(&gg_logits, 8);
    let st_hidden_st_lm_top = topk_logits(&st_hidden_st_lm, 4);
    let st_hidden_gg_lm_top = topk_logits(&st_hidden_gg_lm, 4);
    let gg_hidden_st_lm_top = topk_logits(&gg_hidden_st_lm, 4);
    let gg_hidden_gg_lm_top = topk_logits(&gg_hidden_gg_lm, 4);
    let decode = |id: usize| {
        tokenizer
            .decode(&[id as u32])
            .unwrap_or_else(|_| "<decode err>".into())
    };

    eprintln!("safetensors_top8:");
    for (id, logit) in &st_top {
        eprintln!("  id={id} logit={logit:.6} text={:?}", decode(*id));
    }
    eprintln!("gguf_top8:");
    for (id, logit) in &gg_top {
        eprintln!("  id={id} logit={logit:.6} text={:?}", decode(*id));
    }
    eprintln!("cross_top4 st_hidden + st_lm_head:");
    for (id, logit) in &st_hidden_st_lm_top {
        eprintln!("  id={id} logit={logit:.6} text={:?}", decode(*id));
    }
    eprintln!("cross_top4 st_hidden + gg_lm_head:");
    for (id, logit) in &st_hidden_gg_lm_top {
        eprintln!("  id={id} logit={logit:.6} text={:?}", decode(*id));
    }
    eprintln!("cross_top4 gg_hidden + st_lm_head:");
    for (id, logit) in &gg_hidden_st_lm_top {
        eprintln!("  id={id} logit={logit:.6} text={:?}", decode(*id));
    }
    eprintln!("cross_top4 gg_hidden + gg_lm_head:");
    for (id, logit) in &gg_hidden_gg_lm_top {
        eprintln!("  id={id} logit={logit:.6} text={:?}", decode(*id));
    }

    let st_sampled = gpu_sample_token(
        &st_logits,
        &crate::sampler::SamplingParams {
            temperature: 0.0,
            ..Default::default()
        },
    );
    let gg_sampled = gpu_sample_token(
        &gg_logits,
        &crate::sampler::SamplingParams {
            temperature: 0.0,
            ..Default::default()
        },
    );
    eval(&[&st_sampled, &gg_sampled]);
    let st_id = st_sampled.item_i32() as usize;
    let gg_id = gg_sampled.item_i32() as usize;
    eprintln!(
        "sampled: safetensors id={st_id} text={:?} | gguf id={gg_id} text={:?}",
        decode(st_id),
        decode(gg_id),
    );

    Ok(())
}

#[test]
#[ignore = "debug helper for comparing Qwen3.5 safetensors vs GGUF layer-wise hidden states"]
fn compare_qwen35_0p8b_layerwise_hidden_safetensors_vs_gguf() -> Result<()> {
    let Some(model_path) = qwen35_safetensors_model_path() else {
        eprintln!("QWEN35_MODEL_PATH unset and ../models/Qwen3.5-0.8B missing; skipping");
        return Ok(());
    };
    let Some(gguf_path) = qwen35_gguf_model_path() else {
        eprintln!(
            "QWEN35_GGUF_PATH unset and ../models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf missing; skipping"
        );
        return Ok(());
    };
    let _guard = metal_test_guard();

    let config = load_metal_config(&model_path)?;
    let MetalModelArch::Qwen35(arch) = &config.arch else {
        anyhow::bail!("QWEN35_MODEL_PATH must point to a Qwen3.5 model");
    };
    let tokenizer = Tokenizer::from_file(model_path.to_str().context("invalid model path")?)?;
    let input_ids = tokenizer.encode("Hello")?;
    eprintln!("input_ids={input_ids:?}");

    let gguf = GgufFile::open(gguf_path.to_str().context("invalid GGUF path")?)?;
    let safetensors = load_qwen35_metal_weights(&model_path, &config)?;
    let gguf_weights = load_qwen35_metal_weights_from_gguf(&gguf, &config)?;

    let num_full_layers = arch.num_full_attention_layers();
    let cache_shape = [
        1_i32,
        config.num_key_value_heads as i32,
        input_ids.len() as i32 + 8,
        config.head_dim as i32,
    ];
    let init_kv = || {
        (
            (0..num_full_layers)
                .map(|_| zeros(&cache_shape, Dtype::Bfloat16))
                .collect::<Vec<_>>(),
            (0..num_full_layers)
                .map(|_| zeros(&cache_shape, Dtype::Bfloat16))
                .collect::<Vec<_>>(),
        )
    };
    let target_layers: Vec<usize> = (0..config.num_hidden_layers).collect();

    let (mut st_k, mut st_v) = init_kv();
    let (mut gg_k, mut gg_v) = init_kv();
    let mut st_recurrent =
        MetalRecurrentState::new(arch.num_linear_attention_layers(), &arch.linear);
    let mut gg_recurrent =
        MetalRecurrentState::new(arch.num_linear_attention_layers(), &arch.linear);

    let (_, st_hidden) = qwen35_forward_with_hidden_states(
        &input_ids,
        &safetensors,
        &config,
        arch,
        &mut st_k,
        &mut st_v,
        &mut st_recurrent,
        0,
        &target_layers,
    );
    let (_, gg_hidden) = qwen35_forward_with_hidden_states(
        &input_ids,
        &gguf_weights,
        &config,
        arch,
        &mut gg_k,
        &mut gg_v,
        &mut gg_recurrent,
        0,
        &target_layers,
    );

    let st_hidden = as_dtype(&st_hidden, Dtype::Float32);
    let gg_hidden = as_dtype(&gg_hidden, Dtype::Float32);
    eval(&[&st_hidden, &gg_hidden]);
    let st_slice = st_hidden.as_slice_f32();
    let gg_slice = gg_hidden.as_slice_f32();
    let hidden_size = config.hidden_size;
    for layer_idx in 0..config.num_hidden_layers {
        let start = layer_idx * hidden_size;
        let end = start + hidden_size;
        let st_layer = MlxArray::from_slice_f32(&st_slice[start..end], &[1, hidden_size as i32]);
        let gg_layer = MlxArray::from_slice_f32(&gg_slice[start..end], &[1, hidden_size as i32]);
        print_tensor_diff(&format!("layer{layer_idx}.hidden"), &st_layer, &gg_layer);
    }

    Ok(())
}

fn qwen35_replay_logits(
    input_ids: &[u32],
    weights: &Qwen35MetalWeights,
    config: &MetalModelConfig,
    arch: &MetalQwen35ArchConfig,
) -> Result<MlxArray> {
    let num_full_layers = arch.num_full_attention_layers();
    let kv_capacity = input_ids.len() as i32 + 8;
    let cache_shape = [
        1_i32,
        config.num_key_value_heads as i32,
        kv_capacity,
        config.head_dim as i32,
    ];
    let mut k_caches: Vec<MlxArray> = (0..num_full_layers)
        .map(|_| zeros(&cache_shape, Dtype::Bfloat16))
        .collect();
    let mut v_caches: Vec<MlxArray> = (0..num_full_layers)
        .map(|_| zeros(&cache_shape, Dtype::Bfloat16))
        .collect();
    let mut recurrent = MetalRecurrentState::new(arch.num_linear_attention_layers(), &arch.linear);

    let mut logits = None;
    for (cache_len, &token) in (0_i32..).zip(input_ids.iter()) {
        let token_arr = MlxArray::from_slice_i32(&[token as i32], &[1]);
        logits = Some(qwen35_forward_step(
            &token_arr,
            weights,
            config,
            arch,
            &mut k_caches,
            &mut v_caches,
            &mut recurrent,
            cache_len,
        ));
    }

    logits.context("expected replay sequence to produce logits")
}

fn fresh_cpp_qwen35_flat_state(
    config: &MetalModelConfig,
    arch: &MetalQwen35ArchConfig,
    kv_capacity: i32,
) -> (Vec<MlxArray>, Vec<MlxArray>) {
    let cache_shape = [
        1_i32,
        config.num_key_value_heads as i32,
        kv_capacity,
        config.head_dim as i32,
    ];
    let kv_flat: Vec<MlxArray> = (0..arch.num_full_attention_layers())
        .flat_map(|_| {
            [
                zeros(&cache_shape, Dtype::Bfloat16),
                zeros(&cache_shape, Dtype::Bfloat16),
            ]
        })
        .collect();
    let recurrent = MetalRecurrentState::new(arch.num_linear_attention_layers(), &arch.linear);
    let gdr_flat = recurrent
        .states
        .iter()
        .zip(recurrent.conv_states.iter())
        .flat_map(|(state, conv)| [state.clone(), conv.clone()])
        .collect();
    (kv_flat, gdr_flat)
}

fn greedy_token_id(logits: &MlxArray) -> u32 {
    let sampled = gpu_sample_token(
        logits,
        &crate::sampler::SamplingParams {
            temperature: 0.0,
            ..Default::default()
        },
    );
    eval(&[&sampled]);
    sampled.item_i32() as u32
}

#[test]
#[ignore = "debug helper for comparing Qwen3.5 GGUF incremental decode vs full replay"]
fn compare_qwen35_0p8b_gguf_incremental_decode_vs_full_replay() -> Result<()> {
    let Some(model_path) = qwen35_safetensors_model_path() else {
        eprintln!("QWEN35_MODEL_PATH unset and ../models/Qwen3.5-0.8B missing; skipping");
        return Ok(());
    };
    let Some(gguf_path) = qwen35_gguf_model_path() else {
        eprintln!(
            "QWEN35_GGUF_PATH unset and ../models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf missing; skipping"
        );
        return Ok(());
    };
    let _guard = metal_test_guard();

    let config = load_metal_config(&model_path)?;
    let MetalModelArch::Qwen35(arch) = &config.arch else {
        anyhow::bail!("QWEN35_MODEL_PATH must point to a Qwen3.5 model");
    };
    let tokenizer = Tokenizer::from_file(model_path.to_str().context("invalid model path")?)?;
    let gguf = GgufFile::open(gguf_path.to_str().context("invalid GGUF path")?)?;
    let weights = load_qwen35_metal_weights_from_gguf(&gguf, &config)?;

    let mut sequence = tokenizer.encode("Hello")?;
    anyhow::ensure!(!sequence.is_empty(), "expected non-empty prompt");

    let num_full_layers = arch.num_full_attention_layers();
    let kv_capacity = sequence.len() as i32 + 16;
    let cache_shape = [
        1_i32,
        config.num_key_value_heads as i32,
        kv_capacity,
        config.head_dim as i32,
    ];
    let mut k_caches: Vec<MlxArray> = (0..num_full_layers)
        .map(|_| zeros(&cache_shape, Dtype::Bfloat16))
        .collect();
    let mut v_caches: Vec<MlxArray> = (0..num_full_layers)
        .map(|_| zeros(&cache_shape, Dtype::Bfloat16))
        .collect();
    let mut recurrent = MetalRecurrentState::new(arch.num_linear_attention_layers(), &arch.linear);

    let mut incremental_logits = qwen35_replay_logits(&sequence, &weights, &config, arch)?;
    for (cache_len, &token) in (0_i32..).zip(sequence.iter()) {
        let token_arr = MlxArray::from_slice_i32(&[token as i32], &[1]);
        incremental_logits = qwen35_forward_step(
            &token_arr,
            &weights,
            &config,
            arch,
            &mut k_caches,
            &mut v_caches,
            &mut recurrent,
            cache_len,
        );
    }

    for step_idx in 0..4 {
        let replay_logits = qwen35_replay_logits(&sequence, &weights, &config, arch)?;
        let incremental_token = greedy_token_id(&incremental_logits);
        let replay_token = greedy_token_id(&replay_logits);
        eprintln!(
            "step {step_idx}: incremental id={incremental_token} text={:?} | replay id={replay_token} text={:?}",
            tokenizer.decode(&[incremental_token])?,
            tokenizer.decode(&[replay_token])?,
        );
        anyhow::ensure!(
            incremental_token == replay_token,
            "GGUF incremental decode diverged from full replay at step {step_idx}: incremental={incremental_token} replay={replay_token}"
        );

        sequence.push(incremental_token);
        let token_arr = MlxArray::from_slice_i32(&[incremental_token as i32], &[1]);
        incremental_logits = qwen35_forward_step(
            &token_arr,
            &weights,
            &config,
            arch,
            &mut k_caches,
            &mut v_caches,
            &mut recurrent,
            (sequence.len() - 1) as i32,
        );
    }

    Ok(())
}

#[test]
#[ignore = "requires local Qwen3.5-0.8B safetensors config plus Q4_K_M GGUF weights"]
fn compare_qwen35_0p8b_gguf_cpp_generate_vs_rust_replay() -> Result<()> {
    let Some(model_path) = qwen35_safetensors_model_path() else {
        eprintln!("QWEN35_MODEL_PATH unset and ../models/Qwen3.5-0.8B missing; skipping");
        return Ok(());
    };
    let Some(gguf_path) = qwen35_gguf_model_path() else {
        eprintln!(
            "QWEN35_GGUF_PATH unset and ../models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf missing; skipping"
        );
        return Ok(());
    };
    let _guard = metal_test_guard();

    let config = load_metal_config(&model_path)?;
    let MetalModelArch::Qwen35(arch) = &config.arch else {
        anyhow::bail!("QWEN35_MODEL_PATH must point to a Qwen3.5 model");
    };
    let tokenizer = Tokenizer::from_file(model_path.to_str().context("invalid model path")?)?;
    let gguf = GgufFile::open(gguf_path.to_str().context("invalid GGUF path")?)?;
    let weights = load_qwen35_metal_weights_from_gguf(&gguf, &config)?;
    anyhow::ensure!(
        weights.cpp_model.is_some(),
        "GGUF Qwen3.5 should build the C++ model unless METAL_NO_CPP is set"
    );
    anyhow::ensure!(
        !qwen35_dflash_supported(&weights),
        "GGUF Qwen3.5 C++ path must not advertise DFlash support"
    );

    let prompt_ids = tokenizer.encode("Hello world from the GGUF C++ path.")?;
    anyhow::ensure!(
        prompt_ids.len() > 1,
        "expected a multi-token prompt to exercise C++ GDR fallback prefill"
    );

    let max_new_tokens = 4;
    let mut replay_sequence = prompt_ids.clone();
    let mut expected = Vec::with_capacity(max_new_tokens);
    for step_idx in 0..max_new_tokens {
        let replay_logits = qwen35_replay_logits(&replay_sequence, &weights, &config, arch)?;
        let token = greedy_token_id(&replay_logits);
        eprintln!(
            "replay step {step_idx}: id={token} text={:?}",
            tokenizer.decode(&[token])?,
        );
        expected.push(token);
        replay_sequence.push(token);
    }

    let mut streamed = Vec::new();
    let generated = metal_generate_qwen35(
        &prompt_ids,
        &weights,
        &config,
        None,
        &SamplingParams {
            temperature: 0.7,
            top_k: 1,
            ignore_eos: true,
            ..Default::default()
        },
        max_new_tokens,
        Instant::now(),
        &mut |token| {
            streamed.push(token);
            Ok(())
        },
    )?;

    anyhow::ensure!(
        generated.tokens == expected,
        "GGUF C++ generate diverged from Rust replay: generated={:?} expected={:?}",
        generated.tokens,
        expected
    );
    anyhow::ensure!(
        streamed == expected,
        "GGUF C++ callback tokens diverged from Rust replay: streamed={streamed:?} expected={expected:?}"
    );

    Ok(())
}

#[test]
#[ignore = "requires local Qwen3.5-0.8B safetensors config plus Q4_K_M GGUF weights"]
fn compare_qwen35_0p8b_gguf_cpp_gdr_kernel_vs_fallback_prefill() -> Result<()> {
    let Some(model_path) = qwen35_safetensors_model_path() else {
        eprintln!("QWEN35_MODEL_PATH unset and ../models/Qwen3.5-0.8B missing; skipping");
        return Ok(());
    };
    let Some(gguf_path) = qwen35_gguf_model_path() else {
        eprintln!(
            "QWEN35_GGUF_PATH unset and ../models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf missing; skipping"
        );
        return Ok(());
    };
    let _guard = metal_test_guard();

    let config = load_metal_config(&model_path)?;
    let MetalModelArch::Qwen35(arch) = &config.arch else {
        anyhow::bail!("QWEN35_MODEL_PATH must point to a Qwen3.5 model");
    };
    let tokenizer = Tokenizer::from_file(model_path.to_str().context("invalid model path")?)?;
    let gguf = GgufFile::open(gguf_path.to_str().context("invalid GGUF path")?)?;
    let weights = load_qwen35_metal_weights_from_gguf(&gguf, &config)?;
    let cpp_model = weights
        .cpp_model
        .as_ref()
        .context("GGUF Qwen3.5 C++ model unavailable")?;

    let prompt_ids = tokenizer.encode("Hello world from the GGUF C++ path.")?;
    anyhow::ensure!(prompt_ids.len() > 1, "expected multi-token prompt");
    let prompt_tokens: Vec<i32> = prompt_ids.iter().map(|&id| id as i32).collect();
    let prompt_len = prompt_tokens.len() as i32;
    let prompt_arr = MlxArray::from_slice_i32(&prompt_tokens, &[prompt_len]);
    let kv_capacity = prompt_len + 8;

    let (mut fallback_kv, mut fallback_gdr) =
        fresh_cpp_qwen35_flat_state(&config, arch, kv_capacity);
    unsafe { mlx_sys::qwen35_compiled_set_gdr_metal_kernel_enabled(cpp_model.as_raw(), 0) };
    let fallback_logits = cpp_model.prefill(
        &prompt_arr,
        prompt_len,
        0,
        &mut fallback_kv,
        &mut fallback_gdr,
    )?;

    let (mut kernel_kv, mut kernel_gdr) = fresh_cpp_qwen35_flat_state(&config, arch, kv_capacity);
    unsafe { mlx_sys::qwen35_compiled_set_gdr_metal_kernel_enabled(cpp_model.as_raw(), 1) };
    let kernel_logits =
        cpp_model.prefill(&prompt_arr, prompt_len, 0, &mut kernel_kv, &mut kernel_gdr)?;
    unsafe { mlx_sys::qwen35_compiled_set_gdr_metal_kernel_enabled(cpp_model.as_raw(), 0) };

    let (max_logit_diff, _, _, _) = tensor_diff_stats(&kernel_logits, &fallback_logits);
    let mut max_kv_diff = 0.0_f32;
    for (kernel, fallback) in kernel_kv.iter().zip(fallback_kv.iter()) {
        let (max_abs, _, _, _) = tensor_diff_stats(kernel, fallback);
        max_kv_diff = max_kv_diff.max(max_abs);
    }
    let mut max_gdr_diff = 0.0_f32;
    for (kernel, fallback) in kernel_gdr.iter().zip(fallback_gdr.iter()) {
        let (max_abs, _, _, _) = tensor_diff_stats(kernel, fallback);
        max_gdr_diff = max_gdr_diff.max(max_abs);
    }
    eprintln!(
        "GGUF C++ GDR kernel vs fallback prefill: max_logit={max_logit_diff:.6} max_kv={max_kv_diff:.6} max_gdr={max_gdr_diff:.6}"
    );
    anyhow::ensure!(
        max_logit_diff < 5e-2,
        "GGUF C++ custom GDR prefill logits diverged from fallback by {max_logit_diff}"
    );
    anyhow::ensure!(
        max_kv_diff < 1e-1,
        "GGUF C++ custom GDR prefill KV cache diverged from fallback by {max_kv_diff}"
    );
    anyhow::ensure!(
        max_gdr_diff < 5e-2,
        "GGUF C++ custom GDR prefill state diverged from fallback by {max_gdr_diff}"
    );

    Ok(())
}

#[test]
#[ignore = "debug helper for isolating the first divergent Qwen3.5 GGUF layer0 subgroup"]
fn compare_qwen35_0p8b_layer0_linear_subgroup_ablation() -> Result<()> {
    let Some(model_path) = qwen35_safetensors_model_path() else {
        eprintln!("QWEN35_MODEL_PATH unset and ../models/Qwen3.5-0.8B missing; skipping");
        return Ok(());
    };
    let Some(gguf_path) = qwen35_gguf_model_path() else {
        eprintln!(
            "QWEN35_GGUF_PATH unset and ../models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf missing; skipping"
        );
        return Ok(());
    };
    let _guard = metal_test_guard();

    let config = load_metal_config(&model_path)?;
    let MetalModelArch::Qwen35(arch) = &config.arch else {
        anyhow::bail!("QWEN35_MODEL_PATH must point to a Qwen3.5 model");
    };
    anyhow::ensure!(
        arch.layer_types.first() == Some(&MetalQwen35LayerType::LinearAttention),
        "expected Qwen3.5-0.8B layer0 to be linear attention"
    );

    let tokenizer = Tokenizer::from_file(model_path.to_str().context("invalid model path")?)?;
    let input_ids = tokenizer.encode("Hello")?;
    anyhow::ensure!(
        !input_ids.is_empty(),
        "expected non-empty tokenized prompt for layer0 ablation"
    );
    let token = MlxArray::from_slice_i32(&[input_ids[0] as i32], &[1]);

    let gguf = GgufFile::open(gguf_path.to_str().context("invalid GGUF path")?)?;
    let safetensors = load_qwen35_metal_weights(&model_path, &config)?;
    let gguf_weights = load_qwen35_metal_weights_from_gguf(&gguf, &config)?;

    let x = qwen35_embed_tokens(&safetensors, &token);
    let st_block = clone_linear_block(&safetensors.layers[0])?;
    let gg_block = clone_linear_block(&gguf_weights.layers[0])?;

    let (st_after_attn, st_after_block) =
        forward_linear_block_outputs(&x, &st_block, arch, &config)?;
    let (gg_after_attn, gg_after_block) =
        forward_linear_block_outputs(&x, &gg_block, arch, &config)?;
    print_tensor_diff("layer0.gguf.after_attn", &st_after_attn, &gg_after_attn);
    print_tensor_diff("layer0.gguf.after_block", &st_after_block, &gg_after_block);

    let (
        MetalQwen35Attention::Linear(st_attn),
        MetalQwen35Attention::Linear(gg_attn),
        MlpKind::Dense(st_mlp),
        MlpKind::Dense(gg_mlp),
    ) = (
        &st_block.attention,
        &gg_block.attention,
        &st_block.mlp,
        &gg_block.mlp,
    )
    else {
        anyhow::bail!("expected dense linear-attention layer0");
    };

    {
        let mut mix = clone_linear_block(&st_block)?;
        mix.input_layernorm = gg_block.input_layernorm.clone();
        print_linear_block_mix_diff(
            "layer0.swap_input_layernorm",
            &x,
            &mix,
            &st_after_attn,
            &st_after_block,
            arch,
            &config,
        )?;
    }
    {
        let mut mix = clone_linear_block(&st_block)?;
        if let MetalQwen35Attention::Linear(attn) = &mut mix.attention {
            attn.in_proj_qkvz = gg_attn.in_proj_qkvz.clone();
        }
        print_linear_block_mix_diff(
            "layer0.swap_in_proj_qkvz",
            &x,
            &mix,
            &st_after_attn,
            &st_after_block,
            arch,
            &config,
        )?;
    }
    {
        let mut mix = clone_linear_block(&st_block)?;
        if let MetalQwen35Attention::Linear(attn) = &mut mix.attention {
            attn.in_proj_qkv = gg_attn.in_proj_qkv.clone();
            attn.in_proj_qkvz = Some(concat_dense_weights(
                &gg_attn.in_proj_qkv,
                &st_attn.in_proj_z,
            )?);
        }
        print_linear_block_mix_diff(
            "layer0.swap_in_proj_qkv_only",
            &x,
            &mix,
            &st_after_attn,
            &st_after_block,
            arch,
            &config,
        )?;
    }
    {
        let mut mix = clone_linear_block(&st_block)?;
        if let MetalQwen35Attention::Linear(attn) = &mut mix.attention {
            attn.in_proj_z = gg_attn.in_proj_z.clone();
            attn.in_proj_qkvz = Some(concat_dense_weights(
                &st_attn.in_proj_qkv,
                &gg_attn.in_proj_z,
            )?);
        }
        print_linear_block_mix_diff(
            "layer0.swap_in_proj_z_only",
            &x,
            &mix,
            &st_after_attn,
            &st_after_block,
            arch,
            &config,
        )?;
    }
    {
        let mut mix = clone_linear_block(&st_block)?;
        if let MetalQwen35Attention::Linear(attn) = &mut mix.attention {
            attn.in_proj_ba = gg_attn.in_proj_ba.clone();
        }
        print_linear_block_mix_diff(
            "layer0.swap_in_proj_ba",
            &x,
            &mix,
            &st_after_attn,
            &st_after_block,
            arch,
            &config,
        )?;
    }
    {
        let mut mix = clone_linear_block(&st_block)?;
        if let MetalQwen35Attention::Linear(attn) = &mut mix.attention {
            attn.conv1d_weight = gg_attn.conv1d_weight.clone();
        }
        print_linear_block_mix_diff(
            "layer0.swap_conv1d",
            &x,
            &mix,
            &st_after_attn,
            &st_after_block,
            arch,
            &config,
        )?;
    }
    {
        let mut mix = clone_linear_block(&st_block)?;
        if let MetalQwen35Attention::Linear(attn) = &mut mix.attention {
            attn.dt_bias = gg_attn.dt_bias.clone();
            attn.a_log = gg_attn.a_log.clone();
        }
        print_linear_block_mix_diff(
            "layer0.swap_gate_params",
            &x,
            &mix,
            &st_after_attn,
            &st_after_block,
            arch,
            &config,
        )?;
    }
    {
        let mut mix = clone_linear_block(&st_block)?;
        if let MetalQwen35Attention::Linear(attn) = &mut mix.attention {
            attn.norm_weight = gg_attn.norm_weight.clone();
        }
        print_linear_block_mix_diff(
            "layer0.swap_linear_norm",
            &x,
            &mix,
            &st_after_attn,
            &st_after_block,
            arch,
            &config,
        )?;
    }
    {
        let mut mix = clone_linear_block(&st_block)?;
        if let MetalQwen35Attention::Linear(attn) = &mut mix.attention {
            attn.out_proj = gg_attn.out_proj.clone();
        }
        print_linear_block_mix_diff(
            "layer0.swap_out_proj",
            &x,
            &mix,
            &st_after_attn,
            &st_after_block,
            arch,
            &config,
        )?;
    }
    {
        let mut mix = clone_linear_block(&st_block)?;
        mix.post_attention_layernorm = gg_block.post_attention_layernorm.clone();
        print_linear_block_mix_diff(
            "layer0.swap_post_attention_layernorm",
            &x,
            &mix,
            &st_after_attn,
            &st_after_block,
            arch,
            &config,
        )?;
    }
    {
        let mut mix = clone_linear_block(&st_block)?;
        mix.mlp = MlpKind::Dense(clone_dense_mlp_weights(gg_mlp));
        print_linear_block_mix_diff(
            "layer0.swap_mlp_all",
            &x,
            &mix,
            &st_after_attn,
            &st_after_block,
            arch,
            &config,
        )?;
    }
    {
        let mut mix = clone_linear_block(&st_block)?;
        mix.mlp = MlpKind::Dense(MetalQwen35DenseMlpWeights {
            inputs: MlpInputProjection::Split {
                gate_proj: gg_mlp.gate_proj.clone(),
                up_proj: gg_mlp.up_proj.clone(),
            },
            down_proj: st_mlp.down_proj.clone(),
            gate_proj: gg_mlp.gate_proj.clone(),
            up_proj: gg_mlp.up_proj.clone(),
        });
        print_linear_block_mix_diff(
            "layer0.swap_mlp_gate_up",
            &x,
            &mix,
            &st_after_attn,
            &st_after_block,
            arch,
            &config,
        )?;
    }
    {
        let mut mix = clone_linear_block(&st_block)?;
        mix.mlp = MlpKind::Dense(MetalQwen35DenseMlpWeights {
            inputs: MlpInputProjection::Split {
                gate_proj: st_mlp.gate_proj.clone(),
                up_proj: st_mlp.up_proj.clone(),
            },
            down_proj: gg_mlp.down_proj.clone(),
            gate_proj: st_mlp.gate_proj.clone(),
            up_proj: st_mlp.up_proj.clone(),
        });
        print_linear_block_mix_diff(
            "layer0.swap_mlp_down_proj",
            &x,
            &mix,
            &st_after_attn,
            &st_after_block,
            arch,
            &config,
        )?;
    }
    {
        let mut mix = clone_linear_block(&st_block)?;
        mix.attention = MetalQwen35Attention::Linear(clone_linear_attn_weights(gg_attn));
        print_linear_block_mix_diff(
            "layer0.swap_attention_all",
            &x,
            &mix,
            &st_after_attn,
            &st_after_block,
            arch,
            &config,
        )?;
    }

    Ok(())
}

#[test]
fn verify_block_batched_matches_verify_block_for_b1() -> Result<()> {
    let Some(model_path) = env::var_os("QWEN35_MODEL_PATH").map(PathBuf::from) else {
        eprintln!("QWEN35_MODEL_PATH unset; skipping verify_block_batched B=1 equivalence test");
        return Ok(());
    };
    let _guard = metal_test_guard();

    let config = load_metal_config(&model_path)?;
    let MetalModelArch::Qwen35(arch) = &config.arch else {
        anyhow::bail!("QWEN35_MODEL_PATH must point to a Qwen3.5 model");
    };
    let weights = load_qwen35_metal_weights(&model_path, &config)?;
    let cpp_model = weights
        .cpp_model
        .as_ref()
        .context("Qwen3.5 compiled C++ model unavailable")?;

    let prompt_tokens = [1_i32, 2, 3, 4];
    let block_tokens = [5_i32, 6];
    let prompt_len = prompt_tokens.len() as i32;
    let block_size = block_tokens.len() as i32;
    let kv_capacity = prompt_len + block_size + 4;
    let cache_shape = [
        1_i32,
        config.num_key_value_heads as i32,
        kv_capacity,
        config.head_dim as i32,
    ];

    let num_full_layers = arch.num_full_attention_layers();
    let mut kv_flat: Vec<MlxArray> = (0..num_full_layers)
        .flat_map(|_| {
            [
                zeros(&cache_shape, Dtype::Bfloat16),
                zeros(&cache_shape, Dtype::Bfloat16),
            ]
        })
        .collect();
    let recurrent = MetalRecurrentState::new(arch.num_linear_attention_layers(), &arch.linear);
    let mut gdr_flat: Vec<MlxArray> = recurrent
        .states
        .iter()
        .zip(recurrent.conv_states.iter())
        .flat_map(|(state, conv)| [state.clone(), conv.clone()])
        .collect();

    let prompt_arr = MlxArray::from_slice_i32(&prompt_tokens, &[prompt_len]);
    let prompt_logits =
        cpp_model.prefill(&prompt_arr, prompt_len, 0, &mut kv_flat, &mut gdr_flat)?;
    let mut prompt_refs: Vec<&MlxArray> = Vec::with_capacity(1 + kv_flat.len() + gdr_flat.len());
    prompt_refs.push(&prompt_logits);
    prompt_refs.extend(kv_flat.iter());
    prompt_refs.extend(gdr_flat.iter());
    eval(&prompt_refs);

    let cache_pos = prompt_len;
    let mut verify_kv = kv_flat.clone();
    let mut verify_gdr = gdr_flat.clone();
    let verify_tokens = MlxArray::from_slice_i32(&block_tokens, &[block_size]);
    let verify_logits = cpp_model.verify_block(
        &verify_tokens,
        block_size,
        cache_pos,
        &mut verify_kv,
        &mut verify_gdr,
    )?;

    let mut batched_kv = kv_flat.clone();
    let mut batched_gdr = gdr_flat.clone();
    let batched_tokens = MlxArray::from_slice_i32(&block_tokens, &[1, block_size]);
    let rope_offsets = MlxArray::from_slice_i32(&[cache_pos], &[1]);
    let batched_logits = cpp_model.verify_block_batched(
        &batched_tokens,
        1,
        block_size,
        &[cache_pos],
        &mut batched_kv,
        &mut batched_gdr,
        None,
        &rope_offsets,
    )?;

    let verify_logits_f32 = as_dtype(&verify_logits, Dtype::Float32);
    let batched_logits_f32 = as_dtype(&batched_logits, Dtype::Float32);
    eval(&[&verify_logits_f32, &batched_logits_f32]);

    assert_eq!(verify_logits_f32.shape(), batched_logits_f32.shape());
    for (idx, (lhs, rhs)) in verify_logits_f32
        .as_slice_f32()
        .iter()
        .zip(batched_logits_f32.as_slice_f32().iter())
        .enumerate()
    {
        assert!(
            (lhs - rhs).abs() < 1e-3,
            "logit[{idx}] mismatch: {lhs} vs {rhs}"
        );
    }

    Ok(())
}

#[test]
fn verify_block_summary_matches_batched_sampled_for_b1() -> Result<()> {
    let Some(model_path) = env::var_os("QWEN35_MODEL_PATH").map(PathBuf::from) else {
        eprintln!("QWEN35_MODEL_PATH unset; skipping verify_block_summary B=1 equivalence test");
        return Ok(());
    };
    let _guard = metal_test_guard();

    let config = load_metal_config(&model_path)?;
    let MetalModelArch::Qwen35(arch) = &config.arch else {
        anyhow::bail!("QWEN35_MODEL_PATH must point to a Qwen3.5 model");
    };
    let weights = load_qwen35_metal_weights(&model_path, &config)?;
    let cpp_model = weights
        .cpp_model
        .as_ref()
        .context("Qwen3.5 compiled C++ model unavailable")?;

    let params = SamplingParams {
        temperature: 0.0,
        ..Default::default()
    };
    let prompt_tokens = [1_i32, 2, 3, 4];
    let block_tokens = [5_i32, 6];
    let prompt_len = prompt_tokens.len() as i32;
    let block_size = block_tokens.len() as i32;
    let kv_capacity = prompt_len + block_size + 4;
    let cache_shape = [
        1_i32,
        config.num_key_value_heads as i32,
        kv_capacity,
        config.head_dim as i32,
    ];

    let num_full_layers = arch.num_full_attention_layers();
    let mut kv_flat: Vec<MlxArray> = (0..num_full_layers)
        .flat_map(|_| {
            [
                zeros(&cache_shape, Dtype::Bfloat16),
                zeros(&cache_shape, Dtype::Bfloat16),
            ]
        })
        .collect();
    let recurrent = MetalRecurrentState::new(arch.num_linear_attention_layers(), &arch.linear);
    let mut gdr_flat: Vec<MlxArray> = recurrent
        .states
        .iter()
        .zip(recurrent.conv_states.iter())
        .flat_map(|(state, conv)| [state.clone(), conv.clone()])
        .collect();

    let prompt_arr = MlxArray::from_slice_i32(&prompt_tokens, &[prompt_len]);
    let prompt_logits =
        cpp_model.prefill(&prompt_arr, prompt_len, 0, &mut kv_flat, &mut gdr_flat)?;
    let mut prompt_refs: Vec<&MlxArray> = Vec::with_capacity(1 + kv_flat.len() + gdr_flat.len());
    prompt_refs.push(&prompt_logits);
    prompt_refs.extend(kv_flat.iter());
    prompt_refs.extend(gdr_flat.iter());
    eval(&prompt_refs);

    let cache_pos = prompt_len;
    let mut scalar_kv = kv_flat.clone();
    let mut scalar_gdr = gdr_flat.clone();
    let verify_tokens = MlxArray::from_slice_i32(&block_tokens, &[block_size]);
    let scalar_summary = cpp_model.verify_block_summary(
        &verify_tokens,
        block_size,
        cache_pos,
        &mut scalar_kv,
        &mut scalar_gdr,
        &params,
        None,
    )?;

    let mut batched_kv = kv_flat.clone();
    let mut batched_gdr = gdr_flat.clone();
    let batched_tokens = MlxArray::from_slice_i32(&block_tokens, &[1, block_size]);
    let rope_offsets = MlxArray::from_slice_i32(&[cache_pos], &[1]);
    let batched_sampled = cpp_model.verify_block_batched_sampled(
        &batched_tokens,
        1,
        block_size,
        &[cache_pos],
        &mut batched_kv,
        &mut batched_gdr,
        None,
        &rope_offsets,
        &params,
        None,
    )?;
    let batched_sampled = reshape(&batched_sampled, &[block_size]);

    eval(&[&batched_sampled]);
    let batched_sampled = batched_sampled.as_slice_i32();
    let expected_matched_prefix_len = block_tokens
        .iter()
        .skip(1)
        .zip(batched_sampled.iter())
        .take_while(|(draft, sampled)| draft == sampled)
        .count();
    assert_eq!(
        scalar_summary.matched_prefix_len,
        expected_matched_prefix_len
    );
    assert_eq!(
        scalar_summary.next_token,
        batched_sampled[expected_matched_prefix_len] as u32
    );

    Ok(())
}

#[test]
fn verify_block_batched_matches_independent_verify_block_for_b2() -> Result<()> {
    let Some(model_path) = env::var_os("QWEN35_MODEL_PATH").map(PathBuf::from) else {
        eprintln!("QWEN35_MODEL_PATH unset; skipping verify_block_batched B=2 equivalence test");
        return Ok(());
    };
    let _guard = metal_test_guard();

    let config = load_metal_config(&model_path)?;
    let MetalModelArch::Qwen35(arch) = &config.arch else {
        anyhow::bail!("QWEN35_MODEL_PATH must point to a Qwen3.5 model");
    };
    let weights = load_qwen35_metal_weights(&model_path, &config)?;
    let cpp_model = weights
        .cpp_model
        .as_ref()
        .context("Qwen3.5 compiled C++ model unavailable")?;

    // Two independent rows with DIFFERENT prompts but the SAME prompt
    // length (so both sit at `cache_pos = prompt_len` after prefill and
    // no left-padding / attention-mask plumbing is exercised at this
    // layer — that's the 2c.4 scheduler's job). Different tokens ensure
    // the per-row GDR state actually diverges, catching any accidental
    // cross-row leak in the batched verify.
    let row_prompts: [[i32; 4]; 2] = [[1, 2, 3, 4], [5, 6, 7, 8]];
    let row_blocks: [[i32; 2]; 2] = [[11, 12], [13, 14]];
    let prompt_len = row_prompts[0].len() as i32;
    let block_size = row_blocks[0].len() as i32;
    let kv_capacity = prompt_len + block_size + 4;
    let cache_shape = [
        1_i32,
        config.num_key_value_heads as i32,
        kv_capacity,
        config.head_dim as i32,
    ];

    let num_full_layers = arch.num_full_attention_layers();
    let mut per_row_kv_after_verify: Vec<Vec<MlxArray>> = Vec::with_capacity(2);
    let mut per_row_gdr_after_verify: Vec<Vec<MlxArray>> = Vec::with_capacity(2);
    let mut per_row_verify_logits: Vec<MlxArray> = Vec::with_capacity(2);

    // Also capture the post-prefill (pre-verify) state per row — we'll
    // stack these as the starting point for the batched verify call.
    let mut per_row_kv_pre_verify: Vec<Vec<MlxArray>> = Vec::with_capacity(2);
    let mut per_row_gdr_pre_verify: Vec<Vec<MlxArray>> = Vec::with_capacity(2);

    for (prompt_tokens, block_tokens) in row_prompts.iter().zip(row_blocks.iter()) {
        let mut kv_flat: Vec<MlxArray> = (0..num_full_layers)
            .flat_map(|_| {
                [
                    zeros(&cache_shape, Dtype::Bfloat16),
                    zeros(&cache_shape, Dtype::Bfloat16),
                ]
            })
            .collect();
        let recurrent = MetalRecurrentState::new(arch.num_linear_attention_layers(), &arch.linear);
        let mut gdr_flat: Vec<MlxArray> = recurrent
            .states
            .iter()
            .zip(recurrent.conv_states.iter())
            .flat_map(|(state, conv)| [state.clone(), conv.clone()])
            .collect();

        let prompt_arr = MlxArray::from_slice_i32(prompt_tokens, &[prompt_len]);
        let prompt_logits =
            cpp_model.prefill(&prompt_arr, prompt_len, 0, &mut kv_flat, &mut gdr_flat)?;
        let mut prompt_refs: Vec<&MlxArray> =
            Vec::with_capacity(1 + kv_flat.len() + gdr_flat.len());
        prompt_refs.push(&prompt_logits);
        prompt_refs.extend(kv_flat.iter());
        prompt_refs.extend(gdr_flat.iter());
        eval(&prompt_refs);

        per_row_kv_pre_verify.push(kv_flat.clone());
        per_row_gdr_pre_verify.push(gdr_flat.clone());

        let verify_tokens = MlxArray::from_slice_i32(block_tokens, &[block_size]);
        let mut verify_kv = kv_flat;
        let mut verify_gdr = gdr_flat;
        let verify_logits = cpp_model.verify_block(
            &verify_tokens,
            block_size,
            prompt_len,
            &mut verify_kv,
            &mut verify_gdr,
        )?;
        let mut verify_refs: Vec<&MlxArray> =
            Vec::with_capacity(1 + verify_kv.len() + verify_gdr.len());
        verify_refs.push(&verify_logits);
        verify_refs.extend(verify_kv.iter());
        verify_refs.extend(verify_gdr.iter());
        eval(&verify_refs);

        per_row_verify_logits.push(verify_logits);
        per_row_kv_after_verify.push(verify_kv);
        per_row_gdr_after_verify.push(verify_gdr);
    }

    // Stack the pre-verify per-row states along batch axis for the
    // batched verify call.
    let n_kv = per_row_kv_pre_verify[0].len();
    let n_gdr = per_row_gdr_pre_verify[0].len();
    let mut packed_kv: Vec<MlxArray> = Vec::with_capacity(n_kv);
    for kv_idx in 0..n_kv {
        let stacked: Vec<MlxArray> = per_row_kv_pre_verify
            .iter()
            .map(|row_kv| row_kv[kv_idx].clone())
            .collect();
        packed_kv.push(concatenate_axis(&stacked, 0));
    }
    let mut packed_gdr: Vec<MlxArray> = Vec::with_capacity(n_gdr);
    for gdr_idx in 0..n_gdr {
        let stacked: Vec<MlxArray> = per_row_gdr_pre_verify
            .iter()
            .map(|row_gdr| row_gdr[gdr_idx].clone())
            .collect();
        packed_gdr.push(concatenate_axis(&stacked, 0));
    }

    let batched_block_tokens: Vec<i32> = row_blocks.iter().flatten().copied().collect();
    let batched_tokens = MlxArray::from_slice_i32(&batched_block_tokens, &[2, block_size]);
    let rope_offsets = MlxArray::from_slice_i32(&[prompt_len, prompt_len], &[2]);

    let batched_logits = cpp_model.verify_block_batched(
        &batched_tokens,
        2,
        block_size,
        &[prompt_len, prompt_len],
        &mut packed_kv,
        &mut packed_gdr,
        None,
        &rope_offsets,
    )?;

    let batched_logits_f32 = as_dtype(&batched_logits, Dtype::Float32);
    let mut eval_refs: Vec<&MlxArray> = Vec::with_capacity(1 + packed_kv.len() + packed_gdr.len());
    eval_refs.push(&batched_logits_f32);
    eval_refs.extend(packed_kv.iter());
    eval_refs.extend(packed_gdr.iter());
    eval(&eval_refs);

    assert_eq!(
        batched_logits_f32.shape(),
        &[2, block_size, config.vocab_size as i32]
    );

    let vocab = config.vocab_size;
    let batched_slice = batched_logits_f32.as_slice_f32();
    assert_eq!(
        batched_slice.len(),
        2 * (block_size as usize) * vocab,
        "batched logits elem count mismatch"
    );

    for (row_idx, scalar_logits) in per_row_verify_logits.iter().enumerate() {
        let scalar_f32 = as_dtype(scalar_logits, Dtype::Float32);
        eval(&[&scalar_f32]);
        let scalar_slice = scalar_f32.as_slice_f32();
        let expected_len = (block_size as usize) * vocab;
        assert_eq!(
            scalar_slice.len(),
            expected_len,
            "scalar row {row_idx} logits len mismatch"
        );
        let offset = row_idx * expected_len;
        for (pos, (lhs, rhs)) in scalar_slice
            .iter()
            .zip(batched_slice[offset..offset + expected_len].iter())
            .enumerate()
        {
            assert!(
                (lhs - rhs).abs() < 1e-3,
                "row {row_idx} logit[{pos}] mismatch: scalar={lhs} batched={rhs}"
            );
        }
    }

    Ok(())
}

#[test]
fn packed_decode_batched_sampling_matches_scalar_sampling_for_b4() -> Result<()> {
    let Some(model_path) = env::var_os("QWEN35_MODEL_PATH").map(PathBuf::from) else {
        eprintln!(
            "QWEN35_MODEL_PATH unset; skipping packed decode batched sampling B=4 equivalence test"
        );
        return Ok(());
    };
    let _guard = metal_test_guard();

    let config = load_metal_config(&model_path)?;
    let MetalModelArch::Qwen35(arch) = &config.arch else {
        anyhow::bail!("QWEN35_MODEL_PATH must point to a Qwen3.5 model");
    };
    let weights = load_qwen35_metal_weights(&model_path, &config)?;
    let cpp_model = weights
        .cpp_model
        .as_ref()
        .context("Qwen3.5 compiled C++ model unavailable")?;

    let params = SamplingParams {
        temperature: 0.0,
        ..Default::default()
    };
    let prompt_rows = [
        [1_i32, 2, 3, 4],
        [5_i32, 6, 7, 8],
        [9_i32, 10, 11, 12],
        [13_i32, 14, 15, 16],
    ];
    let batch_size = i32::try_from(prompt_rows.len()).expect("batch size fits in i32");
    let prompt_len = i32::try_from(prompt_rows[0].len()).expect("prompt len fits in i32");
    let kv_capacity = prompt_len + KV_CACHE_CHUNK;
    let cache_shape = [
        1_i32,
        config.num_key_value_heads as i32,
        kv_capacity,
        config.head_dim as i32,
    ];

    let num_full_layers = arch.num_full_attention_layers();
    let mut per_row_kv = Vec::with_capacity(prompt_rows.len());
    let mut per_row_gdr = Vec::with_capacity(prompt_rows.len());
    let mut decode_inputs = Vec::with_capacity(prompt_rows.len());

    for prompt_tokens in prompt_rows {
        let mut kv_flat: Vec<MlxArray> = (0..num_full_layers)
            .flat_map(|_| {
                [
                    zeros(&cache_shape, Dtype::Bfloat16),
                    zeros(&cache_shape, Dtype::Bfloat16),
                ]
            })
            .collect();
        let recurrent = MetalRecurrentState::new(arch.num_linear_attention_layers(), &arch.linear);
        let mut gdr_flat: Vec<MlxArray> = recurrent
            .states
            .iter()
            .zip(recurrent.conv_states.iter())
            .flat_map(|(state, conv)| [state.clone(), conv.clone()])
            .collect();

        let prompt_arr = MlxArray::from_slice_i32(&prompt_tokens, &[prompt_len]);
        let prompt_logits =
            cpp_model.prefill(&prompt_arr, prompt_len, 0, &mut kv_flat, &mut gdr_flat)?;
        let mut prompt_refs: Vec<&MlxArray> =
            Vec::with_capacity(1 + kv_flat.len() + gdr_flat.len());
        prompt_refs.push(&prompt_logits);
        prompt_refs.extend(kv_flat.iter());
        prompt_refs.extend(gdr_flat.iter());
        eval(&prompt_refs);

        let decode_input = gpu_sample_token(&prompt_logits, &params);
        eval(&[&decode_input]);
        decode_inputs.push(decode_input.item_i32());
        per_row_kv.push(kv_flat);
        per_row_gdr.push(gdr_flat);
    }

    let n_kv = i32::try_from(per_row_kv[0].len()).expect("kv len fits in i32");
    let n_gdr = i32::try_from(per_row_gdr[0].len()).expect("gdr len fits in i32");
    let mut packed_kv = Vec::with_capacity(n_kv as usize);
    for kv_idx in 0..n_kv as usize {
        let stacked: Vec<MlxArray> = per_row_kv
            .iter()
            .map(|row_kv| row_kv[kv_idx].clone())
            .collect();
        packed_kv.push(concatenate_axis(&stacked, 0));
    }
    let mut packed_gdr = Vec::with_capacity(n_gdr as usize);
    for gdr_idx in 0..n_gdr as usize {
        let stacked: Vec<MlxArray> = per_row_gdr
            .iter()
            .map(|row_gdr| row_gdr[gdr_idx].clone())
            .collect();
        packed_gdr.push(concatenate_axis(&stacked, 0));
    }

    let decode_tokens = MlxArray::from_slice_i32(&decode_inputs, &[batch_size]);
    let rope_offsets = MlxArray::from_slice_i32(
        &vec![prompt_len; usize::try_from(batch_size).expect("batch size fits in usize")],
        &[batch_size],
    );
    let batched_logits = cpp_model.step_batch_packed(
        &decode_tokens,
        batch_size,
        prompt_len,
        &mut packed_kv,
        n_kv,
        &mut packed_gdr,
        n_gdr,
        None,
        Some(&rope_offsets),
    )?;
    let mut decode_refs: Vec<&MlxArray> =
        Vec::with_capacity(1 + packed_kv.len() + packed_gdr.len());
    decode_refs.push(&batched_logits);
    decode_refs.extend(packed_kv.iter());
    decode_refs.extend(packed_gdr.iter());
    eval(&decode_refs);

    let batched_sampled = gpu_sample_token_batched(&batched_logits, &params);
    eval(&[&batched_sampled]);
    let batched_tokens = batched_sampled.as_slice_i32();

    let mut scalar_sampled = Vec::with_capacity(prompt_rows.len());
    for row_idx in 0..batch_size {
        let row_logits = slice_row_for_sampling(&batched_logits, row_idx);
        scalar_sampled.push(gpu_sample_token(&row_logits, &params));
    }
    let scalar_refs: Vec<&MlxArray> = scalar_sampled.iter().collect();
    eval(&scalar_refs);
    let scalar_tokens: Vec<i32> = scalar_sampled
        .iter()
        .map(|sampled| sampled.item_i32())
        .collect();

    assert_eq!(batched_tokens, scalar_tokens);

    Ok(())
}

#[test]
fn packed_decode_varlen_matches_independent_scalar_decode_for_b2() -> Result<()> {
    let Some(model_path) = env::var_os("QWEN35_MODEL_PATH").map(PathBuf::from) else {
        eprintln!("QWEN35_MODEL_PATH unset; skipping packed decode varlen B=2 equivalence test");
        return Ok(());
    };
    let _guard = metal_test_guard();

    let config = load_metal_config(&model_path)?;
    let MetalModelArch::Qwen35(arch) = &config.arch else {
        anyhow::bail!("QWEN35_MODEL_PATH must point to a Qwen3.5 model");
    };
    let weights = load_qwen35_metal_weights(&model_path, &config)?;
    let cpp_model = weights
        .cpp_model
        .as_ref()
        .context("Qwen3.5 compiled C++ model unavailable")?;

    let params = SamplingParams {
        temperature: 0.0,
        ..Default::default()
    };
    let prompt_rows: [&[i32]; 2] = [&[1, 2, 3, 4], &[5, 6]];
    let prompt_lens: Vec<i32> = prompt_rows
        .iter()
        .map(|tokens| i32::try_from(tokens.len()).expect("prompt len fits in i32"))
        .collect();
    let batch_size = i32::try_from(prompt_rows.len()).expect("batch size fits in i32");
    let batch_cache_len = *prompt_lens
        .iter()
        .max()
        .expect("packed decode varlen test requires rows");
    let kv_capacity = batch_cache_len + KV_CACHE_CHUNK;
    let cache_shape = [
        1_i32,
        config.num_key_value_heads as i32,
        kv_capacity,
        config.head_dim as i32,
    ];

    let num_full_layers = arch.num_full_attention_layers();
    let mut per_row_prefill_kv = Vec::with_capacity(prompt_rows.len());
    let mut per_row_prefill_gdr = Vec::with_capacity(prompt_rows.len());
    let mut decode_inputs = Vec::with_capacity(prompt_rows.len());
    let mut scalar_logits_per_row = Vec::with_capacity(prompt_rows.len());

    for (prompt_tokens, &prompt_len) in prompt_rows.iter().zip(prompt_lens.iter()) {
        let mut kv_flat: Vec<MlxArray> = (0..num_full_layers)
            .flat_map(|_| {
                [
                    zeros(&cache_shape, Dtype::Bfloat16),
                    zeros(&cache_shape, Dtype::Bfloat16),
                ]
            })
            .collect();
        let recurrent = MetalRecurrentState::new(arch.num_linear_attention_layers(), &arch.linear);
        let mut gdr_flat: Vec<MlxArray> = recurrent
            .states
            .iter()
            .zip(recurrent.conv_states.iter())
            .flat_map(|(state, conv)| [state.clone(), conv.clone()])
            .collect();

        let prompt_arr = MlxArray::from_slice_i32(prompt_tokens, &[prompt_len]);
        let prompt_logits =
            cpp_model.prefill(&prompt_arr, prompt_len, 0, &mut kv_flat, &mut gdr_flat)?;
        let mut prompt_refs: Vec<&MlxArray> =
            Vec::with_capacity(1 + kv_flat.len() + gdr_flat.len());
        prompt_refs.push(&prompt_logits);
        prompt_refs.extend(kv_flat.iter());
        prompt_refs.extend(gdr_flat.iter());
        eval(&prompt_refs);

        let decode_input = gpu_sample_token(&prompt_logits, &params);
        eval(&[&decode_input]);
        let decode_input = decode_input.item_i32();
        decode_inputs.push(decode_input);

        let decode_token_arr = MlxArray::from_slice_i32(&[decode_input], &[1]);
        let mut scalar_kv = kv_flat.clone();
        let mut scalar_gdr = gdr_flat.clone();
        let scalar_logits = cpp_model.step(
            &decode_token_arr,
            prompt_len,
            &mut scalar_kv,
            &mut scalar_gdr,
        )?;
        let mut scalar_refs: Vec<&MlxArray> =
            Vec::with_capacity(1 + scalar_kv.len() + scalar_gdr.len());
        scalar_refs.push(&scalar_logits);
        scalar_refs.extend(scalar_kv.iter());
        scalar_refs.extend(scalar_gdr.iter());
        eval(&scalar_refs);

        scalar_logits_per_row.push(as_dtype(&scalar_logits, Dtype::Float32));
        per_row_prefill_kv.push(kv_flat);
        per_row_prefill_gdr.push(gdr_flat);
    }

    let left_padding: Vec<i32> = prompt_lens
        .iter()
        .map(|&prompt_len| batch_cache_len - prompt_len)
        .collect();
    let n_kv = per_row_prefill_kv[0].len();
    let n_gdr = per_row_prefill_gdr[0].len();

    let mut packed_kv = Vec::with_capacity(n_kv);
    for kv_idx in 0..n_kv {
        let stacked: Vec<MlxArray> = per_row_prefill_kv
            .iter()
            .zip(left_padding.iter())
            .zip(prompt_lens.iter())
            .map(|((row_kv, &left_pad), &prompt_len)| {
                if left_pad == 0 {
                    row_kv[kv_idx].clone()
                } else {
                    left_pad_kv_cache_row_for_test(
                        &row_kv[kv_idx],
                        left_pad,
                        prompt_len,
                        kv_capacity,
                    )
                }
            })
            .collect();
        packed_kv.push(concatenate_axis(&stacked, 0));
    }

    let mut packed_gdr = Vec::with_capacity(n_gdr);
    for gdr_idx in 0..n_gdr {
        let stacked: Vec<MlxArray> = per_row_prefill_gdr
            .iter()
            .map(|row_gdr| row_gdr[gdr_idx].clone())
            .collect();
        packed_gdr.push(concatenate_axis(&stacked, 0));
    }

    let decode_tokens = MlxArray::from_slice_i32(&decode_inputs, &[batch_size]);
    let attn_mask =
        crate::backend::metal::mlx::build_varlen_decode_mask(&left_padding, batch_cache_len);
    let rope_offsets = MlxArray::from_slice_i32(&prompt_lens, &[batch_size]);
    let packed_logits = cpp_model.step_batch_packed(
        &decode_tokens,
        batch_size,
        batch_cache_len,
        &mut packed_kv,
        i32::try_from(n_kv).expect("kv len fits in i32"),
        &mut packed_gdr,
        i32::try_from(n_gdr).expect("gdr len fits in i32"),
        Some(&attn_mask),
        Some(&rope_offsets),
    )?;
    let packed_logits = as_dtype(&packed_logits, Dtype::Float32);
    let mut packed_refs: Vec<&MlxArray> =
        Vec::with_capacity(1 + packed_kv.len() + packed_gdr.len());
    packed_refs.push(&packed_logits);
    packed_refs.extend(packed_kv.iter());
    packed_refs.extend(packed_gdr.iter());
    eval(&packed_refs);

    assert_eq!(
        packed_logits.shape(),
        &[batch_size, 1, config.vocab_size as i32]
    );

    for (row_idx, scalar_logits) in scalar_logits_per_row.iter().enumerate() {
        eval(&[scalar_logits]);
        let packed_row = slice_row_for_sampling(
            &packed_logits,
            i32::try_from(row_idx).expect("row index fits in i32"),
        );
        let scalar_slice = scalar_logits.as_slice_f32();
        let packed_slice = packed_row.as_slice_f32();
        assert_eq!(
            scalar_slice.len(),
            packed_slice.len(),
            "row {row_idx} logits len mismatch"
        );
        for (pos, (lhs, rhs)) in scalar_slice.iter().zip(packed_slice.iter()).enumerate() {
            assert!(
                (lhs - rhs).abs() < 1e-3,
                "row {row_idx} logit[{pos}] mismatch: scalar={lhs} packed={rhs}"
            );
        }
    }

    Ok(())
}
