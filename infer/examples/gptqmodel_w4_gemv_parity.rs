#![cfg_attr(not(feature = "cuda"), allow(dead_code, unused_imports))]

#[cfg(feature = "cuda")]
mod app {
    use std::{
        fs,
        path::{Path, PathBuf},
    };

    use anyhow::{Context, Result, bail};
    use cuda_kernels::{ffi, prelude::DeviceContext, tensor::CudaAllocTraceExt};
    use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
    use half::{bf16, f16};
    use memmap2::Mmap;
    use safetensors::{Dtype, SafeTensors};
    use serde_json::json;

    #[derive(Debug)]
    struct Args {
        model_path: PathBuf,
        source_model_path: Option<PathBuf>,
        tensor_base: String,
        seed: u32,
        output: PathBuf,
    }

    pub(crate) fn main() -> Result<()> {
        let args = parse_args()?;
        let qweight_key = format!("{}.qweight", args.tensor_base);
        let scales_key = format!("{}.scales", args.tensor_base);
        let qzeros_key = format!("{}.qzeros", args.tensor_base);
        let g_idx_key = format!("{}.g_idx", args.tensor_base);

        let qweight = load_tensor(&args.model_path, &qweight_key)?;
        let scales = load_tensor(&args.model_path, &scales_key)?;
        let qzeros = load_tensor_optional(&args.model_path, &qzeros_key)?;
        let g_idx = load_tensor_optional(&args.model_path, &g_idx_key)?;

        if !(qweight.dtype == Dtype::I32 || qweight.dtype == Dtype::U32) {
            bail!(
                "{} dtype must be I32/U32, got {:?}",
                qweight_key,
                qweight.dtype
            );
        }
        if !matches!(scales.dtype, Dtype::BF16 | Dtype::F16 | Dtype::F32) {
            bail!(
                "{} dtype must be BF16/F16/F32, got {:?}",
                scales_key,
                scales.dtype
            );
        }
        if qweight.shape.len() != 2 || scales.shape.len() != 2 {
            bail!("qweight and scales tensors must be rank-2");
        }

        let group_size = gptq_group_size(&args.model_path)?;
        let gptq_rows = qweight.shape[0];
        let rows = qweight.shape[1];
        let cols = gptq_rows
            .checked_mul(8)
            .context("GPTQ W4 K dimension overflow")?;
        let num_groups = cols / group_size;
        if !cols.is_multiple_of(group_size) {
            bail!("K={cols} is not divisible by group_size={group_size}");
        }
        if scales.shape != [num_groups, rows] {
            bail!(
                "scales shape mismatch: expected [{num_groups}, {rows}], got {:?}",
                scales.shape
            );
        }

        let qzeros_report = qzeros
            .as_ref()
            .map(|tensor| validate_qzeros(&qzeros_key, tensor))
            .transpose()?;
        let g_idx_report = g_idx
            .as_ref()
            .map(|tensor| validate_g_idx(&g_idx_key, tensor, cols, group_size))
            .transpose()?;

        let layout = convert_gptqmodel_w4_layout(
            &qweight.data,
            &qweight.shape,
            &scales.data,
            scales.dtype,
            &scales.shape,
            group_size,
        )?;
        let input = deterministic_input(cols, args.seed);
        let cpu_gptq = cpu_gptq_reference(
            &qweight.data,
            &qweight.shape,
            &scales.data,
            scales.dtype,
            group_size,
            &input,
        )?;

        let ctx = DeviceContext::new()?;
        let packed_gpu = ctx
            .stream
            .clone_htod(&layout.packed)
            .context("H2D packed W4")?;
        let scales_gpu = ctx
            .stream
            .clone_htod(&layout.scales)
            .context("H2D W4 scales")?;
        let input_gpu = ctx
            .stream
            .clone_htod(&input)
            .context("H2D deterministic input")?;
        let mut cuda_out: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros_traced(rows)
            .context("alloc W4A16 output")?;
        {
            let (packed_ptr, _gp) = packed_gpu.device_ptr(&ctx.stream);
            let (scales_ptr, _gs) = scales_gpu.device_ptr(&ctx.stream);
            let (input_ptr, _gx) = input_gpu.device_ptr(&ctx.stream);
            let (output_ptr, _gy) = cuda_out.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::w4a16_gemv_cuda(
                    packed_ptr as *const u8,
                    scales_ptr as *const ffi::Half,
                    input_ptr as *const ffi::Half,
                    output_ptr as *mut ffi::Half,
                    rows as i32,
                    cols as i32,
                    group_size as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .context("w4a16_gemv_cuda failed")?;
            }
        }
        ctx.sync()?;
        let cuda_host = ctx
            .stream
            .clone_dtoh(&cuda_out)
            .context("D2H CUDA output")?;
        let cuda_values: Vec<f32> = cuda_host.iter().map(|value| value.to_f32()).collect();

        let cuda_vs_cpu = compare(
            "cuda_w4a16_gemv",
            "cpu_faithful_gptq_reference",
            &cuda_values,
            &cpu_gptq,
        );
        let first8_cuda_vs_cpu = first8(
            "cuda_w4a16_gemv",
            "cpu_faithful_gptq_reference",
            &cuda_values,
            &cpu_gptq,
        );

        let source_comparison = if let Some(source_model_path) = &args.source_model_path {
            let source_key = format!("{}.weight", args.tensor_base);
            let source = load_tensor(source_model_path, &source_key)?;
            let source_out = cpu_source_dense_reference(&source_key, &source, rows, cols, &input)?;
            Some(json!({
                "source_model_path": source_model_path,
                "source_key": source_key,
                "comparison": compare(
                    "cpu_faithful_gptq_reference",
                    "bf16_source_dense_reference",
                    &cpu_gptq,
                    &source_out,
                ),
                "first8": first8(
                    "cpu_faithful_gptq_reference",
                    "bf16_source_dense_reference",
                    &cpu_gptq,
                    &source_out,
                ),
            }))
        } else {
            None
        };

        let payload = json!({
            "model_path": args.model_path,
            "tensor_base": args.tensor_base,
            "seed": args.seed,
            "input": {
                "shape": [1, cols],
                "first8": input.iter().take(8).map(|value| value.to_f32()).collect::<Vec<_>>(),
            },
            "projection": {
                "rows": rows,
                "cols": cols,
                "bits": 4,
                "group_size": group_size,
                "qweight_shape": qweight.shape,
                "scales_shape": scales.shape,
                "qzeros": qzeros_report,
                "g_idx": g_idx_report,
            },
            "cuda_w4a16_vs_cpu_gptq_reference": cuda_vs_cpu,
            "first8_cuda_w4a16_vs_cpu_gptq_reference": first8_cuda_vs_cpu,
            "cpu_gptq_reference_vs_source": source_comparison,
        });
        if let Some(parent) = args.output.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create output dir {}", parent.display()))?;
        }
        fs::write(&args.output, serde_json::to_vec_pretty(&payload)?)?;
        println!("{}", serde_json::to_string_pretty(&payload)?);
        Ok(())
    }

    #[derive(Debug)]
    struct TensorData {
        dtype: Dtype,
        shape: Vec<usize>,
        data: Vec<u8>,
    }

    #[derive(Debug)]
    struct GptqLayout {
        packed: Vec<u8>,
        scales: Vec<bf16>,
    }

    fn load_tensor(model_path: &Path, key: &str) -> Result<TensorData> {
        let shard_name = shard_for_key(model_path, key)?;
        let shard_path = model_path.join(&shard_name);
        let file = fs::File::open(&shard_path)
            .with_context(|| format!("open safetensors shard {}", shard_path.display()))?;
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("mmap safetensors shard {}", shard_path.display()))?;
        let tensors = SafeTensors::deserialize(&mmap)
            .with_context(|| format!("parse safetensors shard {}", shard_path.display()))?;
        let view = tensors
            .tensor(key)
            .with_context(|| format!("missing tensor {key} in {shard_name}"))?;
        Ok(TensorData {
            dtype: view.dtype(),
            shape: view.shape().to_vec(),
            data: view.data().to_vec(),
        })
    }

    fn load_tensor_optional(model_path: &Path, key: &str) -> Result<Option<TensorData>> {
        let index_path = model_path.join("model.safetensors.index.json");
        let raw = fs::read_to_string(&index_path)
            .with_context(|| format!("read {}", index_path.display()))?;
        let index: serde_json::Value = serde_json::from_str(&raw)
            .with_context(|| format!("parse {}", index_path.display()))?;
        let Some(shard_name) = index
            .get("weight_map")
            .and_then(|wm| wm.get(key))
            .and_then(|value| value.as_str())
        else {
            return Ok(None);
        };
        let shard_path = model_path.join(shard_name);
        let file = fs::File::open(&shard_path)
            .with_context(|| format!("open safetensors shard {}", shard_path.display()))?;
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("mmap safetensors shard {}", shard_path.display()))?;
        let tensors = SafeTensors::deserialize(&mmap)
            .with_context(|| format!("parse safetensors shard {}", shard_path.display()))?;
        let view = tensors
            .tensor(key)
            .with_context(|| format!("missing tensor {key} in {shard_name}"))?;
        Ok(Some(TensorData {
            dtype: view.dtype(),
            shape: view.shape().to_vec(),
            data: view.data().to_vec(),
        }))
    }

    fn shard_for_key(model_path: &Path, key: &str) -> Result<String> {
        let index_path = model_path.join("model.safetensors.index.json");
        let raw = fs::read_to_string(&index_path)
            .with_context(|| format!("read {}", index_path.display()))?;
        let index: serde_json::Value = serde_json::from_str(&raw)
            .with_context(|| format!("parse {}", index_path.display()))?;
        index
            .get("weight_map")
            .and_then(|wm| wm.get(key))
            .and_then(|value| value.as_str())
            .map(str::to_owned)
            .with_context(|| format!("missing {key} in {}", index_path.display()))
    }

    fn gptq_group_size(model_path: &Path) -> Result<usize> {
        let raw = fs::read_to_string(model_path.join("config.json")).context("read config.json")?;
        let config: serde_json::Value = serde_json::from_str(&raw).context("parse config.json")?;
        let group_size = config
            .get("quantization_config")
            .and_then(|quant| quant.get("group_size"))
            .and_then(|value| value.as_u64())
            .context("config.json missing quantization_config.group_size")?;
        Ok(group_size as usize)
    }

    fn validate_qzeros(name: &str, tensor: &TensorData) -> Result<serde_json::Value> {
        if !(tensor.dtype == Dtype::I32 || tensor.dtype == Dtype::U32) {
            bail!("{name} dtype must be I32/U32, got {:?}", tensor.dtype);
        }
        if !tensor.data.len().is_multiple_of(std::mem::size_of::<u32>()) {
            bail!(
                "{name} byte length {} is not u32-aligned",
                tensor.data.len()
            );
        }
        let mut bad = Vec::new();
        for (word_idx, bytes) in tensor
            .data
            .chunks_exact(std::mem::size_of::<u32>())
            .enumerate()
        {
            let word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            for nibble_idx in 0..8 {
                let nibble = (word >> (nibble_idx * 4)) & 0x0f;
                if nibble != 7 && bad.len() < 8 {
                    bad.push(json!({
                        "word": word_idx,
                        "nibble": nibble_idx,
                        "value": nibble,
                    }));
                }
            }
        }
        Ok(json!({
            "present": true,
            "shape": tensor.shape,
            "all_nibbles_are_7": bad.is_empty(),
            "first_bad": bad,
        }))
    }

    fn validate_g_idx(
        name: &str,
        tensor: &TensorData,
        cols: usize,
        group_size: usize,
    ) -> Result<serde_json::Value> {
        if !(tensor.dtype == Dtype::I32 || tensor.dtype == Dtype::U32) {
            bail!("{name} dtype must be I32/U32, got {:?}", tensor.dtype);
        }
        if tensor.shape != [cols] {
            bail!("{name} shape must be [{cols}], got {:?}", tensor.shape);
        }
        let mut mismatches = Vec::new();
        for k in 0..cols {
            let value = read_u32_le(&tensor.data, k)?;
            let expected = (k / group_size) as u32;
            if value != expected && mismatches.len() < 8 {
                mismatches.push(json!({
                    "k": k,
                    "value": value,
                    "expected": expected,
                }));
            }
        }
        Ok(json!({
            "present": true,
            "shape": tensor.shape,
            "matches_k_div_group_size": mismatches.is_empty(),
            "first_mismatches": mismatches,
        }))
    }

    fn convert_gptqmodel_w4_layout(
        qweight: &[u8],
        qweight_shape: &[usize],
        scales: &[u8],
        scales_dtype: Dtype,
        scales_shape: &[usize],
        group_size: usize,
    ) -> Result<GptqLayout> {
        let gptq_rows = qweight_shape[0];
        let rows = qweight_shape[1];
        let cols = gptq_rows * 8;
        let num_groups = cols / group_size;
        if scales_shape != [num_groups, rows] {
            bail!("scales shape mismatch: expected [{num_groups}, {rows}], got {scales_shape:?}");
        }

        let mut packed = vec![0u8; rows * cols / 2];
        for k in 0..cols {
            let gptq_row = k / 8;
            let bit_pos = (k % 8) * 4;
            for row in 0..rows {
                let word = read_u32_le(qweight, gptq_row * rows + row)?;
                let nibble = ((word >> bit_pos) & 0x0f) as u8;
                let byte_idx = row * (cols / 2) + k / 2;
                if k % 2 == 0 {
                    packed[byte_idx] = (packed[byte_idx] & 0xf0) | nibble;
                } else {
                    packed[byte_idx] = (packed[byte_idx] & 0x0f) | (nibble << 4);
                }
            }
        }

        let mut scale_out = vec![bf16::ZERO; rows * num_groups];
        for group in 0..num_groups {
            for row in 0..rows {
                scale_out[row * num_groups + group] =
                    scale_to_bf16(scales, scales_dtype, group * rows + row)?;
            }
        }
        Ok(GptqLayout {
            packed,
            scales: scale_out,
        })
    }

    fn cpu_gptq_reference(
        qweight: &[u8],
        qweight_shape: &[usize],
        scales: &[u8],
        scales_dtype: Dtype,
        group_size: usize,
        input: &[bf16],
    ) -> Result<Vec<f32>> {
        let gptq_rows = qweight_shape[0];
        let rows = qweight_shape[1];
        let cols = gptq_rows * 8;
        if input.len() != cols {
            bail!("input len {} does not match K={cols}", input.len());
        }
        let mut out = vec![0.0f32; rows];
        for row in 0..rows {
            let mut sum = 0.0f32;
            for k in 0..cols {
                let word = read_u32_le(qweight, (k / 8) * rows + row)?;
                let q = ((word >> ((k % 8) * 4)) & 0x0f) as i32 - 8;
                let scale = scale_to_f32(scales, scales_dtype, (k / group_size) * rows + row)?;
                sum += q as f32 * scale * input[k].to_f32();
            }
            out[row] = sum;
        }
        Ok(out)
    }

    fn cpu_source_dense_reference(
        key: &str,
        source: &TensorData,
        rows: usize,
        cols: usize,
        input: &[bf16],
    ) -> Result<Vec<f32>> {
        if source.shape != [rows, cols] {
            bail!(
                "{key} source shape mismatch: expected [{rows}, {cols}], got {:?}",
                source.shape
            );
        }
        if !matches!(source.dtype, Dtype::BF16 | Dtype::F16 | Dtype::F32) {
            bail!(
                "{key} source dtype must be BF16/F16/F32, got {:?}",
                source.dtype
            );
        }
        let mut out = vec![0.0f32; rows];
        for row in 0..rows {
            let mut sum = 0.0f32;
            for k in 0..cols {
                let weight = tensor_scalar_to_f32(&source.data, source.dtype, row * cols + k)?;
                sum += weight * input[k].to_f32();
            }
            out[row] = sum;
        }
        Ok(out)
    }

    fn read_u32_le(data: &[u8], idx: usize) -> Result<u32> {
        let offset = idx * std::mem::size_of::<u32>();
        let bytes = data
            .get(offset..offset + std::mem::size_of::<u32>())
            .ok_or_else(|| anyhow::anyhow!("u32 index {idx} out of range"))?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn scale_to_bf16(data: &[u8], dtype: Dtype, idx: usize) -> Result<bf16> {
        Ok(bf16::from_f32(scale_to_f32(data, dtype, idx)?))
    }

    fn scale_to_f32(data: &[u8], dtype: Dtype, idx: usize) -> Result<f32> {
        tensor_scalar_to_f32(data, dtype, idx)
    }

    fn tensor_scalar_to_f32(data: &[u8], dtype: Dtype, idx: usize) -> Result<f32> {
        match dtype {
            Dtype::BF16 => {
                let offset = idx * std::mem::size_of::<bf16>();
                let bytes = data
                    .get(offset..offset + std::mem::size_of::<bf16>())
                    .ok_or_else(|| anyhow::anyhow!("bf16 index {idx} out of range"))?;
                Ok(bf16::from_le_bytes([bytes[0], bytes[1]]).to_f32())
            }
            Dtype::F16 => {
                let offset = idx * std::mem::size_of::<f16>();
                let bytes = data
                    .get(offset..offset + std::mem::size_of::<f16>())
                    .ok_or_else(|| anyhow::anyhow!("f16 index {idx} out of range"))?;
                Ok(f16::from_le_bytes([bytes[0], bytes[1]]).to_f32())
            }
            Dtype::F32 => {
                let offset = idx * std::mem::size_of::<f32>();
                let bytes = data
                    .get(offset..offset + std::mem::size_of::<f32>())
                    .ok_or_else(|| anyhow::anyhow!("f32 index {idx} out of range"))?;
                Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            }
            dtype => bail!("unsupported scalar dtype {dtype:?}"),
        }
    }

    fn deterministic_input(len: usize, seed: u32) -> Vec<bf16> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let raw = ((state >> 8) & 0xffff) as f32 / 65_535.0;
                bf16::from_f32((raw - 0.5) * 2.0)
            })
            .collect()
    }

    fn compare(lhs_name: &str, rhs_name: &str, lhs: &[f32], rhs: &[f32]) -> serde_json::Value {
        assert_eq!(lhs.len(), rhs.len());
        let mut max_abs = 0.0f32;
        let mut mean_abs = 0.0f64;
        let mut max_rel = 0.0f32;
        let mut mean_rel = 0.0f64;
        let mut sq = 0.0f64;
        let mut rhs_sq = 0.0f64;
        let mut max_abs_index = 0usize;
        let mut max_rel_index = 0usize;
        for (idx, (&a, &b)) in lhs.iter().zip(rhs.iter()).enumerate() {
            let abs = (a - b).abs();
            let rel = abs / b.abs().max(1.0e-6);
            if abs > max_abs {
                max_abs = abs;
                max_abs_index = idx;
            }
            if rel > max_rel {
                max_rel = rel;
                max_rel_index = idx;
            }
            mean_abs += abs as f64;
            mean_rel += rel as f64;
            sq += ((a - b) as f64).powi(2);
            rhs_sq += (b as f64).powi(2);
        }
        let n = lhs.len() as f64;
        let rmse = (sq / n).sqrt();
        let rhs_rms = (rhs_sq / n).sqrt();
        json!({
            "lhs": lhs_name,
            "rhs": rhs_name,
            "elements": lhs.len(),
            "max_abs": max_abs,
            "max_abs_index": max_abs_index,
            "mean_abs": mean_abs / n,
            "rmse": rmse,
            "rhs_rms": rhs_rms,
            "rmse_over_rhs_rms": rmse / rhs_rms.max(1.0e-12),
            "max_rel": max_rel,
            "max_rel_index": max_rel_index,
            "mean_rel": mean_rel / n,
        })
    }

    fn first8(lhs_name: &str, rhs_name: &str, lhs: &[f32], rhs: &[f32]) -> Vec<serde_json::Value> {
        (0..8.min(lhs.len()))
            .map(|idx| {
                json!({
                    "index": idx,
                    lhs_name: lhs[idx],
                    rhs_name: rhs[idx],
                    "abs_err": (lhs[idx] - rhs[idx]).abs(),
                    "rel_err": (lhs[idx] - rhs[idx]).abs() / rhs[idx].abs().max(1.0e-6),
                })
            })
            .collect()
    }

    fn parse_args() -> Result<Args> {
        let mut model_path = None;
        let mut source_model_path = None;
        let mut tensor_base = Some("model.language_model.layers.0.mlp.gate_proj".to_string());
        let mut seed = 0x5eed1234_u32;
        let mut output = None;

        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model-path" => model_path = Some(PathBuf::from(next_arg(&mut args, &arg)?)),
                "--source-model-path" => {
                    source_model_path = Some(PathBuf::from(next_arg(&mut args, &arg)?));
                }
                "--tensor-base" => tensor_base = Some(next_arg(&mut args, &arg)?),
                "--seed" => seed = next_arg(&mut args, &arg)?.parse()?,
                "--output" => output = Some(PathBuf::from(next_arg(&mut args, &arg)?)),
                "--help" | "-h" => {
                    println!(
                        "usage: cargo run -p infer --example gptqmodel_w4_gemv_parity --release --features cuda -- \
                         --model-path DIR [--source-model-path BF16_DIR] \
                         [--tensor-base model.language_model.layers.0.mlp.gate_proj] \
                         [--seed 1592594996] --output parity.json"
                    );
                    std::process::exit(0);
                }
                _ => bail!("unknown argument `{arg}`"),
            }
        }
        Ok(Args {
            model_path: model_path.context("--model-path is required")?,
            source_model_path,
            tensor_base: tensor_base.context("--tensor-base is required")?,
            seed,
            output: output.context("--output is required")?,
        })
    }

    fn next_arg(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
        args.next()
            .with_context(|| format!("{flag} requires a value"))
    }
}

#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    app::main()
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("gptqmodel_w4_gemv_parity requires --features cuda");
}
