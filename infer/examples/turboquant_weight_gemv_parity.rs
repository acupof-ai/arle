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
    use half::bf16;
    use memmap2::Mmap;
    use safetensors::{Dtype, SafeTensors};
    use serde_json::json;

    #[derive(Debug)]
    struct Args {
        model_path: PathBuf,
        tensor_base: String,
        seed: u32,
        output: PathBuf,
    }

    pub(crate) fn main() -> Result<()> {
        let args = parse_args()?;
        let packed_key = format!("{}.tq_packed", args.tensor_base);
        let scales_key = format!("{}.tq_scales", args.tensor_base);
        let signs_key = format!("{}.tq_signs", args.tensor_base);

        let packed_tensor = load_tensor(&args.model_path, &packed_key)?;
        let scales_tensor = load_tensor(&args.model_path, &scales_key)?;
        let signs_tensor = load_tensor(&args.model_path, &signs_key)?;

        if packed_tensor.dtype != Dtype::U8 {
            bail!(
                "{} dtype must be U8, got {:?}",
                packed_key,
                packed_tensor.dtype
            );
        }
        if scales_tensor.dtype != Dtype::F16 {
            bail!(
                "{} dtype must be F16, got {:?}",
                scales_key,
                scales_tensor.dtype
            );
        }
        if signs_tensor.dtype != Dtype::I8 {
            bail!(
                "{} dtype must be I8, got {:?}",
                signs_key,
                signs_tensor.dtype
            );
        }
        if packed_tensor.shape.len() != 2 || scales_tensor.shape.len() != 2 {
            bail!("packed and scales tensors must be rank-2");
        }
        if signs_tensor.shape.len() != 1 {
            bail!("signs tensor must be rank-1");
        }

        let rows = packed_tensor.shape[0];
        let packed_cols = packed_tensor.shape[1];
        let num_groups = scales_tensor.shape[1];
        let bits = turboquant_bits(&args.model_path)?;
        let group_size = turboquant_group_size(&args.model_path)?;
        let effective_bits = if bits == 3 { 4 } else { bits };
        let orig_k = packed_cols.checked_mul(8).context("packed cols overflow")? / effective_bits;
        if orig_k != signs_tensor.shape[0] {
            bail!(
                "inferred K={} does not match signs shape {}",
                orig_k,
                signs_tensor.shape[0]
            );
        }
        if orig_k / group_size != num_groups {
            bail!(
                "K/group_size mismatch: K={} group_size={} num_groups={}",
                orig_k,
                group_size,
                num_groups
            );
        }

        let scales_u16 = bytes_to_u16(&scales_tensor.data)?;
        let signs_host = bytes_to_i8(&signs_tensor.data);
        let centroids_host = lloyd_max_centroids(bits, group_size);
        let input_host = deterministic_input(orig_k, args.seed);

        let ctx = DeviceContext::new()?;
        let packed_gpu = ctx
            .stream
            .clone_htod(&packed_tensor.data)
            .context("H2D packed")?;
        let scales_gpu = ctx.stream.clone_htod(&scales_u16).context("H2D scales")?;
        let signs_gpu = ctx.stream.clone_htod(&signs_host).context("H2D signs")?;
        let centroids_gpu = ctx
            .stream
            .clone_htod(&centroids_host)
            .context("H2D centroids")?;
        let input_gpu = ctx
            .stream
            .clone_htod(&input_host)
            .context("H2D deterministic input")?;
        let mut fused_out: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros_traced(rows)
            .context("alloc fused output")?;
        let mut reference_out: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros_traced(rows)
            .context("alloc reference output")?;
        let mut dequant_workspace: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros_traced(rows * orig_k)
            .context("alloc dequant workspace")?;

        {
            let (packed_ptr, _g1) = packed_gpu.device_ptr(&ctx.stream);
            let (scales_ptr, _g2) = scales_gpu.device_ptr(&ctx.stream);
            let (signs_ptr, _g3) = signs_gpu.device_ptr(&ctx.stream);
            let (centroids_ptr, _g4) = centroids_gpu.device_ptr(&ctx.stream);
            let (input_ptr, _gx) = input_gpu.device_ptr(&ctx.stream);
            let (fused_ptr, _gy) = fused_out.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::turboquant_weight_gemv_cuda(
                    packed_ptr as *const u8,
                    scales_ptr as *const ffi::Half,
                    signs_ptr as *const i8,
                    centroids_ptr as *const f32,
                    input_ptr as *const ffi::Half,
                    fused_ptr as *mut ffi::Half,
                    rows as i32,
                    orig_k as i32,
                    group_size as i32,
                    packed_cols as i32,
                    num_groups as i32,
                    bits as i32,
                    ctx.stream.cu_stream(),
                );
            }
        }

        {
            let (packed_ptr, _g1) = packed_gpu.device_ptr(&ctx.stream);
            let (scales_ptr, _g2) = scales_gpu.device_ptr(&ctx.stream);
            let (signs_ptr, _g3) = signs_gpu.device_ptr(&ctx.stream);
            let (centroids_ptr, _g4) = centroids_gpu.device_ptr(&ctx.stream);
            let (workspace_ptr, _gw) = dequant_workspace.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::turboquant_weight_dequant_cuda(
                    packed_ptr as *const u8,
                    scales_ptr as *const ffi::Half,
                    signs_ptr as *const i8,
                    centroids_ptr as *const f32,
                    workspace_ptr as *mut ffi::Half,
                    rows as i32,
                    orig_k as i32,
                    group_size as i32,
                    packed_cols as i32,
                    num_groups as i32,
                    bits as i32,
                    ctx.stream.cu_stream(),
                );
            }
        }

        {
            let (workspace_ptr, _gw) = dequant_workspace.device_ptr(&ctx.stream);
            let (input_ptr, _gx) = input_gpu.device_ptr(&ctx.stream);
            let (reference_ptr, _gy) = reference_out.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::gemm_cuda(
                    workspace_ptr as *const ffi::Half,
                    input_ptr as *const ffi::Half,
                    reference_ptr as *mut ffi::Half,
                    rows as i32,
                    1,
                    orig_k as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .context("bulk dequant + cuBLAS GEMM failed")?;
            }
        }

        ctx.sync()?;
        let fused_host = ctx.stream.clone_dtoh(&fused_out).context("D2H fused")?;
        let reference_host = ctx
            .stream
            .clone_dtoh(&reference_out)
            .context("D2H reference")?;
        let fused_values: Vec<f32> = fused_host.iter().map(|value| value.to_f32()).collect();
        let reference_values: Vec<f32> =
            reference_host.iter().map(|value| value.to_f32()).collect();
        let report = compare(&fused_values, &reference_values);

        let first8: Vec<_> = (0..8.min(rows))
            .map(|idx| {
                json!({
                    "index": idx,
                    "fused_gemv": fused_values[idx],
                    "bulk_dequant_cublas": reference_values[idx],
                    "abs_err": (fused_values[idx] - reference_values[idx]).abs(),
                    "rel_err": (fused_values[idx] - reference_values[idx]).abs()
                        / reference_values[idx].abs().max(1.0e-6),
                })
            })
            .collect();

        let payload = json!({
            "model_path": args.model_path,
            "tensor_base": args.tensor_base,
            "seed": args.seed,
            "input": {
                "shape": [1, orig_k],
                "first8": input_host.iter().take(8).map(|value| value.to_f32()).collect::<Vec<_>>(),
            },
            "projection": {
                "rows": rows,
                "cols": orig_k,
                "bits": bits,
                "group_size": group_size,
                "packed_cols": packed_cols,
                "num_groups": num_groups,
            },
            "comparison": report,
            "first8": first8,
        });
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

    fn shard_for_key(model_path: &Path, key: &str) -> Result<String> {
        let index_path = model_path.join("model.safetensors.index.json");
        let raw = fs::read_to_string(&index_path)
            .with_context(|| format!("read {}", index_path.display()))?;
        let index: serde_json::Value = serde_json::from_str(&raw)
            .with_context(|| format!("parse {}", index_path.display()))?;
        let shard = index
            .get("weight_map")
            .and_then(|wm| wm.get(key))
            .and_then(|value| value.as_str())
            .with_context(|| format!("missing {key} in {}", index_path.display()))?;
        Ok(shard.to_string())
    }

    fn turboquant_bits(model_path: &Path) -> Result<usize> {
        let raw = fs::read_to_string(model_path.join("turboquant_config.json"))
            .context("read turboquant_config.json")?;
        let config: serde_json::Value =
            serde_json::from_str(&raw).context("parse turboquant_config.json")?;
        config
            .get("bits")
            .and_then(|value| value.as_u64())
            .map(|value| value as usize)
            .context("turboquant_config.json missing bits")
    }

    fn turboquant_group_size(model_path: &Path) -> Result<usize> {
        let raw = fs::read_to_string(model_path.join("turboquant_config.json"))
            .context("read turboquant_config.json")?;
        let config: serde_json::Value =
            serde_json::from_str(&raw).context("parse turboquant_config.json")?;
        config
            .get("group_size")
            .and_then(|value| value.as_u64())
            .map(|value| value as usize)
            .context("turboquant_config.json missing group_size")
    }

    fn bytes_to_u16(data: &[u8]) -> Result<Vec<u16>> {
        if data.len() % 2 != 0 {
            bail!("u16 tensor byte length must be even, got {}", data.len());
        }
        Ok(data
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect())
    }

    fn bytes_to_i8(data: &[u8]) -> Vec<i8> {
        data.iter().map(|byte| *byte as i8).collect()
    }

    fn lloyd_max_centroids(bits: usize, group_size: usize) -> Vec<f32> {
        let num_levels = 1usize << bits;
        let mut centroids = vec![0.0f32; num_levels];
        let mut boundaries = vec![0.0f32; num_levels + 1];
        unsafe {
            ffi::turboquant_lloyd_max(
                centroids.as_mut_ptr(),
                boundaries.as_mut_ptr(),
                num_levels as i32,
                group_size as i32,
                200,
            );
        }
        centroids
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

    fn compare(lhs: &[f32], rhs: &[f32]) -> serde_json::Value {
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
            "lhs": "fused_turboquant_gemv",
            "rhs": "bulk_dequant_cublas",
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

    fn parse_args() -> Result<Args> {
        let mut model_path = None;
        let mut tensor_base = Some("model.language_model.layers.0.mlp.gate_proj".to_string());
        let mut seed = 0x5eed1234_u32;
        let mut output = None;

        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model-path" => model_path = Some(PathBuf::from(next_arg(&mut args, &arg)?)),
                "--tensor-base" => tensor_base = Some(next_arg(&mut args, &arg)?),
                "--seed" => seed = next_arg(&mut args, &arg)?.parse()?,
                "--output" => output = Some(PathBuf::from(next_arg(&mut args, &arg)?)),
                "--help" | "-h" => {
                    println!(
                        "usage: cargo run -p infer --example turboquant_weight_gemv_parity --release --features cuda -- \
                         --model-path DIR [--tensor-base model.language_model.layers.0.mlp.gate_proj] \
                         [--seed 1592594996] --output parity.json"
                    );
                    std::process::exit(0);
                }
                _ => bail!("unknown argument `{arg}`"),
            }
        }
        Ok(Args {
            model_path: model_path.context("--model-path is required")?,
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
    eprintln!("turboquant_weight_gemv_parity requires --features cuda");
}
