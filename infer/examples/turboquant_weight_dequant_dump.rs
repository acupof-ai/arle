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
    use memmap2::Mmap;
    use safetensors::{Dtype, SafeTensors};
    use serde_json::json;

    #[derive(Debug)]
    struct Args {
        model_path: PathBuf,
        tensor_base: String,
        row_start: usize,
        row_count: usize,
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
        if args.row_count == 0 || args.row_start + args.row_count > rows {
            bail!(
                "invalid row slice start={} count={} for rows={}",
                args.row_start,
                args.row_count,
                rows
            );
        }

        let packed_row_bytes = packed_cols;
        let packed_start = args.row_start * packed_row_bytes;
        let packed_end = packed_start + args.row_count * packed_row_bytes;
        let packed_host = &packed_tensor.data[packed_start..packed_end];

        let scales_u16 = bytes_to_u16(&scales_tensor.data)?;
        let scale_start = args.row_start * num_groups;
        let scale_end = scale_start + args.row_count * num_groups;
        let scales_host = &scales_u16[scale_start..scale_end];

        let signs_host = bytes_to_i8(&signs_tensor.data);
        let centroids_host = lloyd_max_centroids(bits, group_size);

        let ctx = DeviceContext::new()?;
        let packed_gpu = ctx.stream.clone_htod(packed_host).context("H2D packed")?;
        let scales_gpu = ctx.stream.clone_htod(scales_host).context("H2D scales")?;
        let signs_gpu = ctx.stream.clone_htod(&signs_host).context("H2D signs")?;
        let centroids_gpu = ctx
            .stream
            .clone_htod(&centroids_host)
            .context("H2D centroids")?;
        let mut out_gpu: CudaSlice<u16> = ctx
            .stream
            .alloc_zeros_traced(args.row_count * orig_k)
            .context("alloc output")?;

        {
            let (packed_ptr, _g1) = packed_gpu.device_ptr(&ctx.stream);
            let (scales_ptr, _g2) = scales_gpu.device_ptr(&ctx.stream);
            let (signs_ptr, _g3) = signs_gpu.device_ptr(&ctx.stream);
            let (centroids_ptr, _g4) = centroids_gpu.device_ptr(&ctx.stream);
            let (out_ptr, _g5) = out_gpu.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::turboquant_weight_dequant_cuda(
                    packed_ptr as *const u8,
                    scales_ptr as *const ffi::Half,
                    signs_ptr as *const i8,
                    centroids_ptr as *const f32,
                    out_ptr as *mut ffi::Half,
                    args.row_count as i32,
                    orig_k as i32,
                    group_size as i32,
                    packed_cols as i32,
                    num_groups as i32,
                    bits as i32,
                    ctx.stream.cu_stream(),
                );
            }
        }
        ctx.sync()?;
        let out_bits = ctx.stream.clone_dtoh(&out_gpu).context("D2H output")?;
        let values: Vec<f32> = out_bits
            .iter()
            .map(|bits| half::bf16::from_bits(*bits).to_f32())
            .collect();

        let payload = json!({
            "model_path": args.model_path,
            "tensor_base": args.tensor_base,
            "row_start": args.row_start,
            "row_count": args.row_count,
            "shape": [args.row_count, orig_k],
            "bits": bits,
            "group_size": group_size,
            "packed_cols": packed_cols,
            "num_groups": num_groups,
            "centroids": centroids_host,
            "values": values,
        });
        fs::write(&args.output, serde_json::to_vec(&payload)?)?;
        println!(
            "turboquant_weight_dequant_dump tensor={} rows={}..{} shape=[{},{}] output={}",
            args.tensor_base,
            args.row_start,
            args.row_start + args.row_count,
            args.row_count,
            orig_k,
            args.output.display()
        );
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

    fn parse_args() -> Result<Args> {
        let mut model_path = None;
        let mut tensor_base = None;
        let mut row_start = 0usize;
        let mut row_count = 8usize;
        let mut output = None;

        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model-path" => model_path = Some(PathBuf::from(next_arg(&mut args, &arg)?)),
                "--tensor-base" => tensor_base = Some(next_arg(&mut args, &arg)?),
                "--row-start" => row_start = next_arg(&mut args, &arg)?.parse()?,
                "--row-count" => row_count = next_arg(&mut args, &arg)?.parse()?,
                "--output" => output = Some(PathBuf::from(next_arg(&mut args, &arg)?)),
                "--help" | "-h" => {
                    println!(
                        "usage: cargo run -p infer --example turboquant_weight_dequant_dump --release --features cuda -- \
                         --model-path DIR --tensor-base model.language_model.layers.1.mlp.gate_proj \
                         --row-start 0 --row-count 8 --output cuda-dequant.json"
                    );
                    std::process::exit(0);
                }
                _ => bail!("unknown argument `{arg}`"),
            }
        }
        Ok(Args {
            model_path: model_path.context("--model-path is required")?,
            tensor_base: tensor_base.context("--tensor-base is required")?,
            row_start,
            row_count,
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
    eprintln!("turboquant_weight_dequant_dump requires --features cuda");
}
