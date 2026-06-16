#![cfg_attr(
    not(all(feature = "cuda", not(feature = "no-cuda"))),
    allow(dead_code, unused_imports)
)]

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
mod app {
    use std::{env, path::PathBuf, sync::Arc, time::Instant};

    use anyhow::{Context, Result, bail};
    use autograd::{Backend, TensorId, TensorStore, backend_cuda::CudaBackend};
    use train::{
        LoraConfig, LoraTargetSet, qwen35::Qwen35Model, qwen35_loader::load_qwen35_lora_from_hf_dir,
    };

    const DEFAULT_MODEL_DIR: &str = "/data01/models/Qwen3.6-35B-A3B-FP8";

    #[derive(Debug)]
    struct Args {
        model: PathBuf,
        device: usize,
        lora: LoraConfig,
        target_set: LoraTargetSet,
    }

    pub fn main() -> Result<()> {
        let args = parse_args()?;
        println!(
            "qwen36_fp8_lora_load_gate_start model={} device={} rank={} alpha={:.6} target_set={}",
            args.model.display(),
            args.device,
            args.lora.rank,
            args.lora.alpha,
            args.target_set.label()
        );

        let backend = Arc::new(
            CudaBackend::new(args.device)
                .with_context(|| format!("init CUDA backend device {}", args.device))?,
        );
        let (free_before, total) = backend.mem_get_info().context("cuda mem before load")?;
        let mut store = TensorStore::with_backend(backend.clone());

        let started = Instant::now();
        let student =
            load_qwen35_lora_from_hf_dir(&args.model, args.lora, args.target_set, &mut store)
                .with_context(|| format!("load LoRA student from {}", args.model.display()))?;
        backend
            .device_synchronize()
            .context("synchronize after student load")?;
        let load_seconds = started.elapsed().as_secs_f64();
        let (free_after, _) = backend.mem_get_info().context("cuda mem after load")?;

        let all_params = student.all_parameter_ids();
        let trainable_params: Vec<TensorId> = all_params
            .iter()
            .copied()
            .filter(|id| store.get(*id).is_some_and(|tensor| tensor.requires_grad))
            .collect();
        let base_params = all_params.len().saturating_sub(trainable_params.len());
        let adapter_count = student.adapter_name_map().len();
        let cfg = student.config();

        println!(
            "qwen36_fp8_lora_load_gate_result load_seconds={load_seconds:.6} \
             total_vram_mib={:.1} used_delta_mib={:.1} free_before_mib={:.1} free_after_mib={:.1} \
             hidden={} layers={} vocab={} experts={} topk={} moe_intermediate={} shared_intermediate={} \
             all_param_tensors={} frozen_param_tensors={} trainable_param_tensors={} trainable_elements={} adapters={}",
            total as f64 / (1024.0 * 1024.0),
            free_before.saturating_sub(free_after) as f64 / (1024.0 * 1024.0),
            free_before as f64 / (1024.0 * 1024.0),
            free_after as f64 / (1024.0 * 1024.0),
            cfg.hidden_size,
            cfg.num_hidden_layers,
            cfg.vocab_size,
            cfg.num_experts,
            cfg.num_experts_per_tok,
            cfg.moe_intermediate_size,
            cfg.shared_expert_intermediate_size,
            all_params.len(),
            base_params,
            trainable_params.len(),
            element_count(&trainable_params, &store),
            adapter_count
        );

        Ok(())
    }

    fn parse_args() -> Result<Args> {
        let mut model = env::var_os("ARLE_QWEN36_FP8_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_DIR));
        let mut device = env::var("ARLE_CUDA_TEST_DEVICE")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
        let mut rank = 8usize;
        let mut alpha = 16.0f32;
        let mut target_set = LoraTargetSet::AllLinear;

        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model" => {
                    model = PathBuf::from(next_arg("--model", &mut args)?);
                }
                "--device" => {
                    device = next_arg("--device", &mut args)?
                        .parse()
                        .context("parse --device")?;
                }
                "--rank" => {
                    rank = next_arg("--rank", &mut args)?
                        .parse()
                        .context("parse --rank")?;
                }
                "--alpha" => {
                    alpha = next_arg("--alpha", &mut args)?
                        .parse()
                        .context("parse --alpha")?;
                }
                "--target-set" => {
                    target_set = parse_target_set(&next_arg("--target-set", &mut args)?)?;
                }
                "-h" | "--help" => {
                    println!(
                        "usage: cargo run -p train --example qwen36_fp8_lora_load_gate --release --features cuda -- \
                         [--model DIR] [--device N] [--rank N] [--alpha F] [--target-set all-linear|attention-qv]"
                    );
                    std::process::exit(0);
                }
                other => bail!("unknown argument {other:?}"),
            }
        }

        if rank == 0 || !alpha.is_finite() || alpha <= 0.0 {
            bail!("rank must be >0 and alpha must be positive finite");
        }
        Ok(Args {
            model,
            device,
            lora: LoraConfig { rank, alpha },
            target_set,
        })
    }

    fn next_arg(flag: &'static str, args: &mut impl Iterator<Item = String>) -> Result<String> {
        args.next()
            .with_context(|| format!("{flag} requires a value"))
    }

    fn parse_target_set(raw: &str) -> Result<LoraTargetSet> {
        match raw {
            "all-linear" | "all_linear" | "all" => Ok(LoraTargetSet::AllLinear),
            "attention-qv" | "attention_qv" | "qv" => Ok(LoraTargetSet::AttentionQv),
            other => bail!("unknown target set {other:?}"),
        }
    }

    fn element_count(ids: &[TensorId], store: &TensorStore) -> usize {
        ids.iter()
            .filter_map(|id| store.get(*id).map(|tensor| tensor.size))
            .sum()
    }
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
fn main() -> anyhow::Result<()> {
    app::main()
}

#[cfg(not(all(feature = "cuda", not(feature = "no-cuda"))))]
fn main() {
    eprintln!(
        "qwen36_fp8_lora_load_gate requires a real CUDA build: \
         cargo run -p train --example qwen36_fp8_lora_load_gate --release --features cuda"
    );
}
