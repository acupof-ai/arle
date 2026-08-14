#![cfg_attr(
    not(all(feature = "cuda", not(feature = "no-cuda"))),
    allow(dead_code, unused_imports)
)]

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
mod app {
    use std::{
        env,
        path::{Path, PathBuf},
        sync::{Arc, Mutex},
        time::Instant,
    };

    use anyhow::{Context, Result, bail};
    use autograd::{Backend, TensorId, TensorStore, backend_cuda::CudaBackend};
    use infer_api::{EngineLoadConfig, LoadedInferenceEngine};
    use train::{
        LoraConfig, LoraTargetSet, infer_student::InferStudent, qwen35::Qwen35Model,
        qwen35_loader::load_qwen35_lora_from_hf_dir, tokenizer::ChatTokenizer,
    };

    const DEFAULT_MODEL_DIR: &str = "/data01/models/Qwen3.6-35B-A3B-FP8";

    #[derive(Debug)]
    struct Args {
        model: PathBuf,
        device: usize,
        lora: LoraConfig,
        target_set: LoraTargetSet,
        sync_infer: bool,
        infer_model: PathBuf,
        perturb_adapter: Option<String>,
        rollout_smoke_prompt: Option<String>,
        rollout_smoke_tokens: usize,
        expect_substring: Option<String>,
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
        let live_host_mib = store.live_host_bytes() as f64 / (1024.0 * 1024.0);

        println!(
            "qwen36_fp8_lora_load_gate_result load_seconds={load_seconds:.6} \
             total_vram_mib={:.1} used_delta_mib={:.1} free_before_mib={:.1} free_after_mib={:.1} \
             live_host_mib={live_host_mib:.1} \
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

        if args.sync_infer {
            if let Some(name) = args.perturb_adapter.as_deref() {
                perturb_adapter(&mut store, &student, name)?;
            }
            let infer_started = Instant::now();
            let infer_student = load_infer_student(
                &args.infer_model,
                backend.clone(),
                cfg.vocab_size,
                args.model.display().to_string(),
            )?;
            let infer_load_seconds = infer_started.elapsed().as_secs_f64();
            let sync_started = Instant::now();
            infer_student
                .sync_lora_from_store(
                    &mut store,
                    &student.adapter_name_map(),
                    &student.param_name_map(),
                    args.lora,
                )
                .context("sync LoRA into infer student")?;
            backend
                .device_synchronize()
                .context("synchronize after infer LoRA sync")?;
            let sync_seconds = sync_started.elapsed().as_secs_f64();
            println!(
                "qwen36_fp8_lora_sync_gate_result infer_load_seconds={infer_load_seconds:.6} \
                 sync_seconds={sync_seconds:.6} perturb_adapter={}",
                args.perturb_adapter.as_deref().unwrap_or("none")
            );
            if let Some(prompt) = args.rollout_smoke_prompt.as_deref() {
                run_rollout_smoke(
                    &infer_student,
                    &args.infer_model,
                    prompt,
                    args.rollout_smoke_tokens,
                    args.expect_substring.as_deref(),
                )?;
            }
        }

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
        let mut sync_infer = false;
        let mut infer_model = model.clone();
        let mut perturb_adapter =
            Some("model.language_model.layers.0.mlp.experts.0.up_proj.weight.lora_b".to_string());
        let mut rollout_smoke_prompt = None;
        let mut rollout_smoke_tokens = 32usize;
        let mut expect_substring = None;

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
                "--sync-infer" => {
                    sync_infer = true;
                }
                "--infer-model" => {
                    infer_model = PathBuf::from(next_arg("--infer-model", &mut args)?);
                }
                "--perturb-adapter" => {
                    let raw = next_arg("--perturb-adapter", &mut args)?;
                    perturb_adapter = if raw == "none" { None } else { Some(raw) };
                }
                "--rollout-smoke-prompt" => {
                    rollout_smoke_prompt = Some(next_arg("--rollout-smoke-prompt", &mut args)?);
                }
                "--rollout-smoke-tokens" => {
                    rollout_smoke_tokens = next_arg("--rollout-smoke-tokens", &mut args)?
                        .parse()
                        .context("parse --rollout-smoke-tokens")?;
                }
                "--expect-substring" => {
                    expect_substring = Some(next_arg("--expect-substring", &mut args)?);
                }
                "-h" | "--help" => {
                    println!(
                        "usage: cargo run -p train --example qwen36_fp8_lora_load_gate --release --features cuda -- \
                         [--model DIR] [--device N] [--rank N] [--alpha F] \
                         [--target-set all-linear|attention-qv] [--sync-infer] \
                         [--infer-model DIR] [--perturb-adapter NAME|none] \
                         [--rollout-smoke-prompt TEXT] [--rollout-smoke-tokens N] \
                         [--expect-substring TEXT]"
                    );
                    std::process::exit(0);
                }
                other => bail!("unknown argument {other:?}"),
            }
        }

        if rank == 0 || !alpha.is_finite() || alpha <= 0.0 {
            bail!("rank must be >0 and alpha must be positive finite");
        }
        if rollout_smoke_tokens == 0 {
            bail!("--rollout-smoke-tokens must be > 0");
        }
        if rollout_smoke_prompt.is_some() && !sync_infer {
            bail!("--rollout-smoke-prompt requires --sync-infer");
        }
        Ok(Args {
            model,
            device,
            lora: LoraConfig { rank, alpha },
            target_set,
            sync_infer,
            infer_model,
            perturb_adapter,
            rollout_smoke_prompt,
            rollout_smoke_tokens,
            expect_substring,
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

    fn perturb_adapter(
        store: &mut TensorStore,
        student: &Qwen35Model,
        adapter_name: &str,
    ) -> Result<()> {
        let adapters = student.adapter_name_map();
        let id = *adapters
            .get(adapter_name)
            .with_context(|| format!("adapter {adapter_name} not found"))?;
        let tensor = store
            .get_mut(id)
            .with_context(|| format!("adapter tensor {adapter_name} id {id:?} missing"))?;
        let slot = tensor
            .data
            .first_mut()
            .with_context(|| format!("adapter tensor {adapter_name} is empty"))?;
        *slot = 1.0e-3;
        tensor.device_handle = None;
        println!("qwen36_fp8_lora_sync_gate_perturbed adapter={adapter_name} value={slot:.6e}");
        Ok(())
    }

    fn load_infer_student(
        model: &Path,
        train_backend: Arc<dyn Backend>,
        vocab_size: usize,
        label: String,
    ) -> Result<InferStudent> {
        let max_seq_len = 512usize;
        let page_size = 16usize;
        let engine = LoadedInferenceEngine::load_with_config(
            model
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("infer model path is not valid UTF-8"))?,
            true,
            EngineLoadConfig {
                num_slots: 1,
                page_size,
                total_pages: max_seq_len.div_ceil(page_size),
                max_prompt_tokens: max_seq_len,
                max_total_tokens: max_seq_len,
                chunked_prefill_size: Some(max_seq_len),
                ..EngineLoadConfig::default()
            tp_size: None,
        },
        )
        .with_context(|| format!("load infer rollout student from {}", model.display()))?;
        println!("qwen36_fp8_lora_sync_gate_infer_loaded model={label}");
        Ok(InferStudent::new(
            Arc::new(Mutex::new(engine)),
            train_backend,
            vocab_size,
        ))
    }

    fn run_rollout_smoke(
        infer_student: &InferStudent,
        model: &Path,
        prompt: &str,
        rollout_tokens: usize,
        expect_substring: Option<&str>,
    ) -> Result<()> {
        let tokenizer_path = model.join("tokenizer.json");
        let tokenizer = ChatTokenizer::from_file(&tokenizer_path)
            .map_err(|err| anyhow::anyhow!("load tokenizer {}: {err}", tokenizer_path.display()))?;
        let prompt_ids = tokenizer
            .encode(prompt, false)
            .map_err(|err| anyhow::anyhow!("encode rollout smoke prompt: {err}"))?;
        let started = Instant::now();
        let rollout = infer_student
            .generate_rollout(&prompt_ids, rollout_tokens, None)
            .context("generate rollout smoke through InferStudent")?;
        let generated = &rollout[prompt_ids.len()..];
        let generated_text = tokenizer
            .decode(generated, true)
            .map_err(|err| anyhow::anyhow!("decode generated rollout smoke: {err}"))?;
        let full_text = tokenizer
            .decode(&rollout, true)
            .map_err(|err| anyhow::anyhow!("decode full rollout smoke: {err}"))?;
        let smoke_seconds = started.elapsed().as_secs_f64();
        let contains_expect = expect_substring
            .map(|needle| generated_text.contains(needle))
            .unwrap_or(true);
        println!(
            "qwen36_fp8_lora_rollout_smoke_result prompt_tokens={} generated_tokens={} \
             smoke_seconds={smoke_seconds:.6} expect={:?} contains_expect={} \
             generated_text={:?} full_text={:?}",
            prompt_ids.len(),
            generated.len(),
            expect_substring,
            contains_expect,
            generated_text,
            full_text
        );
        if !contains_expect {
            bail!(
                "rollout smoke output did not contain expected substring {:?}",
                expect_substring
            );
        }
        Ok(())
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
