#![cfg_attr(
    not(all(feature = "cuda", not(feature = "no-cuda"))),
    allow(dead_code, unused_imports)
)]

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
mod app {
    use std::{
        collections::HashSet,
        env,
        path::PathBuf,
        sync::Arc,
        time::{Duration, Instant},
    };

    use anyhow::{Context, Result, bail};
    use autograd::{
        Backend, BackwardProfile, Tape, Tensor, TensorId, TensorStore,
        backend_cuda::CudaBackend,
        ops::{mul, sum},
    };
    use train::{
        LoraConfig, LoraTargetSet,
        qwen35::{
            Qwen35AttentionForwardProfile, Qwen35Model, Qwen35MoeRouteSignature,
            Qwen35RolloutForwardProfile,
        },
        qwen35_loader::load_qwen35_lora_from_hf_dir,
    };

    const DEFAULT_MODEL_DIR: &str = "/data01/models/Qwen3.6-35B-A3B-FP8";
    const DEFAULT_TARGET_ADAPTER: &str =
        "model.language_model.layers.0.mlp.shared_expert.up_proj.weight.lora_b";
    const SPARSE_LOGIT_PROBE_PER_ROW: usize = 64;

    #[derive(Debug)]
    struct Args {
        model: PathBuf,
        device: usize,
        lora: LoraConfig,
        target_set: LoraTargetSet,
        target_adapter: String,
        eps: f32,
        tokens: Vec<u32>,
        mode: GateMode,
        layer: usize,
        profile_backward: bool,
        profile_forward_only: bool,
        check_route_stability: bool,
        freeze_base_routes: bool,
        sparse_logit_probe: bool,
        frozen_prompt_kv: bool,
        prompt_len: usize,
        gdr_chunkwise: bool,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum GateMode {
        MlpLayer,
        FullModel,
        Tail,
    }

    struct GateInput {
        hidden: Option<TensorId>,
        probe: TensorId,
    }

    pub fn main() -> Result<()> {
        let args = parse_args()?;
        // autograd flags come from apply_runtime_flags, never the environment, so
        // the FlashQLA GDN backward only engages when this is set explicitly.
        autograd::apply_runtime_flags(&autograd::AutogradRuntimeFlags {
            gdr_chunkwise_prefill: args.gdr_chunkwise,
            ..Default::default()
        });
        if args.target_set != LoraTargetSet::AllLinear {
            bail!(
                "finite-diff gate requires --target-set all-linear so the default Qwen3.6 adapter exists"
            );
        }
        println!(
            "qwen36_fp8_lora_fd_gate_start model={} device={} rank={} alpha={:.6} \
             target_set={} target_adapter={} eps={:.1e} tokens={:?} mode={} layer={} \
             profile_backward={} profile_forward_only={} check_route_stability={} \
             freeze_base_routes={} sparse_logit_probe={} gdr_chunkwise={}",
            args.model.display(),
            args.device,
            args.lora.rank,
            args.lora.alpha,
            args.target_set.label(),
            args.target_adapter,
            args.eps,
            args.tokens,
            args.mode.label(),
            args.layer,
            args.profile_backward,
            args.profile_forward_only,
            args.check_route_stability,
            args.freeze_base_routes,
            args.sparse_logit_probe,
            args.gdr_chunkwise
        );

        let backend = Arc::new(
            CudaBackend::new(args.device)
                .with_context(|| format!("init CUDA backend device {}", args.device))?,
        );
        let mut store = TensorStore::with_backend(backend.clone());
        let load_started = Instant::now();
        let student =
            load_qwen35_lora_from_hf_dir(&args.model, args.lora, args.target_set, &mut store)
                .with_context(|| format!("load LoRA student from {}", args.model.display()))?;
        backend
            .device_synchronize()
            .context("synchronize after student load")?;
        let load_elapsed = load_started.elapsed();

        let positions = (0..args.tokens.len() as u32).collect::<Vec<_>>();
        if args.profile_forward_only {
            profile_forward_only(&student, &mut store, &args, &positions)?;
            return Ok(());
        }

        if args.frozen_prompt_kv {
            // Runtime check deferred to the pod increment (carry-aware device
            // kernel + a real FP8 checkpoint). Here we wire + compile the gate:
            // an FD check on the frozen gen-segment forward at a response-position
            // probe. The host taped carry is gated by Gate A's parity tests; this
            // pod-side gate confirms it on the production FP8 model.
            run_frozen_prompt_kv_gate(&student, &mut store, &args, &backend)?;
            return Ok(());
        }

        let keep_before_probe = student
            .all_parameter_ids()
            .into_iter()
            .collect::<HashSet<_>>();

        let gate_input = make_gate_input(&student, &mut store, &args, &positions)?;
        let mut keep = keep_before_probe;
        if let Some(hidden) = gate_input.hidden {
            keep.insert(hidden);
        }
        keep.insert(gate_input.probe);
        store.retain_ids(&keep);

        let frozen_routes = if args.freeze_base_routes {
            let routes = collect_base_moe_routes(&student, &mut store, &args.tokens, &positions)?;
            let total_slots = routes
                .iter()
                .map(|route| route.indices.len())
                .sum::<usize>();
            println!(
                "qwen36_fp8_lora_route_frozen base_layers={} total_slots={} \
                 tokens={} top_k={} experts={}",
                routes.len(),
                total_slots,
                args.tokens.len(),
                routes.first().map(|route| route.top_k).unwrap_or(0),
                routes.first().map(|route| route.experts).unwrap_or(0)
            );
            store.retain_ids(&keep);
            Some(routes)
        } else {
            None
        };
        let frozen_routes_ref = frozen_routes.as_deref();

        let analytic_started = Instant::now();
        let mut tape = Tape::new();
        let mut base_routes = Vec::new();
        let route_sink = args.check_route_stability.then_some(&mut base_routes);
        let loss = weighted_logits_loss(
            &student,
            &mut store,
            &mut tape,
            args.mode,
            args.layer,
            &args.tokens,
            &positions,
            gate_input.hidden,
            gate_input.probe,
            route_sink,
            frozen_routes_ref,
        )?;
        let loss_base = scalar_host(&mut store, loss)?;
        let (grads, backward_profile) = if args.profile_backward {
            let (grads, profile) = tape.backward_profiled(loss, &mut store)?;
            (grads, Some(profile))
        } else {
            (tape.backward(loss, &mut store)?, None)
        };
        backend
            .device_synchronize()
            .context("synchronize after analytic backward")?;
        let analytic_elapsed = analytic_started.elapsed();
        if let Some(profile) = &backward_profile {
            print_backward_profile(profile);
        }

        let selected = select_target_probe(&student, &grads, &mut store, &args, gate_input.hidden)?;
        store.retain_ids(&keep);

        let original = tensor_value(&mut store, selected.target, selected.index)?;
        let mut plus_routes = Vec::new();
        let plus_started = Instant::now();
        set_tensor_value(
            &mut store,
            selected.target,
            selected.index,
            original + args.eps,
        )?;
        let plus_route_sink = args.check_route_stability.then_some(&mut plus_routes);
        let loss_plus = loss_value(
            &student,
            &mut store,
            &args,
            &positions,
            &gate_input,
            plus_route_sink,
            frozen_routes_ref,
        )?;
        backend
            .device_synchronize()
            .context("synchronize after plus finite diff")?;
        let plus_elapsed = plus_started.elapsed();
        store.retain_ids(&keep);

        let mut minus_routes = Vec::new();
        let minus_started = Instant::now();
        set_tensor_value(
            &mut store,
            selected.target,
            selected.index,
            original - args.eps,
        )?;
        let minus_route_sink = args.check_route_stability.then_some(&mut minus_routes);
        let loss_minus = loss_value(
            &student,
            &mut store,
            &args,
            &positions,
            &gate_input,
            minus_route_sink,
            frozen_routes_ref,
        )?;
        backend
            .device_synchronize()
            .context("synchronize after minus finite diff")?;
        let minus_elapsed = minus_started.elapsed();
        set_tensor_value(&mut store, selected.target, selected.index, original)?;
        store.retain_ids(&keep);

        let numeric = (loss_plus - loss_minus) / (2.0 * args.eps);
        let rel_err = rel_err(selected.analytic, numeric);
        if args.check_route_stability {
            print_route_stability(&base_routes, &plus_routes, &minus_routes);
        }
        let live_host_mib = store.live_host_bytes() as f64 / (1024.0 * 1024.0);
        println!(
            "qwen36_fp8_lora_fd_gate_result load_seconds={:.6} analytic_seconds={:.6} \
             plus_seconds={:.6} minus_seconds={:.6} live_host_mib={live_host_mib:.1} \
             mode={} layer={} route_frozen={} sparse_logit_probe={} target={} index={} eps={:.1e} \
             loss_base={:.9e} loss_minus={:.9e} loss_plus={:.9e} \
             analytic={:.9e} numeric={:.9e} rel_err={:.3e}",
            secs(load_elapsed),
            secs(analytic_elapsed),
            secs(plus_elapsed),
            secs(minus_elapsed),
            args.mode.label(),
            args.layer,
            args.freeze_base_routes,
            args.sparse_logit_probe,
            selected.name,
            selected.index,
            args.eps,
            loss_base,
            loss_minus,
            loss_plus,
            selected.analytic,
            numeric,
            rel_err
        );
        if rel_err > 1.0e-2 {
            bail!(
                "Qwen3.6 FP8 LoRA finite diff failed: target={} analytic={:.9e} numeric={numeric:.9e} rel_err={rel_err:.3e}",
                selected.name,
                selected.analytic
            );
        }
        println!("qwen36_fp8_lora_fd_gate PASS");
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
        let mut target_adapter = DEFAULT_TARGET_ADAPTER.to_string();
        let mut eps = 1.0e-3_f32;
        let mut tokens = vec![1_u32, 3, 8];
        let mut mode = GateMode::MlpLayer;
        let mut layer = 0usize;
        let mut profile_backward = false;
        let mut profile_forward_only = false;
        let mut check_route_stability = false;
        let mut freeze_base_routes = false;
        let mut sparse_logit_probe = false;
        let mut frozen_prompt_kv = false;
        let mut prompt_len = 0usize;
        let mut gdr_chunkwise = false;

        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model" => model = PathBuf::from(next_arg("--model", &mut args)?),
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
                    target_set = parse_target_set(&next_arg("--target-set", &mut args)?)?
                }
                "--target-adapter" => target_adapter = next_arg("--target-adapter", &mut args)?,
                "--eps" => {
                    eps = next_arg("--eps", &mut args)?
                        .parse()
                        .context("parse --eps")?;
                }
                "--tokens" => tokens = parse_tokens(&next_arg("--tokens", &mut args)?)?,
                "--mode" => mode = parse_mode(&next_arg("--mode", &mut args)?)?,
                "--layer" => {
                    layer = next_arg("--layer", &mut args)?
                        .parse()
                        .context("parse --layer")?;
                }
                "--profile-backward" => profile_backward = true,
                "--profile-forward-only" => profile_forward_only = true,
                "--check-route-stability" => check_route_stability = true,
                "--freeze-base-routes" => freeze_base_routes = true,
                "--sparse-logit-probe" => sparse_logit_probe = true,
                "--frozen-prompt-kv" => frozen_prompt_kv = true,
                "--gdr-chunkwise" => gdr_chunkwise = true,
                "--prompt-len" => {
                    prompt_len = next_arg("--prompt-len", &mut args)?
                        .parse()
                        .context("parse --prompt-len")?;
                }
                "-h" | "--help" => {
                    println!(
                        "usage: cargo run -p train --example qwen36_fp8_lora_fd_gate \
                         --release --features cuda -- [--model DIR] [--device N] \
                         [--rank N] [--alpha F] [--target-set all-linear] \
                         [--target-adapter NAME|auto:routed-up] [--eps F] [--tokens CSV] \
                         [--mode mlp-layer|full-model|tail] [--layer N] [--profile-backward] \
                         [--profile-forward-only] [--check-route-stability] \
                         [--freeze-base-routes] [--sparse-logit-probe] \
                         [--frozen-prompt-kv --prompt-len N] [--gdr-chunkwise]"
                    );
                    std::process::exit(0);
                }
                other => bail!("unknown argument {other:?}"),
            }
        }
        if rank == 0 || !alpha.is_finite() || alpha <= 0.0 {
            bail!("rank must be >0 and alpha must be positive finite");
        }
        if !eps.is_finite() || eps <= 0.0 {
            bail!("--eps must be positive finite");
        }
        if tokens.is_empty() {
            bail!("--tokens must contain at least one token id");
        }
        if check_route_stability && mode != GateMode::FullModel {
            bail!("--check-route-stability requires --mode full-model");
        }
        if freeze_base_routes && mode != GateMode::FullModel {
            bail!("--freeze-base-routes requires --mode full-model");
        }
        if sparse_logit_probe && !matches!(mode, GateMode::FullModel | GateMode::Tail) {
            bail!("--sparse-logit-probe requires --mode full-model or --mode tail");
        }
        if frozen_prompt_kv {
            // gen_start = prompt_len - 1, so the prefix needs >= 1 token and the
            // gen segment (boundary ++ generated) needs >= 2 tokens total.
            if prompt_len < 2 {
                bail!("--frozen-prompt-kv requires --prompt-len >= 2");
            }
            if prompt_len >= tokens.len() {
                bail!(
                    "--frozen-prompt-kv requires --prompt-len < tokens ({} >= {})",
                    prompt_len,
                    tokens.len()
                );
            }
        }
        Ok(Args {
            model,
            device,
            lora: LoraConfig { rank, alpha },
            target_set,
            target_adapter,
            eps,
            tokens,
            mode,
            layer,
            profile_backward,
            profile_forward_only,
            check_route_stability,
            freeze_base_routes,
            sparse_logit_probe,
            frozen_prompt_kv,
            prompt_len,
            gdr_chunkwise,
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

    fn parse_tokens(raw: &str) -> Result<Vec<u32>> {
        raw.split(',')
            .map(|part| {
                let trimmed = part.trim();
                if trimmed.is_empty() {
                    bail!("empty token in --tokens {raw:?}");
                }
                trimmed
                    .parse::<u32>()
                    .with_context(|| format!("parse token {trimmed:?}"))
            })
            .collect()
    }

    fn parse_mode(raw: &str) -> Result<GateMode> {
        match raw {
            "mlp-layer" | "mlp" => Ok(GateMode::MlpLayer),
            "full-model" | "full" => Ok(GateMode::FullModel),
            "tail" | "lm-head-tail" => Ok(GateMode::Tail),
            other => bail!("unknown mode {other:?}"),
        }
    }

    impl GateMode {
        fn label(self) -> &'static str {
            match self {
                Self::MlpLayer => "mlp-layer",
                Self::FullModel => "full-model",
                Self::Tail => "tail",
            }
        }
    }

    struct SelectedProbe {
        name: String,
        target: TensorId,
        index: usize,
        analytic: f32,
    }

    fn select_target_probe(
        model: &Qwen35Model,
        grads: &std::collections::HashMap<TensorId, TensorId>,
        store: &mut TensorStore,
        args: &Args,
        hidden: Option<TensorId>,
    ) -> Result<SelectedProbe> {
        if args.mode == GateMode::Tail {
            let target = hidden.context("tail mode missing hidden target")?;
            let grad_id = *grads
                .get(&target)
                .context("missing gradient for diagnostic tail hidden")?;
            let grad = store.to_host(grad_id)?;
            let index = largest_nonzero_index(&grad)
                .context("gradient for diagnostic tail hidden has no non-zero element")?;
            return Ok(SelectedProbe {
                name: "diagnostic.tail_hidden".to_string(),
                target,
                index,
                analytic: grad[index],
            });
        }
        if args.target_adapter == "auto:routed-up" {
            return select_auto_routed_up_probe(model, grads, store, args.layer);
        }
        let target = *model
            .adapter_name_map()
            .get(args.target_adapter.as_str())
            .with_context(|| format!("adapter {} not found", args.target_adapter))?;
        let grad_id = *grads
            .get(&target)
            .with_context(|| format!("missing gradient for {}", args.target_adapter))?;
        let grad = store.to_host(grad_id)?;
        let index = largest_nonzero_index(&grad).with_context(|| {
            format!(
                "gradient for {} has no non-zero element",
                args.target_adapter
            )
        })?;
        Ok(SelectedProbe {
            name: args.target_adapter.clone(),
            target,
            index,
            analytic: grad[index],
        })
    }

    fn select_auto_routed_up_probe(
        model: &Qwen35Model,
        grads: &std::collections::HashMap<TensorId, TensorId>,
        store: &mut TensorStore,
        layer: usize,
    ) -> Result<SelectedProbe> {
        let prefix = format!("model.language_model.layers.{layer}.mlp.experts.");
        let mut candidates = model
            .adapter_name_map()
            .into_iter()
            .filter(|(name, _)| {
                name.starts_with(&prefix) && name.ends_with(".up_proj.weight.lora_b")
            })
            .collect::<Vec<_>>();
        candidates.sort_by_key(|(name, _)| *name);

        let mut best: Option<SelectedProbe> = None;
        for (name, target) in candidates {
            let Some(&grad_id) = grads.get(&target) else {
                continue;
            };
            let grad = store.to_host(grad_id)?;
            let Some(index) = largest_nonzero_index(&grad) else {
                continue;
            };
            let analytic = grad[index];
            let replace = best
                .as_ref()
                .is_none_or(|current| analytic.abs() > current.analytic.abs());
            if replace {
                best = Some(SelectedProbe {
                    name: name.to_string(),
                    target,
                    index,
                    analytic,
                });
            }
        }
        best.with_context(|| {
            format!("auto:routed-up found no non-zero routed expert up_proj LoRA-B gradient in layer {layer}")
        })
    }

    fn make_gate_input(
        model: &Qwen35Model,
        store: &mut TensorStore,
        args: &Args,
        positions: &[u32],
    ) -> Result<GateInput> {
        let hidden = match args.mode {
            GateMode::MlpLayer | GateMode::Tail => Some(store.alloc(Tensor::new(
                deterministic_vec(
                    args.tokens.len() * model.config().hidden_size,
                    0x4781_9f20_aa11_0017,
                    0.02,
                ),
                vec![1, args.tokens.len(), model.config().hidden_size],
                args.mode == GateMode::Tail,
            )?)),
            GateMode::FullModel => None,
        };
        let mut tape = Tape::new();
        tape.set_enabled(false);
        let output = forward_gate_output(
            model,
            store,
            &mut tape,
            args.mode,
            args.layer,
            &args.tokens,
            positions,
            hidden,
            None,
            None,
        )?;
        let shape = store
            .get(output)
            .context("gate output tensor missing")?
            .shape
            .clone();
        let size = store
            .get(output)
            .context("gate output tensor missing")?
            .size;
        let (probe_data, probe_shape) =
            if args.sparse_logit_probe && args.mode == GateMode::FullModel {
                let vocab = *shape
                    .last()
                    .context("full-model output missing vocab dimension")?;
                let prefix_elems = size / vocab;
                let (probe, indices) = deterministic_sparse_logit_probe(
                    prefix_elems,
                    vocab,
                    SPARSE_LOGIT_PROBE_PER_ROW,
                    size,
                );
                let preview = indices.iter().copied().take(16).collect::<Vec<_>>();
                println!(
                    "qwen36_fp8_lora_sparse_logit_probe prefix_elems={} vocab={} \
                 per_row={} nonzero={} scale=1.0 indices_preview={:?}",
                    prefix_elems,
                    vocab,
                    SPARSE_LOGIT_PROBE_PER_ROW,
                    indices.len(),
                    preview
                );
                (probe, shape)
            } else {
                (deterministic_vec(size, 0x8d12_ba11_cafe_f00d, 0.01), shape)
            };
        let probe = store.alloc(Tensor::new(probe_data, probe_shape, false)?);
        store.free(output).ok();
        Ok(GateInput { hidden, probe })
    }

    fn forward_gate_output(
        model: &Qwen35Model,
        store: &mut TensorStore,
        tape: &mut Tape,
        mode: GateMode,
        layer: usize,
        tokens: &[u32],
        positions: &[u32],
        hidden: Option<TensorId>,
        route_sink: Option<&mut Vec<Qwen35MoeRouteSignature>>,
        frozen_routes: Option<&[Qwen35MoeRouteSignature]>,
    ) -> Result<TensorId> {
        match mode {
            GateMode::MlpLayer => {
                let hidden = hidden.context("mlp-layer mode missing hidden input")?;
                Ok(model.forward_mlp_for_diagnostics(
                    layer,
                    hidden,
                    1,
                    tokens.len(),
                    store,
                    tape,
                )?)
            }
            GateMode::Tail => {
                let hidden = hidden.context("tail mode missing hidden input")?;
                Ok(model.forward_lm_head_tail_for_diagnostics(hidden, store, tape)?)
            }
            GateMode::FullModel => {
                if let Some(frozen_routes) = frozen_routes {
                    let output = model.forward_with_frozen_moe_routes_for_diagnostics(
                        store,
                        tape,
                        tokens,
                        positions,
                        frozen_routes,
                    )?;
                    if let Some(route_sink) = route_sink {
                        route_sink.extend(frozen_routes.iter().cloned());
                    }
                    Ok(output)
                } else if let Some(route_sink) = route_sink {
                    let (output, routes) = model
                        .forward_with_moe_routes_for_diagnostics(store, tape, tokens, positions)?;
                    route_sink.extend(routes);
                    Ok(output)
                } else {
                    Ok(model.forward(store, tape, tokens, positions)?)
                }
            }
        }
    }

    fn weighted_logits_loss(
        model: &Qwen35Model,
        store: &mut TensorStore,
        tape: &mut Tape,
        mode: GateMode,
        layer: usize,
        tokens: &[u32],
        positions: &[u32],
        hidden: Option<TensorId>,
        probe: TensorId,
        route_sink: Option<&mut Vec<Qwen35MoeRouteSignature>>,
        frozen_routes: Option<&[Qwen35MoeRouteSignature]>,
    ) -> Result<TensorId> {
        let output = forward_gate_output(
            model,
            store,
            tape,
            mode,
            layer,
            tokens,
            positions,
            hidden,
            route_sink,
            frozen_routes,
        )?;
        let weighted = mul(output, probe, store, tape)?;
        Ok(sum(weighted, store, tape)?)
    }

    fn loss_value(
        model: &Qwen35Model,
        store: &mut TensorStore,
        args: &Args,
        positions: &[u32],
        input: &GateInput,
        route_sink: Option<&mut Vec<Qwen35MoeRouteSignature>>,
        frozen_routes: Option<&[Qwen35MoeRouteSignature]>,
    ) -> Result<f32> {
        let mut tape = Tape::new();
        tape.set_enabled(false);
        let loss = weighted_logits_loss(
            model,
            store,
            &mut tape,
            args.mode,
            args.layer,
            &args.tokens,
            positions,
            input.hidden,
            input.probe,
            route_sink,
            frozen_routes,
        )?;
        scalar_host(store, loss)
    }

    /// Frozen-prompt-KV gate: FD-check the analytic gradient of a probe-weighted
    /// loss over the frozen gen-segment hidden against finite differences on the
    /// target LoRA adapter. `gen_start = prompt_len - 1`; the prefix is captured
    /// off-tape and the gen segment (boundary ++ generated tokens) is forwarded
    /// taped. Confirms the host taped carry on the production FP8 model.
    fn run_frozen_prompt_kv_gate(
        model: &Qwen35Model,
        store: &mut TensorStore,
        args: &Args,
        backend: &Arc<CudaBackend>,
    ) -> Result<()> {
        let gen_start = args.prompt_len - 1;
        let prompt_prefix: Vec<u32> = args.tokens[0..gen_start].to_vec();
        let gen_ids: Vec<u32> = args.tokens[gen_start..].to_vec();
        let prompt_positions: Vec<u32> = (0..gen_start as u32).collect();
        let gen_positions: Vec<u32> = (gen_start as u32..args.tokens.len() as u32).collect();

        // Deterministic probe over the gen hidden [1, gen_len, hidden].
        let gen_len = gen_ids.len();
        let probe_len = gen_len * model.config().hidden_size;
        let probe_data = deterministic_vec(probe_len, 0x9c3f_11ae_5577_d00d, 0.01);
        let probe = store.alloc(Tensor::new(
            probe_data,
            vec![1, gen_len, model.config().hidden_size],
            false,
        )?);

        // Analytic pass.
        let mut tape = Tape::new();
        tape.set_enabled(true);
        let hidden = model
            .forward_hidden_states_gen_segment(
                store,
                &mut tape,
                &prompt_prefix,
                &gen_ids,
                &prompt_positions,
                &gen_positions,
            )
            .context("frozen gen-segment analytic forward")?;
        let weighted = mul(hidden, probe, store, &mut tape)?;
        let loss = sum(weighted, store, &mut tape)?;
        let loss_base = scalar_host(store, loss)?;
        let grads = tape.backward(loss, store)?;
        backend
            .device_synchronize()
            .context("synchronize after frozen analytic backward")?;

        let target = *model
            .adapter_name_map()
            .get(args.target_adapter.as_str())
            .with_context(|| format!("adapter {} not found", args.target_adapter))?;
        let grad_id = *grads
            .get(&target)
            .with_context(|| format!("missing gradient for {}", args.target_adapter))?;
        let grad = store.to_host(grad_id)?;
        let index = largest_nonzero_index(&grad).with_context(|| {
            format!(
                "frozen gen-segment gradient for {} has no non-zero element",
                args.target_adapter
            )
        })?;
        let analytic = grad[index];

        // Finite-diff pass: perturb the selected adapter element.
        let fd_loss = |store: &mut TensorStore| -> Result<f32> {
            let mut tape = Tape::new();
            tape.set_enabled(false);
            let hidden = model
                .forward_hidden_states_gen_segment(
                    store,
                    &mut tape,
                    &prompt_prefix,
                    &gen_ids,
                    &prompt_positions,
                    &gen_positions,
                )
                .context("frozen gen-segment fd forward")?;
            let weighted = mul(hidden, probe, store, &mut tape)?;
            let loss = sum(weighted, store, &mut tape)?;
            scalar_host(store, loss)
        };

        let original = tensor_value(store, target, index)?;
        set_tensor_value(store, target, index, original + args.eps)?;
        let loss_plus = fd_loss(store)?;
        set_tensor_value(store, target, index, original - args.eps)?;
        let loss_minus = fd_loss(store)?;
        set_tensor_value(store, target, index, original)?;
        backend
            .device_synchronize()
            .context("synchronize after frozen finite diff")?;

        let numeric = (loss_plus - loss_minus) / (2.0 * args.eps);
        let rel_err = rel_err(analytic, numeric);
        println!(
            "qwen36_fp8_lora_frozen_prompt_kv_result prompt_len={} gen_start={} gen_len={} \
             target={} index={} eps={:.1e} loss_base={:.9e} loss_minus={:.9e} loss_plus={:.9e} \
             analytic={:.9e} numeric={:.9e} rel_err={:.3e}",
            args.prompt_len,
            gen_start,
            gen_len,
            args.target_adapter,
            index,
            args.eps,
            loss_base,
            loss_minus,
            loss_plus,
            analytic,
            numeric,
            rel_err
        );
        if rel_err > 1.0e-2 {
            bail!(
                "Qwen3.6 FP8 frozen-prompt-KV finite diff failed: target={} analytic={:.9e} \
                 numeric={numeric:.9e} rel_err={rel_err:.3e}",
                args.target_adapter,
                analytic
            );
        }
        println!("qwen36_fp8_lora_frozen_prompt_kv PASS");
        Ok(())
    }

    fn collect_base_moe_routes(
        model: &Qwen35Model,
        store: &mut TensorStore,
        tokens: &[u32],
        positions: &[u32],
    ) -> Result<Vec<Qwen35MoeRouteSignature>> {
        let mut tape = Tape::new();
        tape.set_enabled(false);
        let (output, routes) =
            model.forward_with_moe_routes_for_diagnostics(store, &mut tape, tokens, positions)?;
        store.free(output).ok();
        Ok(routes)
    }

    fn profile_forward_only(
        model: &Qwen35Model,
        store: &mut TensorStore,
        args: &Args,
        positions: &[u32],
    ) -> Result<()> {
        if args.mode != GateMode::FullModel {
            bail!("--profile-forward-only requires --mode full-model");
        }
        let mut tape = Tape::new();
        tape.set_enabled(false);
        let started = Instant::now();
        let (output, profile) = model.forward_profiled_for_diagnostics(
            store,
            &mut tape,
            &args.tokens,
            positions,
            true,
        )?;
        let elapsed = started.elapsed();
        let output_shape = store
            .get(output)
            .context("profile forward output tensor missing")?
            .shape
            .clone();
        print_forward_profile(&profile, elapsed, &output_shape);
        Ok(())
    }

    fn scalar_host(store: &mut TensorStore, id: TensorId) -> Result<f32> {
        let values = store.to_host(id)?;
        values
            .first()
            .copied()
            .with_context(|| format!("tensor {id} is empty"))
    }

    fn largest_nonzero_index(values: &[f32]) -> Option<usize> {
        values
            .iter()
            .enumerate()
            .filter(|(_, value)| value.abs() > 1.0e-9)
            .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
            .map(|(idx, _)| idx)
    }

    fn tensor_value(store: &mut TensorStore, id: TensorId, index: usize) -> Result<f32> {
        store
            .to_host(id)?
            .get(index)
            .copied()
            .with_context(|| format!("tensor {id} missing index {index}"))
    }

    fn set_tensor_value(
        store: &mut TensorStore,
        id: TensorId,
        index: usize,
        value: f32,
    ) -> Result<()> {
        let tensor = store
            .get_mut(id)
            .with_context(|| format!("missing tensor {id}"))?;
        let slot = tensor
            .data
            .get_mut(index)
            .with_context(|| format!("tensor {id} missing index {index}"))?;
        *slot = value;
        Ok(())
    }

    fn rel_err(analytic: f32, numeric: f32) -> f64 {
        let analytic = f64::from(analytic);
        let numeric = f64::from(numeric);
        (analytic - numeric).abs() / analytic.abs().max(numeric.abs()).max(1.0e-12)
    }

    fn print_route_stability(
        base: &[Qwen35MoeRouteSignature],
        plus: &[Qwen35MoeRouteSignature],
        minus: &[Qwen35MoeRouteSignature],
    ) {
        let plus_summary = route_diff_summary(base, plus);
        let minus_summary = route_diff_summary(base, minus);
        println!(
            "qwen36_fp8_lora_route_stability base_layers={} plus_layers={} \
             minus_layers={} plus_changed_layers={} plus_changed_slots={} \
             plus_total_slots={} plus_shape_mismatches={} minus_changed_layers={} \
             minus_changed_slots={} minus_total_slots={} minus_shape_mismatches={}",
            base.len(),
            plus.len(),
            minus.len(),
            plus_summary.changed_layers,
            plus_summary.changed_slots,
            plus_summary.total_slots,
            plus_summary.shape_mismatches,
            minus_summary.changed_layers,
            minus_summary.changed_slots,
            minus_summary.total_slots,
            minus_summary.shape_mismatches
        );
        print_route_diff_layers("plus", base, plus);
        print_route_diff_layers("minus", base, minus);
    }

    #[derive(Debug, Default)]
    struct RouteDiffSummary {
        changed_layers: usize,
        changed_slots: usize,
        total_slots: usize,
        shape_mismatches: usize,
    }

    fn route_diff_summary(
        base: &[Qwen35MoeRouteSignature],
        other: &[Qwen35MoeRouteSignature],
    ) -> RouteDiffSummary {
        let mut summary = RouteDiffSummary::default();
        for (base_layer, other_layer) in base.iter().zip(other.iter()) {
            if base_layer.layer != other_layer.layer
                || base_layer.tokens != other_layer.tokens
                || base_layer.top_k != other_layer.top_k
                || base_layer.experts != other_layer.experts
                || base_layer.indices.len() != other_layer.indices.len()
            {
                summary.shape_mismatches += 1;
                continue;
            }
            let changed = base_layer
                .indices
                .iter()
                .zip(&other_layer.indices)
                .filter(|(left, right)| left != right)
                .count();
            summary.total_slots += base_layer.indices.len();
            summary.changed_slots += changed;
            if changed > 0 {
                summary.changed_layers += 1;
            }
        }
        summary.shape_mismatches += base.len().abs_diff(other.len());
        summary
    }

    fn print_route_diff_layers(
        arm: &str,
        base: &[Qwen35MoeRouteSignature],
        other: &[Qwen35MoeRouteSignature],
    ) {
        let mut printed = 0usize;
        for (base_layer, other_layer) in base.iter().zip(other.iter()) {
            if base_layer.layer != other_layer.layer
                || base_layer.indices.len() != other_layer.indices.len()
            {
                println!(
                    "qwen36_fp8_lora_route_stability_layer arm={arm} \
                     layer={} status=shape_mismatch base_slots={} other_slots={}",
                    base_layer.layer,
                    base_layer.indices.len(),
                    other_layer.indices.len()
                );
                printed += 1;
            } else {
                let changed = base_layer
                    .indices
                    .iter()
                    .zip(&other_layer.indices)
                    .filter(|(left, right)| left != right)
                    .count();
                if changed == 0 {
                    continue;
                }
                println!(
                    "qwen36_fp8_lora_route_stability_layer arm={arm} \
                     layer={} changed_slots={} total_slots={} tokens={} top_k={} \
                     experts={}",
                    base_layer.layer,
                    changed,
                    base_layer.indices.len(),
                    base_layer.tokens,
                    base_layer.top_k,
                    base_layer.experts
                );
                printed += 1;
            }
            if printed >= 16 {
                break;
            }
        }
    }

    fn print_backward_profile(profile: &BackwardProfile) {
        let total_seconds = secs(profile.total_duration);
        let op_seconds = secs(profile.total_op_duration());
        println!(
            "qwen36_fp8_lora_fd_backward_profile total_seconds={total_seconds:.6} \
             op_seconds={op_seconds:.6} prelude_seconds={:.6} merge_grad_seconds={:.6} \
             op_kinds={} site_kinds={}",
            secs(profile.prelude_duration),
            secs(profile.merge_grad_duration),
            profile.op_totals.len(),
            profile.site_totals.len()
        );

        let mut op_rows = profile.op_totals.iter().collect::<Vec<_>>();
        op_rows.sort_by(|(_, left), (_, right)| right.duration.cmp(&left.duration));
        for (rank, (op, stats)) in op_rows.into_iter().take(16).enumerate() {
            let seconds = secs(stats.duration);
            let pct_total = pct(seconds, total_seconds);
            println!(
                "qwen36_fp8_lora_fd_backward_profile_op rank={} op={} count={} \
                 seconds={seconds:.6} pct_total={pct_total:.3}",
                rank + 1,
                op.name(),
                stats.count
            );
        }

        let mut site_rows = profile.site_totals.iter().collect::<Vec<_>>();
        site_rows.sort_by(|(_, left), (_, right)| right.duration.cmp(&left.duration));
        for (rank, ((op, site), stats)) in site_rows.into_iter().take(16).enumerate() {
            let seconds = secs(stats.duration);
            let pct_total = pct(seconds, total_seconds);
            println!(
                "qwen36_fp8_lora_fd_backward_profile_site rank={} op={} site={} count={} \
                 seconds={seconds:.6} pct_total={pct_total:.3}",
                rank + 1,
                op.name(),
                site,
                stats.count
            );
        }
    }

    fn print_forward_profile(
        profile: &Qwen35RolloutForwardProfile,
        wall: Duration,
        output_shape: &[usize],
    ) {
        let profile_seconds = secs(profile.total);
        let wall_seconds = secs(wall);
        println!(
            "qwen36_fp8_lora_fd_forward_profile total_seconds={profile_seconds:.6} \
             wall_seconds={wall_seconds:.6} layers={} output_shape={:?} \
             cache_select_seconds={:.6} embedding_seconds={:.6} input_rmsnorm_seconds={:.6} \
             attention_seconds={:.6} attention_residual_seconds={:.6} \
             post_attention_rmsnorm_seconds={:.6} mlp_seconds={:.6} \
             mlp_residual_seconds={:.6} final_norm_seconds={:.6} lm_head_seconds={:.6}",
            profile.layers.len(),
            output_shape,
            secs(profile.cache_select),
            secs(profile.embedding),
            secs(profile.input_rmsnorm_total()),
            secs(profile.attention_total()),
            secs(profile.attention_residual_total()),
            secs(profile.post_attention_rmsnorm_total()),
            secs(profile.mlp_total()),
            secs(profile.mlp_residual_total()),
            secs(profile.final_norm),
            secs(profile.lm_head)
        );

        for (idx, layer) in profile.layers.iter().enumerate() {
            println!(
                "qwen36_fp8_lora_fd_forward_profile_layer layer={} \
                 input_rmsnorm_seconds={:.6} attention_seconds={:.6} \
                 attention_residual_seconds={:.6} post_attention_rmsnorm_seconds={:.6} \
                 mlp_seconds={:.6} mlp_residual_seconds={:.6}",
                idx,
                secs(layer.input_rmsnorm),
                secs(layer.attention),
                secs(layer.attention_residual),
                secs(layer.post_attention_rmsnorm),
                secs(layer.mlp),
                secs(layer.mlp_residual)
            );
        }

        let attention: Qwen35AttentionForwardProfile =
            profile
                .layers
                .iter()
                .fold(Default::default(), |mut acc, layer| {
                    acc.q_proj += layer.attention_detail.q_proj;
                    acc.q_layout += layer.attention_detail.q_layout;
                    acc.k_proj += layer.attention_detail.k_proj;
                    acc.v_proj += layer.attention_detail.v_proj;
                    acc.kv_split += layer.attention_detail.kv_split;
                    acc.qk_norm += layer.attention_detail.qk_norm;
                    acc.rope += layer.attention_detail.rope;
                    acc.repeat_kv += layer.attention_detail.repeat_kv;
                    acc.append_kv += layer.attention_detail.append_kv;
                    acc.sdpa += layer.attention_detail.sdpa;
                    acc.gate += layer.attention_detail.gate;
                    acc.merge += layer.attention_detail.merge;
                    acc.o_proj += layer.attention_detail.o_proj;
                    acc.linear_qkv_proj += layer.attention_detail.linear_qkv_proj;
                    acc.linear_z_proj += layer.attention_detail.linear_z_proj;
                    acc.linear_b_proj += layer.attention_detail.linear_b_proj;
                    acc.linear_a_proj += layer.attention_detail.linear_a_proj;
                    acc.linear_core += layer.attention_detail.linear_core;
                    acc.linear_out_proj += layer.attention_detail.linear_out_proj;
                    acc
                });
        for (component, duration) in [
            ("q_proj", attention.q_proj),
            ("q_layout", attention.q_layout),
            ("k_proj", attention.k_proj),
            ("v_proj", attention.v_proj),
            ("kv_split", attention.kv_split),
            ("qk_norm", attention.qk_norm),
            ("rope", attention.rope),
            ("repeat_kv", attention.repeat_kv),
            ("append_kv", attention.append_kv),
            ("sdpa", attention.sdpa),
            ("gate", attention.gate),
            ("merge", attention.merge),
            ("o_proj", attention.o_proj),
            ("linear_qkv_proj", attention.linear_qkv_proj),
            ("linear_z_proj", attention.linear_z_proj),
            ("linear_b_proj", attention.linear_b_proj),
            ("linear_a_proj", attention.linear_a_proj),
            ("linear_core", attention.linear_core),
            ("linear_out_proj", attention.linear_out_proj),
        ] {
            let seconds = secs(duration);
            if seconds > 0.0 {
                println!(
                    "qwen36_fp8_lora_fd_forward_profile_attention component={} \
                     seconds={seconds:.6} pct_total={:.3}",
                    component,
                    pct(seconds, profile_seconds)
                );
            }
        }
    }

    fn pct(seconds: f64, total_seconds: f64) -> f64 {
        if total_seconds == 0.0 {
            0.0
        } else {
            seconds / total_seconds * 100.0
        }
    }

    fn secs(duration: Duration) -> f64 {
        duration.as_secs_f64()
    }

    fn deterministic_vec(len: usize, mut state: u64, scale: f32) -> Vec<f32> {
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let bits = ((state >> 32) as u32) as f32 / u32::MAX as f32;
                (bits * 2.0 - 1.0) * scale
            })
            .collect()
    }

    fn deterministic_indices(len: usize, vocab: usize, mut state: u64) -> Vec<usize> {
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(2862933555777941757)
                    .wrapping_add(3037000493);
                ((state >> 32) as usize) % vocab
            })
            .collect()
    }

    fn deterministic_sparse_logit_probe(
        rows: usize,
        vocab: usize,
        per_row: usize,
        full_size: usize,
    ) -> (Vec<f32>, Vec<usize>) {
        let mut probe = vec![0.0_f32; full_size];
        let indices = deterministic_indices(rows * per_row, vocab, 0x5e17_1d5a_baad_f00d);
        let values = deterministic_vec(rows * per_row, 0x8d12_ba11_cafe_f00d, 1.0);
        for row in 0..rows {
            for slot in 0..per_row {
                let flat = row * per_row + slot;
                let index = indices[flat];
                probe[row * vocab + index] += values[flat];
            }
        }
        (probe, indices)
    }
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
fn main() -> anyhow::Result<()> {
    app::main()
}

#[cfg(not(all(feature = "cuda", not(feature = "no-cuda"))))]
fn main() {
    eprintln!(
        "qwen36_fp8_lora_fd_gate requires a real CUDA build: \
         cargo run -p train --example qwen36_fp8_lora_fd_gate --release --features cuda"
    );
}
