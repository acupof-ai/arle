use std::{env, path::PathBuf, sync::Arc};

use anyhow::{Context, Result, bail};
use autograd::{Backend, Tape, TensorId, TensorStore, backend_cuda::CudaBackend};
use cuda_kernels::prelude::{DeviceContext, DeviceVec};
use cudarc::driver::DevicePtr;
use infer::model::Qwen35Model as InferQwen35Model;
use train::qwen35_loader::load_qwen35_from_hf_dir;

const DEFAULT_MODEL_DIR: &str = "/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base";
const TOKEN_ID: u32 = 9419;
const STRICT_DIVERGENCE_THRESHOLD: f32 = 1.0e-4;
const BF16_ABS_GATE: f32 = 1.0e-2;
const BF16_MEAN_RATIO_GATE: f32 = 1.0e-2;
const LM_HEAD_DOMINANT_REL_GATE: f32 = 1.0e-2;
const LM_HEAD_DOMINANT_TOP_K: usize = 64;

struct StageReport {
    name: &'static str,
    len: usize,
    max_abs: f32,
    max_rel: f32,
    mean_abs_ref: f32,
    max_abs_over_mean_abs_ref: f32,
    lm_head_dominant_rel: Option<f32>,
    bf16_gate_pass: bool,
    first_train: f32,
    first_infer: f32,
}

fn main() -> Result<()> {
    let model_dir = env::var_os("ARLE_PARITY_QWEN35_08B_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_DIR));
    println!("model_dir={}", model_dir.display());
    println!("input_ids=[{TOKEN_ID}] positions=[0]");

    let backend: Arc<dyn Backend> = Arc::new(CudaBackend::new(0)?);
    let mut store = TensorStore::with_backend(backend.clone());
    let mut tape = Tape::new();
    tape.set_enabled(false);

    let train_model = load_qwen35_from_hf_dir(&model_dir, &mut store)
        .with_context(|| format!("load train Qwen3.5 from {}", model_dir.display()))?;
    let train_stages =
        train_model.forward_single_token_parity_stages(&mut store, &mut tape, TOKEN_ID, 0)?;

    let infer_model = InferQwen35Model::from_safetensors_with_options(
        model_dir
            .to_str()
            .context("model path is not valid UTF-8")?,
        false,
    )
    .with_context(|| format!("load infer Qwen3.5 from {}", model_dir.display()))?;
    let infer_stages = infer_model.forward_single_token_parity_stages(TOKEN_ID)?;

    let stage_pairs = [
        ("embedding", train_stages.embedding, &infer_stages.embedding),
        (
            "layer0_rmsnorm",
            train_stages.layer0_rmsnorm,
            &infer_stages.layer0_rmsnorm,
        ),
        (
            "layer0_attention",
            train_stages.layer0_attention,
            &infer_stages.layer0_attention,
        ),
        (
            "layer0_ffn",
            train_stages.layer0_ffn,
            &infer_stages.layer0_ffn,
        ),
        (
            "layer0_residual",
            train_stages.layer0_residual,
            &infer_stages.layer0_residual,
        ),
        (
            "final_rmsnorm",
            train_stages.final_rmsnorm,
            &infer_stages.final_rmsnorm,
        ),
        ("lm_head", train_stages.lm_head, &infer_stages.lm_head),
    ];

    let mut reports = Vec::with_capacity(stage_pairs.len());
    for (name, train_id, infer_vec) in stage_pairs {
        reports.push(compare_stage(
            name,
            train_id,
            infer_vec,
            &infer_stages_device(&infer_model),
            &backend,
            &mut store,
        )?);
    }

    println!();
    println!(
        "| stage | len | max_abs | mean_abs_ref | max_abs/mean_abs_ref | max_rel | lm_head_top64_rel | bf16_gate | first train | first infer |"
    );
    println!("| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: |");
    for report in &reports {
        println!(
            "| {} | {} | {:.8e} | {:.8e} | {:.8e} | {:.8e} | {} | {} | {:.8e} | {:.8e} |",
            report.name,
            report.len,
            report.max_abs,
            report.mean_abs_ref,
            report.max_abs_over_mean_abs_ref,
            report.max_rel,
            format_optional_f32(report.lm_head_dominant_rel),
            if report.bf16_gate_pass {
                "PASS"
            } else {
                "FAIL"
            },
            report.first_train,
            report.first_infer
        );
    }

    let first_divergence = reports.iter().find(|report| {
        report.max_abs > STRICT_DIVERGENCE_THRESHOLD || report.max_rel > STRICT_DIVERGENCE_THRESHOLD
    });
    match first_divergence {
        Some(report) => {
            println!(
                "\nfirst_divergence={} max_abs={:.8e} max_rel={:.8e} threshold={:.1e}",
                report.name, report.max_abs, report.max_rel, STRICT_DIVERGENCE_THRESHOLD
            );
        }
        None => {
            println!(
                "\nfirst_divergence=none threshold={:.1e}",
                STRICT_DIVERGENCE_THRESHOLD
            );
        }
    }

    let first_bf16_failure = reports.iter().find(|report| !report.bf16_gate_pass);
    match first_bf16_failure {
        Some(report) => {
            println!(
                "first_bf16_gate_failure={} max_abs={:.8e} max_abs/mean_abs_ref={:.8e}",
                report.name, report.max_abs, report.max_abs_over_mean_abs_ref
            );
        }
        None => {
            println!(
                "first_bf16_gate_failure=none abs_gate={:.1e} mean_ratio_gate={:.1e}",
                BF16_ABS_GATE, BF16_MEAN_RATIO_GATE
            );
        }
    }

    let lm_head_dominant_rel = reports
        .iter()
        .find(|report| report.name == "lm_head")
        .and_then(|report| report.lm_head_dominant_rel)
        .unwrap_or(f32::INFINITY);
    let path_b_commit3_retry =
        first_bf16_failure.is_none() && lm_head_dominant_rel <= LM_HEAD_DOMINANT_REL_GATE;
    println!(
        "lm_head_dominant_top_k={} rel_gate={:.1e} observed={:.8e}",
        LM_HEAD_DOMINANT_TOP_K, LM_HEAD_DOMINANT_REL_GATE, lm_head_dominant_rel
    );
    println!(
        "path_b_commit3_retry={}",
        if path_b_commit3_retry {
            "licensed"
        } else {
            "blocked"
        }
    );

    Ok(())
}

fn format_optional_f32(value: Option<f32>) -> String {
    value
        .map(|value| format!("{value:.8e}"))
        .unwrap_or_else(|| "-".to_string())
}

fn infer_stages_device(model: &InferQwen35Model) -> DeviceContext {
    model.parity_device_context()
}

fn compare_stage(
    name: &'static str,
    train_id: TensorId,
    infer_vec: &DeviceVec,
    infer_ctx: &DeviceContext,
    backend: &Arc<dyn Backend>,
    store: &mut TensorStore,
) -> Result<StageReport> {
    infer_ctx.sync()?;
    let (ptr, _guard) = infer_vec.data.device_ptr(&infer_ctx.stream);
    let imported_handle =
        backend.import_bf16_device_ptr_as_f32(ptr as u64, infer_vec.len, &[infer_vec.len])?;
    let imported = store.alloc_device_tensor(vec![infer_vec.len], imported_handle)?;
    let train_host = store.to_host(train_id)?;
    let infer_host = store.to_host(imported)?;
    if train_host.len() != infer_host.len() {
        bail!(
            "{name}: length mismatch train={} infer={}",
            train_host.len(),
            infer_host.len()
        );
    }

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut ref_abs_sum = 0.0f32;
    for (&train, &infer) in train_host.iter().zip(infer_host.iter()) {
        let abs = (train - infer).abs();
        let denom = train.abs().max(infer.abs()).max(1.0e-6);
        let rel = abs / denom;
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
        ref_abs_sum += infer.abs();
    }
    let mean_abs_ref = if infer_host.is_empty() {
        0.0
    } else {
        ref_abs_sum / infer_host.len() as f32
    };
    let max_abs_over_mean_abs_ref = if mean_abs_ref > 0.0 {
        max_abs / mean_abs_ref
    } else if max_abs == 0.0 {
        0.0
    } else {
        f32::INFINITY
    };
    let bf16_gate_pass =
        max_abs <= BF16_ABS_GATE || max_abs_over_mean_abs_ref <= BF16_MEAN_RATIO_GATE;
    let lm_head_dominant_rel = (name == "lm_head").then(|| {
        max_dominant_relerr(
            &train_host,
            &infer_host,
            LM_HEAD_DOMINANT_TOP_K.min(infer_host.len()),
        )
    });

    Ok(StageReport {
        name,
        len: train_host.len(),
        max_abs,
        max_rel,
        mean_abs_ref,
        max_abs_over_mean_abs_ref,
        lm_head_dominant_rel,
        bf16_gate_pass,
        first_train: train_host.first().copied().unwrap_or(0.0),
        first_infer: infer_host.first().copied().unwrap_or(0.0),
    })
}

fn max_dominant_relerr(train_host: &[f32], infer_host: &[f32], top_k: usize) -> f32 {
    if top_k == 0 {
        return 0.0;
    }
    let mut indices: Vec<usize> = (0..infer_host.len()).collect();
    indices.sort_unstable_by(|&a, &b| infer_host[b].abs().total_cmp(&infer_host[a].abs()));
    indices
        .into_iter()
        .take(top_k)
        .map(|index| {
            let reference = infer_host[index];
            (train_host[index] - reference).abs() / reference.abs().max(1.0e-6)
        })
        .fold(0.0f32, f32::max)
}
