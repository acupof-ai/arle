use std::{env, path::PathBuf, sync::Arc};

use anyhow::{Context, Result, bail};
use autograd::{Backend, Tape, TensorId, TensorStore, backend_cuda::CudaBackend};
use cuda_kernels::prelude::{DeviceContext, DeviceVec};
use cudarc::driver::DevicePtr;
use infer::model::Qwen35Model as InferQwen35Model;
use train::qwen35_loader::load_qwen35_from_hf_dir;

const DEFAULT_MODEL_DIR: &str = "/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base";
const TOKEN_ID: u32 = 9419;
const DIVERGENCE_THRESHOLD: f32 = 1.0e-4;

struct StageReport {
    name: &'static str,
    len: usize,
    max_abs: f32,
    max_rel: f32,
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
    println!("| stage | len | max_abs | max_rel | first train | first infer |");
    println!("| --- | ---: | ---: | ---: | ---: | ---: |");
    for report in &reports {
        println!(
            "| {} | {} | {:.8e} | {:.8e} | {:.8e} | {:.8e} |",
            report.name,
            report.len,
            report.max_abs,
            report.max_rel,
            report.first_train,
            report.first_infer
        );
    }

    let first_divergence = reports.iter().find(|report| {
        report.max_abs > DIVERGENCE_THRESHOLD || report.max_rel > DIVERGENCE_THRESHOLD
    });
    match first_divergence {
        Some(report) => {
            println!(
                "\nfirst_divergence={} max_abs={:.8e} max_rel={:.8e} threshold={:.1e}",
                report.name, report.max_abs, report.max_rel, DIVERGENCE_THRESHOLD
            );
        }
        None => {
            println!(
                "\nfirst_divergence=none threshold={:.1e}",
                DIVERGENCE_THRESHOLD
            );
        }
    }

    Ok(())
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
    for (&train, &infer) in train_host.iter().zip(infer_host.iter()) {
        let abs = (train - infer).abs();
        let denom = train.abs().max(infer.abs()).max(1.0e-6);
        let rel = abs / denom;
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
    }

    Ok(StageReport {
        name,
        len: train_host.len(),
        max_abs,
        max_rel,
        first_train: train_host.first().copied().unwrap_or(0.0),
        first_infer: infer_host.first().copied().unwrap_or(0.0),
    })
}
