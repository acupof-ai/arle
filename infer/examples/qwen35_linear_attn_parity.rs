use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{Context, Result, anyhow, bail};
use infer::model::Qwen35Model;
use serde::{Deserialize, Serialize};

const DEFAULT_MODEL_DIR: &str = "/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base";
const DEFAULT_PYTHON: &str = "/home/ckl/projects/arle/.venv/bin/python";
const DEFAULT_TOKEN_ID: u32 = 9419;
const RMSE_REF_RMS_GATE: f32 = 0.05;

const PYTORCH_REFERENCE: &str = r#"
import argparse
import json
import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from transformers import AutoModelForCausalLM, logging

logging.set_verbosity_error()

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", required=True)
parser.add_argument("--token-id", required=True, type=int)
parser.add_argument("--device", default="cpu")
parser.add_argument("--output", required=True)
args = parser.parse_args()

device = torch.device(args.device)
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
    device_map=None,
    trust_remote_code=True,
).to(device)
model.eval()

config_dict = model.config.to_dict()
text_config = config_dict.get("text_config") or config_dict
layer0_type = text_config["layer_types"][0]
if layer0_type != "linear_attention":
    raise RuntimeError(f"layer 0 is {layer0_type}, expected linear_attention")

with torch.no_grad():
    input_ids = torch.tensor([[args.token_id]], dtype=torch.long, device=device)
    hidden = model.model.embed_tokens(input_ids)
    normed = model.model.layers[0].input_layernorm(hidden)
    out = model.model.layers[0].linear_attn(normed, cache_params=None, attention_mask=None)
    flat = out.reshape(-1).float().cpu().tolist()

with open(args.output, "w", encoding="utf-8") as handle:
    json.dump({"layer0_type": layer0_type, "values": flat}, handle)
"#;

#[derive(Debug)]
struct Args {
    model_path: PathBuf,
    reference_model_path: PathBuf,
    python: PathBuf,
    token_id: u32,
    python_device: String,
    output: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct PyReference {
    layer0_type: String,
    values: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct ParityReport {
    model_path: String,
    reference_model_path: String,
    token_id: u32,
    python_device: String,
    layer0_type: String,
    len: usize,
    finite_pair_count: usize,
    arle_nonfinite_count: usize,
    pytorch_nonfinite_count: usize,
    pytorch_finite_count: usize,
    max_abs: f32,
    max_rel: f32,
    mean_abs: f32,
    rmse: f32,
    ref_rms: f32,
    rmse_over_ref_rms: f32,
    gate_rmse_over_ref_rms: f32,
    gate_pass: bool,
    first8: Vec<Entry>,
}

#[derive(Debug, Serialize)]
struct Entry {
    index: usize,
    arle: f32,
    pytorch: f32,
    abs_err: f32,
    rel_err: f32,
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let layer0_type = read_layer0_type(&args.model_path)?;
    if layer0_type != "linear_attention" {
        bail!("layer 0 is {layer0_type}, expected linear_attention for this parity harness");
    }

    let model_path_str = args
        .model_path
        .to_str()
        .context("model path is not valid UTF-8")?;
    let reference_model_path_str = args
        .reference_model_path
        .to_str()
        .context("reference model path is not valid UTF-8")?;
    let model = Qwen35Model::from_safetensors_with_options(model_path_str, false)
        .with_context(|| format!("load ARLE Qwen3.5 model from {model_path_str}"))?;
    let stages = model.forward_single_token_parity_stages(args.token_id)?;
    let ctx = model.parity_device_context();
    let arle_values = stages.layer0_attention.to_host(&ctx)?;

    let py_reference = run_pytorch_reference(&args)?;
    if py_reference.layer0_type != "linear_attention" {
        bail!(
            "PyTorch layer 0 is {}, expected linear_attention",
            py_reference.layer0_type
        );
    }

    let report = compare(
        model_path_str,
        reference_model_path_str,
        args.token_id,
        &args.python_device,
        &layer0_type,
        &arle_values,
        &py_reference.values,
    )?;

    print_report(&report);
    if let Some(output) = &args.output {
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
        }
        fs::write(output, serde_json::to_string_pretty(&report)?)
            .with_context(|| format!("write {}", output.display()))?;
    }

    if !report.gate_pass {
        bail!(
            "linear_attn parity gate failed: rmse/ref_rms={:.6} gate={:.6}",
            report.rmse_over_ref_rms,
            report.gate_rmse_over_ref_rms
        );
    }

    Ok(())
}

fn parse_args() -> Result<Args> {
    let mut args = Args {
        model_path: env::var_os("ARLE_QWEN35_LINEAR_ATTN_MODEL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_DIR)),
        reference_model_path: env::var_os("ARLE_QWEN35_LINEAR_ATTN_REFERENCE_MODEL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_DIR)),
        python: env::var_os("ARLE_PYTHON")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_PYTHON)),
        token_id: DEFAULT_TOKEN_ID,
        python_device: "cpu".to_string(),
        output: None,
    };

    let mut iter = env::args().skip(1);
    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--model-path" => args.model_path = next_path(&mut iter, "--model-path")?,
            "--reference-model-path" => {
                args.reference_model_path = next_path(&mut iter, "--reference-model-path")?
            }
            "--python" => args.python = next_path(&mut iter, "--python")?,
            "--token-id" => args.token_id = next_value(&mut iter, "--token-id")?.parse()?,
            "--python-device" => args.python_device = next_value(&mut iter, "--python-device")?,
            "--output" => args.output = Some(next_path(&mut iter, "--output")?),
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => bail!("unknown argument: {other}"),
        }
    }

    Ok(args)
}

fn next_value(iter: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    iter.next()
        .ok_or_else(|| anyhow!("{flag} requires a value"))
}

fn next_path(iter: &mut impl Iterator<Item = String>, flag: &str) -> Result<PathBuf> {
    Ok(PathBuf::from(next_value(iter, flag)?))
}

fn print_help() {
    println!(
        "qwen35_linear_attn_parity [--model-path PATH] [--reference-model-path PATH] [--python PATH] \\
         [--python-device cpu|cuda] [--token-id ID] [--output PATH]"
    );
}

fn read_layer0_type(model_path: &Path) -> Result<String> {
    let config_path = model_path.join("config.json");
    let value: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(&config_path)
            .with_context(|| format!("read {}", config_path.display()))?,
    )?;
    value
        .get("text_config")
        .and_then(|text| text.get("layer_types"))
        .and_then(|layers| layers.get(0))
        .and_then(|layer| layer.as_str())
        .map(str::to_string)
        .ok_or_else(|| anyhow!("config missing text_config.layer_types[0]"))
}

fn run_pytorch_reference(args: &Args) -> Result<PyReference> {
    let tmp_dir = env::temp_dir().join(format!("arle-qwen35-linear-attn-{}", std::process::id()));
    fs::create_dir_all(&tmp_dir).with_context(|| format!("create {}", tmp_dir.display()))?;
    let script_path = tmp_dir.join("reference.py");
    let output_path = tmp_dir.join("reference.json");
    fs::write(&script_path, PYTORCH_REFERENCE)
        .with_context(|| format!("write {}", script_path.display()))?;

    let status = Command::new(&args.python)
        .arg(&script_path)
        .arg("--model-path")
        .arg(&args.reference_model_path)
        .arg("--token-id")
        .arg(args.token_id.to_string())
        .arg("--device")
        .arg(&args.python_device)
        .arg("--output")
        .arg(&output_path)
        .status()
        .with_context(|| format!("run {}", args.python.display()))?;
    if !status.success() {
        bail!("PyTorch reference exited with {status}");
    }

    serde_json::from_str(
        &fs::read_to_string(&output_path)
            .with_context(|| format!("read {}", output_path.display()))?,
    )
    .context("parse PyTorch reference JSON")
}

fn compare(
    model_path: &str,
    reference_model_path: &str,
    token_id: u32,
    python_device: &str,
    layer0_type: &str,
    arle: &[f32],
    pytorch: &[f32],
) -> Result<ParityReport> {
    if arle.len() != pytorch.len() {
        bail!(
            "length mismatch: ARLE={} PyTorch={}",
            arle.len(),
            pytorch.len()
        );
    }
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut abs_sum = 0.0f32;
    let mut sq_sum = 0.0f32;
    let mut ref_sq_sum = 0.0f32;
    let mut finite_pair_count = 0usize;
    let mut arle_nonfinite_count = 0usize;
    let mut pytorch_nonfinite_count = 0usize;
    let mut pytorch_finite_count = 0usize;
    let mut first8 = Vec::with_capacity(8.min(arle.len()));

    for (index, (&a, &p)) in arle.iter().zip(pytorch.iter()).enumerate() {
        if !a.is_finite() {
            arle_nonfinite_count += 1;
        }
        if !p.is_finite() {
            pytorch_nonfinite_count += 1;
        } else {
            ref_sq_sum += p * p;
            pytorch_finite_count += 1;
        }
        let abs_err = if a.is_finite() && p.is_finite() {
            let abs_err = (a - p).abs();
            let rel_err = abs_err / p.abs().max(1.0e-6);
            max_abs = max_abs.max(abs_err);
            max_rel = max_rel.max(rel_err);
            abs_sum += abs_err;
            sq_sum += abs_err * abs_err;
            finite_pair_count += 1;
            abs_err
        } else {
            f32::NAN
        };
        let rel_err = if a.is_finite() && p.is_finite() {
            abs_err / p.abs().max(1.0e-6)
        } else {
            f32::NAN
        };
        if first8.len() < 8 {
            first8.push(Entry {
                index,
                arle: a,
                pytorch: p,
                abs_err,
                rel_err,
            });
        }
    }

    let len = arle.len();
    let mean_abs = if finite_pair_count > 0 {
        abs_sum / finite_pair_count as f32
    } else {
        f32::NAN
    };
    let rmse = if finite_pair_count > 0 {
        (sq_sum / finite_pair_count as f32).sqrt()
    } else {
        f32::NAN
    };
    let ref_rms = if pytorch_finite_count > 0 {
        (ref_sq_sum / pytorch_finite_count as f32).sqrt()
    } else {
        f32::NAN
    };
    let rmse_over_ref_rms = rmse / ref_rms.max(1.0e-12);
    let gate_pass = arle_nonfinite_count == 0
        && pytorch_nonfinite_count == 0
        && rmse_over_ref_rms <= RMSE_REF_RMS_GATE;

    Ok(ParityReport {
        model_path: model_path.to_string(),
        reference_model_path: reference_model_path.to_string(),
        token_id,
        python_device: python_device.to_string(),
        layer0_type: layer0_type.to_string(),
        len,
        finite_pair_count,
        arle_nonfinite_count,
        pytorch_nonfinite_count,
        pytorch_finite_count,
        max_abs,
        max_rel,
        mean_abs,
        rmse,
        ref_rms,
        rmse_over_ref_rms,
        gate_rmse_over_ref_rms: RMSE_REF_RMS_GATE,
        gate_pass,
        first8,
    })
}

fn print_report(report: &ParityReport) {
    println!("model_path={}", report.model_path);
    println!("reference_model_path={}", report.reference_model_path);
    println!(
        "token_id={} python_device={} layer0_type={}",
        report.token_id, report.python_device, report.layer0_type
    );
    println!(
        "len={} finite_pairs={} arle_nonfinite={} pytorch_nonfinite={} pytorch_finite={} max_abs={:.8e} max_rel={:.8e} mean_abs={:.8e} rmse={:.8e} ref_rms={:.8e} rmse/ref_rms={:.8e} gate={:.8e} pass={}",
        report.len,
        report.finite_pair_count,
        report.arle_nonfinite_count,
        report.pytorch_nonfinite_count,
        report.pytorch_finite_count,
        report.max_abs,
        report.max_rel,
        report.mean_abs,
        report.rmse,
        report.ref_rms,
        report.rmse_over_ref_rms,
        report.gate_rmse_over_ref_rms,
        report.gate_pass
    );
    println!();
    println!("| index | ARLE | PyTorch | abs_err | rel_err |");
    println!("| ---: | ---: | ---: | ---: | ---: |");
    for entry in &report.first8 {
        println!(
            "| {} | {:.8e} | {:.8e} | {:.8e} | {:.8e} |",
            entry.index, entry.arle, entry.pytorch, entry.abs_err, entry.rel_err
        );
    }
}
