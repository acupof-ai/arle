use std::{
    env, fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow, bail};
use infer::model::Qwen35Model;
use serde::Serialize;

const DEFAULT_MODEL_DIR: &str = "/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base";
const DEFAULT_TOKEN_ID: u32 = 9419;

#[derive(Debug)]
struct Args {
    model_path: PathBuf,
    token_id: u32,
    output: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct SubstageReport {
    model_path: String,
    token_id: u32,
    layer0_type: String,
    stages: Vec<StageSummary>,
    first_nonfinite_stage: Option<String>,
}

#[derive(Debug, Serialize)]
struct StageSummary {
    name: String,
    len: usize,
    finite_count: usize,
    nan_count: usize,
    pos_inf_count: usize,
    neg_inf_count: usize,
    max_abs: f32,
    mean_abs: f32,
    first8: Vec<f32>,
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let layer0_type = read_layer0_type(&args.model_path)?;
    if layer0_type != "linear_attention" {
        bail!("layer 0 is {layer0_type}, expected linear_attention for this substage dump");
    }

    let model_path_str = args
        .model_path
        .to_str()
        .context("model path is not valid UTF-8")?;
    let model = Qwen35Model::from_safetensors_with_options(model_path_str, false)
        .with_context(|| format!("load ARLE Qwen3.5 model from {model_path_str}"))?;
    let tensors = model.layer0_linear_attention_diagnostic_tensors(args.token_id)?;
    let stages: Vec<_> = tensors
        .into_iter()
        .map(|tensor| summarize(tensor.name, &tensor.values))
        .collect();
    let first_nonfinite_stage = stages
        .iter()
        .find(|stage| stage.nan_count + stage.pos_inf_count + stage.neg_inf_count > 0)
        .map(|stage| stage.name.clone());

    let report = SubstageReport {
        model_path: model_path_str.to_string(),
        token_id: args.token_id,
        layer0_type,
        stages,
        first_nonfinite_stage,
    };
    print_report(&report);

    if let Some(output) = &args.output {
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
        }
        fs::write(output, serde_json::to_string_pretty(&report)?)
            .with_context(|| format!("write {}", output.display()))?;
    }

    Ok(())
}

fn parse_args() -> Result<Args> {
    let mut args = Args {
        model_path: env::var_os("ARLE_QWEN35_LINEAR_ATTN_MODEL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_DIR)),
        token_id: DEFAULT_TOKEN_ID,
        output: None,
    };

    let mut iter = env::args().skip(1);
    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--model-path" => args.model_path = next_path(&mut iter, "--model-path")?,
            "--token-id" => args.token_id = next_value(&mut iter, "--token-id")?.parse()?,
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
        "qwen35_linear_attn_substage_dump [--model-path PATH] [--token-id ID] [--output PATH]"
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

fn summarize(name: &str, values: &[f32]) -> StageSummary {
    let mut finite_count = 0usize;
    let mut nan_count = 0usize;
    let mut pos_inf_count = 0usize;
    let mut neg_inf_count = 0usize;
    let mut max_abs = 0.0f32;
    let mut abs_sum = 0.0f32;

    for &value in values {
        if value.is_nan() {
            nan_count += 1;
        } else if value == f32::INFINITY {
            pos_inf_count += 1;
        } else if value == f32::NEG_INFINITY {
            neg_inf_count += 1;
        } else {
            finite_count += 1;
            let abs = value.abs();
            max_abs = max_abs.max(abs);
            abs_sum += abs;
        }
    }

    StageSummary {
        name: name.to_string(),
        len: values.len(),
        finite_count,
        nan_count,
        pos_inf_count,
        neg_inf_count,
        max_abs,
        mean_abs: if finite_count > 0 {
            abs_sum / finite_count as f32
        } else {
            f32::NAN
        },
        first8: values.iter().copied().take(8).collect(),
    }
}

fn print_report(report: &SubstageReport) {
    println!("model_path={}", report.model_path);
    println!(
        "token_id={} layer0_type={} first_nonfinite_stage={}",
        report.token_id,
        report.layer0_type,
        report.first_nonfinite_stage.as_deref().unwrap_or("<none>")
    );
    println!();
    println!("| stage | len | finite | nan | +inf | -inf | max_abs | mean_abs | first8 |");
    println!("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |");
    for stage in &report.stages {
        println!(
            "| {} | {} | {} | {} | {} | {} | {:.8e} | {:.8e} | {:?} |",
            stage.name,
            stage.len,
            stage.finite_count,
            stage.nan_count,
            stage.pos_inf_count,
            stage.neg_inf_count,
            stage.max_abs,
            stage.mean_abs,
            stage.first8,
        );
    }
}
