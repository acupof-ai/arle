use std::{env, path::PathBuf};

use anyhow::{Context, Result, bail};
use infer::model::Qwen35Model;
use serde_json::json;

const DEFAULT_MODEL_DIR: &str = "/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B-TQ4";
const DEFAULT_TOKEN_ID: u32 = 9419;

struct Args {
    model_path: PathBuf,
    token_id: u32,
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let model_path = args
        .model_path
        .to_str()
        .context("model path is not valid UTF-8")?;
    let model = Qwen35Model::from_safetensors_with_options(model_path, false)
        .with_context(|| format!("load Qwen3.5 model from {model_path}"))?;
    let ctx = model.parity_device_context();
    let outputs = model.dense_module_parity_outputs(args.token_id)?;

    let embedding = outputs.embedding.to_host(&ctx)?;
    let final_rmsnorm = outputs.final_rmsnorm.to_host(&ctx)?;
    let lm_head = outputs.lm_head.to_host(&ctx)?;

    let payload = json!({
        "model_path": args.model_path,
        "token_id": args.token_id,
        "input_contract": {
            "embedding": "token embedding for token_id",
            "final_rmsnorm": "deterministic bf16 vector, salt=17, value=((idx*37 + salt*17) % 257 - 128) / 64",
            "lm_head": "deterministic bf16 vector, salt=29, value=((idx*37 + salt*17) % 257 - 128) / 64"
        },
        "embedding": embedding,
        "final_rmsnorm": final_rmsnorm,
        "lm_head": lm_head,
    });
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    std::fs::write(&args.output, serde_json::to_vec(&payload)?)
        .with_context(|| format!("write {}", args.output.display()))?;
    println!(
        "dense_module_dump model={} token_id={} embedding_len={} final_rmsnorm_len={} lm_head_len={} output={}",
        args.model_path.display(),
        args.token_id,
        payload["embedding"].as_array().map_or(0, Vec::len),
        payload["final_rmsnorm"].as_array().map_or(0, Vec::len),
        payload["lm_head"].as_array().map_or(0, Vec::len),
        args.output.display()
    );
    Ok(())
}

fn parse_args() -> Result<Args> {
    let mut model_path = env::var_os("ARLE_QWEN35_DENSE_MODULE_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_DIR));
    let mut token_id = DEFAULT_TOKEN_ID;
    let mut output = None;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model-path" => model_path = PathBuf::from(next_arg(&mut args, &arg)?),
            "--token-id" => token_id = next_arg(&mut args, &arg)?.parse()?,
            "--output" => output = Some(PathBuf::from(next_arg(&mut args, &arg)?)),
            "--help" | "-h" => {
                println!(
                    "usage: cargo run -p infer --example qwen35_dense_module_dump --release --features cuda -- \
                     --model-path DIR --token-id 9419 --output dense-modules.json"
                );
                std::process::exit(0);
            }
            other => bail!("unknown argument `{other}`"),
        }
    }

    Ok(Args {
        model_path,
        token_id,
        output: output.context("--output is required")?,
    })
}

fn next_arg(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    args.next()
        .with_context(|| format!("{flag} requires a value"))
}
