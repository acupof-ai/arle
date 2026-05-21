#![cfg_attr(not(feature = "cuda"), allow(dead_code, unused_imports))]

#[cfg(feature = "cuda")]
mod app {
    use std::{path::PathBuf, time::Instant};

    use anyhow::{Context, Result, bail};
    use infer::server_engine::{InferenceEngineOptions, LoadedInferenceEngine};
    use serde_json::json;

    #[derive(Debug)]
    struct Args {
        model_path: PathBuf,
        input_ids: Vec<u32>,
        output: PathBuf,
        cuda_graph: bool,
    }

    pub(crate) fn main() -> Result<()> {
        let args = parse_args()?;
        let positions = (0..args.input_ids.len() as u32).collect::<Vec<_>>();
        let load_started = Instant::now();
        let engine = LoadedInferenceEngine::load_with_options(
            args.model_path
                .to_str()
                .context("model path is not valid UTF-8")?,
            42,
            InferenceEngineOptions {
                enable_cuda_graph: args.cuda_graph,
            },
        )?;
        let load_seconds = load_started.elapsed().as_secs_f64();

        let forward_started = Instant::now();
        let logits = engine.forward_token_logits(&args.input_ids, &positions)?;
        let forward_seconds = forward_started.elapsed().as_secs_f64();
        let host_started = Instant::now();
        let host = logits.to_host_f32()?;
        let host_seconds = host_started.elapsed().as_secs_f64();

        let payload = json!({
            "model_path": args.model_path,
            "input_ids": args.input_ids,
            "positions": positions,
            "seq_len": logits.seq_len(),
            "vocab_size": logits.vocab_size(),
            "load_seconds": load_seconds,
            "forward_seconds": forward_seconds,
            "host_readback_seconds": host_seconds,
            "logits": host,
        });
        std::fs::write(&args.output, serde_json::to_vec(&payload)?)?;
        println!(
            "raw_logits_dump model={} output={} seq_len={} vocab_size={} load_seconds={load_seconds:.6} forward_seconds={forward_seconds:.6} host_readback_seconds={host_seconds:.6}",
            args.model_path.display(),
            args.output.display(),
            logits.seq_len(),
            logits.vocab_size(),
        );
        Ok(())
    }

    fn parse_args() -> Result<Args> {
        let mut model_path = None;
        let mut input_ids = None;
        let mut output = None;
        let mut cuda_graph = false;

        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model-path" => model_path = Some(PathBuf::from(next_arg(&mut args, &arg)?)),
                "--input-ids" => input_ids = Some(parse_ids(&next_arg(&mut args, &arg)?)?),
                "--output" => output = Some(PathBuf::from(next_arg(&mut args, &arg)?)),
                "--cuda-graph" => cuda_graph = parse_bool(&next_arg(&mut args, &arg)?)?,
                "--help" | "-h" => {
                    println!(
                        "usage: cargo run -p infer --example raw_token_logits_dump --release --features cuda -- \
                         --model-path DIR --input-ids 9419 --output logits.json [--cuda-graph false]"
                    );
                    std::process::exit(0);
                }
                _ => bail!("unknown argument `{arg}`"),
            }
        }

        let model_path = model_path.context("--model-path is required")?;
        let input_ids = input_ids.context("--input-ids is required")?;
        if input_ids.is_empty() {
            bail!("--input-ids must contain at least one id");
        }
        let output = output.context("--output is required")?;
        Ok(Args {
            model_path,
            input_ids,
            output,
            cuda_graph,
        })
    }

    fn next_arg(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
        args.next()
            .with_context(|| format!("{flag} requires a value"))
    }

    fn parse_ids(raw: &str) -> Result<Vec<u32>> {
        raw.split(',')
            .filter(|item| !item.trim().is_empty())
            .map(|item| {
                item.trim()
                    .parse::<u32>()
                    .with_context(|| format!("invalid token id `{}`", item.trim()))
            })
            .collect()
    }

    fn parse_bool(raw: &str) -> Result<bool> {
        match raw {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            _ => bail!("invalid bool `{raw}`"),
        }
    }
}

#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    app::main()
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("raw_token_logits_dump requires --features cuda");
}
