use std::process::ExitCode;

use crate::args::{ModelArgs, ModelCommand, ModelDownloadArgs, ModelSourceArg};

pub(crate) fn run_model(model: ModelArgs) -> ExitCode {
    match model.command {
        ModelCommand::Download(args) => run_model_download(args),
    }
}

fn run_model_download(args: ModelDownloadArgs) -> ExitCode {
    let source_label = match args.source {
        ModelSourceArg::Hf => "hf",
        ModelSourceArg::Modelscope => "modelscope",
    };
    if args.render.dry_run {
        if args.render.json {
            println!(
                "{}",
                serde_json::json!({
                    "command": "model download",
                    "argv": [args.model_id],
                    "source": source_label,
                })
            );
        } else {
            println!("command model download");
            println!("argv {}", args.model_id);
            println!("source {source_label}");
        }
        return ExitCode::SUCCESS;
    }
    let result = match args.source {
        ModelSourceArg::Hf => crate::download::download_model_with_progress(&args.model_id),
        ModelSourceArg::Modelscope => {
            crate::modelscope::download_model_from_modelscope_with_progress(&args.model_id)
        }
    };
    match result {
        Ok(path) => {
            eprintln!(
                "[ARLE model download] downloaded ({source_label}) to: {}",
                path.display()
            );
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("[ARLE model download] error: {err:#}");
            ExitCode::FAILURE
        }
    }
}
