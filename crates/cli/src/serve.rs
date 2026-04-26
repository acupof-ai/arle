use std::{
    env,
    path::PathBuf,
    process::{Command, ExitCode},
};

use crate::{
    args::{Args, BackendArg, ServeArgs},
    hardware::CompiledBackend,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ServeBackend {
    Cuda,
    Metal,
    Cpu,
}

impl ServeBackend {
    fn binary_name(self) -> &'static str {
        match self {
            Self::Cuda => "infer",
            Self::Metal => "metal_serve",
            Self::Cpu => "cpu_serve",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Metal => "metal",
            Self::Cpu => "cpu",
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct ServeInvocation {
    backend: ServeBackend,
    binary: PathBuf,
    argv: Vec<String>,
    bind_warning: Option<String>,
}

pub(crate) fn run_serve(args: &Args, serve_args: ServeArgs) -> ExitCode {
    match resolve_invocation(args, &serve_args) {
        Ok(invocation) => run_invocation(invocation),
        Err(err) => {
            eprintln!("[ARLE serve] error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run_invocation(invocation: ServeInvocation) -> ExitCode {
    if let Some(warning) = invocation.bind_warning.as_deref() {
        eprintln!("[ARLE serve] warning: {warning}");
    }
    eprintln!(
        "[ARLE serve] launching {} backend via {}",
        invocation.backend.label(),
        invocation.binary.display()
    );
    let status = Command::new(&invocation.binary)
        .args(&invocation.argv)
        .status();
    match status {
        Ok(status) if status.success() => ExitCode::SUCCESS,
        Ok(status) => ExitCode::from(status.code().unwrap_or(1) as u8),
        Err(err) => {
            eprintln!(
                "[ARLE serve] error: failed to launch {}: {err}",
                invocation.binary.display()
            );
            ExitCode::FAILURE
        }
    }
}

fn resolve_invocation(args: &Args, serve_args: &ServeArgs) -> Result<ServeInvocation, String> {
    let backend = resolve_backend(serve_args.backend)?;
    let model_path = serve_args
        .model_path
        .as_deref()
        .or(args.model_path.as_deref())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .or_else(model_from_env)
        .ok_or_else(|| {
            "no model selected; pass `arle serve --model-path ...`, top-level `--model-path`, or set ARLE_MODEL".to_string()
        })?;

    let mut argv = vec![
        "--model-path".to_string(),
        model_path,
        "--port".to_string(),
        serve_args.port.to_string(),
    ];

    let bind_warning = if backend == ServeBackend::Metal {
        argv.push("--bind".to_string());
        argv.push(serve_args.bind.clone());
        None
    } else if serve_args.bind != "127.0.0.1" {
        Some(format!(
            "--bind={} is only supported by the Metal serving binary today; {} will use its backend default",
            serve_args.bind,
            backend.binary_name()
        ))
    } else {
        None
    };

    if backend == ServeBackend::Cuda && args.no_cuda_graph {
        argv.push("--cuda-graph".to_string());
        argv.push("false".to_string());
    }

    if let Some(url) = serve_args.train_control_url.as_deref() {
        argv.push("--train-control-url".to_string());
        argv.push(url.to_string());
    }

    argv.extend(serve_args.extra_args.iter().cloned());

    Ok(ServeInvocation {
        backend,
        binary: resolve_binary(backend.binary_name()),
        argv,
        bind_warning,
    })
}

fn resolve_backend(arg: BackendArg) -> Result<ServeBackend, String> {
    match arg {
        BackendArg::Cuda => Ok(ServeBackend::Cuda),
        BackendArg::Metal => Ok(ServeBackend::Metal),
        BackendArg::Cpu => Ok(ServeBackend::Cpu),
        BackendArg::Auto => match CompiledBackend::detect() {
            CompiledBackend::Cuda => Ok(ServeBackend::Cuda),
            CompiledBackend::Metal => Ok(ServeBackend::Metal),
            CompiledBackend::Cpu => Ok(ServeBackend::Cpu),
            #[cfg(not(any(feature = "cuda", feature = "metal", feature = "cpu")))]
            CompiledBackend::None => Err(
                "serve requires a backend build; rebuild with cuda, metal/no-cuda, or cpu/no-cuda"
                    .to_string(),
            ),
        },
    }
}

fn model_from_env() -> Option<String> {
    env::var("ARLE_MODEL")
        .ok()
        .or_else(|| env::var("AGENT_INFER_MODEL").ok())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn resolve_binary(name: &str) -> PathBuf {
    if let Some(sibling) = current_exe_sibling(name)
        && sibling.is_file()
    {
        return sibling;
    }
    PathBuf::from(name)
}

fn current_exe_sibling(name: &str) -> Option<PathBuf> {
    let exe = env::current_exe().ok()?;
    let dir = exe.parent()?;
    Some(dir.join(platform_binary_name(name)))
}

fn platform_binary_name(name: &str) -> String {
    if cfg!(target_os = "windows") {
        format!("{name}.exe")
    } else {
        name.to_string()
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;

    #[test]
    fn serve_uses_subcommand_model_path() {
        let mut args = Args::parse_from([
            "arle",
            "serve",
            "--backend",
            "cpu",
            "--model-path",
            "from-sub",
        ]);
        let serve = match args.command.take().expect("serve command") {
            crate::args::CliCommand::Serve(serve) => *serve,
            _ => panic!("expected serve"),
        };
        let invocation = resolve_invocation(&args, &serve).expect("resolve");
        assert_eq!(invocation.argv[1], "from-sub");
    }

    #[test]
    fn serve_uses_top_level_model_path() {
        let mut args = Args::parse_from([
            "arle",
            "--model-path",
            "from-root",
            "serve",
            "--backend",
            "cpu",
        ]);
        let serve = match args.command.take().expect("serve command") {
            crate::args::CliCommand::Serve(serve) => *serve,
            _ => panic!("expected serve"),
        };
        let invocation = resolve_invocation(&args, &serve).expect("resolve");
        assert_eq!(invocation.argv[1], "from-root");
    }

    #[test]
    fn cuda_serve_forwards_no_cuda_graph() {
        let mut args = Args::parse_from([
            "arle",
            "--no-cuda-graph",
            "serve",
            "--backend",
            "cuda",
            "--model-path",
            "model",
        ]);
        let serve = match args.command.take().expect("serve command") {
            crate::args::CliCommand::Serve(serve) => *serve,
            _ => panic!("expected serve"),
        };
        let invocation = resolve_invocation(&args, &serve).expect("resolve");
        assert!(
            invocation
                .argv
                .windows(2)
                .any(|item| item[0] == "--cuda-graph" && item[1] == "false")
        );
    }
}
