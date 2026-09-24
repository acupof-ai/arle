use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Result;
use clap::{Parser, Subcommand};
use kernel_gate::{DEFAULT_GATE_PREFIX, Registry, check, parity_gates};

/// Validate an operator registry and list its parity gates.
#[derive(Parser)]
#[command(name = "kernel-gate", version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Parse the registry against the schema and validate every correctness gate.
    Check {
        registry: PathBuf,
        /// Directory the gate paths are relative to.
        #[arg(long, default_value = ".")]
        root: PathBuf,
    },
    /// Print one parity example per line (sorted, deduplicated).
    List {
        registry: PathBuf,
        /// Directory the gate paths are relative to.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Append `<TAB>flags` (comma list from {sm90,model}).
        #[arg(long)]
        flags: bool,
        /// Path prefix that marks a standalone parity example.
        #[arg(long, default_value = DEFAULT_GATE_PREFIX)]
        gate_prefix: String,
    },
}

fn run(cli: Cli) -> Result<bool> {
    match cli.command {
        Command::Check { registry, root } => {
            let parsed = Registry::load(&registry)?;
            let errors = check(&parsed, &root, &registry.display().to_string());
            for error in &errors {
                eprintln!("{error}");
            }
            if errors.is_empty() {
                println!(
                    "kernel-gate: {} OK ({} semantic, {} implementation, {} policy)",
                    registry.display(),
                    parsed.semantic.len(),
                    parsed.implementation.len(),
                    parsed.policy.len()
                );
            }
            Ok(errors.is_empty())
        }
        Command::List {
            registry,
            root,
            flags,
            gate_prefix,
        } => {
            let parsed = Registry::load(&registry)?;
            for gate in parity_gates(&parsed, &root, &gate_prefix)? {
                if flags {
                    println!("{}\t{}", gate.name, gate.flags());
                } else {
                    println!("{}", gate.name);
                }
            }
            Ok(true)
        }
    }
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(err) => {
            eprintln!("{err:#}");
            ExitCode::FAILURE
        }
    }
}
