//! CLI: download a single file from a HuggingFace dataset repo and print
//! the resulting local cache path on stdout. Designed to be chained with
//! other trainer binaries:
//!
//! ```bash
//! DATA=$(download_dataset --repo allenai/tulu-3-sft-mixture \
//!                         --file data/train.jsonl)
//! train_sft --data "$DATA" ...
//! ```
//!
//! Stderr is reserved for progress / log output; stdout is *only* the
//! resolved path (with a trailing newline). This lets shell substitution
//! work cleanly.

use std::process::ExitCode;

use train::hub_dataset::download_dataset_file;

fn main() -> ExitCode {
    dispatch_from_args(std::env::args().skip(1).collect::<Vec<_>>())
}

pub(crate) fn dispatch_from_args<I>(args: I) -> ExitCode
where
    I: IntoIterator<Item = String>,
{
    let mut repo: Option<String> = None;
    let mut file: Option<String> = None;

    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--repo" => repo = args.next(),
            "--file" => file = args.next(),
            "--help" | "-h" => {
                eprintln!("usage: download_dataset --repo <hf-dataset-id> --file <path-in-repo>");
                return ExitCode::from(0);
            }
            other => {
                eprintln!("unknown flag: {other}");
                return ExitCode::from(2);
            }
        }
    }

    let Some(repo) = repo else {
        eprintln!("error: --repo is required (e.g. 'allenai/tulu-3-sft-mixture')");
        return ExitCode::from(2);
    };
    let Some(file) = file else {
        eprintln!("error: --file is required (e.g. 'data/train.jsonl')");
        return ExitCode::from(2);
    };

    eprintln!("[download_dataset] fetching '{file}' from dataset '{repo}'");
    match download_dataset_file(&repo, &file) {
        Ok(path) => {
            eprintln!("[download_dataset] ready: {}", path.display());
            println!("{}", path.display());
            ExitCode::from(0)
        }
        Err(err) => {
            eprintln!("[download_dataset] error: {err:#}");
            ExitCode::from(1)
        }
    }
}
