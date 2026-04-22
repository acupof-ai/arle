use std::io::IsTerminal;
use std::path::PathBuf;

use anyhow::Result;
use console::style;

use crate::args::Args;
use crate::hardware::{self, GpuInfo};
use crate::hub_discovery;
use crate::model_catalog;

struct DoctorSnapshot {
    info: hardware::SystemInfo,
    env_model: Option<String>,
    discovered: Option<(String, PathBuf)>,
    snapshots: Vec<hub_discovery::HubSnapshot>,
    recommendations: Vec<&'static model_catalog::CatalogEntry>,
    selected: Result<SelectedModelSource>,
}

pub(crate) fn run(args: &Args) -> Result<()> {
    let snapshot = collect_snapshot(args);

    println!("{}", style("agent-infer doctor").bold().cyan());
    println!();
    println!(
        "{} {}",
        style("version").dim(),
        style(env!("CARGO_PKG_VERSION")).bold()
    );
    println!(
        "{} {}",
        style("compiled backend").dim(),
        style(snapshot.info.compiled_backend.name()).bold()
    );
    println!(
        "{} stdin={} stdout={} stderr={}",
        style("tty").dim(),
        std::io::stdin().is_terminal(),
        std::io::stdout().is_terminal(),
        std::io::stderr().is_terminal()
    );
    println!(
        "{} {} · {} cores",
        style("cpu").dim(),
        snapshot.info.cpu_name,
        snapshot.info.cpu_cores
    );
    println!(
        "{} {:.1} GB total · {:.1} GB free",
        style("ram").dim(),
        snapshot.info.total_ram_gb,
        snapshot.info.available_ram_gb
    );
    println!(
        "{} {:.1} GB",
        style("effective memory").dim(),
        snapshot.info.effective_memory_gb()
    );
    println!("{} {}", style("gpu").dim(), gpu_summary(&snapshot.info.gpu));
    println!();

    println!(
        "{} {}",
        style("env AGENT_INFER_MODEL").dim(),
        snapshot.env_model.as_deref().unwrap_or("<unset>")
    );
    println!(
        "{} {}",
        style("arg --model-path").dim(),
        args.model_path.as_deref().unwrap_or("<unset>")
    );
    println!(
        "{} {}",
        style("hf cache").dim(),
        hub_discovery::hub_cache_root()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "<unavailable>".to_string())
    );
    println!();

    println!("{}", style("Resolution").bold());
    match &snapshot.selected {
        Ok(source) => {
            println!("{} {}", style("selected via").dim(), source.origin_label());
            println!("{} {}", style("selected source").dim(), source.value());
            match source.local_path() {
                Some(path) => println!("{} {}", style("local path").dim(), path.display()),
                None => println!("{} <not present locally>", style("local path").dim()),
            }
        }
        Err(err) => {
            println!("{} {err:#}", style("resolution error").red().bold());
        }
    }
    println!();

    print_discovery_section(&snapshot);
    print_recommendations_section(&snapshot);

    println!("{}", style("Checks").bold());
    if snapshot.info.compiled_backend.supports_inference() {
        println!(
            "{} {} backend is available in this binary",
            style("backend").dim(),
            snapshot.info.compiled_backend.name()
        );
    } else {
        println!(
            "{} rebuild with `cuda`, `metal,no-cuda`, or `cpu,no-cuda` to run inference",
            style("backend").yellow().bold()
        );
    }
    if snapshot.recommendations.is_empty() && snapshot.info.compiled_backend.supports_inference() {
        println!(
            "{} detected memory does not fit any curated model for this backend",
            style("catalog").yellow().bold()
        );
    }
    if snapshot.selected.is_err() {
        println!(
            "{} set `--model-path`, `AGENT_INFER_MODEL`, or cache a supported local model",
            style("resolution").yellow().bold()
        );
    }

    Ok(())
}

pub(crate) fn list_models(args: &Args) -> Result<()> {
    let snapshot = collect_snapshot(args);

    println!("{}", style("agent-infer models").bold().cyan());
    println!();
    println!("{}", style("Resolution").bold());
    match &snapshot.selected {
        Ok(source) => {
            println!("{} {}", style("selected via").dim(), source.origin_label());
            println!("{} {}", style("selected source").dim(), source.value());
            match source.local_path() {
                Some(path) => println!("{} {}", style("local path").dim(), path.display()),
                None => println!("{} <not present locally>", style("local path").dim()),
            }
        }
        Err(err) => {
            println!("{} {err:#}", style("resolution error").red().bold());
        }
    }
    println!();

    print_discovery_section(&snapshot);
    print_recommendations_section(&snapshot);

    Ok(())
}

fn collect_snapshot(args: &Args) -> DoctorSnapshot {
    let info = hardware::detect_system();
    let env_model =
        non_empty_value(std::env::var("AGENT_INFER_MODEL").ok().as_deref()).map(ToOwned::to_owned);
    let discovered = infer::hf_hub::discover_local_model();
    let snapshots = hub_discovery::discover_hub_snapshots();
    let recommendations = model_catalog::recommend_models(&info);
    let selected = select_model_source(
        args.model_path.as_deref(),
        env_model.as_deref(),
        discovered.clone(),
    );
    DoctorSnapshot {
        info,
        env_model,
        discovered,
        snapshots,
        recommendations,
        selected,
    }
}

fn print_discovery_section(snapshot: &DoctorSnapshot) {
    println!("{}", style("Discovery").bold());
    match &snapshot.discovered {
        Some((candidate, path)) => {
            println!(
                "{} {} -> {}",
                style("preferred local model").dim(),
                candidate,
                path.display()
            );
        }
        None => println!("{} <none>", style("preferred local model").dim()),
    }
    if snapshot.snapshots.is_empty() {
        println!("{} <none>", style("supported hub snapshots").dim());
    } else {
        println!(
            "{} {}",
            style("supported hub snapshots").dim(),
            snapshot.snapshots.len()
        );
        for item in snapshot.snapshots.iter().take(5) {
            println!("  - {} -> {}", item.model_id, item.path.display());
        }
        if snapshot.snapshots.len() > 5 {
            println!("  - ... and {} more", snapshot.snapshots.len() - 5);
        }
    }
    println!();
}

fn print_recommendations_section(snapshot: &DoctorSnapshot) {
    println!("{}", style("Recommendations").bold());
    if snapshot.recommendations.is_empty() {
        println!("{} <none fit this machine>", style("catalog").dim());
    } else {
        for entry in snapshot.recommendations.iter().take(5) {
            let quant = entry
                .quantization
                .map(|value| format!(" ({value})"))
                .unwrap_or_default();
            println!(
                "  - {}{} · {} · {:.1} GB download · needs {:.1} GB",
                entry.display_name, quant, entry.hf_id, entry.size_gb, entry.min_memory_gb
            );
        }
    }
    println!();
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SelectedModelSource {
    Explicit(String),
    Environment(String),
    AutoDiscovered { model_id: String, path: PathBuf },
}

impl SelectedModelSource {
    fn origin_label(&self) -> &'static str {
        match self {
            Self::Explicit(_) => "--model-path",
            Self::Environment(_) => "AGENT_INFER_MODEL",
            Self::AutoDiscovered { .. } => "auto-discovered local model",
        }
    }

    fn value(&self) -> &str {
        match self {
            Self::Explicit(value) | Self::Environment(value) => value,
            Self::AutoDiscovered { model_id, .. } => model_id,
        }
    }

    fn local_path(&self) -> Option<PathBuf> {
        match self {
            Self::Explicit(value) | Self::Environment(value) => {
                infer::hf_hub::resolve_local_model_path(value)
            }
            Self::AutoDiscovered { path, .. } => Some(path.clone()),
        }
    }
}

fn select_model_source(
    explicit_model_path: Option<&str>,
    env_model: Option<&str>,
    discovered: Option<(String, PathBuf)>,
) -> Result<SelectedModelSource> {
    if let Some(model_path) = non_empty_value(explicit_model_path) {
        return Ok(SelectedModelSource::Explicit(model_path.to_string()));
    }
    if let Some(model_path) = non_empty_value(env_model) {
        return Ok(SelectedModelSource::Environment(model_path.to_string()));
    }
    if let Some((model_id, path)) = discovered {
        return Ok(SelectedModelSource::AutoDiscovered { model_id, path });
    }

    anyhow::bail!(
        "no model selected; pass --model-path, set AGENT_INFER_MODEL, or place a supported model in the local HuggingFace cache"
    )
}

fn non_empty_value(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

fn gpu_summary(gpu: &GpuInfo) -> String {
    match gpu {
        GpuInfo::Cuda { name, vram_gb } => format!("{name} · {vram_gb:.1} GB VRAM"),
        GpuInfo::Metal {
            chip,
            unified_memory_gb,
        } => format!("{chip} · {unified_memory_gb:.0} GB unified memory"),
        GpuInfo::None => "none detected".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::{SelectedModelSource, select_model_source};
    use std::path::PathBuf;

    #[test]
    fn prefers_explicit_model_path() {
        let selected = select_model_source(
            Some(" /tmp/model "),
            Some("Qwen/Qwen3-4B"),
            Some(("Qwen/Qwen3-0.6B".to_string(), PathBuf::from("/tmp/auto"))),
        )
        .expect("explicit path should win");
        assert_eq!(
            selected,
            SelectedModelSource::Explicit("/tmp/model".to_string())
        );
    }

    #[test]
    fn falls_back_to_env_model() {
        let selected = select_model_source(
            None,
            Some(" Qwen/Qwen3-4B "),
            Some(("Qwen/Qwen3-0.6B".to_string(), PathBuf::from("/tmp/auto"))),
        )
        .expect("env model should win when explicit path is absent");
        assert_eq!(
            selected,
            SelectedModelSource::Environment("Qwen/Qwen3-4B".to_string())
        );
    }

    #[test]
    fn falls_back_to_auto_discovered_model() {
        let selected = select_model_source(
            None,
            None,
            Some(("Qwen/Qwen3-0.6B".to_string(), PathBuf::from("/tmp/auto"))),
        )
        .expect("auto-discovered model should win when explicit/env are absent");
        assert_eq!(
            selected,
            SelectedModelSource::AutoDiscovered {
                model_id: "Qwen/Qwen3-0.6B".to_string(),
                path: PathBuf::from("/tmp/auto"),
            }
        );
    }

    #[test]
    fn errors_when_no_model_source_is_available() {
        let err = select_model_source(None, Some(" "), None)
            .err()
            .expect("missing model source should fail");
        assert!(
            err.to_string()
                .contains("no model selected; pass --model-path")
        );
    }
}
