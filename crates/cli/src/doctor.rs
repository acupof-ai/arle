use std::io::IsTerminal;
use std::path::PathBuf;

use anyhow::Result;
use console::style;
use serde::Serialize;

use crate::args::Args;
use crate::hardware::{self, GpuInfo};
use crate::hub_discovery;
use crate::model_catalog;

const INSPECTION_SCHEMA_VERSION: u32 = 3;
const PRIMARY_MODEL_ENV: &str = "ARLE_MODEL";
const LEGACY_MODEL_ENV: &str = "AGENT_INFER_MODEL";

struct DoctorSnapshot {
    info: hardware::SystemInfo,
    tty: TtyState,
    env_model: Option<String>,
    arg_model_path: Option<String>,
    hf_cache_root: Option<PathBuf>,
    discovered: Option<(String, PathBuf)>,
    snapshots: Vec<hub_discovery::HubSnapshot>,
    recommendations: Vec<&'static model_catalog::CatalogEntry>,
    selected: Result<SelectedModelSource>,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
struct TtyState {
    stdin: bool,
    stdout: bool,
    stderr: bool,
}

pub(crate) fn run(args: &Args) -> Result<()> {
    let snapshot = collect_snapshot(args);
    let report = doctor_report(&snapshot);
    if args.json {
        print_json(&report)?;
        return enforce_strict(args, &report);
    }

    println!("{}", style("ARLE doctor").bold().cyan());
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
        "{} {}",
        style("host accelerator").dim(),
        gpu_summary(&snapshot.info.gpu)
    );
    println!(
        "{} stdin={} stdout={} stderr={}",
        style("tty").dim(),
        snapshot.tty.stdin,
        snapshot.tty.stdout,
        snapshot.tty.stderr
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
        style("effective backend memory").dim(),
        snapshot.info.effective_memory_gb()
    );
    println!();

    println!(
        "{} {}",
        style("env ARLE_MODEL (legacy: AGENT_INFER_MODEL)").dim(),
        snapshot.env_model.as_deref().unwrap_or("<unset>")
    );
    println!(
        "{} {}",
        style("arg --model-path").dim(),
        snapshot.arg_model_path.as_deref().unwrap_or("<unset>")
    );
    println!(
        "{} {}",
        style("hf cache").dim(),
        snapshot
            .hf_cache_root
            .as_ref()
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
    print_tools_section();

    println!("{}", style("Checks").bold());
    if snapshot.info.compiled_backend.supports_inference() {
        println!(
            "{} {} backend is available in this binary",
            style("backend").dim(),
            snapshot.info.compiled_backend.name()
        );
    } else {
        println!(
            "{} {}",
            style("backend").yellow().bold(),
            rebuild_backend_message(&snapshot.info.gpu)
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
            "{} set `--model-path`, `ARLE_MODEL` (legacy `AGENT_INFER_MODEL` also works), or cache a supported local model",
            style("resolution").yellow().bold()
        );
    }

    enforce_strict(args, &report)
}

pub(crate) fn list_models(args: &Args) -> Result<()> {
    let snapshot = collect_snapshot(args);
    if args.json {
        return print_json(&models_report(&snapshot));
    }

    println!("{}", style("ARLE models").bold().cyan());
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
    let tty = TtyState {
        stdin: std::io::stdin().is_terminal(),
        stdout: std::io::stdout().is_terminal(),
        stderr: std::io::stderr().is_terminal(),
    };
    let env_model = preferred_model_env_value();
    let arg_model_path = non_empty_value(args.model_path.as_deref()).map(ToOwned::to_owned);
    let hf_cache_root = hub_discovery::hub_cache_root();
    let discovered = infer::hf_hub::discover_local_model();
    let snapshots = hub_discovery::discover_hub_snapshots();
    let recommendations = model_catalog::recommend_models(&info);
    let selected = select_model_source(
        arg_model_path.as_deref(),
        env_model.as_deref(),
        discovered.clone(),
    );
    DoctorSnapshot {
        info,
        tty,
        env_model,
        arg_model_path,
        hf_cache_root,
        discovered,
        snapshots,
        recommendations,
        selected,
    }
}

#[derive(Debug, Serialize)]
struct DoctorJsonReport {
    schema_version: u32,
    mode: &'static str,
    status: &'static str,
    version: &'static str,
    compiled_backend: String,
    supports_inference: bool,
    tty: TtyState,
    cpu: CpuReport,
    ram_gb: RamReport,
    gpu: GpuReport,
    inputs: InputsReport,
    resolution: ResolutionReport,
    discovery: DiscoveryReport,
    recommendations: Vec<ModelRecommendationReport>,
    tools: tools::ToolRuntimeReport,
    checks: Vec<CheckReport>,
}

#[derive(Debug, Serialize)]
struct ModelsJsonReport {
    schema_version: u32,
    mode: &'static str,
    status: &'static str,
    version: &'static str,
    compiled_backend: String,
    resolution: ResolutionReport,
    discovery: DiscoveryReport,
    recommendations: Vec<ModelRecommendationReport>,
}

#[derive(Debug, Serialize)]
struct CpuReport {
    name: String,
    cores: usize,
}

#[derive(Debug, Serialize)]
struct RamReport {
    total: f64,
    available: f64,
    effective: f64,
}

#[derive(Debug, Serialize)]
struct GpuReport {
    kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    memory_gb: Option<f64>,
}

#[derive(Debug, Serialize)]
struct InputsReport {
    env_model: Option<String>,
    model_path_arg: Option<String>,
    hf_cache_root: Option<String>,
}

#[derive(Debug, Serialize)]
struct ResolutionReport {
    status: &'static str,
    selected: Option<SelectedModelReport>,
    error_code: Option<&'static str>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct SelectedModelReport {
    origin: &'static str,
    value: String,
    local_path: Option<String>,
}

#[derive(Debug, Serialize)]
struct DiscoveryReport {
    preferred_local_model: Option<DiscoveredModelReport>,
    supported_hub_snapshots: Vec<HubSnapshotReport>,
}

#[derive(Debug, Serialize)]
struct DiscoveredModelReport {
    model_id: String,
    path: String,
}

#[derive(Debug, Serialize)]
struct HubSnapshotReport {
    model_id: String,
    path: String,
}

#[derive(Debug, Serialize)]
struct ModelRecommendationReport {
    hf_id: String,
    display_name: String,
    quantization: Option<String>,
    size_gb: f64,
    min_memory_gb: f64,
}

#[derive(Debug, Serialize)]
struct CheckReport {
    code: &'static str,
    name: &'static str,
    status: &'static str,
    message: String,
}

fn print_json<T: Serialize>(value: &T) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

fn enforce_strict(args: &Args, report: &DoctorJsonReport) -> Result<()> {
    if args.strict && report.status != "ok" {
        anyhow::bail!(
            "doctor reported status={} under --strict; inspect the checks above",
            report.status
        );
    }
    Ok(())
}

fn doctor_report(snapshot: &DoctorSnapshot) -> DoctorJsonReport {
    let checks = checks_report(snapshot);
    DoctorJsonReport {
        schema_version: INSPECTION_SCHEMA_VERSION,
        mode: "doctor",
        status: overall_check_status(&checks),
        version: env!("CARGO_PKG_VERSION"),
        compiled_backend: snapshot.info.compiled_backend.name().to_string(),
        supports_inference: snapshot.info.compiled_backend.supports_inference(),
        tty: snapshot.tty,
        cpu: CpuReport {
            name: snapshot.info.cpu_name.clone(),
            cores: snapshot.info.cpu_cores,
        },
        ram_gb: RamReport {
            total: snapshot.info.total_ram_gb,
            available: snapshot.info.available_ram_gb,
            effective: snapshot.info.effective_memory_gb(),
        },
        gpu: gpu_report(&snapshot.info.gpu),
        inputs: InputsReport {
            env_model: snapshot.env_model.clone(),
            model_path_arg: snapshot.arg_model_path.clone(),
            hf_cache_root: snapshot
                .hf_cache_root
                .as_ref()
                .map(|path| path.display().to_string()),
        },
        resolution: resolution_report(&snapshot.selected),
        discovery: discovery_report(snapshot),
        recommendations: recommendation_reports(snapshot),
        tools: tools::tool_runtime_report(),
        checks,
    }
}

fn models_report(snapshot: &DoctorSnapshot) -> ModelsJsonReport {
    let resolution = resolution_report(&snapshot.selected);
    ModelsJsonReport {
        schema_version: INSPECTION_SCHEMA_VERSION,
        mode: "list_models",
        status: resolution.status,
        version: env!("CARGO_PKG_VERSION"),
        compiled_backend: snapshot.info.compiled_backend.name().to_string(),
        resolution,
        discovery: discovery_report(snapshot),
        recommendations: recommendation_reports(snapshot),
    }
}

fn gpu_report(gpu: &GpuInfo) -> GpuReport {
    match gpu {
        GpuInfo::Cuda { name, vram_gb } => GpuReport {
            kind: "cuda",
            name: Some(name.clone()),
            memory_gb: Some(*vram_gb),
        },
        GpuInfo::Metal {
            chip,
            unified_memory_gb,
        } => GpuReport {
            kind: "metal",
            name: Some(chip.clone()),
            memory_gb: Some(*unified_memory_gb),
        },
        GpuInfo::None => GpuReport {
            kind: "none",
            name: None,
            memory_gb: None,
        },
    }
}

fn resolution_report(selected: &Result<SelectedModelSource>) -> ResolutionReport {
    match selected {
        Ok(source) => ResolutionReport {
            status: "ok",
            selected: Some(SelectedModelReport {
                origin: source.origin_key(),
                value: source.value().to_string(),
                local_path: source.local_path().map(|path| path.display().to_string()),
            }),
            error_code: None,
            error: None,
        },
        Err(err) => ResolutionReport {
            status: "warn",
            selected: None,
            error_code: Some(resolution_error_code(err)),
            error: Some(format!("{err:#}")),
        },
    }
}

fn resolution_error_code(err: &anyhow::Error) -> &'static str {
    if err.to_string().starts_with("no model selected;") {
        "no_model_selected"
    } else {
        "resolution_failed"
    }
}

fn discovery_report(snapshot: &DoctorSnapshot) -> DiscoveryReport {
    DiscoveryReport {
        preferred_local_model: snapshot.discovered.as_ref().map(|(model_id, path)| {
            DiscoveredModelReport {
                model_id: model_id.clone(),
                path: path.display().to_string(),
            }
        }),
        supported_hub_snapshots: snapshot
            .snapshots
            .iter()
            .map(|item| HubSnapshotReport {
                model_id: item.model_id.clone(),
                path: item.path.display().to_string(),
            })
            .collect(),
    }
}

fn recommendation_reports(snapshot: &DoctorSnapshot) -> Vec<ModelRecommendationReport> {
    snapshot
        .recommendations
        .iter()
        .map(|entry| ModelRecommendationReport {
            hf_id: entry.hf_id.to_string(),
            display_name: entry.display_name.to_string(),
            quantization: entry.quantization.map(ToOwned::to_owned),
            size_gb: entry.size_gb,
            min_memory_gb: entry.min_memory_gb,
        })
        .collect()
}

fn checks_report(snapshot: &DoctorSnapshot) -> Vec<CheckReport> {
    let mut checks = Vec::new();
    let supports_inference = snapshot.info.compiled_backend.supports_inference();
    if supports_inference {
        checks.push(CheckReport {
            code: "backend_available",
            name: "backend",
            status: "ok",
            message: format!(
                "{} backend is available in this binary",
                snapshot.info.compiled_backend.name()
            ),
        });
    } else {
        checks.push(CheckReport {
            code: "backend_missing",
            name: "backend",
            status: "warn",
            message: rebuild_backend_message(&snapshot.info.gpu),
        });
    }
    if !supports_inference {
        checks.push(CheckReport {
            code: "catalog_backendless",
            name: "catalog",
            status: "warn",
            message:
                "backend-specific recommendations require a build with `cuda`, `metal,no-cuda`, or `cpu,no-cuda`"
                    .to_string(),
        });
    } else if snapshot.recommendations.is_empty() {
        checks.push(CheckReport {
            code: "catalog_no_fit",
            name: "catalog",
            status: "warn",
            message: "detected memory does not fit any curated model for this backend".to_string(),
        });
    } else {
        checks.push(CheckReport {
            code: "catalog_available",
            name: "catalog",
            status: "ok",
            message: format!(
                "{} curated recommendations available",
                snapshot.recommendations.len()
            ),
        });
    }
    if snapshot.selected.is_ok() {
        checks.push(CheckReport {
            code: "resolution_resolved",
            name: "resolution",
            status: "ok",
            message: "resolved a model source".to_string(),
        });
    } else {
        checks.push(CheckReport {
            code: "resolution_missing",
            name: "resolution",
            status: "warn",
            message:
                "set `--model-path`, `ARLE_MODEL` (legacy `AGENT_INFER_MODEL` also works), or cache a supported local model"
                    .to_string(),
        });
    }
    let tools = tools::tool_runtime_report();
    if tools.sandboxed {
        checks.push(CheckReport {
            code: "tool_sandbox_available",
            name: "tool_sandbox",
            status: "ok",
            message: format!("built-in tools will run through {}", tools.sandbox_backend),
        });
    } else {
        checks.push(CheckReport {
            code: "tool_sandbox_missing",
            name: "tool_sandbox",
            status: "warn",
            message:
                "built-in shell/python tools are enabled by default and no supported sandbox backend was detected"
                    .to_string(),
        });
    }
    checks
}

fn overall_check_status(checks: &[CheckReport]) -> &'static str {
    if checks.iter().any(|check| check.status != "ok") {
        "warn"
    } else {
        "ok"
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
    if !snapshot.info.compiled_backend.supports_inference() {
        println!(
            "{} {}",
            style("catalog").dim(),
            rebuild_catalog_message(&snapshot.info.gpu)
        );
    } else if snapshot.recommendations.is_empty() {
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

fn print_tools_section() {
    let report = tools::tool_runtime_report();
    println!("{}", style("Tools").bold());
    println!(
        "{} {}",
        style("enabled by default").dim(),
        report.enabled_by_default
    );
    println!(
        "{} {}",
        style("built-ins").dim(),
        report.builtin_tools.join(", ")
    );
    println!(
        "{} {}{}",
        style("sandbox").dim(),
        report.sandbox_backend,
        if report.sandboxed {
            ""
        } else {
            " (not isolated)"
        }
    );
    println!(
        "{} {}s · {} MiB",
        style("limits").dim(),
        report.timeout_secs,
        report.max_memory_mb
    );
    println!("{} {}", style("python").dim(), report.python);
    if !report.sandboxed {
        println!(
            "{} run with `--no-tools` for prompts that must not execute shell/python code",
            style("safety").yellow().bold()
        );
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
            Self::Environment(_) => "ARLE_MODEL / AGENT_INFER_MODEL",
            Self::AutoDiscovered { .. } => "auto-discovered local model",
        }
    }

    fn origin_key(&self) -> &'static str {
        match self {
            Self::Explicit(_) => "arg_model_path",
            Self::Environment(_) => "env_model",
            Self::AutoDiscovered { .. } => "auto_discovered_local_model",
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
        "no model selected; pass --model-path, set ARLE_MODEL (legacy AGENT_INFER_MODEL also works), or place a supported model in the local HuggingFace cache"
    )
}

fn non_empty_value(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

fn preferred_model_env_value() -> Option<String> {
    non_empty_value(std::env::var(PRIMARY_MODEL_ENV).ok().as_deref())
        .map(ToOwned::to_owned)
        .or_else(|| {
            non_empty_value(std::env::var(LEGACY_MODEL_ENV).ok().as_deref()).map(ToOwned::to_owned)
        })
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

fn rebuild_backend_message(gpu: &GpuInfo) -> String {
    match gpu {
        GpuInfo::Cuda { .. } => {
            "host NVIDIA GPU detected; rebuild with the default `cuda` feature to run inference"
                .to_string()
        }
        GpuInfo::Metal { .. } => {
            "host Apple Silicon GPU detected; rebuild with `--no-default-features --features metal,no-cuda` to run inference"
                .to_string()
        }
        GpuInfo::None => {
            "no local GPU backend compiled; rebuild with `--no-default-features --features cpu,no-cuda` for CPU inference, or with `cuda` / `metal,no-cuda` on a supported host"
                .to_string()
        }
    }
}

fn rebuild_catalog_message(gpu: &GpuInfo) -> String {
    match gpu {
        GpuInfo::Cuda { .. } => {
            "backend-specific recommendations require a CUDA build on this host".to_string()
        }
        GpuInfo::Metal { .. } => {
            "backend-specific recommendations require a Metal build on this host".to_string()
        }
        GpuInfo::None => {
            "backend-specific recommendations require a build with `cuda`, `metal,no-cuda`, or `cpu,no-cuda`"
                .to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DoctorSnapshot, SelectedModelSource, TtyState, doctor_report, models_report,
        select_model_source,
    };
    use crate::hardware::{CompiledBackend, GpuInfo, SystemInfo};
    use crate::{hub_discovery, model_catalog};
    use std::path::PathBuf;

    fn test_snapshot(selected: anyhow::Result<SelectedModelSource>) -> DoctorSnapshot {
        let snapshots = (0..6)
            .map(|index| hub_discovery::HubSnapshot {
                model_id: format!("Qwen/Qwen3-{}B", index + 1),
                path: PathBuf::from(format!("/tmp/hub/{index}")),
            })
            .collect();
        DoctorSnapshot {
            info: SystemInfo {
                cpu_name: "test-cpu".to_string(),
                cpu_cores: 8,
                total_ram_gb: 32.0,
                available_ram_gb: 24.0,
                gpu: GpuInfo::None,
                compiled_backend: CompiledBackend::Cpu,
            },
            tty: TtyState {
                stdin: false,
                stdout: true,
                stderr: true,
            },
            env_model: Some("Qwen/Qwen3-4B".to_string()),
            arg_model_path: None,
            hf_cache_root: Some(PathBuf::from("/tmp/hf-cache")),
            discovered: Some(("Qwen/Qwen3-0.6B".to_string(), PathBuf::from("/tmp/auto"))),
            snapshots,
            recommendations: vec![&model_catalog::CATALOG[2], &model_catalog::CATALOG[3]],
            selected,
        }
    }

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

    #[test]
    fn doctor_report_serializes_machine_readable_fields() {
        let snapshot = test_snapshot(Ok(SelectedModelSource::Environment(
            "Qwen/Qwen3-4B".to_string(),
        )));
        let value = serde_json::to_value(doctor_report(&snapshot)).expect("serialize doctor json");

        assert_eq!(value["schema_version"], 3);
        assert_eq!(value["mode"], "doctor");
        assert_eq!(
            value["status"],
            if snapshot.info.compiled_backend.supports_inference() {
                "ok"
            } else {
                "warn"
            }
        );
        assert_eq!(
            value["compiled_backend"],
            snapshot.info.compiled_backend.name()
        );
        assert_eq!(value["resolution"]["status"], "ok");
        assert_eq!(value["resolution"]["selected"]["origin"], "env_model");
        assert_eq!(value["inputs"]["hf_cache_root"], "/tmp/hf-cache");
        assert!(value["tools"].is_object());
        assert!(value["tools"]["builtin_tools"].is_array());
        assert_eq!(
            value["discovery"]["supported_hub_snapshots"]
                .as_array()
                .expect("snapshot list")
                .len(),
            6
        );
        assert_eq!(
            value["checks"][0]["code"],
            if snapshot.info.compiled_backend.supports_inference() {
                "backend_available"
            } else {
                "backend_missing"
            }
        );
        assert_eq!(value["checks"][0]["name"], "backend");
        assert_eq!(
            value["checks"][0]["status"],
            if snapshot.info.compiled_backend.supports_inference() {
                "ok"
            } else {
                "warn"
            }
        );
    }

    #[test]
    fn models_report_serializes_without_doctor_only_checks() {
        let snapshot = test_snapshot(Err(anyhow::anyhow!("no model selected")));
        let value = serde_json::to_value(models_report(&snapshot)).expect("serialize models json");

        assert_eq!(value["schema_version"], 3);
        assert_eq!(value["mode"], "list_models");
        assert_eq!(value["status"], "warn");
        assert!(value.get("checks").is_none());
        assert_eq!(value["resolution"]["status"], "warn");
        assert!(value["resolution"]["selected"].is_null());
        assert_eq!(value["resolution"]["error_code"], "resolution_failed");
        assert_eq!(value["resolution"]["error"], "no model selected");
    }
}
