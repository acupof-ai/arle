//! Startup banner — minimal hacker aesthetic.
//!
//! Dim labels, bold values, subtle color accents. No box-drawing.

use console::style;

use crate::hardware::{GpuInfo, SystemInfo};

/// Print the startup banner with detected system info.
pub(crate) fn print_startup_banner(info: &SystemInfo) {
    let version = env!("CARGO_PKG_VERSION");

    eprintln!();
    eprintln!(
        "  {}",
        style(format!("agent-infer v{version}")).bold().cyan()
    );
    eprintln!();

    // CPU
    eprintln!(
        "  {}  {} {}",
        style("cpu").dim(),
        style(&info.cpu_name).bold(),
        style(format!("· {} cores", info.cpu_cores)).dim()
    );

    // RAM
    eprintln!(
        "  {}  {} {}",
        style("ram").dim(),
        style(format!("{:.1} GB", info.total_ram_gb)).bold(),
        style(format!("({:.1} GB free)", info.available_ram_gb)).dim()
    );

    // GPU
    match &info.gpu {
        GpuInfo::Cuda { name, vram_gb } => {
            eprintln!(
                "  {}  {} {}",
                style("gpu").dim(),
                style(name).bold().green(),
                style(format!("· {vram_gb:.1} GB VRAM")).dim()
            );
        }
        GpuInfo::Metal {
            chip,
            unified_memory_gb,
        } => {
            eprintln!(
                "  {}  {} {}",
                style("gpu").dim(),
                style(chip).bold().green(),
                style(format!("· {unified_memory_gb:.0} GB unified")).dim()
            );
        }
        GpuInfo::None => {
            eprintln!("  {}  {}", style("gpu").dim(), style("none detected").dim());
        }
    }

    // Backend
    eprintln!(
        "  {}  {}",
        style("mode").dim(),
        style(info.compiled_backend.name()).bold().cyan()
    );
    eprintln!();
}

/// Print a compact model-loaded confirmation line.
pub(crate) fn print_model_loaded(model_id: &str, backend: &str, load_secs: f64) {
    eprintln!(
        "  {} {} {} {}",
        style("loaded").green().bold(),
        style(model_id).bold(),
        style(format!("({backend})")).dim(),
        style(format!("in {load_secs:.1}s")).dim()
    );
    eprintln!();
}
