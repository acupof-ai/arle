use console::style;

use crate::hardware::{GpuInfo, SystemInfo};

pub(crate) fn print_startup_banner(info: &SystemInfo) {
    let version = env!("CARGO_PKG_VERSION");

    eprintln!();
    eprintln!("  {}", style(format!("ARLE v{version}")).bold().cyan());
    eprintln!();

    eprintln!(
        "  {}  {} {}",
        style("cpu").dim(),
        style(&info.cpu_name).bold(),
        style(format!("· {} cores", info.cpu_cores)).dim()
    );

    eprintln!(
        "  {}  {} {}",
        style("ram").dim(),
        style(format!("{:.1} GB", info.total_ram_gb)).bold(),
        style(format!("({:.1} GB free)", info.available_ram_gb)).dim()
    );

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
            recommended_working_set_gb,
        } => {
            let memory = if let Some(working_set) = recommended_working_set_gb {
                format!("· {unified_memory_gb:.0} GB unified · {working_set:.1} GB working set")
            } else {
                format!("· {unified_memory_gb:.0} GB unified")
            };
            eprintln!(
                "  {}  {} {}",
                style("gpu").dim(),
                style(chip).bold().green(),
                style(memory).dim()
            );
        }
        GpuInfo::None => {
            eprintln!("  {}  {}", style("gpu").dim(), style("none detected").dim());
        }
    }

    eprintln!(
        "  {}  {}",
        style("backend").dim(),
        style(info.compiled_backend.name()).bold().cyan()
    );
    eprintln!();
}

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
