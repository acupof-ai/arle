//! Background GPU sampler: polls `nvidia-smi` every 2 s on rank 0, stores
//! the latest sample in a process-wide lock. Zero hot-path cost — the engine
//! thread reads the lock only inside `stats()` (per-tick, not per-step).

use infer_seam::GpuSample;
use std::sync::OnceLock;
use std::sync::RwLock;
use std::time::Duration;

static STORE: RwLock<Option<GpuSample>> = RwLock::new(None);
static INIT: OnceLock<()> = OnceLock::new();

/// Returns the latest GPU sample, or `None` if the sampler is not running
/// (non-zero TP rank, nvidia-smi absent, or first poll not yet completed).
pub(crate) fn latest() -> Option<GpuSample> {
    // Disabled: nvidia-smi every 2s stalls the GPU driver on H20 (~2-5 ms/step
    // under DSpark batched decode). Re-enable with a non-stalling sampler (NVML).
    None
}

fn spawn() {
    let rank = crate::tp::resolve_tp_config_from_env()
        .map(|c| c.rank)
        .unwrap_or(0);
    if rank != 0 {
        return;
    }
    std::thread::spawn(|| {
        loop {
            std::thread::sleep(Duration::from_secs(2));
            let sample = match query_nvidia_smi() {
                Ok(s) => s,
                Err(_) => continue,
            };
            if let Ok(mut guard) = STORE.write() {
                *guard = Some(sample);
            }
        }
    });
}

/// Parses `nvidia-smi --query-gpu=... --format=csv,noheader,nounits` output.
/// One line per GPU: `index, util%, mem_used_mb, mem_total_mb, temp_c, power_w`.
fn parse_nvidia_smi(stdout: &str) -> GpuSample {
    let mut devices = [infer_seam::GpuDeviceSample::default(); 8];
    let mut count = 0u8;
    for line in stdout.lines() {
        if count >= 8 {
            break;
        }
        let parts: Vec<&str> = line.split(',').map(str::trim).collect();
        if parts.len() < 6 {
            continue;
        }
        devices[count as usize] = infer_seam::GpuDeviceSample {
            gpu_index: parts[0].parse().unwrap_or(0),
            util_pct: parts[1].parse().unwrap_or(0),
            memory_used_mb: parts[2].parse().unwrap_or(0),
            memory_total_mb: parts[3].parse().unwrap_or(0),
            temp_c: parts[4].parse().unwrap_or(0),
            power_w: parts[5].parse().unwrap_or(0),
        };
        count += 1;
    }
    GpuSample {
        devices,
        device_count: count,
    }
}

fn query_nvidia_smi() -> std::io::Result<GpuSample> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ])
        .output()?;
    if !output.status.success() {
        return Err(std::io::Error::other("nvidia-smi exited with error"));
    }
    Ok(parse_nvidia_smi(&String::from_utf8_lossy(&output.stdout)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_two_gpus() {
        let out = "0, 95, 12000, 24576, 45, 350\n1, 80, 10000, 24576, 42, 300\n";
        let s = parse_nvidia_smi(out);
        assert_eq!(s.device_count, 2);
        assert_eq!(s.devices[0].gpu_index, 0);
        assert_eq!(s.devices[0].util_pct, 95);
        assert_eq!(s.devices[0].memory_used_mb, 12000);
        assert_eq!(s.devices[0].memory_total_mb, 24576);
        assert_eq!(s.devices[0].temp_c, 45);
        assert_eq!(s.devices[0].power_w, 350);
        assert_eq!(s.devices[1].gpu_index, 1);
        assert_eq!(s.devices[1].util_pct, 80);
    }

    #[test]
    fn parse_empty() {
        let s = parse_nvidia_smi("");
        assert_eq!(s.device_count, 0);
    }

    #[test]
    fn parse_malformed_line_skipped() {
        let out = "garbage\n0, 50, 8000, 16384, 40, 200\n";
        let s = parse_nvidia_smi(out);
        assert_eq!(s.device_count, 1);
        assert_eq!(s.devices[0].util_pct, 50);
    }
}
