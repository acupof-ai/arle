//! Background host sampling + JSONL storage. Zero hot-path cost: the engine
//! thread only writes the snapshot it already publishes per tick.

use crate::execution::CounterSnapshot;
use serde::{Deserialize, Serialize};
use std::os::unix::io::AsRawFd;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HostSample {
    pub cpu_pct: f32,
    pub ram_used_mb: u64,
    pub disk_used_pct: f32,
}

fn disk_used_pct(disks: &sysinfo::Disks) -> f32 {
    for disk in disks.list() {
        if disk.mount_point() == std::path::Path::new("/") {
            let total = disk.total_space();
            if total == 0 {
                return 0.0;
            }
            let avail = disk.available_space();
            return ((total - avail) as f32 / total as f32) * 100.0;
        }
    }
    0.0
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StoredSample {
    pub ts_ms: u64,
    pub active_requests: u32,
    pub queue_depth: u32,
    pub kv_free_pages: u32,
    pub cpu_pct: f32,
    pub ram_used_mb: u64,
    pub disk_used_pct: f32,
    pub gpu: Option<infer_seam::GpuSample>,
    pub generated_tokens: u64,
    pub prefix_lookups: u64,
    pub prefix_hits: u64,
    pub ttft_micros_total: u64,
    pub ttft_count: u64,
    pub tpot_micros_total: u64,
    pub tpot_count: u64,
}

impl StoredSample {
    fn from_snapshot(snap: &CounterSnapshot, host: HostSample, ts_ms: u64) -> Self {
        Self {
            ts_ms,
            active_requests: snap.active_requests as u32,
            queue_depth: snap.queue_depth as u32,
            kv_free_pages: snap.kv_free_pages as u32,
            cpu_pct: host.cpu_pct,
            ram_used_mb: host.ram_used_mb,
            disk_used_pct: host.disk_used_pct,
            gpu: snap.gpu,
            generated_tokens: snap.throughput.generated_tokens,
            prefix_lookups: snap.prefix_cache.lookups,
            prefix_hits: snap.prefix_cache.hits,
            ttft_micros_total: snap.throughput.ttft_micros_total,
            ttft_count: snap.throughput.ttft_count,
            tpot_micros_total: snap.throughput.tpot_micros_total,
            tpot_count: snap.throughput.tpot_count,
        }
    }
}

fn observe_dir() -> std::path::PathBuf {
    std::env::var("ARLE_OBSERVE_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("observe-data"))
}

// Howard Hinnant's days-from-civil algorithm.
fn days_to_ymd(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (y + i64::from(m <= 2), m, d)
}

fn day_string(ts_ms: u64) -> String {
    let (y, m, d) = days_to_ymd((ts_ms / 86_400_000) as i64);
    format!("{y:04}-{m:02}-{d:02}")
}

fn append_sample(dir: &std::path::Path, sample: &StoredSample) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(format!("observe-{}.jsonl", day_string(sample.ts_ms)));
    let line = serde_json::to_string(sample)?;
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(file, "{line}")?;
    Ok(())
}
fn sweep_retention(dir: &std::path::Path, retention_days: u64) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let cutoff = day_string(infer_seam::now_ms().saturating_sub(retention_days * 86_400_000));
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.starts_with("observe-") || !name.ends_with(".jsonl") {
            continue;
        }
        let day = &name["observe-".len()..name.len() - ".jsonl".len()];
        if day < cutoff.as_str() {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

fn retention_days() -> u64 {
    std::env::var("ARLE_OBSERVE_RETENTION_DAYS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30)
}

fn sample_host(sys: &mut sysinfo::System, disks: &mut sysinfo::Disks, tick: u32) -> HostSample {
    sys.refresh_cpu_all();
    sys.refresh_memory();
    if tick.is_multiple_of(30) {
        disks.refresh(true);
    }
    let cpu_pct = if sys.cpus().is_empty() {
        0.0
    } else {
        sys.cpus().iter().map(|c| c.cpu_usage()).sum::<f32>() / sys.cpus().len() as f32
    };
    let ram_used_mb =
        (sys.total_memory() / (1024 * 1024)).saturating_sub(sys.available_memory() / (1024 * 1024));
    HostSample {
        cpu_pct,
        ram_used_mb,
        disk_used_pct: disk_used_pct(disks),
    }
}

fn write_sample(dir: &std::path::Path, snap: &CounterSnapshot, host: HostSample) {
    let sample = StoredSample::from_snapshot(snap, host, infer_seam::now_ms());
    if let Err(e) = append_sample(dir, &sample) {
        log::warn!("observe: append failed: {e}");
    }
}

/// Spawn the background observe task. `snapshot` returns the current counter
/// snapshot, or `None` to skip this tick (e.g. mutex poisoned). The flock
/// singleton ensures only one writer per machine.
pub fn spawn_observe_task<F>(mut snapshot: F)
where
    F: FnMut() -> Option<CounterSnapshot> + Send + 'static,
{
    std::thread::Builder::new()
        .name("arle-observe".to_string())
        .spawn(move || {
            let dir = observe_dir();
            let _lock = match acquire_writer_lock(&dir) {
                Some(f) => f,
                None => return,
            };
            sweep_retention(&dir, retention_days());
            let mut sys = sysinfo::System::new();
            let mut disks = sysinfo::Disks::new_with_refreshed_list();
            let mut tick: u32 = 0;
            loop {
                std::thread::sleep(Duration::from_secs(10));
                let host = sample_host(&mut sys, &mut disks, tick);
                let Some(snap) = snapshot() else {
                    tick += 1;
                    continue;
                };
                write_sample(&dir, &snap, host);
                tick += 1;
            }
        })
        .expect("spawn observe task");
}

// flock ensures only one process per machine writes; the kernel releases it on exit.
fn acquire_writer_lock(dir: &std::path::Path) -> Option<std::fs::File> {
    std::fs::create_dir_all(dir).ok()?;
    let path = dir.join("observe.lock");
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open(&path)
        .ok()?;
    // SAFETY: flock on a valid open fd; LOCK_NB makes it non-blocking.
    let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    (rc == 0).then_some(file)
}

pub fn query(range_ms: u64) -> Vec<StoredSample> {
    let dir = observe_dir();
    let now = infer_seam::now_ms();
    let start = now.saturating_sub(range_ms);
    let mut samples = Vec::new();
    for day in (start / 86_400_000)..=(now / 86_400_000) {
        let path = dir.join(format!("observe-{}.jsonl", day_string(day * 86_400_000)));
        if let Ok(content) = std::fs::read_to_string(&path) {
            for line in content.lines() {
                if let Ok(s) = serde_json::from_str::<StoredSample>(line)
                    && s.ts_ms >= start
                {
                    samples.push(s);
                }
            }
        }
    }
    samples
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ymd_epoch() {
        let (y, m, d) = days_to_ymd(0);
        assert_eq!((y, m, d), (1970, 1, 1));
    }

    #[test]
    fn day_string_known_date() {
        assert_eq!(day_string(1_786_924_800_000), "2026-08-17");
    }

    #[test]
    fn stored_sample_json_roundtrip() {
        let sample = StoredSample {
            ts_ms: 12345,
            active_requests: 3,
            queue_depth: 1,
            kv_free_pages: 100,
            cpu_pct: 45.5,
            ram_used_mb: 32000,
            disk_used_pct: 60.0,
            gpu: None,
            generated_tokens: 100000,
            prefix_lookups: 500,
            prefix_hits: 400,
            ttft_micros_total: 30_000_000,
            ttft_count: 200,
            tpot_micros_total: 7_000_000,
            tpot_count: 100_000,
        };
        let json = serde_json::to_string(&sample).unwrap();
        let parsed: StoredSample = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.ts_ms, sample.ts_ms);
        assert_eq!(parsed.generated_tokens, sample.generated_tokens);
        assert_eq!(parsed.cpu_pct, sample.cpu_pct);
    }

    #[test]
    fn append_and_read_back() {
        let dir = std::env::temp_dir().join(format!("arle-observe-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ts = 1_786_924_800_000 + 3_600_000; // 2026-08-17 01:00 UTC
        let sample = StoredSample {
            ts_ms: ts,
            ..Default::default()
        };
        append_sample(&dir, &sample).unwrap();
        let path = dir.join(format!("observe-{}.jsonl", day_string(ts)));
        let content = std::fs::read_to_string(&path).unwrap();
        let parsed: StoredSample = serde_json::from_str(content.trim()).unwrap();
        assert_eq!(parsed.ts_ms, sample.ts_ms);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
