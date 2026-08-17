//! Background host sampling + JSONL storage. Zero hot-path cost: the engine
//! thread only writes the snapshot it already publishes per tick.

use crate::execution::CounterSnapshot;
use serde::{Deserialize, Serialize};
use std::os::unix::io::AsRawFd;
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HostSample {
    pub cpu_pct: f32,
    pub ram_used_mb: u64,
    pub ram_total_mb: u64,
    pub disk_used_pct: f32,
}

static HOST: RwLock<HostSample> = RwLock::new(HostSample {
    cpu_pct: 0.0,
    ram_used_mb: 0,
    ram_total_mb: 0,
    disk_used_pct: 0.0,
});
static HOST_INIT: OnceLock<()> = OnceLock::new();

pub fn host_sample() -> HostSample {
    HOST_INIT.get_or_init(spawn_host_sampler);
    HOST.read().map(|h| h.clone()).unwrap_or_default()
}

fn spawn_host_sampler() {
    std::thread::spawn(|| {
        let mut sys = sysinfo::System::new();
        loop {
            sys.refresh_cpu_all();
            sys.refresh_memory();
            let cpu_pct = if sys.cpus().is_empty() {
                0.0
            } else {
                sys.cpus().iter().map(|c| c.cpu_usage()).sum::<f32>() / sys.cpus().len() as f32
            };
            let ram_total_mb = sys.total_memory() / (1024 * 1024);
            let ram_used_mb = ram_total_mb.saturating_sub(sys.available_memory() / (1024 * 1024));
            let sample = HostSample {
                cpu_pct,
                ram_used_mb,
                ram_total_mb,
                disk_used_pct: disk_used_pct(),
            };
            if let Ok(mut h) = HOST.write() {
                *h = sample;
            }
            std::thread::sleep(Duration::from_secs(10));
        }
    });
}

fn disk_used_pct() -> f32 {
    let disks = sysinfo::Disks::new_with_refreshed_list();
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
    pub cached_pages: u32,
    pub cpu_pct: f32,
    pub ram_used_mb: u64,
    pub ram_total_mb: u64,
    pub disk_used_pct: f32,
    pub gpu: Option<infer_seam::GpuSample>,
    pub steps: u64,
    pub prefill_tokens: u64,
    pub generated_tokens: u64,
    pub requests_completed: u64,
    pub requests_succeeded: u64,
    pub requests_failed: u64,
    pub ttft_micros_total: u64,
    pub ttft_count: u64,
    pub tpot_micros_total: u64,
    pub tpot_count: u64,
    pub forward_busy_micros: u64,
    pub prefill_forward_steps: u64,
    pub prefill_forward_busy_micros: u64,
    pub decode_forward_steps: u64,
    pub decode_forward_busy_micros: u64,
    pub mixed_forward_steps: u64,
    pub mixed_forward_busy_micros: u64,
    pub prefix_lookups: u64,
    pub prefix_hits: u64,
    pub prefix_hit_tokens: u64,
    pub spec_drafted: u64,
    pub spec_accepted: u64,
    pub kv_demoted_pages: u64,
    pub kv_promoted_pages: u64,
    pub kv_disk_pages: u64,
    pub kv_reuse_hit_disk: u64,
}

impl StoredSample {
    fn from_snapshot(snap: &CounterSnapshot, host: HostSample, ts_ms: u64) -> Self {
        let t = &snap.throughput;
        let p = &snap.prefix_cache;
        let k = &snap.kv_tier;
        let ks = &snap.kv_system;
        Self {
            ts_ms,
            active_requests: snap.active_requests as u32,
            queue_depth: snap.queue_depth as u32,
            kv_free_pages: snap.kv_free_pages as u32,
            cached_pages: p.cached_pages as u32,
            cpu_pct: host.cpu_pct,
            ram_used_mb: host.ram_used_mb,
            ram_total_mb: host.ram_total_mb,
            disk_used_pct: host.disk_used_pct,
            gpu: snap.gpu,
            steps: t.steps,
            prefill_tokens: t.prefill_tokens,
            generated_tokens: t.generated_tokens,
            requests_completed: t.requests_completed,
            requests_succeeded: t.requests_succeeded,
            requests_failed: t.requests_failed,
            ttft_micros_total: t.ttft_micros_total,
            ttft_count: t.ttft_count,
            tpot_micros_total: t.tpot_micros_total,
            tpot_count: t.tpot_count,
            forward_busy_micros: t.forward_busy_micros,
            prefill_forward_steps: t.prefill_forward_steps,
            prefill_forward_busy_micros: t.prefill_forward_busy_micros,
            decode_forward_steps: t.decode_forward_steps,
            decode_forward_busy_micros: t.decode_forward_busy_micros,
            mixed_forward_steps: t.mixed_forward_steps,
            mixed_forward_busy_micros: t.mixed_forward_busy_micros,
            prefix_lookups: p.lookups,
            prefix_hits: p.hits,
            prefix_hit_tokens: p.hit_tokens,
            spec_drafted: snap.spec_decode.drafted,
            spec_accepted: snap.spec_decode.accepted,
            kv_demoted_pages: k.demoted_pages,
            kv_promoted_pages: k.promoted_pages,
            kv_disk_pages: ks.disk_pages as u64,
            kv_reuse_hit_disk: ks.reuse_hit_disk,
        }
    }
}

fn observe_dir() -> std::path::PathBuf {
    std::env::var("ARLE_OBSERVE_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("observe-data"))
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
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
        .truncate(false)
        .open(path)?;
    writeln!(file, "{line}")?;
    file.flush()
}

fn sweep_retention(dir: &std::path::Path, retention_days: u64) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let cutoff_ms = now_ms().saturating_sub(retention_days * 86_400_000);
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.starts_with("observe-") || !name.ends_with(".jsonl") {
            continue;
        }
        let day = &name["observe-".len()..name.len() - ".jsonl".len()];
        let ts = parse_day_to_ms(day).unwrap_or(0);
        if ts < cutoff_ms {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

fn parse_day_to_ms(day: &str) -> Option<u64> {
    let parts: Vec<&str> = day.split('-').collect();
    if parts.len() != 3 {
        return None;
    }
    let y: i64 = parts[0].parse().ok()?;
    let m: u32 = parts[1].parse().ok()?;
    let d: u32 = parts[2].parse().ok()?;
    let days = days_from_ymd(y, m, d)?;
    Some(days as u64 * 86_400_000)
}

fn days_from_ymd(y: i64, m: u32, d: u32) -> Option<i64> {
    let y = y - i64::from(m <= 2);
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let m_adj: i64 = if m > 2 {
        i64::from(m) - 3
    } else {
        i64::from(m) + 9
    };
    let doy = (153 * m_adj + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some(era * 146_097 + doe - 719_468)
}

pub fn spawn_observe_task(counters: Arc<Mutex<CounterSnapshot>>) {
    std::thread::spawn(move || {
        let dir = observe_dir();
        let _lock = match acquire_writer_lock(&dir) {
            Some(f) => f,
            None => return,
        };
        let retention_days: u64 = std::env::var("ARLE_OBSERVE_RETENTION_DAYS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(30);
        sweep_retention(&dir, retention_days);
        loop {
            std::thread::sleep(Duration::from_secs(10));
            let snap = match counters.lock() {
                Ok(s) => s.clone(),
                Err(_) => continue,
            };
            let host = host_sample();
            let sample = StoredSample::from_snapshot(&snap, host, now_ms());
            if let Err(e) = append_sample(&dir, &sample) {
                log::warn!("observe: append failed: {e}");
            }
        }
    });
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
    let now = now_ms();
    let start = now.saturating_sub(range_ms);
    let start_day = day_string(start);
    let end_day = day_string(now);
    let mut samples = Vec::new();
    let mut day = start_day.clone();
    loop {
        let path = dir.join(format!("observe-{day}.jsonl"));
        if let Ok(content) = std::fs::read_to_string(&path) {
            for line in content.lines() {
                if let Ok(s) = serde_json::from_str::<StoredSample>(line)
                    && s.ts_ms >= start
                    && s.ts_ms <= now
                {
                    samples.push(s);
                }
            }
        }
        if day == end_day {
            break;
        }
        let ts = parse_day_to_ms(&day).unwrap_or(0);
        day = day_string(ts + 86_400_000);
    }
    samples
}

pub fn latest() -> Option<StoredSample> {
    let dir = observe_dir();
    let path = dir.join(format!("observe-{}.jsonl", day_string(now_ms())));
    let content = std::fs::read_to_string(&path).ok()?;
    content
        .lines()
        .rev()
        .find_map(|l| serde_json::from_str(l).ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ymd_roundtrip() {
        let days = days_from_ymd(2026, 8, 17).unwrap();
        let (y, m, d) = days_to_ymd(days);
        assert_eq!((y, m, d), (2026, 8, 17));
    }

    #[test]
    fn ymd_epoch() {
        let (y, m, d) = days_to_ymd(0);
        assert_eq!((y, m, d), (1970, 1, 1));
    }

    #[test]
    fn day_string_format() {
        let ts = days_from_ymd(2026, 8, 17).unwrap() as u64 * 86_400_000;
        assert_eq!(day_string(ts), "2026-08-17");
    }

    #[test]
    fn stored_sample_json_roundtrip() {
        let sample = StoredSample {
            ts_ms: 12345,
            active_requests: 3,
            queue_depth: 1,
            kv_free_pages: 100,
            cached_pages: 50,
            cpu_pct: 45.5,
            ram_used_mb: 32000,
            ram_total_mb: 64000,
            disk_used_pct: 60.0,
            gpu: None,
            steps: 1000,
            prefill_tokens: 50000,
            generated_tokens: 100000,
            requests_completed: 200,
            requests_succeeded: 195,
            requests_failed: 5,
            ttft_micros_total: 30_000_000,
            ttft_count: 200,
            tpot_micros_total: 7_000_000,
            tpot_count: 100_000,
            forward_busy_micros: 35_000_000,
            prefill_forward_steps: 200,
            prefill_forward_busy_micros: 20_000_000,
            decode_forward_steps: 1000,
            decode_forward_busy_micros: 15_000_000,
            mixed_forward_steps: 0,
            mixed_forward_busy_micros: 0,
            prefix_lookups: 500,
            prefix_hits: 400,
            prefix_hit_tokens: 40000,
            spec_drafted: 0,
            spec_accepted: 0,
            kv_demoted_pages: 10,
            kv_promoted_pages: 5,
            kv_disk_pages: 10,
            kv_reuse_hit_disk: 3,
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
        let sample = StoredSample {
            ts_ms: days_from_ymd(2026, 8, 17).unwrap() as u64 * 86_400_000 + 3600_000,
            ..Default::default()
        };
        append_sample(&dir, &sample).unwrap();
        let path = dir.join("observe-2026-08-17.jsonl");
        let content = std::fs::read_to_string(&path).unwrap();
        let parsed: StoredSample = serde_json::from_str(content.trim()).unwrap();
        assert_eq!(parsed.ts_ms, sample.ts_ms);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
