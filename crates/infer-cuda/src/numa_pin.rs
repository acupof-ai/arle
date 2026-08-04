//! Pin this TP worker's threads to the NUMA node of its own GPU.
//!
//! Evidence (2026-06-12 trace, 8×H20 B=1 decode): per-rank compute is dead
//! even, but two ranks arrive last at every collective and the other six
//! spin inside their NCCL kernels waiting — ~129 latency-bound 8 KB
//! collectives per token amplify host launch jitter into multiple ms of
//! wall. Unpinned workers wander across sockets; pinning each rank to a
//! disjoint core slice of its GPU's NUMA node removes the scheduler-placement
//! lottery (the documented ±6% session drift is the same lottery).
//!
//! Default ON for multi-rank CUDA workers; `--numa-pin false` opts out. Every
//! decision is logged loudly; ANY failure leaves the process unpinned and
//! boots normally (pinning is never load-bearing for correctness).

#[cfg(target_os = "linux")]
#[cfg_attr(not(feature = "nccl"), allow(dead_code))]
pub(crate) fn pin_to_gpu_numa(ordinal: usize, world_size: usize) {
    if !crate::runtime_flags::numa_pin() {
        log::info!("[numa-pin] disabled via --numa-pin false");
        return;
    }
    match try_pin(ordinal, world_size) {
        Ok(msg) => log::info!("[numa-pin] {msg}"),
        Err(err) => log::warn!("[numa-pin] unpinned (non-fatal): {err:#}"),
    }
}

#[cfg(not(target_os = "linux"))]
#[cfg_attr(not(feature = "nccl"), allow(dead_code))]
pub(crate) fn pin_to_gpu_numa(_ordinal: usize, _world_size: usize) {}

#[cfg(target_os = "linux")]
fn try_pin(ordinal: usize, world_size: usize) -> anyhow::Result<String> {
    use anyhow::{Context, anyhow, ensure};

    // GPU index → PCI bus id → NUMA node, for ALL GPUs (slice assignment
    // needs to know how many ranks share this node). nvidia-smi is a boot
    // one-shot; no per-step cost.
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=index,pci.bus_id", "--format=csv,noheader"])
        .output()
        .context("nvidia-smi spawn failed")?;
    ensure!(out.status.success(), "nvidia-smi exited non-zero");
    let text = String::from_utf8_lossy(&out.stdout);
    let mut nodes: Vec<(usize, i32)> = Vec::new(); // (gpu index, numa node)
    for line in text.lines() {
        let mut parts = line.split(',').map(str::trim);
        let (Some(idx), Some(bus)) = (parts.next(), parts.next()) else {
            continue;
        };
        let idx: usize = idx.parse().context("gpu index parse")?;
        // nvidia-smi prints e.g. 00000000:8A:00.0; sysfs wants lowercase with
        // a 4-hex-digit domain.
        let bus = bus.to_lowercase();
        let bus_tail = bus.split_once(':').map(|(_, t)| t).unwrap_or(&bus);
        let sys = format!("/sys/bus/pci/devices/0000:{bus_tail}/numa_node");
        let node: i32 = std::fs::read_to_string(&sys)
            .with_context(|| format!("read {sys}"))?
            .trim()
            .parse()
            .context("numa_node parse")?;
        nodes.push((idx, node));
    }
    ensure!(!nodes.is_empty(), "no GPUs parsed from nvidia-smi");
    let my_node = nodes
        .iter()
        .find(|(i, _)| *i == ordinal)
        .map(|(_, n)| *n)
        .ok_or_else(|| anyhow!("GPU ordinal {ordinal} not in nvidia-smi list"))?;
    // numa_node = -1 (single-node / unknown): pin to a disjoint slice of the
    // whole online cpu set instead — the win is disjointness + stickiness.
    let cpulist_path = if my_node >= 0 {
        format!("/sys/devices/system/node/node{my_node}/cpulist")
    } else {
        "/sys/devices/system/cpu/online".to_string()
    };
    let cpus = parse_cpulist(
        std::fs::read_to_string(&cpulist_path)
            .with_context(|| format!("read {cpulist_path}"))?
            .trim(),
    )?;
    ensure!(!cpus.is_empty(), "empty cpulist at {cpulist_path}");

    // This rank's slice index among ranks that share the node (GPU-ordinal
    // order); world_size bounds the divisor when nvidia-smi sees more GPUs
    // than the TP group uses.
    let sharers: Vec<usize> = nodes
        .iter()
        .filter(|(i, n)| (*n == my_node || my_node < 0) && *i < world_size)
        .map(|(i, _)| *i)
        .collect();
    let slot = sharers.iter().position(|&i| i == ordinal).unwrap_or(0);
    let nshare = sharers.len().max(1);
    let per = (cpus.len() / nshare).max(1);
    let slice = &cpus[slot * per..((slot + 1) * per).min(cpus.len())];
    ensure!(!slice.is_empty(), "empty core slice");

    // SAFETY: plain libc cpu_set_t manipulation. Pinning the calling thread
    // alone is NOT enough: `sched_setaffinity(0)` only sets the *current*
    // thread, and threads spawned BEFORE this call (the coordinator/rank-0
    // process already runs the tokio HTTP + router pool by the time it builds
    // the engine) do not inherit it — they would keep wandering the whole
    // machine. So build the mask once and apply it to EVERY thread of this
    // process (existing + the calling thread); future threads inherit from the
    // calling thread. This makes the pin automatic and complete for all ranks,
    // including rank 0 in the multi-threaded coordinator.
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    for &c in slice {
        // SAFETY: `set` is a live zeroed cpu_set_t and `c` is a CPU index
        // from the topology query, below CPU_SETSIZE.
        unsafe { libc::CPU_SET(c, &mut set) };
    }
    // Calling thread first (the inheritance anchor for threads spawned later).
    ensure!(
        // SAFETY: `set` is fully initialized above and its size is passed
        // exactly; pid 0 is the calling thread.
        unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) } == 0,
        "sched_setaffinity(self) failed: {}",
        std::io::Error::last_os_error()
    );
    let (pinned, total) = pin_all_threads(&set);
    Ok(format!(
        "rank gpu{ordinal} → numa{my_node}, cores {}..{} ({} of {} on node, {} sharers), {pinned}/{total} threads pinned",
        slice[0],
        slice[slice.len() - 1],
        slice.len(),
        cpus.len(),
        nshare
    ))
}

/// Apply `set` to every thread of this process. Walks `/proc/self/task`; a
/// per-thread failure (a thread exiting mid-walk, or a TID-reuse race) is
/// non-fatal — it just leaves that thread on the previous mask. Returns
/// `(pinned, total_seen)` for the boot log. New threads spawned after this
/// inherit from the already-pinned calling thread, so the walk only has to
/// catch the pre-existing pool.
#[cfg(target_os = "linux")]
fn pin_all_threads(set: &libc::cpu_set_t) -> (usize, usize) {
    let Ok(entries) = std::fs::read_dir("/proc/self/task") else {
        return (0, 0);
    };
    let (mut pinned, mut total) = (0usize, 0usize);
    for entry in entries.flatten() {
        let Ok(tid) = entry.file_name().to_string_lossy().parse::<libc::pid_t>() else {
            continue;
        };
        total += 1;
        // SAFETY: cpu_set_t by const ptr; tid is a thread of this process.
        let rc =
            unsafe { libc::sched_setaffinity(tid, std::mem::size_of::<libc::cpu_set_t>(), set) };
        if rc == 0 {
            pinned += 1;
        }
    }
    (pinned, total)
}

/// Parse a sysfs cpulist like `0-23,96-119` into core ids.
#[cfg(target_os = "linux")]
fn parse_cpulist(s: &str) -> anyhow::Result<Vec<usize>> {
    use anyhow::Context;
    let mut out = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((a, b)) = part.split_once('-') {
            let a: usize = a.trim().parse().context("cpulist range start")?;
            let b: usize = b.trim().parse().context("cpulist range end")?;
            out.extend(a..=b);
        } else {
            out.push(part.parse().context("cpulist entry")?);
        }
    }
    Ok(out)
}
