//! Runtime topology discovery and CPU-affinity helpers.
//!
//! Linux CUDA hosts get best-effort topology from sysfs/procfs. Non-Linux
//! builds keep the same API but return a single fallback CPU group so callers
//! can keep one code path.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
#[cfg(target_os = "linux")]
use std::path::PathBuf;
#[cfg(target_os = "linux")]
use std::process::Command;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NumaNodeTopology {
    pub node: i32,
    pub cpus: Vec<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuTopology {
    pub ordinal: usize,
    pub pci_bus_id: Option<String>,
    pub uuid: Option<String>,
    pub numa_node: Option<i32>,
    pub local_cpus: Vec<usize>,
    pub nearest_nics: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NicTopology {
    pub name: String,
    pub pci_bus_id: Option<String>,
    pub numa_node: Option<i32>,
    pub local_cpus: Vec<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NumaWorkerGroup {
    pub numa_node: Option<i32>,
    pub cpus: Vec<usize>,
    pub worker_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkerPlacement {
    pub worker_id: usize,
    pub gpu_ordinal: usize,
    pub numa_node: Option<i32>,
    pub cpus: Vec<usize>,
    pub nics: Vec<String>,
    pub route_cost: u32,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RuntimeTopology {
    pub numa_nodes: Vec<NumaNodeTopology>,
    pub gpus: Vec<GpuTopology>,
    pub nics: Vec<NicTopology>,
    pub fallback_cpus: Vec<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AffinityApplyResult {
    pub label: String,
    pub applied: bool,
    pub requested_cpus: Vec<usize>,
    pub applied_threads: usize,
    pub failed_threads: usize,
    pub reason: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct NumaMemoryStats {
    pub total_pages: u64,
    pub per_node_pages: Vec<(i32, u64)>,
}

impl RuntimeTopology {
    pub fn discover() -> Self {
        #[cfg(target_os = "linux")]
        {
            let topology = Self::discover_linux(Path::new("/sys"));
            if !topology.fallback_cpus.is_empty()
                || !topology.numa_nodes.is_empty()
                || !topology.gpus.is_empty()
                || !topology.nics.is_empty()
            {
                return topology;
            }
        }
        Self::fallback()
    }

    fn fallback() -> Self {
        let cpus: Vec<usize> = (0..std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1))
            .collect();
        let fallback_cpus = if cpus.is_empty() { vec![0] } else { cpus };
        Self {
            numa_nodes: Vec::new(),
            gpus: Vec::new(),
            nics: Vec::new(),
            fallback_cpus,
        }
    }

    #[cfg(target_os = "linux")]
    fn discover_linux(sysfs: &Path) -> Self {
        let fallback_cpus = read_cpu_list(sysfs.join("devices/system/cpu/online"))
            .filter(|cpus| !cpus.is_empty())
            .unwrap_or_else(|| Self::fallback().fallback_cpus);
        let mut numa_nodes = discover_numa_nodes(sysfs);
        numa_nodes.sort_by_key(|node| node.node);

        let mut nics = discover_nics(sysfs);
        nics.sort_by(|a, b| a.name.cmp(&b.name));
        let mut gpus = discover_gpus(sysfs, &nics);
        gpus.sort_by_key(|gpu| gpu.ordinal);

        Self {
            numa_nodes,
            gpus,
            nics,
            fallback_cpus,
        }
    }

    pub fn log_summary(&self) {
        log::info!(
            "Runtime topology: numa_nodes={} gpus={} nics={} fallback_cpus={}",
            self.numa_nodes.len(),
            self.gpus.len(),
            self.nics.len(),
            format_cpu_list(&self.fallback_cpus),
        );
        for node in &self.numa_nodes {
            log::info!(
                "Runtime topology NUMA node {}: cpus={}",
                node.node,
                format_cpu_list(&node.cpus),
            );
        }
        for gpu in &self.gpus {
            log::info!(
                "Runtime topology GPU {}: pci={} numa={:?} local_cpus={} nearest_nics={}",
                gpu.ordinal,
                gpu.pci_bus_id.as_deref().unwrap_or("unknown"),
                gpu.numa_node,
                format_cpu_list(&gpu.local_cpus),
                if gpu.nearest_nics.is_empty() {
                    "none".to_string()
                } else {
                    gpu.nearest_nics.join(",")
                },
            );
        }
        for nic in &self.nics {
            log::info!(
                "Runtime topology NIC {}: pci={} numa={:?} local_cpus={}",
                nic.name,
                nic.pci_bus_id.as_deref().unwrap_or("unknown"),
                nic.numa_node,
                format_cpu_list(&nic.local_cpus),
            );
        }
    }

    pub fn placement_for_gpu(&self, gpu_ordinal: usize) -> WorkerPlacement {
        self.placement_for_gpu_with_worker(gpu_ordinal, gpu_ordinal)
    }

    fn placement_for_gpu_with_worker(
        &self,
        gpu_ordinal: usize,
        worker_id: usize,
    ) -> WorkerPlacement {
        let gpu = self.gpus.iter().find(|gpu| gpu.ordinal == gpu_ordinal);
        let numa_node = gpu.and_then(|gpu| gpu.numa_node);
        let cpus = gpu
            .map(|gpu| gpu.local_cpus.clone())
            .filter(|cpus| !cpus.is_empty())
            .or_else(|| self.cpus_for_numa(numa_node))
            .unwrap_or_else(|| self.fallback_cpus.clone());
        let nics = gpu
            .map(|gpu| gpu.nearest_nics.clone())
            .filter(|nics| !nics.is_empty())
            .unwrap_or_else(|| self.nics_for_numa(numa_node));
        WorkerPlacement {
            worker_id,
            gpu_ordinal,
            numa_node,
            cpus,
            nics,
            route_cost: 0,
        }
    }

    pub fn placement_for_configured_cuda_device(&self) -> WorkerPlacement {
        let cuda_ordinal = configured_cuda_device_ordinal();
        let gpu = self.resolve_cuda_visible_gpu(
            cuda_ordinal,
            std::env::var("CUDA_VISIBLE_DEVICES").ok().as_deref(),
            std::env::var("CUDA_DEVICE_ORDER").ok().as_deref(),
        );
        match gpu {
            Some(gpu) => self.placement_for_gpu_with_worker(gpu.ordinal, cuda_ordinal),
            None => self.placement_for_gpu_with_worker(cuda_ordinal, cuda_ordinal),
        }
    }

    pub fn preprocess_worker_groups(&self, desired_workers: usize) -> Vec<NumaWorkerGroup> {
        let desired_workers = desired_workers.max(1);
        let nodes: Vec<&NumaNodeTopology> = self
            .numa_nodes
            .iter()
            .filter(|node| !node.cpus.is_empty())
            .collect();
        if nodes.is_empty() {
            return vec![NumaWorkerGroup {
                numa_node: None,
                cpus: self.fallback_cpus.clone(),
                worker_count: desired_workers,
            }];
        }

        let group_count = nodes.len().min(desired_workers);
        let base = desired_workers / group_count;
        let extra = desired_workers % group_count;
        nodes
            .into_iter()
            .take(group_count)
            .enumerate()
            .map(|(idx, node)| NumaWorkerGroup {
                numa_node: Some(node.node),
                cpus: node.cpus.clone(),
                worker_count: base + usize::from(idx < extra),
            })
            .collect()
    }

    pub fn route_cost_from_numa(&self, placement: &WorkerPlacement, ingress: Option<i32>) -> u32 {
        match (placement.numa_node, ingress) {
            (Some(worker), Some(request)) if worker == request => 0,
            (Some(_), Some(_)) => 100,
            _ => 10,
        }
    }

    fn cpus_for_numa(&self, numa_node: Option<i32>) -> Option<Vec<usize>> {
        let node = numa_node?;
        self.numa_nodes
            .iter()
            .find(|topology| topology.node == node)
            .map(|topology| topology.cpus.clone())
            .filter(|cpus| !cpus.is_empty())
    }

    fn nics_for_numa(&self, numa_node: Option<i32>) -> Vec<String> {
        let Some(node) = numa_node else {
            return Vec::new();
        };
        self.nics
            .iter()
            .filter(|nic| nic.numa_node == Some(node))
            .map(|nic| nic.name.clone())
            .collect()
    }

    fn resolve_cuda_visible_gpu(
        &self,
        cuda_ordinal: usize,
        visible_devices: Option<&str>,
        device_order: Option<&str>,
    ) -> Option<&GpuTopology> {
        if let Some(selector) =
            visible_devices.and_then(|value| visible_device_selector(value, cuda_ordinal))
        {
            return self.gpu_for_visible_selector(selector);
        }

        if matches!(device_order, Some(order) if order.eq_ignore_ascii_case("PCI_BUS_ID")) {
            let mut gpus = self
                .gpus
                .iter()
                .filter(|gpu| gpu.pci_bus_id.is_some())
                .collect::<Vec<_>>();
            gpus.sort_by_key(|gpu| gpu.pci_bus_id.as_deref().unwrap_or_default());
            return gpus.get(cuda_ordinal).copied();
        }

        self.gpus.iter().find(|gpu| gpu.ordinal == cuda_ordinal)
    }

    fn gpu_for_visible_selector(&self, selector: &str) -> Option<&GpuTopology> {
        if let Ok(ordinal) = selector.parse::<usize>() {
            return self.gpus.iter().find(|gpu| gpu.ordinal == ordinal);
        }
        if selector.contains(':') {
            let bus_id = normalize_pci_bus_id(selector);
            return self
                .gpus
                .iter()
                .find(|gpu| gpu.pci_bus_id.as_deref() == Some(bus_id.as_str()));
        }
        self.gpus
            .iter()
            .find(|gpu| gpu.uuid.as_deref() == Some(selector))
    }
}

pub fn configured_cuda_device_ordinal() -> usize {
    parse_configured_cuda_device_ordinal(std::env::var("INFER_CUDA_DEVICE").ok().as_deref())
}

fn parse_configured_cuda_device_ordinal(value: Option<&str>) -> usize {
    value
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(0)
}

fn visible_device_selector(visible_devices: &str, cuda_ordinal: usize) -> Option<&str> {
    visible_devices
        .split(',')
        .map(str::trim)
        .filter(|device| !device.is_empty())
        .nth(cuda_ordinal)
}

pub fn bind_current_thread_to_cpus(
    cpus: &[usize],
    label: impl Into<String>,
) -> AffinityApplyResult {
    apply_affinity_to_tids(cpus, [0].into_iter(), label)
}

pub fn bind_process_to_cpus(cpus: &[usize], label: impl Into<String>) -> AffinityApplyResult {
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    {
        let tids = fs::read_dir("/proc/self/task")
            .ok()
            .into_iter()
            .flat_map(|entries| entries.filter_map(Result::ok))
            .filter_map(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .parse::<libc::pid_t>()
                    .ok()
            })
            .collect::<Vec<_>>();
        return apply_affinity_to_tids(cpus, tids.into_iter(), label);
    }

    #[cfg(not(all(target_os = "linux", feature = "cuda")))]
    {
        let _ = cpus;
        AffinityApplyResult {
            label: label.into(),
            applied: false,
            requested_cpus: Vec::new(),
            applied_threads: 0,
            failed_threads: 0,
            reason: "CPU affinity is only applied on linux cuda builds".to_string(),
        }
    }
}

pub fn bind_current_thread_to_placement(
    placement: &WorkerPlacement,
    label: impl Into<String>,
) -> AffinityApplyResult {
    bind_current_thread_to_cpus(&placement.cpus, label)
}

pub fn bind_process_to_placement(
    placement: &WorkerPlacement,
    label: impl Into<String>,
) -> AffinityApplyResult {
    bind_process_to_cpus(&placement.cpus, label)
}

fn apply_affinity_to_tids(
    cpus: &[usize],
    tids: impl Iterator<Item = i32>,
    label: impl Into<String>,
) -> AffinityApplyResult {
    let label = label.into();
    if cpus.is_empty() {
        return AffinityApplyResult {
            label,
            applied: false,
            requested_cpus: Vec::new(),
            applied_threads: 0,
            failed_threads: 0,
            reason: "no CPUs available for affinity".to_string(),
        };
    }

    #[cfg(all(target_os = "linux", feature = "cuda"))]
    {
        let mut applied_threads = 0usize;
        let mut failed_threads = 0usize;
        let mut saw_thread = false;
        for tid in tids {
            saw_thread = true;
            match sched_setaffinity(tid, cpus) {
                Ok(()) => applied_threads += 1,
                Err(err) => {
                    failed_threads += 1;
                    log::warn!("CPU affinity apply failed label={label} tid={tid}: {err}");
                }
            }
        }
        if !saw_thread {
            match sched_setaffinity(0, cpus) {
                Ok(()) => applied_threads += 1,
                Err(err) => {
                    failed_threads += 1;
                    log::warn!("CPU affinity apply failed label={label} tid=0: {err}");
                }
            }
        }
        let applied = applied_threads > 0 && failed_threads == 0;
        return AffinityApplyResult {
            label,
            applied,
            requested_cpus: cpus.to_vec(),
            applied_threads,
            failed_threads,
            reason: if applied {
                "applied".to_string()
            } else {
                "one or more affinity calls failed".to_string()
            },
        };
    }

    #[cfg(not(all(target_os = "linux", feature = "cuda")))]
    {
        let _ = tids;
        AffinityApplyResult {
            label,
            applied: false,
            requested_cpus: cpus.to_vec(),
            applied_threads: 0,
            failed_threads: 0,
            reason: "CPU affinity is only applied on linux cuda builds".to_string(),
        }
    }
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
fn sched_setaffinity(tid: libc::pid_t, cpus: &[usize]) -> std::io::Result<()> {
    let mut set = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
    unsafe {
        libc::CPU_ZERO(&mut set);
    }
    let mut inserted = 0usize;
    for &cpu in cpus {
        if cpu < libc::CPU_SETSIZE as usize {
            unsafe {
                libc::CPU_SET(cpu, &mut set);
            }
            inserted += 1;
        }
    }
    if inserted == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "all requested CPUs exceed CPU_SETSIZE",
        ));
    }
    let rc = unsafe {
        libc::sched_setaffinity(
            tid,
            std::mem::size_of::<libc::cpu_set_t>(),
            std::ptr::addr_of!(set),
        )
    };
    if rc == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

pub fn sample_process_numa_maps() -> Option<NumaMemoryStats> {
    sample_numa_maps(Path::new("/proc/self/numa_maps"))
}

fn sample_numa_maps(path: &Path) -> Option<NumaMemoryStats> {
    let content = fs::read_to_string(path).ok()?;
    let mut per_node: HashMap<i32, u64> = HashMap::new();
    for token in content.split_whitespace() {
        let Some(rest) = token.strip_prefix('N') else {
            continue;
        };
        let Some((node, pages)) = rest.split_once('=') else {
            continue;
        };
        let Ok(node) = node.parse::<i32>() else {
            continue;
        };
        let Ok(pages) = pages.parse::<u64>() else {
            continue;
        };
        *per_node.entry(node).or_default() += pages;
    }
    let mut per_node_pages = per_node.into_iter().collect::<Vec<_>>();
    per_node_pages.sort_by_key(|(node, _)| *node);
    let total_pages = per_node_pages.iter().map(|(_, pages)| *pages).sum();
    Some(NumaMemoryStats {
        total_pages,
        per_node_pages,
    })
}

#[cfg(target_os = "linux")]
fn discover_numa_nodes(sysfs: &Path) -> Vec<NumaNodeTopology> {
    let node_dir = sysfs.join("devices/system/node");
    let Ok(entries) = fs::read_dir(node_dir) else {
        return Vec::new();
    };
    entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            let node = name.strip_prefix("node")?.parse::<i32>().ok()?;
            let cpus = read_cpu_list(entry.path().join("cpulist")).unwrap_or_default();
            Some(NumaNodeTopology { node, cpus })
        })
        .collect()
}

#[cfg(target_os = "linux")]
fn discover_gpus(sysfs: &Path, nics: &[NicTopology]) -> Vec<GpuTopology> {
    let identities_by_bus = nvidia_smi_gpu_identity_map();
    let devices = pci_devices(sysfs);
    let mut gpus = Vec::new();
    for device in devices {
        let vendor = read_trimmed(device.join("vendor")).unwrap_or_default();
        if vendor != "0x10de" {
            continue;
        }
        let class = read_trimmed(device.join("class")).unwrap_or_default();
        if !(class.starts_with("0x03") || class.starts_with("0x12")) {
            continue;
        }
        let Some(bus_id) = device
            .file_name()
            .map(|name| normalize_pci_bus_id(&name.to_string_lossy()))
        else {
            continue;
        };
        let local_cpus = read_cpu_list(device.join("local_cpulist")).unwrap_or_default();
        let numa_node = read_numa_node(device.join("numa_node"));
        let identity = identities_by_bus.get(&bus_id);
        let ordinal = identity
            .map(|identity| identity.ordinal)
            .unwrap_or(gpus.len());
        let uuid = identity.and_then(|identity| identity.uuid.clone());
        let nearest_nics = nearest_nics_for_gpu(numa_node, &local_cpus, nics);
        gpus.push(GpuTopology {
            ordinal,
            pci_bus_id: Some(bus_id),
            uuid,
            numa_node,
            local_cpus,
            nearest_nics,
        });
    }
    gpus
}

#[cfg(target_os = "linux")]
fn discover_nics(sysfs: &Path) -> Vec<NicTopology> {
    pci_devices(sysfs)
        .into_iter()
        .flat_map(|device| {
            let net_dir = device.join("net");
            let names = fs::read_dir(net_dir)
                .ok()
                .into_iter()
                .flat_map(|entries| entries.filter_map(Result::ok))
                .map(|entry| entry.file_name().to_string_lossy().into_owned())
                .collect::<Vec<_>>();
            let pci_bus_id = device
                .file_name()
                .map(|name| normalize_pci_bus_id(&name.to_string_lossy()));
            let numa_node = read_numa_node(device.join("numa_node"));
            let local_cpus = read_cpu_list(device.join("local_cpulist")).unwrap_or_default();
            names.into_iter().map(move |name| NicTopology {
                name,
                pci_bus_id: pci_bus_id.clone(),
                numa_node,
                local_cpus: local_cpus.clone(),
            })
        })
        .collect()
}

#[cfg(target_os = "linux")]
fn nearest_nics_for_gpu(
    numa_node: Option<i32>,
    local_cpus: &[usize],
    nics: &[NicTopology],
) -> Vec<String> {
    let same_numa = nics
        .iter()
        .filter(|nic| numa_node.is_some() && nic.numa_node == numa_node)
        .map(|nic| nic.name.clone())
        .collect::<Vec<_>>();
    if !same_numa.is_empty() {
        return same_numa;
    }
    nics.iter()
        .filter(|nic| has_cpu_intersection(local_cpus, &nic.local_cpus))
        .map(|nic| nic.name.clone())
        .collect()
}

#[cfg(target_os = "linux")]
fn pci_devices(sysfs: &Path) -> Vec<PathBuf> {
    fs::read_dir(sysfs.join("bus/pci/devices"))
        .ok()
        .into_iter()
        .flat_map(|entries| entries.filter_map(Result::ok))
        .map(|entry| entry.path())
        .collect()
}

#[cfg(target_os = "linux")]
struct NvidiaSmiGpuIdentity {
    ordinal: usize,
    uuid: Option<String>,
}

#[cfg(target_os = "linux")]
fn nvidia_smi_gpu_identity_map() -> HashMap<String, NvidiaSmiGpuIdentity> {
    let Ok(output) = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,uuid,pci.bus_id",
            "--format=csv,noheader,nounits",
        ])
        .output()
    else {
        return HashMap::new();
    };
    if !output.status.success() {
        return HashMap::new();
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| {
            let mut fields = line.split(',').map(str::trim);
            let ordinal = fields.next()?.parse::<usize>().ok()?;
            let uuid = fields
                .next()
                .map(str::to_string)
                .filter(|uuid| !uuid.is_empty());
            let bus_id = normalize_pci_bus_id(fields.next()?);
            Some((bus_id, NvidiaSmiGpuIdentity { ordinal, uuid }))
        })
        .collect()
}

#[cfg(target_os = "linux")]
fn read_numa_node(path: impl AsRef<Path>) -> Option<i32> {
    let node = read_trimmed(path)?.parse::<i32>().ok()?;
    (node >= 0).then_some(node)
}

#[cfg(target_os = "linux")]
fn read_cpu_list(path: impl AsRef<Path>) -> Option<Vec<usize>> {
    Some(parse_cpu_list(&read_trimmed(path)?))
}

#[cfg(target_os = "linux")]
fn read_trimmed(path: impl AsRef<Path>) -> Option<String> {
    fs::read_to_string(path)
        .ok()
        .map(|value| value.trim().to_string())
}

pub fn parse_cpu_list(raw: &str) -> Vec<usize> {
    let mut cpus = Vec::new();
    for part in raw
        .trim()
        .split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty())
    {
        if let Some((start, end)) = part.split_once('-') {
            let Ok(start) = start.parse::<usize>() else {
                continue;
            };
            let Ok(end) = end.parse::<usize>() else {
                continue;
            };
            if start <= end {
                cpus.extend(start..=end);
            }
        } else if let Ok(cpu) = part.parse::<usize>() {
            cpus.push(cpu);
        }
    }
    cpus.sort_unstable();
    cpus.dedup();
    cpus
}

pub fn normalize_pci_bus_id(raw: &str) -> String {
    let mut value = raw.trim().to_ascii_lowercase();
    for prefix in ["pci:", "00000000:", "0000:"] {
        if let Some(stripped) = value.strip_prefix(prefix) {
            value = stripped.to_string();
        }
    }
    value
}

#[cfg(target_os = "linux")]
fn has_cpu_intersection(left: &[usize], right: &[usize]) -> bool {
    left.iter().any(|cpu| right.contains(cpu))
}

fn format_cpu_list(cpus: &[usize]) -> String {
    if cpus.is_empty() {
        return "none".to_string();
    }
    cpus.iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_list_parser_handles_ranges_and_sparse_values() {
        assert_eq!(parse_cpu_list("0-3,8,10-11"), vec![0, 1, 2, 3, 8, 10, 11]);
        assert_eq!(parse_cpu_list(" 2,2,4-4 "), vec![2, 4]);
        assert!(parse_cpu_list("").is_empty());
    }

    #[test]
    fn pci_bus_id_normalization_matches_nvidia_smi_and_sysfs() {
        assert_eq!(normalize_pci_bus_id("00000000:17:00.0"), "17:00.0");
        assert_eq!(normalize_pci_bus_id("0000:ca:00.0"), "ca:00.0");
        assert_eq!(normalize_pci_bus_id("PCI:3B:00.0"), "3b:00.0");
    }

    #[test]
    fn gpu_placement_prefers_gpu_numa_then_same_numa_nic() {
        let topology = RuntimeTopology {
            numa_nodes: vec![
                NumaNodeTopology {
                    node: 0,
                    cpus: vec![0, 1],
                },
                NumaNodeTopology {
                    node: 1,
                    cpus: vec![2, 3],
                },
            ],
            gpus: vec![GpuTopology {
                ordinal: 0,
                pci_bus_id: Some("17:00.0".to_string()),
                uuid: None,
                numa_node: Some(1),
                local_cpus: Vec::new(),
                nearest_nics: vec!["mlx5_0".to_string()],
            }],
            nics: Vec::new(),
            fallback_cpus: vec![0, 1, 2, 3],
        };
        let placement = topology.placement_for_gpu(0);
        assert_eq!(placement.numa_node, Some(1));
        assert_eq!(placement.cpus, vec![2, 3]);
        assert_eq!(placement.nics, vec!["mlx5_0"]);
    }

    #[test]
    fn configured_cuda_device_ordinal_parser_uses_zero_default() {
        assert_eq!(parse_configured_cuda_device_ordinal(None), 0);
        assert_eq!(parse_configured_cuda_device_ordinal(Some("2")), 2);
        assert_eq!(parse_configured_cuda_device_ordinal(Some(" 7 ")), 7);
        assert_eq!(parse_configured_cuda_device_ordinal(Some("bad")), 0);
    }

    #[test]
    fn cuda_visible_devices_selector_resolves_physical_gpu() {
        let topology = RuntimeTopology {
            numa_nodes: Vec::new(),
            gpus: vec![
                GpuTopology {
                    ordinal: 0,
                    pci_bus_id: Some("17:00.0".to_string()),
                    uuid: Some("GPU-a".to_string()),
                    numa_node: Some(0),
                    local_cpus: vec![0, 1],
                    nearest_nics: Vec::new(),
                },
                GpuTopology {
                    ordinal: 2,
                    pci_bus_id: Some("ca:00.0".to_string()),
                    uuid: Some("GPU-b".to_string()),
                    numa_node: Some(1),
                    local_cpus: vec![2, 3],
                    nearest_nics: Vec::new(),
                },
            ],
            nics: Vec::new(),
            fallback_cpus: vec![0, 1, 2, 3],
        };

        let gpu = topology
            .resolve_cuda_visible_gpu(0, Some("2,0"), None)
            .unwrap();
        assert_eq!(gpu.ordinal, 2);
        let gpu = topology
            .resolve_cuda_visible_gpu(0, Some("GPU-b"), None)
            .unwrap();
        assert_eq!(gpu.ordinal, 2);
        let gpu = topology
            .resolve_cuda_visible_gpu(1, None, Some("PCI_BUS_ID"))
            .unwrap();
        assert_eq!(gpu.ordinal, 2);
    }

    #[test]
    fn preprocess_groups_split_workers_across_numa_nodes() {
        let topology = RuntimeTopology {
            numa_nodes: vec![
                NumaNodeTopology {
                    node: 0,
                    cpus: vec![0, 1],
                },
                NumaNodeTopology {
                    node: 1,
                    cpus: vec![2, 3],
                },
            ],
            gpus: Vec::new(),
            nics: Vec::new(),
            fallback_cpus: vec![0, 1, 2, 3],
        };
        let groups = topology.preprocess_worker_groups(5);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].worker_count, 3);
        assert_eq!(groups[1].worker_count, 2);
    }

    #[test]
    fn numa_maps_sampler_accumulates_node_pages() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("numa_maps");
        fs::write(&path, "7f N0=2 N1=3 kernelpagesize_kB=4\n8f N1=5 dirty=1\n").unwrap();
        let stats = sample_numa_maps(&path).unwrap();
        assert_eq!(stats.total_pages, 10);
        assert_eq!(stats.per_node_pages, vec![(0, 2), (1, 8)]);
    }
}
