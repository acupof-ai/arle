use std::path::Path;

#[cfg(any(target_os = "linux", test))]
use std::path::PathBuf;

use crate::DiskIoMode;

pub(crate) fn log_gate(root: &Path, mode: DiskIoMode) {
    if mode != DiskIoMode::Direct {
        return;
    }
    match probe(root, mode) {
        Ok(()) => log::info!("GDS hardware gate ready at {}", root.display()),
        Err(reason) => log::info!("GDS hardware gate blocked: {reason}"),
    }
}

fn probe(root: &Path, mode: DiskIoMode) -> Result<(), String> {
    if mode != DiskIoMode::Direct {
        return Err("L3 disk mode is not O_DIRECT".into());
    }
    probe_linux(root)
}

#[cfg(not(target_os = "linux"))]
fn probe_linux(_root: &Path) -> Result<(), String> {
    Err("GDS requires Linux".into())
}

#[cfg(target_os = "linux")]
fn probe_linux(root: &Path) -> Result<(), String> {
    use std::os::unix::fs::FileTypeExt as _;

    if !libcufile_loadable() {
        return Err("libcufile is not loadable".into());
    }
    let config = std::fs::read_to_string("/etc/cufile.json")
        .map_err(|_| "/etc/cufile.json is unavailable")?;
    let compat = json_bool(&config, "allow_compat_mode")
        .ok_or("/etc/cufile.json lacks allow_compat_mode")?;
    let p2pdma =
        json_bool(&config, "use_pci_p2pdma").ok_or("/etc/cufile.json lacks use_pci_p2pdma")?;
    if compat {
        return Err("cuFile compatibility mode is enabled".into());
    }
    if !(Path::new("/sys/module/nvidia_fs").exists()
        || p2pdma && p2pmem_available(Path::new("/sys/bus/pci/devices")))
    {
        return Err("neither nvidia_fs nor PCIe P2PDMA is available".into());
    }
    let root = root
        .canonicalize()
        .map_err(|_| "L3 spill root is unavailable")?;
    let mountinfo = std::fs::read_to_string("/proc/self/mountinfo")
        .map_err(|_| "L3 mount table is unavailable")?;
    let (source, filesystem) = mount_for(&mountinfo, &root).ok_or("L3 mount is unavailable")?;
    if !matches!(filesystem.as_str(), "ext4" | "xfs") {
        return Err(format!("L3 filesystem {filesystem} is not GDS-gated"));
    }
    if !Path::new(&source)
        .metadata()
        .is_ok_and(|metadata| metadata.file_type().is_block_device())
    {
        return Err(format!("L3 mount source {source} is not a block device"));
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn libcufile_loadable() -> bool {
    ["libcufile.so.0", "libcufile.so"].iter().any(|name| {
        let name = std::ffi::CString::new(*name).expect("static library name");
        // SAFETY: `name` is NUL-terminated; a non-null handle is closed once.
        let handle = unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_LAZY | libc::RTLD_LOCAL) };
        if handle.is_null() {
            return false;
        }
        // SAFETY: `handle` was returned by dlopen above and has not been closed.
        unsafe { libc::dlclose(handle) };
        true
    })
}

#[cfg(any(target_os = "linux", test))]
fn json_bool(config: &str, key: &str) -> Option<bool> {
    let value = config
        .split_once(&format!("\"{key}\""))?
        .1
        .split_once(':')?
        .1
        .trim_start();
    value
        .starts_with("true")
        .then_some(true)
        .or_else(|| value.starts_with("false").then_some(false))
}

#[cfg(target_os = "linux")]
fn p2pmem_available(devices: &Path) -> bool {
    std::fs::read_dir(devices).is_ok_and(|entries| {
        entries
            .filter_map(Result::ok)
            .any(|entry| entry.path().join("p2pmem").exists())
    })
}

#[cfg(any(target_os = "linux", test))]
fn mount_for(mountinfo: &str, root: &Path) -> Option<(String, String)> {
    mountinfo
        .lines()
        .filter_map(|line| {
            let (left, right) = line.split_once(" - ")?;
            let point = PathBuf::from(unescape_mount(left.split_whitespace().nth(4)?));
            let mut right = right.split_whitespace();
            let filesystem = right.next()?.to_string();
            let source = unescape_mount(right.next()?);
            root.starts_with(&point)
                .then_some((point, source, filesystem))
        })
        .max_by_key(|(point, _, _)| point.as_os_str().len())
        .map(|(_, source, filesystem)| (source, filesystem))
}

#[cfg(any(target_os = "linux", test))]
fn unescape_mount(value: &str) -> String {
    value
        .replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
}
