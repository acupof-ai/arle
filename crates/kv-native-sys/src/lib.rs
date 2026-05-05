//! Pure-Rust persistence substrate for the KV tier.
//!
//! This crate is POSIX-only (Linux + macOS); it uses `nix`, `memmap2`, and
//! `libc` directly with no FFI of its own. The exported surface — file/block
//! I/O, the WAL, file mmap, POSIX shm, and a host arena — was historically
//! a Zig FFI layer; the migration to native Rust landed in tranches on
//! 2026-05-05 (see `docs/experience/wins/`).

use std::fs::OpenOptions;
use std::io;
use std::io::Write;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::ffi::OsStringExt;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::path::PathBuf;
use std::ptr::NonNull;

pub const KV_MMAP_PATH_CAP: usize = 512;
pub const KV_SHM_NAME_CAP: usize = 128;

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvMmapDescriptor {
    pub len: usize,
    pub path_len: usize,
    pub path: [u8; KV_MMAP_PATH_CAP],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvSharedMemoryDescriptor {
    pub len: usize,
    pub generation: u64,
    pub name_len: usize,
    pub name: [u8; KV_SHM_NAME_CAP],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvHostArenaRegion {
    pub offset: u64,
    pub len: usize,
}

/// Opaque pointer type returned by [`host_arena_create`]. Internally a
/// pointer to a boxed [`KvHostArena`]; the `_private` field keeps the
/// shape stable for any caller that may have stored the bare pointer.
#[repr(C)]
pub struct KvHostArenaHandle {
    _private: [u8; 0],
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct KvWalRecord {
    pub kind: u8,
    pub key: Vec<u8>,
    pub value: Vec<u8>,
}

fn path_bytes(path: &Path) -> &[u8] {
    path.as_os_str().as_bytes()
}

pub fn write_file(path: &Path, bytes: &[u8]) -> io::Result<()> {
    if path.as_os_str().is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "write_file: empty path",
        ));
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o644)
        .open(path)?;
    file.write_all(bytes)
}

pub fn write_file_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    if path.as_os_str().is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "write_file_atomic: empty path",
        ));
    }
    // Compose `<path>.tmp` by appending bytes to the OsString — matches the Zig
    // implementation which used `std.fmt.allocPrint("{s}.tmp", .{path})` on the
    // raw byte slice (no extension semantics).
    let mut tmp_os = path.as_os_str().to_owned();
    tmp_os.push(".tmp");
    let tmp_path = PathBuf::from(tmp_os);

    let result = (|| -> io::Result<()> {
        {
            let mut tmp = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .mode(0o644)
                .open(&tmp_path)?;
            tmp.write_all(bytes)?;
            tmp.sync_data()?;
        }
        std::fs::rename(&tmp_path, path)?;
        // fsync the parent directory so the rename is durable on power loss.
        let parent = path.parent().filter(|p| !p.as_os_str().is_empty());
        let parent = parent.unwrap_or_else(|| Path::new("."));
        let dir = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY)
            .open(parent)?;
        dir.sync_all()?;
        Ok(())
    })();

    if result.is_err() {
        // Best-effort cleanup of the staging file; ignore secondary errors.
        let _ = std::fs::remove_file(&tmp_path);
    }
    result
}

pub fn read_file(path: &Path) -> io::Result<Vec<u8>> {
    if path.as_os_str().is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "read_file: empty path",
        ));
    }
    // Match the Zig semantics: a 0-byte file returns Ok(empty), but a missing
    // file surfaces as NotFound. `std::fs::read` already does the right thing
    // for both — it does NOT raise NotFound on a 0-byte existing file, and it
    // does on a missing one.
    std::fs::read(path)
}

pub fn remove_file(path: &Path, ignore_not_found: bool) -> io::Result<()> {
    if path.as_os_str().is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "remove_file: empty path",
        ));
    }
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == io::ErrorKind::NotFound && ignore_not_found => Ok(()),
        Err(err) => Err(err),
    }
}

pub fn block_path(root: &Path, fingerprint: [u8; 16]) -> io::Result<PathBuf> {
    let mut filename = String::with_capacity(35);
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for byte in fingerprint.iter() {
        filename.push(HEX[(byte >> 4) as usize] as char);
        filename.push(HEX[(byte & 0x0f) as usize] as char);
    }
    filename.push_str(".kv");
    Ok(root.join(filename))
}

pub fn write_block_atomic(root: &Path, fingerprint: [u8; 16], bytes: &[u8]) -> io::Result<()> {
    let path = block_path(root, fingerprint)?;
    write_file_atomic(&path, bytes)
}

pub fn read_block(root: &Path, fingerprint: [u8; 16]) -> io::Result<Vec<u8>> {
    let path = block_path(root, fingerprint)?;
    read_file(&path)
}

/// Like [`read_block`] but returns an owning guard over the heap-allocated
/// payload. The guard derefs to `&[u8]` and frees its buffer on `Drop`.
///
/// Historically this skipped a Zig→Rust copy; with the pure-Rust substrate
/// the payload is a `Vec<u8>` directly, so this is now a thin wrapper that
/// preserves the public API for callers like `DiskStore::get_block`.
pub fn read_block_owned(root: &Path, fingerprint: [u8; 16]) -> io::Result<KvNativeOwnedBytes> {
    let bytes = read_block(root, fingerprint)?;
    Ok(KvNativeOwnedBytes::from_vec(bytes))
}

pub fn remove_block(root: &Path, fingerprint: [u8; 16], ignore_not_found: bool) -> io::Result<()> {
    let path = block_path(root, fingerprint)?;
    remove_file(&path, ignore_not_found)
}

pub fn wal_append(path: &Path, kind: u8, key: &[u8], value: &[u8]) -> io::Result<()> {
    if key.len() > u32::MAX as usize || value.len() > u32::MAX as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "wal_append: key/value exceeds u32::MAX",
        ));
    }
    let mut file = OpenOptions::new()
        .write(true)
        .append(true)
        .create(true)
        .mode(0o644)
        .open(path)?;
    // If the file is fresh (size 0), prepend the WAL magic so that replay can
    // distinguish a valid empty log from a corrupted prefix.
    let metadata = file.metadata()?;
    if metadata.len() == 0 {
        file.write_all(WAL_MAGIC)?;
    }
    let mut header = [0_u8; 9];
    header[0] = kind;
    header[1..5].copy_from_slice(&u32::to_le_bytes(key.len() as u32));
    header[5..9].copy_from_slice(&u32::to_le_bytes(value.len() as u32));
    file.write_all(&header)?;
    file.write_all(key)?;
    file.write_all(value)?;
    file.sync_data()?;
    Ok(())
}

pub fn wal_replay(path: &Path) -> io::Result<Vec<KvWalRecord>> {
    let bytes = match read_file(path) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(err),
    };
    if bytes.is_empty() {
        return Ok(Vec::new());
    }
    decode_wal_records(&bytes)
}

pub fn mmap_create(path: &Path, len: usize) -> io::Result<KvMmapDescriptor> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let path_raw = path_bytes(path);
    if path_raw.len() > KV_MMAP_PATH_CAP {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "mmap_create: path exceeds KV_MMAP_PATH_CAP",
        ));
    }
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o644)
        .open(path)?;
    file.set_len(len as u64)?;

    let mut out = KvMmapDescriptor {
        len,
        path_len: path_raw.len(),
        path: [0; KV_MMAP_PATH_CAP],
    };
    out.path[..path_raw.len()].copy_from_slice(path_raw);
    Ok(out)
}

pub fn mmap_write(desc: &KvMmapDescriptor, offset: usize, bytes: &[u8]) -> io::Result<()> {
    if offset
        .checked_add(bytes.len())
        .is_none_or(|end| end > desc.len)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "mmap_write: offset+bytes_len exceeds descriptor len",
        ));
    }
    let path = desc.path_buf()?;
    let file = OpenOptions::new().read(true).write(true).open(&path)?;
    // Safety: `MmapMut` over a non-shared file. We only borrow the mapping
    // for the duration of this call; no aliasing with other Rust references.
    let mut mapping = unsafe { memmap2::MmapMut::map_mut(&file)? };
    if mapping.len() < desc.len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "mmap_write: backing file shorter than descriptor len",
        ));
    }
    mapping[offset..offset + bytes.len()].copy_from_slice(bytes);
    // `MmapMut::flush` calls msync(MS_SYNC) under the hood, matching the
    // previous Zig path which used `c.msync(mapping, desc.len, c.MS_SYNC)`.
    mapping.flush()?;
    Ok(())
}

pub fn mmap_read(desc: &KvMmapDescriptor, offset: usize, len: usize) -> io::Result<Vec<u8>> {
    if offset.checked_add(len).is_none_or(|end| end > desc.len) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "mmap_read: offset+len exceeds descriptor len",
        ));
    }
    let path = desc.path_buf()?;
    let file = OpenOptions::new().read(true).open(&path)?;
    // Safety: read-only mapping over a regular file; bytes are immutable for
    // the lifetime of the borrow, which ends before the function returns.
    let mapping = unsafe { memmap2::Mmap::map(&file)? };
    if mapping.len() < desc.len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "mmap_read: backing file shorter than descriptor len",
        ));
    }
    Ok(mapping[offset..offset + len].to_vec())
}

// Layout of the in-band shm header that every mapping carries. Mirrors the Zig
// `ShmHeader` struct verbatim (8-byte magic + u64 generation + u64 payload_len),
// so a Rust-side rewrite here is wire-compatible with any pre-port persisted
// segment (none should exist — these segments are ephemeral — but we keep the
// invariant explicit).
const SHM_MAGIC: &[u8; 8] = b"KVSHM001";
const SHM_HEADER_BYTES: usize = 24;
const SHM_CREATE_RETRY_LIMIT: usize = 64;
const SHM_CREATE_RETRY_SLEEP_US: u64 = 1_000;

static SHM_GENERATION_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn next_shm_generation() -> u64 {
    use std::sync::atomic::Ordering;
    let g = SHM_GENERATION_COUNTER.fetch_add(1, Ordering::Relaxed);
    if g == 0 {
        // The Zig path retried once on overflow-to-zero; do the same.
        SHM_GENERATION_COUNTER.fetch_add(1, Ordering::Relaxed)
    } else {
        g
    }
}

fn shm_total_len(payload_len: usize) -> io::Result<usize> {
    SHM_HEADER_BYTES.checked_add(payload_len).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "shm: header + payload_len overflows usize",
        )
    })
}

fn write_shm_header(mapping: &mut memmap2::MmapMut, generation: u64, payload_len: usize) {
    mapping[0..8].copy_from_slice(SHM_MAGIC);
    mapping[8..16].copy_from_slice(&u64::to_le_bytes(generation));
    mapping[16..24].copy_from_slice(&u64::to_le_bytes(payload_len as u64));
}

fn validate_shm_header(mapping: &[u8], desc: &KvSharedMemoryDescriptor) -> io::Result<()> {
    if mapping.len() < SHM_HEADER_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "shm: mapping smaller than header",
        ));
    }
    if &mapping[0..8] != SHM_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "shm: header magic mismatch",
        ));
    }
    let mut buf = [0_u8; 8];
    buf.copy_from_slice(&mapping[8..16]);
    if u64::from_le_bytes(buf) != desc.generation {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "shm: descriptor generation does not match header",
        ));
    }
    buf.copy_from_slice(&mapping[16..24]);
    if u64::from_le_bytes(buf) != desc.len as u64 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "shm: descriptor payload_len does not match header",
        ));
    }
    Ok(())
}

fn shm_open_create_exclusive_retrying(
    name_c: &std::ffi::CString,
) -> io::Result<std::os::fd::OwnedFd> {
    use nix::errno::Errno;
    use nix::fcntl::OFlag;
    use nix::sys::mman::shm_open;
    use nix::sys::stat::Mode;
    let mut retries = 0_usize;
    loop {
        match shm_open(
            name_c.as_c_str(),
            OFlag::O_RDWR | OFlag::O_CREAT | OFlag::O_EXCL,
            Mode::from_bits_truncate(0o600),
        ) {
            Ok(fd) => return Ok(fd),
            Err(Errno::EINTR) => continue,
            Err(Errno::EEXIST) => {
                if retries >= SHM_CREATE_RETRY_LIMIT {
                    return Err(io::Error::from_raw_os_error(libc::EEXIST));
                }
                retries += 1;
                std::thread::sleep(std::time::Duration::from_micros(SHM_CREATE_RETRY_SLEEP_US));
            }
            Err(e) => return Err(io::Error::from_raw_os_error(e as i32)),
        }
    }
}

fn shm_open_existing(name_c: &std::ffi::CString, write: bool) -> io::Result<std::os::fd::OwnedFd> {
    use nix::fcntl::OFlag;
    use nix::sys::mman::shm_open;
    use nix::sys::stat::Mode;
    let flag = if write {
        OFlag::O_RDWR
    } else {
        OFlag::O_RDONLY
    };
    shm_open(name_c.as_c_str(), flag, Mode::from_bits_truncate(0o600))
        .map_err(|e| io::Error::from_raw_os_error(e as i32))
}

pub fn shm_create(name: &str, len: usize) -> io::Result<KvSharedMemoryDescriptor> {
    if name.is_empty() || !name.starts_with('/') || name.len() > KV_SHM_NAME_CAP {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "shm_create: name must start with '/' and fit KV_SHM_NAME_CAP",
        ));
    }
    let total_len = shm_total_len(len)?;
    let name_c = std::ffi::CString::new(name).map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("shm_create: name has interior NUL: {err}"),
        )
    })?;

    let fd = shm_open_create_exclusive_retrying(&name_c)?;
    let file = std::fs::File::from(fd);
    if let Err(err) = file.set_len(total_len as u64) {
        let _ = nix::sys::mman::shm_unlink(name_c.as_c_str());
        return Err(err);
    }
    // Safety: we just created and sized this fd; map_mut ties the mapping to
    // the file's lifetime which we hold for the duration of this call.
    let mut mapping = match unsafe { memmap2::MmapOptions::new().len(total_len).map_mut(&file) } {
        Ok(m) => m,
        Err(err) => {
            let _ = nix::sys::mman::shm_unlink(name_c.as_c_str());
            return Err(err);
        }
    };

    let generation = next_shm_generation();
    write_shm_header(&mut mapping, generation, len);

    let mut out = KvSharedMemoryDescriptor {
        len,
        generation,
        name_len: name.len(),
        name: [0; KV_SHM_NAME_CAP],
    };
    out.name[..name.len()].copy_from_slice(name.as_bytes());
    Ok(out)
}

pub fn shm_write(desc: &KvSharedMemoryDescriptor, offset: usize, bytes: &[u8]) -> io::Result<()> {
    if offset
        .checked_add(bytes.len())
        .is_none_or(|end| end > desc.len)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "shm_write: offset+bytes_len exceeds descriptor len",
        ));
    }
    let name = desc.name()?;
    let total_len = shm_total_len(desc.len)?;
    let name_c = std::ffi::CString::new(name).map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("shm_write: name has interior NUL: {err}"),
        )
    })?;
    let fd = shm_open_existing(&name_c, true)?;
    let file = std::fs::File::from(fd);
    let mut mapping = unsafe { memmap2::MmapOptions::new().len(total_len).map_mut(&file)? };
    validate_shm_header(&mapping, desc)?;
    let payload = &mut mapping[SHM_HEADER_BYTES..];
    payload[offset..offset + bytes.len()].copy_from_slice(bytes);
    mapping.flush()?;
    Ok(())
}

pub fn shm_read(desc: &KvSharedMemoryDescriptor, offset: usize, len: usize) -> io::Result<Vec<u8>> {
    if offset.checked_add(len).is_none_or(|end| end > desc.len) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "shm_read: offset+len exceeds descriptor len",
        ));
    }
    let name = desc.name()?;
    let total_len = shm_total_len(desc.len)?;
    let name_c = std::ffi::CString::new(name).map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("shm_read: name has interior NUL: {err}"),
        )
    })?;
    let fd = shm_open_existing(&name_c, false)?;
    let file = std::fs::File::from(fd);
    let mapping = unsafe { memmap2::MmapOptions::new().len(total_len).map(&file)? };
    validate_shm_header(&mapping, desc)?;
    let payload = &mapping[SHM_HEADER_BYTES..];
    Ok(payload[offset..offset + len].to_vec())
}

pub fn shm_unlink(desc: &KvSharedMemoryDescriptor) -> io::Result<()> {
    let name = desc.name()?;
    let total_len = shm_total_len(desc.len)?;
    let name_c = std::ffi::CString::new(name).map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("shm_unlink: name has interior NUL: {err}"),
        )
    })?;
    // First validate via a read-only mapping that the live segment matches our
    // descriptor — preserves the Zig behavior of returning InvalidData when a
    // stale descriptor tries to unlink a segment that was already recreated by
    // a different generation. This is what `shm_stale_descriptor_is_rejected_after_recreate`
    // pins.
    let fd = shm_open_existing(&name_c, false)?;
    let file = std::fs::File::from(fd);
    let mapping = unsafe { memmap2::MmapOptions::new().len(total_len).map(&file)? };
    validate_shm_header(&mapping, desc)?;
    drop(mapping);
    drop(file);
    nix::sys::mman::shm_unlink(name_c.as_c_str())
        .map_err(|e| io::Error::from_raw_os_error(e as i32))
}

/// Internal arena state. Anonymous mmap + bump pointer + Vec free-list.
/// Held behind a `Box`; the leaked `Box::into_raw` pointer is what
/// [`host_arena_create`] hands back as a `*mut KvHostArenaHandle`.
struct KvHostArena {
    mapping: NonNull<u8>,
    capacity_bytes: usize,
    next_offset: usize,
    reserved_bytes: usize,
    free_list: Vec<KvHostArenaRegion>,
}

fn region_end(region: KvHostArenaRegion) -> Option<usize> {
    let offset: usize = region.offset.try_into().ok()?;
    offset.checked_add(region.len)
}

fn host_arena_rewind_tail(arena: &mut KvHostArena) {
    // Iteratively drain any free-list entry whose end == next_offset, popping
    // next_offset back to that entry's offset. Mirrors the Zig
    // `hostArenaRewindTail` loop verbatim.
    loop {
        let mut found: Option<usize> = None;
        for (idx, free) in arena.free_list.iter().enumerate() {
            let Some(end) = region_end(*free) else {
                continue;
            };
            if end == arena.next_offset {
                found = Some(idx);
                break;
            }
        }
        match found {
            Some(idx) => {
                let free = arena.free_list.swap_remove(idx);
                arena.next_offset = free.offset as usize;
            }
            None => return,
        }
    }
}

pub fn host_arena_create(capacity_bytes: usize) -> io::Result<(*mut KvHostArenaHandle, *mut u8)> {
    if capacity_bytes == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "host_arena_create: capacity_bytes must be > 0",
        ));
    }
    let nz =
        std::num::NonZeroUsize::new(capacity_bytes).expect("capacity_bytes != 0 checked above");
    // Safety: anonymous mapping with no fd; nix::mmap_anonymous is the
    // documented entry for this exact pattern.
    let mapping = unsafe {
        nix::sys::mman::mmap_anonymous(
            None,
            nz,
            nix::sys::mman::ProtFlags::PROT_READ | nix::sys::mman::ProtFlags::PROT_WRITE,
            nix::sys::mman::MapFlags::MAP_PRIVATE,
        )
    }
    .map_err(|e| io::Error::from_raw_os_error(e as i32))?;

    // Best-effort: ask the kernel to back the arena with transparent huge
    // pages on Linux. Failure is silently ignored — the arena still works
    // with 4 KiB pages. Matches the Zig substrate's `madvise(MADV_HUGEPAGE)`
    // call (see kv_native.zig L621-L623 + the 2026-05-04 wins-entry note
    // about the call being a no-op on hosts where THP is disabled).
    #[cfg(target_os = "linux")]
    {
        // Safety: we just received `mapping` from mmap_anonymous and have not
        // shared it; this advisory call is sound regardless of return value.
        let _ = unsafe {
            nix::sys::mman::madvise(
                mapping,
                capacity_bytes,
                nix::sys::mman::MmapAdvise::MADV_HUGEPAGE,
            )
        };
    }

    let base: NonNull<u8> = mapping.cast();
    let arena = Box::new(KvHostArena {
        mapping: base,
        capacity_bytes,
        next_offset: 0,
        reserved_bytes: 0,
        free_list: Vec::new(),
    });
    let handle = Box::into_raw(arena).cast::<KvHostArenaHandle>();
    Ok((handle, base.as_ptr()))
}

/// Destroy a host arena created by [`host_arena_create`].
///
/// # Safety
/// The caller must ensure `handle` came from [`host_arena_create`], is still
/// live, and is not used again after this call returns.
pub unsafe fn host_arena_destroy(handle: *mut KvHostArenaHandle) -> io::Result<()> {
    if handle.is_null() {
        return Ok(());
    }
    // Safety: caller contract — `handle` came from `Box::into_raw` in
    // `host_arena_create` and is still live.
    let arena: Box<KvHostArena> = unsafe { Box::from_raw(handle.cast::<KvHostArena>()) };
    let mapping = arena.mapping.cast::<core::ffi::c_void>();
    let capacity = arena.capacity_bytes;
    drop(arena); // free Vec<KvHostArenaRegion> before munmap
    // Safety: the mapping pointer + length match the pair returned by
    // mmap_anonymous in `host_arena_create`.
    unsafe {
        nix::sys::mman::munmap(mapping, capacity)
            .map_err(|e| io::Error::from_raw_os_error(e as i32))?;
    }
    Ok(())
}

/// Query the number of bytes currently reserved inside a live host arena.
///
/// # Safety
/// The caller must ensure `handle` points to a live arena created by
/// [`host_arena_create`] and remains valid for the duration of this call.
pub unsafe fn host_arena_reserved_bytes(handle: *const KvHostArenaHandle) -> io::Result<usize> {
    if handle.is_null() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "host_arena_reserved_bytes: null handle",
        ));
    }
    // Safety: caller contract.
    let arena = unsafe { &*(handle.cast::<KvHostArena>()) };
    Ok(arena.reserved_bytes)
}

/// Reserve a contiguous region from a live host arena.
///
/// # Safety
/// The caller must ensure `handle` points to a live arena created by
/// [`host_arena_create`] and that any returned region is later released back to
/// the same arena via [`host_arena_release`] or invalidated by
/// [`host_arena_reset`] / [`host_arena_destroy`].
pub unsafe fn host_arena_reserve(
    handle: *mut KvHostArenaHandle,
    len: usize,
) -> io::Result<Option<KvHostArenaRegion>> {
    if len == 0 {
        return Ok(None);
    }
    if handle.is_null() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "host_arena_reserve: null handle",
        ));
    }
    // Safety: caller contract.
    let arena = unsafe { &mut *(handle.cast::<KvHostArena>()) };

    // Free-list first-fit. Mirrors the Zig path: linear scan, swap_remove, push
    // remainder. Both branches bump `reserved_bytes`.
    let mut idx = 0usize;
    while idx < arena.free_list.len() {
        let free = arena.free_list[idx];
        if free.len < len {
            idx += 1;
            continue;
        }
        arena.free_list.swap_remove(idx);
        if free.len > len {
            arena.free_list.push(KvHostArenaRegion {
                offset: free.offset + len as u64,
                len: free.len - len,
            });
        }
        let region = KvHostArenaRegion {
            offset: free.offset,
            len,
        };
        arena.reserved_bytes = arena
            .reserved_bytes
            .checked_add(len)
            .ok_or_else(|| io::Error::other("host_arena_reserve: reserved_bytes overflow"))?;
        return Ok(Some(region));
    }

    // Bump path. OOM = Ok(None), per the existing Rust shim contract.
    let Some(end) = arena.next_offset.checked_add(len) else {
        return Ok(None);
    };
    if end > arena.capacity_bytes {
        return Ok(None);
    }
    let region = KvHostArenaRegion {
        offset: arena.next_offset as u64,
        len,
    };
    arena.next_offset = end;
    arena.reserved_bytes = arena
        .reserved_bytes
        .checked_add(len)
        .ok_or_else(|| io::Error::other("host_arena_reserve: reserved_bytes overflow"))?;
    Ok(Some(region))
}

/// Release a previously reserved region back to the same host arena.
///
/// # Safety
/// The caller must ensure `handle` points to the live arena that originally
/// produced `region`, and that `region` is not released more than once.
pub unsafe fn host_arena_release(
    handle: *mut KvHostArenaHandle,
    region: KvHostArenaRegion,
) -> io::Result<()> {
    if handle.is_null() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "host_arena_release: null handle",
        ));
    }
    let end = region_end(region).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "host_arena_release: region offset/len overflows",
        )
    })?;
    // Safety: caller contract.
    let arena = unsafe { &mut *(handle.cast::<KvHostArena>()) };
    if region.len == 0 || end > arena.capacity_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "host_arena_release: region out of bounds or empty",
        ));
    }
    arena.reserved_bytes = arena
        .reserved_bytes
        .checked_sub(region.len)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "host_arena_release: reserved_bytes underflow",
            )
        })?;
    if end == arena.next_offset {
        arena.next_offset = region.offset as usize;
        host_arena_rewind_tail(arena);
        return Ok(());
    }
    arena.free_list.push(region);
    Ok(())
}

/// Reset a live host arena to its empty state.
///
/// # Safety
/// The caller must ensure `handle` points to a live arena created by
/// [`host_arena_create`] and that no outstanding borrowers continue using
/// regions after the reset.
pub unsafe fn host_arena_reset(handle: *mut KvHostArenaHandle) -> io::Result<()> {
    if handle.is_null() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "host_arena_reset: null handle",
        ));
    }
    // Safety: caller contract.
    let arena = unsafe { &mut *(handle.cast::<KvHostArena>()) };
    arena.next_offset = 0;
    arena.reserved_bytes = 0;
    arena.free_list.clear();
    Ok(())
}

impl KvMmapDescriptor {
    pub fn path_buf(&self) -> io::Result<PathBuf> {
        if self.path_len > self.path.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "kv mmap descriptor path_len exceeds capacity",
            ));
        }
        Ok(PathBuf::from(std::ffi::OsString::from_vec(
            self.path[..self.path_len].to_vec(),
        )))
    }
}

impl KvSharedMemoryDescriptor {
    pub fn name(&self) -> io::Result<&str> {
        if self.name_len > self.name.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "kv shared memory descriptor name_len exceeds capacity",
            ));
        }
        std::str::from_utf8(&self.name[..self.name_len]).map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("kv shared memory descriptor contains invalid utf8: {err}"),
            )
        })
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }
}

/// Owning guard over a heap-allocated payload buffer. Drops via the Rust
/// allocator. The bytes are valid for as long as the guard lives; the caller
/// must not retain a slice past `Drop`.
///
/// Storage is a `Box<[u8]>`. The struct keeps `Send + Sync` and the same
/// public surface (`as_slice`, `len`, `is_empty`, `Deref<Target = [u8]>`)
/// it had under the Zig-FFI substrate, so call sites in
/// `infer/src/kv_tier/transport/disk.rs` continue to compile unchanged.
pub struct KvNativeOwnedBytes {
    data: Box<[u8]>,
}

impl KvNativeOwnedBytes {
    pub(crate) fn from_vec(bytes: Vec<u8>) -> Self {
        Self {
            data: bytes.into_boxed_slice(),
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl std::ops::Deref for KvNativeOwnedBytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

const WAL_MAGIC: &[u8; 8] = b"KVWAL001";

fn decode_wal_records(bytes: &[u8]) -> io::Result<Vec<KvWalRecord>> {
    if bytes.is_empty() {
        return Ok(Vec::new());
    }
    if bytes.len() < WAL_MAGIC.len() || &bytes[..WAL_MAGIC.len()] != WAL_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "kv wal replay: invalid magic",
        ));
    }

    let mut cursor = WAL_MAGIC.len();
    let mut records = Vec::new();
    while cursor < bytes.len() {
        if cursor + 9 > bytes.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "kv wal replay: truncated record header",
            ));
        }
        let kind = bytes[cursor];
        cursor += 1;
        let mut key_len_buf = [0_u8; 4];
        key_len_buf.copy_from_slice(&bytes[cursor..cursor + 4]);
        let key_len = u32::from_le_bytes(key_len_buf) as usize;
        cursor += 4;
        let mut value_len_buf = [0_u8; 4];
        value_len_buf.copy_from_slice(&bytes[cursor..cursor + 4]);
        let value_len = u32::from_le_bytes(value_len_buf) as usize;
        cursor += 4;
        if cursor + key_len + value_len > bytes.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "kv wal replay: truncated record payload",
            ));
        }
        let key = bytes[cursor..cursor + key_len].to_vec();
        cursor += key_len;
        let value = bytes[cursor..cursor + value_len].to_vec();
        cursor += value_len;
        records.push(KvWalRecord { kind, key, value });
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hint::black_box;
    use std::time::Instant;
    use tempfile::tempdir;

    struct TestHostArena {
        handle: *mut KvHostArenaHandle,
        base_ptr: *mut u8,
    }

    impl TestHostArena {
        fn new(capacity_bytes: usize) -> Self {
            let (handle, base_ptr) = host_arena_create(capacity_bytes).unwrap();
            assert!(!handle.is_null());
            assert!(!base_ptr.is_null());
            Self { handle, base_ptr }
        }

        fn reserve(&self, len: usize) -> KvHostArenaRegion {
            unsafe { host_arena_reserve(self.handle, len) }
                .unwrap()
                .unwrap()
        }

        fn release(&self, region: KvHostArenaRegion) {
            unsafe {
                host_arena_release(self.handle, region).unwrap();
            }
        }

        fn reset(&self) {
            unsafe {
                host_arena_reset(self.handle).unwrap();
            }
        }

        fn reserved_bytes(&self) -> usize {
            unsafe { host_arena_reserved_bytes(self.handle.cast_const()) }.unwrap()
        }
    }

    impl Drop for TestHostArena {
        fn drop(&mut self) {
            if !self.handle.is_null() {
                unsafe {
                    host_arena_destroy(self.handle).unwrap();
                }
            }
        }
    }

    #[test]
    fn block_path_is_hex_named() {
        let dir = tempdir().unwrap();
        let path = block_path(dir.path(), [0xAB; 16]).unwrap();
        assert_eq!(
            path.file_name().and_then(std::ffi::OsStr::to_str),
            Some("abababababababababababababababab.kv")
        );
    }

    #[test]
    fn wal_append_and_replay_roundtrip() {
        let dir = tempdir().unwrap();
        let wal = dir.path().join("kv.wal");
        wal_append(&wal, 1, b"alpha", b"payload-a").unwrap();
        wal_append(&wal, 2, b"beta", b"payload-b").unwrap();

        let records = wal_replay(&wal).unwrap();
        assert_eq!(
            records,
            vec![
                KvWalRecord {
                    kind: 1,
                    key: b"alpha".to_vec(),
                    value: b"payload-a".to_vec(),
                },
                KvWalRecord {
                    kind: 2,
                    key: b"beta".to_vec(),
                    value: b"payload-b".to_vec(),
                },
            ]
        );
    }

    #[test]
    fn wal_replay_rejects_truncated_record() {
        let dir = tempdir().unwrap();
        let wal = dir.path().join("corrupt.wal");
        std::fs::write(&wal, b"KVWAL001\x01\x03\x00\x00\x00").unwrap();
        let err = wal_replay(&wal).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn mmap_roundtrip() {
        let dir = tempdir().unwrap();
        let desc = mmap_create(&dir.path().join("segment.bin"), 64).unwrap();
        mmap_write(&desc, 4, b"hello").unwrap();
        assert_eq!(mmap_read(&desc, 4, 5).unwrap(), b"hello");
        assert_eq!(desc.path_buf().unwrap(), dir.path().join("segment.bin"));
    }

    #[test]
    fn shm_roundtrip() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let name = format!("/akv{:x}", nonce ^ u128::from(std::process::id()));
        let desc = shm_create(&name, 64).unwrap();
        shm_write(&desc, 8, b"shared").unwrap();
        assert_eq!(shm_read(&desc, 8, 6).unwrap(), b"shared");
        assert_eq!(desc.name().unwrap(), name);
        assert_ne!(desc.generation(), 0);
        shm_unlink(&desc).unwrap();
    }

    #[test]
    fn shm_stale_descriptor_is_rejected_after_recreate() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let name = format!("/akv{:x}", nonce ^ u128::from(std::process::id()));

        let stale = shm_create(&name, 64).unwrap();
        shm_write(&stale, 0, b"alpha").unwrap();
        shm_unlink(&stale).unwrap();

        let fresh = shm_create(&name, 64).unwrap();
        assert_ne!(stale.generation(), fresh.generation());
        shm_write(&fresh, 0, b"bravo").unwrap();

        let stale_read = shm_read(&stale, 0, 5).unwrap_err();
        assert_eq!(stale_read.kind(), io::ErrorKind::InvalidData);

        let stale_unlink = shm_unlink(&stale).unwrap_err();
        assert_eq!(stale_unlink.kind(), io::ErrorKind::InvalidData);

        assert_eq!(shm_read(&fresh, 0, 5).unwrap(), b"bravo");
        shm_unlink(&fresh).unwrap();
    }

    #[test]
    fn host_arena_reuses_released_regions() {
        let arena = TestHostArena::new(256);
        assert!(!arena.base_ptr.is_null());

        let a = arena.reserve(64);
        let b = arena.reserve(32);
        assert_eq!(a.offset, 0);
        assert_eq!(b.offset, 64);
        assert_eq!(arena.reserved_bytes(), 96);

        arena.release(a);
        let c = arena.reserve(64);
        assert_eq!(c.offset, 0);
        assert_eq!(arena.reserved_bytes(), 96);
    }

    #[test]
    fn host_arena_reset_rewinds_allocator() {
        let arena = TestHostArena::new(128);
        let _ = arena.reserve(96);
        arena.reset();
        let region = arena.reserve(96);
        assert_eq!(region.offset, 0);
        assert_eq!(arena.reserved_bytes(), 96);
    }

    #[test]
    fn host_arena_reverse_release_rewinds_tail_capacity() {
        let arena = TestHostArena::new(192);
        let a = arena.reserve(64);
        let b = arena.reserve(64);
        let c = arena.reserve(64);
        assert_eq!(a.offset, 0);
        assert_eq!(b.offset, 64);
        assert_eq!(c.offset, 128);
        assert_eq!(arena.reserved_bytes(), 192);

        arena.release(c);
        assert_eq!(arena.reserved_bytes(), 128);
        arena.release(b);
        assert_eq!(arena.reserved_bytes(), 64);
        arena.release(a);
        assert_eq!(arena.reserved_bytes(), 0);

        let whole = arena.reserve(192);
        assert_eq!(whole.offset, 0);
        assert_eq!(whole.len, 192);
        assert_eq!(arena.reserved_bytes(), 192);
    }

    #[test]
    fn host_arena_tail_release_collapses_adjacent_free_list_regions() {
        let arena = TestHostArena::new(256);
        let a = arena.reserve(64);
        let b = arena.reserve(64);
        let c = arena.reserve(64);
        let d = arena.reserve(64);
        assert_eq!(a.offset, 0);
        assert_eq!(b.offset, 64);
        assert_eq!(c.offset, 128);
        assert_eq!(d.offset, 192);
        assert_eq!(arena.reserved_bytes(), 256);

        arena.release(b);
        assert_eq!(arena.reserved_bytes(), 192);
        arena.release(c);
        assert_eq!(arena.reserved_bytes(), 128);

        arena.release(d);
        assert_eq!(arena.reserved_bytes(), 64);

        let stitched_tail = arena.reserve(192);
        assert_eq!(stitched_tail.offset, 64);
        assert_eq!(stitched_tail.len, 192);
        assert_eq!(arena.reserved_bytes(), 256);
    }

    #[test]
    fn read_block_owned_roundtrips_payload_via_zig_owned_buffer() {
        let dir = tempdir().unwrap();
        let fp = [0xC7u8; 16];
        let payload = b"payload-bytes-for-owned-roundtrip-test";
        write_block_atomic(dir.path(), fp, payload).unwrap();

        // The owning guard should expose the same bytes as `read_block` and
        // free the Zig-allocated buffer on Drop without panicking.
        let owned = read_block_owned(dir.path(), fp).unwrap();
        assert_eq!(owned.as_slice(), payload);
        assert_eq!(owned.len(), payload.len());
        assert!(!owned.is_empty());
        let slice: &[u8] = &owned;
        assert_eq!(slice, payload);
        drop(owned);

        // After dropping the guard, a fresh `read_block` call must still
        // succeed against the same on-disk file (verifies guard owns the
        // buffer and doesn't disturb the underlying storage).
        let again = read_block(dir.path(), fp).unwrap();
        assert_eq!(again, payload);
    }

    #[test]
    fn read_block_owned_returns_empty_for_empty_payload() {
        let dir = tempdir().unwrap();
        let fp = [0x9Fu8; 16];
        write_block_atomic(dir.path(), fp, b"").unwrap();

        let owned = read_block_owned(dir.path(), fp).unwrap();
        assert!(owned.is_empty());
        assert_eq!(owned.len(), 0);
        assert_eq!(owned.as_slice(), &[] as &[u8]);
        let slice: &[u8] = &owned;
        assert!(slice.is_empty());
    }

    #[test]
    #[ignore = "microbench: cargo test -p kv-native-sys host_arena_bench --release -- --ignored --nocapture"]
    fn host_arena_bench_reserved_bytes_fragmented() {
        const REGION_LEN: usize = 64;
        const REGIONS: usize = 8_192;
        const QUERIES: usize = 5_000_000;
        const WARMUP_QUERIES: usize = 100_000;

        let arena = TestHostArena::new(REGION_LEN * REGIONS);
        let mut regions = Vec::with_capacity(REGIONS);
        for _ in 0..REGIONS {
            regions.push(arena.reserve(REGION_LEN));
        }
        for region in regions.iter().step_by(2) {
            arena.release(*region);
        }

        let expected_reserved = REGION_LEN * (REGIONS / 2);
        let mut checksum = 0usize;
        for _ in 0..WARMUP_QUERIES {
            let reserved = arena.reserved_bytes();
            assert_eq!(reserved, expected_reserved);
            checksum = checksum.wrapping_add(black_box(reserved));
        }

        let start = Instant::now();
        for _ in 0..QUERIES {
            let reserved = arena.reserved_bytes();
            assert_eq!(reserved, expected_reserved);
            checksum = checksum.wrapping_add(black_box(reserved));
        }
        let elapsed = start.elapsed();
        let ns_per_query = elapsed.as_secs_f64() * 1_000_000_000.0 / QUERIES as f64;

        eprintln!(
            "host_arena_bench_reserved_bytes_fragmented regions={REGIONS} holes={} expected_reserved={} queries={QUERIES} total_ns={} ns_per_query={ns_per_query:.2} checksum={checksum}",
            REGIONS / 2,
            expected_reserved,
            elapsed.as_nanos(),
        );
    }
}
