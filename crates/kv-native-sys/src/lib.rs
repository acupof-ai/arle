use std::ffi::c_int;
use std::io;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::ffi::OsStringExt;
use std::path::Path;
use std::path::PathBuf;

pub const KV_MMAP_PATH_CAP: usize = 512;
pub const KV_SHM_NAME_CAP: usize = 128;

#[repr(C)]
struct KvNativeBuffer {
    data: *mut u8,
    len: usize,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum KvNativeStatus {
    Ok = 0,
    InvalidInput = 1,
    InvalidData = 2,
    NotFound = 3,
    Io = 4,
    Oom = 5,
}

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

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct KvWalRecord {
    pub kind: u8,
    pub key: Vec<u8>,
    pub value: Vec<u8>,
}

unsafe extern "C" {
    fn kv_native_write_file(
        path_ptr: *const u8,
        path_len: usize,
        bytes_ptr: *const u8,
        bytes_len: usize,
    ) -> c_int;
    fn kv_native_write_file_atomic(
        path_ptr: *const u8,
        path_len: usize,
        bytes_ptr: *const u8,
        bytes_len: usize,
    ) -> c_int;
    fn kv_native_read_file(path_ptr: *const u8, path_len: usize, out: *mut KvNativeBuffer)
        -> c_int;
    fn kv_native_remove_file(path_ptr: *const u8, path_len: usize, ignore_not_found: bool)
        -> c_int;
    fn kv_native_block_path(
        root_ptr: *const u8,
        root_len: usize,
        fingerprint_ptr: *const u8,
        fingerprint_len: usize,
        out: *mut KvNativeBuffer,
    ) -> c_int;
    fn kv_native_write_block_atomic(
        root_ptr: *const u8,
        root_len: usize,
        fingerprint_ptr: *const u8,
        fingerprint_len: usize,
        bytes_ptr: *const u8,
        bytes_len: usize,
    ) -> c_int;
    fn kv_native_read_block(
        root_ptr: *const u8,
        root_len: usize,
        fingerprint_ptr: *const u8,
        fingerprint_len: usize,
        out: *mut KvNativeBuffer,
    ) -> c_int;
    fn kv_native_remove_block(
        root_ptr: *const u8,
        root_len: usize,
        fingerprint_ptr: *const u8,
        fingerprint_len: usize,
        ignore_not_found: bool,
    ) -> c_int;
    fn kv_native_wal_append(
        path_ptr: *const u8,
        path_len: usize,
        kind: u8,
        key_ptr: *const u8,
        key_len: usize,
        value_ptr: *const u8,
        value_len: usize,
    ) -> c_int;
    fn kv_native_wal_replay(
        path_ptr: *const u8,
        path_len: usize,
        out: *mut KvNativeBuffer,
    ) -> c_int;
    fn kv_native_mmap_create(
        path_ptr: *const u8,
        path_len: usize,
        len: usize,
        out: *mut KvMmapDescriptor,
    ) -> c_int;
    fn kv_native_mmap_write(
        desc: *const KvMmapDescriptor,
        offset: usize,
        bytes_ptr: *const u8,
        bytes_len: usize,
    ) -> c_int;
    fn kv_native_mmap_read(
        desc: *const KvMmapDescriptor,
        offset: usize,
        bytes_len: usize,
        out: *mut KvNativeBuffer,
    ) -> c_int;
    fn kv_native_shm_create(
        name_ptr: *const u8,
        name_len: usize,
        len: usize,
        out: *mut KvSharedMemoryDescriptor,
    ) -> c_int;
    fn kv_native_shm_write(
        desc: *const KvSharedMemoryDescriptor,
        offset: usize,
        bytes_ptr: *const u8,
        bytes_len: usize,
    ) -> c_int;
    fn kv_native_shm_read(
        desc: *const KvSharedMemoryDescriptor,
        offset: usize,
        bytes_len: usize,
        out: *mut KvNativeBuffer,
    ) -> c_int;
    fn kv_native_shm_unlink(desc: *const KvSharedMemoryDescriptor) -> c_int;
    fn kv_native_buffer_free(data: *mut u8);
}

fn path_bytes(path: &Path) -> &[u8] {
    path.as_os_str().as_bytes()
}

fn fingerprint_bytes(fingerprint: [u8; 16]) -> [u8; 16] {
    fingerprint
}

fn status_from_code(code: c_int) -> Option<KvNativeStatus> {
    match code {
        0 => Some(KvNativeStatus::Ok),
        1 => Some(KvNativeStatus::InvalidInput),
        2 => Some(KvNativeStatus::InvalidData),
        3 => Some(KvNativeStatus::NotFound),
        4 => Some(KvNativeStatus::Io),
        5 => Some(KvNativeStatus::Oom),
        _ => None,
    }
}

fn status_to_error(status: c_int, context: &'static str) -> io::Error {
    match status_from_code(status) {
        Some(KvNativeStatus::InvalidInput) => io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{context}: invalid input"),
        ),
        Some(KvNativeStatus::InvalidData) => io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{context}: invalid data"),
        ),
        Some(KvNativeStatus::NotFound) => io::Error::new(
            io::ErrorKind::NotFound,
            format!("{context}: file not found"),
        ),
        Some(KvNativeStatus::Io) => io::Error::other(format!("{context}: native file I/O failure")),
        Some(KvNativeStatus::Oom) => {
            io::Error::other(format!("{context}: native allocator exhausted"))
        }
        Some(KvNativeStatus::Ok) => unreachable!("Ok should not be converted into io::Error"),
        None => io::Error::other(format!("{context}: unknown native status {status}")),
    }
}

fn status_to_result(status: c_int, context: &'static str) -> io::Result<()> {
    match status_from_code(status) {
        Some(KvNativeStatus::Ok) => Ok(()),
        _ => Err(status_to_error(status, context)),
    }
}

pub fn write_file(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let path = path_bytes(path);
    let status =
        unsafe { kv_native_write_file(path.as_ptr(), path.len(), bytes.as_ptr(), bytes.len()) };
    status_to_result(status, "kv_native_write_file")
}

pub fn write_file_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let path = path_bytes(path);
    let status = unsafe {
        kv_native_write_file_atomic(path.as_ptr(), path.len(), bytes.as_ptr(), bytes.len())
    };
    status_to_result(status, "kv_native_write_file_atomic")
}

pub fn read_file(path: &Path) -> io::Result<Vec<u8>> {
    let path = path_bytes(path);
    read_buffer(
        |out| unsafe { kv_native_read_file(path.as_ptr(), path.len(), out) },
        "kv_native_read_file",
    )
}

pub fn remove_file(path: &Path, ignore_not_found: bool) -> io::Result<()> {
    let path = path_bytes(path);
    let status = unsafe { kv_native_remove_file(path.as_ptr(), path.len(), ignore_not_found) };
    status_to_result(status, "kv_native_remove_file")
}

pub fn block_path(root: &Path, fingerprint: [u8; 16]) -> io::Result<PathBuf> {
    let root = path_bytes(root);
    let fingerprint = fingerprint_bytes(fingerprint);
    let bytes = read_buffer(
        |out| unsafe {
            kv_native_block_path(
                root.as_ptr(),
                root.len(),
                fingerprint.as_ptr(),
                fingerprint.len(),
                out,
            )
        },
        "kv_native_block_path",
    )?;
    Ok(PathBuf::from(std::ffi::OsString::from_vec(bytes)))
}

pub fn write_block_atomic(root: &Path, fingerprint: [u8; 16], bytes: &[u8]) -> io::Result<()> {
    let root = path_bytes(root);
    let fingerprint = fingerprint_bytes(fingerprint);
    let status = unsafe {
        kv_native_write_block_atomic(
            root.as_ptr(),
            root.len(),
            fingerprint.as_ptr(),
            fingerprint.len(),
            bytes.as_ptr(),
            bytes.len(),
        )
    };
    status_to_result(status, "kv_native_write_block_atomic")
}

pub fn read_block(root: &Path, fingerprint: [u8; 16]) -> io::Result<Vec<u8>> {
    let root = path_bytes(root);
    let fingerprint = fingerprint_bytes(fingerprint);
    read_buffer(
        |out| unsafe {
            kv_native_read_block(
                root.as_ptr(),
                root.len(),
                fingerprint.as_ptr(),
                fingerprint.len(),
                out,
            )
        },
        "kv_native_read_block",
    )
}

pub fn remove_block(root: &Path, fingerprint: [u8; 16], ignore_not_found: bool) -> io::Result<()> {
    let root = path_bytes(root);
    let fingerprint = fingerprint_bytes(fingerprint);
    let status = unsafe {
        kv_native_remove_block(
            root.as_ptr(),
            root.len(),
            fingerprint.as_ptr(),
            fingerprint.len(),
            ignore_not_found,
        )
    };
    status_to_result(status, "kv_native_remove_block")
}

pub fn wal_append(path: &Path, kind: u8, key: &[u8], value: &[u8]) -> io::Result<()> {
    let path = path_bytes(path);
    let status = unsafe {
        kv_native_wal_append(
            path.as_ptr(),
            path.len(),
            kind,
            key.as_ptr(),
            key.len(),
            value.as_ptr(),
            value.len(),
        )
    };
    status_to_result(status, "kv_native_wal_append")
}

pub fn wal_replay(path: &Path) -> io::Result<Vec<KvWalRecord>> {
    let path = path_bytes(path);
    let bytes = match read_buffer(
        |out| unsafe { kv_native_wal_replay(path.as_ptr(), path.len(), out) },
        "kv_native_wal_replay",
    ) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(err),
    };
    decode_wal_records(&bytes)
}

pub fn mmap_create(path: &Path, len: usize) -> io::Result<KvMmapDescriptor> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let path = path_bytes(path);
    let mut out = KvMmapDescriptor {
        len: 0,
        path_len: 0,
        path: [0; KV_MMAP_PATH_CAP],
    };
    let status = unsafe { kv_native_mmap_create(path.as_ptr(), path.len(), len, &mut out) };
    status_to_result(status, "kv_native_mmap_create")?;
    Ok(out)
}

pub fn mmap_write(desc: &KvMmapDescriptor, offset: usize, bytes: &[u8]) -> io::Result<()> {
    let status = unsafe { kv_native_mmap_write(desc, offset, bytes.as_ptr(), bytes.len()) };
    status_to_result(status, "kv_native_mmap_write")
}

pub fn mmap_read(desc: &KvMmapDescriptor, offset: usize, len: usize) -> io::Result<Vec<u8>> {
    read_buffer(
        |out| unsafe { kv_native_mmap_read(desc, offset, len, out) },
        "kv_native_mmap_read",
    )
}

pub fn shm_create(name: &str, len: usize) -> io::Result<KvSharedMemoryDescriptor> {
    let mut out = KvSharedMemoryDescriptor {
        len: 0,
        generation: 0,
        name_len: 0,
        name: [0; KV_SHM_NAME_CAP],
    };
    let status = unsafe { kv_native_shm_create(name.as_ptr(), name.len(), len, &mut out) };
    status_to_result(status, "kv_native_shm_create")?;
    Ok(out)
}

pub fn shm_write(desc: &KvSharedMemoryDescriptor, offset: usize, bytes: &[u8]) -> io::Result<()> {
    let status = unsafe { kv_native_shm_write(desc, offset, bytes.as_ptr(), bytes.len()) };
    status_to_result(status, "kv_native_shm_write")
}

pub fn shm_read(desc: &KvSharedMemoryDescriptor, offset: usize, len: usize) -> io::Result<Vec<u8>> {
    read_buffer(
        |out| unsafe { kv_native_shm_read(desc, offset, len, out) },
        "kv_native_shm_read",
    )
}

pub fn shm_unlink(desc: &KvSharedMemoryDescriptor) -> io::Result<()> {
    let status = unsafe { kv_native_shm_unlink(desc) };
    status_to_result(status, "kv_native_shm_unlink")
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

fn read_buffer(
    call: impl FnOnce(*mut KvNativeBuffer) -> c_int,
    context: &'static str,
) -> io::Result<Vec<u8>> {
    let mut out = KvNativeBuffer {
        data: std::ptr::null_mut(),
        len: 0,
    };
    let status = call(&mut out);
    if status_from_code(status) != Some(KvNativeStatus::Ok) {
        return Err(status_to_error(status, context));
    }
    if out.len == 0 {
        if !out.data.is_null() {
            unsafe { kv_native_buffer_free(out.data) };
        }
        return Ok(Vec::new());
    }
    if out.data.is_null() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{context}: native layer returned null data for non-empty buffer"),
        ));
    }
    let bytes = unsafe { std::slice::from_raw_parts(out.data.cast_const(), out.len) }.to_vec();
    unsafe { kv_native_buffer_free(out.data) };
    Ok(bytes)
}

fn decode_wal_records(bytes: &[u8]) -> io::Result<Vec<KvWalRecord>> {
    const WAL_MAGIC: &[u8; 8] = b"KVWAL001";

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
    use tempfile::tempdir;

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
}
