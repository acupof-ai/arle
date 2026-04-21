use std::ffi::c_int;
use std::io;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::ffi::OsStringExt;
use std::path::Path;
use std::path::PathBuf;

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
