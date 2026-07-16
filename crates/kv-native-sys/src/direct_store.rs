use std::fs::{File, OpenOptions};
use std::io;
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::ptr::NonNull;
use std::time::Instant;

use io_uring::{IoUring, opcode, types};

pub(crate) const DIRECT_ALIGNMENT: usize = 4096;
const MAX_RETRIES: usize = 3;

pub(crate) struct DirectStore {
    file: File,
    ring: IoUring,
    slot_bytes: usize,
    stride_bytes: usize,
    num_slots: u32,
    free_list: Vec<u32>,
    queue_depth: usize,
}

pub(crate) struct BatchIo {
    pub(crate) submitted_bytes: u64,
    pub(crate) wait_ns: u64,
}

struct AlignedBuffer {
    ptr: NonNull<u8>,
    len: usize,
}

impl AlignedBuffer {
    fn zeroed(len: usize) -> io::Result<Self> {
        let mut ptr = std::ptr::null_mut();
        // SAFETY: posix_memalign initializes `ptr` on success; alignment and size
        // satisfy its contract. The allocation is released by Drop.
        let rc = unsafe { libc::posix_memalign(&mut ptr, DIRECT_ALIGNMENT, len) };
        if rc != 0 {
            return Err(io::Error::from_raw_os_error(rc));
        }
        let ptr = NonNull::new(ptr.cast()).ok_or_else(|| io::Error::other("null allocation"))?;
        // SAFETY: `ptr` owns `len` writable bytes returned above.
        unsafe { std::ptr::write_bytes(ptr.as_ptr(), 0, len) };
        Ok(Self { ptr, len })
    }

    fn as_slice(&self) -> &[u8] {
        // SAFETY: the allocation is live for `self` and contains `len` bytes.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from posix_memalign and is freed exactly once.
        unsafe { libc::free(self.ptr.as_ptr().cast()) };
    }
}

impl DirectStore {
    pub(crate) fn create(path: &Path, num_slots: usize, slot_bytes: usize) -> io::Result<Self> {
        let file = open(path, true)?;
        Self::from_file(file, num_slots, slot_bytes, true)
    }

    pub(crate) fn open(path: &Path, num_slots: usize, slot_bytes: usize) -> io::Result<Self> {
        let file = open(path, false)?;
        Self::from_file(file, num_slots, slot_bytes, false)
    }

    fn from_file(
        file: File,
        num_slots: usize,
        slot_bytes: usize,
        create: bool,
    ) -> io::Result<Self> {
        if slot_bytes == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "slot size must be > 0",
            ));
        }
        let stride_bytes = aligned_len(slot_bytes);
        let num_slots = u32::try_from(num_slots)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "num_slots exceeds u32"))?;
        if num_slots == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "num_slots must be > 0",
            ));
        }
        let total_bytes = u64::from(num_slots)
            .checked_mul(stride_bytes as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "total bytes overflow"))?;
        if create {
            file.set_len(total_bytes)?;
        } else if file.metadata()?.len() != total_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "direct store file size does not match configured slots",
            ));
        }
        let queue_depth = std::env::var("ARLE_KV_IO_URING_QD")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|depth| *depth > 0)
            .unwrap_or(8)
            .min(1024);
        let ring = IoUring::new(queue_depth as u32)?;
        let mut store = Self {
            file,
            ring,
            slot_bytes,
            stride_bytes,
            num_slots,
            free_list: (0..num_slots).collect(),
            queue_depth,
        };
        if create {
            let zero = vec![0; DIRECT_ALIGNMENT];
            store.write_slots(&[(0, zero.as_slice())])?;
            let (read, _) = store.read_slots(&[(0, DIRECT_ALIGNMENT)])?;
            if read[0].as_slice() != zero {
                return Err(io::Error::other("O_DIRECT probe read mismatch"));
            }
        } else {
            store.read_slots(&[(0, DIRECT_ALIGNMENT)])?;
        }
        Ok(store)
    }

    pub(crate) fn num_slots(&self) -> u32 {
        self.num_slots
    }

    pub(crate) fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }

    pub(crate) fn alloc_slot(&mut self) -> Option<u32> {
        self.free_list.pop()
    }

    pub(crate) fn free_slot(&mut self, slot: u32) {
        if !self.free_list.contains(&slot) {
            self.free_list.push(slot);
        }
    }

    pub(crate) fn reserve_indices(&mut self, indices: &[u32]) {
        self.free_list.retain(|index| !indices.contains(index));
    }

    pub(crate) fn write_slots(&mut self, writes: &[(u32, &[u8])]) -> io::Result<BatchIo> {
        let mut submitted_bytes = 0;
        let mut wait_ns = 0;
        for window in writes.chunks(self.queue_depth) {
            let mut buffers = Vec::with_capacity(window.len());
            for (_, bytes) in window {
                if bytes.len() > self.slot_bytes {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "direct write exceeds slot",
                    ));
                }
                let len = aligned_len(bytes.len());
                let mut buffer = AlignedBuffer::zeroed(len)?;
                // SAFETY: both slices are valid and non-overlapping; destination
                // is at least `bytes.len()` bytes by construction.
                unsafe {
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.as_mut_ptr(), bytes.len())
                };
                buffers.push(buffer);
            }
            let batch_bytes = buffers.iter().map(|buffer| buffer.len as u64).sum::<u64>();
            for attempt in 0..=MAX_RETRIES {
                for (index, ((slot, _), buffer)) in window.iter().zip(&buffers).enumerate() {
                    let entry = opcode::Write::new(
                        types::Fd(self.file.as_raw_fd()),
                        buffer.ptr.as_ptr(),
                        buffer.len as u32,
                    )
                    .offset(u64::from(*slot) * self.stride_bytes as u64)
                    .build()
                    .user_data(index as u64);
                    // SAFETY: each buffer and the file descriptor remain live until
                    // every completion is drained below.
                    unsafe { self.ring.submission().push(&entry) }
                        .map_err(|_| io::Error::other("io_uring submission queue full"))?;
                }
                let started = Instant::now();
                self.ring.submit_and_wait(window.len())?;
                wait_ns += elapsed_ns(started);
                submitted_bytes += batch_bytes;
                if let Err(err) = validate_completions(&mut self.ring, &buffers) {
                    if err.raw_os_error() == Some(libc::EAGAIN) && attempt < MAX_RETRIES {
                        continue;
                    }
                    return Err(err);
                }
                break;
            }
        }
        Ok(BatchIo {
            submitted_bytes,
            wait_ns,
        })
    }

    pub(crate) fn read_slots(
        &mut self,
        reads: &[(u32, usize)],
    ) -> io::Result<(Vec<Vec<u8>>, BatchIo)> {
        let mut values = Vec::with_capacity(reads.len());
        let mut submitted_bytes = 0;
        let mut wait_ns = 0;
        for window in reads.chunks(self.queue_depth) {
            let mut buffers = window
                .iter()
                .map(|(_, len)| {
                    if *len > self.slot_bytes {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "direct read exceeds slot",
                        ));
                    }
                    AlignedBuffer::zeroed(aligned_len(*len))
                })
                .collect::<io::Result<Vec<_>>>()?;
            let batch_bytes = buffers.iter().map(|buffer| buffer.len as u64).sum::<u64>();
            for attempt in 0..=MAX_RETRIES {
                for (index, ((slot, _), buffer)) in window.iter().zip(&mut buffers).enumerate() {
                    let entry = opcode::Read::new(
                        types::Fd(self.file.as_raw_fd()),
                        buffer.as_mut_ptr(),
                        buffer.len as u32,
                    )
                    .offset(u64::from(*slot) * self.stride_bytes as u64)
                    .build()
                    .user_data(index as u64);
                    // SAFETY: each buffer and the file descriptor remain live until
                    // every completion is drained below.
                    unsafe { self.ring.submission().push(&entry) }
                        .map_err(|_| io::Error::other("io_uring submission queue full"))?;
                }
                let started = Instant::now();
                self.ring.submit_and_wait(window.len())?;
                wait_ns += elapsed_ns(started);
                submitted_bytes += batch_bytes;
                if let Err(err) = validate_completions(&mut self.ring, &buffers) {
                    if err.raw_os_error() == Some(libc::EAGAIN) && attempt < MAX_RETRIES {
                        continue;
                    }
                    return Err(err);
                }
                break;
            }
            values.extend(
                buffers
                    .iter()
                    .zip(window)
                    .map(|(buffer, (_, len))| buffer.as_slice()[..*len].to_vec()),
            );
        }
        Ok((
            values,
            BatchIo {
                submitted_bytes,
                wait_ns,
            },
        ))
    }
}

fn open(path: &Path, create: bool) -> io::Result<File> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(create)
        .truncate(create)
        .mode(0o644)
        .custom_flags(libc::O_DIRECT)
        .open(path)
}

fn aligned_len(len: usize) -> usize {
    len.max(1).div_ceil(DIRECT_ALIGNMENT) * DIRECT_ALIGNMENT
}

fn elapsed_ns(started: Instant) -> u64 {
    started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
}

fn validate_completions(ring: &mut IoUring, buffers: &[AlignedBuffer]) -> io::Result<()> {
    let mut completed = 0;
    let mut failure = None;
    for completion in ring.completion() {
        let index = completion.user_data() as usize;
        let Some(expected) = buffers.get(index).map(|buffer| buffer.len) else {
            failure.get_or_insert_with(|| io::Error::other("io_uring returned an unknown request"));
            completed += 1;
            continue;
        };
        let result = completion.result();
        if result < 0 {
            failure.get_or_insert_with(|| io::Error::from_raw_os_error(-result));
        } else if result as usize != expected {
            failure.get_or_insert_with(|| {
                io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("io_uring completed {result} B, expected {expected} B"),
                )
            });
        }
        completed += 1;
    }
    if completed != buffers.len() {
        return Err(io::Error::other("io_uring completion count mismatch"));
    }
    failure.map_or(Ok(()), Err)
}
