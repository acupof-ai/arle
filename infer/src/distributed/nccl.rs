//! NCCL group smoke for the single-node multi-GPU F0 foundation.

use std::ffi::{CStr, CString, c_char, c_void};
use std::net::{SocketAddr, ToSocketAddrs};
use std::sync::{Arc, OnceLock};

use anyhow::{Context, Result, anyhow, bail};
use cuda_kernels::ffi::nccl::{
    ncclComm_t, ncclDataType_t, ncclRedOp_t, ncclResult_t, ncclUniqueId,
};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;

use super::{RendezvousClient, RendezvousServer, UNIQUE_ID_BYTES};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NcclInitMethod {
    TcpStore(SocketAddr),
    EnvBootstrap,
}

#[derive(Debug)]
pub struct NcclGroup {
    pub rank: usize,
    pub world_size: usize,
    pub comm: NcclComm,
    stream: Arc<CudaStream>,
}

// SAFETY: ARLE creates one communicator per rank thread and only uses it to
// enqueue NCCL calls on the communicator-owned CUDA stream. NCCL communicators
// are safe to move/share as opaque handles when callers preserve that stream
// ownership discipline; synchronization is handled by CUDA stream ordering.
unsafe impl Send for NcclGroup {}
unsafe impl Sync for NcclGroup {}

impl NcclGroup {
    pub fn new(rank: usize, world_size: usize, init_method: NcclInitMethod) -> Result<Self> {
        if world_size == 0 {
            bail!("NCCL group world_size must be >= 1");
        }
        if rank >= world_size {
            bail!("NCCL rank {rank} must be < world_size {world_size}");
        }

        let ctx = CudaContext::new(rank)
            .with_context(|| format!("failed to create CUDA context for rank/device {rank}"))?;
        let stream = ctx.default_stream();
        Self::new_on_stream(rank, world_size, init_method, stream)
    }

    pub fn new_on_stream(
        rank: usize,
        world_size: usize,
        init_method: NcclInitMethod,
        stream: Arc<CudaStream>,
    ) -> Result<Self> {
        if world_size == 0 {
            bail!("NCCL group world_size must be >= 1");
        }
        if rank >= world_size {
            bail!("NCCL rank {rank} must be < world_size {world_size}");
        }

        let id = exchange_unique_id(rank, world_size, init_method)?;
        let comm = NcclComm::init_rank(rank, world_size, id)
            .with_context(|| format!("ncclCommInitRank failed for rank {rank}/{world_size}"))?;

        Ok(Self {
            rank,
            world_size,
            comm,
            stream,
        })
    }

    pub fn all_reduce_smoke(&self, input: &[f32]) -> Result<Vec<f32>> {
        self.all_reduce_f32(input)
    }

    pub fn all_reduce_bf16_in_place(&self, buffer: &mut CudaSlice<bf16>) -> Result<()> {
        if buffer.len() == 0 {
            return Ok(());
        }
        let count = buffer.len();
        let (ptr, _record) = buffer.device_ptr_mut(&self.stream);
        self.comm.all_reduce(
            ptr as *const c_void,
            ptr as *mut c_void,
            count,
            ncclDataType_t::Bfloat16,
            &self.stream,
        )
    }

    pub fn all_reduce_f32(&self, input: &[f32]) -> Result<Vec<f32>> {
        let send = self
            .stream
            .clone_htod(input)
            .with_context(|| format!("rank {} H2D smoke input copy failed", self.rank))?;
        let mut recv = self
            .stream
            .alloc_zeros::<f32>(input.len())
            .with_context(|| format!("rank {} smoke output allocation failed", self.rank))?;

        {
            let (src, _record_src) = send.device_ptr(&self.stream);
            let (dst, _record_dst) = recv.device_ptr_mut(&self.stream);
            self.comm.all_reduce(
                src as *const c_void,
                dst as *mut c_void,
                input.len(),
                ncclDataType_t::Float32,
                &self.stream,
            )?;
        }
        self.stream
            .synchronize()
            .with_context(|| format!("rank {} stream sync after NCCL failed", self.rank))?;
        self.stream
            .clone_dtoh(&recv)
            .with_context(|| format!("rank {} D2H smoke output copy failed", self.rank))
    }

    pub fn all_gather_f32(&self, input: &[f32], per_rank_count: usize) -> Result<Vec<f32>> {
        if input.len() != per_rank_count {
            bail!(
                "NCCL all_gather rank {} input len {} must equal per-rank count {per_rank_count}",
                self.rank,
                input.len()
            );
        }
        let send = self
            .stream
            .clone_htod(input)
            .with_context(|| format!("rank {} H2D all_gather input copy failed", self.rank))?;
        let mut recv = self
            .stream
            .alloc_zeros::<f32>(per_rank_count * self.world_size)
            .with_context(|| format!("rank {} all_gather output allocation failed", self.rank))?;

        {
            let (src, _record_src) = send.device_ptr(&self.stream);
            let (dst, _record_dst) = recv.device_ptr_mut(&self.stream);
            self.comm.all_gather(
                src as *const c_void,
                dst as *mut c_void,
                per_rank_count,
                ncclDataType_t::Float32,
                &self.stream,
            )?;
        }
        self.stream
            .synchronize()
            .with_context(|| format!("rank {} stream sync after all_gather failed", self.rank))?;
        self.stream
            .clone_dtoh(&recv)
            .with_context(|| format!("rank {} D2H all_gather output copy failed", self.rank))
    }

    pub fn broadcast_f32(&self, input: &[f32], count: usize, root_rank: usize) -> Result<Vec<f32>> {
        if root_rank >= self.world_size {
            bail!(
                "NCCL broadcast root {root_rank} must be < world_size {}",
                self.world_size
            );
        }
        if self.rank == root_rank && input.len() != count {
            bail!(
                "NCCL broadcast root rank {} input len {} must equal count {count}",
                self.rank,
                input.len()
            );
        }
        let mut recv = if self.rank == root_rank {
            self.stream
                .clone_htod(input)
                .with_context(|| format!("rank {} H2D broadcast input copy failed", self.rank))?
        } else {
            self.stream
                .alloc_zeros::<f32>(count)
                .with_context(|| format!("rank {} broadcast output allocation failed", self.rank))?
        };

        {
            let (ptr, _record) = recv.device_ptr_mut(&self.stream);
            self.comm.broadcast(
                ptr as *mut c_void,
                count,
                ncclDataType_t::Float32,
                root_rank,
                &self.stream,
            )?;
        }
        self.stream
            .synchronize()
            .with_context(|| format!("rank {} stream sync after broadcast failed", self.rank))?;
        self.stream
            .clone_dtoh(&recv)
            .with_context(|| format!("rank {} D2H broadcast output copy failed", self.rank))
    }
}

#[derive(Clone, Copy)]
struct Id {
    id: ncclUniqueId,
}

impl Id {
    fn new() -> Result<Self> {
        let mut id = ncclUniqueId { internal: [0; 128] };
        let api = nccl_api()?;
        check_nccl(
            unsafe { (api.get_unique_id)(&mut id) },
            "ncclGetUniqueId",
            api,
        )?;
        Ok(Self { id })
    }

    fn uninit(internal: [c_char; UNIQUE_ID_BYTES]) -> Self {
        Self {
            id: ncclUniqueId { internal },
        }
    }

    fn internal(&self) -> &[c_char; UNIQUE_ID_BYTES] {
        &self.id.internal
    }
}

pub struct NcclComm {
    raw: ncclComm_t,
}

impl std::fmt::Debug for NcclComm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NcclComm").field("raw", &self.raw).finish()
    }
}

impl NcclComm {
    fn init_rank(rank: usize, world_size: usize, id: Id) -> Result<Self> {
        let mut raw = std::ptr::null_mut();
        let api = nccl_api()?;
        check_nccl(
            unsafe { (api.comm_init_rank)(&mut raw, world_size as i32, id.id, rank as i32) },
            "ncclCommInitRank",
            api,
        )?;
        Ok(Self { raw })
    }

    fn all_reduce(
        &self,
        sendbuff: *const c_void,
        recvbuff: *mut c_void,
        count: usize,
        dtype: ncclDataType_t,
        stream: &CudaStream,
    ) -> Result<()> {
        let api = nccl_api()?;
        check_nccl(
            unsafe {
                (api.all_reduce)(
                    sendbuff,
                    recvbuff,
                    count,
                    dtype,
                    ncclRedOp_t::Sum,
                    self.raw,
                    stream.cu_stream() as *mut c_void,
                )
            },
            "ncclAllReduce",
            api,
        )
    }

    fn all_gather(
        &self,
        sendbuff: *const c_void,
        recvbuff: *mut c_void,
        sendcount: usize,
        dtype: ncclDataType_t,
        stream: &CudaStream,
    ) -> Result<()> {
        let api = nccl_api()?;
        check_nccl(
            unsafe {
                (api.all_gather)(
                    sendbuff,
                    recvbuff,
                    sendcount,
                    dtype,
                    self.raw,
                    stream.cu_stream() as *mut c_void,
                )
            },
            "ncclAllGather",
            api,
        )
    }

    fn broadcast(
        &self,
        buffer: *mut c_void,
        count: usize,
        dtype: ncclDataType_t,
        root: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let api = nccl_api()?;
        check_nccl(
            unsafe {
                (api.broadcast)(
                    buffer as *const c_void,
                    buffer,
                    count,
                    dtype,
                    root as i32,
                    self.raw,
                    stream.cu_stream() as *mut c_void,
                )
            },
            "ncclBroadcast",
            api,
        )
    }
}

impl Drop for NcclComm {
    fn drop(&mut self) {
        if !self.raw.is_null()
            && let Some(api) = NCCL_API.get()
        {
            unsafe {
                let _ = (api.comm_destroy)(self.raw);
            }
        }
    }
}

type NcclGetUniqueId = unsafe extern "C" fn(*mut ncclUniqueId) -> ncclResult_t;
type NcclCommInitRank =
    unsafe extern "C" fn(*mut ncclComm_t, i32, ncclUniqueId, i32) -> ncclResult_t;
type NcclCommDestroy = unsafe extern "C" fn(ncclComm_t) -> ncclResult_t;
type NcclAllReduce = unsafe extern "C" fn(
    *const c_void,
    *mut c_void,
    usize,
    ncclDataType_t,
    ncclRedOp_t,
    ncclComm_t,
    *mut c_void,
) -> ncclResult_t;
type NcclAllGather = unsafe extern "C" fn(
    *const c_void,
    *mut c_void,
    usize,
    ncclDataType_t,
    ncclComm_t,
    *mut c_void,
) -> ncclResult_t;
type NcclBroadcast = unsafe extern "C" fn(
    *const c_void,
    *mut c_void,
    usize,
    ncclDataType_t,
    i32,
    ncclComm_t,
    *mut c_void,
) -> ncclResult_t;
type NcclGetErrorString = unsafe extern "C" fn(ncclResult_t) -> *const c_char;

struct NcclApi {
    _handle: *mut c_void,
    get_unique_id: NcclGetUniqueId,
    comm_init_rank: NcclCommInitRank,
    comm_destroy: NcclCommDestroy,
    all_reduce: NcclAllReduce,
    all_gather: NcclAllGather,
    broadcast: NcclBroadcast,
    get_error_string: NcclGetErrorString,
}

// SAFETY: `NcclApi` is a process-lifetime dlopen handle plus immutable function
// pointers. The pointed-to library code is thread-safe per NCCL's API contract.
unsafe impl Send for NcclApi {}
unsafe impl Sync for NcclApi {}

static NCCL_API: OnceLock<NcclApi> = OnceLock::new();

fn nccl_api() -> Result<&'static NcclApi> {
    if let Some(api) = NCCL_API.get() {
        return Ok(api);
    }
    let api = unsafe { NcclApi::open()? };
    let _ = NCCL_API.set(api);
    NCCL_API
        .get()
        .ok_or_else(|| anyhow!("NCCL API initialization failed"))
}

impl NcclApi {
    unsafe fn open() -> Result<Self> {
        let handle = open_nccl_library()?;
        Ok(Self {
            _handle: handle,
            get_unique_id: unsafe { load_symbol(handle, b"ncclGetUniqueId\0")? },
            comm_init_rank: unsafe { load_symbol(handle, b"ncclCommInitRank\0")? },
            comm_destroy: unsafe { load_symbol(handle, b"ncclCommDestroy\0")? },
            all_reduce: unsafe { load_symbol(handle, b"ncclAllReduce\0")? },
            all_gather: unsafe { load_symbol(handle, b"ncclAllGather\0")? },
            broadcast: unsafe { load_symbol(handle, b"ncclBroadcast\0")? },
            get_error_string: unsafe { load_symbol(handle, b"ncclGetErrorString\0")? },
        })
    }
}

fn open_nccl_library() -> Result<*mut c_void> {
    let mut candidates = Vec::new();
    if let Ok(path) = std::env::var("ARLE_NCCL_LIBRARY") {
        candidates.push(path);
    }
    candidates.extend(["libnccl.so".to_string(), "libnccl.so.2".to_string()]);

    let mut errors = Vec::new();
    for name in candidates {
        let cname = CString::new(name.clone())
            .with_context(|| format!("invalid NCCL library path {name:?}"))?;
        let handle = unsafe { libc::dlopen(cname.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if !handle.is_null() {
            return Ok(handle);
        }
        errors.push(format!("{name}: {}", dlerror_string()));
    }
    bail!("failed to load NCCL library; tried {}", errors.join("; "))
}

unsafe fn load_symbol<T: Copy>(handle: *mut c_void, name: &'static [u8]) -> Result<T> {
    let ptr = unsafe { libc::dlsym(handle, name.as_ptr().cast()) };
    if ptr.is_null() {
        let display = CStr::from_bytes_until_nul(name)
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|_| "<invalid symbol>".to_string());
        bail!("failed to load NCCL symbol {display}: {}", dlerror_string());
    }
    Ok(unsafe { std::mem::transmute_copy(&ptr) })
}

fn dlerror_string() -> String {
    let err = unsafe { libc::dlerror() };
    if err.is_null() {
        "unknown dlopen/dlsym error".to_string()
    } else {
        unsafe { CStr::from_ptr(err) }
            .to_string_lossy()
            .into_owned()
    }
}

fn check_nccl(result: ncclResult_t, op: &str, api: &NcclApi) -> Result<()> {
    if result == ncclResult_t::Success {
        return Ok(());
    }
    let msg = unsafe {
        let ptr = (api.get_error_string)(result);
        if ptr.is_null() {
            "unknown NCCL error".into()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };
    bail!("{op} failed: {msg} ({result:?})")
}

fn exchange_unique_id(rank: usize, world_size: usize, init_method: NcclInitMethod) -> Result<Id> {
    let addr = match init_method {
        NcclInitMethod::TcpStore(addr) => addr,
        NcclInitMethod::EnvBootstrap => env_bootstrap_addr()?,
    };

    if rank == 0 {
        let id =
            Id::new().map_err(|err| anyhow!("rank 0 failed to create NCCL unique id: {err:?}"))?;
        let bytes = id_to_bytes(&id);
        let mut server = RendezvousServer::bind(addr, world_size)
            .with_context(|| format!("rank 0 failed to bind NCCL TCP store at {addr}"))?;
        server
            .rendezvous(&bytes)
            .context("rank 0 NCCL TCP-store rendezvous failed")?;
        Ok(id)
    } else {
        let mut client = RendezvousClient::connect(addr)
            .with_context(|| format!("rank {rank} failed to connect NCCL TCP store at {addr}"))?;
        let bytes = client
            .rendezvous()
            .with_context(|| format!("rank {rank} NCCL TCP-store rendezvous failed"))?;
        Ok(id_from_bytes(bytes))
    }
}

fn env_bootstrap_addr() -> Result<SocketAddr> {
    let host = std::env::var("MASTER_ADDR").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port = std::env::var("MASTER_PORT")
        .context("NCCL EnvBootstrap requires MASTER_PORT; set MASTER_ADDR optionally")?;
    let port = port
        .parse::<u16>()
        .with_context(|| format!("invalid MASTER_PORT: {port}"))?;
    (host.as_str(), port)
        .to_socket_addrs()
        .with_context(|| format!("failed to resolve MASTER_ADDR/MASTER_PORT: {host}:{port}"))?
        .next()
        .with_context(|| format!("MASTER_ADDR/MASTER_PORT resolved to zero addrs: {host}:{port}"))
}

fn id_to_bytes(id: &Id) -> [u8; UNIQUE_ID_BYTES] {
    std::array::from_fn(|idx| id.internal()[idx] as u8)
}

fn id_from_bytes(bytes: [u8; UNIQUE_ID_BYTES]) -> Id {
    let internal: [c_char; UNIQUE_ID_BYTES] = std::array::from_fn(|idx| bytes[idx] as c_char);
    Id::uninit(internal)
}
