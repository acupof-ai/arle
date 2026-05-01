//! NCCL group smoke for the single-node multi-GPU F0 foundation.

use std::ffi::c_char;
use std::net::{SocketAddr, ToSocketAddrs};

use anyhow::{Context, Result, anyhow, bail};
use cudarc::driver::CudaContext;
use cudarc::nccl::safe::{Comm, Id, ReduceOp};

use super::{RendezvousClient, RendezvousServer, UNIQUE_ID_BYTES};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NcclInitMethod {
    TcpStore(SocketAddr),
    EnvBootstrap,
}

pub struct NcclGroup {
    pub rank: usize,
    pub world_size: usize,
    pub comm: Comm,
}

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
        let id = exchange_unique_id(rank, world_size, init_method)?;
        let comm = Comm::from_rank(stream, rank, world_size, id).map_err(|err| {
            anyhow!("ncclCommInitRank failed for rank {rank}/{world_size}: {err:?}")
        })?;

        Ok(Self {
            rank,
            world_size,
            comm,
        })
    }

    pub fn all_reduce_smoke(&self, input: &[f32]) -> Result<Vec<f32>> {
        let stream = self.comm.stream();
        let send = stream
            .clone_htod(input)
            .with_context(|| format!("rank {} H2D smoke input copy failed", self.rank))?;
        let mut recv = stream
            .alloc_zeros::<f32>(input.len())
            .with_context(|| format!("rank {} smoke output allocation failed", self.rank))?;

        self.comm
            .all_reduce(&send, &mut recv, &ReduceOp::Sum)
            .map_err(|err| anyhow!("rank {} NCCL all_reduce(sum) failed: {err:?}", self.rank))?;
        stream
            .synchronize()
            .with_context(|| format!("rank {} stream sync after NCCL failed", self.rank))?;
        stream
            .clone_dtoh(&recv)
            .with_context(|| format!("rank {} D2H smoke output copy failed", self.rank))
    }
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
