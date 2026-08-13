use super::*;

impl CudaBackend {
    /// Create a backend bound to the CUDA device at `ordinal`.
    ///
    /// # Errors
    /// Returns an error if the device cannot be opened, cuBLAS cannot be
    /// initialised, or the autograd CUDA kernels fail NVRTC compilation.
    pub fn new(ordinal: usize) -> Result<Self> {
        #[cfg(feature = "no-cuda")]
        {
            let _ = ordinal;
            todo!("GPU required: CudaBackend::new is unavailable under feature no-cuda")
        }

        #[cfg(not(feature = "no-cuda"))]
        {
            let ctx = CudaContext::new(ordinal).map_err(|_| {
                AutogradError::TapeInvariant("CudaContext::new failed (is a GPU present?)")
            })?;
            let stream = ctx.default_stream();
            let blas = CudaBlas::new(stream.clone())
                .map_err(|_| AutogradError::TapeInvariant("CudaBlas::new failed"))?;
            let kernels = KernelCache::new(stream.context())?;
            Ok(Self {
                tape_dtype: AtomicU8::new(TapeDtype::F32 as u8),
                stream,
                blas: Arc::new(blas),
                kernels,
                pinned_checkpoints: Mutex::default(),
                #[cfg(all(feature = "nccl", not(feature = "no-cuda")))]
                nccl: None,
                #[cfg(all(feature = "nccl", not(feature = "no-cuda")))]
                nccl_seq: None,
            })
        }
    }

    /// Create a CUDA backend with an NCCL all-reduce communicator attached.
    ///
    /// The communicator uses the same default stream/context as autograd CUDA
    /// kernels, so surrounding ops and collectives are naturally ordered.
    #[cfg(all(feature = "nccl", not(feature = "no-cuda")))]
    pub fn new_with_nccl(
        ordinal: usize,
        unique_id: cuda_kernels::ffi::nccl::ncclUniqueId,
        world_size: usize,
        rank: usize,
    ) -> Result<Self> {
        let mut backend = Self::new(ordinal)?;
        let nccl = NcclBackend::init_rank(unique_id, world_size, rank).map_err(|err| {
            AutogradError::TapeInvariant(Box::leak(
                format!("NcclBackend::init_rank failed for autograd: {err:#}").into_boxed_str(),
            ))
        })?;
        let nccl = Arc::new(nccl);
        // Default seq comm = world comm; a composed mesh overwrites it.
        backend.nccl_seq = Some(nccl.clone());
        backend.nccl = Some(nccl);
        Ok(backend)
    }

    /// Mesh backend: grad/count all-reduces on the world comm, seq collectives
    /// on a `ncclCommSplit` subgroup `seq_group = (color, size, rank)`. The
    /// caller derives the spec from the one mesh — this crate stays
    /// layout-agnostic. `None` = no subgroup (identical to `new_with_nccl`).
    #[cfg(all(feature = "nccl", not(feature = "no-cuda")))]
    pub fn new_with_mesh(
        ordinal: usize,
        unique_id: cuda_kernels::ffi::nccl::ncclUniqueId,
        world_size: usize,
        world_rank: usize,
        seq_group: Option<(usize, usize, usize)>,
    ) -> Result<Self> {
        let mut backend = Self::new_with_nccl(ordinal, unique_id, world_size, world_rank)?;
        if let Some((color, size, rank)) = seq_group {
            let world_comm = backend.nccl.as_ref().expect("new_with_nccl sets nccl");
            // Collective over the world comm — every rank constructs together.
            let seq = world_comm
                .split(color as i32, rank as i32, size, rank)
                .map_err(|err| {
                    AutogradError::TapeInvariant(Box::leak(
                        format!("ncclCommSplit for the seq subgroup failed: {err:#}")
                            .into_boxed_str(),
                    ))
                })?;
            backend.nccl_seq = Some(Arc::new(seq));
        }
        Ok(backend)
    }

    #[cfg(all(feature = "nccl", not(feature = "no-cuda")))]
    pub(super) fn comm(&self, axis: CommAxis) -> Option<&Arc<NcclBackend>> {
        match axis {
            CommAxis::World | CommAxis::Expert => self.nccl.as_ref(),
            CommAxis::Seq => self.nccl_seq.as_ref(),
        }
    }

    /// No-GPU stub for no-cuda builds that still type-check the nccl feature.
    #[cfg(all(feature = "nccl", feature = "no-cuda"))]
    pub fn new_with_nccl(
        ordinal: usize,
        unique_id: cuda_kernels::ffi::nccl::ncclUniqueId,
        world_size: usize,
        rank: usize,
    ) -> Result<Self> {
        let _ = (ordinal, unique_id, world_size, rank);
        todo!("GPU required: CudaBackend::new_with_nccl is unavailable under feature no-cuda")
    }

    /// No-GPU stub for no-cuda builds that still type-check the nccl feature.
    #[cfg(all(feature = "nccl", feature = "no-cuda"))]
    pub fn new_with_mesh(
        ordinal: usize,
        unique_id: cuda_kernels::ffi::nccl::ncclUniqueId,
        world_size: usize,
        world_rank: usize,
        seq_group: Option<(usize, usize, usize)>,
    ) -> Result<Self> {
        let _ = (ordinal, unique_id, world_size, world_rank, seq_group);
        todo!("GPU required: CudaBackend::new_with_mesh is unavailable under feature no-cuda")
    }

    /// Query device VRAM `(free_bytes, total_bytes)` for the backend's CUDA
    /// context. Used by OPD attribution tooling to log per-phase peak VRAM
    /// without shelling out to `nvidia-smi`.
    ///
    /// # Errors
    /// Returns an error if the driver `cuMemGetInfo` call fails.
    #[cfg(not(feature = "no-cuda"))]
    pub fn mem_get_info(&self) -> Result<(usize, usize)> {
        self.stream
            .context()
            .mem_get_info()
            .map_err(|_| AutogradError::TapeInvariant("cuda mem_get_info failed"))
    }

    /// No-GPU stub — VRAM query is unavailable without a CUDA device.
    #[cfg(feature = "no-cuda")]
    pub fn mem_get_info(&self) -> Result<(usize, usize)> {
        todo!("GPU required: CudaBackend::mem_get_info is unavailable under feature no-cuda")
    }

    /// Whether an NCCL communicator is attached (multi-rank collectives work).
    #[cfg(all(feature = "nccl", not(feature = "no-cuda")))]
    pub fn has_collective(&self) -> bool {
        self.nccl.is_some()
    }

    /// Whether an NCCL communicator is attached (multi-rank collectives work).
    #[cfg(not(all(feature = "nccl", not(feature = "no-cuda"))))]
    pub fn has_collective(&self) -> bool {
        false
    }
}
