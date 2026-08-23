//! First decode call captures the kernel sequence into a graph, later calls
//! replay it. The `AUTO_FREE_ON_LAUNCH` instantiate flag frees async-pool
//! allocations made during capture on launch.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, ensure};
use cudarc::driver::safe::{CudaGraph, CudaStream};
use cudarc::driver::sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
use cudarc::driver::sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL;
use log::debug;

/// The first [`Self::run_or_capture`] call captures the kernel closure into a
/// graph and launches it; every later call launches the captured graph only.
pub struct CudaGraphState {
    /// Compute stream the graph is captured on and replayed to (`end_capture`
    /// needs `&Arc<CudaStream>`, so the `Arc` is held here).
    stream: Arc<CudaStream>,
    graph: Option<CudaGraph>,
    /// Whole-step mode: when true, `run_or_capture` runs kernels eagerly so they
    /// record into an OUTER capture instead of nesting their own.
    bypass: bool,
    /// Eager warm runs remaining before the first capture (default 1).
    ///
    /// Lazy first-use initialization inside the kernel closure — DeepGEMM's
    /// JIT `cuModuleLoad` on a new shape, cuBLAS workspace setup, allocator
    /// pool growth — is ILLEGAL during stream capture (CUDA error 900,
    /// `STREAM_CAPTURE_UNSUPPORTED`). Running the closure eagerly first lets
    /// every lazy init land outside capture. This lives HERE so every graph
    /// user (per-portion attn/moe, tail, whole-step, Qwen dense decode, any
    /// future model) is capture-safe by construction instead of each call
    /// site rediscovering error 900.
    warm_remaining: u32,
}

// SAFETY: the wrapped `CudaGraph` holds `!Send` handles (a CUDA graph must be
// captured, launched, and destroyed on its owning thread/context). The invariant:
// `CudaGraphState` lives inside the single CUDA executor driven from one blocking
// inference thread, so capture and replay always run there. Touching it from
// another thread would race the stream — undefined GPU behaviour.
unsafe impl Send for CudaGraphState {}

impl CudaGraphState {
    #[must_use]
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self {
            stream,
            graph: None,
            bypass: false,
            warm_remaining: 1,
        }
    }

    /// Re-arm `n` eager warm runs without dropping the captured graph. Called
    /// at request boundaries (slot reset): the next step runs eagerly so
    /// host-side per-request work (ring bootstrap, compressed bulk pack) can
    /// execute, then replay resumes — capture cost is paid once per slot, not
    /// once per request.
    pub fn rearm_warm(&mut self, n: u32) {
        self.warm_remaining = self.warm_remaining.max(n);
    }

    #[must_use]
    pub fn is_captured(&self) -> bool {
        self.graph.is_some()
    }

    /// Whether the next [`Self::run_or_capture`] call will run eagerly (a warm
    /// run is still armed) rather than replay/capture. Lets callers count
    /// replays precisely for the reuse-evidence probe.
    #[must_use]
    pub fn is_armed_warm(&self) -> bool {
        self.warm_remaining > 0
    }

    /// `kernels` must be a pure GPU kernel sequence — no host/device sync, no
    /// host allocation — because the work is recorded into a replay-able graph.
    ///
    /// # Errors
    /// Propagates `begin_capture` / `end_capture` / `launch` driver errors.
    pub fn run_or_capture<F>(&mut self, kernels: F) -> Result<()>
    where
        F: FnOnce() -> Result<()>,
    {
        if self.bypass {
            return kernels();
        }
        // Warm runs take precedence over replay: a captured graph can be
        // re-armed (`rearm_warm`) so the next call(s) run eagerly — the
        // per-request boundary hook (slot reset) uses this to run bootstrap /
        // bulk host work once per request while KEEPING the captured graph.
        if self.warm_remaining > 0 {
            self.warm_remaining -= 1;
            return kernels();
        }
        if let Some(graph) = &self.graph {
            graph
                .launch()
                .map_err(|e| anyhow::anyhow!("CUDA Graph launch failed: {e}"))?;
            return Ok(());
        }

        debug!("Capturing CUDA Graph for decode path...");
        self.stream
            .begin_capture(CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| anyhow::anyhow!("begin_capture failed: {e}"))?;

        // A mid-closure error MUST still terminate the capture: leaving the
        // stream in capture mode would record all subsequent (eager-fallback)
        // work into an orphaned capture and poison the stream. Captured work
        // is recorded, not executed, so aborting here leaves device state
        // untouched and the eager fallback re-runs the step cleanly.
        if let Err(e) = kernels() {
            let _ = self
                .stream
                .end_capture(CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
            return Err(e);
        }

        // Walk the still-capturing graph's nodes BEFORE instantiation: a
        // memcpy node whose src/dst is HOST memory re-reads that host address
        // on every replay — when the source was a stack/heap temporary this
        // is a use-after-free that corrupts silently (frozen topk → IMA;
        // frozen active_counts → MoE no-op at fake +24%, 2026-06-10). Audit
        // failure is intentionally fatal: the alternative is silent garbage.
        let audit = audit_capturing_graph(&self.stream);

        let graph = self
            .stream
            .end_capture(CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
            .map_err(|e| anyhow::anyhow!("end_capture failed: {e}"))?;
        match audit {
            Ok(a) => {
                ensure!(
                    a.host_memcpy_nodes == 0 && a.host_fn_nodes == 0,
                    "captured graph is host-coupled: {} host-side memcpy node(s), {} host-callback node(s)                      of {} total — per-step values must be device-derived or pre-replay updates into                      persistent buffers (graph discarded; see errors/2026-06-10 graph rekill entry)",
                    a.host_memcpy_nodes,
                    a.host_fn_nodes,
                    a.total_nodes,
                );
                if a.mem_alloc_nodes > 0 {
                    log::warn!(
                        "captured graph allocates: {} alloc / {} free node(s) of {} — legal                          (AUTO_FREE_ON_LAUNCH) but fragile; prefer persistent scratch",
                        a.mem_alloc_nodes,
                        a.mem_free_nodes,
                        a.total_nodes,
                    );
                }
            }
            Err(e) => debug!("capture audit unavailable: {e}"),
        }
        self.graph = graph;
        debug!("CUDA Graph captured successfully");

        if let Some(graph) = &self.graph {
            graph
                .launch()
                .map_err(|e| anyhow::anyhow!("CUDA Graph first launch failed: {e}"))?;
        }
        Ok(())
    }

    /// Pre-upload the captured graph so the first replay skips setup overhead.
    ///
    /// No-op when nothing has been captured yet.
    ///
    /// # Errors
    /// Propagates the `upload` driver error.
    pub fn upload(&self) -> Result<()> {
        if let Some(graph) = &self.graph {
            graph
                .upload()
                .map_err(|e| anyhow::anyhow!("CUDA Graph upload failed: {e}"))?;
        }
        Ok(())
    }
}

/// One captured decode graph keyed by its decode batch size. Wraps a
/// [`CudaGraphState`] so a later stage can carry per-bucket buffers alongside it.
pub struct CapturedDecodeGraph {
    pub batch_size: usize,
    pub state: CudaGraphState,
}

impl CapturedDecodeGraph {
    #[must_use]
    pub fn new(batch_size: usize, stream: Arc<CudaStream>) -> Self {
        Self {
            batch_size,
            state: CudaGraphState::new(stream),
        }
    }
}

/// Node census of a captured graph, taken via `cuStreamGetCaptureInfo_v2`
/// while capture is still active (the safe `CudaGraph` wrapper does not
/// expose its raw handle).
struct CaptureAudit {
    total_nodes: usize,
    /// Memcpy nodes touching HOST memory on either side — these re-read the
    /// recorded host ADDRESS every replay (use-after-free when the source was
    /// a temporary). Zero legitimate uses in this codebase.
    host_memcpy_nodes: usize,
    /// CU_GRAPH_NODE_TYPE_HOST callback nodes — host fn pointers in a replay.
    host_fn_nodes: usize,
    mem_alloc_nodes: usize,
    mem_free_nodes: usize,
}

fn audit_capturing_graph(stream: &CudaStream) -> Result<CaptureAudit> {
    use cudarc::driver::sys as cu;
    let mut status = cu::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
    let mut id: u64 = 0;
    let mut graph: cu::CUgraph = std::ptr::null_mut();
    let mut deps: *const cu::CUgraphNode = std::ptr::null();
    let mut ndeps: usize = 0;
    // SAFETY: out-params are valid locals; the stream handle is live.
    unsafe {
        cu::cuStreamGetCaptureInfo_v2(
            stream.cu_stream(),
            &mut status,
            &mut id,
            &mut graph,
            &mut deps,
            &mut ndeps,
        )
        .result()?;
    }
    ensure!(
        status == cu::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE && !graph.is_null(),
        "stream not in active capture"
    );
    let mut n: usize = 0;
    // SAFETY: null nodes-out queries the count.
    unsafe { cu::cuGraphGetNodes(graph, std::ptr::null_mut(), &mut n).result()? };
    let mut nodes: Vec<cu::CUgraphNode> = vec![std::ptr::null_mut(); n];
    // SAFETY: nodes has capacity n as reported by the count query.
    unsafe { cu::cuGraphGetNodes(graph, nodes.as_mut_ptr(), &mut n).result()? };
    if std::env::var_os("ARLE_GRAPH_NODE_CENSUS").is_some_and(|v| v != "0") {
        log_kernel_node_census(graph, &nodes[..n]);
    }
    let mut audit = CaptureAudit {
        total_nodes: n,
        host_memcpy_nodes: 0,
        host_fn_nodes: 0,
        mem_alloc_nodes: 0,
        mem_free_nodes: 0,
    };
    for &node in &nodes[..n] {
        let mut ty = cu::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL;
        // SAFETY: node handles come from cuGraphGetNodes on a live graph.
        unsafe { cu::cuGraphNodeGetType(node, &mut ty).result()? };
        match ty {
            cu::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMCPY => {
                // CUDA_MEMCPY3D contains enums with no 0 variant — mem::zeroed
                // aborts (validity check). MaybeUninit + driver-filled out-param.
                let mut p = std::mem::MaybeUninit::<cu::CUDA_MEMCPY3D>::uninit();
                // SAFETY: the driver fully writes the params on success.
                if unsafe { cu::cuGraphMemcpyNodeGetParams(node, p.as_mut_ptr()) }
                    .result()
                    .is_ok()
                {
                    // SAFETY: success above guarantees initialization.
                    let p = unsafe { p.assume_init() };
                    if matches!(p.srcMemoryType, cu::CUmemorytype::CU_MEMORYTYPE_HOST)
                        || matches!(p.dstMemoryType, cu::CUmemorytype::CU_MEMORYTYPE_HOST)
                    {
                        audit.host_memcpy_nodes += 1;
                    }
                }
            }
            cu::CUgraphNodeType::CU_GRAPH_NODE_TYPE_HOST => audit.host_fn_nodes += 1,
            cu::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_ALLOC => {
                audit.mem_alloc_nodes += 1;
                if std::env::var_os("ARLE_GRAPH_NODE_CENSUS").is_some() {
                    let mut p = std::mem::MaybeUninit::<cu::CUDA_MEM_ALLOC_NODE_PARAMS>::uninit();
                    // SAFETY: node is a live MEM_ALLOC node; the driver fills params.
                    if unsafe { cu::cuGraphMemAllocNodeGetParams(node, p.as_mut_ptr()) }
                        .result()
                        .is_ok()
                    {
                        // SAFETY: success above guarantees initialization.
                        let p = unsafe { p.assume_init() };
                        log::info!("[graph-alloc-census] bytes={}", p.bytesize);
                    }
                }
            }
            cu::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_FREE => audit.mem_free_nodes += 1,
            _ => {}
        }
    }
    Ok(audit)
}

/// Dump every kernel node of the capturing graph in execution order, with its
/// launch geometry (`ARLE_GRAPH_NODE_CENSUS=1`, capture-time only).
///
/// A profiler aggregates by kernel name, so one Marlin row covers every
/// projection in the step and its efficiency cannot be split by call site.
/// This census is the join key: `nsys profile --cuda-graph-trace=node` times
/// each node separately, and the ordered list below names what each one is.
fn log_kernel_node_census(
    graph: cudarc::driver::sys::CUgraph,
    nodes: &[cudarc::driver::sys::CUgraphNode],
) {
    use cudarc::driver::sys as cu;
    let order = topological_order(graph, nodes);
    log::info!(
        "[graph-node-census] {} node(s), execution order",
        order.len()
    );
    for (i, &node) in order.iter().enumerate() {
        let mut p = std::mem::MaybeUninit::<cu::CUDA_KERNEL_NODE_PARAMS>::uninit();
        // SAFETY: node handles come from cuGraphGetNodes on a live graph; a
        // non-kernel node fails cleanly and is skipped.
        #[cfg(infer_cuda_cuda_12)]
        let params_ok = unsafe { cu::cuGraphKernelNodeGetParams_v2(node, p.as_mut_ptr()) }
            .result()
            .is_ok();
        #[cfg(not(infer_cuda_cuda_12))]
        // SAFETY: same as above: node handles from cuGraphGetNodes on a live graph.
        let params_ok = unsafe { cu::cuGraphKernelNodeGetParams(node, p.as_mut_ptr()) }
            .result()
            .is_ok();
        if !params_ok {
            let mut ty = cu::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL;
            // SAFETY: node handle from cuGraphGetNodes on a live graph.
            let _ = unsafe { cu::cuGraphNodeGetType(node, &mut ty) };
            if ty == cu::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_ALLOC {
                let mut ap = std::mem::MaybeUninit::<cu::CUDA_MEM_ALLOC_NODE_PARAMS>::uninit();
                // SAFETY: node is a live MEM_ALLOC node; the driver fills params.
                if unsafe { cu::cuGraphMemAllocNodeGetParams(node, ap.as_mut_ptr()) }
                    .result()
                    .is_ok()
                {
                    // SAFETY: success above guarantees initialization.
                    let ap = unsafe { ap.assume_init() };
                    log::info!("[graph-node-census] {i:04} ALLOC bytes={}", ap.bytesize);
                }
            } else if ty == cu::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_FREE {
                log::info!("[graph-node-census] {i:04} FREE");
            }
            continue;
        }
        // SAFETY: success above guarantees initialization.
        let p = unsafe { p.assume_init() };
        #[cfg(infer_cuda_cuda_12)]
        let name = {
            let mut raw: *const std::ffi::c_char = std::ptr::null();
            // A node carries `func` or `kern`, never both.
            let got = if !p.func.is_null() {
                // SAFETY: `p.func` is a valid CUfunction from the captured graph node.
                unsafe { cu::cuFuncGetName(&mut raw, p.func) }
                    .result()
                    .is_ok()
            } else if !p.kern.is_null() {
                // SAFETY: `p.kern` is a valid CUkernel from the captured graph node.
                unsafe { cu::cuKernelGetName(&mut raw, p.kern) }
                    .result()
                    .is_ok()
            } else {
                false
            };
            if got && !raw.is_null() {
                // SAFETY: `raw` was written by cuFuncGetName/cuKernelGetName and is a valid C string.
                unsafe { std::ffi::CStr::from_ptr(raw) }
                    .to_string_lossy()
                    .into_owned()
            } else {
                "<unnamed>".to_string()
            }
        };
        #[cfg(not(infer_cuda_cuda_12))]
        let name = "<kernel>".to_string();
        log::info!(
            "[graph-node-census] {i:04} grid=({},{},{}) block=({},{},{}) smem={} {name}",
            p.gridDimX,
            p.gridDimY,
            p.gridDimZ,
            p.blockDimX,
            p.blockDimY,
            p.blockDimZ,
            p.sharedMemBytes,
        );
    }
}

/// Nodes in execution order. Single-stream capture yields a chain, so a
/// Kahn walk is exact; on any branch the tie is broken by handle order,
/// which keeps the census total-correct if locally out of order.
fn topological_order(
    graph: cudarc::driver::sys::CUgraph,
    nodes: &[cudarc::driver::sys::CUgraphNode],
) -> Vec<cudarc::driver::sys::CUgraphNode> {
    use cudarc::driver::sys as cu;
    let mut n_edges: usize = 0;
    // SAFETY: null from/to queries the edge count.
    if unsafe {
        cu::cuGraphGetEdges(
            graph,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &mut n_edges,
        )
    }
    .result()
    .is_err()
    {
        return nodes.to_vec();
    }
    let mut from = vec![std::ptr::null_mut(); n_edges];
    let mut to = vec![std::ptr::null_mut(); n_edges];
    // SAFETY: both buffers hold n_edges entries as reported by the count query.
    if unsafe { cu::cuGraphGetEdges(graph, from.as_mut_ptr(), to.as_mut_ptr(), &mut n_edges) }
        .result()
        .is_err()
    {
        return nodes.to_vec();
    }
    let index: HashMap<_, _> = nodes.iter().enumerate().map(|(i, &n)| (n, i)).collect();
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
    let mut indeg = vec![0usize; nodes.len()];
    for e in 0..n_edges {
        let (Some(&f), Some(&t)) = (index.get(&from[e]), index.get(&to[e])) else {
            continue;
        };
        succ[f].push(t);
        indeg[t] += 1;
    }
    let mut ready: Vec<usize> = (0..nodes.len()).filter(|&i| indeg[i] == 0).collect();
    let mut out = Vec::with_capacity(nodes.len());
    while let Some(i) = ready.pop() {
        out.push(nodes[i]);
        for &s in &succ[i] {
            indeg[s] -= 1;
            if indeg[s] == 0 {
                ready.push(s);
            }
        }
    }
    if out.len() == nodes.len() {
        out
    } else {
        nodes.to_vec()
    }
}
