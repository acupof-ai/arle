//! Metal KV host bookkeeping.
//!
//! Metal uses the backend-neutral host paged-KV allocator. Device-side MLX KV
//! buffers live in the executor/session state; this type only tracks logical
//! slot/page/token ownership for the shared scheduler seam.

pub type MetalKvPool = infer_seam::HostPagedKvPool;
