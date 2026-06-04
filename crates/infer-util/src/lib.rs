//! Cross-cutting runtime utilities with no backend coupling.
//!
//! - [`hf_hub`] — HuggingFace model-id/path resolution + download (host-only;
//!   no GPU/backend deps), relocated from `infer/src/hf_hub.rs`.
//! - [`logging`] — stderr logger init, relocated from `infer/src/logging.rs`.
//!
//! Lives in its own leaf crate so the legacy `infer` crate can be deleted
//! without losing these, and so host-only commands (cli `doctor`/`download`)
//! can use them without dragging in a backend-gated engine crate.

pub mod hf_hub;
pub mod logging;
