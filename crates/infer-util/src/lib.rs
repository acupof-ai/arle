//! Cross-cutting runtime utilities with no backend coupling.
//!
//! Leaf crate so host-only commands (cli `doctor`/`download`) can use these
//! without dragging in a backend-gated engine crate.

pub mod hf_hub;
pub mod logging;
