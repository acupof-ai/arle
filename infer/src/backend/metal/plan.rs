//! Compatibility aliases for the unified scheduler logical plan IR.
//!
//! Metal still exposes `MetalLogical*` names to keep the scheduler/runtime
//! surface stable during M3, but the row schema now lives in
//! `crate::scheduler::plan`.

pub use crate::scheduler::{
    LogicalBatchShape as MetalLogicalBatchShape, LogicalDecodeRow as MetalLogicalDecodeRow,
    LogicalPrefillRow as MetalLogicalPrefillRow, LogicalServePlan as MetalLogicalServePlan,
};
