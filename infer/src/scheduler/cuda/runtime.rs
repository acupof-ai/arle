//! CUDA scheduler runtime coordinator.
//!
//! This file used to hold the entire single-writer scheduler loop (~2117 LOC).
//! It was split into responsibility-sized submodules in 2026-04-27 (pure
//! structural refactor — no behavior change). The submodules are:
//!
//! - `helpers` — top-level helper fns, structs, enums (waiting-queue
//!   ordering, prefix-admission plans, fetch waiters).
//! - `admission` — admission-side scheduler methods: normalization, prefix
//!   plan building, slot materialization, cold-prefill fallback,
//!   staged-prefix promotion.
//! - `fetch` — fetch-completion + coordinator/event drain methods.
//! - `scheduler_loop` — the public `run` driver, `assign_slots`, `cleanup`,
//!   wakeup orchestration, `free_slots`.
//! - `tests` — unit tests for the helpers and admission ordering.
//!
//! External callers use only `Scheduler<M>::run` (declared in
//! `scheduler_loop`). All other methods are `pub(super)` or private and
//! resolve through the impl blocks across the submodules.

#[path = "runtime/helpers.rs"]
mod helpers;

#[path = "runtime/admission.rs"]
mod admission;

#[path = "runtime/fetch.rs"]
mod fetch;

#[path = "runtime/scheduler_loop.rs"]
mod scheduler_loop;

#[path = "runtime/tests.rs"]
mod tests_mod;

// `WaitingInsertBias` is consumed by sibling submodule `decode` —
// re-export it under the original name `runtime::WaitingInsertBias`.
pub(in crate::scheduler::cuda) use helpers::WaitingInsertBias;

// Tests reach helpers directly via `crate::scheduler::cuda::runtime::helpers::*`.
