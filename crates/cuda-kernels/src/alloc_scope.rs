//! Owner-granularity tracked VRAM registry — scope attribution for device allocations.
//!
//! This is the **scope dimension** on top of the call-site `CUDA_ALLOC_TRACE`
//! scaffold in [`crate::tensor`]. It answers "which subsystem owns this VRAM"
//! at runtime by attributing each byte-dominant device allocation to a logical
//! [`ScopeId`] (weights, KV pool, optimizer, …) and maintaining a *live* gauge
//! (alloc increments, owner-`Drop` decrements) — not the cumulative flow the
//! call-site trace records.
//!
//! Design: `docs/plans/2026-06-26-unified-tracked-allocator-design.md`.
//!
//! ## Three pieces
//!
//! 1. **Scope span** — a thread-local label stack + an RAII [`ScopeGuard`]
//!    ([`scope`]). Subsystem boundaries open a span; every tracked allocation
//!    inside attributes to the top-of-stack scope.
//! 2. **Registry** — a process-global `[AtomicI64; ScopeId::COUNT]` of live
//!    bytes. [`record_scope_alloc`] adds, [`record_scope_free`] subtracts.
//! 3. **Owner free-decrement** — the byte-dominant owners ([`crate::tensor`]
//!    `DeviceVec`/`DeviceMatrix`/`HiddenStates`, the KV pool, the workspace
//!    slots) carry the `(ScopeId, bytes)` they incremented in a [`VramTag`] and
//!    decrement the *same* scope from their `Drop` — because `CudaSlice::Drop`
//!    is foreign and cannot be hooked.
//!
//! ## Gate
//!
//! Everything is gated default-OFF behind `ARLE_VRAM_TRACE` (also honoring the
//! legacy `ARLE_CUDA_ALLOC_TRACE`). When off, `record_scope_*` early-return and
//! the registry is never touched — zero steady-state cost. Decode is
//! allocation-free, so the only call sites hit at all are load / prefill-flip /
//! optimizer-init.

use std::cell::RefCell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicI64, Ordering};

/// Logical owner of a device allocation. Small interned enum so the registry is
/// a fixed-size atomic array keyed by `as usize`.
///
/// `Untracked` is the default for owner buffers that were constructed outside a
/// scope span (transient scratch, clones, externally-built struct literals) —
/// their `Drop` is a no-op decrement.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(u16)]
pub enum ScopeId {
    /// No owning subsystem — buffer was not allocated inside a scope span.
    #[default]
    Untracked = 0,
    /// Model weights (`DeviceMatrix`/`DeviceVec` from the loader).
    Weights = 1,
    /// Paged KV cache slab (`PagedKVPool`/`TokenKVPool`).
    KvPool = 2,
    /// Per-layer attention state.
    AttnState = 3,
    /// MoE scratch / expert routing buffers.
    MoeScratch = 4,
    /// Optimizer moments (AdamW `m`/`v`).
    Optimizer = 5,
    /// Forward activations / workspace slots.
    Activations = 6,
    /// OPD teacher resident weights.
    OpdTeacher = 7,
    /// DeepEP transport scratch (EP path).
    DeepEp = 8,
}

impl ScopeId {
    /// Number of registry buckets (must equal the highest discriminant + 1).
    pub const COUNT: usize = 9;

    /// Stable lowercase label used in the report.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            ScopeId::Untracked => "untracked",
            ScopeId::Weights => "weights",
            ScopeId::KvPool => "kv_pool",
            ScopeId::AttnState => "attn_state",
            ScopeId::MoeScratch => "moe_scratch",
            ScopeId::Optimizer => "optimizer",
            ScopeId::Activations => "activations",
            ScopeId::OpdTeacher => "opd_teacher",
            ScopeId::DeepEp => "deepep",
        }
    }

    /// All scopes in registry order, for reporting.
    #[must_use]
    pub fn all() -> [ScopeId; ScopeId::COUNT] {
        [
            ScopeId::Untracked,
            ScopeId::Weights,
            ScopeId::KvPool,
            ScopeId::AttnState,
            ScopeId::MoeScratch,
            ScopeId::Optimizer,
            ScopeId::Activations,
            ScopeId::OpdTeacher,
            ScopeId::DeepEp,
        ]
    }
}

/// The `(scope, bytes)` an owner buffer incremented at construction, stored so
/// the owner's `Drop` decrements the matching scope independent of the
/// thread-local stack at drop time.
///
/// `Default` is `Untracked` / `0` so adding this field to a struct keeps
/// externally-written struct literals and `#[derive(Default)]` paths valid
/// (their buffers are simply not attributed).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct VramTag {
    scope: ScopeId,
    bytes: u64,
}

impl VramTag {
    /// An untracked tag (no registry effect on drop). Use in struct literals
    /// for buffers built outside a load-phase scope.
    #[must_use]
    pub const fn untracked() -> Self {
        VramTag {
            scope: ScopeId::Untracked,
            bytes: 0,
        }
    }

    /// Record `bytes` against the current top-of-stack scope and return a tag
    /// the owner stores for its `Drop` to decrement. When tracing is off (or the
    /// current scope is `Untracked`) this is a no-op and returns
    /// [`VramTag::untracked`].
    #[must_use]
    pub fn record(bytes: u64) -> Self {
        if !vram_trace_enabled() {
            return VramTag::untracked();
        }
        let scope = current_scope();
        if scope == ScopeId::Untracked || bytes == 0 {
            return VramTag::untracked();
        }
        registry()[scope as usize].fetch_add(bytes as i64, Ordering::Relaxed);
        VramTag { scope, bytes }
    }

    /// Scope this tag is attributed to.
    #[must_use]
    pub fn scope(self) -> ScopeId {
        self.scope
    }

    /// Bytes this tag holds.
    #[must_use]
    pub fn bytes(self) -> u64 {
        self.bytes
    }

    /// Decrement the registry by this tag's bytes. Idempotent for `Untracked`;
    /// the owner `Drop` calls this exactly once (owners move, never double-drop).
    pub fn release(self) {
        if self.scope == ScopeId::Untracked || self.bytes == 0 {
            return;
        }
        registry()[self.scope as usize].fetch_sub(self.bytes as i64, Ordering::Relaxed);
    }
}

static VRAM_TRACE_ENABLED: OnceLock<bool> = OnceLock::new();

/// Whether owner-granularity VRAM tracking is on. Default OFF; enabled by
/// `ARLE_VRAM_TRACE` (and the legacy `ARLE_CUDA_ALLOC_TRACE`, so the existing
/// call-site trace flag also turns the scope registry on).
pub fn vram_trace_enabled() -> bool {
    *VRAM_TRACE_ENABLED.get_or_init(|| {
        let truthy = |v: &str| matches!(v, "1" | "true" | "TRUE" | "yes" | "on" | "ON");
        std::env::var("ARLE_VRAM_TRACE")
            .ok()
            .as_deref()
            .map(truthy)
            .unwrap_or(false)
            || std::env::var("ARLE_CUDA_ALLOC_TRACE")
                .ok()
                .as_deref()
                .map(truthy)
                .unwrap_or(false)
    })
}

fn registry() -> &'static [AtomicI64; ScopeId::COUNT] {
    static REGISTRY: OnceLock<[AtomicI64; ScopeId::COUNT]> = OnceLock::new();
    REGISTRY.get_or_init(|| std::array::from_fn(|_| AtomicI64::new(0)))
}

thread_local! {
    static SCOPE_STACK: RefCell<Vec<ScopeId>> = const { RefCell::new(Vec::new()) };
}

/// Current top-of-stack scope, or `Untracked` if no span is open.
#[must_use]
pub fn current_scope() -> ScopeId {
    SCOPE_STACK.with(|s| s.borrow().last().copied().unwrap_or(ScopeId::Untracked))
}

/// RAII span guard: pops its scope off the thread-local stack on drop.
#[must_use = "the scope closes when this guard is dropped; bind it to a variable"]
pub struct ScopeGuard {
    // Non-Send/Sync: the stack is thread-local and the guard must drop on the
    // same thread that pushed.
    _not_send: std::marker::PhantomData<*const ()>,
}

/// Open a VRAM-attribution span. Every tracked allocation constructed while the
/// returned guard is live attributes to `id`.
///
/// ```ignore
/// let _g = alloc_scope::scope(ScopeId::KvPool);
/// let pool = PagedKVPool::with_format(ctx, /* … */)?; // slab → KvPool
/// ```
pub fn scope(id: ScopeId) -> ScopeGuard {
    SCOPE_STACK.with(|s| s.borrow_mut().push(id));
    ScopeGuard {
        _not_send: std::marker::PhantomData,
    }
}

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        SCOPE_STACK.with(|s| {
            s.borrow_mut().pop();
        });
    }
}

/// Convenience span openers for the named subsystems.
pub struct VramScope;

impl VramScope {
    /// Open a weights-load span.
    pub fn weights() -> ScopeGuard {
        scope(ScopeId::Weights)
    }
    /// Open a KV-pool-build span.
    pub fn kv_pool() -> ScopeGuard {
        scope(ScopeId::KvPool)
    }
    /// Open an attention-state span.
    pub fn attn_state() -> ScopeGuard {
        scope(ScopeId::AttnState)
    }
    /// Open a MoE-scratch span.
    pub fn moe_scratch() -> ScopeGuard {
        scope(ScopeId::MoeScratch)
    }
    /// Open an optimizer span.
    pub fn optimizer() -> ScopeGuard {
        scope(ScopeId::Optimizer)
    }
    /// Open an activations span.
    pub fn activations() -> ScopeGuard {
        scope(ScopeId::Activations)
    }
    /// Open an OPD-teacher span.
    pub fn opd_teacher() -> ScopeGuard {
        scope(ScopeId::OpdTeacher)
    }
    /// Open a DeepEP span.
    pub fn deepep() -> ScopeGuard {
        scope(ScopeId::DeepEp)
    }
}

/// Increment the current scope by `bytes` (used by constructors that don't own a
/// [`VramTag`] field, e.g. the KV slab which records its summed slab size once).
/// Returns the scope it attributed to (for the caller to remember for its own
/// `Drop`), or `Untracked` if tracing is off / no span open.
pub fn record_scope_alloc(bytes: u64) -> ScopeId {
    VramTag::record(bytes).scope()
}

/// Decrement `scope` by `bytes` directly (used by owners that store
/// `(ScopeId, bytes)` separately rather than a [`VramTag`]).
pub fn record_scope_free(scope: ScopeId, bytes: u64) {
    if scope == ScopeId::Untracked || bytes == 0 {
        return;
    }
    registry()[scope as usize].fetch_sub(bytes as i64, Ordering::Relaxed);
}

/// Per-scope live bytes (`alloc − free`). Negative values (double-free /
/// untracked-decrement) are surfaced raw here for leak diagnostics.
#[must_use]
pub fn bytes_by_scope() -> Vec<(ScopeId, i64)> {
    let reg = registry();
    ScopeId::all()
        .into_iter()
        .map(|s| (s, reg[s as usize].load(Ordering::Relaxed)))
        .collect()
}

/// Live bytes for a single scope.
#[must_use]
pub fn scope_bytes(scope: ScopeId) -> i64 {
    registry()[scope as usize].load(Ordering::Relaxed)
}

/// Sum of all per-scope live bytes (≈ `pool.used` when fully wired).
#[must_use]
pub fn registry_total_bytes() -> i64 {
    registry().iter().map(|a| a.load(Ordering::Relaxed)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    // The registry is process-global; force-enable tracing for the unit test
    // without racing the env-gated `OnceLock`. We exercise the bookkeeping
    // through the same code paths the owners use, but drive the gate directly.
    fn enable_for_test() {
        let _ = VRAM_TRACE_ENABLED.set(true);
    }

    fn reset_scope(scope: ScopeId) {
        registry()[scope as usize].store(0, Ordering::Relaxed);
    }

    #[test]
    fn scope_push_record_drop_decrement_roundtrips() {
        enable_for_test();
        // Use a scope unlikely to be touched by any concurrent test.
        let scope_id = ScopeId::OpdTeacher;
        reset_scope(scope_id);

        let n: u64 = 4096;
        {
            let _g = scope(scope_id);
            assert_eq!(current_scope(), scope_id, "span pushes the scope");

            let tag = VramTag::record(n);
            assert_eq!(tag.scope(), scope_id);
            assert_eq!(tag.bytes(), n);
            assert_eq!(
                scope_bytes(scope_id),
                n as i64,
                "record increments the live gauge by N"
            );

            // Owner-Drop decrement.
            tag.release();
            assert_eq!(scope_bytes(scope_id), 0, "release decrements back to 0");
        }
        // Span closed: stack is empty again, so a record outside attributes to
        // Untracked (no-op).
        assert_eq!(current_scope(), ScopeId::Untracked);
        let outside = VramTag::record(n);
        assert_eq!(outside.scope(), ScopeId::Untracked);
        assert_eq!(scope_bytes(scope_id), 0);
    }

    #[test]
    fn direct_record_free_api_roundtrips() {
        // The `record_scope_alloc` / `record_scope_free` pair (used by autograd's
        // backend `zeros` and the raw `SliceSlot`) must balance.
        enable_for_test();
        let scope_id = ScopeId::DeepEp;
        reset_scope(scope_id);

        let _g = scope(scope_id);
        let attributed = record_scope_alloc(2048);
        assert_eq!(attributed, scope_id);
        assert_eq!(scope_bytes(scope_id), 2048);

        record_scope_free(scope_id, 2048);
        assert_eq!(scope_bytes(scope_id), 0);

        // Free against Untracked / zero bytes is a no-op.
        record_scope_free(ScopeId::Untracked, 999);
        record_scope_free(scope_id, 0);
        assert_eq!(scope_bytes(scope_id), 0);
    }

    #[test]
    fn nested_spans_attribute_to_innermost() {
        enable_for_test();
        reset_scope(ScopeId::Weights);
        reset_scope(ScopeId::KvPool);

        let _outer = scope(ScopeId::Weights);
        let w = VramTag::record(100);
        assert_eq!(w.scope(), ScopeId::Weights);
        {
            let _inner = scope(ScopeId::KvPool);
            let k = VramTag::record(200);
            assert_eq!(k.scope(), ScopeId::KvPool, "innermost wins");
            assert_eq!(scope_bytes(ScopeId::KvPool), 200);
            k.release();
        }
        // Back to the outer scope.
        let w2 = VramTag::record(50);
        assert_eq!(w2.scope(), ScopeId::Weights);
        assert_eq!(scope_bytes(ScopeId::Weights), 150);
        w.release();
        w2.release();
        assert_eq!(scope_bytes(ScopeId::Weights), 0);
    }
}
