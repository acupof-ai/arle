//! Composed KV pool seam.
//!
//! [`KvPool`] is the union of the three cohesive sub-traits
//! ([`KvQuery`](crate::KvQuery), [`KvAllocator`](crate::KvAllocator),
//! [`KvPrefixStore`](crate::KvPrefixStore)). A blanket impl gives `KvPool` to
//! any type implementing all three, so backends implement the three pieces and
//! engine-core can still hold `&mut dyn KvPool` without knowing the backend.

use crate::{KvAllocator, KvPrefixStore, KvQuery, KvSlotAccounting};

/// Host-indexed KV pool surface visible to engine-core.
///
/// Every method is expressed in host slot ids, page ids, token counts, and
/// logical positions. The trait is dyn-safe so engine-core can hold
/// `&mut dyn KvPool` without knowing the backend.
///
/// `KvSlotAccounting` is the narrowed write surface `BackendExecutor::submit`
/// takes; it is a strict subset of `KvAllocator`, so the supertrait bound lets
/// `&mut dyn KvPool` upcast to it at the engine-core call site.
pub trait KvPool: KvQuery + KvAllocator + KvPrefixStore + KvSlotAccounting {}

impl<T: KvQuery + KvAllocator + KvPrefixStore> KvPool for T {}
