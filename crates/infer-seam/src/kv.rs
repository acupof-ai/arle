//! Composed KV pool seam.
//!
//! [`KvPool`] is the union of the three cohesive sub-traits
//! ([`KvQuery`](crate::KvQuery), [`KvAllocator`](crate::KvAllocator),
//! [`KvPrefixStore`](crate::KvPrefixStore)). A blanket impl gives `KvPool` to
//! any type implementing all three, so backends implement the three pieces and
//! engine-core can still hold `&mut dyn KvPool` without knowing the backend.

use crate::{KvAllocator, KvPrefixStore, KvQuery};

/// Host-indexed KV pool surface visible to engine-core.
///
/// Every method is expressed in host slot ids, page ids, token counts, and
/// logical positions. The trait is dyn-safe so engine-core can hold
/// `&mut dyn KvPool` without knowing the backend.
///
/// Engine-core is the sole writer: it grows/shrinks slots through the
/// [`KvAllocator`](crate::KvAllocator)/[`KvSlotAccounting`](crate::KvSlotAccounting)
/// supertraits (the supertrait chain lets `&mut dyn KvPool` call them at the
/// engine-core site). Backends receive only the read-only
/// [`KvBatchDescriptor`] view and report reached lengths in `StepOutput::kv_actual`.
pub trait KvPool: KvQuery + KvAllocator + KvPrefixStore {}

impl<T: KvQuery + KvAllocator + KvPrefixStore> KvPool for T {}
