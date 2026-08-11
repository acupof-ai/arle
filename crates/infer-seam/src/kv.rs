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
pub trait KvPool: KvQuery + KvAllocator + KvPrefixStore {}

impl<T: KvQuery + KvAllocator + KvPrefixStore> KvPool for T {}
