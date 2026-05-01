use std::collections::HashMap;
use std::fmt;

use super::hash::MmHash;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum EncoderCacheError {
    EntryExceedsCapacity {
        entry_bytes: u64,
        capacity_bytes: u64,
    },
}

impl fmt::Display for EncoderCacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EntryExceedsCapacity {
                entry_bytes,
                capacity_bytes,
            } => write!(
                f,
                "encoder cache entry has {entry_bytes} bytes, above capacity {capacity_bytes}"
            ),
        }
    }
}

impl std::error::Error for EncoderCacheError {}

struct Entry<T> {
    value: T,
    byte_len: u64,
    last_access: u64,
}

/// Byte-bounded LRU cache for multimodal encoder outputs.
///
/// The cached tensor/buffer type is generic so CUDA, Metal, and CPU tests can
/// own backend-native payloads without a shared trait object on the hot path.
pub struct EncoderCache<T> {
    entries: HashMap<MmHash, Entry<T>>,
    capacity_bytes: u64,
    used_bytes: u64,
    clock: u64,
}

impl<T> EncoderCache<T> {
    #[must_use]
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            entries: HashMap::new(),
            capacity_bytes,
            used_bytes: 0,
            clock: 0,
        }
    }

    #[must_use]
    pub fn capacity_bytes(&self) -> u64 {
        self.capacity_bytes
    }

    #[must_use]
    pub fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[must_use]
    pub fn contains(&self, key: MmHash) -> bool {
        self.entries.contains_key(&key)
    }

    pub fn get(&mut self, key: MmHash) -> Option<&T> {
        self.clock = self.clock.wrapping_add(1);
        let entry = self.entries.get_mut(&key)?;
        entry.last_access = self.clock;
        Some(&entry.value)
    }

    pub fn remove(&mut self, key: MmHash) -> Option<T> {
        let entry = self.entries.remove(&key)?;
        self.used_bytes = self.used_bytes.saturating_sub(entry.byte_len);
        Some(entry.value)
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.used_bytes = 0;
    }

    pub fn insert(
        &mut self,
        key: MmHash,
        value: T,
        byte_len: u64,
    ) -> Result<Option<T>, EncoderCacheError> {
        if byte_len > self.capacity_bytes {
            return Err(EncoderCacheError::EntryExceedsCapacity {
                entry_bytes: byte_len,
                capacity_bytes: self.capacity_bytes,
            });
        }

        self.clock = self.clock.wrapping_add(1);
        let old = self.entries.insert(
            key,
            Entry {
                value,
                byte_len,
                last_access: self.clock,
            },
        );
        let replaced = old.map(|entry| {
            self.used_bytes = self.used_bytes.saturating_sub(entry.byte_len);
            entry.value
        });
        self.used_bytes = self.used_bytes.saturating_add(byte_len);
        self.evict_until_within_capacity();
        Ok(replaced)
    }

    fn evict_until_within_capacity(&mut self) {
        while self.used_bytes > self.capacity_bytes {
            let Some(victim) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_access)
                .map(|(key, _)| *key)
            else {
                break;
            };
            let _ = self.remove(victim);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::hash::hash_canonical_media;

    fn key(label: &[u8]) -> MmHash {
        hash_canonical_media(label, Some("image/png"))
    }

    #[test]
    fn insert_and_get_touch_entry() {
        let mut cache = EncoderCache::new(32);
        let a = key(b"a");
        let b = key(b"b");
        let c = key(b"c");

        cache.insert(a, "a", 8).unwrap();
        cache.insert(b, "b", 8).unwrap();
        assert_eq!(cache.get(a), Some(&"a"));
        cache.insert(c, "c", 24).unwrap();

        assert!(cache.contains(a));
        assert!(!cache.contains(b));
        assert!(cache.contains(c));
    }

    #[test]
    fn replacing_entry_updates_used_bytes() {
        let mut cache = EncoderCache::new(32);
        let a = key(b"a");

        assert_eq!(cache.insert(a, "small", 8).unwrap(), None);
        assert_eq!(cache.used_bytes(), 8);
        assert_eq!(cache.insert(a, "large", 16).unwrap(), Some("small"));
        assert_eq!(cache.used_bytes(), 16);
        assert_eq!(cache.get(a), Some(&"large"));
    }

    #[test]
    fn oversized_entry_is_rejected() {
        let mut cache = EncoderCache::new(8);
        let err = cache.insert(key(b"a"), "a", 16).unwrap_err();

        assert_eq!(
            err,
            EncoderCacheError::EntryExceedsCapacity {
                entry_bytes: 16,
                capacity_bytes: 8
            }
        );
        assert!(cache.is_empty());
    }

    #[test]
    fn clear_drops_accounting() {
        let mut cache = EncoderCache::new(32);
        cache.insert(key(b"a"), "a", 8).unwrap();
        cache.clear();

        assert_eq!(cache.len(), 0);
        assert_eq!(cache.used_bytes(), 0);
    }
}
