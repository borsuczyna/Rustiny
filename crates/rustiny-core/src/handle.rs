//! Generational handles and a slot pool.
//!
//! Engine resources (GPU buffers, textures, assets, ...) are referred to by [`Handle`]s rather
//! than raw pointers or indices. A handle becomes *stale* once its slot is freed, even if the slot
//! is later reused, which turns use-after-free bugs into cheap `None` lookups.

use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

/// Typed, generation-checked reference into a [`Pool<T>`].
pub struct Handle<T> {
    index: u32,
    generation: u32,
    _marker: PhantomData<fn() -> T>,
}

impl<T> Handle<T> {
    /// Slot index (stable for the lifetime of the pooled value).
    pub fn index(self) -> u32 {
        self.index
    }

    /// Generation counter of the slot at the time the handle was issued.
    pub fn generation(self) -> u32 {
        self.generation
    }
}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Handle<T> {}
impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.generation == other.generation
    }
}
impl<T> Eq for Handle<T> {}
impl<T> Hash for Handle<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.index.hash(state);
        self.generation.hash(state);
    }
}
impl<T> fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Handle({}v{})", self.index, self.generation)
    }
}

struct Slot<T> {
    generation: u32,
    value: Option<T>,
}

/// Dense-index pool handing out generation-checked [`Handle`]s.
///
/// Lookup, insert and remove are O(1). Freed slots are recycled in LIFO order, and their
/// generation is bumped so that old handles can never alias the new occupant. The generation
/// counter wraps after 2^32 reuses of one slot, which is acceptable for in-memory resources.
pub struct Pool<T> {
    slots: Vec<Slot<T>>,
    free: Vec<u32>,
    len: usize,
}

impl<T> Default for Pool<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> fmt::Debug for Pool<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pool")
            .field("len", &self.len)
            .field("capacity", &self.slots.len())
            .finish()
    }
}

impl<T> Pool<T> {
    /// Create an empty pool.
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            len: 0,
        }
    }

    /// Number of live values.
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` if the pool holds no live values.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Store `value` and return a handle to it.
    ///
    /// # Panics
    /// Panics if more than `u32::MAX` slots would be required.
    pub fn insert(&mut self, value: T) -> Handle<T> {
        self.len += 1;
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            slot.value = Some(value);
            Handle {
                index,
                generation: slot.generation,
                _marker: PhantomData,
            }
        } else {
            let index = u32::try_from(self.slots.len()).expect("pool exceeded u32::MAX slots");
            self.slots.push(Slot {
                generation: 0,
                value: Some(value),
            });
            Handle {
                index,
                generation: 0,
                _marker: PhantomData,
            }
        }
    }

    /// Remove and return the value, invalidating every outstanding copy of `handle`.
    pub fn remove(&mut self, handle: Handle<T>) -> Option<T> {
        let slot = self.slots.get_mut(handle.index as usize)?;
        if slot.generation != handle.generation {
            return None;
        }
        let value = slot.value.take()?;
        slot.generation = slot.generation.wrapping_add(1);
        self.free.push(handle.index);
        self.len -= 1;
        Some(value)
    }

    /// Borrow the value if the handle is still live.
    pub fn get(&self, handle: Handle<T>) -> Option<&T> {
        let slot = self.slots.get(handle.index as usize)?;
        (slot.generation == handle.generation)
            .then_some(slot.value.as_ref())
            .flatten()
    }

    /// Mutably borrow the value if the handle is still live.
    pub fn get_mut(&mut self, handle: Handle<T>) -> Option<&mut T> {
        let slot = self.slots.get_mut(handle.index as usize)?;
        (slot.generation == handle.generation)
            .then_some(slot.value.as_mut())
            .flatten()
    }

    /// `true` if `handle` still refers to a live value.
    pub fn contains(&self, handle: Handle<T>) -> bool {
        self.get(handle).is_some()
    }

    /// Iterate over live `(handle, value)` pairs in slot order.
    pub fn iter(&self) -> impl Iterator<Item = (Handle<T>, &T)> {
        self.slots.iter().enumerate().filter_map(|(i, slot)| {
            slot.value.as_ref().map(|v| {
                // `i` fits in u32 because `insert` enforces it.
                (
                    Handle {
                        index: i as u32,
                        generation: slot.generation,
                        _marker: PhantomData,
                    },
                    v,
                )
            })
        })
    }

    /// Remove every value, invalidating all handles.
    pub fn clear(&mut self) {
        for (i, slot) in self.slots.iter_mut().enumerate() {
            if slot.value.take().is_some() {
                slot.generation = slot.generation.wrapping_add(1);
                self.free.push(i as u32);
            }
        }
        self.len = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_get_remove() {
        let mut pool = Pool::new();
        let a = pool.insert("a");
        let b = pool.insert("b");
        assert_eq!(pool.len(), 2);
        assert_eq!(pool.get(a), Some(&"a"));
        assert_eq!(pool.remove(a), Some("a"));
        assert_eq!(pool.get(a), None);
        assert_eq!(pool.get(b), Some(&"b"));
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn stale_handle_does_not_alias_reused_slot() {
        let mut pool = Pool::new();
        let old = pool.insert(1);
        pool.remove(old);
        let new = pool.insert(2);
        assert_eq!(old.index(), new.index(), "slot should be recycled");
        assert_ne!(old.generation(), new.generation());
        assert_eq!(pool.get(old), None);
        assert_eq!(pool.get(new), Some(&2));
        assert_eq!(pool.remove(old), None, "stale remove must be a no-op");
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn double_remove_is_noop() {
        let mut pool = Pool::new();
        let h = pool.insert(5);
        assert_eq!(pool.remove(h), Some(5));
        assert_eq!(pool.remove(h), None);
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn iter_and_clear() {
        let mut pool = Pool::new();
        let handles: Vec<_> = (0..4).map(|i| pool.insert(i)).collect();
        pool.remove(handles[1]);
        let seen: Vec<_> = pool.iter().map(|(_, v)| *v).collect();
        assert_eq!(seen, vec![0, 2, 3]);
        pool.clear();
        assert!(pool.is_empty());
        assert!(handles.iter().all(|h| !pool.contains(*h)));
        // Slots are reusable after clear.
        let h = pool.insert(9);
        assert_eq!(pool.get(h), Some(&9));
    }

    #[test]
    fn get_mut_modifies_in_place() {
        let mut pool = Pool::new();
        let h = pool.insert(String::from("x"));
        pool.get_mut(h).unwrap().push('y');
        assert_eq!(pool.get(h).unwrap(), "xy");
    }
}
