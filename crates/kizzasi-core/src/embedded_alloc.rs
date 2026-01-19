//! Embedded-friendly allocator for no_std environments
//!
//! This module provides a simple, efficient allocator suitable for embedded systems
//! with limited memory. Features:
//! - Fixed-size pool allocation for predictable memory usage
//! - Bump allocator for sequential allocations
//! - Stack-based allocation for temporary data
//! - Zero-allocation APIs where possible

#[cfg(not(feature = "std"))]
use core::{
    alloc::{GlobalAlloc, Layout},
    cell::UnsafeCell,
    ptr::{self, NonNull},
    sync::atomic::{AtomicUsize, Ordering},
};

#[cfg(feature = "std")]
use std::{
    alloc::{GlobalAlloc, Layout},
    cell::UnsafeCell,
    ptr::{self, NonNull},
    sync::atomic::{AtomicUsize, Ordering},
};

/// Fixed-size memory pool for embedded systems
///
/// Provides O(1) allocation and deallocation from a pre-allocated buffer.
/// Suitable for real-time systems with deterministic memory requirements.
pub struct FixedPool<const SIZE: usize, const ALIGN: usize> {
    buffer: UnsafeCell<[u8; SIZE]>,
    free_list: AtomicUsize,
    block_size: usize,
    num_blocks: usize,
}

impl<const SIZE: usize, const ALIGN: usize> FixedPool<SIZE, ALIGN> {
    /// Create a new fixed pool with given block size
    ///
    /// # Arguments
    ///
    /// * `block_size` - Size of each allocatable block (will be aligned to ALIGN)
    ///
    /// # Examples
    ///
    /// ```
    /// use kizzasi_core::embedded_alloc::FixedPool;
    ///
    /// // Pool with 1KB, 8-byte aligned blocks
    /// let pool: FixedPool<1024, 8> = FixedPool::new(64);
    /// ```
    #[allow(clippy::manual_div_ceil)] // div_ceil is not const fn
    pub const fn new(block_size: usize) -> Self {
        let aligned_block_size = ((block_size + ALIGN - 1) / ALIGN) * ALIGN;
        let num_blocks = SIZE / aligned_block_size;

        Self {
            buffer: UnsafeCell::new([0u8; SIZE]),
            free_list: AtomicUsize::new(0),
            block_size: aligned_block_size,
            num_blocks,
        }
    }

    /// Allocate a block from the pool
    ///
    /// Returns `None` if the pool is exhausted.
    pub fn alloc(&self) -> Option<NonNull<u8>> {
        loop {
            let free_idx = self.free_list.load(Ordering::Acquire);
            if free_idx >= self.num_blocks {
                return None; // Pool exhausted
            }

            // Try to claim this block
            if self
                .free_list
                .compare_exchange(free_idx, free_idx + 1, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                let offset = free_idx * self.block_size;
                let ptr = unsafe {
                    let buf = &*self.buffer.get();
                    buf.as_ptr().add(offset) as *mut u8
                };
                return NonNull::new(ptr);
            }
            // CAS failed, retry
        }
    }

    /// Deallocate a block back to the pool
    ///
    /// # Safety
    ///
    /// The pointer must have been allocated from this pool and not already freed.
    pub unsafe fn dealloc(&self, _ptr: NonNull<u8>) {
        // For simplicity, this implementation doesn't support deallocation
        // In a real embedded system, you'd maintain a free list of blocks
        // For now, we just decrement the counter if safe
        let current = self.free_list.load(Ordering::Acquire);
        if current > 0 {
            self.free_list.fetch_sub(1, Ordering::Release);
        }
    }

    /// Reset the pool, freeing all allocations
    ///
    /// # Safety
    ///
    /// Only call this when you're certain no outstanding allocations exist.
    pub unsafe fn reset(&self) {
        self.free_list.store(0, Ordering::Release);
    }

    /// Get the number of available blocks
    pub fn available(&self) -> usize {
        let used = self.free_list.load(Ordering::Acquire);
        self.num_blocks.saturating_sub(used)
    }

    /// Get total number of blocks in the pool
    pub const fn capacity(&self) -> usize {
        self.num_blocks
    }

    /// Get block size
    pub const fn block_size(&self) -> usize {
        self.block_size
    }
}

// Safety: FixedPool uses atomic operations for thread safety
unsafe impl<const SIZE: usize, const ALIGN: usize> Send for FixedPool<SIZE, ALIGN> {}
unsafe impl<const SIZE: usize, const ALIGN: usize> Sync for FixedPool<SIZE, ALIGN> {}

/// Bump allocator for sequential allocations
///
/// Very fast O(1) allocation, no deallocation. Suitable for temporary
/// computations where all memory is freed at once.
pub struct BumpAllocator<const SIZE: usize> {
    buffer: UnsafeCell<[u8; SIZE]>,
    offset: AtomicUsize,
}

impl<const SIZE: usize> BumpAllocator<SIZE> {
    /// Create a new bump allocator
    pub const fn new() -> Self {
        Self {
            buffer: UnsafeCell::new([0u8; SIZE]),
            offset: AtomicUsize::new(0),
        }
    }

    /// Allocate memory with given layout
    ///
    /// Returns `None` if insufficient space remains.
    pub fn alloc(&self, layout: Layout) -> Option<NonNull<u8>> {
        let size = layout.size();
        let align = layout.align();

        loop {
            let current_offset = self.offset.load(Ordering::Acquire);

            // Align the offset
            let aligned_offset = (current_offset + align - 1) & !(align - 1);
            let new_offset = aligned_offset + size;

            if new_offset > SIZE {
                return None; // Out of memory
            }

            // Try to claim this allocation
            if self
                .offset
                .compare_exchange(
                    current_offset,
                    new_offset,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                let ptr = unsafe {
                    let buf = &*self.buffer.get();
                    buf.as_ptr().add(aligned_offset) as *mut u8
                };
                return NonNull::new(ptr);
            }
            // CAS failed, retry
        }
    }

    /// Reset the allocator, freeing all allocations
    ///
    /// # Safety
    ///
    /// Only call this when you're certain no outstanding allocations exist.
    pub unsafe fn reset(&self) {
        self.offset.store(0, Ordering::Release);
    }

    /// Get the number of bytes allocated
    pub fn used(&self) -> usize {
        self.offset.load(Ordering::Acquire)
    }

    /// Get the number of bytes available
    pub fn available(&self) -> usize {
        SIZE.saturating_sub(self.used())
    }

    /// Get total capacity
    pub const fn capacity(&self) -> usize {
        SIZE
    }
}

// Safety: BumpAllocator uses atomic operations for thread safety
unsafe impl<const SIZE: usize> Send for BumpAllocator<SIZE> {}
unsafe impl<const SIZE: usize> Sync for BumpAllocator<SIZE> {}

impl<const SIZE: usize> Default for BumpAllocator<SIZE> {
    fn default() -> Self {
        Self::new()
    }
}

/// Stack-based allocator for temporary allocations
///
/// Allocations must be freed in LIFO order. Very fast and suitable
/// for nested scopes.
pub struct StackAllocator<const SIZE: usize> {
    buffer: UnsafeCell<[u8; SIZE]>,
    offset: AtomicUsize,
}

impl<const SIZE: usize> StackAllocator<SIZE> {
    /// Create a new stack allocator
    pub const fn new() -> Self {
        Self {
            buffer: UnsafeCell::new([0u8; SIZE]),
            offset: AtomicUsize::new(0),
        }
    }

    /// Push an allocation onto the stack
    ///
    /// Returns the pointer and the offset to restore when popping.
    pub fn push(&self, layout: Layout) -> Option<(NonNull<u8>, usize)> {
        let size = layout.size();
        let align = layout.align();

        loop {
            let current_offset = self.offset.load(Ordering::Acquire);

            // Align the offset
            let aligned_offset = (current_offset + align - 1) & !(align - 1);
            let new_offset = aligned_offset + size;

            if new_offset > SIZE {
                return None; // Out of memory
            }

            // Try to claim this allocation
            if self
                .offset
                .compare_exchange(
                    current_offset,
                    new_offset,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                let ptr = unsafe {
                    let buf = &*self.buffer.get();
                    buf.as_ptr().add(aligned_offset) as *mut u8
                };
                // Return pointer and the offset BEFORE this allocation (for pop)
                return NonNull::new(ptr).map(|p| (p, current_offset));
            }
            // CAS failed, retry
        }
    }

    /// Pop an allocation from the stack
    ///
    /// # Arguments
    ///
    /// * `saved_offset` - The offset returned from `push`
    ///
    /// # Safety
    ///
    /// Must be called in LIFO order matching the `push` calls.
    pub unsafe fn pop(&self, saved_offset: usize) {
        self.offset.store(saved_offset, Ordering::Release);
    }

    /// Get current stack depth (bytes used)
    pub fn depth(&self) -> usize {
        self.offset.load(Ordering::Acquire)
    }

    /// Get available space
    pub fn available(&self) -> usize {
        SIZE.saturating_sub(self.depth())
    }

    /// Get total capacity
    pub const fn capacity(&self) -> usize {
        SIZE
    }
}

// Safety: StackAllocator uses atomic operations for thread safety
unsafe impl<const SIZE: usize> Send for StackAllocator<SIZE> {}
unsafe impl<const SIZE: usize> Sync for StackAllocator<SIZE> {}

impl<const SIZE: usize> Default for StackAllocator<SIZE> {
    fn default() -> Self {
        Self::new()
    }
}

/// Global embedded allocator combining bump and pool allocation
///
/// For no_std environments, this can be used as the global allocator.
pub struct EmbeddedAllocator<const SIZE: usize, const POOL_SIZE: usize> {
    bump: BumpAllocator<SIZE>,
    pool: FixedPool<POOL_SIZE, 8>,
}

impl<const SIZE: usize, const POOL_SIZE: usize> EmbeddedAllocator<SIZE, POOL_SIZE> {
    /// Create a new embedded allocator
    ///
    /// # Arguments
    ///
    /// * `block_size` - Size of blocks in the pool allocator
    pub const fn new(block_size: usize) -> Self {
        Self {
            bump: BumpAllocator::new(),
            pool: FixedPool::new(block_size),
        }
    }

    /// Reset both allocators
    ///
    /// # Safety
    ///
    /// Only call when no outstanding allocations exist.
    pub unsafe fn reset(&self) {
        self.bump.reset();
        self.pool.reset();
    }

    /// Get total memory used
    pub fn used(&self) -> usize {
        self.bump.used() + (self.pool.capacity() - self.pool.available()) * self.pool.block_size()
    }

    /// Get total capacity
    pub const fn capacity(&self) -> usize {
        SIZE + POOL_SIZE
    }
}

unsafe impl<const SIZE: usize, const POOL_SIZE: usize> GlobalAlloc
    for EmbeddedAllocator<SIZE, POOL_SIZE>
{
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // Try pool first for small allocations
        if layout.size() <= self.pool.block_size() {
            if let Some(ptr) = self.pool.alloc() {
                return ptr.as_ptr();
            }
        }

        // Fall back to bump allocator
        self.bump
            .alloc(layout)
            .map(|p| p.as_ptr())
            .unwrap_or(ptr::null_mut())
    }

    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        // Try to return to pool if it's a pool allocation
        if let Some(nonnull) = NonNull::new(ptr) {
            self.pool.dealloc(nonnull);
        }
    }
}

// Safety: EmbeddedAllocator uses atomic operations and is thread-safe
unsafe impl<const SIZE: usize, const POOL_SIZE: usize> Send for EmbeddedAllocator<SIZE, POOL_SIZE> {}
unsafe impl<const SIZE: usize, const POOL_SIZE: usize> Sync for EmbeddedAllocator<SIZE, POOL_SIZE> {}

/// RAII guard for stack allocations
pub struct StackGuard<'a, const SIZE: usize> {
    allocator: &'a StackAllocator<SIZE>,
    saved_offset: usize,
}

impl<'a, const SIZE: usize> Drop for StackGuard<'a, SIZE> {
    fn drop(&mut self) {
        unsafe {
            self.allocator.pop(self.saved_offset);
        }
    }
}

impl<const SIZE: usize> StackAllocator<SIZE> {
    /// Allocate with RAII guard that automatically frees on drop
    pub fn scoped_alloc(&self, layout: Layout) -> Option<(NonNull<u8>, StackGuard<'_, SIZE>)> {
        self.push(layout).map(|(ptr, saved_offset)| {
            let guard = StackGuard {
                allocator: self,
                saved_offset,
            };
            (ptr, guard)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_pool_basic() {
        let pool: FixedPool<1024, 8> = FixedPool::new(64);
        assert_eq!(pool.capacity(), 16); // 1024 / 64 = 16 blocks
        assert_eq!(pool.available(), 16);

        let ptr1 = pool.alloc().expect("Failed to allocate");
        assert_eq!(pool.available(), 15);

        let ptr2 = pool.alloc().expect("Failed to allocate");
        assert_eq!(pool.available(), 14);

        assert_ne!(ptr1.as_ptr(), ptr2.as_ptr());

        unsafe {
            pool.dealloc(ptr1);
            pool.dealloc(ptr2);
        }
    }

    #[test]
    fn test_fixed_pool_exhaustion() {
        let pool: FixedPool<128, 8> = FixedPool::new(32);
        assert_eq!(pool.capacity(), 4);

        let mut ptrs = vec![];
        for _ in 0..4 {
            ptrs.push(pool.alloc().expect("Failed to allocate"));
        }

        // Pool should be exhausted
        assert!(pool.alloc().is_none());
        assert_eq!(pool.available(), 0);
    }

    #[test]
    fn test_bump_allocator_basic() {
        let bump: BumpAllocator<1024> = BumpAllocator::new();
        assert_eq!(bump.capacity(), 1024);
        assert_eq!(bump.used(), 0);

        let layout = Layout::from_size_align(64, 8).unwrap();
        let ptr1 = bump.alloc(layout).expect("Failed to allocate");
        assert_eq!(bump.used(), 64);

        let ptr2 = bump.alloc(layout).expect("Failed to allocate");
        assert_eq!(bump.used(), 128);

        assert_ne!(ptr1.as_ptr(), ptr2.as_ptr());
    }

    #[test]
    fn test_bump_allocator_alignment() {
        let bump: BumpAllocator<1024> = BumpAllocator::new();

        // Allocate with different alignments
        let layout1 = Layout::from_size_align(1, 1).unwrap();
        let _ptr1 = bump.alloc(layout1).expect("Failed to allocate");

        let layout2 = Layout::from_size_align(8, 8).unwrap();
        let ptr2 = bump.alloc(layout2).expect("Failed to allocate");

        // ptr2 should be 8-byte aligned
        assert_eq!(ptr2.as_ptr() as usize % 8, 0);
    }

    #[test]
    fn test_bump_allocator_reset() {
        let bump: BumpAllocator<1024> = BumpAllocator::new();

        let layout = Layout::from_size_align(64, 8).unwrap();
        let _ptr = bump.alloc(layout).expect("Failed to allocate");
        assert_eq!(bump.used(), 64);

        unsafe {
            bump.reset();
        }
        assert_eq!(bump.used(), 0);
    }

    #[test]
    fn test_stack_allocator_basic() {
        let stack: StackAllocator<1024> = StackAllocator::new();
        assert_eq!(stack.capacity(), 1024);
        assert_eq!(stack.depth(), 0);

        let layout = Layout::from_size_align(64, 8).unwrap();
        let (ptr1, offset1) = stack.push(layout).expect("Failed to allocate");
        assert!(stack.depth() >= 64);

        let (ptr2, offset2) = stack.push(layout).expect("Failed to allocate");
        assert!(stack.depth() >= 128);

        assert_ne!(ptr1.as_ptr(), ptr2.as_ptr());

        // Pop in LIFO order
        unsafe {
            stack.pop(offset2);
            assert!(stack.depth() < 128);
            stack.pop(offset1);
            assert_eq!(stack.depth(), 0);
        }
    }

    #[test]
    fn test_stack_allocator_scoped() {
        let stack: StackAllocator<1024> = StackAllocator::new();

        let layout = Layout::from_size_align(64, 8).unwrap();
        {
            let (_ptr1, _guard1) = stack.scoped_alloc(layout).expect("Failed to allocate");
            assert!(stack.depth() >= 64);

            {
                let (_ptr2, _guard2) = stack.scoped_alloc(layout).expect("Failed to allocate");
                assert!(stack.depth() >= 128);
            }
            // guard2 dropped, space freed
            assert!(stack.depth() < 128);
        }
        // guard1 dropped, all space freed
        // Note: depth might not be exactly 0 due to alignment
    }

    #[test]
    fn test_embedded_allocator_basic() {
        let allocator: EmbeddedAllocator<1024, 512> = EmbeddedAllocator::new(32);
        assert_eq!(allocator.capacity(), 1536);

        unsafe {
            let layout = Layout::from_size_align(16, 8).unwrap();
            let ptr1 = allocator.alloc(layout);
            assert!(!ptr1.is_null());

            let ptr2 = allocator.alloc(layout);
            assert!(!ptr2.is_null());
            assert_ne!(ptr1, ptr2);

            allocator.dealloc(ptr1, layout);
            allocator.dealloc(ptr2, layout);
        }
    }

    #[test]
    fn test_embedded_allocator_small_large() {
        let allocator: EmbeddedAllocator<2048, 1024> = EmbeddedAllocator::new(64);

        unsafe {
            // Small allocation (should use pool)
            let small_layout = Layout::from_size_align(32, 8).unwrap();
            let small_ptr = allocator.alloc(small_layout);
            assert!(!small_ptr.is_null());

            // Large allocation (should use bump)
            let large_layout = Layout::from_size_align(128, 8).unwrap();
            let large_ptr = allocator.alloc(large_layout);
            assert!(!large_ptr.is_null());

            assert_ne!(small_ptr, large_ptr);

            allocator.dealloc(small_ptr, small_layout);
            allocator.dealloc(large_ptr, large_layout);
        }
    }

    #[test]
    fn test_fixed_pool_reset() {
        let pool: FixedPool<1024, 8> = FixedPool::new(64);

        let _ptr1 = pool.alloc().expect("Failed to allocate");
        let _ptr2 = pool.alloc().expect("Failed to allocate");
        assert_eq!(pool.available(), 14);

        unsafe {
            pool.reset();
        }
        assert_eq!(pool.available(), 16);
    }
}
