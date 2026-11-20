//! Host-visible, persistently mapped buffers.
//!
//! Used for the unit box and the per-instance data. Host-visible memory keeps the first slice
//! simple (no staging copies); on a discrete GPU the shader reads it over PCIe, which is fine for
//! the few hundred kilobytes involved. A device-local copy is a later optimisation.

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use rustiny_vulkan::{Device, Result, VkResultExt, VulkanError};
use std::sync::Arc;

/// A `vk::Buffer` with its own mapped allocation.
pub(crate) struct HostBuffer {
    device: Arc<Device>,
    buffer: vk::Buffer,
    allocation: Option<Allocation>,
    size: u64,
}

impl std::fmt::Debug for HostBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostBuffer")
            .field("size", &self.size)
            .finish_non_exhaustive()
    }
}

impl HostBuffer {
    /// Create a buffer of `size` bytes (at least 4) and fill it with `bytes`.
    pub(crate) fn with_data(
        device: &Arc<Device>,
        bytes: &[u8],
        usage: vk::BufferUsageFlags,
        name: &'static str,
    ) -> Result<Self> {
        let size = (bytes.len() as u64).max(4);
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let raw = device.raw();
        // SAFETY: plain object creation on a live device.
        let buffer = unsafe { raw.create_buffer(&info, None) }.vk("vkCreateBuffer(host)")?;
        // SAFETY: the buffer was just created on this device.
        let requirements = unsafe { raw.get_buffer_memory_requirements(buffer) };
        let allocation = device
            .allocator()
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .allocate(&AllocationCreateDesc {
                name,
                requirements,
                location: MemoryLocation::CpuToGpu,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            });
        let mut allocation = match allocation {
            Ok(a) => a,
            Err(e) => {
                // SAFETY: nothing uses the buffer yet.
                unsafe { raw.destroy_buffer(buffer, None) };
                return Err(VulkanError::Allocator(e.to_string()));
            }
        };
        // SAFETY: the allocation satisfies the buffer's requirements.
        let bound =
            unsafe { raw.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()) }
                .vk("vkBindBufferMemory(host)");
        let written = bound.and_then(|()| match allocation.mapped_slice_mut() {
            Some(slice) if slice.len() >= bytes.len() => {
                slice[..bytes.len()].copy_from_slice(bytes);
                Ok(())
            }
            _ => Err(VulkanError::Allocator(
                "host buffer memory is not mapped".into(),
            )),
        });
        if let Err(e) = written {
            let _ = device
                .allocator()
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .free(allocation);
            // SAFETY: nothing uses the buffer.
            unsafe { raw.destroy_buffer(buffer, None) };
            return Err(e);
        }
        Ok(Self {
            device: device.clone(),
            buffer,
            allocation: Some(allocation),
            size,
        })
    }

    /// Create a zero-filled buffer of `size` bytes.
    pub(crate) fn zeroed(
        device: &Arc<Device>,
        size: usize,
        usage: vk::BufferUsageFlags,
        name: &'static str,
    ) -> Result<Self> {
        Self::with_data(device, &vec![0u8; size], usage, name)
    }

    /// Overwrite `bytes.len()` bytes at `offset`. The caller makes sure the GPU is not reading
    /// that range (the renderer only writes records no submitted frame refers to).
    pub(crate) fn write_at(&mut self, offset: usize, bytes: &[u8]) -> Result<()> {
        let out_of_range = || VulkanError::InvalidInput("write past the end of a buffer".into());
        let end = offset.checked_add(bytes.len()).ok_or_else(out_of_range)?;
        match self.allocation.as_mut().and_then(|a| a.mapped_slice_mut()) {
            Some(slice) => slice
                .get_mut(offset..end)
                .ok_or_else(out_of_range)?
                .copy_from_slice(bytes),
            None => {
                return Err(VulkanError::Allocator(
                    "host buffer memory is not mapped".into(),
                ));
            }
        }
        Ok(())
    }

    /// The Vulkan handle.
    pub(crate) fn handle(&self) -> vk::Buffer {
        self.buffer
    }
}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        if let Some(allocation) = self.allocation.take() {
            let _ = self
                .device
                .allocator()
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .free(allocation);
        }
        // SAFETY: the owner guarantees the GPU no longer uses the buffer (see `SceneRenderer`'s
        // retirement queue and its `Drop`, which waits for the device to be idle).
        unsafe { self.device.raw().destroy_buffer(self.buffer, None) };
    }
}
