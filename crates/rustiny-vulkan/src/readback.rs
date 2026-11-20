//! GPU -> CPU readback of the presented frame (used for screenshots and rendering tests).

use crate::device::Device;
use crate::error::{Result, VkResultExt, VulkanError};
use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};

/// A captured frame in tightly packed 8-bit RGBA, top row first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Screenshot {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// `width * height * 4` bytes, `R, G, B, A` order.
    pub rgba: Vec<u8>,
}

/// Host-visible buffer a swapchain image is copied into.
pub(crate) struct ReadbackBuffer {
    buffer: vk::Buffer,
    allocation: Option<Allocation>,
    extent: vk::Extent2D,
    bgra: bool,
}

impl ReadbackBuffer {
    pub(crate) fn new(device: &Device, extent: vk::Extent2D, format: vk::Format) -> Result<Self> {
        let bgra = match format {
            vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB => true,
            vk::Format::R8G8B8A8_UNORM | vk::Format::R8G8B8A8_SRGB => false,
            other => {
                return Err(VulkanError::WindowSystem(format!(
                    "screenshot unsupported for swapchain format {other:?}"
                )));
            }
        };
        let size = u64::from(extent.width) * u64::from(extent.height) * 4;
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let raw = device.raw();
        // SAFETY: plain object creation on a live device.
        let buffer = unsafe { raw.create_buffer(&info, None) }.vk("vkCreateBuffer(readback)")?;
        // SAFETY: buffer was just created on this device.
        let requirements = unsafe { raw.get_buffer_memory_requirements(buffer) };
        let allocation = device
            .allocator()
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .allocate(&AllocationCreateDesc {
                name: "screenshot-readback",
                requirements,
                location: MemoryLocation::GpuToCpu,
                linear: true,
                // A dedicated allocation guarantees offset 0, which keeps cache invalidation
                // trivially aligned.
                allocation_scheme: AllocationScheme::DedicatedBuffer(buffer),
            });
        let allocation = match allocation {
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
                .vk("vkBindBufferMemory(readback)");
        if let Err(e) = bound {
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
            buffer,
            allocation: Some(allocation),
            extent,
            bgra,
        })
    }

    /// Record `vkCmdCopyImageToBuffer`. `image` must be in `TRANSFER_SRC_OPTIMAL`.
    pub(crate) fn record_copy(&self, device: &Device, cmd: vk::CommandBuffer, image: vk::Image) {
        let region = vk::BufferImageCopy::default()
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1),
            )
            .image_extent(vk::Extent3D {
                width: self.extent.width,
                height: self.extent.height,
                depth: 1,
            });
        // SAFETY: the command buffer is recording; the image is in the layout the caller promised.
        unsafe {
            device.raw().cmd_copy_image_to_buffer(
                cmd,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.buffer,
                &[region],
            );
        }
        // Make the transfer write visible to the host once the fence signals.
        let barrier = vk::BufferMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(self.buffer)
            .size(vk::WHOLE_SIZE);
        let barriers = [barrier];
        let dependency = vk::DependencyInfo::default().buffer_memory_barriers(&barriers);
        // SAFETY: recording command buffer.
        unsafe { device.raw().cmd_pipeline_barrier2(cmd, &dependency) };
    }

    /// Read the pixels (the submission that wrote them must have completed) and free the buffer.
    pub(crate) fn finish(mut self, device: &Device) -> Result<Screenshot> {
        let raw = device.raw();
        let Some(allocation) = self.allocation.take() else {
            return Err(VulkanError::Allocator(
                "readback buffer already consumed".into(),
            ));
        };
        // SAFETY: `memory()` is only unsafe because the handle must not be freed by the caller;
        // we only pass it to a query-like call while the allocation is alive.
        let memory = unsafe { allocation.memory() };
        let range = vk::MappedMemoryRange::default()
            .memory(memory)
            .offset(allocation.offset())
            .size(vk::WHOLE_SIZE);
        // SAFETY: the memory is mapped host-visible memory of this device; invalidation is a no-op
        // on coherent memory and required on non-coherent memory.
        unsafe { raw.invalidate_mapped_memory_ranges(&[range]) }
            .vk("vkInvalidateMappedMemoryRanges")?;

        let result = match allocation.mapped_slice() {
            Some(bytes) => {
                let mut rgba = bytes
                    [..(self.extent.width as usize * self.extent.height as usize * 4)]
                    .to_vec();
                if self.bgra {
                    for px in rgba.as_chunks_mut::<4>().0.iter_mut() {
                        px.swap(0, 2);
                    }
                }
                // The swapchain has no meaningful alpha; force opaque so viewers show the image.
                for px in rgba.as_chunks_mut::<4>().0.iter_mut() {
                    px[3] = 255;
                }
                Ok(Screenshot {
                    width: self.extent.width,
                    height: self.extent.height,
                    rgba,
                })
            }
            None => Err(VulkanError::Allocator(
                "readback memory is not host mapped".into(),
            )),
        };
        let _ = device
            .allocator()
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .free(allocation);
        // SAFETY: the submission using the buffer has completed (caller waited on the fence).
        unsafe { raw.destroy_buffer(self.buffer, None) };
        self.buffer = vk::Buffer::null();
        result
    }
}

impl Drop for ReadbackBuffer {
    fn drop(&mut self) {
        // Reached only if `finish` was not called (error path); the device is still alive because
        // the owning renderer holds an Arc to it, but the GPU may still be using the buffer, so
        // leak rather than risk a use-after-free. This is a rare diagnostic-only path.
        if self.allocation.is_some() {
            tracing::warn!("screenshot readback buffer dropped without being read; leaking it");
        }
    }
}
