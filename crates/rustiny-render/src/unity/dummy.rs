//! One-texel images that stand in for every texture a Unity shader asks for and nothing provides:
//! the material's missing maps (white, black, grey, flat normal) and the engine textures it does
//! not have yet (depth, opaque colour, reflection probes, 3D and array textures).
//!
//! They are filled with a clear command in the first recorded pass, so they need no staging data.

use super::TextureFallback;
use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use rustiny_shader::ResourceKind;
use rustiny_vulkan::{Device, Result, VkResultExt, VulkanError};
use std::sync::Arc;

const FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;

/// A 1x1 image, its view and its memory.
struct Image {
    device: Arc<Device>,
    image: vk::Image,
    view: vk::ImageView,
    allocation: Option<Allocation>,
    layers: u32,
    colour: [f32; 4],
}

impl Drop for Image {
    fn drop(&mut self) {
        // SAFETY: the owner guarantees the GPU no longer uses these objects.
        unsafe {
            self.device.raw().destroy_image_view(self.view, None);
            self.device.raw().destroy_image(self.image, None);
        }
        if let Some(allocation) = self.allocation.take() {
            let _ = self
                .device
                .allocator()
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .free(allocation);
        }
    }
}

impl Image {
    fn new(
        device: &Arc<Device>,
        dimension: vk::ImageType,
        view_type: vk::ImageViewType,
        layers: u32,
        colour: [f32; 4],
    ) -> Result<Self> {
        let raw = device.raw();
        let flags = if view_type == vk::ImageViewType::CUBE {
            vk::ImageCreateFlags::CUBE_COMPATIBLE
        } else {
            vk::ImageCreateFlags::empty()
        };
        let info = vk::ImageCreateInfo::default()
            .flags(flags)
            .image_type(dimension)
            .format(FORMAT)
            .extent(vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(layers)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        // SAFETY: plain object creation on a live device.
        let image = unsafe { raw.create_image(&info, None) }.vk("vkCreateImage(dummy)")?;
        // SAFETY: the image was just created on this device.
        let requirements = unsafe { raw.get_image_memory_requirements(image) };
        let allocation = device
            .allocator()
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .allocate(&AllocationCreateDesc {
                name: "unity-dummy-image",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            });
        let allocation = match allocation {
            Ok(a) => a,
            Err(e) => {
                // SAFETY: nothing uses the image yet.
                unsafe { raw.destroy_image(image, None) };
                return Err(VulkanError::Allocator(e.to_string()));
            }
        };
        // SAFETY: the allocation is live and owned by the `Image` built next.
        let (memory, offset) = (unsafe { allocation.memory() }, allocation.offset());
        let mut this = Self {
            device: device.clone(),
            image,
            view: vk::ImageView::null(),
            allocation: Some(allocation),
            layers,
            colour,
        };
        // SAFETY: the allocation satisfies the image's requirements.
        unsafe { raw.bind_image_memory(image, memory, offset) }.vk("vkBindImageMemory(dummy)")?;
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(view_type)
            .format(FORMAT)
            .subresource_range(range(layers));
        // SAFETY: the image is bound and valid.
        this.view =
            unsafe { raw.create_image_view(&view_info, None) }.vk("vkCreateImageView(dummy)")?;
        Ok(this)
    }
}

fn range(layers: u32) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(layers)
}

/// The stand-in images.
pub(crate) struct Dummies {
    white: Image,
    black: Image,
    grey: Image,
    bump: Image,
    cube: Image,
    volume: Image,
    array: Image,
    ready: bool,
}

impl std::fmt::Debug for Dummies {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dummies")
            .field("ready", &self.ready)
            .finish_non_exhaustive()
    }
}

impl Dummies {
    pub(crate) fn new(device: &Arc<Device>) -> Result<Self> {
        let flat = |colour| {
            Image::new(
                device,
                vk::ImageType::TYPE_2D,
                vk::ImageViewType::TYPE_2D,
                1,
                colour,
            )
        };
        Ok(Self {
            white: flat([1.0; 4])?,
            black: flat([0.0, 0.0, 0.0, 1.0])?,
            grey: flat([0.5; 4])?,
            bump: flat([0.5, 0.5, 1.0, 1.0])?,
            cube: Image::new(
                device,
                vk::ImageType::TYPE_2D,
                vk::ImageViewType::CUBE,
                6,
                [0.0, 0.0, 0.0, 1.0],
            )?,
            volume: Image::new(
                device,
                vk::ImageType::TYPE_3D,
                vk::ImageViewType::TYPE_3D,
                1,
                [0.0, 0.0, 0.0, 1.0],
            )?,
            array: Image::new(
                device,
                vk::ImageType::TYPE_2D,
                vk::ImageViewType::TYPE_2D_ARRAY,
                1,
                [0.0, 0.0, 0.0, 1.0],
            )?,
            ready: false,
        })
    }

    fn all(&self) -> [&Image; 7] {
        [
            &self.white,
            &self.black,
            &self.grey,
            &self.bump,
            &self.cube,
            &self.volume,
            &self.array,
        ]
    }

    /// The view that stands in for a texture of `kind` (the material's `fallback` applies to
    /// plain 2D textures; the others are black).
    pub(crate) fn view(&self, kind: ResourceKind, fallback: TextureFallback) -> vk::ImageView {
        match kind {
            ResourceKind::TextureCube => self.cube.view,
            ResourceKind::Texture3D => self.volume.view,
            ResourceKind::Texture2DArray => self.array.view,
            _ => match fallback {
                TextureFallback::White => self.white.view,
                TextureFallback::Black => self.black.view,
                TextureFallback::Grey => self.grey.view,
                TextureFallback::Bump => self.bump.view,
            },
        }
    }

    /// Fill the images (once) and make them readable by shaders. `cmd` must be recording outside
    /// a rendering scope.
    pub(crate) fn record_init(&mut self, cmd: vk::CommandBuffer) {
        if self.ready {
            return;
        }
        let raw = self.white.device.raw();
        let barrier = |image: &Image, old, new, src, dst| {
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(src)
                .src_access_mask(if old == vk::ImageLayout::UNDEFINED {
                    vk::AccessFlags2::NONE
                } else {
                    vk::AccessFlags2::TRANSFER_WRITE
                })
                .dst_stage_mask(dst)
                .dst_access_mask(if new == vk::ImageLayout::TRANSFER_DST_OPTIMAL {
                    vk::AccessFlags2::TRANSFER_WRITE
                } else {
                    vk::AccessFlags2::SHADER_SAMPLED_READ
                })
                .old_layout(old)
                .new_layout(new)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image.image)
                .subresource_range(range(image.layers))
        };
        let to_clear: Vec<_> = self
            .all()
            .iter()
            .map(|image| {
                barrier(
                    image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::PipelineStageFlags2::NONE,
                    vk::PipelineStageFlags2::CLEAR,
                )
            })
            .collect();
        let dependency = vk::DependencyInfo::default().image_memory_barriers(&to_clear);
        // SAFETY: `cmd` is recording outside a rendering scope (caller's contract) and the images
        // belong to this device.
        unsafe { raw.cmd_pipeline_barrier2(cmd, &dependency) };
        for image in self.all() {
            let colour = vk::ClearColorValue {
                float32: image.colour,
            };
            // SAFETY: the image is in TRANSFER_DST_OPTIMAL after the barrier above.
            unsafe {
                raw.cmd_clear_color_image(
                    cmd,
                    image.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &colour,
                    &[range(image.layers)],
                );
            }
        }
        let to_read: Vec<_> = self
            .all()
            .iter()
            .map(|image| {
                barrier(
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::PipelineStageFlags2::CLEAR,
                    vk::PipelineStageFlags2::FRAGMENT_SHADER
                        | vk::PipelineStageFlags2::VERTEX_SHADER,
                )
            })
            .collect();
        let dependency = vk::DependencyInfo::default().image_memory_barriers(&to_read);
        // SAFETY: as above.
        unsafe { raw.cmd_pipeline_barrier2(cmd, &dependency) };
        self.ready = true;
    }
}
