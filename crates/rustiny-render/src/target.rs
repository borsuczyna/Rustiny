//! The offscreen colour and depth images the scene pass renders into.

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use rustiny_vulkan::{Device, Result, VkResultExt, VulkanError};
use std::sync::Arc;

/// The descriptor set layouts and sampler a target's sets are made for.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TargetLayouts {
    /// The final pass: scene colour, bloom, sampler.
    pub present: vk::DescriptorSetLayout,
    /// The bloom passes: source, lower level, sampler.
    pub bloom: vk::DescriptorSetLayout,
    /// Linear clamp.
    pub sampler: vk::Sampler,
}

/// Most bloom levels a target keeps (URP stops at 16 iterations).
const MAX_BLOOM_LEVELS: u32 = 16;

/// The descriptor sets of the bloom passes and of the final pass (see [`Target`]).
#[derive(Debug, Clone, Default)]
pub(crate) struct BloomSets {
    /// Reads the scene colour.
    pub prefilter: vk::DescriptorSet,
    /// `blur_h[i]` reads level `i - 1` going down (index 0 unused).
    pub blur_h: Vec<vk::DescriptorSet>,
    /// `blur_v[i]` reads the horizontal result of level `i`.
    pub blur_v: Vec<vk::DescriptorSet>,
    /// `upsample_from_down[i]`: level `i` with level `i + 1` of the way down below it (for the
    /// lowest level of a chain).
    pub upsample_from_down: Vec<vk::DescriptorSet>,
    /// `upsample_from_up[i]`: level `i` with the upsampled level `i + 1`.
    pub upsample_from_up: Vec<vk::DescriptorSet>,
    /// The final pass with bloom from the upsampled level 0 (chains of 2+ levels)...
    pub present_up: vk::DescriptorSet,
    /// ... or from level 0 going down (a chain of one level).
    pub present_down: vk::DescriptorSet,
}

/// The formats of a target's images.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TargetFormats {
    /// The scene colour (HDR, linear).
    pub hdr: vk::Format,
    /// Depth.
    pub depth: vk::Format,
    /// What the editor shows.
    pub display: vk::Format,
}

struct Image {
    image: vk::Image,
    view: vk::ImageView,
    allocation: Option<Allocation>,
}

/// The images of one viewport size: the HDR colour and the depth the scene is drawn into, their
/// copies for `_CameraOpaqueTexture` and `_CameraDepthTexture`, and the display image the final
/// pass writes (what the editor shows), with the descriptor set that pass reads the HDR image by.
pub(crate) struct Target {
    device: Arc<Device>,
    extent: vk::Extent2D,
    color: Image,
    depth: Image,
    /// Copy of the colour made after the opaque draws (`_CameraOpaqueTexture`).
    color_copy: Image,
    /// Copy of the depth made at the same moment (`_CameraDepthTexture`).
    depth_copy: Image,
    /// The colour converted for display (sRGB values in a UNORM image).
    display: Image,
    /// Bloom levels going down (`down[0]` is half size) and coming back up.
    bloom_down: Vec<Image>,
    bloom_up: Vec<Image>,
    bloom_extents: Vec<vk::Extent2D>,
    pool: vk::DescriptorPool,
    sets: BloomSets,
}

impl std::fmt::Debug for Target {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Target")
            .field("extent", &self.extent)
            .finish_non_exhaustive()
    }
}

fn create_image(
    device: &Device,
    extent: vk::Extent2D,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    aspect: vk::ImageAspectFlags,
    name: &'static str,
) -> Result<Image> {
    let raw = device.raw();
    let info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    // SAFETY: plain object creation on a live device.
    let image = unsafe { raw.create_image(&info, None) }.vk("vkCreateImage(target)")?;
    // SAFETY: the image was just created on this device.
    let requirements = unsafe { raw.get_image_memory_requirements(image) };
    let allocation = device
        .allocator()
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .allocate(&AllocationCreateDesc {
            name,
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::DedicatedImage(image),
        });
    let allocation = match allocation {
        Ok(a) => a,
        Err(e) => {
            // SAFETY: nothing uses the image yet.
            unsafe { raw.destroy_image(image, None) };
            return Err(VulkanError::Allocator(e.to_string()));
        }
    };
    let release = |allocation: Allocation, image: vk::Image| {
        let _ = device
            .allocator()
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .free(allocation);
        // SAFETY: nothing uses the image.
        unsafe { raw.destroy_image(image, None) };
    };
    // SAFETY: the allocation satisfies the image's requirements.
    let bound = unsafe { raw.bind_image_memory(image, allocation.memory(), allocation.offset()) }
        .vk("vkBindImageMemory(target)");
    if let Err(e) = bound {
        release(allocation, image);
        return Err(e);
    }
    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(aspect)
                .level_count(1)
                .layer_count(1),
        );
    // SAFETY: the image is bound and valid.
    match unsafe { raw.create_image_view(&view_info, None) }.vk("vkCreateImageView(target)") {
        Ok(view) => Ok(Image {
            image,
            view,
            allocation: Some(allocation),
        }),
        Err(e) => {
            release(allocation, image);
            Err(e)
        }
    }
}

impl Image {
    fn destroy(&mut self, device: &Device) {
        // SAFETY: the owner guarantees the GPU no longer uses these objects.
        unsafe {
            device.raw().destroy_image_view(self.view, None);
            device.raw().destroy_image(self.image, None);
        }
        if let Some(allocation) = self.allocation.take() {
            let _ = device
                .allocator()
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .free(allocation);
        }
    }
}

impl Target {
    pub(crate) fn new(
        device: &Arc<Device>,
        extent: vk::Extent2D,
        formats: TargetFormats,
        layouts: TargetLayouts,
    ) -> Result<Self> {
        let specs = [
            (
                formats.hdr,
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_SRC,
                vk::ImageAspectFlags::COLOR,
                "scene-color",
            ),
            (
                formats.depth,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
                vk::ImageAspectFlags::DEPTH,
                "scene-depth",
            ),
            (
                formats.hdr,
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                vk::ImageAspectFlags::COLOR,
                "scene-color-copy",
            ),
            (
                formats.depth,
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                vk::ImageAspectFlags::DEPTH,
                "scene-depth-copy",
            ),
            (
                formats.display,
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_SRC,
                vk::ImageAspectFlags::COLOR,
                "scene-display",
            ),
        ];
        let mut made: Vec<Image> = Vec::new();
        let release = |made: Vec<Image>| {
            for mut image in made {
                image.destroy(device);
            }
        };
        for (format, usage, aspect, name) in specs {
            match create_image(device, extent, format, usage, aspect, name) {
                Ok(image) => made.push(image),
                Err(e) => {
                    release(made);
                    return Err(e);
                }
            }
        }
        // Bloom levels: half size, then halving, as long as URP would go.
        let largest = (extent.width >> 1).max(extent.height >> 1).max(1) as f32;
        let levels = ((largest.log2() - 1.0).floor() as i64).clamp(1, i64::from(MAX_BLOOM_LEVELS));
        let mut bloom_extents = Vec::new();
        for level in 0..levels as u32 {
            bloom_extents.push(vk::Extent2D {
                width: (extent.width >> (level + 1)).max(1),
                height: (extent.height >> (level + 1)).max(1),
            });
        }
        let bloom_usage = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED;
        for name in ["bloom-down", "bloom-up"] {
            for size in &bloom_extents {
                match create_image(
                    device,
                    *size,
                    formats.hdr,
                    bloom_usage,
                    vk::ImageAspectFlags::COLOR,
                    name,
                ) {
                    Ok(image) => made.push(image),
                    Err(e) => {
                        release(made);
                        return Err(e);
                    }
                }
            }
        }
        let levels = bloom_extents.len();
        let set_count = 3 + 4 * levels as u32;
        let raw = device.raw();
        let sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(2 * set_count),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLER)
                .descriptor_count(set_count),
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(set_count)
            .pool_sizes(&sizes);
        // SAFETY: plain object creation on a live device.
        let pool = match unsafe { raw.create_descriptor_pool(&pool_info, None) }
            .vk("vkCreateDescriptorPool(target)")
        {
            Ok(pool) => pool,
            Err(e) => {
                release(made);
                return Err(e);
            }
        };
        let mut images = made.into_iter();
        let mut next = || images.next();
        let (Some(color), Some(depth), Some(color_copy), Some(depth_copy), Some(display)) =
            (next(), next(), next(), next(), next())
        else {
            // SAFETY: nothing uses the pool.
            unsafe { raw.destroy_descriptor_pool(pool, None) };
            return Err(VulkanError::Allocator("target images missing".into()));
        };
        let bloom_down: Vec<Image> = (0..levels).filter_map(|_| next()).collect();
        let bloom_up: Vec<Image> = (0..levels).filter_map(|_| next()).collect();
        let mut target = Self {
            device: device.clone(),
            extent,
            color,
            depth,
            color_copy,
            depth_copy,
            display,
            bloom_down,
            bloom_up,
            bloom_extents,
            pool,
            sets: BloomSets::default(),
        };
        target.sets = target.make_sets(layouts)?;
        Ok(target)
    }

    /// Allocate and write every set of the bloom and final passes.
    fn make_sets(&self, layouts: TargetLayouts) -> Result<BloomSets> {
        let raw = self.device.raw();
        let set = |layout: vk::DescriptorSetLayout,
                   first: vk::ImageView,
                   second: vk::ImageView|
         -> Result<vk::DescriptorSet> {
            let set_layouts = [layout];
            let alloc = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.pool)
                .set_layouts(&set_layouts);
            // SAFETY: the pool and layout are live and belong to this device.
            let set = unsafe { raw.allocate_descriptor_sets(&alloc) }
                .vk("vkAllocateDescriptorSets(target)")?[0];
            let image = |view| {
                [vk::DescriptorImageInfo::default()
                    .image_view(view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]
            };
            let (a, b) = (image(first), image(second));
            let sampler = [vk::DescriptorImageInfo::default().sampler(layouts.sampler)];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&a),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&b),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .image_info(&sampler),
            ];
            // SAFETY: the set is new and the views and sampler are live.
            unsafe { raw.update_descriptor_sets(&writes, &[]) };
            Ok(set)
        };
        let down = |i: usize| self.bloom_down[i].view;
        let up = |i: usize| self.bloom_up[i].view;
        let levels = self.bloom_down.len();
        let mut sets = BloomSets {
            prefilter: set(layouts.bloom, self.color.view, self.color.view)?,
            present_up: set(layouts.present, self.color.view, up(0))?,
            present_down: set(layouts.present, self.color.view, down(0))?,
            ..BloomSets::default()
        };
        for i in 0..levels {
            let previous = down(i.saturating_sub(1));
            sets.blur_h.push(set(layouts.bloom, previous, previous)?);
            sets.blur_v.push(set(layouts.bloom, up(i), up(i))?);
            let below = (i + 1).min(levels - 1);
            sets.upsample_from_down
                .push(set(layouts.bloom, down(i), down(below))?);
            sets.upsample_from_up
                .push(set(layouts.bloom, down(i), up(below))?);
        }
        Ok(sets)
    }

    /// The descriptor sets of the post-processing passes.
    pub(crate) fn sets(&self) -> &BloomSets {
        &self.sets
    }

    /// Bloom levels: images going down, coming up, and their sizes.
    pub(crate) fn bloom_levels(&self) -> usize {
        self.bloom_down.len()
    }

    pub(crate) fn bloom_down(&self, level: usize) -> (vk::Image, vk::ImageView) {
        (self.bloom_down[level].image, self.bloom_down[level].view)
    }

    pub(crate) fn bloom_up(&self, level: usize) -> (vk::Image, vk::ImageView) {
        (self.bloom_up[level].image, self.bloom_up[level].view)
    }

    pub(crate) fn bloom_extent(&self, level: usize) -> vk::Extent2D {
        self.bloom_extents[level]
    }

    pub(crate) fn display_image(&self) -> vk::Image {
        self.display.image
    }

    pub(crate) fn display_view(&self) -> vk::ImageView {
        self.display.view
    }

    pub(crate) fn color_copy_image(&self) -> vk::Image {
        self.color_copy.image
    }

    pub(crate) fn color_copy_view(&self) -> vk::ImageView {
        self.color_copy.view
    }

    pub(crate) fn depth_copy_image(&self) -> vk::Image {
        self.depth_copy.image
    }

    pub(crate) fn depth_copy_view(&self) -> vk::ImageView {
        self.depth_copy.view
    }

    pub(crate) fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    pub(crate) fn color_image(&self) -> vk::Image {
        self.color.image
    }

    pub(crate) fn color_view(&self) -> vk::ImageView {
        self.color.view
    }

    pub(crate) fn depth_image(&self) -> vk::Image {
        self.depth.image
    }

    pub(crate) fn depth_view(&self) -> vk::ImageView {
        self.depth.view
    }
}

impl Drop for Target {
    fn drop(&mut self) {
        self.color.destroy(&self.device);
        self.depth.destroy(&self.device);
        self.color_copy.destroy(&self.device);
        self.depth_copy.destroy(&self.device);
        self.display.destroy(&self.device);
        for image in self.bloom_down.iter_mut().chain(self.bloom_up.iter_mut()) {
            image.destroy(&self.device);
        }
        // SAFETY: the owner guarantees the GPU no longer uses the set (the target is retired
        // until its frames are done).
        unsafe { self.device.raw().destroy_descriptor_pool(self.pool, None) };
    }
}
