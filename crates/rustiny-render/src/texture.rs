//! Material textures (albedo, normal, metallic-smoothness): images with mip chains, uploaded in
//! the frame's command buffer, and the descriptor sets that present three of them to a section.
//!
//! Textures are added from any point in the program with [`Textures::add`] (which only creates
//! the image and a staging buffer); the copy into the image is recorded by [`Textures::flush`]
//! at the start of the next scene pass, a few megabytes per frame, so streaming a few thousand
//! textures never stalls a frame. Until a texture's copy has been recorded, draws that name it use
//! a 1x1 white texture instead.
//!
//! A section draws with one descriptor set holding its albedo, normal and metallic-smoothness
//! images. Sets are made on first use for each combination of *effective* textures (the texture
//! itself once its copy has been recorded, the slot's default before) and never updated afterwards,
//! so no set that a submitted frame uses is ever rewritten. Defaults: white (albedo), a flat normal
//! and "no metal, full smoothness" (metallic-smoothness). One shared immutable sampler (linear, repeating, trilinear)
//! lives in the set layout.

use crate::buffer::HostBuffer;
use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use rustiny_vulkan::{Device, Result, VkResultExt, VulkanError};
use std::collections::HashMap;
use std::sync::Arc;

/// Format of colour textures: the sampler decodes sRGB to linear, which the lighting works in.
const SRGB_FORMAT: vk::Format = vk::Format::R8G8B8A8_SRGB;
/// Format of data textures (normals, metallic-smoothness): sampled as stored.
const LINEAR_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;
/// Textures a section binds: albedo, normal, metallic-smoothness.
pub(crate) const SLOTS: usize = 3;
/// Stands for "the slot's default" in a set's cache key; never a real texture key.
const DEFAULT_KEY: u32 = u32::MAX;
/// Largest edge accepted.
const MAX_EDGE: u32 = 8192;
/// Descriptor sets per pool.
const SETS_PER_POOL: u32 = 512;

/// One uploaded (or waiting to be uploaded) texture.
struct Texture {
    device: Arc<Device>,
    image: vk::Image,
    view: vk::ImageView,
    allocation: Option<Allocation>,
    mip_levels: u32,
    /// Staging data and the copies that move it, until the copy is recorded.
    upload: Option<(HostBuffer, Vec<vk::BufferImageCopy>)>,
    /// The copy has been recorded: the image is in `SHADER_READ_ONLY_OPTIMAL` from here on.
    ready: bool,
    bytes: u64,
    /// Order of creation, so uploads happen in the order textures were added.
    serial: u64,
}

impl Drop for Texture {
    fn drop(&mut self) {
        // SAFETY: the owner guarantees the GPU no longer uses these objects (retirement queue);
        // destroying a null view is a no-op.
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

/// Textures and descriptor pools dropped by the renderer, kept until the GPU is done with them.
pub(crate) struct RetiredTextures {
    device: Arc<Device>,
    items: Vec<Texture>,
    pools: Vec<vk::DescriptorPool>,
}

impl std::fmt::Debug for RetiredTextures {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetiredTextures")
            .field("textures", &self.items.len())
            .finish_non_exhaustive()
    }
}

impl Drop for RetiredTextures {
    fn drop(&mut self) {
        self.items.clear();
        for pool in self.pools.drain(..) {
            // SAFETY: as above; destroying a pool frees the sets allocated from it.
            unsafe { self.device.raw().destroy_descriptor_pool(pool, None) };
        }
    }
}

/// What the renderer holds for textures.
pub(crate) struct Textures {
    device: Arc<Device>,
    pub(crate) layout: vk::DescriptorSetLayout,
    sampler: vk::Sampler,
    /// Pools the textures' sets come from; the last one has `pool_free` sets left.
    pools: Vec<vk::DescriptorPool>,
    pool_free: u32,
    items: HashMap<u32, Texture>,
    /// Defaults of the three slots: white (sRGB), flat normal, no metal and full smoothness (linear).
    defaults: Vec<Texture>,
    /// Sets by effective textures (see the module docs).
    sets: HashMap<[u32; SLOTS], vk::DescriptorSet>,
    bytes: u64,
    next_serial: u64,
    /// Changes whenever an image that sets may refer to goes away (a key replaced, all cleared).
    epoch: u64,
}

impl std::fmt::Debug for Textures {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Textures")
            .field("count", &self.items.len())
            .field("bytes", &self.bytes)
            .finish_non_exhaustive()
    }
}

/// Reject data that does not describe a mip chain of the given size.
pub(crate) fn validate(width: u32, height: u32, mips: &[Vec<u8>]) -> Result<()> {
    let bad = |why: String| Err(VulkanError::InvalidInput(why));
    if width == 0 || height == 0 || width > MAX_EDGE || height > MAX_EDGE {
        return bad(format!("texture size {width}x{height} is not supported"));
    }
    if mips.is_empty() || mips.len() > 14 {
        return bad(format!(
            "a texture needs 1 to 14 mip levels, got {}",
            mips.len()
        ));
    }
    for (level, data) in mips.iter().enumerate() {
        let (w, h) = mip_size(width, height, level as u32);
        let expected = u64::from(w) * u64::from(h) * 4;
        if data.len() as u64 != expected {
            return bad(format!(
                "mip {level} of a {width}x{height} texture should be {expected} bytes, got {}",
                data.len()
            ));
        }
    }
    Ok(())
}

/// Size of mip `level`.
pub(crate) fn mip_size(width: u32, height: u32, level: u32) -> (u32, u32) {
    ((width >> level).max(1), (height >> level).max(1))
}

/// Build the mip chain of an RGBA8 image by averaging 2x2 blocks (in the stored, sRGB-encoded
/// values, which is slightly dark for high contrast but cheap). `base` must be
/// `width * height * 4` bytes; the chain stops at 1x1 or after 14 levels.
pub fn mip_chain(width: u32, height: u32, base: Vec<u8>) -> Vec<Vec<u8>> {
    let mut chain = vec![base];
    let (mut w, mut h) = (width.max(1), height.max(1));
    while (w > 1 || h > 1) && chain.len() < 14 {
        let (nw, nh) = mip_size(w, h, 1);
        let mut next = vec![0u8; nw as usize * nh as usize * 4];
        if let Some(prev) = chain.last() {
            let texel = |x: u32, y: u32, c: usize| -> u32 {
                let (x, y) = (x.min(w - 1), y.min(h - 1));
                prev.get((y as usize * w as usize + x as usize) * 4 + c)
                    .copied()
                    .map_or(0, u32::from)
            };
            for y in 0..nh {
                for x in 0..nw {
                    for c in 0..4 {
                        let sum = texel(2 * x, 2 * y, c)
                            + texel(2 * x + 1, 2 * y, c)
                            + texel(2 * x, 2 * y + 1, c)
                            + texel(2 * x + 1, 2 * y + 1, c);
                        next[(y as usize * nw as usize + x as usize) * 4 + c] =
                            ((sum + 2) / 4) as u8;
                    }
                }
            }
        }
        chain.push(next);
        (w, h) = (nw, nh);
    }
    chain
}

fn new_pool(device: &Device, sets: u32) -> Result<vk::DescriptorPool> {
    let sizes = [
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(sets * SLOTS as u32),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::SAMPLER)
            .descriptor_count(sets),
    ];
    let info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(sets)
        .pool_sizes(&sizes);
    // SAFETY: plain object creation on a live device.
    unsafe { device.raw().create_descriptor_pool(&info, None) }
        .vk("vkCreateDescriptorPool(material)")
}

impl Textures {
    pub(crate) fn new(device: &Arc<Device>) -> Result<Self> {
        let raw = device.raw();
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .max_lod(vk::LOD_CLAMP_NONE);
        // SAFETY: plain object creation on a live device.
        let sampler =
            unsafe { raw.create_sampler(&sampler_info, None) }.vk("vkCreateSampler(material)")?;
        let samplers = [sampler];
        let mut bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..SLOTS as u32)
            .map(|binding| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(binding)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            })
            .collect();
        bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(SLOTS as u32)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .immutable_samplers(&samplers),
        );
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY: `samplers` outlives the call; plain object creation.
        let layout = match unsafe { raw.create_descriptor_set_layout(&layout_info, None) }
            .vk("vkCreateDescriptorSetLayout(material)")
        {
            Ok(layout) => layout,
            Err(e) => {
                // SAFETY: the sampler is not used by anything yet.
                unsafe { raw.destroy_sampler(sampler, None) };
                return Err(e);
            }
        };
        let mut textures = Self {
            device: device.clone(),
            layout,
            sampler,
            pools: Vec::new(),
            pool_free: 0,
            items: HashMap::new(),
            defaults: Vec::new(),
            sets: HashMap::new(),
            bytes: 0,
            next_serial: 0,
            epoch: 0,
        };
        // On error `textures` drops and releases everything created so far.
        // White albedo, a flat normal, and a mask with no metal and full smoothness (so that a
        // material whose mask has not arrived yet is a plain non-metal).
        for (data, format, name) in [
            (vec![255u8; 4], SRGB_FORMAT, "albedo-default"),
            (vec![128, 128, 255, 255], LINEAR_FORMAT, "normal-default"),
            (vec![0, 0, 0, 255], LINEAR_FORMAT, "mask-default"),
        ] {
            let texture = textures.create(1, 1, &[data], format, name)?;
            textures.defaults.push(texture);
        }
        Ok(textures)
    }

    /// A set from the current pool, starting a new pool when it is used up.
    fn allocate_set(&mut self) -> Result<vk::DescriptorSet> {
        if self.pool_free == 0 {
            let pool = new_pool(&self.device, SETS_PER_POOL)?;
            self.pools.push(pool);
            self.pool_free = SETS_PER_POOL;
        }
        let Some(&pool) = self.pools.last() else {
            return Err(VulkanError::Allocator("no descriptor pool".into()));
        };
        let layouts = [self.layout];
        let info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        // SAFETY: the pool and layout are live and belong to this device.
        let sets = unsafe { self.device.raw().allocate_descriptor_sets(&info) }
            .vk("vkAllocateDescriptorSets(material)")?;
        let set = sets.first().copied().ok_or_else(|| {
            VulkanError::Allocator("a descriptor set allocation returned no set".into())
        })?;
        self.pool_free -= 1;
        Ok(set)
    }

    /// Create the image, its view and the staging buffer. Nothing is copied yet.
    fn create(
        &mut self,
        width: u32,
        height: u32,
        mips: &[Vec<u8>],
        format: vk::Format,
        name: &'static str,
    ) -> Result<Texture> {
        validate(width, height, mips)?;
        let device = self.device.clone();
        let raw = device.raw();
        let mip_levels = mips.len() as u32;

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(mip_levels)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        // SAFETY: plain object creation on a live device.
        let image = unsafe { raw.create_image(&image_info, None) }.vk("vkCreateImage(material)")?;
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
        // SAFETY: the allocation is live; the memory handle is only used to bind below, while the
        // texture owns the allocation.
        let (memory, offset) = (unsafe { allocation.memory() }, allocation.offset());
        // From here on a `Texture` owns image and allocation, so errors release them by dropping.
        let mut texture = Texture {
            device: device.clone(),
            image,
            view: vk::ImageView::null(),
            allocation: Some(allocation),
            mip_levels,
            upload: None,
            ready: false,
            bytes: mips.iter().map(|m| m.len() as u64).sum(),
            serial: self.next_serial,
        };
        self.next_serial += 1;
        // SAFETY: the allocation satisfies the image's requirements.
        unsafe { raw.bind_image_memory(image, memory, offset) }
            .vk("vkBindImageMemory(material)")?;
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(mip_levels)
                    .layer_count(1),
            );
        // SAFETY: the image is bound and valid.
        texture.view =
            unsafe { raw.create_image_view(&view_info, None) }.vk("vkCreateImageView(material)")?;

        let mut staging = Vec::with_capacity(texture.bytes as usize);
        let mut regions = Vec::with_capacity(mips.len());
        for (level, data) in mips.iter().enumerate() {
            let (w, h) = mip_size(width, height, level as u32);
            regions.push(
                vk::BufferImageCopy::default()
                    .buffer_offset(staging.len() as u64)
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(level as u32)
                            .layer_count(1),
                    )
                    .image_extent(vk::Extent3D {
                        width: w,
                        height: h,
                        depth: 1,
                    }),
            );
            staging.extend_from_slice(data);
        }
        let buffer = HostBuffer::with_data(
            &device,
            &staging,
            vk::BufferUsageFlags::TRANSFER_SRC,
            "material-staging",
        )?;
        texture.upload = Some((buffer, regions));
        Ok(texture)
    }

    /// Add (or replace) the texture under `key`: sRGB for colours, linear for data such as
    /// normals. Returns what it replaced, to be retired.
    pub(crate) fn add(
        &mut self,
        key: u32,
        width: u32,
        height: u32,
        mips: &[Vec<u8>],
        srgb: bool,
    ) -> Result<Option<RetiredTextures>> {
        validate(width, height, mips)?;
        if key == DEFAULT_KEY {
            return Err(VulkanError::InvalidInput(
                "texture key u32::MAX is reserved".into(),
            ));
        }
        let format = if srgb { SRGB_FORMAT } else { LINEAR_FORMAT };
        let texture = self.create(width, height, mips, format, "material-texture")?;
        self.bytes += texture.bytes;
        let old = self.items.insert(key, texture);
        // Cached sets may name the old texture, so they go too (a replaced key is rare: the
        // editor never reuses one).
        Ok(old.map(|old| {
            self.epoch += 1;
            self.bytes -= old.bytes;
            let pools = self.drop_sets();
            RetiredTextures {
                device: self.device.clone(),
                items: vec![old],
                pools,
            }
        }))
    }

    /// Forget every cached set and hand over the pools they came from.
    fn drop_sets(&mut self) -> Vec<vk::DescriptorPool> {
        self.sets.clear();
        self.pool_free = 0;
        std::mem::take(&mut self.pools)
    }

    /// Whether a texture is stored under `key` (uploaded or not).
    pub(crate) fn contains(&self, key: u32) -> bool {
        self.items.contains_key(&key)
    }

    /// Number of textures stored (the defaults not counted).
    pub(crate) fn len(&self) -> usize {
        self.items.len()
    }

    /// Bytes of texel data stored.
    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Textures added whose copy has not been recorded yet.
    pub(crate) fn waiting(&self) -> usize {
        self.items.values().filter(|t| !t.ready).count()
    }

    /// Whether the copy of the texture under `key` has been recorded.
    pub(crate) fn is_ready(&self, key: u32) -> bool {
        self.items.get(&key).is_some_and(|t| t.ready)
    }

    /// The image view of the texture under `key`, once its copy has been recorded.
    pub(crate) fn view_of(&self, key: u32) -> Option<vk::ImageView> {
        self.items.get(&key).filter(|t| t.ready).map(|t| t.view)
    }

    /// Changes whenever a view handed out before may have been destroyed.
    pub(crate) fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Drop every texture and set. The defaults stay.
    pub(crate) fn clear(&mut self) -> RetiredTextures {
        self.bytes = 0;
        self.epoch += 1;
        let pools = self.drop_sets();
        RetiredTextures {
            device: self.device.clone(),
            items: self.items.drain().map(|(_, t)| t).collect(),
            pools,
        }
    }

    /// The descriptor set presenting `keys` (albedo, normal, metallic-smoothness; `None`, or a
    /// texture whose copy has not been recorded yet, stands for the slot's default). Call after
    /// [`flush`](Self::flush) in the same recording.
    pub(crate) fn set_for(&mut self, keys: [Option<u32>; SLOTS]) -> Result<vk::DescriptorSet> {
        let mut effective = [DEFAULT_KEY; SLOTS];
        for (slot, key) in keys.iter().enumerate() {
            if let Some(key) = *key
                && self.items.get(&key).is_some_and(|t| t.ready)
            {
                effective[slot] = key;
            }
        }
        if let Some(&set) = self.sets.get(&effective) {
            return Ok(set);
        }
        let set = self.allocate_set()?;
        let view = |slot: usize| {
            let texture = if effective[slot] == DEFAULT_KEY {
                self.defaults.get(slot)
            } else {
                self.items.get(&effective[slot])
            };
            texture.map_or(vk::ImageView::null(), |t| t.view)
        };
        let infos: [vk::DescriptorImageInfo; SLOTS] = std::array::from_fn(|slot| {
            vk::DescriptorImageInfo::default()
                .image_view(view(slot))
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        });
        let writes: Vec<vk::WriteDescriptorSet> = infos
            .iter()
            .enumerate()
            .map(|(slot, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(slot as u32)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(std::slice::from_ref(info))
            })
            .collect();
        // SAFETY: the set is new (no submitted work uses it) and the views are live.
        unsafe { self.device.raw().update_descriptor_sets(&writes, &[]) };
        self.sets.insert(effective, set);
        Ok(set)
    }

    /// Record the copies of waiting textures into `cmd` (outside a rendering scope), up to
    /// `budget_bytes` (always at least one texture). Returns the staging buffers, which must stay
    /// alive until the submission has finished.
    pub(crate) fn flush(&mut self, cmd: vk::CommandBuffer, budget_bytes: u64) -> Vec<HostBuffer> {
        let Self {
            device,
            items,
            defaults,
            ..
        } = self;
        let mut used: u64 = defaults.iter().filter(|t| !t.ready).map(|t| t.bytes).sum();
        let mut waiting: Vec<(u64, u32)> = items
            .iter()
            .filter(|(_, t)| !t.ready && t.upload.is_some())
            .map(|(key, t)| (t.serial, *key))
            .collect();
        waiting.sort_unstable();
        let mut chosen: Vec<u32> = Vec::new();
        for (_, key) in waiting {
            let bytes = items.get(&key).map_or(0, |t| t.bytes);
            if used > 0 && used + bytes > budget_bytes {
                break;
            }
            used += bytes;
            chosen.push(key);
        }

        let mut jobs: Vec<&mut Texture> = defaults.iter_mut().filter(|t| !t.ready).collect();
        let mut by_key: HashMap<u32, &mut Texture> = items
            .iter_mut()
            .filter(|(key, _)| chosen.contains(key))
            .map(|(key, t)| (*key, t))
            .collect();
        for key in &chosen {
            if let Some(texture) = by_key.remove(key) {
                jobs.push(texture);
            }
        }
        if jobs.is_empty() {
            return Vec::new();
        }

        let raw = device.raw();
        let barrier = |texture: &Texture,
                       (src_stage, src_access, old): (
            vk::PipelineStageFlags2,
            vk::AccessFlags2,
            vk::ImageLayout,
        ),
                       (dst_stage, dst_access, new): (
            vk::PipelineStageFlags2,
            vk::AccessFlags2,
            vk::ImageLayout,
        )| {
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(src_stage)
                .src_access_mask(src_access)
                .dst_stage_mask(dst_stage)
                .dst_access_mask(dst_access)
                .old_layout(old)
                .new_layout(new)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(texture.image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(texture.mip_levels)
                        .layer_count(1),
                )
        };
        let to_copy: Vec<_> = jobs
            .iter()
            .map(|t| {
                barrier(
                    t,
                    (
                        vk::PipelineStageFlags2::NONE,
                        vk::AccessFlags2::NONE,
                        vk::ImageLayout::UNDEFINED,
                    ),
                    (
                        vk::PipelineStageFlags2::COPY,
                        vk::AccessFlags2::TRANSFER_WRITE,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    ),
                )
            })
            .collect();
        let dependency = vk::DependencyInfo::default().image_memory_barriers(&to_copy);
        // SAFETY: `cmd` is recording outside a rendering scope (caller's contract) and the images
        // belong to this device.
        unsafe { raw.cmd_pipeline_barrier2(cmd, &dependency) };
        let mut staging = Vec::with_capacity(jobs.len());
        for texture in &mut jobs {
            if let Some((buffer, regions)) = texture.upload.take() {
                // SAFETY: the image is in TRANSFER_DST_OPTIMAL after the barrier above, the
                // buffer holds the data the regions describe, and the caller keeps the buffer
                // alive until the submission has finished.
                unsafe {
                    raw.cmd_copy_buffer_to_image(
                        cmd,
                        buffer.handle(),
                        texture.image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &regions,
                    );
                }
                staging.push(buffer);
            }
            texture.ready = true;
        }
        let to_sample: Vec<_> = jobs
            .iter()
            .map(|t| {
                barrier(
                    t,
                    (
                        vk::PipelineStageFlags2::COPY,
                        vk::AccessFlags2::TRANSFER_WRITE,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    ),
                    (
                        vk::PipelineStageFlags2::FRAGMENT_SHADER,
                        vk::AccessFlags2::SHADER_SAMPLED_READ,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    ),
                )
            })
            .collect();
        let dependency = vk::DependencyInfo::default().image_memory_barriers(&to_sample);
        // SAFETY: as above.
        unsafe { raw.cmd_pipeline_barrier2(cmd, &dependency) };
        staging
    }
}

impl Drop for Textures {
    fn drop(&mut self) {
        self.items.clear();
        self.defaults.clear();
        let raw = self.device.raw();
        // SAFETY: the owner waited for the device to be idle before dropping (see
        // `SceneRenderer::drop`), so nothing uses these objects any more.
        unsafe {
            for pool in self.pools.drain(..) {
                raw.destroy_descriptor_pool(pool, None);
            }
            raw.destroy_descriptor_set_layout(self.layout, None);
            raw.destroy_sampler(self.sampler, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mip_sizes_halve_down_to_one() {
        assert_eq!(mip_size(256, 64, 0), (256, 64));
        assert_eq!(mip_size(256, 64, 3), (32, 8));
        assert_eq!(mip_size(256, 64, 7), (2, 1));
        assert_eq!(mip_size(256, 64, 9), (1, 1));
    }

    #[test]
    fn mip_chain_averages_blocks_down_to_one_texel() {
        // 2x2 with channel 0 = 0, 100, 200, 100 -> one texel of 100.
        let mut base = vec![0u8; 16];
        for (i, v) in [0u8, 100, 200, 100].into_iter().enumerate() {
            base[i * 4] = v;
            base[i * 4 + 3] = 255;
        }
        let chain = mip_chain(2, 2, base);
        assert_eq!(chain.len(), 2);
        assert_eq!(chain[1], vec![100, 0, 0, 255]);
        // Odd and non-square sizes follow the Vulkan rule and validate.
        let chain = mip_chain(5, 3, vec![7; 5 * 3 * 4]);
        assert_eq!(chain.len(), 3, "5x3 -> 2x1 -> 1x1");
        assert!(validate(5, 3, &chain).is_ok());
        assert!(chain.iter().all(|m| m.iter().all(|&b| b == 7)));
        // A 1x1 image is its own chain.
        assert_eq!(mip_chain(1, 1, vec![1, 2, 3, 4]).len(), 1);
    }

    fn chain(width: u32, height: u32, levels: u32) -> Vec<Vec<u8>> {
        (0..levels)
            .map(|l| {
                let (w, h) = mip_size(width, height, l);
                vec![0; (w * h * 4) as usize]
            })
            .collect()
    }

    #[test]
    fn a_consistent_chain_is_accepted() {
        assert!(validate(4, 4, &chain(4, 4, 1)).is_ok());
        assert!(validate(4, 2, &chain(4, 2, 3)).is_ok());
        assert!(validate(1, 1, &chain(1, 1, 1)).is_ok());
    }

    #[test]
    fn inconsistent_chains_are_rejected() {
        assert!(validate(0, 4, &chain(4, 4, 1)).is_err(), "zero width");
        assert!(validate(9000, 4, &chain(4, 4, 1)).is_err(), "too wide");
        assert!(validate(4, 4, &[]).is_err(), "no mips");
        assert!(validate(4, 4, &chain(4, 4, 15)).is_err(), "too many mips");
        let mut short = chain(4, 4, 2);
        short[1].pop();
        assert!(validate(4, 4, &short).is_err(), "wrong mip size");
        assert!(validate(4, 4, &chain(8, 8, 1)).is_err(), "wrong base size");
    }
}
