//! Material records: what a section is drawn with, kept in one storage buffer the fragment shader
//! indexes with a number pushed per draw.
//!
//! Each section of an uploaded mesh gets one 48-byte record (see [`Surface::record`]); record 0 is
//! the neutral material of the proxy boxes. The buffer is host-visible and only ever appended to,
//! so no record that a submitted frame reads is rewritten. [`Materials::reset`] hands the whole
//! block (buffer, pool, set) over to be retired and starts a new one.

use crate::buffer::HostBuffer;
use crate::mesh::Surface;
use ash::vk;
use rustiny_vulkan::{Device, Result, VkResultExt, VulkanError};
use std::sync::Arc;

/// Bytes of one record: three `vec4`s.
pub(crate) const RECORD_SIZE: usize = 48;
/// Records a block holds (record 0 included).
pub(crate) const CAPACITY: u32 = 1 << 17;

/// Flag bits in a record.
const FLAG_UNLIT: f32 = 1.0;
const FLAG_MASK: f32 = 2.0;

impl Surface {
    /// The record the shader reads: `a` = tint rgb + alpha cutoff, `b` = emission rgb + normal
    /// scale, `c` = metallic, smoothness, flags (1 unlit, 2 has a metallic-smoothness texture), 0.
    pub(crate) fn record(&self) -> [u8; RECORD_SIZE] {
        let mut flags = 0.0;
        if self.unlit {
            flags += FLAG_UNLIT;
        }
        if self.metallic_map.is_some() {
            flags += FLAG_MASK;
        }
        let values = [
            self.tint[0],
            self.tint[1],
            self.tint[2],
            self.cutoff.clamp(0.0, 1.0),
            self.emission[0],
            self.emission[1],
            self.emission[2],
            self.normal_scale,
            self.metallic.clamp(0.0, 1.0),
            self.smoothness.clamp(0.0, 1.0),
            flags,
            0.0,
        ];
        let mut bytes = [0u8; RECORD_SIZE];
        for (chunk, value) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(values) {
            *chunk = value.to_ne_bytes();
        }
        bytes
    }
}

/// One buffer of records with the descriptor set that presents it.
pub(crate) struct MaterialBlock {
    device: Arc<Device>,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    buffer: HostBuffer,
    count: u32,
}

impl std::fmt::Debug for MaterialBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MaterialBlock")
            .field("count", &self.count)
            .finish_non_exhaustive()
    }
}

impl Drop for MaterialBlock {
    fn drop(&mut self) {
        // SAFETY: the owner guarantees the GPU no longer uses the set (retirement queue); the
        // buffer frees itself afterwards.
        unsafe {
            self.device.raw().destroy_descriptor_pool(self.pool, None);
        }
    }
}

/// The renderer's material records.
#[derive(Debug)]
pub(crate) struct Materials {
    device: Arc<Device>,
    pub(crate) layout: vk::DescriptorSetLayout,
    block: MaterialBlock,
}

impl Materials {
    pub(crate) fn new(device: &Arc<Device>) -> Result<Self> {
        let raw = device.raw();
        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
        let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY: plain object creation on a live device.
        let layout = unsafe { raw.create_descriptor_set_layout(&info, None) }
            .vk("vkCreateDescriptorSetLayout(materials)")?;
        match Self::new_block(device, layout) {
            Ok(block) => Ok(Self {
                device: device.clone(),
                layout,
                block,
            }),
            Err(e) => {
                // SAFETY: the layout is not used by anything.
                unsafe { raw.destroy_descriptor_set_layout(layout, None) };
                Err(e)
            }
        }
    }

    /// A buffer holding only the neutral record 0, and the set that presents it.
    fn new_block(device: &Arc<Device>, layout: vk::DescriptorSetLayout) -> Result<MaterialBlock> {
        let mut buffer = HostBuffer::zeroed(
            device,
            CAPACITY as usize * RECORD_SIZE,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "material-records",
        )?;
        buffer.write_at(0, &Surface::default().record())?;

        let raw = device.raw();
        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&sizes);
        // SAFETY: plain object creation on a live device.
        let pool = unsafe { raw.create_descriptor_pool(&pool_info, None) }
            .vk("vkCreateDescriptorPool(materials)")?;
        let layouts = [layout];
        let alloc = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        // SAFETY: the pool and layout are live and belong to this device.
        let set = unsafe { raw.allocate_descriptor_sets(&alloc) }
            .vk("vkAllocateDescriptorSets(materials)")
            .and_then(|sets| {
                sets.first().copied().ok_or_else(|| {
                    VulkanError::Allocator("a descriptor set allocation returned no set".into())
                })
            });
        let set = match set {
            Ok(set) => set,
            Err(e) => {
                // SAFETY: the pool is not used by anything.
                unsafe { raw.destroy_descriptor_pool(pool, None) };
                return Err(e);
            }
        };
        let buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(buffer.handle())
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buffer_info);
        // SAFETY: the set is new (no submitted work uses it) and the buffer is live.
        unsafe { raw.update_descriptor_sets(&[write], &[]) };
        Ok(MaterialBlock {
            device: device.clone(),
            pool,
            set,
            buffer,
            count: 1,
        })
    }

    /// The set to bind as set 1.
    pub(crate) fn set(&self) -> vk::DescriptorSet {
        self.block.set
    }

    /// Records in use (the neutral one included).
    pub(crate) fn len(&self) -> u32 {
        self.block.count
    }

    /// Whether `more` further records fit.
    pub(crate) fn has_room(&self, more: usize) -> bool {
        (self.block.count as usize).saturating_add(more) <= CAPACITY as usize
    }

    /// Append the records of `surfaces`; returns the index of the first.
    pub(crate) fn append<'a>(
        &mut self,
        surfaces: impl ExactSizeIterator<Item = &'a Surface>,
    ) -> Result<u32> {
        if !self.has_room(surfaces.len()) {
            return Err(VulkanError::InvalidInput(format!(
                "more than {CAPACITY} material records"
            )));
        }
        let first = self.block.count;
        let mut next = first;
        for surface in surfaces {
            self.block
                .buffer
                .write_at(next as usize * RECORD_SIZE, &surface.record())?;
            next += 1;
        }
        self.block.count = next;
        Ok(first)
    }

    /// Start over with only the neutral record. The old block is returned so the caller can keep
    /// it alive until the frames that read it have finished.
    pub(crate) fn reset(&mut self) -> Result<MaterialBlock> {
        let fresh = Self::new_block(&self.device, self.layout)?;
        Ok(std::mem::replace(&mut self.block, fresh))
    }
}

impl Drop for Materials {
    fn drop(&mut self) {
        // SAFETY: the owner waited for the device to be idle before dropping.
        unsafe {
            self.device
                .raw()
                .destroy_descriptor_set_layout(self.layout, None);
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn f(bytes: &[u8], i: usize) -> f32 {
        f32::from_ne_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap())
    }

    #[test]
    fn a_record_has_the_layout_the_shader_reads() {
        let surface = Surface {
            tint: [0.1, 0.2, 0.3],
            cutoff: 0.4,
            emission: [1.0, 2.0, 3.0],
            normal_scale: 1.5,
            metallic: 0.6,
            smoothness: 0.7,
            unlit: true,
            metallic_map: Some(9),
            ..Surface::default()
        };
        let bytes = surface.record();
        let values: Vec<f32> = (0..12).map(|i| f(&bytes, i)).collect();
        assert_eq!(
            values,
            vec![0.1, 0.2, 0.3, 0.4, 1.0, 2.0, 3.0, 1.5, 0.6, 0.7, 3.0, 0.0]
        );
    }

    #[test]
    fn the_default_surface_is_neutral_white() {
        let bytes = Surface::default().record();
        assert_eq!((f(&bytes, 0), f(&bytes, 1), f(&bytes, 2)), (1.0, 1.0, 1.0));
        assert_eq!(f(&bytes, 3), 0.0, "no cutoff");
        assert_eq!(f(&bytes, 9), 0.0, "matte");
        assert_eq!(f(&bytes, 10), 0.0, "no flags");
    }

    #[test]
    fn out_of_range_values_are_clamped() {
        let bytes = Surface {
            cutoff: 3.0,
            metallic: -1.0,
            smoothness: 5.0,
            ..Surface::default()
        }
        .record();
        assert_eq!((f(&bytes, 3), f(&bytes, 8), f(&bytes, 9)), (1.0, 0.0, 1.0));
    }
}
