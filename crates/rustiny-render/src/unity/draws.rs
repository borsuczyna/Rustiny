//! Per-object constant data (`UnityPerDraw`) kept on the GPU between frames, the way Unity's SRP
//! Batcher keeps it: each object's block is written once and drawn from many times, instead of
//! being copied again every frame into a per-frame ring that a large scene overflows.
//!
//! A block depends on the pass (its layout and the frame values written into it, the template)
//! and on the object (its matrices). A slot is therefore keyed by (pass, object). A slot is never
//! written while a submitted frame may read it: when a pass's template changes, its objects get
//! new slots further on, and when the buffer is full a bigger one replaces it (the old one is kept
//! until the frames that read it are done) and every slot is written again.

use crate::buffer::HostBuffer;
use ash::vk;
use rustiny_vulkan::{Device, FRAMES_IN_FLIGHT, Result};
use std::collections::HashMap;
use std::sync::Arc;

/// First size of the buffer; it doubles when full.
const FIRST_CAPACITY: u64 = 16 << 20;

/// Slots of draws that change every frame (the skybox follows the camera) are reused after this
/// many frames, when no frame in flight reads them any more.
pub(crate) const RING: u64 = FRAMES_IN_FLIGHT as u64 + 1;

/// Who a slot belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Owner {
    /// An instance of the scene (its index among the renderer's mesh instances): its block is
    /// written once.
    Instance(u32),
    /// A draw rewritten every frame (`index` within the frame), slot `frame % RING`.
    Frame { index: u32, ring: u64 },
}

pub(crate) struct DrawCache {
    device: Arc<Device>,
    buffer: HostBuffer,
    capacity: u64,
    cursor: u64,
    slots: HashMap<(u32, Owner), u32>,
    /// The template each pass's instance slots were written with.
    templates: HashMap<u32, Vec<u8>>,
    /// Changes whenever `buffer` is replaced: descriptor sets naming the old one must go.
    generation: u64,
    old: Vec<(u64, HostBuffer)>,
}

impl std::fmt::Debug for DrawCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DrawCache")
            .field("capacity", &self.capacity)
            .field("used", &self.cursor)
            .field("slots", &self.slots.len())
            .finish_non_exhaustive()
    }
}

impl DrawCache {
    pub(crate) fn new(device: &Arc<Device>) -> Result<Self> {
        Ok(Self {
            device: device.clone(),
            buffer: HostBuffer::zeroed(
                device,
                FIRST_CAPACITY as usize,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                "unity-per-draw",
            )?,
            capacity: FIRST_CAPACITY,
            cursor: 0,
            slots: HashMap::new(),
            templates: HashMap::new(),
            generation: 0,
            old: Vec::new(),
        })
    }

    pub(crate) fn handle(&self) -> vk::Buffer {
        self.buffer.handle()
    }

    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    /// Start of a frame: free replaced buffers no frame reads any more.
    pub(crate) fn begin_frame(&mut self, records: u64) {
        self.old
            .retain(|(stamp, _)| records <= stamp + FRAMES_IN_FLIGHT as u64 + 1);
    }

    /// The scene's instances changed: their blocks are written again (in new slots).
    pub(crate) fn forget_instances(&mut self) {
        self.slots
            .retain(|(_, owner), _| matches!(owner, Owner::Frame { .. }));
    }

    /// The template `pass` is drawn with this frame. A different one than its slots were written
    /// with makes them stale.
    pub(crate) fn use_template(&mut self, pass: u32, template: &[u8]) {
        if self.templates.get(&pass).is_some_and(|t| t == template) {
            return;
        }
        self.slots
            .retain(|(p, owner), _| *p != pass || matches!(owner, Owner::Frame { .. }));
        self.templates.insert(pass, template.to_vec());
    }

    /// Make room for `count` more blocks of `size` bytes, replacing the buffer by a bigger one
    /// when needed (then every slot is gone and the generation changes). `records` stamps the
    /// old buffer for retirement.
    pub(crate) fn reserve(
        &mut self,
        count: u64,
        size: u64,
        align: u64,
        records: u64,
    ) -> Result<()> {
        let need = count * size.div_ceil(align) * align + align;
        if self.cursor + need <= self.capacity {
            return Ok(());
        }
        let mut capacity = self.capacity * 2;
        while capacity < need {
            capacity *= 2;
        }
        let buffer = HostBuffer::zeroed(
            &self.device,
            capacity as usize,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            "unity-per-draw",
        )?;
        let old = std::mem::replace(&mut self.buffer, buffer);
        self.old.push((records, old));
        self.capacity = capacity;
        self.cursor = 0;
        self.slots.clear();
        self.templates.clear();
        self.generation += 1;
        Ok(())
    }

    /// The offset of `owner`'s block for `pass`, writing it with `fill` first when it has no
    /// slot yet or rewrites every frame. `None` when there is no room (call `reserve` first).
    pub(crate) fn slot(
        &mut self,
        pass: u32,
        owner: Owner,
        size: u64,
        align: u64,
        fill: impl FnOnce() -> Vec<u8>,
    ) -> Option<u32> {
        let rewrite = matches!(owner, Owner::Frame { .. });
        if let Some(&at) = self.slots.get(&(pass, owner))
            && !rewrite
        {
            return Some(at);
        }
        let at = match self.slots.get(&(pass, owner)) {
            Some(&at) => u64::from(at),
            None => {
                let start = self.cursor.div_ceil(align) * align;
                if start + size > self.capacity {
                    return None;
                }
                self.cursor = start + size;
                start
            }
        };
        let bytes = fill();
        self.buffer.write_at(at as usize, &bytes).ok()?;
        let at = u32::try_from(at).ok()?;
        self.slots.insert((pass, owner), at);
        Some(at)
    }
}
