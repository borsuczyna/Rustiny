//! GPU time of the scene pass, measured with timestamp queries.
//!
//! Each frame slot owns a begin/end query pair. A slot's result is read back the next time the
//! slot is used (at least `FRAMES_IN_FLIGHT` records later, by which time the frame that wrote it
//! has finished), so reading never waits.

use ash::vk;
use rustiny_vulkan::{Device, FRAMES_IN_FLIGHT, Result, VkResultExt};
use std::sync::Arc;

/// Timestamp query pool with one begin/end pair per frame slot.
pub(crate) struct GpuTimer {
    device: Arc<Device>,
    pool: vk::QueryPool,
    /// Nanoseconds per timestamp tick.
    period_ns: f64,
    /// Mask of the valid bits of a timestamp.
    mask: u64,
    /// Which slots hold a result that has not been read yet.
    written: [bool; FRAMES_IN_FLIGHT],
}

impl std::fmt::Debug for GpuTimer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuTimer")
            .field("period_ns", &self.period_ns)
            .finish_non_exhaustive()
    }
}

impl GpuTimer {
    /// `None` when the graphics queue cannot write timestamps.
    pub(crate) fn new(device: &Arc<Device>) -> Result<Option<Self>> {
        let instance = device.instance().raw();
        // SAFETY: query calls on a live instance and one of its physical devices.
        let (limits, families) = unsafe {
            (
                instance
                    .get_physical_device_properties(device.physical())
                    .limits,
                instance.get_physical_device_queue_family_properties(device.physical()),
            )
        };
        let bits = families
            .get(device.families().graphics as usize)
            .map_or(0, |f| f.timestamp_valid_bits);
        if bits == 0 || limits.timestamp_period <= 0.0 {
            return Ok(None);
        }
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(2 * FRAMES_IN_FLIGHT as u32);
        // SAFETY: plain object creation on a live device.
        let pool =
            unsafe { device.raw().create_query_pool(&info, None) }.vk("vkCreateQueryPool")?;
        Ok(Some(Self {
            device: device.clone(),
            pool,
            period_ns: f64::from(limits.timestamp_period),
            mask: if bits >= 64 {
                u64::MAX
            } else {
                (1u64 << bits) - 1
            },
            written: [false; FRAMES_IN_FLIGHT],
        }))
    }

    /// Read the previous result of the slot used by record number `record`, if it finished.
    /// Returns milliseconds.
    pub(crate) fn collect(&mut self, record: u64) -> Option<f32> {
        let slot = (record % FRAMES_IN_FLIGHT as u64) as usize;
        if !std::mem::take(&mut self.written[slot]) {
            return None;
        }
        // Per query: the value and its availability flag.
        let mut data = [[0u64; 2]; 2];
        // SAFETY: `data` holds two queries of (value, availability) as `TYPE_64 |
        // WITH_AVAILABILITY` writes them; the queries belong to this pool.
        let result = unsafe {
            self.device.raw().get_query_pool_results(
                self.pool,
                (slot * 2) as u32,
                &mut data,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY,
            )
        };
        // NOT_READY is reported through the availability flags below.
        let _ = result;
        if data[0][1] == 0 || data[1][1] == 0 {
            return None;
        }
        let ticks = (data[1][0].wrapping_sub(data[0][0])) & self.mask;
        Some((ticks as f64 * self.period_ns / 1.0e6) as f32)
    }

    /// Reset the slot's queries and write the start timestamp.
    pub(crate) fn begin(&mut self, cmd: vk::CommandBuffer, record: u64) {
        let slot = (record % FRAMES_IN_FLIGHT as u64) as usize;
        let first = (slot * 2) as u32;
        // SAFETY: `cmd` is recording outside a rendering scope; the queries are this pool's and
        // not in use by a pending frame (see the module docs).
        unsafe {
            self.device
                .raw()
                .cmd_reset_query_pool(cmd, self.pool, first, 2);
            self.device.raw().cmd_write_timestamp2(
                cmd,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                self.pool,
                first,
            );
        }
    }

    /// Write the end timestamp; the result becomes readable once the frame has finished.
    pub(crate) fn end(&mut self, cmd: vk::CommandBuffer, record: u64) {
        let slot = (record % FRAMES_IN_FLIGHT as u64) as usize;
        // SAFETY: as in `begin`.
        unsafe {
            self.device.raw().cmd_write_timestamp2(
                cmd,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                self.pool,
                (slot * 2 + 1) as u32,
            );
        }
        self.written[slot] = true;
    }
}

impl Drop for GpuTimer {
    fn drop(&mut self) {
        // SAFETY: the owner waits for the device to be idle before dropping the timer.
        unsafe { self.device.raw().destroy_query_pool(self.pool, None) };
    }
}
