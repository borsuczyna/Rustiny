//! Frame loop driver: frames in flight, acquire/submit/present and swapchain recreation.

use crate::device::Device;
use crate::error::{Result, VkResultExt};
use crate::readback::{ReadbackBuffer, Screenshot};
use crate::swapchain::{Surface, Swapchain, SwapchainState, color_range};
use ash::vk;
use rustiny_core::config::PresentModePreference;
use std::sync::Arc;

/// Number of frames the CPU may record ahead of the GPU.
pub const FRAMES_IN_FLIGHT: usize = 2;

struct FrameSlot {
    pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    image_available: vk::Semaphore,
    in_flight: vk::Fence,
}

/// A frame that has been acquired and is being recorded.
///
/// The swapchain image is already in `COLOR_ATTACHMENT_OPTIMAL` layout; [`Renderer::end_frame`]
/// transitions it to `PRESENT_SRC_KHR`.
#[derive(Debug)]
pub struct Frame {
    /// Primary command buffer in the recording state.
    pub command_buffer: vk::CommandBuffer,
    /// Index of the frame slot (`0..FRAMES_IN_FLIGHT`).
    pub frame_index: usize,
    /// Swapchain image index.
    pub image_index: u32,
    /// Swapchain image.
    pub image: vk::Image,
    /// View of the swapchain image.
    pub view: vk::ImageView,
    /// Size of the image.
    pub extent: vk::Extent2D,
}

/// Owns the surface, swapchain and per-frame resources and runs the frame loop.
///
/// Typical use:
/// ```ignore
/// if let Some(frame) = renderer.begin_frame()? {
///     renderer.begin_rendering(&frame, [0.1, 0.1, 0.1, 1.0]);
///     // ... record draws ...
///     renderer.end_rendering(&frame);
///     renderer.end_frame(frame)?;
/// }
/// ```
pub struct Renderer {
    device: Arc<Device>,
    // Declared before `surface` so that it is dropped first.
    swapchain: Option<Swapchain>,
    surface: Surface,
    frames: Vec<FrameSlot>,
    frame_index: usize,
    upload_pool: vk::CommandPool,
    present_pref: PresentModePreference,
    size: (u32, u32),
    needs_recreate: bool,
    swapchain_generation: u64,
    screenshot_requested: bool,
    screenshot: Option<Screenshot>,
}

impl std::fmt::Debug for Renderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Renderer")
            .field("size", &self.size)
            .field("swapchain", &self.swapchain)
            .finish()
    }
}

impl Renderer {
    /// Build the swapchain and per-frame resources.
    pub fn new(
        device: &Arc<Device>,
        surface: Surface,
        size: (u32, u32),
        present_pref: PresentModePreference,
    ) -> Result<Self> {
        let swapchain = Swapchain::new(device, &surface, size, present_pref, None)?;
        let raw = device.raw();
        let family = device.families().graphics;

        let mut frames = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for _ in 0..FRAMES_IN_FLIGHT {
            let pool_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(family)
                .flags(vk::CommandPoolCreateFlags::TRANSIENT);
            // SAFETY: plain object creation on a live device.
            let pool =
                unsafe { raw.create_command_pool(&pool_info, None) }.vk("vkCreateCommandPool")?;
            let alloc = vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            // SAFETY: pool was just created.
            let command_buffer =
                unsafe { raw.allocate_command_buffers(&alloc) }.vk("vkAllocateCommandBuffers")?[0];
            // SAFETY: plain object creation.
            let image_available =
                unsafe { raw.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }
                    .vk("vkCreateSemaphore")?;
            // Created signalled so the very first wait returns immediately.
            let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
            // SAFETY: plain object creation.
            let in_flight = unsafe { raw.create_fence(&fence_info, None) }.vk("vkCreateFence")?;
            frames.push(FrameSlot {
                pool,
                command_buffer,
                image_available,
                in_flight,
            });
        }

        // Pool for short-lived upload command buffers (UI font atlas, texture uploads).
        let upload_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(family)
            .flags(
                vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER
                    | vk::CommandPoolCreateFlags::TRANSIENT,
            );
        // SAFETY: plain object creation.
        let upload_pool = unsafe { raw.create_command_pool(&upload_info, None) }
            .vk("vkCreateCommandPool(upload)")?;

        Ok(Self {
            device: device.clone(),
            swapchain: Some(swapchain),
            surface,
            frames,
            frame_index: 0,
            upload_pool,
            present_pref,
            size,
            needs_recreate: false,
            swapchain_generation: 0,
            screenshot_requested: false,
            screenshot: None,
        })
    }

    /// The device this renderer draws with.
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Command pool for short-lived upload command buffers. Externally synchronised: only the
    /// thread that drives the frame loop may use it.
    pub fn upload_command_pool(&self) -> vk::CommandPool {
        self.upload_pool
    }

    /// Current swapchain image format.
    ///
    /// # Panics
    /// Never in normal operation; the swapchain only goes missing transiently inside recreation.
    pub fn color_format(&self) -> vk::Format {
        self.swapchain().format()
    }

    /// Current swapchain size in pixels.
    pub fn extent(&self) -> vk::Extent2D {
        self.swapchain().extent()
    }

    /// Current swapchain image count.
    pub fn image_count(&self) -> usize {
        self.swapchain().image_count()
    }

    /// Presentation mode currently in use.
    pub fn present_mode(&self) -> vk::PresentModeKHR {
        self.swapchain().present_mode()
    }

    /// Incremented every time the swapchain is rebuilt (useful for tests and diagnostics).
    pub fn swapchain_generation(&self) -> u64 {
        self.swapchain_generation
    }

    fn swapchain(&self) -> &Swapchain {
        self.swapchain
            .as_ref()
            .expect("swapchain exists outside of recreation")
    }

    /// Tell the renderer the window changed size (in physical pixels). The swapchain is rebuilt
    /// lazily at the start of the next frame.
    pub fn resize(&mut self, width: u32, height: u32) {
        if (width, height) != self.size {
            self.size = (width, height);
            self.needs_recreate = true;
        }
    }

    /// Begin recording a frame. Returns `None` when nothing can be drawn right now (window
    /// minimised, or the swapchain was out of date and has been scheduled for recreation).
    pub fn begin_frame(&mut self) -> Result<Option<Frame>> {
        if self.size.0 == 0 || self.size.1 == 0 {
            return Ok(None);
        }
        if self.needs_recreate {
            self.recreate_swapchain()?;
        }
        let raw = self.device.raw();
        let slot = &self.frames[self.frame_index];

        // SAFETY: valid fence; waiting here is what bounds the CPU to FRAMES_IN_FLIGHT frames.
        unsafe { raw.wait_for_fences(&[slot.in_flight], true, u64::MAX) }.vk("vkWaitForFences")?;

        let Some((image_index, state)) = self.swapchain().acquire(slot.image_available)? else {
            self.needs_recreate = true;
            return Ok(None);
        };
        if state == SwapchainState::Suboptimal {
            self.needs_recreate = true;
        }

        // Only reset the fence once we know a submission will follow, otherwise a failed acquire
        // would leave it unsignalled forever and deadlock the next wait.
        // SAFETY: fence is signalled and not in use.
        unsafe { raw.reset_fences(&[slot.in_flight]) }.vk("vkResetFences")?;
        // SAFETY: the fence wait guarantees the GPU is done with this pool's buffers.
        unsafe { raw.reset_command_pool(slot.pool, vk::CommandPoolResetFlags::empty()) }
            .vk("vkResetCommandPool")?;
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: buffer is in the initial state after the pool reset.
        unsafe { raw.begin_command_buffer(slot.command_buffer, &begin) }
            .vk("vkBeginCommandBuffer")?;

        let sc = self.swapchain();
        let image = sc.image(image_index);
        // Undefined -> attachment: previous contents are cleared anyway.
        transition(
            raw,
            slot.command_buffer,
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );

        Ok(Some(Frame {
            command_buffer: slot.command_buffer,
            frame_index: self.frame_index,
            image_index,
            image,
            view: sc.view(image_index),
            extent: sc.extent(),
        }))
    }

    /// Begin dynamic rendering into the frame's swapchain image, clearing it to `clear`.
    pub fn begin_rendering(&self, frame: &Frame, clear: [f32; 4]) {
        let attachment = vk::RenderingAttachmentInfo::default()
            .image_view(frame.view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue { float32: clear },
            });
        let attachments = [attachment];
        let info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: frame.extent,
            })
            .layer_count(1)
            .color_attachments(&attachments);
        // SAFETY: the command buffer is recording and the view is valid for this frame.
        unsafe {
            self.device
                .raw()
                .cmd_begin_rendering(frame.command_buffer, &info)
        };
    }

    /// End the dynamic rendering scope started by [`Renderer::begin_rendering`].
    pub fn end_rendering(&self, frame: &Frame) {
        // SAFETY: matches a prior `cmd_begin_rendering` on the same command buffer.
        unsafe { self.device.raw().cmd_end_rendering(frame.command_buffer) };
    }

    /// Finish recording, submit and present.
    pub fn end_frame(&mut self, frame: Frame) -> Result<()> {
        let raw = self.device.raw();
        let slot = &self.frames[frame.frame_index];

        let capture = if self.screenshot_requested && self.swapchain().supports_readback() {
            self.screenshot_requested = false;
            Some(ReadbackBuffer::new(
                &self.device,
                frame.extent,
                self.swapchain().format(),
            )?)
        } else {
            None
        };
        if let Some(readback) = &capture {
            barrier(
                raw,
                frame.command_buffer,
                frame.image,
                (
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                ),
                (
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::PipelineStageFlags2::COPY,
                    vk::AccessFlags2::TRANSFER_READ,
                ),
            );
            readback.record_copy(&self.device, frame.command_buffer, frame.image);
            barrier(
                raw,
                frame.command_buffer,
                frame.image,
                (
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::PipelineStageFlags2::COPY,
                    vk::AccessFlags2::TRANSFER_READ,
                ),
                (
                    vk::ImageLayout::PRESENT_SRC_KHR,
                    vk::PipelineStageFlags2::NONE,
                    vk::AccessFlags2::NONE,
                ),
            );
        } else {
            transition(
                raw,
                frame.command_buffer,
                frame.image,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
            );
        }
        // SAFETY: the buffer is recording.
        unsafe { raw.end_command_buffer(frame.command_buffer) }.vk("vkEndCommandBuffer")?;

        let sc = self.swapchain();
        let wait = [vk::SemaphoreSubmitInfo::default()
            .semaphore(slot.image_available)
            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)];
        let signal = [vk::SemaphoreSubmitInfo::default()
            .semaphore(sc.render_finished(frame.image_index))
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let buffers = [vk::CommandBufferSubmitInfo::default().command_buffer(frame.command_buffer)];
        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&wait)
            .command_buffer_infos(&buffers)
            .signal_semaphore_infos(&signal);
        let queue = self.device.graphics_queue();
        // SAFETY: all handles are valid; the fence was reset in `begin_frame`.
        unsafe { raw.queue_submit2(queue, &[submit], slot.in_flight) }.vk("vkQueueSubmit2")?;

        let state = sc.present(queue, frame.image_index)?;
        if state != SwapchainState::Optimal {
            self.needs_recreate = true;
        }
        let slot_fence = self.frames[frame.frame_index].in_flight;
        self.frame_index = (self.frame_index + 1) % FRAMES_IN_FLIGHT;

        if let Some(readback) = capture {
            // Screenshots are rare and explicit, so waiting for this frame here is acceptable.
            // SAFETY: valid fence that was submitted above.
            unsafe {
                self.device
                    .raw()
                    .wait_for_fences(&[slot_fence], true, u64::MAX)
            }
            .vk("vkWaitForFences(screenshot)")?;
            self.screenshot = Some(readback.finish(&self.device)?);
        }
        Ok(())
    }

    /// Ask for the next presented frame to be copied back to the CPU.
    ///
    /// Returns `false` if the swapchain does not support readback.
    pub fn request_screenshot(&mut self) -> bool {
        if self.swapchain().supports_readback() {
            self.screenshot_requested = true;
            true
        } else {
            false
        }
    }

    /// Take the captured frame, if one has completed since the last call.
    pub fn take_screenshot(&mut self) -> Option<Screenshot> {
        self.screenshot.take()
    }

    fn recreate_swapchain(&mut self) -> Result<()> {
        // Rebuilding while images may still be in flight requires the device to be idle. This
        // only happens on resize / out-of-date, never on the steady-state path.
        self.device.wait_idle()?;
        let old = self.swapchain.take();
        match Swapchain::new(
            &self.device,
            &self.surface,
            self.size,
            self.present_pref,
            old.as_ref(),
        ) {
            Ok(new) => {
                self.swapchain = Some(new);
                drop(old);
                self.needs_recreate = false;
                self.swapchain_generation += 1;
                Ok(())
            }
            Err(e) => {
                // Keep the old swapchain so Drop invariants hold; the caller may retry later.
                self.swapchain = old;
                Err(e)
            }
        }
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        if let Err(e) = self.device.wait_idle() {
            tracing::error!("wait before renderer teardown failed: {e}");
        }
        let raw = self.device.raw();
        // SAFETY: the device is idle; each object is destroyed once.
        unsafe {
            for slot in &self.frames {
                raw.destroy_fence(slot.in_flight, None);
                raw.destroy_semaphore(slot.image_available, None);
                raw.destroy_command_pool(slot.pool, None);
            }
            raw.destroy_command_pool(self.upload_pool, None);
        }
        // `swapchain` then `surface` drop afterwards in declaration order.
    }
}

/// Record an arbitrary colour-image layout transition: `(layout, stage, access)` before and after.
fn barrier(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    from: (vk::ImageLayout, vk::PipelineStageFlags2, vk::AccessFlags2),
    to: (vk::ImageLayout, vk::PipelineStageFlags2, vk::AccessFlags2),
) {
    let barrier = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(from.1)
        .src_access_mask(from.2)
        .dst_stage_mask(to.1)
        .dst_access_mask(to.2)
        .old_layout(from.0)
        .new_layout(to.0)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(color_range());
    let barriers = [barrier];
    let dependency = vk::DependencyInfo::default().image_memory_barriers(&barriers);
    // SAFETY: the command buffer is in the recording state.
    unsafe { device.cmd_pipeline_barrier2(cmd, &dependency) };
}

/// Record a colour-image layout transition with a synchronization2 barrier.
fn transition(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    old: vk::ImageLayout,
    new: vk::ImageLayout,
) {
    let (src_stage, src_access, dst_stage, dst_access) = match new {
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::NONE,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        ),
        _ => (
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::NONE,
            vk::AccessFlags2::NONE,
        ),
    };
    let barrier = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access)
        .old_layout(old)
        .new_layout(new)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(color_range());
    let barriers = [barrier];
    let dependency = vk::DependencyInfo::default().image_memory_barriers(&barriers);
    // SAFETY: the command buffer is in the recording state.
    unsafe { device.cmd_pipeline_barrier2(cmd, &dependency) };
}
