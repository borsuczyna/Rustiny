//! Window surface and swapchain.

use crate::device::Device;
use crate::error::{Result, VkResultExt, VulkanError};
use crate::instance::Instance;
use ash::vk;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use rustiny_core::config::PresentModePreference;
use std::sync::Arc;

/// A presentation surface bound to a native window.
pub struct Surface {
    instance: Arc<Instance>,
    loader: ash::khr::surface::Instance,
    raw: vk::SurfaceKHR,
}

impl std::fmt::Debug for Surface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Surface").finish_non_exhaustive()
    }
}

impl Surface {
    /// Create a surface for the given native window.
    ///
    /// # Safety
    /// `display` and `window` must be valid handles of a live window and must stay valid until
    /// this `Surface` is dropped.
    pub unsafe fn new(
        instance: &Arc<Instance>,
        display: RawDisplayHandle,
        window: RawWindowHandle,
    ) -> Result<Self> {
        // SAFETY: forwarded from the caller's contract.
        let raw = unsafe {
            ash_window::create_surface(instance.entry(), instance.raw(), display, window, None)
        }
        .vk("vkCreate*SurfaceKHR")?;
        let loader = ash::khr::surface::Instance::new(instance.entry(), instance.raw());
        Ok(Self {
            instance: instance.clone(),
            loader,
            raw,
        })
    }

    /// Raw surface handle.
    pub fn raw(&self) -> vk::SurfaceKHR {
        self.raw
    }

    /// Surface extension loader.
    pub fn loader(&self) -> &ash::khr::surface::Instance {
        &self.loader
    }

    /// Owning instance.
    pub fn instance(&self) -> &Arc<Instance> {
        &self.instance
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        // SAFETY: the surface was created by us; the swapchain using it is dropped first because
        // the renderer declares it before the surface.
        unsafe { self.loader.destroy_surface(self.raw, None) };
    }
}

/// Outcome of presenting or acquiring: whether the swapchain still matches the surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwapchainState {
    /// Matches the surface exactly.
    Optimal,
    /// Still usable but should be recreated soon (e.g. after a resize).
    Suboptimal,
    /// Unusable; recreate before rendering again.
    OutOfDate,
}

/// Swapchain with its image views and per-image "render finished" semaphores.
///
/// Presentation waits on a semaphore that is indexed by *swapchain image* rather than by frame in
/// flight: a present operation holds its wait semaphore until the image is consumed, so reusing a
/// per-frame semaphore could signal it while a previous present still waits on it.
pub struct Swapchain {
    device: Arc<Device>,
    loader: ash::khr::swapchain::Device,
    raw: vk::SwapchainKHR,
    format: vk::SurfaceFormatKHR,
    extent: vk::Extent2D,
    present_mode: vk::PresentModeKHR,
    supports_readback: bool,
    images: Vec<vk::Image>,
    views: Vec<vk::ImageView>,
    render_finished: Vec<vk::Semaphore>,
}

impl std::fmt::Debug for Swapchain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Swapchain")
            .field("format", &self.format.format)
            .field("extent", &self.extent)
            .field("present_mode", &self.present_mode)
            .field("images", &self.images.len())
            .finish()
    }
}

impl Swapchain {
    /// Create a swapchain for `surface`, optionally recycling `old`.
    pub fn new(
        device: &Arc<Device>,
        surface: &Surface,
        size: (u32, u32),
        preference: PresentModePreference,
        old: Option<&Swapchain>,
    ) -> Result<Self> {
        let physical = device.physical();
        let loader_s = surface.loader();
        // SAFETY: surface and physical device belong to the same instance.
        let caps =
            unsafe { loader_s.get_physical_device_surface_capabilities(physical, surface.raw()) }
                .vk("vkGetPhysicalDeviceSurfaceCapabilitiesKHR")?;
        // SAFETY: as above.
        let formats =
            unsafe { loader_s.get_physical_device_surface_formats(physical, surface.raw()) }
                .vk("vkGetPhysicalDeviceSurfaceFormatsKHR")?;
        // SAFETY: as above.
        let modes =
            unsafe { loader_s.get_physical_device_surface_present_modes(physical, surface.raw()) }
                .vk("vkGetPhysicalDeviceSurfacePresentModesKHR")?;

        // UNORM keeps UI colours identical to what Dear ImGui authors; the scene renderer will
        // render in linear HDR offscreen and convert at the end, so an sRGB swapchain buys nothing.
        let format = formats
            .iter()
            .copied()
            .find(|f| {
                f.format == vk::Format::B8G8R8A8_UNORM
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .or_else(|| {
                formats.iter().copied().find(|f| {
                    f.format == vk::Format::R8G8B8A8_UNORM
                        && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                })
            })
            .or_else(|| formats.first().copied())
            .ok_or_else(|| {
                VulkanError::NoSuitableDevice("surface reports no formats".to_owned())
            })?;

        let wanted_mode = match preference {
            PresentModePreference::Fifo => vk::PresentModeKHR::FIFO,
            PresentModePreference::Mailbox => vk::PresentModeKHR::MAILBOX,
            PresentModePreference::Immediate => vk::PresentModeKHR::IMMEDIATE,
        };
        // FIFO is guaranteed by the specification.
        let present_mode = if modes.contains(&wanted_mode) {
            wanted_mode
        } else {
            vk::PresentModeKHR::FIFO
        };

        let extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            vk::Extent2D {
                width: size
                    .0
                    .clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                height: size
                    .1
                    .clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            }
        };
        if extent.width == 0 || extent.height == 0 {
            return Err(VulkanError::WindowSystem(
                "surface extent is zero (minimised?)".into(),
            ));
        }

        let mut image_count = caps.min_image_count + 1;
        if caps.max_image_count > 0 {
            image_count = image_count.min(caps.max_image_count);
        }

        // TRANSFER_SRC enables screenshots / rendering tests; it is optional.
        let supports_readback = caps
            .supported_usage_flags
            .contains(vk::ImageUsageFlags::TRANSFER_SRC);
        let mut usage = vk::ImageUsageFlags::COLOR_ATTACHMENT;
        if supports_readback {
            usage |= vk::ImageUsageFlags::TRANSFER_SRC;
        }

        let queue_family = [device.families().graphics];
        let info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface.raw())
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(usage)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&queue_family)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(old.map_or(vk::SwapchainKHR::null(), |s| s.raw));

        let loader = ash::khr::swapchain::Device::new(device.instance().raw(), device.raw());
        // SAFETY: `info` references live locals; the old swapchain (if any) is valid.
        let raw = unsafe { loader.create_swapchain(&info, None) }.vk("vkCreateSwapchainKHR")?;

        let mut this = Self {
            device: device.clone(),
            loader,
            raw,
            format,
            extent,
            present_mode,
            supports_readback,
            images: Vec::new(),
            views: Vec::new(),
            render_finished: Vec::new(),
        };
        // From here on `Drop` cleans up partially built state on error.
        // SAFETY: valid swapchain.
        this.images =
            unsafe { this.loader.get_swapchain_images(raw) }.vk("vkGetSwapchainImagesKHR")?;
        for &image in &this.images {
            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format.format)
                .subresource_range(color_range());
            // SAFETY: image belongs to the swapchain just created.
            let view = unsafe { device.raw().create_image_view(&view_info, None) }
                .vk("vkCreateImageView")?;
            this.views.push(view);
            // SAFETY: plain object creation.
            let sem = unsafe {
                device
                    .raw()
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            }
            .vk("vkCreateSemaphore")?;
            this.render_finished.push(sem);
        }

        tracing::info!(
            "swapchain {}x{} {:?} {:?}, {} images",
            extent.width,
            extent.height,
            format.format,
            present_mode,
            this.images.len()
        );
        Ok(this)
    }

    /// Acquire the next image. Returns `(image_index, state)`.
    pub fn acquire(&self, signal: vk::Semaphore) -> Result<Option<(u32, SwapchainState)>> {
        // SAFETY: valid swapchain and semaphore; infinite timeout is fine because FIFO/mailbox
        // acquisition completes as soon as the presentation engine releases an image.
        match unsafe {
            self.loader
                .acquire_next_image(self.raw, u64::MAX, signal, vk::Fence::null())
        } {
            Ok((index, false)) => Ok(Some((index, SwapchainState::Optimal))),
            Ok((index, true)) => Ok(Some((index, SwapchainState::Suboptimal))),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(None),
            Err(e) => Err(e).vk("vkAcquireNextImageKHR"),
        }
    }

    /// Present `image_index`, waiting on its render-finished semaphore.
    pub fn present(&self, queue: vk::Queue, image_index: u32) -> Result<SwapchainState> {
        let wait = [self.render_finished[image_index as usize]];
        let swapchains = [self.raw];
        let indices = [image_index];
        let info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait)
            .swapchains(&swapchains)
            .image_indices(&indices);
        // SAFETY: all handles are valid and the semaphore will be signalled by the submission
        // recorded for this image.
        match unsafe { self.loader.queue_present(queue, &info) } {
            Ok(false) => Ok(SwapchainState::Optimal),
            Ok(true) => Ok(SwapchainState::Suboptimal),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(SwapchainState::OutOfDate),
            Err(e) => Err(e).vk("vkQueuePresentKHR"),
        }
    }

    /// Image count.
    pub fn image_count(&self) -> usize {
        self.images.len()
    }

    /// Image at `index`.
    pub fn image(&self, index: u32) -> vk::Image {
        self.images[index as usize]
    }

    /// View of image `index`.
    pub fn view(&self, index: u32) -> vk::ImageView {
        self.views[index as usize]
    }

    /// Semaphore signalled when rendering to image `index` has finished.
    pub fn render_finished(&self, index: u32) -> vk::Semaphore {
        self.render_finished[index as usize]
    }

    /// Pixel format of the images.
    pub fn format(&self) -> vk::Format {
        self.format.format
    }

    /// Size in pixels.
    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    /// `true` if frames can be copied out of this swapchain.
    pub fn supports_readback(&self) -> bool {
        self.supports_readback
    }

    /// Presentation mode actually in use.
    pub fn present_mode(&self) -> vk::PresentModeKHR {
        self.present_mode
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        let device = self.device.raw();
        // SAFETY: callers wait for the device (or at least all work using these images) before
        // dropping; objects are destroyed exactly once.
        unsafe {
            for &sem in &self.render_finished {
                device.destroy_semaphore(sem, None);
            }
            for &view in &self.views {
                device.destroy_image_view(view, None);
            }
            self.loader.destroy_swapchain(self.raw, None);
        }
    }
}

pub(crate) fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1)
}
