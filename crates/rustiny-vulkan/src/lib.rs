//! Thin, safe-by-construction Vulkan layer for Rustiny built on `ash`.
//!
//! Scope of this crate (Milestone 1): loader/instance with validation, physical-device selection,
//! logical device with a shared GPU allocator, swapchain management and a frames-in-flight loop
//! using dynamic rendering and synchronization2. The render graph, descriptor management and
//! pipeline cache arrive in later milestones.
//!
//! Ownership model: every wrapper holds an `Arc` to the object it depends on
//! (`Surface`/`Device` -> `Instance`, `Swapchain`/`Renderer` -> `Device`), so Vulkan destruction
//! order is correct by construction. Third parties that hold raw clones (for example the UI
//! renderer's allocator handle) must be shut down before the [`Device`] is dropped; the device
//! logs an error if it detects that this was violated.

#![cfg_attr(test, allow(clippy::unwrap_used))]

pub mod debug;
pub mod device;
pub mod error;
pub mod instance;
pub mod readback;
pub mod renderer;
pub mod swapchain;

pub use ash;
pub use debug::ValidationStats;
pub use device::{Device, DeviceConfig, DeviceInfo, PresentTarget, QueueFamilies};
pub use error::{Result, VkResultExt, VulkanError};
pub use instance::{Instance, InstanceConfig};
pub use readback::Screenshot;
pub use renderer::{FRAMES_IN_FLIGHT, Frame, Renderer};
pub use swapchain::{Surface, Swapchain, SwapchainState};
