//! Error type of the Vulkan layer.

use ash::vk;

/// Result alias for this crate.
pub type Result<T, E = VulkanError> = std::result::Result<T, E>;

/// Failures raised while driving Vulkan.
#[derive(Debug, thiserror::Error)]
pub enum VulkanError {
    /// The Vulkan loader library could not be found or loaded.
    #[error("failed to load the Vulkan loader: {0}")]
    Load(#[from] ash::LoadingError),

    /// A Vulkan entry point returned an error code.
    #[error("Vulkan call `{call}` failed: {result:?}")]
    Call {
        /// Name of the failing function.
        call: &'static str,
        /// Result code returned by the driver.
        result: vk::Result,
    },

    /// The logical device was lost (driver crash, GPU reset, device removal).
    ///
    /// This is unrecoverable for the current device; the application must tear down and may
    /// rebuild the whole Vulkan stack.
    #[error("the Vulkan device was lost during `{call}`")]
    DeviceLost {
        /// Name of the call that reported the loss.
        call: &'static str,
    },

    /// No physical device satisfies the engine's requirements.
    #[error("no suitable Vulkan device: {0}")]
    NoSuitableDevice(String),

    /// The GPU memory allocator reported an error.
    #[error("GPU allocator error: {0}")]
    Allocator(String),

    /// Data handed to the renderer is inconsistent (an index out of range, a section past the end
    /// of its mesh, ...). Nothing was created.
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// The window system handles were rejected or unsupported.
    #[error("window system integration error: {0}")]
    WindowSystem(String),
}

impl VulkanError {
    /// `true` if this error means the device is gone.
    pub fn is_device_lost(&self) -> bool {
        matches!(self, Self::DeviceLost { .. })
    }
}

/// Adds `.vk("call")` to raw `ash` results, mapping them into [`VulkanError`].
pub trait VkResultExt<T> {
    /// Attach the name of the Vulkan call for diagnostics.
    fn vk(self, call: &'static str) -> Result<T>;
}

impl<T> VkResultExt<T> for ash::prelude::VkResult<T> {
    fn vk(self, call: &'static str) -> Result<T> {
        self.map_err(|result| match result {
            vk::Result::ERROR_DEVICE_LOST => VulkanError::DeviceLost { call },
            result => VulkanError::Call { call, result },
        })
    }
}
