//! Why a run did not complete normally.

use crate::gui::GuiError;
use rustiny_vulkan::VulkanError;

/// Why a run did not complete normally.
#[derive(Debug, thiserror::Error)]
pub enum RunError {
    /// The machine lacks a prerequisite (Vulkan loader, 1.3 GPU, display). Not a code defect.
    #[error("unsupported environment: {0}")]
    Unsupported(String),
    /// Something broke at runtime.
    #[error("{0}")]
    Failed(String),
}

impl From<VulkanError> for RunError {
    fn from(e: VulkanError) -> Self {
        match e {
            VulkanError::Load(_) | VulkanError::NoSuitableDevice(_) => {
                Self::Unsupported(e.to_string())
            }
            other => Self::Failed(other.to_string()),
        }
    }
}

impl From<GuiError> for RunError {
    fn from(e: GuiError) -> Self {
        Self::Failed(e.to_string())
    }
}
