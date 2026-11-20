//! Validation-layer message handling.

use ash::vk;
use std::ffi::{CStr, c_void};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// Maximum number of distinct error messages retained for later inspection.
const MAX_RETAINED_ERRORS: usize = 16;

/// Counters fed by the validation layer.
///
/// The smoke tests and CI assert `errors() == 0`, which is how the "no validation errors in the
/// basic scenario" acceptance criterion is verified mechanically.
#[derive(Debug, Default)]
pub struct ValidationStats {
    errors: AtomicU64,
    general_errors: AtomicU64,
    warnings: AtomicU64,
    retained: Mutex<Vec<String>>,
}

impl ValidationStats {
    /// Number of `ERROR` severity messages of type *validation* (API misuse found by the
    /// validation layer). This is the counter acceptance tests assert on.
    pub fn errors(&self) -> u64 {
        self.errors.load(Ordering::Relaxed)
    }

    /// Number of `ERROR` severity messages of type *general* (loader / environment problems such
    /// as a broken third-party layer manifest). They are logged but are not API misuse by us.
    pub fn general_errors(&self) -> u64 {
        self.general_errors.load(Ordering::Relaxed)
    }

    /// Number of `WARNING` severity messages seen.
    pub fn warnings(&self) -> u64 {
        self.warnings.load(Ordering::Relaxed)
    }

    /// The first few error messages, for inclusion in failure reports.
    pub fn retained_errors(&self) -> Vec<String> {
        self.retained
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }
}

/// Callback registered with `VK_EXT_debug_utils`.
///
/// # Safety
/// Called by the Vulkan loader. `user_data` must be the pointer to a live [`ValidationStats`]
/// registered by [`crate::instance::Instance`].
pub(crate) unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    user_data: *mut c_void,
) -> vk::Bool32 {
    // SAFETY: the loader passes either null or a valid callback-data pointer for the duration of
    // this call.
    let Some(data) = (unsafe { data.as_ref() }) else {
        return vk::FALSE;
    };
    let message = if data.p_message.is_null() {
        std::borrow::Cow::Borrowed("<no message>")
    } else {
        // SAFETY: `p_message` is a NUL-terminated string valid for this call.
        unsafe { CStr::from_ptr(data.p_message) }.to_string_lossy()
    };
    let kind = if types.contains(vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE) {
        "performance"
    } else if types.contains(vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION) {
        "validation"
    } else {
        "general"
    };

    // SAFETY: `user_data` is the `ValidationStats` we registered; it outlives the messenger.
    let stats = unsafe { (user_data as *const ValidationStats).as_ref() };

    let is_validation = types.contains(vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION);
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) && !is_validation {
        tracing::warn!(target: "vulkan", "[{kind}/environment] {message}");
        if let Some(stats) = stats {
            stats.general_errors.fetch_add(1, Ordering::Relaxed);
        }
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        tracing::error!(target: "vulkan", "[{kind}] {message}");
        if let Some(stats) = stats {
            stats.errors.fetch_add(1, Ordering::Relaxed);
            let mut retained = stats.retained.lock().unwrap_or_else(|p| p.into_inner());
            if retained.len() < MAX_RETAINED_ERRORS {
                retained.push(message.into_owned());
            }
        }
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        tracing::warn!(target: "vulkan", "[{kind}] {message}");
        // Warnings from the loader about third-party layers are environment noise.
        if let Some(stats) = stats.filter(|_| is_validation) {
            stats.warnings.fetch_add(1, Ordering::Relaxed);
        }
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::INFO) {
        tracing::debug!(target: "vulkan", "[{kind}] {message}");
    } else {
        tracing::trace!(target: "vulkan", "[{kind}] {message}");
    }
    vk::FALSE
}
