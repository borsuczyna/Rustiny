//! Vulkan loader, instance and validation setup.

use crate::debug::{ValidationStats, debug_callback};
use crate::error::{Result, VkResultExt, VulkanError};
use ash::{Entry, vk};
use raw_window_handle::RawDisplayHandle;
use std::ffi::{CStr, c_char};
use std::sync::Arc;

const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

/// Instance creation parameters.
#[derive(Debug, Clone)]
pub struct InstanceConfig {
    /// Application name reported to the driver.
    pub app_name: String,
    /// Request the Khronos validation layer (falls back with a warning if it is not installed).
    pub validation: bool,
    /// Additionally enable synchronisation validation.
    pub sync_validation: bool,
    /// Display system handle used to determine the required surface extensions. `None` creates a
    /// headless instance without any presentation support.
    pub display: Option<RawDisplayHandle>,
}

impl Default for InstanceConfig {
    fn default() -> Self {
        Self {
            app_name: "Rustiny".to_owned(),
            validation: false,
            sync_validation: false,
            display: None,
        }
    }
}

struct DebugMessenger {
    loader: ash::ext::debug_utils::Instance,
    raw: vk::DebugUtilsMessengerEXT,
}

/// Owns the Vulkan loader and instance.
pub struct Instance {
    entry: Entry,
    raw: ash::Instance,
    debug: Option<DebugMessenger>,
    stats: Arc<ValidationStats>,
    validation_enabled: bool,
    sync_validation_enabled: bool,
}

impl std::fmt::Debug for Instance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Instance")
            .field("validation", &self.validation_enabled)
            .field("sync_validation", &self.sync_validation_enabled)
            .finish_non_exhaustive()
    }
}

impl Instance {
    /// Load Vulkan and create an instance.
    pub fn new(config: &InstanceConfig) -> Result<Arc<Self>> {
        // SAFETY: loading the system Vulkan loader runs its initialisation code; this is the
        // documented way to obtain an `Entry` and is done once at start-up.
        let entry = unsafe { Entry::load()? };

        // SAFETY: plain query calls on a valid entry.
        let available_layers = unsafe { entry.enumerate_instance_layer_properties() }
            .vk("vkEnumerateInstanceLayerProperties")?;
        let layer_present = available_layers
            .iter()
            .any(|l| l.layer_name_as_c_str().is_ok_and(|n| n == VALIDATION_LAYER));
        let validation = config.validation && layer_present;
        if config.validation && !layer_present {
            tracing::warn!(
                "validation was requested but {} is not installed; continuing without it",
                VALIDATION_LAYER.to_string_lossy()
            );
        }

        // SAFETY: query call.
        let available_exts = unsafe { entry.enumerate_instance_extension_properties(None) }
            .vk("vkEnumerateInstanceExtensionProperties")?;
        let has_ext = |name: &CStr| {
            available_exts
                .iter()
                .any(|e| e.extension_name_as_c_str().is_ok_and(|n| n == name))
        };

        let mut extensions: Vec<*const c_char> = Vec::new();
        if let Some(display) = config.display {
            let required = ash_window::enumerate_required_extensions(display)
                .map_err(|e| VulkanError::WindowSystem(e.to_string()))?;
            extensions.extend_from_slice(required);
        }

        let debug_utils = has_ext(ash::ext::debug_utils::NAME);
        if debug_utils {
            extensions.push(ash::ext::debug_utils::NAME.as_ptr());
        }

        // Synchronisation validation is exposed by the layer through VK_EXT_validation_features.
        let sync_validation = validation && config.sync_validation && {
            // SAFETY: query call with a valid, NUL-terminated layer name.
            let layer_exts =
                unsafe { entry.enumerate_instance_extension_properties(Some(VALIDATION_LAYER)) }
                    .vk("vkEnumerateInstanceExtensionProperties(layer)")?;
            layer_exts.iter().any(|e| {
                e.extension_name_as_c_str()
                    .is_ok_and(|n| n == ash::ext::validation_features::NAME)
            })
        };
        if sync_validation {
            extensions.push(ash::ext::validation_features::NAME.as_ptr());
        }

        let layers: Vec<*const c_char> = if validation {
            vec![VALIDATION_LAYER.as_ptr()]
        } else {
            Vec::new()
        };

        let app_name = std::ffi::CString::new(config.app_name.replace('\0', ""))
            .unwrap_or_else(|_| c"Rustiny".to_owned());
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(c"Rustiny")
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_3);

        let stats = Arc::new(ValidationStats::default());
        let user_data = Arc::as_ptr(&stats) as *mut std::ffi::c_void;
        let messenger_info = messenger_create_info(user_data);

        let enabled_features = [vk::ValidationFeatureEnableEXT::SYNCHRONIZATION_VALIDATION];
        let mut validation_features =
            vk::ValidationFeaturesEXT::default().enabled_validation_features(&enabled_features);
        let mut messenger_for_create = messenger_info;

        let mut create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extensions)
            .enabled_layer_names(&layers);
        if debug_utils {
            // Also reports problems during vkCreateInstance / vkDestroyInstance.
            create_info = create_info.push_next(&mut messenger_for_create);
        }
        if sync_validation {
            create_info = create_info.push_next(&mut validation_features);
        }

        // SAFETY: all pointers in `create_info` reference locals that outlive this call.
        let raw = unsafe { entry.create_instance(&create_info, None) }.vk("vkCreateInstance")?;

        let debug = if debug_utils {
            let loader = ash::ext::debug_utils::Instance::new(&entry, &raw);
            // SAFETY: `user_data` points into `stats`, which this struct keeps alive for at
            // least as long as the messenger (it is destroyed first in `Drop`).
            let messenger = unsafe { loader.create_debug_utils_messenger(&messenger_info, None) };
            match messenger {
                Ok(raw) => Some(DebugMessenger { loader, raw }),
                Err(e) => {
                    tracing::warn!("could not create debug messenger: {e:?}");
                    None
                }
            }
        } else {
            None
        };

        // SAFETY: query call on a valid entry.
        let version = unsafe { entry.try_enumerate_instance_version() }
            .ok()
            .flatten()
            .unwrap_or(vk::API_VERSION_1_0);
        tracing::info!(
            "Vulkan loader {}.{}.{}; validation={}, sync_validation={}",
            vk::api_version_major(version),
            vk::api_version_minor(version),
            vk::api_version_patch(version),
            validation,
            sync_validation
        );

        Ok(Arc::new(Self {
            entry,
            raw,
            debug,
            stats,
            validation_enabled: validation,
            sync_validation_enabled: sync_validation,
        }))
    }

    /// The loader entry points.
    pub fn entry(&self) -> &Entry {
        &self.entry
    }

    /// The raw `ash` instance.
    pub fn raw(&self) -> &ash::Instance {
        &self.raw
    }

    /// Validation message counters.
    pub fn validation_stats(&self) -> &Arc<ValidationStats> {
        &self.stats
    }

    /// `true` if the validation layer is active.
    pub fn validation_enabled(&self) -> bool {
        self.validation_enabled
    }

    /// `true` if synchronisation validation is active.
    pub fn sync_validation_enabled(&self) -> bool {
        self.sync_validation_enabled
    }
}

fn messenger_create_info(
    user_data: *mut std::ffi::c_void,
) -> vk::DebugUtilsMessengerCreateInfoEXT<'static> {
    vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(debug_callback))
        .user_data(user_data)
}

impl Drop for Instance {
    fn drop(&mut self) {
        // SAFETY: the messenger and instance were created by us and are destroyed exactly once,
        // messenger first. Every dependent object holds an `Arc<Instance>`, so none can outlive
        // this call.
        unsafe {
            if let Some(debug) = self.debug.take() {
                debug.loader.destroy_debug_utils_messenger(debug.raw, None);
            }
            self.raw.destroy_instance(None);
        }
    }
}
