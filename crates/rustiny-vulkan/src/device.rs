//! Physical device selection, logical device and GPU memory allocator.

use crate::error::{Result, VkResultExt, VulkanError};
use crate::instance::Instance;
use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use std::ffi::CStr;
use std::sync::{Arc, Mutex};

/// Queue families chosen for the logical device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueFamilies {
    /// Family used for graphics, presentation and (for now) all rendering work.
    pub graphics: u32,
    /// Dedicated transfer-only family, if the GPU exposes one (async uploads).
    pub transfer: Option<u32>,
    /// Dedicated compute-only family, if the GPU exposes one (async compute).
    pub compute: Option<u32>,
}

/// Human readable facts about the selected GPU.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Marketing name reported by the driver.
    pub name: String,
    /// Integrated, discrete, virtual or CPU.
    pub device_type: vk::PhysicalDeviceType,
    /// Highest Vulkan API version the device supports.
    pub api_version: u32,
    /// Driver-specific version number.
    pub driver_version: u32,
    /// PCI vendor id.
    pub vendor_id: u32,
}

/// Surface information needed to pick a device that can present to it.
#[derive(Clone, Copy)]
pub struct PresentTarget<'a> {
    /// Surface extension loader.
    pub loader: &'a ash::khr::surface::Instance,
    /// Surface to test presentation support against.
    pub surface: vk::SurfaceKHR,
}

impl std::fmt::Debug for PresentTarget<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PresentTarget")
            .field("surface", &self.surface)
            .finish_non_exhaustive()
    }
}

/// Selection preferences.
#[derive(Debug, Clone, Default)]
pub struct DeviceConfig {
    /// Case-insensitive substring of the preferred GPU name. Empty = pick the best automatically.
    pub gpu_preference: String,
}

/// Logical device, queues and allocator.
///
/// Dropping the device destroys the allocator and the `VkDevice`. Every object created from the
/// device must have been destroyed by then (the dependent wrappers in this crate hold an
/// `Arc<Device>` to guarantee that), and external users of [`Device::allocator`] must have
/// released their clones.
pub struct Device {
    instance: Arc<Instance>,
    physical: vk::PhysicalDevice,
    raw: ash::Device,
    info: DeviceInfo,
    families: QueueFamilies,
    graphics_queue: vk::Queue,
    transfer_queue: Option<vk::Queue>,
    compute_queue: Option<vk::Queue>,
    allocator: Option<Arc<Mutex<Allocator>>>,
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device")
            .field("info", &self.info)
            .field("families", &self.families)
            .finish()
    }
}

struct Candidate {
    physical: vk::PhysicalDevice,
    info: DeviceInfo,
    families: QueueFamilies,
    score: i64,
    has_swapchain: bool,
}

impl Device {
    /// Select a GPU and create the logical device.
    ///
    /// With `present` set, only devices able to present to that surface are considered and the
    /// swapchain extension is enabled.
    pub fn new(
        instance: &Arc<Instance>,
        present: Option<PresentTarget<'_>>,
        config: &DeviceConfig,
    ) -> Result<Arc<Self>> {
        let raw_instance = instance.raw();
        // SAFETY: query call on a live instance.
        let physical_devices = unsafe { raw_instance.enumerate_physical_devices() }
            .vk("vkEnumeratePhysicalDevices")?;

        let mut rejected = Vec::new();
        let mut candidates = Vec::new();
        for physical in physical_devices {
            match evaluate(instance, physical, present, &config.gpu_preference) {
                Ok(c) => candidates.push(c),
                Err(reason) => rejected.push(reason),
            }
        }
        for reason in &rejected {
            tracing::info!("GPU rejected: {reason}");
        }
        let best = candidates
            .into_iter()
            .max_by_key(|c| c.score)
            .ok_or_else(|| {
                VulkanError::NoSuitableDevice(if rejected.is_empty() {
                    "no Vulkan physical devices found".to_owned()
                } else {
                    rejected.join("; ")
                })
            })?;

        let Candidate {
            physical,
            info,
            families,
            has_swapchain,
            ..
        } = best;
        tracing::info!(
            "selected GPU `{}` ({:?}), Vulkan {}.{}.{}, queues: graphics={} transfer={:?} compute={:?}",
            info.name,
            info.device_type,
            vk::api_version_major(info.api_version),
            vk::api_version_minor(info.api_version),
            vk::api_version_patch(info.api_version),
            families.graphics,
            families.transfer,
            families.compute
        );

        // One queue per distinct family.
        let mut family_ids = vec![families.graphics];
        family_ids.extend(families.transfer);
        family_ids.extend(families.compute);
        family_ids.sort_unstable();
        family_ids.dedup();
        let priorities = [1.0_f32];
        let queue_infos: Vec<_> = family_ids
            .iter()
            .map(|&family| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(family)
                    .queue_priorities(&priorities)
            })
            .collect();

        let mut extensions = Vec::new();
        // Only request the swapchain extension when presenting: it requires VK_KHR_surface at
        // instance level, which headless instances do not enable.
        if present.is_some() && has_swapchain {
            extensions.push(ash::khr::swapchain::NAME.as_ptr());
        }

        let mut features13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);
        let mut features2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut features13);
        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&extensions)
            .push_next(&mut features2);

        // SAFETY: `create_info` and everything it points to are alive for the call.
        let raw = unsafe { raw_instance.create_device(physical, &create_info, None) }
            .vk("vkCreateDevice")?;

        // SAFETY: the queue families/indices were requested in `queue_infos`.
        let graphics_queue = unsafe { raw.get_device_queue(families.graphics, 0) };
        // SAFETY: as above.
        let transfer_queue = families
            .transfer
            .map(|f| unsafe { raw.get_device_queue(f, 0) });
        // SAFETY: as above.
        let compute_queue = families
            .compute
            .map(|f| unsafe { raw.get_device_queue(f, 0) });

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: raw_instance.clone(),
            device: raw.clone(),
            physical_device: physical,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        });
        let allocator = match allocator {
            Ok(a) => a,
            Err(e) => {
                // SAFETY: device was just created and nothing else uses it.
                unsafe { raw.destroy_device(None) };
                return Err(VulkanError::Allocator(e.to_string()));
            }
        };

        Ok(Arc::new(Self {
            instance: instance.clone(),
            physical,
            raw,
            info,
            families,
            graphics_queue,
            transfer_queue,
            compute_queue,
            allocator: Some(Arc::new(Mutex::new(allocator))),
        }))
    }

    /// The owning instance.
    pub fn instance(&self) -> &Arc<Instance> {
        &self.instance
    }

    /// The selected physical device.
    pub fn physical(&self) -> vk::PhysicalDevice {
        self.physical
    }

    /// The raw `ash` device (cheap to clone, but clones must not outlive the [`Device`]).
    pub fn raw(&self) -> &ash::Device {
        &self.raw
    }

    /// Facts about the GPU.
    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    /// Chosen queue families.
    pub fn families(&self) -> QueueFamilies {
        self.families
    }

    /// Graphics + present queue.
    pub fn graphics_queue(&self) -> vk::Queue {
        self.graphics_queue
    }

    /// Dedicated transfer queue, if any.
    pub fn transfer_queue(&self) -> Option<vk::Queue> {
        self.transfer_queue
    }

    /// Dedicated compute queue, if any.
    pub fn compute_queue(&self) -> Option<vk::Queue> {
        self.compute_queue
    }

    /// Shared GPU memory allocator.
    ///
    /// Clones of this `Arc` must be dropped before the [`Device`] is.
    pub fn allocator(&self) -> &Arc<Mutex<Allocator>> {
        self.allocator
            .as_ref()
            .expect("allocator is only taken during Drop")
    }

    /// Block until the device is idle. Use sparingly (resize, shutdown) - never per frame.
    pub fn wait_idle(&self) -> Result<()> {
        // SAFETY: valid device handle.
        unsafe { self.raw.device_wait_idle() }.vk("vkDeviceWaitIdle")
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        if let Err(e) = self.wait_idle() {
            tracing::error!("device wait before destruction failed: {e}");
        }
        if let Some(allocator) = self.allocator.take() {
            let holders = Arc::strong_count(&allocator);
            if holders > 1 {
                // Destroying the device while another owner can still free memory through the
                // allocator would be undefined behaviour later; make the bug loud.
                tracing::error!(
                    "{} extra owner(s) of the GPU allocator outlive the device; \
                     shut down all GPU users before dropping the device",
                    holders - 1
                );
            }
            drop(allocator);
        }
        // SAFETY: the device is idle, the allocator is gone and all dependent objects hold an
        // `Arc<Device>`, so nothing else can still be using it.
        unsafe { self.raw.destroy_device(None) };
    }
}

fn evaluate(
    instance: &Instance,
    physical: vk::PhysicalDevice,
    present: Option<PresentTarget<'_>>,
    preference: &str,
) -> std::result::Result<Candidate, String> {
    let raw = instance.raw();
    // SAFETY: query calls on a physical device obtained from this instance.
    let props = unsafe { raw.get_physical_device_properties(physical) };
    let name = props
        .device_name_as_c_str()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|_| "<unnamed>".to_owned());

    if props.api_version < vk::API_VERSION_1_3 {
        return Err(format!(
            "`{name}` supports only Vulkan {}.{} (1.3 required for dynamic rendering)",
            vk::api_version_major(props.api_version),
            vk::api_version_minor(props.api_version)
        ));
    }

    let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
    let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut f13);
    // SAFETY: chained structs are valid for the call.
    unsafe { raw.get_physical_device_features2(physical, &mut f2) };
    if f13.dynamic_rendering == vk::FALSE || f13.synchronization2 == vk::FALSE {
        return Err(format!("`{name}` lacks dynamicRendering/synchronization2"));
    }

    // SAFETY: query call.
    let extensions = unsafe { raw.enumerate_device_extension_properties(physical) }
        .map_err(|e| format!("`{name}`: cannot enumerate extensions: {e:?}"))?;
    let has_swapchain = extensions.iter().any(|e| {
        e.extension_name_as_c_str()
            .is_ok_and(|n: &CStr| n == ash::khr::swapchain::NAME)
    });
    if present.is_some() && !has_swapchain {
        return Err(format!("`{name}` lacks VK_KHR_swapchain"));
    }

    // SAFETY: query call.
    let queue_props = unsafe { raw.get_physical_device_queue_family_properties(physical) };
    let mut graphics = None;
    for (i, q) in queue_props.iter().enumerate() {
        if !q.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
            continue;
        }
        let index = i as u32;
        if let Some(target) = present {
            // SAFETY: surface and physical device belong to the same instance.
            let ok = unsafe {
                target
                    .loader
                    .get_physical_device_surface_support(physical, index, target.surface)
            }
            .map_err(|e| format!("`{name}`: surface support query failed: {e:?}"))?;
            if !ok {
                continue;
            }
        }
        graphics = Some(index);
        break;
    }
    let graphics = graphics.ok_or_else(|| {
        format!(
            "`{name}` has no graphics queue family{}",
            if present.is_some() {
                " that can present"
            } else {
                ""
            }
        )
    })?;

    let dedicated = |wanted: vk::QueueFlags, unwanted: vk::QueueFlags| {
        queue_props
            .iter()
            .position(|q| q.queue_flags.contains(wanted) && !q.queue_flags.intersects(unwanted))
            .map(|i| i as u32)
    };
    let transfer = dedicated(
        vk::QueueFlags::TRANSFER,
        vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE,
    );
    let compute = dedicated(vk::QueueFlags::COMPUTE, vk::QueueFlags::GRAPHICS);

    let mut score = match props.device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => 1000,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 500,
        vk::PhysicalDeviceType::VIRTUAL_GPU => 250,
        _ => 10,
    };
    if !preference.is_empty() && name.to_lowercase().contains(&preference.to_lowercase()) {
        score += 100_000;
    }

    Ok(Candidate {
        physical,
        info: DeviceInfo {
            name,
            device_type: props.device_type,
            api_version: props.api_version,
            driver_version: props.driver_version,
            vendor_id: props.vendor_id,
        },
        families: QueueFamilies {
            graphics,
            transfer,
            compute,
        },
        score,
        has_swapchain,
    })
}
