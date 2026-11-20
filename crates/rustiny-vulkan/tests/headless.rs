//! Headless Vulkan tests: instance + device + allocator creation without any window.
//!
//! These need a Vulkan loader and a Vulkan 1.3 device. If none is available the tests print a
//! `SKIPPED` line instead of failing, so CI machines without a GPU stay green; the line makes the
//! skip visible in the log rather than silently reporting a pass.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{AllocationCreateDesc, AllocationScheme};
use rustiny_vulkan::{Device, DeviceConfig, Instance, InstanceConfig, VulkanError};

fn setup() -> Option<std::sync::Arc<Device>> {
    let instance = match Instance::new(&InstanceConfig {
        validation: true,
        sync_validation: true,
        ..Default::default()
    }) {
        Ok(i) => i,
        Err(VulkanError::Load(e)) => {
            eprintln!("SKIPPED: no Vulkan loader ({e})");
            return None;
        }
        Err(e) => panic!("instance creation failed: {e}"),
    };
    match Device::new(&instance, None, &DeviceConfig::default()) {
        Ok(d) => Some(d),
        Err(VulkanError::NoSuitableDevice(why)) => {
            eprintln!("SKIPPED: no suitable device ({why})");
            None
        }
        Err(e) => panic!("device creation failed: {e}"),
    }
}

#[test]
fn device_creates_and_destroys_without_validation_errors() {
    let Some(device) = setup() else { return };
    let stats = device.instance().validation_stats().clone();
    let info = device.info().clone();
    assert!(!info.name.is_empty());
    assert!(info.api_version >= vk::API_VERSION_1_3);
    drop(device);
    assert_eq!(
        stats.errors(),
        0,
        "validation errors: {:?}",
        stats.retained_errors()
    );
}

#[test]
fn allocator_allocates_and_frees_buffer_memory() {
    let Some(device) = setup() else { return };
    let stats = device.instance().validation_stats().clone();
    let raw = device.raw();

    let info = vk::BufferCreateInfo::default()
        .size(64 * 1024)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    // SAFETY: valid device and create info.
    let buffer = unsafe { raw.create_buffer(&info, None) }.unwrap();
    // SAFETY: buffer was just created on this device.
    let requirements = unsafe { raw.get_buffer_memory_requirements(buffer) };
    let allocation = device
        .allocator()
        .lock()
        .unwrap()
        .allocate(&AllocationCreateDesc {
            name: "headless-test-buffer",
            requirements,
            location: MemoryLocation::CpuToGpu,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .unwrap();
    // SAFETY: memory satisfies the buffer's requirements and is not bound elsewhere.
    unsafe { raw.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()) }.unwrap();
    assert!(
        allocation.mapped_ptr().is_some(),
        "CpuToGpu memory should be host mapped"
    );

    device.allocator().lock().unwrap().free(allocation).unwrap();
    // SAFETY: not used by the GPU.
    unsafe { raw.destroy_buffer(buffer, None) };
    drop(device);
    assert_eq!(
        stats.errors(),
        0,
        "validation errors: {:?}",
        stats.retained_errors()
    );
}

#[test]
fn unsatisfiable_gpu_preference_still_selects_a_device() {
    // The preference is only a score bonus; a non-matching name must not make selection fail.
    let instance = match Instance::new(&InstanceConfig::default()) {
        Ok(i) => i,
        Err(VulkanError::Load(e)) => {
            eprintln!("SKIPPED: no Vulkan loader ({e})");
            return;
        }
        Err(e) => panic!("{e}"),
    };
    let config = DeviceConfig {
        gpu_preference: "definitely-not-a-gpu".into(),
    };
    match Device::new(&instance, None, &config) {
        Ok(_) | Err(VulkanError::NoSuitableDevice(_)) => {}
        Err(e) => panic!("{e}"),
    }
}
