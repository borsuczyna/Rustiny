//! The last step of the scene pass, URP's "uber" pass: the HDR image the scene is rendered into
//! (linear light, like URP's camera target) gets bloom, vignette, exposure, colour grading and
//! tonemapping and is written to the display image as sRGB. Also owns the linear clamp sampler
//! the post-processing passes read with.

use crate::post::{PRESENT_PUSH, fullscreen_pipelines, layout_with_push, two_textures_and_sampler};
use ash::vk;
use rustiny_vulkan::{Device, Result, VkResultExt};

/// SPIR-V produced by `build.rs` from `shaders/present.wgsl`.
const PRESENT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/present.spv"));

/// The pipeline of the final pass, its layout (scene colour, bloom, sampler), and the sampler.
pub(crate) struct Present {
    pub(crate) set_layout: vk::DescriptorSetLayout,
    pub(crate) layout: vk::PipelineLayout,
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) sampler: vk::Sampler,
}

impl std::fmt::Debug for Present {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Present").finish_non_exhaustive()
    }
}

impl Present {
    pub(crate) fn new(device: &Device, display_format: vk::Format) -> Result<Self> {
        let raw = device.raw();
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
        // SAFETY: plain object creation on a live device.
        let sampler =
            unsafe { raw.create_sampler(&sampler_info, None) }.vk("vkCreateSampler(post)")?;
        let set_layout = match two_textures_and_sampler(device) {
            Ok(l) => l,
            Err(e) => {
                // SAFETY: never used.
                unsafe { raw.destroy_sampler(sampler, None) };
                return Err(e);
            }
        };
        let layout = match layout_with_push(device, set_layout, PRESENT_PUSH) {
            Ok(l) => l,
            Err(e) => {
                // SAFETY: never used.
                unsafe {
                    raw.destroy_descriptor_set_layout(set_layout, None);
                    raw.destroy_sampler(sampler, None);
                }
                return Err(e);
            }
        };
        match fullscreen_pipelines(
            device,
            PRESENT_SPV,
            layout,
            display_format,
            c"present_vs",
            &[c"present_fs"],
        ) {
            Ok(p) => Ok(Self {
                set_layout,
                layout,
                pipeline: p[0],
                sampler,
            }),
            Err(e) => {
                // SAFETY: never used.
                unsafe {
                    raw.destroy_pipeline_layout(layout, None);
                    raw.destroy_descriptor_set_layout(set_layout, None);
                    raw.destroy_sampler(sampler, None);
                }
                Err(e)
            }
        }
    }

    pub(crate) fn destroy(&self, device: &Device) {
        let raw = device.raw();
        // SAFETY: the owner guarantees the GPU no longer uses these objects.
        unsafe {
            raw.destroy_pipeline(self.pipeline, None);
            raw.destroy_pipeline_layout(self.layout, None);
            raw.destroy_descriptor_set_layout(self.set_layout, None);
            raw.destroy_sampler(self.sampler, None);
        }
    }
}
