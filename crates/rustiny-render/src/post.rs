//! URP's post-processing: what the scene's volume profiles ask for ([`PostProcess`]), the bloom
//! pipelines and the chain of images bloom works in.

use ash::vk;
use rustiny_vulkan::{Device, Result, VkResultExt, VulkanError};

/// SPIR-V produced by `build.rs` from `shaders/bloom.wgsl`.
const BLOOM_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/bloom.spv"));

/// Push constant bytes of the bloom passes (four `vec4`s).
pub(crate) const BLOOM_PUSH: u32 = 64;
/// Push constant bytes of the final pass (seven `vec4`s).
pub(crate) const PRESENT_PUSH: u32 = 112;

/// The tonemapping operator (`Tonemapping.mode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Tonemap {
    /// No tonemapping: values are clipped at 1.
    #[default]
    None,
    /// URP's Neutral operator.
    Neutral,
}

/// `Vignette`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vignette {
    /// Linear colour of the darkening.
    pub color: [f32; 3],
    /// Centre in screen space (0..1).
    pub center: [f32; 2],
    /// `intensity` as in the volume (0..1).
    pub intensity: f32,
    /// `smoothness` as in the volume (0..1).
    pub smoothness: f32,
    /// `rounded`: circular whatever the aspect ratio.
    pub rounded: bool,
}

impl Default for Vignette {
    fn default() -> Self {
        Self {
            color: [0.0; 3],
            center: [0.5; 2],
            intensity: 0.0,
            smoothness: 0.2,
            rounded: false,
        }
    }
}

/// `Bloom`, with the values as the volume stores them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bloom {
    /// `threshold`, in gamma space (URP converts it to linear).
    pub threshold: f32,
    /// `intensity`.
    pub intensity: f32,
    /// `scatter` (0..1).
    pub scatter: f32,
    /// `clamp`: brightest value that takes part.
    pub clamp: f32,
    /// `tint`, linear.
    pub tint: [f32; 3],
    /// `highQualityFiltering`.
    pub high_quality: bool,
    /// `skipIterations`.
    pub skip_iterations: u32,
    /// `maxIterations`.
    pub max_iterations: u32,
}

impl Default for Bloom {
    fn default() -> Self {
        Self {
            threshold: 0.9,
            intensity: 0.0,
            scatter: 0.7,
            clamp: 65472.0,
            tint: [1.0; 3],
            high_quality: false,
            skip_iterations: 1,
            max_iterations: 6,
        }
    }
}

/// The post-processing of the camera, as URP's volume stack resolves it. The default does
/// nothing (no tonemapping, no vignette, no bloom).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PostProcess {
    /// `ColorAdjustments.postExposure` in stops.
    pub post_exposure: f32,
    /// `ColorAdjustments.contrast` (-100..100).
    pub contrast: f32,
    /// `ColorAdjustments.colorFilter`, linear.
    pub color_filter: [f32; 3],
    /// `ColorAdjustments.hueShift` in degrees.
    pub hue_shift: f32,
    /// `ColorAdjustments.saturation` (-100..100).
    pub saturation: f32,
    /// `Tonemapping.mode`.
    pub tonemap: Tonemap,
    /// `Vignette`.
    pub vignette: Vignette,
    /// `Bloom`, `None` when its intensity is zero.
    pub bloom: Option<Bloom>,
}

impl Default for PostProcess {
    fn default() -> Self {
        Self {
            post_exposure: 0.0,
            contrast: 0.0,
            color_filter: [1.0; 3],
            hue_shift: 0.0,
            saturation: 0.0,
            tonemap: Tonemap::None,
            vignette: Vignette::default(),
            bloom: None,
        }
    }
}

/// `Mathf.GammaToLinearSpace`.
pub(crate) fn gamma_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else if c < 1.0 {
        ((c + 0.055) / 1.055).powf(2.4)
    } else {
        c.powf(2.2)
    }
}

impl Bloom {
    /// How many levels URP uses for a target of `width` x `height` (half size first).
    pub(crate) fn mip_count(&self, width: u32, height: u32, available: u32) -> u32 {
        let largest = (width >> 1).max(height >> 1).max(1) as f32;
        let iterations = (largest.log2() - 1.0).floor() as i64 - i64::from(self.skip_iterations);
        let wanted = iterations.clamp(1, i64::from(self.max_iterations.max(1)));
        (wanted as u32).min(available).max(1)
    }

    /// `_Params` of the bloom passes: scatter, clamp, threshold (linear), threshold knee.
    pub(crate) fn params(&self) -> [f32; 4] {
        let threshold = gamma_to_linear(self.threshold);
        let scatter = 0.05 + (0.95 - 0.05) * self.scatter;
        [scatter, self.clamp, threshold, threshold * 0.5]
    }

    /// The tint the final pass multiplies bloom by: normalised to unit luminance, times the
    /// intensity.
    pub(crate) fn tint_times_intensity(&self) -> [f32; 3] {
        let [r, g, b] = self.tint;
        let luma = 0.2126729 * r + 0.7151522 * g + 0.072175 * b;
        let tint = if luma > 0.0 {
            [r / luma, g / luma, b / luma]
        } else {
            [1.0; 3]
        };
        tint.map(|c| c * self.intensity)
    }
}

impl PostProcess {
    /// The final pass's push constants (see `present.wgsl`), for a target of `width` x `height`
    /// whose bloom result is `bloom_size` pixels.
    pub(crate) fn present_push(
        &self,
        width: u32,
        height: u32,
        bloom_size: [u32; 2],
    ) -> [u8; PRESENT_PUSH as usize] {
        let v = &self.vignette;
        let aspect = width as f32 / height.max(1) as f32;
        let bloom = self.bloom.filter(|b| b.intensity > 0.0);
        let tint = bloom.map_or([0.0; 3], |b| b.tint_times_intensity());
        let [bw, bh] = bloom_size.map(|s| s.max(1) as f32);
        let values: [f32; 28] = [
            2f32.powf(self.post_exposure),
            self.contrast / 100.0 + 1.0,
            self.hue_shift / 360.0,
            self.saturation / 100.0 + 1.0,
            self.color_filter[0],
            self.color_filter[1],
            self.color_filter[2],
            match self.tonemap {
                Tonemap::None => 0.0,
                Tonemap::Neutral => 1.0,
            },
            v.color[0],
            v.color[1],
            v.color[2],
            if v.rounded { aspect } else { 1.0 },
            v.center[0],
            v.center[1],
            v.intensity * 3.0,
            v.smoothness * 5.0,
            tint[0],
            tint[1],
            tint[2],
            if bloom.is_some() { 1.0 } else { 0.0 },
            1.0 / bw,
            1.0 / bh,
            bw,
            bh,
            if bloom.is_some_and(|b| b.high_quality) {
                1.0
            } else {
                0.0
            },
            0.0,
            0.0,
            0.0,
        ];
        let mut bytes = [0u8; PRESENT_PUSH as usize];
        for (chunk, value) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(values) {
            *chunk = value.to_ne_bytes();
        }
        bytes
    }
}

/// The four bloom pipelines and the layout they share: a source and a lower-level texture, a
/// linear clamp sampler, and [`BLOOM_PUSH`] bytes of push constants.
pub(crate) struct BloomPipelines {
    pub(crate) set_layout: vk::DescriptorSetLayout,
    pub(crate) layout: vk::PipelineLayout,
    pub(crate) prefilter: vk::Pipeline,
    pub(crate) blur_h: vk::Pipeline,
    pub(crate) blur_v: vk::Pipeline,
    pub(crate) upsample: vk::Pipeline,
}

impl std::fmt::Debug for BloomPipelines {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BloomPipelines").finish_non_exhaustive()
    }
}

/// A set layout of two sampled images and a sampler, for the fragment stage.
pub(crate) fn two_textures_and_sampler(device: &Device) -> Result<vk::DescriptorSetLayout> {
    let binding = |index: u32, ty| {
        vk::DescriptorSetLayoutBinding::default()
            .binding(index)
            .descriptor_type(ty)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
    };
    let bindings = [
        binding(0, vk::DescriptorType::SAMPLED_IMAGE),
        binding(1, vk::DescriptorType::SAMPLED_IMAGE),
        binding(2, vk::DescriptorType::SAMPLER),
    ];
    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    // SAFETY: plain object creation on a live device.
    unsafe { device.raw().create_descriptor_set_layout(&info, None) }
        .vk("vkCreateDescriptorSetLayout(post)")
}

/// Fullscreen-triangle pipelines of one shader module writing one `format` colour target.
pub(crate) fn fullscreen_pipelines(
    device: &Device,
    spirv: &[u8],
    layout: vk::PipelineLayout,
    format: vk::Format,
    vertex: &std::ffi::CStr,
    fragments: &[&std::ffi::CStr],
) -> Result<Vec<vk::Pipeline>> {
    let raw = device.raw();
    let words: Vec<u32> = spirv
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    let module_info = vk::ShaderModuleCreateInfo::default().code(&words);
    // SAFETY: `words` is valid SPIR-V produced (and validated) by naga at build time.
    let module =
        unsafe { raw.create_shader_module(&module_info, None) }.vk("vkCreateShaderModule(post)")?;
    let stages: Vec<[vk::PipelineShaderStageCreateInfo<'_>; 2]> = fragments
        .iter()
        .map(|fragment| {
            [
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(module)
                    .name(vertex),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(module)
                    .name(fragment),
            ]
        })
        .collect();
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let attachments = [vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)];
    let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let formats = [format];
    let mut renderings: Vec<vk::PipelineRenderingCreateInfo<'_>> = fragments
        .iter()
        .map(|_| vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&formats))
        .collect();
    let infos: Vec<vk::GraphicsPipelineCreateInfo<'_>> = stages
        .iter()
        .zip(renderings.iter_mut())
        .map(|(stages, rendering)| {
            vk::GraphicsPipelineCreateInfo::default()
                .stages(stages)
                .vertex_input_state(&vertex_input)
                .input_assembly_state(&input_assembly)
                .viewport_state(&viewport)
                .rasterization_state(&raster)
                .multisample_state(&multisample)
                .color_blend_state(&blend)
                .dynamic_state(&dynamic)
                .layout(layout)
                .push_next(rendering)
        })
        .collect();
    // SAFETY: every structure referenced by `infos` outlives this call.
    let created = unsafe { raw.create_graphics_pipelines(vk::PipelineCache::null(), &infos, None) };
    // SAFETY: the pipelines keep what they need from the module.
    unsafe { raw.destroy_shader_module(module, None) };
    match created {
        Ok(pipelines) => Ok(pipelines),
        Err((pipelines, result)) => {
            for pipeline in pipelines.into_iter().filter(|p| *p != vk::Pipeline::null()) {
                // SAFETY: created by the failed call, never used.
                unsafe { raw.destroy_pipeline(pipeline, None) };
            }
            Err(VulkanError::Call {
                call: "vkCreateGraphicsPipelines(post)",
                result,
            })
        }
    }
}

/// A pipeline layout with one set and `push` bytes of fragment push constants.
pub(crate) fn layout_with_push(
    device: &Device,
    set_layout: vk::DescriptorSetLayout,
    push: u32,
) -> Result<vk::PipelineLayout> {
    let set_layouts = [set_layout];
    let ranges = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .offset(0)
        .size(push)];
    let info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&ranges);
    // SAFETY: plain object creation; the set layout is live.
    unsafe { device.raw().create_pipeline_layout(&info, None) }.vk("vkCreatePipelineLayout(post)")
}

impl BloomPipelines {
    pub(crate) fn new(device: &Device, format: vk::Format) -> Result<Self> {
        let raw = device.raw();
        let set_layout = two_textures_and_sampler(device)?;
        let layout = match layout_with_push(device, set_layout, BLOOM_PUSH) {
            Ok(layout) => layout,
            Err(e) => {
                // SAFETY: never used.
                unsafe { raw.destroy_descriptor_set_layout(set_layout, None) };
                return Err(e);
            }
        };
        let pipelines = fullscreen_pipelines(
            device,
            BLOOM_SPV,
            layout,
            format,
            c"bloom_vs",
            &[c"prefilter_fs", c"blur_h_fs", c"blur_v_fs", c"upsample_fs"],
        );
        match pipelines {
            Ok(p) => Ok(Self {
                set_layout,
                layout,
                prefilter: p[0],
                blur_h: p[1],
                blur_v: p[2],
                upsample: p[3],
            }),
            Err(e) => {
                // SAFETY: never used.
                unsafe {
                    raw.destroy_pipeline_layout(layout, None);
                    raw.destroy_descriptor_set_layout(set_layout, None);
                }
                Err(e)
            }
        }
    }

    pub(crate) fn destroy(&self, device: &Device) {
        let raw = device.raw();
        // SAFETY: the owner guarantees the GPU no longer uses these objects.
        unsafe {
            for pipeline in [self.prefilter, self.blur_h, self.blur_v, self.upsample] {
                raw.destroy_pipeline(pipeline, None);
            }
            raw.destroy_pipeline_layout(self.layout, None);
            raw.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bloom_levels_follow_urp() {
        let bloom = Bloom {
            skip_iterations: 0,
            max_iterations: 6,
            ..Bloom::default()
        };
        // 1600x900: half is 800, log2(800) - 1 = 8.6 -> 8, at most 6.
        assert_eq!(bloom.mip_count(1600, 900, 16), 6);
        // 128x96: half 64 -> log2(64) - 1 = 5.
        assert_eq!(bloom.mip_count(128, 96, 16), 5);
        let skip = Bloom {
            skip_iterations: 2,
            ..bloom
        };
        assert_eq!(skip.mip_count(128, 96, 16), 3);
        assert_eq!(bloom.mip_count(4, 4, 16), 1, "at least one level");
        assert_eq!(bloom.mip_count(1600, 900, 3), 3, "no more than there are");
    }

    #[test]
    fn bloom_parameters_are_urps() {
        let bloom = Bloom {
            threshold: 3.8,
            scatter: 0.6,
            intensity: 2.2,
            ..Bloom::default()
        };
        let [scatter, clamp, threshold, knee] = bloom.params();
        assert!((scatter - 0.59).abs() < 1e-6);
        assert_eq!(clamp, 65472.0);
        assert!(
            (threshold - 3.8f32.powf(2.2)).abs() < 1e-3,
            "gamma to linear"
        );
        assert!((knee - threshold * 0.5).abs() < 1e-6);
        let tint = bloom.tint_times_intensity();
        assert!(
            tint.iter().all(|c| (c - 2.2).abs() < 1e-5),
            "white tint has unit luminance"
        );
    }
}
