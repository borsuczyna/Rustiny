//! Shader modules and the two graphics pipelines of the scene pass.

use crate::geometry::VERTEX_STRIDE;
use ash::vk;
use rustiny_vulkan::{Device, Result, VkResultExt, VulkanError};

/// SPIR-V produced by `build.rs` from `shaders/scene.wgsl`.
const SCENE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/scene.spv"));

/// Size in bytes of the push constants: view-projection (64) + eye (16) + params (16) +
/// material block (16: the index of the section's record, then padding).
pub(crate) const PUSH_SIZE: u32 = 112;
/// Offset of the material block, which is pushed again for every section drawn.
pub(crate) const MATERIAL_OFFSET: u32 = 96;

/// Pipeline layout and the pipelines that use it.
pub(crate) struct Pipelines {
    pub(crate) layout: vk::PipelineLayout,
    pub(crate) meshes: vk::Pipeline,
    pub(crate) grid: vk::Pipeline,
}

impl std::fmt::Debug for Pipelines {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipelines").finish_non_exhaustive()
    }
}

fn spirv_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

impl Pipelines {
    pub(crate) fn new(
        device: &Device,
        texture_layout: vk::DescriptorSetLayout,
        material_layout: vk::DescriptorSetLayout,
        color_format: vk::Format,
        depth_format: vk::Format,
    ) -> Result<Self> {
        let raw = device.raw();
        let words = spirv_words(SCENE_SPV);
        let module_info = vk::ShaderModuleCreateInfo::default().code(&words);
        // SAFETY: `words` is valid SPIR-V produced (and validated) by naga at build time.
        let module =
            unsafe { raw.create_shader_module(&module_info, None) }.vk("vkCreateShaderModule")?;

        let result = Self::build(
            device,
            module,
            [texture_layout, material_layout],
            color_format,
            depth_format,
        );
        // SAFETY: pipelines keep what they need from the module; it is not used any more.
        unsafe { raw.destroy_shader_module(module, None) };
        result
    }

    fn build(
        device: &Device,
        module: vk::ShaderModule,
        set_layouts: [vk::DescriptorSetLayout; 2],
        color_format: vk::Format,
        depth_format: vk::Format,
    ) -> Result<Self> {
        let raw = device.raw();
        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(PUSH_SIZE)];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&ranges);
        // SAFETY: plain object creation.
        let layout = unsafe { raw.create_pipeline_layout(&layout_info, None) }
            .vk("vkCreatePipelineLayout")?;

        let stages = |vs: &'static std::ffi::CStr, fs: &'static std::ffi::CStr| {
            [
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(module)
                    .name(vs),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(module)
                    .name(fs),
            ]
        };

        // Binding 0 = per-vertex position, normal, uv; binding 1 = per-instance model, colour,
        // object id. The proxy box and imported meshes share this layout and the pipeline.
        let bindings = [
            vk::VertexInputBindingDescription::default()
                .binding(0)
                .stride(VERTEX_STRIDE as u32)
                .input_rate(vk::VertexInputRate::VERTEX),
            vk::VertexInputBindingDescription::default()
                .binding(1)
                .stride(crate::GpuInstance::SIZE as u32)
                .input_rate(vk::VertexInputRate::INSTANCE),
        ];
        let attribute = |location: u32, binding: u32, format: vk::Format, offset: u32| {
            vk::VertexInputAttributeDescription::default()
                .location(location)
                .binding(binding)
                .format(format)
                .offset(offset)
        };
        let attributes = [
            attribute(0, 0, vk::Format::R32G32B32_SFLOAT, 0),
            attribute(1, 0, vk::Format::R32G32B32_SFLOAT, 12),
            attribute(2, 0, vk::Format::R32G32_SFLOAT, 24),
            attribute(3, 1, vk::Format::R32G32B32A32_SFLOAT, 0),
            attribute(4, 1, vk::Format::R32G32B32A32_SFLOAT, 16),
            attribute(5, 1, vk::Format::R32G32B32A32_SFLOAT, 32),
            attribute(6, 1, vk::Format::R32G32B32A32_SFLOAT, 48),
            attribute(7, 1, vk::Format::R32G32B32A32_SFLOAT, 64),
            attribute(8, 1, vk::Format::R32_UINT, 80),
        ];
        let mesh_vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&bindings)
            .vertex_attribute_descriptions(&attributes);
        let no_vertex_input = vk::PipelineVertexInputStateCreateInfo::default();

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        // No culling: instances may carry mirrored (negative-scale) transforms, and the depth
        // test hides the back faces anyway.
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        // Reverse-Z: nearer fragments have the larger depth value.
        let mesh_depth = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::GREATER);
        let grid_depth = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(false)
            .depth_compare_op(vk::CompareOp::GREATER_OR_EQUAL);

        let opaque = [vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let alpha = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let opaque_blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&opaque);
        let alpha_blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&alpha);

        let color_formats = [color_format];
        let mut mesh_rendering = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_formats)
            .depth_attachment_format(depth_format);
        let mut grid_rendering = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_formats)
            .depth_attachment_format(depth_format);

        let mesh_stages = stages(c"mesh_vs", c"mesh_fs");
        let grid_stages = stages(c"grid_vs", c"grid_fs");
        let infos = [
            vk::GraphicsPipelineCreateInfo::default()
                .stages(&mesh_stages)
                .vertex_input_state(&mesh_vertex_input)
                .input_assembly_state(&input_assembly)
                .viewport_state(&viewport)
                .rasterization_state(&raster)
                .multisample_state(&multisample)
                .depth_stencil_state(&mesh_depth)
                .color_blend_state(&opaque_blend)
                .dynamic_state(&dynamic)
                .layout(layout)
                .push_next(&mut mesh_rendering),
            vk::GraphicsPipelineCreateInfo::default()
                .stages(&grid_stages)
                .vertex_input_state(&no_vertex_input)
                .input_assembly_state(&input_assembly)
                .viewport_state(&viewport)
                .rasterization_state(&raster)
                .multisample_state(&multisample)
                .depth_stencil_state(&grid_depth)
                .color_blend_state(&alpha_blend)
                .dynamic_state(&dynamic)
                .layout(layout)
                .push_next(&mut grid_rendering),
        ];
        // SAFETY: every structure referenced by `infos` outlives this call.
        let created =
            unsafe { raw.create_graphics_pipelines(vk::PipelineCache::null(), &infos, None) };
        match created {
            Ok(pipelines) => Ok(Self {
                layout,
                meshes: pipelines[0],
                grid: pipelines[1],
            }),
            Err((pipelines, result)) => {
                for pipeline in pipelines.into_iter().filter(|p| *p != vk::Pipeline::null()) {
                    // SAFETY: created by the failed call, never used.
                    unsafe { raw.destroy_pipeline(pipeline, None) };
                }
                // SAFETY: never used.
                unsafe { raw.destroy_pipeline_layout(layout, None) };
                Err(VulkanError::Call {
                    call: "vkCreateGraphicsPipelines",
                    result,
                })
            }
        }
    }

    pub(crate) fn destroy(&self, device: &Device) {
        // SAFETY: the owner guarantees the GPU no longer uses the pipelines.
        unsafe {
            device.raw().destroy_pipeline(self.meshes, None);
            device.raw().destroy_pipeline(self.grid, None);
            device.raw().destroy_pipeline_layout(self.layout, None);
        }
    }
}
