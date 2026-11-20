//! One compiled pass of a Unity shader: its descriptor layout (decided from the reflection of both
//! stages), its pipeline and what it needs from the vertex stream.

use super::{Blend, BlendFactor, DepthTest, PassState, ShaderPassDesc};
use ash::vk;
use rustiny_shader::{Member, Reflection, ResourceKind, UniformBlock, rebind, reflect};
use rustiny_vulkan::{Device, Result, VkResultExt, VulkanError};
use std::collections::BTreeMap;

/// The vertex buffer binding that holds the mesh (position, normal, uv).
pub(crate) const MESH_BINDING: u32 = 0;
/// The vertex buffer binding that holds [`CONSTANT_VERTEX`] with a stride of zero, so every vertex
/// reads the same defaults for what the mesh does not have (tangent, colour, extra UV sets).
pub(crate) const CONSTANT_BINDING: u32 = 2;
/// Bytes of the constant vertex data: tangent, colour and four zeros.
pub(crate) const CONSTANT_VERTEX: [f32; 12] = [
    1.0, 0.0, 0.0, 1.0, // tangent
    1.0, 1.0, 1.0, 1.0, // colour
    0.0, 0.0, 0.0, 0.0, // other attributes
];

/// What a descriptor of the pass is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    /// `UnityPerDraw`: written for every draw.
    PerDraw,
    /// `UnityPerMaterial`: written once per material.
    PerMaterial,
    /// Any other constant buffer: written per material and frame.
    Frame,
    /// A texture of this kind.
    Texture(ResourceKind),
    /// A sampler.
    Sampler,
    /// A structured buffer.
    Storage,
}

/// A binding of the pass's descriptor set.
#[derive(Debug, Clone)]
pub(crate) struct Descriptor {
    pub name: String,
    pub binding: u32,
    pub kind: Kind,
    /// The constant buffer's layout.
    pub block: Option<UniformBlock>,
}

impl Descriptor {
    /// Whether the buffer is bound with a dynamic offset.
    pub(crate) fn is_dynamic(&self) -> bool {
        matches!(self.kind, Kind::PerDraw | Kind::Frame)
    }
}

/// The members of `UnityPerDraw` the renderer writes for every draw.
#[derive(Debug, Clone, Default)]
pub(crate) struct DrawMembers {
    pub object_to_world: Option<Member>,
    pub world_to_object: Option<Member>,
    pub transform_params: Option<Member>,
}

/// A pass ready to draw with.
pub(crate) struct Pass {
    pub layout: vk::PipelineLayout,
    pub set_layout: vk::DescriptorSetLayout,
    pub pipeline: vk::Pipeline,
    /// Every descriptor, by binding.
    pub descriptors: Vec<Descriptor>,
    /// Indices (into `descriptors`) of the dynamic buffers in binding order: the order the
    /// dynamic offsets are given in.
    pub dynamic: Vec<usize>,
    /// Indices of the texture descriptors in binding order.
    pub textures: Vec<usize>,
    pub per_draw: Option<usize>,
    pub per_material: Option<usize>,
    pub draw_members: DrawMembers,
    /// The pass blends its output: it is drawn after the opaque ones.
    pub transparent: bool,
    /// The pass samples the camera's depth or opaque colour texture.
    pub camera_textures: bool,
}

impl std::fmt::Debug for Pass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pass")
            .field("descriptors", &self.descriptors.len())
            .finish_non_exhaustive()
    }
}

fn invalid(what: impl std::fmt::Display) -> VulkanError {
    VulkanError::InvalidInput(format!("unity shader pass: {what}"))
}

/// The union of what both stages declare, one descriptor per name.
fn merge(stages: [&Reflection; 2]) -> Result<BTreeMap<String, Descriptor>> {
    let mut found: BTreeMap<String, Descriptor> = BTreeMap::new();
    for reflection in stages {
        for block in &reflection.blocks {
            let kind = match block.name.as_str() {
                "UnityPerDraw" => Kind::PerDraw,
                "UnityPerMaterial" => Kind::PerMaterial,
                _ => Kind::Frame,
            };
            if let Some(existing) = found.get(&block.name) {
                let same = existing
                    .block
                    .as_ref()
                    .is_some_and(|b| b.size == block.size);
                if !same {
                    return Err(invalid(format!(
                        "the stages disagree about the size of {}",
                        block.name
                    )));
                }
                continue;
            }
            found.insert(
                block.name.clone(),
                Descriptor {
                    name: block.name.clone(),
                    binding: 0,
                    kind,
                    block: Some(block.clone()),
                },
            );
        }
        for resource in &reflection.resources {
            let kind = match resource.kind {
                ResourceKind::Sampler => Kind::Sampler,
                ResourceKind::StorageBuffer => Kind::Storage,
                other => Kind::Texture(other),
            };
            if let Some(existing) = found.get(&resource.name) {
                if existing.kind != kind {
                    return Err(invalid(format!(
                        "{} is declared as two different resources",
                        resource.name
                    )));
                }
                continue;
            }
            found.insert(
                resource.name.clone(),
                Descriptor {
                    name: resource.name.clone(),
                    binding: 0,
                    kind,
                    block: None,
                },
            );
        }
    }
    Ok(found)
}

fn depth_op(test: DepthTest) -> vk::CompareOp {
    // The depth buffer is reverse-Z (nearer is larger), so Unity's "less" is Vulkan's "greater".
    match test {
        DepthTest::Never => vk::CompareOp::NEVER,
        DepthTest::Less => vk::CompareOp::GREATER,
        DepthTest::Equal => vk::CompareOp::EQUAL,
        DepthTest::LessEqual => vk::CompareOp::GREATER_OR_EQUAL,
        DepthTest::Greater => vk::CompareOp::LESS,
        DepthTest::NotEqual => vk::CompareOp::NOT_EQUAL,
        DepthTest::GreaterEqual => vk::CompareOp::LESS_OR_EQUAL,
        DepthTest::Always => vk::CompareOp::ALWAYS,
    }
}

fn blend_factor(factor: BlendFactor) -> vk::BlendFactor {
    match factor {
        BlendFactor::Zero => vk::BlendFactor::ZERO,
        BlendFactor::One => vk::BlendFactor::ONE,
        BlendFactor::SrcColor => vk::BlendFactor::SRC_COLOR,
        BlendFactor::OneMinusSrcColor => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
        BlendFactor::SrcAlpha => vk::BlendFactor::SRC_ALPHA,
        BlendFactor::OneMinusSrcAlpha => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        BlendFactor::DstColor => vk::BlendFactor::DST_COLOR,
        BlendFactor::OneMinusDstColor => vk::BlendFactor::ONE_MINUS_DST_COLOR,
        BlendFactor::DstAlpha => vk::BlendFactor::DST_ALPHA,
        BlendFactor::OneMinusDstAlpha => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
    }
}

/// Split `TEXCOORD3` into `("TEXCOORD", 3)`.
fn split_semantic(semantic: &str) -> (&str, u32) {
    let base = semantic.trim_end_matches(|c: char| c.is_ascii_digit());
    let index = semantic[base.len()..].parse().unwrap_or(0);
    (base, index)
}

impl Pass {
    pub(crate) fn new(
        device: &Device,
        desc: &ShaderPassDesc,
        color_format: vk::Format,
        depth_format: vk::Format,
    ) -> Result<Self> {
        let mut vertex = desc.vertex.clone();
        let mut fragment = desc.fragment.clone();
        let vertex_info = reflect(&vertex).map_err(|e| invalid(format!("vertex stage: {e}")))?;
        let fragment_info =
            reflect(&fragment).map_err(|e| invalid(format!("fragment stage: {e}")))?;

        let mut found = merge([&vertex_info, &fragment_info])?;
        for (binding, descriptor) in found.values_mut().enumerate() {
            descriptor.binding = binding as u32;
        }
        let bindings: BTreeMap<&str, u32> = found
            .values()
            .map(|d| (d.name.as_str(), d.binding))
            .collect();
        for words in [&mut vertex, &mut fragment] {
            rebind(words, |name| bindings.get(name).map(|b| (0, *b)))
                .map_err(|e| invalid(e.to_string()))?;
        }
        let descriptors: Vec<Descriptor> = found.into_values().collect();

        let layout_bindings: Vec<vk::DescriptorSetLayoutBinding> = descriptors
            .iter()
            .map(|d| {
                let ty = match d.kind {
                    Kind::PerDraw | Kind::Frame => vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                    Kind::PerMaterial => vk::DescriptorType::UNIFORM_BUFFER,
                    Kind::Texture(_) => vk::DescriptorType::SAMPLED_IMAGE,
                    Kind::Sampler => vk::DescriptorType::SAMPLER,
                    Kind::Storage => vk::DescriptorType::STORAGE_BUFFER,
                };
                vk::DescriptorSetLayoutBinding::default()
                    .binding(d.binding)
                    .descriptor_type(ty)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            })
            .collect();
        let raw = device.raw();
        let set_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&layout_bindings);
        // SAFETY: plain object creation on a live device.
        let set_layout = unsafe { raw.create_descriptor_set_layout(&set_info, None) }
            .vk("vkCreateDescriptorSetLayout(unity)")?;
        let set_layouts = [set_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        // SAFETY: plain object creation.
        let layout = match unsafe { raw.create_pipeline_layout(&layout_info, None) }
            .vk("vkCreatePipelineLayout(unity)")
        {
            Ok(layout) => layout,
            Err(e) => {
                // SAFETY: never used.
                unsafe { raw.destroy_descriptor_set_layout(set_layout, None) };
                return Err(e);
            }
        };
        let pipeline = Self::pipeline(
            device,
            layout,
            &vertex,
            &fragment,
            &vertex_info,
            desc.state,
            color_format,
            depth_format,
        );
        let pipeline = match pipeline {
            Ok(pipeline) => pipeline,
            Err(e) => {
                // SAFETY: never used.
                unsafe {
                    raw.destroy_pipeline_layout(layout, None);
                    raw.destroy_descriptor_set_layout(set_layout, None);
                }
                return Err(e);
            }
        };

        let index_of = |kind: Kind| descriptors.iter().position(|d| d.kind == kind);
        let per_draw = index_of(Kind::PerDraw);
        let member = |name: &str| {
            per_draw
                .and_then(|i| descriptors[i].block.as_ref())
                .and_then(|b| b.members.iter().find(|m| m.name == name))
                .cloned()
        };
        let draw_members = DrawMembers {
            object_to_world: member("unity_ObjectToWorld"),
            world_to_object: member("unity_WorldToObject"),
            transform_params: member("unity_WorldTransformParams"),
        };
        Ok(Self {
            layout,
            set_layout,
            pipeline,
            dynamic: descriptors
                .iter()
                .enumerate()
                .filter(|(_, d)| d.is_dynamic())
                .map(|(i, _)| i)
                .collect(),
            textures: descriptors
                .iter()
                .enumerate()
                .filter(|(_, d)| matches!(d.kind, Kind::Texture(_)))
                .map(|(i, _)| i)
                .collect(),
            per_material: index_of(Kind::PerMaterial),
            per_draw,
            draw_members,
            transparent: desc.state.blend.is_some(),
            camera_textures: descriptors.iter().any(|d| {
                matches!(d.kind, Kind::Texture(_)) && super::camera_texture(&d.name).is_some()
            }),
            descriptors,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn pipeline(
        device: &Device,
        layout: vk::PipelineLayout,
        vertex: &[u32],
        fragment: &[u32],
        vertex_info: &Reflection,
        state: PassState,
        color_format: vk::Format,
        depth_format: vk::Format,
    ) -> Result<vk::Pipeline> {
        let raw = device.raw();
        let make_module = |words: &[u32]| {
            let info = vk::ShaderModuleCreateInfo::default().code(words);
            // SAFETY: the words are SPIR-V DXC produced and `rebind` only changed decorations.
            unsafe { raw.create_shader_module(&info, None) }.vk("vkCreateShaderModule(unity)")
        };
        let vertex_module = make_module(vertex)?;
        let fragment_module = match make_module(fragment) {
            Ok(module) => module,
            Err(e) => {
                // SAFETY: never used.
                unsafe { raw.destroy_shader_module(vertex_module, None) };
                return Err(e);
            }
        };
        // DXC names the entry point after the HLSL function, which is not `main`.
        let (vertex_entry, fragment_entry) = (entry_name(vertex), entry_name(fragment));
        let result = (|| {
            let (Some(vertex_entry), Some(fragment_entry)) = (&vertex_entry, &fragment_entry)
            else {
                return Err(invalid("a stage has no entry point"));
            };
            let stages = [
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(vertex_module)
                    .name(vertex_entry),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(fragment_module)
                    .name(fragment_entry),
            ];
            let bindings = [
                vk::VertexInputBindingDescription::default()
                    .binding(MESH_BINDING)
                    .stride(crate::geometry::VERTEX_STRIDE as u32)
                    .input_rate(vk::VertexInputRate::VERTEX),
                vk::VertexInputBindingDescription::default()
                    .binding(CONSTANT_BINDING)
                    .stride(0)
                    .input_rate(vk::VertexInputRate::VERTEX),
            ];
            let attributes: Vec<_> = vertex_info
                .inputs
                .iter()
                .filter(|input| !input.builtin)
                .map(|input| {
                    let (name, index) = split_semantic(input.semantic.as_deref().unwrap_or(""));
                    let (binding, format, offset) = match (name, index) {
                        ("POSITION", 0) => (MESH_BINDING, vk::Format::R32G32B32_SFLOAT, 0),
                        ("NORMAL", 0) => (MESH_BINDING, vk::Format::R32G32B32_SFLOAT, 12),
                        ("TEXCOORD", 0) => (MESH_BINDING, vk::Format::R32G32_SFLOAT, 24),
                        ("TANGENT", _) => (CONSTANT_BINDING, vk::Format::R32G32B32A32_SFLOAT, 0),
                        ("COLOR", _) => (CONSTANT_BINDING, vk::Format::R32G32B32A32_SFLOAT, 16),
                        _ => (CONSTANT_BINDING, vk::Format::R32G32B32A32_SFLOAT, 32),
                    };
                    vk::VertexInputAttributeDescription::default()
                        .location(input.location)
                        .binding(binding)
                        .format(format)
                        .offset(offset)
                })
                .collect();
            let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(&bindings)
                .vertex_attribute_descriptions(&attributes);
            let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
            let viewport = vk::PipelineViewportStateCreateInfo::default()
                .viewport_count(1)
                .scissor_count(1);
            // Both sides are drawn: the winding of imported meshes has not been checked against
            // Unity's culling convention yet.
            let raster = vk::PipelineRasterizationStateCreateInfo::default()
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(vk::CullModeFlags::NONE)
                .front_face(vk::FrontFace::CLOCKWISE)
                .line_width(1.0);
            let multisample = vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);
            let depth = vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(true)
                .depth_write_enable(state.z_write)
                .depth_compare_op(depth_op(state.z_test));
            let mask = vk::ColorComponentFlags::from_raw(u32::from(state.color_mask & 0xf));
            let attachment = match state.blend {
                None => vk::PipelineColorBlendAttachmentState::default().color_write_mask(mask),
                Some(Blend {
                    src_color,
                    dst_color,
                    src_alpha,
                    dst_alpha,
                }) => vk::PipelineColorBlendAttachmentState::default()
                    .blend_enable(true)
                    .src_color_blend_factor(blend_factor(src_color))
                    .dst_color_blend_factor(blend_factor(dst_color))
                    .color_blend_op(vk::BlendOp::ADD)
                    .src_alpha_blend_factor(blend_factor(src_alpha))
                    .dst_alpha_blend_factor(blend_factor(dst_alpha))
                    .alpha_blend_op(vk::BlendOp::ADD)
                    .color_write_mask(mask),
            };
            let attachments = [attachment];
            let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic =
                vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
            let color_formats = [color_format];
            let mut rendering = vk::PipelineRenderingCreateInfo::default()
                .color_attachment_formats(&color_formats)
                .depth_attachment_format(depth_format);
            let infos = [vk::GraphicsPipelineCreateInfo::default()
                .stages(&stages)
                .vertex_input_state(&vertex_input)
                .input_assembly_state(&input_assembly)
                .viewport_state(&viewport)
                .rasterization_state(&raster)
                .multisample_state(&multisample)
                .depth_stencil_state(&depth)
                .color_blend_state(&blend)
                .dynamic_state(&dynamic)
                .layout(layout)
                .push_next(&mut rendering)];
            // SAFETY: every structure referenced by `infos` outlives this call.
            let created =
                unsafe { raw.create_graphics_pipelines(vk::PipelineCache::null(), &infos, None) };
            match created {
                Ok(pipelines) => pipelines
                    .first()
                    .copied()
                    .ok_or_else(|| invalid("no pipeline was created")),
                Err((pipelines, result)) => {
                    for pipeline in pipelines.into_iter().filter(|p| *p != vk::Pipeline::null()) {
                        // SAFETY: created by the failed call, never used.
                        unsafe { raw.destroy_pipeline(pipeline, None) };
                    }
                    Err(VulkanError::Call {
                        call: "vkCreateGraphicsPipelines(unity)",
                        result,
                    })
                }
            }
        })();
        // SAFETY: pipelines keep what they need from the modules.
        unsafe {
            raw.destroy_shader_module(vertex_module, None);
            raw.destroy_shader_module(fragment_module, None);
        }
        result
    }

    pub(crate) fn destroy(&self, device: &Device) {
        // SAFETY: the owner guarantees the GPU no longer uses these objects.
        unsafe {
            device.raw().destroy_pipeline(self.pipeline, None);
            device.raw().destroy_pipeline_layout(self.layout, None);
            device
                .raw()
                .destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

/// The name of the first entry point of a module (`OpEntryPoint`: execution model, id, name).
fn entry_name(words: &[u32]) -> Option<std::ffi::CString> {
    let mut at = 5;
    while at < words.len() {
        let head = words[at];
        let (count, op) = ((head >> 16) as usize, head & 0xffff);
        if count == 0 || at + count > words.len() {
            return None;
        }
        if op == 15 && count >= 4 {
            let mut bytes = Vec::new();
            'words: for word in &words[at + 3..at + count] {
                for byte in word.to_le_bytes() {
                    if byte == 0 {
                        break 'words;
                    }
                    bytes.push(byte);
                }
            }
            return std::ffi::CString::new(bytes).ok();
        }
        at += count;
    }
    None
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn semantics_split_into_name_and_index() {
        assert_eq!(split_semantic("TEXCOORD3"), ("TEXCOORD", 3));
        assert_eq!(split_semantic("POSITION"), ("POSITION", 0));
        assert_eq!(split_semantic("COLOR0"), ("COLOR", 0));
        assert_eq!(split_semantic(""), ("", 0));
    }

    #[test]
    fn unity_depth_tests_are_flipped_for_reverse_z() {
        assert_eq!(
            depth_op(DepthTest::LessEqual),
            vk::CompareOp::GREATER_OR_EQUAL
        );
        assert_eq!(depth_op(DepthTest::Less), vk::CompareOp::GREATER);
        assert_eq!(depth_op(DepthTest::Greater), vk::CompareOp::LESS);
        assert_eq!(depth_op(DepthTest::Always), vk::CompareOp::ALWAYS);
    }

    #[test]
    fn the_entry_point_name_is_read_from_the_module() {
        // OpEntryPoint Vertex %1 "vert" (name = "vert\0" in one word), no interface.
        let words = [
            0x0723_0203,
            0x0001_0600,
            0,
            10,
            0,
            (4 << 16) | 15,
            0,
            1,
            u32::from_le_bytes(*b"vert"),
        ];
        // The name needs a terminating nul word.
        let mut with_nul = words.to_vec();
        with_nul[5] = (5 << 16) | 15;
        with_nul.push(0);
        assert_eq!(entry_name(&with_nul).unwrap().to_str().unwrap(), "vert");
        assert!(entry_name(&[0x0723_0203, 0, 0, 0, 0]).is_none());
    }
}
