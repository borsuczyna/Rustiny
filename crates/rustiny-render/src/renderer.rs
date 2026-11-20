//! [`SceneRenderer`]: owns the scene pass resources and records it into a command buffer.

use crate::buffer::HostBuffer;
use crate::geometry::{BOX_INDEX_COUNT, GpuInstance, unit_box};
use crate::material::{MaterialBlock, Materials};
use crate::mesh::{GpuMesh, GroupRange, MeshGroup, MeshSection, MeshStats, MeshVertex, Surface};
use crate::pipeline::{MATERIAL_OFFSET, PUSH_SIZE, Pipelines};
use crate::post::{BLOOM_PUSH, BloomPipelines, PostProcess};
use crate::present::Present;
use crate::target::{Target, TargetFormats, TargetLayouts};
use crate::texture::{RetiredTextures, Textures};
use crate::timing::GpuTimer;
use crate::unity::{
    CameraViews, Phase, SectionDraw, ShaderMaterialDesc, ShaderPassDesc, UnityFrame, UnityShaders,
};
use ash::vk;
use rustiny_vulkan::{Device, FRAMES_IN_FLIGHT, Result, VulkanError};
use std::collections::HashMap;
use std::sync::Arc;

/// Format of the offscreen colour image. UNORM: the shaders gamma-encode their output, so the
/// editor can show the image as it is.
pub const COLOR_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;
/// Format the scene is rendered in: linear light with values above 1, like URP's HDR camera
/// target. The final pass converts it into a [`COLOR_FORMAT`] image for display.
pub const HDR_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
/// Format of the depth image (reverse-Z, cleared to 0).
pub const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

/// Per-frame inputs of the scene pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameParams {
    /// Column-major view-projection (see [`crate::matrix`]).
    pub view_proj: [f32; 16],
    /// Camera position in world space (centres the ground grid).
    pub eye: [f32; 3],
    /// Scene object that is selected: its instances are drawn highlighted, even when hidden.
    pub selected: Option<u32>,
    /// Background colour written to the target (UNORM, not gamma-encoded again).
    pub clear: [f32; 4],
    /// Half extent of the ground grid quad around the camera, in world units.
    pub grid_extent: f32,
    /// Size of one grid cell in world units.
    pub grid_cell: f32,
}

/// Something retired but possibly still in use by frames already submitted.
#[derive(Debug)]
enum Retired {
    Buffer(#[allow(dead_code)] HostBuffer),
    Target(#[allow(dead_code)] Box<Target>),
    Mesh(#[allow(dead_code)] GpuMesh),
    Textures(#[allow(dead_code)] RetiredTextures),
    Materials(#[allow(dead_code)] MaterialBlock),
    Unity(#[allow(dead_code)] Box<UnityShaders>),
}

/// Radius of the skybox sphere as a fraction of the far distance the shaders are told about.
const SKY_RADIUS: f32 = 0.9;
/// Latitude rings and longitude segments of the skybox sphere.
const SKY_RINGS: u32 = 24;
const SKY_SEGMENTS: u32 = 48;

/// A unit sphere (positions are also the normals), for the skybox.
fn sky_sphere(rings: u32, segments: u32) -> (Vec<MeshVertex>, Vec<u32>) {
    let mut vertices = Vec::with_capacity(((rings + 1) * (segments + 1)) as usize);
    for ring in 0..=rings {
        let theta = std::f32::consts::PI * ring as f32 / rings as f32;
        for segment in 0..=segments {
            let phi = std::f32::consts::TAU * segment as f32 / segments as f32;
            let p = [
                theta.sin() * phi.cos(),
                theta.cos(),
                theta.sin() * phi.sin(),
            ];
            vertices.push(MeshVertex {
                position: p,
                normal: p,
                uv: [
                    segment as f32 / segments as f32,
                    1.0 - ring as f32 / rings as f32,
                ],
            });
        }
    }
    let mut indices = Vec::with_capacity((rings * segments * 6) as usize);
    let row = segments + 1;
    for ring in 0..rings {
        for segment in 0..segments {
            let a = ring * row + segment;
            let b = a + row;
            indices.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    (vertices, indices)
}

/// Texel bytes copied into images per recorded pass (the first texture always goes through).
const TEXTURE_UPLOAD_BUDGET: u64 = 8 << 20;

/// Renders the scene into an offscreen image.
///
/// Create with [`SceneRenderer::new`], give it objects with [`set_instances`](Self::set_instances)
/// and a size with [`resize`](Self::resize), then call [`record`](Self::record) once per frame
/// from inside a frame that is being recorded (outside any rendering scope). The colour image
/// ends the pass in `SHADER_READ_ONLY_OPTIMAL`, ready to be sampled later in the same submission.
#[derive(Debug)]
pub struct SceneRenderer {
    device: Arc<Device>,
    pipelines: Pipelines,
    present: Present,
    bloom: BloomPipelines,
    /// The camera's post-processing (from the scene's volumes).
    post: PostProcess,
    /// The target generation whose bloom images were given their first layout.
    bloom_ready: Option<u64>,
    textures: Textures,
    materials: Materials,
    box_vertices: HostBuffer,
    box_indices: HostBuffer,
    instances: Option<HostBuffer>,
    instance_count: u32,
    meshes: HashMap<u32, GpuMesh>,
    mesh_instances: Option<HostBuffer>,
    /// The mesh instances again on the CPU: sections drawn with Unity shaders read their matrices
    /// from here.
    mesh_instance_data: Vec<GpuInstance>,
    mesh_groups: Vec<GroupRange>,
    unity: Option<UnityShaders>,
    unity_frame: UnityFrame,
    /// The skybox sphere, drawn with the scene's skybox material.
    sky: Option<GpuMesh>,
    shader_globals: Arc<HashMap<String, Vec<f32>>>,
    stats: MeshStats,
    target: Option<Target>,
    retired: Vec<(u64, Retired)>,
    records: u64,
    /// Changes with every new target; the copies made for the camera textures belong to one.
    target_generation: u64,
    /// The generation whose copy images were moved to their first layout.
    copies_ready: Option<u64>,
    timer: Option<GpuTimer>,
    gpu_ms: Option<f32>,
}

impl SceneRenderer {
    /// Build the pipelines and the unit box.
    pub fn new(device: &Arc<Device>) -> Result<Self> {
        for format in [COLOR_FORMAT, HDR_FORMAT, DEPTH_FORMAT] {
            // SAFETY: query call on a live instance and a physical device of it.
            let properties = unsafe {
                device
                    .instance()
                    .raw()
                    .get_physical_device_format_properties(device.physical(), format)
            };
            let needed = if format == DEPTH_FORMAT {
                vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT
            } else {
                vk::FormatFeatureFlags::COLOR_ATTACHMENT | vk::FormatFeatureFlags::SAMPLED_IMAGE
            };
            if !properties.optimal_tiling_features.contains(needed) {
                return Err(VulkanError::NoSuitableDevice(format!(
                    "format {format:?} cannot be used for the scene target"
                )));
            }
        }
        let textures = Textures::new(device)?;
        let materials = Materials::new(device)?;
        let pipelines = Pipelines::new(
            device,
            textures.layout,
            materials.layout,
            HDR_FORMAT,
            DEPTH_FORMAT,
        )?;
        let present = Present::new(device, COLOR_FORMAT)?;
        let bloom = match BloomPipelines::new(device, HDR_FORMAT) {
            Ok(bloom) => bloom,
            Err(e) => {
                present.destroy(device);
                pipelines.destroy(device);
                return Err(e);
            }
        };
        let timer = GpuTimer::new(device).unwrap_or_else(|e| {
            tracing::warn!("GPU timing unavailable: {e}");
            None
        });
        let (vertices, indices) = unit_box();
        let buffers = HostBuffer::with_data(
            device,
            &vertices,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            "scene-box-vertices",
        )
        .and_then(|v| {
            HostBuffer::with_data(
                device,
                &indices,
                vk::BufferUsageFlags::INDEX_BUFFER,
                "scene-box-indices",
            )
            .map(|i| (v, i))
        });
        match buffers {
            Ok((box_vertices, box_indices)) => Ok(Self {
                device: device.clone(),
                pipelines,
                present,
                textures,
                materials,
                box_vertices,
                box_indices,
                instances: None,
                instance_count: 0,
                meshes: HashMap::new(),
                mesh_instances: None,
                mesh_instance_data: Vec::new(),
                mesh_groups: Vec::new(),
                unity: None,
                unity_frame: UnityFrame::default(),
                sky: None,
                shader_globals: Arc::new(HashMap::new()),
                stats: MeshStats::default(),
                target: None,
                retired: Vec::new(),
                records: 0,
                target_generation: 0,
                copies_ready: None,
                bloom,
                post: PostProcess::default(),
                bloom_ready: None,
                timer,
                gpu_ms: None,
            }),
            Err(e) => {
                pipelines.destroy(device);
                present.destroy(device);
                bloom.destroy(device);
                Err(e)
            }
        }
    }

    /// Replace the proxy boxes. The previous instance buffer is kept alive until frames that
    /// may still read it have finished; nothing here waits for the GPU.
    pub fn set_instances(&mut self, instances: &[GpuInstance]) -> Result<()> {
        let buffer = self.instance_buffer(instances.iter(), instances.len(), "scene-instances")?;
        if let Some(old) = self.instances.take() {
            self.retired.push((self.records, Retired::Buffer(old)));
        }
        self.instances = buffer;
        self.instance_count = u32::try_from(instances.len()).unwrap_or(u32::MAX);
        Ok(())
    }

    fn instance_buffer<'a>(
        &self,
        instances: impl Iterator<Item = &'a GpuInstance>,
        count: usize,
        name: &'static str,
    ) -> Result<Option<HostBuffer>> {
        if count == 0 {
            return Ok(None);
        }
        let mut bytes = vec![0u8; count * GpuInstance::SIZE];
        let (slots, _) = bytes.as_chunks_mut::<{ GpuInstance::SIZE }>();
        for (instance, out) in instances.zip(slots) {
            instance.write_to(out);
        }
        HostBuffer::with_data(
            &self.device,
            &bytes,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            name,
        )
        .map(Some)
    }

    /// Replace the mesh instances. Groups whose mesh has not been uploaded yet are skipped until
    /// it is, so instances can be set first and meshes streamed in afterwards. Uploaded meshes
    /// are kept.
    pub fn set_mesh_instances(&mut self, groups: &[MeshGroup]) -> Result<()> {
        let total: usize = groups.iter().map(|g| g.instances.len()).sum();
        let buffer = self.instance_buffer(
            groups.iter().flat_map(|g| g.instances.iter()),
            total,
            "scene-mesh-instances",
        )?;
        let mut ranges = Vec::with_capacity(groups.len());
        let mut first = 0u32;
        for group in groups {
            let count = u32::try_from(group.instances.len()).unwrap_or(u32::MAX);
            if count > 0 {
                ranges.push(GroupRange {
                    key: group.key,
                    first,
                    count,
                });
            }
            first = first.saturating_add(count);
        }
        if let Some(old) = self.mesh_instances.take() {
            self.retired.push((self.records, Retired::Buffer(old)));
        }
        self.mesh_instances = buffer;
        self.mesh_groups = ranges;
        self.mesh_instance_data.clear();
        self.mesh_instance_data
            .extend(groups.iter().flat_map(|g| g.instances.iter().copied()));
        if let Some(unity) = &mut self.unity {
            // The per-object shader data was written for the old instances.
            unity.forget_instances();
        }
        self.stats.instances = total;
        Ok(())
    }

    /// Add a compiled pass of a Unity shader and return its index for
    /// [`add_shader_material`](Self::add_shader_material). The first call creates the shared
    /// resources (per-frame constant memory, stand-in textures, samplers); instances set before
    /// it are kept and drawn with shaders as soon as their sections name a shader material.
    pub fn add_shader_pass(&mut self, desc: &ShaderPassDesc) -> Result<u32> {
        if self.unity.is_none() {
            self.unity = Some(UnityShaders::new(&self.device, HDR_FORMAT, DEPTH_FORMAT)?);
        }
        match &mut self.unity {
            Some(unity) => unity.add_pass(desc),
            None => Err(VulkanError::Allocator("no shader state".into())),
        }
    }

    /// Add a material for a shader pass and return the id sections name in `Surface::shader`.
    pub fn add_shader_material(&mut self, desc: &ShaderMaterialDesc) -> Result<u32> {
        let unity = self.unity.as_mut().ok_or_else(|| {
            VulkanError::InvalidInput("add a shader pass before shader materials".into())
        })?;
        let id = unity.add_material(desc)?;
        self.stats.shader_materials = unity.material_count();
        Ok(id)
    }

    /// Drop every shader pass and material (they stay alive until the frames using them are
    /// done). Sections that named a shader material are drawn with the native surface look until
    /// their meshes are added again.
    pub fn clear_shaders(&mut self) {
        if let Some(old) = self.unity.take() {
            self.retired
                .push((self.records, Retired::Unity(Box::new(old))));
        }
        self.mesh_instance_data.clear();
        self.stats.shader_materials = 0;
    }

    /// Global shader values (`Shader.SetGlobalFloat` and friends in a Unity project's scripts),
    /// by variable name. A material's own value wins, then the engine's camera, time and light
    /// values, then these.
    pub fn set_shader_globals(&mut self, globals: HashMap<String, Vec<f32>>) {
        self.shader_globals = Arc::new(globals);
    }

    /// Draw the background with the skybox shader material `material` (an id from
    /// [`add_shader_material`](Self::add_shader_material)), or with the clear colour (`None`).
    /// Like Unity, the skybox is a sphere around the camera drawn after the opaque objects.
    pub fn set_skybox(&mut self, material: Option<u32>) -> Result<()> {
        if let Some(old) = self.sky.take() {
            self.retired.push((self.records, Retired::Mesh(old)));
        }
        let Some(material) = material else {
            return Ok(());
        };
        if !self
            .unity
            .as_ref()
            .is_some_and(|u| u.has_material(material))
        {
            return Err(VulkanError::InvalidInput(format!(
                "skybox material {material} has not been added"
            )));
        }
        let (vertices, indices) = sky_sphere(SKY_RINGS, SKY_SEGMENTS);
        let sections = [MeshSection {
            first_index: 0,
            index_count: indices.len() as u32,
            surface: Surface {
                shader: Some(material),
                ..Surface::default()
            },
        }];
        if !self.materials.has_room(1) {
            return Err(VulkanError::InvalidInput(
                "too many materials in the scene".into(),
            ));
        }
        let mesh = GpuMesh::new(
            &self.device,
            &vertices,
            &indices,
            &sections,
            self.materials.len(),
        )?;
        self.materials.append(sections.iter().map(|s| &s.surface))?;
        self.sky = Some(mesh);
        Ok(())
    }

    /// The post-processing the final pass applies (URP's volume stack, resolved).
    pub fn set_post_process(&mut self, post: PostProcess) {
        self.post = post;
    }

    /// The camera, clock and light the Unity shaders see in the next recorded passes.
    pub fn set_unity_frame(&mut self, frame: UnityFrame) {
        self.unity_frame = frame;
    }

    /// Upload a mesh under `key`, replacing any mesh with that key. Rejects inconsistent data
    /// (indices past the last vertex, sections past the index list) without creating anything.
    /// Every section takes one material record; a replaced mesh's records stay in use until
    /// [`clear_meshes`](Self::clear_meshes).
    pub fn add_mesh(
        &mut self,
        key: u32,
        vertices: &[MeshVertex],
        indices: &[u32],
        sections: &[MeshSection],
    ) -> Result<()> {
        if !self.materials.has_room(sections.len()) {
            return Err(VulkanError::InvalidInput(
                "too many materials in the scene".into(),
            ));
        }
        for section in sections {
            if let Some(id) = section.surface.shader
                && !self.unity.as_ref().is_some_and(|u| u.has_material(id))
            {
                return Err(VulkanError::InvalidInput(format!(
                    "section uses shader material {id}, which has not been added"
                )));
            }
        }
        let mesh = GpuMesh::new(
            &self.device,
            vertices,
            indices,
            sections,
            self.materials.len(),
        )?;
        self.materials.append(sections.iter().map(|s| &s.surface))?;
        if let Some(old) = self.meshes.insert(key, mesh) {
            self.retired.push((self.records, Retired::Mesh(old)));
        }
        self.stats.meshes = self.meshes.len();
        self.stats.materials = self.materials.len() as usize;
        Ok(())
    }

    /// Drop every uploaded mesh and material record (they stay alive until the frames using them
    /// are done).
    pub fn clear_meshes(&mut self) {
        let stamp = self.records;
        self.retired.extend(
            self.meshes
                .drain()
                .map(|(_, mesh)| (stamp, Retired::Mesh(mesh))),
        );
        match self.materials.reset() {
            Ok(old) => self.retired.push((stamp, Retired::Materials(old))),
            Err(e) => tracing::warn!("cannot reset the material records: {e}"),
        }
        self.stats.meshes = 0;
        self.stats.materials = self.materials.len() as usize;
    }

    /// Whether a mesh is uploaded under `key`.
    pub fn has_mesh(&self, key: u32) -> bool {
        self.meshes.contains_key(&key)
    }

    /// Add a texture under `key` (replacing one with that key). `mips` is the whole chain of RGBA8
    /// images, largest first (see [`mip_chain`](crate::mip_chain)); every level must have exactly
    /// the size its dimensions give. Row 0 is the bottom of the picture (V up, as in Unity).
    /// `srgb` is `true` for colours (albedo), `false` for data (normals, metallic-smoothness).
    /// The texels are copied to the GPU in the next recorded passes, a few megabytes each; until
    /// then sections that use the texture draw as if they had none.
    pub fn add_texture(
        &mut self,
        key: u32,
        width: u32,
        height: u32,
        mips: &[Vec<u8>],
        srgb: bool,
    ) -> Result<()> {
        if let Some(old) = self.textures.add(key, width, height, mips, srgb)? {
            self.retired.push((self.records, Retired::Textures(old)));
        }
        self.stats.textures = self.textures.len();
        self.stats.texture_bytes = self.textures.bytes();
        Ok(())
    }

    /// Drop every texture (they stay alive until the frames using them are done).
    pub fn clear_textures(&mut self) {
        let old = self.textures.clear();
        self.retired.push((self.records, Retired::Textures(old)));
        self.stats.textures = 0;
        self.stats.texture_bytes = 0;
    }

    /// Whether a texture is stored under `key`.
    pub fn has_texture(&self, key: u32) -> bool {
        self.textures.contains(key)
    }

    /// Textures added whose copy has not been recorded yet; passes keep being needed until it is 0.
    pub fn textures_waiting(&self) -> usize {
        self.textures.waiting()
    }

    /// Whether the texture under `key` has been copied to the GPU by a recorded pass.
    pub fn texture_uploaded(&self, key: u32) -> bool {
        self.textures.is_ready(key)
    }

    /// Mesh counts and what the last recorded pass submitted.
    pub fn mesh_stats(&self) -> MeshStats {
        self.stats
    }

    /// Number of instances currently drawn.
    pub fn instance_count(&self) -> u32 {
        self.instance_count
    }

    /// Make the target `width` x `height` pixels (at least 1x1). Returns `true` when a new target
    /// was created, in which case [`color_view`](Self::color_view) is a different image view and
    /// anything that sampled the old one (an ImGui texture, say) must be pointed at the new one.
    pub fn resize(&mut self, width: u32, height: u32) -> Result<bool> {
        let extent = vk::Extent2D {
            width: width.max(1),
            height: height.max(1),
        };
        if self.target.as_ref().is_some_and(|t| t.extent() == extent) {
            return Ok(false);
        }
        let target = Target::new(
            &self.device,
            extent,
            TargetFormats {
                hdr: HDR_FORMAT,
                depth: DEPTH_FORMAT,
                display: COLOR_FORMAT,
            },
            TargetLayouts {
                present: self.present.set_layout,
                bloom: self.bloom.set_layout,
                sampler: self.present.sampler,
            },
        )?;
        self.target_generation += 1;
        if let Some(old) = self.target.replace(target) {
            self.retired
                .push((self.records, Retired::Target(Box::new(old))));
        }
        Ok(true)
    }

    /// GPU time of a recent scene pass in milliseconds, measured with timestamp queries (a frame
    /// or two old). `None` until the first result arrives or when the queue cannot time.
    pub fn gpu_time_ms(&self) -> Option<f32> {
        self.gpu_ms
    }

    /// Size of the target, if there is one.
    pub fn extent(&self) -> Option<vk::Extent2D> {
        self.target.as_ref().map(Target::extent)
    }

    /// The colour image view to sample, once a target exists.
    pub fn color_view(&self) -> Option<vk::ImageView> {
        self.target.as_ref().map(Target::display_view)
    }

    /// The colour image (for copies in tests and screenshots), once a target exists.
    pub fn color_image(&self) -> Option<vk::Image> {
        self.target.as_ref().map(Target::display_image)
    }

    /// Record the scene pass into `cmd`, which must be recording and outside a rendering scope.
    /// Returns `false` (recording nothing) when there is no target yet.
    pub fn record(&mut self, cmd: vk::CommandBuffer, params: &FrameParams) -> bool {
        self.records += 1;
        let records = self.records;
        self.retired
            .retain(|(stamp, _)| records <= stamp + FRAMES_IN_FLIGHT as u64 + 1);
        let Some(target) = &self.target else {
            return false;
        };
        let raw = self.device.raw();
        let extent = target.extent();
        // Texture copies come first, outside the rendering scope; their staging buffers live as
        // long as frames that may still read them.
        for staging in self.textures.flush(cmd, TEXTURE_UPLOAD_BUDGET) {
            self.retired.push((records, Retired::Buffer(staging)));
        }
        let default_set = match self.textures.set_for([None; crate::texture::SLOTS]) {
            Ok(set) => set,
            Err(e) => {
                tracing::warn!("cannot make the default texture set: {e}");
                return false;
            }
        };
        let mut set_failed = false;
        let has_unity = self.unity.is_some();
        // Passes that blend, or that read the camera's depth or colour, are drawn after the opaque
        // ones, with a copy of both in between.
        let split = self
            .unity
            .as_ref()
            .is_some_and(UnityShaders::needs_camera_copy);
        if let Some(unity) = &mut self.unity {
            unity.begin_frame(
                cmd,
                records,
                &self.unity_frame,
                &params.view_proj,
                &self.textures,
                Some(CameraViews {
                    depth: target.depth_copy_view(),
                    color: target.color_copy_view(),
                    generation: self.target_generation,
                }),
                &self.shader_globals,
            );
        }
        if let Some(timer) = &mut self.timer {
            if let Some(ms) = timer.collect(records) {
                self.gpu_ms = Some(ms);
            }
            timer.begin(cmd, records);
        }

        // Colour and depth start the pass with undefined contents (everything is cleared) and
        // wait for earlier readers of the previous frame's colour.
        let color_to_attachment = image_barrier(
            target.color_image(),
            vk::ImageAspectFlags::COLOR,
            (
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::AccessFlags2::NONE,
                vk::ImageLayout::UNDEFINED,
            ),
            (
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            ),
        );
        let depth_to_attachment = image_barrier(
            target.depth_image(),
            vk::ImageAspectFlags::DEPTH,
            (
                vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                vk::ImageLayout::UNDEFINED,
            ),
            (
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
                    | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
                vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            ),
        );
        let mut barriers = vec![color_to_attachment, depth_to_attachment];
        if split && self.copies_ready != Some(self.target_generation) {
            // Shaders may read the copies before the first copy is made: give them a layout.
            let ready = (
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
            let undefined = (
                vk::PipelineStageFlags2::NONE,
                vk::AccessFlags2::NONE,
                vk::ImageLayout::UNDEFINED,
            );
            barriers.push(image_barrier(
                target.color_copy_image(),
                vk::ImageAspectFlags::COLOR,
                undefined,
                ready,
            ));
            barriers.push(image_barrier(
                target.depth_copy_image(),
                vk::ImageAspectFlags::DEPTH,
                undefined,
                ready,
            ));
            self.copies_ready = Some(self.target_generation);
        }
        let dependency = vk::DependencyInfo::default().image_memory_barriers(&barriers);
        // SAFETY: `cmd` is recording (caller's contract) and the images belong to this device.
        unsafe { raw.cmd_pipeline_barrier2(cmd, &dependency) };

        let color_attachments = [vk::RenderingAttachmentInfo::default()
            .image_view(target.color_view())
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    // The clear colour is given as it should look; the target is linear.
                    float32: srgb_to_linear(params.clear),
                },
            })];
        let depth_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(target.depth_view())
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(if split {
                vk::AttachmentStoreOp::STORE
            } else {
                vk::AttachmentStoreOp::DONT_CARE
            })
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 0.0,
                    stencil: 0,
                },
            });
        let rendering = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent,
            })
            .layer_count(1)
            .color_attachments(&color_attachments)
            .depth_attachment(&depth_attachment);
        // The second part of a split frame keeps what the first one drew.
        let color_load = [color_attachments[0]
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE)];
        let depth_load = depth_attachment
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::DONT_CARE);
        let rendering_again = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent,
            })
            .layer_count(1)
            .color_attachments(&color_load)
            .depth_attachment(&depth_load);
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent,
        };
        let push = push_constants(params);
        let (mut draw_calls, mut triangles) = (0u32, 0u64);
        let (mut shader_draws, mut shader_skipped) = (0u32, 0u32);

        // SAFETY: `cmd` is recording outside a rendering scope; every handle used below is owned
        // by `self` and stays alive until the frames using it have finished (retirement queue).
        unsafe {
            raw.cmd_begin_rendering(cmd, &rendering);
            raw.cmd_set_viewport(cmd, 0, &[viewport]);
            raw.cmd_set_scissor(cmd, 0, &[scissor]);
            raw.cmd_push_constants(
                cmd,
                self.pipelines.layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                &push,
            );
            raw.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipelines.meshes);
            // The boxes use the default textures and record 0.
            let plain = default_set;
            raw.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipelines.layout,
                0,
                &[plain, self.materials.set()],
                &[],
            );
            let mut bound = plain;
            if let Some(instances) = &self.instances {
                raw.cmd_bind_vertex_buffers(
                    cmd,
                    0,
                    &[self.box_vertices.handle(), instances.handle()],
                    &[0, 0],
                );
                raw.cmd_bind_index_buffer(cmd, self.box_indices.handle(), 0, vk::IndexType::UINT32);
                raw.cmd_draw_indexed(cmd, BOX_INDEX_COUNT, self.instance_count, 0, 0, 0);
                draw_calls += 1;
                triangles += u64::from(BOX_INDEX_COUNT / 3) * u64::from(self.instance_count);
            }
            if let Some(instances) = &self.mesh_instances {
                for group in &self.mesh_groups {
                    let Some(mesh) = self.meshes.get(&group.key) else {
                        continue;
                    };
                    raw.cmd_bind_vertex_buffers(
                        cmd,
                        0,
                        &[mesh.vertices.handle(), instances.handle()],
                        &[0, 0],
                    );
                    raw.cmd_bind_index_buffer(cmd, mesh.indices.handle(), 0, vk::IndexType::UINT32);
                    for section in &mesh.sections {
                        if has_unity && section.shader.is_some() {
                            continue;
                        }
                        let set = match self.textures.set_for(section.textures) {
                            Ok(set) => set,
                            Err(e) => {
                                if !set_failed {
                                    tracing::warn!("cannot make a texture set: {e}");
                                    set_failed = true;
                                }
                                continue;
                            }
                        };
                        if set != bound {
                            raw.cmd_bind_descriptor_sets(
                                cmd,
                                vk::PipelineBindPoint::GRAPHICS,
                                self.pipelines.layout,
                                0,
                                &[set],
                                &[],
                            );
                            bound = set;
                        }
                        raw.cmd_push_constants(
                            cmd,
                            self.pipelines.layout,
                            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                            MATERIAL_OFFSET,
                            &material_bytes(section.material),
                        );
                        raw.cmd_draw_indexed(
                            cmd,
                            section.index_count,
                            group.count,
                            section.first_index,
                            0,
                            group.first,
                        );
                        draw_calls += 1;
                        triangles += u64::from(section.index_count / 3) * u64::from(group.count);
                    }
                }
            }
            if let Some(unity) = &mut self.unity {
                let phase = Phase::Opaque;
                let (drawn, left_out, tris) = draw_unity_groups(
                    unity,
                    cmd,
                    &self.textures,
                    &self.meshes,
                    &self.mesh_groups,
                    &self.mesh_instance_data,
                    params.selected,
                    phase,
                );
                shader_draws += drawn;
                shader_skipped += left_out;
                triangles += tris;
                // URP's order: opaque objects, the skybox, then (after the camera textures are
                // copied) the transparent ones.
                if let Some(sky) = &self.sky {
                    let radius = self.unity_frame.far * SKY_RADIUS;
                    let mut model = [0.0f32; 16];
                    model[0] = radius;
                    model[5] = radius;
                    model[10] = radius;
                    model[12] = params.eye[0];
                    model[13] = params.eye[1];
                    model[14] = params.eye[2];
                    model[15] = 1.0;
                    let instance = [GpuInstance::new(model, [1.0; 3], true)];
                    for section in &sky.sections {
                        let (drawn, _) = unity.draw_section(
                            cmd,
                            &self.textures,
                            &SectionDraw {
                                mesh: sky,
                                section,
                                instances: &instance,
                                first_instance: None,
                                selected: None,
                                phase: Phase::Opaque,
                            },
                        );
                        shader_draws += drawn;
                    }
                }
                if split {
                    raw.cmd_end_rendering(cmd);
                    copy_for_camera_textures(raw, cmd, target, extent);
                    raw.cmd_begin_rendering(cmd, &rendering_again);
                }
                {
                    let (drawn, left_out, tris) = draw_unity_groups(
                        unity,
                        cmd,
                        &self.textures,
                        &self.meshes,
                        &self.mesh_groups,
                        &self.mesh_instance_data,
                        params.selected,
                        Phase::Transparent,
                    );
                    shader_draws += drawn;
                    shader_skipped += left_out;
                    triangles += tris;
                }
                if unity.used() {
                    // Another pipeline layout was in use: bring the scene's bindings back.
                    raw.cmd_push_constants(
                        cmd,
                        self.pipelines.layout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        &push,
                    );
                    raw.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipelines.layout,
                        0,
                        &[plain, self.materials.set()],
                        &[],
                    );
                }
            }
            raw.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipelines.grid);
            raw.cmd_draw(cmd, 6, 1, 0, 0);
            raw.cmd_end_rendering(cmd);
        }
        draw_calls += shader_draws;
        self.stats.draw_calls = draw_calls;
        self.stats.shader_draws = shader_draws;
        self.stats.shader_skipped = shader_skipped;
        self.stats.triangles_drawn = triangles;

        // Colour is sampled by the UI pass that follows in the same submission (the tests copy it
        // out with a further barrier of their own).
        self.record_present(cmd, extent);
        if let Some(timer) = &mut self.timer {
            timer.end(cmd, records);
        }
        true
    }
}

impl SceneRenderer {
    /// The final pass: the HDR colour is read and written to the display image as sRGB. The
    /// display image ends in `SHADER_READ_ONLY_OPTIMAL` for the UI pass that samples it.
    fn record_present(&mut self, cmd: vk::CommandBuffer, extent: vk::Extent2D) {
        let Some(target) = self.target.as_ref() else {
            return;
        };
        use vk::AccessFlags2 as A;
        use vk::ImageLayout as L;
        use vk::PipelineStageFlags2 as S;
        let raw = self.device.raw();
        let before = [
            image_barrier(
                target.color_image(),
                vk::ImageAspectFlags::COLOR,
                (
                    S::COLOR_ATTACHMENT_OUTPUT,
                    A::COLOR_ATTACHMENT_WRITE,
                    L::COLOR_ATTACHMENT_OPTIMAL,
                ),
                (
                    S::FRAGMENT_SHADER,
                    A::SHADER_SAMPLED_READ,
                    L::SHADER_READ_ONLY_OPTIMAL,
                ),
            ),
            // The previous frame's display image may still be read by its UI pass.
            image_barrier(
                target.display_image(),
                vk::ImageAspectFlags::COLOR,
                (S::FRAGMENT_SHADER, A::NONE, L::UNDEFINED),
                (
                    S::COLOR_ATTACHMENT_OUTPUT,
                    A::COLOR_ATTACHMENT_WRITE,
                    L::COLOR_ATTACHMENT_OPTIMAL,
                ),
            ),
        ];
        // SAFETY: `cmd` is recording outside a rendering scope; the images belong to the target.
        unsafe {
            raw.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&before),
            );
        }
        if self.bloom_ready != Some(self.target_generation) {
            // Every bloom image is read by some set; give them all a readable layout once.
            let mut first: Vec<vk::ImageMemoryBarrier2<'static>> = Vec::new();
            for level in 0..target.bloom_levels() {
                for (image, _) in [target.bloom_down(level), target.bloom_up(level)] {
                    first.push(image_barrier(
                        image,
                        vk::ImageAspectFlags::COLOR,
                        (S::NONE, A::NONE, L::UNDEFINED),
                        (
                            S::FRAGMENT_SHADER,
                            A::SHADER_SAMPLED_READ,
                            L::SHADER_READ_ONLY_OPTIMAL,
                        ),
                    ));
                }
            }
            // SAFETY: as above.
            unsafe {
                raw.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().image_memory_barriers(&first),
                );
            }
            self.bloom_ready = Some(self.target_generation);
        }
        let levels = match self.post.bloom.filter(|b| b.intensity > 0.0) {
            Some(bloom) => {
                let levels =
                    bloom.mip_count(extent.width, extent.height, target.bloom_levels() as u32)
                        as usize;
                self.record_bloom(cmd, target, &bloom, levels, extent);
                levels
            }
            None => 1,
        };
        let present_set = if levels > 1 {
            target.sets().present_up
        } else {
            target.sets().present_down
        };
        let bloom_extent = target.bloom_extent(0);
        let push = self.post.present_push(
            extent.width,
            extent.height,
            [bloom_extent.width, bloom_extent.height],
        );
        let attachments = [vk::RenderingAttachmentInfo::default()
            .image_view(target.display_view())
            .image_layout(L::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)];
        let rendering = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent,
            })
            .layer_count(1)
            .color_attachments(&attachments);
        let after = [image_barrier(
            target.display_image(),
            vk::ImageAspectFlags::COLOR,
            (
                S::COLOR_ATTACHMENT_OUTPUT,
                A::COLOR_ATTACHMENT_WRITE,
                L::COLOR_ATTACHMENT_OPTIMAL,
            ),
            (
                S::FRAGMENT_SHADER,
                A::SHADER_SAMPLED_READ,
                L::SHADER_READ_ONLY_OPTIMAL,
            ),
        )];
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent,
        };
        // SAFETY: `cmd` is recording outside a rendering scope (the caller ended the scene's);
        // the pipeline, the set and the images are owned by `self` and the target, which outlive
        // the frame.
        unsafe {
            raw.cmd_begin_rendering(cmd, &rendering);
            raw.cmd_set_viewport(cmd, 0, &[viewport]);
            raw.cmd_set_scissor(cmd, 0, &[scissor]);
            raw.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.present.pipeline);
            raw.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.present.layout,
                0,
                &[present_set],
                &[],
            );
            raw.cmd_push_constants(
                cmd,
                self.present.layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                &push,
            );
            raw.cmd_draw(cmd, 3, 1, 0, 0);
            raw.cmd_end_rendering(cmd);
            raw.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&after),
            );
        }
    }
    /// URP's bloom: the scene colour prefiltered into level 0, blurred down level by level, then
    /// upsampled back to level 0 (`levels` levels).
    fn record_bloom(
        &self,
        cmd: vk::CommandBuffer,
        target: &Target,
        bloom: &crate::post::Bloom,
        levels: usize,
        extent: vk::Extent2D,
    ) {
        let params = bloom.params();
        let flags = [f32::from(u8::from(bloom.high_quality)), 0.0, 0.0, 0.0];
        let size = |e: vk::Extent2D| {
            let (w, h) = (e.width.max(1) as f32, e.height.max(1) as f32);
            [1.0 / w, 1.0 / h, w, h]
        };
        let sets = target.sets();
        let pipes = &self.bloom;
        let pass = |pipeline,
                    set,
                    (image, view): (vk::Image, vk::ImageView),
                    out: vk::Extent2D,
                    source: vk::Extent2D,
                    low: vk::Extent2D| {
            let mut push = [0u8; BLOOM_PUSH as usize];
            let values = size(source)
                .into_iter()
                .chain(size(low))
                .chain(params)
                .chain(flags);
            for (chunk, value) in push.as_chunks_mut::<4>().0.iter_mut().zip(values) {
                *chunk = value.to_ne_bytes();
            }
            self.fullscreen_pass(cmd, pipeline, pipes.layout, set, image, view, out, &push);
        };
        pass(
            pipes.prefilter,
            sets.prefilter,
            target.bloom_down(0),
            target.bloom_extent(0),
            extent,
            extent,
        );
        for level in 1..levels {
            let here = target.bloom_extent(level);
            let above = target.bloom_extent(level - 1);
            pass(
                pipes.blur_h,
                sets.blur_h[level],
                target.bloom_up(level),
                here,
                above,
                above,
            );
            pass(
                pipes.blur_v,
                sets.blur_v[level],
                target.bloom_down(level),
                here,
                here,
                here,
            );
        }
        for level in (0..levels.saturating_sub(1)).rev() {
            let set = if level + 2 == levels {
                sets.upsample_from_down[level]
            } else {
                sets.upsample_from_up[level]
            };
            let here = target.bloom_extent(level);
            let below = target.bloom_extent(level + 1);
            pass(
                pipes.upsample,
                set,
                target.bloom_up(level),
                here,
                here,
                below,
            );
        }
    }

    /// One fullscreen draw into `image` (left readable by shaders afterwards).
    #[allow(clippy::too_many_arguments)] // A pass's inputs; called from one place.
    fn fullscreen_pass(
        &self,
        cmd: vk::CommandBuffer,
        pipeline: vk::Pipeline,
        layout: vk::PipelineLayout,
        set: vk::DescriptorSet,
        image: vk::Image,
        view: vk::ImageView,
        extent: vk::Extent2D,
        push: &[u8],
    ) {
        use vk::AccessFlags2 as A;
        use vk::ImageLayout as L;
        use vk::PipelineStageFlags2 as S;
        let raw = self.device.raw();
        let to_write = [image_barrier(
            image,
            vk::ImageAspectFlags::COLOR,
            (S::FRAGMENT_SHADER, A::SHADER_SAMPLED_READ, L::UNDEFINED),
            (
                S::COLOR_ATTACHMENT_OUTPUT,
                A::COLOR_ATTACHMENT_WRITE,
                L::COLOR_ATTACHMENT_OPTIMAL,
            ),
        )];
        let to_read = [image_barrier(
            image,
            vk::ImageAspectFlags::COLOR,
            (
                S::COLOR_ATTACHMENT_OUTPUT,
                A::COLOR_ATTACHMENT_WRITE,
                L::COLOR_ATTACHMENT_OPTIMAL,
            ),
            (
                S::FRAGMENT_SHADER,
                A::SHADER_SAMPLED_READ,
                L::SHADER_READ_ONLY_OPTIMAL,
            ),
        )];
        let attachments = [vk::RenderingAttachmentInfo::default()
            .image_view(view)
            .image_layout(L::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)];
        let area = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent,
        };
        let rendering = vk::RenderingInfo::default()
            .render_area(area)
            .layer_count(1)
            .color_attachments(&attachments);
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        // SAFETY: `cmd` is recording outside a rendering scope; the pipeline, layout, set and
        // image are owned by the renderer and the target, which outlive the frame.
        unsafe {
            raw.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&to_write),
            );
            raw.cmd_begin_rendering(cmd, &rendering);
            raw.cmd_set_viewport(cmd, 0, &[viewport]);
            raw.cmd_set_scissor(cmd, 0, &[area]);
            raw.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
            raw.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                layout,
                0,
                &[set],
                &[],
            );
            raw.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::FRAGMENT, 0, push);
            raw.cmd_draw(cmd, 3, 1, 0, 0);
            raw.cmd_end_rendering(cmd);
            raw.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&to_read),
            );
        }
    }
}

/// The clear colour as linear light (the RGB channels are sRGB-decoded, alpha is kept).
fn srgb_to_linear(c: [f32; 4]) -> [f32; 4] {
    let decode = |v: f32| {
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    };
    [decode(c[0]), decode(c[1]), decode(c[2]), c[3]]
}

/// Record the Unity-shader draws of `phase` for every mesh group. Returns the draw calls made,
/// the instances left out for lack of memory and the triangles drawn.
#[allow(clippy::too_many_arguments)]
fn draw_unity_groups(
    unity: &mut UnityShaders,
    cmd: vk::CommandBuffer,
    textures: &Textures,
    meshes: &HashMap<u32, GpuMesh>,
    groups: &[GroupRange],
    instance_data: &[GpuInstance],
    selected: Option<u32>,
    phase: Phase,
) -> (u32, u32, u64) {
    let (mut draws, mut skipped, mut triangles) = (0u32, 0u32, 0u64);
    for group in groups {
        let Some(mesh) = meshes.get(&group.key) else {
            continue;
        };
        let first = group.first as usize;
        let Some(instances) = instance_data.get(first..first + group.count as usize) else {
            continue;
        };
        for section in mesh.sections.iter().filter(|s| s.shader.is_some()) {
            let (drawn, left_out) = unity.draw_section(
                cmd,
                textures,
                &SectionDraw {
                    mesh,
                    section,
                    instances,
                    first_instance: Some(group.first),
                    selected,
                    phase,
                },
            );
            draws += drawn;
            skipped += left_out;
            triangles += u64::from(section.index_count / 3) * u64::from(drawn);
        }
    }
    (draws, skipped, triangles)
}

/// Copy the colour and depth drawn so far into the images shaders read as `_CameraOpaqueTexture`
/// and `_CameraDepthTexture`, and put the target back into its attachment layouts. `cmd` must be
/// recording outside a rendering scope.
fn copy_for_camera_textures(
    raw: &ash::Device,
    cmd: vk::CommandBuffer,
    target: &Target,
    extent: vk::Extent2D,
) {
    use vk::AccessFlags2 as A;
    use vk::ImageAspectFlags as Aspect;
    use vk::ImageLayout as L;
    use vk::PipelineStageFlags2 as S;
    let sampled = (
        S::FRAGMENT_SHADER,
        A::SHADER_SAMPLED_READ,
        L::SHADER_READ_ONLY_OPTIMAL,
    );
    let before = [
        image_barrier(
            target.color_image(),
            Aspect::COLOR,
            (
                S::COLOR_ATTACHMENT_OUTPUT,
                A::COLOR_ATTACHMENT_WRITE,
                L::COLOR_ATTACHMENT_OPTIMAL,
            ),
            (S::COPY, A::TRANSFER_READ, L::TRANSFER_SRC_OPTIMAL),
        ),
        image_barrier(
            target.depth_image(),
            Aspect::DEPTH,
            (
                S::LATE_FRAGMENT_TESTS,
                A::DEPTH_STENCIL_ATTACHMENT_WRITE,
                L::DEPTH_ATTACHMENT_OPTIMAL,
            ),
            (S::COPY, A::TRANSFER_READ, L::TRANSFER_SRC_OPTIMAL),
        ),
        image_barrier(
            target.color_copy_image(),
            Aspect::COLOR,
            sampled,
            (S::COPY, A::TRANSFER_WRITE, L::TRANSFER_DST_OPTIMAL),
        ),
        image_barrier(
            target.depth_copy_image(),
            Aspect::DEPTH,
            sampled,
            (S::COPY, A::TRANSFER_WRITE, L::TRANSFER_DST_OPTIMAL),
        ),
    ];
    let region = |aspect| {
        let layers = vk::ImageSubresourceLayers::default()
            .aspect_mask(aspect)
            .layer_count(1);
        vk::ImageCopy::default()
            .src_subresource(layers)
            .dst_subresource(layers)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
    };
    let after = [
        image_barrier(
            target.color_image(),
            Aspect::COLOR,
            (S::COPY, A::TRANSFER_READ, L::TRANSFER_SRC_OPTIMAL),
            (
                S::COLOR_ATTACHMENT_OUTPUT,
                A::COLOR_ATTACHMENT_READ | A::COLOR_ATTACHMENT_WRITE,
                L::COLOR_ATTACHMENT_OPTIMAL,
            ),
        ),
        image_barrier(
            target.depth_image(),
            Aspect::DEPTH,
            (S::COPY, A::TRANSFER_READ, L::TRANSFER_SRC_OPTIMAL),
            (
                S::EARLY_FRAGMENT_TESTS | S::LATE_FRAGMENT_TESTS,
                A::DEPTH_STENCIL_ATTACHMENT_READ | A::DEPTH_STENCIL_ATTACHMENT_WRITE,
                L::DEPTH_ATTACHMENT_OPTIMAL,
            ),
        ),
        image_barrier(
            target.color_copy_image(),
            Aspect::COLOR,
            (S::COPY, A::TRANSFER_WRITE, L::TRANSFER_DST_OPTIMAL),
            sampled,
        ),
        image_barrier(
            target.depth_copy_image(),
            Aspect::DEPTH,
            (S::COPY, A::TRANSFER_WRITE, L::TRANSFER_DST_OPTIMAL),
            sampled,
        ),
    ];
    // SAFETY: `cmd` is recording outside a rendering scope (caller's contract) and the four
    // images belong to this device and to `target`, which outlives the recorded frame.
    unsafe {
        raw.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().image_memory_barriers(&before),
        );
        raw.cmd_copy_image(
            cmd,
            target.color_image(),
            L::TRANSFER_SRC_OPTIMAL,
            target.color_copy_image(),
            L::TRANSFER_DST_OPTIMAL,
            &[region(Aspect::COLOR)],
        );
        raw.cmd_copy_image(
            cmd,
            target.depth_image(),
            L::TRANSFER_SRC_OPTIMAL,
            target.depth_copy_image(),
            L::TRANSFER_DST_OPTIMAL,
            &[region(Aspect::DEPTH)],
        );
        raw.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().image_memory_barriers(&after),
        );
    }
}

/// An image barrier over the whole single-mip, single-layer image.
fn image_barrier(
    image: vk::Image,
    aspect: vk::ImageAspectFlags,
    (src_stage, src_access, old_layout): (
        vk::PipelineStageFlags2,
        vk::AccessFlags2,
        vk::ImageLayout,
    ),
    (dst_stage, dst_access, new_layout): (
        vk::PipelineStageFlags2,
        vk::AccessFlags2,
        vk::ImageLayout,
    ),
) -> vk::ImageMemoryBarrier2<'static> {
    vk::ImageMemoryBarrier2::default()
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(aspect)
                .level_count(1)
                .layer_count(1),
        )
}

/// The material block pushed before each section: the index of its record, then padding.
fn material_bytes(record: u32) -> [u8; 16] {
    let mut bytes = [0u8; 16];
    bytes[..4].copy_from_slice(&record.to_ne_bytes());
    bytes
}

/// The push constant block in the layout the shaders declare: view-projection, eye, params, and
/// material record 0 (the neutral one) that the box draw uses as is.
fn push_constants(params: &FrameParams) -> [u8; PUSH_SIZE as usize] {
    let selected = params.selected.map_or(-1.0, |index| index as f32);
    let eye = [params.eye[0], params.eye[1], params.eye[2], 0.0];
    let extra = [selected, params.grid_extent, params.grid_cell, 0.0];
    let mut bytes = [0u8; PUSH_SIZE as usize];
    let (chunks, _) = bytes.as_chunks_mut::<4>();
    let values = params.view_proj.iter().chain(&eye).chain(&extra);
    for (chunk, value) in chunks.iter_mut().zip(values) {
        *chunk = value.to_ne_bytes();
    }
    // The material block (bytes 96..112) stays zero: record 0.
    bytes
}

impl Drop for SceneRenderer {
    fn drop(&mut self) {
        if let Err(e) = self.device.wait_idle() {
            tracing::error!("waiting for the GPU before destroying the scene renderer failed: {e}");
        }
        self.retired.clear();
        self.meshes.clear();
        self.unity = None;
        self.timer = None;
        self.pipelines.destroy(&self.device);
        self.present.destroy(&self.device);
        self.bloom.destroy(&self.device);
        // Buffers, textures and the target free themselves when the fields drop, after this body.
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn push_constants_follow_the_shader_layout() {
        let mut view_proj = [0.0; 16];
        view_proj[0] = 1.0;
        view_proj[15] = 9.0;
        let bytes = push_constants(&FrameParams {
            view_proj,
            eye: [1.0, 2.0, 3.0],
            selected: Some(7),
            clear: [0.0; 4],
            grid_extent: 500.0,
            grid_cell: 10.0,
        });
        let f = |i: usize| f32::from_ne_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
        assert_eq!((f(0), f(15)), (1.0, 9.0));
        assert_eq!((f(16), f(17), f(18)), (1.0, 2.0, 3.0));
        assert_eq!((f(20), f(21), f(22)), (7.0, 500.0, 10.0));
        assert_eq!(bytes.len(), 112);
        assert_eq!(&bytes[96..112], &[0u8; 16], "material record 0");
        assert_eq!(material_bytes(7)[..4], 7u32.to_ne_bytes());
        let none = push_constants(&FrameParams {
            selected: None,
            view_proj,
            eye: [0.0; 3],
            clear: [0.0; 4],
            grid_extent: 1.0,
            grid_cell: 1.0,
        });
        assert_eq!(f32::from_ne_bytes(none[80..84].try_into().unwrap()), -1.0);
    }
}
