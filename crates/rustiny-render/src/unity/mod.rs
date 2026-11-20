//! Drawing with a Unity project's own shaders.
//!
//! The editor compiles a Unity pass's HLSL to SPIR-V (`rustiny-shader`) and hands the two stages
//! to [`SceneRenderer::add_shader_pass`](crate::SceneRenderer::add_shader_pass) together with the
//! pass's render state. The renderer reflects them, gives both stages one descriptor numbering,
//! builds a pipeline and then fills the pass's constant buffers *by name*:
//!
//! * `UnityPerMaterial` (and loose globals) from the material's properties
//!   ([`ShaderMaterialDesc`]), once per material;
//! * `UnityPerDraw` (`unity_ObjectToWorld`, ...) from each instance, for every draw (instances of
//!   these sections are drawn one by one, from a ring of per-frame memory);
//! * every other constant buffer (camera, time, light, ambient) from the [`UnityFrame`].
//!
//! Textures the material names are bound once uploaded; everything else a shader asks for (depth,
//! opaque colour, probes, shadow maps, ...) is a one-texel stand-in. What this does not do yet:
//! shadows and other camera textures, keyword variants other than the editor's choice, culling by
//! face (both sides are drawn), instancing, sorting of transparent draws.

mod draws;
mod dummy;
mod pass;
mod values;

use crate::buffer::HostBuffer;
use crate::geometry::GpuInstance;
use crate::matrix::{invert, multiply};
use crate::mesh::{GpuMesh, GpuSection};
use crate::texture::Textures;
use ash::vk;
use draws::{DrawCache, Owner, RING};
use dummy::Dummies;
use pass::{CONSTANT_BINDING, CONSTANT_VERTEX, Kind, MESH_BINDING, Pass};
use rustiny_vulkan::{Device, FRAMES_IN_FLIGHT, Result, VkResultExt, VulkanError};
use std::collections::HashMap;
use std::sync::Arc;
use values::{Constants, fill_block, handedness, write_member};

/// Which faces a pass culls (Unity's `Cull`). Recorded, but both sides are drawn for now.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CullMode {
    /// No culling.
    Off,
    /// Cull front faces.
    Front,
    /// Cull back faces.
    #[default]
    Back,
}

/// Unity's `ZTest`. "Less" means nearer to the camera.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DepthTest {
    /// Never passes.
    Never,
    /// Nearer than what is there.
    Less,
    /// Equally near.
    Equal,
    /// Nearer or equal.
    #[default]
    LessEqual,
    /// Farther.
    Greater,
    /// Not equally near.
    NotEqual,
    /// Farther or equal.
    GreaterEqual,
    /// Always passes.
    Always,
}

/// Unity's blend factors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendFactor {
    /// 0.
    Zero,
    /// 1.
    One,
    /// Source colour.
    SrcColor,
    /// 1 - source colour.
    OneMinusSrcColor,
    /// Source alpha.
    SrcAlpha,
    /// 1 - source alpha.
    OneMinusSrcAlpha,
    /// Destination colour.
    DstColor,
    /// 1 - destination colour.
    OneMinusDstColor,
    /// Destination alpha.
    DstAlpha,
    /// 1 - destination alpha.
    OneMinusDstAlpha,
}

/// `Blend src dst, srcAlpha dstAlpha`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Blend {
    /// Factor of the shader's colour.
    pub src_color: BlendFactor,
    /// Factor of what is already in the target.
    pub dst_color: BlendFactor,
    /// Factor of the shader's alpha.
    pub src_alpha: BlendFactor,
    /// Factor of the target's alpha.
    pub dst_alpha: BlendFactor,
}

/// The fixed-function state of a pass, with ShaderLab's `[_Property]` references already resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PassState {
    /// `Cull`.
    pub cull: CullMode,
    /// `ZWrite`.
    pub z_write: bool,
    /// `ZTest`.
    pub z_test: DepthTest,
    /// `Blend`, `None` for opaque output.
    pub blend: Option<Blend>,
    /// `ColorMask` as bits: 1 red, 2 green, 4 blue, 8 alpha.
    pub color_mask: u8,
}

impl Default for PassState {
    /// Unity's defaults: cull back, write depth, `LEqual`, no blending, all channels.
    fn default() -> Self {
        Self {
            cull: CullMode::Back,
            z_write: true,
            z_test: DepthTest::LessEqual,
            blend: None,
            color_mask: 0xf,
        }
    }
}

/// A compiled pass: both stages as SPIR-V (any binding numbering) and its state.
#[derive(Debug, Clone)]
pub struct ShaderPassDesc {
    /// The vertex stage.
    pub vertex: Vec<u32>,
    /// The fragment stage.
    pub fragment: Vec<u32>,
    /// Render state.
    pub state: PassState,
}

/// What a texture slot shows until its texture is on the GPU (and for a slot with no texture).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextureFallback {
    /// Unity's `white`.
    #[default]
    White,
    /// `black`.
    Black,
    /// `gray`.
    Grey,
    /// `bump`: a flat normal map.
    Bump,
}

/// A material property as the shader reads it.
#[derive(Debug, Clone, PartialEq)]
pub enum PropertyValue {
    /// A float, colour or vector: scalars use the first component. Colours are as Unity stores
    /// them (gamma-space for non-HDR colour properties is the editor's business).
    Vector([f32; 4]),
    /// A texture, with its tiling and offset (`_ST`: scale x/y, offset x/y).
    Texture {
        /// Texture key given to `add_texture`, `None` for an empty slot.
        key: Option<u32>,
        /// `[scale.x, scale.y, offset.x, offset.y]`.
        scale_offset: [f32; 4],
        /// What to show while there is no texture.
        fallback: TextureFallback,
    },
}

/// A material for a shader pass.
#[derive(Debug, Clone, PartialEq)]
pub struct ShaderMaterialDesc {
    /// Index returned by `add_shader_pass`.
    pub pass: u32,
    /// Properties by name (`_BaseColor`, `_BaseMap`, ...).
    pub properties: Vec<(String, PropertyValue)>,
}

/// Ambient light as Unity's "Gradient" (trilight) mode defines it: the colour seen looking up, at
/// the horizon and down. Linear, with the ambient intensity applied. A flat ambient has three
/// equal colours.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ambient {
    /// Light from above.
    pub sky: [f32; 3],
    /// Light from the horizon.
    pub equator: [f32; 3],
    /// Light from below.
    pub ground: [f32; 3],
}

impl Ambient {
    /// The same light from every direction.
    pub fn flat(color: [f32; 3]) -> Self {
        Self {
            sky: color,
            equator: color,
            ground: color,
        }
    }

    /// The spherical-harmonic coefficients URP's `SampleSH` reads: `unity_SHAr/Ag/Ab` (linear term
    /// in xyz, constant in w), `unity_SHBr/Bg/Bb` and `unity_SHC` (quadratic terms). They are
    /// chosen so that `SampleSH(n)` is a function of the height `n.y` alone, `a + b y + c y^2`,
    /// that gives exactly the sky colour straight up, the equator colour on the horizon and the
    /// ground colour straight down.
    pub fn spherical_harmonics(&self) -> [[f32; 4]; 7] {
        let mut sh = [[0.0f32; 4]; 7];
        for channel in 0..3 {
            let (up, side, down) = (
                self.sky[channel],
                self.equator[channel],
                self.ground[channel],
            );
            let a = side;
            let b = (up - down) * 0.5;
            let c = (up + down) * 0.5 - side;
            // y^2 = (1 - (x^2 - y^2) - z^2) / 2, and SampleSH has (x^2 - y^2) in unity_SHC and
            // z^2 in the third component of unity_SHB*.
            sh[channel] = [0.0, b, 0.0, a + c * 0.5];
            sh[3 + channel][2] = -c * 0.5;
            sh[6][channel] = -c * 0.5;
        }
        sh
    }
}

/// Per-frame values the shaders read: the camera, the clock, the main light and the ambient.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UnityFrame {
    /// World-to-view matrix in Unity's convention (see [`crate::matrix::view_matrix`]).
    pub view: [f32; 16],
    /// View-to-clip matrix (see [`crate::matrix::projection`]).
    pub projection: [f32; 16],
    /// Camera position.
    pub eye: [f32; 3],
    /// Near plane distance.
    pub near: f32,
    /// Far plane distance reported to the shaders (the real far plane is infinite).
    pub far: f32,
    /// Size of the target in pixels.
    pub width: u32,
    /// Size of the target in pixels.
    pub height: u32,
    /// Seconds since the start.
    pub time: f32,
    /// Unit vector pointing towards the main light.
    pub light_direction: [f32; 3],
    /// Colour of the main light, linear.
    pub light_color: [f32; 3],
    /// Ambient light.
    pub ambient: Ambient,
    /// `RenderSettings.m_ReflectionIntensity`: scales the environment reflection.
    pub reflection_intensity: f32,
}

impl Default for UnityFrame {
    fn default() -> Self {
        let mut identity = [0.0; 16];
        for i in 0..4 {
            identity[i * 5] = 1.0;
        }
        let light = [0.4f32, 0.8, -0.35];
        let length = light.iter().map(|v| v * v).sum::<f32>().sqrt();
        Self {
            view: identity,
            projection: identity,
            eye: [0.0; 3],
            near: 0.1,
            far: 5000.0,
            width: 1,
            height: 1,
            time: 0.0,
            light_direction: light.map(|v| v / length),
            light_color: [1.0, 0.96, 0.9],
            ambient: Ambient::flat([0.30, 0.33, 0.38]),
            reflection_intensity: 1.0,
        }
    }
}

/// Which part of a frame's Unity-shader draws to record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Phase {
    /// Passes that do not blend: drawn first.
    Opaque,
    /// Passes that blend: drawn after the skybox and the copy of the camera textures.
    Transparent,
}

/// A texture the camera provides to shaders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CameraTexture {
    /// `_CameraDepthTexture`: the depth of the opaque draws.
    Depth,
    /// `_CameraOpaqueTexture`: the colour of the opaque draws.
    Opaque,
}

/// The camera texture a shader variable name asks for, if it is one.
pub(crate) fn camera_texture(name: &str) -> Option<CameraTexture> {
    match name.to_ascii_lowercase().as_str() {
        "_cameradepthtexture" => Some(CameraTexture::Depth),
        "_cameraopaquetexture" => Some(CameraTexture::Opaque),
        _ => None,
    }
}

/// The images behind the camera textures, and which render target they belong to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CameraViews {
    pub depth: vk::ImageView,
    pub color: vk::ImageView,
    /// Changes whenever the images are replaced (a resize), so cached descriptor sets are dropped.
    pub generation: u64,
}

/// Sets per descriptor pool.
const SETS_PER_POOL: u32 = 64;
/// Marks "no texture" in a set's cache key.
const DEFAULT_KEY: u32 = u32::MAX;
/// Bytes of `UnityPerMaterial` data for all materials.
const MATERIAL_BUFFER: u64 = 16 << 20;
/// Bytes of per-frame constant data (the per-material frame blocks) for one frame, and how many
/// frames the ring holds. Per-object data lives in the [`DrawCache`].
const REGION: u64 = 8 << 20;
const REGIONS: u64 = FRAMES_IN_FLIGHT as u64 + 1;

/// Whether a box (`bounds` in the mesh's space, placed by `model`) lies entirely outside one
/// plane of the view frustum. Conservative: a box that straddles a corner is kept. Clip space is
/// Vulkan's with a reverse-Z infinite far plane, so a point is inside when
/// `-w <= x <= w`, `-w <= y <= w` and `0 <= z <= w`.
fn outside_frustum(view_proj: &[f32; 16], model: &[f32; 16], bounds: &[[f32; 3]; 2]) -> bool {
    if !bounds[0][0].is_finite() {
        return false;
    }
    let mvp = multiply(view_proj, model);
    // Bit per plane: set while every corner so far is outside it.
    let mut outside = 0b11_1111u8;
    for corner in 0..8 {
        let p = [
            bounds[corner & 1][0],
            bounds[(corner >> 1) & 1][1],
            bounds[(corner >> 2) & 1][2],
        ];
        let clip = |row: usize| {
            mvp[row] * p[0] + mvp[4 + row] * p[1] + mvp[8 + row] * p[2] + mvp[12 + row]
        };
        let (x, y, z, w) = (clip(0), clip(1), clip(2), clip(3));
        let mut inside = 0u8;
        if x >= -w {
            inside |= 1;
        }
        if x <= w {
            inside |= 2;
        }
        if y >= -w {
            inside |= 4;
        }
        if y <= w {
            inside |= 8;
        }
        if z >= 0.0 {
            inside |= 16;
        }
        if z <= w {
            inside |= 32;
        }
        outside &= !inside;
        if outside == 0 {
            return false;
        }
    }
    outside != 0
}

/// A material of a shader pass, ready to draw with.
struct UnityMaterial {
    pass: u32,
    properties: HashMap<String, PropertyValue>,
    /// Offset of its `UnityPerMaterial` bytes in the material buffer.
    region: u64,
    /// The pass's texture descriptors, in binding order: what each shows.
    slots: Vec<Slot>,
}

struct Slot {
    key: Option<u32>,
    fallback: TextureFallback,
}

impl UnityMaterial {
    /// The numbers of the property `name` (`<texture>_ST` gives the texture's tiling).
    fn numbers(&self, name: &str) -> Option<&[f32]> {
        match self.properties.get(name) {
            Some(PropertyValue::Vector(v)) => Some(&v[..]),
            Some(PropertyValue::Texture { .. }) => None,
            None => {
                let base = name.strip_suffix("_ST")?;
                match self.properties.get(base)? {
                    PropertyValue::Texture { scale_offset, .. } => Some(&scale_offset[..]),
                    PropertyValue::Vector(_) => None,
                }
            }
        }
    }
}

/// Per-frame memory the dynamic constant buffers are written into.
struct Arena {
    buffer: HostBuffer,
    base: u64,
    cursor: u64,
}

impl Arena {
    /// Reserve `size` bytes aligned to `align`; the offset is absolute in the buffer.
    fn alloc(&mut self, size: u64, align: u64) -> Option<u64> {
        let start = self.cursor.div_ceil(align) * align;
        let end = start + size;
        if end > REGION {
            return None;
        }
        self.cursor = end;
        Some(self.base + start)
    }
}

/// What the renderer holds for Unity shaders. Created with the first pass.
pub(crate) struct UnityShaders {
    device: Arc<Device>,
    color_format: vk::Format,
    depth_format: vk::Format,
    passes: Vec<Pass>,
    materials: Vec<UnityMaterial>,
    material_buffer: HostBuffer,
    material_cursor: u64,
    arena: Arena,
    constant_vertex: HostBuffer,
    storage: HostBuffer,
    /// Repeat, clamp, point + clamp, point + repeat (all linear-mip except the point ones).
    samplers: [vk::Sampler; 4],
    dummies: Dummies,
    alignment: u64,
    pools: Vec<vk::DescriptorPool>,
    pool_free: u32,
    old_pools: Vec<(u64, vk::DescriptorPool)>,
    /// Sets by material, then by the effective texture keys they present.
    sets: HashMap<u32, Vec<(Vec<u32>, vk::DescriptorSet)>>,
    epoch: u64,
    camera: Option<CameraViews>,
    /// Values the project's scripts would set with `Shader.SetGlobal*`.
    globals: Arc<HashMap<String, Vec<f32>>>,
    // Per frame:
    view_proj: [f32; 16],
    constants: Constants,
    frame_offsets: HashMap<u32, Vec<u32>>,
    templates: HashMap<u32, Vec<u8>>,
    bound_pass: Option<u32>,
    warned: bool,
    /// Per-object blocks, kept between frames.
    draws: DrawCache,
    /// The draw cache generation the descriptor sets were made for.
    sets_draw_generation: u64,
    /// The recorded pass this frame belongs to, and how many per-frame draws it made so far.
    records: u64,
    frame_draws: u32,
}

impl std::fmt::Debug for UnityShaders {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnityShaders")
            .field("passes", &self.passes.len())
            .field("materials", &self.materials.len())
            .finish_non_exhaustive()
    }
}

impl Drop for UnityShaders {
    fn drop(&mut self) {
        let raw = self.device.raw();
        // SAFETY: the owner retires this value until the GPU is done with it (or waits for the
        // device to be idle), so nothing uses these objects any more. Buffers and images free
        // themselves afterwards.
        unsafe {
            for pass in &self.passes {
                pass.destroy(&self.device);
            }
            for pool in self.pools.drain(..) {
                raw.destroy_descriptor_pool(pool, None);
            }
            for (_, pool) in self.old_pools.drain(..) {
                raw.destroy_descriptor_pool(pool, None);
            }
            for sampler in self.samplers {
                raw.destroy_sampler(sampler, None);
            }
        }
    }
}

fn sampler(
    device: &Device,
    filter: vk::Filter,
    address: vk::SamplerAddressMode,
) -> Result<vk::Sampler> {
    let info = vk::SamplerCreateInfo::default()
        .mag_filter(filter)
        .min_filter(filter)
        .mipmap_mode(if filter == vk::Filter::LINEAR {
            vk::SamplerMipmapMode::LINEAR
        } else {
            vk::SamplerMipmapMode::NEAREST
        })
        .address_mode_u(address)
        .address_mode_v(address)
        .address_mode_w(address)
        .max_lod(vk::LOD_CLAMP_NONE);
    // SAFETY: plain object creation on a live device.
    unsafe { device.raw().create_sampler(&info, None) }.vk("vkCreateSampler(unity)")
}

impl UnityShaders {
    pub(crate) fn new(
        device: &Arc<Device>,
        color_format: vk::Format,
        depth_format: vk::Format,
    ) -> Result<Self> {
        // SAFETY: query call on a live instance and a physical device of it.
        let limits = unsafe {
            device
                .instance()
                .raw()
                .get_physical_device_properties(device.physical())
        }
        .limits;
        let alignment = limits.min_uniform_buffer_offset_alignment.max(16);
        let constants: Vec<u8> = CONSTANT_VERTEX
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        let linear = vk::Filter::LINEAR;
        let point = vk::Filter::NEAREST;
        let (repeat, clamp) = (
            vk::SamplerAddressMode::REPEAT,
            vk::SamplerAddressMode::CLAMP_TO_EDGE,
        );
        let mut samplers = Vec::new();
        for (filter, address) in [
            (linear, repeat),
            (linear, clamp),
            (point, clamp),
            (point, repeat),
        ] {
            match sampler(device, filter, address) {
                Ok(s) => samplers.push(s),
                Err(e) => {
                    for s in samplers {
                        // SAFETY: never used.
                        unsafe { device.raw().destroy_sampler(s, None) };
                    }
                    return Err(e);
                }
            }
        }
        let samplers: [vk::Sampler; 4] = samplers
            .try_into()
            .map_err(|_| VulkanError::Allocator("sampler creation".into()))?;
        let built = (|| {
            Ok::<_, VulkanError>((
                HostBuffer::zeroed(
                    device,
                    MATERIAL_BUFFER as usize,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    "unity-materials",
                )?,
                HostBuffer::zeroed(
                    device,
                    (REGION * REGIONS) as usize,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    "unity-frame-arena",
                )?,
                HostBuffer::with_data(
                    device,
                    &constants,
                    vk::BufferUsageFlags::VERTEX_BUFFER,
                    "unity-constant-vertex",
                )?,
                HostBuffer::zeroed(
                    device,
                    64 << 10,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    "unity-storage-default",
                )?,
                Dummies::new(device)?,
            ))
        })();
        let draws = DrawCache::new(device)?;
        let (material_buffer, arena, constant_vertex, storage, dummies) = match built {
            Ok(parts) => parts,
            Err(e) => {
                for s in samplers {
                    // SAFETY: never used.
                    unsafe { device.raw().destroy_sampler(s, None) };
                }
                return Err(e);
            }
        };
        Ok(Self {
            device: device.clone(),
            color_format,
            depth_format,
            passes: Vec::new(),
            materials: Vec::new(),
            material_buffer,
            material_cursor: 0,
            arena: Arena {
                buffer: arena,
                base: 0,
                cursor: 0,
            },
            constant_vertex,
            storage,
            samplers,
            dummies,
            alignment,
            pools: Vec::new(),
            pool_free: 0,
            old_pools: Vec::new(),
            sets: HashMap::new(),
            epoch: 0,
            camera: None,
            globals: Arc::new(HashMap::new()),
            view_proj: [0.0; 16],
            constants: Constants::default(),
            frame_offsets: HashMap::new(),
            templates: HashMap::new(),
            bound_pass: None,
            warned: false,
            draws,
            sets_draw_generation: 0,
            records: 0,
            frame_draws: 0,
        })
    }

    pub(crate) fn add_pass(&mut self, desc: &ShaderPassDesc) -> Result<u32> {
        let pass = Pass::new(&self.device, desc, self.color_format, self.depth_format)?;
        self.passes.push(pass);
        Ok(self.passes.len() as u32 - 1)
    }

    pub(crate) fn has_material(&self, id: u32) -> bool {
        (id as usize) < self.materials.len()
    }

    pub(crate) fn material_count(&self) -> usize {
        self.materials.len()
    }

    pub(crate) fn add_material(&mut self, desc: &ShaderMaterialDesc) -> Result<u32> {
        let Some(pass) = self.passes.get(desc.pass as usize) else {
            return Err(VulkanError::InvalidInput(format!(
                "shader pass {} does not exist",
                desc.pass
            )));
        };
        let properties: HashMap<String, PropertyValue> = desc.properties.iter().cloned().collect();
        let slots: Vec<Slot> = pass
            .textures
            .iter()
            .map(|&i| match properties.get(&pass.descriptors[i].name) {
                Some(PropertyValue::Texture { key, fallback, .. }) => Slot {
                    key: *key,
                    fallback: *fallback,
                },
                _ => Slot {
                    key: None,
                    fallback: TextureFallback::Black,
                },
            })
            .collect();
        let mut material = UnityMaterial {
            pass: desc.pass,
            properties,
            region: 0,
            slots,
        };
        if let Some(block) = pass
            .per_material
            .and_then(|i| pass.descriptors[i].block.as_ref())
        {
            let start = self.material_cursor.div_ceil(self.alignment) * self.alignment;
            let size = u64::from(block.size);
            if start + size > MATERIAL_BUFFER {
                return Err(VulkanError::InvalidInput(
                    "too many shader materials".into(),
                ));
            }
            let mut bytes = vec![0u8; block.size as usize];
            fill_block(block, &mut bytes, |name| material.numbers(name));
            self.material_buffer.write_at(start as usize, &bytes)?;
            self.material_cursor = start + size;
            material.region = start;
        }
        self.materials.push(material);
        Ok(self.materials.len() as u32 - 1)
    }

    /// Start of a recorded pass: fill the stand-in textures (once), retire sets that name replaced
    /// textures, and take this frame's part of the memory ring.
    #[allow(clippy::too_many_arguments)] // Everything a frame starts from; called once per frame.
    pub(crate) fn begin_frame(
        &mut self,
        cmd: vk::CommandBuffer,
        records: u64,
        frame: &UnityFrame,
        view_proj: &[f32; 16],
        textures: &Textures,
        camera: Option<CameraViews>,
        globals: &Arc<HashMap<String, Vec<f32>>>,
    ) {
        let raw = self.device.raw();
        self.globals = globals.clone();
        self.old_pools.retain(|(stamp, pool)| {
            let keep = records <= stamp + FRAMES_IN_FLIGHT as u64 + 1;
            if !keep {
                // SAFETY: the frames that could use the pool have finished.
                unsafe { raw.destroy_descriptor_pool(*pool, None) };
            }
            keep
        });
        self.dummies.record_init(cmd);
        self.records = records;
        self.frame_draws = 0;
        self.draws.begin_frame(records);
        if textures.epoch() != self.epoch
            || self.camera.map(|c| c.generation) != camera.map(|c| c.generation)
        {
            self.epoch = textures.epoch();
            self.camera = camera;
            self.drop_sets();
        }
        self.arena.base = (records % REGIONS) * REGION;
        self.arena.cursor = 0;
        self.constants = Constants::new(frame, view_proj);
        self.view_proj = *view_proj;
        self.frame_offsets.clear();
        self.templates.clear();
        self.bound_pass = None;
        self.warned = false;
    }

    /// Retire every descriptor set (they name textures or buffers that changed).
    fn drop_sets(&mut self) {
        let records = self.records;
        self.old_pools
            .extend(self.pools.drain(..).map(|pool| (records, pool)));
        self.pool_free = 0;
        self.sets.clear();
        self.sets_draw_generation = self.draws.generation();
    }

    /// The scene's instances were replaced: their per-object blocks are written again.
    pub(crate) fn forget_instances(&mut self) {
        self.draws.forget_instances();
    }

    /// Whether some pass needs the camera textures or blends, so the frame has to be drawn in two
    /// parts with a copy of colour and depth in between.
    pub(crate) fn needs_camera_copy(&self) -> bool {
        self.passes
            .iter()
            .any(|p| p.transparent || p.camera_textures)
    }

    /// Whether a draw has bound a Unity pipeline since the frame began.
    pub(crate) fn used(&self) -> bool {
        self.bound_pass.is_some()
    }

    fn allocate_set(&mut self, layout: vk::DescriptorSetLayout) -> Result<vk::DescriptorSet> {
        let raw = self.device.raw();
        let layouts = [layout];
        let mut last = None;
        for _ in 0..2 {
            if self.pool_free > 0
                && let Some(&pool) = self.pools.last()
            {
                let info = vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts);
                // SAFETY: the pool and layout are live and belong to this device.
                match unsafe { raw.allocate_descriptor_sets(&info) } {
                    Ok(sets) if !sets.is_empty() => {
                        self.pool_free -= 1;
                        return Ok(sets[0]);
                    }
                    Ok(_) => {}
                    Err(result) => last = Some(result),
                }
            }
            let sizes = [
                (vk::DescriptorType::UNIFORM_BUFFER, 4),
                (vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC, 8),
                (vk::DescriptorType::SAMPLED_IMAGE, 64),
                (vk::DescriptorType::SAMPLER, 16),
                (vk::DescriptorType::STORAGE_BUFFER, 4),
            ]
            .map(|(ty, per_set)| {
                vk::DescriptorPoolSize::default()
                    .ty(ty)
                    .descriptor_count(per_set * SETS_PER_POOL)
            });
            let info = vk::DescriptorPoolCreateInfo::default()
                .max_sets(SETS_PER_POOL)
                .pool_sizes(&sizes);
            // SAFETY: plain object creation on a live device.
            let pool = unsafe { raw.create_descriptor_pool(&info, None) }
                .vk("vkCreateDescriptorPool(unity)")?;
            self.pools.push(pool);
            self.pool_free = SETS_PER_POOL;
        }
        Err(VulkanError::Call {
            call: "vkAllocateDescriptorSets(unity)",
            result: last.unwrap_or(vk::Result::ERROR_OUT_OF_POOL_MEMORY),
        })
    }

    /// The descriptor set that presents `material` with the textures that are ready now.
    fn set_for(&mut self, material: u32, textures: &Textures) -> Result<vk::DescriptorSet> {
        let mat = &self.materials[material as usize];
        let effective: Vec<u32> = mat
            .slots
            .iter()
            .map(|slot| match slot.key {
                Some(key) if textures.is_ready(key) => key,
                _ => DEFAULT_KEY,
            })
            .collect();
        if let Some((_, set)) = self
            .sets
            .get(&material)
            .and_then(|sets| sets.iter().find(|(keys, _)| *keys == effective))
        {
            return Ok(*set);
        }

        enum Write {
            Buffer(u32, vk::DescriptorType, vk::DescriptorBufferInfo),
            Image(u32, vk::DescriptorImageInfo),
            Sampler(u32, vk::DescriptorImageInfo),
        }
        let pass = &self.passes[mat.pass as usize];
        let mut writes = Vec::with_capacity(pass.descriptors.len());
        let mut slot = 0;
        for descriptor in &pass.descriptors {
            let binding = descriptor.binding;
            let size = descriptor
                .block
                .as_ref()
                .map_or(16, |b| u64::from(b.size.max(16)));
            writes.push(match descriptor.kind {
                Kind::PerDraw => Write::Buffer(
                    binding,
                    vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                    vk::DescriptorBufferInfo::default()
                        .buffer(self.draws.handle())
                        .range(size),
                ),
                Kind::Frame => Write::Buffer(
                    binding,
                    vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                    vk::DescriptorBufferInfo::default()
                        .buffer(self.arena.buffer.handle())
                        .range(size),
                ),
                Kind::PerMaterial => Write::Buffer(
                    binding,
                    vk::DescriptorType::UNIFORM_BUFFER,
                    vk::DescriptorBufferInfo::default()
                        .buffer(self.material_buffer.handle())
                        .offset(mat.region)
                        .range(size),
                ),
                Kind::Storage => Write::Buffer(
                    binding,
                    vk::DescriptorType::STORAGE_BUFFER,
                    vk::DescriptorBufferInfo::default()
                        .buffer(self.storage.handle())
                        .range(vk::WHOLE_SIZE),
                ),
                Kind::Texture(kind) => {
                    let wanted = mat.slots.get(slot);
                    let key = effective.get(slot).copied().unwrap_or(DEFAULT_KEY);
                    slot += 1;
                    let camera = camera_texture(&descriptor.name).zip(self.camera);
                    let view = match camera {
                        Some((CameraTexture::Depth, c)) => c.depth,
                        Some((CameraTexture::Opaque, c)) => c.color,
                        None => textures
                            .view_of(key)
                            .filter(|_| key != DEFAULT_KEY)
                            .unwrap_or_else(|| {
                                self.dummies.view(
                                    kind,
                                    wanted.map_or(TextureFallback::Black, |s| s.fallback),
                                )
                            }),
                    };
                    Write::Image(
                        binding,
                        vk::DescriptorImageInfo::default()
                            .image_view(view)
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                    )
                }
                Kind::Sampler => {
                    let name = descriptor.name.to_ascii_lowercase();
                    // Depth is never filtered; the opaque colour is clamped like in URP.
                    let (point, clamp) = if name.contains("cameradepth") {
                        (true, true)
                    } else if name.contains("cameraopaque") {
                        (false, true)
                    } else {
                        (name.contains("point"), name.contains("clamp"))
                    };
                    let which = match (point, clamp) {
                        (true, true) => 2,
                        (true, false) => 3,
                        (false, true) => 1,
                        (false, false) => 0,
                    };
                    Write::Sampler(
                        binding,
                        vk::DescriptorImageInfo::default().sampler(self.samplers[which]),
                    )
                }
            });
        }
        let layout = pass.set_layout;
        let set = self.allocate_set(layout)?;
        let updates: Vec<vk::WriteDescriptorSet> = writes
            .iter()
            .map(|write| match write {
                Write::Buffer(binding, ty, info) => vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(*binding)
                    .descriptor_type(*ty)
                    .buffer_info(std::slice::from_ref(info)),
                Write::Image(binding, info) => vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(*binding)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(std::slice::from_ref(info)),
                Write::Sampler(binding, info) => vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(*binding)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .image_info(std::slice::from_ref(info)),
            })
            .collect();
        // SAFETY: the set is new (no submitted work uses it) and the buffers, views and samplers
        // are live.
        unsafe { self.device.raw().update_descriptor_sets(&updates, &[]) };
        self.sets
            .entry(material)
            .or_default()
            .push((effective, set));
        Ok(set)
    }

    /// The dynamic offsets of the buffers that are not per draw, written for this material once
    /// per frame. The per-draw slot holds `u32::MAX`.
    fn frame_offsets(&mut self, material: u32) -> Option<Vec<u32>> {
        if let Some(offsets) = self.frame_offsets.get(&material) {
            return Some(offsets.clone());
        }
        let mat = &self.materials[material as usize];
        let pass = &self.passes[mat.pass as usize];
        let mut offsets = Vec::with_capacity(pass.dynamic.len());
        for &index in &pass.dynamic {
            let descriptor = &pass.descriptors[index];
            let Some(block) = &descriptor.block else {
                return None;
            };
            if descriptor.kind == Kind::PerDraw {
                offsets.push(u32::MAX);
                continue;
            }
            let at = self.arena.alloc(u64::from(block.size), self.alignment)?;
            let mut bytes = vec![0u8; block.size as usize];
            let (constants, globals) = (&self.constants, &self.globals);
            fill_block(block, &mut bytes, |name| {
                mat.numbers(name)
                    .or_else(|| constants.get(name))
                    .or_else(|| globals.get(name).map(Vec::as_slice))
            });
            self.arena.buffer.write_at(at as usize, &bytes).ok()?;
            offsets.push(at as u32);
        }
        self.frame_offsets.insert(material, offsets.clone());
        Some(offsets)
    }

    /// Draw every visible instance of one section with its shader material. Returns the draw
    /// calls recorded and the instances left out (only when memory for their blocks could not be
    /// had). Call between `cmd_begin_rendering` and `cmd_end_rendering`.
    pub(crate) fn draw_section(
        &mut self,
        cmd: vk::CommandBuffer,
        textures: &Textures,
        draw: &SectionDraw<'_>,
    ) -> (u32, u32) {
        let SectionDraw {
            mesh,
            section,
            instances,
            first_instance,
            selected,
            phase,
        } = *draw;
        let Some(material) = section.shader.filter(|m| self.has_material(*m)) else {
            return (0, 0);
        };
        let pass_index = self.materials[material as usize].pass;
        let transparent = self.passes[pass_index as usize].transparent;
        if (phase == Phase::Opaque && transparent) || (phase == Phase::Transparent && !transparent)
        {
            return (0, 0);
        }
        let Some(offsets) = self.frame_offsets(material) else {
            return (0, instances.len() as u32);
        };
        let pass = &self.passes[pass_index as usize];
        let draw_slot = pass
            .per_draw
            .and_then(|d| pass.dynamic.iter().position(|&i| i == d));
        let draw_block = pass
            .per_draw
            .and_then(|d| pass.descriptors[d].block.as_ref())
            .cloned();
        let draw_size = draw_block.as_ref().map_or(0, |b| u64::from(b.size));
        let template = match &draw_block {
            Some(block) => self
                .templates
                .entry(pass_index)
                .or_insert_with(|| {
                    let mut bytes = vec![0u8; block.size as usize];
                    let constants = &self.constants;
                    fill_block(block, &mut bytes, |name| constants.get(name));
                    bytes
                })
                .clone(),
            None => Vec::new(),
        };
        if draw_size > 0 {
            if first_instance.is_some() {
                self.draws.use_template(pass_index, &template);
            }
            if let Err(e) = self.draws.reserve(
                instances.len() as u64,
                draw_size,
                self.alignment,
                self.records,
            ) {
                if !self.warned {
                    tracing::warn!("no memory for per-object shader data: {e}");
                    self.warned = true;
                }
                return (0, instances.len() as u32);
            }
            if self.draws.generation() != self.sets_draw_generation {
                // The blocks moved to a new buffer: sets naming the old one are retired.
                self.drop_sets();
            }
        }
        let set = match self.set_for(material, textures) {
            Ok(set) => set,
            Err(e) => {
                if !self.warned {
                    tracing::warn!("cannot make a descriptor set for a Unity material: {e}");
                    self.warned = true;
                }
                return (0, 0);
            }
        };
        let pass = &self.passes[pass_index as usize];
        let (layout, pipeline) = (pass.layout, pass.pipeline);

        let device = self.device.clone();
        let raw = device.raw();
        // SAFETY: `cmd` is recording inside a rendering scope (caller's contract); every handle
        // is owned by `self` or the mesh and outlives the frames using it (retirement queue).
        unsafe {
            if self.bound_pass != Some(pass_index) {
                raw.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
                self.bound_pass = Some(pass_index);
            }
            raw.cmd_bind_vertex_buffers(cmd, MESH_BINDING, &[mesh.vertices.handle()], &[0]);
            raw.cmd_bind_vertex_buffers(
                cmd,
                CONSTANT_BINDING,
                &[self.constant_vertex.handle()],
                &[0],
            );
            raw.cmd_bind_index_buffer(cmd, mesh.indices.handle(), 0, vk::IndexType::UINT32);
        }

        let members = self.passes[pass_index as usize].draw_members.clone();
        let (mut draws, mut skipped) = (0, 0);
        let mut offsets = offsets;
        for (index, instance) in instances.iter().enumerate() {
            if instance.color[3] == 0.0 && selected != Some(instance.object) {
                continue;
            }
            if outside_frustum(&self.view_proj, &instance.model, &mesh.bounds) {
                continue;
            }
            if draw_size > 0 {
                let owner = match first_instance {
                    Some(first) => Owner::Instance(first + index as u32),
                    None => {
                        self.frame_draws += 1;
                        Owner::Frame {
                            index: self.frame_draws,
                            ring: self.records % RING,
                        }
                    }
                };
                let fill = || {
                    let mut bytes = template.clone();
                    if let Some(member) = &members.object_to_world {
                        write_member(member, &instance.model, &mut bytes);
                    }
                    if let Some(member) = &members.world_to_object {
                        let inverse = invert(&instance.model).unwrap_or(instance.model);
                        write_member(member, &inverse, &mut bytes);
                    }
                    if let Some(member) = &members.transform_params {
                        let sign = handedness(&instance.model);
                        write_member(member, &[0.0, 0.0, 0.0, sign], &mut bytes);
                    }
                    bytes
                };
                let Some(at) = self
                    .draws
                    .slot(pass_index, owner, draw_size, self.alignment, fill)
                else {
                    skipped += 1;
                    continue;
                };
                if let Some(slot) = draw_slot {
                    offsets[slot] = at;
                }
            }
            // SAFETY: as above; the set was written before and the offsets lie inside the
            // buffers it names.
            unsafe {
                raw.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    layout,
                    0,
                    &[set],
                    &offsets,
                );
                raw.cmd_draw_indexed(cmd, section.index_count, 1, section.first_index, 0, 0);
            }
            draws += 1;
        }
        (draws, skipped)
    }
}

/// One section to draw with its Unity shader material, and the instances of its mesh.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SectionDraw<'a> {
    pub mesh: &'a GpuMesh,
    pub section: &'a GpuSection,
    pub instances: &'a [GpuInstance],
    /// Index of the first of `instances` among the scene's mesh instances: their per-object data
    /// is then written once and kept. `None` for a draw that changes every frame (the skybox).
    pub first_instance: Option<u32>,
    /// The selected scene object: drawn even when hidden.
    pub selected: Option<u32>,
    pub phase: Phase,
}

#[cfg(test)]
mod cull_tests {
    use super::*;

    /// A projection like the renderer's: reverse-Z, infinite far, 90 degrees, square, near 0.1,
    /// looking down +Z from the origin (view = identity).
    fn projection() -> [f32; 16] {
        let mut m = [0.0; 16];
        m[0] = 1.0;
        m[5] = 1.0;
        m[11] = 1.0; // w = z
        m[14] = 0.1; // z_clip = near
        m
    }

    const IDENTITY: [f32; 16] = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];

    fn at(x: f32, y: f32, z: f32) -> [f32; 16] {
        let mut m = IDENTITY;
        m[12] = x;
        m[13] = y;
        m[14] = z;
        m
    }

    const UNIT: [[f32; 3]; 2] = [[-0.5; 3], [0.5; 3]];

    #[test]
    fn boxes_in_front_are_kept_and_boxes_out_of_view_are_culled() {
        let vp = projection();
        assert!(!outside_frustum(&vp, &at(0.0, 0.0, 10.0), &UNIT), "ahead");
        assert!(
            !outside_frustum(&vp, &at(9.0, 0.0, 10.0), &UNIT),
            "near the edge"
        );
        assert!(
            outside_frustum(&vp, &at(30.0, 0.0, 10.0), &UNIT),
            "far to the right"
        );
        assert!(
            outside_frustum(&vp, &at(0.0, -30.0, 10.0), &UNIT),
            "far below"
        );
        assert!(
            outside_frustum(&vp, &at(0.0, 0.0, -10.0), &UNIT),
            "behind the camera"
        );
    }

    #[test]
    fn a_big_box_around_the_camera_is_kept() {
        let big = [[-1000.0; 3], [1000.0; 3]];
        assert!(!outside_frustum(&projection(), &IDENTITY, &big));
    }

    #[test]
    fn empty_bounds_are_never_culled() {
        let none = [[f32::INFINITY; 3], [f32::NEG_INFINITY; 3]];
        assert!(!outside_frustum(&projection(), &at(0.0, 0.0, -5.0), &none));
    }
}
