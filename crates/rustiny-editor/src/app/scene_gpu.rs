//! The GPU Scene view on the application side: owns the scene renderer and the ImGui texture that
//! shows its image, and sequences the two across frames.
//!
//! Per frame: [`prepare`](SceneGpu::prepare) (before the UI is built) applies the size the panel
//! asked for last frame, then the UI is built with the texture, the panel's new request is stored
//! with [`set_request`](SceneGpu::set_request), and [`record`](SceneGpu::record) renders the pass
//! into the frame's command buffer ahead of the swapchain pass that samples it.

use crate::meshes::{DecodedTexture, MeshPlan, Upload};
use crate::shaders::{SceneLightValues, ShaderHub};
use crate::ui::SceneViewRequest;
use dear_imgui_ash::AshRenderer;
use dear_imgui_rs::TextureId;
use rustiny_render::{FrameParams, GpuInstance, MeshStats, SceneRenderer, UnityFrame};
use rustiny_vulkan::ash::vk;
use rustiny_vulkan::{Device, Result};
use std::sync::Arc;

/// Background of the viewport (matches the CPU fallback's).
const BACKGROUND: [f32; 4] = [0.09, 0.095, 0.105, 1.0];
/// A new viewport size must be requested this many frames in a row before the target is rebuilt,
/// so dragging a panel edge does not recreate it (and wait for the GPU) on every frame.
const RESIZE_SETTLE_FRAMES: u32 = 3;

/// The scene renderer plus the texture the UI draws it with.
#[derive(Debug)]
pub(super) struct SceneGpu {
    renderer: SceneRenderer,
    texture: Option<TextureId>,
    request: Option<SceneViewRequest>,
    /// A size the panel keeps asking for, and for how many frames in a row.
    pending_size: Option<([u32; 2], u32)>,
    passes: u64,
    /// How much of the shader hub's registry has been sent to the renderer, and the renderer's id
    /// for each hub pass and material that made it (`None` for those that failed).
    shader_passes_sent: usize,
    shader_materials_sent: usize,
    gpu_passes: Vec<Option<u32>>,
    gpu_materials: Vec<Option<u32>>,
    started: std::time::Instant,
    /// The open scene's main light and ambient (`None`: the renderer's defaults).
    lighting: Option<SceneLightValues>,
}

impl SceneGpu {
    pub(super) fn new(device: &Arc<Device>) -> Result<Self> {
        let mut renderer = SceneRenderer::new(device)?;
        renderer.set_shader_globals(crate::shaders::project_globals(
            crate::shaders::AUTHORED_TIME_OF_DAY,
        ));
        Ok(Self {
            renderer,
            texture: None,
            request: None,
            pending_size: None,
            passes: 0,
            shader_passes_sent: 0,
            shader_materials_sent: 0,
            gpu_passes: Vec::new(),
            gpu_materials: Vec::new(),
            started: std::time::Instant::now(),
            lighting: None,
        })
    }

    /// The texture showing the scene, once a target exists.
    pub(super) fn texture(&self) -> Option<TextureId> {
        self.texture
    }

    /// GPU time of a recent scene pass in milliseconds.
    pub(super) fn gpu_time_ms(&self) -> Option<f32> {
        self.renderer.gpu_time_ms()
    }

    /// How many scene passes have been recorded.
    pub(super) fn passes(&self) -> u64 {
        self.passes
    }

    /// Replace the proxy boxes. Failures are logged; the view then shows what it had.
    pub(super) fn set_instances(&mut self, instances: &[GpuInstance]) {
        if let Err(e) = self.renderer.set_instances(instances) {
            tracing::warn!("cannot upload the scene to the GPU: {e}");
        }
    }

    /// Show a new plan: its boxes and mesh instances replace the old ones and every uploaded mesh
    /// is dropped (the models of the new scene stream in afterwards).
    /// The post-processing of the open scene.
    pub(super) fn set_post_process(&mut self, post: rustiny_render::PostProcess) {
        self.renderer.set_post_process(post);
    }

    /// The lighting of the open scene.
    pub(super) fn set_lighting(&mut self, lighting: Option<SceneLightValues>) {
        self.lighting = lighting;
    }

    pub(super) fn set_plan(&mut self, plan: &MeshPlan) {
        self.renderer.clear_meshes();
        self.renderer.clear_textures();
        self.renderer.clear_shaders();
        self.shader_passes_sent = 0;
        self.shader_materials_sent = 0;
        self.gpu_passes.clear();
        self.gpu_materials.clear();
        if let Err(e) = self.renderer.set_skybox(None) {
            tracing::warn!("cannot remove the skybox: {e}");
        }
        self.set_instances(&plan.boxes);
        if let Err(e) = self.renderer.set_mesh_instances(&plan.groups) {
            tracing::warn!("cannot upload the mesh instances to the GPU: {e}");
        }
    }

    /// Send the passes and materials the hub has registered since the last call to the renderer,
    /// then rewrite the shader ids in `upload` (the hub's) to the renderer's. A section whose
    /// shader material could not be added keeps the native look.
    pub(super) fn use_shaders(&mut self, hub: &ShaderHub, upload: &mut Upload) {
        let (passes, materials) =
            hub.take_since(self.shader_passes_sent, self.shader_materials_sent);
        for pass in &passes {
            let id = match self.renderer.add_shader_pass(pass) {
                Ok(id) => Some(id),
                Err(e) => {
                    tracing::warn!("a shader pass cannot be used: {e}");
                    None
                }
            };
            self.gpu_passes.push(id);
        }
        self.shader_passes_sent += passes.len();
        for mut material in materials {
            let id = self
                .gpu_passes
                .get(material.pass as usize)
                .copied()
                .flatten()
                .and_then(|pass| {
                    material.pass = pass;
                    match self.renderer.add_shader_material(&material) {
                        Ok(id) => Some(id),
                        Err(e) => {
                            tracing::warn!("a shader material cannot be used: {e}");
                            None
                        }
                    }
                });
            self.gpu_materials.push(id);
        }
        self.shader_materials_sent += self.gpu_materials.len() - self.shader_materials_sent;
        for section in &mut upload.sections {
            section.surface.shader = section
                .surface
                .shader
                .and_then(|hub_id| self.gpu_materials.get(hub_id as usize).copied().flatten());
        }
    }

    /// Draw the background with the hub's material `hub_id` (the scene's skybox).
    pub(super) fn set_skybox(&mut self, hub: &ShaderHub, hub_id: u32) {
        let mut nothing = Upload::default();
        self.use_shaders(hub, &mut nothing);
        let id = self.gpu_materials.get(hub_id as usize).copied().flatten();
        match id {
            Some(id) => match self.renderer.set_skybox(Some(id)) {
                Ok(()) => tracing::info!("skybox drawn with its material"),
                Err(e) => tracing::warn!("cannot draw the skybox: {e}"),
            },
            None => tracing::warn!("the skybox material did not reach the GPU"),
        }
    }

    /// Upload one model under `key`.
    pub(super) fn add_mesh(&mut self, key: u32, upload: &Upload) -> Result<()> {
        self.renderer
            .add_mesh(key, &upload.vertices, &upload.indices, &upload.sections)
    }

    /// Upload one texture under `key` (its copy to the GPU is recorded in the next passes).
    pub(super) fn add_texture(
        &mut self,
        key: u32,
        texture: &DecodedTexture,
        srgb: bool,
    ) -> Result<()> {
        self.renderer
            .add_texture(key, texture.width, texture.height, &texture.mips, srgb)
    }

    /// Textures added but not yet copied to the GPU: frames are still needed while this is not 0.
    pub(super) fn textures_waiting(&self) -> usize {
        self.renderer.textures_waiting()
    }

    /// What the renderer holds and drew in its last pass.
    pub(super) fn mesh_stats(&self) -> MeshStats {
        self.renderer.mesh_stats()
    }

    /// Remember what the panel wants drawn this frame (`None`: it is not showing).
    pub(super) fn set_request(&mut self, request: Option<SceneViewRequest>) {
        self.request = request;
    }

    /// Apply the viewport size the panel asked for in the previous frame. Returns `true` while
    /// more frames are needed to finish a pending resize.
    pub(super) fn prepare(&mut self, ui_renderer: &mut AshRenderer) -> Result<bool> {
        let Some(request) = self.request else {
            return Ok(false);
        };
        let wanted = request.size_px;
        let current = self.renderer.extent().map(|e| [e.width, e.height]);
        if current == Some(wanted) {
            self.pending_size = None;
            return Ok(false);
        }
        let frames = match self.pending_size {
            Some((size, n)) if size == wanted => n + 1,
            _ => 1,
        };
        self.pending_size = Some((wanted, frames));
        // The very first target is made at once; later ones wait for the size to settle.
        if current.is_some() && frames < RESIZE_SETTLE_FRAMES {
            return Ok(true);
        }
        self.pending_size = None;
        if !self.renderer.resize(wanted[0], wanted[1])? {
            return Ok(false);
        }
        let Some(view) = self.renderer.color_view() else {
            return Ok(false);
        };
        let layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        // `view` is a live sampled view of the scene renderer's new colour image on the device the
        // UI renderer uses. It stays alive while the renderer owns the target, the target is only
        // replaced through this method (which re-points the texture first), and the scene pass
        // leaves the image in `layout` before the UI samples it.
        match self.texture {
            Some(id) => {
                // SAFETY: see above. The call waits for the device to be idle, so no submitted
                // frame still uses the old view.
                unsafe { ui_renderer.update_external_texture(id, view, layout) }
                    .map_err(|e| ui_error("update_external_texture", &e))?;
            }
            None => {
                // SAFETY: see above; the texture is unregistered in `release` before the UI
                // renderer and the scene renderer are torn down.
                let id = unsafe { ui_renderer.register_external_texture(view, layout) }
                    .map_err(|e| ui_error("register_external_texture", &e))?;
                self.texture = Some(id);
            }
        }
        Ok(true)
    }

    /// Record the scene pass into `cmd` (recording, outside any rendering scope). Returns whether
    /// anything was recorded.
    pub(super) fn record(&mut self, cmd: vk::CommandBuffer) -> bool {
        let Some(request) = &self.request else {
            return false;
        };
        let params = FrameParams {
            view_proj: request.view_proj,
            eye: request.eye,
            selected: request.selected,
            clear: BACKGROUND,
            grid_extent: request.grid_extent,
            grid_cell: request.grid_cell,
        };
        self.renderer.set_unity_frame(UnityFrame {
            view: request.view,
            projection: request.projection,
            eye: request.eye,
            width: request.size_px[0],
            height: request.size_px[1],
            time: self.started.elapsed().as_secs_f32(),
            ..self
                .lighting
                .map_or_else(UnityFrame::default, |l| UnityFrame {
                    light_direction: l.light_direction,
                    light_color: l.light_color,
                    ambient: l.ambient,
                    reflection_intensity: l.reflection_intensity,
                    ..UnityFrame::default()
                })
        });
        let recorded = self.renderer.record(cmd, &params);
        self.passes += u64::from(recorded);
        recorded
    }

    /// Unregister the texture from the UI renderer. Call before the UI renderer shuts down.
    pub(super) fn release(&mut self, ui_renderer: &mut AshRenderer) {
        if let Some(id) = self.texture.take() {
            // SAFETY: the texture is no longer shown (the caller is shutting the view down) and
            // `unregister_texture` waits for submitted work that may still sample it.
            if let Err(e) = unsafe { ui_renderer.unregister_texture(id) } {
                tracing::warn!("cannot unregister the scene texture: {e}");
            }
        }
    }
}

fn ui_error(call: &'static str, error: &dyn std::fmt::Display) -> rustiny_vulkan::VulkanError {
    rustiny_vulkan::VulkanError::WindowSystem(format!("UI renderer `{call}` failed: {error}"))
}
