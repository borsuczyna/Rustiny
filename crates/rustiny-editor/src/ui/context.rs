//! Data exchanged between the application shell and the UI each frame.

use dear_imgui_rs::TextureId;
use rustiny_core::config::Theme;
use rustiny_core::logging::LogBuffer;
use rustiny_profiler::FrameProfiler;

/// Read-only data the UI needs each frame.
#[derive(Debug)]
pub struct UiContext<'a> {
    /// Captured log records.
    pub log: &'a LogBuffer,
    /// Frame timing.
    pub profiler: &'a FrameProfiler,
    /// Static facts about the renderer.
    pub gpu: &'a GpuSummary,
    /// Validation counters.
    pub validation_errors: u64,
    /// Validation warnings.
    pub validation_warnings: u64,
    /// Whether the editor redraws continuously.
    pub continuous_redraw: bool,
    /// The GPU-rendered Scene view, once the application has one to show.
    pub scene_texture: Option<TextureId>,
}

/// Facts about the GPU/swapchain shown in the Profiler panel.
#[derive(Debug, Clone, Default)]
pub struct GpuSummary {
    /// GPU name.
    pub name: String,
    /// Driver/API description.
    pub api: String,
    /// Present mode in use.
    pub present_mode: String,
    /// Swapchain size in pixels.
    pub extent: (u32, u32),
    /// Number of swapchain images.
    pub image_count: usize,
    /// Times the swapchain has been recreated.
    pub swapchain_generation: u64,
    /// Whether validation layers are active.
    pub validation_active: bool,
    /// GPU time of a recent Scene view pass in milliseconds (timestamp queries).
    pub scene_gpu_ms: Option<f32>,
}

/// What the Scene panel wants the GPU scene pass to draw this frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneViewRequest {
    /// Size of the viewport in physical pixels (the target is resized to this).
    pub size_px: [u32; 2],
    /// Column-major view-projection of the panel's camera.
    pub view_proj: [f32; 16],
    /// World-to-view matrix in Unity's convention, for the project's own shaders.
    pub view: [f32; 16],
    /// View-to-clip matrix that goes with `view`.
    pub projection: [f32; 16],
    /// Camera position in world space.
    pub eye: [f32; 3],
    /// Index of the selected object, which is drawn highlighted.
    pub selected: Option<u32>,
    /// Size of one ground-grid cell in world units.
    pub grid_cell: f32,
    /// Half extent of the ground grid in world units.
    pub grid_extent: f32,
}

/// Actions the UI asks the application to perform.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct UiRequests {
    /// Quit the editor.
    pub quit: bool,
    /// Toggle continuous redraw.
    pub toggle_continuous: bool,
    /// Change the theme.
    pub theme: Option<Theme>,
    /// Open this scene, given as a path relative to the open project's root.
    pub open_scene: Option<String>,
    /// Draw the Scene view with the GPU pass this frame (absent when the panel is not showing).
    pub scene_view: Option<SceneViewRequest>,
}
