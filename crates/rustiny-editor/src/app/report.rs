//! Facts gathered over a run; printed by smoke mode and asserted by tests.

/// Facts gathered over a run.
#[derive(Debug, Clone, Default)]
pub struct RunReport {
    /// Frames actually presented.
    pub frames: u64,
    /// Validation errors (API misuse) reported by the Vulkan layer.
    pub validation_errors: u64,
    /// Validation warnings.
    pub validation_warnings: u64,
    /// Loader/environment errors that are not caused by this application.
    pub environment_errors: u64,
    /// Times the swapchain was recreated.
    pub swapchain_recreations: u64,
    /// GPU used.
    pub gpu: String,
    /// `(files, folders, issues)` of the opened project, once its background scan finished.
    pub project: Option<(usize, usize, usize)>,
    /// `(objects, roots, issues)` of the opened scene, once its background load finished.
    pub scene: Option<(usize, usize, usize)>,
    /// Number of project scans that finished (the first one plus every refresh).
    pub project_scans: u32,
    /// Scene passes rendered on the GPU for the Scene view.
    pub scene_gpu_passes: u64,
    /// GPU time of the last measured scene pass, in milliseconds.
    pub scene_gpu_ms: Option<f32>,
    /// Models of the opened scene: what was planned, uploaded and failed, and what the last scene
    /// pass drew.
    pub meshes: Option<MeshReport>,
}

/// How the scene's models fared.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MeshReport {
    /// Distinct model files the scene uses.
    pub planned: usize,
    /// Uploaded to the GPU.
    pub uploaded: usize,
    /// Could not be imported or uploaded (shown as boxes).
    pub failed: usize,
    /// Triangles submitted by the last scene pass (instances times triangles).
    pub triangles_drawn: u64,
    /// Draw calls of the last scene pass.
    pub draw_calls: u32,
    /// Material slots of the imported models.
    pub material_slots: usize,
    /// Slots for which a `.mat` of the same name was found.
    pub materials_matched: usize,
    /// Textures the materials asked for.
    pub textures_requested: usize,
    /// Textures uploaded to the GPU.
    pub textures_uploaded: usize,
    /// Textures that could not be decoded or uploaded.
    pub textures_failed: usize,
    /// Bytes of texture memory on the GPU.
    pub texture_bytes: u64,
}
