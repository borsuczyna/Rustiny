//! GPU scene renderer: an offscreen viewport target and an instanced scene pass.
//!
//! Milestone 5: scene objects are drawn as instanced unit boxes or, when the editor has imported
//! their model, as instanced meshes with per-section tints, over a procedural ground grid, with
//! depth testing and one directional light. The result lands in an offscreen colour image the
//! editor shows inside a panel.
//!
//! * `renderer` - [`SceneRenderer`]: owns the pipelines, buffers and target, records the pass.
//! * `target`   - the offscreen colour + depth images.
//! * `buffer`   - host-visible buffers.
//! * `timing`   - GPU time of the pass (timestamp queries).
//! * `geometry` - the unit box and the instance layout.
//! * `mesh`     - mesh vertices, sections, instance groups and their GPU buffers.
//! * `texture`  - material textures: mip chains, descriptor sets, uploads in the frame's commands.
//! * `material` - the storage buffer of per-section material records.
//! * `matrix`   - view-projection for the shaders' conventions.
//! * `unity`    - drawing with a Unity project's own shaders (compiled SPIR-V, constant buffers filled by name).
//!
//! Shaders are WGSL in `shaders/`, translated to SPIR-V by `build.rs` with `naga`.
//!
//! Synchronisation: the pass is recorded into the caller's command buffer with its own image
//! barriers, so it can sit in front of the swapchain pass of the same frame. A replaced instance
//! buffer or target is kept for a few records until the frames that may use it have finished;
//! everything is destroyed after `wait_idle` when the renderer is dropped.

mod buffer;
mod geometry;
mod material;
pub mod matrix;
mod mesh;
mod pipeline;
mod post;
mod present;
mod renderer;
mod target;
mod texture;
mod timing;
mod unity;

pub use geometry::GpuInstance;
pub use mesh::{MeshGroup, MeshSection, MeshStats, MeshVertex, Surface};
pub use post::{Bloom, PostProcess, Tonemap, Vignette};
pub use renderer::{COLOR_FORMAT, DEPTH_FORMAT, FrameParams, HDR_FORMAT, SceneRenderer};
pub use texture::mip_chain;
pub use unity::{
    Ambient, Blend, BlendFactor, CullMode, DepthTest, PassState, PropertyValue, ShaderMaterialDesc,
    ShaderPassDesc, TextureFallback, UnityFrame,
};
