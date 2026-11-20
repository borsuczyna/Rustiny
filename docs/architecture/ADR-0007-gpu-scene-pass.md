# ADR-0007: The Scene view is drawn by a GPU pass into an offscreen image

**Status:** accepted. Supersedes [ADR-0006](ADR-0006-scene-view-draw-list.md) as the main path. Extended by [ADR-0008](ADR-0008-streamed-models-and-textures.md) (meshes, materials, textures).

## Context

ADR-0006 deferred a GPU pass until a shader toolchain existed and meshes could be imported. The user asked for GPU rendering now. Meshes still cannot be imported, so the pass has to draw stand-ins.

## Decision

* New crate `rustiny-render`. It owns an offscreen colour (`R8G8B8A8_UNORM`) and depth (`D32_SFLOAT`) target, two pipelines (instanced box, ground grid), a unit-box vertex/index buffer and the per-instance buffer. It records into the caller's command buffer with its own barriers and knows nothing about ImGui or the scene model.
* **Shaders are WGSL translated to SPIR-V by `naga` in `build.rs`.** No Vulkan SDK or external compiler is needed to build; a shader error fails the build with naga's message. naga's default "adjust coordinate space" (a Y flip for WebGPU) is switched off because the view-projection already produces Vulkan clip space. naga 30 calls push constants "immediates" (`var<immediate>`).
* **Reverse-Z with an infinite far plane**: depth 1 at the near plane, 0 at infinity, compare `GREATER`, clear 0. The matrix is `rustiny_render::matrix::view_projection`; a unit test in the editor checks it lands on the same pixel as the CPU pinhole projection used for picking and overlays.
* **Every scene object is one instance of a unit box** (`GpuInstance`: world matrix + colour, 80 bytes; the instance index equals the object id). Boxes are placeholders for meshes. Culling is off because instances may be mirrored.
* The editor shows the target as an ImGui external texture. The panel only asks (`SceneViewRequest` through `UiRequests`); `app/scene_gpu.rs` sequences it: apply the previous frame's size, build the UI, record the pass before the swapchain pass.
* Replaced instance buffers and targets are retired for `FRAMES_IN_FLIGHT + 1` records instead of waiting; re-pointing the ImGui texture after a resize waits for the device (the backend's `update_external_texture` does), so resizes are debounced by 3 frames.
* Picking stays on the CPU: ray vs oriented box over all active objects on click (no matrix inverse), then the nearest marker for tiny objects. GPU picking (an object-id attachment) is later work.
* The CPU marker drawing of ADR-0006 stays as the fallback when no GPU target exists (headless tests, renderer creation failure).
* GPU time of the pass is measured with timestamp queries (one begin/end pair per frame slot, read back when the slot is reused, so nothing waits) and shown in the Profiler panel and the smoke report.

## Consequences

* Real transforms (including rotation and scale) are visible, with depth and lighting, but the view does not show real models. It must not be described as rendering the game's art.
* Instances sit in host-visible memory and are replaced as a whole when a scene loads (about 4 MB for 50 000 objects); a device-local copy and partial updates are later optimisations.
* The next steps toward real rendering are mesh and material import, a depth-aware selection outline, culling, and shadows.
