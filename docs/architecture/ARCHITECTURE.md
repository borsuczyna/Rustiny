# Rustiny architecture (current state: Milestones 0-1, slices of 3, 4 and 5)

## Crates and boundaries

| Crate | Responsibility | Depends on |
|---|---|---|
| `rustiny-core` | errors, config, logging, atomic fs, generational handles | no GPU / window code |
| `rustiny-profiler` | frame and scope timing with median/P95/P99 | nothing |
| `rustiny-vulkan` | instance + validation, device + allocator, swapchain, frame loop, GPU readback | `rustiny-core`, `ash`, `gpu-allocator` |
| `rustiny-assets` | Unity project detection, `.meta`/GUID reader, read-only asset index | nothing (std only) |
| `rustiny-scene` | UnityYAML reader, read-only scene model (GameObjects, transforms, hierarchy), `.mat` reader | nothing (std only) |
| `rustiny-model` | binary FBX reader and Unity-convention mesh import (flattened one mesh per file) | `flate2`, `thiserror` |
| `rustiny-shader` | Unity shaders without a GPU: ShaderLab parser, `UsePass` resolution, HLSL to SPIR-V through the external `dxc` (optional, run time), package-include mirror, SPIR-V reflection (ADR-0010) | `thiserror`, `tracing` |
| `rustiny-render` | offscreen scene pass: instanced meshes and boxes, albedo textures, grid, target, pipelines, GPU timing, WGSL to SPIR-V at build time | `rustiny-vulkan`, `ash`, `gpu-allocator`, `naga` (build) |
| `rustiny-editor` | `rustiny` binary: winit window, Dear ImGui shell, panels | all of the above |

Rules that hold today:

* The renderer layer (`rustiny-vulkan`) knows nothing about the editor or ImGui.
* `rustiny-core`, `rustiny-profiler`, `rustiny-assets` and `rustiny-scene` can be used headless; `rustiny-vulkan` and `rustiny-render` have headless GPU tests (`tests/headless.rs`, `tests/gpu.rs`).
* `rustiny-render` knows nothing about the scene model, FBX or ImGui: it takes meshes (vertices, indices, sections), RGBA textures with mip chains, instances (matrix + colour + object id) and a view-projection; the editor builds those from `rustiny-scene` and `rustiny-model`.
* Files are read, parsed and decoded on worker threads (`rustiny-editor/src/meshes`); the UI thread only takes finished results within a per-frame time budget and uploads them (see ADR-0008).
* Crates are added only when a real system exists (the planned `-scripting`, ... do not exist yet on purpose).

## Source layout and conventions

Same shape as the crates: small modules with one reason to change.

```
rustiny-editor/src
  main.rs, cli.rs, gui.rs, screenshot.rs
  app/        mod.rs (run, App)  handler.rs (winit events, redraw policy)  running.rs (window+GPU, frame)
              startup.rs (off-thread start-up, phase timing)  project_scan.rs (scan, watch, rescan)  scene_load.rs  scene_gpu.rs (scene renderer + ImGui texture)  smoke.rs  events/error/report.rs
  ui/         mod.rs (re-exports only)  editor/ (EditorUi, menu)  layout.rs  context.rs  project_state.rs  scene_state.rs  theme.rs
    panels/   console.rs  profiler.rs  project.rs  hierarchy.rs  inspector.rs  placeholder.rs  style.rs
              scene_view/ (mod.rs input+state, camera.rs, gpu.rs instances+box picking, points.rs, draw.rs)  headless_tests.rs (ImGui without a GPU)   (one file per panel: `draw` + `State`)
rustiny-render/
  build.rs (naga: shaders/*.wgsl to SPIR-V)  shaders/scene.wgsl (URP Lit-style surface, ADR-0009)  material.rs (storage buffer of per-section records)  texture.rs (3-slot sets)  tests/gpu.rs (offscreen render + pixel readback)
  src/ renderer.rs (SceneRenderer)  target.rs  pipeline.rs  buffer.rs  geometry.rs  mesh.rs  texture.rs  matrix.rs  timing.rs
rustiny-model/src
  fbx/ (binary FBX reader, conversion to Unity conventions, import settings from .meta)
rustiny-editor/src/meshes
  plan.rs (scene -> instances per model)  stream.rs (model workers)  materials.rs (.mat by name, shader properties)  textures.rs (PNG decode, shrink, mips)  convert.rs  pick.rs
rustiny-assets/src
  guid.rs  meta.rs  project.rs  watch.rs (file watcher + debounce)
  database/   mod.rs (AssetDatabase)  entry.rs  issue.rs  scanner.rs (walk + parallel .meta read)  tests.rs
rustiny-scene/src
  classes.rs  math.rs (Mat4)  model.rs (Scene, SceneObject)
  unity_yaml/ reader.rs (documents)  fields.rs (top-level fields)  scalar.rs (values)
  load/       mod.rs (stream)  collect.rs (fields -> raw records)  build.rs (hierarchy)  tests.rs
```

Conventions:

* A new panel is one file in `ui/panels/` exposing `draw(...)` and, if it keeps view state, `State`; it is wired up in `ui/editor/mod.rs`. Panels never touch Vulkan or the window.
* `mod.rs` holds the module's public face (types, re-exports, a short layout note), not the implementation.
* Soft limit: about 400 lines of non-test code per file. When a file grows past it, split by responsibility instead of adding sections. Large test suites live in `tests.rs` next to the module.
* Anything that could block (disk scans, loader work) runs on a worker thread and reports back through the event loop proxy; the UI thread never waits on it.
* Known files still over the limit: `rustiny-vulkan/src/renderer.rs` (~510 lines) and `device.rs` (~420); they are cohesive today and will be split when they next change substantially.

## Vulkan ownership model

Every wrapper owns an `Arc` to what it depends on (`Surface`/`Device` -> `Instance`, `Swapchain`/`Renderer` -> `Device`), so destruction order is correct by construction. The ImGui renderer backend holds clones of the device handle and allocator; `ImGuiLayer::shutdown` must run before the `Renderer` is dropped, and `Device::drop` logs an error if a stray allocator owner remains.

## Frame loop

1. `prepare_frame` (ImGui texture uploads) -> 2. `begin_frame` (wait this slot's fence, acquire, record layout barrier) -> 3. dynamic rendering + ImGui draw -> 4. `end_frame` (barrier, `vkQueueSubmit2`, present).

* 2 frames in flight; presentation waits on a render-finished semaphore **per swapchain image**.
* `vkDeviceWaitIdle` is used only for swapchain recreation (resize/out-of-date) and shutdown, never per frame.
* Out-of-date/suboptimal swapchains are rebuilt lazily at the next frame; zero-size (minimised) windows skip rendering.
* `ERROR_DEVICE_LOST` maps to `VulkanError::DeviceLost` and terminates the editor with a logged error. Rebuilding the device is future work.

## Scene view frame flow

`SceneGpu::prepare` (apply last frame's requested size; debounced) -> build the UI (the Scene panel shows the texture and publishes a `SceneViewRequest`: size, view-projection, selection) -> `SceneGpu::record` into the frame's command buffer, before the swapchain pass, with its own image barriers (colour ends in `SHADER_READ_ONLY_OPTIMAL`) -> ImGui samples it. Instance buffers and targets that are replaced are retired for a few frames instead of waiting for the GPU; only re-pointing the UI texture after a resize waits for the device to be idle.

## Idle behaviour

The editor draws on demand: input, resize or a background wake-up triggers a frame, followed by a one second 30 Hz settle period for hover/tooltip timing; afterwards the thread blocks in the OS event queue. Continuous mode (`--continuous`, config) is for profiling.

## Verification hooks

* `rustiny --smoke-frames N [--screenshot out.png]` renders N frames (resizing mid-run), prints a `SMOKE_RESULT` line and exits non-zero on validation errors; exit code 77 means "machine cannot run it".
* Validation messages are classified: `validation` type counts as an error; loader/environment messages (e.g. a broken third-party layer manifest) are logged and counted separately.
