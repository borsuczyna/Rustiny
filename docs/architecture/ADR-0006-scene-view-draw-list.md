# ADR-0006: The first Scene view is CPU-projected and drawn with the ImGui draw list

**Status:** superseded by [ADR-0007](ADR-0007-gpu-scene-pass.md) as the main path; still the fallback when no GPU target exists.

## Context

The editor needs a Scene view that shows a loaded scene before there is any asset import (no meshes, textures or materials can be loaded yet), and the project has no shader toolchain (no shader compiler in the dependency tree, no SPIR-V in the repository).

## Decision

* Project object positions on the CPU (pinhole camera, left-handed Unity space, near-plane clipping) and draw markers, a grid and gizmos with the panel's `ImDrawList`. This needs no new Vulkan pipeline and works with the existing, validated frame loop.
* Keep per-frame vertex counts bounded: Dear ImGui's draw lists use 16-bit indices in this binding (`DrawIdx = u16`) and the Vulkan backend does not advertise vertex-offset support, so one list must stay under ~65 000 vertices. At most 10 000 four-vertex markers are drawn; if more are in view the nearest are kept (`select_nth_unstable`, O(n)) and the panel says so.
* Camera, projection and marker selection are plain functions with unit tests; the panel's interaction is tested with a headless Dear ImGui context and injected events.

## Consequences

* The view shows positions, not geometry. It must not be presented as rendering. Cost is 0.7 ms median per frame for 49 624 objects (release, one machine).
* Replacing it with a real renderer (instanced meshes, depth, materials) changes `draw.rs` and the data fed to it, not the camera, picking logic or panel interaction.
* A GPU pass will need a shader compilation step (glslc/naga decision) and an offscreen render target shown as an ImGui texture; that decision is deferred to when meshes can be imported.
