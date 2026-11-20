# ADR-0012: URP frame order, camera textures, HDR, skybox and scene lighting

Status: accepted (limits below)

## Context

The river of the real project rendered almost black and flickered. Its shader
(`SummerHideout/Water`) is URP-style: it reads `_CameraDepthTexture` and `_CameraOpaqueTexture`,
reflects the environment probe, depends on `_GlobalDaylightVisibility` set by a script, and uses
`SV_IsFrontFace`. None of that was provided: the camera textures were one-texel stand-ins, the
globals were zero, and every mesh was drawn with the wrong front face, so the shader took its
"seen from below the water" branch (no foam, no duckweed, a flickering mirror). After that was
fixed, the ground turned orange: Unity shaders write linear light, and it was shown without
conversion to sRGB.

## Decisions

* **Frame order as in URP**: opaque draws, then the skybox, then a copy of colour and depth, then
  the transparent draws (`unity::Phase`). A pass is transparent when it blends. The copies are
  bound to `_CameraOpaqueTexture` and `_CameraDepthTexture` (matched by variable name; depth is
  sampled with a point/clamp sampler). Copies happen only when some pass blends or reads them.
* **HDR target**: the scene renders into `R16G16B16A16_SFLOAT` (linear light), the final pass
  (`present.wgsl`) writes the display image with the exact sRGB curve. The native WGSL shader
  outputs linear light too; overlay colours (grid, selection) are decoded to linear.
  Blending therefore happens in linear light, as in URP.
* **Front faces**: Unity's front faces are clockwise; Unity passes now use
  `FrontFace::CLOCKWISE`, so `SV_IsFrontFace` is right. Culling still draws both sides.
* **Depth parameters**: the projection is reverse-Z with an infinite far plane, so
  `_ZBufferParams = (far/near, 0, 1/near, 0)` makes `LinearEyeDepth` exact;
  `unity_OrthoParams.w = 0` (it was 1, which told shaders the camera was orthographic and made
  `GetWorldSpaceViewDir` a constant).
* **Scene lighting from the scene file**: `RenderSettings` (class 104) and `Light` components
  (class 108) are read (`rustiny_scene::lighting`). The main light is `m_Sun` (else the first
  enabled directional light); its direction is minus its forward axis, its colour linear times
  intensity. The trilight ambient becomes `unity_SHA*/SHB*/SHC` that give exactly the sky colour
  up, the equator colour sideways and the ground colour down (`Ambient::spherical_harmonics`,
  tested against URP's `SampleSH`). `m_ReflectionIntensity` scales `unity_SpecCube0_HDR`.
* **Skybox**: `RenderSettings.m_SkyboxMaterial` is compiled by the shader hub like any project
  material (on a worker thread) and drawn on a sphere around the camera after the opaque objects.
* **Project globals**: values the project's scripts set with `Shader.SetGlobal*`
  (`_GlobalDaylightVisibility`, wind, water multipliers) are supplied as the authored scene would
  have them (`shaders::globals`); a material's own value wins over them.
* **Shader include mirror heals itself**: the copy of package includes is checked file by file
  (size) on every start; a cleaned temporary folder had left marker files without their
  contents, and every new shader variant failed to compile.

* **Per-object data kept on the GPU** (`unity::draws::DrawCache`), like the SRP Batcher: each
  object's `UnityPerDraw` block is written once, keyed by (pass, instance), and reused every
  frame; the buffer doubles when full (old one retired with the frame stamp, sets rebuilt). The
  48 MB per-frame ring had overflowed: 40 170 of 105 000 draws were dropped in a forest view,
  which made the skybox, the terrain and objects vanish depending on the angle. The skybox,
  which follows the camera, uses a ring of per-frame slots.
* **Screen to world as URP does it on Vulkan**: `ComputeClipSpacePosition` negates y
  (`UNITY_UV_STARTS_AT_TOP`) because Unity's targets are stored upside down; ours are not, so
  `unity_MatrixInvVP`/`unity_MatrixInvP` include that flip. Without it the water found its floor
  through a mirrored ray: wrong depth, no transparency, no shore foam.
* **Scene camera** flies at 10 m/s by default; the wheel while flying changes the speed (as in
  Unity), Shift is four times faster.

## Verified

GPU tests (validation on): a transparent pass reads the colour and the depth drawn before it; URP's own `ComputeScreenPos`/`ComputeWorldSpacePosition` find the world position of what is behind (it was mirrored in y before the fix); 120 000 instances are all drawn after the per-object buffer grows, and moved instances are drawn at their new place;
blending and the Unity shader test values recomputed for linear light (e.g. half green over the
clear colour = sRGB (16, 187, 54), measured exactly). Unit tests: `RenderSettings`/`Light`
parsing, trilight SH, sun from the scene, gamma curve, daylight function, mirror healing.
Real scene: skybox compiled and drawn, water drawn from above with ripples, duckweed and sky
reflection, no validation errors. Judged by screenshot, not compared with Unity.

## Limits (not hidden)

* No reflection probe yet: `unity_SpecCube0` is still a black stand-in (the skybox is not
  rendered into a cubemap), so reflections fall back to the ambient SH.
* No shadows (no `ShadowCaster` passes, no shadow maps).
* No post-processing (Volume profiles: tonemapping, colour adjustments); values above 1 are
  clipped. The sun's colour temperature (`m_UseColorTemperature`) is not applied (logged).
* The ambient for `m_AmbientMode` = Skybox uses the stored gradient colours, not a probe computed
  from the sky. Lights inside prefab instances are not read.
* Project globals are the authored scene's values, not computed by its scripts each frame
  (time of day is fixed at the scene's `timeOfDay`).
* Meshes still lack tangents, vertex colours and UV1 on the GPU (the water's normal detail uses
  its tangent frame).
* Transparent draws are not sorted back to front.
* The scene's Volume profile (Neutral tonemapping, bloom, vignette, colour adjustments) is not
  applied yet; the ground looks warmer than it would after URP's tonemapping.
