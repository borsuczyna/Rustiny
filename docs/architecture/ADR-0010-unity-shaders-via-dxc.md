# ADR-0010: Unity shaders are compiled from their own HLSL with DXC

**Status:** accepted; compilation (`crates/rustiny-shader`) and drawing (`crates/rustiny-render/src/unity`, `crates/rustiny-editor/src/shaders`) implemented for HLSL passes with default/material keywords. Amends [ADR-0007](ADR-0007-gpu-scene-pass.md) and [ADR-0009](ADR-0009-native-urp-lit-surface.md) on one point: there is now an optional run-time shader compiler.

## Context

ADR-0009 gave URP Lit a native implementation, but a project's own shaders (terrain layer mixing, trees, foliage, water, sky, atmosphere, ...) cannot be reproduced by hand one by one, and the user wants Unity shaders themselves to work. Those shaders are ShaderLab files whose passes are HLSL that `#include`s URP's shader library (`Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl`, ...). `naga` has no HLSL front end.

Measured on the real project (read-only): 36 `.shader` files; 31 use `HLSLPROGRAM`, 4 use `CGPROGRAM` (built-in pipeline, `UnityCG.cginc`); 36 include URP's `Core.hlsl`; URP/Core/Config packages are present in `Library/PackageCache`; the Vulkan SDK on the machine ships `dxc` and `slangc`.

## Decision

* **DXC, as a run-time tool, not a build dependency.** `rustiny-shader` runs `dxc -spirv` (found through `RUSTINY_DXC`, `VULKAN_SDK` or `PATH`) on a Unity pass's HLSL. No FFI and no `unsafe`: a child process, with SPIR-V cached by a hash of the source, entry point, keywords and recipe. Without `dxc` the editor keeps the native URP Lit path of ADR-0009; the engine's own shaders still go through `naga` at build time.
* **Packages are mirrored, never touched.** `#include "Packages/<name>/..."` is resolved by copying the HLSL files of `Library/PackageCache/<name>@<hash>` and embedded packages into a directory of ours (`Packages/<name>/...`, 59 packages, 427 files for this project) and passing it as an include root. Project-relative includes use the shader's folder. Nothing is written in the project.
* **Unity's dialect, not DXC's default:** `-HV 2018` (the 2021 rules reject URP code such as short-circuit `?:` on vectors), `SHADER_API_VULKAN`, `UNITY_VERSION`, `SHADER_STAGE_*`, one `-D` per enabled keyword, and `#include_with_pragmas` rewritten to `#include`.
* **ShaderLab is parsed by us** (`shaderlab.rs`): properties with attributes and defaults, sub-shaders and tags, passes, render state (`Cull`/`ZWrite`/`ZTest`/`Blend`/`ColorMask`/`AlphaToMask`, each possibly `[_Property]`), `UsePass` (resolved across files by `ShaderLibrary`), `HLSLINCLUDE`, and each program's `#pragma vertex/fragment/target/multi_compile/shader_feature`. Unknown constructs (`Stencil`, `GrabPass`, fixed function) are skipped, not fatal.
* **Reflection reads the SPIR-V** (`reflect.rs`, no dependency): vertex inputs with their semantics, constant buffers with member names/offsets/sizes, textures, samplers, structured buffers. That is what lets the renderer fill `UnityPerMaterial`, `UnityPerDraw` and `$Globals` *by name*.

## Result so far (all measured, `cargo run --release -p rustiny-shader --example compile_project -- <project> <work-dir> --reflect`)

* 36 of 36 shader files parse.
* 71 of 71 passes that have both a vertex and a fragment entry (HLSL, default keywords: first option of each `multi_compile`, `shader_feature` off) compile to SPIR-V and reflect: 10 s cold on 8 threads, instant from the cache. Not attempted: 4 `CGPROGRAM` passes, 6 passes without a vertex+fragment pragma pair, other keyword variants, compute shaders.
* What the shaders ask the engine for: vertex inputs POSITION, NORMAL, TANGENT, COLOR, TEXCOORD0-2; `UnityPerDraw` (34 members: `unity_ObjectToWorld`, `unity_WorldToObject`, SH coefficients, probe data, ...); `UnityPerMaterial` (275 distinct members over the project, filled from the `.mat`); `$Globals` (209 distinct members: Unity's camera/light/fog values plus the project's own `_Atmosphere*`/`_Global*` values set by its scripts); `AdditionalLights`; 15 kinds of global textures (`_CameraDepthTexture`, `_CameraOpaqueTexture`, `unity_SpecCube0`, ...); 3 structured buffers.

## Drawing (implemented)

* The renderer reflects both stages of a pass, merges the descriptors **by name** into one set 0, assigns bindings by sorted name and patches the SPIR-V (`rebind`). `UnityPerDraw` and other blocks are dynamic uniform buffers from a per-frame arena (3 regions of 12 MB); `UnityPerMaterial` lives in a 16 MB material buffer; textures and 4 shared samplers (linear/point x repeat/clamp, chosen by name) are bound, storage buffers get a zero buffer. Constant buffers are filled by member name: camera matrices and position, `_Time`, main light, a flat ambient, handedness, per-draw `unity_ObjectToWorld`/`unity_WorldToObject`.
* Vertex inputs come from the reflected semantics: POSITION/NORMAL/TEXCOORD0 from the mesh, TANGENT (1,0,0,1), COLOR (1,1,1,1) and the rest zero from a stride-0 binding. One draw per visible instance. Depth test is mapped for reverse-Z; blend and colour mask come from the pass state; both faces are drawn.
* Editor `ShaderHub`: on a worker it locates dxc, mirrors the packages, parses the project's shaders and compiles each (pass, keywords) variant once; pass choice UniversalForward > UniversalForwardOnly > SRPDefaultUnlit > untagged; keywords from the material (enabled `shader_feature`s, `Toggle`/`KeywordEnum` properties) else the first `multi_compile` option; colours go gamma->linear, PNG textures only. Materials whose shader is missing, not HLSL or rejected by dxc keep the native look.
* Measured on the real project (release), after fixing two bugs found by looking at the result (a ShaderLab parser bug that silently dropped the program of every pass with `AlphaToMask[_Prop]`, which hid LitNormalFade's main pass, and `unity_LightData.z`, which URP reads as the main light's attenuation and was 0, so nothing got direct light): 36 shaders parsed, 15 pass variants compiled, 0 rejected by dxc, 363 materials drawn with them (14127 draws per frame, 0 left out), 0 validation errors, scene pass about 7 ms. The earlier "71 of 71 passes compile" figure counted only passes the parser had kept a program for; with the fix it is 73 HLSL passes. Headless GPU tests cover matrix layout, frame values, blending, hidden instances and the fallback.

### Known limits of the drawing stage

* **The terrain is not drawn with its shader at all**: the tiles (`Terrain_600_xx_yy`) are prefab instances whose mesh is a serialized Unity `Mesh` (`.asset`, 394 of them in the project) and whose material is an override of the prefab (`m_Materials.Array.data[0]`, `TerrainPaint.mat` with the `MeshTerrain` shader); the blend weights (`_Control0`) are a `Texture2D` stored as hex in a `.asset`. None of these is read yet: `.prefab` expansion with property overrides, `.asset` meshes, `MeshFilter`/`MeshRenderer` values and YAML textures are the missing chain, so terrain tiles remain proxy boxes.

* **Procedural instancing and storage buffers are not supported**: shaders such as the project's `Foliage` read per-instance data from a storage buffer that is bound to zeros, so they render black (seen on the real scene). Nothing here evaluates `#pragma instancing_options procedural:`.
* No shadows, depth/opaque textures, reflection probes or lightmaps (stand-in textures); flat ambient instead of probes; project globals (`_Atmosphere*`, `_Global*`) are 0.
* Face winding/culling not verified (both faces drawn); no transparency sorting; no selection highlight on shader sections; at most ~12 MB of per-draw constants per frame.
* The look was **not compared with Unity**; `CGPROGRAM` and Shader Graph are unsupported; dxc is required.

## Not done (original stage list, for reference)

1. **Drawing** (done, see above): a renderer path for "shader" sections: pipeline from the pass's SPIR-V and ShaderLab state, descriptor layout from the reflection, `UnityPerDraw` per draw (our instances are drawn one by one for these), `UnityPerMaterial` from the `.mat`, `$Globals` from per-frame values (camera matrices, time, main light, fog; project globals default to 0), default textures for everything the engine does not provide (depth, opaque, shadow maps, probes), vertex inputs the mesh lacks (tangent, colour, UV1/2) fed from a constant buffer.
2. **Keywords:** (done for the material's own keywords; arbitrary runtime keyword changes are not modelled.)
3. **`CGPROGRAM`** shaders: needs the Editor's `CGIncludes` and a `sampler2D` shim, since DXC has no legacy texture sampling.
4. Shadows, depth/opaque textures, reflection probes, transparency sorting; compute shaders; Shader Graph.

## Consequences

* Shader coverage becomes data (a report over the project) instead of a guess; the first stage is verified by unit tests (parser, `UsePass`, mirror, a real DXC compile with caching, reflection of a known shader) and by compiling every pass of a real project.
* The editor gains an optional dependency on an installed `dxc`; when absent, behaviour is unchanged.
* "Draws" is not "renders like Unity": no pixel comparison with Unity exists.
