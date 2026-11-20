# ADR-0009: URP Lit is implemented natively; other Unity shaders are not translated (yet)

**Status:** accepted. Extends [ADR-0008](ADR-0008-streamed-models-and-textures.md); keeps the rule of [ADR-0007](ADR-0007-gpu-scene-pass.md) that shaders are WGSL compiled by `naga` at build time, with no external shader compiler.

## Context

Unity shaders are written in ShaderLab around HLSL, or generated from Shader Graph. Translating them in general needs an HLSL front end; `naga` has none, so that would mean bundling DXC or Slang, which ADR-0007 deliberately avoided. In the real project (about 2 100 `.mat` files) most materials use URP's own Lit shader (1 274), then a few project shaders (LitNormalFade 534, TreeBillboard 119, TreeFoliage 99, MeshTerrain 43, ...). What Lit does with a material is a fixed, documented set of inputs: albedo, normal map, metallic/smoothness, emission, alpha clip.

## Decision

* The renderer implements **one surface model** that follows URP Lit's metallic workflow (`rustiny_render::Surface`): tint, albedo, normal map (+ scale), metallic-smoothness map or the two scalars, emission, alpha cutoff, and an `unlit` flag for URP Unlit. The editor maps `.mat` properties to it (`meshes/materials.rs`); a project shader only contributes the properties it declares (unchanged rule), a shader that is not a project file (URP's) contributes all of them.
* **Per-section records in a storage buffer.** Each section of an uploaded mesh gets a 48-byte record in one host-visible buffer (`material.rs`); a draw pushes only the record's index (4 bytes of the push block). The buffer is append-only so no record a submitted frame reads is rewritten; `clear_meshes` retires the whole block (buffer, pool, set) and starts a new one. Capacity is 131 072 records; adding beyond it is an error, not a silent wrap.
* **Three texture slots per section** (albedo sRGB, normal and metallic-smoothness linear). One descriptor set per distinct *effective* triple, created on first use and never rewritten; a slot whose texture is absent or not yet copied uses a 1x1 default (white, flat normal, no metal / full smoothness). One immutable trilinear repeat sampler. The same PNG used as colour and as data is two textures.
* **No vertex tangents.** The fragment shader builds the tangent frame from screen-space derivatives of position and UV, so imported meshes need nothing extra. It lights the side of a face that faces the camera.
* **Lighting is deliberately simple:** one directional light, a constant ambient term, Blinn-Phong for the specular lobe (scaled by smoothness squared; metals take their colour from the albedo and reflect a flat ambient "sky" because there is no environment map). It is not URP's BRDF and will not match Unity pixel for pixel.
* **Texture rows:** PNG decoding flips rows so that row 0 is the bottom, matching V-up UVs (before this ADR textures were sampled upside down; tiled textures hid it).
* **Not translated:** custom ShaderLab/HLSL and Shader Graph shaders still fall back to the default grey (or to the albedo/colour they declare). A general translator (ShaderLab parser + HLSL front end + migration report) stays a later step; hand-made equivalents for the project's terrain mixing, tree billboards and water are the next candidates.

## Consequences

* Most materials of the real scene now show normal-mapped, metallic and emissive surfaces, and Unlit materials are drawn unlit.
* A shader this does not implement is silently approximated by the closest declared properties; the resolver does not yet report which shader each fallback came from.
* No shadows, no environment/reflection probes, no occlusion map, no detail maps, no parallax, no real transparency (alpha testing only), no lightmaps. These change the look noticeably against Unity and are listed in `MILESTONES.md`.
