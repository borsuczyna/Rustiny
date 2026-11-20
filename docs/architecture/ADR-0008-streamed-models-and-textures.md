# ADR-0008: Models and textures are loaded on workers and streamed to the GPU

**Status:** accepted. Extends [ADR-0007](ADR-0007-gpu-scene-pass.md).

## Context

ADR-0007 drew every object as a box because nothing could be imported. The real scene has 49 624 objects, 3 787 distinct model files (FBX), about 1 200 distinct albedo textures (PNG, 154 MB on disk for the project's materials) and 2 000+ `.mat` files. Reading, parsing and decoding that on the UI thread would freeze the editor for seconds.

## Decision

* **`rustiny-model`** reads binary FBX (no SDK) and converts to Unity's conventions (left-handed, Y up, metres, `globalScale`/`useFileUnits` from the model's `.meta`). One model file becomes one flattened mesh with one section per material. No skins, no animation, no blend shapes.
* **Plan first, load later.** On the scene-load worker, `build_plan` groups prefab instances of `.fbx` models by file and builds the instance lists, so the renderer draws all instances of a model with one instanced call per section. Objects without a model stay proxy boxes. Files named `..._LODn` (n >= 1) are skipped in favour of `..._LOD0`: a file-name heuristic, not Unity LOD groups.
* **Workers import, the UI uploads.** `MeshStream` imports models on `available_parallelism - 2` threads (1..6) and hands them over a bounded channel; the UI thread takes results for at most 5 ms per frame and uploads them. A model that fails keeps a proxy box. The scene is usable immediately and fills in.
* **Materials by name.** The material names of a model are matched, case-insensitively, to the names of the project's `.mat` files (closest folder wins). This is a heuristic: real Unity resolves through the importer's remap table or the prefab renderers' material lists, neither of which is read yet. The match rate is measured and reported (`materials_matched` / `material_slots`).
* **Shaders are checked.** A project shader's `.shader` file is scanned for the properties it declares; `_BaseMap`/`_BaseColor` of a material are only trusted if its shader declares them. Textures named like weight maps (`..._Control0...`) are not albedo and are ignored (terrain-mixing shaders are not evaluated).
* **Textures.** A second worker pool decodes PNGs, shrinks them by halving to at most 256 px and builds the mip chain. The renderer keeps one descriptor set per texture (sampled image; one immutable trilinear repeat sampler in the set layout) and a 1x1 white texture for untextured sections. `add_texture` creates the image and a staging buffer; the copy is recorded into the frame's command buffer by the next scene pass, at most 8 MB of texels per pass, so streaming never stalls a frame. Sections whose texture has not arrived draw plain.
* **Alpha.** Alpha-clipped materials discard below their `_Cutoff`; transparent materials discard below 0.5 of the texture alpha (the mesh pipeline has no blending); clear transparent materials without a texture are not drawn.
* **Colour.** Material colours are converted to linear (`powf(2.2)`) because the shader gamma-encodes its output; textures are `R8G8B8A8_SRGB`.

## Consequences

* Scene open to all models on the GPU took about 1 s on the real scene; textures follow.
* GPU texture memory is about 275 MB for the 1 184 textures the real scene's matched materials use (256 px cap). There is no eviction.
* Not done: `.prefab` expansion (large prefab instances are still boxes), `MeshFilter`/`MeshRenderer` values, per-renderer material lists, normal maps, shadows, frustum culling, real LOD groups, transparent blending.
