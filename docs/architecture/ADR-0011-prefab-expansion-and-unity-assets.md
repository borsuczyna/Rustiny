# ADR-0011: Prefab expansion, `MeshFilter`/`MeshRenderer`, `Mesh` and `Texture2D` assets

Status: accepted (limits below)

## Context

The real project's terrain is not an FBX drawn through a name-matched material. A terrain tile is an
instance of an FBX "prefab" whose `m_Mesh` and `m_Materials` are overridden by the scene to a Unity
`Mesh` asset (`.asset`) and a material whose shader mixes eight layers with a splat map that is itself
a YAML `Texture2D` asset. Earlier slices read none of these, so the terrain was drawn as a grey box or as
the model's own material.

## Decisions

* **Prefab expansion** (`rustiny-scene::prefab`). A `PrefabInstance` is its source asset plus
  `m_Modification`. `Library` reads the source (`.prefab`, or an `.fbx` which acts as a prefab with
  name-hash object IDs), recursively for nested prefabs and variants (nested object ID = source ID XOR
  instance ID), applies `m_Modifications` (`m_Name`, `m_IsActive`, `m_Mesh`, `m_Enabled`,
  `m_Materials.Array.data[N]`, local position/rotation/scale), honours removed components/objects and
  composes world matrices. Overrides whose target cannot be found are counted, not guessed.
* **`MeshFilter` + `MeshRenderer`** are read from scenes and prefabs; the plan draws one entry per
  renderer with the materials of its slots (submesh index = material slot).
* **`Mesh` `.asset`** (text serialization): `rustiny-model::unity_mesh` reads sub-meshes, vertex channels
  (position, normal, tangent, colour, uv0, uv1), streams and index buffers (u16/u32). 431 of 431 real
  meshes import.
* **`Texture2D` `.asset`** (text serialization): `meshes::textures::decode_texture_asset` reads the first
  level of the uncompressed formats Alpha8, RGB24, RGBA32, ARGB32, BGRA32, R8, RG16 and rebuilds the mip
  chain. Compressed formats are an error (reported, not guessed). Texture properties that name a `.asset`
  are loaded like PNGs (linear for control/normal/height maps).
* **Frustum culling** per instance with the mesh's bounds, and a 48 MB per-frame arena region, so the
  tens of thousands of draws of the real scene fit.
* **Picking** uses one box per mesh of an object (prefabs have several).

## Model files as prefabs (follow-up)

Found by comparing the render with the scene's data, not assumed:

* **Mesh IDs are hashes of names.** Since Unity 2019.1 the `fileID` of a sub-asset of a model is
  `xxh64("Type:Mesh->" + node name + "0")` as a signed 64-bit number (`rustiny_model::fbx::mesh_file_id`).
  It depends only on the type and the name, so `chassis` has the same ID in `sizuro.fbx` and in
  `chassis 1.fbx`. Checked against the real scene. A renderer's `m_Mesh` into a model file therefore
  draws that one mesh in its own space (`fbx::Part::Mesh(id)`), not the whole flattened file; an ID
  no node has draws nothing (the object keeps its proxy box) instead of a whole vehicle.
* **Object IDs in a model used as a prefab** are `xxh64("Type:GameObject->//RootNode/root/<path>0")`
  and `xxh64("Type:<Transform|MeshFilter|MeshRenderer>-><path>/<type>0")`
  (`fbx::object_id`). A file with several nodes is expanded into one node per FBX node with its
  own local transform in Unity space (`fbx::nodes`), so overrides for different nodes (roof, windows,
  frames) no longer collapse onto one. With one top-level node that node is the prefab root; with
  several the root is an empty object named like the file. A file with a single node stays one
  wildcard node drawn as the flattened model. Verified: all 16 override targets of the scene's
  `house-01.fbx` instances are found by the computed IDs.
* **Material slots.** A renderer's slot N belongs to sub-mesh N, and Unity makes one sub-mesh per
  *used* material in the order of first use among the triangles. The material list of the file can
  hold unused materials in another order (`wall, roof, window, door` in the file, `wall, window,
  door, roof` in use), so a slot is not an index into it (`slot_of_materials`). Before this, such
  a house had window materials on its roof.

## Materials stored inside `.asset` files (white terrain patches)

Large white patches of ground came from terrain tiles whose renderer names a `Material` that lives
*inside* `Definitions/TerrainPainting/Edited/Terrain_*.asset`, next to the tile's weight textures. The
plan only accepted `.mat` files, so such a slot dropped to the default (white) look. Now:

* A renderer's material is a `MaterialRef { path, file_id }`: a `.mat`, or an `.asset` plus the
  object ID. `parse_material_object(text, Some(id))` reads the `--- !u!21 &id` document.
* A texture slot with `m_Texture: {fileID: N}` and no GUID is a `Texture2D` of the same file
  (`MaterialTexture.file_id`); the shader hub names it `#N` and the resolver turns it into
  `file.asset#N`. `TextureJob.sub` makes the decoder read only that `--- !u!28 &N` document
  (`textures::document_of`). A GUID plus a non-2800000 file ID selects an object of another `.asset`.
* Checked on the real scene by screenshot (cottage view and terrain overview): no white ground left,
  the ground blends its layers. FBX-embedded materials (a renderer pointing at a material inside an
  `.fbx`) are still not resolved and fall back to the by-name match.

## Measured on the real 305 MB scene

29 026 instances expanded (1 unreadable), 389 946 overrides applied, 33 without a target; 4071 meshes,
all imported; 1864 textures requested, all uploaded; 0 shader draws left out of 1669 shader draws in the
final frame. The terrain now draws with `SummerHideout/MeshTerrain` and shows blended ground textures.

## Limits (not hidden)

* The result was never compared with Unity's own rendering; some areas (water, a very dark large patch
  of ground in the overview) still look wrong and are not diagnosed.
* A prefab instance parented under a nested child transform of another prefab attaches to the root.
* Single-node FBX files are still matched to prefab object IDs by a "wildcard by property kind" rule.
  Multi-node files use exact IDs; 40 of 389 979 overrides found no target (7 more than before the
  node expansion; not yet analysed). Skinned models (characters) are not given bones.
* UV1/tangents/colours are read from meshes but not all reach the GPU vertex format yet.
* Textures are downscaled to 256 px on the longest edge. Compressed `Texture2D` assets are unsupported.
* No shadows, no depth/opaque textures, no probes, no transparency sorting, no procedural instancing
  (Foliage renders black).
