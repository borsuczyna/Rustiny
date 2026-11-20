# ADR-0005: A purpose-built streaming reader for UnityYAML scenes

**Status:** accepted.

## Context

Scenes are text-serialized as UnityYAML: a `%YAML 1.1` preamble, then one document per object, each introduced by `--- !u!<classId> &<fileId>` (optionally `stripped`). The one real scene available for testing (`SampleScene.unity` of a Unity 6000.4 project) is 305 MB with 212 704 documents. Opening it must not block the UI and should not need gigabytes of memory.

## Decision

* **No general YAML crate.** A YAML library would build a value tree for every document (hundreds of thousands of allocations for fields the hierarchy never looks at). Unity writes a very regular subset, so `rustiny-scene::unity_yaml` reads exactly that: fixed two-space indent, sequences at the key's indent, `{fileID: N}` / `{x: 0, y: 1, z: 2}` flow mappings, `'single'` and `"double"` quoted scalars (including Unity's `\xXX`, `\uXXXX` escapes).
* **Streaming, header first.** `DocumentReader` hands out one document header at a time. The caller decides from the class ID whether it wants the body: only `GameObject`, `Transform`/`RectTransform`, `PrefabInstance`, `SceneRoots` and stripped transforms are collected; every other document (MonoBehaviours, renderers, materials, ...) is skipped line by line without being stored. Fields are then iterated by borrowing from the body (`fields()`), allocating nothing.
* **Hierarchy from transforms.** Unity links objects through transform file IDs (`m_Father`, `m_Children`). Prefab instances appear in the scene through a `stripped` transform pointing at the `PrefabInstance` document, so those transform IDs resolve to a single prefab-instance node. Sibling order follows the parent's `m_Children`; root order follows `SceneRoots`.
* **Tolerant, never silent.** Missing transforms, dangling parents, bad vectors and parent loops do not abort the load. The object is kept (loops are cut and the object shown as a root) and a `SceneIssue` is recorded (at most 1000 are stored, the rest counted).
* **Read-only.** The file is opened for reading only. Loading runs on a worker thread in the editor.

## Consequences

* On the real scene: 291.7 MiB read in 0.67-0.80 s (release, warm cache, n=6; ~370-440 MiB/s), sampled peak working set ~34 MiB. The editor (debug-profile build, project scan running at the same time) reports about 1 s.
* Counts were cross-checked against `grep` on the file: 212 704 documents, 49 624 transforms (20 597 plain + 29 027 stripped), 29 027 prefab instances, 12 roots, all equal to what the loader reports.
* Not handled: binary-serialized scenes (rejected with a clear error), multi-line plain scalars (a name wrapped over lines is truncated), YAML anchors/aliases (Unity does not write them), and anything about component *values*. These are deliberate limits of this slice.
* A prefab instance is one node: its contents (the prefab's own objects and the scene's overrides) are not expanded. Its name is the first `m_Name` override found, otherwise the editor shows the source prefab's file name resolved through the asset index by GUID. The first `m_Name` override can belong to a nested object of the prefab rather than its root; resolving that needs the prefab asset and is future work (prefab support milestone).
* If component values or scene writing are added later, the same `fields()` iterator can serve them; writing will need a round-trip-exact emitter, which this reader is not.
