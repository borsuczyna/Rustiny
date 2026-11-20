//! Second pass: turn the collected documents into a hierarchy.
//!
//! Unity links objects by *transform* file IDs: a transform names its father and its children.
//! Prefab instances appear in the hierarchy through a `stripped` transform, so those transform
//! IDs resolve to the prefab instance node.

use super::collect::{Collected, RawPrefab};
use crate::math::Mat4;
use crate::model::{
    Component, LoadStats, ObjectId, ObjectKind, Scene, SceneIssue, SceneLight, SceneLighting,
    SceneObject, TransformData,
};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

pub(super) fn assemble(mut raw: Collected, name: String, bytes: u64, started: Instant) -> Scene {
    let mut nodes: Vec<SceneObject> =
        Vec::with_capacity(raw.game_objects.len() + raw.prefabs.len());
    // Per node: the transform file ID of its parent (0 = none) and its listed children.
    let mut parent_refs: Vec<i64> = Vec::with_capacity(nodes.capacity());
    let mut listed_children: Vec<Vec<i64>> = Vec::with_capacity(nodes.capacity());
    // Transform file ID -> node.
    let mut by_transform: HashMap<i64, ObjectId> = HashMap::new();

    let mut transform_of_object: HashMap<i64, usize> = HashMap::with_capacity(raw.transforms.len());
    for (index, transform) in raw.transforms.iter().enumerate() {
        transform_of_object.insert(transform.game_object, index);
    }

    for game_object in std::mem::take(&mut raw.game_objects) {
        let id = nodes.len();
        let transform = transform_of_object
            .get(&game_object.file_id)
            .map(|&i| &raw.transforms[i]);
        let (data, parent_ref, children) = match transform {
            Some(t) => {
                by_transform.insert(t.file_id, id);
                (Some(t.data), t.father, t.children.clone())
            }
            None => {
                raw.issue(SceneIssue::GameObjectWithoutTransform {
                    file_id: game_object.file_id,
                });
                (None::<TransformData>, 0, Vec::new())
            }
        };
        let components = game_object
            .components
            .iter()
            .map(|&file_id| Component {
                file_id,
                class_id: raw.class_of.get(&file_id).copied(),
            })
            .collect();
        nodes.push(SceneObject {
            file_id: game_object.file_id,
            name: game_object.name,
            active: game_object.active,
            layer: game_object.layer,
            tag: game_object.tag,
            kind: ObjectKind::GameObject,
            mesh: mesh_of(&raw, game_object.file_id),
            renderer: raw
                .renderers
                .get(&game_object.file_id)
                .map(|(state, _)| state.clone()),
            overrides: None,
            transform: data,
            parent: None,
            children: Vec::new(),
            components,
        });
        parent_refs.push(parent_ref);
        listed_children.push(children);
    }

    // Transform IDs that some parent (or `SceneRoots`) lists: the root transform of a prefab
    // instance is one of these.
    let listed: HashSet<i64> = raw
        .transforms
        .iter()
        .flat_map(|t| t.children.iter().copied())
        .chain(raw.roots.iter().flatten().copied())
        .collect();
    let mut stripped_of: HashMap<i64, Vec<(i64, i64)>> = HashMap::new();
    for (&transform, stripped) in &raw.stripped_transforms {
        stripped_of
            .entry(stripped.prefab)
            .or_default()
            .push((transform, stripped.source));
    }
    for candidates in stripped_of.values_mut() {
        candidates.sort_unstable();
    }

    let mut prefab_node: HashMap<i64, ObjectId> = HashMap::with_capacity(raw.prefabs.len());
    for prefab in std::mem::take(&mut raw.prefabs) {
        let id = nodes.len();
        prefab_node.insert(prefab.file_id, id);
        let candidates = stripped_of
            .get(&prefab.file_id)
            .map_or(&[][..], Vec::as_slice);
        let transform = prefab_transform(&prefab, candidates, &listed);
        nodes.push(SceneObject {
            file_id: prefab.file_id,
            name: prefab.name.unwrap_or_default(),
            active: true,
            layer: 0,
            tag: String::new(),
            kind: ObjectKind::PrefabInstance {
                source_guid: prefab.source_guid,
            },
            transform,
            parent: None,
            children: Vec::new(),
            components: Vec::new(),
            mesh: None,
            renderer: None,
            overrides: Some(Box::new(prefab.overrides)),
        });
        parent_refs.push(prefab.parent);
        listed_children.push(Vec::new());
    }
    for (&transform, stripped) in &raw.stripped_transforms {
        if let Some(&node) = prefab_node.get(&stripped.prefab) {
            by_transform.insert(transform, node);
        }
    }

    // Parent links.
    for id in 0..nodes.len() {
        let parent_ref = parent_refs[id];
        if parent_ref == 0 {
            continue;
        }
        match by_transform.get(&parent_ref) {
            Some(&parent) if parent != id => nodes[id].parent = Some(parent),
            _ => raw.issue(SceneIssue::UnknownParent {
                file_id: nodes[id].file_id,
                parent: parent_ref,
            }),
        }
    }

    // Children: first in the order the parent's transform lists them, then any others in file order.
    let mut placed = vec![false; nodes.len()];
    for (parent, listed) in listed_children.iter().enumerate() {
        for transform in listed {
            if let Some(&child) = by_transform.get(transform)
                && nodes[child].parent == Some(parent)
                && !placed[child]
            {
                placed[child] = true;
                nodes[parent].children.push(child);
            }
        }
    }
    for child in 0..nodes.len() {
        if let Some(parent) = nodes[child].parent
            && !placed[child]
        {
            nodes[parent].children.push(child);
        }
    }

    let roots = order_roots(&mut nodes, &mut raw, &by_transform);
    let world = world_matrices(&nodes, &roots);

    let stats = LoadStats {
        documents: raw.documents,
        game_objects: u64::try_from(nodes.len() - prefab_node.len()).unwrap_or(u64::MAX),
        transforms: raw.transform_documents,
        prefab_instances: u64::try_from(prefab_node.len()).unwrap_or(u64::MAX),
        bytes,
        issues_dropped: raw.issues_dropped,
        duration: started.elapsed(),
    };
    let object_of: HashMap<i64, ObjectId> = nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.kind == ObjectKind::GameObject)
        .map(|(i, n)| (n.file_id, i))
        .collect();
    let lights = std::mem::take(&mut raw.lights)
        .into_iter()
        .map(|light| SceneLight {
            object: object_of.get(&light.game_object).copied(),
            light,
        })
        .collect();
    let volumes = std::mem::take(&mut raw.volumes)
        .into_iter()
        .map(|volume| (object_of.get(&volume.game_object).copied(), volume))
        .collect();
    let lighting = SceneLighting {
        settings: raw.render_settings.take(),
        lights,
        volumes,
    };
    Scene::new(name, nodes, roots, raw.issues, stats, world).with_lighting(lighting)
}

/// The mesh of a game object: its `MeshFilter`'s, else its skinned renderer's own.
fn mesh_of(raw: &Collected, game_object: i64) -> Option<crate::prefab::AssetRef> {
    raw.filters
        .get(&game_object)
        .cloned()
        .flatten()
        .or_else(|| {
            raw.renderers
                .get(&game_object)
                .and_then(|(_, mesh)| mesh.clone())
        })
}

/// Roots in `SceneRoots` order (then any others in file order), with loops in the parent links cut.
fn order_roots(
    nodes: &mut [SceneObject],
    raw: &mut Collected,
    by_transform: &HashMap<i64, ObjectId>,
) -> Vec<ObjectId> {
    let mut roots: Vec<ObjectId> = Vec::new();
    let mut is_root = vec![false; nodes.len()];

    for transform in raw.roots.take().unwrap_or_default() {
        match by_transform.get(&transform) {
            Some(&node) if nodes[node].parent.is_none() && !is_root[node] => {
                is_root[node] = true;
                roots.push(node);
            }
            Some(_) => {}
            None => raw.issue(SceneIssue::UnknownRoot { transform }),
        }
    }
    for node in 0..nodes.len() {
        if nodes[node].parent.is_none() && !is_root[node] {
            is_root[node] = true;
            roots.push(node);
        }
    }

    // Anything not reachable from a root hangs off a loop: cut it loose and make it a root.
    let mut visited = vec![false; nodes.len()];
    let mut stack: Vec<ObjectId> = Vec::new();
    for &root in &roots {
        mark_reachable(nodes, &mut visited, &mut stack, root);
    }
    for node in 0..nodes.len() {
        if visited[node] {
            continue;
        }
        if let Some(parent) = nodes[node].parent.take() {
            nodes[parent].children.retain(|&c| c != node);
        }
        raw.issue(SceneIssue::Cycle {
            file_id: nodes[node].file_id,
        });
        roots.push(node);
        mark_reachable(nodes, &mut visited, &mut stack, node);
    }
    roots
}

fn mark_reachable(
    nodes: &[SceneObject],
    visited: &mut [bool],
    stack: &mut Vec<ObjectId>,
    start: ObjectId,
) {
    stack.push(start);
    while let Some(node) = stack.pop() {
        if std::mem::replace(&mut visited[node], true) {
            continue;
        }
        stack.extend_from_slice(&nodes[node].children);
    }
}

/// Local transform of a prefab instance, from the scene's overrides of its root transform.
///
/// The root transform is one of the instance's `stripped` transforms. When several exist, the
/// best candidate is one that a parent (or `SceneRoots`) lists *and* that has overrides, then
/// any with overrides, then the first. Values the scene does not override keep Unity's defaults
/// (the prefab asset's own values are not read). `None` when nothing is overridden.
fn prefab_transform(
    prefab: &RawPrefab,
    candidates: &[(i64, i64)],
    listed: &HashSet<i64>,
) -> Option<TransformData> {
    let has_mods = |source: i64| prefab.transform_mods.iter().any(|m| m.target == source);
    let pick = candidates
        .iter()
        .find(|(transform, source)| listed.contains(transform) && has_mods(*source))
        .or_else(|| candidates.iter().find(|(_, source)| has_mods(*source)))?;
    let mut data = TransformData::default();
    for m in prefab.transform_mods.iter().filter(|m| m.target == pick.1) {
        match m.slot {
            0..=2 => data.position[usize::from(m.slot)] = m.value,
            3..=6 => data.rotation[usize::from(m.slot - 3)] = m.value,
            _ => data.scale[usize::from(m.slot - 7)] = m.value,
        }
    }
    Some(data)
}

/// Local-to-world matrix of every node: parents are applied before children.
fn world_matrices(nodes: &[SceneObject], roots: &[ObjectId]) -> Vec<Mat4> {
    let local = |node: &SceneObject| {
        node.transform.map_or(Mat4::IDENTITY, |t| {
            Mat4::from_trs(t.position, t.rotation, t.scale)
        })
    };
    let mut world = vec![Mat4::IDENTITY; nodes.len()];
    let mut stack: Vec<ObjectId> = Vec::new();
    for &root in roots {
        world[root] = local(&nodes[root]);
        stack.push(root);
    }
    while let Some(parent) = stack.pop() {
        let parent_world = world[parent];
        for &child in &nodes[parent].children {
            world[child] = parent_world.mul(&local(&nodes[child]));
            stack.push(child);
        }
    }
    world
}
