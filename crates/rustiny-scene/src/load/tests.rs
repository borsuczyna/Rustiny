//! Loader tests on small synthetic scenes written the way Unity writes them.

use super::*;
use crate::model::{ObjectKind, SceneIssue};
use std::io::Cursor;

const PREAMBLE: &str = "%YAML 1.1\n%TAG !u! tag:unity3d.com,2011:\n";

fn game_object(id: i64, name: &str, components: &[i64]) -> String {
    let mut s = format!(
        "--- !u!1 &{id}\nGameObject:\n  m_ObjectHideFlags: 0\n  serializedVersion: 6\n  m_Component:\n"
    );
    for c in components {
        s.push_str(&format!("  - component: {{fileID: {c}}}\n"));
    }
    s.push_str(&format!(
        "  m_Layer: 0\n  m_Name: {name}\n  m_TagString: Untagged\n  m_IsActive: 1\n"
    ));
    s
}

fn transform(id: i64, go: i64, father: i64, children: &[i64], pos: [f32; 3]) -> String {
    let mut s = format!(
        "--- !u!4 &{id}\nTransform:\n  m_GameObject: {{fileID: {go}}}\n  m_LocalRotation: {{x: 0, y: 0, z: 0, w: 1}}\n  m_LocalPosition: {{x: {}, y: {}, z: {}}}\n  m_LocalScale: {{x: 1, y: 1, z: 1}}\n",
        pos[0], pos[1], pos[2]
    );
    if children.is_empty() {
        s.push_str("  m_Children: []\n");
    } else {
        s.push_str("  m_Children:\n");
        for c in children {
            s.push_str(&format!("  - {{fileID: {c}}}\n"));
        }
    }
    s.push_str(&format!("  m_Father: {{fileID: {father}}}\n"));
    s
}

fn load(text: &str) -> Scene {
    read_scene(Cursor::new(text.as_bytes().to_vec()), "Test").unwrap()
}

/// Root `Parent` (go 10 / tf 11) with children `B` (go 20 / tf 21) and `A` (go 30 / tf 31), in
/// that sibling order, plus a camera component.
fn simple_scene() -> String {
    let mut s = String::from(PREAMBLE);
    s.push_str("--- !u!20 &12\nCamera:\n  m_Enabled: 1\n");
    s.push_str(&game_object(10, "Parent", &[11, 12]));
    s.push_str(&transform(11, 10, 0, &[21, 31], [1.0, 2.0, 3.0]));
    s.push_str(&game_object(30, "A", &[31]));
    s.push_str(&transform(31, 30, 11, &[], [0.0; 3]));
    s.push_str(&game_object(20, "B", &[21]));
    s.push_str(&transform(21, 20, 11, &[], [0.0; 3]));
    s
}

#[test]
fn builds_a_hierarchy_in_sibling_order() {
    let scene = load(&simple_scene());
    assert_eq!(scene.name(), "Test");
    assert_eq!(scene.len(), 3);
    assert_eq!(scene.roots().len(), 1);

    let parent = scene.object(scene.roots()[0]).unwrap();
    assert_eq!(parent.name, "Parent");
    assert_eq!(parent.parent, None);
    let children: Vec<_> = parent
        .children
        .iter()
        .map(|&c| scene.object(c).unwrap().name.as_str())
        .collect();
    // The order comes from the parent's `m_Children`, not from the file order (A comes first there).
    assert_eq!(children, ["B", "A"]);
    assert_eq!(scene.descendant_count(scene.roots()[0]), 2);
    assert!(scene.issues().is_empty(), "{:?}", scene.issues());

    let b = scene.object(parent.children[0]).unwrap();
    assert_eq!(b.parent, Some(scene.roots()[0]));
}

#[test]
fn reads_transform_and_components() {
    let scene = load(&simple_scene());
    let parent = scene.object(scene.roots()[0]).unwrap();
    let t = parent.transform.unwrap();
    assert_eq!(t.position, [1.0, 2.0, 3.0]);
    assert_eq!(t.rotation, [0.0, 0.0, 0.0, 1.0]);
    assert_eq!(t.scale, [1.0; 3]);
    assert!(!t.is_rect);
    assert!(parent.active);
    assert_eq!(parent.tag, "Untagged");

    let classes: Vec<_> = parent.components.iter().map(|c| c.class_id).collect();
    assert_eq!(classes, [Some(4), Some(20)], "transform then camera");
}

#[test]
fn stats_count_documents_and_bytes() {
    let text = simple_scene();
    let scene = load(&text);
    let stats = scene.stats();
    assert_eq!(stats.documents, 7);
    assert_eq!(stats.game_objects, 3);
    assert_eq!(stats.transforms, 3);
    assert_eq!(stats.prefab_instances, 0);
    assert_eq!(stats.bytes, text.len() as u64);
}

#[test]
fn scene_roots_define_root_order() {
    let mut s = String::from(PREAMBLE);
    for (go, tf, name) in [(1, 2, "First"), (3, 4, "Second"), (5, 6, "Third")] {
        s.push_str(&game_object(go, name, &[tf]));
        s.push_str(&transform(tf, go, 0, &[], [0.0; 3]));
    }
    s.push_str("--- !u!1660057539 &9\nSceneRoots:\n  m_ObjectHideFlags: 0\n  m_Roots:\n  - {fileID: 6}\n  - {fileID: 2}\n  - {fileID: 4}\n");
    let scene = load(&s);
    let names: Vec<_> = scene
        .roots()
        .iter()
        .map(|&r| scene.object(r).unwrap().name.as_str())
        .collect();
    assert_eq!(names, ["Third", "First", "Second"]);
}

#[test]
fn without_scene_roots_roots_follow_file_order() {
    let mut s = String::from(PREAMBLE);
    for (go, tf, name) in [(1, 2, "X"), (3, 4, "Y")] {
        s.push_str(&game_object(go, name, &[tf]));
        s.push_str(&transform(tf, go, 0, &[], [0.0; 3]));
    }
    let scene = load(&s);
    let names: Vec<_> = scene
        .roots()
        .iter()
        .map(|&r| scene.object(r).unwrap().name.as_str())
        .collect();
    assert_eq!(names, ["X", "Y"]);
}

#[test]
fn prefab_instances_become_nodes_and_adopt_children() {
    let mut s = String::from(PREAMBLE);
    // A root game object whose transform lists the stripped transform of a prefab instance.
    s.push_str(&game_object(1, "Holder", &[2]));
    s.push_str(&transform(2, 1, 0, &[50], [0.0; 3]));
    s.push_str("--- !u!4 &50 stripped\nTransform:\n  m_CorrespondingSourceObject: {fileID: 4, guid: abcdefabcdefabcdefabcdefabcdefab, type: 3}\n  m_PrefabInstance: {fileID: 60}\n  m_PrefabAsset: {fileID: 0}\n");
    s.push_str("--- !u!1001 &60\nPrefabInstance:\n  m_ObjectHideFlags: 0\n  serializedVersion: 2\n  m_Modification:\n    serializedVersion: 3\n    m_TransformParent: {fileID: 2}\n    m_Modifications:\n    - target: {fileID: 1, guid: abcdefabcdefabcdefabcdefabcdefab, type: 3}\n      propertyPath: m_Name\n      value: Tree 'big'\n      objectReference: {fileID: 0}\n    m_RemovedComponents: []\n  m_SourcePrefab: {fileID: 100100000, guid: abcdefabcdefabcdefabcdefabcdefab, type: 3}\n");
    // A scene object parented to the prefab's (stripped) transform.
    s.push_str(&game_object(70, "Attached", &[71]));
    s.push_str(&transform(71, 70, 50, &[], [0.0; 3]));

    let scene = load(&s);
    assert!(scene.issues().is_empty(), "{:?}", scene.issues());
    assert_eq!(scene.stats().prefab_instances, 1);
    assert_eq!(scene.stats().game_objects, 2);

    let holder = scene.object(scene.roots()[0]).unwrap();
    assert_eq!(holder.name, "Holder");
    assert_eq!(holder.children.len(), 1);
    let prefab = scene.object(holder.children[0]).unwrap();
    match &prefab.kind {
        ObjectKind::PrefabInstance { source_guid } => {
            assert_eq!(
                source_guid.as_deref(),
                Some("abcdefabcdefabcdefabcdefabcdefab")
            );
        }
        other => panic!("expected prefab instance, got {other:?}"),
    }
    assert_eq!(prefab.name, "Tree 'big'");
    assert_eq!(prefab.parent, Some(scene.roots()[0]));

    assert_eq!(
        prefab.children.len(),
        1,
        "a scene object attached to the prefab"
    );
    assert_eq!(scene.object(prefab.children[0]).unwrap().name, "Attached");
}

#[test]
fn a_prefab_instance_without_a_parent_is_a_root() {
    let mut s = String::from(PREAMBLE);
    s.push_str("--- !u!1001 &5\nPrefabInstance:\n  m_Modification:\n    m_TransformParent: {fileID: 0}\n    m_Modifications: []\n  m_SourcePrefab: {fileID: 100100000, guid: 00000000000000000000000000000001, type: 3}\n");
    let scene = load(&s);
    assert_eq!(scene.roots().len(), 1);
    let prefab = scene.object(scene.roots()[0]).unwrap();
    assert_eq!(prefab.name, "");
}

#[test]
fn problems_are_reported_but_loading_continues() {
    let mut s = String::from(PREAMBLE);
    // No transform at all.
    s.push_str(&game_object(1, "Orphan", &[]));
    // Parent that does not exist.
    s.push_str(&game_object(3, "Lost", &[4]));
    s.push_str(&transform(4, 3, 999, &[], [0.0; 3]));
    // A roots list naming a missing transform.
    s.push_str(
        "--- !u!1660057539 &9\nSceneRoots:\n  m_Roots:\n  - {fileID: 777}\n  - {fileID: 4}\n",
    );
    let scene = load(&s);

    assert_eq!(scene.len(), 2);
    assert_eq!(scene.roots().len(), 2, "both objects are still shown");
    assert!(
        scene
            .issues()
            .contains(&SceneIssue::GameObjectWithoutTransform { file_id: 1 })
    );
    assert!(scene.issues().contains(&SceneIssue::UnknownParent {
        file_id: 3,
        parent: 999
    }));
    assert!(
        scene
            .issues()
            .contains(&SceneIssue::UnknownRoot { transform: 777 })
    );
}

#[test]
fn parent_loops_are_cut() {
    let mut s = String::from(PREAMBLE);
    s.push_str(&game_object(1, "A", &[2]));
    s.push_str(&transform(2, 1, 4, &[4], [0.0; 3]));
    s.push_str(&game_object(3, "B", &[4]));
    s.push_str(&transform(4, 3, 2, &[2], [0.0; 3]));
    // A transform that is its own father.
    s.push_str(&game_object(5, "Self", &[6]));
    s.push_str(&transform(6, 5, 6, &[6], [0.0; 3]));
    let scene = load(&s);

    assert_eq!(scene.len(), 3);
    // Every object must be reachable from the roots exactly once.
    let mut seen = vec![0; scene.len()];
    let mut stack: Vec<_> = scene.roots().to_vec();
    while let Some(id) = stack.pop() {
        seen[id] += 1;
        assert!(seen[id] == 1, "object {id} reached twice");
        stack.extend_from_slice(&scene.object(id).unwrap().children);
    }
    assert!(seen.iter().all(|&n| n == 1), "{seen:?}");
    assert!(
        scene
            .issues()
            .iter()
            .any(|i| matches!(i, SceneIssue::Cycle { .. }))
    );
}

#[test]
fn names_may_be_quoted_and_escaped() {
    let mut s = String::from(PREAMBLE);
    s.push_str(&game_object(1, "'Door: front'", &[]));
    s.push_str(&game_object(2, r#""Zaż\xF3ł""#, &[]));
    let scene = load(&s);
    let names: Vec<_> = scene.objects().iter().map(|o| o.name.as_str()).collect();
    assert_eq!(names, ["Door: front", "Zażół"]);
}

#[test]
fn inactive_layer_tag_and_rect_transform() {
    let mut s = String::from(PREAMBLE);
    s.push_str("--- !u!1 &1\nGameObject:\n  m_Component:\n  - component: {fileID: 2}\n  m_Layer: 5\n  m_Name: Button\n  m_TagString: Respawn\n  m_IsActive: 0\n");
    s.push_str("--- !u!224 &2\nRectTransform:\n  m_GameObject: {fileID: 1}\n  m_LocalRotation: {x: 0, y: 0, z: 0, w: 1}\n  m_LocalPosition: {x: 0, y: 0, z: 0}\n  m_LocalScale: {x: 2, y: 2, z: 2}\n  m_Children: []\n  m_Father: {fileID: 0}\n");
    let scene = load(&s);
    let button = scene.object(0).unwrap();
    assert!(!button.active);
    assert_eq!(button.layer, 5);
    assert_eq!(button.tag, "Respawn");
    let t = button.transform.unwrap();
    assert!(t.is_rect);
    assert_eq!(t.scale, [2.0; 3]);
}

#[test]
fn an_object_below_an_inactive_parent_is_inactive_too() {
    let mut s = String::from(PREAMBLE);
    // Parent (go 10) is off; Child (go 20) is on by itself; Other (go 30) is a separate root.
    s.push_str(&game_object(10, "Parent", &[11]).replace("m_IsActive: 1", "m_IsActive: 0"));
    s.push_str(&transform(11, 10, 0, &[21], [0.0; 3]));
    s.push_str(&game_object(20, "Child", &[21]));
    s.push_str(&transform(21, 20, 11, &[], [0.0; 3]));
    s.push_str(&game_object(30, "Other", &[31]));
    s.push_str(&transform(31, 30, 0, &[], [0.0; 3]));
    let scene = load(&s);
    let id = |name: &str| scene.objects().iter().position(|o| o.name == name).unwrap();
    assert!(!scene.is_active(id("Parent")));
    assert!(
        scene.object(id("Child")).unwrap().active,
        "its own flag is set"
    );
    assert!(!scene.is_active(id("Child")), "but its parent is off");
    assert!(scene.is_active(id("Other")));
    assert!(!scene.is_active(999), "unknown ids are not active");
}

#[test]
fn unreadable_vectors_fall_back_to_identity_and_are_reported() {
    let mut s = String::from(PREAMBLE);
    s.push_str(&game_object(1, "Odd", &[2]));
    s.push_str("--- !u!4 &2\nTransform:\n  m_GameObject: {fileID: 1}\n  m_LocalPosition: {x: nope, y: 0, z: 0}\n  m_Children: []\n  m_Father: {fileID: 0}\n");
    let scene = load(&s);
    assert_eq!(
        scene.object(0).unwrap().transform.unwrap().position,
        [0.0; 3]
    );
    assert!(scene.issues().contains(&SceneIssue::BadValue {
        file_id: 2,
        field: "m_LocalPosition"
    }));
}

#[test]
fn components_pointing_nowhere_have_no_class() {
    let mut s = String::from(PREAMBLE);
    s.push_str(&game_object(1, "X", &[2, 404]));
    s.push_str(&transform(2, 1, 0, &[], [0.0; 3]));
    let scene = load(&s);
    let classes: Vec<_> = scene
        .object(0)
        .unwrap()
        .components
        .iter()
        .map(|c| c.class_id)
        .collect();
    assert_eq!(classes, [Some(4), None]);
}

#[test]
fn crlf_scenes_load_the_same() {
    let lf = simple_scene();
    let crlf = lf.replace('\n', "\r\n");
    let a = load(&lf);
    let b = load(&crlf);
    assert_eq!(a.len(), b.len());
    for (x, y) in a.objects().iter().zip(b.objects()) {
        assert_eq!(x.name, y.name);
        assert_eq!(x.children, y.children);
        assert_eq!(x.transform, y.transform);
    }
}

#[test]
fn rejects_files_that_are_not_text_scenes() {
    let err = read_scene(Cursor::new(b"UnityFS\0binary".to_vec()), "x").unwrap_err();
    assert!(matches!(err, SceneError::NotUnityYaml));
}

#[test]
fn an_empty_but_valid_scene_loads() {
    let scene = load("%YAML 1.1\n%TAG !u! tag:unity3d.com,2011:\n");
    assert!(scene.is_empty());
    assert!(scene.roots().is_empty());
}

#[test]
fn load_scene_reads_a_file_and_reports_a_missing_one() {
    let dir = std::env::temp_dir().join(format!("rustiny-scene-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("Level.unity");
    std::fs::write(&path, simple_scene()).unwrap();
    let scene = load_scene(&path).unwrap();
    assert_eq!(scene.name(), "Level");
    assert_eq!(scene.len(), 3);

    let err = load_scene(&dir.join("missing.unity")).unwrap_err();
    assert!(matches!(err, SceneError::Open { .. }));
    std::fs::remove_dir_all(&dir).unwrap();
}

fn transform_with(id: i64, go: i64, father: i64, pos: [f32; 3], scale: f32) -> String {
    format!(
        "--- !u!4 &{id}\nTransform:\n  m_GameObject: {{fileID: {go}}}\n  m_LocalRotation: {{x: 0, y: 0, z: 0, w: 1}}\n  m_LocalPosition: {{x: {}, y: {}, z: {}}}\n  m_LocalScale: {{x: {scale}, y: {scale}, z: {scale}}}\n  m_Children: []\n  m_Father: {{fileID: {father}}}\n",
        pos[0], pos[1], pos[2]
    )
}

#[test]
fn world_matrices_follow_the_hierarchy() {
    let mut s = String::from(PREAMBLE);
    s.push_str(&game_object(1, "Parent", &[2]));
    s.push_str(&transform_with(2, 1, 0, [10.0, 0.0, 0.0], 2.0));
    s.push_str(&game_object(3, "Child", &[4]));
    s.push_str(&transform_with(4, 3, 2, [1.0, 0.0, 0.0], 1.0));
    s.push_str(&game_object(5, "Grandchild", &[6]));
    s.push_str(&transform_with(6, 5, 4, [0.0, 1.0, 0.0], 1.0));
    let scene = load(&s);
    let pos = |name: &str| {
        let id = scene.objects().iter().position(|o| o.name == name).unwrap();
        scene.world_position(id).unwrap()
    };
    assert_eq!(pos("Parent"), [10.0, 0.0, 0.0]);
    // The child's local offset is scaled by the parent's scale of 2.
    assert_eq!(pos("Child"), [12.0, 0.0, 0.0]);
    assert_eq!(pos("Grandchild"), [12.0, 2.0, 0.0]);
}

/// A prefab instance placed under `Holder` with position, rotation and scale overrides on its
/// root transform (source id 7), plus an override on another transform (source id 8).
fn prefab_scene(stripped_first: bool) -> String {
    let mut s = String::from(PREAMBLE);
    s.push_str(&game_object(1, "Holder", &[2]));
    s.push_str(
        &transform_with(2, 1, 0, [100.0, 0.0, 0.0], 1.0)
            .replace("m_Children: []", "m_Children:\n  - {fileID: 50}"),
    );
    let stripped = "--- !u!4 &50 stripped\nTransform:\n  m_CorrespondingSourceObject: {fileID: 7, guid: abcdefabcdefabcdefabcdefabcdefab, type: 3}\n  m_PrefabInstance: {fileID: 60}\n";
    let other = "--- !u!4 &51 stripped\nTransform:\n  m_CorrespondingSourceObject: {fileID: 8, guid: abcdefabcdefabcdefabcdefabcdefab, type: 3}\n  m_PrefabInstance: {fileID: 60}\n";
    let mut mods = String::new();
    for (target, path, value) in [
        (7, "m_LocalPosition.x", "1"),
        (7, "m_LocalPosition.y", "2"),
        (7, "m_LocalPosition.z", "3"),
        (7, "m_LocalScale.x", "4"),
        (8, "m_LocalPosition.x", "999"),
    ] {
        mods.push_str(&format!(
            "    - target: {{fileID: {target}, guid: abcdefabcdefabcdefabcdefabcdefab, type: 3}}\n      propertyPath: {path}\n      value: {value}\n      objectReference: {{fileID: 0}}\n"
        ));
    }
    mods.push_str("    - target: {fileID: 7, guid: abcdefabcdefabcdefabcdefabcdefab, type: 3}\n      propertyPath: 'm_Materials.Array.data[0]'\n      value: \n      objectReference: {fileID: 2100000, guid: 4a619f174c571d241b742cd20172b9cc, type: 2}\n");
    let instance = format!(
        "--- !u!1001 &60\nPrefabInstance:\n  m_Modification:\n    m_TransformParent: {{fileID: 2}}\n    m_Modifications:\n{mods}    m_RemovedComponents: []\n  m_SourcePrefab: {{fileID: 100100000, guid: abcdefabcdefabcdefabcdefabcdefab, type: 3}}\n"
    );
    if stripped_first {
        s.push_str(other);
        s.push_str(stripped);
        s.push_str(&instance);
    } else {
        s.push_str(&instance);
        s.push_str(stripped);
        s.push_str(other);
    }
    s
}

#[test]
fn prefab_instances_get_their_transform_from_the_overrides() {
    for stripped_first in [false, true] {
        let scene = load(&prefab_scene(stripped_first));
        assert!(scene.issues().is_empty(), "{:?}", scene.issues());
        let id = scene
            .objects()
            .iter()
            .position(|o| matches!(o.kind, ObjectKind::PrefabInstance { .. }))
            .unwrap();
        let t = scene.object(id).unwrap().transform.unwrap();
        // The listed root transform (50 -> source 7) is used, not the other one (51 -> source 8).
        assert_eq!(
            t.position,
            [1.0, 2.0, 3.0],
            "stripped_first={stripped_first}"
        );
        assert_eq!(t.scale, [4.0, 1.0, 1.0], "only x was overridden");
        assert_eq!(t.rotation, [0.0, 0.0, 0.0, 1.0], "defaults for the rest");
        // Placed under Holder at x = 100.
        assert_eq!(scene.world_position(id).unwrap(), [101.0, 2.0, 3.0]);
    }
}

#[test]
fn prefab_instances_without_transform_overrides_have_no_transform() {
    let mut s = String::from(PREAMBLE);
    s.push_str("--- !u!1001 &5\nPrefabInstance:\n  m_Modification:\n    m_TransformParent: {fileID: 0}\n    m_Modifications:\n    - target: {fileID: 9, guid: abcdefabcdefabcdefabcdefabcdefab, type: 3}\n      propertyPath: m_Name\n      value: Plain\n      objectReference: {fileID: 0}\n  m_SourcePrefab: {fileID: 100100000, guid: abcdefabcdefabcdefabcdefabcdefab, type: 3}\n");
    let scene = load(&s);
    let prefab = scene.object(0).unwrap();
    assert_eq!(prefab.name, "Plain");
    assert!(prefab.transform.is_none());
    assert_eq!(scene.world_position(0), Some([0.0; 3]));
}
