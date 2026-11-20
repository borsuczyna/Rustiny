//! Names of Unity's numeric class IDs (the `N` in `--- !u!N &fileID`), for the common built-in
//! types. Unknown IDs are shown as `Class <id>`.

/// Unity class ID of `GameObject`.
pub const GAME_OBJECT: u32 = 1;
/// Unity class ID of `Transform`.
pub const TRANSFORM: u32 = 4;
/// Unity class ID of `RectTransform`.
pub const RECT_TRANSFORM: u32 = 224;
/// Unity class ID of `MeshFilter`.
pub const MESH_FILTER: u32 = 33;
/// Unity class ID of `MeshRenderer`.
pub const MESH_RENDERER: u32 = 23;
/// Unity class ID of `SkinnedMeshRenderer`.
pub const SKINNED_MESH_RENDERER: u32 = 137;
/// Unity class ID of `SpriteRenderer`.
pub const SPRITE_RENDERER: u32 = 212;
/// Unity class ID of `MonoBehaviour`.
pub const MONO_BEHAVIOUR: u32 = 114;
/// Unity class ID of `Light`.
pub const LIGHT: u32 = 108;
/// Unity class ID of `RenderSettings`.
pub const RENDER_SETTINGS: u32 = 104;
/// Unity class ID of `PrefabInstance`.
pub const PREFAB_INSTANCE: u32 = 1001;
/// Unity class ID of `SceneRoots`.
pub const SCENE_ROOTS: u32 = 1_660_057_539;

/// Name of a built-in class, if known.
pub fn class_name(id: u32) -> Option<&'static str> {
    Some(match id {
        1 => "GameObject",
        2 => "Component",
        4 => "Transform",
        20 => "Camera",
        21 => "Material",
        23 => "MeshRenderer",
        25 => "Renderer",
        33 => "MeshFilter",
        43 => "Mesh",
        54 => "Rigidbody",
        56 => "Collider",
        64 => "MeshCollider",
        65 => "BoxCollider",
        81 => "AudioListener",
        82 => "AudioSource",
        95 => "Animator",
        108 => "Light",
        111 => "Animation",
        114 => "MonoBehaviour",
        120 => "LineRenderer",
        135 => "SphereCollider",
        136 => "CapsuleCollider",
        137 => "SkinnedMeshRenderer",
        198 => "ParticleSystem",
        199 => "ParticleSystemRenderer",
        205 => "LODGroup",
        212 => "SpriteRenderer",
        222 => "CanvasRenderer",
        223 => "Canvas",
        224 => "RectTransform",
        1001 => "PrefabInstance",
        1660057539 => "SceneRoots",
        _ => return None,
    })
}

/// Display name: the built-in name or `Class <id>`.
pub fn display_name(id: u32) -> String {
    class_name(id).map_or_else(|| format!("Class {id}"), str::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_and_unknown_ids() {
        assert_eq!(class_name(GAME_OBJECT), Some("GameObject"));
        assert_eq!(class_name(SCENE_ROOTS), Some("SceneRoots"));
        assert_eq!(display_name(114), "MonoBehaviour");
        assert_eq!(display_name(99999), "Class 99999");
    }
}
