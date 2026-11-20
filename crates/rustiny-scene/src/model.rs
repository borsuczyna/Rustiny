//! The scene as the editor sees it: a forest of objects with parent/child links.

use crate::math::Mat4;
use crate::prefab::{AssetRef, Overrides, RendererState};
use std::time::Duration;

/// Index of an object inside a [`Scene`] (stable for the life of that `Scene`).
pub type ObjectId = usize;

/// Local transform of an object, as stored in the file.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransformData {
    /// Local position.
    pub position: [f32; 3],
    /// Local rotation as a quaternion `[x, y, z, w]`.
    pub rotation: [f32; 4],
    /// Local scale.
    pub scale: [f32; 3],
    /// Whether this is a `RectTransform` (UI) rather than a plain `Transform`.
    pub is_rect: bool,
}

impl Default for TransformData {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
            is_rect: false,
        }
    }
}

/// What an object in the hierarchy is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObjectKind {
    /// A `GameObject` stored in the scene file.
    GameObject,
    /// An instance of a prefab. Its contents live in the prefab asset and are not expanded here.
    PrefabInstance {
        /// GUID of the source prefab asset, when the file names one.
        source_guid: Option<String>,
    },
}

/// A component attached to a game object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Component {
    /// File ID of the component's document.
    pub file_id: i64,
    /// Unity class ID, or `None` when the file has no such document.
    pub class_id: Option<u32>,
}

/// One node of the hierarchy.
#[derive(Debug, Clone, PartialEq)]
pub struct SceneObject {
    /// File ID of the `GameObject` (or `PrefabInstance`) document.
    pub file_id: i64,
    /// Name. Empty for a prefab instance whose name is not overridden in the scene.
    pub name: String,
    /// `m_IsActive`.
    pub active: bool,
    /// Layer index.
    pub layer: u32,
    /// Tag, `Untagged` by default.
    pub tag: String,
    /// Game object or prefab instance.
    pub kind: ObjectKind,
    /// Local transform. For a prefab instance it is assembled from the instance's overrides of
    /// its root transform (`None` when the scene overrides none of them); `None` also for a game
    /// object without a transform.
    pub transform: Option<TransformData>,
    /// Parent, `None` for roots.
    pub parent: Option<ObjectId>,
    /// Children in sibling order.
    pub children: Vec<ObjectId>,
    /// Components in file order (the transform comes first in Unity's files).
    pub components: Vec<Component>,
    /// `MeshFilter.m_Mesh` of a game object (or its skinned renderer's mesh).
    pub mesh: Option<AssetRef>,
    /// The mesh or skinned mesh renderer of a game object.
    pub renderer: Option<RendererState>,
    /// For a prefab instance: the overrides that affect drawing.
    pub overrides: Option<Box<Overrides>>,
}

/// Something in the file that did not add up. Loading continues; the object is kept as-is.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SceneIssue {
    /// A game object has no `Transform` document pointing at it.
    #[error("game object {file_id} has no transform")]
    GameObjectWithoutTransform {
        /// File ID of the game object.
        file_id: i64,
    },
    /// An object's parent transform does not exist in the file; it is shown as a root.
    #[error("object {file_id} refers to missing parent transform {parent}")]
    UnknownParent {
        /// File ID of the object.
        file_id: i64,
        /// The transform file ID that was not found.
        parent: i64,
    },
    /// `SceneRoots` names a transform that does not exist.
    #[error("scene roots list refers to missing transform {transform}")]
    UnknownRoot {
        /// The transform file ID that was not found.
        transform: i64,
    },
    /// The parent links form a loop; the loop was cut at this object.
    #[error("parent links of object {file_id} form a cycle; shown as a root")]
    Cycle {
        /// File ID of the object that became a root.
        file_id: i64,
    },
    /// A field could not be parsed; the default was used.
    #[error("object {file_id}: cannot read {field}")]
    BadValue {
        /// File ID of the document.
        file_id: i64,
        /// Field name.
        field: &'static str,
    },
}

/// A light of the scene with the object it sits on.
#[derive(Debug, Clone, PartialEq)]
pub struct SceneLight {
    /// The object, `None` when its game object is not in the scene file.
    pub object: Option<ObjectId>,
    /// The component.
    pub light: crate::lighting::Light,
}

/// The scene's `RenderSettings` (`None` in a file without one) and its lights. Lights inside
/// prefab instances are not included.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SceneLighting {
    /// `RenderSettings`.
    pub settings: Option<crate::lighting::RenderSettings>,
    /// `Light` components of the scene's own game objects.
    pub lights: Vec<SceneLight>,
    /// URP `Volume` components, with their object (`None` when it is not in the scene file).
    pub volumes: Vec<(Option<ObjectId>, crate::volume::SceneVolume)>,
}

/// What loading did, for the status bar and for honest measurements.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LoadStats {
    /// Documents in the file.
    pub documents: u64,
    /// `GameObject` documents.
    pub game_objects: u64,
    /// `Transform` and `RectTransform` documents (including stripped ones).
    pub transforms: u64,
    /// `PrefabInstance` documents.
    pub prefab_instances: u64,
    /// Bytes read from the file.
    pub bytes: u64,
    /// Issues beyond the stored limit (not kept in [`Scene::issues`]).
    pub issues_dropped: u64,
    /// Wall-clock time to read and assemble.
    pub duration: Duration,
}

/// A loaded scene. Immutable; the editor shares it behind an `Arc`.
#[derive(Debug, Clone, PartialEq)]
pub struct Scene {
    name: String,
    lighting: SceneLighting,
    objects: Vec<SceneObject>,
    roots: Vec<ObjectId>,
    issues: Vec<SceneIssue>,
    stats: LoadStats,
    world: Vec<Mat4>,
    active: Vec<bool>,
}

impl Scene {
    pub(crate) fn new(
        name: String,
        objects: Vec<SceneObject>,
        roots: Vec<ObjectId>,
        issues: Vec<SceneIssue>,
        stats: LoadStats,
        world: Vec<Mat4>,
    ) -> Self {
        debug_assert_eq!(objects.len(), world.len());
        let active = effective_active(&objects, &roots);
        Self {
            name,
            lighting: SceneLighting::default(),
            objects,
            roots,
            issues,
            stats,
            world,
            active,
        }
    }

    /// Whether an object is active in the hierarchy: its own `m_IsActive` and every ancestor's.
    /// Unity does not draw or update an object below an inactive parent even when the object's own
    /// flag is set. `false` for an unknown id.
    pub fn is_active(&self, id: ObjectId) -> bool {
        self.active.get(id).copied().unwrap_or(false)
    }

    /// The scene's lighting setup.
    pub(crate) fn with_lighting(mut self, lighting: SceneLighting) -> Self {
        self.lighting = lighting;
        self
    }

    /// `RenderSettings` and the lights.
    pub fn lighting(&self) -> &SceneLighting {
        &self.lighting
    }

    /// The sun: the light `RenderSettings.m_Sun` names, else the first enabled directional light
    /// of an active object (what URP picks as the main light).
    pub fn sun(&self) -> Option<&SceneLight> {
        use crate::lighting::LightType;
        let named = self
            .lighting
            .settings
            .as_ref()
            .and_then(|s| s.sun)
            .and_then(|id| self.lighting.lights.iter().find(|l| l.light.file_id == id));
        named.or_else(|| {
            self.lighting.lights.iter().find(|l| {
                l.light.enabled
                    && l.light.kind == LightType::Directional
                    && l.object.is_some_and(|o| self.is_active(o))
            })
        })
    }

    /// Scene name (the file name without extension).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Every object, in file order. Indexed by [`ObjectId`].
    pub fn objects(&self) -> &[SceneObject] {
        &self.objects
    }

    /// One object.
    pub fn object(&self, id: ObjectId) -> Option<&SceneObject> {
        self.objects.get(id)
    }

    /// Local-to-world matrix of an object: its parents' transforms applied down the hierarchy.
    /// Objects without a known transform (a prefab instance with no position override) count as
    /// the identity relative to their parent.
    pub fn world_matrix(&self, id: ObjectId) -> Option<&Mat4> {
        self.world.get(id)
    }

    /// World position of an object.
    pub fn world_position(&self, id: ObjectId) -> Option<[f32; 3]> {
        self.world.get(id).map(Mat4::translation)
    }

    /// Root objects in scene order.
    pub fn roots(&self) -> &[ObjectId] {
        &self.roots
    }

    /// Problems found while loading (at most a bounded number are kept).
    pub fn issues(&self) -> &[SceneIssue] {
        &self.issues
    }

    /// Load statistics.
    pub fn stats(&self) -> &LoadStats {
        &self.stats
    }

    /// Number of objects in the hierarchy.
    pub fn len(&self) -> usize {
        self.objects.len()
    }

    /// Whether the scene has no objects.
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }

    /// Number of objects below `id`, not counting `id` itself.
    pub fn descendant_count(&self, id: ObjectId) -> usize {
        let mut count = 0;
        let mut stack: Vec<ObjectId> = self
            .objects
            .get(id)
            .map(|o| o.children.clone())
            .unwrap_or_default();
        while let Some(next) = stack.pop() {
            count += 1;
            if let Some(object) = self.objects.get(next) {
                stack.extend_from_slice(&object.children);
            }
        }
        count
    }
}

/// `active` of every object combined with its ancestors', parents before children.
fn effective_active(objects: &[SceneObject], roots: &[ObjectId]) -> Vec<bool> {
    let mut active = vec![false; objects.len()];
    let mut stack: Vec<ObjectId> = Vec::new();
    for &root in roots {
        active[root] = objects[root].active;
        stack.push(root);
    }
    while let Some(parent) = stack.pop() {
        let parent_active = active[parent];
        for &child in &objects[parent].children {
            active[child] = parent_active && objects[child].active;
            stack.push(child);
        }
    }
    active
}
