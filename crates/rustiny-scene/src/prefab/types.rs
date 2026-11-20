//! Data shared by the scene loader and the prefab expander.

use crate::model::TransformData;
use crate::unity_yaml::{file_id_of, flow_value};

/// A reference to an asset (`{fileID: 4300000, guid: ..., type: 2}`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct AssetRef {
    /// File ID of the object inside the asset.
    pub file_id: i64,
    /// GUID of the asset.
    pub guid: String,
}

impl AssetRef {
    /// Read `{fileID: N, guid: G, ...}`. `None` for a null reference (`fileID: 0`) or one without
    /// a GUID (an object of the same file).
    pub fn parse(text: &str) -> Option<Self> {
        let file_id = file_id_of(text)?;
        let guid = flow_value(text, "guid")?;
        (file_id != 0 && !guid.is_empty()).then(|| Self {
            file_id,
            guid: guid.to_owned(),
        })
    }
}

/// The parts of a game object an override can name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Part {
    /// The `GameObject` (`m_Name`, `m_IsActive`).
    GameObject,
    /// The `Transform` (`m_LocalPosition`, ...).
    Transform,
    /// The `MeshFilter` (`m_Mesh`).
    Filter,
    /// The renderer (`m_Materials`, `m_Enabled`).
    Renderer,
}

impl Part {
    pub(super) fn index(self) -> usize {
        self as usize
    }

    /// The part a property path belongs to, `None` for properties that do not matter for drawing.
    pub fn of_property(path: &str) -> Option<Part> {
        Some(match path {
            "m_Name" | "m_IsActive" => Part::GameObject,
            "m_Mesh" => Part::Filter,
            "m_Enabled" => Part::Renderer,
            p if p.starts_with("m_LocalPosition.")
                || p.starts_with("m_LocalRotation.")
                || p.starts_with("m_LocalScale.") =>
            {
                Part::Transform
            }
            p if p.starts_with("m_Materials.Array.data[") => Part::Renderer,
            _ => return None,
        })
    }
}

/// One property override of a prefab instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Modification {
    /// File ID of the overridden object in the space of the instanced prefab.
    pub target: i64,
    /// The property, e.g. `m_LocalPosition.x` or `m_Materials.Array.data[2]`.
    pub path: String,
    /// The new value for plain properties.
    pub value: String,
    /// The new value for references (`m_Mesh`, material slots).
    pub reference: Option<AssetRef>,
}

/// The overrides of one instance that affect what is drawn.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Overrides {
    /// Property overrides, in file order.
    pub mods: Vec<Modification>,
    /// IDs of removed components and game objects.
    pub removed: Vec<i64>,
}

/// A renderer's state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RendererState {
    /// `m_Enabled`.
    pub enabled: bool,
    /// `m_Materials`: one entry per slot (`None` for an empty or unresolvable slot). An empty list
    /// means "the model's own materials".
    pub materials: Vec<Option<AssetRef>>,
}

impl Default for RendererState {
    fn default() -> Self {
        Self {
            enabled: true,
            materials: Vec::new(),
        }
    }
}

/// A game object of an expanded prefab.
#[derive(Debug, Clone)]
pub struct Node {
    /// `m_Name`.
    pub name: String,
    /// `m_IsActive` (of this object only).
    pub active: bool,
    /// Local transform.
    pub transform: TransformData,
    /// Index of the parent in the same list (parents come first); `None` for roots.
    pub parent: Option<usize>,
    /// `MeshFilter.m_Mesh` (or the skinned renderer's mesh).
    pub mesh: Option<AssetRef>,
    /// GUID of the model file this node stands for; its flattened mesh is used when `mesh` is
    /// not overridden.
    pub model: Option<String>,
    /// The renderer, when the object has one.
    pub renderer: Option<RendererState>,
    /// Removed by an override (the object and everything below it is gone).
    pub removed: bool,
    pub(super) ids: [i64; 4],
    pub(super) xor: i64,
    pub(super) wildcard: bool,
}

impl Node {
    pub(super) fn new(name: String) -> Self {
        Self {
            name,
            active: true,
            transform: TransformData::default(),
            parent: None,
            mesh: None,
            model: None,
            renderer: None,
            removed: false,
            ids: [0; 4],
            xor: 0,
            wildcard: false,
        }
    }

    /// The node's ID for one part, in the space of the prefab it was read from.
    pub(super) fn id(&self, part: Part) -> i64 {
        self.ids[part.index()]
    }
}

/// Whether a file extension is a 3D model file that acts as a prefab.
pub fn is_model_extension(extension: &str) -> bool {
    ["fbx", "obj", "blend", "dae", "3ds", "dxf"]
        .iter()
        .any(|e| e.eq_ignore_ascii_case(extension))
}
