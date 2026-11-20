//! First pass: pull the few fields the hierarchy needs out of each interesting document.

use crate::classes;
use crate::model::{SceneIssue, TransformData};
use crate::prefab::{AssetRef, Overrides, RendererState, parse_modification};
use crate::unity_yaml::{
    Field, Header, fields, file_id_of, flow_value, parse_vec3, parse_vec4, unquote,
};
use std::collections::HashMap;

/// How many issues are stored; the rest are only counted.
const MAX_STORED_ISSUES: usize = 1000;

pub(super) struct RawGameObject {
    pub file_id: i64,
    pub name: String,
    pub active: bool,
    pub layer: u32,
    pub tag: String,
    pub components: Vec<i64>,
}

pub(super) struct RawTransform {
    pub file_id: i64,
    pub game_object: i64,
    pub father: i64,
    pub children: Vec<i64>,
    pub data: TransformData,
}

/// One override of a transform value on a prefab instance (`m_LocalPosition.x` and friends).
pub(super) struct TransformMod {
    /// File ID of the transform inside the source prefab.
    pub target: i64,
    /// 0-2 position xyz, 3-6 rotation xyzw, 7-9 scale xyz.
    pub slot: u8,
    pub value: f32,
}

/// A `stripped` transform: stands for a transform that lives inside a prefab.
pub(super) struct StrippedTransform {
    pub prefab: i64,
    /// The transform's file ID inside the source prefab.
    pub source: i64,
}

pub(super) struct RawPrefab {
    pub file_id: i64,
    pub overrides: Overrides,
    pub transform_mods: Vec<TransformMod>,
    pub name: Option<String>,
    pub source_guid: Option<String>,
    /// The transform this instance is attached to (0 = scene root).
    pub parent: i64,
}

#[derive(Default)]
pub(super) struct Collected {
    pub game_objects: Vec<RawGameObject>,
    pub transforms: Vec<RawTransform>,
    pub prefabs: Vec<RawPrefab>,
    /// `MeshFilter`s: game object -> mesh.
    pub filters: HashMap<i64, Option<AssetRef>>,
    /// Renderers: game object -> (state, own mesh for skinned renderers).
    pub renderers: HashMap<i64, (RendererState, Option<AssetRef>)>,
    /// Stripped transform file ID -> the prefab instance it belongs to.
    pub stripped_transforms: HashMap<i64, StrippedTransform>,
    /// Class of every document, to name components.
    pub class_of: HashMap<i64, u32>,
    /// `SceneRoots` order, if the file has one.
    pub roots: Option<Vec<i64>>,
    pub issues: Vec<SceneIssue>,
    pub issues_dropped: u64,
    pub render_settings: Option<crate::lighting::RenderSettings>,
    pub lights: Vec<crate::lighting::Light>,
    pub volumes: Vec<crate::volume::SceneVolume>,
    pub documents: u64,
    pub transform_documents: u64,
}

impl Collected {
    pub fn note_document(&mut self, header: Header) {
        self.documents += 1;
        self.class_of.insert(header.file_id, header.class_id);
    }

    pub fn add(&mut self, header: Header, body: &str) {
        match header.class_id {
            classes::GAME_OBJECT => self.game_object(header.file_id, body),
            classes::TRANSFORM | classes::RECT_TRANSFORM => self.transform(header, body),
            classes::PREFAB_INSTANCE => self.prefab_instance(header.file_id, body),
            classes::MESH_FILTER => self.mesh_filter(body),
            classes::MESH_RENDERER | classes::SKINNED_MESH_RENDERER => self.renderer(body),
            classes::SCENE_ROOTS => self.scene_roots(body),
            classes::RENDER_SETTINGS => {
                self.render_settings = Some(crate::lighting::parse_render_settings(body));
            }
            classes::MONO_BEHAVIOUR => {
                // Only URP volumes are kept; the script is known from the body.
                if body.contains(crate::volume::VOLUME_SCRIPT_GUID)
                    && let Some(volume) = crate::volume::parse_volume(body)
                {
                    self.volumes.push(volume);
                }
            }
            classes::LIGHT => self
                .lights
                .push(crate::lighting::parse_light(header.file_id, body)),
            _ => {}
        }
    }

    pub fn issue(&mut self, issue: SceneIssue) {
        if self.issues.len() < MAX_STORED_ISSUES {
            self.issues.push(issue);
        } else {
            self.issues_dropped += 1;
        }
    }

    fn game_object(&mut self, file_id: i64, body: &str) {
        let mut object = RawGameObject {
            file_id,
            name: String::new(),
            active: true,
            layer: 0,
            tag: "Untagged".to_owned(),
            components: Vec::new(),
        };
        for field in fields(body) {
            match field.key {
                "m_Name" => object.name = unquote(field.value).into_owned(),
                "m_IsActive" => object.active = field.value != "0",
                "m_Layer" => match field.value.parse() {
                    Ok(layer) => object.layer = layer,
                    Err(_) => self.issue(SceneIssue::BadValue {
                        file_id,
                        field: "m_Layer",
                    }),
                },
                "m_TagString" => object.tag = unquote(field.value).into_owned(),
                "m_Component" => object.components = field.items().filter_map(file_id_of).collect(),
                _ => {}
            }
        }
        self.game_objects.push(object);
    }

    fn transform(&mut self, header: Header, body: &str) {
        self.transform_documents += 1;
        if header.stripped {
            // Only the links to the prefab instance and to the source transform matter.
            let (mut prefab, mut source) = (None, 0);
            for field in fields(body) {
                match field.key {
                    "m_PrefabInstance" => prefab = file_id_of(field.value),
                    "m_CorrespondingSourceObject" => source = file_id_of(field.value).unwrap_or(0),
                    _ => {}
                }
            }
            if let Some(prefab) = prefab {
                self.stripped_transforms
                    .insert(header.file_id, StrippedTransform { prefab, source });
            }
            return;
        }

        let file_id = header.file_id;
        let mut raw = RawTransform {
            file_id,
            game_object: 0,
            father: 0,
            children: Vec::new(),
            data: TransformData {
                is_rect: header.class_id == classes::RECT_TRANSFORM,
                ..TransformData::default()
            },
        };
        for field in fields(body) {
            match field.key {
                "m_GameObject" => raw.game_object = file_id_of(field.value).unwrap_or(0),
                "m_Father" => raw.father = file_id_of(field.value).unwrap_or(0),
                "m_Children" => raw.children = field.items().filter_map(file_id_of).collect(),
                "m_LocalPosition" => match parse_vec3(field.value) {
                    Some(v) => raw.data.position = v,
                    None => self.bad_value(file_id, "m_LocalPosition"),
                },
                "m_LocalRotation" => match parse_vec4(field.value) {
                    Some(v) => raw.data.rotation = v,
                    None => self.bad_value(file_id, "m_LocalRotation"),
                },
                "m_LocalScale" => match parse_vec3(field.value) {
                    Some(v) => raw.data.scale = v,
                    None => self.bad_value(file_id, "m_LocalScale"),
                },
                _ => {}
            }
        }
        self.transforms.push(raw);
    }

    fn bad_value(&mut self, file_id: i64, field: &'static str) {
        self.issue(SceneIssue::BadValue { file_id, field });
    }

    fn mesh_filter(&mut self, body: &str) {
        let (mut go, mut mesh) = (0, None);
        for field in fields(body) {
            match field.key {
                "m_GameObject" => go = file_id_of(field.value).unwrap_or(0),
                "m_Mesh" => mesh = AssetRef::parse(field.value),
                _ => {}
            }
        }
        self.filters.insert(go, mesh);
    }

    fn renderer(&mut self, body: &str) {
        let (mut go, mut mesh) = (0, None);
        let mut state = RendererState::default();
        for field in fields(body) {
            match field.key {
                "m_GameObject" => go = file_id_of(field.value).unwrap_or(0),
                "m_Enabled" => state.enabled = field.value != "0",
                "m_Mesh" => mesh = AssetRef::parse(field.value),
                "m_Materials" => state.materials = field.items().map(AssetRef::parse).collect(),
                _ => {}
            }
        }
        self.renderers.insert(go, (state, mesh));
    }

    fn prefab_instance(&mut self, file_id: i64, body: &str) {
        let mut prefab = RawPrefab {
            file_id,
            overrides: Overrides::default(),
            transform_mods: Vec::new(),
            name: None,
            source_guid: None,
            parent: 0,
        };
        for field in fields(body) {
            match field.key {
                "m_SourcePrefab" => {
                    prefab.source_guid = flow_value(field.value, "guid").map(str::to_owned);
                }
                "m_Modification" => {
                    read_modification(field, &mut prefab);
                    prefab.overrides = parse_modification(field.block()).1;
                }
                _ => {}
            }
        }
        self.prefabs.push(prefab);
    }

    fn scene_roots(&mut self, body: &str) {
        for field in fields(body) {
            if field.key == "m_Roots" {
                self.roots = Some(field.items().filter_map(file_id_of).collect());
            }
        }
    }
}

/// `m_Modification` holds the parent transform and the property overrides. From the overrides
/// this keeps the transform values (position, rotation, scale) and the first `m_Name`, taken as
/// the instance's name; without the source prefab loaded the name is a best effort (it can
/// belong to a nested object of the prefab).
fn read_modification(field: Field<'_>, prefab: &mut RawPrefab) {
    let mut target: Option<i64> = None;
    let mut slot: Option<u8> = None;
    let mut name_next = false;
    for line in field.block().lines().map(str::trim) {
        if let Some(rest) = line.strip_prefix("m_TransformParent:") {
            prefab.parent = file_id_of(rest).unwrap_or(0);
        } else if let Some(rest) = line.strip_prefix("- target:") {
            target = file_id_of(rest);
            slot = None;
            name_next = false;
        } else if let Some(path) = line.strip_prefix("propertyPath:") {
            let path = unquote(path);
            slot = transform_slot(&path);
            name_next = prefab.name.is_none() && path == "m_Name";
        } else if let Some(value) = line.strip_prefix("value:") {
            if std::mem::take(&mut name_next) {
                prefab.name = Some(unquote(value).into_owned());
            }
            if let (Some(target), Some(slot)) = (target, slot.take())
                && let Ok(value) = value.trim().parse::<f32>()
            {
                prefab.transform_mods.push(TransformMod {
                    target,
                    slot,
                    value,
                });
            }
        }
    }
}

fn transform_slot(path: &str) -> Option<u8> {
    Some(match path {
        "m_LocalPosition.x" => 0,
        "m_LocalPosition.y" => 1,
        "m_LocalPosition.z" => 2,
        "m_LocalRotation.x" => 3,
        "m_LocalRotation.y" => 4,
        "m_LocalRotation.z" => 5,
        "m_LocalRotation.w" => 6,
        "m_LocalScale.x" => 7,
        "m_LocalScale.y" => 8,
        "m_LocalScale.z" => 9,
        _ => return None,
    })
}
