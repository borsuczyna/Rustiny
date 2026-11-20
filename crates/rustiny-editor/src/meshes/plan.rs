//! From a scene to what the GPU draws: instances grouped by model, plus proxy boxes for the rest.

use rustiny_render::{GpuInstance, MeshGroup};
use rustiny_scene::prefab::{Library, compose_world};
use rustiny_scene::{
    AssetRef, Mat4, ObjectId, ObjectKind, RendererState, Scene, SceneObject, classes,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// What a mesh source is stored as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceKind {
    /// A model file (`.fbx`), imported as one flattened mesh.
    Model,
    /// A Unity `Mesh` asset (`.asset`).
    MeshAsset,
}

/// A material a renderer names: a `.mat` file, or a `Material` object stored in an `.asset` file
/// (terrain tiles keep theirs next to their mesh).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MaterialRef {
    /// The `.mat` or `.asset` file.
    pub path: PathBuf,
    /// The object inside an `.asset`; `None` for a `.mat` file.
    pub file_id: Option<i64>,
}

/// A mesh to import, identified by the key its instances use. Instances of one mesh with
/// different renderer materials are different sources.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshSource {
    /// Key shared with the [`MeshGroup`] of its instances and with the renderer.
    pub key: u32,
    /// The model or mesh asset file.
    pub path: PathBuf,
    /// For a renderer that names one mesh of a model file: that mesh's `fileID`, which Unity
    /// derives from the node's name. `None`: the whole model, flattened.
    pub part: Option<i64>,
    /// How the file is read.
    pub kind: SourceKind,
    /// The renderer's materials by slot. Empty: the model's own material names are matched to
    /// `.mat` files instead.
    pub materials: Vec<Option<MaterialRef>>,
}

/// What the scene needs drawn.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct MeshPlan {
    /// Distinct meshes, in key order.
    pub sources: Vec<MeshSource>,
    /// Instances per mesh, ready for the renderer.
    pub groups: Vec<MeshGroup>,
    /// Proxy boxes for objects that have no mesh (and hidden ones, so a selection still shows).
    pub boxes: Vec<GpuInstance>,
    /// For each key, the scene object of each instance (same order as the group's instances).
    pub objects_of: Vec<Vec<u32>>,
    /// For each key, the world matrix of each instance (same order).
    pub worlds_of: Vec<Vec<Mat4>>,
    /// Instances of lower-detail variants that were left out.
    pub lods_skipped: usize,
    /// Prefab instances that were opened and expanded.
    pub prefabs_expanded: usize,
    /// Prefab instances whose source could not be opened.
    pub prefabs_unreadable: usize,
}

/// The level of detail a model file name asks for: `Name_LOD2.fbx` is 2. `None` for names without
/// the suffix (those are drawn as the highest detail).
pub fn lod_of(file_name: &str) -> Option<u32> {
    let stem = file_name
        .rsplit_once('.')
        .map_or(file_name, |(stem, _)| stem);
    let (_, digits) = stem.rsplit_once("_LOD")?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

/// The level a node name asks for: `LOD1` is 1.
fn node_lod(name: &str) -> Option<u32> {
    let digits = name.strip_prefix("LOD")?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

fn kind_of(path: &Path) -> Option<SourceKind> {
    let extension = path.extension()?.to_str()?;
    if extension.eq_ignore_ascii_case("asset") {
        Some(SourceKind::MeshAsset)
    } else if extension.eq_ignore_ascii_case("fbx") {
        Some(SourceKind::Model)
    } else {
        None
    }
}

/// Whether an object shows something by itself when expansion did not give it meshes: a prefab
/// instance that could not be opened, or a game object with a renderer. Empty game objects
/// (managers, colliders, area markers) draw nothing in Unity's Scene view either, so their proxy
/// box is hidden unless selected.
fn draws_something(object: &SceneObject) -> bool {
    match object.kind {
        ObjectKind::PrefabInstance { .. } => true,
        ObjectKind::GameObject => object.components.iter().any(|c| {
            matches!(
                c.class_id,
                Some(
                    classes::MESH_RENDERER
                        | classes::SKINNED_MESH_RENDERER
                        | classes::SPRITE_RENDERER
                )
            )
        }),
    }
}

fn box_color(kind: &ObjectKind) -> [f32; 3] {
    match kind {
        ObjectKind::PrefabInstance { .. } => [1.0, 0.55, 0.2],
        ObjectKind::GameObject => [0.62, 0.7, 0.85],
    }
}

/// The proxy box of one object: its world matrix, a colour by kind, `visible` or hidden unless
/// selected.
pub fn proxy_box(scene: &Scene, id: ObjectId, visible: bool) -> GpuInstance {
    let model = scene.world_matrix(id).map_or(Mat4::IDENTITY.0, |m| m.0);
    let color = scene
        .object(id)
        .map_or([0.8; 3], |object| box_color(&object.kind));
    GpuInstance::new(model, color, visible).with_object(u32::try_from(id).unwrap_or(u32::MAX))
}

type GroupKey = (PathBuf, SourceKind, Option<i64>, Vec<Option<MaterialRef>>);

/// Collects mesh instances into groups.
struct Groups {
    plan: MeshPlan,
    key_of: HashMap<GroupKey, u32>,
}

/// What one renderer draws.
struct Drawing {
    path: PathBuf,
    part: Option<i64>,
    kind: SourceKind,
    materials: Vec<Option<MaterialRef>>,
}

impl Groups {
    fn add(&mut self, drawing: Drawing, world: &Mat4, object: u32, visible: bool) {
        let plan = &mut self.plan;
        let key = *self
            .key_of
            .entry((
                drawing.path.clone(),
                drawing.kind,
                drawing.part,
                drawing.materials.clone(),
            ))
            .or_insert_with(|| {
                let key = u32::try_from(plan.sources.len()).unwrap_or(u32::MAX);
                plan.sources.push(MeshSource {
                    key,
                    path: drawing.path,
                    part: drawing.part,
                    kind: drawing.kind,
                    materials: drawing.materials,
                });
                plan.groups.push(MeshGroup {
                    key,
                    instances: Vec::new(),
                });
                plan.objects_of.push(Vec::new());
                plan.worlds_of.push(Vec::new());
                key
            });
        plan.groups[key as usize]
            .instances
            .push(GpuInstance::new(world.0, [1.0; 3], visible).with_object(object));
        plan.objects_of[key as usize].push(object);
        plan.worlds_of[key as usize].push(*world);
    }
}

/// What one renderer draws: the mesh file and the `.mat` of each slot.
fn renderer_source(
    mesh: Option<&AssetRef>,
    model: Option<&str>,
    renderer: &RendererState,
    resolve: &impl Fn(&str) -> Option<PathBuf>,
) -> Option<Drawing> {
    let path = match (mesh, model) {
        (Some(mesh), _) => resolve(&mesh.guid)?,
        (None, Some(model)) => resolve(model)?,
        (None, None) => return None,
    };
    let kind = kind_of(&path)?;
    // A mesh reference into a model file is one mesh of it, in its own space.
    let part = mesh
        .filter(|_| kind == SourceKind::Model)
        .map(|mesh| mesh.file_id);
    let materials = renderer
        .materials
        .iter()
        .map(|m| {
            let m = m.as_ref()?;
            let path = resolve(&m.guid)?;
            let extension = path.extension()?.to_str()?.to_ascii_lowercase();
            match extension.as_str() {
                "mat" => Some(MaterialRef {
                    path,
                    file_id: None,
                }),
                "asset" => Some(MaterialRef {
                    path,
                    file_id: Some(m.file_id),
                }),
                _ => None,
            }
        })
        .collect();
    Some(Drawing {
        path,
        part,
        kind,
        materials,
    })
}

/// Whether a drawing is a coarser level of detail.
fn is_coarse(drawing: &Drawing, name: &str) -> bool {
    let file = drawing
        .path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    lod_of(file).is_some_and(|l| l > 0) || node_lod(name).is_some_and(|l| l > 0)
}

/// Decide what every object of `scene` is drawn as. `library` opens prefabs and models by GUID;
/// `resolve` maps a mesh or material GUID to its file.
///
/// * A prefab instance is expanded: the prefab's game objects with the scene's overrides applied
///   (mesh, materials, names, transforms). Each node with a mesh and an enabled renderer becomes
///   an instance of that mesh at its world matrix, owned by the instance's scene object.
/// * A game object with a `MeshFilter` and a renderer is an instance of its mesh.
/// * Meshes named `..._LOD1`, `..._LOD2` ... and nodes named `LOD1`... are the coarser variants
///   of `LOD0`; they sit on the same spot, so only the highest detail is drawn. This is a
///   file-name heuristic, not Unity's LOD group evaluation.
/// * Everything else keeps a proxy box: visible for a prefab that cannot be opened or an object
///   with a renderer whose mesh is not readable, hidden (visible only when selected) otherwise.
///
/// An object is drawn only while it and all its parents are active.
///
/// The work is spread over the machine's cores: the prefab and model files are read in parallel,
/// then the objects are expanded in parallel, and the results are put together in scene order,
/// so the plan does not depend on how many threads did it.
pub fn build_plan<R: Fn(&str) -> Option<PathBuf> + Sync>(
    scene: &Scene,
    library: &Library<R>,
    resolve: impl Fn(&str) -> Option<PathBuf> + Sync,
) -> MeshPlan {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get().clamp(1, 16));
    // Which files the scene instantiates, models apart from prefabs.
    let mut models: Vec<String> = Vec::new();
    let mut prefabs: Vec<String> = Vec::new();
    let mut model_instances: Vec<ObjectId> = Vec::new();
    {
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for (id, object) in scene.objects().iter().enumerate() {
            let ObjectKind::PrefabInstance {
                source_guid: Some(guid),
            } = &object.kind
            else {
                continue;
            };
            let is_model = resolve(guid)
                .as_deref()
                .and_then(kind_of)
                .is_some_and(|k| k == SourceKind::Model);
            if is_model {
                model_instances.push(id);
            }
            if seen.insert(guid) {
                if is_model {
                    models.push(guid.clone());
                } else {
                    prefabs.push(guid.clone());
                }
            }
        }
    }
    // Model files first. Expanding the scene's own instances of them teaches the library which
    // override IDs name which part of a single-node model; prefabs that nest the same models
    // rely on that, so they are read only afterwards. This keeps the result independent of the
    // order the threads run in.
    library.prefetch(&models, threads);
    parallel_map(&model_instances, threads, |&id| {
        if let Some(object) = scene.object(id)
            && let ObjectKind::PrefabInstance {
                source_guid: Some(guid),
            } = &object.kind
        {
            let overrides = object.overrides.as_deref().cloned().unwrap_or_default();
            let _ = library.expand(guid, &overrides);
        }
    });
    // Those expansions are made again below; count each override once.
    library.reset_override_counts();
    library.prefetch(&prefabs, threads);

    let ids: Vec<ObjectId> = (0..scene.len()).collect();
    let planned = parallel_map(&ids, threads, |&id| {
        plan_object(scene, library, &resolve, id)
    });

    let mut groups = Groups {
        plan: MeshPlan::default(),
        key_of: HashMap::new(),
    };
    for (id, object) in planned.into_iter().enumerate() {
        let object_id = u32::try_from(id).unwrap_or(u32::MAX);
        if object.expanded {
            groups.plan.prefabs_expanded += 1;
        }
        if object.unreadable {
            groups.plan.prefabs_unreadable += 1;
        }
        groups.plan.lods_skipped += object.coarse;
        let drawn = object.drawings.len();
        for (drawing, world, visible) in object.drawings {
            groups.add(drawing, &world, object_id, visible);
        }
        if drawn == 0 {
            groups.plan.boxes.push(object.proxy);
        }
    }
    groups.plan
}

/// `work` applied to every item on `threads` threads; the results in the order of `items`.
fn parallel_map<T: Sync, U: Send>(
    items: &[T],
    threads: usize,
    work: impl Fn(&T) -> U + Sync,
) -> Vec<U> {
    const CHUNK: usize = 256;
    let chunks: Vec<&[T]> = items.chunks(CHUNK).collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let done: std::sync::Mutex<Vec<(usize, Vec<U>)>> = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..threads.clamp(1, chunks.len().max(1)) {
            scope.spawn(|| {
                loop {
                    let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(chunk) = chunks.get(index) else {
                        break;
                    };
                    let results: Vec<U> = chunk.iter().map(&work).collect();
                    if let Ok(mut done) = done.lock() {
                        done.push((index, results));
                    }
                }
            });
        }
    });
    let mut done = done.into_inner().unwrap_or_else(|p| p.into_inner());
    done.sort_unstable_by_key(|(index, _)| *index);
    done.into_iter().flat_map(|(_, results)| results).collect()
}

/// What one scene object contributes to the plan.
struct ObjectPlan {
    /// Mesh instances: what, where, and whether it shows.
    drawings: Vec<(Drawing, Mat4, bool)>,
    /// Coarser variants left out.
    coarse: usize,
    /// A prefab instance that was expanded.
    expanded: bool,
    /// A prefab instance whose source could not be opened.
    unreadable: bool,
    /// The box kept when nothing is drawn.
    proxy: GpuInstance,
}

fn plan_object<R: Fn(&str) -> Option<PathBuf>>(
    scene: &Scene,
    library: &Library<R>,
    resolve: &impl Fn(&str) -> Option<PathBuf>,
    id: ObjectId,
) -> ObjectPlan {
    let mut out = ObjectPlan {
        drawings: Vec::new(),
        coarse: 0,
        expanded: false,
        unreadable: false,
        proxy: proxy_box(scene, id, false),
    };
    let Some(object) = scene.object(id) else {
        return out;
    };
    let active = scene.is_active(id);
    match &object.kind {
        ObjectKind::PrefabInstance {
            source_guid: Some(guid),
        } => {
            let overrides = object.overrides.as_deref().cloned().unwrap_or_default();
            if let Some(nodes) = library.expand(guid, &overrides) {
                out.expanded = true;
                let parent_world = object
                    .parent
                    .and_then(|p| scene.world_matrix(p))
                    .copied()
                    .unwrap_or(Mat4::IDENTITY);
                let world = compose_world(&nodes, &parent_world);
                // A node is live when it and every ancestor is active and not removed.
                let mut live: Vec<bool> = Vec::with_capacity(nodes.len());
                for node in &nodes {
                    let parent_live = node.parent.is_none_or(|p| live[p]);
                    live.push(parent_live && node.active && !node.removed);
                }
                for (i, node) in nodes.iter().enumerate() {
                    let Some(renderer) = &node.renderer else {
                        continue;
                    };
                    if node.removed {
                        continue;
                    }
                    let Some(drawing) = renderer_source(
                        node.mesh.as_ref(),
                        node.model.as_deref(),
                        renderer,
                        resolve,
                    ) else {
                        continue;
                    };
                    if is_coarse(&drawing, &node.name) {
                        out.coarse += 1;
                        continue;
                    }
                    let visible = active && live[i] && renderer.enabled;
                    out.drawings.push((drawing, world[i], visible));
                }
            } else {
                out.unreadable = true;
            }
        }
        ObjectKind::GameObject => {
            if let (Some(renderer), Some(mesh)) = (&object.renderer, &object.mesh)
                && let Some(drawing) = renderer_source(Some(mesh), None, renderer, resolve)
            {
                if is_coarse(&drawing, &object.name) {
                    out.coarse += 1;
                } else {
                    let world = scene.world_matrix(id).copied().unwrap_or(Mat4::IDENTITY);
                    out.drawings
                        .push((drawing, world, active && renderer.enabled));
                }
            }
        }
        ObjectKind::PrefabInstance { .. } => {}
    }
    if out.drawings.is_empty() {
        // Nothing is drawn for this object: keep a box, visible only when it should show
        // something that could not be drawn.
        let show = active && out.coarse == 0 && !out.expanded && draws_something(object);
        out.proxy = proxy_box(scene, id, show);
    }
    out
}

/// Objects the plan draws as meshes (every instance of every group).
impl MeshPlan {
    /// Number of mesh instances.
    pub fn mesh_instances(&self) -> usize {
        self.groups.iter().map(|g| g.instances.len()).sum()
    }

    /// The model file objects use, by object id (only objects that have one).
    pub fn source_of(&self, object: ObjectId) -> Option<&MeshSource> {
        let object = u32::try_from(object).ok()?;
        self.objects_of
            .iter()
            .position(|objects| objects.contains(&object))
            .and_then(|key| self.sources.get(key))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustiny_scene::read_scene;
    use std::io::Cursor;

    const MODEL_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const MODEL_A_LOD1: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const TEXTURE: &str = "cccccccccccccccccccccccccccccccc";

    fn prefab(id: i64, guid: &str, x: f32) -> String {
        prefab_under(id, guid, x, 0)
    }

    fn prefab_under(id: i64, guid: &str, x: f32, parent: i64) -> String {
        format!(
            "--- !u!1001 &{id}\nPrefabInstance:\n  m_Modification:\n    m_TransformParent: {{fileID: {parent}}}\n    m_Modifications:\n    - target: {{fileID: 5, guid: {guid}, type: 3}}\n      propertyPath: m_LocalPosition.x\n      value: {x}\n      objectReference: {{fileID: 0}}\n  m_SourcePrefab: {{fileID: 100100000, guid: {guid}, type: 3}}\n--- !u!4 &{stripped} stripped\nTransform:\n  m_CorrespondingSourceObject: {{fileID: 5, guid: {guid}, type: 3}}\n  m_PrefabInstance: {{fileID: {id}}}\n  m_PrefabAsset: {{fileID: 0}}\n",
            stripped = id + 1000
        )
    }

    fn game_object(id: i64, active: bool, father: i64, children: &[i64]) -> String {
        let tf = id + 1;
        let renderer = id + 2;
        let mut s = format!(
            "--- !u!1 &{id}\nGameObject:\n  m_Component:\n  - component: {{fileID: {tf}}}\n  - component: {{fileID: {renderer}}}\n  m_Name: G{id}\n  m_IsActive: {}\n--- !u!23 &{renderer}\nMeshRenderer:\n  m_GameObject: {{fileID: {id}}}\n",
            u8::from(active)
        );
        s.push_str(&format!(
            "--- !u!4 &{tf}\nTransform:\n  m_GameObject: {{fileID: {id}}}\n  m_LocalRotation: {{x: 0, y: 0, z: 0, w: 1}}\n  m_LocalPosition: {{x: 0, y: 0, z: 0}}\n  m_LocalScale: {{x: 1, y: 1, z: 1}}\n"
        ));
        if children.is_empty() {
            s.push_str("  m_Children: []\n");
        } else {
            s.push_str("  m_Children:\n");
            for c in children {
                s.push_str(&format!("  - {{fileID: {}}}\n", c + 1));
            }
        }
        s.push_str(&format!("  m_Father: {{fileID: {father}}}\n"));
        s
    }

    fn resolve(guid: &str) -> Option<PathBuf> {
        match guid {
            MODEL_A => Some("Models/Tower_LOD0.fbx".into()),
            MODEL_A_LOD1 => Some("Models/Tower_LOD1.fbx".into()),
            TEXTURE => Some("Textures/wall.png".into()),
            _ => None,
        }
    }

    fn plan_of(s: &Scene) -> MeshPlan {
        let library = Library::new(resolve);
        build_plan(s, &library, resolve)
    }

    fn scene(documents: &[String]) -> Scene {
        let mut text = String::from("%YAML 1.1\n%TAG !u! tag:unity3d.com,2011:\n");
        for d in documents {
            text.push_str(d);
        }
        read_scene(Cursor::new(text.into_bytes()), "T").unwrap()
    }

    #[test]
    fn lod_numbers_come_from_the_file_name() {
        assert_eq!(lod_of("Tower_LOD0.fbx"), Some(0));
        assert_eq!(lod_of("Tower_LOD12.fbx"), Some(12));
        assert_eq!(lod_of("db8_LOD2.FBX"), Some(2));
        assert_eq!(lod_of("Tower.fbx"), None);
        assert_eq!(lod_of("Tower_LOD.fbx"), None);
        assert_eq!(lod_of("Tower_LODx.fbx"), None);
        assert_eq!(
            lod_of("Tower_LOD1_old.fbx"),
            None,
            "the suffix must end the name"
        );
    }

    #[test]
    fn instances_of_one_model_share_a_group_and_keep_their_object_ids() {
        let s = scene(&[prefab(1, MODEL_A, 1.0), prefab(2, MODEL_A, 7.0)]);
        let plan = plan_of(&s);
        assert_eq!(plan.sources.len(), 1);
        assert_eq!(plan.sources[0].path, PathBuf::from("Models/Tower_LOD0.fbx"));
        assert_eq!(plan.groups.len(), 1);
        let instances = &plan.groups[0].instances;
        assert_eq!(instances.len(), 2);
        assert_eq!(
            (instances[0].object, instances[1].object),
            (0, 1),
            "object ids are scene object ids"
        );
        assert_eq!(instances[1].model[12], 7.0, "world position");
        assert!(plan.boxes.is_empty(), "mesh objects get no proxy box");
        assert_eq!(plan.objects_of, vec![vec![0, 1]]);
        assert_eq!(plan.mesh_instances(), 2);
        assert_eq!(plan.source_of(1).unwrap().key, 0);
        assert!(plan.source_of(5).is_none());
    }

    /// A game object "Wheel" with a mesh filter that refers to one mesh of a model file.
    fn wheel_with_model_mesh(guid: &str) -> Vec<String> {
        vec![format!(
            "--- !u!1 &1
GameObject:
  m_Component:
  - component: {{fileID: 2}}
  - component: {{fileID: 3}}
  - component: {{fileID: 4}}
  m_Name: Wheel
  m_IsActive: 1
--- !u!4 &2
Transform:
  m_GameObject: {{fileID: 1}}
  m_LocalRotation: {{x: 0, y: 0, z: 0, w: 1}}
  m_LocalPosition: {{x: 4, y: 0, z: 0}}
  m_LocalScale: {{x: 1, y: 1, z: 1}}
  m_Children: []
  m_Father: {{fileID: 0}}
--- !u!33 &3
MeshFilter:
  m_GameObject: {{fileID: 1}}
  m_Mesh: {{fileID: -4256789094099829062, guid: {guid}, type: 3}}
--- !u!23 &4
MeshRenderer:
  m_GameObject: {{fileID: 1}}
  m_Enabled: 1
  m_Materials: []
"
        )]
    }

    #[test]
    fn a_mesh_filter_on_a_model_draws_that_one_mesh_not_the_whole_file() {
        let plan = plan_of(&scene(&wheel_with_model_mesh(MODEL_A)));
        assert_eq!(plan.sources.len(), 1);
        assert_eq!(plan.sources[0].kind, SourceKind::Model);
        assert_eq!(
            plan.sources[0].part,
            Some(-4_256_789_094_099_829_062),
            "the mesh is looked up by its fileID"
        );
        assert_eq!(plan.groups[0].instances[0].model[12], 4.0);
        // A prefab instance of the same file is the model as a whole.
        let plan = plan_of(&scene(&[prefab(1, MODEL_A, 0.0)]));
        assert_eq!(plan.sources[0].part, None);
    }

    #[test]
    fn a_material_inside_an_asset_file_is_kept_with_its_object_id() {
        let asset = |guid: &str, id: i64| {
            Some(AssetRef {
                file_id: id,
                guid: guid.to_owned(),
            })
        };
        let resolve = |guid: &str| match guid {
            "11" => Some(PathBuf::from("Terrain/Tile.asset")),
            "22" => Some(PathBuf::from("Materials/Wall.mat")),
            "33" => Some(PathBuf::from("Textures/wall.png")),
            MODEL_A => Some(PathBuf::from("Models/Tower_LOD0.fbx")),
            _ => None,
        };
        let renderer = RendererState {
            enabled: true,
            materials: vec![
                asset("11", -77),
                asset("22", 2_100_000),
                asset("33", 2_800_000),
                None,
            ],
        };
        let drawing = renderer_source(None, Some(MODEL_A), &renderer, &resolve).unwrap();
        assert_eq!(
            drawing.materials,
            vec![
                Some(MaterialRef {
                    path: "Terrain/Tile.asset".into(),
                    file_id: Some(-77)
                }),
                Some(MaterialRef {
                    path: "Materials/Wall.mat".into(),
                    file_id: None
                }),
                None,
                None
            ]
        );
    }

    #[test]
    fn coarser_variants_are_skipped_but_stay_selectable() {
        let s = scene(&[prefab(1, MODEL_A, 0.0), prefab(2, MODEL_A_LOD1, 0.0)]);
        let plan = plan_of(&s);
        assert_eq!(plan.lods_skipped, 1);
        assert_eq!(plan.sources.len(), 1);
        assert_eq!(plan.boxes.len(), 1);
        assert_eq!(plan.boxes[0].object, 1);
        assert_eq!(plan.boxes[0].color[3], 0.0, "hidden unless selected");
    }

    #[test]
    fn other_objects_stay_boxes_and_inactive_ones_are_hidden() {
        let s = scene(&[
            game_object(10, true, 0, &[]),
            game_object(20, false, 0, &[30]),
            game_object(30, true, 21, &[]),
            game_object(40, true, 0, &[]).replace(
                "  - component: {fileID: 42}
",
                "",
            ),
            prefab(1, TEXTURE, 0.0),
            prefab(2, "dddddddddddddddddddddddddddddddd", 0.0),
        ]);
        let plan = plan_of(&s);
        assert!(
            plan.sources.is_empty(),
            "a texture or an unknown GUID is not a model"
        );
        assert_eq!(plan.boxes.len(), s.len(), "every object keeps a box");
        let alpha = |name: &str| {
            let id = s.objects().iter().position(|o| o.name == name).unwrap();
            plan.boxes
                .iter()
                .find(|b| b.object as usize == id)
                .unwrap()
                .color[3]
        };
        assert_eq!(alpha("G10"), 1.0);
        assert_eq!(alpha("G20"), 0.0, "inactive");
        assert_eq!(alpha("G30"), 0.0, "active itself but its parent is off");
        assert_eq!(alpha("G40"), 0.0, "no renderer: nothing to draw");
    }

    #[test]
    fn an_inactive_model_instance_is_hidden_not_dropped() {
        let s = scene(&[
            game_object(10, false, 0, &[]),
            prefab_under(1, MODEL_A, 0.0, 11),
            prefab(2, MODEL_A, 0.0),
        ]);
        let plan = plan_of(&s);
        let instances = &plan.groups[0].instances;
        assert_eq!(instances.len(), 2, "both are kept");
        let under_off = instances
            .iter()
            .find(|i| {
                s.object(i.object as usize)
                    .is_some_and(|o| o.parent.is_some())
            })
            .unwrap();
        assert_eq!(under_off.color[3], 0.0, "below an inactive object");
        let root = instances
            .iter()
            .find(|i| i.object != under_off.object)
            .unwrap();
        assert_eq!(root.color[3], 1.0, "a root prefab is active");
    }

    #[test]
    fn parallel_work_keeps_the_order_of_the_items() {
        let items: Vec<usize> = (0..10_000).collect();
        for threads in [1, 3, 16] {
            let out = parallel_map(&items, threads, |&i| i * 2);
            assert_eq!(out.len(), items.len());
            assert!(
                out.iter().enumerate().all(|(i, &v)| v == i * 2),
                "{threads} threads"
            );
        }
        assert!(parallel_map(&[] as &[usize], 4, |&i| i).is_empty());
    }

    #[test]
    fn the_lookup_runs_once_per_guid_for_prefab_sources() {
        let s = scene(&[
            prefab(1, MODEL_A, 0.0),
            prefab(2, MODEL_A, 1.0),
            prefab(3, MODEL_A, 2.0),
        ]);
        let calls = std::sync::atomic::AtomicUsize::new(0);
        let counting = |g: &str| {
            calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            resolve(g)
        };
        let library = Library::new(&counting);
        let plan = build_plan(&s, &library, resolve);
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "the library caches the source"
        );
        assert_eq!(plan.mesh_instances(), 3);
    }
}
