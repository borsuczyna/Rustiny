//! Opening prefab assets and applying overrides.

use super::overrides::parse_modification;
use super::types::{AssetRef, Node, Overrides, Part, RendererState, is_model_extension};
use crate::classes;
use crate::math::Mat4;
use crate::model::TransformData;
use crate::unity_yaml::{
    DocumentReader, fields, file_id_of, flow_value, parse_vec3, parse_vec4, unquote,
};
use rustiny_model::ImportSettings;
use rustiny_model::fbx::{self, ROOT_PATH, object_id};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Nesting deeper than this is treated as a cycle.
const MAX_DEPTH: usize = 16;

/// How many overrides found their target.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExpandStats {
    /// Overrides applied.
    pub applied: usize,
    /// Overrides whose target object could not be found.
    pub unmatched: usize,
    /// Prefab assets that could not be opened or read.
    pub unreadable: usize,
}

/// Opens prefab assets by GUID and expands instances of them. Templates are cached, so each file
/// is read once.
pub struct Library<R> {
    resolve: R,
    cache: Mutex<HashMap<String, Option<Arc<Vec<Node>>>>>,
    /// Model node IDs seen in overrides of direct instances, with the part they name.
    learned: Mutex<HashMap<i64, Part>>,
    applied: AtomicUsize,
    unmatched: AtomicUsize,
    unreadable: AtomicUsize,
}

impl<R> std::fmt::Debug for Library<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Library")
            .field("applied", &self.applied.load(Ordering::Relaxed))
            .field("unmatched", &self.unmatched.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

struct RawGo {
    id: i64,
    name: String,
    active: bool,
}

struct RawTransform {
    go: i64,
    father: i64,
    data: TransformData,
}

struct RawInstance {
    id: i64,
    source: String,
    parent: i64,
    overrides: Overrides,
}

#[derive(Default)]
struct Parsed {
    gos: Vec<RawGo>,
    transforms: HashMap<i64, RawTransform>,
    filters: HashMap<i64, (i64, Option<AssetRef>)>,
    renderers: HashMap<i64, (i64, RendererState, Option<AssetRef>)>,
    instances: Vec<RawInstance>,
}

impl<R: Fn(&str) -> Option<PathBuf>> Library<R> {
    /// A library that finds the file of a prefab or model GUID with `resolve`.
    pub fn new(resolve: R) -> Self {
        Self {
            resolve,
            cache: Mutex::new(HashMap::new()),
            learned: Mutex::new(HashMap::new()),
            applied: AtomicUsize::new(0),
            unmatched: AtomicUsize::new(0),
            unreadable: AtomicUsize::new(0),
        }
    }

    /// Forget the override counts so far (after expansions that were only made to learn from,
    /// whose results are made again). Unreadable files stay counted.
    pub fn reset_override_counts(&self) {
        self.applied.store(0, Ordering::Relaxed);
        self.unmatched.store(0, Ordering::Relaxed);
    }

    /// Counters so far.
    pub fn stats(&self) -> ExpandStats {
        ExpandStats {
            applied: self.applied.load(Ordering::Relaxed),
            unmatched: self.unmatched.load(Ordering::Relaxed),
            unreadable: self.unreadable.load(Ordering::Relaxed),
        }
    }

    /// The game objects an instance of `guid` with `overrides` stands for, parents first. `None`
    /// when the asset is not a prefab or model, or cannot be read.
    pub fn expand(&self, guid: &str, overrides: &Overrides) -> Option<Vec<Node>> {
        self.expand_at(guid, overrides, 0)
    }

    fn expand_at(&self, guid: &str, overrides: &Overrides, depth: usize) -> Option<Vec<Node>> {
        let template = self.template(guid, depth)?;
        let mut nodes = (*template).clone();
        self.apply(&mut nodes, overrides);
        Some(nodes)
    }

    fn template(&self, guid: &str, depth: usize) -> Option<Arc<Vec<Node>>> {
        if let Ok(cache) = self.cache.lock()
            && let Some(known) = cache.get(guid)
        {
            return known.clone();
        }
        let built = if depth > MAX_DEPTH {
            None
        } else {
            self.build(guid, depth).map(Arc::new)
        };
        if let Ok(mut cache) = self.cache.lock() {
            cache.insert(guid.to_owned(), built.clone());
        }
        built
    }

    fn build(&self, guid: &str, depth: usize) -> Option<Vec<Node>> {
        let path = (self.resolve)(guid)?;
        let extension = path.extension()?.to_str()?.to_owned();
        if is_model_extension(&extension) {
            if let Some(nodes) = model_nodes(guid, &path) {
                return Some(nodes);
            }
            let name = path
                .file_stem()
                .map_or_else(String::new, |s| s.to_string_lossy().into_owned());
            let mut node = Node::new(name);
            node.model = Some(guid.to_owned());
            node.renderer = Some(RendererState::default());
            node.wildcard = true;
            return Some(vec![node]);
        }
        if !extension.eq_ignore_ascii_case("prefab") {
            return None;
        }
        let parsed = match read_prefab(&path) {
            Ok(parsed) => parsed,
            Err(why) => {
                self.unreadable.fetch_add(1, Ordering::Relaxed);
                tracing::debug!("prefab {}: {why}", path.display());
                return None;
            }
        };
        Some(self.assemble(parsed, depth))
    }

    fn assemble(&self, parsed: Parsed, depth: usize) -> Vec<Node> {
        let mut nodes: Vec<Node> = Vec::with_capacity(parsed.gos.len());
        let mut transform_owner: HashMap<i64, usize> = HashMap::new();
        // (node, the transform it hangs from) to resolve once nested instances are in.
        let mut pending: Vec<(usize, i64)> = Vec::new();

        let mut filter_of: HashMap<i64, (i64, &Option<AssetRef>)> = HashMap::new();
        for (&id, (go, mesh)) in &parsed.filters {
            filter_of.insert(*go, (id, mesh));
        }
        let mut renderer_of: HashMap<i64, (i64, &RendererState, &Option<AssetRef>)> =
            HashMap::new();
        for (&id, (go, state, mesh)) in &parsed.renderers {
            renderer_of.insert(*go, (id, state, mesh));
        }
        let mut transform_of: HashMap<i64, (i64, &RawTransform)> = HashMap::new();
        for (&id, t) in &parsed.transforms {
            transform_of.insert(t.go, (id, t));
        }

        for go in &parsed.gos {
            let mut node = Node::new(go.name.clone());
            node.active = go.active;
            node.ids[Part::GameObject.index()] = go.id;
            if let Some(&(id, t)) = transform_of.get(&go.id) {
                node.ids[Part::Transform.index()] = id;
                node.transform = t.data;
                transform_owner.insert(id, nodes.len());
                if t.father != 0 {
                    pending.push((nodes.len(), t.father));
                }
            }
            if let Some(&(id, mesh)) = filter_of.get(&go.id) {
                node.ids[Part::Filter.index()] = id;
                node.mesh = mesh.clone();
            }
            if let Some(&(id, state, mesh)) = renderer_of.get(&go.id) {
                node.ids[Part::Renderer.index()] = id;
                node.renderer = Some(state.clone());
                if node.mesh.is_none() {
                    node.mesh = mesh.clone();
                }
            }
            nodes.push(node);
        }

        // Nested instances: their nodes join this prefab with IDs composed by the instance ID.
        for instance in &parsed.instances {
            let Some(mut sub) = self.expand_at(&instance.source, &instance.overrides, depth + 1)
            else {
                continue;
            };
            let base = nodes.len();
            for node in &mut sub {
                for id in &mut node.ids {
                    if *id != 0 {
                        *id ^= instance.id;
                    }
                }
                node.xor ^= instance.id;
                match node.parent {
                    Some(parent) => node.parent = Some(parent + base),
                    None => {
                        if instance.parent != 0 {
                            pending.push((nodes.len(), instance.parent));
                        }
                    }
                }
                nodes.push(node.clone());
            }
        }

        for (node_index, node) in nodes.iter().enumerate() {
            let id = node.ids[Part::Transform.index()];
            if id != 0 {
                transform_owner.entry(id).or_insert(node_index);
            }
        }
        for (node, father) in pending {
            if let Some(&parent) = transform_owner.get(&father)
                && parent != node
            {
                nodes[node].parent = Some(parent);
            }
        }
        parents_first(nodes)
    }

    /// Apply `overrides` to `nodes` in place.
    fn apply(&self, nodes: &mut [Node], overrides: &Overrides) {
        for m in &overrides.mods {
            let Some(part) = Part::of_property(&m.path) else {
                continue;
            };
            let Some(index) = self.find(nodes, m.target, part) else {
                self.unmatched.fetch_add(1, Ordering::Relaxed);
                continue;
            };
            self.applied.fetch_add(1, Ordering::Relaxed);
            let node = &mut nodes[index];
            match m.path.as_str() {
                "m_Name" => node.name.clone_from(&m.value),
                "m_IsActive" => node.active = m.value.trim() != "0",
                "m_Mesh" => {
                    node.mesh = m.reference.clone();
                    if m.reference.is_none() {
                        node.model = None;
                    }
                }
                "m_Enabled" => {
                    node.renderer
                        .get_or_insert_with(RendererState::default)
                        .enabled = m.value.trim() != "0";
                }
                path if path.starts_with("m_Materials.Array.data[") => {
                    let slot = path
                        .trim_start_matches("m_Materials.Array.data[")
                        .trim_end_matches(']')
                        .parse::<usize>()
                        .ok();
                    if let Some(slot) = slot.filter(|&s| s < 4096) {
                        let renderer = node.renderer.get_or_insert_with(RendererState::default);
                        if renderer.materials.len() <= slot {
                            renderer.materials.resize(slot + 1, None);
                        }
                        renderer.materials[slot].clone_from(&m.reference);
                    }
                }
                path => {
                    if let (Some(slot), Ok(value)) = (transform_slot(path), m.value.trim().parse())
                    {
                        set_transform(&mut node.transform, slot, value);
                    }
                }
            }
        }
        for &id in &overrides.removed {
            for part in [Part::Renderer, Part::Filter, Part::GameObject] {
                if let Some(index) = self.find(nodes, id, part) {
                    let node = &mut nodes[index];
                    match part {
                        Part::Renderer => node.renderer = None,
                        Part::Filter => {
                            node.mesh = None;
                            node.model = None;
                        }
                        _ => node.removed = true,
                    }
                    break;
                }
            }
        }
    }

    fn find(&self, nodes: &[Node], target: i64, part: Part) -> Option<usize> {
        let exact = nodes.iter().position(|n| {
            !n.wildcard
                && (n.id(part) == target
                    // A skinned renderer carries its own mesh.
                    || (part == Part::Filter && n.id(Part::Renderer) == target))
        });
        if exact.is_some() {
            return exact;
        }
        let mut learned = self.learned.lock().ok()?;
        let wild: Vec<usize> = nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.wildcard)
            .map(|(i, _)| i)
            .collect();
        for &i in &wild {
            let raw = target ^ nodes[i].xor;
            if nodes[i].xor == 0 {
                learned.insert(raw, part);
                return Some(i);
            }
            if learned.get(&raw) == Some(&part) {
                return Some(i);
            }
        }
        (wild.len() == 1).then(|| wild[0])
    }
}

impl<R: Fn(&str) -> Option<PathBuf> + Sync> Library<R> {
    /// Read the prefab or model files `guids` name on `threads` threads, so that later
    /// expansions find them cached. Nested prefabs are read as part of the prefab that holds them.
    pub fn prefetch(&self, guids: &[String], threads: usize) {
        let next = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..threads.clamp(1, guids.len().max(1)) {
                scope.spawn(|| {
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        let Some(guid) = guids.get(index) else {
                            break;
                        };
                        let _ = self.template(guid, 0);
                    }
                });
            }
        });
    }
}

/// How many nodes a binary FBX file has, counted without parsing it (`<name>\0\x01Model` is how
/// every model node's name is stored).
fn count_model_nodes(bytes: &[u8]) -> usize {
    const TAG: &[u8] = b"\0\x01Model";
    bytes.windows(TAG.len()).filter(|w| *w == TAG).count()
}

/// The game objects Unity makes of a model file with several nodes: one per node, with the
/// node's own transform and mesh and the IDs Unity gives them (`xxh64` of a type and a path under
/// `//RootNode/root`, see `rustiny_model::fbx::mesh_file_id`). A file with one top-level node
/// uses it as the prefab's root; with several, the root is an empty object named like the file.
///
/// `None` for a file with a single node (it stays one wildcard node, drawn as the flattened
/// model) or one that cannot be read.
fn model_nodes(guid: &str, path: &std::path::Path) -> Option<Vec<Node>> {
    if !path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("fbx"))
    {
        return None;
    }
    let bytes = std::fs::read(path).ok()?;
    if count_model_nodes(&bytes) < 2 {
        return None;
    }
    let mut meta = path.as_os_str().to_owned();
    meta.push(".meta");
    let settings = std::fs::read_to_string(&meta)
        .map(|text| ImportSettings::from_meta(&text))
        .unwrap_or_default();
    let listed = fbx::nodes(&bytes, settings).ok()?;
    if listed.is_empty() {
        return None;
    }
    let file_name = path
        .file_stem()
        .map_or_else(String::new, |s| s.to_string_lossy().into_owned());
    let tops = listed.iter().filter(|n| n.parent.is_none()).count();
    let synthetic = tops != 1;
    let shift = usize::from(synthetic);

    let mut nodes: Vec<Node> = Vec::with_capacity(listed.len() + shift);
    let mut paths: Vec<String> = Vec::with_capacity(listed.len() + shift);
    if synthetic {
        let mut root = Node::new(file_name.clone());
        root.ids[Part::GameObject.index()] = object_id("GameObject", ROOT_PATH);
        root.ids[Part::Transform.index()] = object_id("Transform", ROOT_PATH);
        nodes.push(root);
        paths.push(ROOT_PATH.to_owned());
    }
    for listed_node in &listed {
        let is_root = !synthetic && listed_node.parent.is_none();
        let node_path = if is_root {
            ROOT_PATH.to_owned()
        } else {
            let parent = listed_node.parent.map_or(0, |p| p + shift);
            format!("{}/{}", paths[parent], listed_node.name)
        };
        let mut node = Node::new(if is_root {
            file_name.clone()
        } else {
            listed_node.name.clone()
        });
        node.transform = TransformData {
            position: listed_node.position,
            rotation: listed_node.rotation,
            scale: listed_node.scale,
            is_rect: false,
        };
        node.parent = match listed_node.parent {
            Some(p) => Some(p + shift),
            None => synthetic.then_some(0),
        };
        node.ids[Part::GameObject.index()] = object_id("GameObject", &node_path);
        node.ids[Part::Transform.index()] = object_id("Transform", &node_path);
        if listed_node.has_mesh {
            node.ids[Part::Filter.index()] = object_id("MeshFilter", &node_path);
            node.ids[Part::Renderer.index()] = object_id("MeshRenderer", &node_path);
            node.mesh = Some(AssetRef {
                file_id: fbx::mesh_file_id(&listed_node.name),
                guid: guid.to_owned(),
            });
            node.renderer = Some(RendererState::default());
        }
        nodes.push(node);
        paths.push(node_path);
    }
    Some(nodes)
}

/// World matrices of expanded nodes: `root_world` is the matrix of the space the roots sit in.
pub fn compose_world(nodes: &[Node], root_world: &Mat4) -> Vec<Mat4> {
    let mut world: Vec<Mat4> = Vec::with_capacity(nodes.len());
    for node in nodes {
        let t = node.transform;
        let local = Mat4::from_trs(t.position, t.rotation, t.scale);
        let parent = node.parent.and_then(|p| world.get(p)).unwrap_or(root_world);
        world.push(parent.mul(&local));
    }
    world
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

fn set_transform(t: &mut TransformData, slot: u8, value: f32) {
    match slot {
        0..=2 => t.position[usize::from(slot)] = value,
        3..=6 => t.rotation[usize::from(slot - 3)] = value,
        _ => t.scale[usize::from(slot - 7)] = value,
    }
}

/// Reorder so that every node comes after its parent (stable otherwise), fixing the indices.
fn parents_first(nodes: Vec<Node>) -> Vec<Node> {
    let n = nodes.len();
    let mut order: Vec<usize> = Vec::with_capacity(n);
    let mut placed = vec![false; n];
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut roots = Vec::new();
    for (i, node) in nodes.iter().enumerate() {
        match node.parent.filter(|&p| p < n && p != i) {
            Some(p) => children[p].push(i),
            None => roots.push(i),
        }
    }
    let mut stack: Vec<usize> = Vec::new();
    for root in roots {
        stack.push(root);
        while let Some(i) = stack.pop() {
            if std::mem::replace(&mut placed[i], true) {
                continue;
            }
            order.push(i);
            stack.extend(children[i].iter().rev());
        }
    }
    // Nodes on a parent cycle: make them roots.
    for (i, done) in placed.iter_mut().enumerate() {
        if !*done {
            *done = true;
            order.push(i);
        }
    }
    let mut new_index = vec![0usize; n];
    for (new, &old) in order.iter().enumerate() {
        new_index[old] = new;
    }
    let mut slots: Vec<Option<Node>> = nodes.into_iter().map(Some).collect();
    order
        .iter()
        .filter_map(|&old| {
            let mut node = slots[old].take()?;
            node.parent = node
                .parent
                .filter(|&p| p < n && p != old)
                .map(|p| new_index[p])
                .filter(|&p| p < new_index[old]);
            Some(node)
        })
        .collect()
}

fn read_prefab(path: &std::path::Path) -> Result<Parsed, String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let mut documents = DocumentReader::new(BufReader::with_capacity(1 << 16, file));
    if !documents
        .looks_like_unity_yaml()
        .map_err(|e| e.to_string())?
    {
        return Err("not text-serialized".into());
    }
    let mut parsed = Parsed::default();
    let mut body = String::new();
    while let Some(header) = documents.next_header().map_err(|e| e.to_string())? {
        let wanted = !header.stripped
            && matches!(
                header.class_id,
                classes::GAME_OBJECT
                    | classes::TRANSFORM
                    | classes::RECT_TRANSFORM
                    | classes::MESH_FILTER
                    | classes::MESH_RENDERER
                    | classes::SKINNED_MESH_RENDERER
                    | classes::PREFAB_INSTANCE
            );
        if !wanted {
            continue;
        }
        documents.read_body(&mut body).map_err(|e| e.to_string())?;
        match header.class_id {
            classes::GAME_OBJECT => {
                let mut go = RawGo {
                    id: header.file_id,
                    name: String::new(),
                    active: true,
                };
                for field in fields(&body) {
                    match field.key {
                        "m_Name" => go.name = unquote(field.value).into_owned(),
                        "m_IsActive" => go.active = field.value != "0",
                        _ => {}
                    }
                }
                parsed.gos.push(go);
            }
            classes::TRANSFORM | classes::RECT_TRANSFORM => {
                let mut t = RawTransform {
                    go: 0,
                    father: 0,
                    data: TransformData::default(),
                };
                for field in fields(&body) {
                    match field.key {
                        "m_GameObject" => t.go = file_id_of(field.value).unwrap_or(0),
                        "m_Father" => t.father = file_id_of(field.value).unwrap_or(0),
                        "m_LocalPosition" => {
                            t.data.position = parse_vec3(field.value).unwrap_or(t.data.position);
                        }
                        "m_LocalRotation" => {
                            t.data.rotation = parse_vec4(field.value).unwrap_or(t.data.rotation);
                        }
                        "m_LocalScale" => {
                            t.data.scale = parse_vec3(field.value).unwrap_or(t.data.scale);
                        }
                        _ => {}
                    }
                }
                parsed.transforms.insert(header.file_id, t);
            }
            classes::MESH_FILTER => {
                let (mut go, mut mesh) = (0, None);
                for field in fields(&body) {
                    match field.key {
                        "m_GameObject" => go = file_id_of(field.value).unwrap_or(0),
                        "m_Mesh" => mesh = AssetRef::parse(field.value),
                        _ => {}
                    }
                }
                parsed.filters.insert(header.file_id, (go, mesh));
            }
            classes::MESH_RENDERER | classes::SKINNED_MESH_RENDERER => {
                let (mut go, mut mesh) = (0, None);
                let mut state = RendererState::default();
                for field in fields(&body) {
                    match field.key {
                        "m_GameObject" => go = file_id_of(field.value).unwrap_or(0),
                        "m_Enabled" => state.enabled = field.value != "0",
                        "m_Mesh" => mesh = AssetRef::parse(field.value),
                        "m_Materials" => {
                            state.materials = field.items().map(AssetRef::parse).collect();
                        }
                        _ => {}
                    }
                }
                parsed.renderers.insert(header.file_id, (go, state, mesh));
            }
            classes::PREFAB_INSTANCE => {
                let mut instance = RawInstance {
                    id: header.file_id,
                    source: String::new(),
                    parent: 0,
                    overrides: Overrides::default(),
                };
                for field in fields(&body) {
                    match field.key {
                        "m_SourcePrefab" => {
                            instance.source = flow_value(field.value, "guid")
                                .unwrap_or_default()
                                .to_owned();
                        }
                        "m_Modification" => {
                            let (parent, overrides) = parse_modification(field.block());
                            instance.parent = parent;
                            instance.overrides = overrides;
                        }
                        _ => {}
                    }
                }
                if !instance.source.is_empty() {
                    parsed.instances.push(instance);
                }
            }
            _ => {}
        }
    }
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prefab::Modification;
    use std::path::Path;

    fn write(dir: &Path, name: &str, text: &str) {
        std::fs::write(dir.join(name), text).unwrap();
    }

    fn temp_dir(tag: &str) -> PathBuf {
        static N: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "rustiny-prefab-{tag}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    const BASE: &str = "%YAML 1.1
--- !u!1 &10
GameObject:
  m_Component:
  - component: {fileID: 11}
  - component: {fileID: 12}
  - component: {fileID: 13}
  m_Name: Rock
  m_IsActive: 1
--- !u!4 &11
Transform:
  m_GameObject: {fileID: 10}
  m_LocalRotation: {x: 0, y: 0, z: 0, w: 1}
  m_LocalPosition: {x: 1, y: 2, z: 3}
  m_LocalScale: {x: 1, y: 1, z: 1}
  m_Father: {fileID: 0}
--- !u!33 &12
MeshFilter:
  m_GameObject: {fileID: 10}
  m_Mesh: {fileID: 4300000, guid: mesh1, type: 2}
--- !u!23 &13
MeshRenderer:
  m_GameObject: {fileID: 10}
  m_Enabled: 1
  m_Materials:
  - {fileID: 2100000, guid: mat1, type: 2}
  - {fileID: 2100000, guid: mat2, type: 2}
--- !u!1 &20
GameObject:
  m_Name: Child
  m_IsActive: 1
--- !u!4 &21
Transform:
  m_GameObject: {fileID: 20}
  m_LocalRotation: {x: 0, y: 0, z: 0, w: 1}
  m_LocalPosition: {x: 0, y: 1, z: 0}
  m_LocalScale: {x: 1, y: 1, z: 1}
  m_Father: {fileID: 11}
";

    fn library(dir: &Path) -> Library<impl Fn(&str) -> Option<PathBuf>> {
        let dir = dir.to_owned();
        Library::new(move |guid: &str| {
            let name = match guid {
                "base" => "base.prefab",
                "variant" => "variant.prefab",
                "model" => "tree.fbx",
                _ => return None,
            };
            Some(dir.join(name))
        })
    }

    fn mod_of(target: i64, path: &str, value: &str, reference: Option<AssetRef>) -> Modification {
        Modification {
            target,
            path: path.into(),
            value: value.into(),
            reference,
        }
    }

    fn asset(file_id: i64, guid: &str) -> Option<AssetRef> {
        Some(AssetRef {
            file_id,
            guid: guid.into(),
        })
    }

    #[test]
    fn a_prefab_file_becomes_nodes_parents_first() {
        let dir = temp_dir("base");
        write(&dir, "base.prefab", BASE);
        let nodes = library(&dir).expand("base", &Overrides::default()).unwrap();
        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[0].name, "Rock");
        assert_eq!(nodes[0].transform.position, [1.0, 2.0, 3.0]);
        assert_eq!(nodes[0].mesh.as_ref().unwrap().guid, "mesh1");
        assert_eq!(nodes[0].renderer.as_ref().unwrap().materials.len(), 2);
        assert_eq!(nodes[1].parent, Some(0));
        let world = compose_world(&nodes, &Mat4::IDENTITY);
        assert_eq!(world[1].translation(), [1.0, 3.0, 3.0]);
    }

    #[test]
    fn overrides_change_mesh_materials_name_and_transform() {
        let dir = temp_dir("mods");
        write(&dir, "base.prefab", BASE);
        let overrides = Overrides {
            mods: vec![
                mod_of(10, "m_Name", "Boulder", None),
                mod_of(11, "m_LocalPosition.x", "9", None),
                mod_of(12, "m_Mesh", "", asset(4_300_000, "other")),
                mod_of(
                    13,
                    "m_Materials.Array.data[1]",
                    "",
                    asset(2_100_000, "mat9"),
                ),
                mod_of(999, "m_Name", "nobody", None),
            ],
            removed: vec![20],
        };
        let lib = library(&dir);
        let nodes = lib.expand("base", &overrides).unwrap();
        assert_eq!(nodes[0].name, "Boulder");
        assert_eq!(nodes[0].transform.position, [9.0, 2.0, 3.0]);
        assert_eq!(nodes[0].mesh.as_ref().unwrap().guid, "other");
        let materials = &nodes[0].renderer.as_ref().unwrap().materials;
        assert_eq!(materials[0].as_ref().unwrap().guid, "mat1");
        assert_eq!(materials[1].as_ref().unwrap().guid, "mat9");
        assert!(nodes[1].removed, "the child was removed");
        let stats = lib.stats();
        assert_eq!((stats.applied, stats.unmatched), (4, 1));
    }

    #[test]
    fn a_variant_applies_its_overrides_and_composes_ids_for_the_outer_level() {
        let dir = temp_dir("variant");
        write(&dir, "base.prefab", BASE);
        // The variant is one instance (id 500) of `base` that renames the rock.
        write(
            &dir,
            "variant.prefab",
            "%YAML 1.1
--- !u!1001 &500
PrefabInstance:
  m_Modification:
    m_TransformParent: {fileID: 0}
    m_Modifications:
    - target: {fileID: 10, guid: base, type: 3}
      propertyPath: m_Name
      value: Variant rock
      objectReference: {fileID: 0}
  m_SourcePrefab: {fileID: 100100000, guid: base, type: 3}
",
        );
        let lib = library(&dir);
        // A scene overrides the variant's rock by the composed ID 11 ^ 500 (its transform).
        let overrides = Overrides {
            mods: vec![mod_of(11 ^ 500, "m_LocalPosition.y", "7", None)],
            removed: vec![],
        };
        let nodes = lib.expand("variant", &overrides).unwrap();
        assert_eq!(nodes[0].name, "Variant rock");
        assert_eq!(nodes[0].transform.position, [1.0, 7.0, 3.0]);
        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[1].parent, Some(0));
    }

    #[test]
    fn a_model_is_one_wildcard_node_that_takes_mesh_and_material_overrides() {
        let dir = temp_dir("model");
        write(&dir, "tree.fbx", "not parsed here");
        let lib = library(&dir);
        let overrides = Overrides {
            mods: vec![
                mod_of(-5754, "m_Mesh", "", asset(4_300_000, "terrain")),
                mod_of(
                    -7511,
                    "m_Materials.Array.data[0]",
                    "",
                    asset(2_100_000, "paint"),
                ),
            ],
            removed: vec![],
        };
        let nodes = lib.expand("model", &overrides).unwrap();
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].mesh.as_ref().unwrap().guid, "terrain");
        assert_eq!(
            nodes[0].renderer.as_ref().unwrap().materials[0]
                .as_ref()
                .unwrap()
                .guid,
            "paint"
        );
        assert_eq!(nodes[0].model.as_deref(), Some("model"));
    }

    #[test]
    fn unknown_and_cyclic_sources_do_not_hang_or_panic() {
        let dir = temp_dir("cycle");
        write(
            &dir,
            "base.prefab",
            "%YAML 1.1
--- !u!1001 &1
PrefabInstance:
  m_Modification:
    m_TransformParent: {fileID: 0}
    m_Modifications: []
  m_SourcePrefab: {fileID: 100100000, guid: base, type: 3}
",
        );
        let lib = library(&dir);
        assert!(lib.expand("nothing", &Overrides::default()).is_none());
        let nodes = lib.expand("base", &Overrides::default()).unwrap();
        assert!(nodes.is_empty());
    }
}
