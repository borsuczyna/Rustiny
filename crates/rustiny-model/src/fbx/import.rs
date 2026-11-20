//! From the node tree to one Unity-space mesh.

use super::FbxError;
use super::file_id::mesh_file_id;
use super::geometry::{self, Corner};
use super::reader::{Node, Prop, parse};
use crate::math::Mat;
use crate::mesh::{MeshData, Section};
use std::collections::HashMap;

/// Importer settings that live in the `.meta` file next to the model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImportSettings {
    /// `globalScale`.
    pub global_scale: f64,
    /// `useFileScale`: honour the file's unit (centimetres, metres, ...) and convert to metres.
    pub use_file_scale: bool,
}

impl Default for ImportSettings {
    fn default() -> Self {
        Self {
            global_scale: 1.0,
            use_file_scale: true,
        }
    }
}

impl ImportSettings {
    /// The settings a model's `.meta` file holds: the `globalScale` and `useFileUnits` lines of
    /// its `ModelImporter` block. Anything missing or unreadable keeps the default, which is also
    /// what Unity writes for a freshly imported model.
    pub fn from_meta(text: &str) -> Self {
        let mut settings = Self::default();
        for line in text.lines() {
            let Some((key, value)) = line.trim().split_once(':') else {
                continue;
            };
            let value = value.trim();
            match key {
                "globalScale" => {
                    if let Ok(scale) = value.parse::<f64>()
                        && scale.is_finite()
                        && scale > 0.0
                    {
                        settings.global_scale = scale;
                    }
                }
                "useFileUnits" => settings.use_file_scale = value != "0",
                _ => {}
            }
        }
        settings
    }
}

/// Which part of a file to import.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Part {
    /// Every mesh of the file, each placed by its node's transform from the file root: the model
    /// as one flattened mesh.
    Whole,
    /// One mesh in the space of its own node, the way Unity stores the `Mesh` sub-asset that a
    /// `MeshFilter` refers to (the node's transform is the `Transform` of the game object, so it
    /// is not baked into the vertices). The mesh is picked by its `fileID`, which Unity derives
    /// from the node's name (see [`mesh_file_id`]); a file with a single mesh needs no match.
    Mesh(i64),
}

/// Parse and convert a binary FBX file into one mesh.
pub fn import(bytes: &[u8], settings: ImportSettings) -> Result<MeshData, FbxError> {
    import_part(bytes, settings, Part::Whole)
}

/// Parse and convert a binary FBX file into one mesh: all of it, or one node's mesh.
pub fn import_part(
    bytes: &[u8],
    settings: ImportSettings,
    part: Part,
) -> Result<MeshData, FbxError> {
    let (_, top) = parse(bytes)?;
    let find = |name: &str| top.iter().find(|n| n.name == name);

    let conversion = unity_conversion(find("GlobalSettings"), settings);
    let objects = find("Objects").ok_or(FbxError::NoMesh)?;

    // Objects by id.
    let mut models: HashMap<i64, &Node> = HashMap::new();
    let mut geometries: HashMap<i64, &Node> = HashMap::new();
    let mut material_names: HashMap<i64, String> = HashMap::new();
    let mut model_order: Vec<i64> = Vec::new();
    for node in &objects.children {
        let Some(id) = node.props.first().and_then(Prop::as_i64) else {
            continue;
        };
        match node.name.as_str() {
            "Model" => {
                models.insert(id, node);
                model_order.push(id);
            }
            "Geometry" => {
                geometries.insert(id, node);
            }
            "Material" => {
                material_names.insert(id, object_name(node));
            }
            _ => {}
        }
    }

    // Object-object connections: `C: "OO", child, parent`.
    let mut parent_of: HashMap<i64, i64> = HashMap::new();
    let mut geometries_of: HashMap<i64, Vec<i64>> = HashMap::new();
    let mut materials_of: HashMap<i64, Vec<i64>> = HashMap::new();
    if let Some(connections) = find("Connections") {
        for c in connections.children_named("C") {
            let [kind, child, parent, ..] = c.props.as_slice() else {
                continue;
            };
            if kind.as_str() != Some("OO") {
                continue;
            }
            let (Some(child), Some(parent)) = (child.as_i64(), parent.as_i64()) else {
                continue;
            };
            if models.contains_key(&child) && models.contains_key(&parent) {
                parent_of.insert(child, parent);
            } else if geometries.contains_key(&child) && models.contains_key(&parent) {
                geometries_of.entry(parent).or_default().push(child);
            } else if material_names.contains_key(&child) && models.contains_key(&parent) {
                materials_of.entry(parent).or_default().push(child);
            }
        }
    }

    // The nodes to import, and whether they keep their place in the file.
    let with_geometry: Vec<i64> = model_order
        .iter()
        .copied()
        .filter(|id| geometries_of.contains_key(id))
        .collect();
    let chosen: Vec<i64> = match part {
        Part::Whole => with_geometry,
        Part::Mesh(id) => {
            let by_id: Vec<i64> = with_geometry
                .iter()
                .copied()
                .filter(|model| mesh_file_id(&object_name(models[model])) == id)
                .collect();
            match (with_geometry.as_slice(), by_id.as_slice()) {
                ([only], _) => vec![*only],
                (_, [first, ..]) => vec![*first],
                _ => return Err(FbxError::NoMesh),
            }
        }
    };

    let mut world_cache: HashMap<i64, Mat> = HashMap::new();
    let mut out = Builder::default();
    let mut material_index: HashMap<String, usize> = HashMap::new();
    for model_id in &chosen {
        let Some(list) = geometries_of.get(model_id) else {
            continue;
        };
        let world = match part {
            Part::Whole => node_world(*model_id, &models, &parent_of, &mut world_cache, 0),
            Part::Mesh(_) => Mat::IDENTITY,
        };
        let transform = conversion
            .mul(&world)
            .mul(&geometric_transform(models[model_id]));
        let slots: Vec<Option<usize>> = materials_of
            .get(model_id)
            .map(|ids| {
                ids.iter()
                    .map(|id| {
                        let name = material_names.get(id).cloned().unwrap_or_default();
                        let next = material_index.len();
                        let index = *material_index.entry(name.clone()).or_insert(next);
                        if index == out.materials.len() {
                            out.materials.push(name);
                        }
                        Some(index)
                    })
                    .collect()
            })
            .unwrap_or_default();
        for geometry_id in list {
            if let Some(geometry) = geometries
                .get(geometry_id)
                .and_then(|n| geometry::decode(n))
            {
                out.add(&geometry, &transform, &slots);
            }
        }
    }
    out.finish()
}

/// One node of a model file as Unity makes a game object of it.
#[derive(Debug, Clone, PartialEq)]
pub struct FbxNode {
    /// The node's name.
    pub name: String,
    /// Index of the parent in the same list; `None` for the file's top-level nodes. Parents come
    /// first.
    pub parent: Option<usize>,
    /// Local position in Unity space, metres.
    pub position: [f32; 3],
    /// Local rotation as a quaternion `[x, y, z, w]` in Unity space.
    pub rotation: [f32; 4],
    /// Local scale.
    pub scale: [f32; 3],
    /// Whether the node has a mesh (a `MeshFilter` and a renderer in Unity).
    pub has_mesh: bool,
}

/// The node hierarchy of a binary FBX file with each node's local transform in Unity's space
/// (the same conversion the meshes get: axes turned and the unit and `globalScale` applied to
/// translations, so that `Part::Mesh` vertices placed by these nodes sit where `Part::Whole`
/// puts them).
pub fn nodes(bytes: &[u8], settings: ImportSettings) -> Result<Vec<FbxNode>, FbxError> {
    let (_, top) = parse(bytes)?;
    let find = |name: &str| top.iter().find(|n| n.name == name);
    let conversion = unity_conversion(find("GlobalSettings"), settings);
    let objects = find("Objects").ok_or(FbxError::NoMesh)?;
    let mut models: HashMap<i64, &Node> = HashMap::new();
    let mut with_geometry: std::collections::HashSet<i64> = std::collections::HashSet::new();
    let mut geometry_ids: std::collections::HashSet<i64> = std::collections::HashSet::new();
    let mut order: Vec<i64> = Vec::new();
    for node in &objects.children {
        let Some(id) = node.props.first().and_then(Prop::as_i64) else {
            continue;
        };
        match node.name.as_str() {
            "Model" => {
                models.insert(id, node);
                order.push(id);
            }
            "Geometry" => {
                geometry_ids.insert(id);
            }
            _ => {}
        }
    }
    let mut parent_of: HashMap<i64, i64> = HashMap::new();
    if let Some(connections) = find("Connections") {
        for c in connections.children_named("C") {
            let [kind, child, parent, ..] = c.props.as_slice() else {
                continue;
            };
            if kind.as_str() != Some("OO") {
                continue;
            }
            let (Some(child), Some(parent)) = (child.as_i64(), parent.as_i64()) else {
                continue;
            };
            if models.contains_key(&child) && models.contains_key(&parent) {
                parent_of.insert(child, parent);
            } else if geometry_ids.contains(&child) && models.contains_key(&parent) {
                with_geometry.insert(parent);
            }
        }
    }

    // The scale in the conversion, and the axis turn without it.
    let s = (conversion.0[0].powi(2) + conversion.0[1].powi(2) + conversion.0[2].powi(2)).sqrt();
    if !s.is_finite() || s <= 0.0 {
        return Err(FbxError::NoMesh);
    }
    let mut axes = conversion;
    for v in &mut axes.0 {
        *v /= s;
    }
    axes.0[15] = 1.0;

    // Depth first from the top-level nodes, parents before children.
    let mut children: HashMap<Option<i64>, Vec<i64>> = HashMap::new();
    for id in &order {
        let parent = parent_of.get(id).copied();
        children.entry(parent).or_default().push(*id);
    }
    let mut out: Vec<FbxNode> = Vec::with_capacity(order.len());
    let mut placed: HashMap<i64, usize> = HashMap::new();
    let mut stack: Vec<(i64, Option<usize>)> = children
        .get(&None)
        .into_iter()
        .flatten()
        .rev()
        .map(|id| (*id, None))
        .collect();
    while let Some((id, parent)) = stack.pop() {
        if placed.contains_key(&id) || out.len() > 1 << 20 {
            continue;
        }
        let model = models[&id];
        let mut local = local_matrix(model);
        for v in &mut local.0[12..15] {
            *v *= s;
        }
        let unity = axes.mul(&local).mul(&axes.transposed());
        let (position, rotation, scale) = decompose(&unity);
        let index = out.len();
        placed.insert(id, index);
        out.push(FbxNode {
            name: object_name(model),
            parent,
            position,
            rotation,
            scale,
            has_mesh: with_geometry.contains(&id),
        });
        for child in children.get(&Some(id)).into_iter().flatten().rev() {
            stack.push((*child, Some(index)));
        }
    }
    Ok(out)
}

/// Translation, rotation (quaternion `x, y, z, w`) and scale of an affine matrix.
fn decompose(m: &Mat) -> ([f32; 3], [f32; 4], [f32; 3]) {
    let col = |c: usize| [m.0[c * 4], m.0[c * 4 + 1], m.0[c * 4 + 2]];
    let len = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let mut scale = [len(col(0)), len(col(1)), len(col(2))];
    if m.det3() < 0.0 {
        scale[0] = -scale[0];
    }
    let axis = |c: usize| {
        let v = col(c);
        let l = if scale[c].abs() > 1e-12 {
            scale[c]
        } else {
            1.0
        };
        [v[0] / l, v[1] / l, v[2] / l]
    };
    let (x, y, z) = (axis(0), axis(1), axis(2));
    // Rotation matrix with columns x, y, z -> quaternion.
    let (m00, m10, m20) = (x[0], x[1], x[2]);
    let (m01, m11, m21) = (y[0], y[1], y[2]);
    let (m02, m12, m22) = (z[0], z[1], z[2]);
    let trace = m00 + m11 + m22;
    let q = if trace > 0.0 {
        let k = (trace + 1.0).sqrt() * 2.0;
        [(m21 - m12) / k, (m02 - m20) / k, (m10 - m01) / k, 0.25 * k]
    } else if m00 > m11 && m00 > m22 {
        let k = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
        [0.25 * k, (m01 + m10) / k, (m02 + m20) / k, (m21 - m12) / k]
    } else if m11 > m22 {
        let k = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
        [(m01 + m10) / k, 0.25 * k, (m12 + m21) / k, (m02 - m20) / k]
    } else {
        let k = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
        [(m02 + m20) / k, (m12 + m21) / k, 0.25 * k, (m10 - m01) / k]
    };
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let q = if norm > 1e-12 {
        [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm]
    } else {
        [0.0, 0.0, 0.0, 1.0]
    };
    (
        [m.0[12] as f32, m.0[13] as f32, m.0[14] as f32],
        q.map(|v| v as f32),
        scale.map(|v| v as f32),
    )
}

/// The matrix taking FBX space (in file units) to Unity space in metres.
///
/// Unity's importer converts the file's axes to its own left-handed, Y-up, Z-forward space.
/// With the file's right, up and front directions `R`, `U`, `F` (unit vectors in file space), a
/// point `p` becomes `(-R.p, U.p, F.p)`; for the common Y-up file that is `(-x, y, z)`.
fn unity_conversion(global: Option<&Node>, settings: ImportSettings) -> Mat {
    let int = |name: &str, default: i64| {
        global
            .and_then(|g| p70(g, name))
            .and_then(|v| v.first().and_then(Prop::as_f64))
            .map_or(default, |v| v.round() as i64)
    };
    let unit = global
        .and_then(|g| p70(g, "UnitScaleFactor"))
        .and_then(|v| v.first().and_then(Prop::as_f64))
        .filter(|u| u.is_finite() && *u > 0.0)
        .unwrap_or(1.0);
    let axis = |axis: i64, sign: i64| {
        let mut v = [0.0; 3];
        if let Ok(i) = usize::try_from(axis)
            && i < 3
        {
            v[i] = if sign < 0 { -1.0 } else { 1.0 };
        }
        v
    };
    let right = axis(int("CoordAxis", 0), int("CoordAxisSign", 1));
    let up = axis(int("UpAxis", 1), int("UpAxisSign", 1));
    let front = axis(int("FrontAxis", 2), int("FrontAxisSign", 1));
    let scale = if settings.use_file_scale {
        unit * 0.01
    } else {
        1.0
    } * settings.global_scale;
    let rows = [[-right[0], -right[1], -right[2]], up, front];
    let mut m = Mat::IDENTITY;
    for (row, values) in rows.iter().enumerate() {
        for (col, v) in values.iter().enumerate() {
            m.0[col * 4 + row] = v * scale;
        }
    }
    m
}

/// Name of an object node: the first property is `"Name\0\x01Class"`.
fn object_name(node: &Node) -> String {
    node.props
        .get(1)
        .and_then(Prop::as_str)
        .map(|s| s.split('\0').next().unwrap_or("").to_owned())
        .unwrap_or_default()
}

/// The values of a `Properties70` entry (everything after the four descriptor strings).
fn p70<'a>(node: &'a Node, name: &str) -> Option<&'a [Prop]> {
    node.child("Properties70")?
        .children_named("P")
        .find(|p| p.props.first().and_then(Prop::as_str) == Some(name))
        .and_then(|p| p.props.get(4..))
}

fn vec3(node: &Node, name: &str, default: [f64; 3]) -> [f64; 3] {
    match p70(node, name) {
        Some([x, y, z, ..]) => [
            x.as_f64().unwrap_or(default[0]),
            y.as_f64().unwrap_or(default[1]),
            z.as_f64().unwrap_or(default[2]),
        ],
        _ => default,
    }
}

/// The order Euler angles are applied in, from FBX's `RotationOrder` enum.
fn rotation_order(node: &Node) -> [usize; 3] {
    let value = p70(node, "RotationOrder")
        .and_then(|v| v.first())
        .and_then(Prop::as_i64)
        .unwrap_or(0);
    match value {
        1 => [0, 2, 1],
        2 => [1, 2, 0],
        3 => [1, 0, 2],
        4 => [2, 0, 1],
        5 => [2, 1, 0],
        _ => [0, 1, 2],
    }
}

/// `T * Rpre * R * Rpost^-1 * S` for a model node (pivots and offsets are not applied).
fn local_matrix(node: &Node) -> Mat {
    let order = rotation_order(node);
    let t = vec3(node, "Lcl Translation", [0.0; 3]);
    let r = vec3(node, "Lcl Rotation", [0.0; 3]);
    let s = vec3(node, "Lcl Scaling", [1.0; 3]);
    let pre = vec3(node, "PreRotation", [0.0; 3]);
    let post = vec3(node, "PostRotation", [0.0; 3]);
    let has = |name: &str| p70(node, name).is_some();
    let mut m = Mat::translation(t);
    if has("PreRotation") {
        m = m.mul(&Mat::euler_degrees(pre, [0, 1, 2]));
    }
    m = m.mul(&Mat::euler_degrees(r, order));
    if has("PostRotation") {
        m = m.mul(&Mat::euler_degrees(post, [0, 1, 2]).transposed());
    }
    m.mul(&Mat::scaling(s))
}

/// The geometric transform applies to a node's own geometry but not to its children.
fn geometric_transform(node: &Node) -> Mat {
    let t = vec3(node, "GeometricTranslation", [0.0; 3]);
    let r = vec3(node, "GeometricRotation", [0.0; 3]);
    let s = vec3(node, "GeometricScaling", [1.0; 3]);
    Mat::translation(t)
        .mul(&Mat::euler_degrees(r, [0, 1, 2]))
        .mul(&Mat::scaling(s))
}

/// A node's transform relative to the file root: its parents' local matrices applied down.
fn node_world(
    id: i64,
    models: &HashMap<i64, &Node>,
    parent_of: &HashMap<i64, i64>,
    cache: &mut HashMap<i64, Mat>,
    depth: usize,
) -> Mat {
    if let Some(m) = cache.get(&id) {
        return *m;
    }
    // A cycle (or an absurdly deep chain) is cut by treating the node as a root.
    let parent = if depth < 256 {
        parent_of.get(&id)
    } else {
        None
    };
    let local = models.get(&id).map_or(Mat::IDENTITY, |n| local_matrix(n));
    let world = match parent {
        Some(&p) => node_world(p, models, parent_of, cache, depth + 1).mul(&local),
        None => local,
    };
    cache.insert(id, world);
    world
}

/// Collects vertices and triangles from every geometry of the file.
#[derive(Default)]
struct Builder {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    /// Triangles per material (`None` = no material), in first-seen order.
    buckets: Vec<(Option<usize>, Vec<u32>)>,
    materials: Vec<String>,
}

impl Builder {
    fn bucket(&mut self, material: Option<usize>) -> &mut Vec<u32> {
        let at = match self.buckets.iter().position(|(m, _)| *m == material) {
            Some(i) => i,
            None => {
                self.buckets.push((material, Vec::new()));
                self.buckets.len() - 1
            }
        };
        &mut self.buckets[at].1
    }

    fn add(&mut self, geometry: &geometry::Geometry, transform: &Mat, slots: &[Option<usize>]) {
        let normal_matrix = transform.normal_matrix();
        let mirrored = transform.det3() < 0.0;
        let world: Vec<[f64; 3]> = geometry
            .positions
            .iter()
            .map(|p| transform.point(*p))
            .collect();
        // Vertices are shared between triangles that agree on position, normal and UV.
        let mut seen: HashMap<(u32, [u32; 3], [u32; 2]), u32> = HashMap::new();
        for triangle in &geometry.triangles {
            let material = slots.get(triangle.slot).copied().flatten();
            let flat = flat_normal(
                world[triangle.corners[0].vertex as usize],
                world[triangle.corners[1].vertex as usize],
                world[triangle.corners[2].vertex as usize],
                mirrored,
            );
            let mut ids = [0u32; 3];
            for (k, corner) in triangle.corners.iter().enumerate() {
                let normal = corner
                    .normal
                    .map_or(flat, |n| transform_normal(&normal_matrix, n, flat));
                ids[k] = self.vertex(&mut seen, &world, corner, normal);
            }
            let order = if mirrored { [0, 2, 1] } else { [0, 1, 2] };
            let bucket = self.bucket(material);
            bucket.extend(order.map(|k| ids[k]));
        }
    }

    fn vertex(
        &mut self,
        seen: &mut HashMap<(u32, [u32; 3], [u32; 2]), u32>,
        world: &[[f64; 3]],
        corner: &Corner,
        normal: [f32; 3],
    ) -> u32 {
        let uv = [corner.uv[0] as f32, corner.uv[1] as f32];
        let key = (
            corner.vertex,
            normal.map(f32::to_bits),
            uv.map(f32::to_bits),
        );
        *seen.entry(key).or_insert_with(|| {
            let p = world[corner.vertex as usize];
            self.positions.push([p[0] as f32, p[1] as f32, p[2] as f32]);
            self.normals.push(normal);
            self.uvs.push(uv);
            (self.positions.len() - 1) as u32
        })
    }

    fn finish(self) -> Result<MeshData, FbxError> {
        let mut indices = Vec::new();
        let mut sections = Vec::new();
        for (material, tris) in self.buckets {
            if tris.is_empty() {
                continue;
            }
            sections.push(Section {
                material,
                first_index: indices.len() as u32,
                index_count: tris.len() as u32,
            });
            indices.extend(tris);
        }
        if indices.is_empty() {
            return Err(FbxError::NoMesh);
        }
        Ok(MeshData {
            positions: self.positions,
            normals: self.normals,
            uvs: self.uvs,
            indices,
            sections,
            materials: self.materials,
            ..MeshData::default()
        })
    }
}

fn normalised(v: [f64; 3]) -> Option<[f32; 3]> {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    (len.is_finite() && len > 1e-12).then(|| {
        [
            (v[0] / len) as f32,
            (v[1] / len) as f32,
            (v[2] / len) as f32,
        ]
    })
}

/// Outward face normal of a triangle given in its original (counter-clockwise seen from
/// outside, right-handed) order, after the vertices were moved to Unity space. A mirroring
/// transform (`mirrored`, negative determinant) turns the cross product of the moved edges
/// around, so it is negated back.
fn flat_normal(a: [f64; 3], b: [f64; 3], c: [f64; 3], mirrored: bool) -> [f32; 3] {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let mut n = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    if mirrored {
        n = [-n[0], -n[1], -n[2]];
    }
    normalised(n).unwrap_or([0.0, 1.0, 0.0])
}

fn transform_normal(m: &[[f64; 3]; 3], n: [f64; 3], fallback: [f32; 3]) -> [f32; 3] {
    let v = [
        m[0][0] * n[0] + m[0][1] * n[1] + m[0][2] * n[2],
        m[1][0] * n[0] + m[1][1] * n[1] + m[1][2] * n[2],
        m[2][0] * n[0] + m[2][1] * n[1] + m[2][2] * n[2],
    ];
    normalised(v).unwrap_or(fallback)
}
