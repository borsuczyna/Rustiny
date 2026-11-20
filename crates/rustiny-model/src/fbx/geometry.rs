//! One FBX `Geometry` node as triangles: polygon decoding, layer lookup and triangulation.

use super::reader::Node;

/// A polygon corner: which vertex, plus the layer values found for it.
#[derive(Debug, Clone, Copy)]
pub(super) struct Corner {
    pub vertex: u32,
    pub normal: Option<[f64; 3]>,
    pub uv: [f64; 2],
}

/// One triangle of a geometry, with its material slot.
#[derive(Debug, Clone, Copy)]
pub(super) struct Triangle {
    pub corners: [Corner; 3],
    pub slot: usize,
}

/// A decoded geometry.
#[derive(Debug, Default)]
pub(super) struct Geometry {
    /// Control points, file units.
    pub positions: Vec<[f64; 3]>,
    pub triangles: Vec<Triangle>,
}

/// How a layer maps its values to the mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mapping {
    PolygonVertex,
    Vertex,
    Polygon,
    AllSame,
}

struct Layer {
    mapping: Mapping,
    values: Vec<f64>,
    /// `IndexToDirect` indices, when the layer has them.
    index: Option<Vec<i32>>,
    width: usize,
}

impl Layer {
    /// The value for a corner given its counters.
    fn get(&self, polygon_vertex: usize, vertex: usize, polygon: usize) -> Option<&[f64]> {
        let slot = match self.mapping {
            Mapping::PolygonVertex => polygon_vertex,
            Mapping::Vertex => vertex,
            Mapping::Polygon => polygon,
            Mapping::AllSame => 0,
        };
        let slot = match &self.index {
            Some(index) => usize::try_from(*index.get(slot)?).ok()?,
            None => slot,
        };
        self.values
            .get(slot * self.width..slot * self.width + self.width)
    }
}

fn layer(geometry: &Node, element: &str, values: &str, index: &str, width: usize) -> Option<Layer> {
    let node = geometry.child(element)?;
    let mapping = match node.child("MappingInformationType")?.first_str()? {
        "ByPolygonVertex" => Mapping::PolygonVertex,
        "ByVertice" | "ByVertex" => Mapping::Vertex,
        "ByPolygon" => Mapping::Polygon,
        "AllSame" => Mapping::AllSame,
        _ => return None,
    };
    Some(Layer {
        mapping,
        values: node.child(values)?.f64_array()?,
        index: node.child(index).and_then(Node::i32_array),
        width,
    })
}

/// Decode a `Geometry` node. `None` when it has no vertices or polygons (a non-mesh geometry).
pub(super) fn decode(node: &Node) -> Option<Geometry> {
    let flat = node.child("Vertices")?.f64_array()?;
    let positions: Vec<[f64; 3]> = flat.as_chunks::<3>().0.to_vec();
    let polygon_indices = node.child("PolygonVertexIndex")?.i32_array()?;

    let normals = layer(node, "LayerElementNormal", "Normals", "NormalsIndex", 3);
    let uvs = layer(node, "LayerElementUV", "UV", "UVIndex", 2);
    // Material numbers are stored directly, one per polygon (or one for all).
    let materials = node.child("LayerElementMaterial").and_then(|m| {
        let mapping = m.child("MappingInformationType")?.first_str()?;
        let values = m.child("Materials")?.i32_array()?;
        Some((mapping == "AllSame", values))
    });

    let mut triangles = Vec::new();
    let mut polygon: Vec<Corner> = Vec::new();
    let (mut polygon_vertex, mut polygon_number) = (0usize, 0usize);
    for &raw in &polygon_indices {
        let end = raw < 0;
        let vertex = if end { !raw } else { raw };
        let vertex = u32::try_from(vertex).ok()?;
        if vertex as usize >= positions.len() {
            return None;
        }
        let normal = normals
            .as_ref()
            .and_then(|l| l.get(polygon_vertex, vertex as usize, polygon_number))
            .map(|v| [v[0], v[1], v[2]]);
        let uv = uvs
            .as_ref()
            .and_then(|l| l.get(polygon_vertex, vertex as usize, polygon_number))
            .map_or([0.0, 0.0], |v| [v[0], v[1]]);
        polygon.push(Corner { vertex, normal, uv });
        polygon_vertex += 1;
        if end {
            let slot = materials.as_ref().map_or(0, |(all_same, values)| {
                let i = if *all_same { 0 } else { polygon_number };
                values
                    .get(i)
                    .and_then(|v| usize::try_from(*v).ok())
                    .unwrap_or(0)
            });
            // Fan triangulation: exact for convex polygons, which is what exporters write.
            for i in 1..polygon.len().saturating_sub(1) {
                triangles.push(Triangle {
                    corners: [polygon[0], polygon[i], polygon[i + 1]],
                    slot,
                });
            }
            polygon.clear();
            polygon_number += 1;
        }
    }
    Some(Geometry {
        positions,
        triangles,
    })
}
