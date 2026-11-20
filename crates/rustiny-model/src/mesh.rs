//! The importer's output: one triangle mesh per model file, in Unity's space.

/// A run of triangles that use one material.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Section {
    /// Index into [`MeshData::materials`], or `None` when the file names no material.
    pub material: Option<usize>,
    /// First index in [`MeshData::indices`].
    pub first_index: u32,
    /// Number of indices (a multiple of three).
    pub index_count: u32,
}

/// An axis-aligned box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    /// Smallest corner.
    pub min: [f32; 3],
    /// Largest corner.
    pub max: [f32; 3],
}

impl Bounds {
    /// Centre of the box.
    pub fn centre(&self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }

    /// Half the size along each axis.
    pub fn half_extent(&self) -> [f32; 3] {
        [
            (self.max[0] - self.min[0]) * 0.5,
            (self.max[1] - self.min[1]) * 0.5,
            (self.max[2] - self.min[2]) * 0.5,
        ]
    }
}

/// An indexed triangle mesh with one vertex stream.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MeshData {
    /// Vertex positions, metres, Unity axes (left-handed, +Y up).
    pub positions: Vec<[f32; 3]>,
    /// Unit normals, one per vertex.
    pub normals: Vec<[f32; 3]>,
    /// Texture coordinates, one per vertex (`[0, 0]` when the file has none). V points up as in
    /// Unity (the file's V is used as is).
    pub uvs: Vec<[f32; 2]>,
    /// Triangle list, grouped by section.
    pub indices: Vec<u32>,
    /// Triangle runs per material, in `indices` order.
    pub sections: Vec<Section>,
    /// Material names the sections refer to.
    pub materials: Vec<String>,
    /// Second texture coordinates (lightmap UVs), empty when the source has none.
    pub uv1: Vec<[f32; 2]>,
    /// Tangents with handedness in `w`, empty when the source has none.
    pub tangents: Vec<[f32; 4]>,
    /// Vertex colours, empty when the source has none.
    pub colors: Vec<[f32; 4]>,
}

impl MeshData {
    /// Number of triangles.
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Bounding box of the positions, or `None` for an empty mesh.
    pub fn bounds(&self) -> Option<Bounds> {
        let first = *self.positions.first()?;
        let (mut min, mut max) = (first, first);
        for p in &self.positions {
            for axis in 0..3 {
                min[axis] = min[axis].min(p[axis]);
                max[axis] = max[axis].max(p[axis]);
            }
        }
        Some(Bounds { min, max })
    }

    /// Whether every array is consistent and every index is in range. The importer guarantees
    /// this; it is checked in tests and before upload.
    pub fn is_valid(&self) -> bool {
        let n = self.positions.len();
        self.normals.len() == n
            && self.uvs.len() == n
            && self.indices.len().is_multiple_of(3)
            && self.indices.iter().all(|&i| (i as usize) < n)
            && self
                .sections
                .iter()
                .map(|s| s.index_count as usize)
                .sum::<usize>()
                == self.indices.len()
    }
}
