//! The unit box, the vertex layout shared with meshes, and the per-instance data layout.

/// Per-instance data as the shaders read it: a column-major model matrix, a colour whose alpha
/// says whether the instance is drawn (1) or hidden unless selected (0), and the id of the scene
/// object the instance belongs to (what `FrameParams::selected` is compared with). 84 bytes.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct GpuInstance {
    /// Column-major model (local-to-world) matrix.
    pub model: [f32; 16],
    /// `[r, g, b, visible]`. Multiplied with the material tint of the section being drawn.
    pub color: [f32; 4],
    /// Scene object this instance stands for.
    pub object: u32,
}

impl GpuInstance {
    /// Size in bytes of one instance in the instance buffer.
    pub const SIZE: usize = 84;

    /// An instance with a colour (object id 0); `visible = false` hides it unless it is the
    /// selected one.
    pub fn new(model: [f32; 16], rgb: [f32; 3], visible: bool) -> Self {
        Self {
            model,
            color: [rgb[0], rgb[1], rgb[2], if visible { 1.0 } else { 0.0 }],
            object: 0,
        }
    }

    /// The same instance standing for another scene object.
    pub fn with_object(mut self, object: u32) -> Self {
        self.object = object;
        self
    }

    /// The instance as bytes in native order, ready to copy into the instance buffer.
    pub(crate) fn write_to(&self, out: &mut [u8]) {
        let (chunks, _) = out.as_chunks_mut::<4>();
        let floats = self.model.iter().chain(&self.color);
        for (chunk, value) in chunks.iter_mut().zip(floats) {
            *chunk = value.to_ne_bytes();
        }
        if let Some(chunk) = chunks.get_mut(20) {
            *chunk = self.object.to_ne_bytes();
        }
    }
}

/// Bytes per vertex of the box and of meshes: position, normal (3 floats each), uv (2 floats).
pub(crate) const VERTEX_STRIDE: usize = 32;

/// Number of indices of the unit box.
pub(crate) const BOX_INDEX_COUNT: u32 = 36;

/// A unit box centred on the origin: vertex bytes (24 vertices, 4 per face so normals are flat)
/// and index bytes (36 `u32` indices).
pub(crate) fn unit_box() -> (Vec<u8>, Vec<u8>) {
    // (normal, tangent u, tangent v): each face is the square spanned by u and v at +-0.5 along
    // the normal, wound counter-clockwise seen from outside.
    const FACES: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
        ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
        ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
        ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    ];
    let mut vertices = Vec::with_capacity(24 * VERTEX_STRIDE);
    let mut indices = Vec::with_capacity(36 * 4);
    for (face, (n, u, v)) in FACES.iter().enumerate() {
        for (su, sv) in [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)] {
            let position = [
                n[0] * 0.5 + u[0] * su + v[0] * sv,
                n[1] * 0.5 + u[1] * su + v[1] * sv,
                n[2] * 0.5 + u[2] * su + v[2] * sv,
            ];
            let uv = [su + 0.5, sv + 0.5];
            for value in position.iter().chain(n).chain(&uv) {
                vertices.extend_from_slice(&value.to_ne_bytes());
            }
        }
        let base = (face * 4) as u32;
        for offset in [0u32, 1, 2, 0, 2, 3] {
            indices.extend_from_slice(&(base + offset).to_ne_bytes());
        }
    }
    (vertices, indices)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn floats(bytes: &[u8]) -> Vec<f32> {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_ne_bytes(*c))
            .collect()
    }

    fn indices_of(bytes: &[u8]) -> Vec<usize> {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_ne_bytes(*c) as usize)
            .collect()
    }

    #[test]
    fn the_box_has_24_vertices_and_36_indices() {
        let (vertices, indices) = unit_box();
        assert_eq!(vertices.len(), 24 * VERTEX_STRIDE);
        assert_eq!(indices.len() as u32, BOX_INDEX_COUNT * 4);
        assert!(indices_of(&indices).iter().all(|&i| i < 24));
    }

    #[test]
    fn faces_wind_counter_clockwise_seen_from_outside() {
        let (vertices, indices) = unit_box();
        let v = floats(&vertices);
        for triangle in indices_of(&indices).as_chunks::<3>().0 {
            let p = |i: usize| [v[i * 8], v[i * 8 + 1], v[i * 8 + 2]];
            let n = [
                v[triangle[0] * 8 + 3],
                v[triangle[0] * 8 + 4],
                v[triangle[0] * 8 + 5],
            ];
            let (a, b, c) = (p(triangle[0]), p(triangle[1]), p(triangle[2]));
            let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let cross = [
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            ];
            let along_normal = cross[0] * n[0] + cross[1] * n[1] + cross[2] * n[2];
            assert!(along_normal > 0.0, "triangle {triangle:?} faces inwards");
        }
    }

    #[test]
    fn vertices_lie_on_the_unit_box_and_normals_are_unit_axes() {
        let (vertices, _) = unit_box();
        for vertex in floats(&vertices).as_chunks::<8>().0 {
            assert!(vertex[..3].iter().all(|c| c.abs() <= 0.5 + 1e-6));
            assert!(
                vertex[6..].iter().all(|c| (0.0..=1.0).contains(c)),
                "uv lies in 0..1"
            );
            let n = &vertex[3..6];
            assert_eq!(n.iter().map(|c| c * c).sum::<f32>(), 1.0);
            let dot: f32 = vertex[..3].iter().zip(n).map(|(p, n)| p * n).sum();
            assert!((dot - 0.5).abs() < 1e-6, "vertex is not on its face");
        }
    }

    #[test]
    fn instances_pack_to_84_bytes_in_field_order() {
        let mut model = [0.0; 16];
        model[0] = 2.0;
        model[15] = 1.0;
        let instance = GpuInstance::new(model, [0.1, 0.2, 0.3], true).with_object(77);
        let mut bytes = [0u8; GpuInstance::SIZE];
        instance.write_to(&mut bytes);
        let f = floats(&bytes);
        assert_eq!(f.len(), 21);
        assert_eq!(f[0], 2.0);
        assert_eq!(f[15], 1.0);
        assert_eq!(&f[16..20], &[0.1, 0.2, 0.3, 1.0]);
        assert_eq!(f[20].to_bits(), 77, "the object id follows the colour");
        assert_eq!(GpuInstance::new(model, [0.0; 3], false).color[3], 0.0);
        assert_eq!(std::mem::size_of::<GpuInstance>(), GpuInstance::SIZE);
    }
}
