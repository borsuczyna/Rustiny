//! Meshes the scene pass draws: the vertex layout, material sections and the GPU buffers.
//!
//! A mesh is uploaded once under a caller-chosen `key`; instances of it are given separately as a
//! [`MeshGroup`] with the same key, so the renderer draws every instance of a mesh with one call
//! per section.

use crate::buffer::HostBuffer;
use crate::geometry::{GpuInstance, VERTEX_STRIDE};
use ash::vk;
use rustiny_vulkan::{Device, Result, VulkanError};
use std::sync::Arc;

/// One vertex of a mesh. 32 bytes, the same layout as the proxy box.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct MeshVertex {
    /// Position in the mesh's local space, metres.
    pub position: [f32; 3],
    /// Unit normal.
    pub normal: [f32; 3],
    /// Texture coordinates, V up (as in Unity).
    pub uv: [f32; 2],
}

impl MeshVertex {
    pub(crate) fn write_to(&self, out: &mut [u8]) {
        let (chunks, _) = out.as_chunks_mut::<4>();
        let values = self.position.iter().chain(&self.normal).chain(&self.uv);
        for (chunk, value) in chunks.iter_mut().zip(values) {
            *chunk = value.to_ne_bytes();
        }
    }
}

/// How a surface is shaded: a simplified form of Unity's URP Lit inputs (metallic workflow).
///
/// Texture keys refer to textures added with `SceneRenderer::add_texture`; while a texture has not
/// been uploaded yet the slot behaves as if the surface had none.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Surface {
    /// Colour the albedo is multiplied with, linear.
    pub tint: [f32; 3],
    /// Albedo texture (sRGB), or `None` for a plain colour.
    pub albedo: Option<u32>,
    /// Tangent-space normal map (linear, Unity's convention: green up), or `None`.
    pub normal: Option<u32>,
    /// Metallic in red and smoothness in alpha (linear), or `None`.
    pub metallic_map: Option<u32>,
    /// Pixels whose albedo alpha is below this are not drawn; 0 draws everything.
    pub cutoff: f32,
    /// Strength of the normal map (1 = as stored).
    pub normal_scale: f32,
    /// Metallic when there is no `metallic_map`, 0 to 1.
    pub metallic: f32,
    /// Smoothness (times the map's alpha when there is one), 0 to 1. 0 is matte.
    pub smoothness: f32,
    /// Light the surface emits, linear.
    pub emission: [f32; 3],
    /// Draw the albedo as is, without lighting (Unity's Unlit shaders).
    pub unlit: bool,
    /// A shader material from `SceneRenderer::add_shader_material`: the section is drawn with a
    /// Unity shader and the fields above (apart from the textures it names) are ignored.
    pub shader: Option<u32>,
}

impl Default for Surface {
    /// White, matte, opaque, lit.
    fn default() -> Self {
        Self {
            tint: [1.0; 3],
            albedo: None,
            normal: None,
            metallic_map: None,
            cutoff: 0.0,
            normal_scale: 1.0,
            metallic: 0.0,
            smoothness: 0.0,
            emission: [0.0; 3],
            unlit: false,
            shader: None,
        }
    }
}

impl Surface {
    /// The three texture slots (albedo, normal, metallic-smoothness).
    pub(crate) fn textures(&self) -> [Option<u32>; crate::texture::SLOTS] {
        [self.albedo, self.normal, self.metallic_map]
    }
}

/// A run of triangles drawn with one surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshSection {
    /// First index in the mesh's index list.
    pub first_index: u32,
    /// Number of indices (a multiple of three).
    pub index_count: u32,
    /// How the triangles are shaded.
    pub surface: Surface,
}

/// A section as the renderer keeps it: where to draw, which record, which textures.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct GpuSection {
    pub first_index: u32,
    pub index_count: u32,
    /// Index of the section's record in the material store.
    pub material: u32,
    pub textures: [Option<u32>; crate::texture::SLOTS],
    /// The shader material the section is drawn with, if it has one.
    pub shader: Option<u32>,
}

/// All instances of one mesh.
#[derive(Debug, Clone, PartialEq)]
pub struct MeshGroup {
    /// The key the mesh was uploaded under.
    pub key: u32,
    /// Where the mesh is drawn. Each instance names the scene object it belongs to.
    pub instances: Vec<GpuInstance>,
}

/// Where a group's instances sit in the shared instance buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GroupRange {
    pub key: u32,
    pub first: u32,
    pub count: u32,
}

/// What the renderer holds for meshes, for the status bar and the tests.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MeshStats {
    /// Meshes uploaded.
    pub meshes: usize,
    /// Instances in the mesh instance buffer, whether or not their mesh has been uploaded yet.
    pub instances: usize,
    /// Triangles submitted by the last recorded pass (instances x triangles).
    pub triangles_drawn: u64,
    /// Draw calls submitted by the last recorded pass, boxes included.
    pub draw_calls: u32,
    /// Material records in use (the neutral one of the boxes included).
    pub materials: usize,
    /// Draw calls of the last recorded pass that went to Unity shaders (a subset of `draw_calls`).
    pub shader_draws: u32,
    /// Instances of Unity-shader sections the last pass left out for lack of per-frame memory.
    pub shader_skipped: u32,
    /// Shader materials added.
    pub shader_materials: usize,
    /// Textures stored (created, whether or not their copy has been recorded yet).
    pub textures: usize,
    /// Texture memory of those, in bytes (texels, mip chains included).
    pub texture_bytes: u64,
}

/// Uploaded geometry and its sections.
pub(crate) struct GpuMesh {
    /// Smallest and largest vertex position, for culling.
    pub bounds: [[f32; 3]; 2],
    pub vertices: HostBuffer,
    pub indices: HostBuffer,
    pub sections: Vec<GpuSection>,
    pub triangles: u32,
}

impl std::fmt::Debug for GpuMesh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuMesh")
            .field("sections", &self.sections.len())
            .field("triangles", &self.triangles)
            .finish_non_exhaustive()
    }
}

impl GpuMesh {
    /// Check the data, then upload it. Section `i` uses the material record `first_material + i`.
    pub(crate) fn new(
        device: &Arc<Device>,
        vertices: &[MeshVertex],
        indices: &[u32],
        sections: &[MeshSection],
        first_material: u32,
    ) -> Result<Self> {
        validate(vertices.len(), indices, sections)?;
        let mut vertex_bytes = vec![0u8; vertices.len() * VERTEX_STRIDE];
        let (slots, _) = vertex_bytes.as_chunks_mut::<VERTEX_STRIDE>();
        for (vertex, out) in vertices.iter().zip(slots) {
            vertex.write_to(out);
        }
        let mut index_bytes = Vec::with_capacity(indices.len() * 4);
        for index in indices {
            index_bytes.extend_from_slice(&index.to_ne_bytes());
        }
        let mut bounds = [[f32::INFINITY; 3], [f32::NEG_INFINITY; 3]];
        for vertex in vertices {
            for (axis, &value) in vertex.position.iter().enumerate() {
                bounds[0][axis] = bounds[0][axis].min(value);
                bounds[1][axis] = bounds[1][axis].max(value);
            }
        }
        Ok(Self {
            bounds,
            vertices: HostBuffer::with_data(
                device,
                &vertex_bytes,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                "mesh-vertices",
            )?,
            indices: HostBuffer::with_data(
                device,
                &index_bytes,
                vk::BufferUsageFlags::INDEX_BUFFER,
                "mesh-indices",
            )?,
            sections: sections
                .iter()
                .zip(first_material..)
                .map(|(s, material)| GpuSection {
                    first_index: s.first_index,
                    index_count: s.index_count,
                    material,
                    textures: s.surface.textures(),
                    shader: s.surface.shader,
                })
                .collect(),
            triangles: u32::try_from(indices.len() / 3).unwrap_or(u32::MAX),
        })
    }
}

/// Reject data the GPU would read out of bounds.
fn validate(vertex_count: usize, indices: &[u32], sections: &[MeshSection]) -> Result<()> {
    let bad = |why: &str| Err(VulkanError::InvalidInput(why.to_owned()));
    if vertex_count == 0 || indices.is_empty() {
        return bad("a mesh needs vertices and indices");
    }
    if !indices.len().is_multiple_of(3) {
        return bad("the index count is not a multiple of three");
    }
    if indices.iter().any(|&i| i as usize >= vertex_count) {
        return bad("an index points past the last vertex");
    }
    for section in sections {
        let end = u64::from(section.first_index) + u64::from(section.index_count);
        if end > indices.len() as u64 || !section.index_count.is_multiple_of(3) {
            return bad("a section reaches past the index list or is not whole triangles");
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn section(first: u32, count: u32) -> MeshSection {
        MeshSection {
            first_index: first,
            index_count: count,
            surface: Surface::default(),
        }
    }

    #[test]
    fn valid_data_passes() {
        assert!(validate(3, &[0, 1, 2], &[section(0, 3)]).is_ok());
        assert!(validate(4, &[0, 1, 2, 2, 1, 3], &[section(0, 3), section(3, 3)]).is_ok());
        assert!(
            validate(3, &[0, 1, 2], &[]).is_ok(),
            "no sections draws nothing"
        );
    }

    #[test]
    fn inconsistent_data_is_rejected() {
        assert!(validate(0, &[0, 1, 2], &[]).is_err(), "no vertices");
        assert!(validate(3, &[], &[]).is_err(), "no indices");
        assert!(validate(3, &[0, 1], &[]).is_err(), "not whole triangles");
        assert!(validate(3, &[0, 1, 3], &[]).is_err(), "index past the end");
        assert!(
            validate(3, &[0, 1, 2], &[section(0, 6)]).is_err(),
            "section too long"
        );
        assert!(
            validate(3, &[0, 1, 2], &[section(3, 3)]).is_err(),
            "section past the end"
        );
        assert!(
            validate(3, &[0, 1, 2], &[section(0, 2)]).is_err(),
            "partial triangle"
        );
        assert!(
            validate(3, &[0, 1, 2], &[section(u32::MAX, 3)]).is_err(),
            "no overflow when adding"
        );
    }

    #[test]
    fn vertices_pack_to_32_bytes() {
        let v = MeshVertex {
            position: [1.0, 2.0, 3.0],
            normal: [0.0, 1.0, 0.0],
            uv: [0.25, 0.75],
        };
        let mut bytes = [0u8; VERTEX_STRIDE];
        v.write_to(&mut bytes);
        let f = |i: usize| f32::from_ne_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
        assert_eq!((f(0), f(2), f(4), f(6), f(7)), (1.0, 3.0, 1.0, 0.25, 0.75));
        assert_eq!(std::mem::size_of::<MeshVertex>(), VERTEX_STRIDE);
    }
}
