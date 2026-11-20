//! Imported model data in the renderer's vertex and section format.

use rustiny_model::MeshData;
use rustiny_render::{MeshSection, MeshVertex, Surface};

/// Colour of a section whose model names no usable material.
pub const DEFAULT_TINT: [f32; 3] = [0.72, 0.72, 0.74];

/// How one material of a model is drawn.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Look {
    /// The shading inputs; texture keys are the ones the renderer will know the textures by, once
    /// they are (or will be) uploaded.
    pub surface: Surface,
    /// `false` for surfaces that are see-through (clear glass): the section is not drawn.
    pub visible: bool,
}

impl Look {
    /// A plain, slightly glossy grey, for models whose material is unknown.
    pub const DEFAULT: Look = Look {
        surface: Surface {
            tint: DEFAULT_TINT,
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
        },
        visible: true,
    };
}

/// Everything `SceneRenderer::add_mesh` takes.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Upload {
    /// Vertices in the model's local space.
    pub vertices: Vec<MeshVertex>,
    /// Triangle list.
    pub indices: Vec<u32>,
    /// One section per run of triangles with one material.
    pub sections: Vec<MeshSection>,
}

/// Convert an imported mesh. `look` gives how a section is drawn from the index of its material
/// in [`MeshData::materials`] (`None` for triangles without one). Sections with no triangles, and
/// sections whose look is not [`visible`](Look::visible), are dropped.
pub fn to_upload(mesh: &MeshData, look: impl Fn(Option<usize>) -> Look) -> Upload {
    let vertices = mesh
        .positions
        .iter()
        .zip(&mesh.normals)
        .zip(&mesh.uvs)
        .map(|((&position, &normal), &uv)| MeshVertex {
            position,
            normal,
            uv,
        })
        .collect();
    let sections = mesh
        .sections
        .iter()
        .filter(|s| s.index_count > 0)
        .filter_map(|s| {
            let look = look(s.material);
            look.visible.then_some(MeshSection {
                first_index: s.first_index,
                index_count: s.index_count,
                surface: look.surface,
            })
        })
        .collect();
    Upload {
        vertices,
        indices: mesh.indices.clone(),
        sections,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustiny_model::Section;

    fn triangle() -> MeshData {
        MeshData {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, -1.0]; 3],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            indices: vec![0, 1, 2, 2, 1, 0],
            sections: vec![
                Section {
                    material: Some(0),
                    first_index: 0,
                    index_count: 3,
                },
                Section {
                    material: None,
                    first_index: 3,
                    index_count: 3,
                },
                Section {
                    material: Some(1),
                    first_index: 6,
                    index_count: 0,
                },
            ],
            materials: vec!["Wall".into(), "Roof".into()],
            ..MeshData::default()
        }
    }

    #[test]
    fn vertices_and_sections_carry_over_and_empty_sections_are_dropped() {
        let up = to_upload(&triangle(), |m| match m {
            Some(0) => Look {
                surface: Surface {
                    tint: [1.0, 0.0, 0.0],
                    albedo: Some(4),
                    normal: Some(5),
                    cutoff: 0.5,
                    ..Surface::default()
                },
                visible: true,
            },
            _ => Look::DEFAULT,
        });
        assert_eq!(up.vertices.len(), 3);
        assert_eq!(up.vertices[1].position, [1.0, 0.0, 0.0]);
        assert_eq!(up.vertices[2].uv, [0.0, 1.0]);
        assert_eq!(up.vertices[0].normal, [0.0, 0.0, -1.0]);
        assert_eq!(up.indices, vec![0, 1, 2, 2, 1, 0]);
        assert_eq!(up.sections.len(), 2, "the empty section is gone");
        let first = up.sections[0].surface;
        assert_eq!(first.tint, [1.0, 0.0, 0.0]);
        assert_eq!((first.albedo, first.normal), (Some(4), Some(5)));
        assert_eq!(first.cutoff, 0.5);
        assert_eq!(up.sections[1].surface.tint, DEFAULT_TINT);
        assert_eq!(up.sections[1].first_index, 3);
    }

    #[test]
    fn invisible_sections_are_not_uploaded() {
        let up = to_upload(&triangle(), |m| Look {
            visible: m != Some(0),
            ..Look::DEFAULT
        });
        assert_eq!(up.sections.len(), 1);
        assert_eq!(up.sections[0].first_index, 3, "only the unmaterialled one");
    }
}
