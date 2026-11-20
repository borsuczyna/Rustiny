//! Unity's own serialized `Mesh` asset (`.asset`, class 43) as written in the text format.
//!
//! Terrain tiles, roads and many generated meshes of a project are stored this way rather than
//! as model files. The text form keeps the index buffer and the vertex data as one hex string
//! each, next to a description of the vertex channels. Compressed meshes (`m_MeshCompression`
//! other than 0) and meshes whose data lives in a separate `.resS` stream are reported as
//! unsupported rather than guessed at.

use crate::mesh::{MeshData, Section};

/// Why a mesh asset could not be read.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MeshAssetError {
    /// The text has no `Mesh:` document.
    #[error("not a Unity Mesh asset")]
    NotAMesh,
    /// A field the format always has is missing or unreadable.
    #[error("mesh asset field `{0}` is missing or malformed")]
    Field(&'static str),
    /// The mesh uses a feature that is not implemented.
    #[error("unsupported mesh asset: {0}")]
    Unsupported(&'static str),
    /// The buffers do not match the sizes the channels and sub-meshes describe.
    #[error("inconsistent mesh asset: {0}")]
    Inconsistent(&'static str),
}

/// Channel order in `m_Channels`.
const POSITION: usize = 0;
const NORMAL: usize = 1;
const TANGENT: usize = 2;
const COLOR: usize = 3;
const UV0: usize = 4;
const UV1: usize = 5;

#[derive(Debug, Clone, Copy, Default)]
struct Channel {
    stream: usize,
    offset: usize,
    format: u32,
    dimension: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct SubMesh {
    first_byte: usize,
    index_count: usize,
    topology: u32,
    base_vertex: i64,
    first_vertex: usize,
    vertex_count: usize,
}

fn hex_value(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Decode a hex string (the YAML form of a byte array).
fn decode_hex(text: &str) -> Option<Vec<u8>> {
    let bytes = text.trim().as_bytes();
    let (pairs, rest) = bytes.as_chunks::<2>();
    if !rest.is_empty() {
        return None;
    }
    pairs
        .iter()
        .map(|pair| Some(hex_value(pair[0])? << 4 | hex_value(pair[1])?))
        .collect()
}

/// The text after `key:` on the first line (at any indentation) that starts with it.
fn value_of<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    text.lines().find_map(|line| {
        line.trim_start()
            .strip_prefix(key)
            .and_then(|rest| rest.strip_prefix(':'))
            .map(str::trim)
    })
}

fn number<T: std::str::FromStr>(text: &str, key: &'static str) -> Result<T, MeshAssetError> {
    value_of(text, key)
        .and_then(|v| v.parse().ok())
        .ok_or(MeshAssetError::Field(key))
}

/// The lines of the block that starts at the line `  <key>:` (the lines indented deeper, or list
/// items at the same depth), as one string.
fn block<'a>(text: &'a str, key: &str, indent: usize) -> Option<&'a str> {
    let header = format!("{}{key}:", " ".repeat(indent));
    let mut start = None;
    let mut offset = 0;
    for line in text.split_inclusive('\n') {
        let trimmed = line.trim_end_matches(['\n', '\r']);
        match start {
            None => {
                if trimmed.trim_end() == header || trimmed.starts_with(&format!("{header} ")) {
                    start = Some(offset + line.len());
                }
            }
            Some(begin) => {
                let depth = trimmed.len() - trimmed.trim_start().len();
                let is_item = trimmed.trim_start().starts_with("- ");
                if !trimmed.is_empty() && (depth < indent || (depth == indent && !is_item)) {
                    return Some(&text[begin..offset]);
                }
            }
        }
        offset += line.len();
    }
    start.map(|begin| &text[begin..])
}

/// Split a list block into items (each starts at a `- ` line at its own indentation).
fn items(block: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for line in block.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("- ") {
            out.push(format!("{rest}\n"));
        } else if let Some(last) = out.last_mut() {
            last.push_str(trimmed);
            last.push('\n');
        }
    }
    out
}

fn format_size(format: u32) -> Option<usize> {
    Some(match format {
        0 | 10 | 11 => 4,
        1 | 4 | 5 | 8 | 9 => 2,
        2 | 3 | 6 | 7 => 1,
        _ => return None,
    })
}

/// Read one component of a channel as `f32` (normalised formats map to 0..1 or -1..1).
fn component(data: &[u8], at: usize, format: u32) -> Option<f32> {
    let b = |n: usize| data.get(at..at + n);
    Some(match format {
        0 => f32::from_le_bytes(b(4)?.try_into().ok()?),
        1 => f16_to_f32(u16::from_le_bytes(b(2)?.try_into().ok()?)),
        2 => f32::from(*b(1)?.first()?) / 255.0,
        3 => (f32::from(i8::from_le_bytes([*b(1)?.first()?])) / 127.0).max(-1.0),
        4 => f32::from(u16::from_le_bytes(b(2)?.try_into().ok()?)) / 65535.0,
        5 => (f32::from(i16::from_le_bytes(b(2)?.try_into().ok()?)) / 32767.0).max(-1.0),
        6 => f32::from(*b(1)?.first()?),
        7 => f32::from(i8::from_le_bytes([*b(1)?.first()?])),
        8 => f32::from(u16::from_le_bytes(b(2)?.try_into().ok()?)),
        9 => f32::from(i16::from_le_bytes(b(2)?.try_into().ok()?)),
        10 => u32::from_le_bytes(b(4)?.try_into().ok()?) as f32,
        11 => i32::from_le_bytes(b(4)?.try_into().ok()?) as f32,
        _ => return None,
    })
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exponent = u32::from((h >> 10) & 0x1f);
    let fraction = u32::from(h & 0x3ff);
    let bits = match (exponent, fraction) {
        (0, 0) => sign,
        (0, f) => {
            // Subnormal: normalise.
            let shift = f.leading_zeros() - 21;
            sign | ((127 - 15 - shift + 1) << 23) | (((f << shift) & 0x3ff) << 13)
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, f) => sign | 0x7f80_0000 | (f << 13),
        (e, f) => sign | ((e + 127 - 15) << 23) | (f << 13),
    };
    f32::from_bits(bits)
}

/// Read a Unity `Mesh` asset in the text serialization into triangle data in Unity's space. Each
/// sub-mesh becomes a [`Section`] whose `material` is the sub-mesh index (the renderer's
/// `m_Materials` slot), so `materials` has one placeholder name per sub-mesh.
pub fn import_mesh_asset(text: &str) -> Result<MeshData, MeshAssetError> {
    if !text.contains("\nMesh:") && !text.starts_with("Mesh:") {
        return Err(MeshAssetError::NotAMesh);
    }
    if number::<u32>(text, "m_MeshCompression").unwrap_or(0) != 0 {
        return Err(MeshAssetError::Unsupported("compressed mesh"));
    }
    if let Some(stream) = block(text, "m_StreamData", 2)
        && number::<u64>(stream, "size").unwrap_or(0) != 0
    {
        return Err(MeshAssetError::Unsupported("mesh data in a .resS stream"));
    }

    let sub_block = block(text, "m_SubMeshes", 2).ok_or(MeshAssetError::Field("m_SubMeshes"))?;
    let sub_meshes: Vec<SubMesh> = items(sub_block)
        .iter()
        .map(|item| {
            Ok(SubMesh {
                first_byte: number(item, "firstByte")?,
                index_count: number(item, "indexCount")?,
                topology: number(item, "topology")?,
                base_vertex: number(item, "baseVertex")?,
                first_vertex: number(item, "firstVertex")?,
                vertex_count: number(item, "vertexCount")?,
            })
        })
        .collect::<Result<_, MeshAssetError>>()?;

    let vertex_block =
        block(text, "m_VertexData", 2).ok_or(MeshAssetError::Field("m_VertexData"))?;
    let vertex_count: usize = number(vertex_block, "m_VertexCount")?;
    let channel_block =
        block(vertex_block, "m_Channels", 4).ok_or(MeshAssetError::Field("m_Channels"))?;
    let channels: Vec<Channel> = items(channel_block)
        .iter()
        .map(|item| {
            Ok(Channel {
                stream: number(item, "stream")?,
                offset: number(item, "offset")?,
                format: number(item, "format")?,
                dimension: number::<usize>(item, "dimension")? & 0xf,
            })
        })
        .collect::<Result<_, MeshAssetError>>()?;
    let data = decode_hex(
        value_of(vertex_block, "_typelessdata").ok_or(MeshAssetError::Field("_typelessdata"))?,
    )
    .ok_or(MeshAssetError::Field("_typelessdata"))?;
    let index_bytes =
        decode_hex(value_of(text, "m_IndexBuffer").ok_or(MeshAssetError::Field("m_IndexBuffer"))?)
            .ok_or(MeshAssetError::Field("m_IndexBuffer"))?;
    let wide_indices = number::<u32>(text, "m_IndexFormat").unwrap_or(0) == 1;

    // Streams are stored one after another, each `stride * vertex_count` bytes, padded to 16.
    let mut stream_start = Vec::new();
    let mut stream_stride = Vec::new();
    let streams = channels
        .iter()
        .filter(|c| c.dimension > 0)
        .map(|c| c.stream)
        .max()
        .map_or(0, |s| s + 1);
    let mut cursor = 0usize;
    for stream in 0..streams {
        let stride = channels
            .iter()
            .filter(|c| c.stream == stream && c.dimension > 0)
            .map(|c| c.offset + c.dimension * format_size(c.format).unwrap_or(4))
            .max()
            .unwrap_or(0);
        let stride = (stride + 3) & !3;
        stream_start.push(cursor);
        stream_stride.push(stride);
        cursor += stride * vertex_count;
        cursor = (cursor + 15) & !15;
    }

    let read = |channel: usize, vertex: usize, lane: usize| -> Option<f32> {
        let c = channels.get(channel)?;
        if c.dimension == 0 || lane >= c.dimension {
            return None;
        }
        let size = format_size(c.format)?;
        let at = stream_start.get(c.stream)?
            + vertex * stream_stride.get(c.stream)?
            + c.offset
            + lane * size;
        component(&data, at, c.format)
    };
    let has = |channel: usize| channels.get(channel).is_some_and(|c| c.dimension > 0);
    if !has(POSITION) {
        return Err(MeshAssetError::Inconsistent("no position channel"));
    }
    if cursor > data.len() + 15 {
        return Err(MeshAssetError::Inconsistent(
            "vertex data is shorter than described",
        ));
    }

    let mut mesh = MeshData {
        positions: Vec::with_capacity(vertex_count),
        normals: Vec::with_capacity(vertex_count),
        uvs: Vec::with_capacity(vertex_count),
        ..MeshData::default()
    };
    for v in 0..vertex_count {
        let p = |lane| read(POSITION, v, lane).ok_or(MeshAssetError::Inconsistent("position"));
        mesh.positions.push([p(0)?, p(1)?, p(2)?]);
        mesh.normals.push(if has(NORMAL) {
            let n = |lane| read(NORMAL, v, lane).unwrap_or(0.0);
            [n(0), n(1), n(2)]
        } else {
            [0.0, 1.0, 0.0]
        });
        mesh.uvs.push(if has(UV0) {
            [
                read(UV0, v, 0).unwrap_or(0.0),
                read(UV0, v, 1).unwrap_or(0.0),
            ]
        } else {
            [0.0; 2]
        });
        if has(UV1) {
            mesh.uv1.push([
                read(UV1, v, 0).unwrap_or(0.0),
                read(UV1, v, 1).unwrap_or(0.0),
            ]);
        }
        if has(TANGENT) {
            let t = |lane| read(TANGENT, v, lane).unwrap_or(0.0);
            mesh.tangents.push([
                t(0),
                t(1),
                t(2),
                if channels[TANGENT].dimension > 3 {
                    t(3)
                } else {
                    1.0
                },
            ]);
        }
        if has(COLOR) {
            let c = |lane| read(COLOR, v, lane).unwrap_or(1.0);
            mesh.colors.push([
                c(0),
                c(1),
                c(2),
                if channels[COLOR].dimension > 3 {
                    c(3)
                } else {
                    1.0
                },
            ]);
        }
    }

    let index_size = if wide_indices { 4 } else { 2 };
    let index_at = |i: usize| -> Option<u32> {
        let at = i * index_size;
        let bytes = index_bytes.get(at..at + index_size)?;
        Some(if wide_indices {
            u32::from_le_bytes(bytes.try_into().ok()?)
        } else {
            u32::from(u16::from_le_bytes(bytes.try_into().ok()?))
        })
    };
    for (slot, sub) in sub_meshes.iter().enumerate() {
        mesh.materials.push(format!("submesh{slot}"));
        let first = mesh.indices.len();
        let triangles = match sub.topology {
            0 => sub.index_count / 3 * 3,
            _ => {
                return Err(MeshAssetError::Unsupported(
                    "sub-mesh topology other than triangles",
                ));
            }
        };
        for i in 0..triangles {
            let raw = index_at(sub.first_byte / index_size + i)
                .ok_or(MeshAssetError::Inconsistent("index buffer"))?;
            let vertex = i64::from(raw) + sub.base_vertex;
            let valid = usize::try_from(vertex).is_ok_and(|v| v < vertex_count);
            if !valid {
                return Err(MeshAssetError::Inconsistent("index out of range"));
            }
            mesh.indices.push(vertex as u32);
        }
        let _ = (sub.first_vertex, sub.vertex_count);
        mesh.sections.push(Section {
            material: Some(slot),
            first_index: first as u32,
            index_count: (mesh.indices.len() - first) as u32,
        });
    }
    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// A two-triangle quad in two sub-meshes: positions + normals + uv0 in stream 0 (32 bytes per
    /// vertex), 16-bit indices.
    fn quad() -> String {
        let mut data = Vec::new();
        let corners = [[0.0f32, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        for [x, z] in corners {
            for f in [x, 2.0, z, 0.0, 1.0, 0.0, x, z] {
                data.extend_from_slice(&f.to_le_bytes());
            }
        }
        let indices: Vec<u8> = [0u16, 1, 2, 0, 2, 3]
            .iter()
            .flat_map(|i| i.to_le_bytes())
            .collect();
        let channel = |stream, offset, dim| {
            format!(
                "    - stream: {stream}\n      offset: {offset}\n      format: 0\n      dimension: {dim}\n"
            )
        };
        format!(
            "%YAML 1.1\n--- !u!43 &4300000\nMesh:\n  m_Name: Quad\n  serializedVersion: 12\n  m_SubMeshes:\n  - serializedVersion: 2\n    firstByte: 0\n    indexCount: 3\n    topology: 0\n    baseVertex: 0\n    firstVertex: 0\n    vertexCount: 4\n    localAABB:\n      m_Center: {{x: 0, y: 0, z: 0}}\n  - serializedVersion: 2\n    firstByte: 6\n    indexCount: 3\n    topology: 0\n    baseVertex: 0\n    firstVertex: 0\n    vertexCount: 4\n  m_MeshCompression: 0\n  m_IndexFormat: 0\n  m_IndexBuffer: {}\n  m_VertexData:\n    serializedVersion: 3\n    m_VertexCount: 4\n    m_Channels:\n{}{}{}{}{}{}    m_DataSize: {}\n    _typelessdata: {}\n  m_StreamData:\n    serializedVersion: 2\n    offset: 0\n    size: 0\n    path: \n",
            hex(&indices),
            channel(0, 0, 3),
            channel(0, 12, 3),
            channel(0, 0, 0),
            channel(0, 0, 0),
            channel(0, 24, 2),
            channel(0, 0, 0),
            data.len(),
            hex(&data)
        )
    }

    #[test]
    fn a_quad_with_two_sub_meshes_is_read() {
        let mesh = import_mesh_asset(&quad()).unwrap();
        assert_eq!(mesh.positions.len(), 4);
        assert_eq!(mesh.positions[2], [1.0, 2.0, 1.0]);
        assert_eq!(mesh.normals[1], [0.0, 1.0, 0.0]);
        assert_eq!(mesh.uvs[3], [0.0, 1.0]);
        assert_eq!(mesh.indices, vec![0, 1, 2, 0, 2, 3]);
        assert_eq!(mesh.sections.len(), 2);
        assert_eq!(mesh.sections[1].first_index, 3);
        assert_eq!(mesh.sections[1].material, Some(1));
    }

    #[test]
    fn compressed_and_foreign_text_are_refused() {
        let compressed = quad().replace("m_MeshCompression: 0", "m_MeshCompression: 2");
        assert!(matches!(
            import_mesh_asset(&compressed),
            Err(MeshAssetError::Unsupported(_))
        ));
        assert_eq!(
            import_mesh_asset("--- !u!21 &1\nMaterial:\n  m_Name: x\n"),
            Err(MeshAssetError::NotAMesh)
        );
    }

    #[test]
    fn an_index_past_the_vertices_is_an_error_not_a_panic() {
        let bad = quad().replace(
            "indexCount: 3\n    topology: 0\n    baseVertex: 0",
            "indexCount: 3\n    topology: 0\n    baseVertex: 9",
        );
        assert!(matches!(
            import_mesh_asset(&bad),
            Err(MeshAssetError::Inconsistent(_))
        ));
    }

    #[test]
    fn half_floats_decode() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!((f16_to_f32(0x0001) - 5.960_464_5e-8).abs() < 1e-12);
    }
}
