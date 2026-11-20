//! Binary FBX import.
//!
//! * `reader.rs`   - the container: nodes and typed properties.
//! * `geometry.rs` - one `Geometry` node as triangles (polygons, normals, UVs, material slots).
//! * `import.rs`   - objects, connections, node transforms and the conversion to Unity space.
//!
//! What this is and is not: it reproduces the parts of Unity's FBX importer that decide where a
//! model's triangles end up (unit scale, axis conversion to Unity's left-handed Y-up space, the
//! node hierarchy's transforms) and flattens the whole file into one mesh. It does not read
//! skins, animation, cameras, lights or blend shapes, and it has not been compared against
//! Unity's output vertex by vertex.

mod file_id;
mod geometry;
mod import;
mod reader;
#[cfg(test)]
mod tests;

pub use file_id::{ROOT_PATH, mesh_file_id, object_id, xxh64};
pub use import::{FbxNode, ImportSettings, Part, import, import_part, nodes};
pub use reader::{Node, Prop, parse};

/// Why a file could not be imported.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FbxError {
    /// The file does not start with the binary FBX header (ASCII FBX is not supported).
    #[error("not a binary FBX file")]
    NotBinaryFbx,
    /// The file ends inside a record, or a record's sizes do not add up.
    #[error("the FBX file is truncated or corrupt")]
    Truncated,
    /// Nodes are nested deeper than any real file.
    #[error("FBX nodes are nested too deeply")]
    TooDeep,
    /// An unknown property type letter.
    #[error("unknown FBX property type {0:#04x}")]
    BadPropertyType(u8),
    /// An unknown array encoding.
    #[error("unknown FBX array encoding {0}")]
    BadEncoding(u32),
    /// An array claims to be larger than is reasonable.
    #[error("an FBX array is implausibly large")]
    ArrayTooLarge,
    /// Inflating a compressed array failed.
    #[error("cannot decompress an FBX array: {0}")]
    Decompress(String),
    /// The file has no mesh with triangles.
    #[error("the FBX file contains no triangle mesh")]
    NoMesh,
}
