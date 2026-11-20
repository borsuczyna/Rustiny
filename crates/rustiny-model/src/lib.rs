//! Importing 3D model files into triangle meshes in Unity's coordinate space.
//!
//! * [`fbx`] - binary FBX: container reader, geometry decoding, unit and axis conversion.
//! * [`unity_mesh`] - Unity's serialized `Mesh` assets (`.asset`).
//! * [`mesh`] - the output: [`MeshData`] with per-material [`Section`]s and [`Bounds`].
//!
//! Everything here is pure computation on bytes (no files, no GPU, no threads), so it can run on
//! any worker thread and is tested without a window.

#![cfg_attr(test, allow(clippy::unwrap_used))]

pub mod fbx;
mod math;
pub mod mesh;
pub mod unity_mesh;

pub use fbx::{FbxError, ImportSettings};
pub use mesh::{Bounds, MeshData, Section};
pub use unity_mesh::{MeshAssetError, import_mesh_asset};
