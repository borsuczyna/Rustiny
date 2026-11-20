//! Unity scene model and a streaming reader for UnityYAML (`.unity`) files.
//!
//! * `unity_yaml` - the file format: splitting a file into documents and reading their fields.
//! * `model`      - [`Scene`] and [`SceneObject`]: the hierarchy as the editor sees it.
//! * `load`       - turning documents into a [`Scene`].
//! * `classes`    - names of Unity's numeric class IDs.
//! * `material`   - the handful of `.mat` properties a viewport needs.
//!
//! Read-only: scenes are loaded, never written. Unity scenes can be hundreds of megabytes, so the
//! reader streams the file one document at a time and keeps only what the model needs.

#![cfg_attr(test, allow(clippy::unwrap_used))]

pub mod classes;
pub mod lighting;
mod load;
pub mod material;
pub mod math;
pub mod model;
pub mod prefab;
pub mod unity_yaml;
pub mod volume;

pub use load::{SceneError, load_scene, read_scene};
pub use material::{Material, MaterialTexture, parse_material, parse_material_object};
pub use math::Mat4;
pub use model::{
    Component, LoadStats, ObjectId, ObjectKind, Scene, SceneIssue, SceneLight, SceneLighting,
    SceneObject, TransformData,
};
pub use prefab::{AssetRef, Overrides, RendererState};
