//! What the Hierarchy and Inspector panels are showing.

use crate::meshes::PickShapes;
use rustiny_scene::Scene;
use std::sync::Arc;

/// Lifecycle of the open scene, as seen by the UI.
#[derive(Debug, Clone, Default)]
pub enum SceneState {
    /// No scene was opened.
    #[default]
    None,
    /// The background load of this path is running.
    Loading(String),
    /// A scene is loaded. It is read-only and shared with the loader's result.
    Ready {
        /// The loaded scene.
        scene: Arc<Scene>,
        /// The file it came from, for display.
        path: String,
        /// How objects with imported meshes are clicked; filled in as the models arrive.
        shapes: Arc<PickShapes>,
    },
    /// The file could not be read or is not a text-serialized scene.
    Failed(String),
}
