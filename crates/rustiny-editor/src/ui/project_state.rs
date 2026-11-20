//! What the Project panel is showing.

use rustiny_assets::{AssetDatabase, UnityProject};
use std::sync::Arc;

/// Lifecycle of the opened project, as seen by the UI.
#[derive(Debug, Clone, Default)]
pub enum ProjectState {
    /// No project was opened (start with `--project <path>`).
    #[default]
    None,
    /// The background scan of this path is running.
    Scanning(String),
    /// A project is open and indexed.
    Ready {
        /// Detected project.
        project: UnityProject,
        /// Its asset index.
        db: Arc<AssetDatabase>,
        /// Whether file changes are being watched (a change triggers a rescan).
        watching: bool,
    },
    /// The path is not a usable project, or scanning failed.
    Failed(String),
}
