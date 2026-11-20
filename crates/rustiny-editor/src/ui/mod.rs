//! Editor user interface: dockspace, menu bar and panels.
//!
//! * `editor`   - [`EditorUi`], the owner of all UI state, plus the menu bar.
//! * `layout`   - panel identities, visibility and the default dock arrangement.
//! * `panels`   - one module per dockable panel.
//! * `context`  - data passed between the application shell and the UI.
//! * `theme`    - colours and style.
//!
//! Panels with real data behind them: Console, Profiler and Project. The rest are explicit
//! placeholders that name the milestone in which they gain functionality.

mod context;
mod editor;
mod layout;
pub mod panels;
mod project_state;
mod scene_state;
pub mod theme;

pub use context::{GpuSummary, SceneViewRequest, UiContext, UiRequests};
pub use editor::EditorUi;
pub use layout::PanelVisibility;
pub use project_state::ProjectState;
pub use scene_state::SceneState;
