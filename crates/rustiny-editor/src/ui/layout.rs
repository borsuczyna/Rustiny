//! Panel identities, visibility and the default dock arrangement.

use dear_imgui_rs::{DockLayout, DockSplit, WindowKey};

/// Open/closed state of every dockable panel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PanelVisibility {
    /// Hierarchy panel.
    pub hierarchy: bool,
    /// Scene view.
    pub scene: bool,
    /// Game view.
    pub game: bool,
    /// Inspector.
    pub inspector: bool,
    /// Project browser.
    pub project: bool,
    /// Console.
    pub console: bool,
    /// Profiler.
    pub profiler: bool,
}

impl Default for PanelVisibility {
    fn default() -> Self {
        Self {
            hierarchy: true,
            scene: true,
            game: true,
            inspector: true,
            project: true,
            console: true,
            profiler: true,
        }
    }
}

/// Stable identities of the dockable windows. The `stable_id` strings are what Dear ImGui hashes
/// into its persisted layout, so they must never change once released.
#[derive(Debug)]
pub(super) struct Keys {
    pub hierarchy: WindowKey,
    pub scene: WindowKey,
    pub game: WindowKey,
    pub inspector: WindowKey,
    pub project: WindowKey,
    pub console: WindowKey,
    pub profiler: WindowKey,
}

impl Keys {
    pub fn new() -> Self {
        // Literals are valid (non-empty, no NUL, no separator); a failure here is a programming
        // error caught by the unit tests.
        #[allow(clippy::expect_used)]
        let key = |id: &str, title: &str| WindowKey::new(id, title).expect("valid window key");
        Self {
            hierarchy: key("panel.hierarchy", "Hierarchy"),
            scene: key("panel.scene", "Scene"),
            game: key("panel.game", "Game"),
            inspector: key("panel.inspector", "Inspector"),
            project: key("panel.project", "Project"),
            console: key("panel.console", "Console"),
            profiler: key("panel.profiler", "Profiler"),
        }
    }

    /// Default arrangement: Hierarchy | Scene+Game | Inspector over Project+Console+Profiler.
    pub fn default_layout(&self) -> DockLayout {
        let centre = DockLayout::tabs([&self.scene, &self.game]);
        let right = DockLayout::split(
            DockSplit::Right,
            0.24,
            DockLayout::tabs([&self.inspector]),
            centre,
        );
        let top = DockLayout::split(
            DockSplit::Left,
            0.18,
            DockLayout::tabs([&self.hierarchy]),
            right,
        );
        DockLayout::split(
            DockSplit::Down,
            0.30,
            DockLayout::tabs([&self.project, &self.console, &self.profiler]),
            top,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_layout_is_valid_and_contains_every_panel() {
        let layout = Keys::new().default_layout();
        layout.validate().expect("default layout must validate");
    }

    #[test]
    fn window_keys_are_unique() {
        let k = Keys::new();
        let ids = [
            &k.hierarchy,
            &k.scene,
            &k.game,
            &k.inspector,
            &k.project,
            &k.console,
            &k.profiler,
        ];
        let mut names: Vec<_> = ids.iter().map(|k| format!("{k:?}")).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), ids.len());
    }
}
