//! `EditorUi`: owns all UI state and builds one frame of the editor.

mod menu;

use super::layout::{Keys, PanelVisibility};
use super::{ProjectState, SceneState, UiContext, UiRequests, panels};
use dear_imgui_rs::{DockLayout, DockLayoutApply, Ui};

/// All editor UI state.
#[derive(Debug)]
pub struct EditorUi {
    /// Which panels are open.
    pub visibility: PanelVisibility,
    keys: Keys,
    layout: DockLayout,
    apply_layout: DockLayoutApply,
    console: panels::console::State,
    project: ProjectState,
    project_panel: panels::project::State,
    scene: SceneState,
    hierarchy_panel: panels::hierarchy::State,
    scene_view: panels::scene_view::State,
    profiler_plot: Vec<f32>,
    show_about: bool,
}

impl Default for EditorUi {
    fn default() -> Self {
        Self::new()
    }
}

impl EditorUi {
    /// Create the UI state with the default layout.
    pub fn new() -> Self {
        let keys = Keys::new();
        let layout = keys.default_layout();
        Self {
            visibility: PanelVisibility::default(),
            keys,
            layout,
            apply_layout: DockLayoutApply::IfMissing,
            console: panels::console::State::default(),
            project: ProjectState::None,
            project_panel: panels::project::State::default(),
            scene: SceneState::None,
            hierarchy_panel: panels::hierarchy::State::default(),
            scene_view: panels::scene_view::State::default(),
            profiler_plot: Vec::new(),
            show_about: false,
        }
    }

    /// Show a different project in the Project panel.
    pub fn set_project(&mut self, project: ProjectState) {
        self.project_panel = panels::project::State {
            focus_in_frames: if matches!(project, ProjectState::None) {
                0
            } else {
                3
            },
            ..panels::project::State::default()
        };
        self.project = project;
    }

    /// Replace the project data without disturbing the panel (selection and open folders stay).
    pub fn update_project(&mut self, project: ProjectState) {
        self.project = project;
    }

    /// Show a scene (or its loading/failed state) in the Hierarchy and Inspector.
    pub fn set_scene(&mut self, scene: SceneState) {
        self.hierarchy_panel.clear();
        self.scene_view.clear();
        self.scene = scene;
    }

    /// Place the Scene view's camera when the next scene is shown (command line `--view`).
    pub fn set_initial_view(&mut self, view: Option<[f32; 5]>) {
        self.scene_view.set_initial_view(view);
    }

    /// Replace the dock arrangement with the default one on the next frame.
    pub fn reset_layout(&mut self) {
        self.visibility = PanelVisibility::default();
        self.apply_layout = DockLayoutApply::Replace;
    }

    /// Build the whole editor UI for one frame.
    pub fn draw(&mut self, ui: &Ui, cx: &UiContext<'_>) -> UiRequests {
        let mut requests = UiRequests::default();
        self.main_menu(ui, cx, &mut requests);

        let apply = std::mem::take(&mut self.apply_layout);
        if let Err(e) = ui.dockspace().layout(&self.layout, apply).build() {
            // A layout error must never take the editor down; log once per occurrence.
            tracing::error!("dockspace: {e}");
        }

        let k = &self.keys;
        let v = &mut self.visibility;
        panels::hierarchy::draw(
            ui,
            &k.hierarchy,
            &mut v.hierarchy,
            &self.scene,
            &self.project,
            &mut self.hierarchy_panel,
        );
        self.scene_view.set_texture(cx.scene_texture);
        panels::scene_view::draw(
            ui,
            &k.scene,
            &mut v.scene,
            &self.scene,
            &self.project,
            &mut self.hierarchy_panel,
            &mut self.scene_view,
        );
        requests.scene_view = self.scene_view.take_request();
        panels::placeholder::draw(ui, &k.game, &mut v.game, "Game view", "Milestone 5");
        panels::inspector::draw(
            ui,
            &k.inspector,
            &mut v.inspector,
            &self.scene,
            &self.project,
            &self.hierarchy_panel,
        );
        panels::project::draw(
            ui,
            &k.project,
            &mut v.project,
            &self.project,
            &mut self.project_panel,
            &mut requests,
        );
        panels::console::draw(ui, &k.console, &mut v.console, cx.log, &mut self.console);
        panels::profiler::draw(
            ui,
            &k.profiler,
            &mut v.profiler,
            cx,
            &mut self.profiler_plot,
            &mut requests,
        );

        self.about(ui);
        requests
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_restores_visibility_and_requests_replace() {
        let mut ui = EditorUi::new();
        ui.visibility.console = false;
        ui.reset_layout();
        assert!(ui.visibility.console);
        assert_eq!(ui.apply_layout, DockLayoutApply::Replace);
    }
}
