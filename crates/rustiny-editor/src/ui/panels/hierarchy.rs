//! Hierarchy panel: read-only tree of the open scene's objects.

use super::style::{COLOR_ERROR, COLOR_WARN};
use crate::ui::{ProjectState, SceneState};
use dear_imgui_rs::{Ui, WindowKey};
use rustiny_assets::Guid;
use rustiny_scene::{ObjectId, ObjectKind, Scene, SceneObject};

/// Selection state of the Hierarchy panel.
#[derive(Debug, Default)]
pub struct State {
    /// The selected object (shared with the Inspector).
    pub selected: Option<ObjectId>,
    /// Show the list of load problems.
    pub show_issues: bool,
}

impl State {
    /// Forget the selection (a different scene was opened).
    pub fn clear(&mut self) {
        *self = Self::default();
    }
}

/// The Hierarchy panel.
///
/// Only expanded nodes are walked, so cost follows what is visible rather than scene size. A
/// single node with many thousands of children, expanded, would still be drawn in full; row
/// virtualisation is future work.
pub fn draw(
    ui: &Ui,
    key: &WindowKey,
    open: &mut bool,
    scene: &SceneState,
    project: &ProjectState,
    state: &mut State,
) {
    if !*open {
        return;
    }
    ui.window(key).opened(open).build(|| match scene {
        SceneState::None => {
            ui.text_disabled("No scene open.");
            ui.text_disabled("Double-click a .unity file in the Project panel,");
            ui.text_disabled("or start Rustiny with  --scene <path to a .unity file>.");
        }
        SceneState::Loading(path) => ui.text_disabled(format!("Loading {path} ...")),
        SceneState::Failed(why) => ui.text_colored(COLOR_ERROR, why),
        SceneState::Ready { scene, path, .. } => {
            draw_scene(ui, scene, path, project, state);
        }
    });
}

fn draw_scene(ui: &Ui, scene: &Scene, path: &str, project: &ProjectState, state: &mut State) {
    let stats = scene.stats();
    ui.text(scene.name());
    ui.text_disabled(format!(
        "{} objects | {:.1} MiB read in {:.0} ms",
        scene.len(),
        stats.bytes as f64 / (1024.0 * 1024.0),
        stats.duration.as_secs_f64() * 1000.0
    ));
    if ui.is_item_hovered() {
        ui.tooltip_text(path);
    }
    let issues = scene.issues().len() as u64 + stats.issues_dropped;
    if issues > 0 {
        ui.text_colored(COLOR_WARN, format!("{issues} problems found"));
        ui.same_line();
        ui.checkbox("show", &mut state.show_issues);
        if state.show_issues {
            for issue in scene.issues() {
                ui.text_colored(COLOR_WARN, issue.to_string());
            }
        }
    }
    ui.separator();
    for &root in scene.roots() {
        draw_object(ui, scene, root, project, state);
    }
}

fn draw_object(ui: &Ui, scene: &Scene, id: ObjectId, project: &ProjectState, state: &mut State) {
    let Some(object) = scene.object(id) else {
        return;
    };
    let label = display_name(object, project);
    let id_text = id.to_string();
    let node = ui
        .tree_node_config(id_text.as_str())
        .label(label.as_str())
        .open_on_arrow(true)
        .span_full_width(true)
        .leaf(object.children.is_empty())
        .selected(state.selected == Some(id))
        .push();
    if ui.is_item_clicked() {
        state.selected = Some(id);
    }
    if ui.is_item_hovered() {
        ui.tooltip_text(tooltip(scene, id, object));
    }
    if let Some(token) = node {
        for &child in &object.children {
            draw_object(ui, scene, child, project, state);
        }
        drop(token);
    }
}

/// The name shown for an object: its own, or for an unnamed prefab instance the file name of its
/// source prefab when the project index knows it.
pub fn display_name(object: &SceneObject, project: &ProjectState) -> String {
    match &object.kind {
        ObjectKind::GameObject => {
            if object.name.is_empty() {
                "(unnamed)".to_owned()
            } else {
                object.name.clone()
            }
        }
        ObjectKind::PrefabInstance { source_guid } => {
            let name = if object.name.is_empty() {
                source_prefab_name(source_guid.as_deref(), project)
                    .unwrap_or_else(|| "(prefab instance)".to_owned())
            } else {
                object.name.clone()
            };
            format!("[prefab] {name}")
        }
    }
}

/// File name (without extension) of the prefab asset with this GUID, if the project is indexed.
pub fn source_prefab_name(guid: Option<&str>, project: &ProjectState) -> Option<String> {
    let ProjectState::Ready { db, .. } = project else {
        return None;
    };
    let guid: Guid = guid?.parse().ok()?;
    let entry = db.by_guid(guid)?;
    let name = entry.name();
    Some(name.strip_suffix(".prefab").unwrap_or(name).to_owned())
}

fn tooltip(scene: &Scene, id: ObjectId, object: &SceneObject) -> String {
    let below = scene.descendant_count(id);
    match &object.kind {
        ObjectKind::GameObject => format!(
            "File ID {}\n{} components, {below} objects below",
            object.file_id,
            object.components.len()
        ),
        ObjectKind::PrefabInstance { source_guid } => format!(
            "Prefab instance, file ID {}\nSource prefab GUID: {}\nContents are not expanded yet.",
            object.file_id,
            source_guid.as_deref().unwrap_or("unknown")
        ),
    }
}
