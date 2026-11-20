//! Inspector panel: read-only facts about the object selected in the Hierarchy.
//!
//! Only what the scene loader reads is shown (name, flags, local transform, component types).
//! Component *values* are not parsed yet, and nothing here can be edited.

use super::hierarchy::{self, State as HierarchyState};
use crate::ui::{ProjectState, SceneState};
use dear_imgui_rs::{Ui, WindowKey};
use rustiny_scene::classes;
use rustiny_scene::{ObjectKind, SceneObject};

/// The Inspector panel.
pub fn draw(
    ui: &Ui,
    key: &WindowKey,
    open: &mut bool,
    scene: &SceneState,
    project: &ProjectState,
    selection: &HierarchyState,
) {
    if !*open {
        return;
    }
    ui.window(key).opened(open).build(|| {
        let SceneState::Ready { scene, .. } = scene else {
            ui.text_disabled("No scene open.");
            return;
        };
        let Some(object) = selection.selected.and_then(|id| scene.object(id)) else {
            ui.text_disabled("Select an object in the Hierarchy.");
            return;
        };
        draw_object(ui, object, project);
    });
}

fn draw_object(ui: &Ui, object: &SceneObject, project: &ProjectState) {
    ui.text(hierarchy::display_name(object, project));
    ui.text_disabled(format!("File ID {}", object.file_id));
    ui.separator();

    match &object.kind {
        ObjectKind::PrefabInstance { source_guid } => {
            ui.text("Prefab instance");
            let source = source_guid.as_deref();
            match hierarchy::source_prefab_name(source, project) {
                Some(name) => ui.text(format!("Source: {name}")),
                None => ui.text_disabled(format!("Source GUID: {}", source.unwrap_or("unknown"))),
            }
            ui.text_disabled("Overrides and the prefab's contents are not read yet.");
        }
        ObjectKind::GameObject => {
            ui.text(format!(
                "Active: {}",
                if object.active { "yes" } else { "no" }
            ));
            ui.text(format!("Tag: {}    Layer: {}", object.tag, object.layer));
            if let Some(t) = &object.transform {
                ui.separator();
                ui.text(if t.is_rect {
                    "RectTransform (local)"
                } else {
                    "Transform (local)"
                });
                ui.text(format!(
                    "Position  {:.3}  {:.3}  {:.3}",
                    t.position[0], t.position[1], t.position[2]
                ));
                ui.text(format!(
                    "Rotation  {:.3}  {:.3}  {:.3}  {:.3}  (quaternion)",
                    t.rotation[0], t.rotation[1], t.rotation[2], t.rotation[3]
                ));
                ui.text(format!(
                    "Scale     {:.3}  {:.3}  {:.3}",
                    t.scale[0], t.scale[1], t.scale[2]
                ));
            }
            ui.separator();
            ui.text(format!("Components ({})", object.components.len()));
            for component in &object.components {
                match component.class_id {
                    Some(class) => ui.bullet_text(classes::display_name(class)),
                    None => ui.bullet_text(format!("missing (file ID {})", component.file_id)),
                }
            }
            ui.text_disabled("Component values are not read yet.");
        }
    }
}
