//! Main menu bar and the About window.

use super::EditorUi;
use crate::ui::{UiContext, UiRequests};
use dear_imgui_rs::Ui;
use rustiny_core::config::Theme;

impl EditorUi {
    pub(super) fn main_menu(&mut self, ui: &Ui, _cx: &UiContext<'_>, requests: &mut UiRequests) {
        ui.main_menu_bar(|| {
            ui.menu("File", || {
                if ui.menu_item_with_shortcut("Quit", "Alt+F4") {
                    requests.quit = true;
                }
            });
            ui.menu("Window", || {
                let v = &mut self.visibility;
                ui.menu_item_toggle_no_shortcut("Hierarchy", &mut v.hierarchy, true);
                ui.menu_item_toggle_no_shortcut("Scene", &mut v.scene, true);
                ui.menu_item_toggle_no_shortcut("Game", &mut v.game, true);
                ui.menu_item_toggle_no_shortcut("Inspector", &mut v.inspector, true);
                ui.menu_item_toggle_no_shortcut("Project", &mut v.project, true);
                ui.menu_item_toggle_no_shortcut("Console", &mut v.console, true);
                ui.menu_item_toggle_no_shortcut("Profiler", &mut v.profiler, true);
                ui.separator();
                if ui.menu_item("Reset Layout") {
                    self.reset_layout();
                }
                ui.separator();
                ui.menu("Theme", || {
                    if ui.menu_item("Dark") {
                        requests.theme = Some(Theme::Dark);
                    }
                    if ui.menu_item("Light") {
                        requests.theme = Some(Theme::Light);
                    }
                });
            });
            ui.menu("Help", || {
                if ui.menu_item("About Rustiny") {
                    self.show_about = true;
                }
            });
        });
    }

    pub(super) fn about(&mut self, ui: &Ui) {
        if !self.show_about {
            return;
        }
        let mut open = true;
        ui.window("About Rustiny").opened(&mut open).build(|| {
            ui.text(format!("Rustiny {}", rustiny_core::ENGINE_VERSION));
            ui.text_disabled("High-performance, Unity-compatible game engine written in Rust.");
            ui.separator();
            ui.text("Window, Vulkan, Dear ImGui docking; read-only Unity asset index.");
        });
        self.show_about = open;
    }
}
