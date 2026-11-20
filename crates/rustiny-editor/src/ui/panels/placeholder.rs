//! Placeholder for panels whose implementation lands in a later milestone.

use dear_imgui_rs::{Ui, WindowKey};

/// Panel whose real implementation lands in a later milestone.
pub fn draw(ui: &Ui, key: &WindowKey, open: &mut bool, what: &str, milestone: &str) {
    if !*open {
        return;
    }
    ui.window(key).opened(open).build(|| {
        ui.text_disabled(what);
        ui.text_disabled(format!("Not implemented yet - planned for {milestone}."));
    });
}
