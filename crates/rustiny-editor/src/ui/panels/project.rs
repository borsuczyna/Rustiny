//! Project panel: read-only tree of the opened Unity project's `Assets/` folder.

use super::style::{COLOR_ERROR, COLOR_WARN};
use crate::ui::{ProjectState, UiRequests};
use dear_imgui_rs::{MouseButton, Ui, WindowKey};
use rustiny_assets::{AssetEntry, Issue};

/// Selection and view state of the Project panel.
#[derive(Debug, Default)]
pub struct State {
    /// Project-relative path of the selected asset.
    pub selected: Option<String>,
    /// Show the list of scan problems.
    pub show_issues: bool,
    /// Frames left before the panel's tab is brought to the front (0 = nothing pending). Delayed
    /// because a request in the very first frame is overridden by the default dock layout.
    pub focus_in_frames: u8,
}

/// The Project panel: a read-only tree of the opened project's `Assets/` folder.
///
/// Only expanded folders are walked, so cost follows what is visible rather than project size
/// (an expanded folder with tens of thousands of files would still be drawn in full; row
/// virtualisation is future work).
pub fn draw(
    ui: &Ui,
    key: &WindowKey,
    open: &mut bool,
    state: &ProjectState,
    panel: &mut State,
    requests: &mut UiRequests,
) {
    if !*open {
        return;
    }
    let focus = panel.focus_in_frames == 1;
    panel.focus_in_frames = panel.focus_in_frames.saturating_sub(1);
    ui.window(key)
        .opened(open)
        .focused(focus)
        .build(|| match state {
            ProjectState::None => {
                ui.text_disabled("No project open.");
                ui.text_disabled("Start Rustiny with  --project <path to a Unity project>.");
            }
            ProjectState::Scanning(path) => ui.text_disabled(format!("Scanning {path} ...")),
            ProjectState::Failed(why) => ui.text_colored(COLOR_ERROR, why),
            ProjectState::Ready {
                project,
                db,
                watching,
            } => {
                let stats = db.stats();
                ui.text(project.name());
                ui.same_line();
                ui.text_disabled(format!(
                    "Unity {} | {} files, {} folders | scanned in {:.0} ms ({} of {} .meta reused){}",
                    project
                        .editor_version
                        .as_deref()
                        .unwrap_or("version unknown"),
                    stats.files,
                    stats.folders,
                    stats.duration.as_secs_f64() * 1000.0,
                    stats.metas_reused,
                    stats.metas_reused + stats.metas_parsed,
                    if *watching {
                        " | watching for changes"
                    } else {
                        ""
                    }
                ));
                let issues = db.issues().len();
                if issues > 0 {
                    ui.text_colored(COLOR_WARN, format!("{issues} problems found"));
                    ui.same_line();
                    ui.checkbox("show", &mut panel.show_issues);
                }
                ui.separator();
                if panel.show_issues {
                    for issue in db.issues() {
                        ui.text_colored(issue_color(issue), issue.to_string());
                    }
                    ui.separator();
                }
                draw_assets(ui, db.entries(), 0, db.entries().len(), panel, requests);
            }
        });
}

fn issue_color(issue: &Issue) -> [f32; 4] {
    match issue {
        Issue::InvalidMeta { .. } | Issue::DuplicateGuid { .. } | Issue::Unreadable { .. } => {
            COLOR_ERROR
        }
        _ => COLOR_WARN,
    }
}

/// Draw `entries[start..end]` (one directory level and, recursively, its open sub-folders).
fn draw_assets(
    ui: &Ui,
    entries: &[AssetEntry],
    start: usize,
    end: usize,
    panel: &mut State,
    requests: &mut UiRequests,
) {
    let mut i = start;
    while i < end {
        let entry = &entries[i];
        let selected = panel.selected.as_deref() == Some(entry.path.as_str());
        if entry.is_folder {
            let node = ui
                .tree_node_config(entry.path.as_str())
                .label(entry.name())
                .open_on_arrow(true)
                .span_full_width(true)
                .selected(selected)
                .push();
            if ui.is_item_clicked() {
                panel.selected = Some(entry.path.clone());
            }
            if let Some(token) = node {
                draw_assets(ui, entries, i + 1, entry.subtree_end, panel, requests);
                drop(token);
            }
            i = entry.subtree_end;
        } else {
            let label = format!("{}##{}", entry.name(), entry.path);
            if ui.selectable_config(label).selected(selected).build() {
                panel.selected = Some(entry.path.clone());
            }
            if ui.is_item_hovered() {
                ui.tooltip_text(asset_tooltip(entry));
                if is_scene(entry) && ui.is_mouse_double_clicked(MouseButton::Left) {
                    requests.open_scene = Some(entry.path.clone());
                }
            }
            i += 1;
        }
    }
}

fn is_scene(entry: &AssetEntry) -> bool {
    entry.name().ends_with(".unity")
}

fn asset_tooltip(entry: &AssetEntry) -> String {
    let guid = entry
        .guid
        .map_or_else(|| "none (no valid .meta)".to_owned(), |g| g.to_string());
    let hint = if is_scene(entry) {
        "\nDouble-click to open the scene (read-only)."
    } else {
        ""
    };
    format!(
        "{}\nGUID: {guid}\nImporter: {}\nSize: {} bytes{hint}",
        entry.path,
        entry.importer.as_deref().unwrap_or("-"),
        entry.size
    )
}
