//! Rustiny visual identity (own colours, not a Unity imitation).

use dear_imgui_rs::{Context, StyleColor};
use rustiny_core::config::Theme;

/// Rust-orange accent used for selection and focus.
const ACCENT: [f32; 4] = [0.85, 0.40, 0.17, 1.0];

fn with_alpha(c: [f32; 4], a: f32) -> [f32; 4] {
    [c[0], c[1], c[2], a]
}

/// Apply a theme to the style of `context`.
pub fn apply(context: &mut Context, theme: Theme) {
    // SAFETY: these functions reset the colour table of the *current* Dear ImGui context to the
    // built-in palette. `context` is the only context in the process and was made current when it
    // was created; passing null selects the current style. They are called between frames.
    unsafe {
        match theme {
            Theme::Dark => dear_imgui_rs::sys::igStyleColorsDark(std::ptr::null_mut()),
            Theme::Light => dear_imgui_rs::sys::igStyleColorsLight(std::ptr::null_mut()),
        }
    }
    let style = context.style_mut();
    style.set_window_rounding(3.0);
    style.set_frame_rounding(3.0);
    style.set_tab_rounding(3.0);
    style.set_scrollbar_rounding(3.0);
    style.set_frame_border_size(0.0);

    let accent = ACCENT;
    let tinted = [
        (StyleColor::CheckMark, accent),
        (StyleColor::SliderGrab, accent),
        (StyleColor::SliderGrabActive, with_alpha(accent, 1.0)),
        (StyleColor::Header, with_alpha(accent, 0.35)),
        (StyleColor::HeaderHovered, with_alpha(accent, 0.55)),
        (StyleColor::HeaderActive, with_alpha(accent, 0.75)),
        (StyleColor::TabSelected, with_alpha(accent, 0.65)),
        (StyleColor::TabSelectedOverline, accent),
        (StyleColor::TabDimmedSelected, with_alpha(accent, 0.40)),
        (
            StyleColor::TabDimmedSelectedOverline,
            with_alpha(accent, 0.7),
        ),
        (StyleColor::TabHovered, with_alpha(accent, 0.80)),
        (StyleColor::DockingPreview, with_alpha(accent, 0.55)),
        (StyleColor::SeparatorHovered, with_alpha(accent, 0.78)),
        (StyleColor::SeparatorActive, accent),
        (StyleColor::ResizeGripHovered, with_alpha(accent, 0.67)),
        (StyleColor::ResizeGripActive, with_alpha(accent, 0.95)),
        (StyleColor::TextSelectedBg, with_alpha(accent, 0.35)),
        (StyleColor::NavCursor, accent),
        (StyleColor::PlotLines, accent),
        (StyleColor::PlotHistogram, accent),
    ];
    for (slot, colour) in tinted {
        style.set_color(slot, colour);
    }

    // Neutral chrome: replace the stock blue title bars / tabs so only the accent stands out.
    let neutral: [(StyleColor, [f32; 4]); 5] = match theme {
        Theme::Dark => [
            (StyleColor::TitleBgActive, [0.20, 0.15, 0.12, 1.0]),
            (StyleColor::Tab, [0.17, 0.17, 0.18, 1.0]),
            (StyleColor::TabDimmed, [0.13, 0.13, 0.14, 1.0]),
            (StyleColor::TabDimmedSelected, [0.25, 0.19, 0.15, 1.0]),
            (StyleColor::DockingEmptyBg, [0.10, 0.10, 0.11, 1.0]),
        ],
        Theme::Light => [
            (StyleColor::TitleBgActive, [0.93, 0.85, 0.80, 1.0]),
            (StyleColor::Tab, [0.84, 0.84, 0.85, 1.0]),
            (StyleColor::TabDimmed, [0.89, 0.89, 0.90, 1.0]),
            (StyleColor::TabDimmedSelected, [0.95, 0.88, 0.83, 1.0]),
            (StyleColor::DockingEmptyBg, [0.80, 0.80, 0.81, 1.0]),
        ],
    };
    for (slot, colour) in neutral {
        style.set_color(slot, colour);
    }
}
