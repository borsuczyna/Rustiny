//! Profiler panel: frame statistics, GPU facts and validation counters.

use super::style::{COLOR_ERROR, COLOR_INFO};
use crate::ui::{UiContext, UiRequests};
use dear_imgui_rs::{TreeNodeFlags, Ui, WindowKey};

/// Frame statistics, GPU facts and validation counters.
pub fn draw(
    ui: &Ui,
    key: &WindowKey,
    open: &mut bool,
    cx: &UiContext<'_>,
    plot: &mut Vec<f32>,
    requests: &mut UiRequests,
) {
    if !*open {
        return;
    }
    ui.window(key).opened(open).build(|| {
        let p = cx.profiler;
        let cpu = p.cpu_time().stats();
        ui.text(format!("Frames rendered: {}", p.frame_count()));
        let mut continuous = cx.continuous_redraw;
        if ui.checkbox("Continuous redraw (profiling)", &mut continuous) {
            requests.toggle_continuous = true;
        }
        if !cx.continuous_redraw {
            ui.text_disabled("On-demand mode: the editor only draws when something changes.");
        }
        ui.separator();

        if cpu.count > 0 {
            ui.text("CPU time per frame (ms)");
            p.cpu_time().copy_f32_into(plot);
            ui.plot_lines("##cpu", plot);
            ui.text(format!(
                "median {:.3}  p95 {:.3}  p99 {:.3}  max {:.3}  (n={})",
                cpu.median, cpu.p95, cpu.p99, cpu.max, cpu.count
            ));
        } else {
            ui.text_disabled("Waiting for frame samples...");
        }
        if cx.continuous_redraw {
            let iv = p.frame_interval().stats();
            if iv.count > 0 {
                ui.text(format!(
                    "Frame interval ms: median {:.3}  p95 {:.3}  p99 {:.3}  => {:.1} FPS",
                    iv.median,
                    iv.p95,
                    iv.p99,
                    if iv.median > 0.0 {
                        1000.0 / iv.median
                    } else {
                        0.0
                    }
                ));
            }
        }

        let scopes = p.scopes_sorted();
        if !scopes.is_empty() && ui.collapsing_header("Scopes (ms)", TreeNodeFlags::empty()) {
            for (name, window) in scopes {
                let s = window.stats();
                ui.text(format!(
                    "{name:<14} median {:.3}  p95 {:.3}  p99 {:.3}",
                    s.median, s.p95, s.p99
                ));
            }
        }

        ui.separator();
        let g = cx.gpu;
        ui.text(format!("GPU: {}", g.name));
        ui.text(format!("API: {}", g.api));
        ui.text(format!(
            "Swapchain: {}x{}, {} images, {} (recreated {}x)",
            g.extent.0, g.extent.1, g.image_count, g.present_mode, g.swapchain_generation
        ));
        match g.scene_gpu_ms {
            Some(ms) => ui.text(format!("Scene view pass: {ms:.2} ms on the GPU")),
            None => ui.text_disabled("Scene view pass: not measured"),
        }
        if g.validation_active {
            let colour = if cx.validation_errors > 0 {
                COLOR_ERROR
            } else {
                COLOR_INFO
            };
            ui.text_colored(
                colour,
                format!(
                    "Validation: {} errors, {} warnings",
                    cx.validation_errors, cx.validation_warnings
                ),
            );
        } else {
            ui.text_disabled("Validation layer: off");
        }
    });
}
