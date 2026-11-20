//! Smoke-mode bookkeeping: mid-run resize, screenshot capture and the exit condition.

use super::{App, RunError, SMOKE_TIMEOUT};

/// Scene passes a smoke run waits for (when a scene is open and the GPU view exists).
const GPU_PASSES_WANTED: u64 = 8;
use winit::dpi::PhysicalSize;
use winit::event_loop::ActiveEventLoop;

impl App {
    /// Called after each drawn frame; does nothing outside smoke mode.
    pub(super) fn after_frame(&mut self, event_loop: &ActiveEventLoop) {
        let Some(target) = self.opts.smoke_frames else {
            return;
        };
        let Some(run) = self.running.as_mut() else {
            return;
        };

        // Exercise the swapchain recreation path half-way through.
        if !run.resize_tested && run.frames >= u64::from(target / 2) {
            run.resize_tested = true;
            let current = run.window.inner_size();
            let smaller = PhysicalSize::new(
                (current.width * 3 / 4).max(640),
                (current.height * 3 / 4).max(360),
            );
            tracing::info!(
                "smoke: resizing window to {}x{}",
                smaller.width,
                smaller.height
            );
            let _ = run.window.request_inner_size(smaller);
        }
        let want_shot = self.opts.screenshot.is_some();
        // A scene that is still loading is waited for, so the report and the screenshot show it.
        // Once it is loaded, the GPU view needs a few frames to create its target and render.
        // Only the smoke timeout bounds this wait: how many frames a load takes depends on the
        // frame rate, which says nothing about the load.
        let meshes_pending = self.mesh_load.as_ref().is_some_and(|l| !l.is_finished());
        let scene_pending = self.scene_rx.is_some()
            || self.pending_scene.is_some()
            || meshes_pending
            || (self.report.scene.is_some()
                && run.scene_gpu.is_some()
                && run.scene_gpu_passes() < GPU_PASSES_WANTED
                && run.frames < u64::from(target) + 300);
        // Request the capture one frame before the end so it is available when we stop.
        if want_shot
            && !scene_pending
            && !run.screenshot_requested
            && run.frames + 1 >= u64::from(target)
        {
            run.screenshot_requested = run.gfx.request_screenshot();
            if !run.screenshot_requested {
                tracing::warn!(
                    "swapchain does not support readback; no screenshot will be written"
                );
            }
        }
        if let Some(shot) = run.gfx.take_screenshot() {
            if let Some(path) = &self.opts.screenshot {
                match crate::screenshot::save_png(path, &shot) {
                    Ok(()) => tracing::info!("screenshot written to {}", path.display()),
                    Err(e) => {
                        let error = RunError::Failed(format!("screenshot: {e}"));
                        self.fail(event_loop, error);
                        return;
                    }
                }
            }
            run.screenshot_saved = true;
        }
        let shot_pending = want_shot && run.screenshot_requested && !run.screenshot_saved;
        if run.frames >= u64::from(target) && !shot_pending && !scene_pending {
            event_loop.exit();
        } else if self.started.elapsed() > SMOKE_TIMEOUT {
            self.fail(
                event_loop,
                RunError::Failed("smoke test timed out".to_owned()),
            );
        }
    }
}
