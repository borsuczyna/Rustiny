//! The winit event handler: window events, redraw policy and shutdown.

use super::report::MeshReport;
use super::running::{Running, StartParams};
use super::scene_gpu::SceneGpu;
use super::{App, RunError, SETTLE_INTERVAL, UserEvent};
use std::sync::atomic::Ordering;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{StartCause, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow};
use winit::window::WindowId;

impl App {
    pub(super) fn fail(&mut self, event_loop: &ActiveEventLoop, error: RunError) {
        tracing::error!("fatal: {error}");
        self.error.get_or_insert(error);
        event_loop.exit();
    }

    fn create_running(&mut self, event_loop: &ActiveEventLoop) -> Result<Running, RunError> {
        let instance_job = self
            .instance_job
            .take()
            .ok_or_else(|| RunError::Failed("Vulkan instance job missing".to_owned()))?;
        let continuous = self.config.ui.continuous_redraw
            || self.opts.continuous
            || self.opts.smoke_frames.is_some();
        let started = Running::create(
            event_loop,
            StartParams {
                config: &self.config,
                layout_path: &self.layout_path,
                continuous,
                process_start: self.process_start,
                instance_job,
            },
        )?;
        self.stats = Some(started.stats);
        self.report.gpu = started.gpu_name;
        Ok(started.running)
    }
}

impl ApplicationHandler<UserEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.running.is_some() {
            return;
        }
        match self.create_running(event_loop) {
            Ok(running) => {
                tracing::info!("editor ready");
                self.running = Some(running);
                if let Some(run) = self.running.as_mut() {
                    run.editor_ui.set_initial_view(self.opts.view);
                }
                if let Some(path) = self.opts.project.clone() {
                    self.open_project(path);
                }
                if let Some(path) = self.opts.scene.clone() {
                    self.open_scene(path);
                }
            }
            Err(e) => self.fail(event_loop, e),
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::Wake => {
                self.wake_pending.store(false, Ordering::Release);
                self.poll_project();
                self.poll_scene();
                if let Some(run) = self.running.as_mut() {
                    run.redraw_pending = true;
                }
            }
            UserEvent::ProjectChanged => self.on_project_changed(),
        }
    }

    fn new_events(&mut self, _event_loop: &ActiveEventLoop, cause: StartCause) {
        // The settle timer fired: draw even though no OS event arrived.
        if matches!(cause, StartCause::ResumeTimeReached { .. })
            && let Some(run) = self.running.as_mut()
        {
            run.redraw_pending = true;
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        let Some(run) = self.running.as_mut() else {
            return;
        };
        if id != run.window.id() {
            return;
        }

        match run
            .imgui
            .platform
            .handle_window_event(&mut run.imgui.context, id, &event)
        {
            Ok(outcome) => {
                if outcome.needs_frame {
                    run.mark_active();
                }
            }
            Err(e) => {
                let error = RunError::Failed(format!("platform event: {e}"));
                self.fail(event_loop, error);
                return;
            }
        }

        match event {
            WindowEvent::CloseRequested => {
                tracing::info!("close requested");
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                run.gfx.resize(size.width, size.height);
                run.mark_active();
            }
            WindowEvent::ScaleFactorChanged { .. }
            | WindowEvent::Focused(_)
            | WindowEvent::Occluded(false)
            | WindowEvent::ThemeChanged(_) => run.mark_active(),
            WindowEvent::RedrawRequested => {
                // Models that finished importing go to the GPU first, so this frame shows them.
                if let (Some(load), Some(gpu)) = (self.mesh_load.as_mut(), run.scene_gpu.as_mut())
                    && load.pump(gpu)
                {
                    run.redraw_pending = true;
                }
                let stats = self.stats.clone();
                match run.draw_frame(&self.log, stats.as_ref()) {
                    Ok(outcome) => {
                        run.last_draw = Instant::now();
                        if let Some(theme) = outcome.theme {
                            self.config.ui.theme = theme;
                            self.config_dirty = true;
                        }
                        if outcome.quit {
                            event_loop.exit();
                        }
                        if let Some(relative) = outcome.open_scene {
                            self.open_project_scene(&relative);
                        }
                    }
                    Err(e) => {
                        self.fail(event_loop, e);
                        return;
                    }
                }
                self.after_frame(event_loop);
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let Some(run) = self.running.as_mut() else {
            event_loop.set_control_flow(ControlFlow::Wait);
            return;
        };
        let size = run.window.inner_size();
        if size.width == 0 || size.height == 0 {
            // Minimised: nothing to draw, block until the window is restored.
            event_loop.set_control_flow(ControlFlow::Wait);
            return;
        }
        let now = Instant::now();
        if run.continuous || run.redraw_pending {
            run.redraw_pending = false;
            run.window.request_redraw();
            event_loop.set_control_flow(ControlFlow::Wait);
        } else if now < run.active_until {
            // Settle period: redraw at a modest rate so time-based UI behaviour completes.
            let next = run.last_draw + SETTLE_INTERVAL;
            if now >= next {
                run.window.request_redraw();
                event_loop.set_control_flow(ControlFlow::Wait);
            } else {
                event_loop.set_control_flow(ControlFlow::WaitUntil(next));
            }
        } else {
            event_loop.set_control_flow(ControlFlow::Wait);
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        // Stop watching before the GPU teardown starts.
        self.open_project = None;
        if let Some(mut run) = self.running.take() {
            self.report.frames = run.frames;
            self.report.swapchain_recreations = run.gfx.swapchain_generation();
            self.report.scene_gpu_passes = run.scene_gpu_passes();
            self.report.scene_gpu_ms = run.gpu.scene_gpu_ms;
            self.report.meshes = self.mesh_load.as_ref().map(|load| {
                let stats = run.scene_gpu.as_ref().map(SceneGpu::mesh_stats);
                MeshReport {
                    planned: load.total(),
                    uploaded: load.uploaded(),
                    failed: load.failed(),
                    triangles_drawn: stats.map_or(0, |s| s.triangles_drawn),
                    draw_calls: stats.map_or(0, |s| s.draw_calls),
                    material_slots: load.resolve_stats().slots,
                    materials_matched: load.resolve_stats().matched,
                    textures_requested: load.textures().requested,
                    textures_uploaded: load.textures().uploaded,
                    textures_failed: load.textures().failed,
                    texture_bytes: stats.map_or(0, |s| s.texture_bytes),
                }
            });
            run.release_scene_view();
            run.imgui.shutdown();
            // Dropping `run` now tears down the swapchain, surface and device in order.
        }
        if self.config_dirty
            && self.opts.smoke_frames.is_none()
            && let Err(e) = self.config.save(&self.config_path)
        {
            tracing::warn!("could not save configuration: {e}");
        }
    }
}
