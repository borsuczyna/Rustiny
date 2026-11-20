//! State that exists only while the window and GPU are alive: creation and per-frame drawing.

use super::scene_gpu::SceneGpu;
use super::startup::{InstanceJob, PhaseTimer, wait_for_instance};
use super::{CLEAR_COLOR, RunError, SETTLE_DURATION};
use crate::gui::ImGuiLayer;
use crate::meshes::MeshPlan;
use crate::ui::{EditorUi, GpuSummary, SceneState, UiContext, theme};
use rustiny_core::config::{EditorConfig, Theme};
use rustiny_core::logging::LogBuffer;
use rustiny_profiler::FrameProfiler;
use rustiny_vulkan::{Device, DeviceConfig, PresentTarget, Renderer, Surface, ValidationStats};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use winit::dpi::LogicalSize;
use winit::event_loop::ActiveEventLoop;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::Window;

/// What the application must do after a frame.
#[derive(Debug, Clone)]
pub(super) struct FrameOutcome {
    pub quit: bool,
    pub theme: Option<Theme>,
    /// Scene the user asked to open, relative to the project root.
    pub open_scene: Option<String>,
}

/// Everything that only exists while the window and GPU are alive.
///
/// Field order is drop order: UI backends first (they hold GPU objects), then the scene renderer,
/// then the Vulkan renderer (swapchain, surface, device), and the window last.
pub(super) struct Running {
    pub imgui: ImGuiLayer,
    /// The GPU Scene view; absent when it could not be created (the panel then draws markers).
    pub scene_gpu: Option<SceneGpu>,
    pub gfx: Renderer,
    pub window: Arc<Window>,
    pub editor_ui: EditorUi,
    pub profiler: FrameProfiler,
    pub gpu: GpuSummary,
    pub continuous: bool,
    pub redraw_pending: bool,
    pub active_until: Instant,
    pub last_draw: Instant,
    pub frames: u64,
    pub resize_tested: bool,
    pub screenshot_requested: bool,
    pub screenshot_saved: bool,
}

/// Inputs for [`Running::create`].
pub(super) struct StartParams<'a> {
    pub config: &'a EditorConfig,
    pub layout_path: &'a Path,
    pub continuous: bool,
    pub process_start: Instant,
    pub instance_job: InstanceJob,
}

/// A freshly created [`Running`] and facts the application shell keeps.
pub(super) struct Started {
    pub running: Running,
    pub stats: Arc<ValidationStats>,
    pub gpu_name: String,
}

impl Running {
    /// Create the window, the Vulkan objects and the UI backend.
    pub fn create(event_loop: &ActiveEventLoop, p: StartParams<'_>) -> Result<Started, RunError> {
        let mut phases = PhaseTimer::new(p.process_start);
        let w = &p.config.window;
        let attributes = Window::default_attributes()
            .with_title("Rustiny")
            .with_inner_size(LogicalSize::new(f64::from(w.width), f64::from(w.height)))
            .with_min_inner_size(LogicalSize::new(640.0, 360.0))
            .with_maximized(w.maximized);
        let window = Arc::new(
            event_loop
                .create_window(attributes)
                .map_err(|e| RunError::Unsupported(format!("cannot create window: {e}")))?,
        );
        phases.mark("window created");
        let display = window
            .display_handle()
            .map_err(|e| RunError::Failed(format!("display handle: {e}")))?
            .as_raw();
        let window_handle = window
            .window_handle()
            .map_err(|e| RunError::Failed(format!("window handle: {e}")))?
            .as_raw();

        let render = &p.config.render;
        let (instance, worker_time) = wait_for_instance(p.instance_job)?;
        let stats = instance.validation_stats().clone();
        phases.mark("wait for instance");
        tracing::info!(
            "vulkan instance created on worker in {:.0}ms",
            worker_time.as_secs_f64() * 1000.0
        );

        // SAFETY: the handles come from `window`, which is kept alive (and stays attached to the
        // surface's lifetime) by the `Running` struct that also owns the `Renderer`/`Surface`.
        let surface = unsafe { Surface::new(&instance, display, window_handle) }?;
        phases.mark("surface");
        let device = Device::new(
            &instance,
            Some(PresentTarget {
                loader: surface.loader(),
                surface: surface.raw(),
            }),
            &DeviceConfig {
                gpu_preference: render.gpu_preference.clone(),
            },
        )?;

        phases.mark("device");
        let size = window.inner_size();
        let gfx = Renderer::new(
            &device,
            surface,
            (size.width, size.height),
            render.present_mode,
        )?;
        drop(device); // `gfx` keeps the device alive.

        phases.mark("swapchain+frames");
        let mut imgui = ImGuiLayer::new(&window, &gfx, p.layout_path)?;
        theme::apply(&mut imgui.context, p.config.ui.theme);

        phases.mark("imgui");
        let scene_gpu = match SceneGpu::new(gfx.device()) {
            Ok(gpu) => Some(gpu),
            Err(e) => {
                tracing::warn!("GPU scene view unavailable, drawing markers instead: {e}");
                None
            }
        };
        phases.mark("scene renderer");
        let info = gfx.device().info().clone();
        let api = format!(
            "Vulkan {}.{}.{}, driver 0x{:08x}",
            rustiny_vulkan::ash::vk::api_version_major(info.api_version),
            rustiny_vulkan::ash::vk::api_version_minor(info.api_version),
            rustiny_vulkan::ash::vk::api_version_patch(info.api_version),
            info.driver_version
        );
        let gpu_name = info.name.clone();
        let gpu = GpuSummary {
            name: info.name,
            api,
            validation_active: instance.validation_enabled(),
            ..GpuSummary::default()
        };

        phases.log();
        let now = Instant::now();
        window.request_redraw();
        Ok(Started {
            running: Self {
                imgui,
                scene_gpu,
                gfx,
                window,
                editor_ui: EditorUi::new(),
                profiler: FrameProfiler::new(600),
                gpu,
                continuous: p.continuous,
                redraw_pending: true,
                active_until: now + SETTLE_DURATION,
                last_draw: now,
                frames: 0,
                resize_tested: false,
                screenshot_requested: false,
                screenshot_saved: false,
            },
            stats,
            gpu_name,
        })
    }

    /// Show a scene (or its loading/failed state) in the UI and, when there is one, on the GPU.
    pub fn show_scene(&mut self, state: SceneState, plan: &MeshPlan) {
        let lighting = match &state {
            SceneState::Ready { scene, .. } => crate::shaders::scene_light_values(scene),
            _ => None,
        };
        self.editor_ui.set_scene(state);
        if let Some(gpu) = self.scene_gpu.as_mut() {
            gpu.set_lighting(lighting);
            gpu.set_plan(plan);
        }
        self.redraw_pending = true;
    }

    /// The post-processing the Scene view applies (the scene's volumes).
    pub fn set_post_process(&mut self, post: rustiny_render::PostProcess) {
        if let Some(gpu) = self.scene_gpu.as_mut() {
            gpu.set_post_process(post);
        }
    }

    /// Scene passes rendered so far.
    pub fn scene_gpu_passes(&self) -> u64 {
        self.scene_gpu.as_ref().map_or(0, SceneGpu::passes)
    }

    /// Unregister the Scene view's texture from the UI renderer (before it shuts down).
    pub fn release_scene_view(&mut self) {
        if let Some(gpu) = self.scene_gpu.as_mut() {
            gpu.release(&mut self.imgui.renderer);
        }
    }

    /// Input arrived: redraw now and keep redrawing through the settle period.
    pub fn mark_active(&mut self) {
        self.redraw_pending = true;
        self.active_until = Instant::now() + SETTLE_DURATION;
    }

    /// Build and render one frame.
    pub fn draw_frame(
        &mut self,
        log: &LogBuffer,
        stats: Option<&Arc<ValidationStats>>,
    ) -> Result<FrameOutcome, RunError> {
        self.profiler.begin_frame();
        let (requests, presented) = {
            let ImGuiLayer {
                renderer,
                platform,
                context,
                ..
            } = &mut self.imgui;

            platform
                .prepare_frame(context, &self.window)
                .map_err(|e| RunError::Failed(format!("platform.prepare_frame: {e}")))?;

            // Refresh GPU facts that can change at runtime.
            let extent = self.gfx.extent();
            self.gpu.extent = (extent.width, extent.height);
            self.gpu.image_count = self.gfx.image_count();
            self.gpu.present_mode = format!("{:?}", self.gfx.present_mode());
            self.gpu.swapchain_generation = self.gfx.swapchain_generation();
            self.gpu.scene_gpu_ms = self.scene_gpu.as_ref().and_then(SceneGpu::gpu_time_ms);

            if let Some(gpu) = self.scene_gpu.as_mut() {
                match gpu.prepare(renderer) {
                    Ok(more) => self.redraw_pending |= more,
                    Err(e) => {
                        tracing::error!("GPU scene view disabled: {e}");
                        gpu.release(renderer);
                        self.scene_gpu = None;
                    }
                }
            }

            let ui_start = Instant::now();
            let ui = context.frame();
            let cx = UiContext {
                log,
                profiler: &self.profiler,
                gpu: &self.gpu,
                validation_errors: stats.map_or(0, |s| s.errors()),
                validation_warnings: stats.map_or(0, |s| s.warnings()),
                continuous_redraw: self.continuous,
                scene_texture: self.scene_gpu.as_ref().and_then(SceneGpu::texture),
            };
            let requests = self.editor_ui.draw(ui, &cx);
            if let Some(gpu) = self.scene_gpu.as_mut() {
                gpu.set_request(requests.scene_view);
            }
            platform
                .prepare_render(ui, &self.window)
                .map_err(|e| RunError::Failed(format!("platform.prepare_render: {e}")))?;
            let consumer = renderer
                .renderer_consumer()
                .map_err(|e| RunError::Failed(format!("renderer consumer: {e}")))?;
            let pending = context.render(consumer);
            self.profiler.record_scope("ui_build", ui_start.elapsed());

            // Texture uploads happen here, before the swapchain image is acquired.
            let (reconciled, retirement) = renderer
                .prepare_frame(pending)
                .map_err(|e| RunError::Failed(format!("renderer.prepare_frame: {e}")))?;

            let mut presented = false;
            let record_start = Instant::now();
            if let Some(frame) = self.gfx.begin_frame()? {
                if let Some(gpu) = self.scene_gpu.as_mut() {
                    let scene_start = Instant::now();
                    if gpu.record(frame.command_buffer) {
                        self.profiler
                            .record_scope("scene_pass", scene_start.elapsed());
                    }
                }
                self.gfx.begin_rendering(&frame, CLEAR_COLOR);
                // SAFETY: `frame.command_buffer` is recording inside a dynamic-rendering scope whose
                // colour format matches the renderer configuration; `reconciled` comes from this
                // renderer's `prepare_frame` for the current frame, and the frame-in-flight fence
                // wait in `begin_frame` guarantees the renderer's mesh slot is idle.
                unsafe { renderer.cmd_draw(frame.command_buffer, reconciled) }
                    .map_err(|e| RunError::Failed(format!("renderer.cmd_draw: {e}")))?;
                self.gfx.end_rendering(&frame);
                self.profiler.record_scope("record", record_start.elapsed());
                let submit_start = Instant::now();
                self.gfx.end_frame(frame)?;
                self.profiler
                    .record_scope("submit+present", submit_start.elapsed());
                presented = true;
            }
            if let Some(batch) = retirement {
                renderer
                    .wait_for_texture_retirements(batch)
                    .map_err(|e| RunError::Failed(format!("texture retirement: {e}")))?;
            }

            (requests, presented)
        };

        self.profiler.end_frame();
        if presented {
            self.frames += 1;
        }
        if requests.toggle_continuous {
            self.continuous = !self.continuous;
            self.profiler.reset();
        }
        if let Some(t) = requests.theme {
            theme::apply(&mut self.imgui.context, t);
        }
        self.redraw_pending |= requests.theme.is_some() || requests.toggle_continuous;
        Ok(FrameOutcome {
            quit: requests.quit,
            theme: requests.theme,
            open_scene: requests.open_scene,
        })
    }
}
