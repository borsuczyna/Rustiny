//! Dear ImGui context, winit platform backend and Vulkan renderer backend, owned together so they
//! are torn down in the one order that is correct.

use dear_imgui_ash::{AshRenderer, AshRendererConfig, DynamicRendering, Options};
use dear_imgui_rs::{ConfigFlags, Context};
use dear_imgui_winit::{HiDpiMode, WinitPlatform};
use rustiny_vulkan::{FRAMES_IN_FLIGHT, Renderer};
use std::path::Path;
use std::sync::Arc;
use winit::window::Window;

/// Failures while bringing up the UI layer.
#[derive(Debug, thiserror::Error)]
pub enum GuiError {
    /// Dear ImGui context setup failed.
    #[error("Dear ImGui: {0}")]
    ImGui(String),
    /// Winit platform backend failed.
    #[error("ImGui platform backend: {0}")]
    Platform(String),
    /// Vulkan renderer backend failed.
    #[error("ImGui Vulkan backend: {0}")]
    Renderer(String),
}

/// The UI stack. Field order matters: the renderer backend must drop before the context.
pub struct ImGuiLayer {
    /// Vulkan backend.
    pub renderer: AshRenderer,
    /// Winit backend.
    pub platform: WinitPlatform,
    /// Dear ImGui context.
    pub context: Context,
    shut_down: bool,
}

impl std::fmt::Debug for ImGuiLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImGuiLayer")
            .field("shut_down", &self.shut_down)
            .finish_non_exhaustive()
    }
}

impl ImGuiLayer {
    /// Create the context and both backends.
    pub fn new(window: &Arc<Window>, gfx: &Renderer, ini_path: &Path) -> Result<Self, GuiError> {
        let mut context = Context::create();
        context
            .set_ini_filename(Some(ini_path.to_path_buf()))
            .map_err(|e| GuiError::ImGui(e.to_string()))?;
        let flags = context.io().config_flags()
            | ConfigFlags::DOCKING_ENABLE
            | ConfigFlags::NAV_ENABLE_KEYBOARD;
        context.io_mut().set_config_flags(flags);

        let mut platform =
            WinitPlatform::new(&mut context).map_err(|e| GuiError::Platform(e.to_string()))?;
        platform
            .attach_window(window.clone(), HiDpiMode::Default, &mut context)
            .map_err(|e| GuiError::Platform(e.to_string()))?;

        let device = gfx.device();
        let config = AshRendererConfig::with_dynamic_rendering(
            device.raw().clone(),
            device.graphics_queue(),
            gfx.upload_command_pool(),
            DynamicRendering {
                color_attachment_format: gfx.color_format(),
                depth_attachment_format: None,
            },
        )
        .with_options(Options {
            in_flight_frames: FRAMES_IN_FLIGHT,
            framebuffer_srgb: false,
            ..Options::default()
        });
        // SAFETY: the device, queue and command pool all come from `gfx`, which outlives this
        // layer (see `shutdown`); the queue and pool are only used from the thread that drives
        // the frame loop; the colour format matches the swapchain the draws are recorded into.
        let renderer = unsafe {
            AshRenderer::with_gpu_allocator(device.allocator().clone(), config, &mut context)
        }
        .map_err(|e| GuiError::Renderer(e.to_string()))?;

        Ok(Self {
            renderer,
            platform,
            context,
            shut_down: false,
        })
    }

    /// Release GPU resources and detach the backends. Idempotent; must run before the Vulkan
    /// [`Renderer`] / device are dropped.
    pub fn shutdown(&mut self) {
        if self.shut_down {
            return;
        }
        self.shut_down = true;
        if let Err(e) = self.renderer.shutdown(&mut self.context) {
            tracing::error!("ImGui renderer shutdown: {e}");
        }
        if let Err(e) = self.platform.shutdown(&mut self.context) {
            tracing::error!("ImGui platform shutdown: {e}");
        }
    }
}

impl Drop for ImGuiLayer {
    fn drop(&mut self) {
        self.shutdown();
    }
}
