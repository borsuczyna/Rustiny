//! Application shell: window, event loop, frame orchestration.
//!
//! Redraw policy (Rustiny's idle-performance requirement): the editor draws *on demand*. It
//! redraws when input arrives, the window changes, or a background thread wakes it (new log
//! record), and keeps a short 30 Hz "settle" period afterwards so hover delays, tooltips and
//! animations complete. After that it blocks in the OS event queue and uses no CPU. `continuous`
//! mode (config, `--continuous`, smoke tests) redraws every vsync instead, for profiling.
//!
//! Layout of this module:
//! * `mod.rs`          - [`run`], the [`App`] state and shared constants.
//! * `handler.rs`      - the winit `ApplicationHandler` (events, redraw policy, shutdown).
//! * `running.rs`      - window + GPU state: creation and per-frame drawing.
//! * `startup.rs`      - work moved off the main thread at start-up, and phase timing.
//! * `project_scan.rs` - opening a Unity project on a worker thread.
//! * `scene_load.rs`   - opening a scene file on a worker thread.
//! * `scene_gpu.rs`    - the GPU Scene view: scene renderer and its ImGui texture.
//! * `mesh_load.rs`    - taking imported models and uploading them within a per-frame budget.
//! * `smoke.rs`        - the `--smoke-frames` test harness behaviour.
//! * `events.rs`, `error.rs`, `report.rs` - small shared types.

mod error;
mod events;
mod handler;
mod mesh_load;
mod project_scan;
mod report;
mod running;
mod scene_gpu;
mod scene_load;
mod smoke;
mod startup;

pub use error::RunError;
pub use events::UserEvent;
pub use report::{MeshReport, RunReport};

use crate::cli::CliOptions;
use running::Running;
use rustiny_core::config::EditorConfig;
use rustiny_core::logging::LogBuffer;
use rustiny_core::paths::UserDirs;
use rustiny_vulkan::ValidationStats;
use startup::InstanceJob;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};
use winit::event_loop::{EventLoop, EventLoopProxy};
use winit::raw_window_handle::HasDisplayHandle;

/// How long after the last interesting event the editor keeps redrawing at [`SETTLE_INTERVAL`].
const SETTLE_DURATION: Duration = Duration::from_millis(1000);
const SETTLE_INTERVAL: Duration = Duration::from_millis(33);
/// Safety valve so a hung smoke test fails instead of blocking CI forever.
const SMOKE_TIMEOUT: Duration = Duration::from_secs(120);
const CLEAR_COLOR: [f32; 4] = [0.06, 0.06, 0.07, 1.0];

struct App {
    opts: CliOptions,
    config: EditorConfig,
    config_path: PathBuf,
    config_dirty: bool,
    /// Where compiled shaders and mirrored package includes are kept (never inside a project).
    shader_cache: PathBuf,
    log: Arc<LogBuffer>,
    layout_path: PathBuf,
    wake_pending: Arc<AtomicBool>,
    running: Option<Running>,
    /// Vulkan instance being created on a worker thread; taken when the window is created.
    instance_job: Option<InstanceJob>,
    proxy: EventLoopProxy<UserEvent>,
    /// Result of the background project scan, if one is running.
    project_rx: Option<Receiver<project_scan::ScanResult>>,
    scan_in_flight: bool,
    /// Changes arrived while a scan was running; scan again when it finishes.
    rescan_queued: bool,
    /// The scanned project and its file watcher, once the first scan succeeded.
    open_project: Option<project_scan::OpenProject>,
    /// Result of the background scene load, if one is running.
    scene_rx: Option<Receiver<scene_load::SceneResult>>,
    /// A scene asked for while the project it belongs to is still being scanned: its model GUIDs
    /// cannot be looked up before the scan finishes, so it is opened then.
    pending_scene: Option<PathBuf>,
    /// The open scene's models, while they are imported and uploaded (and afterwards for counts).
    mesh_load: Option<mesh_load::MeshLoad>,
    stats: Option<Arc<ValidationStats>>,
    error: Option<RunError>,
    started: Instant,
    process_start: Instant,
    report: RunReport,
}

/// Inputs to [`run`].
#[derive(Debug)]
pub struct RunInputs {
    /// Parsed command line.
    pub opts: CliOptions,
    /// Loaded configuration.
    pub config: EditorConfig,
    /// Resolved user directories.
    pub dirs: UserDirs,
    /// Log capture buffer for the Console panel.
    pub log: Arc<LogBuffer>,
    /// Moment `main` began, for start-up timing.
    pub process_start: Instant,
}

/// Run the editor until the window closes (or the smoke run finishes).
pub fn run(inputs: RunInputs) -> Result<RunReport, RunError> {
    let event_loop = EventLoop::<UserEvent>::with_user_event()
        .build()
        .map_err(|e| RunError::Unsupported(format!("cannot create event loop: {e}")))?;
    let wake_pending = Arc::new(AtomicBool::new(false));
    events::install_log_waker(&inputs.log, event_loop.create_proxy(), wake_pending.clone());

    // Start creating the Vulkan instance now so it overlaps with window creation.
    let display = event_loop
        .display_handle()
        .map_err(|e| RunError::Failed(format!("display handle: {e}")))?
        .as_raw();
    let instance_job = startup::spawn_instance_job(
        inputs.config.render.validation.enabled() && !inputs.opts.no_validation,
        inputs.config.render.sync_validation,
        display,
    )?;

    let mut app = App {
        instance_job: Some(instance_job),
        proxy: event_loop.create_proxy(),
        project_rx: None,
        scan_in_flight: false,
        rescan_queued: false,
        open_project: None,
        scene_rx: None,
        mesh_load: None,
        pending_scene: None,
        shader_cache: inputs.dirs.config.join("shader-cache"),
        config_path: inputs.dirs.config_file(),
        layout_path: inputs.dirs.layout_file(),
        opts: inputs.opts,
        config: inputs.config,
        config_dirty: false,
        log: inputs.log,
        wake_pending,
        running: None,
        stats: None,
        error: None,
        started: Instant::now(),
        process_start: inputs.process_start,
        report: RunReport::default(),
    };
    event_loop
        .run_app(&mut app)
        .map_err(|e| RunError::Failed(format!("event loop: {e}")))?;

    // Everything GPU-related has been dropped by now, so teardown validation is included.
    if let Some(stats) = &app.stats {
        app.report.validation_errors = stats.errors();
        app.report.validation_warnings = stats.warnings();
        app.report.environment_errors = stats.general_errors();
    }
    match app.error {
        Some(e) => Err(e),
        None => Ok(app.report),
    }
}
