//! Start-up work that runs off the main thread, and the timing that keeps it honest.
//!
//! Loading the Vulkan loader and creating the instance takes several hundred milliseconds (layer
//! manifests, ICD discovery) and does not need the window, so it runs on a worker thread while the
//! window is created.

use super::RunError;
use rustiny_vulkan::{Instance, InstanceConfig, VulkanError};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use winit::raw_window_handle::RawDisplayHandle;

/// Result of the background instance creation: the instance and how long it took on the worker.
type InstanceResult = Result<(Arc<Instance>, Duration), VulkanError>;

/// A Vulkan instance being created on a worker thread.
pub(super) type InstanceJob = JoinHandle<InstanceResult>;

/// A display handle that may be moved to the worker thread.
struct SendDisplay(RawDisplayHandle);

// SAFETY: the worker only inspects which kind of display this is (to choose the surface
// extensions); it never dereferences the pointers inside the handle.
unsafe impl Send for SendDisplay {}

/// Start creating the Vulkan instance on its own thread.
pub(super) fn spawn_instance_job(
    validation: bool,
    sync_validation: bool,
    display: RawDisplayHandle,
) -> Result<InstanceJob, RunError> {
    let display = SendDisplay(display);
    std::thread::Builder::new()
        .name("vulkan-instance".to_owned())
        .spawn(move || {
            // Using the whole wrapper (not just its field) makes the closure capture `SendDisplay`.
            let display = display;
            let config = InstanceConfig {
                app_name: "Rustiny Editor".to_owned(),
                validation,
                sync_validation,
                display: Some(display.0),
            };
            let start = Instant::now();
            Instance::new(&config).map(|i| (i, start.elapsed()))
        })
        .map_err(|e| RunError::Failed(format!("cannot start the Vulkan init thread: {e}")))
}

/// Wait for the instance job; returns the instance and the time it took on the worker.
pub(super) fn wait_for_instance(job: InstanceJob) -> Result<(Arc<Instance>, Duration), RunError> {
    let result = job
        .join()
        .map_err(|_| RunError::Failed("Vulkan init thread panicked".to_owned()))?;
    Ok(result?)
}

/// Logs how long each start-up phase took (and the time since process start), so start-up work
/// is measured rather than guessed.
pub(super) struct PhaseTimer {
    process_start: Instant,
    last: Instant,
    phases: Vec<(&'static str, Duration)>,
}

impl PhaseTimer {
    pub fn new(process_start: Instant) -> Self {
        Self {
            process_start,
            last: Instant::now(),
            phases: Vec::new(),
        }
    }

    pub fn mark(&mut self, name: &'static str) {
        let now = Instant::now();
        self.phases.push((name, now - self.last));
        self.last = now;
    }

    pub fn log(&self) {
        let detail: Vec<String> = self
            .phases
            .iter()
            .map(|(n, d)| format!("{n}={:.0}ms", d.as_secs_f64() * 1000.0))
            .collect();
        tracing::info!(
            "startup phases: {} | since process start: {:.0}ms",
            detail.join(" "),
            self.process_start.elapsed().as_secs_f64() * 1000.0
        );
    }
}
