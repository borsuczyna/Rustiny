//! Events injected into the winit loop from other threads.

use rustiny_core::logging::LogBuffer;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use winit::event_loop::EventLoopProxy;

/// Events injected into the winit loop from other threads.
#[derive(Debug, Clone, Copy)]
pub enum UserEvent {
    /// A background thread produced something worth redrawing for (e.g. a log record).
    Wake,
    /// Files under the opened project's `Assets/` changed (already debounced).
    ProjectChanged,
}

/// Wake the event loop when a log record arrives, coalescing bursts into one event.
pub(super) fn install_log_waker(
    log: &LogBuffer,
    proxy: EventLoopProxy<UserEvent>,
    pending: Arc<AtomicBool>,
) {
    log.set_notifier(move || {
        if !pending.swap(true, Ordering::AcqRel) {
            // The loop may already be gone during shutdown; that is fine.
            let _ = proxy.send_event(UserEvent::Wake);
        }
    });
}
