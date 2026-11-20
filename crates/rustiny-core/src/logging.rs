//! Structured logging built on `tracing`.
//!
//! Besides the usual stderr/file sinks this module provides an in-memory [`LogBuffer`] that the
//! editor's Console panel renders. The buffer is bounded, so a log storm cannot grow memory
//! without limit, and it can wake the UI event loop through a notifier instead of being polled.

use crate::error::{Result, RustinyError};
use std::collections::VecDeque;
use std::fmt::Write as _;
use std::fs::File;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};
use tracing::field::{Field, Visit};
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

/// One captured log record.
#[derive(Debug, Clone)]
pub struct LogEntry {
    /// Monotonic sequence number (never reused, survives `clear`).
    pub seq: u64,
    /// Severity.
    pub level: Level,
    /// Module path / target that emitted the record.
    pub target: String,
    /// Formatted message including structured fields.
    pub message: String,
    /// Time since the buffer was created.
    pub elapsed: Duration,
}

type Notifier = Arc<dyn Fn() + Send + Sync>;

/// Bounded, thread-safe ring buffer of recent log records.
pub struct LogBuffer {
    entries: Mutex<VecDeque<LogEntry>>,
    capacity: usize,
    next_seq: AtomicU64,
    start: Instant,
    notifier: Mutex<Option<Notifier>>,
}

impl std::fmt::Debug for LogBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LogBuffer")
            .field("capacity", &self.capacity)
            .finish_non_exhaustive()
    }
}

impl LogBuffer {
    /// Create a buffer that keeps at most `capacity` records (minimum 1).
    pub fn new(capacity: usize) -> Arc<Self> {
        let capacity = capacity.max(1);
        Arc::new(Self {
            entries: Mutex::new(VecDeque::with_capacity(capacity.min(1024))),
            capacity,
            next_seq: AtomicU64::new(0),
            start: Instant::now(),
            notifier: Mutex::new(None),
        })
    }

    fn lock(&self) -> MutexGuard<'_, VecDeque<LogEntry>> {
        // A panic while holding the lock cannot leave the deque in an invalid state, so a
        // poisoned lock is safe to recover from. Logging must never itself panic.
        self.entries.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Register a callback invoked (from the logging thread) after every new record.
    pub fn set_notifier(&self, notifier: impl Fn() + Send + Sync + 'static) {
        *self.notifier.lock().unwrap_or_else(|p| p.into_inner()) = Some(Arc::new(notifier));
    }

    /// Append a record, evicting the oldest one when full.
    pub fn push(&self, level: Level, target: &str, message: String) {
        let entry = LogEntry {
            seq: self.next_seq.fetch_add(1, Ordering::Relaxed),
            level,
            target: target.to_owned(),
            message,
            elapsed: self.start.elapsed(),
        };
        {
            let mut entries = self.lock();
            if entries.len() == self.capacity {
                entries.pop_front();
            }
            entries.push_back(entry);
        }
        let notifier = self
            .notifier
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone();
        if let Some(notify) = notifier {
            notify();
        }
    }

    /// Number of records currently held.
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    /// `true` if no records are held.
    pub fn is_empty(&self) -> bool {
        self.lock().is_empty()
    }

    /// Sequence number the next record will receive (cheap change detector for the UI).
    pub fn next_seq(&self) -> u64 {
        self.next_seq.load(Ordering::Relaxed)
    }

    /// Run `f` over the held records without cloning them.
    pub fn with_entries<R>(&self, f: impl FnOnce(&VecDeque<LogEntry>) -> R) -> R {
        f(&self.lock())
    }

    /// Copy out the held records.
    pub fn snapshot(&self) -> Vec<LogEntry> {
        self.lock().iter().cloned().collect()
    }

    /// Drop all held records.
    pub fn clear(&self) {
        self.lock().clear();
    }
}

struct CaptureLayer {
    buffer: Arc<LogBuffer>,
}

#[derive(Default)]
struct MessageVisitor {
    message: String,
    fields: String,
}

impl Visit for MessageVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            let _ = write!(self.message, "{value:?}");
        } else {
            if !self.fields.is_empty() {
                self.fields.push(' ');
            }
            let _ = write!(self.fields, "{}={value:?}", field.name());
        }
    }
}

impl<S: Subscriber> Layer<S> for CaptureLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let mut visitor = MessageVisitor::default();
        event.record(&mut visitor);
        let mut message = visitor.message;
        if !visitor.fields.is_empty() {
            if !message.is_empty() {
                message.push(' ');
            }
            message.push_str(&visitor.fields);
        }
        self.buffer.push(
            *event.metadata().level(),
            event.metadata().target(),
            message,
        );
    }
}

/// Parameters for [`init`].
#[derive(Debug, Clone)]
pub struct LogOptions {
    /// Filter directive used when `RUST_LOG` is unset.
    pub filter: String,
    /// Optional log file. An existing file is rotated to `<name>.prev` first.
    pub file: Option<PathBuf>,
    /// Number of records kept for the in-editor console.
    pub buffer_capacity: usize,
}

impl Default for LogOptions {
    fn default() -> Self {
        Self {
            filter: "info".to_owned(),
            file: None,
            buffer_capacity: 5000,
        }
    }
}

/// Handle returned by [`init`].
#[derive(Debug, Clone)]
pub struct LogHandle {
    /// In-memory records for the Console panel.
    pub buffer: Arc<LogBuffer>,
    /// Log file in use, if any.
    pub file: Option<PathBuf>,
}

/// Install the global `tracing` subscriber (stderr + optional file + in-memory capture).
///
/// May be called at most once per process.
pub fn init(options: &LogOptions) -> Result<LogHandle> {
    let filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new(&options.filter))
        .map_err(|e| RustinyError::Logging(format!("bad filter `{}`: {e}", options.filter)))?;

    let buffer = LogBuffer::new(options.buffer_capacity);
    let stderr_layer = tracing_subscriber::fmt::layer()
        .with_target(true)
        .with_writer(std::io::stderr);

    let mut used_file = None;
    let file_layer = match &options.file {
        Some(path) => match open_log_file(path) {
            Ok(file) => {
                used_file = Some(path.clone());
                Some(
                    tracing_subscriber::fmt::layer()
                        .with_ansi(false)
                        .with_writer(Mutex::new(file)),
                )
            }
            Err(e) => {
                // Not fatal: the editor must still start if the log directory is read-only.
                eprintln!("warning: cannot open log file {}: {e}", path.display());
                None
            }
        },
        None => None,
    };

    tracing_subscriber::registry()
        .with(filter)
        .with(stderr_layer)
        .with(file_layer)
        .with(CaptureLayer {
            buffer: buffer.clone(),
        })
        .try_init()
        .map_err(|e| RustinyError::Logging(e.to_string()))?;

    Ok(LogHandle {
        buffer,
        file: used_file,
    })
}

fn open_log_file(path: &std::path::Path) -> std::io::Result<File> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if path.exists() {
        let mut prev = path.as_os_str().to_owned();
        prev.push(".prev");
        // Rotation failure should not prevent logging; ignore and truncate below.
        let _ = std::fs::rename(path, prev);
    }
    File::create(path)
}

/// Route panics through `tracing` so that crashes end up in the log file with a backtrace.
///
/// The previously installed hook (usually the default one printing to stderr) still runs.
pub fn install_panic_hook() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let backtrace = std::backtrace::Backtrace::force_capture();
        tracing::error!(target: "rustiny::panic", "PANIC: {info}\n{backtrace}");
        previous(info);
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_buffer_evicts_oldest_and_keeps_sequence() {
        let buf = LogBuffer::new(3);
        for i in 0..5 {
            buf.push(Level::INFO, "t", format!("m{i}"));
        }
        let msgs: Vec<_> = buf.snapshot().into_iter().map(|e| e.message).collect();
        assert_eq!(msgs, ["m2", "m3", "m4"]);
        assert_eq!(buf.next_seq(), 5);
        buf.clear();
        assert!(buf.is_empty());
        buf.push(Level::WARN, "t", "after clear".into());
        assert_eq!(
            buf.snapshot()[0].seq,
            5,
            "sequence numbers are never reused"
        );
    }

    #[test]
    fn notifier_fires_per_record() {
        let buf = LogBuffer::new(8);
        let count = Arc::new(AtomicU64::new(0));
        let c = count.clone();
        buf.set_notifier(move || {
            c.fetch_add(1, Ordering::SeqCst);
        });
        buf.push(Level::INFO, "t", "a".into());
        buf.push(Level::INFO, "t", "b".into());
        assert_eq!(count.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn capture_layer_records_message_and_fields() {
        let buffer = LogBuffer::new(16);
        let subscriber = tracing_subscriber::registry().with(CaptureLayer {
            buffer: buffer.clone(),
        });
        tracing::subscriber::with_default(subscriber, || {
            tracing::warn!(answer = 42, "hello {}", "world");
        });
        let entries = buffer.snapshot();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].level, Level::WARN);
        assert_eq!(entries[0].message, "hello world answer=42");
    }

    #[test]
    fn push_from_many_threads_is_bounded() {
        let buf = LogBuffer::new(50);
        std::thread::scope(|s| {
            for t in 0..8 {
                let buf = &buf;
                s.spawn(move || {
                    for i in 0..200 {
                        buf.push(Level::DEBUG, "t", format!("{t}-{i}"));
                    }
                });
            }
        });
        assert_eq!(buf.len(), 50);
        assert_eq!(buf.next_seq(), 1600);
    }
}
