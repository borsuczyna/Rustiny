//! Opening a Unity project and keeping it up to date, always off the UI thread.
//!
//! Flow: `--project` starts a worker that detects the project, starts the file watcher and only
//! then scans (so a change made during the scan is not missed: it queues a follow-up scan). Each
//! burst of changes under `Assets/` triggers a rescan that reuses the `.meta` results of unchanged
//! files. At most one scan runs at a time; changes that arrive meanwhile are folded into a single
//! follow-up scan. Everything here only reads the project.

use super::{App, UserEvent};
use crate::ui::ProjectState;
use rustiny_assets::{AssetDatabase, ProjectWatcher, UnityProject};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;
use winit::event_loop::EventLoopProxy;

/// How long the file system must be quiet before a rescan starts.
const QUIET_BEFORE_RESCAN: Duration = Duration::from_millis(300);

/// What a scan worker should do.
pub(super) enum ScanRequest {
    /// Detect, start watching, and fully scan the project at this path.
    Open(PathBuf),
    /// Scan again, reusing what `previous` already knows.
    Refresh {
        project: UnityProject,
        previous: Arc<AssetDatabase>,
    },
}

/// What a scan worker hands back.
pub(super) struct ScanResult {
    state: ProjectState,
    /// Set by an `Open` request whose watcher started successfully.
    watcher: Option<ProjectWatcher>,
}

/// A project that has been scanned successfully at least once.
pub(super) struct OpenProject {
    project: UnityProject,
    db: Arc<AssetDatabase>,
    /// Kept alive to keep receiving change notifications; `None` if watching could not start.
    watcher: Option<ProjectWatcher>,
}

impl OpenProject {
    /// The project's root folder.
    pub(super) fn root(&self) -> &Path {
        &self.project.root
    }

    /// The asset index of the last successful scan.
    pub(super) fn db(&self) -> &Arc<AssetDatabase> {
        &self.db
    }
}

fn spawn_scan(request: ScanRequest, proxy: EventLoopProxy<UserEvent>) -> Receiver<ScanResult> {
    let (tx, rx) = mpsc::channel();
    let spawned = std::thread::Builder::new()
        .name("project-scan".to_owned())
        .spawn({
            let tx = tx.clone();
            move || {
                let result = run_scan(request, &proxy);
                // The receiver is gone only during shutdown.
                let _ = tx.send(result);
                let _ = proxy.send_event(UserEvent::Wake);
            }
        });
    if let Err(e) = spawned {
        let _ = tx.send(ScanResult {
            state: ProjectState::Failed(format!("cannot start scan thread: {e}")),
            watcher: None,
        });
    }
    rx
}

fn start_watcher(
    project: &UnityProject,
    proxy: &EventLoopProxy<UserEvent>,
) -> Option<ProjectWatcher> {
    let proxy = proxy.clone();
    match ProjectWatcher::start(project, QUIET_BEFORE_RESCAN, move || {
        // The loop is gone only during shutdown.
        let _ = proxy.send_event(UserEvent::ProjectChanged);
    }) {
        Ok(w) => Some(w),
        Err(e) => {
            tracing::warn!("not watching the project for changes: {e}");
            None
        }
    }
}

fn run_scan(request: ScanRequest, proxy: &EventLoopProxy<UserEvent>) -> ScanResult {
    let (project, watcher, result) = match request {
        ScanRequest::Open(path) => match UnityProject::detect(&path) {
            Ok(project) => {
                // Watch first: anything that changes from here on is reported, even while the
                // initial scan is still walking the tree.
                let watcher = start_watcher(&project, proxy);
                let result = AssetDatabase::scan(&project);
                (project, watcher, result)
            }
            Err(e) => {
                return ScanResult {
                    state: ProjectState::Failed(e.to_string()),
                    watcher: None,
                };
            }
        },
        ScanRequest::Refresh { project, previous } => {
            let result = previous.rescan(&project);
            (project, None, result)
        }
    };
    let state = match result {
        Ok(db) => ProjectState::Ready {
            project,
            db: Arc::new(db),
            watching: false, // filled in by `project_ready`, which owns the watcher
        },
        Err(e) => ProjectState::Failed(e.to_string()),
    };
    ScanResult { state, watcher }
}

impl App {
    /// Start opening the project at `path` (called once the window exists).
    pub(super) fn open_project(&mut self, path: PathBuf) {
        if let Some(run) = self.running.as_mut() {
            run.editor_ui
                .set_project(ProjectState::Scanning(path.display().to_string()));
        }
        self.start_scan(ScanRequest::Open(path));
    }

    fn start_scan(&mut self, request: ScanRequest) {
        self.scan_in_flight = true;
        self.project_rx = Some(spawn_scan(request, self.proxy.clone()));
    }

    /// Take a finished scan, if there is one, and show it. Called when the loop is woken.
    pub(super) fn poll_project(&mut self) {
        let Some(result) = self.project_rx.as_ref().and_then(|rx| rx.try_recv().ok()) else {
            return;
        };
        self.project_rx = None;
        self.scan_in_flight = false;

        match result.state {
            ProjectState::Ready { project, db, .. } => {
                self.project_ready(project, db, result.watcher);
                if let Some(path) = self.pending_scene.take() {
                    self.open_scene(path);
                }
            }
            ProjectState::Failed(why) => {
                if self.open_project.is_some() {
                    // A refresh failed (e.g. `Assets/` vanished briefly); keep showing the last
                    // good index rather than an error.
                    tracing::warn!("project rescan failed, keeping the previous index: {why}");
                } else if let Some(run) = self.running.as_mut() {
                    run.editor_ui.set_project(ProjectState::Failed(why));
                }
                // A scene that waited for this project opens without being able to find models.
                if let Some(path) = self.pending_scene.take() {
                    self.open_scene(path);
                }
            }
            other => {
                if let Some(run) = self.running.as_mut() {
                    run.editor_ui.update_project(other);
                }
            }
        }

        if std::mem::take(&mut self.rescan_queued) {
            self.on_project_changed();
        }
    }

    fn project_ready(
        &mut self,
        project: UnityProject,
        db: Arc<AssetDatabase>,
        new_watcher: Option<ProjectWatcher>,
    ) {
        let stats = db.stats();
        self.report.project = Some((stats.files, stats.folders, db.issues().len()));
        self.report.project_scans += 1;

        let first = self.open_project.is_none();
        let watcher =
            new_watcher.or_else(|| self.open_project.as_mut().and_then(|o| o.watcher.take()));
        let watching = watcher.is_some();
        self.open_project = Some(OpenProject {
            project: project.clone(),
            db: db.clone(),
            watcher,
        });

        if let Some(run) = self.running.as_mut() {
            let state = ProjectState::Ready {
                project,
                db,
                watching,
            };
            if first {
                run.editor_ui.set_project(state);
            } else {
                run.editor_ui.update_project(state);
            }
            run.redraw_pending = true;
        }
    }

    /// The watcher reported changes: rescan now, or right after the scan that is running.
    pub(super) fn on_project_changed(&mut self) {
        if self.scan_in_flight {
            self.rescan_queued = true;
            return;
        }
        let Some(open) = &self.open_project else {
            return;
        };
        let request = ScanRequest::Refresh {
            project: open.project.clone(),
            previous: open.db.clone(),
        };
        self.start_scan(request);
    }
}
