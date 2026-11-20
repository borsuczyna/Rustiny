//! Watching a project's `Assets/` folder for changes.
//!
//! Strictly read-only: the OS is asked to report changes, nothing is ever written. Raw file system
//! events are noisy (one save can produce several), so they are collapsed: after the first
//! relevant event the watcher waits until things have been quiet for a moment and then calls the
//! callback once.

use crate::database::is_ignored_name;
use crate::project::UnityProject;
use notify::{EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use std::path::{Component, Path, PathBuf};
use std::sync::mpsc;
use std::time::Duration;

/// Why watching could not start.
#[derive(Debug, thiserror::Error)]
pub enum WatchError {
    /// The OS watcher failed.
    #[error("cannot watch `{}`: {message}", path.display())]
    Notify {
        /// The folder.
        path: PathBuf,
        /// Error text.
        message: String,
    },
    /// The debounce thread could not be started.
    #[error("cannot start the watcher thread: {0}")]
    Thread(std::io::Error),
}

/// Reports changes under a project's `Assets/` folder until dropped.
#[derive(Debug)]
pub struct ProjectWatcher {
    // Dropping the watcher drops its callback and with it the channel sender, which ends the
    // debounce thread.
    _watcher: RecommendedWatcher,
}

impl ProjectWatcher {
    /// Start watching `project/Assets`. `on_change` runs on a background thread once per burst of
    /// changes, after no relevant event has arrived for `quiet`.
    pub fn start(
        project: &UnityProject,
        quiet: Duration,
        on_change: impl Fn() + Send + 'static,
    ) -> Result<Self, WatchError> {
        let assets = project.root.join("Assets");
        let (tx, rx) = mpsc::channel::<()>();

        let filter_root = assets.clone();
        let mut watcher =
            notify::recommended_watcher(move |event: notify::Result<notify::Event>| {
                if let Ok(event) = event
                    && is_relevant(&filter_root, &event)
                {
                    // The receiver only disappears while shutting down.
                    let _ = tx.send(());
                }
            })
            .map_err(|e| WatchError::Notify {
                path: assets.clone(),
                message: e.to_string(),
            })?;
        watcher
            .watch(&assets, RecursiveMode::Recursive)
            .map_err(|e| WatchError::Notify {
                path: assets.clone(),
                message: e.to_string(),
            })?;

        std::thread::Builder::new()
            .name("project-watch".to_owned())
            .spawn(move || {
                // Block for the first event, then keep absorbing until a quiet period passes.
                while rx.recv().is_ok() {
                    loop {
                        match rx.recv_timeout(quiet) {
                            Ok(()) => {}
                            Err(mpsc::RecvTimeoutError::Timeout) => break,
                            Err(mpsc::RecvTimeoutError::Disconnected) => return,
                        }
                    }
                    on_change();
                }
            })
            .map_err(WatchError::Thread)?;
        Ok(Self { _watcher: watcher })
    }
}

/// An event matters if it changes something (not just reads it) and no part of the path below
/// `Assets/` is a name Unity ignores.
fn is_relevant(assets: &Path, event: &notify::Event) -> bool {
    if matches!(event.kind, EventKind::Access(_)) {
        return false;
    }
    event.paths.iter().any(|path| {
        let Ok(rel) = path.strip_prefix(assets) else {
            return true;
        };
        rel.components().all(|c| match c {
            Component::Normal(name) => name.to_str().is_none_or(|n| !is_ignored_name(n)),
            _ => true,
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Instant;

    fn fixture(name: &str) -> (PathBuf, UnityProject) {
        let root =
            std::env::temp_dir().join(format!("rustiny-watch-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("Assets")).unwrap();
        fs::create_dir_all(root.join("ProjectSettings")).unwrap();
        let project = UnityProject::detect(&root).unwrap();
        (root, project)
    }

    fn wait_until(limit: Duration, mut done: impl FnMut() -> bool) -> bool {
        let start = Instant::now();
        while start.elapsed() < limit {
            if done() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        done()
    }

    #[test]
    fn a_burst_of_changes_calls_back_once_and_hidden_files_are_ignored() {
        let (root, project) = fixture("burst");
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = calls.clone();
        let _watcher = ProjectWatcher::start(&project, Duration::from_millis(150), move || {
            counter.fetch_add(1, Ordering::SeqCst);
        })
        .unwrap();

        for i in 0..5 {
            fs::write(root.join(format!("Assets/f{i}.txt")), "x").unwrap();
        }
        assert!(
            wait_until(Duration::from_secs(10), || calls.load(Ordering::SeqCst)
                >= 1),
            "no change was reported"
        );
        std::thread::sleep(Duration::from_millis(500));
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "the burst must be collapsed"
        );

        // Hidden files and `~` folders are not assets, so they must not trigger anything.
        fs::write(root.join("Assets/.hidden"), "x").unwrap();
        fs::create_dir_all(root.join("Assets/Cache~")).unwrap();
        fs::write(root.join("Assets/Cache~/data.bin"), "x").unwrap();
        std::thread::sleep(Duration::from_millis(700));
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "ignored paths must not report"
        );

        fs::write(root.join("Assets/f0.txt"), "changed").unwrap();
        assert!(
            wait_until(Duration::from_secs(10), || calls.load(Ordering::SeqCst)
                >= 2),
            "a later change was not reported"
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn relevance_filter() {
        let assets = Path::new("/p/Assets");
        let ev = |kind, path: &str| notify::Event::new(kind).add_path(PathBuf::from(path));
        let modify = EventKind::Modify(notify::event::ModifyKind::Any);
        assert!(is_relevant(assets, &ev(modify, "/p/Assets/a/b.png")));
        assert!(!is_relevant(assets, &ev(modify, "/p/Assets/.git/x")));
        assert!(!is_relevant(assets, &ev(modify, "/p/Assets/Cache~/x")));
        assert!(!is_relevant(
            assets,
            &ev(
                EventKind::Access(notify::event::AccessKind::Any),
                "/p/Assets/a.png"
            )
        ));
    }
}
