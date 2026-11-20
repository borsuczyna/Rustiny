//! Importing models on worker threads and handing them to the UI thread as they finish.
//!
//! The UI thread never reads or parses a model: it only drains finished meshes with a time budget
//! per frame and uploads them. The channel is bounded, so workers wait when the UI is slower than
//! they are instead of piling up decoded meshes in memory.

use super::convert::Look;
use super::plan::{MeshSource, SourceKind};
use rustiny_model::{ImportSettings, MeshData, fbx, import_mesh_asset};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::time::{Duration, Instant};

/// Finished meshes waiting for the UI thread at most.
const IN_FLIGHT: usize = 32;

/// How the materials of an imported model are drawn: from the model file and the mesh, one
/// [`Look`] per entry of [`MeshData::materials`]. Runs on the worker threads.
pub type LooksFn = dyn Fn(&MeshSource, &MeshData) -> Vec<Look> + Send + Sync;

/// An imported model and how its materials look.
#[derive(Debug)]
pub struct LoadedMesh {
    /// The geometry.
    pub mesh: MeshData,
    /// One look per material of the mesh.
    pub looks: Vec<Look>,
}

/// One model's result.
#[derive(Debug)]
pub struct Loaded {
    /// The key of the [`MeshSource`] it came from.
    pub key: u32,
    /// The model, or why the file could not be imported.
    pub result: Result<LoadedMesh, String>,
}

/// Models being imported in the background.
#[derive(Debug)]
pub struct MeshStream {
    rx: Receiver<Loaded>,
    total: usize,
    received: usize,
    cancel: Arc<AtomicBool>,
}

impl MeshStream {
    /// Start importing `sources` on `workers` threads, resolving materials with `looks`. `wake`
    /// is called (from a worker) after each result so the UI can come and collect it.
    pub fn start(
        sources: Vec<MeshSource>,
        workers: usize,
        looks: Arc<LooksFn>,
        wake: impl Fn() + Send + Sync + 'static,
    ) -> Self {
        let total = sources.len();
        let (tx, rx) = mpsc::sync_channel(IN_FLIGHT);
        let cancel = Arc::new(AtomicBool::new(false));
        let sources = Arc::new(sources);
        let next = Arc::new(AtomicUsize::new(0));
        let wake = Arc::new(wake);
        for n in 0..workers.clamp(1, 16).min(total.max(1)) {
            let (tx, sources, next, cancel, wake, looks) = (
                tx.clone(),
                sources.clone(),
                next.clone(),
                cancel.clone(),
                wake.clone(),
                looks.clone(),
            );
            let spawned = std::thread::Builder::new()
                .name(format!("mesh-import-{n}"))
                .spawn(move || {
                    while !cancel.load(Ordering::Relaxed) {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(source) = sources.get(i) else { break };
                        let result = import_file(source).map(|mesh| {
                            let looks = looks(source, &mesh);
                            LoadedMesh { mesh, looks }
                        });
                        let loaded = Loaded {
                            key: source.key,
                            result,
                        };
                        // The receiver is gone when a newer scene replaced this stream.
                        if tx.send(loaded).is_err() {
                            break;
                        }
                        wake();
                    }
                });
            if let Err(e) = spawned {
                tracing::warn!("cannot start a model import thread: {e}");
            }
        }
        Self {
            rx,
            total,
            received: 0,
            cancel,
        }
    }

    /// Models the stream was asked to import.
    pub fn total(&self) -> usize {
        self.total
    }

    /// Results taken so far (successes and failures).
    pub fn received(&self) -> usize {
        self.received
    }

    /// Whether every model has been taken.
    pub fn is_done(&self) -> bool {
        self.received >= self.total
    }

    /// Take finished results and pass them to `sink` until `budget` has been used up (at least
    /// one is taken when any is ready). Returns whether more are expected.
    pub fn drain(&mut self, budget: Duration, mut sink: impl FnMut(Loaded)) -> bool {
        let deadline = Instant::now() + budget;
        loop {
            match self.rx.try_recv() {
                Ok(loaded) => {
                    self.received += 1;
                    sink(loaded);
                }
                Err(TryRecvError::Empty) => return !self.is_done(),
                // Every worker has finished (or none could start).
                Err(TryRecvError::Disconnected) => {
                    self.received = self.total;
                    return false;
                }
            }
            if Instant::now() >= deadline {
                return !self.is_done();
            }
        }
    }
}

impl Drop for MeshStream {
    fn drop(&mut self) {
        // Workers notice at their next file (or when a send fails) and end on their own.
        self.cancel.store(true, Ordering::Relaxed);
    }
}

/// Read and import one mesh: a model file, or a Unity `Mesh` asset.
fn import_file(source: &MeshSource) -> Result<MeshData, String> {
    match source.kind {
        SourceKind::Model => import_model(&source.path, source.part),
        SourceKind::MeshAsset => {
            let text = std::fs::read_to_string(&source.path)
                .map_err(|e| format!("{}: {e}", source.path.display()))?;
            import_mesh_asset(&text).map_err(|e| format!("{}: {e}", source.path.display()))
        }
    }
}

/// Read and import one model; its `.meta` (next to it) supplies the scale settings.
fn import_model(path: &Path, part: Option<i64>) -> Result<MeshData, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut meta = path.as_os_str().to_owned();
    meta.push(".meta");
    let settings = std::fs::read_to_string(&meta)
        .map(|text| ImportSettings::from_meta(&text))
        .unwrap_or_default();
    // A renderer that names one mesh of the model by its fileID gets that mesh alone, in its own
    // space. When no node of the file has that ID the model is not drawn at all (the object keeps
    // its proxy box): drawing the whole file in its place would show a whole vehicle for one part.
    if let Some(id) = part {
        return fbx::import_part(&bytes, settings, fbx::Part::Mesh(id))
            .map_err(|e| format!("{} (mesh {id}): {e}", path.display()));
    }
    fbx::import(&bytes, settings).map_err(|e| format!("{}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("rustiny-stream-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn no_looks() -> Arc<LooksFn> {
        Arc::new(|_, _| Vec::new())
    }

    fn drain_all(stream: &mut MeshStream) -> Vec<Loaded> {
        let mut out = Vec::new();
        let start = Instant::now();
        while !stream.is_done() {
            stream.drain(Duration::from_millis(5), |l| out.push(l));
            assert!(start.elapsed() < Duration::from_secs(20), "stream stalled");
            std::thread::sleep(Duration::from_millis(1));
        }
        out
    }

    #[test]
    fn every_source_reports_back_even_when_it_cannot_be_imported() {
        let dir = temp_dir("failures");
        let garbage = dir.join("garbage.fbx");
        std::fs::write(&garbage, b"this is not an FBX file at all").unwrap();
        let sources = vec![
            MeshSource {
                key: 0,
                path: garbage,
                part: None,
                kind: SourceKind::Model,
                materials: Vec::new(),
            },
            MeshSource {
                key: 1,
                path: dir.join("missing.fbx"),
                part: None,
                kind: SourceKind::Model,
                materials: Vec::new(),
            },
        ];
        let woken = Arc::new(AtomicU32::new(0));
        let counter = woken.clone();
        let mut stream = MeshStream::start(sources, 2, no_looks(), move || {
            counter.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(stream.total(), 2);
        let mut results = drain_all(&mut stream);
        results.sort_by_key(|l| l.key);
        assert_eq!(results.len(), 2);
        assert!(
            results[0]
                .result
                .as_ref()
                .unwrap_err()
                .contains("not a binary FBX")
        );
        assert!(
            results[1].result.is_err(),
            "a missing file is an error, not a panic"
        );
        assert_eq!(stream.received(), 2);
        assert_eq!(woken.load(Ordering::Relaxed), 2, "one wake per result");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn an_empty_stream_is_done_at_once() {
        let mut stream = MeshStream::start(Vec::new(), 4, no_looks(), || {});
        assert!(stream.is_done());
        assert!(!stream.drain(Duration::from_millis(1), |_| panic!("nothing to take")));
    }

    #[test]
    fn dropping_the_stream_stops_the_workers_without_blocking() {
        let dir = temp_dir("cancel");
        let sources: Vec<MeshSource> = (0..200)
            .map(|key| MeshSource {
                key,
                path: dir.join(format!("nothing-{key}.fbx")),
                part: None,
                kind: SourceKind::Model,
                materials: Vec::new(),
            })
            .collect();
        let stream = MeshStream::start(sources, 2, no_looks(), || {});
        let start = Instant::now();
        drop(stream);
        assert!(start.elapsed() < Duration::from_secs(1));
        let _ = std::fs::remove_dir_all(dir);
    }
}
