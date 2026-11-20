//! Opening a scene file on a worker thread. The UI thread never touches the file: a 300 MB scene
//! takes a noticeable fraction of a second to read and must not stall frames.
//!
//! Loading is read-only. Opening another scene while one is loading simply supersedes it: the
//! older worker finishes on its own and its result is discarded.

use super::mesh_load::MeshLoad;
use super::{App, UserEvent};
use crate::meshes::{MaterialIndex, MeshPlan, PickShapes, build_plan};
use crate::shaders::ShaderHub;
use crate::ui::SceneState;
use rustiny_assets::{AssetDatabase, Guid};
use rustiny_render::PostProcess;
use rustiny_scene::prefab::Library;
use rustiny_scene::{Scene, load_scene};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver};
use winit::event_loop::EventLoopProxy;

/// What a scene load hands back.
pub(super) struct SceneResult {
    path: PathBuf,
    /// The scene, what to draw for it and the project's materials (all prepared on the worker so
    /// the UI thread does not).
    outcome: Result<(Scene, MeshPlan, Option<MaterialIndex>, PostProcess), String>,
}

/// The project a scene is opened in: where model GUIDs are looked up.
type ModelLookup = Option<(PathBuf, Arc<AssetDatabase>)>;

/// The file a GUID names, when the project knows it.
fn resolve_guid(lookup: &ModelLookup, guid: &str) -> Option<PathBuf> {
    let (root, db) = lookup.as_ref()?;
    let guid: Guid = guid.parse().ok()?;
    let entry = db.by_guid(guid)?;
    Some(root.join(&entry.path))
}

fn spawn_load(
    path: PathBuf,
    proxy: EventLoopProxy<UserEvent>,
    lookup: ModelLookup,
) -> Receiver<SceneResult> {
    let (tx, rx) = mpsc::channel();
    let spawned = std::thread::Builder::new()
        .name("scene-load".to_owned())
        .spawn({
            let tx = tx.clone();
            let path = path.clone();
            move || {
                let outcome = load_scene(&path)
                    .map(|scene| {
                        let resolve = |guid: &str| resolve_guid(&lookup, guid);
                        let library = Library::new(&resolve);
                        let planning = std::time::Instant::now();
                        let plan = build_plan(&scene, &library, resolve);
                        let prefabs = library.stats();
                        tracing::info!(
                            "prefabs: {} instances expanded, {} unreadable; {} overrides applied, {} without a target; {} meshes; planned in {} ms",
                            plan.prefabs_expanded,
                            plan.prefabs_unreadable,
                            prefabs.applied,
                            prefabs.unmatched,
                            plan.sources.len(),
                            planning.elapsed().as_millis()
                        );
                        let materials = lookup
                            .as_ref()
                            .map(|(root, db)| MaterialIndex::build(root.clone(), db.clone()));
                        // URP's volume stack: the default profile, then the scene's volumes.
                        let post = lookup.as_ref().map_or_else(PostProcess::default, |(root, _)| {
                            let stack = crate::shaders::scene_volume_stack(root, &scene, resolve);
                            let post = crate::shaders::post_process_of(&stack);
                            tracing::info!("post-processing from the scene's volumes: {post:?}");
                            post
                        });
                        (scene, plan, materials, post)
                    })
                    .map_err(|e| e.to_string());
                // The receiver is gone when a newer load superseded this one, or at shutdown.
                let _ = tx.send(SceneResult { path, outcome });
                let _ = proxy.send_event(UserEvent::Wake);
            }
        });
    if let Err(e) = spawned {
        let _ = tx.send(SceneResult {
            path,
            outcome: Err(format!("cannot start scene loading thread: {e}")),
        });
    }
    rx
}

impl App {
    /// Start loading the scene at `path` (any `.unity` file; it does not have to be inside the
    /// open project).
    pub(super) fn open_scene(&mut self, path: PathBuf) {
        self.mesh_load = None;
        if self.open_project.is_none() && self.scan_in_flight {
            if let Some(run) = self.running.as_mut() {
                run.show_scene(
                    SceneState::Loading(path.display().to_string()),
                    &MeshPlan::default(),
                );
            }
            self.pending_scene = Some(path);
            return;
        }
        self.pending_scene = None;
        if let Some(run) = self.running.as_mut() {
            run.show_scene(
                SceneState::Loading(path.display().to_string()),
                &MeshPlan::default(),
            );
        }
        let lookup = self
            .open_project
            .as_ref()
            .map(|open| (open.root().to_path_buf(), open.db().clone()));
        self.scene_rx = Some(spawn_load(path, self.proxy.clone(), lookup));
    }

    /// The user picked a scene in the Project panel: its path is relative to the project root.
    pub(super) fn open_project_scene(&mut self, relative: &str) {
        let Some(open) = &self.open_project else {
            return;
        };
        let path = open.root().join(relative);
        self.open_scene(path);
    }

    /// Take a finished scene load, if there is one, and show it. Called when the loop is woken.
    pub(super) fn poll_scene(&mut self) {
        let Some(result) = self.scene_rx.as_ref().and_then(|rx| rx.try_recv().ok()) else {
            return;
        };
        self.scene_rx = None;

        let (state, plan, loading) = match result.outcome {
            Ok((scene, plan, materials, post)) => {
                if let Some(run) = self.running.as_mut() {
                    run.set_post_process(post);
                }
                tracing::info!(
                    "scene {} loaded: {} objects, {} roots, {} problems, {:.0} ms",
                    result.path.display(),
                    scene.len(),
                    scene.roots().len(),
                    scene.issues().len(),
                    scene.stats().duration.as_secs_f64() * 1000.0
                );
                tracing::info!(
                    "scene plan: {} model files, {} model instances, {} proxy boxes, {} coarser variants left out",
                    plan.sources.len(),
                    plan.mesh_instances(),
                    plan.boxes.len(),
                    plan.lods_skipped
                );
                self.report.scene = Some((scene.len(), scene.roots().len(), scene.issues().len()));
                let scene = Arc::new(scene);
                let shapes = Arc::new(PickShapes::default());
                let state = SceneState::Ready {
                    scene: scene.clone(),
                    path: display_path(&result.path, self.open_project_root()),
                    shapes: shapes.clone(),
                };
                (state, plan, Some((scene, shapes, materials)))
            }
            Err(why) => {
                tracing::warn!("cannot open scene {}: {why}", result.path.display());
                (SceneState::Failed(why), MeshPlan::default(), None)
            }
        };
        if let Some(run) = self.running.as_mut() {
            run.show_scene(state, &plan);
        }
        if let Some((scene, shapes, materials)) = loading
            && !plan.sources.is_empty()
        {
            let workers = std::thread::available_parallelism()
                .map_or(2, |n| n.get().saturating_sub(2).clamp(1, 6));
            let proxy = self.proxy.clone();
            let hub = self.open_project.as_ref().map(|open| {
                Arc::new(ShaderHub::new(
                    open.root().to_path_buf(),
                    open.db().clone(),
                    self.shader_cache.clone(),
                ))
            });
            self.mesh_load = Some(MeshLoad::start(
                &plan,
                scene,
                shapes,
                materials,
                hub,
                workers,
                move || {
                    // The loop is gone only during shutdown.
                    let _ = proxy.send_event(UserEvent::Wake);
                },
            ));
        }
    }

    fn open_project_root(&self) -> Option<&Path> {
        self.open_project.as_ref().map(|o| o.root())
    }
}

/// The path as shown to the user: relative to the project when it lies inside it.
fn display_path(path: &Path, root: Option<&Path>) -> String {
    root.and_then(|r| path.strip_prefix(r).ok())
        .unwrap_or(path)
        .display()
        .to_string()
}
