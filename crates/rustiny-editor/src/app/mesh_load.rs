//! Bringing a scene's models and their textures onto the GPU: workers import and decode them (see
//! `crate::meshes`), this takes the finished ones on the UI thread within a small time budget per
//! frame and uploads them.
//!
//! A model that cannot be imported keeps (or gets back) its proxy box, so a broken file never
//! makes an object vanish. A texture that cannot be decoded leaves its sections with the plain
//! tint of their material.

use super::scene_gpu::SceneGpu;
use crate::meshes::{
    Look, MaterialIndex, MeshPlan, MeshStream, PickShapes, ResolveStats, Resolver, TextureStream,
    pick_matrix, to_upload,
};
use crate::shaders::ShaderHub;
use rustiny_render::GpuInstance;
use rustiny_scene::Scene;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// How long one frame may spend taking and uploading finished models.
const BUDGET: Duration = Duration::from_millis(5);
/// How long one frame may spend taking finished textures.
const TEXTURE_BUDGET: Duration = Duration::from_millis(3);
/// Failures logged one by one; later ones are only counted.
const LOGGED_FAILURES: usize = 10;

/// What happened to the scene's textures.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct TextureCounts {
    /// Textures the materials asked for.
    pub requested: usize,
    /// Textures uploaded.
    pub uploaded: usize,
    /// Textures that could not be decoded or uploaded.
    pub failed: usize,
}

/// One scene's models, from the first import to the last upload.
pub(super) struct MeshLoad {
    stream: MeshStream,
    textures: TextureStream,
    resolver: Arc<Resolver>,
    hub: Option<Arc<ShaderHub>>,
    scene: Arc<Scene>,
    shapes: Arc<PickShapes>,
    objects_of: Vec<Vec<u32>>,
    worlds_of: Vec<Vec<rustiny_scene::Mat4>>,
    boxes: Vec<GpuInstance>,
    failed: Vec<u32>,
    uploaded: usize,
    uploaded_triangles: u64,
    texture_counts: TextureCounts,
    started: Instant,
    finished: bool,
    /// The skybox material being compiled on a worker: the hub's id, once known.
    sky: Option<mpsc::Receiver<Option<u32>>>,
}

impl MeshLoad {
    /// Start importing the plan's models. `materials` finds each model's `.mat` files (`None`
    /// draws everything in the default grey).
    pub(super) fn start(
        plan: &MeshPlan,
        scene: Arc<Scene>,
        shapes: Arc<PickShapes>,
        materials: Option<MaterialIndex>,
        hub: Option<Arc<ShaderHub>>,
        workers: usize,
        wake: impl Fn() + Clone + Send + Sync + 'static,
    ) -> Self {
        let (jobs_tx, jobs_rx) = mpsc::channel();
        let resolver =
            Arc::new(Resolver::new(materials.unwrap_or_default(), jobs_tx).with_hub(hub.clone()));
        let looks = {
            let resolver = resolver.clone();
            Arc::new(
                move |source: &crate::meshes::MeshSource, mesh: &rustiny_model::MeshData| {
                    resolver.looks(source, mesh)
                },
            )
        };
        let texture_workers = workers.div_ceil(2).max(1);
        // The skybox material: reading and compiling it is file and compiler work, so it runs on
        // a thread of its own; `pump` hands the result to the renderer.
        let sky = scene
            .lighting()
            .settings
            .as_ref()
            .and_then(|s| s.skybox.clone())
            .filter(|_| hub.is_some())
            .and_then(|skybox| {
                let (tx, rx) = mpsc::channel();
                let resolver = resolver.clone();
                let wake = wake.clone();
                std::thread::Builder::new()
                    .name("skybox-material".into())
                    .spawn(move || {
                        let id = resolver.hub_material(&skybox.guid, skybox.file_id);
                        if id.is_none() {
                            tracing::warn!(
                                "the skybox material {} cannot be drawn with its shader",
                                skybox.guid
                            );
                        }
                        let _ = tx.send(id);
                        wake();
                    })
                    .ok()
                    .map(|_| rx)
            });
        Self {
            stream: MeshStream::start(plan.sources.clone(), workers, looks, wake.clone()),
            textures: TextureStream::start(jobs_rx, texture_workers, wake),
            resolver,
            hub,
            scene,
            shapes,
            objects_of: plan.objects_of.clone(),
            worlds_of: plan.worlds_of.clone(),
            boxes: plan.boxes.clone(),
            failed: Vec::new(),
            uploaded: 0,
            uploaded_triangles: 0,
            texture_counts: TextureCounts::default(),
            started: Instant::now(),
            finished: plan.sources.is_empty(),
            sky,
        }
    }

    /// Models the scene uses.
    pub(super) fn total(&self) -> usize {
        self.stream.total()
    }

    /// Models uploaded to the GPU so far.
    pub(super) fn uploaded(&self) -> usize {
        self.uploaded
    }

    /// Models that could not be imported or uploaded.
    pub(super) fn failed(&self) -> usize {
        self.failed.len()
    }

    /// Textures requested, uploaded and failed so far.
    pub(super) fn textures(&self) -> TextureCounts {
        TextureCounts {
            requested: self.resolver.submitted(),
            ..self.texture_counts
        }
    }

    /// How many material slots found a `.mat` of the same name.
    pub(super) fn resolve_stats(&self) -> ResolveStats {
        self.resolver.stats()
    }

    /// Whether every model and texture has been dealt with.
    pub(super) fn is_finished(&self) -> bool {
        self.finished
    }

    /// Take what the workers have finished and upload it. Returns whether more is expected, so
    /// the caller keeps drawing frames until it is all there (textures need a few frames after
    /// their upload for the copy to the GPU).
    pub(super) fn pump(&mut self, gpu: &mut SceneGpu) -> bool {
        if let Some(rx) = &self.sky
            && let Ok(id) = rx.try_recv()
        {
            self.sky = None;
            if let (Some(id), Some(hub)) = (id, &self.hub) {
                gpu.set_skybox(hub, id);
            }
        }
        if self.finished {
            return self.sky.is_some();
        }
        let Self {
            stream,
            textures,
            hub,
            shapes,
            objects_of,
            worlds_of,
            failed,
            uploaded,
            uploaded_triangles,
            texture_counts,
            ..
        } = self;
        let models_more = stream.drain(BUDGET, |loaded| {
            let key = loaded.key;
            let model = match loaded.result {
                Ok(model) => model,
                Err(why) => {
                    if failed.len() < LOGGED_FAILURES {
                        tracing::warn!("model not imported: {why}");
                    }
                    failed.push(key);
                    return;
                }
            };
            let mesh = model.mesh;
            let looks = model.looks;
            let mut upload = to_upload(&mesh, |material| {
                material
                    .and_then(|i| looks.get(i))
                    .copied()
                    .unwrap_or(Look::DEFAULT)
            });
            if let Some(hub) = hub {
                gpu.use_shaders(hub, &mut upload);
            }
            if let Err(e) = gpu.add_mesh(key, &upload) {
                if failed.len() < LOGGED_FAILURES {
                    tracing::warn!("model not uploaded: {e}");
                }
                failed.push(key);
                return;
            }
            *uploaded += 1;
            *uploaded_triangles += mesh.triangle_count() as u64;
            if let (Some(bounds), Some(objects), Some(worlds)) = (
                mesh.bounds(),
                objects_of.get(key as usize),
                worlds_of.get(key as usize),
            ) {
                for (&object, world) in objects.iter().zip(worlds) {
                    shapes.add(object, pick_matrix(world, &bounds));
                }
            }
        });
        textures.drain(TEXTURE_BUDGET, |loaded| {
            let key = loaded.key;
            match loaded.result {
                Ok(texture) => match gpu.add_texture(key, &texture, loaded.srgb) {
                    Ok(()) => texture_counts.uploaded += 1,
                    Err(e) => {
                        if texture_counts.failed < LOGGED_FAILURES {
                            tracing::warn!("texture not uploaded: {e}");
                        }
                        texture_counts.failed += 1;
                    }
                },
                Err(why) => {
                    if texture_counts.failed < LOGGED_FAILURES {
                        tracing::warn!("texture not decoded: {why}");
                    }
                    texture_counts.failed += 1;
                }
            }
        });
        let texture_results = self.textures.received();
        let textures_more = texture_results < self.resolver.submitted();
        // Once everything is in, passes are still needed until the copies have been recorded.
        let more = models_more || textures_more || gpu.textures_waiting() > 0;
        if !more {
            self.finish(gpu);
        }
        more
    }

    fn finish(&mut self, gpu: &mut SceneGpu) {
        self.finished = true;
        let resolve = self.resolver.stats();
        let textures = self.textures();
        let stats = gpu.mesh_stats();
        tracing::info!(
            "models: {} of {} uploaded ({} triangles), {} failed, {:.1} s since the scene opened",
            self.uploaded,
            self.stream.total(),
            self.uploaded_triangles,
            self.failed.len(),
            self.started.elapsed().as_secs_f64()
        );
        if let Some(hub) = &self.hub {
            let shaders = hub.stats();
            tracing::info!(
                "project shaders: {} parsed, {} pass variants compiled, {} rejected by dxc, {} \
                 materials drawn with them ({} draws, {} left out last frame)",
                shaders.shaders,
                shaders.compiled,
                shaders.failed,
                shaders.materials,
                stats.shader_draws,
                stats.shader_skipped
            );
        }
        tracing::info!(
            "materials: {} of {} slots matched a .mat by name; textures: {} requested, {} uploaded, \
             {} failed, {:.1} MB on the GPU",
            resolve.matched,
            resolve.slots,
            textures.requested,
            textures.uploaded,
            textures.failed,
            stats.texture_bytes as f64 / 1e6
        );
        if self.failed.is_empty() {
            return;
        }
        // Objects whose model failed are shown as boxes again.
        let mut boxes = std::mem::take(&mut self.boxes);
        for &key in &self.failed {
            for &object in self.objects_of.get(key as usize).into_iter().flatten() {
                boxes.push(crate::meshes::proxy_box(
                    &self.scene,
                    object as usize,
                    self.scene.is_active(object as usize),
                ));
            }
        }
        gpu.set_instances(&boxes);
        self.boxes = boxes;
    }
}
