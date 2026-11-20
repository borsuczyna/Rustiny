//! Scene panel: a free-fly view of the open scene.
//!
//! Two ways of drawing, chosen by whether the application has a GPU texture for the view:
//!
//! * **GPU** (normal): the panel only owns the camera and the interaction. It publishes a
//!   [`SceneViewRequest`] each frame; the application renders every object as an instanced unit
//!   box (a stand-in for its real mesh, which needs asset import) over a ground grid into an
//!   offscreen image, and the panel shows that image. Clicks pick the nearest box under the mouse.
//! * **CPU markers** (no GPU target yet, and in headless tests): every object is a small marker at
//!   its *world* position drawn with the panel's ImGui draw list. Clicks pick the nearest marker.
//!
//! Either way a selected object also gets a ring, its local axes and its name as an overlay.
//!
//! Controls: hold the right mouse button over the view to look around and fly with W A S D
//! (Q / E down / up, Shift faster), mouse wheel to dolly, left click to select, `F` to frame
//! the selection, `Home` to frame the whole scene.

mod camera;
mod draw;
mod gpu;
mod points;

use super::hierarchy::{self, State as HierarchyState};
use crate::meshes::PickShapes;
use crate::ui::{ProjectState, SceneState, SceneViewRequest};
use camera::Camera;
use dear_imgui_rs::{Key, MouseButton, TextureId, Ui, WindowKey};
use points::Point;
use rustiny_scene::Scene;

/// Most markers drawn per frame (see `points` for why).
const MARKER_BUDGET: usize = 10_000;
/// Pixels within which a click selects a marker.
const PICK_RADIUS: f32 = 9.0;
/// Fly speed multiplier while Shift is held.
const FAST: f32 = 4.0;
/// Fly speed a view starts with, in metres per second.
const DEFAULT_SPEED: f32 = 10.0;
/// The wheel turned while flying changes the speed by this factor per notch, as in Unity.
const SPEED_STEP: f32 = 1.25;
/// Slowest and fastest fly speed.
const SPEED_RANGE: (f32, f32) = (0.1, 2000.0);

/// View state of the Scene panel.
#[derive(Debug)]
pub struct State {
    camera: Camera,
    /// Metres per second of flying; changed with the wheel while flying.
    speed: f32,
    /// The scene's extent, once computed: `(centre, radius)`.
    extent: Option<([f32; 3], f32)>,
    /// Frame the whole scene at the next opportunity.
    frame_pending: bool,
    /// The right mouse button is held and the camera is being steered.
    looking: bool,
    visible: Vec<Point>,
    /// The GPU-rendered view, when the application has one (set before each frame).
    texture: Option<TextureId>,
    /// What the panel wants drawn on the GPU this frame (taken by the application after).
    request: Option<SceneViewRequest>,
    /// A camera placement from the command line, used instead of framing the first scene opened:
    /// `[x, y, z, yaw degrees, pitch degrees]`.
    initial_view: Option<[f32; 5]>,
}

impl Default for State {
    fn default() -> Self {
        Self {
            camera: Camera::default(),
            speed: DEFAULT_SPEED,
            extent: None,
            frame_pending: true,
            looking: false,
            visible: Vec::new(),
            texture: None,
            request: None,
            initial_view: None,
        }
    }
}

impl State {
    /// Where the camera is, in world space.
    pub fn eye(&self) -> [f32; 3] {
        self.camera.eye
    }

    /// Forget everything about the previous scene (a different scene was opened).
    pub fn clear(&mut self) {
        let texture = self.texture;
        let initial_view = self.initial_view;
        *self = Self::default();
        self.texture = texture;
        self.initial_view = initial_view;
    }

    /// Start the camera at `view` (`[x, y, z, yaw degrees, pitch degrees]`) when the next scene
    /// is shown, instead of framing it.
    pub fn set_initial_view(&mut self, view: Option<[f32; 5]>) {
        self.initial_view = view;
    }

    /// Tell the panel which GPU texture shows the scene (`None`: draw CPU markers instead).
    pub fn set_texture(&mut self, texture: Option<TextureId>) {
        self.texture = texture;
    }

    /// The GPU draw the panel asked for this frame, if it showed a viewport.
    pub fn take_request(&mut self) -> Option<SceneViewRequest> {
        self.request.take()
    }
}

/// The Scene panel.
pub fn draw(
    ui: &Ui,
    key: &WindowKey,
    open: &mut bool,
    scene: &SceneState,
    project: &ProjectState,
    selection: &mut HierarchyState,
    state: &mut State,
) {
    if !*open {
        return;
    }
    ui.window(key).opened(open).build(|| match scene {
        SceneState::Ready { scene, shapes, .. } => {
            draw_viewport(ui, scene, shapes, project, selection, state);
        }
        SceneState::Loading(_) => ui.text_disabled("Loading scene ..."),
        _ => {
            ui.text_disabled("No scene open.");
            ui.text_disabled("Open a .unity file to see where its objects are.");
        }
    });
}

fn draw_viewport(
    ui: &Ui,
    scene: &Scene,
    shapes: &PickShapes,
    project: &ProjectState,
    selection: &mut HierarchyState,
    state: &mut State,
) {
    let size = ui.content_region_avail();
    if size[0] < 64.0 || size[1] < 64.0 {
        return;
    }
    let origin = ui.cursor_screen_pos();
    let gpu = state.texture;
    if let Some(texture) = gpu {
        ui.image(texture, size);
        ui.set_cursor_screen_pos(origin);
    }
    ui.invisible_button("##scene-viewport", size);
    let hovered = ui.is_item_hovered();

    let (centre, radius) = *state.extent.get_or_insert_with(|| scene_extent(scene));
    if std::mem::take(&mut state.frame_pending) {
        tracing::info!("scene view: bulk of the scene is {radius:.0} m around {centre:?}");
        if let Some([x, y, z, yaw, pitch]) = state.initial_view.take() {
            state.camera.eye = [x, y, z];
            state.camera.yaw = yaw.to_radians();
            state.camera.pitch = pitch.to_radians();
        } else {
            // Look down at the scene from a distance, which reads better than a level view.
            state.camera.pitch = -0.7;
            state.camera.frame(centre, radius);
        }
    }
    steer(ui, scene, selection, state, hovered, (centre, radius));

    // Ask for a GPU draw at the panel's physical size; the image appears from the next frame on.
    let scale = ui.io().display_framebuffer_scale();
    let (grid_cell, grid_extent) = gpu::grid_for(state.camera.eye[1]);
    state.request = Some(SceneViewRequest {
        size_px: [
            (size[0] * scale[0]).round().max(1.0) as u32,
            (size[1] * scale[1]).round().max(1.0) as u32,
        ],
        view_proj: state.camera.view_projection(size[0] / size[1]),
        view: state.camera.view_matrix(),
        projection: state.camera.projection(size[0] / size[1]),
        eye: state.camera.eye,
        selected: selection.selected.and_then(|id| u32::try_from(id).ok()),
        grid_cell,
        grid_extent,
    });

    let project_markers = |camera: &Camera, out: &mut Vec<Point>| {
        points::collect_visible(
            camera,
            (0..scene.len()).filter_map(|id| scene.world_position(id).map(|p| (id, p))),
            size,
            MARKER_BUDGET,
            out,
        )
    };
    // With the GPU image the markers are only needed to pick small objects on a click.
    let stats = gpu
        .is_none()
        .then(|| project_markers(&state.camera, &mut state.visible));
    if hovered && !state.looking && ui.is_mouse_clicked(MouseButton::Left) {
        let mouse = ui.mouse_pos();
        let local = [mouse[0] - origin[0], mouse[1] - origin[1]];
        selection.selected = if gpu.is_some() {
            gpu::pick_box(&state.camera, size, local, scene, shapes).or_else(|| {
                project_markers(&state.camera, &mut state.visible);
                points::pick(&state.visible, local, PICK_RADIUS)
            })
        } else {
            points::pick(&state.visible, local, PICK_RADIUS)
        };
    }

    let dl = ui.get_window_draw_list();
    let end = [origin[0] + size[0], origin[1] + size[1]];
    dl.with_clip_rect(origin, end, || {
        if stats.is_some() {
            draw::background(&dl, origin, size);
            draw::grid(&dl, &state.camera, origin, size);
            draw::markers(&dl, origin, &state.visible, |id| {
                scene.object(id).map_or([1.0; 4], draw::marker_color)
            });
        }
        if let Some(id) = selection.selected
            && let (Some(world), Some(object)) = (scene.world_matrix(id), scene.object(id))
        {
            let name = hierarchy::display_name(object, project);
            draw::selection(&dl, &state.camera, origin, size, world, &name);
        }
        let info = match &stats {
            Some(stats) => format!(
                "{} objects | CPU markers: {} in view{} | speed {:.0}/s",
                scene.len(),
                stats.on_screen,
                if stats.kept < stats.on_screen {
                    format!(" (nearest {} drawn)", stats.kept)
                } else {
                    String::new()
                },
                state.speed
            ),
            None => format!(
                "{} objects (GPU: models where imported, boxes otherwise) | speed {:.0}/s",
                scene.len(),
                state.speed
            ),
        };
        dl.add_text(
            [origin[0] + 8.0, origin[1] + 6.0],
            [0.75, 0.75, 0.8, 0.9],
            info,
        );
        dl.add_text(
            [origin[0] + 8.0, end[1] - 20.0],
            [0.6, 0.6, 0.65, 0.8],
            "RMB + WASD fly (Shift faster, wheel sets speed) | wheel dolly | click select | F frame selection | Home frame all",
        );
    });
}

/// Mouse look, fly keys, wheel dolly and the framing shortcuts.
fn steer(
    ui: &Ui,
    scene: &Scene,
    selection: &HierarchyState,
    state: &mut State,
    hovered: bool,
    (centre, radius): ([f32; 3], f32),
) {
    let right_down = ui.is_mouse_down(MouseButton::Right);
    if right_down {
        state.looking |= hovered;
    } else {
        state.looking = false;
    }

    let io = ui.io();
    if state.looking {
        let [dx, dy] = io.mouse_delta();
        state.camera.look(dx, dy);
        let boost = if ui.is_key_down(Key::LeftShift) {
            FAST
        } else {
            1.0
        };
        let step = state.speed * boost * io.delta_time().min(0.1);
        let axis = |plus: Key, minus: Key| {
            f32::from(i8::from(ui.is_key_down(plus)) - i8::from(ui.is_key_down(minus)))
        };
        state.camera.fly(
            axis(Key::W, Key::S) * step,
            axis(Key::D, Key::A) * step,
            axis(Key::E, Key::Q) * step,
        );
    }
    if hovered {
        let wheel = io.mouse_wheel();
        if wheel != 0.0 && state.looking {
            // While flying the wheel sets the speed (Unity's Scene view does the same).
            state.speed =
                (state.speed * SPEED_STEP.powf(wheel)).clamp(SPEED_RANGE.0, SPEED_RANGE.1);
        } else if wheel != 0.0 {
            state.camera.fly(wheel * state.speed * 0.25, 0.0, 0.0);
        }
        if ui.is_key_pressed(Key::Home) {
            state.camera.frame(centre, radius);
        }
        if ui.is_key_pressed(Key::F)
            && let Some(position) = selection.selected.and_then(|id| scene.world_position(id))
        {
            state.camera.frame(position, (radius * 0.01).max(2.0));
        }
    }
}

/// Centre and radius of the bulk of the scene. Outliers (a stray object far away) are ignored by
/// using the 2nd to 98th percentile of every axis rather than the min and max.
fn scene_extent(scene: &Scene) -> ([f32; 3], f32) {
    let mut lo = [0.0; 3];
    let mut hi = [0.0; 3];
    for axis in 0..3 {
        let mut values: Vec<f32> = (0..scene.len())
            .filter_map(|id| scene.world_position(id))
            .map(|p| p[axis])
            .filter(|v| v.is_finite())
            .collect();
        if values.is_empty() {
            continue;
        }
        values.sort_by(f32::total_cmp);
        let at = |q: f64| values[((values.len() - 1) as f64 * q) as usize];
        lo[axis] = at(0.02);
        hi[axis] = at(0.98);
    }
    let centre = [
        (lo[0] + hi[0]) * 0.5,
        (lo[1] + hi[1]) * 0.5,
        (lo[2] + hi[2]) * 0.5,
    ];
    let half = [
        (hi[0] - lo[0]) * 0.5,
        (hi[1] - lo[1]) * 0.5,
        (hi[2] - lo[2]) * 0.5,
    ];
    let radius = (half[0] * half[0] + half[1] * half[1] + half[2] * half[2]).sqrt();
    (centre, radius.max(5.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustiny_scene::read_scene;
    use std::io::Cursor;

    fn scene_with_positions(positions: &[[f32; 3]]) -> Scene {
        let mut text = String::from("%YAML 1.1\n%TAG !u! tag:unity3d.com,2011:\n");
        for (i, p) in positions.iter().enumerate() {
            let (go, tf) = (i * 2 + 1, i * 2 + 2);
            text.push_str(&format!(
                "--- !u!1 &{go}\nGameObject:\n  m_Component:\n  - component: {{fileID: {tf}}}\n  m_Name: O{i}\n--- !u!4 &{tf}\nTransform:\n  m_GameObject: {{fileID: {go}}}\n  m_LocalRotation: {{x: 0, y: 0, z: 0, w: 1}}\n  m_LocalPosition: {{x: {}, y: {}, z: {}}}\n  m_LocalScale: {{x: 1, y: 1, z: 1}}\n  m_Children: []\n  m_Father: {{fileID: 0}}\n",
                p[0], p[1], p[2]
            ));
        }
        read_scene(Cursor::new(text.into_bytes()), "T").unwrap()
    }

    #[test]
    fn extent_ignores_far_outliers() {
        let mut positions: Vec<[f32; 3]> =
            (0..100).map(|i| [i as f32, 0.0, (i % 10) as f32]).collect();
        positions.push([1.0e6, 0.0, 0.0]);
        let (centre, radius) = scene_extent(&scene_with_positions(&positions));
        assert!((centre[0] - 50.0).abs() < 5.0, "{centre:?}");
        assert!(
            radius < 200.0,
            "one outlier must not blow up the extent: {radius}"
        );
    }

    #[test]
    fn extent_of_a_tiny_scene_has_a_minimum_size() {
        let (_, radius) = scene_extent(&scene_with_positions(&[[0.0; 3]]));
        assert!(radius >= 5.0);
        let (centre, radius) = scene_extent(&scene_with_positions(&[]));
        assert_eq!(centre, [0.0; 3]);
        assert!(radius >= 5.0);
    }

    #[test]
    fn framing_the_scene_shows_most_objects() {
        let positions: Vec<[f32; 3]> = (0..400)
            .map(|i| [(i % 20) as f32 * 10.0, 0.0, (i / 20) as f32 * 10.0])
            .collect();
        let scene = scene_with_positions(&positions);
        let (centre, radius) = scene_extent(&scene);
        let mut camera = Camera::default();
        camera.frame(centre, radius);
        let mut out = Vec::new();
        let stats = points::collect_visible(
            &camera,
            (0..scene.len()).filter_map(|id| scene.world_position(id).map(|p| (id, p))),
            [800.0, 600.0],
            10_000,
            &mut out,
        );
        assert!(
            stats.on_screen >= 380,
            "only {} of 400 in view",
            stats.on_screen
        );
    }
}

/// Cost of the per-frame CPU work on a real scene. Ignored by default; run with
/// `RUSTINY_BENCH_SCENE=<file.unity> cargo test --release -p rustiny-editor --lib -- --ignored --nocapture bench`.
#[cfg(test)]
mod bench {
    use super::*;
    use std::time::Instant;

    #[test]
    #[ignore = "needs RUSTINY_BENCH_SCENE"]
    fn bench_projection_on_a_real_scene() {
        let Some(path) = std::env::var_os("RUSTINY_BENCH_SCENE") else {
            eprintln!("RUSTINY_BENCH_SCENE is not set; nothing measured");
            return;
        };
        let scene = rustiny_scene::load_scene(std::path::Path::new(&path)).unwrap();
        let (centre, radius) = scene_extent(&scene);
        let mut camera = Camera {
            pitch: -0.7,
            ..Camera::default()
        };
        camera.frame(centre, radius);
        let mut out = Vec::new();
        let mut times = Vec::new();
        for _ in 0..50 {
            let start = Instant::now();
            points::collect_visible(
                &camera,
                (0..scene.len()).filter_map(|id| scene.world_position(id).map(|p| (id, p))),
                [880.0, 600.0],
                MARKER_BUDGET,
                &mut out,
            );
            times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(f64::total_cmp);
        eprintln!(
            "collect_visible over {} objects: median {:.3} ms, worst {:.3} ms ({} kept)",
            scene.len(),
            times[times.len() / 2],
            times[times.len() - 1],
            out.len()
        );
        let start = Instant::now();
        let _ = scene_extent(&scene);
        eprintln!(
            "scene_extent (once per scene): {:.1} ms",
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
}
