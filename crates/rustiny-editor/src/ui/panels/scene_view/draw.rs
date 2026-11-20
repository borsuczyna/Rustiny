//! Drawing the viewport contents with the window's ImGui draw list.
//!
//! This is CPU-projected 2D drawing, not a GPU scene renderer: it shows where objects are, not
//! what they look like. Everything stays within a fixed vertex budget (see `points`).

use super::camera::Camera;
use super::points::Point;
use dear_imgui_rs::DrawListMut;
use rustiny_scene::{Mat4, ObjectKind, SceneObject};

const GRID_CELLS: i32 = 20;
const GRID_COLOR: [f32; 4] = [1.0, 1.0, 1.0, 0.07];
const AXIS_EXTENT: f32 = 20_000.0;
const SELECTED_COLOR: [f32; 4] = [1.0, 0.95, 0.2, 1.0];
const AXIS_COLORS: [[f32; 4]; 3] = [
    [0.95, 0.3, 0.3, 1.0],
    [0.4, 0.9, 0.4, 1.0],
    [0.35, 0.55, 1.0, 1.0],
];

fn at(origin: [f32; 2], p: [f32; 2]) -> [f32; 2] {
    [origin[0] + p[0], origin[1] + p[1]]
}

/// Dark background of the viewport.
pub(super) fn background(dl: &DrawListMut<'_>, origin: [f32; 2], size: [f32; 2]) {
    dl.add_rect(origin, at(origin, size), [0.09, 0.095, 0.105, 1.0])
        .filled(true)
        .build();
}

/// A ground grid on the XZ plane around the camera, with the world X and Z axes highlighted.
/// The spacing is a power of ten that follows the camera's height.
pub(super) fn grid(dl: &DrawListMut<'_>, cam: &Camera, origin: [f32; 2], size: [f32; 2]) {
    let height = cam.eye[1].abs().max(1.0);
    let spacing = 10.0_f32.powf(height.log10().floor());
    let centre = [
        (cam.eye[0] / spacing).round() * spacing,
        (cam.eye[2] / spacing).round() * spacing,
    ];
    let reach = GRID_CELLS as f32 * spacing;
    let line = |a: [f32; 3], b: [f32; 3], color: [f32; 4]| {
        if let Some((p, q)) = cam.project_segment(a, b, size) {
            dl.add_line(at(origin, p), at(origin, q), color).build();
        }
    };
    for i in -GRID_CELLS..=GRID_CELLS {
        let offset = i as f32 * spacing;
        let x = centre[0] + offset;
        let z = centre[1] + offset;
        line(
            [x, 0.0, centre[1] - reach],
            [x, 0.0, centre[1] + reach],
            GRID_COLOR,
        );
        line(
            [centre[0] - reach, 0.0, z],
            [centre[0] + reach, 0.0, z],
            GRID_COLOR,
        );
    }
    line(
        [-AXIS_EXTENT, 0.0, 0.0],
        [AXIS_EXTENT, 0.0, 0.0],
        [AXIS_COLORS[0][0], AXIS_COLORS[0][1], AXIS_COLORS[0][2], 0.5],
    );
    line(
        [0.0, 0.0, -AXIS_EXTENT],
        [0.0, 0.0, AXIS_EXTENT],
        [AXIS_COLORS[2][0], AXIS_COLORS[2][1], AXIS_COLORS[2][2], 0.5],
    );
}

/// Colour of an object's marker.
pub(super) fn marker_color(object: &SceneObject) -> [f32; 4] {
    match (&object.kind, object.active) {
        (ObjectKind::PrefabInstance { .. }, _) => [1.0, 0.65, 0.25, 0.9],
        (ObjectKind::GameObject, true) => [0.78, 0.82, 0.92, 0.9],
        (ObjectKind::GameObject, false) => [0.45, 0.45, 0.5, 0.6],
    }
}

/// One small square per object.
pub(super) fn markers(
    dl: &DrawListMut<'_>,
    origin: [f32; 2],
    points: &[Point],
    color_of: impl Fn(usize) -> [f32; 4],
) {
    for p in points {
        let c = at(origin, p.pos);
        dl.add_rect(
            [c[0] - 1.5, c[1] - 1.5],
            [c[0] + 1.5, c[1] + 1.5],
            color_of(p.id),
        )
        .filled(true)
        .build();
    }
}

/// Highlight the selected object: a ring, its local axes (constant size on screen) and its name.
pub(super) fn selection(
    dl: &DrawListMut<'_>,
    cam: &Camera,
    origin: [f32; 2],
    size: [f32; 2],
    world: &Mat4,
    name: &str,
) {
    let position = world.translation();
    let Some([x, y, depth]) = cam.project(position, size) else {
        return;
    };
    let centre = at(origin, [x, y]);
    dl.add_circle(centre, 6.0, SELECTED_COLOR)
        .thickness(1.5)
        .build();
    dl.add_text([centre[0] + 9.0, centre[1] - 18.0], SELECTED_COLOR, name);

    let length = depth * 0.08;
    for (i, color) in AXIS_COLORS.iter().enumerate() {
        let axis = world.axis(i);
        let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        if !norm.is_finite() || norm <= 1e-9 {
            continue;
        }
        let end = [
            position[0] + axis[0] / norm * length,
            position[1] + axis[1] / norm * length,
            position[2] + axis[2] / norm * length,
        ];
        if let Some((a, b)) = cam.project_segment(position, end, size) {
            dl.add_line(at(origin, a), at(origin, b), *color)
                .thickness(2.0)
                .build();
        }
    }
}
