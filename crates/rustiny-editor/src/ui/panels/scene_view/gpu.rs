//! What the GPU scene pass needs from the scene view: the grid size and box picking.
//!
//! Which instances are drawn is decided by `crate::meshes::build_plan`; here, a click is matched
//! against the box each object is drawn with: its model's bounding box when it has one (see
//! `PickShapes`), otherwise the unit box placed by its world matrix.

use super::camera::Camera;
use crate::meshes::PickShapes;
use rustiny_scene::{Mat4, Scene};

/// Ground grid cells drawn each side of the camera.
const GRID_CELLS: f32 = 40.0;

/// Grid cell size and half extent for a camera at `height` above the ground: the cell is a power
/// of ten that follows the height so lines stay readable from close up and from far away.
pub fn grid_for(height: f32) -> (f32, f32) {
    let height = if height.is_finite() {
        height.abs().max(1.0)
    } else {
        1.0
    };
    let cell = 10.0_f32.powf(height.log10().floor());
    (cell, cell * GRID_CELLS)
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Distance along the ray `origin + t * dir` at which it enters the unit box placed by `world`
/// (an oriented box: no matrix inverse needed), or `None` for a miss, a box behind the origin or
/// a degenerate (zero or non-finite scale) one.
fn ray_box(origin: [f32; 3], dir: [f32; 3], world: &Mat4) -> Option<f32> {
    let centre = world.translation();
    let offset = [
        centre[0] - origin[0],
        centre[1] - origin[1],
        centre[2] - origin[2],
    ];
    let (mut near, mut far) = (f32::NEG_INFINITY, f32::INFINITY);
    for axis in 0..3 {
        let a = world.axis(axis);
        let length = dot(a, a).sqrt();
        if !length.is_finite() || length < 1e-6 {
            return None;
        }
        let u = [a[0] / length, a[1] / length, a[2] / length];
        let half = 0.5 * length;
        let along = dot(offset, u);
        let speed = dot(dir, u);
        if speed.abs() < 1e-9 {
            if along.abs() > half {
                return None;
            }
            continue;
        }
        let (t1, t2) = ((along - half) / speed, (along + half) / speed);
        near = near.max(t1.min(t2));
        far = far.min(t1.max(t2));
    }
    // `far` stays infinite only for a zero direction, which never happens for a mouse ray.
    if !far.is_finite() || near > far || far <= 0.0 {
        return None;
    }
    Some(if near > 0.0 { near } else { far })
}

/// The nearest active object whose box the mouse ray hits. `mouse` is in viewport pixels
/// (origin top-left), `size` the viewport size in the same units. Objects below an inactive
/// parent are not drawn and are skipped.
pub fn pick_box(
    camera: &Camera,
    size: [f32; 2],
    mouse: [f32; 2],
    scene: &Scene,
    shapes: &PickShapes,
) -> Option<usize> {
    let tan = (camera.fov_y * 0.5).tan();
    let ndc = [
        mouse[0] / size[0] * 2.0 - 1.0,
        mouse[1] / size[1] * 2.0 - 1.0,
    ];
    let (f, r, u) = (camera.forward(), camera.right(), camera.up());
    let sx = ndc[0] * (size[0] / size[1]) * tan;
    let sy = -ndc[1] * tan;
    let dir = [
        f[0] + r[0] * sx + u[0] * sy,
        f[1] + r[1] * sx + u[1] * sy,
        f[2] + r[2] * sx + u[2] * sy,
    ];
    (0..scene.len())
        .filter(|&id| scene.is_active(id))
        .filter_map(|id| {
            let own = u32::try_from(id).map(|o| shapes.get(o)).unwrap_or_default();
            let t = if own.is_empty() {
                ray_box(camera.eye, dir, scene.world_matrix(id)?)?
            } else {
                own.iter()
                    .filter_map(|m| ray_box(camera.eye, dir, m))
                    .min_by(f32::total_cmp)?
            };
            Some((id, t))
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(id, _)| id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustiny_scene::read_scene;
    use std::io::Cursor;

    /// A flat scene of objects given as `(position, scale, active)`.
    fn scene_of(objects: &[([f32; 3], f32, bool)]) -> Scene {
        let mut text = String::from("%YAML 1.1\n%TAG !u! tag:unity3d.com,2011:\n");
        for (i, (p, s, active)) in objects.iter().enumerate() {
            let (go, tf) = (i * 2 + 1, i * 2 + 2);
            text.push_str(&format!(
                "--- !u!1 &{go}\nGameObject:\n  m_Component:\n  - component: {{fileID: {tf}}}\n  m_Name: O{i}\n  m_IsActive: {}\n--- !u!4 &{tf}\nTransform:\n  m_GameObject: {{fileID: {go}}}\n  m_LocalRotation: {{x: 0, y: 0, z: 0, w: 1}}\n  m_LocalPosition: {{x: {}, y: {}, z: {}}}\n  m_LocalScale: {{x: {s}, y: {s}, z: {s}}}\n  m_Children: []\n  m_Father: {{fileID: 0}}\n",
                u8::from(*active),
                p[0],
                p[1],
                p[2]
            ));
        }
        read_scene(Cursor::new(text.into_bytes()), "T").unwrap()
    }

    fn camera() -> Camera {
        Camera {
            eye: [0.0; 3],
            yaw: 0.0,
            pitch: 0.0,
            fov_y: 60.0_f32.to_radians(),
        }
    }

    const SIZE: [f32; 2] = [800.0, 600.0];

    #[test]
    fn grid_cells_follow_the_camera_height() {
        assert_eq!(grid_for(0.0), (1.0, 40.0));
        assert_eq!(grid_for(5.0).0, 1.0);
        assert_eq!(grid_for(50.0).0, 10.0);
        assert_eq!(grid_for(-500.0).0, 100.0);
        assert_eq!(grid_for(f32::NAN), (1.0, 40.0));
    }

    #[test]
    fn a_ray_hits_a_box_in_front_and_misses_beside_or_behind() {
        let at = |x: f32, z: f32| Mat4::from_trs([x, 0.0, z], [0.0, 0.0, 0.0, 1.0], [2.0; 3]);
        let hit = ray_box([0.0; 3], [0.0, 0.0, 1.0], &at(0.0, 10.0)).unwrap();
        assert!(
            (hit - 9.0).abs() < 1e-4,
            "the near face of a 2m box at 10 is at 9: {hit}"
        );
        assert!(ray_box([0.0; 3], [0.0, 0.0, 1.0], &at(3.0, 10.0)).is_none());
        assert!(ray_box([0.0; 3], [0.0, 0.0, 1.0], &at(0.0, -10.0)).is_none());
        // Inside the box the exit point counts.
        let inside = ray_box([0.0; 3], [0.0, 0.0, 1.0], &at(0.0, 0.0)).unwrap();
        assert!((inside - 1.0).abs() < 1e-4);
        // A degenerate (zero-scale) box is never hit.
        let flat = Mat4::from_trs([0.0, 0.0, 10.0], [0.0, 0.0, 0.0, 1.0], [0.0; 3]);
        assert!(ray_box([0.0; 3], [0.0, 0.0, 1.0], &flat).is_none());
    }

    #[test]
    fn picking_takes_the_nearest_active_box_under_the_mouse() {
        let scene = scene_of(&[
            ([0.0, 0.0, 20.0], 4.0, true),
            ([0.0, 0.0, 8.0], 2.0, true),
            ([0.0, 0.0, 5.0], 2.0, false),
            ([40.0, 0.0, 20.0], 2.0, true),
        ]);
        let centre = [SIZE[0] / 2.0, SIZE[1] / 2.0];
        assert_eq!(
            pick_box(&camera(), SIZE, centre, &scene, &PickShapes::default()),
            Some(1),
            "the nearer active box wins and the inactive one in front is ignored"
        );
        assert_eq!(
            pick_box(&camera(), SIZE, [5.0, 5.0], &scene, &PickShapes::default()),
            None
        );
        // Object 3 sits off to the right: aim at it using the CPU projection.
        let p = camera().project([40.0, 0.0, 20.0], SIZE).unwrap();
        assert_eq!(
            pick_box(
                &camera(),
                SIZE,
                [p[0], p[1]],
                &scene,
                &PickShapes::default()
            ),
            Some(3)
        );
    }

    #[test]
    fn the_box_pick_agrees_with_where_the_box_is_drawn() {
        // A box's corner, projected with the same camera the GPU matrix comes from, is a pixel
        // that must pick the box; a pixel just outside must not.
        let scene = scene_of(&[([0.0, 0.0, 10.0], 2.0, true)]);
        let cam = camera();
        let corner = cam.project([0.99, 0.99, 9.0], SIZE).unwrap();
        assert_eq!(
            pick_box(
                &cam,
                SIZE,
                [corner[0], corner[1]],
                &scene,
                &PickShapes::default()
            ),
            Some(0)
        );
        let outside = cam.project([1.5, 0.0, 9.0], SIZE).unwrap();
        assert_eq!(
            pick_box(
                &cam,
                SIZE,
                [outside[0], outside[1]],
                &scene,
                &PickShapes::default()
            ),
            None
        );
    }

    #[test]
    fn a_pick_shape_replaces_the_unit_box_and_inactive_parents_hide_objects() {
        // Object 0 is a 1 m box at z = 10, but its mesh is a 10 m wide wall (pick shape).
        let scene = scene_of(&[([0.0, 0.0, 10.0], 1.0, true)]);
        let wide_aim = camera().project([3.0, 0.0, 10.0], SIZE).unwrap();
        let aim = [wide_aim[0], wide_aim[1]];
        assert_eq!(
            pick_box(&camera(), SIZE, aim, &scene, &PickShapes::default()),
            None
        );
        let shapes = PickShapes::default();
        shapes.add(
            0,
            Mat4::from_trs([0.0, 0.0, 10.0], [0.0, 0.0, 0.0, 1.0], [10.0, 1.0, 1.0]),
        );
        assert_eq!(pick_box(&camera(), SIZE, aim, &scene, &shapes), Some(0));
    }
}
