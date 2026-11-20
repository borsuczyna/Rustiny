//! Choosing which objects to draw in the viewport.
//!
//! ImGui draw lists use 16-bit indices here, so one window can hold at most ~65 000 vertices.
//! Objects are therefore drawn as small quads under a fixed budget: everything on screen if it
//! fits, otherwise the nearest ones.

use super::camera::Camera;

/// An object that landed inside the viewport.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    /// Index of the object in the scene.
    pub id: usize,
    /// Viewport pixel position.
    pub pos: [f32; 2],
    /// Distance along the view direction.
    pub depth: f32,
}

/// What [`collect_visible`] found.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VisibleStats {
    /// Objects in front of the camera and inside the viewport.
    pub on_screen: usize,
    /// How many of them were kept (the nearest, up to the budget).
    pub kept: usize,
}

/// Project `positions` and keep those inside a `size` viewport, nearest first up to `budget`.
/// `out` is cleared and reused so a frame allocates nothing once it has grown.
pub fn collect_visible(
    camera: &Camera,
    positions: impl Iterator<Item = (usize, [f32; 3])>,
    size: [f32; 2],
    budget: usize,
    out: &mut Vec<Point>,
) -> VisibleStats {
    out.clear();
    for (id, world) in positions {
        let Some([x, y, depth]) = camera.project(world, size) else {
            continue;
        };
        if (0.0..size[0]).contains(&x) && (0.0..size[1]).contains(&y) {
            out.push(Point {
                id,
                pos: [x, y],
                depth,
            });
        }
    }
    let on_screen = out.len();
    if on_screen > budget {
        out.select_nth_unstable_by(budget, |a, b| a.depth.total_cmp(&b.depth));
        out.truncate(budget);
    }
    VisibleStats {
        on_screen,
        kept: out.len(),
    }
}

/// The point closest to `mouse` within `radius` pixels (the nearer one to the camera on ties).
pub fn pick(points: &[Point], mouse: [f32; 2], radius: f32) -> Option<usize> {
    let limit = radius * radius;
    points
        .iter()
        .map(|p| {
            let (dx, dy) = (p.pos[0] - mouse[0], p.pos[1] - mouse[1]);
            (dx * dx + dy * dy, p)
        })
        .filter(|(d2, _)| *d2 <= limit)
        .min_by(|(a, pa), (b, pb)| a.total_cmp(b).then(pa.depth.total_cmp(&pb.depth)))
        .map(|(_, p)| p.id)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIZE: [f32; 2] = [800.0, 600.0];

    fn camera() -> Camera {
        Camera {
            eye: [0.0; 3],
            yaw: 0.0,
            pitch: 0.0,
            fov_y: 90.0_f32.to_radians(),
        }
    }

    #[test]
    fn only_points_in_front_and_on_screen_are_kept() {
        let positions = [
            (0, [0.0, 0.0, 10.0]),    // centre
            (1, [0.0, 0.0, -10.0]),   // behind
            (2, [1000.0, 0.0, 10.0]), // far to the right, off screen
            (3, [f32::NAN, 0.0, 10.0]),
            (4, [5.0, 5.0, 10.0]),
        ];
        let mut out = Vec::new();
        let stats = collect_visible(&camera(), positions.into_iter(), SIZE, 100, &mut out);
        let mut ids: Vec<_> = out.iter().map(|p| p.id).collect();
        ids.sort_unstable();
        assert_eq!(ids, [0, 4]);
        assert_eq!(
            stats,
            VisibleStats {
                on_screen: 2,
                kept: 2
            }
        );
    }

    #[test]
    fn over_budget_keeps_the_nearest() {
        let positions = (0..100).map(|i| (i, [0.0, 0.0, 1.0 + i as f32]));
        let mut out = Vec::new();
        let stats = collect_visible(&camera(), positions, SIZE, 10, &mut out);
        assert_eq!(
            stats,
            VisibleStats {
                on_screen: 100,
                kept: 10
            }
        );
        assert!(out.iter().all(|p| p.id < 10), "{out:?}");
    }

    #[test]
    fn the_buffer_is_reused_between_frames() {
        let mut out = Vec::new();
        collect_visible(
            &camera(),
            (0..50).map(|i| (i, [0.0, 0.0, 5.0])),
            SIZE,
            100,
            &mut out,
        );
        let capacity = out.capacity();
        collect_visible(
            &camera(),
            (0..10).map(|i| (i, [0.0, 0.0, 5.0])),
            SIZE,
            100,
            &mut out,
        );
        assert_eq!(out.len(), 10);
        assert_eq!(out.capacity(), capacity);
    }

    #[test]
    fn picking_takes_the_closest_within_the_radius() {
        let points = [
            Point {
                id: 1,
                pos: [100.0, 100.0],
                depth: 5.0,
            },
            Point {
                id: 2,
                pos: [104.0, 100.0],
                depth: 5.0,
            },
            Point {
                id: 3,
                pos: [300.0, 300.0],
                depth: 1.0,
            },
        ];
        assert_eq!(pick(&points, [103.0, 100.0], 10.0), Some(2));
        assert_eq!(pick(&points, [300.0, 305.0], 10.0), Some(3));
        assert_eq!(pick(&points, [200.0, 200.0], 10.0), None);
        // Same distance: the nearer to the camera wins.
        let tie = [
            Point {
                id: 7,
                pos: [10.0, 10.0],
                depth: 9.0,
            },
            Point {
                id: 8,
                pos: [10.0, 10.0],
                depth: 2.0,
            },
        ];
        assert_eq!(pick(&tie, [10.0, 10.0], 5.0), Some(8));
        assert_eq!(pick(&[], [0.0, 0.0], 5.0), None);
    }
}
