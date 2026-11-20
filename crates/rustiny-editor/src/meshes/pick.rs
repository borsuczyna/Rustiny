//! The shapes objects are clicked with.
//!
//! A proxy box is clicked through the object's world matrix. A mesh object is clicked through the
//! mesh's bounding box, which is only known once the model has been imported, so the shapes are
//! filled in as models arrive. The panel reads them on the UI thread; the lock is only there so
//! the scene state can be shared and cloned.

use rustiny_model::Bounds;
use rustiny_scene::Mat4;
use std::collections::HashMap;
use std::sync::RwLock;

/// Thinnest a pick box is made along any axis (local units), so flat meshes such as roads can
/// still be hit.
const MIN_THICKNESS: f32 = 0.05;

/// Pick boxes of objects that are not plain proxy boxes, by object id. An object made of several
/// meshes (a prefab with parts) has one box per mesh.
#[derive(Debug, Default)]
pub struct PickShapes {
    boxes: RwLock<HashMap<u32, Vec<Mat4>>>,
}

impl PickShapes {
    /// Add `matrix` (a unit box placed in the world) as a box that picks `object`.
    pub fn add(&self, object: u32, matrix: Mat4) {
        self.boxes
            .write()
            .unwrap_or_else(|p| p.into_inner())
            .entry(object)
            .or_default()
            .push(matrix);
    }

    /// The pick boxes of `object` (empty when it has none of its own).
    pub fn get(&self, object: u32) -> Vec<Mat4> {
        self.boxes
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .get(&object)
            .cloned()
            .unwrap_or_default()
    }

    /// Number of objects with their own pick box.
    pub fn len(&self) -> usize {
        self.boxes.read().unwrap_or_else(|p| p.into_inner()).len()
    }

    /// Whether no object has its own pick box.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The unit-box matrix that covers `bounds` of a mesh placed by `world`.
pub fn pick_matrix(world: &Mat4, bounds: &Bounds) -> Mat4 {
    let centre = bounds.centre();
    let half = bounds.half_extent();
    let size = [
        (half[0] * 2.0).max(MIN_THICKNESS),
        (half[1] * 2.0).max(MIN_THICKNESS),
        (half[2] * 2.0).max(MIN_THICKNESS),
    ];
    world.mul(&Mat4::from_trs(centre, [0.0, 0.0, 0.0, 1.0], size))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-4)
    }

    #[test]
    fn the_pick_box_covers_the_bounds_in_the_world() {
        let bounds = Bounds {
            min: [-1.0, 0.0, -2.0],
            max: [1.0, 4.0, 2.0],
        };
        let world = Mat4::from_trs([10.0, 0.0, 5.0], [0.0, 0.0, 0.0, 1.0], [2.0, 2.0, 2.0]);
        let m = pick_matrix(&world, &bounds);
        // The unit box's corners land on the scaled and moved bounds' corners.
        assert!(close(m.transform_point([0.5, 0.5, 0.5]), [12.0, 8.0, 9.0]));
        assert!(close(
            m.transform_point([-0.5, -0.5, -0.5]),
            [8.0, 0.0, 1.0]
        ));
    }

    #[test]
    fn flat_meshes_get_a_thin_but_real_box() {
        let flat = Bounds {
            min: [-5.0, 0.0, -5.0],
            max: [5.0, 0.0, 5.0],
        };
        let m = pick_matrix(&Mat4::IDENTITY, &flat);
        let top = m.transform_point([0.0, 0.5, 0.0]);
        let bottom = m.transform_point([0.0, -0.5, 0.0]);
        assert!((top[1] - bottom[1] - MIN_THICKNESS).abs() < 1e-6);
    }

    #[test]
    fn shapes_are_stored_per_object() {
        let shapes = PickShapes::default();
        assert!(shapes.is_empty());
        assert!(shapes.get(3).is_empty());
        shapes.add(3, Mat4::IDENTITY);
        shapes.add(3, Mat4::IDENTITY);
        assert_eq!(shapes.get(3), vec![Mat4::IDENTITY; 2]);
        assert_eq!(shapes.len(), 1);
    }
}
