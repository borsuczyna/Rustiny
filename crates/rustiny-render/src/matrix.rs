//! The view-projection matrix for the scene shaders.
//!
//! Matches the shaders' conventions: left-handed world (+Y up, +Z forward), column vectors,
//! column-major storage, Vulkan clip space (Y points down), and a reverse-Z infinite-far depth
//! range (depth 1 at the near plane, approaching 0 with distance, so the depth test is GREATER).

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// View-projection from a camera: its position, orthonormal basis (right, up, forward in world
/// space), vertical field of view in radians, aspect ratio (width / height) and near distance.
pub fn view_projection(
    eye: [f32; 3],
    right: [f32; 3],
    up: [f32; 3],
    forward: [f32; 3],
    fov_y: f32,
    aspect: f32,
    near: f32,
) -> [f32; 16] {
    let focal = 1.0 / (fov_y * 0.5).tan();
    let (fx, fy) = (focal / aspect, -focal);
    let mut m = [0.0; 16];
    let mut set_row = |row: usize, values: [f32; 4]| {
        for (col, value) in values.iter().enumerate() {
            m[col * 4 + row] = *value;
        }
    };
    set_row(
        0,
        [
            fx * right[0],
            fx * right[1],
            fx * right[2],
            -fx * dot(right, eye),
        ],
    );
    set_row(1, [fy * up[0], fy * up[1], fy * up[2], -fy * dot(up, eye)]);
    // z_clip = near and w = view-space depth, so ndc z = near / depth (reverse-Z, infinite far).
    set_row(2, [0.0, 0.0, 0.0, near]);
    set_row(3, [forward[0], forward[1], forward[2], -dot(forward, eye)]);
    m
}

/// The world-to-view matrix in Unity's convention (the camera looks down -Z of view space), for
/// shaders that read `unity_MatrixV`. Column-major.
pub fn view_matrix(eye: [f32; 3], right: [f32; 3], up: [f32; 3], forward: [f32; 3]) -> [f32; 16] {
    let back = [-forward[0], -forward[1], -forward[2]];
    let mut m = [0.0; 16];
    for (row, axis) in [right, up, back].iter().enumerate() {
        for col in 0..3 {
            m[col * 4 + row] = axis[col];
        }
        m[12 + row] = -dot(*axis, eye);
    }
    m[15] = 1.0;
    m
}

/// The projection that goes with [`view_matrix`]: `projection * view_matrix` equals
/// [`view_projection`] of the same camera (Vulkan clip space, reverse-Z, infinite far plane).
/// Column-major.
pub fn projection(fov_y: f32, aspect: f32, near: f32) -> [f32; 16] {
    let focal = 1.0 / (fov_y * 0.5).tan();
    let mut m = [0.0; 16];
    m[0] = focal / aspect;
    m[5] = -focal;
    m[14] = near;
    m[11] = -1.0;
    m
}

/// `a * b` for column-major matrices.
pub fn multiply(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut out = [0.0; 16];
    for col in 0..4 {
        for row in 0..4 {
            out[col * 4 + row] = (0..4).map(|k| a[k * 4 + row] * b[col * 4 + k]).sum();
        }
    }
    out
}

/// The inverse of a column-major matrix, `None` when it is singular.
pub fn invert(m: &[f32; 16]) -> Option<[f32; 16]> {
    // Gauss-Jordan elimination with partial pivoting, in f64.
    let mut a = [[0.0f64; 8]; 4];
    for (row, line) in a.iter_mut().enumerate() {
        for col in 0..4 {
            line[col] = f64::from(m[col * 4 + row]);
        }
        line[4 + row] = 1.0;
    }
    for col in 0..4 {
        let pivot = (col..4).max_by(|&x, &y| a[x][col].abs().total_cmp(&a[y][col].abs()))?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        let scale = a[col][col];
        for value in &mut a[col] {
            *value /= scale;
        }
        let source = a[col];
        for (row, line) in a.iter_mut().enumerate() {
            if row != col {
                let factor = line[col];
                for (value, from) in line.iter_mut().zip(source) {
                    *value -= factor * from;
                }
            }
        }
    }
    let mut out = [0.0; 16];
    for row in 0..4 {
        for col in 0..4 {
            out[col * 4 + row] = a[row][4 + col] as f32;
        }
    }
    Some(out)
}

/// Apply a column-major matrix to a world point and divide by w. `None` when w is not positive
/// (the point is behind the camera).
pub fn project(m: &[f32; 16], p: [f32; 3]) -> Option<[f32; 3]> {
    let row = |r: usize| m[r] * p[0] + m[4 + r] * p[1] + m[8 + r] * p[2] + m[12 + r];
    let w = row(3);
    (w > 0.0).then(|| [row(0) / w, row(1) / w, row(2) / w])
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn level_camera(eye: [f32; 3]) -> [f32; 16] {
        view_projection(
            eye,
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            90.0_f32.to_radians(),
            2.0,
            0.1,
        )
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn a_point_straight_ahead_is_at_the_clip_centre() {
        let p = project(&level_camera([0.0; 3]), [0.0, 0.0, 10.0]).unwrap();
        assert!(close(p[0], 0.0) && close(p[1], 0.0));
        assert!(
            close(p[2], 0.1 / 10.0),
            "reverse-Z depth is near / distance: {p:?}"
        );
    }

    #[test]
    fn depth_is_one_at_the_near_plane_and_shrinks_with_distance() {
        let m = level_camera([0.0; 3]);
        let near = project(&m, [0.0, 0.0, 0.1]).unwrap()[2];
        let mid = project(&m, [0.0, 0.0, 5.0]).unwrap()[2];
        let far = project(&m, [0.0, 0.0, 5000.0]).unwrap()[2];
        assert!(close(near, 1.0));
        assert!(near > mid && mid > far && far > 0.0);
    }

    #[test]
    fn right_is_positive_x_and_up_is_negative_y_in_vulkan_clip_space() {
        let m = level_camera([0.0; 3]);
        assert!(project(&m, [5.0, 0.0, 5.0]).unwrap()[0] > 0.0);
        let up = project(&m, [0.0, 5.0, 5.0]).unwrap();
        assert!(up[1] < 0.0, "Vulkan NDC y grows downwards: {up:?}");
        // 90 degree vertical FOV: height 5 at distance 5 is the top edge; aspect 2 makes the
        // horizontal half-extent twice as wide.
        assert!(close(up[1], -1.0));
        assert!(close(project(&m, [10.0, 0.0, 5.0]).unwrap()[0], 1.0));
    }

    #[test]
    fn view_and_projection_multiply_to_the_view_projection() {
        let (eye, right, up) = ([1.0, 2.0, -3.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]);
        let forward = [-1.0, 0.0, 0.0];
        let fov = 60.0_f32.to_radians();
        let combined = view_projection(eye, right, up, forward, fov, 1.5, 0.1);
        let split = multiply(
            &projection(fov, 1.5, 0.1),
            &view_matrix(eye, right, up, forward),
        );
        for (a, b) in combined.iter().zip(&split) {
            assert!(close(*a, *b), "{combined:?} vs {split:?}");
        }
        // Unity's view space looks down -Z: a point ahead has a negative view-space z.
        let v = view_matrix(eye, right, up, forward);
        let ahead = [0.0, 2.0, -3.0];
        assert!(v[2] * ahead[0] + v[6] * ahead[1] + v[10] * ahead[2] + v[14] < 0.0);
    }

    #[test]
    fn invert_undoes_a_transform_and_rejects_singular_matrices() {
        let mut m = level_camera([3.0, -2.0, 7.0]);
        m[1] += 0.3;
        let inverse = invert(&m).unwrap();
        let product = multiply(&m, &inverse);
        for (i, v) in product.iter().enumerate() {
            let expected = if i % 5 == 0 { 1.0 } else { 0.0 };
            assert!((v - expected).abs() < 1e-3, "{product:?}");
        }
        assert!(invert(&[0.0; 16]).is_none());
    }

    #[test]
    fn moving_the_eye_moves_the_view() {
        let p = project(&level_camera([3.0, 0.0, 0.0]), [3.0, 0.0, 10.0]).unwrap();
        assert!(close(p[0], 0.0));
    }

    #[test]
    fn points_behind_the_camera_are_rejected() {
        assert!(project(&level_camera([0.0; 3]), [0.0, 0.0, -1.0]).is_none());
        assert!(project(&level_camera([0.0; 3]), [0.0, 0.0, 0.0]).is_none());
    }
}
