//! Minimal 4x4 matrix maths for world transforms.
//!
//! Column-major storage, column-vector convention (`world = parent * local`), the layout Unity
//! and most GPU APIs use. Unity's space is left-handed with +Y up and +Z forward; the matrices
//! here do not care about handedness, only the camera that consumes them does.

/// A 4x4 matrix, column-major: element `(row, col)` is at `col * 4 + row`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4(pub [f32; 16]);

impl Mat4 {
    /// The identity matrix.
    pub const IDENTITY: Self = Self([
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]);

    /// Translation * rotation * scale. `rotation` is a quaternion `[x, y, z, w]`; it is
    /// normalised here, and a zero or non-finite quaternion counts as "no rotation".
    pub fn from_trs(position: [f32; 3], rotation: [f32; 4], scale: [f32; 3]) -> Self {
        let [x, y, z, w] = normalised(rotation);
        let r = [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ];
        let mut m = [0.0; 16];
        for col in 0..3 {
            for row in 0..3 {
                m[col * 4 + row] = r[row][col] * scale[col];
            }
        }
        m[12..15].copy_from_slice(&position);
        m[15] = 1.0;
        Self(m)
    }

    /// `self * rhs`: applies `rhs` first, then `self`.
    pub fn mul(&self, rhs: &Self) -> Self {
        let mut out = [0.0; 16];
        for col in 0..4 {
            for row in 0..4 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += self.0[k * 4 + row] * rhs.0[col * 4 + k];
                }
                out[col * 4 + row] = sum;
            }
        }
        Self(out)
    }

    /// The translation part.
    pub fn translation(&self) -> [f32; 3] {
        [self.0[12], self.0[13], self.0[14]]
    }

    /// One of the three basis columns (the local X, Y or Z axis in the parent's space, scaled).
    pub fn axis(&self, index: usize) -> [f32; 3] {
        let i = index.min(2) * 4;
        [self.0[i], self.0[i + 1], self.0[i + 2]]
    }

    /// Transform a point (w = 1).
    pub fn transform_point(&self, p: [f32; 3]) -> [f32; 3] {
        let m = &self.0;
        [
            m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
            m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
            m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
        ]
    }
}

impl Default for Mat4 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

fn normalised(q: [f32; 4]) -> [f32; 4] {
    let len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if len.is_finite() && len > 1e-12 {
        [q[0] / len, q[1] / len, q[2] / len, q[3] / len]
    } else {
        [0.0, 0.0, 0.0, 1.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5)
    }

    const SQRT_HALF: f32 = std::f32::consts::FRAC_1_SQRT_2;

    #[test]
    fn identity_and_translation() {
        let t = Mat4::from_trs([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0], [1.0; 3]);
        assert_eq!(t.translation(), [1.0, 2.0, 3.0]);
        assert!(close(t.transform_point([1.0, 1.0, 1.0]), [2.0, 3.0, 4.0]));
        assert_eq!(Mat4::IDENTITY.mul(&t), t);
        assert_eq!(Mat4::default(), Mat4::IDENTITY);
    }

    #[test]
    fn quarter_turn_about_y_maps_forward_to_right() {
        // Unity (left-handed): +90 degrees about Y turns +Z into +X.
        let r = Mat4::from_trs([0.0; 3], [0.0, SQRT_HALF, 0.0, SQRT_HALF], [1.0; 3]);
        assert!(close(r.transform_point([0.0, 0.0, 1.0]), [1.0, 0.0, 0.0]));
        assert!(close(r.transform_point([1.0, 0.0, 0.0]), [0.0, 0.0, -1.0]));
        assert!(close(r.transform_point([0.0, 1.0, 0.0]), [0.0, 1.0, 0.0]));
    }

    #[test]
    fn scale_applies_before_rotation_and_translation() {
        let m = Mat4::from_trs(
            [10.0, 0.0, 0.0],
            [0.0, SQRT_HALF, 0.0, SQRT_HALF],
            [2.0, 1.0, 3.0],
        );
        // (0,0,1) -> scaled (0,0,3) -> rotated (3,0,0) -> moved (13,0,0)
        assert!(close(m.transform_point([0.0, 0.0, 1.0]), [13.0, 0.0, 0.0]));
        assert!(close(m.axis(0), [0.0, 0.0, -2.0]));
    }

    #[test]
    fn child_world_is_parent_times_local() {
        let parent = Mat4::from_trs([5.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [2.0; 3]);
        let child = Mat4::from_trs([1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0; 3]);
        let world = parent.mul(&child);
        assert!(close(world.translation(), [7.0, 0.0, 0.0]));
        assert!(close(world.axis(0), [2.0, 0.0, 0.0]), "scale is inherited");
    }

    #[test]
    fn degenerate_rotations_are_identity() {
        for q in [
            [0.0; 4],
            [f32::NAN, 0.0, 0.0, 1.0],
            [f32::INFINITY, 0.0, 0.0, 0.0],
        ] {
            let m = Mat4::from_trs([0.0; 3], q, [1.0; 3]);
            assert!(
                close(m.transform_point([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0]),
                "{q:?}"
            );
        }
        // Un-normalised input is normalised.
        let m = Mat4::from_trs(
            [0.0; 3],
            [0.0, 2.0 * SQRT_HALF, 0.0, 2.0 * SQRT_HALF],
            [1.0; 3],
        );
        assert!(close(m.transform_point([0.0, 0.0, 1.0]), [1.0, 0.0, 0.0]));
    }
}
