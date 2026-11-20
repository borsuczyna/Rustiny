//! Small `f64` matrix helpers for composing FBX node transforms.
//!
//! Column-major, column vectors: `world = parent * local`.

/// A 4x4 matrix, column-major: element `(row, col)` is at `col * 4 + row`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat(pub [f64; 16]);

impl Mat {
    pub const IDENTITY: Self = Self([
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]);

    pub fn translation(t: [f64; 3]) -> Self {
        let mut m = Self::IDENTITY;
        m.0[12..15].copy_from_slice(&t);
        m
    }

    pub fn scaling(s: [f64; 3]) -> Self {
        let mut m = Self::IDENTITY;
        m.0[0] = s[0];
        m.0[5] = s[1];
        m.0[10] = s[2];
        m
    }

    /// Rotation about one axis (0 = X, 1 = Y, 2 = Z) by `radians`, right-handed.
    fn axis_rotation(axis: usize, radians: f64) -> Self {
        let (s, c) = radians.sin_cos();
        let mut m = Self::IDENTITY;
        let (i, j) = match axis {
            0 => (1, 2),
            1 => (2, 0),
            _ => (0, 1),
        };
        // Rotate in the (i, j) plane: i -> j.
        m.0[i * 4 + i] = c;
        m.0[j * 4 + j] = c;
        m.0[i * 4 + j] = s;
        m.0[j * 4 + i] = -s;
        m
    }

    /// Rotation from Euler angles in degrees. `order` lists the axes in the order they are
    /// applied (`[0, 1, 2]` = X first, then Y, then Z, which is FBX's default `XYZ`).
    pub fn euler_degrees(angles: [f64; 3], order: [usize; 3]) -> Self {
        let mut m = Self::IDENTITY;
        for axis in order {
            m = Self::axis_rotation(axis, angles[axis].to_radians()).mul(&m);
        }
        m
    }

    /// `self * rhs`: applies `rhs` first.
    pub fn mul(&self, rhs: &Self) -> Self {
        let mut out = [0.0; 16];
        for col in 0..4 {
            for row in 0..4 {
                out[col * 4 + row] = (0..4)
                    .map(|k| self.0[k * 4 + row] * rhs.0[col * 4 + k])
                    .sum();
            }
        }
        Self(out)
    }

    /// Transpose of the inverse of the upper-left 3x3 (the matrix that transforms normals), or
    /// the identity when the 3x3 is singular. Returned row-major as `[[f64; 3]; 3]` acting on
    /// column vectors.
    pub fn normal_matrix(&self) -> [[f64; 3]; 3] {
        let a = |r: usize, c: usize| self.0[c * 4 + r];
        let cof = |r0: usize, c0: usize, r1: usize, c1: usize| {
            a(r0, c0) * a(r1, c1) - a(r0, c1) * a(r1, c0)
        };
        // Cofactor matrix; the inverse transpose is the cofactor matrix over the determinant.
        let c = [
            [cof(1, 1, 2, 2), -cof(1, 0, 2, 2), cof(1, 0, 2, 1)],
            [-cof(0, 1, 2, 2), cof(0, 0, 2, 2), -cof(0, 0, 2, 1)],
            [cof(0, 1, 1, 2), -cof(0, 0, 1, 2), cof(0, 0, 1, 1)],
        ];
        let det = a(0, 0) * c[0][0] + a(0, 1) * c[0][1] + a(0, 2) * c[0][2];
        if det.abs() < 1e-18 || !det.is_finite() {
            return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        }
        let mut out = c;
        for row in &mut out {
            for v in row {
                *v /= det;
            }
        }
        out
    }

    /// Transpose (the inverse of a pure rotation).
    pub fn transposed(&self) -> Self {
        let mut out = [0.0; 16];
        for col in 0..4 {
            for row in 0..4 {
                out[row * 4 + col] = self.0[col * 4 + row];
            }
        }
        Self(out)
    }

    /// Determinant of the upper-left 3x3; negative for a mirroring transform.
    pub fn det3(&self) -> f64 {
        let m = &self.0;
        m[0] * (m[5] * m[10] - m[9] * m[6]) - m[4] * (m[1] * m[10] - m[9] * m[2])
            + m[8] * (m[1] * m[6] - m[5] * m[2])
    }

    /// Transform a point (w = 1).
    pub fn point(&self, p: [f64; 3]) -> [f64; 3] {
        let m = &self.0;
        [
            m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
            m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
            m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
        ]
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn close(a: [f64; 3], b: [f64; 3]) -> bool {
        a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-9)
    }

    #[test]
    fn rotations_are_right_handed() {
        let rz = Mat::euler_degrees([0.0, 0.0, 90.0], [0, 1, 2]);
        assert!(close(rz.point([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0]));
        let rx = Mat::euler_degrees([90.0, 0.0, 0.0], [0, 1, 2]);
        assert!(close(rx.point([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0]));
        let ry = Mat::euler_degrees([0.0, 90.0, 0.0], [0, 1, 2]);
        assert!(close(ry.point([0.0, 0.0, 1.0]), [1.0, 0.0, 0.0]));
    }

    #[test]
    fn euler_order_applies_the_first_axis_first() {
        // X then Y: (0,1,0) -> X90 -> (0,0,1) -> Y90 -> (1,0,0).
        let xy = Mat::euler_degrees([90.0, 90.0, 0.0], [0, 1, 2]);
        assert!(close(xy.point([0.0, 1.0, 0.0]), [1.0, 0.0, 0.0]));
        // Y then X: (0,1,0) -> Y90 -> (0,1,0) -> X90 -> (0,0,1).
        let yx = Mat::euler_degrees([90.0, 90.0, 0.0], [1, 0, 2]);
        assert!(close(yx.point([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0]));
    }

    #[test]
    fn multiplication_applies_the_right_hand_side_first() {
        let m = Mat::translation([1.0, 0.0, 0.0]).mul(&Mat::scaling([2.0, 2.0, 2.0]));
        assert!(close(m.point([1.0, 1.0, 1.0]), [3.0, 2.0, 2.0]));
    }

    #[test]
    fn the_normal_matrix_undoes_non_uniform_scale() {
        // A plane tilted 45 degrees stretched along X: its normal must lean away from X.
        let n = Mat::scaling([4.0, 1.0, 1.0]).normal_matrix();
        let v = [1.0, 1.0, 0.0];
        let out = [
            n[0][0] * v[0] + n[0][1] * v[1] + n[0][2] * v[2],
            n[1][0] * v[0] + n[1][1] * v[1] + n[1][2] * v[2],
        ];
        assert!((out[0] - 0.25).abs() < 1e-9 && (out[1] - 1.0).abs() < 1e-9);
        let singular = Mat::scaling([0.0, 1.0, 1.0]).normal_matrix();
        assert_eq!(singular[0][0], 1.0);
    }
}
