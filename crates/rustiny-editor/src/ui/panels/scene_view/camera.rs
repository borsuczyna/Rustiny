//! A free-fly editor camera and the projection of world points to the viewport.
//!
//! Unity's space is left-handed: +X right, +Y up, +Z forward. Yaw 0 looks along +Z, positive
//! yaw turns towards +X. Projection is a plain pinhole (no matrices needed for points and lines).

/// Distance of the near plane; points closer than this to the camera plane are not drawn.
pub const NEAR: f32 = 0.1;
const MAX_PITCH: f32 = 89.0_f32.to_radians();
const LOOK_SENSITIVITY: f32 = 0.0035;

/// Position and orientation of the viewport camera.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    /// World position.
    pub eye: [f32; 3],
    /// Rotation about +Y in radians (0 = looking along +Z).
    pub yaw: f32,
    /// Elevation in radians (positive = looking up).
    pub pitch: f32,
    /// Vertical field of view in radians.
    pub fov_y: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            eye: [0.0, 5.0, -10.0],
            yaw: 0.0,
            pitch: -0.3,
            fov_y: 60.0_f32.to_radians(),
        }
    }
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

impl Camera {
    /// Unit vector the camera looks along.
    pub fn forward(&self) -> [f32; 3] {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        [cp * sy, sp, cp * cy]
    }

    /// Unit vector to the camera's right (always horizontal).
    pub fn right(&self) -> [f32; 3] {
        let (sy, cy) = self.yaw.sin_cos();
        [cy, 0.0, -sy]
    }

    /// Unit vector up from the camera's point of view.
    pub fn up(&self) -> [f32; 3] {
        let f = self.forward();
        let r = self.right();
        // forward x right, which is `up` in a left-handed system.
        [
            f[1] * r[2] - f[2] * r[1],
            f[2] * r[0] - f[0] * r[2],
            f[0] * r[1] - f[1] * r[0],
        ]
    }

    /// Turn the camera by a mouse movement in pixels.
    pub fn look(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * LOOK_SENSITIVITY;
        self.pitch = (self.pitch - dy * LOOK_SENSITIVITY).clamp(-MAX_PITCH, MAX_PITCH);
        self.yaw = self.yaw.rem_euclid(std::f32::consts::TAU);
    }

    /// Move along the camera's forward, right and (world) up axes by `amount` each, in world units.
    pub fn fly(&mut self, forward: f32, right: f32, up: f32) {
        let f = self.forward();
        let r = self.right();
        for i in 0..3 {
            let world_up = if i == 1 { 1.0 } else { 0.0 };
            self.eye[i] += f[i] * forward + r[i] * right + world_up * up;
        }
    }

    /// Place the camera so a sphere of `radius` around `center` fills the view, keeping the
    /// current viewing direction.
    pub fn frame(&mut self, center: [f32; 3], radius: f32) {
        let distance = radius.max(0.5) / (self.fov_y * 0.5).tan() * 1.2;
        let f = self.forward();
        for i in 0..3 {
            self.eye[i] = center[i] - f[i] * distance;
        }
    }

    /// Column-major view-projection for the GPU scene pass (Vulkan clip space, reverse-Z,
    /// infinite far plane); `aspect` is viewport width / height.
    pub fn view_projection(&self, aspect: f32) -> [f32; 16] {
        rustiny_render::matrix::view_projection(
            self.eye,
            self.right(),
            self.up(),
            self.forward(),
            self.fov_y,
            aspect,
            NEAR,
        )
    }

    /// World-to-view matrix in Unity's convention (for the project's shaders).
    pub fn view_matrix(&self) -> [f32; 16] {
        rustiny_render::matrix::view_matrix(self.eye, self.right(), self.up(), self.forward())
    }

    /// The projection that goes with [`view_matrix`](Self::view_matrix).
    pub fn projection(&self, aspect: f32) -> [f32; 16] {
        rustiny_render::matrix::projection(self.fov_y, aspect, NEAR)
    }

    fn view_of(&self, p: [f32; 3]) -> [f32; 3] {
        let v = sub(p, self.eye);
        [
            dot(v, self.right()),
            dot(v, self.up()),
            dot(v, self.forward()),
        ]
    }

    fn focal(&self, size: [f32; 2]) -> f32 {
        size[1] * 0.5 / (self.fov_y * 0.5).tan()
    }

    fn view_to_screen(&self, v: [f32; 3], size: [f32; 2]) -> [f32; 2] {
        let f = self.focal(size);
        [
            size[0] * 0.5 + v[0] / v[2] * f,
            size[1] * 0.5 - v[1] / v[2] * f,
        ]
    }

    /// Project a world point to viewport pixels (origin top-left). Returns `(x, y, depth)` with
    /// `depth` the distance along the view direction, or `None` when the point is behind the
    /// near plane or not finite.
    pub fn project(&self, p: [f32; 3], size: [f32; 2]) -> Option<[f32; 3]> {
        let v = self.view_of(p);
        if !v.iter().all(|c| c.is_finite()) || v[2] < NEAR {
            return None;
        }
        let s = self.view_to_screen(v, size);
        Some([s[0], s[1], v[2]])
    }

    /// Project a world-space line segment, clipped against the near plane.
    pub fn project_segment(
        &self,
        a: [f32; 3],
        b: [f32; 3],
        size: [f32; 2],
    ) -> Option<([f32; 2], [f32; 2])> {
        let (mut va, mut vb) = (self.view_of(a), self.view_of(b));
        if !va.iter().chain(vb.iter()).all(|c| c.is_finite()) {
            return None;
        }
        if va[2] < NEAR && vb[2] < NEAR {
            return None;
        }
        // Move the endpoint that is behind the near plane onto it.
        let clip = |inside: [f32; 3], outside: &mut [f32; 3]| {
            let t = (NEAR - outside[2]) / (inside[2] - outside[2]);
            for i in 0..3 {
                outside[i] += (inside[i] - outside[i]) * t;
            }
            outside[2] = NEAR;
        };
        if va[2] < NEAR {
            clip(vb, &mut va);
        } else if vb[2] < NEAR {
            clip(va, &mut vb);
        }
        Some((self.view_to_screen(va, size), self.view_to_screen(vb, size)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIZE: [f32; 2] = [800.0, 600.0];

    fn level() -> Camera {
        Camera {
            eye: [0.0; 3],
            yaw: 0.0,
            pitch: 0.0,
            fov_y: 90.0_f32.to_radians(),
        }
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-3
    }

    #[test]
    fn basis_is_orthonormal_and_left_handed() {
        let c = Camera {
            yaw: 0.7,
            pitch: 0.4,
            ..level()
        };
        let (f, r, u) = (c.forward(), c.right(), c.up());
        for v in [f, r, u] {
            assert!(close(dot(v, v), 1.0));
        }
        assert!(close(dot(f, r), 0.0) && close(dot(f, u), 0.0) && close(dot(r, u), 0.0));
        let flat = level();
        assert_eq!(flat.forward(), [0.0, 0.0, 1.0]);
        assert_eq!(flat.right(), [1.0, 0.0, 0.0]);
        assert!(close(flat.up()[1], 1.0));
    }

    #[test]
    fn a_point_straight_ahead_is_in_the_centre() {
        let p = level().project([0.0, 0.0, 10.0], SIZE).unwrap();
        assert!(close(p[0], 400.0) && close(p[1], 300.0) && close(p[2], 10.0));
    }

    #[test]
    fn right_is_right_and_up_is_up_on_screen() {
        let c = level();
        // 90 degree vertical FOV: a point at (z, z) up/right offset lands at the edge.
        let right = c.project([10.0, 0.0, 10.0], SIZE).unwrap();
        assert!(right[0] > 400.0 && close(right[1], 300.0));
        let up = c.project([0.0, 10.0, 10.0], SIZE).unwrap();
        assert!(up[1] < 300.0, "screen y grows downwards");
        assert!(
            close(up[1], 0.0),
            "half the FOV at distance 10 and height 10"
        );
    }

    #[test]
    fn yaw_turns_towards_plus_x() {
        let c = Camera {
            yaw: std::f32::consts::FRAC_PI_2,
            ..level()
        };
        let p = c.project([10.0, 0.0, 0.0], SIZE).unwrap();
        assert!(close(p[0], 400.0), "+X is now straight ahead: {p:?}");
    }

    #[test]
    fn points_behind_or_invalid_are_rejected() {
        let c = level();
        assert!(c.project([0.0, 0.0, -5.0], SIZE).is_none());
        assert!(c.project([0.0, 0.0, 0.05], SIZE).is_none());
        assert!(c.project([f32::NAN, 0.0, 5.0], SIZE).is_none());
        assert!(c.project([0.0, f32::INFINITY, 5.0], SIZE).is_none());
    }

    #[test]
    fn segments_are_clipped_at_the_near_plane() {
        let c = level();
        assert!(
            c.project_segment([0.0, 0.0, -1.0], [0.0, 0.0, -9.0], SIZE)
                .is_none()
        );
        let both = c
            .project_segment([-1.0, 0.0, 5.0], [1.0, 0.0, 5.0], SIZE)
            .unwrap();
        assert!(both.0[0] < both.1[0]);
        // One end behind the camera: the visible part is still returned and finite.
        let (a, b) = c
            .project_segment([0.0, 0.0, 10.0], [0.0, 0.0, -10.0], SIZE)
            .unwrap();
        assert!(a.iter().chain(b.iter()).all(|v| v.is_finite()));
        assert!(close(a[0], 400.0) && close(b[0], 400.0));
        // Vertical line near the camera plane stays bounded by the near-plane projection.
        let (_, near) = c
            .project_segment([2.0, 0.0, 10.0], [2.0, 0.0, -10.0], SIZE)
            .unwrap();
        assert!(near[0] > 400.0);
    }

    #[test]
    fn the_gpu_matrix_agrees_with_the_pinhole_projection() {
        // The CPU projection (picking, overlays) and the GPU matrix (the image) must put every
        // point on the same pixel, or clicks would not land on what is drawn.
        let cam = Camera {
            eye: [3.0, 7.0, -11.0],
            yaw: 0.8,
            pitch: -0.45,
            fov_y: 60.0_f32.to_radians(),
        };
        let matrix = cam.view_projection(SIZE[0] / SIZE[1]);
        let points = [
            [3.0, 7.0, 0.0],
            [10.0, 0.0, 10.0],
            [-4.0, 2.0, 30.0],
            [25.0, -3.0, 5.0],
            [0.0, 0.0, 0.0],
        ];
        let mut compared = 0;
        for p in points {
            let cpu = cam.project(p, SIZE);
            let gpu = rustiny_render::matrix::project(&matrix, p);
            let (Some(cpu), Some(gpu)) = (cpu, gpu) else {
                assert_eq!(cpu.is_some(), gpu.is_some(), "visibility of {p:?} differs");
                continue;
            };
            let px = [
                (gpu[0] + 1.0) * 0.5 * SIZE[0],
                (gpu[1] + 1.0) * 0.5 * SIZE[1],
            ];
            assert!(
                (px[0] - cpu[0]).abs() < 0.05 && (px[1] - cpu[1]).abs() < 0.05,
                "{p:?}: gpu {px:?} vs cpu {cpu:?}"
            );
            assert!(
                (gpu[2] - NEAR / cpu[2]).abs() < 1e-5,
                "reverse-Z depth is near / view depth"
            );
            compared += 1;
        }
        assert!(compared >= 3, "too few visible points to compare");
    }

    #[test]
    fn look_clamps_pitch_and_wraps_yaw() {
        let mut c = level();
        c.look(0.0, -1.0e6);
        assert!(close(c.pitch, MAX_PITCH));
        c.look(0.0, 1.0e6);
        assert!(close(c.pitch, -MAX_PITCH));
        c.look(1.0e6, 0.0);
        assert!((0.0..std::f32::consts::TAU).contains(&c.yaw));
    }

    #[test]
    fn fly_moves_along_the_view_and_world_up() {
        let mut c = level();
        c.fly(5.0, 2.0, 1.0);
        assert!(close(c.eye[0], 2.0) && close(c.eye[1], 1.0) && close(c.eye[2], 5.0));
        // Looking up, forward motion climbs but "up" stays the world axis.
        let mut c = Camera {
            pitch: 0.5,
            ..level()
        };
        c.fly(1.0, 0.0, 0.0);
        assert!(c.eye[1] > 0.4);
    }

    #[test]
    fn framing_puts_the_sphere_in_front_of_the_camera() {
        let mut c = level();
        c.frame([100.0, 0.0, 200.0], 50.0);
        let p = c.project([100.0, 0.0, 200.0], SIZE).unwrap();
        assert!(close(p[0], 400.0) && close(p[1], 300.0));
        let edge = c.project([150.0, 0.0, 200.0], SIZE).unwrap();
        assert!(
            edge[0] < 800.0,
            "the whole radius fits horizontally: {edge:?}"
        );
    }
}
