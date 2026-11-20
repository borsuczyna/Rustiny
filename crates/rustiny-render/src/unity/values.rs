//! Filling constant buffers of Unity shaders by member name.
//!
//! A pass's reflection says where `_BaseColor`, `unity_ObjectToWorld`, `_MainLightColor`, ... sit
//! in their constant buffers; the values come from the material, from the frame (camera, time,
//! light) or from the draw (model matrix). Members nothing provides stay zero.

use super::UnityFrame;
use rustiny_shader::{Member, Scalar, Shape, UniformBlock};
use std::collections::HashMap;

fn put(dest: &mut [u8], at: usize, kind: Scalar, value: f32) {
    let bytes = match kind {
        Scalar::Float => value.to_ne_bytes(),
        Scalar::Int => (value as i32).to_ne_bytes(),
        Scalar::Uint => (value as u32).to_ne_bytes(),
        Scalar::Bool => u32::from(value != 0.0).to_ne_bytes(),
    };
    if let Some(slot) = dest.get_mut(at..at + 4) {
        slot.copy_from_slice(&bytes);
    }
}

/// Write `values` into the bytes of `member` inside `out` (the block's bytes). A scalar takes the
/// first value, a vector as many as it has, a matrix needs all 16 (column-major). Arrays and
/// structures are left alone.
pub(crate) fn write_member(member: &Member, values: &[f32], out: &mut [u8]) {
    let start = member.offset as usize;
    let Some(dest) = out.get_mut(start..start + member.size as usize) else {
        return;
    };
    match &member.shape {
        Shape::Scalar(kind) => put(dest, 0, *kind, values.first().copied().unwrap_or(0.0)),
        Shape::Vector(kind, count) => {
            for i in 0..*count as usize {
                put(dest, i * 4, *kind, values.get(i).copied().unwrap_or(0.0));
            }
        }
        Shape::Matrix(columns, rows) if values.len() == 16 => {
            let stride = dest.len() / (*columns as usize).max(1);
            for column in 0..*columns as usize {
                for row in 0..(*rows as usize).min(4) {
                    put(
                        dest,
                        column * stride + row * 4,
                        Scalar::Float,
                        values[column.min(3) * 4 + row],
                    );
                }
            }
        }
        _ => {}
    }
}

/// Fill the members of `block` that `lookup` knows.
pub(crate) fn fill_block<'a>(
    block: &UniformBlock,
    out: &mut [u8],
    mut lookup: impl FnMut(&str) -> Option<&'a [f32]>,
) {
    for member in &block.members {
        if let Some(values) = lookup(&member.name) {
            write_member(member, values, out);
        }
    }
}

/// Sign of the determinant of the upper 3x3 of a column-major matrix: -1 for a mirrored transform.
pub(crate) fn handedness(m: &[f32; 16]) -> f32 {
    let det = m[0] * (m[5] * m[10] - m[9] * m[6]) - m[4] * (m[1] * m[10] - m[9] * m[2])
        + m[8] * (m[1] * m[6] - m[5] * m[2]);
    if det < 0.0 { -1.0 } else { 1.0 }
}

/// The values Unity shaders expect to find ready: camera, time, main light, ambient, probes.
#[derive(Debug, Default)]
pub(crate) struct Constants {
    map: HashMap<&'static str, Vec<f32>>,
}

impl Constants {
    pub(crate) fn new(frame: &UnityFrame, view_proj: &[f32; 16]) -> Self {
        use crate::matrix::{invert, multiply};
        let mut map: HashMap<&'static str, Vec<f32>> = HashMap::new();
        let mut set = |name: &'static str, values: &[f32]| {
            map.insert(name, values.to_vec());
        };
        let identity = {
            let mut m = [0.0f32; 16];
            for i in 0..4 {
                m[i * 5] = 1.0;
            }
            m
        };
        let inverse = |m: &[f32; 16]| invert(m).unwrap_or(identity);

        set("unity_MatrixV", &frame.view);
        set("unity_WorldToCamera", &frame.view);
        set("unity_MatrixInvV", &inverse(&frame.view));
        set("unity_CameraToWorld", &inverse(&frame.view));
        set("glstate_matrix_projection", &frame.projection);
        set("unity_CameraProjection", &frame.projection);
        // URP turns a screen position back into a view or world position with
        // `ComputeClipSpacePosition`, which on Vulkan (`UNITY_UV_STARTS_AT_TOP`) negates y: Unity's
        // own render targets are stored upside down. Ours are not (texture row 0 is the top of
        // the view, as `ComputeScreenPos` with `_ProjectionParams.x = 1` expects), so the inverse
        // matrices undo that negation first.
        let flip_y = {
            let mut m = identity;
            m[5] = -1.0;
            m
        };
        set(
            "unity_MatrixInvP",
            &inverse(&multiply(&flip_y, &frame.projection)),
        );
        set("unity_CameraInvProjection", &inverse(&frame.projection));
        set("unity_MatrixVP", view_proj);
        set("unity_MatrixInvVP", &inverse(&multiply(&flip_y, view_proj)));
        // Kept for shaders that build the product themselves.
        set(
            "unity_MatrixVPFromParts",
            &multiply(&frame.projection, &frame.view),
        );

        let [x, y, z] = frame.eye;
        set("_WorldSpaceCameraPos", &[x, y, z, 0.0]);
        let (w, h) = (frame.width.max(1) as f32, frame.height.max(1) as f32);
        set("_ScreenParams", &[w, h, 1.0 + 1.0 / w, 1.0 + 1.0 / h]);
        set("_ScaledScreenParams", &[w, h, 1.0 + 1.0 / w, 1.0 + 1.0 / h]);
        let (near, far) = (frame.near, frame.far);
        set("_ProjectionParams", &[1.0, near, far, 1.0 / far]);
        // The projection is reverse-Z with an infinite far plane: depth = near / eye depth, so
        // LinearEyeDepth (1 / (z * depth + w)) is exact with z = 1 / near, w = 0, and
        // Linear01Depth (1 / (x * depth + y)) is eye depth over `far` with x = far / near, y = 0.
        set("_ZBufferParams", &[far / near, 0.0, 1.0 / near, 0.0]);
        // w is 1 for an orthographic camera; this one is a perspective camera.
        set("unity_OrthoParams", &[1.0, 1.0, 0.0, 0.0]);
        set("_GlobalMipBias", &[0.0, 1.0, 0.0, 0.0]);

        let t = frame.time;
        set("_Time", &[t / 20.0, t, t * 2.0, t * 3.0]);
        set(
            "_SinTime",
            &[(t / 8.0).sin(), (t / 4.0).sin(), (t / 2.0).sin(), t.sin()],
        );
        set(
            "_CosTime",
            &[(t / 8.0).cos(), (t / 4.0).cos(), (t / 2.0).cos(), t.cos()],
        );
        set("_TimeParameters", &[t, t.sin(), t.cos(), 0.0]);
        set("unity_DeltaTime", &[0.016, 62.5, 0.016, 62.5]);

        let [lx, ly, lz] = frame.light_direction;
        let [r, g, b] = frame.light_color;
        set("_MainLightPosition", &[lx, ly, lz, 0.0]);
        set("_MainLightColor", &[r, g, b, 1.0]);
        set("_MainLightLayerMask", &[u32::MAX as f32]);
        set("_WorldSpaceLightPos0", &[lx, ly, lz, 0.0]);
        set("_LightColor0", &[r, g, b, 1.0]);
        set("_AdditionalLightsCount", &[0.0, 0.0, 0.0, 0.0]);
        // z is 1 when the light is not culled by the object's mask: URP reads it as the main light's distance attenuation.
        set("unity_LightData", &[0.0, 0.0, 1.0, 0.0]);

        let ambient = &frame.ambient;
        let sh = ambient.spherical_harmonics();
        for (name, values) in [
            "unity_SHAr",
            "unity_SHAg",
            "unity_SHAb",
            "unity_SHBr",
            "unity_SHBg",
            "unity_SHBb",
            "unity_SHC",
        ]
        .into_iter()
        .zip(sh)
        {
            set(name, &values);
        }
        let rgba = |[r, g, b]: [f32; 3]| [r, g, b, 1.0];
        set("unity_AmbientSky", &rgba(ambient.sky));
        set("unity_AmbientEquator", &rgba(ambient.equator));
        set("unity_AmbientGround", &rgba(ambient.ground));
        // URP: the ambient probe's constant term times the reflection intensity, used where no
        // reflection probe is sampled.
        let i = frame.reflection_intensity;
        set(
            "_GlossyEnvironmentColor",
            &[sh[0][3] * i, sh[1][3] * i, sh[2][3] * i, 1.0],
        );
        // Decode instructions of the environment cubemap: x scales it (the reflection intensity),
        // w = 0 because the alpha channel holds no HDR multiplier.
        set("unity_SpecCube0_HDR", &[i, 1.0, 0.0, 0.0]);
        set("unity_SpecCube1_HDR", &[i, 1.0, 0.0, 0.0]);
        set("unity_RenderingLayer", &[u32::MAX as f32, 0.0, 0.0, 0.0]);
        set("unity_LightmapST", &[1.0, 1.0, 0.0, 0.0]);
        set("unity_DynamicLightmapST", &[1.0, 1.0, 0.0, 0.0]);
        Self { map }
    }

    /// The value called `name`, if the frame provides one.
    pub(crate) fn get(&self, name: &str) -> Option<&[f32]> {
        self.map.get(name).map(Vec::as_slice)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn block(members: Vec<Member>, size: u32) -> UniformBlock {
        UniformBlock {
            name: "B".into(),
            set: 0,
            binding: 0,
            size,
            members,
        }
    }

    fn member(name: &str, offset: u32, size: u32, shape: Shape) -> Member {
        Member {
            name: name.into(),
            offset,
            size,
            shape,
        }
    }

    fn floats(bytes: &[u8]) -> Vec<f32> {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_ne_bytes(*c))
            .collect()
    }

    #[test]
    fn members_are_written_at_their_offsets_by_shape() {
        let b = block(
            vec![
                member("_Color", 0, 16, Shape::Vector(Scalar::Float, 4)),
                member("_Cutoff", 16, 4, Shape::Scalar(Scalar::Float)),
                member("_Mode", 20, 4, Shape::Scalar(Scalar::Int)),
                member("_Pos", 32, 12, Shape::Vector(Scalar::Float, 3)),
                member("_M", 48, 64, Shape::Matrix(4, 4)),
                member("_Skipped", 112, 16, Shape::Vector(Scalar::Float, 4)),
            ],
            128,
        );
        let matrix: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut out = vec![0u8; 128];
        fill_block(&b, &mut out, |name| match name {
            "_Color" => Some(&[1.0, 0.5, 0.25, 1.0][..]),
            "_Cutoff" => Some(&[0.3][..]),
            "_Mode" => Some(&[2.0][..]),
            "_Pos" => Some(&[7.0, 8.0, 9.0, 10.0][..]),
            "_M" => Some(&matrix[..]),
            _ => None,
        });
        let f = floats(&out);
        assert_eq!(&f[0..4], &[1.0, 0.5, 0.25, 1.0]);
        assert_eq!(f[4], 0.3);
        assert_eq!(i32::from_ne_bytes(out[20..24].try_into().unwrap()), 2);
        assert_eq!(&f[8..11], &[7.0, 8.0, 9.0], "a float3 takes 3 values");
        assert_eq!(f[11], 0.0, "the padding after a float3 is untouched");
        assert_eq!(
            &f[12..28],
            &matrix[..],
            "the matrix keeps its column-major order"
        );
        assert_eq!(&f[28..32], &[0.0; 4], "unknown members stay zero");
    }

    #[test]
    fn writes_past_the_block_are_ignored() {
        let m = member("_X", 28, 16, Shape::Vector(Scalar::Float, 4));
        let mut out = vec![0u8; 32];
        write_member(&m, &[1.0; 4], &mut out);
        assert_eq!(out, vec![0u8; 32]);
    }

    #[test]
    fn a_mirrored_transform_has_negative_handedness() {
        let mut m = [0.0f32; 16];
        m[0] = 1.0;
        m[5] = 1.0;
        m[10] = 1.0;
        m[15] = 1.0;
        assert_eq!(handedness(&m), 1.0);
        m[0] = -1.0;
        assert_eq!(handedness(&m), -1.0);
    }

    /// URP's SampleSH: L0 + L1 + L2.
    fn sample_sh(sh: &[[f32; 4]; 7], n: [f32; 3]) -> [f32; 3] {
        let [x, y, z] = n;
        let mut out = [0.0; 3];
        for (c, value) in out.iter_mut().enumerate() {
            let a = sh[c];
            let b = sh[3 + c];
            let l01 = a[0] * x + a[1] * y + a[2] * z + a[3];
            let vb = [x * y, y * z, z * z, z * x];
            let l2 = b.iter().zip(vb).map(|(k, v)| k * v).sum::<f32>() + sh[6][c] * (x * x - y * y);
            *value = l01 + l2;
        }
        out
    }

    #[test]
    fn trilight_ambient_gives_sky_up_equator_sideways_and_ground_down() {
        use super::super::Ambient;
        let ambient = Ambient {
            sky: [0.4, 0.5, 0.9],
            equator: [0.3, 0.3, 0.3],
            ground: [0.1, 0.05, 0.0],
        };
        let sh = ambient.spherical_harmonics();
        let close = |a: [f32; 3], b: [f32; 3]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5);
        assert!(close(sample_sh(&sh, [0.0, 1.0, 0.0]), ambient.sky));
        assert!(close(sample_sh(&sh, [0.0, -1.0, 0.0]), ambient.ground));
        assert!(close(sample_sh(&sh, [1.0, 0.0, 0.0]), ambient.equator));
        assert!(close(sample_sh(&sh, [0.0, 0.0, 1.0]), ambient.equator));
        let s = std::f32::consts::FRAC_1_SQRT_2;
        assert!(
            close(sample_sh(&sh, [s, 0.0, s]), ambient.equator),
            "every horizontal direction is the equator colour"
        );
        let flat = Ambient::flat([0.2, 0.3, 0.4]).spherical_harmonics();
        assert!(close(sample_sh(&flat, [0.3, 0.8, -0.52]), [0.2, 0.3, 0.4]));
    }

    #[test]
    fn the_frame_provides_camera_light_and_time() {
        let mut view_proj = [0.0f32; 16];
        view_proj[0] = 2.0;
        let frame = UnityFrame {
            eye: [1.0, 2.0, 3.0],
            width: 200,
            height: 100,
            time: 4.0,
            light_direction: [0.0, 1.0, 0.0],
            ..UnityFrame::default()
        };
        let c = Constants::new(&frame, &view_proj);
        assert_eq!(
            c.get("_WorldSpaceCameraPos"),
            Some(&[1.0, 2.0, 3.0, 0.0][..])
        );
        assert_eq!(c.get("_ScreenParams").unwrap()[..2], [200.0, 100.0]);
        assert_eq!(c.get("_Time").unwrap()[1], 4.0);
        assert_eq!(c.get("_MainLightPosition").unwrap()[..3], [0.0, 1.0, 0.0]);
        assert_eq!(c.get("unity_MatrixVP").unwrap()[0], 2.0);
        assert!(c.get("_NotAUnityValue").is_none());
    }
}
