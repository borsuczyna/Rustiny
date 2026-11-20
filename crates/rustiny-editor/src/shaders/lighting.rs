//! The main light and ambient light of a scene, as URP hands them to shaders.
//!
//! * The main light is `RenderSettings.m_Sun`, else the first enabled directional light
//!   ([`Scene::sun`]). `_MainLightPosition` is the direction towards it (minus the light's
//!   forward axis); `_MainLightColor` is its colour converted to linear times its intensity (URP
//!   uses linear light intensities).
//! * The ambient light comes from `RenderSettings`: the gradient (trilight) colours or the flat
//!   colour, converted to linear and scaled by `m_AmbientIntensity`.

use rustiny_render::Ambient;
use rustiny_scene::Scene;
use rustiny_scene::lighting::{AmbientMode, RenderSettings};

/// `Mathf.GammaToLinearSpace` for one channel (the exact sRGB curve).
pub fn gamma_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else if c < 1.0 {
        ((c + 0.055) / 1.055).powf(2.4)
    } else {
        c.powf(2.2)
    }
}

fn linear_rgb(c: [f32; 4]) -> [f32; 3] {
    [
        gamma_to_linear(c[0]),
        gamma_to_linear(c[1]),
        gamma_to_linear(c[2]),
    ]
}

/// What the shaders need to know about a scene's lighting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneLightValues {
    /// Unit vector towards the main light.
    pub light_direction: [f32; 3],
    /// Linear colour times intensity; black without a main light.
    pub light_color: [f32; 3],
    /// Ambient light.
    pub ambient: Ambient,
    /// `m_ReflectionIntensity`.
    pub reflection_intensity: f32,
}

/// The ambient light `settings` define. The skybox mode needs the ambient probe Unity computes
/// from the sky; until that is computed here, the gradient colours stored next to it are used.
pub fn ambient_of(settings: &RenderSettings) -> Ambient {
    let scale = |c: [f32; 4]| linear_rgb(c).map(|v| v * settings.ambient_intensity);
    match settings.ambient_mode {
        AmbientMode::Flat => Ambient::flat(scale(settings.ambient_sky)),
        AmbientMode::Trilight | AmbientMode::Skybox => Ambient {
            sky: scale(settings.ambient_sky),
            equator: scale(settings.ambient_equator),
            ground: scale(settings.ambient_ground),
        },
    }
}

/// The lighting of `scene`. `None` when the scene has neither `RenderSettings` nor a sun, so the
/// renderer keeps its defaults.
pub fn scene_light_values(scene: &Scene) -> Option<SceneLightValues> {
    let settings = scene.lighting().settings.clone();
    let sun = scene.sun();
    if settings.is_none() && sun.is_none() {
        return None;
    }
    let settings = settings.unwrap_or_default();
    let (light_direction, light_color) = match sun {
        Some(sun) => {
            let forward = sun
                .object
                .and_then(|o| scene.world_matrix(o))
                .map_or([0.0, 0.0, 1.0], |m| m.axis(2));
            let length = forward.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
            let towards = forward.map(|v| -v / length);
            if sun.light.use_color_temperature {
                tracing::warn!(
                    "the sun uses a colour temperature ({} K), which is not applied yet",
                    sun.light.color_temperature
                );
            }
            let on = sun.light.enabled && sun.object.is_none_or(|o| scene.is_active(o));
            let intensity = if on { sun.light.intensity } else { 0.0 };
            (towards, linear_rgb(sun.light.color).map(|v| v * intensity))
        }
        None => ([0.0, 1.0, 0.0], [0.0; 3]),
    };
    Some(SceneLightValues {
        light_direction,
        light_color,
        ambient: ambient_of(&settings),
        reflection_intensity: settings.reflection_intensity,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use rustiny_scene::read_scene;

    const SCENE: &str = "%YAML 1.1\n%TAG !u! tag:unity3d.com,2011:\n--- !u!104 &2\nRenderSettings:\n  m_AmbientSkyColor: {r: 1, g: 0.5, b: 0, a: 1}\n  m_AmbientEquatorColor: {r: 0.5, g: 0.5, b: 0.5, a: 1}\n  m_AmbientGroundColor: {r: 0, g: 0, b: 0, a: 1}\n  m_AmbientIntensity: 2\n  m_AmbientMode: 1\n  m_ReflectionIntensity: 0.5\n  m_Sun: {fileID: 12}\n--- !u!1 &10\nGameObject:\n  m_Component:\n  - component: {fileID: 11}\n  - component: {fileID: 12}\n  m_Name: Sun\n  m_IsActive: 1\n--- !u!4 &11\nTransform:\n  m_GameObject: {fileID: 10}\n  m_LocalRotation: {x: 0.70710677, y: 0, z: 0, w: 0.70710677}\n  m_LocalPosition: {x: 0, y: 0, z: 0}\n  m_LocalScale: {x: 1, y: 1, z: 1}\n  m_Children: []\n  m_Father: {fileID: 0}\n--- !u!108 &12\nLight:\n  m_GameObject: {fileID: 10}\n  m_Enabled: 1\n  m_Type: 1\n  m_Color: {r: 1, g: 0.5, b: 1, a: 1}\n  m_Intensity: 2\n";

    #[test]
    fn the_sun_and_the_gradient_ambient_come_from_the_scene() {
        let scene = read_scene(SCENE.as_bytes(), "s").unwrap();
        let v = scene_light_values(&scene).unwrap();
        // Rotated 90 degrees about X, the light's forward axis points down: the light is above.
        assert!(
            (v.light_direction[1] - 1.0).abs() < 1e-5,
            "{:?}",
            v.light_direction
        );
        let half = gamma_to_linear(0.5);
        assert!((v.light_color[0] - 2.0).abs() < 1e-5);
        assert!((v.light_color[1] - 2.0 * half).abs() < 1e-5);
        assert_eq!(v.ambient.sky, [2.0, 2.0 * half, 0.0]);
        assert_eq!(v.ambient.equator, [2.0 * half; 3]);
        assert_eq!(v.ambient.ground, [0.0; 3]);
        assert_eq!(v.reflection_intensity, 0.5);
    }

    #[test]
    fn gamma_to_linear_follows_the_srgb_curve() {
        assert_eq!(gamma_to_linear(0.0), 0.0);
        assert!((gamma_to_linear(1.0) - 1.0).abs() < 1e-6);
        assert!((gamma_to_linear(0.5) - 0.214_041).abs() < 1e-5);
    }
}
