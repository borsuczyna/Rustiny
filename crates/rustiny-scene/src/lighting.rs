//! A scene's lighting setup: its `RenderSettings` document (ambient, fog, skybox, reflections,
//! sun) and its `Light` components.

use crate::prefab::AssetRef;
use crate::unity_yaml::{fields, file_id_of, flow_value};

/// How the ambient light is defined (`m_AmbientMode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AmbientMode {
    /// From the skybox (Unity's default; mode 0).
    #[default]
    Skybox,
    /// Three colours: sky, equator, ground (mode 1).
    Trilight,
    /// One colour, the sky colour (mode 3).
    Flat,
}

/// Where reflections come from (`m_DefaultReflectionMode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReflectionSource {
    /// A cubemap rendered from the skybox (mode 0).
    #[default]
    Skybox,
    /// The cubemap in `m_CustomReflection` (mode 1).
    Custom,
}

/// The scene's `RenderSettings` document. Colours are as Unity stores them: gamma-space values,
/// the alpha unused.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderSettings {
    /// `m_Fog`.
    pub fog: bool,
    /// `m_FogColor`.
    pub fog_color: [f32; 4],
    /// `m_FogMode`: 1 linear, 2 exponential, 3 exponential squared.
    pub fog_mode: u32,
    /// `m_FogDensity`.
    pub fog_density: f32,
    /// `m_LinearFogStart`.
    pub fog_start: f32,
    /// `m_LinearFogEnd`.
    pub fog_end: f32,
    /// `m_AmbientMode`.
    pub ambient_mode: AmbientMode,
    /// `m_AmbientSkyColor`.
    pub ambient_sky: [f32; 4],
    /// `m_AmbientEquatorColor`.
    pub ambient_equator: [f32; 4],
    /// `m_AmbientGroundColor`.
    pub ambient_ground: [f32; 4],
    /// `m_AmbientIntensity`.
    pub ambient_intensity: f32,
    /// `m_SkyboxMaterial`.
    pub skybox: Option<AssetRef>,
    /// `m_DefaultReflectionMode`.
    pub reflection_source: ReflectionSource,
    /// `m_DefaultReflectionResolution`.
    pub reflection_resolution: u32,
    /// `m_ReflectionIntensity`.
    pub reflection_intensity: f32,
    /// `m_CustomReflection`.
    pub custom_reflection: Option<AssetRef>,
    /// `m_Sun`: the file ID of the `Light` component that is the sun.
    pub sun: Option<i64>,
}

impl Default for RenderSettings {
    /// Unity's defaults for a new scene.
    fn default() -> Self {
        let grey = |v: f32| [v, v, v, 1.0];
        Self {
            fog: false,
            fog_color: grey(0.5),
            fog_mode: 3,
            fog_density: 0.01,
            fog_start: 0.0,
            fog_end: 300.0,
            ambient_mode: AmbientMode::Skybox,
            ambient_sky: [0.212, 0.227, 0.259, 1.0],
            ambient_equator: [0.114, 0.125, 0.133, 1.0],
            ambient_ground: [0.047, 0.043, 0.035, 1.0],
            ambient_intensity: 1.0,
            skybox: None,
            reflection_source: ReflectionSource::Skybox,
            reflection_resolution: 128,
            reflection_intensity: 1.0,
            custom_reflection: None,
            sun: None,
        }
    }
}

/// `{r: .., g: .., b: .., a: ..}`.
pub(crate) fn parse_color(text: &str) -> Option<[f32; 4]> {
    let channel = |key| flow_value(text, key).and_then(|v| v.trim().parse::<f32>().ok());
    Some([
        channel("r")?,
        channel("g")?,
        channel("b")?,
        channel("a").unwrap_or(1.0),
    ])
}

fn number<T: std::str::FromStr>(text: &str) -> Option<T> {
    text.trim().parse().ok()
}

/// Read a `RenderSettings` document body.
pub(crate) fn parse_render_settings(body: &str) -> RenderSettings {
    let mut s = RenderSettings::default();
    for field in fields(body) {
        let v = field.value;
        match field.key {
            "m_Fog" => s.fog = v.trim() != "0",
            "m_FogColor" => s.fog_color = parse_color(v).unwrap_or(s.fog_color),
            "m_FogMode" => s.fog_mode = number(v).unwrap_or(s.fog_mode),
            "m_FogDensity" => s.fog_density = number(v).unwrap_or(s.fog_density),
            "m_LinearFogStart" => s.fog_start = number(v).unwrap_or(s.fog_start),
            "m_LinearFogEnd" => s.fog_end = number(v).unwrap_or(s.fog_end),
            "m_AmbientMode" => {
                s.ambient_mode = match number::<u32>(v) {
                    Some(1) => AmbientMode::Trilight,
                    Some(3) => AmbientMode::Flat,
                    _ => AmbientMode::Skybox,
                }
            }
            "m_AmbientSkyColor" => s.ambient_sky = parse_color(v).unwrap_or(s.ambient_sky),
            "m_AmbientEquatorColor" => {
                s.ambient_equator = parse_color(v).unwrap_or(s.ambient_equator);
            }
            "m_AmbientGroundColor" => s.ambient_ground = parse_color(v).unwrap_or(s.ambient_ground),
            "m_AmbientIntensity" => s.ambient_intensity = number(v).unwrap_or(s.ambient_intensity),
            "m_SkyboxMaterial" => s.skybox = AssetRef::parse(v),
            "m_DefaultReflectionMode" => {
                s.reflection_source = if number::<u32>(v) == Some(1) {
                    ReflectionSource::Custom
                } else {
                    ReflectionSource::Skybox
                }
            }
            "m_DefaultReflectionResolution" => {
                s.reflection_resolution = number(v).unwrap_or(s.reflection_resolution);
            }
            "m_ReflectionIntensity" => {
                s.reflection_intensity = number(v).unwrap_or(s.reflection_intensity);
            }
            "m_CustomReflection" => s.custom_reflection = AssetRef::parse(v),
            "m_Sun" => s.sun = file_id_of(v).filter(|id| *id != 0),
            _ => {}
        }
    }
    s
}

/// `Light.m_Type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightType {
    /// 0.
    Spot,
    /// 1.
    Directional,
    /// 2.
    Point,
    /// 3 (baked only).
    Area,
}

/// `Light.m_Shadows.m_Type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightShadows {
    /// 0.
    None,
    /// 1.
    Hard,
    /// 2.
    Soft,
}

/// A `Light` component.
#[derive(Debug, Clone, PartialEq)]
pub struct Light {
    /// File ID of the component.
    pub file_id: i64,
    /// File ID of its game object.
    pub game_object: i64,
    /// `m_Enabled`.
    pub enabled: bool,
    /// `m_Type`.
    pub kind: LightType,
    /// `m_Color`, gamma-space.
    pub color: [f32; 4],
    /// `m_Intensity`.
    pub intensity: f32,
    /// `m_Range`.
    pub range: f32,
    /// `m_SpotAngle` in degrees.
    pub spot_angle: f32,
    /// `m_Shadows.m_Type`.
    pub shadows: LightShadows,
    /// `m_Shadows.m_Strength`.
    pub shadow_strength: f32,
    /// `m_Shadows.m_Bias`.
    pub shadow_bias: f32,
    /// `m_Shadows.m_NormalBias`.
    pub shadow_normal_bias: f32,
    /// `m_ColorTemperature` in kelvin, used when `use_color_temperature`.
    pub color_temperature: f32,
    /// `m_UseColorTemperature`.
    pub use_color_temperature: bool,
}

/// Read a `Light` document body.
pub(crate) fn parse_light(file_id: i64, body: &str) -> Light {
    let mut light = Light {
        file_id,
        game_object: 0,
        enabled: true,
        kind: LightType::Point,
        color: [1.0; 4],
        intensity: 1.0,
        range: 10.0,
        spot_angle: 30.0,
        shadows: LightShadows::None,
        shadow_strength: 1.0,
        shadow_bias: 0.05,
        shadow_normal_bias: 0.4,
        color_temperature: 6570.0,
        use_color_temperature: false,
    };
    for field in fields(body) {
        let v = field.value;
        match field.key {
            "m_GameObject" => light.game_object = file_id_of(v).unwrap_or(0),
            "m_Enabled" => light.enabled = v.trim() != "0",
            "m_Type" => {
                light.kind = match number::<u32>(v) {
                    Some(0) => LightType::Spot,
                    Some(1) => LightType::Directional,
                    Some(3) => LightType::Area,
                    _ => LightType::Point,
                }
            }
            "m_Color" => light.color = parse_color(v).unwrap_or(light.color),
            "m_Intensity" => light.intensity = number(v).unwrap_or(light.intensity),
            "m_Range" => light.range = number(v).unwrap_or(light.range),
            "m_SpotAngle" => light.spot_angle = number(v).unwrap_or(light.spot_angle),
            "m_ColorTemperature" => {
                light.color_temperature = number(v).unwrap_or(light.color_temperature);
            }
            "m_UseColorTemperature" => light.use_color_temperature = v.trim() != "0",
            "m_Shadows" => {
                for line in field.block().lines().map(str::trim) {
                    let Some((key, value)) = line.split_once(':') else {
                        continue;
                    };
                    match key {
                        "m_Type" => {
                            light.shadows = match number::<u32>(value) {
                                Some(1) => LightShadows::Hard,
                                Some(2) => LightShadows::Soft,
                                _ => LightShadows::None,
                            }
                        }
                        "m_Strength" => {
                            light.shadow_strength = number(value).unwrap_or(light.shadow_strength);
                        }
                        "m_Bias" => light.shadow_bias = number(value).unwrap_or(light.shadow_bias),
                        "m_NormalBias" => {
                            light.shadow_normal_bias =
                                number(value).unwrap_or(light.shadow_normal_bias);
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    light
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_settings_are_read_with_their_ambient_skybox_and_sun() {
        let body = "RenderSettings:\n  m_Fog: 1\n  m_FogColor: {r: 0.5, g: 0.4, b: 0.3, a: 1}\n  m_FogMode: 3\n  m_FogDensity: 0.006\n  m_AmbientSkyColor: {r: 0.37, g: 0.38, b: 0.5, a: 0.8}\n  m_AmbientEquatorColor: {r: 0.4, g: 0.33, b: 0.33, a: 0.7}\n  m_AmbientGroundColor: {r: 0.14, g: 0.11, b: 0.1, a: 0.18}\n  m_AmbientIntensity: 1\n  m_AmbientMode: 1\n  m_SkyboxMaterial: {fileID: 2100000, guid: d579acb4e449c8a499d70e1145ccb07f, type: 2}\n  m_DefaultReflectionMode: 0\n  m_DefaultReflectionResolution: 128\n  m_ReflectionIntensity: 0.51\n  m_CustomReflection: {fileID: 0}\n  m_Sun: {fileID: 410087040}\n";
        let s = parse_render_settings(body);
        assert!(s.fog);
        assert_eq!(s.fog_color, [0.5, 0.4, 0.3, 1.0]);
        assert_eq!(s.ambient_mode, AmbientMode::Trilight);
        assert_eq!(s.ambient_sky[..3], [0.37, 0.38, 0.5]);
        assert_eq!(
            s.skybox.as_ref().map(|r| r.guid.as_str()),
            Some("d579acb4e449c8a499d70e1145ccb07f")
        );
        assert_eq!(s.reflection_source, ReflectionSource::Skybox);
        assert_eq!(s.reflection_intensity, 0.51);
        assert!(s.custom_reflection.is_none());
        assert_eq!(s.sun, Some(410_087_040));
    }

    #[test]
    fn a_directional_light_is_read_with_its_shadow_settings() {
        let body = "Light:\n  m_GameObject: {fileID: 7}\n  m_Enabled: 1\n  m_Type: 1\n  m_Color: {r: 1, g: 0.9, b: 0.8, a: 1}\n  m_Intensity: 1.3\n  m_Range: 10\n  m_SpotAngle: 30\n  m_Shadows:\n    m_Type: 2\n    m_Resolution: -1\n    m_Strength: 0.8\n    m_Bias: 0.05\n    m_NormalBias: 0.4\n  m_ColorTemperature: 5000\n  m_UseColorTemperature: 1\n";
        let l = parse_light(5, body);
        assert_eq!((l.file_id, l.game_object), (5, 7));
        assert_eq!(l.kind, LightType::Directional);
        assert_eq!(l.color, [1.0, 0.9, 0.8, 1.0]);
        assert_eq!(l.intensity, 1.3);
        assert_eq!(l.shadows, LightShadows::Soft);
        assert_eq!(l.shadow_strength, 0.8);
        assert!(l.use_color_temperature);
        assert_eq!(l.color_temperature, 5000.0);
    }
}
