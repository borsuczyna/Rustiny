//! URP's volume framework: `Volume` components of a scene and the `VolumeProfile` assets they
//! name. A profile is a list of components (`Bloom`, `Tonemapping`, ...), each a set of
//! parameters that are either overridden (`m_OverrideState: 1`) or left to what is below it.

use crate::prefab::AssetRef;
use crate::unity_yaml::{fields, file_id_of, flow_value};
use std::collections::HashMap;

/// GUID of URP's `Volume` script (`com.unity.render-pipelines.core`, `Volume.cs`).
pub const VOLUME_SCRIPT_GUID: &str = "172515602e62fb746b5d573b38a5fe58";

/// A `Volume` component of the scene.
#[derive(Debug, Clone, PartialEq)]
pub struct SceneVolume {
    /// File ID of its game object.
    pub game_object: i64,
    /// `m_Enabled`.
    pub enabled: bool,
    /// `m_IsGlobal`: applies everywhere (local volumes apply inside their colliders).
    pub global: bool,
    /// `priority`: higher volumes override lower ones.
    pub priority: f32,
    /// `weight`.
    pub weight: f32,
    /// `blendDistance`.
    pub blend_distance: f32,
    /// `sharedProfile`.
    pub profile: Option<AssetRef>,
}

/// Read a `Volume` MonoBehaviour body. `None` when the document is another script.
pub fn parse_volume(body: &str) -> Option<SceneVolume> {
    let mut volume = SceneVolume {
        game_object: 0,
        enabled: true,
        global: true,
        priority: 0.0,
        weight: 1.0,
        blend_distance: 0.0,
        profile: None,
    };
    let mut is_volume = false;
    for field in fields(body) {
        let v = field.value.trim();
        match field.key {
            "m_Script" => is_volume = flow_value(v, "guid") == Some(VOLUME_SCRIPT_GUID),
            "m_GameObject" => volume.game_object = file_id_of(v).unwrap_or(0),
            "m_Enabled" => volume.enabled = v != "0",
            "m_IsGlobal" => volume.global = v != "0",
            "priority" => volume.priority = v.parse().unwrap_or(0.0),
            "weight" => volume.weight = v.parse().unwrap_or(1.0),
            "blendDistance" => volume.blend_distance = v.parse().unwrap_or(0.0),
            "sharedProfile" => volume.profile = AssetRef::parse(v),
            _ => {}
        }
    }
    is_volume.then_some(volume)
}

/// One component of a profile: its overridden parameters, by name, as their YAML text
/// (`3.8`, `{r: 1, g: 1, b: 1, a: 1}`, ...).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct VolumeComponent {
    /// The component type (`Bloom`, `Tonemapping`, ...), from `m_Name`.
    pub name: String,
    /// `active`.
    pub active: bool,
    /// Overridden parameters.
    pub overrides: HashMap<String, String>,
}

/// The components of a `VolumeProfile` asset (`.asset` text). Components are the MonoBehaviour
/// documents next to the profile; the profile document itself (with `components:`) is skipped.
pub fn parse_volume_profile(text: &str) -> Vec<VolumeComponent> {
    let mut out = Vec::new();
    for document in text.split("\n--- !u!").skip(1) {
        let Some((_, body)) = document.split_once('\n') else {
            continue;
        };
        if !body.starts_with("MonoBehaviour:") || body.contains("\n  components:") {
            continue;
        }
        let mut component = VolumeComponent {
            active: true,
            ..VolumeComponent::default()
        };
        let mut lines = body.lines().peekable();
        while let Some(line) = lines.next() {
            if let Some(name) = line.strip_prefix("  m_Name: ") {
                component.name = name.trim().to_owned();
            } else if let Some(active) = line.strip_prefix("  active: ") {
                component.active = active.trim() != "0";
            } else if let Some(key) = line.strip_prefix("  ").and_then(|l| l.strip_suffix(':'))
                && !key.starts_with(' ')
            {
                // A parameter block: `m_OverrideState` then `m_Value`.
                let mut overridden = false;
                let mut value = None;
                while let Some(next) = lines.peek() {
                    if !next.starts_with("    ") {
                        break;
                    }
                    let next = lines.next().unwrap_or_default().trim();
                    if let Some(state) = next.strip_prefix("m_OverrideState:") {
                        overridden = state.trim() != "0";
                    } else if let Some(v) = next.strip_prefix("m_Value:") {
                        value = Some(v.trim().to_owned());
                    }
                }
                if overridden && let Some(value) = value {
                    component.overrides.insert(key.to_owned(), value);
                }
            }
        }
        if !component.name.is_empty() {
            out.push(component);
        }
    }
    out
}

/// The parameters in effect: components of several profiles laid over each other, lowest
/// priority first, each overridden parameter replacing (or, with a weight below 1, blending
/// towards, for numbers) what is below.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct VolumeStack {
    values: HashMap<(String, String), String>,
    active: HashMap<String, bool>,
}

impl VolumeStack {
    /// Lay `components` over the stack with `weight` (0..1).
    pub fn apply(&mut self, components: &[VolumeComponent], weight: f32) {
        for component in components {
            if !component.active {
                continue;
            }
            self.active.insert(component.name.clone(), true);
            for (key, value) in &component.overrides {
                let slot = (component.name.clone(), key.clone());
                let blended = match (self.values.get(&slot), weight < 1.0) {
                    (Some(below), true) => blend(below, value, weight),
                    _ => value.clone(),
                };
                self.values.insert(slot, blended);
            }
        }
    }

    /// The text of `component.parameter`, if any profile overrides it.
    pub fn get(&self, component: &str, parameter: &str) -> Option<&str> {
        self.values
            .get(&(component.to_owned(), parameter.to_owned()))
            .map(String::as_str)
    }

    /// A number parameter, else `default`.
    pub fn number(&self, component: &str, parameter: &str, default: f32) -> f32 {
        self.get(component, parameter)
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }

    /// A colour parameter (`{r, g, b, a}`), else `default`.
    pub fn color(&self, component: &str, parameter: &str, default: [f32; 4]) -> [f32; 4] {
        self.get(component, parameter)
            .and_then(crate::lighting::parse_color)
            .unwrap_or(default)
    }

    /// A vector parameter (`{x, y, ...}`), else `default`.
    pub fn vector(&self, component: &str, parameter: &str, default: [f32; 4]) -> [f32; 4] {
        let Some(text) = self.get(component, parameter) else {
            return default;
        };
        let mut out = default;
        for (i, key) in ["x", "y", "z", "w"].into_iter().enumerate() {
            if let Some(v) = flow_value(text, key).and_then(|v| v.trim().parse().ok()) {
                out[i] = v;
            }
        }
        out
    }

    /// Whether some profile has the component (active).
    pub fn has(&self, component: &str) -> bool {
        self.active.get(component).copied().unwrap_or(false)
    }
}

/// `below` moved towards `above` by `t`, for plain numbers; other values switch at one half.
fn blend(below: &str, above: &str, t: f32) -> String {
    match (below.parse::<f32>(), above.parse::<f32>()) {
        (Ok(a), Ok(b)) => (a + (b - a) * t).to_string(),
        _ if t >= 0.5 => above.to_owned(),
        _ => below.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROFILE: &str = "%YAML 1.1\n%TAG !u! tag:unity3d.com,2011:\n--- !u!114 &1\nMonoBehaviour:\n  m_Name: Bloom\n  active: 1\n  threshold:\n    m_OverrideState: 1\n    m_Value: 3.8\n  intensity:\n    m_OverrideState: 1\n    m_Value: 2.2\n  tint:\n    m_OverrideState: 0\n    m_Value: {r: 1, g: 0, b: 0, a: 1}\n--- !u!114 &2\nMonoBehaviour:\n  m_Name: SampleSceneProfile\n  components:\n  - {fileID: 1}\n--- !u!114 &3\nMonoBehaviour:\n  m_Name: Tonemapping\n  active: 1\n  mode:\n    m_OverrideState: 1\n    m_Value: 1\n--- !u!114 &4\nMonoBehaviour:\n  m_Name: MotionBlur\n  active: 0\n  intensity:\n    m_OverrideState: 1\n    m_Value: 0.6\n";

    #[test]
    fn a_profile_keeps_only_overridden_parameters() {
        let components = parse_volume_profile(PROFILE);
        let names: Vec<&str> = components.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["Bloom", "Tonemapping", "MotionBlur"]);
        assert_eq!(
            components[0].overrides.get("threshold").map(String::as_str),
            Some("3.8")
        );
        assert!(
            !components[0].overrides.contains_key("tint"),
            "not overridden"
        );
        assert!(!components[2].active);
    }

    #[test]
    fn higher_volumes_override_and_inactive_components_do_nothing() {
        let mut stack = VolumeStack::default();
        let base = parse_volume_profile(
            "x\n--- !u!114 &1\nMonoBehaviour:\n  m_Name: Bloom\n  active: 1\n  intensity:\n    m_OverrideState: 1\n    m_Value: 0\n  scatter:\n    m_OverrideState: 1\n    m_Value: 0.7\n",
        );
        stack.apply(&base, 1.0);
        stack.apply(&parse_volume_profile(PROFILE), 1.0);
        assert_eq!(stack.number("Bloom", "intensity", 0.0), 2.2);
        assert_eq!(
            stack.number("Bloom", "scatter", 0.0),
            0.7,
            "kept from below"
        );
        assert_eq!(stack.number("Tonemapping", "mode", 0.0), 1.0);
        assert_eq!(stack.number("MotionBlur", "intensity", 0.0), 0.0);
        let mut half = VolumeStack::default();
        half.apply(&base, 1.0);
        half.apply(&parse_volume_profile(PROFILE), 0.5);
        assert!((half.number("Bloom", "intensity", 0.0) - 1.1).abs() < 1e-6);
    }

    #[test]
    fn a_volume_component_is_recognised_by_its_script() {
        let body = "MonoBehaviour:\n  m_GameObject: {fileID: 9}\n  m_Enabled: 1\n  m_Script: {fileID: 11500000, guid: 172515602e62fb746b5d573b38a5fe58, type: 3}\n  m_IsGlobal: 1\n  priority: 2\n  blendDistance: 0\n  weight: 0.5\n  sharedProfile: {fileID: 11400000, guid: 10fc4df2da32a41aaa32d77bc913491c, type: 2}\n";
        let v = parse_volume(body).unwrap();
        assert!(v.global && v.enabled);
        assert_eq!((v.game_object, v.priority, v.weight), (9, 2.0, 0.5));
        assert_eq!(
            v.profile.map(|p| p.guid),
            Some("10fc4df2da32a41aaa32d77bc913491c".to_owned())
        );
        assert!(parse_volume(&body.replace("172515602e", "000000000e")).is_none());
    }
}
