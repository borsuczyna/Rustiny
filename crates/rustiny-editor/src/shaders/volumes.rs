//! The camera's post-processing as URP's volume system resolves it for a scene: the default
//! volume profile of the URP global settings at the bottom, then the scene's enabled global
//! volumes by priority, each with its weight. Local volumes (which need the camera inside their
//! colliders) are not evaluated.

use super::lighting::gamma_to_linear;
use rustiny_render::{Bloom, PostProcess, Tonemap, Vignette};
use rustiny_scene::Scene;
use rustiny_scene::unity_yaml::flow_value;
use rustiny_scene::volume::{VolumeStack, parse_volume_profile};
use std::path::{Path, PathBuf};

/// The `guid` of the asset `key` names in `text` (`key: {fileID: .., guid: .., type: ..}`),
/// looking from the first occurrence of `after` on.
fn guid_after<'a>(text: &'a str, after: &str, key: &str) -> Option<&'a str> {
    let start = text.find(after)?;
    let rest = &text[start..];
    let line = rest.lines().find(|l| l.trim_start().starts_with(key))?;
    flow_value(line.split_once(':')?.1, "guid")
}

/// The default volume profile of the URP global settings
/// (`ProjectSettings/GraphicsSettings.asset` names the global settings asset).
fn default_profile(root: &Path, path_of: &impl Fn(&str) -> Option<PathBuf>) -> Option<PathBuf> {
    let graphics =
        std::fs::read_to_string(root.join("ProjectSettings").join("GraphicsSettings.asset"))
            .ok()?;
    let settings_guid = guid_after(
        &graphics,
        "m_RenderPipelineGlobalSettingsMap:",
        "UnityEngine.Rendering.Universal.UniversalRenderPipeline:",
    )?;
    let settings = std::fs::read_to_string(path_of(settings_guid)?).ok()?;
    let profile_guid = guid_after(
        &settings,
        "URPDefaultVolumeProfileSettings",
        "m_VolumeProfile:",
    )
    .or_else(|| {
        guid_after(
            &settings,
            "m_DefaultVolumeProfile:",
            "m_DefaultVolumeProfile:",
        )
    })?;
    path_of(profile_guid)
}

/// The volume stack of `scene`. `path_of` gives the file of an asset GUID. Reads files: call it
/// off the UI thread.
pub fn scene_volume_stack(
    root: &Path,
    scene: &Scene,
    path_of: impl Fn(&str) -> Option<PathBuf>,
) -> VolumeStack {
    let mut stack = VolumeStack::default();
    let read = |path: PathBuf| {
        std::fs::read_to_string(&path)
            .map(|text| parse_volume_profile(&text))
            .map_err(|e| tracing::warn!("volume profile {}: {e}", path.display()))
            .ok()
    };
    if let Some(components) = default_profile(root, &path_of).and_then(read) {
        stack.apply(&components, 1.0);
    }
    let mut volumes: Vec<_> = scene
        .lighting()
        .volumes
        .iter()
        .filter(|(object, v)| {
            v.enabled && v.global && v.weight > 0.0 && object.is_none_or(|o| scene.is_active(o))
        })
        .collect();
    let local = scene
        .lighting()
        .volumes
        .iter()
        .filter(|(_, v)| !v.global)
        .count();
    if local > 0 {
        tracing::info!("{local} local volumes are not evaluated (only global ones are)");
    }
    // Lower priority first, so higher ones override; equal priorities keep the scene order.
    volumes.sort_by(|a, b| a.1.priority.total_cmp(&b.1.priority));
    for (_, volume) in volumes {
        if let Some(components) = volume
            .profile
            .as_ref()
            .and_then(|p| path_of(&p.guid))
            .and_then(read)
        {
            stack.apply(&components, volume.weight.clamp(0.0, 1.0));
        }
    }
    stack
}

fn linear(c: [f32; 4]) -> [f32; 3] {
    [
        gamma_to_linear(c[0]),
        gamma_to_linear(c[1]),
        gamma_to_linear(c[2]),
    ]
}

/// The post-processing the stack asks for, with URP's defaults for what no profile overrides.
pub fn post_process_of(stack: &VolumeStack) -> PostProcess {
    let mut post = PostProcess::default();
    if stack.has("ColorAdjustments") {
        let c = "ColorAdjustments";
        post.post_exposure = stack.number(c, "postExposure", 0.0);
        post.contrast = stack.number(c, "contrast", 0.0);
        post.color_filter = linear(stack.color(c, "colorFilter", [1.0; 4]));
        post.hue_shift = stack.number(c, "hueShift", 0.0);
        post.saturation = stack.number(c, "saturation", 0.0);
    }
    if stack.has("Tonemapping") {
        post.tonemap = match stack.number("Tonemapping", "mode", 0.0) as i32 {
            1 => Tonemap::Neutral,
            2 => {
                tracing::warn!(
                    "ACES tonemapping is not reproduced yet; the image is not tonemapped"
                );
                Tonemap::None
            }
            _ => Tonemap::None,
        };
    }
    if stack.has("Vignette") {
        let v = "Vignette";
        let color = stack.color(v, "color", [0.0, 0.0, 0.0, 1.0]);
        let center = stack.vector(v, "center", [0.5, 0.5, 0.0, 0.0]);
        post.vignette = Vignette {
            color: [color[0], color[1], color[2]],
            center: [center[0], center[1]],
            intensity: stack.number(v, "intensity", 0.0),
            smoothness: stack.number(v, "smoothness", 0.2),
            rounded: stack.number(v, "rounded", 0.0) != 0.0,
        };
    }
    if stack.has("Bloom") {
        let b = "Bloom";
        let bloom = Bloom {
            threshold: stack.number(b, "threshold", 0.9),
            intensity: stack.number(b, "intensity", 0.0),
            scatter: stack.number(b, "scatter", 0.7),
            clamp: stack.number(b, "clamp", 65472.0),
            tint: linear(stack.color(b, "tint", [1.0; 4])),
            high_quality: stack.number(b, "highQualityFiltering", 0.0) != 0.0,
            skip_iterations: stack.number(b, "skipIterations", 1.0).max(0.0) as u32,
            max_iterations: stack.number(b, "maxIterations", 6.0).max(1.0) as u32,
        };
        post.bloom = (bloom.intensity > 0.0).then_some(bloom);
    }
    post
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn the_scene_profile_becomes_urps_settings() {
        let profile = "x\n--- !u!114 &1\nMonoBehaviour:\n  m_Name: Bloom\n  active: 1\n  skipIterations:\n    m_OverrideState: 1\n    m_Value: 0\n  threshold:\n    m_OverrideState: 1\n    m_Value: 3.8\n  intensity:\n    m_OverrideState: 1\n    m_Value: 2.2\n  scatter:\n    m_OverrideState: 1\n    m_Value: 0.6\n  highQualityFiltering:\n    m_OverrideState: 1\n    m_Value: 1\n--- !u!114 &2\nMonoBehaviour:\n  m_Name: Vignette\n  active: 1\n  intensity:\n    m_OverrideState: 1\n    m_Value: 0.2\n--- !u!114 &3\nMonoBehaviour:\n  m_Name: Tonemapping\n  active: 1\n  mode:\n    m_OverrideState: 1\n    m_Value: 1\n";
        let mut stack = VolumeStack::default();
        stack.apply(&parse_volume_profile(profile), 1.0);
        let post = post_process_of(&stack);
        assert_eq!(post.tonemap, Tonemap::Neutral);
        assert_eq!(post.vignette.intensity, 0.2);
        assert_eq!(post.vignette.smoothness, 0.2, "URP's default");
        let bloom = post.bloom.unwrap();
        assert_eq!(
            (bloom.threshold, bloom.intensity, bloom.scatter),
            (3.8, 2.2, 0.6)
        );
        assert!(bloom.high_quality);
        assert_eq!((bloom.skip_iterations, bloom.max_iterations), (0, 6));
        assert_eq!(post.post_exposure, 0.0);
    }

    #[test]
    fn nothing_overridden_means_no_post_processing() {
        assert_eq!(
            post_process_of(&VolumeStack::default()),
            PostProcess::default()
        );
    }

    #[test]
    fn the_default_profile_is_found_through_the_graphics_settings() {
        let graphics = "GraphicsSettings:\n  m_RenderPipelineGlobalSettingsMap:\n    UnityEngine.Rendering.Universal.UniversalRenderPipeline: {fileID: 11400000, guid: 18dc0cd2c080841dea60987a38ce93fa, type: 2}\n";
        assert_eq!(
            guid_after(
                graphics,
                "m_RenderPipelineGlobalSettingsMap:",
                "UnityEngine.Rendering.Universal.UniversalRenderPipeline:"
            ),
            Some("18dc0cd2c080841dea60987a38ce93fa")
        );
        let settings = "    - rid: 1\n      type: {class: URPDefaultVolumeProfileSettings}\n      data:\n        m_VolumeProfile: {fileID: 11400000, guid: ab09877e2e707104187f6f83e2f62510, type: 2}\n";
        assert_eq!(
            guid_after(
                settings,
                "URPDefaultVolumeProfileSettings",
                "m_VolumeProfile:"
            ),
            Some("ab09877e2e707104187f6f83e2f62510")
        );
    }
}
