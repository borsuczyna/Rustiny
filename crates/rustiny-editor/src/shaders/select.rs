//! Choosing what to draw from a parsed Unity shader and a material, without a GPU or a compiler:
//! which pass, which keywords, and the fixed-function state with `[_Property]` references resolved.

use rustiny_render::{Blend, BlendFactor, CullMode, DepthTest, PassState};
use rustiny_shader::{
    BlendSetting, Compare, CullMode as LabCull, KeywordGroup, KeywordKind, Language, Property,
    PropertyKind, RenderState, ResolvedPass, Setting,
};

/// The pass that draws the object with the shader's own lighting, as an index into `passes`.
///
/// URP's forward pass first, then its forward-only and unlit variants, then a pass with no
/// `LightMode` at all. Shadow, depth, meta and G-buffer passes are never drawn here. Only passes
/// with an HLSL vertex and fragment program count.
pub(crate) fn choose_pass(passes: &[ResolvedPass]) -> Option<usize> {
    let rank = |pass: &ResolvedPass| -> Option<u32> {
        let program = pass.pass.program.as_ref()?;
        if program.language != Language::Hlsl
            || program.vertex.is_none()
            || program.fragment.is_none()
        {
            return None;
        }
        match pass.light_mode.as_deref().map(str::to_ascii_lowercase) {
            Some(mode) => match mode.as_str() {
                "universalforward" => Some(0),
                "universalforwardonly" => Some(1),
                "srpdefaultunlit" => Some(2),
                "always" => Some(3),
                _ => None,
            },
            None => Some(3),
        }
    };
    passes
        .iter()
        .enumerate()
        .filter_map(|(index, pass)| rank(pass).map(|r| (r, index)))
        .min()
        .map(|(_, index)| index)
}

/// The keywords to compile with: from each `multi_compile` / `shader_feature` group the option the
/// material enables, else (for `multi_compile`) the first option, which Unity uses when nothing
/// selects another. The option `_` stands for "none of them" and adds no keyword.
pub(crate) fn enabled_keywords(groups: &[KeywordGroup], material: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for group in groups {
        let chosen = group
            .options
            .iter()
            .find(|option| option.as_str() != "_" && material.contains(option))
            .or_else(|| {
                (group.kind == KeywordKind::MultiCompile)
                    .then(|| group.options.first())
                    .flatten()
            });
        if let Some(keyword) = chosen.filter(|k| k.as_str() != "_")
            && !out.contains(keyword)
        {
            out.push(keyword.clone());
        }
    }
    out.sort();
    out
}

/// Keywords that float properties switch on: `[Toggle(_KEYWORD)] _Prop` enables `_KEYWORD` when
/// the value is 1, `[Toggle] _Prop` enables `_PROP_ON`, and `[KeywordEnum(A, B)] _Prop` enables
/// `_PROP_A` or `_PROP_B` by the value's index.
pub(crate) fn property_keywords(
    properties: &[Property],
    number: impl Fn(&str) -> Option<f32>,
) -> Vec<String> {
    let mut out = Vec::new();
    for property in properties {
        let value = number(&property.name).unwrap_or(match property.default {
            rustiny_shader::PropertyDefault::Number(n) => n,
            _ => 0.0,
        });
        for attribute in &property.attributes {
            let (name, args) = match attribute.split_once('(') {
                Some((name, rest)) => (name.trim(), rest.trim_end_matches(')').trim()),
                None => (attribute.trim(), ""),
            };
            match name {
                "Toggle" if value >= 0.5 => {
                    out.push(if args.is_empty() {
                        format!("{}_ON", property.name.to_ascii_uppercase())
                    } else {
                        args.to_owned()
                    });
                }
                "KeywordEnum" => {
                    let options: Vec<&str> = args.split(',').map(str::trim).collect();
                    if let Some(option) = options.get(value.round().max(0.0) as usize) {
                        out.push(format!(
                            "{}_{}",
                            property.name.to_ascii_uppercase(),
                            option.to_ascii_uppercase()
                        ));
                    }
                }
                _ => {}
            }
        }
    }
    out
}

fn compare(value: Compare) -> DepthTest {
    match value {
        Compare::Never => DepthTest::Never,
        Compare::Less => DepthTest::Less,
        Compare::Equal => DepthTest::Equal,
        Compare::LEqual => DepthTest::LessEqual,
        Compare::Greater => DepthTest::Greater,
        Compare::NotEqual => DepthTest::NotEqual,
        Compare::GEqual => DepthTest::GreaterEqual,
        Compare::Always => DepthTest::Always,
    }
}

fn factor(value: rustiny_shader::BlendFactor) -> BlendFactor {
    use rustiny_shader::BlendFactor as F;
    match value {
        F::Zero => BlendFactor::Zero,
        F::One => BlendFactor::One,
        F::DstColor => BlendFactor::DstColor,
        F::SrcColor => BlendFactor::SrcColor,
        F::OneMinusDstColor => BlendFactor::OneMinusDstColor,
        F::SrcAlpha | F::SrcAlphaSaturate => BlendFactor::SrcAlpha,
        F::OneMinusSrcColor => BlendFactor::OneMinusSrcColor,
        F::DstAlpha => BlendFactor::DstAlpha,
        F::OneMinusDstAlpha => BlendFactor::OneMinusDstAlpha,
        F::OneMinusSrcAlpha => BlendFactor::OneMinusSrcAlpha,
    }
}

/// Unity's `BlendMode` enum values, as materials store them in `_SrcBlend`/`_DstBlend`.
fn factor_of_number(value: f32) -> BlendFactor {
    match value.round() as i32 {
        0 => BlendFactor::Zero,
        2 => BlendFactor::DstColor,
        3 => BlendFactor::SrcColor,
        4 => BlendFactor::OneMinusDstColor,
        5 | 9 => BlendFactor::SrcAlpha,
        6 => BlendFactor::OneMinusSrcColor,
        7 => BlendFactor::DstAlpha,
        8 => BlendFactor::OneMinusDstAlpha,
        10 => BlendFactor::OneMinusSrcAlpha,
        _ => BlendFactor::One,
    }
}

/// Unity's `ColorMask` text as channel bits (1 red, 2 green, 4 blue, 8 alpha).
fn color_mask(text: &str) -> u8 {
    if text.trim() == "0" {
        return 0;
    }
    text.chars().fold(0, |bits, c| {
        bits | match c.to_ascii_uppercase() {
            'R' => 1,
            'G' => 2,
            'B' => 4,
            'A' => 8,
            _ => 0,
        }
    })
}

/// The pass's render state with `[_Property]` settings read through `number` (the material's
/// value of a float property, if it has one).
pub(crate) fn pass_state(state: &RenderState, number: impl Fn(&str) -> Option<f32>) -> PassState {
    let mut out = PassState::default();
    match &state.cull {
        Some(Setting::Fixed(mode)) => out.cull = cull(*mode),
        Some(Setting::Property(name)) => {
            if let Some(value) = number(name) {
                out.cull = match value.round() as i32 {
                    0 => CullMode::Off,
                    1 => CullMode::Front,
                    _ => CullMode::Back,
                };
            }
        }
        None => {}
    }
    match &state.z_write {
        Some(Setting::Fixed(on)) => out.z_write = *on,
        Some(Setting::Property(name)) => {
            if let Some(value) = number(name) {
                out.z_write = value >= 0.5;
            }
        }
        None => {}
    }
    match &state.z_test {
        Some(Setting::Fixed(test)) => out.z_test = compare(*test),
        Some(Setting::Property(name)) => {
            if let Some(value) = number(name) {
                // CompareFunction: 0 disabled, 1 never, 2 less, ... 8 always.
                out.z_test = match value.round() as i32 {
                    1 => DepthTest::Never,
                    2 => DepthTest::Less,
                    3 => DepthTest::Equal,
                    5 => DepthTest::Greater,
                    6 => DepthTest::NotEqual,
                    7 => DepthTest::GreaterEqual,
                    8 | 0 => DepthTest::Always,
                    _ => DepthTest::LessEqual,
                };
            }
        }
        None => {}
    }
    if let BlendSetting::On(blend) = &state.blend {
        let resolve = |setting: &Setting<rustiny_shader::BlendFactor>| match setting {
            Setting::Fixed(f) => Some(factor(*f)),
            Setting::Property(name) => number(name).map(factor_of_number),
        };
        if let (Some(src), Some(dst)) = (resolve(&blend.src), resolve(&blend.dst)) {
            let (src_alpha, dst_alpha) = match &blend.alpha {
                Some((a, b)) => (resolve(a).unwrap_or(src), resolve(b).unwrap_or(dst)),
                None => (src, dst),
            };
            out.blend = Some(Blend {
                src_color: src,
                dst_color: dst,
                src_alpha,
                dst_alpha,
            });
        }
    }
    if let Some(mask) = &state.color_mask {
        out.color_mask = color_mask(mask);
    }
    out
}

fn cull(mode: LabCull) -> CullMode {
    match mode {
        LabCull::Back => CullMode::Back,
        LabCull::Front => CullMode::Front,
        LabCull::Off => CullMode::Off,
    }
}

/// Whether a texture property holds colour (sRGB) or data (linear): normal maps, masks and the
/// like are data. Decided from the property's default texture, attributes and name.
pub(crate) fn texture_is_srgb(property: &Property) -> bool {
    if let rustiny_shader::PropertyDefault::Texture(default) = &property.default
        && default.eq_ignore_ascii_case("bump")
    {
        return false;
    }
    if property
        .attributes
        .iter()
        .any(|a| a.eq_ignore_ascii_case("Normal"))
    {
        return false;
    }
    let name = property.name.to_ascii_lowercase();
    ![
        "normal",
        "bump",
        "mask",
        "control",
        "metal",
        "height",
        "noise",
        "rough",
        "occlusion",
        "data",
        "lut",
        "splat",
        "weight",
    ]
    .iter()
    .any(|word| name.contains(word))
}

/// Whether a property is a texture.
pub(crate) fn is_texture(kind: &PropertyKind) -> bool {
    matches!(
        kind,
        PropertyKind::Texture2D
            | PropertyKind::Texture3D
            | PropertyKind::Cube
            | PropertyKind::Texture2DArray
            | PropertyKind::CubeArray
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use rustiny_shader::{ShaderLibrary, parse_shader};

    fn passes(text: &str) -> Vec<ResolvedPass> {
        let shader = parse_shader(text).unwrap();
        let mut library = ShaderLibrary::new();
        library.add(shader.clone());
        library.resolve_passes(&shader, &shader.sub_shaders[0])
    }

    const SHADER: &str = r#"Shader "T/S" {
        Properties {
            [Toggle(_FANCY)] _Fancy ("Fancy", Float) = 0
            [Toggle] _Plain ("Plain", Float) = 0
            [KeywordEnum(Low, High)] _Quality ("Q", Float) = 1
            [Normal] _Detail ("D", 2D) = "white" {}
            _BumpMap ("B", 2D) = "bump" {}
            _Albedo ("A", 2D) = "white" {}
        }
        SubShader {
            Cull [_Cull]
            Pass { Name "ShadowCaster" Tags { "LightMode"="ShadowCaster" }
                HLSLPROGRAM
                #pragma vertex v
                #pragma fragment f
                ENDHLSL }
            Pass { Name "Depth" Tags { "LightMode"="DepthOnly" }
                HLSLPROGRAM
                #pragma vertex v
                #pragma fragment f
                ENDHLSL }
            Pass { Name "Fwd" Tags { "LightMode"="UniversalForward" }
                Blend [_Src] [_Dst], One OneMinusSrcAlpha
                ZWrite [_ZWrite] ColorMask RGB
                HLSLPROGRAM
                #pragma vertex v
                #pragma fragment f
                #pragma multi_compile _ _A _B
                #pragma multi_compile_fog
                #pragma shader_feature_local _NORMALMAP
                ENDHLSL }
        }
    }"#;

    #[test]
    fn the_forward_pass_is_chosen_over_shadow_and_depth() {
        let p = passes(SHADER);
        assert_eq!(p.len(), 3);
        assert_eq!(p[choose_pass(&p).unwrap()].name.as_deref(), Some("Fwd"));
        let only_shadow = passes(
            "Shader \"T/X\" { SubShader { Pass { Tags { \"LightMode\"=\"ShadowCaster\" }
HLSLPROGRAM
#pragma vertex v
#pragma fragment f
ENDHLSL
 } } }",
        );
        assert_eq!(choose_pass(&only_shadow), None);
        let untagged = passes(
            "Shader \"T/U\" { SubShader { Pass {
HLSLPROGRAM
#pragma vertex v
#pragma fragment f
ENDHLSL
 } } }",
        );
        assert_eq!(choose_pass(&untagged), Some(0));
        let cg = passes(
            "Shader \"T/C\" { SubShader { Pass {
CGPROGRAM
#pragma vertex v
#pragma fragment f
ENDCG
 } } }",
        );
        assert_eq!(choose_pass(&cg), None, "CG passes are not compiled");
    }

    #[test]
    fn keywords_follow_the_material_and_the_first_option() {
        let p = passes(SHADER);
        let groups = &p[2].pass.program.as_ref().unwrap().keywords;
        let none: Vec<String> = Vec::new();
        assert_eq!(
            enabled_keywords(groups, &none),
            Vec::<String>::new(),
            "multi_compile takes its first option (`_`), shader_feature stays off"
        );
        let on = vec!["_B".to_owned(), "_NORMALMAP".to_owned()];
        assert_eq!(enabled_keywords(groups, &on), ["_B", "_NORMALMAP"]);
        let fog = passes(
            "Shader \"T/F\" { SubShader { Pass { Tags { \"LightMode\"=\"UniversalForward\" }
HLSLPROGRAM
#pragma vertex v
#pragma fragment f
#pragma multi_compile_fog
ENDHLSL
 } } }",
        );
        let groups = &fog[0].pass.program.as_ref().unwrap().keywords;
        assert_eq!(enabled_keywords(groups, &none), Vec::<String>::new());
        assert_eq!(
            enabled_keywords(groups, &["FOG_EXP".to_owned()]),
            ["FOG_EXP"]
        );
    }

    #[test]
    fn toggles_and_keyword_enums_become_keywords() {
        let shader = parse_shader(SHADER).unwrap();
        let values = |name: &str| match name {
            "_Fancy" => Some(1.0),
            "_Plain" => Some(1.0),
            _ => None,
        };
        assert_eq!(
            property_keywords(&shader.properties, values),
            ["_FANCY", "_PLAIN_ON", "_QUALITY_HIGH"],
            "the enum falls back to its default of 1"
        );
        assert!(
            property_keywords(&shader.properties, |_| Some(0.0))
                .iter()
                .all(|k| k.starts_with("_QUALITY"))
        );
    }

    #[test]
    fn render_state_reads_material_properties() {
        let p = passes(SHADER);
        let pass = &p[2];
        let numbers = |name: &str| match name {
            "_Cull" => Some(0.0),
            "_Src" => Some(5.0),
            "_Dst" => Some(10.0),
            "_ZWrite" => Some(0.0),
            _ => None,
        };
        let state = pass_state(&pass.state, numbers);
        assert_eq!(state.cull, CullMode::Off);
        assert!(!state.z_write);
        assert_eq!(state.color_mask, 7);
        assert_eq!(
            state.blend,
            Some(Blend {
                src_color: BlendFactor::SrcAlpha,
                dst_color: BlendFactor::OneMinusSrcAlpha,
                src_alpha: BlendFactor::One,
                dst_alpha: BlendFactor::OneMinusSrcAlpha,
            })
        );
        // Properties the material lacks leave Unity's defaults.
        let plain = pass_state(&pass.state, |_| None);
        assert_eq!(plain.cull, CullMode::Back);
        assert!(plain.z_write);
        assert_eq!(plain.blend, None);
        assert_eq!(color_mask("0"), 0);
        assert_eq!(color_mask("RGBA"), 15);
    }

    #[test]
    fn data_textures_are_linear_and_colour_textures_srgb() {
        let shader = parse_shader(SHADER).unwrap();
        let by = |n: &str| shader.properties.iter().find(|p| p.name == n).unwrap();
        assert!(!texture_is_srgb(by("_Detail")), "[Normal]");
        assert!(!texture_is_srgb(by("_BumpMap")), "default bump");
        assert!(texture_is_srgb(by("_Albedo")));
        assert!(is_texture(&by("_Albedo").kind) && !is_texture(&by("_Fancy").kind));
    }
}
