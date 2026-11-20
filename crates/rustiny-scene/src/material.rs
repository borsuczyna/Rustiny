//! Reading the few `.mat` properties a viewport needs: base colour and texture, normal map,
//! metallic/smoothness, emission colour and how the surface treats alpha.
//!
//! Unity material files are UnityYAML like scenes. This reads the `Material` document line by
//! line and keeps `_BaseMap`/`_MainTex` (the albedo texture), `_BaseColor`/`_Color`, `_BumpMap`,
//! `_BumpScale`, `_MetallicGlossMap`, `_Metallic`, `_Smoothness`/`_Glossiness`, `_EmissionColor`,
//! `_Cutoff`, `_AlphaClip` and `_Surface`. Every float, colour and texture slot (with tiling) and
//! the enabled keywords are kept as well, for the project's own shaders, which read them by name.

use crate::unity_yaml::{file_id_of, flow_value, unquote};

/// Unity class ID of `Material` documents.
const MATERIAL_CLASS: &str = "21";
/// File ID Unity uses for the main object of a texture asset.
const TEXTURE_FILE_ID: i64 = 2_800_000;

/// A texture slot of a material: which texture and how it is tiled.
#[derive(Debug, Clone, PartialEq)]
pub struct MaterialTexture {
    /// Property name (`_BaseMap`).
    pub name: String,
    /// GUID of the texture asset, `None` when the slot is empty or refers to an object of the
    /// file the material itself is in (see [`file_id`](Self::file_id)).
    pub guid: Option<String>,
    /// File ID of the texture inside its asset: 2800000 for a texture file, anything else for a
    /// `Texture2D` stored next to the material in one `.asset`. 0 when the slot is empty.
    pub file_id: i64,
    /// `m_Scale` (tiling).
    pub scale: [f32; 2],
    /// `m_Offset`.
    pub offset: [f32; 2],
}

/// What a viewport needs to know about a material.
#[derive(Debug, Clone, PartialEq)]
pub struct Material {
    /// `m_Name`.
    pub name: String,
    /// GUID of the shader asset (which tells URP Lit from a custom shader).
    pub shader_guid: Option<String>,
    /// GUID of the albedo texture (`_BaseMap`, else `_MainTex`).
    pub base_map: Option<String>,
    /// `_BaseColor` (else `_Color`), as stored (not converted between colour spaces); white when
    /// the material sets none.
    pub color: [f32; 4],
    /// `_AlphaClip`: pixels below [`cutoff`](Self::cutoff) are discarded.
    pub alpha_clip: bool,
    /// `_Cutoff`.
    pub cutoff: f32,
    /// `_Surface` is 1 (transparent) rather than 0 (opaque).
    pub transparent: bool,
    /// GUID of the normal map (`_BumpMap`).
    pub bump_map: Option<String>,
    /// `_BumpScale`.
    pub bump_scale: f32,
    /// GUID of the metallic (R) and smoothness (A) map (`_MetallicGlossMap`).
    pub metallic_map: Option<String>,
    /// `_Metallic`; used when there is no metallic map.
    pub metallic: f32,
    /// `_Smoothness` (`_Glossiness` in older materials): a value, or the scale of the map's alpha.
    pub smoothness: f32,
    /// `_EmissionColor` without alpha, as stored (may exceed 1 for HDR colours).
    pub emission: [f32; 3],
    /// Every float (and integer) property in file order, for shaders of the project's own.
    pub floats: Vec<(String, f32)>,
    /// Every colour and vector property in file order.
    pub colors: Vec<(String, [f32; 4])>,
    /// Every texture slot in file order, empty ones included.
    pub textures: Vec<MaterialTexture>,
    /// Keywords the material enables (`m_ValidKeywords`, or the older `m_ShaderKeywords`).
    pub keywords: Vec<String>,
    /// `m_CustomRenderQueue`: `None` (stored as -1) takes the shader's queue.
    pub render_queue: Option<i32>,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            name: String::new(),
            shader_guid: None,
            base_map: None,
            color: [1.0; 4],
            alpha_clip: false,
            cutoff: 0.5,
            transparent: false,
            bump_map: None,
            bump_scale: 1.0,
            metallic_map: None,
            metallic: 0.0,
            smoothness: 0.5,
            emission: [0.0; 3],
            floats: Vec::new(),
            colors: Vec::new(),
            textures: Vec::new(),
            keywords: Vec::new(),
            render_queue: None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Block {
    Other,
    Textures,
    Floats,
    Colors,
    Keywords,
}

/// `{x: 1, y: 2}` as a pair.
fn pair(text: &str) -> Option<[f32; 2]> {
    Some([
        flow_value(text, "x")?.parse().ok()?,
        flow_value(text, "y")?.parse().ok()?,
    ])
}

/// Parse the `Material` document of a `.mat` file. `None` when the text has none.
pub fn parse_material(text: &str) -> Option<Material> {
    parse_material_object(text, None)
}

/// Parse one `Material` document: the one with `file_id` (a material stored as a sub-asset of an
/// `.asset` file next to other objects), or the first when `None`.
pub fn parse_material_object(text: &str, file_id: Option<i64>) -> Option<Material> {
    let mut lines = text.lines();
    // Skip to the Material document (the first one is often URP's version marker).
    loop {
        let line = lines.next()?;
        if let Some(rest) = line.strip_prefix("--- !u!") {
            let mut words = rest.split_whitespace();
            let class_matches = words.next() == Some(MATERIAL_CLASS);
            let id_matches = file_id.is_none_or(|wanted| {
                words
                    .next()
                    .and_then(|w| w.strip_prefix('&'))
                    .and_then(|w| w.parse::<i64>().ok())
                    == Some(wanted)
            });
            if class_matches && id_matches {
                break;
            }
        }
    }

    let mut material = Material::default();
    let (mut block, mut texture_key) = (Block::Other, "");
    let (mut base_map, mut main_tex) = (None, None);
    let (mut bump_map, mut metallic_map) = (None, None);
    let (mut smoothness, mut glossiness) = (None, None);
    let (mut base_color, mut color) = (None, None);
    for line in lines {
        if line.starts_with("--- ") {
            break;
        }
        let trimmed = line.trim();
        let indent = line.len() - line.trim_start().len();
        if let Some(entry) = trimmed.strip_prefix("- ") {
            if block == Block::Keywords {
                material.keywords.push(unquote(entry.trim()).into_owned());
                continue;
            }
            let Some((key, value)) = entry.split_once(':') else {
                continue;
            };
            let value = value.trim();
            match block {
                Block::Textures => {
                    texture_key = key;
                    material.textures.push(MaterialTexture {
                        name: key.to_owned(),
                        guid: None,
                        file_id: 0,
                        scale: [1.0; 2],
                        offset: [0.0; 2],
                    });
                }
                Block::Floats => {
                    let number = value.parse::<f32>().ok();
                    if let Some(v) = number {
                        material.floats.push((key.to_owned(), v));
                    }
                    match (key, number) {
                        ("_Cutoff", Some(v)) => material.cutoff = v,
                        ("_AlphaClip", Some(v)) => material.alpha_clip = v >= 0.5,
                        ("_Surface", Some(v)) => material.transparent = v >= 0.5,
                        ("_BumpScale", Some(v)) => material.bump_scale = v,
                        ("_Metallic", Some(v)) => material.metallic = v.clamp(0.0, 1.0),
                        ("_Smoothness", Some(v)) => smoothness = Some(v.clamp(0.0, 1.0)),
                        ("_Glossiness", Some(v)) => glossiness = Some(v.clamp(0.0, 1.0)),
                        _ => {}
                    }
                }
                Block::Colors => {
                    let channel = |name: &str| flow_value(value, name)?.parse::<f32>().ok();
                    if let (Some(r), Some(g), Some(b), Some(a)) =
                        (channel("r"), channel("g"), channel("b"), channel("a"))
                    {
                        material.colors.push((key.to_owned(), [r, g, b, a]));
                        match key {
                            "_BaseColor" => base_color = Some([r, g, b, a]),
                            "_Color" => color = Some([r, g, b, a]),
                            "_EmissionColor" => material.emission = [r, g, b],
                            _ => {}
                        }
                    }
                }
                Block::Other | Block::Keywords => {}
            }
        } else if indent == 2 {
            block = Block::Other;
            if let Some(name) = trimmed.strip_prefix("m_Name:") {
                material.name = unquote(name.trim()).into_owned();
            } else if let Some(shader) = trimmed.strip_prefix("m_Shader:") {
                material.shader_guid = flow_value(shader, "guid").map(str::to_owned);
            } else if let Some(list) = trimmed.strip_prefix("m_ValidKeywords:") {
                if list.trim().is_empty() {
                    block = Block::Keywords;
                }
            } else if let Some(list) = trimmed.strip_prefix("m_ShaderKeywords:") {
                material
                    .keywords
                    .extend(list.split_whitespace().map(str::to_owned));
            } else if let Some(queue) = trimmed.strip_prefix("m_CustomRenderQueue:") {
                material.render_queue = queue.trim().parse::<i32>().ok().filter(|q| *q >= 0);
            }
        } else if indent == 4 {
            block = match trimmed {
                "m_TexEnvs:" => Block::Textures,
                "m_Floats:" | "m_Ints:" => Block::Floats,
                "m_Colors:" => Block::Colors,
                _ => Block::Other,
            };
        } else if block == Block::Textures {
            if let Some(reference) = trimmed.strip_prefix("m_Texture:")
                && let Some(id) = file_id_of(reference).filter(|id| *id != 0)
                && flow_value(reference, "guid").is_none()
            {
                // A texture stored in the same file as the material.
                if let Some(slot) = material.textures.last_mut() {
                    slot.file_id = id;
                }
            } else if let Some(reference) = trimmed.strip_prefix("m_Texture:")
                && let Some(id) =
                    file_id_of(reference).filter(|id| *id != 0 && *id != TEXTURE_FILE_ID)
                && let Some(guid) = flow_value(reference, "guid")
            {
                // A texture object inside another asset: only the project's own shaders read it.
                if let Some(slot) = material.textures.last_mut() {
                    slot.guid = Some(guid.to_owned());
                    slot.file_id = id;
                }
            } else if let Some(reference) = trimmed.strip_prefix("m_Texture:")
                && file_id_of(reference) == Some(TEXTURE_FILE_ID)
                && let Some(guid) = flow_value(reference, "guid")
            {
                if let Some(slot) = material.textures.last_mut() {
                    slot.guid = Some(guid.to_owned());
                    slot.file_id = TEXTURE_FILE_ID;
                }
                match texture_key {
                    "_BaseMap" => base_map = Some(guid.to_owned()),
                    "_MainTex" => main_tex = Some(guid.to_owned()),
                    "_BumpMap" => bump_map = Some(guid.to_owned()),
                    "_MetallicGlossMap" => metallic_map = Some(guid.to_owned()),
                    _ => {}
                }
            } else if let Some(scale) = trimmed.strip_prefix("m_Scale:")
                && let Some(slot) = material.textures.last_mut()
            {
                slot.scale = pair(scale).unwrap_or(slot.scale);
            } else if let Some(offset) = trimmed.strip_prefix("m_Offset:")
                && let Some(slot) = material.textures.last_mut()
            {
                slot.offset = pair(offset).unwrap_or(slot.offset);
            }
        }
    }
    material.base_map = base_map.or(main_tex);
    material.bump_map = bump_map;
    material.metallic_map = metallic_map;
    material.smoothness = smoothness.or(glossiness).unwrap_or(0.5);
    material.color = base_color.or(color).unwrap_or([1.0; 4]);
    Some(material)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
--- !u!114 &-9016090471998378729
MonoBehaviour:
  m_Enabled: 1
  version: 10
--- !u!21 &2100000
Material:
  serializedVersion: 8
  m_Name: 'spruce-bark[wood]'
  m_Shader: {fileID: 4800000, guid: 9dad8ac6d58073b41a9d53d83923371d, type: 3}
  m_SavedProperties:
    serializedVersion: 3
    m_TexEnvs:
    - _BaseMap:
        m_Texture: {fileID: 2800000, guid: c20b987c3a38284429a6ff6091c699db, type: 3}
        m_Scale: {x: 1, y: 1}
    - _BumpMap:
        m_Texture: {fileID: 2800000, guid: 11111111111111111111111111111111, type: 3}
        m_Scale: {x: 1, y: 1}
    - _MainTex:
        m_Texture: {fileID: 2800000, guid: 22222222222222222222222222222222, type: 3}
    m_Floats:
    - _AlphaClip: 1
    - _Cutoff: 0.35
    - _Surface: 0
    m_Colors:
    - _BaseColor: {r: 0.5, g: 0.25, b: 1, a: 1}
    - _Color: {r: 0, g: 0, b: 0, a: 1}
--- !u!1 &9
GameObject:
  m_Name: after
";

    const EMBEDDED: &str = "%YAML 1.1
--- !u!21 &-100
Material:
  m_Name: first
  m_Shader: {fileID: 4800000, guid: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, type: 3}
  m_SavedProperties:
    m_TexEnvs:
    - _Control0:
        m_Texture: {fileID: -5}
        m_Scale: {x: 2, y: 2}
        m_Offset: {x: 0, y: 0}
    - _Layer0:
        m_Texture: {fileID: 2800000, guid: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb, type: 3}
    - _Layer1:
        m_Texture: {fileID: 0}
--- !u!21 &-200
Material:
  m_Name: second
  m_Shader: {fileID: 4800000, guid: cccccccccccccccccccccccccccccccc, type: 3}
  m_SavedProperties:
    m_TexEnvs:
    - _Control0:
        m_Texture: {fileID: -6}
";

    #[test]
    fn a_material_object_is_picked_by_its_file_id() {
        let first = parse_material_object(EMBEDDED, Some(-100)).unwrap();
        assert_eq!(first.name, "first");
        let second = parse_material_object(EMBEDDED, Some(-200)).unwrap();
        assert_eq!(second.name, "second");
        assert!(parse_material_object(EMBEDDED, Some(7)).is_none());
        assert_eq!(parse_material(EMBEDDED).unwrap().name, "first");
    }

    #[test]
    fn a_texture_stored_next_to_the_material_is_a_local_reference() {
        let material = parse_material_object(EMBEDDED, Some(-100)).unwrap();
        let slot = |name: &str| material.textures.iter().find(|t| t.name == name).unwrap();
        assert_eq!(
            (slot("_Control0").guid.clone(), slot("_Control0").file_id),
            (None, -5)
        );
        assert_eq!(slot("_Layer0").file_id, 2_800_000);
        assert!(slot("_Layer0").guid.is_some());
        assert_eq!(slot("_Layer1").file_id, 0);
        let other = parse_material_object(EMBEDDED, Some(-200)).unwrap();
        assert_eq!(other.textures[0].file_id, -6);
    }

    #[test]
    fn reads_the_properties_a_viewport_needs() {
        let m = parse_material(SAMPLE).unwrap();
        assert_eq!(m.name, "spruce-bark[wood]");
        assert_eq!(
            m.shader_guid.as_deref(),
            Some("9dad8ac6d58073b41a9d53d83923371d")
        );
        assert_eq!(
            m.base_map.as_deref(),
            Some("c20b987c3a38284429a6ff6091c699db"),
            "_BaseMap wins over _MainTex and other textures are ignored"
        );
        assert_eq!(
            m.color,
            [0.5, 0.25, 1.0, 1.0],
            "_BaseColor wins over _Color"
        );
        assert!(m.alpha_clip);
        assert_eq!(m.cutoff, 0.35);
        assert!(!m.transparent);
    }

    #[test]
    fn reads_normal_metallic_and_emission_properties() {
        let text = "--- !u!21 &1
Material:
  m_Name: Pbr
  m_SavedProperties:
    m_TexEnvs:
    - _BumpMap:
        m_Texture: {fileID: 2800000, guid: bb, type: 3}
    - _MetallicGlossMap:
        m_Texture: {fileID: 2800000, guid: cc, type: 3}
    - _OcclusionMap:
        m_Texture: {fileID: 2800000, guid: dd, type: 3}
    m_Floats:
    - _BumpScale: 0.7
    - _Metallic: 0.9
    - _Glossiness: 0.25
    m_Colors:
    - _EmissionColor: {r: 2, g: 1, b: 0.5, a: 1}
";
        let m = parse_material(text).unwrap();
        assert_eq!(m.bump_map.as_deref(), Some("bb"));
        assert_eq!(m.metallic_map.as_deref(), Some("cc"));
        assert_eq!((m.bump_scale, m.metallic, m.smoothness), (0.7, 0.9, 0.25));
        assert_eq!(m.emission, [2.0, 1.0, 0.5]);
        let bare = parse_material(
            "--- !u!21 &1
Material:
  m_Name: B
",
        )
        .unwrap();
        assert_eq!(
            (bare.bump_scale, bare.metallic, bare.smoothness),
            (1.0, 0.0, 0.5)
        );
        assert_eq!((bare.bump_map, bare.emission), (None, [0.0; 3]));
    }

    #[test]
    fn falls_back_to_main_tex_and_color_and_defaults() {
        let text = "--- !u!21 &1\nMaterial:\n  m_Name: Old\n  m_SavedProperties:\n    m_TexEnvs:\n    - _MainTex:\n        m_Texture: {fileID: 2800000, guid: aa, type: 3}\n    - _BaseMap:\n        m_Texture: {fileID: 0}\n    m_Colors:\n    - _Color: {r: 1, g: 0, b: 0, a: 0.5}\n    m_Floats:\n    - _Surface: 1\n";
        let m = parse_material(text).unwrap();
        assert_eq!(m.base_map.as_deref(), Some("aa"));
        assert_eq!(m.color, [1.0, 0.0, 0.0, 0.5]);
        assert!(m.transparent);
        assert!(!m.alpha_clip);
        assert_eq!(m.cutoff, 0.5, "default cutoff");

        let bare = parse_material("--- !u!21 &1\nMaterial:\n  m_Name: Bare\n").unwrap();
        assert_eq!(bare.base_map, None);
        assert_eq!(bare.color, [1.0; 4]);
    }

    #[test]
    fn keeps_every_property_for_custom_shaders() {
        let text = "--- !u!21 &1
Material:
  m_Name: Custom
  m_ValidKeywords:
  - _NORMALMAP
  - _ALPHATEST_ON
  m_InvalidKeywords: []
  m_CustomRenderQueue: 2450
  m_SavedProperties:
    serializedVersion: 3
    m_TexEnvs:
    - _MaskMap:
        m_Texture: {fileID: 2800000, guid: aa, type: 3}
        m_Scale: {x: 2, y: 3}
        m_Offset: {x: 0.5, y: -1}
    - _Empty:
        m_Texture: {fileID: 0}
        m_Scale: {x: 1, y: 1}
        m_Offset: {x: 0, y: 0}
    m_Ints:
    - _Mode: 2
    m_Floats:
    - _Tiling: 4
    m_Colors:
    - _Tint: {r: 1, g: 0.5, b: 0, a: 1}
";
        let m = parse_material(text).unwrap();
        assert_eq!(m.keywords, ["_NORMALMAP", "_ALPHATEST_ON"]);
        assert_eq!(m.render_queue, Some(2450));
        assert_eq!(m.floats, [("_Mode".into(), 2.0), ("_Tiling".into(), 4.0)]);
        assert_eq!(m.colors, [("_Tint".into(), [1.0, 0.5, 0.0, 1.0])]);
        assert_eq!(m.textures.len(), 2);
        assert_eq!(m.textures[0].name, "_MaskMap");
        assert_eq!(m.textures[0].guid.as_deref(), Some("aa"));
        assert_eq!(m.textures[0].scale, [2.0, 3.0]);
        assert_eq!(m.textures[0].offset, [0.5, -1.0]);
        assert_eq!(m.textures[1].guid, None);
        assert_eq!(m.textures[1].scale, [1.0, 1.0]);

        let old = parse_material(
            "--- !u!21 &1\nMaterial:\n  m_Name: Old\n  m_ShaderKeywords: _A _B\n  m_CustomRenderQueue: -1\n",
        )
        .unwrap();
        assert_eq!(old.keywords, ["_A", "_B"]);
        assert_eq!(old.render_queue, None);
    }

    #[test]
    fn text_without_a_material_is_none() {
        assert!(parse_material("").is_none());
        assert!(parse_material("--- !u!1 &1\nGameObject:\n  m_Name: x\n").is_none());
        assert!(parse_material("not yaml at all").is_none());
    }

    #[test]
    fn a_texture_that_is_not_a_texture_asset_is_ignored() {
        // fileID 0 means "no texture"; other file ids are sub-assets, not the texture itself.
        let text = "--- !u!21 &1\nMaterial:\n  m_SavedProperties:\n    m_TexEnvs:\n    - _BaseMap:\n        m_Texture: {fileID: 21300000, guid: aa, type: 3}\n";
        assert_eq!(parse_material(text).unwrap().base_map, None);
    }
}
