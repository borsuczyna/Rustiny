//! Finding the `.mat` file a model's material stands for, and turning it into a [`Look`].
//!
//! Unity links a model's materials to project assets through the model importer's remap table
//! (`externalObjects`) or, when that is empty, through a name search. Most models of real projects
//! have an empty table, and their prefabs override the materials on the renderer instead. This
//! reads neither: it matches the model's material *name* against the names of the project's
//! `.mat` files and, when several share it, takes the one closest to the model's folder. That is a
//! heuristic and the coverage is reported (see [`ResolveStats`]), not assumed.
//!
//! Custom shaders give the same property names other meanings (a terrain painter's `_BaseMap` is
//! its weight map, drawn as a red field if taken for the albedo). Materials of a project shader
//! are therefore only read for the properties that shader declares; a shader declaring neither
//! `_BaseMap`/`_MainTex` nor `_BaseColor`/`_Color` gets the default grey. Shaders that are not
//! project files (URP's Lit, Simple Lit, Unlit) are the ones the renderer implements natively:
//! URP Lit's inputs (normal map, metallic, smoothness, emission) are read; Unlit is drawn unlit.

use super::convert::Look;
use super::plan::MeshSource;
use super::textures::TextureJob;
use crate::shaders::ShaderHub;
use rustiny_assets::{AssetDatabase, Guid};
use rustiny_model::MeshData;
use rustiny_render::Surface;
use rustiny_scene::{Material, parse_material_object};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};

/// Project files that are `.mat`, by lower-case name without extension.
#[derive(Debug, Default, Clone)]
pub struct MaterialIndex {
    root: PathBuf,
    by_name: HashMap<String, Vec<String>>,
    db: Option<Arc<AssetDatabase>>,
}

fn lower_stem(path: &str) -> Option<String> {
    let name = path.rsplit('/').next()?;
    let stem = name
        .strip_suffix(".mat")
        .or_else(|| name.strip_suffix(".MAT"))?;
    Some(stem.to_lowercase())
}

/// Directory components of a `/` separated path.
fn dirs_of(path: &str) -> Vec<&str> {
    let mut parts: Vec<&str> = path.split('/').collect();
    parts.pop();
    parts
}

impl MaterialIndex {
    /// Index every `.mat` of `db`; `root` is the project folder paths are relative to.
    pub fn build(root: PathBuf, db: Arc<AssetDatabase>) -> Self {
        let mut by_name: HashMap<String, Vec<String>> = HashMap::new();
        for entry in db.entries().iter().filter(|e| !e.is_folder) {
            if let Some(stem) = lower_stem(&entry.path) {
                by_name.entry(stem).or_default().push(entry.path.clone());
            }
        }
        Self {
            root,
            by_name,
            db: Some(db),
        }
    }

    /// Number of distinct material names.
    pub fn names(&self) -> usize {
        self.by_name.len()
    }

    /// The `.mat` (project-relative path) named `material` that lies closest to `model`: the most
    /// shared leading folders wins, then the shorter path, then the alphabetically first.
    pub fn find(&self, model: &Path, material: &str) -> Option<&str> {
        let candidates = self.by_name.get(&material.to_lowercase())?;
        let model_rel = model.strip_prefix(&self.root).unwrap_or(model);
        let model_rel = model_rel.to_string_lossy().replace('\\', "/");
        let model_dirs = dirs_of(&model_rel);
        candidates
            .iter()
            .max_by_key(|candidate| {
                let shared = dirs_of(candidate)
                    .iter()
                    .zip(&model_dirs)
                    .take_while(|(a, b)| a.eq_ignore_ascii_case(b))
                    .count();
                (
                    shared,
                    std::cmp::Reverse(candidate.len()),
                    std::cmp::Reverse((*candidate).clone()),
                )
            })
            .map(String::as_str)
    }

    /// The project-relative path a GUID names.
    fn path_of(&self, guid: &str) -> Option<String> {
        let guid: Guid = guid.parse().ok()?;
        self.db.as_ref()?.by_guid(guid).map(|e| e.path.clone())
    }
}

/// GUID of URP's Unlit shader.
const URP_UNLIT: &str = "650dd9526735d5b46b79224bc6e94025";

/// Which of the properties the viewport reads a shader declares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShaderProps {
    /// `_BaseMap` or `_MainTex`.
    pub base_map: bool,
    /// `_BaseColor` or `_Color`.
    pub base_color: bool,
    /// `_BumpMap`.
    pub bump_map: bool,
    /// `_MetallicGlossMap`.
    pub metallic_map: bool,
    /// `_Metallic`.
    pub metallic: bool,
    /// `_Smoothness` or `_Glossiness`.
    pub smoothness: bool,
    /// `_EmissionColor`.
    pub emission: bool,
}

impl ShaderProps {
    /// Shaders that are not project files (URP's, the built-in ones) have all of them.
    pub const ALL: Self = Self {
        base_map: true,
        base_color: true,
        bump_map: true,
        metallic_map: true,
        metallic: true,
        smoothness: true,
        emission: true,
    };
    /// A shader that declares none of them.
    #[cfg(test)]
    pub const NONE: Self = Self {
        base_map: false,
        base_color: false,
        bump_map: false,
        metallic_map: false,
        metallic: false,
        smoothness: false,
        emission: false,
    };
}

/// Whether `text` declares property `name` (as a whole word, e.g. `_BaseMap ("Base", 2D)`).
fn declares(text: &str, name: &str) -> bool {
    text.match_indices(name).any(|(at, _)| {
        let before = text[..at].chars().next_back();
        let after = text[at + name.len()..].chars().next();
        !before.is_some_and(|c| c.is_alphanumeric() || c == '_')
            && after.is_some_and(|c| c == '(' || (c.is_whitespace() && c != '\n'))
    })
}

/// The viewport's properties a ShaderLab file declares (looked for before its first `SubShader`).
pub fn declared_properties(shader_text: &str) -> ShaderProps {
    let head = shader_text
        .split_once("SubShader")
        .map_or(shader_text, |(head, _)| head);
    ShaderProps {
        base_map: declares(head, "_BaseMap") || declares(head, "_MainTex"),
        base_color: declares(head, "_BaseColor") || declares(head, "_Color"),
        bump_map: declares(head, "_BumpMap"),
        metallic_map: declares(head, "_MetallicGlossMap"),
        metallic: declares(head, "_Metallic"),
        smoothness: declares(head, "_Smoothness") || declares(head, "_Glossiness"),
        emission: declares(head, "_EmissionColor"),
    }
}

/// The renderer slot of each material of `mesh` (`None` for a material no triangle uses).
///
/// Unity gives a mesh one sub-mesh per material that is really used, in the order the materials
/// first appear among the triangles, and a renderer's material slot N belongs to sub-mesh N. The
/// model file's own material list can hold unused materials and be in another order, so a slot is
/// not an index into it: for a house whose file lists `wall, roof, window, door` but whose
/// triangles use `wall, window, door, roof` in that order, slot 1 is the window.
pub fn slot_of_materials(mesh: &MeshData) -> Vec<Option<usize>> {
    let mut slots = vec![None; mesh.materials.len()];
    for (slot, section) in mesh.sections.iter().enumerate() {
        if let Some(entry) = section.material.and_then(|m| slots.get_mut(m))
            && entry.is_none()
        {
            *entry = Some(slot);
        }
    }
    slots
}

/// Whether a texture file is a splat/weight map ("SH_Control0_ab12.png") rather than a picture.
fn is_weight_map(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path).to_lowercase();
    name.contains("_control") || name.starts_with("control")
}

/// The renderer keys of the textures a material uses (`None` for a texture it has none of).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LookTextures {
    /// Albedo (colour).
    pub albedo: Option<u32>,
    /// Normal map (data).
    pub normal: Option<u32>,
    /// Metallic in red, smoothness in alpha (data).
    pub metallic_map: Option<u32>,
}

/// How a parsed material is drawn. `textures` are the keys the textures will have in the renderer.
///
/// * The tint is the material colour converted to linear (the files store it as shown in the
///   inspector); emission is taken as stored (HDR colours are linear).
/// * Alpha-clipped materials discard below their cutoff; transparent ones approximately below 0.5
///   of the texture's alpha, because the pass has no blending for meshes. A transparent material
///   with no texture and a mostly transparent colour (clear glass) is not drawn at all.
/// * `unlit` draws the albedo without lighting (URP's Unlit shader).
pub fn look_of(material: &Material, textures: LookTextures, unlit: bool) -> Look {
    let linear = |c: f32| c.clamp(0.0, 1.0).powf(2.2);
    let texture = textures.albedo;
    let cutoff = if texture.is_none() {
        0.0
    } else if material.alpha_clip {
        material.cutoff.clamp(0.01, 1.0)
    } else if material.transparent {
        0.5
    } else {
        0.0
    };
    Look {
        surface: Surface {
            tint: [
                linear(material.color[0]),
                linear(material.color[1]),
                linear(material.color[2]),
            ],
            albedo: texture,
            normal: textures.normal,
            metallic_map: textures.metallic_map,
            cutoff,
            normal_scale: material.bump_scale,
            metallic: material.metallic,
            smoothness: material.smoothness,
            emission: material.emission.map(|c| c.max(0.0)),
            unlit,
            shader: None,
        },
        visible: !(material.transparent && texture.is_none() && material.color[3] < 0.35),
    }
}

/// How well names were matched, for the log and the smoke report.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResolveStats {
    /// Material slots looked at (one per material of every imported model).
    pub slots: usize,
    /// Slots for which a `.mat` of the same name was found.
    pub matched: usize,
    /// Distinct textures asked for.
    pub textures: usize,
}

/// A texture as the resolver tells textures apart: file, object inside it, colour or data.
type TextureId = (PathBuf, Option<i64>, bool);

/// Splits `file#id` into the file and the object ID (`None` without the suffix).
fn split_key(key: &str) -> (&str, Option<i64>) {
    match key.rsplit_once('#') {
        Some((file, id)) => match id.parse() {
            Ok(id) => (file, Some(id)),
            Err(_) => (key, None),
        },
        None => (key, None),
    }
}

/// Resolves the materials of imported models. Shared by the import worker threads.
pub struct Resolver {
    index: MaterialIndex,
    materials: Mutex<HashMap<String, Option<Arc<Material>>>>,
    shaders: Mutex<HashMap<String, ShaderProps>>,
    textures: Mutex<HashMap<TextureId, u32>>,
    jobs: Sender<TextureJob>,
    /// Compiles the project's own shaders; without it (no `dxc`) only the native look is used.
    hub: Option<Arc<ShaderHub>>,
    slots: AtomicUsize,
    matched: AtomicUsize,
}

impl std::fmt::Debug for Resolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Resolver")
            .field("names", &self.index.names())
            .finish_non_exhaustive()
    }
}

impl Resolver {
    /// Resolve against `index`; textures that are asked for are queued on `jobs`.
    pub fn new(index: MaterialIndex, jobs: Sender<TextureJob>) -> Self {
        Self {
            index,
            materials: Mutex::new(HashMap::new()),
            shaders: Mutex::new(HashMap::new()),
            textures: Mutex::new(HashMap::new()),
            jobs,
            hub: None,
            slots: AtomicUsize::new(0),
            matched: AtomicUsize::new(0),
        }
    }

    /// Draw materials of the project's own shaders with those shaders.
    #[must_use]
    pub fn with_hub(mut self, hub: Option<Arc<ShaderHub>>) -> Self {
        self.hub = hub;
        self
    }

    /// Distinct textures queued so far.
    pub fn submitted(&self) -> usize {
        self.textures.lock().map_or(0, |t| t.len())
    }

    /// Matching numbers so far.
    pub fn stats(&self) -> ResolveStats {
        ResolveStats {
            slots: self.slots.load(Ordering::Relaxed),
            matched: self.matched.load(Ordering::Relaxed),
            textures: self.submitted(),
        }
    }

    /// The material named by `key`: a project-relative `.mat` path, or `file.asset#id` for a
    /// `Material` object inside an `.asset`.
    fn material(&self, key: &str) -> Option<Arc<Material>> {
        if let Ok(cache) = self.materials.lock()
            && let Some(known) = cache.get(key)
        {
            return known.clone();
        }
        let (file, id) = split_key(key);
        let parsed = std::fs::read_to_string(self.index.root.join(file))
            .ok()
            .and_then(|text| parse_material_object(&text, id))
            .map(Arc::new);
        if let Ok(mut cache) = self.materials.lock() {
            cache.insert(key.to_owned(), parsed.clone());
        }
        parsed
    }

    /// What the shader a material uses declares (everything, for shaders that are not project
    /// files).
    fn shader_props(&self, guid: Option<&str>) -> ShaderProps {
        let Some(guid) = guid else {
            return ShaderProps::ALL;
        };
        if let Ok(cache) = self.shaders.lock()
            && let Some(known) = cache.get(guid)
        {
            return *known;
        }
        let props = self
            .index
            .path_of(guid)
            .filter(|path| path.to_ascii_lowercase().ends_with(".shader"))
            .and_then(|path| std::fs::read_to_string(self.index.root.join(path)).ok())
            .map_or(ShaderProps::ALL, |text| declared_properties(&text));
        if let Ok(mut cache) = self.shaders.lock() {
            cache.insert(guid.to_owned(), props);
        }
        props
    }

    /// The renderer key of the texture at `relative` (queueing it on first use). The same file
    /// used as a colour and as data is two textures.
    fn texture_key(&self, relative: &str, srgb: bool) -> Option<u32> {
        let (file, sub) = split_key(relative);
        let path = self.index.root.join(file);
        let mut known = self.textures.lock().ok()?;
        if let Some(&key) = known.get(&(path.clone(), sub, srgb)) {
            return Some(key);
        }
        let key = u32::try_from(known.len()).ok()?;
        known.insert((path.clone(), sub, srgb), key);
        // The receiver is gone when the scene was replaced; the key is then never used.
        let _ = self.jobs.send(TextureJob {
            key,
            path,
            sub,
            srgb,
        });
        Some(key)
    }

    /// Project-relative path of the PNG a texture reference names, when the shader uses it.
    fn png_of(&self, guid: Option<&str>, declared: bool) -> Option<String> {
        guid.filter(|_| declared)
            .and_then(|guid| self.index.path_of(guid))
            .filter(|path| path.to_ascii_lowercase().ends_with(".png"))
    }

    /// The project-relative form of an absolute path inside the project.
    fn relative(&self, path: &Path) -> Option<String> {
        let relative = path.strip_prefix(&self.index.root).ok()?;
        Some(relative.to_string_lossy().replace('\\', "/"))
    }

    /// The shader hub's id for `material` (read from `key`), when its shader is one of the
    /// project's and compiles.
    fn hub_material_of(&self, material: &Material, key: &str) -> Option<u32> {
        let hub = self.hub.as_ref()?;
        hub.material(material, key, &|path, srgb| {
            // `#id` is a texture stored in the material's own file.
            match path.strip_prefix('#') {
                Some(id) => self.texture_key(&format!("{}#{id}", split_key(key).0), srgb),
                None => self.texture_key(path, srgb),
            }
        })
    }

    /// The shader hub's id for the material asset with `guid` (its `fileID` is `file_id` when
    /// the material is an object inside an `.asset`). Reads the file: call it off the UI thread.
    pub fn hub_material(&self, guid: &str, file_id: i64) -> Option<u32> {
        let path = self.index.path_of(guid)?;
        let key = if path.to_ascii_lowercase().ends_with(".mat") {
            path
        } else {
            format!("{path}#{file_id}")
        };
        let material = self.material(&key)?;
        self.hub_material_of(&material, &key)
    }

    /// One [`Look`] per material of `mesh`, in the order of [`MeshData::materials`]. A material
    /// the renderer names a `.mat` for (by its slot, see [`slot_of_materials`]) uses it; the
    /// others are matched by the model's material name.
    pub fn looks(&self, source: &MeshSource, mesh: &MeshData) -> Vec<Look> {
        let slots = slot_of_materials(mesh);
        mesh.materials
            .iter()
            .enumerate()
            .map(|(index, name)| {
                self.slots.fetch_add(1, Ordering::Relaxed);
                let assigned = slots[index]
                    .and_then(|slot| source.materials.get(slot))
                    .and_then(Option::as_ref)
                    .and_then(|reference| {
                        let file = self.relative(&reference.path)?;
                        Some(match reference.file_id {
                            Some(id) => format!("{file}#{id}"),
                            None => file,
                        })
                    });
                let Some(relative) =
                    assigned.or_else(|| self.index.find(&source.path, name).map(str::to_owned))
                else {
                    return Look::DEFAULT;
                };
                let relative = relative.as_str();
                let Some(material) = self.material(relative) else {
                    return Look::DEFAULT;
                };
                self.matched.fetch_add(1, Ordering::Relaxed);
                // A shader of the project: compile it and draw with it (its id is the hub's; the
                // UI thread turns it into a renderer id before the mesh is uploaded).
                if let Some(id) = self.hub_material_of(&material, relative) {
                    return Look {
                        surface: Surface {
                            shader: Some(id),
                            ..Look::DEFAULT.surface
                        },
                        visible: true,
                    };
                }
                let props = self.shader_props(material.shader_guid.as_deref());
                if !props.base_map && !props.base_color {
                    return Look::DEFAULT;
                }
                let mut material = (*material).clone();
                if !props.base_color {
                    material.color = [1.0; 4];
                }
                if !props.metallic {
                    material.metallic = 0.0;
                }
                if !props.smoothness {
                    material.smoothness = 0.0;
                }
                if !props.emission {
                    material.emission = [0.0; 3];
                }
                let base_map = self.png_of(material.base_map.as_deref(), props.base_map);
                if base_map.as_deref().is_some_and(is_weight_map) {
                    // A terrain-mixing material: its "base map" holds the weights of the layers
                    // the custom shader blends, which this viewport does not evaluate.
                    return Look::DEFAULT;
                }
                let normal = self.png_of(material.bump_map.as_deref(), props.bump_map);
                let mask = self.png_of(material.metallic_map.as_deref(), props.metallic_map);
                let textures = LookTextures {
                    albedo: base_map.and_then(|path| self.texture_key(&path, true)),
                    normal: normal.and_then(|path| self.texture_key(&path, false)),
                    metallic_map: mask.and_then(|path| self.texture_key(&path, false)),
                };
                let unlit = material.shader_guid.as_deref() == Some(URP_UNLIT);
                look_of(&material, textures, unlit)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_renderer_slot_is_the_position_of_the_sub_mesh_not_of_the_file_material() {
        // The file lists wall, roof, window, door; the triangles use wall, window, door, roof.
        let mesh = MeshData {
            materials: ["wall", "roof", "window", "door", "unused"]
                .map(String::from)
                .to_vec(),
            sections: [0, 2, 3, 1]
                .map(|material| rustiny_model::Section {
                    material: Some(material),
                    first_index: 0,
                    index_count: 3,
                })
                .to_vec(),
            ..MeshData::default()
        };
        assert_eq!(
            slot_of_materials(&mesh),
            vec![Some(0), Some(3), Some(1), Some(2), None]
        );
    }

    fn index(paths: &[&str]) -> MaterialIndex {
        let mut by_name: HashMap<String, Vec<String>> = HashMap::new();
        for path in paths {
            by_name
                .entry(lower_stem(path).unwrap())
                .or_default()
                .push((*path).to_owned());
        }
        MaterialIndex {
            root: PathBuf::from("/proj"),
            by_name,
            db: None,
        }
    }

    #[test]
    fn names_match_case_insensitively_and_ignore_other_files() {
        let i = index(&["Assets/Materials/Tynk.mat"]);
        let model = Path::new("/proj/Assets/Meshes/a.fbx");
        assert_eq!(i.find(model, "tynk"), Some("Assets/Materials/Tynk.mat"));
        assert_eq!(i.find(model, "TYNK"), Some("Assets/Materials/Tynk.mat"));
        assert_eq!(i.find(model, "beton"), None);
        assert_eq!(lower_stem("Assets/a.png"), None);
        assert_eq!(i.names(), 1);
    }

    #[test]
    fn the_material_closest_to_the_model_wins() {
        let i = index(&[
            "Assets/Materials/Other/wood.mat",
            "Assets/Materials/SHMapImport/wood.mat",
            "Assets/Materials/wood.mat",
        ]);
        let model = Path::new("/proj/Assets/Materials/SHMapImport/Current/x.fbx");
        assert_eq!(
            i.find(model, "wood"),
            Some("Assets/Materials/SHMapImport/wood.mat")
        );
        let elsewhere = Path::new("/proj/Assets/Meshes/x.fbx");
        assert_eq!(
            i.find(elsewhere, "wood"),
            Some("Assets/Materials/wood.mat"),
            "no folder in common: the shortest path"
        );
    }

    #[test]
    fn shaders_declare_the_properties_they_use() {
        let terrain = "Shader \"Custom/MeshTerrain\" {
    Properties {
        _Control0 (\"Weights\", 2D) = \"red\" {}
        _Tint0 (\"Tint\", Color) = (1,1,1,1)
    }
    SubShader { _BaseMap }
}";
        assert_eq!(
            declared_properties(terrain),
            ShaderProps::NONE,
            "names used after SubShader do not count"
        );
        let lit = "Shader \"X\" {
 Properties {
  [MainTexture] _BaseMap(\"Albedo\", 2D) = \"white\" {}
  _Color (\"C\", Color) = (1,1,1,1)
 }
 SubShader {}
}";
        assert_eq!(
            declared_properties(lit),
            ShaderProps {
                base_map: true,
                base_color: true,
                ..ShaderProps::NONE
            }
        );
        let pbr = "Properties { _BumpMap (\"n\", 2D) = \"bump\" {}
 _MetallicGlossMap (\"m\", 2D) = \"white\" {} _Metallic (\"m\", Range(0,1)) = 0
 _Glossiness (\"s\", Range(0,1)) = 0.5 [HDR] _EmissionColor (\"e\", Color) = (0,0,0) } SubShader {}";
        assert_eq!(
            declared_properties(pbr),
            ShaderProps {
                bump_map: true,
                metallic_map: true,
                metallic: true,
                smoothness: true,
                emission: true,
                ..ShaderProps::NONE
            }
        );
        // A longer name that merely starts with the property is not the property.
        let other = "Properties { _BaseMapScale (\"s\", Float) = 1
 _MyColor (\"c\", Color) = (1,1,1,1) } SubShader {}";
        assert_eq!(declared_properties(other), ShaderProps::NONE);
    }

    #[test]
    fn weight_maps_are_recognised_by_name() {
        assert!(is_weight_map("Assets/Textures/SH_Control0_05627e5c.png"));
        assert!(is_weight_map("Assets/T/control1.png"));
        assert!(!is_weight_map("Assets/Textures/red-paint.png"));
        assert!(!is_weight_map("Assets/Controller/wall.png"));
    }

    fn albedo(key: u32) -> LookTextures {
        LookTextures {
            albedo: Some(key),
            ..LookTextures::default()
        }
    }

    fn material(color: [f32; 4]) -> Material {
        Material {
            color,
            ..Material::default()
        }
    }

    #[test]
    fn opaque_materials_get_a_linear_tint_and_no_cutoff() {
        let look = look_of(&material([0.5, 1.0, 0.0, 1.0]), albedo(3), false);
        let surface = look.surface;
        assert!((surface.tint[0] - 0.5_f32.powf(2.2)).abs() < 1e-6);
        assert_eq!((surface.tint[1], surface.tint[2]), (1.0, 0.0));
        assert_eq!(
            (surface.albedo, surface.cutoff, look.visible),
            (Some(3), 0.0, true)
        );
    }

    #[test]
    fn alpha_clip_and_transparency_decide_the_cutoff_and_visibility() {
        let clipped = Material {
            alpha_clip: true,
            cutoff: 0.3,
            ..Material::default()
        };
        assert_eq!(look_of(&clipped, albedo(1), false).surface.cutoff, 0.3);
        assert_eq!(
            look_of(&clipped, LookTextures::default(), false)
                .surface
                .cutoff,
            0.0,
            "no texture, nothing to clip"
        );

        let glass = Material {
            transparent: true,
            color: [1.0, 1.0, 1.0, 0.2],
            ..Material::default()
        };
        let none = LookTextures::default();
        assert!(
            !look_of(&glass, none, false).visible,
            "clear glass is not drawn"
        );
        assert_eq!(look_of(&glass, albedo(1), false).surface.cutoff, 0.5);
        assert!(look_of(&glass, albedo(1), false).visible);
        let tinted = Material {
            color: [1.0, 1.0, 1.0, 0.8],
            ..glass
        };
        assert!(look_of(&tinted, none, false).visible, "mostly opaque stays");
    }

    #[test]
    fn pbr_inputs_and_unlit_reach_the_surface() {
        let pbr = Material {
            bump_scale: 0.5,
            metallic: 0.8,
            smoothness: 0.6,
            emission: [0.5, -1.0, 2.0],
            ..Material::default()
        };
        let textures = LookTextures {
            albedo: Some(1),
            normal: Some(2),
            metallic_map: Some(3),
        };
        let surface = look_of(&pbr, textures, true).surface;
        assert_eq!(
            (surface.normal, surface.metallic_map, surface.unlit),
            (Some(2), Some(3), true)
        );
        assert_eq!(
            (surface.normal_scale, surface.metallic, surface.smoothness),
            (0.5, 0.8, 0.6)
        );
        assert_eq!(
            surface.emission,
            [0.5, 0.0, 2.0],
            "negative emission is cut"
        );
    }
}
