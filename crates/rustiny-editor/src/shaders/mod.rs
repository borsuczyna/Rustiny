//! The project's own Unity shaders, compiled on demand and handed to the renderer.
//!
//! [`ShaderHub`] is built for one open project and shared by the model import workers. The first
//! time a material needs a project shader it
//!
//! 1. finds the shader by GUID among the parsed `.shader` files of the project,
//! 2. picks the pass that draws it (`select::choose_pass`), the keywords the material enables
//!    and the render state its properties select,
//! 3. compiles both stages with DXC (cached on disk, outside the project), and
//! 4. records a pass and a material in a registry.
//!
//! The UI thread copies new registry entries to the renderer with [`ShaderHub::take_since`]
//! before it uploads the meshes that name them. Nothing here touches the GPU or the project's
//! files (they are only read), and a shader that cannot be compiled simply yields no material, so
//! the section keeps the native look.

mod globals;
mod lighting;
mod select;
mod volumes;

pub use globals::{AUTHORED_TIME_OF_DAY, daylight_visibility, project_globals};
pub use lighting::{SceneLightValues, ambient_of, gamma_to_linear, scene_light_values};
pub use volumes::{post_process_of, scene_volume_stack};

use rustiny_assets::{AssetDatabase, Guid};
use rustiny_render::{PropertyValue, ShaderMaterialDesc, ShaderPassDesc, TextureFallback};
use rustiny_scene::Material;
use rustiny_shader::{
    Compiler, Property, PropertyDefault, PropertyKind, Request, ResolvedPass, ShaderFile,
    ShaderLibrary, Stage, locate_dxc, mirror_packages, parse_shader,
};
use select::{
    choose_pass, enabled_keywords, is_texture, pass_state, property_keywords, texture_is_srgb,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// How much of the project's shader work succeeded, for the log and the smoke report.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HubStats {
    /// Shader files parsed.
    pub shaders: usize,
    /// Distinct pass variants compiled.
    pub compiled: usize,
    /// Pass variants DXC rejected.
    pub failed: usize,
    /// Materials registered (drawn with their shader).
    pub materials: usize,
}

/// Compiled SPIR-V of both stages.
#[derive(Debug)]
struct Variant {
    vertex: Vec<u32>,
    fragment: Vec<u32>,
}

/// Parsed shaders and the compiler; made once, on first use, on a worker thread.
struct Inner {
    compiler: Compiler,
    library: ShaderLibrary,
    /// Folder of each shader's file, by lower-case shader name (its own includes live there).
    dirs: HashMap<String, PathBuf>,
    /// Shader name by lower-case project-relative path.
    by_path: HashMap<String, String>,
    /// Shader name by GUID, for the shaders of packages (URP's `Lit`, `Unlit`, ...), which the
    /// asset database of `Assets/` does not know.
    by_guid: HashMap<String, String>,
}

#[derive(Default)]
struct Registry {
    passes: Vec<ShaderPassDesc>,
    pass_ids: HashMap<String, usize>,
    materials: Vec<ShaderMaterialDesc>,
    /// Registry id of each `.mat` (`None`: it cannot be drawn with a shader).
    material_ids: HashMap<String, Option<u32>>,
}

/// One variant being compiled once; `None` inside means dxc rejected it.
type VariantSlot = Arc<OnceLock<Option<Arc<Variant>>>>;

/// Compiles and registers the project's shaders.
pub struct ShaderHub {
    root: PathBuf,
    db: Arc<AssetDatabase>,
    cache: PathBuf,
    inner: OnceLock<Option<Inner>>,
    variants: Mutex<HashMap<String, VariantSlot>>,
    /// Messages already logged (one line per shader and reason, not per material).
    said: Mutex<std::collections::HashSet<String>>,
    registry: Mutex<Registry>,
    compiled: AtomicUsize,
    failed: AtomicUsize,
}

impl std::fmt::Debug for ShaderHub {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShaderHub")
            .field("root", &self.root)
            .finish_non_exhaustive()
    }
}

impl ShaderHub {
    /// A hub for the project at `root`. `cache` is a directory of ours (never inside the project)
    /// where package includes are mirrored and SPIR-V is cached. Nothing is read until the first
    /// [`material`](Self::material) call.
    pub fn new(root: PathBuf, db: Arc<AssetDatabase>, cache: PathBuf) -> Self {
        Self {
            root,
            db,
            cache,
            inner: OnceLock::new(),
            variants: Mutex::new(HashMap::new()),
            said: Mutex::new(std::collections::HashSet::new()),
            registry: Mutex::new(Registry::default()),
            compiled: AtomicUsize::new(0),
            failed: AtomicUsize::new(0),
        }
    }

    /// What has been compiled so far.
    pub fn stats(&self) -> HubStats {
        let (materials, shaders) = (
            self.registry.lock().map_or(0, |r| {
                r.material_ids.values().filter(|id| id.is_some()).count()
            }),
            self.inner
                .get()
                .and_then(Option::as_ref)
                .map_or(0, |i| i.library.len()),
        );
        HubStats {
            shaders,
            compiled: self.compiled.load(Ordering::Relaxed),
            failed: self.failed.load(Ordering::Relaxed),
            materials,
        }
    }

    fn inner(&self) -> Option<&Inner> {
        self.inner
            .get_or_init(|| init(&self.root, &self.db, &self.cache))
            .as_ref()
    }

    /// Log `message` the first time only.
    fn say(&self, message: String) {
        if let Ok(mut said) = self.said.lock()
            && said.insert(message.clone())
        {
            tracing::info!("{message}");
        }
    }

    fn path_of(&self, guid: &str) -> Option<String> {
        let guid: Guid = guid.parse().ok()?;
        self.db.by_guid(guid).map(|e| e.path.clone())
    }

    /// The registry id of the shader material for the `.mat` at `relative`, compiling its shader
    /// if needed. `None` when the material's shader is not one of the project's, is not HLSL, or
    /// does not compile. `texture` gives the renderer key of a project-relative PNG (queueing the
    /// load) for colour (`true`) or data (`false`).
    pub fn material(
        &self,
        material: &Material,
        relative: &str,
        texture: &dyn Fn(&str, bool) -> Option<u32>,
    ) -> Option<u32> {
        if let Ok(registry) = self.registry.lock()
            && let Some(known) = registry.material_ids.get(relative)
        {
            return *known;
        }
        let id = self.build(material, texture);
        if let Ok(mut registry) = self.registry.lock() {
            let id = id.map(|(pass, properties)| {
                registry.materials.push(ShaderMaterialDesc {
                    pass: pass as u32,
                    properties,
                });
                registry.materials.len() as u32 - 1
            });
            registry.material_ids.insert(relative.to_owned(), id);
            return id;
        }
        None
    }

    /// Registry entries added after the first `passes` passes and `materials` materials.
    pub fn take_since(
        &self,
        passes: usize,
        materials: usize,
    ) -> (Vec<ShaderPassDesc>, Vec<ShaderMaterialDesc>) {
        self.registry.lock().map_or_else(
            |_| (Vec::new(), Vec::new()),
            |r| {
                (
                    r.passes.get(passes..).unwrap_or_default().to_vec(),
                    r.materials.get(materials..).unwrap_or_default().to_vec(),
                )
            },
        )
    }

    /// Compile what `material` needs and register the pass: the registry pass id and the
    /// material's properties.
    fn build(
        &self,
        material: &Material,
        texture: &dyn Fn(&str, bool) -> Option<u32>,
    ) -> Option<(usize, Vec<(String, PropertyValue)>)> {
        let inner = self.inner()?;
        let guid = material.shader_guid.as_deref()?;
        let name = match self.path_of(guid) {
            Some(path) => inner.by_path.get(&path.to_ascii_lowercase())?,
            None => inner.by_guid.get(guid)?,
        };
        let Some(shader) = inner.library.get(name) else {
            self.say(format!("shader {name}: not in the library"));
            return None;
        };
        let Some(sub) = shader.sub_shaders.first() else {
            self.say(format!("shader {name}: no sub-shader, drawn natively"));
            return None;
        };
        let passes = inner.library.resolve_passes(shader, sub);
        let Some(index) = choose_pass(&passes) else {
            self.say(format!(
                "shader {name}: no HLSL pass to draw, drawn natively"
            ));
            return None;
        };
        let chosen = &passes[index];
        let program = chosen.pass.program.as_ref()?;
        self.say(format!(
            "shader {name}: drawing with pass {} ({} keyword groups)",
            chosen.name.as_deref().unwrap_or("(unnamed)"),
            program.keywords.len()
        ));

        let number = |property: &str| {
            material
                .floats
                .iter()
                .find(|(n, _)| n == property)
                .map(|(_, v)| *v)
        };
        let mut keywords = enabled_keywords(&program.keywords, &material.keywords);
        for keyword in property_keywords(&shader.properties, number) {
            if !keywords.contains(&keyword) {
                keywords.push(keyword);
            }
        }
        keywords.sort();
        let state = pass_state(&chosen.state, number);
        let variant = self.variant(inner, chosen, &keywords)?;

        let variant_key = variant_key(chosen, &keywords);
        let pass_key = format!("{variant_key}|{state:?}");
        let mut registry = self.registry.lock().ok()?;
        let pass = match registry.pass_ids.get(&pass_key) {
            Some(&id) => id,
            None => {
                registry.passes.push(ShaderPassDesc {
                    vertex: variant.vertex.clone(),
                    fragment: variant.fragment.clone(),
                    state,
                });
                let id = registry.passes.len() - 1;
                registry.pass_ids.insert(pass_key, id);
                id
            }
        };
        drop(registry);
        Some((pass, self.properties(shader, material, texture)))
    }

    /// The compiled variant of a pass, compiled by the first caller while others wait.
    fn variant(
        &self,
        inner: &Inner,
        pass: &ResolvedPass,
        keywords: &[String],
    ) -> Option<Arc<Variant>> {
        let key = variant_key(pass, keywords);
        let cell = self.variants.lock().ok()?.entry(key).or_default().clone();
        cell.get_or_init(|| {
            let compiled = self.compile(inner, pass, keywords);
            match &compiled {
                Some(_) => {
                    tracing::info!(
                        "shader {} pass {} compiled with [{}]",
                        pass.from_shader,
                        pass.name.as_deref().unwrap_or("(unnamed)"),
                        keywords.join(" ")
                    );
                    self.compiled.fetch_add(1, Ordering::Relaxed)
                }
                None => self.failed.fetch_add(1, Ordering::Relaxed),
            };
            compiled
        })
        .clone()
    }

    fn compile(
        &self,
        inner: &Inner,
        pass: &ResolvedPass,
        keywords: &[String],
    ) -> Option<Arc<Variant>> {
        let program = pass.pass.program.as_ref()?;
        let (vertex, fragment) = (program.vertex.as_deref()?, program.fragment.as_deref()?);
        let dir = inner
            .dirs
            .get(&pass.from_shader.to_lowercase())
            .cloned()
            .unwrap_or_default();
        let include_dirs = [dir, self.root.clone()];
        let run = |entry: &str, stage| {
            inner.compiler.compile(&Request {
                source: &program.source,
                entry,
                stage,
                keywords,
                include_dirs: &include_dirs,
            })
        };
        match (run(vertex, Stage::Vertex), run(fragment, Stage::Fragment)) {
            (Ok(v), Ok(f)) => Some(Arc::new(Variant {
                vertex: v.spirv,
                fragment: f.spirv,
            })),
            (Err(e), _) | (_, Err(e)) => {
                let text = e.to_string();
                tracing::warn!(
                    "shader {} pass {}: not compiled: {}",
                    pass.from_shader,
                    pass.name.as_deref().unwrap_or("(unnamed)"),
                    text.chars().take(500).collect::<String>()
                );
                None
            }
        }
    }

    /// The material's values for every property the shader declares (defaults for the rest).
    fn properties(
        &self,
        shader: &ShaderFile,
        material: &Material,
        texture: &dyn Fn(&str, bool) -> Option<u32>,
    ) -> Vec<(String, PropertyValue)> {
        shader
            .properties
            .iter()
            .map(|property| {
                let value = if is_texture(&property.kind) {
                    self.texture_value(property, material, texture)
                } else {
                    numeric_value(property, material)
                };
                (property.name.clone(), value)
            })
            .collect()
    }

    fn texture_value(
        &self,
        property: &Property,
        material: &Material,
        texture: &dyn Fn(&str, bool) -> Option<u32>,
    ) -> PropertyValue {
        let slot = material.textures.iter().find(|t| t.name == property.name);
        let named = slot.and_then(|t| match t.guid.as_deref() {
            // A `Texture2D` stored next to the material: the caller knows the file.
            None if t.file_id != 0 => Some(format!("#{}", t.file_id)),
            None => None,
            Some(guid) => {
                let path = self.path_of(guid)?;
                let lower = path.to_ascii_lowercase();
                if lower.ends_with(".png") {
                    Some(path)
                } else if lower.ends_with(".asset") {
                    // A texture inside an `.asset` with other objects is picked by its ID.
                    Some(if t.file_id == 2_800_000 || t.file_id == 0 {
                        path
                    } else {
                        format!("{path}#{}", t.file_id)
                    })
                } else {
                    None
                }
            }
        });
        let key = named.and_then(|path| texture(&path, texture_is_srgb(property)));
        let (scale, offset) = slot.map_or(([1.0; 2], [0.0; 2]), |t| (t.scale, t.offset));
        let fallback = match &property.default {
            PropertyDefault::Texture(name) => match name.to_ascii_lowercase().as_str() {
                "white" => TextureFallback::White,
                "black" => TextureFallback::Black,
                "bump" => TextureFallback::Bump,
                _ => TextureFallback::Grey,
            },
            _ => TextureFallback::Grey,
        };
        PropertyValue::Texture {
            key,
            scale_offset: [scale[0], scale[1], offset[0], offset[1]],
            fallback,
        }
    }
}

fn variant_key(pass: &ResolvedPass, keywords: &[String]) -> String {
    format!(
        "{}|{}|{}",
        pass.from_shader,
        pass.name.as_deref().unwrap_or(""),
        keywords.join(",")
    )
}

/// A float, vector or colour property: the material's value, else the shader's default. Colours
/// are converted to linear like the native path does (files store them as the inspector shows
/// them), except HDR ones.
fn numeric_value(property: &Property, material: &Material) -> PropertyValue {
    let find_color = || {
        material
            .colors
            .iter()
            .find(|(n, _)| n == &property.name)
            .map(|(_, c)| *c)
    };
    let value = match (&property.kind, &property.default) {
        (PropertyKind::Color, default) => {
            let stored = find_color().unwrap_or(match default {
                PropertyDefault::Vector(v) => *v,
                _ => [1.0; 4],
            });
            let hdr = property
                .attributes
                .iter()
                .any(|a| a.eq_ignore_ascii_case("HDR") || a.eq_ignore_ascii_case("Gamma"));
            if hdr {
                stored
            } else {
                let linear = |c: f32| c.clamp(0.0, 1.0).powf(2.2);
                [
                    linear(stored[0]),
                    linear(stored[1]),
                    linear(stored[2]),
                    stored[3],
                ]
            }
        }
        (PropertyKind::Vector, default) => find_color().unwrap_or(match default {
            PropertyDefault::Vector(v) => *v,
            _ => [0.0; 4],
        }),
        (_, default) => {
            let stored = material
                .floats
                .iter()
                .find(|(n, _)| n == &property.name)
                .map(|(_, v)| *v)
                .unwrap_or(match default {
                    PropertyDefault::Number(n) => *n,
                    _ => 0.0,
                });
            [stored, 0.0, 0.0, 0.0]
        }
    };
    PropertyValue::Vector(value)
}

fn init(root: &Path, db: &AssetDatabase, cache: &Path) -> Option<Inner> {
    let Some(dxc) = locate_dxc() else {
        tracing::info!(
            "dxc not found (set RUSTINY_DXC or install the Vulkan SDK): the project's own shaders \
             are not drawn"
        );
        return None;
    };
    let mirror = cache.join("mirror");
    match mirror_packages(root, &mirror) {
        Ok(stats) => tracing::info!(
            "shader includes: {} packages, {} copied ({} files)",
            stats.packages,
            stats.copied,
            stats.files
        ),
        Err(e) => {
            tracing::warn!("cannot mirror the project's shader packages: {e}");
            return None;
        }
    }
    let mut library = ShaderLibrary::new();
    let mut dirs = HashMap::new();
    let mut by_path = HashMap::new();
    for entry in db
        .entries()
        .iter()
        .filter(|e| !e.is_folder && e.path.to_ascii_lowercase().ends_with(".shader"))
    {
        let full = root.join(&entry.path);
        let Ok(text) = std::fs::read_to_string(&full) else {
            continue;
        };
        match parse_shader(&text) {
            Ok(shader) => {
                dirs.insert(
                    shader.name.to_lowercase(),
                    full.parent().map(Path::to_path_buf).unwrap_or_default(),
                );
                by_path.insert(entry.path.to_ascii_lowercase(), shader.name.clone());
                library.add(shader);
            }
            Err(e) => tracing::warn!("shader {} not parsed: {e}", entry.path),
        }
    }
    let project_shaders = library.len();
    let by_guid = add_package_shaders(root, &mut library, &mut dirs);
    tracing::info!(
        "{project_shaders} project shaders and {} package shaders parsed",
        by_guid.len()
    );
    Some(Inner {
        compiler: Compiler::new(dxc, mirror, cache.join("work")),
        library,
        dirs,
        by_path,
        by_guid,
    })
}

/// Parse the `.shader` files of the project's packages (`Library/PackageCache/<name>@<hash>`
/// and embedded `Packages/<name>`), read-only, and return their names by GUID (from the `.meta`
/// next to each file).
fn add_package_shaders(
    root: &Path,
    library: &mut ShaderLibrary,
    dirs: &mut HashMap<String, PathBuf>,
) -> HashMap<String, String> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
            } else if path
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("shader"))
            {
                out.push(path);
            }
        }
    }
    let mut files = Vec::new();
    for packages in [
        root.join("Library").join("PackageCache"),
        root.join("Packages"),
    ] {
        walk(&packages, &mut files);
    }
    let mut by_guid = HashMap::new();
    for file in files {
        let meta = PathBuf::from(format!("{}.meta", file.display()));
        let Some(guid) = std::fs::read_to_string(&meta).ok().and_then(|text| {
            text.lines()
                .find_map(|l| l.strip_prefix("guid:").map(|g| g.trim().to_owned()))
        }) else {
            continue;
        };
        let Ok(text) = std::fs::read_to_string(&file) else {
            continue;
        };
        match parse_shader(&text) {
            Ok(shader) => {
                dirs.insert(
                    shader.name.to_lowercase(),
                    file.parent().map(Path::to_path_buf).unwrap_or_default(),
                );
                by_guid.insert(guid, shader.name.clone());
                library.add(shader);
            }
            Err(e) => tracing::debug!("package shader {} not parsed: {e}", file.display()),
        }
    }
    by_guid
}
