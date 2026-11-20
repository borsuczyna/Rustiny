//! HLSL to SPIR-V with DXC, and what Unity's HLSL needs around it.
//!
//! Unity's SRP shaders `#include "Packages/com.unity.render-pipelines.universal/..."`. Those files
//! live in the project's `Library/PackageCache/<name>@<hash>/`; [`mirror_packages`] copies their
//! HLSL into a directory of ours laid out as `Packages/<name>/...`, which is then an include root
//! for DXC. The project itself is only read.
//!
//! [`Compiler::compile`] writes the source to a work directory, runs `dxc` and caches the SPIR-V
//! by a hash of everything that went in, so an unchanged shader costs a file read the next time.

use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::Hasher;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use thiserror::Error;

/// Which stage an entry point is compiled for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Vertex shader (`vs_6_0`).
    Vertex,
    /// Fragment shader (`ps_6_0`).
    Fragment,
}

impl Stage {
    fn profile(self) -> &'static str {
        match self {
            Stage::Vertex => "vs_6_0",
            Stage::Fragment => "ps_6_0",
        }
    }

    fn define(self) -> &'static str {
        match self {
            Stage::Vertex => "SHADER_STAGE_VERTEX",
            Stage::Fragment => "SHADER_STAGE_FRAGMENT",
        }
    }
}

/// Why a shader did not compile.
#[derive(Debug, Error)]
pub enum CompileError {
    /// `dxc` could not be started or its files could not be written.
    #[error("compiler input/output failed: {0}")]
    Io(#[from] io::Error),
    /// DXC rejected the source; the text is its first error lines.
    #[error("{0}")]
    Rejected(String),
    /// DXC said nothing is wrong but produced no usable SPIR-V.
    #[error("DXC produced no SPIR-V")]
    NoOutput,
}

/// A compiled entry point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Compiled {
    /// The SPIR-V words.
    pub spirv: Vec<u32>,
    /// Whether the result came from the cache.
    pub cached: bool,
}

/// What [`mirror_packages`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MirrorStats {
    /// Packages found in the project.
    pub packages: usize,
    /// Packages that had at least one file copied this time.
    pub copied: usize,
    /// Files copied.
    pub files: usize,
}

/// Find `dxc`: `RUSTINY_DXC`, then `VULKAN_SDK/Bin`, then `PATH`.
pub fn locate_dxc() -> Option<PathBuf> {
    let exe = if cfg!(windows) { "dxc.exe" } else { "dxc" };
    if let Some(path) = std::env::var_os("RUSTINY_DXC").map(PathBuf::from)
        && path.is_file()
    {
        return Some(path);
    }
    if let Some(sdk) = std::env::var_os("VULKAN_SDK") {
        let path = Path::new(&sdk).join("Bin").join(exe);
        if path.is_file() {
            return Some(path);
        }
    }
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|dir| dir.join(exe))
            .find(|path| path.is_file())
    })
}

fn is_shader_include(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()).is_some_and(|e| {
        matches!(
            e.to_ascii_lowercase().as_str(),
            "hlsl" | "cginc" | "glslinc"
        )
    })
}

fn copy_includes(from: &Path, to: &Path, files: &mut usize) -> io::Result<()> {
    for entry in fs::read_dir(from)? {
        let entry = entry?;
        let path = entry.path();
        let kind = entry.file_type()?;
        if kind.is_dir() {
            copy_includes(&path, &to.join(entry.file_name()), files)?;
        } else if kind.is_file() && is_shader_include(&path) {
            let target = to.join(entry.file_name());
            // Files already there with the same size are kept; anything missing (a cleaned
            // temporary folder deletes old files one by one) or changed is copied again.
            let same = fs::metadata(&target)
                .ok()
                .is_some_and(|m| m.len() == entry.metadata().map_or(u64::MAX, |e| e.len()));
            if !same {
                fs::create_dir_all(to)?;
                fs::copy(&path, &target)?;
                *files += 1;
            }
        }
    }
    Ok(())
}

/// Copy the HLSL include files of the project's packages into `mirror_root/Packages/<name>/`.
///
/// Packages are taken from `Library/PackageCache/<name>@<hash>` and from embedded packages in
/// `Packages/<name>`. Every run checks each include file and copies only those that are missing
/// or have another size, so a partly deleted mirror heals itself. Only `mirror_root` is written.
///
/// # Errors
///
/// Any I/O error while copying.
pub fn mirror_packages(project_root: &Path, mirror_root: &Path) -> io::Result<MirrorStats> {
    let mut stats = MirrorStats::default();
    let target_root = mirror_root.join("Packages");
    fs::create_dir_all(&target_root)?;
    let sources = [
        project_root.join("Library").join("PackageCache"),
        project_root.join("Packages"),
    ];
    for source in sources {
        let Ok(entries) = fs::read_dir(&source) else {
            continue;
        };
        for entry in entries {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let folder = entry.file_name().to_string_lossy().into_owned();
            let name = folder.split('@').next().unwrap_or(&folder).to_owned();
            if !name.starts_with("com.") {
                continue;
            }
            stats.packages += 1;
            let marker = target_root.join(format!(".mirrored-{folder}"));
            let before = stats.files;
            copy_includes(&entry.path(), &target_root.join(&name), &mut stats.files)?;
            if !marker.exists() {
                fs::write(&marker, b"")?;
            }
            if stats.files > before {
                stats.copied += 1;
            }
        }
    }
    Ok(stats)
}

/// Everything one compile depends on.
#[derive(Debug, Clone, Copy)]
pub struct Request<'a> {
    /// The program text (a pass's `HLSLPROGRAM` with its includes).
    pub source: &'a str,
    /// Entry point name.
    pub entry: &'a str,
    /// Vertex or fragment.
    pub stage: Stage,
    /// Keywords switched on (each becomes `-D KEYWORD=1`).
    pub keywords: &'a [String],
    /// Extra include directories (the shader's own folder, the project root).
    pub include_dirs: &'a [PathBuf],
}

/// Runs `dxc`.
#[derive(Debug, Clone)]
pub struct Compiler {
    dxc: PathBuf,
    mirror_root: PathBuf,
    work_dir: PathBuf,
}

/// Changes when anything about how sources are compiled changes, so old cache entries are
/// not reused.
const COMPILE_RECIPE: &str = "v1 hv2018 vulkan1.3 SHADER_API_VULKAN UNITY_VERSION=60004";

impl Compiler {
    /// Compile with `dxc`; includes starting `Packages/` come from `mirror_root`, scratch files
    /// and the cache go below `work_dir`.
    pub fn new(
        dxc: impl Into<PathBuf>,
        mirror_root: impl Into<PathBuf>,
        work_dir: impl Into<PathBuf>,
    ) -> Self {
        Self {
            dxc: dxc.into(),
            mirror_root: mirror_root.into(),
            work_dir: work_dir.into(),
        }
    }

    /// Compile one entry point to SPIR-V.
    ///
    /// # Errors
    ///
    /// [`CompileError::Rejected`] with DXC's first errors when the source does not compile.
    pub fn compile(&self, request: &Request<'_>) -> Result<Compiled, CompileError> {
        let mut hasher = DefaultHasher::new();
        for part in [
            COMPILE_RECIPE,
            request.source,
            request.entry,
            request.stage.profile(),
        ] {
            hasher.write(part.as_bytes());
            hasher.write_u8(0);
        }
        let mut keywords: Vec<&String> = request.keywords.iter().collect();
        keywords.sort();
        for keyword in &keywords {
            hasher.write(keyword.as_bytes());
            hasher.write_u8(0);
        }
        // The include files are part of the input: the mirror's markers name the package
        // versions, so a package update starts a new cache generation.
        let mut markers: Vec<String> = fs::read_dir(self.mirror_root.join("Packages"))
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with(".mirrored-"))
            .collect();
        markers.sort();
        for marker in &markers {
            hasher.write(marker.as_bytes());
            hasher.write_u8(0);
        }
        for dir in request.include_dirs {
            hasher.write(dir.to_string_lossy().as_bytes());
            hasher.write_u8(0);
        }
        let key = format!("{:016x}", hasher.finish());
        let cache_dir = self.work_dir.join("spv");
        let cached_path = cache_dir.join(format!("{key}.spv"));
        if let Ok(bytes) = fs::read(&cached_path)
            && let Some(spirv) = words(&bytes)
        {
            return Ok(Compiled {
                spirv,
                cached: true,
            });
        }

        let src_dir = self.work_dir.join("src");
        fs::create_dir_all(&src_dir)?;
        fs::create_dir_all(&cache_dir)?;
        let src_path = src_dir.join(format!("{key}.hlsl"));
        // `#include_with_pragmas` is Unity's: the file's `#pragma`s count for the shader. DXC
        // knows only `#include`, and ignores the pragmas it does not know.
        fs::write(
            &src_path,
            request.source.replace("#include_with_pragmas", "#include"),
        )?;
        let out_path = cache_dir.join(format!("{key}.tmp"));

        let mut command = Command::new(&self.dxc);
        command
            .args(["-spirv", "-HV", "2018", "-fspv-target-env=vulkan1.3"])
            .args(["-T", request.stage.profile(), "-E", request.entry])
            .args(["-Wno-ignored-attributes", "-Wno-conversion"])
            .arg("-I")
            .arg(&self.mirror_root);
        for dir in request.include_dirs {
            command.arg("-I").arg(dir);
        }
        command
            .args(["-D", "SHADER_API_VULKAN=1", "-D", "UNITY_VERSION=60004"])
            .arg("-D")
            .arg(format!("{}=1", request.stage.define()));
        for keyword in &keywords {
            if keyword.as_str() != "_" {
                command.arg("-D").arg(format!("{keyword}=1"));
            }
        }
        command.arg(&src_path).arg("-Fo").arg(&out_path);
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            // CREATE_NO_WINDOW: no console flashes up when the editor runs the compiler.
            command.creation_flags(0x0800_0000);
        }
        let output = command.output()?;
        let mut text = String::from_utf8_lossy(&output.stderr).into_owned();
        text.push_str(&String::from_utf8_lossy(&output.stdout));
        if !output.status.success() {
            let _ = fs::remove_file(&out_path);
            return Err(CompileError::Rejected(first_errors(&text)));
        }
        let bytes = fs::read(&out_path).map_err(|_| CompileError::NoOutput)?;
        let spirv = words(&bytes).ok_or(CompileError::NoOutput)?;
        // A rename keeps readers from ever seeing half a file.
        fs::rename(&out_path, &cached_path)?;
        let _ = fs::remove_file(&src_path);
        Ok(Compiled {
            spirv,
            cached: false,
        })
    }
}

/// The error lines of DXC's output (at most four), shortened.
fn first_errors(text: &str) -> String {
    let lines: Vec<String> = text
        .lines()
        .filter(|l| l.contains("error:") || l.contains("fatal error"))
        .take(4)
        .map(|l| {
            // Drop the long path in front: `dir\file.hlsl:12:5: error: ...`.
            let trimmed = l.rsplit(['\\', '/']).next().unwrap_or(l);
            trimmed.trim().to_owned()
        })
        .collect();
    if lines.is_empty() {
        text.lines().take(3).collect::<Vec<_>>().join(" | ")
    } else {
        lines.join(" | ")
    }
}

fn words(bytes: &[u8]) -> Option<Vec<u32>> {
    if bytes.len() < 20 || !bytes.len().is_multiple_of(4) {
        return None;
    }
    let (chunks, _) = bytes.as_chunks::<4>();
    let spirv: Vec<u32> = chunks.iter().map(|c| u32::from_le_bytes(*c)).collect();
    (spirv.first() == Some(&0x0723_0203)).then_some(spirv)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("rustiny-shader-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn spirv_words_need_the_magic_number() {
        let mut bytes = vec![0u8; 20];
        assert!(words(&bytes).is_none());
        bytes[..4].copy_from_slice(&0x0723_0203u32.to_le_bytes());
        assert_eq!(words(&bytes).unwrap().len(), 5);
        assert!(words(&bytes[..18]).is_none());
        assert!(words(&[]).is_none());
    }

    #[test]
    fn error_text_is_cut_down_to_the_error_lines() {
        let text = "In file included from a.hlsl:1:\nC:\\x\\y\\z.hlsl:12:5: error: unknown type 'foo'\nmore\nC:/q/r.hlsl:3:1: fatal error: 'a.hlsl' file not found\n";
        assert_eq!(
            first_errors(text),
            "z.hlsl:12:5: error: unknown type 'foo' | r.hlsl:3:1: fatal error: 'a.hlsl' file not found"
        );
        assert_eq!(first_errors("plain failure"), "plain failure");
    }

    #[test]
    fn packages_are_mirrored_once_and_only_their_hlsl() {
        let project = temp("proj");
        let cache = project.join("Library").join("PackageCache");
        let pkg = cache.join("com.unity.render-pipelines.core@abc123");
        fs::create_dir_all(pkg.join("ShaderLibrary")).unwrap();
        fs::write(pkg.join("ShaderLibrary").join("Common.hlsl"), "// c").unwrap();
        fs::write(pkg.join("ShaderLibrary").join("Big.png"), "x").unwrap();
        fs::create_dir_all(cache.join("not-a-package@1")).unwrap();
        let mirror = temp("mirror");

        let first = mirror_packages(&project, &mirror).unwrap();
        assert_eq!((first.packages, first.copied, first.files), (1, 1, 1));
        let copied = mirror
            .join("Packages")
            .join("com.unity.render-pipelines.core")
            .join("ShaderLibrary");
        assert!(copied.join("Common.hlsl").is_file());
        assert!(
            !copied.join("Big.png").exists(),
            "only include files are copied"
        );
        let again = mirror_packages(&project, &mirror).unwrap();
        assert_eq!((again.packages, again.copied, again.files), (1, 0, 0));
        // A copy deleted behind its marker's back is made again.
        fs::remove_dir_all(
            mirror
                .join("Packages")
                .join("com.unity.render-pipelines.core"),
        )
        .unwrap();
        let restored = mirror_packages(&project, &mirror).unwrap();
        assert_eq!((restored.copied, restored.files), (1, 1));
        assert!(copied.join("Common.hlsl").is_file());
        // So is a single file deleted from a mirrored package.
        fs::remove_file(copied.join("Common.hlsl")).unwrap();
        let healed = mirror_packages(&project, &mirror).unwrap();
        assert_eq!((healed.copied, healed.files), (1, 1));
        assert!(copied.join("Common.hlsl").is_file());
        let _ = fs::remove_dir_all(project);
        let _ = fs::remove_dir_all(mirror);
    }

    /// Needs `dxc`; passes trivially (and says so) when it is not installed.
    #[test]
    fn a_small_shader_compiles_and_the_result_is_cached() {
        let Some(dxc) = locate_dxc() else {
            eprintln!("dxc not found: skipping");
            return;
        };
        let work = temp("work");
        let mirror = temp("mirror2");
        fs::create_dir_all(mirror.join("Packages")).unwrap();
        let compiler = Compiler::new(dxc, &mirror, &work);
        let source = "#include \"Packages/x/inc.hlsl\"\nfloat4 main(float4 p : POSITION) : SV_Position { return p * SCALE; }";
        let pkg = mirror.join("Packages").join("x");
        fs::create_dir_all(&pkg).unwrap();
        fs::write(
            pkg.join("inc.hlsl"),
            "#ifndef SCALE\n#define SCALE 1\n#endif\n",
        )
        .unwrap();
        let request = Request {
            source,
            entry: "main",
            stage: Stage::Vertex,
            keywords: &[],
            include_dirs: &[],
        };
        let first = compiler.compile(&request).unwrap();
        assert!(!first.cached);
        assert_eq!(first.spirv[0], 0x0723_0203);
        let second = compiler.compile(&request).unwrap();
        assert!(second.cached);
        assert_eq!(first.spirv, second.spirv);

        // A keyword changes the key and the output.
        let scaled = Request {
            keywords: &["SCALE".to_owned()],
            ..request
        };
        assert!(!compiler.compile(&scaled).unwrap().cached);

        // Errors come back as text, not panics.
        let broken = Request {
            source: "float4 main( : SV_Position { }",
            ..request
        };
        let Err(CompileError::Rejected(text)) = compiler.compile(&broken) else {
            panic!("a broken shader must be rejected");
        };
        assert!(text.contains("error"), "{text}");
        let _ = fs::remove_dir_all(work);
        let _ = fs::remove_dir_all(mirror);
    }
}
