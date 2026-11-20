//! Compile every pass of every `.shader` of a Unity project and report what worked.
//!
//! ```text
//! cargo run --release -p rustiny-shader --example compile_project -- <project> <work-dir>
//! ```
//!
//! The project is only read. Package includes are mirrored and SPIR-V is cached under `<work-dir>`
//! (which must not be inside the project). Each pass is built with its default keywords (the first
//! option of each `multi_compile`, `shader_feature` off).

use rustiny_shader::{
    CompileError, Compiler, Language, Reflection, Request, ShaderLibrary, Stage, locate_dxc,
    mirror_packages, parse_shader, reflect,
};

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A pass's index and what compiling it gave: the reflection of both stages, or why not.
type Outcome = (usize, Result<[Reflection; 2], String>);

struct Job {
    shader: String,
    pass: String,
    source: String,
    vertex: String,
    fragment: String,
    dir: PathBuf,
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk(&path, out);
        } else if path.extension().is_some_and(|e| e == "shader") {
            out.push(path);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(project), Some(work)) = (args.first(), args.get(1)) else {
        eprintln!("usage: compile_project <project> <work-dir>");
        std::process::exit(2);
    };
    let (project, work) = (PathBuf::from(project), PathBuf::from(work));
    if work.starts_with(&project) {
        eprintln!("the work directory must not be inside the project");
        std::process::exit(2);
    }
    let Some(dxc) = locate_dxc() else {
        eprintln!("dxc not found (set RUSTINY_DXC or install the Vulkan SDK)");
        std::process::exit(1);
    };
    let mirror = work.join("mirror");
    match mirror_packages(&project, &mirror) {
        Ok(stats) => println!(
            "packages: {} found, {} copied, {} files",
            stats.packages, stats.copied, stats.files
        ),
        Err(e) => {
            eprintln!("cannot mirror packages: {e}");
            std::process::exit(1);
        }
    }

    let mut files = Vec::new();
    walk(&project.join("Assets"), &mut files);
    files.sort();
    let mut library = ShaderLibrary::new();
    let mut dirs = std::collections::HashMap::new();
    let mut parse_failures = 0;
    for file in &files {
        let text = std::fs::read_to_string(file).unwrap_or_default();
        match parse_shader(&text) {
            Ok(shader) => {
                dirs.insert(
                    shader.name.to_lowercase(),
                    file.parent().map(Path::to_path_buf).unwrap_or_default(),
                );
                library.add(shader);
            }
            Err(e) => {
                parse_failures += 1;
                println!("PARSE FAIL {}: {e}", file.display());
            }
        }
    }
    println!(
        "shaders: {} files, {} parsed, {} not parsed",
        files.len(),
        library.len(),
        parse_failures
    );

    let mut jobs = Vec::new();
    let (mut cg, mut no_program) = (0usize, 0usize);
    for shader in library.shaders() {
        let Some(sub) = shader.sub_shaders.first() else {
            continue;
        };
        for pass in library.resolve_passes(shader, sub) {
            let Some(program) = &pass.pass.program else {
                no_program += 1;
                continue;
            };
            if program.language == Language::Cg {
                cg += 1;
                continue;
            }
            let (Some(vertex), Some(fragment)) = (&program.vertex, &program.fragment) else {
                no_program += 1;
                continue;
            };
            jobs.push(Job {
                shader: shader.name.clone(),
                pass: pass.name.clone().unwrap_or_else(|| "(unnamed)".into()),
                source: program.source.clone(),
                vertex: vertex.clone(),
                fragment: fragment.clone(),
                dir: dirs
                    .get(&pass.from_shader.to_lowercase())
                    .cloned()
                    .unwrap_or_default(),
            });
        }
    }
    println!(
        "passes to compile: {} HLSL (skipped: {cg} CGPROGRAM, {no_program} without vertex+fragment)",
        jobs.len()
    );

    let compiler = Compiler::new(dxc, &mirror, work.join("work"));
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<Outcome>> = Mutex::new(Vec::new());
    let started = std::time::Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..8 {
            scope.spawn(|| {
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(job) = jobs.get(index) else { break };
                    let include_dirs = [job.dir.clone(), project.clone()];
                    let run = |entry: &str, stage| {
                        compiler.compile(&Request {
                            source: &job.source,
                            entry,
                            stage,
                            keywords: &[],
                            include_dirs: &include_dirs,
                        })
                    };
                    let vs = run(&job.vertex, Stage::Vertex);
                    let ps = run(&job.fragment, Stage::Fragment);
                    let outcome = match (vs, ps) {
                        (Ok(a), Ok(b)) => match (reflect(&a.spirv), reflect(&b.spirv)) {
                            (Ok(x), Ok(y)) => Ok([x, y]),
                            _ => Err("reflection failed".to_owned()),
                        },
                        (Err(CompileError::Rejected(e)), _) => Err(format!("VS: {e}")),
                        (_, Err(CompileError::Rejected(e))) => Err(format!("PS: {e}")),
                        (Err(e), _) | (_, Err(e)) => Err(e.to_string()),
                    };
                    if let Ok(mut all) = results.lock() {
                        all.push((index, outcome));
                    }
                }
            });
        }
    });
    let mut results = results.into_inner().unwrap_or_default();
    results.sort_by_key(|(i, _)| *i);
    let (mut ok, mut failed) = (0, 0);
    let mut blocks: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    let mut textures: BTreeMap<String, usize> = BTreeMap::new();
    let mut semantics: BTreeMap<String, usize> = BTreeMap::new();
    for (index, outcome) in &results {
        let job = &jobs[*index];
        match outcome {
            Ok(reflections) => {
                ok += 1;
                for (n, r) in reflections.iter().enumerate() {
                    for block in &r.blocks {
                        let entry = blocks.entry(block.name.clone()).or_default();
                        for member in &block.members {
                            *entry.entry(member.name.clone()).or_default() += 1;
                        }
                    }
                    for res in &r.resources {
                        *textures
                            .entry(format!("{:?} {}", res.kind, res.name))
                            .or_default() += 1;
                    }
                    if n == 0 {
                        for input in r.inputs.iter().filter(|i| !i.builtin) {
                            *semantics
                                .entry(input.semantic.clone().unwrap_or_default())
                                .or_default() += 1;
                        }
                    }
                }
            }
            Err(why) => {
                failed += 1;
                println!("FAIL {} / {}: {why}", job.shader, job.pass);
            }
        }
    }
    if std::env::args().any(|a| a == "--reflect") {
        println!("== vertex inputs (semantic: passes)");
        for (name, n) in &semantics {
            println!("  {name}: {n}");
        }
        println!("== constant buffers (block: members, how many stage compiles use each)");
        for (name, members) in &blocks {
            let list: Vec<String> = members.iter().map(|(m, n)| format!("{m}x{n}")).collect();
            println!("  [{name}] {} members: {}", members.len(), list.join(" "));
        }
        println!("== textures and samplers (kind name: stage compiles)");
        for (name, n) in &textures {
            println!("  {name}: {n}");
        }
    }
    println!(
        "compiled: {ok} ok, {failed} failed, of {} passes ({:.1} s)",
        jobs.len(),
        started.elapsed().as_secs_f64()
    );
}
