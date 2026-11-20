//! Import every `.fbx` under a folder (read-only) and report failures and timing:
//! `cargo run --release -p rustiny-model --example import_all -- <folder>`.

use rustiny_model::{ImportSettings, fbx};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(&path, out);
        } else if path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("fbx"))
        {
            out.push(path);
        }
    }
}

fn main() {
    let Some(folder) = std::env::args().nth(1) else {
        eprintln!("usage: import_all <folder>");
        return;
    };
    let mut files = Vec::new();
    collect(Path::new(&folder), &mut files);
    files.sort();
    let start = Instant::now();
    let (mut ok, mut failed, mut bytes, mut triangles, mut vertices) = (0, 0, 0u64, 0usize, 0usize);
    let mut reasons: std::collections::BTreeMap<String, usize> = Default::default();
    let mut tallest = (0.0f32, PathBuf::new());
    for path in &files {
        let Ok(data) = std::fs::read(path) else {
            failed += 1;
            continue;
        };
        bytes += data.len() as u64;
        match fbx::import(&data, ImportSettings::default()) {
            Ok(mesh) => {
                ok += 1;
                triangles += mesh.triangle_count();
                vertices += mesh.positions.len();
                if let Some(b) = mesh.bounds() {
                    let size = (b.max[0] - b.min[0])
                        .max(b.max[1] - b.min[1])
                        .max(b.max[2] - b.min[2]);
                    if size > tallest.0 {
                        tallest = (size, path.clone());
                    }
                }
            }
            Err(e) => {
                failed += 1;
                *reasons.entry(e.to_string()).or_default() += 1;
            }
        }
    }
    println!(
        "{} files, {} imported, {} failed, {:.0} MiB read, {} vertices, {} triangles, {:.2} s",
        files.len(),
        ok,
        failed,
        bytes as f64 / 1048576.0,
        vertices,
        triangles,
        start.elapsed().as_secs_f64()
    );
    for (reason, n) in reasons {
        println!("  {n} x {reason}");
    }
    println!("largest extent {:.1} m: {}", tallest.0, tallest.1.display());
}
