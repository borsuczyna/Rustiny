//! Read every `.asset` under a directory as a Unity Mesh and report what worked (read-only).
//!
//! ```text
//! cargo run --release -p rustiny-model --example check_mesh_assets -- <dir>
//! ```

use rustiny_model::{MeshAssetError, import_mesh_asset};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk(&path, out);
        } else if path.extension().is_some_and(|e| e == "asset") {
            out.push(path);
        }
    }
}

fn main() {
    let Some(dir) = std::env::args().nth(1) else {
        eprintln!("usage: check_mesh_assets <dir>");
        std::process::exit(2);
    };
    let mut files = Vec::new();
    walk(Path::new(&dir), &mut files);
    let (mut meshes, mut triangles, mut skipped) = (0usize, 0usize, 0usize);
    let mut failures: BTreeMap<String, usize> = BTreeMap::new();
    for file in &files {
        let Ok(text) = std::fs::read_to_string(file) else {
            skipped += 1;
            continue;
        };
        match import_mesh_asset(&text) {
            Ok(mesh) => {
                meshes += 1;
                triangles += mesh.indices.len() / 3;
            }
            Err(MeshAssetError::NotAMesh) => skipped += 1,
            Err(e) => {
                *failures.entry(e.to_string()).or_default() += 1;
                eprintln!("{}: {e}", file.display());
            }
        }
    }
    println!(
        "{} .asset files: {meshes} meshes read ({triangles} triangles), {skipped} not meshes or unreadable",
        files.len()
    );
    for (why, n) in failures {
        println!("  {n} x {why}");
    }
}
