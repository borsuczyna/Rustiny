//! Print what one model contains (sections, material names, bounds):
//! `cargo run --release -p rustiny-model --example fbx_info -- <file.fbx>`.

use rustiny_model::{ImportSettings, fbx};

fn main() {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: fbx_info <file.fbx>");
        return;
    };
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{path}: {e}");
            return;
        }
    };
    match fbx::import(&bytes, ImportSettings::default()) {
        Ok(mesh) => {
            println!(
                "{} vertices, {} triangles, bounds {:?}",
                mesh.positions.len(),
                mesh.triangle_count(),
                mesh.bounds()
            );
            for (i, s) in mesh.sections.iter().enumerate() {
                let name = s
                    .material
                    .and_then(|m| mesh.materials.get(m))
                    .map_or("-", String::as_str);
                println!(
                    "  section {i}: material {:?} `{name}`, {} indices",
                    s.material, s.index_count
                );
            }
        }
        Err(e) => eprintln!("{path}: {e}"),
    }
}
