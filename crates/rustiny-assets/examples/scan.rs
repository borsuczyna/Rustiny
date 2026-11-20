//! Read-only command line scan: `cargo run --release -p rustiny-assets --example scan -- <project>`.

use rustiny_assets::{AssetDatabase, UnityProject};
use std::process::ExitCode;

fn main() -> ExitCode {
    let Some(path) = std::env::args_os().nth(1) else {
        eprintln!("usage: scan <unity project folder>");
        return ExitCode::from(2);
    };
    let project = match UnityProject::detect(std::path::Path::new(&path)) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "project `{}`, Unity {}, package manifest: {}",
        project.name(),
        project.editor_version.as_deref().unwrap_or("?"),
        project.has_package_manifest
    );
    match AssetDatabase::scan(&project) {
        Ok(db) => {
            let s = db.stats();
            println!(
                "{} files, {} folders, {} metas parsed, {} issues, {:.0} ms",
                s.files,
                s.folders,
                s.metas_parsed,
                db.issues().len(),
                s.duration.as_secs_f64() * 1000.0
            );
            for issue in db.issues().iter().take(40) {
                println!("  {issue}");
            }
            match db.rescan(&project) {
                Ok(again) => {
                    let s = again.stats();
                    println!(
                        "rescan: {} metas read, {} reused, same index: {}, {:.0} ms",
                        s.metas_parsed,
                        s.metas_reused,
                        again.entries() == db.entries(),
                        s.duration.as_secs_f64() * 1000.0
                    );
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("rescan error: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
