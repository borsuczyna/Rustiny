//! `rustiny` - the Rustiny editor executable.

use rustiny_core::config::EditorConfig;
use rustiny_core::logging::{self, LogOptions};
use rustiny_core::paths::UserDirs;
use rustiny_editor::app::{self, RunError, RunInputs};
use rustiny_editor::cli::{CliOptions, EXIT_SKIPPED, USAGE};
use std::process::ExitCode;

fn main() -> ExitCode {
    let process_start = std::time::Instant::now();
    let opts = match CliOptions::parse(std::env::args().skip(1)) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("error: {e}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    if opts.help {
        print!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    if opts.version {
        println!("rustiny {}", rustiny_core::ENGINE_VERSION);
        return ExitCode::SUCCESS;
    }

    // Smoke runs use a private directory so they never read or write the user's real settings.
    let smoke_dir = opts
        .smoke_frames
        .filter(|_| opts.config_dir.is_none())
        .map(|_| std::env::temp_dir().join(format!("rustiny-smoke-{}", std::process::id())));
    let dirs = match (&opts.config_dir, &smoke_dir) {
        (Some(dir), _) | (None, Some(dir)) => UserDirs::rooted_at(dir.clone()),
        (None, None) => match UserDirs::resolve() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::FAILURE;
            }
        },
    };

    let (config, config_warning) = EditorConfig::load_or_default(&dirs.config_file());
    let log_handle = match logging::init(&LogOptions {
        filter: config.logging.filter.clone(),
        file: config.logging.file.then(|| dirs.logs.join("rustiny.log")),
        ..LogOptions::default()
    }) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    logging::install_panic_hook();
    tracing::info!("Rustiny {} starting", rustiny_core::ENGINE_VERSION);
    if let Some(warning) = config_warning {
        tracing::warn!("configuration problem, using defaults: {warning}");
    }
    if let Some(file) = &log_handle.file {
        tracing::info!("logging to {}", file.display());
    }

    let smoke = opts.smoke_frames;
    let result = app::run(RunInputs {
        opts,
        config,
        dirs,
        log: log_handle.buffer,
        process_start,
    });

    if let Some(dir) = smoke_dir {
        let _ = std::fs::remove_dir_all(dir);
    }

    match (result, smoke) {
        (Ok(report), Some(target)) => {
            let ok = report.validation_errors == 0 && report.frames >= u64::from(target);
            let scans = report.project_scans;
            let project = report
                .project
                .map_or(String::new(), |(files, folders, issues)| {
                    format!(
                        " project_files={files} project_folders={folders} \
                         project_issues={issues} project_scans={scans}"
                    )
                });
            let scene = report
                .scene
                .map_or(String::new(), |(objects, roots, issues)| {
                    format!(" scene_objects={objects} scene_roots={roots} scene_issues={issues}")
                });
            let meshes = report.meshes.map_or(String::new(), |m| {
                format!(
                    concat!(
                        " meshes_planned={} meshes_uploaded={} meshes_failed={}",
                        " mesh_triangles_drawn={} mesh_draw_calls={} material_slots={}",
                        " materials_matched={} textures_requested={} textures_uploaded={}",
                        " textures_failed={} texture_mb={:.1}"
                    ),
                    m.planned,
                    m.uploaded,
                    m.failed,
                    m.triangles_drawn,
                    m.draw_calls,
                    m.material_slots,
                    m.materials_matched,
                    m.textures_requested,
                    m.textures_uploaded,
                    m.textures_failed,
                    m.texture_bytes as f64 / 1e6
                )
            });
            println!(
                "SMOKE_RESULT ok={ok} frames={} validation_errors={} validation_warnings={} \
                 environment_errors={} swapchain_recreations={} scene_gpu_passes={} scene_gpu_ms={} gpu=\"{}\"{}{}{}",
                report.frames,
                report.validation_errors,
                report.validation_warnings,
                report.environment_errors,
                report.swapchain_recreations,
                report.scene_gpu_passes,
                report
                    .scene_gpu_ms
                    .map_or("none".to_owned(), |ms| format!("{ms:.3}")),
                report.gpu,
                project,
                scene,
                meshes
            );
            if ok {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        (Ok(_), None) => ExitCode::SUCCESS,
        (Err(RunError::Unsupported(why)), Some(_)) => {
            println!("SMOKE_RESULT skipped=true reason=\"{why}\"");
            ExitCode::from(EXIT_SKIPPED as u8)
        }
        (Err(e), _) => {
            tracing::error!("{e}");
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
