//! Read-only scene inspector: `cargo run --release -p rustiny-scene --example scene -- <file.unity>`
//!
//! Loads the scene exactly like the editor does and prints what it found. It never writes
//! anything; the scene file is only opened for reading.

use rustiny_scene::{ObjectKind, Scene, load_scene};
use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let Some(path) = std::env::args_os().nth(1).map(PathBuf::from) else {
        eprintln!("usage: scene <file.unity> [--tree N]");
        return ExitCode::from(2);
    };
    let tree_limit: usize = std::env::args()
        .skip_while(|a| a != "--tree")
        .nth(1)
        .and_then(|n| n.parse().ok())
        .unwrap_or(0);

    let scene = match load_scene(&path) {
        Ok(scene) => scene,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };

    let stats = scene.stats();
    let mib = stats.bytes as f64 / (1024.0 * 1024.0);
    let secs = stats.duration.as_secs_f64();
    println!("scene:            {}", scene.name());
    println!(
        "size:             {mib:.1} MiB in {} documents",
        stats.documents
    );
    println!(
        "time:             {:.2} s ({:.0} MiB/s)",
        secs,
        if secs > 0.0 { mib / secs } else { 0.0 }
    );
    println!("game objects:     {}", stats.game_objects);
    println!("transforms:       {}", stats.transforms);
    println!("prefab instances: {}", stats.prefab_instances);
    println!("hierarchy nodes:  {}", scene.len());
    println!("roots:            {}", scene.roots().len());
    println!(
        "issues:           {} stored, {} dropped",
        scene.issues().len(),
        stats.issues_dropped
    );
    for issue in scene.issues().iter().take(10) {
        println!("  - {issue}");
    }
    println!("max depth:        {}", max_depth(&scene));
    print_extent(&scene);

    if tree_limit > 0 {
        let mut printed = 0;
        for &root in scene.roots() {
            print_tree(&scene, root, 0, tree_limit, &mut printed);
        }
    }
    ExitCode::SUCCESS
}

fn max_depth(scene: &Scene) -> usize {
    let mut deepest = 0;
    let mut stack: Vec<(usize, usize)> = scene.roots().iter().map(|&r| (r, 1)).collect();
    while let Some((id, depth)) = stack.pop() {
        deepest = deepest.max(depth);
        if let Some(object) = scene.object(id) {
            stack.extend(object.children.iter().map(|&c| (c, depth + 1)));
        }
    }
    deepest
}

fn print_tree(scene: &Scene, id: usize, depth: usize, limit: usize, printed: &mut usize) {
    if *printed >= limit {
        return;
    }
    let Some(object) = scene.object(id) else {
        return;
    };
    *printed += 1;
    let label = match &object.kind {
        ObjectKind::GameObject => object.name.clone(),
        ObjectKind::PrefabInstance { source_guid } => format!(
            "[prefab {}] {}",
            source_guid.as_deref().unwrap_or("?"),
            object.name
        ),
    };
    println!(
        "{}{} ({} components, {} children)",
        "  ".repeat(depth),
        label,
        object.components.len(),
        object.children.len()
    );
    for &child in &object.children {
        print_tree(scene, child, depth + 1, limit, printed);
    }
}

/// World-space extent of the objects, to sanity-check transforms on a real scene.
fn print_extent(scene: &Scene) {
    let prefabs = scene
        .objects()
        .iter()
        .filter(|o| matches!(o.kind, ObjectKind::PrefabInstance { .. }));
    let (with, total) = prefabs.fold((0, 0), |(w, t), o| {
        (w + usize::from(o.transform.is_some()), t + 1)
    });
    println!("prefabs with a transform override: {with} of {total}");
    for axis in 0..3 {
        let mut values: Vec<f32> = (0..scene.len())
            .filter_map(|id| scene.world_position(id))
            .map(|p| p[axis])
            .filter(|v| v.is_finite())
            .collect();
        if values.is_empty() {
            continue;
        }
        values.sort_by(f32::total_cmp);
        let at = |q: f64| values[((values.len() - 1) as f64 * q) as usize];
        println!(
            "world {}: min {:.1}  p1 {:.1}  median {:.1}  p99 {:.1}  max {:.1}",
            ["x", "y", "z"][axis],
            at(0.0),
            at(0.01),
            at(0.5),
            at(0.99),
            at(1.0)
        );
    }
}
