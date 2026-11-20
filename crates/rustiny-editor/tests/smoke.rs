//! End-to-end smoke test: launches the real editor binary against the real GPU.
//!
//! Verifies the Milestone 1 acceptance criteria mechanically:
//! * a window opens and frames are presented,
//! * the Vulkan validation layer (including synchronisation validation) reports zero errors,
//! * the swapchain survives a window resize,
//! * the frame read back from the GPU actually contains rendered UI (not a blank clear colour),
//! * the dock layout file is persisted.
//!
//! If the machine cannot run the editor at all (no Vulkan loader / GPU / display) the binary exits
//! with code 77 and this test prints `SKIPPED` rather than failing.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashSet;
use std::fs::File;
use std::path::PathBuf;
use std::process::Command;

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("rustiny-smoke-test-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let pat = format!("{key}=");
    let start = line.find(&pat)? + pat.len();
    let rest = &line[start..];
    Some(rest.split_whitespace().next().unwrap_or(""))
}

#[test]
fn editor_renders_frames_without_validation_errors() {
    let dir = scratch("render");
    let png_path = dir.join("frame.png");
    let output = Command::new(env!("CARGO_BIN_EXE_rustiny"))
        .args(["--smoke-frames", "60", "--config-dir"])
        .arg(&dir)
        .arg("--screenshot")
        .arg(&png_path)
        .output()
        .expect("failed to launch rustiny");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if output.status.code() == Some(77) {
        eprintln!("SKIPPED: machine cannot run the editor: {stdout}");
        return;
    }
    assert!(
        output.status.success(),
        "editor failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );

    let line = stdout
        .lines()
        .find(|l| l.starts_with("SMOKE_RESULT"))
        .unwrap_or_else(|| panic!("no SMOKE_RESULT line in:\n{stdout}"));
    assert_eq!(field(line, "ok"), Some("true"), "{line}");
    assert_eq!(
        field(line, "validation_errors"),
        Some("0"),
        "{line}\n{stderr}"
    );
    let frames: u64 = field(line, "frames").unwrap().parse().unwrap();
    assert!(frames >= 60, "{line}");
    let recreations: u64 = field(line, "swapchain_recreations")
        .unwrap()
        .parse()
        .unwrap();
    assert!(
        recreations >= 1,
        "resize must have recreated the swapchain: {line}"
    );

    // The captured frame must contain real UI: many distinct colours, not one flat clear colour.
    let decoder = png::Decoder::new(std::io::BufReader::new(File::open(&png_path).unwrap()));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();
    assert!(
        info.width >= 640 && info.height >= 360,
        "{}x{}",
        info.width,
        info.height
    );
    let colours: HashSet<[u8; 3]> = buf[..info.buffer_size()]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| [p[0], p[1], p[2]])
        .collect();
    assert!(
        colours.len() > 8,
        "frame looks blank: only {} distinct colours",
        colours.len()
    );

    // Layout persistence: Dear ImGui must have written the dock layout.
    let layout = std::fs::read_to_string(dir.join("layout.ini")).expect("layout.ini not written");
    assert!(
        layout.contains("[Docking][Data]"),
        "no dock data persisted:\n{layout}"
    );
    for panel in [
        "hierarchy",
        "scene",
        "game",
        "inspector",
        "project",
        "console",
        "profiler",
    ] {
        assert!(
            layout.contains(&format!("panel.{panel}")),
            "panel.{panel} missing from layout"
        );
    }
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn usage_errors_exit_with_code_two() {
    let status = Command::new(env!("CARGO_BIN_EXE_rustiny"))
        .arg("--bogus")
        .output()
        .unwrap();
    assert_eq!(status.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&status.stderr).contains("--bogus"));
}

#[test]
fn help_and_version_do_not_open_a_window() {
    let help = Command::new(env!("CARGO_BIN_EXE_rustiny"))
        .arg("--help")
        .output()
        .unwrap();
    assert!(help.status.success());
    assert!(String::from_utf8_lossy(&help.stdout).contains("--smoke-frames"));
    let version = Command::new(env!("CARGO_BIN_EXE_rustiny"))
        .arg("--version")
        .output()
        .unwrap();
    assert!(String::from_utf8_lossy(&version.stdout).starts_with("rustiny 0."));
}

#[test]
fn opens_a_unity_project_read_only_and_indexes_it() {
    let dir = scratch("project-config");
    let project = scratch("project");
    let write = |rel: &str, text: &str| {
        let path = project.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    };
    write(
        "ProjectSettings/ProjectVersion.txt",
        "m_EditorVersion: 2022.3.10f1\n",
    );
    write(
        "Assets/Scripts.meta",
        "fileFormatVersion: 2\nguid: 00000000000000000000000000000001\nfolderAsset: yes\nDefaultImporter:\n",
    );
    write("Assets/Scripts/Player.cs", "class Player {}");
    write(
        "Assets/Scripts/Player.cs.meta",
        "fileFormatVersion: 2\nguid: 00000000000000000000000000000002\nMonoImporter:\n",
    );
    write("Assets/Unmanaged.txt", "no meta file next to me");

    let list = |root: &PathBuf| -> Vec<(String, Vec<u8>)> {
        let mut out = Vec::new();
        let mut stack = vec![root.clone()];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(d).unwrap() {
                let p = e.unwrap().path();
                if p.is_dir() {
                    stack.push(p);
                } else {
                    out.push((p.display().to_string(), std::fs::read(&p).unwrap()));
                }
            }
        }
        out.sort();
        out
    };
    let before = list(&project);

    let output = Command::new(env!("CARGO_BIN_EXE_rustiny"))
        .args(["--smoke-frames", "90", "--no-validation", "--config-dir"])
        .arg(&dir)
        .arg("--project")
        .arg(&project)
        .output()
        .expect("failed to launch rustiny");
    let stdout = String::from_utf8_lossy(&output.stdout);
    if output.status.code() == Some(77) {
        eprintln!("SKIPPED: machine cannot run the editor: {stdout}");
        return;
    }
    assert!(
        output.status.success(),
        "{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let line = stdout
        .lines()
        .find(|l| l.starts_with("SMOKE_RESULT"))
        .unwrap();
    assert_eq!(field(line, "project_files"), Some("2"), "{line}");
    assert_eq!(field(line, "project_folders"), Some("1"), "{line}");
    assert_eq!(
        field(line, "project_issues"),
        Some("1"),
        "missing meta: {line}"
    );

    assert_eq!(
        list(&project),
        before,
        "opening a project must not modify it"
    );
}

#[test]
fn project_changes_made_while_running_are_picked_up() {
    use std::io::{BufRead, BufReader, Read};
    use std::process::Stdio;
    use std::sync::mpsc;
    use std::time::Duration;

    let config = scratch("live-config");
    let project = scratch("live-project");
    let write = |rel: &str, text: &str| {
        let path = project.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    };
    let meta = |n: u32| format!("fileFormatVersion: 2\nguid: {n:032x}\nDefaultImporter:\n");
    write(
        "ProjectSettings/ProjectVersion.txt",
        "m_EditorVersion: 2022.3.10f1\n",
    );
    write("Assets/A.txt", "a");
    write("Assets/A.txt.meta", &meta(1));

    let mut child = Command::new(env!("CARGO_BIN_EXE_rustiny"))
        .args(["--smoke-frames", "900", "--no-validation", "--config-dir"])
        .arg(&config)
        .arg("--project")
        .arg(&project)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to launch rustiny");
    let stderr = child.stderr.take().unwrap();
    let (tx, rx) = mpsc::channel::<String>();
    std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            if tx.send(line).is_err() {
                break;
            }
        }
    });

    // Wait for the first scan to finish; only then change the project.
    let mut first_scan_seen = false;
    loop {
        match rx.recv_timeout(Duration::from_secs(60)) {
            Ok(line) if line.contains("scanned ") => {
                first_scan_seen = true;
                break;
            }
            Ok(_) => {}
            Err(_) => break,
        }
    }
    if !first_scan_seen {
        let status = child.wait().unwrap();
        if status.code() == Some(77) {
            eprintln!("SKIPPED: machine cannot run the editor");
            return;
        }
        panic!("the project was never scanned (exit status {status})");
    }
    write("Assets/B.txt", "b");
    write("Assets/B.txt.meta", &meta(2));
    write("Assets/Sub/C.txt", "c");
    write("Assets/Sub/C.txt.meta", &meta(3));
    write("Assets/Sub.meta", &format!("{}folderAsset: yes\n", meta(4)));

    let mut stdout = String::new();
    child
        .stdout
        .take()
        .unwrap()
        .read_to_string(&mut stdout)
        .unwrap();
    let status = child.wait().unwrap();
    if !status.success() {
        // Keep the evidence: this test failed once without its output being captured.
        std::thread::sleep(Duration::from_millis(200));
        let log: Vec<String> = rx.try_iter().collect();
        panic!(
            "editor exited with {status}\nstdout:\n{stdout}\nstderr after the first scan:\n{}",
            log.join("\n")
        );
    }
    let line = stdout
        .lines()
        .find(|l| l.starts_with("SMOKE_RESULT"))
        .unwrap();
    assert_eq!(
        field(line, "project_files"),
        Some("3"),
        "A, B and C: {line}"
    );
    assert_eq!(field(line, "project_folders"), Some("1"), "{line}");
    let scans: u32 = field(line, "project_scans").unwrap().parse().unwrap();
    assert!(scans >= 2, "a rescan must have followed the change: {line}");
}

#[test]
fn opens_a_scene_read_only_and_builds_its_hierarchy() {
    let dir = scratch("scene-config");
    let project = scratch("scene-project");
    let scene_dir = project.join("Assets").join("Scenes");
    std::fs::create_dir_all(&scene_dir).unwrap();
    std::fs::create_dir_all(project.join("ProjectSettings")).unwrap();
    std::fs::write(
        project.join("ProjectSettings").join("ProjectVersion.txt"),
        "m_EditorVersion: 6000.0.0f1\n",
    )
    .unwrap();

    // Root "Level" with child "Crate", plus one prefab instance attached to the root.
    let scene_text = [
        "%YAML 1.1",
        "%TAG !u! tag:unity3d.com,2011:",
        "--- !u!1 &10",
        "GameObject:",
        "  m_Component:",
        "  - component: {fileID: 11}",
        "  m_Layer: 0",
        "  m_Name: Level",
        "  m_TagString: Untagged",
        "  m_IsActive: 1",
        "--- !u!4 &11",
        "Transform:",
        "  m_GameObject: {fileID: 10}",
        "  m_LocalRotation: {x: 0, y: 0, z: 0, w: 1}",
        "  m_LocalPosition: {x: 0, y: 0, z: 0}",
        "  m_LocalScale: {x: 1, y: 1, z: 1}",
        "  m_Children:",
        "  - {fileID: 21}",
        "  - {fileID: 31}",
        "  m_Father: {fileID: 0}",
        "--- !u!1 &20",
        "GameObject:",
        "  m_Component:",
        "  - component: {fileID: 21}",
        "  m_Layer: 0",
        "  m_Name: Crate",
        "  m_TagString: Untagged",
        "  m_IsActive: 1",
        "--- !u!4 &21",
        "Transform:",
        "  m_GameObject: {fileID: 20}",
        "  m_LocalRotation: {x: 0, y: 0, z: 0, w: 1}",
        "  m_LocalPosition: {x: 1, y: 2, z: 3}",
        "  m_LocalScale: {x: 1, y: 1, z: 1}",
        "  m_Children: []",
        "  m_Father: {fileID: 11}",
        "--- !u!4 &31 stripped",
        "Transform:",
        "  m_PrefabInstance: {fileID: 40}",
        "--- !u!1001 &40",
        "PrefabInstance:",
        "  m_Modification:",
        "    m_TransformParent: {fileID: 11}",
        "    m_Modifications: []",
        "  m_SourcePrefab: {fileID: 100100000, guid: 0000000000000000000000000000000a, type: 3}",
        "",
    ]
    .join("\n");
    let scene_path = scene_dir.join("Level.unity");
    std::fs::write(&scene_path, &scene_text).unwrap();

    // Validation stays on: the GPU Scene view must render this scene without a single error.
    let output = Command::new(env!("CARGO_BIN_EXE_rustiny"))
        .args(["--smoke-frames", "60", "--config-dir"])
        .arg(&dir)
        .arg("--project")
        .arg(&project)
        .arg("--scene")
        .arg(&scene_path)
        .output()
        .expect("failed to launch rustiny");
    let stdout = String::from_utf8_lossy(&output.stdout);
    if output.status.code() == Some(77) {
        eprintln!("SKIPPED: machine cannot run the editor: {stdout}");
        return;
    }
    assert!(
        output.status.success(),
        "{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let line = stdout
        .lines()
        .find(|l| l.starts_with("SMOKE_RESULT"))
        .unwrap();
    assert_eq!(field(line, "scene_objects"), Some("3"), "{line}");
    assert_eq!(field(line, "scene_roots"), Some("1"), "{line}");
    assert_eq!(field(line, "scene_issues"), Some("0"), "{line}");
    assert_eq!(field(line, "validation_errors"), Some("0"), "{line}");
    let passes: u64 = field(line, "scene_gpu_passes").unwrap().parse().unwrap();
    assert!(
        passes >= 1,
        "the GPU Scene view must have rendered the scene: {line}"
    );
    assert_eq!(
        std::fs::read_to_string(&scene_path).unwrap(),
        scene_text,
        "opening a scene must not modify it"
    );
}

#[test]
fn a_scene_that_cannot_be_read_does_not_crash_the_editor() {
    let dir = scratch("badscene-config");
    let bad = dir.join("Broken.unity");
    std::fs::write(&bad, b"UnityFS\0 this is not text").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_rustiny"))
        .args(["--smoke-frames", "20", "--no-validation", "--config-dir"])
        .arg(&dir)
        .arg("--scene")
        .arg(&bad)
        .output()
        .expect("failed to launch rustiny");
    let stdout = String::from_utf8_lossy(&output.stdout);
    if output.status.code() == Some(77) {
        eprintln!("SKIPPED: machine cannot run the editor: {stdout}");
        return;
    }
    assert!(output.status.success(), "{stdout}");
    let line = stdout
        .lines()
        .find(|l| l.starts_with("SMOKE_RESULT"))
        .unwrap();
    assert_eq!(
        field(line, "scene_objects"),
        None,
        "no scene was loaded: {line}"
    );
}
