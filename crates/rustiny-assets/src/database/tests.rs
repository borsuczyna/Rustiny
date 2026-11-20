//! Scan tests on generated project fixtures.

use super::scanner::META_HEAD_BYTES;
use super::{AssetDatabase, Issue, ScanError};
use crate::project::UnityProject;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

/// A throw-away directory removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        static N: AtomicU32 = AtomicU32::new(0);
        let dir = std::env::temp_dir().join(format!(
            "rustiny-assets-test-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    fn write(&self, rel: &str, content: &str) {
        let path = self.0.join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, content).unwrap();
    }

    fn project(&self) -> UnityProject {
        UnityProject::detect(&self.0).unwrap()
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn meta(guid: &str, importer: &str) -> String {
    format!("fileFormatVersion: 2\nguid: {guid}\n{importer}:\n  userData: \n")
}

fn folder_meta(guid: &str) -> String {
    format!(
        "fileFormatVersion: 2\nguid: {guid}\nfolderAsset: yes\nDefaultImporter:\n  userData: \n"
    )
}

fn g(n: u32) -> String {
    format!("{n:032x}")
}

fn new_project() -> TempDir {
    let t = TempDir::new();
    t.write(
        "ProjectSettings/ProjectVersion.txt",
        "m_EditorVersion: 2022.3.10f1\n",
    );
    t
}

#[test]
fn detect_requires_assets_and_project_settings() {
    let t = TempDir::new();
    assert!(matches!(
        UnityProject::detect(&t.0),
        Err(crate::ProjectError::MissingFolder {
            missing: "Assets",
            ..
        })
    ));
    fs::create_dir_all(t.0.join("Assets")).unwrap();
    assert!(matches!(
        UnityProject::detect(&t.0),
        Err(crate::ProjectError::MissingFolder {
            missing: "ProjectSettings",
            ..
        })
    ));
    fs::create_dir_all(t.0.join("ProjectSettings")).unwrap();
    let p = UnityProject::detect(&t.0).unwrap();
    assert_eq!(p.editor_version, None);
    assert!(!p.has_package_manifest);
    assert!(matches!(
        UnityProject::detect(&t.0.join("nope")),
        Err(crate::ProjectError::NotADirectory(_))
    ));
}

#[test]
fn indexes_a_small_project_in_tree_order() {
    let t = new_project();
    t.write("Assets/Textures.meta", &folder_meta(&g(1)));
    t.write("Assets/Textures/b.png", "x");
    t.write(
        "Assets/Textures/b.png.meta",
        &meta(&g(2), "TextureImporter"),
    );
    t.write("Assets/Textures/A.png", "xyz");
    t.write(
        "Assets/Textures/A.png.meta",
        &meta(&g(3), "TextureImporter"),
    );
    t.write("Assets/Main.cs", "class Main {}");
    t.write("Assets/Main.cs.meta", &meta(&g(4), "MonoImporter"));
    t.write("Assets/.hidden", "ignored");
    t.write("Assets/Cache~/ignored.txt", "ignored");
    t.write("Assets/scratch.tmp", "ignored");

    let db = AssetDatabase::scan(&t.project()).unwrap();
    let paths: Vec<&str> = db.entries().iter().map(|e| e.path.as_str()).collect();
    assert_eq!(
        paths,
        [
            "Assets/Textures",
            "Assets/Textures/A.png",
            "Assets/Textures/b.png",
            "Assets/Main.cs"
        ]
    );
    assert!(db.issues().is_empty(), "{:?}", db.issues());
    assert_eq!(db.stats().files, 3);
    assert_eq!(db.stats().folders, 1);
    assert_eq!(db.stats().metas_parsed, 4);

    let folder = &db.entries()[0];
    assert!(folder.is_folder);
    assert_eq!(folder.subtree_end, 3);
    assert_eq!(db.entries()[1].depth, 1);
    assert_eq!(db.entries()[1].size, 3);

    let main = db.by_path("Assets/Main.cs").unwrap();
    assert_eq!(main.importer.as_deref(), Some("MonoImporter"));
    assert_eq!(main.name(), "Main.cs");
    assert_eq!(
        db.by_guid(g(3).parse().unwrap()).unwrap().path,
        "Assets/Textures/A.png"
    );
}

#[test]
fn scan_never_writes_into_the_project() {
    let t = new_project();
    t.write("Assets/NoMeta.txt", "hello");
    let before = snapshot(&t.0);
    let db = AssetDatabase::scan(&t.project()).unwrap();
    assert_eq!(db.issues().len(), 1);
    assert_eq!(
        snapshot(&t.0),
        before,
        "the scan must not create or change files"
    );
}

fn snapshot(root: &Path) -> Vec<(String, u64)> {
    fn walk(dir: &Path, root: &Path, out: &mut Vec<(String, u64)>) {
        for e in fs::read_dir(dir).unwrap() {
            let e = e.unwrap();
            let p = e.path();
            let rel = p
                .strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            if p.is_dir() {
                out.push((rel, 0));
                walk(&p, root, out);
            } else {
                out.push((rel, fs::metadata(&p).unwrap().len()));
            }
        }
    }
    let mut out = Vec::new();
    walk(root, root, &mut out);
    out.sort();
    out
}

#[test]
fn reports_every_kind_of_problem_without_aborting() {
    let t = new_project();
    t.write("Assets/NoMeta.txt", "x");
    t.write("Assets/Gone.txt.meta", &meta(&g(10), "DefaultImporter"));
    t.write("Assets/Broken.txt", "x");
    t.write("Assets/Broken.txt.meta", "fileFormatVersion: 2\n");
    t.write("Assets/One.txt", "x");
    t.write("Assets/One.txt.meta", &meta(&g(20), "DefaultImporter"));
    t.write("Assets/Two.txt", "x");
    t.write("Assets/Two.txt.meta", &meta(&g(20), "DefaultImporter"));
    t.write("Assets/Flag.txt", "x");
    t.write("Assets/Flag.txt.meta", &folder_meta(&g(30)));

    let db = AssetDatabase::scan(&t.project()).unwrap();
    let issues = db.issues();
    assert!(issues.contains(&Issue::MissingMeta {
        path: "Assets/NoMeta.txt".into()
    }));
    assert!(issues.contains(&Issue::OrphanMeta {
        path: "Assets/Gone.txt.meta".into()
    }));
    assert!(issues.iter().any(|i| matches!(i,
        Issue::InvalidMeta { path, .. } if path == "Assets/Broken.txt.meta")));
    assert!(issues.contains(&Issue::KindMismatch {
        path: "Assets/Flag.txt".into()
    }));
    let dup = issues.iter().find_map(|i| match i {
        Issue::DuplicateGuid { paths, .. } => Some(paths.clone()),
        _ => None,
    });
    assert_eq!(dup.unwrap(), ["Assets/One.txt", "Assets/Two.txt"]);
    // The first user wins the lookup, deterministically.
    assert_eq!(
        db.by_guid(g(20).parse().unwrap()).unwrap().path,
        "Assets/One.txt"
    );
    // Entries with a broken meta are still listed, just without identity.
    assert_eq!(db.by_path("Assets/Broken.txt").unwrap().guid, None);
    assert_eq!(db.by_path("Assets/NoMeta.txt").unwrap().guid, None);
}

#[test]
fn empty_assets_folder_is_fine_and_missing_one_is_an_error() {
    let t = new_project();
    fs::create_dir_all(t.0.join("Assets")).unwrap();
    let db = AssetDatabase::scan(&t.project()).unwrap();
    assert!(db.entries().is_empty());

    let p = t.project();
    fs::remove_dir_all(t.0.join("Assets")).unwrap();
    assert!(matches!(AssetDatabase::scan(&p), Err(ScanError::Io { .. })));
}

#[test]
fn huge_meta_files_are_read_by_their_head_only() {
    let t = new_project();
    // Importer settings far larger than the head that is read, with a deliberately
    // misleading nested `guid:` and a line cut by the read limit.
    let mut body = meta(&g(7), "TextureImporter");
    for i in 0..2000 {
        body.push_str(&format!(
            "  sprite{i}:\n    guid: ffffffffffffffffffffffffffffffff\n"
        ));
    }
    t.write("Assets/Big.png", "x");
    t.write("Assets/Big.png.meta", &body);
    assert!(body.len() > META_HEAD_BYTES * 10);

    let db = AssetDatabase::scan(&t.project()).unwrap();
    assert!(db.issues().is_empty(), "{:?}", db.issues());
    let big = db.by_path("Assets/Big.png").unwrap();
    assert_eq!(big.guid, Some(g(7).parse().unwrap()));
    assert_eq!(big.importer.as_deref(), Some("TextureImporter"));
}

#[test]
fn a_cut_off_line_is_never_misread() {
    let t = new_project();
    // Pad so a top-level `guid:` line straddles the read limit.
    let mut body = String::from("fileFormatVersion: 2\n");
    body.push_str(&format!("guid: {}\n", g(9)));
    body.push_str("DefaultImporter:\n");
    while body.len() < META_HEAD_BYTES - 10 {
        body.push_str("# padding\n");
    }
    body.push_str("guid: 0123456789abcdef0123456789abcdef\n");
    t.write("Assets/Edge.txt", "x");
    t.write("Assets/Edge.txt.meta", &body);
    let db = AssetDatabase::scan(&t.project()).unwrap();
    assert!(db.issues().is_empty(), "{:?}", db.issues());
    assert_eq!(
        db.by_path("Assets/Edge.txt").unwrap().guid,
        Some(g(9).parse().unwrap())
    );
}

#[test]
fn scan_is_deterministic() {
    let t = new_project();
    for i in 0..20 {
        t.write(&format!("Assets/d{}/f{}.txt", i % 4, i), "x");
        t.write(
            &format!("Assets/d{}/f{}.txt.meta", i % 4, i),
            &meta(&g(100 + i), "DefaultImporter"),
        );
    }
    let a: Vec<String> = AssetDatabase::scan(&t.project())
        .unwrap()
        .entries()
        .iter()
        .map(|e| e.path.clone())
        .collect();
    let b: Vec<String> = AssetDatabase::scan(&t.project())
        .unwrap()
        .entries()
        .iter()
        .map(|e| e.path.clone())
        .collect();
    assert_eq!(a, b);
}

fn make_many(t: &TempDir, n: u32) {
    for i in 0..n {
        t.write(&format!("Assets/d{}/f{i}.txt", i % 5), "x");
        t.write(
            &format!("Assets/d{}/f{i}.txt.meta", i % 5),
            &meta(&g(1000 + i), "DefaultImporter"),
        );
    }
}

#[test]
fn rescan_without_changes_reads_no_metas_and_gives_the_same_index() {
    let t = new_project();
    make_many(&t, 30);
    let first = AssetDatabase::scan(&t.project()).unwrap();
    let second = first.rescan(&t.project()).unwrap();
    assert_eq!(first.stats().metas_parsed, 30);
    assert_eq!(first.stats().metas_reused, 0);
    assert_eq!(
        second.stats().metas_parsed,
        0,
        "nothing should have been re-read"
    );
    assert_eq!(second.stats().metas_reused, 30);
    assert_eq!(first.entries(), second.entries());
    assert_eq!(first.issues(), second.issues());
}

#[test]
fn rescan_picks_up_added_changed_and_removed_assets() {
    let t = new_project();
    make_many(&t, 10);
    let first = AssetDatabase::scan(&t.project()).unwrap();
    assert!(first.by_guid(g(1003).parse().unwrap()).is_some());

    // Add an asset, change one meta (different length, so the change is detectable), remove one.
    t.write("Assets/New.txt", "new");
    t.write("Assets/New.txt.meta", &meta(&g(5000), "MonoImporter"));
    t.write(
        "Assets/d0/f0.txt.meta",
        &meta(&g(6000), "TextureImporter"), // longer importer name than before
    );
    fs::remove_file(t.0.join("Assets/d1/f1.txt")).unwrap();
    fs::remove_file(t.0.join("Assets/d1/f1.txt.meta")).unwrap();

    let second = first.rescan(&t.project()).unwrap();
    assert_eq!(
        second.stats().metas_parsed,
        2,
        "only the new and the changed meta are read"
    );
    assert_eq!(second.stats().metas_reused, 8);
    assert_eq!(
        second.by_path("Assets/New.txt").unwrap().guid,
        Some(g(5000).parse().unwrap())
    );
    let changed = second.by_path("Assets/d0/f0.txt").unwrap();
    assert_eq!(changed.guid, Some(g(6000).parse().unwrap()));
    assert_eq!(changed.importer.as_deref(), Some("TextureImporter"));
    assert!(second.by_path("Assets/d1/f1.txt").is_none());
    assert!(second.by_guid(g(1001).parse().unwrap()).is_none());

    // The result must equal a from-scratch scan of the same tree.
    let fresh = AssetDatabase::scan(&t.project()).unwrap();
    assert_eq!(second.entries(), fresh.entries());
    assert_eq!(second.issues(), fresh.issues());
}

#[test]
fn rescan_still_reports_a_kind_mismatch_from_reused_data() {
    let t = new_project();
    t.write("Assets/Flag.txt", "x");
    t.write("Assets/Flag.txt.meta", &folder_meta(&g(30)));
    let first = AssetDatabase::scan(&t.project()).unwrap();
    let second = first.rescan(&t.project()).unwrap();
    assert_eq!(second.stats().metas_reused, 1);
    assert_eq!(first.issues(), second.issues());
    assert!(second.issues().contains(&Issue::KindMismatch {
        path: "Assets/Flag.txt".into()
    }));
}
