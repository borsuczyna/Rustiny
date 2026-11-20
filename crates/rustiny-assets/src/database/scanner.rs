//! The directory walk and the parallel `.meta` reader behind [`AssetDatabase::scan`].
//!
//! [`AssetDatabase::scan`]: super::AssetDatabase::scan

use super::entry::MetaStamp;
use super::{AssetDatabase, AssetEntry, Issue, ScanError};
use crate::meta::MetaData;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::SystemTime;

/// Only this many leading bytes of a `.meta` file are read. Identity and importer name are always
/// at the top; the rest (importer settings, sprite sheets) can be megabytes and is not needed yet.
pub(super) const META_HEAD_BYTES: usize = 4096;
/// Upper bound on worker threads reading `.meta` files (the work is I/O latency bound).
const MAX_READERS: usize = 8;

pub(super) struct Scanner<'a> {
    pub(super) db: &'a mut AssetDatabase,
    /// The previous scan of the same project, if any; its `.meta` results are reused when the
    /// files are unchanged.
    pub(super) previous: Option<&'a AssetDatabase>,
    /// `.meta` files still to be read, in entry order.
    pub(super) pending: Vec<PendingMeta>,
}

/// A `.meta` file found during the directory walk, read afterwards (in parallel).
pub(super) struct PendingMeta {
    entry: usize,
    file: PathBuf,
    meta_rel: String,
    is_folder: bool,
    size: u64,
    modified: Option<SystemTime>,
}

/// `true` for names Unity does not import (hidden, `~` folders, `cvs`, `.tmp`).
pub fn is_ignored_name(name: &str) -> bool {
    name.starts_with('.')
        || name.ends_with('~')
        || name.eq_ignore_ascii_case("cvs")
        || name.ends_with(".tmp")
}

struct Child {
    name: String,
    is_dir: bool,
    size: u64,
    modified: Option<SystemTime>,
}

impl Scanner<'_> {
    pub(super) fn scan_dir(
        &mut self,
        dir: &Path,
        rel: &str,
        depth: u16,
        is_root: bool,
    ) -> Result<(), ScanError> {
        let read = match fs::read_dir(dir) {
            Ok(r) => r,
            Err(source) if is_root => {
                return Err(ScanError::Io {
                    path: dir.to_owned(),
                    source,
                });
            }
            Err(e) => {
                self.db.issues.push(Issue::Unreadable {
                    path: rel.to_owned(),
                    reason: e.to_string(),
                });
                return Ok(());
            }
        };

        let mut children = Vec::new();
        for item in read {
            let Ok(item) = item else { continue };
            let Some(name) = item.file_name().to_str().map(str::to_owned) else {
                self.db.issues.push(Issue::NonUnicodeName {
                    path: format!("{rel}/{}", item.file_name().to_string_lossy()),
                });
                continue;
            };
            let Ok(file_type) = item.file_type() else {
                continue;
            };
            if file_type.is_symlink() {
                self.db.issues.push(Issue::SymlinkSkipped {
                    path: format!("{rel}/{name}"),
                });
                continue;
            }
            // On Windows the directory listing already carries size and times, so this is free.
            let (size, modified) = item
                .metadata()
                .map(|m| (m.len(), m.modified().ok()))
                .unwrap_or((0, None));
            children.push(Child {
                name,
                is_dir: file_type.is_dir(),
                size,
                modified,
            });
        }

        let meta_names: HashMap<&str, (u64, Option<SystemTime>)> = children
            .iter()
            .filter(|c| !c.is_dir && c.name.ends_with(".meta"))
            .map(|c| (c.name.as_str(), (c.size, c.modified)))
            .collect();
        let mut assets: Vec<&Child> = children
            .iter()
            .filter(|c| !(is_ignored_name(&c.name) || (!c.is_dir && c.name.ends_with(".meta"))))
            .collect();
        // Folders first, then case-insensitive name order; exact name breaks ties.
        assets.sort_by(|a, b| {
            b.is_dir
                .cmp(&a.is_dir)
                .then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
                .then_with(|| a.name.cmp(&b.name))
        });

        let asset_names: HashSet<&str> = children
            .iter()
            .filter(|c| !(!c.is_dir && c.name.ends_with(".meta")))
            .map(|c| c.name.as_str())
            .collect();
        let mut orphans: Vec<&str> = meta_names
            .keys()
            .copied()
            .filter(|m| {
                m.strip_suffix(".meta")
                    .is_some_and(|asset| !asset_names.contains(asset))
            })
            .collect();
        orphans.sort();
        for m in orphans {
            self.db.issues.push(Issue::OrphanMeta {
                path: format!("{rel}/{m}"),
            });
        }

        for child in assets {
            let path = format!("{rel}/{}", child.name);
            let full = dir.join(&child.name);
            let meta_name = format!("{}.meta", child.name);
            let index = self.db.entries.len();
            let mut guid = None;
            let mut importer = None;
            let mut meta_stamp = None;
            if let Some(&(size, modified)) = meta_names.get(meta_name.as_str()) {
                if let Some((prev_guid, prev_importer, stamp)) =
                    self.reusable(&path, size, modified)
                {
                    // Unchanged since the last scan: no need to open the file.
                    self.db.stats.metas_reused += 1;
                    if stamp.folder_flag != child.is_dir {
                        self.db
                            .issues
                            .push(Issue::KindMismatch { path: path.clone() });
                    }
                    guid = Some(prev_guid);
                    importer = prev_importer;
                    meta_stamp = Some(stamp);
                } else {
                    self.pending.push(PendingMeta {
                        entry: index,
                        file: dir.join(&meta_name),
                        meta_rel: format!("{path}.meta"),
                        is_folder: child.is_dir,
                        size,
                        modified,
                    });
                }
            } else {
                self.db
                    .issues
                    .push(Issue::MissingMeta { path: path.clone() });
            }

            self.db.entries.push(AssetEntry {
                path: path.clone(),
                depth,
                is_folder: child.is_dir,
                guid,
                importer,
                size: if child.is_dir { 0 } else { child.size },
                modified: if child.is_dir { None } else { child.modified },
                subtree_end: index + 1,
                meta_stamp,
            });
            if child.is_dir {
                self.db.stats.folders += 1;
                self.scan_dir(&full, &path, depth.saturating_add(1), false)?;
                self.db.entries[index].subtree_end = self.db.entries.len();
            } else {
                self.db.stats.files += 1;
            }
        }
        Ok(())
    }

    /// What the previous scan learned about `asset_path`'s `.meta`, if that file is unchanged.
    /// Requires a modification time: without one a change cannot be detected.
    fn reusable(
        &self,
        asset_path: &str,
        size: u64,
        modified: Option<SystemTime>,
    ) -> Option<(crate::Guid, Option<String>, MetaStamp)> {
        modified?;
        let prev = self.previous?.by_path(asset_path)?;
        let stamp = prev.meta_stamp.as_ref()?;
        if stamp.size != size || stamp.modified != modified {
            return None;
        }
        Some((prev.guid?, prev.importer.clone(), stamp.clone()))
    }

    /// Read every pending `.meta` (in parallel) and apply the results in entry order, so the
    /// outcome does not depend on thread timing.
    pub(super) fn resolve_metas(&mut self) {
        let pending = std::mem::take(&mut self.pending);
        let threads = std::thread::available_parallelism()
            .map_or(4, usize::from)
            .clamp(1, MAX_READERS)
            .min(pending.len().max(1));
        let next = AtomicUsize::new(0);
        let mut outcomes: Vec<Option<Result<MetaData, String>>> = Vec::new();
        outcomes.resize_with(pending.len(), || None);

        std::thread::scope(|scope| {
            let workers: Vec<_> = (0..threads)
                .map(|_| {
                    scope.spawn(|| {
                        let mut done = Vec::new();
                        loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            let Some(job) = pending.get(i) else { break };
                            done.push((i, read_meta(&job.file)));
                        }
                        done
                    })
                })
                .collect();
            for worker in workers {
                match worker.join() {
                    Ok(done) => {
                        for (i, outcome) in done {
                            outcomes[i] = Some(outcome);
                        }
                    }
                    Err(_) => tracing::error!("a .meta reader thread panicked"),
                }
            }
        });

        for (job, outcome) in pending.iter().zip(outcomes) {
            let outcome = outcome.unwrap_or_else(|| Err("reader thread failed".to_owned()));
            match outcome {
                Ok(meta) => {
                    self.db.stats.metas_parsed += 1;
                    let entry = &mut self.db.entries[job.entry];
                    if meta.folder_asset != job.is_folder {
                        self.db.issues.push(Issue::KindMismatch {
                            path: entry.path.clone(),
                        });
                    }
                    entry.guid = Some(meta.guid);
                    entry.importer = meta.importer;
                    entry.meta_stamp = Some(MetaStamp {
                        size: job.size,
                        modified: job.modified,
                        folder_flag: meta.folder_asset,
                    });
                }
                Err(reason) => self.db.issues.push(Issue::InvalidMeta {
                    path: job.meta_rel.clone(),
                    reason,
                }),
            }
        }
    }
}

/// Read and parse the head of one `.meta` file.
fn read_meta(file: &Path) -> Result<MetaData, String> {
    let mut buf = Vec::with_capacity(META_HEAD_BYTES + 1);
    fs::File::open(file)
        .and_then(|f| f.take(META_HEAD_BYTES as u64 + 1).read_to_end(&mut buf))
        .map_err(|e| e.to_string())?;
    if buf.len() > META_HEAD_BYTES {
        // The file continues: drop the cut-off last line so it cannot be misread.
        let end = buf[..META_HEAD_BYTES]
            .iter()
            .rposition(|&b| b == b'\n')
            .ok_or_else(|| format!("first line is longer than {META_HEAD_BYTES} bytes"))?;
        buf.truncate(end + 1);
    }
    MetaData::parse(&buf).map_err(|e| e.to_string())
}
