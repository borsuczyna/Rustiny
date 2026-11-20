//! The asset database: a read-only index of a Unity project's `Assets/` folder.
//!
//! The scan never writes. Entries are stored in depth-first order (folders before files, names
//! compared case-insensitively) so a tree can be drawn from the flat list: a folder at index `i`
//! owns the entries `i + 1 .. subtree_end`.
//!
//! * `entry`   - [`AssetEntry`] and [`ScanStats`].
//! * `issue`   - [`Issue`], the problems a scan reports.
//! * `scanner` - directory walk and parallel `.meta` reading.

mod entry;
mod issue;
mod scanner;
#[cfg(test)]
mod tests;

pub use entry::{AssetEntry, ScanStats};
pub use issue::Issue;
pub use scanner::is_ignored_name;

use crate::guid::Guid;
use crate::project::UnityProject;
use scanner::Scanner;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

/// Why a scan could not start.
#[derive(Debug, thiserror::Error)]
pub enum ScanError {
    /// `Assets/` could not be listed.
    #[error("cannot read `{}`: {source}", path.display())]
    Io {
        /// The folder.
        path: PathBuf,
        /// OS error.
        #[source]
        source: std::io::Error,
    },
}

/// Index of everything under `Assets/`.
#[derive(Debug, Default)]
pub struct AssetDatabase {
    entries: Vec<AssetEntry>,
    by_guid: HashMap<Guid, usize>,
    by_path: HashMap<String, usize>,
    issues: Vec<Issue>,
    stats: ScanStats,
}

impl AssetDatabase {
    /// Scan `project/Assets`. Read-only.
    pub fn scan(project: &UnityProject) -> Result<Self, ScanError> {
        Self::scan_inner(project, None)
    }

    /// Scan again, reusing the `.meta` results of `self` for files whose size and modification
    /// time did not change. The directory tree is walked again in full, which is cheap; reading
    /// the `.meta` files is what dominates a cold scan. Read-only.
    pub fn rescan(&self, project: &UnityProject) -> Result<Self, ScanError> {
        Self::scan_inner(project, Some(self))
    }

    fn scan_inner(project: &UnityProject, previous: Option<&Self>) -> Result<Self, ScanError> {
        let started = Instant::now();
        let assets_dir = project.root.join("Assets");
        let mut db = Self::default();
        let mut scanner = Scanner {
            db: &mut db,
            previous,
            pending: Vec::new(),
        };
        scanner.scan_dir(&assets_dir, "Assets", 0, true)?;
        scanner.resolve_metas();

        let mut guid_users: HashMap<Guid, Vec<usize>> = HashMap::new();
        for (i, entry) in db.entries.iter().enumerate() {
            if let Some(g) = entry.guid {
                guid_users.entry(g).or_default().push(i);
            }
        }
        let mut shared: Vec<(&Guid, &Vec<usize>)> = guid_users
            .iter()
            .filter(|(_, users)| users.len() > 1)
            .collect();
        shared.sort_by_key(|(guid, _)| **guid);
        let duplicates: Vec<Issue> = shared
            .into_iter()
            .map(|(guid, users)| Issue::DuplicateGuid {
                guid: *guid,
                paths: users.iter().map(|&i| db.entries[i].path.clone()).collect(),
            })
            .collect();
        db.issues.extend(duplicates);

        for (guid, users) in &guid_users {
            // First user in path order wins lookups; the duplicate is reported above.
            db.by_guid.insert(*guid, users[0]);
        }
        db.by_path = db
            .entries
            .iter()
            .enumerate()
            .map(|(i, e)| (e.path.clone(), i))
            .collect();
        db.stats.duration = started.elapsed();
        tracing::info!(
            "scanned {}: {} files, {} folders, {} issues ({} metas read, {} reused) in {:.1} ms",
            project.root.display(),
            db.stats.files,
            db.stats.folders,
            db.issues.len(),
            db.stats.metas_parsed,
            db.stats.metas_reused,
            db.stats.duration.as_secs_f64() * 1000.0
        );
        Ok(db)
    }

    /// All entries in depth-first display order.
    pub fn entries(&self) -> &[AssetEntry] {
        &self.entries
    }

    /// Problems found by the scan.
    pub fn issues(&self) -> &[Issue] {
        &self.issues
    }

    /// Scan statistics.
    pub fn stats(&self) -> ScanStats {
        self.stats
    }

    /// Entry owning `guid`.
    pub fn by_guid(&self, guid: Guid) -> Option<&AssetEntry> {
        self.by_guid.get(&guid).map(|&i| &self.entries[i])
    }

    /// Entry at a project-relative `/` path.
    pub fn by_path(&self, path: &str) -> Option<&AssetEntry> {
        self.by_path.get(path).map(|&i| &self.entries[i])
    }
}
