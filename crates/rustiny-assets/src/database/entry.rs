//! Entries of the asset index and scan statistics.

use crate::guid::Guid;
use std::time::{Duration, SystemTime};

/// The state of a `.meta` file at the time it was read. If the file still has the same size and
/// modification time, a rescan reuses what was learned instead of reading it again.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MetaStamp {
    pub size: u64,
    pub modified: Option<SystemTime>,
    /// `folderAsset: yes` as read from the file.
    pub folder_flag: bool,
}

/// One asset or folder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssetEntry {
    /// Path relative to the project root with `/` separators, e.g. `Assets/Textures/a.png`.
    pub path: String,
    /// Nesting depth below `Assets` (children of `Assets/` have depth 0).
    pub depth: u16,
    /// `true` for directories.
    pub is_folder: bool,
    /// Identity from the `.meta` file; `None` if it is missing or unreadable (see [`Issue`]).
    pub guid: Option<Guid>,
    /// Importer block name from the `.meta` file.
    pub importer: Option<String>,
    /// File size in bytes (0 for folders).
    pub size: u64,
    /// Last modification time of the asset file, if the platform reports one.
    pub modified: Option<SystemTime>,
    /// Index one past the last descendant (equals own index + 1 for files and empty folders).
    pub subtree_end: usize,
    /// Cache key for the `.meta` this entry was read from (`None` if it had none or it was bad).
    pub(crate) meta_stamp: Option<MetaStamp>,
}

impl AssetEntry {
    /// Final path component.
    pub fn name(&self) -> &str {
        self.path.rsplit('/').next().unwrap_or(&self.path)
    }
}

/// Numbers describing one scan.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ScanStats {
    /// Files (not folders) indexed.
    pub files: usize,
    /// Folders indexed (not counting `Assets` itself).
    pub folders: usize,
    /// `.meta` files read from disk and parsed by this scan.
    pub metas_parsed: usize,
    /// `.meta` files whose size and modification time were unchanged since the previous scan, so
    /// their contents were reused without reading them.
    pub metas_reused: usize,
    /// Wall-clock time of the scan.
    pub duration: Duration,
}
