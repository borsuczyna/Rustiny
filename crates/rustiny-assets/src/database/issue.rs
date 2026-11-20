//! Problems a scan can report.

use crate::guid::Guid;
use std::fmt;

/// A problem found while scanning. None of these stops the scan, and none is "repaired" on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Issue {
    /// An asset has no `.meta` file. Unity would create one (a new GUID); Rustiny does not.
    MissingMeta {
        /// Asset path.
        path: String,
    },
    /// A `.meta` file whose asset does not exist.
    OrphanMeta {
        /// The `.meta` path.
        path: String,
    },
    /// A `.meta` file that could not be read or understood.
    InvalidMeta {
        /// The `.meta` path.
        path: String,
        /// What was wrong.
        reason: String,
    },
    /// The same GUID is used by several assets.
    DuplicateGuid {
        /// The shared GUID.
        guid: Guid,
        /// Every asset that claims it.
        paths: Vec<String>,
    },
    /// `folderAsset` in the `.meta` does not match what is on disk.
    KindMismatch {
        /// Asset path.
        path: String,
    },
    /// A directory could not be listed.
    Unreadable {
        /// The directory.
        path: String,
        /// OS error text.
        reason: String,
    },
    /// Symbolic links are not followed (they can form loops).
    SymlinkSkipped {
        /// The link.
        path: String,
    },
    /// A file name that is not valid Unicode.
    NonUnicodeName {
        /// Lossy rendering of the path.
        path: String,
    },
}

impl fmt::Display for Issue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingMeta { path } => write!(f, "{path}: no .meta file"),
            Self::OrphanMeta { path } => write!(f, "{path}: .meta without an asset"),
            Self::InvalidMeta { path, reason } => write!(f, "{path}: invalid .meta ({reason})"),
            Self::DuplicateGuid { guid, paths } => {
                write!(
                    f,
                    "GUID {guid} is used by {} assets: {}",
                    paths.len(),
                    paths.join(", ")
                )
            }
            Self::KindMismatch { path } => {
                write!(
                    f,
                    "{path}: .meta folder flag does not match the file system"
                )
            }
            Self::Unreadable { path, reason } => write!(f, "{path}: cannot list ({reason})"),
            Self::SymlinkSkipped { path } => write!(f, "{path}: symbolic link not followed"),
            Self::NonUnicodeName { path } => write!(f, "{path}: name is not valid Unicode"),
        }
    }
}
