//! Filesystem helpers with crash-safety and resource-limit guarantees.

use crate::error::{Result, RustinyError};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Atomically replace `path` with `bytes`.
///
/// The data is written to a sibling temporary file, flushed to stable storage and then renamed
/// over the destination. Readers therefore observe either the complete old content or the
/// complete new content, never a torn write, even if the process dies half-way through.
pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().filter(|p| !p.as_os_str().is_empty());
    if let Some(parent) = parent {
        fs::create_dir_all(parent).map_err(|e| RustinyError::io(parent, e))?;
    }

    let tmp = temp_sibling(path);
    let write = || -> std::io::Result<()> {
        let mut file = File::create(&tmp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        Ok(())
    };
    if let Err(e) = write() {
        // Best effort cleanup; the original error is what matters to the caller.
        let _ = fs::remove_file(&tmp);
        return Err(RustinyError::io(&tmp, e));
    }
    if let Err(e) = fs::rename(&tmp, path) {
        let _ = fs::remove_file(&tmp);
        return Err(RustinyError::io(path, e));
    }
    Ok(())
}

fn temp_sibling(path: &Path) -> PathBuf {
    let mut name = path
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_default();
    name.push(format!(".{}.tmp", std::process::id()));
    path.with_file_name(name)
}

/// Read a UTF-8 text file, refusing files larger than `limit` bytes.
///
/// The size check happens on metadata *before* any allocation, so hostile or corrupt inputs
/// cannot force a huge read.
pub fn read_to_string_limited(path: &Path, limit: u64) -> Result<String> {
    let meta = fs::metadata(path).map_err(|e| RustinyError::io(path, e))?;
    if meta.len() > limit {
        return Err(RustinyError::FileTooLarge {
            path: path.to_path_buf(),
            size: meta.len(),
            limit,
        });
    }
    fs::read_to_string(path).map_err(|e| RustinyError::io(path, e))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rustiny-test-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn atomic_write_creates_parents_and_replaces_content() {
        let dir = scratch("atomic");
        let target = dir.join("nested/deeper/file.txt");
        atomic_write(&target, b"first").unwrap();
        assert_eq!(fs::read(&target).unwrap(), b"first");
        atomic_write(&target, b"second, longer").unwrap();
        assert_eq!(fs::read(&target).unwrap(), b"second, longer");
        // No temp files may be left behind.
        let leftovers: Vec<_> = fs::read_dir(target.parent().unwrap())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(leftovers.len(), 1, "unexpected files: {leftovers:?}");
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn limited_read_rejects_oversized_files() {
        let dir = scratch("limit");
        let file = dir.join("big.txt");
        fs::write(&file, vec![b'a'; 100]).unwrap();
        assert!(matches!(
            read_to_string_limited(&file, 99),
            Err(RustinyError::FileTooLarge {
                size: 100,
                limit: 99,
                ..
            })
        ));
        assert_eq!(read_to_string_limited(&file, 100).unwrap().len(), 100);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn limited_read_reports_missing_file() {
        let missing = std::env::temp_dir().join("rustiny-definitely-missing-file.toml");
        assert!(matches!(
            read_to_string_limited(&missing, 10),
            Err(RustinyError::Io { .. })
        ));
    }
}
