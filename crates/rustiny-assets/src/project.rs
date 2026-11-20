//! Detecting a Unity project folder.

use std::path::{Path, PathBuf};

/// A folder recognised as a Unity project.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnityProject {
    /// Project root (the folder that contains `Assets/`).
    pub root: PathBuf,
    /// `m_EditorVersion` from `ProjectSettings/ProjectVersion.txt`, if readable.
    pub editor_version: Option<String>,
    /// Whether `Packages/manifest.json` exists.
    pub has_package_manifest: bool,
}

/// Why a folder is not usable as a project.
#[derive(Debug, thiserror::Error)]
pub enum ProjectError {
    /// The path is not an existing directory.
    #[error("`{}` is not a directory", .0.display())]
    NotADirectory(PathBuf),
    /// A required sub-folder is missing.
    #[error("`{}` does not look like a Unity project: missing `{missing}/`", root.display())]
    MissingFolder {
        /// The inspected folder.
        root: PathBuf,
        /// `Assets` or `ProjectSettings`.
        missing: &'static str,
    },
}

impl UnityProject {
    /// Check that `path` has the layout of a Unity project (`Assets/` and `ProjectSettings/`).
    ///
    /// Only reads; never creates or changes anything.
    pub fn detect(path: &Path) -> Result<Self, ProjectError> {
        if !path.is_dir() {
            return Err(ProjectError::NotADirectory(path.to_owned()));
        }
        for missing in ["Assets", "ProjectSettings"] {
            if !path.join(missing).is_dir() {
                return Err(ProjectError::MissingFolder {
                    root: path.to_owned(),
                    missing,
                });
            }
        }
        let editor_version =
            std::fs::read_to_string(path.join("ProjectSettings").join("ProjectVersion.txt"))
                .ok()
                .and_then(|text| parse_editor_version(&text));
        Ok(Self {
            root: path.to_owned(),
            editor_version,
            has_package_manifest: path.join("Packages").join("manifest.json").is_file(),
        })
    }

    /// Folder name, for display.
    pub fn name(&self) -> String {
        self.root.file_name().map_or_else(
            || self.root.display().to_string(),
            |n| n.to_string_lossy().into_owned(),
        )
    }
}

fn parse_editor_version(text: &str) -> Option<String> {
    text.lines().find_map(|line| {
        let value = line.strip_prefix("m_EditorVersion:")?.trim();
        (!value.is_empty()).then(|| value.to_owned())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn editor_version_is_read_from_the_right_line() {
        let text = "m_EditorVersion: 2022.3.10f1\nm_EditorVersionWithRevision: 2022.3.10f1 (ff3792e53c62)\n";
        assert_eq!(parse_editor_version(text).as_deref(), Some("2022.3.10f1"));
        assert_eq!(parse_editor_version("nothing here"), None);
        assert_eq!(parse_editor_version("m_EditorVersion:\n"), None);
    }
}
