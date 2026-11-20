//! Per-user directories used by the editor.
//!
//! The location can be overridden with the `RUSTINY_CONFIG_DIR` environment variable, which the
//! test-suite and smoke tests use to avoid touching the real user profile.

use crate::error::{Result, RustinyError};
use std::path::PathBuf;

/// Environment variable that overrides the per-user configuration root.
pub const CONFIG_DIR_ENV: &str = "RUSTINY_CONFIG_DIR";

/// Resolved set of directories Rustiny reads and writes outside of a project.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserDirs {
    /// Root directory for configuration (`config.toml`, ImGui layout, ...).
    pub config: PathBuf,
    /// Directory for log files.
    pub logs: PathBuf,
}

impl UserDirs {
    /// Build the directory set rooted at an explicit path.
    pub fn rooted_at(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        Self {
            logs: root.join("logs"),
            config: root,
        }
    }

    /// Resolve the directories honouring [`CONFIG_DIR_ENV`], falling back to the OS convention
    /// (`%APPDATA%\Rustiny` on Windows, `~/.config/Rustiny` on Linux).
    pub fn resolve() -> Result<Self> {
        if let Some(root) = std::env::var_os(CONFIG_DIR_ENV).filter(|v| !v.is_empty()) {
            return Ok(Self::rooted_at(PathBuf::from(root)));
        }
        let dirs = directories::ProjectDirs::from("", "", "Rustiny")
            .ok_or(RustinyError::NoUserDirectory("Rustiny"))?;
        Ok(Self::rooted_at(dirs.config_dir()))
    }

    /// Path of the main configuration file.
    pub fn config_file(&self) -> PathBuf {
        self.config.join("config.toml")
    }

    /// Path of the Dear ImGui layout (`.ini`) file.
    pub fn layout_file(&self) -> PathBuf {
        self.config.join("layout.ini")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rooted_layout_is_derived_from_root() {
        let dirs = UserDirs::rooted_at("/tmp/x");
        assert_eq!(dirs.config_file(), PathBuf::from("/tmp/x/config.toml"));
        assert_eq!(dirs.layout_file(), PathBuf::from("/tmp/x/layout.ini"));
        assert_eq!(dirs.logs, PathBuf::from("/tmp/x/logs"));
    }
}
