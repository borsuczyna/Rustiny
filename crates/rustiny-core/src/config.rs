//! Editor configuration: typed, validated, size-limited and atomically persisted.

use crate::error::{Result, RustinyError};
use crate::fs::{atomic_write, read_to_string_limited};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Upper bound on the size of a configuration file we are willing to parse.
pub const MAX_CONFIG_BYTES: u64 = 1024 * 1024;

/// Swapchain presentation preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PresentModePreference {
    /// V-Sync. Always available; lowest power use.
    #[default]
    Fifo,
    /// Low latency without tearing, if the driver offers it (falls back to FIFO).
    Mailbox,
    /// No synchronisation; may tear (falls back to FIFO).
    Immediate,
}

/// Whether the Vulkan validation layer should be requested.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ValidationMode {
    /// On in debug builds, off in release builds.
    #[default]
    Auto,
    /// Always request validation; fall back with a warning if the layer is missing.
    On,
    /// Never request validation.
    Off,
}

impl ValidationMode {
    /// Resolve `Auto` against the current build profile.
    pub fn enabled(self) -> bool {
        match self {
            Self::On => true,
            Self::Off => false,
            Self::Auto => cfg!(debug_assertions),
        }
    }
}

/// Colour theme of the editor UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Theme {
    /// Dark theme (default).
    #[default]
    Dark,
    /// Light theme.
    Light,
}

/// Main window settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WindowConfig {
    /// Initial client width in logical pixels.
    pub width: u32,
    /// Initial client height in logical pixels.
    pub height: u32,
    /// Start maximised.
    pub maximized: bool,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            width: 1600,
            height: 900,
            maximized: false,
        }
    }
}

/// Rendering settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RenderConfig {
    /// Presentation mode preference.
    pub present_mode: PresentModePreference,
    /// Validation layer policy.
    pub validation: ValidationMode,
    /// Also enable Vulkan synchronisation validation when validation is active.
    pub sync_validation: bool,
    /// Substring of the GPU name to prefer (case-insensitive). Empty = automatic.
    pub gpu_preference: String,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            present_mode: PresentModePreference::default(),
            validation: ValidationMode::default(),
            sync_validation: true,
            gpu_preference: String::new(),
        }
    }
}

/// Editor UI behaviour.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct UiConfig {
    /// Colour theme.
    pub theme: Theme,
    /// Redraw every frame instead of only when something changed. Intended for profiling.
    pub continuous_redraw: bool,
    /// Frame cap while `continuous_redraw` is on (0 = uncapped, i.e. vsync-bound).
    pub max_fps: u32,
}

/// Logging settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LoggingConfig {
    /// `tracing` filter directive used when `RUST_LOG` is not set.
    pub filter: String,
    /// Mirror the log into `logs/rustiny.log`.
    pub file: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            filter: "info".to_owned(),
            file: true,
        }
    }
}

/// Complete editor configuration (`config.toml`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EditorConfig {
    /// Main window settings.
    pub window: WindowConfig,
    /// Rendering settings.
    pub render: RenderConfig,
    /// UI settings.
    pub ui: UiConfig,
    /// Logging settings.
    pub logging: LoggingConfig,
}

impl EditorConfig {
    /// Smallest accepted window dimension.
    pub const MIN_WINDOW_DIM: u32 = 320;
    /// Largest accepted window dimension.
    pub const MAX_WINDOW_DIM: u32 = 16384;

    /// Check every value against its permitted range.
    pub fn validate(&self) -> Result<()> {
        let range = Self::MIN_WINDOW_DIM..=Self::MAX_WINDOW_DIM;
        if !range.contains(&self.window.width) || !range.contains(&self.window.height) {
            return Err(RustinyError::ConfigInvalid(format!(
                "window size {}x{} must be within {}..={} on both axes",
                self.window.width,
                self.window.height,
                Self::MIN_WINDOW_DIM,
                Self::MAX_WINDOW_DIM
            )));
        }
        if self.ui.max_fps > 1000 {
            return Err(RustinyError::ConfigInvalid(format!(
                "ui.max_fps {} must be at most 1000",
                self.ui.max_fps
            )));
        }
        if self.logging.filter.len() > 1024 {
            return Err(RustinyError::ConfigInvalid(
                "logging.filter is unreasonably long".to_owned(),
            ));
        }
        Ok(())
    }

    /// Parse and validate configuration text. `origin` is only used for diagnostics.
    pub fn from_toml_str(text: &str, origin: &Path) -> Result<Self> {
        let cfg: Self = toml::from_str(text).map_err(|e| RustinyError::ConfigParse {
            path: origin.to_path_buf(),
            message: e.to_string(),
        })?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Load and validate a configuration file.
    pub fn load(path: &Path) -> Result<Self> {
        let text = read_to_string_limited(path, MAX_CONFIG_BYTES)?;
        Self::from_toml_str(&text, path)
    }

    /// Load the configuration, or fall back to defaults.
    ///
    /// * A missing file silently yields defaults.
    /// * A corrupt or invalid file is preserved as `<name>.corrupt` (so a later save cannot
    ///   destroy the user's data) and defaults are returned together with a warning message.
    pub fn load_or_default(path: &Path) -> (Self, Option<String>) {
        if !path.exists() {
            return (Self::default(), None);
        }
        match Self::load(path) {
            Ok(cfg) => (cfg, None),
            Err(err) => {
                let mut backup = path.as_os_str().to_owned();
                backup.push(".corrupt");
                let note = match std::fs::copy(path, &backup) {
                    Ok(_) => format!(
                        "{err}; original preserved as {}",
                        Path::new(&backup).display()
                    ),
                    Err(copy_err) => {
                        format!("{err}; additionally failed to back it up: {copy_err}")
                    }
                };
                (Self::default(), Some(note))
            }
        }
    }

    /// Validate and atomically persist the configuration.
    pub fn save(&self, path: &Path) -> Result<()> {
        self.validate()?;
        let text = toml::to_string_pretty(self)
            .map_err(|e| RustinyError::ConfigInvalid(format!("cannot serialise config: {e}")))?;
        atomic_write(path, text.as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("rustiny-cfg-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn defaults_are_valid() {
        EditorConfig::default().validate().unwrap();
    }

    #[test]
    fn roundtrip_through_toml() {
        let dir = scratch("roundtrip");
        let path = dir.join("config.toml");
        let mut cfg = EditorConfig::default();
        cfg.window.width = 1280;
        cfg.render.present_mode = PresentModePreference::Mailbox;
        cfg.ui.theme = Theme::Light;
        cfg.save(&path).unwrap();
        assert_eq!(EditorConfig::load(&path).unwrap(), cfg);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn partial_files_fill_in_defaults() {
        let cfg = EditorConfig::from_toml_str("[window]\nwidth = 800\n", Path::new("t")).unwrap();
        assert_eq!(cfg.window.width, 800);
        assert_eq!(cfg.window.height, WindowConfig::default().height);
    }

    #[test]
    fn unknown_keys_are_rejected() {
        let err =
            EditorConfig::from_toml_str("[window]\nwidht = 800\n", Path::new("t")).unwrap_err();
        assert!(matches!(err, RustinyError::ConfigParse { .. }), "{err}");
    }

    #[test]
    fn out_of_range_values_are_rejected() {
        let err =
            EditorConfig::from_toml_str("[window]\nwidth = 10\n", Path::new("t")).unwrap_err();
        assert!(matches!(err, RustinyError::ConfigInvalid(_)), "{err}");
        let err =
            EditorConfig::from_toml_str("[ui]\nmax_fps = 5000\n", Path::new("t")).unwrap_err();
        assert!(matches!(err, RustinyError::ConfigInvalid(_)), "{err}");
    }

    #[test]
    fn corrupt_file_is_backed_up_and_defaults_returned() {
        let dir = scratch("corrupt");
        let path = dir.join("config.toml");
        fs::write(&path, "this is = = not toml").unwrap();
        let (cfg, warning) = EditorConfig::load_or_default(&path);
        assert_eq!(cfg, EditorConfig::default());
        assert!(warning.unwrap().contains("preserved"));
        assert_eq!(
            fs::read_to_string(dir.join("config.toml.corrupt")).unwrap(),
            "this is = = not toml"
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn missing_file_yields_defaults_without_warning() {
        let dir = scratch("missing");
        let (cfg, warning) = EditorConfig::load_or_default(&dir.join("nope.toml"));
        assert_eq!(cfg, EditorConfig::default());
        assert!(warning.is_none());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn oversized_file_is_refused() {
        let dir = scratch("oversized");
        let path = dir.join("config.toml");
        fs::write(&path, vec![b'#'; (MAX_CONFIG_BYTES + 1) as usize]).unwrap();
        assert!(matches!(
            EditorConfig::load(&path),
            Err(RustinyError::FileTooLarge { .. })
        ));
        let _ = fs::remove_dir_all(dir);
    }
}
