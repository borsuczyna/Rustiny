//! Engine-wide error type.

use std::path::PathBuf;

/// Convenience alias used throughout the engine.
pub type Result<T, E = RustinyError> = std::result::Result<T, E>;

/// Errors produced by the foundational layer.
///
/// Higher level crates define their own error enums and convert into this type at API
/// boundaries where a single error type is required.
#[derive(Debug, thiserror::Error)]
pub enum RustinyError {
    /// An I/O operation failed.
    #[error("I/O error on `{}`: {source}", path.display())]
    Io {
        /// Path involved in the failed operation.
        path: PathBuf,
        /// Underlying error.
        #[source]
        source: std::io::Error,
    },

    /// A file exceeded the configured size limit before it was even parsed.
    #[error("`{}` is {size} bytes, which exceeds the {limit} byte limit", path.display())]
    FileTooLarge {
        /// Offending file.
        path: PathBuf,
        /// Actual size in bytes.
        size: u64,
        /// Maximum accepted size in bytes.
        limit: u64,
    },

    /// A configuration file could not be parsed.
    #[error("failed to parse configuration `{}`: {message}", path.display())]
    ConfigParse {
        /// Offending file.
        path: PathBuf,
        /// Parser diagnostic.
        message: String,
    },

    /// A configuration value is outside its permitted range.
    #[error("invalid configuration: {0}")]
    ConfigInvalid(String),

    /// The logging subsystem could not be initialised.
    #[error("logging initialisation failed: {0}")]
    Logging(String),

    /// No suitable per-user directory could be determined.
    #[error("could not determine a per-user directory for {0}")]
    NoUserDirectory(&'static str),
}

impl RustinyError {
    /// Wrap an [`std::io::Error`] together with the path it relates to.
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}
