//! Foundational building blocks shared by every Rustiny crate.
//!
//! This crate deliberately has no graphics, windowing or scripting dependencies so that it can be
//! used from headless tests, importers and the standalone player alike.

#![cfg_attr(test, allow(clippy::unwrap_used))]

pub mod config;
pub mod error;
pub mod fs;
pub mod handle;
pub mod logging;
pub mod paths;

pub use error::{Result, RustinyError};
pub use handle::{Handle, Pool};

/// Human readable engine name.
pub const ENGINE_NAME: &str = "Rustiny";

/// Engine version as declared in the workspace manifest.
pub const ENGINE_VERSION: &str = env!("CARGO_PKG_VERSION");
