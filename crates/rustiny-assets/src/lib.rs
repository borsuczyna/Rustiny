//! Unity project detection, `.meta` / GUID handling and the asset database.
//!
//! Compatibility rule enforced here: **Rustiny never writes to a Unity project while scanning**.
//! GUIDs are only ever read from existing `.meta` files; an asset without a valid `.meta` is
//! reported as an [`Issue`], not silently given a new identity.

#![cfg_attr(test, allow(clippy::unwrap_used))]

pub mod database;
pub mod guid;
pub mod meta;
pub mod project;
pub mod watch;

pub use database::{AssetDatabase, AssetEntry, Issue, ScanStats, is_ignored_name};
pub use guid::Guid;
pub use meta::{MetaData, MetaError};
pub use project::{ProjectError, UnityProject};
pub use watch::{ProjectWatcher, WatchError};
