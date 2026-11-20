//! Rustiny editor: windowing, Vulkan presentation and the Dear ImGui based editor shell.

#![cfg_attr(test, allow(clippy::unwrap_used))]

pub mod app;
pub mod cli;
pub mod gui;
pub mod meshes;
pub mod screenshot;
pub mod shaders;
pub mod ui;
