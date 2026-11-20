//! Individual editor panels. Each panel is a module with a `draw` function and, if it keeps
//! view state between frames, a `State` type.

pub mod console;
#[cfg(test)]
mod headless_tests;
pub mod hierarchy;
pub mod inspector;
pub mod placeholder;
pub mod profiler;
pub mod project;
pub mod scene_view;
mod style;
