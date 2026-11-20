//! Prefabs: what a `PrefabInstance` stands for.
//!
//! A scene (or another prefab) lists a prefab instance as a source asset plus a list of property
//! overrides. To know what is drawn, the source has to be opened and the overrides applied:
//!
//! * a `.prefab` file holds game objects, transforms and components, and possibly *nested*
//!   instances of other prefabs (a prefab variant is a file that is nothing but one instance);
//! * a model file (`.fbx`) acts as a prefab whose nodes carry meshes. The model importer flattens
//!   a file into one mesh, so a model is one node here, and overrides of its mesh and materials
//!   apply to that node.
//!
//! Objects inside a prefab are named by *file IDs*. When prefab B is nested in prefab A through
//! instance `P`, an object of B with ID `b` is called `b ^ P` in A. Overrides in a scene name
//! objects in the space of the prefab they instance, so they are matched against the composed
//! IDs. Model files have node IDs derived from names that are not reproduced here: for them an
//! override is matched by the kind of property it sets (mesh, materials, name, transform), after
//! undoing the composition, using the IDs seen in overrides of direct instances.

mod expand;
mod overrides;
mod types;

pub use expand::{ExpandStats, Library, compose_world};
pub use overrides::parse_modification;
pub use types::{AssetRef, Modification, Node, Overrides, Part, RendererState, is_model_extension};
