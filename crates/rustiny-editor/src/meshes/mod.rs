//! Real geometry in the Scene view: which objects have a model, loading those models in the
//! background and handing them to the GPU.
//!
//! * `plan`    - [`MeshPlan`]: from a scene and a GUID lookup, which objects draw which model and
//!   which stay proxy boxes. Pure and testable without a GPU or a project on disk.
//! * `stream`  - [`MeshStream`]: imports the planned models on worker threads, never on the UI
//!   thread, and hands the results over a bounded channel.
//! * `convert` - imported [`MeshData`](rustiny_model::MeshData) to the renderer's vertex format.
//! * `materials` - matching a model's material names to `.mat` files and turning them into looks.
//! * `textures` - decoding the albedo PNGs on worker threads.
//! * `pick`    - [`PickShapes`]: the box each mesh object is clicked with.
//!
//! What this covers: prefab instances of `.fbx` models, the highest-detail variant only, with the
//! tint, albedo texture and alpha cutoff of the `.mat` whose name matches the model's material.
//! It does not expand `.prefab` assets, read `MeshFilter` components or the renderers' material
//! lists yet.

mod convert;
mod materials;
mod pick;
mod plan;
mod stream;
mod textures;

pub use convert::{DEFAULT_TINT, Look, Upload, to_upload};
pub use materials::{
    LookTextures, MaterialIndex, ResolveStats, Resolver, look_of, slot_of_materials,
};
pub use pick::{PickShapes, pick_matrix};
pub use plan::{MaterialRef, MeshPlan, MeshSource, SourceKind, build_plan, lod_of, proxy_box};
pub use stream::{Loaded, LoadedMesh, LooksFn, MeshStream};
pub use textures::{
    DecodedTexture, LoadedTexture, MAX_EDGE, TextureJob, TextureStream, decode_png,
    decode_texture_asset, document_of,
};
