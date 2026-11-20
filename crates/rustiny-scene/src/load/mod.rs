//! Loading a scene: stream the documents, keep what the hierarchy needs, then assemble it.

mod build;
mod collect;
#[cfg(test)]
mod tests;

use crate::classes;
use crate::model::Scene;
use crate::unity_yaml::{DocumentReader, YamlError};
use collect::Collected;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

/// Read buffer for scene files; large reads keep the 300 MB case I/O-efficient.
const READ_BUFFER_BYTES: usize = 1 << 20;

/// Why a scene could not be loaded.
#[derive(Debug, thiserror::Error)]
pub enum SceneError {
    /// The file could not be opened.
    #[error("cannot open {path}: {source}")]
    Open {
        /// The path that was tried.
        path: String,
        /// The OS error.
        #[source]
        source: io::Error,
    },
    /// Reading or parsing failed part-way.
    #[error(transparent)]
    Yaml(#[from] YamlError),
    /// The file is not text-serialized Unity data (it has no `%YAML` directive).
    #[error("not a text-serialized Unity scene (no %YAML header; binary scenes are unsupported)")]
    NotUnityYaml,
}

/// Load the scene at `path`. The file is opened read-only and never modified.
pub fn load_scene(path: &Path) -> Result<Scene, SceneError> {
    let file = File::open(path).map_err(|source| SceneError::Open {
        path: path.display().to_string(),
        source,
    })?;
    let name = path
        .file_stem()
        .map_or_else(String::new, |s| s.to_string_lossy().into_owned());
    let scene = read_scene(BufReader::with_capacity(READ_BUFFER_BYTES, file), name)?;
    tracing::debug!(
        path = %path.display(),
        objects = scene.len(),
        documents = scene.stats().documents,
        ms = scene.stats().duration.as_millis(),
        "scene loaded"
    );
    Ok(scene)
}

/// Load a scene from any buffered reader (used by tests and by [`load_scene`]).
pub fn read_scene<R: BufRead>(reader: R, name: impl Into<String>) -> Result<Scene, SceneError> {
    let started = Instant::now();
    let mut documents = DocumentReader::new(reader);
    if !documents.looks_like_unity_yaml().map_err(YamlError::from)? {
        return Err(SceneError::NotUnityYaml);
    }

    let mut collected = Collected::default();
    let mut body = String::new();
    while let Some(header) = documents.next_header()? {
        collected.note_document(header);
        let needed = match (header.class_id, header.stripped) {
            (classes::GAME_OBJECT | classes::PREFAB_INSTANCE, stripped) => !stripped,
            (
                classes::TRANSFORM
                | classes::RECT_TRANSFORM
                | classes::SCENE_ROOTS
                | classes::RENDER_SETTINGS,
                _,
            ) => true,
            (classes::LIGHT | classes::MONO_BEHAVIOUR, stripped) => !stripped,
            (
                classes::MESH_FILTER | classes::MESH_RENDERER | classes::SKINNED_MESH_RENDERER,
                stripped,
            ) => !stripped,
            _ => false,
        };
        if needed {
            documents.read_body(&mut body)?;
            collected.add(header, &body);
        }
    }

    let bytes = documents.bytes_read();
    Ok(build::assemble(collected, name.into(), bytes, started))
}
