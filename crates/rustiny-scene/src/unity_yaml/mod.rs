//! The UnityYAML text format: documents, fields and scalar values.
//!
//! Not a general YAML parser. Unity writes a very regular subset (fixed two-space indent, flow
//! mappings for references and vectors), and scenes can be hundreds of megabytes, so this
//! module reads exactly that subset line by line and borrows from the input.

mod fields;
mod reader;
mod scalar;

pub use fields::{Field, Fields, fields, type_name};
pub use reader::{DocumentReader, Header, YamlError};
pub use scalar::{file_id_of, flow_value, parse_vec3, parse_vec4, unquote};
