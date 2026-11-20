//! What a compiled shader needs, read from its SPIR-V: vertex inputs, constant buffers with their
//! members and offsets, textures and samplers.
//!
//! This is a small reader for the instructions DXC emits for Unity-style HLSL (no `spirv-cross`
//! or other dependency). It is meant for filling constant buffers by name and building descriptor
//! set layouts, not for validating SPIR-V.

use std::collections::HashMap;
use thiserror::Error;

/// The SPIR-V could not be read.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ReflectError {
    /// Wrong magic number or fewer words than a header.
    #[error("not SPIR-V")]
    NotSpirv,
    /// An instruction's length reaches past the end of the module.
    #[error("truncated SPIR-V instruction")]
    Truncated,
}

/// The scalar kind of a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scalar {
    /// `float`.
    Float,
    /// `int`.
    Int,
    /// `uint`.
    Uint,
    /// `bool`.
    Bool,
}

/// The shape of a value in a constant buffer or a stage input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Shape {
    /// `float`, `int`, ...
    Scalar(Scalar),
    /// `float4`, `int2`, ...
    Vector(Scalar, u32),
    /// `float4x4`: columns, rows.
    Matrix(u32, u32),
    /// An array of `count` elements, `stride` bytes apart.
    Array(Box<Shape>, u32, u32),
    /// A structure (its members are not expanded).
    Struct,
}

/// One member of a constant buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Member {
    /// Name in the source (`_BaseColor`).
    pub name: String,
    /// Byte offset in the buffer.
    pub offset: u32,
    /// Byte size.
    pub size: u32,
    /// Type.
    pub shape: Shape,
}

/// A constant buffer (`cbuffer`, or the `$Globals` block DXC makes of loose globals).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UniformBlock {
    /// Name of the block (`UnityPerMaterial`, `$Globals`, ...).
    pub name: String,
    /// Descriptor set.
    pub set: u32,
    /// Binding in the set.
    pub binding: u32,
    /// Size in bytes.
    pub size: u32,
    /// Members in declaration order.
    pub members: Vec<Member>,
}

/// What kind of descriptor a resource is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceKind {
    /// A sampled 2D texture.
    Texture2D,
    /// A sampled 3D texture.
    Texture3D,
    /// A sampled cube texture.
    TextureCube,
    /// A 2D texture array.
    Texture2DArray,
    /// Another image type (buffers, storage images, multisampled, ...).
    OtherImage,
    /// A sampler.
    Sampler,
    /// A structured buffer (`StructuredBuffer<T>`).
    StorageBuffer,
}

/// A texture, sampler or storage buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resource {
    /// Name in the source (`_BaseMap`, `sampler_BaseMap`).
    pub name: String,
    /// Descriptor set.
    pub set: u32,
    /// Binding in the set.
    pub binding: u32,
    /// Kind.
    pub kind: ResourceKind,
}

/// A vertex shader input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageInput {
    /// Location.
    pub location: u32,
    /// HLSL semantic (`POSITION`, `TEXCOORD0`, `SV_InstanceID`, ...), if DXC kept it.
    pub semantic: Option<String>,
    /// Variable name (`in.var.POSITION`).
    pub name: String,
    /// Type.
    pub shape: Shape,
    /// A built-in (`SV_VertexID`, `SV_InstanceID`) rather than a vertex attribute.
    pub builtin: bool,
}

/// Everything one entry point needs.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Reflection {
    /// Stage inputs (vertex attributes for a vertex shader; interpolants for a fragment shader).
    pub inputs: Vec<StageInput>,
    /// Constant buffers.
    pub blocks: Vec<UniformBlock>,
    /// Textures, samplers and storage buffers.
    pub resources: Vec<Resource>,
    /// Size in bytes of the push constant block, 0 if none.
    pub push_constant_size: u32,
}

impl Reflection {
    /// The constant buffer named `name`.
    pub fn block(&self, name: &str) -> Option<&UniformBlock> {
        self.blocks.iter().find(|b| b.name == name)
    }

    /// The texture, sampler or buffer named `name`.
    pub fn resource(&self, name: &str) -> Option<&Resource> {
        self.resources.iter().find(|r| r.name == name)
    }
}

const OP_NAME: u32 = 5;
const OP_MEMBER_NAME: u32 = 6;
const OP_ENTRY_POINT: u32 = 15;
const OP_TYPE_BOOL: u32 = 20;
const OP_TYPE_INT: u32 = 21;
const OP_TYPE_FLOAT: u32 = 22;
const OP_TYPE_VECTOR: u32 = 23;
const OP_TYPE_MATRIX: u32 = 24;
const OP_TYPE_IMAGE: u32 = 25;
const OP_TYPE_SAMPLER: u32 = 26;
const OP_TYPE_SAMPLED_IMAGE: u32 = 27;
const OP_TYPE_ARRAY: u32 = 28;
const OP_TYPE_RUNTIME_ARRAY: u32 = 29;
const OP_TYPE_STRUCT: u32 = 30;
const OP_TYPE_POINTER: u32 = 32;
const OP_CONSTANT: u32 = 43;
const OP_VARIABLE: u32 = 59;
const OP_DECORATE: u32 = 71;
const OP_MEMBER_DECORATE: u32 = 72;
const OP_DECORATE_STRING: u32 = 5632;

const DEC_BLOCK: u32 = 2;
const DEC_ARRAY_STRIDE: u32 = 6;
const DEC_BUILTIN: u32 = 11;
const DEC_LOCATION: u32 = 30;
const DEC_BINDING: u32 = 33;
const DEC_DESCRIPTOR_SET: u32 = 34;
const DEC_OFFSET: u32 = 35;
const DEC_MATRIX_STRIDE: u32 = 7;
const DEC_USER_SEMANTIC: u32 = 5635;

const SC_UNIFORM_CONSTANT: u32 = 0;
const SC_INPUT: u32 = 1;
const SC_UNIFORM: u32 = 2;
const SC_PUSH_CONSTANT: u32 = 9;
const SC_STORAGE_BUFFER: u32 = 12;

#[derive(Debug, Clone)]
enum Ty {
    Bool,
    Int {
        signed: bool,
    },
    Float,
    Vector(u32, u32),
    Matrix(u32, u32),
    Image {
        dim: u32,
        arrayed: bool,
        sampled: u32,
    },
    Sampler,
    SampledImage,
    Array(u32, u32),
    RuntimeArray(u32),
    Struct(Vec<u32>),
    Pointer(u32),
}

#[derive(Default)]
struct Module {
    names: HashMap<u32, String>,
    member_names: HashMap<(u32, u32), String>,
    decorations: HashMap<u32, Vec<(u32, u32)>>,
    member_decorations: HashMap<(u32, u32), Vec<(u32, u32)>>,
    semantics: HashMap<u32, String>,
    types: HashMap<u32, Ty>,
    constants: HashMap<u32, u32>,
    variables: Vec<(u32, u32, u32)>,
    entry_interface: Vec<u32>,
}

fn string_of(words: &[u32]) -> String {
    let mut bytes = Vec::new();
    'outer: for word in words {
        for byte in word.to_le_bytes() {
            if byte == 0 {
                break 'outer;
            }
            bytes.push(byte);
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

impl Module {
    fn decoration(&self, id: u32, which: u32) -> Option<u32> {
        self.decorations
            .get(&id)?
            .iter()
            .find(|(d, _)| *d == which)
            .map(|(_, v)| *v)
    }

    fn has_decoration(&self, id: u32, which: u32) -> bool {
        self.decorations
            .get(&id)
            .is_some_and(|d| d.iter().any(|(k, _)| *k == which))
    }

    fn member_decoration(&self, id: u32, member: u32, which: u32) -> Option<u32> {
        self.member_decorations
            .get(&(id, member))?
            .iter()
            .find(|(d, _)| *d == which)
            .map(|(_, v)| *v)
    }

    fn scalar_of(&self, id: u32) -> Option<Scalar> {
        match self.types.get(&id)? {
            Ty::Bool => Some(Scalar::Bool),
            Ty::Int { signed: true } => Some(Scalar::Int),
            Ty::Int { signed: false } => Some(Scalar::Uint),
            Ty::Float => Some(Scalar::Float),
            _ => None,
        }
    }

    fn shape(&self, id: u32) -> Shape {
        match self.types.get(&id) {
            Some(Ty::Vector(component, count)) => {
                Shape::Vector(self.scalar_of(*component).unwrap_or(Scalar::Float), *count)
            }
            Some(Ty::Matrix(column, columns)) => match self.types.get(column) {
                Some(Ty::Vector(_, rows)) => Shape::Matrix(*columns, *rows),
                _ => Shape::Struct,
            },
            Some(Ty::Array(element, length)) => {
                let count = self.constants.get(length).copied().unwrap_or(0);
                let stride = self.decoration(id, DEC_ARRAY_STRIDE).unwrap_or(0);
                Shape::Array(Box::new(self.shape(*element)), count, stride)
            }
            Some(Ty::Struct(_)) => Shape::Struct,
            Some(_) => self.scalar_of(id).map_or(Shape::Struct, Shape::Scalar),
            None => Shape::Struct,
        }
    }

    /// Size in bytes of a type laid out in a constant buffer.
    fn size_of(&self, id: u32) -> u32 {
        match self.types.get(&id) {
            Some(Ty::Bool | Ty::Int { .. } | Ty::Float) => 4,
            Some(Ty::Vector(_, count)) => 4 * count,
            Some(Ty::Matrix(column, columns)) => self.size_of(*column) * columns,
            Some(Ty::Array(element, length)) => {
                let count = self.constants.get(length).copied().unwrap_or(0);
                let stride = self
                    .decoration(id, DEC_ARRAY_STRIDE)
                    .unwrap_or_else(|| self.size_of(*element));
                stride * count
            }
            Some(Ty::Struct(members)) => {
                let mut end = 0;
                for (index, member) in members.iter().enumerate() {
                    let offset = self
                        .member_decoration(id, index as u32, DEC_OFFSET)
                        .unwrap_or(0);
                    end = end.max(offset + self.member_size(id, index as u32, *member));
                }
                end
            }
            _ => 0,
        }
    }

    fn member_size(&self, parent: u32, index: u32, member: u32) -> u32 {
        if let Some(Ty::Matrix(column, columns)) = self.types.get(&member) {
            let stride = self
                .member_decoration(parent, index, DEC_MATRIX_STRIDE)
                .unwrap_or_else(|| self.size_of(*column));
            return stride * columns;
        }
        if let Some(Ty::Array(element, length)) = self.types.get(&member) {
            let count = self.constants.get(length).copied().unwrap_or(0);
            let stride = self
                .decoration(member, DEC_ARRAY_STRIDE)
                .unwrap_or_else(|| self.size_of(*element));
            // An array of matrices takes its element stride from the array decoration.
            return stride * count;
        }
        self.size_of(member)
    }

    fn name(&self, id: u32) -> String {
        self.names.get(&id).cloned().unwrap_or_default()
    }

    fn resource_kind(&self, pointee: u32) -> Option<ResourceKind> {
        let mut ty = self.types.get(&pointee)?;
        // A texture array of resources (`Texture2D x[4]`) is one resource of that kind here.
        if let Ty::Array(element, _) | Ty::RuntimeArray(element) = ty {
            ty = self.types.get(element)?;
        }
        Some(match ty {
            Ty::Sampler => ResourceKind::Sampler,
            Ty::SampledImage => ResourceKind::Texture2D,
            Ty::Image {
                dim,
                arrayed,
                sampled,
            } => match (*dim, *arrayed, *sampled) {
                (1, false, 1) => ResourceKind::Texture2D,
                (1, true, 1) => ResourceKind::Texture2DArray,
                (2, false, 1) => ResourceKind::Texture3D,
                (3, false, 1) => ResourceKind::TextureCube,
                _ => ResourceKind::OtherImage,
            },
            Ty::RuntimeArray(_) | Ty::Struct(_) => ResourceKind::StorageBuffer,
            _ => return None,
        })
    }
}

/// The HLSL semantic of an input: the one DXC recorded, or the one in the variable's name
/// (`in.var.TEXCOORD0`).
fn semantic_of(module: &Module, var: u32) -> Option<String> {
    module.semantics.get(&var).cloned().or_else(|| {
        module
            .names
            .get(&var)
            .and_then(|n| n.strip_prefix("in.var."))
            .map(str::to_owned)
    })
}

fn parse(words: &[u32]) -> Result<Module, ReflectError> {
    if words.len() < 5 || words[0] != 0x0723_0203 {
        return Err(ReflectError::NotSpirv);
    }
    let mut module = Module::default();
    let mut at = 5;
    while at < words.len() {
        let head = words[at];
        let (count, op) = ((head >> 16) as usize, head & 0xffff);
        if count == 0 || at + count > words.len() {
            return Err(ReflectError::Truncated);
        }
        let w = &words[at + 1..at + count];
        let get = |i: usize| w.get(i).copied().unwrap_or(0);
        match op {
            OP_NAME => {
                module.names.insert(get(0), string_of(&w[1.min(w.len())..]));
            }
            OP_MEMBER_NAME => {
                module
                    .member_names
                    .insert((get(0), get(1)), string_of(&w[2.min(w.len())..]));
            }
            OP_ENTRY_POINT => {
                // execution model, function id, name (nul-terminated words), then the interface.
                let name_words = w[2.min(w.len())..]
                    .iter()
                    .position(|word| word.to_le_bytes().contains(&0))
                    .map_or(0, |p| p + 1);
                module.entry_interface = w[(2 + name_words).min(w.len())..].to_vec();
            }
            OP_TYPE_BOOL => {
                module.types.insert(get(0), Ty::Bool);
            }
            OP_TYPE_INT => {
                module.types.insert(
                    get(0),
                    Ty::Int {
                        signed: get(2) == 1,
                    },
                );
            }
            OP_TYPE_FLOAT => {
                module.types.insert(get(0), Ty::Float);
            }
            OP_TYPE_VECTOR => {
                module.types.insert(get(0), Ty::Vector(get(1), get(2)));
            }
            OP_TYPE_MATRIX => {
                module.types.insert(get(0), Ty::Matrix(get(1), get(2)));
            }
            OP_TYPE_IMAGE => {
                module.types.insert(
                    get(0),
                    Ty::Image {
                        dim: get(2),
                        arrayed: get(4) == 1,
                        sampled: get(6),
                    },
                );
            }
            OP_TYPE_SAMPLER => {
                module.types.insert(get(0), Ty::Sampler);
            }
            OP_TYPE_SAMPLED_IMAGE => {
                module.types.insert(get(0), Ty::SampledImage);
            }
            OP_TYPE_ARRAY => {
                module.types.insert(get(0), Ty::Array(get(1), get(2)));
            }
            OP_TYPE_RUNTIME_ARRAY => {
                module.types.insert(get(0), Ty::RuntimeArray(get(1)));
            }
            OP_TYPE_STRUCT => {
                module
                    .types
                    .insert(get(0), Ty::Struct(w[1.min(w.len())..].to_vec()));
            }
            OP_TYPE_POINTER => {
                module.types.insert(get(0), Ty::Pointer(get(2)));
            }
            OP_CONSTANT => {
                module.constants.insert(get(1), get(2));
            }
            OP_VARIABLE => module.variables.push((get(1), get(0), get(2))),
            OP_DECORATE => {
                let extra = if w.len() > 2 { get(2) } else { 0 };
                module
                    .decorations
                    .entry(get(0))
                    .or_default()
                    .push((get(1), extra));
            }
            OP_MEMBER_DECORATE => {
                let extra = if w.len() > 3 { get(3) } else { 0 };
                module
                    .member_decorations
                    .entry((get(0), get(1)))
                    .or_default()
                    .push((get(2), extra));
            }
            OP_DECORATE_STRING if get(1) == DEC_USER_SEMANTIC => {
                module
                    .semantics
                    .insert(get(0), string_of(&w[2.min(w.len())..]));
            }
            _ => {}
        }
        at += count;
    }
    Ok(module)
}

/// Read what the module needs.
///
/// # Errors
///
/// [`ReflectError`] when `words` is not a well-formed SPIR-V module.
pub fn reflect(words: &[u32]) -> Result<Reflection, ReflectError> {
    let module = parse(words)?;
    let mut out = Reflection::default();
    for &(var, pointer_ty, storage) in &module.variables {
        let Some(Ty::Pointer(pointee)) = module.types.get(&pointer_ty).cloned() else {
            continue;
        };
        match storage {
            SC_INPUT if module.entry_interface.contains(&var) => {
                let builtin = module.has_decoration(var, DEC_BUILTIN);
                if let Some(location) = module.decoration(var, DEC_LOCATION) {
                    out.inputs.push(StageInput {
                        location,
                        semantic: semantic_of(&module, var),
                        name: module.name(var),
                        shape: module.shape(pointee),
                        builtin: false,
                    });
                } else if builtin {
                    out.inputs.push(StageInput {
                        location: u32::MAX,
                        semantic: semantic_of(&module, var),
                        name: module.name(var),
                        shape: module.shape(pointee),
                        builtin: true,
                    });
                }
            }
            SC_UNIFORM
                if module
                    .types
                    .get(&pointee)
                    .is_some_and(|t| matches!(t, Ty::Struct(_)))
                    && module.has_decoration(pointee, DEC_BLOCK) =>
            {
                let Some(Ty::Struct(members)) = module.types.get(&pointee) else {
                    continue;
                };
                let block_members: Vec<Member> = members
                    .iter()
                    .enumerate()
                    .map(|(index, &member)| {
                        let index = index as u32;
                        Member {
                            name: module
                                .member_names
                                .get(&(pointee, index))
                                .cloned()
                                .unwrap_or_default(),
                            offset: module
                                .member_decoration(pointee, index, DEC_OFFSET)
                                .unwrap_or(0),
                            size: module.member_size(pointee, index, member),
                            shape: module.shape(member),
                        }
                    })
                    .collect();
                // The block's name is on the type for `cbuffer`s (`type.UnityPerMaterial`).
                let name = module.name(pointee);
                let name = name.strip_prefix("type.").unwrap_or(&name).to_owned();
                out.blocks.push(UniformBlock {
                    name: if name.is_empty() {
                        module.name(var)
                    } else {
                        name
                    },
                    set: module.decoration(var, DEC_DESCRIPTOR_SET).unwrap_or(0),
                    binding: module.decoration(var, DEC_BINDING).unwrap_or(0),
                    size: module.size_of(pointee),
                    members: block_members,
                });
            }
            SC_UNIFORM_CONSTANT | SC_STORAGE_BUFFER | SC_UNIFORM => {
                if let Some(kind) = module.resource_kind(pointee) {
                    out.resources.push(Resource {
                        name: module.name(var),
                        set: module.decoration(var, DEC_DESCRIPTOR_SET).unwrap_or(0),
                        binding: module.decoration(var, DEC_BINDING).unwrap_or(0),
                        kind,
                    });
                }
            }
            SC_PUSH_CONSTANT => out.push_constant_size = module.size_of(pointee),
            _ => {}
        }
    }
    out.inputs.sort_by_key(|i| (i.builtin, i.location));
    out.blocks.sort_by_key(|b| (b.set, b.binding));
    out.resources.sort_by_key(|r| (r.set, r.binding));
    Ok(out)
}

/// Give the descriptors of a module new `(set, binding)` pairs, chosen by name.
///
/// DXC numbers the bindings of each stage on its own, so a vertex and a fragment shader that both
/// use `UnityPerMaterial` can disagree about where it is. The renderer reflects both stages,
/// decides one numbering for the pass and patches both modules with this function. `assign` gets
/// the name as [`reflect`] reports it (the block name for constant buffers, the variable name for
/// textures, samplers and buffers); a descriptor it returns `None` for keeps its binding. Returns
/// how many descriptors were changed.
///
/// # Errors
///
/// [`ReflectError`] when `words` is not a well-formed SPIR-V module.
pub fn rebind(
    words: &mut [u32],
    assign: impl Fn(&str) -> Option<(u32, u32)>,
) -> Result<usize, ReflectError> {
    let module = parse(words)?;
    let mut targets: HashMap<u32, (u32, u32)> = HashMap::new();
    for &(var, pointer_ty, storage) in &module.variables {
        if !matches!(
            storage,
            SC_UNIFORM_CONSTANT | SC_UNIFORM | SC_STORAGE_BUFFER
        ) {
            continue;
        }
        let Some(Ty::Pointer(pointee)) = module.types.get(&pointer_ty) else {
            continue;
        };
        if module.decoration(var, DEC_BINDING).is_none() {
            continue;
        }
        let name = descriptor_name(&module, var, *pointee);
        if let Some(target) = assign(&name) {
            targets.insert(var, target);
        }
    }
    let mut changed = 0;
    let mut at = 5;
    while at < words.len() {
        let head = words[at];
        let (count, op) = ((head >> 16) as usize, head & 0xffff);
        if count == 0 || at + count > words.len() {
            return Err(ReflectError::Truncated);
        }
        if op == OP_DECORATE
            && count >= 4
            && let Some(&(set, binding)) = targets.get(&words[at + 1])
        {
            match words[at + 2] {
                DEC_BINDING if words[at + 3] != binding => {
                    words[at + 3] = binding;
                    changed += 1;
                }
                DEC_DESCRIPTOR_SET if words[at + 3] != set => words[at + 3] = set,
                _ => {}
            }
        }
        at += count;
    }
    Ok(changed)
}

/// The name [`reflect`] gives a descriptor variable.
fn descriptor_name(module: &Module, var: u32, pointee: u32) -> String {
    let is_block = matches!(module.types.get(&pointee), Some(Ty::Struct(_)))
        && module.has_decoration(pointee, DEC_BLOCK);
    if is_block {
        let name = module.name(pointee);
        let name = name.strip_prefix("type.").unwrap_or(&name);
        if !name.is_empty() {
            return name.to_owned();
        }
    }
    module.name(var)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::{Compiler, Request, Stage, locate_dxc};

    const SOURCE: &str = r#"
cbuffer UnityPerMaterial { float4 _BaseColor; float _Cutoff; float4x4 _Matrix; float4 _Array[3]; };
float4 _Loose;
Texture2D _BaseMap; SamplerState sampler_BaseMap;
TextureCube _Cube;
struct A { float3 p : POSITION; float3 n : NORMAL; float2 uv : TEXCOORD0; uint id : SV_InstanceID; };
struct V { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
V vert(A a) { V v; v.pos = mul(_Matrix, float4(a.p, 1)) * _Cutoff + _Array[a.id % 3]; v.uv = a.uv + a.n.xy; return v; }
float4 frag(V v) : SV_Target { return _BaseMap.Sample(sampler_BaseMap, v.uv) * _BaseColor * _Loose + _Cube.Sample(sampler_BaseMap, float3(0,0,1)); }
"#;

    fn compile(entry: &str, stage: Stage) -> Option<Vec<u32>> {
        let dxc = locate_dxc()?;
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "rustiny-reflect-{}-{entry}-{n}",
            std::process::id()
        ));
        std::fs::create_dir_all(dir.join("Packages")).unwrap();
        let compiler = Compiler::new(dxc, &dir, &dir);
        let spirv = compiler
            .compile(&Request {
                source: SOURCE,
                entry,
                stage,
                keywords: &[],
                include_dirs: &[],
            })
            .unwrap()
            .spirv;
        let _ = std::fs::remove_dir_all(dir);
        Some(spirv)
    }

    #[test]
    fn garbage_is_an_error() {
        assert_eq!(reflect(&[1, 2, 3]), Err(ReflectError::NotSpirv));
        assert_eq!(
            reflect(&[0x0723_0203, 0, 0, 0, 0, 0x0005_0005]),
            Err(ReflectError::Truncated)
        );
    }

    #[test]
    fn a_vertex_shader_reports_inputs_blocks_and_offsets() {
        let Some(spirv) = compile("vert", Stage::Vertex) else {
            eprintln!("dxc not found: skipping");
            return;
        };
        let r = reflect(&spirv).unwrap();
        let semantics: Vec<(u32, Option<&str>)> = r
            .inputs
            .iter()
            .filter(|i| !i.builtin)
            .map(|i| (i.location, i.semantic.as_deref()))
            .collect();
        assert_eq!(
            semantics,
            vec![
                (0, Some("POSITION")),
                (1, Some("NORMAL")),
                (2, Some("TEXCOORD0"))
            ]
        );
        assert!(
            r.inputs.iter().any(|i| i.builtin),
            "SV_InstanceID is a built-in"
        );
        assert_eq!(r.inputs[0].shape, Shape::Vector(Scalar::Float, 3));

        let block = r.block("UnityPerMaterial").unwrap();
        let by_name = |n: &str| block.members.iter().find(|m| m.name == n).unwrap();
        assert_eq!(by_name("_BaseColor").offset, 0);
        assert_eq!(by_name("_BaseColor").shape, Shape::Vector(Scalar::Float, 4));
        assert_eq!(by_name("_Cutoff").offset, 16);
        assert_eq!(by_name("_Matrix").size, 64);
        assert_eq!(by_name("_Matrix").shape, Shape::Matrix(4, 4));
        assert_eq!(by_name("_Array").size, 48);
        assert!(block.size >= 16 + 16 + 64 + 48 - 12);
        assert_eq!(r.push_constant_size, 0);
    }

    #[test]
    fn a_fragment_shader_reports_textures_samplers_and_loose_globals() {
        let Some(spirv) = compile("frag", Stage::Fragment) else {
            eprintln!("dxc not found: skipping");
            return;
        };
        let r = reflect(&spirv).unwrap();
        assert_eq!(
            r.resource("_BaseMap").unwrap().kind,
            ResourceKind::Texture2D
        );
        assert_eq!(
            r.resource("sampler_BaseMap").unwrap().kind,
            ResourceKind::Sampler
        );
        assert_eq!(r.resource("_Cube").unwrap().kind, ResourceKind::TextureCube);
        let globals = r.block("$Globals").expect("loose globals become $Globals");
        assert!(globals.members.iter().any(|m| m.name == "_Loose"));
        // Every resource has its own binding in a set.
        let mut keys: Vec<(u32, u32)> = r.resources.iter().map(|x| (x.set, x.binding)).collect();
        keys.extend(r.blocks.iter().map(|b| (b.set, b.binding)));
        let count = keys.len();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), count, "no two resources share a binding");
    }

    #[test]
    fn rebind_moves_descriptors_by_name() {
        let Some(mut spirv) = compile("frag", Stage::Fragment) else {
            eprintln!("dxc not found: skipping");
            return;
        };
        let before = reflect(&spirv).unwrap();
        let changed = rebind(&mut spirv, |name| match name {
            "_BaseMap" => Some((0, 40)),
            "UnityPerMaterial" => Some((0, 41)),
            _ => None,
        })
        .unwrap();
        let after = reflect(&spirv).unwrap();
        assert!(changed >= 1);
        assert_eq!(after.resource("_BaseMap").unwrap().binding, 40);
        assert_eq!(
            after.resource("_Cube").unwrap().binding,
            before.resource("_Cube").unwrap().binding,
            "descriptors the closure ignores stay"
        );
        if let Some(block) = after.block("UnityPerMaterial") {
            assert_eq!(block.binding, 41);
        }
        assert_eq!(
            rebind(&mut [1, 2, 3], |_| None),
            Err(ReflectError::NotSpirv)
        );
    }
}
