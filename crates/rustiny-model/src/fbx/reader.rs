//! Binary FBX container: a tree of named nodes, each with typed properties.
//!
//! Layout: a 27-byte header (`Kaydara FBX Binary  \0`, `0x1A 0x00`, version), then node records
//! until a null record. A record is `end offset, property count, property bytes, name length`
//! (32-bit fields before version 7500, 64-bit from 7500), the name, the properties and the nested
//! records. Every read is bounds-checked; declared array sizes are checked against the bytes that
//! are really there, so a corrupt or hostile file cannot make this allocate wildly or panic.

use super::FbxError;
use flate2::read::ZlibDecoder;
use std::io::Read;

const MAGIC: &[u8] = b"Kaydara FBX Binary  \0";
const HEADER_LEN: usize = 27;
const MAX_DEPTH: usize = 64;
/// Largest decompressed array accepted, in bytes.
const MAX_ARRAY_BYTES: u64 = 1 << 30;

/// A property value.
#[derive(Debug, Clone, PartialEq)]
pub enum Prop {
    /// `C`
    Bool(bool),
    /// `Y`
    I16(i16),
    /// `I`
    I32(i32),
    /// `L`
    I64(i64),
    /// `F`
    F32(f32),
    /// `D`
    F64(f64),
    /// `S`
    Str(String),
    /// `R`
    Raw(Vec<u8>),
    /// `i`
    I32s(Vec<i32>),
    /// `l`
    I64s(Vec<i64>),
    /// `f`
    F32s(Vec<f32>),
    /// `d`
    F64s(Vec<f64>),
    /// `b`
    Bools(Vec<bool>),
}

impl Prop {
    /// Any integer or float property as `f64`.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Bool(v) => Some(f64::from(u8::from(*v))),
            Self::I16(v) => Some(f64::from(*v)),
            Self::I32(v) => Some(f64::from(*v)),
            Self::I64(v) => Some(*v as f64),
            Self::F32(v) => Some(f64::from(*v)),
            Self::F64(v) => Some(*v),
            _ => None,
        }
    }

    /// Any integer property as `i64`.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Bool(v) => Some(i64::from(*v)),
            Self::I16(v) => Some(i64::from(*v)),
            Self::I32(v) => Some(i64::from(*v)),
            Self::I64(v) => Some(*v),
            _ => None,
        }
    }

    /// A string property.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s),
            _ => None,
        }
    }
}

/// One node of the tree.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Node {
    /// Node name (`Objects`, `Geometry`, `P`, ...).
    pub name: String,
    /// Properties in file order.
    pub props: Vec<Prop>,
    /// Nested nodes in file order.
    pub children: Vec<Node>,
}

impl Node {
    /// The first child with this name.
    pub fn child(&self, name: &str) -> Option<&Node> {
        self.children.iter().find(|c| c.name == name)
    }

    /// Every child with this name.
    pub fn children_named<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a Node> {
        self.children.iter().filter(move |c| c.name == name)
    }

    /// The array in the first property, as `f64`s (a float array of either width).
    pub fn f64_array(&self) -> Option<Vec<f64>> {
        match self.props.first()? {
            Prop::F64s(v) => Some(v.clone()),
            Prop::F32s(v) => Some(v.iter().map(|x| f64::from(*x)).collect()),
            Prop::I32s(v) => Some(v.iter().map(|x| f64::from(*x)).collect()),
            _ => None,
        }
    }

    /// The array in the first property, as `i32`s.
    pub fn i32_array(&self) -> Option<Vec<i32>> {
        match self.props.first()? {
            Prop::I32s(v) => Some(v.clone()),
            Prop::I64s(v) => Some(v.iter().filter_map(|x| i32::try_from(*x).ok()).collect()),
            _ => None,
        }
    }

    /// The string in the first property.
    pub fn first_str(&self) -> Option<&str> {
        self.props.first()?.as_str()
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], FbxError> {
        let end = self.pos.checked_add(n).ok_or(FbxError::Truncated)?;
        let slice = self.bytes.get(self.pos..end).ok_or(FbxError::Truncated)?;
        self.pos = end;
        Ok(slice)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], FbxError> {
        let mut out = [0u8; N];
        out.copy_from_slice(self.take(N)?);
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8, FbxError> {
        Ok(self.array::<1>()?[0])
    }

    fn u32(&mut self) -> Result<u32, FbxError> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64, FbxError> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    /// An offset or count: 32 bits before version 7500, 64 bits after.
    fn word(&mut self, wide: bool) -> Result<u64, FbxError> {
        if wide {
            self.u64()
        } else {
            self.u32().map(u64::from)
        }
    }
}

/// Parse a binary FBX file into its top-level nodes; returns them with the file version.
pub fn parse(bytes: &[u8]) -> Result<(u32, Vec<Node>), FbxError> {
    if bytes.len() < HEADER_LEN || !bytes.starts_with(MAGIC) {
        return Err(FbxError::NotBinaryFbx);
    }
    let mut cursor = Cursor {
        bytes,
        pos: MAGIC.len() + 2,
    };
    let version = cursor.u32()?;
    let wide = version >= 7500;
    let mut nodes = Vec::new();
    while cursor.pos < bytes.len() {
        match read_node(&mut cursor, wide, 0)? {
            Some(node) => nodes.push(node),
            None => break,
        }
    }
    Ok((version, nodes))
}

/// One record, or `None` for the null record that ends a list.
fn read_node(c: &mut Cursor<'_>, wide: bool, depth: usize) -> Result<Option<Node>, FbxError> {
    if depth > MAX_DEPTH {
        return Err(FbxError::TooDeep);
    }
    let end = c.word(wide)?;
    let count = c.word(wide)?;
    let _props_len = c.word(wide)?;
    let name_len = usize::from(c.u8()?);
    if end == 0 {
        return Ok(None);
    }
    let end = usize::try_from(end).map_err(|_| FbxError::Truncated)?;
    if end > c.bytes.len() {
        return Err(FbxError::Truncated);
    }
    let name = String::from_utf8_lossy(c.take(name_len)?).into_owned();
    // Every property takes at least one byte, which bounds the allocation.
    let remaining = end.saturating_sub(c.pos);
    let count = usize::try_from(count).map_err(|_| FbxError::Truncated)?;
    if count > remaining {
        return Err(FbxError::Truncated);
    }
    let mut props = Vec::with_capacity(count);
    for _ in 0..count {
        props.push(read_prop(c)?);
    }
    let mut children = Vec::new();
    while c.pos < end {
        match read_node(c, wide, depth + 1)? {
            Some(child) => children.push(child),
            None => break,
        }
    }
    // Records say where they end; trust that over what the children consumed.
    c.pos = end;
    Ok(Some(Node {
        name,
        props,
        children,
    }))
}

fn read_prop(c: &mut Cursor<'_>) -> Result<Prop, FbxError> {
    let kind = c.u8()?;
    Ok(match kind {
        b'C' => Prop::Bool(c.u8()? != 0),
        b'Y' => Prop::I16(i16::from_le_bytes(c.array()?)),
        b'I' => Prop::I32(i32::from_le_bytes(c.array()?)),
        b'L' => Prop::I64(i64::from_le_bytes(c.array()?)),
        b'F' => Prop::F32(f32::from_le_bytes(c.array()?)),
        b'D' => Prop::F64(f64::from_le_bytes(c.array()?)),
        b'S' | b'R' => {
            let len = c.u32()? as usize;
            let data = c.take(len)?;
            if kind == b'S' {
                Prop::Str(String::from_utf8_lossy(data).into_owned())
            } else {
                Prop::Raw(data.to_vec())
            }
        }
        b'i' => Prop::I32s(read_array(c, 4, |b| {
            i32::from_le_bytes([b[0], b[1], b[2], b[3]])
        })?),
        b'l' => Prop::I64s(read_array(c, 8, |b| {
            i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
        })?),
        b'f' => Prop::F32s(read_array(c, 4, |b| {
            f32::from_le_bytes([b[0], b[1], b[2], b[3]])
        })?),
        b'd' => Prop::F64s(read_array(c, 8, |b| {
            f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
        })?),
        b'b' => Prop::Bools(read_array(c, 1, |b| b[0] != 0)?),
        other => return Err(FbxError::BadPropertyType(other)),
    })
}

fn read_array<T>(
    c: &mut Cursor<'_>,
    width: usize,
    decode: impl Fn(&[u8]) -> T,
) -> Result<Vec<T>, FbxError> {
    let count = u64::from(c.u32()?);
    let encoding = c.u32()?;
    let stored = c.u32()? as usize;
    let data = c.take(stored)?;
    let expected = count.checked_mul(width as u64).ok_or(FbxError::Truncated)?;
    if expected > MAX_ARRAY_BYTES {
        return Err(FbxError::ArrayTooLarge);
    }
    let raw: std::borrow::Cow<'_, [u8]> = match encoding {
        0 => data.into(),
        1 => {
            let mut out = Vec::new();
            ZlibDecoder::new(data)
                .take(expected)
                .read_to_end(&mut out)
                .map_err(|e| FbxError::Decompress(e.to_string()))?;
            out.into()
        }
        other => return Err(FbxError::BadEncoding(other)),
    };
    if (raw.len() as u64) < expected {
        return Err(FbxError::Truncated);
    }
    Ok(raw[..expected as usize]
        .chunks_exact(width)
        .map(decode)
        .collect())
}
