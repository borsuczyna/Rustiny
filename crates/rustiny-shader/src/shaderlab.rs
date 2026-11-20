//! The ShaderLab file format: the outer language of Unity `.shader` files.
//!
//! A file is `Shader "Name" { Properties {...} SubShader { Tags {...} Pass {...} ... } Fallback ... }`.
//! Passes hold render state (`Cull`, `ZWrite`, `Blend`, ...) and one `HLSLPROGRAM`/`CGPROGRAM`
//! block. The program text is kept verbatim (with its `#pragma`s read out); this module does not
//! look inside it beyond that. Constructs it does not know (`Stencil`, `GrabPass`, fixed-function
//! blocks, ...) are skipped, not rejected, so a file with a rarely used feature still yields its
//! passes.

use thiserror::Error;

/// A ShaderLab file that could not be read.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ParseError {
    /// The file does not start with `Shader "name" {`.
    #[error("not a ShaderLab file: {0}")]
    NotShaderLab(String),
    /// Braces or parentheses do not match, or the file ends inside a block.
    #[error("unexpected end of input while reading {0}")]
    UnexpectedEnd(&'static str),
    /// A token that cannot appear where it was found.
    #[error("unexpected `{found}` while reading {context}")]
    Unexpected {
        /// What was read.
        found: String,
        /// What the parser was reading.
        context: &'static str,
    },
    /// A `HLSLPROGRAM`/`CGPROGRAM` block without its `ENDHLSL`/`ENDCG`.
    #[error("a shader program block is not closed")]
    UnclosedProgram,
}

/// The type of a material property.
#[derive(Debug, Clone, PartialEq)]
pub enum PropertyKind {
    /// `Color`.
    Color,
    /// `Vector`.
    Vector,
    /// `Float`.
    Float,
    /// `Int`.
    Int,
    /// `Range(min, max)`.
    Range(f32, f32),
    /// `2D`.
    Texture2D,
    /// `3D`.
    Texture3D,
    /// `Cube`.
    Cube,
    /// `2DArray`.
    Texture2DArray,
    /// `CubeArray`.
    CubeArray,
}

/// A property's default value.
#[derive(Debug, Clone, PartialEq)]
pub enum PropertyDefault {
    /// A number (`Float`, `Int`, `Range`).
    Number(f32),
    /// Four components (`Color`, `Vector`; missing ones are 0, alpha of a colour 1).
    Vector([f32; 4]),
    /// The name of a built-in texture (`white`, `black`, `bump`, `grey`, `red`, or empty).
    Texture(String),
}

/// One entry of a shader's `Properties` block.
#[derive(Debug, Clone, PartialEq)]
pub struct Property {
    /// Name as the shader code and the `.mat` files use it (`_BaseMap`).
    pub name: String,
    /// Label in the inspector.
    pub display: String,
    /// Type.
    pub kind: PropertyKind,
    /// Default value.
    pub default: PropertyDefault,
    /// Attributes in square brackets (`HDR`, `MainTexture`, `Normal`, `Toggle(_X)`, ...).
    pub attributes: Vec<String>,
}

/// A value that is written in the file or comes from a material property (`Cull [_Cull]`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Setting<T> {
    /// Written in the file.
    Fixed(T),
    /// Taken from the material property of this name.
    Property(String),
}

/// `Cull`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CullMode {
    /// Cull back faces (the default).
    Back,
    /// Cull front faces.
    Front,
    /// Draw both.
    Off,
}

/// `ZTest`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum Compare {
    Never,
    Less,
    Equal,
    LEqual,
    Greater,
    NotEqual,
    GEqual,
    Always,
}

/// A blend factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum BlendFactor {
    Zero,
    One,
    DstColor,
    SrcColor,
    OneMinusDstColor,
    SrcAlpha,
    OneMinusSrcColor,
    DstAlpha,
    OneMinusDstAlpha,
    SrcAlphaSaturate,
    OneMinusSrcAlpha,
}

/// Blend factors; the alpha pair is `None` when the file gives only the colour pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Blend {
    /// Source factor of colour.
    pub src: Setting<BlendFactor>,
    /// Destination factor of colour.
    pub dst: Setting<BlendFactor>,
    /// Source and destination factors of alpha.
    pub alpha: Option<(Setting<BlendFactor>, Setting<BlendFactor>)>,
}

/// What a pass says about blending.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum BlendSetting {
    /// No `Blend` line: opaque.
    #[default]
    Unset,
    /// `Blend Off`.
    Off,
    /// `Blend src dst [, srcA dstA]`.
    On(Blend),
}

/// Render state a pass (or a sub-shader, for all its passes) sets. `None` leaves Unity's default.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RenderState {
    /// `Cull` (default Back).
    pub cull: Option<Setting<CullMode>>,
    /// `ZWrite` (default On).
    pub z_write: Option<Setting<bool>>,
    /// `ZTest` (default LEqual).
    pub z_test: Option<Setting<Compare>>,
    /// `Blend`.
    pub blend: BlendSetting,
    /// `AlphaToMask On`.
    pub alpha_to_mask: Option<bool>,
    /// `ColorMask` text (`RGB`, `0`, ...).
    pub color_mask: Option<String>,
}

impl RenderState {
    /// This state with every setting that `base` has and this one lacks taken from `base`.
    #[must_use]
    pub fn over(&self, base: &RenderState) -> RenderState {
        RenderState {
            cull: self.cull.clone().or_else(|| base.cull.clone()),
            z_write: self.z_write.clone().or_else(|| base.z_write.clone()),
            z_test: self.z_test.clone().or_else(|| base.z_test.clone()),
            blend: if self.blend == BlendSetting::Unset {
                base.blend.clone()
            } else {
                self.blend.clone()
            },
            alpha_to_mask: self.alpha_to_mask.or(base.alpha_to_mask),
            color_mask: self.color_mask.clone().or_else(|| base.color_mask.clone()),
        }
    }
}

/// The language of a program block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    /// `HLSLPROGRAM`: SRP shaders (URP, HDRP).
    Hlsl,
    /// `CGPROGRAM`: the built-in pipeline's dialect (`UnityCG.cginc`).
    Cg,
}

/// How a keyword group is compiled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeywordKind {
    /// `multi_compile`: every option is built.
    MultiCompile,
    /// `shader_feature`: only options a material uses are built.
    ShaderFeature,
}

/// One `#pragma multi_compile`/`shader_feature` line: exactly one option is on at a time. The
/// option `_` means "none of these keywords".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeywordGroup {
    /// Whether all options are built or only used ones.
    pub kind: KeywordKind,
    /// The options, `_` included when present.
    pub options: Vec<String>,
}

/// A `HLSLPROGRAM`/`CGPROGRAM` block.
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    /// HLSL or CG.
    pub language: Language,
    /// The text (shared `HLSLINCLUDE` blocks first), without the begin/end markers.
    pub source: String,
    /// Vertex entry point (`#pragma vertex`).
    pub vertex: Option<String>,
    /// Fragment entry point (`#pragma fragment`).
    pub fragment: Option<String>,
    /// `#pragma target`, e.g. `4.5`.
    pub target: Option<String>,
    /// Keyword groups, in file order.
    pub keywords: Vec<KeywordGroup>,
}

/// One pass.
#[derive(Debug, Clone, PartialEq)]
pub struct Pass {
    /// `Name "..."`.
    pub name: Option<String>,
    /// `Tags {...}` of the pass (`LightMode`, ...).
    pub tags: Vec<(String, String)>,
    /// Render state of the pass alone (the sub-shader's is applied by [`SubShader::state`]).
    pub state: RenderState,
    /// The shader program, if the pass has one.
    pub program: Option<Program>,
    /// `UsePass "Shader/Name/PASS"`: the pass is this one of another shader.
    pub use_pass: Option<String>,
}

impl Pass {
    /// Value of the pass tag `key` (case-insensitive key).
    pub fn tag(&self, key: &str) -> Option<&str> {
        self.tags
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(key))
            .map(|(_, v)| v.as_str())
    }
}

/// A `SubShader`.
#[derive(Debug, Clone, PartialEq)]
pub struct SubShader {
    /// `Tags {...}` (`RenderType`, `Queue`, `RenderPipeline`, ...).
    pub tags: Vec<(String, String)>,
    /// `LOD n` (0 when absent).
    pub lod: u32,
    /// Render state set outside the passes; it applies to every pass.
    pub state: RenderState,
    /// The passes, `UsePass` entries included, in file order.
    pub passes: Vec<Pass>,
}

impl SubShader {
    /// Value of the sub-shader tag `key` (case-insensitive key).
    pub fn tag(&self, key: &str) -> Option<&str> {
        self.tags
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(key))
            .map(|(_, v)| v.as_str())
    }
}

/// A parsed `.shader` file.
#[derive(Debug, Clone, PartialEq)]
pub struct ShaderFile {
    /// The name in `Shader "..."` (`SummerHideout/Foliage`).
    pub name: String,
    /// The `Properties` block.
    pub properties: Vec<Property>,
    /// The sub-shaders (those inside a `Category` included).
    pub sub_shaders: Vec<SubShader>,
    /// `Fallback "name"` (`None` for `Fallback Off` or no line).
    pub fallback: Option<String>,
    /// `CustomEditor "..."`.
    pub custom_editor: Option<String>,
}

// ---------------------------------------------------------------------------------------------
// Reading

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Word(String),
    Str(String),
    P(char),
}

impl Tok {
    fn show(&self) -> String {
        match self {
            Tok::Word(w) => w.clone(),
            Tok::Str(s) => format!("\"{s}\""),
            Tok::P(c) => c.to_string(),
        }
    }
}

/// A program block cut out of the file text.
struct Block {
    language: Language,
    include: bool,
    text: String,
}

/// Replace `HLSLPROGRAM..ENDHLSL` (and the other three marker pairs) by `@BLOCK:n@` words and
/// return the blocks, so braces inside shader code do not confuse the structure parser.
fn cut_blocks(text: &str) -> Result<(String, Vec<Block>), ParseError> {
    const MARKERS: [(&str, &str, Language, bool); 4] = [
        ("HLSLPROGRAM", "ENDHLSL", Language::Hlsl, false),
        ("CGPROGRAM", "ENDCG", Language::Cg, false),
        ("HLSLINCLUDE", "ENDHLSL", Language::Hlsl, true),
        ("CGINCLUDE", "ENDCG", Language::Cg, true),
    ];
    let mut skeleton = String::with_capacity(text.len());
    let mut blocks = Vec::new();
    let mut rest = text;
    loop {
        // The earliest marker that starts at a word boundary and is not inside a `//` comment.
        let mut best: Option<(usize, usize)> = None;
        for (index, (begin, ..)) in MARKERS.iter().enumerate() {
            let mut from = 0;
            while let Some(found) = rest[from..].find(begin) {
                let at = from + found;
                let before = rest[..at].chars().next_back();
                let after = rest[at + begin.len()..].chars().next();
                let boundary = !before.is_some_and(|c| c.is_alphanumeric() || c == '_')
                    && !after.is_some_and(|c| c.is_alphanumeric() || c == '_');
                let line_start = rest[..at].rfind('\n').map_or(0, |n| n + 1);
                let commented = rest[line_start..at].contains("//");
                if boundary && !commented {
                    if best.is_none_or(|(b, _)| at < b) {
                        best = Some((at, index));
                    }
                    break;
                }
                from = at + begin.len();
            }
        }
        let Some((at, index)) = best else {
            skeleton.push_str(rest);
            break;
        };
        let (begin, end, language, include) = MARKERS[index];
        skeleton.push_str(&rest[..at]);
        let body_start = at + begin.len();
        let Some(body_len) = rest[body_start..].find(end) else {
            return Err(ParseError::UnclosedProgram);
        };
        blocks.push(Block {
            language,
            include,
            text: rest[body_start..body_start + body_len].to_owned(),
        });
        skeleton.push_str(&format!(" @BLOCK:{}@ ", blocks.len() - 1));
        rest = &rest[body_start + body_len + end.len()..];
    }
    Ok((skeleton, blocks))
}

fn tokenize(text: &str) -> Vec<Tok> {
    let mut toks = Vec::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            c if c.is_whitespace() || c == '\u{feff}' => {}
            '/' if chars.peek() == Some(&'/') => {
                for next in chars.by_ref() {
                    if next == '\n' {
                        break;
                    }
                }
            }
            '/' if chars.peek() == Some(&'*') => {
                chars.next();
                let mut prev = ' ';
                for next in chars.by_ref() {
                    if prev == '*' && next == '/' {
                        break;
                    }
                    prev = next;
                }
            }
            '"' => {
                let mut s = String::new();
                for next in chars.by_ref() {
                    if next == '"' {
                        break;
                    }
                    s.push(next);
                }
                toks.push(Tok::Str(s));
            }
            '{' | '}' | '(' | ')' | '[' | ']' | '=' | ',' => toks.push(Tok::P(c)),
            _ => {
                let mut w = String::from(c);
                while let Some(&next) = chars.peek() {
                    if next.is_whitespace() || "{}()[]=,\"".contains(next) {
                        break;
                    }
                    if next == '/' {
                        break;
                    }
                    w.push(next);
                    chars.next();
                }
                toks.push(Tok::Word(w));
            }
        }
    }
    toks
}

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
    blocks: Vec<Block>,
}

type Res<T> = Result<T, ParseError>;

fn parse_number(s: &str) -> Option<f32> {
    s.trim_start_matches('+').parse().ok()
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn next(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        self.pos += usize::from(t.is_some());
        t
    }

    fn peek_word(&self) -> Option<&str> {
        match self.peek() {
            Some(Tok::Word(w)) => Some(w),
            _ => None,
        }
    }

    fn expect_p(&mut self, c: char, context: &'static str) -> Res<()> {
        match self.next() {
            Some(Tok::P(p)) if p == c => Ok(()),
            Some(other) => Err(ParseError::Unexpected {
                found: other.show(),
                context,
            }),
            None => Err(ParseError::UnexpectedEnd(context)),
        }
    }

    fn eat_p(&mut self, c: char) -> bool {
        if self.peek() == Some(&Tok::P(c)) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect_str(&mut self, context: &'static str) -> Res<String> {
        match self.next() {
            Some(Tok::Str(s)) => Ok(s),
            Some(other) => Err(ParseError::Unexpected {
                found: other.show(),
                context,
            }),
            None => Err(ParseError::UnexpectedEnd(context)),
        }
    }

    /// Skip a `{ ... }` block whose `{` is next (nested blocks included).
    fn skip_block(&mut self, context: &'static str) -> Res<()> {
        self.expect_p('{', context)?;
        let mut depth = 1;
        while depth > 0 {
            match self.next() {
                Some(Tok::P('{')) => depth += 1,
                Some(Tok::P('}')) => depth -= 1,
                Some(_) => {}
                None => return Err(ParseError::UnexpectedEnd(context)),
            }
        }
        Ok(())
    }

    /// Skip tokens up to (not including) the next word that is in `stops`, `}` or `{`.
    fn skip_until_keyword(&mut self, stops: &[&str]) {
        while let Some(tok) = self.peek() {
            match tok {
                Tok::P('}') | Tok::P('{') => return,
                Tok::Word(w) if w.starts_with("@BLOCK:") => return,
                Tok::Word(w) if stops.iter().any(|s| s.eq_ignore_ascii_case(w)) => return,
                _ => self.pos += 1,
            }
        }
    }

    /// Skip an unknown command: its word, its arguments up to the next known keyword, and a
    /// `{ ... }` block if it has one.
    fn skip_command(&mut self, stops: &[&str], context: &'static str) -> Res<()> {
        self.pos += 1;
        self.skip_until_keyword(stops);
        if self.peek() == Some(&Tok::P('{')) {
            self.skip_block(context)?;
        }
        Ok(())
    }

    fn parse_tags(&mut self) -> Res<Vec<(String, String)>> {
        self.expect_p('{', "Tags")?;
        let mut tags = Vec::new();
        loop {
            match self.next() {
                Some(Tok::P('}')) => return Ok(tags),
                Some(Tok::Str(key)) => {
                    self.expect_p('=', "Tags")?;
                    tags.push((key, self.expect_str("Tags")?));
                }
                Some(other) => {
                    return Err(ParseError::Unexpected {
                        found: other.show(),
                        context: "Tags",
                    });
                }
                None => return Err(ParseError::UnexpectedEnd("Tags")),
            }
        }
    }

    fn parse_properties(&mut self) -> Res<Vec<Property>> {
        self.expect_p('{', "Properties")?;
        let mut properties = Vec::new();
        loop {
            let mut attributes = Vec::new();
            while self.eat_p('[') {
                let mut text = String::new();
                let mut previous_word = false;
                let mut depth = 1;
                while depth > 0 {
                    match self.next() {
                        Some(Tok::P('[')) => depth += 1,
                        Some(Tok::P(']')) => depth -= 1,
                        Some(t) => {
                            let word = matches!(t, Tok::Word(_));
                            if word && previous_word {
                                text.push(' ');
                            }
                            previous_word = word;
                            text.push_str(&t.show());
                        }
                        None => return Err(ParseError::UnexpectedEnd("a property attribute")),
                    }
                }
                attributes.push(text);
            }
            let name = match self.next() {
                Some(Tok::P('}')) => return Ok(properties),
                Some(Tok::Word(w)) => w,
                Some(other) => {
                    return Err(ParseError::Unexpected {
                        found: other.show(),
                        context: "Properties",
                    });
                }
                None => return Err(ParseError::UnexpectedEnd("Properties")),
            };
            self.expect_p('(', "a property")?;
            let display = self.expect_str("a property")?;
            self.expect_p(',', "a property")?;
            let mut type_toks = Vec::new();
            let mut depth = 1;
            loop {
                match self.next() {
                    Some(Tok::P('(')) => {
                        depth += 1;
                        type_toks.push(Tok::P('('));
                    }
                    Some(Tok::P(')')) => {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                        type_toks.push(Tok::P(')'));
                    }
                    Some(t) => type_toks.push(t),
                    None => return Err(ParseError::UnexpectedEnd("a property type")),
                }
            }
            let kind = property_kind(&type_toks).ok_or_else(|| ParseError::Unexpected {
                found: type_toks.first().map_or_else(String::new, Tok::show),
                context: "a property type",
            })?;
            self.expect_p('=', "a property")?;
            let default = self.parse_default(&kind)?;
            properties.push(Property {
                name,
                display,
                kind,
                default,
                attributes,
            });
        }
    }

    fn parse_default(&mut self, kind: &PropertyKind) -> Res<PropertyDefault> {
        match self.next() {
            Some(Tok::P('(')) => {
                let mut values = Vec::new();
                loop {
                    match self.next() {
                        Some(Tok::P(')')) => break,
                        Some(Tok::P(',')) => {}
                        Some(Tok::Word(w)) => values.push(parse_number(&w).unwrap_or(0.0)),
                        Some(other) => {
                            return Err(ParseError::Unexpected {
                                found: other.show(),
                                context: "a property default",
                            });
                        }
                        None => return Err(ParseError::UnexpectedEnd("a property default")),
                    }
                }
                let alpha = if *kind == PropertyKind::Color {
                    1.0
                } else {
                    0.0
                };
                let mut v = [0.0, 0.0, 0.0, alpha];
                for (slot, value) in v.iter_mut().zip(values) {
                    *slot = value;
                }
                Ok(PropertyDefault::Vector(v))
            }
            Some(Tok::Str(name)) => {
                if self.peek() == Some(&Tok::P('{')) {
                    self.skip_block("a texture default")?;
                }
                Ok(PropertyDefault::Texture(name))
            }
            Some(Tok::Word(w)) => Ok(PropertyDefault::Number(parse_number(&w).unwrap_or(0.0))),
            Some(other) => Err(ParseError::Unexpected {
                found: other.show(),
                context: "a property default",
            }),
            None => Err(ParseError::UnexpectedEnd("a property default")),
        }
    }

    /// `Name` or `[_Property]`, converted by `convert`.
    fn parse_setting<T>(&mut self, convert: fn(&str) -> Option<T>) -> Option<Setting<T>> {
        let save = self.pos;
        if self.eat_p('[') {
            if let (Some(Tok::Word(name)), Some(Tok::P(']'))) = (self.next(), self.next()) {
                return Some(Setting::Property(name));
            }
            self.pos = save;
            return None;
        }
        if let Some(Tok::Word(w)) = self.peek().cloned()
            && let Some(value) = convert(&w)
        {
            self.pos += 1;
            return Some(Setting::Fixed(value));
        }
        None
    }

    /// Try to read one render-state command whose keyword is the next word.
    fn parse_state(&mut self, state: &mut RenderState, keyword: &str) -> bool {
        let lower = keyword.to_ascii_lowercase();
        let save = self.pos;
        self.pos += 1;
        let done = match lower.as_str() {
            "cull" => self.parse_setting(cull_mode).map(|v| state.cull = Some(v)),
            "zwrite" => self.parse_setting(on_off).map(|v| state.z_write = Some(v)),
            "ztest" => self.parse_setting(compare).map(|v| state.z_test = Some(v)),
            "alphatomask" => match self.parse_setting(on_off) {
                Some(Setting::Fixed(v)) => {
                    state.alpha_to_mask = Some(v);
                    Some(())
                }
                _ => None,
            },
            "colormask" => match self.next() {
                Some(Tok::Word(w)) => {
                    state.color_mask = Some(w);
                    // An optional render target index follows.
                    if matches!(self.peek(), Some(Tok::Word(i)) if i.parse::<u32>().is_ok()) {
                        self.pos += 1;
                    }
                    Some(())
                }
                _ => None,
            },
            "blend" => self.parse_blend().map(|b| state.blend = b),
            _ => None,
        };
        if done.is_none() {
            self.pos = save;
        }
        done.is_some()
    }

    fn parse_blend(&mut self) -> Option<BlendSetting> {
        // An optional render target index (`Blend 1 One One`).
        if matches!(self.peek(), Some(Tok::Word(w)) if w.parse::<u32>().is_ok()) {
            self.pos += 1;
        }
        if self
            .peek_word()
            .is_some_and(|w| w.eq_ignore_ascii_case("off"))
        {
            self.pos += 1;
            return Some(BlendSetting::Off);
        }
        let src = self.parse_setting(blend_factor)?;
        let dst = self.parse_setting(blend_factor)?;
        let mut alpha = None;
        if self.eat_p(',') {
            let a = self.parse_setting(blend_factor)?;
            let b = self.parse_setting(blend_factor)?;
            alpha = Some((a, b));
        }
        Some(BlendSetting::On(Blend { src, dst, alpha }))
    }

    fn parse_pass(&mut self, includes: &[String]) -> Res<Pass> {
        self.expect_p('{', "Pass")?;
        let mut pass = Pass {
            name: None,
            tags: Vec::new(),
            state: RenderState::default(),
            program: None,
            use_pass: None,
        };
        loop {
            let Some(tok) = self.peek().cloned() else {
                return Err(ParseError::UnexpectedEnd("Pass"));
            };
            match tok {
                Tok::P('}') => {
                    self.pos += 1;
                    return Ok(pass);
                }
                Tok::Word(w) if w.starts_with("@BLOCK:") => {
                    self.pos += 1;
                    if let Some(block) = block_index(&w).and_then(|i| self.blocks.get(i)) {
                        let program = make_program(block, includes);
                        pass.program = Some(program);
                    }
                }
                Tok::Word(w) if w.eq_ignore_ascii_case("name") => {
                    self.pos += 1;
                    pass.name = Some(self.expect_str("Pass Name")?);
                }
                Tok::Word(w) if w.eq_ignore_ascii_case("tags") => {
                    self.pos += 1;
                    pass.tags = self.parse_tags()?;
                }
                Tok::Word(w) => {
                    if !self.parse_state(&mut pass.state, &w) {
                        // Unknown command (Stencil, GrabPass, fixed-function blocks, ...).
                        self.skip_command(STATE_KEYWORDS, "a pass command")?;
                    }
                }
                other => {
                    return Err(ParseError::Unexpected {
                        found: other.show(),
                        context: "Pass",
                    });
                }
            }
        }
    }

    fn parse_sub_shader(&mut self, shader_includes: &[String]) -> Res<SubShader> {
        self.expect_p('{', "SubShader")?;
        let mut sub = SubShader {
            tags: Vec::new(),
            lod: 0,
            state: RenderState::default(),
            passes: Vec::new(),
        };
        let mut includes: Vec<String> = shader_includes.to_vec();
        loop {
            let Some(tok) = self.peek().cloned() else {
                return Err(ParseError::UnexpectedEnd("SubShader"));
            };
            match tok {
                Tok::P('}') => {
                    self.pos += 1;
                    return Ok(sub);
                }
                Tok::Word(w) if w.starts_with("@BLOCK:") => {
                    self.pos += 1;
                    if let Some(block) = block_index(&w).and_then(|i| self.blocks.get(i))
                        && block.include
                    {
                        includes.push(block.text.clone());
                    }
                }
                Tok::Word(w) if w.eq_ignore_ascii_case("tags") => {
                    self.pos += 1;
                    sub.tags = self.parse_tags()?;
                }
                Tok::Word(w) if w.eq_ignore_ascii_case("lod") => {
                    self.pos += 1;
                    if let Some(Tok::Word(n)) = self.next() {
                        sub.lod = n.parse().unwrap_or(0);
                    }
                }
                Tok::Word(w) if w.eq_ignore_ascii_case("pass") => {
                    self.pos += 1;
                    sub.passes.push(self.parse_pass(&includes)?);
                }
                Tok::Word(w) if w.eq_ignore_ascii_case("usepass") => {
                    self.pos += 1;
                    let target = self.expect_str("UsePass")?;
                    sub.passes.push(Pass {
                        name: None,
                        tags: Vec::new(),
                        state: RenderState::default(),
                        program: None,
                        use_pass: Some(target),
                    });
                }
                Tok::Word(w) => {
                    if !self.parse_state(&mut sub.state, &w) {
                        self.skip_command(STATE_KEYWORDS, "a sub-shader command")?;
                    }
                }
                other => {
                    return Err(ParseError::Unexpected {
                        found: other.show(),
                        context: "SubShader",
                    });
                }
            }
        }
    }

    fn parse_shader(&mut self) -> Res<ShaderFile> {
        match self.next() {
            Some(Tok::Word(w)) if w == "Shader" => {}
            other => {
                return Err(ParseError::NotShaderLab(
                    other.map_or_else(|| "empty file".to_owned(), |t| t.show()),
                ));
            }
        }
        let name = self.expect_str("Shader")?;
        self.expect_p('{', "Shader")?;
        let mut shader = ShaderFile {
            name,
            properties: Vec::new(),
            sub_shaders: Vec::new(),
            fallback: None,
            custom_editor: None,
        };
        let mut includes: Vec<String> = Vec::new();
        self.parse_shader_items(&mut shader, &mut includes, false)?;
        Ok(shader)
    }

    /// The items of a `Shader` (or `Category`) body, up to and including its `}`.
    fn parse_shader_items(
        &mut self,
        shader: &mut ShaderFile,
        includes: &mut Vec<String>,
        in_category: bool,
    ) -> Res<()> {
        loop {
            let Some(tok) = self.peek().cloned() else {
                return Err(ParseError::UnexpectedEnd("Shader"));
            };
            match tok {
                Tok::P('}') => {
                    self.pos += 1;
                    return Ok(());
                }
                Tok::Word(w) if w.starts_with("@BLOCK:") => {
                    self.pos += 1;
                    if let Some(block) = block_index(&w).and_then(|i| self.blocks.get(i))
                        && block.include
                    {
                        includes.push(block.text.clone());
                    }
                }
                Tok::Word(w) if w.eq_ignore_ascii_case("properties") => {
                    self.pos += 1;
                    shader.properties = self.parse_properties()?;
                }
                Tok::Word(w) if w.eq_ignore_ascii_case("subshader") => {
                    self.pos += 1;
                    let sub = self.parse_sub_shader(includes)?;
                    shader.sub_shaders.push(sub);
                }
                Tok::Word(w) if w.eq_ignore_ascii_case("category") && !in_category => {
                    self.pos += 1;
                    self.expect_p('{', "Category")?;
                    let mut inner = includes.clone();
                    self.parse_shader_items(shader, &mut inner, true)?;
                }
                Tok::Word(w) if w.eq_ignore_ascii_case("fallback") => {
                    self.pos += 1;
                    match self.next() {
                        Some(Tok::Str(s)) => shader.fallback = Some(s),
                        Some(Tok::Word(_)) => {}
                        _ => return Err(ParseError::UnexpectedEnd("Fallback")),
                    }
                }
                Tok::Word(w) if w.eq_ignore_ascii_case("customeditor") => {
                    self.pos += 1;
                    shader.custom_editor = Some(self.expect_str("CustomEditor")?);
                }
                Tok::Word(_) => {
                    self.skip_command(&["subshader", "properties", "fallback"], "a shader item")?;
                }
                other => {
                    return Err(ParseError::Unexpected {
                        found: other.show(),
                        context: "Shader",
                    });
                }
            }
        }
    }
}

const STATE_KEYWORDS: &[&str] = &[
    "cull",
    "zwrite",
    "ztest",
    "blend",
    "colormask",
    "alphatomask",
    "offset",
    "stencil",
    "name",
    "tags",
    "lod",
    "pass",
    "usepass",
    "grabpass",
    "lighting",
    "fog",
    "bindchannels",
];

fn block_index(word: &str) -> Option<usize> {
    word.strip_prefix("@BLOCK:")?
        .strip_suffix('@')?
        .parse()
        .ok()
}

fn property_kind(toks: &[Tok]) -> Option<PropertyKind> {
    let Some(Tok::Word(first)) = toks.first() else {
        return None;
    };
    Some(match first.to_ascii_lowercase().as_str() {
        "color" => PropertyKind::Color,
        "vector" => PropertyKind::Vector,
        "float" => PropertyKind::Float,
        "int" => PropertyKind::Int,
        "2d" => PropertyKind::Texture2D,
        "3d" => PropertyKind::Texture3D,
        "cube" => PropertyKind::Cube,
        "2darray" => PropertyKind::Texture2DArray,
        "cubearray" => PropertyKind::CubeArray,
        w if w.starts_with("range") => {
            let numbers: Vec<f32> = toks
                .iter()
                .skip(1)
                .filter_map(|t| match t {
                    Tok::Word(w) => parse_number(w),
                    _ => None,
                })
                .collect();
            // `Range(0,1)` may arrive as the word `Range` + `(` + numbers, or `range(0` glued.
            let glued = w.trim_start_matches("range").trim_start_matches('(');
            let mut all: Vec<f32> = parse_number(glued).into_iter().collect();
            all.extend(numbers);
            PropertyKind::Range(*all.first()?, *all.get(1)?)
        }
        _ => return None,
    })
}

fn on_off(w: &str) -> Option<bool> {
    match w.to_ascii_lowercase().as_str() {
        "on" => Some(true),
        "off" => Some(false),
        _ => None,
    }
}

fn cull_mode(w: &str) -> Option<CullMode> {
    match w.to_ascii_lowercase().as_str() {
        "back" => Some(CullMode::Back),
        "front" => Some(CullMode::Front),
        "off" => Some(CullMode::Off),
        _ => None,
    }
}

fn compare(w: &str) -> Option<Compare> {
    Some(match w.to_ascii_lowercase().as_str() {
        "never" => Compare::Never,
        "less" => Compare::Less,
        "equal" => Compare::Equal,
        "lequal" => Compare::LEqual,
        "greater" => Compare::Greater,
        "notequal" => Compare::NotEqual,
        "gequal" => Compare::GEqual,
        "always" => Compare::Always,
        _ => return None,
    })
}

fn blend_factor(w: &str) -> Option<BlendFactor> {
    Some(match w.to_ascii_lowercase().as_str() {
        "zero" => BlendFactor::Zero,
        "one" => BlendFactor::One,
        "dstcolor" => BlendFactor::DstColor,
        "srccolor" => BlendFactor::SrcColor,
        "oneminusdstcolor" => BlendFactor::OneMinusDstColor,
        "srcalpha" => BlendFactor::SrcAlpha,
        "oneminussrccolor" => BlendFactor::OneMinusSrcColor,
        "dstalpha" => BlendFactor::DstAlpha,
        "oneminusdstalpha" => BlendFactor::OneMinusDstAlpha,
        "srcalphasaturate" => BlendFactor::SrcAlphaSaturate,
        "oneminussrcalpha" => BlendFactor::OneMinusSrcAlpha,
        _ => return None,
    })
}

/// Build a pass's program: shared includes first, the pass text after; `#pragma`s read from both.
fn make_program(block: &Block, includes: &[String]) -> Program {
    let mut source = String::new();
    for include in includes {
        source.push_str(include);
        source.push('\n');
    }
    source.push_str(&block.text);
    let mut program = Program {
        language: block.language,
        source,
        vertex: None,
        fragment: None,
        target: None,
        keywords: Vec::new(),
    };
    for line in program.source.clone().lines() {
        read_pragma(&mut program, line.trim());
    }
    program
}

fn read_pragma(program: &mut Program, line: &str) {
    let Some(rest) = line.strip_prefix('#') else {
        return;
    };
    let mut words = rest.split_whitespace();
    if words.next() != Some("pragma") {
        return;
    }
    let Some(directive) = words.next() else {
        return;
    };
    let args: Vec<String> = words
        .take_while(|w| !w.starts_with("//"))
        .map(str::to_owned)
        .collect();
    match directive {
        "vertex" => program.vertex = args.first().cloned(),
        "fragment" => program.fragment = args.first().cloned(),
        "target" => program.target = args.first().cloned(),
        "multi_compile_fog" => program.keywords.push(KeywordGroup {
            kind: KeywordKind::MultiCompile,
            options: ["_", "FOG_LINEAR", "FOG_EXP", "FOG_EXP2"]
                .map(str::to_owned)
                .to_vec(),
        }),
        "multi_compile_instancing" => program.keywords.push(KeywordGroup {
            kind: KeywordKind::MultiCompile,
            options: ["_", "INSTANCING_ON"].map(str::to_owned).to_vec(),
        }),
        d if (d == "multi_compile" || d.starts_with("multi_compile_")) && !args.is_empty() => {
            program.keywords.push(KeywordGroup {
                kind: KeywordKind::MultiCompile,
                options: args,
            });
        }
        d if (d == "shader_feature" || d.starts_with("shader_feature_")) && !args.is_empty() => {
            program.keywords.push(KeywordGroup {
                kind: KeywordKind::ShaderFeature,
                options: args,
            });
        }
        _ => {}
    }
}

/// Read a `.shader` file.
///
/// # Errors
///
/// [`ParseError`] when the text is not ShaderLab or its blocks are not closed.
pub fn parse_shader(text: &str) -> Result<ShaderFile, ParseError> {
    let (skeleton, blocks) = cut_blocks(text)?;
    let mut parser = Parser {
        toks: tokenize(&skeleton),
        pos: 0,
        blocks,
    };
    parser.parse_shader()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
// leading comment
Shader "Test/Sample"
{
    Properties
    {
        [MainTexture] _BaseMap ("Albedo (RGBA)", 2D) = "white" {}
        [MainColor][HDR] _BaseColor ("Tint", Color) = (1,0.5,0.25,1)
        _Cutoff ("Alpha // Cutoff", Range(0.0, 1)) = 0.5
        [Toggle(_FOO)] _Foo ("Foo", Float) = 1
        _V ("V", Vector) = (1,2,3,4)
        [Enum(UnityEngine.Rendering.CullMode)] _Cull ("Cull", Float) = 2
    }
    HLSLINCLUDE
    #define SHARED 1
    ENDHLSL
    SubShader
    {
        Tags { "RenderType" = "Opaque" "RenderPipeline" = "UniversalPipeline" "Queue"="Geometry" }
        LOD 300
        Cull Off
        Pass
        {
            Name "ForwardLit"
            Tags { "LightMode" = "UniversalForward" }
            Blend SrcAlpha OneMinusSrcAlpha, One Zero
            ZWrite [_ZWrite]
            ZTest LEqual
            Cull [_Cull]
            Stencil { Ref 1 Comp Always }
            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag
            #pragma target 4.5
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE
            #pragma multi_compile_fog
            #pragma shader_feature_local _NORMALMAP
            struct A { float4 p : POSITION; };
            float4 Vert(A a) : SV_POSITION { return a.p; }  // braces { } in code
            float4 Frag() : SV_Target { return 1; }
            ENDHLSL
        }
        UsePass "Test/Other/SHADOWCASTER"
        Pass
        {
            Name "Depth"
            ColorMask 0
            ZWrite On
            CGPROGRAM
            #pragma vertex vert
            ENDCG
        }
    }
    Fallback "Hidden/Universal Render Pipeline/FallbackError"
    CustomEditor "My.Editor"
}
"#;

    #[test]
    fn name_properties_and_attributes_are_read() {
        let shader = parse_shader(SAMPLE).unwrap();
        assert_eq!(shader.name, "Test/Sample");
        assert_eq!(shader.properties.len(), 6);
        let base = &shader.properties[0];
        assert_eq!(base.name, "_BaseMap");
        assert_eq!(base.kind, PropertyKind::Texture2D);
        assert_eq!(base.default, PropertyDefault::Texture("white".into()));
        assert_eq!(base.attributes, vec!["MainTexture"]);
        let color = &shader.properties[1];
        assert_eq!(color.kind, PropertyKind::Color);
        assert_eq!(
            color.default,
            PropertyDefault::Vector([1.0, 0.5, 0.25, 1.0])
        );
        assert_eq!(color.attributes, vec!["MainColor", "HDR"]);
        let cutoff = &shader.properties[2];
        assert_eq!(
            cutoff.display, "Alpha // Cutoff",
            "// inside a string is not a comment"
        );
        assert_eq!(cutoff.kind, PropertyKind::Range(0.0, 1.0));
        assert_eq!(cutoff.default, PropertyDefault::Number(0.5));
        assert_eq!(shader.properties[3].attributes, vec!["Toggle(_FOO)"]);
        assert_eq!(
            shader.properties[4].default,
            PropertyDefault::Vector([1.0, 2.0, 3.0, 4.0])
        );
        assert_eq!(shader.properties[5].kind, PropertyKind::Float);
        assert_eq!(
            shader.fallback.as_deref(),
            Some("Hidden/Universal Render Pipeline/FallbackError")
        );
        assert_eq!(shader.custom_editor.as_deref(), Some("My.Editor"));
    }

    #[test]
    fn sub_shaders_passes_state_and_tags_are_read() {
        let shader = parse_shader(SAMPLE).unwrap();
        let sub = &shader.sub_shaders[0];
        assert_eq!(sub.tag("renderpipeline"), Some("UniversalPipeline"));
        assert_eq!(sub.tag("Queue"), Some("Geometry"));
        assert_eq!(sub.lod, 300);
        assert_eq!(sub.state.cull, Some(Setting::Fixed(CullMode::Off)));
        assert_eq!(sub.passes.len(), 3);

        let forward = &sub.passes[0];
        assert_eq!(forward.name.as_deref(), Some("ForwardLit"));
        assert_eq!(forward.tag("LightMode"), Some("UniversalForward"));
        assert_eq!(
            forward.state.z_write,
            Some(Setting::Property("_ZWrite".into()))
        );
        assert_eq!(forward.state.z_test, Some(Setting::Fixed(Compare::LEqual)));
        assert_eq!(forward.state.cull, Some(Setting::Property("_Cull".into())));
        let BlendSetting::On(blend) = &forward.state.blend else {
            panic!("blend not read");
        };
        assert_eq!(blend.src, Setting::Fixed(BlendFactor::SrcAlpha));
        assert_eq!(blend.dst, Setting::Fixed(BlendFactor::OneMinusSrcAlpha));
        assert_eq!(
            blend.alpha,
            Some((
                Setting::Fixed(BlendFactor::One),
                Setting::Fixed(BlendFactor::Zero)
            ))
        );

        assert_eq!(
            sub.passes[1].use_pass.as_deref(),
            Some("Test/Other/SHADOWCASTER")
        );
        let depth = &sub.passes[2];
        assert_eq!(depth.state.color_mask.as_deref(), Some("0"));
        assert_eq!(depth.program.as_ref().unwrap().language, Language::Cg);
    }

    #[test]
    fn an_unknown_state_command_does_not_swallow_the_program() {
        // `AlphaToMask[_Prop]` (a property, not a fixed value) is not a state we model; skipping
        // it must stop at the program that follows, or the pass silently loses its shader.
        let text = r#"Shader "T" { SubShader { Pass {
            Tags { "LightMode" = "UniversalForward" }
            Blend[_Src][_Dst], [_SrcA][_DstA]
            AlphaToMask[_AlphaToMask]
            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag
            ENDHLSL
        } } }"#;
        let shader = parse_shader(text).unwrap();
        let program = shader.sub_shaders[0].passes[0].program.as_ref().unwrap();
        assert_eq!(program.vertex.as_deref(), Some("Vert"));
        assert_eq!(program.fragment.as_deref(), Some("Frag"));
    }

    #[test]
    fn programs_keep_their_text_and_pragmas() {
        let shader = parse_shader(SAMPLE).unwrap();
        let program = shader.sub_shaders[0].passes[0].program.as_ref().unwrap();
        assert_eq!(program.language, Language::Hlsl);
        assert_eq!(program.vertex.as_deref(), Some("Vert"));
        assert_eq!(program.fragment.as_deref(), Some("Frag"));
        assert_eq!(program.target.as_deref(), Some("4.5"));
        assert!(
            program.source.contains("#define SHARED 1"),
            "HLSLINCLUDE comes first"
        );
        assert!(program.source.contains("braces { } in code"));
        assert_eq!(program.keywords.len(), 3);
        assert_eq!(program.keywords[0].kind, KeywordKind::MultiCompile);
        assert_eq!(
            program.keywords[0].options,
            vec!["_", "_MAIN_LIGHT_SHADOWS", "_MAIN_LIGHT_SHADOWS_CASCADE"]
        );
        assert_eq!(program.keywords[1].options[1], "FOG_LINEAR");
        assert_eq!(program.keywords[2].kind, KeywordKind::ShaderFeature);
        assert_eq!(program.keywords[2].options, vec!["_NORMALMAP"]);
    }

    #[test]
    fn state_commands_over_a_base_fill_gaps_only() {
        let shader = parse_shader(SAMPLE).unwrap();
        let sub = &shader.sub_shaders[0];
        let depth = sub.passes[2].state.over(&sub.state);
        assert_eq!(
            depth.cull,
            Some(Setting::Fixed(CullMode::Off)),
            "from the sub-shader"
        );
        assert_eq!(depth.z_write, Some(Setting::Fixed(true)), "the pass's own");
        let forward = sub.passes[0].state.over(&sub.state);
        assert_eq!(
            forward.cull,
            Some(Setting::Property("_Cull".into())),
            "the pass wins"
        );
    }

    #[test]
    fn category_sub_shaders_are_found_and_fixed_function_is_skipped() {
        let text = r#"Shader "Old/Style" {
            Category {
                Tags { "Queue"="Transparent" }
                SubShader {
                    Pass {
                        Lighting Off
                        Material { Diffuse (1,1,1,1) }
                        SetTexture [_MainTex] { combine texture }
                    }
                }
            }
            Fallback Off
        }"#;
        let shader = parse_shader(text).unwrap();
        assert_eq!(shader.sub_shaders.len(), 1);
        assert_eq!(shader.sub_shaders[0].passes.len(), 1);
        assert_eq!(shader.fallback, None);
    }

    #[test]
    fn broken_files_are_errors_not_panics() {
        assert!(matches!(parse_shader(""), Err(ParseError::NotShaderLab(_))));
        assert!(matches!(
            parse_shader("Not a shader"),
            Err(ParseError::NotShaderLab(_))
        ));
        assert!(parse_shader("Shader \"X\" { SubShader {").is_err());
        assert_eq!(
            parse_shader("Shader \"X\" { SubShader { Pass { HLSLPROGRAM void f(){} } } }"),
            Err(ParseError::UnclosedProgram)
        );
        assert!(parse_shader("Shader \"X\" { Properties { _A (\"a\", Bogus) = 1 } }").is_err());
    }

    #[test]
    fn a_commented_out_program_marker_is_ignored() {
        let text = "Shader \"X\" { SubShader { Pass {\n // HLSLPROGRAM old\n HLSLPROGRAM\n #pragma vertex v\n ENDHLSL } } }";
        let shader = parse_shader(text).unwrap();
        let program = shader.sub_shaders[0].passes[0].program.as_ref().unwrap();
        assert_eq!(program.vertex.as_deref(), Some("v"));
    }
}
