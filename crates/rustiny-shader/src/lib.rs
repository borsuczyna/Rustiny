//! Unity shaders without a GPU: reading ShaderLab and turning its HLSL into SPIR-V.
//!
//! * [`shaderlab`] - the ShaderLab file format: properties, sub-shaders, passes, render state and
//!   the `HLSLPROGRAM`/`CGPROGRAM` blocks with their `#pragma`s ([`parse_shader`]).
//! * [`library`] - several parsed shaders, so `UsePass` can be resolved ([`ShaderLibrary`]).
//! * [`reflect`] - what a compiled shader needs: inputs, constant buffers, textures ([`reflect::reflect`]).
//! * [`compile`] - HLSL to SPIR-V with DXC ([`Compiler`]), a mirror of the project's packages
//!   so `#include "Packages/..."` resolves ([`mirror_packages`]), and a result cache.
//!
//! Compiling needs `dxc` (the DirectX Shader Compiler, shipped with the Vulkan SDK) at run time;
//! it is found through `RUSTINY_DXC`, `VULKAN_SDK` or `PATH`. Without it everything here except
//! [`Compiler`] still works. Nothing in this crate writes inside a Unity project: sources are only
//! read, and files are written to a directory the caller names.

#![cfg_attr(test, allow(clippy::unwrap_used))]

pub mod compile;
pub mod library;
pub mod reflect;
pub mod shaderlab;

pub use compile::{
    CompileError, Compiled, Compiler, MirrorStats, Request, Stage, locate_dxc, mirror_packages,
};
pub use library::{ResolvedPass, ShaderLibrary};
pub use reflect::{
    Member, ReflectError, Reflection, Resource, ResourceKind, Scalar, Shape, StageInput,
    UniformBlock, rebind, reflect,
};
pub use shaderlab::{
    Blend, BlendFactor, BlendSetting, Compare, CullMode, KeywordGroup, KeywordKind, Language,
    ParseError, Pass, Program, Property, PropertyDefault, PropertyKind, RenderState, Setting,
    ShaderFile, SubShader, parse_shader,
};
