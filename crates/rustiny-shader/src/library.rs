//! A set of parsed shaders, so a pass can name a pass of another shader (`UsePass`).

use crate::shaderlab::{Pass, RenderState, ShaderFile, SubShader};
use std::collections::HashMap;

/// All the shaders of a project, by name.
#[derive(Debug, Default, Clone)]
pub struct ShaderLibrary {
    shaders: Vec<ShaderFile>,
    by_name: HashMap<String, usize>,
}

/// A pass with its `UsePass` indirection followed and the sub-shader's render state merged in.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedPass {
    /// The pass's name (the used pass's name for a `UsePass`).
    pub name: Option<String>,
    /// The `LightMode` tag.
    pub light_mode: Option<String>,
    /// Render state: the pass's over the sub-shader's.
    pub state: RenderState,
    /// The pass that has the program (the used pass for a `UsePass`).
    pub pass: Pass,
    /// Name of the shader the pass came from.
    pub from_shader: String,
}

impl ShaderLibrary {
    /// An empty library.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a shader. A second shader with the same name (case-insensitive) replaces the first.
    pub fn add(&mut self, shader: ShaderFile) {
        let key = shader.name.to_lowercase();
        if let Some(&index) = self.by_name.get(&key) {
            self.shaders[index] = shader;
        } else {
            self.by_name.insert(key, self.shaders.len());
            self.shaders.push(shader);
        }
    }

    /// Number of shaders.
    pub fn len(&self) -> usize {
        self.shaders.len()
    }

    /// Whether there are no shaders.
    pub fn is_empty(&self) -> bool {
        self.shaders.is_empty()
    }

    /// The shaders in the order they were added.
    pub fn shaders(&self) -> &[ShaderFile] {
        &self.shaders
    }

    /// The shader called `name` (case-insensitive).
    pub fn get(&self, name: &str) -> Option<&ShaderFile> {
        self.by_name
            .get(&name.to_lowercase())
            .map(|&i| &self.shaders[i])
    }

    /// The passes of `sub`, each with `UsePass` replaced by the pass it names and render state
    /// merged. A `UsePass` that cannot be found (or that leads to itself) is left out.
    pub fn resolve_passes(&self, owner: &ShaderFile, sub: &SubShader) -> Vec<ResolvedPass> {
        let mut out = Vec::new();
        for pass in &sub.passes {
            match &pass.use_pass {
                None => out.push(resolved(owner, sub, pass, pass)),
                Some(target) => {
                    if let Some((other, used_sub, used)) = self.find_pass(target) {
                        // The used pass keeps its own sub-shader's state, as Unity compiles it
                        // there; the using sub-shader's tags and state do not apply.
                        out.push(resolved(other, used_sub, used, used));
                    }
                }
            }
        }
        out
    }

    /// `"Shader/Name/PASSNAME"`: the shader is everything before the last `/`. The pass name
    /// matches case-insensitively. The first sub-shader that has the pass is used, and a pass that
    /// is itself a `UsePass` is not followed (Unity does not either).
    fn find_pass(&self, target: &str) -> Option<(&ShaderFile, &SubShader, &Pass)> {
        let (shader_name, pass_name) = target.rsplit_once('/')?;
        let shader = self.get(shader_name)?;
        for sub in &shader.sub_shaders {
            for pass in &sub.passes {
                let named = pass
                    .name
                    .as_deref()
                    .is_some_and(|n| n.eq_ignore_ascii_case(pass_name));
                if named && pass.use_pass.is_none() {
                    return Some((shader, sub, pass));
                }
            }
        }
        None
    }
}

fn resolved(owner: &ShaderFile, sub: &SubShader, from: &Pass, pass: &Pass) -> ResolvedPass {
    ResolvedPass {
        name: from.name.clone(),
        light_mode: from.tag("LightMode").map(str::to_owned),
        state: pass.state.over(&sub.state),
        pass: pass.clone(),
        from_shader: owner.name.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shaderlab::{CullMode, Setting, parse_shader};

    fn library() -> ShaderLibrary {
        let mut lib = ShaderLibrary::new();
        lib.add(
            parse_shader(
                r#"Shader "A/Base" { SubShader { Cull Off
                Pass { Name "ForwardLit" Tags { "LightMode"="UniversalForward" } HLSLPROGRAM #pragma vertex v ENDHLSL }
                Pass { Name "ShadowCaster" HLSLPROGRAM #pragma vertex s ENDHLSL } } }"#,
            )
            .unwrap(),
        );
        lib.add(
            parse_shader(
                r#"Shader "A/User" { SubShader { Cull Front
                UsePass "A/Base/FORWARDLIT" UsePass "A/Base/Missing" UsePass "Nope/X/Y"
                Pass { Name "Own" HLSLPROGRAM #pragma vertex o ENDHLSL } } }"#,
            )
            .unwrap(),
        );
        lib
    }

    #[test]
    fn use_pass_takes_the_pass_of_the_other_shader_with_its_state() {
        let lib = library();
        let user = lib.get("a/user").unwrap();
        let passes = lib.resolve_passes(user, &user.sub_shaders[0]);
        assert_eq!(
            passes.len(),
            2,
            "two UsePass lines cannot be resolved and are dropped"
        );
        assert_eq!(passes[0].name.as_deref(), Some("ForwardLit"));
        assert_eq!(passes[0].light_mode.as_deref(), Some("UniversalForward"));
        assert_eq!(passes[0].from_shader, "A/Base");
        assert_eq!(
            passes[0].state.cull,
            Some(Setting::Fixed(CullMode::Off)),
            "base's state"
        );
        assert_eq!(
            passes[0].pass.program.as_ref().unwrap().vertex.as_deref(),
            Some("v")
        );
        assert_eq!(passes[1].name.as_deref(), Some("Own"));
        assert_eq!(passes[1].state.cull, Some(Setting::Fixed(CullMode::Front)));
    }

    #[test]
    fn adding_a_shader_twice_replaces_it() {
        let mut lib = library();
        assert_eq!(lib.len(), 2);
        lib.add(parse_shader(r#"Shader "a/base" { }"#).unwrap());
        assert_eq!(lib.len(), 2);
        assert!(lib.get("A/Base").unwrap().sub_shaders.is_empty());
        assert!(!lib.is_empty());
    }
}
