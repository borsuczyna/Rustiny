//! Compiles the WGSL shaders in `shaders/` to SPIR-V with `naga` (no Vulkan SDK needed), writing
//! `$OUT_DIR/<name>.spv`. A shader that does not parse or validate fails the build with naga's
//! own message.

use naga::back::spv;
use naga::valid::{Capabilities, ValidationFlags, Validator};
use std::error::Error;
use std::path::PathBuf;

const SHADERS: &[&str] = &["scene", "present", "bloom"];

fn main() -> Result<(), Box<dyn Error>> {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
    for name in SHADERS {
        let path = PathBuf::from("shaders").join(format!("{name}.wgsl"));
        println!("cargo:rerun-if-changed={}", path.display());
        let source = std::fs::read_to_string(&path)?;

        let module = naga::front::wgsl::parse_str(&source)
            .map_err(|e| format!("{}: {}", path.display(), e.emit_to_string(&source)))?;
        let info = Validator::new(ValidationFlags::all(), Capabilities::IMMEDIATES)
            .validate(&module)
            .map_err(|e| format!("{}: {e:?}", path.display()))?;
        // The shaders already emit Vulkan clip space (Y down, reverse-Z), so naga's default
        // coordinate-space adjustment (which flips Y for WebGPU sources) must stay off.
        let options = spv::Options {
            flags: spv::WriterFlags::empty(),
            ..spv::Options::default()
        };
        let words = spv::write_vec(&module, &info, &options, None)
            .map_err(|e| format!("{}: SPIR-V: {e}", path.display()))?;

        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        std::fs::write(out_dir.join(format!("{name}.spv")), bytes)?;
    }
    println!("cargo:rerun-if-changed=build.rs");
    Ok(())
}
