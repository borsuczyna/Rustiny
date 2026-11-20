# Rustiny

A high-performance, Unity-compatible game engine and editor written in Rust (Vulkan + Dear ImGui). **Early stage: Milestones 0-1 are complete, Milestones 3, 4 and 5 have started; see `MILESTONES.md` for exactly what exists.** It can index a Unity project's assets and GUIDs and show a scene's hierarchy and its objects in a GPU-rendered Scene view (all read-only; prefab instances of FBX models are drawn as the real models with name-matched materials and albedo textures, everything else as boxes), but cannot expand `.prefab` assets, import other formats, edit anything or run scripts yet.

## Build and run

Requirements: Rust (version pinned in `rust-toolchain.toml`), CMake and a C++ toolchain (Dear ImGui is compiled from source), a Vulkan 1.3 capable GPU/driver. The Vulkan SDK is recommended for the validation layers.

```bash
cargo run -p rustiny-editor            # start the editor
cargo test --workspace                 # unit + headless Vulkan + end-to-end smoke test
cargo run -p rustiny-editor -- --smoke-frames 120 --screenshot out.png
```

Open a Unity project (read-only, indexed in the background, shown in the Project panel and kept up to date as files change): `cargo run -p rustiny-editor -- --project <path>`. Open a scene (text-serialized `.unity`, read-only) in the Hierarchy and Inspector with `--scene <file.unity>` or by double-clicking it in the Project panel. In the Scene panel hold the right mouse button and use W A S D to fly; click an object's box to select it.

Options: `--continuous`, `--no-validation`, `--config-dir <path>`; see `--help`. Configuration lives in `%APPDATA%\Rustiny` (Windows) / `~/.config/Rustiny` (Linux).

## Docs

* `docs/architecture/ARCHITECTURE.md`, `ADR-*.md` - design and decisions
* `docs/compatibility/UNITY_COMPATIBILITY.md` - compatibility matrix
* `MILESTONES.md` - progress

Licence: MIT OR Apache-2.0.
