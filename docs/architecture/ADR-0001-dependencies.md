# ADR-0001: Dependency selection

Versions were checked against crates.io on 2026-11-15 (not assumed).

| Need | Choice | Version | Licence | Why / limits |
|---|---|---|---|---|
| Vulkan bindings | `ash` | 0.38 | MIT/Apache-2.0 | Full control, thin. Version is dictated by the ImGui backend (it targets ash 0.38). |
| Surface creation | `ash-window` | 0.13 | MIT/Apache-2.0 | Matches ash 0.38 / raw-window-handle 0.6. |
| GPU memory | `gpu-allocator` | 0.28 | MIT/Apache-2.0 | Pure Rust; shared with the ImGui backend. |
| Window/events | `winit` | 0.30.13 | Apache-2.0 | 0.31 is still beta; the ImGui winit backend targets 0.30. |
| Editor UI | `dear-imgui-rs` (+ `-winit`, `-ash`) | 0.22 | MIT/Apache-2.0 | Docking, managed textures (ImGui 1.92), dynamic-rendering Vulkan backend, multi-viewport route. Rejected: `imgui` 0.12 (older ImGui, no maintained winit-0.30/ash-0.38 multi-viewport story). **Risk:** the backend self-describes as experimental; young crate. Mitigated by the smoke test. |
| Logging | `tracing`, `tracing-subscriber` | 0.1 / 0.3 | MIT | Structured, layered (console capture layer). |
| Errors | `thiserror` | 2 | MIT/Apache-2.0 | Typed errors; no `anyhow` in libraries. |
| Config | `serde`, `toml` | 1 / 1 | MIT/Apache-2.0 | |
| User dirs | `directories` | 6 | MIT/Apache-2.0 | |
| PNG output | `png` | 0.18 | MIT/Apache-2.0 | Screenshot files. |

Deferred until needed: `rapier3d` (0.36, Apache-2.0) for physics, `rusqlite` (0.43, MIT) for the asset database, `notify` (9.0 is still `rc`; re-evaluate) for file watching, `glam` for math, ECS (`hecs` 0.11 / `bevy_ecs` 0.20 to be evaluated in Milestone 4).

Project licence: `MIT OR Apache-2.0` (matches every dependency above; permits later commercial licensing decisions to be revisited). No Unity code, binaries or assets are used.
