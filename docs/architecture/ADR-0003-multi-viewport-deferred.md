# ADR-0003: ImGui multi-viewport deferred

**Status:** accepted for Milestone 1.

The requirement is "multi-viewport, if it works stably". `dear-imgui-ash` offers a multi-viewport route, but it replaces the simple `prepare_frame` / `cmd_draw` flow with a transactional protocol (open frame token, per-viewport surfaces and swapchains, completion capabilities that must outlive every submission). Adopting it means restructuring the frame loop and its tests.

Decision: ship Milestone 1 with docking in a single OS window, with the crate features (`multi-viewport`) already enabled so the switch is local to `rustiny-editor::gui`/`app`. Revisit at the start of Milestone 2 with its own validation-clean smoke test, because secondary windows multiply swapchain/synchronisation risk.

Consequence: panels cannot yet be dragged out of the main window.
