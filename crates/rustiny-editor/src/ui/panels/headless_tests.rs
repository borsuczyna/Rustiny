//! Panel interaction tests that run real Dear ImGui frames without a window or GPU.
//!
//! A frame is built, mouse events are injected the way the platform backend would deliver them,
//! and the resulting selection is checked. Dear ImGui allows only one context per process, so
//! everything lives in a single test function.

use super::hierarchy::{self, State as HierarchyState};
use super::{inspector, scene_view};
use crate::ui::{ProjectState, SceneState};
use dear_imgui_rs::{Condition, Context, Key, MouseButton, Ui, WindowKey};
use rustiny_scene::read_scene;
use std::io::Cursor;
use std::sync::Arc;

fn game_object(id: usize, name: &str, father: usize, children: &[usize], pos: [f32; 3]) -> String {
    let (go, tf) = (id * 2 + 1, id * 2 + 2);
    let father = if father == usize::MAX {
        0
    } else {
        father * 2 + 2
    };
    let mut text = format!(
        "--- !u!1 &{go}\nGameObject:\n  m_Component:\n  - component: {{fileID: {tf}}}\n  m_Name: {name}\n--- !u!4 &{tf}\nTransform:\n  m_GameObject: {{fileID: {go}}}\n  m_LocalRotation: {{x: 0, y: 0, z: 0, w: 1}}\n  m_LocalPosition: {{x: {}, y: {}, z: {}}}\n  m_LocalScale: {{x: 1, y: 1, z: 1}}\n",
        pos[0], pos[1], pos[2]
    );
    if children.is_empty() {
        text.push_str("  m_Children: []\n");
    } else {
        text.push_str("  m_Children:\n");
        for c in children {
            text.push_str(&format!("  - {{fileID: {}}}\n", c * 2 + 2));
        }
    }
    text.push_str(&format!("  m_Father: {{fileID: {father}}}\n"));
    text
}

fn scene(objects: &[String]) -> SceneState {
    let mut text = String::from("%YAML 1.1\n%TAG !u! tag:unity3d.com,2011:\n");
    for o in objects {
        text.push_str(o);
    }
    let scene = read_scene(Cursor::new(text.into_bytes()), "Test").unwrap();
    SceneState::Ready {
        scene: Arc::new(scene),
        path: "Test.unity".to_owned(),
        shapes: Default::default(),
    }
}

/// Give a window a fixed place and size by opening it once, empty, before the panel does.
fn place(ui: &Ui, key: &WindowKey) {
    ui.window(key)
        .position([0.0, 0.0], Condition::Always)
        .size([400.0, 300.0], Condition::Always)
        .build(|| {});
}

struct Harness {
    ctx: Context,
}

impl Harness {
    fn new() -> Self {
        let mut ctx = Context::create();
        ctx.set_ini_filename(None::<std::path::PathBuf>).unwrap();
        ctx.font_atlas()
            .try_claim_legacy_renderer()
            .expect("font atlas")
            .build();
        ctx.io_mut().set_display_size([800.0, 600.0]);
        ctx.io_mut().set_delta_time(1.0 / 60.0);
        Self { ctx }
    }

    fn frame(&mut self, build: impl FnOnce(&Ui)) {
        let ui = self.ctx.frame();
        build(ui);
        assert!(self.ctx.end_frame());
    }

    /// Move the mouse, press and release the left button, each in its own frame.
    fn click(&mut self, pos: [f32; 2], mut build: impl FnMut(&Ui)) {
        self.ctx.io_mut().add_mouse_pos_event(pos);
        self.frame(&mut build);
        self.ctx
            .io_mut()
            .add_mouse_button_event(MouseButton::Left, true);
        self.frame(&mut build);
        self.ctx
            .io_mut()
            .add_mouse_button_event(MouseButton::Left, false);
        self.frame(&mut build);
    }
}

#[test]
fn clicking_selects_objects_in_the_hierarchy_and_the_scene_view() {
    let mut h = Harness::new();
    let key = |id: &str, title: &str| WindowKey::new(id, title).unwrap();
    let hierarchy_key = key("test.hierarchy", "Hierarchy");
    let inspector_key = key("test.inspector", "Inspector");
    let scene_key = key("test.scene", "Scene");
    let project = ProjectState::None;

    // --- Hierarchy: Alpha (with child Beta) and Gamma. ---
    let state = scene(&[
        game_object(0, "Alpha", usize::MAX, &[1], [0.0; 3]),
        game_object(1, "Beta", 0, &[], [1.0, 0.0, 0.0]),
        game_object(2, "Gamma", usize::MAX, &[], [5.0, 0.0, 0.0]),
    ]);
    let SceneState::Ready { scene: loaded, .. } = &state else {
        unreachable!()
    };
    let (alpha, gamma) = (loaded.roots()[0], loaded.roots()[1]);
    assert_eq!(loaded.object(alpha).unwrap().name, "Alpha");
    assert_eq!(loaded.object(gamma).unwrap().name, "Gamma");

    let mut selection = HierarchyState::default();
    let mut open = true;
    let mut draw_hierarchy = |ui: &Ui, selection: &mut HierarchyState| {
        place(ui, &hierarchy_key);
        hierarchy::draw(ui, &hierarchy_key, &mut open, &state, &project, selection);
    };

    h.frame(|ui| draw_hierarchy(ui, &mut selection));
    assert_eq!(
        selection.selected, None,
        "nothing is selected before a click"
    );

    // The rows are at unknown heights, so sweep down the window until a click lands on one.
    let mut selected_in_order = Vec::new();
    let mut y = 20.0;
    while y < 200.0 && selected_in_order.len() < 2 {
        h.click([120.0, y], |ui| draw_hierarchy(ui, &mut selection));
        if let Some(id) = selection.selected
            && selected_in_order.last() != Some(&id)
        {
            selected_in_order.push(id);
        }
        y += 4.0;
    }
    assert_eq!(
        selected_in_order,
        [alpha, gamma],
        "the first row is Alpha and, with Alpha collapsed, the next one is Gamma"
    );

    // The Inspector draws the selected object without trouble (and the empty case too).
    let mut open_inspector = true;
    for selected in [Some(gamma), None] {
        let shown = HierarchyState {
            selected,
            ..HierarchyState::default()
        };
        h.frame(|ui| {
            place(ui, &inspector_key);
            inspector::draw(
                ui,
                &inspector_key,
                &mut open_inspector,
                &state,
                &project,
                &shown,
            );
        });
    }

    // --- Scene view: one object at the origin, which the first frame centres in the view. ---
    let single = scene(&[game_object(0, "Only", usize::MAX, &[], [0.0; 3])]);
    let mut scene_selection = HierarchyState::default();
    let mut view = scene_view::State::default();
    let mut open_view = true;
    let mut draw_view = |ui: &Ui, selection: &mut HierarchyState, view: &mut scene_view::State| {
        place(ui, &scene_key);
        scene_view::draw(
            ui,
            &scene_key,
            &mut open_view,
            &single,
            &project,
            selection,
            view,
        );
    };
    h.frame(|ui| draw_view(ui, &mut scene_selection, &mut view));

    // Click far from the marker (low in the view, away from the title bar): nothing to pick.
    h.click([30.0, 250.0], |ui| {
        draw_view(ui, &mut scene_selection, &mut view)
    });
    assert_eq!(scene_selection.selected, None);

    // Sweep the window; exactly the marker near the centre of the view selects the object.
    let mut hit = None;
    'sweep: for y in (30..300).step_by(6) {
        for x in (10..390).step_by(6) {
            h.click([x as f32, y as f32], |ui| {
                draw_view(ui, &mut scene_selection, &mut view)
            });
            if scene_selection.selected.is_some() {
                hit = Some([x, y]);
                break 'sweep;
            }
        }
    }
    let hit = hit.expect("some click near the centre must select the only object");
    assert_eq!(scene_selection.selected, Some(0));
    assert!(
        (100..300).contains(&hit[0]) && (60..220).contains(&hit[1]),
        "the object was framed near the middle of the 400x300 window, but the hit was at {hit:?}"
    );
    // Flying: hold the right mouse button over the view and press W; the camera moves forward.
    let before = view.eye();
    h.ctx.io_mut().add_mouse_pos_event([200.0, 160.0]);
    h.frame(|ui| draw_view(ui, &mut scene_selection, &mut view));
    h.ctx
        .io_mut()
        .add_mouse_button_event(MouseButton::Right, true);
    h.ctx.io_mut().add_key_event(Key::W, true);
    for _ in 0..10 {
        h.frame(|ui| draw_view(ui, &mut scene_selection, &mut view));
    }
    h.ctx.io_mut().add_key_event(Key::W, false);
    h.ctx
        .io_mut()
        .add_mouse_button_event(MouseButton::Right, false);
    h.frame(|ui| draw_view(ui, &mut scene_selection, &mut view));
    let after = view.eye();
    assert_ne!(
        before, after,
        "W with the right button held must fly the camera"
    );
    // Without the right button the keys do nothing.
    h.ctx.io_mut().add_key_event(Key::W, true);
    for _ in 0..5 {
        h.frame(|ui| draw_view(ui, &mut scene_selection, &mut view));
    }
    h.ctx.io_mut().add_key_event(Key::W, false);
    h.frame(|ui| draw_view(ui, &mut scene_selection, &mut view));
    assert_eq!(view.eye(), after);

    // Clicking empty space again clears the selection.
    h.click([30.0, 250.0], |ui| {
        draw_view(ui, &mut scene_selection, &mut view)
    });
    assert_eq!(scene_selection.selected, None);
}
