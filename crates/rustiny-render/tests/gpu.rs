//! Headless GPU tests: render the scene pass offscreen and read pixels back.
//!
//! Needs a Vulkan loader and a Vulkan 1.3 device; without one each test prints a `SKIPPED` line
//! and returns, so machines without a GPU stay green while the skip stays visible in the log.
//! Validation and synchronisation validation are on, and every test ends by asserting that they
//! reported nothing.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{AllocationCreateDesc, AllocationScheme};
use rustiny_render::matrix::{project, view_projection};
use rustiny_render::{
    FrameParams, GpuInstance, MeshGroup, MeshSection, MeshVertex, SceneRenderer, Surface,
};
use rustiny_vulkan::{Device, DeviceConfig, Instance, InstanceConfig, VulkanError};
use std::sync::Arc;

const CLEAR: [f32; 4] = [0.1, 0.2, 0.3, 1.0];
const CLEAR_BYTES: [u8; 3] = [26, 51, 77];
const W: u32 = 128;
const H: u32 = 96;

fn device() -> Option<Arc<Device>> {
    let instance = match Instance::new(&InstanceConfig {
        validation: true,
        sync_validation: true,
        ..Default::default()
    }) {
        Ok(i) => i,
        Err(VulkanError::Load(e)) => {
            eprintln!("SKIPPED: no Vulkan loader ({e})");
            return None;
        }
        Err(e) => panic!("instance creation failed: {e}"),
    };
    match Device::new(&instance, None, &DeviceConfig::default()) {
        Ok(d) => Some(d),
        Err(VulkanError::NoSuitableDevice(why)) => {
            eprintln!("SKIPPED: no suitable device ({why})");
            None
        }
        Err(e) => panic!("device creation failed: {e}"),
    }
}

/// Column-major model matrix: uniform `scale` and a translation.
fn model(scale: f32, at: [f32; 3]) -> [f32; 16] {
    let mut m = [0.0; 16];
    m[0] = scale;
    m[5] = scale;
    m[10] = scale;
    m[15] = 1.0;
    m[12..15].copy_from_slice(&at);
    m
}

/// Camera at `eye` looking along +Z tilted by `pitch` (yaw 0), as the editor's camera defines it.
fn view_proj(eye: [f32; 3], pitch: f32) -> [f32; 16] {
    let (sp, cp) = pitch.sin_cos();
    view_projection(
        eye,
        [1.0, 0.0, 0.0],
        [0.0, cp, -sp],
        [0.0, sp, cp],
        60.0_f32.to_radians(),
        W as f32 / H as f32,
        0.1,
    )
}

fn params(view_proj: [f32; 16], eye: [f32; 3]) -> FrameParams {
    FrameParams {
        view_proj,
        eye,
        selected: None,
        clear: CLEAR,
        grid_extent: 200.0,
        grid_cell: 1.0,
    }
}

struct Pixels(Vec<u8>);

impl Pixels {
    fn at(&self, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * W + x) * 4) as usize;
        [self.0[i], self.0[i + 1], self.0[i + 2], self.0[i + 3]]
    }

    fn is_clear(&self, x: u32, y: u32) -> bool {
        let p = self.at(x, y);
        (0..3).all(|c| p[c].abs_diff(CLEAR_BYTES[c]) <= 1) && p[3] == 255
    }

    /// The pixel in the `radius` window around `(x, y)` for which `score` is highest.
    fn best(&self, x: i32, y: i32, radius: i32, score: impl Fn([u8; 4]) -> i32) -> [u8; 4] {
        let mut best = (i32::MIN, [0; 4]);
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let (px, py) = (x + dx, y + dy);
                if (0..W as i32).contains(&px) && (0..H as i32).contains(&py) {
                    let p = self.at(px as u32, py as u32);
                    best = best.max((score(p), p));
                }
            }
        }
        best.1
    }
}

/// A device, a renderer and a command pool; renders one frame at a time and reads it back.
struct Rig {
    device: Arc<Device>,
    pool: vk::CommandPool,
    renderer: Option<SceneRenderer>,
}

impl Rig {
    fn new() -> Option<Self> {
        let device = device()?;
        let mut renderer = SceneRenderer::new(&device).expect("scene renderer");
        assert!(renderer.resize(W, H).expect("target"));
        assert!(
            !renderer.resize(W, H).expect("same size"),
            "an unchanged size must not recreate the target"
        );
        let info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(device.families().graphics)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: plain object creation on a live device.
        let pool = unsafe { device.raw().create_command_pool(&info, None) }.unwrap();
        Some(Self {
            device,
            pool,
            renderer: Some(renderer),
        })
    }

    fn renderer(&mut self) -> &mut SceneRenderer {
        self.renderer.as_mut().unwrap()
    }

    /// Record the scene pass, copy the colour image to a host buffer, wait, and return the pixels.
    fn render(&mut self, params: &FrameParams) -> Pixels {
        let raw = self.device.raw().clone();
        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: the pool was created on this device and is used from one thread.
        let cmd = unsafe { raw.allocate_command_buffers(&alloc) }.unwrap()[0];

        let size = u64::from(W) * u64::from(H) * 4;
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST);
        // SAFETY: plain object creation.
        let buffer = unsafe { raw.create_buffer(&buffer_info, None) }.unwrap();
        // SAFETY: the buffer was just created.
        let requirements = unsafe { raw.get_buffer_memory_requirements(buffer) };
        let mut allocation = self
            .device
            .allocator()
            .lock()
            .unwrap()
            .allocate(&AllocationCreateDesc {
                name: "test-readback",
                requirements,
                location: MemoryLocation::GpuToCpu,
                linear: true,
                allocation_scheme: AllocationScheme::DedicatedBuffer(buffer),
            })
            .unwrap();
        // SAFETY: the allocation satisfies the requirements.
        unsafe { raw.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()) }
            .unwrap();

        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: fresh command buffer.
        unsafe { raw.begin_command_buffer(cmd, &begin) }.unwrap();
        let renderer = self.renderer.as_mut().unwrap();
        assert!(renderer.record(cmd, params), "a target exists");
        let image = renderer.color_image().unwrap();

        let to_copy = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
            .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1),
            );
        let barriers = [to_copy];
        let dependency = vk::DependencyInfo::default().image_memory_barriers(&barriers);
        let region = vk::BufferImageCopy::default()
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1),
            )
            .image_extent(vk::Extent3D {
                width: W,
                height: H,
                depth: 1,
            });
        let to_host = vk::BufferMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(buffer)
            .size(vk::WHOLE_SIZE);
        let buffer_barriers = [to_host];
        let host_dependency =
            vk::DependencyInfo::default().buffer_memory_barriers(&buffer_barriers);
        // SAFETY: the command buffer is recording; the image is in the layout the barrier names.
        unsafe {
            raw.cmd_pipeline_barrier2(cmd, &dependency);
            raw.cmd_copy_image_to_buffer(
                cmd,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                buffer,
                &[region],
            );
            raw.cmd_pipeline_barrier2(cmd, &host_dependency);
            raw.end_command_buffer(cmd).unwrap();
        }

        // SAFETY: plain object creation.
        let fence = unsafe { raw.create_fence(&vk::FenceCreateInfo::default(), None) }.unwrap();
        let cmds = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&cmds);
        // SAFETY: the submission and fence are valid; waiting makes the readback safe.
        unsafe {
            raw.queue_submit(self.device.graphics_queue(), &[submit], fence)
                .unwrap();
            raw.wait_for_fences(&[fence], true, u64::MAX).unwrap();
            raw.destroy_fence(fence, None);
            raw.free_command_buffers(self.pool, &cmds);
        }
        let pixels = allocation.mapped_slice().unwrap()[..size as usize].to_vec();
        self.device
            .allocator()
            .lock()
            .unwrap()
            .free(std::mem::take(&mut allocation))
            .unwrap();
        // SAFETY: the fence wait above means the GPU is done with the buffer.
        unsafe { raw.destroy_buffer(buffer, None) };
        Pixels(pixels)
    }

    /// Tear down in the right order and assert validation stayed silent.
    fn finish(mut self) {
        let stats = self.device.instance().validation_stats().clone();
        self.renderer = None;
        // SAFETY: the renderer waited for idle in its Drop; the pool is no longer used.
        unsafe { self.device.raw().destroy_command_pool(self.pool, None) };
        let device = self.device;
        drop(device);
        assert_eq!(
            stats.errors(),
            0,
            "validation errors: {:?}",
            stats.retained_errors()
        );
    }
}

#[test]
fn a_box_is_drawn_lit_over_the_clear_colour() {
    let Some(mut rig) = Rig::new() else { return };
    let eye = [0.0, 0.0, 0.0];
    let p = params(view_proj(eye, 0.0), eye);
    rig.renderer()
        .set_instances(&[GpuInstance::new(
            model(2.0, [0.0, 0.0, 5.0]),
            [1.0, 0.0, 0.0],
            true,
        )])
        .unwrap();
    assert_eq!(rig.renderer().instance_count(), 1);
    let pixels = rig.render(&p);

    // Centre: the red face turned towards the camera, lambert 0.364 with ambient 0.35, gamma 2.2
    // gives about 200.
    let centre = pixels.at(W / 2, H / 2);
    assert!(
        (190..=210).contains(&centre[0]) && centre[1] < 4 && centre[2] < 4 && centre[3] == 255,
        "unexpected box colour {centre:?}"
    );
    for (x, y) in [(2, 2), (W - 3, 2), (2, H - 3), (W - 3, H - 3), (W / 2, 3)] {
        assert!(pixels.is_clear(x, y), "({x},{y}) is {:?}", pixels.at(x, y));
    }
    // The box (2 wide at distance 4) covers a bit under half the view's width: 60 degree vertical
    // FOV puts the image half-height at 4*tan(30) = 2.3, so it spans about 2/4.6 of the height.
    let covered = (0..W).filter(|&x| pixels.at(x, H / 2)[0] > 150).count() as f32;
    let expected = 2.0 / (4.0 * 30.0_f32.to_radians().tan() * 2.0) * H as f32;
    assert!(
        (covered - expected).abs() < 4.0,
        "box covers {covered} px across, expected about {expected}"
    );
    rig.finish();
}

#[test]
fn nearer_objects_hide_farther_ones_regardless_of_draw_order() {
    let Some(mut rig) = Rig::new() else { return };
    let eye = [0.0, 0.0, 0.0];
    let p = params(view_proj(eye, 0.0), eye);
    // The near green box is drawn first, the far red one second: without a depth test the red
    // one would overwrite it.
    rig.renderer()
        .set_instances(&[
            GpuInstance::new(model(1.0, [0.0, 0.0, 3.0]), [0.0, 1.0, 0.0], true),
            GpuInstance::new(model(6.0, [0.0, 0.0, 12.0]), [1.0, 0.0, 0.0], true),
        ])
        .unwrap();
    let pixels = rig.render(&p);
    let centre = pixels.at(W / 2, H / 2);
    assert!(
        centre[1] > 150 && centre[0] < 4,
        "the near green box must win at the centre: {centre:?}"
    );
    // Off to the side only the large far red box is visible: the green one spans about +-16 px
    // from the centre (1 unit at distance 2.5), the red one about +-28 px (6 units at 9).
    let side = pixels.best((W / 2) as i32 + 22, (H / 2) as i32, 1, |p| i32::from(p[0]));
    assert!(
        side[0] > 150 && side[1] < 4,
        "far red box beside it: {side:?}"
    );
    rig.finish();
}

#[test]
fn hidden_instances_are_skipped_unless_selected() {
    let Some(mut rig) = Rig::new() else { return };
    let eye = [0.0, 0.0, 0.0];
    let mut p = params(view_proj(eye, 0.0), eye);
    rig.renderer()
        .set_instances(&[GpuInstance::new(
            model(2.0, [0.0, 0.0, 5.0]),
            [1.0, 0.0, 0.0],
            false,
        )])
        .unwrap();
    let hidden = rig.render(&p);
    assert!(
        hidden.is_clear(W / 2, H / 2),
        "hidden: {:?}",
        hidden.at(W / 2, H / 2)
    );

    p.selected = Some(0);
    let selected = rig.render(&p);
    let centre = selected.at(W / 2, H / 2);
    assert!(
        centre[0] > 150 && centre[1] > 150 && centre[2] < 120,
        "a selected instance is drawn yellow even when hidden: {centre:?}"
    );

    // Replacing the instances while earlier frames are still retired keeps working.
    rig.renderer().set_instances(&[]).unwrap();
    assert_eq!(rig.renderer().instance_count(), 0);
    p.selected = None;
    assert!(rig.render(&p).is_clear(W / 2, H / 2));
    rig.finish();
}

#[test]
fn the_ground_grid_shows_the_world_axes() {
    let Some(mut rig) = Rig::new() else { return };
    let eye = [0.0, 5.0, -8.0];
    let vp = view_proj(eye, -0.5);
    let p = params(vp, eye);
    let pixels = rig.render(&p);

    let screen = |world: [f32; 3]| {
        let ndc = project(&vp, world).expect("in front of the camera");
        (
            ((ndc[0] + 1.0) * 0.5 * W as f32) as i32,
            ((ndc[1] + 1.0) * 0.5 * H as f32) as i32,
        )
    };
    // The X axis is red, the Z axis blue (both blended over the clear colour).
    let (x, y) = screen([3.0, 0.0, 0.0]);
    let red = pixels.best(x, y, 2, |p| i32::from(p[0]) - i32::from(p[2]));
    assert!(
        red[0] > red[2] + 20,
        "X axis should be reddish: {red:?} at ({x},{y})"
    );
    let (x, y) = screen([0.0, 0.0, 3.0]);
    let blue = pixels.best(x, y, 2, |p| i32::from(p[2]) - i32::from(p[0]));
    assert!(
        blue[2] > 120 && blue[2] > blue[0] + 40,
        "Z axis should be blue: {blue:?} at ({x},{y})"
    );
    // The sky above the horizon is untouched.
    assert!(pixels.is_clear(W / 2, 1), "{:?}", pixels.at(W / 2, 1));
    rig.finish();
}

#[test]
fn resizing_creates_a_new_target_and_keeps_rendering() {
    let Some(mut rig) = Rig::new() else { return };
    let eye = [0.0, 0.0, 0.0];
    let p = params(view_proj(eye, 0.0), eye);
    rig.renderer()
        .set_instances(&[GpuInstance::new(
            model(2.0, [0.0, 0.0, 5.0]),
            [1.0, 0.0, 0.0],
            true,
        )])
        .unwrap();
    let before = rig.renderer().color_view().unwrap();
    assert!(rig.renderer().resize(W, H + 8).unwrap());
    assert_ne!(rig.renderer().color_view().unwrap(), before);
    assert!(rig.renderer().resize(W, H).unwrap());
    // Zero sizes are clamped rather than failing.
    assert!(rig.renderer().resize(0, 0).unwrap());
    assert_eq!(
        rig.renderer().extent().map(|e| (e.width, e.height)),
        Some((1, 1))
    );
    assert!(rig.renderer().resize(W, H).unwrap());
    let pixels = rig.render(&p);
    assert!(pixels.at(W / 2, H / 2)[0] > 150);
    rig.finish();
}

/// A 2x2 quad at z = 0 facing the camera (normal -Z): two triangles, lower-right then upper-left.
fn quad() -> (Vec<MeshVertex>, Vec<u32>) {
    let v = |x: f32, y: f32| MeshVertex {
        position: [x, y, 0.0],
        normal: [0.0, 0.0, -1.0],
        uv: [x * 0.5 + 0.5, y * 0.5 + 0.5],
    };
    (
        vec![v(-1.0, -1.0), v(1.0, -1.0), v(1.0, 1.0), v(-1.0, 1.0)],
        vec![0, 1, 2, 0, 2, 3],
    )
}

fn section(first: u32, tint: [f32; 3]) -> MeshSection {
    surface_section(
        first,
        Surface {
            tint,
            ..Surface::default()
        },
    )
}

fn surface_section(first: u32, surface: Surface) -> MeshSection {
    MeshSection {
        first_index: first,
        index_count: 3,
        surface,
    }
}

/// Both triangles of the quad with the same surface.
fn both(surface: Surface) -> [MeshSection; 2] {
    [surface_section(0, surface), surface_section(3, surface)]
}

/// Both triangles of the quad, textured.
fn textured(texture: u32, tint: [f32; 3], cutoff: f32) -> [MeshSection; 2] {
    both(Surface {
        tint,
        albedo: Some(texture),
        cutoff,
        ..Surface::default()
    })
}

/// A 2x2 texture, texel (0,0) red, (1,0) green, (0,1) blue, (1,1) white, with the given alpha for
/// the left column, and its one-texel mip.
fn quadrants(left_alpha: u8) -> Vec<Vec<u8>> {
    let base = vec![
        255, 0, 0, left_alpha, //
        0, 255, 0, 255, //
        0, 0, 255, left_alpha, //
        255, 255, 255, 255,
    ];
    rustiny_render::mip_chain(2, 2, base)
}

/// Put one instance of mesh `key` (the quad) 5 units ahead of the camera.
fn place_quad(r: &mut SceneRenderer, key: u32) {
    r.set_mesh_instances(&[MeshGroup {
        key,
        instances: vec![GpuInstance::new(
            model(1.0, [0.0, 0.0, 5.0]),
            [1.0; 3],
            true,
        )],
    }])
    .unwrap();
}

fn screen(vp: &[f32; 16], world: [f32; 3]) -> (u32, u32) {
    let ndc = project(vp, world).expect("in front of the camera");
    (
        ((ndc[0] + 1.0) * 0.5 * W as f32) as u32,
        ((ndc[1] + 1.0) * 0.5 * H as f32) as u32,
    )
}

#[test]
fn a_mesh_is_drawn_per_section_with_each_sections_tint() {
    let Some(mut rig) = Rig::new() else { return };
    let eye = [0.0, 0.0, 0.0];
    let vp = view_proj(eye, 0.0);
    let (vertices, indices) = quad();
    let r = rig.renderer();
    r.add_mesh(
        7,
        &vertices,
        &indices,
        &[section(0, [1.0, 0.0, 0.0]), section(3, [0.0, 1.0, 0.0])],
    )
    .unwrap();
    r.set_mesh_instances(&[MeshGroup {
        key: 7,
        instances: vec![GpuInstance::new(
            model(1.0, [0.0, 0.0, 5.0]),
            [1.0; 3],
            true,
        )],
    }])
    .unwrap();
    assert!(r.has_mesh(7) && !r.has_mesh(8));
    let pixels = rig.render(&params(vp, eye));

    let (x, y) = screen(&vp, [0.6, -0.6, 5.0]);
    let lower_right = pixels.at(x, y);
    assert!(
        lower_right[0] > 150 && lower_right[1] < 4,
        "the first section is red: {lower_right:?}"
    );
    let (x, y) = screen(&vp, [-0.6, 0.6, 5.0]);
    let upper_left = pixels.at(x, y);
    assert!(
        upper_left[1] > 150 && upper_left[0] < 4,
        "the second section is green: {upper_left:?}"
    );
    let stats = rig.renderer().mesh_stats();
    assert_eq!(
        (
            stats.meshes,
            stats.instances,
            stats.triangles_drawn,
            stats.draw_calls
        ),
        (1, 1, 2, 2),
        "one instance, two sections of one triangle each"
    );
    rig.finish();
}

#[test]
fn instances_of_one_mesh_are_one_draw_and_selection_follows_the_object_id() {
    let Some(mut rig) = Rig::new() else { return };
    let eye = [0.0, 0.0, 0.0];
    let vp = view_proj(eye, 0.0);
    let (vertices, indices) = quad();
    let r = rig.renderer();
    r.add_mesh(
        1,
        &vertices,
        &indices,
        &[section(0, [0.0, 0.0, 1.0]), section(3, [0.0, 0.0, 1.0])],
    )
    .unwrap();
    // Two copies side by side; the ids are scene object ids, unrelated to their buffer position.
    r.set_mesh_instances(&[MeshGroup {
        key: 1,
        instances: vec![
            GpuInstance::new(model(1.0, [-2.0, 0.0, 6.0]), [1.0; 3], true).with_object(40),
            GpuInstance::new(model(1.0, [2.0, 0.0, 6.0]), [1.0; 3], true).with_object(9000),
        ],
    }])
    .unwrap();
    let mut p = params(vp, eye);
    let plain = rig.render(&p);
    let (lx, ly) = screen(&vp, [-2.0, 0.0, 6.0]);
    let (rx, ry) = screen(&vp, [2.0, 0.0, 6.0]);
    for (x, y) in [(lx, ly), (rx, ry)] {
        let c = plain.at(x, y);
        assert!(
            c[2] > 150 && c[0] < 4 && c[1] < 4,
            "blue copy at ({x},{y}): {c:?}"
        );
    }
    assert_eq!(
        rig.renderer().mesh_stats().draw_calls,
        2,
        "one draw per section, not per instance"
    );
    assert_eq!(rig.renderer().mesh_stats().triangles_drawn, 4);

    p.selected = Some(9000);
    let selected = rig.render(&p);
    let c = selected.at(rx, ry);
    assert!(
        c[0] > 150 && c[1] > 150,
        "the right copy turns yellow: {c:?}"
    );
    let c = selected.at(lx, ly);
    assert!(c[2] > 150 && c[0] < 4, "the left copy stays blue: {c:?}");
    rig.finish();
}

#[test]
fn instances_wait_for_their_mesh_and_a_replaced_mesh_takes_over() {
    let Some(mut rig) = Rig::new() else { return };
    let eye = [0.0, 0.0, 0.0];
    let vp = view_proj(eye, 0.0);
    let (vertices, indices) = quad();
    let r = rig.renderer();
    // Instances first, mesh not there yet: nothing is drawn and nothing breaks.
    r.set_mesh_instances(&[MeshGroup {
        key: 5,
        instances: vec![GpuInstance::new(
            model(1.0, [0.0, 0.0, 5.0]),
            [1.0; 3],
            true,
        )],
    }])
    .unwrap();
    let p = params(vp, eye);
    assert!(rig.render(&p).is_clear(W / 2, H / 2));
    assert_eq!(rig.renderer().mesh_stats().draw_calls, 0);

    // The mesh streams in later and appears.
    rig.renderer()
        .add_mesh(
            5,
            &vertices,
            &indices,
            &[section(0, [1.0, 0.0, 0.0]), section(3, [1.0, 0.0, 0.0])],
        )
        .unwrap();
    assert!(rig.render(&p).at(W / 2, H / 2)[0] > 150);

    // Replacing the mesh under the same key switches to the new geometry and colour.
    rig.renderer()
        .add_mesh(
            5,
            &vertices,
            &indices,
            &[section(0, [0.0, 1.0, 0.0]), section(3, [0.0, 1.0, 0.0])],
        )
        .unwrap();
    let green = rig.render(&p).at(W / 2, H / 2);
    assert!(green[1] > 150 && green[0] < 4, "{green:?}");

    // Clearing drops it again; instances stay and wait for the next upload.
    rig.renderer().clear_meshes();
    assert!(rig.render(&p).is_clear(W / 2, H / 2));
    assert_eq!(rig.renderer().mesh_stats().meshes, 0);
    rig.finish();
}

#[test]
fn inconsistent_meshes_are_rejected_without_side_effects() {
    let Some(mut rig) = Rig::new() else { return };
    let (vertices, indices) = quad();
    let r = rig.renderer();
    assert!(
        r.add_mesh(1, &vertices, &[0, 1, 9], &[]).is_err(),
        "index past the end"
    );
    assert!(
        r.add_mesh(
            1,
            &vertices,
            &indices,
            &[section(3, [1.0; 3]), section(6, [1.0; 3])]
        )
        .is_err(),
        "section past the end"
    );
    assert!(r.add_mesh(1, &[], &[], &[]).is_err(), "empty mesh");
    assert!(!r.has_mesh(1));
    assert_eq!(r.mesh_stats().meshes, 0);
    rig.finish();
}

#[test]
fn a_textured_section_shows_the_texture_and_the_tint_multiplies_it() {
    let Some(mut rig) = Rig::new() else { return };
    let eye = [0.0, 0.0, 0.0];
    let vp = view_proj(eye, 0.0);
    let (vertices, indices) = quad();
    let r = rig.renderer();
    r.add_texture(3, 2, 2, &quadrants(255), true).unwrap();
    r.add_mesh(1, &vertices, &indices, &textured(3, [1.0; 3], 0.0))
        .unwrap();
    place_quad(r, 1);
    assert!(r.has_texture(3) && !r.has_texture(4));
    assert!(
        !r.texture_uploaded(3),
        "the copy is recorded by the first pass, not by add_texture"
    );
    assert_eq!(
        (r.mesh_stats().textures, r.mesh_stats().texture_bytes),
        (1, 16 + 4)
    );
    let p = params(vp, eye);
    let pixels = rig.render(&p);
    assert!(rig.renderer().texture_uploaded(3));

    // Texel centres sit at the quad's quadrant centres (V up, row 0 at the bottom).
    let at = |wx: f32, wy: f32| {
        let (x, y) = screen(&vp, [wx, wy, 5.0]);
        pixels.at(x, y)
    };
    let red = at(-0.5, -0.5);
    assert!(red[0] > 150 && red[1] < 40 && red[2] < 40, "red {red:?}");
    let green = at(0.5, -0.5);
    assert!(
        green[1] > 150 && green[0] < 40 && green[2] < 40,
        "green {green:?}"
    );
    let blue = at(-0.5, 0.5);
    assert!(
        blue[2] > 150 && blue[0] < 40 && blue[1] < 40,
        "blue {blue:?}"
    );
    let white = at(0.5, 0.5);
    assert!(white.iter().take(3).all(|&c| c > 150), "white {white:?}");

    // A green tint keeps only the green channel of the texture.
    rig.renderer()
        .add_mesh(1, &vertices, &indices, &textured(3, [0.0, 1.0, 0.0], 0.0))
        .unwrap();
    let pixels = rig.render(&p);
    let (x, y) = screen(&vp, [-0.5, -0.5, 5.0]);
    let tinted_red = pixels.at(x, y);
    assert!(
        tinted_red[0] < 4 && tinted_red[1] < 50,
        "red texel times green tint is (nearly) black, a little green leaks in from the          linearly filtered neighbours: {tinted_red:?}"
    );
    let (x, y) = screen(&vp, [0.5, 0.5, 5.0]);
    let tinted_white = pixels.at(x, y);
    assert!(
        tinted_white[1] > 150 && tinted_white[0] < 4 && tinted_white[2] < 4,
        "white texel times green tint is green: {tinted_white:?}"
    );
    rig.finish();
}

#[test]
fn texels_below_the_cutoff_are_discarded() {
    let Some(mut rig) = Rig::new() else { return };
    let eye = [0.0, 0.0, 0.0];
    let vp = view_proj(eye, 0.0);
    let (vertices, indices) = quad();
    let r = rig.renderer();
    // The left column is transparent.
    r.add_texture(1, 2, 2, &quadrants(0), true).unwrap();
    r.add_mesh(1, &vertices, &indices, &textured(1, [1.0; 3], 0.5))
        .unwrap();
    place_quad(r, 1);
    let pixels = rig.render(&params(vp, eye));
    let (x, y) = screen(&vp, [-0.5, -0.5, 5.0]);
    assert!(
        pixels.is_clear(x, y),
        "transparent texel shows the background"
    );
    let (x, y) = screen(&vp, [-0.5, 0.5, 5.0]);
    assert!(pixels.is_clear(x, y), "second transparent texel too");
    let (x, y) = screen(&vp, [0.5, -0.5, 5.0]);
    assert!(pixels.at(x, y)[1] > 150, "opaque texel is drawn");

    // Without a cutoff the same texture draws everywhere.
    rig.renderer()
        .add_mesh(1, &vertices, &indices, &textured(1, [1.0; 3], 0.0))
        .unwrap();
    let pixels = rig.render(&params(vp, eye));
    let (x, y) = screen(&vp, [-0.5, -0.5, 5.0]);
    assert!(!pixels.is_clear(x, y), "cutoff 0 draws every texel");
    rig.finish();
}

#[test]
fn sections_with_a_missing_texture_draw_plain_and_clearing_drops_textures() {
    let Some(mut rig) = Rig::new() else { return };
    let eye = [0.0, 0.0, 0.0];
    let vp = view_proj(eye, 0.0);
    let (vertices, indices) = quad();
    let r = rig.renderer();
    // The mesh names texture 9, which is never added: the white default stands in.
    r.add_mesh(1, &vertices, &indices, &textured(9, [1.0, 0.0, 0.0], 0.0))
        .unwrap();
    place_quad(r, 1);
    let p = params(vp, eye);
    let c = rig.render(&p).at(W / 2, H / 2);
    assert!(c[0] > 150 && c[1] < 4, "plain red tint: {c:?}");

    // Add it later: the section picks it up without touching the mesh.
    rig.renderer()
        .add_texture(9, 2, 2, &quadrants(255), true)
        .unwrap();
    let pixels = rig.render(&p);
    let (x, y) = screen(&vp, [0.5, -0.5, 5.0]);
    let c = pixels.at(x, y);
    assert!(
        c[0] < 60 && c[1] < 4,
        "red tint times a green texel is (nearly) black: {c:?}"
    );

    // Replacing the texture under the same key switches to the new one.
    let solid = rustiny_render::mip_chain(1, 1, vec![255, 255, 255, 255]);
    rig.renderer().add_texture(9, 1, 1, &solid, true).unwrap();
    assert_eq!(rig.renderer().mesh_stats().textures, 1);
    let c = rig.render(&p).at(W / 2, H / 2);
    assert!(c[0] > 150 && c[1] < 4, "white texture, red tint: {c:?}");

    rig.renderer().clear_textures();
    assert!(!rig.renderer().has_texture(9));
    assert_eq!(rig.renderer().mesh_stats().textures, 0);
    let c = rig.render(&p).at(W / 2, H / 2);
    assert!(c[0] > 150 && c[1] < 4, "back to plain: {c:?}");
    // The default survives clearing, and textures can be added again.
    rig.renderer()
        .add_texture(9, 2, 2, &quadrants(255), true)
        .unwrap();
    rig.render(&p);
    assert!(rig.renderer().texture_uploaded(9));
    rig.finish();
}

#[test]
fn inconsistent_textures_are_rejected_without_side_effects() {
    let Some(mut rig) = Rig::new() else { return };
    let r = rig.renderer();
    assert!(
        r.add_texture(1, 0, 4, &[vec![0; 16]], true).is_err(),
        "empty size"
    );
    assert!(r.add_texture(1, 2, 2, &[], true).is_err(), "no mips");
    assert!(
        r.add_texture(1, 2, 2, &[vec![0; 15]], true).is_err(),
        "wrong byte count"
    );
    assert!(
        r.add_texture(1, 2, 2, &[vec![0; 16], vec![0; 8]], true)
            .is_err(),
        "wrong mip size"
    );
    assert!(!r.has_texture(1));
    assert_eq!(r.mesh_stats().textures, 0);
    rig.finish();
}

#[test]
fn many_textures_stream_in_over_several_passes() {
    let Some(mut rig) = Rig::new() else { return };
    let eye = [0.0, 0.0, 0.0];
    let p = params(view_proj(eye, 0.0), eye);
    let r = rig.renderer();
    // 700 textures of 256x256 (about 350 KB each with mips) exceed one pass's budget and one
    // descriptor pool.
    let count = 700u32;
    for key in 0..count {
        let base = vec![(key % 251) as u8; 256 * 256 * 4];
        r.add_texture(
            key,
            256,
            256,
            &rustiny_render::mip_chain(256, 256, base),
            true,
        )
        .unwrap();
    }
    assert_eq!(r.mesh_stats().textures, count as usize);
    assert!(!r.texture_uploaded(count - 1));
    let mut passes = 0;
    while !rig.renderer().texture_uploaded(count - 1) {
        rig.render(&p);
        passes += 1;
        assert!(passes < 200, "textures never finished uploading");
    }
    assert!(
        passes > 1,
        "the per-pass budget spreads the upload over several passes"
    );
    assert!((0..count).all(|k| rig.renderer().texture_uploaded(k)));
    rig.finish();
}

/// A 1x1 linear texture holding `texel`.
fn solid(texel: [u8; 4]) -> Vec<Vec<u8>> {
    vec![texel.to_vec()]
}

/// The colour at the centre of the lower-right and of the upper-left triangle of the quad.
fn both_triangles(pixels: &Pixels, vp: &[f32; 16]) -> ([u8; 4], [u8; 4]) {
    let (x, y) = screen(vp, [0.6, -0.6, 5.0]);
    let lower_right = pixels.at(x, y);
    let (x, y) = screen(vp, [-0.6, 0.6, 5.0]);
    (lower_right, pixels.at(x, y))
}

#[test]
fn a_normal_map_changes_the_lighting_of_a_flat_quad() {
    let Some(mut rig) = Rig::new() else { return };
    let eye = [0.0, 0.0, 0.0];
    let vp = view_proj(eye, 0.0);
    let (vertices, indices) = quad();
    let r = rig.renderer();
    // Tilted towards the light (+x, +y) and away from it (-x, -y); the quad's own normal is -Z,
    // which the sun (0.4, 0.8, -0.35) lights a little.
    r.add_texture(1, 1, 1, &solid([204, 204, 195, 255]), false)
        .unwrap();
    r.add_texture(2, 1, 1, &solid([51, 51, 195, 255]), false)
        .unwrap();
    let with = |normal: Option<u32>, first| {
        surface_section(
            first,
            Surface {
                normal,
                ..Surface::default()
            },
        )
    };
    r.add_mesh(
        1,
        &vertices,
        &indices,
        &[with(Some(1), 0), with(Some(2), 3)],
    )
    .unwrap();
    r.add_mesh(2, &vertices, &indices, &[with(None, 0), with(None, 3)])
        .unwrap();
    place_quad(r, 1);
    let (towards, away) = both_triangles(&rig.render(&params(vp, eye)), &vp);
    assert!(
        towards[0] > 215,
        "tilted towards the light is bright: {towards:?}"
    );
    assert!(
        (140..175).contains(&away[0]),
        "tilted away leaves only the ambient light: {away:?}"
    );

    place_quad(rig.renderer(), 2);
    let (flat_a, flat_b) = both_triangles(&rig.render(&params(vp, eye)), &vp);
    assert!(
        (190..212).contains(&flat_a[0]) && flat_a == flat_b,
        "without a normal map the quad is between the two: {flat_a:?} {flat_b:?}"
    );
    assert!(away[0] < flat_a[0] && flat_a[0] < towards[0]);
    rig.finish();
}

#[test]
fn metallic_smooth_surfaces_reflect_and_matte_ones_do_not() {
    let Some(mut rig) = Rig::new() else { return };
    let eye = [0.0, 0.0, 0.0];
    let vp = view_proj(eye, 0.0);
    let (vertices, indices) = quad();
    let red = |metallic: f32, smoothness: f32, map: Option<u32>| Surface {
        tint: [1.0, 0.0, 0.0],
        metallic,
        smoothness,
        metallic_map: map,
        ..Surface::default()
    };
    let r = rig.renderer();
    r.add_texture(5, 1, 1, &solid([255, 0, 0, 255]), false)
        .unwrap();
    // Matte, metal, and metal through a map (red = metallic, alpha = smoothness).
    r.add_mesh(1, &vertices, &indices, &both(red(0.0, 0.0, None)))
        .unwrap();
    r.add_mesh(2, &vertices, &indices, &both(red(1.0, 1.0, None)))
        .unwrap();
    r.add_mesh(3, &vertices, &indices, &both(red(0.0, 1.0, Some(5))))
        .unwrap();
    let mut shades = Vec::new();
    for key in [1, 2, 3] {
        place_quad(rig.renderer(), key);
        shades.push(rig.render(&params(vp, eye)).at(W / 2, H / 2));
    }
    let (matte, metal, mapped) = (shades[0], shades[1], shades[2]);
    assert!(
        (190..212).contains(&matte[0]) && matte[1] < 4,
        "matte red, diffuse light only: {matte:?}"
    );
    assert!(
        metal[0] > 225 && metal[1] < 4,
        "a smooth metal reflects the sky in its own colour: {metal:?}"
    );
    assert!(
        mapped[0].abs_diff(metal[0]) < 4,
        "the map's red channel makes it metallic: {mapped:?} vs {metal:?}"
    );
    rig.finish();
}

#[test]
fn unlit_surfaces_ignore_the_light_and_emission_adds_to_it() {
    let Some(mut rig) = Rig::new() else { return };
    let eye = [0.0, 0.0, 0.0];
    let vp = view_proj(eye, 0.0);
    let (vertices, indices) = quad();
    let r = rig.renderer();
    let green = Surface {
        tint: [0.0, 1.0, 0.0],
        ..Surface::default()
    };
    r.add_mesh(1, &vertices, &indices, &both(green)).unwrap();
    r.add_mesh(
        2,
        &vertices,
        &indices,
        &both(Surface {
            unlit: true,
            ..green
        }),
    )
    .unwrap();
    r.add_mesh(
        3,
        &vertices,
        &indices,
        &both(Surface {
            tint: [0.0; 3],
            emission: [0.0, 0.5, 0.0],
            ..Surface::default()
        }),
    )
    .unwrap();
    let mut shades = Vec::new();
    for key in [1, 2, 3] {
        place_quad(rig.renderer(), key);
        shades.push(rig.render(&params(vp, eye)).at(W / 2, H / 2));
    }
    assert!((190..212).contains(&shades[0][1]), "lit: {:?}", shades[0]);
    assert!(
        shades[1][1] > 250,
        "unlit is the full colour: {:?}",
        shades[1]
    );
    assert!(
        (175..200).contains(&shades[2][1]) && shades[2][0] < 4,
        "black with a half-strength green emission: {:?}",
        shades[2]
    );
    rig.finish();
}

#[test]
fn hundreds_of_texture_combinations_need_more_than_one_descriptor_pool() {
    let Some(mut rig) = Rig::new() else { return };
    let eye = [0.0, 0.0, 0.0];
    let vp = view_proj(eye, 0.0);
    let (vertices, indices) = quad();
    let r = rig.renderer();
    let count = 700u32;
    for key in 0..count {
        let green = if key == 0 { 255 } else { 0 };
        r.add_texture(key, 1, 1, &[vec![0, green, 0, 255]], true)
            .unwrap();
    }
    // One section per texture, all over the same triangle: every section needs its own set. The
    // first one drawn wins (the depth test is strict), and its texture is the green one.
    let sections: Vec<MeshSection> = (0..count)
        .map(|key| {
            surface_section(
                0,
                Surface {
                    albedo: Some(key),
                    ..Surface::default()
                },
            )
        })
        .collect();
    r.add_mesh(1, &vertices, &indices, &sections).unwrap();
    place_quad(r, 1);
    let p = params(vp, eye);
    let mut passes = 0;
    while rig.renderer().textures_waiting() > 0 || passes == 0 {
        rig.render(&p);
        passes += 1;
        assert!(passes < 100, "textures never finished uploading");
    }
    let pixels = rig.render(&p);
    let (x, y) = screen(&vp, [0.6, -0.6, 5.0]);
    let c = pixels.at(x, y);
    assert!(c[1] > 150 && c[0] < 4, "the first section's texture: {c:?}");
    assert_eq!(rig.renderer().mesh_stats().draw_calls, count);
    assert_eq!(rig.renderer().mesh_stats().materials, count as usize + 1);
    rig.finish();
}

#[test]
fn clearing_meshes_starts_the_material_records_over() {
    let Some(mut rig) = Rig::new() else { return };
    let eye = [0.0, 0.0, 0.0];
    let vp = view_proj(eye, 0.0);
    let (vertices, indices) = quad();
    let p = params(vp, eye);
    for round in 0..3 {
        let r = rig.renderer();
        r.clear_meshes();
        assert_eq!(r.mesh_stats().materials, 1, "only the neutral record");
        r.add_mesh(
            1,
            &vertices,
            &indices,
            &[section(0, [1.0, 0.0, 0.0]), section(3, [1.0, 0.0, 0.0])],
        )
        .unwrap();
        place_quad(r, 1);
        assert_eq!(r.mesh_stats().materials, 3, "round {round}");
        let c = rig.render(&p).at(W / 2, H / 2);
        assert!(c[0] > 150 && c[1] < 4, "round {round}: {c:?}");
    }
    rig.finish();
}

/// A Unity-style pass: `UnityPerDraw`, `UnityPerMaterial`, loose globals the renderer fills from the
/// frame, a texture with its sampler, and the vertex inputs a real URP shader asks for (tangent and
/// colour, which the meshes do not have).
const UNITY_HLSL: &str = r#"
cbuffer UnityPerDraw { float4x4 unity_ObjectToWorld; float4x4 unity_WorldToObject; float4 unity_WorldTransformParams; };
cbuffer UnityPerMaterial { float4 _BaseColor; float4 _BaseMap_ST; };
float4x4 unity_MatrixVP;
float4 _MainLightColor;
Texture2D _BaseMap;
SamplerState sampler_BaseMap;
struct A { float3 p : POSITION; float3 n : NORMAL; float2 uv : TEXCOORD0; float4 t : TANGENT; float4 c : COLOR; };
struct V { float4 pos : SV_Position; float2 uv : TEXCOORD0; float4 c : TEXCOORD1; float3 n : TEXCOORD2; };
V vert(A a) {
    V v;
    v.pos = mul(unity_MatrixVP, mul(unity_ObjectToWorld, float4(a.p, 1)));
    v.uv = a.uv * _BaseMap_ST.xy + _BaseMap_ST.zw;
    v.c = float4(a.c.xyz, a.t.x);
    v.n = a.n;
    return v;
}
float4 frag(V v) : SV_Target {
    float4 albedo = _BaseMap.Sample(sampler_BaseMap, v.uv) * _BaseColor;
    return float4(albedo.rgb * v.c.rgb * v.c.w * _MainLightColor.rgb, albedo.a);
}
"#;

fn compile_unity_pass() -> Option<(Vec<u32>, Vec<u32>)> {
    compile_hlsl(UNITY_HLSL)
}

fn compile_hlsl(source: &str) -> Option<(Vec<u32>, Vec<u32>)> {
    use rustiny_shader::{Compiler, Request, Stage, locate_dxc};
    let Some(dxc) = locate_dxc() else {
        eprintln!("SKIPPED: dxc not found");
        return None;
    };
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("rustiny-gpu-unity-{}-{n}", std::process::id()));
    std::fs::create_dir_all(dir.join("Packages")).unwrap();
    let compiler = Compiler::new(dxc, &dir, &dir);
    let stage = |entry: &str, stage| {
        compiler
            .compile(&Request {
                source,
                entry,
                stage,
                keywords: &[],
                include_dirs: &[],
            })
            .unwrap()
            .spirv
    };
    let result = (stage("vert", Stage::Vertex), stage("frag", Stage::Fragment));
    let _ = std::fs::remove_dir_all(dir);
    Some(result)
}

fn shader_surface(material: u32) -> Surface {
    Surface {
        shader: Some(material),
        ..Surface::default()
    }
}

#[test]
fn a_compiled_unity_shader_is_drawn_with_its_material_matrices_and_frame_values() {
    use rustiny_render::{
        PassState, PropertyValue, ShaderMaterialDesc, ShaderPassDesc, TextureFallback,
    };
    let Some((vertex, fragment)) = compile_unity_pass() else {
        return;
    };
    let Some(mut rig) = Rig::new() else { return };
    let eye = [0.0, 0.0, 0.0];
    let vp = view_proj(eye, 0.0);
    let (vertices, indices) = quad();
    let r = rig.renderer();
    r.add_texture(3, 2, 2, &quadrants(255), true).unwrap();
    let pass = r
        .add_shader_pass(&ShaderPassDesc {
            vertex,
            fragment,
            state: PassState::default(),
        })
        .unwrap();
    let property = |name: &str, value| (name.to_owned(), value);
    let textured_material = r
        .add_shader_material(&ShaderMaterialDesc {
            pass,
            properties: vec![
                property("_BaseColor", PropertyValue::Vector([1.0; 4])),
                property(
                    "_BaseMap",
                    PropertyValue::Texture {
                        key: Some(3),
                        scale_offset: [1.0, 1.0, 0.0, 0.0],
                        fallback: TextureFallback::White,
                    },
                ),
            ],
        })
        .unwrap();
    let green_material = r
        .add_shader_material(&ShaderMaterialDesc {
            pass,
            properties: vec![
                property("_BaseColor", PropertyValue::Vector([0.0, 1.0, 0.0, 1.0])),
                property(
                    "_BaseMap",
                    PropertyValue::Texture {
                        key: None,
                        scale_offset: [1.0, 1.0, 0.0, 0.0],
                        fallback: TextureFallback::White,
                    },
                ),
            ],
        })
        .unwrap();
    assert!(
        r.add_shader_material(&ShaderMaterialDesc {
            pass: 99,
            properties: Vec::new()
        })
        .is_err(),
        "a material needs an existing pass"
    );
    assert!(
        r.add_mesh(9, &vertices, &indices, &both(shader_surface(77)))
            .is_err(),
        "a section cannot name a shader material that does not exist"
    );
    r.add_mesh(
        1,
        &vertices,
        &indices,
        &both(shader_surface(textured_material)),
    )
    .unwrap();
    r.add_mesh(
        2,
        &vertices,
        &indices,
        &both(shader_surface(green_material)),
    )
    .unwrap();
    r.set_mesh_instances(&[
        MeshGroup {
            key: 1,
            instances: vec![GpuInstance::new(
                model(1.0, [0.0, 0.0, 5.0]),
                [1.0; 3],
                true,
            )],
        },
        MeshGroup {
            key: 2,
            instances: vec![
                // Twice the size, to the right: the model matrix reaches the vertex shader.
                GpuInstance::new(model(2.0, [6.0, 0.0, 10.0]), [1.0; 3], true),
                // Hidden, so not drawn.
                GpuInstance::new(model(1.0, [-6.0, 0.0, 5.0]), [1.0; 3], false),
            ],
        },
    ])
    .unwrap();
    let pixels = rig.render(&params(vp, eye));
    let stats = rig.renderer().mesh_stats();
    assert_eq!(
        (
            stats.shader_draws,
            stats.shader_skipped,
            stats.shader_materials
        ),
        (4, 0, 2),
        "two instances drawn, two sections each; the hidden one is skipped"
    );
    assert_eq!(stats.draw_calls, 4);

    let at = |wx: f32, wy: f32, wz: f32| {
        let (x, y) = screen(&vp, [wx, wy, wz]);
        pixels.at(x, y)
    };
    // The light colour (1, 0.96, 0.9) is multiplied in. The shader writes linear light and the
    // final pass encodes it as sRGB, so a fully lit channel reads 255, 249 or 243. The sRGB curve
    // also lifts the little that filtering bleeds in from the neighbouring texels.
    let red = at(-0.5, -0.5, 5.0);
    assert!(
        red[0] > 245 && red[1] < 60 && red[2] < 60,
        "the texture's red texel: {red:?}"
    );
    let green = at(0.5, -0.5, 5.0);
    assert!(
        (245..=252).contains(&green[1]) && green[0] < 60 && green[2] < 60,
        "green texel times the frame's light colour (0.96): {green:?}"
    );
    let blue = at(-0.5, 0.5, 5.0);
    assert!(
        (238..=248).contains(&blue[2]) && blue[0] < 60 && blue[1] < 60,
        "blue texel times 0.9: {blue:?}"
    );

    // Material B: green, white texture. The scale-2 instance centred at x = 6, z = 10 reaches
    // world x from 4 to 8 at that depth.
    let near_edge = at(7.5, 1.5, 10.0);
    assert!(
        near_edge[1] > 245 && near_edge[0] < 40 && near_edge[2] < 40,
        "the scaled instance covers its far corner: {near_edge:?}"
    );
    let outside = at(9.5, 0.0, 10.0);
    let close_to_clear = |p: [u8; 4]| (0..3).all(|c| p[c].abs_diff(CLEAR_BYTES[c]) <= 1);
    assert!(
        close_to_clear(outside),
        "nothing is drawn past the scaled instance: {outside:?}"
    );
    let hidden = at(-6.0, 0.0, 5.0);
    assert!(
        close_to_clear(hidden),
        "the hidden instance stays hidden: {hidden:?}"
    );
    rig.finish();
}

#[test]
fn blending_follows_the_pass_state_and_cleared_shaders_fall_back_to_the_native_look() {
    use rustiny_render::{
        Blend, BlendFactor, PassState, PropertyValue, ShaderMaterialDesc, ShaderPassDesc,
        TextureFallback,
    };
    let Some((vertex, fragment)) = compile_unity_pass() else {
        return;
    };
    let Some(mut rig) = Rig::new() else { return };
    let eye = [0.0, 0.0, 0.0];
    let vp = view_proj(eye, 0.0);
    let (vertices, indices) = quad();
    let r = rig.renderer();
    let pass = r
        .add_shader_pass(&ShaderPassDesc {
            vertex,
            fragment,
            state: PassState {
                blend: Some(Blend {
                    src_color: BlendFactor::SrcAlpha,
                    dst_color: BlendFactor::OneMinusSrcAlpha,
                    src_alpha: BlendFactor::One,
                    dst_alpha: BlendFactor::OneMinusSrcAlpha,
                }),
                z_write: false,
                ..PassState::default()
            },
        })
        .unwrap();
    let material = r
        .add_shader_material(&ShaderMaterialDesc {
            pass,
            properties: vec![
                (
                    "_BaseColor".to_owned(),
                    PropertyValue::Vector([0.0, 1.0, 0.0, 0.5]),
                ),
                (
                    "_BaseMap".to_owned(),
                    PropertyValue::Texture {
                        key: None,
                        scale_offset: [1.0, 1.0, 0.0, 0.0],
                        fallback: TextureFallback::White,
                    },
                ),
            ],
        })
        .unwrap();
    r.add_mesh(1, &vertices, &indices, &both(shader_surface(material)))
        .unwrap();
    place_quad(r, 1);
    let p = params(vp, eye);
    let (x, y) = screen(&vp, [0.0, 0.0, 5.0]);
    let blended = rig.render(&p).at(x, y);
    // Half of the lit green (0.96) over the clear colour (0.1, 0.2, 0.3 as shown), mixed in linear
    // light like URP does: linear (0.005, 0.496, 0.036), shown as sRGB (16, 187, 54).
    assert!(
        blended[1].abs_diff(187) <= 4
            && blended[0].abs_diff(16) <= 4
            && blended[2].abs_diff(54) <= 4,
        "alpha 0.5 blends with what is behind: {blended:?}"
    );

    rig.renderer().clear_shaders();
    let after = rig.render(&p);
    let fallback = after.at(x, y);
    assert!(
        !after.is_clear(x, y) && fallback[0] == fallback[1],
        "sections of cleared shaders fall back to the native grey look: {fallback:?}"
    );
    assert_eq!(rig.renderer().mesh_stats().shader_materials, 0);
    rig.finish();
}

/// Reads the camera textures: the colour drawn before it, channels rotated (blue out of red), and
/// whether the depth there holds something (green).
const CAMERA_HLSL: &str = r#"
cbuffer UnityPerDraw { float4x4 unity_ObjectToWorld; float4x4 unity_WorldToObject; float4 unity_WorldTransformParams; };
float4x4 unity_MatrixVP;
Texture2D _CameraOpaqueTexture;
Texture2D<float> _CameraDepthTexture;
struct A { float3 p : POSITION; };
struct V { float4 pos : SV_Position; };
V vert(A a) {
    V v;
    v.pos = mul(unity_MatrixVP, mul(unity_ObjectToWorld, float4(a.p, 1)));
    return v;
}
float4 frag(V v) : SV_Target {
    int3 c = int3(v.pos.xy, 0);
    float3 opaque = _CameraOpaqueTexture.Load(c).rgb;
    float depth = _CameraDepthTexture.Load(c);
    return float4(opaque.b, depth > 0 ? 1 : 0, opaque.r, 1);
}
"#;

#[test]
fn transparent_passes_read_the_colour_and_depth_drawn_before_them() {
    use rustiny_render::{Blend, BlendFactor, PassState, ShaderMaterialDesc, ShaderPassDesc};
    let Some((vertex, fragment)) = compile_hlsl(CAMERA_HLSL) else {
        return;
    };
    let Some(mut rig) = Rig::new() else { return };
    let eye = [0.0, 0.0, 0.0];
    let vp = view_proj(eye, 0.0);
    let (vertices, indices) = quad();
    let r = rig.renderer();
    // An opaque red quad, drawn natively, 5 units ahead.
    r.add_mesh(
        1,
        &vertices,
        &indices,
        &both(Surface {
            tint: [1.0, 0.0, 0.0],
            unlit: true,
            ..Surface::default()
        }),
    )
    .unwrap();
    let pass = r
        .add_shader_pass(&ShaderPassDesc {
            vertex,
            fragment,
            state: PassState {
                blend: Some(Blend {
                    src_color: BlendFactor::SrcAlpha,
                    dst_color: BlendFactor::OneMinusSrcAlpha,
                    src_alpha: BlendFactor::One,
                    dst_alpha: BlendFactor::OneMinusSrcAlpha,
                }),
                z_write: false,
                ..PassState::default()
            },
        })
        .unwrap();
    let material = r
        .add_shader_material(&ShaderMaterialDesc {
            pass,
            properties: Vec::new(),
        })
        .unwrap();
    // A transparent quad covering the whole view, nearer than the red one.
    r.add_mesh(2, &vertices, &indices, &both(shader_surface(material)))
        .unwrap();
    r.set_mesh_instances(&[
        MeshGroup {
            key: 1,
            instances: vec![GpuInstance::new(
                model(1.0, [0.0, 0.0, 5.0]),
                [1.0; 3],
                true,
            )],
        },
        MeshGroup {
            key: 2,
            instances: vec![GpuInstance::new(
                model(20.0, [0.0, 0.0, 3.0]),
                [1.0; 3],
                true,
            )],
        },
    ])
    .unwrap();
    let pixels = rig.render(&params(vp, eye));
    assert_eq!(rig.renderer().mesh_stats().shader_draws, 2);

    let (x, y) = screen(&vp, [0.0, 0.0, 5.0]);
    let over_red = pixels.at(x, y);
    assert!(
        over_red[2] > 240 && over_red[0] < 10 && over_red[1] > 240,
        "the red drawn first comes back as blue, and the depth there is set: {over_red:?}"
    );
    let corner = pixels.at(2, 2);
    // The background: the clear colour (0.1, 0.2, 0.3 as shown; linear 0.010, 0.033, 0.073)
    // with blue and red swapped, shown as sRGB, and no depth.
    assert!(
        corner[0].abs_diff(77) <= 3 && corner[1] == 0 && corner[2].abs_diff(26) <= 3,
        "the clear colour comes back rotated, and the far plane has no depth: {corner:?}"
    );
    rig.finish();
}

#[test]
fn per_object_shader_data_grows_with_the_scene_and_follows_moved_instances() {
    use rustiny_render::{
        PassState, PropertyValue, ShaderMaterialDesc, ShaderPassDesc, TextureFallback,
    };
    let Some((vertex, fragment)) = compile_unity_pass() else {
        return;
    };
    let Some(mut rig) = Rig::new() else { return };
    let eye = [0.0, 0.0, 0.0];
    let vp = view_proj(eye, 0.0);
    let (vertices, indices) = quad();
    let r = rig.renderer();
    let pass = r
        .add_shader_pass(&ShaderPassDesc {
            vertex,
            fragment,
            state: PassState::default(),
        })
        .unwrap();
    let material = r
        .add_shader_material(&ShaderMaterialDesc {
            pass,
            properties: vec![
                (
                    "_BaseColor".to_owned(),
                    PropertyValue::Vector([0.0, 1.0, 0.0, 1.0]),
                ),
                (
                    "_BaseMap".to_owned(),
                    PropertyValue::Texture {
                        key: None,
                        scale_offset: [1.0, 1.0, 0.0, 0.0],
                        fallback: TextureFallback::White,
                    },
                ),
            ],
        })
        .unwrap();
    // One section, so every instance is one draw.
    r.add_mesh(
        1,
        &vertices,
        &indices,
        &[MeshSection {
            first_index: 0,
            index_count: 6,
            surface: shader_surface(material),
        }],
    )
    .unwrap();
    // More instances than the per-object buffer first holds: 16 MB, and this pass's block is
    // 144 bytes (at least 192 with the usual offset alignment), so 120 000 need about 23 MB.
    const COUNT: usize = 120_000;
    let at = |x: f32| GpuInstance::new(model(1.0, [x, 0.0, 5.0]), [1.0; 3], true);
    r.set_mesh_instances(&[MeshGroup {
        key: 1,
        instances: vec![at(-2.0); COUNT],
    }])
    .unwrap();
    for frame in 0..2 {
        let pixels = rig.render(&params(vp, eye));
        let stats = rig.renderer().mesh_stats();
        assert_eq!(
            (stats.shader_draws, stats.shader_skipped),
            (COUNT as u32, 0),
            "frame {frame}: every instance is drawn, none left out"
        );
        let (x, y) = screen(&vp, [-2.0, 0.0, 5.0]);
        assert!(pixels.at(x, y)[1] > 200, "frame {frame}: green at x = -2");
    }
    // Moving the instances writes their data again.
    rig.renderer()
        .set_mesh_instances(&[MeshGroup {
            key: 1,
            instances: vec![at(2.0); 10],
        }])
        .unwrap();
    let pixels = rig.render(&params(vp, eye));
    let (old_x, old_y) = screen(&vp, [-2.0, 0.0, 5.0]);
    let (new_x, new_y) = screen(&vp, [2.0, 0.0, 5.0]);
    assert!(
        pixels.is_clear(old_x, old_y),
        "nothing left at the old place"
    );
    assert!(pixels.at(new_x, new_y)[1] > 200, "green at the new place");
    rig.finish();
}

/// URP's own functions (Core RP `Common.hlsl`, URP `ShaderVariablesFunctions.hlsl`), as compiled
/// for Vulkan (`UNITY_UV_STARTS_AT_TOP`, reversed Z): the screen position of a vertex and the world
/// position of the depth found there. Writes the world position of what lies behind.
const RECONSTRUCT_HLSL: &str = r#"
#define UNITY_UV_STARTS_AT_TOP 1
cbuffer UnityPerDraw { float4x4 unity_ObjectToWorld; float4x4 unity_WorldToObject; float4 unity_WorldTransformParams; };
float4x4 unity_MatrixVP;
float4x4 unity_MatrixInvVP;
float4 _ProjectionParams;
float4 _ScreenParams;
Texture2D<float> _CameraDepthTexture;
float4 ComputeScreenPos(float4 positionCS) {
    float4 o = positionCS * 0.5f;
    o.xy = float2(o.x, o.y * _ProjectionParams.x) + o.w;
    o.zw = positionCS.zw;
    return o;
}
float4 ComputeClipSpacePosition(float2 positionNDC, float deviceDepth) {
    float4 positionCS = float4(positionNDC * 2.0 - 1.0, deviceDepth, 1.0);
#if UNITY_UV_STARTS_AT_TOP
    positionCS.y = -positionCS.y;
#endif
    return positionCS;
}
float3 ComputeWorldSpacePosition(float2 positionNDC, float deviceDepth, float4x4 invViewProjMatrix) {
    float4 positionCS = ComputeClipSpacePosition(positionNDC, deviceDepth);
    float4 hpositionWS = mul(invViewProjMatrix, positionCS);
    return hpositionWS.xyz / hpositionWS.w;
}
struct A { float3 p : POSITION; };
struct V { float4 pos : SV_Position; float4 screen : TEXCOORD0; };
V vert(A a) {
    V v;
    v.pos = mul(unity_MatrixVP, mul(unity_ObjectToWorld, float4(a.p, 1)));
    v.screen = ComputeScreenPos(v.pos);
    return v;
}
float4 frag(V v) : SV_Target {
    float2 uv = v.screen.xy / v.screen.w;
    float depth = _CameraDepthTexture.Load(int3(uv * _ScreenParams.xy, 0));
    float3 world = ComputeWorldSpacePosition(uv, depth, unity_MatrixInvVP);
    return float4(saturate(world.z / 10.0), saturate(world.y + 0.5), saturate(world.x + 0.5), 1);
}
"#;

#[test]
fn urp_finds_the_world_position_of_what_is_behind_from_the_camera_depth() {
    use rustiny_render::{
        Blend, BlendFactor, PassState, ShaderMaterialDesc, ShaderPassDesc, UnityFrame,
    };
    let Some((vertex, fragment)) = compile_hlsl(RECONSTRUCT_HLSL) else {
        return;
    };
    let Some(mut rig) = Rig::new() else { return };
    let eye = [0.0, 0.0, 0.0];
    let vp = view_proj(eye, 0.0);
    let (vertices, indices) = quad();
    let r = rig.renderer();
    r.set_unity_frame(UnityFrame {
        width: W,
        height: H,
        ..UnityFrame::default()
    });
    r.add_mesh(1, &vertices, &indices, &both(Surface::default()))
        .unwrap();
    let pass = r
        .add_shader_pass(&ShaderPassDesc {
            vertex,
            fragment,
            state: PassState {
                blend: Some(Blend {
                    src_color: BlendFactor::One,
                    dst_color: BlendFactor::Zero,
                    src_alpha: BlendFactor::One,
                    dst_alpha: BlendFactor::Zero,
                }),
                z_write: false,
                ..PassState::default()
            },
        })
        .unwrap();
    let material = r
        .add_shader_material(&ShaderMaterialDesc {
            pass,
            properties: Vec::new(),
        })
        .unwrap();
    r.add_mesh(2, &vertices, &indices, &both(shader_surface(material)))
        .unwrap();
    r.set_mesh_instances(&[
        MeshGroup {
            key: 1,
            instances: vec![GpuInstance::new(
                model(1.0, [0.0, 0.0, 5.0]),
                [1.0; 3],
                true,
            )],
        },
        MeshGroup {
            key: 2,
            instances: vec![GpuInstance::new(
                model(20.0, [0.0, 0.0, 3.0]),
                [1.0; 3],
                true,
            )],
        },
    ])
    .unwrap();
    let pixels = rig.render(&params(vp, eye));
    // The opaque quad at z = 5 spans x and y from -1 to 1. Shown as sRGB: z / 10 = 0.5 -> 188;
    // y + 0.5 = 1 (top) -> 255, 0 (bottom) -> 0.
    for (wx, wy) in [(-0.5f32, 0.5f32), (0.5, -0.5), (0.5, 0.5), (-0.5, -0.5)] {
        let (x, y) = screen(&vp, [wx, wy, 5.0]);
        let p = pixels.at(x, y);
        let srgb = |v: f32| {
            let v = v.clamp(0.0, 1.0);
            let s = if v <= 0.003_130_8 {
                v * 12.92
            } else {
                1.055 * v.powf(1.0 / 2.4) - 0.055
            };
            (s * 255.0).round() as u8
        };
        let want = [srgb(0.5), srgb(wy + 0.5), srgb(wx + 0.5)];
        assert!(
            (0..3).all(|c| p[c].abs_diff(want[c]) <= 6),
            "world ({wx}, {wy}, 5) is found again: got {p:?}, want {want:?}"
        );
    }
    rig.finish();
}

/// Core RP's `NeutralTonemap`, for checking the final pass.
fn neutral_tonemap(x: f32) -> f32 {
    let (a, b, c, d, e, f) = (0.2f32, 0.29, 0.24, 0.272, 0.02, 0.3);
    let curve = |x: f32| ((x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f)) - e / f;
    let white_scale = 1.0 / curve(5.3);
    curve(x * white_scale) * white_scale
}

fn to_srgb_byte(v: f32) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let s = if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0).round() as u8
}

/// An unlit quad whose colour is `value` in every channel (emission adds to the black tint).
fn glowing_quad(r: &mut SceneRenderer, key: u32, value: f32, at: [f32; 3], scale: f32) {
    let (vertices, indices) = quad();
    r.add_mesh(
        key,
        &vertices,
        &indices,
        &both(Surface {
            tint: [0.0; 3],
            emission: [value; 3],
            unlit: true,
            ..Surface::default()
        }),
    )
    .unwrap();
    r.set_mesh_instances(&[MeshGroup {
        key,
        instances: vec![GpuInstance::new(model(scale, at), [1.0; 3], true)],
    }])
    .unwrap();
}

#[test]
fn neutral_tonemapping_and_exposure_follow_urp() {
    use rustiny_render::{PostProcess, Tonemap};
    let Some(mut rig) = Rig::new() else { return };
    let eye = [0.0, 0.0, 0.0];
    let vp = view_proj(eye, 0.0);
    glowing_quad(rig.renderer(), 1, 0.8, [0.0, 0.0, 5.0], 1.0);
    let (x, y) = screen(&vp, [0.0, 0.0, 5.0]);
    let plain = rig.render(&params(vp, eye)).at(x, y);
    assert!(
        plain[0].abs_diff(to_srgb_byte(0.8)) <= 2,
        "no tonemapping: {plain:?}"
    );

    rig.renderer().set_post_process(PostProcess {
        tonemap: Tonemap::Neutral,
        post_exposure: 1.0,
        ..PostProcess::default()
    });
    let mapped = rig.render(&params(vp, eye)).at(x, y);
    // One stop of exposure doubles 0.8 to 1.6, which the Neutral curve brings below 1.
    let want = to_srgb_byte(neutral_tonemap(1.6));
    assert!(
        mapped[0].abs_diff(want) <= 2 && mapped[1] == mapped[0],
        "Neutral(1.6) = {want}: {mapped:?}"
    );
    rig.finish();
}

#[test]
fn the_vignette_darkens_the_corners_only() {
    use rustiny_render::{PostProcess, Vignette};
    let Some(mut rig) = Rig::new() else { return };
    let eye = [0.0, 0.0, 0.0];
    let vp = view_proj(eye, 0.0);
    glowing_quad(rig.renderer(), 1, 0.5, [0.0, 0.0, 5.0], 20.0);
    rig.renderer().set_post_process(PostProcess {
        vignette: Vignette {
            intensity: 0.4,
            ..Vignette::default()
        },
        ..PostProcess::default()
    });
    let pixels = rig.render(&params(vp, eye));
    let centre = pixels.at(W / 2, H / 2);
    let corner = pixels.at(1, 1);
    assert!(
        centre[0].abs_diff(to_srgb_byte(0.5)) <= 2,
        "centre untouched: {centre:?}"
    );
    assert!(
        corner[0] + 20 < centre[0],
        "corner darker: {corner:?} vs {centre:?}"
    );
    rig.finish();
}

#[test]
fn bloom_spreads_pixels_above_the_threshold_and_fades_with_distance() {
    use rustiny_render::{Bloom, PostProcess};
    let Some(mut rig) = Rig::new() else { return };
    let eye = [0.0, 0.0, 0.0];
    let vp = view_proj(eye, 0.0);
    let bloom = PostProcess {
        bloom: Some(Bloom {
            // 3.8 in gamma space is about 18.9 in linear light.
            threshold: 3.8,
            intensity: 2.2,
            scatter: 0.6,
            high_quality: true,
            skip_iterations: 0,
            ..Bloom::default()
        }),
        ..PostProcess::default()
    };
    let (x, y) = screen(&vp, [0.0, 0.0, 5.0]);
    let (near, far) = ((x + 12, y), (x + 40, y));

    // Below the threshold (and its knee) nothing spreads.
    glowing_quad(rig.renderer(), 1, 8.0, [0.0, 0.0, 5.0], 0.2);
    rig.renderer().set_post_process(bloom);
    let dim = rig.render(&params(vp, eye));
    assert!(
        dim.is_clear(near.0, near.1),
        "below the threshold: {:?}",
        dim.at(near.0, near.1)
    );

    // Far above it, the quad glows onto its surroundings, less with distance.
    glowing_quad(rig.renderer(), 2, 200.0, [0.0, 0.0, 5.0], 0.2);
    let bright = rig.render(&params(vp, eye));
    let (a, b) = (bright.at(near.0, near.1), bright.at(far.0, far.1));
    assert!(
        (0..3).all(|c| a[c] > CLEAR_BYTES[c] + 30),
        "glow next to the quad: {a:?}"
    );
    assert!(
        (0..3).all(|c| a[c] > b[c]),
        "less glow further away: {a:?} vs {b:?}"
    );
    rig.renderer().set_post_process(PostProcess::default());
    let off = rig.render(&params(vp, eye));
    assert!(off.is_clear(near.0, near.1), "no glow without bloom");
    rig.finish();
}
