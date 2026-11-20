// Scene pass shaders: instanced boxes and a procedural ground grid.
//
// Conventions: Unity's left-handed, Y-up world; reverse-Z infinite far plane (depth 1 at the near
// plane, 0 at infinity, compare GREATER); the view-projection matrix already produces Vulkan
// clip space (Y down). Output is linear light: the target is HDR and the final pass converts it.

struct Push {
    view_proj: mat4x4<f32>,
    // xyz: camera position.
    eye: vec4<f32>,
    // x: selected object id (negative = none), y: grid half extent,
    // z: grid cell size, w: unused.
    params: vec4<f32>,
    // x: index of the section's record in `materials`; the rest is padding.
    material: vec4<u32>,
};
// naga 30 calls push constants "immediates"; the SPIR-V backend emits them as PushConstant.
var<immediate> push: Push;

// One section's surface, written by the renderer (see material.rs).
struct Material {
    // rgb: tint (linear), w: alpha cutoff (0 = opaque).
    a: vec4<f32>,
    // rgb: emission (linear), w: normal map strength.
    b: vec4<f32>,
    // x: metallic, y: smoothness, z: flags (1 = unlit, 2 = has a metallic-smoothness texture).
    c: vec4<f32>,
};
@group(1) @binding(0) var<storage, read> materials: array<Material>;

// Textures of the section being drawn; each is a 1x1 default (white, flat normal, no metal and full
// smoothness) when the section has none.
@group(0) @binding(0) var albedo_tex: texture_2d<f32>;
@group(0) @binding(1) var normal_tex: texture_2d<f32>;
@group(0) @binding(2) var mask_tex: texture_2d<f32>;
@group(0) @binding(3) var material_sampler: sampler;

struct MeshIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    // Per-instance model matrix columns.
    @location(3) m0: vec4<f32>,
    @location(4) m1: vec4<f32>,
    @location(5) m2: vec4<f32>,
    @location(6) m3: vec4<f32>,
    // rgb: instance colour, a: 1 = draw, 0 = hidden unless selected.
    @location(7) color: vec4<f32>,
    // Scene object the instance stands for.
    @location(8) object: u32,
};

struct MeshOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) selected: f32,
    @location(3) uv: vec2<f32>,
    @location(4) world: vec3<f32>,
};

@vertex
fn mesh_vs(in: MeshIn) -> MeshOut {
    var out: MeshOut;
    let is_selected = push.params.x >= 0.0 && u32(push.params.x) == in.object;
    if (in.color.a < 0.5 && !is_selected) {
        // Outside the clip volume: the triangle is discarded.
        out.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.normal = vec3<f32>(0.0, 1.0, 0.0);
        out.color = vec3<f32>(0.0);
        out.selected = 0.0;
        out.uv = vec2<f32>(0.0);
        out.world = vec3<f32>(0.0);
        return out;
    }
    let model = mat4x4<f32>(in.m0, in.m1, in.m2, in.m3);
    let world = model * vec4<f32>(in.position, 1.0);
    out.clip = push.view_proj * world;
    out.normal = normalize((model * vec4<f32>(in.normal, 0.0)).xyz);
    out.color = in.color.rgb;
    out.selected = select(0.0, 1.0, is_selected);
    out.uv = in.uv;
    out.world = world.xyz;
    return out;
}

// The sun, and how much light comes from everywhere else.
const LIGHT: vec3<f32> = vec3<f32>(0.4, 0.8, -0.35);
const AMBIENT: f32 = 0.35;

@fragment
fn mesh_fs(in: MeshOut) -> @location(0) vec4<f32> {
    let m = materials[push.material.x];
    // Everything that needs derivatives or implicit LODs happens in uniform control flow, before
    // the alpha test.
    let texel = textureSample(albedo_tex, material_sampler, in.uv);
    let packed_normal = textureSample(normal_tex, material_sampler, in.uv).xyz;
    let mask = textureSample(mask_tex, material_sampler, in.uv);
    let dp1 = dpdx(in.world);
    let dp2 = dpdy(in.world);
    let duv1 = dpdx(in.uv);
    let duv2 = dpdy(in.uv);
    if (m.a.w > 0.0 && texel.a < m.a.w) {
        discard;
    }

    let flags = u32(m.c.z + 0.5);
    let unlit = (flags & 1u) != 0u;
    let has_mask = (flags & 2u) != 0u;
    let albedo = in.color * m.a.rgb * texel.rgb;

    // Faces are lit on the side that faces the camera (leaves, cloth and other one-sided meshes
    // are seen from both sides).
    let to_eye = normalize(push.eye.xyz - in.world);
    var n = normalize(in.normal);
    if (dot(n, to_eye) < 0.0) {
        n = -n;
    }
    // Normal map: tangent space from the screen-space derivatives of position and UV, so meshes
    // do not need tangents. V is up in both the UVs and the stored normals.
    let tangent_normal = vec3<f32>(
        (packed_normal.xy * 2.0 - 1.0) * m.b.w,
        packed_normal.z * 2.0 - 1.0,
    );
    let dp2perp = cross(dp2, n);
    let dp1perp = cross(n, dp1);
    let t = dp2perp * duv1.x + dp1perp * duv2.x;
    let b = dp2perp * duv1.y + dp1perp * duv2.y;
    let scale = inverseSqrt(max(max(dot(t, t), dot(b, b)), 1e-20));
    var shading_normal = n;
    if (dot(t, t) + dot(b, b) > 1e-20) {
        shading_normal = normalize(t * scale * tangent_normal.x + b * scale * tangent_normal.y + n * tangent_normal.z);
    }

    var lit: vec3<f32>;
    if (unlit) {
        lit = albedo;
    } else {
        var metallic = m.c.x;
        var smoothness = m.c.y;
        if (has_mask) {
            metallic = mask.r;
            smoothness = mask.a * m.c.y;
        }
        let light = normalize(LIGHT);
        let lambert = max(dot(shading_normal, light), 0.0);
        let half_vector = normalize(light + to_eye);
        let n_dot_h = max(dot(shading_normal, half_vector), 0.0);
        // Blinn-Phong lobe, wider the rougher the surface; zero for a matte (smoothness 0) one.
        let shininess = mix(8.0, 256.0, smoothness * smoothness);
        let specular_color = mix(vec3<f32>(0.04), albedo, metallic);
        let lobe = pow(n_dot_h, shininess) * (shininess + 8.0) / 25.1 * lambert;
        let gloss = smoothness * smoothness;
        let diffuse = albedo * (1.0 - 0.75 * metallic) * (AMBIENT + 0.65 * lambert);
        // Without an environment map, metals reflect a flat ambient sky instead.
        let reflected = specular_color * (AMBIENT * 2.0 * gloss + lobe * gloss);
        lit = diffuse + reflected;
    }
    lit = lit + m.b.rgb;
    // The selection yellow (1, 0.92, 0.25 as shown) in linear light.
    lit = mix(lit, vec3<f32>(1.0, 0.832, 0.047) * (0.6 + 0.4 * max(dot(n, normalize(LIGHT)), 0.0)), in.selected);
    return vec4<f32>(max(lit, vec3<f32>(0.0)), 1.0);
}

struct GridOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
};

@vertex
fn grid_vs(@builtin(vertex_index) index: u32) -> GridOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let corner = corners[index];
    let extent = push.params.y;
    let world = vec3<f32>(push.eye.x + corner.x * extent, 0.0, push.eye.z + corner.y * extent);
    var out: GridOut;
    out.clip = push.view_proj * vec4<f32>(world, 1.0);
    out.world = world;
    return out;
}

@fragment
fn grid_fs(in: GridOut) -> @location(0) vec4<f32> {
    let cell = push.params.z;
    let coord = in.world.xz / cell;
    let width = max(fwidth(coord), vec2<f32>(1e-4));
    let distance_to_line = abs(fract(coord - 0.5) - 0.5) / width;
    let line = 1.0 - min(min(distance_to_line.x, distance_to_line.y), 1.0);

    // World axes: X red, Z blue, a little thicker than the grid lines.
    let axis_width = width * cell * 1.5;
    let on_x_axis = 1.0 - min(abs(in.world.z) / max(axis_width.y, 1e-4), 1.0);
    let on_z_axis = 1.0 - min(abs(in.world.x) / max(axis_width.x, 1e-4), 1.0);

    let fade = 1.0 - smoothstep(0.35 * push.params.y, push.params.y, length(in.world.xz - push.eye.xz));
    var colour = vec3<f32>(1.0);
    var alpha = line * 0.16;
    colour = mix(colour, vec3<f32>(0.95, 0.3, 0.3), on_x_axis);
    colour = mix(colour, vec3<f32>(0.35, 0.55, 1.0), on_z_axis);
    alpha = max(alpha, max(on_x_axis, on_z_axis) * 0.55);
    // The overlay colours are chosen as they should look: decode them to the linear target.
    return vec4<f32>(pow(colour, vec3<f32>(2.2)), alpha * fade);
}
