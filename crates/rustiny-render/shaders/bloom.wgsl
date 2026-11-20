// URP's bloom (Bloom.shader of the Universal Render Pipeline), pass by pass:
//
// * prefilter: the HDR scene at half size, clamped, with the soft threshold applied (13 taps
//   with high-quality filtering, else one);
// * blur_h: a 9-tap gaussian along x, writing the next (half as big) level;
// * blur_v: an optimised 5-tap bilinear gaussian along y, same size;
// * upsample: the level below (bicubic with high-quality filtering) mixed into this level by
//   the scatter amount.

struct Push {
    // xy: 1 / source size, zw: source size.
    source: vec4<f32>,
    // xy: 1 / lower level size, zw: lower level size.
    low: vec4<f32>,
    // x: scatter, y: clamp, z: threshold, w: threshold knee.
    params: vec4<f32>,
    // x: 1 for high-quality filtering.
    flags: vec4<f32>,
};
var<immediate> push: Push;

@group(0) @binding(0) var source_tex: texture_2d<f32>;
@group(0) @binding(1) var low_tex: texture_2d<f32>;
@group(0) @binding(2) var linear_clamp: sampler;

struct Out {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn bloom_vs(@builtin(vertex_index) index: u32) -> Out {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    var out: Out;
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    // Vulkan's clip space has y down, so texture v follows clip y directly.
    out.uv = uv;
    return out;
}

fn sample_source(uv: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(source_tex, linear_clamp, uv, 0.0).rgb;
}

fn sample_low(uv: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(low_tex, linear_clamp, uv, 0.0).rgb;
}

@fragment
fn prefilter_fs(in: Out) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let t = push.source.xy;
    var color: vec3<f32>;
    if (push.flags.x > 0.5) {
        let a = sample_source(uv + t * vec2<f32>(-1.0, -1.0));
        let b = sample_source(uv + t * vec2<f32>(0.0, -1.0));
        let c = sample_source(uv + t * vec2<f32>(1.0, -1.0));
        let d = sample_source(uv + t * vec2<f32>(-0.5, -0.5));
        let e = sample_source(uv + t * vec2<f32>(0.5, -0.5));
        let f = sample_source(uv + t * vec2<f32>(-1.0, 0.0));
        let g = sample_source(uv);
        let h = sample_source(uv + t * vec2<f32>(1.0, 0.0));
        let i = sample_source(uv + t * vec2<f32>(-0.5, 0.5));
        let j = sample_source(uv + t * vec2<f32>(0.5, 0.5));
        let k = sample_source(uv + t * vec2<f32>(-1.0, 1.0));
        let l = sample_source(uv + t * vec2<f32>(0.0, 1.0));
        let m = sample_source(uv + t * vec2<f32>(1.0, 1.0));
        let div = (1.0 / 4.0) * vec2<f32>(0.5, 0.125);
        color = (d + e + i + j) * div.x;
        color = color + (a + b + g + f) * div.y;
        color = color + (b + c + h + g) * div.y;
        color = color + (f + g + l + k) * div.y;
        color = color + (g + h + m + l) * div.y;
    } else {
        color = sample_source(uv);
    }
    // User-controlled clamp to limit crazy high broken specular values.
    color = min(vec3<f32>(push.params.y), color);
    // Thresholding with a soft knee.
    let threshold = push.params.z;
    let knee = push.params.w;
    let brightness = max(max(color.r, color.g), color.b);
    var softness = clamp(brightness - threshold + knee, 0.0, 2.0 * knee);
    softness = (softness * softness) / (4.0 * knee + 1e-4);
    let multiplier = max(brightness - threshold, softness) / max(brightness, 1e-4);
    color = max(color * multiplier, vec3<f32>(0.0));
    return vec4<f32>(color, 1.0);
}

@fragment
fn blur_h_fs(in: Out) -> @location(0) vec4<f32> {
    let texel = push.source.x * 2.0;
    let uv = in.uv;
    let c0 = sample_source(uv - vec2<f32>(texel * 4.0, 0.0));
    let c1 = sample_source(uv - vec2<f32>(texel * 3.0, 0.0));
    let c2 = sample_source(uv - vec2<f32>(texel * 2.0, 0.0));
    let c3 = sample_source(uv - vec2<f32>(texel * 1.0, 0.0));
    let c4 = sample_source(uv);
    let c5 = sample_source(uv + vec2<f32>(texel * 1.0, 0.0));
    let c6 = sample_source(uv + vec2<f32>(texel * 2.0, 0.0));
    let c7 = sample_source(uv + vec2<f32>(texel * 3.0, 0.0));
    let c8 = sample_source(uv + vec2<f32>(texel * 4.0, 0.0));
    let color = c0 * 0.01621622 + c1 * 0.05405405 + c2 * 0.12162162 + c3 * 0.19459459
        + c4 * 0.22702703
        + c5 * 0.19459459 + c6 * 0.12162162 + c7 * 0.05405405 + c8 * 0.01621622;
    return vec4<f32>(color, 1.0);
}

@fragment
fn blur_v_fs(in: Out) -> @location(0) vec4<f32> {
    let texel = push.source.y;
    let uv = in.uv;
    let c0 = sample_source(uv - vec2<f32>(0.0, texel * 3.23076923));
    let c1 = sample_source(uv - vec2<f32>(0.0, texel * 1.38461538));
    let c2 = sample_source(uv);
    let c3 = sample_source(uv + vec2<f32>(0.0, texel * 1.38461538));
    let c4 = sample_source(uv + vec2<f32>(0.0, texel * 3.23076923));
    let color = c0 * 0.07027027 + c1 * 0.31621622 + c2 * 0.22702703 + c3 * 0.31621622
        + c4 * 0.07027027;
    return vec4<f32>(color, 1.0);
}

// Core RP's B-spline bicubic filter as four bilinear taps (Filtering.hlsl).
fn bspline_rightmost(x: vec2<f32>) -> vec2<f32> {
    return 0.16666667 + x * (-0.5 + x * (0.5 - x * 0.16666667));
}

fn bspline_middle_right(x: vec2<f32>) -> vec2<f32> {
    return 0.66666667 + x * (-1.0 + 0.5 * x) * x;
}

fn bspline_middle_left(x: vec2<f32>) -> vec2<f32> {
    return 0.16666667 + x * (0.5 + x * (0.5 - x * 0.5));
}

fn sample_low_bicubic(uv: vec2<f32>) -> vec3<f32> {
    let size = push.low.zw;
    let texel = push.low.xy;
    let xy = uv * size + 0.5;
    let ic = floor(xy);
    let fc = fract(xy);
    let r = bspline_rightmost(fc);
    let mr = bspline_middle_right(fc);
    let ml = bspline_middle_left(fc);
    let l = 1.0 - mr - ml - r;
    let w0 = r + mr;
    let w1 = ml + l;
    let o0 = -1.0 + mr / w0;
    let o1 = 1.0 + l / w1;
    let s00 = sample_low((ic + vec2<f32>(o0.x, o0.y) - 0.5) * texel);
    let s10 = sample_low((ic + vec2<f32>(o1.x, o0.y) - 0.5) * texel);
    let s01 = sample_low((ic + vec2<f32>(o0.x, o1.y) - 0.5) * texel);
    let s11 = sample_low((ic + vec2<f32>(o1.x, o1.y) - 0.5) * texel);
    return w0.y * (w0.x * s00 + w1.x * s10) + w1.y * (w0.x * s01 + w1.x * s11);
}

@fragment
fn upsample_fs(in: Out) -> @location(0) vec4<f32> {
    let high = sample_source(in.uv);
    var low: vec3<f32>;
    if (push.flags.x > 0.5) {
        low = sample_low_bicubic(in.uv);
    } else {
        low = sample_low(in.uv);
    }
    return vec4<f32>(mix(high, low, push.params.x), 1.0);
}
