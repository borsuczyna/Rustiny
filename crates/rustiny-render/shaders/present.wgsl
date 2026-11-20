// Final pass of the scene, URP's "uber" post-processing pass: the HDR scene image (linear light)
// gets the bloom added, the vignette, the post exposure, the colour grading and the tonemapping,
// in that order, and is written to the display image as sRGB.

struct Push {
    // x: post exposure (multiplier), y: contrast (1 = none), z: hue shift (turns),
    // w: saturation (1 = none).
    grading: vec4<f32>,
    // rgb: colour filter (linear), w: tonemapping (0 none, 1 neutral).
    color_filter: vec4<f32>,
    // rgb: vignette colour, w: roundness.
    vignette_color: vec4<f32>,
    // xy: vignette centre, z: intensity (x3 already), w: smoothness (x5 already).
    vignette: vec4<f32>,
    // rgb: bloom tint times intensity, w: 1 when there is bloom.
    bloom: vec4<f32>,
    // xy: 1 / bloom texture size, zw: bloom texture size; then x: 1 for bicubic bloom sampling.
    bloom_size: vec4<f32>,
    flags: vec4<f32>,
};
var<immediate> push: Push;

@group(0) @binding(0) var scene_hdr: texture_2d<f32>;
@group(0) @binding(1) var bloom_tex: texture_2d<f32>;
@group(0) @binding(2) var linear_clamp: sampler;

struct PresentOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// One triangle that covers the whole target.
@vertex
fn present_vs(@builtin(vertex_index) index: u32) -> PresentOut {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    var out: PresentOut;
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = uv;
    return out;
}

// The exact sRGB transfer function (what an sRGB render target does in hardware).
fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let low = c * 12.92;
    let high = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(high, low, c <= vec3<f32>(0.0031308));
}

// ALEXA LogC (EI 1000), as Unity's colour grading uses it for contrast.
const LOGC_CUT: f32 = 0.011361;
const LOGC_A: f32 = 5.555556;
const LOGC_B: f32 = 0.047996;
const LOGC_C: f32 = 0.244161;
const LOGC_D: f32 = 0.386036;
const LOGC_E: f32 = 5.301883;
const LOGC_F: f32 = 0.092819;
const ACESCC_MIDGRAY: f32 = 0.4135884;

fn linear_to_logc(x: vec3<f32>) -> vec3<f32> {
    let log_part = LOGC_C * log2(LOGC_A * x + LOGC_B) / log2(10.0) + LOGC_D;
    return select(LOGC_E * x + LOGC_F, log_part, x > vec3<f32>(LOGC_CUT));
}

fn logc_to_linear(x: vec3<f32>) -> vec3<f32> {
    let exp_part = (pow(vec3<f32>(10.0), (x - LOGC_D) / LOGC_C) - LOGC_B) / LOGC_A;
    return select((x - LOGC_F) / LOGC_E, exp_part, x > vec3<f32>(LOGC_E * LOGC_CUT + LOGC_F));
}

fn rgb_to_hsv(c: vec3<f32>) -> vec3<f32> {
    let k = vec4<f32>(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    let p = select(vec4<f32>(c.gb, k.xy), vec4<f32>(c.bg, k.wz), c.g < c.b);
    let q = select(vec4<f32>(c.r, p.yzx), vec4<f32>(p.xyw, c.r), c.r < p.x);
    let d = q.x - min(q.w, q.y);
    let e = 1.0e-4;
    return vec3<f32>(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

fn hsv_to_rgb(c: vec3<f32>) -> vec3<f32> {
    let k = vec4<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    let p = abs(fract(c.xxx + k.xyz) * 6.0 - k.www);
    return c.z * mix(k.xxx, clamp(p - k.xxx, vec3<f32>(0.0), vec3<f32>(1.0)), c.y);
}

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126729, 0.7151522, 0.0721750));
}

// URP's colour grading for the parameters a ColorAdjustments override sets (LutBuilderHdr).
fn grade(color_in: vec3<f32>) -> vec3<f32> {
    var color = color_in;
    let contrast = push.grading.y;
    if (contrast != 1.0) {
        let log_color = (linear_to_logc(color) - ACESCC_MIDGRAY) * contrast + ACESCC_MIDGRAY;
        color = logc_to_linear(log_color);
    }
    color = max(color * push.color_filter.rgb, vec3<f32>(0.0));
    let hue_shift = push.grading.z;
    if (hue_shift != 0.0) {
        var hsv = rgb_to_hsv(color);
        hsv.x = fract(hsv.x + hue_shift);
        color = hsv_to_rgb(hsv);
    }
    let luma = luminance(color);
    return vec3<f32>(luma) + push.grading.w * (color - vec3<f32>(luma));
}

// Core RP's NeutralTonemap (Color.hlsl).
fn neutral_curve(x: vec3<f32>, a: f32, b: f32, c: f32, d: f32, e: f32, f: f32) -> vec3<f32> {
    return ((x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f)) - e / f;
}

fn neutral_tonemap(x_in: vec3<f32>) -> vec3<f32> {
    let a = 0.2;
    let b = 0.29;
    let c = 0.24;
    let d = 0.272;
    let e = 0.02;
    let f = 0.3;
    let white_level = 5.3;
    let white_clip = 1.0;
    let white_scale = vec3<f32>(1.0) / neutral_curve(vec3<f32>(white_level), a, b, c, d, e, f);
    var x = neutral_curve(x_in * white_scale, a, b, c, d, e, f);
    x = x * white_scale;
    return x / white_clip;
}

fn sample_bloom(uv: vec2<f32>) -> vec3<f32> {
    if (push.flags.x < 0.5) {
        return textureSampleLevel(bloom_tex, linear_clamp, uv, 0.0).rgb;
    }
    // Bicubic B-spline as four bilinear taps, as URP samples bloom with high-quality filtering.
    let size = push.bloom_size.zw;
    let texel = push.bloom_size.xy;
    let xy = uv * size + 0.5;
    let ic = floor(xy);
    let fc = fract(xy);
    let r = 0.16666667 + fc * (-0.5 + fc * (0.5 - fc * 0.16666667));
    let mr = 0.66666667 + fc * (-1.0 + 0.5 * fc) * fc;
    let ml = 0.16666667 + fc * (0.5 + fc * (0.5 - fc * 0.5));
    let l = 1.0 - mr - ml - r;
    let w0 = r + mr;
    let w1 = ml + l;
    let o0 = -1.0 + mr / w0;
    let o1 = 1.0 + l / w1;
    let s00 = textureSampleLevel(bloom_tex, linear_clamp, (ic + vec2<f32>(o0.x, o0.y) - 0.5) * texel, 0.0).rgb;
    let s10 = textureSampleLevel(bloom_tex, linear_clamp, (ic + vec2<f32>(o1.x, o0.y) - 0.5) * texel, 0.0).rgb;
    let s01 = textureSampleLevel(bloom_tex, linear_clamp, (ic + vec2<f32>(o0.x, o1.y) - 0.5) * texel, 0.0).rgb;
    let s11 = textureSampleLevel(bloom_tex, linear_clamp, (ic + vec2<f32>(o1.x, o1.y) - 0.5) * texel, 0.0).rgb;
    return w0.y * (w0.x * s00 + w1.x * s10) + w1.y * (w0.x * s01 + w1.x * s11);
}

@fragment
fn present_fs(in: PresentOut) -> @location(0) vec4<f32> {
    var color = textureLoad(scene_hdr, vec2<i32>(in.position.xy), 0).rgb;
    if (push.bloom.w > 0.5) {
        color = color + sample_bloom(in.uv) * push.bloom.rgb;
    }
    // Vignette (URP's ApplyVignette).
    if (push.vignette.z > 0.0) {
        var dist = abs(in.uv - push.vignette.xy) * push.vignette.z;
        dist.x = dist.x * push.vignette_color.w;
        let factor = pow(clamp(1.0 - dot(dist, dist), 0.0, 1.0), push.vignette.w);
        color = color * mix(push.vignette_color.rgb, vec3<f32>(1.0), factor);
    }
    color = grade(color * push.grading.x);
    // ACES is not reproduced yet: the renderer is given "none" for it (and says so).
    if (push.color_filter.w > 0.5) {
        color = neutral_tonemap(color);
    }
    let linear = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(linear_to_srgb(linear), 1.0);
}
