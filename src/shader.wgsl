struct Uniforms {
    view_proj: mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var base_tex: texture_2d<f32>;
@group(0) @binding(2) var base_sampler: sampler;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip_pos = uniforms.view_proj * vec4<f32>(in.position, 1.0);
    out.world_pos = in.position;
    out.normal = in.normal;
    out.uv = in.uv;
    return out;
}

const LIGHT_DIR: vec3<f32> = normalize(vec3<f32>(0.5, 0.7, 0.8));
const LIGHT_COLOR: vec3<f32> = vec3<f32>(1.0, 0.97, 0.92);

fn linear_from_gamma_rgb(g: vec3<f32>) -> vec3<f32> {
    return select(
        pow((g + vec3<f32>(0.055)) / vec3<f32>(1.055), vec3<f32>(2.4)),
        g / vec3<f32>(12.92),
        g < vec3<f32>(0.04045),
    );
}

fn gamma_from_linear_rgb(l: vec3<f32>) -> vec3<f32> {
    return select(
        vec3<f32>(1.055) * pow(l, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055),
        l * vec3<f32>(12.92),
        l < vec3<f32>(0.0031308),
    );
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    var n = normalize(in.normal);

    // The albedo atlas (from PNG) is sRGB-encoded; decode to linear before
    // lighting. Alpha (texel.a) drives the erased/transparent regions.
    let texel = textureSample(base_tex, base_sampler, in.uv);
    let a = texel.a;

    // Pixel is TRANSPARENT (erased / translucent) unless alpha is 1: skip it
    // entirely so it is never rendered — nothing fills the transparent part.
    // The pixel keeps whatever is behind the surface (the clear backdrop).
    if (a < 1.0) {
        discard;
    }

    let color = linear_from_gamma_rgb(texel.rgb);

    // Directional key light + gentle hemisphere so the texture reads clearly
    // even on faces that point away from the key light.
    var light = 0.5;
    if (length(n) > 0.0) {
        let diff = max(dot(n, LIGHT_DIR), 0.0);
        let hemi = n.y * 0.5 + 0.5;
        light = 0.42 + 0.38 * diff + 0.22 * hemi;
    }
    let shaded = color * LIGHT_COLOR * light;
    let lit = clamp(shaded, vec3<f32>(0.0), vec3<f32>(1.0));

    // egui-wgpu displays this registered native texture as sRGB/gamma-space
    // data (treats it as NOT sRGB-aware and converts to linear itself). Write
    // GAMMA-ENCODED pixels so the conversion round-trips. Only fully opaque
    // fragments reach here, so the output is alpha 1.
    return vec4<f32>(gamma_from_linear_rgb(lit), 1.0);
}