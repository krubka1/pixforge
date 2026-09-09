struct Uniforms {
    view_proj: mat4x4<f32>,
    // Fill composited under transparent (erased) texels, drawn flat onto the
    // mesh (same look as the Texture preview). bg_a/bg_b are the two checker
    // colors (identical = solid fill); checker_scale = squares per UV axis.
    bg_a: vec4<f32>,
    bg_b: vec4<f32>,
    checker_scale: f32,
    checker_on: f32,
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
    let color = linear_from_gamma_rgb(texel.rgb);
    let a = texel.a;

    // Directional key light + gentle hemisphere so the texture reads clearly
    // even on faces that point away from the key light.
    var light = 0.5;
    if (length(n) > 0.0) {
        let diff = max(dot(n, LIGHT_DIR), 0.0);
        let hemi = n.y * 0.5 + 0.5;
        light = 0.42 + 0.38 * diff + 0.22 * hemi;
    }
    let shaded = color * LIGHT_COLOR * light;
    let lit = vec4<f32>(clamp(shaded, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);

    // Erased texels are filled with the chosen background style (checkerboard
    // or solid, hex/sRGB values) exactly like the Texture preview. The fill
    // colors are decoded to linear so the composite happens in linear space.
    var bg_raw = uniforms.bg_a.rgb;
    if (uniforms.checker_on > 0.5) {
        let cell = floor(in.uv * uniforms.checker_scale);
        if ((cell.x + cell.y) % 2.0 > 0.5) {
            bg_raw = uniforms.bg_b.rgb;
        }
    }
    let bg = vec4<f32>(linear_from_gamma_rgb(bg_raw), 1.0);
    let out = mix(bg, lit, a);

    // egui-wgpu displays this registered native texture as sRGB/gamma-space
    // data (it treats it as NOT sRGB-aware and converts to linear itself).
    // Write GAMMA-ENCODED pixels so that conversion round-trips and the flat
    // fill doesn't collapse toward black. Output stays opaque (alpha 1).
    return vec4<f32>(gamma_from_linear_rgb(out.rgb), 1.0);
}