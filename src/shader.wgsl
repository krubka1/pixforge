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

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    var n = normalize(in.normal);

    // Load the model's actual albedo texture.
    let color = textureSample(base_tex, base_sampler, in.uv).rgb;

    // Directional key light + gentle hemisphere so the texture reads clearly
    // even on faces that point away from the key light.
    var light = 0.5;
    if (length(n) > 0.0) {
        let diff = max(dot(n, LIGHT_DIR), 0.0);
        let hemi = n.y * 0.5 + 0.5;
        light = 0.42 + 0.38 * diff + 0.22 * hemi;
    }
    let shaded = color * LIGHT_COLOR * light;

    return vec4<f32>(clamp(shaded, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
