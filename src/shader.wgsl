struct Uniforms {
    view_proj: mat4x4<f32>,
    /// 0 = opaque pass, 1 = translucent pass. The opaque pass draws only
    /// fully-opaque texels (so translucent ones write no depth and cannot
    /// occlude the surface behind them); the translucent pass then source-over
    /// blends the 0 < alpha < 1 texels over whatever the opaque pass wrote.
    pass_mode: u32,
    /// UV debug overlay: bit 0 = checkerboard, bit 1 = UV grid.
    uv_overlay: u32,
    /// PBR material: x = roughness, y = metallic, z = emissive intensity,
    /// w = ambient occlusion.
    material: vec4<f32>,
    /// Sun direction (xyz, toward the sun) and intensity (w).
    sun: vec4<f32>,
    /// Sun color (rgb).
    sun_color: vec4<f32>,
    /// Environment: x = environment intensity, y = exposure,
    /// z = camera-fill light intensity.
    env: vec4<f32>,
    /// Camera position (xyz) for view-dependent lighting.
    camera_pos: vec4<f32>,
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

const PI: f32 = 3.141592653589793;

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

/// Narkowicz ACES filmic tone mapping: maps the HDR result into [0,1] with a
/// pleasant shoulder that keeps over-brights from clipping to flat white.
fn aces(x: vec3<f32>) -> vec3<f32> {
    const A: f32 = 2.51;
    const B: f32 = 0.03;
    const C: f32 = 2.43;
    const D: f32 = 0.59;
    const E: f32 = 0.14;
    return clamp(
        (x * (A * x + vec3<f32>(B))) / (x * (C * x + vec3<f32>(D)) + vec3<f32>(E)),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
}

fn ggx_ndf(ndoth: f32, rough: f32) -> f32 {
    let a2 = rough * rough;
    let a4 = a2 * a2;
    let d = ndoth * ndoth * (a4 - 1.0) + 1.0;
    return a4 / (PI * d * d);
}

/// Height-correlated Smith geometry (visibility) term.
fn geometry_smith(ndotl: f32, ndotv: f32, rough: f32) -> f32 {
    let a2 = rough * rough;
    let a4 = a2 * a2;
    let lambda_v = sqrt(ndotv * ndotv * (1.0 - a4) + a4);
    let lambda_l = sqrt(ndotl * ndotl * (1.0 - a4) + a4);
    return 0.5 / max(lambda_v + lambda_l, 1e-4);
}

fn schlick_f(vdoth: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - vdoth, 0.0, 1.0), 5.0);
}

/// Van der Corput radical inverse for the i-th of `n` Hammersley samples.
fn hammersley(i: u32, n: u32) -> vec2<f32> {
    var bits = i;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    let vdc = f32(bits) * 2.3283064365386963e-10;
    return vec2<f32>(f32(i) / f32(n), vdc);
}

/// Importance-samples the GGX normal distribution around +Z.
fn ggx_importance_sample(xi: vec2<f32>, rough: f32) -> vec3<f32> {
    let a = rough * rough;
    let phi = 2.0 * PI * xi.x;
    let cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    return vec3<f32>(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

/// An orthonormal frame whose Z axis is `n` (used to sample lobes around a
/// general direction).
fn tangent_frame(n: vec3<f32>) -> mat3x3<f32> {
    let up = select(
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(1.0, 0.0, 0.0),
        abs(n.y) > 0.999,
    );
    let t = normalize(cross(up, n));
    let b = cross(n, t);
    return mat3x3<f32>(t, b, n);
}

/// The stylized analytic sky: a cool horizon-to-zenith gradient plus a warm
/// sun disc. Below the horizon it falls off to a soft warm-grey "ground" tone,
/// so mirror-like (metallic) reflections pointing downward never read as a
/// broken black void. Cheap to evaluate anywhere, so it doubles as the
/// environment map for image-based lighting without any textures.
fn sky(rd: vec3<f32>) -> vec3<f32> {
    let h = clamp(rd.y, -1.0, 1.0);
    let horizon = vec3<f32>(0.60, 0.66, 0.72);
    let zenith = vec3<f32>(0.20, 0.34, 0.60);
    let grad = mix(horizon, zenith, pow(smoothstep(-0.1, 0.5, h), 0.6));
    let ground = mix(horizon, vec3<f32>(0.35, 0.31, 0.28), smoothstep(0.0, -0.6, h));
    let above = mix(ground, grad, smoothstep(-0.02, 0.02, h));
    // Broad sun disc: a very high exponent pinpricks into a sharp specular
    // glint that low roughness turns into noisy star blotches; keep it soft so
    // GGX sky reflections sample the highlight smoothly.
    let vis = smoothstep(-0.03, 0.03, rd.y);
    let sun_disk = pow(max(dot(rd, normalize(uniforms.sun.xyz)), 0.0), 140.0)
        * vis * uniforms.sun_color.xyz * 1.5;
    return above + sun_disk;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // No backface culling: orient the geometric normal toward the camera so
    // every visible surface (front caps, far interior walls seen through an
    // erased hole, two-sided planes) gets view-correct lighting with
    // ndotv >= 0 — keying the flip on `front_facing` fails for cavity views,
    // where the winding faces the camera but the authored normal points away.
    let n_geo = normalize(in.normal);
    let v = normalize(uniforms.camera_pos.xyz - in.world_pos);
    let n = select(-n_geo, n_geo, dot(n_geo, v) >= 0.0);
    let ndotv = saturate(dot(n, v));

    // The albedo atlas (from PNG) is sRGB-encoded; decode to linear before
    // lighting. Alpha (texel.a) drives the erased/transparent regions.
    let texel = textureSample(base_tex, base_sampler, in.uv);
    let a = texel.a;

    // The render is split into two draws. The opaque pass emits only texels
    // with alpha ~ 1, writing depth — so a translucent texel never writes depth
    // and thus never occludes the surface behind it (the far interior wall of a
    // hole). The translucent pass then draws the 0 < alpha < 1 texels
    // source-over with depth writes off, blending over whatever the opaque pass
    // put there (the far wall) instead of over the clear backdrop.
    const ALPHA_EPS: f32 = 0.02;
    let opaque_texel = a >= 1.0 - 0.001;
    if (uniforms.pass_mode == 0u) {
        if (!opaque_texel) {
            discard;
        }
    } else {
        // Translucent pass: skip near-erased slivers and fully-opaque texels.
        if (opaque_texel || a <= ALPHA_EPS) {
            discard;
        }
    }

    let albedo = linear_from_gamma_rgb(texel.rgb);
    // PBR material from uniforms (paintable maps arrive later; constants now).
    let roughness = clamp(uniforms.material.x, 0.03, 1.0);
    let metallic = clamp(uniforms.material.y, 0.0, 1.0);
    let ao = clamp(uniforms.material.w, 0.0, 1.0);
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    // ------- Direct sun (GGX metallic-roughness) -------
    let l = normalize(uniforms.sun.xyz);
    let h = normalize(l + v);
    let ndotl = saturate(dot(n, l));
    let ndoth = saturate(dot(n, h));
    let vdoth = saturate(dot(v, h));
    var direct = vec3<f32>(0.0);
    if (ndotl > 0.0) {
        let d = ggx_ndf(ndoth, roughness);
        let g = geometry_smith(ndotl, ndotv, roughness);
        let f = schlick_f(vdoth, f0);
        let spec = d * g * f / max(4.0 * ndotl * ndotv, 1e-4);
        let diff = (1.0 - f) * (1.0 - metallic) * albedo / PI;
        direct = (diff + spec) * uniforms.sun_color.xyz * uniforms.sun.w * ndotl;
    }

    // ------- Camera fill light -------
    // Low-poly stylization: a soft, warm light from the camera direction that
    // fades in where the sun isn't hitting. This keeps erased-hole interiors
    // readable from any angle — with pure GGX sun the far wall's normal (flipped
    // toward the camera) often points away from the sun, so it would collapse to
    // near-backdrop darkness. The `ndotl_soft` factor backs it off to zero on
    // sun-lit faces so front-facing shading keeps its contrast.
    let ndotl_soft = saturate(1.0 - 1.5 * ndotl);
    direct += ndotl_soft * (1.0 - metallic) * albedo * uniforms.env.z
        * vec3<f32>(1.0, 0.93, 0.85) * (ndotv * 0.66 + 0.34);

    // ------- Diffuse ambient: average the sky over a small cosine-weighted
    // patch around the normal (cheap, smooth; no precomputed irradiance). -------
    // Build a tangent frame to tilt sample directions toward the horizon.
    let up_t = select(
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(1.0, 0.0, 0.0),
        abs(n.y) > 0.999,
    );
    let t_axis = normalize(cross(up_t, n));
    let b_axis = cross(n, t_axis);
    var sky_diff = sky(n);
    sky_diff += sky(normalize(n * 0.8 + t_axis * 0.6));
    sky_diff += sky(normalize(n * 0.8 - t_axis * 0.6));
    sky_diff += sky(normalize(n * 0.8 + b_axis * 0.6));
    sky_diff *= 0.25;
    let env_diff = (1.0 - metallic) * albedo * sky_diff / PI;

    // ------- Specular ambient: GGX importance-sample the sky around the
    // reflection, normalized by the sampled weights. Rarefies to a single
    // mirror ray on smooth surfaces via a constant loop bound + early break. -------
    var env_spec = vec3<f32>(0.0);
    var wsum = 0.0;
    const NSPEC: u32 = 48u;
    var n_sp = NSPEC;
    if (roughness < 0.04) {
        n_sp = 1u;
    }
    let frame = tangent_frame(reflect(-v, n));
    for (var i = 0u; i < NSPEC; i += 1u) {
        if (i >= n_sp) {
            break;
        }
        let h_imp = frame * ggx_importance_sample(hammersley(i, n_sp), roughness);
        let l_imp = normalize(2.0 * dot(v, h_imp) * h_imp - v);
        let ndotl_imp = saturate(dot(n, l_imp));
        if (ndotl_imp > 0.0) {
            let ndoth_imp = saturate(dot(n, h_imp));
            let vdoth_imp = saturate(dot(v, h_imp));
            let d = ggx_ndf(ndoth_imp, roughness);
            let g = geometry_smith(ndotl_imp, ndotv, roughness);
            let f = schlick_f(vdoth_imp, f0);
            // Sample weight (without the Fresnel term, which is the filter).
            let w = (d * g) / max(4.0 * ndotl_imp * ndotv, 1e-4) * ndotl_imp;
            env_spec += f * w * sky(l_imp);
            wsum += w;
        }
    }
    env_spec = env_spec / max(wsum, 1e-6);

    // AO darkens only the diffuse ambient (crevice shading); it must not scale
// the specular environment reflection, or metals (whose colour comes purely
// from reflections here) go black and look broken when the slider moves.
let amb = env_diff * ao * uniforms.env.x + env_spec * uniforms.env.x;
    let lit = aces((direct + amb) * uniforms.env.y);
    var out = clamp(lit + albedo * uniforms.material.z, vec3<f32>(0.0), vec3<f32>(1.0));

    // UV debug overlays, drawn over the lit surface so seams and distortion
    // are visible while painting. Applied before gamma encoding, matching the
    // way the checker palette is mixed in the 2D atlas preview.
    if (uniforms.uv_overlay & 2u) != 0u {
        // UV grid every 1/8 of the [0,1] UV square.
        let uv = in.uv * 8.0;
        let grid = min(fract(uv).x, fract(uv).y);
        if (grid < 0.015) {
            out *= 0.45;
        }
    }
    if (uniforms.uv_overlay & 1u) != 0u {
        // Checkerboard (two shades) at 8x8 across the UV square.
        if ((floor(in.uv.x * 8.0) + floor(in.uv.y * 8.0)) % 2.0) == 0.0 {
            out *= 1.15;
        } else {
            out *= 0.62;
        }
    }

    // egui-wgpu displays this registered native texture as sRGB/gamma-space
    // data (treats it as NOT sRGB-aware and converts to linear itself). Write
    // GAMMA-ENCODED pixels so the conversion round-trips. The surface alpha is
    // carried through and blended in the pipeline (source-over) against the
    // clear backdrop, so semi-transparent texels reveal what is behind them.
    return vec4<f32>(gamma_from_linear_rgb(out), a);
}