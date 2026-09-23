struct Uniforms {
    view_proj: mat4x4<f32>,
    /// Inverse of `view_proj`, used by the background quad to un-project each
    /// pixel back to a world ray so the analytic sky / loaded environment map
    /// shows correctly behind the model.
    view_proj_inv: mat4x4<f32>,
    /// 0 = opaque pass, 1 = translucent pass. The opaque pass draws only
    /// fully-opaque texels (so translucent ones write no depth and cannot
    /// occlude the surface behind them); the translucent pass then source-over
    /// blends the 0 < alpha < 1 texels over whatever the opaque pass wrote.
    pass_mode: u32,
    /// UV debug overlay: bit 0 = checkerboard, bit 1 = UV grid.
    uv_overlay: u32,
    /// PBR material: x = roughness, y = metallic, z = emissive intensity,
    /// w = viewport parallax strength (0 disables the relief shift).
    material: vec4<f32>,
    /// Sun direction (xyz, toward the sun) and intensity (w).
    sun: vec4<f32>,
    /// Sun color (rgb).
    sun_color: vec4<f32>,
    /// Environment: x = environment intensity, y = exposure,
    /// z = camera-fill light intensity, w = height-map resolution in texels
    /// (0 when no height map is bound) used to floor the bump gradient step.
    env: vec4<f32>,
    /// Camera position (xyz) for view-dependent lighting. w = environment mip
    /// count minus one (0 = no environment bound; samples the analytic sky).
    camera_pos: vec4<f32>,
    /// Environment rotation around the vertical axis in radians (x).
    env_rot: vec4<f32>,
    /// Uniform color of the analytic sky (used when `camera_pos.w == 0`,
    /// i.e. no environment map is bound): ambient light + viewport backdrop.
    sky_color: vec4<f32>,
    /// Brush cursor mask drawn over the surface. xyz = brush center in world
    /// space, w = 1 when the mask is active (fragment discards otherwise).
    overlay_center: vec4<f32>,
    /// xyz = brush-local U axis (unit vector on the surface tangent plane),
    /// w = brush radius in world units.
    overlay_u: vec4<f32>,
    /// xyz = brush-local V axis (unit, tangent ⊥ U), w = footprint shape:
    /// 0 = round, 1 = square, 2 = diamond, 3 = texture (mirrors the paint
    /// footprints).
    overlay_v: vec4<f32>,
    /// RGBA tint of the mask (gamma-space, like the user-picked brush color).
    overlay_color: vec4<f32>,
    /// Texture-shape mask: xy = brush sprite size in texels, z = stamp rotation
    /// in radians, w = flip_x | (flip_y << 1). Ignored by the other shapes.
    overlay_sprite: vec4<f32>,
    /// x = 1 when drawing the dark scrim pass, 2 for the emissive glow pass.
    /// The shader switches its output blend accordingly so the cursor reads
    /// as a dark+emissive footprint distinct from the underlying material.
    overlay_params: vec4<f32>,
    /// Anchored texture-pattern frame (sampled by `shape == 3`): xyz = the
    /// stroke-start anchor position, w = the phase divider = the anchor dab's
    /// world radius (or the locked texture size) divided by `texture_scale`, so
    /// the preview's `r / aa_chart.w` reproduces the stamp's
    /// `texture_scale * r / divider` phase scale exactly. The pattern phase of a
    /// fragment is its world position relative to this anchor, so the preview is
    /// glued to the exact seam the stamp will paint.
    overlay_anchor: vec4<f32>,
    /// xyz = anchor U axis, w = 1 when the anchored pattern preview is active
    /// (0 restores the plain rubber-stamp dab reading for this sprite).
    overlay_anchor_u: vec4<f32>,
    /// xyz = anchor V axis (orthonormal to the U axis on the tangent plane).
    overlay_anchor_v: vec4<f32>,
    /// xyz = unit view direction (camera forward) shared by all parallel rays
    /// of an orthographic view; w = 1 in ortho, 0 in perspective. Shading,
    /// parallax and the background/environment rays branch on it so an
    /// orthographic camera (whose per-pixel direction is constant, not
    /// `camera_pos - position`) lights up correctly instead of inverting.
    view_forward: vec4<f32>,
}
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var base_tex: texture_2d<f32>;
@group(0) @binding(2) var base_sampler: sampler;
@group(0) @binding(3) var material_tex: texture_2d<f32>;
@group(0) @binding(4) var material_sampler: sampler;
@group(0) @binding(5) var height_tex: texture_2d<f32>;
@group(0) @binding(6) var env_tex: texture_2d<f32>;
@group(0) @binding(7) var env_sampler: sampler;
@group(0) @binding(8) var brush_tex: texture_2d<f32>;
@group(0) @binding(9) var brush_sampler: sampler;
@group(0) @binding(10) var extras_tex: texture_2d<f32>;

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

// Overlay vertex inputs: the mesh's position/normal/uv from slot 0 plus the
// per-vertex surface phase `(u, v)` from the overlay slot-1 phase buffer —
// the geodesic unwrap field the texture-sprite cursor reads. When no phase
// field is uploaded (flag clear) every vertex reads (0,0) and the fragment
// falls back to the analytical anchor-plane chord.
struct VsOverlayIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) phase: vec2<f32>,
};

struct VsOverlayOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) phase: vec2<f32>,
};

@vertex
fn overlay_vs_main(in: VsOverlayIn) -> VsOverlayOut {
    var out: VsOverlayOut;
    out.clip_pos = uniforms.view_proj * vec4<f32>(in.position, 1.0);
    out.world_pos = in.position;
    out.normal = in.normal;
    out.uv = in.uv;
    out.phase = in.phase;
    return out;
}

const PI: f32 = 3.141592653589793;

// The height atlas stores the per-texel bump strength in its G channel
// divided by 8 (the slider max) so it packs into a u8; multiply back here.
const BUMP_SCALE: f32 = 8.0;

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

/// Maps a world direction to equirectangular UVs (u wraps, v = vertical angle).
fn dir_to_eqrect(rd: vec3<f32>) -> vec2<f32> {
    let u = 0.5 + (atan2(rd.z, rd.x) + uniforms.env_rot.x) / (2.0 * PI);
    let v = 0.5 - asin(clamp(rd.y, -1.0, 1.0)) / PI;
    return vec2<f32>(u, v);
}

/// The analytic sky/environment: a single uniform color picked by the user
/// (ambient-only, no up/down gradient). With the direct sun on, the splash of
/// sunlight carries the shading; with it off, uniform ambient means "shadow
/// and light side" disappear and the model reads flat, lit everywhere by the
/// same skylight. Cheap to evaluate anywhere.
fn env_sky(rd: vec3<f32>) -> vec3<f32> {
    if uniforms.camera_pos.w > 0.0 {
        return textureSample(env_tex, env_sampler, dir_to_eqrect(rd)).rgb;
    }
    return uniforms.sky_color.rgb;
}

/// Specular-environment lookup: a roughness-stepped mip of the env map (higher
/// LOD for rougher surfaces = a wider, dimmer reflection lobe), the uniform
/// analytic-sky color as fallback.
fn env_sky_lod(rd: vec3<f32>, roughness: f32) -> vec3<f32> {
    if uniforms.camera_pos.w > 0.0 {
        let lod = uniforms.camera_pos.w * clamp(roughness, 0.0, 1.0);
        return textureSampleLevel(env_tex, env_sampler, dir_to_eqrect(rd), lod).rgb;
    }
    return uniforms.sky_color.rgb;
}

// Height/bump helpers (Mikkelsen surface gradients, [Mikkelsen 2020] JCGT
// 9(3)). The height map's R channel stores a signed height (128 = flat). The
// UV-space gradient is taken by central differences whose step is clamped to
// at least one top-level texel — differencing a magnified 2x2 bilinear block
// reads a piecewise-linear signal whose derivative is faceted (the classic
// "low resolution" bump artefact). The gradient is then converted to screen
// space with the chain rule (dpdx/dpdy of uv) and the tangent frame comes
// from screen-space world-position derivatives, so the bump follows zoom and
// the atlas tiling rate instead of a fixed uv offset.
fn height_at(uv: vec2<f32>) -> f32 {
    return textureSample(height_tex, material_sampler, uv).r * 2.0 - 1.0;
}

fn perturb_normal(
    surf_pos: vec3<f32>,
    surf_n: vec3<f32>,
    dhdx: f32,
    dhdy: f32,
    bump: f32,
) -> vec3<f32> {
    let dpx = dpdx(surf_pos);
    let dpy = dpdy(surf_pos);
    let r1 = cross(dpy, surf_n);
    let r2 = cross(surf_n, dpx);
    let det = dot(dpx, r1);
    if (abs(det) > 1e-12) {
        let s = select(-1.0, 1.0, det >= 0.0) / max(1e-12, abs(det));
        let perturb = (r1 * dhdx + r2 * dhdy) * (s * bump);
        // Cap the horizontal tilt so the steep height gradient at a paint
        // boundary can't swing the normal to grazing incidence. Near-tangent
        // normals flip the Fresnel term toward 1 and stack the specular
        // reflection, painting a whitish rim on semi-transparent stroke edges
        // in direct light. The cap keeps the relief while bounding the tilt
        // to about 40 degrees (matches how renderers clamp perturbed normals
        // to keep them in the upper hemisphere).
        const kMaxTilt: f32 = 0.85;
        let plen = length(perturb);
        let capped = select(
            perturb * (kMaxTilt / plen),
            perturb,
            plen <= kMaxTilt,
        );
        return normalize(surf_n - capped);
    }
    return surf_n;
}

/// Unit vector from a surface point toward the camera: the per-point
/// `camera_pos - p` direction in perspective, or the constant (parallel-ray)
/// view direction in orthographic views — a single point can't stand in for a
/// parallel camera without inverting/streaking the lighting.
fn view_dir(p: vec3<f32>) -> vec3<f32> {
    let f = uniforms.view_forward;
    if (f.w > 0.5) {
        return -f.xyz;
    }
    return normalize(uniforms.camera_pos.xyz - p);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // No backface culling: orient the geometric normal toward the camera so
    // every visible surface (front caps, far interior walls seen through an
    // erased hole, two-sided planes) gets view-correct lighting with
    // ndotv >= 0 — keying the flip on `front_facing` fails for cavity views,
    // where the winding faces the camera but the authored normal points away.
    let n_geo = normalize(in.normal);
    let v = view_dir(in.world_pos);
    var n = select(-n_geo, n_geo, dot(n_geo, v) >= 0.0);

    let h_center = textureSample(height_tex, material_sampler, in.uv);
    let bump = h_center.g * BUMP_SCALE;
    if (bump > 0.0) {
        let duvdx = dpdx(in.uv);
        let duvdy = dpdy(in.uv);
        // Screen-space footprint magnitude of each uv axis (uv per pixel).
        let su = length(vec2<f32>(duvdx.x, duvdy.x));
        let sv = length(vec2<f32>(duvdx.y, duvdy.y));
        // Differentiate over at least one top-level texel (resolution passed
        // in uniforms.env.w) so a heavily magnified map keeps a gradient.
        let texel = 1.0 / max(uniforms.env.w, 1.0);
        let u_step = max(su, texel);
        let v_step = max(sv, texel);
        let dhdu = (height_at(in.uv + vec2<f32>(u_step, 0.0))
            - height_at(in.uv - vec2<f32>(u_step, 0.0)))
            / (2.0 * u_step);
        let dhdv = (height_at(in.uv + vec2<f32>(0.0, v_step))
            - height_at(in.uv - vec2<f32>(0.0, v_step)))
            / (2.0 * v_step);
        // Chain rule: UV-space gradient → screen-space height derivatives.
        let dhdx = dhdu * duvdx.x + dhdv * duvdx.y;
        let dhdy = dhdu * duvdy.x + dhdv * duvdy.y;
        n = perturb_normal(in.world_pos, n, dhdx, dhdy, bump);
    }

    // Parallax relief: nudge the per-texel sample opposite to the view tilt,
    // scaled by the signed height and the viewport parallax strength
    // (`uniforms.material.w`; 0 disables). The shift runs along the surface
    // plane — the view ray is resolved into the UV axes through the
    // screen-space world/uv derivatives (least-squares), so it follows the
    // mesh's real layout instead of assuming some tangent frame.
    var p_uv = in.uv;
    if (uniforms.material.w > 0.0) {
        let signed_h = h_center.r * 2.0 - 1.0;
        let tilt = v - n_geo * dot(v, n_geo);
        let tilt_len = length(tilt);
        if (tilt_len > 1e-4) {
            let disp = tilt / tilt_len * (signed_h * uniforms.material.w);
            let ddu = dpdx(in.uv);
            let ddv = dpdy(in.uv);
            let ddwu = dpdx(in.world_pos);
            let ddwv = dpdy(in.world_pos);
            let a11 = dot(ddwu, ddwu);
            let b11 = dot(ddwu, ddwv);
            let c11 = dot(ddwv, ddwv);
            let det = max(a11 * c11 - b11 * b11, 1e-6);
            let k1 = (c11 * dot(disp, ddwu) - b11 * dot(disp, ddwv)) / det;
            let k2 = (a11 * dot(disp, ddwv) - b11 * dot(disp, ddwu)) / det;
            p_uv = in.uv + ddu * k1 + ddv * k2;
        }
    }

    let ndotv = saturate(dot(n, v));

    // The albedo atlas (from PNG) is sRGB-encoded; decode to linear before
    // lighting. Alpha (texel.a) drives the erased/transparent regions.
    let texel = textureSample(base_tex, base_sampler, p_uv);
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
    // PBR material from the per-layer material map (RGBA = roughness, metallic,
    // emissive/3, ambient occlusion). The extras map (R = clearcoat roughness,
    // GBA = emissive colour) is sampled alongside; clearcoat intensity and the
    // specular IOR piggyback the height atlas (B/A). All samplers follow the
    // parallax offset so the relief shifts the whole shading consistently.
    let mtl = textureSample(material_tex, material_sampler, p_uv);
    let roughness = clamp(mtl.r, 0.03, 1.0);
    let metallic = clamp(mtl.g, 0.0, 1.0);
    let emiss = mtl.b * 3.0;
    let ao = clamp(mtl.a, 0.0, 1.0);
    let ext = textureSample(extras_tex, material_sampler, p_uv);
    let h_p = textureSample(height_tex, material_sampler, p_uv);
    // Specular IOR (1.0..2.5): exact dielectric Schlick f0, so 1.5 (the
    // default) reproduces the old fixed 0.04 fresnel.
    let ior = mix(1.0, 2.5, h_p.a);
    let ior_f0 = (ior - 1.0) / (ior + 1.0);
    let f0 = mix(vec3<f32>(ior_f0 * ior_f0), albedo, metallic);
    let clearcoat = h_p.b;
    let cc_rough = clamp(ext.r, 0.03, 1.0);
    let ecolor = ext.gba;

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
    var sky_diff = env_sky(n);
    sky_diff += env_sky(normalize(n * 0.8 + t_axis * 0.6));
    sky_diff += env_sky(normalize(n * 0.8 - t_axis * 0.6));
    sky_diff += env_sky(normalize(n * 0.8 + b_axis * 0.6));
    sky_diff *= 0.25;
    let env_diff = (1.0 - metallic) * albedo * sky_diff / PI;

    // ------- Specular ambient: the sky reflected at the mirror direction,
    // weighted by the view-angle Fresnel term. No importance sampling, no
    // hemisphere guard. A roughness-0 metal should look like chrome: the
    // reflection of a raking camera comes from *below* the surface horizon
    // (ground radiance), and a naive `ndotl > 0` rejection would throw that
    // ray away and turn the whole material black except for the direct sun
    // glint — the "ghostly model-shaped" hole in the PBR. The sky is a smooth,
    // low-range analytic radiance (see sky()), so a lobe sample collapses to
    // the mirror direction and Fresnel keeps grazing edges reflective while
    // rough dielectrics stay mostly diffuse. The direct sun lobe below
    // is where roughness actually shows.
    let env_spec = schlick_f(ndotv, f0) * env_sky_lod(reflect(-v, n), roughness);

    // AO darkens only the diffuse ambient (crevice shading); it must not scale
// the specular environment reflection, or metals (whose colour comes purely
// from reflections here) go black and look broken when the slider moves.
let amb = env_diff * ao * uniforms.env.x + env_spec * uniforms.env.x;

    // Coverage-correct translucency: a partially-covered texel (0 < a < 1,
    // drawn in pass 2) only *partially* reflects light. Shading it at full
    // radiance and blending it over the pass-1 backdrop (the lit surface
    // behind it) double-adds light through the feather, which reads as whitish
    // fog in direct sun. Scaling the radiance by the coverage keeps the tint
    // without the bright halo — opaque texels are unchanged (coverage = 1).
    let coverage = select(a, 1.0, opaque_texel);

    // Clearcoat: a second GGX specular lobe over the base at its own
    // roughness, layered with the KHR_materials_clearcoat Fresnel (a 1.5-IOR
    // dielectric). Zero clearcoat leaves the base untouched.
    let fres_coat = 0.04 + 0.96 * pow(1.0 - abs(ndotv), 5.0);
    let cem = clearcoat * fres_coat;
    var coat = vec3<f32>(0.0);
    if (ndotl > 0.0) {
        let cc_a = cc_rough * cc_rough;
        let g_cc = geometry_smith(ndotl, ndotv, cc_a);
        let cc_spec = ggx_ndf(ndoth, cc_a) * g_cc / max(4.0 * ndotl * ndotv, 1e-4);
        coat = cc_spec * uniforms.sun_color.xyz * uniforms.sun.w * ndotl;
    }
    // The coat reflects the environment as well as the sun. Without this the
    // base radiance is attenuated by (1 - cem) at grazing angles while the coat
    // only re-adds the narrow sun lobe — leaving a hard dark (or, into the
    // glint, hard bright) ring across the silhouette. `cem` already carries the
    // coat Fresnel, so the sky term is added unweighted here.
    coat += env_sky_lod(reflect(-v, n), cc_rough) * uniforms.env.x;
    // Energy-conserving layering: the base radiance parks under the coat
    // ((1 - cem)) while the coat lobe sits on top, scaled by clearcoat.
    let radi = (direct + amb) * (1.0 - cem) + coat * cem;

    let lit = aces(radi * coverage * uniforms.env.y);
    // Emission: the tinted colour (extras GBA) modulated by the white emissive
    // intensity, attenuated by the clearcoat Fresnel reflection of the coat.
    var out = clamp(
        lit + mix(albedo, ecolor, saturate(emiss)) * emiss * (1.0 - cem),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );

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

// ------- Background: the analytic sky / loaded environment behind the model.
// A fullscreen triangle (no vertex buffers; positions come from vertex_index)
// reconstructs a world ray per pixel through the inverse view-projection and
// samples the same `env_sky` the IBL uses, so the skybox shows behind the
// model (and around erased/transparent texels) instead of a flat clear color.

struct BgVsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn bg_vs(@builtin(vertex_index) vi: u32) -> BgVsOut {
    // Cover the whole clip space with one oversized triangle.
    let corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    let p = corners[vi];
    var out: BgVsOut;
    out.clip = vec4<f32>(p, 0.0, 1.0);
    out.uv = vec2<f32>(p.x * 0.5 + 0.5, p.y * 0.5 + 0.5);
    return out;
}

@fragment
fn bg_fs(in: BgVsOut) -> @location(0) vec4<f32> {
    // The far-plane clip position; un-project to a world point. Perspective
    // subtracts the eye for its view ray; orthographic rays are parallel, so
    // the view uses the bound constant forward direction.
    let world = uniforms.view_proj_inv * vec4<f32>(in.uv * 2.0 - 1.0, 1.0, 1.0);
    let f = uniforms.view_forward;
    let rd = select(
        normalize(world.xyz / world.w - uniforms.camera_pos.xyz),
        f.xyz,
        f.w > 0.5,
    );
    // Match the IBL treatment (environment intensity + exposure + tonemap +
    // gamma) so the backdrop and the surfaces lit by it stay consistent.
    return vec4<f32>(gamma_from_linear_rgb(aces(env_sky(rd) * uniforms.env.x * uniforms.env.y)), 1.0);
}

// ------- Brush cursor: a translucent mask laid over the *visible* surface.
// Retraces the mesh with the same vertex shader, depth-tests against the
// opaque pass (so the far side of a wall, hidden behind the near surface,
// never gets tinted) and keeps only the fragments inside the brush footprint.
// Because it sits on the actual geometry, the mask conforms to the model's
// curvature exactly — no flat-sprite or 2D-projection approximation.
@fragment
fn overlay_fs(in: VsOverlayOut) -> @location(0) vec4<f32> {
    if (uniforms.overlay_center.w < 0.5) {
        discard;
    }
    let pos = in.world_pos;
    // The mask must sit on the *visible* surface: a fragment facing away from
    // the eye is the back/far side of the model, which the depth test alone
    // only hides while a nearer surface covers it (it leaks around the
    // silhouette). Same criterion as the stamp's convex occlusion, so the
    // cursor and the painted result agree — and a curved footprint (projected
    // onto the brush's tangent plane) can otherwise admit far-side points.
    if (dot(normalize(in.normal), view_dir(pos)) <= 0.0) {
        discard;
    }
    let rel = pos - uniforms.overlay_center.xyz;
    let r = uniforms.overlay_u.w;
    let tu = dot(rel, uniforms.overlay_u.xyz);
    let tv = dot(rel, uniforms.overlay_v.xyz);
    let shape = u32(uniforms.overlay_v.w);
    var coverage = 0.0;
    if (shape == 0u) {
        // Round: same "within a sphere of radius r" test as the stamp's
        // footprint (3D distance, not just the tangent plane).
        let d = length(rel);
        if (d <= r) { coverage = 1.0; }
    } else if (shape == 1u) {
        // Square: axis-aligned in the brush-local tangent plane.
        if (abs(tu) <= r && abs(tv) <= r) { coverage = 1.0; }
    } else if (shape == 2u) {
        // Diamond.
        let m = abs(tu) + abs(tv);
        if (m <= r) { coverage = 1.0; }
    } else if (shape == 3u) {
        // Texture: the sprite's alpha is the coverage, resolved exactly like
        // the stamp. With the anchored-pattern frame set (overlay_anchor_u.w
        // >= 0.5) the sprite is read at each fragment's world position relative
        // to the stroke-start anchor; the phase then wraps so the cursor shows
        // the real texture glued to the surface, and once a stroke is running
        // it previews the exact anchored pattern that will be painted. Both
        // paths tile the sprite and mask it by the selected paint-window shape,
        // matching the stamp's every-dab frame.
        let sw = u32(uniforms.overlay_sprite.x);
        let sh = u32(uniforms.overlay_sprite.y);
        let s = f32(max(max(sw, sh), 1u));
        let ww = f32(sw) * (2.0 * r) / s;
        let wh = f32(sh) * (2.0 * r) / s;
        let rot = uniforms.overlay_sprite.z;
        let sr = sin(rot);
        let cr = cos(rot);
        let anchored = uniforms.overlay_anchor_u.w >= 0.5;
        // Pattern phase: world space relative to the captured anchor frame.
        // With a per-vertex phase field uploaded (`overlay_anchor_v.w`), the
        // fragment reads the interpolated *geodesic* phase instead — the
        // along-surface seam the stroke paints on curved bodies is the same
        // field, so the cursor preview matches the stamp texel-for-texel.
        var bx: f32;
        var by: f32;
        if (anchored) {
            let scale = r / max(uniforms.overlay_anchor.w, 1e-6);
            if (uniforms.overlay_anchor_v.w >= 0.5) {
                bx = in.phase.x * scale;
                by = in.phase.y * scale;
            } else {
                let ad = pos - uniforms.overlay_anchor.xyz;
                bx = dot(ad, uniforms.overlay_anchor_u.xyz) * scale;
                by = dot(ad, uniforms.overlay_anchor_v.xyz) * scale;
            }
        } else {
            bx = tu;
            by = tv;
        }
        let x = bx * cr - by * sr;
        let y = bx * sr + by * cr;
        let flip = u32(uniforms.overlay_sprite.w);
        let rx = select(x, -x, (flip & 1u) == 1u);
        let ry = select(y, -y, ((flip >> 1u) & 1u) == 1u);
        var u = 0.5 + rx / ww;
        var v = 0.5 - ry / wh;
        // Seam writing: repeat the sprite infinitely (the same rem_euclid-style
        // wrap the stamp uses) under the user's paint-window mask (0 round /
        // 1 square / 2 diamond — riding in overlay_params.y) so the cursor reads
        // as a continuous texture framed by the selected shape, whether or not
        // the anchored-pattern frame is set. The frame shapes every texture dab
        // (locked or not), so the cursor must mask both paths identically. Same
        // flat-core + C¹ smoothstep skirt as `Window::mask`.
        u = fract(u);
        v = fract(v);
        let wshape = u32(uniforms.overlay_params.y);
        var ttm: f32;
        if (wshape == 1u) {
            ttm = max(abs(tu), abs(tv)) / r;
        } else if (wshape == 2u) {
            ttm = (abs(tu) + abs(tv)) / r;
        } else {
            ttm = length(rel) / r;
        }
        let tt = clamp(ttm, 0.0, 1.0);
        var win = 1.0;
        if (tt > 0.5) {
            let xw = (tt - 0.5) * 2.0;
            win = 1.0 - xw * xw * (3.0 - 2.0 * xw);
        }
        let sx = min(u32(u * f32(sw)), sw - 1u);
        let sy = min(u32(v * f32(sh)), sh - 1u);
        coverage = textureLoad(brush_tex, vec2<i32>(i32(sx), i32(sy)), 0).a * win;
    }
    let mode = u32(uniforms.overlay_params.x);
    if (coverage <= 0.003) {
        discard;
    }
    // Two-pass dark+emissive overlay: the dark scrim pass (mode 1) darkens the
    // surface so the footprint reads in daylight; the emissive glow pass
    // (mode 2) adds a bright coloured tint visible in dark scenes.  Together
    // they give the cursor a distinctive "dark+emissive" look on any material.
    if (mode == 1u) {
        // Dark scrim: black tint, source-over blended (the pipeline is set to
        // SrcAlpha / OneMinusSrcAlpha). Alpha controls how strongly the scrim
        // darkens the underlying surface.
        return vec4<f32>(0.0, 0.0, 0.0, 0.45 * coverage);
    } else if (mode == 2u) {
        // Emissive glow: brush colour boosted, additive blend (the pipeline
        // uses SrcAlpha / One so the output brightens the surface directly).
        let c = uniforms.overlay_color.rgb * 1.8;
        return vec4<f32>(c, 0.4 * coverage);
    }
    // Fallback (should not be reached with the two-pass draw).
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}