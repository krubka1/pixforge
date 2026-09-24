use glam::Vec3;
use rayon::prelude::*;
use std::collections::HashMap;

/// Writes a PNG of the albedo atlas at `path`, encoding the raw CPU rows
/// top-down (row 0 = texture top, matching glTF UV v=0).
pub fn save_atlas_png(path: &str, tex: &TextureData) -> std::io::Result<()> {
    let img = image::RgbaImage::from_raw(tex.width, tex.height, tex.rgba.clone())
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "bad atlas size"))?;
    img.save(path)
        .map_err(|e| std::io::Error::other(e.to_string()))
}

/// Encodes an RGBA atlas into PNG bytes (same encoder `project` uses).
fn png_bytes(width: u32, height: u32, rgba: &[u8]) -> Vec<u8> {
    use image::ImageEncoder;
    let mut buf = Vec::new();
    let enc = image::codecs::png::PngEncoder::new(&mut buf);
    let _ = enc.write_image(rgba, width, height, image::ExtendedColorType::Rgba8);
    buf
}

/// Re-packs the material map into glTF's ORM layout: R = ambient occlusion,
/// G = roughness, B = metallic (the app's material atlas stores
/// R = roughness, G = metallic, B = emissive/3, A = ao).
fn orm_from_material(tex: &TextureData) -> Vec<u8> {
    let (w, h) = (tex.width, tex.height);
    let mut out = vec![0u8; (w * h * 4) as usize];
    for i in (0..(w * h * 4) as usize).step_by(4) {
        out[i] = tex.rgba[i + 3]; // ao
        out[i + 1] = tex.rgba[i]; // roughness
        out[i + 2] = tex.rgba[i + 1]; // metallic
        out[i + 3] = 255;
    }
    out
}

/// Extracts the emissive texture: the tinted emissive colour (extras atlas
/// G/B/A, straight sRGB) scaled by the per-texel emissive intensity (stored /3
/// in the material atlas B channel), un-clamped back to the 0..=1 glTF range.
fn emissive_from_maps(material: &TextureData, extras: &TextureData) -> Vec<u8> {
    let (w, h) = (
        material.width.max(extras.width),
        material.height.max(extras.height),
    );
    let mut out = vec![0u8; (w * h * 4) as usize];
    for i in (0..(w * h * 4) as usize).step_by(4) {
        let e = (material.rgba[i + 2] as f32 / 255.0) * 3.0;
        for c in 0..3 {
            out[i + c] = (e * (extras.rgba[i + 1 + c] as f32 / 255.0) * 255.0)
                .round()
                .clamp(0.0, 255.0) as u8;
        }
        out[i + 3] = 255;
    }
    out
}

/// Packs the clearcoat channel set into one PNG shared by both
/// KHR_materials_clearcoat textures: R = clearcoat intensity (height atlas B),
/// G = clearcoat roughness (extras atlas R), B = specular IOR (height atlas A,
/// encoded (ior - 1) / 1.5), A = 255.
fn coat_from_maps(height: &TextureData, extras: &TextureData) -> Vec<u8> {
    let (w, h) = (height.width, height.height);
    let mut out = vec![0u8; (w * h * 4) as usize];
    for i in (0..(w * h * 4) as usize).step_by(4) {
        out[i] = height.rgba[i + 2];
        out[i + 1] = extras.rgba[i];
        out[i + 2] = height.rgba[i + 3];
        out[i + 3] = 255;
    }
    out
}

/// Bakes a tangent-space normal map from the height atlas so the painted
/// relief survives in other viewers. R holds a signed height ((h + 1) / 2,
/// 128 = flat) and G holds bump strength /8. The 2-texel gradient is scaled
/// by strength * 8 (mirroring `BUMP_SCALE` in the shader) and the horizontal
/// tilt is capped to 0.85 (mirroring the `perturb_normal` tilt cap), then the
/// result is packed as a normal map: xy = -dH, z = 1, normalized.
fn normal_map_from_height(tex: &TextureData) -> Vec<u8> {
    let w = tex.width.max(1);
    let h = tex.height.max(1);
    // The output atlas must match the source size exactly (the caller encodes
    // it at `tex.width`/`tex.height`). Sampling clamps to the same bounds, so a
    // 1×1 (or N×1 / 1×N) atlas produces a flat normal instead of reading past
    // the source buffer.
    let mut out = vec![0u8; (w * h * 4) as usize];
    let val = |x: i64, y: i64, ch: usize| -> f32 {
        let x = x.clamp(0, w as i64 - 1);
        let y = y.clamp(0, h as i64 - 1);
        tex.rgba[((y * w as i64 + x) as usize) * 4 + ch] as f32 / 255.0
    };
    for y in 0..h {
        for x in 0..w {
            let (xx, yy) = (x as i64, y as i64);
            // Gradient of the ENCODED height (shades per texel).
            let dh_u = (val(xx + 1, yy, 0) - val(xx - 1, yy, 0)) * 0.5;
            let dh_v = (val(xx, yy + 1, 0) - val(xx, yy - 1, 0)) * 0.5;
            let strength = val(xx, yy, 1);
            let (gx, gy) = (-dh_u * strength * 8.0, -dh_v * strength * 8.0);
            let m = (gx * gx + gy * gy).sqrt();
            let (gx, gy) = if m > 0.85 {
                (gx * 0.85 / m, gy * 0.85 / m)
            } else {
                (gx, gy)
            };
            let ilen = 1.0 / (gx * gx + gy * gy + 1.0).sqrt();
            let n = [gx * ilen, gy * ilen, ilen];
            let i = ((y * w + x) as usize) * 4;
            out[i] = ((n[0] * 0.5 + 0.5) * 255.0).round() as u8;
            out[i + 1] = ((n[1] * 0.5 + 0.5) * 255.0).round() as u8;
            out[i + 2] = ((n[2] * 0.5 + 0.5) * 255.0).round() as u8;
            out[i + 3] = 255;
        }
    }
    out
}

fn white_atlas(width: u32, height: u32) -> Vec<u8> {
    vec![255; (width * height * 4) as usize]
}

/// Saves the mesh with its baked layer stack as a binary glTF (.glb): one
/// primitive with positions/normals/UVs, and five embedded PNG atlases —
/// base color (flattened albedo, straight alpha, `alphaMode: BLEND` so
/// erased holes survive), metallic-roughness + occlusion (ORM repack), a
/// tinted emissive map (extras color × material intensity), a tangent-space
/// normal map baked from the height atlas, and a coat map sharing clearcoat /
/// clearcoat-roughness / IOR channels (KHR_materials_clearcoat + ior).
pub fn save_glb(path: &str, mesh: &MeshData) -> std::io::Result<()> {
    let nv = mesh.positions.len();
    if nv == 0 || mesh.indices.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "mesh has no geometry to export",
        ));
    }

    // Bake the layer stack into export atlases (fall back to neutral sheets
    // when a mesh has no paint/layers).
    let albedo = mesh.flattened_atlas().unwrap_or(TextureData {
        width: 1,
        height: 1,
        rgba: white_atlas(1, 1),
    });
    let material = mesh
        .flattened_material_atlas()
        .map(|t| t.rgba)
        .unwrap_or_else(|| {
            let mut v = vec![0u8; (albedo.width * albedo.height * 4) as usize];
            for i in (0..v.len()).step_by(4) {
                v[i] = 140; // roughness 0.55
                v[i + 3] = 255; // ao 1.0
            }
            v
        });
    let height = match mesh.flattened_height_atlas() {
        Some(t) => t.rgba,
        None => [128, 0, 0, 85].repeat((albedo.width * albedo.height) as usize),
    };
    let extras = match mesh.flattened_extras_atlas() {
        Some(t) => t.rgba,
        // Seed: satin 0.6 coat roughness, untinted white emission.
        None => [153, 255, 255, 255].repeat((albedo.width * albedo.height) as usize),
    };
    let base_png = png_bytes(albedo.width, albedo.height, &albedo.rgba);
    let orm_png = png_bytes(
        albedo.width,
        albedo.height,
        &orm_from_material(&TextureData {
            width: albedo.width,
            height: albedo.height,
            rgba: material.clone(),
        }),
    );
    let emissive_png = png_bytes(
        albedo.width,
        albedo.height,
        &emissive_from_maps(
            &TextureData {
                width: albedo.width,
                height: albedo.height,
                rgba: material,
            },
            &TextureData {
                width: albedo.width,
                height: albedo.height,
                rgba: extras.clone(),
            },
        ),
    );
    let coat_png = png_bytes(
        albedo.width,
        albedo.height,
        &coat_from_maps(
            &TextureData {
                width: albedo.width,
                height: albedo.height,
                rgba: height.clone(),
            },
            &TextureData {
                width: albedo.width,
                height: albedo.height,
                rgba: extras,
            },
        ),
    );
    let normal_png = png_bytes(
        albedo.width,
        albedo.height,
        &normal_map_from_height(&TextureData {
            width: albedo.width,
            height: albedo.height,
            rgba: height,
        }),
    );
    if orm_png.is_empty()
        || emissive_png.is_empty()
        || coat_png.is_empty()
        || normal_png.is_empty()
        || base_png.is_empty()
    {
        return Err(std::io::Error::other("failed to bake export textures"));
    }

    // Interleaved vertex stream: position (12B) + normal (12B) + uv (8B).
    let mut attrib = Vec::with_capacity(nv * 32);
    for i in 0..nv {
        let mut row = [0u8; 32];
        row[0..12].copy_from_slice(bytemuck::cast_slice(&mesh.positions[i].to_array()));
        row[12..24].copy_from_slice(bytemuck::cast_slice(&mesh.normals[i].to_array()));
        let uv = [mesh.uvs[i].0, mesh.uvs[i].1];
        row[24..32].copy_from_slice(bytemuck::cast_slice(&uv));
        attrib.extend_from_slice(&row);
    }
    let index_bytes = bytemuck::cast_slice(&mesh.indices);

    let align4 = |n: usize| (n + 3) & !3;
    let mut bin: Vec<u8> = Vec::new();
    let mut add = |data: &[u8]| -> (u32, u32) {
        bin.resize(align4(bin.len()), 0);
        let start = bin.len() as u32;
        bin.extend_from_slice(data);
        (start, data.len() as u32)
    };
    let (attr_off, attr_len) = add(&attrib);
    let (idx_off, idx_len) = add(index_bytes);
    let (i0_off, i0_len) = add(&base_png);
    let (i1_off, i1_len) = add(&orm_png);
    let (i2_off, i2_len) = add(&emissive_png);
    let (i3_off, i3_len) = add(&normal_png);
    let (i4_off, i4_len) = add(&coat_png);

    let (min, max) = {
        let mut mn = [f32::INFINITY; 3];
        let mut mx = [f32::NEG_INFINITY; 3];
        for p in &mesh.positions {
            mn[0] = mn[0].min(p.x);
            mn[1] = mn[1].min(p.y);
            mn[2] = mn[2].min(p.z);
            mx[0] = mx[0].max(p.x);
            mx[1] = mx[1].max(p.y);
            mx[2] = mx[2].max(p.z);
        }
        (mn, mx)
    };

    let nidx = mesh.indices.len();
    let json = serde_json::json!({
        "asset": { "version": "2.0", "generator": "pixforge" },
        "scene": 0,
        "scenes": [{ "nodes": [0], "name": "pixforge_scene" }],
        "nodes": [{ "mesh": 0, "name": "pixforge_mesh" }],
        "meshes": [{
            "primitives": [{
                "attributes": {
                    "POSITION": 0,
                    "NORMAL": 1,
                    "TEXCOORD_0": 2
                },
                "indices": 3,
                "material": 0,
                "mode": 4
            }]
        }],
        "materials": [{
            "name": "PixForgeMaterial",
            "pbrMetallicRoughness": {
                "baseColorTexture": { "index": 0 },
                "metallicRoughnessTexture": { "index": 1 }
            },
            "normalTexture": { "index": 3 },
            "emissiveTexture": { "index": 2 },
            "alphaMode": "BLEND",
            "doubleSided": true,
            "extensions": {
                "KHR_materials_clearcoat": {
                    "clearcoatFactor": 1.0,
                    "clearcoatTexture": { "index": 4 },
                    "clearcoatRoughnessFactor": 1.0,
                    "clearcoatRoughnessTexture": { "index": 4 }
                },
                "KHR_materials_ior": { "ior": 1.5 }
            }
        }],
        "extensionsUsed": ["KHR_materials_clearcoat", "KHR_materials_ior"],
        "buffers": [{ "byteLength": bin.len() }],
        "bufferViews": [
            { "buffer": 0, "byteOffset": attr_off, "byteLength": attr_len, "byteStride": 32, "target": 34962 },
            { "buffer": 0, "byteOffset": idx_off, "byteLength": idx_len, "target": 34963 },
            { "buffer": 0, "byteOffset": i0_off, "byteLength": i0_len },
            { "buffer": 0, "byteOffset": i1_off, "byteLength": i1_len },
            { "buffer": 0, "byteOffset": i2_off, "byteLength": i2_len },
            { "buffer": 0, "byteOffset": i3_off, "byteLength": i3_len },
            { "buffer": 0, "byteOffset": i4_off, "byteLength": i4_len }
        ],
        "accessors": [
            { "bufferView": 0, "byteOffset": 0,  "componentType": 5126, "count": nv,   "type": "VEC3", "min": min, "max": max },
            { "bufferView": 0, "byteOffset": 12, "componentType": 5126, "count": nv,   "type": "VEC3" },
            { "bufferView": 0, "byteOffset": 24, "componentType": 5126, "count": nv,   "type": "VEC2" },
            { "bufferView": 1, "byteOffset": 0,  "componentType": 5125, "count": nidx, "type": "SCALAR" }
        ],
        "images": [
            { "bufferView": 2, "mimeType": "image/png" },
            { "bufferView": 3, "mimeType": "image/png" },
            { "bufferView": 4, "mimeType": "image/png" },
            { "bufferView": 5, "mimeType": "image/png" },
            { "bufferView": 6, "mimeType": "image/png" }
        ],
        "samplers": [{
            "magFilter": 9729,
            "minFilter": 9987,
            "wrapS": 33071,
            "wrapT": 33071
        }],
        "textures": [
            { "sampler": 0, "source": 0 },
            { "sampler": 0, "source": 1 },
            { "sampler": 0, "source": 2 },
            { "sampler": 0, "source": 3 },
            { "sampler": 0, "source": 4 }
        ]
    });

    let json_bytes = serde_json::to_vec(&json).map_err(|e| std::io::Error::other(e.to_string()))?;
    let json_len = align4(json_bytes.len());
    let bin_len = align4(bin.len());
    let total = 12u32 + 8 + json_len as u32 + 8 + bin_len as u32;

    let mut out = Vec::with_capacity(total as usize);
    out.extend_from_slice(&0x46546C67u32.to_le_bytes()); // "glTF"
    out.extend_from_slice(&2u32.to_le_bytes()); // version
    out.extend_from_slice(&total.to_le_bytes());
    out.extend_from_slice(&(json_len as u32).to_le_bytes());
    out.extend_from_slice(&0x4E4F534Au32.to_le_bytes()); // "JSON"
    out.extend_from_slice(&json_bytes);
    out.resize(12 + 8 + json_len, 0x20); // pad JSON with spaces to 4-byte chunks
    out.extend_from_slice(&(bin_len as u32).to_le_bytes());
    out.extend_from_slice(&0x004E4942u32.to_le_bytes()); // "BIN\0"
    out.extend_from_slice(&bin);
    out.resize(total as usize, 0);

    std::fs::write(path, out)
}

/// A decoded equirectangular environment map: a full mip chain (box-filtered
/// on the CPU) of RGBA float16 rows, top-down, so the renderer can upload it
/// once and sample rough reflections with a roughness-driven texture LOD.
pub struct EnvironmentMips {
    pub width: u32,
    pub height: u32,
    /// One entry per mip level (level 0 = full resolution), row-major RGBA f16.
    pub mips: Vec<Vec<u8>>,
}

/// Lossy f32 → f16 bit conversion (f32 subnormals collapse to zero, NaN → inf);
/// plenty for environment radiance.
pub(crate) fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;
    match exp {
        // f32 subnormals: collapse to the signed zero.
        0..=110 => sign,
        // ±inf and NaN propagate.
        143..=255 => sign | 0x7c00,
        _ => {
            let e = exp - 127 + 15;
            if e >= 1 {
                let e = (e as u32) << 10;
                sign | e as u16 | (mant >> 13) as u16
            } else {
                // e <= 0: the value lies in the f16 subnormal range (below the
                // smallest normal f16, 2^-14). Each subnormal bit is worth
                // 2^-24, so scale the magnitude by 2^24 and truncate. This is
                // the branch that previously computed `-1 << 10 == 0xFC00` and
                // mapped every tiny positive value to f16 -inf.
                let sub = (v.abs() * 16777216.0) as u32;
                sign | sub.min(1023) as u16
            }
        }
    }
}

/// Loads an equirectangular environment map in any decodable image format
/// (Radiance .hdr, png, jpg, webp, …), peak-normalizing it so the visual
/// range matches the analytic sky (the environment-intensity slider still
/// scales it), and pre-builds the box-filtered mip chain for rough
/// reflections.
pub fn load_environment(path: &str) -> Result<EnvironmentMips, Box<dyn std::error::Error>> {
    let img = image::ImageReader::open(path)?.decode()?.to_rgb32f();
    let (w, h) = img.dimensions();
    let raw = img.as_raw();
    if w == 0 || h == 0 || raw.len() < (w * h * 3) as usize {
        return Err("environment map has zero size".into());
    }

    // Peak-normalize: one environment map's absolute radiance isn't meaningful
    // to the stylized shader; matching the analytic sky's ~[0,1] range keeps
    // the existing environment/exposure knobs working the same way.
    let count = (w * h) as usize;
    let mut cur: Vec<f32> = raw[..count * 3].to_vec();
    let mut peak: f32 = 0.0;
    for i in (0..count * 3).step_by(3) {
        peak = peak.max(cur[i]).max(cur[i + 1]).max(cur[i + 2]);
    }
    if peak > 0.0 && peak.is_finite() {
        let s = 1.0 / peak;
        for v in cur.iter_mut() {
            *v *= s;
        }
    }

    let mut levels: Vec<(u32, u32, Vec<f32>)> = Vec::new();
    let (mut cw, mut ch) = (w, h);
    loop {
        levels.push((cw, ch, cur.clone()));
        if cw <= 1 && ch <= 1 {
            break;
        }
        let (nw, nh) = ((cw / 2).max(1), (ch / 2).max(1));
        let mut down = vec![0.0f32; (nw * nh * 3) as usize];
        for y in 0..nh {
            for x in 0..nw {
                let mut acc = [0.0f32; 3];
                let mut cnt = 0u32;
                for sy in 0..2 {
                    for sx in 0..2 {
                        let px = (x * 2 + sx).min(cw - 1);
                        let py = (y * 2 + sy).min(ch - 1);
                        let i = ((py * cw + px) as usize) * 3;
                        acc[0] += cur[i];
                        acc[1] += cur[i + 1];
                        acc[2] += cur[i + 2];
                        cnt += 1;
                    }
                }
                let oi = ((y * nw + x) as usize) * 3;
                down[oi] = acc[0] / cnt as f32;
                down[oi + 1] = acc[1] / cnt as f32;
                down[oi + 2] = acc[2] / cnt as f32;
            }
        }
        cw = nw;
        ch = nh;
        cur = down;
    }

    let mips = levels
        .iter()
        .map(|(mw, mh, data)| {
            let mut bytes = Vec::with_capacity((mw * mh * 4) as usize * 2);
            for i in (0..data.len()).step_by(3) {
                for ch in 0..3 {
                    bytes.extend_from_slice(&f32_to_f16(data[i + ch]).to_le_bytes());
                }
                bytes.extend_from_slice(&f32_to_f16(1.0).to_le_bytes());
            }
            bytes
        })
        .collect();
    Ok(EnvironmentMips {
        width: w,
        height: h,
        mips,
    })
}

/// Loads a PNG (or any image `image` can decode) from `path`, scaling it with
/// nearest-neighbor resampling into a `w`x`h` atlas (the active layer's size).
pub fn load_image_into_atlas(
    path: &str,
    w: u32,
    h: u32,
) -> Result<TextureData, Box<dyn std::error::Error>> {
    let img = image::ImageReader::open(path)?.decode()?.to_rgba8();
    let (iw, ih) = img.dimensions();
    let mut out = TextureData {
        width: w,
        height: h,
        rgba: vec![0; (w * h * 4) as usize],
    };
    if w == 0 || h == 0 || iw == 0 || ih == 0 {
        return Ok(out);
    }
    let raw = img.as_raw();
    for y in 0..h {
        let sy = (y * ih) / h;
        for x in 0..w {
            let sx = (x * iw) / w;
            let si = (sy * iw + sx) as usize * 4;
            let di = (y * w + x) as usize * 4;
            out.rgba[di..di + 4].copy_from_slice(&raw[si..si + 4]);
        }
    }
    Ok(out)
}

/// Loads a PNG and normalizes it into a brush sprite: transparent pixels are
/// ignored; an opaque image has its luminance inverted (dark = strong paint).
/// Returns a `TextureData` whose RGBA is white with the computed coverage in
/// the alpha channel.
pub fn brush_sprite(path: &str) -> Result<TextureData, Box<dyn std::error::Error>> {
    let img = image::ImageReader::open(path)?.decode()?.to_rgba8();
    let (w, h) = img.dimensions();
    let raw = img.as_raw();
    let has_alpha = raw.iter().skip(3).step_by(4).any(|&a| a < 255);
    let mut rgba = vec![255u8; (w * h * 4) as usize];
    for i in 0..(w * h) as usize {
        let j = i * 4;
        let a = raw[j + 3] as f32 / 255.0;
        let cov = if has_alpha {
            a
        } else {
            let lum =
                (raw[j] as f32 * 0.2126 + raw[j + 1] as f32 * 0.7152 + raw[j + 2] as f32 * 0.0722)
                    / 255.0;
            (1.0 - lum).clamp(0.0, 1.0)
        };
        rgba[j + 3] = (cov * 255.0 + 0.5) as u8;
    }
    Ok(TextureData {
        width: w,
        height: h,
        rgba,
    })
}

/// Source-over blends `src` (scaled by `opacity`) into the accumulation atlas
/// `acc` (both straight alpha, same dimensions — mismatched layers are skipped).
/// Blend mode applied when a layer composite sits on the stack beneath it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    Normal,
    Multiply,
    Screen,
    Overlay,
}

impl BlendMode {
    pub const ALL: [BlendMode; 4] = [
        BlendMode::Normal,
        BlendMode::Multiply,
        BlendMode::Screen,
        BlendMode::Overlay,
    ];

    pub fn name(self) -> &'static str {
        match self {
            BlendMode::Normal => "Normal",
            BlendMode::Multiply => "Multiply",
            BlendMode::Screen => "Screen",
            BlendMode::Overlay => "Overlay",
        }
    }

    /// Compact label for a one-row blend picker inside the narrow Layers panel.
    pub fn short_name(self) -> &'static str {
        match self {
            BlendMode::Normal => "Nrm",
            BlendMode::Multiply => "Mul",
            BlendMode::Screen => "Scr",
            BlendMode::Overlay => "Ovl",
        }
    }

    pub fn to_byte(self) -> u8 {
        match self {
            BlendMode::Normal => 0,
            BlendMode::Multiply => 1,
            BlendMode::Screen => 2,
            BlendMode::Overlay => 3,
        }
    }

    pub fn from_byte(b: u8) -> Self {
        match b {
            1 => BlendMode::Multiply,
            2 => BlendMode::Screen,
            3 => BlendMode::Overlay,
            _ => BlendMode::Normal,
        }
    }
}

/// Photoshop-style blend function on straight 0..=1 channel values
/// (`cd` = backdrop/destination, `cs` = source).
fn blend_channel(mode: BlendMode, cd: f32, cs: f32) -> f32 {
    match mode {
        BlendMode::Normal => cs,
        BlendMode::Multiply => cs * cd,
        BlendMode::Screen => cs + cd - cs * cd,
        BlendMode::Overlay => {
            if cd <= 0.5 {
                2.0 * cs * cd
            } else {
                1.0 - 2.0 * (1.0 - cs) * (1.0 - cd)
            }
        }
    }
}

fn src_over(acc: &mut TextureData, src: &TextureData, opacity: f32, mode: BlendMode) {
    if src.width != acc.width || src.height != acc.height {
        return;
    }
    let (w, h) = (acc.width as usize, acc.height as usize);
    if w == 0 || h == 0 {
        return;
    }
    // Row-parallel source-over: every output row is independent, so rayon can
    // blend the whole atlas in ~h/(n_cpus) passes. Small atlases (tests,
    // thumbnails) avoid the dispatch overhead and stay scalar.
    if w * h >= 16 * 1024 {
        let row = w * 4;
        acc.rgba
            .par_chunks_exact_mut(row)
            .zip(src.rgba.par_chunks_exact(row))
            .for_each(|(ap, sp)| {
                for i in (0..row).step_by(4) {
                    let mut apx: [u8; 4] = ap[i..i + 4].try_into().unwrap();
                    let spx: [u8; 4] = sp[i..i + 4].try_into().unwrap();
                    src_over_px(&mut apx, &spx, opacity, mode);
                    ap[i..i + 4].copy_from_slice(&apx);
                }
            });
    } else {
        for (ap, sp) in acc.rgba.chunks_exact_mut(4).zip(src.rgba.chunks_exact(4)) {
            let (ap, sp): (&mut [u8; 4], &[u8; 4]) =
                (ap.try_into().unwrap(), sp.try_into().unwrap());
            src_over_px(ap, sp, opacity, mode);
        }
    }
}

/// Composites one source texel over one destination texel (straight alpha,
/// source-over) applying `opacity` to the source and `mode` to its rgb.
fn src_over_px(dst: &mut [u8; 4], src: &[u8; 4], opacity: f32, mode: BlendMode) {
    let sa = src[3] as f32 / 255.0 * opacity;
    if sa <= 0.0 {
        return;
    }
    let da = dst[3] as f32 / 255.0;
    let oa = sa + da * (1.0 - sa);
    if oa <= 0.0 {
        dst.copy_from_slice(&[0, 0, 0, 0]);
        return;
    }
    for c in 0..3 {
        let s = src[c] as f32 / 255.0;
        let d = dst[c] as f32 / 255.0;
        let bs = blend_channel(mode, d, s);
        let oc = (bs * sa + d * da * (1.0 - sa)) / oa;
        dst[c] = (oc * 255.0).round().clamp(0.0, 255.0) as u8;
    }
    dst[3] = (oa * 255.0).round().clamp(0.0, 255.0) as u8;
}

/// Blends one layer's scalar material value onto the accumulating material
/// map. The coverage per texel is the *paint* alpha from the layer's atlas
/// (times its opacity) — the standard non-blend-mode source-over mix on every
/// data channel. Unlike `src_over`, the map's 4th byte (ambient occlusion) is
/// data, not alpha, so it gets the same mix instead of becoming coverage.
fn src_over_material(acc: &mut TextureData, cover: &TextureData, rgba: [u8; 4], opacity: f32) {
    if cover.width != acc.width || cover.height != acc.height {
        return;
    }
    let (w, h) = (acc.width as usize, acc.height as usize);
    if w == 0 || h == 0 {
        return;
    }
    let row = w * 4;
    if w * h >= 16 * 1024 {
        // Row-parallel (independent output rows) for large atlases.
        acc.rgba
            .par_chunks_exact_mut(row)
            .zip(cover.rgba.par_chunks_exact(row))
            .for_each(|(ap, sp)| {
                for i in (0..row).step_by(4) {
                    let mut apx: [u8; 4] = ap[i..i + 4].try_into().unwrap();
                    let sa = sp[i + 3] as f32 / 255.0 * opacity;
                    if sa <= 0.0 {
                        continue;
                    }
                    for c in 0..4 {
                        let s = rgba[c] as f32 / 255.0;
                        let d = apx[c] as f32 / 255.0;
                        apx[c] = ((s * sa + d * (1.0 - sa)) * 255.0)
                            .round()
                            .clamp(0.0, 255.0) as u8;
                    }
                    ap[i..i + 4].copy_from_slice(&apx);
                }
            });
    } else {
        for (ap, sp) in acc.rgba.chunks_exact_mut(4).zip(cover.rgba.chunks_exact(4)) {
            let ap: &mut [u8; 4] = ap.try_into().unwrap();
            let sa = sp[3] as f32 / 255.0 * opacity;
            if sa <= 0.0 {
                continue;
            }
            for c in 0..4 {
                let s = rgba[c] as f32 / 255.0;
                let d = ap[c] as f32 / 255.0;
                ap[c] = ((s * sa + d * (1.0 - sa)) * 255.0)
                    .round()
                    .clamp(0.0, 255.0) as u8;
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TextureData {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// One stacked albedo atlas. All layers of a mesh share the same atlas size;
/// the renderer shows the flattened composite of the visible layers.
#[derive(Debug, Clone)]
pub struct Layer {
    pub name: String,
    pub visible: bool,
    /// 0..=1 applied during compositing (source-over).
    pub opacity: f32,
    /// How this layer's rgb merges with the stack below it.
    pub blend: BlendMode,
    /// Protected layer: content and property edits are rejected. Toggling
    /// visibility and renaming are still allowed.
    pub locked: bool,
    /// Per-layer surface material. These are scalar sliders — not painted
    /// per-texel maps — and they follow the same source-over stacking as the
    /// albedo: wherever a layer covers the model, its roughness/metallic/
    /// emissive/AO shape the shading (an opaque stroke fully takes over, a
    /// 50%-opacity stroke blends halfway). Stored 0..=1 except `emissive`,
    /// which is a multiplier on the albedo and may bloom past white.
    pub roughness: f32,
    pub metallic: f32,
    pub emissive: f32,
    pub ambient_occlusion: f32,
    /// Signed surface height carried by this layer's paint, -1 (carved/recessed)
    /// to 1 (raised). Composited into a separate height atlas (like the
    /// material map) and used by the shader to perturb the normal — 0 leaves
    /// the surface untouched.
    pub height: f32,
    /// Exaggeration of this layer's height: how strongly its painted texels
    /// perturb the surface normal. 0 disables the bump even where height is
    /// set. Data channel of the height atlas (G, stored /8 so it packs into a
    /// u8), like the per-layer material values.
    pub bump_strength: f32,
    /// 0..=1 clearcoat layer intensity (an extra GGX lobe on top of the base
    /// material, per the glTF KHR_materials_clearcoat layering). Composited
    /// into the height atlas B channel together with the per-layer height.
    pub clearcoat: f32,
    /// 0..=1 smoothness of the clearcoat lobe (lower = glossier coat).
    /// Composited into the extras atlas R channel (defaults to a satin 0.6).
    pub clearcoat_roughness: f32,
    /// Index-of-refraction of the clearcoat/dielectric layer, 1.0..=2.5. Drives
    /// the dielectric fresnel f0 = ((ior-1)/(ior+1))^2 (1.5 -> 0.04, matching
    /// the historical hardcoded value). Composited into the height atlas A
    /// channel, encoded (ior - 1.0) / 1.5.
    pub specular_ior: f32,
    /// Tint of this layer's emission, applied where the emissive intensity
    /// (0..3 scalar above) is positive: glow = mix(albedo, tint, strength) * strength.
    /// Composited into the extras atlas G/B/A channels (RGB, straight sRGB).
    pub emissive_color: [f32; 3],
    pub texture: TextureData,
}

impl Layer {
    pub fn new(name: impl Into<String>, texture: TextureData) -> Self {
        Self {
            name: name.into(),
            visible: true,
            opacity: 1.0,
            blend: BlendMode::Normal,
            locked: false,
            roughness: 0.55,
            metallic: 0.0,
            emissive: 0.0,
            ambient_occlusion: 1.0,
            height: 0.0,
            bump_strength: 2.0,
            clearcoat: 0.0,
            clearcoat_roughness: 0.6,
            specular_ior: 1.5,
            emissive_color: [1.0, 1.0, 1.0],
            texture,
        }
    }

    pub fn blank(name: impl Into<String>, width: u32, height: u32, fill: [u8; 4]) -> Self {
        Layer::new(name, blank_atlas(width, height, fill))
    }
}

#[derive(Debug, Clone, Default)]
pub struct MeshData {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<(f32, f32)>,
    pub indices: Vec<u32>,
    /// Stacked albedo atlases (bottom = index 0), all sharing one atlas size.
    pub layers: Vec<Layer>,
    /// Index of the layer that receives paint / erase / fill edits.
    pub active_layer: usize,
    /// Inclusive texel bounding `(x0, y0, x1, y1)` of texels edited since the
    /// last GPU upload (`None` = nothing pending, do a full upload). Set by
    /// paint/erase/fill so a frame only recomposites and re-uploads the region
    /// the brush actually touched instead of the whole atlas.
    pub dirty: Option<(u32, u32, u32, u32)>,
}

impl MeshData {
    pub fn with_texture(mut self, texture: TextureData) -> Self {
        self.layers = vec![Layer::new("Layer 1", texture)];
        self.active_layer = 0;
        self
    }

    pub fn active_layer_texture(&self) -> Option<&TextureData> {
        self.layers.get(self.active_layer).map(|l| &l.texture)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn active_layer_texture_mut(&mut self) -> Option<&mut TextureData> {
        self.layers
            .get_mut(self.active_layer)
            .map(|l| &mut l.texture)
    }

    /// Composites the visible layers (bottom to top, source-over, each scaled
    /// by its opacity) into a single atlas for rendering / export / preview.
    ///
    /// Returns `None` only when the mesh has no layers at all (rendered as the
    /// untextured white fallback). A mesh with layers always yields a composite
    /// — even a fully transparent one, which renders as a see-through hole.
    pub fn flattened_atlas(&self) -> Option<TextureData> {
        let visible: Vec<&Layer> = self
            .layers
            .iter()
            .filter(|l| l.visible && l.opacity > 0.0)
            .collect();
        let Some(&first) = visible.first() else {
            if self.layers.is_empty() {
                return None;
            }
            let tex = &self.layers[0].texture;
            return Some(TextureData {
                width: tex.width,
                height: tex.height,
                rgba: vec![0; (tex.width * tex.height * 4) as usize],
            });
        };
        // Exactly one visible layer at full opacity with the plain Normal blend
        // *is* the composite: source-over onto a transparent bottom copies the
        // source unchanged (including its alpha). Returning the layer's own
        // pixels avoids a full-atlas allocation + blend pass for the common
        // single-layer mesh.
        if visible.len() == 1 && first.opacity == 1.0 && first.blend == BlendMode::Normal {
            return Some(first.texture.clone());
        }
        let (w, h) = (first.texture.width, first.texture.height);
        let mut acc = TextureData {
            width: w,
            height: h,
            rgba: vec![0; (w * h * 4) as usize],
        };
        for layer in visible {
            src_over(&mut acc, &layer.texture, layer.opacity, layer.blend);
        }
        Some(acc)
    }

    /// Composites the per-layer material parameters into one RGBA atlas, the
    /// "material map": R = roughness, G = metallic, B = emissive (scaled so
    /// the shader multiplies back: stored value * 3 = emissive), A = ambient
    /// occlusion. One layer's scalar material follows source-over at each
    /// texel, weighted by that layer's paint coverage (its albedo alpha ×
    /// opacity): an opaque stroke fully takes over the material underneath, a
    /// 50%-opacity stroke blends halfway, unpainted texels keep the material
    /// below them. The default material seeds the whole sheet, so surfaces the
    /// stack never covers read as the default. Albado blend modes do not touch
    /// the material map — these are physical properties, not colors.
    pub fn flattened_material_atlas(&self) -> Option<TextureData> {
        let first = self.layers.iter().find(|l| l.visible && l.opacity > 0.0)?;
        let (w, h) = (first.texture.width, first.texture.height);
        if w == 0 || h == 0 {
            return None;
        }
        // Default material seed (matches render::Material::default's surface
        // params): roughness 0.55, metallic 0, emissive 0, ao 1.
        let default = [0.55f32, 0.0, 0.0, 1.0];
        let rgba = default
            .iter()
            .cycle()
            .take((w * h * 4) as usize)
            .map(|v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
            .collect();
        let mut acc = TextureData {
            width: w,
            height: h,
            rgba,
        };
        for layer in self.layers.iter().filter(|l| l.visible && l.opacity > 0.0) {
            if layer.texture.width != w || layer.texture.height != h {
                continue;
            }
            let texel = [
                (layer.roughness * 255.0).round().clamp(0.0, 255.0) as u8,
                (layer.metallic * 255.0).round().clamp(0.0, 255.0) as u8,
                (layer.emissive / 3.0 * 255.0).round().clamp(0.0, 255.0) as u8,
                (layer.ambient_occlusion * 255.0).round().clamp(0.0, 255.0) as u8,
            ];
            src_over_material(&mut acc, &layer.texture, texel, layer.opacity);
        }
        Some(acc)
    }

    /// Composites the per-layer height/bump into a separate RGBA atlas: R
    /// stores a signed height encoded as ((height + 1) * 0.5) so that 0
    /// (flat) maps to the byte value 128 and -1/+1 map to 0/255; G stores
    /// that layer's bump strength (/8 for the u8, same trick as emissive /3).
    /// B stores the layer's clearcoat intensity (0..=1) and A the specular IOR
    /// (encoded (ior - 1.0) / 1.5 so 1.0 -> 0 and 2.5 -> 255; 1.5 -> 85).
    /// Same source-over semantics as the material map: a layer's values apply
    /// only where its paint covers the surface, blended by opacity, and the
    /// sheet seeds to flat 0 (R = 128), no coat (B = 0) and the neutral IOR of
    /// 1.5 (A = 85). The shader decodes R to a signed height, takes its
    /// gradient, and scales it by the per-texel strength G.
    pub fn flattened_height_atlas(&self) -> Option<TextureData> {
        let first = self.layers.iter().find(|l| l.visible && l.opacity > 0.0)?;
        let (w, h) = (first.texture.width, first.texture.height);
        if w == 0 || h == 0 {
            return None;
        }
        // Seed: flat surface (signed height 0 → byte 128), no clearcoat, the
        // neutral 1.5 IOR (85).
        let mut acc = TextureData {
            width: w,
            height: h,
            rgba: [128, 0, 0, 85].repeat((w * h) as usize),
        };
        for layer in self.layers.iter().filter(|l| l.visible && l.opacity > 0.0) {
            if layer.texture.width != w || layer.texture.height != h {
                continue;
            }
            let h_byte = ((layer.height + 1.0) * 0.5 * 255.0)
                .round()
                .clamp(0.0, 255.0) as u8;
            let s_byte = (layer.bump_strength / 8.0 * 255.0)
                .round()
                .clamp(0.0, 255.0) as u8;
            let cc_byte = (layer.clearcoat * 255.0).round().clamp(0.0, 255.0) as u8;
            let ior_byte = ((layer.specular_ior - 1.0) / 1.5 * 255.0)
                .round()
                .clamp(0.0, 255.0) as u8;
            src_over_material(
                &mut acc,
                &layer.texture,
                [h_byte, s_byte, cc_byte, ior_byte],
                layer.opacity,
            );
        }
        Some(acc)
    }

    /// Composites the per-layer clearcoat roughness (R) and emissive color
    /// (G/B/A, straight sRGB, white = no tint) into an RGBA atlas with the same
    /// source-over semantics as the material and height maps. Seeds to a satin
    /// 0.6 coat roughness and an untinted white emission.
    pub fn flattened_extras_atlas(&self) -> Option<TextureData> {
        let first = self.layers.iter().find(|l| l.visible && l.opacity > 0.0)?;
        let (w, h) = (first.texture.width, first.texture.height);
        if w == 0 || h == 0 {
            return None;
        }
        // Seed: satin 0.6 coat roughness, untinted white emission (matches the
        // material-map seeding, which is independent of the layer values).
        let mut acc = TextureData {
            width: w,
            height: h,
            rgba: [153, 255, 255, 255].repeat((w * h) as usize),
        };
        for layer in self.layers.iter().filter(|l| l.visible && l.opacity > 0.0) {
            if layer.texture.width != w || layer.texture.height != h {
                continue;
            }
            let texel = [
                (layer.clearcoat_roughness * 255.0)
                    .round()
                    .clamp(0.0, 255.0) as u8,
                (layer.emissive_color[0].clamp(0.0, 1.0) * 255.0).round() as u8,
                (layer.emissive_color[1].clamp(0.0, 1.0) * 255.0).round() as u8,
                (layer.emissive_color[2].clamp(0.0, 1.0) * 255.0).round() as u8,
            ];
            src_over_material(&mut acc, &layer.texture, texel, layer.opacity);
        }
        Some(acc)
    }

    /// Composites only the `w`x`h` texel region starting at `(x0, y0)` into a
    /// small atlas (source-over, same rules as `flattened_atlas`).
    ///
    /// Used with `Self::dirty` after a stroke: a frame recomposites just the
    /// texels the brush touched instead of the whole atlas. The region is
    /// clamped to the atlas bounds; with no layers it returns `None`.
    pub fn flattened_atlas_region(&self, x0: u32, y0: u32, w: u32, h: u32) -> Option<TextureData> {
        let first = self.layers.first()?;
        let (tw, th) = (first.texture.width, first.texture.height);
        if w == 0 || h == 0 || tw == 0 || th == 0 {
            return Some(TextureData {
                width: w,
                height: h,
                rgba: vec![],
            });
        }
        let x0 = x0.min(tw - 1);
        let y0 = y0.min(th - 1);
        let ww = w.min(tw - x0);
        let hh = h.min(th - y0);
        let mut acc = TextureData {
            width: ww,
            height: hh,
            rgba: vec![0; (ww * hh * 4) as usize],
        };
        for layer in self.layers.iter().filter(|l| l.visible && l.opacity > 0.0) {
            // Guard against mismatched layer sizes the same way `flattened_atlas`
            // does: a layer with different dimensions is skipped, never indexed
            // with the first layer's stride (which would read past its buffer).
            if layer.texture.width != tw || layer.texture.height != th {
                continue;
            }
            let src = &layer.texture;
            let row_bytes = (ww * 4) as usize;
            // Each output row only ever touches that row's destination bytes, so
            // large dirty rects blend row-parallel; small ones (most strokes)
            // stay scalar to avoid dispatch overhead.
            if ww * hh >= 16 * 1024 {
                acc.rgba
                    .par_chunks_exact_mut(row_bytes)
                    .enumerate()
                    .for_each(|(yy, dp)| {
                        let src_row = ((y0 + yy as u32) * tw + x0) as usize * 4;
                        for xx in 0..ww as usize {
                            let si = src_row + xx * 4;
                            let mut dpx =
                                [dp[xx * 4], dp[xx * 4 + 1], dp[xx * 4 + 2], dp[xx * 4 + 3]];
                            let sp = [
                                src.rgba[si],
                                src.rgba[si + 1],
                                src.rgba[si + 2],
                                src.rgba[si + 3],
                            ];
                            src_over_px(&mut dpx, &sp, layer.opacity, layer.blend);
                            dp[xx * 4..xx * 4 + 4].copy_from_slice(&dpx);
                        }
                    });
            } else {
                for yy in 0..hh {
                    let src_row = ((y0 + yy) * tw + x0) as usize * 4;
                    let dst_row = (yy * ww) as usize * 4;
                    for xx in 0..ww {
                        let si = src_row + xx as usize * 4;
                        let di = dst_row + xx as usize * 4;
                        let sp = [
                            src.rgba[si],
                            src.rgba[si + 1],
                            src.rgba[si + 2],
                            src.rgba[si + 3],
                        ];
                        let mut dp = [
                            acc.rgba[di],
                            acc.rgba[di + 1],
                            acc.rgba[di + 2],
                            acc.rgba[di + 3],
                        ];
                        src_over_px(&mut dp, &sp, layer.opacity, layer.blend);
                        acc.rgba[di..di + 4].copy_from_slice(&dp);
                    }
                }
            }
        }
        Some(acc)
    }

    /// Builds a low-poly UV sphere (positions, normals, UVs, indices).
    ///
    /// The seam column duplicates its vertices (u=1 column = u=0 column in
    /// world space) so the texture wraps cleanly with fully covered texels
    /// instead of a stretched, degenerate column at the wrap.
    pub fn uv_sphere(radius: f32, rows: u32, cols: u32) -> Self {
        let pi = std::f32::consts::PI;
        // One extra column per ring: [0, cols] u values, where column `cols`
        // is the duplicated seam (world position of column 0, u = 1.0).
        let ring_verts = cols + 1;
        let verts = ((rows + 1) * ring_verts) as usize;
        let mut positions = Vec::with_capacity(verts);
        let mut normals = Vec::with_capacity(verts);
        let mut uvs = Vec::with_capacity(verts);

        for i in 0..=rows {
            let theta = pi * (i as f32 / rows as f32);
            let (st, ct) = theta.sin_cos();
            for j in 0..ring_verts {
                // The last (duplicated) column maps back to column 0's angles.
                let jc = (j % cols) as f32 / cols as f32;
                let phi = 2.0 * pi * jc;
                let (sp, cp) = phi.sin_cos();
                let n = Vec3::new(st * cp, ct, st * sp);
                positions.push(n * radius);
                normals.push(n);
                uvs.push((j as f32 / cols as f32, i as f32 / rows as f32));
            }
        }

        let mut indices = Vec::with_capacity((rows * cols * 6) as usize);
        for i in 0..rows {
            for j in 0..cols {
                let a = i * ring_verts + j;
                let b = (i + 1) * ring_verts + j;
                let c = (i + 1) * ring_verts + j + 1;
                let d = i * ring_verts + j + 1;
                indices.extend_from_slice(&[a, d, b, b, d, c]);
            }
        }

        Self {
            positions,
            normals,
            uvs,
            indices,
            layers: vec![],
            active_layer: 0,
            dirty: None,
        }
    }
}

/// Default albedo texture shown on the startup sphere: a classic two-tone
/// checkerboard so the UV layout and distortion read clearly at a glance.
pub fn default_albedo() -> TextureData {
    const S: u32 = 512;
    const CELLS: u32 = 8;
    let cell = S / CELLS;

    let light = [246u8, 241, 232];
    let dark = [205u8, 198, 186];

    let mut rgba = Vec::with_capacity((S * S * 4) as usize);
    for y in 0..S {
        for x in 0..S {
            if (x / cell + y / cell).is_multiple_of(2) {
                rgba.extend_from_slice(&dark);
            } else {
                rgba.extend_from_slice(&light);
            }
            rgba.push(255);
        }
    }

    TextureData {
        width: S,
        height: S,
        rgba,
    }
}

/// Resamples the atlas so its longest side becomes `longest_side`, preserving
/// aspect ratio. Nearest-neighbor sampling keeps the pixel-art look (no blur
/// when up-scaling, no bleeding when down-scaling).
pub fn resize_atlas(tex: &TextureData, longest_side: u32) -> TextureData {
    let longest_side = longest_side.max(1);
    let max_dim = tex.width.max(tex.height).max(1);
    let scale = longest_side as f32 / max_dim as f32;
    let new_w = (tex.width as f32 * scale).round().max(1.0) as u32;
    let new_h = (tex.height as f32 * scale).round().max(1.0) as u32;
    let mut rgba = vec![0u8; (new_w * new_h * 4) as usize];
    for y in 0..new_h {
        for x in 0..new_w {
            let src_x = ((x as f32 + 0.5) / scale - 0.5)
                .round()
                .clamp(0.0, tex.width as f32 - 1.0) as u32;
            let src_y = ((y as f32 + 0.5) / scale - 0.5)
                .round()
                .clamp(0.0, tex.height as f32 - 1.0) as u32;
            let si = ((src_y * tex.width + src_x) * 4) as usize;
            let di = ((y * new_w + x) * 4) as usize;
            rgba[di..di + 4].copy_from_slice(&tex.rgba[si..si + 4]);
        }
    }
    TextureData {
        width: new_w,
        height: new_h,
        rgba,
    }
}

/// Creates a new flat-colored atlas of the given size.
pub fn blank_atlas(width: u32, height: u32, fill: [u8; 4]) -> TextureData {
    let (w, h) = (width.max(1), height.max(1));
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    for px in rgba.chunks_exact_mut(4) {
        px.copy_from_slice(&fill);
    }
    TextureData {
        width: w,
        height: h,
        rgba,
    }
}

pub enum LoadedModel {
    Mesh(MeshData),
    Invalid,
}

fn next_pow2(v: u32) -> u32 {
    v.max(1).next_power_of_two()
}

/// Converts gltf decoded image data into RGBA8.
fn to_rgba(img: &gltf::image::Data) -> Option<(u32, u32, Vec<u8>)> {
    let n = (img.width as usize) * (img.height as usize);
    let out = match img.format {
        gltf::image::Format::R8G8B8A8 => img.pixels.clone(),
        gltf::image::Format::R8G8B8 => {
            let mut out = Vec::with_capacity(n * 4);
            for px in img.pixels.chunks_exact(3) {
                out.extend_from_slice(&[px[0], px[1], px[2], 255]);
            }
            out
        }
        gltf::image::Format::R8 => {
            let mut out = Vec::with_capacity(n * 4);
            for &g in &img.pixels {
                out.extend_from_slice(&[g, g, g, 255]);
            }
            out
        }
        _ => return None,
    };
    Some((img.width, img.height, out))
}

pub fn load_gltf(path: &str) -> LoadedModel {
    let (document, buffers, images) = match gltf::import(path) {
        Ok(v) => v,
        Err(_) => return LoadedModel::Invalid,
    };

    // Pass 1: collect the distinct base-color images used by any primitive.
    let mut slot_of_image: HashMap<usize, usize> = HashMap::new();
    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            let tex = primitive
                .material()
                .pbr_metallic_roughness()
                .base_color_texture()
                .map(|info| info.texture());
            if let Some(tex) = tex {
                let img = tex.source().index();
                if img < images.len() {
                    let next = slot_of_image.len();
                    slot_of_image.entry(img).or_insert(next);
                }
            }
        }
    }

    // Decode every used image into RGBA8 and pack it into a power-of-two atlas.
    let n_slots = slot_of_image.len();
    let mut layouts: Vec<Option<(u32, u32, Vec<u8>)>> = Vec::with_capacity(n_slots);
    let mut max_dim = 0u32;
    if n_slots > 0 {
        let mut pairs: Vec<(usize, usize)> = slot_of_image.iter().map(|(k, v)| (*k, *v)).collect();
        pairs.sort_by_key(|(_, slot)| *slot);
        for (img_idx, _slot) in pairs {
            match images.get(img_idx) {
                Some(data) => match to_rgba(data) {
                    Some((w, h, rgba)) => {
                        max_dim = max_dim.max(w).max(h);
                        layouts.push(Some((w, h, rgba)));
                    }
                    None => layouts.push(None),
                },
                None => layouts.push(None),
            }
        }
    }

    let mut atlas_texture: Option<TextureData> = None;
    let mut slot_regions: Vec<(u32, u32, u32, u32)> = Vec::with_capacity(n_slots); // (x, y, w, h)
    if n_slots > 0 {
        let cell = next_pow2(max_dim.max(1));
        let cols = (n_slots as f32).sqrt().ceil() as u32;
        let rows = (n_slots as u32).div_ceil(cols);
        let atlas_w = next_pow2(cols * cell);
        let atlas_h = next_pow2(rows * cell);
        if atlas_w > 8192 || atlas_h > 8192 {
            return LoadedModel::Invalid;
        }

        let mut atlas = vec![0u8; (atlas_w * atlas_h * 4) as usize];
        for (slot, layout) in layouts.iter().enumerate() {
            let x = (slot as u32 % cols) * cell;
            let y = (slot as u32 / cols) * cell;
            match layout {
                Some((w, h, rgba)) => {
                    for row in 0..*h {
                        let src = (row * *w * 4) as usize;
                        let dst = ((y + row) * atlas_w + x) as usize * 4;
                        atlas[dst..dst + (*w as usize) * 4]
                            .copy_from_slice(&rgba[src..src + (*w as usize) * 4]);
                    }
                    slot_regions.push((x, y, *w, *h));
                }
                None => slot_regions.push((x, y, 1, 1)),
            }
        }
        atlas_texture = Some(TextureData {
            width: atlas_w,
            height: atlas_h,
            rgba: atlas,
        });
    }

    // Pass 2: fill mesh data, applying the scene-graph world transform so
    // multi-node models (separate body parts, rig bones) render in their
    // proper positions.  Skinned meshes are not supported; when a node has a
    // skin the vertices are left in the bind pose, which is still a large
    // improvement over gluing every part at the local-space origin.
    let mut positions: Vec<Vec3> = Vec::new();
    let mut normals: Vec<Vec3> = Vec::new();
    let mut uvs: Vec<(f32, f32)> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    let aw = atlas_texture.as_ref().map(|t| t.width).unwrap_or(1) as f32;
    let ah = atlas_texture.as_ref().map(|t| t.height).unwrap_or(1) as f32;

    // Accumulate world matrices for every node in the default scene.
    let mut scene_nodes: Vec<(gltf::Node, glam::Mat4)> = Vec::new();
    if let Some(scene) = document.default_scene() {
        let mut stack: Vec<(gltf::Node, glam::Mat4)> =
            scene.nodes().map(|n| (n, glam::Mat4::IDENTITY)).collect();
        while let Some((node, parent)) = stack.pop() {
            let (t, r, s) = node.transform().decomposed();
            let local = glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::from_array(s),
                glam::Quat::from_array(r),
                glam::Vec3::from_array(t),
            );
            let world = parent * local;
            stack.extend(node.children().map(|c| (c, world)));
            scene_nodes.push((node, world));
        }
    }

    let push_primitive = |world: glam::Mat4,
                          primitive: gltf::Primitive,
                          positions: &mut Vec<Vec3>,
                          normals: &mut Vec<Vec3>,
                          uvs: &mut Vec<(f32, f32)>,
                          indices: &mut Vec<u32>|
     -> Result<(), ()> {
        let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
        let base = positions.len() as u32;

        let mut src_pos = match reader.read_positions() {
            Some(iter) => iter
                .map(|v| Vec3::new(v[0], v[1], v[2]))
                .collect::<Vec<_>>(),
            None => return Err(()),
        };
        let mut src_nrm = match reader.read_normals() {
            // A normal accessor whose count disagrees with the position count
            // would misalign every downstream array — treat it as absent rather
            // than swallowing a malformed mesh.
            Some(iter) => {
                let n = iter
                    .map(|v| Vec3::new(v[0], v[1], v[2]))
                    .collect::<Vec<_>>();
                if n.len() == src_pos.len() {
                    n
                } else {
                    vec![Vec3::ZERO; src_pos.len()]
                }
            }
            None => vec![Vec3::ZERO; src_pos.len()],
        };

        if world != glam::Mat4::IDENTITY {
            let normal_mat = glam::Mat3::from_mat4(world).inverse().transpose();
            for v in &mut src_pos {
                *v = world.transform_point3(*v);
            }
            for n in &mut src_nrm {
                *n = (normal_mat * *n).normalize_or_zero();
            }
        }

        // Captured before `src_pos` is moved into `positions` below; the UV
        // count guard and the no-UV fallback both need the vertex count.
        let src_len = src_pos.len();
        positions.extend(src_pos);
        normals.extend(src_nrm);

        // UV scale/offset for this primitive's atlas slot; identity if no atlas.
        let (ox, oy, sx, sy) = match primitive
            .material()
            .pbr_metallic_roughness()
            .base_color_texture()
            .map(|info| info.texture().source().index())
        {
            Some(img_idx) => match slot_of_image.get(&img_idx) {
                Some(slot) => {
                    let (x, y, w, h) = slot_regions[*slot];
                    (x as f32 / aw, y as f32 / ah, w as f32 / aw, h as f32 / ah)
                }
                None => (0.0, 0.0, 1.0, 1.0),
            },
            None => (0.0, 0.0, 1.0, 1.0),
        };

        match reader.read_tex_coords(0) {
            Some(uv) => {
                let uv = uv
                    .into_f32()
                    .map(|v| (v[0] * sx + ox, v[1] * sy + oy))
                    .collect::<Vec<_>>();
                // Guard against a UV accessor whose count disagrees with the
                // position count (malformed files): misaligned UVs make paint
                // land on the wrong texels, so fall back to zeros like a
                // missing accessor would.
                if uv.len() == src_len {
                    uvs.extend(uv);
                } else {
                    uvs.extend(std::iter::repeat_n((0.0, 0.0), src_len));
                }
            }
            None => uvs.extend(std::iter::repeat_n((0.0, 0.0), src_len)),
        }

        match reader.read_indices() {
            Some(ind) => indices.extend(ind.into_u32().map(|i| i + base)),
            None => {
                let n = reader.read_positions().map(|it| it.len()).unwrap_or(0);
                indices.extend(base..base + n as u32)
            }
        }
        Ok(())
    };

    if scene_nodes.is_empty() {
        // No scene graph: use identity for every mesh, like before.
        for mesh in document.meshes() {
            for primitive in mesh.primitives() {
                if push_primitive(
                    glam::Mat4::IDENTITY,
                    primitive,
                    &mut positions,
                    &mut normals,
                    &mut uvs,
                    &mut indices,
                )
                .is_err()
                {
                    return LoadedModel::Invalid;
                }
            }
        }
    } else {
        for (node, world) in &scene_nodes {
            let Some(mesh) = node.mesh() else {
                continue;
            };
            for primitive in mesh.primitives() {
                if push_primitive(
                    *world,
                    primitive,
                    &mut positions,
                    &mut normals,
                    &mut uvs,
                    &mut indices,
                )
                .is_err()
                {
                    return LoadedModel::Invalid;
                }
            }
        }
    }

    if positions.is_empty() {
        return LoadedModel::Invalid;
    }

    LoadedModel::Mesh(MeshData {
        positions,
        normals,
        uvs,
        indices,
        layers: atlas_texture
            .map(|tex| vec![Layer::new("Layer 1", tex)])
            .unwrap_or_default(),
        active_layer: 0,
        dirty: None,
    })
}

/// One island's packed box in the remapped atlas (normalized [0,1] UVs) plus
/// the UV extent it occupied before the remap — enough to re-bake the atlas
/// content into the new layout.
#[derive(Clone, Copy)]
pub(crate) struct IslandBox {
    /// Packed box in final UV space.
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    /// Old UV extent this island covered before the remap.
    pub u0: f32,
    pub v0: f32,
    pub du: f32,
    pub dv: f32,
}

/// Mirrors the texel rows/columns of `tex` in place: `flip_x` mirrors each
/// row left↔right, `flip_y` swaps top↔bottom rows. Both may be set; the RGB and
/// alpha channels flip together as whole pixels.
pub fn flip_texture(tex: &mut TextureData, flip_x: bool, flip_y: bool) {
    if !flip_x && !flip_y {
        return;
    }
    let w = tex.width as usize;
    let h = tex.height as usize;
    if w == 0 || h == 0 {
        return;
    }
    let mut flipped = vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let src = (y * w + x) * 4;
            let dx = if flip_x { w - 1 - x } else { x };
            let dy = if flip_y { h - 1 - y } else { y };
            let dst = (dy * w + dx) * 4;
            flipped[dst..dst + 4].copy_from_slice(&tex.rgba[src..src + 4]);
        }
    }
    tex.rgba = flipped;
}

/// Re-unwraps the mesh so every UV island gets a texture budget proportional to
/// its 3D surface area (uniform texel density): a big body part no longer
/// squanders the same atlas space as a tiny detail.
///
/// Islands are the connected components of the triangle graph (triangles glued
/// by shared vertices). Each island keeps its own UV-aspect ratio; a shelf
/// packer lays the islands out and a uniform scale fits the result into
/// `[0,1]²` — never overlapping, leaving the seam duplicates of wrapped meshes
/// intact.
///
/// Because the remap re-locates every island, the atlas *content* is re-baked
/// in the same call: each layer's textures are resampled so their pixels land
/// back under the same 3D surface (the painting survives the rewrap instead of
/// sliding off the model). `mesh.dirty` is set to the full atlas so the GPU
/// re-uploads everything. Callers must also re-upload the geometry
/// (`Renderer::set_mesh`) so the vertex buffer actually carries the new UVs.
///
/// Returns the island count (0 = empty mesh, 1 = single connected part where
/// there is nothing to repack and no texture change happens).
#[cfg_attr(not(test), allow(dead_code))] // used by the Remake UV tests only
pub fn remake_uv(mesh: &mut MeshData) -> usize {
    let n_tri = mesh.indices.len() / 3;
    if mesh.positions.is_empty() || n_tri == 0 {
        return 0;
    }
    let (w_tex, h_tex) = match mesh.active_layer_texture() {
        Some(t) if t.width > 0 && t.height > 0 => (t.width as f32, t.height as f32),
        _ => (256.0, 256.0),
    };

    // Triangle islands = connected components via shared vertex indices.
    let mut parent: Vec<usize> = (0..n_tri).collect();
    let mut vertex_first = vec![usize::MAX; mesh.positions.len()];
    for (tri, tri_idx) in mesh.indices.chunks_exact(3).enumerate() {
        for &vi in tri_idx {
            let vi = vi as usize;
            match vertex_first[vi] {
                usize::MAX => vertex_first[vi] = tri,
                other => union(&mut parent, tri, other),
            }
        }
    }
    // Compact union-find roots to 0..n_islands.
    let mut key_of = std::collections::HashMap::new();
    let island_of: Vec<usize> = (0..n_tri)
        .map(|t| {
            let root = find(&mut parent, t);
            let n = key_of.len();
            *key_of.entry(root).or_insert(n)
        })
        .collect();
    let n_islands = key_of.len();
    if n_islands <= 1 {
        // A single island already covers the atlas with uniform density and
        // there is no repack to do — UVs and texture stay untouched.
        return 1;
    }

    #[derive(Clone, Copy)]
    struct Island {
        area3d: f32,
        u0: f32,
        u1: f32,
        v0: f32,
        v1: f32,
    }
    let mut island = vec![
        Island {
            area3d: 0.0,
            u0: f32::INFINITY,
            u1: f32::NEG_INFINITY,
            v0: f32::INFINITY,
            v1: f32::NEG_INFINITY,
        };
        n_islands
    ];

    for (tri, tri_idx) in mesh.indices.chunks_exact(3).enumerate() {
        let (i0, i1, i2) = (
            tri_idx[0] as usize,
            tri_idx[1] as usize,
            tri_idx[2] as usize,
        );
        let (a, b, c) = (mesh.positions[i0], mesh.positions[i1], mesh.positions[i2]);
        let isl = &mut island[island_of[tri]];
        isl.area3d += (b - a).cross(c - a).length() * 0.5;
        for i in [i0, i1, i2] {
            let (u, v) = mesh.uvs[i];
            isl.u0 = isl.u0.min(u);
            isl.u1 = isl.u1.max(u);
            isl.v0 = isl.v0.min(v);
            isl.v1 = isl.v1.max(v);
        }
    }

    let area_total: f32 = island.iter().map(|i| i.area3d).sum();
    if area_total <= 0.0 {
        return 0;
    }

    // Target texel budget ∝ surface area; keep each island's current texture
    // aspect ratio ((Δu·W)/(Δv·H)), clamped so degenerate slivers stay sane.
    let mut boxes_px: Vec<(f32, f32, f32, f32, usize)> = Vec::new(); // x_px, y_px, w_px, h_px, island
    for (id, isl) in island.iter().enumerate() {
        let area_share = isl.area3d / area_total;
        let n_tex = (w_tex * h_tex * area_share).max(1.0);
        let (du, dv) = ((isl.u1 - isl.u0).abs(), (isl.v1 - isl.v0).abs());
        let aspect = if du > 1e-6 && dv > 1e-6 {
            ((du * w_tex) / (dv * h_tex)).clamp(1e-3, 1e3)
        } else {
            1.0
        };
        let h_px = (n_tex / aspect).sqrt();
        let w_px = (n_tex * aspect).sqrt();
        boxes_px.push((0.0, 0.0, w_px, h_px, id));
    }
    // Shelf packing, tallest first.
    boxes_px.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
    let use_w = w_tex;
    let mut cur_x = 0.0;
    let mut cur_y = 0.0;
    let mut shelf_h = 0.0;
    let mut used_w_max: f32 = 0.0;
    for b in &mut boxes_px {
        if cur_x + b.2 > use_w && cur_x > 0.0 {
            cur_x = 0.0;
            cur_y += shelf_h;
            shelf_h = 0.0;
        }
        b.0 = cur_x;
        b.1 = cur_y;
        cur_x += b.2;
        shelf_h = shelf_h.max(b.3);
        used_w_max = used_w_max.max(b.0 + b.2);
    }
    let total_h = cur_y + shelf_h;
    // Uniform scale so the sheet fits the atlas (grows if there is room).
    let s = (use_w / used_w_max.max(1.0)).min((h_tex / total_h.max(1.0)).max(0.0));
    let s = if s > 0.0 && s.is_finite() { s } else { 1.0 };

    // Remap every island into its packed box, recording the transform so the
    // atlas content can be re-baked afterwards.
    let uvs_orig = mesh.uvs.clone();
    let mut island_boxes: Vec<IslandBox> = Vec::with_capacity(n_islands);
    for &(bx, by, w_px, h_px, id) in &boxes_px {
        let isl = &island[id];
        let (du, dv) = (isl.u1 - isl.u0, isl.v1 - isl.v0);
        let w_norm = (w_px * s) / w_tex;
        let h_norm = (h_px * s) / h_tex;
        // The uniform sheet scale must shrink/grow the box *origins* too, not
        // just the sizes: scaling sizes around unscaled origins would leave the
        // sheet overflowing the atlas when s < 1 (and leave lopsided gutters).
        let x_norm = (bx * s) / w_tex;
        let y_norm = (by * s) / h_tex;
        island_boxes.push(IslandBox {
            x: x_norm,
            y: y_norm,
            w: w_norm,
            h: h_norm,
            u0: isl.u0,
            v0: isl.v0,
            du,
            dv,
        });
        for (tri, tri_idx) in mesh.indices.chunks_exact(3).enumerate() {
            if island_of[tri] != id {
                continue;
            }
            for &vi in tri_idx {
                let vi = vi as usize;
                // Read original UVs: shared vertices are visited more than once
                // and must always be remapped from the pre-remap position.
                let (u, v) = uvs_orig[vi];
                let u_n = if du.abs() > 1e-6 {
                    (u - isl.u0) / du
                } else {
                    0.0
                };
                let v_n = if dv.abs() > 1e-6 {
                    (v - isl.v0) / dv
                } else {
                    0.0
                };
                mesh.uvs[vi] = (x_norm + u_n * w_norm, y_norm + v_n * h_norm);
            }
        }
    }

    // Re-bake every layer's atlas into the new layout so the paint follows the
    // islands (a remap without this would scramble the texture over the mesh).
    let old_layers: Vec<TextureData> = mesh.layers.iter().map(|l| l.texture.clone()).collect();
    for (li, layer) in mesh.layers.iter_mut().enumerate() {
        let old = &old_layers[li];
        if old.width == 0 || old.height == 0 {
            continue;
        }
        layer.texture = refit_texture(&island_boxes, old, old.width, old.height);
    }

    // Full-atlas dirty rect in the app's (x0, y0, x1, y1) inclusive min/max
    // convention so `flush_paint_edit` re-uploads every texel. (Storing the raw
    // sheet dimensions as (w, h, 0, 0) would read as an inverted rect there and
    // collapse the region upload to one corner pixel.)
    let dirty = (
        0,
        0,
        (w_tex as u32).saturating_sub(1),
        (h_tex as u32).saturating_sub(1),
    );
    mesh.dirty = Some(dirty);
    n_islands
}

/// Resamples the old atlas content into the remapped layout: for every texel
/// that falls inside a packed island box, pull the texel the island's OLD UV
/// range mapped to; sheet gaps stay transparent.
fn refit_texture(boxes: &[IslandBox], old: &TextureData, w: u32, h: u32) -> TextureData {
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    if w == 0 || h == 0 {
        return TextureData {
            width: w,
            height: h,
            rgba,
        };
    }
    for py in 0..h {
        for px in 0..w {
            let u = (px as f32 + 0.5) / w as f32;
            let v = (py as f32 + 0.5) / h as f32;
            for b in boxes {
                if u >= b.x && u < b.x + b.w && v >= b.y && v < b.y + b.h {
                    let nu = if b.w > 1e-6 { (u - b.x) / b.w } else { 0.0 };
                    let nv = if b.h > 1e-6 { (v - b.y) / b.h } else { 0.0 };
                    let ou = b.u0 + nu * b.du;
                    let ov = b.v0 + nv * b.dv;
                    let di = (py as usize * w as usize + px as usize) * 4;
                    rgba[di..di + 4].copy_from_slice(&sample_bilinear(old, ou, ov));
                    break;
                }
            }
        }
    }
    TextureData {
        width: w,
        height: h,
        rgba,
    }
}

/// Bilinear sample of `tex` at normalized UVs `u`, `v` (clamped to the texel
/// centers). 1-texel textures return their sole pixel.
fn sample_bilinear(tex: &TextureData, u: f32, v: f32) -> [u8; 4] {
    let (dw, dh) = (tex.width, tex.height);
    if dw == 0 || dh == 0 {
        return [0, 0, 0, 0];
    }
    if dw == 1 && dh == 1 {
        return tex.rgba[..4].try_into().unwrap();
    }
    let fx = (u.clamp(0.0, 1.0) * dw as f32 - 0.5).max(0.0);
    let fy = (v.clamp(0.0, 1.0) * dh as f32 - 0.5).max(0.0);
    let x0 = fx.floor() as u32;
    let y0 = fy.floor() as u32;
    let x1 = (x0 + 1).min(dw - 1);
    let y1 = (y0 + 1).min(dh - 1);
    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;
    let at = |x: u32, y: u32| {
        let i = (y * dw + x) as usize * 4;
        &tex.rgba[i..i + 4]
    };
    let mut out = [0u8; 4];
    for (c, o) in out.iter_mut().enumerate() {
        let top = at(x0, y0)[c] as f32 * (1.0 - tx) + at(x1, y0)[c] as f32 * tx;
        let bot = at(x0, y1)[c] as f32 * (1.0 - tx) + at(x1, y1)[c] as f32 * tx;
        *o = (top * (1.0 - ty) + bot * ty).round().clamp(0.0, 255.0) as u8;
    }
    out
}

fn union(parent: &mut [usize], a: usize, b: usize) {
    let (ra, rb) = (find(parent, a), find(parent, b));
    if ra != rb {
        parent[ra] = rb;
    }
}

fn find(parent: &mut [usize], mut x: usize) -> usize {
    let mut root = x;
    while parent[root] != root {
        root = parent[root];
    }
    while parent[x] != root {
        let next = parent[x];
        parent[x] = root;
        x = next;
    }
    root
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;

    // Minimal glTF 2.0: one triangle with positions, normals, uv, u16 indices.
    // Buffer base64-encoded inline (data URI) so no external .bin is needed.
    const TRIANGE_B64: &str = "AAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAAAAAgD8AAAAAAAAAAAAAAAAAAIA/AAAAAAAAAAAAAIA/AAAAAAAAAAAAAIA/AAAAAAAAAAAAAIA/AAAAAAAAAAAAAIA/AAABAAIA";

    fn write_temp_gltf() -> std::path::PathBuf {
        let json = format!(
            r#"{{
  "asset": {{"version": "2.0"}},
  "scene": 0,
  "scenes": [{{"nodes": [0]}}],
  "nodes": [{{"mesh": 0}}],
  "meshes": [{{"primitives": [{{"attributes": {{"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}}, "indices": 3}}]}}],
  "buffers": [{{"uri": "data:application/octet-stream;base64,{B64}", "byteLength": 102}}],
  "bufferViews": [
    {{"buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962}},
    {{"buffer": 0, "byteOffset": 36, "byteLength": 36, "target": 34962}},
    {{"buffer": 0, "byteOffset": 72, "byteLength": 24, "target": 34962}},
    {{"buffer": 0, "byteOffset": 96, "byteLength": 6, "target": 34963}}
  ],
  "accessors": [
    {{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0,0,0], "max": [1,1,0]}},
    {{"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3"}},
    {{"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC2"}},
    {{"bufferView": 3, "componentType": 5123, "count": 3, "type": "SCALAR"}}
  ]
}}"#,
            B64 = TRIANGE_B64
        );

        let path = std::env::temp_dir().join(format!("pixforge_test_{}.gltf", std::process::id()));
        let mut f = File::create(&path).unwrap();
        f.write_all(json.as_bytes()).unwrap();
        path
    }

    #[test]
    fn loads_triangle_mesh() {
        let path = write_temp_gltf();
        match load_gltf(path.to_str().unwrap()) {
            LoadedModel::Mesh(m) => {
                assert_eq!(m.positions.len(), 3);
                assert_eq!(m.normals.len(), 3);
                assert_eq!(m.uvs.len(), 3);
                assert_eq!(m.indices, vec![0, 1, 2]);
                assert!((m.normals[0].z - 1.0).abs() < 1e-6);
                assert!(m.layers.is_empty(), "no material => no layers");
            }
            LoadedModel::Invalid => panic!("expected a valid mesh"),
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn rejects_missing_file() {
        match load_gltf("/nonexistent/pixforge.gltf") {
            LoadedModel::Invalid => {}
            LoadedModel::Mesh(_) => panic!("expected Invalid for missing file"),
        }
    }

    #[test]
    fn hdr_environment_round_trips_and_prebuilds_mips() {
        let (w, h) = (16u32, 8u32);
        let mut img = image::Rgb32FImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let px = img.get_pixel_mut(x, y);
                px.0 = [2.0, 1.0, 0.5];
            }
        }
        let path = std::env::temp_dir().join(format!("pixforge_hdri_{}.hdr", std::process::id()));
        img.save_with_format(&path, image::ImageFormat::Hdr)
            .unwrap();

        let env = load_environment(path.to_str().unwrap()).expect("load hdri environment");
        assert_eq!(env.width, w);
        assert_eq!(env.height, h);
        assert!(
            env.mips.len() >= 4,
            "chain should run down to 1x1, got {}",
            env.mips.len()
        );
        // Level 0 was peak-normalized (peak 2.0 → 1.0); f16(1.0) ≈ 0x3C00.
        let l0 = &env.mips[0];
        assert_eq!(l0.len(), (w * h * 4) as usize * 2);
        assert_eq!(
            &l0[0..2],
            b"\x00\x3c",
            "red channel should read ~1.0 after normalize"
        );
        assert_eq!(&l0[6..8], b"\x00\x3c", "alpha should be 1.0");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn plain_image_works_as_environment() {
        // A regular 8-bit PNG is not a Radiance map: the encode/decode path must
        // still yield a valid float16 mip chain (peak ≈ 0.5 scales to 1.0).
        let (w, h) = (8u32, 4u32);
        let mut img = image::RgbImage::new(w, h);
        for (_, _, px) in img.enumerate_pixels_mut() {
            *px = image::Rgb([128, 64, 32]);
        }
        let path = std::env::temp_dir().join(format!("pixforge_env_{}.png", std::process::id()));
        img.save(&path).unwrap();

        let env = load_environment(path.to_str().unwrap()).expect("load png environment");
        assert_eq!(env.width, w);
        assert_eq!(env.height, h);
        assert!(
            env.mips.len() >= 3,
            "chain should run down to 1x1, got {}",
            env.mips.len()
        );
        let l0 = &env.mips[0];
        fn f16_to_f32(bits: u16) -> f32 {
            f32::from_bits(
                (u32::from(bits & 0x8000) << 16)
                    | (u32::from(((bits >> 10) & 0x1f).saturating_add(112)) << 23)
                    | (u32::from(bits & 0x3ff) << 13),
            )
        }
        let red = f16_to_f32(u16::from_le_bytes([l0[0], l0[1]]));
        assert!(red > 0.99, "peak-normalized red should be ~1.0, got {red}");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn glb_export_roundtrips_geometry_and_layers() {
        let mut mesh = MeshData::uv_sphere(0.6, 10, 14);
        let (w, h) = (32u32, 32u32);
        let mut layer = Layer::blank("paint", w, h, [200, 20, 20, 255]);
        layer.roughness = 0.3;
        layer.metallic = 0.5;
        layer.height = 0.5;
        layer.bump_strength = 4.0;
        mesh.layers = vec![layer];
        mesh.active_layer = 0;

        let path = std::env::temp_dir().join(format!("pixforge_glb_{}.glb", std::process::id()));
        save_glb(path.to_str().unwrap(), &mesh).expect("save_glb");

        let bytes = std::fs::read(&path).expect("read glb");
        assert_eq!(&bytes[0..4], b"glTF");
        assert_eq!(&bytes[4..8], &2u32.to_le_bytes());
        assert_eq!(
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize,
            bytes.len(),
            "GLB total length must match the file"
        );

        match load_gltf(path.to_str().unwrap()) {
            LoadedModel::Mesh(m) => {
                assert_eq!(m.positions.len(), mesh.positions.len());
                assert_eq!(m.indices, mesh.indices);
                assert_eq!(m.layers.len(), 1, "baked base color should create a layer");
                let t = &m.layers[0].texture;
                assert!(!t.rgba.is_empty());
                // The painted center texel survives the PNG round-trip.
                let mid = t.rgba[((t.height / 2 * t.width + t.width / 2) as usize) * 4];
                assert!(
                    mid >= 150,
                    "center should stay reddish after export, got {mid}"
                );
            }
            LoadedModel::Invalid => panic!("exported glb failed to reload"),
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn default_sphere_is_solid() {
        let (rows, cols) = (12u32, 16u32);
        let s = MeshData::uv_sphere(0.6, rows, cols);
        // One duplicated seam column per ring.
        assert_eq!(s.positions.len(), ((rows + 1) * (cols + 1)) as usize);
        assert_eq!(s.positions.len(), s.normals.len());
        assert_eq!(s.positions.len(), s.uvs.len());
        assert_eq!(s.indices.len() % 3, 0);
        assert_eq!(s.indices.len(), (rows * cols * 6) as usize);
        for n in &s.normals {
            assert!((n.length() - 1.0).abs() < 1e-4, "normals must be unit");
        }
        // UVs cover the full [0,1]^2 square and never skip texel columns.
        let covers_full = s.uvs.iter().any(|&(u, _)| u >= 1.0)
            && s.uvs.iter().any(|&(u, _)| u <= 0.0)
            && s.uvs.iter().any(|&(_, v)| v <= 0.0)
            && s.uvs.iter().any(|&(_, v)| v >= 1.0);
        assert!(covers_full, "sphere UVs must span the whole atlas");
        // Every index references a valid vertex.
        assert!(s.indices.iter().all(|&i| (i as usize) < s.positions.len()));

        let tex = default_albedo();
        assert_eq!(tex.rgba.len(), (tex.width * tex.height * 4) as usize);
        assert!(tex.rgba.chunks_exact(4).all(|px| px[3] == 255));
    }

    #[test]
    fn resize_atlas_preserves_aspect_and_nearest_mapping() {
        // 4x2 -> longest side 8: becomes 8x4, aspect held.
        let src = TextureData {
            width: 4,
            height: 2,
            rgba: (0..(4 * 2 * 4)).map(|i| i as u8).collect::<Vec<u8>>(),
        };
        let big = resize_atlas(&src, 8);
        assert_eq!((big.width, big.height), (8, 4));
        // Nearest sampling: dst (0,0) maps to src (0,0), dst (7,0) to src (3,0).
        let px = |t: &TextureData, x: u32, y: u32| {
            let i = ((y * t.width + x) * 4) as usize;
            t.rgba[i..i + 4].to_vec()
        };
        assert_eq!(px(&big, 0, 0), px(&src, 0, 0));
        assert_eq!(px(&big, 7, 0), px(&src, 3, 0));
        assert_eq!(px(&big, 0, 3), px(&src, 0, 1));
        assert_eq!(px(&big, 7, 3), px(&src, 3, 1));

        // Down-scaling back to longest side 2 gives 2x1.
        let small = resize_atlas(&src, 2);
        assert_eq!((small.width, small.height), (2, 1));

        // Solid fields stay solid through round-trips.
        let solid = blank_atlas(16, 16, [7, 13, 29, 255]);
        let r = resize_atlas(&solid, 5);
        assert!(
            r.rgba.chunks_exact(4).all(|px| px[..4] == [7, 13, 29, 255]),
            "resized atlas must stay solid"
        );
        assert_eq!((r.width, r.height), (5, 5));
    }

    #[test]
    fn blank_atlas_has_requested_size_and_fill() {
        let b = blank_atlas(32, 16, [1, 2, 3, 4]);
        assert_eq!((b.width, b.height), (32, 16));
        assert_eq!(b.rgba.len(), (32 * 16 * 4) as usize);
        assert!(b.rgba.chunks_exact(4).all(|px| px == [1, 2, 3, 4]));
    }

    /// Helper: a tiny 1x1 atlas of a given RGBA color.
    fn px1(color: [u8; 4]) -> TextureData {
        TextureData {
            width: 1,
            height: 1,
            rgba: vec![color[0], color[1], color[2], color[3]],
        }
    }

    /// A mesh with a single layer wrapping a 1x1 atlas.
    fn mesh_one(color: [u8; 4]) -> MeshData {
        MeshData::default() // positions/indices are irrelevant for composites
            .with_texture(px1(color))
    }

    #[test]
    fn blend_modes_darken_lighten_and_overlay() {
        let acc = px1([200, 0, 0, 255]);
        let top = px1([128, 255, 0, 255]);

        let mut norm = acc.clone();
        src_over(&mut norm, &top, 1.0, BlendMode::Normal);
        assert_eq!(&norm.rgba, &[128, 255, 0, 255], "Normal copies src");

        let mut mul = px1([200, 0, 0, 255]);
        src_over(&mut mul, &top, 1.0, BlendMode::Multiply);
        // 200*128/255 ≈ 100 red; green 0*1 = 0; blue 0.
        assert_eq!(&mul.rgba, &[100, 0, 0, 255], "Multiply darkens");

        let mut scr = px1([200, 0, 0, 255]);
        src_over(&mut scr, &top, 1.0, BlendMode::Screen);
        // 1 - (1-200/255)(1-128/255) ≈ 228; green 1-0 = 1.
        assert_eq!(&scr.rgba, &[228, 255, 0, 255], "Screen lightens");

        let mut ovl = px1([200, 0, 0, 255]);
        src_over(&mut ovl, &top, 1.0, BlendMode::Overlay);
        // cd>0.5 -> 1 - 2(1-cd)(1-cs) ≈ 200; green cd=0<=0.5 -> 2*cs*cd = 0.
        assert_eq!(&ovl.rgba, &[200, 0, 0, 255], "Overlay hard lights");
    }

    #[test]
    fn flattened_region_matches_full_atlas() {
        const N: u32 = 16;
        let mut bottom = vec![0u8; (N * N * 4) as usize];
        for i in 0..(N * N) as usize {
            bottom[i * 4..i * 4 + 4].copy_from_slice(&[(i as u8).wrapping_mul(3), 20, 30, 255]);
        }
        let mut top = vec![0u8; (N * N * 4) as usize];
        for i in 0..(N * N) as usize {
            let (x, y) = (i as u32 % N, i as u32 / N);
            let a = if (x + y) % 3 == 0 { 200 } else { 0 };
            top[i * 4..i * 4 + 4].copy_from_slice(&[255, a, 10, a]);
        }
        let mesh = MeshData {
            layers: vec![
                Layer {
                    name: "bottom".into(),
                    visible: true,
                    opacity: 1.0,
                    locked: false,
                    blend: BlendMode::Normal,
                    roughness: 0.55,
                    metallic: 0.0,
                    emissive: 0.0,
                    ambient_occlusion: 1.0,
                    height: 0.0,
                    bump_strength: 2.0,
                    clearcoat: 0.0,
                    clearcoat_roughness: 0.6,
                    specular_ior: 1.5,
                    emissive_color: [1.0, 1.0, 1.0],
                    texture: TextureData {
                        width: N,
                        height: N,
                        rgba: bottom,
                    },
                },
                Layer {
                    name: "top".into(),
                    visible: true,
                    opacity: 0.5,
                    locked: false,
                    blend: BlendMode::Multiply,
                    roughness: 0.1,
                    metallic: 0.9,
                    emissive: 0.0,
                    ambient_occlusion: 1.0,
                    height: 0.0,
                    bump_strength: 2.0,
                    clearcoat: 0.0,
                    clearcoat_roughness: 0.6,
                    specular_ior: 1.5,
                    emissive_color: [1.0, 1.0, 1.0],
                    texture: TextureData {
                        width: N,
                        height: N,
                        rgba: top,
                    },
                },
            ],
            ..Default::default()
        };

        let full = mesh.flattened_atlas().unwrap();
        for (x, y, w, h) in [
            (0, 0, N, N),
            (3, 5, 7, 9),
            (9, 1, 6, 2),
            (0, 0, 1, 1),
            (N - 1, N - 1, 4, 4), // clamped to the atlas edge
        ] {
            let reg = mesh.flattened_atlas_region(x, y, w, h).unwrap();
            let exp_w = w.min(N - x);
            let exp_h = h.min(N - y);
            assert_eq!((reg.width, reg.height), (exp_w, exp_h));
            for yy in 0..reg.height {
                for xx in 0..reg.width {
                    let si = (((y + yy) * N + (x + xx)) as usize) * 4;
                    let di = ((yy * reg.width + xx) as usize) * 4;
                    assert_eq!(
                        &reg.rgba[di..di + 4],
                        &full.rgba[si..si + 4],
                        "pixel ({}, {}) differs",
                        x + xx,
                        y + yy
                    );
                }
            }
        }
    }

    #[test]
    fn flattened_composites_bottom_to_top() {
        let mut mesh = mesh_one([255, 0, 0, 255]);
        mesh.layers.push(Layer::new("L2", px1([0, 0, 255, 255])));
        let flat = mesh.flattened_atlas().unwrap();
        // Red under opaque blue -> solid blue.
        assert_eq!(flat.rgba, [0, 0, 255, 255]);
    }

    #[test]
    fn flattened_respects_layer_opacity() {
        let mut mesh = mesh_one([255, 0, 0, 255]); // opaque red bottom
        mesh.layers.push(Layer {
            name: "L2".into(),
            visible: true,
            opacity: 0.5,
            locked: false,
            blend: BlendMode::Normal,
            roughness: 0.55,
            metallic: 0.0,
            emissive: 0.0,
            ambient_occlusion: 1.0,
            height: 0.0,
            bump_strength: 2.0,
            clearcoat: 0.0,
            clearcoat_roughness: 0.6,
            specular_ior: 1.5,
            emissive_color: [1.0, 1.0, 1.0],
            texture: px1([0, 0, 255, 255]), // 50% blue on top
        });
        let flat = mesh.flattened_atlas().unwrap();
        // src-over: out = blue*0.5 + red*0.5, alpha = 0.5 + 0.5 = 1.0.
        let expect = [
            (0.5_f32 * 0.0 + 0.5_f32 * 255.0).round() as u8,
            (0.5_f32 * 0.0 + 0.5_f32 * 0.0).round() as u8,
            (0.5_f32 * 255.0 + 0.5_f32 * 0.0).round() as u8,
            255,
        ];
        assert_eq!(flat.rgba, expect);
    }

    #[test]
    fn flattened_skips_hidden_and_zero_opacity_layers() {
        let mut mesh = mesh_one([255, 0, 0, 255]);
        mesh.layers.push(Layer {
            name: "hidden".into(),
            visible: false,
            opacity: 1.0,
            locked: false,
            blend: BlendMode::Normal,
            roughness: 0.55,
            metallic: 0.0,
            emissive: 0.0,
            ambient_occlusion: 1.0,
            height: 0.0,
            bump_strength: 2.0,
            clearcoat: 0.0,
            clearcoat_roughness: 0.6,
            specular_ior: 1.5,
            emissive_color: [1.0, 1.0, 1.0],
            texture: px1([0, 255, 0, 255]), // green, invisible
        });
        mesh.layers.push(Layer {
            name: "zero".into(),
            visible: true,
            opacity: 0.0,
            locked: false,
            blend: BlendMode::Normal,
            roughness: 0.55,
            metallic: 0.0,
            emissive: 0.0,
            ambient_occlusion: 1.0,
            height: 0.0,
            bump_strength: 2.0,
            clearcoat: 0.0,
            clearcoat_roughness: 0.6,
            specular_ior: 1.5,
            emissive_color: [1.0, 1.0, 1.0],
            texture: px1([255, 255, 0, 255]), // yellow, fully transparent
        });
        let flat = mesh.flattened_atlas().unwrap();
        assert_eq!(flat.rgba, [255, 0, 0, 255]);
    }

    #[test]
    fn flatten_all_transparent_gives_transparent_composite() {
        let mesh = mesh_one([0, 0, 0, 0]);
        let flat = mesh.flattened_atlas().unwrap();
        assert_eq!(flat.rgba, [0, 0, 0, 0]);
    }

    #[test]
    fn flatten_empty_stack_returns_none() {
        let mesh = MeshData::default();
        assert!(mesh.flattened_atlas().is_none());
    }

    #[test]
    fn layers_with_mismatched_sizes_are_skipped() {
        let mut mesh = mesh_one([255, 0, 0, 255]);
        let mut wide = px1([0, 0, 255, 255]);
        wide.width = 2; // different atlas size -> skip
        mesh.layers.push(Layer::new("L2", wide));
        let flat = mesh.flattened_atlas().unwrap();
        assert_eq!(flat.rgba, [255, 0, 0, 255]);
    }

    #[test]
    fn material_atlas_unpainted_lands_on_defaults() {
        let mesh = mesh_one([255, 0, 0, 255]); // no paint, default material
        let flat = mesh.flattened_material_atlas().unwrap();
        let rough_d = (0.55f32 * 255.0).round() as u8;
        assert_eq!(flat.rgba[0], rough_d);
        assert_eq!(flat.rgba[1], 0); // metallic
        assert_eq!(flat.rgba[2], 0); // emissive/3
        assert_eq!(flat.rgba[3], 255); // ao
    }

    #[test]
    fn material_atlas_reads_active_surface_params() {
        let mut mesh = mesh_one([255, 0, 0, 255]);
        mesh.layers[0].roughness = 0.2;
        mesh.layers[0].metallic = 0.8;
        mesh.layers[0].emissive = 1.5;
        mesh.layers[0].ambient_occlusion = 0.4;
        let flat = mesh.flattened_material_atlas().unwrap();
        assert_eq!(flat.rgba[0], (0.2f32 * 255.0).round() as u8);
        assert_eq!(flat.rgba[1], (0.8f32 * 255.0).round() as u8);
        assert_eq!(flat.rgba[2], (0.5f32 * 255.0).round() as u8); // emissive is stored /3
        assert_eq!(flat.rgba[3], (0.4f32 * 255.0).round() as u8);
    }

    #[test]
    fn material_atlas_stacks_layers_by_opacity() {
        let mut mesh = mesh_one([255, 0, 0, 255]);
        mesh.layers[0].roughness = 0.3;
        mesh.layers.push(Layer {
            name: "L2".into(),
            visible: true,
            opacity: 0.5, // half-strength stroke blends halfway
            locked: false,
            blend: BlendMode::Normal,
            roughness: 0.9,
            metallic: 1.0,
            emissive: 0.0,
            ambient_occlusion: 0.5,
            height: 0.7,
            bump_strength: 2.0,
            clearcoat: 0.0,
            clearcoat_roughness: 0.6,
            specular_ior: 1.5,
            emissive_color: [1.0, 1.0, 1.0],
            texture: px1([255, 255, 255, 255]),
        });
        let flat = mesh.flattened_material_atlas().unwrap();
        // acc = over*0.5 + under*(1-0.5), evaluated in the byte domain because
        // each layer's material is quantized to a u8 before compositing.
        let byte = |v: f32| (v * 255.0).round().clamp(0.0, 255.0) as u8;
        let exp = |a: f32, b: f32| (byte(a) as f32 * 0.5 + byte(b) as f32 * 0.5).round() as u8;
        assert_eq!(flat.rgba[0], exp(0.9, 0.3));
        assert_eq!(flat.rgba[1], exp(1.0, 0.0));
        assert_eq!(flat.rgba[3], exp(0.5, 1.0));
    }

    #[test]
    fn material_atlas_skips_hidden_and_transparent_layers() {
        let mut mesh = mesh_one([255, 0, 0, 255]);
        mesh.layers.push(Layer {
            name: "off".into(),
            visible: false,
            opacity: 1.0,
            locked: false,
            blend: BlendMode::Normal,
            roughness: 0.1,
            metallic: 0.9,
            emissive: 0.0,
            ambient_occlusion: 0.2,
            height: 0.0,
            bump_strength: 2.0,
            clearcoat: 0.0,
            clearcoat_roughness: 0.6,
            specular_ior: 1.5,
            emissive_color: [1.0, 1.0, 1.0],
            texture: px1([255, 255, 255, 255]),
        });
        let flat = mesh.flattened_material_atlas().unwrap();
        let rough_d = (0.55f32 * 255.0).round() as u8;
        assert_eq!(flat.rgba[0], rough_d);
        assert_eq!(flat.rgba[1], 0);
        assert_eq!(flat.rgba[3], 255);
    }

    #[test]
    fn height_atlas_follows_paint_coverage() {
        let mut mesh = mesh_one([255, 0, 0, 255]); // fully painted by default
        let enc = |h: f32| ((h + 1.0) * 0.5 * 255.0).round().clamp(0.0, 255.0) as u8;
        // Zero height on a flat sheet -> R encodes signed 0 = byte 128 (center).
        let flat = mesh.flattened_height_atlas().unwrap();
        assert_eq!(
            flat.rgba[0],
            enc(0.0),
            "unpainted/no-height sheet must be flat"
        );
        assert_eq!(
            flat.rgba[1],
            (2.0f32 / 8.0 * 255.0).round() as u8,
            "default layer strength"
        );
        // A full-coverage 0.4-height layer raises every texel.
        mesh.layers[0].height = 0.4;
        let flat = mesh.flattened_height_atlas().unwrap();
        assert_eq!(flat.rgba[0], enc(0.4));
        assert_eq!(flat.rgba[1], (2.0f32 / 8.0 * 255.0).round() as u8);
        // A negative height carves: -0.4 decodes below center.
        mesh.layers[0].height = -0.4;
        assert_eq!(mesh.flattened_height_atlas().unwrap().rgba[0], enc(-0.4));
        mesh.layers[0].height = 0.4; // restore for blend test
                                     // A half-opacity 0.8-height / 4.0-strength layer over the base blends
                                     // halfway: the paint alpha (255) times opacity 0.5 gives sa = 0.5.
        mesh.layers.push(Layer {
            name: "L2".into(),
            visible: true,
            opacity: 0.5,
            locked: false,
            blend: BlendMode::Normal,
            roughness: 0.0,
            metallic: 0.0,
            emissive: 0.0,
            ambient_occlusion: 1.0,
            height: 0.8,
            bump_strength: 4.0,
            clearcoat: 0.0,
            clearcoat_roughness: 0.6,
            specular_ior: 1.5,
            emissive_color: [1.0, 1.0, 1.0],
            texture: px1([255, 255, 255, 255]),
        });
        let flat = mesh.flattened_height_atlas().unwrap();
        assert_eq!(
            flat.rgba[0],
            (enc(0.8) as f32 * 0.5 + enc(0.4) as f32 * 0.5).round() as u8
        );
        let byte_s = |v: f32| (v / 8.0 * 255.0).round().clamp(0.0, 255.0) as u8;
        assert_eq!(
            flat.rgba[1],
            (byte_s(4.0) as f32 * 0.5 + byte_s(2.0) as f32 * 0.5).round() as u8
        );
    }

    /// Two quads: one 1x1 and one 2x2 in world space, both squeezed into the
    /// same 0.25×1 UV box — so before a remake they get equal texel budgets.
    fn uneven_panels() -> MeshData {
        let mut m = MeshData::default().with_texture(TextureData {
            width: 128,
            height: 128,
            rgba: [240, 240, 240, 255].repeat(128 * 128),
        });
        let add_quad = |m: &mut MeshData, world: f32, y: f32, uv: [(f32, f32); 4]| {
            let s = world / 2.0;
            let base = m.positions.len() as u32;
            m.positions.extend_from_slice(&[
                glam::Vec3::new(-s, y, -s),
                glam::Vec3::new(s, y, -s),
                glam::Vec3::new(s, y, s),
                glam::Vec3::new(-s, y, s),
            ]);
            m.normals.extend(std::iter::repeat_n(glam::Vec3::Y, 4));
            m.uvs.extend_from_slice(&uv);
            m.indices
                .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        };
        add_quad(
            &mut m,
            1.0,
            0.0,
            [(0.0, 0.0), (0.25, 0.0), (0.25, 1.0), (0.0, 1.0)],
        );
        add_quad(
            &mut m,
            2.0,
            0.0,
            [(0.25, 0.0), (0.5, 0.0), (0.5, 1.0), (0.25, 1.0)],
        );
        m
    }

    fn island_uv_extent(mesh: &MeshData, quad_idx: usize) -> (f32, f32) {
        let v_a = mesh.indices[quad_idx * 6] as usize; // base (u=0 corner)
        let v_b = mesh.indices[quad_idx * 6 + 1] as usize; // base+1 (max-u corner)
        let v_c = mesh.indices[quad_idx * 6 + 2] as usize; // base+2 (max-v corner)
        let u_extent = (mesh.uvs[v_b].0 - mesh.uvs[v_a].0).abs() * 128.0;
        let v_extent = (mesh.uvs[v_c].1 - mesh.uvs[v_a].1).abs() * 128.0;
        (u_extent, v_extent)
    }

    /// Each pixel is a unique 4-byte cell so a mirror is verifiable texel-by-texel.
    #[test]
    fn flip_texture_mirrors_columns_and_rows() {
        let mut tex = TextureData {
            width: 3,
            height: 2,
            rgba: (0..6)
                .flat_map(|i| [i as u8, i as u8 + 1, i as u8 + 2, 255])
                .collect(),
        };
        let row = |tex: &TextureData, y: usize| -> Vec<u8> {
            tex.rgba[y * tex.width as usize * 4..(y + 1) * tex.width as usize * 4]
                .chunks_exact(4)
                .map(|px| px[0])
                .collect()
        };

        flip_texture(&mut tex, true, false);
        // Row-major left/right mirror: [0,1,2] → [2,1,0], [3,4,5] → [5,4,3].
        assert_eq!(row(&tex, 0), vec![2, 1, 0]);
        assert_eq!(row(&tex, 1), vec![5, 4, 3]);

        flip_texture(&mut tex, false, true);
        // Top/bottom swap on top of the horizontal flip: vertical mirror.
        assert_eq!(row(&tex, 0), vec![5, 4, 3]);
        assert_eq!(row(&tex, 1), vec![2, 1, 0]);

        // No-op when neither axis is requested.
        let snapshot = tex.rgba.clone();
        flip_texture(&mut tex, false, false);
        assert_eq!(tex.rgba, snapshot);
    }

    #[test]
    fn remake_uv_gives_big_parts_more_texels() {
        let mut mesh = uneven_panels();
        let before = island_uv_extent(&mesh, 0);
        let before_big = island_uv_extent(&mesh, 1);
        assert_eq!(before, before_big, "equal budgets before the remake");

        let islands = remake_uv(&mut mesh);
        assert_eq!(islands, 2);

        let small = island_uv_extent(&mesh, 0);
        let big = island_uv_extent(&mesh, 1);
        // The 2x2 world quad is 4× the surface area → 4× the texels, i.e. 2×
        // the linear texel extent. Assert a clear margin despite packing scale.
        assert!(
            big.0 > small.0 * 1.5 && big.1 > small.1 * 1.5,
            "big part should get ~2x linear texels, got small={small:?} big={big:?}"
        );
        assert!(
            big.0 * big.1 > small.0 * small.1 * 3.0,
            "big part should get ~4x texel count, got small={small:?} big={big:?}"
        );
    }

    #[test]
    fn remake_uv_keeps_everything_in_unit_square() {
        let mut mesh = uneven_panels();
        remake_uv(&mut mesh);
        for (u, v) in &mesh.uvs {
            assert!(*u >= 0.0 && *u <= 1.0, "u={u} out of range");
            assert!(*v >= 0.0 && *v <= 1.0, "v={v} out of range");
        }
    }

    #[test]
    fn remake_uv_single_island_is_a_noop() {
        let mut mesh = MeshData::uv_sphere(0.6, 4, 6).with_texture(TextureData {
            width: 64,
            height: 64,
            rgba: vec![0u8; 64 * 64 * 4],
        });
        let uvs_before = mesh.uvs.clone();
        let tex_before = mesh.layers[0].texture.clone();
        assert_eq!(remake_uv(&mut mesh), 1);
        assert_eq!(mesh.uvs, uvs_before, "single island remaps to itself");
        assert_eq!(
            mesh.layers[0].texture, tex_before,
            "single island leaves the texture alone"
        );
        assert!(mesh.dirty.is_none(), "no-op must not dirty the atlas");
    }

    #[test]
    fn remake_uv_rebakes_atlas_content_onto_islands() {
        let mut mesh = uneven_panels();
        // Paint the big panel with a vertical gradient in its UV box; after the
        // remake, the same world surface must still sample the same gradient.
        let w = 128u32;
        let h = 128u32;
        let rgba = vec![240u8; (w * h * 4) as usize];
        mesh.layers = vec![Layer {
            name: "Base".into(),
            visible: true,
            opacity: 1.0,
            locked: false,
            blend: BlendMode::Normal,
            roughness: 0.3,
            metallic: 0.0,
            emissive: 0.0,
            ambient_occlusion: 1.0,
            height: 0.0,
            bump_strength: 2.0,
            clearcoat: 0.0,
            clearcoat_roughness: 0.6,
            specular_ior: 1.5,
            emissive_color: [1.0, 1.0, 1.0],
            texture: TextureData {
                width: w,
                height: h,
                rgba,
            },
        }];
        let tex = &mut mesh.layers[0].texture;
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) as usize * 4;
                tex.rgba[i] = (x as f32 / w as f32 * 255.0).round() as u8;
                tex.rgba[i + 3] = 255;
            }
        }
        let mesh_before = mesh.clone();
        // The 2x2 quad is the second island `uneven_panels` pushes (verts 4..7);
        // vertex indices are stable across the remap, only the UVs change.
        let big_idxs: Vec<usize> = (4..8).collect();
        remake_uv(&mut mesh);
        let (lo_b, hi_b) = {
            let mut vs = mesh_before.uvs[..]
                .iter()
                .enumerate()
                .filter(|(i, _)| big_idxs.contains(i))
                .map(|(_, &(u, _))| u);
            let (mut lo, mut hi) = (vs.next().unwrap(), vs.next().unwrap());
            for u in vs {
                lo = lo.min(u);
                hi = hi.max(u);
            }
            (lo, hi)
        };
        let (lo_a, hi_a) = mesh
            .uvs
            .iter()
            .enumerate()
            .filter(|(i, _)| big_idxs.contains(i))
            .map(|(_, &(u, _))| u)
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), u| {
                (lo.min(u), hi.max(u))
            });
        assert!((lo_b - 0.25).abs() < 1e-3 && (hi_b - 0.5).abs() < 1e-3);
        assert_eq!(lo_a, 0.0, "big island should start at the atlas origin");
        // Sample both meshes at matching world positions on the BIG quad's
        // interior; the remapped mesh must land on the same gradient value.
        let tex_before = &mesh_before.layers[0].texture;
        let tex_after = &mesh.layers[0].texture;
        let sample = |tex: &TextureData, u: f32, v: f32| {
            let x = (u * tex.width as f32) as usize % tex.width as usize;
            let y = (v * tex.height as f32) as usize % tex.height as usize;
            tex.rgba[(y * tex.width as usize + x) * 4]
        };
        for t in [0.2, 0.4, 0.6] {
            let u_b = lo_b + t * (hi_b - lo_b);
            let u_a = lo_a + t * (hi_a - lo_a);
            let r0 = sample(tex_before, u_b, 0.5);
            let r1 = sample(tex_after, u_a, 0.5);
            assert!(
                (r0 as i32 - r1 as i32).abs() <= 1,
                "paint must survive the rewrap at t={t}: before={r0} after={r1}"
            );
        }
    }
}
