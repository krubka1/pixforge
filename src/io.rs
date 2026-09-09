use glam::Vec3;
use std::collections::HashMap;

/// Writes a PNG of the albedo atlas at `path`, encoding the raw CPU rows
/// top-down (row 0 = texture top, matching glTF UV v=0).
pub fn save_atlas_png(path: &str, tex: &TextureData) -> std::io::Result<()> {
    let img = image::RgbaImage::from_raw(tex.width, tex.height, tex.rgba.clone())
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "bad atlas size"))?;
    img.save(path)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
}

#[derive(Debug, Clone)]
pub struct TextureData {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct MeshData {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<(f32, f32)>,
    pub indices: Vec<u32>,
    /// Albedo texture atlas for the mesh (all base-color images packed in, UVs remapped).
    pub texture: Option<TextureData>,
}

impl MeshData {
    pub fn with_texture(mut self, texture: TextureData) -> Self {
        self.texture = Some(texture);
        self
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
                indices.extend_from_slice(&[a, b, d, b, c, d]);
            }
        }

        Self {
            positions,
            normals,
            uvs,
            indices,
            texture: None,
        }
    }
}

/// Default albedo texture shown on the startup sphere: a light base with
/// pixel-art "paint blobs", fitting PixForge's painter identity.
pub fn default_albedo() -> TextureData {
    const S: u32 = 256;
    let mut rgba = vec![246u8, 241, 232, 255];
    rgba.resize((S * S * 4) as usize, 255);

    let base_light = [246u8, 241, 232];
    let base_dark = [238u8, 231, 219];
    let grid = [250u8, 247, 240];

    // Blobs: (cx, cy, radius, color).
    let blobs: [(f32, f32, f32, [u8; 3]); 4] = [
        (64.0, 64.0, 26.0, [214, 65, 65]),
        (160.0, 52.0, 22.0, [61, 139, 214]),
        (96.0, 176.0, 30.0, [70, 160, 92]),
        (188.0, 168.0, 24.0, [224, 161, 60]),
    ];

    for y in 0..S {
        for x in 0..S {
            let mut c = if ((x / 64 + y / 64) % 2) == 0 {
                base_dark
            } else {
                base_light
            };
            if x % 64 < 2 || y % 64 < 2 {
                c = grid;
            }

            let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
            for &(cx, cy, r, color) in &blobs {
                let d = ((fx - cx).powi(2) + (fy - cy).powi(2)).sqrt();
                if d <= r {
                    // Hard pixel-edges: darker 2px border then flat fill.
                    c = if r - d < 2.0 {
                        [darker(color), darker(color), darker(color)]
                    } else {
                        color
                    };
                }
            }

            let i = ((y * S + x) * 4) as usize;
            rgba[i..i + 3].copy_from_slice(&c);
            rgba[i + 3] = 255;
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
    TextureData { width: w, height: h, rgba }
}

fn darker(c: [u8; 3]) -> u8 {
    (c[0] as f32 * 0.55) as u8
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
                        atlas[dst..dst + (*w as usize) * 4].copy_from_slice(&rgba[src..src + (*w as usize) * 4]);
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

    // Pass 2: fill mesh data, remapping each primitive's UVs into its atlas slot.
    let mut positions: Vec<Vec3> = Vec::new();
    let mut normals: Vec<Vec3> = Vec::new();
    let mut uvs: Vec<(f32, f32)> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
            let base = positions.len() as u32;

            match reader.read_positions() {
                Some(iter) => positions.extend(iter.map(|v| Vec3::new(v[0], v[1], v[2]))),
                None => return LoadedModel::Invalid,
            }

            match reader.read_normals() {
                Some(iter) => normals.extend(iter.map(|v| Vec3::new(v[0], v[1], v[2]))),
                None => normals.extend(std::iter::repeat(Vec3::ZERO).take(positions.len() as usize)),
            }

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
                        let aw = atlas_texture.as_ref().map(|t| t.width).unwrap_or(1) as f32;
                        let ah = atlas_texture.as_ref().map(|t| t.height).unwrap_or(1) as f32;
                        (x as f32 / aw, y as f32 / ah, w as f32 / aw, h as f32 / ah)
                    }
                    None => (0.0, 0.0, 1.0, 1.0),
                },
                None => (0.0, 0.0, 1.0, 1.0),
            };

            match reader.read_tex_coords(0) {
                Some(uv) => uvs.extend(uv.into_f32().map(|v| (v[0] * sx + ox, v[1] * sy + oy))),
                None => uvs.extend(std::iter::repeat((0.0, 0.0)).take(positions.len() as usize)),
            }

            match reader.read_indices() {
                Some(ind) => indices.extend(ind.into_u32().map(|i| i + base)),
                None => {
                    let n = reader.read_positions().map(|it| it.len()).unwrap_or(0);
                    indices.extend(base..base + n as u32)
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
        texture: atlas_texture,
    })
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

        let path = std::env::temp_dir().join(format!(
            "pixforge_test_{}.gltf",
            std::process::id()
        ));
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
                assert!(m.texture.is_none(), "no material => no texture");
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
        assert!(
            s.indices
                .iter()
                .all(|&i| (i as usize) < s.positions.len())
        );

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
            rgba: (0..(4 * 2 * 4))
                .map(|i| i as u8)
                .collect::<Vec<u8>>(),
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
        assert_eq!(r.rgba.chunks_exact(4).all(|px| px[..4] == [7, 13, 29, 255]), true);
        assert_eq!((r.width, r.height), (5, 5));
    }

    #[test]
    fn blank_atlas_has_requested_size_and_fill() {
        let b = blank_atlas(32, 16, [1, 2, 3, 4]);
        assert_eq!((b.width, b.height), (32, 16));
        assert_eq!(b.rgba.len(), (32 * 16 * 4) as usize);
        assert!(b.rgba.chunks_exact(4).all(|px| px == [1, 2, 3, 4]));
    }
}