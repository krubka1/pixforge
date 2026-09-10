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
            let lum = (raw[j] as f32 * 0.2126
                + raw[j + 1] as f32 * 0.7152
                + raw[j + 2] as f32 * 0.0722)
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
    for (ap, sp) in acc.rgba.chunks_exact_mut(4).zip(src.rgba.chunks_exact(4)) {
        let (ap, sp) = (ap.try_into().unwrap(), sp.try_into().unwrap());
        src_over_px(ap, sp, opacity, mode);
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

#[derive(Debug, Clone)]
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
    pub texture: TextureData,
}

impl Layer {
    pub fn new(name: impl Into<String>, texture: TextureData) -> Self {
        Self {
            name: name.into(),
            visible: true,
            opacity: 1.0,
            blend: BlendMode::Normal,
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
        let Some(first) = self.layers.iter().find(|l| l.visible && l.opacity > 0.0) else {
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
        let (w, h) = (first.texture.width, first.texture.height);
        let mut acc = TextureData {
            width: w,
            height: h,
            rgba: vec![0; (w * h * 4) as usize],
        };
        for layer in self.layers.iter().filter(|l| l.visible && l.opacity > 0.0) {
            src_over(&mut acc, &layer.texture, layer.opacity, layer.blend);
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
            let src = &layer.texture;
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

/// Default albedo texture shown on the startup sphere: a light base with
/// pixel-art "paint blobs", fitting PixForge's painter identity.
pub fn default_albedo() -> TextureData {
    const S: u32 = 512;
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
    TextureData {
        width: w,
        height: h,
        rgba,
    }
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
            Some(iter) => iter
                .map(|v| Vec3::new(v[0], v[1], v[2]))
                .collect::<Vec<_>>(),
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
        assert_eq!(
            r.rgba.chunks_exact(4).all(|px| px[..4] == [7, 13, 29, 255]),
            true
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
                    blend: BlendMode::Normal,
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
                    blend: BlendMode::Multiply,
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
            blend: BlendMode::Normal,
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
            blend: BlendMode::Normal,
            texture: px1([0, 255, 0, 255]), // green, invisible
        });
        mesh.layers.push(Layer {
            name: "zero".into(),
            visible: true,
            opacity: 0.0,
            blend: BlendMode::Normal,
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
}
