//! Brush library: a folder-backed collection of brush presets.
//!
//! The library always exposes a set of built-in brushes (the geometric shapes
//! plus a couple of procedurally generated texture stamps) and augments them
//! with every supported brush file found under the brush folder. Subfolders
//! become categories, so dropping a downloaded brush pack into its own folder
//! yields a filterable group.
//!
//! Supported file formats:
//! * Image masks (PNG/JPG/BMP/WebP/GIF) — transparent pixels are the dab; an
//!   opaque image is used inverted (dark = strong paint). See
//!   [`crate::io::brush_sprite`].
//! * GIMP `.gbr` v1 brushes (grayscale opacity or RGBA).

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::io::TextureData;

/// Category label used for the brushes shipped with PixForge.
pub const BUILTIN_CATEGORY: &str = "Built-in";
/// Category label for files that sit directly in the brush folder root.
pub const ROOT_CATEGORY: &str = "Root";

const IMAGE_EXTS: [&str; 6] = ["png", "jpg", "jpeg", "bmp", "webp", "gif"];

/// How a brush produces its footprint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrushKind {
    /// A plain geometric shape (round/square/diamond); no image needed.
    Shape(crate::paint::BrushShape),
    /// An image stamp — the entry's `sprite` is the coverage mask.
    Texture,
}

/// One selectable brush preset.
#[derive(Clone)]
pub struct BrushEntry {
    pub name: String,
    pub category: String,
    pub kind: BrushKind,
    /// Normalized coverage mask (white RGB, coverage in alpha). Every entry has
    /// one so it doubles as the thumbnail source.
    pub sprite: TextureData,
    /// Filesystem path for file-backed brushes; `None` for built-ins.
    pub path: Option<PathBuf>,
}

/// Folder-backed brush collection with throttled rescans.
pub struct BrushLibrary {
    pub folder: PathBuf,
    pub entries: Vec<BrushEntry>,
    /// Index into `entries` of the last brush applied, if any.
    pub selected: Option<usize>,
    /// Most recent skip error (unsupported/corrupt file), shown in the panel.
    pub last_error: Option<String>,
    signature: String,
    last_refresh: Option<Instant>,
}

impl BrushLibrary {
    pub fn new(folder: PathBuf) -> Self {
        let _ = std::fs::create_dir_all(&folder);
        let entries = builtin_entries();
        let signature = signature_of(&entries);
        Self {
            folder,
            entries,
            selected: None,
            last_error: None,
            signature,
            last_refresh: None,
        }
    }

    /// Walks the folder and rebuilds the entry list when its contents changed.
    /// Throttled so it is cheap to call every frame.
    pub fn refresh(&mut self) {
        if let Some(t) = self.last_refresh {
            if t.elapsed() < Duration::from_millis(1500) {
                return;
            }
        }
        self.last_refresh = Some(Instant::now());

        let mut entries = builtin_entries();
        let mut paths = Vec::new();
        collect_brush_files(&self.folder, &mut paths);
        paths.sort();

        let mut error = None;
        for path in paths {
            match load_entry(&self.folder, &path) {
                Ok(entry) => entries.push(entry),
                Err(msg) => error = Some(msg),
            }
        }

        let signature = signature_of(&entries);
        if signature != self.signature {
            self.entries = entries;
            self.signature = signature;
            if self.selected.is_some_and(|i| i >= self.entries.len()) {
                self.selected = None;
            }
        }
        self.last_error = error;
    }

    /// Forces an immediate rescan (used by the panel's Rescan button).
    pub fn force_refresh(&mut self) {
        self.last_refresh = None;
        self.refresh();
    }

    /// Unique category labels, `Built-in` first then alphabetical.
    pub fn categories(&self) -> Vec<String> {
        let mut cats: Vec<String> = Vec::new();
        for entry in &self.entries {
            if !cats.contains(&entry.category) {
                cats.push(entry.category.clone());
            }
        }
        cats.sort();
        if let Some(pos) = cats.iter().position(|c| c == BUILTIN_CATEGORY) {
            let builtin = cats.remove(pos);
            cats.insert(0, builtin);
        }
        cats
    }

    pub fn signature(&self) -> &str {
        &self.signature
    }
}

/// Reads one brush file into an entry, normalizing it to a coverage mask.
fn load_entry(root: &Path, path: &Path) -> Result<BrushEntry, String> {
    let name = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "brush".to_string());
    let category = category_of(root, path);
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();

    let sprite = if ext == "gbr" {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        parse_gbr(&bytes).map_err(|e| format!("{}: {e}", path.display()))?
    } else {
        crate::io::brush_sprite(&path.to_string_lossy())
            .map_err(|e| format!("{}: {e}", path.display()))?
    };

    Ok(BrushEntry {
        name,
        category,
        kind: BrushKind::Texture,
        sprite,
        path: Some(path.to_path_buf()),
    })
}

/// Recursively collects supported brush files under `dir`.
fn collect_brush_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(read) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in read.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_brush_files(&path, out);
        } else {
            let ext = path
                .extension()
                .map(|e| e.to_string_lossy().to_ascii_lowercase())
                .unwrap_or_default();
            if ext == "gbr" || IMAGE_EXTS.contains(&ext.as_str()) {
                out.push(path);
            }
        }
    }
}

/// The category for a file: its top-level folder under `root`, or `Root`.
fn category_of(root: &Path, path: &Path) -> String {
    let rel = path.strip_prefix(root).unwrap_or(path);
    let mut comps = rel.components();
    match (comps.next(), comps.next()) {
        (Some(first), Some(_)) => first.as_os_str().to_string_lossy().into_owned(),
        _ => ROOT_CATEGORY.to_string(),
    }
}

/// A stable identity for the entry list, used to detect folder changes.
fn signature_of(entries: &[BrushEntry]) -> String {
    let mut out = String::new();
    for e in entries {
        let kind = match e.kind {
            BrushKind::Shape(s) => s.label(),
            BrushKind::Texture => "tex",
        };
        out.push_str(&format!("{}|{}|{}\n", e.category, e.name, kind));
    }
    out
}

fn builtin_entries() -> Vec<BrushEntry> {
    use crate::paint::BrushShape;
    let mut entries = Vec::new();
    for shape in [BrushShape::Round, BrushShape::Square, BrushShape::Diamond] {
        entries.push(BrushEntry {
            name: shape.label().to_string(),
            category: BUILTIN_CATEGORY.to_string(),
            kind: BrushKind::Shape(shape),
            sprite: shape_sprite(shape),
            path: None,
        });
    }
    for (name, gen_fn) in [
        ("Splotch", splotch_sprite as fn(u32, u32) -> TextureData),
        ("Grain", grain_sprite),
        ("Wood", wood_sprite),
        ("Marble", marble_sprite),
        ("Rust", rust_sprite),
    ] {
        entries.push(BrushEntry {
            name: name.to_string(),
            category: BUILTIN_CATEGORY.to_string(),
            kind: BrushKind::Texture,
            sprite: gen_fn(MASK_SIZE, (name.len() as u32).wrapping_mul(0xD651_7F53)),
            path: None,
        });
    }
    entries
}

// ---------------------------------------------------------------------------
// Procedural sprite generation
// ---------------------------------------------------------------------------

const MASK_SIZE: u32 = 64;

/// Builds a `TextureData` whose RGB is white and alpha is `coverage(x, y)`.
fn make_sprite(w: u32, h: u32, coverage: impl Fn(u32, u32) -> f32) -> TextureData {
    let mut rgba = vec![255u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let cov = coverage(x, y).clamp(0.0, 1.0);
            let i = ((y * w + x) * 4 + 3) as usize;
            rgba[i] = (cov * 255.0 + 0.5) as u8;
        }
    }
    TextureData {
        width: w,
        height: h,
        rgba,
    }
}

fn shape_sprite(shape: crate::paint::BrushShape) -> TextureData {
    use crate::paint::BrushShape;
    make_sprite(MASK_SIZE, MASK_SIZE, |x, y| {
        let (nx, ny) = centered(x, y, MASK_SIZE);
        let d = match shape {
            BrushShape::Round | BrushShape::Texture => (nx * nx + ny * ny).sqrt(),
            BrushShape::Square => nx.abs().max(ny.abs()),
            BrushShape::Diamond => nx.abs() + ny.abs(),
        };
        edge(d, 0.92, 0.14)
    })
}

fn splotch_sprite(size: u32, seed: u32) -> TextureData {
    let mut rng = Lcg::new(seed);
    let blobs: Vec<(f32, f32, f32)> = (0..6)
        .map(|_| {
            (
                rng.range(-0.55, 0.55),
                rng.range(-0.55, 0.55),
                rng.range(0.20, 0.42),
            )
        })
        .collect();
    make_sprite(size, size, |x, y| {
        let (nx, ny) = centered(x, y, size);
        let mut v = 0.0f32;
        for &(bx, by, r) in &blobs {
            let d = ((nx - bx).powi(2) + (ny - by).powi(2)).sqrt();
            v = v.max(edge(d, r, r * 0.8));
        }
        v
    })
}

fn grain_sprite(size: u32, seed: u32) -> TextureData {
    let mut rng = Lcg::new(seed);
    let noise: Vec<f32> = (0..(size * size)).map(|_| rng.next_f32()).collect();
    make_sprite(size, size, |x, y| {
        let (nx, ny) = centered(x, y, size);
        let radial = edge((nx * nx + ny * ny).sqrt(), 0.92, 0.14);
        let mut sum = 0.0;
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let xx = (x as i32 + dx).clamp(0, size as i32 - 1) as u32;
                let yy = (y as i32 + dy).clamp(0, size as i32 - 1) as u32;
                sum += noise[(yy * size + xx) as usize];
            }
        }
        let smoothed = sum / 9.0;
        radial * (0.32 + 0.68 * smoothed)
    })
}

/// Plank-face wood grain: long wavy streaks running down the board, with
/// denser "figure" patches and a couple of knots, instead of end-grain rings.
/// Coverage bridges the streaks (high) and dips into the dark groove lines.
fn wood_sprite(size: u32, seed: u32) -> TextureData {
    let mut rng = Lcg::new(seed);
    // Knots are small spots whose field pull bends the grain around them.
    let knots: Vec<(f32, f32, f32)> = (0..2)
        .map(|_| {
            (
                rng.range(-0.55, 0.55),
                rng.range(-0.55, 0.55),
                rng.range(0.10, 0.18),
            )
        })
        .collect();
    make_sprite(size, size, |x, y| {
        let (nx, ny) = centered(x, y, size);
        let radial = edge((nx * nx + ny * ny).sqrt(), 0.95, 0.2);
        let wobble = bilinear_noise(nx * 2.5, ny * 1.2, 5, seed);
        let figure = bilinear_noise(nx * 0.9 + 4.0, ny * 0.7, 3, seed ^ 0x00F1_0ECE);
        // Grain field: lines run along y, slowly curving with x (lengthwise
        // streaks), bent by low-frequency noise and pulled by the knots.
        let mut warp = (wobble - 0.5) * 2.2 + (nx * 1.6 + 0.5).sin() * 0.35;
        for &(kx, ky, kr) in &knots {
            let d = ((nx - kx).powi(2) + (ny - ky).powi(2)).sqrt();
            warp += (-d / kr).exp() * (ny - ky);
        }
        let field = ny * 24.0 + warp * 6.0;
        let streak = 0.5 + 0.5 * field.sin();
        // Grooves are the dark lines: thicker in noisy figure, thin in plain.
        let groove = edge(streak, 0.20 + 0.18 * figure.powf(3.0), 0.14);
        let grain =
            0.80 + 0.20 * bilinear_noise(nx * 4.0 + 2.0, ny * 4.0 + 1.0, 5, seed ^ 0x05EE_D10F);
        radial * (0.97 - 0.68 * groove) * grain
    })
}

/// Wavy striated marble: a noise-warped fold field carves thin dark seams
/// through a bright ground, like polished veined stone.
fn marble_sprite(size: u32, seed: u32) -> TextureData {
    make_sprite(size, size, |x, y| {
        let (nx, ny) = centered(x, y, size);
        let radial = edge((nx * nx + ny * ny).sqrt(), 0.95, 0.2);
        let n1 = bilinear_noise(nx * 1.5 + 2.0, ny * 1.5, 5, seed);
        let n2 = bilinear_noise(nx * 4.0 + 3.0, ny * 4.0 + 1.0, 6, seed ^ 0xA5A5_A5A5);
        // Fold field: horizontal strata wobbled by noise and a gentle arc.
        let field = ny * 11.0 + 2.2 * (n1 - 0.5) + 0.45 * (nx * 1.7 + 1.0).sin() + 0.6;
        let v = field.sin() * 0.5 + 0.5;
        let vein = edge(v, 0.30, 0.16);
        let haze = 0.86 + 0.14 * n2;
        radial * (0.93 - 0.72 * vein) * haze
    })
}

/// Blotchy, pitted corrosion: rough high coverage with patchy dark pits where
/// the rust has eaten through.
fn rust_sprite(size: u32, seed: u32) -> TextureData {
    let mut rng = Lcg::new(seed);
    let pits: Vec<(f32, f32, f32)> = (0..9)
        .map(|_| {
            (
                rng.range(-0.6, 0.6),
                rng.range(-0.6, 0.6),
                rng.range(0.10, 0.24),
            )
        })
        .collect();
    make_sprite(size, size, |x, y| {
        let (nx, ny) = centered(x, y, size);
        let radial = edge((nx * nx + ny * ny).sqrt(), 0.95, 0.2);
        let fine = bilinear_noise(nx * 5.0, ny * 5.0, 5, seed ^ 0x0FEE_D00D);
        let mut pit = 0.0f32;
        for &(px, py, pr) in &pits {
            let d = ((nx - px).powi(2) + (ny - py).powi(2)).sqrt();
            pit = pit.max(edge(d, pr, pr * 1.4) * (0.5 + 0.5 * fine));
        }
        let base = 0.9 + 0.1 * fine;
        radial * (base - 0.62 * pit)
    })
}

/// Deterministic 2-D value noise: bilinear interpolation of a hashed lattice
/// with `cell` divisions across the unit square, so all three textures render
/// identically every run.
fn bilinear_noise(x: f32, y: f32, cell: u32, seed: u32) -> f32 {
    let xi = x * cell as f32;
    let yi = y * cell as f32;
    let x0 = xi.floor();
    let y0 = yi.floor();
    let fx = xi - x0;
    let fy = yi - y0;
    let sx = fx * fx * (3.0 - 2.0 * fx);
    let sy = fy * fy * (3.0 - 2.0 * fy);
    let v = |ix: i64, iy: i64| {
        let mut h: u64 = (ix as u64).wrapping_mul(0x9E37_79B1_97F4_A7C7)
            ^ (iy as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
            ^ (seed as u64).wrapping_mul(0x1656_67B1_9E37_79B1);
        h ^= h >> 33;
        h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
        h ^= h >> 33;
        ((h & 0xFF_FFFF) as f32) / 16_777_216.0
    };
    let (ix0, iy0) = (x0 as i64, y0 as i64);
    let n00 = v(ix0, iy0);
    let n10 = v(ix0 + 1, iy0);
    let n01 = v(ix0, iy0 + 1);
    let n11 = v(ix0 + 1, iy0 + 1);
    let nx0 = n00 + (n10 - n00) * sx;
    let nx1 = n01 + (n11 - n01) * sx;
    nx0 + (nx1 - nx0) * sy
}

/// Maps pixel `(x, y)` to coordinates in `[-1, 1]` with the center at 0.
fn centered(x: u32, y: u32, size: u32) -> (f32, f32) {
    let n = (size - 1).max(1) as f32;
    (x as f32 / n * 2.0 - 1.0, y as f32 / n * 2.0 - 1.0)
}

/// 1 inside `radius`, fading to 0 over `soft` beyond it.
fn edge(dist: f32, radius: f32, soft: f32) -> f32 {
    ((radius - dist) / soft.max(1e-4)).clamp(0.0, 1.0)
}

/// Tiny deterministic LCG so generated brushes look the same every run.
struct Lcg(u32);

impl Lcg {
    fn new(seed: u32) -> Self {
        Self(seed.wrapping_mul(2_654_435_761).wrapping_add(1))
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / 16_777_216.0
    }
    fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.next_f32()
    }
}

// ---------------------------------------------------------------------------
// GIMP .gbr parsing
// ---------------------------------------------------------------------------

/// Parses a GIMP `.gbr` brush into a white coverage mask. The header (per the
/// GIMP `gbr.txt` spec) is a sequence of big-endian fields:
///
/// ```text
/// [ 0..4)  header_size  (28 + name length; 20 + name length in v1)
/// [ 4..8)  version      (1 or 2)
/// [ 8..12) width
/// [12..16) height
/// [16..20) depth        (1 = grayscale, 4 = RGBA)
/// [20..24) magic        "GIMP"  (v2 only)
/// [24..28) spacing      (v2 only)
/// [28..header_size)     brush name (v2); in v1 the name starts at byte 20
/// [header_size..)       pixel data, width*height*depth bytes
/// ```
///
/// Grayscale brushes store opacity per byte; RGBA brushes use the alpha
/// channel. Version 3 (CinePaint, 16-bit float) and animated `.gih` pipes are
/// not supported.
pub fn parse_gbr(bytes: &[u8]) -> Result<TextureData, String> {
    if bytes.len() < 20 {
        return Err("not a GIMP brush".to_string());
    }
    let version = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    if version != 1 && version != 2 {
        return Err(format!("unsupported GBR version {version}"));
    }
    // Only v2 carries the "GIMP" magic (v1 jumps straight to the name).
    if version == 2 && &bytes[20..24] != b"GIMP" {
        return Err("not a GIMP brush".to_string());
    }
    let header_size = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let width = u32::from_be_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    let height = u32::from_be_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
    let depth = u32::from_be_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
    if depth != 1 && depth != 4 {
        return Err(format!("unsupported GBR depth {depth}"));
    }
    if width == 0 || height == 0 {
        return Err("empty brush".to_string());
    }
    // Pixels start exactly after the header (header_size already includes the
    // brush-name bytes), for both v1 and v2.
    let pixels_start = header_size.max(20);
    let pixels = (width as usize).saturating_mul(height as usize);
    let data = &bytes[pixels_start.min(bytes.len())..];
    if data.len() < pixels * depth as usize {
        return Err("truncated brush data".to_string());
    }

    let stride = depth as usize;
    let mut rgba = vec![255u8; pixels * 4];
    for i in 0..pixels {
        let cov = if depth == 1 {
            data[i] as f32 / 255.0
        } else {
            data[i * stride + 3] as f32 / 255.0
        };
        rgba[i * 4 + 3] = (cov * 255.0 + 0.5) as u8;
    }
    Ok(TextureData {
        width,
        height,
        rgba,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a canonical `.gbr` header: u32 fields (header_size, version, width,
    /// height, depth), then for v2 the "GIMP" magic + spacing, then the name.
    fn gbr_header(w: u32, h: u32, depth: u32, version: u32, name: &str) -> Vec<u8> {
        let mut b = Vec::new();
        let name_overhead = if version == 2 { 28 } else { 20 };
        b.extend_from_slice(&((name_overhead + name.len()) as u32).to_be_bytes());
        b.extend_from_slice(&version.to_be_bytes());
        b.extend_from_slice(&w.to_be_bytes());
        b.extend_from_slice(&h.to_be_bytes());
        b.extend_from_slice(&depth.to_be_bytes());
        if version == 2 {
            b.extend_from_slice(b"GIMP");
            b.extend_from_slice(&75u32.to_be_bytes()); // spacing (ignored)
        }
        b.extend_from_slice(name.as_bytes());
        b
    }

    #[test]
    fn gbr_parses_grayscale_opacity() {
        let mut bytes = gbr_header(2, 2, 1, 2, "");
        bytes.extend_from_slice(&[0, 64, 128, 255]);
        let tex = parse_gbr(&bytes).expect("parse");
        assert_eq!((tex.width, tex.height), (2, 2));
        assert_eq!(tex.rgba[3], 0);
        assert_eq!(tex.rgba[7], 64);
        assert_eq!(tex.rgba[11], 128);
        assert_eq!(tex.rgba[15], 255);
    }

    #[test]
    fn gbr_parses_rgba_alpha() {
        let mut bytes = gbr_header(2, 1, 4, 2, "");
        bytes.extend_from_slice(&[10, 20, 30, 0, 40, 50, 60, 200]);
        let tex = parse_gbr(&bytes).expect("parse");

        // After the pixel data, texels should reflect the expected alpha.
        assert_eq!(tex.rgba[3], 0);
        assert_eq!(tex.rgba[7], 200);
    }

    #[test]
    fn gbr_v2_with_name_parses() {
        // The brush name lives between the header and the pixels; header_size
        // includes its length, so pixels must start at header_size.
        let mut bytes = gbr_header(2, 1, 4, 2, "chalky");
        bytes.extend_from_slice(&[10, 20, 30, 0, 40, 50, 60, 200]);
        let tex = parse_gbr(&bytes).expect("parse");
        assert_eq!((tex.width, tex.height), (2, 1));
        assert_eq!(tex.rgba[3], 0);
        assert_eq!(tex.rgba[7], 200);
    }

    #[test]
    fn gbr_v1_puts_name_right_after_the_depth() {
        // v1 skips magic+spacing, so the name starts at byte 20.
        let mut bytes = gbr_header(2, 2, 1, 1, "old");
        bytes.extend_from_slice(&[0, 64, 128, 255]);
        let tex = parse_gbr(&bytes).expect("parse");
        assert_eq!(tex.rgba[3], 0);
        assert_eq!(tex.rgba[7], 64);
        assert_eq!(tex.rgba[11], 128);
    }

    #[test]
    fn gbr_rejects_unknown_version() {
        let mut bytes = gbr_header(2, 2, 1, 9, "");
        bytes.extend_from_slice(&[0; 4]);
        assert!(parse_gbr(&bytes).is_err());
    }

    #[test]
    fn builtins_include_shapes_and_textures() {
        let lib = BrushLibrary::new(std::env::temp_dir().join("pixforge_no_such_brushes"));
        assert!(lib
            .entries
            .iter()
            .any(|e| matches!(e.kind, BrushKind::Shape(crate::paint::BrushShape::Round))));
        assert!(lib
            .entries
            .iter()
            .any(|e| matches!(e.kind, BrushKind::Texture)));
        assert_eq!(lib.categories()[0], BUILTIN_CATEGORY);
    }

    #[test]
    fn folder_scan_finds_files_by_category() {
        let dir =
            std::env::temp_dir().join(format!("pixforge_brushes_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("accent")).unwrap();
        let img = image::RgbaImage::from_pixel(4, 4, image::Rgba([255, 0, 0, 255]));
        img.save(dir.join("hard_round.png")).unwrap();
        img.save(dir.join("accent").join("dot.png")).unwrap();

        let mut lib = BrushLibrary::new(dir.clone());
        lib.refresh();

        assert!(lib
            .entries
            .iter()
            .any(|e| { e.name == "hard_round" && e.category == ROOT_CATEGORY }));
        assert!(lib
            .entries
            .iter()
            .any(|e| { e.name == "dot" && e.category == "accent" }));
        let dot = lib.entries.iter().find(|e| e.name == "dot").unwrap();
        assert!(matches!(dot.kind, BrushKind::Texture));

        // A rescan with no changes keeps the same entry list (no churn).
        let sig_before = lib.signature().to_string();
        lib.refresh();
        assert_eq!(lib.signature(), sig_before);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn ships_brush_folder_files_all_load() {
        // The repo's ./brushes folder (if present) must not hold files the
        // library silently fails on: every entry must carry a valid sprite.
        let dir = std::path::Path::new("brushes");
        if !dir.is_dir() {
            eprintln!("skipping: no ./brushes folder in the workdir");
            return;
        }
        let mut lib = BrushLibrary::new(dir.to_path_buf());
        lib.refresh();
        assert!(!lib.entries.is_empty(), "expected shipped brushes to load");
        let file_entries: Vec<_> = lib.entries.iter().filter(|e| e.path.is_some()).collect();
        assert!(!file_entries.is_empty(), "expected files under ./brushes");
        for e in &file_entries {
            assert!(
                e.sprite.width > 0 && e.sprite.height > 0 && !e.sprite.rgba.is_empty(),
                "brush {:?} ({}) has no sprite data",
                e.name,
                e.path.as_ref().unwrap().display()
            );
        }
        // Directory categories become Brush entries with those names.
        for dir_name in [
            "Spot",
            "Stroke",
            "Dots",
            "Spiky",
            "Grain",
            // New categories added in the second wave
            "Material",
        ] {
            assert!(
                lib.entries.iter().any(|e| e.category == dir_name),
                "expected a further {dir_name} category from ./brushes/{dir_name}"
            );
        }
    }
}
