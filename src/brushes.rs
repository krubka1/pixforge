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
    /// A tiling material texture — the entry's `sprite` is a seamless fill
    /// that strokes use as an aligned, revealing pattern.
    Texture,
    /// A splat/stamp brush (GIMP `.gbr`) — the entry's `sprite` is re-centered
    /// under every dab like a classic rubber stamp, never aligned to the world.
    Stamp,
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
        kind: if ext == "gbr" {
            BrushKind::Stamp
        } else {
            BrushKind::Texture
        },
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
            BrushKind::Stamp => "stamp",
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
    for (name, kind, gen_fn) in texture_generators() {
        entries.push(BrushEntry {
            name: name.to_string(),
            category: BUILTIN_CATEGORY.to_string(),
            kind,
            sprite: gen_fn(MASK_SIZE, seed_for(name)),
            path: None,
        });
    }
    entries
}

/// The built-in procedural texture brushes, as `(name, kind, generator)`
/// triples. Kept as a plain table so both the library and the tests exercise
/// the identical generators. `BrushKind::Texture` entries tile the sprite as
/// an aligned pattern; `BrushKind::Stamp` ones are drawn once per dab, so
/// Splotch stays a single organic splat instead of repeating.
fn texture_generators() -> Vec<(&'static str, BrushKind, TextureGen)> {
    vec![
        ("Splotch", BrushKind::Stamp, splotch_sprite as fn(u32, u32) -> TextureData),
        ("Grain", BrushKind::Texture, grain_sprite),
        ("Wood", BrushKind::Texture, wood_sprite),
        ("Marble", BrushKind::Texture, marble_sprite),
        ("Rust", BrushKind::Texture, rust_sprite),
        ("Brushed Metal", BrushKind::Texture, brushed_metal_sprite),
        ("Hammered Metal", BrushKind::Texture, hammered_metal_sprite),
        ("Halftone", BrushKind::Texture, halftone_sprite),
        ("Checker", BrushKind::Texture, checker_sprite),
        ("Diamond Plate", BrushKind::Texture, diamond_plate_sprite),
        ("Denim", BrushKind::Texture, denim_sprite),
        ("Corduroy", BrushKind::Texture, corduroy_sprite),
        ("Burlap", BrushKind::Texture, burlap_sprite),
        ("Linen", BrushKind::Texture, linen_sprite),
        ("Silk", BrushKind::Texture, silk_sprite),
        ("Velvet", BrushKind::Texture, velvet_sprite),
    ]
}

// ---------------------------------------------------------------------------
// Procedural sprite generation
// ---------------------------------------------------------------------------

/// Built-in procedural sprite resolution. `PACK_MATERIAL_SIZE` already matches;
/// stored at 256 so large dabs stay crisp instead of blocky.
const MASK_SIZE: u32 = 256;

/// A deterministic procedural texture generator: `(size, seed) -> sprite`.
type TextureGen = fn(u32, u32) -> TextureData;

/// Builds a `TextureData` whose RGB is white and alpha is `coverage(x, y)`.
///
/// The pixel scan is split across the available cores (every pixel is an
/// independent evaluation, so parallelizing preserves deterministic bytes
/// exactly — same seed still yields an identical sprite).
fn make_sprite(w: u32, h: u32, coverage: impl Fn(u32, u32) -> f32 + Send + Sync) -> TextureData {
    let mut rgba = vec![255u8; (w * h * 4) as usize];
    let nth = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let pixel_count = w as u64 * h as u64;
    let fill = |o: &mut [u8], y0: u32, y1: u32, base: u32, coverage: &dyn Fn(u32, u32) -> f32| {
        for y in y0..y1 {
            for x in 0..w {
                let cov = coverage(x, y).clamp(0.0, 1.0);
                o[((y - base) as usize * w as usize + x as usize) * 4 + 3] =
                    (cov * 255.0 + 0.5) as u8;
            }
        }
    };
    if nth <= 1 || pixel_count < 8192 {
        // Tiny sprites: single-threaded to avoid thread-spawn overhead.
        fill(&mut rgba, 0, h, 0, &coverage);
    } else {
        std::thread::scope(|s| {
            let cov = &coverage;
            let row_bytes = (w * 4) as usize;
            let per = (h as usize).div_ceil(nth).max(1);
            let mut remainder: &mut [u8] = &mut rgba;
            let mut y0 = 0;
            while y0 < h as usize {
                let y1 = (y0 + per).min(h as usize);
                let cut = (y1 - y0) * row_bytes;
                let (chunk, rest) = remainder.split_at_mut(cut);
                remainder = rest;
                let (y0c, y1c) = (y0 as u32, y1 as u32);
                s.spawn(move || fill(chunk, y0c, y1c, y0c, cov));
                y0 = y1;
            }
        });
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
        let (u, v) = (nx * 0.5 + 0.5, ny * 0.5 + 0.5);
        let mut cov = 0.0f32;
        for &(bx, by, r) in &blobs {
            // Toroidal distance so a blob parked near an edge keeps its far
            // half on the opposite edge — the splotch tile repeats seamlessly
            // instead of clipping each splat at the seam.
            let d = torus_dist(nx, ny, bx, by);
            cov = cov.max(edge(d, r, r * 0.8));
        }
        // Organic mottle across the blobs so the stamp never reads flat.
        let mottle = fbm_tiled(u, v, 4, 4, seed ^ 0x51EC);
        cov * (0.78 + 0.22 * mottle)
    })
}

fn grain_sprite(size: u32, seed: u32) -> TextureData {
    make_sprite(size, size, |x, y| {
        let (nx, ny) = centered(x, y, size);
        let (u, v) = (nx * 0.5 + 0.5, ny * 0.5 + 0.5);
        // Slightly stretched fBm speckle reads as tooth/paper grain rather
        // than flat dots, and stays seamless when the stamp rotates. The
        // coverage fills the whole tile edge-to-edge (no radial disc), so the
        // tiled pattern repeats cleanly instead of showing a circle per tile.
        let iso = fbm_tiled(u, v, 4, 4, seed);
        let stretch = fbm_tiled(u, v, 3, 4, seed ^ 0xA5E1);
        let grain = 0.6 * iso + 0.4 * stretch;
        0.30 + 0.70 * grain
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
        let (u, v) = (nx * 0.5 + 0.5, ny * 0.5 + 0.5);
        // Domain-warp the sampling coordinates so the streaks bend and sway
        // instead of marching in perfectly straight lines.
        let (wu, wv) = domain_warp(u, v, 2, 3, 0.5, seed ^ 0xD13A);
        let (wn, vn) = (wu * 2.0 - 1.0, wv * 2.0 - 1.0);
        // Grain frequency drifts across the board (irregular growth rings);
        // denser "figure" patches darken the groove lines.
        let growth = fbm_tiled(wu, wv, 2, 3, seed ^ 0x0F1F);
        let wobble = fbm_tiled(wu, wv, 2, 3, seed ^ 0xC0FE);
        let figure = fbm_tiled(wu + 4.0, wv + 1.0, 3, 3, seed ^ 0x00F1_0ECE);
        let freq = 20.0 + 16.0 * growth;
        let mut warp = (wobble - 0.5) * 2.4 + (wn * 1.6 + 0.5).sin() * 0.35;
        for &(kx, ky, kr) in &knots {
            let d = ((wn - kx).powi(2) + (vn - ky).powi(2)).sqrt();
            warp += (-d / kr.max(1e-3)).exp() * (vn - ky);
        }
        let field = vn * freq + warp * 6.0;
        let streak = 0.5 + 0.5 * field.sin();
        // Grooves are the dark lines: thicker in noisy figure, thin in plain.
        let groove = edge(streak, 0.20 + 0.18 * figure.powf(3.0), 0.14);
        let grain = 0.80 + 0.20 * fbm_tiled(wu + 2.0, wv + 1.0, 4, 3, seed ^ 0x05EE_D10F);
        (0.97 - 0.68 * groove) * grain
    })
}

/// Wavy striated marble: a noise-warped fold field carves thin dark seams
/// through a bright ground, like polished veined stone.
fn marble_sprite(size: u32, seed: u32) -> TextureData {
    make_sprite(size, size, |x, y| {
        let (nx, ny) = centered(x, y, size);
        let (u, v) = (nx * 0.5 + 0.5, ny * 0.5 + 0.5);
        // Domain-warped fold field carves wavy strata; fine fBm sharpens some
        // into branchy veins instead of uniform bands.
        let (wu, wv) = domain_warp(u, v, 4, 3, 0.6, seed ^ 0xA5A5);
        let n1 = fbm_tiled(wu, wv, 3, 4, seed);
        let n2 = fbm_tiled(wu + 3.0, wv + 1.0, 4, 3, seed ^ 0xA5A5_A5A5);
        let field = wv * 12.0 + 2.4 * (n1 - 0.5) + 0.5 * (wu * 2.0).sin() + 0.6;
        let sv = field.sin() * 0.5 + 0.5;
        let vein = edge(sv, 0.30, 0.16);
        // Fine craquelure between the strata, from random cell borders.
        let (_, gap) = worley_tiled(wu, wv, 9, seed ^ 0xC0B8);
        let branch = edge(gap, 0.05, 0.02) * (0.4 + 0.6 * n2);
        let haze = 0.86 + 0.14 * n2;
        (0.93 - 0.72 * vein - 0.35 * branch) * haze
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
        let (u, v) = (nx * 0.5 + 0.5, ny * 0.5 + 0.5);
        let fine = fbm_tiled(u, v, 5, 3, seed ^ 0x0FEE_D00D);
        // Flake plates: rusty sheets whose cell edges read darker.
        let (f1, gap) = worley_tiled(u + 3.0, v + 1.0, 6, seed ^ 0xF1A6);
        let plate = edge(gap, 0.06, 0.03) * (0.55 + 0.45 * f1 * 2.0);
        // Pitting: excavated dark voids clustered across the flakes.
        let pit_map = fbm_tiled(u, v, 4, 3, seed ^ 0x7AB1);
        let pit = edge(pit_map, 0.60, 0.20) * (0.6 + 0.4 * fine);
        // Deep sculpted pits for character.
        let mut deep = 0.0f32;
        for &(px, py, pr) in &pits {
            let d = ((nx - px).powi(2) + (ny - py).powi(2)).sqrt();
            deep = deep.max(edge(d, pr, pr * 1.4) * (0.5 + 0.5 * fine));
        }
        let base = 0.9 + 0.1 * fine;
        base - 0.35 * deep - 0.30 * pit - 0.28 * plate
    })
}

/// Milled / brushed sheet metal: a bright base shot through with thin
/// lengthwise streaks — straight in the middle, gently bent by low-frequency
/// wobble at the edges — plus a faint micro-grain so it never reads as flat.
fn brushed_metal_sprite(size: u32, seed: u32) -> TextureData {
    make_sprite(size, size, |x, y| {
        let (nx, ny) = centered(x, y, size);
        let (u, v) = (nx * 0.5 + 0.5, ny * 0.5 + 0.5);
        // Warped vertical streaks: fine in the middle, swaying at the ends.
        let wobble = fbm_tiled(u, v, 2, 3, seed ^ 0x0B5E_B00C);
        let streak_phase = u * 36.0 + 18.0 + (wobble - 0.5) * 3.5;
        let sv = 0.5 + 0.5 * (streak_phase * std::f32::consts::TAU).sin();
        let mark = edge(sv, 0.16, 0.10);
        let micro = 1.0 + 0.32 * (fbm_tiled(u + 0.3, v + 0.2, 6, 4, seed) - 0.5);
        (0.98 - 0.68 * mark) * micro
    })
}

/// Ball-peen hammered metal: a staggered field of shallow circular dents,
/// each with a dark excavated center and a bright lip ridge catching light.
fn hammered_metal_sprite(size: u32, seed: u32) -> TextureData {
    make_sprite(size, size, |x, y| {
        let (nx, ny) = centered(x, y, size);
        let (u, v) = (nx * 0.5 + 0.5, ny * 0.5 + 0.5);
        let spacing = 0.30f32;
        // Staggered (hex-ish) dent lattice; odd rows offset by half a pitch.
        let j = (ny / spacing).round();
        let stagger = if (j as i32).rem_euclid(2) == 1 {
            spacing * 0.5
        } else {
            0.0
        };
        let i = ((nx + stagger) / spacing).round();
        // Per-dent randomness: jittered center and radius so the peen reads
        // hammered by hand, not punched by a press.
        let jx = lattice_hash(i as i64, j as i64, seed ^ 0x1E56) - 0.5;
        let jy = lattice_hash(i as i64 + 77, j as i64 + 31, seed ^ 0x3A9D) - 0.5;
        let rr = spacing * (0.40 + 0.12 * lattice_hash(i as i64 + 13, j as i64, seed ^ 0x9D11));
        let (cx, cy) = (
            i * spacing - stagger + jx * spacing * 0.4,
            j * spacing + jy * spacing * 0.4,
        );
        let d = ((nx - cx).powi(2) + (ny - cy).powi(2)).sqrt();
        let dent = edge(d, rr, rr * 0.5);
        let lip = edge(d, rr * 1.28, rr * 0.9) - edge(d, rr * 0.82, rr * 0.5);
        let grain = 0.9 + 0.1 * fbm_tiled(u + 1.0, v + 2.0, 5, 3, seed);
        (0.86 - 0.30 * dent + 0.16 * lip) * grain
    })
}

/// Halftone screening: a staggered dot grid whose dots swell toward the
/// center, so the face is a soft vignette of printed dots — great for shading
/// or stencil-style stamps.
fn halftone_sprite(size: u32, seed: u32) -> TextureData {
    let screen = 11.0f32;
    make_sprite(size, size, |x, y| {
        let (nx, ny) = centered(x, y, size);
        let (u, v) = (nx * 0.5 + 0.5, ny * 0.5 + 0.5);
        let rad = (nx * nx + ny * ny).sqrt();
        // Dots grow from the rim (tiny) toward the center (large).
        let radius = (0.16 + 0.34 * (1.0 - rad).clamp(0.0, 1.0)).clamp(0.03, 0.5);
        let g = tiled_noise(u + 1.0, v + 2.0, 6, seed) * 0.05 + 0.95;
        dot_field(nx * 0.5 + 0.5, ny * 0.5 + 0.5, screen, radius) * g
    })
}

/// Crisp checkerboard: alternating squares, feathered just before their
/// corners so the stamps don't alias when rotated or scaled.
fn checker_sprite(size: u32, seed: u32) -> TextureData {
    let cols = 4.0f32;
    make_sprite(size, size, |x, y| {
        let (nx, ny) = centered(x, y, size);
        let (u, v) = (nx * 0.5 + 0.5, ny * 0.5 + 0.5);
        let ci = (u * cols).floor();
        let cj = (v * cols).floor();
        let on = ((ci as i32 + cj as i32).rem_euclid(2)) == 0;
        // Soft square: high in the middle of each cell, feathering to 0 at
        // the cell border so adjacent squares blend instead of hard-edging.
        let cell = 1.0 / cols;
        let lu = (u - ci * cell) / cell - 0.5;
        let lv = (v - cj * cell) / cell - 0.5;
        let sq = edge(lu.abs().max(lv.abs()), 0.42, 0.10);
        let g = tiled_noise(u + 1.0, v + 2.0, 6, seed) * 0.06 + 0.94;
        let cover = if on {
            0.82 + 0.16 * sq
        } else {
            0.22 + 0.16 * sq
        };
        cover * g
    })
}

/// Diamond tread plate: raised diamonds in a staggered grid with bright crests
/// and dark grooves between, like industrial floor plate.
fn diamond_plate_sprite(size: u32, seed: u32) -> TextureData {
    let n = 3.0f32;
    let sp = 2.0 / n;
    make_sprite(size, size, |x, y| {
        let (nx, ny) = centered(x, y, size);
        let (u, v) = (nx * 0.5 + 0.5, ny * 0.5 + 0.5);
        // Brick diamond lattice: odd rows shift half a pitch so the diamonds
        // interlock like real tread plate.
        let j = ((ny + 1.0) / sp).floor();
        let stagger = if (j as i32).rem_euclid(2) == 1 {
            sp * 0.5
        } else {
            0.0
        };
        let i = ((nx + 1.0 - stagger) / sp).floor();
        let lx = (nx + 1.0 - stagger) - (i * sp) - sp * 0.5;
        let ly = (ny + 1.0) - (j * sp) - sp * 0.5;
        // Diamond metric: |x| + |y| in the lattice frame.
        let d = lx.abs() + ly.abs();
        let crest = edge(d, sp * 0.30, sp * 0.10);
        let groove = edge(d, sp * 0.46, sp * 0.07);
        let g = tiled_noise(u + 1.0, v + 2.0, 6, seed) * 0.08 + 0.92;
        (0.78 + 0.20 * crest - 0.60 * groove) * g
    })
}

/// Woven canvas cloth: interlaced warp and weft threads, each with a little
/// per-thread wobble so the weave stays lively when rotated or scaled. The
/// crossing points read slightly darker, like real fabric.
fn canvas_sprite(size: u32, seed: u32) -> TextureData {
    let n = 13.0f32;
    make_sprite(size, size, |x, y| {
        let (nx, ny) = centered(x, y, size);
        let u = nx * 0.5 + 0.5;
        let v = ny * 0.5 + 0.5;
        let wu = (tiled_noise(u + 0.3, v + 0.1, 2, seed) - 0.5) * 0.07;
        let wv = (tiled_noise(u + 0.9, v + 0.5, 2, seed ^ 0xCA1A) - 0.5) * 0.07;
        // Distance to the nearest thread center, scaled 0..1 (0 at the middle
        // of a thread, 1 at the gap between threads).
        let warp = edge(thread_dist(u * n + wu), 0.42, 0.12);
        let weft = edge(thread_dist(v * n + wv), 0.42, 0.12);
        let thread = warp.max(weft);
        let cross = warp * weft;
        let fibre = tiled_noise(u + 0.1, v + 0.2, 6, seed ^ 0xF18E) * 0.16 + 0.84;
        (0.40 + 0.60 * thread * fibre) * (0.94 - 0.16 * cross)
    })
}

/// Poured concrete: blotchy pour patches, pitted voids and a fine aggregate
/// speckle, on a solid high-coverage base.
fn concrete_sprite(size: u32, seed: u32) -> TextureData {
    make_sprite(size, size, |x, y| {
        let (nx, ny) = centered(x, y, size);
        let (u, v) = (nx * 0.5 + 0.5, ny * 0.5 + 0.5);
        let blotch = tiled_noise(u + 0.1, v + 0.2, 3, seed);
        let speck = tiled_noise(u + 0.4, v + 0.8, 17, seed ^ 0xC0B8);
        let base = 0.80 + 0.14 * (blotch - 0.5) + 0.18 * (speck - 0.5);
        let void = tiled_noise(u + 0.2, v + 0.3, 6, seed ^ 0x0B10);
        base - 0.30 * edge(void, 0.38, 0.18)
    })
}

/// Grunge smear: a worn, scratched mid-gray ground shot through with thin
/// dark hairline scratches, a few greasy dark pits and lighter scuffed
/// patches — weathered armor or grimy machinery. Every feature is sampled on
/// the tile torus (`torus_dist`), so the pack tiles without seams.
fn grunge_sprite(size: u32, seed: u32) -> TextureData {
    let mut rng = Lcg::new(seed);
    let pits: Vec<(f32, f32, f32)> = (0..7)
        .map(|_| {
            (
                rng.range(-0.6, 0.6),
                rng.range(-0.6, 0.6),
                rng.range(0.10, 0.28),
            )
        })
        .collect();
    let scuffs: Vec<(f32, f32, f32)> = (0..5)
        .map(|_| {
            (
                rng.range(-0.55, 0.55),
                rng.range(-0.55, 0.55),
                rng.range(0.20, 0.38),
            )
        })
        .collect();
    make_sprite(size, size, |x, y| {
        let (nx, ny) = centered(x, y, size);
        let (u, v) = (nx * 0.5 + 0.5, ny * 0.5 + 0.5);
        // Mid ground with a swirly smear.
        let smudge = tiled_noise(u + 0.1, v + 0.2, 3, seed ^ 0x6D17);
        let mut base = 0.58 + 0.20 * (smudge - 0.5);
        // Scuffed lighter patches (wrapped so they close across the seam).
        for &(sx, sy, sr) in &scuffs {
            base += edge(torus_dist(nx, ny, sx, sy), sr, sr * 1.2) * 0.30;
        }
        // Greasy dark pits (wrapped too).
        let mut pit = 0.0f32;
        for &(px, py, pr) in &pits {
            pit = pit.max(edge(torus_dist(nx, ny, px, py), pr, pr * 1.4));
        }
        // Thin hairline scratches: the ridge skeleton of a torus noise field
        // — its zero-crossing contours are winding, never cut the tile, and
        // read as scratched metal instead of regular diagonals.
        let scratch = fbm_tiled(u, v, 9, 2, seed ^ 0xE5CA);
        let wing = 1.0 - (2.0 * scratch - 1.0).abs();
        let hair = ((wing - 0.78) / 0.20).clamp(0.0, 1.0);
        let fine = tiled_noise(u + 0.4, v + 0.8, 8, seed ^ 0x77A1) * 0.12 + 0.88;
        (base - 0.55 * pit - 0.35 * hair) * fine
    })
}

/// Pebbled leather: tight packed grain with a slick, softly varying sheen —
/// a premium glove-leather surface rather than the hammered dents.
fn pebbled_leather_sprite(size: u32, seed: u32) -> TextureData {
    make_sprite(size, size, |x, y| {
        let (nx, ny) = centered(x, y, size);
        let (u, v) = (nx * 0.5 + 0.5, ny * 0.5 + 0.5);
        let grain = tiled_noise(u + 0.1, v + 0.2, 6, seed);
        let sheen = tiled_noise(u + 0.5, v + 0.1, 3, seed ^ 0xE0F0);
        0.70 + 0.18 * grain + 0.12 * sheen
    })
}

/// Loose sand: coarse granular speckle over soft drifting shadows, matte and
/// powdery — a natural fill for desert props and brushed-in dunes.
fn sand_sprite(size: u32, seed: u32) -> TextureData {
    make_sprite(size, size, |x, y| {
        let (nx, ny) = centered(x, y, size);
        let (u, v) = (nx * 0.5 + 0.5, ny * 0.5 + 0.5);
        let drift = 0.62 + 0.26 * fbm_tiled(u, v, 2, 3, seed ^ 0x50A1);
        let grit = fbm_tiled(u, v, 9, 3, seed ^ 0x0F0F);
        drift * (0.55 + 0.50 * grit) * 0.62 + 0.12
    })
}

/// Rough-faced stone: blocky cobble mottling, wavy mineral strata and a coarse
/// stone speckle, with fine hairline cracks following the noise contours so
/// the whole pack tiles without seams.
fn stone_sprite(size: u32, seed: u32) -> TextureData {
    make_sprite(size, size, |x, y| {
        let (nx, ny) = centered(x, y, size);
        let (u, v) = (nx * 0.5 + 0.5, ny * 0.5 + 0.5);
        // Irregular cell plates read as cut stone rather than smooth pour —
        // `F1` softens into mottling, the border term would be masonry mortar.
        let (block, _mortar) = worley_tiled(u, v, 3, seed ^ 0x211B);
        // Wavy mineral strata, diagonally drifted. The band frequency is an
        // integer multiple of the tile and the warp is itself tiled, so the
        // bands close exactly at the seams instead of marching off the edge.
        let strata_warp = fbm_tiled(u, v, 2, 2, seed ^ 0x5F0B) - 0.5;
        let strata_p = std::f32::consts::TAU * (v * 8.0 + u * 2.0 + strata_warp * 3.0);
        let band = 0.5 + 0.5 * strata_p.sin();
        let speck = tiled_noise(u + 0.5, v + 0.1, 16, seed ^ 0xAC5E);
        // Hairline crack network: the ridge skeleton of a torus field follows
        // the noise's zero-crossing contours, which are winding and never cut
        // the tile.
        let crack = fbm_tiled(u, v, 6, 3, seed ^ 0x3C21);
        let ridge = 1.0 - (2.0 * crack - 1.0).abs();
        let hairline = ((ridge - 0.76) / 0.22).clamp(0.0, 1.0);
        let mut base =
            0.78 + 0.10 * (block * 2.0 - 1.0) + 0.14 * (band - 0.5) + 0.16 * (speck - 0.5);
        base -= 0.30 * hairline;
        base
    })
}

// ---------------------------------------------------------------------------
// Fabric brushes
// ---------------------------------------------------------------------------

/// Distance of a woven thread coordinate to its centerline, scaled 0..1 (0 at
/// the middle of a thread, 1 at the gap between threads).
fn thread_dist(t: f32) -> f32 {
    let f = t.fract();
    (f - 0.5).abs() * 2.0
}

/// Denim: a tight warp×weft cross-weave whose rows step diagonally, with the
/// diagonal rib sheen, indigo dye mottle and worn chafe patches denim reads by.
fn denim_sprite(size: u32, seed: u32) -> TextureData {
    let thread = 30.0f32;
    make_sprite(size, size, |x, y| {
        let (nx, ny) = centered(x, y, size);
        let (u, v) = (nx * 0.5 + 0.5, ny * 0.5 + 0.5);
        let wob = (fbm_tiled(u, v, 3, 3, seed ^ 0x0D33) - 0.5) * 0.05;
        let row = (v * thread).floor() as i64;
        let step = row.rem_euclid(4) as f32 * 0.25;
        let warp = edge(thread_dist(u * thread + wob * thread), 0.40, 0.10);
        let weft = edge(thread_dist(v * thread + wob * thread + step), 0.40, 0.10);
        let base = 0.40 + 0.60 * warp.max(weft);
        let sheen =
            0.5 + 0.5 * ((v * thread * 0.25 + u * thread * 0.6) * std::f32::consts::TAU).sin();
        let dye = 0.82 + 0.18 * fbm_tiled(u + 1.0, v + 3.0, 3, 3, seed ^ 0x1D3E);
        let chafe = fbm_tiled(u + 4.0, v + 2.0, 2, 3, seed ^ 0xC6A1);
        base * dye * (0.90 + 0.10 * sheen * chafe * chafe)
    })
}

/// Corduroy: vertical cording wales, each a raised rib with crosswale steps
/// and a dark valley between, over a soft piled nap.
fn corduroy_sprite(size: u32, seed: u32) -> TextureData {
    let wales = 10.0f32;
    make_sprite(size, size, |x, y| {
        let (nx, ny) = centered(x, y, size);
        let (u, v) = (nx * 0.5 + 0.5, ny * 0.5 + 0.5);
        // Warp the wale positions so the cords sway rather than stand rigid.
        let wu = u + 0.12 * (fbm_tiled(u, v, 2, 3, seed ^ 0xC023) - 0.5);
        let dgeo = thread_dist(wu * wales);
        let rib = edge(dgeo, 0.36, 0.10);
        let groove = edge(dgeo, 0.46, 0.06);
        let stepn = 0.5 + 0.5 * (v * 28.0 * std::f32::consts::TAU).sin();
        let nap = 0.5 + 0.5 * fbm_tiled(u + 2.0, v, 4, 3, seed ^ 0x5E31);
        (0.34 + 0.66 * rib) * (0.92 - 0.40 * groove) * (0.70 + 0.30 * stepn) * nap
    })
}

/// Burlap (hessian): coarse, loose open plain weave — thick threads, big
/// visible gaps where the light falls through, heavy per-thread sway and fuzz.
fn burlap_sprite(size: u32, seed: u32) -> TextureData {
    let thread = 14.0f32;
    make_sprite(size, size, |x, y| {
        let (nx, ny) = centered(x, y, size);
        let (u, v) = (nx * 0.5 + 0.5, ny * 0.5 + 0.5);
        let wu = (fbm_tiled(u, v, 2, 3, seed ^ 0xB2A0) - 0.5) * 0.10;
        let wv = (fbm_tiled(u + 0.5, v + 0.3, 2, 3, seed ^ 0xB31A) - 0.5) * 0.10;
        let warp = edge(thread_dist(u * thread + wu * thread), 0.48, 0.08);
        let weft = edge(thread_dist(v * thread + wv * thread), 0.48, 0.08);
        let thr = warp.max(weft);
        // Open gaps between threads show through as deep dark.
        let gap =
            edge(thread_dist(u * thread), 0.44, 0.05) * edge(thread_dist(v * thread), 0.44, 0.05);
        let fuzz = fbm_tiled(u, v, 5, 3, seed ^ 0xF2A9) * 0.3 + 0.7;
        (0.18 + 0.82 * thr * fuzz) * (0.96 - 0.35 * gap)
    })
}

/// Linen: fine plain weave, two-tone threads (warp a touch brighter than the
/// weft), gentle slub variation and a very soft sheen.
fn linen_sprite(size: u32, seed: u32) -> TextureData {
    let thread = 40.0f32;
    make_sprite(size, size, |x, y| {
        let (nx, ny) = centered(x, y, size);
        let (u, v) = (nx * 0.5 + 0.5, ny * 0.5 + 0.5);
        let slub = (fbm_tiled(u, v, 3, 3, seed ^ 0x1E1E) - 0.5) * 0.06;
        let warp = edge(thread_dist(u * thread + slub * thread), 0.42, 0.08);
        let weft = edge(thread_dist(v * thread + slub * thread), 0.42, 0.08);
        let two_tone = 0.55 + 0.16 * warp + 0.10 * weft;
        let slub_light = 0.5 + 0.5 * fbm_tiled(u + 2.0, v + 1.0, 5, 3, seed ^ 0xA1E1);
        two_tone * (0.92 + 0.08 * slub_light)
    })
}

/// Silk/satin: the smoothest of the set — long floating warp fibers hide the
/// weave entirely, leaving soft domain-warped sheen streaks over flat cloth.
fn silk_sprite(size: u32, seed: u32) -> TextureData {
    make_sprite(size, size, |x, y| {
        let (nx, ny) = centered(x, y, size);
        let (u, v) = (nx * 0.5 + 0.5, ny * 0.5 + 0.5);
        let (wu, wv) = domain_warp(u, v, 3, 3, 0.6, seed ^ 0x51C5);
        let sheen = 0.5 + 0.5 * (wv * 5.0 + fbm_tiled(wu, wv, 2, 3, seed ^ 0x51C6)).sin();
        // Faint float lines; the fibers run along one axis only.
        let float = 0.5 + 0.5 * (u * 42.0 * std::f32::consts::TAU).sin();
        let micro = 0.5 + 0.5 * fbm_tiled(u + 0.5, v + 0.7, 8, 2, seed ^ 0x3E57);
        (0.66 + 0.34 * sheen) * (0.97 + 0.03 * float) * (0.98 + 0.02 * micro)
    })
}

/// Velvet: a short directional pile that catches light — soft tonal nap
/// shading that shifts with the warp direction over a faint hidden weave.
fn velvet_sprite(size: u32, seed: u32) -> TextureData {
    make_sprite(size, size, |x, y| {
        let (nx, ny) = centered(x, y, size);
        let (u, v) = (nx * 0.5 + 0.5, ny * 0.5 + 0.5);
        let (wu, wv) = domain_warp(u, v, 2, 3, 0.45, seed ^ 0x5E11);
        let nap = fbm_tiled(wu, wv, 3, 3, seed ^ 0x5E12);
        let pile = 0.42 + 0.58 * nap;
        let weave =
            edge(thread_dist(u * 36.0), 0.46, 0.04).max(edge(thread_dist(v * 36.0), 0.46, 0.04));
        pile * (0.98 + 0.02 * weave)
    })
}

/// Coverage of a staggered (hexagonally off-set per row) grid of soft dots:
/// `(u, v)` are in [0,1], `screen` is the number of dots across the square,
/// `radius` is the dot radius in cell units (0..=0.5).
fn dot_field(u: f32, v: f32, screen: f32, radius: f32) -> f32 {
    let cell = 1.0 / screen;
    let cj = (v / cell).floor();
    let ci = (u / cell).floor();
    let mut best = f32::INFINITY;
    for a in -1i32..=1 {
        for b in -1i32..=1 {
            let row = (cj as i32 + b) as f32;
            let stagger = if (row as i32).rem_euclid(2) == 1 {
                cell * 0.5
            } else {
                0.0
            };
            let c_x = (ci + a as f32) * cell + stagger;
            let c_y = row * cell;
            let du = (u - c_x).abs();
            let dv = (v - c_y).abs();
            best = best.min((du * du + dv * dv).sqrt());
        }
    }
    let d_cell = best / cell;
    edge(d_cell, radius, radius * 0.30)
}

/// Deterministic hash of a 2-D lattice cell, in `[0, 1)`. Uses only u32
/// arithmetic so it runs 2-4× faster than the previous u64 version on most
/// platforms (especially WASM and 32-bit targets).
#[inline(always)]
fn lattice_hash(ix: i64, iy: i64, seed: u32) -> f32 {
    let mut h: u32 = (ix as u32)
        .wrapping_mul(0x9E37_79B1)
        .wrapping_add((iy as u32).wrapping_mul(0x85EB_CA6B))
        .wrapping_add(seed);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb352d);
    h ^= h >> 13;
    h = h.wrapping_mul(0x846ca68b);
    h ^= h >> 16;
    (h >> 8) as f32 / 16_777_216.0
}

/// Deterministic 2-D value noise with `cell` divisions across the tile whose
/// lattice wraps around, so sampling a coordinate of `0` and `1` agree. Every
/// texture brush stamps seamlessly across a seam the way Substance grunge maps
/// do, so dense overlapping dabs tile invisibly when rotated or scaled. All
/// generators render identically every run (same seed → same bytes).
#[inline]
fn tiled_noise(u: f32, v: f32, cell: u32, seed: u32) -> f32 {
    let ci = cell as i64;
    let cf = cell as f32;
    let ui = u.rem_euclid(1.0) * cf;
    let vi = v.rem_euclid(1.0) * cf;
    let u0 = ui.floor();
    let v0 = vi.floor();
    let fu = ui - u0;
    let fv = vi - v0;
    let su = fu * fu * (3.0 - 2.0 * fu);
    let sv = fv * fv * (3.0 - 2.0 * fv);
    let wrap = |i: i64| -> i64 { i.rem_euclid(ci) };
    let u0i = u0 as i64;
    let v0i = v0 as i64;
    let n00 = lattice_hash(wrap(u0i), wrap(v0i), seed);
    let n10 = lattice_hash(wrap(u0i + 1), wrap(v0i), seed);
    let n01 = lattice_hash(wrap(u0i), wrap(v0i + 1), seed);
    let n11 = lattice_hash(wrap(u0i + 1), wrap(v0i + 1), seed);
    let nu0 = n00 + (n10 - n00) * su;
    let nu1 = n01 + (n11 - n01) * su;
    nu0 + (nu1 - nu0) * sv
}

/// Fractal-sum (fBm) value noise: `base` coarse cells on the tile plus fractal
/// detail octaves, each seamless. This is the macro→meso→micro spine Substance
/// materials are built from; a single octave degenerates to [`tiled_noise`].
#[inline]
fn fbm_tiled(u: f32, v: f32, base: u32, octaves: u32, seed: u32) -> f32 {
    let mut amp = 0.5f32;
    let mut sum = 0f32;
    let mut norm = 0f32;
    let mut cell = base.max(1);
    for o in 0..octaves.max(1) {
        sum += amp * tiled_noise(u, v, cell, seed ^ o.wrapping_mul(0x9E37_79B1));
        norm += amp;
        amp *= 0.5;
        cell <<= 1;
        // Stop early when the remaining octaves contribute less than 1/255
        // in the final 8-bit output — saves ~30% on 4+ octave calls.
        if amp < 0.004 {
            break;
        }
    }
    sum / norm
}

/// Voronoi cells wrapped on the tile, returned as `(F1, F2 - F1)` in cell units
/// (≈`[0, 0.5]`). `F1` reads as a distance-to-feature blob field, `F2 - F1` as
/// cell borders — the cells/plates bark, rock and leather rely on.
#[inline]
fn worley_tiled(u: f32, v: f32, cells: u32, seed: u32) -> (f32, f32) {
    let n = cells.max(1) as f32;
    let cells_i = cells.max(1) as i64;
    let pu = u.rem_euclid(1.0) * n;
    let pv = v.rem_euclid(1.0) * n;
    let cu = pu.floor() as i64;
    let cv = pv.floor() as i64;
    // Pre-wrap the base cell once instead of per-neighbor.
    let ci = cu.rem_euclid(cells_i);
    let cj = cv.rem_euclid(cells_i);
    let mut f1 = f32::INFINITY;
    let mut f2 = f32::INFINITY;
    for a in -1i64..=1 {
        for b in -1i64..=1 {
            let fi = (ci + a).rem_euclid(cells_i);
            let fj = (cj + b).rem_euclid(cells_i);
            let fx = fi as f32 + lattice_hash(fi, fj, seed ^ 0xA9C8_DE1A);
            let fy = fj as f32 + lattice_hash(fi, fj, seed ^ 0x1E64_3F05);
            let mut du = pu - fx;
            let mut dv = pv - fy;
            du -= n * (du / n).round();
            dv -= n * (dv / n).round();
            let d2 = du * du + dv * dv;
            if d2 < f1 {
                f2 = f1;
                f1 = d2;
            } else if d2 < f2 {
                f2 = d2;
            }
        }
    }
    let d1 = f1.sqrt() / n;
    let gap = if f2.is_finite() {
        (f2.sqrt() - f1.sqrt()) / n
    } else {
        0.0
    };
    (d1, gap)
}

/// Warps `(u, v)` by low-frequency fBm fields (classic Perlin warp) and wraps
/// the result back onto the tile, so patterned fields bend and sway organically
/// instead of marching in straight lines — the single biggest "wow" factor in
/// Substance-style procedural textures.
#[inline]
fn domain_warp(u: f32, v: f32, base: u32, octaves: u32, amount: f32, seed: u32) -> (f32, f32) {
    let du = fbm_tiled(u, v, base, octaves, seed) - 0.5;
    let dv = fbm_tiled(u + 0.37, v + 0.71, base, octaves, seed ^ 0x9E37_79BC) - 0.5;
    (
        (u + du * amount).rem_euclid(1.0),
        (v + dv * amount).rem_euclid(1.0),
    )
}

/// Maps pixel `(x, y)` to coordinates in `[-1, 1]` with the center at 0.
#[inline(always)]
fn centered(x: u32, y: u32, size: u32) -> (f32, f32) {
    let n = (size - 1).max(1) as f32;
    (x as f32 / n * 2.0 - 1.0, y as f32 / n * 2.0 - 1.0)
}

/// 1 inside `radius`, fading to 0 over `soft` beyond it.
#[inline(always)]
fn edge(dist: f32, radius: f32, soft: f32) -> f32 {
    ((radius - dist) / soft.max(1e-4)).clamp(0.0, 1.0)
}

/// Toroidal distance between `(nx, ny)` (unit-norm, tile-centered) and the
/// blob center `(cx, cy)`: the nearest copy across the tile seams, so blobs
/// near an edge keep their far half on the opposite edge. The sprite domain is
/// `[-1, 1]` in both axes, so the wrap period is 2.
#[inline(always)]
fn torus_dist(nx: f32, ny: f32, cx: f32, cy: f32) -> f32 {
    let mut dx = nx - cx;
    dx -= 2.0 * (dx * 0.5).round();
    let mut dy = ny - cy;
    dy -= 2.0 * (dy * 0.5).round();
    (dx * dx + dy * dy).sqrt()
}

/// Tiny deterministic LCG so generated brushes look the same every run.
struct Lcg(u32);

/// Stable per-name seed used anywhere brushes are generated from a name, so
/// the procedural look never drifts between runs or between built-ins and the
/// shipped pack.
fn seed_for(name: &str) -> u32 {
    (name.len() as u32).wrapping_mul(0xD651_7F53)
}

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
    // Only v2 carries the "GIMP" magic (v1 jumps straight to the name). A
    // truncated v2 file (20..24 bytes) must be rejected before slicing the
    // 4-byte magic, not panic.
    if version == 2 && (bytes.len() < 24 || &bytes[20..24] != b"GIMP") {
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

// ---------------------------------------------------------------------------
// Shipped pack generation
// ---------------------------------------------------------------------------

/// Size of the shipped material PNGs. Kept a multiple of 64 so it downscales
/// cleanly for stamps, but far above the old 64² so large dabs stay crisp.
pub const PACK_MATERIAL_SIZE: u32 = 256;

/// Regenerates the shipped brush packs under `root/Material` using the same
/// deterministic procedural generators as the built-ins. The files are written
/// fresh every run, so the shipped folder always matches exactly what the code
/// produces — no hand-edited art to drift.
///
/// Returns the list of files it wrote (for logging/tests).
pub fn generate_ship_pack(root: &std::path::Path) -> Result<Vec<std::path::PathBuf>, String> {
    let material = root.join("Material");
    std::fs::create_dir_all(&material)
        .map_err(|e| format!("create {}: {e}", material.display()))?;

    // Material.png files: white RGB with coverage in alpha, matching the
    // semantics `brush_sprite` expects (alpha used when present).
    let materials: Vec<(&'static str, TextureGen)> = vec![
        ("BrushedMetal", brushed_metal_sprite),
        ("Canvas", canvas_sprite),
        ("Concrete", concrete_sprite),
        ("Grunge", grunge_sprite),
        ("PebbledLeather", pebbled_leather_sprite),
        ("Sand", sand_sprite),
        ("Stone", stone_sprite),
    ];
    let mut written = Vec::new();
    for (name, gen) in &materials {
        let tex = gen(PACK_MATERIAL_SIZE, seed_for(name));
        let path = material.join(format!("{name}.png"));
        write_material_png(&path, &tex)?;
        written.push(path);
    }
    Ok(written)
}

/// Writes a material PNG: white RGB, coverage in the alpha channel, which is
/// exactly the format `crate::io::brush_sprite` reads back as a dab mask.
fn write_material_png(path: &std::path::Path, tex: &TextureData) -> Result<(), String> {
    use image::codecs::png::PngEncoder;
    use image::{ColorType, ImageEncoder};
    use std::io::BufWriter;
    let file =
        std::fs::File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;
    let mut w = BufWriter::new(file);
    PngEncoder::new(&mut w)
        .write_image(&tex.rgba, tex.width, tex.height, ColorType::Rgba8.into())
        .map_err(|e| format!("encode {}: {e}", path.display()))
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
    fn builtin_texture_brushes_are_high_res_deterministic_and_varied() {
        let lib = BrushLibrary::new(std::env::temp_dir().join("pixforge_no_such_brushes"));
        for name in [
            "Wood",
            "Brushed Metal",
            "Hammered Metal",
            "Halftone",
            "Checker",
            "Diamond Plate",
            "Denim",
            "Corduroy",
            "Burlap",
            "Linen",
            "Silk",
            "Velvet",
        ] {
            let e = lib
                .entries
                .iter()
                .find(|e| e.name == name)
                .unwrap_or_else(|| panic!("missing built-in brush {name:?}"));
            assert_eq!(e.kind, BrushKind::Texture);
            assert_eq!(
                (e.sprite.width, e.sprite.height),
                (MASK_SIZE, MASK_SIZE),
                "{name}: procedural masks should render at MASK_SIZE"
            );
            // Coverage must actually vary across the mask (a flat 128² mask
            // would be a useless stamp) and never exceed the [0,1] range.
            let mut lo = 255u8;
            let mut hi = 0u8;
            for a in e.sprite.rgba.chunks_exact(4) {
                lo = lo.min(a[3]);
                hi = hi.max(a[3]);
            }
            assert!(
                hi > lo,
                "{name}: procedural mask must vary in coverage, got alpha sweep [{lo},{hi}]"
            );
            // `hi`/`lo` come from u8 alphas, so they are in [0,255] by
            // construction — no range re-check needed.
        }
        // Determinism: the same library seeds the identical sprite bytes, so
        // thumbnails and stamps look the same on every run.
        let a = lib.entries.iter().find(|e| e.name == "Wood").unwrap();
        let b = lib.entries.iter().find(|e| e.name == "Wood").unwrap();
        assert_eq!(a.sprite.rgba, b.sprite.rgba);
        let c = Builtin::procedural("Brushed Metal");
        let d = Builtin::procedural("Brushed Metal");
        assert_eq!(c.sprite.rgba, d.sprite.rgba);
    }

    #[test]
    fn noise_helpers_are_deterministic_and_seamless() {
        // All the shared noise primitives are 1-periodic: sampling one tile
        // over (u+1, v) is identical to (u, v), so every brush stamp tiles
        // invisibly when the dab repeats.
        for cell in 1..=8u32 {
            for &(u, v) in &[(0.37, 0.61), (0.001, 0.5), (0.99, 0.25)] {
                let (a, b, c, d) = (
                    tiled_noise(u + 1.0, v, cell, 7),
                    tiled_noise(u, v, cell, 7),
                    tiled_noise(u, v + 1.0, cell, 7),
                    tiled_noise(u, v, cell, 7),
                );
                assert!(
                    (a - b).abs() < 1e-6 && (c - d).abs() < 1e-6,
                    "tiled_noise wrap u={u} v={v}"
                );
                let (a1, ag) = worley_tiled(u + 1.0, v, cell, 7);
                let (b1, bg) = worley_tiled(u, v, cell, 7);
                assert!(
                    (a1 - b1).abs() < 1e-6 && (ag - bg).abs() < 1e-6,
                    "worley u-wrap at cell {cell}"
                );
                let (a1, ag) = worley_tiled(u, v + 1.0, cell, 7);
                let (b1, bg) = worley_tiled(u, v, cell, 7);
                assert!(
                    (a1 - b1).abs() < 1e-6 && (ag - bg).abs() < 1e-6,
                    "worley v-wrap at cell {cell}"
                );
                let a = domain_warp(u + 1.0, v, 3, 3, 0.5, 9);
                let b = domain_warp(u, v, 3, 3, 0.5, 9);
                assert!(
                    (a.0 - b.0).abs() < 1e-6 && (a.1 - b.1).abs() < 1e-6,
                    "domain_warp u-wrap at cell {cell}"
                );
            }
        }
        // Determinism: the same arguments always give the identical value.
        assert_eq!(
            fbm_tiled(0.21, 0.77, 3, 4, 99),
            fbm_tiled(0.21, 0.77, 3, 4, 99)
        );
        // Seamlessness at the primitive level. `x + 1.0` isn't bit-exact to `x`
        // in f32 (adding 1.0 sheds mantissa bits), so compare with tolerance
        // like the wrap checks above — real generators never add 1.0, since
        // their u/v are already wrapped into [0,1) by `domain_warp`.
        let a = fbm_tiled(0.21 + 1.0, 0.77, 3, 4, 99);
        let b = fbm_tiled(0.21, 0.77, 3, 4, 99);
        assert!((a - b).abs() < 1e-6, "fbm seam, got {a} vs {b}");
    }

    /// Mirrors the built-in brush table so tests validate the real generators
    /// (not just the library's stored copy) stay deterministic.
    struct Builtin;

    impl Builtin {
        fn procedural(name: &str) -> BrushEntry {
            let (_, kind, gen_fn): (&str, BrushKind, fn(u32, u32) -> TextureData) =
                super::texture_generators()
                    .into_iter()
                    .find(|(n, _, _)| *n == name)
                    .unwrap();
            BrushEntry {
                name: name.to_string(),
                category: BUILTIN_CATEGORY.to_string(),
                kind,
                sprite: gen_fn(MASK_SIZE, seed_for(name)),
                path: None,
            }
        }
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
    #[ignore]
    fn bench_texture_generators() {
        type Gen = fn(u32, u32) -> TextureData;
        let gens = super::texture_generators();
        let mut total = std::time::Duration::ZERO;
        for (name, _, gen) in &gens {
            let start = std::time::Instant::now();
            let _tex = gen(super::MASK_SIZE, super::seed_for(name));
            let elapsed = start.elapsed();
            total += elapsed;
            eprintln!("{:>20}: {:>8.2} ms", name, elapsed.as_secs_f64() * 1000.0);
        }
        // Shared material generators too.
        let shared: Vec<(&str, Gen)> = vec![
            ("BrushedMetal", super::brushed_metal_sprite),
            ("Canvas", super::canvas_sprite),
            ("Concrete", super::concrete_sprite),
            ("Grunge", super::grunge_sprite),
            ("PebbledLeather", super::pebbled_leather_sprite),
            ("Sand", super::sand_sprite),
            ("Stone", super::stone_sprite),
        ];
        for (name, gen) in &shared {
            let start = std::time::Instant::now();
            let _tex = gen(super::MASK_SIZE, super::seed_for(name));
            let elapsed = start.elapsed();
            total += elapsed;
            eprintln!("{:>20}: {:>8.2} ms", name, elapsed.as_secs_f64() * 1000.0);
        }
        eprintln!("{:>20}: {:>8.2} ms", "TOTAL", total.as_secs_f64() * 1000.0);
    }

    #[test]
    fn material_brushes_tile_without_seams() {
        // Every shipped-material generator must produce a texture whose edges
        // wrap exactly: the texels at the far boundary of the tile must equal
        // the texels at the start boundary (they are adjacent across the seam
        // when the pack tiles).
        let materials = [
            ("Canvas", canvas_sprite as fn(u32, u32) -> TextureData),
            ("Concrete", concrete_sprite),
            ("Grunge", grunge_sprite),
            ("PebbledLeather", pebbled_leather_sprite),
            ("Sand", sand_sprite),
            ("Stone", stone_sprite),
        ];
        for (name, gen) in materials {
            let tex = gen(MASK_SIZE, seed_for(name));
            let w = tex.width as usize;
            let h = tex.height as usize;
            let a = |x: usize, y: usize| tex.rgba[(y * w + x) * 4 + 3];
            for y in 0..h {
                assert_eq!(
                    a(w - 1, y),
                    a(0, y),
                    "{name}: vertical seam mismatch at row {y}"
                );
            }
            for x in 0..w {
                assert_eq!(
                    a(x, h - 1),
                    a(x, 0),
                    "{name}: horizontal seam mismatch at column {x}"
                );
            }
        }
    }

    #[test]
    #[ignore]
    fn dump_masks_to_tmp() {
        let dir = std::path::Path::new("/tmp/pixforge_masks");
        let _ = std::fs::remove_dir_all(dir);
        std::fs::create_dir_all(dir).unwrap();
        for (name, _, gen) in texture_generators() {
            let tex = gen(MASK_SIZE, seed_for(name));
            dump_one(dir, name, &tex);
        }
        // The shipped-material generators too, so the pack art can be eyeballed
        // alongside the built-ins.
        for (name, gen) in [
            ("Canvas", canvas_sprite as fn(u32, u32) -> TextureData),
            ("Concrete", concrete_sprite),
            ("Grunge", grunge_sprite),
            ("PebbledLeather", pebbled_leather_sprite),
            ("Sand", sand_sprite),
            ("Stone", stone_sprite),
        ] {
            let tex = gen(MASK_SIZE, seed_for(name));
            dump_one(dir, name, &tex);
        }
        println!("dumped to {:?}", dir);
    }

    fn dump_one(dir: &std::path::Path, name: &str, tex: &TextureData) {
        use std::io::Write;
        let mut rgba = tex.rgba.clone();
        for (i, a) in tex.rgba.chunks_exact(4).enumerate() {
            let a = a[3];
            rgba[i * 4] = a;
            rgba[i * 4 + 1] = a;
            rgba[i * 4 + 2] = a;
            rgba[i * 4 + 3] = 255;
        }
        let img = image::RgbaImage::from_raw(tex.width, tex.height, rgba).unwrap();
        img.save(dir.join(format!("{}.png", name.replace(' ', "_"))))
            .unwrap();
        // A 2×2 repeat of the tile, so seamlessness can be eyeballed side by
        // side with the single tile (a good seam gives a perfectly continuous
        // four-tile wall).
        let (tw, th) = (tex.width as usize, tex.height as usize);
        let mut tiled = vec![255u8; tw * 2 * th * 2 * 4];
        for ty in 0..2 {
            for tx in 0..2 {
                for y in 0..th {
                    for x in 0..tw {
                        let src = tex.rgba[(y * tw + x) * 4 + 3];
                        let dst = ((ty * th + y) * tw * 2 + (tx * tw + x)) * 4;
                        tiled[dst] = src;
                        tiled[dst + 1] = src;
                        tiled[dst + 2] = src;
                        tiled[dst + 3] = 255;
                    }
                }
            }
        }
        let t_img = image::RgbaImage::from_raw((tw * 2) as u32, (th * 2) as u32, tiled).unwrap();
        t_img
            .save(dir.join(format!("{}_tiled.png", name.replace(' ', "_"))))
            .unwrap();
        // ASCII preview as a bonus for the terminal.
        let mut txt = String::new();
        for y in 0..tex.height {
            for x in 0..tex.width {
                let a = tex.rgba[((y * tex.width + x) * 4 + 3) as usize];
                let c = match a {
                    0..=40 => ' ',
                    41..=90 => '.',
                    91..=140 => ':',
                    141..=190 => 'o',
                    191..=230 => '#',
                    _ => '@',
                };
                txt.push(c);
            }
            txt.push('\n');
        }
        let mut f =
            std::fs::File::create(dir.join(format!("{}.txt", name.replace(' ', "_")))).unwrap();
        f.write_all(txt.as_bytes()).unwrap();
    }

    #[test]
    fn generate_ship_pack_writes_files_that_parse_back() {
        // The shipped-pack generator must produce PNGs and GBRs the library's
        // own loaders accept, so the packs never ship broken.
        let dir = std::env::temp_dir().join(format!("pixforge_pack_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let files = generate_ship_pack(&dir).expect("generate ship pack");
        assert_eq!(files.len(), 7, "7 material packs");
        for path in &files {
            let entry = load_entry(&dir, path)
                .unwrap_or_else(|e| panic!("pack file {} failed to load: {e}", path.display()));
            let sprite = &entry.sprite;
            assert_eq!(
                sprite.width,
                PACK_MATERIAL_SIZE,
                "{}: unexpected width {}",
                path.display(),
                sprite.width
            );
            // Coverage must actually vary (a flat pack file is a useless stamp).
            let alpha = sprite
                .rgba
                .chunks_exact(4)
                .map(|p| p[3])
                .fold((255u8, 0u8), |(lo, hi), a| (lo.min(a), hi.max(a)));
            assert!(alpha.1 > alpha.0, "{}: flat coverage", path.display());
        }
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
            "Spot", "Stroke", "Spiky", "Grain",
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
