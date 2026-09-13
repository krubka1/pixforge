use glam::{Vec2, Vec3};

use crate::io::{MeshData, TextureData};

/// A 3D hit from a ray cast against a mesh's triangles.
pub struct Hit {
    pub position: Vec3,
    pub uv: (f32, f32),
    /// Index of the triangle (`mesh.indices[triangle*3 .. *3+3]`).
    pub triangle: usize,
}

/// Casts a ray against the mesh and returns the nearest hit, if any.
/// First surface hit along `origin + dir * t` — the nearest triangle, its
/// position and interpolated UV. Used for picking and as the per-texel sight
/// line in `stamp_texels` (the accelerated variant is the occlusion grid built
/// once per stroke).
pub fn mesh_raycast(mesh: &MeshData, origin: Vec3, dir: Vec3) -> Option<Hit> {
    let dir = dir.normalize_or_zero();
    if dir == Vec3::ZERO {
        return None;
    }
    let mut best: Option<(f32, Hit)> = None;
    for (tri, indices) in mesh.indices.chunks_exact(3).enumerate() {
        let (i0, i1, i2) = (
            indices[0] as usize,
            indices[1] as usize,
            indices[2] as usize,
        );
        let (a, b, c) = (mesh.positions[i0], mesh.positions[i1], mesh.positions[i2]);
        let Some(t) = ray_triangle(origin, dir, a, b, c) else {
            continue;
        };
        if best.as_ref().is_some_and(|(bt, _)| t >= *bt) {
            continue;
        }
        let position = origin + dir * t;
        let (w0, w1, w2) = barycentric_3d(position, a, b, c);
        let uv = (
            w0 * mesh.uvs[i0].0 + w1 * mesh.uvs[i1].0 + w2 * mesh.uvs[i2].0,
            w0 * mesh.uvs[i0].1 + w1 * mesh.uvs[i1].1 + w2 * mesh.uvs[i2].1,
        );
        best = Some((
            t,
            Hit {
                position,
                uv,
                triangle: tri,
            },
        ));
    }
    best.map(|(_, h)| h)
}

/// Return the bounding-box center when `positions`/`indices` form a closed,
/// convex solid, else `None`.
///
/// Used to replace the per-texel occlusion raycast with a dot-product test on
/// the fast path: a point of a convex solid is visible from an external eye
/// iff its outward normal points toward the eye, and such a mesh can never
/// hide a visible texel behind another part of itself.  Walls, panels and
/// anything with openings fail the watertight test and keep the slow (grid)
/// path.
fn mesh_is_convex(positions: &[Vec3], indices: &[u32]) -> Option<Vec3> {
    let n_tris = indices.len() / 3;
    if n_tris < 4 {
        return None;
    }
    // Quantize identical-nearby vertices (collapsed pole ring of a UV sphere,
    // the seam column, and floating-point noise like `sin(π) ≈ 8.7e-8`) into
    // one logical vertex so the topology test sees a real closed solid instead
    // of spurious boundary edges.  The step is ~1e-6 of the bounding span:
    // far below any real feature size, so distinct vertices never merge.
    let (mut bb_min, mut bb_max) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
    for p in positions {
        bb_min = bb_min.min(*p);
        bb_max = bb_max.max(*p);
    }
    let span = (bb_max - bb_min).length().max(1e-9);
    let quant = span * 1e-6;
    let q = |c: f32| {
        if c == 0.0 {
            0
        } else {
            (c / quant).round() as i32
        }
    };

    let mut weld_ids = vec![0u32; positions.len()];
    let mut weld: std::collections::HashMap<[i32; 3], u32> = std::collections::HashMap::new();
    for (i, p) in positions.iter().enumerate() {
        let key = [q(p.x), q(p.y), q(p.z)];
        if let Some(&id) = weld.get(&key) {
            weld_ids[i] = id;
        } else {
            let id = weld.len() as u32;
            weld.insert(key, id);
            weld_ids[i] = id;
        }
    }

    // Directed-edge counts over welded vertices, from non-degenerate faces.
    let mut edges: std::collections::HashMap<(u32, u32), u32> = std::collections::HashMap::new();
    for ch in indices.chunks_exact(3) {
        let (i0, i1, i2) = (ch[0] as usize, ch[1] as usize, ch[2] as usize);
        let (a, b, c) = (weld_ids[i0], weld_ids[i1], weld_ids[i2]);
        if a == b || b == c || c == a {
            continue; // degenerate face (zero area)
        }
        for (u, v) in [(a, b), (b, c), (c, a)] {
            *edges.entry((u, v)).or_insert(0) += 1;
        }
    }
    // Watertight: every directed edge is paired with an equal reverse count.
    for (&(a, b), &count) in edges.iter() {
        if edges.get(&(b, a)).copied() != Some(count) {
            return None;
        }
    }

    // Support-plane test: every vertex must lie on the interior side of every
    // face, i.e. the intersection of the face half-spaces is the polyhedron.
    // The reference point is the bounding-box center: the vertex average is
    // biased by the duplicated collar of a UV sphere (collapsed pole rings and
    // a repeated seam column), which skews normal orientation enough to mark a
    // plainly visible front texel as back-facing.
    let centroid = (bb_min + bb_max) * 0.5;
    let eps = (bb_max - bb_min).length() * 1e-5;
    for ch in indices.chunks_exact(3) {
        let (i0, i1, i2) = (ch[0] as usize, ch[1] as usize, ch[2] as usize);
        let (v0, v1, v2) = (positions[i0], positions[i1], positions[i2]);
        let mut n = (v1 - v0).cross(v2 - v0);
        if n.length_squared() < 1e-20 {
            continue;
        }
        if n.dot(v0 - centroid) < 0.0 {
            n = -n; // orient outward from the centroid
        }
        for p in positions {
            if n.dot(*p - v0) > eps {
                return None;
            }
        }
    }
    Some(centroid)
}

/// Uniform-grid index of the mesh triangles, built once per stroke so the
/// per-texel occlusion raycast only tests triangles whose AABB overlaps the
/// sight-line segment instead of scanning the whole mesh every texel.
///
/// Every triangle is inserted into each grid cell its AABB overlaps; a hit
/// closer than `max_dist` along a ray must have its AABB overlap that
/// segment's bounding box, so checking just the cells under the segment AABB
/// cannot miss an occluder.
///
/// Cell layout shared with [`OwnedOcclusionGrid`]; the owned variant also
/// carries cloned geometry so a stroke can cache the index across dabs.
struct GridIndex {
    cell: f32,
    min: Vec3,
    cells: std::collections::HashMap<(i32, i32, i32), Vec<u32>>,
}

fn grid_index(positions: &[Vec3], indices: &[u32]) -> GridIndex {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for p in positions {
        min = min.min(*p);
        max = max.max(*p);
    }
    let size = max - min;
    let cell = size.max_element().max(1e-4) / 8.0;
    let mut cells: std::collections::HashMap<(i32, i32, i32), Vec<u32>> =
        std::collections::HashMap::new();
    for (ti, ch) in indices.chunks_exact(3).enumerate() {
        let a = positions[ch[0] as usize];
        let b = positions[ch[1] as usize];
        let c = positions[ch[2] as usize];
        let tmin = a.min(b).min(c);
        let tmax = a.max(b).max(c);
        let (c0, c1) = (cell_index(tmin, min, cell), cell_index(tmax, min, cell));
        for i in c0.0..=c1.0 {
            for j in c0.1..=c1.1 {
                for k in c0.2..=c1.2 {
                    cells.entry((i, j, k)).or_default().push(ti as u32);
                }
            }
        }
    }
    GridIndex { cell, min, cells }
}

fn cell_index(p: Vec3, min: Vec3, cell: f32) -> (i32, i32, i32) {
    let v = (p - min) / cell;
    (v.x.floor() as i32, v.y.floor() as i32, v.z.floor() as i32)
}

/// Nearest hit distance along the ray that is (strictly) reachable before
/// `max_dist`, ignoring triangles whose surface sits at/behind it.
/// `visited`/`qid` are reused per stamp; distinct queries bump `qid`.
#[allow(clippy::too_many_arguments)]
fn grid_nearest_before(
    data: &GridIndex,
    positions: &[Vec3],
    indices: &[u32],
    origin: Vec3,
    dir: Vec3,
    max_dist: f32,
    visited: &mut [u32],
    qid: &mut u32,
) -> Option<f32> {
    let end = origin + dir * max_dist;
    let smin = origin.min(end);
    let smax = origin.max(end);
    let (lo, hi) = (
        cell_index(smin, data.min, data.cell),
        cell_index(smax, data.min, data.cell),
    );
    let mut best: Option<f32> = None;
    for i in lo.0..=hi.0 {
        for j in lo.1..=hi.1 {
            for k in lo.2..=hi.2 {
                let Some(list) = data.cells.get(&(i, j, k)) else {
                    continue;
                };
                *qid = qid.wrapping_add(1);
                if *qid == 0 {
                    visited.fill(0);
                    *qid = 1;
                }
                for &ti in list {
                    let ti = ti as usize;
                    if visited[ti] == *qid {
                        continue;
                    }
                    visited[ti] = *qid;
                    let ch = &indices[ti * 3..ti * 3 + 3];
                    let (a, b, c) = (
                        positions[ch[0] as usize],
                        positions[ch[1] as usize],
                        positions[ch[2] as usize],
                    );
                    if let Some(t) = ray_triangle(origin, dir, a, b, c) {
                        if best.is_none_or(|b| t < b) {
                            best = Some(t);
                        }
                    }
                }
            }
        }
    }
    best
}

/// Owned copy of the occlusion index (positions + indices cloned) so a stroke
/// caches the grid across dabs without borrowing the live mesh.
struct OwnedOcclusionGrid {
    data: GridIndex,
    positions: Vec<Vec3>,
    indices: Vec<u32>,
}

impl OwnedOcclusionGrid {
    fn new(positions: &[Vec3], indices: &[u32]) -> Self {
        Self {
            data: grid_index(positions, indices),
            positions: positions.to_vec(),
            indices: indices.to_vec(),
        }
    }

    fn nearest_before(
        &self,
        origin: Vec3,
        dir: Vec3,
        max_dist: f32,
        visited: &mut [u32],
        qid: &mut u32,
    ) -> Option<f32> {
        grid_nearest_before(
            &self.data,
            &self.positions,
            &self.indices,
            origin,
            dir,
            max_dist,
            visited,
            qid,
        )
    }
}

/// Per-geometry acceleration built once per stroke and reused by every dab so
/// the O(V·F) convexity scan, the occlusion index and the split-lock
/// components are not recomputed for each dab of a stroke. Geometry does not
/// change while painting, so the cached values stay valid for the stroke's
/// whole lifetime.
pub struct StampAccel {
    convex_centroid: Option<Vec3>,
    bounds_center: Vec3,
    bounds_radius: f32,
    occ: Option<OwnedOcclusionGrid>,
    split: Option<(usize, Vec<usize>)>,
}

impl StampAccel {
    pub fn new(mesh: &MeshData, split_seed: Option<usize>) -> Self {
        let convex_centroid = mesh_is_convex(&mesh.positions, &mesh.indices);
        let (bounds_center, bounds_radius) = match convex_centroid {
            Some(c) => {
                let r = mesh
                    .positions
                    .iter()
                    .map(|p| (*p - c).length())
                    .fold(0.0f32, f32::max);
                (c, r)
            }
            None => {
                let mut min = Vec3::splat(f32::INFINITY);
                let mut max = Vec3::splat(f32::NEG_INFINITY);
                for p in &mesh.positions {
                    min = min.min(*p);
                    max = max.max(*p);
                }
                ((min + max) * 0.5, (max - min).length() * 0.5)
            }
        };
        let occ = if convex_centroid.is_none() {
            Some(OwnedOcclusionGrid::new(&mesh.positions, &mesh.indices))
        } else {
            None
        };
        let split = split_seed.and_then(|seed| {
            if seed >= mesh.indices.len() / 3 {
                return None;
            }
            let comps = triangle_components_edge(&mesh.indices);
            comps.get(seed).copied().map(|c| (c, comps))
        });
        Self {
            convex_centroid,
            bounds_center,
            bounds_radius,
            occ,
            split,
        }
    }
}

/// Maps a UV (0,0 = top-left of the atlas, as in glTF) to a texel index.
pub fn texel_from_uv(uv: (f32, f32), width: u32, height: u32) -> (u32, u32) {
    let x = (uv.0.clamp(0.0, 1.0) * width as f32)
        .floor()
        .clamp(0.0, (width.saturating_sub(1)) as f32) as u32;
    let y = (uv.1.clamp(0.0, 1.0) * height as f32)
        .floor()
        .clamp(0.0, (height.saturating_sub(1)) as f32) as u32;
    (x, y)
}

/// Center of a texel in UV space.
pub fn uv_from_texel(x: u32, y: u32, width: u32, height: u32) -> (f32, f32) {
    (
        (x as f32 + 0.5) / width.max(1) as f32,
        (y as f32 + 0.5) / height.max(1) as f32,
    )
}

/// Places brush dab positions along a stroke segment `from -> to` (screen
/// pixels), stepping every `spacing_px`. The endpoint `to` is always included;
/// `spacing_px <= 0` returns just `[to]` (one dab per frame, i.e. freehand at
/// the current pointer position). The live stroke uses an accumulator instead;
/// this helper remains the tested contract for fill/tool paths.
#[allow(dead_code)] // exercised by tests; the interactive stroke accumulates distances
pub fn stamp_positions(from: Vec2, to: Vec2, spacing_px: f32) -> Vec<Vec2> {
    if spacing_px <= 0.0 {
        return vec![to];
    }
    let delta = to - from;
    let dist = delta.length();
    let steps = (dist / spacing_px).floor() as u32;
    let mut out = Vec::with_capacity(steps as usize + 2);
    let step = if steps >= 1 {
        Some(delta.normalize() * spacing_px)
    } else {
        None
    };
    match step {
        Some(step) => {
            for i in 0..=steps {
                out.push(from + step * i as f32);
            }
        }
        None => out.push(from),
    }
    if out.last() != Some(&to) {
        out.push(to);
    }
    out
}

/// Feeds one cursor sample of an in-progress stroke into the accumulator and
/// returns the dabs now owed. Dabs are spaced `spacing` screen px apart along
/// the motion; `acc` carries the leftover travel since the last dab, so a slow
/// drag never clumps dabs together and a fast flick still gets evenly spaced
/// dabs along its path. Returns `(dabs, new_acc, new_last_dab)`.
pub fn spaced_freehand_dabs(
    cursor_prev: egui::Pos2,
    cursor_now: egui::Pos2,
    last_dab: egui::Pos2,
    acc: f32,
    spacing: f32,
) -> (Vec<egui::Pos2>, f32, egui::Pos2) {
    if spacing <= 0.0 || cursor_prev == cursor_now {
        return (Vec::new(), acc, last_dab);
    }
    let delta = cursor_now - cursor_prev;
    let dir = delta / delta.length();
    let mut acc = acc + delta.length();
    let mut last_dab = last_dab;
    let mut dabs = Vec::new();
    while acc >= spacing {
        acc -= spacing;
        last_dab += dir * spacing;
        dabs.push(last_dab);
    }
    (dabs, acc, last_dab)
}

/// World units per texel, averaged over the hit triangle's edges.
#[allow(dead_code)] // texel-size brush API, currently exercised by tests
pub fn hit_texel_scale(mesh: &MeshData, hit: &Hit, width: u32, height: u32) -> f32 {
    let (i0, i1, i2) = tri_indices(mesh, hit.triangle);
    triangle_texel_scale(&mesh.positions, &mesh.uvs, i0, i1, i2, width, height)
}

/// Converts a brush size in texels to a world-space radius at the hit point.
#[allow(dead_code)] // texel-size brush API, currently exercised by tests
pub fn brush_radius_world(mesh: &MeshData, hit: &Hit, width: u32, height: u32, texels: f32) -> f32 {
    texels * hit_texel_scale(mesh, hit, width, height)
}

/// How a stroke interacts with the existing texels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StampMode {
    Paint,
    Erase,
}

/// The 2D footprint of a single brush dab, drawn in the brush-local plane
/// (perpendicular to the brush ray through the stamp center).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrushShape {
    /// Soft round dab (radius = `radius_world`).
    Round,
    /// Axis-aligned square (diameter 2 × `radius_world`).
    Square,
    /// Axis-aligned diamond inscribed in the round dab.
    Diamond,
    /// Arbitrary image stamp from `BrushStyle::sprite`.
    Texture,
}

impl BrushShape {
    pub const ALL: [BrushShape; 4] = [
        BrushShape::Round,
        BrushShape::Square,
        BrushShape::Diamond,
        BrushShape::Texture,
    ];

    pub fn label(self) -> &'static str {
        match self {
            BrushShape::Round => "Round",
            BrushShape::Square => "Square",
            BrushShape::Diamond => "Diamond",
            BrushShape::Texture => "Texture",
        }
    }
}

/// A complete brush definition: footprint shape plus an optional image stamp.
#[derive(Clone, Debug)]
pub struct BrushStyle {
    pub shape: BrushShape,
    /// Image stamp for [`BrushShape::Texture`]. Its alpha is the coverage when
    /// it has transparency; a fully opaque image uses inverted luminance
    /// (dark = strong paint) instead. Build one with `io::brush_sprite`.
    pub sprite: Option<crate::io::TextureData>,
    /// Stamp rotation in radians (texture shapes only).
    pub rotation: f32,
    pub flip_x: bool,
    pub flip_y: bool,
}

impl Default for BrushStyle {
    fn default() -> Self {
        Self {
            shape: BrushShape::Round,
            sprite: None,
            rotation: 0.0,
            flip_x: false,
            flip_y: false,
        }
    }
}

/// Applies the given brush style instead of the default round dab:
/// `apply_stamp` with a `Texture` shape driven by `style`.
#[allow(clippy::too_many_arguments)]
pub fn apply_stamp_with(
    mesh: &mut MeshData,
    center: Vec3,
    radius_world: f32,
    eye: Vec3,
    view_dir: Vec3,
    color: [u8; 4],
    opacity: f32,
    hardness: f32,
    mode: StampMode,
    style: &BrushStyle,
    accel: Option<&StampAccel>,
    accumulate: bool,
    stroke_alpha: Option<&mut [u8]>,
) {
    stamp_texels(
        mesh,
        center,
        radius_world,
        eye,
        view_dir,
        color,
        opacity,
        hardness,
        mode,
        None,
        Some(style),
        accel,
        accumulate,
        stroke_alpha,
    );
}

/// Applies a 3D-projected disc brush centered at `center` to the albedo
/// texture: every texel whose 3D surface position is within `radius_world` is
/// blended with `color` (soft falloff shaped by `hardness`, strength by
/// `opacity`).
///
/// `eye` and `view_dir` describe the brush ray (origin + normalized direction).
/// Only texels that actually face the brush — `dot(face_normal, -view_dir) > 0`
/// — AND are the nearest surface along their own sight line (not hidden behind
/// a nearer part of the mesh) are touched, so a stroke never paints or erases
/// "through" the object: the far side of a wall, the same-facing far wall
/// visible across an opening, and the outer back of a solid all stay untouched.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)] // public convenience wrapper; exercised by tests
pub fn apply_stamp(
    mesh: &mut MeshData,
    center: Vec3,
    radius_world: f32,
    eye: Vec3,
    view_dir: Vec3,
    color: [u8; 4],
    opacity: f32,
    hardness: f32,
    mode: StampMode,
) {
    stamp_texels(
        mesh,
        center,
        radius_world,
        eye,
        view_dir,
        color,
        opacity,
        hardness,
        mode,
        None,
        None,
        None,
        true,
        None,
    );
}

/// Like [`apply_stamp`], but the footprint is a rectangle (in world units)
/// aligned to the brush ray: texels within `half` of the surface plane along
/// two screen-aligned axes are stamped. `half_w`/`half_h` are half-extents.
#[allow(clippy::too_many_arguments)]
pub fn apply_stamp_rect(
    mesh: &mut MeshData,
    center: Vec3,
    half_w: f32,
    half_h: f32,
    eye: Vec3,
    view_dir: Vec3,
    color: [u8; 4],
    opacity: f32,
    hardness: f32,
    mode: StampMode,
    accel: Option<&StampAccel>,
    accumulate: bool,
    stroke_alpha: Option<&mut [u8]>,
) {
    stamp_texels(
        mesh,
        center,
        half_w.max(half_h),
        eye,
        view_dir,
        color,
        opacity,
        hardness,
        mode,
        Some((half_w, half_h)),
        None,
        accel,
        accumulate,
        stroke_alpha,
    );
}

/// Samples `sprite` at canvas texel `(x, y)` as a *world-space tiled* pattern:
/// the coordinate is the texel's global position mod the sprite size (with the
/// brush's flip/rotation applied to the lattice), NOT an offset from the dab
/// center. The texel is anchored to the canvas/atlas grid, so dragging a
/// texture brush reveals a fixed pattern instead of sliding a static stamp
/// frame along the stroke path.
fn pattern_alpha_at(
    sprite: &TextureData,
    x: i32,
    y: i32,
    rotation: f32,
    flip_x: bool,
    flip_y: bool,
) -> f32 {
    let (sw, sh) = (sprite.width.max(1) as usize, sprite.height.max(1) as usize);
    let (mut gx, mut gy) = (x as f32, y as f32);
    if flip_x {
        gx = -gx;
    }
    if flip_y {
        gy = -gy;
    }
    if rotation != 0.0 {
        let (sr, cr) = rotation.sin_cos();
        let (rx, ry) = (gx * cr - gy * sr, gx * sr + gy * cr);
        gx = rx;
        gy = ry;
    }
    let sx = ((gx.rem_euclid(sw as f32) + 0.5) as usize) % sw;
    let sy = ((gy.rem_euclid(sh as f32) + 0.5) as usize) % sh;
    sprite.rgba[(sy * sw + sx) * 4 + 3] as f32 / 255.0
}

/// Shared core of the stamp brushes (`rect = None` → disc of `radius_world`;
/// `Some((half_w, half_h))` → axis-aligned rectangle). `style` overrides the
/// disc footprint with the given brush shape; `rect` and `style` are mutually
/// exclusive (the latter wins when both are `Some` is not possible).
/// A texture brush stamps a *round* dab whose falloff only controls where the
/// stroke lands; the sprite is sampled as a world-space tiled pattern at each
/// texel's global position (see [`pattern_alpha_at`]) and multiplies the dab.
/// If `accumulate` is false, `stroke_alpha` (when Some) acts as the stroke's
/// `stroke_buffer`: it tracks the MAX target alpha per texel
/// (`min(opacity, dab × pattern)`), so later dabs cap rather than stack, and
/// each texel is blended toward that value through an exact source-over step —
/// live preview is identical to compositing the final buffer once.
#[allow(clippy::too_many_arguments)]
fn stamp_texels(
    mesh: &mut MeshData,
    center: Vec3,
    radius_world: f32,
    eye: Vec3,
    view_dir: Vec3,
    color: [u8; 4],
    opacity: f32,
    hardness: f32,
    mode: StampMode,
    rect: Option<(f32, f32)>,
    style: Option<&BrushStyle>,
    accel: Option<&StampAccel>,
    accumulate: bool,
    mut stroke_alpha: Option<&mut [u8]>,
) {
    let Some(texture) = mesh.active_layer_texture() else {
        return;
    };
    let ok = texture.width > 0 && texture.height > 0 && opacity > 0.0 && radius_world > 0.0;
    if !ok {
        return;
    }
    // Take the active layer's texture out so the per-texel occlusion raycast
    // can borrow the mesh immutably at the same time; it is put back before
    // returning.
    let layer_idx = mesh.active_layer;
    let (tw, th) = {
        let tex = &mesh.layers[layer_idx].texture;
        (tex.width as i32, tex.height as i32)
    };
    let mut tex = std::mem::replace(
        &mut mesh.layers[layer_idx].texture,
        crate::io::blank_atlas(tw as u32, th as u32, [0, 0, 0, 0]),
    );
    let (w, h) = (tw, th);
    let mut dirty = mesh.dirty.unwrap_or((tw as u32, th as u32, 0, 0));
    let radius = radius_world.max(1e-4);
    #[derive(Clone, Copy)]
    enum Footprint {
        Rect(f32, f32),
        Round(f32),
        Square(f32),
        Diamond(f32),
    }
    let sprite = style.and_then(|s| s.sprite.as_ref());
    let footprint = if let Some((hw, hh)) = rect {
        Footprint::Rect(hw, hh)
    } else if let Some(s) = style {
        match s.shape {
            BrushShape::Round => Footprint::Round(radius),
            BrushShape::Square => Footprint::Square(radius),
            BrushShape::Diamond => Footprint::Diamond(radius),
            BrushShape::Texture => Footprint::Round(radius), // round dab; pattern multiplies below
        }
    } else {
        Footprint::Round(radius)
    };
    let sprite_footprint =
        sprite.is_some() && style.is_some_and(|s| matches!(s.shape, BrushShape::Texture));
    let footprint_radius = match footprint {
        Footprint::Rect(hw, hh) => (hw * hw + hh * hh).sqrt(),
        Footprint::Square(r) => r * std::f32::consts::SQRT_2,
        _ => radius,
    };

    // A zero view direction falls back to "touch everything" (no gates).
    let facing_gate = view_dir.length_squared() > 1e-12;
    let occlusion_gate = eye.length_squared() > 1e-12;
    let away = -view_dir;
    // Brush-local axes: the plane follows the surface tangent around the stamp
    // (with the camera's screen-right projected onto it so the pattern stays
    // upright relative to the view); both axes are zero for the analytic
    // "touch everything" fallback. `brush_axes` mirrors the camera-plane choice
    // for the cursor preview, so the on-screen cursor matches the painted
    // footprint.
    let (axis_u, axis_v) = brush_axes(
        &mesh.positions,
        &mesh.indices,
        center,
        footprint_radius,
        view_dir,
    );

    let positions = &mesh.positions;
    let uvs = &mesh.uvs;
    // Reuse the stroke's acceleration when one was built; otherwise build a
    // throwaway copy (tests / one-shot stamps). Geometry never changes while
    // painting, so the cached convexity / bounds / occlusion index / split
    // components stay valid for the whole stroke.
    let accel_local;
    let accel: &StampAccel = match accel {
        Some(a) => a,
        None => {
            accel_local = StampAccel::new(mesh, None);
            &accel_local
        }
    };
    let convex_centroid = accel.convex_centroid;
    // The dot-product shortcut presupposes the eye sits outside the solid, so
    // only take it when the eye clears the bounding sphere; an eye at or
    // inside the volume falls back to the grid, which handles it regardless.
    let eye_outside = match convex_centroid {
        Some(_) => (eye - accel.bounds_center).length() > accel.bounds_radius + 1e-6,
        None => false,
    };
    let use_fast_occ = occlusion_gate && convex_centroid.is_some() && eye_outside;
    // Occlusion needs the nearest surface along each texel's sight line; the
    // grid is built once per stroke by the accel and reused across dabs.
    let occ_grid = if occlusion_gate && !use_fast_occ {
        accel.occ.as_ref()
    } else {
        None
    };
    let mut occ_visited: Vec<u32> = vec![0; (mesh.indices.len() / 3).max(1)];
    let mut occ_qid = 0u32;

    // Split lock: the accel carries the seed component and the per-triangle
    // edge-components already computed at stroke start. Every texel not on a
    // triangle of the seed part is skipped, so a brush never bleeds onto a
    // separate model part that happens to fall inside its radius.
    let split_components = accel.split.as_ref();

    for (tri, indices) in mesh.indices.chunks_exact(3).enumerate() {
        if let Some((seed_comp, comps)) = &split_components {
            if comps[tri] != *seed_comp {
                continue;
            }
        }
        let (i0, i1, i2) = (
            indices[0] as usize,
            indices[1] as usize,
            indices[2] as usize,
        );
        // Skip triangles the brush cannot "see" the front of. A back-facing
        // triangle is the hidden side of a wall (or the far wall across an
        // opening), so painting/erasing it would go through walls.
        //
        // Degenerate triangles (zero area, length_squared < 1e-12) are kept:
        // at each pole all ring-0 vertices collapse to the same position, so
        // one of the two row-0 quads per column is degenerate in 3D.  Its UV
        // area is still valid — together with the non-degenerate half it
        // covers the full pole texel strip — so skipping it would leave half
        // the pole unpaintable.
        if facing_gate {
            let n = (positions[i1] - positions[i0]).cross(positions[i2] - positions[i0]);
            if n.length_squared() >= 1e-12 && n.dot(away) <= 0.0 {
                continue;
            }
        }
        let (a, b, c) = (positions[i0], positions[i1], positions[i2]);

        // Outward-facing normal for the fast convex occlusion test, oriented
        // away from the centroid.  Degenerate (collapsed pole) triangles yield
        // none and are left paintable, matching the facing-gate behavior.
        let occ_normal = if use_fast_occ {
            match convex_centroid {
                Some(centroid) => {
                    let mut n = (b - a).cross(c - a);
                    if n.length_squared() > 1e-12 {
                        if n.dot(a - centroid) < 0.0 {
                            n = -n;
                        }
                        Some(n)
                    } else {
                        None
                    }
                }
                None => None, // unreachable when use_fast_occ
            }
        } else {
            None
        };

        // Conservative culling of whole triangles outside the footprint.
        if let Footprint::Rect(hw, hh) = footprint {
            let (mut min_u, mut max_u) = (f32::INFINITY, f32::NEG_INFINITY);
            let (mut min_v, mut max_v) = (f32::INFINITY, f32::NEG_INFINITY);
            for v in [a, b, c] {
                let rel = v - center;
                let (tu, tv) = (rel.dot(axis_u), rel.dot(axis_v));
                min_u = min_u.min(tu);
                max_u = max_u.max(tu);
                min_v = min_v.min(tv);
                max_v = max_v.max(tv);
            }
            if max_u < -hw || min_u > hw || max_v < -hh || min_v > hh {
                continue;
            }
        } else if dist_point_to_triangle(center, a, b, c) > footprint_radius {
            continue;
        }

        // Per-triangle texel scale -> UV bounding box.
        //
        // The old code expanded the box by `margin = extent / scale` to catch
        // texels whose 3D position fell inside the brush despite their UV
        // center being outside the triangle.  That is impossible: the 3-D
        // position is only computed *after* `uv_barycentric` succeeds, and
        // barycentric rejects texels outside the triangle.  The margin just
        // inflated the loop from ~700 to ~400K iterations per triangle with
        // a 300 px brush, all immediately rejected — a ~500× waste.
        let (u0, u1, u2) = (uvs[i0].0, uvs[i1].0, uvs[i2].0);
        let (v0, v1, v2) = (uvs[i0].1, uvs[i1].1, uvs[i2].1);
        let min_u = u0.min(u1).min(u2) * w as f32;
        let max_u = u0.max(u1).max(u2) * w as f32;
        let min_v = v0.min(v1).min(v2) * h as f32;
        let max_v = v0.max(v1).max(v2) * h as f32;

        let x0 = (min_u.floor().max(0.0) as i32).min(w - 1);
        let x1 = (max_u.ceil().min((w - 1) as f32) as i32).max(x0);
        let y0 = (min_v.floor().max(0.0) as i32).min(h - 1);
        let y1 = (max_v.ceil().min((h - 1) as f32) as i32).max(y0);

        let t0 = Vec2::new(uvs[i0].0, uvs[i0].1);
        let t1 = Vec2::new(uvs[i1].0, uvs[i1].1);
        let t2 = Vec2::new(uvs[i2].0, uvs[i2].1);

        for y in y0..=y1 {
            for x in x0..=x1 {
                let uv = uv_from_texel(x as u32, y as u32, tex.width, tex.height);
                let p = Vec2::new(uv.0, uv.1);
                let Some((bb0, bb1)) = uv_barycentric(p, t0, t1, t2) else {
                    continue;
                };
                let pos_3d = a + (b - a) * bb0 + (c - a) * bb1;

                // Footprint inclusion + edge falloff (disc / rect / square / diamond /
                // sprite). `tu`/`tv` are the texel's position in the brush-local
                // plane (the plane through `center` perpendicular to the ray).
                let rel = pos_3d - center;
                let (tu, tv) = if facing_gate {
                    (rel.dot(axis_u), rel.dot(axis_v))
                } else {
                    (0.0, 0.0)
                };
                let (t, inside) = match footprint {
                    Footprint::Rect(hw, hh) => {
                        let (atu, atv) = (tu.abs(), tv.abs());
                        if atu > hw || atv > hh {
                            (0.0, false)
                        } else {
                            ((atu / hw.max(1e-6)).max(atv / hh.max(1e-6)), true)
                        }
                    }
                    Footprint::Round(r) => {
                        let dd = rel.length_squared();
                        let r2 = r * r;
                        if dd > r2 {
                            (0.0, false)
                        } else {
                            (dd.sqrt() / r, true)
                        }
                    }
                    Footprint::Square(r) => {
                        if tu.abs() > r || tv.abs() > r {
                            (0.0, false)
                        } else {
                            (tu.abs().max(tv.abs()) / r, true)
                        }
                    }
                    Footprint::Diamond(r) => {
                        let m = tu.abs() + tv.abs();
                        if m > r {
                            (0.0, false)
                        } else {
                            (m / r, true)
                        }
                    }
                };
                if !inside {
                    continue;
                }

                // Occlusion: the texel must be the NEAREST surface along its own
                // sight line from the brush eye. A same-facing surface sitting
                // behind a wall (e.g. the far interior wall beyond a hole) is
                // occluded and must not be painted "through" the nearer one.
                if use_fast_occ {
                    // Convex mesh seen from outside: a texel is hidden exactly
                    // when its outward normal points away from the eye. This is
                    // the closed-form equivalent of the grid raycast below and
                    // costs a single dot product per texel instead.
                    if occ_normal.is_some_and(|n| n.dot(eye - pos_3d) <= 0.0) {
                        continue;
                    }
                } else if occ_grid.is_some() {
                    let to_point = pos_3d - eye;
                    let dist = to_point.length();
                    if dist > 1e-9 {
                        let occ_eps = (radius_world * 0.001).max(1e-4);
                        let blocked = occ_grid
                            .as_ref()
                            .and_then(|g| {
                                g.nearest_before(
                                    eye,
                                    to_point / dist,
                                    dist,
                                    &mut occ_visited,
                                    &mut occ_qid,
                                )
                            })
                            .is_some_and(|t| t < dist - occ_eps);
                        if blocked {
                            continue;
                        }
                    }
                }

                // dab_alpha: classic distance falloff — full strength out to
                // `hardness` of the radius, then a linear fade to the edge
                // (1.0 = hard rim, 0.0 = gradient from the center outward).
                // The eraser feathers with a fully-transparent core instead.
                let dab = if mode == StampMode::Erase {
                    let core = 0.55;
                    let t = t.min(1.0);
                    if t <= core {
                        1.0
                    } else {
                        (1.0 - t) / (1.0 - core)
                    }
                } else {
                    let core_t = hardness.clamp(0.0, 0.999);
                    if t <= core_t {
                        1.0
                    } else {
                        ((1.0 - t) / (1.0 - core_t)).clamp(0.0, 1.0)
                    }
                };
                // pattern_alpha: the sprite at the texel's GLOBAL atlas
                // position. Sample at (x, y), not relative to the dab, so a
                // dragged texture stroke reveals a surface-anchored pattern.
                let pattern = if sprite_footprint {
                    let s = style.expect("sprite footprint ⇒ style is present");
                    let spr = sprite.expect("sprite footprint ⇒ sprite present");
                    pattern_alpha_at(spr, x, y, s.rotation, s.flip_x, s.flip_y)
                } else {
                    1.0
                };
                let raw = dab * pattern;
                if raw <= 0.0 {
                    continue;
                }
                let idx = (y as u32 * tex.width + x as u32) as usize * 4;
                let texel_idx = (y as u32 * tex.width + x as u32) as usize;
                // Non-accumulative stroke blend: the stroke buffer holds the
                // MAX of `min(opacity, dab × pattern)` per texel, exactly like
                // compositing the final buffer once. The exact source-over step
                // (new_alpha − current)/(1 − current) reaches that value when
                // blended incrementally, so the live preview already matches.
                let mut effective_opacity = opacity * raw;
                let mut new_stroke_alpha = 0u8;
                if !accumulate {
                    if let Some(sa) = stroke_alpha.as_mut() {
                        let current_stroke_alpha = sa[texel_idx] as f32 / 255.0;
                        let new_alpha = raw.min(opacity).max(current_stroke_alpha);
                        if new_alpha <= current_stroke_alpha {
                            continue;
                        }
                        effective_opacity =
                            (new_alpha - current_stroke_alpha) / (1.0 - current_stroke_alpha);
                        new_stroke_alpha = (new_alpha * 255.0).round() as u8;
                    }
                }
                let mut px = [
                    tex.rgba[idx],
                    tex.rgba[idx + 1],
                    tex.rgba[idx + 2],
                    tex.rgba[idx + 3],
                ];
                match mode {
                    StampMode::Paint => blend_pixel(&mut px, color, effective_opacity),
                    StampMode::Erase => erase_pixel(&mut px, effective_opacity),
                }
                tex.rgba[idx..idx + 4].copy_from_slice(&px);
                if !accumulate {
                    if let Some(sa) = stroke_alpha.as_mut() {
                        sa[texel_idx] = new_stroke_alpha;
                    }
                }
                dirty.0 = dirty.0.min(x as u32);
                dirty.1 = dirty.1.min(y as u32);
                dirty.2 = dirty.2.max(x as u32);
                dirty.3 = dirty.3.max(y as u32);
            }
        }
    }
    mesh.layers[layer_idx].texture = tex;
    if dirty.0 <= dirty.2 {
        mesh.dirty = Some(dirty);
    }
}

/// Applies a brush stamp directly in 2D UV space onto `texture` (the 2D
/// texture preview) without any 3D mesh involvement. `center_uv` is the brush
/// center in [0,1] UV coordinates; `radius_px` is the footprint radius in
/// texels (the preview scales screen pixels 1:1 with texels when zoomed to
/// fit). `shape` drives the footprint (rect tools pass [`BrushShape::Square`]);
/// `style` supplies the sprite, rotation and flips for the texture shape: the
/// sprite is sampled as a world-space tiled pattern at each texel's global
/// (x, y) position. The hardness profile / opacity / mode semantics mirror the
/// 3D stamps. `dirty` is expanded to the touched texel rect so the caller can do
/// a region upload instead of a full one.
/// If `accumulate` is false, `stroke_alpha` (when Some) tracks the maximum
/// alpha this stroke has applied per texel, capping further dabs within the
/// same stroke so opacity doesn't stack beyond `opacity`.
#[allow(clippy::too_many_arguments)]
pub fn stamp_2d(
    texture: &mut TextureData,
    center_uv: (f32, f32),
    radius_px: f32,
    shape: BrushShape,
    color: [u8; 4],
    opacity: f32,
    hardness: f32,
    mode: StampMode,
    style: &BrushStyle,
    dirty: &mut Option<(u32, u32, u32, u32)>,
    accumulate: bool,
    mut stroke_alpha: Option<&mut [u8]>,
) {
    let (w, h) = (texture.width as i32, texture.height as i32);
    if w <= 0 || h <= 0 || opacity <= 0.0 || radius_px <= 0.0 {
        return;
    }
    let (cx, cy) = (center_uv.0 * w as f32, center_uv.1 * h as f32);
    let r = radius_px;
    let r_inv = 1.0 / r;

    let x0 = (cx - r).floor() as i32;
    let x1 = (cx + r).ceil() as i32;
    let y0 = (cy - r).floor() as i32;
    let y1 = (cy + r).ceil() as i32;

    let sprite = style.sprite.as_ref();
    for y in y0..=y1 {
        if y < 0 || y >= h {
            continue;
        }
        for x in x0..=x1 {
            if x < 0 || x >= w {
                continue;
            }
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            // `t` is the normalized distance to the dab center (the footprint
            // falloff input); the dab profile is resolved after the match.
            let (t, inside) = match shape {
                BrushShape::Round => {
                    let dd = dx * dx + dy * dy;
                    let r2 = r * r;
                    if dd > r2 {
                        (0.0, false)
                    } else {
                        (dd.sqrt() * r_inv, true)
                    }
                }
                BrushShape::Square => {
                    let (adx, ady) = (dx.abs(), dy.abs());
                    if adx > r || ady > r {
                        (0.0, false)
                    } else {
                        (adx.max(ady) * r_inv, true)
                    }
                }
                BrushShape::Diamond => {
                    let m = dx.abs() + dy.abs();
                    if m > r {
                        (0.0, false)
                    } else {
                        (m * r_inv, true)
                    }
                }
                BrushShape::Texture => {
                    // The sprite is sampled as a world-space tiled pattern at
                    // each texel's global (x, y) below; the footprint itself is
                    // a round dab (like the 3D stamp).
                    let dd = dx * dx + dy * dy;
                    let r2 = r * r;
                    if dd > r2 {
                        (0.0, false)
                    } else {
                        (dd.sqrt() * r_inv, true)
                    }
                }
            };
            if !inside || t <= 0.0 {
                continue;
            }

            // dab_alpha: distance falloff — hardness keeps full strength out to
            // `hardness` of the radius then fades; the eraser feathers with a
            // transparent core. Mirrors the 3D stamp exactly.
            let dab = if mode == StampMode::Erase {
                let core = 0.55;
                let t = t.min(1.0);
                if t <= core {
                    1.0
                } else {
                    (1.0 - t) / (1.0 - core)
                }
            } else {
                let core_t = hardness.clamp(0.0, 0.999);
                if t <= core_t {
                    1.0
                } else {
                    ((1.0 - t) / (1.0 - core_t)).clamp(0.0, 1.0)
                }
            };
            // pattern_alpha: the sprite tiled at the texel's GLOBAL canvas
            // position (x mod w, y mod h), anchored to the canvas so a dragged
            // texture stroke reveals a fixed pattern instead of a moving frame.
            let pattern = if matches!(shape, BrushShape::Texture) {
                if let Some(spr) = sprite {
                    pattern_alpha_at(spr, x, y, style.rotation, style.flip_x, style.flip_y)
                } else {
                    1.0 // texture shape with no sprite → plain round brush
                }
            } else {
                1.0
            };
            let raw = dab * pattern;
            if raw <= 0.0 {
                continue;
            }
            let idx = (y as u32 * texture.width + x as u32) as usize * 4;
            let texel_idx = (y as u32 * texture.width + x as u32) as usize;
            // Non-accumulative stroke blend: `stroke_buffer` keeps the MAX of
            // `min(opacity, dab × pattern)` per texel; the exact source-over
            // step reaches compositing the final buffer once, so live preview
            // already equals the finished stroke.
            let mut effective_opacity = opacity * raw;
            let mut new_stroke_alpha = 0u8;
            if !accumulate {
                if let Some(sa) = stroke_alpha.as_mut() {
                    let current_stroke_alpha = sa[texel_idx] as f32 / 255.0;
                    let new_alpha = raw.min(opacity).max(current_stroke_alpha);
                    if new_alpha <= current_stroke_alpha {
                        continue;
                    }
                    effective_opacity =
                        (new_alpha - current_stroke_alpha) / (1.0 - current_stroke_alpha);
                    new_stroke_alpha = (new_alpha * 255.0).round() as u8;
                }
            }
            let mut px = [
                texture.rgba[idx],
                texture.rgba[idx + 1],
                texture.rgba[idx + 2],
                texture.rgba[idx + 3],
            ];
            match mode {
                StampMode::Paint => blend_pixel(&mut px, color, effective_opacity),
                StampMode::Erase => erase_pixel(&mut px, effective_opacity),
            }
            texture.rgba[idx..idx + 4].copy_from_slice(&px);
            if !accumulate {
                if let Some(sa) = stroke_alpha.as_mut() {
                    sa[texel_idx] = new_stroke_alpha;
                }
            }
            let (ux, uy) = (x as u32, y as u32);
            match dirty {
                Some(d) => {
                    d.0 = d.0.min(ux);
                    d.1 = d.1.min(uy);
                    d.2 = d.2.max(ux);
                    d.3 = d.3.max(uy);
                }
                None => *dirty = Some((ux, uy, ux, uy)),
            }
        }
    }
}

/// Flood-fills a connected region of similar texels starting at `seed_uv` in
/// the 2D texture preview. Every pixel within the fill tolerance of the seed
/// color is blended toward `color` (source-over, `opacity` strength). The seed
/// is matched on RGB + alpha, so an erased (fully transparent) area is filled
/// like any colored one. Uses a scanline flood fill (O(n) stack, no recursion).
/// `dirty` is expanded to the touched texel rect for the region upload.
pub fn stamp_fill_2d(
    texture: &mut TextureData,
    seed_uv: (f32, f32),
    color: [u8; 4],
    opacity: f32,
    dirty: &mut Option<(u32, u32, u32, u32)>,
) {
    let (w, h) = (texture.width as i32, texture.height as i32);
    if w <= 0 || h <= 0 || opacity <= 0.0 {
        return;
    }
    let sx = (seed_uv.0 * w as f32).round().clamp(0.0, (w - 1) as f32) as i32;
    let sy = (seed_uv.1 * h as f32).round().clamp(0.0, (h - 1) as f32) as i32;
    let n = (w * h) as usize;

    // Match threshold: a single call reproduces bucket fills; painted color
    // steps are kept apart so it doesn't bleed through soft gradients.
    const TOL: i32 = 48;
    let seed_idx = ((sy * w + sx) as usize) * 4;
    let seed = [
        texture.rgba[seed_idx],
        texture.rgba[seed_idx + 1],
        texture.rgba[seed_idx + 2],
        texture.rgba[seed_idx + 3],
    ];

    let mut visited = vec![false; n];
    let mut stack: Vec<(i32, i32)> = vec![(sx, sy)];
    let mut min_x = w as u32;
    let mut min_y = h as u32;
    let mut max_x = 0u32;
    let mut max_y = 0u32;

    let in_region = |tex: &[u8], x: i32, y: i32| -> bool {
        if x < 0 || y < 0 || x >= w || y >= h {
            return false;
        }
        let idx = ((y * w + x) as usize) * 4;
        let d_r = tex[idx] as i32 - seed[0] as i32;
        let d_g = tex[idx + 1] as i32 - seed[1] as i32;
        let d_b = tex[idx + 2] as i32 - seed[2] as i32;
        let d_a = tex[idx + 3] as i32 - seed[3] as i32;
        d_r.abs() + d_g.abs() + d_b.abs() + d_a.abs() <= TOL
    };

    while let Some((x, y)) = stack.pop() {
        let idx = (y * w + x) as usize;
        if visited[idx] {
            continue;
        }
        visited[idx] = true;
        if !in_region(&texture.rgba, x, y) {
            continue;
        }
        min_x = min_x.min(x as u32);
        min_y = min_y.min(y as u32);
        max_x = max_x.max(x as u32);
        max_y = max_y.max(y as u32);
        for (nx, ny) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] {
            if nx >= 0 && ny >= 0 && nx < w && ny < h && !visited[(ny * w + nx) as usize] {
                stack.push((nx, ny));
            }
        }
    }

    if max_x >= min_x {
        // Blend the whole touched region toward the fill color at `opacity`.
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let t = in_region(&texture.rgba, x as i32, y as i32);
                if !t {
                    continue;
                }
                let idx = (((y as i32 * w) + x as i32) as usize) * 4;
                let blend = opacity;
                let inv = 1.0 - blend;
                texture.rgba[idx] = (color[0] as f32 * blend + texture.rgba[idx] as f32 * inv)
                    .round()
                    .min(255.0) as u8;
                texture.rgba[idx + 1] = (color[1] as f32 * blend
                    + texture.rgba[idx + 1] as f32 * inv)
                    .round()
                    .min(255.0) as u8;
                texture.rgba[idx + 2] = (color[2] as f32 * blend
                    + texture.rgba[idx + 2] as f32 * inv)
                    .round()
                    .min(255.0) as u8;
                texture.rgba[idx + 3] = (color[3] as f32 * blend
                    + texture.rgba[idx + 3] as f32 * inv)
                    .round()
                    .min(255.0) as u8;
            }
        }
        match dirty {
            Some(d) => {
                d.0 = d.0.min(min_x);
                d.1 = d.1.min(min_y);
                d.2 = d.2.max(max_x);
                d.3 = d.3.max(max_y);
            }
            None => *dirty = Some((min_x, min_y, max_x, max_y)),
        }
    }
}

/// Flood-fills every texel covered by triangles in the same connected
/// component (UV island) as `seed_triangle`.
pub fn fill_region(mesh: &mut MeshData, seed_triangle: usize, color: [u8; 4], opacity: f32) {
    let Some(tex) = mesh
        .layers
        .get_mut(mesh.active_layer)
        .map(|l| &mut l.texture)
    else {
        return;
    };
    let (w, h) = (tex.width as i32, tex.height as i32);
    if w <= 0 || h <= 0 || opacity <= 0.0 {
        return;
    }
    let positions = &mesh.positions;
    let uvs = &mesh.uvs;
    let indices = &mesh.indices;
    let comps = triangle_components(positions.len(), indices);
    if seed_triangle >= comps.len() {
        return;
    }
    let seed = comps[seed_triangle];

    // Union bounding box (in texel space) of the seed component to bound the loop.
    let mut min_u = f32::MAX;
    let mut max_u = f32::MIN;
    let mut min_v = f32::MAX;
    let mut max_v = f32::MIN;
    for (tri, tri_idx) in indices.chunks_exact(3).enumerate() {
        if comps[tri] != seed {
            continue;
        }
        for k in 0..3 {
            let idx = tri_idx[k] as usize;
            min_u = min_u.min(uvs[idx].0);
            max_u = max_u.max(uvs[idx].0);
            min_v = min_v.min(uvs[idx].1);
            max_v = max_v.max(uvs[idx].1);
        }
    }
    if min_u > max_u {
        return;
    }

    let x0 = ((min_u * w as f32).floor().max(0.0) as i32).min(w - 1);
    let x1 = ((max_u * w as f32).ceil().min(w as f32) as i32)
        .max(x0)
        .min(w - 1);
    let y0 = ((min_v * h as f32).floor().max(0.0) as i32).min(h - 1);
    let y1 = ((max_v * h as f32).ceil().min(h as f32) as i32)
        .max(y0)
        .min(h - 1);

    let mut tri_uvs: Vec<(Vec2, Vec2, Vec2)> = Vec::new();
    for (tri, tri_idx) in indices.chunks_exact(3).enumerate() {
        if comps[tri] != seed {
            continue;
        }
        let (a, b, c) = (
            uvs[tri_idx[0] as usize],
            uvs[tri_idx[1] as usize],
            uvs[tri_idx[2] as usize],
        );
        tri_uvs.push((
            Vec2::new(a.0, a.1),
            Vec2::new(b.0, b.1),
            Vec2::new(c.0, c.1),
        ));
    }

    let mut dmin_x = w as u32;
    let mut dmin_y = h as u32;
    let mut dmax_x = 0u32;
    let mut dmax_y = 0u32;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let uv = uv_from_texel(x as u32, y as u32, tex.width, tex.height);
            let p = Vec2::new(uv.0, uv.1);
            let inside = tri_uvs
                .iter()
                .any(|&(a, b, c)| uv_barycentric(p, a, b, c).is_some());
            if !inside {
                continue;
            }
            let idx = (y as u32 * tex.width + x as u32) as usize * 4;
            let mut px = [
                tex.rgba[idx],
                tex.rgba[idx + 1],
                tex.rgba[idx + 2],
                tex.rgba[idx + 3],
            ];
            blend_pixel(&mut px, color, opacity);
            tex.rgba[idx..idx + 4].copy_from_slice(&px);
            dmin_x = dmin_x.min(x as u32);
            dmin_y = dmin_y.min(y as u32);
            dmax_x = dmax_x.max(x as u32);
            dmax_y = dmax_y.max(y as u32);
        }
    }
    if dmin_x <= dmax_x {
        let dirty = mesh.dirty.unwrap_or((w as u32, h as u32, 0, 0));
        mesh.dirty = Some((
            dirty.0.min(dmin_x),
            dirty.1.min(dmin_y),
            dirty.2.max(dmax_x),
            dirty.3.max(dmax_y),
        ));
    }
}

/// Reads the texel color at a hit's UV position.
pub fn pick_color(tex: &TextureData, hit: &Hit) -> [u8; 4] {
    if tex.rgba.len() < 4 {
        return [0; 4];
    }
    let (x, y) = texel_from_uv(hit.uv, tex.width, tex.height);
    let i = (y * tex.width + x) as usize * 4;
    [
        tex.rgba[i],
        tex.rgba[i + 1],
        tex.rgba[i + 2],
        tex.rgba[i + 3],
    ]
}

// ---------------------------------------------------------------------------
// Internal math helpers.
// ---------------------------------------------------------------------------

fn tri_indices(mesh: &MeshData, triangle: usize) -> (usize, usize, usize) {
    let i = triangle * 3;
    (
        mesh.indices[i] as usize,
        mesh.indices[i + 1] as usize,
        mesh.indices[i + 2] as usize,
    )
}

/// Area-weighted average of the geometric normals of every triangle whose
/// surface lies within `radius` of `center`: a smooth "surface at the brush"
/// normal. Used to align the brush footprint to the model (so dabs follow the
/// surface contour instead of being slapped flat along the camera plane).
fn local_surface_normal(
    positions: &[Vec3],
    indices: &[u32],
    center: Vec3,
    radius: f32,
) -> Option<Vec3> {
    let mut sum = Vec3::ZERO;
    let mut count = 0u32;
    for tri in indices.chunks_exact(3) {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let (a, b, c) = (positions[i0], positions[i1], positions[i2]);
        if dist_point_to_triangle(center, a, b, c) > radius {
            continue;
        }
        let n = (b - a).cross(c - a);
        let l = n.length();
        if l > 1e-12 {
            sum += n;
            count += 1;
        }
    }
    if count > 0 && sum.length_squared() > 1e-12 {
        Some(sum.normalize_or_zero())
    } else {
        None
    }
}

/// Brush-local (u, v) axes for a stamp centered at `center` on a mesh: the
/// brush plane follows the area-weighted surface normal around the dab, with
/// the camera's screen-right projected onto the tangent plane so the pattern
/// stays upright relative to the view while conforming to the surface. Both
/// axes are zero when the view direction is degenerate (the analytic
/// "touch everything" fallback). Mirrored by the cursor preview so the
/// on-screen cursor matches the painted footprint.
pub fn brush_axes(
    positions: &[Vec3],
    indices: &[u32],
    center: Vec3,
    radius: f32,
    view_dir: Vec3,
) -> (Vec3, Vec3) {
    if view_dir.length_squared() <= 1e-12 {
        return (Vec3::ZERO, Vec3::ZERO);
    }
    let up = if view_dir.y.abs() > 0.9 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let cam_u = view_dir.cross(up).normalize_or_zero();
    let cam_v = view_dir.cross(cam_u).normalize_or_zero();
    match local_surface_normal(positions, indices, center, radius) {
        Some(n) => {
            let right_on_surf = (cam_u - n * cam_u.dot(n)).normalize_or_zero();
            if right_on_surf.length_squared() > 1e-6 {
                (right_on_surf, n.cross(right_on_surf).normalize_or_zero())
            } else {
                (cam_u, cam_v)
            }
        }
        None => (cam_u, cam_v),
    }
}

/// 3D outline of a round brush dab sitting on the surface at `center`: a ring
/// of `segments` points in the brush-local tangent plane, laid out exactly like
/// the painted footprint (same radius and axes). The caller projects these to
/// the viewport to draw the brush cursor as a surface-following mask. Empty
/// when the view direction is degenerate.
///
/// This was used for the earlier 2D-projection cursor preview and is now kept
/// as a utility; the cursor overlay is rendered in 3D by the GPU shader.
#[allow(dead_code)]
pub fn brush_plane_circle(
    mesh: &MeshData,
    center: Vec3,
    radius: f32,
    view_dir: Vec3,
    segments: usize,
) -> Vec<Vec3> {
    let (u, v) = brush_axes(&mesh.positions, &mesh.indices, center, radius, view_dir);
    if u.length_squared() <= 1e-12 {
        return Vec::new();
    }
    (0..segments)
        .map(|i| {
            let a = i as f32 / segments.max(1) as f32 * std::f32::consts::TAU;
            center + u * (a.cos() * radius) + v * (a.sin() * radius)
        })
        .collect()
}

fn triangle_texel_scale(
    positions: &[Vec3],
    uvs: &[(f32, f32)],
    i0: usize,
    i1: usize,
    i2: usize,
    width: u32,
    height: u32,
) -> f32 {
    let mut acc = 0.0;
    let mut n = 0;
    for (a, b) in [(i0, i1), (i1, i2), (i2, i0)] {
        let world = (positions[b] - positions[a]).length();
        let du = (uvs[b].0 - uvs[a].0) * width as f32;
        let dv = (uvs[b].1 - uvs[a].1) * height as f32;
        let texel_len = (du * du + dv * dv).sqrt();
        if texel_len > 1e-6 {
            acc += world / texel_len;
            n += 1;
        }
    }
    acc / n.max(1) as f32
}

fn ray_triangle(origin: Vec3, dir: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Option<f32> {
    let e1 = b - a;
    let e2 = c - a;
    let p = dir.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-8 {
        return None;
    }
    let inv = 1.0 / det;
    let tvec = origin - a;
    let u = tvec.dot(p) * inv;
    if u < 0.0 || u > 1.0 {
        return None;
    }
    let q = tvec.cross(e1);
    let v = dir.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(q) * inv;
    if t < 0.0 {
        None
    } else {
        Some(t)
    }
}

/// Barycentric weights of `p` expressed in the plane of (a, b, c).
fn barycentric_3d(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> (f32, f32, f32) {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;
    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);
    let d20 = v2.dot(v0);
    let d21 = v2.dot(v1);
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1e-8 {
        return (0.0, 1.0, 0.0);
    }
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    (1.0 - v - w, v, w)
}

/// Barycentric weights of `p` in the 2-D triangle (a, b, c).
///
/// Computes all three weights via the closed-form cross-product formula and
/// returns `(weight_at_b, weight_at_c)`.  Faster than the Gram-matrix
/// approach (fewer dot products, no intermediate `Vec2` temporaries).
fn uv_barycentric(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> Option<(f32, f32)> {
    let d = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
    if d.abs() < 1e-8 {
        return None;
    }
    let inv = 1.0 / d;
    let wa = ((b.y - c.y) * (p.x - c.x) + (c.x - b.x) * (p.y - c.y)) * inv;
    let wb = ((c.y - a.y) * (p.x - c.x) + (a.x - c.x) * (p.y - c.y)) * inv;
    let wc = 1.0 - wa - wb;
    const EPS: f32 = 1e-4;
    if wa >= -EPS && wb >= -EPS && wc >= -EPS {
        Some((wb, wc))
    } else {
        None
    }
}

fn dist_point_to_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> f32 {
    let n = (b - a).cross(c - a).normalize_or_zero();
    if n == Vec3::ZERO {
        return dist_point_segment(p, a, b)
            .min(dist_point_segment(p, b, c))
            .min(dist_point_segment(p, c, a));
    }
    let centroid = (a + b + c) / 3.0;
    let d = (p - centroid).dot(n);
    let proj = p - n * d;
    let (w0, w1, w2) = barycentric_3d(proj, a, b, c);
    const EPS: f32 = 1e-4;
    if w0 >= -EPS && w1 >= -EPS && w2 >= -EPS {
        d.abs()
    } else {
        dist_point_segment(p, a, b)
            .min(dist_point_segment(p, b, c))
            .min(dist_point_segment(p, c, a))
    }
}

fn dist_point_segment(p: Vec3, a: Vec3, b: Vec3) -> f32 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0);
    (p - (a + ab * t)).length()
}

/// Applies a brush/fill (straight-alpha color) to a texel: the texel is pulled
/// toward the material by `t` (opacity × cover). BOTH rgb and alpha lerp toward
/// the brush, so a translucent color really produces a translucent texel
/// (glass) — with source-over it would stay opaque over opaque content and the
/// alpha channel would be invisible in the 3D viewport.
/// Blends `src` into `dst` (both straight alpha) with strength `t` in 0..=1.
///
/// Where the destination is fully transparent, the rgb is kept at the full
/// brush color and only alpha is scaled: interpolating rgb toward the texel's
/// transparent black would darken every partial stroke stored on an empty
/// layer, and the compositor would then read that darkened color over the
/// base layer as a muddy, near-black brush (the alpha below already makes the
/// stroke look faded — scaling rgb too double-darkens it).
fn blend_pixel(dst: &mut [u8; 4], src: [u8; 4], t: f32) {
    let t = t.clamp(0.0, 1.0);
    if t <= 0.0 {
        return;
    }
    if dst[3] == 0 {
        // Empty texel: straight color with scaled alpha. rgb stays full so the
        // first stroke on a fresh layer reads exactly like painting opaque.
        dst[..3].copy_from_slice(&src[..3]);
        dst[3] = (src[3] as f32 * t).round().clamp(0.0, 255.0) as u8;
        return;
    }
    for i in 0..4 {
        let s = src[i] as f32;
        let d = dst[i] as f32;
        dst[i] = (d + (s - d) * t).round().clamp(0.0, 255.0) as u8;
    }
}

/// Fades alpha toward transparent while PRESERVING rgb (used by the eraser).
/// Scaling the color too would leave a dark fringe in partially erased texels:
/// after the 3D pass blends (0 < a < 1) toward the backdrop, that dark rgb
/// would read as a black halo around the erased region.
fn erase_pixel(dst: &mut [u8; 4], a: f32) {
    let f = 1.0 - a.clamp(0.0, 1.0);
    dst[3] = (dst[3] as f32 * f).round() as u8;
}

/// Per-triangle connected-component ids (triangles sharing a vertex index are
/// in the same component).
fn triangle_components(positions_len: usize, indices: &[u32]) -> Vec<usize> {
    let n = indices.len() / 3;
    let mut parent: Vec<usize> = (0..n).collect();
    let mut vertex_first = vec![usize::MAX; positions_len];

    for (tri, tri_idx) in indices.chunks_exact(3).enumerate() {
        for &vi in tri_idx {
            let vi = vi as usize;
            match vertex_first[vi] {
                usize::MAX => vertex_first[vi] = tri,
                other => union(&mut parent, tri, other),
            }
        }
    }
    (0..n).map(|t| find(&mut parent, t)).collect()
}

/// Edge-connected-component ids for split lock: two triangles belong to the
/// same part only when they share a full edge (two vertices), not merely a
/// single corner vertex — so two surfaces that touch at a point stay separate
/// and a stroke locked to one part can never bleed onto the other.
fn triangle_components_edge(indices: &[u32]) -> Vec<usize> {
    let n = indices.len() / 3;
    let mut parent: Vec<usize> = (0..n).collect();
    let mut edge_first = std::collections::HashMap::with_capacity(indices.len());

    for (tri, tri_idx) in indices.chunks_exact(3).enumerate() {
        for k in 0..3 {
            let (a, b) = (tri_idx[k], tri_idx[(k + 1) % 3]);
            let edge = if a <= b { (a, b) } else { (b, a) };
            match edge_first.entry(edge) {
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(tri);
                }
                std::collections::hash_map::Entry::Occupied(e) => {
                    union(&mut parent, tri, *e.get());
                }
            }
        }
    }
    (0..n).map(|t| find(&mut parent, t)).collect()
}

fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

fn union(parent: &mut [usize], a: usize, b: usize) {
    let (ra, rb) = (find(parent, a), find(parent, b));
    if ra != rb {
        parent[ra] = rb;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::Layer;

    fn solid_texture(w: u32, h: u32, color: [u8; 4]) -> TextureData {
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for px in rgba.chunks_exact_mut(4) {
            px.copy_from_slice(&color);
        }
        TextureData {
            width: w,
            height: h,
            rgba,
        }
    }

    fn push_quad(mesh: &mut MeshData, corners: [Vec3; 4], uvs: [(f32, f32); 4], normal: Vec3) {
        let base = mesh.positions.len() as u32;
        for (i, &p) in corners.iter().enumerate() {
            mesh.positions.push(p);
            mesh.normals.push(normal);
            mesh.uvs.push(uvs[i]);
        }
        mesh.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    /// A 24-vertex cube (±0.5), each face its own disconnected UV island
    /// spanning the full [0,1]² texture: /-faces share no vertices.
    fn unit_cube() -> MeshData {
        let mut m = MeshData {
            positions: vec![],
            normals: vec![],
            uvs: vec![],
            indices: vec![],
            layers: vec![Layer::new(
                "Layer 1",
                solid_texture(64, 64, [90, 90, 90, 255]),
            )],
            active_layer: 0,
            dirty: None,
        };
        let (s, uv_face) = (0.5f32, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]);
        push_quad(
            &mut m,
            [
                Vec3::new(-s, -s, s),
                Vec3::new(s, -s, s),
                Vec3::new(s, s, s),
                Vec3::new(-s, s, s),
            ],
            uv_face,
            Vec3::Z,
        );
        push_quad(
            &mut m,
            [
                Vec3::new(s, -s, -s),
                Vec3::new(-s, -s, -s),
                Vec3::new(-s, s, -s),
                Vec3::new(s, s, -s),
            ],
            uv_face,
            -Vec3::Z,
        );
        push_quad(
            &mut m,
            [
                Vec3::new(s, -s, s),
                Vec3::new(s, -s, -s),
                Vec3::new(s, s, -s),
                Vec3::new(s, s, s),
            ],
            uv_face,
            Vec3::X,
        );
        push_quad(
            &mut m,
            [
                Vec3::new(-s, -s, -s),
                Vec3::new(-s, -s, s),
                Vec3::new(-s, s, s),
                Vec3::new(-s, s, -s),
            ],
            uv_face,
            -Vec3::X,
        );
        push_quad(
            &mut m,
            [
                Vec3::new(s, s, s),
                Vec3::new(-s, s, s),
                Vec3::new(-s, s, -s),
                Vec3::new(s, s, -s),
            ],
            uv_face,
            Vec3::Y,
        );
        push_quad(
            &mut m,
            [
                Vec3::new(-s, -s, s),
                Vec3::new(s, -s, s),
                Vec3::new(s, -s, -s),
                Vec3::new(-s, -s, -s),
            ],
            uv_face,
            -Vec3::Y,
        );
        m
    }

    /// Two disconnected quads on the XZ plane with distinct UV halves: panel 0
    /// in u∈[0,0.5], panel 1 in u∈[0.5,1].
    fn two_panels() -> MeshData {
        let mut m = MeshData {
            positions: vec![],
            normals: vec![],
            uvs: vec![],
            indices: vec![],
            layers: vec![Layer::new(
                "Layer 1",
                solid_texture(64, 64, [200, 200, 200, 255]),
            )],
            active_layer: 0,
            dirty: None,
        };
        push_quad(
            &mut m,
            [
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 1.0),
                Vec3::new(0.0, 0.0, 1.0),
            ],
            [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)],
            Vec3::Y,
        );
        push_quad(
            &mut m,
            [
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(3.0, 0.0, 0.0),
                Vec3::new(3.0, 0.0, 1.0),
                Vec3::new(2.0, 0.0, 1.0),
            ],
            [(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)],
            Vec3::Y,
        );
        m
    }

    /// Like `two_panels`, but wound +Y (counter-clockwise when viewed from
    /// above), so both quads face an eye looking down at the y=0 plane — the
    /// facing gate of the stamp is otherwise back-coneulled and paints nothing.
    fn two_panels_up() -> MeshData {
        let mut m = MeshData {
            positions: vec![],
            normals: vec![],
            uvs: vec![],
            indices: vec![],
            layers: vec![Layer::new(
                "Layer 1",
                solid_texture(64, 64, [200, 200, 200, 255]),
            )],
            active_layer: 0,
            dirty: None,
        };
        push_quad(
            &mut m,
            [
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(1.0, 0.0, 1.0),
                Vec3::new(1.0, 0.0, 0.0),
            ],
            [(0.0, 0.0), (0.0, 1.0), (0.5, 1.0), (0.5, 0.0)],
            Vec3::Y,
        );
        push_quad(
            &mut m,
            [
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 1.0),
                Vec3::new(3.0, 0.0, 1.0),
                Vec3::new(3.0, 0.0, 0.0),
            ],
            [(0.5, 0.0), (0.5, 1.0), (1.0, 1.0), (1.0, 0.0)],
            Vec3::Y,
        );
        m
    }

    fn texel(m: &MeshData, x: u32, y: u32) -> [u8; 4] {
        let t = m.active_layer_texture().unwrap();
        let i = (y * t.width + x) as usize * 4;
        [t.rgba[i], t.rgba[i + 1], t.rgba[i + 2], t.rgba[i + 3]]
    }

    #[test]
    fn ray_hits_cube_face() {
        let mesh = unit_cube();
        let hit = mesh_raycast(&mesh, Vec3::new(0.0, 0.0, 3.0), Vec3::new(0.0, 0.0, -1.0))
            .expect("should hit +Z face");
        assert!((hit.position.x).abs() < 1e-6);
        assert!((hit.position.y).abs() < 1e-6);
        assert!((hit.position.z - 0.5).abs() < 1e-6);
        assert!((hit.uv.0 - 0.5).abs() < 1e-6);
        assert!((hit.uv.1 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn stamp_positions_step_at_spacing_and_end_on_target() {
        // 25px segment, 10px spacing -> dabs at 0, 10, 20, then the endpoint.
        let pts = stamp_positions(Vec2::ZERO, Vec2::new(25.0, 0.0), 10.0);
        assert_eq!(pts.len(), 4);
        for p in &pts {
            assert!((p.y).abs() < 1e-6);
        }
        assert!((pts[0].x - 0.0).abs() < 1e-6);
        assert!((pts[1].x - 10.0).abs() < 1e-6);
        assert!((pts[2].x - 20.0).abs() < 1e-6);
        assert!((pts[3].x - 25.0).abs() < 1e-6);
        // Monotonic non-decreasing along the axis.
        for w in pts.windows(2) {
            assert!(w[1].x >= w[0].x, "positions must be ordered");
        }
    }

    #[test]
    fn spaced_dabs_stay_uniform_even_on_slow_jittery_drag() {
        // 1px of drift per frame with spacing 6: exactly one dab every 6px of
        // travel (6, 12, 18), never one per frame.
        use egui::pos2;
        let mut acc = 0.0;
        let mut prev = pos2(0.0, 0.0);
        let mut last_dab = prev;
        let mut all = Vec::new();
        for i in 1..=20u32 {
            let now = pos2(i as f32, 0.0);
            let (dabs, a, ld) = spaced_freehand_dabs(prev, now, last_dab, acc, 6.0);
            all.extend(dabs);
            acc = a;
            last_dab = ld;
            prev = now;
        }
        assert_eq!(all, vec![pos2(6.0, 0.0), pos2(12.0, 0.0), pos2(18.0, 0.0)]);
    }

    #[test]
    fn local_surface_normal_hugs_the_curve() {
        let m = MeshData::uv_sphere(1.0, 24, 32);
        // On the equator (phi = 0) the surface normal at (1,0,0) is +X; the
        // footprint-averaged normal must stay close to it, not to the camera.
        let n = local_surface_normal(&m.positions, &m.indices, Vec3::new(1.0, 0.0, 0.0), 0.35)
            .expect("footprint around (1,0,0) has surface");
        assert!(
            n.dot(Vec3::X) > 0.98,
            "limb normal should hug the surface, got {n:?}"
        );
        // The north pole keeps a mostly-up normal even with a wide footprint.
        let nq = local_surface_normal(&m.positions, &m.indices, Vec3::new(0.0, 1.0, 0.0), 0.3)
            .expect("footprint around (0,1,0) has surface");
        assert!(
            nq.dot(Vec3::Y) > 0.97,
            "pole normal should point up, got {nq:?}"
        );
        // Away from the mesh there is nothing to estimate.
        assert!(
            local_surface_normal(&m.positions, &m.indices, Vec3::new(5.0, 5.0, 5.0), 0.1).is_none()
        );
    }

    #[test]
    fn spaced_dabs_fast_flick_gets_evenly_spaced_dabs() {
        // A 50px flick in one frame with spacing 10 -> dabs every 10px.
        use egui::pos2;
        let (dabs, acc, last) =
            spaced_freehand_dabs(pos2(0.0, 0.0), pos2(50.0, 0.0), pos2(0.0, 0.0), 0.0, 10.0);
        assert_eq!(
            dabs,
            vec![
                pos2(10.0, 0.0),
                pos2(20.0, 0.0),
                pos2(30.0, 0.0),
                pos2(40.0, 0.0),
                pos2(50.0, 0.0),
            ]
        );
        assert_eq!(acc, 0.0);
        assert_eq!(last, pos2(50.0, 0.0));
    }

    #[test]
    fn stamp_positions_short_segment_and_disabled_spacing() {
        // Shorter than one step -> dab at both ends (keeps the stroke gapless).
        let pts = stamp_positions(Vec2::ZERO, Vec2::new(3.0, 4.0), 10.0);
        assert_eq!(pts, vec![Vec2::ZERO, Vec2::new(3.0, 4.0)]);
        // spacing <= 0 -> single dab at the target (per-frame freehand).
        let pts = stamp_positions(Vec2::ZERO, Vec2::new(50.0, 5.0), 0.0);
        assert_eq!(pts.len(), 1);
        assert_eq!(pts[0], Vec2::new(50.0, 5.0));
    }

    #[test]
    fn ray_misses_empty_space() {
        let mesh = unit_cube();
        assert!(mesh_raycast(&mesh, Vec3::new(0.0, 0.0, 3.0), Vec3::new(0.0, 0.0, 1.0)).is_none());
    }

    #[test]
    fn stamp_does_not_reach_through_the_object() {
        // Brushing the front of a solid must not paint/erase the surfaces that
        // face away from the brush — the hidden back of the object that a big
        // dab's world radius would otherwise reach "through the wall".
        use crate::io::MeshData;

        let mut mesh = MeshData::uv_sphere(0.6, 12, 16).with_texture(solid_texture(
            64,
            64,
            [246, 241, 232, 255],
        ));
        let hit = mesh_raycast(&mesh, Vec3::new(0.0, 0.0, 2.6), Vec3::new(0.0, 0.0, -1.0))
            .expect("hits the sphere front");

        // A dab far larger than the whole sphere, centered on the front point
        // (+Z pole): every texel on the mesh is within its world radius.
        apply_stamp(
            &mut mesh,
            hit.position,
            4.0,
            Vec3::new(0.0, 0.0, 2.6),
            Vec3::new(0.0, 0.0, -1.0),
            [0, 0, 0, 0],
            1.0,
            1.0,
            StampMode::Erase,
        );

        // A truly visible front texel (the +Z-facing center: u≈0.25, v=0.5) is,
        // of course, erased...
        let front = texel(&mesh, 16, 32);
        assert!(
            front[3] < 30,
            "visible front texel should be erased, got {front:?}"
        );
        // ...but the -Z-facing far side (u = 0.75) is untouched: the dab must
        // not slice through the whole ball.
        let back = texel(&mesh, 48, 32);
        assert_eq!(
            back,
            [246, 241, 232, 255],
            "back-facing texel must not be erased through the object, got {back:?}"
        );
        // The +X limb (u=0) sits just below the visible horizon from this eye
        // (apparent limb radius < 0.6), so it is hidden by the nearer bulge of
        // the same surface and must not be erased "around" it either.
        let limb = texel(&mesh, 0, 32);
        assert_eq!(
            limb,
            [246, 241, 232, 255],
            "texel hidden below the visible horizon must not be erased, got {limb:?}"
        );
    }

    #[test]
    fn stamp_does_not_reach_same_facing_surface_behind_a_wall() {
        // A surface can be hidden even when it faces the brush: a second,
        // same-facing wall behind the near one. Its texels are within the
        // brush's world radius (and pass the facing test) but are occluded by
        // the near wall, so they must NOT be stamped "through" the wall.
        let mut m = MeshData {
            positions: vec![],
            normals: vec![],
            uvs: vec![],
            indices: vec![],
            layers: vec![Layer::new(
                "Layer 1",
                solid_texture(64, 64, [246, 241, 232, 255]),
            )],
            active_layer: 0,
            dirty: None,
        };
        // Near quad: full extent, left half of the atlas. Far quad: 0.1 behind,
        // same facing (+Z), right half of the atlas.
        push_quad(
            &mut m,
            [
                Vec3::new(-0.5, -0.5, 0.0),
                Vec3::new(0.5, -0.5, 0.0),
                Vec3::new(0.5, 0.5, 0.0),
                Vec3::new(-0.5, 0.5, 0.0),
            ],
            [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)],
            Vec3::Z,
        );
        push_quad(
            &mut m,
            [
                Vec3::new(-0.5, -0.5, -0.1),
                Vec3::new(0.5, -0.5, -0.1),
                Vec3::new(0.5, 0.5, -0.1),
                Vec3::new(-0.5, 0.5, -0.1),
            ],
            [(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)],
            Vec3::Z,
        );
        let hit = mesh_raycast(&m, Vec3::new(0.0, 0.0, 3.0), Vec3::new(0.0, 0.0, -1.0))
            .expect("hits the near quad");

        // A dab covering both quads (half-extents well past the 0.51 max reach).
        apply_stamp_rect(
            &mut m,
            hit.position,
            1.0,
            1.0,
            Vec3::new(0.0, 0.0, 3.0),
            Vec3::new(0.0, 0.0, -1.0),
            [255, 0, 0, 255],
            1.0,
            0.5,
            StampMode::Paint,
            None,
            true,
            None,
        );

        let near = texel(&m, 16, 32); // left half (near quad) center
        assert!(
            near[0] > 230 && near[1] < 60,
            "near, visible texel should be painted, got {near:?}"
        );
        let far = texel(&m, 48, 32); // right half (occluded far quad) center
        assert_eq!(
            far,
            [246, 241, 232, 255],
            "same-facing texel hidden behind the wall must not be painted, got {far:?}"
        );
    }

    #[test]
    fn rect_stamp_paints_a_rectangle_footprint() {
        let mut m = MeshData {
            positions: vec![],
            normals: vec![],
            uvs: vec![],
            indices: vec![],
            layers: vec![Layer::new(
                "Layer 1",
                solid_texture(64, 64, [90, 90, 90, 255]),
            )],
            active_layer: 0,
            dirty: None,
        };
        push_quad(
            &mut m,
            [
                Vec3::new(-0.5, -0.5, 0.0),
                Vec3::new(0.5, -0.5, 0.0),
                Vec3::new(0.5, 0.5, 0.0),
                Vec3::new(-0.5, 0.5, 0.0),
            ],
            [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            Vec3::Z,
        );

        // A 0.3 x 0.1 (world) rectangle centered on the quad.
        apply_stamp_rect(
            &mut m,
            Vec3::ZERO,
            0.3,
            0.1,
            Vec3::new(0.0, 0.0, 3.0),
            Vec3::new(0.0, 0.0, -1.0),
            [255, 0, 0, 255],
            1.0,
            0.5,
            StampMode::Paint,
            None,
            true,
            None,
        );

        // Center is inside the rectangle → painted.
        let center = texel(&m, 32, 32);
        assert!(
            center[0] > 230,
            "rect center should be painted, got {center:?}"
        );
        // World x=0.35 (uv u≈0.85, texel x 54) is beyond half_w=0.3 → untouched.
        let outside_x = texel(&m, 54, 32);
        assert_eq!(
            outside_x,
            [90, 90, 90, 255],
            "texel outside the rectangle's half-width must be untouched, got {outside_x:?}"
        );
        // World y=0.35 (uv v≈0.85, texel y 54) is beyond half_h=0.1 → untouched.
        let outside_y = texel(&m, 32, 54);
        assert_eq!(
            outside_y,
            [90, 90, 90, 255],
            "texel outside the rectangle's half-height must be untouched, got {outside_y:?}"
        );
    }

    /// A flat XY quad spanning 4×4 world units, UV-mapped to the full [0,1]²
    /// texture: world x = 4u − 2, world y = 4v − 2. Front (+Z) facing.
    fn uv_quad_plane() -> MeshData {
        let mut m = MeshData {
            positions: vec![],
            normals: vec![],
            uvs: vec![],
            indices: vec![],
            layers: vec![Layer::new(
                "Layer 1",
                solid_texture(64, 64, [246, 241, 232, 255]),
            )],
            active_layer: 0,
            dirty: None,
        };
        push_quad(
            &mut m,
            [
                Vec3::new(-2.0, -2.0, 0.0),
                Vec3::new(2.0, -2.0, 0.0),
                Vec3::new(2.0, 2.0, 0.0),
                Vec3::new(-2.0, 2.0, 0.0),
            ],
            [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            Vec3::Z,
        );
        m
    }

    /// 4×4 sprite: the left 2 columns are opaque white, the right 2 transparent.
    fn hl_sprite(left_white: bool) -> TextureData {
        let mut rgba = vec![0u8; 4 * 4 * 4];
        for y in 0..4u32 {
            for x in 0..4u32 {
                let i = ((y * 4 + x) * 4) as usize;
                let opaque = if left_white { x < 2 } else { x >= 2 };
                rgba[i..i + 4].copy_from_slice(&if opaque {
                    [255, 255, 255, 255]
                } else {
                    [255, 255, 255, 0]
                });
            }
        }
        TextureData {
            width: 4,
            height: 4,
            rgba,
        }
    }

    fn style_with(shape: BrushShape, sprite: Option<TextureData>, flip_x: bool) -> BrushStyle {
        BrushStyle {
            shape,
            sprite,
            rotation: 0.0,
            flip_x,
            flip_y: false,
        }
    }

    fn paint_once(m: &mut MeshData, style: &BrushStyle, mode: StampMode) {
        apply_stamp_with(
            m,
            Vec3::ZERO,
            1.0,
            Vec3::new(0.0, 0.0, 3.0),
            Vec3::new(0.0, 0.0, -1.0),
            [255, 0, 0, 255],
            1.0,
            0.0,
            mode,
            style,
            None,
            true,
            None,
        );
    }

    #[test]
    fn square_stamp_covers_corners_that_round_misses() {
        let mut sq = uv_quad_plane();
        let mut rd = uv_quad_plane();
        paint_once(
            &mut sq,
            &style_with(BrushShape::Square, None, false),
            StampMode::Paint,
        );
        apply_stamp(
            &mut rd,
            Vec3::ZERO,
            1.0,
            Vec3::new(0.0, 0.0, 3.0),
            Vec3::new(0.0, 0.0, -1.0),
            [255, 0, 0, 255],
            1.0,
            0.0,
            StampMode::Paint,
        );

        // Corner texel (world x≈0.72, y≈0.84): inside the square's |x|,|y| ≤ 1
        // footprint but outside the round disc (x²+y² > 1).
        assert_ne!(
            texel(&sq, 43, 45),
            [246, 241, 232, 255],
            "square footprint should reach its corner texel"
        );
        assert_eq!(
            texel(&rd, 43, 45),
            [246, 241, 232, 255],
            "round footprint must NOT reach that corner texel"
        );
        // On the +X axis beyond the square edge (world x≈1.22): neither paints.
        assert_eq!(texel(&sq, 51, 32), [246, 241, 232, 255]);
        assert_eq!(texel(&rd, 51, 32), [246, 241, 232, 255]);
    }

    #[test]
    fn diamond_stamp_rejects_diagonal_corners_round_keeps() {
        let mut dm = uv_quad_plane();
        let mut rd = uv_quad_plane();
        paint_once(
            &mut dm,
            &style_with(BrushShape::Diamond, None, false),
            StampMode::Paint,
        );
        apply_stamp(
            &mut rd,
            Vec3::ZERO,
            1.0,
            Vec3::new(0.0, 0.0, 3.0),
            Vec3::new(0.0, 0.0, -1.0),
            [255, 0, 0, 255],
            1.0,
            0.0,
            StampMode::Paint,
        );

        // Diagonal texel (world x=y≈0.59): |x|+|y|≈1.19 > 1 → outside the
        // diamond, yet x²+y²≈0.70 < 1 → inside the round disc.
        assert_eq!(
            texel(&dm, 41, 41),
            [246, 241, 232, 255],
            "diamond must NOT paint its diagonal corners"
        );
        assert_ne!(
            texel(&rd, 41, 41),
            [246, 241, 232, 255],
            "round footprint reaches that diagonal texel"
        );
        // Axis texel (world x≈0.84, y≈0): |x|+|y|≈0.84 ≤ 1 → inside the diamond.
        assert_ne!(
            texel(&dm, 45, 32),
            [246, 241, 232, 255],
            "diamond paints its axis arms"
        );
    }

    /// 4×4 sprite with a single opaque pixel anywhere (all else transparent).
    fn dot_sprite(ox: u32, oy: u32) -> TextureData {
        let mut rgba = vec![0u8; 4 * 4 * 4];
        let i = ((oy * 4 + ox) * 4) as usize;
        rgba[i..i + 4].copy_from_slice(&[255, 255, 255, 255]);
        TextureData {
            width: 4,
            height: 4,
            rgba,
        }
    }

    #[test]
    fn texture_stamp_tiles_the_pattern_in_world_space() {
        // A texture brush samples its sprite at each texel's GLOBAL atlas
        // coordinate (x mod w, y mod h) — NOT relative to the dab center — so
        // the pattern is anchored to the surface: texels on the same atlas
        // cell repeat the sprite wherever the dab lands. `paint_once` paints a
        // confirmed round dab over the whole quad, so only the pattern decides.
        let mut m = uv_quad_plane();
        paint_once(
            &mut m,
            &style_with(BrushShape::Texture, Some(dot_sprite(0, 0)), false),
            StampMode::Paint,
        );
        // Opaque at cell (0,0) → texels with x%4==0 && y%4==0 paint...
        assert_ne!(
            texel(&m, 32, 32),
            [246, 241, 232, 255],
            "opaque cell (32%4, 32%4)=(0,0) at the dab center paints"
        );
        assert_ne!(
            texel(&m, 28, 32),
            [246, 241, 232, 255],
            "the same cell repeats every 4 texels at (28,32)"
        );
        // ...other cells stay clear.
        assert_eq!(
            texel(&m, 46, 32),
            [246, 241, 232, 255],
            "transparent cell (46%4, 32%4)=(2,0) stays clear"
        );
        assert_eq!(
            texel(&m, 34, 32),
            [246, 241, 232, 255],
            "transparent cell (34%4, 32%4)=(2,0) stays clear"
        );

        // flip_x mirrors the tiled lattice (x → −x mod 4): the (1,0) opaque
        // cell shifts from columns x%4==1 to x%4==3.
        let one = BrushStyle {
            sprite: Some(dot_sprite(1, 0)),
            ..style_with(BrushShape::Texture, None, false)
        };
        let mut mf = uv_quad_plane();
        let one_mirrored = BrushStyle {
            flip_x: true,
            ..one.clone()
        };
        paint_once(&mut mf, &one_mirrored, StampMode::Paint);
        assert_eq!(
            texel(&mf, 33, 32),
            [246, 241, 232, 255],
            "flip_x moves the (1,0) cell off columns x%4==1"
        );
        assert_ne!(
            texel(&mf, 35, 32),
            [246, 241, 232, 255],
            "flip_x paints columns x%4==3 instead"
        );

        // rotation spins the whole lattice; a 90° turn maps the (1,0) cell to
        // gu=−y, gv=x → opaque where x%4==0 && y%4==3.
        let mut mro = uv_quad_plane();
        let one_rot = BrushStyle {
            rotation: std::f32::consts::FRAC_PI_2,
            ..one
        };
        paint_once(&mut mro, &one_rot, StampMode::Paint);
        assert_ne!(
            texel(&mro, 16, 35),
            [246, 241, 232, 255],
            "90° rotation puts the (1,0) cell at x%4==0 && y%4==3"
        );
        assert_eq!(
            texel(&mro, 17, 32),
            [246, 241, 232, 255],
            "the cell (17%4, 32%4)=(1,0) is empty after rotation"
        );
    }

    fn texel2d(t: &TextureData, x: u32, y: u32) -> [u8; 4] {
        let i = (y * t.width + x) as usize * 4;
        [t.rgba[i], t.rgba[i + 1], t.rgba[i + 2], t.rgba[i + 3]]
    }

    #[test]
    fn stamp_2d_texture_tiles_pattern_at_global_texels() {
        // A 4×1 sprite with only column 0 opaque, stamped into a 32² canvas at
        // (16,16) r=8. The pattern is anchored to the CANVAS: texels whose
        // column ≡ 0 (mod 4) inside the round dab paint, the rest stay clear —
        // not a dab-centered image mapped into the footprint.
        let bg = [246, 241, 232, 255];
        let mut tex = solid_texture(32, 32, bg);
        let mut sprite_rgba = vec![0u8; 4 * 1 * 4];
        sprite_rgba[..4].copy_from_slice(&[255, 255, 255, 255]);
        let style = BrushStyle {
            shape: BrushShape::Texture,
            sprite: Some(TextureData {
                width: 4,
                height: 1,
                rgba: sprite_rgba,
            }),
            rotation: 0.0,
            flip_x: false,
            flip_y: false,
        };
        stamp_2d(
            &mut tex,
            (0.5, 0.5),
            8.0,
            BrushShape::Texture,
            [255, 0, 0, 255],
            1.0,
            1.0,
            StampMode::Paint,
            &style,
            &mut None,
            true,
            None,
        );

        // Columns ≡ 0 (mod 4) paint wherever the round dab covers them…
        assert_ne!(texel2d(&tex, 16, 12), bg, "center-region stripe paints");
        assert_ne!(texel2d(&tex, 12, 16), bg, "stripe x=12 paints");
        assert_ne!(texel2d(&tex, 20, 16), bg, "stripe x=20 paints");
        assert_ne!(
            texel2d(&tex, 16, 20),
            bg,
            "stripe row doesn't matter (height 1)"
        );
        // …columns between the stripes stay clear…
        assert_eq!(
            texel2d(&tex, 18, 16),
            bg,
            "x=18 (outside the stripe) stays clear"
        );
        // …and the dab footprint still bounds the stroke: x=24 is on the rim.
        assert_eq!(
            texel2d(&tex, 24, 16),
            bg,
            "rim texel stays clear (stripe, but dab=0)"
        );
        assert_eq!(texel2d(&tex, 16, 7), bg, "above the disc stays clear");
    }

    #[test]
    fn stamp_2d_non_accumulate_caps_the_max_pattern_alpha() {
        // The stroke blend spec: per-texel `target = min(opacity, dab ×
        // pattern)` and `stroke_buffer[x,y] = max(stroke_buffer[x,y], target)`.
        // A second overlapping dab must not deepen a texel already at its
        // target, and the result exactly equals a single dab.
        let bg = [246, 241, 232, 255];
        let style = BrushStyle {
            shape: BrushShape::Texture,
            sprite: Some(TextureData {
                width: 2,
                height: 2,
                rgba: [255u8, 255, 255, 255].repeat(4), // fully opaque pattern
            }),
            rotation: 0.0,
            flip_x: false,
            flip_y: false,
        };
        let dab = |t: &mut TextureData, sa: &mut [u8]| {
            stamp_2d(
                t,
                (0.5, 0.5),
                8.0,
                BrushShape::Texture,
                [255, 0, 0, 255],
                0.5,
                1.0,
                StampMode::Paint,
                &style,
                &mut None,
                false,
                Some(sa),
            );
        };
        let mut single = solid_texture(32, 32, bg);
        dab(&mut single, &mut vec![0u8; (32 * 32) as usize]);
        let single_center = texel2d(&single, 16, 12);

        let mut capped = solid_texture(32, 32, bg);
        let mut buf = vec![0u8; (32 * 32) as usize];
        dab(&mut capped, &mut buf);
        dab(&mut capped, &mut buf);
        assert_eq!(
            texel2d(&capped, 16, 12),
            single_center,
            "non-accumulate caps at the max: a second overlapping dab must not deepen the texel"
        );
    }

    #[test]
    fn texture_stamp_paints_only_the_front_side() {
        use crate::io::MeshData;
        let mut mesh = MeshData::uv_sphere(1.0, 48, 64).with_texture(solid_texture(
            64,
            64,
            [246, 241, 232, 255],
        ));
        let eye = Vec3::new(0.0, 0.0, 3.5);
        let view_dir = Vec3::new(0.0, 0.0, -1.0);
        let hit = mesh_raycast(&mesh, eye, view_dir).expect("center ray hits the sphere front");
        // Fully opaque sprite: whatever the footprint test lets through paints.
        let sprite = TextureData {
            width: 4,
            height: 4,
            rgba: [255u8, 255, 255, 255].repeat(16),
        };
        let style = BrushStyle {
            shape: BrushShape::Texture,
            sprite: Some(sprite),
            rotation: 0.0,
            flip_x: false,
            flip_y: false,
        };
        apply_stamp_with(
            &mut mesh,
            hit.position,
            0.45,
            eye,
            view_dir,
            [255, 0, 0, 255],
            1.0,
            1.0,
            StampMode::Paint,
            &style,
            None,
            true,
            None,
        );

        // Map every painted texel back to its 3D position (uv_sphere puts the
        // seam at u=0/1 and the row = polar angle) and assert it faces the eye
        // — no texel on the far side (normals pointing away from the brush)
        // may be stamped through the object.
        let tex = &mesh.layers[0].texture;
        let pi = std::f32::consts::PI;
        let (w, h) = (tex.width as f32, tex.height as f32);
        let mut painted = 0u32;
        for y in 0..tex.height {
            for x in 0..tex.width {
                let p = (y as usize * tex.width as usize + x as usize) * 4;
                let px = &tex.rgba[p..p + 4];
                if px[3] < 255 || px[0] == 246 {
                    continue;
                }
                painted += 1;
                let u = (x as f32 + 0.5) / w;
                let v = (y as f32 + 0.5) / h;
                let theta = pi * v;
                let phi = 2.0 * pi * u;
                let (st, ct) = theta.sin_cos();
                let (sp, cp) = phi.sin_cos();
                let pos = Vec3::new(st * cp, ct, st * sp);
                assert!(
                    pos.dot(eye - pos) > 0.0,
                    "texture stamp painted a far-side texel at ({x},{y}) (faces away from the eye)"
                );
            }
        }
        assert!(
            painted > 100,
            "the sprite stamp must paint the front, got {painted}"
        );
    }

    #[test]
    fn texture_eraser_clears_only_the_opaque_side() {
        let mut m = uv_quad_plane();
        paint_once(
            &mut m,
            &style_with(BrushShape::Texture, Some(hl_sprite(true)), false),
            StampMode::Erase,
        );
        // hl_sprite(left-white) tiles: sprite column 0 (x%4==0) is opaque and
        // sits at dab_alpha=1 (dist 0.44 ≤ 0.55 core) → fully erased.
        assert_eq!(
            texel(&m, 24, 32)[3],
            0,
            "opaque sprite cell erased to full transparency"
        );
        // Sprite column 2 (x%4==2) is transparent → untouched.
        assert_eq!(
            texel(&m, 46, 32),
            [246, 241, 232, 255],
            "transparent sprite cell untouched by eraser"
        );
    }

    #[test]
    fn texel_conversions_round_trip_and_clamp() {
        let (w, h) = (64, 32);
        let (x, y) = texel_from_uv((0.5, 0.5), w, h);
        assert_eq!((x, y), (32, 16));
        let uv = uv_from_texel(32, 16, w, h);
        assert!((uv.0 - 0.5078125).abs() < 1e-6);
        assert!((uv.1 - 0.515625).abs() < 1e-6);
        assert_eq!(texel_from_uv((-3.0, 99.0), w, h), (0, h - 1));
    }

    #[test]
    fn stamp_records_dirty_rect() {
        let mut mesh = unit_cube();
        let hit = mesh_raycast(&mesh, Vec3::new(0.0, 0.0, 3.0), Vec3::new(0.0, 0.0, -1.0)).unwrap();
        let radius = brush_radius_world(&mesh, &hit, 64, 64, 8.0);
        apply_stamp(
            &mut mesh,
            hit.position,
            radius,
            Vec3::new(0.0, 0.0, 3.0),
            Vec3::new(0.0, 0.0, -1.0),
            [255, 0, 0, 255],
            1.0,
            0.5,
            StampMode::Paint,
        );

        let (x0, y0, x1, y1) = mesh.dirty.expect("stamp must record a dirty rect");
        assert!(
            x0 <= 32 && 32 <= x1 && y0 <= 32 && 32 <= y1,
            "dirty rect must contain the stamped texel, got ({x0},{y0})-({x1},{y1})"
        );
        // An 8px brush only touches a handful of texels, not the whole atlas.
        assert!(
            x1 - x0 <= 24 && y1 - y0 <= 24,
            "dirty rect should be brush-sized, got ({x0},{y0})-({x1},{y1})"
        );

        // A second dab merges into the same rect without resetting it.
        let hit2 =
            mesh_raycast(&mesh, Vec3::new(0.0, 0.0, 3.0), Vec3::new(0.1, 0.0, -1.0)).unwrap();
        apply_stamp(
            &mut mesh,
            hit2.position,
            radius,
            Vec3::new(0.0, 0.0, 3.0),
            Vec3::new(0.0, 0.0, -1.0),
            [255, 0, 0, 255],
            1.0,
            0.5,
            StampMode::Paint,
        );
        let d2 = mesh.dirty.unwrap();
        assert!(
            d2.2 >= x1 && d2.0 <= x0,
            "second dab must extend (not shrink) the dirty rect: {d2:?}"
        );
    }

    #[test]
    fn stamp_paints_center_color() {
        let mut mesh = unit_cube();
        let hit = mesh_raycast(&mesh, Vec3::new(0.0, 0.0, 3.0), Vec3::new(0.0, 0.0, -1.0)).unwrap();
        let radius = brush_radius_world(&mesh, &hit, 64, 64, 8.0);
        assert!(radius > 0.05 && radius < 0.2, "radius {radius}");

        apply_stamp(
            &mut mesh,
            hit.position,
            radius,
            Vec3::new(0.0, 0.0, 3.0),
            Vec3::new(0.0, 0.0, -1.0),
            [255, 0, 0, 255],
            1.0,
            0.5,
            StampMode::Paint,
        );

        let c = texel(&mesh, 32, 32);
        assert!(
            c[0] > 230 && c[1] < 40 && c[3] > 200,
            "center texel should be strongly painted, got {c:?} (seam texels are double-blended)"
        );

        // Well outside the 8-texel brush (d ~ 19px world 0.3) stays untouched.
        assert_eq!(texel(&mesh, 58, 32), [90, 90, 90, 255]);
        assert_eq!(texel(&mesh, 32, 2), [90, 90, 90, 255]);
    }

    #[test]
    fn stamp_falloff_is_monotonic() {
        let mut mesh = unit_cube();
        let hit = mesh_raycast(&mesh, Vec3::new(0.0, 0.0, 3.0), Vec3::new(0.0, 0.0, -1.0)).unwrap();
        let radius = brush_radius_world(&mesh, &hit, 64, 64, 10.0);
        apply_stamp(
            &mut mesh,
            hit.position,
            radius,
            Vec3::new(0.0, 0.0, 3.0),
            Vec3::new(0.0, 0.0, -1.0),
            [200, 0, 0, 255],
            1.0,
            0.6,
            StampMode::Paint,
        );

        let row: Vec<u8> = (0..64).map(|x| texel(&mesh, x, 32)[0]).collect();
        for x in 1..64 {
            if row[x] > row[x - 1] && row[x] <= 90 {
                // Background is 90; the red channel rises toward the center and
                // falls away again — catch a rise after it started falling.
                panic!("red channel not monotonic decreasing from the center at x={x}");
            }
        }
        let center = (26..38)
            .map(|x| (x, row[x]))
            .max_by_key(|&(_, r)| r)
            .unwrap()
            .0;
        assert!(
            (25..40).contains(&center),
            "peak should sit near brush center"
        );
    }

    #[test]
    fn hardness_keeps_full_strength_core() {
        // Classic hardness semantics: the inner `hardness` fraction of the
        // radius paints at full strength, then fades linearly to the edge.
        let stamp = |hardness: f32| -> usize {
            let mut mesh = unit_cube();
            let hit =
                mesh_raycast(&mesh, Vec3::new(0.0, 0.0, 3.0), Vec3::new(0.0, 0.0, -1.0)).unwrap();
            let radius = brush_radius_world(&mesh, &hit, 64, 64, 10.0);
            apply_stamp(
                &mut mesh,
                hit.position,
                radius,
                Vec3::new(0.0, 0.0, 3.0),
                Vec3::new(0.0, 0.0, -1.0),
                [200, 0, 0, 255],
                1.0,
                hardness,
                StampMode::Paint,
            );
            (0..64)
                .map(|x| texel(&mesh, x, 32)[0])
                .filter(|&r| r >= 199)
                .count()
        };

        let hard = stamp(1.0);
        let soft = stamp(0.0);
        assert!(
            hard > soft,
            "a harder brush must keep more of the dab at full strength (hard {hard} vs soft {soft})"
        );
        assert!(
            hard >= 8,
            "hardness 1 should leave a wide full-strength plateau, got {hard}"
        );
        assert!(
            soft <= 4,
            "hardness 0 should reach full strength only near the center, got {soft}"
        );
    }

    #[test]
    fn eraser_clears_alpha() {
        let mut mesh = unit_cube();
        let hit = mesh_raycast(&mesh, Vec3::new(0.0, 0.0, 3.0), Vec3::new(0.0, 0.0, -1.0)).unwrap();
        let radius = brush_radius_world(&mesh, &hit, 64, 64, 6.0);
        apply_stamp(
            &mut mesh,
            hit.position,
            radius,
            Vec3::new(0.0, 0.0, 3.0),
            Vec3::new(0.0, 0.0, -1.0),
            [0, 0, 0, 0],
            1.0,
            0.5,
            StampMode::Erase,
        );
        let c = texel(&mesh, 32, 32);
        assert!(c[3] < 30, "center should be mostly erased, got {c:?}");
        // Far corner unaffected.
        assert_eq!(texel(&mesh, 2, 2), [90, 90, 90, 255]);
    }

    #[test]
    fn stamp_paints_translucent_color() {
        // Painting a semi-transparent brush over OPAQUE content must still
        // produce a semi-transparent texel (glass look) — source-over would
        // keep alpha at 255 and the 3D viewport would show no alpha at all.
        let mut mesh = MeshData {
            positions: vec![],
            normals: vec![],
            uvs: vec![],
            indices: vec![],
            layers: vec![Layer::new(
                "Layer 1",
                solid_texture(64, 64, [246, 241, 232, 255]),
            )],
            active_layer: 0,
            dirty: None,
        };
        push_quad(
            &mut mesh,
            [
                Vec3::new(-0.5, -0.5, 0.0),
                Vec3::new(0.5, -0.5, 0.0),
                Vec3::new(0.5, 0.5, 0.0),
                Vec3::new(-0.5, 0.5, 0.0),
            ],
            [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            Vec3::Z,
        );
        let hit = mesh_raycast(&mesh, Vec3::new(0.0, 0.0, 2.0), Vec3::new(0.0, 0.0, -1.0)).unwrap();
        let radius = brush_radius_world(&mesh, &hit, 64, 64, 8.0);
        apply_stamp(
            &mut mesh,
            hit.position,
            radius,
            Vec3::new(0.0, 0.0, 2.0),
            Vec3::new(0.0, 0.0, -1.0),
            [200, 60, 60, 128],
            1.0,
            0.5,
            StampMode::Paint,
        );

        let c = texel(&mesh, 32, 32);
        assert!(
            c[3] >= 118 && c[3] <= 140,
            "os: center texel alpha should follow the brush (~128), got {c:?}"
        );
        assert!(
            c[0] > 180 && c[1] < 90 && c[2] < 90,
            "center texel should take the translucent color, got {c:?}"
        );
    }

    #[test]
    fn paint_on_empty_layer_keeps_straight_color() {
        // A soft stroke on a fresh transparent layer must store the full brush
        // color with scaled alpha — NOT an rgb darkened toward the texel's
        // transparent black (which used to read as a muddy near-black brush in
        // the 3D view once the compositor blended it over the base layer).
        let mut mesh = MeshData {
            positions: vec![],
            normals: vec![],
            uvs: vec![],
            indices: vec![],
            layers: vec![
                Layer::new("Base", solid_texture(64, 64, [246, 241, 232, 255])),
                Layer::new("Layer 2", solid_texture(64, 64, [0, 0, 0, 0])),
            ],
            active_layer: 1,
            dirty: None,
        };
        push_quad(
            &mut mesh,
            [
                Vec3::new(-0.5, -0.5, 0.0),
                Vec3::new(0.5, -0.5, 0.0),
                Vec3::new(0.5, 0.5, 0.0),
                Vec3::new(-0.5, 0.5, 0.0),
            ],
            [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            Vec3::Z,
        );
        let hit = mesh_raycast(&mesh, Vec3::new(0.0, 0.0, 2.0), Vec3::new(0.0, 0.0, -1.0)).unwrap();
        let radius = brush_radius_world(&mesh, &hit, 64, 64, 8.0);
        apply_stamp(
            &mut mesh,
            hit.position,
            radius,
            Vec3::new(0.0, 0.0, 2.0),
            Vec3::new(0.0, 0.0, -1.0),
            [255, 0, 0, 255],
            0.5,
            0.5,
            StampMode::Paint,
        );

        // Center texel: full red rgb, alpha ~50% (the brush opacity). Texel
        // (31,32) sits beside the quad's shared diagonal so it is painted by a
        // single triangle (32,32 doubles up and lands a second dab).
        let c = texel(&mesh, 31, 32);
        assert!(
            c[0] > 230 && c[1] < 30 && c[2] < 30,
            "rgb must keep the full brush color, got {c:?}"
        );
        assert!(
            (100..=135).contains(&c[3]),
            "alpha should follow the brush opacity (~0.5), got {c:?}"
        );

        // Composite over the opaque base layer must read pinkish-red (straight
        // alpha), not a darkened brown.
        let flat = mesh.flattened_atlas().unwrap();
        let i = (32 * 64 + 31) as usize * 4;
        let f = &flat.rgba[i..i + 4];
        assert!(
            f[0] > 230 && f[1] < 155 && f[2] < 155,
            "composite should be a bright pink-red, got {f:?}"
        );
    }

    #[test]
    fn pick_returns_texel_color() {
        let mut mesh = two_panels();
        let hit = mesh_raycast(&mesh, Vec3::new(0.5, 2.0, 0.5), Vec3::new(0.0, -1.0, 0.0)).unwrap();
        // Pre-paint a texel then sample it back.
        let tex = mesh.active_layer_texture_mut().unwrap();
        let i = (32 * 64 + 16) as usize * 4;
        tex.rgba[i..i + 3].copy_from_slice(&[12, 34, 56]);
        assert_eq!(
            pick_color(mesh.active_layer_texture().unwrap(), &hit),
            [12, 34, 56, 255]
        );
    }

    #[test]
    fn fill_respects_islands() {
        let mut mesh = two_panels();
        let hit = mesh_raycast(&mesh, Vec3::new(0.5, 2.0, 0.5), Vec3::new(0.0, -1.0, 0.0)).unwrap();
        fill_region(&mut mesh, hit.triangle, [0, 128, 255, 255], 1.0);

        for y in 0..64 {
            for x in 0..32 {
                let c = texel(&mesh, x, y);
                assert_eq!(
                    &c[..3],
                    &[0, 128, 255],
                    "panel 0 should be filled at ({x},{y})"
                );
            }
            for x in 32..64 {
                let c = texel(&mesh, x, y);
                assert_eq!(
                    c,
                    [200, 200, 200, 255],
                    "panel 1 must stay untouched at ({x},{y})"
                );
            }
        }
    }

    #[test]
    fn split_lock_stops_a_stamp_bleeding_onto_another_part() {
        // Two disconnected panels: panel 0 in u∈[0,0.5] (world x∈[0,1]),
        // panel 1 in u∈[0.5,1] (world x∈[2,3]), both on the y=0 plane.
        let eye = Vec3::new(0.5, 2.0, 0.5);
        let dir = Vec3::new(0.0, -1.0, 0.0);
        let center = Vec3::new(0.5, 0.0, 0.5);

        // Unlocked: a big round stamp covering both panels paints both.
        let mut open = two_panels_up();
        apply_stamp(
            &mut open,
            center,
            2.0,
            eye,
            dir,
            [255, 0, 0, 255],
            1.0,
            1.0,
            StampMode::Paint,
        );
        assert_eq!(
            texel(&open, 16, 32),
            [255, 0, 0, 255],
            "panel 0 gets painted"
        );
        assert_eq!(
            texel(&open, 38, 32),
            [255, 0, 0, 255],
            "without split lock the stamp reaches panel 1"
        );

        // Locked to the face under the brush center: only panel 0's connected
        // part is painted; panel 1 sits inside the radius but stays untouched.
        let hit = mesh_raycast(&open, eye, dir).expect("brush center hits panel 0");
        assert_eq!(
            hit.triangle, 0,
            "sanity: the seed is panel 0's first triangle"
        );
        let mut locked = two_panels_up();
        let accel = StampAccel::new(&locked, Some(hit.triangle));
        apply_stamp_with(
            &mut locked,
            center,
            2.0,
            eye,
            dir,
            [255, 0, 0, 255],
            1.0,
            1.0,
            StampMode::Paint,
            &style_with(BrushShape::Round, None, false),
            Some(&accel),
            true,
            None,
        );
        assert_eq!(
            texel(&locked, 16, 32),
            [255, 0, 0, 255],
            "split lock still paints the seeded part"
        );
        assert_eq!(
            texel(&locked, 38, 32),
            [200, 200, 200, 255],
            "split lock keeps the stamp off the separate panel"
        );
    }

    /// Two overlapping passes of the same disc over the same texel.
    fn double_dab(m: &mut MeshData, accumulate: bool, mut stroke_alpha: Option<&mut [u8]>) {
        for _ in 0..2 {
            apply_stamp_with(
                m,
                Vec3::ZERO,
                1.0,
                Vec3::new(0.0, 0.0, 3.0),
                Vec3::new(0.0, 0.0, -1.0),
                [255, 0, 0, 255],
                0.5,
                1.0,
                StampMode::Paint,
                &style_with(BrushShape::Round, None, false),
                None,
                accumulate,
                stroke_alpha.as_deref_mut(),
            );
        }
    }

    #[test]
    fn non_accumulate_caps_coverage_while_new_strokes_layer() {
        // Reference: exactly one dab at 0.5 opacity on the base (opaque) layer.
        let mut single = uv_quad_plane();
        {
            let tw = single.layers[0].texture.width as usize;
            let th = single.layers[0].texture.height as usize;
            let mut stroke_alpha = vec![0u8; tw * th];
            apply_stamp_with(
                &mut single,
                Vec3::ZERO,
                1.0,
                Vec3::new(0.0, 0.0, 3.0),
                Vec3::new(0.0, 0.0, -1.0),
                [255, 0, 0, 255],
                0.5,
                1.0,
                StampMode::Paint,
                &style_with(BrushShape::Round, None, false),
                None,
                false,
                Some(&mut stroke_alpha),
            );
        }
        let single_center = texel(&single, 32, 40);

        // Accumulate OFF with a shared stroke buffer: a second overlapping dab
        // is skipped, so the center texel equals a single pass exactly.
        let mut capped = uv_quad_plane();
        {
            let tw = capped.layers[0].texture.width as usize;
            let th = capped.layers[0].texture.height as usize;
            let mut stroke_alpha = vec![0u8; tw * th];
            double_dab(&mut capped, false, Some(&mut stroke_alpha));
        }
        assert_eq!(
            texel(&capped, 32, 40),
            single_center,
            "non-accumulate: the second overlapping dab must not change the texel, got {:?} vs {single_center:?}",
            texel(&capped, 32, 40)
        );

        // Accumulate ON: the same two dabs build up → rgb pulled well past one
        // dab's worth.
        let mut built = uv_quad_plane();
        double_dab(&mut built, true, None);
        let built_center = texel(&built, 32, 40);
        assert!(
            built_center[1] < single_center[1] || built_center[0] > single_center[0],
            "accumulate must stack past one dab, got {built_center:?} vs {single_center:?}"
        );

        // A fresh stroke (new buffer) layers over the previous stroke's result:
        // two non-accumulate passes in *separate* strokes equal two dabs stacked.
        let mut layered = uv_quad_plane();
        {
            let tw = layered.layers[0].texture.width as usize;
            let th = layered.layers[0].texture.height as usize;
            let mut stroke_alpha = vec![0u8; tw * th];
            double_dab(&mut layered, false, Some(&mut stroke_alpha));
            let mut stroke_alpha = vec![0u8; tw * th];
            double_dab(&mut layered, false, Some(&mut stroke_alpha));
        }
        assert_eq!(
            texel(&layered, 32, 40),
            built_center,
            "two capped strokes must layer exactly like accumulation, got {:?} vs {built_center:?}",
            texel(&layered, 32, 40)
        );
    }
}
