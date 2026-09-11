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
/// line in `stamp_texels` (see `OcclusionGrid` for the accelerated variant).
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

/// Uniform-grid index of the mesh triangles, built once per stamp so the
/// per-texel occlusion raycast only tests triangles whose AABB overlaps the
/// sight-line segment instead of scanning the whole mesh every texel.
///
/// Every triangle is inserted into each grid cell its AABB overlaps; a hit
/// closer than `max_dist` along a ray must have its AABB overlap that
/// segment's bounding box, so checking just the cells under the segment AABB
/// cannot miss an occluder.
struct OcclusionGrid<'a> {
    cell: f32,
    min: Vec3,
    positions: &'a [Vec3],
    indices: &'a [u32],
    cells: std::collections::HashMap<(i32, i32, i32), Vec<u32>>,
}

impl<'a> OcclusionGrid<'a> {
    fn build(positions: &'a [Vec3], indices: &'a [u32]) -> Self {
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
            let (c0, c1) = (
                Self::cell_index(tmin, min, cell),
                Self::cell_index(tmax, min, cell),
            );
            for i in c0.0..=c1.0 {
                for j in c0.1..=c1.1 {
                    for k in c0.2..=c1.2 {
                        cells.entry((i, j, k)).or_default().push(ti as u32);
                    }
                }
            }
        }
        Self {
            cell,
            min,
            positions,
            indices,
            cells,
        }
    }

    fn cell_index(p: Vec3, min: Vec3, cell: f32) -> (i32, i32, i32) {
        let v = (p - min) / cell;
        (v.x.floor() as i32, v.y.floor() as i32, v.z.floor() as i32)
    }

    /// Nearest hit distance along the ray that is (strictly) reachable before
    /// `max_dist`, ignoring triangles whose surface sits at/behind it.
    /// `visited`/`qid` are reused per stamp; distinct queries bump `qid`.
    fn nearest_before(
        &self,
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
            Self::cell_index(smin, self.min, self.cell),
            Self::cell_index(smax, self.min, self.cell),
        );
        let mut best: Option<f32> = None;
        for i in lo.0..=hi.0 {
            for j in lo.1..=hi.1 {
                for k in lo.2..=hi.2 {
                    let Some(list) = self.cells.get(&(i, j, k)) else {
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
                        let ch = &self.indices[ti * 3..ti * 3 + 3];
                        let (a, b, c) = (
                            self.positions[ch[0] as usize],
                            self.positions[ch[1] as usize],
                            self.positions[ch[2] as usize],
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
/// the current pointer position).
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
    );
}

/// Shared core of the stamp brushes (`rect = None` → disc of `radius_world`;
/// `Some((half_w, half_h))` → axis-aligned rectangle). `style` overrides the
/// disc footprint with the given brush shape; `rect` and `style` are mutually
/// exclusive (the latter wins when both are `Some` is not possible).
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
        Sprite(f32),
    }
    let sprite = style.and_then(|s| s.sprite.as_ref());
    let footprint = if let Some((hw, hh)) = rect {
        Footprint::Rect(hw, hh)
    } else if let Some(s) = style {
        match s.shape {
            BrushShape::Round => Footprint::Round(radius),
            BrushShape::Square => Footprint::Square(radius),
            BrushShape::Diamond => Footprint::Diamond(radius),
            BrushShape::Texture if sprite.is_some() => Footprint::Sprite(radius),
            BrushShape::Texture => Footprint::Round(radius), // no sprite → round
        }
    } else {
        Footprint::Round(radius)
    };
    let sprite_footprint = matches!(footprint, Footprint::Sprite(_));
    let footprint_radius = match footprint {
        Footprint::Rect(hw, hh) => (hw * hw + hh * hh).sqrt(),
        Footprint::Square(r) => r * std::f32::consts::SQRT_2,
        _ => radius,
    };

    // A zero view direction falls back to "touch everything" (no gates).
    let facing_gate = view_dir.length_squared() > 1e-12;
    let occlusion_gate = eye.length_squared() > 1e-12;
    let away = -view_dir;
    // Screen-aligned axes for the rectangle footprint (and world-up fallback).
    let axis_u = if facing_gate {
        let up = if view_dir.y.abs() > 0.9 {
            Vec3::X
        } else {
            Vec3::Y
        };
        view_dir.cross(up).normalize_or_zero()
    } else {
        Vec3::ZERO
    };
    let axis_v = if facing_gate {
        view_dir.cross(axis_u).normalize_or_zero()
    } else {
        Vec3::ZERO
    };

    let positions = &mesh.positions;
    let uvs = &mesh.uvs;
    // Occlusion stops a stroke from painting "through" the object (the far
    // side of a wall, the far interior wall across a hole, the far side of a
    // solid).  For a convex, watertight mesh viewed from outside, occlusion is
    // exact and cheaper to test per texel than via a raycast grid: a point of
    // a convex solid is visible from an external eye `e` iff its outward
    // normal points towards the eye, `normal · (e - p) > 0`, and the surface
    // seen by the brush (eye on the outward side, back-faces culled) can never
    // be hidden behind itself.  Detecting convexity is O(V·F), once per stamp,
    // and replaces the per-texel raycast — ~50× the dominant cost of a big
    // brush on the default sphere.
    let convex_centroid = mesh_is_convex(positions, &mesh.indices);
    // The dot-product shortcut presupposes the eye sits outside the solid, so
    // only take it when the eye clears the bounding sphere; an eye at or
    // inside the volume falls back to the grid, which handles it regardless.
    let eye_outside = match convex_centroid {
        Some(c) => {
            let r = positions
                .iter()
                .map(|p| (*p - c).length())
                .fold(0.0f32, f32::max);
            (eye - c).length() > r + 1e-6
        }
        None => false,
    };
    let use_fast_occ = occlusion_gate && convex_centroid.is_some() && eye_outside;
    // Occlusion needs the nearest surface along each texel's sight line; index
    // the triangles once so big brushes don't rescan the whole mesh per texel.
    let occ_grid = if occlusion_gate && !use_fast_occ {
        Some(OcclusionGrid::build(positions, &mesh.indices))
    } else {
        None
    };
    let mut occ_visited: Vec<u32> = vec![0; (mesh.indices.len() / 3).max(1)];
    let mut occ_qid = 0u32;

    for indices in mesh.indices.chunks_exact(3) {
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
                    Footprint::Sprite(r) => {
                        if !facing_gate || r <= 0.0 {
                            (0.0, false)
                        } else {
                            let s = style.expect("Sprite footprint ⇒ style is present");
                            let (x, y) = if s.rotation != 0.0 {
                                let (sr, cr) = s.rotation.sin_cos();
                                (tu * cr - tv * sr, tu * sr + tv * cr)
                            } else {
                                (tu, tv)
                            };
                            let (rx, ry) = (
                                if s.flip_x { -x } else { x },
                                if s.flip_y { -y } else { y },
                            );
                            let u = 0.5 + rx / (2.0 * r);
                            let v = 0.5 + ry / (2.0 * r);
                            if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
                                (0.0, false)
                            } else {
                                let spr = sprite.expect("Sprite footprint ⇒ sprite present");
                                let (sw, sh) = (spr.width.max(1), spr.height.max(1));
                                let (sx, sy) = (
                                    ((u * sw as f32).floor() as u32).min(sw - 1),
                                    ((v * sh as f32).floor() as u32).min(sh - 1),
                                );
                                let a = spr.rgba[((sy * sw + sx) as usize) * 4 + 3] as f32
                                    / 255.0;
                                if a <= 0.0 {
                                    (0.0, false)
                                } else {
                                    (a, true)
                                }
                            }
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

                let cover = if sprite_footprint {
                    // Texture stamp: the sprite's coverage *is* the dab profile,
                    // for both painting and erasing.
                    t
                } else if mode == StampMode::Erase {
                    // Eraser: a fully-transparent core (alpha 0, so the surface
                    // is discarded and whatever is behind it shows through) with
                    // a linear feather over the outer part of the dab. A smooth
                    // falloff everywhere would leave tiny residual alphas that
                    // write depth and occlude the far interior wall of a hole.
                    let core = 0.55;
                    let t = t.min(1.0);
                    if t <= core {
                        1.0
                    } else {
                        (1.0 - t) / (1.0 - core)
                    }
                } else {
                    // Classic hardness: full strength out to `hardness` of the
                    // radius, then a linear fade to the edge (1.0 = hard rim,
                    // 0.0 = gradient fading from the center outward).
                    let core_t = hardness.clamp(0.0, 0.999);
                    if t <= core_t {
                        1.0
                    } else {
                        ((1.0 - t) / (1.0 - core_t)).clamp(0.0, 1.0)
                    }
                };
                if cover <= 0.0 {
                    continue;
                }
                let idx = (y as u32 * tex.width + x as u32) as usize * 4;
                let mut px = [
                    tex.rgba[idx],
                    tex.rgba[idx + 1],
                    tex.rgba[idx + 2],
                    tex.rgba[idx + 3],
                ];
                match mode {
                    StampMode::Paint => blend_pixel(&mut px, color, opacity * cover),
                    StampMode::Erase => erase_pixel(&mut px, opacity * cover),
                }
                tex.rgba[idx..idx + 4].copy_from_slice(&px);
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

    fn style_with(
        shape: BrushShape,
        sprite: Option<TextureData>,
        flip_x: bool,
    ) -> BrushStyle {
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
        );
    }

    #[test]
    fn square_stamp_covers_corners_that_round_misses() {
        let mut sq = uv_quad_plane();
        let mut rd = uv_quad_plane();
        paint_once(&mut sq, &style_with(BrushShape::Square, None, false), StampMode::Paint);
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
        paint_once(&mut dm, &style_with(BrushShape::Diamond, None, false), StampMode::Paint);
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

    #[test]
    fn texture_stamp_paints_only_the_sprites_opaque_side() {
        // 4×4 sprite: the left 2 columns are opaque, the right 2 transparent.
        let mut m = uv_quad_plane();
        paint_once(
            &mut m,
            &style_with(
                BrushShape::Texture,
                Some(hl_sprite(true)),
                false,
            ),
            StampMode::Paint,
        );
        assert_ne!(
            texel(&m, 16, 32),
            [246, 241, 232, 255],
            "left (opaque) sprite half should paint, world x≈−0.97"
        );
        assert_eq!(
            texel(&m, 44, 32),
            [246, 241, 232, 255],
            "right (transparent) sprite half must stay clear, world x≈0.78"
        );

        // Flip swaps which world side the opaque half covers.
        let mut mf = uv_quad_plane();
        paint_once(
            &mut mf,
            &style_with(BrushShape::Texture, Some(hl_sprite(true)), true),
            StampMode::Paint,
        );
        assert_eq!(
            texel(&mf, 16, 32),
            [246, 241, 232, 255],
            "flip_x moves the opaque half off the left"
        );
        assert_ne!(
            texel(&mf, 44, 32),
            [246, 241, 232, 255],
            "flip_x paints the right side instead"
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
        assert_eq!(
            texel(&m, 16, 32)[3],
            0,
            "opaque sprite half erased to full transparency"
        );
        assert_eq!(
            texel(&m, 44, 32),
            [246, 241, 232, 255],
            "transparent sprite half untouched by eraser"
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
}
