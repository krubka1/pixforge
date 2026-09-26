use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use glam::{Vec2, Vec3};
use rayon::prelude::*;

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
    mesh_raycast_filtered(mesh, origin, dir, |_| true)
}

/// Casts a ray against the mesh considering only triangles that pass `filter`.
/// Useful for mesh-linked isolation to ignore foreground objects.
pub fn mesh_raycast_filtered<F: Fn(usize) -> bool>(
    mesh: &MeshData,
    origin: Vec3,
    dir: Vec3,
    filter: F,
) -> Option<Hit> {
    let dir = dir.normalize_or_zero();
    if dir == Vec3::ZERO {
        return None;
    }
    let mut best: Option<(f32, Hit)> = None;
    for (tri, indices) in mesh.indices.chunks_exact(3).enumerate() {
        if !filter(tri) {
            continue;
        }
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
    size: Vec3,
    /// Cell counts per axis — a sight-line DDA can cross at most
    /// nx+ny+nz cells, so they bound the march cap tightly.
    dims: (i32, i32, i32),
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
    let mut dims = (1i32, 1i32, 1i32);
    for (ti, ch) in indices.chunks_exact(3).enumerate() {
        let a = positions[ch[0] as usize];
        let b = positions[ch[1] as usize];
        let c = positions[ch[2] as usize];
        let tmin = a.min(b).min(c);
        let tmax = a.max(b).max(c);
        let (c0, c1) = (cell_index(tmin, min, cell), cell_index(tmax, min, cell));
        dims.0 = dims.0.max((c1.0 - c0.0 + 1).max(0));
        dims.1 = dims.1.max((c1.1 - c0.1 + 1).max(0));
        dims.2 = dims.2.max((c1.2 - c0.2 + 1).max(0));
        for i in c0.0..=c1.0 {
            for j in c0.1..=c1.1 {
                for k in c0.2..=c1.2 {
                    cells.entry((i, j, k)).or_default().push(ti as u32);
                }
            }
        }
    }
    GridIndex {
        cell,
        min,
        size,
        dims,
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
///
/// Cells are marched in increasing distance order (DDA) instead of iterating
/// the whole segment bounding box, so off-path cells are never touched and the
/// first cell along the ray that yields a hit is the nearest one.
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
    let (min, cell) = (data.min, data.cell);
    let box_max = min + data.size;
    // Slab ray/box: the t-range inside the grid, clamped to the query segment.
    let mut t0 = 0.0f32;
    let mut t1 = max_dist;
    for axis in 0..3 {
        let (o, d, lo, hi) = (origin[axis], dir[axis], min[axis], box_max[axis]);
        if d.abs() < 1e-12 {
            if o < lo || o > hi {
                return None;
            }
        } else {
            let inv = 1.0 / d;
            let mut a = (lo - o) * inv;
            let mut b = (hi - o) * inv;
            if a > b {
                std::mem::swap(&mut a, &mut b);
            }
            t0 = t0.max(a);
            t1 = t1.min(b);
        }
    }
    if t0 > t1 {
        return None;
    }
    let mut t0 = t0.max(0.0);
    // Start just inside the entry cell so a hit exactly on a cell wall is not
    // skipped by boundary rounding.
    let start = origin + dir * (t0 + cell.max(1e-6) * 1e-4);
    let (mut cx, mut cy, mut cz) = cell_index(start, min, cell);
    let step_x = if dir.x >= 0.0 { 1 } else { -1 };
    let step_y = if dir.y >= 0.0 { 1 } else { -1 };
    let step_z = if dir.z >= 0.0 { 1 } else { -1 };
    let delta_x = cell / dir.x.abs().max(1e-30);
    let delta_y = cell / dir.y.abs().max(1e-30);
    let delta_z = cell / dir.z.abs().max(1e-30);
    let wall_x = min.x + cell * (cx + if step_x > 0 { 1 } else { 0 }) as f32;
    let wall_y = min.y + cell * (cy + if step_y > 0 { 1 } else { 0 }) as f32;
    let wall_z = min.z + cell * (cz + if step_z > 0 { 1 } else { 0 }) as f32;
    let mut t_max_x = if dir.x.abs() > 1e-30 {
        (wall_x - start.x) / dir.x
    } else {
        f32::INFINITY
    };
    let mut t_max_y = if dir.y.abs() > 1e-30 {
        (wall_y - start.y) / dir.y
    } else {
        f32::INFINITY
    };
    let mut t_max_z = if dir.z.abs() > 1e-30 {
        (wall_z - start.z) / dir.z
    } else {
        f32::INFINITY
    };
    // Safety cap on the march (path cells only; hugely generous).
    let mut steps = 0u32;
    let mut best: Option<f32> = None;
    loop {
        // Cells are visited in increasing entry-distance order. Once the
        // current cell's entry is past the nearest hit so far, no later cell
        // can contain anything closer — stop.
        if best.is_some_and(|b| t0 >= b) || t0 > t1 {
            break;
        }
        if let Some(list) = data.cells.get(&(cx, cy, cz)) {
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
                    best = Some(best.map_or(t, |b| b.min(t)));
                }
            }
        }
        // Advance to the nearest cell wall.
        if t_max_x < t_max_y && t_max_x < t_max_z {
            cx += step_x;
            t_max_x += delta_x;
            t0 = t_max_x;
        } else if t_max_y < t_max_z {
            cy += step_y;
            t_max_y += delta_y;
            t0 = t_max_y;
        } else {
            cz += step_z;
            t_max_z += delta_z;
            t0 = t_max_z;
        }
        steps += 1;
        // The DDA advances one cell per step along a monotonically increasing
        // path: it crosses at most nx+ny+nz cell walls (the precomputed grid
        // dims). The generous ×4 margin is a safety factor for corner grazes;
        // the old fixed 4096 was unreachable in practice. `best` / `t1` still
        // terminate the march far earlier whenever a hit or the segment end is
        // reached.
        if steps > ((data.dims.0 + data.dims.1 + data.dims.2) * 4).max(16) as u32 {
            break;
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
    mesh_iso: Option<(usize, Vec<usize>)>,
    bvh: TriangleBvh,
    /// Reusable per-dab candidate list so the BVH query does not allocate a
    /// fresh heap vec on every stamp inside a stroke.
    candidates: RefCell<Vec<u32>>,
}

/// A bounding-volume hierarchy over the mesh's triangles. Built once per
/// stroke (geometry is immutable while painting) so each dab only visits the
/// triangles whose AABB can touch the brush sphere instead of scanning every
/// triangle in the mesh.
pub(crate) struct TriangleBvh {
    nodes: Vec<BvhNode>,
    tris: Vec<u32>,
}

#[derive(Clone, Copy)]
struct BvhNode {
    min: Vec3,
    max: Vec3,
    child0: u32,
    child1: u32,
    first: u32,
    count: u32,
}

const BVH_LEAF: usize = 12;

impl TriangleBvh {
    fn new(positions: &[Vec3], indices: &[u32]) -> Self {
        let n = indices.len() / 3;
        let tris: Vec<u32> = (0..n as u32).collect();
        let mut nodes = Vec::with_capacity(n * 2);
        if n == 0 {
            return Self { nodes, tris };
        }
        let mut tris = tris;
        Self::build_node(positions, indices, &mut tris, &mut nodes, 0, n, 0);
        Self { nodes, tris }
    }

    fn build_node(
        positions: &[Vec3],
        indices: &[u32],
        tris: &mut [u32],
        nodes: &mut Vec<BvhNode>,
        start: usize,
        end: usize,
        depth: usize,
    ) -> u32 {
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for &t in &tris[start..end] {
            let t = t as usize;
            let (va, vb, vc) = (
                positions[indices[t * 3] as usize],
                positions[indices[t * 3 + 1] as usize],
                positions[indices[t * 3 + 2] as usize],
            );
            min = min.min(va).min(vb).min(vc);
            max = max.max(va).max(vb).max(vc);
        }
        let node_index = nodes.len() as u32;
        nodes.push(BvhNode {
            min,
            max,
            child0: 0,
            child1: 0,
            first: start as u32,
            count: 0,
        });
        let count = end - start;
        if count <= BVH_LEAF || depth >= 24 {
            let n = &mut nodes[node_index as usize];
            n.first = start as u32;
            n.count = count as u32;
            return node_index;
        }
        // Binned Surface-Area Heuristic: for each of the 3 axes, bin the
        // triangle centroids into 8 buckets, accumulate each bucket's own
        // AABB, and sweep all 7 split points for the cheapest
        // left_n·SA(left) + right_n·SA(right). O(N) per node (a single pass
        // plus constant bucket work) instead of the O(N log N) centroid sort
        // the median split used, and it yields measurably better trees on
        // non-uniform meshes. Falls back to a plain count split when every
        // axis is degenerate (all centroids collapse to one point).
        let splits = [
            Self::best_sah_split(positions, indices, &tris[start..end], 0),
            Self::best_sah_split(positions, indices, &tris[start..end], 1),
            Self::best_sah_split(positions, indices, &tris[start..end], 2),
        ];
        let mut best: Option<(usize, usize)> = None; // (axis, split bin)
        let mut best_cost = f32::INFINITY;
        for (axis, split) in splits.into_iter().enumerate() {
            if let Some((k, cost)) = split {
                if cost < best_cost {
                    best_cost = cost;
                    best = Some((axis, k));
                }
            }
        }
        let mid_in_slice = match best {
            Some((axis, k)) => {
                // Partition the slice: bin < k goes left, the rest right. The
                // predicate is recomputed per element for the chosen axis.
                let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
                for &t in &tris[start..end] {
                    let c = Self::tri_centroid_axis(positions, indices, t as usize, axis);
                    lo = lo.min(c);
                    hi = hi.max(c);
                }
                let span = (hi - lo).max(1e-9);
                let inv = 8.0 / span;
                let sub = &mut tris[start..end];
                let bin = |t: &u32| {
                    let c = Self::tri_centroid_axis(positions, indices, *t as usize, axis);
                    (((c - lo) * inv).floor() as usize).min(7)
                };
                // In-place partition (stable `partition_in_place` is not
                // available on slices): bin < k goes to the left side.
                let (mut l, mut r) = (0usize, sub.len());
                while l < r {
                    if bin(&sub[l]) < k {
                        l += 1;
                        continue;
                    }
                    r -= 1;
                    sub.swap(l, r);
                }
                l
            }
            None => count / 2,
        };
        let mid = start + mid_in_slice;
        let left = Self::build_node(positions, indices, tris, nodes, start, mid, depth + 1);
        let right = Self::build_node(positions, indices, tris, nodes, mid, end, depth + 1);
        let n = &mut nodes[node_index as usize];
        n.child0 = left;
        n.child1 = right;
        node_index
    }

    /// Binned-SAH score for one axis: bins the slice's centroids into
    /// `BINS` buckets (8), accumulates each bucket's own AABB and count, then
    /// sweeps the 7 split points for the cheapest
    /// `left_n·SA(left) + right_n·SA(right)`. Returns `(split_bin, cost)` or
    /// `None` when the axis has no centroid spread (degenerate).
    fn best_sah_split(
        positions: &[Vec3],
        indices: &[u32],
        tris: &[u32],
        axis: usize,
    ) -> Option<(usize, f32)> {
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &t in tris {
            let c = Self::tri_centroid_axis(positions, indices, t as usize, axis);
            lo = lo.min(c);
            hi = hi.max(c);
        }
        if hi - lo <= 1e-9 {
            return None;
        }
        let inv = 8.0 / (hi - lo);
        let mut counts = [0u32; 8];
        let mut bmin = [Vec3::splat(f32::INFINITY); 8];
        let mut bmax = [Vec3::splat(f32::NEG_INFINITY); 8];
        for &t in tris {
            let t = t as usize;
            let i = t * 3;
            let (a, b, c) = (
                positions[indices[i] as usize],
                positions[indices[i + 1] as usize],
                positions[indices[i + 2] as usize],
            );
            let centroid = (a + b + c) * (1.0 / 3.0);
            let bin = (((centroid[axis] - lo) * inv).floor() as usize).min(7);
            counts[bin] += 1;
            bmin[bin] = bmin[bin].min(a).min(b).min(c);
            bmax[bin] = bmax[bin].max(a).max(b).max(c);
        }
        // Whole-bucket right-side suffix (AABB, count) for every split point.
        let total: u32 = counts.iter().sum();
        let mut rcount = [0u32; 8];
        let mut rmin = [Vec3::splat(f32::INFINITY); 8];
        let mut rmax = [Vec3::splat(f32::NEG_INFINITY); 8];
        for k in (0..8).rev() {
            rcount[k] = counts[k] + if k + 1 < 8 { rcount[k + 1] } else { 0 };
            rmin[k] = bmin[k].min(if k + 1 < 8 {
                rmin[k + 1]
            } else {
                Vec3::splat(f32::INFINITY)
            });
            rmax[k] = bmax[k].max(if k + 1 < 8 {
                rmax[k + 1]
            } else {
                Vec3::splat(f32::NEG_INFINITY)
            });
        }
        let sa = |min: Vec3, max: Vec3| {
            let d = max - min;
            2.0 * (d.x * d.y + d.y * d.z + d.x * d.z)
        };
        let mut best: Option<(usize, f32)> = None;
        let (mut lcount, mut lmin, mut lmax) = (
            0u32,
            Vec3::splat(f32::INFINITY),
            Vec3::splat(f32::NEG_INFINITY),
        );
        for k in 0..7 {
            lcount += counts[k];
            lmin = lmin.min(bmin[k]);
            lmax = lmax.max(bmax[k]);
            let split_bin = k + 1;
            let rn = total - lcount;
            if lcount == 0 || rn == 0 {
                continue;
            }
            let cost =
                lcount as f32 * sa(lmin, lmax) + rn as f32 * sa(rmin[split_bin], rmax[split_bin]);
            if best.is_none_or(|(_, c)| cost < c) {
                best = Some((split_bin, cost));
            }
        }
        best
    }

    /// Returns the triangle's centroid's `axis` coordinate.
    fn tri_centroid_axis(positions: &[Vec3], indices: &[u32], t: usize, axis: usize) -> f32 {
        let i = t * 3;
        let (a, b, c) = (
            positions[indices[i] as usize],
            positions[indices[i + 1] as usize],
            positions[indices[i + 2] as usize],
        );
        ((a + b + c) * (1.0 / 3.0))[axis]
    }

    /// Pushes the indices of every triangle whose AABB intersects the sphere
    /// `(center, radius)` into `out`. Conservative: a triangle may be reported
    /// even if the sphere misses it; the exact `dist_point_to_triangle` cull
    /// still runs per candidate in the stamp loop.
    fn query(&self, center: Vec3, radius: f32, out: &mut Vec<u32>) {
        if self.nodes.is_empty() && self.tris.is_empty() {
            return;
        }
        let r2 = radius * radius;
        let mut stack = vec![0u32];
        while let Some(idx) = stack.pop() {
            let node = &self.nodes[idx as usize];
            let closest = center.clamp(node.min, node.max);
            if (closest - center).length_squared() > r2 {
                continue;
            }
            if node.count != 0 {
                let first = node.first as usize;
                out.extend_from_slice(&self.tris[first..first + node.count as usize]);
            } else {
                stack.push(node.child0);
                stack.push(node.child1);
            }
        }
    }
}

impl StampAccel {
    pub fn new(mesh: &MeshData, split_seed: Option<usize>) -> Self {
        Self::with_isolation(mesh, split_seed, None)
    }

    pub fn with_isolation(
        mesh: &MeshData,
        split_seed: Option<usize>,
        mesh_seed: Option<usize>,
    ) -> Self {
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
        let mesh_iso = mesh_seed.and_then(|seed| {
            if seed >= mesh.indices.len() / 3 {
                return None;
            }
            let comps = triangle_components_mesh(&mesh.positions, &mesh.indices);
            comps.get(seed).copied().map(|c| (c, comps))
        });
        let bvh = TriangleBvh::new(&mesh.positions, &mesh.indices);
        Self {
            convex_centroid,
            bounds_center,
            bounds_radius,
            occ,
            split,
            mesh_iso,
            bvh,
            candidates: RefCell::new(Vec::new()),
        }
    }

    /// The mesh-isolation lock captured at stroke start, if any: the seed
    /// component id and the per-triangle welded-component map. The app consults
    /// it to pick the brush ray *through* foreground objects (triangles of
    /// other parts are skipped by the raycast), so a stroke stays anchored to
    /// the seeded part and never drifts onto a nearer separate part.
    pub fn mesh_isolation(&self) -> Option<(usize, &[usize])> {
        self.mesh_iso
            .as_ref()
            .map(|(seed, comps)| (*seed, comps.as_slice()))
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
    let dist = cursor_now.distance(cursor_prev);
    let mut acc = acc + dist;
    let mut last_dab = last_dab;
    let mut dabs = Vec::new();
    // Every dab advances `spacing` from the *previous dab toward the current
    // cursor*, re-aiming each step, rather than extrapolating along this
    // frame's chord (`dir * spacing`). On curved or jittery drags that kept the
    // dabs fanning straight lines off the pointer's actual path; steering at
    // the cursor keeps the resampled stroke on the curve it was drawn along.
    while acc >= spacing {
        acc -= spacing;
        let to_cur = cursor_now - last_dab;
        if to_cur.length_sq() < 1e-12 {
            // No progress left between the last dab and the cursor (e.g. the
            // pointer snapped back on top of a dab that was just placed): bail
            // out rather than stack identical dabs.
            break;
        }
        last_dab += to_cur / to_cur.length() * spacing;
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

/// How a stroke interacts with the existing texels. Lives with the brush
/// engine; re-exported here so existing call sites keep resolving.
pub use crate::brush::StampMode;

/// Legacy brush-shape enum kept for the library brush packs (`brushes.rs`) and
/// the paint test suite; the app now stores [`crate::brush::FootprintKind`].
#[allow(dead_code)] // legacy pack/test-only surface
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

#[allow(dead_code)] // legacy pack/test-only surface
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
/// Superseded by [`crate::brush::Brush`] in the app; kept for the tests and
/// pack tooling.
#[allow(dead_code)] // legacy pack/test-only surface
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

/// Builds a [`crate::brush::Brush`] from the legacy scalar brush args used by
/// the wrappers below. Sprite copies only happen where a `&TextureData` is
/// passed in; the app's dab loops hand `&core.brush` straight to
/// [`apply_brush_stamp`] so no per-dab copy occurs there.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)] // legacy builder for the wrappers below
fn style_brush(
    kind: crate::brush::FootprintKind,
    size: f32,
    hardness: f32,
    opacity: f32,
    color: [u8; 4],
    mode: StampMode,
    sprite: Option<&TextureData>,
    rotation: f32,
    flip_x: bool,
    flip_y: bool,
) -> crate::brush::Brush {
    crate::brush::Brush {
        kind,
        size,
        hardness,
        spacing: 0.0,
        opacity,
        accumulate: true,
        color,
        mode,
        sprite: sprite.cloned(),
        pattern_lock: crate::brush::PatternLock::Dab,
        rotation,
        flip_x,
        flip_y,
        texture_scale: 1.0,
        texture_locked: false,
        texture_size_lock: 0.0,
        texture_window: crate::brush::Window::Round,
    }
}

impl From<BrushShape> for crate::brush::FootprintKind {
    fn from(shape: BrushShape) -> Self {
        match shape {
            BrushShape::Round => crate::brush::FootprintKind::Round,
            BrushShape::Square => crate::brush::FootprintKind::Square,
            BrushShape::Diamond => crate::brush::FootprintKind::Diamond,
            BrushShape::Texture => crate::brush::FootprintKind::Sprite,
        }
    }
}

/// Engine entry for the 3D paint tools: stamps one dab of `brush` at `center`
/// (`radius_world` is the dab's spatial size; the brush carries the shape,
/// sprite, falloff, color, opacity, mode and accumulate flag). `rect =
/// Some((half_w, half_h))` overrides the brush footprint with the rect tool's
/// axis-aligned rectangle — the rect wins over the round/square/... footprint.
/// `accel` is the per-surface occlusion precompute; `stroke_alpha` is the
/// non-accumulative stroke buffer (see [`stamp_texels`]).
///
/// Pattern-aligned strokes paint a classic per-dab footprint: each dab samples
/// the world-locked sprite phase through its dab-local window, and the shared
/// `stroke_alpha` buffer makes the stroke a flat *replace* — overlapping dabs
/// composite exactly as if the strongest coverage were painted once, so the
/// anchored texture never stacks alpha or builds density spikes.
#[allow(clippy::too_many_arguments)]
pub fn apply_brush_stamp(
    mesh: &mut MeshData,
    center: Vec3,
    radius_world: f32,
    eye: Vec3,
    view_dir: Vec3,
    rect: Option<(f32, f32)>,
    brush: &crate::brush::Brush,
    accel: Option<&StampAccel>,
    stroke_alpha: Option<&mut [u8]>,
    pattern: Option<&crate::brush::PatternAnchor>,
    unwrap: Option<&SurfaceUnwrap>,
) {
    stamp_texels(
        mesh,
        center,
        radius_world,
        eye,
        view_dir,
        rect,
        brush,
        accel,
        stroke_alpha,
        pattern,
        unwrap,
    );
}

/// Applies the given brush style instead of the default round dab:
/// `apply_stamp` with a `Texture` shape driven by `style`. Kept for the test
/// suite; the app passes `&core.brush` to [`apply_brush_stamp`] directly.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)] // legacy wrapper; exercised by tests
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
    let mut brush = style_brush(
        style.shape.into(),
        radius_world,
        hardness,
        opacity,
        color,
        mode,
        style.sprite.as_ref(),
        style.rotation,
        style.flip_x,
        style.flip_y,
    );
    brush.accumulate = accumulate;
    apply_brush_stamp(
        mesh,
        center,
        radius_world,
        eye,
        view_dir,
        None,
        &brush,
        accel,
        stroke_alpha,
        None,
        None,
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
    let brush = style_brush(
        crate::brush::FootprintKind::Round,
        radius_world,
        hardness,
        opacity,
        color,
        mode,
        None,
        0.0,
        false,
        false,
    );
    apply_brush_stamp(
        mesh,
        center,
        radius_world,
        eye,
        view_dir,
        None,
        &brush,
        None,
        None,
        None,
        None,
    );
}

/// Like [`apply_stamp`], but the footprint is a rectangle (in world units)
/// aligned to the brush ray: texels within `half` of the surface plane along
/// two screen-aligned axes are stamped. `half_w`/`half_h` are half-extents.
/// Kept for the test suite; the app's rect tool passes a rect override to
/// [`apply_brush_stamp`] instead.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)] // legacy wrapper; exercised by tests
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
    let radius = half_w.max(half_h);
    let mut brush = style_brush(
        crate::brush::FootprintKind::Rect,
        radius,
        hardness,
        opacity,
        color,
        mode,
        None,
        0.0,
        false,
        false,
    );
    brush.accumulate = accumulate;
    apply_brush_stamp(
        mesh,
        center,
        radius,
        eye,
        view_dir,
        Some((half_w, half_h)),
        &brush,
        accel,
        stroke_alpha,
        None,
        None,
    );
}

/// Samples `sprite` at canvas texel `(x, y)` as a *world-space tiled* pattern:
/// Shared core of the stamp brushes (`rect = None` → disc of `radius_world`;
/// `Some((half_w, half_h))` → axis-aligned rectangle; the rect wins over the
/// brush footprint). `brush` supplies the footprint kind, sprite, falloff
/// profile, color, opacity, mode and accumulate flag.
/// A texture brush stamps its sprite ONCE per dab, centered on the footprint,
/// the sprite's alpha being the coverage; the sprite is not tiled across the
/// surface.
/// If `brush.accumulate` is false, `stroke_alpha` (when Some) acts as the
/// stroke's `stroke_buffer`: it tracks the MAX target alpha per texel
/// (`min(opacity, cover)`), so later dabs cap rather than stack, and each
/// texel is blended toward that value through an exact source-over step — live
/// preview is identical to compositing the final buffer once. Pattern-aligned
/// strokes *always* honor the buffer (see `pattern`), so the anchored texture
/// is a flat replace: overlapping dabs can never "fill itself up".
#[allow(clippy::too_many_arguments)]
fn stamp_texels(
    mesh: &mut MeshData,
    center: Vec3,
    radius_world: f32,
    eye: Vec3,
    view_dir: Vec3,
    rect: Option<(f32, f32)>,
    brush: &crate::brush::Brush,
    accel: Option<&StampAccel>,
    mut stroke_alpha: Option<&mut [u8]>,
    pattern: Option<&crate::brush::PatternAnchor>,
    unwrap: Option<&SurfaceUnwrap>,
) {
    let Some(texture) = mesh.active_layer_texture() else {
        return;
    };
    let ok = texture.width > 0 && texture.height > 0 && brush.opacity > 0.0 && radius_world > 0.0;
    if !ok {
        return;
    }
    // Take the active layer's texture out so the per-texel occlusion raycast
    // can borrow the mesh immutably at the same time; it is put back before
    // returning.
    let layer_idx = mesh.active_layer;
    // A zero view direction falls back to "touch everything" (no gates).
    let facing_gate = view_dir.length_squared() > 1e-12;
    let occlusion_gate = eye.length_squared() > 1e-12;
    let away = -view_dir;
    let (tw, th) = {
        let tex = &mesh.layers[layer_idx].texture;
        (tex.width as i32, tex.height as i32)
    };
    // Reuse the stroke's acceleration when one was built; otherwise build a
    // throwaway copy (tests / one-shot stamps). Geometry never changes while
    // painting, so the cached BVH / convexity / bounds / occlusion index /
    // split components stay valid for the whole stroke. Built before the
    // texture is mutably borrowed in place below (it takes the whole mesh).
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
    let mut occ_visited: Vec<u32> = if occ_grid.is_some() {
        vec![0; (mesh.indices.len() / 3).max(1)]
    } else {
        Vec::new()
    };
    let mut occ_qid = 0u32;
    let tex = &mut mesh.layers[layer_idx].texture;
    // Mutably borrow the active layer's texture in place: the loop reads the
    // mesh's positions/uvs/indices (disjoint fields) at the same time, and no
    // full-texture buffer is allocated or memcpy'd per dab.
    let (w, h) = (tw, th);
    let (tex_w, tex_h) = (w as u32, h as u32);
    let mut dirty = mesh.dirty.unwrap_or((tw as u32, th as u32, 0, 0));
    let radius = radius_world.max(1e-4);
    // Pattern-anchor world→pattern scale ratios. `radius / anchor_r` keeps the
    // texture's world size glued to the stroke-start dab; the brush's
    // `texture_scale` multiplier resizes that tile, and `texture_locked` swaps
    // the anchor radius for a fixed captured size so resizing the brush never
    // stretches the texture. Resolved once per dab, not per texel.
    let anchored_uv_scale = match pattern {
        Some(crate::brush::PatternAnchor::Uv {
            radius: anchor_r, ..
        })
        | Some(crate::brush::PatternAnchor::Canvas {
            radius: anchor_r, ..
        }) => brush.texture_scale * radius / anchor_r.max(1e-6),
        _ => 0.0,
    };
    let anchored_surface_scale = match pattern {
        Some(crate::brush::PatternAnchor::Surface {
            radius: anchor_r, ..
        }) => {
            let divider = if brush.texture_locked && brush.texture_size_lock > 0.0 {
                brush.texture_size_lock
            } else {
                anchor_r.max(1e-6)
            };
            brush.texture_scale * radius / divider
        }
        _ => 0.0,
    };
    // Resolve the brush state into the shared pure footprint + profile once
    // per dab; the per-texel loop delegates every shape/falloff decision to
    // `brush::local_coverage`, the same evaluator the 2D stamp uses.
    // `Footprint::for_dab` / `DabProfile::for_dab` hold the fallback rules
    // (rect tool wins, sprite-without-image → round, sprite alpha IS the
    // coverage, eraser feathers) so both stampers share them.
    let footprint = match rect {
        Some((half_w, half_h)) => crate::brush::Footprint::Rect { half_w, half_h },
        None => match pattern {
            Some(_) => brush.pattern_footprint(radius),
            None => brush.footprint(radius),
        },
    };
    let profile = brush.falloff();
    let footprint_radius = footprint.outer_radius();

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
        Some(&accel.bvh),
    );

    let positions = &mesh.positions;
    let uvs = &mesh.uvs;

    // Split lock / mesh isolation: skip triangles not on the seed part.
    let split_components = accel.split.as_ref();
    let mesh_iso_components = accel.mesh_iso.as_ref();

    // Broad phase: the accel's BVH (built once per stroke) only hands back the
    // triangles whose AABB can touch the brush sphere, replacing a full-mesh
    // scan (O(T) `dist_point_to_triangle` calls per dab) with a tree walk.
    // The candidate list lives on the accel and is cleared per dab, so rapid
    // dabs inside one stroke never reallocate the buffer.
    let mut candidates = accel.candidates.borrow_mut();
    candidates.clear();
    accel.bvh.query(center, footprint_radius, &mut candidates);

    for &tri in candidates.iter() {
        let tri = tri as usize;
        let indices = &mesh.indices[tri * 3..][..3];
        if let Some((seed_comp, comps)) = &split_components {
            if comps[tri] != *seed_comp {
                continue;
            }
        }
        if let Some((seed_comp, comps)) = &mesh_iso_components {
            if comps[tri] != *seed_comp {
                continue;
            }
        }
        // Surface-anchored pattern: the per-texel phase comes from the
        // geodesic unwrap's per-vertex phases (barycentric along the texel's
        // barycentric frame), or straight from the anchor-plane chord when the
        // triangle lies outside the unfolded patch.
        let tri_phases: Option<[Vec2; 3]> = unwrap.and_then(|u| u.tri(tri));
        let (i0, i1, i2) = (
            indices[0] as usize,
            indices[1] as usize,
            indices[2] as usize,
        );
        // Skip triangles the brush cannot "see" the front of. A back-facing
        // triangle is the hidden side of a wall (or the far wall across an
        // opening), so painting/erasing it would go through walls.
        // Also compute angle falloff to smoothly fade paint at grazing
        // incidence, preventing the texture from stretching across edge-on
        // faces.
        //
        // Degenerate triangles (zero area, length_squared < 1e-12) are always
        // kept (angle_factor stays 1): at each pole all ring-0 vertices
        // collapse to the same position, so one of the two row-0 quads per
        // column is degenerate in 3D.  Its UV area is still valid — together
        // with the non-degenerate half it covers the full pole texel strip —
        // so skipping it would leave half the pole unpaintable. A zero cross
        // product must NOT fall through to the back-facing cull, or the pole
        // strip silently goes dead.
        let mut angle_factor = 1.0f32;
        if facing_gate {
            let n = (positions[i1] - positions[i0]).cross(positions[i2] - positions[i0]);
            let nl = n.length();
            if nl >= 1e-12 {
                let cos_angle = (n / nl).dot(away);
                if cos_angle <= 0.02 {
                    continue;
                }
                if cos_angle < 0.22 {
                    let t = ((cos_angle - 0.02) / 0.20).clamp(0.0, 1.0);
                    angle_factor = t * t * (3.0 - 2.0 * t);
                }
            }
        }
        let (a, b, c) = (positions[i0], positions[i1], positions[i2]);

        // Per-triangle surface frame for plane-shaped dabs (plain square /
        // diamond / sprite, no pattern): the mask is measured in each texel's
        // OWN face plane instead of the brush axis plane. When the brush is
        // bigger than a face, `local_surface_normal` blends the face with its
        // neighbors and tilts the axis plane, so the projected (tu, tv) are
        // compressed along the face and the whole shape paints stretched past
        // the cursor. Reading (tu, tv) in the face's own plane keeps the shape
        // true-size on every face. The brush axes are projected onto the plane
        // and re-orthonormalized so square stays square.
        let surf_frame: Option<(Vec3, Vec3, Vec3)> = if facing_gate
            && pattern.is_none()
            && !matches!(footprint, crate::brush::Footprint::Round { .. })
            && !matches!(footprint, crate::brush::Footprint::Rect { .. })
        {
            let n = (b - a).cross(c - a);
            let nl = n.length();
            if nl >= 1e-12 {
                let nu = n / nl;
                let su0 = axis_u - nu * axis_u.dot(nu);
                let sul = su0.length();
                if sul >= 1e-4 {
                    let su = su0 / sul;
                    let sv0 = axis_v - nu * axis_v.dot(nu);
                    let sv = sv0 - su * sv0.dot(su);
                    let svl = sv.length();
                    if svl >= 1e-4 {
                        Some((nu, su, sv / svl))
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // World-stable surface frame for a rubber-stamp texture dab's sprite
        // read: the pattern lies IN the face plane but is oriented by the
        // face's own normal (cross with a world axis — the same triplanar
        // construction the shader's ambient uses) instead of the projected
        // brush/camera axes. So the pattern's direction tracks the surface as
        // it curves away and never reads like a screen-space projection of the
        // cursor. `up_ref` swaps to another world axis when the face already
        // runs ~parallel to world up (pole faces); degenerate faces yield None
        // and the read falls back to the on-face brush-plane point.
        let stable_frame: Option<(Vec3, Vec3)> = if facing_gate
            && pattern.is_none()
            && matches!(footprint, crate::brush::Footprint::Sprite { .. })
        {
            let n = (b - a).cross(c - a);
            let nl = n.length();
            if nl >= 1e-12 {
                let nu = n / nl;
                let up_ref = if nu.y.abs() > 0.999 { Vec3::X } else { Vec3::Y };
                let su0 = up_ref.cross(nu);
                let sul = su0.length();
                if sul >= 1e-4 {
                    let su = su0 / sul;
                    Some((su, nu.cross(su)))
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };
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
        if let crate::brush::Footprint::Rect { half_w, half_h } = footprint {
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
            if max_u < -half_w || min_u > half_w || max_v < -half_h || min_v > half_h {
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

        // Per-texel body for one horizontal line of this triangle. `row` is
        // the texture's row bytes for `tri_y` (already offset), `sa_row` that
        // row's stroke-alpha bytes (when tracked), and both are shared by the
        // serial and the rayon drivers below so the two paths stay
        // bit-identical. Every texel of a triangle is visited at most once and
        // a row only touches ITS OWN bytes, so rows are mutually exclusive.
        let stamp_tex_row = |row: &mut [u8],
                             mut sa_row: Option<&mut [u8]>,
                             tri_y: i32,
                             bx0: i32,
                             bx1: i32,
                             mut occ_pack: Option<(&mut Vec<u32>, &mut u32)>|
         -> Option<(u32, u32, u32, u32)> {
            let mut rect: Option<(u32, u32, u32, u32)> = None;
            for x in bx0..=bx1 {
                let uv = uv_from_texel(x as u32, tri_y as u32, tex_w, tex_h);
                let p = Vec2::new(uv.0, uv.1);
                let Some((bb0, bb1)) = uv_barycentric(p, t0, t1, t2) else {
                    continue;
                };
                let pos_3d = a + (b - a) * bb0 + (c - a) * bb1;

                // Footprint inclusion + edge falloff (disc / rect / square /
                // diamond / sprite) is evaluated by the shared pure engine.
                // `tu`/`tv` are the texel's position in the brush-local plane
                // (the plane through `center` perpendicular to the ray). The
                // analytic "touch everything" fallback collapses them to (0,0)
                // so every analytic shape treats each visited texel as under
                // the brush center; a sprite stamp is skipped there — it has
                // no plane position to sample.
                let rel = pos_3d - center;
                let (tu, tv) = if facing_gate {
                    (rel.dot(axis_u), rel.dot(axis_v))
                } else {
                    (0.0, 0.0)
                };
                // 3-D sphere gate for Square and Diamond: the axis-plane
                // projection of `rel` can be small even for texels on the far
                // side of a curved or complex mesh, so without this gate the
                // square/diamond footprint paints texels that are far away in
                // world space — causing bleed-through on the opposite side of
                // the mesh and apparent stretching near surface creases.
                // Round already uses `rel.length()` directly in `local`; Rect
                // and Sprite have their own bounds. Only Square and Diamond
                // need the extra clamp.
                if facing_gate
                    && matches!(
                        footprint.kind(),
                        crate::brush::FootprintKind::Square | crate::brush::FootprintKind::Diamond
                    )
                    && rel.length_squared() > footprint_radius * footprint_radius
                {
                    continue;
                }
                if !facing_gate && matches!(footprint.kind(), crate::brush::FootprintKind::Sprite) {
                    continue;
                }
                // The mask's frame: a ROUND window (the plain round dab and the
                // pattern-locked round window) measures the WORLD distance to the
                // brush center — exactly like the GPU cursor's sphere test — not
                // the projection into the brush-local plane. Across a hard crease
                // (cube edges) `local_surface_normal` blends the two faces and
                // tilts that plane, so projected distances compress and the stamp
                // would paint far past the cursor ring. A square / diamond window
                // must keep the plane coordinates — collapsing `local` to a
                // radial distance would erase the frame shape and paint a round
                // stroke, diverging from the cursor's (tu, tv) frame.
                let local = if !facing_gate {
                    Vec2::new(0.0, 0.0)
                } else if matches!(footprint, crate::brush::Footprint::Round { .. })
                    || (pattern.is_some()
                        && matches!(
                            footprint,
                            crate::brush::Footprint::Sprite {
                                window: crate::brush::Window::Round,
                                ..
                            }
                        ))
                {
                    Vec2::new(rel.length(), 0.0)
                } else if let Some((nu, su, sv)) = surf_frame {
                    let lift = rel.dot(nu);
                    let rh = rel - nu * lift;
                    Vec2::new(rh.dot(su), rh.dot(sv))
                } else {
                    Vec2::new(tu, tv)
                };
                // The rubber-stamp texture dab reads its sprite in each texel's
                // own surface plane (the same `surf_frame` a square / diamond
                // footprint uses), so the pattern lies flat along the surface
                // instead of squashing onto the brush axis plane. `face_local`
                // is that point; when the face is degenerate it falls back to
                // the brush-axis projection.
                let face_local_pt = match surf_frame {
                    Some((nu, su, sv)) => {
                        let lift = rel.dot(nu);
                        let rh = rel - nu * lift;
                        Vec2::new(rh.dot(su), rh.dot(sv))
                    }
                    None => Vec2::new(tu, tv),
                };
                // The pattern phase for the rubber-stamp texture dab's sprite
                // read: the face-plane position expressed in the world-stable
                // surface frame (see `stable_frame`), so the pattern lies flat
                // along the face and its direction follows the surface rather
                // than the brush/camera. Falls back to the on-face brush point
                // on degenerate faces.
                let surface_pt = match stable_frame {
                    Some((su, sv)) => Vec2::new(rel.dot(su), rel.dot(sv)),
                    None => face_local_pt,
                };
                // `local_coverage` classifies the point against the footprint,
                // then applies the falloff profile (sprite alpha IS the
                // coverage; hardness / eraser feather the normalized
                // distance). Pattern-locked stamps additionally read the sprite
                // at the anchored `pattern` frame (captured axes at stroke
                // start), scaled so the pattern's world size stays constant
                // even if the dab radius varies mid-stroke.
                let raw = match pattern {
                    Some(crate::brush::PatternAnchor::Uv {
                        x: anchor_x,
                        y: anchor_y,
                        ..
                    })
                    | Some(crate::brush::PatternAnchor::Canvas {
                        x: anchor_x,
                        y: anchor_y,
                        ..
                    }) => {
                        let anchored = Vec2::new(
                            (x as f32 - anchor_x) * anchored_uv_scale,
                            (tri_y as f32 - anchor_y) * anchored_uv_scale,
                        );
                        crate::brush::pattern_coverage(&profile, &footprint, local, anchored)
                    }
                    Some(crate::brush::PatternAnchor::Surface {
                        pos,
                        axis_u: anchor_u,
                        axis_v: anchor_v,
                        ..
                    }) if facing_gate => {
                        // Geodesic arc along the surface when the triangle was
                        // unfolded, anchor-plane chord otherwise.
                        let anchored = match tri_phases {
                            Some(ts) => {
                                let su =
                                    ts[0].x + (ts[1].x - ts[0].x) * bb0 + (ts[2].x - ts[0].x) * bb1;
                                let sv =
                                    ts[0].y + (ts[1].y - ts[0].y) * bb0 + (ts[2].y - ts[0].y) * bb1;
                                Vec2::new(su, sv) * anchored_surface_scale
                            }
                            None => {
                                let d = pos_3d - *pos;
                                Vec2::new(
                                    d.dot(*anchor_u) * anchored_surface_scale,
                                    d.dot(*anchor_v) * anchored_surface_scale,
                                )
                            }
                        };
                        crate::brush::pattern_coverage(&profile, &footprint, local, anchored)
                    }
                    // A rubber-stamp texture dab (no pattern anchor) paints
                    // through the same paint window the cursor previews, not a
                    // bare sprite box: the mask and the sprite read are split
                    // like a pattern dab whose anchor is the dab itself.
                    //   - SpriteBounds: the classic decal — the sprite's own UV
                    //     box (clamped) is the mask.
                    //   - Round window (the default texture brush): the mask is
                    //     the WORLD sphere `rel.length() <= r` — exactly the GPU
                    //     cursor's circle — so on a curved surface or crease the
                    //     mark never paints past the cursor the way a face-plane
                    //     disc stretches across the silhouette or balloons at the
                    //     pole; the sprite is read at the surface-plane phase so
                    //     the pattern itself stays flat along the mesh.
                    //   - Square / Diamond windows: the mask and read share the
                    //     surface-plane frame the window mask describes.
                    None => match &footprint {
                        crate::brush::Footprint::Sprite {
                            window: crate::brush::Window::SpriteBounds,
                            ..
                        } => crate::brush::local_coverage(&profile, &footprint, face_local_pt),
                        crate::brush::Footprint::Sprite {
                            window: crate::brush::Window::Round,
                            ..
                        } => crate::brush::pattern_coverage(
                            &profile,
                            &footprint,
                            Vec2::new(rel.length(), 0.0),
                            surface_pt,
                        ),
                        crate::brush::Footprint::Sprite {
                            window: crate::brush::Window::Square | crate::brush::Window::Diamond,
                            ..
                        } => crate::brush::pattern_coverage(
                            &profile,
                            &footprint,
                            face_local_pt,
                            surface_pt,
                        ),
                        _ => crate::brush::local_coverage(&profile, &footprint, local),
                    },
                    _ => crate::brush::local_coverage(&profile, &footprint, local),
                } * angle_factor;
                if raw <= 0.0 {
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
                        // The occlusion grid needs per-query scratch along the
                        // shared `occ_visited`/`occ_qid`; these arrive through
                        // `occ_pack` so the row worker stays a plain `Fn` and the
                        // row-parallel path above (occ_grid == None) can hand it
                        // nothing at all. It is always `Some` when this branch
                        // runs; `None` is treated as unblocked.
                        let blocked = match occ_pack.as_mut() {
                            Some((v, q)) => occ_grid
                                .as_ref()
                                .and_then(|g| g.nearest_before(eye, to_point / dist, dist, v, q))
                                .is_some_and(|t| t < dist - occ_eps),
                            None => false,
                        };
                        if blocked {
                            continue;
                        }
                    }
                }

                let lx = x as usize * 4;
                // Non-accumulative stroke blend: the stroke buffer holds the
                // MAX of `min(opacity, cover)` per texel, exactly like
                // compositing the final buffer once. The exact source-over step
                // (new_alpha − current)/(1 − current) reaches that value when
                // blended incrementally, so the live preview already matches.
                let mut effective_opacity = brush.opacity * raw;
                let mut new_stroke_alpha = 0u8;
                if !brush.accumulate || pattern.is_some() {
                    if let Some(sa) = sa_row.as_mut() {
                        let current_stroke_alpha = sa[x as usize] as f32 / 255.0;
                        let new_alpha = raw.min(brush.opacity).max(current_stroke_alpha);
                        if new_alpha <= current_stroke_alpha {
                            continue;
                        }
                        effective_opacity =
                            (new_alpha - current_stroke_alpha) / (1.0 - current_stroke_alpha);
                        new_stroke_alpha = (new_alpha * 255.0).round() as u8;
                    }
                }
                let mut px = [row[lx], row[lx + 1], row[lx + 2], row[lx + 3]];
                match brush.mode {
                    StampMode::Paint => blend_pixel(&mut px, brush.color, effective_opacity),
                    StampMode::Erase => erase_pixel(&mut px, effective_opacity),
                }
                row[lx..lx + 4].copy_from_slice(&px);
                if !brush.accumulate || pattern.is_some() {
                    if let Some(sa) = sa_row.as_mut() {
                        sa[x as usize] = new_stroke_alpha;
                    }
                }
                let (ux, uy) = (x as u32, tri_y as u32);
                match rect {
                    Some(d) => rect = Some((d.0.min(ux), d.1.min(uy), d.2.max(ux), d.3.max(uy))),
                    None => rect = Some((ux, uy, ux, uy)),
                }
            }
            rect
        };

        // Run the triangle's rows either in parallel (no occlusion grid — the
        // convex fast path — and enough texels to amortize the dispatch) or
        // serially row by row. Both call the same row worker, so the pixels
        // are identical either way; the triangle iteration itself stays serial
        // to preserve the exact seam-overlap ordering between triangles.
        let n_rows = y1 - y0 + 1;
        let area = n_rows as i64 * (x1 - x0 + 1) as i64;
        if occ_grid.is_none() && n_rows >= 4 && area >= 16 * 1024 {
            let row_bytes = tex_w as usize * 4;
            let tex_it = tex.rgba.par_chunks_exact_mut(row_bytes);
            if let Some(sa) = stroke_alpha.as_deref_mut() {
                let sa_it = sa.par_chunks_exact_mut(tex_w as usize);
                let results: Vec<Option<(u32, u32, u32, u32)>> = tex_it
                    .zip(sa_it)
                    .enumerate()
                    .map(|(y, (row, sa_row))| {
                        if (y as i32) < y0 || (y as i32) > y1 {
                            return None;
                        }
                        stamp_tex_row(row, Some(sa_row), y as i32, x0, x1, None)
                    })
                    .collect();
                for r in results.into_iter().flatten() {
                    dirty.0 = dirty.0.min(r.0);
                    dirty.1 = dirty.1.min(r.1);
                    dirty.2 = dirty.2.max(r.2);
                    dirty.3 = dirty.3.max(r.3);
                }
            } else {
                let results: Vec<Option<(u32, u32, u32, u32)>> = tex_it
                    .enumerate()
                    .map(|(y, row)| {
                        if (y as i32) < y0 || (y as i32) > y1 {
                            return None;
                        }
                        stamp_tex_row(row, None, y as i32, x0, x1, None)
                    })
                    .collect();
                for r in results.into_iter().flatten() {
                    dirty.0 = dirty.0.min(r.0);
                    dirty.1 = dirty.1.min(r.1);
                    dirty.2 = dirty.2.max(r.2);
                    dirty.3 = dirty.3.max(r.3);
                }
            }
        } else {
            for y in y0..=y1 {
                let rs = (y as usize) * tex_w as usize;
                let row = &mut tex.rgba[rs * 4..(rs + tex_w as usize) * 4];
                let sa_row = stroke_alpha
                    .as_deref_mut()
                    .map(|s| &mut s[rs..rs + tex_w as usize]);
                if let Some(r) = stamp_tex_row(
                    row,
                    sa_row,
                    y,
                    x0,
                    x1,
                    Some((&mut occ_visited, &mut occ_qid)),
                ) {
                    dirty.0 = dirty.0.min(r.0);
                    dirty.1 = dirty.1.min(r.1);
                    dirty.2 = dirty.2.max(r.2);
                    dirty.3 = dirty.3.max(r.3);
                }
            }
        }
    }
    if dirty.0 <= dirty.2 {
        // Dilate painted pixels 1 texel outward into any completely
        // transparent neighbours within the dirty rect. This fills the
        // sub-pixel seam gaps that appear on rotated UV islands (where a
        // diagonal triangle edge passes through pixels whose center lies
        // just outside the triangle). All professional texture painters
        // apply this step; it does not affect interior texels (they are
        // already painted) and only touches alpha-0 neighbours.
        // Skip for Erase mode: the eraser intentionally zeros alpha, and
        // dilation would immediately refill those pixels from neighbours.
        if brush.mode == StampMode::Paint {
            dilate_seams(&mut mesh.layers[layer_idx].texture, dirty);
        }
        mesh.dirty = Some(dirty);
    }
}

/// Applies a brush stamp directly in 2D UV space onto `texture` (the 2D
/// texture preview) without any 3D mesh involvement. `center_uv` is the brush
/// center in [0,1] UV coordinates; `radius_px` is the footprint radius in
/// texels (the preview scales screen pixels 1:1 with texels when zoomed to
/// fit). `shape` drives the footprint (rect tools pass [`BrushShape::Square`]);
/// Stamps one dab of `brush` into a 2D texture at `center_uv` with radius
/// `radius_px` (texture texels). `brush` supplies the footprint kind (with its
/// optional sprite + transform), the falloff profile, color, opacity, mode and
/// accumulate flag; the round/square/diamond/sprite semantics mirror the 3D
/// stamps — both share `brush::local_coverage`. `dirty` is expanded to the
/// touched texel rect so the caller can do a region upload instead of a full
/// one.
/// If `brush.accumulate` is false, `stroke_alpha` (when Some) tracks the
/// maximum alpha this stroke has applied per texel, capping further dabs
/// within the same stroke so opacity doesn't stack beyond `brush.opacity`.
/// Pattern-aligned strokes *always* honor it, so the anchored texture is a
/// flat replace: overlapping dabs can never "fill itself up".
/// Stamps one horizontal line of a 2D dab. `row` is the texture's row bytes
/// for `y` (already offset by the caller); `sa_row` is that row's stroke-alpha
/// bytes when the stroke tracks non-accumulating coverage. Returns the texels
/// the row actually painted (a unit-height rect) or `None` when it painted
/// nothing. This is the shared per-row body of [`stamp_2d`], driven either
/// serially or with rayon: because each texel is visited at most once per dab
/// and a row only touches ITS OWN bytes, rows are mutually exclusive and the
/// two drivers are bit-identical.
#[allow(clippy::too_many_arguments)]
fn stamp_2d_row(
    row: &mut [u8],
    mut sa_row: Option<&mut [u8]>,
    y: i32,
    x0: i32,
    x1: i32,
    ww: i32,
    cx: f32,
    cy: f32,
    brush: &crate::brush::Brush,
    footprint: &crate::brush::Footprint,
    profile: &crate::brush::DabProfile,
    anchored_frame: Option<(f32, f32, f32)>,
    has_pattern: bool,
) -> Option<(u32, u32, u32, u32)> {
    let mut rect: Option<(u32, u32, u32, u32)> = None;
    for x in x0..=x1 {
        if x < 0 || x >= ww {
            continue;
        }
        let dx = x as f32 - cx;
        let dy = y as f32 - cy;
        // Brush-local plane with +y = canvas-up: the footprint's sprite
        // sampling treats +y as the sprite's top, so a texel below the dab
        // center must arrive here with a negative y or the stamped sprite
        // lands vertically mirrored against the 2D cursor (and rotates the
        // wrong way at 0deg/180deg).
        let local = Vec2::new(dx, -dy);
        let cover = match anchored_frame {
            Some((anchor_x, anchor_y, scale)) => {
                // The texture phase is read straight from the texel position
                // minus the stroke-start anchor (never the moving dab center),
                // so every dab paints the same stationary square-tiled grid.
                // The radius ratio cancels the dab-relative sampling scale in
                // `alpha_at_wrapped` — the pattern's tile size in texels is
                // fixed at `2·anchor_r` for the whole stroke — while the
                // dab-local soft mask stays a pure distance reveal.
                let anchored = Vec2::new(
                    (x as f32 - anchor_x) * scale,
                    -(y as f32 - anchor_y) * scale,
                );
                crate::brush::pattern_coverage(profile, footprint, local, anchored)
            }
            None => crate::brush::local_coverage(profile, footprint, local),
        };
        if cover <= 0.0 {
            continue;
        }
        let lx = (x as usize) * 4;
        let texel_idx = x as usize;
        // Non-accumulative stroke blend: `stroke_buffer` keeps the MAX of
        // `min(opacity, cover)` per texel; the exact source-over step
        // reaches compositing the final buffer once, so live preview
        // already equals the finished stroke.
        let mut effective_opacity = brush.opacity * cover;
        let mut new_stroke_alpha = 0u8;
        if !brush.accumulate || has_pattern {
            if let Some(sa) = sa_row.as_mut() {
                let current_stroke_alpha = sa[texel_idx] as f32 / 255.0;
                let new_alpha = cover.min(brush.opacity).max(current_stroke_alpha);
                if new_alpha <= current_stroke_alpha {
                    continue;
                }
                effective_opacity =
                    (new_alpha - current_stroke_alpha) / (1.0 - current_stroke_alpha);
                new_stroke_alpha = (new_alpha * 255.0).round() as u8;
            }
        }
        let mut px = [row[lx], row[lx + 1], row[lx + 2], row[lx + 3]];
        match brush.mode {
            StampMode::Paint => blend_pixel(&mut px, brush.color, effective_opacity),
            StampMode::Erase => erase_pixel(&mut px, effective_opacity),
        }
        row[lx..lx + 4].copy_from_slice(&px);
        if !brush.accumulate || has_pattern {
            if let Some(sa) = sa_row.as_mut() {
                sa[texel_idx] = new_stroke_alpha;
            }
        }
        let (ux, uy) = (x as u32, y as u32);
        match rect {
            Some(d) => rect = Some((d.0.min(ux), d.1.min(uy), d.2.max(ux), d.3.max(uy))),
            None => rect = Some((ux, uy, ux, uy)),
        }
    }
    rect
}

#[allow(clippy::too_many_arguments)]
pub fn stamp_2d(
    texture: &mut TextureData,
    center_uv: (f32, f32),
    radius_px: f32,
    brush: &crate::brush::Brush,
    dirty: &mut Option<(u32, u32, u32, u32)>,
    mut stroke_alpha: Option<&mut [u8]>,
    pattern: Option<&crate::brush::PatternAnchor>,
) {
    let (w, h) = (texture.width as i32, texture.height as i32);
    if w <= 0 || h <= 0 || brush.opacity <= 0.0 || radius_px <= 0.0 {
        return;
    }
    let (cx, cy) = (center_uv.0 * w as f32, center_uv.1 * h as f32);
    let r = radius_px;

    let x0 = (cx - r).floor() as i32;
    let x1 = (cx + r).ceil() as i32;
    let y0 = (cy - r).floor() as i32;
    let y1 = (cy + r).ceil() as i32;

    // Resolve the brush state into the shared pure footprint + profile once
    // per dab; the per-texel loop then delegates everything to
    // `brush::local_coverage` (the same evaluator the 3D stamp uses).
    // `Brush::footprint`/`Brush::falloff` hold the fallback rules
    // (sprite-without-image → round, sprite alpha IS the coverage, eraser
    // feathers) so both stampers share them.
    let footprint = match pattern {
        Some(_) => brush.pattern_footprint(r),
        None => brush.footprint(r),
    };
    let profile = brush.falloff();
    let has_pattern = pattern.is_some();

    // Absolute world-space / UV-space pattern frame: the anchor and the scale
    // ratio are constant for the whole dab, so they are resolved once before
    // the loop (the per-texel work is just the two anchored coordinates).
    // Surface anchors (3D-only) carry no constant anchor x/y — they are
    // resolved per texel from the world position instead.
    let anchored_frame: Option<(f32, f32, f32)> = match pattern {
        Some(crate::brush::PatternAnchor::Uv { x, y, radius })
        | Some(crate::brush::PatternAnchor::Canvas { x, y, radius }) => {
            Some((*x, *y, brush.texture_scale * r / radius.max(1e-6)))
        }
        _ => None,
    };

    // Clamp the row range to the canvas (the per-row body still bounds-checks
    // x, matching the original loop).
    let ay0 = y0.max(0).min(h - 1);
    let ay1 = y1.max(0).min(h - 1);
    if ay0 > ay1 {
        return;
    }
    let ww = w as usize;
    let row_bytes = ww * 4;
    let n_rows = (ay1 - ay0 + 1) as usize;
    let xl = x0.max(0).min(w - 1);
    let xr = x1.max(0).min(w - 1);
    let area = (xr - xl + 1) as usize * n_rows;

    let merge_dirty =
        |dirty: &mut Option<(u32, u32, u32, u32)>, r: (u32, u32, u32, u32)| match dirty {
            Some(d) => {
                d.0 = d.0.min(r.0);
                d.1 = d.1.min(r.1);
                d.2 = d.2.max(r.2);
                d.3 = d.3.max(r.3);
            }
            None => *dirty = Some(r),
        };

    if area >= 16 * 1024 && n_rows >= 4 {
        // Large dab: stamp rows in parallel. Each row writes only its own
        // texture row bytes and its own stroke-alpha row bytes, so the
        // rows are mutually exclusive (rayon's par_chunks guarantees disjoint
        // slices) and the result is bit-identical to the serial pass. Row
        // order is irrelevant because no two rows touch the same texel.
        let tex_iter = texture.rgba.par_chunks_exact_mut(row_bytes);
        if let Some(sa) = stroke_alpha {
            let sa_iter = sa.par_chunks_exact_mut(ww);
            let results: Vec<Option<(u32, u32, u32, u32)>> = tex_iter
                .zip(sa_iter)
                .enumerate()
                .map(|(y, (row, sa_row))| {
                    if (y as i32) < ay0 || (y as i32) > ay1 {
                        return None;
                    }
                    stamp_2d_row(
                        row,
                        Some(sa_row),
                        y as i32,
                        x0,
                        x1,
                        w,
                        cx,
                        cy,
                        brush,
                        &footprint,
                        &profile,
                        anchored_frame,
                        has_pattern,
                    )
                })
                .collect();
            for r in results.into_iter().flatten() {
                merge_dirty(dirty, r);
            }
        } else {
            let results: Vec<Option<(u32, u32, u32, u32)>> = tex_iter
                .enumerate()
                .map(|(y, row)| {
                    if (y as i32) < ay0 || (y as i32) > ay1 {
                        return None;
                    }
                    stamp_2d_row(
                        row,
                        None,
                        y as i32,
                        x0,
                        x1,
                        w,
                        cx,
                        cy,
                        brush,
                        &footprint,
                        &profile,
                        anchored_frame,
                        has_pattern,
                    )
                })
                .collect();
            for r in results.into_iter().flatten() {
                merge_dirty(dirty, r);
            }
        }
        return;
    }

    for y in ay0..=ay1 {
        let rs = (y as usize) * row_bytes;
        let tex_row = &mut texture.rgba[rs..rs + row_bytes];
        let sa_row = stroke_alpha
            .as_deref_mut()
            .map(|s| &mut s[(y as usize) * ww..(y as usize) * ww + ww]);
        if let Some(r) = stamp_2d_row(
            tex_row,
            sa_row,
            y,
            x0,
            x1,
            w,
            cx,
            cy,
            brush,
            &footprint,
            &profile,
            anchored_frame,
            has_pattern,
        ) {
            merge_dirty(dirty, r);
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
        for &idx in tri_idx {
            let idx = idx as usize;
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
        // Same seam-dilation pass as stamp_texels: fill the 1-pixel gap that
        // appears along diagonal UV triangle edges after a flood fill.
        let fill_dirty = (dmin_x, dmin_y, dmax_x, dmax_y);
        let li = mesh.active_layer.min(mesh.layers.len().saturating_sub(1));
        dilate_seams(&mut mesh.layers[li].texture, fill_dirty);
        let dirty = mesh.dirty.unwrap_or((w as u32, h as u32, 0, 0));
        mesh.dirty = Some((
            dirty.0.min(dmin_x),
            dirty.1.min(dmin_y),
            dirty.2.max(dmax_x),
            dirty.3.max(dmax_y),
        ));
    }
}

/// Flood-fills every texel covered by triangles in the same connected 3D mesh
/// component (welded geometry, ignoring UV seams) as `seed_triangle`.
pub fn fill_region_mesh(mesh: &mut MeshData, seed_triangle: usize, color: [u8; 4], opacity: f32) {
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
    let comps = triangle_components_mesh(positions, indices);
    if seed_triangle >= comps.len() {
        return;
    }
    let seed = comps[seed_triangle];

    let mut min_u = f32::MAX;
    let mut max_u = f32::MIN;
    let mut min_v = f32::MAX;
    let mut max_v = f32::MIN;
    for (tri, tri_idx) in indices.chunks_exact(3).enumerate() {
        if comps[tri] != seed {
            continue;
        }
        for &idx in tri_idx {
            let idx = idx as usize;
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
        let fill_dirty = (dmin_x, dmin_y, dmax_x, dmax_y);
        let li = mesh.active_layer.min(mesh.layers.len().saturating_sub(1));
        dilate_seams(&mut mesh.layers[li].texture, fill_dirty);
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
    bvh: Option<&TriangleBvh>,
) -> Option<Vec3> {
    let mut sum = Vec3::ZERO;
    let mut count = 0u32;
    let mut visit = |tri: usize| {
        let t = tri * 3;
        let (i0, i1, i2) = (
            indices[t] as usize,
            indices[t + 1] as usize,
            indices[t + 2] as usize,
        );
        let (a, b, c) = (positions[i0], positions[i1], positions[i2]);
        if dist_point_to_triangle(center, a, b, c) > radius {
            return;
        }
        let n = (b - a).cross(c - a);
        let l = n.length();
        if l > 1e-12 {
            sum += n;
            count += 1;
        }
    };
    match bvh {
        // A stroke owns a BVH build once and hands it down; the normal then
        // only considers triangles the brush sphere can reach instead of
        // scanning the whole mesh per dab.
        Some(bvh) => {
            let mut cand = Vec::new();
            bvh.query(center, radius, &mut cand);
            for &tri in &cand {
                visit(tri as usize);
            }
        }
        None => {
            for (tri, _) in indices.chunks_exact(3).enumerate() {
                visit(tri);
            }
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
    bvh: Option<&TriangleBvh>,
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
    match local_surface_normal(positions, indices, center, radius, bvh) {
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

// ---------------------------------------------------------------------------
// Surface-anchored geodesic pattern phase
// ---------------------------------------------------------------------------

/// The aligned texture pattern's phase for the 3D surface stamp, measured
/// *along the surface* instead of through the anchor plane.
///
/// A pattern-locked texture stroke tiles a world-uniform grid anchored at the
/// click point. Naively the phase of a texel is its world chord from the
/// anchor projected onto the anchor's tangent axes — that is a flat,
/// camera-facing plane. On a curved surface such a chord cuts through depth:
/// the pattern compresses on the near bulge and, worse, the far side of the
/// body folds back onto the near side (front and back of a sphere land on the
/// same phase), so curved forward/backwards surfaces paint wrong.
///
/// `SurfaceUnwrap` instead unfolds the mesh patch around the anchor, triangle
/// by triangle, conserving every world edge length (the discrete exponential
/// map): each new triangle's far vertex is placed at the two-circle
/// intersection of its world edge lengths, on the far side of the shared edge.
/// The phase is therefore the *arc* the pattern travels over the surface — it
/// wraps curvature instead of projecting through it, never folds, and keeps
/// the tile world-uniform everywhere. On a locally flat patch the unfold is
/// the same affine world-plane map, so flat surfaces stay bit-identical to the
/// old chord behaviour.
#[derive(Clone)]
pub struct SurfaceUnwrap {
    /// Phase at each visited vertex; `u`/`v` are world arc-lengths along the
    /// anchor's U/V axes, before the dab-radius scaling.
    vertex_phases: HashMap<u32, Vec2>,
    /// The three vertex phases per visited triangle, in triangle index order.
    tri_phases: HashMap<u32, [Vec2; 3]>,
    /// Vertex-index span the patch covers (bounds the GPU cursor upload);
    /// inspected by the tests and the `span` helper.
    #[cfg_attr(not(test), allow(dead_code))]
    min_vertex: u32,
    #[cfg_attr(not(test), allow(dead_code))]
    max_vertex: u32,
    /// World radius the patch was unfolded within; texels in triangles beyond
    /// it fall back to the old anchor-plane chord.
    radius: f32,
}

impl SurfaceUnwrap {
    /// The world radius the patch was unfolded within; texels in triangles
    /// beyond it fall back to the old anchor-plane chord.
    #[inline]
    pub fn radius(&self) -> f32 {
        self.radius
    }

    /// The phases of a triangle's three vertices, in index order, if the
    /// triangle lies in the unfolded patch.
    #[inline]
    pub fn tri(&self, tri: usize) -> Option<[Vec2; 3]> {
        self.tri_phases.get(&(tri as u32)).copied()
    }

    /// The phase at one vertex (used to fill the cursor-preview buffer).
    #[cfg_attr(not(test), allow(dead_code))] // exercised by the unfold tests
    #[inline]
    pub fn vertex_phase(&self, v: u32) -> Option<Vec2> {
        self.vertex_phases.get(&v).copied()
    }

    /// The vertex-index span covered by the patch.
    #[cfg_attr(not(test), allow(dead_code))] // exercised by the unfold tests
    #[inline]
    pub fn span(&self) -> Option<(u32, u32)> {
        if self.vertex_phases.is_empty() {
            None
        } else {
            Some((self.min_vertex, self.max_vertex))
        }
    }

    /// Full-length per-vertex phase data for the GPU cursor: one `(u, v)` pair
    /// (two `f32`) per mesh vertex, zero (phase 0) for vertices the unfold
    /// never reached — the fragment shader's `phase_active` flag gates which
    /// vertices actually read the field, so untouched regions fall back to the
    /// chord instead of sampling the pattern origin.
    pub fn upload(&self, vertex_count: usize) -> Vec<f32> {
        let mut data = vec![0.0f32; vertex_count.saturating_mul(2)];
        for (&v, &ph) in &self.vertex_phases {
            let i = v as usize * 2;
            if i + 1 < data.len() {
                data[i] = ph.x;
                data[i + 1] = ph.y;
            }
        }
        data
    }
}

/// Places the far vertex of a neighbor triangle in the phase plane, conserving
/// the world edge lengths to the shared edge's two vertices (two-circle
/// intersection), opened on the far side of the shared edge from `pc` — the
/// phase of the triangle we came from's opposite vertex. This is the discrete
/// exponential map: the surface arc from `a` to the new vertex is preserved,
/// so curvature accumulates as the patch unfolds instead of collapsing into
/// the anchor plane. Degenerate edges fall back to the anchor-plane chord.
///
/// Which of the two symmetric intersections to use is chosen from the WORLD
/// side of the new vertex about the shared edge (measured in the current
/// face's plane). Comparing the two near-identical phase positions against
/// `pc`'s phase instead is fragile: at a crease — a flat wall attached to a
/// curved region — `pc`'s phase compresses onto the edge line, so that test
/// becomes a coin flip and mirrors the wall behind it.
#[allow(clippy::too_many_arguments)] // phase triple + position pair + anchor frame
fn unfold_vertex(
    pa: Vec2,
    pb: Vec2,
    pc: Vec2,
    pos_w: Vec3,
    pos_a: Vec3,
    pos_b: Vec3,
    pos_c: Vec3,
    anchor: Vec3,
    axis_u: Vec3,
    axis_v: Vec3,
) -> Vec2 {
    let dir = pb - pa;
    let len_ab = dir.length();
    if len_ab < 1e-9 {
        return Vec2::new((pos_w - anchor).dot(axis_u), (pos_w - anchor).dot(axis_v));
    }
    let d_aw = (pos_w - pos_a).length();
    let d_bw = (pos_w - pos_b).length();
    let base = (d_aw * d_aw - d_bw * d_bw + len_ab * len_ab) / (2.0 * len_ab);
    let h = (d_aw * d_aw - base * base).max(0.0).sqrt();
    let u = dir / len_ab;
    let side = Vec2::new(-u.y, u.x);
    let q1 = pa + u * base + side * h;
    let q2 = pa + u * base - side * h;
    // World side of `w` (and of `c`, the current face's far vertex) across the
    // shared edge, measured by the in-plane perpendicular of the edge. When
    // both are unambiguous, `w` follows `c`'s WORLD side: opposite a manifold
    // neighbor, same side where the mesh genuinely folds back. Only when the
    // world sides collapse (w on the edge line) do we fall back to the
    // historical phase-plane rule (opposite of `pc`).
    let world_n = (pos_c - pos_a).cross(pos_b - pos_a);
    let perp = world_n.cross(pos_b - pos_a).normalize_or_zero();
    let (sw, sc) = if perp.length_squared() > 1e-12 {
        (
            ((pos_w - pos_a).dot(perp)).signum() as i8,
            ((pos_c - pos_a).dot(perp)).signum() as i8,
        )
    } else {
        (0, 0)
    };
    let s_ref = dir.x * (pc.y - pa.y) - dir.y * (pc.x - pa.x);
    let s1 = dir.x * (q1.y - pa.y) - dir.y * (q1.x - pa.x);
    // Target phase side for `w`. A manifold neighbor puts `w` on the far side
    // of the shared edge from `c`, so it unfolds opposite `pc`; only when the
    // world sides agree does the region genuinely fold back onto `c`'s side.
    // Without a measurable world geometry we keep the historical rule
    // (opposite of `pc`).
    let target = match (sw, sc) {
        (wv, cv) if wv != 0 && cv != 0 && wv != cv => -s_ref.signum(),
        (wv, cv) if wv != 0 && cv != 0 => s_ref.signum(),
        _ => -s_ref.signum(),
    };
    if target == s1.signum() {
        q1
    } else {
        q2
    }
}

/// Builds a surface-following anchor unwrap starting at `anchor_tri` — the
/// triangle that owns the hit point — unfolding every triangle within `radius`
/// of `anchor`. Returns `None` when the seed triangle is missing. Texels in
/// triangles outside the unfolded patch (dents, far-flung dabs) keep the
/// anchor-plane chord via the stamp's fallback.
pub fn surface_unwrap(
    positions: &[Vec3],
    indices: &[u32],
    anchor: Vec3,
    axis_u: Vec3,
    axis_v: Vec3,
    anchor_tri: usize,
    radius: f32,
) -> Option<SurfaceUnwrap> {
    let n_tris = indices.len() / 3;
    if anchor_tri >= n_tris || radius <= 0.0 {
        return None;
    }
    // Candidate patch: triangles within `radius` of the anchor. An infinite
    // radius unfolds the whole connected mesh (every dab of a stroke shares
    // one field, however far it drags); a finite radius keeps the unfold local
    // to the brush, e.g. the cheap hover-preview unwrap.
    let whole_mesh = radius.is_infinite();
    let mut tri_cand: Vec<u32> = Vec::with_capacity(if whole_mesh { n_tris } else { 0 });
    for (ti, tri) in indices.chunks_exact(3).enumerate() {
        if whole_mesh {
            tri_cand.push(ti as u32);
            continue;
        }
        let (a, b, c) = (
            positions[tri[0] as usize],
            positions[tri[1] as usize],
            positions[tri[2] as usize],
        );
        if dist_point_to_triangle(anchor, a, b, c) <= radius {
            tri_cand.push(ti as u32);
        }
    }
    if !tri_cand.contains(&(anchor_tri as u32)) {
        tri_cand.push(anchor_tri as u32);
    }
    // Undirected edge -> incident candidate triangles (a list tolerates
    // non-manifold edges).
    let mut edge_tris: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    for &ti in &tri_cand {
        let s = ti as usize * 3;
        let (i0, i1, i2) = (indices[s], indices[s + 1], indices[s + 2]);
        for (a, b) in [(i0, i1), (i1, i2), (i2, i0)] {
            let key = if a < b { (a, b) } else { (b, a) };
            edge_tris.entry(key).or_default().push(ti);
        }
    }

    let mut vertex_phases: HashMap<u32, Vec2> = HashMap::new();
    let mut tri_phases: HashMap<u32, [Vec2; 3]> = HashMap::new();
    let mut visited: HashSet<u32> = HashSet::new();
    let mut queue: Vec<u32> = Vec::new();

    // The anchor lies on `anchor_tri`, so that triangle's phase is exact: the
    // in-plane displacement from the anchor along the captured axes.
    let chord = |v: Vec3| Vec2::new((v - anchor).dot(axis_u), (v - anchor).dot(axis_v));
    {
        let s = anchor_tri * 3;
        let idxs = [indices[s], indices[s + 1], indices[s + 2]];
        let p = [
            chord(positions[idxs[0] as usize]),
            chord(positions[idxs[1] as usize]),
            chord(positions[idxs[2] as usize]),
        ];
        for (i, ph) in idxs.into_iter().zip(p) {
            vertex_phases.insert(i, ph);
        }
        tri_phases.insert(anchor_tri as u32, p);
        visited.insert(anchor_tri as u32);
        queue.push(anchor_tri as u32);
    }

    let mut head = 0usize;
    while head < queue.len() {
        let ti = queue[head];
        head += 1;
        let s = ti as usize * 3;
        let idxs = [indices[s], indices[s + 1], indices[s + 2]];
        // Each edge (idxs[ea], idxs[eb]) lies opposite vertex idxs[ec].
        for (ea, eb, ec) in [(0usize, 1usize, 2usize), (1, 2, 0), (2, 0, 1)] {
            let (a, b, c) = (idxs[ea], idxs[eb], idxs[ec]);
            let key = if a < b { (a, b) } else { (b, a) };
            let Some(neighbors) = edge_tris.get(&key) else {
                continue;
            };
            let (Some(pa), Some(pb), Some(pc)) = (
                vertex_phases.get(&a).copied(),
                vertex_phases.get(&b).copied(),
                vertex_phases.get(&c).copied(),
            ) else {
                continue;
            };
            for &nt in neighbors {
                if visited.contains(&nt) {
                    continue;
                }
                let ns = nt as usize * 3;
                let (n0, n1, n2) = (indices[ns], indices[ns + 1], indices[ns + 2]);
                // `nt` shares the (a, b) edge; its far vertex is whichever of
                // its three is neither a nor b.
                let Some(w) = [n0, n1, n2].into_iter().find(|&v| v != a && v != b) else {
                    continue; // degenerate triangle (two identical indices)
                };
                // A vertex's phase is assigned exactly once. On a closed
                // curved surface the unfold eventually wraps around and meets
                // itself; recomputing/overwriting an already-placed vertex
                // would make incident triangles disagree at that vertex, so
                // the stamp's barycentric phase would jump across the shared
                // edge — a visible seam. Keeping the first assignment instead
                // leaves the field single-valued (continuous everywhere); the
                // loop-closing triangles shear rather than tear.
                vertex_phases.entry(w).or_insert_with(|| {
                    unfold_vertex(
                        pa,
                        pb,
                        pc,
                        positions[w as usize],
                        positions[a as usize],
                        positions[b as usize],
                        positions[c as usize],
                        anchor,
                        axis_u,
                        axis_v,
                    )
                });
                let np = [vertex_phases[&n0], vertex_phases[&n1], vertex_phases[&n2]];
                tri_phases.insert(nt, np);
                visited.insert(nt);
                queue.push(nt);
            }
        }
    }

    // Triangles the geodesic flood never reached lie on mesh components that
    // are disconnected from the seed part (cracked seams, duplicated strips,
    // separate objects). The old anchor-plane chord stretched or collapsed
    // them: a panel tilted relative to the click plane was projected onto it,
    // smearing the pattern into a line instead of the dab. Develop each such
    // component isometrically in its own plane, aligned to the anchor axes
    // through its first triangle's centroid, so a flat panel paints a
    // world-uniform, non-stretched pattern. Only components inside the patch
    // radius matter for a finite unfold; their phases feed the stamp's
    // per-triangle lookup, which then never falls back to the chord.
    let candidate: Option<Vec<bool>> = if whole_mesh {
        None
    } else {
        let mut c = vec![false; n_tris];
        for &t in &tri_cand {
            c[t as usize] = true;
        }
        Some(c)
    };
    let mut closed: HashSet<u32> = visited;
    for ti in 0..n_tris as u32 {
        if closed.contains(&ti) {
            continue;
        }
        if let Some(c) = &candidate {
            if !c[ti as usize] {
                continue;
            }
        }
        let mut component = Vec::new();
        let mut queue = vec![ti];
        closed.insert(ti);
        while let Some(t) = queue.pop() {
            component.push(t);
            let s = t as usize * 3;
            let (i0, i1, i2) = (indices[s], indices[s + 1], indices[s + 2]);
            for (a, b) in [(i0, i1), (i1, i2), (i2, i0)] {
                let key = if a < b { (a, b) } else { (b, a) };
                if let Some(nbs) = edge_tris.get(&key) {
                    for &nt in nbs {
                        if closed.insert(nt) {
                            queue.push(nt);
                        }
                    }
                }
            }
        }
        // The component's first triangle's plane, spun so its U axis is the
        // anchor U axis projected onto the plane (V = plane normal × U). This
        // keeps the pattern's world orientation on flat panels while staying
        // isometric (no stretch, no shear).
        let s0 = component[0] as usize * 3;
        let (a, b, c) = (
            positions[indices[s0] as usize],
            positions[indices[s0 + 1] as usize],
            positions[indices[s0 + 2] as usize],
        );
        let n = (b - a).cross(c - a);
        let n = if n.length_squared() > 1e-12 {
            n.normalize()
        } else {
            axis_u.cross(axis_v).normalize_or_zero()
        };
        let fu = axis_u - n * axis_u.dot(n);
        let (fu, fv) = if fu.length_squared() > 1e-12 {
            (fu.normalize(), n.cross(fu).normalize())
        } else {
            let fv = axis_v - n * axis_v.dot(n);
            if fv.length_squared() > 1e-12 {
                (n.cross(fv).normalize(), fv.normalize())
            } else {
                (axis_u.normalize_or_zero(), axis_v.normalize_or_zero())
            }
        };
        let origin = (a + b + c) / 3.0;
        let offset = Vec2::new((origin - anchor).dot(axis_u), (origin - anchor).dot(axis_v));
        for &t in &component {
            let s = t as usize * 3;
            let idxs = [indices[s], indices[s + 1], indices[s + 2]];
            let mut nps = [Vec2::ZERO; 3];
            for (k, &v) in idxs.iter().enumerate() {
                let rel = positions[v as usize] - origin;
                let ph = offset + Vec2::new(rel.dot(fu), rel.dot(fv));
                vertex_phases.entry(v).or_insert(ph);
                nps[k] = ph;
            }
            tri_phases.insert(t, nps);
        }
    }

    let (mut min_vertex, mut max_vertex) = (u32::MAX, 0u32);
    for &v in vertex_phases.keys() {
        min_vertex = min_vertex.min(v);
        max_vertex = max_vertex.max(v);
    }

    Some(SurfaceUnwrap {
        vertex_phases,
        tri_phases,
        min_vertex,
        max_vertex,
        radius,
    })
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
    let (u, v) = brush_axes(
        &mesh.positions,
        &mesh.indices,
        center,
        radius,
        view_dir,
        None,
    );
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
    if !(0.0..=1.0).contains(&u) {
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
/// Pushes painted colour 1 texel outward into any completely transparent
/// neighbour pixel within (and 1 pixel around) `dirty`. Only alpha-0 texels
/// are written; already-painted texels are never overwritten. This fills the
/// sub-pixel seam gaps that occur when a UV triangle edge is not axis-aligned
/// and cuts diagonally through a pixel whose center falls just outside the
/// triangle boundary.
fn dilate_seams(tex: &mut TextureData, dirty: (u32, u32, u32, u32)) {
    let (dx0, dy0, dx1, dy1) = dirty;
    let bx0 = dx0.saturating_sub(1);
    let by0 = dy0.saturating_sub(1);
    let bx1 = (dx1 + 1).min(tex.width.saturating_sub(1));
    let by1 = (dy1 + 1).min(tex.height.saturating_sub(1));
    let w = tex.width as usize;
    // Collect writes separately so reads are not affected by in-progress writes.
    let mut writes: Vec<(usize, [u8; 4])> = Vec::new();
    for y in by0..=by1 {
        for x in bx0..=bx1 {
            let idx = (y as usize * w + x as usize) * 4;
            // Only dilate into unpainted (alpha == 0) texels.
            if tex.rgba[idx + 3] != 0 {
                continue;
            }
            // Check 4-connected neighbours; copy the first painted one found.
            let neighbours: [(u32, u32); 4] = [
                (x.wrapping_sub(1), y),
                (x + 1, y),
                (x, y.wrapping_sub(1)),
                (x, y + 1),
            ];
            for (nx, ny) in neighbours {
                if nx >= tex.width || ny >= tex.height {
                    continue;
                }
                let ni = (ny as usize * w + nx as usize) * 4;
                if tex.rgba[ni + 3] > 0 {
                    writes.push((
                        idx,
                        [
                            tex.rgba[ni],
                            tex.rgba[ni + 1],
                            tex.rgba[ni + 2],
                            tex.rgba[ni + 3],
                        ],
                    ));
                    break;
                }
            }
        }
    }
    for (idx, px) in writes {
        tex.rgba[idx..idx + 4].copy_from_slice(&px);
    }
}

fn uv_barycentric(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> Option<(f32, f32)> {
    let d = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
    if d.abs() < 1e-8 {
        return None;
    }
    let inv = 1.0 / d;
    let wa = ((b.y - c.y) * (p.x - c.x) + (c.x - b.x) * (p.y - c.y)) * inv;
    let wb = ((c.y - a.y) * (p.x - c.x) + (a.x - c.x) * (p.y - c.y)) * inv;
    let wc = 1.0 - wa - wb;
    // Half-texel bias: at 1024px one texel ≈ 1/1024 ≈ 0.001 UV units.
    // 5e-4 lets a pixel whose center sits within ~half a milli-UV of a
    // triangle edge be included, which is the typical sub-pixel gap on a
    // 45°-rotated UV island. The original 1e-4 was too tight and caused
    // a 1-pixel-wide unpainted fringe along all diagonal triangle edges.
    const EPS: f32 = 5e-4;
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

/// Quantizes 3D vertex positions to a fine spatial grid to identify coincident
/// vertices across UV seams and weld boundaries.
pub fn weld_vertex_positions(positions: &[Vec3]) -> Vec<u32> {
    if positions.is_empty() {
        return Vec::new();
    }
    let mut bb_min = Vec3::splat(f32::INFINITY);
    let mut bb_max = Vec3::splat(f32::NEG_INFINITY);
    for p in positions {
        bb_min = bb_min.min(*p);
        bb_max = bb_max.max(*p);
    }
    let span = (bb_max - bb_min).length().max(1e-9);
    let quant = (span * 1e-5).clamp(1e-6, 1e-3);
    let q = |c: f32| (c / quant).round() as i32;

    let mut weld_ids = vec![0u32; positions.len()];
    let mut weld: std::collections::HashMap<[i32; 3], u32> =
        std::collections::HashMap::with_capacity(positions.len());
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
    weld_ids
}

/// Component ids for mesh-linked isolation: two triangles belong to the same
/// connected 3D geometry part if they share edges in 3D space (using welded 3D
/// positions). This ignores UV seams so continuous surfaces with multiple UV
/// charts stay in one component, while physically separate geometry (e.g. hair
/// or accessories over a face) remain in distinct components.
pub fn triangle_components_mesh(positions: &[Vec3], indices: &[u32]) -> Vec<usize> {
    let n = indices.len() / 3;
    let mut parent: Vec<usize> = (0..n).collect();
    let weld_ids = weld_vertex_positions(positions);
    let mut edge_first = std::collections::HashMap::with_capacity(indices.len());

    for (tri, tri_idx) in indices.chunks_exact(3).enumerate() {
        let w = [
            weld_ids
                .get(tri_idx[0] as usize)
                .copied()
                .unwrap_or(tri_idx[0]),
            weld_ids
                .get(tri_idx[1] as usize)
                .copied()
                .unwrap_or(tri_idx[1]),
            weld_ids
                .get(tri_idx[2] as usize)
                .copied()
                .unwrap_or(tri_idx[2]),
        ];
        for k in 0..3 {
            let (a, b) = (w[k], w[(k + 1) % 3]);
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
        let n = local_surface_normal(
            &m.positions,
            &m.indices,
            Vec3::new(1.0, 0.0, 0.0),
            0.35,
            None,
        )
        .expect("footprint around (1,0,0) has surface");
        assert!(
            n.dot(Vec3::X) > 0.98,
            "limb normal should hug the surface, got {n:?}"
        );
        // The north pole keeps a mostly-up normal even with a wide footprint.
        let nq = local_surface_normal(
            &m.positions,
            &m.indices,
            Vec3::new(0.0, 1.0, 0.0),
            0.3,
            None,
        )
        .expect("footprint around (0,1,0) has surface");
        assert!(
            nq.dot(Vec3::Y) > 0.97,
            "pole normal should point up, got {nq:?}"
        );
        // Away from the mesh there is nothing to estimate.
        assert!(local_surface_normal(
            &m.positions,
            &m.indices,
            Vec3::new(5.0, 5.0, 5.0),
            0.1,
            None
        )
        .is_none());
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

    #[test]
    fn texture_stamp_paints_the_sprite_once_centered_on_the_dab() {
        // A texture brush stamps its sprite ONCE per dab, centered on the
        // footprint, the sprite's alpha being the coverage — there is no
        // world-space tiling, so a sprite with a transparent half leaves the
        // corresponding half of the footprint clear wherever the dab lands.
        let sprite = hl_sprite(true); // left 2 columns opaque, right 2 clear
        let mut m = uv_quad_plane();
        paint_once(
            &mut m,
            &style_with(BrushShape::Texture, Some(sprite), false),
            StampMode::Paint,
        );
        let bg = [246, 241, 232, 255];
        // Left half of the stamp paints (columns 0..2 cover rx < 0)…
        assert_ne!(texel(&m, 24, 32), bg, "left half of the sprite paints");
        assert_ne!(texel(&m, 31, 32), bg, "paint reaches the dab center line");
        // …the right half stays clear (columns 2..4 are transparent).
        assert_eq!(
            texel(&m, 33, 32),
            bg,
            "right half of the sprite stays clear"
        );
        assert_eq!(texel(&m, 40, 32), bg, "right half stays clear at its edge");
        // The sprite is anchored to the DAB, not tiled across the surface: a
        // texel outside the stamp footprint stays clear even though global
        // position x%4==0 would paint it under world-space tiling.
        assert_eq!(texel(&m, 4, 32), bg, "no tiling beyond the stamp footprint");
    }

    #[test]
    fn texture_stamp_cursor_agrees_with_stamp_for_nonsquare_sprites() {
        // Regression: the GPU cursor resolves the sprite once per dab exactly
        // like the stamp. For a non-square sprite (4 wide × 2 tall), the opaque
        // row occupies the top half of the footprint while the clear row shows
        // through underneath — proving the sprite is drawn at its native aspect
        // and centered, not stretched into the full square footprint.
        let bg = [246, 241, 232, 255];
        let mut m = uv_quad_plane();
        let mut rgba = vec![0u8; 4 * 2 * 4];
        for x in 0..4u32 {
            let i = (x * 4) as usize;
            rgba[i..i + 4].copy_from_slice(&[255, 255, 255, 255]);
        }
        let sprite = TextureData {
            width: 4,
            height: 2,
            rgba,
        };
        let style = BrushStyle {
            shape: BrushShape::Texture,
            sprite: Some(sprite),
            rotation: 0.0,
            flip_x: false,
            flip_y: false,
        };
        apply_stamp_with(
            &mut m,
            Vec3::ZERO,
            1.0,
            Vec3::new(0.0, 0.0, 3.0),
            Vec3::new(0.0, 0.0, -1.0),
            [255, 0, 0, 255],
            1.0,
            1.0,
            StampMode::Paint,
            &style,
            None,
            true,
            None,
        );
        // The sprite's opaque row maps to v_sprite < 0.5 → ry = tv > 0, so the
        // band just above the center line paints…
        assert_ne!(
            texel(&m, 32, 33),
            bg,
            "opaque sprite row paints just above the center line"
        );
        // …the clear row shows through just below center, and the sprite's
        // native 4:2 aspect (half the footprint height) means the full upper
        // region stays clear instead of the sprite being stretched to fill it.
        assert_eq!(
            texel(&m, 32, 31),
            bg,
            "clear sprite row keeps the region below center untouched"
        );
        assert_eq!(
            texel(&m, 32, 40),
            bg,
            "sprite is not stretched: the full upper region stays clear"
        );
    }

    fn texel2d(t: &TextureData, x: u32, y: u32) -> [u8; 4] {
        let i = (y * t.width + x) as usize * 4;
        [t.rgba[i], t.rgba[i + 1], t.rgba[i + 2], t.rgba[i + 3]]
    }

    #[test]
    fn sprite_dab_at_the_uv_seam_wraps_to_column_zero() {
        // A sprite dab straddling the uv_sphere's duplicated seam column (world
        // position of column 0, u = 1.0) must paint both sides of the wrap:
        // the right-edge columns AND column zero — a seamless continuation, with
        // nothing smeared into the texture interior. This locks in the user
        // report "texture stretching around the UV cut": the stamp wraps the
        // sprite footprint instead of clamping at the image edge.
        let mut mesh = crate::io::MeshData::uv_sphere(0.6, 12, 16).with_texture(solid_texture(
            64,
            64,
            [0, 0, 0, 255],
        ));
        // A 4x4 sprite with the TOP half opaque, bottom transparent.
        let mut rgba = vec![0u8; 4 * 4 * 4];
        for y in 0..4u32 {
            for x in 0..4u32 {
                let i = ((y * 4 + x) * 4) as usize;
                let on = y < 2;
                rgba[i..i + 4].copy_from_slice(&if on {
                    [255, 255, 255, 255]
                } else {
                    [255, 255, 255, 0]
                });
            }
        }
        let sprite = TextureData {
            width: 4,
            height: 4,
            rgba,
        };
        let brush = crate::brush::Brush {
            kind: crate::brush::FootprintKind::Sprite,
            size: 0.14,
            hardness: 0.0,
            spacing: 0.0,
            opacity: 1.0,
            accumulate: false,
            color: [255, 0, 0, 255],
            mode: StampMode::Paint,
            sprite: Some(sprite),
            pattern_lock: crate::brush::PatternLock::Dab,
            rotation: 0.0,
            flip_x: false,
            flip_y: false,
            texture_scale: 1.0,
            texture_locked: false,
            texture_size_lock: 0.0,
            texture_window: crate::brush::Window::SpriteBounds,
        };
        // Park the dab just shy of the seam (u ≈ 0.97), eye/view facing it so it
        // behaves like a real viewport stroke at the seam.
        let theta = std::f32::consts::PI * 0.5;
        let phi = 0.97 * 2.0 * std::f32::consts::PI;
        let p = Vec3::new(
            theta.sin() * phi.cos() * 0.6,
            theta.cos() * 0.6,
            theta.sin() * phi.sin() * 0.6,
        );
        let n = p.normalize();
        crate::paint::apply_brush_stamp(
            &mut mesh,
            p,
            0.14,
            p + n * 3.0,
            -n,
            None,
            &brush,
            None,
            None,
            None,
            None,
        );
        let t = mesh.active_layer_texture().unwrap();
        let mut painted = std::collections::BTreeSet::new();
        for y in 0..64u32 {
            for x in 0..64u32 {
                let i = (y * 64 + x) as usize * 4;
                if t.rgba[i] > 0 {
                    painted.insert((x, y));
                }
            }
        }
        assert!(!painted.is_empty(), "the seam dab must paint texels");
        assert!(
            painted.iter().any(|(x, _)| *x == 0),
            "the sprite must wrap into column zero at the seam"
        );
        for (x, _) in &painted {
            assert!(
                *x >= 58 || *x == 0,
                "sprite texel at column {x} must stay inside the seam wrap"
            );
        }
    }

    #[test]
    fn pole_spray_round_and_sprite_smear_identically() {
        // Painting toward the sphere's pole legitimately widens into a warm
        // cap: the top texture rows collapse onto the single pole point, so any
        // dab reaching it covers the whole converging band. This is inherent
        // pole UV convergence, NOT the sprite brush stretching: the plain round
        // brush must smear the same texels. Regression for the "dragging a
        // texture brush to the edges widens/stretches it" report.
        let spray =
            |sprite: Option<TextureData>| -> (usize, usize) {
                let mut mesh = crate::io::MeshData::uv_sphere(0.6, 12, 16)
                    .with_texture(solid_texture(64, 64, [0, 0, 0, 255]));
                let brush = crate::brush::Brush {
                    kind: match sprite {
                        Some(_) => crate::brush::FootprintKind::Sprite,
                        None => crate::brush::FootprintKind::Round,
                    },
                    size: 0.14,
                    hardness: 0.0,
                    spacing: 0.0,
                    opacity: 1.0,
                    accumulate: false,
                    color: [255, 0, 0, 255],
                    mode: StampMode::Paint,
                    sprite,
                    pattern_lock: crate::brush::PatternLock::Dab,
                    rotation: 0.0,
                    flip_x: false,
                    flip_y: false,
                    texture_scale: 1.0,
                    texture_locked: false,
                    texture_size_lock: 0.0,
                    texture_window: crate::brush::Window::Round,
                };
                for step in 0..40 {
                    let theta = 1.2 - 1.1 * (step as f32 / 39.0);
                    let phi = std::f32::consts::PI;
                    let p = Vec3::new(
                        theta.sin() * phi.cos() * 0.6,
                        theta.cos() * 0.6,
                        theta.sin() * phi.sin() * 0.6,
                    );
                    let n = p.normalize();
                    crate::paint::apply_brush_stamp(
                        &mut mesh,
                        p,
                        0.14,
                        p + n * 3.0,
                        -n,
                        None,
                        &brush,
                        None,
                        None,
                        None,
                        None,
                    );
                }
                let t = mesh.active_layer_texture().unwrap();
                let mut texels = 0;
                let mut top_row_span = 0;
                for x in 0..64u32 {
                    if t.rgba[(x * 4) as usize] > 0 {
                        top_row_span += 1;
                        texels += 1;
                    }
                }
                for y in 1..64u32 {
                    for x in 0..64u32 {
                        if t.rgba[(y * 64 + x) as usize * 4] > 0 {
                            texels += 1;
                        }
                    }
                }
                (texels, top_row_span)
            };
        let (round_texels, round_top) = spray(None);
        let (sprite_texels, sprite_top) = spray(Some(TextureData {
            width: 4,
            height: 4,
            rgba: vec![255u8; 4 * 4 * 4],
        }));
        // The converging band reaches the full texture width at the pole for
        // both brushes — the "wide smear" is the sphere's own pole layout.
        assert_eq!(round_top, 64, "round pole reach must span the full top row");
        assert_eq!(
            sprite_top, 64,
            "sprite pole reach must match the round band"
        );
        // And the sprite adds no stretch beyond the round footprint.
        let ratio = round_texels as f32 / sprite_texels as f32;
        assert!(
            (0.85..=1.15).contains(&ratio),
            "round {round_texels} vs sprite {sprite_texels} texels must match"
        );
    }

    #[test]
    fn stamp_2d_texture_wraps_the_sprite_under_the_round_frame() {
        // A 4×1 sprite with only column 0 opaque, stamped into a 32² canvas at
        // (16,16) r=8 through the default Round frame. The sprite's cells scale
        // to ww=2r=16 px wide, and the frame wraps them: the opaque column
        // becomes a vertical band through the left of the dab AND reappears one
        // full cell-period lower (vertical wrap of the 1-row sprite). The round
        // disc then trims the wrapped field: the band's own rim position (x=8,
        // the sprite's left edge) sits at the disc rim and is masked away.
        let bg = [246, 241, 232, 255];
        let mut tex = solid_texture(32, 32, bg);
        let mut sprite_rgba = vec![0u8; 4 * 4];
        sprite_rgba[..4].copy_from_slice(&[255, 255, 255, 255]);
        let brush = crate::brush::Brush {
            kind: crate::brush::FootprintKind::Sprite,
            size: 8.0,
            hardness: 1.0,
            spacing: 0.0,
            opacity: 1.0,
            accumulate: true,
            color: [255, 0, 0, 255],
            mode: StampMode::Paint,
            sprite: Some(TextureData {
                width: 4,
                height: 1,
                rgba: sprite_rgba,
            }),
            pattern_lock: crate::brush::PatternLock::Dab,
            rotation: 0.0,
            flip_x: false,
            flip_y: false,
            texture_scale: 1.0,
            texture_locked: false,
            texture_size_lock: 0.0,
            texture_window: crate::brush::Window::Round,
        };
        stamp_2d(&mut tex, (0.5, 0.5), 8.0, &brush, &mut None, None, None);

        // The sprite spans ww = 4·2·r/4 = 2r = 16 px horizontally, with column
        // 0 opaque over u ∈ [0, 0.25] → dx ∈ [-8, -4]. Inside the disc's flat
        // core (|rel| ≤ r/2 = 4) that is x = 12..; the band is strongest left
        // of center where the round skirt still allows it.
        assert_ne!(
            texel2d(&tex, 10, 16),
            bg,
            "opaque column paints its wrapped band left of center"
        );
        assert_eq!(
            texel2d(&tex, 16, 16),
            bg,
            "center of the dab is clear (column 2 of the sprite)"
        );
        assert_eq!(
            texel2d(&tex, 12, 16),
            bg,
            "column 1 maps to u=0.25→sx=1, also clear at x=12"
        );
        // The wrapped sprite repeats: the opaque column's next tile starts at
        // dx=+8, exactly the disc rim, so it is trimmed away there…
        assert_eq!(
            texel2d(&tex, 24, 16),
            bg,
            "the next tile's opaque column starts at the rim and is masked away"
        );
        // …and the 1-row sprite wraps vertically to the row one cell below.
        assert_ne!(
            texel2d(&tex, 10, 12),
            bg,
            "the sprite wraps to the row one vertical cell below the band"
        );
        // The round frame trims the sprite's own band at the disc rim: the
        // sprite's left edge (dx=-8) coincides with the frame rim, so the otherwise
        // opaque column is masked out at the silhouette.
        assert_eq!(
            texel2d(&tex, 8, 16),
            bg,
            "the round frame trims the sprite band at the disc rim"
        );
        assert_eq!(texel2d(&tex, 16, 7), bg, "above the disc stays clear");
    }

    #[test]
    fn stamp_2d_sprite_is_upright_and_rotates_with_the_cursor() {
        // Regression: the 2D stamp's local frame used texel-down as +y, but the
        // sprite plan maps +y to the sprite's *top* — so the stamped sprite was
        // vertically mirrored against the 2D cursor, and rotation appeared to
        // spin the wrong way at 0deg/180deg. With the canvas-up frame the top
        // row of this top-opaque 2x2 sprite paints above the dab center, and a
        // 90deg rotation moves it to the right side — exactly the cursor's.
        let bg = [246, 241, 232, 255];
        let mut sprite_rgba = vec![0u8; 2 * 2 * 4];
        for c in 0..2u32 {
            let i = (c * 4) as usize;
            sprite_rgba[i..i + 4].copy_from_slice(&[255, 255, 255, 255]);
        }
        let brush = crate::brush::Brush {
            kind: crate::brush::FootprintKind::Sprite,
            size: 4.0,
            hardness: 1.0,
            spacing: 0.0,
            opacity: 1.0,
            accumulate: true,
            color: [255, 0, 0, 255],
            mode: StampMode::Paint,
            sprite: Some(TextureData {
                width: 2,
                height: 2,
                rgba: sprite_rgba,
            }),
            pattern_lock: crate::brush::PatternLock::Dab,
            rotation: 0.0,
            flip_x: false,
            flip_y: false,
            texture_scale: 1.0,
            texture_locked: false,
            texture_size_lock: 0.0,
            texture_window: crate::brush::Window::Round,
        };
        let mut tex = solid_texture(16, 16, bg);
        stamp_2d(&mut tex, (0.5, 0.5), 4.0, &brush, &mut None, None, None);
        assert_eq!(
            texel2d(&tex, 8, 12),
            bg,
            "rotation 0: the opaque top row must not land below the dab center"
        );
        assert_ne!(
            texel2d(&tex, 8, 6),
            bg,
            "rotation 0: the opaque top row paints above the dab center (upright)"
        );
        assert_eq!(
            texel2d(&tex, 8, 4),
            bg,
            "rotation 0: the round frame trims the opaque row at the disc rim"
        );

        let mut b = brush.clone();
        b.rotation = std::f32::consts::FRAC_PI_2;
        let mut tex = solid_texture(16, 16, bg);
        stamp_2d(&mut tex, (0.5, 0.5), 4.0, &b, &mut None, None, None);
        assert_eq!(
            texel2d(&tex, 5, 8),
            bg,
            "rotation 90: the top row must not land on the left of the dab"
        );
        assert_ne!(
            texel2d(&tex, 11, 7),
            bg,
            "rotation 90: the top row rotates to the right, matching the cursor"
        );
    }

    #[test]
    fn stamp_2d_square_frame_masks_like_a_square_not_a_disc() {
        // A fully-opaque 1×1 sprite (so the sprite alpha can't influence the
        // silhouette) stamped at r=8 through the Square frame. The square
        // window's mask distance is max(|dx|,|dy|)/r, so a diagonal texel at
        // (dx,dy)=(7,7) sits at 0.875r — inside the square, faintly painted.
        // A Round frame disc of radius 8 would put that same texel at |rel|≈9.9
        // — fully outside and clear. Painting the diagonal corner while the
        // axis-aligned edge beyond r stays trimmed is exactly "masks as a
        // square", not a round.
        let bg = [246, 241, 232, 255];
        let square = crate::brush::Brush {
            kind: crate::brush::FootprintKind::Sprite,
            size: 8.0,
            hardness: 1.0,
            spacing: 0.0,
            opacity: 1.0,
            accumulate: true,
            color: [255, 0, 0, 255],
            mode: StampMode::Paint,
            sprite: Some(TextureData {
                width: 1,
                height: 1,
                rgba: vec![255, 255, 255, 255],
            }),
            pattern_lock: crate::brush::PatternLock::Dab,
            rotation: 0.0,
            flip_x: false,
            flip_y: false,
            texture_scale: 1.0,
            texture_locked: false,
            texture_size_lock: 0.0,
            texture_window: crate::brush::Window::Square,
        };
        let mut tex = solid_texture(32, 32, bg);
        stamp_2d(&mut tex, (0.5, 0.5), 8.0, &square, &mut None, None, None);
        // (10,10): dx=−6, dy=−6 → 0.75r along both axes. Inside the square
        // (mask = max(6,6)/8 → skirt ≈ 0.5) but the point's radial distance
        // (√72 ≈ 8.5) is OUTSIDE a radius-8 disc.
        assert_ne!(
            texel2d(&tex, 10, 10),
            bg,
            "square frame paints the diagonal corner a round disc would clip"
        );
        assert_ne!(
            texel2d(&tex, 14, 16),
            bg,
            "axis point inside the square paints"
        );
        assert_eq!(
            texel2d(&tex, 16, 8),
            bg,
            "just beyond the square edge (|dy|=r) is trimmed"
        );

        let mut round = square.clone();
        round.texture_window = crate::brush::Window::Round;
        let mut tex = solid_texture(32, 32, bg);
        stamp_2d(&mut tex, (0.5, 0.5), 8.0, &round, &mut None, None, None);
        assert_eq!(
            texel2d(&tex, 10, 10),
            bg,
            "the same diagonal corner is clear under the round frame"
        );
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
            let brush = crate::brush::Brush {
                kind: crate::brush::FootprintKind::Sprite,
                size: 8.0,
                hardness: 1.0,
                spacing: 0.0,
                opacity: 0.5,
                accumulate: false,
                color: [255, 0, 0, 255],
                mode: StampMode::Paint,
                sprite: style.sprite.clone(),
                pattern_lock: crate::brush::PatternLock::Dab,
                rotation: 0.0,
                flip_x: false,
                flip_y: false,
                texture_scale: 1.0,
                texture_locked: false,
                texture_size_lock: 0.0,
                texture_window: crate::brush::Window::Round,
            };
            stamp_2d(t, (0.5, 0.5), 8.0, &brush, &mut None, Some(sa), None);
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

    /// A physically-continuous strip of two quads sharing the x=1 seam in 3D
    /// (coincident vertices, duplicated indices for a UV seam cut), plus a
    /// third quad fully separate at x∈[3,4]. The seam charts join in welded
    /// space but not in index space.
    fn seam_strip_plus_loose_panel() -> MeshData {
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
        // Left chart: x∈[0,1], u∈[0,0.5]. The x=1 edge (corners 3,2) is the seam.
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
        // Right chart: x∈[1,2], u∈[0.5,1]. Shares the x=1 positions with the
        // left chart (coincident, duplicated) — welded in 3D, cut in index space.
        push_quad(
            &mut m,
            [
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 1.0),
                Vec3::new(2.0, 0.0, 1.0),
                Vec3::new(2.0, 0.0, 0.0),
            ],
            [(0.5, 0.0), (0.5, 1.0), (1.0, 1.0), (1.0, 0.0)],
            Vec3::Y,
        );
        // Loose panel: fully separate geometry at x∈[3,4], u∈[0,0.5].
        push_quad(
            &mut m,
            [
                Vec3::new(3.0, 0.0, 0.0),
                Vec3::new(3.0, 0.0, 1.0),
                Vec3::new(4.0, 0.0, 1.0),
                Vec3::new(4.0, 0.0, 0.0),
            ],
            [(0.0, 0.0), (0.0, 1.0), (0.5, 1.0), (0.5, 0.0)],
            Vec3::Y,
        );
        m
    }

    #[test]
    fn mesh_isolation_crosses_a_uv_seam_but_keeps_separate_parts() {
        // Two physically-one charts split by a UV seam, plus a loose part. The
        // mesh connectivity measure must weld the seam (one continuous surface)
        // while keeping the loose panel isolated: edge connectivity is blind to
        // coincidence and wrongly cuts the strip at the seam into two "parts".
        let m = seam_strip_plus_loose_panel();
        let edge = triangle_components_edge(&m.indices);
        let welded = triangle_components_mesh(&m.positions, &m.indices);
        assert!(
            edge[0] == edge[1] && edge[2] == edge[3],
            "each chart's two triangles stay connected"
        );
        assert_ne!(
            edge[0], edge[2],
            "edge components must cut the UV seam into separate parts"
        );
        assert_ne!(
            edge[2], edge[4],
            "edge components also separate the loose panel"
        );
        assert_eq!(
            welded[0], welded[2],
            "welded components must join the two seam charts into one part"
        );
        assert_ne!(
            welded[0], welded[4],
            "the loose panel stays a distinct welded component"
        );
        let distinct: std::collections::HashSet<usize> = welded.iter().copied().collect();
        assert_eq!(distinct.len(), 2, "strip + loose panel = two welded parts");
    }

    #[test]
    fn mesh_isolation_locks_a_stamp_to_the_welded_part_crossing_the_seam() {
        // Seed the left chart; a stamp spanning x∈[0,2] must paint both charts
        // (the seam is welded, not a part boundary). The old edge-based split
        // lock stops at the seam and paints only the left chart.
        let eye = Vec3::new(0.5, 3.0, 0.5);
        let dir = Vec3::new(0.0, -1.0, 0.0);
        let center = Vec3::new(1.0, 0.0, 0.5);
        let seed = mesh_raycast(&seam_strip_plus_loose_panel(), eye, dir)
            .expect("brush center hits the strip")
            .triangle;
        assert_eq!(
            seed, 0,
            "sanity: the seed is the left chart's first triangle"
        );

        let mut isolated = seam_strip_plus_loose_panel();
        let accel = StampAccel::with_isolation(&isolated, None, Some(seed));
        apply_stamp_with(
            &mut isolated,
            center,
            2.2,
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
            texel(&isolated, 16, 32),
            [255, 0, 0, 255],
            "mesh isolation paints the seeded left chart"
        );
        assert_eq!(
            texel(&isolated, 48, 32),
            [255, 0, 0, 255],
            "mesh isolation crosses the UV seam onto the right chart"
        );

        let mut split = seam_strip_plus_loose_panel();
        let accel = StampAccel::new(&split, Some(seed));
        apply_stamp_with(
            &mut split,
            center,
            2.2,
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
            texel(&split, 16, 32),
            [255, 0, 0, 255],
            "edge split paints the seeded left chart"
        );
        assert_eq!(
            texel(&split, 48, 32),
            [200, 200, 200, 255],
            "edge split stops at the UV seam"
        );
    }

    #[test]
    fn mesh_isolation_locks_a_stamp_off_a_separate_welded_part() {
        // Two disconnected panels (world x∈[0,1] and x∈[2,3], UVs covering the
        // full image in disjoint halves): mesh isolation must keep a big stamp
        // on the seeded panel exactly like edge split lock, now via welded
        // components instead of edge components.
        let eye = Vec3::new(0.5, 2.0, 0.5);
        let dir = Vec3::new(0.0, -1.0, 0.0);
        let center = Vec3::new(0.5, 0.0, 0.5);
        let seed = mesh_raycast(&two_panels_up(), eye, dir)
            .expect("brush center hits panel 0")
            .triangle;
        assert_eq!(seed, 0, "sanity: the seed is panel 0's first triangle");

        let mut locked = two_panels_up();
        let accel = StampAccel::with_isolation(&locked, None, Some(seed));
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
            "mesh isolation still paints the seeded part"
        );
        assert_eq!(
            texel(&locked, 38, 32),
            [200, 200, 200, 255],
            "mesh isolation keeps the stamp off the separate panel"
        );
    }

    #[test]
    fn fill_region_mesh_crosses_a_uv_seam_but_spares_separate_parts() {
        // Welded fills reach across UV seams (unlike the edge-based fill that
        // stops at the seam cut) while still sparing physically separate parts.
        let eye = Vec3::new(0.5, 3.0, 0.5);
        let dir = Vec3::new(0.0, -1.0, 0.0);
        let mut strip = seam_strip_plus_loose_panel();
        let seed = mesh_raycast(&strip, eye, dir).unwrap().triangle;
        fill_region_mesh(&mut strip, seed, [190, 120, 30, 255], 1.0);
        assert_eq!(
            texel(&strip, 16, 32),
            [190, 120, 30, 255],
            "welded fill reaches the seeded left chart"
        );
        assert_eq!(
            texel(&strip, 48, 32),
            [190, 120, 30, 255],
            "welded fill crosses the UV seam onto the right chart"
        );

        // Separate panels (disjoint UVs): filling panel 0 leaves panel 1 intact.
        let mut panels = two_panels_up();
        let seed = mesh_raycast(&panels, eye, dir).unwrap().triangle;
        assert_eq!(seed, 0, "seed lands on panel 0");
        fill_region_mesh(&mut panels, seed, [190, 120, 30, 255], 1.0);
        assert_eq!(
            texel(&panels, 16, 32),
            [190, 120, 30, 255],
            "welded fill paints the seeded panel"
        );
        assert_eq!(
            texel(&panels, 38, 32),
            [200, 200, 200, 255],
            "welded fill leaves the separate panel untouched"
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

    #[test]
    fn stamp_2d_pattern_lock_replace_tiles_the_anchored_phase_across_dabs() {
        // A texture brush (left half opaque, right half transparent) over a
        // two-frame sweep as a plain dab train: a click dab at (32,8) then a
        // continuation dab at (38,8). Each dab's paint window is a disc of the
        // brush radius, and the sprite is read at the anchored, UV-wrapped
        // phase — the pattern tiles through the brush footprint without
        // re-centering, so two overlapping dabs sample identical texels and
        // the shared stroke buffer caps every texel once (no "fill up").
        let bg = [246, 241, 232, 255];
        let mut tex = solid_texture(64, 16, bg);
        let mut sprite_rgba = vec![0u8; 2 * 2 * 4];
        for y in 0..2u32 {
            sprite_rgba[((y * 2) as usize) * 4..((y * 2 + 1) as usize) * 4]
                .copy_from_slice(&[255, 255, 255, 255]);
        }
        let brush = crate::brush::Brush {
            kind: crate::brush::FootprintKind::Sprite,
            size: 8.0,
            hardness: 1.0,
            spacing: 0.0,
            opacity: 1.0,
            accumulate: true,
            color: [0, 128, 0, 255],
            mode: StampMode::Paint,
            sprite: Some(TextureData {
                width: 2,
                height: 2,
                rgba: sprite_rgba,
            }),
            pattern_lock: crate::brush::PatternLock::Aligned,
            rotation: 0.0,
            flip_x: false,
            flip_y: false,
            texture_scale: 1.0,
            texture_locked: false,
            texture_size_lock: 0.0,
            texture_window: crate::brush::Window::Round,
        };
        let mut stroke_alpha = vec![0u8; 64 * 16];
        let anchor = crate::brush::PatternAnchor::Canvas {
            x: 32.0f32,
            y: 8.0,
            radius: 4.0,
        };
        // Frame 1 (click): one disc at (32,8).
        stamp_2d(
            &mut tex,
            (32.0 / 64.0, 8.0 / 16.0),
            4.0,
            &brush,
            &mut None,
            Some(&mut stroke_alpha),
            Some(&anchor),
        );
        // Frame 2 (continuation): a dab stepped 6 texels to (38,8).
        stamp_2d(
            &mut tex,
            (38.0 / 64.0, 8.0 / 16.0),
            4.0,
            &brush,
            &mut None,
            Some(&mut stroke_alpha),
            Some(&anchor),
        );

        // The sprite spans 8 texels (diameter 2r); wrapped by period 8, its
        // left-opaque half covers anchored positions where (x - 32) mod 8 ∈
        // [-4, 0) ∪ [4, 8), the other half is transparent. Over the two dabs
        // (x ∈ [28, 36] and [34, 42]) the anchored tiling reveals itself:
        // x = 28..31 and 36..39 painted, x = 32..35 and 40..41 clear — an
        // anchored rubber-stamp across the footprint, no disc rhythm.
        assert_ne!(
            texel2d(&tex, 30, 8),
            bg,
            "the click dab paints through its anchored left-opaque phase"
        );
        assert_eq!(
            texel2d(&tex, 33, 8),
            bg,
            "the tiled transparent half stays clear inside the dab disc"
        );
        assert_eq!(
            texel2d(&tex, 34, 8),
            bg,
            "both dabs read the same transparent anchored phase"
        );
        assert_ne!(
            texel2d(&tex, 39, 8),
            bg,
            "the wrapped anchored phase tiles into the second dab one period on"
        );
        assert_eq!(
            texel2d(&tex, 40, 8),
            bg,
            "past the wrapped opaque band the pattern is transparent again"
        );
        // Each dab's window is its own disc: (31,5) is inside the click dab
        // (dx=-1, dy=-3) at the opaque phase, (31,3) is past the radius.
        assert_ne!(
            texel2d(&tex, 31, 5),
            bg,
            "the dab disc sweeps up near its center row"
        );
        assert_eq!(
            texel2d(&tex, 31, 3),
            bg,
            "outside the dab radius the stroke stops cleanly"
        );
        // Outside the dabs no texels change (no global tiling).
        assert_eq!(
            texel2d(&tex, 1, 8),
            bg,
            "nothing paints beyond the drawn dabs"
        );
    }

    #[test]
    fn stamp_2d_pattern_lock_replace_caps_overlap_like_a_clean_stencil() {
        // Three overlapping pattern dabs with a solid opaque sprite at 50%
        // opacity, accumulate = true (the classic dab train would stack the
        // overlap to 75%). The stroke max-buffer caps every texel at one dab's
        // worth, so a texel covered by several dabs is pixel-identical to one
        // covered once — overlapping pattern dabs never "fill up".
        let bg = [246, 241, 232, 255];
        let mut tex = solid_texture(64, 16, bg);
        let brush = crate::brush::Brush {
            kind: crate::brush::FootprintKind::Sprite,
            size: 12.0,
            hardness: 1.0,
            spacing: 0.0,
            opacity: 0.5,
            accumulate: true,
            color: [0, 128, 0, 255],
            mode: StampMode::Paint,
            sprite: Some(TextureData {
                width: 1,
                height: 1,
                rgba: vec![255, 255, 255, 255],
            }),
            pattern_lock: crate::brush::PatternLock::Aligned,
            rotation: 0.0,
            flip_x: false,
            flip_y: false,
            texture_scale: 1.0,
            texture_locked: false,
            texture_size_lock: 0.0,
            texture_window: crate::brush::Window::Round,
        };
        let anchor = crate::brush::PatternAnchor::Canvas {
            x: 20.0f32,
            y: 8.0,
            radius: 6.0,
        };
        let mut stroke_alpha = vec![0u8; 64 * 16];
        // Three half-overlapping dabs (radius 6, centers 6 apart): (20,8),
        // (26,8), (32,8) — the middle one is covered by all three.
        for x in [20.0f32, 26.0, 32.0] {
            stamp_2d(
                &mut tex,
                (x / 64.0, 8.0 / 16.0),
                6.0,
                &brush,
                &mut None,
                Some(&mut stroke_alpha),
                Some(&anchor),
            );
        }
        // (17,8) is covered only by the first dab, (26,8) sits in the overlap,
        // (35,8) only by the final dab. All three must hold exactly one dab's
        // worth of green (50% -> 75% would be stacking, 50% == onced == flat).
        let once = texel2d(&tex, 17, 8);
        assert_ne!(once, bg, "the first dab paints through its disc");
        for (x, label) in [(26u32, "triple overlap"), (35u32, "last dab only")] {
            assert_eq!(
                texel2d(&tex, x, 8),
                once,
                "{label} texel must equal a once-painted texel (no stacking)"
            );
        }
        // (17,15) is 7 texels from every center — outside every dab disc.
        assert_eq!(
            texel2d(&tex, 17, 15),
            bg,
            "outside every dab disc nothing paints"
        );
    }

    #[test]
    fn stamp_2d_pattern_lock_dense_dabs_melt_into_a_flat_stroke() {
        // The coin-chain fix: aligned-pattern strokes ride a dab train spaced
        // at `Brush::pattern_spacing` (half the brush radius) with a soft
        // window mask (flat core through 0.5r, C¹ skirt to the rim). Every
        // texel inside the stroke band sits within the flat core of some dab,
        // so the max-combined envelope is perfectly flat — no seams, beads,
        // density dips, or crisp disc arcs at the dab boundaries.
        let bg = [246, 241, 232, 255];
        let mut tex = solid_texture(64, 16, bg);
        let brush = crate::brush::Brush {
            kind: crate::brush::FootprintKind::Sprite,
            size: 8.0, // r = 4
            hardness: 1.0,
            spacing: 0.0,
            opacity: 1.0,
            accumulate: true,
            color: [0, 128, 0, 255],
            mode: StampMode::Paint,
            sprite: Some(TextureData {
                width: 1,
                height: 1,
                rgba: vec![255, 255, 255, 255],
            }),
            pattern_lock: crate::brush::PatternLock::Aligned,
            rotation: 0.0,
            flip_x: false,
            flip_y: false,
            texture_scale: 1.0,
            texture_locked: false,
            texture_size_lock: 0.0,
            texture_window: crate::brush::Window::Round,
        };
        let anchor = crate::brush::PatternAnchor::Canvas {
            x: 32.0f32,
            y: 8.0,
            radius: 4.0,
        };
        let mut stroke_alpha = vec![0u8; 64 * 16];
        // The stroke lays dabs every `pattern_spacing()` texels — half the
        // brush radius, well inside every dab's flat core.
        let step = brush.pattern_spacing();
        for x in (24..=40).step_by(step as usize) {
            stamp_2d(
                &mut tex,
                (x as f32 / 64.0, 8.0 / 16.0),
                4.0,
                &brush,
                &mut None,
                Some(&mut stroke_alpha),
                Some(&anchor),
            );
        }
        // The whole band is a solid, seam-free strip: a dab-center texel and a
        // midway-between-dabs texel are pixel-identical, and so is every texel
        // inside the stroke (no chain-link cresting along the spine).
        let center = texel2d(&tex, 32, 8);
        assert_ne!(center, bg, "the stroke paints its own spine");
        for x in 24..=40 {
            assert_eq!(
                texel2d(&tex, x, 8),
                center,
                "texel {x} along the spine must be flat (no bead between dabs)"
            );
        }
        // The stroke halts cleanly: a texel beyond every dab's support stays
        // untouched, and the feather zone one texel past the core is softer,
        // not a crisp rim.
        assert_eq!(
            texel2d(&tex, 2, 8),
            bg,
            "nothing paints far outside the drawn stroke"
        );
        assert_eq!(
            texel2d(&tex, 46, 8),
            bg,
            "the stroke's far end stops at the last dab's radius"
        );
    }

    #[test]
    fn stamp_2d_pattern_lock_reveals_a_stationary_grid_under_dab_resize() {
        // The aligned pattern is an *absolute* canvas-space grid: the texture
        // phase comes only from the texel position minus the stroke-start
        // anchor, never the dab center or the dab's radius. Two dabs stamped
        // at the same spot with different radii must therefore reveal
        // pixel-identical texture over their overlap — only the reach of the
        // pure distance mask changes, the texture underneath stays glued.
        let bg = [246, 241, 232, 255];
        let mut sprite_rgba = vec![0u8; 2 * 2 * 4];
        for y in 0..2u32 {
            sprite_rgba[((y * 2) as usize) * 4..((y * 2 + 1) as usize) * 4]
                .copy_from_slice(&[255, 255, 255, 255]);
        }
        let brush = crate::brush::Brush {
            kind: crate::brush::FootprintKind::Sprite,
            size: 16.0, // nominal; the dabs below pass their own radii
            hardness: 1.0,
            spacing: 0.0,
            opacity: 1.0,
            accumulate: true,
            color: [0, 128, 0, 255],
            mode: StampMode::Paint,
            sprite: Some(TextureData {
                width: 2,
                height: 2,
                rgba: sprite_rgba,
            }),
            pattern_lock: crate::brush::PatternLock::Aligned,
            rotation: 0.0,
            flip_x: false,
            flip_y: false,
            texture_scale: 1.0,
            texture_locked: false,
            texture_size_lock: 0.0,
            texture_window: crate::brush::Window::Round,
        };
        // Anchor radius 8 -> one square tile = 2·8 = 16 texels in the canvas.
        let anchor = crate::brush::PatternAnchor::Canvas {
            x: 8.0f32,
            y: 8.0,
            radius: 8.0,
        };

        // A 4-texel and an 8-texel dab at the same center, separate strokes.
        let mut small = solid_texture(64, 16, bg);
        let mut sa = vec![0u8; 64 * 16];
        stamp_2d(
            &mut small,
            (16.0 / 64.0, 8.0 / 16.0),
            4.0,
            &brush,
            &mut None,
            Some(&mut sa),
            Some(&anchor),
        );
        let mut large = solid_texture(64, 16, bg);
        let mut la = vec![0u8; 64 * 16];
        stamp_2d(
            &mut large,
            (16.0 / 64.0, 8.0 / 16.0),
            8.0,
            &brush,
            &mut None,
            Some(&mut la),
            Some(&anchor),
        );

        const FULL: [u8; 4] = [0, 128, 0, 255];
        let mut same = 0;
        let mut revealed = 0;
        for y in 4..=12u32 {
            for x in 12..=20u32 {
                let s = texel2d(&small, x, y);
                let l = texel2d(&large, x, y);
                if l == FULL {
                    revealed += 1;
                }
                if s == FULL {
                    // Inside every dab's fully-unmasking region the texture
                    // phase is byte-identical under either radius: the grid
                    // stays glued, only the distance mask's reach changes.
                    assert_eq!(
                        s,
                        l,
                        "fully-unmasked texel ({x},{y}) must reveal the same grid under either dab radius"
                    );
                    same += 1;
                }
            }
        }
        assert!(same > 0, "the small disc fully reveals part of the grid");
        assert!(
            revealed > same,
            "the larger dab reaches further (distance mask) without moving the texture"
        );
    }

    #[test]
    fn apply_brush_stamp_pattern_lock_replace_does_not_stack_overlapping_dabs() {
        // Two identical pattern-locked dabs overlap the same region through a
        // shared stroke buffer: the replace blend records the MAX per texel,
        // so re-stamping an already-painted area (the "pattern fills itself
        // up" case on a slow drag / wiggle) is pixel-identical to a single
        // dab — exactly the flat non-accumulating overwrite the aligned
        // pattern mode wants. Radius 0.5 world = 8 texels on the 64² quad.
        let anchor = crate::brush::PatternAnchor::Surface {
            pos: Vec3::new(0.5, 0.0, 0.0),
            axis_u: Vec3::new(1.0, 0.0, 0.0),
            axis_v: Vec3::new(0.0, 1.0, 0.0),
            radius: 0.5,
        };

        let ray = |x: f32| (Vec3::new(x, 0.0, 3.0), Vec3::new(0.0, 0.0, -1.0));

        // A 50%-opacity pattern brush (a solid opaque sprite); accumulation
        // would otherwise stack an overlap to 75%.
        let mut brush = pattern_brush();
        brush.opacity = 0.5;

        // Control: the same dab applied exactly once through a fresh stroke
        // buffer (the quad's shared diagonal means a single dab can visit a
        // texel twice across its two triangles — the buffer caps that).
        let mut once = uv_quad_plane();
        let tw = once.layers[0].texture.width as usize;
        let th = once.layers[0].texture.height as usize;
        let mut once_sa = vec![0u8; tw * th];
        let (o, d) = ray(0.5);
        apply_brush_stamp(
            &mut once,
            Vec3::new(0.5, 0.0, 0.0),
            0.5,
            o,
            d,
            None,
            &brush,
            None,
            Some(&mut once_sa),
            Some(&anchor),
            None,
        );

        // Double-dab the identical center through the shared max buffer.
        let mut twice = uv_quad_plane();
        let tw = twice.layers[0].texture.width as usize;
        let th = twice.layers[0].texture.height as usize;
        let mut stroke_alpha = vec![0u8; tw * th];
        for _ in 0..2 {
            let (o, d) = ray(0.5);
            apply_brush_stamp(
                &mut twice,
                Vec3::new(0.5, 0.0, 0.0),
                0.5,
                o,
                d,
                None,
                &brush,
                None,
                Some(&mut stroke_alpha),
                Some(&anchor),
                None,
            );
        }

        // Every texel of the doubled stroke equals the once-painted control:
        // re-stamping can never deepen the pattern.
        for y in 0..th {
            for x in 0..tw {
                assert_eq!(
                    texel(&twice, x as u32, y as u32),
                    texel(&once, x as u32, y as u32),
                    "a doubled pattern dab must equal a single dab at texel ({x},{y})"
                );
            }
        }

        // Two separated dabs (disc train): the swept band between them stays
        // clear — there is no ribbon band, each dab is its own disc.
        let mut train = uv_quad_plane();
        let mut stroke_alpha = vec![0u8; tw * th];
        for x in [0.5f32, 1.5] {
            let (o, d) = ray(x);
            apply_brush_stamp(
                &mut train,
                Vec3::new(x, 0.0, 0.0),
                0.5,
                o,
                d,
                None,
                &brush,
                None,
                Some(&mut stroke_alpha),
                Some(&anchor),
                None,
            );
        }
        // (46,39): world ≈ (0.906, 0.469) sits between the two discs (≥0.4
        // from each center, > the 0.5 brush radius) — a plain dab train does
        // not paint it.
        assert_eq!(
            texel(&train, 46, 39),
            [246, 241, 232, 255],
            "a dab train leaves the band between two discs clear"
        );
        // The doubly-stamped center texel is the flat once-painted value.
        assert_eq!(
            texel(&twice, 40, 32),
            texel(&once, 40, 32),
            "the anchor texel stays at one dab's worth after re-stamping"
        );
    }

    #[test]
    fn apply_brush_stamp_surface_pattern_is_world_exact_on_a_flat_face() {
        // The exact pathway the app uses on a perfectly flat face: a real
        // geodesic `SurfaceUnwrap` seeded at the click triangle, a
        // `PatternAnchor::Surface` pinning a 4x4 half-opaque sprite, and a
        // disc dab. The anchored phase on a flat face must equal the texel's
        // world chord (the unwrap is bit-equivalent there), so the painted
        // pattern is pinned 1:1 to the world: the sprite's opaque half covers
        // world x < 0 everywhere inside the disc, and the seam sits exactly at
        // world x = 0 — no stretch, no drift, no rotation.
        let mut mesh = uv_quad_plane();
        let anchor = crate::brush::PatternAnchor::Surface {
            pos: Vec3::ZERO,
            axis_u: Vec3::X,
            axis_v: Vec3::Y,
            radius: 0.5,
        };
        // left 2 columns opaque, right 2 transparent (4x4).
        let mut sprite_rgba = vec![0u8; 4 * 4 * 4];
        for y in 0..4u32 {
            for x in 0..4u32 {
                let i = ((y * 4 + x) * 4) as usize;
                sprite_rgba[i..i + 4].copy_from_slice(&if x < 2 {
                    [255, 255, 255, 255]
                } else {
                    [255, 255, 255, 0]
                });
            }
        }
        let mut brush = pattern_brush();
        brush.sprite = Some(TextureData {
            width: 4,
            height: 4,
            rgba: sprite_rgba,
        });
        let unwrap = surface_unwrap(
            &mesh.positions,
            &mesh.indices,
            Vec3::ZERO,
            Vec3::X,
            Vec3::Y,
            0,
            f32::INFINITY,
        )
        .expect("flat quad unfolds");
        let tw = mesh.layers[0].texture.width as usize;
        let th = mesh.layers[0].texture.height as usize;
        let mut stroke_alpha = vec![0u8; tw * th];
        let (o, d) = (Vec3::new(0.0, 0.0, 3.0), Vec3::new(0.0, 0.0, -1.0));
        apply_brush_stamp(
            &mut mesh,
            Vec3::ZERO,
            0.5,
            o,
            d,
            None,
            &brush,
            None,
            Some(&mut stroke_alpha),
            Some(&anchor),
            Some(&unwrap),
        );
        let bg = [246, 241, 232, 255];
        let world = |i: u32| -2.0 + (i as f32 + 0.5) * 4.0 / 64.0;
        let painted = |m: &MeshData, x: u32, y: u32, what: &str| {
            let px = texel(m, x, y);
            assert_ne!(
                px,
                bg,
                "{what}: texel ({x},{y}) world ({}, {}) must be painted, got {px:?}",
                world(x),
                world(y)
            );
        };
        let clear = |m: &MeshData, x: u32, y: u32, what: &str| {
            let px = texel(m, x, y);
            assert_eq!(
                px,
                bg,
                "{what}: texel ({x},{y}) world ({}, {}) must stay clear, got {px:?}",
                world(x),
                world(y)
            );
        };
        // Centre row: the seam between the sprite's halves falls exactly on
        // world x = 0. Texel 31 → x ≈ -0.031 (opaque half, inside disc);
        // texel 32 → x ≈ +0.031 (transparent half).
        painted(
            &mesh,
            31,
            32,
            "sprite's opaque half reaches the seam at world x=0",
        );
        clear(
            &mesh,
            32,
            32,
            "the transparent half starts exactly past x=0",
        );
        // Above/below the seam along the centre column: still pinned to x=0.
        painted(&mesh, 31, 24, "seam pinned on the upper row");
        clear(&mesh, 32, 24, "transparent side pinned on the upper row");
        // Screen-anchored within the disc: left half painted near the rims…
        painted(&mesh, 24, 32, "left inside the disc is painted");
        // …the right half of the disc is transparent (not a mirror, not an
        // over-big sprite): x≈+1.98 stays clear.
        clear(
            &mesh,
            40,
            32,
            "right inside the disc is the transparent sprite half",
        );
        // Outside the disc nothing paints.
        clear(&mesh, 31, 0, "above the disc nothing paints");
        // The pattern is world-pinned, so a dab whose radius differs from the
        // stroke-start anchor radius still lays the SAME world grid: re-stamp
        // with a larger radius and the seam must stay at x=0, not drift.
        let mut stroke_alpha = vec![0u8; tw * th];
        apply_brush_stamp(
            &mut mesh,
            Vec3::ZERO,
            1.0,
            o,
            d,
            None,
            &brush,
            None,
            Some(&mut stroke_alpha),
            Some(&anchor),
            Some(&unwrap),
        );
        painted(
            &mesh,
            31,
            32,
            "seam stays at x=0 under a different dab radius",
        );
        clear(
            &mesh,
            32,
            32,
            "transparent half still at x>0 under a different dab radius",
        );
        painted(
            &mesh,
            31,
            16,
            "the bigger disc paints further up while staying pinned",
        );
        clear(&mesh, 32, 16, "still transparent past x=0");
        // The anchored grid keeps its 1.0-world period and its world phase even
        // under the bigger dab: texel 23 (world x ≈ -0.53) is the transparent
        // half of the neighbouring tile, texel 40 (world x ≈ 0.53) is the
        // opaque half of tile +1.
        clear(
            &mesh,
            23,
            32,
            "x≈-0.53 is the transparent half of the neighbouring tile",
        );
        painted(
            &mesh,
            40,
            32,
            "x≈0.53 is the opaque half of tile +1, still world-pinned",
        );
    }

    fn pattern_brush() -> crate::brush::Brush {
        crate::brush::Brush {
            kind: crate::brush::FootprintKind::Sprite,
            size: 8.0,
            hardness: 1.0,
            spacing: 0.0,
            opacity: 1.0,
            accumulate: true,
            color: [0, 128, 0, 255],
            mode: StampMode::Paint,
            sprite: Some(TextureData {
                width: 1,
                height: 1,
                rgba: vec![255, 255, 255, 255],
            }),
            pattern_lock: crate::brush::PatternLock::Aligned,
            rotation: 0.0,
            flip_x: false,
            flip_y: false,
            texture_scale: 1.0,
            texture_locked: false,
            texture_size_lock: 0.0,
            texture_window: crate::brush::Window::Round,
        }
    }

    #[test]
    fn apply_brush_stamp_uv_pattern_anchor_does_not_stack_overlapping_dabs() {
        let anchor = crate::brush::PatternAnchor::Uv {
            x: 32.0,
            y: 32.0,
            radius: 8.0,
        };
        let ray = |x: f32| (Vec3::new(x, 0.0, 3.0), Vec3::new(0.0, 0.0, -1.0));

        let mut brush = pattern_brush();
        brush.opacity = 0.5;

        let mut once = uv_quad_plane();
        let tw = once.layers[0].texture.width as usize;
        let th = once.layers[0].texture.height as usize;
        let mut once_sa = vec![0u8; tw * th];
        let (o, d) = ray(0.5);
        apply_brush_stamp(
            &mut once,
            Vec3::new(0.5, 0.0, 0.0),
            0.5,
            o,
            d,
            None,
            &brush,
            None,
            Some(&mut once_sa),
            Some(&anchor),
            None,
        );

        let mut twice = uv_quad_plane();
        let mut stroke_alpha = vec![0u8; tw * th];
        for _ in 0..2 {
            let (o, d) = ray(0.5);
            apply_brush_stamp(
                &mut twice,
                Vec3::new(0.5, 0.0, 0.0),
                0.5,
                o,
                d,
                None,
                &brush,
                None,
                Some(&mut stroke_alpha),
                Some(&anchor),
                None,
            );
        }

        // Overlapping dabs do not stack or destroy the pattern
        for y in 0..th {
            for x in 0..tw {
                assert_eq!(
                    texel(&twice, x as u32, y as u32),
                    texel(&once, x as u32, y as u32),
                    "a doubled UV pattern dab must equal a single dab at texel ({x},{y})"
                );
            }
        }
    }

    #[test]
    fn apply_brush_stamp_pattern_square_window_paints_by_the_frame_not_a_disc() {
        // Regression: the 3D stamp collapsed a pattern-locked dab's window-local
        // coordinates to a single radial distance, so the Square/Diamond frame
        // painted a round disc (both mask terms vanished) and never matched the
        // cursor's (tu, tv) frame. The window mask must keep the plane
        // coordinates.
        let anchor = crate::brush::PatternAnchor::Uv {
            x: 32.0,
            y: 32.0,
            radius: 8.0,
        };
        let stamp = |window: crate::brush::Window| {
            let mut mesh = uv_quad_plane();
            let tw = mesh.layers[0].texture.width as usize;
            let th = mesh.layers[0].texture.height as usize;
            let mut stroke_alpha = vec![0u8; tw * th];
            let mut brush = pattern_brush();
            brush.texture_window = window;
            apply_brush_stamp(
                &mut mesh,
                Vec3::ZERO,
                0.5,
                Vec3::new(0.0, 0.0, 3.0),
                Vec3::new(0.0, 0.0, -1.0),
                None,
                &brush,
                None,
                Some(&mut stroke_alpha),
                Some(&anchor),
                None,
            );
            mesh
        };
        let background = [246, 241, 232, 255];
        let square = stamp(crate::brush::Window::Square);
        let round = stamp(crate::brush::Window::Round);
        let mut frame_corners = 0;
        for y in 0..64u32 {
            for x in 0..64u32 {
                let painted_by_square = texel(&square, x, y) != background;
                let painted_by_round = texel(&round, x, y) != background;
                if painted_by_square && !painted_by_round {
                    // Extra square paint must sit off-axis at a rim corner the
                    // round disc leaves clear — not along an axis.
                    let (dx, dy) = (x as i32 - 32, y as i32 - 32);
                    assert!(
                        dx != 0 && dy != 0,
                        "extra square paint must sit off-axis at a rim corner"
                    );
                    frame_corners += 1;
                }
            }
        }
        assert!(
            frame_corners > 0,
            "the square frame must paint rim corners the round frame leaves clear"
        );
    }

    #[test]
    fn apply_brush_stamp_uv_pattern_anchor_seamlessly_paints_across_neighboring_triangles() {
        // uv_quad_plane is made of two triangles sharing a diagonal.
        // A stroke using PatternAnchor::Uv unmasks the tiled pattern seamlessly
        // across both triangles based on their continuous UV coordinates.
        let anchor = crate::brush::PatternAnchor::Uv {
            x: 32.0,
            y: 32.0,
            radius: 16.0,
        };
        let mut mesh = uv_quad_plane();
        let tw = mesh.layers[0].texture.width as usize;
        let th = mesh.layers[0].texture.height as usize;
        let mut stroke_alpha = vec![0u8; tw * th];

        let brush = pattern_brush();
        let (o, d) = (Vec3::new(0.0, 0.0, 3.0), Vec3::new(0.0, 0.0, -1.0));
        apply_brush_stamp(
            &mut mesh,
            Vec3::new(0.0, 0.0, 0.0),
            0.8,
            o,
            d,
            None,
            &brush,
            None,
            Some(&mut stroke_alpha),
            Some(&anchor),
            None,
        );

        // Verify that texels on BOTH triangles of the quad are painted
        let mut painted_count = 0;
        for y in 0..th {
            for x in 0..tw {
                let px = texel(&mesh, x as u32, y as u32);
                if px == [0, 128, 0, 255] {
                    painted_count += 1;
                }
            }
        }
        assert!(
            painted_count > 50,
            "both neighboring triangles must have the UV pattern seamlessly painted"
        );
    }

    #[test]
    fn surface_unwrap_flat_patch_matches_anchor_plane() {
        // On a flat quad the geodesic unfold conserves the exact edge lengths,
        // so every vertex phase is the anchor-plane chord — and because the
        // seed triangle is anchored by chord, the whole patch is bit-identical
        // to the flat-projection it replaces.
        let m = uv_quad_plane();
        let anchor = Vec3::ZERO;
        let (axis_u, axis_v) = (Vec3::X, Vec3::Y);
        let unwrap = surface_unwrap(&m.positions, &m.indices, anchor, axis_u, axis_v, 0, 5.0)
            .expect("flat quad within radius");
        let expected = [
            Vec2::new(-2.0, -2.0),
            Vec2::new(2.0, -2.0),
            Vec2::new(2.0, 2.0),
            Vec2::new(-2.0, 2.0),
        ];
        for (v, want) in expected.iter().enumerate() {
            let got = unwrap
                .vertex_phase(v as u32)
                .unwrap_or_else(|| panic!("vertex {v} must be unfolded"));
            assert!(
                (got - *want).length() < 1e-4,
                "flat vertex {v} phase {got:?} must equal its chord {want:?}"
            );
        }
        let (min, max) = unwrap.span().expect("patch has vertices");
        assert_eq!((min, max), (0, 3), "both quad triangles are covered");
        // tri_phases follow the quad's winding: tri 0 is (0,1,2), tri 1 is
        // (0,2,3) — same values the stamp interpolates per texel.
        assert_eq!(unwrap.tri(0), Some([expected[0], expected[1], expected[2]]));
        let t1 = unwrap.tri(1).expect("quad's second triangle unfolded");
        for (k, want) in [expected[0], expected[2], expected[3]].iter().enumerate() {
            assert!(
                (t1[k] - *want).length() < 1e-4,
                "tri(1) vertex {k} must equal its chord {want:?}, got {:?}",
                t1[k]
            );
        }
    }

    #[test]
    fn surface_unwrap_open_book_does_not_fold_onto_anchor() {
        // Two triangles sharing the X axis, folded 90° apart (an open book):
        // the anchor sits on the flat page at the hinge, and the folded page
        // peaks along +Z. The far vertex D=(0,0,2) has ZERO chord against the
        // anchor axes (the whole folded page collapses onto the hinge line in
        // the anchor plane — the sphere-back-side bug). The geodesic unfold
        // instead keeps the world edge length from the hinge, so D unfolds to
        // (0,-2) phase: the page opens out flat instead of folding.
        let m = MeshData {
            positions: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(0.0, 2.0, 0.0),
                Vec3::new(0.0, 0.0, 2.0),
            ],
            normals: vec![Vec3::Z, Vec3::Z, Vec3::Z, Vec3::new(0.0, -1.0, 0.0)],
            uvs: vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 1.0)],
            indices: vec![0, 1, 2, 0, 1, 3],
            layers: vec![Layer::new("L", solid_texture(8, 8, [255, 255, 255, 255]))],
            active_layer: 0,
            dirty: None,
        };
        let anchor = Vec3::ZERO;
        let (axis_u, axis_v) = (Vec3::X, Vec3::Y);
        // Sanity: the folded vertex's chord against the anchor plane is zero —
        // this is exactly the collapse the unfold must repair.
        let d = m.positions[3] - anchor;
        assert!(
            (d.dot(axis_u).abs() + d.dot(axis_v).abs()) < 1e-6,
            "folded vertex must sit on the anchor normal (chord zero)"
        );
        let unwrap = surface_unwrap(&m.positions, &m.indices, anchor, axis_u, axis_v, 0, 3.0)
            .expect("open book within radius");
        let hinge0 = unwrap.vertex_phase(0).expect("hinge vertex 0");
        let hinge1 = unwrap.vertex_phase(1).expect("hinge vertex 1");
        let folded = unwrap.vertex_phase(3).expect("folded vertex unfolded");
        // Hinge stays put: phase (0,0) and (2,0).
        assert!(
            hinge0.length() < 1e-4,
            "hinge vertex 0 must stay at the anchor"
        );
        assert!(
            (hinge1 - Vec2::new(2.0, 0.0)).length() < 1e-4,
            "hinge vertex 1 must stay at (2,0)"
        );
        // The folded page opens to (0,-2): world edge lengths to the hinge are
        // preserved, so the pattern runs around the fold instead of through it.
        assert!(
            (folded - Vec2::new(0.0, -2.0)).length() < 1e-3,
            "folded vertex must unfold to (0,-2), got {folded:?}"
        );
        assert!(
            folded.y < 0.0,
            "folded vertex must open away from the flat page, not fold onto it"
        );
        let ts = unwrap.tri(1).expect("folded triangle unfolded");
        assert_eq!(ts, [hinge0, hinge1, folded]);
    }

    #[test]
    fn stamp_on_a_crease_paints_only_within_the_true_radius() {
        // Two flat quads joined at 90° along a shared edge (a dihedron). A dab
        // centered on the hinge: `local_surface_normal` blends the two face
        // normals (~45° tilt), so a round mask measuring the *projected*
        // in-plane distance paints texels far past the dab — the "whole
        // triangle, where the brush never touched" report on cubes. The round
        // mask must measure the true 3D radius (the GPU cursor's sphere test),
        // so no texel beyond it paints on either quad.
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
        // Quad A: z=0 plane, x∈[-1,1], y∈[-1,1]. Its top edge (v3-v2) is the
        // hinge along x at y=1.
        let a = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
        push_quad(
            &mut m,
            [
                Vec3::new(a[0].0, a[0].1, 0.0),
                Vec3::new(a[1].0, a[1].1, 0.0),
                Vec3::new(a[2].0, a[2].1, 0.0),
                Vec3::new(a[3].0, a[3].1, 0.0),
            ],
            [(0.05, 0.0), (0.45, 0.0), (0.45, 1.0), (0.05, 1.0)],
            Vec3::Z,
        );
        // Quad B: y=1 plane, x∈[-1,1], z∈[-1,1], sharing the hinge edge with A
        // (new corners, reused hinge vertices 2/3).
        push_quad(
            &mut m,
            [
                Vec3::new(-1.0, 1.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(1.0, 1.0, 1.0),
                Vec3::new(-1.0, 1.0, 1.0),
            ],
            [(0.55, 0.0), (0.95, 0.0), (0.95, 1.0), (0.55, 1.0)],
            Vec3::new(0.0, 1.0, 0.0),
        );
        let bg = [246, 241, 232, 255];
        let radius = 0.8f32;
        let center = Vec3::new(0.0, 1.0, 0.0); // on the hinge
        let (o, d) = (Vec3::new(0.0, 1.0, 3.0), Vec3::new(0.0, 0.0, -1.0));
        let brush = crate::brush::Brush {
            kind: crate::brush::FootprintKind::Round,
            size: 8.0,
            hardness: 1.0,
            spacing: 0.0,
            opacity: 1.0,
            accumulate: true,
            color: [40, 80, 200, 255],
            mode: StampMode::Paint,
            sprite: None,
            pattern_lock: crate::brush::PatternLock::Dab,
            rotation: 0.0,
            flip_x: false,
            flip_y: false,
            texture_scale: 1.0,
            texture_locked: false,
            texture_size_lock: 0.0,
            texture_window: crate::brush::Window::Round,
        };
        let accel = crate::paint::StampAccel::new(&m, None);
        apply_brush_stamp(
            &mut m,
            center,
            radius,
            o,
            d,
            None,
            &brush,
            Some(&accel),
            None,
            None,
            None,
        );
        // Map a texel back to its 3D point on the quad and check the sphere.
        let on_quad_a = |u: f32, v: f32| Vec3::new((u - 0.25) * 5.0, (v - 0.5) * 2.0, 0.0);
        let on_quad_b = |u: f32, v: f32| Vec3::new((u - 0.75) * 5.0, 1.0, v - 0.5);
        let mut painted_any = false;
        let tw = m.layers[0].texture.width as usize;
        let th = m.layers[0].texture.height as usize;
        for y in 0..th {
            for x in 0..tw {
                let px = texel(&m, x as u32, y as u32);
                if px == bg {
                    continue;
                }
                let uv = uv_from_texel(x as u32, y as u32, tw as u32, th as u32);
                let u = uv.0;
                let v = uv.1;
                let world = if (0.05..0.45).contains(&u) {
                    on_quad_a(u, v)
                } else if (0.55..0.95).contains(&u) {
                    on_quad_b(u, v)
                } else {
                    continue;
                };
                painted_any = true;
                let dist = (world - center).length();
                assert!(
                    dist <= radius + 0.05,
                    "texel ({x},{y}) world {world:?} painted at {dist} > {radius} inside no dab"
                );
            }
        }
        assert!(
            painted_any,
            "the disk around the hinge must paint something"
        );
    }

    #[test]
    fn square_dab_on_a_crease_stays_within_its_true_extent() {
        // A square brush larger than the surrounding faces, dabbed on a 90°
        // hinge. `local_surface_normal` blends the two face normals (~45°), so
        // the brush axis plane tilts and a mask measured at projected (tu, tv)
        // paints each face's texels up to `half / cos45 ≈ 1.13·half` — the
        // square looks stretched past the cursor on both faces. Every texel
        // must instead stay within the square's true extent measured in its
        // OWN face plane.
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
        let a = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
        push_quad(
            &mut m,
            [
                Vec3::new(a[0].0, a[0].1, 0.0),
                Vec3::new(a[1].0, a[1].1, 0.0),
                Vec3::new(a[2].0, a[2].1, 0.0),
                Vec3::new(a[3].0, a[3].1, 0.0),
            ],
            [(0.05, 0.0), (0.45, 0.0), (0.45, 1.0), (0.05, 1.0)],
            Vec3::Z,
        );
        push_quad(
            &mut m,
            [
                Vec3::new(-1.0, 1.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(1.0, 1.0, 1.0),
                Vec3::new(-1.0, 1.0, 1.0),
            ],
            [(0.55, 0.0), (0.95, 0.0), (0.95, 1.0), (0.55, 1.0)],
            Vec3::new(0.0, 1.0, 0.0),
        );
        let bg = [246, 241, 232, 255];
        let half = 0.8f32;
        let center = Vec3::new(0.0, 1.0, 0.0); // on the hinge
        let (o, d) = (Vec3::new(0.0, 1.0, 3.0), Vec3::new(0.0, 0.0, -1.0));
        let brush = crate::brush::Brush {
            kind: crate::brush::FootprintKind::Square,
            size: 8.0,
            hardness: 1.0,
            spacing: 0.0,
            opacity: 1.0,
            accumulate: true,
            color: [40, 80, 200, 255],
            mode: StampMode::Paint,
            sprite: None,
            pattern_lock: crate::brush::PatternLock::Dab,
            rotation: 0.0,
            flip_x: false,
            flip_y: false,
            texture_scale: 1.0,
            texture_locked: false,
            texture_size_lock: 0.0,
            texture_window: crate::brush::Window::Round,
        };
        let accel = StampAccel::new(&m, None);
        apply_brush_stamp(
            &mut m,
            center,
            half,
            o,
            d,
            None,
            &brush,
            Some(&accel),
            None,
            None,
            None,
        );
        let on_quad_a = |u: f32, v: f32| Vec3::new((u - 0.25) * 5.0, (v - 0.5) * 2.0, 0.0);
        let on_quad_b = |u: f32, v: f32| Vec3::new((u - 0.75) * 5.0, 1.0, v - 0.5);
        let mut painted_any = false;
        let tw = m.layers[0].texture.width as usize;
        let th = m.layers[0].texture.height as usize;
        for y in 0..th {
            for x in 0..tw {
                let px = texel(&m, x as u32, y as u32);
                if px == bg {
                    continue;
                }
                let uv = uv_from_texel(x as u32, y as u32, tw as u32, th as u32);
                let (u, v) = (uv.0, uv.1);
                if (0.05..0.45).contains(&u) {
                    let p = on_quad_a(u, v);
                    painted_any = true;
                    let (dx, dy) = (p.x - center.x, p.y - center.y);
                    assert!(
                        dx.abs() <= half + 0.05 && dy.abs() <= half + 0.05,
                        "quad A texel ({x},{y}) painted at face offset ({dx},{dy}) past the square"
                    );
                } else if (0.55..0.95).contains(&u) {
                    let p = on_quad_b(u, v);
                    painted_any = true;
                    let (dx, dz) = (p.x - center.x, p.z - center.z);
                    assert!(
                        dx.abs() <= half + 0.05 && dz.abs() <= half + 0.05,
                        "quad B texel ({x},{y}) painted at face offset ({dx},{dz}) past the square"
                    );
                }
            }
        }
        assert!(
            painted_any,
            "the square around the hinge must paint something"
        );
    }

    #[test]
    fn surface_unwrap_full_mesh_far_side_follows_arc() {
        // A stroke started near one rim of a sphere and dragged across to the
        // other rim must keep running *along the surface*: with a whole-mesh
        // unfold the antipodal ring vertex gets a phase comparable to the
        // half-circumference arc, whereas a dab-sized patch never reaches it
        // and falls back to the anchor-plane chord (~0) — the reported
        // "stretches at the far edge; fine from the center" regression.
        // The unfold is a greedy tree flattening, so it cannot hit the ideal
        // pi*R on a closed positive-curvature sphere (no isometric flattening
        // exists); the invariant we hold is "multi-radius, definitively not
        // collapsed, whole mesh covered".
        let m = MeshData::uv_sphere(1.0, 24, 32);
        let ring_verts = 32 + 1;
        let anchor_i = 12 * ring_verts; // equator, phi = 0 -> (1, 0, 0)
        let far_i = 12 * ring_verts + 16; // equator, phi = pi -> (-1, 0, 0)
        let anchor = m.positions[anchor_i];
        let (axis_u, axis_v) = (Vec3::Z, Vec3::new(0.0, -1.0, 0.0));
        // Seed triangle: quad (12, 0), whose first triangle shares the anchor
        // vertex, so the unfold fans out around the equator ring.
        let anchor_tri = 12 * (32 * 2);
        let unwrap = surface_unwrap(
            &m.positions,
            &m.indices,
            anchor,
            axis_u,
            axis_v,
            anchor_tri,
            f32::INFINITY,
        )
        .expect("whole sphere unfolds");
        for v in 0..m.positions.len() {
            assert!(
                unwrap.vertex_phase(v as u32).is_some(),
                "whole-mesh unfold must reach every vertex, missing {v}"
            );
        }
        let ch = |v: Vec3| Vec2::new((v - anchor).dot(axis_u), (v - anchor).dot(axis_v));
        let far = m.positions[far_i];
        let chord = ch(far);
        assert!(
            chord.x.abs() < 1e-3 && chord.y.abs() < 1e-3,
            "antipode chord must vanish on the anchor plane, got {chord:?}"
        );
        let ph = unwrap.vertex_phase(far_i as u32).unwrap();
        assert!(
            ph.length() > 1.0 && ph.length() < 4.0,
            "far-side vertex must keep a surface-following phase (~arc length), got {ph:?} (chord {chord:?})"
        );
        // The immediate neighbor of the anchor is only one ring away, so its
        // phase is a short exact arc — this is the geodesic (vs chord) core of
        // the feature and must hold precisely.
        let near_i = 12 * ring_verts + 1;
        let near_ph = unwrap.vertex_phase(near_i as u32).unwrap();
        let near_chord = ch(m.positions[near_i]);
        assert!(
            (near_ph - near_chord).length() < 0.1,
            "one-step neighbor must be near its chord segment root, got {near_ph:?} vs {near_chord:?}"
        );
    }

    #[test]
    fn surface_unwrap_detached_tilted_panel_develops_isometrically_not_stretched() {
        // A panel glued to the mesh as a separate component (cracked seam /
        // separate object) is never reached by the geodesic flood. Projecting
        // it onto the click plane — the old chord fallback — collapses a panel
        // standing perpendicular to the anchor plane into a line (infinite
        // stretch, "the whole line over the triangle"). The unfold must
        // instead develop it in its own plane: every phase edge equals its
        // world edge, so the pattern on the panel stays world-uniform.
        let mut m = MeshData {
            positions: vec![],
            normals: vec![],
            uvs: vec![],
            indices: vec![],
            layers: vec![Layer::new(
                "Layer 1",
                solid_texture(32, 32, [255, 255, 255, 255]),
            )],
            active_layer: 0,
            dirty: None,
        };
        // Click panel: the z=0 plane (anchor axes X/Y). Triangles 0-1.
        push_quad(
            &mut m,
            [
                Vec3::ZERO,
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(2.0, 2.0, 0.0),
                Vec3::new(0.0, 2.0, 0.0),
            ],
            [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            Vec3::Z,
        );
        // Detached panel: the y=0 plane, perpendicular to the anchor plane and
        // sharing no vertex or edge with the click panel (vertices 4-7,
        // triangles 2-3). The old chord put every phase on the x-axis —
        // collapsed the panel's height to zero.
        push_quad(
            &mut m,
            [
                Vec3::new(2.0, 0.0, -1.0),
                Vec3::new(4.0, 0.0, -1.0),
                Vec3::new(4.0, 0.0, 1.0),
                Vec3::new(2.0, 0.0, 1.0),
            ],
            [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            Vec3::Y,
        );
        let unwrap = surface_unwrap(
            &m.positions,
            &m.indices,
            Vec3::ZERO,
            Vec3::X,
            Vec3::Y,
            0,
            f32::INFINITY,
        )
        .expect("two-component mesh unfolds");
        // Every triangle carries phases now — the detached panel is developed,
        // no triangle keeps the stretched chord fallback.
        assert!(
            (0..m.indices.len() / 3).all(|t| unwrap.tri(t).is_some()),
            "the detached panel must get an isometric development"
        );
        // Per-triangle isometry: each phase edge equals its world edge.
        for (t, tri) in m.indices.chunks_exact(3).enumerate() {
            let tp = unwrap.tri(t).unwrap();
            for k in 0..3 {
                let (a, b) = (tri[k], tri[(k + 1) % 3]);
                let world = (m.positions[a as usize] - m.positions[b as usize]).length();
                let phase = (tp[k] - tp[(k + 1) % 3]).length();
                assert!(
                    (phase - world).abs() < 1e-3,
                    "triangle {t} edge {k} stretches: phase {phase} vs world {world}"
                );
            }
        }
        // The perpendicular panel keeps its own height: the corner (2,0,1) is
        // exactly 2 units from the base (2,0,-1), and so must their phases be;
        // the old chord collapsed that span to zero.
        let base_ph = unwrap.vertex_phase(4).expect("panel base phased");
        let corner_ph = unwrap.vertex_phase(7).expect("panel corner phased");
        assert!(
            ((corner_ph - base_ph).length() - 2.0).abs() < 1e-3,
            "perpendicular panel keeps its own height, got {} (chord collapsed it to 0)",
            (corner_ph - base_ph).length()
        );
    }

    #[test]
    fn surface_unwrap_phase_is_single_valued_across_incident_triangles() {
        // Diagnostic: the greedy BFS must assign each vertex ONE phase. If a
        // later triangle overwrites an already-assigned far vertex, incident
        // triangles disagree at that vertex and the stamp interpolates a
        // discontinuous (seamed) pattern field.
        let m = MeshData::uv_sphere(1.0, 24, 32);
        let ring_verts = 32 + 1;
        let anchor_i = 12 * ring_verts;
        let anchor = m.positions[anchor_i];
        let (axis_u, axis_v) = (Vec3::Z, Vec3::new(0.0, -1.0, 0.0));
        let anchor_tri = 12 * (32 * 2);
        let unwrap = surface_unwrap(
            &m.positions,
            &m.indices,
            anchor,
            axis_u,
            axis_v,
            anchor_tri,
            f32::INFINITY,
        )
        .expect("whole sphere unfolds");
        let mut mismatches = 0usize;
        for (ti, tri) in m.indices.chunks_exact(3).enumerate() {
            let Some(tp) = unwrap.tri(ti) else { continue };
            for k in 0..3 {
                let v = tri[k];
                if let Some(ph) = unwrap.vertex_phase(v) {
                    if (tp[k] - ph).length() > 1e-4 {
                        mismatches += 1;
                    }
                }
            }
        }
        assert_eq!(
            mismatches, 0,
            "phase field is multivalued at {mismatches} incident vertex copies (seams)"
        );
    }

    #[test]
    fn surface_unwrap_flat_strip_probe() {
        // Probe: does a flat coplanar strip (skinny triangles) scramble the
        // phase past the two-circle's side pick? Every non-curved vertex must
        // still equal its anchor-plane chord.
        for (cols, rows, scale) in [(12usize, 2usize, 0.01f32), (12, 6, 0.05), (12, 6, 0.5)] {
            let mut m = MeshData {
                positions: vec![],
                normals: vec![],
                uvs: vec![],
                indices: vec![],
                layers: vec![Layer::new("L", solid_texture(8, 8, [255, 255, 255, 255]))],
                active_layer: 0,
                dirty: None,
            };
            for r in 0..=rows {
                for c in 0..=cols {
                    m.positions.push(Vec3::new(c as f32, r as f32 * scale, 0.0));
                    m.uvs.push((0.0, 0.0));
                    m.normals.push(Vec3::Z);
                }
            }
            let w = cols + 1;
            for r in 0..rows {
                for c in 0..cols {
                    let a = (r * w + c) as u32;
                    let b = (r * w + c + 1) as u32;
                    let d = ((r + 1) * w + c) as u32;
                    let e = ((r + 1) * w + c + 1) as u32;
                    m.indices.extend_from_slice(&[a, b, e, a, e, d]);
                }
            }
            let center = Vec3::new(cols as f32 * 0.5, rows as f32 * scale * 0.5, 0.0);
            let unwrap = surface_unwrap(
                &m.positions,
                &m.indices,
                center,
                Vec3::X,
                Vec3::Y,
                0,
                f32::INFINITY,
            )
            .expect("strip unfolds");
            let mut worst = 0.0f32;
            let mut bad = 0usize;
            for (v, p) in m.positions.iter().enumerate() {
                let ph = unwrap.vertex_phase(v as u32).unwrap();
                let want = Vec2::new(p.x - center.x, p.y - center.y);
                let d = (ph - want).length();
                worst = worst.max(d);
                if d > 1e-3 {
                    bad += 1;
                }
            }
            assert!(
                worst < 1e-3,
                "strip {cols}x{rows} scale {scale}: worst {worst}, bad {bad} of {}",
                m.positions.len()
            );
        }
        // Plain coplanar fan, no crack.
        let mut f = MeshData {
            positions: vec![Vec3::ZERO],
            normals: vec![],
            uvs: vec![(0.0, 0.0)],
            indices: vec![],
            layers: vec![Layer::new("L", solid_texture(8, 8, [255, 255, 255, 255]))],
            active_layer: 0,
            dirty: None,
        };
        for k in 0..24u32 {
            let a = k as f32 * std::f32::consts::TAU / 24.0;
            f.positions
                .push(Vec3::new(8.0 * a.cos(), 8.0 * a.sin(), 0.0));
            f.normals.push(Vec3::Z);
            f.uvs.push((0.0, 0.0));
        }
        for k in 0..24u32 {
            f.indices.extend_from_slice(&[0, k + 1, (k + 1) % 24 + 1]);
        }
        let unwrap = surface_unwrap(
            &f.positions,
            &f.indices,
            Vec3::ZERO,
            Vec3::X,
            Vec3::Y,
            0,
            f32::INFINITY,
        )
        .expect("fan unfolds");
        let (mut worst, mut bad) = (0.0f32, 0usize);
        for (v, p) in f.positions.iter().enumerate() {
            let ph = unwrap.vertex_phase(v as u32).unwrap_or_default();
            let want = Vec2::new(p.x, p.y);
            let d = (ph - want).length();
            worst = worst.max(d);
            if d > 1e-3 {
                bad += 1;
            }
        }
        assert!(
            worst < 1e-3,
            "plain coplanar fan: worst {worst}, bad {bad} of {}",
            f.positions.len()
        );
        // Flat wall reached across a curved crease: a half-cylinder lip of
        // radius 0.2 along x=0 rides above a flat coplanar wall x∈[0,10].
        // Anchor sits on the lip; every wall vertex must still land on its
        // anchor-plane chord (x, y).
        let mut w = MeshData {
            positions: vec![],
            normals: vec![],
            uvs: vec![],
            indices: vec![],
            layers: vec![Layer::new("L", solid_texture(8, 8, [255, 255, 255, 255]))],
            active_layer: 0,
            dirty: None,
        };
        let segs = 24u32;
        let bands = 3u32;
        let wcols = 12u32;
        for b in 0..=bands {
            for s in 0..=segs {
                let a = (s as f32 / segs as f32) * std::f32::consts::PI;
                let y = b as f32 * 0.7 - bands as f32 * 0.35;
                w.positions.push(Vec3::new(0.2 * a.cos(), y, 0.2 * a.sin()));
                w.uvs.push((0.0, 0.0));
                w.normals.push(Vec3::Z);
            }
        }
        let n0 = segs + 1;
        for b in 0..bands {
            for s in 0..segs {
                let a = b * n0 + s;
                w.indices
                    .extend_from_slice(&[a, a + 1, a + n0, a + n0, a + 1, a + n0 + 1]);
            }
        }
        // Now the flat wall: it SHARES the lip's s=0 column (x=0.2, z=0) so
        // the BFS can cross the crease, then extends to x=10 in the z=0 plane.
        let lip_row = |r: u32| r * (segs + 1); // vertex id of lip row r, col 0
        let mut wid = Vec::new(); // wall vertex id for (r, c)
        for r in 0..=bands {
            for c in 0..=wcols {
                if c == 0 {
                    wid.push(lip_row(r));
                } else {
                    let y = r as f32 * 0.7 - bands as f32 * 0.35;
                    w.positions.push(Vec3::new(
                        0.2 + c as f32 * (10.0 - 0.2) / wcols as f32,
                        y,
                        0.0,
                    ));
                    w.uvs.push((0.0, 0.0));
                    w.normals.push(Vec3::Z);
                    wid.push((w.positions.len() - 1) as u32);
                }
            }
        }
        let wcnt = wcols + 1;
        for r in 0..bands {
            for c in 0..wcols {
                let (a, b, d, e) = (
                    wid[(r * wcnt + c) as usize],
                    wid[(r * wcnt + c + 1) as usize],
                    wid[((r + 1) * wcnt + c) as usize],
                    wid[((r + 1) * wcnt + c + 1) as usize],
                );
                w.indices.extend_from_slice(&[a, b, e, a, e, d]);
            }
        }
        let wall_first = wid[0];
        let wall_only_start = wid[1]; // first vertex pushed exclusively for the wall
        let anchor = Vec3::new(0.2, 0.0, 0.0);
        let seed = 0; // lip triangle touching the shared crease column
        let unwrap = surface_unwrap(
            &w.positions,
            &w.indices,
            anchor,
            Vec3::X,
            Vec3::Y,
            seed,
            f32::INFINITY,
        )
        .expect("lip+wall unfolds");
        let (mut worst, mut bad) = (0.0f32, 0usize);
        for (v, p) in w.positions.iter().enumerate() {
            if (v as u32) < wall_only_start {
                continue; // lip is curved; only check the flat wall's own vertices
            }
            let ph = unwrap.vertex_phase(v as u32).unwrap_or_default();
            let want = Vec2::new(p.x - anchor.x, p.y - anchor.y);
            let d = (ph - want).length();
            worst = worst.max(d);
            if d > 1e-3 {
                bad += 1;
            }
        }
        assert!(
            worst < 0.05,
            "wall reached across a curved lip: worst {worst}, bad {bad} of {}",
            w.positions.len() - wall_first as usize
        );
        // Reflex crease: an overhanging wall leans back OVER the floor its
        // edges live on, so across the shared crease edge the wall's far
        // vertices sit on the SAME world side as the floor's (folded back).
        // The unfold must lay them on the floor's own phase side (negative x),
        // never mirror them onto positive x.
        let mut r = MeshData {
            positions: vec![],
            normals: vec![],
            uvs: vec![],
            indices: vec![],
            layers: vec![Layer::new("L", solid_texture(8, 8, [255, 255, 255, 255]))],
            active_layer: 0,
            dirty: None,
        };
        let (fc, fr) = (2u32, 2u32); // floor: 3 cols x∈[-1,0], 3 rows y∈[-1,1]
        for i in 0..=fr {
            for j in 0..=fc {
                let y = i as f32 - 1.0;
                let x = j as f32 * 0.5 - 1.0;
                r.positions.push(Vec3::new(x, y, 0.0));
                r.uvs.push((0.0, 0.0));
                r.normals.push(Vec3::Z);
            }
        }
        let fw = fc + 1;
        for i in 0..fr {
            for j in 0..fc {
                let (a, b, d, e) = (
                    (i * fw + j),
                    (i * fw + j + 1),
                    ((i + 1) * fw + j),
                    ((i + 1) * fw + j + 1),
                );
                r.indices.extend_from_slice(&[a, e, b, a, d, e]);
            }
        }
        // Leaning wall shares the floor's last column (x=0) and reaches back
        // over it: col c sits at (-0.5c, y, 0.5c).
        let last_col = |i: u32| i * fw + fc;
        let mut rid = Vec::new();
        for i in 0..=fr {
            for c in 0..=2u32 {
                if c == 0 {
                    rid.push(last_col(i));
                } else {
                    let y = i as f32 - 1.0;
                    r.positions
                        .push(Vec3::new(-0.5 * c as f32, y, 0.5 * c as f32));
                    r.uvs.push((0.0, 0.0));
                    r.normals.push(Vec3::Z);
                    rid.push((r.positions.len() - 1) as u32);
                }
            }
        }
        for i in 0..fr {
            for c in 0..2u32 {
                let (a, b, d, e) = (
                    rid[(i * 3 + c) as usize],
                    rid[(i * 3 + c + 1) as usize],
                    rid[((i + 1) * 3 + c) as usize],
                    rid[((i + 1) * 3 + c + 1) as usize],
                );
                r.indices.extend_from_slice(&[a, e, b, a, d, e]);
            }
        }
        let r_anchor = Vec3::new(0.0, 0.0, 0.0);
        let r_unwrap = surface_unwrap(
            &r.positions,
            &r.indices,
            r_anchor,
            Vec3::X,
            Vec3::Y,
            0,
            f32::INFINITY,
        )
        .expect("reflex crease unfolds");
        let (mut worst, mut bad) = (0.0f32, 0usize);
        for (v, p) in r.positions.iter().enumerate() {
            if (v as u32) <= last_col(fr) {
                continue; // floor only; check the overhanging wall
            }
            let ph = r_unwrap.vertex_phase(v as u32).unwrap_or_default();
            let d_crease = (p.x * p.x + p.z * p.z).sqrt(); // body distance from the crease column
            let want = Vec2::new(-d_crease, p.y);
            let d = (ph - want).length();
            worst = worst.max(d);
            if d > 1e-2 {
                bad += 1;
            }
        }
        assert!(
            worst < 1e-2,
            "folded-back wall must unfold onto the floor side (not mirror): worst {worst}, bad {bad} of {}",
            r.positions.len() - (last_col(fr) + 1) as usize
        );
    }

    #[test]
    #[ignore = "performance probe: cargo test --release -- --ignored bench_stamp"]
    fn bench_stamp_dab_scan() {
        // Per-dab cost on a large grid plane with a small brush: previously the
        // texel loop scanned every triangle of the mesh per dab; now the stroke
        // BVH narrows each dab to the triangles near its sphere.
        let n = 96usize;
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
        for j in 0..n {
            for i in 0..n {
                let (u0, u1) = (i as f32 / n as f32, (i + 1) as f32 / n as f32);
                let (v0, v1) = (j as f32 / n as f32, (j + 1) as f32 / n as f32);
                push_quad(
                    &mut m,
                    [
                        Vec3::new(-1.0 + 2.0 * u0, -1.0 + 2.0 * v0, 0.0),
                        Vec3::new(-1.0 + 2.0 * u1, -1.0 + 2.0 * v0, 0.0),
                        Vec3::new(-1.0 + 2.0 * u1, -1.0 + 2.0 * v1, 0.0),
                        Vec3::new(-1.0 + 2.0 * u0, -1.0 + 2.0 * v1, 0.0),
                    ],
                    [(u0, v0), (u1, v0), (u1, v1), (u0, v1)],
                    Vec3::Z,
                );
            }
        }
        let tris = m.indices.len() / 3;
        let accel = StampAccel::new(&m, None);
        let brush = crate::brush::Brush {
            kind: crate::brush::FootprintKind::Round,
            size: 8.0,
            hardness: 1.0,
            spacing: 0.0,
            opacity: 1.0,
            accumulate: true,
            color: [40, 80, 200, 255],
            mode: StampMode::Paint,
            sprite: None,
            pattern_lock: crate::brush::PatternLock::Dab,
            rotation: 0.0,
            flip_x: false,
            flip_y: false,
            texture_scale: 1.0,
            texture_locked: false,
            texture_size_lock: 0.0,
            texture_window: crate::brush::Window::Round,
        };
        let (o, d) = (Vec3::new(0.0, 0.0, 3.0), Vec3::new(0.0, 0.0, -1.0));
        let mut run = |radius: f32| {
            let t = std::time::Instant::now();
            let dabs = 400usize;
            for k in 0..dabs {
                let cx = -0.9 + (k % 37) as f32 * 0.05;
                let cy = -0.9 + (k % 29) as f32 * 0.05;
                apply_brush_stamp(
                    &mut m,
                    Vec3::new(cx, cy, 0.0),
                    radius,
                    o,
                    d,
                    None,
                    &brush,
                    Some(&accel),
                    None,
                    None,
                    None,
                );
            }
            let dt = t.elapsed();
            println!(
                "r={radius}: {dabs} dabs on {tris} tris: {:?} per dab",
                dt / dabs as u32
            );
        };
        // Small local dab (the interactive case) and a whole-face dab.
        run(0.06);
        run(2.0);
        assert!(m.layers[0].texture.rgba.iter().any(|&a| a != 0), "sanity");
    }

    #[test]
    fn texture_dab_never_paints_past_the_brush_sphere() {
        // Regression for the "dragging a texture brush to the edges widens /
        // stretches it" reports. A rubber-stamp texture dab (the default Round
        // paint window) masks by the WORLD sphere exactly like the GPU cursor,
        // not by a face-plane disc: on a curved surface or a crease the old
        // disc dilated the footprint past the brush circle (measured ~1.3×R on
        // a sphere's silhouette and up to ~1.4×R across a crease), the sprite
        // reading its rim texels over too much surface. Every painted texel
        // must sit within the brush sphere on flat ground, curved silhouette,
        // pole and crease alike — then the mark matches the cursor circle and
        // the pattern can no longer stretch past it.
        let bg = [0, 0, 0, 255];
        let mk_flat = || {
            let mut m = MeshData {
                positions: vec![],
                normals: vec![],
                uvs: vec![],
                indices: vec![],
                layers: vec![Layer::new("Layer 1", solid_texture(64, 64, bg))],
                active_layer: 0,
                dirty: None,
            };
            push_quad(
                &mut m,
                [
                    Vec3::new(-1.0, -1.0, 0.0),
                    Vec3::new(1.0, -1.0, 0.0),
                    Vec3::new(1.0, 1.0, 0.0),
                    Vec3::new(-1.0, 1.0, 0.0),
                ],
                [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
                Vec3::Z,
            );
            m
        };
        let mk_dihedron = || {
            let mut m = MeshData {
                positions: vec![],
                normals: vec![],
                uvs: vec![],
                indices: vec![],
                layers: vec![Layer::new("Layer 1", solid_texture(64, 64, bg))],
                active_layer: 0,
                dirty: None,
            };
            push_quad(
                &mut m,
                [
                    Vec3::new(-1.0, -1.0, 0.0),
                    Vec3::new(1.0, -1.0, 0.0),
                    Vec3::new(1.0, 1.0, 0.0),
                    Vec3::new(-1.0, 1.0, 0.0),
                ],
                [(0.05, 0.0), (0.45, 0.0), (0.45, 1.0), (0.05, 1.0)],
                Vec3::Z,
            );
            push_quad(
                &mut m,
                [
                    Vec3::new(-1.0, 1.0, 0.0),
                    Vec3::new(1.0, 1.0, 0.0),
                    Vec3::new(1.0, 1.0, 1.0),
                    Vec3::new(-1.0, 1.0, 1.0),
                ],
                [(0.55, 0.0), (0.95, 0.0), (0.95, 1.0), (0.55, 1.0)],
                Vec3::new(0.0, 1.0, 0.0),
            );
            m
        };
        let sprite = TextureData {
            width: 8,
            height: 8,
            rgba: vec![255u8; 8 * 8 * 4],
        };
        let brush = crate::brush::Brush {
            kind: crate::brush::FootprintKind::Sprite,
            size: 0.5,
            hardness: 0.0,
            spacing: 0.0,
            opacity: 1.0,
            accumulate: false,
            color: [255, 0, 0, 255],
            mode: StampMode::Paint,
            sprite: Some(sprite),
            pattern_lock: crate::brush::PatternLock::Dab,
            rotation: 0.0,
            flip_x: false,
            flip_y: false,
            texture_scale: 1.0,
            texture_locked: false,
            texture_size_lock: 0.0,
            texture_window: crate::brush::Window::Round,
        };
        let probe =
            |m: &mut MeshData, center: Vec3, radius: f32, eye: Vec3, dir: Vec3, label: &str| {
                apply_brush_stamp(
                    m, center, radius, eye, dir, None, &brush, None, None, None, None,
                );
                let idx = m.active_layer;
                let mut max_dist = 0.0f32;
                for y in 0..64u32 {
                    for x in 0..64u32 {
                        let i = (y * 64 + x) as usize * 4;
                        if m.layers[idx].texture.rgba[i] > 0 {
                            // The stamp wrote this texel from SOME triangle whose UV
                            // holds it (a seam column can be shared by two). Any one
                            // of those is a legitimate reading of the texel's surface
                            // sample, so it is only a violation when EVERY holding
                            // triangle sits outside the brush sphere.
                            let mut closest = f32::INFINITY;
                            for (b0, b1, tri) in uv_hits(m, x, y) {
                                closest =
                                    closest.min((pos_3d_at(m, tri, b0, b1) - center).length());
                            }
                            max_dist = max_dist.max(closest);
                        }
                    }
                }
                // Allow a small epsilon for the spherical cap's rim texels whose
                // centers sit a fraction of a texel past the circle; the previous
                // behavior measured 1.3×–1.4× the radius here.
                assert!(
                    max_dist <= radius * 1.08,
                    "{label}: texture dab painted up to {max_dist}; brush is {radius}"
                );
            };
        probe(
            &mut mk_flat(),
            Vec3::ZERO,
            0.5,
            Vec3::new(0.0, 0.0, 2.0),
            Vec3::new(0.0, 0.0, -1.0),
            "flat",
        );
        probe(
            &mut mk_flat(),
            Vec3::new(-0.35, 0.0, 0.0),
            0.5,
            Vec3::new(0.0, 0.0, 2.0),
            Vec3::new(0.0, 0.0, -1.0),
            "flat-near-rim",
        );
        probe(
            &mut crate::io::MeshData::uv_sphere(0.6, 24, 32)
                .with_texture(solid_texture(64, 64, bg)),
            Vec3::new(0.6, 0.0, 0.0),
            0.3,
            Vec3::new(2.4, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            "sphere-side",
        );
        probe(
            &mut crate::io::MeshData::uv_sphere(0.6, 24, 32)
                .with_texture(solid_texture(64, 64, bg)),
            Vec3::new(0.0, 0.6, 0.0),
            0.3,
            Vec3::new(0.0, 2.4, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            "sphere-pole",
        );
        probe(
            &mut mk_dihedron(),
            Vec3::new(0.0, 1.0, 0.0),
            0.6,
            Vec3::new(0.0, 1.0, 3.0),
            Vec3::new(0.0, 0.0, -1.0),
            "crease",
        );
    }

    #[test]
    fn rubber_stamp_texture_paint_is_camera_independent_on_a_flat_face() {
        // The texture dab's pattern phase must lie in each face's plane,
        // anchored to the surface's own direction (world-stable frame:
        // per-triangle `T = cross(up, n)`, `B = cross(n, T)`) — NOT projected
        // from the brush/camera axes. So painting the same dab on the same
        // flat plane from two different view directions must produce
        // identical texels: the pattern follows the surface, not the camera.
        // A quad tilted in the WORLD (pitched about the in-plane X axis). On a
        // flat +Z face any camera-roll is projected away so the camera-derived
        // and the world-stable frames coincide; a tilted face makes the
        // difference visible.
        let c = 3.0_f32.sqrt() / 2.0;
        let s3 = 0.5;
        let mk = |label: &str| {
            let mut m = MeshData {
                positions: vec![],
                normals: vec![],
                uvs: vec![],
                indices: vec![],
                layers: vec![Layer::new(label, solid_texture(64, 64, [10, 10, 10, 255]))],
                active_layer: 0,
                dirty: None,
            };
            push_quad(
                &mut m,
                [
                    Vec3::new(-1.0, -c, -s3),
                    Vec3::new(1.0, -c, -s3),
                    Vec3::new(1.0, c, s3),
                    Vec3::new(-1.0, c, s3),
                ],
                [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
                Vec3::new(0.0, -s3, c),
            );
            m
        };
        // 2x2 checker sprite (4x4 pixel cells) so the read phase is visible.
        let mut rgba = vec![0u8; 8 * 8 * 4];
        for y in 0..8u32 {
            for x in 0..8u32 {
                let i = ((y * 8 + x) * 4) as usize;
                let cell = (x / 4 + y / 4) % 2 == 0;
                rgba[i..i + 4].copy_from_slice(&[255, 255, 255, if cell { 255 } else { 0 }]);
            }
        }
        let brush = crate::brush::Brush {
            kind: crate::brush::FootprintKind::Sprite,
            size: 0.5,
            hardness: 0.0,
            spacing: 0.0,
            opacity: 1.0,
            accumulate: false,
            color: [255, 255, 255, 255],
            mode: StampMode::Paint,
            sprite: Some(TextureData {
                width: 8,
                height: 8,
                rgba,
            }),
            pattern_lock: crate::brush::PatternLock::Dab,
            rotation: 0.0,
            flip_x: false,
            flip_y: false,
            texture_scale: 1.0,
            texture_locked: false,
            texture_size_lock: 0.0,
            texture_window: crate::brush::Window::Round,
        };
        let stamp = |eye: Vec3, dir: Vec3| {
            let mut m = mk("paint");
            crate::paint::apply_brush_stamp(
                &mut m,
                Vec3::ZERO,
                brush.size,
                eye,
                dir,
                None,
                &brush,
                None,
                None,
                None,
                None,
            );
            m
        };
        let head_on = stamp(Vec3::new(0.0, 0.0, 3.0), Vec3::new(0.0, 0.0, -1.0));
        let from_the_side = stamp(
            Vec3::new(1.8, 1.1, 3.0),
            Vec3::new(-1.8, -1.1, -3.0).normalize(),
        );
        assert!(
            head_on.layers[0].texture.rgba.iter().any(|&v| v > 100),
            "the dab must actually paint"
        );
        assert_eq!(
            head_on.layers[0].texture.rgba, from_the_side.layers[0].texture.rgba,
            "swapping the view must not change the painted pattern: the \
             rubber-stamp sprite read is anchored to the surface, not the camera"
        );
    }

    /// Every triangle whose UV (texel-center frac coords) holds `uv`, with the
    /// barycentric coords into it (test scaffolding for the sphere-bound probe).
    fn uv_hits(m: &MeshData, x: u32, y: u32) -> Vec<(f32, f32, usize)> {
        // Texel CENTER — the same point the stamp rasters (`uv_from_texel`).
        let uv = ((x as f32 + 0.5) / 64.0, (y as f32 + 0.5) / 64.0);
        let mut out = Vec::new();
        for tri in 0..m.indices.len() / 3 {
            let (i0, i1, i2) = (
                m.indices[tri * 3] as usize,
                m.indices[tri * 3 + 1] as usize,
                m.indices[tri * 3 + 2] as usize,
            );
            let (t0, t1, t2) = (
                Vec2::new(m.uvs[i0].0, m.uvs[i0].1),
                Vec2::new(m.uvs[i1].0, m.uvs[i1].1),
                Vec2::new(m.uvs[i2].0, m.uvs[i2].1),
            );
            let p = Vec2::new(uv.0, uv.1);
            if let Some((b0, b1)) = uv_barycentric(p, t0, t1, t2) {
                let b2 = 1.0 - b0 - b1;
                if b0 >= -1e-4 && b1 >= -1e-4 && b2 >= -1e-4 {
                    out.push((b0, b1, tri));
                }
            }
        }
        out
    }

    fn pos_3d_at(m: &MeshData, tri: usize, b0: f32, b1: f32) -> Vec3 {
        let (i0, i1, i2) = (
            m.indices[tri * 3] as usize,
            m.indices[tri * 3 + 1] as usize,
            m.indices[tri * 3 + 2] as usize,
        );
        let (a, b, c) = (m.positions[i0], m.positions[i1], m.positions[i2]);
        a + (b - a) * b0 + (c - a) * b1
    }
}
