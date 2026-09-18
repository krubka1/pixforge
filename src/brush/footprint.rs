//! Pure footprint geometry for brush dabs.
//!
//! A footprint is a shape evaluated in the *brush-local plane* — the tangent
//! plane through the stamp center, spanned by the `(axis_u, axis_v)` frame that
//! `paint::brush_axes` produces. Both stampers agree on this plane: the 3D
//! stamp projects each texel's world position onto it, the 2D stamp treats
//! texel coordinates directly as plane coordinates, and the GPU cursor overlay
//! samples the same footprint so the preview always equals the output.
//!
//! This module is deliberately geometry-free: it knows only `glam::Vec2` plane
//! coordinates and a coverage mask, so every instrument here is unit testable
//! without a `MeshData` or a GPU.

use glam::Vec2;

use crate::io::TextureData;

/// The footprint families a brush can produce.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FootprintKind {
    Round,
    Square,
    Diamond,
    Rect,
    Sprite,
}

impl FootprintKind {
    /// Stable serialization code for the persisted config (legacy `BrushShape
    /// as u8`: Round 0, Square 1, Diamond 2, Texture 3 — `Texture` became
    /// `Sprite`, `Rect` shares the `Texture` slot since the rect *tool* never
    /// stored a shape).
    pub fn shape_code(self) -> u8 {
        match self {
            FootprintKind::Round => 0,
            FootprintKind::Square => 1,
            FootprintKind::Diamond => 2,
            FootprintKind::Rect | FootprintKind::Sprite => 3,
        }
    }

    /// Inverse of [`FootprintKind::shape_code`]; unknown codes fall back to
    /// `Round` (matching the legacy loader).
    pub fn from_shape_code(code: u8) -> FootprintKind {
        match code {
            0 => FootprintKind::Round,
            1 => FootprintKind::Square,
            2 => FootprintKind::Diamond,
            _ => FootprintKind::Sprite,
        }
    }

    /// Human-readable name for UI labels.
    pub fn label(self) -> &'static str {
        match self {
            FootprintKind::Round => "Round",
            FootprintKind::Square => "Square",
            FootprintKind::Diamond => "Diamond",
            FootprintKind::Rect => "Rect",
            FootprintKind::Sprite => "Texture",
        }
    }
}

/// The image backing a [`Footprint::Sprite`]: a coverage mask plus the stamp
/// transform (rotation / flips). The stamp is drawn ONCE per dab, centered on
/// the footprint, its longest side spanning the dab diameter — the sprite's
/// alpha is the coverage.
///
/// This borrows the sprite (a stroke keeps the same texture for every dab) so
/// footprints are cheap to build per dab and never copy the pixel buffer.
#[derive(Clone, Copy, Debug)]
pub struct SpriteStamp<'a> {
    /// White RGB, coverage in alpha (the layout `io::brush_sprite` and
    /// `brushes::parse_gbr` normalize to).
    pub sprite: &'a TextureData,
    /// Rotation in radians applied to the sprite before stamping.
    pub rotation: f32,
    pub flip_x: bool,
    pub flip_y: bool,
}

impl<'a> SpriteStamp<'a> {
    /// Coverage alpha at a brush-local plane point for a dab of `radius`.
    ///
    /// Mirrors the 3D stamp / 2D stamp / GPU cursor sampling exactly: rotate
    /// by `rotation`, apply the flips, map to UV 0..1 (the longest sprite side
    /// spanning the stamp diameter), nearest-texel sample. `None` when the
    /// point is outside the stamped sprite or the sprite texel is fully
    /// transparent.
    pub fn alpha_at(&self, local: Vec2, radius: f32) -> Option<f32> {
        self.sample_text(SampleEdge::Clamp, local, radius)
    }

    /// Like [`SpriteStamp::alpha_at`], but UV coordinates outside `0..=1` wrap
    /// around with `rem_euclid(1.0)` (the fractional-coordinate repeat). A
    /// pattern-locked dab therefore tiles the sprite infinitely across its
    /// paint window — the anchored phase is preserved, so two dabs that
    /// overlap read the exact same wrapped texels instead of re-centering and
    /// smearing the pattern.
    pub fn alpha_at_wrapped(&self, local: Vec2, radius: f32) -> Option<f32> {
        self.sample_text(SampleEdge::Wrap, local, radius)
    }

    fn sample_text(&self, edge: SampleEdge, local: Vec2, radius: f32) -> Option<f32> {
        let spr = &self.sprite;
        let (swi, shi) = (spr.width.max(1), spr.height.max(1));
        let (sw, sh) = (swi as f32, shi as f32);
        let m = sw.max(sh);
        let (ww, wh) = (sw * 2.0 * radius / m, sh * 2.0 * radius / m);
        let (x, y) = if self.rotation != 0.0 {
            let (sr, cr) = self.rotation.sin_cos();
            (local.x * cr - local.y * sr, local.x * sr + local.y * cr)
        } else {
            (local.x, local.y)
        };
        let (rx, ry) = (
            if self.flip_x { -x } else { x },
            if self.flip_y { -y } else { y },
        );
        let mut u = 0.5 + rx / ww;
        let mut v = 0.5 - ry / wh;
        match edge {
            SampleEdge::Clamp => {
                if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
                    return None;
                }
            }
            SampleEdge::Wrap => {
                u = u.rem_euclid(1.0);
                v = v.rem_euclid(1.0);
            }
        }
        let (sx, sy) = (
            ((u * sw).floor() as u32).min(swi - 1),
            ((v * sh).floor() as u32).min(shi - 1),
        );
        let a = spr.rgba[((sy * swi + sx) as usize) * 4 + 3] as f32 / 255.0;
        if a <= 0.0 {
            None
        } else {
            Some(a)
        }
    }
}

/// How a sprite maps UV coordinates that fall outside `0..=1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SampleEdge {
    /// Outside the sprite → `None` (classic rubber-stamp dab).
    Clamp,
    /// Repeat the sprite (pattern-locked dabs tile across the window).
    Wrap,
}

/// The result of evaluating a footprint at one point of the brush-local plane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FootprintSample {
    /// Outside the footprint — stamps nothing.
    Outside,
    /// Normalized feature distance `0..=1` (0 = dab center, 1 = rim). This is
    /// the input to the hardness / eraser falloff curves.
    Distance(f32),
    /// Direct coverage (sprite alpha). Bypasses the falloff curve entirely —
    /// the sprite's alpha IS the coverage.
    Coverage(f32),
}

/// A concrete footprint instance at dab time. Distances are world units (the
/// 3D stamp) or texels (the 2D stamp); the local plane itself is unit-agnostic.
#[derive(Clone, Copy, Debug)]
pub enum Footprint<'a> {
    Round {
        radius: f32,
    },
    /// Axis-aligned square of half-extent `half` — the inscribed reach, not
    /// the circumscribed `half * sqrt(2)` (see [`Footprint::outer_radius`]).
    Square {
        half: f32,
    },
    /// Axis-aligned diamond inscribed in the `half` square.
    Diamond {
        half: f32,
    },
    Rect {
        half_w: f32,
        half_h: f32,
    },
    Sprite {
        stamp: SpriteStamp<'a>,
        radius: f32,
        /// Where the dab paints. [`Window::SpriteBounds`] is the classic
        /// rubber-stamp window (the sprite's own bounds + alpha is what shows);
        /// the shape windows ([`Window::Round`] is what `Brush::pattern_footprint`
        /// produces) clip a pattern-locked dab to a dab-local geometric region
        /// whose content samples the anchored phase with wrapping UVs.
        window: Window,
    },
}

/// The paint window of a [`Footprint::Sprite`]: what region of the dab-local
/// plane the texture stamp applies to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Window {
    /// Classic texture dab: the sprite itself is the paint window (rubber
    /// stamp), sampled with clamped UVs exactly like the historic stampers.
    /// This is what [`Footprint::for_dab`] and `Brush::footprint` produce.
    SpriteBounds,
    /// Pattern-locked dab: a soft-edged disc of `radius` is the paint window
    /// (see [`Window::mask`] — flat core, C¹ skirt), and the sprite's UV
    /// **wraps** so the anchored pattern tiles infinitely across the stroke
    /// area instead of clipping to one sprite box. Overlapping pattern dabs
    /// therefore blend into a continuous world-locked stroke with no crisp
    /// dab-boundary arcs.
    Round,
    /// Pattern-locked dab clipped to the inscribed `radius` square, soft-edged
    /// like `Round`.
    #[allow(dead_code)] // reserved for future pattern windows
    Square,
    /// Pattern-locked dab clipped to the inscribed diamond, soft-edged like
    /// `Round`.
    #[allow(dead_code)] // reserved for future pattern windows
    Diamond,
}

impl Window {
    /// Smooth, value-continuous window coverage. `1.0` across the flat core
    /// (up to half the radius), then a C¹ smoothstep skirt down to `0.0` at
    /// the rim. The flat core keeps the max-combined envelope of consecutive
    /// dabs level between centers (no density dip); the skirt makes adjacent
    /// dab masks overlap within their own support and melt into a continuous
    /// stroke edge instead of a chain-link of crisp coin arcs.
    fn mask(self, local: Vec2, radius: f32) -> f32 {
        let d = match self {
            // SpriteBounds windows are bounded by the sprite's own UV box,
            // never masked here.
            Window::SpriteBounds => return 1.0,
            Window::Round => local.length(),
            Window::Square => local.x.abs().max(local.y.abs()),
            Window::Diamond => local.x.abs() + local.y.abs(),
        };
        let t = (d / radius).clamp(0.0, 1.0);
        if t <= 0.5 {
            1.0
        } else {
            // Inverted smoothstep over [0.5, 1]: `1 - x²(3-2x)`.
            let x = (t - 0.5) * 2.0;
            1.0 - x * x * (3.0 - 2.0 * x)
        }
    }
}

/// The perpendicular offset of `p` from the segment `a -> b`. A zero-length
/// segment collapses to the classic `p - a` disc. `a`, `b` and `p` are any
/// consistent brush-local plane frame; the result's magnitude is exactly the
/// plane distance to the segment.
#[allow(dead_code)]
pub fn perp_to_segment(p: Vec2, a: Vec2, b: Vec2) -> Vec2 {
    let ab = b - a;
    let len2 = ab.length_squared();
    if len2 <= f32::EPSILON {
        return p - a;
    }
    let t = ((p - a).dot(ab) / len2).clamp(0.0, 1.0);
    p - (a + ab * t)
}

impl<'a> Footprint<'a> {
    /// Classifies a point of the brush-local plane against the footprint.
    pub fn sample(&self, local: Vec2) -> FootprintSample {
        match self {
            Footprint::Round { radius } => {
                let dd = local.length_squared();
                let r2 = radius * radius;
                if dd > r2 {
                    FootprintSample::Outside
                } else {
                    FootprintSample::Distance(dd.sqrt() / radius)
                }
            }
            Footprint::Square { half } => {
                let (adx, ady) = (local.x.abs(), local.y.abs());
                if adx > *half || ady > *half {
                    FootprintSample::Outside
                } else {
                    FootprintSample::Distance(adx.max(ady) / half)
                }
            }
            Footprint::Diamond { half } => {
                let m = local.x.abs() + local.y.abs();
                if m > *half {
                    FootprintSample::Outside
                } else {
                    FootprintSample::Distance(m / half)
                }
            }
            Footprint::Rect { half_w, half_h } => {
                let (adx, ady) = (local.x.abs(), local.y.abs());
                if adx > *half_w || ady > *half_h {
                    FootprintSample::Outside
                } else {
                    FootprintSample::Distance((adx / half_w).max(ady / half_h))
                }
            }
            Footprint::Sprite {
                stamp,
                radius,
                window,
            } => {
                let anchored = match window {
                    Window::SpriteBounds => stamp.alpha_at(local, *radius),
                    win => {
                        let mask = win.mask(local, *radius);
                        if mask <= 0.0 {
                            return FootprintSample::Outside;
                        }
                        stamp.alpha_at_wrapped(local, *radius).map(|a| a * mask)
                    }
                };
                match anchored {
                    Some(a) => FootprintSample::Coverage(a),
                    None => FootprintSample::Outside,
                }
            }
        }
    }

    /// Pattern-locked sampling: `dab_local` classifies the point against the
    /// *paint window* (where this dab applies) while `pattern` is the anchored
    /// sprite frame (where the pattern is read). With a [`Window::Round`] —
    /// what `Brush::pattern_footprint` produces — the sprite is read at the
    /// anchored, **UV-wrapped** phase: the texture tiles infinitely across the
    /// stroke area and overlapping dabs sample the identical phase, so the
    /// pattern never smears or shifts. [`Window::SpriteBounds`] ignores
    /// `dab_local` and clamps the sprite to its own box (the rubber-stamp /
    /// decal path — passing `pattern == dab_local` reduces it to the classic
    /// dab). Non-sprite footprints ignore `pattern` and classify `dab_local`
    /// exactly like [`Footprint::sample`].
    pub fn sample_pattern(&self, dab_local: Vec2, pattern: Vec2) -> FootprintSample {
        match self {
            Footprint::Sprite {
                stamp,
                radius,
                window,
            } => {
                let anchored = match window {
                    Window::SpriteBounds => stamp.alpha_at(pattern, *radius),
                    win => {
                        // The mask lives in the dab-local frame (where this dab
                        // applies); the sprite is read at the anchored frame
                        // (where the pattern lies). Multiplying the anchored
                        // alpha by the dab-local soft window makes consecutive
                        // dabs overlap into a continuous stroke envelope.
                        let mask = win.mask(dab_local, *radius);
                        if mask <= 0.0 {
                            return FootprintSample::Outside;
                        }
                        stamp.alpha_at_wrapped(pattern, *radius).map(|a| a * mask)
                    }
                };
                match anchored {
                    Some(a) => FootprintSample::Coverage(a),
                    None => FootprintSample::Outside,
                }
            }
            other => other.sample(dab_local),
        }
    }

    /// Conservative outer reach of the footprint, used to cull whole triangles
    /// / bounding boxes before the per-texel loop.
    pub fn outer_radius(&self) -> f32 {
        match self {
            Footprint::Round { radius } => *radius,
            Footprint::Square { half } => half * std::f32::consts::SQRT_2,
            Footprint::Diamond { half } => *half,
            Footprint::Rect { half_w, half_h } => (half_w * half_w + half_h * half_h).sqrt(),
            Footprint::Sprite { radius, .. } => *radius,
        }
    }

    /// The footprint family.
    pub fn kind(&self) -> FootprintKind {
        match self {
            Footprint::Round { .. } => FootprintKind::Round,
            Footprint::Square { .. } => FootprintKind::Square,
            Footprint::Diamond { .. } => FootprintKind::Diamond,
            Footprint::Rect { .. } => FootprintKind::Rect,
            Footprint::Sprite { .. } => FootprintKind::Sprite,
        }
    }

    /// Mask selector consumed by the GPU cursor pipeline (0 = round,
    /// 1 = square, 2 = diamond, 3 = sprite). The Rect tool renders the square
    /// mask, matching the current overlay.
    #[allow(dead_code)] // the cursor overlay consumes this in a later increment
    pub fn cursor_shape(&self) -> u32 {
        match self {
            Footprint::Round { .. } => 0,
            Footprint::Square { .. } | Footprint::Rect { .. } => 1,
            Footprint::Diamond { .. } => 2,
            Footprint::Sprite { .. } => 3,
        }
    }

    /// The one constructor every stamp uses, so the fallback rules live in a
    /// single place. An explicit rectangle (`rect` tool) wins over everything;
    /// otherwise `kind` selects the family and `sprite` supplies the image for
    /// [`FootprintKind::Sprite`]. A sprite brush with no sprite degrades to a
    /// plain round dab. `radius` is the dab's half-size in the caller's units
    /// (world units for the 3D stamp, texels for the 2D stamp).
    pub fn for_dab(
        rect: Option<(f32, f32)>,
        kind: Option<FootprintKind>,
        sprite: Option<SpriteStamp<'a>>,
        radius: f32,
    ) -> Footprint<'a> {
        if let Some((half_w, half_h)) = rect {
            return Footprint::Rect { half_w, half_h };
        }
        match kind {
            None | Some(FootprintKind::Round) => Footprint::Round { radius },
            Some(FootprintKind::Square) => Footprint::Square { half: radius },
            Some(FootprintKind::Diamond) => Footprint::Diamond { half: radius },
            Some(FootprintKind::Rect) => Footprint::Rect {
                half_w: radius,
                half_h: radius,
            },
            Some(FootprintKind::Sprite) => match sprite {
                Some(stamp) => Footprint::Sprite {
                    stamp,
                    radius,
                    window: Window::SpriteBounds,
                },
                None => Footprint::Round { radius },
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    fn solid_sprite(w: u32, h: u32, alpha: u8) -> TextureData {
        let mut rgba = vec![255u8; (w * h * 4) as usize];
        for px in rgba.chunks_exact_mut(4) {
            px[3] = alpha;
        }
        TextureData {
            width: w,
            height: h,
            rgba,
        }
    }

    #[test]
    fn perp_to_segment_sweeps_a_disc_into_a_capsule() {
        // A point directly above the segment's middle: perpendicular offset is
        // pure screen-up, magnitude equals the plane distance to the segment.
        let off = perp_to_segment(
            Vec2::new(0.0, 3.0),
            Vec2::new(-4.0, 0.0),
            Vec2::new(4.0, 0.0),
        );
        assert!((off.x - 0.0).abs() < 1e-6 && (off.y - 3.0).abs() < 1e-6);
        // Beyond the `b` endpoint the window becomes the end disc again.
        let past = perp_to_segment(Vec2::new(6.0, 3.0), Vec2::ZERO, Vec2::new(4.0, 0.0));
        assert!((past - Vec2::new(2.0, 3.0)).length() < 1e-6);
        // Past the `a` endpoint is clamped to the start disc.
        let behind = perp_to_segment(Vec2::new(-7.0, 1.0), Vec2::ZERO, Vec2::new(4.0, 0.0));
        assert!((behind - Vec2::new(-7.0, 1.0)).length() < 1e-6);
        // A degenerate segment is exactly the `p - a` disc.
        let dot = perp_to_segment(
            Vec2::new(2.0, 5.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, 1.0),
        );
        assert!((dot - Vec2::new(1.0, 4.0)).length() < 1e-6);
    }

    /// Left `n` columns fully opaque, the rest fully transparent.
    fn left_opaque(w: u32, h: u32, n: u32) -> TextureData {
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..n {
                let i = ((y * w + x) * 4) as usize;
                rgba[i..i + 4].copy_from_slice(&[255, 255, 255, 255]);
            }
        }
        TextureData {
            width: w,
            height: h,
            rgba,
        }
    }

    fn assert_distance(sample: FootprintSample, expected: f32) {
        match sample {
            FootprintSample::Distance(t) => assert!(
                (t - expected).abs() < 1e-6,
                "Distance({t}) != expected {expected}"
            ),
            other => panic!("expected Distance({expected}), got {other:?}"),
        }
    }

    #[test]
    fn round_center_zero_rim_one_diagonal_inside() {
        let fp = Footprint::Round { radius: 10.0 };
        assert_eq!(fp.sample(Vec2::ZERO), FootprintSample::Distance(0.0));
        assert_distance(fp.sample(Vec2::new(10.0, 0.0)), 1.0);
        assert_eq!(fp.sample(Vec2::new(10.01, 0.0)), FootprintSample::Outside);
        // At (7,7) the chord is √98 ≈ 9.899 < 10: inside, t ≈ 0.9899.
        let d = (7.0f32 * 7.0 * 2.0).sqrt();
        assert_distance(fp.sample(Vec2::new(7.0, 7.0)), d / 10.0);
    }

    #[test]
    fn square_reaches_corners_that_round_misses() {
        let half = 10.0f32;
        let sq = Footprint::Square { half };
        let rd = Footprint::Round { radius: half };
        // (8,8): inside the square (|x|,|y| <= half) but outside the disc
        // (8² + 8² = 128 > 100).
        assert_distance(sq.sample(Vec2::new(8.0, 8.0)), 0.8);
        assert_eq!(rd.sample(Vec2::new(8.0, 8.0)), FootprintSample::Outside);
        // Beyond the half-extent on an axis: outside both.
        assert_eq!(sq.sample(Vec2::new(10.1, 0.0)), FootprintSample::Outside);
        assert_eq!(rd.sample(Vec2::new(10.1, 0.0)), FootprintSample::Outside);
        // Distance along the axis is max(|x|,|y|)/half.
        assert_distance(sq.sample(Vec2::new(7.0, 3.0)), 0.7);
    }

    #[test]
    fn diamond_rejects_diagonal_corners_round_keeps() {
        let half = 1.0f32;
        let dm = Footprint::Diamond { half };
        let rd = Footprint::Round { radius: half };
        // Diagonal (0.59,0.59): |x|+|y| = 1.19 > 1 → outside the diamond, yet
        // 0.59²·2 = 0.70 < 1 → inside the disc. Mirrors the existing stamp test.
        assert_eq!(dm.sample(Vec2::new(0.59, 0.59)), FootprintSample::Outside);
        assert_distance(
            rd.sample(Vec2::new(0.59, 0.59)),
            (0.59f32 * 0.59 * 2.0).sqrt(),
        );
        // Axis arm (0.84, 0): |x|+|y| = 0.84 <= 1 → inside.
        assert_distance(dm.sample(Vec2::new(0.84, 0.0)), 0.84);
        // Rim along an axis is t = 1.
        assert_distance(dm.sample(Vec2::new(1.0, 0.0)), 1.0);
        assert_eq!(dm.sample(Vec2::new(0.6, 0.6)), FootprintSample::Outside);
    }

    #[test]
    fn rect_uses_independent_half_extents() {
        // Mirrors paint.rs `rect_stamp_paints_a_rectangle_footprint`
        // (half_w = 0.3, half_h = 0.1).
        let fp = Footprint::Rect {
            half_w: 0.3,
            half_h: 0.1,
        };
        // Inside; the major axis drives the normalized distance.
        assert_distance(fp.sample(Vec2::new(0.25, 0.05)), 0.25 / 0.3);
        assert_distance(fp.sample(Vec2::new(0.1, 0.09)), 0.9);
        assert_distance(fp.sample(Vec2::new(0.3, 0.1)), 1.0);
        // Outside beyond either half-extent.
        assert_eq!(fp.sample(Vec2::new(0.35, 0.0)), FootprintSample::Outside);
        assert_eq!(fp.sample(Vec2::new(0.0, 0.11)), FootprintSample::Outside);
    }

    #[test]
    fn outer_radius_matches_current_culling_bounds() {
        let r = Footprint::Round { radius: 10.0 };
        assert!((r.outer_radius() - 10.0).abs() < 1e-6);
        let s = Footprint::Square { half: 10.0 };
        assert!((s.outer_radius() - 10.0 * std::f32::consts::SQRT_2).abs() < 1e-6);
        let d = Footprint::Diamond { half: 10.0 };
        assert!((d.outer_radius() - 10.0).abs() < 1e-6);
        let rect = Footprint::Rect {
            half_w: 0.3,
            half_h: 0.1,
        };
        assert!((rect.outer_radius() - (0.09f32 + 0.01).sqrt()).abs() < 1e-6);
        let sprite = solid_sprite(2, 2, 255);
        let spr = Footprint::Sprite {
            stamp: SpriteStamp {
                sprite: &sprite,
                rotation: 0.0,
                flip_x: false,
                flip_y: false,
            },
            radius: 4.0,
            window: Window::SpriteBounds,
        };
        assert!((spr.outer_radius() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn cursor_shape_codes_match_the_overlay_shader() {
        assert_eq!(Footprint::Round { radius: 1.0 }.cursor_shape(), 0);
        assert_eq!(Footprint::Square { half: 1.0 }.cursor_shape(), 1);
        assert_eq!(Footprint::Diamond { half: 1.0 }.cursor_shape(), 2);
        assert_eq!(
            Footprint::Rect {
                half_w: 1.0,
                half_h: 1.0
            }
            .cursor_shape(),
            1
        );
        let sprite = solid_sprite(2, 2, 255);
        assert_eq!(
            Footprint::Sprite {
                stamp: SpriteStamp {
                    sprite: &sprite,
                    rotation: 0.0,
                    flip_x: false,
                    flip_y: false,
                },
                radius: 1.0,
                window: Window::SpriteBounds,
            }
            .cursor_shape(),
            3
        );
    }

    #[test]
    fn for_dab_matches_historic_sprite_and_rect_fallbacks() {
        // Rect wins over kind.
        let r = Footprint::for_dab(Some((2.0, 1.0)), Some(FootprintKind::Square), None, 5.0);
        assert!(matches!(
            r,
            Footprint::Rect {
                half_w: 2.0,
                half_h: 1.0
            }
        ));
        // Sprite brush with a sprite → sprite; without → degrades to round.
        let sprite = solid_sprite(2, 2, 255);
        let stamp = SpriteStamp {
            sprite: &sprite,
            rotation: 0.0,
            flip_x: false,
            flip_y: false,
        };
        let s = Footprint::for_dab(None, Some(FootprintKind::Sprite), Some(stamp), 4.0);
        assert!(matches!(s, Footprint::Sprite { radius: 4.0, .. }));
        let fallback = Footprint::for_dab(None, Some(FootprintKind::Sprite), None, 4.0);
        assert!(matches!(fallback, Footprint::Round { radius: 4.0 }));
        // None kind → round; squares/diamonds keep their half-extent.
        assert!(matches!(
            Footprint::for_dab(None, None, None, 3.0),
            Footprint::Round { radius: 3.0 }
        ));
        assert!(matches!(
            Footprint::for_dab(None, Some(FootprintKind::Diamond), None, 3.0),
            Footprint::Diamond { half: 3.0 }
        ));
    }

    #[test]
    fn sprite_stamps_once_centered_not_tiled() {
        // Mirrors `texture_stamp_paints_the_sprite_once_centered_on_the_dab`:
        // a 4×4 sprite with the left 2 columns opaque, stamped at radius 1.
        let sprite = left_opaque(4, 4, 2);
        let stamp = SpriteStamp {
            sprite: &sprite,
            rotation: 0.0,
            flip_x: false,
            flip_y: false,
        };
        let fp = Footprint::Sprite {
            stamp,
            radius: 1.0,
            window: Window::SpriteBounds,
        };
        // Left half samples the opaque columns.
        match fp.sample(Vec2::new(-0.4, 0.0)) {
            FootprintSample::Coverage(a) => assert_eq!(a, 1.0),
            other => panic!("expected Coverage(1.0), got {other:?}"),
        }
        // Right half is the transparent columns → outside.
        assert_eq!(fp.sample(Vec2::new(0.4, 0.0)), FootprintSample::Outside);
        // Away from the centre only the sprite band footprint exists: no
        // world-space tiling, so the mirrored position never re-paints.
        assert_eq!(fp.sample(Vec2::new(0.9, 0.0)), FootprintSample::Outside);
        // Outside the square span of the stamp.
        assert_eq!(fp.sample(Vec2::new(1.5, 0.0)), FootprintSample::Outside);
    }

    #[test]
    fn sprite_keeps_native_aspect_not_stretched() {
        // Mirrors `texture_stamp_cursor_agrees_with_stamp_for_nonsquare_sprites`:
        // a 4-wide × 2-tall sprite, top row opaque, bottom row clear. The stamp
        // is centered with its longest side spanning the diameter; the opaque
        // band occupies only the top half of the footprint.
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
        let stamp = SpriteStamp {
            sprite: &sprite,
            rotation: 0.0,
            flip_x: false,
            flip_y: false,
        };
        let fp = Footprint::Sprite {
            stamp,
            radius: 1.0,
            window: Window::SpriteBounds,
        };
        // Opaque row maps to v_sprite < 0.5 → ry = tv > 0.
        match fp.sample(Vec2::new(0.0, 0.3)) {
            FootprintSample::Coverage(a) => assert_eq!(a, 1.0),
            other => panic!("expected Coverage(1.0), got {other:?}"),
        }
        // Clear row shows through below center.
        assert_eq!(fp.sample(Vec2::new(0.0, -0.3)), FootprintSample::Outside);
        // The sprite's native 4:2 aspect ends at half the footprint height — the
        // upper region is clear instead of the sprite being stretched to fill it.
        assert_eq!(fp.sample(Vec2::new(0.0, 0.6)), FootprintSample::Outside);
    }

    #[test]
    fn sprite_rotation_and_flips_transform_the_stamp() {
        // 2×2 with the left column opaque. After a 90° rotation the opaque
        // column lies along the +y local direction (mirroring the stamp math in
        // paint.rs: `x = tu·cr - tv·sr`).
        let sprite = left_opaque(2, 2, 1);
        let rot = SpriteStamp {
            sprite: &sprite,
            rotation: FRAC_PI_2,
            flip_x: false,
            flip_y: false,
        };
        let fpr = Footprint::Sprite {
            stamp: rot,
            radius: 1.0,
            window: Window::SpriteBounds,
        };
        match fpr.sample(Vec2::new(0.0, 0.4)) {
            FootprintSample::Coverage(a) => assert_eq!(a, 1.0),
            other => panic!("expected Coverage(1.0) after rotation, got {other:?}"),
        }
        assert_eq!(
            fpr.sample(Vec2::new(0.0, -0.4)),
            FootprintSample::Outside,
            "rotation must move the opaque band off the +y half"
        );

        // flip_x mirrors the band across the dab center.
        let sprite = left_opaque(2, 2, 1);
        let flip = SpriteStamp {
            sprite: &sprite,
            rotation: 0.0,
            flip_x: true,
            flip_y: false,
        };
        let fpf = Footprint::Sprite {
            stamp: flip,
            radius: 1.0,
            window: Window::SpriteBounds,
        };
        assert_eq!(fpf.sample(Vec2::new(-0.4, 0.0)), FootprintSample::Outside);
        match fpf.sample(Vec2::new(0.4, 0.0)) {
            FootprintSample::Coverage(a) => assert_eq!(a, 1.0),
            other => panic!("expected Coverage(1.0) after flip_x, got {other:?}"),
        }
    }

    #[test]
    fn sprite_alpha_is_the_coverage_not_a_bool() {
        // A half-alpha texel comes back as Coverage(0.5): the falloff must not
        // destroy partial sprite opacity.
        let mut rgba = vec![255u8; 4];
        rgba[3] = 128;
        let sprite = TextureData {
            width: 1,
            height: 1,
            rgba,
        };
        let stamp = SpriteStamp {
            sprite: &sprite,
            rotation: 0.0,
            flip_x: false,
            flip_y: false,
        };
        let fp = Footprint::Sprite {
            stamp,
            radius: 1.0,
            window: Window::SpriteBounds,
        };
        match fp.sample(Vec2::new(0.0, 0.0)) {
            FootprintSample::Coverage(a) => assert!((a - 128.0 / 255.0).abs() < 1e-6),
            other => panic!("expected Coverage(~0.5), got {other:?}"),
        }
    }

    #[test]
    fn pattern_window_fades_softly_to_the_rim() {
        // An all-opaque texture: the window is a soft-edged disc of `radius` —
        // full coverage across the flat core, a C¹ skirt down to the rim, and
        // nothing beyond. The soft skirt keeps consecutive pattern dabs from
        // leaving crisp coin-edge arcs.
        let sprite = solid_sprite(4, 4, 255);
        let stamp = SpriteStamp {
            sprite: &sprite,
            rotation: 0.0,
            flip_x: false,
            flip_y: false,
        };
        let fp = Footprint::Sprite {
            stamp,
            radius: 1.0,
            window: Window::Round,
        };
        // Flat core: full coverage up to half the radius.
        match fp.sample(Vec2::new(-0.4, 0.0)) {
            FootprintSample::Coverage(a) => assert_eq!(a, 1.0),
            other => panic!("expected Coverage(1.0), got {other:?}"),
        }
        // In-window corner at 0.848 radius: inside the disc but in the skirt —
        // partial, exactly the inverted-smoothstep mask value.
        let t = (0.6f32 * 0.6 * 2.0).sqrt();
        let x = (t - 0.5) * 2.0;
        let expected = 1.0 - x * x * (3.0 - 2.0 * x);
        match fp.sample(Vec2::new(0.6, 0.6)) {
            FootprintSample::Coverage(a) => assert!(
                (a - expected).abs() < 1e-6,
                "skirt coverage {a} must equal the mask {expected}"
            ),
            other => panic!("expected Coverage(~{expected}), got {other:?}"),
        }
        // Beyond the disc the dab paints nothing.
        assert_eq!(fp.sample(Vec2::new(1.01, 0.0)), FootprintSample::Outside);
        assert_eq!(fp.sample(Vec2::new(0.0, 1.01)), FootprintSample::Outside);
    }

    #[test]
    fn pattern_phase_is_anchored_while_membership_stays_dab_local() {
        // Same anchored pattern offset read at two different dab positions
        // yields identical coverage: the phase is glued. A different anchored
        // offset (the transparent right half) reads transparent at the same
        // dab position.
        let sprite = left_opaque(2, 2, 1);
        let stamp = SpriteStamp {
            sprite: &sprite,
            rotation: 0.0,
            flip_x: false,
            flip_y: false,
        };
        let fp = Footprint::Sprite {
            stamp,
            radius: 1.0,
            window: Window::Round,
        };
        let at = |dab: Vec2, pattern: Vec2| fp.sample_pattern(dab, pattern);
        // Two different dab centers, the same anchored phase → identical alpha.
        assert_eq!(
            at(Vec2::ZERO, Vec2::new(-0.4, 0.0)),
            at(Vec2::new(0.4, 0.0), Vec2::new(-0.4, 0.0))
        );
        // The dab-local window still gates membership: move the dab out while
        // the anchored phase stays the same → nothing.
        assert_eq!(
            at(Vec2::new(1.01, 0.0), Vec2::new(-0.4, 0.0)),
            FootprintSample::Outside
        );
        // Same dab, other phase → the transparent half of the sprite.
        assert_eq!(
            at(Vec2::ZERO, Vec2::new(0.4, 0.0)),
            FootprintSample::Outside
        );
    }

    #[test]
    fn pattern_wrap_repeats_the_sprite_under_the_dab() {
        // An all-opaque 2×4 sprite (2 wide × 4 tall). Wrapped sampling returns
        // the same texel for a local position and that position + a full sprite
        // width of travel in the anchored frame.
        let sprite = solid_sprite(2, 4, 255);
        let stamp = SpriteStamp {
            sprite: &sprite,
            rotation: 0.0,
            flip_x: false,
            flip_y: false,
        };
        let fp = Footprint::Sprite {
            stamp,
            radius: 1.0,
            window: Window::Round,
        };
        // Sprite half-width ww = 2·sw·r/4 = r, so `1.6` sits one wrapped period
        // away from `-0.4` in anchored space: both map to the same texel.
        let near = fp.sample_pattern(Vec2::ZERO, Vec2::new(-0.4, 0.0));
        let far = fp.sample_pattern(Vec2::ZERO, Vec2::new(1.6, 0.0));
        assert_eq!(near, far, "wrapped phase must repeat the same sprite texel");
        assert!(matches!(near, FootprintSample::Coverage(1.0)));
    }
}
