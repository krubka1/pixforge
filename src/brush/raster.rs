//! Shared per-texel coverage evaluation.
//!
//! Both the 3D and 2D stampers resolve a texel's contribution through this
//! single function, so the footprint geometry and falloff curve live in exactly
//! one place. The only input is a point on the brush-local plane dressed as a
//! `glam::Vec2`.
//!
//!   - 3D stamp: `local = (pos_3d - center).dot(axis_u), · · .dot(axis_v)`
//!   - 2D stamp: `local = (x - cx, -(y - cy))` — texel space with +y = canvas-up
//!     so the sprite's top maps to the canvas top, mirroring the 2D cursor.

use glam::Vec2;

use super::falloff::DabProfile;
use super::footprint::Footprint;

/// The coverage contribution of one texel whose plane position is `local`.
/// `0.0` means the texel is outside the footprint (or fully transparent).
pub fn local_coverage(profile: &DabProfile, footprint: &Footprint, local: Vec2) -> f32 {
    profile.coverage(footprint.sample(local))
}

/// Coverage for a pattern-locked dab: `dab_local` is the paint-window frame
/// (where this dab applies), `pattern` is the anchored pattern frame (where the
/// sprite is read). Equivalent to [`local_coverage`] when `pattern ==
/// dab_local`, so rubber-stamp and non-sprite dabs keep using that. With a
/// tiling window ([`Window::Round`], what `Brush::pattern_footprint`
/// produces) the sprite is read with wrapping UVs, so the anchored pattern
/// repeats infinitely across the stroke area and overlapping dabs read the
/// identical phase.
pub fn pattern_coverage(
    profile: &DabProfile,
    footprint: &Footprint,
    dab_local: Vec2,
    pattern: Vec2,
) -> f32 {
    profile.coverage(footprint.sample_pattern(dab_local, pattern))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brush::footprint::{SpriteStamp, Window};

    fn solid_sprite(w: u32, h: u32, alpha: u8) -> crate::io::TextureData {
        let mut rgba = vec![255u8; (w * h * 4) as usize];
        for px in rgba.chunks_exact_mut(4) {
            px[3] = alpha;
        }
        crate::io::TextureData {
            width: w,
            height: h,
            rgba,
        }
    }

    /// Left `n` columns fully opaque, the rest fully transparent.
    fn left_opaque(w: u32, h: u32, n: u32) -> crate::io::TextureData {
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..n {
                let i = ((y * w + x) * 4) as usize;
                rgba[i..i + 4].copy_from_slice(&[255, 255, 255, 255]);
            }
        }
        crate::io::TextureData {
            width: w,
            height: h,
            rgba,
        }
    }

    #[test]
    fn hard_square_dab_is_full_in_the_core_only() {
        let fp = Footprint::Square { half: 10.0 };
        // Hard rim: full strength across the whole square, nothing outside.
        assert!(local_coverage(&DabProfile::hardness(1.0), &fp, Vec2::ZERO) > 0.999);
        assert!(local_coverage(&DabProfile::hardness(1.0), &fp, Vec2::new(9.9, 0.0)) > 0.999);
        assert!(local_coverage(&DabProfile::hardness(1.0), &fp, Vec2::new(20.0, 0.0)) < 1e-6);
        // Soft rim: full only at the center, faded at the edge.
        let soft = local_coverage(&DabProfile::hardness(0.0), &fp, Vec2::new(5.0, 0.0));
        assert!(
            (soft - 0.5).abs() < 1e-6,
            "soft square covers 0.5 at half extent {soft}"
        );
    }

    #[test]
    fn sprite_dab_bypasses_the_hardness_curve() {
        // An opaque sprite stamped with a soft hardness 0 brush still paints
        // at full coverage where its alpha is 1 — the sprite IS the coverage.
        let sprite = solid_sprite(2, 2, 255);
        let fp = Footprint::sprite(
            SpriteStamp {
                sprite: &sprite,
                rotation: 0.0,
                flip_x: false,
                flip_y: false,
            },
            4.0,
            Window::SpriteBounds,
        );
        let soft = DabProfile::hardness(0.0);
        assert!((local_coverage(&soft, &fp, Vec2::new(-0.5, 0.0)) - 1.0).abs() < 1e-6);
        // Beyond the stamp span: nothing.
        assert!(local_coverage(&soft, &fp, Vec2::new(4.5, 0.0)) < 1e-6);
    }

    #[test]
    fn eraser_dab_uses_the_feather_curve() {
        let fp = Footprint::Round { radius: 8.0 };
        let p = DabProfile::eraser();
        assert!(
            local_coverage(&p, &fp, Vec2::ZERO) > 0.999,
            "full erase at center"
        );
        assert!(
            local_coverage(&p, &fp, Vec2::new(7.6, 0.0)) < 1.0,
            "rim feather"
        );
        assert!(
            local_coverage(&p, &fp, Vec2::new(8.1, 0.0)) < 1e-6,
            "outside erases nothing"
        );
    }

    #[test]
    fn pattern_coverage_locks_the_phase_across_dab_positions() {
        let sprite = left_opaque(2, 2, 1);
        let stamp = SpriteStamp {
            sprite: &sprite,
            rotation: 0.0,
            flip_x: false,
            flip_y: false,
        };
        let fp = Footprint::sprite(stamp, 1.0, Window::Round);
        let p = DabProfile::sprite();
        // Two different dab positions reading the same anchored phase agree.
        let a = pattern_coverage(&p, &fp, Vec2::new(0.0, 0.0), Vec2::new(-0.4, 0.0));
        let b = pattern_coverage(&p, &fp, Vec2::new(0.5, 0.0), Vec2::new(-0.4, 0.0));
        assert!(
            (a - b).abs() < 1e-6,
            "phase must be anchored, got {a} vs {b}"
        );
        assert!(a > 0.999, "the anchored phase samples the opaque texel {a}");
        // The dab-local window still gates membership wherever the phase is.
        assert!(pattern_coverage(&p, &fp, Vec2::new(1.01, 0.0), Vec2::new(-0.4, 0.0)) < 1e-6);
        // Wrapping: an anchored position a wrapped period away re-reads the
        // same texel (2-px sprite, radius 1 → span 2 → +1.6 ≡ -0.4).
        let wrapped = pattern_coverage(&p, &fp, Vec2::new(0.0, 0.0), Vec2::new(1.6, 0.0));
        assert!(
            (wrapped - a).abs() < 1e-6,
            "wrapped {wrapped} vs anchored {a}"
        );
    }

    #[test]
    fn pattern_coverage_tiles_infinitely_across_periods() {
        // The anchored pattern repeats forever: a phase any whole number of
        // sprite-spans away samples the identical texel, so a stroke covering
        // many periods paints a seamless, anchored tiling.
        let sprite = left_opaque(2, 2, 1);
        let stamp = SpriteStamp {
            sprite: &sprite,
            rotation: 0.0,
            flip_x: false,
            flip_y: false,
        };
        let fp = Footprint::sprite(stamp, 1.0, Window::Round);
        let p = DabProfile::sprite();
        // Span of a 2-px sprite at radius 1 is 2.0 pattern units, so only
        // phases a whole number of spans from the anchor tile identically.
        let anchor = pattern_coverage(&p, &fp, Vec2::ZERO, Vec2::new(-0.6, 0.0));
        assert!(anchor > 0.999);
        for phase in [1.4f32, 3.4, 5.4, -2.6, -4.6, 7.4] {
            let c = pattern_coverage(&p, &fp, Vec2::ZERO, Vec2::new(phase, 0.0));
            assert!(
                (c - anchor).abs() < 1e-6,
                "phase {phase} must tile to the anchored texel, got {c}"
            );
        }
        // The tiling is seamless, not all-out: phases landing on the sprite's
        // transparent half stay clear wherever they occur.
        for phase in [0.4f32, 2.4, 4.4, -1.6] {
            assert!(
                pattern_coverage(&p, &fp, Vec2::ZERO, Vec2::new(phase, 0.0)) < 1e-6,
                "transparent half must stay clear at phase {phase}"
            );
        }
    }

    #[test]
    fn pattern_coverage_equals_local_coverage_for_rubber_stamps() {
        // RB path: pattern == dab_local goes through the classic SpriteBounds
        // window, identical to `local_coverage`.
        let sprite = left_opaque(4, 4, 2);
        let stamp = SpriteStamp {
            sprite: &sprite,
            rotation: 0.0,
            flip_x: false,
            flip_y: false,
        };
        let fp = Footprint::sprite(stamp, 1.0, Window::SpriteBounds);
        let p = DabProfile::sprite();
        for local in [
            Vec2::new(-0.4, 0.0),
            Vec2::new(0.4, 0.0),
            Vec2::new(1.5, 0.0),
        ] {
            assert_eq!(
                pattern_coverage(&p, &fp, local, local),
                local_coverage(&p, &fp, local),
                "at {local:?}"
            );
        }
    }
}
