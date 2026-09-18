//! Dab falloff curves.
//!
//! A falloff maps a footprint's normalized feature distance
//! ([`FootprintSample::Distance`]) to a coverage value in `0..=1`. Sprite
//! coverage passes through untouched: the sprite's alpha IS the coverage, so
//! the distance curve never touches it — this matches both current stampers.

use super::footprint::FootprintSample;

/// How a dab's coverage falls off between the center and the rim.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DabProfile {
    /// Classic brush hardness: full strength out to `hardness` of the radius,
    /// then a linear fade to the edge. `1.0` = hard rim, `0.0` = a gradient
    /// falling from the center outward.
    Hardness { hardness: f32 },
    /// Eraser: feathers with a fully-transparent core — full erasure out to
    /// `core` of the radius, then a linear fade in toward the rim.
    Eraser { core: f32 },
    /// A sprite stamp: the sprite's alpha is the coverage; there is no
    /// distance curve.
    SpriteCoverage,
}

impl DabProfile {
    /// Classic hardness curve with the hardness clamped to `[0, 0.999]` (a
    /// core of 1.0 would divide by zero at the rim).
    pub fn hardness(hardness: f32) -> Self {
        Self::Hardness {
            hardness: hardness.clamp(0.0, 0.999),
        }
    }

    /// The eraser's transparent-core feather (core fraction 0.55).
    pub fn eraser() -> Self {
        Self::Eraser { core: 0.55 }
    }

    /// A sprite stamp: alpha is coverage, no distance curve.
    pub const fn sprite() -> Self {
        Self::SpriteCoverage
    }

    /// The falloff every stamp applies. A sprite stamp carries its own
    /// coverage and needs no distance curve; erasing feathers with a
    /// transparent core; anything else uses the hardness curve. `sprite`
    /// wins over `erasing`, exactly like the historic stampers.
    pub fn for_dab(sprite: bool, erasing: bool, hardness: f32) -> Self {
        if sprite {
            Self::sprite()
        } else if erasing {
            Self::eraser()
        } else {
            Self::hardness(hardness)
        }
    }

    /// Resolves a footprint sample into a coverage value in `0..=1`
    /// (`0.0` = nothing to paint).
    pub fn coverage(&self, sample: FootprintSample) -> f32 {
        match sample {
            FootprintSample::Outside => 0.0,
            // Sprite alpha is already a coverage; no curve applies.
            FootprintSample::Coverage(a) => a,
            FootprintSample::Distance(t) => match self {
                Self::Hardness { hardness } => {
                    let core = hardness.clamp(0.0, 0.999);
                    if t <= core {
                        1.0
                    } else {
                        ((1.0 - t) / (1.0 - core)).clamp(0.0, 1.0)
                    }
                }
                Self::Eraser { core } => {
                    let t = t.min(1.0);
                    let core = core.min(1.0);
                    if t <= core {
                        1.0
                    } else {
                        ((1.0 - t) / (1.0 - core)).clamp(0.0, 1.0)
                    }
                }
                Self::SpriteCoverage => 0.0,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn for_dab_matches_historic_stamper_choices() {
        // Sprite wins over erasing; eraser feathers over hardness.
        assert_eq!(DabProfile::for_dab(true, true, 0.5), DabProfile::sprite());
        assert_eq!(DabProfile::for_dab(false, true, 0.5), DabProfile::eraser());
        assert_eq!(
            DabProfile::for_dab(false, false, 0.5),
            DabProfile::hardness(0.5)
        );
    }

    #[test]
    fn hardness_keeps_full_strength_core_then_fades() {
        // hardness 0.6: full out to t=0.6, linear to 0 at t=1.
        let p = DabProfile::hardness(0.6);
        assert!((p.coverage(FootprintSample::Distance(0.0)) - 1.0).abs() < 1e-6);
        assert!((p.coverage(FootprintSample::Distance(0.6)) - 1.0).abs() < 1e-6);
        assert!((p.coverage(FootprintSample::Distance(0.8)) - 0.5).abs() < 1e-6);
        assert!((p.coverage(FootprintSample::Distance(1.0)) - 0.0).abs() < 1e-6);
        assert!((p.coverage(FootprintSample::Outside) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn hardness_is_monotonic_in_distance() {
        for hardness in [0.0f32, 0.3, 0.6, 0.999] {
            let p = DabProfile::hardness(hardness);
            let mut prev = f32::INFINITY;
            for i in 0..=100 {
                let t = i as f32 / 100.0;
                let c = p.coverage(FootprintSample::Distance(t));
                assert!(
                    c <= prev + 1e-6,
                    "coverage must not rise with distance at {t}"
                );
                prev = c;
            }
        }
    }

    #[test]
    fn hardness_one_is_a_hard_rim() {
        let hard = DabProfile::hardness(1.0);
        assert!((hard.coverage(FootprintSample::Distance(0.9)) - 1.0).abs() < 1e-6);
        assert!((hard.coverage(FootprintSample::Distance(1.0)) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn eraser_feathers_from_a_transparent_core() {
        let p = DabProfile::eraser();
        assert!((p.coverage(FootprintSample::Distance(0.0)) - 1.0).abs() < 1e-6);
        assert!((p.coverage(FootprintSample::Distance(0.55)) - 1.0).abs() < 1e-6);
        // Mid-fade: (1 - 0.8) / (1 - 0.55) = 0.2 / 0.45 ≈ 0.444.
        let mid = p.coverage(FootprintSample::Distance(0.8));
        assert!((mid - 0.2 / 0.45).abs() < 1e-6, "mid-fade coverage {mid}");
        assert!((p.coverage(FootprintSample::Distance(1.0)) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn sprite_coverage_bypasses_the_curve() {
        // Even a hardness profile must treat sprite coverage verbatim — the
        // stampers resolve sprite alpha as the coverage before any falloff.
        for p in [
            DabProfile::hardness(0.0),
            DabProfile::hardness(1.0),
            DabProfile::eraser(),
            DabProfile::sprite(),
        ] {
            assert!((p.coverage(FootprintSample::Coverage(0.5)) - 0.5).abs() < 1e-6);
            assert!((p.coverage(FootprintSample::Coverage(1.0)) - 1.0).abs() < 1e-6);
            assert!((p.coverage(FootprintSample::Outside) - 0.0).abs() < 1e-6);
        }
    }

    #[test]
    fn sprite_profile_ignores_distance() {
        assert!((DabProfile::sprite().coverage(FootprintSample::Distance(0.3)) - 0.0).abs() < 1e-6);
    }
}
