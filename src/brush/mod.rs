//! Pure brush-engine core: footprints, falloff profiles, the brush value type
//! and the shared per-texel coverage evaluator.
//!
//! Everything here is geometry-free — it knows only brush-local plane
//! coordinates and coverage masks, never a `MeshData` or a GPU — so every
//! piece is unit testable in isolation. The 3D/2D stamping orchestration (in
//! `paint`) lives on top of these primitives.
//!
//! The stamp pipeline is:
//!
//! ```text
//! Brush ─► footprint()/falloff()     dip: Footprint, profile: DabProfile
//! plane point ─► Footprint::sample → FootprintSample
//!                                     ─► DabProfile::coverage → coverage
//! ```

pub mod falloff;
pub mod footprint;
pub mod raster;

use glam::Vec3;

use crate::io::TextureData;

pub use falloff::DabProfile;
pub use footprint::{Footprint, FootprintKind, SpriteStamp, Window};
pub use raster::{local_coverage, pattern_coverage};

/// How a dab interacts with the existing texels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StampMode {
    Paint,
    Erase,
}

/// Whether a texture brush anchors its pattern to a fixed source point for the
/// whole stroke or re-centers the sprite on every dab.
///
/// - [`PatternLock::Dab`] is the classic rubber stamp: each dab centers the
///   sprite on itself, so overlapping dabs re-sample the sprite and a drag
///   smears the pattern.
/// - [`PatternLock::Aligned`] samples the sprite at a fixed stroke-start anchor
///   (the [`PatternAnchor`]), so the phase stays glued under the brush as it
///   moves — the beside-dab reads show the same region of the source texture.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PatternLock {
    #[default]
    Dab,
    Aligned,
}

/// Stroke-time anchor of a pattern-locked stroke: pure data the stampers turn
/// into per-texel pattern coordinates, never geometry itself.
#[derive(Clone, Copy, Debug)]
pub enum PatternAnchor {
    /// 2D: an anchored texture texel plus the brush radius (in texels) of the
    /// click dab. The 2D stamp derives the anchored phase directly from the
    /// absolute texel position (never the moving dab center), so the pattern
    /// is an infinite, stationary, square-tiled grid glued to the canvas; the
    /// captured radius fixes the grid's tile size for the whole stroke (a
    /// mid-stroke resize re-scales the dab mask, not the texture).
    Canvas { x: f32, y: f32, radius: f32 },
    /// 3D: a world point plus the tangent basis captured once at stroke start
    /// (the phase follows the fixed axes, so it stays glued across curvature),
    /// and the world radius of that first dab — later dabs divide by it so the
    /// pattern's world size stays constant even if the dab radius varies.
    Surface {
        pos: Vec3,
        axis_u: Vec3,
        axis_v: Vec3,
        radius: f32,
    },
    /// Seamless UV anchor: maps the tiled texture across the mesh's UV coordinates
    /// and unmasks it through the circle brush. Neighboring surfaces with continuous
    /// UV mapping will have the texture seamlessly drawn across them in 3D and 2D.
    Uv { x: f32, y: f32, radius: f32 },
}

/// A brush: pure *properties*, no evaluation logic. Every routine that paints
/// resolves a dab through [`Brush::footprint`] + [`Brush::falloff`] and then
/// evaluates texels with [`local_coverage`].
#[derive(Clone, Debug)]
pub struct Brush {
    pub kind: FootprintKind,
    /// Brush radius in screen px (the 3D stamp converts it to world units at
    /// apply time via `screen_to_world_radius`; the 2D stamp uses it directly
    /// in texels).
    pub size: f32,
    /// 0..=1 softness of the distance falloff (hardness curve plateau).
    pub hardness: f32,
    /// Distance (px) between dab centers along a stroke; `<= 0` = continuous
    /// (step at half the brush radius so successive dabs overlap).
    pub spacing: f32,
    /// Stroke strength applied per texel (the falloff scales it).
    pub opacity: f32,
    /// When false, the stroke's opacity is capped at `opacity`: repeated passes
    /// over the same area in one stroke do not stack.
    pub accumulate: bool,
    /// RGBA brush color (Paint/Fill; transparent for Erase).
    pub color: [u8; 4],
    pub mode: StampMode,
    /// Image stamp + transform for [`FootprintKind::Sprite`]; `None` degrades
    /// a sprite brush to a round dab.
    pub sprite: Option<TextureData>,
    /// How a texture brush positions its pattern while stamping:
    /// [`PatternLock::Dab`] re-centers the sprite on every dab (rubber stamp);
    /// [`PatternLock::Aligned`] glues it to a fixed stroke-start anchor.
    pub pattern_lock: PatternLock,
    /// Stamp rotation in radians (sprite brushes only).
    pub rotation: f32,
    pub flip_x: bool,
    pub flip_y: bool,
}

impl Default for Brush {
    fn default() -> Self {
        Self {
            kind: FootprintKind::Round,
            size: 24.0,
            hardness: 0.5,
            spacing: 6.0,
            opacity: 1.0,
            accumulate: true,
            color: [90, 160, 255, 255],
            mode: StampMode::Paint,
            sprite: None,
            pattern_lock: PatternLock::Dab,
            rotation: 0.0,
            flip_x: false,
            flip_y: false,
        }
    }
}

impl Brush {
    /// The dab's footprint at `radius` in the caller's units. A sprite brush
    /// with no stored sprite degrades to Round.
    pub fn footprint(&self, radius: f32) -> Footprint<'_> {
        Footprint::for_dab(None, Some(self.kind), self.sprite_stamp(), radius)
    }

    /// The pattern-locked dab footprint for [`PatternLock::Aligned`]: the dab
    /// paints inside a dab-local [`Window::Round`] disc of `radius`, while
    /// `sample_pattern` reads the sprite at the anchored (world/screen-locked)
    /// phase with **wrapping** UVs — the texture tiles infinitely across the
    /// stroke area instead of clamping to a single sprite box. Overlapping dabs
    /// sample the same anchored phase, so the pattern never smears or shifts.
    /// Degrades to the plain footprint when the brush carries no texture to
    /// lock.
    pub fn pattern_footprint(&self, radius: f32) -> Footprint<'_> {
        if self.kind != FootprintKind::Sprite || self.sprite.is_none() {
            return self.footprint(radius);
        }
        match self.sprite_stamp() {
            Some(stamp) => Footprint::sprite(stamp, radius, Window::Round),
            None => Footprint::Round { radius },
        }
    }

    /// The dab's falloff: a sprite stamp carries its own coverage (no distance
    /// curve); erasing feathers with a transparent core; otherwise hardness.
    pub fn falloff(&self) -> DabProfile {
        DabProfile::for_dab(
            self.kind == FootprintKind::Sprite && self.sprite.is_some(),
            self.mode == StampMode::Erase,
            self.hardness,
        )
    }

    /// The sprite stamp borrowing this brush's sprite (none unless
    /// [`FootprintKind::Sprite`] with an image).
    pub fn sprite_stamp(&self) -> Option<SpriteStamp<'_>> {
        self.sprite.as_ref().map(|sprite| SpriteStamp {
            sprite,
            rotation: self.rotation,
            flip_x: self.flip_x,
            flip_y: self.flip_y,
        })
    }

    /// With a [`PatternLock::Aligned`] world-locked texture the stroke must
    /// ride a much denser dab train than a rubber-stamp brush: consecutive
    /// dab centers land within the flat core of each soft window mask, so the
    /// max-combined envelope stays level everywhere (no scalloped
    /// "chain of coins" between dabs). Half the brush radius.
    pub fn pattern_spacing(&self) -> f32 {
        (self.size / 4.0).max(1.0)
    }

    /// Dab spacing in screen px; `<= 0` (continuous) steps at half the brush
    /// radius so dabs always overlap.
    pub fn effective_spacing(&self) -> f32 {
        if self.spacing > 0.0 {
            self.spacing
        } else {
            (self.size / 2.0).max(1.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_brush_is_a_soft_round_paint() {
        let b = Brush::default();
        assert!(matches!(
            b.footprint(10.0),
            Footprint::Round { radius: 10.0 }
        ));
        assert_eq!(b.falloff(), DabProfile::hardness(0.5));
        assert_eq!(b.mode, StampMode::Paint);
        assert!(b.sprite_stamp().is_none());
    }

    #[test]
    fn sprite_brush_without_image_degrades_to_round() {
        let mut b = Brush::default();
        b.kind = FootprintKind::Sprite;
        assert!(matches!(
            b.footprint(10.0),
            Footprint::Round { radius: 10.0 }
        ));
        // ... and its falloff is hardness, not sprite coverage.
        assert_ne!(b.falloff(), DabProfile::sprite());
    }

    #[test]
    fn sprite_brush_with_image_paints_through_the_sprite() {
        let mut b = Brush::default();
        b.kind = FootprintKind::Sprite;
        let mut rgba = vec![255u8; 4];
        rgba[3] = 128;
        b.sprite = Some(TextureData {
            width: 1,
            height: 1,
            rgba,
        });
        b.hardness = 0.0;
        let fp = b.footprint(4.0);
        assert!(matches!(fp, Footprint::Sprite { radius: 4.0, .. }));
        // The sprite's half-alpha IS the coverage, immune to the hardness curve.
        let c = local_coverage(&b.falloff(), &fp, glam::Vec2::ZERO);
        assert!(
            (c - 128.0 / 255.0).abs() < 1e-6,
            "sprite alpha bypasses hardness {c}"
        );
    }

    #[test]
    fn effective_spacing_falls_back_to_half_the_radius() {
        let mut b = Brush::default();
        b.spacing = 8.0;
        assert_eq!(b.effective_spacing(), 8.0);
        b.spacing = 0.0;
        b.size = 24.0;
        assert_eq!(b.effective_spacing(), 12.0);
    }

    #[test]
    fn eraser_falloff_feathers() {
        let mut b = Brush::default();
        b.mode = StampMode::Erase;
        assert_eq!(b.falloff(), DabProfile::eraser());
    }

    #[test]
    fn shape_codes_round_trip_with_the_legacy_layout() {
        // Legacy BrushShape order: Round 0, Square 1, Diamond 2, Texture 3.
        assert_eq!(FootprintKind::Round.shape_code(), 0);
        assert_eq!(FootprintKind::Square.shape_code(), 1);
        assert_eq!(FootprintKind::Diamond.shape_code(), 2);
        assert_eq!(FootprintKind::Sprite.shape_code(), 3);
        assert_eq!(FootprintKind::Rect.shape_code(), 3);
        assert_eq!(FootprintKind::from_shape_code(0), FootprintKind::Round);
        assert_eq!(FootprintKind::from_shape_code(1), FootprintKind::Square);
        assert_eq!(FootprintKind::from_shape_code(2), FootprintKind::Diamond);
        assert_eq!(FootprintKind::from_shape_code(3), FootprintKind::Sprite);
        assert_eq!(FootprintKind::from_shape_code(9), FootprintKind::Sprite);
    }

    #[test]
    fn pattern_lock_defaults_to_rubber_stamp() {
        assert_eq!(Brush::default().pattern_lock, PatternLock::Dab);
        assert_eq!(PatternLock::default(), PatternLock::Dab);
    }

    #[test]
    fn pattern_footprint_only_locks_when_a_texture_is_present() {
        // A plain round brush ignores pattern lock entirely.
        let mut b = Brush::default();
        b.pattern_lock = PatternLock::Aligned;
        assert!(matches!(
            b.pattern_footprint(6.0),
            Footprint::Round { radius: 6.0 }
        ));
        // A sprite brush with no image degrades the same way.
        b.kind = FootprintKind::Sprite;
        assert!(matches!(
            b.pattern_footprint(6.0),
            Footprint::Round { radius: 6.0 }
        ));
        // With an image the locked dab samples a dab-local disc (Round window)
        // and the anchored phase + wrapping UVs tile the sprite infinitely.
        let rgba = vec![255u8; 4];
        b.sprite = Some(TextureData {
            width: 1,
            height: 1,
            rgba,
        });
        match b.pattern_footprint(6.0) {
            Footprint::Sprite {
                radius,
                window,
                ..
            } => {
                assert_eq!(radius, 6.0);
                assert_eq!(window, Window::Round);
                // Wrapping: a phase a full sprite-span away still samples the
                // texel (clamped sampling would fall outside the span).
                let stamp = b.sprite_stamp().unwrap();
                assert!(stamp
                    .alpha_at_wrapped(glam::Vec2::new(11.9, 0.0), 6.0)
                    .is_some());
                assert!(stamp.alpha_at(glam::Vec2::new(11.9, 0.0), 6.0).is_none());
            }
            other => panic!("expected a windowed sprite footprint, got {other:?}"),
        }
    }
}
