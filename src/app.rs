use std::collections::HashMap;
use std::sync::Arc;

use egui::{ColorImage, Id, TextureHandle, Ui, WidgetText};
use egui_dock::{DockArea, DockState, NodeIndex, TabViewer};

use crate::io::MeshData;
use crate::render::{Camera, Renderer, ViewportTextures};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Panel {
    Viewport,
    Channels,
    Texture,
    Layers,
    Lighting,
    Brushes,
    Palette,
    Preferences,
}

impl Panel {
    const ALL: [Panel; 8] = [
        Panel::Viewport,
        Panel::Channels,
        Panel::Texture,
        Panel::Layers,
        Panel::Lighting,
        Panel::Brushes,
        Panel::Palette,
        Panel::Preferences,
    ];

    fn title(&self) -> &'static str {
        match self {
            Panel::Viewport => "3D Viewport",
            Panel::Channels => "Material",
            Panel::Texture => "Texture Editor",
            Panel::Layers => "Layers",
            Panel::Lighting => "Lighting",
            Panel::Brushes => "Brushes",
            Panel::Palette => "Palette",
            Panel::Preferences => "Preferences",
        }
    }

    fn index(&self) -> usize {
        Self::ALL.iter().position(|p| p == self).unwrap()
    }
}

pub struct PixForgeApp {
    dock_state: DockState<Panel>,
    core: Core,
}

struct Core {
    device: wgpu::Device,
    egui_renderer: Arc<egui::epaint::mutex::RwLock<egui_wgpu::Renderer>>,
    renderer: Renderer,
    viewport: Option<ViewportResources>,
    mesh: Option<MeshData>,
    center: glam::Vec3,
    bounds_radius: f32,
    needs_fit: bool,
    /// Per-panel tab visibility (View menu). Kept as a `Vec` so older saved
    /// layouts (a smaller array) still load; entries past the stored length
    /// default to visible.
    panel_visible: Vec<bool>,
    active_tool: usize,
    channels: [bool; 6],
    /// Set when the layer stack's material compositing changed, so `update`
    /// re-uploads the material map (roughness/metallic/emissive/ao) to the GPU
    /// texture separate from the albedo atlas.
    needs_material_upload: bool,
    /// Wall clock of the last material/height map GPU upload, used to coalesce
    /// the (expensive) full-atlas recomposites across rapid paint dabs.
    last_material_upload: std::time::Instant,
    /// Viewport render-resolution scale. While the user interacts (orbit, pan,
    /// zoom, paint) the 3D scene renders at a reduced internal resolution for
    /// FPS; it snaps back to 1.0 (full res) the moment interaction stops.
    vp_scale: f32,
    /// The active brush: footprint kind, size (screen px), hardness, spacing,
    /// opacity, color, build-up flag, Paint/Erase mode and the optional
    /// texture stamp (+ rotation/flips). Persisted through `UiMemory`'s legacy
    /// scalar fields (translated via shape codes at load/save); the sprite is
    /// session-only.
    brush: crate::brush::Brush,
    /// Loaded environment/skybox map (HDRI or plain equirect photo; session-only);
    /// `None` = analytic sky.
    env_path: Option<String>,
    /// Folder-backed brush library (built-ins + files dropped in `brushes/`).
    brushes: crate::brushes::BrushLibrary,
    /// Cached thumbnail textures keyed by `brushes.entries` index; rebuilt
    /// (keyed off the library signature) when the folder contents change.
    brush_thumbs: HashMap<usize, TextureHandle>,
    brush_thumb_sig: String,
    /// Selected category filter ("All" or a category name) for the panel.
    brush_filter: String,
    /// Color palette library (persisted to a sidecar file next to the UI
    /// layout). The active palette feeds the RMB brush menu's swatch grid.
    palettes: Vec<crate::palette::Palette>,
    active_palette: usize,
    /// Undo/redo stacks for palette-library edits (transient). Palette changes
    /// snapshot the whole library; the Ctrl+Z / Ctrl+Y handlers route here
    /// whenever the most recent action was a palette edit, so a mistakenly
    /// cleared / deleted palette can always be restored.
    palette_prev: Vec<Vec<crate::palette::Palette>>,
    palette_redo: Vec<Vec<crate::palette::Palette>>,
    palette_edit_pending: bool,
    /// Armed confirmation for destructive palette actions (Clear / Delete):
    /// an action id + the instant it was armed. The button reads "Confirm?"
    /// until the arming deadline passes.
    palette_confirm: Option<(u8, std::time::Instant)>,
    /// PBR material (roughness/metallic/emissive/AO + lighting), persisted.
    material: crate::render::Material,
    /// Runtime state of the brush stroke in progress.
    stroke: Option<StrokeState>,
    /// Cached geodesic unwrap of the hover pose (anchor focus): rebuilt when
    /// the cursor crosses into a new triangle or at most every 100 ms, so the
    /// cursor overlay previews the along-surface phase field without paying
    /// for a full unfold every frame. The stored anchor position/triangle
    /// stamp the field it was built for.
    hover_unwrap: Option<(
        std::time::Instant,
        u32,
        glam::Vec3,
        crate::paint::SurfaceUnwrap,
    )>,
    /// Cached egui texture of the Brushes-panel live preview (the current
    /// stamp rasterized off-screen: footprint shape, color/opacity, and for
    /// texture brushes the tiled sprite with its repeat size, rotation and
    /// flips). Keyed by the brush state that changes the look.
    settings_preview: Option<(u64, TextureHandle)>,
    /// Cached egui texture of the 2D-canvas cursor / 3D viewport cursor /
    /// T-bar brush icon: the exact stamp the current brush paints (frame
    /// shape + tiled sprite + rotation + flips, tinted, no checkerboard), so
    /// the floating cursors show the WYSIWYG masked dab — never the raw round
    /// sprite — and selecting Square/Diamond changes the frame live. Keyed by
    /// `settings_preview_sig` plus the dab radius in px.
    settings_preview_flat: Option<(u64, u32, TextureHandle)>,
    /// Undo/redo history of full-texture snapshots.
    history: EditHistory,
    /// Target atlas resolution (longest side) for Resize / Blank (not persisted).
    atlas_res: u32,
    /// Right-click brush menu popped up over the viewport (not persisted).
    brush_menu_open: bool,
    brush_menu_pos: Option<egui::Pos2>,
    /// One-shot guard so the floating brush menu is rendered by exactly one
    /// panel per frame (the 3D viewport and the 2D texture canvas can both be
    /// docked — a second Area with the same id would collide). Reset each frame.
    brush_menu_rendered: bool,
    /// Lazily-loaded eyedropper icon (lucide pipette, ISC licensed).
    pick_icon: Option<TextureHandle>,
    /// Lazily-loaded shared UI icons (lucide, ISC licensed).
    icons: Option<IconSet>,
    /// Set when the CPU atlas changed and must be re-uploaded before the next render.
    needs_texture_upload: bool,
    /// Cached egui copy of the current albedo texture, rebuilt when `preview_gen` bumps.
    texture_preview: Option<PreviewTexture>,
    preview_gen: u64,
    /// Dirty albedo region `(x0, y0, w, h, atlas_w, atlas_h)` captured at the
    /// last flush so the 2D preview can patch just that rectangle into its
    /// cached egui texture (`set_partial`) instead of recompositing the whole
    /// atlas. The stored atlas dims guard against a resize/blank making the
    /// stale region invalid; a full rebuild is used then.
    preview_patch: Option<(u32, u32, u32, u32, u32, u32)>,
    show_uv_overlay: bool,
    /// 3D viewport UV checkerboard / grid overlays (shader-driven).
    show_uv_checker_3d: bool,
    show_uv_grid_3d: bool,
    /// Split lock: restrict 3D stamps to the mesh part connected to the face
    /// under the brush, so a stroke never bleeds onto a separate model part.
    split_lock: bool,
    /// Show the in-viewport vertical tool strip (its translucent T-bar).
    show_tool_strip: bool,
    /// Slide-in/out animation progress of the T-bar: 0 = fully hidden off the
    /// left edge, 1 = fully visible (transient, not persisted).
    tool_strip_anim: f32,
    /// Show the Texture preview's in-panel brush picker (a second tool strip,
    /// just like the 3D viewport's T-bar) — but toggled per-panel with T.
    show_brush_picker: bool,
    /// Slide-in/out animation progress of the 2D brush picker (0..1).
    brush_picker_anim: f32,
    /// Keep the 3D viewport's horizontal overlay bar (UV checker / grid
    /// toggles) visible at the top-right. The bar is only ever shown while
    /// this is set — never on hover. Transient.
    show_vp_overlay_bar: bool,
    /// UI chrome theme, chosen in Preferences (persisted).
    theme_pref: ThemePref,
    /// Category open in the Preferences panel (transient).
    prefs_tab: PrefsTab,
    /// Remappable keyboard shortcuts (persisted).
    shortcuts: Shortcuts,
    /// A shortcut capture in progress from the Preferences window; while set,
    /// the app's own key handlers yield so the pressed key is captured instead.
    recording: Option<ShortcutAction>,
    /// Armed confirmation for "Reset all shortcuts" (Preferences): the instant
    /// the arm button was clicked. The button reads "Confirm?" until the
    /// deadline passes, mirroring the palette panel's clear/delete flow.
    prefs_reset_armed: Option<std::time::Instant>,
    /// Layer currently being renamed (transient): shows an inline text field in
    /// place of its label. `None` when no rename is in progress.
    renaming: Option<usize>,
    /// Edit buffer for the in-progress rename.
    rename_buf: String,
    /// One-shot flag: request keyboard focus in the rename field on the frame
    /// right after a rename is started (it is cleared as soon as it is used).
    rename_grab_focus: bool,
    /// Stroke in progress inside the 2D texture preview (screen-space paint
    /// positions), kept separate from the 3D viewport's `stroke`.
    stroke_2d: Option<StrokeState>,
    /// 2D texture-preview camera (pan + zoom), image-editor style.
    canvas2d: Canvas2D,
    /// Camera view to restore on the first frame (from a previous session).
    restore_view: Option<(glam::Vec3, glam::Vec3, f32, bool)>,
    status: String,
}

/// Brush stroke in progress. Dabs are spaced a fixed number of screen pixels
/// apart (`Core::brush_spacing`); `acc`/`next_t` carry the leftover travel so
/// slow or jittery cursor motion doesn't clump dabs together, and fast flicks
/// still get evenly spaced dabs along their path.
struct StrokeState {
    /// Cursor position on the previous frame (distance accrues from here).
    last: egui::Pos2,
    /// Where the stroke began: with Shift held, dabs ride the straight line
    /// from here to the cursor instead of following the drag.
    start: egui::Pos2,
    /// Where the most recent freehand dab landed; the next one is `spacing`
    /// past it along the current travel direction.
    last_dab: egui::Pos2,
    /// Freehand: path length travelled since the last dab was laid down.
    acc: f32,
    /// Shift-line: distance along the straight line already covered by dabs.
    next_t: f32,
    /// Per-geometry acceleration (convexity, bounding sphere, occlusion grid,
    /// split-lock components) built once at stroke start from the mesh under
    /// the brush, so the O(V·F) convexity scan and the occlusion index are not
    /// recomputed for every dab of the stroke. The split lock rides inside it:
    /// when locked, every dab of this stroke stays on the part connected to
    /// the seed face captured on the mouse-down press — it never chases the
    /// cursor onto a different part (dabs that land elsewhere paint nothing).
    accel: Option<crate::paint::StampAccel>,
    /// Per-stroke alpha buffer for the replace blend: tracks the maximum alpha
    /// this stroke has applied to each texel, so later dabs cap rather than
    /// stack. Allocated for non-accumulative brushes and for pattern-aligned
    /// texture strokes (the anchored sprite must stay flat across overlap).
    /// Indexed as `y * width + x`.
    stroke_alpha: Option<Vec<u8>>,
    /// Pattern-locked anchor captured at the stroke's first dab
    /// ([`crate::brush::PatternLock::Aligned`] texture brushes only): the
    /// sprite phase stays glued to this point for the whole stroke.
    pattern: Option<crate::brush::PatternAnchor>,
    /// Geodesic surface unwrap built once at stroke start for a
    /// [`PatternAnchor::Surface`] stroke: per-vertex phases that unfold the
    /// mesh patch around the anchor along the surface, so curved bodies paint
    /// the world-uniform tile without the fold/chord collapse, and the cursor
    /// overlay previews the very same field.
    unwrap: Option<crate::paint::SurfaceUnwrap>,
}

/// The 2D Texture preview shows the classic alpha checkerboard behind
/// transparent (erased) texels — a preview backdrop only. The 3D viewport
/// renders fully erased texels transparent; semi-transparent ones show the
/// surface behind them (source-over), nothing is painted over them.
struct PreviewTexture {
    gen: u64,
    handle: TextureHandle,
}

/// Camera of the 2D texture-preview canvas: how the atlas is placed and scaled
/// inside the panel, image-editor style (pan with middle mouse, zoom with the
/// scroll wheel, everything stays hover-gated so it never fights the 3D tab).
struct Canvas2D {
    /// Canvas-local offset of the image center from the panel center.
    center: egui::Vec2,
    /// Screen px per texel (1.0 = 100%, zoomed out = small).
    zoom: f32,
    /// Re-fit to the panel on the next frame (set after load/resize/blank).
    needs_fit: bool,
}

impl Default for Canvas2D {
    fn default() -> Self {
        Self {
            center: egui::Vec2::ZERO,
            zoom: 1.0,
            needs_fit: true,
        }
    }
}

/// A full copy of one layer's atlas + attributes, used to restore layer state.
#[derive(Clone)]
struct LayerSnapshot {
    name: String,
    visible: bool,
    locked: bool,
    opacity: f32,
    blend: crate::io::BlendMode,
    roughness: f32,
    metallic: f32,
    emissive: f32,
    ambient_occlusion: f32,
    height: f32,
    bump_strength: f32,
    clearcoat: f32,
    clearcoat_roughness: f32,
    specular_ior: f32,
    emissive_color: [f32; 3],
    texture: crate::io::TextureData,
}

/// A full copy of the whole layer stack (all layers + the active index), used
/// to restore texture state for undo/redo. One snapshot covers every edit:
/// paint strokes, layer add/delete/duplicate/reorder, opacity/visibility.
#[derive(Clone)]
struct LayerStackSnapshot {
    layers: Vec<LayerSnapshot>,
    active_layer: usize,
}

/// Bounded undo/redo history. `undo` holds states that can restore *to*; the
/// most recent is last. Pushing a new snapshot clears the redo stack.
struct EditHistory {
    undo: Vec<LayerStackSnapshot>,
    redo: Vec<LayerStackSnapshot>,
    limit: usize,
}

impl EditHistory {
    fn new(limit: usize) -> Self {
        Self {
            undo: Vec::new(),
            redo: Vec::new(),
            limit: limit.max(1),
        }
    }

    fn record(&mut self, snap: LayerStackSnapshot) {
        self.undo.push(snap);
        if self.undo.len() > self.limit {
            self.undo.remove(0);
        }
        self.redo.clear();
    }

    fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// Pops the state to restore to, pushing `current` onto the redo stack.
    fn undo(&mut self, current: LayerStackSnapshot) -> Option<LayerStackSnapshot> {
        let snap = self.undo.pop()?;
        self.redo.push(current);
        Some(snap)
    }

    /// Pops the state to restore to, pushing `current` onto the undo stack.
    fn redo(&mut self, current: LayerStackSnapshot) -> Option<LayerStackSnapshot> {
        let snap = self.redo.pop()?;
        self.undo.push(current);
        Some(snap)
    }

    fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
    }
}

fn snapshot_of(mesh: &MeshData) -> LayerStackSnapshot {
    LayerStackSnapshot {
        layers: mesh
            .layers
            .iter()
            .map(|l| LayerSnapshot {
                name: l.name.clone(),
                visible: l.visible,
                locked: l.locked,
                opacity: l.opacity,
                blend: l.blend,
                roughness: l.roughness,
                metallic: l.metallic,
                emissive: l.emissive,
                ambient_occlusion: l.ambient_occlusion,
                height: l.height,
                bump_strength: l.bump_strength,
                clearcoat: l.clearcoat,
                clearcoat_roughness: l.clearcoat_roughness,
                specular_ior: l.specular_ior,
                emissive_color: l.emissive_color,
                texture: l.texture.clone(),
            })
            .collect(),
        active_layer: mesh.active_layer,
    }
}

/// Persisted UI state (dock layout, tool settings, camera).
#[derive(serde::Serialize, serde::Deserialize)]
struct UiMemory {
    dock: DockState<Panel>,
    panel_visible: Vec<bool>,
    active_tool: usize,
    channels: [bool; 6],
    brush_size: f32,
    brush_hardness: f32,
    brush_opacity: f32,
    brush_spacing: f32,
    brush_color: [u8; 4],
    show_uv_overlay: bool,
    show_uv_checker_3d: bool,
    show_uv_grid_3d: bool,
    /// The brush footprint (shape + rotation/flip; the sprite itself is not
    /// persisted since it lives in the user's filesystem).
    brush_shape: u8,
    brush_rotation: f32,
    brush_flip_x: bool,
    brush_flip_y: bool,
    /// [`crate::brush::PatternLock`] as a `u8` (0 = Dab, 1 = Aligned).
    /// Defaults on older configs that predate pattern-lock.
    #[serde(default)]
    brush_pattern_lock: u8,
    /// Texture-repeat multiplier + fixed-size lock for sprite/texture brushes.
    /// Defaults (scale) on configs that predate texture-size controls.
    #[serde(default)]
    brush_texture_scale: f32,
    #[serde(default)]
    brush_texture_locked: bool,
    material: crate::render::Material,
    show_tool_strip: bool,
    camera: Option<CameraState>,
    /// Theme preference as a `u8` (0 = System, 1 = Dark, 2 = Light). Defaults
    /// on older configs that predate Preferences.
    #[serde(default)]
    theme: u8,
    /// Remappable shortcut layout. Defaults on older configs.
    #[serde(default, deserialize_with = "shortcuts_or_default")]
    shortcuts: Shortcuts,
}

/// Decode `Shortcuts`, falling back to the stock defaults if the stored blob
/// is absent, older/corrupt (pre-map format was positional and mis-reordered
/// on upgrade), or otherwise unreadable — so a bad shortcuts section can never
/// wipe the rest of the UI layout.
fn shortcuts_or_default<'de, D>(d: D) -> Result<Shortcuts, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(<Shortcuts as serde::Deserialize>::deserialize(d).unwrap_or_default())
}

#[derive(serde::Serialize, serde::Deserialize)]
struct CameraState {
    eye: [f32; 3],
    target: [f32; 3],
    radius: f32,
    /// Orthographic projection flag (nav-gizmo axis views). Defaults on older
    /// configs that predate the gizmo.
    #[serde(default)]
    ortho: bool,
}

/// The app's UI chrome theme, chosen in Preferences (persisted as a `u8`).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
enum ThemePref {
    /// Follow the OS light/dark preference.
    #[default]
    System,
    Dark,
    Light,
}

/// Category shown in the Preferences panel's tabbed layout (transient UI
/// state — which page the user is on, not a persisted preference).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
enum PrefsTab {
    /// Theme selection (System / Dark / Light).
    #[default]
    Appearance,
    /// Viewport / canvas display toggles.
    Viewport,
    /// Remappable keyboard shortcuts.
    Shortcuts,
}

pub const ACCENT: egui::Color32 = egui::Color32::from_rgb(82, 158, 228);
pub const ACCENT_HOVER: egui::Color32 = egui::Color32::from_rgb(110, 180, 244);
pub const ACCENT_DIM: egui::Color32 = egui::Color32::from_rgb(52, 108, 162);

/// Shared corner radii so every panel, card and pill agrees.
pub const RADIUS_CARD: u8 = 6;
pub const RADIUS_CONTROL: u8 = 5;
pub const RADIUS_CHIP: u8 = 4;
pub const RADIUS_PILL: u8 = 10;

/// Theme-aware colors for the app's custom-painted "chrome" — the toolbar,
/// viewport overlays, tool strip, layer cards and canvases. egui's stock
/// widgets follow [`blender_dark_visuals`] / [`blender_light_visuals`]; this is
/// the matching palette for everything drawn by hand, so both themes read as
/// intentional instead of only the dark one working.
#[derive(Clone, Copy)]
pub struct UiPalette {
    pub dark: bool,
    /// Toolbar / menu frame background and its text.
    pub chrome: egui::Color32,
    pub chrome_text: egui::Color32,
    pub chrome_text_weak: egui::Color32,
    /// Panel list cards (layers, brush grid, palette).
    pub card: egui::Color32,
    pub card_active: egui::Color32,
    pub card_border: egui::Color32,
    pub card_border_active: egui::Color32,
    /// Recessed image wells (texture canvas, thumbnail backdrops).
    pub well: egui::Color32,
    pub well_border: egui::Color32,
    /// Translucent floating chrome over the viewport.
    pub overlay: egui::Color32,
    pub overlay_border: egui::Color32,
    pub overlay_text: egui::Color32,
    /// Tool-strip buttons.
    pub control: egui::Color32,
    pub control_hover: egui::Color32,
    pub control_border: egui::Color32,
    pub control_text: egui::Color32,
    pub warn: egui::Color32,
    pub checker_a: egui::Color32,
    pub checker_b: egui::Color32,
    pub axis_x: egui::Color32,
    pub axis_y: egui::Color32,
    pub axis_z: egui::Color32,
}

impl UiPalette {
    pub fn of(ui: &Ui) -> Self {
        Self::for_dark(ui.visuals().dark_mode)
    }

    pub fn for_dark(dark: bool) -> Self {
        if dark {
            Self {
                dark,
                chrome: egui::Color32::from_rgb(22, 24, 28),
                chrome_text: egui::Color32::from_rgb(214, 218, 226),
                chrome_text_weak: egui::Color32::from_rgb(146, 151, 161),
                card: egui::Color32::from_rgb(30, 32, 37),
                card_active: egui::Color32::from_rgb(42, 45, 52),
                card_border: egui::Color32::from_rgb(42, 45, 52),
                card_border_active: egui::Color32::from_rgb(76, 84, 96),
                well: egui::Color32::from_rgb(18, 19, 23),
                well_border: egui::Color32::from_rgb(52, 56, 64),
                overlay: egui::Color32::from_rgba_unmultiplied(16, 18, 23, 205),
                overlay_border: egui::Color32::from_white_alpha(26),
                overlay_text: egui::Color32::from_rgb(228, 231, 237),
                control: egui::Color32::from_rgb(36, 39, 45),
                control_hover: egui::Color32::from_rgb(52, 56, 64),
                control_border: egui::Color32::from_rgb(56, 60, 68),
                control_text: egui::Color32::from_rgb(230, 233, 239),
                warn: egui::Color32::from_rgb(233, 169, 98),
                checker_a: egui::Color32::from_gray(96),
                checker_b: egui::Color32::from_gray(80),
                axis_x: egui::Color32::from_rgb(255, 82, 96),
                axis_y: egui::Color32::from_rgb(112, 232, 112),
                axis_z: egui::Color32::from_rgb(92, 158, 255),
            }
        } else {
            Self {
                dark,
                chrome: egui::Color32::from_rgb(230, 232, 237),
                chrome_text: egui::Color32::from_rgb(38, 42, 50),
                chrome_text_weak: egui::Color32::from_rgb(104, 110, 121),
                card: egui::Color32::from_rgb(237, 239, 243),
                card_active: egui::Color32::from_rgb(220, 226, 236),
                card_border: egui::Color32::from_rgb(208, 212, 220),
                card_border_active: egui::Color32::from_rgb(150, 178, 205),
                well: egui::Color32::from_rgb(198, 202, 210),
                well_border: egui::Color32::from_rgb(172, 177, 187),
                overlay: egui::Color32::from_rgba_unmultiplied(248, 249, 252, 232),
                overlay_border: egui::Color32::from_black_alpha(28),
                overlay_text: egui::Color32::from_rgb(40, 44, 52),
                control: egui::Color32::from_rgb(224, 227, 233),
                control_hover: egui::Color32::from_rgb(206, 212, 221),
                control_border: egui::Color32::from_rgb(196, 201, 210),
                control_text: egui::Color32::from_rgb(38, 42, 50),
                warn: egui::Color32::from_rgb(176, 112, 32),
                checker_a: egui::Color32::from_gray(211),
                checker_b: egui::Color32::from_gray(190),
                axis_x: egui::Color32::from_rgb(214, 54, 72),
                axis_y: egui::Color32::from_rgb(54, 170, 74),
                axis_z: egui::Color32::from_rgb(52, 116, 214),
            }
        }
    }
}

/// A small accent tick followed by a bold label — the one heading style used
/// at the top of every panel, so panels share a visual rhythm.
fn panel_heading(ui: &mut Ui, text: &str) {
    let pal = UiPalette::of(ui);
    ui.horizontal(|ui| {
        let (rect, _) = ui.allocate_exact_size(egui::vec2(3.0, 15.0), egui::Sense::hover());
        ui.painter().rect_filled(rect, RADIUS_CHIP, ACCENT);
        ui.add_space(2.0);
        ui.label(
            egui::RichText::new(text)
                .size(13.0)
                .strong()
                .color(pal.chrome_text),
        );
    });
}

/// A full-width rule with the label centred on it — used to divide shortcut
/// categories in the Preferences panel, where a plain text sub-heading would
/// read as another row.
fn ruled_heading(ui: &mut Ui, text: &str) {
    let pal = UiPalette::of(ui);
    ui.add_space(8.0);
    ui.horizontal(|ui| {
        let (rect, _) =
            ui.allocate_exact_size(egui::vec2(ui.available_width(), 1.0), egui::Sense::hover());
        let mid = rect.center();
        ui.painter().hline(
            rect.left()..=mid.x - 30.0,
            mid.y,
            egui::Stroke::new(1.0, pal.card_border),
        );
        ui.colored_label(
            pal.chrome_text_weak,
            egui::RichText::new(text).size(10.5).strong(),
        );
        ui.painter().hline(
            mid.x + 30.0..=rect.right(),
            mid.y,
            egui::Stroke::new(1.0, pal.card_border),
        );
    });
    ui.add_space(4.0);
}

fn dark_shadows(dark: bool) -> (egui::epaint::Shadow, egui::epaint::Shadow) {
    use egui::epaint::Shadow;
    if dark {
        (
            Shadow {
                offset: [0, 10],
                blur: 28,
                spread: 0,
                color: egui::Color32::from_black_alpha(130),
            },
            Shadow {
                offset: [0, 6],
                blur: 18,
                spread: 0,
                color: egui::Color32::from_black_alpha(110),
            },
        )
    } else {
        (
            Shadow {
                offset: [0, 8],
                blur: 24,
                spread: 0,
                color: egui::Color32::from_black_alpha(38),
            },
            Shadow {
                offset: [0, 5],
                blur: 16,
                spread: 0,
                color: egui::Color32::from_black_alpha(30),
            },
        )
    }
}

pub fn blender_dark_visuals() -> egui::Visuals {
    let mut v = egui::Visuals::dark();
    v.panel_fill = egui::Color32::from_rgb(31, 33, 37);
    v.window_fill = egui::Color32::from_rgb(24, 26, 30);
    v.faint_bg_color = egui::Color32::from_rgb(37, 39, 44);
    v.extreme_bg_color = egui::Color32::from_rgb(15, 16, 19);
    v.code_bg_color = egui::Color32::from_rgb(23, 24, 28);

    v.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(34, 36, 41);
    v.widgets.noninteractive.weak_bg_fill = egui::Color32::from_rgb(30, 32, 36);
    v.widgets.noninteractive.bg_stroke =
        egui::Stroke::new(1.0, egui::Color32::from_rgb(46, 48, 54));
    v.widgets.noninteractive.fg_stroke =
        egui::Stroke::new(1.0, egui::Color32::from_rgb(176, 179, 187));
    v.widgets.noninteractive.corner_radius = egui::CornerRadius::same(RADIUS_CONTROL);

    v.widgets.inactive.bg_fill = egui::Color32::from_rgb(47, 50, 57);
    v.widgets.inactive.weak_bg_fill = egui::Color32::from_rgb(40, 42, 48);
    v.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(57, 61, 69));
    v.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(219, 222, 230));
    v.widgets.inactive.corner_radius = egui::CornerRadius::same(RADIUS_CONTROL);

    v.widgets.hovered.bg_fill = egui::Color32::from_rgb(62, 66, 74);
    v.widgets.hovered.weak_bg_fill = egui::Color32::from_rgb(55, 59, 66);
    v.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(82, 88, 99));
    v.widgets.hovered.fg_stroke = egui::Stroke::new(1.0, egui::Color32::WHITE);
    v.widgets.hovered.corner_radius = egui::CornerRadius::same(RADIUS_CONTROL);

    v.widgets.active.bg_fill = ACCENT;
    v.widgets.active.weak_bg_fill = ACCENT_DIM;
    v.widgets.active.bg_stroke = egui::Stroke::new(1.0, ACCENT);
    v.widgets.active.fg_stroke = egui::Stroke::new(1.0, egui::Color32::WHITE);
    v.widgets.active.corner_radius = egui::CornerRadius::same(RADIUS_CONTROL);

    v.widgets.open.bg_fill = egui::Color32::from_rgb(41, 44, 50);
    v.widgets.open.corner_radius = egui::CornerRadius::same(RADIUS_CONTROL);

    v.selection.bg_fill = ACCENT;
    v.selection.stroke = egui::Stroke::new(1.0, egui::Color32::WHITE);

    v.window_corner_radius = egui::CornerRadius::same(RADIUS_CARD);
    v.menu_corner_radius = egui::CornerRadius::same(RADIUS_CARD);
    v.window_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(46, 49, 56));
    v.text_cursor.stroke = egui::Stroke::new(2.0, ACCENT);
    (v.window_shadow, v.popup_shadow) = dark_shadows(true);

    v
}

pub fn blender_light_visuals() -> egui::Visuals {
    let mut v = egui::Visuals::light();
    v.panel_fill = egui::Color32::from_rgb(241, 242, 246);
    v.window_fill = egui::Color32::from_rgb(233, 235, 240);
    v.faint_bg_color = egui::Color32::from_rgb(248, 248, 250);
    v.extreme_bg_color = egui::Color32::from_rgb(214, 217, 224);
    v.code_bg_color = egui::Color32::from_rgb(238, 239, 243);

    v.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(233, 235, 240);
    v.widgets.noninteractive.weak_bg_fill = egui::Color32::from_rgb(239, 240, 244);
    v.widgets.noninteractive.bg_stroke =
        egui::Stroke::new(1.0, egui::Color32::from_rgb(214, 217, 224));
    v.widgets.noninteractive.fg_stroke =
        egui::Stroke::new(1.0, egui::Color32::from_rgb(96, 100, 110));
    v.widgets.noninteractive.corner_radius = egui::CornerRadius::same(RADIUS_CONTROL);

    v.widgets.inactive.bg_fill = egui::Color32::from_rgb(226, 228, 234);
    v.widgets.inactive.weak_bg_fill = egui::Color32::from_rgb(233, 235, 240);
    v.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(206, 209, 217));
    v.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(46, 49, 56));
    v.widgets.inactive.corner_radius = egui::CornerRadius::same(RADIUS_CONTROL);

    v.widgets.hovered.bg_fill = egui::Color32::from_rgb(202, 207, 216);
    v.widgets.hovered.weak_bg_fill = egui::Color32::from_rgb(211, 215, 223);
    v.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(178, 184, 195));
    v.widgets.hovered.fg_stroke = egui::Stroke::new(1.0, egui::Color32::BLACK);
    v.widgets.hovered.corner_radius = egui::CornerRadius::same(RADIUS_CONTROL);

    v.widgets.active.bg_fill = ACCENT_DIM;
    v.widgets.active.weak_bg_fill = ACCENT_DIM;
    v.widgets.active.bg_stroke = egui::Stroke::new(1.0, ACCENT_DIM);
    v.widgets.active.fg_stroke = egui::Stroke::new(1.0, egui::Color32::WHITE);
    v.widgets.active.corner_radius = egui::CornerRadius::same(RADIUS_CONTROL);

    v.widgets.open.bg_fill = egui::Color32::from_rgb(233, 235, 240);
    v.widgets.open.corner_radius = egui::CornerRadius::same(RADIUS_CONTROL);

    v.selection.bg_fill = ACCENT;
    v.selection.stroke = egui::Stroke::new(1.0, egui::Color32::WHITE);

    v.window_corner_radius = egui::CornerRadius::same(RADIUS_CARD);
    v.menu_corner_radius = egui::CornerRadius::same(RADIUS_CARD);
    v.window_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(200, 203, 211));
    v.text_cursor.stroke = egui::Stroke::new(2.0, ACCENT_DIM);
    (v.window_shadow, v.popup_shadow) = dark_shadows(false);

    v
}

fn apply_theme(ctx: &egui::Context, pref: ThemePref) {
    let (theme, visuals) = match pref {
        ThemePref::Light => (egui::Theme::Light, blender_light_visuals()),
        ThemePref::Dark | ThemePref::System => (egui::Theme::Dark, blender_dark_visuals()),
    };
    ctx.set_theme(theme);
    ctx.set_visuals_of(theme, visuals);
    ctx.style_mut_of(theme, |style| {
        // One spacing scale for the whole app: comfortable hit targets, a
        // uniform gutter, and a consistent default slider width (panels may
        // narrow it). Kept compact so selection highlights fit inside cards.
        style.spacing.item_spacing = egui::vec2(8.0, 4.0);
        style.spacing.button_padding = egui::vec2(8.0, 3.0);
        style.spacing.interact_size = egui::vec2(36.0, 20.0);
        style.spacing.slider_width = 120.0;
        style.spacing.combo_width = 132.0;
        style.spacing.indent = 18.0;
        style.spacing.menu_margin = egui::Margin::same(6);
    });
}

/// A remappable keyboard binding. The key is stored as its index into
/// [`egui::Key::ALL`] (egui's `Key` only serializes behind its `serde`
/// feature) and the modifiers as plain bits, so a bind round-trips through the
/// persisted UI config.
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
struct KeyBind {
    /// Index into [`egui::Key::ALL`]; `usize::MAX` = unbound.
    key: usize,
    ctrl: bool,
    shift: bool,
    alt: bool,
    /// Command (⌘ on macOS); mirrors Ctrl on Windows/Linux.
    command: bool,
    /// The Mac ⌘ key specifically (false on other platforms).
    mac_cmd: bool,
}

impl KeyBind {
    fn unbound() -> Self {
        Self {
            key: usize::MAX,
            ctrl: false,
            shift: false,
            alt: false,
            command: false,
            mac_cmd: false,
        }
    }

    fn new(key: egui::Key, modifiers: egui::Modifiers) -> Self {
        Self {
            key: egui::Key::ALL
                .iter()
                .position(|k| *k == key)
                .unwrap_or(usize::MAX),
            ctrl: modifiers.ctrl,
            shift: modifiers.shift,
            alt: modifiers.alt,
            command: modifiers.command,
            mac_cmd: modifiers.mac_cmd,
        }
    }

    /// Ctrl/Meta (and never Shift/Alt) — the classic "Ctrl+<key>" binding.
    fn ctrl(key: egui::Key) -> Self {
        Self::new(
            key,
            egui::Modifiers {
                ctrl: true,
                shift: false,
                alt: false,
                command: true,
                mac_cmd: false,
            },
        )
    }

    /// Ctrl/Meta+Shift+<key>.
    fn ctrl_shift(key: egui::Key) -> Self {
        Self::new(
            key,
            egui::Modifiers {
                ctrl: true,
                shift: true,
                alt: false,
                command: true,
                mac_cmd: false,
            },
        )
    }

    fn is_bound(&self) -> bool {
        self.key < egui::Key::ALL.len()
    }

    fn key_of(&self) -> egui::Key {
        egui::Key::ALL
            .get(self.key)
            .copied()
            .unwrap_or(egui::Key::A)
    }

    fn modifiers_of(&self) -> egui::Modifiers {
        egui::Modifiers {
            ctrl: self.ctrl,
            shift: self.shift,
            alt: self.alt,
            command: self.command,
            mac_cmd: self.mac_cmd,
        }
    }

    fn label(&self) -> String {
        if !self.is_bound() {
            return "None".to_string();
        }
        let mut parts: Vec<String> = Vec::new();
        if self.command && !self.ctrl {
            parts.push("Cmd".to_string());
        } else if self.ctrl {
            parts.push("Ctrl".to_string());
        }
        if self.alt {
            parts.push("Alt".to_string());
        }
        if self.shift {
            parts.push("Shift".to_string());
        }
        parts.push(key_name(self.key_of()));
        parts.join("+")
    }
}

fn key_name(key: egui::Key) -> String {
    key.name().to_string()
}

/// Exposed, remappable actions. The actual key layout is user-configurable from
/// Preferences; defaults match the historic hard-coded keys.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ShortcutAction {
    /// Open a .gltf/.glb model.
    OpenModel,
    /// Open a .pixforge project file.
    OpenProject,
    /// Save the current project.
    SaveProject,
    /// Load an environment/skybox HDRI.
    OpenEnvironment,
    /// Select the Brush tool.
    SelectBrush,
    /// Select the Eraser tool.
    SelectEraser,
    /// Select the Fill tool.
    SelectFill,
    /// Select the Pick (eyedropper) tool.
    SelectPicker,
    /// Select the Rect stamp tool.
    SelectRect,
    /// Increase the brush size.
    BrushSizeUp,
    /// Decrease the brush size.
    BrushSizeDown,
    /// Increase the brush opacity.
    BrushOpacityUp,
    /// Decrease the brush opacity.
    BrushOpacityDown,
    /// 3D viewport T-bar (tool strip).
    ToggleTools3d,
    /// Texture editor brush picker strip.
    ToggleTools2d,
    /// Show/hide the 3D viewport overlay bar (UV checker / grid).
    ToggleOverlayBar,
    /// Fit the 3D camera to the model.
    Fit3d,
    /// Fit the 2D canvas to the panel.
    Fit2d,
    /// Redo the last undone edit (checked before Undo — it is a modifier
    /// superset, and egui's logical match ignores the extra Shift).
    Redo,
    /// Undo the last texture edit.
    Undo,
}

impl ShortcutAction {
    const ALL: [Self; 20] = [
        Self::OpenModel,
        Self::OpenProject,
        Self::SaveProject,
        Self::OpenEnvironment,
        Self::SelectBrush,
        Self::SelectEraser,
        Self::SelectFill,
        Self::SelectPicker,
        Self::SelectRect,
        Self::BrushSizeUp,
        Self::BrushSizeDown,
        Self::BrushOpacityUp,
        Self::BrushOpacityDown,
        Self::ToggleTools3d,
        Self::ToggleTools2d,
        Self::ToggleOverlayBar,
        Self::Fit3d,
        Self::Fit2d,
        Self::Redo,
        Self::Undo,
    ];

    fn category(self) -> &'static str {
        match self {
            Self::OpenModel | Self::OpenProject | Self::SaveProject | Self::OpenEnvironment => {
                "File"
            }
            Self::SelectBrush
            | Self::SelectEraser
            | Self::SelectFill
            | Self::SelectPicker
            | Self::SelectRect => "Tools",
            Self::BrushSizeUp
            | Self::BrushSizeDown
            | Self::BrushOpacityUp
            | Self::BrushOpacityDown => "Brush",
            Self::ToggleTools3d
            | Self::ToggleTools2d
            | Self::ToggleOverlayBar
            | Self::Fit3d
            | Self::Fit2d => "Viewport",
            Self::Undo | Self::Redo => "Edit",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::OpenModel => "Open Model…",
            Self::OpenProject => "Open Project…",
            Self::SaveProject => "Save Project…",
            Self::OpenEnvironment => "Open Environment…",
            Self::SelectBrush => "Select Brush",
            Self::SelectEraser => "Select Eraser",
            Self::SelectFill => "Select Fill",
            Self::SelectPicker => "Select Pick",
            Self::SelectRect => "Select Rect",
            Self::BrushSizeUp => "Brush size up",
            Self::BrushSizeDown => "Brush size down",
            Self::BrushOpacityUp => "Brush opacity up",
            Self::BrushOpacityDown => "Brush opacity down",
            Self::ToggleTools3d => "Toggle 3D tools bar",
            Self::ToggleTools2d => "Toggle 2D tools bar",
            Self::ToggleOverlayBar => "Toggle UV overlay bar (3D)",
            Self::Fit3d => "Fit camera (3D)",
            Self::Fit2d => "Fit canvas (2D)",
            Self::Undo => "Undo",
            Self::Redo => "Redo",
        }
    }

    fn description(self) -> &'static str {
        match self {
            Self::OpenModel => "Open a .gltf/.glb model from disk",
            Self::OpenProject => "Open a saved .pixforge project (geometry + all layers)",
            Self::SaveProject => "Save the model geometry and all layers to a .pixforge project",
            Self::OpenEnvironment => "Load an HDR/equirect environment map for the 3D scene",
            Self::SelectBrush => "Activate the brush (paint) tool",
            Self::SelectEraser => "Activate the eraser tool",
            Self::SelectFill => "Activate the fill (bucket) tool",
            Self::SelectPicker => "Activate the pick (eyedropper) tool",
            Self::SelectRect => "Activate the rectangular stamp tool",
            Self::BrushSizeUp => "Increase the brush radius (1/10th of size, min 1 px)",
            Self::BrushSizeDown => "Decrease the brush radius (1/10th of size, min 1 px)",
            Self::BrushOpacityUp => "Increase the brush opacity by 5%",
            Self::BrushOpacityDown => "Decrease the brush opacity by 5%",
            Self::ToggleTools3d => {
                "Show/hide the vertical tool strip over the 3D viewport's left edge"
            }
            Self::ToggleTools2d => "Show/hide the brush picker over the texture editor's left edge",
            Self::ToggleOverlayBar => {
                "Pin/unpin the UV checker / grid overlay bar at the 3D viewport's top right"
            }
            Self::Fit3d => "Frame the loaded model in the 3D viewport",
            Self::Fit2d => "Fit the texture atlas to the texture editor panel",
            Self::Undo => "Undo the last texture edit",
            Self::Redo => "Redo the last undone edit",
        }
    }

    /// The brush-tool index this action selects, if it is a tool selection.
    fn tool_index(self) -> Option<usize> {
        match self {
            Self::SelectBrush => Some(0),
            Self::SelectEraser => Some(1),
            Self::SelectFill => Some(2),
            Self::SelectPicker => Some(3),
            Self::SelectRect => Some(4),
            _ => None,
        }
    }

    /// Position in [`Self::ALL`], used to index parallel arrays.
    fn index(self) -> usize {
        Self::ALL.iter().position(|a| *a == self).unwrap()
    }

    /// Stable, human-friendly id used as the persistence key. Never rename
    /// these — adding a new one is fine, and old entries simply become unknown
    /// (silently dropped) or unbound-but-defaulted.
    fn serial(self) -> &'static str {
        match self {
            Self::OpenModel => "open_model",
            Self::OpenProject => "open_project",
            Self::SaveProject => "save_project",
            Self::OpenEnvironment => "open_environment",
            Self::SelectBrush => "select_brush",
            Self::SelectEraser => "select_eraser",
            Self::SelectFill => "select_fill",
            Self::SelectPicker => "select_picker",
            Self::SelectRect => "select_rect",
            Self::BrushSizeUp => "brush_size_up",
            Self::BrushSizeDown => "brush_size_down",
            Self::BrushOpacityUp => "brush_opacity_up",
            Self::BrushOpacityDown => "brush_opacity_down",
            Self::ToggleTools3d => "toggle_tools_3d",
            Self::ToggleTools2d => "toggle_tools_2d",
            Self::ToggleOverlayBar => "toggle_overlay_bar",
            Self::Fit3d => "fit_3d",
            Self::Fit2d => "fit_2d",
            Self::Undo => "undo",
            Self::Redo => "redo",
        }
    }

    fn from_serial(s: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|a| a.serial() == s)
    }
}

/// The persisted shortcut layout (serialized as a *named map* of action id →
/// binding).
///
/// msgpack's struct encoding is positional, so a plain derived struct would
/// silently mis-assign stored bindings whenever fields were added or moved
/// between builds. A keyed map avoids that entirely: adding actions later is
/// backward compatible, unknown keys are ignored, and any action the stored
/// map does not mention falls back to its default binding.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Shortcuts {
    open_model: KeyBind,
    open_project: KeyBind,
    save_project: KeyBind,
    open_environment: KeyBind,
    select_brush: KeyBind,
    select_eraser: KeyBind,
    select_fill: KeyBind,
    select_picker: KeyBind,
    select_rect: KeyBind,
    brush_size_up: KeyBind,
    brush_size_down: KeyBind,
    brush_opacity_up: KeyBind,
    brush_opacity_down: KeyBind,
    toggle_tools_3d: KeyBind,
    toggle_tools_2d: KeyBind,
    toggle_overlay_bar: KeyBind,
    fit_3d: KeyBind,
    fit_2d: KeyBind,
    undo: KeyBind,
    redo: KeyBind,
}

impl Shortcuts {
    /// A fresh layout with every action unbound (used as the base when reading
    /// a partial/older map from disk).
    fn default_binds_unset() -> Self {
        Self {
            open_model: KeyBind::unbound(),
            open_project: KeyBind::unbound(),
            save_project: KeyBind::unbound(),
            open_environment: KeyBind::unbound(),
            select_brush: KeyBind::unbound(),
            select_eraser: KeyBind::unbound(),
            select_fill: KeyBind::unbound(),
            select_picker: KeyBind::unbound(),
            select_rect: KeyBind::unbound(),
            brush_size_up: KeyBind::unbound(),
            brush_size_down: KeyBind::unbound(),
            brush_opacity_up: KeyBind::unbound(),
            brush_opacity_down: KeyBind::unbound(),
            toggle_tools_3d: KeyBind::unbound(),
            toggle_tools_2d: KeyBind::unbound(),
            toggle_overlay_bar: KeyBind::unbound(),
            fit_3d: KeyBind::unbound(),
            fit_2d: KeyBind::unbound(),
            undo: KeyBind::unbound(),
            redo: KeyBind::unbound(),
        }
    }

    fn default_binds() -> Self {
        Self {
            open_model: KeyBind::ctrl(egui::Key::O),
            open_project: KeyBind::ctrl_shift(egui::Key::O),
            save_project: KeyBind::ctrl(egui::Key::S),
            open_environment: KeyBind::unbound(),
            select_brush: KeyBind::new(egui::Key::B, egui::Modifiers::NONE),
            select_eraser: KeyBind::new(egui::Key::E, egui::Modifiers::NONE),
            select_fill: KeyBind::new(egui::Key::G, egui::Modifiers::NONE),
            select_picker: KeyBind::new(egui::Key::I, egui::Modifiers::NONE),
            select_rect: KeyBind::new(egui::Key::R, egui::Modifiers::NONE),
            brush_size_up: KeyBind::new(egui::Key::CloseBracket, egui::Modifiers::NONE),
            brush_size_down: KeyBind::new(egui::Key::OpenBracket, egui::Modifiers::NONE),
            brush_opacity_up: KeyBind::new(egui::Key::CloseCurlyBracket, egui::Modifiers::SHIFT),
            brush_opacity_down: KeyBind::new(egui::Key::OpenCurlyBracket, egui::Modifiers::SHIFT),
            toggle_tools_3d: KeyBind::new(egui::Key::T, egui::Modifiers::NONE),
            toggle_tools_2d: KeyBind::new(egui::Key::T, egui::Modifiers::NONE),
            toggle_overlay_bar: KeyBind::unbound(),
            fit_3d: KeyBind::new(egui::Key::F, egui::Modifiers::NONE),
            fit_2d: KeyBind::new(egui::Key::F, egui::Modifiers::NONE),
            undo: KeyBind::ctrl(egui::Key::Z),
            redo: KeyBind::ctrl_shift(egui::Key::Z),
        }
    }

    fn get(&self, action: ShortcutAction) -> &KeyBind {
        match action {
            ShortcutAction::OpenModel => &self.open_model,
            ShortcutAction::OpenProject => &self.open_project,
            ShortcutAction::SaveProject => &self.save_project,
            ShortcutAction::OpenEnvironment => &self.open_environment,
            ShortcutAction::SelectBrush => &self.select_brush,
            ShortcutAction::SelectEraser => &self.select_eraser,
            ShortcutAction::SelectFill => &self.select_fill,
            ShortcutAction::SelectPicker => &self.select_picker,
            ShortcutAction::SelectRect => &self.select_rect,
            ShortcutAction::BrushSizeUp => &self.brush_size_up,
            ShortcutAction::BrushSizeDown => &self.brush_size_down,
            ShortcutAction::BrushOpacityUp => &self.brush_opacity_up,
            ShortcutAction::BrushOpacityDown => &self.brush_opacity_down,
            ShortcutAction::ToggleTools3d => &self.toggle_tools_3d,
            ShortcutAction::ToggleTools2d => &self.toggle_tools_2d,
            ShortcutAction::ToggleOverlayBar => &self.toggle_overlay_bar,
            ShortcutAction::Fit3d => &self.fit_3d,
            ShortcutAction::Fit2d => &self.fit_2d,
            ShortcutAction::Undo => &self.undo,
            ShortcutAction::Redo => &self.redo,
        }
    }

    fn get_mut(&mut self, action: ShortcutAction) -> &mut KeyBind {
        match action {
            ShortcutAction::OpenModel => &mut self.open_model,
            ShortcutAction::OpenProject => &mut self.open_project,
            ShortcutAction::SaveProject => &mut self.save_project,
            ShortcutAction::OpenEnvironment => &mut self.open_environment,
            ShortcutAction::SelectBrush => &mut self.select_brush,
            ShortcutAction::SelectEraser => &mut self.select_eraser,
            ShortcutAction::SelectFill => &mut self.select_fill,
            ShortcutAction::SelectPicker => &mut self.select_picker,
            ShortcutAction::SelectRect => &mut self.select_rect,
            ShortcutAction::BrushSizeUp => &mut self.brush_size_up,
            ShortcutAction::BrushSizeDown => &mut self.brush_size_down,
            ShortcutAction::BrushOpacityUp => &mut self.brush_opacity_up,
            ShortcutAction::BrushOpacityDown => &mut self.brush_opacity_down,
            ShortcutAction::ToggleTools3d => &mut self.toggle_tools_3d,
            ShortcutAction::ToggleTools2d => &mut self.toggle_tools_2d,
            ShortcutAction::ToggleOverlayBar => &mut self.toggle_overlay_bar,
            ShortcutAction::Fit3d => &mut self.fit_3d,
            ShortcutAction::Fit2d => &mut self.fit_2d,
            ShortcutAction::Undo => &mut self.undo,
            ShortcutAction::Redo => &mut self.redo,
        }
    }
}

impl Default for Shortcuts {
    fn default() -> Self {
        Self::default_binds()
    }
}

impl serde::Serialize for Shortcuts {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = s.serialize_map(Some(ShortcutAction::ALL.len()))?;
        for action in ShortcutAction::ALL {
            map.serialize_entry(action.serial(), self.get(action))?;
        }
        map.end()
    }
}

impl<'de> serde::Deserialize<'de> for Shortcuts {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct Vis;
        impl<'de> serde::de::Visitor<'de> for Vis {
            type Value = Shortcuts;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "a map of shortcut action → binding")
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> Result<Self::Value, A::Error> {
                use serde::de::IgnoredAny;
                let mut sc = Shortcuts::default_binds_unset();
                let mut present = vec![false; ShortcutAction::ALL.len()];
                while let Some(key) = map.next_key::<String>()? {
                    match ShortcutAction::from_serial(&key) {
                        Some(action) => {
                            let bind: KeyBind = map.next_value()?;
                            *sc.get_mut(action) = bind;
                            present[action.index()] = true;
                        }
                        // Unknown/obsolete action ids are skipped, not fatal, so
                        // older or newer configs stay readable either way.
                        None => {
                            let _: IgnoredAny = map.next_value()?;
                        }
                    }
                }
                // Any action the stored map did not mention is treated as
                // "never configured": give it its default binding. This is what
                // makes newly added shortcuts appear with sane defaults when an
                // older config is loaded (and unbound-by-choice entries persist,
                // because those exist as explicit map entries).
                let defaults = Shortcuts::default();
                for action in ShortcutAction::ALL {
                    if !present[action.index()] && defaults.get(action).is_bound() {
                        *sc.get_mut(action) = *defaults.get(action);
                    }
                }
                Ok(sc)
            }
        }
        d.deserialize_map(Vis)
    }
}

struct ViewportResources {
    textures: ViewportTextures,
    texture_id: Option<egui::TextureId>,
    camera: Camera,
}

const TOOLS: [&str; 5] = ["Brush", "Eraser", "Fill", "Pick", "Rect"];

/// In-viewport vertical tool strip (T-bar) dimensions.
const STRIP_W: f32 = 36.0;
const STRIP_TOP_INSET: f32 = 10.0;
const STRIP_PAD: f32 = 6.0;
/// Gap between stacked tool buttons. Must stay in sync with the height math in
/// `tool_strip_rect` so the pill always encloses the last button.
const STRIP_GAP: f32 = 2.0;
/// Seconds to slide the T-bar in/out.
const STRIP_ANIM_S: f32 = 0.16;
/// Extra off-screen distance the bar travels so it fully clears the viewport
/// edge before disappearing (no lingering sliver).
const STRIP_HIDE_EXTRA: f32 = 20.0;

fn load_pick_icon(ctx: &egui::Context) -> Option<TextureHandle> {
    let bytes: &[u8] = include_bytes!("../assets/pipette.png");
    let img = image::load_from_memory(bytes).ok()?.to_rgba8();
    let color_image = egui::ColorImage::from_rgba_unmultiplied(
        [img.width() as usize, img.height() as usize],
        img.as_raw(),
    );
    Some(ctx.load_texture("pick_icon", color_image, egui::TextureOptions::LINEAR))
}

/// Shared UI icon textures (lucide, ISC licensed — same source as the pipette
/// icon). Rasterized from white strokes so every use can be tinted to the
/// current theme. Loaded once, lazily, on the first frame that shows them.
#[allow(dead_code)]
pub struct IconSet {
    pub eye: TextureHandle,
    pub eye_off: TextureHandle,
    pub lock: TextureHandle,
    pub lock_open: TextureHandle,
    pub grip: TextureHandle,
    pub plus: TextureHandle,
    pub copy: TextureHandle,
    pub trash: TextureHandle,
    pub arrow_up: TextureHandle,
    pub arrow_down: TextureHandle,
    pub brush: TextureHandle,
    pub eraser: TextureHandle,
    pub fill: TextureHandle,
    pub pick: TextureHandle,
    pub rect: TextureHandle,
    pub grid: TextureHandle,
    pub undo: TextureHandle,
    pub redo: TextureHandle,
    pub sun: TextureHandle,
    pub layers: TextureHandle,
    pub sliders: TextureHandle,
}

/// Rasterized-64px PNG bytes of the icon set (rendered white for tinting).
const ICON_EYE: &[u8] = include_bytes!("../assets/icons/eye.png");
const ICON_EYE_OFF: &[u8] = include_bytes!("../assets/icons/eye-off.png");
const ICON_LOCK: &[u8] = include_bytes!("../assets/icons/lock.png");
const ICON_LOCK_OPEN: &[u8] = include_bytes!("../assets/icons/lock-open.png");
const ICON_GRIP: &[u8] = include_bytes!("../assets/icons/grip-vertical.png");
const ICON_PLUS: &[u8] = include_bytes!("../assets/icons/plus.png");
const ICON_COPY: &[u8] = include_bytes!("../assets/icons/copy.png");
const ICON_TRASH: &[u8] = include_bytes!("../assets/icons/trash-2.png");
const ICON_ARROW_UP: &[u8] = include_bytes!("../assets/icons/arrow-up.png");
const ICON_ARROW_DOWN: &[u8] = include_bytes!("../assets/icons/arrow-down.png");
const ICON_BRUSH: &[u8] = include_bytes!("../assets/icons/paintbrush.png");
const ICON_ERASER: &[u8] = include_bytes!("../assets/icons/eraser.png");
const ICON_FILL: &[u8] = include_bytes!("../assets/icons/paint-bucket.png");
const ICON_PICK: &[u8] = include_bytes!("../assets/icons/pipette.png");
const ICON_RECT: &[u8] = include_bytes!("../assets/icons/square.png");
const ICON_GRID: &[u8] = include_bytes!("../assets/icons/grid-2x2.png");
const ICON_UNDO: &[u8] = include_bytes!("../assets/icons/undo.png");
const ICON_REDO: &[u8] = include_bytes!("../assets/icons/redo.png");
const ICON_SUN: &[u8] = include_bytes!("../assets/icons/sun.png");
const ICON_LAYERS: &[u8] = include_bytes!("../assets/icons/layers.png");
const ICON_SLIDERS: &[u8] = include_bytes!("../assets/icons/sliders.png");

fn icon_texture(ctx: &egui::Context, name: &str, bytes: &[u8]) -> Option<TextureHandle> {
    let img = image::load_from_memory(bytes).ok()?.to_rgba8();
    let color_image = egui::ColorImage::from_rgba_unmultiplied(
        [img.width() as usize, img.height() as usize],
        img.as_raw(),
    );
    Some(ctx.load_texture(name, color_image, egui::TextureOptions::LINEAR))
}

fn load_icons(ctx: &egui::Context) -> Option<IconSet> {
    Some(IconSet {
        eye: icon_texture(ctx, "icon_eye", ICON_EYE)?,
        eye_off: icon_texture(ctx, "icon_eye_off", ICON_EYE_OFF)?,
        lock: icon_texture(ctx, "icon_lock", ICON_LOCK)?,
        lock_open: icon_texture(ctx, "icon_lock_open", ICON_LOCK_OPEN)?,
        grip: icon_texture(ctx, "icon_grip", ICON_GRIP)?,
        plus: icon_texture(ctx, "icon_plus", ICON_PLUS)?,
        copy: icon_texture(ctx, "icon_copy", ICON_COPY)?,
        trash: icon_texture(ctx, "icon_trash", ICON_TRASH)?,
        arrow_up: icon_texture(ctx, "icon_arrow_up", ICON_ARROW_UP)?,
        arrow_down: icon_texture(ctx, "icon_arrow_down", ICON_ARROW_DOWN)?,
        brush: icon_texture(ctx, "icon_brush", ICON_BRUSH)?,
        eraser: icon_texture(ctx, "icon_eraser", ICON_ERASER)?,
        fill: icon_texture(ctx, "icon_fill", ICON_FILL)?,
        pick: icon_texture(ctx, "icon_pick", ICON_PICK)?,
        rect: icon_texture(ctx, "icon_rect", ICON_RECT)?,
        grid: icon_texture(ctx, "icon_grid", ICON_GRID)?,
        undo: icon_texture(ctx, "icon_undo", ICON_UNDO)?,
        redo: icon_texture(ctx, "icon_redo", ICON_REDO)?,
        sun: icon_texture(ctx, "icon_sun", ICON_SUN)?,
        layers: icon_texture(ctx, "icon_layers", ICON_LAYERS)?,
        sliders: icon_texture(ctx, "icon_sliders", ICON_SLIDERS)?,
    })
}

/// Icon clickable in a compact strip: tinted to `tint`, shows a hover
/// background + pointing-hand cursor when `enabled`, and carries a tooltip.
/// Disabled icons drop the click sense so they can never fire.
fn icon_button(
    ui: &mut Ui,
    tex: &TextureHandle,
    size: f32,
    enabled: bool,
    tint: egui::Color32,
    tooltip: &str,
) -> egui::Response {
    // Allocate + interact first so the hover backdrop can be painted *under*
    // the glyph — the frame painter emits in call order, so a backdrop emitted
    // after the image would cover the icon (drawing pictures an opaque box
    // over every hovered glyph).
    let sense = if enabled {
        egui::Sense::click()
    } else {
        egui::Sense::hover()
    };
    let (rect, mut resp) = ui.allocate_exact_size(egui::vec2(size, size), sense);
    if enabled && resp.hovered() {
        ui.painter().rect_filled(
            rect.expand(3.0),
            4.0,
            ui.visuals().widgets.hovered.weak_bg_fill,
        );
    }
    ui.painter().image(
        tex.id(),
        rect,
        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
        tint,
    );
    resp = resp.on_hover_text(tooltip);
    if enabled && resp.hovered() {
        resp = resp.on_hover_cursor(egui::CursorIcon::PointingHand);
    }
    resp
}

/// Cheap stable hash of a brush sprite, used to key the cached preview texture.
fn sprite_sig(sprite: &crate::io::TextureData) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in &sprite.rgba {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h ^= (sprite.width as u64) << 32 | sprite.height as u64;
    h
}

impl Core {
    fn pick_icon_tex(&mut self, ctx: &egui::Context) -> Option<&TextureHandle> {
        if self.pick_icon.is_none() {
            self.pick_icon = load_pick_icon(ctx);
        }
        self.pick_icon.as_ref()
    }
}

impl PixForgeApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let rs = cc
            .wgpu_render_state
            .as_ref()
            .expect("PixForge requires the wgpu backend");
        let device = rs.device.clone();
        let queue = rs.queue.clone();
        let egui_renderer = rs.renderer.clone();

        let mut renderer = Renderer::new(device.clone(), queue.clone());

        let (mut dock_state, memory) = match load_ui_memory() {
            Some((dock, mem)) => (dock, Some(mem)),
            None => (default_dock(), None),
        };

        let default_mesh =
            crate::io::MeshData::uv_sphere(0.6, 12, 16).with_texture(crate::io::default_albedo());
        let center = mesh_center(&default_mesh);
        let radius = mesh_bounds_radius(&default_mesh, center);
        renderer.set_mesh(&default_mesh);

        let mut core = Core {
            device,
            egui_renderer,
            renderer,
            viewport: None,
            mesh: Some(default_mesh),
            center,
            bounds_radius: radius,
            needs_fit: true,
            panel_visible: vec![true; Panel::ALL.len()],
            active_tool: 0,
            channels: [true, true, false, false, false, false],
            brush: crate::brush::Brush::default(),
            env_path: None,
            brushes: crate::brushes::BrushLibrary::new(brushes_folder()),
            brush_thumbs: HashMap::new(),
            brush_thumb_sig: String::new(),
            brush_filter: "All".to_string(),
            palettes: load_palettes(),
            active_palette: 0,
            palette_prev: Vec::new(),
            palette_redo: Vec::new(),
            palette_edit_pending: false,
            palette_confirm: None,
            material: crate::render::Material::default(),
            stroke: None,
            hover_unwrap: None,
            settings_preview: None,
            settings_preview_flat: None,
            history: EditHistory::new(24),
            atlas_res: 512,
            brush_menu_open: false,
            brush_menu_pos: None,
            brush_menu_rendered: false,
            pick_icon: None,
            icons: None,
            needs_texture_upload: false,
            needs_material_upload: false,
            last_material_upload: std::time::Instant::now(),
            vp_scale: 1.0,
            texture_preview: None,
            preview_gen: 1,
            preview_patch: None,
            show_uv_overlay: true,
            show_uv_checker_3d: false,
            show_uv_grid_3d: false,
            split_lock: false,
            show_tool_strip: true,
            tool_strip_anim: 1.0,
            show_brush_picker: true,
            brush_picker_anim: 1.0,
            show_vp_overlay_bar: true,
            theme_pref: ThemePref::default(),
            prefs_tab: PrefsTab::default(),
            shortcuts: Shortcuts::default(),
            recording: None,
            prefs_reset_armed: None,
            renaming: None,
            rename_buf: String::new(),
            rename_grab_focus: false,
            stroke_2d: None,
            canvas2d: Canvas2D::default(),
            restore_view: None,
            status: "Default sphere and material — File > Open to load a .gltf/.glb".to_string(),
        };

        if let Some(mem) = memory {
            core.panel_visible = mem.panel_visible;
            // New panels added after a saved layout are visible by default.
            while core.panel_visible.len() < Panel::ALL.len() {
                core.panel_visible.push(true);
            }
            core.active_tool = mem.active_tool;
            core.channels = mem.channels;
            core.brush.size = mem.brush_size;
            core.brush.hardness = mem.brush_hardness;
            core.brush.opacity = mem.brush_opacity;
            core.brush.spacing = mem.brush_spacing;
            core.brush.color = mem.brush_color;
            core.show_uv_overlay = mem.show_uv_overlay;
            core.show_uv_checker_3d = mem.show_uv_checker_3d;
            core.show_uv_grid_3d = mem.show_uv_grid_3d;
            // Legacy `BrushShape` code (Round/Square/Diamond/Texture) →
            // footprint kind; `Texture` (3) loads as a sprite brush (whose
            // session sprite, if any, degrades gracefully to round).
            core.brush.kind = crate::brush::FootprintKind::from_shape_code(mem.brush_shape);
            core.brush.rotation = mem.brush_rotation;
            core.brush.flip_x = mem.brush_flip_x;
            core.brush.flip_y = mem.brush_flip_y;
            core.brush.pattern_lock = match mem.brush_pattern_lock {
                1 => crate::brush::PatternLock::Aligned,
                _ => crate::brush::PatternLock::Dab,
            };
            // Old configs default `brush_texture_scale` to 0.0; a 0 multiplier
            // would collapse every repeat, so only adopt a positive stored scale.
            if mem.brush_texture_scale > 0.0 {
                core.brush.texture_scale = mem.brush_texture_scale;
            }
            core.brush.texture_locked = mem.brush_texture_locked;
            core.material = mem.material;
            core.show_tool_strip = mem.show_tool_strip;
            core.tool_strip_anim = if core.show_tool_strip { 1.0 } else { 0.0 };
            core.theme_pref = match mem.theme {
                1 => ThemePref::Dark,
                2 => ThemePref::Light,
                _ => ThemePref::System,
            };
            core.shortcuts = mem.shortcuts;
            // Hidden panels were removed from the dock when they were unchecked;
            // re-apply that so a restored layout doesn't resurrect closed tabs.
            for (i, panel) in Panel::ALL.iter().enumerate() {
                if !core.panel_visible.get(i).copied().unwrap_or(true) && *panel != Panel::Viewport
                {
                    let paths: Vec<_> = dock_state
                        .iter_all_tabs()
                        .filter(|(_, tab)| **tab == *panel)
                        .map(|(path, _)| path)
                        .collect();
                    for path in paths {
                        dock_state.remove_tab(path);
                    }
                }
            }
            if let Some(cam) = mem.camera {
                core.restore_view = Some((
                    glam::Vec3::from(cam.eye),
                    glam::Vec3::from(cam.target),
                    cam.radius,
                    cam.ortho,
                ));
                core.needs_fit = false;
            }
        }

        apply_theme(&cc.egui_ctx, core.theme_pref);

        Self { dock_state, core }
    }

    fn open_model(&mut self, path: &str) {
        match crate::io::load_gltf(path) {
            crate::io::LoadedModel::Mesh(mesh) => {
                let center = mesh_center(&mesh);
                let radius = mesh_bounds_radius(&mesh, center);
                self.core.renderer.set_mesh(&mesh);
                self.core.mesh = Some(mesh);
                self.core.center = center;
                self.core.bounds_radius = radius;
                self.core.needs_fit = true;
                self.core.canvas2d.needs_fit = true;
                self.core.preview_gen += 1;
                self.core.stroke = None;
                self.core.history.clear();
                self.core.status = format!("Loaded {path}");
            }
            crate::io::LoadedModel::Invalid => {
                self.core.status = format!("Failed to load {path}");
            }
        }
    }

    fn export_albedo(&mut self, path: &str) {
        match self.core.mesh.as_ref().and_then(|m| m.flattened_atlas()) {
            Some(tex) => {
                let (tw, th) = (tex.width, tex.height);
                match crate::io::save_atlas_png(path, &tex) {
                    Ok(()) => {
                        self.core.status = format!("Exported albedo atlas ({tw}x{th}) to {path}");
                    }
                    Err(e) => self.core.status = format!("Export failed: {e}"),
                }
            }
            None => self.core.status = "Nothing to export — no layers".to_string(),
        }
    }

    fn export_glb(&mut self, path: &str) {
        match self.core.mesh.as_ref() {
            Some(mesh) => match crate::io::save_glb(path, mesh) {
                Ok(()) => {
                    self.core.status = format!(
                        "Exported .glb with baked layers ({}) to {path}",
                        mesh.layers.len()
                    );
                }
                Err(e) => self.core.status = format!("Export failed: {e}"),
            },
            None => self.core.status = "Nothing to export — no model loaded".to_string(),
        }
    }

    fn save_project(&mut self, path: &str) {
        match self.core.mesh.as_ref() {
            Some(mesh) => match crate::project::save_project(path, mesh) {
                Ok(()) => self.core.status = format!("Saved project to {path}"),
                Err(e) => self.core.status = format!("Save failed: {e}"),
            },
            None => self.core.status = "Nothing to save — no model loaded".to_string(),
        }
    }

    fn open_project(&mut self, path: &str) {
        match crate::project::load_project(path) {
            Ok(mesh) => {
                let center = mesh_center(&mesh);
                let radius = mesh_bounds_radius(&mesh, center);
                self.core.renderer.set_mesh(&mesh);
                self.core.mesh = Some(mesh);
                self.core.center = center;
                self.core.bounds_radius = radius;
                self.core.needs_fit = true;
                self.core.canvas2d.needs_fit = true;
                self.core.preview_gen += 1;
                self.core.stroke = None;
                self.core.history.clear();
                self.core.status = format!("Opened project {path}");
            }
            Err(e) => self.core.status = format!("Failed to open project: {e}"),
        }
    }

    fn import_image_to_layer(&mut self, path: &str) {
        let Some(mesh) = self.core.mesh.as_mut() else {
            self.core.status = "Import failed — no model loaded".to_string();
            return;
        };
        if mesh.layers.is_empty() {
            self.core.status = "Import failed — add a layer first".to_string();
            return;
        }
        if active_layer_locked(mesh) {
            self.core.status = "Import failed — active layer is locked".to_string();
            return;
        }
        let (w, h) = {
            let tex = mesh.active_layer_texture().unwrap();
            (tex.width, tex.height)
        };
        match crate::io::load_image_into_atlas(path, w, h) {
            Ok(img) => {
                self.core.history.record(snapshot_of(mesh));
                let li = mesh.active_layer;
                mesh.layers[li].texture = img;
                self.core.stroke = None;
                self.core.needs_texture_upload = true;
                self.core.needs_material_upload = true;
                self.core.preview_gen += 1;
                self.core.status = format!("Imported image onto layer {}", li + 1);
            }
            Err(e) => self.core.status = format!("Import failed: {e}"),
        }
    }

    fn save_ui_memory(&self) {
        let mem = UiMemory {
            dock: self.dock_state.clone(),
            panel_visible: self.core.panel_visible.clone(),
            active_tool: self.core.active_tool,
            channels: self.core.channels,
            brush_size: self.core.brush.size,
            brush_hardness: self.core.brush.hardness,
            brush_opacity: self.core.brush.opacity,
            brush_spacing: self.core.brush.spacing,
            brush_color: self.core.brush.color,
            show_uv_overlay: self.core.show_uv_overlay,
            show_uv_checker_3d: self.core.show_uv_checker_3d,
            show_uv_grid_3d: self.core.show_uv_grid_3d,
            brush_shape: self.core.brush.kind.shape_code(),
            brush_rotation: self.core.brush.rotation,
            brush_flip_x: self.core.brush.flip_x,
            brush_flip_y: self.core.brush.flip_y,
            brush_pattern_lock: match self.core.brush.pattern_lock {
                crate::brush::PatternLock::Dab => 0,
                crate::brush::PatternLock::Aligned => 1,
            },
            brush_texture_scale: self.core.brush.texture_scale,
            brush_texture_locked: self.core.brush.texture_locked,
            material: self.core.material,
            show_tool_strip: self.core.show_tool_strip,
            theme: match self.core.theme_pref {
                ThemePref::System => 0,
                ThemePref::Dark => 1,
                ThemePref::Light => 2,
            },
            shortcuts: self.core.shortcuts.clone(),
            camera: self.core.viewport.as_ref().map(|vp| CameraState {
                eye: vp.camera.eye.into(),
                target: vp.camera.target.into(),
                radius: vp.camera.radius,
                ortho: vp.camera.ortho,
            }),
        };

        let path = ui_memory_path();
        // MessagePack (not JSON): egui_dock's node rects hold infinite values
        // that JSON cannot represent but rmp-serde round-trips fine.
        let bytes = match rmp_serde::to_vec(&mem) {
            Ok(b) => b,
            Err(e) => {
                log::warn!("failed to serialize UI config: {e}");
                return;
            }
        };
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Err(e) = std::fs::write(&path, bytes) {
            log::warn!("failed to save UI config to {}: {e}", path.display());
        }
    }

    fn set_panel_visible(&mut self, panel: Panel, visible: bool) {
        let idx = panel.index();
        if self.core.panel_visible.len() <= idx {
            self.core.panel_visible.resize(Panel::ALL.len(), true);
        }
        self.core.panel_visible[idx] = visible;

        let paths: Vec<_> = self
            .dock_state
            .iter_all_tabs()
            .filter(|(_, tab)| **tab == panel)
            .map(|(path, _)| path)
            .collect();
        for path in paths {
            self.dock_state.remove_tab(path);
        }
        if visible {
            // Re-open the panel by docking it into the first available leaf.
            // The 3D Viewport is re-shown the same way, so unchecking it in the
            // View menu no longer hides it forever (a docked Viewport is just a
            // leaf holding that tab; `viewport_ui` is independent of whether it
            // shares the leaf with other panels).
            self.dock_state.push_to_first_leaf(panel);
            // If the viewport leaf was emptied, ensure the new tab has company.
            if self.dock_state.iter_all_tabs().count() == 1 {
                self.dock_state.push_to_first_leaf(Panel::Viewport);
            }
        }
    }
}

fn mesh_center(mesh: &MeshData) -> glam::Vec3 {
    if mesh.positions.is_empty() {
        return glam::Vec3::ZERO;
    }
    let mut min = mesh.positions[0];
    let mut max = mesh.positions[0];
    for p in &mesh.positions {
        min = min.min(*p);
        max = max.max(*p);
    }
    (min + max) * 0.5
}

fn mesh_bounds_radius(mesh: &MeshData, center: glam::Vec3) -> f32 {
    mesh.positions
        .iter()
        .map(|p| (p - center).length())
        .fold(0.0, f32::max)
}

/// Projects a world point to viewport pixel coordinates. Returns `None` if the
/// point is behind the camera.
fn project_to_screen(
    cam: &Camera,
    world: glam::Vec3,
    rect: egui::Rect,
    w: u32,
    h: u32,
) -> Option<egui::Pos2> {
    let clip = cam.view_proj().project_point3(world);
    if clip.z > 1.0 || clip.z < 0.0 {
        return None;
    }
    let ndc_x = (clip.x * 0.5 + 0.5) * w as f32;
    let ndc_y = (1.0 - (clip.y * 0.5 + 0.5)) * h as f32;
    Some(egui::pos2(rect.left() + ndc_x, rect.top() + ndc_y))
}

/// Converts a brush radius given in screen pixels into a world-space radius at
/// the depth of `world` (e.g. a mesh hit point).
/// Converts a viewport point (in `rect` coordinates) to normalized device coords.
fn viewport_ndc(x: f32, y: f32, rect: egui::Rect) -> (f32, f32) {
    let nx = ((x - rect.left()) / rect.width().max(1.0)) * 2.0 - 1.0;
    let ny = 1.0 - ((y - rect.top()) / rect.height().max(1.0)) * 2.0;
    (nx, ny)
}

fn screen_to_world_radius(
    cam: &Camera,
    world: glam::Vec3,
    screen_px: f32,
    rect: egui::Rect,
    w: u32,
    h: u32,
) -> f32 {
    let eye_dir = (cam.target - cam.eye).normalize_or_zero();
    // A tangent that is guaranteed perpendicular to the view direction — the
    // camera's right vector. Using `eye_dir.cross(world)` (the hit *position*)
    // was wrong: near the sphere's silhouette the position is no longer
    // perpendicular to the view ray, so the measured pixels-per-world unit was
    // compressed along one axis and the brush flattened in 3D.
    let tangent = {
        let up = if eye_dir.y.abs() > 0.9 {
            glam::Vec3::X
        } else {
            glam::Vec3::Y
        };
        eye_dir.cross(up).normalize_or_zero()
    };
    // Measure how many screen pixels a 1.0 world-unit probe maps to.
    let (Some(center), Some(edge)) = (
        project_to_screen(cam, world, rect, w, h),
        project_to_screen(cam, world + tangent, rect, w, h),
    ) else {
        return screen_px;
    };
    let px_per_world = (edge - center).length().max(1e-6);
    screen_px / px_per_world
}

/// The layout used when no persisted UI config exists (or it fails to load).
fn default_dock() -> DockState<Panel> {
    let mut dock_state = DockState::new(vec![Panel::Viewport]);
    let main = dock_state.main_surface_mut();
    let [_old, left] = main.split_left(NodeIndex::root(), 0.2, vec![Panel::Channels]);
    let [_, lighting] = main.split_below(left, 0.42, vec![Panel::Lighting]);
    let [_old, brushes] = main.split_below(lighting, 0.5, vec![Panel::Brushes]);
    let _ = main.split_below(brushes, 1.0, vec![Panel::Palette]);
    let [_old, right] = main.split_right(NodeIndex::root(), 0.28, vec![Panel::Texture]);
    let [_old, layers] = main.split_below(right, 0.5, vec![Panel::Layers]);
    let _ = main.split_below(layers, 0.5, vec![Panel::Preferences]);
    dock_state
}

fn ui_memory_path() -> std::path::PathBuf {
    if let Some(home) = std::env::var_os("HOME") {
        return std::path::PathBuf::from(home).join(".config/pixforge/ui_layout.msgpack");
    }
    std::path::PathBuf::from("ui_layout.msgpack")
}

/// Reads the persisted UI state, if any. Returns `None` when the file is
/// missing or unreadable so the caller can fall back to the default layout.
fn load_ui_memory() -> Option<(DockState<Panel>, UiMemory)> {
    let path = ui_memory_path();
    let bytes = std::fs::read(&path).ok()?;
    let mem: UiMemory = match rmp_serde::from_slice(&bytes) {
        Ok(m) => m,
        Err(e) => {
            log::warn!("ignoring unreadable UI config {}: {e}", path.display());
            return None;
        }
    };
    // Make sure every panel is present even if the file came from another version.
    let mut dock = mem.dock.clone();
    for panel in Panel::ALL {
        if !dock.iter_all_tabs().any(|(_, tab)| *tab == panel) {
            dock.push_to_first_leaf(panel);
        }
    }
    Some((dock, mem))
}

fn palettes_path() -> std::path::PathBuf {
    if let Some(home) = std::env::var_os("HOME") {
        return std::path::PathBuf::from(home).join(".config/pixforge/palettes.msgpack");
    }
    std::path::PathBuf::from("palettes.msgpack")
}

/// Loads the palette library; falls back to the built-in default palette when
/// the file is missing or unreadable (never fails — palettes are a convenience).
/// Restores every stock palette that is missing or — when `force` — not
/// pristine. This is the "factory default" floor for the built-in set: a stock
/// palette that was deleted or fully cleared comes back automatically on load,
/// and the "Restore stock" button also resets edited stock palettes (forced).
fn restore_stock_palettes(palettes: &mut Vec<crate::palette::Palette>, force: bool) {
    for stock in crate::palette::builtin_palettes() {
        match palettes.iter_mut().find(|p| p.name == stock.name) {
            Some(p) => {
                if force || p.colors.is_empty() {
                    p.colors = stock.colors;
                }
            }
            None => palettes.push(stock),
        }
    }
}

fn load_palettes() -> Vec<crate::palette::Palette> {
    let path = palettes_path();
    let bytes = std::fs::read(&path);
    let bytes = match bytes {
        Ok(b) => b,
        Err(_) => return default_palettes(),
    };
    let mut palettes: Vec<crate::palette::Palette> =
        rmp_serde::from_slice(&bytes).unwrap_or_default();
    if palettes.is_empty() {
        return default_palettes();
    }
    // Auto-restore stock palettes that were deleted or cleared (see
    // restore_stock_palettes). Custom palettes are never touched.
    restore_stock_palettes(&mut palettes, false);
    palettes
}

fn save_palettes(palettes: &[crate::palette::Palette]) {
    let path = palettes_path();
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    match rmp_serde::to_vec_named(palettes) {
        Ok(bytes) => {
            if let Err(e) = std::fs::write(&path, bytes) {
                log::warn!("could not save palettes to {}: {e}", path.display());
            }
        }
        Err(e) => log::warn!("could not serialize palettes: {e}"),
    }
}

fn default_palettes() -> Vec<crate::palette::Palette> {
    crate::palette::builtin_palettes()
}

impl eframe::App for PixForgeApp {
    fn clear_color(&self, visuals: &egui::Visuals) -> [f32; 4] {
        if visuals.dark_mode {
            [0.055, 0.06, 0.07, 1.0]
        } else {
            [0.90, 0.91, 0.93, 1.0]
        }
    }

    fn on_exit(&mut self) {
        self.save_ui_memory();
        save_palettes(&self.core.palettes);
    }

    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        // The floating brush menu may be hosted by whichever panel is first to
        // draw it (3D viewport or 2D canvas); re-allow it this frame.
        self.core.brush_menu_rendered = false;

        // Capture a pending shortcut binding before the app's own key handlers
        // run, so a key meant for recording never triggers an action.
        self.capture_binding(ui);
        self.handle_shortcuts(ui);
        self.menu_bar(ui);

        // Persistent header toolbar (color + brush sliders) pinned under the
        // menu bar, always visible regardless of dock layout.
        egui::Panel::top("main_toolbar").show(ui, |ui| toolbar_ui(ui, &mut self.core));

        egui::CentralPanel::default().show(ui, |ui| {
            let (dock_state, core) = (&mut self.dock_state, &mut self.core);
            DockArea::new(dock_state).show_inside(ui, &mut PixForgeTabViewer { core });
        });
    }
}

impl PixForgeApp {
    fn menu_bar(&mut self, ui: &mut Ui) {
        egui::Panel::top("menu_bar")
            .exact_size(ui.text_style_height(&egui::TextStyle::Button) + 10.0)
            .show(ui, |ui| {
                egui::MenuBar::new().ui(ui, |ui| {
                    ui.menu_button("File", |ui| {
                        if ui.button("Open Model…").clicked() {
                            ui.close();
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("3D models", &["gltf", "glb"])
                                .pick_file()
                            {
                                self.open_model(&path.to_string_lossy());
                            }
                        }
                        ui.separator();
                        ui.add_enabled(
                            self.core.mesh.is_some(),
                            egui::Button::new("Save Project…"),
                        )
                        .on_hover_text(
                            "Save the model geometry and all layers to a .pixforge project file",
                        )
                        .clicked()
                        .then(|| {
                            ui.close();
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("PixForge project", &["pixforge"])
                                .set_file_name("untitled.pixforge")
                                .save_file()
                            {
                                self.save_project(&path.to_string_lossy());
                            }
                        });
                        if ui
                            .button("Open Project…")
                            .on_hover_text("Open a .pixforge project with layers intact")
                            .clicked()
                        {
                            ui.close();
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("PixForge project", &["pixforge"])
                                .pick_file()
                            {
                                self.open_project(&path.to_string_lossy());
                            }
                        }
                        ui.separator();

if ui.button("Open Environment / Skybox…").clicked() {
                            ui.close();
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter(
                                    "Environment maps",
                                    &["hdr", "png", "jpg", "jpeg", "bmp", "webp"],
                                )
                                .pick_file()
                            {
                                match crate::io::load_environment(&path.to_string_lossy()) {
                                    Ok(env) => {
                                        self.core.renderer.set_environment(Some(env));
                                        self.core.env_path =
                                            Some(path.to_string_lossy().to_string());
                                        self.core.status = format!(
                                            "Loaded environment from {}",
                                            path.display()
                                        );
                                    }
                                    Err(e) => {
                                        self.core.status = format!("Environment load failed: {e}")
                                    }
                                }
                            }
                        }
                        ui.add_enabled(
                            self.core.env_path.is_some(),
                            egui::Button::new("Clear Skybox"),
                        )
                        .on_hover_text("Fall back to the analytic sky")
                        .clicked()
                        .then(|| {
                            ui.close();
                            self.core.renderer.set_environment(None);
                            self.core.env_path = None;
                            self.core.status =
                                "Cleared environment — analytic sky".to_string();
                        });

                        ui.separator();
                        ui.add_enabled(
                            self.core
                                .mesh
                                .as_ref()
                                .is_some_and(|m| !m.layers.is_empty()),
                            egui::Button::new("Export Albedo Atlas…"),
                        )
                        .on_hover_text("Save the painted albedo atlas as a PNG")
                        .clicked()
                        .then(|| {
                            ui.close();
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("PNG image", &["png"])
                                .set_file_name("pixforge_albedo.png")
                                .save_file()
                            {
                                self.export_albedo(&path.to_string_lossy());
                            }
                        });
                        if ui
                            .add_enabled(
                                self.core.mesh.is_some(),
                                egui::Button::new("Export GLB…"),
                            )
                            .on_hover_text(
                                "Bake all layers (albedo, material, height/bump) and save as a .glb",
                            )
                            .clicked()
                        {
                            ui.close();
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("glTF binary", &["glb"])
                                .set_file_name("pixforge_model.glb")
                                .save_file()
                            {
                                self.export_glb(&path.to_string_lossy());
                            }
                        }
                        ui.separator();
                        ui.add_enabled(
                            self.core
                                .mesh
                                .as_ref()
                                .is_some_and(|m| !m.layers.is_empty()),
                            egui::Button::new("Import Image to Layer…"),
                        )
                        .on_hover_text("Fit a PNG/image onto the active layer")
                        .clicked()
                        .then(|| {
                            ui.close();
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("Images", &["png", "jpg", "jpeg", "bmp", "webp"])
                                .pick_file()
                            {
                                self.import_image_to_layer(&path.to_string_lossy());
                            }
                        });
                        ui.separator();
                        if ui.button("Quit").clicked() {
                            ui.close();
                            ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    });

                    ui.menu_button("Edit", |ui| {
                        let can_undo = self.core.history.can_undo();
                        let can_redo = self.core.history.can_redo();
                        if ui
                            .add_enabled(
                                can_undo,
                                egui::Button::new("Undo").shortcut_text("Ctrl+Z"),
                            )
                            .clicked()
                        {
                            ui.close();
                            self.undo();
                        }
                        if ui
                            .add_enabled(
                                can_redo,
                                egui::Button::new("Redo").shortcut_text("Ctrl+Shift+Z"),
                            )
                            .clicked()
                        {
                            ui.close();
                            self.redo();
                        }
                        ui.separator();
                        if ui.button("Preferences…").clicked() {
                            ui.close();
                            let present = self
                                .dock_state
                                .iter_all_tabs()
                                .any(|(_, tab)| *tab == Panel::Preferences)
                                && self
                                    .core
                                    .panel_visible
                                    .get(Panel::Preferences.index())
                                    .copied()
                                    .unwrap_or(true);
                            if !present {
                                self.set_panel_visible(Panel::Preferences, true);
                            }
                        }
                    });

                    ui.menu_button("View", |ui| {
                        for panel in Panel::ALL {
                            let idx = panel.index();
                            let mut visible = self
                                .core
                                .panel_visible
                                .get(idx)
                                .copied()
                                .unwrap_or(true);
                            if ui.checkbox(&mut visible, panel.title()).changed() {
                                self.set_panel_visible(panel, visible);
                            }
                        }
                        ui.separator();
                        ui.checkbox(&mut self.core.show_tool_strip, "In-viewport tools (T)");
                    });

                    ui.menu_button("Help", |ui| {
                        ui.label("PixForge — stylized 3D texture painter");
                    });
                });
            });
    }

    /// App-level keyboard shortcuts (Ctrl+Z / Ctrl+Shift+Z / Ctrl+Y). Ignored
    /// App-level keyboard shortcuts (tools, brush size/opacity, file ops,
    /// undo/redo). Ignored while an egui text widget has keyboard focus
    /// (e.g. typing in a field) or a shortcut capture is pending.
    fn handle_shortcuts(&mut self, ui: &mut Ui) {
        if self.core.recording.is_some() {
            return;
        }
        if ui.ctx().egui_wants_keyboard_input() {
            return;
        }
        // Order matters: modifiers match logically (pattern ⊆ pressed), so the
        // Shift-superset variants must be checked before their bare counterparts
        // (Redo before Undo, Open Project before Open Model).
        let mut do_undo = false;
        let mut do_redo = false;
        let mut open_model = false;
        let mut open_project = false;
        let mut save_project = false;
        let mut open_env = false;
        let mut pick_tool: Option<usize> = None;
        let mut brush_delta = 0.0f32;
        let mut opacity_delta = 0.0f32;
        ui.ctx().input_mut(|i| {
            let redo = *self.core.shortcuts.get(ShortcutAction::Redo);
            if redo.is_bound() {
                do_redo = i.consume_key(redo.modifiers_of(), redo.key_of());
            }
            let undo = *self.core.shortcuts.get(ShortcutAction::Undo);
            if undo.is_bound() && !do_redo {
                do_undo = i.consume_key(undo.modifiers_of(), undo.key_of());
            }

            let project = *self.core.shortcuts.get(ShortcutAction::OpenProject);
            if project.is_bound() {
                open_project = i.consume_key(project.modifiers_of(), project.key_of());
            }
            let model = *self.core.shortcuts.get(ShortcutAction::OpenModel);
            if model.is_bound() && !open_project {
                open_model = i.consume_key(model.modifiers_of(), model.key_of());
            }
            let save = *self.core.shortcuts.get(ShortcutAction::SaveProject);
            if save.is_bound() {
                save_project = i.consume_key(save.modifiers_of(), save.key_of());
            }
            let env = *self.core.shortcuts.get(ShortcutAction::OpenEnvironment);
            if env.is_bound() {
                open_env = i.consume_key(env.modifiers_of(), env.key_of());
            }

            for action in ShortcutAction::ALL {
                if let Some(index) = action.tool_index() {
                    let bind = *self.core.shortcuts.get(action);
                    if bind.is_bound() && i.consume_key(bind.modifiers_of(), bind.key_of()) {
                        pick_tool = Some(index);
                    }
                }
            }
            let size_up = *self.core.shortcuts.get(ShortcutAction::BrushSizeUp);
            if size_up.is_bound() && i.consume_key(size_up.modifiers_of(), size_up.key_of()) {
                brush_delta = self.core.brush.size * 0.1;
            }
            let size_down = *self.core.shortcuts.get(ShortcutAction::BrushSizeDown);
            if size_down.is_bound() && i.consume_key(size_down.modifiers_of(), size_down.key_of()) {
                brush_delta = -self.core.brush.size * 0.1;
            }
            let op_up = *self.core.shortcuts.get(ShortcutAction::BrushOpacityUp);
            if op_up.is_bound() && i.consume_key(op_up.modifiers_of(), op_up.key_of()) {
                opacity_delta = 0.05;
            }
            let op_down = *self.core.shortcuts.get(ShortcutAction::BrushOpacityDown);
            if op_down.is_bound() && i.consume_key(op_down.modifiers_of(), op_down.key_of()) {
                opacity_delta = -0.05;
            }
        });

        if let Some(index) = pick_tool {
            self.core.active_tool = index;
            self.core.status = format!("Tool: {}", TOOLS[index]);
        }
        if brush_delta != 0.0 {
            self.core.brush.size = (self.core.brush.size + brush_delta).clamp(1.0, 300.0);
            self.core.status = format!("Brush size: {:.0}px", self.core.brush.size);
        }
        if opacity_delta != 0.0 {
            self.core.brush.opacity = (self.core.brush.opacity + opacity_delta).clamp(0.0, 1.0);
            self.core.status = format!("Brush opacity: {:.0}%", self.core.brush.opacity * 100.0);
        }
        if open_model {
            self.prompt_open_model();
        }
        if open_project {
            self.prompt_open_project();
        }
        if save_project {
            self.prompt_save_project();
        }
        if open_env {
            self.prompt_open_environment();
        }
        if do_undo {
            self.undo();
        }
        if do_redo {
            self.redo();
        }
    }

    /// Open a model via a native file dialog (used by File menu + shortcut).
    fn prompt_open_model(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("3D models", &["gltf", "glb"])
            .pick_file()
        {
            self.open_model(&path.to_string_lossy());
        }
    }

    /// Open a saved project via a native file dialog.
    fn prompt_open_project(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("PixForge project", &["pixforge"])
            .pick_file()
        {
            self.open_project(&path.to_string_lossy());
        }
    }

    /// Save the project (Save-As semantics via a native dialog).
    fn prompt_save_project(&mut self) {
        if self.core.mesh.is_none() {
            self.core.status = "Nothing to save — no model loaded".to_string();
            return;
        }
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("PixForge project", &["pixforge"])
            .set_file_name("untitled.pixforge")
            .save_file()
        {
            self.save_project(&path.to_string_lossy());
        }
    }

    /// Load an environment map via a native file dialog.
    fn prompt_open_environment(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter(
                "Environment maps",
                &["hdr", "png", "jpg", "jpeg", "bmp", "webp"],
            )
            .pick_file()
        {
            match crate::io::load_environment(&path.to_string_lossy()) {
                Ok(env) => {
                    self.core.renderer.set_environment(Some(env));
                    self.core.env_path = Some(path.to_string_lossy().to_string());
                    self.core.status = format!("Loaded environment from {}", path.display());
                }
                Err(e) => self.core.status = format!("Environment load failed: {e}"),
            }
        }
    }

    fn undo(&mut self) {
        undo_action(&mut self.core);
    }

    fn redo(&mut self) {
        redo_action(&mut self.core);
    }
}

fn undo_action(core: &mut Core) {
    // Most recent action was a palette edit -> undo the palette library first,
    // so a cleared / deleted palette restores instead of being lost forever.
    if core.palette_edit_pending && !core.palette_prev.is_empty() {
        if let Some(prev) = core.palette_prev.pop() {
            core.palette_redo.push(core.palettes.clone());
            core.palettes = prev;
            if core.active_palette >= core.palettes.len() {
                core.active_palette = core.palettes.len().saturating_sub(1);
            }
            save_palettes(&core.palettes);
            core.palette_confirm = None;
            let left = core.palette_prev.len();
            core.status = if left > 0 {
                "Undo palette".to_string()
            } else {
                "Undo palette — no more history".to_string()
            };
            return;
        }
    }
    core.palette_edit_pending = false;
    let current = snapshot_of_current(core);
    if let Some(snap) = core.history.undo(current) {
        restore_snapshot(core, snap);
        let left = core.history.can_undo();
        core.status = if left {
            "Undo".to_string()
        } else {
            "Undo — history empty".to_string()
        };
    }
}

fn redo_action(core: &mut Core) {
    // Mirror the palette-undo routing for Ctrl+Y / Ctrl+Shift+Z.
    if core.palette_edit_pending && !core.palette_redo.is_empty() {
        if let Some(next) = core.palette_redo.pop() {
            core.palette_prev.push(core.palettes.clone());
            core.palettes = next;
            if core.active_palette >= core.palettes.len() {
                core.active_palette = core.palettes.len().saturating_sub(1);
            }
            save_palettes(&core.palettes);
            core.palette_confirm = None;
            let left = core.palette_redo.len();
            core.status = if left > 0 {
                "Redo palette".to_string()
            } else {
                "Redo palette — nothing to redo".to_string()
            };
            return;
        }
    }
    core.palette_edit_pending = false;
    let current = snapshot_of_current(core);
    if let Some(snap) = core.history.redo(current) {
        restore_snapshot(core, snap);
        let left = core.history.can_redo();
        core.status = if left {
            "Redo".to_string()
        } else {
            "Redo — nothing to redo".to_string()
        };
    }
}

impl PixForgeApp {
    /// Capture the next key press as a shortcut binding (Preferences window).
    /// Esc cancels; a raw modifier key alone never binds.
    fn capture_binding(&mut self, ui: &mut Ui) {
        let Some(action) = self.core.recording else {
            return;
        };
        // Modifier keys arrive as their own *first* `Event::Key` press when the
        // user types a combination (Ctrl, then Z). Ignore them — a binding with
        // only a modifier key is useless, and without this check "Ctrl+Z" would
        // bind a lone "ControlLeft". The modifier is still captured implicitly
        // in the modifiers state on the non-modifier key's event.
        let is_modifier = |k: egui::Key| {
            matches!(
                k,
                egui::Key::ShiftLeft
                    | egui::Key::ShiftRight
                    | egui::Key::ControlLeft
                    | egui::Key::ControlRight
                    | egui::Key::AltLeft
                    | egui::Key::AltRight
                    | egui::Key::SuperLeft
                    | egui::Key::SuperRight
            )
        };
        let Some((key, modifiers)) = ui.ctx().input(|i| {
            i.events.iter().find_map(|e| match e {
                egui::Event::Key {
                    key,
                    pressed: true,
                    modifiers,
                    ..
                } if !is_modifier(*key) => Some((*key, *modifiers)),
                _ => None,
            })
        }) else {
            return;
        };

        // Consume the event so the same key press can't also fire the action it
        // was just bound to (or a panel-local handler) later this frame.
        ui.ctx().input_mut(|i| i.consume_key(modifiers, key));

        if key == egui::Key::Escape {
            self.core.recording = None;
            self.core.status = "Shortcut capture cancelled".to_string();
            return;
        }

        let bind = KeyBind::new(key, modifiers);
        *self.core.shortcuts.get_mut(action) = bind;
        self.core.recording = None;
        self.core.status = format!("{} bound to {}", action.label(), bind.label());
    }
}

/// Bold page title at the top of each Preferences sub-page.
fn page_title(ui: &mut Ui, text: &str) {
    let pal = UiPalette::of(ui);
    ui.label(
        egui::RichText::new(text)
            .size(13.0)
            .strong()
            .color(pal.chrome_text),
    );
}

/// Renders the Preferences panel as a two-column, tabbed layout: a vertical
/// category list on the left, the selected category's dedicated settings page
/// on the right. The Shortcuts page starts a capture via `Core::recording`,
/// which `capture_binding` resolves each frame before the app's own key
/// handlers run.
fn prefs_ui(ui: &mut Ui, core: &mut Core) {
    ui.add_space(6.0);
    panel_heading(ui, "Preferences");
    ui.add_space(6.0);

    // --- Left: the vertical category list ---
    egui::Panel::left("prefs_category")
        .exact_size(104.0)
        .resizable(false)
        .show_separator_line(true)
        .show(ui, |ui| {
            let pal = UiPalette::of(ui);
            ui.spacing_mut().item_spacing = egui::vec2(0.0, 4.0);
            ui.add_space(2.0);
            let tabs = [
                (PrefsTab::Appearance, "Appearance"),
                (PrefsTab::Viewport, "Viewport"),
                (PrefsTab::Shortcuts, "Shortcuts"),
            ];
            for (tab, label) in tabs {
                let active = core.prefs_tab == tab;
                let row = egui::Frame::new()
                    .fill(if active { ACCENT } else { pal.card })
                    .stroke(egui::Stroke::new(
                        1.0,
                        if active { ACCENT_HOVER } else { pal.card_border },
                    ))
                    .corner_radius(egui::CornerRadius::same(RADIUS_CHIP))
                    .inner_margin(egui::Margin::symmetric(8, 6))
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.colored_label(
                            if active {
                                egui::Color32::WHITE
                            } else {
                                pal.chrome_text
                            },
                            egui::RichText::new(label).strong(),
                        );
                    })
                    .response
                    .interact(egui::Sense::click());
                row.clone().on_hover_cursor(egui::CursorIcon::PointingHand);
                if row.clicked() && !active {
                    core.prefs_tab = tab;
                    core.status = format!("Preferences: {label}");
                }
            }
        });

    // --- Right: the settings page for the selected category ---
    egui::CentralPanel::default().show(ui, |ui| {
        ui.add_space(6.0);
        egui::ScrollArea::vertical()
            .id_salt("prefs_page")
            .auto_shrink([false, true])
            .show(ui, |ui| match core.prefs_tab {
                PrefsTab::Appearance => appearance_prefs_ui(ui, core),
                PrefsTab::Viewport => viewport_prefs_ui(ui, core),
                PrefsTab::Shortcuts => shortcuts_prefs_ui(ui, core),
            });
    });
}

/// Preferences page "Appearance": the UI chrome theme as a pill toggle group —
/// the active choice gets the accent fill and reads as one unit.
fn appearance_prefs_ui(ui: &mut Ui, core: &mut Core) {
    let pal = UiPalette::of(ui);
    page_title(ui, "Theme");
    ui.label(
        egui::RichText::new("Switches the UI chrome between the two built-in themes.")
            .small()
            .color(pal.chrome_text_weak),
    );
    ui.add_space(6.0);
    let themes = [
        (ThemePref::System, "System"),
        (ThemePref::Dark, "Dark"),
        (ThemePref::Light, "Light"),
    ];
    ui.horizontal(|ui| {
        for (pref, label) in themes {
            let active = core.theme_pref == pref;
            let fill = if active { ACCENT } else { pal.control };
            let text_col = if active {
                egui::Color32::WHITE
            } else {
                pal.control_text
            };
            let frame = egui::Frame::new()
                .fill(fill)
                .corner_radius(egui::CornerRadius::same(RADIUS_PILL))
                .inner_margin(egui::Margin::symmetric(14, 5));
            let resp = frame
                .show(ui, |ui| ui.colored_label(text_col, label))
                .response
                .interact(egui::Sense::click());
            resp.clone().on_hover_cursor(egui::CursorIcon::PointingHand);
            if resp.clicked() {
                core.theme_pref = pref;
                apply_theme(ui.ctx(), pref);
                core.status = format!("Theme: {:?}", pref);
            }
        }
    });
}

/// Preferences page "Viewport": display toggles for the 3D viewport and the
/// 2D texture canvas (all persisted).
fn viewport_prefs_ui(ui: &mut Ui, core: &mut Core) {
    let pal = UiPalette::of(ui);
    page_title(ui, "Viewport");
    ui.label(
        egui::RichText::new("What the 3D viewport and the 2D texture canvas display.")
            .small()
            .color(pal.chrome_text_weak),
    );
    ui.add_space(6.0);

    let mut tool_strip = core.show_tool_strip;
    if ui
        .checkbox(&mut tool_strip, "In-viewport tool strip (T)")
        .on_hover_text(
            "Blender-style vertical tool bar (Brush / Eraser / Fill / Pick / Rect) \
             floating over the 3D viewport's left edge. Also toggled live with T.",
        )
        .changed()
    {
        core.show_tool_strip = tool_strip;
        core.tool_strip_anim = if tool_strip { 1.0 } else { 0.0 };
        core.status = if tool_strip {
            "In-viewport tools: on".to_string()
        } else {
            "In-viewport tools: off".to_string()
        };
    }

    let mut uv_overlay = core.show_uv_overlay;
    if ui
        .checkbox(&mut uv_overlay, "UV wireframe (2D canvas)")
        .on_hover_text("Show the UV island wireframe over the 2D texture canvas.")
        .changed()
    {
        core.show_uv_overlay = uv_overlay;
    }

    let mut checker = core.show_uv_checker_3d;
    if ui
        .checkbox(&mut checker, "UV checkerboard (3D)")
        .on_hover_text(
            "Overlay a UV checkerboard on the 3D model to check texel density \
             and seam stretching.",
        )
        .changed()
    {
        core.show_uv_checker_3d = checker;
    }

    let mut grid = core.show_uv_grid_3d;
    if ui
        .checkbox(&mut grid, "UV grid (3D)")
        .on_hover_text("Overlay a UV grid on the 3D model.")
        .changed()
    {
        core.show_uv_grid_3d = grid;
    }
}

/// Preferences page "Shortcuts": remappable keyboard bindings with a reset.
/// Click a binding to start a capture; `capture_binding` resolves it.
fn shortcuts_prefs_ui(ui: &mut Ui, core: &mut Core) {
    let pal = UiPalette::of(ui);
    page_title(ui, "Shortcuts");
    ui.label("Click a binding, then press a key. Esc cancels.");
    ui.add_space(4.0);
    let mut last_category: Option<&'static str> = None;
    let mut zebra = false;
    for action in ShortcutAction::ALL {
        let category = action.category();
        if Some(category) != last_category {
            last_category = Some(category);
            zebra = false;
            ruled_heading(ui, category);
        }
        if core.recording == Some(action) {
            ui.horizontal(|ui| {
                ui.label(action.label());
                ui.colored_label(ui.visuals().warn_fg_color, "Listening… press a key");
            });
            continue;
        }
        let bind = *core.shortcuts.get(action);
        let frame_fill = if zebra { pal.card_active } else { pal.card };
        zebra = !zebra;
        let frame = egui::Frame::new()
            .fill(frame_fill)
            .corner_radius(egui::CornerRadius::same(RADIUS_CHIP))
            .inner_margin(egui::Margin::symmetric(8, 4));
        let row = frame
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(action.label()).on_hover_text(action.description());
                    let btn = ui
                        .add(
                            egui::Button::new(bind.label()).min_size(egui::vec2(70.0, 0.0)),
                        )
                        .on_hover_text(action.description());
                    if btn.clicked() {
                        core.recording = Some(action);
                    }
                    if bind.is_bound()
                        && ui
                            .small_button("×")
                            .on_hover_text("Remove this binding")
                            .clicked()
                    {
                        *core.shortcuts.get_mut(action) = KeyBind::unbound();
                    }
                });
            })
            .response;
        row.on_hover_cursor(egui::CursorIcon::PointingHand);
    }
    ui.add_space(8.0);
    // Reset shortcuts with an armed two-step confirm (mirrors the
    // palette panel's Clear/Delete flow) instead of a bare button.
    let armed = core
        .prefs_reset_armed
        .is_some_and(|t| t.elapsed().as_secs_f32() < 3.0);
    if !armed {
        core.prefs_reset_armed = None;
    }
    let label = if armed {
        "⚠ Confirm reset?"
    } else {
        "Reset shortcuts…"
    };
    if ui
        .button(egui::RichText::new(label).color(if armed {
            pal.warn
        } else {
            pal.control_text
        }))
        .clicked()
    {
        if armed {
            core.shortcuts = Shortcuts::default();
            core.prefs_reset_armed = None;
            core.status = "Shortcuts reset to defaults".to_string();
        } else {
            core.prefs_reset_armed = Some(std::time::Instant::now());
        }
    }
}

fn snapshot_of_current(core: &Core) -> LayerStackSnapshot {
    core.mesh
        .as_ref()
        .map(snapshot_of)
        .unwrap_or_else(|| LayerStackSnapshot {
            layers: Vec::new(),
            active_layer: 0,
        })
}

/// True when the mesh has an active layer that is protected from edits. A mesh
/// with no layers is not "locked" — there is simply nothing to edit.
fn active_layer_locked(mesh: &MeshData) -> bool {
    mesh.layers.get(mesh.active_layer).is_some_and(|l| l.locked)
}

/// Drag-and-drop payload for reordering the layer stack: the storage index of
/// the layer being dragged.
#[derive(Clone, Copy)]
struct LayerDrag {
    from: usize,
}

/// Moves the layer at `from` relative to `target` (either above or below it in UI order),
/// returning the new active layer index.
fn move_layer_relative(
    layers: &mut Vec<crate::io::Layer>,
    from: usize,
    target: usize,
    above_in_ui: bool,
    active: usize,
) -> usize {
    if layers.len() <= 1 || from >= layers.len() || target >= layers.len() || from == target {
        return active;
    }
    let moved = layers.remove(from);
    let target_idx = if from < target { target - 1 } else { target };
    // In UI, layers are displayed in reverse (top of list = highest storage index).
    // So "above_in_ui" means higher storage index (+1).
    let insert_at = if above_in_ui {
        target_idx + 1
    } else {
        target_idx
    };
    let insert_at = insert_at.min(layers.len());
    layers.insert(insert_at, moved);

    let new_active = if active == from {
        insert_at
    } else {
        let a = if active > from { active - 1 } else { active };
        if a >= insert_at {
            a + 1
        } else {
            a
        }
    };
    new_active.min(layers.len().saturating_sub(1))
}

/// Moves the layer at `from` so it lands at storage index `to` (dropped onto
/// the row displayed there), returning the corrected active-layer index.
fn reorder_layers(
    layers: &mut Vec<crate::io::Layer>,
    from: usize,
    to: usize,
    active: usize,
) -> usize {
    debug_assert!(from < layers.len());
    let to = to.min(layers.len().saturating_sub(1));
    let moved = layers.remove(from);
    layers.insert(to, moved);
    let new_active = if active == from {
        to
    } else {
        // Compose the two index shifts (remove shifts down, insert shifts up).
        let x = if active > from { active - 1 } else { active };
        if x >= to {
            x + 1
        } else {
            x
        }
    };
    new_active.min(layers.len().saturating_sub(1))
}

/// Restores a snapshot as the mesh's layer stack, scheduling a GPU re-upload
/// and preview rebuild. Handles dimension changes (update_texture recreates
/// the texture when the size differs).
fn restore_snapshot(core: &mut Core, snap: LayerStackSnapshot) {
    if let Some(mesh) = core.mesh.as_mut() {
        mesh.layers = snap
            .layers
            .into_iter()
            .map(|l| crate::io::Layer {
                name: l.name,
                visible: l.visible,
                locked: l.locked,
                opacity: l.opacity,
                blend: l.blend,
                roughness: l.roughness,
                metallic: l.metallic,
                emissive: l.emissive,
                ambient_occlusion: l.ambient_occlusion,
                height: l.height,
                bump_strength: l.bump_strength,
                clearcoat: l.clearcoat,
                clearcoat_roughness: l.clearcoat_roughness,
                specular_ior: l.specular_ior,
                emissive_color: l.emissive_color,
                texture: l.texture,
            })
            .collect();
        mesh.active_layer = snap.active_layer.min(mesh.layers.len().saturating_sub(1));
    }
    core.stroke = None;
    core.stroke_2d = None;
    core.needs_texture_upload = true;
    core.needs_material_upload = true;
    core.preview_gen += 1;
}

/// Common bookkeeping after a texture swap (resize / blank): abort any active
/// stroke, schedule a GPU upload and preview rebuild, update the status line.
fn finish_texture_change(core: &mut Core, status: String) {
    core.stroke = None;
    core.stroke_2d = None;
    core.canvas2d.needs_fit = true;
    core.needs_texture_upload = true;
    core.needs_material_upload = true;
    core.preview_gen += 1;
    core.status = status;
}

/// Pushes a CPU-side paint edit (3D stamp or 2D preview stamp) to the GPU:
/// recomposite + upload just the dirty texel rect when one was recorded,
/// otherwise fall back to a full atlas upload. Bumps the preview generation and
/// schedules the material-map recomposite + upload. Shared by the 3D viewport
/// and the 2D texture preview so both paint paths end with the same pipeline.
fn flush_paint_edit(core: &mut Core) {
    let mut uploaded_region = false;
    if let Some((x0, y0, x1, y1)) = core.mesh.as_ref().and_then(|m| m.dirty) {
        let (fw, fh, rx, ry, rw, rh) = {
            let mesh = core.mesh.as_ref().unwrap();
            let (fw, fh) = mesh
                .layers
                .first()
                .map(|l| (l.texture.width, l.texture.height))
                .unwrap_or((0, 0));
            (
                fw,
                fh,
                x0,
                y0,
                x1.saturating_sub(x0) + 1,
                y1.saturating_sub(y0) + 1,
            )
        };
        let region = core
            .mesh
            .as_ref()
            .and_then(|m| m.flattened_atlas_region(rx, ry, rw, rh));
        if let Some(region) = region {
            uploaded_region = core.renderer.update_texture_region(&region, rx, ry, fw, fh);
        }
        if uploaded_region {
            core.preview_patch = Some((rx, ry, rw, rh, fw, fh));
            if let Some(m) = core.mesh.as_mut() {
                m.dirty = None;
            }
        } else {
            core.preview_patch = None;
            core.needs_texture_upload = true;
        }
    } else {
        core.preview_patch = None;
        core.needs_texture_upload = true;
    }
    core.preview_gen += 1;
    core.needs_material_upload = true;
}

struct PixForgeTabViewer<'a> {
    core: &'a mut Core,
}

impl TabViewer for PixForgeTabViewer<'_> {
    type Tab = Panel;

    fn id(&mut self, tab: &mut Panel) -> Id {
        Id::new(tab.title())
    }

    fn title(&mut self, tab: &mut Panel) -> WidgetText {
        tab.title().into()
    }

    fn ui(&mut self, ui: &mut Ui, tab: &mut Panel) {
        let core: &mut Core = self.core;
        match tab {
            // Viewport and Texture manage their own full-size / pan-zoom
            // canvases, and Brushes/Prefs scroll internally already.
            Panel::Viewport => viewport_ui(ui, core),
            Panel::Texture => texture_ui(ui, core),
            Panel::Brushes => brushes_ui(ui, core),
            Panel::Preferences => prefs_ui(ui, core),
            // Everything else gets a uniform vertical scroll so long panels
            // never clip — a single place to keep that behavior consistent.
            panel => {
                egui::ScrollArea::vertical()
                    .id_salt(panel.title())
                    .auto_shrink([false, false])
                    .show(ui, |ui| match panel {
                        Panel::Channels => channels_ui(ui, core),
                        Panel::Layers => layers_ui(ui, core),
                        Panel::Lighting => lighting_ui(ui, core),
                        Panel::Palette => palette_ui(ui, core),
                        _ => {}
                    });
            }
        }
    }
}

fn viewport_ui(ui: &mut Ui, core: &mut Core) {
    let full_rect = ui.max_rect();

    // Remappable per-panel shortcuts: T toggles the 3D T-bar, and (when bound)
    // the UV overlay bar pin. Both consume only while the pointer is over the
    // viewport, so the texture editor keeps its own T for its picker.
    let bind_tools = *core.shortcuts.get(ShortcutAction::ToggleTools3d);
    let pressed_tools = bind_tools.is_bound()
        && core.recording.is_none()
        && ui
            .ctx()
            .input_mut(|i| i.consume_key(bind_tools.modifiers_of(), bind_tools.key_of()));
    if ui.rect_contains_pointer(full_rect) && pressed_tools {
        core.show_tool_strip = !core.show_tool_strip;
        core.status = if core.show_tool_strip {
            "In-viewport tools: on (toggle: Preferences)".to_string()
        } else {
            "In-viewport tools: off (toggle: Preferences)".to_string()
        };
    }
    let bind_overlay = *core.shortcuts.get(ShortcutAction::ToggleOverlayBar);
    let pressed_overlay = bind_overlay.is_bound()
        && core.recording.is_none()
        && ui
            .ctx()
            .input_mut(|i| i.consume_key(bind_overlay.modifiers_of(), bind_overlay.key_of()));
    if ui.rect_contains_pointer(full_rect) && pressed_overlay {
        core.show_vp_overlay_bar = !core.show_vp_overlay_bar;
        core.status = if core.show_vp_overlay_bar {
            "UV overlay bar: pinned".to_string()
        } else {
            "UV overlay bar: hidden".to_string()
        };
    }

    // Blender-style T-bar: a slim vertical tool strip floating over the
    // viewport's left edge (translucent, rounded, toggled with T). It slides
    // in/out horizontally. The 3D scene renders under the *whole* viewport
    // (including beneath the strip), so the strip reads as translucency over
    // the image. Its backdrop is sized to the buttons and only that box blocks
    // pointer interaction.
    let target = if core.show_tool_strip { 1.0 } else { 0.0 };
    if (core.tool_strip_anim - target).abs() > 1e-3 {
        let dt = ui.input(|i| i.stable_dt).clamp(0.0, 0.1);
        // Linear-in-time ramp to the target. A constant-speed slide (rather
        // than an exponential settle) guarantees the bar always clears the
        // viewport edge — it never stalls half-visible.
        let dir = if target > core.tool_strip_anim {
            1.0
        } else {
            -1.0
        };
        core.tool_strip_anim = (core.tool_strip_anim + dir * dt / STRIP_ANIM_S).clamp(0.0, 1.0);
        ui.ctx().request_repaint();
    }
    let anim = core.tool_strip_anim;
    let strip_rect = tool_strip_rect(full_rect.min, anim);

    // Top-right corner: the navigation gizmo (six orthographic axis views + a
    // perspective home hub, left of the overlay pin which is now smaller).
    // Pointer gating is precise — only a small radius around each dot/hub
    // blocks painting, so the rest of the corner stays paintable.
    let pin_rect = egui::Rect::from_min_size(
        egui::pos2(full_rect.right() - 30.0, full_rect.top() + 10.0),
        egui::vec2(20.0, 20.0),
    );
    let gizmo_size = 84.0;
    let gizmo_rect = egui::Rect::from_min_size(
        egui::pos2(pin_rect.left() - 8.0 - gizmo_size, full_rect.top() + 6.0),
        egui::vec2(gizmo_size, gizmo_size),
    );
    let vp_bar_anchor = egui::Rect::from_min_max(
        egui::pos2(full_rect.right() - 420.0, full_rect.top() + 8.0),
        egui::pos2(gizmo_rect.left() - 8.0, full_rect.top() + 48.0),
    );
    let vp_bar_active = core.show_vp_overlay_bar
        && ui
            .input(|i| i.pointer.hover_pos())
            .is_some_and(|p| vp_bar_anchor.contains(p));
    let vp_toggle_active = ui
        .input(|i| i.pointer.hover_pos())
        .is_some_and(|p| pin_rect.contains(p) || vp_bar_anchor.contains(p));
    let nav = core.viewport.as_ref().map(|vp| {
        nav_gizmo_build(
            &vp.camera,
            gizmo_rect.center(),
            gizmo_size * 0.30,
            UiPalette::of(ui),
        )
    });
    let gizmo_active = ui
        .input(|i| i.pointer.hover_pos())
        .is_some_and(|q| nav.as_ref().is_some_and(|n| n.hit(q).is_some()));

    // The offscreen texture always spans the full viewport.
    let size = full_rect.size();
    let (w, h) = (size.x.max(1.0) as u32, size.y.max(1.0) as u32);

    // Dynamic resolution: while the user is interacting (orbiting / panning /
    // zooming / painting) render the 3D scene at a reduced internal resolution
    // and let egui upscale it — motion hides the softness, and the visible FPS
    // jump is large. The instant interaction stops the scale snaps back to 1.0,
    // so the resting frame is full quality (nothing is ever degraded at rest).
    let interacting = ui.input(|i| {
        i.pointer.middle_down()
            || i.pointer.primary_down()
            || i.pointer.secondary_down()
            || i.smooth_scroll_delta != egui::Vec2::ZERO
    });
    let target_scale = if interacting { 0.75 } else { 1.0 };
    if core.vp_scale != target_scale {
        // Cheap, known & bounded transitions only: idle -> interacting and back.
        core.vp_scale = target_scale;
    }
    let (tw, th) = (
        ((w as f32 * core.vp_scale).ceil() as u32).max(2) & !1,
        ((h as f32 * core.vp_scale).ceil() as u32).max(2) & !1,
    );

    // Ensure viewport resources exist and match the current size.
    let needs_create = core.viewport.is_none();
    if let Some(vp) = core.viewport.as_mut() {
        if vp.textures.resize(&core.device, tw, th) {
            // The offscreen texture was recreated, so the egui native texture must be too.
            if let Some(old) = vp.texture_id.take() {
                core.egui_renderer.write().free_texture(&old);
            }
        }
        vp.camera.aspect = w as f32 / h.max(1) as f32;
    }
    let just_created = core.viewport.is_none();

    if needs_create || just_created {
        let textures = ViewportTextures::new(&core.device, tw, th);
        let camera = Camera::new(w as f32 / h.max(1) as f32);
        core.viewport = Some(ViewportResources {
            textures,
            texture_id: None,
            camera,
        });
    }

    // Restore the camera from a previous session (once) instead of fitting.
    if let Some((eye, target, radius, ortho)) = core.restore_view.take() {
        let vp = core.viewport.as_mut().unwrap();
        vp.camera.eye = eye;
        vp.camera.target = target;
        vp.camera.radius = radius;
        vp.camera.ortho = ortho;
        core.needs_fit = false;
    }

    // Re-register the viewport texture with egui when it changes size. Linear
    // filtering smooths the upscale of the reduced interaction-resolution frame
    // (and keeps panel resizes from looking blocky).
    let vp = core.viewport.as_mut().unwrap();
    if vp.texture_id.is_none() {
        let id = core.egui_renderer.write().register_native_texture(
            &core.device,
            &vp.textures.color_view,
            wgpu::FilterMode::Linear,
        );
        vp.texture_id = Some(id);
    }
    let _ = vp;

    // Handle camera interaction. LMB is painting; MMB orbits and Shift+MMB
    // pans. RMB is reserved for future tools. Wheel zooms the camera, or
    // (with Shift) resizes the brush. No widget is allocated here — inputs are
    // read straight from the context while the pointer is over the viewport rect.
    let rect = full_rect;
    let hovered = ui.rect_contains_pointer(rect)
        && !ui.rect_contains_pointer(strip_rect)
        && !vp_bar_active
        && !vp_toggle_active
        && !gizmo_active;

    if core.needs_fit {
        let vp = core.viewport.as_mut().unwrap();
        vp.camera.fit(core.center, core.bounds_radius);
        core.needs_fit = false;
    }

    let (delta, scroll, m_middle, shift) = ui.input(|i| {
        (
            i.pointer.delta(),
            i.smooth_scroll_delta,
            i.pointer.middle_down(),
            i.modifiers.shift,
        )
    });

    let bind_fit = *core.shortcuts.get(ShortcutAction::Fit3d);
    if core.recording.is_none()
        && !ui.ctx().egui_wants_keyboard_input()
        && ui.rect_contains_pointer(full_rect)
        && bind_fit.is_bound()
        && ui
            .ctx()
            .input_mut(|i| i.consume_key(bind_fit.modifiers_of(), bind_fit.key_of()))
    {
        core.needs_fit = true;
    }

    let navigating = hovered && m_middle;
    if navigating {
        // Keep frames flowing while a navigation button is held so orbit/pan is smooth
        // even though no egui widget is capturing the drag.
        ui.ctx().request_repaint();
        let vp = core.viewport.as_mut().unwrap();
        vp.camera.aspect = w as f32 / h.max(1) as f32;
        if shift {
            vp.camera.pan(delta.x, delta.y, h as f32);
        } else {
            vp.camera.orbit(delta.x * 0.008, -delta.y * 0.008);
        }
    }

    if hovered && (scroll.x != 0.0 || scroll.y != 0.0) {
        ui.ctx().request_repaint();
        // On some platforms Shift+wheel surfaces as horizontal scroll, so pick
        // the dominant axis (Shift -> horizontal/brush, plain -> vertical/zoom).
        let amount = if shift {
            if scroll.x.abs() > scroll.y.abs() {
                scroll.x
            } else {
                scroll.y
            }
        } else if scroll.y.abs() > scroll.x.abs() {
            scroll.y
        } else {
            scroll.x
        };
        if shift {
            // Shift+wheel resizes the brush (screen-space radius).
            core.brush.size = (core.brush.size + amount * 0.8).clamp(1.0, 300.0);
        } else {
            let vp = core.viewport.as_mut().unwrap();
            // Blender-style: scroll up (positive egui delta) zooms in.
            vp.camera.zoom((-amount * 0.0015).exp());
            vp.camera.aspect = w as f32 / h.max(1) as f32;
        }
    }

    // RMB pops up the brush menu (color for now, brush types later).
    let secondary_clicked = ui.input(|i| i.pointer.secondary_clicked());
    if hovered && !navigating && secondary_clicked && !core.brush_menu_open {
        if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
            core.brush_menu_open = true;
            core.brush_menu_pos = Some(pos);
            ui.ctx().request_repaint();
        }
    }

    // Tool interaction: LMB paints / erases / fills / picks on the mesh.
    // Suppressed while the right-click brush menu is open so a click inside it
    // doesn't also paint on the model underneath. Locked layers still allow the
    // (read-only) picker, but reject paint / erase / fill.
    let edits_locked =
        core.active_tool != 3 && core.mesh.as_ref().map(active_layer_locked).unwrap_or(false);
    if hovered
        && !navigating
        && !core.brush_menu_open
        && !edits_locked
        && ui.input(|i| i.pointer.primary_down())
    {
        if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
            let mut painted = false;
            let mut picked: Option<[u8; 4]> = None;
            {
                let vp = core.viewport.as_ref();
                let mesh = core.mesh.as_mut();
                if let (Some(vp), Some(mesh)) = (vp, mesh) {
                    let (ndc_x, ndc_y) = viewport_ndc(pos.x, pos.y, rect);
                    let (origin, dir) = vp.camera.ray(ndc_x, ndc_y);
                    if let Some(hit) = crate::paint::mesh_raycast(mesh, origin, dir) {
                        let began = core.stroke.is_none();
                        match core.active_tool {
                            0 | 1 | 2 | 4 if began => {
                                // One undo step per stroke (or per fill press).
                                core.palette_edit_pending = false;
                                core.history.record(snapshot_of(mesh));
                                // The fill tool (2) is a one-shot flood fill: it
                                // consumes no dab loop and no occlusion/pattern
                                // machinery, so skip the mesh-wide StampAccel
                                // build, the unwrap and the whole-texture stroke
                                // buffer. A StrokeState is still recorded so the
                                // `began` lifecycle (one undo per press) is
                                // unchanged; its `accel` simply stays `None`.
                                let accel = if core.active_tool == 2 {
                                    None
                                } else {
                                    Some(crate::paint::StampAccel::new(
                                        mesh,
                                        if core.split_lock {
                                            Some(hit.triangle)
                                        } else {
                                            None
                                        },
                                    ))
                                };
                                // Pattern-locked texture strokes pin a tiled seamless
                                // fill to the click point through the captured
                                // world-space tangent frame: each texel's phase is
                                // its world position relative to that anchor, so
                                // the grid stays world-uniform no matter how the
                                // mesh's UV chart is laid out — no stretching across
                                // faces of differing texel density, and the
                                // pattern's world size is constant even if the dab
                                // radius changes mid-stroke.
                                let (pattern, unwrap) = if core.active_tool != 2
                                    && core.active_tool != 4
                                    && core.brush.pattern_lock == crate::brush::PatternLock::Aligned
                                    && core.brush.kind == crate::brush::FootprintKind::Sprite
                                    && core.brush.sprite.is_some()
                                {
                                    let world_r = screen_to_world_radius(
                                        &vp.camera,
                                        hit.position,
                                        core.brush.size,
                                        rect,
                                        w,
                                        h,
                                    );
                                    let (axis_u, axis_v) = crate::paint::brush_axes(
                                        &mesh.positions,
                                        &mesh.indices,
                                        hit.position,
                                        world_r,
                                        dir,
                                        None,
                                    );
                                    let pattern = crate::brush::PatternAnchor::Surface {
                                        pos: hit.position,
                                        axis_u,
                                        axis_v,
                                        radius: world_r,
                                    };
                                    // A "locked texture size" keeps the pattern
                                    // tile at the size it had when locking was
                                    // toggled on, no matter how the brush is
                                    // resized afterwards. Capture that size from
                                    // this first dab's world radius the first
                                    // time a locked stroke starts.
                                    if core.brush.texture_locked
                                        && core.brush.texture_size_lock <= 0.0
                                    {
                                        core.brush.texture_size_lock = world_r.max(1e-6);
                                    }
                                    // Unfold the whole reachable mesh once per
                                    // stroke so every dab (and the cursor
                                    // overlay) shares one field: phases run
                                    // *along the surface*, so a curved tile
                                    // wraps the curvature instead of collapsing
                                    // onto the anchor plane. A full unfold (not
                                    // a dab-sized patch) is what keeps the
                                    // pattern uniform all the way to the far
                                    // endpoint of a long drag — with a tiny
                                    // patch the far dabs fell back to the
                                    // chord and the stretching reappeared.
                                    // Triangles the unfold can't reach (split
                                    // parts) keep the anchor-plane chord via
                                    // the fallback.
                                    let unwrap = crate::paint::surface_unwrap(
                                        &mesh.positions,
                                        &mesh.indices,
                                        hit.position,
                                        axis_u,
                                        axis_v,
                                        hit.triangle,
                                        f32::INFINITY,
                                    );
                                    (Some(pattern), unwrap)
                                } else {
                                    (None, None)
                                };
                                // Pattern-locked strokes always allocate the stroke buffer: the
                                // replace blend must cap every overlap at one
                                // dab's worth, so the anchored texture never
                                // "fills up". Plain non-accumulative brushes
                                // allocate as before.
                                let stroke_alpha = if core.active_tool != 2 {
                                    if core.brush.accumulate && pattern.is_none() {
                                        None
                                    } else {
                                        let (tw, th) = mesh
                                            .active_layer_texture()
                                            .map(|t| (t.width as usize, t.height as usize))
                                            .unwrap_or((0, 0));
                                        if tw > 0 && th > 0 {
                                            Some(vec![0u8; tw * th])
                                        } else {
                                            None
                                        }
                                    }
                                } else {
                                    None
                                };
                                core.stroke = Some(StrokeState {
                                    last: pos,
                                    start: pos,
                                    last_dab: pos,
                                    acc: 0.0,
                                    next_t: 0.0,
                                    accel,
                                    stroke_alpha,
                                    pattern,
                                    unwrap,
                                });
                            }
                            _ => {}
                        }
                        match core.active_tool {
                            0 | 1 | 4 => {
                                // Dabs are spaced `brush_spacing` screen px apart,
                                // carrying the leftover travel across frames so a
                                // slow drag doesn't clump dabs and a fast flick
                                // still gets evenly spaced dabs along its path.
                                // Spacing 0 = continuous: step at half the brush
                                // radius so successive dabs always overlap.
                                let st = core.stroke.as_mut().expect("stroke recorded right above");
                                // `.effective_spacing()`: 0 (continuous) steps at
                                // half the brush radius so dabs always overlap.
                                // World-locked pattern strokes ride an even denser
                                // train (half the brush radius) so consecutive soft
                                // window masks overlap into one continuous stroke —
                                // no coin-edge chain, no density dip.
                                let spacing = if st.pattern.is_some() {
                                    core.brush.pattern_spacing()
                                } else {
                                    core.brush.effective_spacing()
                                };
                                let shift = ui.input(|i| i.modifiers.shift);
                                let mut dabs: Vec<egui::Pos2> = Vec::new();
                                if began {
                                    // The click itself is a dab.
                                    dabs.push(pos);
                                }
                                if shift {
                                    // Ride the straight line start -> cursor; emit
                                    // only dabs not yet covered by previous frames.
                                    let delta = pos - st.start;
                                    let t = delta.length();
                                    if t > 0.0 {
                                        let dir = delta / t;
                                        while st.next_t + spacing <= t {
                                            st.next_t += spacing;
                                            dabs.push(st.start + dir * st.next_t);
                                        }
                                    }
                                } else {
                                    let (dabs_here, acc, last_dab) =
                                        crate::paint::spaced_freehand_dabs(
                                            st.last,
                                            pos,
                                            st.last_dab,
                                            st.acc,
                                            spacing,
                                        );
                                    st.acc = acc;
                                    st.last_dab = last_dab;
                                    dabs.extend(dabs_here);
                                }
                                st.last = pos;
                                let is_rect = core.active_tool == 4;
                                let mode = if core.active_tool == 0 || core.active_tool == 4 {
                                    crate::paint::StampMode::Paint
                                } else {
                                    crate::paint::StampMode::Erase
                                };
                                // The brush object stays the single source of
                                // stamp properties (shape/sprite/color/…); only
                                // the mode varies per tool. Every dab below
                                // stamps `&core.brush` directly (no per-dab
                                // copies); the rect tool overrides the footprint
                                // with a square via `rect`. Pattern-aligned
                                // texture strokes ride the SAME spaced dab loop
                                // (each dab samples the anchored sprite phase
                                // through its footprint); the shared
                                // `stroke_alpha` buffer overlaps them into a
                                // flat non-accumulating replace.
                                core.brush.mode = mode;
                                for dab in dabs {
                                    let (dx, dy) = viewport_ndc(dab.x, dab.y, rect);
                                    let (o, d) = vp.camera.ray(dx, dy);
                                    if let Some(hi) = crate::paint::mesh_raycast(mesh, o, d) {
                                        let world_r = screen_to_world_radius(
                                            &vp.camera,
                                            hi.position,
                                            core.brush.size,
                                            rect,
                                            w,
                                            h,
                                        );
                                        crate::paint::apply_brush_stamp(
                                            mesh,
                                            hi.position,
                                            world_r,
                                            o,
                                            d,
                                            if is_rect {
                                                Some((world_r, world_r))
                                            } else {
                                                None
                                            },
                                            &core.brush,
                                            st.accel.as_ref(),
                                            st.stroke_alpha.as_deref_mut(),
                                            st.pattern.as_ref(),
                                            st.unwrap.as_ref(),
                                        );
                                        painted = true;
                                    }
                                }
                            }
                            2 => {
                                crate::paint::fill_region(
                                    mesh,
                                    hit.triangle,
                                    core.brush.color,
                                    core.brush.opacity,
                                );
                                painted = true;
                            }
                            3 => {
                                if let Some(tex) = mesh.flattened_atlas() {
                                    picked = Some(crate::paint::pick_color(&tex, &hit));
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            if let Some(c) = picked {
                core.brush.color = c;
                core.active_tool = 0;
                core.status = format!("Picked rgb({}, {}, {}) — back to Brush", c[0], c[1], c[2]);
            }
            if painted {
                // Paint/erase/fill only touch a bounded texel rect on the active
                // layer; recomposite + re-upload just that region instead of the
                // whole atlas. Fall back to a full upload when no dirty rect was
                // recorded or the GPU texture isn't ready at full size yet.
                flush_paint_edit(core);
                if core.active_tool == 1 {
                    core.status = "Erased — fully transparent (alpha 0) in the 3D view".to_string();
                }
                ui.ctx().request_repaint();
            }
        }
    }
    // End the stroke when the button is released or the pointer leaves.
    if !ui.input(|i| i.pointer.primary_down()) || !hovered {
        core.stroke = None;
    }

    // Push any freshly painted texels to the GPU before the render below: the
    // layers are flattened on the CPU, so upload is the composited atlas.
    if core.needs_texture_upload {
        if let (Some(tex), renderer) = (
            core.mesh.as_ref().and_then(|m| m.flattened_atlas()),
            &mut core.renderer,
        ) {
            renderer.update_texture(&tex);
        }
        core.needs_texture_upload = false;
    }
    // Same for the per-layer material map (roughness/metallic/emissive/ao),
    // which is a second texture sampled by the shader, and the height/bump
    // map that feeds the normal perturbation. Both are full-atlas
    // recomposites; during a fast paint stroke they're coalesced to at most
    // one upload per ~30 ms (the albedo region upload above stays per-dab), so
    // drawing frames stay cheap. When the stroke ends the pending upload
    // flushes immediately.
    if core.needs_material_upload {
        let due =
            core.stroke.is_none() || core.last_material_upload.elapsed().as_millis() as u64 >= 30;
        if due {
            let material_map = core
                .mesh
                .as_ref()
                .and_then(|m| m.flattened_material_atlas());
            core.renderer.update_material_map(material_map.as_ref());
            let height_map = core.mesh.as_ref().and_then(|m| m.flattened_height_atlas());
            core.renderer.update_height_map(height_map.as_ref());
            let extras_map = core.mesh.as_ref().and_then(|m| m.flattened_extras_atlas());
            core.renderer.update_extras_map(extras_map.as_ref());
            core.needs_material_upload = false;
            core.last_material_upload = std::time::Instant::now();
        }
    }

    // Erased texels (alpha 0) are discarded by the shader — nothing is
    // composited over them; semi-transparent ones (0 < alpha < 1) are blended
    // source-over so the backdrop shows through.

    // Render the 3D scene into the offscreen viewport texture.
    let uv_mode = (core.show_uv_checker_3d as u32 * crate::render::UV_OVERLAY_CHECKER)
        | (core.show_uv_grid_3d as u32 * crate::render::UV_OVERLAY_GRID);
    core.renderer.set_uv_overlay(uv_mode);
    core.renderer.set_material(core.material);
    // 3D brush-cursor mask: the fragment shader traces the mesh a third time
    // and keeps only the surface fragments inside the brush footprint, so the
    // cursor reads as a stencil sitting on the model. Texture brush shapes get
    // a `shape == 3` mask that samples the sprite (its alpha is the coverage,
    // exactly like the stamp).
    if let Some(sprite) = core.brush.sprite.as_ref() {
        let sig = sprite_sig(sprite);
        core.renderer.set_brush_sprite(sig, sprite);
    }
    // The active 3D stroke's pattern-lock anchor, captured at stroke
    // start; the cursor preview uses it so the texture is glued to the
    // exact seam being painted. While hovering (no stroke) the anchor
    // falls back to the current hover pose below.
    let stroke_anchor = match &core.stroke {
        Some(st) => match st.pattern {
            Some(crate::brush::PatternAnchor::Surface {
                pos,
                axis_u,
                axis_v,
                radius,
            }) => Some((pos, axis_u, axis_v, radius)),
            _ => None,
        },
        None => None,
    };
    // Per-vertex surface phase field for the cursor overlay. While an aligned
    // sprite stroke is live the field is the stroke's own geodesic unwrap, so
    // the cursor previews the exact seam being painted; hovering, a throttled
    // unwrap of the hover pose keeps the preview glued to the cursor. The
    // shader interpolates these vertex phases per fragment, mirroring the
    // per-texel CPU phase on the very same triangles.
    let overlay_phases: Option<Vec<f32>> = {
        let anchored = core.brush.pattern_lock == crate::brush::PatternLock::Aligned
            && core.brush.kind == crate::brush::FootprintKind::Sprite
            && core.brush.sprite.is_some();
        if !anchored || !hovered || navigating || core.brush_menu_open {
            None
        } else {
            ui.input(|i| i.pointer.hover_pos()).and_then(|p| {
                let vp = core.viewport.as_ref()?;
                let mesh = core.mesh.as_ref()?;
                let (nx, ny) = viewport_ndc(p.x, p.y, rect);
                let (o, d) = vp.camera.ray(nx, ny);
                let hit = crate::paint::mesh_raycast(mesh, o, d)?;
                let r =
                    screen_to_world_radius(&vp.camera, hit.position, core.brush.size, rect, w, h);
                // Live aligned sprite stroke: the cursor previews the
                // stroke's own geodesic unwrap (the exact seam being
                // painted) while it covers the cursor, mirroring the CPU
                // fallback beyond the patch.
                let live = match &core.stroke {
                    Some(st) => match (&st.pattern, &st.unwrap) {
                        (Some(crate::brush::PatternAnchor::Surface { pos, .. }), Some(u)) => {
                            Some((*pos, u.clone()))
                        }
                        _ => None,
                    },
                    None => None,
                };
                if let Some((apos, u)) = &live {
                    if (hit.position - *apos).length() <= u.radius() {
                        return Some(u.upload(mesh.positions.len()));
                    }
                    return None;
                }
                // Hover: a throttled unwrap of the hover pose keeps the
                // preview glued to the cursor. Rebuilt when the cursor
                // crosses into a new triangle, wanders the brush radius,
                // or the field ages past the throttle window.
                let (axis_u, axis_v) = crate::paint::brush_axes(
                    &mesh.positions,
                    &mesh.indices,
                    hit.position,
                    r,
                    d,
                    None,
                );
                let now = std::time::Instant::now();
                let fresh = match &core.hover_unwrap {
                    Some((at, tri, pos, _)) => {
                        *tri != hit.triangle as u32
                            || (*pos - hit.position).length() > r * 0.5
                            || now.duration_since(*at) >= std::time::Duration::from_millis(100)
                    }
                    None => true,
                };
                let hover = if fresh {
                    let u = crate::paint::surface_unwrap(
                        &mesh.positions,
                        &mesh.indices,
                        hit.position,
                        axis_u,
                        axis_v,
                        hit.triangle,
                        2.5 * r,
                    );
                    core.hover_unwrap = u
                        .clone()
                        .map(|uu| (now, hit.triangle as u32, hit.position, uu));
                    (hit.position, u)
                } else {
                    let u = core.hover_unwrap.as_ref().map(|(_, _, _, uu)| uu.clone());
                    let anchor_pos = core
                        .hover_unwrap
                        .as_ref()
                        .map(|(_, _, pos, _)| *pos)
                        .unwrap_or(hit.position);
                    (anchor_pos, u)
                };
                hover
                    .1
                    .filter(|u| (hit.position - hover.0).length() <= u.radius())
                    .map(|u| u.upload(mesh.positions.len()))
            })
        }
    };
    core.renderer.brush_overlay = {
        // 0 = round, 1 = square, 2 = diamond, 3 = texture-sprite. Rect tools
        // stamp a square footprint, so they get the square mask regardless of
        // the brush shape; a sprite shape with no sprite loads no mask (the
        // flat screen-space fallback cursor remains). Every sprite brush is
        // drawn as a texture-sprite: aligned texture brushes sample the
        // anchored, world-locked seam through it, dab brushes the plain
        // sprite.
        let overlay_shape: Option<u32> = match core.active_tool {
            4 => Some(1),
            0 | 1 => match core.brush.kind {
                crate::brush::FootprintKind::Round => Some(0),
                crate::brush::FootprintKind::Square => Some(1),
                crate::brush::FootprintKind::Diamond => Some(2),
                crate::brush::FootprintKind::Sprite if core.brush.sprite.is_some() => Some(3),
                crate::brush::FootprintKind::Sprite | crate::brush::FootprintKind::Rect => None,
            },
            _ => None,
        };
        if overlay_shape.is_some() && hovered && !navigating && !core.brush_menu_open {
            let pos = ui.input(|i| i.pointer.hover_pos());
            pos.and_then(|p| {
                let vp = core.viewport.as_ref()?;
                let mesh = core.mesh.as_ref()?;
                let (nx, ny) = viewport_ndc(p.x, p.y, rect);
                let (o, d) = vp.camera.ray(nx, ny);
                crate::paint::mesh_raycast(mesh, o, d).map(|hit| {
                    let r = screen_to_world_radius(
                        &vp.camera,
                        hit.position,
                        core.brush.size,
                        rect,
                        w,
                        h,
                    );
                    let (axis_u, axis_v) = crate::paint::brush_axes(
                        &mesh.positions,
                        &mesh.indices,
                        hit.position,
                        r,
                        d,
                        None,
                    );
                    let shape = overlay_shape.expect("checked just above");
                    // Anchored pattern preview: while an aligned texture stroke
                    // is live the frame rides the stroke-start anchor, so the
                    // cursor shows the exact seam being painted; hovering, it
                    // centers the repeat on the cursor. Dab sprites preview as
                    // plain rubber stamps (no anchor).
                    let anchored = core.brush.pattern_lock == crate::brush::PatternLock::Aligned;
                    let anchor = if anchored {
                        let (ap, au, av, ar) =
                            stroke_anchor.unwrap_or((hit.position, axis_u, axis_v, r));
                        // Same "phase divider" rule as the 3D stamp
                        // (`paint.rs stamp_texels`): the pattern's world size
                        // locks to the stroke-start dab, or the captured fixed
                        // size when texture-locked. The shader cancels this
                        // against `texture_scale` via `r / anchor.w`.
                        let divider =
                            if core.brush.texture_locked && core.brush.texture_size_lock > 0.0 {
                                core.brush.texture_size_lock
                            } else {
                                ar.max(1e-6)
                            };
                        Some(crate::render::OverlayAnchor {
                            pos: ap,
                            axis_u: au,
                            axis_v: av,
                            radius: divider / core.brush.texture_scale.max(1e-6),
                        })
                    } else {
                        None
                    };
                    crate::render::BrushOverlay {
                        center: hit.position,
                        axis_u,
                        axis_v,
                        radius: r,
                        shape,
                        window: window_code(core.brush.texture_window),
                        color: [
                            core.brush.color[0] as f32 / 255.0,
                            core.brush.color[1] as f32 / 255.0,
                            core.brush.color[2] as f32 / 255.0,
                            0.23,
                        ],
                        rotation: core.brush.rotation,
                        flip_x: core.brush.flip_x,
                        flip_y: core.brush.flip_y,
                        anchor,
                        phases: overlay_phases,
                    }
                })
            })
        } else {
            None
        }
    };
    if let Some(vp) = core.viewport.as_ref() {
        core.renderer
            .render(&vp.camera, &vp.textures.color_view, &vp.textures.depth_view);
    }

    // Draw the offscreen texture across the whole viewport (under the T-bar).
    // The render is fully opaque — transparency was already resolved to the
    // backdrop by skipping alpha<1 fragments entirely.
    if let Some(vp) = core.viewport.as_ref() {
        if let Some(tex_id) = vp.texture_id {
            ui.painter().image(
                tex_id,
                full_rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
        }
    }

    // Left-edge vertical tool strip (drawn on top of the render, Blender-style).
    // Hidden when the slide animation has fully exited.
    if core.tool_strip_anim > 0.001 {
        view_tool_strip(ui, core, strip_rect);
    }

    // Top-edge horizontal overlay bar (UV checker / grid toggles) + its
    // visibility button, floating over the render like the T-bar floats over
    // the left edge. The bar only draws while explicitly pinned.
    if core.show_vp_overlay_bar {
        vp_overlay_bar(ui, core, vp_bar_anchor, full_rect);
    }
    vp_overlay_toggle(ui, core, pin_rect);
    viewport_nav_gizmo(ui, core, gizmo_rect, nav.as_ref());

    // Brush preview shape: a fixed-size ring in screen pixels matching the brush
    // radius (Paint/Eraser) or a square outline (Rect). Shift+wheel
    // in the viewport resizes it.
    if (core.active_tool == 0 || core.active_tool == 1 || core.active_tool == 4)
        && hovered
        && !navigating
    {
        let pos = ui.input(|i| i.pointer.hover_pos());
        if let Some(pos) = pos {
            // Shape brushes, texture brushes with a sprite, and rect tools get
            // their cursor drawn by the renderer's 3D overlay pipeline whenever
            // the pointer is over the mesh, so no flat screen-space cursor is
            // needed there — showing one would double the mask. The egui
            // fallback below only appears when no surface mask exists (over
            // empty space, a sprite-less texture brush, or while the menu is
            // open).
            let on_surface_mask = !core.brush_menu_open
                && (core.active_tool == 4
                    || matches!(
                        core.brush.kind,
                        crate::brush::FootprintKind::Round
                            | crate::brush::FootprintKind::Square
                            | crate::brush::FootprintKind::Diamond
                    )
                    || (matches!(core.brush.kind, crate::brush::FootprintKind::Sprite)
                        && core.brush.sprite.is_some()))
                && core
                    .mesh
                    .as_ref()
                    .zip(core.viewport.as_ref())
                    .is_some_and(|(mesh, vp)| {
                        let (nx, ny) = viewport_ndc(pos.x, pos.y, rect);
                        let (o, d) = vp.camera.ray(nx, ny);
                        crate::paint::mesh_raycast(mesh, o, d).is_some()
                    });
            if on_surface_mask {
                return;
            }
            let screen_r = core.brush.size;
            let painting = core.active_tool == 0 || core.active_tool == 4;
            let (fill, stroke, dot) = if painting {
                (
                    egui::Color32::from_rgba_unmultiplied(
                        core.brush.color[0],
                        core.brush.color[1],
                        core.brush.color[2],
                        60,
                    ),
                    egui::Color32::from_rgba_unmultiplied(
                        core.brush.color[0],
                        core.brush.color[1],
                        core.brush.color[2],
                        255,
                    ),
                    egui::Color32::WHITE,
                )
            } else {
                (
                    egui::Color32::from_rgba_unmultiplied(255, 255, 255, 30),
                    egui::Color32::WHITE,
                    egui::Color32::WHITE,
                )
            };
            // Fallback cursor for cases where the renderer draws no 3D mask:
            // rect outlines, texture-sprite previews, or hovering empty space.
            if core.active_tool == 4 {
                let square =
                    egui::Rect::from_center_size(pos, egui::vec2(screen_r * 2.0, screen_r * 2.0));
                ui.painter().rect_filled(square, 0.0, fill);
                ui.painter().rect_stroke(
                    square,
                    0.0,
                    egui::Stroke::new(1.5, stroke),
                    egui::StrokeKind::Outside,
                );
            } else {
                match core.brush.kind {
                    crate::brush::FootprintKind::Round => {
                        ui.painter().circle_filled(pos, screen_r, fill);
                        ui.painter()
                            .circle_stroke(pos, screen_r, egui::Stroke::new(1.5, stroke));
                    }
                    crate::brush::FootprintKind::Square | crate::brush::FootprintKind::Rect => {
                        let square = egui::Rect::from_center_size(
                            pos,
                            egui::vec2(screen_r * 2.0, screen_r * 2.0),
                        );
                        ui.painter().rect_filled(square, 0.0, fill);
                        ui.painter().rect_stroke(
                            square,
                            0.0,
                            egui::Stroke::new(1.5, stroke),
                            egui::StrokeKind::Outside,
                        );
                    }
                    crate::brush::FootprintKind::Sprite => {
                        // WYSIWYG cursor: rasterize the actual stamp (frame
                        // shape + tiled sprite + rotation + flips, tinted) at
                        // the dab radius, so selecting Square/Diamond shows the
                        // square/diamond frame live under the cursor and it
                        // exactly matches the painted dab.
                        let Some(_sprite) = &core.brush.sprite else {
                            let square = egui::Rect::from_center_size(
                                pos,
                                egui::vec2(screen_r * 2.0, screen_r * 2.0),
                            );
                            ui.painter().rect_filled(square, 0.0, fill);
                            ui.painter().rect_stroke(
                                square,
                                0.0,
                                egui::Stroke::new(1.5, stroke),
                                egui::StrokeKind::Outside,
                            );
                            return;
                        };
                        let tex = brush_cursor_preview(ui.ctx(), core, screen_r);
                        let side = (screen_r * 2.0).max(1.0);
                        let square = egui::Rect::from_center_size(pos, egui::vec2(side, side));
                        ui.painter().image(
                            tex,
                            square,
                            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                            egui::Color32::WHITE,
                        );
                    }
                    crate::brush::FootprintKind::Diamond => {
                        let r = screen_r;
                        let pts = vec![
                            pos + egui::vec2(0.0, -r),
                            pos + egui::vec2(r, 0.0),
                            pos + egui::vec2(0.0, r),
                            pos + egui::vec2(-r, 0.0),
                        ];
                        ui.painter().add(egui::Shape::convex_polygon(
                            pts,
                            fill,
                            egui::Stroke::new(1.5, stroke),
                        ));
                    }
                }
            }
            ui.painter()
                .circle_stroke(pos, 2.0, egui::Stroke::new(1.0, dot));
            ui.ctx().request_repaint();
        }
    }

    // Pick tool: hide the OS cursor and draw the pipette icon (lucide, ISC)
    // centered so its tip sits near the pointer.
    if core.active_tool == 3 && hovered && !navigating && !core.brush_menu_open {
        ui.ctx().set_cursor_icon(egui::CursorIcon::None);
        if let (Some(icon), Some(pos)) = (
            core.pick_icon_tex(ui.ctx()),
            ui.input(|i| i.pointer.hover_pos()),
        ) {
            draw_pick_icon(ui.ctx(), icon, pos);
            ui.ctx().request_repaint();
        }
    }

    brush_menu_popup(ui, core);

    // Status bar pinned to the viewport's bottom (staying clear of the T-bar).
    let status_h = 44.0;
    let status_rect = egui::Rect::from_min_max(
        full_rect.left_bottom() - egui::vec2(0.0, status_h),
        full_rect.right_bottom(),
    );
    ui.scope_builder(
        egui::UiBuilder::new()
            .max_rect(status_rect)
            .layout(egui::Layout::bottom_up(egui::Align::LEFT)),
        |ui| {
            let pal = UiPalette::of(ui);
            ui.add_space(4.0);
            let mut hint = String::from(
                "LMB paint  |  Shift+LMB: straight stroke  |  MMB drag: orbit  |  Shift+MMB drag: pan  |  Wheel: zoom  |  Shift+Wheel: brush size  |  RMB: brush menu",
            );
            let fit = *core.shortcuts.get(ShortcutAction::Fit3d);
            if fit.is_bound() {
                hint.push_str(&format!("  |  {}: fit", fit.label()));
            }
            let tools = *core.shortcuts.get(ShortcutAction::ToggleTools3d);
            if tools.is_bound() {
                hint.push_str(&format!("  |  {}: tools on/off", tools.label()));
            }
            ui.label(
                egui::RichText::new(hint)
                    .small()
                    .color(pal.chrome_text_weak),
            );
            ui.add_space(6.0);
            ui.label(
                egui::RichText::new(&core.status)
                    .color(pal.chrome_text_weak),
            );
        },
    );
}

/// Where a tool strip (the 3D T-bar, or the Texture preview's brush picker)
/// sits for a given anchor (top-left of the hosting panel) and slide progress:
/// 0 = fully hidden, pushed `STRIP_HIDE_EXTRA` past the left edge, 1 = flush
/// with the anchor. Constant-speed sliding means it never stalls half-visible.
fn tool_strip_rect(anchor_min: egui::Pos2, anim: f32) -> egui::Rect {
    let side = (STRIP_W - 6.0).max(18.0);
    // Each button advances the cursor by `side` plus the trailing item spacing,
    // so the pill must reserve `STRIP_GAP` per button (not just between them) or
    // the bottom button pokes past the rounded background.
    let content_h = STRIP_PAD * 2.0 + TOOLS.len() as f32 * (side + STRIP_GAP);
    let total = STRIP_W + STRIP_HIDE_EXTRA;
    egui::Rect::from_min_size(
        anchor_min + egui::vec2(-total * (1.0 - anim), STRIP_TOP_INSET),
        egui::vec2(STRIP_W, content_h),
    )
}

/// Blender-style vertical T-bar overlaid on the viewport's left edge: a slim
/// translucent pill (fully rounded corners) with one compact icon per tool.
fn view_tool_strip(ui: &mut Ui, core: &mut Core, strip_rect: egui::Rect) {
    // Eagerly ensure icons are loaded before we try to paint buttons
    if core.icons.is_none() {
        core.icons = load_icons(ui.ctx());
    }

    // One consistent corner radius everywhere so the backdrop and the buttons
    // read as a single pill.
    let pal = UiPalette::of(ui);
    let corner = egui::CornerRadius::same(RADIUS_PILL);

    // Translucent glass that flips with the theme.
    ui.painter().rect_filled(strip_rect, corner, pal.overlay);
    // Subtle accent highlight across the top edge.
    let highlight_rect =
        egui::Rect::from_min_size(strip_rect.left_top(), egui::vec2(strip_rect.width(), 2.0));
    ui.painter().rect_filled(
        highlight_rect,
        egui::CornerRadius {
            nw: RADIUS_PILL,
            ne: RADIUS_PILL,
            sw: 0,
            se: 0,
        },
        egui::Color32::from_rgba_unmultiplied(ACCENT.r(), ACCENT.g(), ACCENT.b(), 70),
    );
    ui.painter().rect_stroke(
        strip_rect,
        corner,
        egui::Stroke::new(1.0, pal.overlay_border),
        egui::StrokeKind::Inside,
    );
    ui.scope_builder(
        egui::UiBuilder::new()
            .max_rect(strip_rect)
            .layout(egui::Layout::top_down(egui::Align::Center)),
        |ui| {
            ui.set_min_height(strip_rect.height());
            // No gap between buttons; STRIP_PAD breathing room on both ends so
            // the pill hugs the icons (first & last button stay inside).
            ui.spacing_mut().item_spacing = egui::vec2(0.0, STRIP_GAP);
            ui.add_space(STRIP_PAD);
            for index in 0..TOOLS.len() {
                tool_strip_button(ui, core, index, strip_rect.width());
            }
            ui.add_space(STRIP_PAD);
        },
    );
}

/// The compact horizontal toolbar pill shown (pinned) at the 3D viewport's
/// top-right, left of the visibility button. Carries the viewport overlay
/// toggles (UV checkerboard / UV grid). The backdrop hugs the contents with
/// ~10px padding, so its width is whatever the toggles need — never fills the
/// viewport. Clipped to the viewport so it can't bleed over other panels.
fn vp_overlay_bar(ui: &mut Ui, core: &mut Core, anchor: egui::Rect, viewport: egui::Rect) {
    ui.scope_builder(
        egui::UiBuilder::new()
            .max_rect(anchor)
            .layout(egui::Layout::right_to_left(egui::Align::Center)),
        |ui| {
            ui.set_clip_rect(viewport);
            let pal = UiPalette::of(ui);
            egui::Frame::new()
                .fill(pal.overlay)
                .stroke(egui::Stroke::new(1.0, pal.overlay_border))
                .corner_radius(egui::CornerRadius::same(RADIUS_CARD))
                .inner_margin(egui::Margin::symmetric(10, 6))
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(6.0, 0.0);
                    ui.checkbox(&mut core.show_uv_checker_3d, "UV checker")
                        .on_hover_text("Checkerboard overlay, mapped through the UVs");
                    ui.checkbox(&mut core.show_uv_grid_3d, "UV grid")
                        .on_hover_text("UV grid overlay, mapped through the UVs");
                    ui.checkbox(&mut core.split_lock, "Split lock")
                        .on_hover_text(
                            "Restrict a stroke to the mesh part connected to the face under the \
                             brush, so it can't bleed onto separate parts in reach",
                        );
                });
        },
    );
}

/// The small floating button at the 3D viewport's top-right that toggles the
/// horizontal overlay bar's visibility: anchors the bar open, or (when
/// unpinned) temporarily re-shows it via hover.
fn vp_overlay_toggle(ui: &mut Ui, core: &mut Core, rect: egui::Rect) {
    let resp = ui.allocate_rect(rect, egui::Sense::click());
    let pal = UiPalette::of(ui);
    let corner = egui::CornerRadius::same(RADIUS_CONTROL);
    let hovered = resp.hovered();
    let p = ui.painter();
    p.rect_filled(
        rect,
        corner,
        if hovered {
            pal.control_hover
        } else {
            pal.overlay
        },
    );
    p.rect_stroke(
        rect,
        corner,
        egui::Stroke::new(1.0, pal.overlay_border),
        egui::StrokeKind::Inside,
    );
    let icon = if core.show_vp_overlay_bar { "–" } else { "+" };
    let tip = if core.show_vp_overlay_bar {
        "Hide the 3D overlay bar"
    } else {
        "Show the 3D overlay bar"
    };
    p.text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        icon,
        egui::FontId::proportional(12.0),
        pal.overlay_text,
    );
    if resp.clicked() {
        core.show_vp_overlay_bar = !core.show_vp_overlay_bar;
    }
    resp.clone().on_hover_text(tip);
}

/// Precomputed screen-space layout for the viewport navigation gizmo: six
/// orthographic axis views (front/back/right/left/top/bottom) plus the
/// perspective "home" hub. Pin positions are the world axis projected
/// orthographically onto the camera's right/up plane, so the widget mirrors the
/// model's visible orientation and each pin glides through the hub as its axis
/// swings toward the view direction (no snapping).
struct NavGizmo {
    center: egui::Pos2,
    /// Screen distance from the hub to each dot, in points.
    arm: f32,
    /// (view label, axis to look along, dot position, dot color).
    views: [(&'static str, glam::Vec3, egui::Pos2, egui::Color32); 6],
    /// Indices into `views` that are the positive (fully colored) axes.
    positives: [usize; 3],
    /// Per-view depth: `axis · forward`. Positive = toward the camera (front
    /// hemisphere, drawn solid); negative = behind it (drawn faint).
    depths: [f32; 6],
}

impl NavGizmo {
    /// Hit radius around each axis dot, in points.
    const DOT_R: f32 = 15.0;
    /// Hit radius around the home hub, in points. Kept smaller than `DOT_R`
    /// so the (more important) axis pins win any overlap at the center.
    const HUB_R: f32 = 9.0;

    /// Interactive target under `q`: `Some(6)` for the home hub, the index of
    /// a dot, or `None` for a dead corner where painting passes through. When
    /// a front pin passes over the hub (axis nearly view-aligned) the nearest
    /// target wins.
    fn hit(&self, q: egui::Pos2) -> Option<usize> {
        let mut best: Option<(usize, f32)> = None;
        let hub = self.center.distance(q);
        if hub <= Self::HUB_R {
            best = Some((6, hub));
        }
        for (i, (_, _, pt, _)) in self.views.iter().enumerate() {
            // Bias toward the front hemisphere so a faint back pin never steals
            // a click from the solid front pin it overlaps.
            let bias = if self.depths[i] < 0.0 { 8.0 } else { 0.0 };
            let d = pt.distance(q) + bias;
            if d <= Self::DOT_R && best.is_none_or(|(_, bd)| d < bd) {
                best = Some((i, d));
            }
        }
        best.map(|(i, _)| i)
    }
}

/// Position for a pin on the gizmo face. Orthographic projection of the world
/// axis onto the camera's right/up plane — deliberately NOT normalized: a pin's
/// distance from the hub is the axis' perpendicular component, so as an axis
/// swings toward the view direction its pin glides smoothly in to the center
/// (and back out the far side) instead of snapping/flickering on the rim.
fn nav_gizmo_build(
    camera: &crate::render::Camera,
    center: egui::Pos2,
    arm: f32,
    pal: UiPalette,
) -> NavGizmo {
    let forward = (camera.target - camera.eye).normalize_or_zero();
    let right = forward.cross(camera.up).normalize_or_zero();
    let up = right.cross(forward).normalize_or_zero();
    let pin_of = |dir: glam::Vec3| {
        let x = dir.dot(right);
        let y = -dir.dot(up);
        egui::pos2(center.x + x * arm, center.y + y * arm)
    };
    let axes = [
        glam::Vec3::X,
        glam::Vec3::NEG_X,
        glam::Vec3::Y,
        glam::Vec3::NEG_Y,
        glam::Vec3::Z,
        glam::Vec3::NEG_Z,
    ];
    // Vibrant, Blender-like axis hues (theme-tuned so they read on light too).
    // Both signs share the axis color (depth shading conveys near/far); every
    // pin carries its sign in the label.
    let red = pal.axis_x;
    let green = pal.axis_y;
    let blue = pal.axis_z;
    NavGizmo {
        center,
        arm,
        views: [
            ("Right", glam::Vec3::X, pin_of(glam::Vec3::X), red),
            ("Left", glam::Vec3::NEG_X, pin_of(glam::Vec3::NEG_X), red),
            ("Top", glam::Vec3::Y, pin_of(glam::Vec3::Y), green),
            (
                "Bottom",
                glam::Vec3::NEG_Y,
                pin_of(glam::Vec3::NEG_Y),
                green,
            ),
            ("Back", glam::Vec3::Z, pin_of(glam::Vec3::Z), blue),
            ("Front", glam::Vec3::NEG_Z, pin_of(glam::Vec3::NEG_Z), blue),
        ],
        positives: [0, 2, 4],
        depths: axes.map(|a| a.dot(forward)),
    }
}

/// Draws the viewport navigation gizmo as a Blender-style wireframe globe (a
/// translucent sphere with meridian rings, colored axis pins on the rim, home
/// hub in the center) and handles its clicks: an axis pin snaps the camera to
/// that orthographic view, the home hub returns to perspective. Only the
/// pins/hub block pointer input (see `NavGizmo::hit`), so the surrounding
/// corner of the viewport stays paintable.
fn viewport_nav_gizmo(ui: &mut Ui, core: &mut Core, rect: egui::Rect, nav: Option<&NavGizmo>) {
    let Some(nav) = nav else {
        return;
    };
    let resp = ui.allocate_rect(rect, egui::Sense::click());
    let p = ui.painter();
    let hovered = ui.input(|i| i.pointer.hover_pos()).and_then(|q| nav.hit(q));

    let c = nav.center;
    let pal = UiPalette::of(ui);
    // Globe body: a soft glass ball with a gentle top-left sheen and a bright
    // rim, sized a little past the pin radius so the vibrant axis pins sit
    // comfortably inside it. Kept light so the scene still shows through.
    let rim = nav.arm * 1.5;
    let (glass, sheen, ring_outer, ring_inner, ink) = if pal.dark {
        (
            egui::Color32::from_black_alpha(60),
            egui::Color32::from_white_alpha(16),
            egui::Color32::from_black_alpha(70),
            egui::Color32::from_white_alpha(150),
            egui::Color32::WHITE,
        )
    } else {
        (
            egui::Color32::from_white_alpha(88),
            egui::Color32::from_white_alpha(140),
            egui::Color32::from_black_alpha(36),
            egui::Color32::from_black_alpha(64),
            egui::Color32::from_rgb(32, 36, 43),
        )
    };
    p.circle_filled(c, rim, glass);
    p.circle_filled(c + egui::vec2(-rim * 0.22, -rim * 0.22), rim * 0.72, sheen);
    p.circle_stroke(c, rim + 1.0, egui::Stroke::new(2.0, ring_outer));
    p.circle_stroke(c, rim, egui::Stroke::new(1.4, ring_inner));

    // Home hub: a small, understated house at the center (perspective reset).
    // Deliberately quieter than the axis pins — the axes are the primary
    // targets and drawn on top of the hub, so the hub never competes with them.
    let home_r = if hovered == Some(6) { 9.0 } else { 8.0 };
    p.circle_filled(c, home_r, glass);
    p.circle_stroke(c, home_r, egui::Stroke::new(1.3, ring_inner));
    let src = if hovered == Some(6) { 235 } else { 150 };
    let glyph = |a: u8| ink.gamma_multiply(if pal.dark { 1.0 } else { a as f32 / 255.0 });
    p.add(egui::Shape::convex_polygon(
        vec![
            egui::pos2(c.x, c.y - 3.6),
            egui::pos2(c.x - 4.0, c.y - 0.4),
            egui::pos2(c.x + 4.0, c.y - 0.4),
        ],
        glyph(src),
        egui::Stroke::NONE,
    ));
    p.rect_filled(
        egui::Rect::from_min_max(
            egui::pos2(c.x - 2.8, c.y - 0.4),
            egui::pos2(c.x + 2.8, c.y + 2.4),
        ),
        0.0,
        glyph(src),
    );

    // Axis spokes + pins — the primary targets, drawn last so they sit on top
    // of the hub. Every axis is a labeled bead; front-hemisphere axes (depth >
    // 0) are larger/brighter, back-hemisphere ones fade for correct near/far.
    let labels = ["X", "-X", "Y", "-Y", "Z", "-Z"];
    for (i, (_, _, pt, color)) in nav.views.iter().enumerate() {
        // Smooth near/far fade across the whole sphere (never a hard cut).
        let a = (0.66 + 0.34 * nav.depths[i]).clamp(0.3, 1.0);
        let col = color.gamma_multiply(a);
        let hover = hovered == Some(i);
        let pos = nav.positives.contains(&i);
        let r = if pos {
            if hover {
                8.6
            } else {
                7.0
            }
        } else if hover {
            7.6
        } else {
            6.2
        };
        let base = c + (*pt - c) * 0.5;
        p.line_segment(
            [base, *pt],
            egui::Stroke::new(if pos { 2.6 } else { 2.0 }, col),
        );
        // Layered halos fake a soft glow (no blur in egui).
        p.circle_filled(*pt, r + 5.0, color.gamma_multiply(0.16 * a));
        p.circle_filled(*pt, r + 2.6, color.gamma_multiply(0.28 * a));
        p.circle_filled(*pt, r, col);
        // Top-left specular so the pin reads as a glossy bead.
        p.circle_filled(
            *pt - egui::vec2(r * 0.32, r * 0.32),
            r * 0.4,
            egui::Color32::from_white_alpha((110.0 * a) as u8),
        );
        p.circle_stroke(
            *pt,
            r,
            egui::Stroke::new(1.4, egui::Color32::from_white_alpha((150.0 * a) as u8)),
        );
        p.text(
            *pt + egui::vec2(-11.0, 9.5),
            egui::Align2::CENTER_CENTER,
            labels[i],
            egui::FontId::proportional(11.0),
            col,
        );
    }

    if resp.clicked() {
        let q = resp.interact_pointer_pos();
        if let Some(h) = q.and_then(|q| nav.hit(q)) {
            let vp = core.viewport.as_mut().unwrap();
            if h == 6 {
                vp.camera.view_home();
                core.status = "Perspective view".to_string();
            } else {
                let (label, axis, _, _) = &nav.views[h];
                vp.camera.look_along(*axis);
                core.status = format!("View: {} (orthographic)", label);
            }
        }
    }
    if let Some(h) = hovered {
        let label = if h == 6 {
            "Home (perspective)"
        } else {
            nav.views[h].0
        };
        resp.on_hover_text(label);
    }
}

/// One square tool button inside the in-viewport T-bar (also reused by the
/// Texture preview's brush picker) using crisp Lucide icons and Blender styling.
fn tool_strip_button(ui: &mut Ui, core: &mut Core, index: usize, strip_width: f32) {
    if core.icons.is_none() {
        core.icons = load_icons(ui.ctx());
    }

    let side = (strip_width - 6.0).max(22.0);
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(side, side), egui::Sense::click());
    let pal = UiPalette::of(ui);
    let corner = egui::CornerRadius::same(RADIUS_CONTROL);
    let is_active = core.active_tool == index;
    let p = ui.painter();

    if is_active {
        p.rect_filled(rect, corner, ACCENT);
        p.rect_stroke(
            rect,
            corner,
            egui::Stroke::new(1.0, egui::Color32::WHITE),
            egui::StrokeKind::Inside,
        );
    } else if resp.hovered() {
        p.rect_filled(rect, corner, pal.control_hover);
        p.rect_stroke(
            rect,
            corner,
            egui::Stroke::new(1.0, ACCENT_HOVER),
            egui::StrokeKind::Inside,
        );
    } else {
        p.rect_filled(rect, corner, pal.control);
        p.rect_stroke(
            rect,
            corner,
            egui::Stroke::new(1.0, pal.control_border),
            egui::StrokeKind::Inside,
        );
    }

    let tint = if is_active {
        egui::Color32::WHITE
    } else if resp.hovered() {
        pal.control_text
    } else {
        pal.chrome_text_weak
    };

    let center = rect.center();
    let icon_size = (side - 8.0).max(16.0);
    let icon_rect = egui::Rect::from_center_size(center, egui::vec2(icon_size, icon_size));

    if let Some(icons) = &core.icons {
        match index {
            0 => {
                if core.brush.sprite.is_some()
                    || core.brush.kind == crate::brush::FootprintKind::Sprite
                {
                    let tex = brush_cursor_preview(ui.ctx(), core, 32.0);
                    p.image(
                        tex,
                        icon_rect,
                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                        tint,
                    );
                } else {
                    p.image(
                        icons.brush.id(),
                        icon_rect,
                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                        tint,
                    );
                }
            }
            1 => {
                p.image(
                    icons.eraser.id(),
                    icon_rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    tint,
                );
            }
            2 => {
                p.image(
                    icons.fill.id(),
                    icon_rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    tint,
                );
            }
            3 => {
                p.image(
                    icons.pick.id(),
                    icon_rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    tint,
                );
            }
            4 => {
                p.image(
                    icons.rect.id(),
                    icon_rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    tint,
                );
            }
            _ => {}
        }
    }

    let name = match index {
        0 => "Brush",
        1 => "Eraser",
        2 => "Fill Bucket",
        3 => "Pipette / Pick Color",
        4 => "Rectangle Stamp",
        _ => TOOLS.get(index).copied().unwrap_or("Tool"),
    };
    // Reflect the *actual* (remappable) shortcut instead of a stale literal.
    let tooltip = ShortcutAction::ALL
        .iter()
        .find(|a| a.tool_index() == Some(index))
        .map(|a| core.shortcuts.get(*a))
        .filter(|b| b.is_bound())
        .map(|b| format!("{} [{}]", name, b.label()))
        .unwrap_or_else(|| name.to_string());

    if resp.clicked() {
        core.active_tool = index;
        core.status = format!("Tool: {}", TOOLS[index]);
    }
    resp.on_hover_text(tooltip);
}

/// Renders the right-click brush menu as a floating popup over the viewport.
/// Closes on Escape or any click outside it.
fn brush_menu_popup(ui: &mut Ui, core: &mut Core) {
    if !core.brush_menu_open {
        return;
    }
    // Drawn at most once a frame even when multiple panels are visible.
    if core.brush_menu_rendered {
        return;
    }
    core.brush_menu_rendered = true;
    let pos = core.brush_menu_pos.unwrap_or_else(|| {
        ui.input(|i| i.pointer.hover_pos())
            .unwrap_or_else(|| ui.max_rect().left_top())
    });
    ui.ctx().request_repaint();
    let inner = egui::Area::new(egui::Id::new("viewport_brush_menu"))
        .order(egui::Order::Foreground)
        .fixed_pos(pos)
        .constrain(true)
        .show(ui.ctx(), |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                draw_brush_menu(ui, core);
            });
        });
    close_brush_menu_on_outside_click(ui, core, inner.response.rect);
}

/// Closes the brush menu on Escape or a new press outside the popup. Closing
/// is keyed to fresh presses (not the live cursor position) so dragging the
/// color picker around doesn't flicker the menu open/closed.
fn close_brush_menu_on_outside_click(ui: &mut Ui, core: &mut Core, popup_rect: egui::Rect) {
    let escape = ui.input(|i| i.key_pressed(egui::Key::Escape));
    let press_outside = ui.input(|i| {
        (i.pointer.primary_pressed() || i.pointer.secondary_pressed())
            && i.pointer
                .interact_pos()
                .is_some_and(|p| !popup_rect.expand(4.0).contains(p))
    });
    if escape || press_outside {
        core.brush_menu_open = false;
        core.brush_menu_pos = None;
    }
}

fn draw_brush_menu(ui: &mut Ui, core: &mut Core) {
    // Give the inline color picker room to breathe (SV box + hue slider sized
    // from this) while keeping the whole popup compact.
    ui.spacing_mut().slider_width = 132.0;
    ui.spacing_mut().item_spacing.y = 6.0;
    ui.set_min_width(168.0);
    ui.set_max_width(200.0);

    ui.strong("Brush");
    ui.separator();

    // Active palette quick-picks (from the Palette panel).
    if let Some(pal) = core.palettes.get(core.active_palette) {
        if !pal.colors.is_empty() {
            ui.label(egui::RichText::new(&pal.name).small().weak());
            ui.horizontal_wrapped(|ui| {
                for c in pal.colors.iter().take(60) {
                    let rgb = [c[0], c[1], c[2]];
                    if swatch_button(ui, rgb).clicked() {
                        core.brush.color = [rgb[0], rgb[1], rgb[2], core.brush.color[3]];
                        core.brush_menu_open = false;
                        core.brush_menu_pos = None;
                    }
                }
            });
            ui.separator();
        }
    }

    // Custom color: chip + hex, then the live picker right below.
    let cur = core.brush.color;
    ui.horizontal(|ui| {
        let pal = UiPalette::of(ui);
        let chip = egui::CornerRadius::same(RADIUS_CHIP);
        let (rect, _) = ui.allocate_exact_size(egui::vec2(18.0, 18.0), egui::Sense::hover());
        ui.painter().rect_filled(
            rect,
            chip,
            egui::Color32::from_rgba_unmultiplied(cur[0], cur[1], cur[2], cur[3]),
        );
        ui.painter().rect_stroke(
            rect,
            chip,
            egui::Stroke::new(1.0, pal.card_border),
            egui::StrokeKind::Inside,
        );
        ui.label(format!(
            "#{:02X}{:02X}{:02X}   α{:02X}",
            cur[0], cur[1], cur[2], cur[3]
        ));
    });
    let mut col = egui::Color32::from_rgba_unmultiplied(cur[0], cur[1], cur[2], cur[3]);
    // Blend alpha is exposed so you can paint translucent colors for glasses
    // and other semi-opaque materials (alpha<255 blends toward the fill).
    let changed = egui::color_picker::color_picker_color32(
        ui,
        &mut col,
        egui::color_picker::Alpha::OnlyBlend,
    );
    if changed {
        core.brush.color = col.to_srgba_unmultiplied();
        core.status = format!(
            "Brush color #{:02X}{:02X}{:02X} α{:02X}",
            core.brush.color[0], core.brush.color[1], core.brush.color[2], core.brush.color[3]
        );
    }

    ui.separator();

    // Convenience: sample a color straight from the model.
    if core.pick_icon_tex(ui.ctx()).is_some() {
        let icon = core.pick_icon_tex(ui.ctx()).unwrap();
        let img = egui::Image::new(icon)
            .fit_to_exact_size(egui::vec2(16.0, 16.0))
            .tint(ui.visuals().text_color());
        let pick_btn = egui::Button::image_and_text(img, "Pick from model");
        let resp = ui
            .add(pick_btn)
            .on_hover_text("Switches to the Pick tool — click a spot on the model to sample it");
        if resp.clicked() {
            core.active_tool = 3;
            core.brush_menu_open = false;
            core.brush_menu_pos = None;
            core.status = "Pick: click a spot on the model to sample its color".to_string();
        }
    } else {
        // Fallback: text-only button when the icon failed to load.
        if ui
            .button("Pick from model")
            .on_hover_text("Switches to the Pick tool — click a spot on the model to sample it")
            .clicked()
        {
            core.active_tool = 3;
            core.brush_menu_open = false;
            core.brush_menu_pos = None;
            core.status = "Pick: click a spot on the model to sample its color".to_string();
        }
    }

    ui.separator();

    // Brush footprint shape + optional texture stamp.
    ui.label("Brush type");
    ui.horizontal_wrapped(|ui| {
        for kind in [
            crate::brush::FootprintKind::Round,
            crate::brush::FootprintKind::Square,
            crate::brush::FootprintKind::Diamond,
            crate::brush::FootprintKind::Sprite,
        ] {
            let sel = core.brush.kind == kind;
            if ui.selectable_label(sel, kind.label()).clicked() {
                core.brush.kind = kind;
                core.brush_menu_open = false;
                core.brush_menu_pos = None;
            }
        }
    });

    // Brush build-up: when off, a single stroke's coverage is capped at the
    // brush opacity (overlapping dabs never stack). New strokes layer over old
    // ones normally — the cap lives on a per-stroke alpha mask, dropped with
    // the stroke. When on, dabs build up exactly like a layered airbrush.
    ui.checkbox(&mut core.brush.accumulate, "Accumulate")
        .on_hover_text(
            "Build-up: on = dabs stack within a stroke; off = coverage is capped at \
             the brush opacity, repeated passes over the same texels don't darken",
        );

    if load_brush_png_button(ui, core) {
        core.brush_menu_open = false;
        core.brush_menu_pos = None;
    }
    if core.brush.sprite.is_some() {
        texture_brush_settings_ui(ui, core);
    }
}

/// File-picker button "Load brush PNG…": promotes the brush to a sprite stamp
/// and loads the chosen image. Returns `true` when a sprite was actually loaded
/// (callers close their floating menu on success). Shared by the RMB brush menu
/// and the Brushes panel, so both accept image brushes the same way.
fn load_brush_png_button(ui: &mut Ui, core: &mut Core) -> bool {
    let mut loaded = false;
    ui.horizontal(|ui| {
        if ui.button("Load brush PNG…").clicked() {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("PNG", &["png"])
                .pick_file()
            {
                let path_str = path.to_string_lossy().into_owned();
                match crate::io::brush_sprite(&path_str) {
                    Ok(sprite) => {
                        core.brush.kind = crate::brush::FootprintKind::Sprite;
                        core.brush.sprite = Some(sprite);
                        core.status = format!("Loaded brush sprite: {path_str}");
                        loaded = true;
                    }
                    Err(e) => core.status = format!("Brush load failed: {e}"),
                }
            }
        }
    });
    loaded
}

/// Rotate slider: the brush stores radians (what the stamp math and shader
/// `sin_cos` expect), but a raw −π..=π bound is hostile to point-and-drag.
/// Drive the slider in degrees and translate at the boundary, so the readout
/// shows something a human expects (e.g. 90°, not 1.5708).
fn rotation_slider_ui(ui: &mut Ui, rad_deg: &mut f32) {
    let mut deg = rad_deg.to_degrees();
    if ui
        .add(egui::Slider::new(&mut deg, -180.0..=180.0).suffix("°"))
        .changed()
    {
        *rad_deg = deg.to_radians();
    }
}

/// Sprite/pattern controls shared by the RMB brush menu and the Brushes panel:
/// stamp rotation + flips, texture-size (world repeat) multiplier, the lock
/// toggle, and pattern-lock (world-space sampling). Only called when a sprite
/// is loaded, so the two UIs can never drift apart.
fn texture_brush_settings_ui(ui: &mut Ui, core: &mut Core) {
    ui.horizontal(|ui| {
        rotation_slider_ui(ui, &mut core.brush.rotation);
    });
    ui.horizontal(|ui| {
        ui.checkbox(&mut core.brush.flip_x, "Flip X");
        ui.checkbox(&mut core.brush.flip_y, "Flip Y");
    });
    ui.horizontal(|ui| {
        ui.label("Texture size");
        ui.add(
            egui::Slider::new(&mut core.brush.texture_scale, 0.25..=4.0)
                .logarithmic(true)
                .show_value(true),
        )
        .on_hover_text(
            "World size of one texture repeat: bigger = fewer, larger \
             repeats per dab; smaller = a denser, finer pattern",
        );
    });
    // Lock the pattern's world size so resizing the brush never stretches
    // the texture — the brush radius only changes the paint window. The
    // locked size is captured from the next stroke started after toggling on.
    let mut locked = core.brush.texture_locked;
    if ui
        .checkbox(&mut locked, "Lock texture size")
        .on_hover_text(
            "Keep the pattern at a fixed world size regardless of brush size: \
             resizing the brush grows/shrinks the paint window, not the \
             texture tile",
        )
        .changed()
    {
        if locked {
            core.brush.texture_size_lock = 0.0; // recapture at next stroke start
        }
        core.brush.texture_locked = locked;
    }
    // Pattern-lock: Aligned pins the sprite phase to the stroke-start
    // anchor so a sweep keeps the pattern glued (world/screen-space
    // sampling); Dab re-centers the sprite on every dab (rubber stamp,
    // which smears textures on overlap).
    let mut aligned = core.brush.pattern_lock == crate::brush::PatternLock::Aligned;
    if ui
        .checkbox(&mut aligned, "Pattern-lock (world-space sampling)")
        .on_hover_text(
            "Anchor the sprite phase to the stroke start instead of \
             re-centering it on every dab, so dragging keeps the pattern \
             glued instead of smearing it",
        )
        .changed()
    {
        core.brush.pattern_lock = if aligned {
            crate::brush::PatternLock::Aligned
        } else {
            crate::brush::PatternLock::Dab
        };
    }
    texture_window_ui(ui, core);
}

/// Small tappable color square for the palette grid.
fn swatch_button(ui: &mut Ui, rgb: [u8; 3]) -> egui::Response {
    let size = egui::vec2(20.0, 20.0);
    let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::click());
    if ui.is_rect_visible(rect) {
        let pal = UiPalette::of(ui);
        let radius = egui::CornerRadius::same(RADIUS_CHIP);
        ui.painter().rect_filled(
            rect,
            radius,
            egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]),
        );
        let stroke = if resp.hovered() {
            egui::Stroke::new(2.0, egui::Color32::WHITE)
        } else {
            egui::Stroke::new(1.0, pal.card_border)
        };
        ui.painter()
            .rect_stroke(rect, radius, stroke, egui::StrokeKind::Inside);
        resp.clone()
            .on_hover_text(format!("#{:02X}{:02X}{:02X}", rgb[0], rgb[1], rgb[2]));
    }
    resp
}

/// Draws the pipette icon (lucide, ISC) at the cursor as the Pick-mode
/// pointer. The tip of the pipette (lower-left) sits near the actual click
/// point; the rest extends up-right. Drawn on a dedicated tooltip layer.
fn draw_pick_icon(ctx: &egui::Context, icon: &TextureHandle, cursor: egui::Pos2) {
    let id = egui::Id::new("pick_cursor");
    let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Tooltip, id));
    let size = 22.0;
    let offset = egui::vec2(8.0, 6.0); // shift so tip lines up with cursor
    let rect = egui::Rect::from_center_size(cursor + offset, egui::vec2(size, size));
    painter.image(
        icon.id(),
        rect,
        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
        egui::Color32::WHITE,
    );
}

/// Dim styled label used inside the toolbar pill.
fn toolbar_label(ui: &mut egui::Ui, text: &str) {
    let pal = UiPalette::of(ui);
    ui.label(
        egui::RichText::new(text)
            .color(pal.chrome_text_weak)
            .size(10.5)
            .strong(),
    );
}

fn toolbar_ui(ui: &mut Ui, core: &mut Core) {
    if core.icons.is_none() {
        core.icons = load_icons(ui.ctx());
    }
    let pal = UiPalette::of(ui);
    let theme = pal.chrome_text;
    let dim = ui.visuals().weak_text_color();

    // The toolbar owns a distinct chrome band. Everything here reads from the
    // active palette so it flips with the light/dark theme.
    egui::Frame::NONE
        .fill(pal.chrome)
        .inner_margin(egui::Margin::symmetric(10, 6))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 7.0;
                ui.spacing_mut().slider_width = 78.0;

                let mut do_undo_clicked = false;
                let mut do_redo_clicked = false;
                if let Some(icons) = &core.icons {
                    let can_undo = core.history.can_undo();
                    let can_redo = core.history.can_redo();

                    if icon_button(
                        ui,
                        &icons.undo,
                        17.0,
                        can_undo,
                        if can_undo { theme } else { dim },
                        "Undo (Ctrl+Z)",
                    )
                    .clicked()
                    {
                        do_undo_clicked = true;
                    }

                    if icon_button(
                        ui,
                        &icons.redo,
                        17.0,
                        can_redo,
                        if can_redo { theme } else { dim },
                        "Redo (Ctrl+Y / Ctrl+Shift+Z)",
                    )
                    .clicked()
                    {
                        do_redo_clicked = true;
                    }

                    ui.add(egui::Separator::default().vertical().spacing(10.0));
                }

                if do_undo_clicked {
                    undo_action(core);
                }
                if do_redo_clicked {
                    redo_action(core);
                }

                let mut color = core.brush.color;
                if ui
                    .color_edit_button_srgba_unmultiplied(&mut color)
                    .on_hover_text("Brush color — right-click the viewport for a picker & presets")
                    .changed()
                {
                    core.brush.color = color;
                }

                ui.add(egui::Separator::default().vertical().spacing(10.0));

                toolbar_label(ui, "SIZE");
                ui.add(
                    egui::Slider::new(&mut core.brush.size, 1.0..=300.0)
                        .suffix("px")
                        .logarithmic(true)
                        .max_decimals(0),
                );

                toolbar_label(ui, "HARDNESS");
                ui.add(egui::Slider::new(&mut core.brush.hardness, 0.0..=1.0)
                    .custom_formatter(|v, _| format!("{:.0}%", v * 100.0)))
                    .on_hover_text(
                        "Fraction of the radius at full strength; it fades to the edge beyond that. 100% = hard edge.",
                    );

                toolbar_label(ui, "OPACITY");
                ui.add(egui::Slider::new(&mut core.brush.opacity, 0.0..=1.0)
                    .custom_formatter(|v, _| format!("{:.0}%", v * 100.0)));

                toolbar_label(ui, "SPACING");
                ui.add(
                    egui::Slider::new(&mut core.brush.spacing, 0.0..=200.0)
                        .suffix("px")
                        .step_by(1.0)
                        .max_decimals(0),
                )
                .on_hover_text("0 = continuous (dabs overlap, tuned to brush size). Positive = fixed distance (px) between dabs along a stroke.");
            });
        });
}

/// Layer material panel. Surface parameters (roughness, metallic, emissive,
/// AO) are per-layer sliders: they shape the shading wherever the layer covers
/// the model, following the same source-over stacking as the albedo.
fn channels_ui(ui: &mut Ui, core: &mut Core) {
    panel_heading(ui, "Material");
    let Some(mesh) = core.mesh.as_mut() else {
        let pal = UiPalette::of(ui);
        ui.add_space(24.0);
        ui.vertical_centered(|ui| {
            ui.label(
                egui::RichText::new("No model loaded")
                    .size(13.0)
                    .color(pal.chrome_text_weak),
            );
            ui.add_space(4.0);
            ui.label(
                egui::RichText::new("File → Open to load a .gltf or .glb")
                    .size(11.0)
                    .color(pal.card_border_active),
            );
        });
        return;
    };
    if mesh.layers.is_empty() {
        ui.add_space(24.0);
        ui.vertical_centered(|ui| {
            ui.label(
                egui::RichText::new("No layers yet")
                    .size(13.0)
                    .color(UiPalette::of(ui).chrome_text_weak),
            );
            ui.add_space(4.0);
            ui.label(
                egui::RichText::new("Add a layer to set its material.")
                    .size(11.0)
                    .color(UiPalette::of(ui).card_border_active),
            );
        });
        return;
    }
    ui.spacing_mut().slider_width = 132.0;
    let li = mesh.active_layer.min(mesh.layers.len() - 1);

    // Two-phase slider block: read the layer's surface into locals, preview the
    // edits live, then write back once. Recording the undo snapshot happens
    // *after* the block so it never fights the mutable borrow of `mesh.layers`,
    // and it fires exactly once per interaction (drag start or preset click).
    let old_surface: (f32, f32, f32, f32, f32, f32, f32, f32, f32, [f32; 3]) = {
        let l = &mesh.layers[li];
        // Active-layer badge instead of a plain "Layer: …" label, so it is
        // impossible to miss which layer the sliders edit.
        let pal = UiPalette::of(ui);
        let badge = egui::Frame::new()
            .fill(pal.card_active)
            .corner_radius(egui::CornerRadius::same(RADIUS_CHIP))
            .inner_margin(egui::Margin::symmetric(8, 3));
        badge.show(ui, |ui| {
            ui.label(
                egui::RichText::new(&l.name)
                    .strong()
                    .size(12.0)
                    .color(ACCENT),
            );
        });
        (
            l.roughness,
            l.metallic,
            l.emissive,
            l.ambient_occlusion,
            l.height,
            l.bump_strength,
            l.clearcoat,
            l.clearcoat_roughness,
            l.specular_ior,
            l.emissive_color,
        )
    };
    let locked = mesh.layers[li].locked;
    let mut surface = old_surface;
    let mut interaction_started = false;

    if locked {
        let i_lock = {
            if core.icons.is_none() {
                core.icons = load_icons(ui.ctx());
            }
            core.icons.as_ref().unwrap().lock.clone()
        };
        ui.horizontal(|ui| {
            ui.add(
                egui::Image::new(&i_lock)
                    .fit_to_exact_size(egui::vec2(14.0, 14.0))
                    .tint(ui.visuals().text_color()),
            );
            ui.label("Locked — unlock to edit material.");
        });
    }

    let slider =
        |ui: &mut Ui, value: &mut f32, range: std::ops::RangeInclusive<f32>, text: &str| {
            let mut s = egui::Slider::new(value, range).text(text);
            if text == "Roughness" || text == "Clearcoat roughness" {
                s = s.logarithmic(true);
            }
            ui.add_enabled(!locked, s).drag_started()
        };

    egui::CollapsingHeader::new("Surface")
        .default_open(true)
        .show(ui, |ui| {
            interaction_started |= slider(ui, &mut surface.0, 0.03..=1.0, "Roughness");
            interaction_started |= slider(ui, &mut surface.1, 0.0..=1.0, "Metallic");
            interaction_started |= slider(ui, &mut surface.2, 0.0..=3.0, "Emissive glow");
            interaction_started |= slider(ui, &mut surface.3, 0.0..=1.0, "Ambient occlusion");
            ui.horizontal(|ui| {
                ui.label("Emissive color");
                if !locked {
                    let col = surface.9;
                    let mut emc = egui::Color32::from_rgb(
                        (col[0].clamp(0.0, 1.0) * 255.0).round() as u8,
                        (col[1].clamp(0.0, 1.0) * 255.0).round() as u8,
                        (col[2].clamp(0.0, 1.0) * 255.0).round() as u8,
                    );
                    if ui.color_edit_button_srgba(&mut emc).changed() {
                        interaction_started = true;
                        surface.9 = [
                            emc.r() as f32 / 255.0,
                            emc.g() as f32 / 255.0,
                            emc.b() as f32 / 255.0,
                        ];
                        core.status = "Emissive color changed".to_string();
                    } else {
                        let swatch = egui::Frame::default()
                            .fill(emc)
                            .corner_radius(3.0)
                            .inner_margin(egui::Margin::same(8));
                        swatch.show(ui, |ui| {
                            ui.add_space(0.0);
                        });
                    }
                } else {
                    let col = surface.9;
                    let swatch = egui::Frame::default()
                        .fill(egui::Color32::from_rgb(
                            (col[0].clamp(0.0, 1.0) * 255.0).round() as u8,
                            (col[1].clamp(0.0, 1.0) * 255.0).round() as u8,
                            (col[2].clamp(0.0, 1.0) * 255.0).round() as u8,
                        ))
                        .corner_radius(3.0)
                        .inner_margin(egui::Margin::same(8));
                    swatch.show(ui, |ui| {
                        ui.add_space(0.0);
                    });
                }
            });
        });
    egui::CollapsingHeader::new("Advanced")
        .default_open(false)
        .show(ui, |ui| {
            interaction_started |= slider(ui, &mut surface.4, -1.0..=1.0, "Height");
            interaction_started |= slider(ui, &mut surface.5, 0.0..=8.0, "Bump strength");
            interaction_started |= slider(ui, &mut surface.6, 0.0..=1.0, "Clearcoat");
            interaction_started |= slider(ui, &mut surface.7, 0.1..=1.0, "Clearcoat roughness");
            interaction_started |= slider(ui, &mut surface.8, 1.0..=2.5, "Specular IOR");
        });

    // Presets as a 2×2 chip grid (a material-tint dot + name), replacing the
    // horizontally-wrapped selectable labels that wrap awkwardly in panels.
    ui.add_space(6.0);
    let pal = UiPalette::of(ui);
    ui.label(
        egui::RichText::new("PRESETS")
            .size(10.0)
            .color(pal.chrome_text_weak)
            .strong(),
    );
    egui::Grid::new("material_presets")
        .num_columns(2)
        .spacing([6.0, 4.0])
        .show(ui, |ui| {
            let presets = [
                (
                    "Clay",
                    egui::Color32::from_rgb(193, 127, 90),
                    (0.85, 0.0, 0.0, 1.0),
                ),
                (
                    "Glossy",
                    egui::Color32::from_rgb(255, 255, 255),
                    (0.18, 0.0, 0.0, 1.0),
                ),
                (
                    "Brushed metal",
                    egui::Color32::from_rgb(200, 208, 216),
                    (0.35, 1.0, 0.0, 1.0),
                ),
                (
                    "Cold metal",
                    egui::Color32::from_rgb(184, 200, 216),
                    (0.25, 1.0, 0.1, 1.0),
                ),
            ];
            for (i, (name, dot_color, (rough, metal, emiss, ao))) in presets.iter().enumerate() {
                let selected =
                    (surface.0, surface.1, surface.2, surface.3) == (*rough, *metal, *emiss, *ao);
                let chip = egui::Frame::new()
                    .fill(if selected { pal.card_active } else { pal.card })
                    .stroke(egui::Stroke::new(
                        1.0,
                        if selected {
                            pal.card_border_active
                        } else {
                            pal.card_border
                        },
                    ))
                    .corner_radius(egui::CornerRadius::same(RADIUS_CHIP))
                    .inner_margin(egui::Margin::symmetric(8, 4));
                let clicked = chip
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            let (rect, _) =
                                ui.allocate_exact_size(egui::vec2(8.0, 8.0), egui::Sense::hover());
                            ui.painter().circle_filled(rect.center(), 3.5, *dot_color);
                            ui.add_space(2.0);
                            ui.label(if selected {
                                egui::RichText::new(*name).strong()
                            } else {
                                egui::RichText::new(*name)
                            });
                        });
                    })
                    .response
                    .interact(egui::Sense::click())
                    .clicked();
                if clicked && !locked {
                    interaction_started = true;
                    surface.0 = *rough;
                    surface.1 = *metal;
                    surface.2 = *emiss;
                    surface.3 = *ao;
                    core.status = format!("Material preset: {name}");
                }
                if i % 2 == 1 {
                    ui.end_row();
                }
            }
        });

    // Snapshot *before* any write-back so an interaction always restores to the
    // true pre-edit surface (even if the drags' first nudge already landed in
    // `surface`, the mesh hasn't been touched yet).
    if interaction_started {
        core.history.record(snapshot_of(mesh));
    }
    if surface != old_surface {
        mesh.layers[li].roughness = surface.0;
        mesh.layers[li].metallic = surface.1;
        mesh.layers[li].emissive = surface.2;
        mesh.layers[li].ambient_occlusion = surface.3;
        mesh.layers[li].height = surface.4;
        mesh.layers[li].bump_strength = surface.5;
        mesh.layers[li].clearcoat = surface.6;
        mesh.layers[li].clearcoat_roughness = surface.7;
        mesh.layers[li].specular_ior = surface.8;
        mesh.layers[li].emissive_color = surface.9;
        core.needs_material_upload = true;
    }
}

/// Viewport-wide lighting. Sun on/off + direction + color drive the direct key
/// light; the environment (analytic sky or a loaded skybox map) lights diffuse
/// and specular reflections and shows as the background. Turning the sun off
/// leaves the skybox as the sole light source.
fn lighting_ui(ui: &mut Ui, core: &mut Core) {
    panel_heading(ui, "Lighting");
    ui.spacing_mut().slider_width = 150.0;
    let material = &mut core.material;
    let pal = UiPalette::of(ui);

    let sun_was = (
        material.sun_enabled,
        material.sun_elevation,
        material.sun_azimuth,
    );
    let env_was = (
        material.env_intensity,
        material.env_rotation,
        material.sky_color,
    );

    let card = egui::Frame::new()
        .fill(pal.card)
        .corner_radius(egui::CornerRadius::same(RADIUS_CARD))
        .inner_margin(egui::Margin::same(10));

    // --- Sun card ---
    card.show(ui, |ui| {
        // Header row: accent tick + bold "Sun" with the on/off toggle pinned to
        // the right edge (replaces the verbose "Sun (directional key light)").
        ui.horizontal(|ui| {
            let (rect, _) = ui.allocate_exact_size(egui::vec2(3.0, 15.0), egui::Sense::hover());
            ui.painter().rect_filled(
                rect,
                RADIUS_CHIP,
                if material.sun_enabled {
                    ACCENT
                } else {
                    pal.card_border
                },
            );
            ui.add_space(2.0);
            ui.label(
                egui::RichText::new("Sun")
                    .size(13.0)
                    .strong()
                    .color(pal.chrome_text),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.checkbox(&mut material.sun_enabled, "")
                    .on_hover_text("Turn off so the environment/skybox alone lights the scene.");
            });
        });
        if !material.sun_enabled {
            ui.add_space(2.0);
            ui.label(
                egui::RichText::new("off — skybox only")
                    .small()
                    .color(pal.chrome_text_weak),
            );
        }
        ui.add_enabled(
            material.sun_enabled,
            egui::Slider::new(&mut material.sun_elevation, -89.9..=89.9).text("Sun elevation"),
        );
        ui.add_enabled(
            material.sun_enabled,
            egui::Slider::new(&mut material.sun_azimuth, 0.0..=360.0).text("Sun azimuth"),
        );
        ui.add_enabled(
            material.sun_enabled,
            egui::Slider::new(&mut material.sun_intensity, 0.0..=8.0).text("Sun intensity"),
        );
        if material.sun_enabled {
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                ui.label("Sun color");
                if ui.color_edit_button_rgb(&mut material.sun_color).changed() {
                    core.status = "Sun color changed".to_string();
                }
            });
        }
    });
    ui.add_space(6.0);

    // --- Environment card ---
    card.show(ui, |ui| {
        let env_label = match &core.env_path {
            Some(p) => format!(
                "Skybox: {}",
                p.rsplit(['/', '\\']).next().unwrap_or(p.as_str())
            ),
            None => "Skybox: analytic (color) sky".to_string(),
        };
        ui.label(
            egui::RichText::new(env_label)
                .strong()
                .color(pal.chrome_text),
        );
        ui.add_space(2.0);
        ui.horizontal(|ui| {
            ui.label("Sky color");
            if ui.color_edit_button_rgb(&mut material.sky_color).changed() {
                core.status = "Sky color changed".to_string();
            }
        });
        ui.add(egui::Slider::new(&mut material.env_intensity, 0.0..=2.0).text("Sky light"));
        ui.add(egui::Slider::new(&mut material.env_rotation, 0.0..=360.0).text("Skybox rotation"))
            .on_hover_text("Rotates the loaded environment map around the vertical axis.");
        ui.add(egui::Slider::new(&mut material.exposure, 0.1..=4.0).text("Exposure"));
        ui.add(egui::Slider::new(&mut material.parallax, 0.0..=0.1).text("Height relief"))
            .on_hover_text(
                "Viewport-only parallax: offsets the shading by the painted height map \
                 along the view ray. 0 flattens relief; ~0.02-0.06 gives depth without \
                 smearing.",
            );
        ui.add_enabled(
            material.sun_enabled,
            egui::Slider::new(&mut material.fill_intensity, 0.0..=1.5).text("Interior fill"),
        )
        .on_hover_text(
            "Camera-direction fill light that keeps shadow interiors readable. Tied to the sun: \
             disabled while the sun is off, so that mode is pure skybox lighting.",
        );
    });

    if (sun_was.0, sun_was.1, sun_was.2)
        != (
            material.sun_enabled,
            material.sun_elevation,
            material.sun_azimuth,
        )
    {
        core.status = "Sun updated".to_string();
    }
    if env_was
        != (
            material.env_intensity,
            material.env_rotation,
            material.sky_color,
        )
    {
        core.status = "Environment updated".to_string();
    }
}

/// Live preview for the brush settings: the current stamp drawn on a
/// checkerboard — footprint shape + color/opacity, and for texture brushes the
/// sprite repeated at `texture_scale` with rotation and flips applied, masked
/// to the round dab window (exactly the coverage the stamp uses: sprite alpha
/// masked, painted in the brush tint). Rasterized off-screen and cached, so it
/// re-renders only when the brush actually changes.
fn brush_preview_ui(ui: &mut Ui, core: &mut Core) {
    const RASTER: u32 = 256;
    const SHOWN: f32 = 72.0;
    const CARD_H: f32 = 94.0;
    let pal = UiPalette::of(ui);
    let size = egui::vec2(SHOWN + 40.0, CARD_H);
    let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::hover());
    if !ui.is_rect_visible(rect) {
        return;
    }
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, egui::CornerRadius::same(RADIUS_CARD), pal.card);
    painter.rect_stroke(
        rect,
        egui::CornerRadius::same(RADIUS_CARD),
        egui::Stroke::new(1.0, pal.card_border),
        egui::StrokeKind::Inside,
    );

    let img_rect = egui::Rect::from_center_size(
        egui::pos2(rect.center().x, rect.top() + SHOWN * 0.5 + 6.0),
        egui::vec2(SHOWN, SHOWN),
    );
    let sig = settings_preview_sig(core);
    let stale = core
        .settings_preview
        .as_ref()
        .map(|(s, _)| *s != sig)
        .unwrap_or(true);
    if stale {
        let image = render_settings_preview(core, RASTER, None, true);
        let handle = ui.ctx().load_texture(
            format!("settings_preview_{sig:016x}"),
            image,
            egui::TextureOptions::LINEAR,
        );
        core.settings_preview = Some((sig, handle));
    }
    let handle = core.settings_preview.as_ref().unwrap().1.clone();
    painter.image(
        handle.id(),
        img_rect,
        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
        egui::Color32::WHITE,
    );

    let caption = match &core.brush.sprite {
        Some(s) => format!(
            "{} × {}  ·  {:.0}°",
            s.width,
            s.height,
            core.brush.rotation.to_degrees().rem_euclid(360.0),
        ),
        None => match core.brush.kind {
            crate::brush::FootprintKind::Round => "Round dab".to_string(),
            crate::brush::FootprintKind::Square => "Square dab".to_string(),
            crate::brush::FootprintKind::Diamond => "Diamond dab".to_string(),
            crate::brush::FootprintKind::Rect => "Rect tool".to_string(),
            crate::brush::FootprintKind::Sprite => "Texture dab".to_string(),
        },
    };
    painter.text(
        egui::pos2(rect.center().x, rect.bottom() - 12.0),
        egui::Align2::CENTER_CENTER,
        caption,
        egui::FontId::proportional(11.0),
        pal.chrome_text,
    );
    resp.on_hover_text(format!(
        "Live stamp preview — {}",
        match &core.brush.sprite {
            Some(_) => {
                "repeat size, rotation, flips and the frame (mask) shape follow \
                 the texture brush settings"
            }
            None => "footprint shape and color follow the brush",
        }
    ));
}

/// Numeric code for the texture paint-window shape (0 round, 1 square, 2
/// diamond), shared by the sig, the preview and the GPU cursor overlay.
fn window_code(win: crate::brush::Window) -> u32 {
    match win {
        crate::brush::Window::Round | crate::brush::Window::SpriteBounds => 0,
        crate::brush::Window::Square => 1,
        crate::brush::Window::Diamond => 2,
    }
}

/// The paint-window falloff, mirroring `brush::Window::mask`: `1.0` across the
/// flat core (up to half the radius), then a C¹ smoothstep skirt to `0.0` at
/// the rim (`t >= 1`). The preview applies this to the same normalized
/// `t` (`shape distance / radius`) the stamp uses, so the square/diamond
/// frames read exactly as painted.
fn window_mask(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    if t <= 0.5 {
        1.0
    } else {
        let x = (t - 0.5) * 2.0;
        1.0 - x * x * (3.0 - 2.0 * x)
    }
}

/// Fingerprint of everything that changes the settings preview's look.
fn settings_preview_sig(core: &Core) -> u64 {
    let b = &core.brush;
    let mut h = 0x6a09_e667_f3bc_c909u64;
    h = h.rotate_left(17) ^ (b.kind as u8 as u64);
    h = h.rotate_left(17) ^ b.rotation.to_bits() as u64;
    h = h.rotate_left(17) ^ b.texture_scale.to_bits() as u64;
    h = h.rotate_left(17) ^ b.opacity.to_bits() as u64;
    h = h.rotate_left(17) ^ window_code(b.texture_window) as u64;
    // Combine the two flip flags and the four RGBA bytes into disjoint bit
    // ranges and xor them in at once, so every field change is guaranteed to
    // alter the signature (naively mixing `^` and `|` leaves a field's bits at
    // the mercy of whatever the accumulator already holds).
    let flips = (b.flip_x as u64) | ((b.flip_y as u64) << 1);
    let rgba = ((b.color[3] as u64) << 8)
        | ((b.color[0] as u64) << 16)
        | ((b.color[1] as u64) << 24)
        | ((b.color[2] as u64) << 32);
    h = h.rotate_left(17) ^ flips ^ rgba;
    if let Some(s) = &b.sprite {
        h = h.rotate_left(17) ^ sprite_sig(s);
    }
    h
}

/// Rasterizes the current brush into `px`×`px`. Geometric footprints paint
/// their signed-distance rim in the brush tint at `opacity`; texture brushes
/// sample a tiling of the sprite (rotation + flips per tile) masked by the
/// paint window (`texture_window`) — the sprite's alpha is the coverage and
/// the tint color the paint, mirroring how a stamp lands. The mask uses the
/// same normalized distance + [`window_mask`] falloff as `brush::Window::mask`,
/// so square and diamond frames read their exact painted shapes.
///
/// `dab_radius` (canvas px) makes the frame fill the image and sizes the tile
/// cell from the sprite-to-dab mapping the stamp uses (`sw/m · 2r`), so the
/// 2D canvas cursor shows the stamp texel-for-texel. `None` uses the panel's
/// fixed `CELL_BASE × texture_scale` repeat. `checker` draws the transparency
/// checkerboard (panel) or leaves texels transparent (cursor overlay).
fn render_settings_preview(
    core: &Core,
    px: u32,
    dab_radius: Option<f32>,
    checker: bool,
) -> egui::ColorImage {
    let b = &core.brush;
    let n = px as f32;
    let r = match dab_radius {
        Some(_) => n * 0.5,
        None => n * 0.30,
    };
    let tint = b.color;
    let fa = ((tint[3] as f32 / 255.0) * b.opacity.clamp(0.0, 1.0)).clamp(0.06, 1.0);
    let (tw, th) = match &b.sprite {
        Some(s) => (s.width.max(1) as usize, s.height.max(1) as usize),
        None => (1, 1),
    };
    let m = (tw.max(th) as f32).max(1.0);
    let (cellx, celly) = match dab_radius {
        // The stamp maps the sprite's longest side to the dab diameter (`2r`),
        // so in image px (drawn at `2·dab_radius`) the tile is `sw/m · n`.
        Some(_) => (tw as f32 / m * n, th as f32 / m * n),
        None => {
            let cell = (CELL_BASE * b.texture_scale.clamp(0.1, 6.0)).max(10.0);
            (cell, cell)
        }
    };
    let (sr, cr) = b.rotation.sin_cos();
    let mut out = Vec::with_capacity((px * px) as usize);
    for y in 0..px as usize {
        for x in 0..px as usize {
            let dx = x as f32 + 0.5 - n * 0.5;
            let dy = y as f32 + 0.5 - n * 0.5;
            // Normalized distance to the footprint rim in dab radii (t = 1 at
            // the rim). Geometric brushes shape by kind; texture brushes shape
            // by the paint-window mask (`texture_window`), matching
            // `brush::Window::mask` exactly (square: inscribed square, diamond:
            // inscribed diamond with vertices on the axes).
            let t = match b.kind {
                crate::brush::FootprintKind::Round => (dx * dx + dy * dy).sqrt() / r,
                crate::brush::FootprintKind::Diamond => (dx.abs() + dy.abs()) / r,
                crate::brush::FootprintKind::Square | crate::brush::FootprintKind::Rect => {
                    (dx.abs() / r).max(dy.abs() / r)
                }
                crate::brush::FootprintKind::Sprite => match b.texture_window {
                    crate::brush::Window::Round | crate::brush::Window::SpriteBounds => {
                        (dx * dx + dy * dy).sqrt() / r
                    }
                    crate::brush::Window::Square => (dx.abs() / r).max(dy.abs() / r),
                    crate::brush::Window::Diamond => (dx.abs() + dy.abs()) / r,
                },
            };
            let cov = window_mask(t);
            // Checkerboard backdrop.
            let cb = if checker {
                if ((x / 8) + (y / 8)) % 2 == 0 {
                    [232, 232, 232]
                } else {
                    [208, 208, 208]
                }
            } else {
                [0, 0, 0]
            };
            let (fc, alpha) = if let Some(sp) = &b.sprite {
                let (sx, sy, sz, sa) = if dab_radius.is_some() {
                    // Cursor / stamp mode: FIT ONE SPRITE inside the painted
                    // dab (rubber-stamp style), so the preview reads as the
                    // single framed brush — never a repeated tile. The sprite
                    // box is the same aspect-correct fit the brush paints
                    // (longest side = the dab, rotated + flipped), and
                    // whatever falls outside the box is transparent, so the
                    // frame mask alone shapes the visible stamp.
                    let hw = r * tw as f32 / m;
                    let hh = r * th as f32 / m;
                    // Inverse sprite↔screen rotation (matches the painted
                    // stamp: sprite-local (x,y) → screen (x·cr − y·sr, …)),
                    // then flips, then the boxed fit test.
                    let x_local = dx * cr + dy * sr;
                    let y_local = -dx * sr + dy * cr;
                    let mut su = x_local / hw;
                    let mut sv = y_local / hh;
                    if b.flip_x {
                        su = -su;
                    }
                    if b.flip_y {
                        sv = -sv;
                    }
                    if su.abs() > 1.0 || sv.abs() > 1.0 {
                        (0u8, 0u8, 0u8, 0.0f32)
                    } else {
                        let u = ((su * 0.5 + 0.5) * tw as f32).floor().max(0.0) as usize % tw;
                        let v = ((sv * 0.5 + 0.5) * th as f32).floor().max(0.0) as usize % th;
                        let idx = (v * tw + u) * 4;
                        let a = sp.rgba[idx + 3] as f32 / 255.0;
                        (sp.rgba[idx], sp.rgba[idx + 1], sp.rgba[idx + 2], a)
                    }
                } else {
                    // Panel mode: show the tiled repeat (world-size) look.
                    // Tile the plane: local coords in the cell around the dab,
                    // rotated + flipped into sprite space, nearest-texel read.
                    let lx = (dx + cellx * 0.5).rem_euclid(cellx) - cellx * 0.5;
                    let ly = (dy + celly * 0.5).rem_euclid(celly) - celly * 0.5;
                    let (mut rx, mut ry) = (lx * cr - ly * sr, lx * sr + ly * cr);
                    if b.flip_x {
                        rx = -rx;
                    }
                    if b.flip_y {
                        ry = -ry;
                    }
                    let su =
                        ((rx + cellx * 0.5) / cellx * tw as f32).floor().max(0.0) as usize % tw;
                    let sv =
                        ((ry + celly * 0.5) / celly * th as f32).floor().max(0.0) as usize % th;
                    let idx = (sv * tw + su) * 4;
                    // Sprite alpha = coverage; sprite rgb modulates the tint.
                    let a = sp.rgba[idx + 3] as f32 / 255.0;
                    (sp.rgba[idx], sp.rgba[idx + 1], sp.rgba[idx + 2], a)
                };
                (
                    [
                        (sx as f32 * tint[0] as f32 / 255.0) as u8,
                        (sy as f32 * tint[1] as f32 / 255.0) as u8,
                        (sz as f32 * tint[2] as f32 / 255.0) as u8,
                    ],
                    cov * fa * sa,
                )
            } else {
                ([tint[0], tint[1], tint[2]], cov * fa)
            };
            let a = alpha.clamp(0.0, 1.0);
            if checker {
                let r8 = (cb[0] as f32 + (fc[0] as f32 - cb[0] as f32) * a) as u8;
                let g8 = (cb[1] as f32 + (fc[1] as f32 - cb[1] as f32) * a) as u8;
                let b8 = (cb[2] as f32 + (fc[2] as f32 - cb[2] as f32) * a) as u8;
                out.push(egui::Color32::from_rgba_unmultiplied(r8, g8, b8, 255));
            } else {
                out.push(egui::Color32::from_rgba_unmultiplied(
                    fc[0],
                    fc[1],
                    fc[2],
                    (a * 255.0).round() as u8,
                ));
            }
        }
    }
    egui::ColorImage::new([px as usize, px as usize], out)
}

/// Cached WYSIWYG dab texture for the floating cursors and the T-bar brush
/// button: the exact stamp the current brush paints, rasterized at `dab_px`
/// so the frame fills the image (see `render_settings_preview`). Reuses
/// `settings_preview_flat`, keyed on the settings signature plus the radius;
/// the 3D viewport cursor, 2D canvas cursor, and toolbar icon all draw the
/// same masked dab so none of them lingers on the raw round sprite.
fn brush_cursor_preview(ctx: &egui::Context, core: &mut Core, dab_px: f32) -> egui::TextureId {
    const RASTER: u32 = 128;
    let sig = settings_preview_sig(core);
    let rkey = (dab_px.clamp(1.0, 512.0) * 2.0) as u32;
    let stale = core
        .settings_preview_flat
        .as_ref()
        .map(|(s, r, _)| *s != sig || *r != rkey)
        .unwrap_or(true);
    if stale {
        let image = render_settings_preview(core, RASTER, Some(dab_px), false);
        let handle = ctx.load_texture(
            format!("settings_preview_flat_{sig:016x}_{rkey}"),
            image,
            egui::TextureOptions::LINEAR,
        );
        core.settings_preview_flat = Some((sig, rkey, handle));
    }
    core.settings_preview_flat.as_ref().unwrap().2.id()
}

/// Paint-window ("frame") selector for texture brushes: the shape that frames
/// a pattern-locked texture dab (round default, square or diamond). Shown as
/// shape glyphs (no labels to read — the shape IS the label). Shared by the
/// Brushes-panel settings strip and the RMB texture menu so the two never
/// drift apart.
fn texture_window_ui(ui: &mut Ui, core: &mut Core) {
    ui.horizontal(|ui| {
        ui.label("Frame");
        ui.selectable_value(
            &mut core.brush.texture_window,
            crate::brush::Window::Round,
            "Round",
        )
        .on_hover_text("Round frame: the texture dab paints through a soft disc");
        ui.selectable_value(
            &mut core.brush.texture_window,
            crate::brush::Window::Square,
            "Square",
        )
        .on_hover_text("Square frame: the texture dab paints through a soft square");
        ui.selectable_value(
            &mut core.brush.texture_window,
            crate::brush::Window::Diamond,
            "Diamond",
        )
        .on_hover_text("Diamond frame: the texture dab paints through a soft diamond");
    });
}

/// Base repeat-cell size (in preview px at `texture_scale = 1`) for the
/// texture-brush preview.
const CELL_BASE: f32 = 46.0;

/// Compact brush settings strip for the Brushes panel, shown above the gallery.
/// Only the texture-behavior settings live here — texture size (world repeat),
/// its lock, and pattern-lock, plus the stamp transform when a texture brush
/// has a sprite. The toolbar / RMB menu already carry color, size, hardness,
/// opacity and spacing, so those aren't duplicated here.
fn brush_settings_ui(ui: &mut Ui, core: &mut Core) {
    ui.set_width(ui.available_width());
    ui.spacing_mut().slider_width = 88.0;
    panel_heading(ui, "Brush");

    egui::Grid::new("brush_tex")
        .num_columns(2)
        .spacing([8.0, 4.0])
        .show(ui, |ui| {
            ui.label("Texture size");
            ui.add(
                egui::Slider::new(&mut core.brush.texture_scale, 0.25..=4.0)
                    .logarithmic(true)
                    .show_value(true),
            )
            .on_hover_text(
                "World size of one texture repeat: bigger = fewer, larger \
                 repeats per dab; smaller = a denser, finer pattern",
            );
            ui.end_row();

            // Lock the pattern's world size so resizing the brush never
            // stretches the texture — the brush radius only changes the paint
            // window. The locked size is captured from the next stroke started
            // after toggling on.
            ui.label("Lock size");
            let mut locked = core.brush.texture_locked;
            let lock_changed = ui
                .checkbox(&mut locked, "")
                .on_hover_text(
                    "Keep the pattern at a fixed world size regardless of brush size: \
                     resizing the brush grows/shrinks the paint window, not the \
                     texture tile",
                )
                .changed();
            if lock_changed {
                if locked {
                    core.brush.texture_size_lock = 0.0; // recapture at next stroke start
                }
                core.brush.texture_locked = locked;
            }
            ui.end_row();

            // Pattern-lock: Aligned pins the sprite phase to the stroke-start
            // anchor so a sweep keeps the pattern glued (world/screen-space
            // sampling); Dab re-centers the sprite on every dab (rubber stamp,
            // which smears textures on overlap).
            ui.label("Pattern-lock");
            let mut aligned = core.brush.pattern_lock == crate::brush::PatternLock::Aligned;
            if ui
                .checkbox(&mut aligned, "")
                .on_hover_text(
                    "Anchor the sprite phase to the stroke start instead of \
                     re-centering it on every dab, so dragging keeps the pattern \
                     glued instead of smearing it",
                )
                .changed()
            {
                core.brush.pattern_lock = if aligned {
                    crate::brush::PatternLock::Aligned
                } else {
                    crate::brush::PatternLock::Dab
                };
            }
            ui.end_row();
        });

    texture_window_ui(ui, core);

    // Stamp transform + load button for texture brushes (or a Texture brush
    // that still lacks an image, which phases in the load button).
    if core.brush.sprite.is_some() || core.brush.kind == crate::brush::FootprintKind::Sprite {
        ui.horizontal_wrapped(|ui| {
            let _ = load_brush_png_button(ui, core);
            ui.toggle_value(&mut core.brush.flip_x, "↔")
                .on_hover_text("Flip sprite horizontally");
            ui.toggle_value(&mut core.brush.flip_y, "↕")
                .on_hover_text("Flip sprite vertically");
            rotation_slider_ui(ui, &mut core.brush.rotation);
        });
    }
}

/// Browsable brush library tab. Lists the current brush, the folder it reads
/// from, a category filter (subfolders become categories), a compact settings
/// strip ([`brush_settings_ui`]), and a grid of thumbnails.
/// Click a brush to make it the active brush.
fn brushes_ui(ui: &mut Ui, core: &mut Core) {
    // Cheap-ish folder walk (throttled inside) so dropping a new file onto the
    // folder shows up while this panel is on screen.
    core.brushes.refresh();

    ui.horizontal(|ui| {
        panel_heading(ui, "Brushes");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui
                .button("Rescan")
                .on_hover_text("Re-read the brush folder now")
                .clicked()
            {
                core.brushes.force_refresh();
                ui.ctx().request_repaint();
            }
        });
    });
    ui.label(
        egui::RichText::new(core.brushes.folder.display().to_string())
            .small()
            .weak(),
    )
    .on_hover_text("The folder PixForge reads brushes from: PIXFORGE_BRUSHES, else ./brushes");
    ui.label(
        egui::RichText::new("Drop PNG/GBR brush files here — subfolders become categories.")
            .small()
            .weak(),
    );
    if let Some(err) = core.brushes.last_error.clone() {
        let warn = UiPalette::of(ui).warn;
        ui.colored_label(warn, egui::RichText::new(format!("Skipped: {err}")).small());
    }

    if let Some(sel) = core.brushes.selected {
        if let Some(entry) = core.brushes.entries.get(sel) {
            // The live preview sits next to the active brush. Copy out the
            // strings first: the preview borrows `core` mutably (its texture
            // cache), which must not overlap the immutable brush entry borrow.
            let name = entry.name.clone();
            let category = entry.category.clone();
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.label("Active:");
                        ui.strong(name);
                    });
                    ui.label(
                        egui::RichText::new(format!(
                            "{} px · {}",
                            core.brush.size as u32, category
                        ))
                        .small()
                        .weak(),
                    );
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::TOP), |ui| {
                    brush_preview_ui(ui, core);
                });
            });
        }
    }
    ui.separator();

    // Compact settings strip sits directly under the active brush, so whatever
    // is being tweaked is never hidden below a filter bar.
    brush_settings_ui(ui, core);
    ui.separator();

    // Category filter pills live between the settings and the gallery.
    let mut filter = core.brush_filter.clone();
    let cats = core.brushes.categories();
    ui.horizontal_wrapped(|ui| {
        let all_sel = filter == "All";
        if ui.selectable_label(all_sel, "All").clicked() {
            filter = "All".to_string();
        }
        for cat in &cats {
            let sel = filter == *cat;
            if ui.selectable_label(sel, cat).clicked() {
                filter = cat.clone();
            }
        }
    });
    core.brush_filter = filter;
    ui.separator();

    let indices: Vec<usize> = core
        .brushes
        .entries
        .iter()
        .enumerate()
        .filter(|(_, e)| core.brush_filter == "All" || e.category == core.brush_filter)
        .map(|(i, _)| i)
        .collect();

    // Rebuild thumbnail textures when the entry list changed.
    let sig = core.brushes.signature().to_string();
    if sig != core.brush_thumb_sig {
        core.brush_thumbs.clear();
        core.brush_thumb_sig = sig;
    }
    for &i in &indices {
        if !core.brush_thumbs.contains_key(&i) {
            let img = brush_thumb_image(&core.brushes.entries[i].sprite);
            core.brush_thumbs.insert(
                i,
                ui.ctx().load_texture(
                    format!("brush_thumb_{i}"),
                    img,
                    egui::TextureOptions::LINEAR,
                ),
            );
        }
    }

    let mut clicked: Option<usize> = None;
    let pal = UiPalette::of(ui);
    egui::ScrollArea::vertical()
        .auto_shrink([false, true])
        .show(ui, |ui| {
            if indices.is_empty() {
                ui.label("No brushes in this category.");
                return;
            }
            ui.horizontal_wrapped(|ui| {
                for &i in &indices {
                    let entry = &core.brushes.entries[i];
                    let selected = core.brushes.selected == Some(i);
                    let (rect, resp) =
                        ui.allocate_exact_size(egui::vec2(80.0, 100.0), egui::Sense::click());
                    let painter = ui.painter_at(rect);

                    painter.rect_filled(
                        rect,
                        egui::CornerRadius::same(RADIUS_CARD),
                        if selected || resp.hovered() {
                            pal.card_active
                        } else {
                            pal.card
                        },
                    );

                    let img_rect = egui::Rect::from_center_size(
                        egui::pos2(rect.center().x, rect.top() + 34.0),
                        egui::vec2(62.0, 62.0),
                    );
                    if let Some(thumb) = core.brush_thumbs.get(&i) {
                        painter.image(
                            thumb.id(),
                            img_rect,
                            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                            egui::Color32::WHITE,
                        );
                    }
                    // Subtle frame so the thumbnail reads on any card fill.
                    painter.rect_stroke(
                        img_rect,
                        egui::CornerRadius::same(RADIUS_CHIP),
                        egui::Stroke::new(1.0, pal.card_border),
                        egui::StrokeKind::Inside,
                    );
                    // Selection/hover highlight wraps the whole card so it sits
                    // flush with the background rectangle.
                    let stroke = if selected {
                        egui::Stroke::new(2.0, ACCENT)
                    } else if resp.hovered() {
                        egui::Stroke::new(1.5, ACCENT_HOVER)
                    } else {
                        egui::Stroke::NONE
                    };
                    if stroke != egui::Stroke::NONE {
                        painter.rect_stroke(
                            rect,
                            egui::CornerRadius::same(RADIUS_CARD),
                            stroke,
                            egui::StrokeKind::Inside,
                        );
                    }

                    painter.text(
                        egui::pos2(rect.center().x, rect.bottom() - 10.0),
                        egui::Align2::CENTER_CENTER,
                        truncate_mid(&entry.name, 13),
                        egui::FontId::proportional(11.0),
                        if selected { ACCENT } else { pal.chrome_text },
                    );

                    if resp.clicked() {
                        clicked = Some(i);
                    }
                    if let Some(path) = &entry.path {
                        let _ = resp.clone().on_hover_text(format!(
                            "{}\n{}",
                            entry.name,
                            path.display()
                        ));
                    } else {
                        let _ = resp
                            .clone()
                            .on_hover_text(format!("{} (built-in)", entry.name));
                    }
                }
            });
        });

    if let Some(i) = clicked {
        apply_brush(core, i);
    }
}

/// Color palette panel: add/remove palettes and swatches, pick colors into the
/// brush, and import/export palettes (GIMP `.gpl`, JASC `.pal`, or plain text).
/// The palette library is persisted next to the UI layout; the active palette
/// also feeds the RMB brush menu's quick-pick grid.
/// Snapshot the palette library before a mutation so Ctrl+Z can restore it.
/// Keeps at most 64 steps; redo is cleared on a fresh edit.
fn record_palette(core: &mut Core) {
    core.palette_prev.push(core.palettes.clone());
    if core.palette_prev.len() > 64 {
        core.palette_prev.remove(0);
    }
    core.palette_redo.clear();
    core.palette_edit_pending = true;
}

/// Armed-confirmation ids for the destructive palette buttons.
const PALETTE_CONFIRM_CLEAR: u8 = 1;
const PALETTE_CONFIRM_DELETE: u8 = 2;
const PALETTE_CONFIRM_ARM_MS: u128 = 2000;

/// Drains a stale armed confirmation (first click on a destructive button arms
/// it; if the user walks away the label turns back to normal after ~2 s).
fn expire_palette_confirm(core: &mut Core) {
    let stale = core
        .palette_confirm
        .as_ref()
        .is_some_and(|(_, t)| t.elapsed().as_millis() > PALETTE_CONFIRM_ARM_MS);
    if stale {
        core.palette_confirm = None;
    }
}

fn palette_ui(ui: &mut Ui, core: &mut Core) {
    if core.palettes.is_empty() {
        core.palettes = default_palettes();
        core.active_palette = 0;
    }
    if core.active_palette >= core.palettes.len() {
        core.active_palette = 0;
    }

    expire_palette_confirm(core);

    panel_heading(ui, "Palette");

    let mut dirty = false;

    // Select / create / import / export / delete palettes.
    let mut selected = core.active_palette;
    ui.horizontal(|ui| {
        let names: Vec<String> = core.palettes.iter().map(|p| p.name.clone()).collect();
        egui::ComboBox::from_id_salt("palette_choose")
            .selected_text(names.get(selected).map(String::as_str).unwrap_or(""))
            .show_ui(ui, |ui| {
                for (i, name) in names.iter().enumerate() {
                    if ui.selectable_label(i == selected, name).clicked() {
                        selected = i;
                    }
                }
            });
        if ui
            .button("New")
            .on_hover_text("Create an empty palette")
            .clicked()
        {
            record_palette(core);
            core.palettes.push(crate::palette::Palette {
                name: format!("Palette {}", core.palettes.len() + 1),
                colors: Vec::new(),
            });
            selected = core.palettes.len() - 1;
            dirty = true;
        }
        if ui
            .button("Import…")
            .on_hover_text("Load a .gpl, .pal or text palette")
            .clicked()
        {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Palettes", &["gpl", "pal", "txt"])
                .pick_file()
            {
                match std::fs::read(&path) {
                    Ok(bytes) => {
                        let stem = path
                            .file_stem()
                            .map(|s| s.to_string_lossy().to_string())
                            .unwrap_or_else(|| "Imported".to_string());
                        let pal = crate::palette::parse_palette(&bytes, &stem);
                        core.status = format!(
                            "Imported palette “{}” — {} colors",
                            pal.name,
                            pal.colors.len()
                        );
                        record_palette(core);
                        core.palettes.push(pal);
                        selected = core.palettes.len() - 1;
                        dirty = true;
                    }
                    Err(e) => core.status = format!("Import failed: {e}"),
                }
            }
        }
        if ui
            .button("Export…")
            .on_hover_text("Save the active palette as a GIMP .gpl file")
            .clicked()
        {
            let name = core.palettes[selected].name.clone();
            let file_name = sanitize_file_name(&name) + ".gpl";
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("GIMP palette", &["gpl"])
                .set_file_name(&file_name)
                .save_file()
            {
                let gpl = core.palettes[selected].to_gpl();
                match std::fs::write(&path, gpl.as_bytes()) {
                    Ok(()) => core.status = format!("Exported palette “{name}”"),
                    Err(e) => core.status = format!("Export failed: {e}"),
                }
            }
        }
        if ui
            .button("Restore stock")
            .on_hover_text(
                "Re-add every built-in palette; overwrites stock palettes that \
                 were cleared or edited (Ctrl+Z restores the edited library)",
            )
            .clicked()
        {
            record_palette(core);
            restore_stock_palettes(&mut core.palettes, true);
            core.status = "Restored stock palettes".to_string();
            dirty = true;
        }
        let delete_armed = core
            .palette_confirm
            .as_ref()
            .is_some_and(|(id, _)| *id == PALETTE_CONFIRM_DELETE);
        let delete_label = if delete_armed {
            "Delete? Click again"
        } else {
            "Delete"
        };
        let delete_resp = ui
            .add_enabled(core.palettes.len() > 1, egui::Button::new(delete_label))
            .on_hover_text("Remove the active palette (Ctrl+Z restores)");
        if delete_resp.clicked() {
            if delete_armed {
                record_palette(core);
                core.palettes.remove(selected);
                selected = selected.min(core.palettes.len() - 1);
                core.palette_confirm = None;
                core.status = "Deleted palette".to_string();
                dirty = true;
            } else {
                core.palette_confirm = Some((PALETTE_CONFIRM_DELETE, std::time::Instant::now()));
            }
        }
    });
    core.active_palette = selected;

    // Rename the active palette inline. Snapshot the library *before* the
    // widget so Ctrl+Z walks the rename back keystroke by keystroke.
    let rename_stash = core.palettes.clone();
    ui.horizontal(|ui| {
        ui.label("Name:");
        let resp = ui.text_edit_singleline(&mut core.palettes[core.active_palette].name);
        if resp.changed() {
            core.palette_prev.push(rename_stash);
            if core.palette_prev.len() > 64 {
                core.palette_prev.remove(0);
            }
            core.palette_redo.clear();
            core.palette_edit_pending = true;
            dirty = true;
        }
    });
    ui.separator();

    // Add the current brush color to the palette.
    let cur = core.brush.color;
    let at_cap = core.palettes[core.active_palette].colors.len() >= 256;
    ui.horizontal(|ui| {
        let pal = UiPalette::of(ui);
        let chip = egui::CornerRadius::same(RADIUS_CHIP);
        let (rect, _) = ui.allocate_exact_size(egui::vec2(22.0, 22.0), egui::Sense::hover());
        ui.painter().rect_filled(
            rect,
            chip,
            egui::Color32::from_rgba_unmultiplied(cur[0], cur[1], cur[2], cur[3]),
        );
        ui.painter().rect_stroke(
            rect,
            chip,
            egui::Stroke::new(1.0, pal.card_border),
            egui::StrokeKind::Inside,
        );
        ui.label(format!(
            "#{:02X}{:02X}{:02X} α{:02X}",
            cur[0], cur[1], cur[2], cur[3]
        ));
        if ui
            .add_enabled(!at_cap, egui::Button::new("Add current"))
            .on_hover_text("Adds the brush color to the active palette")
            .clicked()
        {
            record_palette(core);
            core.palettes[core.active_palette].colors.push(cur);
            core.status = "Added brush color to palette".to_string();
            dirty = true;
        }
        let clear_armed = core
            .palette_confirm
            .as_ref()
            .is_some_and(|(id, _)| *id == PALETTE_CONFIRM_CLEAR);
        let clear_label = if clear_armed {
            "Clear? Click again"
        } else {
            "Clear"
        };
        if ui
            .button(clear_label)
            .on_hover_text("Remove every color from the active palette (Ctrl+Z restores)")
            .clicked()
        {
            if clear_armed {
                record_palette(core);
                core.palettes[core.active_palette].colors.clear();
                core.palette_confirm = None;
                core.status = "Cleared palette".to_string();
                dirty = true;
            } else {
                core.palette_confirm = Some((PALETTE_CONFIRM_CLEAR, std::time::Instant::now()));
            }
        }
    });
    ui.separator();

    // Swatch grid: click picks into the brush, right-click removes a color.
    let count = core.palettes[core.active_palette].colors.len();
    ui.label(
        egui::RichText::new(format!("{count} colors — click picks, right-click removes"))
            .small()
            .weak(),
    );
    let colors: Vec<[u8; 4]> = core.palettes[core.active_palette].colors.clone();
    let spotlight = core.icons.as_ref().map(|s| s.trash.clone());
    let mut remove: Option<usize> = None;
    ui.horizontal_wrapped(|ui| {
        for (i, c) in colors.iter().enumerate() {
            let rgb = [c[0], c[1], c[2]];
            let resp = swatch_button(ui, rgb);
            if resp.secondary_clicked() {
                remove = Some(i);
            }
            if resp.hovered() {
                if let Some(tex) = &spotlight {
                    let size = 13.0;
                    let img_rect = egui::Rect::from_min_size(
                        resp.rect.right_top() + egui::vec2(-size, 0.0),
                        egui::vec2(size, size),
                    );
                    ui.painter()
                        .rect_filled(img_rect, 2.0, egui::Color32::from_black_alpha(160));
                    ui.painter().image(
                        tex.id(),
                        img_rect,
                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                        egui::Color32::WHITE,
                    );
                }
            }
            if resp.clicked() {
                core.brush.color = *c;
                core.status = format!("Palette #{:02X}{:02X}{:02X}", c[0], c[1], c[2]);
            }
        }
    });
    if let Some(i) = remove {
        record_palette(core);
        core.palettes[core.active_palette].colors.remove(i);
        core.status = "Removed color from palette".to_string();
        dirty = true;
    }
    if colors.is_empty() {
        ui.label(
            egui::RichText::new("No colors yet — pick a color and use “Add current”.")
                .small()
                .weak(),
        );
    }

    if dirty {
        save_palettes(&core.palettes);
    }
}

/// Strips characters that would be awkward in a file name used as an export
/// suggestion (the exporter still lets the user pick any path).
fn sanitize_file_name(name: &str) -> String {
    let mut out: String = name
        .chars()
        .map(|c| {
            if c.is_whitespace() || "\\/:*?\"<>|".contains(c) {
                '_'
            } else {
                c
            }
        })
        .collect();
    if out.is_empty() {
        out = "palette".to_string();
    }
    out
}

/// Sets the brush from a library entry, keeping the current tool active (the
/// footprint changes; an eraser in use stays an eraser) and clearing any stale
/// stamp transform.
fn apply_brush(core: &mut Core, index: usize) {
    use crate::brushes::BrushKind;
    let Some(entry) = core.brushes.entries.get(index) else {
        return;
    };
    match entry.kind {
        BrushKind::Shape(shape) => {
            core.brush.kind = shape.into();
            core.brush.sprite = None;
        }
        BrushKind::Texture => {
            core.brush.kind = crate::brush::FootprintKind::Sprite;
            core.brush.sprite = Some(entry.sprite.clone());
            core.brush.pattern_lock = crate::brush::PatternLock::Aligned;
        }
        BrushKind::Stamp => {
            core.brush.kind = crate::brush::FootprintKind::Sprite;
            core.brush.sprite = Some(entry.sprite.clone());
            core.brush.pattern_lock = crate::brush::PatternLock::Dab;
        }
    }
    core.brush.rotation = 0.0;
    core.brush.flip_x = false;
    core.brush.flip_y = false;
    // Keep the current tool: picking a brush in Eraser mode stays Eraser, in
    // Fill stays Fill, etc. Only the footprint changes.
    core.status = format!("Brush: {} · {}", entry.name, entry.category);
    core.brushes.selected = Some(index);
}

/// Renders a brush's coverage mask as dark marks on a light ground, the
/// classic way brush packs show their stamps.
fn brush_thumb_image(sprite: &crate::io::TextureData) -> ColorImage {
    let pixels = sprite
        .rgba
        .chunks_exact(4)
        .map(|px| {
            let cov = px[3] as f32 / 255.0;
            let v = (235.0 * (1.0 - cov) + 16.0 * cov).round() as u8;
            egui::Color32::from_rgb(v, v, v)
        })
        .collect();
    ColorImage::new([sprite.width as usize, sprite.height as usize], pixels)
}

/// Shortens a brush name for its thumbnail label, keeping the ends.
fn truncate_mid(s: &str, max: usize) -> String {
    let count = s.chars().count();
    if count <= max {
        return s.to_string();
    }
    let keep = max.saturating_sub(1);
    let cutoff = keep / 2;
    let head: String = s.chars().take(cutoff).collect();
    let tail: String = s.chars().skip(count - cutoff).collect();
    format!("{head}…{tail}")
}

/// Where the brush library lives: `$PIXFORGE_BRUSHES` if set, else `./brushes`.
fn brushes_folder() -> std::path::PathBuf {
    if let Some(path) = std::env::var_os("PIXFORGE_BRUSHES") {
        return std::path::PathBuf::from(path);
    }
    std::env::current_dir()
        .unwrap_or_else(|_| std::path::PathBuf::from("."))
        .join("brushes")
}

fn texture_ui(ui: &mut Ui, core: &mut Core) {
    panel_heading(ui, "Texture Editor");

    let res_options = [256u32, 512, 1024];
    if core.mesh.is_some() {
        ui.horizontal(|ui| {
            ui.label("Longest side:");
            for res in res_options {
                if ui
                    .selectable_label(core.atlas_res == res, res.to_string())
                    .clicked()
                {
                    core.atlas_res = res;
                }
            }
        });
        ui.horizontal(|ui| {
            let has_tex = core.mesh.as_ref().is_some_and(|m| !m.layers.is_empty());
            let resize_clicked = has_tex && ui.button("Resize").clicked();
            let blank_clicked = ui
                .button("Blank…")
                .on_hover_text("Replace with a clean canvas at the selected size")
                .clicked();
            if resize_clicked {
                let target = core.atlas_res;
                if let Some(mesh) = core.mesh.as_mut() {
                    if let Some(tex) = mesh.active_layer_texture() {
                        if tex.width.max(tex.height) != target {
                            core.history.record(snapshot_of(mesh));
                            for layer in &mut mesh.layers {
                                layer.texture = crate::io::resize_atlas(&layer.texture, target);
                            }
                            finish_texture_change(core, format!("Resized atlas to {target}px"));
                        }
                    }
                }
            }
            if blank_clicked {
                if let Some(mesh) = core.mesh.as_mut() {
                    let (cw, ch) = mesh
                        .active_layer_texture()
                        .map_or((0, 0), |t| (t.width, t.height));
                    let target = core.atlas_res;
                    let (bw, bh) = if cw.max(ch) == 0 {
                        (target, target)
                    } else {
                        let scale = target as f32 / cw.max(ch).max(1) as f32;
                        (
                            (cw as f32 * scale).round().max(1.0) as u32,
                            (ch as f32 * scale).round().max(1.0) as u32,
                        )
                    };
                    core.history.record(snapshot_of(mesh));
                    mesh.layers.clear();
                    mesh.layers.push(crate::io::Layer::blank(
                        "Layer 1",
                        bw,
                        bh,
                        [240, 240, 240, 255],
                    ));
                    mesh.active_layer = 0;
                    finish_texture_change(core, format!("New blank canvas {bw}x{bh}"));
                }
            }
        });
        ui.separator();
    }

    let gen = core.preview_gen;
    let dims = core
        .mesh
        .as_ref()
        .and_then(|m| m.layers.first())
        .map(|l| (l.texture.width, l.texture.height));

    if let Some((tw, th)) = dims {
        if core.texture_preview.as_ref().map(|p| p.gen) != Some(gen) {
            // Incremental: a stroke flush recomposites + records the dirty
            // texel rect; patch exactly that rect into the cached egui texture
            // so a paint edit never recomposites or re-uploads the whole atlas.
            let patched = core
                .preview_patch
                .take()
                .and_then(|(x0, y0, w, h, fw, fh)| {
                    if fw != tw || fh != th {
                        // Atlas changed size (resize/blank/load); the stale
                        // region is invalid — fall through to a full rebuild.
                        return None;
                    }
                    let region = core.mesh.as_ref()?.flattened_atlas_region(x0, y0, w, h)?;
                    let p = core.texture_preview.as_mut()?;
                    let pixels = region
                        .rgba
                        .chunks_exact(4)
                        .map(|px| egui::Color32::from_rgba_unmultiplied(px[0], px[1], px[2], px[3]))
                        .collect();
                    let img =
                        ColorImage::new([region.width as usize, region.height as usize], pixels);
                    p.handle.set_partial(
                        [x0 as usize, y0 as usize],
                        img,
                        egui::TextureOptions::NEAREST,
                    );
                    p.gen = gen;
                    Some(())
                })
                .is_some();
            if !patched {
                // Full rebuild: a structure change (resize/blank/layer ops/
                // visibility) or no patch — flatten the whole atlas into a
                // fresh egui copy. Only runs when the preview actually changes.
                if let Some(tex) = core.mesh.as_ref().and_then(|m| m.flattened_atlas()) {
                    let pixels = tex
                        .rgba
                        .chunks_exact(4)
                        .map(|px| egui::Color32::from_rgba_unmultiplied(px[0], px[1], px[2], px[3]))
                        .collect();
                    let img = ColorImage::new([tex.width as usize, tex.height as usize], pixels);
                    match core.texture_preview.as_mut() {
                        Some(p) => {
                            p.handle.set(img, egui::TextureOptions::NEAREST);
                            p.gen = gen;
                        }
                        None => {
                            let handle = ui.ctx().load_texture(
                                format!("albedo_preview_{gen}"),
                                img,
                                egui::TextureOptions::NEAREST,
                            );
                            core.texture_preview = Some(PreviewTexture { gen, handle });
                        }
                    }
                }
            }
        }

        if let Some(handle_id) = core.texture_preview.as_ref().map(|p| p.handle.id()) {
            let (tw_f, th_f) = (tw.max(1) as f32, th.max(1) as f32);

            // Re-fit after load/resize/blank: center the atlas and scale it to
            // fill the available panel area.
            if core.canvas2d.needs_fit {
                let avail = ui.available_size();
                let zoom = (avail.x / tw_f).min(avail.y / th_f);
                core.canvas2d.zoom = zoom.clamp(0.01, 64.0);
                core.canvas2d.center = egui::Vec2::ZERO;
                core.canvas2d.needs_fit = false;
            }

            // Static toolbar row: live zoom readout, Fit / 100%, and the
            // 2D UV-overlay toggle. The 3D-viewport overlays (UV checker /
            // UV grid) live in the 3D viewport's own floating bar.
            ui.horizontal(|ui| {
                ui.label(format!("{:.0}%", core.canvas2d.zoom * 100.0));
                if ui.button("Fit").clicked() {
                    core.canvas2d.needs_fit = true;
                }
                if ui.button("100%").clicked() {
                    core.canvas2d.zoom = 1.0;
                    core.canvas2d.center = egui::Vec2::ZERO;
                }
                ui.separator();
                ui.checkbox(&mut core.show_uv_overlay, "Show UV overlay")
                    .on_hover_text("UV wireframe over the 2D canvas");
            });
            ui.separator();

            // The canvas widget fills the whole panel; a dark backdrop sits
            // behind the atlas.
            let (rect, _resp) =
                ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
            let pal = UiPalette::of(ui);
            ui.painter().rect_filled(rect, 0.0, pal.well);

            // Toggleable brush picker (tool strip) over the preview's left edge,
            // sliding in/out on T — same animation as the 3D viewport's T-bar.
            let strip_target = if core.show_brush_picker { 1.0 } else { 0.0 };
            if (core.brush_picker_anim - strip_target).abs() > 1e-3 {
                let dt = ui.input(|i| i.stable_dt).clamp(0.0, 0.1);
                let dir = if strip_target > core.brush_picker_anim {
                    1.0
                } else {
                    -1.0
                };
                core.brush_picker_anim =
                    (core.brush_picker_anim + dir * dt / STRIP_ANIM_S).clamp(0.0, 1.0);
                ui.ctx().request_repaint();
            }
            let anim = core.brush_picker_anim;
            let strip_rect = tool_strip_rect(rect.min, anim);

            // Per-panel T toggle for the 2D brush picker: consume the key only while
            // the pointer hovers the canvas (or the picker itself), so the 3D
            // viewport keeps its own instance of the same action.
            let bind_2d = *core.shortcuts.get(ShortcutAction::ToggleTools2d);
            if core.recording.is_none()
                && (ui.rect_contains_pointer(rect)
                    || ui
                        .input(|i| i.pointer.hover_pos())
                        .is_some_and(|p| strip_rect.expand(2.0).contains(p)))
                && bind_2d.is_bound()
                && ui
                    .ctx()
                    .input_mut(|i| i.consume_key(bind_2d.modifiers_of(), bind_2d.key_of()))
            {
                core.show_brush_picker = !core.show_brush_picker;
                core.status = if core.show_brush_picker {
                    "Texture tools: on (toggle: Preferences)".to_string()
                } else {
                    "Texture tools: off (toggle: Preferences)".to_string()
                };
            }

            // Camera + painting input, all from *raw* pointer events read straight off
            // the context — exactly like the 3D viewport (which has always
            // painted reliably) — never egui widget `Response` flags, which the
            // dock & scroll layers occasionally interfere with. The
            // `click_and_drag` allocation above stays purely so the canvas
            // claims the drag and the enclosing ScrollArea doesn't scroll the
            // panel mid-stroke.
            let canvas_hovered = ui.rect_contains_pointer(rect);
            let strip_active = ui.rect_contains_pointer(strip_rect);
            let hovered = canvas_hovered && !strip_active;
            let pointer = ui.input(|i| i.pointer.hover_pos());
            let primary_down = ui.input(|i| i.pointer.primary_down());
            let pressed = ui.input(|i| i.pointer.button_pressed(egui::PointerButton::Primary));
            let released = ui.input(|i| i.pointer.button_released(egui::PointerButton::Primary));

            // RMB pops up the brush menu over the canvas, exactly like the 3D
            // viewport (color, pick, brush type, accumulate, load PNG).
            let secondary_clicked = ui.input(|i| i.pointer.secondary_clicked());
            if hovered && secondary_clicked && !core.brush_menu_open {
                if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
                    core.brush_menu_open = true;
                    core.brush_menu_pos = Some(pos);
                    ui.ctx().request_repaint();
                }
            }

            // Fit the 2D canvas via the remappable shortcut (default F), like
            // the 3D viewport's Fit.
            let bind_fit_2d = *core.shortcuts.get(ShortcutAction::Fit2d);
            if core.recording.is_none()
                && hovered
                && !ui.ctx().egui_wants_keyboard_input()
                && bind_fit_2d.is_bound()
                && ui
                    .ctx()
                    .input_mut(|i| i.consume_key(bind_fit_2d.modifiers_of(), bind_fit_2d.key_of()))
            {
                core.canvas2d.needs_fit = true;
            }

            // Pan: hold the middle mouse button and drag (Krita/Blender style).
            if hovered && ui.input(|i| i.pointer.middle_down()) {
                let delta = ui.input(|i| i.pointer.delta());
                if delta != egui::Vec2::ZERO {
                    core.canvas2d.center += egui::vec2(delta.x, delta.y);
                    ui.ctx().request_repaint();
                }
            }

            // Zoom: plain scroll resizes about the cursor; shift+scroll keeps
            // resizing the brush (same modifier mapping as the 3D viewport).
            let scroll = ui.input(|i| i.smooth_scroll_delta);
            let shift = ui.input(|i| i.modifiers.shift);
            let scroll_amount = if scroll.y.abs() > scroll.x.abs() {
                scroll.y
            } else {
                scroll.x
            };
            if hovered && scroll_amount != 0.0 {
                if shift {
                    let factor = 1.0 - scroll_amount * 0.015;
                    core.brush.size = (core.brush.size * factor).clamp(1.0, 300.0);
                } else {
                    // Scale about the cursor: the texel under the pointer stays
                    // fixed on screen while everything else zooms around it.
                    let new_zoom =
                        (core.canvas2d.zoom * (scroll_amount * 0.0015).exp()).clamp(0.01, 64.0);
                    if new_zoom != core.canvas2d.zoom {
                        let factor = new_zoom / core.canvas2d.zoom;
                        let new_center = if let Some(p) = pointer {
                            let before_size =
                                egui::vec2(tw_f * core.canvas2d.zoom, th_f * core.canvas2d.zoom);
                            let before_min =
                                rect.center() + core.canvas2d.center - before_size * 0.5;
                            let fu = ((p.x - before_min.x) / before_size.x).clamp(0.0, 1.0);
                            let fv = ((p.y - before_min.y) / before_size.y).clamp(0.0, 1.0);
                            let after_size = before_size * factor;
                            let after_min =
                                egui::pos2(p.x - fu * after_size.x, p.y - fv * after_size.y);
                            after_min + after_size * 0.5 - rect.center()
                        } else {
                            core.canvas2d.center
                        };
                        core.canvas2d.center = new_center;
                    }
                    core.canvas2d.zoom = new_zoom;
                    ui.ctx().request_repaint();
                }
            }

            // Image placement from the final camera: a single on-screen rect
            // for the whole atlas at `zoom` screen px per texel.
            let img_size = egui::vec2(tw_f * core.canvas2d.zoom, th_f * core.canvas2d.zoom);
            let img_rect =
                egui::Rect::from_center_size(rect.center() + core.canvas2d.center, img_size);
            let p = ui.painter_at(rect);
            draw_checkerboard(ui, img_rect, p.clip_rect());
            p.image(
                handle_id,
                img_rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
            draw_uv_overlay(ui, img_rect, p.clip_rect(), core);
            p.rect_stroke(
                img_rect,
                0.0,
                egui::Stroke::new(1.0, pal.well_border),
                egui::StrokeKind::Outside,
            );
            p.text(
                rect.right_top() + egui::vec2(-6.0, 6.0),
                egui::Align2::RIGHT_TOP,
                "MMB drag: pan · wheel: zoom · shift+wheel: brush size",
                egui::FontId::proportional(12.0),
                pal.chrome_text_weak,
            );
            p.text(
                rect.left_bottom() + egui::vec2(6.0, -6.0),
                egui::Align2::LEFT_BOTTOM,
                format!("{} × {} @ {:.0}%", tw, th, core.canvas2d.zoom * 100.0),
                egui::FontId::proportional(12.0),
                pal.chrome_text,
            );

            // The brush picker (T-bar) floats on top of everything, like the
            // 3D viewport's tool strip.
            if core.show_brush_picker || anim > 0.0 {
                view_tool_strip(ui, core, strip_rect);
            }

            let mut painted = false;
            // Painting only happens over the atlas itself (not the empty canvas
            // around it, where pan/zoom still work) so a stroke outside the
            // texture never gets force-clamped to its edge.
            let editing_locked = core.active_tool != 3
                && core.mesh.as_ref().map(active_layer_locked).unwrap_or(false);
            let over_image = pointer.is_some_and(|p| img_rect.contains(p));
            // With no layers there is nothing to paint into: the 3D stamp path
            // returns early on the same condition (paint.rs stamp_texels), and
            // the `active_layer_texture_mut().unwrap()` calls below would panic.
            let has_layer = core.mesh.as_ref().is_some_and(|m| !m.layers.is_empty());
            if hovered
                && over_image
                && !editing_locked
                && has_layer
                && !core.brush_menu_open
                && (primary_down || pressed || released)
            {
                let pw = img_rect.width();
                let ph = img_rect.height();
                if pw > 0.0 && ph > 0.0 {
                    if let Some(p_pos) = pointer {
                        let u = ((p_pos.x - img_rect.min.x) / pw).clamp(0.0, 1.0);
                        let v = ((p_pos.y - img_rect.min.y) / ph).clamp(0.0, 1.0);

                        let mode = if core.active_tool == 0 || core.active_tool == 4 {
                            crate::paint::StampMode::Paint
                        } else {
                            crate::paint::StampMode::Erase
                        };
                        // Stamping happens through `&core.brush` directly (no per-dab copies):
                        // only the mode and the footprint kind are tool-driven —
                        // eraser vs paint, and rect tools stamp a Square in 2D.
                        // The chosen shape is restored after the stroke branch.
                        core.brush.mode = mode;
                        let prev_kind = core.brush.kind;
                        if core.active_tool == 4 {
                            core.brush.kind = crate::brush::FootprintKind::Square;
                        }

                        // Brush radius in texels: the on-screen footprint stays
                        // `brush_size` px at any zoom, so the cursor ring always
                        // matches the stamp it previews.
                        let brush_r_texels = core.brush.size * (tw as f32 / pw);

                        if core.active_tool == 3 {
                            // Pick tool: sample directly from the atlas at the
                            // cursor UV and switch back to Brush. The composite
                            // (what's on screen) is flattened lazily only on
                            // press — a discrete action, not per frame.
                            if pressed {
                                if let Some(flat) =
                                    core.mesh.as_ref().and_then(|m| m.flattened_atlas())
                                {
                                    let px = (u * flat.width as f32).round() as u32;
                                    let py = (v * flat.height as f32).round() as u32;
                                    let px = px.min(flat.width.saturating_sub(1));
                                    let py = py.min(flat.height.saturating_sub(1));
                                    let idx = ((py * flat.width + px) * 4) as usize;
                                    if idx + 3 < flat.rgba.len() {
                                        let c: [u8; 4] = [
                                            flat.rgba[idx],
                                            flat.rgba[idx + 1],
                                            flat.rgba[idx + 2],
                                            flat.rgba[idx + 3],
                                        ];
                                        core.brush.color = c;
                                        core.active_tool = 0;
                                        core.status = format!(
                                            "Picked rgba({}, {}, {}, {}) — back to Brush",
                                            c[0], c[1], c[2], c[3]
                                        );
                                    }
                                }
                            }
                        } else if core.active_tool == 2 {
                            // Fill tool: flood-fill the region around the click
                            // point with the current brush color.
                            if pressed {
                                core.palette_edit_pending = false;
                                if let Some(m) = core.mesh.as_ref() {
                                    core.history.record(snapshot_of(m));
                                }
                                if let Some(mesh) = core.mesh.as_mut() {
                                    let mut dirty = mesh.dirty;
                                    crate::paint::stamp_fill_2d(
                                        mesh.active_layer_texture_mut().unwrap(),
                                        (u, v),
                                        core.brush.color,
                                        core.brush.opacity,
                                        &mut dirty,
                                    );
                                    mesh.dirty = dirty;
                                }
                                painted = true;
                            }
                        } else if primary_down {
                            // Brush / Eraser / Rect: the press stamps the initial dab
                            // (so a click-and-hold with no travel still leaves a mark),
                            // then freehand dabs continue while the button is held.
                            // Rect (tool 4) rides the same path with a square
                            // footprint — same continuous behavior as the 3D
                            // viewport. The stroke self-heals: if the pointer
                            // briefly leaves the canvas mid-drag, a new stroke
                            // begins on re-entry.
                            if core.stroke_2d.is_none() {
                                core.palette_edit_pending = false;
                                if let Some(m) = core.mesh.as_ref() {
                                    core.history.record(snapshot_of(m));
                                }
                                // Pattern-locked 2D texture strokes pin the sprite phase to the
                                // stroke-start texel, mirroring the 3D anchor.
                                let pattern = if core.brush.pattern_lock
                                    == crate::brush::PatternLock::Aligned
                                    && core.brush.kind == crate::brush::FootprintKind::Sprite
                                    && core.brush.sprite.is_some()
                                    && core.active_tool != 4
                                {
                                    Some(crate::brush::PatternAnchor::Uv {
                                        x: u * tw as f32,
                                        y: v * th as f32,
                                        radius: brush_r_texels,
                                    })
                                } else {
                                    None
                                };
                                // Pattern-locked strokes always allocate the
                                // stroke buffer so the replace blend caps every
                                // overlap at one dab's worth (never "fill up");
                                // plain non-accumulative brushes allocate as before.
                                let stroke_alpha = if core.brush.accumulate && pattern.is_none() {
                                    None
                                } else if let Some(mesh) = core.mesh.as_ref() {
                                    let (tw, th) = mesh
                                        .active_layer_texture()
                                        .map(|t| (t.width as usize, t.height as usize))
                                        .unwrap_or((0, 0));
                                    if tw > 0 && th > 0 {
                                        Some(vec![0u8; tw * th])
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                };
                                core.stroke_2d = Some(StrokeState {
                                    last: egui::pos2(u, v),
                                    start: egui::pos2(u, v),
                                    last_dab: egui::pos2(u, v),
                                    acc: 0.0,
                                    next_t: 0.0,
                                    accel: None,
                                    stroke_alpha,
                                    pattern,
                                    unwrap: None,
                                });
                                if let Some(mesh) = core.mesh.as_mut() {
                                    let mut dirty = mesh.dirty;
                                    let pattern = core.stroke_2d.as_ref().unwrap().pattern;
                                    crate::paint::stamp_2d(
                                        mesh.active_layer_texture_mut().unwrap(),
                                        (u, v),
                                        brush_r_texels,
                                        &core.brush,
                                        &mut dirty,
                                        core.stroke_2d
                                            .as_mut()
                                            .unwrap()
                                            .stroke_alpha
                                            .as_deref_mut(),
                                        pattern.as_ref(),
                                    );
                                    mesh.dirty = dirty;
                                }
                                painted = true;
                            } else if let Some(st) = core.stroke_2d.as_mut() {
                                // Dab-to-dab travel is accumulated in *texel*
                                // space: spacing is in screen px (the 3D
                                // viewport semantics), but the 2D cursor lives
                                // in normalized UV — on a 1024² atlas a "6px"
                                // step must be 6 texels, not 6 UV units.
                                // World-locked pattern strokes ride an even
                                // denser train (half the brush radius) so the
                                // soft window masks overlap into a continuous
                                // anchored stroke — no coin-edge chain.
                                // Pattern-aligned texture strokes share this
                                // spaced dab loop and the `stroke_alpha`
                                // replace buffer, so overlaps stay flat.
                                let spacing = if st.pattern.is_some() {
                                    core.brush.pattern_spacing()
                                } else {
                                    core.brush.effective_spacing()
                                };
                                let mut dabs: Vec<egui::Pos2> = Vec::new();
                                let to_px = |x: f32, y: f32| egui::pos2(x * tw_f, y * th_f);

                                if ui.input(|i| i.modifiers.shift) {
                                    let start_px = to_px(st.start.x, st.start.y);
                                    let now_px = to_px(u, v);
                                    let dir = (now_px - start_px).normalized();
                                    let total = (now_px - start_px).length();
                                    while st.next_t + spacing <= total {
                                        st.next_t += spacing;
                                        let dab = start_px + dir * st.next_t;
                                        dabs.push(egui::pos2(dab.x / tw_f, dab.y / th_f));
                                    }
                                } else {
                                    let (dabs_here, acc, last_dab) =
                                        crate::paint::spaced_freehand_dabs(
                                            to_px(st.last.x, st.last.y),
                                            to_px(u, v),
                                            to_px(st.last_dab.x, st.last_dab.y),
                                            st.acc,
                                            spacing,
                                        );
                                    st.acc = acc;
                                    st.last_dab = egui::pos2(last_dab.x / tw_f, last_dab.y / th_f);
                                    dabs.extend(
                                        dabs_here
                                            .into_iter()
                                            .map(|p| egui::pos2(p.x / tw_f, p.y / th_f)),
                                    );
                                }
                                st.last = egui::pos2(u, v);

                                if let Some(mesh) = core.mesh.as_mut() {
                                    for dab in &dabs {
                                        let mut dirty = mesh.dirty;
                                        crate::paint::stamp_2d(
                                            mesh.active_layer_texture_mut().unwrap(),
                                            (dab.x, dab.y),
                                            brush_r_texels,
                                            &core.brush,
                                            &mut dirty,
                                            st.stroke_alpha.as_deref_mut(),
                                            st.pattern.as_ref(),
                                        );
                                        mesh.dirty = dirty;
                                    }
                                    painted = !dabs.is_empty();
                                }
                            }
                        }
                        core.brush.kind = prev_kind;
                    }
                }
            }

            if painted {
                flush_paint_edit(core);
                if core.active_tool == 1 {
                    core.status = "Erased — fully transparent (alpha 0) in 2D preview".to_string();
                }
                ui.ctx().request_repaint();
            }

            // End the stroke on pointer release or when the pointer leaves the
            // painting surface — the whole canvas, or the atlas itself while it
            // wanders through the surrounding margin (mirrors the 3D viewport's
            // hover-gating). Keeping the stroke alive across a margin excursion
            // would re-interpolate a straight chord between the exit and re-entry
            // dabs over the image.
            if !(hovered && over_image && primary_down) {
                core.stroke_2d = None;
            }

            // The 2D brush cursor mirrors the 3D viewport's: same fill/stroke colors,
            // same shape handling, and the actual brush sprite (shape + pattern +
            // rotation/flips) for texture brushes — all clipped to the canvas so
            // it never bleeds over the toolbar above.
            if hovered {
                if let Some(p_pos) = pointer {
                    let brush_r = core.brush.size;
                    let painting = core.active_tool == 0 || core.active_tool == 4;
                    let (fill, stroke, dot) = if painting {
                        (
                            egui::Color32::from_rgba_unmultiplied(
                                core.brush.color[0],
                                core.brush.color[1],
                                core.brush.color[2],
                                60,
                            ),
                            egui::Color32::from_rgba_unmultiplied(
                                core.brush.color[0],
                                core.brush.color[1],
                                core.brush.color[2],
                                255,
                            ),
                            egui::Color32::WHITE,
                        )
                    } else {
                        (
                            egui::Color32::from_rgba_unmultiplied(255, 255, 255, 30),
                            egui::Color32::WHITE,
                            egui::Color32::WHITE,
                        )
                    };
                    match core.active_tool {
                        4 => {
                            let square = egui::Rect::from_center_size(
                                p_pos,
                                egui::vec2(brush_r * 2.0, brush_r * 2.0),
                            );
                            p.rect_filled(square, 0.0, fill);
                            p.rect_stroke(
                                square,
                                0.0,
                                egui::Stroke::new(1.5, stroke),
                                egui::StrokeKind::Outside,
                            );
                        }
                        _ => match core.brush.kind {
                            crate::brush::FootprintKind::Round => {
                                p.circle_filled(p_pos, brush_r, fill);
                                p.circle_stroke(p_pos, brush_r, egui::Stroke::new(1.5, stroke));
                            }
                            crate::brush::FootprintKind::Square
                            | crate::brush::FootprintKind::Rect => {
                                let square = egui::Rect::from_center_size(
                                    p_pos,
                                    egui::vec2(brush_r * 2.0, brush_r * 2.0),
                                );
                                p.rect_filled(square, 0.0, fill);
                                p.rect_stroke(
                                    square,
                                    0.0,
                                    egui::Stroke::new(1.5, stroke),
                                    egui::StrokeKind::Outside,
                                );
                            }
                            crate::brush::FootprintKind::Diamond => {
                                let r = brush_r;
                                let pts = vec![
                                    p_pos + egui::vec2(0.0, -r),
                                    p_pos + egui::vec2(r, 0.0),
                                    p_pos + egui::vec2(0.0, r),
                                    p_pos + egui::vec2(-r, 0.0),
                                ];
                                p.add(egui::Shape::convex_polygon(
                                    pts,
                                    fill,
                                    egui::Stroke::new(1.5, stroke),
                                ));
                            }
                            crate::brush::FootprintKind::Sprite => {
                                // WYSIWYG cursor: rasterize the actual
                                // stamp (frame shape + tiled sprite +
                                // rotation + flips, tinted) at the real
                                // dab radius, so selecting
                                // Square/Diamond shows the
                                // square/diamond frame live under the
                                // cursor, matching exactly what a dab
                                // paints.
                                let tex = brush_cursor_preview(ui.ctx(), core, brush_r);
                                let side = (brush_r * 2.0).max(1.0);
                                let square =
                                    egui::Rect::from_center_size(p_pos, egui::vec2(side, side));
                                p.image(
                                    tex,
                                    square,
                                    egui::Rect::from_min_max(
                                        egui::pos2(0.0, 0.0),
                                        egui::pos2(1.0, 1.0),
                                    ),
                                    egui::Color32::WHITE,
                                );
                            }
                        },
                    }
                    p.circle_stroke(p_pos, 2.0, egui::Stroke::new(1.0, dot));
                    ui.ctx().request_repaint();
                }
            }
        }
    } else {
        let avail = ui.available_size();
        let img_size = egui::vec2(avail.x, avail.x.min(avail.y));
        let (rect, _) = ui.allocate_exact_size(img_size, egui::Sense::hover());
        ui.painter().rect_filled(rect, 0.0, UiPalette::of(ui).well);
        draw_uv_overlay(ui, rect, rect, core);
        ui.label("This model has no material texture.");
    }

    if let Some(mesh) = core.mesh.as_ref() {
        ui.label(format!(
            "{} triangles, {} vertices",
            mesh.indices.len() / 3,
            mesh.positions.len()
        ));
    }

    brush_menu_popup(ui, core);
}

/// Layer stack panel: per-layer visibility / lock / rename / opacity /
/// selection plus the structural operations (add, duplicate, delete, reorder).
/// The topmost layer is listed first. Rows reorder by dragging the ⠿ handle.
/// Every mutation here is undoable: visibility, lock, rename, blend, opacity
/// (recorded once per drag) and all structural edits.
fn layers_ui(ui: &mut Ui, core: &mut Core) {
    let Some(mesh) = core.mesh.as_mut() else {
        return;
    };

    let len = mesh.layers.len();
    let active = mesh.active_layer;
    let res = core.atlas_res;
    let active_locked = active_layer_locked(mesh);
    // A stale rename target (layer removed/reordered underneath it) is dropped.
    if core.renaming.is_some_and(|r| r >= len) {
        core.renaming = None;
        core.rename_grab_focus = false;
    }

    if core.icons.is_none() {
        core.icons = load_icons(ui.ctx());
    }
    let (i_eye, i_eye_off, i_lock, i_lock_open, i_grip, i_plus, i_copy, i_trash, i_up, i_down) = {
        let s = core.icons.as_ref().unwrap();
        (
            s.eye.clone(),
            s.eye_off.clone(),
            s.lock.clone(),
            s.lock_open.clone(),
            s.grip.clone(),
            s.plus.clone(),
            s.copy.clone(),
            s.trash.clone(),
            s.arrow_up.clone(),
            s.arrow_down.clone(),
        )
    };
    let theme = ui.visuals().text_color();
    let dim = ui.visuals().weak_text_color();

    let mut add = false;
    let mut duplicate = false;
    let mut delete = false;
    let mut move_up = false;
    let mut move_down = false;
    let mut flip_x = false;
    let mut flip_y = false;
    // Toolbar split across two rows so it never wraps on narrow panels:
    // creation/destruction first, then stack order + flips.
    ui.horizontal_wrapped(|ui| {
        add = icon_button(ui, &i_plus, 15.0, true, theme, "Add layer").clicked();
        duplicate = icon_button(
            ui,
            &i_copy,
            15.0,
            active < len && !active_locked,
            if active < len && !active_locked {
                theme
            } else {
                dim
            },
            "Duplicate layer",
        )
        .clicked();
        delete = icon_button(
            ui,
            &i_trash,
            15.0,
            len > 0 && !active_locked,
            if len > 0 && !active_locked {
                theme
            } else {
                dim
            },
            "Delete layer",
        )
        .clicked();
    });
    ui.horizontal_wrapped(|ui| {
        move_up = icon_button(
            ui,
            &i_up,
            15.0,
            active > 0 && !active_locked,
            if active > 0 && !active_locked {
                theme
            } else {
                dim
            },
            "Move layer up",
        )
        .clicked();
        move_down = icon_button(
            ui,
            &i_down,
            15.0,
            active + 1 < len && !active_locked,
            if active + 1 < len && !active_locked {
                theme
            } else {
                dim
            },
            "Move layer down",
        )
        .clicked();
        ui.separator();
        let can_flip = active < len && !active_locked;
        flip_x = ui
            .add_enabled(can_flip, egui::Button::new("Flip X"))
            .on_hover_text("Mirror the layer's paint left/right")
            .clicked();
        flip_y = ui
            .add_enabled(can_flip, egui::Button::new("Flip Y"))
            .on_hover_text("Mirror the layer's paint top/bottom")
            .clicked();
    });

    if len > 0 && !active_locked {
        let active_mode = mesh.layers[mesh.active_layer].blend;
        let mut new_mode = active_mode;
        egui::ComboBox::from_id_salt("blend_mode")
            .selected_text(active_mode.name())
            .width(ui.available_width())
            .show_ui(ui, |ui| {
                for m in crate::io::BlendMode::ALL {
                    ui.selectable_value(&mut new_mode, m, m.name());
                }
            });
        if new_mode != active_mode {
            core.history.record(snapshot_of(mesh));
            mesh.layers[mesh.active_layer].blend = new_mode;
            core.needs_texture_upload = true;
            core.needs_material_upload = true;
            core.preview_gen += 1;
            core.status = format!("Layer blend: {}", new_mode.name());
        }
    } else if active_locked {
        ui.horizontal(|ui| {
            ui.label("Blend:");
            ui.add(
                egui::Image::new(&i_lock)
                    .fit_to_exact_size(egui::vec2(14.0, 14.0))
                    .tint(theme),
            );
            ui.label(mesh.layers[mesh.active_layer].blend.short_name())
                .on_hover_text("Locked layer — unlock to change blending");
        });
    }
    ui.separator();

    if add {
        core.history.record(snapshot_of(mesh));
        let (w, h) = mesh
            .active_layer_texture()
            .map(|t| (t.width, t.height))
            .unwrap_or((res, res));
        mesh.layers.push(crate::io::Layer::blank(
            format!("Layer {}", len + 1),
            w,
            h,
            [0, 0, 0, 0],
        ));
        mesh.active_layer = mesh.layers.len() - 1;
        core.renaming = None;
        core.stroke = None;
        core.stroke_2d = None;
        core.needs_texture_upload = true;
        core.needs_material_upload = true;
        core.preview_gen += 1;
        core.status = format!("Added layer {}", mesh.layers.len());
    }
    if duplicate {
        if let Some(src) = mesh.layers.get(active) {
            core.history.record(snapshot_of(mesh));
            let mut copy = src.clone();
            copy.name = format!("{} copy", src.name);
            mesh.layers.insert(active + 1, copy);
            mesh.active_layer = active + 1;
            core.renaming = None;
            core.stroke = None;
            core.stroke_2d = None;
            core.needs_texture_upload = true;
            core.needs_material_upload = true;
            core.preview_gen += 1;
            core.status = "Duplicated layer".to_string();
        }
    }
    if delete {
        core.history.record(snapshot_of(mesh));
        mesh.layers.remove(active.min(mesh.layers.len() - 1));
        if mesh.layers.is_empty() {
            mesh.active_layer = 0;
        } else {
            mesh.active_layer = mesh.active_layer.min(mesh.layers.len() - 1);
        }
        core.renaming = None;
        core.stroke = None;
        core.stroke_2d = None;
        core.needs_texture_upload = true;
        core.needs_material_upload = true;
        core.preview_gen += 1;
        core.status = "Deleted layer".to_string();
    }
    if move_up {
        core.history.record(snapshot_of(mesh));
        mesh.layers.swap(active, active - 1);
        mesh.active_layer = active - 1;
        core.renaming = None;
        core.stroke = None;
        core.stroke_2d = None;
        core.needs_texture_upload = true;
        core.needs_material_upload = true;
        core.preview_gen += 1;
        core.status = "Layer moved up".to_string();
    }
    if move_down {
        core.history.record(snapshot_of(mesh));
        mesh.layers.swap(active, active + 1);
        mesh.active_layer = active + 1;
        core.renaming = None;
        core.stroke = None;
        core.stroke_2d = None;
        core.needs_texture_upload = true;
        core.needs_material_upload = true;
        core.preview_gen += 1;
        core.status = "Layer moved down".to_string();
    }

    if mesh.layers.is_empty() {
        let pal = UiPalette::of(ui);
        ui.add_space(24.0);
        ui.vertical_centered(|ui| {
            ui.label(
                egui::RichText::new("No layers yet")
                    .size(13.0)
                    .color(pal.chrome_text_weak),
            );
            ui.add_space(4.0);
            ui.label(
                egui::RichText::new("Press + to add one and start painting.")
                    .size(11.0)
                    .color(pal.card_border_active),
            );
        });
        return;
    }

    let mut needs_refresh = false;
    let mut hover_target: Option<usize> = None;
    let mut drop_target: Option<(usize, bool)> = None;
    let drag_payload = egui::DragAndDrop::payload::<LayerDrag>(ui.ctx()).map(|p| *p);
    let is_dragging = drag_payload.is_some();
    let drag_from = drag_payload.map(|p| p.from);
    let pointer_pos = ui.input(|i| i.pointer.hover_pos());

    // Topmost layer listed first: iterate the stack in reverse. The whole list
    // is one drop zone; each unlocked row has a grip-icon drag handle as its
    // DnD source. Locked rows are rendered inert (no handle, no edits).
    let dropped = {
        ui.dnd_drop_zone::<LayerDrag, _>(egui::Frame::NONE, |ui| {
            ui.spacing_mut().item_spacing.y = 4.0;
            for li in (0..mesh.layers.len()).rev() {
                let (name, visible, locked, opacity) = {
                    let l = &mesh.layers[li];
                    (l.name.clone(), l.visible, l.locked, l.opacity)
                };
                let is_active = li == mesh.active_layer;
                let is_renaming = core.renaming == Some(li);
                let grab = core.rename_grab_focus && is_renaming;
                let mut local_buf = if is_renaming {
                    core.rename_buf.clone()
                } else {
                    name.clone()
                };

                let mut toggled = false;
                let mut lock_toggled = false;
                let mut selected = false;
                let mut start_rename = false;
                let mut commit_rename = false;
                let mut cancel_rename = false;
                let mut op = opacity;
                let mut op_changed = false;
                let mut op_drag = false;

                // Blender-style card container
                let pal = UiPalette::of(ui);
                let is_being_dragged = drag_from == Some(li);
                let card_corner = egui::CornerRadius::same(RADIUS_CHIP);
                let card_bg = if is_being_dragged {
                    // Origin slot looks vacated / ghosted
                    egui::Color32::from_rgba_unmultiplied(
                        pal.card.r(),
                        pal.card.g(),
                        pal.card.b(),
                        90,
                    )
                } else if is_active {
                    pal.card_active
                } else {
                    pal.card
                };
                let card_stroke = if is_being_dragged {
                    egui::Stroke::new(
                        1.0,
                        egui::Color32::from_rgba_unmultiplied(
                            ACCENT.r(),
                            ACCENT.g(),
                            ACCENT.b(),
                            120,
                        ),
                    )
                } else if is_active {
                    egui::Stroke::new(1.0, pal.card_border_active)
                } else {
                    egui::Stroke::new(1.0, pal.card_border)
                };

                let card_resp = egui::Frame::NONE
                    .fill(card_bg)
                    .stroke(card_stroke)
                    .corner_radius(card_corner)
                    .inner_margin(egui::Margin::symmetric(6, 6))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 5.0;
                            if !locked {
                                ui.dnd_drag_source(
                                    Id::new(("pixforge_layer_row", li)),
                                    LayerDrag { from: li },
                                    |ui| {
                                        let img = egui::Image::new(&i_grip)
                                            .fit_to_exact_size(egui::vec2(14.0, 14.0))
                                            .tint(if is_active { ACCENT } else { dim });
                                        let resp =
                                            ui.add(img).on_hover_cursor(egui::CursorIcon::Grab);
                                        resp.on_hover_text("Drag to reorder");
                                    },
                                );
                            } else {
                                ui.add_space(14.0);
                            }

                            toggled = icon_button(
                                ui,
                                if visible { &i_eye } else { &i_eye_off },
                                15.0,
                                true,
                                if visible { theme } else { dim },
                                if visible { "Hide layer" } else { "Show layer" },
                            )
                            .clicked();

                            lock_toggled = icon_button(
                                ui,
                                if locked { &i_lock } else { &i_lock_open },
                                15.0,
                                true,
                                if locked { ACCENT } else { dim },
                                if locked {
                                    "Lock layer — protects it from paint, fill and property edits"
                                } else {
                                    "Lock layer"
                                },
                            )
                            .clicked();

                            if is_renaming {
                                let r = ui.add(
                                    egui::TextEdit::singleline(&mut local_buf)
                                        .desired_width(95.0)
                                        .clip_text(true),
                                );
                                if grab {
                                    r.request_focus();
                                }
                                if r.changed() {
                                    core.rename_buf = local_buf.clone();
                                }
                                if r.lost_focus() {
                                    commit_rename = true;
                                }
                                if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                                    cancel_rename = true;
                                }
                            } else {
                                let name_label = egui::RichText::new(&name)
                                    .color(if is_active {
                                        ACCENT
                                    } else if visible {
                                        theme
                                    } else {
                                        dim
                                    })
                                    .strong();
                                let r = ui
                                    .add(
                                        egui::Label::new(name_label)
                                            .sense(egui::Sense::click())
                                            .truncate(),
                                    )
                                    .on_hover_cursor(egui::CursorIcon::PointingHand);
                                if r.clicked() {
                                    selected = true;
                                }
                                if r.double_clicked() {
                                    start_rename = true;
                                }
                                r.on_hover_text("Click to select • Double-click to rename");
                            }

                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    // Opacity percentage badge shown to the right of the slider
                                    let pct_text =
                                        egui::RichText::new(format!("{:.0}%", op * 100.0))
                                            .size(10.0)
                                            .color(if locked {
                                                ui.visuals().weak_text_color()
                                            } else {
                                                pal.chrome_text
                                            });
                                    ui.label(pct_text);
                                    let sr = ui.add_enabled(
                                        !locked,
                                        egui::Slider::new(&mut op, 0.0..=1.0).show_value(false),
                                    );
                                    op_changed = sr.changed();
                                    op_drag = sr.drag_started();
                                    sr.on_hover_text(format!("Opacity: {:.0}%", op * 100.0));
                                },
                            );
                        });
                    });

                let card_rect = card_resp.response.rect;
                if ui.rect_contains_pointer(card_rect) {
                    hover_target = Some(li);
                }

                // Draw active layer left amber accent stripe (Blender Outliner style)
                if is_active {
                    let stripe_rect = egui::Rect::from_min_max(
                        card_rect.left_top(),
                        egui::pos2(card_rect.left() + 3.0, card_rect.bottom()),
                    );
                    ui.painter().rect_filled(
                        stripe_rect,
                        egui::CornerRadius {
                            nw: RADIUS_CHIP,
                            sw: RADIUS_CHIP,
                            ne: 0,
                            se: 0,
                        },
                        ACCENT,
                    );
                }

                // Drag-and-drop indicator & target detection
                if is_dragging {
                    if let Some(pos) = pointer_pos {
                        if card_rect.expand2(egui::vec2(15.0, 3.0)).contains(pos) {
                            let above_in_ui = pos.y < card_rect.center().y;
                            if drag_from != Some(li) {
                                drop_target = Some((li, above_in_ui));

                                // Draw Blender-style insertion indicator line
                                let line_y = if above_in_ui {
                                    card_rect.top() - 2.0
                                } else {
                                    card_rect.bottom() + 2.0
                                };
                                let p = ui.painter();
                                p.line_segment(
                                    [
                                        egui::pos2(card_rect.left() + 4.0, line_y),
                                        egui::pos2(card_rect.right() - 4.0, line_y),
                                    ],
                                    egui::Stroke::new(2.5, ACCENT),
                                );
                                p.circle_filled(
                                    egui::pos2(card_rect.left() + 6.0, line_y),
                                    3.5,
                                    ACCENT,
                                );
                            }
                        }
                    }
                }

                if toggled {
                    core.history.record(snapshot_of(mesh));
                    mesh.layers[li].visible = !mesh.layers[li].visible;
                    needs_refresh = true;
                }
                if lock_toggled {
                    core.history.record(snapshot_of(mesh));
                    mesh.layers[li].locked = !mesh.layers[li].locked;
                    needs_refresh = true;
                    core.status = if mesh.layers[li].locked {
                        "Layer locked".to_string()
                    } else {
                        "Layer unlocked".to_string()
                    };
                }
                if selected {
                    mesh.active_layer = li;
                    let l = &mesh.layers[li];
                    core.status = format!(
                        "Editing {} ({})",
                        l.name,
                        if l.visible { "visible" } else { "hidden" }
                    );
                }
                if start_rename {
                    core.renaming = Some(li);
                    core.rename_buf = name.clone();
                    core.rename_grab_focus = true;
                }
                if cancel_rename {
                    core.renaming = None;
                    core.rename_grab_focus = false;
                }
                if commit_rename {
                    core.renaming = None;
                    core.rename_grab_focus = false;
                    if local_buf.trim().is_empty() {
                        local_buf = name.clone();
                    }
                    if local_buf != name {
                        core.history.record(snapshot_of(mesh));
                        mesh.layers[li].name = local_buf;
                        core.status = format!("Renamed layer to {}", mesh.layers[li].name);
                    }
                }
                if op_drag {
                    core.history.record(snapshot_of(mesh));
                }
                if op_changed {
                    mesh.layers[li].opacity = op;
                    needs_refresh = true;
                }
                if is_renaming {
                    core.rename_grab_focus = false;
                }
            }
        })
        .1
    };

    // --- Floating ghost card during drag -------------------------------------------
    // When a layer is being dragged, render a semi-transparent copy of its card
    // following the pointer so the user feels like they are truly holding the layer.
    if is_dragging {
        if let (Some(drag_idx), Some(ptr)) = (drag_from, pointer_pos) {
            if drag_idx < mesh.layers.len() {
                let ghost_layer = &mesh.layers[drag_idx];
                let ghost_name = ghost_layer.name.clone();
                let ghost_visible = ghost_layer.visible;
                let ghost_opacity = ghost_layer.opacity;
                egui::Area::new(egui::Id::new("layer_dnd_ghost"))
                    .order(egui::Order::Tooltip)
                    .fixed_pos(ptr + egui::vec2(14.0, -10.0))
                    .interactable(false)
                    .show(ui.ctx(), |ui| {
                        let pal = UiPalette::of(ui);
                        let ghost_bg = egui::Color32::from_rgba_unmultiplied(
                            pal.card_active.r(),
                            pal.card_active.g(),
                            pal.card_active.b(),
                            235,
                        );
                        egui::Frame::NONE
                            .fill(ghost_bg)
                            .stroke(egui::Stroke::new(1.5, ACCENT))
                            .corner_radius(egui::CornerRadius::same(RADIUS_CONTROL))
                            .inner_margin(egui::Margin::symmetric(8, 5))
                            .show(ui, |ui| {
                                ui.set_max_width(160.0);
                                ui.horizontal(|ui| {
                                    ui.spacing_mut().item_spacing.x = 6.0;
                                    // drag-handle dot accent
                                    let (r, _) = ui.allocate_exact_size(
                                        egui::vec2(8.0, 14.0),
                                        egui::Sense::hover(),
                                    );
                                    for dy in [-3.0f32, 0.0, 3.0] {
                                        ui.painter().circle_filled(
                                            r.center() + egui::vec2(0.0, dy),
                                            1.8,
                                            ACCENT,
                                        );
                                    }
                                    let vis_color = if ghost_visible {
                                        pal.chrome_text
                                    } else {
                                        pal.chrome_text_weak
                                    };
                                    ui.label(
                                        egui::RichText::new(&ghost_name)
                                            .color(pal.chrome_text)
                                            .strong()
                                            .size(12.0),
                                    );
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            ui.label(
                                                egui::RichText::new(format!(
                                                    "{:.0}%",
                                                    ghost_opacity * 100.0
                                                ))
                                                .color(vis_color)
                                                .size(10.0),
                                            );
                                        },
                                    );
                                });
                            });
                    });
            }
        }
    }
    // ---------------------------------------------------------------------------------

    if let Some(payload) = dropped {
        if let Some((target_li, above_in_ui)) = drop_target {
            if payload.from < mesh.layers.len()
                && target_li < mesh.layers.len()
                && payload.from != target_li
            {
                core.history.record(snapshot_of(mesh));
                let new_active = move_layer_relative(
                    &mut mesh.layers,
                    payload.from,
                    target_li,
                    above_in_ui,
                    mesh.active_layer,
                );
                mesh.active_layer = new_active;
                core.renaming = None;
                core.stroke = None;
                core.stroke_2d = None;
                needs_refresh = true;
                core.status = format!("Reordered layer {}", mesh.layers[new_active].name);
            }
        } else if let Some(to) = hover_target.filter(|&t| t != payload.from) {
            if payload.from < mesh.layers.len() && to < mesh.layers.len() {
                core.history.record(snapshot_of(mesh));
                let new_active =
                    reorder_layers(&mut mesh.layers, payload.from, to, mesh.active_layer);
                mesh.active_layer = new_active;
                core.renaming = None;
                core.stroke = None;
                core.stroke_2d = None;
                needs_refresh = true;
                core.status = format!("Reordered layer {}", mesh.layers[new_active].name);
            }
        }
    }

    if needs_refresh {
        core.needs_texture_upload = true;
        core.needs_material_upload = true;
        core.preview_gen += 1;
        ui.ctx().request_repaint();
    }

    // Flip X/Y mirror the active layer's painted texture in place. Each flip
    // records an undo snapshot and re-uploads the changed atlas.
    if flip_x || flip_y {
        core.history.record(snapshot_of(mesh));
        crate::io::flip_texture(&mut mesh.layers[active].texture, flip_x, flip_y);
        core.needs_texture_upload = true;
        core.preview_gen += 1;
        core.status = match (flip_x, flip_y) {
            (true, true) => "Flipped layer horizontally & vertically".to_string(),
            (true, false) => "Flipped layer horizontally".to_string(),
            _ => "Flipped layer vertically".to_string(),
        };
    }
}

fn draw_checkerboard(ui: &Ui, rect: egui::Rect, clip: egui::Rect) {
    let square = (rect.width() / 24.0).ceil().max(8.0);
    let pal = UiPalette::of(ui);
    let colors = [pal.checker_a, pal.checker_b];
    let painter = ui.painter_at(clip);
    let mut row = 0;
    let mut y = rect.top();
    while y < rect.bottom() {
        let h = square.min(rect.bottom() - y);
        let mut col = 0;
        let mut x = rect.left();
        while x < rect.right() {
            let w = square.min(rect.right() - x);
            let c = colors[(row + col) % 2];
            painter.rect_filled(
                egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(w, h)),
                0.0,
                c,
            );
            col += 1;
            x += square;
        }
        row += 1;
        y += square;
    }
}

/// Draws the mesh's UV layout (its islands/wireframe as seen in a UV editor)
/// over the preview rect. Boundary edges get a bright thick stroke; shared
/// interior edges stay thin and faded. `clip` keeps stray UV lines from
/// painting over the surrounding UI when the atlas overhangs the canvas.
fn draw_uv_overlay(ui: &mut Ui, rect: egui::Rect, clip: egui::Rect, core: &Core) {
    if !core.show_uv_overlay {
        return;
    }
    let Some(mesh) = core.mesh.as_ref() else {
        return;
    };
    if mesh.uvs.is_empty() || mesh.indices.is_empty() {
        return;
    }

    let mut edge_count: HashMap<(u32, u32), u32> = HashMap::new();
    for tri in mesh.indices.chunks_exact(3) {
        for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            let key = if a < b { (a, b) } else { (b, a) };
            *edge_count.entry(key).or_insert(0) += 1;
        }
    }

    let to_pos = |u: f32, v: f32| -> egui::Pos2 {
        egui::pos2(
            rect.left() + u * rect.width(),
            rect.top() + v * rect.height(),
        )
    };

    let painter = ui.painter_at(clip);
    let pal = UiPalette::of(ui);
    let (boundary_col, interior_col) = if pal.dark {
        (
            egui::Color32::from_rgb(255, 214, 96),
            egui::Color32::from_rgba_unmultiplied(120, 190, 255, 190),
        )
    } else {
        (
            egui::Color32::from_rgb(176, 122, 12),
            egui::Color32::from_rgba_unmultiplied(30, 110, 190, 200),
        )
    };
    let boundary_stroke = egui::Stroke::new(2.0, boundary_col);
    let interior_stroke = egui::Stroke::new(1.0, interior_col);

    for ((a, b), count) in &edge_count {
        let (u0, v0) = mesh.uvs[*a as usize];
        let (u1, v1) = mesh.uvs[*b as usize];
        painter.line_segment(
            [to_pos(u0, v0), to_pos(u1, v1)],
            if *count == 1 {
                boundary_stroke
            } else {
                interior_stroke
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A config saved *before* a set of actions existed encodes them as a
    /// missing-key map; loading must give those actions their default bindings
    /// while preserving explicit (including explicitly-unbound) entries.
    #[test]
    fn shortcuts_missing_actions_fall_back_to_defaults() {
        use serde::ser::SerializeMap;
        use serde::Serializer;
        let mut w = Vec::new();
        let mut se = rmp_serde::Serializer::new(&mut w);
        let mut map = se.serialize_map(Some(2)).unwrap();
        map.serialize_entry(
            ShortcutAction::Undo.serial(),
            &KeyBind::new(egui::Key::A, egui::Modifiers::NONE),
        )
        .unwrap();
        map.serialize_entry(
            ShortcutAction::ToggleOverlayBar.serial(),
            &KeyBind::unbound(),
        )
        .unwrap();
        map.end().unwrap();

        let sc: Shortcuts = rmp_serde::from_slice(&w).unwrap();
        // Explicit entries survive verbatim…
        assert_eq!(sc.get(ShortcutAction::Undo).label(), "A");
        assert!(!sc.get(ShortcutAction::ToggleOverlayBar).is_bound());
        // …and actions the map never mentioned get their default bindings.
        assert_eq!(sc.get(ShortcutAction::OpenModel).label(), "Ctrl+O");
        assert_eq!(sc.get(ShortcutAction::SelectBrush).label(), "B");
        // An unknown action id is skipped, not fatal.
        let mut w2 = Vec::new();
        let mut se2 = rmp_serde::Serializer::new(&mut w2);
        let mut map2 = se2.serialize_map(Some(1)).unwrap();
        map2.serialize_entry("obsolete_action", &KeyBind::unbound())
            .unwrap();
        map2.end().unwrap();
        let sc2: Shortcuts = rmp_serde::from_slice(&w2).unwrap();
        assert_eq!(sc2, Shortcuts::default());
    }

    #[test]
    fn ui_memory_round_trips() {
        let dock = default_dock();
        let mem = UiMemory {
            dock,
            panel_visible: vec![true, true, false, true, true, false],
            active_tool: 2,
            channels: [true, false, true, false, false, true],
            brush_size: 42.0,
            brush_hardness: 0.25,
            brush_opacity: 0.75,
            brush_spacing: 12.0,
            brush_color: [12, 34, 56, 255],
            show_uv_overlay: false,
            show_uv_checker_3d: true,
            show_uv_grid_3d: false,
            brush_shape: 2,
            brush_rotation: 0.5,
            brush_flip_x: true,
            brush_flip_y: false,
            brush_pattern_lock: 1,
            brush_texture_scale: 1.75,
            brush_texture_locked: true,
            material: crate::render::Material {
                roughness: 0.3,
                metallic: 1.0,
                emissive: 0.5,
                ambient_occlusion: 0.7,
                sun_intensity: 3.5,
                sun_color: [0.9, 0.8, 0.7],
                sun_enabled: false,
                sun_elevation: 45.0,
                sun_azimuth: 180.0,
                env_rotation: 90.0,
                sky_color: [0.55, 0.44, 0.33],
                env_intensity: 0.4,
                exposure: 1.6,
                fill_intensity: 0.3,
                parallax: 0.03,
            },
            show_tool_strip: false,
            theme: 2,
            shortcuts: Shortcuts::default(),
            camera: Some(CameraState {
                eye: [1.0, 2.0, 3.0],
                target: [0.5, 0.5, 0.5],
                radius: 4.25,
                ortho: false,
            }),
        };

        let bytes = rmp_serde::to_vec(&mem).expect("serialize");
        let back: UiMemory = rmp_serde::from_slice(&bytes).expect("deserialize");

        assert_eq!(
            back.dock.iter_all_tabs().count(),
            mem.dock.iter_all_tabs().count()
        );
        assert_eq!(back.panel_visible, mem.panel_visible);
        assert_eq!(back.active_tool, 2);
        assert_eq!(back.channels, mem.channels);
        assert_eq!(back.brush_size, 42.0);
        assert_eq!(back.brush_spacing, 12.0);
        assert!(!back.show_uv_overlay);
        assert!(back.show_uv_checker_3d);
        assert!(!back.show_uv_grid_3d);
        assert_eq!(back.brush_shape, 2);
        assert_eq!(back.brush_rotation, 0.5);
        assert!(back.brush_flip_x);
        assert!(!back.brush_flip_y);
        assert_eq!(back.brush_pattern_lock, 1);
        assert_eq!(back.material, mem.material);
        assert_eq!(back.material.roughness, 0.3);
        assert_eq!(back.material.exposure, 1.6);
        assert_eq!(back.camera.unwrap().radius, 4.25);
        assert_eq!(back.theme, 2);
        assert_eq!(back.shortcuts, mem.shortcuts);
        assert!(back
            .shortcuts
            .get(ShortcutAction::Undo)
            .label()
            .to_ascii_lowercase()
            .contains("ctrl"));
        assert!(!back
            .shortcuts
            .get(ShortcutAction::ToggleOverlayBar)
            .is_bound());

        // The fixup path must keep every panel present.
        let all: Vec<Panel> = back.dock.iter_all_tabs().map(|(_, tab)| *tab).collect();
        for panel in Panel::ALL {
            assert!(
                all.contains(&panel),
                "panel {panel:?} missing after restore"
            );
        }
    }

    fn snap(r: u8) -> LayerStackSnapshot {
        LayerStackSnapshot {
            active_layer: 0,
            layers: vec![LayerSnapshot {
                name: "Layer 1".to_string(),
                visible: true,
                locked: false,
                opacity: 1.0,
                blend: crate::io::BlendMode::Normal,
                roughness: 0.55,
                metallic: 0.0,
                emissive: 0.0,
                ambient_occlusion: 1.0,
                height: 0.0,
                bump_strength: 2.0,
                clearcoat: 0.0,
                clearcoat_roughness: 0.6,
                specular_ior: 1.5,
                emissive_color: [1.0, 1.0, 1.0],
                texture: crate::io::TextureData {
                    width: 2,
                    height: 2,
                    rgba: vec![r; 16],
                },
            }],
        }
    }

    #[test]
    fn reorder_layers_moves_and_keeps_active_valid() {
        let mut layers = vec![
            crate::io::Layer::blank("a", 2, 2, [0, 0, 0, 0]),
            crate::io::Layer::blank("b", 2, 2, [0, 0, 0, 0]),
            crate::io::Layer::blank("c", 2, 2, [0, 0, 0, 0]),
            crate::io::Layer::blank("d", 2, 2, [0, 0, 0, 0]),
        ];
        // Move top-of-storage (index 3) down-visually → land on row 0: d,a,b,c.
        let na = reorder_layers(&mut layers, 3, 0, 3);
        let names: Vec<_> = layers.iter().map(|l| l.name.clone()).collect();
        assert_eq!(names, ["d", "a", "b", "c"]);
        assert_eq!(na, 0); // the moved layer is active, now at 0

        // Active elsewhere shifts with the move.
        let mut layers2 = vec![
            crate::io::Layer::blank("a", 2, 2, [0, 0, 0, 0]),
            crate::io::Layer::blank("b", 2, 2, [0, 0, 0, 0]),
            crate::io::Layer::blank("c", 2, 2, [0, 0, 0, 0]),
            crate::io::Layer::blank("d", 2, 2, [0, 0, 0, 0]),
        ];
        // Move c (index 2) to land on row 1 → a,c,b,d; active was b (1).
        let na = reorder_layers(&mut layers2, 2, 1, 1);
        let names2: Vec<_> = layers2.iter().map(|l| l.name.clone()).collect();
        assert_eq!(names2, ["a", "c", "b", "d"]);
        assert_eq!(na, 2); // b slid from index 1 to 2

        // Move up-in-storage a (0) onto row 3 (top of display) → b,c,d,a; active a → 3.
        let mut layers3 = vec![
            crate::io::Layer::blank("a", 2, 2, [0, 0, 0, 0]),
            crate::io::Layer::blank("b", 2, 2, [0, 0, 0, 0]),
            crate::io::Layer::blank("c", 2, 2, [0, 0, 0, 0]),
            crate::io::Layer::blank("d", 2, 2, [0, 0, 0, 0]),
        ];
        let na = reorder_layers(&mut layers3, 0, 3, 0);
        let names3: Vec<_> = layers3.iter().map(|l| l.name.clone()).collect();
        assert_eq!(names3, ["b", "c", "d", "a"]);
        assert_eq!(na, 3);
    }

    #[test]
    fn move_layer_relative_tests() {
        let mut layers = vec![
            crate::io::Layer::blank("a", 2, 2, [0, 0, 0, 0]),
            crate::io::Layer::blank("b", 2, 2, [0, 0, 0, 0]),
            crate::io::Layer::blank("c", 2, 2, [0, 0, 0, 0]),
        ];
        // Move c (storage 2) below a (storage 0) in UI -> storage index 0
        let new_active = move_layer_relative(&mut layers, 2, 0, false, 2);
        let names: Vec<_> = layers.iter().map(|l| l.name.clone()).collect();
        assert_eq!(names, ["c", "a", "b"]);
        assert_eq!(new_active, 0);

        // Move c (storage 0) above b (storage 2) in UI -> storage index 2
        let new_active = move_layer_relative(&mut layers, 0, 2, true, 0);
        let names: Vec<_> = layers.iter().map(|l| l.name.clone()).collect();
        assert_eq!(names, ["a", "b", "c"]);
        assert_eq!(new_active, 2);

        // Move c (storage 2) below b (storage 1) in UI -> storage index 1
        let new_active = move_layer_relative(&mut layers, 2, 1, false, 0);
        let names: Vec<_> = layers.iter().map(|l| l.name.clone()).collect();
        assert_eq!(names, ["a", "c", "b"]);
        assert_eq!(new_active, 0);
    }

    #[test]
    fn snapshot_round_trips_locked_and_rename() {
        let mut mesh = crate::io::MeshData::uv_sphere(0.5, 3, 5);
        mesh.layers
            .push(crate::io::Layer::blank("base", 4, 4, [0, 0, 0, 0]));
        mesh.layers[0].locked = true;
        mesh.layers[0].name = "armor".to_string();

        let snap = snapshot_of(&mesh);
        let snap_layer = &snap.layers[0];
        assert!(snap_layer.locked);
        assert_eq!(snap_layer.name, "armor");

        // Rebuild a layer from the snapshot exactly like restore_snapshot does
        // (minus the GPU upload bits), and confirm locked + name survive.
        let rebuilt: Vec<crate::io::Layer> = snap
            .layers
            .iter()
            .map(|l| crate::io::Layer {
                name: l.name.clone(),
                visible: l.visible,
                locked: l.locked,
                opacity: l.opacity,
                blend: l.blend,
                roughness: l.roughness,
                metallic: l.metallic,
                emissive: l.emissive,
                ambient_occlusion: l.ambient_occlusion,
                height: l.height,
                bump_strength: l.bump_strength,
                clearcoat: l.clearcoat,
                clearcoat_roughness: l.clearcoat_roughness,
                specular_ior: l.specular_ior,
                emissive_color: l.emissive_color,
                texture: l.texture.clone(),
            })
            .collect();
        assert!(rebuilt[0].locked);
        assert_eq!(rebuilt[0].name, "armor");
        assert_eq!(rebuilt[0].texture.width, mesh.layers[0].texture.width);
        assert_eq!(rebuilt[0].texture.rgba, mesh.layers[0].texture.rgba);
    }

    #[test]
    fn edit_history_undo_redo_round_trip() {
        let mut h = EditHistory::new(8);
        assert!(!h.can_undo());
        h.record(snap(1));
        h.record(snap(2));
        assert!(h.can_undo());

        // Undo twice: restores 2 then 1, pushes current onto redo.
        let cur = snap(3);
        let s1 = h.undo(cur).expect("undo 1");
        assert_eq!(s1.layers[0].texture.rgba, snap(2).layers[0].texture.rgba);
        assert!(h.can_redo());

        let cur = snap(2);
        let s2 = h.undo(cur).expect("undo 2");
        assert_eq!(s2.layers[0].texture.rgba, snap(1).layers[0].texture.rgba);
        assert!(h.can_redo());

        // Redo restores the most recent undone state.
        let cur = snap(1);
        let r1 = h.redo(cur).expect("redo 1");
        assert_eq!(r1.layers[0].texture.rgba, snap(2).layers[0].texture.rgba);
        assert!(h.can_redo());

        let cur = snap(2);
        let r2 = h.redo(cur).expect("redo 2");
        assert_eq!(r2.layers[0].texture.rgba, snap(3).layers[0].texture.rgba);
        assert!(!h.can_redo());
    }

    #[test]
    fn record_clears_redo_and_caps_undo() {
        let mut h = EditHistory::new(3);
        h.record(snap(1));
        h.record(snap(2));
        h.record(snap(3));
        let _ = h.undo(snap(9));
        assert!(h.can_redo());

        // A new record must clear the redo stack.
        h.record(snap(4));
        assert!(!h.can_redo());

        // Cap: only the last `limit` states survive. After the 5 extra records
        // plus the earlier manipulations, popping everything left exactly 3.
        for i in 1..=5 {
            h.record(snap(i));
        }
        while h.can_undo() {
            h.undo(snap(0));
        }
        assert!(!h.can_undo());
        assert_eq!(h.redo.len(), 3);
        h.clear();
        assert!(!h.can_undo());
        assert!(!h.can_redo());
    }

    #[test]
    fn snapshot_of_clones_texture_state() {
        let mesh =
            crate::io::MeshData::uv_sphere(0.6, 4, 6).with_texture(crate::io::default_albedo());
        let s = snapshot_of(&mesh);
        assert_eq!(s.layers.len(), mesh.layers.len());
        assert_eq!(s.active_layer, mesh.active_layer);
        let src = &s.layers[0].texture;
        let dst = &mesh.layers[0].texture;
        assert_eq!(src.width, dst.width);
        assert_eq!(src.height, dst.height);
        assert_eq!(src.rgba, dst.rgba);
    }

    /// The nav-gizmo orthographic camera must unproject viewport rays that
    /// actually sweep the model: moving the pointer across the viewport must
    /// move the raycast hit accordingly (a frozen hit here would mean a brush
    /// that "doesn't move" in orthographic views).
    #[test]
    fn ortho_view_ray_sweeps_the_model() {
        use crate::io::{default_albedo, MeshData};
        use crate::paint::mesh_raycast;
        let mesh = MeshData::uv_sphere(0.6, 12, 16).with_texture(default_albedo());
        let mut cam = Camera::new(1920.0 / 1080.0);
        let c = mesh_center(&mesh);
        let r = mesh_bounds_radius(&mesh, c);
        cam.fit(c, r);
        cam.look_along(glam::Vec3::NEG_Z);

        let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1920.0, 1080.0));
        let hits: Vec<glam::Vec3> = (0..=10)
            .map(|i| {
                // Sweep the central half of the viewport (inside the sphere's
                // screen span), so every column hits the surface.
                let nx = -0.5 + i as f32 / 10.0;
                let (o, d) = cam.ray(nx, 0.0);
                mesh_raycast(&mesh, o, d).unwrap().position
            })
            .collect();
        let _ = rect;
        // A frozen sweep here (all hits identical) is the "brush doesn't move"
        // bug; breaking it means the columns actually spread. They must spread
        // along the camera's geometric screen-right vector and nowhere else.
        let f = (cam.target - cam.eye).normalize();
        let right = f.cross(cam.up).normalize();
        let drift = hits[0] - hits[10];
        let along = drift.dot(right).abs();
        assert!(along > 0.5, "ortho ray sweep stalled: drift = {drift:?}");
        assert!(
            (drift - right * drift.dot(right)).length() < 1e-3,
            "ortho sweep not purely horizontal: drift = {drift:?}"
        );
        for pair in hits.windows(2) {
            assert_ne!(pair[0], pair[1], "ortho sweep froze mid-column");
        }
    }
}
