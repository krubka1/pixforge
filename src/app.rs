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
}

impl Panel {
    const ALL: [Panel; 6] = [
        Panel::Viewport,
        Panel::Channels,
        Panel::Texture,
        Panel::Layers,
        Panel::Lighting,
        Panel::Brushes,
    ];

    fn title(&self) -> &'static str {
        match self {
            Panel::Viewport => "3D Viewport",
            Panel::Channels => "Channels",
            Panel::Texture => "Texture Editor",
            Panel::Layers => "Layers",
            Panel::Lighting => "Lighting",
            Panel::Brushes => "Brushes",
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
    brush_size: f32,
    brush_hardness: f32,
    brush_opacity: f32,
    /// Distance (px) between dab centers along a stroke; `<= 0` = single dab per frame.
    brush_spacing: f32,
    /// RGBA brush color (persisted; used by Paint/Fill, set by Pick).
    brush_color: [u8; 4],
    /// Loaded environment/skybox map (HDRI or plain equirect photo; session-only);
    /// `None` = analytic sky.
    env_path: Option<String>,
    /// Brush footprint + optional texture stamp. The sprite is session-only;
    /// shape/rotation/flip are persisted via `UiMemory`.
    brush_style: crate::paint::BrushStyle,
    /// Folder-backed brush library (built-ins + files dropped in `brushes/`).
    brushes: crate::brushes::BrushLibrary,
    /// Cached thumbnail textures keyed by `brushes.entries` index; rebuilt
    /// (keyed off the library signature) when the folder contents change.
    brush_thumbs: HashMap<usize, TextureHandle>,
    brush_thumb_sig: String,
    /// Selected category filter ("All" or a category name) for the panel.
    brush_filter: String,
    /// PBR material (roughness/metallic/emissive/AO + lighting), persisted.
    material: crate::render::Material,
    /// Runtime state of the brush stroke in progress.
    stroke: Option<StrokeState>,
    /// Cached egui texture of the active brush sprite, drawn as the viewport
    /// cursor for texture brushes (keyed by a content hash of the sprite).
    brush_preview: Option<(u64, TextureHandle)>,
    /// Undo/redo history of full-texture snapshots.
    history: EditHistory,
    /// Target atlas resolution (longest side) for Resize / Blank (not persisted).
    atlas_res: u32,
    /// Right-click brush menu popped up over the viewport (not persisted).
    brush_menu_open: bool,
    brush_menu_pos: Option<egui::Pos2>,
    /// Lazily-loaded eyedropper icon (lucide pipette, ISC licensed).
    pick_icon: Option<TextureHandle>,
    /// Set when the CPU atlas changed and must be re-uploaded before the next render.
    needs_texture_upload: bool,
    /// Cached egui copy of the current albedo texture, rebuilt when `preview_gen` bumps.
    texture_preview: Option<PreviewTexture>,
    preview_gen: u64,
    show_uv_overlay: bool,
    /// 3D viewport UV checkerboard / grid overlays (shader-driven).
    show_uv_checker_3d: bool,
    show_uv_grid_3d: bool,
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
    /// Stroke in progress inside the 2D texture preview (screen-space paint
    /// positions), kept separate from the 3D viewport's `stroke`.
    stroke_2d: Option<StrokeState>,
    /// 2D texture-preview camera (pan + zoom), image-editor style.
    canvas2d: Canvas2D,
    /// Camera view to restore on the first frame (from a previous session).
    restore_view: Option<(glam::Vec3, glam::Vec3, f32)>,
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
    opacity: f32,
    blend: crate::io::BlendMode,
    roughness: f32,
    metallic: f32,
    emissive: f32,
    ambient_occlusion: f32,
    height: f32,
    bump_strength: f32,
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
                opacity: l.opacity,
                blend: l.blend,
                roughness: l.roughness,
                metallic: l.metallic,
                emissive: l.emissive,
                ambient_occlusion: l.ambient_occlusion,
                height: l.height,
                bump_strength: l.bump_strength,
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
    material: crate::render::Material,
    show_tool_strip: bool,
    camera: Option<CameraState>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct CameraState {
    eye: [f32; 3],
    target: [f32; 3],
    radius: f32,
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
/// Quick-pick palette shown in the right-click (RMB) brush menu.
const PRESET_COLORS: [[u8; 3]; 12] = [
    [255, 255, 255], // white
    [0, 0, 0],       // black
    [90, 160, 255],  // pixforge blue
    [214, 65, 65],   // red
    [70, 160, 92],   // green
    [61, 139, 214],  // blue
    [224, 161, 60],  // amber
    [160, 90, 210],  // purple
    [214, 160, 160], // pink
    [90, 90, 90],    // gray
    [246, 241, 232], // cream
    [60, 200, 200],  // teal
];

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
            panel_visible: vec![true, true, true, false, true, true],
            active_tool: 0,
            channels: [true, true, false, false, false, false],
            brush_size: 24.0,
            brush_hardness: 0.5,
            brush_opacity: 1.0,
            brush_spacing: 6.0,
            brush_color: [90, 160, 255, 255],
            env_path: None,
            brush_style: crate::paint::BrushStyle::default(),
            brushes: crate::brushes::BrushLibrary::new(brushes_folder()),
            brush_thumbs: HashMap::new(),
            brush_thumb_sig: String::new(),
            brush_filter: "All".to_string(),
            material: crate::render::Material::default(),
            stroke: None,
            brush_preview: None,
            history: EditHistory::new(24),
            atlas_res: 512,
            brush_menu_open: false,
            brush_menu_pos: None,
            pick_icon: None,
            needs_texture_upload: false,
            needs_material_upload: false,
            last_material_upload: std::time::Instant::now(),
            vp_scale: 1.0,
            texture_preview: None,
            preview_gen: 1,
            show_uv_overlay: true,
            show_uv_checker_3d: false,
            show_uv_grid_3d: false,
            show_tool_strip: true,
            tool_strip_anim: 1.0,
            show_brush_picker: true,
            brush_picker_anim: 1.0,
            show_vp_overlay_bar: true,
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
            core.brush_size = mem.brush_size;
            core.brush_hardness = mem.brush_hardness;
            core.brush_opacity = mem.brush_opacity;
            core.brush_spacing = mem.brush_spacing;
            core.brush_color = mem.brush_color;
            core.show_uv_overlay = mem.show_uv_overlay;
            core.show_uv_checker_3d = mem.show_uv_checker_3d;
            core.show_uv_grid_3d = mem.show_uv_grid_3d;
            core.brush_style.shape = crate::paint::BrushShape::ALL
                .get(mem.brush_shape as usize)
                .copied()
                .unwrap_or(crate::paint::BrushShape::Round);
            core.brush_style.rotation = mem.brush_rotation;
            core.brush_style.flip_x = mem.brush_flip_x;
            core.brush_style.flip_y = mem.brush_flip_y;
            core.material = mem.material;
            core.show_tool_strip = mem.show_tool_strip;
            core.tool_strip_anim = if core.show_tool_strip { 1.0 } else { 0.0 };
            // Hidden panels were removed from the dock when they were unchecked;
            // re-apply that so a restored layout doesn't resurrect closed tabs.
            for (i, panel) in Panel::ALL.iter().enumerate() {
                if !core.panel_visible.get(i).copied().unwrap_or(true) && *panel != Panel::Viewport {
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
                ));
                core.needs_fit = false;
            }
        }

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
                    self.core.status =
                        format!("Exported .glb with baked layers ({}) to {path}", mesh.layers.len());
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
            brush_size: self.core.brush_size,
            brush_hardness: self.core.brush_hardness,
            brush_opacity: self.core.brush_opacity,
            brush_spacing: self.core.brush_spacing,
            brush_color: self.core.brush_color,
            show_uv_overlay: self.core.show_uv_overlay,
            show_uv_checker_3d: self.core.show_uv_checker_3d,
            show_uv_grid_3d: self.core.show_uv_grid_3d,
            brush_shape: self.core.brush_style.shape as u8,
            brush_rotation: self.core.brush_style.rotation,
            brush_flip_x: self.core.brush_style.flip_x,
            brush_flip_y: self.core.brush_style.flip_y,
            material: self.core.material,
            show_tool_strip: self.core.show_tool_strip,
            camera: self.core.viewport.as_ref().map(|vp| CameraState {
                eye: vp.camera.eye.into(),
                target: vp.camera.target.into(),
                radius: vp.camera.radius,
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
        if visible && panel != Panel::Viewport {
            // Re-open the panel by docking it into the first available leaf.
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
    let tangent = {
        let t = eye_dir.cross(world);
        let t = if t.length_squared() < 1e-6 {
            let up = if eye_dir.y.abs() > 0.9 {
                glam::Vec3::X
            } else {
                glam::Vec3::Y
            };
            eye_dir.cross(up)
        } else {
            t
        };
        t.normalize_or_zero()
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
    let _ = main.split_below(lighting, 0.5, vec![Panel::Brushes]);
    let [_old, right] = main.split_right(NodeIndex::root(), 0.28, vec![Panel::Texture]);
    let _ = main.split_below(right, 0.5, vec![Panel::Layers]);
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

impl eframe::App for PixForgeApp {
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        [0.09, 0.09, 0.11, 1.0]
    }

    fn on_exit(&mut self) {
        self.save_ui_memory();
    }

    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
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
                        ui.add_enabled(false, egui::Button::new("Preferences"));
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
    /// while an egui text widget has keyboard focus (e.g. typing in a field).
    fn handle_shortcuts(&mut self, ui: &mut Ui) {
        if ui.ctx().egui_wants_keyboard_input() {
            return;
        }
        let undo_cmd = egui::Modifiers::COMMAND;
        let redo_cmd = egui::Modifiers::COMMAND | egui::Modifiers::SHIFT;
        let mut do_undo = false;
        let mut do_redo = false;
        ui.ctx().input_mut(|i| {
            do_undo = i.consume_key(undo_cmd, egui::Key::Z);
            do_redo = (i.modifiers.command && i.modifiers.shift && i.key_pressed(egui::Key::Z))
                || i.consume_key(redo_cmd, egui::Key::Z)
                || i.consume_key(undo_cmd, egui::Key::Y);
        });
        if do_undo {
            self.undo();
        }
        if do_redo {
            self.redo();
        }
    }

    fn undo(&mut self) {
        let current = snapshot_of_current(&self.core);
        if let Some(snap) = self.core.history.undo(current) {
            restore_snapshot(&mut self.core, snap);
            let left = self.core.history.can_undo();
            self.core.status = if left {
                "Undo".to_string()
            } else {
                "Undo — history empty".to_string()
            };
        }
    }

    fn redo(&mut self) {
        let current = snapshot_of_current(&self.core);
        if let Some(snap) = self.core.history.redo(current) {
            restore_snapshot(&mut self.core, snap);
            let left = self.core.history.can_redo();
            self.core.status = if left {
                "Redo".to_string()
            } else {
                "Redo — nothing to redo".to_string()
            };
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
                opacity: l.opacity,
                blend: l.blend,
                roughness: l.roughness,
                metallic: l.metallic,
                emissive: l.emissive,
                ambient_occlusion: l.ambient_occlusion,
                height: l.height,
                bump_strength: l.bump_strength,
                texture: l.texture,
            })
            .collect();
        mesh.active_layer = snap.active_layer.min(mesh.layers.len().saturating_sub(1));
    }
    core.stroke = None;
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
            if let Some(m) = core.mesh.as_mut() {
                m.dirty = None;
            }
        } else {
            core.needs_texture_upload = true;
        }
    } else {
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
            Panel::Viewport => viewport_ui(ui, core),
            Panel::Channels => channels_ui(ui, core),
            Panel::Texture => texture_ui(ui, core),
            Panel::Layers => layers_ui(ui, core),
            Panel::Lighting => lighting_ui(ui, core),
            Panel::Brushes => brushes_ui(ui, core),
        }
    }
}

fn viewport_ui(ui: &mut Ui, core: &mut Core) {
    let full_rect = ui.max_rect();

    // Per-panel T toggle: with the pointer anywhere in the viewport, T flips
    // the 3D T-bar (the Texture preview consumes T separately when the pointer
    // hovers *its* canvas + brush picker).
    if ui.rect_contains_pointer(full_rect)
        && ui.ctx().input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::T))
    {
        core.show_tool_strip = !core.show_tool_strip;
        core.status = if core.show_tool_strip {
            "In-viewport tools: on (T to toggle)".to_string()
        } else {
            "In-viewport tools: off (T to toggle)".to_string()
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
        let dt = ui.input(|i| i.stable_dt).clamp(0.0, 0.1) as f32;
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

    // The horizontal overlay bar (UV checker / grid toggles) + its visibility
    // button (top-right pill). The bar sits pinned, left of the button, and is
    // ONLY ever shown while explicitly pinned — never on hover. Its
    // right-anchored box is reserved up front so the pointer gating below
    // excludes it from orbit/paint/zoom.
    let vp_bar_anchor = egui::Rect::from_min_size(
        egui::pos2(full_rect.right() - 360.0, full_rect.top() + 8.0),
        egui::vec2(360.0 - 44.0, 40.0),
    );
    let vp_toggle_rect = egui::Rect::from_min_size(
        egui::pos2(full_rect.right() - 36.0, full_rect.top() + 8.0),
        egui::vec2(28.0, 28.0),
    );
    let vp_bar_active = core.show_vp_overlay_bar
        && ui
            .input(|i| i.pointer.hover_pos())
            .is_some_and(|p| vp_bar_anchor.contains(p));
    let vp_toggle_active = ui
        .input(|i| i.pointer.hover_pos())
        .is_some_and(|p| vp_toggle_rect.contains(p));

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
    if let Some((eye, target, radius)) = core.restore_view.take() {
        let vp = core.viewport.as_mut().unwrap();
        vp.camera.eye = eye;
        vp.camera.target = target;
        vp.camera.radius = radius;
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
        && !vp_toggle_active;

    if core.needs_fit {
        let vp = core.viewport.as_mut().unwrap();
        vp.camera.fit(core.center, core.bounds_radius);
        core.needs_fit = false;
    }

    let (delta, scroll, m_middle, shift, f_pressed) = ui.input(|i| {
        (
            i.pointer.delta(),
            i.smooth_scroll_delta,
            i.pointer.middle_down(),
            i.modifiers.shift,
            i.key_pressed(egui::Key::F),
        )
    });

    if f_pressed {
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
            core.brush_size = (core.brush_size + amount * 0.8).clamp(1.0, 300.0);
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
    // doesn't also paint on the model underneath.
    if hovered && !navigating && !core.brush_menu_open && ui.input(|i| i.pointer.primary_down()) {
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
                                core.history.record(snapshot_of(mesh));
                                core.stroke = Some(StrokeState {
                                    last: pos,
                                    start: pos,
                                    last_dab: pos,
                                    acc: 0.0,
                                    next_t: 0.0,
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
                                let st = core
                                    .stroke
                                    .as_mut()
                                    .expect("stroke recorded right above");
                                let spacing = if core.brush_spacing > 0.0 {
                                    core.brush_spacing
                                } else {
                                    (core.brush_size / 2.0).max(1.0)
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
                                let color = if core.active_tool == 0 || core.active_tool == 4 {
                                    core.brush_color
                                } else {
                                    [0, 0, 0, 0]
                                };
                                for dab in dabs {
                                    let (dx, dy) = viewport_ndc(dab.x, dab.y, rect);
                                    let (o, d) = vp.camera.ray(dx, dy);
                                    if let Some(hi) = crate::paint::mesh_raycast(mesh, o, d) {
                                        let world_r = screen_to_world_radius(
                                            &vp.camera,
                                            hi.position,
                                            core.brush_size,
                                            rect,
                                            w,
                                            h,
                                        );
                                        if is_rect {
                                            crate::paint::apply_stamp_rect(
                                                mesh,
                                                hi.position,
                                                world_r,
                                                world_r,
                                                o,
                                                d,
                                                color,
                                                core.brush_opacity,
                                                core.brush_hardness,
                                                mode,
                                            );
                                        } else {
                                            crate::paint::apply_stamp_with(
                                                mesh,
                                                hi.position,
                                                world_r,
                                                o,
                                                d,
                                                color,
                                                core.brush_opacity,
                                                core.brush_hardness,
                                                mode,
                                                &core.brush_style,
                                            );
                                        }
                                        painted = true;
                                    }
                                }
                            }
                            2 => {
                                crate::paint::fill_region(
                                    mesh,
                                    hit.triangle,
                                    core.brush_color,
                                    core.brush_opacity,
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
                core.brush_color = c;
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
        let due = core.stroke.is_none()
            || core.last_material_upload.elapsed().as_millis() as u64 >= 30;
        if due {
            let material_map = core.mesh.as_ref().and_then(|m| m.flattened_material_atlas());
            core.renderer.update_material_map(material_map.as_ref());
            let height_map = core.mesh.as_ref().and_then(|m| m.flattened_height_atlas());
            core.renderer.update_height_map(height_map.as_ref());
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
    if let Some(sprite) = core.brush_style.sprite.as_ref() {
        let sig = sprite_sig(sprite);
        core.renderer.set_brush_sprite(sig, sprite);
    }
    core.renderer.brush_overlay = {
        // 0 = round, 1 = square, 2 = diamond, 3 = texture-sprite. Rect tools
        // stamp a square footprint, so they get the square mask regardless of
        // the brush shape; a texture shape with no sprite loads no mask (the
        // flat screen-space fallback cursor remains).
        let overlay_shape: Option<u32> = match core.active_tool {
            4 => Some(1),
            0 | 1 => match core.brush_style.shape {
                crate::paint::BrushShape::Round => Some(0),
                crate::paint::BrushShape::Square => Some(1),
                crate::paint::BrushShape::Diamond => Some(2),
                crate::paint::BrushShape::Texture if core.brush_style.sprite.is_some() => Some(3),
                crate::paint::BrushShape::Texture => None,
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
                        core.brush_size,
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
                    );
                    let shape = overlay_shape.expect("checked just above");
                    crate::render::BrushOverlay {
                        center: hit.position,
                        axis_u,
                        axis_v,
                        radius: r,
                        shape,
                        color: [
                            core.brush_color[0] as f32 / 255.0,
                            core.brush_color[1] as f32 / 255.0,
                            core.brush_color[2] as f32 / 255.0,
                            0.23,
                        ],
                        rotation: core.brush_style.rotation,
                        flip_x: core.brush_style.flip_x,
                        flip_y: core.brush_style.flip_y,
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
    vp_overlay_toggle(ui, core, full_rect);

    // Brush preview shape: a fixed-size ring in screen pixels matching the brush
    // radius (Paint/Eraser) or a square outline (Rect). Shift+wheel
    // in the viewport resizes it.
    if (core.active_tool == 0
        || core.active_tool == 1
        || core.active_tool == 4)
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
                        core.brush_style.shape,
                        crate::paint::BrushShape::Round
                            | crate::paint::BrushShape::Square
                            | crate::paint::BrushShape::Diamond
                    )
                    || (matches!(
                        core.brush_style.shape,
                        crate::paint::BrushShape::Texture
                    ) && core.brush_style.sprite.is_some()))
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
            let screen_r = core.brush_size;
            let painting = core.active_tool == 0 || core.active_tool == 4;
            let (fill, stroke, dot) = if painting {
                (
                    egui::Color32::from_rgba_unmultiplied(
                        core.brush_color[0],
                        core.brush_color[1],
                        core.brush_color[2],
                        60,
                    ),
                    egui::Color32::from_rgba_unmultiplied(
                        core.brush_color[0],
                        core.brush_color[1],
                        core.brush_color[2],
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
                match core.brush_style.shape {
                    crate::paint::BrushShape::Round => {
                        ui.painter().circle_filled(pos, screen_r, fill);
                        ui.painter()
                            .circle_stroke(pos, screen_r, egui::Stroke::new(1.5, stroke));
                    }
                    crate::paint::BrushShape::Square => {
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
                    crate::paint::BrushShape::Texture => {
                        // Show the actual brush sprite (shape + pattern) instead
                        // of a plain square outline.
                        let Some(sprite) = &core.brush_style.sprite else {
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
                        let sig = sprite_sig(sprite);
                        if core
                            .brush_preview
                            .as_ref()
                            .map(|(s, _)| *s != sig)
                            .unwrap_or(true)
                        {
                            let img = egui::ColorImage::new(
                                [sprite.width as usize, sprite.height as usize],
                                sprite
                                    .rgba
                                    .chunks_exact(4)
                                    .map(|p| {
                                        egui::Color32::from_rgba_unmultiplied(
                                            p[0],
                                            p[1],
                                            p[2],
                                            p[3],
                                        )
                                    })
                                    .collect(),
                            );
                            let handle = ui.ctx().load_texture(
                                format!("brush_preview_{:016x}", sig),
                                img,
                                egui::TextureOptions::LINEAR,
                            );
                            core.brush_preview = Some((sig, handle));
                        }
                        let handle = core.brush_preview.as_ref().unwrap().1.clone();
                        let aspect = (sprite.height.max(1) as f32)
                            / (sprite.width.max(1) as f32);
                        let hw = screen_r;
                        let hh = screen_r * aspect;
                        let (sr, cr) = core.brush_style.rotation.sin_cos();
                        let flip_x = core.brush_style.flip_x;
                        let flip_y = core.brush_style.flip_y;
                        // Sprite mesh with applied rotation + flips, drawn first
                        // scaled up by the stroke width to leave an outline rim.
                        let sprite_mesh = |scale: f32, color: egui::Color32| {
                            let mut m = egui::Mesh::default();
                            m.texture_id = handle.id();
                            let corners = [
                                (1.0, 1.0),
                                (1.0, -1.0),
                                (-1.0, -1.0),
                                (-1.0, 1.0),
                            ];
                            let uvs = [
                                (1.0, 1.0),
                                (1.0, 0.0),
                                (0.0, 0.0),
                                (0.0, 1.0),
                            ];
                            for (i, (fx, fy)) in corners.into_iter().enumerate() {
                                let (mut x, mut y) = (fx * hw * scale, fy * hh * scale);
                                if flip_x {
                                    x = -x;
                                }
                                if flip_y {
                                    y = -y;
                                }
                                let (xr, yr) = (x * cr - y * sr, x * sr + y * cr);
                                let (ux, uy) = uvs[i];
                                m.vertices.push(egui::epaint::Vertex {
                                    pos: pos + egui::vec2(xr, yr),
                                    uv: egui::pos2(ux, uy),
                                    color,
                                });
                            }
                            m.indices = vec![0, 1, 2, 0, 2, 3];
                            m
                        };
                        let outline = egui::Color32::from_rgba_unmultiplied(
                            core.brush_color[0],
                            core.brush_color[1],
                            core.brush_color[2],
                            200,
                        );
                        let fill_tex = egui::Color32::from_rgba_unmultiplied(
                            core.brush_color[0],
                            core.brush_color[1],
                            core.brush_color[2],
                            96,
                        );
                        let rim = 1.5 / (hw + hh).max(1.0) + 1.0;
                        ui.painter()
                            .add(egui::Shape::Mesh(sprite_mesh(rim, outline).into()));
                        ui.painter()
                            .add(egui::Shape::Mesh(sprite_mesh(1.0, fill_tex).into()));
                    }
                    crate::paint::BrushShape::Diamond => {
                        let r = screen_r * std::f32::consts::SQRT_2;
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
            ui.add_space(2.0);
            ui.label(&core.status);
            ui.label("LMB paint  |  Shift+LMB: straight stroke  |  MMB drag: orbit  |  Shift+MMB drag: pan  |  Wheel: zoom  |  Shift+Wheel: brush size  |  RMB: brush menu  |  F: fit  |  T: tools on/off");
        },
    );
}

/// Where a tool strip (the 3D T-bar, or the Texture preview's brush picker)
/// sits for a given anchor (top-left of the hosting panel) and slide progress:
/// 0 = fully hidden, pushed `STRIP_HIDE_EXTRA` past the left edge, 1 = flush
/// with the anchor. Constant-speed sliding means it never stalls half-visible.
fn tool_strip_rect(anchor_min: egui::Pos2, anim: f32) -> egui::Rect {
    let side = (STRIP_W - 6.0).max(18.0);
    let content_h = STRIP_PAD * 2.0 + TOOLS.len() as f32 * side;
    let total = STRIP_W + STRIP_HIDE_EXTRA;
    egui::Rect::from_min_size(
        anchor_min + egui::vec2(-total * (1.0 - anim), STRIP_TOP_INSET),
        egui::vec2(STRIP_W, content_h),
    )
}

/// Blender-style vertical T-bar overlaid on the viewport's left edge: a slim
/// translucent pill (fully rounded corners) with one compact icon per tool.
fn view_tool_strip(ui: &mut Ui, core: &mut Core, strip_rect: egui::Rect) {
    // One consistent corner radius everywhere so the backdrop and the buttons
    // read as a single pill.
    let corner = egui::CornerRadius::same(8);
    ui.painter()
        .rect_filled(strip_rect, corner, egui::Color32::from_black_alpha(140));
    ui.painter().rect_stroke(
        strip_rect,
        corner,
        egui::Stroke::new(1.0, egui::Color32::from_white_alpha(24)),
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
            ui.spacing_mut().item_spacing = egui::vec2(0.0, 0.0);
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
            egui::Frame::new()
                .fill(egui::Color32::from_black_alpha(150))
                .stroke(egui::Stroke::new(1.0, egui::Color32::from_white_alpha(24)))
                .corner_radius(egui::CornerRadius::same(8))
                .inner_margin(egui::Margin::symmetric(10, 6))
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(6.0, 0.0);
                    ui.checkbox(&mut core.show_uv_checker_3d, "UV checker")
                        .on_hover_text("Checkerboard overlay, mapped through the UVs");
                    ui.checkbox(&mut core.show_uv_grid_3d, "UV grid")
                        .on_hover_text("UV grid overlay, mapped through the UVs");
                });
        },
    );
}

/// The small floating button at the 3D viewport's top-right that toggles the
/// horizontal overlay bar's visibility: anchors the bar open, or (when
/// unpinned) temporarily re-shows it via hover.
fn vp_overlay_toggle(ui: &mut Ui, core: &mut Core, viewport: egui::Rect) {
    let rect = egui::Rect::from_min_size(
        egui::pos2(viewport.right() - 36.0, viewport.top() + 8.0),
        egui::vec2(28.0, 28.0),
    );
    let resp = ui.allocate_rect(rect, egui::Sense::click());
    let corner = egui::CornerRadius::same(8);
    let p = ui.painter_at(viewport);
    p.rect_filled(rect, corner, egui::Color32::from_black_alpha(150));
    p.rect_stroke(
        rect,
        corner,
        egui::Stroke::new(1.0, egui::Color32::from_white_alpha(24)),
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
        egui::FontId::proportional(14.0),
        ui.visuals().text_color(),
    );
    if resp.clicked() {
        core.show_vp_overlay_bar = !core.show_vp_overlay_bar;
    }
    resp.clone().on_hover_text(tip);
}

/// One square tool button inside the in-viewport T-bar (also reused by the
/// Texture preview's brush picker). Icons are hand-drawn except the Pick tool,
/// which reuses the lucide pipette image (ISC licensed).
fn tool_strip_button(ui: &mut Ui, core: &mut Core, index: usize, strip_width: f32) {
    let side = (strip_width - 6.0).max(18.0);
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(side, side), egui::Sense::click());

    if resp.hovered() {
        ui.painter()
            .rect_filled(rect, 8.0, egui::Color32::from_white_alpha(26));
    }
    if core.active_tool == index {
        ui.painter()
            .rect_filled(rect, 8.0, ui.visuals().selection.bg_fill);
    }

    let c = ui.visuals().text_color();
    let center = rect.center();
    let p = ui.painter();
    match index {
        0 => {
            // Brush: show the active brush's actual footprint (a round dab for
            // Round, a square for Square, the sprite itself for texture
            // brushes) instead of a fixed glyph, so the strip previews the
            // brush that will stamp.
            let sprite = core
                .brushes
                .selected
                .and_then(|i| core.brushes.entries.get(i))
                .map(|e| e.sprite.clone());
            if let Some(sprite) = sprite {
                let sig = sprite_sig(&sprite);
                if core
                    .brush_preview
                    .as_ref()
                    .map(|(s, _)| *s != sig)
                    .unwrap_or(true)
                {
                    let img = egui::ColorImage::new(
                        [sprite.width as usize, sprite.height as usize],
                        sprite
                            .rgba
                            .chunks_exact(4)
                            .map(|px| {
                                egui::Color32::from_rgba_unmultiplied(
                                    px[0], px[1], px[2], px[3],
                                )
                            })
                            .collect(),
                    );
                    let handle = ui.ctx().load_texture(
                        format!("brush_preview_strip_{:016x}", sig),
                        img,
                        egui::TextureOptions::LINEAR,
                    );
                    core.brush_preview = Some((sig, handle));
                }
                let handle = core.brush_preview.as_ref().unwrap().1.clone();
                let icon = egui::Rect::from_center_size(
                    center,
                    egui::vec2(rect.width() - 8.0, rect.width() - 8.0),
                );
                p.image(
                    handle.id(),
                    icon,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    c,
                );
            } else {
                p.circle_filled(center, 3.0, c);
                p.circle_stroke(center, 6.0, egui::Stroke::new(1.5, c));
            }
        }
        1 => {
            // Eraser: ring with a diagonal slash.
            p.circle_stroke(center, 6.0, egui::Stroke::new(1.5, c));
            p.line_segment(
                [
                    center + egui::vec2(-7.0, -7.0),
                    center + egui::vec2(7.0, 7.0),
                ],
                egui::Stroke::new(1.5, c),
            );
        }
        2 => {
            // Fill: slanted bucket trapezoid with a drip below.
            let maker = |dx: f32, dy: f32| center + egui::vec2(dx, dy);
            p.add(egui::Shape::convex_polygon(
                vec![
                    maker(-6.0, -5.0),
                    maker(6.0, -5.0),
                    maker(6.0, 1.0),
                    maker(-6.0, 1.0),
                ],
                c,
                egui::Stroke::NONE,
            ));
            p.circle_filled(maker(3.0, 4.5), 1.7, c);
        }
        3 => {
            // Pick: the lucide pipette image (ISC).
            if let Some(icon) = core.pick_icon_tex(ui.ctx()) {
                let img_rect =
                    egui::Rect::from_center_size(center, egui::vec2(side - 8.0, side - 8.0));
                p.image(
                    icon.id(),
                    img_rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    c,
                );
            } else {
                p.text(
                    center,
                    egui::Align2::CENTER_CENTER,
                    "Pick",
                    egui::FontId::proportional(9.0),
                    c,
                );
            }
        }
        4 => {
            // Rect: a filled-outline square (rectangular stamp).
            let square = egui::Rect::from_center_size(center, egui::vec2(11.0, 11.0));
            p.rect_filled(square, 1.0, egui::Color32::from_black_alpha(20));
            p.rect_stroke(
                square,
                1.0,
                egui::Stroke::new(1.5, c),
                egui::StrokeKind::Outside,
            );
        }
        _ => {}
    }

    if resp.clicked() {
        core.active_tool = index;
        core.status = format!("Tool: {}", TOOLS[index]);
    }
    resp.on_hover_text(format!("Tool: {}", TOOLS[index]));
}

/// Renders the right-click brush menu as a floating popup over the viewport.
/// Closes on Escape or any click outside it.
fn brush_menu_popup(ui: &mut Ui, core: &mut Core) {
    if !core.brush_menu_open {
        return;
    }
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
                .map_or(false, |p| !popup_rect.expand(4.0).contains(p))
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

    // Fixed palette quick-picks.
    ui.horizontal_wrapped(|ui| {
        for rgb in PRESET_COLORS {
            if swatch_button(ui, rgb).clicked() {
                core.brush_color = [rgb[0], rgb[1], rgb[2], core.brush_color[3]];
                core.brush_menu_open = false;
                core.brush_menu_pos = None;
            }
        }
    });

    ui.separator();

    // Custom color: chip + hex, then the live picker right below.
    let cur = core.brush_color;
    ui.horizontal(|ui| {
        let (rect, _) = ui.allocate_exact_size(egui::vec2(18.0, 18.0), egui::Sense::hover());
        ui.painter().rect_filled(
            rect,
            3.0,
            egui::Color32::from_rgba_unmultiplied(cur[0], cur[1], cur[2], cur[3]),
        );
        ui.painter().rect_stroke(
            rect,
            3.0,
            egui::Stroke::new(1.0, egui::Color32::from_gray(120)),
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
        core.brush_color = col.to_srgba_unmultiplied();
        core.status = format!(
            "Brush color #{:02X}{:02X}{:02X} α{:02X}",
            core.brush_color[0], core.brush_color[1], core.brush_color[2], core.brush_color[3]
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
        for shape in crate::paint::BrushShape::ALL {
            let sel = core.brush_style.shape == shape;
            if ui.selectable_label(sel, shape.label()).clicked() {
                core.brush_style.shape = shape;
                core.brush_menu_open = false;
                core.brush_menu_pos = None;
            }
        }
    });
    ui.horizontal(|ui| {
        if ui.button("Load brush PNG…").clicked() {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("PNG", &["png"])
                .pick_file()
            {
                let path_str = path.to_string_lossy().into_owned();
                match crate::io::brush_sprite(&path_str) {
                    Ok(sprite) => {
                        core.brush_style.shape = crate::paint::BrushShape::Texture;
                        core.brush_style.sprite = Some(sprite);
                        core.brush_menu_open = false;
                        core.brush_menu_pos = None;
                        core.status = format!("Loaded brush sprite: {path_str}");
                    }
                    Err(e) => core.status = format!("Brush load failed: {e}"),
                }
            }
        }
    });
    if core.brush_style.sprite.is_some() {
        ui.horizontal(|ui| {
            ui.label("Rotate");
            ui.add(egui::Slider::new(
                &mut core.brush_style.rotation,
                -std::f32::consts::PI..=std::f32::consts::PI,
            ));
        });
        ui.horizontal(|ui| {
            ui.checkbox(&mut core.brush_style.flip_x, "Flip X");
            ui.checkbox(&mut core.brush_style.flip_y, "Flip Y");
        });
    }
}

/// Small tappable color square for the palette grid.
fn swatch_button(ui: &mut Ui, rgb: [u8; 3]) -> egui::Response {
    let size = egui::vec2(20.0, 20.0);
    let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::click());
    if ui.is_rect_visible(rect) {
        ui.painter()
            .rect_filled(rect, 3.0, egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]));
        let stroke = if resp.hovered() {
            egui::Stroke::new(2.0, egui::Color32::WHITE)
        } else {
            egui::Stroke::new(1.0, egui::Color32::from_gray(120))
        };
        ui.painter()
            .rect_stroke(rect, 3.0, stroke, egui::StrokeKind::Inside);
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

fn toolbar_ui(ui: &mut Ui, core: &mut Core) {
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().slider_width = 70.0;

        let mut color = core.brush_color;
        if ui
            .color_edit_button_srgba_unmultiplied(&mut color)
            .on_hover_text("Brush color — right-click the viewport for a picker & presets")
            .changed()
        {
            core.brush_color = color;
        }
        ui.label("Size");
        ui.add(
            egui::Slider::new(&mut core.brush_size, 1.0..=300.0)
                .suffix("px")
                .logarithmic(true)
                .max_decimals(0),
        );
        ui.label("Hardness");
        ui.add(egui::Slider::new(&mut core.brush_hardness, 0.0..=1.0)).on_hover_text(
            "Fraction of the radius at full strength; it fades to the edge beyond that. 100% = hard edge.",
        );
        ui.label("Opacity");
        ui.add(egui::Slider::new(&mut core.brush_opacity, 0.0..=1.0));
        ui.label("Spacing");
        ui.add(
            egui::Slider::new(&mut core.brush_spacing, 0.0..=200.0)
                .suffix("px")
                // Linear + whole-pixel steps: a log scale collapses the useful
                // 1..12 range into the left edge, and log(0) is undefined so
                // the "continuous" value near zero behaved erratically.
                .step_by(1.0)
                .max_decimals(0),
        )
        .on_hover_text("0 = continuous (dabs overlap, tuned to brush size). Positive = fixed distance (px) between dabs along a stroke.");
    });
}

/// Layer material panel. Surface parameters (roughness, metallic, emissive,
/// AO) are per-layer sliders: they shape the shading wherever the layer covers
/// the model, following the same source-over stacking as the albedo.
fn channels_ui(ui: &mut Ui, core: &mut Core) {
    ui.heading("Material");
    let Some(mesh) = core.mesh.as_mut() else {
        ui.separator();
        ui.label("Open a model to edit its layer material.");
        return;
    };
    if mesh.layers.is_empty() {
        ui.separator();
        ui.label("No layers yet — add a layer to set its material.");
        return;
    }
    ui.spacing_mut().slider_width = 132.0;
    let li = mesh.active_layer.min(mesh.layers.len() - 1);

    let surface_changed = {
        let l = &mut mesh.layers[li];
        ui.label(format!("Layer: {}", l.name));
        let old_surface = (
            l.roughness,
            l.metallic,
            l.emissive,
            l.ambient_occlusion,
            l.height,
            l.bump_strength,
        );

        ui.add(
            egui::Slider::new(&mut l.roughness, 0.03..=1.0)
                .text("Roughness")
                .logarithmic(true),
        );
        ui.add(egui::Slider::new(&mut l.metallic, 0.0..=1.0).text("Metallic"));
        ui.add(egui::Slider::new(&mut l.emissive, 0.0..=3.0).text("Emissive glow"));
        ui.add(
            egui::Slider::new(&mut l.ambient_occlusion, 0.0..=1.0).text("Ambient occlusion"),
        );
        ui.add(egui::Slider::new(&mut l.height, -1.0..=1.0).text("Height"));
        ui.add(
            egui::Slider::new(&mut l.bump_strength, 0.0..=8.0)
                .logarithmic(true)
                .text("Bump strength"),
        );

        ui.horizontal_wrapped(|ui| {
            ui.label("Presets:");
            let presets: [(&str, (f32, f32, f32, f32)); 4] = [
                ("Clay", (0.85, 0.0, 0.0, 1.0)),
                ("Glossy", (0.18, 0.0, 0.0, 1.0)),
                ("Brushed metal", (0.35, 1.0, 0.0, 1.0)),
                ("Cold metal", (0.25, 1.0, 0.1, 1.0)),
            ];
            for (name, (rough, metal, emiss, ao)) in presets {
                if ui
                    .selectable_label(
                        (l.roughness, l.metallic, l.emissive, l.ambient_occlusion)
                            == (rough, metal, emiss, ao),
                        name,
                    )
                    .clicked()
                {
                    l.roughness = rough;
                    l.metallic = metal;
                    l.emissive = emiss;
                    l.ambient_occlusion = ao;
                    core.status = format!("Material preset: {name}");
                }
            }
        });

        (l.roughness, l.metallic, l.emissive, l.ambient_occlusion, l.height, l.bump_strength)
            != old_surface
    };

    if surface_changed {
        core.needs_material_upload = true;
    }
}

/// Viewport-wide lighting. Sun on/off + direction + color drive the direct key
/// light; the environment (analytic sky or a loaded skybox map) lights diffuse
/// and specular reflections and shows as the background. Turning the sun off
/// leaves the skybox as the sole light source.
fn lighting_ui(ui: &mut Ui, core: &mut Core) {
    ui.heading("Lighting");
    ui.spacing_mut().slider_width = 150.0;
    let material = &mut core.material;

    let sun_was = (material.sun_enabled, material.sun_elevation, material.sun_azimuth);
    ui.horizontal(|ui| {
        ui.checkbox(&mut material.sun_enabled, "Sun (directional key light)")
            .on_hover_text("Turn off so the environment/skybox alone lights the scene.");
        if !material.sun_enabled {
            ui.label("off — skybox only");
        }
    });
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
    ui.add_enabled_ui(material.sun_enabled, |ui| {
        ui.horizontal(|ui| {
            ui.label("Sun color");
            if ui.color_edit_button_rgb(&mut material.sun_color).changed() {
                core.status = "Sun color changed".to_string();
            }
        });
    });

    ui.separator();
    ui.heading("Environment");
    let env_label = match &core.env_path {
        Some(p) => format!(
            "Skybox: {}",
            p.rsplit(['/', '\\']).next().unwrap_or(p.as_str())
        ),
        None => "Skybox: analytic (color) sky".to_string(),
    };
    ui.label(env_label);
    let env_was = (material.env_intensity, material.env_rotation, material.sky_color);
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
    ui.add_enabled(
        material.sun_enabled,
        egui::Slider::new(&mut material.fill_intensity, 0.0..=1.5).text("Interior fill"),
    )
    .on_hover_text(
        "Camera-direction fill light that keeps shadow interiors readable. Tied to the sun: \
         disabled while the sun is off, so that mode is pure skybox lighting.",
    );

    if (sun_was.0, sun_was.1, sun_was.2) != (
        material.sun_enabled,
        material.sun_elevation,
        material.sun_azimuth,
    ) {
        core.status = "Sun updated".to_string();
    }
    if env_was != (material.env_intensity, material.env_rotation, material.sky_color) {
        core.status = "Environment updated".to_string();
    }
}

/// Browsable brush library tab. Lists the current brush, the folder it reads
/// from, a category filter (subfolders become categories), and a grid of
/// thumbnails. Click a brush to make it the active brush.
fn brushes_ui(ui: &mut Ui, core: &mut Core) {
    // Cheap-ish folder walk (throttled inside) so dropping a new file onto the
    // folder shows up while this panel is on screen.
    core.brushes.refresh();

    ui.horizontal(|ui| {
        ui.heading("Brushes");
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
        ui.colored_label(
            egui::Color32::from_rgb(235, 165, 100),
            egui::RichText::new(format!("Skipped: {err}")).small(),
        );
    }

    if let Some(sel) = core.brushes.selected {
        if let Some(entry) = core.brushes.entries.get(sel) {
            ui.horizontal(|ui| {
                ui.label("Active:");
                ui.strong(&entry.name);
            });
        }
    }
    ui.separator();

    // Category filter pills.
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

    if indices.is_empty() {
        ui.label("No brushes in this category.");
        return;
    }

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
                ui.ctx()
                    .load_texture(format!("brush_thumb_{i}"), img, egui::TextureOptions::LINEAR),
            );
        }
    }

    let mut clicked: Option<usize> = None;
    ui.horizontal_wrapped(|ui| {
        for &i in &indices {
            let entry = &core.brushes.entries[i];
            let selected = core.brushes.selected == Some(i);
            let (rect, resp) =
                ui.allocate_exact_size(egui::vec2(80.0, 100.0), egui::Sense::click());
            let painter = ui.painter_at(rect);

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
            let stroke = if selected {
                egui::Stroke::new(2.0, egui::Color32::from_rgb(90, 160, 255))
            } else if resp.hovered() {
                egui::Stroke::new(1.0, egui::Color32::WHITE)
            } else {
                egui::Stroke::new(1.0, egui::Color32::from_gray(90))
            };
            painter.rect_stroke(img_rect.expand(2.0), 4.0, stroke, egui::StrokeKind::Inside);

            painter.text(
                egui::pos2(rect.center().x, rect.bottom() - 10.0),
                egui::Align2::CENTER_CENTER,
                truncate_mid(&entry.name, 13),
                egui::FontId::proportional(11.0),
                ui.visuals().text_color(),
            );

            if resp.clicked() {
                clicked = Some(i);
            }
            if let Some(path) = &entry.path {
                let _ = resp.clone().on_hover_text(format!("{}\n{}", entry.name, path.display()));
            } else {
                let _ = resp
                    .clone()
                    .on_hover_text(format!("{} (built-in)", entry.name));
            }
        }
    });

    if let Some(i) = clicked {
        apply_brush(core, i);
    }
}

/// Sets `brush_style` from a library entry, keeping the current tool active
/// (the footprint changes; an eraser in use stays an eraser) and clearing any
/// stale stamp transform.
fn apply_brush(core: &mut Core, index: usize) {
    use crate::brushes::BrushKind;
    let Some(entry) = core.brushes.entries.get(index) else {
        return;
    };
    match entry.kind {
        BrushKind::Shape(shape) => {
            core.brush_style.shape = shape;
            core.brush_style.sprite = None;
        }
        BrushKind::Texture => {
            core.brush_style.shape = crate::paint::BrushShape::Texture;
            core.brush_style.sprite = Some(entry.sprite.clone());
        }
    }
    core.brush_style.rotation = 0.0;
    core.brush_style.flip_x = false;
    core.brush_style.flip_y = false;
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
    ui.heading("Texture Editor");
    ui.separator();

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
            let has_tex = core.mesh.as_ref().map_or(false, |m| !m.layers.is_empty());
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
    let tex = core.mesh.as_ref().and_then(|m| m.flattened_atlas());

    if let Some(tex) = tex {
        if core.texture_preview.as_ref().map(|p| p.gen) != Some(gen) {
            // The 3D view shows the GPU atlas live during a stroke; defer the
            // full-size CPU -> egui copy until the stroke ends so dragging a
            // 3D brush doesn't recomposite the whole atlas every frame. The 2D
            // preview repaints live during *its own* strokes so ink is visible
            // while dragging — the handle is reused via `set`, so there's no
            // per-frame texture churn.
            let defer_3d = core.stroke.is_some();
            if !defer_3d {
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

        if let Some(handle_id) = core.texture_preview.as_ref().map(|p| p.handle.id()) {
            let (tw_f, th_f) = (tex.width.max(1) as f32, tex.height.max(1) as f32);

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
            let (rect, _resp) = ui.allocate_exact_size(
                ui.available_size(),
                egui::Sense::click_and_drag(),
            );
            ui.painter().rect_filled(rect, 0.0, egui::Color32::from_gray(22));

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

            // Per-panel T toggle for the 2D brush picker: consume T only while
            // the pointer hovers the canvas (or the picker itself), so the 3D
            // viewport keeps its own T for its T-bar.
            if (ui.rect_contains_pointer(rect)
                || ui
                    .input(|i| i.pointer.hover_pos())
                    .is_some_and(|p| strip_rect.expand(2.0).contains(p)))
                && ui.ctx().input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::T))
            {
                core.show_brush_picker = !core.show_brush_picker;
                core.status = if core.show_brush_picker {
                    "Texture tools: on (T to toggle)".to_string()
                } else {
                    "Texture tools: off (T to toggle)".to_string()
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
                    core.brush_size = (core.brush_size * factor).clamp(1.0, 300.0);
                } else {
                    // Scale about the cursor: the texel under the pointer stays
                    // fixed on screen while everything else zooms around it.
                    let new_zoom = (core.canvas2d.zoom * (scroll_amount * 0.0015).exp())
                        .clamp(0.01, 64.0);
                    if new_zoom != core.canvas2d.zoom {
                        let factor = new_zoom / core.canvas2d.zoom;
                        let new_center = if let Some(p) = pointer {
                            let before_size = egui::vec2(
                                tw_f * core.canvas2d.zoom,
                                th_f * core.canvas2d.zoom,
                            );
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
            let img_rect = egui::Rect::from_center_size(
                rect.center() + core.canvas2d.center,
                img_size,
            );
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
                egui::Stroke::new(1.0, egui::Color32::from_gray(70)),
                egui::StrokeKind::Outside,
            );
            p.text(
                rect.right_top() + egui::vec2(-6.0, 6.0),
                egui::Align2::RIGHT_TOP,
                "MMB drag: pan · wheel: zoom · shift+wheel: brush size",
                egui::FontId::proportional(12.0),
                egui::Color32::from_gray(255).gamma_multiply(0.35),
            );
            p.text(
                rect.left_bottom() + egui::vec2(6.0, -6.0),
                egui::Align2::LEFT_BOTTOM,
                format!(
                    "{} × {} @ {:.0}%",
                    tex.width,
                    tex.height,
                    core.canvas2d.zoom * 100.0
                ),
                egui::FontId::proportional(12.0),
                egui::Color32::from_gray(200).gamma_multiply(0.9),
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
            let over_image = pointer.is_some_and(|p| img_rect.contains(p));
            if hovered && over_image && (primary_down || pressed || released) {
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
                        let color = if core.active_tool == 0 || core.active_tool == 4 {
                            core.brush_color
                        } else {
                            [0, 0, 0, 0]
                        };

                        // Rect tools stamp a square footprint in 2D.
                        let shape = if core.active_tool == 4 {
                            crate::paint::BrushShape::Square
                        } else {
                            core.brush_style.shape
                        };

                        // Brush radius in texels: the on-screen footprint stays
                        // `brush_size` px at any zoom, so the cursor ring always
                        // matches the stamp it previews.
                        let brush_r_texels = core.brush_size * (tex.width as f32 / pw);

                        if core.active_tool == 3 {
                            // Pick tool: sample directly from the atlas at the
                            // cursor UV and switch back to Brush.
                            if pressed {
                                let px = (u * tex.width as f32).round() as u32;
                                let py = (v * tex.height as f32).round() as u32;
                                let px = px.min(tex.width.saturating_sub(1));
                                let py = py.min(tex.height.saturating_sub(1));
                                let idx = ((py * tex.width + px) * 4) as usize;
                                if idx + 3 < tex.rgba.len() {
                                    let c: [u8; 4] = [
                                        tex.rgba[idx],
                                        tex.rgba[idx + 1],
                                        tex.rgba[idx + 2],
                                        tex.rgba[idx + 3],
                                    ];
                                    core.brush_color = c;
                                    core.active_tool = 0;
                                    core.status = format!(
                                        "Picked rgba({}, {}, {}, {}) — back to Brush",
                                        c[0], c[1], c[2], c[3]
                                    );
                                }
                            }
                        } else if core.active_tool == 2 {
                            // Fill tool: flood-fill the region around the click
                            // point with the current brush color.
                            if pressed {
                                if let Some(m) = core.mesh.as_ref() {
                                    core.history.record(snapshot_of(m));
                                }
                                if let Some(mesh) = core.mesh.as_mut() {
                                    let mut dirty = mesh.dirty;
                                    crate::paint::stamp_fill_2d(
                                        mesh.active_layer_texture_mut().unwrap(),
                                        (u, v),
                                        core.brush_color,
                                        core.brush_opacity,
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
                                if let Some(m) = core.mesh.as_ref() {
                                    core.history.record(snapshot_of(m));
                                }
                                core.stroke_2d = Some(StrokeState {
                                    last: egui::pos2(u, v),
                                    start: egui::pos2(u, v),
                                    last_dab: egui::pos2(u, v),
                                    acc: 0.0,
                                    next_t: 0.0,
                                });
                                if let Some(mesh) = core.mesh.as_mut() {
                                    let mut dirty = mesh.dirty;
                                    crate::paint::stamp_2d(
                                        mesh.active_layer_texture_mut().unwrap(),
                                        (u, v),
                                        brush_r_texels,
                                        shape,
                                        color,
                                        core.brush_opacity,
                                        core.brush_hardness,
                                        mode,
                                        &core.brush_style,
                                        &mut dirty,
                                    );
                                    mesh.dirty = dirty;
                                }
                                painted = true;
                            } else if let Some(st) = core.stroke_2d.as_mut() {
                                // Dab-to-dab travel is accumulated in *texel*
                                // space: spacing is in screen px (the 3D viewport
                                // semantics), but the 2D cursor lives in
                                // normalized UV — on a 1024² atlas a "6px" step
                                // must be 6 texels, not 6 UV units.
                                let spacing = if core.brush_spacing > 0.0 {
                                    core.brush_spacing
                                } else {
                                    (brush_r_texels / 2.0).max(1.0)
                                };
                                let mut dabs: Vec<egui::Pos2> = Vec::new();
                                let to_px =
                                    |x: f32, y: f32| egui::pos2(x * tw_f, y * th_f);

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
                                    st.last_dab = egui::pos2(
                                        last_dab.x / tw_f,
                                        last_dab.y / th_f,
                                    );
                                    dabs.extend(
                                        dabs_here
                                            .into_iter()
                                            .map(|p| {
                                                egui::pos2(p.x / tw_f, p.y / th_f)
                                            }),
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
                                            shape,
                                            color,
                                            core.brush_opacity,
                                            core.brush_hardness,
                                            mode,
                                            &core.brush_style,
                                            &mut dirty,
                                        );
                                        mesh.dirty = dirty;
                                    }
                                    painted = !dabs.is_empty();
                                }
                            }
                        }
                    }
                }
            }

            if painted {
                flush_paint_edit(core);
                if core.active_tool == 1 {
                    core.status = "Erased — fully transparent (alpha 0) in 2D preview"
                        .to_string();
                }
                ui.ctx().request_repaint();
            }

            // End the stroke on pointer release or when the pointer leaves the
            // preview area (mirrors the 3D viewport's hover-gating).
            if !(hovered && primary_down) {
                core.stroke_2d = None;
            }

            // The 2D brush cursor mirrors the 3D viewport's: same fill/stroke colors,
            // same shape handling, and the actual brush sprite (shape + pattern +
            // rotation/flips) for texture brushes — all clipped to the canvas so
            // it never bleeds over the toolbar above.
            if hovered {
                if let Some(p_pos) = pointer {
                    let brush_r = core.brush_size;
                    let painting = core.active_tool == 0 || core.active_tool == 4;
                    let (fill, stroke, dot) = if painting {
                        (
                            egui::Color32::from_rgba_unmultiplied(
                                core.brush_color[0],
                                core.brush_color[1],
                                core.brush_color[2],
                                60,
                            ),
                            egui::Color32::from_rgba_unmultiplied(
                                core.brush_color[0],
                                core.brush_color[1],
                                core.brush_color[2],
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
                        _ => match core.brush_style.shape {
                            crate::paint::BrushShape::Round => {
                                p.circle_filled(p_pos, brush_r, fill);
                                p.circle_stroke(
                                    p_pos,
                                    brush_r,
                                    egui::Stroke::new(1.5, stroke),
                                );
                            }
                            crate::paint::BrushShape::Square => {
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
                            crate::paint::BrushShape::Diamond => {
                                let r = brush_r * std::f32::consts::SQRT_2;
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
                            crate::paint::BrushShape::Texture => {
                                // Show the actual brush sprite (shape + pattern)
                                // instead of a plain outline, exactly like the 3D
                                // viewport's textured cursor.
                                if let Some(sprite) = &core.brush_style.sprite {
                                let sig = sprite_sig(sprite);
                                if core
                                    .brush_preview
                                    .as_ref()
                                    .map(|(s, _)| *s != sig)
                                    .unwrap_or(true)
                                {
                                    let img = egui::ColorImage::new(
                                        [sprite.width as usize, sprite.height as usize],
                                        sprite
                                            .rgba
                                            .chunks_exact(4)
                                            .map(|c| {
                                                egui::Color32::from_rgba_unmultiplied(
                                                    c[0], c[1], c[2], c[3],
                                                )
                                            })
                                            .collect(),
                                    );
                                    let handle = ui.ctx().load_texture(
                                        format!("brush_preview_{:016x}", sig),
                                        img,
                                        egui::TextureOptions::LINEAR,
                                    );
                                    core.brush_preview = Some((sig, handle));
                                }
                                let handle = core
                                    .brush_preview
                                    .as_ref()
                                    .unwrap()
                                    .1
                                    .clone();
                                let aspect = (sprite.height.max(1) as f32)
                                    / (sprite.width.max(1) as f32);
                                let hw = brush_r;
                                let hh = brush_r * aspect;
                                let (sr, cr) = core.brush_style.rotation.sin_cos();
                                let flip_x = core.brush_style.flip_x;
                                let flip_y = core.brush_style.flip_y;
                                let sprite_mesh = |scale: f32, color: egui::Color32| {
                                    let mut m = egui::Mesh::with_texture(handle.id());
                                    let corners = [
                                        (1.0, 1.0),
                                        (1.0, -1.0),
                                        (-1.0, -1.0),
                                        (-1.0, 1.0),
                                    ];
                                    let uvs = [
                                        (1.0, 1.0),
                                        (1.0, 0.0),
                                        (0.0, 0.0),
                                        (0.0, 1.0),
                                    ];
                                    for (i, (fx, fy)) in corners
                                        .into_iter()
                                        .enumerate()
                                    {
                                        let (mut x, mut y) =
                                            (fx * hw * scale, fy * hh * scale);
                                        if flip_x {
                                            x = -x;
                                        }
                                        if flip_y {
                                            y = -y;
                                        }
                                        let (xr, yr) =
                                            (x * cr - y * sr, x * sr + y * cr);
                                        let (ux, uy) = uvs[i];
                                        m.vertices.push(egui::epaint::Vertex {
                                            pos: p_pos + egui::vec2(xr, yr),
                                            uv: egui::pos2(ux, uy),
                                            color,
                                        });
                                    }
                                    m.indices = vec![0, 1, 2, 0, 2, 3];
                                    m
                                };
                                let outline = egui::Color32::from_rgba_unmultiplied(
                                    core.brush_color[0],
                                    core.brush_color[1],
                                    core.brush_color[2],
                                    200,
                                );
                                let fill_tex = egui::Color32::from_rgba_unmultiplied(
                                    core.brush_color[0],
                                    core.brush_color[1],
                                    core.brush_color[2],
                                    96,
                                );
                                let rim = 1.5 / (hw + hh).max(1.0) + 1.0;
                                p.add(egui::Shape::Mesh(sprite_mesh(rim, outline).into()));
                                p.add(egui::Shape::Mesh(sprite_mesh(1.0, fill_tex).into()));
                                } else {
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
        ui.painter()
            .rect_filled(rect, 0.0, egui::Color32::from_gray(40));
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
}

/// Layer stack panel: per-layer visibility / opacity / selection plus the
/// structural operations (add, duplicate, delete, reorder). The topmost layer
/// is listed first. Visibility and structural edits are undoable; the opacity
/// slider edits live (each frame it changes merely re-composites).
fn layers_ui(ui: &mut Ui, core: &mut Core) {
    let Some(mesh) = core.mesh.as_mut() else {
        return;
    };

    let len = mesh.layers.len();
    let active = mesh.active_layer;
    let res = core.atlas_res;

    let mut add = false;
    let mut duplicate = false;
    let mut delete = false;
    let mut move_up = false;
    let mut move_down = false;
    let mut remake_uvs = false;
    ui.horizontal(|ui| {
        add = ui.button("Add").clicked();
        duplicate = ui
            .add_enabled(active < len, egui::Button::new("Duplicate"))
            .clicked();
        delete = ui
            .add_enabled(len > 0, egui::Button::new("Delete"))
            .clicked();
        move_up = ui
            .add_enabled(active > 0, egui::Button::new("Up"))
            .clicked();
        move_down = ui
            .add_enabled(active + 1 < len, egui::Button::new("Down"))
            .clicked();
        ui.separator();
        remake_uvs = ui
            .button("Remake UV")
            .on_hover_text(
                "Re-pack the UV islands so texture resolution is proportional to surface area. The atlas content is re-baked to follow the islands.",
            )
            .clicked();
    });

    if len > 0 {
        let active_mode = mesh.layers[mesh.active_layer].blend;
        let mut new_mode = active_mode;
        ui.horizontal(|ui| {
            ui.label("Blend:");
            for m in crate::io::BlendMode::ALL {
                if ui
                    .selectable_label(new_mode == m, m.short_name())
                    .on_hover_text(m.name())
                    .clicked()
                {
                    new_mode = m;
                }
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
        core.stroke = None;
        core.stroke = None;
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
            core.stroke = None;
        core.stroke = None;
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
        core.stroke = None;
        core.stroke = None;
        core.needs_texture_upload = true;
        core.needs_material_upload = true;
        core.preview_gen += 1;
        core.status = "Deleted layer".to_string();
    }
    if move_up {
        core.history.record(snapshot_of(mesh));
        mesh.layers.swap(active, active - 1);
        mesh.active_layer = active - 1;
        core.stroke = None;
        core.stroke = None;
        core.needs_texture_upload = true;
        core.needs_material_upload = true;
        core.preview_gen += 1;
        core.status = "Layer moved up".to_string();
    }
    if move_down {
        core.history.record(snapshot_of(mesh));
        mesh.layers.swap(active, active + 1);
        mesh.active_layer = active + 1;
        core.stroke = None;
        core.stroke = None;
        core.needs_texture_upload = true;
        core.needs_material_upload = true;
        core.preview_gen += 1;
        core.status = "Layer moved down".to_string();
    }

    if mesh.layers.is_empty() {
        ui.label("No layers yet — add one to start painting.");
        return;
    }

    let mut needs_refresh = false;
    // Topmost layer listed first: iterate the stack in reverse.
    for li in (0..mesh.layers.len()).rev() {
        let (name, visible, opacity) = {
            let l = &mesh.layers[li];
            (l.name.clone(), l.visible, l.opacity)
        };
        let is_active = li == mesh.active_layer;

        let mut toggled = false;
        let mut selected = false;
        let mut op = opacity;
        let mut op_changed = false;
        ui.horizontal(|ui| {
            toggled = ui
                .button(if visible { "👁" } else { "🚫" })
                .on_hover_text(if visible { "Hide layer" } else { "Show layer" })
                .clicked();
            selected = ui.selectable_label(is_active, name).clicked();
            let slider = egui::Slider::new(&mut op, 0.0..=1.0)
                .show_value(false)
                .suffix("%");
            op_changed = ui.add(slider).changed();
        });

        if toggled {
            core.history.record(snapshot_of(mesh));
            mesh.layers[li].visible = !mesh.layers[li].visible;
            needs_refresh = true;
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
        if op_changed {
            mesh.layers[li].opacity = op;
            needs_refresh = true;
        }
    }

    if needs_refresh {
        core.needs_texture_upload = true;
        core.needs_material_upload = true;
        core.preview_gen += 1;
        ui.ctx().request_repaint();
    }

    // Remake UV: recompute the island packing (both the mesh UVs and, by
    // re-baking every layer atlas, the paint so it stays glued to the surface),
    // then upload the new geometry + textures to the GPU together.
    if remake_uvs {
        if let Some(mut m) = core.mesh.take() {
            core.history.record(snapshot_of(&m));
            let n = crate::io::remake_uv(&mut m);
            if n > 1 {
                core.stroke = None;
        core.stroke = None;
                core.needs_texture_upload = true;
                core.needs_material_upload = true;
                core.preview_gen += 1;
                core.status = format!("Remade UVs — {n} islands repacked");
            } else if n == 0 {
                core.status = "Remake UV: nothing to do".to_string();
            } else {
                core.status = "Remake UV: mesh is one connected island".to_string();
            }
            // set_mesh uploads the new vertex UVs AND re-bakes both textures,
            // so the flags above are belt-and-braces for the next frame.
            core.renderer.set_mesh(&m);
            core.mesh = Some(m);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            },
            show_tool_strip: false,
            camera: Some(CameraState {
                eye: [1.0, 2.0, 3.0],
                target: [0.5, 0.5, 0.5],
                radius: 4.25,
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
        assert_eq!(back.material, mem.material);
        assert_eq!(back.material.roughness, 0.3);
        assert_eq!(back.material.exposure, 1.6);
        assert_eq!(back.camera.unwrap().radius, 4.25);

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
                opacity: 1.0,
                blend: crate::io::BlendMode::Normal,
                roughness: 0.55,
                metallic: 0.0,
                emissive: 0.0,
                ambient_occlusion: 1.0,
                height: 0.0,
                bump_strength: 2.0,
                texture: crate::io::TextureData {
                    width: 2,
                    height: 2,
                    rgba: vec![r; 16],
                },
            }],
        }
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
}

/// Classic two-tone alpha checkerboard drawn behind the atlas preview so
/// transparent (erased) texels are clearly visible.
fn draw_checkerboard(ui: &Ui, rect: egui::Rect, clip: egui::Rect) {
    let square = (rect.width() / 24.0).ceil().max(8.0);
    let colors = [egui::Color32::from_gray(96), egui::Color32::from_gray(80)];
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
    let boundary_stroke = egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 214, 96));
    let interior_stroke = egui::Stroke::new(
        1.0,
        egui::Color32::from_rgba_unmultiplied(120, 190, 255, 190),
    );

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
