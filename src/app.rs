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
}

impl Panel {
    const ALL: [Panel; 3] = [Panel::Viewport, Panel::Channels, Panel::Texture];

    fn title(&self) -> &'static str {
        match self {
            Panel::Viewport => "3D Viewport",
            Panel::Channels => "Channels",
            Panel::Texture => "Texture",
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
    panel_visible: [bool; 3],
    active_tool: usize,
    channels: [bool; 6],
    brush_size: f32,
    brush_hardness: f32,
    brush_opacity: f32,
    /// Distance (px) between dab centers along a stroke; `<= 0` = single dab per frame.
    brush_spacing: f32,
    /// RGBA brush color (persisted; used by Paint/Fill, set by Pick).
    brush_color: [u8; 4],
    /// Last pointer position of the active stroke (for dab interpolation).
    stroke_last: Option<egui::Pos2>,
    /// True while the primary button is held down and edits are happening.
    stroke_active: bool,
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
    /// How the Texture preview fills transparent (erased) texels.
    preview_bg: TexturePreviewBg,
    /// Fill color for `TexturePreviewBg::Custom` (alpha included, so alpha 0
    /// fills with full transparency).
    preview_bg_color: [u8; 4],
    /// Show the in-viewport vertical tool strip (its translucent T-bar).
    show_tool_strip: bool,
    /// Slide-in/out animation progress of the T-bar: 0 = fully hidden off the
    /// left edge, 1 = fully visible (transient, not persisted).
    tool_strip_anim: f32,
    /// Camera view to restore on the first frame (from a previous session).
    restore_view: Option<(glam::Vec3, glam::Vec3, f32)>,
    status: String,
}

/// How the Texture preview fills the transparent (erased) regions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
enum TexturePreviewBg {
    /// Classic alpha checkerboard.
    #[default]
    Checker,
    /// User-selected solid color.
    Custom,
}

struct PreviewTexture {
    gen: u64,
    handle: TextureHandle,
}

/// A full copy of the albedo atlas at some point in time, used to restore
/// texture state for undo/redo.
#[derive(Clone)]
struct TextureSnapshot {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

/// Bounded undo/redo history. `undo` holds states that can restore *to*; the
/// most recent is last. Pushing a new snapshot clears the redo stack.
struct EditHistory {
    undo: Vec<TextureSnapshot>,
    redo: Vec<TextureSnapshot>,
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

    fn record(&mut self, snap: TextureSnapshot) {
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
    fn undo(&mut self, current: TextureSnapshot) -> Option<TextureSnapshot> {
        let snap = self.undo.pop()?;
        self.redo.push(current);
        Some(snap)
    }

    /// Pops the state to restore to, pushing `current` onto the undo stack.
    fn redo(&mut self, current: TextureSnapshot) -> Option<TextureSnapshot> {
        let snap = self.redo.pop()?;
        self.undo.push(current);
        Some(snap)
    }

    fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
    }
}

fn snapshot_of(mesh: &MeshData) -> Option<TextureSnapshot> {
    let tex = mesh.texture.as_ref()?;
    Some(TextureSnapshot {
        width: tex.width,
        height: tex.height,
        rgba: tex.rgba.clone(),
    })
}

/// Persisted UI state (dock layout, tool settings, camera).
#[derive(serde::Serialize, serde::Deserialize)]
struct UiMemory {
    dock: DockState<Panel>,
    panel_visible: [bool; 3],
    active_tool: usize,
    channels: [bool; 6],
    brush_size: f32,
    brush_hardness: f32,
    brush_opacity: f32,
    brush_spacing: f32,
    brush_color: [u8; 4],
    show_uv_overlay: bool,
    preview_bg: TexturePreviewBg,
    preview_bg_color: [u8; 4],
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

const TOOLS: [&str; 4] = ["Brush", "Eraser", "Fill", "Pick"];

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
    let color_image =
        egui::ColorImage::from_rgba_unmultiplied([img.width() as usize, img.height() as usize], img.as_raw());
    Some(ctx.load_texture(
        "pick_icon",
        color_image,
        egui::TextureOptions::LINEAR,
    ))
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
const CHANNELS: [&str; 6] = [
    "Albedo / Color",
    "Normal",
    "Height / Displacement",
    "Roughness",
    "Metallic",
    "Emissive",
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
            panel_visible: [true, true, false],
            active_tool: 0,
            channels: [true, true, false, false, false, false],
            brush_size: 24.0,
            brush_hardness: 0.5,
            brush_opacity: 1.0,
            brush_spacing: 6.0,
            brush_color: [90, 160, 255, 255],
            stroke_last: None,
            stroke_active: false,
            history: EditHistory::new(24),
            atlas_res: 512,
            brush_menu_open: false,
            brush_menu_pos: None,
            pick_icon: None,
            needs_texture_upload: false,
            texture_preview: None,
            preview_gen: 1,
            show_uv_overlay: true,
            preview_bg: TexturePreviewBg::Checker,
            preview_bg_color: [128, 128, 128, 255],
            show_tool_strip: true,
            tool_strip_anim: 1.0,
            restore_view: None,
            status: "Default sphere and material — File > Open to load a .gltf/.glb".to_string(),
        };

        if let Some(mem) = memory {
            core.panel_visible = mem.panel_visible;
            core.active_tool = mem.active_tool;
            core.channels = mem.channels;
            core.brush_size = mem.brush_size;
            core.brush_hardness = mem.brush_hardness;
            core.brush_opacity = mem.brush_opacity;
            core.brush_spacing = mem.brush_spacing;
            core.brush_color = mem.brush_color;
            core.show_uv_overlay = mem.show_uv_overlay;
            core.preview_bg = mem.preview_bg;
            core.preview_bg_color = mem.preview_bg_color;
            core.show_tool_strip = mem.show_tool_strip;
            core.tool_strip_anim = if core.show_tool_strip { 1.0 } else { 0.0 };
            // Hidden panels were removed from the dock when they were unchecked;
            // re-apply that so a restored layout doesn't resurrect closed tabs.
            for (i, panel) in Panel::ALL.iter().enumerate() {
                if !core.panel_visible[i] && *panel != Panel::Viewport {
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
                self.core.preview_gen += 1;
                self.core.stroke_active = false;
                self.core.stroke_last = None;
                self.core.history.clear();
                self.core.status = format!("Loaded {path}");
            }
            crate::io::LoadedModel::Invalid => {
                self.core.status = format!("Failed to load {path}");
            }
        }
    }

    fn export_albedo(&mut self, path: &str) {
        match self.core.mesh.as_ref().and_then(|m| m.texture.as_ref()) {
            Some(tex) => {
                let (tw, th) = (tex.width, tex.height);
                match crate::io::save_atlas_png(path, tex) {
                    Ok(()) => {
                        self.core.status =
                            format!("Exported albedo atlas ({tw}x{th}) to {path}");
                    }
                    Err(e) => self.core.status = format!("Export failed: {e}"),
                }
            }
            None => self.core.status = "Nothing to export — no texture atlas".to_string(),
        }
    }

    fn save_ui_memory(&self) {
        let mem = UiMemory {
            dock: self.dock_state.clone(),
            panel_visible: self.core.panel_visible,
            active_tool: self.core.active_tool,
            channels: self.core.channels,
            brush_size: self.core.brush_size,
            brush_hardness: self.core.brush_hardness,
            brush_opacity: self.core.brush_opacity,
            brush_spacing: self.core.brush_spacing,
            brush_color: self.core.brush_color,
            show_uv_overlay: self.core.show_uv_overlay,
            preview_bg: self.core.preview_bg,
            preview_bg_color: self.core.preview_bg_color,
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
        self.core.panel_visible[panel.index()] = visible;

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
    let [_old, _left] = main.split_left(NodeIndex::root(), 0.2, vec![Panel::Channels]);
    let [_old, _right] = main.split_right(NodeIndex::root(), 0.25, vec![Panel::Texture]);
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
    // A legacy session may have persisted Custom as the old near-black default
    // ([15,15,15,255]); upgrade it so erased texels aren't "black" anymore.
    let mut mem = mem;
    if mem.preview_bg == TexturePreviewBg::Custom && mem.preview_bg_color == [15, 15, 15, 255] {
        mem.preview_bg_color = [128, 128, 128, 255];
    }
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

        // Blender-style header: a persistent tool strip pinned under the menu
        // bar, always visible regardless of dock layout.
        // Blender-style header: a persistent tool strip pinned under the menu
        // bar, always visible regardless of dock layout.
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
                            self.core
                                .mesh
                                .as_ref()
                                .and_then(|m| m.texture.as_ref())
                                .is_some(),
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
                            .add_enabled(can_undo, egui::Button::new("Undo").shortcut_text("Ctrl+Z"))
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
                            let mut visible = self.core.panel_visible[idx];
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
        let mut toggle_tool_strip = false;
        ui.ctx().input_mut(|i| {
            do_undo = i.consume_key(undo_cmd, egui::Key::Z);
            do_redo = (i.modifiers.command && i.modifiers.shift && i.key_pressed(egui::Key::Z))
                || i.consume_key(redo_cmd, egui::Key::Z)
                || i.consume_key(undo_cmd, egui::Key::Y);
            toggle_tool_strip = i.consume_key(egui::Modifiers::NONE, egui::Key::T);
        });
        if do_undo {
            self.undo();
        }
        if do_redo {
            self.redo();
        }
        if toggle_tool_strip {
            self.core.show_tool_strip = !self.core.show_tool_strip;
            self.core.status = if self.core.show_tool_strip {
                "In-viewport tools: on (T to toggle)".to_string()
            } else {
                "In-viewport tools: off (T to toggle)".to_string()
            };
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

fn snapshot_of_current(core: &Core) -> TextureSnapshot {
    core.mesh
        .as_ref()
        .and_then(|m| m.texture.as_ref())
        .map(|tex| TextureSnapshot {
            width: tex.width,
            height: tex.height,
            rgba: tex.rgba.clone(),
        })
        .unwrap_or_else(|| TextureSnapshot {
            width: 0,
            height: 0,
            rgba: Vec::new(),
        })
}

/// Restores a snapshot as the mesh's texture, scheduling a GPU re-upload and
/// preview rebuild. Handles dimension changes (update_texture recreates the
/// texture when the size differs).
fn restore_snapshot(core: &mut Core, snap: TextureSnapshot) {
    if let Some(mesh) = core.mesh.as_mut() {
        mesh.texture = Some(crate::io::TextureData {
            width: snap.width,
            height: snap.height,
            rgba: snap.rgba,
        });
    }
    core.stroke_active = false;
    core.stroke_last = None;
    core.needs_texture_upload = true;
    core.preview_gen += 1;
}

/// Common bookkeeping after a texture swap (resize / blank): abort any active
/// stroke, schedule a GPU upload and preview rebuild, update the status line.
fn finish_texture_change(core: &mut Core, status: String) {
    core.stroke_active = false;
    core.stroke_last = None;
    core.needs_texture_upload = true;
    core.preview_gen += 1;
    core.status = status;
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
        }
    }
}

fn viewport_ui(ui: &mut Ui, core: &mut Core) {
    let full_rect = ui.max_rect();

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
        let dir = if target > core.tool_strip_anim { 1.0 } else { -1.0 };
        core.tool_strip_anim = (core.tool_strip_anim + dir * dt / STRIP_ANIM_S).clamp(0.0, 1.0);
        ui.ctx().request_repaint();
    }
    let anim = core.tool_strip_anim;
    let side = (STRIP_W - 6.0).max(18.0);
    let content_h = STRIP_PAD * 2.0 + TOOLS.len() as f32 * side;
    // Slides between "flush with the left edge" and "fully hidden, pushed
    // STRIP_HIDE_EXTRA past the edge" — so when it hides, nothing stays on
    // screen at the edge.
    let total = STRIP_W + STRIP_HIDE_EXTRA;
    let strip_rect = egui::Rect::from_min_size(
        full_rect.min + egui::vec2(-total * (1.0 - anim), STRIP_TOP_INSET),
        egui::vec2(STRIP_W, content_h),
    );

    // The offscreen texture always spans the full viewport.
    let size = full_rect.size();
    let (w, h) = (size.x.max(1.0) as u32, size.y.max(1.0) as u32);

    // Ensure viewport resources exist and match the current size.
    let needs_create = core.viewport.is_none();
    if let Some(vp) = core.viewport.as_mut() {
        if vp.textures.resize(&core.device, w, h) {
            // The offscreen texture was recreated, so the egui native texture must be too.
            if let Some(old) = vp.texture_id.take() {
                core.egui_renderer.write().free_texture(&old);
            }
        }
    }
    let just_created = core.viewport.is_none();

    if needs_create || just_created {
        let textures = ViewportTextures::new(&core.device, w, h);
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

    // Re-register the viewport texture with egui when it changes size.
    let vp = core.viewport.as_mut().unwrap();
    if vp.texture_id.is_none() {
        let id = core.egui_renderer.write().register_native_texture(
            &core.device,
            &vp.textures.color_view,
            wgpu::FilterMode::Nearest,
        );
        vp.texture_id = Some(id);
    }
    let _ = vp;

    // Handle camera interaction. LMB is painting; MMB orbits and Shift+MMB
    // pans. RMB is reserved for future tools. Wheel zooms the camera, or
    // (with Shift) resizes the brush. No widget is allocated here — inputs are
    // read straight from the context while the pointer is over the viewport rect.
    let rect = full_rect;
    let hovered = ui.rect_contains_pointer(rect) && !ui.rect_contains_pointer(strip_rect);

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
                        match core.active_tool {
                            0 | 1 | 2 => {
                                // One undo step per stroke (or per fill press).
                                if !core.stroke_active {
                                    if let Some(snap) = snapshot_of(mesh) {
                                        core.history.record(snap);
                                    }
                                    core.stroke_active = true;
                                }
                            }
                            _ => {}
                        }
                        match core.active_tool {
                            0 | 1 => {
                                // Step dabs along the drag so fast strokes don't gap.
                                let from = core.stroke_last.unwrap_or(pos);
                                // Spacing 0 = continuous: step at half the brush
                                // radius so successive dabs always overlap; a
                                // positive value steps at that many pixels.
                                let spacing = if core.brush_spacing > 0.0 {
                                    core.brush_spacing
                                } else {
                                    (core.brush_size / 2.0).max(1.0)
                                };
                                let dabs = crate::paint::stamp_positions(
                                    glam::Vec2::new(from.x, from.y),
                                    glam::Vec2::new(pos.x, pos.y),
                                    spacing,
                                );
                                core.stroke_last = Some(pos);
                                let mode = if core.active_tool == 0 {
                                    crate::paint::StampMode::Paint
                                } else {
                                    crate::paint::StampMode::Erase
                                };
                                let color = if core.active_tool == 0 {
                                    core.brush_color
                                } else {
                                    [0, 0, 0, 0]
                                };
                                for dab in dabs {
                                    let (dx, dy) = viewport_ndc(dab.x, dab.y, rect);
                                    let (o, d) = vp.camera.ray(dx, dy);
                                    if let Some(hi) = crate::paint::mesh_raycast(mesh, o, d) {
                                        let world_r = screen_to_world_radius(
                                            &vp.camera, hi.position, core.brush_size, rect, w, h,
                                        );
                                        crate::paint::apply_stamp(
                                            mesh,
                                            hi.position,
                                            world_r,
                                            color,
                                            core.brush_opacity,
                                            core.brush_hardness,
                                            mode,
                                        );
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
                                if let Some(tex) = mesh.texture.as_ref() {
                                    picked = Some(crate::paint::pick_color(tex, &hit));
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
                core.status = format!(
                    "Picked rgb({}, {}, {}) — back to Brush",
                    c[0], c[1], c[2]
                );
            }
            if painted {
                core.needs_texture_upload = true;
                core.preview_gen += 1;
                if core.active_tool == 1 {
                    let fill = match core.preview_bg {
                        TexturePreviewBg::Checker => "Checker".to_string(),
                        TexturePreviewBg::Custom => {
                            let [r, g, b, _] = core.preview_bg_color;
                            format!("#{r:02X}{g:02X}{b:02X}")
                        }
                    };
                    core.status = format!("Erased — transparent fill: {fill}");
                }
                ui.ctx().request_repaint();
            }
        }
    }
    // End the stroke when the button is released or the pointer leaves.
    if !ui.input(|i| i.pointer.primary_down()) || !hovered {
        core.stroke_active = false;
        core.stroke_last = None;
    }

    // Push any freshly painted texels to the GPU before the render below.
    if core.needs_texture_upload {
        if let (Some(tex), renderer) = (
            core.mesh.as_ref().and_then(|m| m.texture.as_ref()),
            &mut core.renderer,
        ) {
            renderer.update_texture(tex);
        }
        core.needs_texture_upload = false;
    }

    // Transparent (erased) texels are filled with the same style the Texture
    // preview shows, composited flat onto the mesh inside the render.
    let bg = match core.preview_bg {
        TexturePreviewBg::Checker => crate::render::Background {
            color_a: [0.38, 0.38, 0.40],
            color_b: [0.31, 0.31, 0.33],
            checker_on: true,
            scale: 16.0,
        },
        TexturePreviewBg::Custom => {
            let [r, g, b, _] = core.preview_bg_color;
            crate::render::Background {
                color_a: [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0],
                color_b: [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0],
                checker_on: false,
                scale: 16.0,
            }
        }
    };
    core.renderer.set_background(bg);

    // Render the 3D scene into the offscreen viewport texture.
    if let Some(vp) = core.viewport.as_ref() {
        core.renderer.render(
            &vp.camera,
            &vp.textures.color_view,
            &vp.textures.depth_view,
        );
    }

    // Draw the offscreen texture across the whole viewport (under the T-bar).
    // The render is fully opaque — transparency was already composited into it.
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

    // Brush preview circle: a fixed-size ring in screen pixels matching the
    // brush radius (Paint/Eraser only). Shift+wheel in the viewport resizes it.
    if (core.active_tool == 0 || core.active_tool == 1)
        && hovered
        && !navigating
    {
        let pos = ui.input(|i| i.pointer.hover_pos());
        if let Some(pos) = pos {
            let screen_r = core.brush_size;
            let (fill, stroke, dot) = if core.active_tool == 0 {
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
            ui.painter().circle_filled(pos, screen_r, fill);
            ui.painter().circle_stroke(pos, screen_r, egui::Stroke::new(1.5, stroke));
            ui.painter().circle_stroke(pos, 2.0, egui::Stroke::new(1.0, dot));
            ui.ctx().request_repaint();
        }
    }

    // Pick tool: hide the OS cursor and draw the pipette icon (lucide, ISC)
    // centered so its tip sits near the pointer.
    if core.active_tool == 3 && hovered && !navigating && !core.brush_menu_open {
        ui.ctx().set_cursor_icon(egui::CursorIcon::None);
        if let (Some(icon), Some(pos)) = (core.pick_icon_tex(ui.ctx()), ui.input(|i| i.pointer.hover_pos())) {
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
            ui.label("LMB paint  |  MMB drag: orbit  |  Shift+MMB drag: pan  |  Wheel: zoom  |  Shift+Wheel: brush size  |  RMB: brush menu  |  F: fit  |  T: tools on/off");
        },
    );
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

/// One square tool button inside the in-viewport T-bar. Icons are hand-drawn
/// except the Pick tool, which reuses the lucide pipette image (ISC licensed).
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
            // Brush: filled dab with a guide ring.
            p.circle_filled(center, 3.0, c);
            p.circle_stroke(center, 6.0, egui::Stroke::new(1.5, c));
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
                let img_rect = egui::Rect::from_center_size(
                    center,
                    egui::vec2(side - 8.0, side - 8.0),
                );
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
        ui.label(format!("#{:02X}{:02X}{:02X}", cur[0], cur[1], cur[2]));
    });
    let mut col = egui::Color32::from_rgba_unmultiplied(cur[0], cur[1], cur[2], cur[3]);
    let changed = egui::color_picker::color_picker_color32(
        ui,
        &mut col,
        egui::color_picker::Alpha::Opaque,
    );
    if changed {
        core.brush_color = col.to_srgba_unmultiplied();
        core.status = format!(
            "Brush color #{:02X}{:02X}{:02X}",
            core.brush_color[0], core.brush_color[1], core.brush_color[2]
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
        let resp = ui.add(pick_btn).on_hover_text(
            "Switches to the Pick tool — click a spot on the model to sample it",
        );
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

    ui.weak("Brush types — coming soon");
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
        resp.clone().on_hover_text(format!("#{:02X}{:02X}{:02X}", rgb[0], rgb[1], rgb[2]));
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
        ui.add(egui::Slider::new(&mut core.brush_hardness, 0.0..=1.0));
        ui.label("Opacity");
        ui.add(egui::Slider::new(&mut core.brush_opacity, 0.0..=1.0));
        ui.label("Spacing");
        ui.add(
            egui::Slider::new(&mut core.brush_spacing, 0.0..=64.0)
                .suffix("px")
                .logarithmic(true)
                .max_decimals(0),
        )
        .on_hover_text("0 = continuous (steps tuned to brush size). Positive = fixed distance (px) between dabs along a stroke.");
    });
}

fn channels_ui(ui: &mut Ui, core: &mut Core) {
    ui.heading("Material Channels");
    ui.separator();
    for (i, label) in CHANNELS.iter().enumerate() {
        ui.checkbox(&mut core.channels[i], *label);
    }
}

fn texture_ui(ui: &mut Ui, core: &mut Core) {
    ui.heading("Texture Preview");
    ui.horizontal(|ui| {
        ui.checkbox(&mut core.show_uv_overlay, "Show UV overlay");
        ui.separator();
        ui.label("Transparent fill:");
        ui.selectable_value(&mut core.preview_bg, TexturePreviewBg::Checker, "Checker");
        ui.selectable_value(&mut core.preview_bg, TexturePreviewBg::Custom, "Custom");
        if core.preview_bg == TexturePreviewBg::Custom {
            if ui
                .color_edit_button_srgba_unmultiplied(&mut core.preview_bg_color)
                .changed()
            {
                // Edited in place (straight alpha), nothing else to do.
            }
        }
    });

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
            let has_tex = core
                .mesh
                .as_ref()
                .and_then(|m| m.texture.as_ref())
                .is_some();
            let resize_clicked = has_tex && ui.button("Resize").clicked();
            let blank_clicked = ui
                .button("Blank…")
                .on_hover_text("Replace with a clean canvas at the selected size")
                .clicked();
            if resize_clicked {
                let target = core.atlas_res;
                if let Some(mesh) = core.mesh.as_mut() {
                    if let Some(tex) = mesh.texture.as_ref() {
                        if tex.width.max(tex.height) != target {
                            let snap = snapshot_of(mesh);
                            let new_tex = crate::io::resize_atlas(tex, target);
                            if let Some(snap) = snap {
                                core.history.record(snap);
                            }
                            mesh.texture = Some(new_tex);
                            finish_texture_change(core, format!("Resized atlas to {target}px"));
                        }
                    }
                }
            }
            if blank_clicked {
                if let Some(mesh) = core.mesh.as_mut() {
                    let (cw, ch) = mesh
                        .texture
                        .as_ref()
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
                    let snap = snapshot_of(mesh);
                    let new_tex = crate::io::blank_atlas(bw, bh, [240, 240, 240, 255]);
                    if let Some(snap) = snap {
                        core.history.record(snap);
                    }
                    mesh.texture = Some(new_tex);
                    finish_texture_change(core, format!("New blank canvas {bw}x{bh}"));
                }
            }
        });
        ui.separator();
    }

    let gen = core.preview_gen;
    let tex = core.mesh.as_ref().and_then(|m| m.texture.as_ref());

    if let Some(tex) = tex {
        if core.texture_preview.as_ref().map(|p| p.gen) != Some(gen) {
            let pixels = tex
                .rgba
                .chunks_exact(4)
                .map(|px| egui::Color32::from_rgba_unmultiplied(px[0], px[1], px[2], px[3]))
                .collect();
            let img = ColorImage::new([tex.width as usize, tex.height as usize], pixels);
            let handle = ui.ctx().load_texture(
                format!("albedo_preview_{gen}"),
                img,
                egui::TextureOptions::NEAREST,
            );
            core.texture_preview = Some(PreviewTexture { gen, handle });
        }

        if let Some(p) = core.texture_preview.as_ref() {
            let avail = ui.available_size();
            let img_size = egui::vec2(avail.x, avail.x.min(avail.y));
            let (rect, _) = ui.allocate_exact_size(img_size, egui::Sense::hover());
            // Fill the transparent (erased) land behind the atlas per the
            // chosen style; the alpha-blended image is drawn on top.
            match core.preview_bg {
                TexturePreviewBg::Checker => draw_checkerboard(ui, rect),
                TexturePreviewBg::Custom => {
                    let [r, g, b, a] = core.preview_bg_color;
                    ui.painter().rect_filled(
                        rect,
                        0.0,
                        egui::Color32::from_rgba_unmultiplied(r, g, b, a),
                    );
                }
            }
            ui.painter().image(
                p.handle.id(),
                rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
            draw_uv_overlay(ui, rect, core);
            ui.label(format!("{} × {} (packed atlas)", tex.width, tex.height));
        }
    } else {
        let avail = ui.available_size();
        let img_size = egui::vec2(avail.x, avail.x.min(avail.y));
        let (rect, _) = ui.allocate_exact_size(img_size, egui::Sense::hover());
        ui.painter().rect_filled(rect, 0.0, egui::Color32::from_gray(40));
        draw_uv_overlay(ui, rect, core);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ui_memory_round_trips() {
        let dock = default_dock();
        let mem = UiMemory {
            dock,
            panel_visible: [true, true, false],
            active_tool: 2,
            channels: [true, false, true, false, false, true],
            brush_size: 42.0,
            brush_hardness: 0.25,
            brush_opacity: 0.75,
            brush_spacing: 12.0,
            brush_color: [12, 34, 56, 255],
            show_uv_overlay: false,
            preview_bg: TexturePreviewBg::Checker,
            preview_bg_color: [128, 128, 128, 255],
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
        assert_eq!(back.show_uv_overlay, false);
        assert_eq!(back.camera.unwrap().radius, 4.25);

        // The fixup path must keep every panel present.
        let all: Vec<Panel> = back
            .dock
            .iter_all_tabs()
            .map(|(_, tab)| *tab)
            .collect();
        for panel in Panel::ALL {
            assert!(all.contains(&panel), "panel {panel:?} missing after restore");
        }
    }

    fn snap(r: u8) -> TextureSnapshot {
        TextureSnapshot {
            width: 2,
            height: 2,
            rgba: vec![r; 16],
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
        assert_eq!(s1.rgba, snap(2).rgba);
        assert!(h.can_redo());

        let cur = snap(2);
        let s2 = h.undo(cur).expect("undo 2");
        assert_eq!(s2.rgba, snap(1).rgba);
        assert!(h.can_redo());

        // Redo restores the most recent undone state.
        let cur = snap(1);
        let r1 = h.redo(cur).expect("redo 1");
        assert_eq!(r1.rgba, snap(2).rgba);
        assert!(h.can_redo());

        let cur = snap(2);
        let r2 = h.redo(cur).expect("redo 2");
        assert_eq!(r2.rgba, snap(3).rgba);
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
        let mesh = crate::io::MeshData::uv_sphere(0.6, 4, 6).with_texture(crate::io::default_albedo());
        let s = snapshot_of(&mesh).expect("texture");
        assert_eq!(s.width, mesh.texture.as_ref().unwrap().width);
        assert_eq!(s.height, mesh.texture.as_ref().unwrap().height);
        assert_eq!(s.rgba.len(), mesh.texture.as_ref().unwrap().rgba.len());
    }
}

/// Classic two-tone alpha checkerboard drawn behind the atlas preview so
/// transparent (erased) texels are clearly visible.
fn draw_checkerboard(ui: &Ui, rect: egui::Rect) {
    let square = (rect.width() / 24.0).ceil().max(8.0);
    let colors = [egui::Color32::from_gray(96), egui::Color32::from_gray(80)];
    let painter = ui.painter();
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
/// interior edges stay thin and faded.
fn draw_uv_overlay(ui: &mut Ui, rect: egui::Rect, core: &Core) {
    if !core.show_uv_overlay {
        return;
    }
    let Some(mesh) = core.mesh.as_ref() else { return };
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

    let painter = ui.painter_at(rect);
    let boundary_stroke = egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 214, 96));
    let interior_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(120, 190, 255, 190));

    for ((a, b), count) in &edge_count {
        let (u0, v0) = mesh.uvs[*a as usize];
        let (u1, v1) = mesh.uvs[*b as usize];
        painter.line_segment(
            [to_pos(u0, v0), to_pos(u1, v1)],
            if *count == 1 { boundary_stroke } else { interior_stroke },
        );
    }
}
