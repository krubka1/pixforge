# PixForge Architecture

PixForge is a stylized 3D texture painter: you paint straight onto the albedo
texture of a .gltf/.glb model in a PBR-shaded 3D viewport, with a classic 2D
texture editor alongside. Everything is CPU-side texture editing; the GPU is
only used for display and for the brush cursor overlay.

Rust 2021, `egui`/`eframe` (egui-wgpu) for UI and windowing, `wgpu` for the
3D viewport, `glam` for math, `image` for PNG encoding, `gltf` for model
loading. See `Cargo.toml` for the exact set.

## Source layout

```
src/main.rs       entry point (there is also a hidden --gen-brush-packs CLI)
src/app.rs        the whole UI: docked panels, tools, strokes, undo/redo, state
src/paint.rs      the painting engine (3D and 2D stamps, raycast, fills)
src/brushes.rs    brush library + procedural texture generators
src/io.rs         data model (TextureData/Layer/MeshData) + load/save + compositing
src/render.rs     wgpu renderer, camera, PBR material, viewport, brush overlay
src/project.rs    custom .pixforge binary save/load format
src/shader.wgsl   WGSL shaders (mesh PBR, background, overlay)
assets/           pipette + lucide icon PNGs
brushes/          shipped brush packs (procedurally generated PNGs)
```

## Data model (`io.rs`)

- `TextureData { width, height, rgba }` — one flat RGBA8 atlas.
- `Layer { name, visible, opacity, blend, locked, roughness, metallic,
  emissive, ambient_occlusion, height, bump_strength, texture }` — a stacked
  albedo atlas plus per-layer scalar surface material / height / bump values.
  All layers of one mesh share one atlas size.
- `MeshData { positions, normals, uvs, indices, layers, active_layer, dirty }` —
  geometry + the layer stack. `dirty` is an inclusive texel rect
  `(x0, y0, x1, y1)` of texels edited since the last GPU upload, so a frame
  only recomposites/re-uploads the touched region.
- `BlendMode { Normal, Multiply, Screen, Overlay }` — per-layer albedo blending.

### Compositing (all CPU, in `io.rs`)

- `flattened_atlas()` — the visible layers, bottom to top, source-over,
  each scaled by opacity and blended by `blend`. Returns `None` only for a
  mesh with zero layers (renders as white fallback).
- `flattened_material_atlas()` — RGBA texture where R=roughness, G=metallic,
  B=emissive/3, A=ambient occlusion, seeded with the default material and
  composited source-over weighted by each layer's paint coverage (its albedo
  alpha × opacity).
- `flattened_height_atlas()` — separate RGBA atlas: R = signed height encoded
  as `(height+1)*0.5` (128 = flat), G = bump strength /8, same source-over
  weighting as the material map.
- `flattened_atlas_region(x0,y0,w,h)` — region-only compositing used to patch
  the 2D preview texture after a stroke.

## Painting engine (`paint.rs`)

Core entry points:

- `apply_stamp_with` / `apply_stamp` / `apply_stamp_rect` — 3D stamps over a
  `MeshData`, delegating to the private `stamp_texels`.
- `stamp_2d` — 2D UV-space stamp on a bare `TextureData` (no mesh), for the
  texture editor.
- `stamp_fill_2d` and `fill_region` — scanline flood fills (2D texture / 3D
  mesh seed triangle).
- `mesh_raycast` — nearest-triangle ray hit with interpolated position + UV,
  used both for picking and (via the accelerated occlusion grid) per-texel
  sight lines.
- `brush_axes`, `brush_plane_circle`, `brush_radius_world`,
  `hit_texel_scale`, `stamp_positions`, `spaced_freehand_dabs`.

### 3D stamping (`stamp_texels`)

For each dab it:

1. Determines the **footprint**: `Round`/`Square`/`Diamond`/`Sprite` from
   `BrushStyle`, or `Rect` from the rect tool. `Textured` with no sprite falls
   back to a round dab.
2. Computes **brush-local axes** (`brush_axes`) — a tangent-plane frame that is
   screen-upright, shared with the cursor overlay so preview matches output.
3. Iterates only the triangles your stamp can *see* the front of
   (`facing_gate`: `normal·(-view_dir) > 0`), conservatively culled by
   bounding distance / rect overlap.
4. For each texel inside the triangle's UV bounding box it computes the 3D
   position by UV barycentric coordinates, applies footprint inclusion +
   edge falloff, then an **occlusion gate**:
   - fast path: a closed convex mesh seen from outside the bounding sphere
     hides a texel exactly when its outward normal points away from the eye
     (single dot product). Detected by `mesh_is_convex` (welds near-duplicate
     pole/seam vertices, checks water tightness and the support-plane test).
   - slow path: a uniform grid index over triangles
     (`OwnedOcclusionGrid`, built once per stroke) raycasts the texel's sight
     line against only the cells under the segment.
5. Applies the **dab profile**: hardness falloff for brushes, transparent-core
   feather for the eraser, and for sprite stamps the sprite's own alpha as the
   coverage (the sprite stamps once per dab, stretched to span the dab
   diameter, rotated/flipped by `BrushStyle`).
6. Blends with `blend_pixel` (straight-alpha pull toward the brush color: BOTH
   rgb and alpha are lerped, so a translucent color really produces a
   translucent texel) or `erase_pixel`.
7. In non-accumulate mode uses a per-stroke `stroke_alpha` buffer holding the
   max `min(opacity, cover)` per texel, so overlapping dabs cap instead of
   stacking and the live preview equals the final composited stroke.

### Occlusion / acceleration (`StampAccel`)

Built once per stroke and reused by every dab: convexity + bounding sphere,
the occlusion grid (when non-convex), and the split-lock connected components.
Geometry never changes during a stroke, so the cache is valid for its whole
lifetime. `GridIndex` (per-cell triangle lists) + `grid_nearest_before`
(the segment-AABB cell walk, wrapped by `OwnedOcclusionGrid::nearest_before`)
implement the slow-path occlusion test.

### Split lock

`stamp_texels` skips every triangle not in the same connected component as the
face under the brush at stroke start (`triangle_components_edge` gives
per-triangle components; the accel caches the seed component). This prevents a
stroke from bleeding onto a separate model part that happens to fall inside
its radius.

## Rendering (`render.rs`)

- `Camera` — orbit/zoom/pan turntable around `target`, `ray()` for picking,
  DirectX-style RH perspective.
- `Renderer` — wgpu pipelines: opaque mesh, translucent mesh, background
  (sky env), and two brush-overlay pipelines (dark scrim pass + emissive glow
  pass). Owns the GPU buffers/textures for albedo, material map, height map,
  environment map and the brush sprite.
- `Material` — the PBR + lighting state (roughness/metallic/emissive/ao,
  sun, env, exposure, fill light). The per-layer surface params ride the
  material atlas; these are the viewport-wide lighting knobs.
- `UV_OVERLAY_CHECKER` / `UV_OVERLAY_GRID` — shader-driven UV debug overlays.
- `BrushOverlay` — the cursor mask. The mesh is retraced with a fragment
  shader that discards everything outside the footprint, so the cursor
  conforms to the model instead of being a projected 2D shape. The overlay
  uniforms are emitted from `app.rs` (`core.renderer.brush_overlay`).
- `ViewportTextures` — offscreen color/depth render targets rendered each
  frame by `Renderer::render(...)`, registered as an egui native texture and
  shown as the "3D Viewport" panel.

### Shader (`shader.wgsl`)

- `vs_main`/`fs_main`: GGX metallic-roughness shading with an analytic sky
  (`env_sky*`) and, when an environment map is bound, mipmapped
  rgba16f equirect IBL. Height-map-driven normal perturbation
  (`perturb_normal`/`height_at`). Two passes: opaque pass draws only fully
  opaque texels (so translucent texels write no depth and never occlude
  surfaces behind them); the translucent pass source-over blends
  `0 < alpha < 1` texels over the result. ACES tone mapping.
- `bg_vs`/`bg_fs`: fullscreen background quad un-projecting to world rays to
  render the sky.
- `overlay_fs`: the brush-cursor mask (scrim/glow pass selected by uniform,
  sprite sampled for texture shapes).

## UI (`app.rs`)

`PixForgeApp { dock_state, core: Core }`. `Core` holds everything: renderer,
mesh, tools, brush parameters, material, stroke state, undo history, 2D canvas.

### Panels (docked tabs)

`Panel::{ Viewport, Channels, Texture, Layers, Lighting, Brushes, Preferences }`
layout comes from `egui_dock` (`DockState`, persisted as part of UI memory).
Each `*_ui` function renders one tab.

- **Viewport** — the 3D scene; hover-orbit/pan/zoom; stroke input; Blender-style
  sliding T-bar of tools; a horizontal UV-overlay bar; right-click brush menu.
- **Channels** — per-layer material sliders (roughness/metallic/emissive/AO).
- **Texture** — the 2D draw canvas (`Canvas2D` camera: pan/zoom, checkerboard
  backdrop), resolution resizer (`Resize`/`Blank`), and its own brush picker
  strip. Painting calls the 2D `stamp_*` functions.
- **Layers** — layer list: add/delete/duplicate/reorder, rename, visibility,
  opacity, blend mode, lock, and flip.
- **Lighting** — sun/ environment / exposure / fill controls (the `Material`).
- **Brushes** — the `BrushLibrary` browser (thumbnails, category filter, Rescan).
- **Preferences** — theme + remappable shortcuts.

### Tools

`TOOLS = ["Brush","Eraser","Fill","Pick","Rect"]` (enums `ShortcutAction`
mirror them). `active_tool` drives both viewports' strips and the 3D
overlay shape.

### Strokes

`StrokeState` tracks the in-progress stroke: previous cursor pos, stroke
start (Shift = straight line), the spacing accumulator (`spaced_freehand_dabs`,
dabs `brush_spacing` screen px apart, `<=0` = one dab per frame), the shared
`StampAccel`, and the non-accumulate stroke-alpha buffer. `stroke` is for the
3D viewport, `stroke_2d` for the texture editor. Brush size is converted
screen→world at the hit point via `screen_to_world_radius`.

### Undo/redo

`EditHistory` keeps full `LayerStackSnapshot`s (bounded, 24 steps). Every
edit records `snapshot_of(mesh)` before mutating; undo/redo swap whole layer
stacks and trigger a texture/material re-upload.

### Incremental GPU uploads

After a stamp the mesh's `dirty` region is used for a region upload
(`update_texture_region`) and the 2D preview is patched with
`set_partial` (`preview_patch`). The material/height map recomposites are
coalesced to at most one upload per ~30 ms while a stroke is active. The 3D
viewport renders at a reduced internal resolution (`vp_scale`) during
interaction and snaps back to 1.0 when idle.

### Persistence

- `~/.config/pixforge/ui_layout.msgpack` (msgpack via `rmp-serde`): `UiMemory`
  with dock layout, panel visibility, tool/brush/channel state, material,
  theme, camera, and `Shortcuts`.
- Shortcuts are serialized as a *named map* (action id → binding) so struct
  changes never mis-assign stored bindings.
- Model/layers go to `.pixforge` via `project.rs`.

## Brush library (`brushes.rs`)

- `BrushLibrary { folder, entries, selected, ... }` — always carries a set of
  built-in brushes and augments them with every supported file under
  `<cwd>/brushes` (or `$PIXFORGE_BRUSHES`). Subfolders become categories.
  Rescans are throttled to ~1.5 s; `signature` change triggers a rebuild.
- `BrushEntry { name, category, kind, sprite, path }`, `BrushKind::Shape|Texture`.
- Supported files: image masks (PNG/JPG/BMP/WebP/GIF; transparent = dab, fully
  opaque uses inverted luminance) and GIMP `.gbr` v1 brushes (`parse_gbr`).
- Built-in shapes: round/square/diamond plus procedural texture stamps.
- `generate_ship_pack()` — the hidden CLI that regenerates every PNG under
  `brushes/` from the procedural generators (`main.rs --gen-brush-packs`).

## Project format (`project.rs`)

`.pixforge` magic `PIXFORGE\0` + little-endian scalar fields for geometry
(positions/normals/UVs/indices), active layer, then per layer: name,
visibility, opacity, atlas, blend byte, four material f32s, height+bump f32s,
locked byte. Atlas bytes are embedded as PNGs so a multi-megabyte canvas stays
small. Versioned (currently 5); readers default missing fields (older files
load fine) and reject newer versions.

## I/O (`io.rs`)

- `load_gltf`: imports a .gltf/.glb, packs each distinct base-color image into
  a power-of-two atlas, remaps UVs into slots, and applies the scene-graph
  world transforms (skinned meshes get bind pose). Multi-node models keep
  their layout.
- `save_glb`: bakes the flattened albedo material + height atlases into a .glb.
- `save_atlas_png` / `load_image_into_atlas` / `brush_sprite`
  (`import_image_to_layer` in app).
- `load_environment`: HDR or LDR equirect → mipmapped rgba16f `EnvironmentMips`
  for IBL.
- `remake_uv`/`flip_texture`/`resize_atlas`/`blank_atlas`/`default_albedo`/
  `uv_sphere` (with a duplicated seam column so the texture wraps cleanly).

## Tests

99 passing unit tests + 2 ignored (CPU-side; wgpu tests run in one module
using a headless adapter). The suites cover: stamp footprints and falloff,
occlusion/through-walls, sprite stamps, stroke-spacing accumulation,
non-accumulate capping, split lock, fills, texel↔UV conversions, project
round-trips + versioning, renderer overlays/lighting, and undo/redo history.

## Performance notes baked into the design

- Per-stroke `StampAccel` avoids repeated O(V·F) convexity scans and rebuilds
  the occlusion grid / split components per stroke, not per dab.
- Convex-mesh fast occlusion path replaces per-texel grid raycasts with a
  dot product.
- Dirty-rect region uploads (albedo upload per-dab; material/height coalesced)
  and incremental 2D preview patches avoid full-atlas recomposites per dab.
- Reduced-resolution 3D rendering while interacting.
- `uv_barycentric` clamps texels to triangle interiors before any 3D
  reconstruction, so the per-triangle loop stays at ~700 iterations rather
  than ~400K per triangle.