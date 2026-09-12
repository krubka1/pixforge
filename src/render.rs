use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use crate::io::{MeshData, TextureData};

/// Brush-cursor mask drawn over the surface by `Renderer`. The mesh is retraced
/// with a dedicated fragment shader that discards everything outside the brush
/// footprint, so the cursor conforms to the model exactly instead of being
/// approximated as a projected 2D shape.
#[derive(Clone, Copy, Debug)]
pub struct BrushOverlay {
    /// Brush center in world space (the surface hit point).
    pub center: Vec3,
    /// Brush-local U/V axes (unit, on the surface tangent plane, matching what
    /// a stamp paints with — see `paint::brush_axes`).
    pub axis_u: Vec3,
    pub axis_v: Vec3,
    /// Footprint radius in world units for round/square, square corner radius
    /// for diamond (same scale the stamp uses).
    pub radius: f32,
    /// 0 = round, 1 = square, 2 = diamond, 3 = texture (mirrors the paint
    /// footprints; the texture mask uses the sprite's alpha as coverage).
    pub shape: u32,
    /// RGBA tint of the mask (gamma-space, like the user-picked brush color).
    pub color: [f32; 4],
    /// Texture-brush stamp rotation in radians (used by `shape == 3`).
    pub rotation: f32,
    /// Texture-brush stamp flips (used by `shape == 3`).
    pub flip_x: bool,
    pub flip_y: bool,
}

/// Physically-based material parameters for the metallic-roughness shading in
/// the viewport. The surface properties (roughness/metallic/emissive/ao) are
/// per-layer and carried to the GPU as the material atlas; the remaining
/// fields here are viewport-wide lighting that the CPU writes each frame.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Material {
    /// 0..=1 surface shininess (lower = glossier).
    pub roughness: f32,
    /// 0..=1 dielectric-to-metal blend.
    pub metallic: f32,
    /// Emission multiplier on the albedo (added after tone mapping, so it can
    /// bloom past white).
    pub emissive: f32,
    /// 0..=1 ambient occlusion applied to the diffuse sky light (crevice
    /// shading). Deliberately does not touch the specular environment
    /// reflection, so metals keep their reflective look while AO only darkens
    /// recessed non-metallic surfaces.
    pub ambient_occlusion: f32,
    /// Key-light (sun) intensity.
    pub sun_intensity: f32,
    /// Key-light (sun) color.
    pub sun_color: [f32; 3],
    /// Master switch for the directional sun. Turn it off so the
    /// environment/skybox alone lights the scene (direct term drops to zero).
    pub sun_enabled: bool,
    /// Sun elevation above the horizon in degrees (-90 = straight down,
    /// +90 = overhead).
    pub sun_elevation: f32,
    /// Sun azimuth around the vertical axis in degrees (0 = toward +Z).
    pub sun_azimuth: f32,
    /// Rotation of the environment/skybox around the vertical axis in degrees.
    pub env_rotation: f32,
    /// Uniform color of the analytic sky (no environment map loaded): ambient
    /// light and viewport backdrop. Ambient-only; carries no direction, so
    /// with the sun off the surface shading from it is flat.
    pub sky_color: [f32; 3],
    /// How strongly the analytic sky lights the surface.
    pub env_intensity: f32,
    /// Exposure multiplier applied before tone mapping.
    pub exposure: f32,
    /// Camera-direction fill light strength (keeps shadow interiors readable).
/// Directional styling — zeroed while the sun is off, leaving pure skybox light.
    pub fill_intensity: f32,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            roughness: 0.55,
            metallic: 0.0,
            emissive: 0.0,
            ambient_occlusion: 1.0,
            sun_intensity: 2.6,
            sun_color: [1.0, 0.97, 0.90],
            sun_enabled: true,
            // Matches the classic fixed key-light direction (0.5, 0.7, 0.8).
            sun_elevation: 36.6,
            sun_azimuth: 32.0,
            env_rotation: 0.0,
            sky_color: [0.45, 0.48, 0.56],
            env_intensity: 1.0,
            exposure: 1.0,
            fill_intensity: 0.5,
        }
    }
}

/// World-space direction *toward* the sun, from elevation/azimuth in degrees
/// (azimuth 0 = +Z, positive turns toward +X).
fn sun_direction(elevation_deg: f32, azimuth_deg: f32) -> [f32; 3] {
    let el = elevation_deg.to_radians();
    let az = azimuth_deg.to_radians();
    [az.sin() * el.cos(), el.sin(), az.cos() * el.cos()]
}

pub struct Camera {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov_y: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
    pub radius: f32,
}

impl Camera {
    pub fn new(aspect: f32) -> Self {
        Self {
            eye: Vec3::new(3.0, 2.0, 3.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            fov_y: 45.0_f32.to_radians(),
            aspect,
            near: 0.01,
            far: 100.0,
            radius: 5.0,
        }
    }

    pub fn fit(&mut self, center: Vec3, bounds_radius: f32) {
        self.target = center;
        self.radius = bounds_radius.max(0.1) * 2.5;
        let dir = (self.eye - self.target).normalize_or_zero();
        self.eye = self.target + dir * self.radius;
    }

    pub fn orbit(&mut self, delta_azimuth: f32, delta_polar: f32) {
        let dir = (self.eye - self.target).normalize_or_zero();
        let azimuth = dir.z.atan2(dir.x) + delta_azimuth;
        // `dir` is normalized, so dir.y is already in [-1, 1]; using it directly
        // lets the view arc all the way over the top (a previous `dir.y / radius`
        // here clamped vertical travel to a narrow band around the equator).
        let polar = (dir.y.clamp(-1.0, 1.0)).acos() + delta_polar;
        let polar = polar.clamp(0.02, std::f32::consts::PI - 0.02);
        self.eye = self.target
            + Vec3::new(
                self.radius * polar.sin() * azimuth.cos(),
                self.radius * polar.cos(),
                self.radius * polar.sin() * azimuth.sin(),
            );
    }

    pub fn zoom(&mut self, factor: f32) {
        self.radius = (self.radius * factor).clamp(0.1, 50.0);
        let dir = (self.eye - self.target).normalize_or_zero();
        self.eye = self.target + dir * self.radius;
    }

    /// Pans the camera in screen space: positive dx drags content right,
    /// positive dy drags content down.
    pub fn pan(&mut self, dx_px: f32, dy_px: f32, viewport_height_px: f32) {
        let fwd = (self.target - self.eye).normalize_or_zero();
        let right = fwd.cross(Vec3::Y).normalize_or_zero();
        let up = right.cross(fwd).normalize_or_zero();
        let scale = 2.0 * self.radius * (self.fov_y * 0.5).tan() / viewport_height_px.max(1.0);
        let offset = (right * -dx_px + up * dy_px) * scale;
        self.eye += offset;
        self.target += offset;
    }

    /// World-space ray through a normalized device coordinate (x right, y up,
    /// both in [-1, 1], from the viewport center).
    pub fn ray(&self, ndc_x: f32, ndc_y: f32) -> (Vec3, Vec3) {
        let inv = self.view_proj().inverse();
        let near = inv.project_point3(Vec3::new(ndc_x, ndc_y, 0.0));
        let far = inv.project_point3(Vec3::new(ndc_x, ndc_y, 1.0));
        (self.eye, (far - near).normalize_or_zero())
    }

    pub fn view_proj(&self) -> Mat4 {
        let proj = glam::camera::rh::proj::directx::perspective(
            self.fov_y,
            self.aspect,
            self.near,
            self.far,
        );
        let view = glam::camera::rh::view::look_at_mat4(self.eye, self.target, self.up);
        proj * view
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
}

fn mesh_vertices(mesh: &MeshData) -> (Vec<Vertex>, Vec<u32>) {
    let vertices: Vec<Vertex> = (0..mesh.positions.len())
        .map(|i| Vertex {
            position: mesh.positions[i].to_array(),
            normal: mesh.normals[i].to_array(),
            uv: [mesh.uvs[i].0, mesh.uvs[i].1],
        })
        .collect();
    (vertices, mesh.indices.clone())
}

/// The mesh is rendered with real transparency in two passes: fully-erased
/// texels (alpha 0) are discarded and the pixel keeps whatever is behind them;
/// fully-opaque texels are drawn first writing depth; texels with
/// 0 < alpha < 1 are then blended source-over (no depth write) on top of that,
/// so translucent paint reveals the lit surface behind it, not just the
/// backdrop.
pub struct Renderer {
    pipeline: wgpu::RenderPipeline,
    translucent_pipeline: wgpu::RenderPipeline,
    /// Draws the analytic sky / loaded environment fullscreen behind the mesh.
    background_pipeline: wgpu::RenderPipeline,
    /// Retraces the mesh with a fragment shader that keeps only the fragments
    /// inside the brush footprint and tints them, so the brush cursor reads as
    /// a mask lying on the model's surface (conforming to its curvature).
    /// Two pipelines share `overlay_fs`: `overlay_pipeline` draws the dark scrim
    /// (source-over), `overlay_glow_pipeline` adds the emissive colour glow
    /// (additive blend), giving the cursor a dark+emissive look readable on
    /// any material in any lighting.
    overlay_pipeline: wgpu::RenderPipeline,
    overlay_glow_pipeline: wgpu::RenderPipeline,
    /// Tiny (32-byte) copy-source buffer holding the two overlay pass
    /// selectors, copied into the uniform buffer before each overlay draw so
    /// each pass lands on the right mode ({1 = scrim, 2 = glow}) in GPU order.
    overlay_mode_upload: wgpu::Buffer,
    /// The active brush-cursor mask, written into the overlay uniforms each
    /// frame; `None` skips the extra draw entirely.
    pub brush_overlay: Option<BrushOverlay>,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    default_bind_group: wgpu::BindGroup,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    texture: Option<wgpu::Texture>,
    texture_view: Option<wgpu::TextureView>,
    /// Per-layer material map texture (RGBA = roughness/metallic/emissive/ao),
    /// kept separate from the albedo atlas. Its bind group binding falls back
    /// to the white view when no map is present.
    material_texture: Option<wgpu::Texture>,
    material_view: Option<wgpu::TextureView>,
    /// Per-layer height/bump atlas (R = signed height, encoded centered so 128
    /// is flat; G = per-texel bump strength /8). Sampled with the material
    /// sampler; falls back to the black view (flat) when absent.
    height_texture: Option<wgpu::Texture>,
    height_view: Option<wgpu::TextureView>,
    /// Resolution (longest side, texels) of the height map, sent to the
    /// shader in `env.w` so the bump gradient step is floored to one top-level
    /// texel regardless of atlas size or zoom.
    height_map_size: f32,
    /// Persistent 1x1 white texture whose view doubles as the albedo and
    /// material fallback.
    white_view: wgpu::TextureView,
    /// Persistent 1x1 black texture used as the height-map fallback (flat).
    black_view: wgpu::TextureView,
    /// Equirectangular environment map (mipmapped rgba16f) driving IBL when
    /// loaded; the shader falls back to the analytic sky when it is absent.
    env_texture: Option<wgpu::Texture>,
    env_view: Option<wgpu::TextureView>,
    env_sampler: wgpu::Sampler,
    /// Mip count minus one (the max texture LOD), sent to the shader in
    /// `camera_pos.w`; 0 means no environment is bound (analytic sky path).
    env_lods: f32,
    /// Texture-brush sprite uploaded for the `shape == 3` overlay mask
    /// (sampled nearest so the mask picks exactly the texels the stamp does).
    brush_sprite_view: Option<wgpu::TextureView>,
    brush_sprite_sampler: wgpu::Sampler,
    /// Signature of the sprite currently in `brush_sprite_view`.
    brush_sprite_sig: u64,
    /// Sprite size in texels, sent to the shader for texel addressing.
    brush_sprite_dims: [u32; 2],
    device: wgpu::Device,
    queue: wgpu::Queue,
    uv_overlay: u32,
    material: Material,
}

/// Uniform buffer contents: the view-projection matrix (64 bytes), the 32-bit
/// pass mode (opaque = 0, translucent = 1), the 32-bit UV debug overlay
/// (bit 0 = checkerboard, bit 1 = UV grid), then the PBR uniform vec4s
/// (material, sun, sun color, environment, camera position).
const UNIFORM_BYTES: u64 = 352;
const UNIFORM_FLOATS: usize = 88;
const PASS_MODE_OFFSET: u64 = 128;
const PASS_OPAQUE: u32 = 0;
const PASS_TRANSLUCENT: u32 = 1;
const UV_OVERLAY_OFFSET: u64 = 132;
const MATERIAL_OFFSET: u64 = 144;
const SUN_OFFSET: u64 = 160;
const SUN_COLOR_OFFSET: u64 = 176;
const ENV_OFFSET: u64 = 192;
const CAMERA_OFFSET: u64 = 208;
const ENV_ROT_OFFSET: u64 = 224;
const SKY_COLOR_OFFSET: u64 = 240;
// Brush-cursor mask block (see the overlay_* fields in shader.wgsl).
const OVERLAY_CENTER_OFFSET: u64 = 256;
const OVERLAY_U_OFFSET: u64 = 272;
const OVERLAY_V_OFFSET: u64 = 288;
const OVERLAY_COLOR_OFFSET: u64 = 304;
const OVERLAY_SPRITE_OFFSET: u64 = 320;
const OVERLAY_PARAMS_OFFSET: u64 = 336;

/// UV debug overlay flags for the 3D viewport.
pub const UV_OVERLAY_CHECKER: u32 = 1;
pub const UV_OVERLAY_GRID: u32 = 2;

impl Renderer {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        const VIEWPORT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mesh_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh_pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let vert_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![
                0 => Float32x3,
                1 => Float32x3,
                2 => Float32x2,
            ],
        };

        let make_pipeline = |write_depth: bool| {
            // The opaque pass writes depth for alpha-1 texels. Adjacent
            // triangles on a curved surface are never coplanar, so along a
            // shared edge one polygon can sit a hair "behind" the plane of its
            // neighbor and lose the depth test there, leaving pixel cracks
            // that reveal whatever is behind (the backdrop or the far wall).
            // A small constant depth bias (toward the camera) on this pass
            // closes those seams. The slope term is left at 0: on grazing
            // silhouette polygons the depth slope is huge, so a slope-scaled
            // bias can shove those fragments far past their neighbors and
            // make curved silhouettes visibly jitter.
            let bias = if write_depth {
                wgpu::DepthBiasState {
                    constant: -1,
                    slope_scale: 0.0,
                    clamp: 0.0,
                }
            } else {
                wgpu::DepthBiasState::default()
            };
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(if write_depth {
                    "mesh_pipeline"
                } else {
                    "mesh_pipeline_translucent"
                }),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[Some(vert_layout.clone())],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: VIEWPORT_FORMAT,
                        blend: Some(wgpu::BlendState {
                            // Source-over: semi-transparent texels (0 < a < 1) blend
                            // toward whatever is behind them (the clear backdrop),
                            // so translucent paint shows the background through it
                            // instead of vanishing or turning black.
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::SrcAlpha,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                            // Keep the stored alpha fully opaque: the viewport is
                            // displayed as a plain texture by egui, and the
                            // semi-transparency has already been resolved onto the
                            // backdrop inside this pass.
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    // No backface culling: looking through an erased/transparent
                    // hole shows the object's far interior, not empty space. The
                    // shader lights back-facing fragments with geometric normals
                    // plus a strong fill so the interior reads clearly from any
                    // angle (a view-dependent normal flip would darken it).
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    // The translucent pass does NOT write depth: translucent
                    // texels are depth-tested but never record their own depth,
                    // so they blend over the opaque surface behind them instead
                    // of occluding it.
                    depth_write_enabled: Some(write_depth),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: Default::default(),
                    bias,
                }),
                multisample: wgpu::MultisampleState::default(),
                cache: None,
                multiview_mask: None,
            })
        };
        let pipeline = make_pipeline(true);
        let translucent_pipeline = make_pipeline(false);

        // Brush-cursor mask: retraces the mesh, discarding fragments outside
        // the brush footprint (shader.wgsl `overlay_fs`). Depth-test LE against
        // the opaque pass with the same constant bias as it, so the mask lands
        // exactly on the visible surface — the half of the footprint hidden
        // behind the near wall fails the test and never shows through, while
        // the depths coincide when the same geometry is pixel-aligned. Writing
        // depth off lets it blend source-over like any translucent overlay.
        let overlay_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("overlay_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(vert_layout.clone())],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("overlay_fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: VIEWPORT_FORMAT,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: Default::default(),
                // Same constant bias as the opaque pass so coincident geometry
                // passes the LE test instead of landing a hair behind it.
                bias: wgpu::DepthBiasState {
                    constant: -1,
                    slope_scale: 0.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            cache: None,
            multiview_mask: None,
        });

        // Emissive glow pass: same mesh + overlay_fs, but additive blend so
        // the brush colour brightens the surface directly — visible even in
        // dark scenes where the dark scrim alone would be invisible.
        let overlay_glow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("overlay_glow_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(vert_layout)],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("overlay_fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: VIEWPORT_FORMAT,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: Default::default(),
                bias: wgpu::DepthBiasState {
                    constant: -1,
                    slope_scale: 0.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            cache: None,
            multiview_mask: None,
        });

        // The background: a fullscreen triangle (no vertex buffers) sampling the
        // analytic sky / loaded environment, drawn first so it sits behind the
        // mesh (and shows through erased/transparent texels). It never writes or
        // tests depth — the mesh passes own the depth buffer.
        let background_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("background_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("bg_vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("bg_fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: VIEWPORT_FORMAT,
                    // Fully replaces the cleared backdrop (opaque sky).
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            cache: None,
            multiview_mask: None,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniform_buffer"),
            size: UNIFORM_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // One 16-byte mode selector per overlay draw ({1 = scrim, 2 = glow}), copied
        // into the uniform buffer between draws so the pass mode is committed in
        // GPU order instead of being clobbered by the last queue write before
        // submission.
        let overlay_mode_upload = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("overlay_mode_upload"),
            size: 2 * 16,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let modes: [f32; 8] = [
            1.0, 0.0, 0.0, 0.0, //
            2.0, 0.0, 0.0, 0.0, //
        ];
        queue.write_buffer(&overlay_mode_upload, 0, bytemuck::cast_slice(&modes));

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mesh_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // 1x1 white fallback: models without a texture render as-is (white albedo).
        let white = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("white_tex"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &white,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[255, 255, 255, 255],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        let white_view = white.create_view(&Default::default());

        // 1x1 black fallback: the height map reads flat 0 when absent.
        let black = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("black_tex"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &black,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[0, 0, 0, 255],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let black_view = black.create_view(&Default::default());

        // Mipmapped wrap sampler for the equirectangular environment: `Repeat` lets
        // the direction-to-uv mapping's seam interpolate across 0/1 instead of
        // popping to the clamped edge, and mip-lod sampling fades roughness.
        let env_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("env_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        // Nearest sampler for the brush sprite: the mask must hit exactly the
        // texels the stamp targets (which indexes with `floor(u * width)`), so
        // bilinear smoothing is disabled here.
        let brush_sprite_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("brush_sprite_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let make_bind_group =
            |uniform_buffer: &wgpu::Buffer,
             base_view: &wgpu::TextureView,
             material_view: &wgpu::TextureView,
             height_view: &wgpu::TextureView,
             env_view: &wgpu::TextureView,
             brush_view: &wgpu::TextureView| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("mesh_bind_group"),
                    layout: &bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: uniform_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(base_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(material_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::Sampler(&sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::TextureView(height_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: wgpu::BindingResource::TextureView(env_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: wgpu::BindingResource::Sampler(&env_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 8,
                            resource: wgpu::BindingResource::TextureView(brush_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 9,
                            resource: wgpu::BindingResource::Sampler(&brush_sprite_sampler),
                        },
                    ],
                })
            };

        let default_bind_group =
            make_bind_group(&uniform_buffer, &white_view, &white_view, &black_view, &black_view, &white_view);

        let (vertex_buffer, index_buffer, index_count) = empty_buffers(&device);

        Self {
            pipeline,
            translucent_pipeline,
            background_pipeline,
            overlay_pipeline,
            overlay_glow_pipeline,
            overlay_mode_upload,
            brush_overlay: None,
            vertex_buffer,
            index_buffer,
            index_count,
            uniform_buffer,
            bind_group: default_bind_group.clone(),
            default_bind_group,
            bind_group_layout: bgl,
            sampler,
            texture: None,
            texture_view: None,
            material_texture: None,
            material_view: None,
            height_texture: None,
            height_view: None,
            height_map_size: 0.0,
            white_view,
            black_view,
            env_texture: None,
            env_view: None,
            env_sampler,
            env_lods: 0.0,
            brush_sprite_view: None,
            brush_sprite_sampler,
            brush_sprite_sig: 0,
            brush_sprite_dims: [1, 1],
            device,
            queue,
            uv_overlay: 0,
            material: Material::default(),
        }
    }

    pub fn set_uv_overlay(&mut self, mode: u32) {
        self.uv_overlay = mode;
    }

    pub fn set_material(&mut self, material: Material) {
        self.material = material;
    }

    /// Uploads (or replaces, when the sprite changed) the brush sprite used by
    /// the texture-shape overlay mask. The upload is skipped when `sig` matches
    /// the sprite already on the GPU, so calling it every frame is cheap.
    pub fn set_brush_sprite(&mut self, sig: u64, sprite: &crate::io::TextureData) {
        if self.brush_sprite_sig == sig {
            return;
        }
        let (padded, stride) = crate::render::padding::rgba_with_padded_rows(
            &sprite.rgba,
            sprite.width,
            sprite.height,
        );
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("brush_sprite_texture"),
            size: wgpu::Extent3d {
                width: sprite.width,
                height: sprite.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &padded,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(stride),
                rows_per_image: Some(sprite.height),
            },
            wgpu::Extent3d {
                width: sprite.width,
                height: sprite.height,
                depth_or_array_layers: 1,
            },
        );
        self.brush_sprite_view = Some(texture.create_view(&Default::default()));
        self.brush_sprite_sig = sig;
        self.brush_sprite_dims = [sprite.width, sprite.height];
        self.rebind();
    }

    /// Rebuilds the current bind group from the base-albedo and material-map
    /// views (falling back to the white texture when either is absent).
    fn rebind(&mut self) {
        if self.texture_view.is_none() {
            self.bind_group = self.default_bind_group.clone();
            return;
        }
        let base = self.texture_view.as_ref().unwrap();
        let material = self.material_view.as_ref().unwrap_or(&self.white_view);
        let height = self.height_view.as_ref().unwrap_or(&self.black_view);
        let env = self.env_view.as_ref().unwrap_or(&self.black_view);
        let brush = self.brush_sprite_view.as_ref().unwrap_or(&self.white_view);
        let binding = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh_bind_group_t"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(base),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(material),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(height),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(env),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Sampler(&self.env_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(brush),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::Sampler(&self.brush_sprite_sampler),
                },
            ],
        });
        self.bind_group = binding;
    }

    pub fn set_mesh(&mut self, mesh: &MeshData) {
        let (vertices, indices) = mesh_vertices(mesh);
        self.vertex_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh_vertices"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
        self.index_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh_indices"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            });
        self.index_count = indices.len() as u32;

        match mesh.flattened_atlas() {
            Some(tex) => self.update_texture(&tex),
            None => {
                self.texture = None;
                self.texture_view = None;
            }
        }
        let material_map = mesh.flattened_material_atlas();
        self.update_material_map(material_map.as_ref());
        let height_map = mesh.flattened_height_atlas();
        self.update_height_map(height_map.as_ref());
    }

    /// Re-uploads (or recreates) the per-layer material map texture. `None`
    /// clears it, falling back to the default material the shader reads from
    /// the white texture.
    pub fn update_material_map(&mut self, tex: Option<&TextureData>) {
        let Some(tex) = tex else {
            self.material_texture = None;
            self.material_view = None;
            self.rebind();
            return;
        };
        let (padded, stride) = padding::rgba_with_padded_rows(&tex.rgba, tex.width, tex.height);
        let recreate = match &self.material_texture {
            Some(existing) => existing.width() != tex.width || existing.height() != tex.height,
            None => true,
        };
        if recreate {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("material_texture"),
                size: wgpu::Extent3d {
                    width: tex.width,
                    height: tex.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.material_view = Some(texture.create_view(&Default::default()));
            self.material_texture = Some(texture);
        }
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: self.material_texture.as_ref().unwrap(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &padded,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(stride),
                rows_per_image: Some(tex.height),
            },
            wgpu::Extent3d {
                width: tex.width,
                height: tex.height,
                depth_or_array_layers: 1,
            },
        );
        self.rebind();
    }

    /// Re-uploads (or recreates) the per-layer height/bump map texture. `None`
    /// clears it, falling back to the black view (flat surface) in the shader.
    pub fn update_height_map(&mut self, tex: Option<&TextureData>) {
        let Some(tex) = tex else {
            self.height_texture = None;
            self.height_view = None;
            self.height_map_size = 0.0;
            self.rebind();
            return;
        };
        self.height_map_size = tex.width.max(tex.height) as f32;
        let (padded, stride) = padding::rgba_with_padded_rows(&tex.rgba, tex.width, tex.height);
        let recreate = match &self.height_texture {
            Some(existing) => existing.width() != tex.width || existing.height() != tex.height,
            None => true,
        };
        if recreate {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("height_texture"),
                size: wgpu::Extent3d {
                    width: tex.width,
                    height: tex.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.height_view = Some(texture.create_view(&Default::default()));
            self.height_texture = Some(texture);
        }
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: self.height_texture.as_ref().unwrap(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &padded,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(stride),
                rows_per_image: Some(tex.height),
            },
            wgpu::Extent3d {
                width: tex.width,
                height: tex.height,
                depth_or_array_layers: 1,
            },
        );
        self.rebind();
    }

    /// Uploads (or clears) the equirectangular environment map (HDRI or plain
    /// image) with its full CPU-prebuilt mip chain. `None` drops it; the shader
    /// falls back to the analytic sky and `camera_pos.w` reports 0 LODs.
    pub fn set_environment(&mut self, env: Option<crate::io::EnvironmentMips>) {
        self.env_texture = None;
        self.env_view = None;
        let Some(env) = env else {
            self.env_lods = 0.0;
            self.rebind();
            return;
        };
        let mip_count = env.mips.len().max(1) as u32;
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("env_texture"),
            size: wgpu::Extent3d {
                width: env.width,
                height: env.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // rgba16f is filterable out of the box (unlike rgba32f, which needs
            // the float32-filterable feature for linear minification).
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        for (level, bytes) in env.mips.iter().enumerate() {
            let (mw, mh) = ((env.width >> level).max(1), (env.height >> level).max(1));
            let row = mw as usize * 8;
            let stride = row.div_ceil(256) * 256;
            let mut padded = Vec::with_capacity(stride * mh as usize);
            for rows in bytes.chunks_exact(row) {
                padded.extend_from_slice(rows);
                padded.resize(padded.len() + (stride - row), 0);
            }
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: level as u32,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &padded,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(stride as u32),
                    rows_per_image: Some(mh),
                },
                wgpu::Extent3d {
                    width: mw,
                    height: mh,
                    depth_or_array_layers: 1,
                },
            );
        }
        self.env_view = Some(texture.create_view(&Default::default()));
        self.env_texture = Some(texture);
        self.env_lods = (mip_count - 1) as f32;
        self.rebind();
    }

    /// Uploads (or recreates, if the dimensions changed) the mesh atlas texture
    /// from CPU `TextureData.rgba` rows, keeping the existing GPU resource
    /// alive for in-place updates between frames.
    pub fn update_texture(&mut self, tex: &TextureData) {
        let (padded, stride) = padding::rgba_with_padded_rows(&tex.rgba, tex.width, tex.height);

        let recreate = match &self.texture {
            Some(existing) => existing.width() != tex.width || existing.height() != tex.height,
            None => true,
        };

        if recreate {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("mesh_texture"),
                size: wgpu::Extent3d {
                    width: tex.width,
                    height: tex.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.texture_view = Some(texture.create_view(&Default::default()));
            self.texture = Some(texture);
        }

        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: self.texture.as_ref().unwrap(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &padded,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(stride),
                rows_per_image: Some(tex.height),
            },
            wgpu::Extent3d {
                width: tex.width,
                height: tex.height,
                depth_or_array_layers: 1,
            },
        );
        self.rebind();
    }

    /// Uploads only a sub-rect of the mesh atlas, leaving the rest untouched.
    ///
    /// `region` holds the composited texels for `x..x+region.width`,
    /// `y..y+region.height` of a `full_width`x`full_height` atlas. The GPU
    /// texture must already exist at the full size (a prior `update_texture`
    /// or `set_mesh`); if it does not the call is a no-op and returns `false`,
    /// and the caller should fall back to a full upload.
    pub fn update_texture_region(
        &mut self,
        region: &TextureData,
        x: u32,
        y: u32,
        full_width: u32,
        full_height: u32,
    ) -> bool {
        let Some(texture) = self.texture.as_ref() else {
            return false;
        };
        if texture.width() != full_width
            || texture.height() != full_height
            || region.width == 0
            || region.height == 0
            || x > full_width
            || y > full_height
            || region.width > full_width - x
            || region.height > full_height - y
        {
            return false;
        }
        let (padded, stride) =
            padding::rgba_with_padded_rows(&region.rgba, region.width, region.height);
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            &padded,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(stride),
                rows_per_image: Some(region.height),
            },
            wgpu::Extent3d {
                width: region.width,
                height: region.height,
                depth_or_array_layers: 1,
            },
        );
        true
    }

    /// Renders the mesh into the given color/depth texture views in two passes.
    ///
    /// Pass 1 (`opaque`) draws only fully-opaque texels (alpha ~ 1) and writes
    /// depth: a translucent texel is skipped there and so cannot occlude the
    /// opaque surface behind it (e.g. the far interior wall of an erased hole).
    /// Pass 2 (`translucent`) then depth-tests but does not write depth, and
    /// source-over blends the 0 < alpha < 1 texels over whatever pass 1 put at
    /// that pixel — the lit far wall, or the backdrop when nothing is behind.
    pub fn render(
        &self,
        camera: &Camera,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
    ) {
        let vp = camera.view_proj().to_cols_array_2d();
        let vp_inv = camera.view_proj().inverse().to_cols_array_2d();
        let mut data = [0.0f32; UNIFORM_FLOATS];
        for (dst, row) in data.iter_mut().zip(vp.iter().flat_map(|r| r.iter())) {
            *dst = *row;
        }
        for (dst, v) in data.iter_mut().skip(16).zip(vp_inv.iter().flat_map(|r| r.iter())) {
            *dst = *v;
        }

        // Pass-mode defaults to opaque; written explicitly with each submit
        // below so the opaque (0) / translucent (1) flip is ordered correctly.
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&data));
        self.queue.write_buffer(
            &self.uniform_buffer,
            PASS_MODE_OFFSET,
            &PASS_OPAQUE.to_le_bytes(),
        );
        self.queue.write_buffer(
            &self.uniform_buffer,
            UV_OVERLAY_OFFSET,
            &self.uv_overlay.to_le_bytes(),
        );
        let m = &self.material;
        let material_vec: [f32; 4] = [
            m.roughness,
            m.metallic,
            m.emissive,
            m.ambient_occlusion,
        ];
        let sun_dir = sun_direction(m.sun_elevation, m.sun_azimuth);
        let sun_vec: [f32; 4] = [
            sun_dir[0],
            sun_dir[1],
            sun_dir[2],
            if m.sun_enabled { m.sun_intensity } else { 0.0 },
        ];
        let sun_color_vec: [f32; 4] = [m.sun_color[0], m.sun_color[1], m.sun_color[2], 0.0];
        let env_vec: [f32; 4] = [
            m.env_intensity,
            m.exposure,
            // The fill light is directional styling (it keys off the sun
            // direction, so it would cast an "anti-sun" shadow with the sun
            // off). Turning the sun off means *pure* skybox lighting, so the
            // fill is zeroed too.
            if m.sun_enabled { m.fill_intensity } else { 0.0 },
            self.height_map_size,
        ];
        let camera_vec: [f32; 4] = [camera.eye.x, camera.eye.y, camera.eye.z, self.env_lods];
        self.queue.write_buffer(
            &self.uniform_buffer,
            MATERIAL_OFFSET,
            bytemuck::cast_slice(&material_vec),
        );
        self.queue.write_buffer(
            &self.uniform_buffer,
            SUN_OFFSET,
            bytemuck::cast_slice(&sun_vec),
        );
        self.queue.write_buffer(
            &self.uniform_buffer,
            SUN_COLOR_OFFSET,
            bytemuck::cast_slice(&sun_color_vec),
        );
        self.queue.write_buffer(
            &self.uniform_buffer,
            ENV_OFFSET,
            bytemuck::cast_slice(&env_vec),
        );
        self.queue.write_buffer(
            &self.uniform_buffer,
            CAMERA_OFFSET,
            bytemuck::cast_slice(&camera_vec),
        );
        let env_rot_vec: [f32; 4] = [m.env_rotation.to_radians(), 0.0, 0.0, 0.0];
        self.queue.write_buffer(
            &self.uniform_buffer,
            ENV_ROT_OFFSET,
            bytemuck::cast_slice(&env_rot_vec),
        );
        let sky_color_vec: [f32; 4] = [m.sky_color[0], m.sky_color[1], m.sky_color[2], 0.0];
        self.queue.write_buffer(
            &self.uniform_buffer,
            SKY_COLOR_OFFSET,
            bytemuck::cast_slice(&sky_color_vec),
        );

        // Brush-cursor mask uniforms. The shader discards when the enable flag
        // is clear, so whatever the app left is harmless when no mask is set.
        if let Some(bo) = &self.brush_overlay {
            let center: [f32; 4] = [bo.center.x, bo.center.y, bo.center.z, 1.0];
            let u: [f32; 4] = [bo.axis_u.x, bo.axis_u.y, bo.axis_u.z, bo.radius];
            let v: [f32; 4] = [bo.axis_v.x, bo.axis_v.y, bo.axis_v.z, bo.shape as f32];
            self.queue.write_buffer(
                &self.uniform_buffer,
                OVERLAY_CENTER_OFFSET,
                bytemuck::cast_slice(&center),
            );
            self.queue.write_buffer(
                &self.uniform_buffer,
                OVERLAY_U_OFFSET,
                bytemuck::cast_slice(&u),
            );
            self.queue.write_buffer(
                &self.uniform_buffer,
                OVERLAY_V_OFFSET,
                bytemuck::cast_slice(&v),
            );
            self.queue.write_buffer(
                &self.uniform_buffer,
                OVERLAY_COLOR_OFFSET,
                bytemuck::cast_slice(&bo.color),
            );
            // Texture-shape mask: sprite size (texels), rotation (radians) and
            // flips packed into w. Ignored by the round/square/diamond shapes.
            let flip = bo.flip_x as u32 | (bo.flip_y as u32) << 1;
            let sprite_vec: [f32; 4] = [
                self.brush_sprite_dims[0] as f32,
                self.brush_sprite_dims[1] as f32,
                bo.rotation,
                flip as f32,
            ];
            self.queue.write_buffer(
                &self.uniform_buffer,
                OVERLAY_SPRITE_OFFSET,
                bytemuck::cast_slice(&sprite_vec),
            );
            // Reset the pass-mode to the default (fallback) so the uniform is
            // deterministic between the two overlay draws below (written again
            // right before each).
            let idle: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
            self.queue.write_buffer(
                &self.uniform_buffer,
                OVERLAY_PARAMS_OFFSET,
                bytemuck::cast_slice(&idle),
            );
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("scene_encoder"),
            });

        // Pass 1 — opaque texels only, depth written so pass 2 can be tested
        // against them.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene_pass_opaque"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.5,
                            g: 0.52,
                            b: 0.55,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            pass.set_pipeline(&self.background_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..3, 0..1);

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            if self.index_count > 0 {
                pass.draw_indexed(0..self.index_count, 0, 0..1);
            }
        }
        self.queue.submit(Some(encoder.finish()));

        // Pass 2 — translucent texels (0 < alpha < 1), no depth writes, blending
        // source-over against the opaque pass output. The pass-mode uniform is
        // flipped and the second encoder submitted after the first, so the GPU
        // executes them in this order.
        self.queue.write_buffer(
            &self.uniform_buffer,
            PASS_MODE_OFFSET,
            &PASS_TRANSLUCENT.to_le_bytes(),
        );
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("scene_encoder_translucent"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene_pass_translucent"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            pass.set_pipeline(&self.translucent_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            if self.index_count > 0 {
                pass.draw_indexed(0..self.index_count, 0, 0..1);
            }
        }

        // Brush-cursor mask: retrace the mesh and let the overlay shader keep
        // only the fragments inside the footprint, so the cursor sits on the
        // surface (depth-tested against the opaque pass). Two ordered passes
        // give the dark+emissive look: a source-over dark scrim (mode 1)
        // darkens the footprint in daylight, and an additive colour glow
        // (mode 2) shines over it so the cursor reads in dark scenes. Each
        // pass first copies its own mode selector into the uniform buffer in
        // GPU order, so the draws genuinely pick their pass instead of every
        // one reading the last queue write.
        if self.brush_overlay.is_some() {
            for (slot, pipeline) in [
                (0, &self.overlay_pipeline),
                (1, &self.overlay_glow_pipeline),
            ] {
                encoder.copy_buffer_to_buffer(
                    &self.overlay_mode_upload,
                    slot * 16,
                    &self.uniform_buffer,
                    OVERLAY_PARAMS_OFFSET,
                    16,
                );
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("scene_pass_overlay"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: color_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                if self.index_count > 0 {
                    pass.draw_indexed(0..self.index_count, 0, 0..1);
                }
            }
        }

        self.queue.submit(Some(encoder.finish()));
    }
}

fn empty_buffers(device: &wgpu::Device) -> (wgpu::Buffer, wgpu::Buffer, u32) {
    let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("empty_vertex"),
        size: 4,
        usage: wgpu::BufferUsages::VERTEX,
        mapped_at_creation: false,
    });
    let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("empty_index"),
        size: 4,
        usage: wgpu::BufferUsages::INDEX,
        mapped_at_creation: false,
    });
    (vertex_buffer, index_buffer, 0)
}

pub struct ViewportTextures {
    // Kept alive for the lifetime of the views (the views hold their own refs too).
    #[allow(dead_code)]
    pub color: wgpu::Texture,
    pub color_view: wgpu::TextureView,
    #[allow(dead_code)]
    pub depth: wgpu::Texture,
    pub depth_view: wgpu::TextureView,
    pub size: (u32, u32),
}

impl ViewportTextures {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewport_color"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let color_view = color.create_view(&Default::default());

        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewport_depth"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth.create_view(&Default::default());

        Self {
            color,
            color_view,
            depth,
            depth_view,
            size: (width, height),
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) -> bool {
        // Quantize to even pixels so a ±1px layout jitter doesn't recreate the
        // texture (and re-register the egui native texture) every frame, which
        // otherwise shows up as viewport flicker.
        let (w, h) = (width.max(2) & !1, height.max(2) & !1);
        if self.size != (w, h) {
            *self = Self::new(device, w, h);
            true
        } else {
            false
        }
    }
}

pub(crate) mod padding {
    /// Pads each row of RGBA data to the wgpu 256-byte row alignment required by
    /// `Queue::write_texture`, returning (data, bytes_per_row).
    pub fn rgba_with_padded_rows(rgba: &[u8], width: u32, height: u32) -> (Vec<u8>, u32) {
        let row = width as usize * 4;
        let stride = row.div_ceil(256) * 256;
        let mut out = Vec::with_capacity(stride * height as usize);
        for rows in rgba.chunks_exact(row) {
            out.extend_from_slice(rows);
            out.resize(out.len() + (stride - row), 0);
        }
        (out, stride as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{load_gltf, Layer, TextureData};
    use glam::Vec3;

    fn device_and_queue() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .expect("no adapter available");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("pixforge_test"),
            ..Default::default()
        }))
        .expect("no device");
        (device, queue)
    }

    const SIZE: u32 = 512;

    fn render_and_read(device: &wgpu::Device, queue: &wgpu::Queue, mesh: &MeshData) -> Vec<u8> {
        render_and_read_material(device, queue, mesh, 0, Material::default())
    }

    fn render_and_read_with_overlay(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh: &MeshData,
        uv_overlay: u32,
    ) -> Vec<u8> {
        render_and_read_material(device, queue, mesh, uv_overlay, Material::default())
    }

    fn render_and_read_material(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh: &MeshData,
        uv_overlay: u32,
        material: Material,
    ) -> Vec<u8> {
        let color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test_color"),
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test_depth"),
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let mut renderer = Renderer::new(device.clone(), queue.clone());
        // Exercise the per-texel material path: surface params are read from
        // the layer material map, so fold the test `Material`'s surface values
        // into the first layer before uploading.
        let mut mesh = mesh.clone();
        if let Some(layer) = mesh.layers.first_mut() {
            layer.roughness = material.roughness;
            layer.metallic = material.metallic;
            layer.emissive = material.emissive;
            layer.ambient_occlusion = material.ambient_occlusion;
        }
        renderer.set_mesh(&mesh);
        renderer.set_uv_overlay(uv_overlay);
        renderer.set_material(material);

        let mut camera = Camera::new(1.0);
        if mesh.positions.len() == 3 {
            // Gimbal triangle around the origin.
            camera.fit(Vec3::ZERO, 1.0);
        } else {
            let min = mesh.positions.iter().fold(Vec3::MAX, |a, b| a.min(*b));
            let max = mesh.positions.iter().fold(Vec3::MIN, |a, b| a.max(*b));
            let center = (min + max) * 0.5;
            let radius = mesh
                .positions
                .iter()
                .map(|p| (p - center).length())
                .fold(0.0, f32::max);
            camera.fit(center, radius);
        }

        renderer.render(
            &camera,
            &color.create_view(&Default::default()),
            &depth.create_view(&Default::default()),
        );

        read_pixels(device, queue, &color)
    }

    fn read_pixels(device: &wgpu::Device, queue: &wgpu::Queue, color: &wgpu::Texture) -> Vec<u8> {
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_readback"),
            size: (SIZE * SIZE * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(SIZE * 4),
                    rows_per_image: Some(SIZE),
                },
            },
            wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let data = slice.get_mapped_range().expect("map failed");
        data.to_vec()
    }

    fn non_background_pixels(pixels: &[u8]) -> usize {
        // Background clear color is (0.50, 0.52, 0.55).
        const BG: [u8; 3] = [128, 133, 140];
        pixels
            .chunks_exact(4)
            .filter(|p| {
                [p[0], p[1], p[2]]
                    .iter()
                    .zip(BG.iter())
                    .any(|(a, b)| a.abs_diff(*b) > 16)
            })
            .count()
    }

    fn distinct_colors(pixels: &[u8]) -> usize {
        let mut seen = std::collections::HashSet::new();
        for p in pixels.chunks_exact(4) {
            seen.insert(u32::from_le_bytes([p[0], p[1], p[2], p[3]]));
        }
        seen.len()
    }

    fn manual_triangle() -> MeshData {
        MeshData {
            positions: vec![
                Vec3::new(-1.0, -1.0, 0.0),
                Vec3::new(1.0, -1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            normals: vec![Vec3::Z, Vec3::Z, Vec3::Z],
            uvs: vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
            indices: vec![0, 1, 2],
            layers: vec![],
            active_layer: 0,
            dirty: None,
        }
    }

    /// A fully white equirect environment (every texel ~1.0 after peak normalize)
    /// built in float16 with a 4-level mip chain, mirroring `load_environment`'s
    /// layout.
    fn white_environment() -> crate::io::EnvironmentMips {
        fn px() -> Vec<u8> {
            let b = crate::io::f32_to_f16(1.0).to_le_bytes();
            [b[0], b[1], b[0], b[1], b[0], b[1], b[0], b[1]].to_vec()
        }
        let mut mips = Vec::new();
        for dims in [(8u32, 4u32), (4, 2), (2, 1)] {
            let n = (dims.0 * dims.1) as usize;
            let mut lvl = Vec::with_capacity(n * 8);
            for _ in 0..n {
                lvl.extend_from_slice(&px());
            }
            mips.push(lvl);
        }
        mips.push(px());
        crate::io::EnvironmentMips {
            width: 8,
            height: 4,
            mips,
        }
    }

    #[test]
    fn environment_renders_as_background() {
        // The loaded environment (or analytic sky) must show BEHIND the model,
        // not only in its reflections: a fully transparent quad leaves the whole
        // frame as the backdrop, which flips between the analytic sky gradient
        // and a white environment map.
        let (device, queue) = device_and_queue();
        let mesh = textured_quad([0, 0, 0, 0]);
        let mut renderer = Renderer::new(device.clone(), queue.clone());
        renderer.set_mesh(&mesh);
        let mut camera = Camera::new(1.0);
        camera.fit(Vec3::ZERO, 1.0);
        let color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test_color"),
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test_depth"),
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let render = |renderer: &mut Renderer, color: &wgpu::Texture, depth: &wgpu::Texture| {
            renderer.render(
                &camera,
                &color.create_view(&Default::default()),
                &depth.create_view(&Default::default()),
            );
            read_pixels(&device, &queue, color)
        };
        let total = (SIZE * SIZE) as usize;

        let sky = render(&mut renderer, &color, &depth);
        assert!(
            sky_backdrop(&sky) > total / 2,
            "analytic sky should fill the transparent frame as the backdrop"
        );

        renderer.set_environment(Some(white_environment()));
        let white = render(&mut renderer, &color, &depth);
        let bright = white
            .chunks_exact(4)
            .filter(|p| p[0] > 230 && p[1] > 230 && p[2] > 230 && p[3] == 255)
            .count();
        assert!(
            bright > total / 2,
            "a white environment map should wash the backdrop white, got {bright}"
        );

        renderer.set_environment(None);
        let back_to_sky = render(&mut renderer, &color, &depth);
        assert!(
            sky_backdrop(&back_to_sky) > total / 2,
            "clearing the environment must restore the analytic sky backdrop"
        );
    }

    #[test]
    fn environment_changes_the_lighting_and_clears_back_to_sky() {
        let (device, queue) = device_and_queue();
        let mesh = manual_triangle().with_texture(crate::io::default_albedo());
        let mut renderer = Renderer::new(device.clone(), queue.clone());
        renderer.set_mesh(&mesh);

        let color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test_color"),
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test_depth"),
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let mut camera = Camera::new(1.0);
        camera.fit(Vec3::ZERO, 1.0);

        let render = |renderer: &mut Renderer,
                      camera: &Camera,
                      color: &wgpu::Texture,
                      depth: &wgpu::Texture| {
            renderer.render(
                camera,
                &color.create_view(&Default::default()),
                &depth.create_view(&Default::default()),
            );
            read_pixels(&device, &queue, color)
        };

        let sky_lit = render(&mut renderer, &camera, &color, &depth);

        renderer.set_environment(Some(white_environment()));
        let env_lit = render(&mut renderer, &camera, &color, &depth);

        let mut diff = 0usize;
        for (b, a) in sky_lit.iter().zip(env_lit.iter()) {
            if b.abs_diff(*a) > 8 {
                diff += 1;
            }
        }
        let total: usize = (SIZE * SIZE * 3) as usize;
        assert!(
            diff > total / 100,
            "white environment (env≈1.0) vs analytic sky (≈0.6) should repaint >1% of channels, diff = {diff}"
        );

        renderer.set_environment(None);
        let back_to_sky = render(&mut renderer, &camera, &color, &depth);
        let mut close = 0usize;
        for (b, a) in sky_lit.iter().zip(back_to_sky.iter()) {
            if b.abs_diff(*a) <= 8 {
                close += 1;
            }
        }
        assert!(close > total - 64, "clearing the environment should restore the sky lighting");
    }

    /// A full-screen quad (single UV island spanning the whole atlas) with a
    /// solid 64x64 albedo, for GPU round-trip update tests.
    fn textured_quad(color: [u8; 4]) -> MeshData {
        let rgba = color.repeat((64 * 64) as usize);
        MeshData {
            positions: vec![
                Vec3::new(-1.0, -1.0, 0.0),
                Vec3::new(1.0, -1.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(-1.0, 1.0, 0.0),
            ],
            normals: vec![Vec3::Z; 4],
            uvs: vec![(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)],
            indices: vec![0, 1, 2, 0, 2, 3],
            layers: vec![Layer::new(
                "Layer 1",
                TextureData {
                    width: 64,
                    height: 64,
                    rgba,
                },
            )],
            active_layer: 0,
            dirty: None,
        }
    }

    fn red_dominant(pixels: &[u8]) -> usize {
        // Strong red that clearly dominates green and blue (a painted red blob),
        // not the analytic-sky backdrop whose warm end is only a few 8-bit steps
        // above the other channels.
        pixels
            .chunks_exact(4)
            .filter(|p| p[0] > 150 && p[0] as u16 > p[1] as u16 + 40 && p[0] as u16 > p[2] as u16 + 40)
            .count()
    }

    fn sky_backdrop(pixels: &[u8]) -> usize {
        // The freshly drawn analytic sky (exposure/env 1.0) reads bright and
        // near-neutral in gamma space (~205-225 across the gradient); the old
        // flat clear color is gone, so this is the "what's behind" detector.
        pixels
            .chunks_exact(4)
            .filter(|p| {
                p[3] == 255
                    && (190..=235).contains(&p[0])
                    && (190..=235).contains(&p[1])
                    && (190..=235).contains(&p[2])
                    && p[0].abs_diff(p[1]) < 30
                    && p[1].abs_diff(p[2]) < 30
            })
            .count()
    }

    #[test]
    fn transparent_texels_leave_the_backdrop() {
        // A fully transparent albedo (alpha 0) must render NOTHING: its
        // fragments are discarded, so the pixel keeps the clear backdrop.
        // Nothing is composited over the transparent part.
        let (device, queue) = device_and_queue();
        let mesh = textured_quad([0, 0, 0, 0]);
        let (color, depth) = {
            let mk = |format, usage| {
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("test_view"),
                    size: wgpu::Extent3d {
                        width: SIZE,
                        height: SIZE,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                })
            };
            (
                mk(
                    wgpu::TextureFormat::Rgba8Unorm,
                    wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                ),
                mk(
                    wgpu::TextureFormat::Depth32Float,
                    wgpu::TextureUsages::RENDER_ATTACHMENT,
                ),
            )
        };

        let mut renderer = Renderer::new(device.clone(), queue.clone());
        renderer.set_mesh(&mesh);
        let mut camera = Camera::new(1.0);
        camera.fit(Vec3::ZERO, 1.0);
        renderer.render(
            &camera,
            &color.create_view(&Default::default()),
            &depth.create_view(&Default::default()),
        );

        let px = read_pixels(&device, &queue, &color);
        // The transparent quad is discarded everywhere, so the whole frame is
        // the analytic-sky backdrop (bright, near-neutral gradient) — nothing
        // dark, and nothing composited over the transparent part.
        let backdrop = sky_backdrop(&px);
        let total = (SIZE * SIZE) as usize;
        assert!(
            backdrop > total / 2,
            "fully transparent texels must be discarded, leaving the sky backdrop, got {backdrop}/{total}"
        );
    }

    #[test]
    fn intermediate_alpha_blends_toward_the_backdrop() {
        // A texel at alpha 128 renders as a semi-transparent surface: it is
        // source-over blended toward whatever is behind it (the clear
        // backdrop). It must NOT vanish (alpha == 0 discard) and must NOT turn
        // black or stay fully red — the backdrop shows through it.
        let (device, queue) = device_and_queue();
        let mesh = textured_quad([200, 0, 0, 128]);
        let (color, depth) = {
            let mk = |format, usage| {
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("test_view"),
                    size: wgpu::Extent3d {
                        width: SIZE,
                        height: SIZE,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                })
            };
            (
                mk(
                    wgpu::TextureFormat::Rgba8Unorm,
                    wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                ),
                mk(
                    wgpu::TextureFormat::Depth32Float,
                    wgpu::TextureUsages::RENDER_ATTACHMENT,
                ),
            )
        };

        let mut renderer = Renderer::new(device.clone(), queue.clone());
        renderer.set_mesh(&mesh);
        let mut camera = Camera::new(1.0);
        camera.fit(Vec3::ZERO, 1.0);
        renderer.render(
            &camera,
            &color.create_view(&Default::default()),
            &depth.create_view(&Default::default()),
        );

        let px = read_pixels(&device, &queue, &color);
        // Fully opaque red at 200 + alpha 0.5 blends toward the backdrop:
        // red rises above the backdrop while staying well below the opaque red
        // (i.e. ~100 in gamma space). The quad leaves a backdrop margin at the
        // viewport corners, so only that thin band stays at the clear color.
        let backdrop = px
            .chunks_exact(4)
            .filter(|p| {
                (116..=140).contains(&p[0])
                    && (120..=145).contains(&p[1])
                    && (125..=155).contains(&p[2])
                    && p[3] == 255
            })
            .count();
        let blended_red = px
            .chunks_exact(4)
            .filter(|p| {
                (140..=198).contains(&p[0])
                    && p[0] > p[1] + 20
                    && p[0] > p[2] + 20
                    && p[3] == 255
            })
            .count();
        let any_black = px
            .chunks_exact(4)
            .filter(|p| p[0] < 10 && p[1] < 10 && p[2] < 10 && p[3] == 255)
            .count();
        assert!(
            backdrop < 130_000,
            "alpha 128 must render (not be discarded) across the quad, kept backdrop {backdrop}"
        );
        assert!(
            blended_red > 100_000,
            "semi-transparent texels must blend toward the backdrop, got {blended_red} shaded px"
        );
        assert_eq!(
            any_black, 0,
            "semi-transparent texels must not turn black, got {any_black}"
        );
    }

    #[test]
    fn render_then_render_after_texture_update_changes_texels() {
        use crate::paint::{apply_stamp, brush_radius_world, mesh_raycast, StampMode};

        let (device, queue) = device_and_queue();
        let mesh = textured_quad([50, 100, 150, 255]);
        let (color, depth) = {
            let mk = |format, usage| {
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("test_view"),
                    size: wgpu::Extent3d {
                        width: SIZE,
                        height: SIZE,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                })
            };
            (
                mk(
                    wgpu::TextureFormat::Rgba8Unorm,
                    wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                ),
                mk(
                    wgpu::TextureFormat::Depth32Float,
                    wgpu::TextureUsages::RENDER_ATTACHMENT,
                ),
            )
        };

        let mut renderer = Renderer::new(device.clone(), queue.clone());
        renderer.set_mesh(&mesh);
        let mut camera = Camera::new(1.0);
        camera.fit(Vec3::ZERO, 1.0);

        let render = |renderer: &Renderer, color: &wgpu::Texture| {
            renderer.render(
                &camera,
                &color.create_view(&Default::default()),
                &depth.create_view(&Default::default()),
            );
            read_pixels(&device, &queue, color)
        };

        let before = render(&renderer, &color);
        assert_eq!(red_dominant(&before), 0, "no red pixels before painting");

        // Paint a red blob into the middle of the atlas, then push it to the GPU.
        let mut mesh = mesh;
        let hit = mesh_raycast(&mesh, Vec3::new(0.0, 0.0, 2.0), Vec3::new(0.0, 0.0, -1.0))
            .expect("hits the quad");
        let radius = brush_radius_world(&mesh, &hit, 64, 64, 24.0);
        apply_stamp(
            &mut mesh,
            hit.position,
            radius,
            Vec3::new(0.0, 0.0, 2.0),
            Vec3::new(0.0, 0.0, -1.0),
            [255, 0, 0, 255],
            1.0,
            1.0,
            StampMode::Paint,
        );
        let tex = mesh.active_layer_texture().unwrap();
        renderer.update_texture(tex);

        let after = render(&renderer, &color);
        let red = red_dominant(&after);
        assert!(
            red > 4_000,
            "painted texels should be visible as red, got {red} red-dominant pixels"
        );
    }

    #[test]
    fn update_texture_region_patches_only_the_rect() {
        let (device, queue) = device_and_queue();
        let mesh = textured_quad([50, 100, 150, 255]);
        let (color, depth) = {
            let mk = |format, usage| {
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("test_view"),
                    size: wgpu::Extent3d {
                        width: SIZE,
                        height: SIZE,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                })
            };
            (
                mk(
                    wgpu::TextureFormat::Rgba8Unorm,
                    wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                ),
                mk(
                    wgpu::TextureFormat::Depth32Float,
                    wgpu::TextureUsages::RENDER_ATTACHMENT,
                ),
            )
        };

        let mut renderer = Renderer::new(device.clone(), queue.clone());
        renderer.set_mesh(&mesh);
        let mut camera = Camera::new(1.0);
        camera.fit(Vec3::ZERO, 1.0);

        let render = |renderer: &Renderer, color: &wgpu::Texture| {
            renderer.render(
                &camera,
                &color.create_view(&Default::default()),
                &depth.create_view(&Default::default()),
            );
            read_pixels(&device, &queue, color)
        };

        // Stamp a red 12x12 texel patch at (4,4) of the 64x64 atlas, then push
        // ONLY that sub-rect to the GPU.
        let region = crate::io::TextureData {
            width: 12,
            height: 12,
            rgba: [255, 0, 0, 255].repeat((12 * 12) as usize),
        };
        assert!(renderer.update_texture_region(&region, 4, 4, 64, 64));

        let px = render(&renderer, &color);
        let red = red_dominant(&px);
        let blue = px
            .chunks_exact(4)
            .filter(|p| p[3] == 255 && p[2] > 100 && p[2] > p[0] && p[2] > p[1])
            .count();
        let total = px.len() / 4;
        // 12/64 of the viewport in each axis -> roughly 9k of 512^2 pixels.
        assert!(
            red > 4_000,
            "the patched sub-rect should show up red, got {red}"
        );
        assert!(
            red < total / 4,
            "the patch must stay a small region, got {red}/{total} red"
        );
        assert!(
            blue > total / 2,
            "the untouched atlas must stay blue, got {blue}/{total}"
        );

        // Region outside the atlas is rejected, not silently dropped.
        let miss = crate::io::TextureData {
            width: 8,
            height: 8,
            rgba: [0, 255, 0, 255].repeat((8 * 8) as usize),
        };
        assert!(!renderer.update_texture_region(&miss, 60, 60, 64, 64));
    }

    #[test]
    fn renders_triangle_offscreen() {
        let (device, queue) = device_and_queue();
        let px = render_and_read(&device, &queue, &manual_triangle());
        let n = non_background_pixels(&px);
        assert!(
            n > 1000,
            "expected a filled triangle, only {n} non-background pixels"
        );
    }

    #[test]
    fn camera_pan_preserves_view_distance() {
        let mut cam = Camera::new(1.0);
        cam.fit(Vec3::ZERO, 1.0);
        let before = (cam.target - cam.eye).length();
        let target_before = cam.target;
        cam.pan(40.0, -15.0, 600.0);
        assert!(
            ((cam.target - cam.eye).length() - before).abs() < 1e-3,
            "panning must not change zoom distance"
        );
        assert_ne!(cam.target, target_before, "panning must move the target");
    }

    #[test]
    fn orbit_reaches_over_the_top() {
        let mut cam = Camera::new(1.0);
        cam.fit(Vec3::ZERO, 1.0);
        // A long vertical drag must arc the view well above the equator.
        for _ in 0..400 {
            cam.orbit(0.0, -0.02);
        }
        let dir = (cam.target - cam.eye).normalize_or_zero();
        assert!(
            dir.y < -0.8,
            "expected the view to swing over the top, dir.y={:.2}",
            dir.y
        );
        // And back down past the other side.
        for _ in 0..800 {
            cam.orbit(0.0, 0.02);
        }
        let dir = (cam.target - cam.eye).normalize_or_zero();
        assert!(
            dir.y > 0.8,
            "expected the view to swing under the bottom, dir.y={:.2}",
            dir.y
        );
    }

    #[test]
    fn through_hole_shows_lit_interior_from_multiple_angles() {
        // Erasing a hole must reveal the FAR interior wall through it, lit and
        // readable, from every camera angle — not the backdrop color and not a
        // dark/normal-flipped surface (the flip made back-face lighting depend
        // on the view, darkening the interior from some angles).
        use crate::io::{default_albedo, MeshData};
        use crate::paint::{apply_stamp, mesh_raycast, StampMode};

        let mut mesh = MeshData::uv_sphere(0.6, 12, 16).with_texture(default_albedo());

        let (device, queue) = device_and_queue();
        let mut renderer = Renderer::new(device.clone(), queue.clone());
        renderer.set_mesh(&mesh);

        let mut cam = Camera::new(1.0);
        cam.fit(Vec3::ZERO, 0.6);
        let (o, d) = cam.ray(0.0, 0.0);
        let hit = mesh_raycast(&mesh, o, d).expect("hit");
        apply_stamp(
            &mut mesh,
            hit.position,
            0.25,
            o,
            d,
            [0, 0, 0, 0],
            1.0,
            1.0,
            StampMode::Erase,
        );
        renderer.update_texture(mesh.active_layer_texture().unwrap());
        let center = hit.position;

        // Cameras aimed directly AT the erased point: the hole is at screen
        // center and only the far interior wall is behind it. Skip the grazing
        // silhouette angles (whose exit lands on a triangle edge apart) and
        // require a bright, lit wall in every non-degenerate view.
        for (name, off) in [
            ("front", (0.0_f32, 0.0_f32)),
            ("bottom", (0.0, -0.6)),
            ("top", (0.0, 0.6)),
            ("right", (0.6, 0.0)),
        ] {
            let mut c = Camera::new(1.0);
            c.target = center;
            let dir = Vec3::new(off.0, off.1, 1.0).normalize_or_zero();
            c.radius = 0.8;
            c.eye = center + dir * c.radius;
            let (color, depth) = {
                let mk = |format, usage| {
                    device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("v"),
                        size: wgpu::Extent3d {
                            width: SIZE,
                            height: SIZE,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format,
                        usage,
                        view_formats: &[],
                    })
                };
                (
                    mk(
                        wgpu::TextureFormat::Rgba8Unorm,
                        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                    ),
                    mk(
                        wgpu::TextureFormat::Depth32Float,
                        wgpu::TextureUsages::RENDER_ATTACHMENT,
                    ),
                )
            };
            renderer.render(
                &c,
                &color.create_view(&Default::default()),
                &depth.create_view(&Default::default()),
            );
            let px = read_pixels(&device, &queue, &color);
            let mid: usize = ((SIZE / 2) * SIZE + SIZE / 2) as usize * 4;
            let [r, g, b, _] = [px[mid], px[mid + 1], px[mid + 2], px[mid + 3]];
            assert!(
                [r, g, b].iter().all(|c| *c > 150),
                "{name}: far interior wall through the hole should be lit, got ({r},{g},{b})"
            );
            assert!(
                r + 4 >= b,
                "{name}: lit cream interior must not read as the bluish backdrop, got ({r},{g},{b})"
            );
        }
    }

    #[test]
    fn feathered_erase_rim_shows_lit_interior_everywhere() {
        // The eraser feather (outer 45% of the dab) leaves texels with
        // 0 < alpha < 1. Those translucent texels must NOT occlude the lit far
        // interior wall behind them: before the two-pass render they wrote
        // depth, failed-out the wall's fragments, and then blended over the
        // cool backdrop — "anything between 0 and 1 still has the issue".
        use crate::io::{default_albedo, MeshData};
        use crate::paint::{apply_stamp, mesh_raycast, StampMode};

        // Solid cream albedo keeps the warm interior / cool backdrop split
        // unambiguous (no checkerboard's own dark squares).
        let mut solid = default_albedo();
        for px in solid.rgba.chunks_exact_mut(4) {
            px[0] = 246;
            px[1] = 241;
            px[2] = 232;
        }
        let mut mesh = MeshData::uv_sphere(0.6, 12, 16).with_texture(solid);

        let (device, queue) = device_and_queue();
        let mut renderer = Renderer::new(device.clone(), queue.clone());
        renderer.set_mesh(&mesh);

        let mut cam = Camera::new(1.0);
        cam.fit(Vec3::ZERO, 0.6);
        let (o, d) = cam.ray(0.0, 0.0);
        let hit = mesh_raycast(&mesh, o, d).expect("hit");
        apply_stamp(
            &mut mesh,
            hit.position,
            0.25,
            o,
            d,
            [0, 0, 0, 0],
            1.0,
            1.0,
            StampMode::Erase,
        );
        renderer.update_texture(mesh.active_layer_texture().unwrap());

        // Pull the camera back so the feather ring projects fully on screen.
        let mut c = Camera::new(1.0);
        c.target = hit.position;
        c.eye = hit.position + Vec3::Z * 1.6;
        c.radius = 1.6;

        // Project the erased disc's rim onto the frame to know its screen width.
        let rim_x = |lat: f32| {
            let z = (0.6f32 * 0.6 - lat * lat).max(0.0).sqrt();
            let clip = c.view_proj() * Vec3::new(lat, 0.0, z).extend(1.0);
            (clip.x / clip.w * 0.5 + 0.5) * SIZE as f32
        };
        let cx = rim_x(0.0);
        let half = (rim_x(0.0) - rim_x(0.25)).abs();

        let (color, depth) = {
            let mk = |format, usage| {
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("feather_view"),
                    size: wgpu::Extent3d {
                        width: SIZE,
                        height: SIZE,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                })
            };
            (
                mk(
                    wgpu::TextureFormat::Rgba8Unorm,
                    wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                ),
                mk(
                    wgpu::TextureFormat::Depth32Float,
                    wgpu::TextureUsages::RENDER_ATTACHMENT,
                ),
            )
        };
        renderer.render(
            &c,
            &color.create_view(&Default::default()),
            &depth.create_view(&Default::default()),
        );
        let px = read_pixels(&device, &queue, &color);

        // Sweep the center row across the whole disc: the alpha-0 core AND the
        // 0<alpha<1 feather ring must read as the warm lit interior (bright,
        // r >= b) — never the cool backdrop (b > r, dimmer).
        let mid_y = (SIZE / 2) as usize;
        let mut bad = Vec::new();
        let start = (cx - half * 1.05).max(0.0) as usize;
        let end = ((cx + half * 1.05).min(SIZE as f32 - 1.0)) as usize;
        for x in 0..(end.max(start) - start) {
            let px_idx = (mid_y * SIZE as usize + (start + x)) * 4;
            let (r, g, b) = (px[px_idx], px[px_idx + 1], px[px_idx + 2]);
            if !(r > 150 && g > 150 && b > 150 && r >= b) {
                bad.push((start + x, r, g, b));
            }
        }
        assert!(
            bad.is_empty(),
            "center-row pixels across the erased disc must show the warm lit interior; \
             backdrop-tinted pixels at {:?} (first 5)",
            &bad[..bad.len().min(5)]
        );
    }

    #[test]
    fn erase_on_default_sphere_leaves_backdrop() {
        // Mirrors the app exactly: startup sphere, erase a blob on the front,
        // push the atlas, render. Erased texels (alpha 0) are discarded and
        // the backdrop shows through — nothing is composited over them.
        use crate::io::{default_albedo, MeshData};
        use crate::paint::{apply_stamp, mesh_raycast, StampMode};

        // Solid cream albedo: any dark pixel here is a lighting artifact, not
        // the checkerboard's own darker squares.
        let mut solid = default_albedo();
        for px in solid.rgba.chunks_exact_mut(4) {
            px[0] = 246;
            px[1] = 241;
            px[2] = 232;
        }
        let mut mesh = MeshData::uv_sphere(0.6, 12, 16).with_texture(solid);

        let (device, queue) = device_and_queue();
        let mut renderer = Renderer::new(device.clone(), queue.clone());
        renderer.set_mesh(&mesh);

        // Camera framing the sphere front (as in the app), hit dead-center.
        let mut camera = Camera::new(1.0);
        camera.fit(Vec3::ZERO, 0.6);
        let (o, d) = camera.ray(0.0, 0.0);
        let hit = mesh_raycast(&mesh, o, d).expect("ray should hit the sphere");

        // Erase a large disc on the near hemisphere.
        apply_stamp(
            &mut mesh,
            hit.position,
            0.45,
            o,
            d,
            [0, 0, 0, 0],
            1.0,
            1.0,
            StampMode::Erase,
        );
        renderer.update_texture(mesh.active_layer_texture().unwrap());

        let (color, depth) = {
            let mk = |format, usage| {
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("test_view"),
                    size: wgpu::Extent3d {
                        width: SIZE,
                        height: SIZE,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                })
            };
            (
                mk(
                    wgpu::TextureFormat::Rgba8Unorm,
                    wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                ),
                mk(
                    wgpu::TextureFormat::Depth32Float,
                    wgpu::TextureUsages::RENDER_ATTACHMENT,
                ),
            )
        };
        renderer.render(
            &camera,
            &color.create_view(&Default::default()),
            &depth.create_view(&Default::default()),
        );

        let px = read_pixels(&device, &queue, &color);
        // The analytic-sky backdrop surrounds the sphere.
        let backdrop = sky_backdrop(&px);
        // The sphere (front, and the far interior shown through the erased
        // disc) is the lit cream albedo — clearly brighter than the backdrop.
        let lit_sphere = px
            .chunks_exact(4)
            .filter(|p| p[3] == 255 && p[0] > 170 && p[1] > 160 && p[2] > 145)
            .count();
        // Through the erased disc the far side must be visible and fully lit —
        // never a dark/black floor from un-lit backfaces.
        let dark = px
            .chunks_exact(4)
            .filter(|p| p[3] == 255 && p[0] < 120 && p[1] < 120 && p[2] < 120)
            .count();
        // Nothing from the old checker palette (magenta / blue).
        let any_checker = px
            .chunks_exact(4)
            .filter(|p| p[3] == 255 && p[0] > 180 && p[2] > 180 && p[1] < 60)
            .count();
        assert!(
            backdrop > 1000,
            "the sphere should sit on the backdrop, got {backdrop}"
        );
        assert!(
            lit_sphere > 2000,
            "the far interior must render through the erased hole, got {lit_sphere}"
        );
        assert_eq!(
            dark, 0,
            "every visible backface through the hole must be lit (geometric normals + backfill), got {dark} dark px"
        );
        assert_eq!(
            any_checker, 0,
            "erased sphere texels must not be checker-filled, got {any_checker}"
        );
    }

    #[test]
    fn erase_top_reveals_bottom_through_composite() {
        // A quad with TWO solid layers: blue on the bottom, green on top with
        // a transparent hole in the middle. The flattened composite must show
        // blue through the hole while the rest stays green.
        use crate::io::{Layer, MeshData, TextureData};

        let mut bottom = vec![0u8; 64 * 64 * 4];
        let mut top = vec![0u8; 64 * 64 * 4];
        for px in bottom.chunks_exact_mut(4) {
            px.copy_from_slice(&[0, 0, 255, 255]); // solid blue
        }
        for i in 0..(64 * 64) {
            let (x, y) = (i % 64, i / 64);
            if (24..40).contains(&x) && (24..40).contains(&y) {
                top[i * 4..i * 4 + 4].copy_from_slice(&[0, 0, 0, 0]); // erased hole
            } else {
                top[i * 4..i * 4 + 4].copy_from_slice(&[0, 255, 0, 255]);
            }
        }
        let mesh = MeshData {
            positions: vec![
                Vec3::new(-1.0, -1.0, 0.0),
                Vec3::new(1.0, -1.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(-1.0, 1.0, 0.0),
            ],
            normals: vec![Vec3::Z; 4],
            uvs: vec![(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)],
            indices: vec![0, 1, 2, 0, 2, 3],
            layers: vec![
                Layer::new(
                    "bottom",
                    TextureData {
                        width: 64,
                        height: 64,
                        rgba: bottom,
                    },
                ),
                Layer::new(
                    "top",
                    TextureData {
                        width: 64,
                        height: 64,
                        rgba: top,
                    },
                ),
            ],
            active_layer: 1,
            dirty: None,
        };

        let (device, queue) = device_and_queue();
        let px = render_and_read(&device, &queue, &mesh);

        let blue: usize = px
            .chunks_exact(4)
            .filter(|p| p[3] == 255 && p[2] > 100 && p[2] > p[0] && p[2] > p[1])
            .count();
        let green: usize = px
            .chunks_exact(4)
            .filter(|p| p[3] == 255 && p[1] > 100 && p[1] > p[0] && p[1] > p[2])
            .count();

        // The erased hole must show the blue layer through the green top, so
        // both colors appear on screen in substantial amounts.
        assert!(
            blue > 1_000,
            "blue should show through the erased hole, got {blue} blue pixels"
        );
        assert!(
            green > 1_000,
            "un-erased area should stay green, got {green} green pixels"
        );
    }

    #[test]
    fn renders_glb_from_env() {
        let path = match std::env::var("PIXFORGE_TEST_MODEL") {
            Ok(p) => p,
            Err(_) => return, // optional test, run with PIXFORGE_TEST_MODEL=/path/model.glb
        };
        let mesh = match load_gltf(&path) {
            crate::io::LoadedModel::Mesh(m) => m,
            crate::io::LoadedModel::Invalid => panic!("load_gltf returned Invalid for {path}"),
        };
        assert!(
            mesh.flattened_atlas().is_some_and(|t| t.rgba.len() >= 4),
            "expected a base-color texture atlas for {path}"
        );
        let (device, queue) = device_and_queue();
        let px = render_and_read(&device, &queue, &mesh);
        let n = non_background_pixels(&px);
        assert!(
            n > 1000,
            "expected wall.glb to fill the viewport, only {n} non-background pixels"
        );
        let colors = distinct_colors(&px);
        assert!(
            colors > 100,
            "expected rich albedo colors, only {colors} distinct colors (texture may not be sampling)"
        );
    }

    #[test]
    fn uv_overlays_add_distinct_tones() {
        // A solid flat quad renders with a view-dependent PBR gradient; the shader
        // UV checker and grid overlays must each visibly recolor the frame,
        // add tone variety, and differ from each other.
        let (device, queue) = device_and_queue();
        let mesh = textured_quad([160, 160, 160, 255]);
        let base = render_and_read_with_overlay(&device, &queue, &mesh, 0);
        let checker = render_and_read_with_overlay(&device, &queue, &mesh, UV_OVERLAY_CHECKER);
        let grid = render_and_read_with_overlay(&device, &queue, &mesh, UV_OVERLAY_GRID);
        let base_colors = distinct_colors(&base);
        assert_ne!(checker, base, "checker overlay must change the render");
        assert_ne!(grid, base, "grid overlay must change the render");
        assert_ne!(checker, grid, "checker and grid overlays must differ");
        assert!(
            distinct_colors(&checker) > base_colors,
            "checker overlay should add tones ({base_colors} base)"
        );
        assert!(
            distinct_colors(&grid) > base_colors,
            "grid overlay should add tones ({base_colors} base)"
        );
    }

    #[test]
    fn material_params_reshape_the_lighting() {
        // The Material uniforms must actually reach the shader: exposure,
        // sun intensity, sky light, metallic and the interior fill each change
        // the lit quad's output. The quad faces the camera, so the sun term
        // (from +X/+Y/+Z) is strongest off-center — just require visible deltas.
        let (device, queue) = device_and_queue();
        let mesh = textured_quad([180, 180, 180, 255]);
        let center_px = |px: &[u8]| {
            let mid = ((SIZE / 2) * SIZE + SIZE / 2) as usize * 4;
            (px[mid], px[mid + 1], px[mid + 2])
        };

        let base = render_and_read_material(&device, &queue, &mesh, 0, Material::default());
        let exposed = render_and_read_material(
            &device,
            &queue,
            &mesh,
            0,
            Material { exposure: 3.0, ..Material::default() },
        );
        let no_sun = render_and_read_material(
            &device,
            &queue,
            &mesh,
            0,
            Material { sun_intensity: 0.0, ..Material::default() },
        );
        let no_sky = render_and_read_material(
            &device,
            &queue,
            &mesh,
            0,
            Material { env_intensity: 0.0, ..Material::default() },
        );
        let metallic = render_and_read_material(
            &device,
            &queue,
            &mesh,
            0,
            Material { metallic: 1.0, roughness: 0.15, ..Material::default() },
        );
        // AO darkens only diffuse sky light; on a purely metallic surface it
        // must leave the (specular) reflection untouched.
        let metal_ao0 = render_and_read_material(
            &device,
            &queue,
            &mesh,
            0,
            Material { metallic: 1.0, roughness: 0.15, ambient_occlusion: 0.0, ..Material::default() },
        );
        let metal_ao1 = render_and_read_material(
            &device,
            &queue,
            &mesh,
            0,
            Material { metallic: 1.0, roughness: 0.15, ..Material::default() },
        );
        let glowing = render_and_read_material(
            &device,
            &queue,
            &mesh,
            0,
            Material { emissive: 1.2, ..Material::default() },
        );

        assert_eq!(metal_ao0, metal_ao1, "AO must not dim metallic reflections");

        assert_ne!(center_px(&exposed), center_px(&base), "exposure must brighten");
        assert_ne!(center_px(&no_sun), center_px(&base), "sun intensity must matter");
        assert_ne!(center_px(&no_sky), center_px(&base), "sky light must matter");
        assert_ne!(center_px(&metallic), center_px(&base), "metallic look must differ");
        assert_ne!(center_px(&glowing), center_px(&base), "emission must show");
        assert!(
            mean_luma(&glowing) > mean_luma(&base) + 30.0,
            "emissive quad should be clearly brighter"
        );
    }

    #[test]
    fn sun_off_removes_all_directional_shading() {
        // Normals vary a lot on a sphere, so any remaining directional light
        // shows up as a bright/dark split. With the sun toggled off the direct
        // term AND the camera fill (which keys off the sun direction) must
        // both vanish — leaving only the uniform skybox on every face.
        let (device, queue) = device_and_queue();
        let mut sphere = MeshData::uv_sphere(1.0, 48, 64);
        // No albedo layer = pure white texture that clips the sun to flat 255
        // and hides all shading; a mid-gray layer keeps the PBR gradient visible.
        sphere.layers.push(Layer::new(
            "base",
            TextureData {
                width: 8,
                height: 8,
                rgba: vec![80u8, 80, 80, 255].repeat(64),
            },
        ));
        let sun_on = render_and_read_material(&device, &queue, &sphere, 0, Material::default());
        let sun_off = render_and_read_material(
            &device,
            &queue,
            &sphere,
            0,
            Material { sun_enabled: false, ..Material::default() },
        );

        // Mean luma over the top and bottom quarters of the central column
        // (well inside the sphere, away from background corners).
        let s = SIZE as usize;
        let range = s / 3..2 * s / 3;
        let half_luma = |px: &[u8], y0: usize, y1: usize| -> f32 {
            let (mut sum, mut n) = (0.0f32, 0.0f32);
            for y in y0..y1 {
                for x in range.clone() {
                    let p = (y * s + x) * 4;
                    sum += 0.2126 * px[p] as f32 + 0.7152 * px[p + 1] as f32 + 0.0722 * px[p + 2] as f32;
                    n += 1.0;
                }
            }
            sum / n
        };

        let gap = |px: &[u8]| half_luma(px, s / 4, s * 2 / 5) - half_luma(px, s * 3 / 5, 3 * s / 4);
        let sun_on_gap = (gap(&sun_on) as f32).abs();
        let sun_off_gap = (gap(&sun_off) as f32).abs();
        assert!(
            sun_on_gap > 6.0,
            "sun should create a visible top/bottom split, got {sun_on_gap}"
        );
        assert!(
            sun_off_gap < 3.0,
            "sun off must leave flat uniform ambient, got top/bottom gap {sun_off_gap}"
        );
    }

    fn mean_luma(px: &[u8]) -> f32 {
        let region = SIZE / 4..SIZE - SIZE / 4;
        let (mut sum, mut n) = (0.0f32, 0.0f32);
        for y in region.clone() {
            for x in region.clone() {
                let p = (y * SIZE + x) as usize * 4;
                sum += 0.2126 * px[p] as f32 + 0.7152 * px[p + 1] as f32 + 0.0722 * px[p + 2] as f32;
                n += 1.0;
            }
        }
        sum / n
    }

    fn render_sphere_with_overlay(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh: &MeshData,
        overlay: Option<BrushOverlay>,
        sprite: Option<(&crate::io::TextureData, u64)>,
    ) -> (Vec<u8>, Camera) {
        let color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("overlay_test_color"),
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("overlay_test_depth"),
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let mut renderer = Renderer::new(device.clone(), queue.clone());
        renderer.set_mesh(mesh);
        if let Some((s, sig)) = sprite {
            renderer.set_brush_sprite(sig, s);
        }
        renderer.brush_overlay = overlay;
        let mut camera = Camera::new(1.0);
        let min = mesh.positions.iter().fold(Vec3::MAX, |a, b| a.min(*b));
        let max = mesh.positions.iter().fold(Vec3::MIN, |a, b| a.max(*b));
        let center = (min + max) * 0.5;
        let radius = mesh
            .positions
            .iter()
            .map(|p| (p - center).length())
            .fold(0.0, f32::max);
        camera.fit(center, radius);
        renderer.render(
            &camera,
            &color.create_view(&Default::default()),
            &depth.create_view(&Default::default()),
        );
        (read_pixels(device, queue, &color), camera)
    }

    #[test]
    fn brush_overlay_texture_cursor_never_tints_the_back_side() {
        // The cursor mask must never tint the far/back side of the model (nor
        // anything outside the brush footprint). The worst case is a brush
        // parked near the silhouette, where front and back depths are equal;
        // a convex model like this sphere still must show the mask only on the
        // visible surface fragments.
        let (device, queue) = device_and_queue();
        let mut sphere = MeshData::uv_sphere(1.0, 48, 64);
        sphere.layers.push(Layer::new(
            "base",
            TextureData {
                width: 8,
                height: 8,
                rgba: [120u8, 120, 120, 255].repeat(64),
            },
        ));
        let (center, radius) = {
            let min = sphere.positions.iter().fold(Vec3::MAX, |a, b| a.min(*b));
            let max = sphere.positions.iter().fold(Vec3::MIN, |a, b| a.max(*b));
            let c = (min + max) * 0.5;
            let r = sphere
                .positions
                .iter()
                .map(|p| (p - c).length())
                .fold(0.0, f32::max);
            (c, r)
        };
        let r = 0.45;
        let mut camera = Camera::new(1.0);
        camera.fit(center, radius);
        let v = (camera.eye - center).normalize_or_zero();
        let up = Vec3::Y;
        let sil_dir = (up - v * up.dot(v)).normalize_or_zero();
        // Brush at 75° from the view axis: just short of the silhouette, so a
        // healthy part of the footprint lies on the visible front.
        let hit_pos = radius * (v * 75f32.to_radians().cos() + sil_dir * 75f32.to_radians().sin());
        let (axis_u, axis_v) =
            crate::paint::brush_axes(&sphere.positions, &sphere.indices, hit_pos, r, v);
        let sprite = TextureData {
            width: 4,
            height: 4,
            rgba: [255u8, 255, 255, 255].repeat(16),
        };
        let (_, cam) = render_sphere_with_overlay(&device, &queue, &sphere, None, None);
        let s = SIZE as usize;

        // Reconstruct the sphere surface point under a pixel so we can check
        // the shader's own footprint test (|tu|,|tv| <= r) and the back-face
        // criterion (normal facing the eye) exactly as the mask must.
        let surface_point = |px: f32, py: f32| -> Option<(Vec3, Vec3)> {
            let ndc_x = px / SIZE as f32 * 2.0 - 1.0;
            let ndc_y = 1.0 - py / SIZE as f32 * 2.0;
            let (o, d) = cam.ray(ndc_x, ndc_y);
            let b = o.dot(d);
            let c = o.length_squared() - 1.0;
            let disc = b * b - c;
            if disc <= 0.0 {
                return None;
            }
            let t = -b - disc.sqrt();
            let pos = o + d * t;
            Some((pos, pos.normalize_or_zero()))
        };

        for shape in [1u32, 2u32, 3u32] {
            let o = BrushOverlay {
                center: hit_pos,
                axis_u,
                axis_v,
                radius: r,
                shape,
                color: [1.0, 0.0, 0.0, 0.8],
                rotation: 0.0,
                flip_x: false,
                flip_y: false,
            };
            let sig = 10 + shape as u64;
            let (ref_img, _) =
                render_sphere_with_overlay(&device, &queue, &sphere, None, Some((&sprite, sig)));
            let (full, _) = render_sphere_with_overlay(
                &device,
                &queue,
                &sphere,
                Some(o),
                Some((&sprite, sig)),
            );
            let mut tinted = 0u32;
            let mut outside_footprint = 0u32;
            let mut back_facing = 0u32;
            for y in 0..s as i32 {
                for x in 0..s as i32 {
                    let i = (y as usize * s + x as usize) * 4;
                    let delta = (ref_img[i] as i32 - full[i] as i32).abs()
                        + (ref_img[i + 1] as i32 - full[i + 1] as i32).abs()
                        + (ref_img[i + 2] as i32 - full[i + 2] as i32).abs();
                    if delta <= 12 {
                        continue;
                    }
                    let (p, n) = surface_point(x as f32 + 0.5, y as f32 + 0.5)
                        .expect("a tinted pixel must lie on the sphere");
                    tinted += 1;
                    let tu = (p - hit_pos).dot(axis_u);
                    let tv = (p - hit_pos).dot(axis_v);
                    if tu.abs() > r + 1e-2 || tv.abs() > r + 1e-2 {
                        outside_footprint += 1;
                    }
                    if n.dot(cam.eye - p) <= 0.0 {
                        back_facing += 1;
                    }
                }
            }
            assert!(tinted > 30, "shape {shape}: the cursor must actually tint the surface");
            assert_eq!(
                outside_footprint, 0,
                "shape {shape}: the cursor tinted pixels outside the brush footprint"
            );
            assert_eq!(
                back_facing, 0,
                "shape {shape}: the cursor tinted pixels on the far/back side of the model"
            );
        }
    }

    #[test]
    fn brush_overlay_conforms_to_the_surface() {
        // The cursor mask must tint exactly the surface fragments inside the
        // brush footprint — nothing outside it, and never 'through' the model.
        let (device, queue) = device_and_queue();
        let mut sphere = MeshData::uv_sphere(1.0, 48, 64);
        sphere.layers.push(Layer::new(
            "base",
            TextureData {
                width: 8,
                height: 8,
                rgba: vec![120u8, 120, 120, 255].repeat(64),
            },
        ));

        // Camera looks at the origin from (3,2,3)/|.|·radius; the sphere's
        // nearest point to the eye sits on that ray and projects to image
        // center. Park a round footprint of radius 0.4 there.
        let fit = {
            let min = sphere.positions.iter().fold(Vec3::MAX, |a, b| a.min(*b));
            let max = sphere.positions.iter().fold(Vec3::MIN, |a, b| a.max(*b));
            let center = (min + max) * 0.5;
            let radius = sphere
                .positions
                .iter()
                .map(|p| (p - center).length())
                .fold(0.0, f32::max);
            (center, radius)
        };
        let dir = Vec3::new(3.0, 2.0, 3.0).normalize();
        let front = fit.0 + dir * fit.1;
        let axis_u = dir.cross(Vec3::Y).normalize();
        let axis_v = dir.cross(axis_u).normalize();

        let (base, camera) = render_sphere_with_overlay(&device, &queue, &sphere, None, None);
        let (over, camera2) = render_sphere_with_overlay(
            &device,
            &queue,
            &sphere,
            Some(BrushOverlay {
                center: front,
                axis_u,
                axis_v,
                radius: 0.4,
                shape: 0,
                color: [1.0, 0.0, 0.0, 0.8],
                rotation: 0.0,
                flip_x: false,
                flip_y: false,
            }),
            None,
        );
        assert_eq!(camera.eye, camera2.eye, "camera must match across renders");

        // Project a point on the footprint's rim so we know how many pixels the
        // mask may legally cover on screen.
        let rim = camera
            .view_proj()
            .project_point3(front + axis_u * 0.4);
        let rim_px = ((rim.x * 0.5 + 0.5) * SIZE as f32).round() as i32;
        let c = SIZE as i32 / 2;
        let rim_dist = (rim_px - c).abs().max(1) as f32;
        let allow = rim_dist * 1.6 + 3.0;

        let s = SIZE as usize;
        let (mut changed, mut max_core_dist, mut max_far_dist) = (0u32, 0f32, 0f32);
        for y in 0..s {
            for x in 0..s {
                let p = (y * s + x) * 4;
                if (base[p] as i32 - over[p] as i32).abs()
                    + (base[p + 1] as i32 - over[p + 1] as i32).abs()
                    + (base[p + 2] as i32 - over[p + 2] as i32).abs()
                    < 12
                {
                    continue;
                }
                changed += 1;
                let d = (((x as i32 - c).pow(2) + (y as i32 - c).pow(2)) as f32).sqrt();
                if d >= allow {
                    max_far_dist = max_far_dist.max(d);
                } else {
                    max_core_dist = max_core_dist.max(d);
                }
            }
        }
        assert!(
            changed > 1000,
            "the footprint must tint a real patch of the surface, got {changed} px"
        );
        assert_eq!(
            max_far_dist, 0.0,
            "no pixel outside the footprint may be tinted (nearest leak {max_far_dist:.1}px, allowance {allow:.1}px)"
        );
        // Center of the mask must actually get the brush tint.
        let pc = (c as usize) * s + c as usize;
        assert!(
            over[pc * 4] as i32 - base[pc * 4] as i32 > 25,
            "center of the mask must redden (base {} over {})",
            base[pc * 4],
            over[pc * 4]
        );
        // And the mask must be *round* on a sphere viewed straight on: the
        // painted patch is a disc, so its widest core sample ≈ its radius.
        assert!(
            max_core_dist > rim_dist * 0.8,
            "mask should extend near the footprint rim (core {max_core_dist:.1}px rim {rim_dist:.1}px)"
        );
    }

    #[test]
    fn brush_overlay_texture_uses_sprite_alpha() {
        // The texture-shape mask must tint exactly the sprite's covered texels:
        // a central-dot sprite tints a central patch and LEAVES the surrounding
        // square within the footprint untouched (the sprite's transparent alpha
        // gates it), matching the real stamp.
        let (device, queue) = device_and_queue();
        let mut sphere = MeshData::uv_sphere(1.0, 48, 64);
        sphere.layers.push(Layer::new(
            "base",
            TextureData {
                width: 8,
                height: 8,
                rgba: vec![120u8, 120, 120, 255].repeat(64),
            },
        ));

        // 4x4 sprite: only the central 2x2 block is opaque.
        let mut rgba = vec![0u8; 4 * 4 * 4];
        for y in 0..4u32 {
            for x in 0..4u32 {
                let i = ((y * 4 + x) * 4) as usize;
                if x >= 1 && x <= 2 && y >= 1 && y <= 2 {
                    rgba[i..i + 4].copy_from_slice(&[255, 255, 255, 255]);
                }
            }
        }
        let sprite = TextureData {
            width: 4,
            height: 4,
            rgba,
        };

        let fit = {
            let min = sphere.positions.iter().fold(Vec3::MAX, |a, b| a.min(*b));
            let max = sphere.positions.iter().fold(Vec3::MIN, |a, b| a.max(*b));
            let center = (min + max) * 0.5;
            let radius = sphere
                .positions
                .iter()
                .map(|p| (p - center).length())
                .fold(0.0, f32::max);
            (center, radius)
        };
        let dir = Vec3::new(3.0, 2.0, 3.0).normalize();
        let front = fit.0 + dir * fit.1;
        let axis_u = dir.cross(Vec3::Y).normalize();
        let axis_v = dir.cross(axis_u).normalize();

        let overlay = BrushOverlay {
            center: front,
            axis_u,
            axis_v,
            radius: 0.4,
            shape: 3,
            color: [1.0, 0.0, 0.0, 0.8],
            rotation: 0.0,
            flip_x: false,
            flip_y: false,
        };
        let (base, camera) = render_sphere_with_overlay(&device, &queue, &sphere, None, None);
        let (over, camera2) =
            render_sphere_with_overlay(&device, &queue, &sphere, Some(overlay), Some((&sprite, 9)));
        assert_eq!(camera.eye, camera2.eye, "camera must match across renders");

        let c = SIZE as i32 / 2;
        let s = SIZE as usize;
        // Project the footprint square's corner and the central dot's corner so
        // we know the two radii to expect on screen (pixel distance from the
        // image center, which is where the sphere's front point lands).
        let to_px = |p: Vec3| -> f32 {
            let clip = camera.view_proj().project_point3(p);
            let x = (clip.x * 0.5 + 0.5) * SIZE as f32;
            let y = (clip.y * 0.5 + 0.5) * SIZE as f32;
            (((x - c as f32).powi(2) + (y - c as f32).powi(2)) as f32).sqrt()
        };
        let allow_sq = to_px(front + (axis_u + axis_v) * 0.4) * 1.4 + 4.0;
        let dot_radius = to_px(front + (axis_u + axis_v) * 0.2) * 1.35 + 3.0;

        let (mut changed, mut unchanged_in_sq, mut max_leak) = (0u32, 0u32, 0f32);
        for y in 0..s {
            for x in 0..s {
                let d = (((x as i32 - c).pow(2) + (y as i32 - c).pow(2)) as f32).sqrt();
                if d > allow_sq {
                    continue;
                }
                let p = (y * s + x) * 4;
                let delta = (base[p] as i32 - over[p] as i32).abs()
                    + (base[p + 1] as i32 - over[p + 1] as i32).abs()
                    + (base[p + 2] as i32 - over[p + 2] as i32).abs();
                if delta < 12 {
                    unchanged_in_sq += 1;
                } else {
                    changed += 1;
                    if d > dot_radius {
                        max_leak = max_leak.max(d);
                    }
                }
            }
        }
        assert!(
            changed > 200,
            "the sprite's opaque center must tint a central patch, got {changed} px"
        );
        assert_eq!(
            max_leak, 0.0,
            "sprite alpha must gate the mask: no tint outside the central dot (leak {max_leak:.1}px > dot {dot_radius:.1}px)"
        );
        assert!(
            unchanged_in_sq > 200,
            "the square ring inside the footprint must stay unpainted, got only {unchanged_in_sq} px"
        );
    }

    #[test]
    fn brush_overlay_texture_sprite_is_not_mirrored() {
        // Regression: the texture-shape cursor mask must map the sprite onto
        // the footprint upright — the sprite's top-left tile must reach the
        // image's top-left corner, matching the 2D preview — not a mirrored
        // position (the v-coordinate used to be flipped in the overlay shader).
        let (device, queue) = device_and_queue();
        let mut sphere = MeshData::uv_sphere(1.0, 48, 64);
        sphere.layers.push(Layer::new(
            "base",
            TextureData {
                width: 8,
                height: 8,
                rgba: vec![120u8, 120, 120, 255].repeat(64),
            },
        ));

        let fit = {
            let min = sphere.positions.iter().fold(Vec3::MAX, |a, b| a.min(*b));
            let max = sphere.positions.iter().fold(Vec3::MIN, |a, b| a.max(*b));
            let center = (min + max) * 0.5;
            let radius = sphere
                .positions
                .iter()
                .map(|p| (p - center).length())
                .fold(0.0, f32::max);
            (center, radius)
        };
        let r = 0.45;
        let mut camera = Camera::new(1.0);
        camera.fit(fit.0, fit.1);
        let (o, d) = camera.ray(0.0, 0.0);
        let hit = crate::paint::mesh_raycast(&sphere, o, d).expect("center ray must hit");
        let (axis_u, axis_v) =
            crate::paint::brush_axes(&sphere.positions, &sphere.indices, hit.position, r, d);

        // 4x4 sprite with a single opaque tile at its top-left corner (0,0).
        let mut rgba = vec![0u8; 4 * 4 * 4];
        rgba[0..4].copy_from_slice(&[255, 255, 255, 255]);
        let sprite = TextureData {
            width: 4,
            height: 4,
            rgba,
        };
        let overlay = BrushOverlay {
            center: hit.position,
            axis_u,
            axis_v,
            radius: r,
            shape: 3,
            color: [1.0, 0.0, 0.0, 0.8],
            rotation: 0.0,
            flip_x: false,
            flip_y: false,
        };
        let (base, cam) = render_sphere_with_overlay(&device, &queue, &sphere, None, None);
        let (over, cam2) =
            render_sphere_with_overlay(&device, &queue, &sphere, Some(overlay), Some((&sprite, 7)));
        assert_eq!(cam.eye, cam2.eye, "camera must match across renders");

        let clip = cam.view_proj().project_point3(hit.position);
        let (icx, icy) = (
            (clip.x * 0.5 + 0.5) * SIZE as f32,
            (clip.y * 0.5 + 0.5) * SIZE as f32,
        );
        let s = SIZE as usize;
        let (mut sum_x, mut sum_y, mut count) = (0.0, 0.0, 0u32);
        for y in 0..s {
            for x in 0..s {
                let i = (y * s + x) * 4;
                let delta = (base[i] as i32 - over[i] as i32).abs()
                    + (base[i + 1] as i32 - over[i + 1] as i32).abs()
                    + (base[i + 2] as i32 - over[i + 2] as i32).abs();
                if delta > 12 {
                    sum_x += x as f32;
                    sum_y += y as f32;
                    count += 1;
                }
            }
        }
        assert!(count > 100, "the sprite's opaque tile must tint some pixels");
        let (gx, gy) = (sum_x / count as f32, sum_y / count as f32);
        assert!(
            gx < icx - 30.0 && gy < icy - 30.0,
            "sprite tile (0,0) must map to the footprint's top-left ({gx:.1},{gy:.1} vs center {icx:.1},{icy:.1}); the mask is mirrored"
        );
        assert!(
            !(gx > icx && gy > icy),
            "the tile must not land in the bottom-right corner (pre-fix mirror position)"
        );
    }
}
