use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use crate::io::{MeshData, TextureData};

/// Physically-based material parameters for the metallic-roughness shading in
/// the viewport. Painted maps come later; for now the values are global (and
/// still stylizable for low-poly looks — nothing here forces realism).
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
    /// How strongly the analytic sky lights the surface.
    pub env_intensity: f32,
    /// Exposure multiplier applied before tone mapping.
    pub exposure: f32,
    /// Camera-direction fill light strength (keeps shadow interiors readable).
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
            env_intensity: 0.7,
            exposure: 1.0,
            fill_intensity: 0.5,
        }
    }
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
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    default_bind_group: wgpu::BindGroup,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    texture: Option<wgpu::Texture>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    uv_overlay: u32,
    material: Material,
}

/// Uniform buffer contents: the view-projection matrix (64 bytes), the 32-bit
/// pass mode (opaque = 0, translucent = 1), the 32-bit UV debug overlay
/// (bit 0 = checkerboard, bit 1 = UV grid), then the PBR uniform vec4s
/// (material, sun, sun color, environment, camera position).
const UNIFORM_BYTES: u64 = 160;
const UNIFORM_FLOATS: usize = 40;
const PASS_MODE_OFFSET: u64 = 64;
const PASS_OPAQUE: u32 = 0;
const PASS_TRANSLUCENT: u32 = 1;
const UV_OVERLAY_OFFSET: u64 = 68;
const MATERIAL_OFFSET: u64 = 80;
const SUN_OFFSET: u64 = 96;
const SUN_COLOR_OFFSET: u64 = 112;
const ENV_OFFSET: u64 = 128;
const CAMERA_OFFSET: u64 = 144;

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

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniform_buffer"),
            size: UNIFORM_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

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

        let default_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh_bind_group"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &white.create_view(&Default::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let (vertex_buffer, index_buffer, index_count) = empty_buffers(&device);

        Self {
            pipeline,
            translucent_pipeline,
            vertex_buffer,
            index_buffer,
            index_count,
            uniform_buffer,
            bind_group: default_bind_group.clone(),
            default_bind_group,
            bind_group_layout: bgl,
            sampler,
            texture: None,
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
                self.bind_group = self.default_bind_group.clone();
                self.texture = None;
            }
        }
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
            let view = texture.create_view(&Default::default());
            self.bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mesh_bind_group_tex"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
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
        let mut data = [0.0f32; UNIFORM_FLOATS];
        for (dst, row) in data.iter_mut().zip(vp.iter().flat_map(|r| r.iter())) {
            *dst = *row;
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
        let sun_vec: [f32; 4] = [0.5, 0.7, 0.8, m.sun_intensity];
        let sun_color_vec: [f32; 4] = [m.sun_color[0], m.sun_color[1], m.sun_color[2], 0.0];
        let env_vec: [f32; 4] = [m.env_intensity, m.exposure, m.fill_intensity, 0.0];
        let camera_vec: [f32; 4] = [camera.eye.x, camera.eye.y, camera.eye.z, 0.0];
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
        renderer.set_mesh(mesh);
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
        pixels
            .chunks_exact(4)
            .filter(|p| p[0] > 100 && p[0] > p[1] && p[0] > p[2])
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
        // Backdrop clear color (0.50, 0.52, 0.55) rendered as Rgba8Unorm.
        let backdrop = px
            .chunks_exact(4)
            .filter(|p| {
                (116..=140).contains(&p[0])
                    && (120..=145).contains(&p[1])
                    && (125..=155).contains(&p[2])
                    && p[3] == 255
            })
            .count();
        assert!(
            backdrop > 4000,
            "fully transparent texels must be discarded, leaving the backdrop, got {backdrop}"
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
            .filter(|p| (140..=190).contains(&p[0]) && p[1] < 90 && p[2] < 90 && p[3] == 255)
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
        // Backdrop clear color (0.50, 0.52, 0.55) surrounds the sphere.
        let backdrop = px
            .chunks_exact(4)
            .filter(|p| {
                (116..=140).contains(&p[0])
                    && (120..=145).contains(&p[1])
                    && (125..=155).contains(&p[2])
                    && p[3] == 255
            })
            .count();
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
}
