use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use crate::io::{MeshData, TextureData};

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

pub struct Renderer {
    pipeline: wgpu::RenderPipeline,
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
}

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

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(vert_layout)],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: VIEWPORT_FORMAT,
                    blend: Some(wgpu::BlendState::REPLACE),
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
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            cache: None,
            multiview_mask: None,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniform_buffer"),
            size: std::mem::size_of::<[[f32; 4]; 4]>() as u64,
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
        }
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

        match &mesh.texture {
            Some(tex) => self.update_texture(tex),
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

    /// Renders the mesh into the given color/depth texture views.
    pub fn render(
        &self,
        camera: &Camera,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
    ) {
        let data: [[f32; 4]; 4] = camera.view_proj().to_cols_array_2d();
        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&data));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("scene_encoder"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.13,
                            g: 0.14,
                            b: 0.17,
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
    use crate::io::{load_gltf, TextureData};
    use glam::Vec3;

    fn device_and_queue() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            },
        ))
        .expect("no adapter available");
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("pixforge_test"),
                ..Default::default()
            },
        ))
        .expect("no device");
        (device, queue)
    }

    const SIZE: u32 = 512;

    fn render_and_read(device: &wgpu::Device, queue: &wgpu::Queue, mesh: &MeshData) -> Vec<u8> {
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

        let mut camera = Camera::new(1.0);
        if mesh.positions.len() == 3 {
            // Gimbal triangle around the origin.
            camera.fit(Vec3::ZERO, 1.0);
        } else {
            let min = mesh
                .positions
                .iter()
                .fold(Vec3::MAX, |a, b| a.min(*b));
            let max = mesh
                .positions
                .iter()
                .fold(Vec3::MIN, |a, b| a.max(*b));
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
        // Background clear color is (0.13, 0.14, 0.17).
        const BG: [u8; 3] = [33, 35, 43];
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
            positions: vec![Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, 0.0), Vec3::new(0.0, 1.0, 0.0)],
            normals: vec![Vec3::Z, Vec3::Z, Vec3::Z],
            uvs: vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
            indices: vec![0, 1, 2],
            texture: None,
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
            texture: Some(TextureData {
                width: 64,
                height: 64,
                rgba,
            }),
        }
    }

    fn red_dominant(pixels: &[u8]) -> usize {
        pixels
            .chunks_exact(4)
            .filter(|p| p[0] > 100 && p[0] > p[1] && p[0] > p[2])
            .count()
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
                mk(wgpu::TextureFormat::Depth32Float, wgpu::TextureUsages::RENDER_ATTACHMENT),
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
        assert_eq!(
            red_dominant(&before),
            0,
            "no red pixels before painting"
        );

        // Paint a red blob into the middle of the atlas, then push it to the GPU.
        let mut mesh = mesh;
        let hit = mesh_raycast(&mesh, Vec3::new(0.0, 0.0, 2.0), Vec3::new(0.0, 0.0, -1.0))
            .expect("hits the quad");
        let radius = brush_radius_world(&mesh, &hit, 64, 64, 24.0);
        apply_stamp(
            &mut mesh,
            hit.position,
            radius,
            [255, 0, 0, 255],
            1.0,
            1.0,
            StampMode::Paint,
        );
        let tex = mesh.texture.as_ref().unwrap();
        renderer.update_texture(tex);

        let after = render(&renderer, &color);
        let red = red_dominant(&after);
        assert!(
            red > 4_000,
            "painted texels should be visible as red, got {red} red-dominant pixels"
        );
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
            mesh.texture.is_some() && mesh.texture.as_ref().unwrap().rgba.len() >= 4,
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
}
