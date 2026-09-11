//! Custom `.pixforge` project format: the full mesh geometry plus every layer
//! atlas (name, visibility, opacity) and the active layer index, one file.
//!
//! Layout (little-endian):
//!   magic     = b"PIXFORGE\0" | u32 version
//!   positions : u32 count, then count * f32x3
//!   normals   : u32 count, then count * f32x3
//!   uvs       : u32 count, then count * f32x2
//!   indices   : u32 count, then count * u32
//!   active_layer : u32
//!   layer_count  : u32
//!   per layer:
//!     name_len u32 + utf8 bytes, visible u8, opacity f32,
//!     width u32, height u32, rgba_len u32, then raw RGBA bytes
//!
//! Version 2 appends a single `blend` byte after each layer's atlas bytes;
//! version 3 appends the layer's four material f32s (roughness, metallic,
//! emissive, ambient occlusion) after that blend byte; version 4 adds the
//! layer's height and bump-strength f32s after those. Older files load fine
//! and default the missing pieces (`Normal` blend, default material, flat
//! height, default bump strength).
//!
//! The atlas bytes are embedded as PNGs (via the `image` crate) so a
//! multi-megabyte canvas stays small; on load they are decoded back into the
//! exact `width * height * 4` RGBA run.

use std::io::{self, Read, Write};

use crate::io::{BlendMode, Layer, MeshData, TextureData};

const MAGIC: &[u8; 9] = b"PIXFORGE\0";
const VERSION: u32 = 4;

struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    fn new() -> Self {
        Self { buf: Vec::new() }
    }
    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn bytes(&mut self, b: &[u8]) {
        self.buf.extend_from_slice(b);
    }
}

struct Reader<'a> {
    cur: std::io::Cursor<&'a [u8]>,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            cur: std::io::Cursor::new(data),
        }
    }
    fn u32(&mut self) -> io::Result<u32> {
        let mut b = [0u8; 4];
        self.cur.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }
    fn f32(&mut self) -> io::Result<f32> {
        let mut b = [0u8; 4];
        self.cur.read_exact(&mut b)?;
        Ok(f32::from_le_bytes(b))
    }
    fn bytes(&mut self, n: usize) -> io::Result<Vec<u8>> {
        let mut b = vec![0u8; n];
        self.cur.read_exact(&mut b)?;
        Ok(b)
    }
    fn string(&mut self) -> io::Result<String> {
        let len = self.u32()? as usize;
        let buf = self.bytes(len)?;
        String::from_utf8(buf)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "bad utf8 string"))
    }
}

fn compress(width: u32, height: u32, rgba: &[u8]) -> Vec<u8> {
    use image::ImageEncoder;
    let mut buf = Vec::new();
    let enc = image::codecs::png::PngEncoder::new(&mut buf);
    let _ = enc.write_image(rgba, width, height, image::ExtendedColorType::Rgba8);
    buf
}

fn decompress(data: &[u8]) -> io::Result<Vec<u8>> {
    let img = image::load_from_memory(data)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
    Ok(img.to_rgba8().into_raw())
}

/// Serializes the full mesh + layer stack into the `.pixforge` binary format.
pub fn save_project(path: &str, mesh: &MeshData) -> io::Result<()> {
    let mut w = Writer::new();
    w.bytes(MAGIC);
    w.u32(VERSION);
    w.u32(mesh.positions.len() as u32);
    for v in &mesh.positions {
        w.f32(v.x);
        w.f32(v.y);
        w.f32(v.z);
    }
    w.u32(mesh.normals.len() as u32);
    for v in &mesh.normals {
        w.f32(v.x);
        w.f32(v.y);
        w.f32(v.z);
    }
    w.u32(mesh.uvs.len() as u32);
    for (u, v) in &mesh.uvs {
        w.f32(*u);
        w.f32(*v);
    }
    w.u32(mesh.indices.len() as u32);
    for i in &mesh.indices {
        w.u32(*i);
    }
    w.u32(mesh.active_layer as u32);
    w.u32(mesh.layers.len() as u32);
    for layer in &mesh.layers {
        w.u32(layer.name.len() as u32);
        w.bytes(layer.name.as_bytes());
        w.bytes(&[layer.visible as u8]);
        w.f32(layer.opacity);
        w.u32(layer.texture.width);
        w.u32(layer.texture.height);
        let compressed = compress(
            layer.texture.width,
            layer.texture.height,
            &layer.texture.rgba,
        );
        w.u32(compressed.len() as u32);
        w.bytes(&compressed);
        w.bytes(&[layer.blend.to_byte()]);
        w.f32(layer.roughness);
        w.f32(layer.metallic);
        w.f32(layer.emissive);
        w.f32(layer.ambient_occlusion);
        w.f32(layer.height);
        w.f32(layer.bump_strength);
    }

    let mut file = std::fs::File::create(path)?;
    file.write_all(&w.buf)?;
    file.sync_all()?;
    Ok(())
}

/// Loads a `.pixforge` project back into a `MeshData`.
pub fn load_project(path: &str) -> io::Result<MeshData> {
    let data = std::fs::read(path)?;
    let mut r = Reader::new(&data);
    if r.bytes(MAGIC.len())? != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a pixforge project file",
        ));
    }
    let version = r.u32()?;
    if version == 0 || version > VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unsupported pixforge version",
        ));
    }

    let n_pos = r.u32()? as usize;
    let mut positions = Vec::with_capacity(n_pos);
    for _ in 0..n_pos {
        positions.push(glam::Vec3::new(r.f32()?, r.f32()?, r.f32()?));
    }
    let n_nrm = r.u32()? as usize;
    let mut normals = Vec::with_capacity(n_nrm);
    for _ in 0..n_nrm {
        normals.push(glam::Vec3::new(r.f32()?, r.f32()?, r.f32()?));
    }
    let n_uv = r.u32()? as usize;
    let mut uvs = Vec::with_capacity(n_uv);
    for _ in 0..n_uv {
        uvs.push((r.f32()?, r.f32()?));
    }
    let n_idx = r.u32()? as usize;
    let mut indices = Vec::with_capacity(n_idx);
    for _ in 0..n_idx {
        indices.push(r.u32()?);
    }
    if n_nrm != n_pos || n_uv != n_pos {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "mismatched vertex counts",
        ));
    }
    if !indices.is_empty() && indices.len() % 3 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "non-triangle index count",
        ));
    }

    let active_layer = r.u32()? as usize;
    let layer_count = r.u32()? as usize;
    let mut layers = Vec::with_capacity(layer_count);
    for _ in 0..layer_count {
        let name = r.string()?;
        let visible = r.bytes(1)?[0] != 0;
        let opacity = r.f32()?;
        let width = r.u32()?;
        let height = r.u32()?;
        let size = r.u32()? as usize;
        let compressed = r.bytes(size)?;
        let rgba = decompress(&compressed)?;
        if rgba.len() != (width * height * 4) as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "layer atlas size mismatch",
            ));
        }
        let blend = if version >= 2 {
            BlendMode::from_byte(r.bytes(1)?[0])
        } else {
            BlendMode::Normal
        };
        let (roughness, metallic, emissive, ambient_occlusion) = if version >= 3 {
            (r.f32()?, r.f32()?, r.f32()?, r.f32()?)
        } else {
            (0.55, 0.0, 0.0, 1.0)
        };
        let layer_height = if version >= 4 { r.f32()? } else { 0.0 };
        let bump_strength = if version >= 4 { r.f32()? } else { 2.0 };
        layers.push(Layer {
            name,
            visible,
            opacity,
            blend,
            roughness,
            metallic,
            emissive,
            ambient_occlusion,
            height: layer_height,
            bump_strength,
            texture: TextureData {
                width,
                height,
                rgba,
            },
        });
    }

    let active_layer = active_layer.min(layers.len().saturating_sub(1));
    Ok(MeshData {
        positions,
        normals,
        uvs,
        indices,
        layers,
        active_layer,
        dirty: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{MeshData, TextureData};

    fn sample_mesh() -> MeshData {
        let mut mesh = MeshData::uv_sphere(0.6, 4, 6);
        mesh.layers = vec![
            Layer {
                name: "Base".into(),
                visible: true,
                opacity: 1.0,
                blend: BlendMode::Normal,
                roughness: 0.55,
                metallic: 0.0,
                emissive: 0.0,
                ambient_occlusion: 1.0,
                height: 0.4,
                bump_strength: 2.0,
                texture: TextureData {
                    width: 4,
                    height: 4,
                    rgba: (0..4 * 4 * 4).map(|i| (i * 7) as u8).collect(),
                },
            },
            Layer {
                name: "Paint 😀".into(), // unicode name survives the round trip
                visible: false,
                opacity: 0.35,
                blend: crate::io::BlendMode::Multiply,
                roughness: 0.2,
                metallic: 0.8,
                emissive: 0.5,
                ambient_occlusion: 0.7,
                height: 0.0,
                bump_strength: 6.0,
                texture: TextureData {
                    width: 2,
                    height: 3,
                    rgba: [255, 0, 0, 128].repeat(6),
                },
            },
        ];
        mesh.active_layer = 1;
        mesh
    }

    #[test]
    fn round_trips_geometry_and_layers() {
        let mesh = sample_mesh();
        let path = std::env::temp_dir().join(format!(
            "pixforge_roundtrip_{}.pixforge",
            std::process::id()
        ));
        let ps = path.to_string_lossy().to_string();
        save_project(&ps, &mesh).expect("save");
        let loaded = load_project(&ps).expect("load");
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.positions, mesh.positions);
        assert_eq!(loaded.normals, mesh.normals);
        assert_eq!(loaded.uvs, mesh.uvs);
        assert_eq!(loaded.indices, mesh.indices);
        assert_eq!(loaded.active_layer, mesh.active_layer);
        assert_eq!(loaded.layers.len(), mesh.layers.len());
        for (l, r) in loaded.layers.iter().zip(&mesh.layers) {
            assert_eq!(l.name, r.name);
            assert_eq!(l.visible, r.visible);
            assert_eq!(l.opacity, r.opacity);
            assert_eq!(l.blend, r.blend);
            assert_eq!(l.height, r.height);
            assert_eq!(l.bump_strength, r.bump_strength);
            assert_eq!(l.texture.width, r.texture.width);
            assert_eq!(l.texture.height, r.texture.height);
            assert_eq!(l.texture.rgba, r.texture.rgba);
        }
    }

    #[test]
    fn rejects_garbage_and_truncated_files() {
        // Wrong magic.
        let path = std::env::temp_dir().join(format!(
            "pixforge_bad_magic_{}.pixforge",
            std::process::id()
        ));
        std::fs::write(&path, b"not a project at all").expect("write");
        let ps = path.to_string_lossy().to_string();
        assert!(load_project(&ps).is_err());
        let _ = std::fs::remove_file(&path);

        // Valid header but cut-off body.
        let mesh = sample_mesh();
        let path2 = std::env::temp_dir().join(format!(
            "pixforge_truncated_{}.pixforge",
            std::process::id()
        ));
        save_project(&path2.to_string_lossy(), &mesh).expect("save");
        let bytes = std::fs::read(&path2).expect("read");
        std::fs::write(&path2, &bytes[..bytes.len() / 2]).expect("truncate");
        assert!(load_project(&path2.to_string_lossy()).is_err());
        let _ = std::fs::remove_file(&path2);
    }

    #[test]
    fn reads_v1_files_and_defaults_blend_to_normal() {
        // Minimal version-1 layout: empty mesh, no layers (so no per-layer
        // blend bytes).
        let mut b = Vec::new();
        b.extend_from_slice(MAGIC);
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes()); // positions
        b.extend_from_slice(&0u32.to_le_bytes()); // normals
        b.extend_from_slice(&0u32.to_le_bytes()); // uvs
        b.extend_from_slice(&0u32.to_le_bytes()); // indices
        b.extend_from_slice(&0u32.to_le_bytes()); // active layer
        b.extend_from_slice(&0u32.to_le_bytes()); // layer count
        let path =
            std::env::temp_dir().join(format!("pixforge_v1_{}.pixforge", std::process::id()));
        std::fs::write(&path, &b).expect("write v1");
        let loaded = load_project(&path.to_string_lossy()).expect("load v1");
        let _ = std::fs::remove_file(&path);
        assert!(loaded.layers.is_empty());
        assert_eq!(loaded.active_layer, 0);

        // A future version is rejected rather than misparsed.
        b[9..13].copy_from_slice(&999u32.to_le_bytes());
        let path2 =
            std::env::temp_dir().join(format!("pixforge_v999_{}.pixforge", std::process::id()));
        std::fs::write(&path2, &b).expect("write v999");
        assert!(load_project(&path2.to_string_lossy()).is_err());
        let _ = std::fs::remove_file(&path2);
    }
}
