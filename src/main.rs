#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod brush;
mod brushes;
mod io;
mod paint;
mod project;
mod render;

use eframe::egui;

fn main() -> eframe::Result {
    // Hidden helper: regenerate the shipped brush packs in `./brushes`
    // (or `$PIXFORGE_BRUSHES`/an explicit path) from the procedural
    // generators in `brushes.rs`.
    let mut args = std::env::args();
    let _prog = args.next();
    while let Some(a) = args.next() {
        if a == "--gen-brush-packs" {
            let dir = args.next().unwrap_or_else(|| {
                std::env::var_os("PIXFORGE_BRUSHES")
                    .map(|p| p.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "brushes".to_string())
            });
            return match crate::brushes::generate_ship_pack(std::path::Path::new(&dir)) {
                Ok(files) => {
                    println!("regenerated {} brush-pack files in {dir}:", files.len());
                    for f in &files {
                        println!("  {}", f.display());
                    }
                    Ok(())
                }
                Err(e) => {
                    eprintln!("error: {e}");
                    Err(eframe::Error::AppCreation(e.to_string().into()))
                }
            };
        }
    }

    env_logger::init();

    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: egui::ViewportBuilder::default()
            .with_title("PixForge — Stylized 3D Texture Painter")
            .with_inner_size([1400.0, 900.0])
            .with_min_inner_size([900.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "PixForge",
        options,
        Box::new(|cc| Ok(Box::new(app::PixForgeApp::new(cc)))),
    )
}
