#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod io;
mod paint;
mod project;
mod render;

use eframe::egui;

fn main() -> eframe::Result {
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
