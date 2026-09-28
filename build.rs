//! Embeds the application icon and Windows version metadata into `pixforge.exe`.
//!
//! Icon + version info live in a Win32 resource section, so without this the
//! binary shows the generic Windows "blank application" icon in Explorer, the
//! taskbar and Alt-Tab, and `wmic datafile get Version` reports 0.0.0.0. The MSI
//! in turn inherits its own icon and version from the binary, so this is what
//! makes an installed build look like a real application.
//!
//! Deliberately *not* embedded here: an application manifest declaring
//! DPI-awareness. winit/egui already put the process into per-monitor-V2 DPI
//! awareness at startup, and a manifest asking for system-DPI awareness would
//! win over that, making the UI blurry on scaled displays. Letting the runtime
//! ask is the correct order.

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=assets/icon.ico");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=Cargo.toml");

    // Resources are a Win32 concept. On other hosts the file is a no-op, which
    // keeps non-Windows builds (and CI on Linux/macOS) working.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }

    let (major, minor, patch) = version_parts();
    // Four 16-bit words: MAJOR << 48 | MINOR << 32 | PATCH << 16 | RELEASE.
    // The final word stays 0; a pre-release is reflected in the string
    // FileVersion below rather than by faking a fourth component.
    let quad = (major << 48) | (minor << 32) | (patch << 16);

    let display = format!("{major}.{minor}.{patch}");

    let mut res = winresource::WindowsResource::new();
    res.set_icon_with_id(icon_path().to_str().expect("utf-8 icon path"), "1")
        .set("FileDescription", "PixForge - Stylized 3D Texture Painter")
        .set("ProductName", "PixForge")
        .set("CompanyName", "PixForge")
        .set(
            "LegalCopyright",
            "Licensed under the GNU GPL v3.0 or later.",
        )
        .set("InternalName", "pixforge")
        .set("OriginalFilename", "pixforge.exe")
        .set(
            "Comments",
            "Paint directly on your 3D mesh with a stylized PBR preview.",
        )
        .set_version_info(winresource::VersionInfo::FILEVERSION, quad)
        .set_version_info(winresource::VersionInfo::PRODUCTVERSION, quad)
        .set_version_info(winresource::VersionInfo::FILEOS, 0x0004_0004) // VOS_NT_WINDOWS32
        .set_version_info(winresource::VersionInfo::FILETYPE, 0x0001) // VFT_APP
        .set_version_info(winresource::VersionInfo::FILESUBTYPE, 0x0000)
        .set("FileVersion", &display);

    if let Err(e) = res.compile() {
        // A missing resource compiler should not fail the whole build: the exe
        // is still perfectly usable, it just loses its icon and version info.
        // Warn, except for a release build on a Windows host, which is what
        // actually ships and so must not pass silently.
        let msg = format!("failed to embed Windows icon/version info: {e}");
        if cfg!(windows) && std::env::var("PROFILE").as_deref() == Ok("release") {
            panic!("{msg} (this must not happen for a release build)");
        }
        println!("cargo:warning={msg}");
    }
}

/// `assets/icon.ico`, located relative to the manifest dir rather than the CWD
/// so the build is independent of where cargo was invoked.
fn icon_path() -> PathBuf {
    PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"))
        .join("assets")
        .join("icon.ico")
}

/// Reads `major.minor.patch` out of CARGO_PKG_VERSION, ignoring any pre-release
/// or build suffix (`1.2.3-rc.1` -> `1.2.3`).
fn version_parts() -> (u64, u64, u64) {
    let v = std::env::var("CARGO_PKG_VERSION").expect("CARGO_PKG_VERSION");
    let core = v.split(['-', '+']).next().unwrap_or(&v);
    let mut it = core.split('.').map(|p| p.parse::<u64>().unwrap_or(0));
    (
        it.next().unwrap_or(0),
        it.next().unwrap_or(0),
        it.next().unwrap_or(0),
    )
}
