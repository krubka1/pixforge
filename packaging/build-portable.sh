#!/usr/bin/env bash
# Build a Windows portable build from a non-Windows host.
#
# Produces target/package/PixForge-<version>-portable-windows-x64.zip containing
# the exe (with icon + version info), the brush library and the licence.
#
# This is NOT a substitute for the MSI or the NSIS installer - see RELEASE.md.
# It exists so testers can grab a build without anyone needing the WiX toolset
# or a Windows box. The real installers still have to be built on Windows.
#
# Requirements:
#   rustup target add x86_64-pc-windows-msvc
#   cargo install cargo-xwin          # fetches the MSVC CRT + Windows SDK
#   wine                              # only used to run rcedit's .exe
#   npm install rcedit@3              # in a scratch dir; see RCEDIT below
#
# Usage: packaging/build-portable.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET=x86_64-pc-windows-msvc
PROFILE=release
PKG_DIR="$ROOT/target/package"
STAGE="$ROOT/target/portable-stage"

# Icon and version strings are duplicated from build.rs on purpose: build.rs
# needs rc.exe, which does not exist off Windows, so on a non-Windows host the
# resources are applied afterwards with rcedit instead.
ICON="$ROOT/assets/icon.ico"

cd "$ROOT"

VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)"
if [ -z "$VERSION" ]; then
  echo "error: could not read version from Cargo.toml" >&2
  exit 1
fi

# rcedit is a dev dependency of this script, not of the crate, so it is looked
# up rather than vendored. Point RCEDIT at your own install if you have one.
RCEDIT_ROOT="$ROOT/target/rcedit"
RCEDIT="${RCEDIT:-$RCEDIT_ROOT/node_modules/rcedit/bin/rcedit-x64.exe}"
if [ ! -f "$RCEDIT" ]; then
  echo "==> fetching rcedit (used to embed the icon on a non-Windows host)"
  npm --prefix "$RCEDIT_ROOT" install rcedit@3 --no-audit --no-fund
fi
if [ ! -f "$RCEDIT" ]; then
  echo "error: rcedit not found at $RCEDIT" >&2
  exit 1
fi

# cross-spawn-windows-exe looks for wine64; Debian's wine package ships `wine`
# only. Provide the name it expects without needing to install anything.
if ! command -v wine64 >/dev/null 2>&1; then
  if command -v wine >/dev/null 2>&1; then
    SHIM_DIR="$ROOT/target/rcedit/shim"
    mkdir -p "$SHIM_DIR"
    ln -sf "$(command -v wine)" "$SHIM_DIR/wine64"
    export PATH="$SHIM_DIR:$PATH"
  else
    echo "error: wine is required to run rcedit on this host" >&2
    exit 1
  fi
fi

echo "==> building the $TARGET exe"
# +crt-static is not optional here. Without it the exe imports VCRUNTIME140.dll
# and the api-ms-win-crt-* forwarders, and a machine that does not already have
# the VC++ Redistributable installed fails to launch it with "VCRUNTIME140.dll is
# missing". The portable build is the one artifact testers get without running
# an installer, so it cannot assume the redistributable is present. dist sets
# the same flag for the MSI (msvc-crt-static in dist-workspace.toml).
RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=+crt-static" \
  cargo xwin build --profile "$PROFILE" --target "$TARGET"

EXE="$ROOT/target/$TARGET/$PROFILE/pixforge.exe"
[ -f "$EXE" ] || { echo "error: $EXE was not produced" >&2; exit 1; }

echo "==> embedding icon and version info"
rm -rf "$STAGE"
mkdir -p "$STAGE"
cp "$EXE" "$STAGE/pixforge.exe"

WINEDEBUG=-all wine "$RCEDIT" "$STAGE/pixforge.exe" \
  --set-icon "$ICON" \
  --set-file-version "$VERSION" \
  --set-product-version "$VERSION" \
  --set-version-string FileDescription "PixForge - Stylized 3D Texture Painter" \
  --set-version-string ProductName "PixForge" \
  --set-version-string CompanyName "PixForge" \
  --set-version-string LegalCopyright "Licensed under the GNU GPL v3.0 or later." \
  --set-version-string OriginalFilename "pixforge.exe" \
  --set-version-string InternalName "pixforge"

echo "==> staging payload"
# brushes/ must sit next to the exe: brushes_folder() in src/app.rs resolves it
# relative to the executable, not the working directory.
cp -r "$ROOT/brushes" "$STAGE/brushes"
cp "$ROOT/LICENSE" "$STAGE/LICENSE.txt"
cp "$ROOT/packaging/PORTABLE-README.txt" "$STAGE/README.txt"

echo "==> zipping"
mkdir -p "$PKG_DIR"
OUT="$PKG_DIR/PixForge-$VERSION-portable-windows-x64.zip"
rm -f "$OUT"
( cd "$STAGE" && zip -qr9 "$OUT" . )

echo
echo "wrote $OUT"
echo "Note: no installer, no Start Menu or Desktop shortcut, no uninstaller."
echo "Give it to testers as-is; build the MSI/NSIS installer on Windows for release."
