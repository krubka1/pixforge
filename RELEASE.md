# Building a Windows release

The MSI is the release artifact. It is built by [`dist`](https://opensource.axo.dev/cargo-dist/)
(via GitHub Actions on a Windows runner, or locally on Windows), and it ships
`pixforge.exe` plus the `brushes/` library beside it.

| | MSI (release) | Portable ZIP (tester drop) |
| --- | --- | --- |
| Built by | `dist` on `windows-2022` | `packaging/build-portable.sh` on Linux/macOS |
| Output | GitHub Release asset | `target/package/PixForge-<version>-portable-windows-x64.zip` |
| Needs Windows | yes (a runner counts) | no |
| Shortcuts | Start Menu + Desktop | none |
| Uninstall | Add/Remove Programs | delete the folder |

There is also an NSIS `setup.exe` script, `packaging/pixforge.nsi`. dist does not
produce NSIS output, so that script is maintained separately and needs NSIS
installed. See [NSIS](#nsis-setup-exe-optional) below.

## What gets installed

```
%LOCALAPPDATA%\Programs\PixForge\
├── bin\
│   ├── pixforge.exe      # icon + version info embedded
│   └── brushes\          # stock brush library
│       ├── Grain\…
│       ├── Material\…
│       ├── Spiky\…
│       ├── Spot\…
│       └── Stroke\…
└── LICENSE.txt
```

Plus Start Menu and Desktop shortcuts, both pointing at `bin\pixforge.exe`.

Note the `bin\` level. dist's stock WiX template installs the exe to `bin\` and
this repo keeps that, which means `brushes/` has to be `bin\brushes\` — see
[Why brushes sit next to the exe](#why-brushes-sit-next-to-the-exe).

The install root is `LocalAppDataFolder`, not `Program Files`, and that has to
stay in step with `InstallScope='perUser'`. A per-user MSI writing to Program
Files needs elevation to install and again to uninstall, which is the opposite
of the point. The stock template pairs `perMachine` with `ProgramFiles64Folder`
for that reason.

## Releasing

Push a tag. The generated workflow builds the MSI, signs it if configured, and
attaches it to a GitHub Release.

```sh
# bump version in Cargo.toml first
cargo test && python3 packaging/gen_wix_brushes.py && python3 packaging/check_wix.py
git commit -am "release 0.1.0"
git tag v0.1.0 && git push origin main --tags
```

Watch the `release` workflow in the Actions tab. Failures are almost always a
WiX compile error from `wix/`; run `packaging/check_wix.py` locally first, it
catches the common cases without needing Windows.

To build without releasing:

```sh
dist build --artifacts=all
```

This still needs the WiX toolset, so from Linux/macOS it only produces the
archive. `dist plan` works anywhere and shows what a release would produce.

## Why brushes sit next to the exe

`brushes_folder()` in `src/app.rs` resolves the library relative to
`current_exe()`, because the CWD is `C:\Windows\System32` when the app is
launched from a Start Menu shortcut. A CWD-relative lookup would come up empty in
an installed build.

So the install layout has to be `bin\pixforge.exe` + `bin\brushes\`. The
`<DirectoryRef Id='Bin'>` in `wix/brushes.wxs` and the `Bin` directory in
`wix/main.wxs` are what enforce that. If you move the exe in `main.wxs`, move
the brushes with it.

User state (UI layout, custom palettes) is written to `%APPDATA%\pixforge` and is
left alone on uninstall. Override the brush location with `PIXFORGE_BRUSHES` to
run side-by-side copies.

## dist configuration

`dist-workspace.toml` holds the settings that matter:

- `targets` — Windows only. `dist init` also offered the macOS and Linux
  triples, but nothing has ever been built or tested on them, and the MSI needs
  a Windows runner regardless. Add a triple when there is a runner for it.
- `msvc-crt-static = true` — pinned explicitly even though it is dist's default.
  Without it the exe imports `VCRUNTIME140.dll` and the `api-ms-win-crt-*`
  forwarders, so it only runs on machines that already have the VC++
  Redistributable. It is a one-word change to a flag nobody would notice
  removing, and the failure shows up as "VCRUNTIME140.dll is missing" on someone
  else's machine.
- `allow-dirty = ["msi"]` — `wix/main.wxs` is hand-maintained, see below.
- `include` — the `brushes/` folder and `RELEASE.md` in the **archive** only.

### `include` does not reach the MSI

dist's `include` feeds the standalone archives. The MSI is rendered separately
from `wix/main.wxs`, which references files one at a time, so brushes have to be
listed there explicitly. `wix/brushes.wxs` does that.

### `wix/main.wxs` is hand-maintained

dist regenerates `wix/main.wxs` from a stock template on `dist init` and
`dist generate`. The stock template is wrong for this project in four ways:
`InstallScope='perMachine'` with a `ProgramFiles64Folder` root (needs
elevation), a **machine-wide PATH modification**, no `LICENSE.txt`, and no
brushes. So the file is edited by hand and `allow-dirty = ["msi"]` stops dist
from treating that as drift and clobbering it.

If you ever run `dist init` or `dist generate` and see `wix/main.wxs` replaced by
the stock template, restore it from git. The CI workflow does not run
`dist generate`, so this only bites locally.

The parts that must not drift:

- `InstallScope='perUser'` and the `LocalAppDataFolder` root together
- the `LICENSE.txt` component
- `brushes/` under `Bin` (in `wix/brushes.wxs`, generated)
- the Start Menu / Desktop shortcut components
- the `UpgradeCode`

## Regenerating the brush WiX

`wix/brushes.wxs` is generated and committed, so the MSI builds without Python
installed. After adding, renaming or removing anything under `brushes/`:

```sh
python3 packaging/gen_wix_brushes.py
```

WiX has no recursive-copy primitive, so every brush is its own `<File>`; the
generator avoids hand-maintaining 53 of them. It also splices the matching
`<ComponentRef>` entries into `wix/main.wxs` between the generated markers, so
the two files cannot drift apart.

`packaging/check_wix.py` validates the result without Windows: XML well-formedness,
dangling `ComponentRef`/`DirectoryRef` targets, duplicate WiX ids, unresolved
`Source=` paths, a `perUser` scope that disagrees with the install root, a
reintroduced machine-wide PATH write, and that the brush library is actually
present in the MSI. It cannot replace a real build.

## Portable build (no Windows needed, for testers)

Neither installer can be built off Windows, but testers can still get a
properly-branded build from Linux or macOS. `packaging/build-portable.sh` does
the whole thing:

```sh
rustup target add x86_64-pc-windows-msvc
cargo install cargo-xwin
./packaging/build-portable.sh
```

Output: `target/package/PixForge-<version>-portable-windows-x64.zip` — the exe
plus `brushes/`, the licence and a readme for testers.

It cross-compiles the exe *without* resources, then applies the icon and version
info afterwards with [`rcedit`](https://github.com/electron/rcedit) under Wine,
working around the missing `rc.exe`. Needs `wine` (Debian ships `wine` but not
`wine64`; the script shims the name) and `npm` for the one-off `rcedit` install.
Wine emits a wall of `ndis.sys`/`MESA` noise on the way — the useful signal is
the final `wrote …zip` line.

Two things the script gets right that are easy to get wrong:

- **`+crt-static`.** Without it the portable exe imports `VCRUNTIME140.dll` and
  only runs where the VC++ Redistributable happens to be installed. The portable
  build is the one artifact testers get *without* running an installer, so it
  cannot assume that. The script sets the flag and dist sets the equivalent for
  the MSI.
- **`brushes/` next to the exe**, not at the zip root.

Verify a build rather than trusting it — both failure modes are silent:

```sh
# 1. resources landed
python3 - <<'EOF'
import zipfile, struct
d = zipfile.ZipFile('target/package/PixForge-0.1.0-portable-windows-x64.zip').read('pixforge.exe')
pe = struct.unpack_from('<I', d, 0x3c)[0]
off = pe + 24 + struct.unpack_from('<H', d, pe + 20)[0]
for _ in range(struct.unpack_from('<H', d, pe + 6)[0]):
    nm = d[off:off+8].rstrip(b'\0').decode()
    _, _, rs, ra = struct.unpack_from('<IIII', d, off + 8)
    if nm == '.rsrc':
        print('.rsrc', rs, 'bytes; ProductName present:',
              'PixForge'.encode('utf-16-le') in d[ra:ra+rs])
    off += 40
EOF

# 2. no CRT DLL imports (expects no output)
cd /tmp && rm -rf pv && mkdir pv && cd pv
unzip -q "$OLDPWD/target/package/PixForge-0.1.0-portable-windows-x64.zip"
objdump -p pixforge.exe | grep -i 'DLL Name' | grep -Ei 'vcruntime|api-ms-win-crt|msvcp'
```

What testers give up versus a real install: no shortcuts, no Add/Remove Programs
entry, no uninstaller (delete the folder), and SmartScreen warns on first run
because the binary is unsigned. Fine for a tester drop, not for a public release.

## Prerequisites

- Rust stable with the `x86_64-pc-windows-msvc` target.
- For the MSI, on whichever machine builds it: the
  [WiX Toolset](https://wixtoolset.org/) v3.14.1+ (`candle.exe` and `light.exe` on
  `PATH`). WiX v4/v5 also works, but cargo-wix defaults to the v3 schema this
  repo's WiX sources are written against.
- For NSIS: NSIS 3.x.

```sh
rustup target add x86_64-pc-windows-msvc
cargo install dist cargo-xwin
```

GitHub Actions installs `dist` itself; you only need it locally to iterate.

## MSI internals

The WXS is committed — no `cargo wix init`, and never `dist init` over it (see
above). To build by hand on Windows:

```sh
dist build --artifacts=msi
```

## NSIS `setup.exe` (optional)

dist does not emit NSIS, so this is a separate path. Stage the payload, then
compile:

```sh
cargo build --release --target x86_64-pc-windows-msvc

rm -rf staging && mkdir -p staging
cp target/x86_64-pc-windows-msvc/release/pixforge.exe staging/
cp -r brushes staging/

makensis -DAPP_VERSION=0.1.0 -DAPP_PUBLISHER=krubka1 packaging/pixforge.nsi
```

Output: `target/package/PixForge-<version>-setup.exe`. It installs per-user
without elevation via `RequestExecutionLevel user`; change that to `admin` at the
top of the script for a machine-wide install, which also switches the registry
hive for the Add/Remove Programs entry.

Note this path does **not** link the CRT statically unless you pass
`-C target-feature=+crt-static` yourself, and it has not been compiled in this
environment — `makensis` needs MinGW, which is not available here. Treat it as
unverified.

## Cross-compiling from Linux or macOS

The exe cross-compiles, but **the icon and version info are Windows-host only**,
because that step needs `rc.exe`. On a non-Windows host the build prints:

```
warning: failed to embed Windows icon/version info: No such file or directory
```

and produces a working but undecorated binary. `build.rs` degrades to a warning
rather than failing so cross-compiles still work; `build-portable.sh` then
backfills the resources with `rcedit`.

With [`cargo-xwin`](https://github.com/rust-cross/cargo-xwin) you get the MSVC
CRT and Windows SDK without a Windows machine:

```sh
cargo install cargo-xwin
rustup target add x86_64-pc-windows-msvc
cargo xwin build --release --target x86_64-pc-windows-msvc
```

WiX is Windows-only regardless, so the MSI cannot be built off Windows — hence
the CI runner.

## Version numbers

The version comes from `version` in `Cargo.toml`, read by all of:

- `build.rs` → the exe's `FILEVERSION` / `PRODUCTVERSION`, shown in Properties
  and `wmic datafile`.
- dist → the MSI's `Product/@Version` and the archive filenames. MSI stores only
  three numeric components, so `0.1.0-rc.1` is truncated to `0.1.0`; prereleases
  are not really expressible in an MSI version.
- `makensis -DAPP_VERSION=…` → the NSIS filename and version info.

## GUIDs that must stay stable

If these move, upgrades break rather than replacing the old install — MSI treats
the new component as unrelated and installs a second copy alongside the old one.

- `upgrade-guid` in `[package.metadata.wix]` → the MSI's `UpgradeCode`, pinned at
  `0B9863A7-557A-430E-8AD6-ED76EE3598E0`. cargo-wix otherwise rolls a fresh
  random GUID per run and every build becomes a separate product.
- `Guid='*'` in the WiX sources → WiX derives each autogenerated GUID from the
  **directory and filename of that component's keypath**. Adding files to a
  component is fine. Renaming a keypath file, moving its folder, or moving
  `KeyPath='yes'` to a different file all change the GUID.
- The brush category GUIDs in `wix/brushes.wxs` are explicit UUIDv5 values
  derived from the category name, so they are independent of which brushes
  exist. That is deliberate: WiX's `Guid='*'` would key off the first file in
  each category, and a new brush sorting earlier would shift it.

## Icon

`assets/icon.ico` is generated by `assets/make_icon.py` (Pillow). It holds
16/24/32/48/64/128/256 px entries, which is what Explorer, the taskbar, Alt-Tab
and Add/Remove Programs each want.

```sh
python3 assets/make_icon.py   # rewrites assets/icon.ico and assets/icon.png
```

Consumers, all of which need regenerating if it changes:

- `build.rs` → the exe's icon and version info.
- `wix/main.wxs` → `<Icon Id='ProductICO'>` and both MSI shortcuts.
- `packaging/pixforge.nsi` → `MUI_ICON` / `MUI_UNICON`.

## Signing

Unsigned installers trigger SmartScreen, and SmartScreen reputation is per-file,
so a new build of the same app warns again until users click through. Worth
doing before any public release.

dist has first-class support for [SSL.com
signing](https://opensource.axo.dev/cargo-dist/book/signings/ssl-com/) and
[Azure Trusted Signing](https://opensource.axo.dev/cargo-dist/book/signings/),
configured in `dist-workspace.toml`. With either set, CI signs the MSI
automatically on release.

To sign by hand, `cargo wix sign` (needs `signtool` from the Windows SDK), or
`signtool sign /fd SHA256 …` on the built MSI.

## Checklist before publishing

1. `cargo build --release` — clean, no warnings.
2. `cargo test`.
3. `python3 packaging/gen_wix_brushes.py` — no diff expected if brushes unchanged.
4. `python3 packaging/check_wix.py` — structural checks on the WiX sources.
5. `git push origin main --tags` after tagging; watch the `release` workflow.
6. Install the MSI on a clean Windows VM.
7. Launch it **from the Start Menu shortcut**, not by double-clicking the exe.
   This is the case that exposes path bugs: the CWD will be `System32`.
8. Check the brush categories load (Texture Editor → Brush Library).
9. Check the UI layout and custom palettes survive a restart, then a reinstall.
10. Confirm the icon shows in Explorer, the taskbar and Add/Remove Programs.
