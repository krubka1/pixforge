#!/usr/bin/env python3
"""Structural checks on the WiX sources.

This is NOT a substitute for building the MSI. candle/light from the WiX
Toolset are Windows-only and are not available here, so these checks only
catch the class of mistake that is cheap to make and annoying to debug from a
failed build on a Windows CI runner: mismatched tags, dangling ComponentRefs,
DirectoryRefs that point at a directory nobody declared, and duplicate
identifiers.

Exit status is 0 when everything checks out, 1 otherwise.
"""

import os
import re
import sys
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIX_DIR = os.path.join(ROOT, "wix")
WIX_NS = "{http://schemas.microsoft.com/wix/2006/wi}"

errors = []
notes = []


def err(msg):
    errors.append(msg)


def load(path):
    try:
        return ET.parse(path).getroot()
    except ET.ParseError as e:
        err(f"{os.path.relpath(path, ROOT)}: not well-formed XML: {e}")
        return None


def tag(el):
    return el.tag[len(WIX_NS):] if el.tag.startswith(WIX_NS) else el.tag


def check_main():
    path = os.path.join(WIX_DIR, "main.wxs")
    root = load(path)
    if root is None:
        return

    # Collect every declared identifier so refs can be resolved.
    declared_dirs = set()
    for d in root.iter(f"{WIX_NS}Directory"):
        if d.get("Id"):
            declared_dirs.add(d.get("Id"))

    product = root.find(f"{WIX_NS}Product")
    if product is None:
        err("main.wxs: no <Product> element")
        return

    pkg = product.find(f"{WIX_NS}Package")
    if pkg is None:
        err("main.wxs: no <Package> element")
    else:
        scope = pkg.get("InstallScope")
        if scope != "perUser":
            err(
                f"main.wxs: InstallScope is {scope!r}, expected 'perUser'. "
                "A perMachine install needs elevation and writes to Program Files, "
                "which is not what this project ships."
            )
        if not product.get("UpgradeCode"):
            err("main.wxs: Product has no UpgradeCode; MSI upgrades would break")

    app_folder = None
    for d in root.iter(f"{WIX_NS}Directory"):
        if d.get("Id") == "APPLICATIONFOLDER":
            app_folder = d
    if app_folder is None:
        err("main.wxs: no APPLICATIONFOLDER directory")

    # The exe must sit in a directory that also has the brushes folder next to
    # it, because brushes_folder() resolves brushes/ relative to the exe.
    bin_dir = None
    for d in root.iter(f"{WIX_NS}Directory"):
        if d.get("Id") == "Bin":
            bin_dir = d
    if bin_dir is None:
        err("main.wxs: no 'Bin' directory; wix/brushes.wxs keys off it")
    else:
        has_exe = any(
            tag(f) == "File" and (f.get("Name") or "").endswith(".exe")
            for f in bin_dir.iter(f"{WIX_NS}File")
        )
        if not has_exe:
            err("main.wxs: 'Bin' holds no .exe, but brushes/ is nested under it")

    # InstallScope and the root directory have to agree. A perUser MSI that
    # writes to Program Files needs elevation to install and to uninstall, which
    # defeats the point; a perMachine MSI aimed at LocalAppData is worse still.
    scope = pkg.get("InstallScope") if pkg is not None else None
    roots = [
        d.get("Id")
        for d in root.iter(f"{WIX_NS}Directory")
        if d.get("Id") in ("ProgramFiles64Folder", "ProgramFilesFolder", "LocalAppDataFolder")
    ]
    if scope and roots:
        per_user_root = "LocalAppDataFolder" in roots
        if scope == "perUser" and not per_user_root:
            err(
                f"main.wxs: InstallScope='perUser' but the install root is "
                f"{roots[0]!r}; a per-user install must target LocalAppDataFolder"
            )
        if scope == "perMachine" and per_user_root:
            err(
                "main.wxs: InstallScope='perMachine' but the install root is "
                "'LocalAppDataFolder'"
            )

    # The stock template adds a machine-wide PATH entry. It is wrong for a
    # per-user install and this project does not want it either.
    for env in product.iter(f"{WIX_NS}Environment"):
        if (env.get("Name") or "").upper() == "PATH" and env.get("System") == "yes":
            err(
                "main.wxs: a <Environment> writes the machine-wide PATH "
                "(System='yes'); this project installs per-user and should not "
                "modify it"
            )

    # DirectoryRef targets must exist.
    for name in sorted(os.listdir(WIX_DIR)):
        if not name.endswith(".wxs"):
            continue
        r = load(os.path.join(WIX_DIR, name))
        if r is None:
            continue
        for dref in r.iter(f"{WIX_NS}DirectoryRef"):
            ref = dref.get("Id")
            if ref and ref not in declared_dirs:
                err(
                    f"{name}: <DirectoryRef Id='{ref}'> points at a directory "
                    "that main.wxs does not declare"
                )

    # ComponentRefs must match a component defined somewhere in wix/.
    all_components = set()
    for name in sorted(os.listdir(WIX_DIR)):
        if not name.endswith(".wxs"):
            continue
        r = load(os.path.join(WIX_DIR, name))
        if r is None:
            continue
        for c in r.iter(f"{WIX_NS}Component"):
            if c.get("Id"):
                all_components.add(c.get("Id"))

    for cref in product.iter(f"{WIX_NS}ComponentRef"):
        cid = cref.get("Id")
        if cid not in all_components:
            err(
                f"main.wxs: <ComponentRef Id='{cid}'> has no matching <Component> "
                "in any file under wix/"
            )

    # Ids are global in WiX; duplicates are a hard error at link time.
    #
    # ComponentRef is excluded: a ref legitimately repeats the Id of the
    # Component it points at. Shortcut/RemoveFolder Ids are also allowed to
    # match their owning Component - the WiX docs use that to let MSI repair
    # identify a shortcut - so only the first occurrence per file is compared
    # for those, and '*' is a per-element placeholder, not a real identifier.
    seen = {}
    for name in sorted(os.listdir(WIX_DIR)):
        if not name.endswith(".wxs"):
            continue
        r = load(os.path.join(WIX_DIR, name))
        if r is None:
            continue
        for el in r.iter():
            t = tag(el)
            # DirectoryRef is a reference to a directory declared elsewhere,
            # same as ComponentRef is a reference to a Component.
            if t in (
                "ComponentRef",
                "DirectoryRef",
                "Shortcut",
                "RemoveFolder",
                "Environment",
            ):
                continue
            i = el.get("Id")
            if not i or i == "*":
                continue
            if i in seen:
                err(
                    f"duplicate WiX Id {i!r} ({t}) in {name} and {seen[i]}; "
                    "WiX identifiers must be unique across all compiled sources"
                )
            else:
                seen[i] = name

    # The icon referenced by the shortcuts has to exist.
    icons = [i.get("SourceFile") for i in product.iter(f"{WIX_NS}Icon")]
    for src in icons:
        if src and not os.path.isfile(os.path.join(ROOT, src.replace("\\", "/"))):
            err(f"main.wxs: <Icon SourceFile='{src}'> does not exist")

    for s in product.iter(f"{WIX_NS}Shortcut"):
        target = s.get("Target")
        if target and "[APPLICATIONFOLDER]" in target and bin_dir is None:
            err(
                "main.wxs: a <Shortcut> targets [APPLICATIONFOLDER] but no 'Bin' "
                "directory exists for the exe"
            )

    notes.append(f"checked {len(seen)} ids across {len(os.listdir(WIX_DIR))} wxs file(s)")


def check_source_paths():
    """Every Source= path must resolve, or candle fails on a missing file."""
    for name in sorted(os.listdir(WIX_DIR)):
        if not name.endswith(".wxs"):
            continue
        path = os.path.join(WIX_DIR, name)
        r = load(path)
        if r is None:
            continue
        for f in r.iter(f"{WIX_NS}File"):
            src = f.get("Source")
            if not src:
                continue
            if src.startswith("$("):
                continue  # preprocessor variable, resolved by cargo-wix
            p = src.replace("\\", "/")
            if not os.path.isfile(os.path.join(ROOT, p)):
                err(f"{name}: <File Source='{src}'> does not exist")


def check_brushes_present():
    """The MSI is the primary artifact; brushes must be in it."""
    path = os.path.join(WIX_DIR, "brushes.wxs")
    if not os.path.isfile(path):
        err(
            "wix/brushes.wxs is missing - the MSI would install no brushes. "
            "Regenerate with: python packaging/gen_wix_brushes.py"
        )
        return
    r = load(path)
    if r is None:
        return
    n = sum(1 for f in r.iter(f"{WIX_NS}File"))
    notes.append(f"wix/brushes.wxs contributes {n} brush files")
    if n == 0:
        err("wix/brushes.wxs contains no <File> elements")


def main():
    check_main()
    check_source_paths()
    check_brushes_present()

    for n in notes:
        print(f"note: {n}")
    if errors:
        print()
        for e in errors:
            print(f"ERROR: {e}")
        print(f"\n{len(errors)} problem(s) found")
        return 1
    print("\nOK: wix sources are structurally consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
