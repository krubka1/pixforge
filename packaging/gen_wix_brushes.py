#!/usr/bin/env python3
"""Generate wix/brushes.wxs from the contents of brushes/.

The MSI has to list every shipped brush file individually: WiX has no recursive
"copy this folder" primitive the way NSIS' `File /r` does, and neither dist nor
cargo-wix has a `resources` setting - dist's `include` only reaches the archives,
not the installer. Rather than hand-maintain 50-odd <File> entries, this walks
the folder and emits the <Fragment>.

cargo-wix compiles every .wxs in wix/, so this file is picked up alongside
main.wxs without being referenced from it. The <DirectoryRef> target must match
the directory in main.wxs that holds pixforge.exe, since brushes_folder() in
src/app.rs resolves brushes/ relative to the executable.

Re-run after adding, renaming or removing any brush:

    python packaging/gen_wix_brushes.py

The output is committed so the MSI can be built without Python installed.
"""


import os
import re
import sys
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRUSHES = os.path.join(ROOT, "brushes")
OUT = os.path.join(ROOT, "wix", "brushes.wxs")
MAIN_WXS = os.path.join(ROOT, "wix", "main.wxs")

# Fixed project namespace, so the category GUIDs below never move. Changing this
# value would give every component a new identity and make the MSI install a
# second copy of the brush library instead of upgrading the existing one.
NAMESPACE = uuid.UUID("6f2a9d4c-0b1e-4a3f-9c7d-1e8b5a2f0c31")


def category_guid(cat):
    """A stable GUID per category.

    WiX's Guid='*' is derived from the keypath file's directory and name, so
    adding a brush that sorts ahead of the current first file in a category
    would change the component's identity and make MSI install a second copy
    rather than upgrade. Deriving the GUID from the category name instead makes
    it independent of which brushes exist.
    """
    return str(uuid.uuid5(NAMESPACE, f"pixforge/brushes/{cat}")).upper()


def wixtid(name):
    """WiX identifiers must be unique; a stable hash suffix avoids collisions
    between same-named files in different category folders and stays identical
    across runs so a regenerated file produces a byte-identical diff."""
    slug = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not slug or not (slug[0].isalpha() or slug[0] == "_"):
        slug = "f" + slug
    return slug[:48]


def main():
    if not os.path.isdir(BRUSHES):
        sys.exit(f"error: {BRUSHES} not found")

    cats = sorted(
        d for d in os.listdir(BRUSHES)
        if os.path.isdir(os.path.join(BRUSHES, d))
    )
    if not cats:
        sys.exit(f"error: no category folders under {BRUSHES}")

    lines = [
        "<!-- GENERATED FILE - do not edit by hand.",
        "     Regenerate with: python packaging/gen_wix_brushes.py -->",
        "<Wix xmlns='http://schemas.microsoft.com/wix/2006/wi'>",
        "    <Fragment>",
    ]

    total = 0
    for cat in cats:
        cat_path = os.path.join(BRUSHES, cat)
        files = sorted(f for f in os.listdir(cat_path)
                       if os.path.isfile(os.path.join(cat_path, f)))
        if not files:
            continue
        # Relative to the workspace root, which is the directory the WiX
        # compiler runs in (cargo-wix passes wxs paths relative to it).
        rel_src = os.path.relpath(cat_path, ROOT).replace("/", "\\")
        lines.append(f"            <!-- {cat} -->")
        lines.append(f"            <Directory Id='BRUSH_{wixtid(cat)}' Name='{cat}'>")
        lines.append(
            f"                <Component Id='BrushSet_{wixtid(cat)}' "
            f"Guid='{category_guid(cat)}'>"
        )
        for i, f in enumerate(files):
            total += 1
            # The first file of each component is its keypath. WiX can infer
            # this, but stating it keeps candle from warning and documents
            # which file MSI repair uses to verify the component.
            keypath = " KeyPath='yes'" if i == 0 else ""
            lines.append(
                f"                    <File Id='Brush_{wixtid(cat)}_{wixtid(f)}' "
                f"Name='{f}' DiskId='1' Source='{rel_src}\\{f}'{keypath} />"
            )
        lines.append("                </Component>")
        lines.append("            </Directory>")

    # All the category directories go inside one 'brushes' folder, which itself
    # sits next to pixforge.exe in 'bin'. brushes_folder() in src/app.rs looks
    # for <exe-dir>/brushes, so getting this wrong installs a library the app
    # cannot find. 'Bin' is the directory declared in main.wxs that holds the
    # exe - keep the two in step.
    body = lines[4:]
    lines = lines[:4] + [
        "        <DirectoryRef Id='Bin'>",
        "            <Directory Id='BRUSHES' Name='brushes'>",
    ] + body + [
        "            </Directory>",
        "        </DirectoryRef>",
    ]

    lines.append("    </Fragment>")
    lines.append("</Wix>")
    lines.append("")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))

    # The components live in a Fragment, so main.wxs still has to reference them
    # for them to land in the Binaries feature. Splice the refs in between the
    # markers rather than letting the two files drift apart.
    refs = [
        f"            <ComponentRef Id='BrushSet_{wixtid(c)}'/>"
        for c in cats
    ]
    update_main_refs(refs)

    print(f"wrote {OUT}: {len(cats)} categories, {total} brush files")
    print(f"updated {MAIN_WXS}: {len(refs)} ComponentRef(s)")


REFS_BEGIN = "<!-- BEGIN GENERATED BRUSH COMPONENT REFS -->"
REFS_END = "<!-- END GENERATED BRUSH COMPONENT REFS -->"


def update_main_refs(refs):
    with open(MAIN_WXS, "r", encoding="utf-8") as fh:
        text = fh.read()

    begin = text.find(REFS_BEGIN)
    end = text.find(REFS_END)
    if begin < 0 or end < 0 or end < begin:
        sys.exit(
            f"error: could not find the ComponentRef markers in {MAIN_WXS}.\n"
            "They must read exactly:\n"
            f"  {REFS_BEGIN}\n"
            f"  {REFS_END}"
        )

    body = "\n".join([REFS_BEGIN] + refs)
    text = text[:begin] + body + "\n            " + text[end:]
    with open(MAIN_WXS, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


if __name__ == "__main__":
    main()
