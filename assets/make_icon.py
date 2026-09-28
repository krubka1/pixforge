#!/usr/bin/env python3
"""Generate the PixForge application icon (.png + multi-size .ico).

Motif: an isometric mesh cube (the 3D surface you paint on) with a paint
stroke swept across its top face. Colours are taken from the app's own dark
UI palette so the icon matches the window chrome.

Rendered at 4x supersample and downscaled, so run this if the icon changes.
"""

import os
from PIL import Image, ImageDraw, ImageFilter

S = 4              # supersample factor
N = 256            # final size
C = N * S          # canvas size

# App palette (src/app.rs)
BG_TOP = (34, 38, 46, 255)
BG_BOT = (16, 18, 22, 255)
ACCENT = (82, 158, 228, 255)       # ACCENT
FACE_TOP = (108, 179, 241, 255)
FACE_LEFT = (58, 122, 187, 255)
FACE_RIGHT = (40, 88, 141, 255)
STROKE_A = (233, 169, 98, 255)     # warn / warm paint
STROKE_B = (245, 206, 140, 255)
EDGE = (14, 16, 20, 190)


def lerp(a, b, t):
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(4))


def rounded_mask(size, radius):
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=255)
    return m


def vgrad(size, top, bot):
    """Vertical linear gradient, generated small then upscaled for speed."""
    strip = Image.new("RGBA", (1, 256))
    px = strip.load()
    for y in range(256):
        px[0, y] = lerp(top, bot, y / 255.0)
    return strip.resize((size, size), Image.BILINEAR)


def draw_cube(d, cx, cy, w, rh, side):
    """Isometric cube. cy is the y of the top vertex; rh the rhombus half-height."""
    top = (cx, cy)
    right = (cx + w, cy + rh)
    bottom = (cx, cy + 2 * rh)
    left = (cx - w, cy + rh)

    d.polygon([top, right, bottom, left], fill=FACE_TOP)                      # top
    d.polygon([left, bottom, (bottom[0], bottom[1] + side),
               (left[0], left[1] + side)], fill=FACE_LEFT)                    # left
    d.polygon([right, bottom, (bottom[0], bottom[1] + side),
               (right[0], right[1] + side)], fill=FACE_RIGHT)                 # right

    # Thin seams so the faces read separately at 16px.
    d.line([left, (left[0], left[1] + side)], fill=EDGE, width=max(1, S))
    d.line([right, (right[0], right[1] + side)], fill=EDGE, width=max(1, S))
    d.line([bottom, (bottom[0], bottom[1] + side)], fill=EDGE, width=max(1, S))
    d.line([top, right, bottom, left, top], fill=EDGE, width=max(1, S))
    return top, right, bottom, left


def paint_stroke(d, pts, width):
    """Tapered stroke: a few overlapping segments plus a round cap at each end."""
    d.line(pts, fill=STROKE_A, width=width, joint="curve")
    # Soft inner highlight for a wet-paint look.
    inner = [(x, y - width * 0.16) for x, y in pts]
    d.line(inner, fill=STROKE_B, width=max(1, int(width * 0.34)), joint="curve")
    for (x, y) in (pts[0], pts[-1]):
        d.ellipse([x - width / 2, y - width / 2, x + width / 2, y + width / 2], fill=STROKE_A)


def main():
    img = Image.new("RGBA", (C, C), (0, 0, 0, 0))
    img.paste(vgrad(C, BG_TOP, BG_BOT), (0, 0))
    d = ImageDraw.Draw(img)

    # Cube geometry, sized so it fills the tile with a comfortable margin.
    w = int(C * 0.30)
    rh = int(C * 0.155)
    side = int(C * 0.30)
    cx = C // 2
    cy = int(C * 0.235)
    draw_cube(d, cx, cy, w, rh, side)

    # Stroke swept across the top face, from the left vertex up to the right.
    stroke_w = int(C * 0.105)
    paint_stroke(
        d,
        [
            (cx - w * 0.80, cy + rh * 1.02),
            (cx - w * 0.30, cy + rh * 0.44),
            (cx + w * 0.28, cy + rh * 0.60),
            (cx + w * 0.78, cy + rh * 1.00),
        ],
        stroke_w,
    )

    # Top-left specular on the tile for a little depth.
    gloss = Image.new("RGBA", (C, C), (0, 0, 0, 0))
    ImageDraw.Draw(gloss).ellipse([-C * 0.35, -C * 0.85, C * 0.95, C * 0.30],
                                  fill=(255, 255, 255, 16))
    gloss = gloss.filter(ImageFilter.GaussianBlur(C * 0.10))
    img.alpha_composite(gloss)

    img.putalpha(rounded_mask(C, int(C * 0.225)))

    out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon-256.png")
    img.resize((N, N), Image.LANCZOS).save(out_png)
    print("wrote", out_png)

    # .ico: 16/24/32/48/64/128/256 so Explorer, taskbar, Alt-Tab and the
    # installer all get a correctly sized variant. PIL resamples each entry
    # from the supersampled master, so quality is as good as the source.
    out_ico = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.ico")
    img.save(out_ico, format="ICO",
             sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("wrote", out_ico)


if __name__ == "__main__":
    main()
