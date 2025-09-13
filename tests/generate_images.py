#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate simple black-on-white test images:
- line.png: thin diagonal line
- sine.png: thin sine curve
- rectangle.png: filled rectangle
- circle.png: filled circle
- sierpinski.png: Sierpiński triangle (fractal)

Usage:
  python tests/generate_images.py --line --sine --rectangle --circle --sierpinski --outdir assets
  # or individually:
  python tests/generate_images.py --line --outdir assets
"""
import argparse, os
import numpy as np
from PIL import Image, ImageDraw

def gen_line(w=1024, h=1024, margin=100, width=1):
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    d.line([(margin, h-margin), (w-margin, margin)], fill=0, width=width)
    return img

def gen_sine(w=1024, h=1024, margin=80, width=1, wavelength_px=400):
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    A = h//3 - margin
    y0 = h//2
    xs = np.arange(margin, w - margin)
    freq = 2*np.pi / float(wavelength_px)
    pts = [(int(x), int(y0 + A*np.sin(freq*x))) for x in xs]
    d.line(pts, fill=0, width=width)
    return img

def gen_rectangle(w=1024, h=1024, side_frac=0.35, line_width=6, filled=True):
    """Rectangle centered on a white background (smaller size).
    side_frac: rectangle side length as a fraction of min(w, h).
    If filled=True, draw a solid black rectangle (area set, D≈2 expected); else outline-only.
    """
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    side = max(8, int(side_frac * min(w, h)))
    cx, cy = w // 2, h // 2
    x0, y0 = cx - side // 2, cy - side // 2
    x1, y1 = cx + side // 2, cy + side // 2
    if filled:
        d.rectangle([x0, y0, x1, y1], fill=0, outline=0, width=1)
    else:
        d.rectangle([x0, y0, x1, y1], fill=None, outline=0, width=line_width)
    return img

def gen_circle(w=1024, h=1024, margin=150):
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    cx, cy = w // 2, h // 2
    r = min(w, h) // 2 - margin
    bbox = [cx - r, cy - r, cx + r, cy + r]
    d.ellipse(bbox, fill=0, outline=0, width=1)
    return img

def gen_sierpinski(w=1024, h=1024, margin=60, depth=7):
    """Generate a Sierpiński triangle by recursive subdivision (filled black set)."""
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)

    # Equilateral triangle centered horizontally, sitting above bottom margin
    s = min(w, h) - 2 * margin  # side length in pixels
    s = max(2, s)
    cx = w / 2.0
    yb = h - margin
    htri = s * (3 ** 0.5) / 2.0
    yt = yb - htri
    p1 = (cx - s / 2.0, yb)
    p2 = (cx + s / 2.0, yb)
    p3 = (cx, yt)

    def mid(a, b):
        return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)

    def draw_tri(p1, p2, p3):
        d.polygon([p1, p2, p3], fill=0)

    def rec(p1, p2, p3, k):
        if k <= 0:
            draw_tri(p1, p2, p3)
            return
        m12 = mid(p1, p2)
        m23 = mid(p2, p3)
        m31 = mid(p3, p1)
        rec(p1, m12, m31, k - 1)
        rec(m12, p2, m23, k - 1)
        rec(m31, m23, p3, k - 1)

    rec(p1, p2, p3, int(depth))
    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", action="store_true")
    ap.add_argument("--sine", action="store_true")
    ap.add_argument("--rectangle", action="store_true")
    ap.add_argument("--circle", action="store_true")
    ap.add_argument("--outdir", default="assets")
    ap.add_argument("--sierpinski", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    selected = any([args.line, args.sine, args.rectangle, args.circle, args.sierpinski])
    if not selected:
        # Default: generate all
        args.line = args.sine = args.rectangle = args.circle = args.sierpinski = True

    if args.line:
        img = gen_line()
        img.save(os.path.join(args.outdir, "line.png"))
        print("[ok] wrote", os.path.join(args.outdir, "line.png"))
    if args.sine:
        img = gen_sine()
        img.save(os.path.join(args.outdir, "sine.png"))
        print("[ok] wrote", os.path.join(args.outdir, "sine.png"))
    if args.rectangle:
        img = gen_rectangle()
        img.save(os.path.join(args.outdir, "rectangle.png"))
        print("[ok] wrote", os.path.join(args.outdir, "rectangle.png"))
    if args.circle:
        img = gen_circle()
        img.save(os.path.join(args.outdir, "circle.png"))
        print("[ok] wrote", os.path.join(args.outdir, "circle.png"))
    if args.sierpinski:
        img = gen_sierpinski()
        img.save(os.path.join(args.outdir, "sierpinski.png"))
        print("[ok] wrote", os.path.join(args.outdir, "sierpinski.png"))

if __name__ == "__main__":
    main()
