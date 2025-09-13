#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate canonical fractal examples as black-on-white PNGs for box counting.

Usage examples:
  python tests/generate_fractals.py --name koch --iters 4 --outdir assets/fractals
  python tests/generate_fractals.py --list  # show available names

Fractals implemented (aliases in parentheses):
  cantor1d, cantordust, koch, snowflake (triflake), terdragon,
  vicsek, qkoch1, minkowski, dragon, sierpinski, arrowhead,
  tsquare, carpet, peano, pythagoras,
  hexaflake, penrose_approx (area-filling placeholder)

All outputs are L-mode images (white background, black curves/fills).
"""
import argparse, math, os
from typing import Dict, Tuple, List
import numpy as np
from PIL import Image, ImageDraw


def save_L(img: Image.Image, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.convert("L").save(path)


# ---------- Generic turtle + L-system ----------
def lsystem(axiom: str, rules: Dict[str, str], iters: int) -> str:
    s = axiom
    for _ in range(max(0, iters)):
        s = "".join(rules.get(ch, ch) for ch in s)
    return s


def draw_turtle(seq: str, angle_deg: float, w: int = 1024, h: int = 1024,
                margin: int = 32, step_scale: float = 1.0, start_heading: float = 0.0) -> Image.Image:
    # First pass: collect points with unit step, then scale to fit
    x, y, th = 0.0, 0.0, math.radians(start_heading)
    pts: List[Tuple[float, float]] = [(x, y)]
    stack: List[Tuple[float, float, float]] = []
    for ch in seq:
        if ch in "Ff":
            x += math.cos(th)
            y += math.sin(th)
            if ch == 'F':
                pts.append((x, y))
        elif ch == '+':
            th += math.radians(angle_deg)
        elif ch == '-':
            th -= math.radians(angle_deg)
        elif ch == '[':
            stack.append((x, y, th))
        elif ch == ']':
            x, y, th = stack.pop() if stack else (x, y, th)
            pts.append((x, y))
        else:
            # Ignore other symbols (variables)
            pass
    if len(pts) < 2:
        img = Image.new("L", (w, h), 255)
        return img
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    sx = (w - 2*margin) / max(1e-9, (maxx - minx))
    sy = (h - 2*margin) / max(1e-9, (maxy - miny))
    s = min(sx, sy) * step_scale
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    prev = None
    for (px, py) in pts:
        xi = int(margin + (px - minx) * s)
        yi = int(margin + (py - miny) * s)
        if prev is not None:
            d.line([prev, (xi, yi)], fill=0, width=1)
        prev = (xi, yi)
    return img


# ---------- Specific generators ----------
def cantor1d(w=1024, h=256, iters=6) -> Image.Image:
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    def rec(x0, x1, y, k):
        if k == 0:
            d.line([(x0, y), (x1, y)], fill=0, width=2)
            return
        L = (x1 - x0) / 3.0
        rec(x0, x0+L, y, k-1)
        rec(x0+2*L, x1, y, k-1)
    for i in range(iters):
        y = int((i+1) * h/(iters+1))
        rec(32, w-32, y, i)
    return img


def cantordust(w=1024, h=1024, iters=4) -> Image.Image:
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    def rec(x, y, size, k):
        if k == 0:
            d.rectangle([x, y, x+size, y+size], fill=0)
            return
        step = size // 3
        for dy in range(3):
            for dx in range(3):
                if dx == 1 or dy == 1:
                    continue
                rec(x+dx*step, y+dy*step, step, k-1)
    side = min(w, h) - 64
    rec((w-side)//2, (h-side)//2, side, iters)
    return img


def koch_curve(iters=4, w=1024, h=512):
    rules = {"F": "F+F--F+F"}
    seq = lsystem("F", rules, iters)
    return draw_turtle(seq, angle_deg=60, w=w, h=h, start_heading=0.0)


def snowflake(iters=4, w=1024, h=1024):
    # 3 rotated Koch curves
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    rules = {"F": "F+F--F+F"}
    seq = lsystem("F", rules, iters)
    for rot in (0, 120, 240):
        part = draw_turtle(seq, 60, w, h, start_heading=rot)
        d.bitmap((0, 0), part, fill=0)
    return img


def terdragon(iters=6, w=1024, h=768):
    seq = lsystem("F", {"F": "F+F-F"}, iters)
    return draw_turtle(seq, 120, w, h)


def vicsek(iters=3, w=1024, h=1024):
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    def rec(x, y, size, k):
        if k == 0:
            d.rectangle([x, y, x+size, y+size], fill=0)
            return
        s3 = size // 3
        # center and 4 corners
        rec(x, y, s3, k-1)
        rec(x+2*s3, y, s3, k-1)
        rec(x+s3, y+s3, s3, k-1)
        rec(x, y+2*s3, s3, k-1)
        rec(x+2*s3, y+2*s3, s3, k-1)
    side = min(w, h) - 64
    rec((w-side)//2, (h-side)//2, side, iters)
    return img


def qkoch1(iters=4, w=1024, h=768):
    # Quadratic Koch (type 1): F -> F+F−F−F+F with 90°
    seq = lsystem("F", {"F": "F+F-F-F+F"}, iters)
    return draw_turtle(seq, 90, w, h)


def minkowski(iters=3, w=1024, h=768):
    # Minkowski sausage: F -> F+F−F−F+F+F+F−F (90°)
    seq = lsystem("F", {"F": "F+F-F-F+F+F+F-F"}, iters)
    return draw_turtle(seq, 90, w, h)


def dragon(iters=12, w=1024, h=768):
    rules = {"X": "X+YF+", "Y": "-FX-Y"}
    seq = lsystem("FX", rules, iters)
    # interpret F only
    seq2 = "".join(ch for ch in seq if ch in "F+-[]")
    return draw_turtle(seq2, 90, w, h)


def sierpinski_triangle(iters=7, w=1024, h=1024):
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    def rec(x, y, size, k):
        if k == 0:
            d.polygon([(x, y+size), (x+size, y+size), (x+size/2, y)], fill=0)
            return
        s2 = size/2
        rec(x, y+size/2, s2, k-1)
        rec(x+size/2, y+size/2, s2, k-1)
        rec(x+size/4, y, s2, k-1)
    side = min(w, h) - 64
    rec((w-side)//2, (h-side)//2, float(side), iters)
    return img


def arrowhead(iters=7, w=1024, h=1024):
    rules = {"A": "B-A-B", "B": "A+B+A"}
    seq = lsystem("A", rules, iters)
    seq2 = seq.replace('A', 'F').replace('B', 'F')
    return draw_turtle(seq2, 60, w, h)


def tsquare(iters=5, w=1024, h=1024):
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    def rec(x, y, size, k):
        if k == 0:
            d.rectangle([x, y, x+size, y+size], outline=0, width=1)
            return
        d.rectangle([x, y, x+size, y+size], outline=0, width=1)
        s2 = size // 2
        rec(x - s2//2, y - s2//2, s2, k-1)
        rec(x + size - s2//2, y - s2//2, s2, k-1)
        rec(x - s2//2, y + size - s2//2, s2, k-1)
        rec(x + size - s2//2, y + size - s2//2, s2, k-1)
    side = min(w, h) // 3
    rec((w-side)//2, (h-side)//2, side, iters)
    return img


def carpet(iters=4, w=1024, h=1024):
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    def rec(x, y, size, k):
        if k == 0:
            d.rectangle([x, y, x+size, y+size], fill=0)
            return
        s3 = size // 3
        for dy in range(3):
            for dx in range(3):
                if dx == 1 and dy == 1:
                    continue
                rec(x+dx*s3, y+dy*s3, s3, k-1)
    side = min(w, h) - 64
    rec((w-side)//2, (h-side)//2, side, iters)
    return img

def peano(iters=3, w=1024, h=1024):
    # Peano curve L-system
    rules = {"X": "XFYFX+F+YFXFY-F-XFYFX", "Y": "YFXFY-F-XFYFX+F+YFXFY"}
    seq = lsystem("X", rules, iters)
    seq2 = "".join(ch if ch in "F+-[]" else ('F' if ch in 'XY' else '') for ch in seq)
    return draw_turtle(seq2, 90, w, h)

def pythagoras(iters=8, w=1024, h=1024):
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    def rec(x, y, size, k):
        if k == 0 or size < 2:
            return
        d.rectangle([x, y-size, x+size, y], fill=0)
        s2 = int(size * 0.7)
        rec(x - s2//2, y - size, s2, k-1)
        rec(x + size - s2//2, y - size, s2, k-1)
    side = min(w, h) // 6
    rec(w//2 - side//2, h - 32, side, iters)
    return img

def hexaflake(iters=3, w=1024, h=1024):
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    R = (min(w, h) - 64) // 3
    cx, cy = w//2, h//2
    def hexagon(cx, cy, r):
        return [(cx + r*math.cos(math.radians(60*k)), cy + r*math.sin(math.radians(60*k))) for k in range(6)]
    def rec(cx, cy, r, k):
        if k == 0 or r < 2:
            d.polygon(hexagon(cx, cy, r), outline=0)
            return
        d.polygon(hexagon(cx, cy, r), outline=0)
        for ang in range(0, 360, 60):
            nx = cx + r*math.cos(math.radians(ang))
            ny = cy + r*math.sin(math.radians(ang))
            rec(nx, ny, r/3, k-1)
        rec(cx, cy, r/3, k-1)
    rec(cx, cy, R, iters)
    return img


def penrose_approx(iters: int = 3, w: int = 1024, h: int = 1024):
    # Placeholder: dense hatch fill to approximate 2D coverage (D≈2)
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    step = max(2, 8 - int(iters))  # rough control via iters
    for y in range(32, h-32, step):
        d.line([(32, y), (w-32, y)], fill=0)
    for x in range(32, w-32, step):
        d.line([(x, 32), (x, h-32)], fill=0)
    return img

NAME_MAP = {
    "cantor1d": cantor1d,
    "cantordust": cantordust,
    "koch": koch_curve,
    "vicsek": vicsek,
    "qkoch1": qkoch1,
    "sierpinski": sierpinski_triangle,
    "carpet": carpet,
    "peano": peano,
    "hexaflake": hexaflake,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", help="Fractal name; use --list to see options")
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--outdir", default="assets/fractals")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list or not args.name:
        print("Available:")
        for k in sorted(NAME_MAP.keys()):
            print(" ", k)
        return

    name = args.name.lower()
    if name not in NAME_MAP:
        raise SystemExit(f"Unknown fractal name: {name}")

    fn = NAME_MAP[name]
    # Call with common signature where possible
    if name in ("cantor1d",):
        img = fn(iters=args.iters)
    elif name in ("koch", "terdragon", "qkoch1", "minkowski", "dragon"):
        img = fn(iters=args.iters)
    elif name in ("hilbert", "moore"):
        img = fn(iters=max(1, args.iters))
    else:
        img = fn(iters=args.iters)

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"{name}.png")
    save_L(img, out)
    print("[ok] wrote", out)


if __name__ == "__main__":
    main()
