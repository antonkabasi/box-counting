#!/usr/bin/env python3
"""
Clean and normalize an input image to a consistent binary form for box counting.

Features
- Flattens transparency onto white or black
- Extracts structure from grayscale or color (optionally target red/green/blue)
- Thresholding: fixed or Otsu (no external deps)
- Morphology: open/close (denoise, bridge gaps) via 3x3 kernel
- Borderize: convert filled areas to a 1px-ish outline (mask − erode(mask))
- Crop to foreground bounding box (optional)

Examples
- Normalize any image into a thin outline mask:
  python scripts/clean_image.py --in in/uk.png --out in/uk_clean.png \
    --alpha-bg white --threshold otsu --invert --close 1 --borderize 1 --crop

- Extract red lines on white background:
  python scripts/clean_image.py --in in/red_curve.png --out in/red_curve_clean.png \
    --target-color red --color-delta 40 --threshold fixed --fixed-thresh 32 --borderize 1
"""
from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
from PIL import Image, ImageFilter


def otsu_threshold_u8(img_u8: np.ndarray) -> int:
    hist = np.bincount(img_u8.ravel(), minlength=256).astype(np.float64)
    total = img_u8.size
    if total == 0:
        return 127
    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b2 = (mu_t * omega - mu) ** 2 / np.where(denom == 0, np.nan, denom)
    k = int(np.nanargmax(sigma_b2))
    return k


def flatten_alpha(im: Image.Image, bg: str = "white") -> Image.Image:
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        bgc = (255, 255, 255) if bg.lower() == "white" else (0, 0, 0)
        base = Image.new("RGBA", im.size, bgc + (255,))
        im = Image.alpha_composite(base, im.convert("RGBA")).convert("RGB")
    return im


def to_grayscale_u8(im: Image.Image, target_color: str | None, color_delta: int) -> np.ndarray:
    im = im.convert("RGB")
    arr = np.array(im, dtype=np.uint8)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    if target_color in ("red", "green", "blue"):
        if target_color == "red":
            dom = r.astype(int) - np.maximum(g, b).astype(int)
        elif target_color == "green":
            dom = g.astype(int) - np.maximum(r, b).astype(int)
        else:
            dom = b.astype(int) - np.maximum(r, g).astype(int)
        dom = np.clip(dom, 0, 255).astype(np.uint8)
        # Threshold dominance as a quick pre-filter
        mask = dom >= int(color_delta)
        # Map to grayscale with high contrast for dominant pixels
        gray = np.where(mask, 255, 0).astype(np.uint8)
        return gray
    # Default: standard luminance
    im_gray = im.convert("L")
    return np.array(im_gray, dtype=np.uint8)


def morph_open_close(mask: Image.Image, open_iters: int, close_iters: int) -> Image.Image:
    img = mask
    for _ in range(max(0, int(open_iters))):
        img = img.filter(ImageFilter.MinFilter(3))  # erode
        img = img.filter(ImageFilter.MaxFilter(3))  # dilate
    for _ in range(max(0, int(close_iters))):
        img = img.filter(ImageFilter.MaxFilter(3))  # dilate
        img = img.filter(ImageFilter.MinFilter(3))  # erode
    return img


def borderize_mask(mask_bin: np.ndarray, width: int = 1) -> np.ndarray:
    # Erode repeatedly and subtract from original to get a thin boundary
    img = Image.fromarray((mask_bin * 255).astype(np.uint8))
    eroded = img
    for _ in range(max(1, int(width))):
        eroded = eroded.filter(ImageFilter.MinFilter(3))
    er = np.array(eroded, dtype=np.uint8) >= 128
    outline = mask_bin & (~er)
    return outline


def crop_to_bbox(mask_bin: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask_bin)
    if ys.size == 0:
        return mask_bin
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return mask_bin[y0:y1, x0:x1]


def main():
    ap = argparse.ArgumentParser(description="Normalize an input image into a binary mask or outline")
    ap.add_argument("--in", dest="inp", required=True, help="Input image path")
    ap.add_argument("--out", dest="out", required=True, help="Output image path (PNG)")
    ap.add_argument("--alpha-bg", choices=["white", "black"], default="white", help="Background for flattening transparency")
    ap.add_argument("--target-color", choices=["auto", "gray", "red", "green", "blue"], default="auto", help="How to extract signal from color")
    ap.add_argument("--color-delta", type=int, default=40, help="Dominance threshold for color extraction (red/green/blue)")
    ap.add_argument("--threshold", choices=["otsu", "fixed"], default="otsu")
    ap.add_argument("--fixed-thresh", type=int, default=128)
    ap.add_argument("--invert", action="store_true", help="Invert after thresholding")
    ap.add_argument("--open", dest="open_iters", type=int, default=0, help="Morphological open iterations (denoise)")
    ap.add_argument("--close", dest="close_iters", type=int, default=0, help="Morphological close iterations (bridge gaps)")
    ap.add_argument("--borderize", type=int, default=0, help="Convert filled areas to outline of this width (0=off)")
    ap.add_argument("--crop", action="store_true", help="Crop to foreground bounding box")
    ap.add_argument("--out-fg", choices=["white", "black"], default="white", help="Foreground (structure) color in output PNG")
    args = ap.parse_args()

    if not os.path.exists(args.inp):
        alt = os.path.join("in", os.path.basename(args.inp))
        if os.path.exists(alt):
            args.inp = alt
        else:
            raise SystemExit(f"[error] input image not found: {args.inp}")

    im = Image.open(args.inp)
    im = flatten_alpha(im, bg=args.alpha_bg)

    # Choose extraction strategy
    targ = None if args.target_color in ("auto", "gray") else args.target_color
    gray = to_grayscale_u8(im, targ, args.color_delta)

    # If auto and image seems red-dominant, switch to red target
    if args.target_color == "auto" and im.mode == "RGB":
        arr = np.array(im.convert("RGB"), dtype=np.uint8)
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        dom_r = (r.astype(int) - np.maximum(g, b).astype(int)).mean()
        if dom_r > args.color_delta:
            gray = to_grayscale_u8(im, "red", args.color_delta)

    # Threshold
    if args.threshold == "otsu":
        t = otsu_threshold_u8(gray)
    else:
        t = int(np.clip(args.fixed_thresh, 0, 255))
    mask = gray > t
    if args.invert:
        mask = ~mask

    # Morphology
    if args.open_iters or args.close_iters:
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img = morph_open_close(mask_img, args.open_iters, args.close_iters)
        mask = np.array(mask_img, dtype=np.uint8) >= 128

    # Borderize
    if args.borderize and args.borderize > 0:
        mask = borderize_mask(mask, width=args.borderize)

    # Crop
    if args.crop:
        mask = crop_to_bbox(mask)

    # Save
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if args.out_fg == "white":
        out_arr = (mask * 255).astype(np.uint8)
    else:
        out_arr = ((~mask) * 255).astype(np.uint8)
    Image.fromarray(out_arr).save(args.out)
    print(f"[ok] wrote {args.out}")


if __name__ == "__main__":
    main()
