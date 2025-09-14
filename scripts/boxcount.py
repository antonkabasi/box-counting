#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Box-counting (area) fractal dimension for binary microstructures from images.

Features:
- Otsu or fixed threshold (no heavy deps).
- Grid-origin averaging to reduce bias.
- Geometric ladder of box sizes.
- CSV export: box_size_px, N_boxes_mean, N_boxes_std, log columns.
- Linear fit on log N vs log(1/ε) with R².
- Optional PNG plot.

Usage examples:
  python boxcount.py --image sample_sem.png --out out/sem --plot
  python boxcount.py --image sample_sem.png --threshold fixed --fixed-thresh 140 --invert --plot
  python boxcount.py --image sample_sem.png --min-box 2 --max-box 512 --scales 12 --grid-averages 4 --plot
  python boxcount.py --image sample_sem.png --pixel-size 5e-8 --plot   # 50 nm/pixel
"""
import argparse
import os
import subprocess
from datetime import datetime, timezone
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from colorama import Fore, Style, init as colorama_init

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    def tqdm(x, **kwargs):  # minimal fallback
        return x


def otsu_threshold_u8(img_u8: np.ndarray) -> int:
    hist = np.bincount(img_u8.ravel(), minlength=256).astype(np.float64)
    total = img_u8.size
    if total == 0:
        return 127
    if np.count_nonzero(hist) <= 1:
        return 127
    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b2 = (mu_t * omega - mu) ** 2 / np.where(denom == 0, np.nan, denom)
    if np.all(~np.isfinite(sigma_b2)):
        return 127
    k = int(np.nanargmax(sigma_b2))
    return int(k)


def load_grayscale_u8(path: str) -> np.ndarray:
    im = Image.open(path).convert("L")
    return np.array(im, dtype=np.uint8)


def binarize(
    img_u8: np.ndarray,
    method: str = "otsu",
    fixed_thresh: int = 128,
    invert: bool = False,
) -> np.ndarray:
    if method == "otsu":
        t = otsu_threshold_u8(img_u8)
    elif method == "fixed":
        t = int(np.clip(fixed_thresh, 0, 255))
    else:
        raise ValueError("threshold method must be 'otsu' or 'fixed'.")
    mask = img_u8 > t
    if invert:
        mask = ~mask
    return mask


def crop_to_bbox(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return mask
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return mask[y0:y1, x0:x1]


def geometric_box_sizes(min_box: int, max_box: int, scales: int) -> List[int]:
    min_box = max(1, int(min_box))
    max_box = max(min_box, int(max_box))
    if scales <= 1:
        return [min_box]
    r = (max_box / min_box) ** (1.0 / (scales - 1))
    vals = []
    for k in range(scales):
        s = int(round(min_box * (r ** k)))
        vals.append(max(1, s))
    vals = sorted(set(vals))
    return vals


def make_offsets(s: int, n: int, rng: np.random.Generator) -> List[Tuple[int, int]]:
    base = [(0, 0), (s // 2, 0), (0, s // 2), (s // 2, s // 2)]
    if n <= 4:
        return base[:n]
    extras = set()
    while len(extras) < (n - 4):
        ox = int(rng.integers(0, s))
        oy = int(rng.integers(0, s))
        if (ox, oy) not in base and (ox, oy) not in extras:
            extras.add((ox, oy))
    return base + sorted(list(extras))


def count_boxes_with_offsets(mask: np.ndarray, s: int, offsets: List[Tuple[int, int]],
                             occupancy: str = "any", frac_tau: float = 0.05,
                             use_integral: bool = False) -> Tuple[float, float, int]:
    H, W = mask.shape
    counts = []
    for (ox, oy) in offsets:
        H2 = ((H - oy + s - 1) // s) * s + oy
        W2 = ((W - ox + s - 1) // s) * s + ox
        pad = np.zeros((H2, W2), dtype=bool)
        pad[:H, :W] = mask
        sub = pad[oy:H2, ox:W2]
        h2, w2 = sub.shape
        if occupancy == "center":
            cy = s // 2
            cx = s // 2
            centers = sub[cy::s, cx::s]
            occ = centers
        else:
            if use_integral:
                a = sub.astype(np.int32)
                sat = a.cumsum(axis=0).cumsum(axis=1)
                sat2 = np.zeros((h2 + 1, w2 + 1), dtype=np.int64)
                sat2[1:, 1:] = sat
                ys = np.arange(0, h2, s)
                xs = np.arange(0, w2, s)
                Y0, X0 = np.meshgrid(ys, xs, indexing='ij')
                Y1 = Y0 + s
                X1 = X0 + s
                sums = sat2[Y1, X1] - sat2[Y0, X1] - sat2[Y1, X0] + sat2[Y0, X0]
                if occupancy == "frac":
                    occ = sums >= (frac_tau * (s * s))
                else:
                    occ = sums > 0
            else:
                blocks = sub.reshape(h2 // s, s, w2 // s, s)
                if occupancy == "frac":
                    tsum = blocks.sum(axis=(1, 3))
                    occ = tsum >= (frac_tau * (s * s))
                else:
                    occ = blocks.any(axis=(1, 3))
        counts.append(int(occ.sum()))
    counts = np.array(counts, dtype=float)
    return float(np.mean(counts)), float(np.std(counts, ddof=1) if len(counts) > 1 else 0.0), int(len(counts))


@dataclass
class FitResult:
    slope: float
    intercept: float
    r2: float
    n: int
    slope_stderr: float
    intercept_stderr: float


def linear_fit(x: np.ndarray, y: np.ndarray) -> FitResult:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m, b = np.polyfit(x, y, 1)
    yhat = m * x + b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    n = len(x)
    sigma2 = ss_res / (n - 2) if n > 2 else 0.0
    Sxx = float(np.sum((x - x.mean()) ** 2))
    slope_stderr = float(np.sqrt(sigma2 / Sxx)) if Sxx > 0 else 0.0
    intercept_stderr = float(np.sqrt(sigma2 * (1.0 / n + (x.mean() ** 2) / Sxx))) if (n > 0 and Sxx > 0) else 0.0
    return FitResult(slope=m, intercept=b, r2=r2, n=n, slope_stderr=slope_stderr, intercept_stderr=intercept_stderr)


def weighted_linear_fit(x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray]) -> FitResult:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if w is None:
        return linear_fit(x, y)
    w = np.asarray(w, dtype=float)
    X = np.column_stack([x, np.ones_like(x)])
    sw = np.sqrt(np.maximum(w, 0.0))
    Xw = X * sw[:, None]
    yw = y * sw
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    m, b = float(beta[0]), float(beta[1])
    yhat = m * x + b
    r = y - yhat
    n = len(x)
    dof = max(1, n - 2)
    ss_res_w = float(np.sum(w * r * r))
    XtWX = X.T @ (w[:, None] * X)
    try:
        XtWX_inv = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        XtWX_inv = np.linalg.pinv(XtWX)
    sigma2 = ss_res_w / dof
    cov = sigma2 * XtWX_inv
    slope_stderr = float(np.sqrt(max(cov[0, 0], 0.0)))
    intercept_stderr = float(np.sqrt(max(cov[1, 1], 0.0)))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(r ** 2)) / ss_tot if ss_tot > 0 else 0.0
    return FitResult(slope=m, intercept=b, r2=r2, n=n, slope_stderr=slope_stderr, intercept_stderr=intercept_stderr)


def main():
    ap = argparse.ArgumentParser(description="Box-counting fractal dimension on images.")
    ap.add_argument("--image", required=True, help="Path to input image (png/jpg/tiff...).")
    ap.add_argument("--out", default="boxcount_output", help="Output prefix (without extension).")
    ap.add_argument("--threshold", choices=["otsu", "fixed"], default="otsu", help="Binarization method.")
    ap.add_argument("--fixed-thresh", type=int, default=128, help="Fixed threshold (0..255) if --threshold fixed.")
    ap.add_argument("--invert", action="store_true", help="Invert mask after thresholding (useful if foreground is dark).")
    ap.add_argument("--crop", action="store_true", help="Crop to bounding box of foreground before counting.")
    ap.add_argument("--pad", type=int, default=0, help="Pad this many background pixels on all sides after crop")
    ap.add_argument("--min-box", type=int, default=2, help="Smallest box size in pixels.")
    ap.add_argument("--max-box", type=int, default=512, help="Largest box size in pixels.")
    ap.add_argument("--scales", type=int, default=10, help="Number of geometric scales between min and max.")
    ap.add_argument("--grid-averages", type=int, default=4, help="Grid-origin averages per scale (1..N).")
    ap.add_argument("--pixel-size", type=float, default=None, help="Physical pixel size (e.g., meters/pixel).")
    ap.add_argument("--drop-head", type=int, default=0, help="Drop this many smallest scales from the fit.")
    ap.add_argument("--drop-tail", type=int, default=0, help="Drop this many largest scales from the fit.")
    ap.add_argument("--plot", action="store_true", help="Save a PNG plot of log N vs log(1/ε) and the linear fit.")
    ap.add_argument("--plot-linear", action="store_true", help="Also save a non-log plot: N vs 1/ε (power-law curve)")
    ap.add_argument("--ci", type=int, choices=[90, 95, 99], default=95, help="Confidence/prediction interval level (90/95/99)")
    ap.add_argument("--bootstrap", type=int, default=0, help="Bootstrap replicates over grid offsets (0=off)")
    ap.add_argument("--boot-grid-averages", type=int, default=None, help="Offsets per scale for bootstrap (defaults to --grid-averages)")
    ap.add_argument("--boot-seed", type=int, default=123, help="Random seed for bootstrap offsets")
    ap.add_argument("--band", choices=["prediction","fit"], default="prediction",
                    help="Shaded band type on log–log plot: prediction (default) or fit (line CI)")
    ap.add_argument("--save-grids", action="store_true", help="Save grid overlay images under a 'grids/' subfolder next to outputs")
    ap.add_argument("--grids-max-offsets", type=int, default=1, help="Max number of grid offsets to render per box size (default: 1)")
    ap.add_argument("--seed", type=int, default=12345, help="RNG seed for grid offsets (reproducibility)")
    ap.add_argument("--auto-window", action="store_true", help="Auto-pick fit window (min width ≥5) by max R², tie-break by minimal curvature")
    ap.add_argument("--min-window", type=int, default=5, help="Minimum number of scales in the fit window when --auto-window")
    ap.add_argument("--integral", action="store_true", help="Use summed-area table optimisation (optional)")
    ap.add_argument("--occupancy", choices=["any","center","frac"], default="any", help="Box occupancy rule")
    ap.add_argument("--frac", type=float, default=0.05, help="τ for occupancy=frac (fraction of pixels in a box)")
    ap.add_argument("--progress", action="store_true", help="Show per-scale progress if tqdm is available")
    ap.add_argument("--grid-overlay", choices=["raw","edge"], default="raw",
                    help="When saving grids: 'raw' overlays grid on the original grayscale (no edits); 'edge' uses binarized edge on white.")
    args = ap.parse_args()

    colorama_init(autoreset=True)
    OK = Fore.GREEN + "[ok]" + Style.RESET_ALL
    INFO = Fore.CYAN + "[info]" + Style.RESET_ALL
    RES = Fore.GREEN + "[result]" + Style.RESET_ALL
    ERR = Fore.RED + "[error]" + Style.RESET_ALL

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    image_path = args.image
    if not os.path.exists(image_path):
        alt = os.path.join("in", image_path)
        if os.path.exists(alt):
            image_path = alt
    if not os.path.exists(image_path):
        raise SystemExit(f"[error] Image not found: {args.image}")
    if os.path.isdir(image_path):
        raise SystemExit(f"[error] Provided path is a directory, not a file: {image_path}")

    # Keep raw grayscale for grid overlays; use a working copy for binarization
    img_u8_raw = load_grayscale_u8(image_path)
    img_u8 = img_u8_raw.copy()

    # Threshold to binary for counting
    mask = binarize(img_u8, method=args.threshold, fixed_thresh=args.fixed_thresh, invert=args.invert)

    # sanity: ensure non-trivial mask BEFORE cropping
    occ_raw = float(mask.mean())
    if occ_raw <= 0.0 or occ_raw >= 1.0:
        raise SystemExit(
            f"{ERR} Thresholding produced foreground fraction={occ_raw:.4f}. "
            f"Adjust threshold/invert or ensure image has features."
        )

    if args.crop:
        mask = crop_to_bbox(mask)
    if args.pad and args.pad > 0:
        padw = int(args.pad)
        mask = np.pad(mask, padw, mode='constant', constant_values=False)

    sizes = geometric_box_sizes(args.min_box, args.max_box, args.scales)

    rows = []
    rng_main = np.random.default_rng(int(args.seed))
    iter_sizes = tqdm(sizes, desc="scales") if args.progress else sizes
    warnings_list = []
    all_offsets = {}
    for s in iter_sizes:
        offs = make_offsets(s, max(1, args.grid_averages), rng_main)
        all_offsets[int(s)] = offs
        meanN, stdN, nrep = count_boxes_with_offsets(mask, int(s), offs,
                                                     occupancy=args.occupancy,
                                                     frac_tau=float(args.frac),
                                                     use_integral=bool(args.integral))
        if meanN <= 0.0:
            warnings_list.append(f"scale s={s}: mean count is zero; dropped from fit")
        rows.append((s, meanN, stdN, nrep))

        # Optional: save grid overlay images
        if args.save_grids and args.out:
            grids_dir = os.path.join(os.path.dirname(args.out), "grids")
            os.makedirs(grids_dir, exist_ok=True)

            # Decide background for grid overlay
            if args.grid_overlay == "raw":
                base = Image.fromarray(img_u8_raw).convert("RGB")
                w, h = base.size
            else:
                h, w = mask.shape
                base = Image.new("RGB", (w, h), (255, 255, 255))
                try:
                    mask_img = Image.fromarray((mask.astype(np.uint8) * 255))
                    eroded = mask_img.filter(ImageFilter.MinFilter(3))
                    er = np.array(eroded, dtype=np.uint8) >= 128
                    edge = mask & (~er)
                    edge_y, edge_x = np.where(edge)
                    if edge_y.size:
                        d_edge = ImageDraw.Draw(base)
                        for yy, xx in zip(edge_y.tolist(), edge_x.tolist()):
                            d_edge.point((int(xx), int(yy)), fill=(80, 80, 80))
                except Exception:
                    pass

            m = max(1, int(args.grids_max_offsets))
            for (ox, oy) in offs[:m]:
                img_overlay = base.copy()
                d = ImageDraw.Draw(img_overlay)
                x = int(ox)
                while x <= w:
                    d.line([(x, 0), (x, h)], fill=(0, 200, 0), width=1)
                    x += int(s)
                y = int(oy)
                while y <= h:
                    d.line([(0, y), (w, y)], fill=(0, 200, 0), width=1)
                    y += int(s)
                grid_name = os.path.join(grids_dir, f"grid_s{s}_ox{ox}_oy{oy}.png")
                img_overlay.save(grid_name)

    # build table
    arr = np.array(rows, dtype=float)  # columns: s, meanN, stdN, nrep
    s_px = arr[:, 0]
    N_mean = arr[:, 1]
    N_std = arr[:, 2]
    nrep = arr[:, 3]

    if args.pixel_size is not None and args.pixel_size > 0:
        eps = s_px * args.pixel_size
        eps_label = "epsilon_phys"
    else:
        eps = s_px
        eps_label = "epsilon_px"

    inv_eps = 1.0 / eps
    with np.errstate(divide="ignore"):
        log_inv_eps = np.log(inv_eps)
        log_N = np.log(N_mean)

    header = ["box_size_px", "N_boxes_mean", "N_boxes_std", "n_grid_averages",
              eps_label, "inv_epsilon", "log_inv_epsilon", "log_N", "in_fit_window", "offsets"]
    in_fit = np.zeros_like(s_px, dtype=bool)
    offsets_col = np.array([";".join([f"{ox}:{oy}" for (ox,oy) in all_offsets[int(s)]]) for s in s_px], dtype=object)
    table = np.column_stack([s_px, N_mean, N_std, nrep, eps, inv_eps, log_inv_eps, log_N, in_fit, offsets_col])

    csv_path = args.out + ".csv"
    if pd is not None:
        df = pd.DataFrame(table, columns=header)
        df.to_csv(csv_path, index=False)
    else:
        np.savetxt(csv_path, table, delimiter=",", header=",".join(header), comments="", fmt="%.8g")
    try:
        rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        rev = "unknown"
    stamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(csv_path, "a", encoding="utf-8") as fcsv:
        fcsv.write(f"\n# generated-by: boxcount (rev={rev}, utc={stamp})\n")

    k0 = int(np.clip(args.drop_head, 0, len(s_px)))
    k1 = int(np.clip(len(s_px) - args.drop_tail, k0 + 2, len(s_px)))
    if args.auto_window and len(s_px) >= max(5, args.min_window):
        best = None
        x_all = log_inv_eps
        y_all = log_N
        for i in range(0, len(s_px) - args.min_window + 1):
            for j in range(i + args.min_window, len(s_px) + 1):
                xi = x_all[i:j]
                yi = y_all[i:j]
                var_i = (N_std[i:j] / np.maximum(N_mean[i:j], 1e-12)) ** 2
                wi = 1.0 / np.maximum(var_i, 1e-8)
                fr = weighted_linear_fit(xi, yi, wi)
                try:
                    a2 = np.polyfit(xi, yi, 2)[0]
                    curv = abs(a2)
                except Exception:
                    curv = 1e9
                score = (fr.r2, -curv)
                if (best is None) or (score > best[0]):
                    best = (score, i, j)
        if best is not None:
            k0, k1 = best[1], best[2]
    x = log_inv_eps[k0:k1]
    y = log_N[k0:k1]

    with np.errstate(divide='ignore', invalid='ignore'):
        var_logN = (N_std / np.maximum(N_mean, 1e-12)) ** 2
    w = 1.0 / np.maximum(var_logN[k0:k1], 1e-8)

    fit = weighted_linear_fit(x, y, w)
    dim = fit.slope
    in_fit = np.zeros(len(s_px), dtype=bool)
    in_fit[k0:k1] = True
    if pd is not None:
        df["in_fit_window"] = in_fit
        df["offsets"] = offsets_col
        df.to_csv(csv_path, index=False)

    boot = []
    boot_lines = None
    boot_x = None
    boot_pred_lo = boot_pred_hi = boot_pred_med = None
    if args.bootstrap and args.bootstrap > 0:
        B = int(args.bootstrap)
        Kb = int(args.boot_grid_averages) if args.boot_grid_averages is not None else int(max(1, args.grid_averages))
        rng = np.random.default_rng(int(args.boot_seed))
        boot_x = np.linspace(log_inv_eps.min(), log_inv_eps.max(), 200)
        ylines = []
        ypts = []
        for b in range(B):
            rows_b = []
            for s in geometric_box_sizes(int(args.min_box), int(args.max_box), int(args.scales)):
                offs_b = [(int(rng.integers(0, s)), int(rng.integers(0, s))) for _ in range(Kb)]
                meanN_b, stdN_b, _ = count_boxes_with_offsets(mask, s, offs_b,
                                                              occupancy=args.occupancy,
                                                              frac_tau=float(args.frac),
                                                              use_integral=bool(args.integral))
                rows_b.append((s, meanN_b, stdN_b))
            tb = np.array(rows_b, dtype=float)
            s_b = tb[:, 0]
            inv_eps_b = 1.0 / s_b
            x_b = np.log(inv_eps_b)
            y_b = np.log(np.maximum(tb[:, 1], 1e-12))
            var_b = (tb[:, 2] / np.maximum(tb[:, 1], 1e-12)) ** 2
            idx = np.argsort(s_b)
            x_b = x_b[idx]
            y_b = y_b[idx]
            var_b = var_b[idx]
            x_fit = x_b[k0:k1]
            y_fit = y_b[k0:k1]
            w_fit = 1.0 / np.maximum(var_b[k0:k1], 1e-8)
            fb = weighted_linear_fit(x_fit, y_fit, w_fit)
            boot.append(fb)
            ylines.append(fb.slope * boot_x + fb.intercept)
            ypts.append(y_b)
        boot_lines = np.vstack(ylines)
        Y = np.vstack(ypts)
        lo_q, hi_q = (0.05, 0.95) if args.ci == 90 else ((0.005, 0.995) if args.ci == 99 else (0.025, 0.975))
        boot_pred_lo = np.quantile(Y, lo_q, axis=0)
        boot_pred_hi = np.quantile(Y, hi_q, axis=0)
        boot_pred_med = np.median(Y, axis=0)

    if args.plot and plt is not None:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        se_N = N_std / np.sqrt(np.maximum(nrep, 1))
        yerr_log = np.clip(se_N / np.maximum(N_mean, 1e-12), 0.0, np.inf)
        ax.errorbar(log_inv_eps, log_N, yerr=yerr_log, fmt='o', ms=4, elinewidth=1.0,
                    capsize=2, color='tab:blue', ecolor='tab:blue', alpha=0.9, label="data")
        xx = np.linspace(log_inv_eps.min(), log_inv_eps.max(), 200)
        yy = fit.slope * xx + fit.intercept
        ax.plot(xx, yy, label=f"fit: D={dim:.4f} ± {fit.slope_stderr:.4f}, R²={fit.r2:.4f}")
        if args.band == "fit" and boot_lines is not None:
            lo_q, hi_q = (0.05, 0.95) if args.ci == 90 else ((0.005, 0.995) if args.ci == 99 else (0.025, 0.975))
            lo = np.quantile(boot_lines, lo_q, axis=0)
            hi = np.quantile(boot_lines, hi_q, axis=0)
            med = np.median(boot_lines, axis=0)
            ax.fill_between(boot_x, lo, hi, color='tab:blue', alpha=0.15, label=f"{args.ci}% CI (bootstrap fit)")
            ax.plot(boot_x, med, color='tab:blue', alpha=0.35, linewidth=2, linestyle='--', label='bootstrap median')
        elif args.band == "prediction" and boot_pred_lo is not None:
            ax.fill_between(log_inv_eps, boot_pred_lo, boot_pred_hi, color='tab:blue', alpha=0.15, label=f"{args.ci}% PI (bootstrap)")
            ax.plot(log_inv_eps, boot_pred_med, color='tab:blue', alpha=0.35, linewidth=2, linestyle='--', label='bootstrap median')
        ax.set_xlabel("log(1/ε)")
        ax.set_ylabel("log N(ε)")
        ax.legend()
        ax.grid(True, which="both", linewidth=0.5, alpha=0.5)
        try:
            rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            rev = "unknown"
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        fig.text(0.01, 0.01, f"boxcount rev {rev} · {stamp}", fontsize=6, color="#555", alpha=0.7)
        fig.tight_layout()
        fig.savefig(args.out + ".png")
        plt.close(fig)

    if args.plot_linear and plt is not None:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        x_lin = inv_eps
        y_lin = N_mean
        se_N = N_std / np.sqrt(np.maximum(nrep, 1))
        ax.errorbar(x_lin, y_lin, yerr=se_N, fmt='o', ms=4, elinewidth=1.0,
                    capsize=2, color='tab:blue', ecolor='tab:blue', alpha=0.9, label="data")
        if args.band == "prediction" and boot_pred_lo is not None:
            lo_lin = np.exp(boot_pred_lo)
            hi_lin = np.exp(boot_pred_hi)
            ax.fill_between(x_lin, lo_lin, hi_lin, color='tab:blue', alpha=0.15, label=f"{args.ci}% PI (bootstrap)")
        ax.set_xlabel("1/ε")
        ax.set_ylabel("N(ε)")
        ax.legend()
        ax.grid(True, which="both", linewidth=0.5, alpha=0.5)
        try:
            rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            rev = "unknown"
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        fig.text(0.01, 0.01, f"boxcount rev {rev} · {stamp}", fontsize=6, color="#555", alpha=0.7)
        fig.tight_layout()
        fig.savefig(args.out + "_linear.png")
        plt.close(fig)

    txt_path = args.out + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Image: {args.image}\n")
        f.write(f"Threshold: {args.threshold} (fixed={args.fixed_thresh}), invert={args.invert}\n")
        f.write(f"Crop: {args.crop}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Occupancy rule: {args.occupancy} (tau={args.frac})\n")
        f.write(f"Fit window indices: [{k0}, {k1}) (min-window={args.min_window}, auto={args.auto_window})\n")
        f.write(f"Box sizes (px): {', '.join(map(lambda v: str(int(v)), s_px))}\n")
        f.write(f"Fit window: drop_head={args.drop_head}, drop_tail={args.drop_tail}\n")
        f.write(f"Fractal dimension (slope of log N vs log(1/ε)): {dim:.6f}\n")
        f.write(f"StdErr(slope): {fit.slope_stderr:.6f}\n")
        zmap = {90: 1.6449, 95: 1.9599, 99: 2.5758}
        z = zmap.get(args.ci, 1.9599)
        lo = fit.slope - z * fit.slope_stderr
        hi = fit.slope + z * fit.slope_stderr
        f.write(f"{args.ci}% CI for slope: [{lo:.6f}, {hi:.6f}]\n")
        # Write the exact token expected by the test:
        f.write(f"R^2: {fit.r2:.6f}\n")
        f.write(f"Fit points: {fit.n}\n")
        if args.pixel_size:
            eps_phys = s_px * args.pixel_size
            f.write(f"Pixel size: {args.pixel_size} (physical units per pixel)\n")
            f.write(f"Physical epsilon range: [{eps_phys.min():.6g}, {eps_phys.max():.6g}]\n")
        if warnings_list:
            f.write("Warnings:\n")
            for wmsg in warnings_list:
                f.write(f"- {wmsg}\n")

    print(f"{OK} CSV: {csv_path}")
    if args.plot and plt is not None:
        print(f"{OK} Plot: {args.out}.png")
    if args.plot_linear and plt is not None:
        print(f"{OK} Linear plot: {args.out}_linear.png")
    print(f"{OK} Summary: {txt_path}")
    zmap = {90: 1.6449, 95: 1.9599, 99: 2.5758}
    z = zmap.get(args.ci, 1.9599)
    ci_lo = fit.slope - z * fit.slope_stderr
    ci_hi = fit.slope + z * fit.slope_stderr
    msg = f"{RES} D = {Fore.MAGENTA}{fit.slope:.6f}{Style.RESET_ALL} ± {fit.slope_stderr:.6f} ({args.ci}% CI {ci_lo:.6f}..{ci_hi:.6f}, R^2={fit.r2:.4f}, points={fit.n})"
    if boot:
        slopes = np.array([b.slope for b in boot], dtype=float)
        mu = float(np.mean(slopes))
        sd = float(np.std(slopes, ddof=1)) if len(slopes) > 1 else 0.0
        lo_q, hi_q = (0.05, 0.95) if args.ci == 90 else ((0.005, 0.995) if args.ci == 99 else (0.025, 0.975))
        lo_b = float(np.quantile(slopes, lo_q))
        hi_b = float(np.quantile(slopes, hi_q))
        msg += f"\n{INFO} Bootstrap D: mean={mu:.6f}, sd={sd:.6f}, {args.ci}% CI {lo_b:.6f}..{hi_b:.6f} (B={len(slopes)})"
    print(msg)

    try:
        meta = {
            "image": args.image,
            "out": args.out,
            "seed": int(args.seed),
            "args": vars(args),
            "window": {"k0": int(k0), "k1": int(k1)},
            "D": float(fit.slope),
            "SE": float(fit.slope_stderr),
            "CI": {"level": int(args.ci), "lo": float(lo), "hi": float(hi)},
            "R2_unweighted": float(fit.r2),
            "scales_used": [int(v) for v in s_px.tolist()],
        }
        with open(args.out + ".json", "w", encoding="utf-8") as fj:
            json.dump(meta, fj, indent=2)
    except Exception:
        pass


if __name__ == "__main__":
    main()
