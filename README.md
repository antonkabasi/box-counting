# Box Counting Fractal Dimension with Uncertainty

![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)

## TL;DR
- **Binary (black/white)**: put your file in `in/` and run `make analyze_in`. Results land in `out/results/<name>/` (plots, CSV/TXT, and `grids/`). Open `out/results/<name>/<name>.png`.
- **Grayscale/colored**: try preprocessing with `make analyze_in PREP=true`. Inspect:
  - `out/results/<name>/preprocessed.png` (the exact image that will be analyzed)
  - `out/results/<name>/grids/` (grid overlays for sanity checks)
  If it doesn’t look right, tweak cleaner knobs (see **Preprocess**) or clean manually and rerun.
- **Compare both modes**: `make all` runs **without** prep and **with** prep, saving to separate folders:
  - No-prep: `out/examples_noprep/**` and `out/results/noprep/**`
  - Prep: `out/examples_prep/**` and `out/results/prep/**`

---

## Scholarly Use Notice
- Scholarly use requires prior notice and citation. Read `ACADEMIC_USE_LICENSE.md` (see end).
- Contact: **anton.kabasi.gm@gmail.com**
- Cite using `CITATION.cff`.

This tool estimates the (area) **box-counting fractal dimension** from images, with a focus on electron/AFM microstructures. It’s tested on classic fractals and 2D primitives; results match the literature within tolerances.

It includes helpers and tests for simple primitives (line, rectangle, sine, circle) and the Sierpiński triangle.

---

## ✨ Features
- Otsu or fixed thresholding (no heavy deps)
- Grid-origin averaging to reduce placement bias
- Geometric ladder of box sizes
- CSV results and a TXT summary (R², fit points)
- Optional log–log PNG + linear PNG (N vs 1/ε)
- Uncertainty: fit stderr + bootstrap bands over grid offsets
- Grid overlays for visual checks  
  - **PREP=false**: overlays drawn over the **raw image**  
  - **PREP=true**: overlays drawn over a light **edge outline** (easier to see filled shapes)

---

## 📏 Grid Origins (Offsets)
- For each box size `s`, the grid can start at `(ox, oy)` with `ox∈[0,s)`, `oy∈[0,s)`.
- We average over **K** origins per scale (`--grid-averages`, default **4**):
  - For each scale `s`, place K grids, count boxes, compute `N_mean(s)` and `N_std(s)`.
- Bootstrap uses **Kb** origins per scale per replicate (`--boot-grid-averages`, default = K). Offsets are redrawn independently each replicate.
- **Rule of thumb**: K=4–8, Kb=K. Higher values stabilize results (at higher cost).

---

## 🧮 How It Works (short math)
- Box counting: `N(ε) ≈ C·ε^{-D}` ⇒ with `x = log(1/ε)` and `y = log N(ε)`, the slope `D` is the dimension.
- Per-scale uncertainty (K placements):
  - `se_N = N_std / √K`, and by delta method `se_log ≈ se_N / N_mean`.
- Weighted least squares (WLS) on a **fit window**:
  - Weights `w = 1/Var[log N]`. We report slope `D̂`, stderr, R², and #points.
- Bootstrap prediction interval (PI):
  - Redraw K offsets per scale for B replicates; form `log N` curves and take percentiles at each observed `x` to get the shaded band (dashed = bootstrap median).

---

## 🔧 Defaults (Make vs CLI)
- **Makefile** (examples & your runs): `--grid-averages 4`, `--bootstrap 50`, `--band prediction`. When `PREP=true`, we also pad the analysis window (`PAD_ON_PREP=8`).
- **CLI**: `--bootstrap 0` (off), `--band prediction`, `--ci 95`. Override anything as needed.

**Overlays** are saved under `.../grids/`. Mode is **raw** when PREP=false and **edge** when PREP=true.

---

## 🚀 Quick Start
- Python **3.9+** recommended.

### Install deps
```bash
make deps
```

### Generate sample primitives
```bash
make generate
```

### Analyze your images
- Drop images into `in/` and run:
```bash
make analyze_in
# or a single file, from anywhere:
make analyze_image IMAGE=/path/to/image.png
# positional form:
make analyze_image /path/to/image.png
```
- If `in/` is empty, we seed `assets/line.png` automatically.  
  `make seed_in` copies all `assets/*.png` to `in/`.

### Preprocess + analyze
```bash
make analyze_in PREP=true
# or
make analyze_image IMAGE=in/sample.png PREP=true
```
This also writes `out/results/<name>/preprocessed.png` (the image that gets analyzed).

### Run built-in examples
```bash
make examples
```
Writes to `out/examples/<name>/`:
- `<name>.png` (log–log)
- `<name>_linear.png` (N vs 1/ε)
- `<name>.csv` (per-scale data)
- `<name>.txt` (fit summary)
- `grids/` overlays

### Compare without/with preprocessing
```bash
make all
```
- No-prep outputs: `out/examples_noprep/**`, `out/results/noprep/**`  
- Prep outputs: `out/examples_prep/**`, `out/results/prep/**`

### Open plots quickly
```bash
make show
```

---

## 🧰 CLI
- Analyzer: `scripts/boxcount.py`  
- Preprocess helper: `scripts/clean_image.py`

**Example:**
```bash
.venv/bin/python scripts/boxcount.py \
  --image assets/line.png \
  --out out/examples/line/line \
  --threshold fixed --fixed-thresh 128 --invert --crop \
  --min-box 2 --max-box 128 --scales 11 --grid-averages 4 \
  --drop-head 2 --drop-tail 1 --bootstrap 50 --ci 95 \
  --seed 12345 --auto-window --band prediction --plot --plot-linear
```

---

## 📦 Inputs & Outputs
- **Inputs:** any Pillow-readable image. Thresholding yields a binary mask (foreground = structure). Samples: `assets/` (`line.png`, `rectangle.png`, `circle.png`, `sine.png`, `sierpinski.png`).
- **Outputs:**
  - Examples → `out/examples/<name>/<name>.{csv,txt,png,_linear.png}`
  - Your runs → `out/results/<name>/<name>.{csv,txt,png,_linear.png}`
    - Grid overlays → `out/results/<name>/grids/grid_s<size>_ox<ox>_oy<oy>.png`
    - Optional run tag: `make analyze_in RUN=2025-09-13` → `out/results/2025-09-13/<name>/<name>.*`

---

## 🧹 Preprocess (Clean) Inputs
Use `scripts/clean_image.py` to convert diverse inputs (transparent PNGs, colored lines, filled shapes) into comparable masks/outlines.

**Examples**
```bash
# Outline via Otsu + invert + light closing
.venv/bin/python scripts/clean_image.py \
  --in in/uk.png --out in/uk_clean.png \
  --alpha-bg white --threshold otsu --invert --close 1 --borderize 1 --crop --out-fg white

# Extract red lines on white
.venv/bin/python scripts/clean_image.py \
  --in in/red_curve.png --out in/red_curve_clean.png \
  --target-color red --color-delta 40 \
  --threshold fixed --fixed-thresh 32 --borderize 1 --out-fg black
```

Then analyze the cleaned image:
```bash
make analyze_image IMAGE=in/uk_clean.png
```

**Make integration**
```bash
make analyze_image IMAGE=in/file.png PREP=true
make analyze_in PREP=true
```
Cleaner knobs (booleans accept true/false, yes/no, on/off, 1/0):
```
CLEAN_ALPHA_BG=white CLEAN_TARGET_COLOR=auto CLEAN_COLOR_DELTA=40
CLEAN_THRESHOLD=otsu   CLEAN_FIXED_THRESH=128  CLEAN_INVERT=true
CLEAN_OPEN=0 CLEAN_CLOSE=0 CLEAN_BORDERIZE=1 CLEAN_CROP=true CLEAN_OUT_FG=white|black
```
When `PREP=true`, padding is applied before analysis: `PAD_ON_PREP=8` (pixels on all sides).

---

## 🚦 Typical Run (PREP=true)
```bash
# put images in in/
make analyze_in PREP=true
# or single file
make analyze_image IMAGE=in/sample.png PREP=true
```
Per image you get:
- `<name>.png` (log–log) with fit + error bars + bootstrap PI band
- `<name>_linear.png` (N vs 1/ε) with error bars + exp(PI) band
- `<name>.csv` (per-scale counts/logs/weights)
- `<name>.txt` (fit stats + uncertainty)
- `preprocessed.png` (what was analyzed)
- `grids/` overlays for quick checks

Adjust via CLI flags or Make vars:
- Padding `PAD_ON_PREP=16`
- Bootstrap reps `--bootstrap 200`
- Grid moves per scale `--grid-averages 8`

---

## 📚 Glossary (one-liners)
- `ε` (epsilon): box size; equals pixels unless `--pixel-size` is given.
- `s`: integer box size (px) for counting (`ε=s` by default).
- Grid origin `(ox, oy)`: upper-left corner of a grid; each in `[0, s)`.
- `K`: grid moves per scale (`--grid-averages`, default 4).
- `Kb`: grid moves per scale per bootstrap replicate (`--boot-grid-averages`, default = K).
- `B`: bootstrap replicates (`--bootstrap`; Make default 50; CLI default 0).
- SEM: standard error of the mean across `K` placements (`se_N = N_std/√K`; `se_log ≈ se_N/N_mean`).
- WLS: weighted least squares on `y = log N_mean` vs `x = log(1/ε)` with `w=1/Var[log N]`.
- CI: confidence interval for the fitted line (mean response).
- PI: prediction interval for new observations (new grid placements).

---

## ✅ Recommended Defaults (cheatsheet)
- Quick looks: `K=4`, `B=50`, pad `8` with PREP.
- Publication: `K=6–8`, `Kb=K`, `B=200–500`, pad `8–16`.
- Fit window: `drop_head=1–2`, `drop_tail=1–2`.
- Band: `--band prediction` (default); use `--band fit` for line CI.

---

## 🧭 Choosing the Fit Window
- Drop small scales where pixel/threshold effects bend the curve.
- Drop large scales where counts saturate/flatten.
- Use R² and residuals; aim for a slope stable across nearby windows.

---

## 🩺 Troubleshooting
- Foreground fraction ~0% or ~100%: adjust `--threshold/--invert`; try `PREP=true`.
- Points outside the band: increase `K`, `B`, or confirm `--band prediction`.
- Band tilted vs fit line: expected (bootstrap median vs single WLS line). Use `--band fit` for a CI around the line.
- Geography too coarse: supply a better dataset via `SOURCE`/`SOURCE_URL`.
- Wrong polarity (black/white): set `CLEAN_OUT_FG=white|black` or `CLEAN_INVERT=true|false`.
- Slow: reduce `B`, `K`, `--scales`, or image size.

---

## ⚙️ Performance & Reproducibility
- Complexity ≈ `O(B × K × #scales × image_area/s²)` (per scale).
- Determinism: set `--boot-seed` and keep inputs/params fixed. When `PREP=true`, intermediates are saved; reruns are consistent.

---

## 🤝 Contributing
- Tests: `make test`
- Code layout: CLI in `scripts/`, generators in `tests/`, examples/assets in `assets/`.
- PRs/issues: include commands, params, and a small input that reproduces your observation.

**Clean only (no analysis):**
```bash
make clean_image IMAGE=in/file.png [OUT=in/file_clean.png] [CLEAN_* vars...]
```

---

## 🧪 Primitives / Tests
Generated by `tests/generate_images.py`:
- `assets/line.png` (thin diagonal)
- `assets/sine.png` (thin sine curve)
- `assets/rectangle.png` (filled rectangle)
- `assets/circle.png` (filled circle)
- `assets/sierpinski.png` (Sierpiński triangle, depth=7)

`make examples` runs these through the pipeline.

---

## 📈 Automated Tests
```bash
make test
```
- Validates estimated dimensions within tolerances:
  - Line & sine ≈ 1.0
  - Rectangle & circle ≈ 2.0
  - Sierpiński ≈ 1.585

---

## 📊 Uncertainty (how it’s computed)
Two sources are reported:

1. **Fit stderr** — standard error of the slope from WLS on `log N` vs `log(1/ε)` within the chosen window. Reported as  
   `D ± z·StdErr(D)` with z = 1.645/1.960/2.576 for 90/95/99%.

2. **Measurement uncertainty (bootstrap over grid origins)** — captures how `D` varies as we move the grid per scale.  
   For each replicate:
   - Draw `Kb` offsets `(ox, oy)∈[0,s)×[0,s)` per scale.
   - Recompute `N_mean(s)`, build `log N` and refit on the same window.
   - The distribution of `{D_b}` yields a bootstrap mean/SD and percentile CI.  
   The log–log band is a **prediction interval**: at each observed `x`, take percentiles of `{log N_b(x)}` and fill between curves (dashed = bootstrap median).

**Notes**
- The shaded band represents observation variability; it needn’t be centered on the single WLS line.
- Padding (`--pad N` or `PAD_ON_PREP`) reduces edge bias when moving grids.

---

## 🧹 Repository Hygiene
- Example PNGs are kept.
- Analysis artifacts are not: `out/` and `.csv` / plots are ignored by git.

---

## 🛠️ Make Targets
- `make deps` — create `.venv/` and install `requirements.txt`
- `make generate` — create primitive images in `assets/`
- `make examples` — run box counting on primitives → `out/examples/`
- `make analyze_image IMAGE=path/to/image` (or positional form) — analyze one image → `out/results/<name>/<name>.*` + `grids/`
- `make analyze_in` (optionally `RUN=tag`) — analyze every supported image under `in/` → `out/results[/tag]/<name>/<name>.*`
- `make all` — run both **no-prep** and **prep** pipelines (examples + `in/`) and save to separate folders
- `make dims_results` (`dims_real`) — print fitted D from summaries under `out/results[/tag]/`
- `make geography uk` (or `make geography-uk`) — generate a country silhouette and analyze it (results under `out/results/geography/uk/`)
  - With a richer dataset: `make geography uk SOURCE=assets/data/countries.geojson`
- `make show` — open the example plots
- `make line|sine|rectangle|circle|sierpinski` — run a single case
- `make dims` — print fitted D from example summaries
- `make clean` — remove outputs in `out/`
- `make distclean` — also remove generated images

---

## 📎 Notes
- Estimator: slope of `log N` vs `log(1/ε)` over a chosen window.
- For SEM/AFM, you may need `--invert` if features are dark on bright background.
- Tune `--drop-head` / `--drop-tail` to capture the most linear region.
- Geography samples include a coarse placeholder; provide `SOURCE`/`SOURCE_URL` for proper coastlines.

---

## 📝 License
- Code license: **BSD 3-Clause**, see `LICENSE`.
- **Academic Use Addendum**: For scholarly publications, see `ACADEMIC_USE_LICENSE.md`.
  - Prior notice before submission and citation are required.
  - Co-authorship is expected when the Author provides substantial additional intellectual contributions (new features, methods, analysis design, interpretation).
  - If you cannot accept the addendum, contact the Author for a separate license.

---

## ✉️ Prior Notice Template (for Scholarly Use)

**Subject:** Prior notice of scholarly use — box-counting software

**Body:**
- Title (working): `<your paper title>`
- Venue and timeline: `<journal/conference>`, `<submission date>`
- Software use: `<briefly describe what was run and for which results/figures>`
- Modifications: `<any code changes or parameters>`
- Expected contributions from author (if any): `<e.g., methods advice, new features>`
- Authorship plan: `<acknowledgment or proposed co-authorship>`
- Contact person: `<name, affiliation, email>`