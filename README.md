# 🌈 Box Counting Fractal Dimension

Estimate the (area) box-counting fractal dimension from images, with a focus on electron microscope microstructures. Includes helpers and tests for simple primitives (line, rectangle, sine, circle) and a classic fractal (Sierpiński triangle).

✨ Features
- Otsu or fixed thresholding (no heavy deps)
- Grid-origin averaging to reduce bias
- Geometric ladder of box sizes
- CSV results and a TXT summary (R², fit points)
- Optional log–log PNG plot and linear plot (N vs 1/ε)
 - Uncertainty bands (bootstrap over grid offsets) and fit stderr

🚀 Quick Start
- Python 3.9+ recommended.
- Create a virtual environment and install deps:
  - `make deps`
- Generate or view sample primitives (kept in `assets/`):
  - `make generate`
- Analyze your own images: drop them in the `in/` directory, then run:
  - `make analyze_in` (processes all `in/*.png|jpg|jpeg|tif|tiff`)
  - Or a single file from anywhere: `make analyze_image IMAGE=/path/to/image.png` or `make analyze_image /path/to/image.png`
  - If `in/` is empty, the Makefile seeds `assets/line.png` into `in/` automatically. You can also run `make seed_in` to copy all `assets/*.png` into `in/`.
  - Preprocess first: add `PREP=true` to auto-clean inputs before analysis (see below)
- Run analysis on built-in examples and write outputs under `out/examples/<name>/` (ignored by git). This writes for each example:
  - log–log plot: `out/examples/<name>/<name>.png`
  - linear plot: `out/examples/<name>/<name>_linear.png`
  - CSV: `out/examples/<name>/<name>.csv`
  - TXT summary: `out/examples/<name>/<name>.txt`
  - Grid overlays: `out/examples/<name>/grids/grid_s<size>_ox<ox>_oy<oy>.png`
  - Use: `make examples`
- Open the plots:
  - `make show`

🧰 CLI
- Script: `scripts/boxcount.py`
 - Preprocess helper: `scripts/clean_image.py` (normalizes varied inputs to a binary mask/outline)
- Example:
  - ``
    .venv/bin/python scripts/boxcount.py \
      --image assets/line.png \
      --out out/examples/line \
      --threshold fixed --fixed-thresh 128 --invert --crop \
      --min-box 2 --max-box 128 --scales 11 --grid-averages 4 \
      --drop-head 2 --drop-tail 1 --bootstrap 50 --ci 95 --plot
    ``

📦 Inputs & Outputs
- Input: any image readable by Pillow; thresholding yields a binary mask (foreground = structure). Sample images are in `assets/` (`line.png`, `rectangle.png`, `circle.png`, and generated `sine.png`).
- Outputs are structured by purpose:
  - Examples: `out/examples/<name>/<name>.{csv,txt,png,_linear.png}`
  - Your runs: `out/results/<name>/<name>.{csv,txt,png,_linear.png}`
    - Grid overlays per image: `out/results/<name>/grids/grid_s<size>_ox<ox>_oy<oy>.png`
    - Optional run tag: `make analyze_in RUN=2025-09-13` → `out/results/2025-09-13/<name>/<name>.*`

🧹 Preprocess (Clean) Inputs
- Use `scripts/clean_image.py` to convert diverse inputs (transparent PNGs, red-on-white drawings, filled shapes) into comparable masks/outlines.
- Examples:
  - Outline from any input with Otsu + invert + light closing:
    - `.venv/bin/python scripts/clean_image.py --in in/uk.png --out in/uk_clean.png --alpha-bg white --threshold otsu --invert --close 1 --borderize 1 --crop --out-fg white`
  - Extract red lines on white:
    - `.venv/bin/python scripts/clean_image.py --in in/red_curve.png --out in/red_curve_clean.png --target-color red --color-delta 40 --threshold fixed --fixed-thresh 32 --borderize 1 --out-fg black`
- Then analyze the cleaned image with `make analyze_image in/<file>`, results under `out/results/<name>/…`
- Make integration:
  - `make analyze_image IMAGE=in/file.png PREP=true` (auto-cleans to `out/results/file/preprocessed.png` then analyzes; boxcount auto-configures invert based on `CLEAN_OUT_FG` and applies padding by default)
  - `make analyze_in PREP=true` (auto-cleans every image under `in/` before analysis; padding is applied by default)
  - Tweak cleaner via variables (defaults shown; booleans accept true/false, yes/no, on/off, 1/0):
    - `CLEAN_ALPHA_BG=white CLEAN_TARGET_COLOR=auto CLEAN_COLOR_DELTA=40`
    - `CLEAN_THRESHOLD=otsu CLEAN_FIXED_THRESH=128 CLEAN_INVERT=true`
    - `CLEAN_OPEN=0 CLEAN_CLOSE=0 CLEAN_BORDERIZE=1 CLEAN_CROP=true CLEAN_OUT_FG=white|black`
  - Analysis padding when PREP=true (defaults):
    - `PAD_ON_PREP=8` (pixels added on all sides before counting)
  - Clean only (no analysis):
    - `make clean_image IMAGE=in/file.png [OUT=in/file_clean.png] [CLEAN_* vars...]`

🧪 Primitives / Tests
- Generate baseline images via `tests/generate_images.py`:
  - `assets/line.png` (thin diagonal)
  - `assets/sine.png` (thin sine curve)
  - `assets/rectangle.png` (filled rectangle)
  - `assets/circle.png` (filled circle)
  - `assets/sierpinski.png` (Sierpiński triangle, depth=7)
- The `Makefile` targets use these for quick checks, and `make examples` runs all of them.

🧫 Automated Tests
- Run unit tests: `make test`
- Tests generate primitives and validate estimated dimensions within tolerances:
  - Line and sine ≈ 1.0, rectangle and circle ≈ 2.0, Sierpiński ≈ 1.585.

📈 Uncertainty (How It’s Computed)
- Two sources are reported:
  - Fit stderr: standard error of the fitted slope from (weighted) least squares on log N vs log(1/ε).
  - Measurement uncertainty (default in Make): bootstrap across grid offsets (for each box size s, re‑draw grid origins ox∈[0,s), oy∈[0,s) and recompute counts). The band on the log–log plot is the percentile envelope over bootstrap replicates.
- Why per‑scale offset sampling: the box count depends on how the grid is placed. For each scale, we move the grid around, so the resulting D reflects variability from placement. With smaller boxes (small s), the offset range [0,s) is smaller and distinct placements are fewer; the uncertainty tends to shrink compared to larger s (but image structure matters).
- Controls (CLI):
  - `--bootstrap B` (replicates; 0=off), `--boot-grid-averages K` (offsets per scale per replicate; defaults to `--grid-averages`), `--boot-seed` (reproducible randomness), `--ci` (90/95/99 for bands and printed CIs).
- Make defaults enable `--bootstrap 50` for examples and your runs; change in the Makefile or pass flags explicitly.

🧹 Repository Hygiene
- Example PNGs are kept under version control.
- Analysis artifacts are not: `out/` is ignored, as well as any `.csv` and `*_loglog.png` files.

🛠️ Make Targets
- `make deps`: create `.venv/` and install `requirements.txt`
- `make generate`: create primitive test images in `assets/`
- `make examples`: run box counting on primitives; writes to `out/examples/`
- `make analyze_image IMAGE=path/to/image` or `make analyze_image path/to/image`: analyze a single image → `out/results/<name>/<name>.*` and grids in `out/results/<name>/grids/`
- `make analyze_in` (optionally with `RUN=tag`): analyze every supported image under `in/` → `out/results[/tag]/<name>/<name>.*` and grids
- `make dims_results` (or legacy `dims_real`): print fitted D from summaries under `out/results[/tag]/`
- `make geography uk` (space-separated): generate a country silhouette via GeoPandas and analyze it (results under `out/results/geography/uk/`)
  - Also works as `make geography-uk` (hyphenated)
  - For realistic coastlines, supply a dataset path: `make geography uk SOURCE=assets/data/countries.geojson`
    - SOURCE accepts any local GeoJSON/GeoPackage/Shapefile readable by GeoPandas
    - Without SOURCE, a bundled placeholder dataset is used (coarse rectangles for demo only)
- `make show`: open the plots for quick visual check
- `make line|sine|rectangle|circle|sierpinski`: run a single case
- `make dims`: print fitted D from summaries
- `make clean`: remove outputs in `out/` and `out/examples/`
- `make distclean`: also remove generated images

📎 Notes
- The estimator is standard box-counting: slope of log N vs log(1/ε) over a configurable window.
- For SEM images, you might need `--invert` if features are dark on bright background.
- Tune `--drop-head` / `--drop-tail` to focus on the roughly linear region.
- Geography: the repo includes a minimal offline placeholder (very coarse). For realistic coastlines, provide `SOURCE` or `SOURCE_URL` to a richer dataset; see Make target docs above.

📝 License
- Code license: BSD 3‑Clause (OSI‑approved), see `LICENSE`.
- Academic Use Addendum: For use in scholarly publications, see `ACADEMIC_USE_LICENSE.md`.
  - Prior notice before submission and citation are required.
  - Co‑authorship is expected when the Author provides substantial additional
    intellectual contributions (new features, methods, analysis design, interpretation).
  - If you cannot accept the addendum terms, contact the Author for a separate license.
- Contact: anton.kabasi.gm@gmail.com

✉️ Prior Notice Template (for Scholarly Use)
Please email the following to anton.kabasi.gm@gmail.com at least 14 days before submission:

Subject: Prior notice of scholarly use — box-counting software

Body:
- Title (working): <your paper title>
- Venue and timeline: <journal/conference>, <submission date>
- Software use: <briefly describe what was run and for which results/figures>
- Modifications: <any code changes or parameters>
- Expected contributions from author (if any): <e.g., methods advice, new features>
- Authorship plan: <acknowledgment or proposed co-authorship>
- Contact person: <name, affiliation, email>
