# Box Counting Fractal Dimension with uncertainty calculation

![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)

TL;DR
- Binary (black/white) images: put your file in `in/` and run `make analyze_in`. Results land in `out/results/<name>/` (plots, CSV/TXT, and `grids/`). Open `out/results/<name>/<name>.png`.
- Grayscale images: try preprocessing — run `make analyze_in PREP=true`. Then check `out/results/<name>/grids/` to see if the cleaned mask/outline looks right. If not, tweak cleaner knobs (see “Preprocess” below) or do it manually in an image editor of choice, rerun, and compare again.

- Play around: you won’t break anything!

Scholarly Use Notice
- Scholarly use requires prior notice and citation. Read `ACADEMIC_USE_LICENSE.md`.
- Contact: anton.kabasi.gm@gmail.com
- Cite using `CITATION.cff`.

Estimate the (area) box-counting fractal dimension from images, with a focus on electron microscope and atomic force microscope microstructures. Tested to work on famous fractals, and 2D primitives - results are consistent with literature. 

Includes helpers and tests for simple primitives (line, rectangle, sine, circle) and a classic fractal (Sierpiński triangle).

✨ Features
- Otsu or fixed thresholding (no heavy deps)
- Grid-origin averaging to reduce bias
- Geometric ladder of box sizes
- CSV results and a TXT summary (R², fit points)
- Optional log–log PNG plot and linear plot (N vs 1/ε)
- Uncertainty bands (bootstrap over grid offsets) and fit stderr

📏 Grid Origins (Offsets)
- For each box size `s`, the counting grid can start at any integer origin `(ox, oy)` with `ox∈[0, s)`, `oy∈[0, s)`.
- We average counts over K distinct origins per scale to reduce placement bias. K is set by `--grid-averages` (default `4`). That means:
  - Per scale `s` we perform K "grid moves" (K different `(ox, oy)` placements) and count boxes for each.
  - We compute the sample mean `N_mean(s)` and sample standard deviation `N_std(s)` across these K placements.
- In the bootstrap, each replicate uses `--boot-grid-averages` Kb origins per scale (defaults to K). We redraw these origins independently every replicate.
- Practical guidance: K=4–8 is a good default; Kb should usually match K. Increasing them improves stability but costs time.

🧩 Methodology at a Glance
- For each geometric ladder of box sizes `s`:
  1) Do K grid moves `(ox, oy)` per scale; record K counts; compute `N_mean(s)` and `N_std(s)`.
  2) Build regression data in the log domain: `x = log(1/ε)`, `y = log N_mean(s)` (with `ε=s` unless `--pixel-size` is given).
  3) Fit slope `D` on a chosen window (drop_head/tail) via weighted least squares (WLS) with weights `w_i = 1 / Var[log N_i]` where
     - `se_N(s) = N_std(s)/sqrt(K)` (standard error of the mean across K placements)
     - `se_log(s) ≈ se_N(s) / N_mean(s)` (delta method) → `Var[log N(s)] ≈ se_log(s)^2`.
  4) Report the fit with:
     - `D` and its WLS standard error (fit CI, line-mean uncertainty)
     - Error bars on points: in log plot `yerr = se_log`, in linear plot `yerr = se_N`.
  5) Measurement uncertainty (bootstrap) with `B` replicates:
     - For each replicate, redraw Kb placements per scale, recompute `N_mean_b(s)` and `y_b(s)=log N_mean_b(s)` and refit WLS on the same window.
     - Prediction interval (PI) on log plot: for each observed `x`, take percentiles of `{y_b(x)}` across replicates; the shaded band is the per‑x envelope; dashed line is the bootstrap median.
     - The linear plot shows the exponentiated PI (multiplicative band).

🔧 Defaults in Make vs CLI
- Makefile defaults (examples and your runs): `--grid-averages 4`, `--bootstrap 50`, `--band prediction` and, when `PREP=true`, padding `PAD_ON_PREP=8` is applied before counting.
- CLI defaults: `--bootstrap 0` (off), `--band prediction`, `--ci 95`. You can override any of these.

🧠 Interpreting the visuals
- Error bars are SEM-based (uncertainty of the mean over K placements). The shaded band is a bootstrap prediction interval (variability across freshly redrawn placements). As a result, bars should typically lie within the band at the chosen level.
- The dashed bootstrap median line does not have to coincide with the single WLS fit line; that difference reflects actual placement variability, not a plotting error.
- Padding (`--pad N` or `PAD_ON_PREP`) helps reduce edge bias when moving grids; it does not alter the structure, only adds whitespace.

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

🚦 Typical Run (PREP=true)
- Put your images in `in/` (e.g., `in/sample.png`).
- Run preprocessing + analysis with padding and bootstrap prediction bands:
  - `make analyze_in PREP=true`  # cleans all `in/*.png|jpg|jpeg|tif|tiff` and analyzes
  - or a single file: `make analyze_image IMAGE=in/sample.png PREP=true`
- What you get per image under `out/results/<name>/`:
  - `<name>.png` (log–log) with fit line, error bars, and a bootstrap prediction interval band
  - `<name>_linear.png` (N vs 1/ε) with error bars and the exponentiated PI band (no fit line)
  - `<name>.csv` (per-scale counts, logs, and weights)
  - `<name>.txt` (fit stats + uncertainty)
  - `grids/` (grid-overlay images for quick sanity checks)
  - Adjust on the command line if needed:
  - Padding: `PAD_ON_PREP=16`  •  Bootstrap reps: `--bootstrap 200`  •  Grid moves per scale: `--grid-averages 8`

📚 Glossary (One‑liners)
- `ε` (epsilon): box size; defaults to pixels unless `--pixel-size` is given.
- `s`: integer box size in pixels for counting; `ε=s` by default.
- Grid origin `(ox, oy)`: upper‑left corner of a counting grid; each in `[0, s)`.
- `K`: grid moves per scale used for the main estimate (`--grid-averages`, default 4).
- `Kb`: grid moves per scale per bootstrap replicate (`--boot-grid-averages`, default = K).
- `B`: bootstrap replicates (`--bootstrap`, Make defaults to 50; CLI default 0).
- SEM: standard error of the mean across `K` placements (linear: `se_N = N_std/√K`; log: `se_log ≈ se_N/N_mean`).
- WLS: weighted least squares on `y = log N_mean` vs `x = log(1/ε)` using weights `w=1/Var[log N]`.
- CI: confidence interval for the fitted line (mean response).
- PI: prediction interval for observations (new grid placements); shaded band on plots.

✅ Recommended Defaults (Cheatsheet)
- Quick looks: `K=4`, `B=50` (Make default), pad `8` with PREP.
- Publication quality: `K=6–8`, `Kb=K`, `B=200–500`, pad `8–16`.
- Fit window: `drop_head=1–2`, `drop_tail=1–2` (inspect residual curvature).
- Band: prediction (`--band prediction`, default); leave `--ci 95`.

🧭 Choosing the Fit Window
- Drop the head where the smallest scales deviate from linearity (thresholding/pixel effects).
- Drop the tail where counts saturate/flatten at large boxes.
- Use R² and residuals in the chosen window; aim for visually stable slope across nearby windows.

🩺 Troubleshooting
- Foreground fraction 0%/100%: adjust `--threshold/--invert`; try PREP (`make analyze_in PREP=true`).
- Points outside the band: increase `K` (grid moves), `B` (replicates), or check that `--band prediction` is used.
- Band looks tilted vs fit line: expected (bootstrap median vs single WLS); use `--band fit` if you need a line CI instead.
- Geography shapes too coarse: provide `SOURCE`/`SOURCE_URL` to a richer dataset.
- Wrong polarity (black/white): use `CLEAN_OUT_FG=white|black` or `CLEAN_INVERT=true|false` in PREP.
- Slow runs: reduce `B` and/or `K`; fewer scales (`--scales`) or smaller images help.

⚙️ Performance & Reproducibility
- Complexity ~ `O(B × K × #scales × image_area/s²)` (per scale). Start small; scale up for finals.
- Determinism: set `--boot-seed` and keep inputs/params fixed. PREP saves intermediates; reruns are consistent.

🤝 Contributing
- Tests: `make test` (unit tests for primitives, outputs, cleaner, geography generator).
- Code layout: CLI in `scripts/`, generators in `tests/`, examples/assets in `assets/`.
- PRs/issues: include commands, params, and a small input that reproduces your observation.
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
  - Measurement uncertainty (default in Make): bootstrap across grid offsets (for each box size s, re‑draw grid origins ox∈[0,s), oy∈[0,s) and recompute counts). The band on the log–log plot is a bootstrap prediction interval (PI): for each x=log(1/ε) we take percentiles of the bootstrapped log N values at that x.
- Why per‑scale offset sampling: the box count depends on how the grid is placed. For each scale, we move the grid around, so the resulting D reflects variability from placement. With smaller boxes (small s), the offset range [0,s) is smaller and distinct placements are fewer; the uncertainty tends to shrink compared to larger s (but image structure matters).
- Controls (CLI):
  - `--bootstrap B` (replicates; 0=off), `--boot-grid-averages K` (offsets per scale per replicate; defaults to `--grid-averages`), `--boot-seed` (reproducible randomness), `--ci` (90/95/99), `--band prediction|fit` (default prediction).
- Make defaults enable `--bootstrap 50` for examples and your runs; change in the Makefile or pass flags explicitly.

🔬 Uncertainty Details (Step‑by‑Step)
- Data we fit: for each geometric box size `s` we estimate a mean count `N_mean(s)` by averaging over several grid origins `(ox, oy)` and compute its spread `N_std(s)` across those origins. We then work in log domain:
  - `x = log(1/ε)` with `ε = s` (or physical size if `--pixel-size` was given)
  - `y = log N_mean(s)`

- Per‑point uncertainty (used for error bars and weights):
  - We compute the standard error of the mean across K grid origins per scale: `se_N = N_std / sqrt(K)`.
  - Delta method for log: `se_log ≈ se_N / N_mean`.
  - Error bars in the log–log plot show `yerr = se_log`; in the linear plot they show `yerr = se_N`.

- Weighted least squares (fit stderr): we estimate the slope `D` by weighted linear regression of `y` on `x` in the chosen fit window (drop_head/tail):
  - Weights: `w_i = 1 / max(Var[log N_i], 1e-8)`
  - From the WLS normal equations `(XᵀWX)^{-1}`, we compute standard errors:
    - `StdErr(D) = sqrt(σ² / Σ w_i (x_i − x̄_w)^2)` with `σ² = RSS_w / (n−2)`
  - We report `D ± z·StdErr(D)` as a fit‑based CI (z = 1.645, 1.960, 2.576 for 90/95/99%).

- Measurement uncertainty (bootstrap over grid origins): captures how `D` varies as we move the grid per scale.
  1) For each replicate `b = 1..B`:
     - For each scale `s`, draw `K` fresh offsets `(ox, oy)` uniformly in `[0, s)` and recompute `N_mean(s), N_std(s)`.
     - Build `x_b, y_b` as above and run a WLS fit on the same fit window → get `D_b, intercept_b`.
  2) Aggregate replicate fits:
     - Distribution of `{D_b}` gives a bootstrap mean/SD and a percentile CI (e.g., 2.5–97.5% for 95%).
     - To draw a band on the log–log plot, we compute a prediction interval: at each observed x point (all scales), we take percentiles of the bootstrapped `y_b(x)` values and fill between the lower/upper curves. The dashed line is the bootstrap median.

- Important nuances:
  - The shaded band represents measurement variability; it is not forced to be centered on the single displayed WLS line (they answer different questions). If you prefer cosmetic centering, we can add an option to recenter the envelope at the main fit.
  - Padding reduces edge bias when moving grids: use `--pad N` on the CLI (default 0). In the Makefile, when `PREP=true`, padding is applied by default via `PAD_ON_PREP` (default `8`).
  - The trend of uncertainty with scale is data‑dependent: small `s` has more boxes (often stabilizing counts), but the offset domain `[0,s)` is smaller; the bootstrap samples both effects by re‑drawing offsets per scale.

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
