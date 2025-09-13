# Makefile for running box-count tests (line, sine, rectangle)
# Usage:
#   make deps        # install Python deps (numpy, pillow, pandas, matplotlib)
#   make generate    # create test images
#   make analyze     # run box-count + save CSV/PNG/TXT
#   make show        # open the generated log–log plot PNGs
#   make line        # generate & analyze just the line case (opens the PNG)
#   make sine        # generate & analyze just the sine case (opens the PNG)
#   make rectangle   # generate & analyze just the rectangle case
#   make dims        # print fitted D from summaries
#   make clean       # remove outputs
#   make distclean   # also remove generated images

VENV      := .venv
PY        := $(VENV)/bin/python
PIP       := $(VENV)/bin/pip
IMAGEDIR  := assets
OUTDIR    := out
RESULTS_OUTDIR := $(OUTDIR)/results
RESULTS_OUT_PREFIX := $(if $(RUN),$(RESULTS_OUTDIR)/$(RUN),$(RESULTS_OUTDIR))
EX_OUTDIR := $(OUTDIR)/examples
GEODIR    := $(IMAGEDIR)/geography
EX_GEO_DIR:= $(EX_OUTDIR)/geography
RES_GEO_DIR:= $(RESULTS_OUTDIR)/geography
GEO_DATA  := assets/data/countries_simple.geojson
SRC_DATA  := $(if $(SOURCE),$(SOURCE),$(GEO_DATA))
SRC_URL   := $(if $(SOURCE_URL),$(SOURCE_URL),https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson)

IMAGES    := $(IMAGEDIR)/line.png $(IMAGEDIR)/sine.png $(IMAGEDIR)/rectangle.png $(IMAGEDIR)/circle.png $(IMAGEDIR)/sierpinski.png
PLOTS     := $(EX_OUTDIR)/line/line.png $(EX_OUTDIR)/sine/sine.png $(EX_OUTDIR)/rectangle/rectangle.png $(EX_OUTDIR)/circle/circle.png $(EX_OUTDIR)/sierpinski/sierpinski.png

.PHONY: all venv deps generate examples analyze show line sine rectangle dims clean distclean seed_in

all: deps generate examples

venv:
	@test -d $(VENV) || python3 -m venv $(VENV)
	@$(PIP) -q install --upgrade pip setuptools wheel

deps: venv requirements.txt
	@$(PIP) -q install -r requirements.txt
	@echo "[ok] dependencies installed in $(VENV)/"

$(IMAGEDIR):
	@mkdir -p $(IMAGEDIR)

$(OUTDIR):
	@mkdir -p $(OUTDIR)

$(EX_OUTDIR): | $(OUTDIR)
	@mkdir -p $(EX_OUTDIR)

$(GEODIR): | $(IMAGEDIR)
	@mkdir -p $(GEODIR)

$(EX_GEO_DIR): | $(EX_OUTDIR)
	@mkdir -p $(EX_GEO_DIR)

$(RES_GEO_DIR): | $(RESULTS_OUTDIR)
	@mkdir -p $(RES_GEO_DIR)

INDIR := in
$(INDIR):
	@mkdir -p $(INDIR)

$(RESULTS_OUTDIR): | $(OUTDIR)
	@mkdir -p $(RESULTS_OUTDIR)

generate: $(IMAGES)
	@echo "[ok] test images generated in $(IMAGEDIR)/"

# Cleaning (preprocess) defaults (override on CLI). Booleans accept: true/false, yes/no, on/off, 1/0.
PREP ?= false
CLEAN_ALPHA_BG ?= white
CLEAN_TARGET_COLOR ?= auto
CLEAN_COLOR_DELTA ?= 40
CLEAN_THRESHOLD ?= otsu
CLEAN_FIXED_THRESH ?= 128
CLEAN_INVERT ?= true
CLEAN_OPEN ?= 0
CLEAN_CLOSE ?= 0
CLEAN_BORDERIZE ?= 1
CLEAN_CROP ?= true
CLEAN_OUT_FG ?= auto
# When PREP=true, add this many pixels of padding to analysis (boxcount --pad)
PAD_ON_PREP ?= 8

# Normalize booleans to 1/0
bool = $(if $(filter yes YES true TRUE on ON 1,$(strip $(1))),1,0)
PREP_BOOL := $(call bool,$(PREP))
CLEAN_INVERT_BOOL := $(call bool,$(CLEAN_INVERT))
CLEAN_CROP_BOOL := $(call bool,$(CLEAN_CROP))

# Compose cleaner args
# Resolve output foreground color: auto maps invert=true→white, invert=false→black
ifeq ($(strip $(CLEAN_OUT_FG)),auto)
  # For PREP=true, flip the default colors: produce black foreground on white.
  ifeq ($(PREP_BOOL),1)
    ifeq ($(CLEAN_INVERT_BOOL),1)
      CLEAN_OUT_FG_RES := black
    else
      CLEAN_OUT_FG_RES := white
    endif
  else
    ifeq ($(CLEAN_INVERT_BOOL),1)
      CLEAN_OUT_FG_RES := white
    else
      CLEAN_OUT_FG_RES := black
    endif
  endif
else
  CLEAN_OUT_FG_RES := $(CLEAN_OUT_FG)
endif

CLEAN_ARGS = --alpha-bg $(CLEAN_ALPHA_BG) --target-color $(CLEAN_TARGET_COLOR) --color-delta $(CLEAN_COLOR_DELTA) \
  --threshold $(CLEAN_THRESHOLD) --fixed-thresh $(CLEAN_FIXED_THRESH) --open $(CLEAN_OPEN) --close $(CLEAN_CLOSE) \
  --borderize $(CLEAN_BORDERIZE) --out-fg $(CLEAN_OUT_FG_RES)
ifeq ($(CLEAN_INVERT_BOOL),1)
  CLEAN_ARGS += --invert
endif
ifeq ($(CLEAN_CROP_BOOL),1)
  CLEAN_ARGS += --crop
endif

# If we preprocess (PREP=1), the cleaned image is already binary with
# foreground as white; do not invert in boxcount. Otherwise, keep --invert.
# Determine whether to pass --invert to boxcount based on PREP and output FG color
BOX_INV :=
ifeq ($(PREP_BOOL),1)
  ifneq ($(strip $(CLEAN_OUT_FG_RES)),white)
    BOX_INV := --invert
  endif
else
  BOX_INV := --invert
endif

# Pad only when PREP is enabled
BOX_PAD :=
ifeq ($(PREP_BOOL),1)
  BOX_PAD := --pad $(PAD_ON_PREP)
endif

# Image generation rules
$(IMAGEDIR)/line.png: tests/generate_images.py | $(IMAGEDIR) venv
	$(PY) tests/generate_images.py --line --outdir $(IMAGEDIR)

$(IMAGEDIR)/sine.png: tests/generate_images.py | $(IMAGEDIR) venv
	$(PY) tests/generate_images.py --sine --outdir $(IMAGEDIR)

$(IMAGEDIR)/rectangle.png: tests/generate_images.py | $(IMAGEDIR) venv
	$(PY) tests/generate_images.py --rectangle --outdir $(IMAGEDIR)

$(IMAGEDIR)/circle.png: tests/generate_images.py | $(IMAGEDIR) venv
	$(PY) tests/generate_images.py --circle --outdir $(IMAGEDIR)

$(IMAGEDIR)/sierpinski.png: tests/generate_images.py | $(IMAGEDIR) venv
	$(PY) tests/generate_images.py --sierpinski --outdir $(IMAGEDIR)

# Geography images (from Natural Earth via GeoPandas)
$(GEODIR)/%.png: tests/generate_geography.py | $(GEODIR) venv
	$(PY) tests/generate_geography.py --country $* --outdir $(GEODIR) --mode outline --line-width 3 --source $(SRC_DATA) --source-url $(SRC_URL) --crs auto

# Examples: runs boxcount.py, saves per-example folders under out/examples/<name>/<name>.*
examples: generate $(PLOTS)
	@echo "[ok] examples complete. See $(EX_OUTDIR)/{line,sine,rectangle,circle,sierpinski}/<name>.{csv,txt,png,_linear.png} and grids/"

# Back-compat alias
analyze: examples

# Produce main PNGs (recipes also create linear PNGs, CSV/TXT, and grids)
$(EX_OUTDIR)/line/line.png: $(IMAGEDIR)/line.png | $(EX_OUTDIR) venv
	@mkdir -p $(EX_OUTDIR)/line
	$(PY) scripts/boxcount.py \
	  --image $< \
	  --out $(EX_OUTDIR)/line/line \
	  --threshold fixed --fixed-thresh 128 --invert --crop \
	  --min-box 2 --max-box 128 --scales 11 --grid-averages 4 --bootstrap 50 \
	  --drop-head 2 --drop-tail 1 --plot --plot-linear --save-grids --grids-max-offsets 1

$(EX_OUTDIR)/sine/sine.png: $(IMAGEDIR)/sine.png | $(EX_OUTDIR) venv
	@mkdir -p $(EX_OUTDIR)/sine
	$(PY) scripts/boxcount.py \
	  --image $< \
	  --out $(EX_OUTDIR)/sine/sine \
	  --threshold fixed --fixed-thresh 128 --invert --crop \
	  --min-box 2 --max-box 128 --scales 11 --grid-averages 4 --bootstrap 50 \
	  --drop-head 2 --drop-tail 1 --plot --plot-linear --save-grids --grids-max-offsets 1

$(EX_OUTDIR)/rectangle/rectangle.png: $(IMAGEDIR)/rectangle.png | $(EX_OUTDIR) venv
	@mkdir -p $(EX_OUTDIR)/rectangle
	$(PY) scripts/boxcount.py \
	  --image $< \
	  --out $(EX_OUTDIR)/rectangle/rectangle \
	  --threshold fixed --fixed-thresh 128 --invert --crop \
	  --min-box 2 --max-box 128 --scales 11 --grid-averages 4 --bootstrap 50 \
	  --drop-head 2 --drop-tail 1 --plot --plot-linear --save-grids --grids-max-offsets 1

$(EX_OUTDIR)/circle/circle.png: $(IMAGEDIR)/circle.png | $(EX_OUTDIR) venv
	@mkdir -p $(EX_OUTDIR)/circle
	$(PY) scripts/boxcount.py \
	  --image $< \
	  --out $(EX_OUTDIR)/circle/circle \
	  --threshold fixed --fixed-thresh 128 --invert --crop \
	  --min-box 2 --max-box 128 --scales 11 --grid-averages 4 --bootstrap 50 \
	  --drop-head 2 --drop-tail 1 --plot --plot-linear --save-grids --grids-max-offsets 1

$(EX_OUTDIR)/sierpinski/sierpinski.png: $(IMAGEDIR)/sierpinski.png | $(EX_OUTDIR) venv
	@mkdir -p $(EX_OUTDIR)/sierpinski
	$(PY) scripts/boxcount.py \
	  --image $< \
	  --out $(EX_OUTDIR)/sierpinski/sierpinski \
	  --threshold fixed --fixed-thresh 128 --invert --crop \
	  --min-box 2 --max-box 128 --scales 11 --grid-averages 4 --bootstrap 50 \
	  --drop-head 2 --drop-tail 1 --plot --plot-linear --save-grids --grids-max-offsets 1

show: examples
	@if command -v xdg-open >/dev/null 2>&1; then \
	  xdg-open $(EX_OUTDIR)/line/line.png; \
	  xdg-open $(EX_OUTDIR)/sine/sine.png; \
	  xdg-open $(EX_OUTDIR)/rectangle/rectangle.png; \
	  xdg-open $(EX_OUTDIR)/circle/circle.png; \
	  xdg-open $(EX_OUTDIR)/sierpinski/sierpinski.png; \
	elif command -v open >/dev/null 2>&1; then \
	  open $(EX_OUTDIR)/line/line.png; \
	  open $(EX_OUTDIR)/sine/sine.png; \
	  open $(EX_OUTDIR)/rectangle/rectangle.png; \
	  open $(EX_OUTDIR)/circle/circle.png; \
	  open $(EX_OUTDIR)/sierpinski/sierpinski.png; \
	else \
	  echo "Open the PNGs in $(EX_OUTDIR)/<name>/<name>.png manually (no opener found)."; \
	fi

line: $(IMAGEDIR)/line.png $(EX_OUTDIR)/line/line.png
	@if command -v xdg-open >/dev/null 2>&1; then xdg-open $(EX_OUTDIR)/line/line.png; elif command -v open >/dev/null 2>&1; then open $(EX_OUTDIR)/line/line.png; fi

sine: $(IMAGEDIR)/sine.png $(EX_OUTDIR)/sine/sine.png
	@if command -v xdg-open >/dev/null 2>&1; then xdg-open $(EX_OUTDIR)/sine/sine.png; elif command -v open >/dev/null 2>&1; then open $(EX_OUTDIR)/sine/sine.png; fi

rectangle: $(IMAGEDIR)/rectangle.png $(EX_OUTDIR)/rectangle/rectangle.png
	@if command -v xdg-open >/dev/null 2>&1; then xdg-open $(EX_OUTDIR)/rectangle/rectangle.png; elif command -v open >/dev/null 2>&1; then open $(EX_OUTDIR)/rectangle/rectangle.png; fi

circle: $(IMAGEDIR)/circle.png $(EX_OUTDIR)/circle/circle.png
	@if command -v xdg-open >/dev/null 2>&1; then xdg-open $(EX_OUTDIR)/circle/circle.png; elif command -v open >/dev/null 2>&1; then open $(EX_OUTDIR)/circle/circle.png; fi

sierpinski: $(IMAGEDIR)/sierpinski.png $(EX_OUTDIR)/sierpinski/sierpinski.png
	@if command -v xdg-open >/dev/null 2>&1; then xdg-open $(EX_OUTDIR)/sierpinski/sierpinski.png; elif command -v open >/dev/null 2>&1; then open $(EX_OUTDIR)/sierpinski/sierpinski.png; fi

# Analyze an arbitrary image: place a copy in out/ and write outputs in out/<base>
.PHONY: analyze_image

# Allow positional form: `make analyze_image path/to/image.png`
ifneq ($(filter analyze_image,$(MAKECMDGOALS)),)
IMAGE := $(if $(IMAGE),$(IMAGE),$(word 2,$(MAKECMDGOALS)))
# Prevent make from treating the second word as a target
$(IMAGE): ; @:
.PHONY: $(IMAGE)
endif

analyze_image: | venv $(OUTDIR) $(RESULTS_OUTDIR)
	@if [ -z "$(IMAGE)" ]; then echo "Usage: make analyze_image IMAGE=/path/to/image or make analyze_image /path/to/image"; exit 2; fi
	@SRC="$(IMAGE)"; \
	if [ ! -f "$$SRC" ] && [ -f "$(INDIR)/$(notdir $(IMAGE))" ]; then \
	  SRC="$(INDIR)/$(notdir $(IMAGE))"; \
	fi; \
	if [ -d "$$SRC" ]; then \
	  echo "[error] provided path is a directory, not a file: $$SRC"; exit 2; \
	fi; \
	if [ ! -f "$$SRC" ]; then \
	  echo "[error] image not found: $(IMAGE) (also tried $(INDIR)/$(notdir $(IMAGE)))"; exit 2; \
	fi; \
	b=$$(basename "$$SRC"); n=$${b%.*}; \
	mkdir -p "$(RESULTS_OUT_PREFIX)/$$n"; \
	# Optional preprocessing
	if [ "$(PREP)" = "1" ]; then \
	  CLEAN_OUT="$(RESULTS_OUT_PREFIX)/$$n/preprocessed.png"; \
	  $(PY) scripts/clean_image.py --in "$$SRC" --out "$$CLEAN_OUT" $(CLEAN_ARGS); \
	  SRC="$$CLEAN_OUT"; \
	fi; \
	$(PY) scripts/boxcount.py \
	  --image "$$SRC" \
	  --out "$(RESULTS_OUT_PREFIX)/$$n/$$n" \
	  --threshold fixed --fixed-thresh 128 $(BOX_INV) --crop $(BOX_PAD) --bootstrap 50 \
	  --min-box 2 --max-box 128 --scales 11 --grid-averages 4 \
	  --drop-head 2 --drop-tail 1 --plot --plot-linear --save-grids --grids-max-offsets 1 || true

.PHONY: analyze_in
analyze_in: | venv $(OUTDIR) $(INDIR) $(RESULTS_OUTDIR)
	@echo "[info] scanning 'in/' for images..."
	@if [ -z "$$(find $(INDIR) -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.tif' -o -iname '*.tiff' \) -print -quit)" ]; then \
	  echo "[warn] no input images found in '$(INDIR)/'. Seeding a sample from assets..."; \
	  if [ ! -f "assets/line.png" ]; then \
	    echo "[info] generating assets via 'make generate'"; \
	    $(MAKE) -s generate; \
	  fi; \
	  cp -f assets/line.png $(INDIR)/ 2>/dev/null || true; \
	  echo "[ok] placed assets/line.png into '$(INDIR)/'. Rerun will analyze it."; \
	fi
	@find $(INDIR) -maxdepth 1 -type f \( \
	  -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.tif' -o -iname '*.tiff' \
	\) -print0 | xargs -0 -I{} sh -c '\
	  f="$${1}"; b="$$(basename "$$f")"; n="$${b%.*}"; \
	  mkdir -p "$(RESULTS_OUT_PREFIX)/$$n"; \
	  echo "[info] analyzing $$f -> $(RESULTS_OUT_PREFIX)/$$n"; \
	  SRC="$$f"; \
	  if [ "$(PREP)" = "1" ]; then \
	    CLEAN_OUT="$(RESULTS_OUT_PREFIX)/$$n/preprocessed.png"; \
	    $(PY) scripts/clean_image.py --in "$$SRC" --out "$$CLEAN_OUT" $(CLEAN_ARGS); \
	    SRC="$$CLEAN_OUT"; \
	  fi; \
	  $(PY) scripts/boxcount.py \
	    --image "$$SRC" \
	    --out "$(RESULTS_OUT_PREFIX)/$$n/$$n" \
	    --threshold fixed --fixed-thresh 128 $(BOX_INV) --crop $(BOX_PAD) --bootstrap 50 \
	    --min-box 2 --max-box 128 --scales 11 --grid-averages 4 \
	    --drop-head 2 --drop-tail 1 --plot --plot-linear --save-grids --grids-max-offsets 1 || true \
	' _ {}

.PHONY: test
test: deps
	@$(PY) -m pytest

.PHONY: clean_image
# Allow positional: `make clean_image path/to/image.png`
ifneq ($(filter clean_image,$(MAKECMDGOALS)),)
IMAGE := $(if $(IMAGE),$(IMAGE),$(word 2,$(MAKECMDGOALS)))
$(IMAGE): ; @:
.PHONY: $(IMAGE)
endif

# Use: make clean_image IMAGE=in/file.png [OUT=in/file_clean.png] [CLEAN_* vars...]
clean_image: | venv $(INDIR)
	@if [ -z "$(IMAGE)" ]; then echo "Usage: make clean_image IMAGE=/path/to/image [OUT=path] or make clean_image /path/to/image"; exit 2; fi
	@SRC="$(IMAGE)"; OUTPATH="$(OUT)"; \
	if [ -z "$$OUTPATH" ]; then \
	  b=$$(basename "$$SRC"); n=$${b%.*}; OUTPATH="$(INDIR)/$${n}_clean.png"; \
	fi; \
	$(PY) scripts/clean_image.py --in "$$SRC" --out "$$OUTPATH" $(CLEAN_ARGS); \
	echo "[ok] cleaned -> $$OUTPATH"

# Manually seed 'in/' with all current assets/* images
seed_in: | $(INDIR)
	@set -e; \
	if [ -z "$$(ls -1 assets/*.png 2>/dev/null | head -n1)" ]; then \
	  echo "[info] no assets found; generating samples"; \
	  $(MAKE) -s generate; \
	fi; \
	cp -f assets/*.png $(INDIR)/ 2>/dev/null || true; \
	echo "[ok] copied assets/*.png into '$(INDIR)/'"

.PHONY: dims_results dims_real
dims_results:
	@echo "[info] scanning $(RESULTS_OUTDIR) for summaries..."
	@find $(RESULTS_OUTDIR) $(if $(RUN),-path "$(RESULTS_OUTDIR)/$(RUN)/*" -o -maxdepth 1 -type f -name "*.skip",) -type f -name "*.txt" 2>/dev/null | while read f; do \
	  echo "==== $$f ===="; \
	  grep -E "Fractal dimension" "$$f" || true; \
	done

# Back-compat alias
dims_real: dims_results

# Geography: analyze a specific country (e.g., make geography-uk)
.PHONY: geography geography-% uk croatia

# Space-separated form: `make geography uk`
ifneq ($(filter geography,$(MAKECMDGOALS)),)
COUNTRY := $(word 2,$(MAKECMDGOALS))
SLUG := $(shell printf '%s' "$(COUNTRY)" | tr '[:upper:]' '[:lower:]' | tr ' ' '-')
# prevent make from trying to build the second goal as a target
$(COUNTRY): ; @:
endif

geography: | venv $(GEODIR) $(RES_GEO_DIR)
	@if [ -z "$(COUNTRY)" ]; then echo "Usage: make geography <country> (e.g., uk, croatia, united-kingdom)"; exit 2; fi
	$(PY) tests/generate_geography.py --country "$(COUNTRY)" --outdir $(GEODIR) --outfile $(SLUG).png --mode outline --line-width 3 --source $(SRC_DATA) --source-url $(SRC_URL) --crs auto
	@mkdir -p $(RES_GEO_DIR)/$(SLUG)
	$(PY) scripts/boxcount.py \
	  --image $(GEODIR)/$(SLUG).png \
	  --out $(RES_GEO_DIR)/$(SLUG)/$(SLUG) \
	  --threshold fixed --fixed-thresh 128 --invert --crop \
	  --min-box 2 --max-box 128 --scales 11 --grid-averages 4 \
	  --drop-head 2 --drop-tail 1 --plot --plot-linear --save-grids --grids-max-offsets 1
	@if command -v xdg-open >/dev/null 2>&1; then xdg-open $(RES_GEO_DIR)/$(SLUG)/$(SLUG).png; elif command -v open >/dev/null 2>&1; then open $(RES_GEO_DIR)/$(SLUG)/$(SLUG).png; fi

# Hyphenated form: `make geography-uk`
geography-%: $(GEODIR)/%.png | $(RES_GEO_DIR)
	@mkdir -p $(RES_GEO_DIR)/$*
	$(PY) scripts/boxcount.py \
	  --image $(GEODIR)/$*.png \
	  --out $(RES_GEO_DIR)/$*/$* \
	  --threshold fixed --fixed-thresh 128 --invert --crop \
	  --min-box 2 --max-box 128 --scales 11 --grid-averages 4 \
	  --drop-head 2 --drop-tail 1 --plot --plot-linear --save-grids --grids-max-offsets 1
	@if command -v xdg-open >/dev/null 2>&1; then xdg-open $(RES_GEO_DIR)/$*/$*.png; elif command -v open >/dev/null 2>&1; then open $(RES_GEO_DIR)/$*/$*.png; fi

uk: geography-uk
croatia: geography-croatia


dims: examples
	@for f in $(EX_OUTDIR)/*/*.txt; do \
	  echo "==== $$f ===="; \
	  grep -E "Fractal dimension" "$$f" || true; \
	done




clean:
	@rm -rf $(EX_OUTDIR) $(RESULTS_OUTDIR)
	@echo "[ok] cleaned $(OUTDIR)/ artifacts"

distclean: clean
	@rm -rf $(IMAGEDIR) tests/__pycache__
	@echo "[ok] removed generated images too"
