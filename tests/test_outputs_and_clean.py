import os
import sys
import re
import tempfile
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

BOXCOUNT = str(Path(__file__).resolve().parent.parent / "scripts" / "boxcount.py")
CLEAN = str(Path(__file__).resolve().parent.parent / "scripts" / "clean_image.py")
GEN_GEO = str(Path(__file__).resolve().parent / "generate_geography.py")
GEO_FALLBACK = str(Path(__file__).resolve().parent.parent / "assets" / "data" / "countries_simple.geojson")


def run(cmd):
    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")
    return subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def run_ok(cmd):
    r = run(cmd)
    assert r.returncode == 0, f"cmd failed: {cmd}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    return r


def test_cli_outputs_and_schema(tmp_path):
    # small synthetic line
    img = Image.new("L", (256, 256), 255)
    d = ImageDraw.Draw(img)
    d.line([(10, 246), (246, 10)], fill=0, width=2)
    ip = tmp_path / "line.png"
    img.save(ip)

    outp = tmp_path / "out" / "line"
    outp.parent.mkdir(parents=True, exist_ok=True)
    run_ok([sys.executable, BOXCOUNT, "--image", str(ip), "--out", str(outp),
            "--threshold", "fixed", "--fixed-thresh", "128", "--invert", "--crop",
            "--min-box", "2", "--max-box", "64", "--scales", "8", "--grid-averages", "2",
            "--drop-head", "1", "--drop-tail", "1", "--plot", "--plot-linear"]) 

    # files exist
    assert (outp.with_suffix(".png")).exists()
    linear_path = Path(str(outp) + "_linear.png")
    assert linear_path.exists()
    assert (outp.with_suffix(".csv")).exists()
    txtp = outp.with_suffix(".txt")
    assert txtp.exists()
    txt = txtp.read_text()
    assert "R^2:" in txt
    assert "Fit window:" in txt

    # CSV header has expected columns
    header = (outp.with_suffix(".csv")).read_text().splitlines()[0]
    for col in ["box_size_px", "N_boxes_mean", "inv_epsilon", "log_N"]:
        assert col in header


def test_clean_polarity(tmp_path):
    # black rectangle on white
    base = Image.new("L", (128, 128), 255)
    d = ImageDraw.Draw(base)
    d.rectangle([32, 32, 96, 96], outline=0, width=3)
    src = tmp_path / "rect.png"
    base.save(src)

    wout = tmp_path / "rect_white.png"
    bout = tmp_path / "rect_black.png"

    # out-fg white
    run_ok([sys.executable, CLEAN, "--in", str(src), "--out", str(wout),
            "--alpha-bg", "white", "--threshold", "otsu", "--invert", "--borderize", "1", "--out-fg", "white"]) 
    # out-fg black
    run_ok([sys.executable, CLEAN, "--in", str(src), "--out", str(bout),
            "--alpha-bg", "white", "--threshold", "otsu", "--invert", "--borderize", "1", "--out-fg", "black"]) 

    wa = Image.open(wout).convert("L"); wb = list(wa.getdata())
    ba = Image.open(bout).convert("L"); bb = list(ba.getdata())
    # white-foreground image should contain some 255s and mostly 0s (background)
    assert max(wb) == 255 and min(wb) == 0
    # black-foreground image should contain some 0s and mostly 255s (background)
    assert max(bb) == 255 and min(bb) == 0
    # polarity differs at many pixels
    diffs = sum(1 for x, y in zip(wb, bb) if (x == 255) != (y == 255))
    assert diffs > 50


def test_geography_outputs_and_grids(tmp_path):
    # Use offline placeholder and outline mode
    geo_img = tmp_path / "uk.png"
    run_ok([sys.executable, GEN_GEO, "--country", "uk", "--outdir", str(tmp_path),
            "--outfile", geo_img.name, "--mode", "outline", "--source", GEO_FALLBACK])

    outp = tmp_path / "out" / "uk"
    run_ok([sys.executable, BOXCOUNT, "--image", str(geo_img), "--out", str(outp),
            "--threshold", "fixed", "--fixed-thresh", "128", "--invert", "--crop",
            "--min-box", "2", "--max-box", "64", "--scales", "8", "--grid-averages", "2",
            "--drop-head", "1", "--drop-tail", "1", "--plot"]) 

    # existence checks
    assert (outp.with_suffix(".png")).exists()
    # grids are produced by Make recipes; here we exercise the CLI only, so skip grids check


def test_error_on_blank_image(tmp_path):
    blank = tmp_path / "blank.png"
    Image.new("L", (128, 128), 255).save(blank)
    outp = tmp_path / "out" / "blank"
    r = run([sys.executable, BOXCOUNT, "--image", str(blank), "--out", str(outp)])
    assert r.returncode != 0
    assert "foreground fraction" in (r.stderr + r.stdout)
