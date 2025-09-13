import os
import re
import tempfile
import subprocess
import sys
from pathlib import Path

from PIL import Image

import importlib.util


def load_module_by_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


BOXCOUNT_PATH = str(Path(__file__).resolve().parent.parent / "scripts" / "boxcount.py")
GEN_IMAGES_PATH = str(Path(__file__).resolve().parent / "generate_images.py")


def run_boxcount_on(path: str, out_prefix: str):
    cmd = [
        sys.executable,
        BOXCOUNT_PATH,
        "--image",
        path,
        "--out",
        out_prefix,
        "--threshold",
        "fixed",
        "--fixed-thresh",
        "128",
        "--invert",
        "--crop",
        "--min-box",
        "2",
        "--max-box",
        "128",
        "--scales",
        "11",
        "--grid-averages",
        "4",
        "--drop-head",
        "2",
        "--drop-tail",
        "1",
        "--plot",
    ]
    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")
    subprocess.check_call(cmd, env=env)
    txt = Path(out_prefix + ".txt").read_text()
    m = re.search(r"Fractal dimension \(slope.*\): ([0-9.]+)", txt)
    assert m, "Could not parse dimension from summary file"
    return float(m.group(1))


def gen_and_save(img, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def test_primitives_dimensions():
    gen = load_module_by_path("gen_images", GEN_IMAGES_PATH)
    with tempfile.TemporaryDirectory() as td:
        # Generate primitives
        line_path = os.path.join(td, "line.png")
        sine_path = os.path.join(td, "sine.png")
        rect_path = os.path.join(td, "rectangle.png")
        circ_path = os.path.join(td, "circle.png")
        gen_and_save(gen.gen_line(), line_path)
        gen_and_save(gen.gen_sine(), sine_path)
        gen_and_save(gen.gen_rectangle(), rect_path)
        gen_and_save(gen.gen_circle(), circ_path)

        # Analyze
        d_line = run_boxcount_on(line_path, os.path.join(td, "out", "line"))
        d_sine = run_boxcount_on(sine_path, os.path.join(td, "out", "sine"))
        d_rect = run_boxcount_on(rect_path, os.path.join(td, "out", "rectangle"))
        d_circ = run_boxcount_on(circ_path, os.path.join(td, "out", "circle"))

        # Expectations
        assert abs(d_line - 1.0) < 0.1
        assert abs(d_sine - 1.0) < 0.1
        assert 1.8 < d_rect < 2.1
        assert 1.8 < d_circ < 2.1


def test_sierpinski_dimension():
    gen = load_module_by_path("gen_images", GEN_IMAGES_PATH)
    with tempfile.TemporaryDirectory() as td:
        sp_path = os.path.join(td, "sierpinski.png")
        gen_and_save(gen.gen_sierpinski(), sp_path)
        d_sp = run_boxcount_on(sp_path, os.path.join(td, "out", "sierpinski"))
        target = 1.58496  # log(3)/log(2)
        assert abs(d_sp - target) < 0.1
