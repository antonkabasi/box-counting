#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a black-on-white filled country silhouette PNG using GeoPandas
and the bundled Natural Earth low-res dataset.

Examples:
  python tests/generate_geography.py --country uk --outdir assets/geography
  python tests/generate_geography.py --country croatia --outdir assets/geography

Output is a grayscale image (white background=255, country filled=0),
sized by --width/--height with a margin.
"""
import argparse
import os
from typing import Tuple

import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from pyproj import CRS
from shapely.geometry.base import BaseGeometry
from PIL import Image, ImageDraw


ALIASES = {
    "uk": "United Kingdom",
    "gb": "United Kingdom",
    "great-britain": "United Kingdom",
    "great britain": "United Kingdom",
    "britain": "United Kingdom",
    "united-kingdom": "United Kingdom",
    "united kingdom": "United Kingdom",
    "usa": "United States of America",
    "us": "United States of America",
    "united-states": "United States of America",
    "united states": "United States of America",
    "south-africa": "South Africa",
    "south africa": "South Africa",
    "republic of south africa": "South Africa",
}


def canonicalize(name: str) -> str:
    nm = name.strip().lower().replace("_", " ").replace("-", " ")
    nm = " ".join(nm.split())
    return ALIASES.get(nm, name)


def geom_bounds(geom: BaseGeometry) -> Tuple[float, float, float, float]:
    minx, miny, maxx, maxy = geom.bounds
    return minx, miny, maxx, maxy


def draw_polygon(img: Image.Image, poly: Polygon, scale: float, minx: float, maxy: float, margin: int, mode: str = "outline", line_width: int = 2):
    d = ImageDraw.Draw(img)

    def tx(coords):
        return [
            (
                int(round(margin + (x - minx) * scale)),
                int(round(margin + (maxy - y) * scale)),
            )
            for (x, y) in coords
        ]

    if mode == "fill":
        # exterior filled black
        d.polygon(tx(poly.exterior.coords), fill=0, outline=0)
        # holes (interiors) filled white
        for interior in poly.interiors:
            d.polygon(tx(interior.coords), fill=255, outline=255)
    else:
        # outline only: approximate coastline as polygon boundary
        ext = tx(poly.exterior.coords)
        if len(ext) >= 2:
            d.line(ext, fill=0, width=max(1, int(line_width)))
        for interior in poly.interiors:
            ring = tx(interior.coords)
            if len(ring) >= 2:
                d.line(ring, fill=0, width=max(1, int(line_width)))


def _bundled_geojson_path() -> str | None:
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, os.pardir))
    bundled = os.path.join(repo_root, "assets", "data", "countries_simple.geojson")
    return bundled if os.path.exists(bundled) else None


def _read_world(source_path: str | None, source_url: str | None):
    # 1) explicit path or URL
    if source_path:
        try:
            return gpd.read_file(source_path)
        except Exception:
            pass
    # 2) explicit URL
    if source_url:
        try:
            return gpd.read_file(source_url)
        except Exception:
            pass
    # 3) bundled offline fallback
    bundled = _bundled_geojson_path()
    if bundled:
        try:
            return gpd.read_file(bundled)
        except Exception:
            pass
    # 4) legacy helpers
    try:
        return gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        from geodatasets import get_path as gd_get_path  # type: ignore
        return gpd.read_file(gd_get_path("naturalearth_lowres"))
    except Exception:
        pass
    raise SystemExit("[error] Could not load a countries dataset. Provide --source or --source-url.")


def _name_column(gdf: gpd.GeoDataFrame) -> str:
    for col in ("name", "NAME", "admin", "ADMIN", "NAME_LONG", "name_long"):
        if col in gdf.columns:
            return col
    # default: first non-geometry string column
    for col in gdf.columns:
        if col != gdf.geometry.name and gdf[col].dtype == object:
            return col
    return "name"


def _choose_crs_for_geom(geom_lonlat: BaseGeometry, crs_arg: str | None) -> CRS:
    if crs_arg and crs_arg.lower() != "auto":
        return CRS.from_user_input(crs_arg)
    c = geom_lonlat.centroid
    lon, lat = float(c.x), float(c.y)
    zone = int((lon + 180.0) // 6 + 1)
    epsg_base = 326 if lat >= 0 else 327
    return CRS.from_epsg(int(f"{epsg_base}{zone:02d}"))


def render_country(name: str, width: int, height: int, margin: int, source: str | None = None, mode: str = "outline", line_width: int = 2, source_url: str | None = None, crs: str | None = None) -> Image.Image:
    world = _read_world(source, source_url)
    # match by case-insensitive name
    cname = canonicalize(name)
    name_col = _name_column(world)
    series = world[name_col].astype(str).str.lower()
    canon = cname.strip().lower().replace("-", " ")
    mask = world[series == canon]
    if mask.empty:
        # try contains substring
        mask = world[series.str.contains(canon)]
    if mask.empty and source_url:
        try:
            world2 = gpd.read_file(source_url)
            name_col2 = _name_column(world2)
            series2 = world2[name_col2].astype(str).str.lower()
            mask2 = world2[series2 == canon]
            if mask2.empty:
                mask2 = world2[series2.str.contains(canon)]
            if not mask2.empty:
                world = world2
                mask = mask2
        except Exception:
            pass
    if mask.empty:
        # helpful hint
        sample = ", ".join(sorted(set(world[_name_column(world)].astype(str).head(10))))
        raise SystemExit(f"[error] Country not found: {name}. Try a different name or provide --source/--source-url. Examples: {sample} …")

    # ensure source CRS known; assume WGS84 if missing
    if world.crs is None:
        world = world.set_crs("EPSG:4326")
        mask = mask.set_crs("EPSG:4326")

    geom_lonlat = mask.to_crs("EPSG:4326").unary_union
    target_crs = _choose_crs_for_geom(geom_lonlat, crs)
    geom = mask.to_crs(target_crs).unary_union
    minx, miny, maxx, maxy = geom_bounds(geom)
    if maxx - minx <= 0 or maxy - miny <= 0:
        raise SystemExit("[error] Invalid geometry bounds for selected country")

    # map lon/lat to pixels with uniform scale and margin
    usable_w = width - 2 * margin
    usable_h = height - 2 * margin
    sx = usable_w / float(maxx - minx)
    sy = usable_h / float(maxy - miny)
    scale = min(sx, sy)

    img = Image.new("L", (width, height), 255)
    if isinstance(geom, Polygon):
        draw_polygon(img, geom, scale, minx, maxy, margin, mode=mode, line_width=line_width)
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            draw_polygon(img, poly, scale, minx, maxy, margin, mode=mode, line_width=line_width)
    else:
        # fallback: try to iterate
        try:
            for poly in geom:
                if isinstance(poly, Polygon):
                    draw_polygon(img, poly, scale, minx, maxy, margin)
        except Exception:
            raise SystemExit("[error] Unsupported geometry type for rendering")

    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", required=True, help="Country name or alias (e.g., uk, croatia, united-kingdom)")
    ap.add_argument("--outdir", default="assets/geography")
    ap.add_argument("--outfile", default=None, help="Optional explicit output filename (e.g., uk.png)")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--margin", type=int, default=32)
    ap.add_argument("--source", default=None, help="Optional path/URL to a countries file (GeoJSON/GeoPackage/shapefile)")
    ap.add_argument("--source-url", default="https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson", help="Optional URL to a countries dataset (default: datasets/geo-countries)")
    ap.add_argument("--mode", choices=["outline", "fill"], default="outline", help="Render outline (coastline-like) or filled silhouette")
    ap.add_argument("--line-width", type=int, default=3, help="Line width in pixels for outline mode")
    ap.add_argument("--crs", default="auto", help="Target CRS for rendering (e.g., EPSG:3857 or auto for local UTM)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    img = render_country(
        args.country, args.width, args.height, args.margin,
        source=args.source, mode=args.mode, line_width=args.line_width,
        source_url=args.source_url, crs=args.crs,
    )
    if args.outfile:
        out = os.path.join(args.outdir, args.outfile)
    else:
        out = os.path.join(args.outdir, f"{args.country.lower().replace(' ', '-')}.png")
    img.save(out)
    print("[ok] wrote", out)


if __name__ == "__main__":
    main()
