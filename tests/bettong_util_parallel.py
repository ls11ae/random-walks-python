#!/usr/bin/env python3
"""
Parallel per-segment habitat-use (WorldCover) from segment utilization distributions.

What it does
------------
For each consecutive pair of fixes bettong[i] -> bettong[i+1] (15-min segments):
  1) Build a local segment grid around the two points (+/- padding)
  2) Compute segment utilization distribution (UD) using CorrelatedWalker
  3) Attribute each UD cell to an ESA WorldCover class
  4) Aggregate UD mass by class for that segment
  5) Write one big CSV with one row per (segment, wc_code)

Key design
----------
- Uses multiprocessing with "spawn" (more stable for native extensions than fork)
- Avoids pickling non-pickleable kernel objects:
    -> main computes m_1/m_2/m_3 matrices (numpy arrays)
    -> each worker builds kernels locally from matrices in the initializer
- Each worker loads WorldCover band into a numpy array once in initializer (fast lookup)

You may need to set flip_y=True if your UD grid y-axis is inverted relative to map northing.
"""

import os
import csv
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import rowcol

from pystac_client import Client
import planetary_computer
import rioxarray
from pyproj import Transformer

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# -------------------------
# random_walk_package imports
# -------------------------
from random_walk_package.bindings.data_structures import kernels
from random_walk_package.core.BiasedWalker import BiasedWalker
from random_walk_package.core.CorrelatedWalker import *
from random_walk_package.bindings.mixed_walk import *
from random_walk_package.bindings.plotter import *
from random_walk_package import matrix_generator_gaussian_pdf
from random_walk_package.bindings.brownian_walk import *

# -------------------------
# WorldCover classes + CRS
# -------------------------
WC_CLASSES = {
    0: ("No Data", "#000000"),
    10: ("Tree cover", "#006400"),
    20: ("Shrubland", "#ffbb22"),
    30: ("Grassland", "#ffff4c"),
    40: ("Cropland", "#f096ff"),
    50: ("Built-up", "#fa0000"),
    60: ("Bare / sparse vegetation", "#b4b4b4"),
    70: ("Snow and ice", "#f0f0f0"),
    80: ("Permanent water bodies", "#0064c8"),
    90: ("Herbaceous wetland", "#0096a0"),
    95: ("Mangroves", "#00cf75"),
    100: ("Moss and lichen", "#fae6a0"),
}

CRS_MGA55 = "EPSG:28355"
CRS_WGS84 = "EPSG:4326"

to_wgs84 = Transformer.from_crs(CRS_MGA55, CRS_WGS84, always_xy=True)
to_mga55 = Transformer.from_crs(CRS_WGS84, CRS_MGA55, always_xy=True)

# -------------------------
# CSV processing (your ID date rules)
# -------------------------
MD_IDS = {
    "lagartha", "bjorn", "baldur", "sifa", "floki", "freya", "andive",
    "beetroot", "durian", "parsnip", "potato", "pumpkin", "raddish", "sprout",
    "swede", "turnip", "tomato"
}
DM_IDS = {"dot", "edwina", "egbert", "maud", "olga", "othello", "percy", "renet"}

def process_bettongs(file_path: str):
    data = pd.read_csv(file_path)
    bettongs = {}
    for _, row in data.iterrows():
        bid = row["ID"]
        if bid in MD_IDS:
            dt = datetime.strptime(row["datetime"], "%m/%d/%y %H:%M")
        elif bid in DM_IDS:
            dt = datetime.strptime(row["datetime"], "%d/%m/%y %H:%M")
        else:
            raise ValueError(f"UNKNOWN ID: {bid}")

        state = int(row["states"])
        bettongs.setdefault(bid, []).append((int(row["x"]), int(row["y"]), dt, state))

    for bid in bettongs:
        bettongs[bid].sort(key=lambda t: t[2])
    return bettongs

# -------------------------
# Step model fitting (GMM -> matrices)
# -------------------------
rnge = 40
reso = rnge * 2 + 1

def delta_vector(a, b, step_size=1):
    d = np.array(a) - np.array(b)
    return d / step_size

def calculate_steps_grouped(bettongs, step_size=1):
    steps = [[], [], []]
    count_total = 0
    count_discarded = 0

    for _, entries in bettongs.items():
        for i in range(1, len(entries)):
            count_total += 1
            time_diff_0 = (entries[i][2] - entries[i - 1][2]).total_seconds() / 60
            if time_diff_0 == 15:
                # Your original used i-2; guard i>=2
                if i < 2:
                    count_discarded += 1
                    continue
                a = (entries[i - 2][0], entries[i - 2][1])
                b = (entries[i - 1][0], entries[i - 1][1])
                steps[entries[i - 1][3] - 1].append(delta_vector(b, a, step_size))
            else:
                count_discarded += 1

    if count_total > 0:
        print(f"  total of {count_total} steps, {count_discarded / count_total}% discarded")
    return steps

def fit_matrix_from_steps(steps):
    data = np.array(steps, dtype=float)
    if data.size == 0:
        # Fallback: tiny Gaussian-ish blob to avoid crashes
        Z = np.zeros((reso, reso), dtype=float)
        Z[reso // 2, reso // 2] = 1.0
        return Z

    gmm = GaussianMixture(n_components=3, covariance_type="full")
    gmm.fit(data)

    x = np.linspace(-rnge, rnge, reso)
    y = np.linspace(-rnge, rnge, reso)
    X, Y = np.meshgrid(x, y)
    grid = np.column_stack([X.ravel(), Y.ravel()])

    log_density = gmm.score_samples(grid)
    density = np.exp(log_density)
    Z = density.reshape(X.shape)
    return Z

def pure_grouped_matrices(bettongs, step_size=15):
    a, b, c = calculate_steps_grouped(bettongs, step_size)
    m_a = fit_matrix_from_steps(a)
    m_b = fit_matrix_from_steps(b)
    m_c = fit_matrix_from_steps(c)
    return m_a, m_b, m_c

# -------------------------
# WorldCover fetch / clip / reproject
# -------------------------
def bbox_mga55_to_wgs84(minx, miny, maxx, maxy):
    corners = [(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]
    lons_lats = [to_wgs84.transform(x, y) for x, y in corners]
    lons, lats = zip(*lons_lats)
    return (min(lons), min(lats), max(lons), max(lats))

def compute_bbox_from_points_mga55(points, pad_m=100):
    xs = [x for (x, y, *_rest) in points]
    ys = [y for (x, y, *_rest) in points]
    return (min(xs) - pad_m, min(ys) - pad_m, max(xs) + pad_m, max(ys) + pad_m)

def reproject_raster(in_tif, out_tif, dst_crs=CRS_MGA55):
    with rasterio.open(in_tif) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height
        })

        with rasterio.open(out_tif, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest
                )
    return out_tif

def fetch_landcover_data_worldcover(bbox_wgs84, year=2021, output_filename="worldcover_clip_4326.tif"):
    print(f"Fetching ESA WorldCover for bbox (WGS84): {bbox_wgs84} and year {year}...")

    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    dt = f"{year}-01-01/{year+1}-01-01"
    search = catalog.search(
        collections=["esa-worldcover"],
        bbox=list(bbox_wgs84),
        datetime=dt,
    )

    items = search.item_collection()
    if not items:
        raise RuntimeError("No ESA WorldCover items found for the given AOI/year.")

    item = items[0]
    print(f"Found {len(items)} item(s). Using: {item.id}")

    asset_href = item.assets["map"].href
    xds = rioxarray.open_rasterio(asset_href).rio.write_crs(CRS_WGS84)

    clipped = xds.rio.clip_box(
        minx=bbox_wgs84[0],
        miny=bbox_wgs84[1],
        maxx=bbox_wgs84[2],
        maxy=bbox_wgs84[3],
        crs=CRS_WGS84
    )

    clipped.rio.to_raster(output_filename, compress="LZW", dtype="uint8")
    print(f"Saved clipped WorldCover to: {output_filename}")
    return output_filename

def worldcover_trajectory_pipeline(points, pad_m=100, year=2021,
                                  clip_4326="worldcover_clip_4326.tif",
                                  clip_28355="worldcover_clip_28355.tif"):
    bbox_mga = compute_bbox_from_points_mga55(points, pad_m=pad_m)
    bbox_wgs = bbox_mga55_to_wgs84(*bbox_mga)
    tif_4326 = fetch_landcover_data_worldcover(bbox_wgs, year=year, output_filename=clip_4326)
    tif_28355 = reproject_raster(tif_4326, clip_28355, dst_crs=CRS_MGA55)
    return tif_28355

# -------------------------
# Worker globals (set in initializer)
# -------------------------
G_BETTONG = None
G_PADDING = None
G_T = None
G_FLIP_Y = None
G_IGNORE_CODES = None

G_KERNELS = None  # built in worker from matrices
G_WC_BAND = None
G_WC_TRANSFORM = None
G_WC_HEIGHT = None
G_WC_WIDTH = None
G_WC_BOUNDS = None
G_WC_NODATA = None

def _points_to_np_float64(points, n: int) -> np.ndarray:
    # Robust conversion from bindings to numpy without buffer protocol assumptions
    try:
        arr = np.fromiter(points, dtype=np.float64, count=n)
        if arr.size == n:
            return arr
    except TypeError:
        pass
    return np.fromiter((float(points[i]) for i in range(n)), dtype=np.float64, count=n)

def _init_worker(bettong, padding, T, flip_y, ignore_codes, m1, m2, m3, wc_path):
    """
    Runs inside each worker process (spawn). Safe place to:
      - set globals
      - build kernels (non-pickleable) from matrices
      - load WorldCover raster band into numpy
    """
    global G_BETTONG, G_PADDING, G_T, G_FLIP_Y, G_IGNORE_CODES
    global G_KERNELS, G_WC_BAND, G_WC_TRANSFORM, G_WC_HEIGHT, G_WC_WIDTH, G_WC_BOUNDS, G_WC_NODATA

    G_BETTONG = bettong
    G_PADDING = int(padding)
    G_T = int(T)
    G_FLIP_Y = bool(flip_y)
    G_IGNORE_CODES = set(ignore_codes) if ignore_codes is not None else set()

    # Build kernels locally in worker (avoids pickling kernel objects)
    k1 = correlated_kernels_from_matrix(m1, reso, reso, 1)
    k2 = correlated_kernels_from_matrix(m2, reso, reso, 1)
    k3 = correlated_kernels_from_matrix(m3, reso, reso, 1)
    G_KERNELS = [k1, k2, k3]

    # Load WorldCover band (fast repeated lookup)
    with rasterio.open(wc_path) as src:
        G_WC_BAND = src.read(1)
        G_WC_TRANSFORM = src.transform
        G_WC_HEIGHT = src.height
        G_WC_WIDTH = src.width
        G_WC_BOUNDS = src.bounds
        G_WC_NODATA = src.nodata

def _compute_segment_rows(i: int):
    """
    Worker task:
      - compute segment UD on local grid
      - aggregate UD mass by WorldCover class
      - return list[dict] rows for CSV
    """
    # Pull segment endpoints
    x0, y0, t0, s0 = G_BETTONG[i]
    x1, y1, t1, s1 = G_BETTONG[i + 1]

    # Only accept 15-min steps (same as your earlier logic)
    if (t1 - t0).total_seconds() / 60 != 15:
        return []

    # Define local segment grid (explicit padding)
    p = G_PADDING
    seg_min_x = min(x0, x1) - p
    seg_min_y = min(y0, y1) - p
    seg_max_x = max(x0, x1) + p
    seg_max_y = max(y0, y1) + p
    W = int(seg_max_x - seg_min_x)
    H = int(seg_max_y - seg_min_y)

    # Map coords -> local pixel coords
    sx, sy = x0 - seg_min_x, y0 - seg_min_y
    tx, ty = x1 - seg_min_x, y1 - seg_min_y

    if not (0 <= sx < W and 0 <= sy < H and 0 <= tx < W and 0 <= ty < H):
        return []

    # Compute UD for this segment
    walker = CorrelatedWalker(S=25, kernel=G_KERNELS[s0 - 1], D=1, W=W, H=H, T=G_T)
    walker.generate(start_x=sx, start_y=sy, use_serialization=False)
    utilization_distribution = walker.utilize(end_x=tx, end_y=ty)

    n = W * H
    acc = np.zeros(n, dtype=np.float64)
    for t in range(G_T):
        pts = utilization_distribution[t][0].data[0][0].data.points
        acc += _points_to_np_float64(pts, n)
    acc /= G_T
    seg = acc.reshape((H, W))

    # Aggregate by WorldCover code (vectorized)
    rr, cc = np.nonzero(seg > 0)
    if rr.size == 0:
        return []

    vals = seg[rr, cc].astype(np.float64, copy=False)

    xs = seg_min_x + cc.astype(np.int64)
    if G_FLIP_Y:
        ys = seg_min_y + (H - 1 - rr.astype(np.int64))
    else:
        ys = seg_min_y + rr.astype(np.int64)

    # Convert x,y to WorldCover row/col
    wc_r, wc_c = rowcol(G_WC_TRANSFORM, xs, ys)

    wc_r = np.asarray(wc_r, dtype=np.int64)
    wc_c = np.asarray(wc_c, dtype=np.int64)

    inb = (wc_r >= 0) & (wc_r < G_WC_HEIGHT) & (wc_c >= 0) & (wc_c < G_WC_WIDTH)
    if not np.any(inb):
        return []

    wc_r = wc_r[inb]
    wc_c = wc_c[inb]
    vals = vals[inb]

    codes = G_WC_BAND[wc_r, wc_c].astype(np.int64, copy=False)

    # nodata / ignore masks
    mask = np.ones_like(codes, dtype=bool)
    if G_WC_NODATA is not None:
        mask &= (codes != int(G_WC_NODATA))
    if G_IGNORE_CODES:
        # vectorized "not in" via isin
        mask &= ~np.isin(codes, list(G_IGNORE_CODES))

    if not np.any(mask):
        return []

    codes = codes[mask]
    vals = vals[mask]

    total_mass = float(vals.sum())
    if total_mass <= 0:
        return []

    # Use bincount (codes are small: max 100)
    max_code = int(codes.max())
    bc = np.bincount(codes, weights=vals, minlength=max_code + 1)

    rows = []
    for code in np.nonzero(bc)[0]:
        mass = float(bc[code])
        if mass <= 0:
            continue
        prop = mass / total_mass
        name = WC_CLASSES.get(int(code), ("Unknown", None))[0]

        rows.append({
            "segment_index": i,
            "t0": t0.isoformat(sep=" "),
            "t1": t1.isoformat(sep=" "),
            "x0": x0, "y0": y0,
            "x1": x1, "y1": y1,
            "state0": s0, "state1": s1,
            "segment_total_mass": total_mass,
            "wc_code": int(code),
            "wc_name": name,
            "mass": mass,
            "proportion": prop,
        })

    # Sort rows by mass descending (optional)
    rows.sort(key=lambda d: d["mass"], reverse=True)

    print(f"Segment {i} finished")

    return rows

# -------------------------
# Main entry
# -------------------------
def main():
    # --- user parameters ---
    file_path = "/home/mart/Code/eco/Gardiner- Habitat.Perception.Bettongs.csv"
    bettong_id = "tomato"

    padding = 100
    worldcover_year = 2020

    T = 15  # walker timesteps
    flip_y = False           # set True if class attribution looks flipped
    ignore_codes = set()     # e.g. {0} to drop "No Data"
    max_workers = None       # None => let executor choose

    out_csv = "segment_worldcover_mass_parallel.csv"

    # --- load tracks ---
    bettongs = process_bettongs(file_path)
    bettong = bettongs[bettong_id]
    if len(bettong) < 2:
        raise ValueError("Need at least 2 fixes for segments.")

    # --- fetch/reproject WorldCover covering this track ---
    wc_tif = worldcover_trajectory_pipeline(
        bettong,
        pad_m=padding,
        year=worldcover_year,
        clip_4326="worldcover_clip_4326.tif",
        clip_28355="worldcover_clip_28355.tif",
    )

    # --- fit step matrices once (main process) ---
    m1, m2, m3 = pure_grouped_matrices(bettongs, step_size=15)

    # --- parallel segment processing ---
    total_segments = len(bettong) - 1
    print(f"Processing {total_segments} segments in parallel...")
    indices = list(range(total_segments))

    ctx = mp.get_context("spawn")
    rows_all = []

    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(bettong, padding, T, flip_y, ignore_codes, m1, m2, m3, wc_tif),
    ) as ex:
        futures = [ex.submit(_compute_segment_rows, i) for i in indices]
        for fut in as_completed(futures):
            rows = fut.result()
            if rows:
                rows_all.extend(rows)

    # --- write CSV ---
    if not rows_all:
        print("No rows produced (no valid segments or empty UD). Nothing to write.")
        return

    df = pd.DataFrame(rows_all)
    df.sort_values(["segment_index", "mass"], ascending=[True, False], inplace=True)
    df.to_csv(out_csv, index=False)

    print(f"Wrote {len(df)} rows to {out_csv}")
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
