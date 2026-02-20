#!/usr/bin/env python3
"""
bettong_util_canopy_cont.py  (spawn-safe, 31 workers)

Compute continuous canopy summaries for step-wise UDs and home-range masks.

Outputs:
- csv_out/proper/<id>_segment_pgreen_continuous.csv
- csv_out/proper/<id>_homerange_pgreen_continuous.csv
- (optional) csv_out/proper/<id>_homerange_pgreen_continuous_plot.png

Notes:
- Uses EPSG:28355 (MGA55) for animal coords and UD grids.
- Seasonal canopy rasters are EPSG:3577; we transform MGA55 -> raster CRS for lookup.
- Reads only the AOI window of each seasonal raster.
- Uses "spawn" multiprocessing; kernels are built INSIDE each worker process.
"""

# ---- make fork/spawn safer with BLAS/OpenMP ----
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import glob
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from rasterio.windows import transform as window_transform
from pyproj import Transformer

# -------------------------
# random_walk_package imports
# -------------------------
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

from random_walk_package.bindings.data_structures import kernels
from random_walk_package.core.BiasedWalker import BiasedWalker
from random_walk_package.core.CorrelatedWalker import *
from random_walk_package.bindings.mixed_walk import *
from random_walk_package.bindings.plotter import *
from random_walk_package import matrix_generator_gaussian_pdf
from random_walk_package.bindings.brownian_walk import *


# -------------------------
# Constants / CRS
# -------------------------
CRS_MGA55 = "EPSG:28355"

# Your tracking CSV has inconsistent datetime formats by animal group
MD_IDS = {
    "lagartha", "bjorn", "baldur", "sifa", "floki", "freya", "andive",
    "beetroot", "durian", "parsnip", "potato", "pumpkin", "raddish", "sprout",
    "swede", "turnip", "tomato"
}
DM_IDS = {"dot", "edwina", "egbert", "maud", "olga", "othello", "percy", "renet"}

# -------------------------
# GMM / step-fitting parameters (same as your workflow)
# -------------------------
rnge = 40
reso = rnge * 2 + 1


# ============================================================
# I/O: track parsing
# ============================================================
def process_bettongs(file_path: str) -> Dict[str, List[Tuple[int, int, datetime, int]]]:
    data = pd.read_csv(file_path)
    bettongs: Dict[str, List[Tuple[int, int, datetime, int]]] = {}

    for _, row in data.iterrows():
        bettong_id = str(row["ID"]).lower()

        if bettong_id in MD_IDS:
            dt = datetime.strptime(row["datetime"], "%m/%d/%y %H:%M")
        elif bettong_id in DM_IDS:
            dt = datetime.strptime(row["datetime"], "%d/%m/%y %H:%M")
        else:
            raise ValueError(f"UNKNOWN ID: {bettong_id}")

        state = int(row["states"])
        bettongs.setdefault(bettong_id, []).append((int(row["x"]), int(row["y"]), dt, state))

    for bid in bettongs:
        bettongs[bid].sort(key=lambda e: e[2])

    return bettongs


# ============================================================
# Step model fitting (GMM) — returns matrices (numpy), not kernels
# ============================================================
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
                if i < 2:
                    count_discarded += 1
                    continue
                a = (entries[i - 2][0], entries[i - 2][1])
                b = (entries[i - 1][0], entries[i - 1][1])
                steps[entries[i - 1][3] - 1].append(delta_vector(b, a, step_size))
            else:
                count_discarded += 1

    if count_total > 0:
        print(f"    total of {count_total} steps, {count_discarded / count_total:.2%} discarded")

    return steps


def fit_gmm_matrix(coords, n_components=3):
    coords = np.asarray(coords, dtype=float)
    gmm = GaussianMixture(n_components=n_components, covariance_type="full")
    gmm.fit(coords)

    x = np.linspace(-rnge, rnge, reso)
    y = np.linspace(-rnge, rnge, reso)
    X, Y = np.meshgrid(x, y)
    grid = np.column_stack([X.ravel(), Y.ravel()])

    log_density = gmm.score_samples(grid)
    density = np.exp(log_density)
    Z = density.reshape(X.shape)
    return Z


def build_step_matrices(bettongs, step_size=15):
    a, b, c = calculate_steps_grouped(bettongs, step_size)
    m_a = fit_gmm_matrix(a)
    m_b = fit_gmm_matrix(b)
    m_c = fit_gmm_matrix(c)
    return m_a, m_b, m_c


# ============================================================
# Season logic matching your filenames
# ============================================================
def season_tag_for_date(dt: datetime) -> str:
    y, m = dt.year, dt.month
    if m in (12, 1, 2):  # DJF
        start_y = y if m == 12 else y - 1
        start = f"{start_y}12"
        end = f"{start_y + 1}02"
    elif m in (3, 4, 5):  # MAM
        start = f"{y}03"
        end = f"{y}05"
    elif m in (6, 7, 8):  # JJA
        start = f"{y}06"
        end = f"{y}08"
    else:  # SON
        start = f"{y}09"
        end = f"{y}11"
    return f"{start}{end}"


def build_filename_for_date(dt: datetime) -> str:
    return f"lztmre_tas_m{season_tag_for_date(dt)}_dp7a2.tif"


def season_tif_path_for_date(dt: datetime, tif_dir: str) -> str:
    expected = os.path.join(tif_dir, build_filename_for_date(dt))
    if os.path.exists(expected):
        return expected

    tag = season_tag_for_date(dt)
    candidates = glob.glob(os.path.join(tif_dir, f"*m{tag}*.tif"))
    if candidates:
        candidates.sort()
        return candidates[0]

    raise FileNotFoundError(
        f"No tif found for date={dt.isoformat()} season_tag={tag}. "
        f"Expected: {expected} or pattern *m{tag}*.tif in {tif_dir}"
    )


# ============================================================
# Raster window context (continuous raw values)
# ============================================================
def _bbox_mga55_to_raster_bounds(
    bbox_mga55: Tuple[float, float, float, float],
    to_raster: Optional[Transformer],
) -> Tuple[float, float, float, float]:
    minx, miny, maxx, maxy = bbox_mga55
    if to_raster is None:
        return (minx, miny, maxx, maxy)

    xs = np.array([minx, minx, maxx, maxx], dtype=np.float64)
    ys = np.array([miny, maxy, miny, maxy], dtype=np.float64)
    xr, yr = to_raster.transform(xs, ys)
    left, right = float(np.min(xr)), float(np.max(xr))
    bottom, top = float(np.min(yr)), float(np.max(yr))
    return (left, bottom, right, top)


def build_ctx_for_season_window_continuous(
    season_tif: str,
    bbox_mga55: Tuple[float, float, float, float],
    nodata_val_fallback: float = 255.0,
) -> Dict[str, Any]:
    """
    Read ONLY the raster window overlapping bbox_mga55 and return:
      {
        "raw": float32 array of canopy values,
        "valid": boolean mask,
        "nodata": nodata value,
        "inv_transform": inverse window transform (raster CRS),
        "height","width",
        "to_raster": MGA55->raster transformer (or None),
        "extent": extent for plotting
      }
    """
    with rasterio.open(season_tif) as src:
        if src.crs is None:
            raise RuntimeError(f"Raster has no CRS: {season_tif}")

        to_raster = None
        if str(src.crs).upper() != CRS_MGA55:
            to_raster = Transformer.from_crs(CRS_MGA55, src.crs, always_xy=True)

        left, bottom, right, top = _bbox_mga55_to_raster_bounds(bbox_mga55, to_raster)

        win = from_bounds(left, bottom, right, top, transform=src.transform)
        win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))

        raw = src.read(1, window=win).astype(np.float32)
        win_tr = window_transform(win, src.transform)

        # plotting extent (raster CRS)
        h, w = raw.shape
        x0, y0 = win_tr * (0, 0)
        x1, y1 = win_tr * (w, h)
        extent = (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))

        nd = src.nodata
        if nd is None:
            nd = nodata_val_fallback

    valid = (raw != float(nd))
    raw = np.clip(raw, 0.0, 100.0)

    return {
        "raw": raw,
        "valid": valid,
        "nodata": float(nd),
        "inv_transform": ~win_tr,
        "height": raw.shape[0],
        "width": raw.shape[1],
        "to_raster": to_raster,
        "extent": extent,
    }


# ============================================================
# UD helpers
# ============================================================
def ud_isopleth_mask(ud, p=0.9):
    ud = np.asarray(ud, dtype=float)
    ud = np.clip(ud, 0, None)
    s = ud.sum()
    if s <= 0:
        return np.zeros_like(ud, dtype=bool), np.nan

    udn = ud / s
    flat = udn.ravel()
    idx = np.argsort(flat)[::-1]
    csum = np.cumsum(flat[idx])
    k = np.searchsorted(csum, p, side="left")
    level = flat[idx[k]]
    mask = (udn >= level)
    return mask, float(level)


def points_to_np_float64(points, n: int) -> np.ndarray:
    # robust extraction from bound vector-like structure
    try:
        arr = np.fromiter(points, dtype=np.float64, count=n)
        if arr.size == n:
            return arr
    except TypeError:
        pass
    return np.fromiter((float(points[i]) for i in range(n)), dtype=np.float64, count=n)


def compute_segment_util(
    bettong, kernels, i: int,
    min_x: int, min_y: int, *,
    W: int, H: int, T: int
):
    x0, y0, t0, s0 = bettong[i]
    x1, y1, t1, s1 = bettong[i + 1]

    if (t1 - t0).total_seconds() / 60 != 15:
        return None

    sx, sy = x0 - min_x, y0 - min_y
    tx, ty = x1 - min_x, y1 - min_y

    if not (0 <= sx < W and 0 <= sy < H and 0 <= tx < W and 0 <= ty < H):
        return None

    walker = CorrelatedWalker(S=25, kernel=kernels[s0 - 1], D=1, W=W, H=H, T=T)
    walker.generate(start_x=sx, start_y=sy, use_serialization=False)
    utilization_distribution = walker.utilize(end_x=tx, end_y=ty)

    n = W * H
    acc = np.zeros(n, dtype=np.float64)
    for t in range(T):
        pts = utilization_distribution[t][0].data[0][0].data.points
        acc += points_to_np_float64(pts, n)

    acc /= T
    return acc.reshape((H, W))


def add_with_offset(total_arr, seg_arr, dx, dy):
    Ht, Wt = total_arr.shape
    Hs, Ws = seg_arr.shape
    x0 = max(dx, 0); y0 = max(dy, 0)
    x1 = min(dx + Ws, Wt); y1 = min(dy + Hs, Ht)
    if x0 >= x1 or y0 >= y1:
        return
    sx0 = x0 - dx; sy0 = y0 - dy
    sx1 = sx0 + (x1 - x0); sy1 = sy0 + (y1 - y0)
    total_arr[y0:y1, x0:x1] += seg_arr[sy0:sy1, sx0:sx1]


def or_mask_with_offset(global_mask, local_mask, dx, dy):
    Ht, Wt = global_mask.shape
    Hs, Ws = local_mask.shape
    x0 = max(dx, 0); y0 = max(dy, 0)
    x1 = min(dx + Ws, Wt); y1 = min(dy + Hs, Ht)
    if x0 >= x1 or y0 >= y1:
        return
    sx0 = x0 - dx; sy0 = y0 - dy
    sx1 = sx0 + (x1 - x0); sy1 = sy0 + (y1 - y0)
    global_mask[y0:y1, x0:x1] |= local_mask[sy0:sy1, sx0:sx1]


# ============================================================
# Continuous canopy summaries (UD-weighted mean/sd)
# ============================================================
def _weighted_mean_sd(values: np.ndarray, weights: np.ndarray):
    wsum = float(np.sum(weights))
    if wsum <= 0:
        return np.nan, np.nan, 0.0
    mu = float(np.sum(weights * values) / wsum)
    var = float(np.sum(weights * (values - mu) ** 2) / wsum)
    sd = float(np.sqrt(max(var, 0.0)))
    return mu, sd, wsum


def segment_canopy_continuous_fast_ctx(seg_ud, seg_min_x, seg_min_y, ctx):
    """
    For all UD cells with seg_ud > 0:
      - sample canopy value
      - compute UD-weighted mean/sd
    Returns: (ud_mass, mean, sd, n_valid)
      - ud_mass is sum of UD weights over valid sampled cells
      - n_valid is number of valid sampled cells
    """
    raw = ctx["raw"]
    valid = ctx["valid"]
    inv = ctx["inv_transform"]
    Hc, Wc = ctx["height"], ctx["width"]
    to_raster = ctx.get("to_raster", None)

    seg_ud = np.asarray(seg_ud, dtype=np.float64)
    rr, cc = np.nonzero(seg_ud > 0)
    if rr.size == 0:
        return 0.0, np.nan, np.nan, 0

    weights = seg_ud[rr, cc].astype(np.float64)
    xs = seg_min_x + cc.astype(np.float64)
    ys = seg_min_y + rr.astype(np.float64)

    if to_raster is not None:
        xs, ys = to_raster.transform(xs, ys)

    cc_f, rr_f = inv * (xs, ys)
    cci = np.floor(cc_f).astype(np.int64)
    rri = np.floor(rr_f).astype(np.int64)

    ok = (rri >= 0) & (rri < Hc) & (cci >= 0) & (cci < Wc)
    if not np.any(ok):
        return 0.0, np.nan, np.nan, 0

    rri = rri[ok]; cci = cci[ok]; weights = weights[ok]

    ok2 = valid[rri, cci]
    if not np.any(ok2):
        return 0.0, np.nan, np.nan, 0

    rri = rri[ok2]; cci = cci[ok2]; weights = weights[ok2]
    vals = raw[rri, cci].astype(np.float64)

    mu, sd, wsum = _weighted_mean_sd(vals, weights)
    return float(wsum), mu, sd, int(vals.size)


def core_canopy_continuous_fast_ctx(seg_ud, core_mask, seg_min_x, seg_min_y, ctx):
    """
    Same as above but restricted to core_mask == True.
    Returns: (core_area, core_mean, core_sd, core_n_valid)
      - core_area: number of core cells (mask true) in UD grid (regardless of raster validity)
      - core_n_valid: number of those core cells that could be sampled with valid raster
      - mean/sd are UD-weighted within the valid sampled subset
    """
    seg_ud = np.asarray(seg_ud, dtype=np.float64)
    core_mask = np.asarray(core_mask, dtype=bool)

    core_rr, core_cc = np.nonzero(core_mask)
    core_area = int(core_rr.size)
    if core_area == 0:
        return 0, np.nan, np.nan, 0

    # restrict to UD>0 as well (should be true usually, but be safe)
    keep = seg_ud[core_rr, core_cc] > 0
    core_rr = core_rr[keep]
    core_cc = core_cc[keep]
    if core_rr.size == 0:
        return core_area, np.nan, np.nan, 0

    # weights/values sampled like segment_canopy...
    raw = ctx["raw"]
    valid = ctx["valid"]
    inv = ctx["inv_transform"]
    Hc, Wc = ctx["height"], ctx["width"]
    to_raster = ctx.get("to_raster", None)

    weights = seg_ud[core_rr, core_cc].astype(np.float64)
    xs = seg_min_x + core_cc.astype(np.float64)
    ys = seg_min_y + core_rr.astype(np.float64)

    if to_raster is not None:
        xs, ys = to_raster.transform(xs, ys)

    cc_f, rr_f = inv * (xs, ys)
    cci = np.floor(cc_f).astype(np.int64)
    rri = np.floor(rr_f).astype(np.int64)

    ok = (rri >= 0) & (rri < Hc) & (cci >= 0) & (cci < Wc)
    if not np.any(ok):
        return core_area, np.nan, np.nan, 0

    rri = rri[ok]; cci = cci[ok]; weights = weights[ok]

    ok2 = valid[rri, cci]
    if not np.any(ok2):
        return core_area, np.nan, np.nan, 0

    rri = rri[ok2]; cci = cci[ok2]; weights = weights[ok2]
    vals = raw[rri, cci].astype(np.float64)

    mu, sd, wsum = _weighted_mean_sd(vals, weights)
    return core_area, mu, sd, int(vals.size)


def home_range_canopy_continuous_ctx(home_mask, global_min_x, global_min_y, ctx):
    """
    Unweighted mean/sd of canopy within home_mask cells.
    Returns: (mask_area, mean, sd, n_valid)
    """
    raw = ctx["raw"]
    valid = ctx["valid"]
    inv = ctx["inv_transform"]
    Hc, Wc = ctx["height"], ctx["width"]
    to_raster = ctx.get("to_raster", None)

    home_mask = np.asarray(home_mask, dtype=bool)
    rr, cc = np.nonzero(home_mask)
    mask_area = int(rr.size)
    if mask_area == 0:
        return 0, np.nan, np.nan, 0

    xs = global_min_x + cc.astype(np.float64)
    ys = global_min_y + rr.astype(np.float64)

    if to_raster is not None:
        xs, ys = to_raster.transform(xs, ys)

    cc_f, rr_f = inv * (xs, ys)
    cci = np.floor(cc_f).astype(np.int64)
    rri = np.floor(rr_f).astype(np.int64)

    ok = (rri >= 0) & (rri < Hc) & (cci >= 0) & (cci < Wc)
    if not np.any(ok):
        return mask_area, np.nan, np.nan, 0

    rri = rri[ok]; cci = cci[ok]
    ok2 = valid[rri, cci]
    if not np.any(ok2):
        return mask_area, np.nan, np.nan, 0

    vals = raw[rri[ok2], cci[ok2]].astype(np.float64)
    mu = float(np.mean(vals))
    sd = float(np.std(vals, ddof=0))
    return mask_area, mu, sd, int(vals.size)


# ============================================================
# Worker globals (spawn initializer fills these)
# ============================================================
G_BETTONG = None
G_PADDING = None
G_GLOBAL_MIN_X = None
G_GLOBAL_MIN_Y = None
G_T = None
G_TIF_DIR = None
G_BBOX_MGA55 = None

G_KERNELS = None  # created inside worker
G_MATS = None     # mats passed in, then kernels built in initializer

WORKER_CTX_CACHE: Dict[str, Dict[str, Any]] = {}


def _init_worker(bettong, mats, padding, global_min_x, global_min_y, T, tif_dir, bbox_mga55):
    """
    Spawn initializer: runs inside each worker process.
    Creates native kernels inside worker (critical).
    """
    global G_BETTONG, G_PADDING, G_GLOBAL_MIN_X, G_GLOBAL_MIN_Y, G_T
    global G_TIF_DIR, G_BBOX_MGA55, G_KERNELS, G_MATS, WORKER_CTX_CACHE

    G_BETTONG = bettong
    G_PADDING = int(padding)
    G_GLOBAL_MIN_X = int(global_min_x)
    G_GLOBAL_MIN_Y = int(global_min_y)
    G_T = int(T)

    G_TIF_DIR = str(tif_dir)
    G_BBOX_MGA55 = tuple(bbox_mga55)

    # Build kernels INSIDE worker from numpy matrices (no native pointers passed)
    m1, m2, m3 = mats
    kernel_1 = correlated_kernels_from_matrix(m1, reso, reso, 1)
    kernel_2 = correlated_kernels_from_matrix(m2, reso, reso, 1)
    kernel_3 = correlated_kernels_from_matrix(m3, reso, reso, 1)
    G_KERNELS = [kernel_1, kernel_2, kernel_3]

    WORKER_CTX_CACHE = {}


def get_ctx_for_datetime(dt: datetime):
    season_tif = season_tif_path_for_date(dt, G_TIF_DIR)
    if season_tif in WORKER_CTX_CACHE:
        return WORKER_CTX_CACHE[season_tif], season_tif

    ctx = build_ctx_for_season_window_continuous(
        season_tif=season_tif,
        bbox_mga55=G_BBOX_MGA55,
        nodata_val_fallback=255.0,
    )
    WORKER_CTX_CACHE[season_tif] = ctx
    return ctx, season_tif


def compute_one_segment_job(i: int, b_id: str):
    # all globals are worker-local under spawn
    x0, y0, t0, s0 = G_BETTONG[i]
    x1, y1, t1, s1 = G_BETTONG[i + 1]

    seg_min_x = min(x0, x1) - G_PADDING
    seg_min_y = min(y0, y1) - G_PADDING
    seg_max_x = max(x0, x1) + G_PADDING
    seg_max_y = max(y0, y1) + G_PADDING

    seg_W = seg_max_x - seg_min_x
    seg_H = seg_max_y - seg_min_y

    seg = compute_segment_util(
        G_BETTONG, G_KERNELS, i, seg_min_x, seg_min_y,
        W=seg_W, H=seg_H, T=G_T
    )
    if seg is None:
        return None

    core_mask, _ = ud_isopleth_mask(seg, p=0.9)

    dx = seg_min_x - G_GLOBAL_MIN_X
    dy = seg_min_y - G_GLOBAL_MIN_Y

    # season by t0
    ctx, season_tif_used = get_ctx_for_datetime(t0)

    ud_mass, ud_mu, ud_sd, ud_n_valid = segment_canopy_continuous_fast_ctx(seg, seg_min_x, seg_min_y, ctx)
    core_area, core_mu, core_sd, core_n_valid = core_canopy_continuous_fast_ctx(seg, core_mask, seg_min_x, seg_min_y, ctx)

    row = {
        "segment_index": i,
        "t0": t0,
        "t1": t1,
        "x0": x0, "y0": y0,
        "x1": x1, "y1": y1,
        "state0": s0,
        "state1": s1,
        "bettong_id": b_id,
        "season_tif": os.path.basename(season_tif_used),

        "ud_mass": float(np.sum(seg)),
        "ud_mean_canopy": ud_mu,
        "ud_sd_canopy": ud_sd,
        "ud_n_valid": int(ud_n_valid),

        "core_area": int(core_area),
        "core_mean_canopy": core_mu,
        "core_sd_canopy": core_sd,
        "core_n_valid": int(core_n_valid),
    }

    return (i, seg, core_mask, dx, dy, row, season_tif_used)


# ============================================================
# MAIN
# ============================================================
def main():
    # Force spawn for the whole program
    mp.set_start_method("spawn", force=True)

    # --------- user config ----------
    file_path = "/home/mart/Code/eco/Gardiner- Habitat.Perception.Bettongs.csv"
    target_id = sys.argv[1].lower()

    padding = 100
    T = 15

    tif_dir = "csv_out/tif_in"
    out_folder = "csv_out/proper"
    os.makedirs(out_folder, exist_ok=True)

    out_csv_segments = f"{out_folder}/{target_id}_segment_pgreen_continuous.csv"
    out_csv_home = f"{out_folder}/{target_id}_homerange_pgreen_continuous.csv"
    out_plot = f"{out_folder}/{target_id}_homerange_pgreen_continuous_plot.png"
    # --------------------------------

    # Load tracks
    bettongs = process_bettongs(file_path)
    if target_id not in bettongs:
        raise KeyError(f"Unknown bettong_id '{target_id}'. Available: {sorted(bettongs.keys())[:10]} ...")
    bettong = bettongs[target_id]
    print(f"  Selected bettong '{target_id}' with {len(bettong)} fixes")

    # Fit step models -> matrices (picklable)
    m_1, m_2, m_3 = build_step_matrices(bettongs, step_size=15)
    mats = (m_1, m_2, m_3)

    # Global UD grid extent (MGA55)
    xs = [x for (x, y, *_rest) in bettong]
    ys = [y for (x, y, *_rest) in bettong]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    W = (max_x - min_x) + 2 * padding
    H = (max_y - min_y) + 2 * padding
    global_min_x = min_x - padding
    global_min_y = min_y - padding
    print(f"  Global W: {W}, H: {H}")

    bbox_mga55 = (
        float(global_min_x),
        float(global_min_y),
        float(global_min_x + W),
        float(global_min_y + H),
    )

    # Global accumulators (in parent only; parent never touches native kernels)
    total_utilization = np.zeros((H, W), dtype=np.float64)
    home_range_mask = np.zeros((H, W), dtype=bool)

    n_segments = len(bettong) - 1

    # Keep exactly 31 workers as requested
    max_workers = 31
    print(f"  Parallelizing {n_segments} segment(s) with max_workers={max_workers} (spawn)...")

    ctx_mp = mp.get_context("spawn")

    rows = []
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx_mp,
        initializer=_init_worker,
        initargs=(bettong, mats, padding, global_min_x, global_min_y, T, tif_dir, bbox_mga55),
    ) as ex:
        futures = [ex.submit(compute_one_segment_job, i, target_id) for i in range(n_segments)]
        for fut in as_completed(futures):
            res = fut.result()
            if res is None:
                continue
            i, seg, core_mask, dx, dy, row, season_tif_used = res
            add_with_offset(total_utilization, seg, dx, dy)
            or_mask_with_offset(home_range_mask, core_mask, dx, dy)
            rows.append(row)
            print(f"  Finished segment {i}            ", end="\r")

    # Write per-segment continuous CSV
    if rows:
        df_seg = pd.DataFrame(rows).sort_values("segment_index")
        df_seg.to_csv(out_csv_segments, index=False)
        print(f"  \nWrote {len(df_seg)} rows to {out_csv_segments}")
    else:
        print("  \nNo valid segments produced results; nothing written (segments).")

    # Home-range continuous summary: use first-fix season for the home-range raster context
    first_dt = bettong[0][2]
    first_season_tif = season_tif_path_for_date(first_dt, tif_dir)
    home_ctx = build_ctx_for_season_window_continuous(
        season_tif=first_season_tif,
        bbox_mga55=bbox_mga55,
        nodata_val_fallback=255.0,
    )

    mask_area, mu_hr, sd_hr, n_valid = home_range_canopy_continuous_ctx(
        home_mask=home_range_mask,
        global_min_x=global_min_x,
        global_min_y=global_min_y,
        ctx=home_ctx,
    )

    df_home = pd.DataFrame([{
        "bettong_id": target_id,
        "season_tif": os.path.basename(first_season_tif),
        "mask_area": int(mask_area),
        "mask_mean_canopy": mu_hr,
        "mask_sd_canopy": sd_hr,
        "mask_n_valid": int(n_valid),
    }])
    df_home.to_csv(out_csv_home, index=False)
    print("  Home-range continuous canopy summary:")
    print(df_home.to_string(index=False))
    print(f"  Wrote home-range table to {out_csv_home}")

    # Optional plot: raster window + home-range outline
    try:
        raw = home_ctx["raw"]
        extent = home_ctx["extent"]
        plt.figure(figsize=(9, 9))
        plt.imshow(raw, extent=extent, origin="upper")
        plt.title(f"{target_id}: canopy (season {os.path.basename(first_season_tif)}) + home-range mask")
        plt.tight_layout()
        plt.savefig(out_plot, dpi=200)
        plt.close()
        print(f"  Wrote plot to {out_plot}")
    except Exception as e:
        print(f"  Plot skipped due to error: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m tests.bettong_util_canopy_cont <bettong_id>")
    main()
