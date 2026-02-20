#!/usr/bin/env python3
"""
bettong_ndvi_ud_onefile.py

End-to-end, single-file script that:

1) Reads bettong CSV (handles MD/DM datetime formats by ID sets)
2) Fits 3 GMMs (one per state) from step vectors (as in your workflow)
3) Builds correlated kernels from those fitted matrices (random_walk_package)
4) Fetches Landsat Collection 2 L2 SR for a selected time window (e.g., 2016-06),
   computes NDVI, and writes a float NDVI raster in MGA55 (EPSG:28355)
5) Classifies NDVI into K discrete classes (uint8 raster, nodata=255) so it behaves
   like your WorldCover raster (categorical grid)
6) Computes per-segment utilization distributions on per-segment local grids (PARALLEL)
7) Aggregates per-segment UDs into a global UD and a union 95% mask (SERIAL aggregation)
8) Summarizes UD mass by NDVI-class code (same logic as your WorldCover mass table)
9) Optionally plots the NDVI-class raster + UD/home-range contours + trajectory

Requirements:
- random_walk_package importable
- pystac_client, planetary_computer, stackstac, rioxarray, rasterio, pyproj installed
- scikit-learn installed (GaussianMixture + MiniBatchKMeans)
- Linux recommended for fork-based multiprocessing (your current approach)
"""

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from sklearn.mixture import GaussianMixture
from sklearn.cluster import MiniBatchKMeans

from pystac_client import Client
import planetary_computer
import stackstac
import rioxarray  # registers .rio accessor
from scipy.ndimage import gaussian_filter

import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling


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
from tests.bettong_util_parallel import segment_worldcover_mass_fast


# -------------------------
# CRS helpers
# -------------------------
CRS_MGA55 = "EPSG:28355"
CRS_WGS84 = "EPSG:4326"

to_wgs84 = Transformer.from_crs(CRS_MGA55, CRS_WGS84, always_xy=True)


def bbox_mga55_to_wgs84(minx, miny, maxx, maxy):
    corners = [(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]
    lons_lats = [to_wgs84.transform(x, y) for x, y in corners]
    lons, lats = zip(*lons_lats)
    return (min(lons), min(lats), max(lons), max(lats))


def compute_bbox_from_points_mga55(points, pad_m=100):
    xs = [x for (x, y, *_rest) in points]
    ys = [y for (x, y, *_rest) in points]
    return (min(xs) - pad_m, min(ys) - pad_m, max(xs) + pad_m, max(ys) + pad_m)


# -------------------------
# CSV processing
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
        bettong_id = row["ID"]

        if bettong_id in MD_IDS:
            date_time_obj = datetime.strptime(row["datetime"], "%m/%d/%y %H:%M")
        elif bettong_id in DM_IDS:
            date_time_obj = datetime.strptime(row["datetime"], "%d/%m/%y %H:%M")
        else:
            raise ValueError(f"UNKNOWN ID: {bettong_id}")

        state = int(row["states"])
        bettongs.setdefault(bettong_id, []).append((int(row["x"]), int(row["y"]), date_time_obj, state))

    for bettong_id in bettongs:
        bettongs[bettong_id].sort(key=lambda entry: entry[2])

    return bettongs


# -------------------------
# Step model fitting (your GMM workflow)
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
                if i < 2:
                    count_discarded += 1
                    continue
                a = (entries[i - 2][0], entries[i - 2][1])
                b = (entries[i - 1][0], entries[i - 1][1])
                steps[entries[i - 1][3] - 1].append(delta_vector(b, a, step_size))
            else:
                count_discarded += 1

    if count_total > 0:
        print(f"  total of {count_total} steps, {count_discarded / count_total:.2%} discarded")

    return steps


def fit_data(axs, steps):
    data = np.array(steps)
    n_components = 3
    gmm = GaussianMixture(n_components=n_components, covariance_type="full")
    gmm.fit(data)

    x = np.linspace(-rnge, rnge, reso)
    y = np.linspace(-rnge, rnge, reso)
    X, Y = np.meshgrid(x, y)
    grid = np.column_stack([X.ravel(), Y.ravel()])

    log_density = gmm.score_samples(grid)
    density = np.exp(log_density)
    Z = density.reshape(X.shape)

    axs.imshow(
        Z,
        extent=(-rnge, rnge, -rnge, rnge),
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
    )
    return Z


def generate_heatmap(axs, coords):
    coords = np.array(coords)
    x_edges = np.linspace(-rnge, rnge, reso)
    y_edges = np.linspace(-rnge, rnge, reso)
    heatmap, _, _ = np.histogram2d(coords[:, 0], coords[:, 1], bins=[x_edges, y_edges])

    axs.imshow(
        heatmap.T,
        extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
        origin="lower",
        cmap="viridis",
    )


def pure_grouped(bettongs, step_size=1):
    a, b, c = calculate_steps_grouped(bettongs, step_size)

    fig, axs = plt.subplots(2, 3, figsize=(12, 6))
    generate_heatmap(axs[0, 0], a)
    generate_heatmap(axs[0, 1], b)
    generate_heatmap(axs[0, 2], c)

    m_a = fit_data(axs[1, 0], a)
    m_b = fit_data(axs[1, 1], b)
    m_c = fit_data(axs[1, 2], c)

    plt.tight_layout()
    return m_a, m_b, m_c

def fetch_ndvi_s2_mga55(
    bbox_wgs84,
    bbox_mga55,
    start_date,
    end_date,
    max_cloud=80,
    output_filename="ndvi_s2_float_28355.tif",
):
    """
    Sentinel-2 L2A (10m) NDVI composite in MGA55.

    - Uses B04 (red, 10m) + B08 (nir, 10m)
    - Uses SCL (20m) for cloud/shadow masking, resampled to 10m
    - Outputs median NDVI over time window as float32 GeoTIFF in EPSG:28355

    bbox_wgs84: (minlon, minlat, maxlon, maxlat) for STAC search
    bbox_mga55: (minx,  miny,  maxx,  maxy)      for stackstac(bounds=...)
    """

    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=list(bbox_wgs84),
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )

    items = list(search.items())
    if not items:
        raise RuntimeError("No Sentinel-2 L2A items found for that AOI/date range.")

    # Stack B04, B08, SCL into a 3-band cube. Use projected bounds to avoid out_bounds=None.
    # band_coords=False avoids weird band coordinate types; we'll select by index with isel().
    stack = stackstac.stack(
        items,
        assets=["B04", "B08", "SCL"],
        epsg=28355,
        resolution=10,
        bounds=bbox_mga55,
        chunksize=2048,
        band_coords=False,
        resampling=Resampling.bilinear,   # important for SCL (categorical)
    ).astype("float32")

    # IMPORTANT: use isel, not sel(band="B04") (avoids your xarray dtype crash)
    red = stack.isel(band=0) * 1e-4
    nir = stack.isel(band=1) * 1e-4
    scl = stack.isel(band=2).astype("uint8")

    # SCL mask: keep only “clear-ish” classes
    # Common mask-out: 0 no data, 1 saturated/defective, 3 cloud shadow,
    # 7/8/9 clouds, 10 cirrus, 11 snow/ice.
    bad = (scl == 0) | (scl == 1) | (scl == 3) | (scl == 7) | (scl == 8) | (scl == 9) | (scl == 10) | (scl == 11)
    clear = ~bad

    ndvi = (nir - red) / (nir + red + 1e-6)
    ndvi = ndvi.where(clear)

    # Composite (median over time)
    ndvi_med = ndvi.median(dim="time", skipna=True)

    # Write CRS + save
    ndvi_med = ndvi_med.rio.write_crs(CRS_MGA55)
    ndvi_med.rio.to_raster(output_filename, compress="LZW", dtype="float32")

    return output_filename


def debug_raster_stats(tif):
    with rasterio.open(tif) as src:
        arr = src.read(1).astype("float32")
        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)

        print("Raster CRS:", src.crs)
        print("Raster bounds:", src.bounds)
        print("Raster shape:", arr.shape)
        print("NaN %:", float(np.isnan(arr).mean() * 100))
        if np.isfinite(arr).any():
            print("Min/Max:", float(np.nanmin(arr)), float(np.nanmax(arr)))
        else:
            print("All NaN/nodata (window too cloudy / mask too strict).")


# -------------------------
# NDVI classification -> uint8 class raster like WorldCover
# -------------------------
def classify_ndvi_raster_kmeans(
    ndvi_tif_in: str,
    class_tif_out: str,
    k: int = 10,
    sample_n: int = 250_000,
    nodata_val: int = 255,
    random_state: int = 0,
):
    """
    Classify NDVI float raster into k ordered classes 0..k-1 (low NDVI -> high NDVI).
    Writes uint8 raster with nodata=255.
    Returns class_info dict: class_id -> center label (for CSV readability).
    """
    with rasterio.open(ndvi_tif_in) as src:
        ndvi = src.read(1).astype("float32")
        prof = src.profile.copy()
        nd_nodata = src.nodata

    valid = np.isfinite(ndvi)
    if nd_nodata is not None:
        valid &= (ndvi != nd_nodata)

    vals = ndvi[valid]
    if vals.size == 0:
        raise RuntimeError("NDVI raster has no valid pixels to classify.")

    vals = np.clip(vals, -1, 1)

    if vals.size > sample_n:
        rng = np.random.default_rng(random_state)
        sample = rng.choice(vals, size=sample_n, replace=False)
    else:
        sample = vals

    km = MiniBatchKMeans(
        n_clusters=k,
        random_state=random_state,
        batch_size=8192,
        n_init="auto",
    )
    km.fit(sample.reshape(-1, 1))

    centers = km.cluster_centers_.reshape(-1)
    order = np.argsort(centers)  # low -> high
    remap = np.empty(k, dtype=np.uint8)
    remap[order] = np.arange(k, dtype=np.uint8)

    cls = np.full(ndvi.shape, nodata_val, dtype=np.uint8)
    raw = km.predict(np.clip(ndvi[valid], -1, 1).reshape(-1, 1)).astype(np.int32)
    cls[valid] = remap[raw]

    # Build class info (center per ordered class)
    class_info = {}
    for c in range(k):
        raw_id = int(np.where(remap == c)[0][0])
        class_info[int(c)] = {
            "center": float(centers[raw_id]),
            "label": f"NDVI center {centers[raw_id]:.2f}",
        }

    prof.update(dtype="uint8", count=1, nodata=nodata_val, compress="LZW")
    with rasterio.open(class_tif_out, "w", **prof) as dst:
        dst.write(cls, 1)

    return class_info


def make_ndvi_classes_dict(class_info, nodata_val=255):
    """
    Create a mapping similar to WC_CLASSES: code -> (name, color_hex_or_None)
    Colors are optional; you can ignore them.
    """
    out = {nodata_val: ("No Data", "#000000")}
    for c, info in class_info.items():
        out[int(c)] = (info["label"], None)
    return out


# -------------------------
# Utilization distribution helpers (your existing logic)
# -------------------------
def ud_isopleth_level(ud, p=0.95):
    ud = np.asarray(ud, dtype=float)
    ud = np.clip(ud, 0, None)

    total = ud.sum()
    if total <= 0:
        return np.nan

    udn = ud / total
    flat = udn.ravel()
    idx = np.argsort(flat)[::-1]
    csum = np.cumsum(flat[idx])

    k = np.searchsorted(csum, p, side="left")
    level = flat[idx[k]]
    return level


def ud_isopleth_mask(ud, p=0.95):
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
    return mask, level


def points_to_np_float64(points, n: int) -> np.ndarray:
    try:
        arr = np.fromiter(points, dtype=np.float64, count=n)
        if arr.size == n:
            return arr
    except TypeError:
        pass
    return np.fromiter((float(points[i]) for i in range(n)), dtype=np.float64, count=n)


def compute_segment_util(bettong, kernels, i: int, min_x: int, min_y: int, *, W: int, H: int, T: int):
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

    x0 = max(dx, 0)
    y0 = max(dy, 0)
    x1 = min(dx + Ws, Wt)
    y1 = min(dy + Hs, Ht)
    if x0 >= x1 or y0 >= y1:
        return

    sx0 = x0 - dx
    sy0 = y0 - dy
    sx1 = sx0 + (x1 - x0)
    sy1 = sy0 + (y1 - y0)

    total_arr[y0:y1, x0:x1] += seg_arr[sy0:sy1, sx0:sx1]


def or_mask_with_offset(global_mask, local_mask, dx, dy):
    Ht, Wt = global_mask.shape
    Hs, Ws = local_mask.shape

    x0 = max(dx, 0)
    y0 = max(dy, 0)
    x1 = min(dx + Ws, Wt)
    y1 = min(dy + Hs, Ht)
    if x0 >= x1 or y0 >= y1:
        return

    sx0 = x0 - dx
    sy0 = y0 - dy
    sx1 = sx0 + (x1 - x0)
    sy1 = sy0 + (y1 - y0)

    global_mask[y0:y1, x0:x1] |= local_mask[sy0:sy1, sx0:sx1]


# -------------------------
# FAST categorical-mass summary (works for NDVI classes too)
# -------------------------
G_CLASS_BAND = None
G_CLASS_INV_TRANSFORM = None
G_CLASS_HEIGHT = None
G_CLASS_WIDTH = None
G_CLASS_LABELS = None  # dict: code -> name


def segment_class_mass_fast(seg, seg_min_x, seg_min_y):
    """
    Vectorized aggregation of UD mass by categorical raster code.
    Uses globally loaded class raster (uint8).
    """
    global G_CLASS_BAND, G_CLASS_INV_TRANSFORM, G_CLASS_HEIGHT, G_CLASS_WIDTH, G_CLASS_LABELS

    seg = np.asarray(seg, dtype=np.float64)
    H, W = seg.shape
    rows, cols = np.nonzero(seg > 0)
    if rows.size == 0:
        return pd.DataFrame(columns=["code", "name", "mass", "proportion"]), 0.0

    vals = seg[rows, cols]

    xs = seg_min_x + cols.astype(np.float64)
    ys = seg_min_y + rows.astype(np.float64)

    cc_f, rr_f = G_CLASS_INV_TRANSFORM * (xs, ys)
    cc = np.floor(cc_f).astype(np.int64)
    rr = np.floor(rr_f).astype(np.int64)

    ok = (rr >= 0) & (rr < G_CLASS_HEIGHT) & (cc >= 0) & (cc < G_CLASS_WIDTH)
    total_mass = float(np.sum(seg))
    if not np.any(ok):
        return pd.DataFrame(columns=["code", "name", "mass", "proportion"]), total_mass

    rr = rr[ok]
    cc = cc[ok]
    vals = vals[ok]

    codes = G_CLASS_BAND[rr, cc].astype(np.int64)

    df = pd.DataFrame({"code": codes, "mass": vals})
    g = df.groupby("code", as_index=False)["mass"].sum()
    g["proportion"] = g["mass"] / total_mass if total_mass > 0 else np.nan
    g["name"] = g["code"].map(lambda c: G_CLASS_LABELS.get(int(c), "Unknown"))
    g = g.sort_values("mass", ascending=False)
    return g[["code", "name", "mass", "proportion"]], total_mass


# -------------------------
# PARALLEL worker globals (Linux fork inherits these)
# -------------------------
G_BETTONG = None
G_KERNELS = None
G_PADDING = None
G_GLOBAL_MIN_X = None
G_GLOBAL_MIN_Y = None
G_T = None


def compute_one_segment_job(i: int, b_id: str):
    global G_BETTONG, G_KERNELS, G_PADDING, G_GLOBAL_MIN_X, G_GLOBAL_MIN_Y, G_T

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

    seg_mask_local, _ = ud_isopleth_mask(seg, p=0.9)

    dx = seg_min_x - G_GLOBAL_MIN_X
    dy = seg_min_y - G_GLOBAL_MIN_Y

    df, total_mass = segment_class_mass_fast(seg, seg_min_x, seg_min_y)
    
    df, total_mass = segment_class_mass_fast(seg, seg_min_x, seg_min_y)
    df2, total_mass2 = segment_class_mass_fast(seg_mask_local.astype(np.float64), seg_min_x, seg_min_y)

    df2 = df2.rename(columns={"mass": "mask_mass"})
    df2 = df2.rename(columns={"proportion": "mask_proportion"})

    df = df.merge(df2, on=["code", "name"], how="outer")
    df[["mass","proportion","mask_mass","mask_proportion"]] = df[
        ["mass","proportion","mask_mass","mask_proportion"]
    ].fillna(0)
    
    df.insert(0, "segment_index", i)
    df["t0"] = t0
    df["t1"] = t1
    df["x0"] = x0
    df["y0"] = y0
    df["x1"] = x1
    df["y1"] = y1
    df["state0"] = s0
    df["state1"] = s1
    df["segment_total_mass"] = float(total_mass)
    df["bettong_id"] = b_id

    return (i, seg, seg_mask_local, dx, dy, df)


# -------------------------
# Plot background categorical raster + contours
# -------------------------
def load_class_raster_for_plot(raster_tif_mga55):
    with rasterio.open(raster_tif_mga55) as src:
        arr = src.read(1)
        extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
        nodata = src.nodata
    return arr, extent, nodata


def vegindex_cmap_paper_like():
    """
    11 discrete colors for VegIndex 0..10, matching the provided colorbar:
    light gray -> pink -> tan/brown -> yellow -> yellow-green -> dark green
    """
    colors = [
        "#f2f2f2",  # 0 very light gray
        "#f1edec",  # 1 pale pink
        "#e4b9ac",  # 2 pink
        "#e8a37a",  # 3 muted rose
        "#deaa3d",  # 4 tan/brown
        "#dbcc14",  # 5 tan/yellow-brown
        "#beda14",  # 6 yellow
        "#75c705",  # 7 yellow-green
        "#4cb80d",  # 8 light green
        "#21ad02",  # 9 green
        "#2b8828",  # 10 dark green
    ]
    return ListedColormap(colors, name="VegIndexPaperLike")

def vegindex_norm(n_classes=11):
    # boundaries for integer classes 0..10
    boundaries = np.arange(-0.5, n_classes + 0.5, 1.0)
    return BoundaryNorm(boundaries, n_classes)

def plot_class_with_trajectory_and_contour(
    bettong, raster_tif_mga55,
    utilization_distribution, home_range_mask,
    global_min_x, global_min_y,
    class_labels: dict,         # optional: used only if you want class names on ticks
    n_classes=11,               # VegIndex 0..10
    tick_values=(2, 4, 6, 8, 10),
    out_path=None,
):
    xs = [x for (x, y, *_rest) in bettong]
    ys = [y for (x, y, *_rest) in bettong]

    arr, extent, nodata = load_class_raster_for_plot(raster_tif_mga55)

    # Prepare array for plotting; mask nodata as NaN so cmap.set_bad applies
    arr_plot = arr.astype("float32")
    if nodata is not None:
        arr_plot[arr == nodata] = np.nan

    cmap = vegindex_cmap_paper_like()
    norm = vegindex_norm(n_classes=n_classes)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(
        arr_plot,
        extent=extent,
        origin="upper",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    # ---- Contours ----
    ud = np.asarray(utilization_distribution, dtype=float)
    H, W = ud.shape
    x = global_min_x + np.arange(W)
    y = global_min_y + np.arange(H)

    # Home-range contour (union mask)
    mask_f = gaussian_filter(home_range_mask.astype(float), sigma=1.0)
    ax.contour(x, y, mask_f, levels=[0.5], colors="purple", linewidths=3, alpha=0.9)

    # # 95% UD contour
    # level95 = ud_isopleth_level(ud, p=0.95)
    # if np.isfinite(level95) and ud.sum() > 0:
    #     ax.contour(x, y, ud / ud.sum(), levels=[level95], colors="orange", linewidths=2, alpha=0.9)

    # ---- Trajectory ----
    # ax.plot(xs, ys, linewidth=2, color="black")
    ax.scatter(xs, ys, s=12, color="black")

    ax.set_title("Trajectory over VegIndex (NDVI k-means 0–10) + UD + Home-range contour")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    # Colorbar: match your reference (ticks at 2,4,6,8,10)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=list(tick_values))
    cbar.set_label("VegIndex (k-means NDVI class)")

    # Optional: if you want tick labels to be the class label strings instead of numbers:
    # cbar.ax.set_yticklabels([class_labels.get(int(t), str(int(t))) for t in tick_values])

    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, dpi=300)
    else:
        plt.show()


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    # --------- user config ----------
    file_path = "/home/mart/Code/eco/Gardiner- Habitat.Perception.Bettongs.csv"
    target_id = sys.argv[1]

    start_date = "2017-04-01"
    end_date = "2017-08-30"
    padding = 150
    T = 15
    K_CLASSES = 11

    ndvi_float_tif = f"ndvi_{start_date}_{end_date}_float_28355.tif"
    ndvi_class_tif = f"ndvi_{start_date}_{end_date}_classK{K_CLASSES}_28355.tif"
    out_folder = "csv_out/proper"
    out_csv_segments = f"{out_folder}/{target_id}_segment_ndvi_class_mass_.csv"
    out_csv_home = f"{out_folder}/{target_id}_homerange_ndvi_class_mass.csv"
    out_homerange_plot = f"{out_folder}/{target_id}_homerange_ndvi_plot.png"
    # --------------------------------

    bettongs = process_bettongs(file_path)
    bettong = bettongs[target_id]
    print(f"Selected bettong '{target_id}' with {len(bettong)} fixes")

    # Fit step models and build kernels (global across animals, your original approach)
    m_1, m_2, m_3 = pure_grouped(bettongs, 15)
    kernel_1 = correlated_kernels_from_matrix(m_1, reso, reso, 1)
    kernel_2 = correlated_kernels_from_matrix(m_2, reso, reso, 1)
    kernel_3 = correlated_kernels_from_matrix(m_3, reso, reso, 1)
    kernels = [kernel_1, kernel_2, kernel_3]

    # Global (track-level) grid definition
    xs, ys = zip(*[(x, y) for x, y, *_ in bettong])
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    W = (max_x - min_x) + 2 * padding
    H = (max_y - min_y) + 2 * padding
    global_min_x = min_x - padding
    global_min_y = min_y - padding
    print(f"Global W: {W}, H: {H}")

    bbox_mga = compute_bbox_from_points_mga55(bettong, pad_m=padding)
    bbox_wgs = bbox_mga55_to_wgs84(*bbox_mga)

    ndvi_float = fetch_ndvi_s2_mga55(
        bbox_wgs84=bbox_wgs,
        bbox_mga55=bbox_mga,
        start_date=start_date,
        end_date=end_date,
        max_cloud=80,
        output_filename="ndvi_s2_2016-06_float_28355.tif",
    )

    # Classify NDVI into K classes (uint8 grid like WorldCover)
    class_info = classify_ndvi_raster_kmeans(
        ndvi_tif_in=ndvi_float,
        class_tif_out=ndvi_class_tif,
        k=K_CLASSES,
        sample_n=250_000,
        nodata_val=255,
        random_state=0,
    )
    ndvi_classes = make_ndvi_classes_dict(class_info, nodata_val=255)
    class_labels = {code: name for code, (name, _color) in ndvi_classes.items()}
    print("NDVI class labels:")
    for k, v in sorted(class_labels.items()):
        print(f"  {k}: {v}")

    # Load class raster once in the parent BEFORE forking (shared copy-on-write)
    with rasterio.open(ndvi_class_tif) as src:
        class_band = src.read(1)
        class_inv = ~src.transform
        class_h, class_w = src.height, src.width

    # Set class globals used by segment_class_mass_fast
    G_CLASS_BAND = class_band
    G_CLASS_INV_TRANSFORM = class_inv
    G_CLASS_HEIGHT = class_h
    G_CLASS_WIDTH = class_w
    G_CLASS_LABELS = class_labels

    # Segment UD parameters and global arrays
    total_utilization = np.zeros((H, W), dtype=np.float64)
    home_range_mask = np.zeros((H, W), dtype=bool)

    # Set worker globals BEFORE forking
    G_BETTONG = bettong
    G_KERNELS = kernels
    G_PADDING = padding
    G_GLOBAL_MIN_X = global_min_x
    G_GLOBAL_MIN_Y = global_min_y
    G_T = T

    n_segments = len(bettong) - 1
    max_workers = max(1, (os.cpu_count() or 2) - 1)

    # Linux: use fork explicitly
    ctx = mp.get_context("fork")

    all_dfs = []

    print(f"Parallelizing {n_segments} segment(s) with max_workers={max_workers}...")

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        futures = [ex.submit(compute_one_segment_job, i, target_id) for i in range(n_segments)]
        for fut in as_completed(futures):
            res = fut.result()
            if res is None:
                continue

            i, seg, seg_mask_local, dx, dy, df = res

            # Aggregate serially
            add_with_offset(total_utilization, seg, dx, dy)
            or_mask_with_offset(home_range_mask, seg_mask_local, dx, dy)

            all_dfs.append(df)
            print(f"Finished segment {i}")

    total_sum = float(np.sum(total_utilization))
    print(f"Total sum of total_utilization: {total_sum}")

    # Write per-segment class-mass CSV
    if all_dfs:
        big = pd.concat(all_dfs, ignore_index=True)
        big.to_csv(out_csv_segments, index=False)
        print(f"Wrote {len(big)} rows to {out_csv_segments}")
    else:
        print("No valid segments produced any results; nothing written (segments).")

    # Home-range (union mask) composition (area-based: 1 inside mask)
    df_home, total_mass_home = segment_class_mass_fast(
        home_range_mask.astype(np.float64),
        global_min_x,
        global_min_y
    )
    df_home.insert(0, "bettong_id", target_id)
    df_home.to_csv(out_csv_home, index=False)
    print("Home-range (union mask) class composition:")
    print(df_home)
    print(f"Wrote home-range table to {out_csv_home}")

    # Optional plot
    plot_class_with_trajectory_and_contour(
        bettong,
        ndvi_class_tif,
        total_utilization,
        home_range_mask,
        global_min_x,
        global_min_y,
        class_labels=class_labels,
        out_path=out_homerange_plot
    )
