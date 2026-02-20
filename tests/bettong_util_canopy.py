#!/usr/bin/env python3
"""
bettong_persistent_green_ud_onefile.py

Uses seasonal persistent-green-cover GeoTIFFs in csv_out/tif_in, chooses season by t0 date,
classifies percent->classes, and summarizes UD mass by class.

CRITICAL: raster CRS is EPSG:3577 while bettong fixes are EPSG:28355 (MGA55).
This script reprojects bettong-based coordinates to raster CRS for raster lookups and plotting.
No NDVI logic is used.
"""

import os
import sys
import glob
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

import rasterio
from scipy.ndimage import gaussian_filter
from sklearn.mixture import GaussianMixture
from pyproj import Transformer


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
# Constants / CRS
# -------------------------
CRS_MGA55 = "EPSG:28355"


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


# -------------------------
# Season logic matching your filenames
# -------------------------
def season_tag_for_date(dt: datetime) -> str:
    """
    Return the YYYYMMYYYYMM season tag used in filenames:
      DJF: Dec–Feb => m(prevDec)(Feb)
      MAM: Mar–May => m(Mar)(May)
      JJA: Jun–Aug => m(Jun)(Aug)
      SON: Sep–Nov => m(Sep)(Nov)
    """
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
    # Your actual files are dp7a2 (from `ls`)
    return f"lztmre_tas_m{season_tag_for_date(dt)}_dp7a2.tif"


def season_tif_path_for_date(dt: datetime, tif_dir: str) -> str:
    """
    Resolve full path for the season tif. If exact expected filename doesn't exist,
    fall back to searching by season tag.
    """
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


# -------------------------
# Persistent green cover classification (percent -> classes)
# -------------------------
def classify_persistent_green_equal_bins(
    tif_in: str,
    tif_out: str,
    k: int = 11,
    nodata_val: int = 255,
):
    """
    Input: persistent green cover in percent (0..100), nodata=255.
    Output: uint8 classes 0..k-1, nodata=255, equal-width bins on [0,100].
    """
    with rasterio.open(tif_in) as src:
        arr = src.read(1)
        prof = src.profile.copy()
        src_nodata = src.nodata

    nd = nodata_val
    if src_nodata is not None:
        nd = int(src_nodata)

    out = np.full(arr.shape, nodata_val, dtype=np.uint8)
    valid = (arr != nd)

    vals = arr[valid].astype(np.float32)
    vals = np.clip(vals, 0.0, 100.0)

    bin_w = 100.0 / float(k)
    cls = np.floor(vals / bin_w).astype(np.int32)
    cls = np.clip(cls, 0, k - 1).astype(np.uint8)
    out[valid] = cls

    prof.update(dtype="uint8", count=1, nodata=nodata_val, compress="LZW")
    os.makedirs(os.path.dirname(tif_out), exist_ok=True)
    with rasterio.open(tif_out, "w", **prof) as dst:
        dst.write(out, 1)

    labels = {nodata_val: "No Data"}
    for c in range(k):
        lo = c * bin_w
        hi = (c + 1) * bin_w
        if c == k - 1:
            hi = 100.0
        labels[c] = f"Persistent green {lo:.1f}–{hi:.1f}%"

    return labels


def ensure_class_tif_for_date(dt: datetime, tif_dir: str, cache_dir: str, k_classes: int, nodata_val: int = 255):
    """
    For a datetime, choose season tif, then create/reuse a classified tif in cache_dir.
    Returns (class_tif_path, labels_dict, season_tif_path).
    """
    season_tif = season_tif_path_for_date(dt, tif_dir)
    base = os.path.splitext(os.path.basename(season_tif))[0]
    class_tif = os.path.join(cache_dir, f"{base}_classK{k_classes}.tif")

    if os.path.exists(class_tif):
        bin_w = 100.0 / float(k_classes)
        labels = {nodata_val: "No Data"}
        for c in range(k_classes):
            lo = c * bin_w
            hi = (c + 1) * bin_w
            if c == k_classes - 1:
                hi = 100.0
            labels[c] = f"Persistent green {lo:.1f}–{hi:.1f}%"
        return class_tif, labels, season_tif

    labels = classify_persistent_green_equal_bins(
        tif_in=season_tif,
        tif_out=class_tif,
        k=k_classes,
        nodata_val=nodata_val,
    )
    return class_tif, labels, season_tif


# -------------------------
# UD helpers
# -------------------------
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
# FAST categorical-mass summary (CRS-safe)
# -------------------------
def segment_class_mass_fast_ctx(seg, seg_min_x, seg_min_y, ctx):
    """
    seg is on a local grid whose coordinates are MGA55 meters:
      x = seg_min_x + col
      y = seg_min_y + row

    Raster in ctx may be EPSG:3577 (or other). We transform (x,y) from MGA55 -> raster_crs
    before applying inv_transform into raster row/col.
    """
    band = ctx["band"]
    inv = ctx["inv_transform"]
    Hc = ctx["height"]
    Wc = ctx["width"]
    labels = ctx["labels"]
    to_raster = ctx.get("to_raster", None)

    seg = np.asarray(seg, dtype=np.float64)
    H, W = seg.shape
    rows, cols = np.nonzero(seg > 0)
    if rows.size == 0:
        return pd.DataFrame(columns=["code", "name", "mass", "proportion"]), 0.0

    vals = seg[rows, cols]

    xs = seg_min_x + cols.astype(np.float64)
    ys = seg_min_y + rows.astype(np.float64)

    # Reproject MGA55 -> raster CRS if needed
    if to_raster is not None:
        xs, ys = to_raster.transform(xs, ys)

    cc_f, rr_f = inv * (xs, ys)
    cc = np.floor(cc_f).astype(np.int64)
    rr = np.floor(rr_f).astype(np.int64)

    ok = (rr >= 0) & (rr < Hc) & (cc >= 0) & (cc < Wc)
    total_mass = float(np.sum(seg))
    if not np.any(ok):
        return pd.DataFrame(columns=["code", "name", "mass", "proportion"]), total_mass

    rr = rr[ok]
    cc = cc[ok]
    vals = vals[ok]

    codes = band[rr, cc].astype(np.int64)

    df = pd.DataFrame({"code": codes, "mass": vals})
    g = df.groupby("code", as_index=False)["mass"].sum()
    g["proportion"] = g["mass"] / total_mass if total_mass > 0 else np.nan
    g["name"] = g["code"].map(lambda c: labels.get(int(c), "Unknown"))
    g = g.sort_values("mass", ascending=False)
    return g[["code", "name", "mass", "proportion"]], total_mass


# -------------------------
# Plot helpers (CRS-safe)
# -------------------------
def vegindex_cmap_paper_like():
    colors = [
        "#f2f2f2",
        "#f1edec",
        "#e4b9ac",
        "#e8a37a",
        "#deaa3d",
        "#dbcc14",
        "#beda14",
        "#75c705",
        "#4cb80d",
        "#21ad02",
        "#2b8828",
    ]
    return ListedColormap(colors, name="VegIndexPaperLike")


def vegindex_norm(n_classes=11):
    boundaries = np.arange(-0.5, n_classes + 0.5, 1.0)
    return BoundaryNorm(boundaries, n_classes)


def load_class_raster_for_plot(raster_tif):
    with rasterio.open(raster_tif) as src:
        arr = src.read(1)
        extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
        nodata = src.nodata
        crs = src.crs
    return arr, extent, nodata, crs


from rasterio.windows import from_bounds
from rasterio.windows import transform as window_transform

def load_class_raster_for_plot_window(
    raster_tif: str,
    bbox_mga55: tuple[float, float, float, float],  # (minx, miny, maxx, maxy) in EPSG:28355
    to_raster: Transformer | None,
    pad: float = 0.0,
):
    """
    Read only the raster window overlapping bbox_mga55 (optionally padded), reprojecting bbox to raster CRS.
    Returns: (array, extent, nodata, raster_crs)
      extent is in raster CRS: (left, right, bottom, top)
    """
    minx, miny, maxx, maxy = bbox_mga55
    minx -= pad; miny -= pad; maxx += pad; maxy += pad

    # Transform bbox to raster CRS for windowing
    if to_raster is not None:
        # Transform all 4 corners then take min/max
        xs = np.array([minx, minx, maxx, maxx], dtype=np.float64)
        ys = np.array([miny, maxy, miny, maxy], dtype=np.float64)
        xr, yr = to_raster.transform(xs, ys)
        left, right = float(np.min(xr)), float(np.max(xr))
        bottom, top = float(np.min(yr)), float(np.max(yr))
    else:
        left, bottom, right, top = float(minx), float(miny), float(maxx), float(maxy)

    with rasterio.open(raster_tif) as src:
        raster_crs = src.crs
        nodata = src.nodata

        # Build a window from bounds in raster CRS
        win = from_bounds(left, bottom, right, top, transform=src.transform)

        # Clip window to raster bounds to avoid empty / out-of-range reads
        win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))

        arr = src.read(1, window=win)
        win_tr = window_transform(win, src.transform)

        # Build extent for imshow from window transform
        # Note: win_tr gives top-left of window; pixel sizes from transform
        w = arr.shape[1]
        h = arr.shape[0]
        x0, y0 = win_tr * (0, 0)       # upper-left corner
        x1, y1 = win_tr * (w, h)       # lower-right corner (because y pixel size is negative typically)

        # Ensure extent is (left, right, bottom, top)
        extent = (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))

    return arr, extent, nodata, raster_crs


def plot_class_with_trajectory_and_contour(
    bettong,
    raster_tif,
    utilization_distribution,
    home_range_mask,
    global_min_x_mga55,
    global_min_y_mga55,
    to_raster: Transformer | None,
    n_classes=11,
    tick_values=(2, 4, 6, 8, 10),
    out_path=None,
    title_suffix="Persistent green cover (binned)",
):
    ud = np.asarray(utilization_distribution, dtype=float)
    H, W = ud.shape

    # This is exactly the bbox you asked for
    bbox_mga55 = (
        float(global_min_x_mga55),
        float(global_min_y_mga55),
        float(global_min_x_mga55 + W),
        float(global_min_y_mga55 + H),
    )

    # Read only the raster window covering the UD bbox (+ optional pad in meters in MGA55)
    arr, extent, nodata, raster_crs = load_class_raster_for_plot_window(
        raster_tif=raster_tif,
        bbox_mga55=bbox_mga55,
        to_raster=to_raster,
        pad=0.0,   # set e.g. 200 if you want some margin
    )

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

    # Build UD grid coords in MGA55, transform to raster CRS for contour overlay
    xs_mga = global_min_x_mga55 + np.arange(W, dtype=np.float64)
    ys_mga = global_min_y_mga55 + np.arange(H, dtype=np.float64)
    X_mga, Y_mga = np.meshgrid(xs_mga, ys_mga)

    if to_raster is not None:
        Xr, Yr = to_raster.transform(X_mga, Y_mga)
    else:
        Xr, Yr = X_mga, Y_mga

    mask_f = gaussian_filter(home_range_mask.astype(float), sigma=1.0)
    ax.contour(Xr, Yr, mask_f, levels=[0.5], colors="purple", linewidths=3, alpha=0.9)

    # Trajectory points (transform MGA55 -> raster CRS)
    bx = np.array([x for x, y, *_ in bettong], dtype=np.float64)
    by = np.array([y for x, y, *_ in bettong], dtype=np.float64)
    if to_raster is not None:
        bx, by = to_raster.transform(bx, by)
    ax.scatter(bx, by, s=12, color="black")

    # Zoom axes to the same plotted bbox (in raster CRS)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    ax.set_title(f"Trajectory over classes + UD + Home-range contour\n{title_suffix}")
    ax.set_xlabel(f"X ({raster_crs})")
    ax.set_ylabel(f"Y ({raster_crs})")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=list(tick_values))
    cbar.set_label("Class (binned persistent green %)")

    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, dpi=300)
    else:
        plt.show()


# -------------------------
# PARALLEL worker globals
# -------------------------
G_BETTONG = None
G_KERNELS = None
G_PADDING = None
G_GLOBAL_MIN_X = None
G_GLOBAL_MIN_Y = None
G_T = None

G_TIF_DIR = None
G_CLASS_CACHE_DIR = None
G_K_CLASSES = None

# Worker-local cache
WORKER_RASTER_CTX_CACHE = {}  # key: class_tif_path -> ctx


def get_raster_ctx_for_datetime(dt: datetime):
    class_tif, labels, season_tif = ensure_class_tif_for_date(
        dt=dt,
        tif_dir=G_TIF_DIR,
        cache_dir=G_CLASS_CACHE_DIR,
        k_classes=G_K_CLASSES,
        nodata_val=255,
    )

    if class_tif in WORKER_RASTER_CTX_CACHE:
        return WORKER_RASTER_CTX_CACHE[class_tif], class_tif, season_tif

    with rasterio.open(class_tif) as src:
        raster_crs = src.crs
        if raster_crs is None:
            raise RuntimeError(f"Raster has no CRS: {class_tif}")

        to_raster = None
        # bettong coords are MGA55
        if str(raster_crs).upper() != CRS_MGA55:
            to_raster = Transformer.from_crs(CRS_MGA55, raster_crs, always_xy=True)

        ctx = {
            "band": src.read(1),
            "inv_transform": ~src.transform,
            "height": src.height,
            "width": src.width,
            "labels": labels,
            "to_raster": to_raster,
            "raster_crs": raster_crs,
        }

    WORKER_RASTER_CTX_CACHE[class_tif] = ctx
    return ctx, class_tif, season_tif


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

    # Season raster context by t0
    ctx, class_tif_used, season_tif_used = get_raster_ctx_for_datetime(t0)

    df, total_mass = segment_class_mass_fast_ctx(seg, seg_min_x, seg_min_y, ctx)

    df2, _ = segment_class_mass_fast_ctx(seg_mask_local.astype(np.float64), seg_min_x, seg_min_y, ctx)
    df2 = df2.rename(columns={"mass": "mask_mass", "proportion": "mask_proportion"})

    df = df.merge(df2[["code", "name", "mask_mass", "mask_proportion"]], on=["code", "name"], how="outer")
    df[["mass", "proportion", "mask_mass", "mask_proportion"]] = df[
        ["mass", "proportion", "mask_mass", "mask_proportion"]
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
    df["season_tif"] = os.path.basename(season_tif_used)
    df["season_class_tif"] = os.path.basename(class_tif_used)

    return (i, seg, seg_mask_local, dx, dy, df, class_tif_used)


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    # --------- user config ----------
    file_path = "/home/mart/Code/eco/Gardiner- Habitat.Perception.Bettongs.csv"
    target_id = sys.argv[1]

    padding = 150
    T = 15
    K_CLASSES = 11

    tif_dir = "csv_out/tif_in"
    class_cache_dir = os.path.join(tif_dir, "_class_cache")

    out_folder = "csv_out/proper"
    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(class_cache_dir, exist_ok=True)

    out_csv_segments = f"{out_folder}/{target_id}_segment_pgreen_class_mass.csv"
    out_csv_home = f"{out_folder}/{target_id}_homerange_pgreen_class_mass.csv"
    out_homerange_plot = f"{out_folder}/{target_id}_homerange_pgreen_plot.png"
    # out_homerange_plot = None
    # --------------------------------

    bettongs = process_bettongs(file_path)
    bettong = bettongs[target_id][20:45]
    print(f"Selected bettong '{target_id}' with {len(bettong)} fixes")

    # Fit step models and build kernels
    m_1, m_2, m_3 = pure_grouped(bettongs, 15)
    kernel_1 = correlated_kernels_from_matrix(m_1, reso, reso, 1)
    kernel_2 = correlated_kernels_from_matrix(m_2, reso, reso, 1)
    kernel_3 = correlated_kernels_from_matrix(m_3, reso, reso, 1)
    kernels = [kernel_1, kernel_2, kernel_3]

    # Global UD grid in MGA55
    xs, ys = zip(*[(x, y) for x, y, *_ in bettong])
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    W = (max_x - min_x) + 2 * padding
    H = (max_y - min_y) + 2 * padding
    global_min_x = min_x - padding
    global_min_y = min_y - padding
    print(f"Global W: {W}, H: {H}")

    # Choose representative season for plotting/home-range: first fix time
    track_dt0 = bettong[0][2]
    plot_class_tif, plot_labels, plot_season_tif = ensure_class_tif_for_date(
        dt=track_dt0,
        tif_dir=tif_dir,
        cache_dir=class_cache_dir,
        k_classes=K_CLASSES,
        nodata_val=255,
    )

    with rasterio.open(plot_class_tif) as src:
        raster_crs = src.crs
        print("RASTER CRS:", raster_crs)
        print("RASTER BOUNDS:", src.bounds)

    # Track bbox for sanity
    print("TRACK MGA55 BBOX:", (min(xs), min(ys), max(xs), max(ys)))
    to_raster_plot = None
    if raster_crs is not None and str(raster_crs).upper() != CRS_MGA55:
        to_raster_plot = Transformer.from_crs(CRS_MGA55, raster_crs, always_xy=True)

    # Parent ctx for home-range (CRS-safe)
    with rasterio.open(plot_class_tif) as src:
        plot_ctx = {
            "band": src.read(1),
            "inv_transform": ~src.transform,
            "height": src.height,
            "width": src.width,
            "labels": plot_labels,
            "to_raster": to_raster_plot,
            "raster_crs": src.crs,
        }

    # Global accumulators (still in MGA55 grid coordinates)
    total_utilization = np.zeros((H, W), dtype=np.float64)
    home_range_mask = np.zeros((H, W), dtype=bool)

    # Worker globals
    G_BETTONG = bettong
    G_KERNELS = kernels
    G_PADDING = padding
    G_GLOBAL_MIN_X = global_min_x
    G_GLOBAL_MIN_Y = global_min_y
    G_T = T

    G_TIF_DIR = tif_dir
    G_CLASS_CACHE_DIR = class_cache_dir
    G_K_CLASSES = K_CLASSES

    n_segments = len(bettong) - 1
    max_workers = max(1, (os.cpu_count() or 2) - 1)
    ctx_mp = mp.get_context("fork")

    all_dfs = []
    print(f"Parallelizing {n_segments} segment(s) with max_workers={max_workers}...")

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx_mp) as ex:
        futures = [ex.submit(compute_one_segment_job, i, target_id) for i in range(n_segments)]
        for fut in as_completed(futures):
            res = fut.result()
            if res is None:
                continue

            i, seg, seg_mask_local, dx, dy, df, class_tif_used = res

            add_with_offset(total_utilization, seg, dx, dy)
            or_mask_with_offset(home_range_mask, seg_mask_local, dx, dy)

            all_dfs.append(df)
            print(f"Finished segment {i}                       ", end="\r")

    total_sum = float(np.sum(total_utilization))
    print(f"Total sum of total_utilization: {total_sum}")

    # Per-segment CSV
    if all_dfs:
        big = pd.concat(all_dfs, ignore_index=True)
        big.to_csv(out_csv_segments, index=False)
        print(f"Wrote {len(big)} rows to {out_csv_segments}")
    else:
        print("No valid segments produced any results; nothing written (segments).")

    # Home-range composition (uses first-fix season raster), CRS-safe
    df_home, _ = segment_class_mass_fast_ctx(
        home_range_mask.astype(np.float64),
        global_min_x,
        global_min_y,
        plot_ctx,
    )
    df_home.insert(0, "bettong_id", target_id)
    df_home["season_tif"] = os.path.basename(plot_season_tif)
    df_home["season_class_tif"] = os.path.basename(plot_class_tif)
    df_home.to_csv(out_csv_home, index=False)

    print("Home-range (union mask) class composition (using first-fix season raster):")
    print(df_home)
    print(f"Wrote home-range table to {out_csv_home}")

    # Plot (CRS-safe overlays)
    plot_class_with_trajectory_and_contour(
        bettong=bettong,
        raster_tif=plot_class_tif,
        utilization_distribution=total_utilization,
        home_range_mask=home_range_mask,
        global_min_x_mga55=global_min_x,
        global_min_y_mga55=global_min_y,
        to_raster=to_raster_plot,
        n_classes=K_CLASSES,
        out_path=out_homerange_plot,
        title_suffix=f"Persistent green (binned), season: {os.path.basename(plot_season_tif)}",
    )
