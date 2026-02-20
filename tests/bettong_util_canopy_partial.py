#!/usr/bin/env python3
"""
bettong_persistent_green_ud_onefile.py

Uses seasonal persistent-green-cover GeoTIFFs in csv_out/tif_in, chooses season by t0 date,
then (IMPORTANT) defines classes ONLY from pixels inside the AOI bounds:
    [global_min_x, global_min_y, global_min_x + W, global_min_y + H]  (EPSG:28355)

Key points:
- Bettong fixes + UD grids are in EPSG:28355 (MGA55)
- Seasonal rasters are EPSG:3577 (confirmed), so we reproject coordinates MGA55 -> raster CRS for lookup
- Raster is read only in the AOI window, and classes are derived only from valid pixels inside that window
- No NDVI logic is used anywhere

Outputs:
- csv_out/proper/<id>_segment_pgreen_class_mass.csv
- csv_out/proper/<id>_homerange_pgreen_class_mass.csv
- csv_out/proper/<id>_homerange_pgreen_plot.png
"""

# ---- make fork safer with BLAS/OpenMP (prevents SIGABRT) ----
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
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

import rasterio
from rasterio.windows import from_bounds
from rasterio.windows import transform as window_transform
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
# AOI window read + classing ONLY within bounds
# -------------------------
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


def build_ctx_for_season_window(
    season_tif: str,
    bbox_mga55: Tuple[float, float, float, float],  # (minx, miny, maxx, maxy) in EPSG:28355
    k_classes: int = 11,
    nodata_val: int = 255,
) -> Dict:
    """
    Read ONLY the raster window overlapping bbox_mga55 and compute class breaks ONLY from valid
    pixels in that window. Uses quantile bins (equal-count bins) within the window.

    Returns ctx:
      {
        "band": uint8 classified window (0..k-1, nodata=255),
        "inv_transform": inverse Affine of window transform (raster CRS),
        "height": int,
        "width": int,
        "labels": dict code->label (includes 255),
        "to_raster": Transformer MGA55->raster CRS (or None),
        "extent": (left,right,bottom,top) in raster CRS for plotting
      }
    """
    with rasterio.open(season_tif) as src:
        if src.crs is None:
            raise RuntimeError(f"Raster has no CRS: {season_tif}")

        to_raster = None
        if str(src.crs).upper() != CRS_MGA55:
            to_raster = Transformer.from_crs(CRS_MGA55, src.crs, always_xy=True)

        # Window bounds in raster CRS
        left, bottom, right, top = _bbox_mga55_to_raster_bounds(bbox_mga55, to_raster)

        win = from_bounds(left, bottom, right, top, transform=src.transform)
        win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))

        raw = src.read(1, window=win)
        win_tr = window_transform(win, src.transform)

        # Build extent in raster CRS for imshow
        h, w = raw.shape
        x0, y0 = win_tr * (0, 0)      # upper-left
        x1, y1 = win_tr * (w, h)      # lower-right (likely y decreases)
        extent = (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))

        src_nodata = src.nodata
        nd = nodata_val if src_nodata is None else int(src_nodata)

    valid = (raw != nd)
    vals = raw[valid].astype(np.float32)
    vals = np.clip(vals, 0.0, 100.0)

    if vals.size == 0:
        cls = np.full(raw.shape, nodata_val, dtype=np.uint8)
        labels = {nodata_val: "No Data"}
        for c in range(k_classes):
            labels[c] = f"Class {c}"
        return {
            "band": cls,
            "inv_transform": ~win_tr,
            "height": cls.shape[0],
            "width": cls.shape[1],
            "labels": labels,
            "to_raster": to_raster,
            "extent": extent,
        }

    # Quantile edges (k classes -> k+1 edges)
    qs = np.linspace(0.0, 1.0, k_classes + 1)
    edges = np.quantile(vals, qs)
    edges[0] = 0.0
    edges[-1] = 100.0

    # Ensure strictly increasing edges (handle repeated quantiles)
    for j in range(1, len(edges)):
        if edges[j] <= edges[j - 1]:
            edges[j] = edges[j - 1] + 1e-6

    bins = edges[1:-1]  # k-1 cutpoints
    cls = np.full(raw.shape, nodata_val, dtype=np.uint8)
    cls_vals = np.digitize(vals, bins, right=False).astype(np.uint8)  # 0..k-1
    cls[valid] = cls_vals

    labels = {nodata_val: "No Data"}
    for c in range(k_classes):
        lo = float(edges[c])
        hi = float(edges[c + 1])
        labels[c] = f"{lo:.1f}–{hi:.1f}% (AOI-quantile)"

    return {
        "band": cls,
        "inv_transform": ~win_tr,   # inverse of WINDOW transform
        "height": cls.shape[0],
        "width": cls.shape[1],
        "labels": labels,
        "to_raster": to_raster,     # MGA55->raster CRS before inv_transform
        "extent": extent,
    }


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
# FAST categorical-mass summary (CRS-safe, window-aware)
# -------------------------
def segment_class_mass_fast_ctx(seg, seg_min_x, seg_min_y, ctx):
    """
    seg is on a local grid with MGA55 coordinates:
      x = seg_min_x + col
      y = seg_min_y + row

    Raster ctx is a WINDOW in raster CRS, so:
      - transform (x,y) MGA55 -> raster CRS using ctx["to_raster"] (EPSG:3577)
      - apply ctx["inv_transform"] (inverse of WINDOW transform) to get window pixel indices
      - index ctx["band"] (classified window array)
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
# Plot helpers (plots only the AOI window)
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


def plot_class_with_trajectory_and_contour(
    bettong,
    ctx,
    utilization_distribution,
    home_range_mask,
    global_min_x_mga55,
    global_min_y_mga55,
    n_classes=11,
    tick_values=(2, 4, 6, 8, 10),
    out_path=None,
    title_suffix="Persistent green cover (AOI-quantile classes)",
):
    arr = ctx["band"]
    extent = ctx["extent"]
    nodata = 255
    to_raster = ctx.get("to_raster", None)

    arr_plot = arr.astype("float32")
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

    # UD + mask grid are defined in MGA55; build their coordinates and transform to raster CRS for contouring
    ud = np.asarray(utilization_distribution, dtype=float)
    H, W = ud.shape

    xs_mga = global_min_x_mga55 + np.arange(W, dtype=np.float64)
    ys_mga = global_min_y_mga55 + np.arange(H, dtype=np.float64)
    X_mga, Y_mga = np.meshgrid(xs_mga, ys_mga)

    if to_raster is not None:
        Xr, Yr = to_raster.transform(X_mga, Y_mga)
        bx, by = to_raster.transform(
            np.array([x for x, y, *_ in bettong], dtype=np.float64),
            np.array([y for x, y, *_ in bettong], dtype=np.float64),
        )
    else:
        Xr, Yr = X_mga, Y_mga
        bx = np.array([x for x, y, *_ in bettong], dtype=np.float64)
        by = np.array([y for x, y, *_ in bettong], dtype=np.float64)

    mask_f = gaussian_filter(home_range_mask.astype(float), sigma=1.0)
    ax.contour(Xr, Yr, mask_f, levels=[0.5], colors="purple", linewidths=3, alpha=0.9)
    ax.scatter(bx, by, s=12, color="black")

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    ax.set_title(f"Trajectory over classes + UD + Home-range contour\n{title_suffix}")
    ax.set_xlabel("X (raster CRS)")
    ax.set_ylabel("Y (raster CRS)")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=list(tick_values))
    cbar.set_label("Class (AOI-quantile persistent green %)")

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
G_K_CLASSES = None
G_BBOX_MGA55 = None

# Worker-local cache: season_tif -> ctx (classified window + transforms)
WORKER_CTX_CACHE: Dict[str, Dict] = {}


def get_ctx_for_datetime(dt: datetime):
    season_tif = season_tif_path_for_date(dt, G_TIF_DIR)
    if season_tif in WORKER_CTX_CACHE:
        return WORKER_CTX_CACHE[season_tif], season_tif

    ctx = build_ctx_for_season_window(
        season_tif=season_tif,
        bbox_mga55=G_BBOX_MGA55,
        k_classes=G_K_CLASSES,
        nodata_val=255,
    )
    WORKER_CTX_CACHE[season_tif] = ctx
    return ctx, season_tif


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

    # Choose season by entry date t0; classes are AOI-window-derived for that season
    ctx, season_tif_used = get_ctx_for_datetime(t0)

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

    return (i, seg, seg_mask_local, dx, dy, df, season_tif_used)


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
    out_folder = "csv_out/proper"
    os.makedirs(out_folder, exist_ok=True)

    out_csv_segments = f"{out_folder}/{target_id}_segment_pgreen_class_mass.csv"
    out_csv_home = f"{out_folder}/{target_id}_homerange_pgreen_class_mass.csv"
    out_homerange_plot = f"{out_folder}/{target_id}_homerange_pgreen_plot.png"
    # --------------------------------

    # Load tracks
    bettongs = process_bettongs(file_path)
    bettong = bettongs[target_id]
    print(f"Selected bettong '{target_id}' with {len(bettong)} fixes")

    # Fit step models and build kernels
    m_1, m_2, m_3 = pure_grouped(bettongs, 15)
    kernel_1 = correlated_kernels_from_matrix(m_1, reso, reso, 1)
    kernel_2 = correlated_kernels_from_matrix(m_2, reso, reso, 1)
    kernel_3 = correlated_kernels_from_matrix(m_3, reso, reso, 1)
    kernels = [kernel_1, kernel_2, kernel_3]

    # Global UD grid extent (MGA55)
    xs = [x for (x, y, *_rest) in bettong]
    ys = [y for (x, y, *_rest) in bettong]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    W = (max_x - min_x) + 2 * padding
    H = (max_y - min_y) + 2 * padding
    global_min_x = min_x - padding
    global_min_y = min_y - padding
    print(f"Global W: {W}, H: {H}")

    # AOI bounds for windowing/classing/plotting (exactly what you requested)
    bbox_mga55 = (
        float(global_min_x),
        float(global_min_y),
        float(global_min_x + W),
        float(global_min_y + H),
    )
    print("AOI MGA55 BBOX:", bbox_mga55)

    # Global accumulators (UD grid in MGA55 index-space)
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
    G_K_CLASSES = K_CLASSES
    G_BBOX_MGA55 = bbox_mga55

    n_segments = len(bettong) - 1
    max_workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"Parallelizing {n_segments} segment(s) with max_workers={max_workers}...")

    # Use fork, but BLAS threads are forced to 1 (top of file), preventing SIGABRT
    ctx_mp = mp.get_context("fork")

    all_dfs = []
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx_mp) as ex:
        futures = [ex.submit(compute_one_segment_job, i, target_id) for i in range(n_segments)]
        for fut in as_completed(futures):
            res = fut.result()
            if res is None:
                continue

            i, seg, seg_mask_local, dx, dy, df, season_tif_used = res

            add_with_offset(total_utilization, seg, dx, dy)
            or_mask_with_offset(home_range_mask, seg_mask_local, dx, dy)

            all_dfs.append(df)
            print(f"Finished segment {i}            ", end="\r")

    total_sum = float(np.sum(total_utilization))
    print(f"Total sum of total_utilization: {total_sum}")

    # Per-segment CSV
    if all_dfs:
        big = pd.concat(all_dfs, ignore_index=True)
        big.to_csv(out_csv_segments, index=False)
        print(f"Wrote {len(big)} rows to {out_csv_segments}")
    else:
        print("No valid segments produced any results; nothing written (segments).")

    # Home-range composition + plot: use first-fix season (classes derived from AOI window for that season)
    first_dt = bettong[0][2]
    first_season_tif = season_tif_path_for_date(first_dt, tif_dir)
    home_ctx = build_ctx_for_season_window(
        season_tif=first_season_tif,
        bbox_mga55=bbox_mga55,
        k_classes=K_CLASSES,
        nodata_val=255,
    )

    df_home, _ = segment_class_mass_fast_ctx(
        home_range_mask.astype(np.float64),
        global_min_x,
        global_min_y,
        home_ctx,
    )
    df_home.insert(0, "bettong_id", target_id)
    df_home["season_tif"] = os.path.basename(first_season_tif)
    df_home.to_csv(out_csv_home, index=False)

    print("Home-range (union mask) class composition (AOI-window-derived classes):")
    print(df_home)
    print(f"Wrote home-range table to {out_csv_home}")

    # Plot (cropped to AOI window)
    plot_class_with_trajectory_and_contour(
        bettong=bettong,
        ctx=home_ctx,
        utilization_distribution=total_utilization,
        home_range_mask=home_range_mask,
        global_min_x_mga55=global_min_x,
        global_min_y_mga55=global_min_y,
        n_classes=K_CLASSES,
        out_path=out_homerange_plot,
        title_suffix=f"Persistent green (AOI-quantile), season: {os.path.basename(first_season_tif)}",
    )
