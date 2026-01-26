#!/usr/bin/env python3
"""
Single-file, self-contained script that:

1) Reads bettong CSV (handles MD/DM datetime formats by ID sets)
2) Fits 3 GMMs (one per state) from step vectors (as in your existing logic)
3) Builds correlated kernels from those fitted matrices (via random_walk_package bindings)
4) Fetches and clips ESA WorldCover from Planetary Computer for the selected track
5) Prints WorldCover class for each fix
6) Computes per-segment utilization distributions on per-segment local grids
7) Pastes each segment UD into a global (track-level) UD array with correct offsets + clipping
8) Plots WorldCover background with UD contours and trajectory overlay

Requirements:
- random_walk_package must be importable
- pystac_client, planetary_computer, rioxarray, rasterio, pyproj installed

Run:
    python bettong_worldcover_ud_onefile.py
"""

import os
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

from sklearn.mixture import GaussianMixture

from pystac_client import Client
import planetary_computer
from scipy.ndimage import gaussian_filter
import rioxarray

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

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

# -------------------------
# WorldCover classes
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

# -------------------------
# CRS helpers
# -------------------------
CRS_MGA55 = "EPSG:28355"
CRS_WGS84 = "EPSG:4326"

to_wgs84 = Transformer.from_crs(CRS_MGA55, CRS_WGS84, always_xy=True)
to_mga55 = Transformer.from_crs(CRS_WGS84, CRS_MGA55, always_xy=True)

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
# Raster reprojection
# -------------------------
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

# -------------------------
# Fetch ESA WorldCover via Planetary Computer
# -------------------------
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
# WorldCover utilities
# -------------------------
def load_worldcover_raster_mga55(raster_tif_mga55):
    with rasterio.open(raster_tif_mga55) as src:
        arr = src.read(1)
        extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
    return arr, extent

def print_worldcover_per_step(bettong_traj, raster_tif_mga55):
    with rasterio.open(raster_tif_mga55) as src:
        band1 = src.read(1)
        nodata = src.nodata

        for i, (x, y, dt, state) in enumerate(bettong_traj, start=1):
            if not (src.bounds.left <= x <= src.bounds.right and src.bounds.bottom <= y <= src.bounds.top):
                print(f"{i:03d} | x={x}, y={y} | time={dt} | state={state} | OUTSIDE raster extent")
                continue

            row, col = src.index(x, y)
            if row < 0 or row >= band1.shape[0] or col < 0 or col >= band1.shape[1]:
                print(f"{i:03d} | x={x}, y={y} | time={dt} | state={state} | OUTSIDE raster array")
                continue

            code = int(band1[row, col])
            if nodata is not None and code == nodata:
                cls_name = "No Data"
            else:
                cls_name = WC_CLASSES.get(code, ("Unknown", None))[0]

            print(f"{i:03d} | x={x}, y={y} | time={dt} | state={state} | class={code} ({cls_name})")


def ud_isopleth_level(ud, p=0.95):
    ud = np.asarray(ud, dtype=float)
    ud = np.clip(ud, 0, None)

    total = ud.sum()
    if total <= 0:
        return np.nan  # nothing to contour

    # Normalize to probability mass per cell (safe even if already normalized)
    udn = ud / total

    flat = udn.ravel()
    idx = np.argsort(flat)[::-1]              # descending
    csum = np.cumsum(flat[idx])               # cumulative probability mass

    k = np.searchsorted(csum, p, side="left") # first index reaching p
    level = flat[idx[k]]
    return level

def plot_worldcover_with_trajectory_and_contour(bettong, raster_tif_mga55,
                                               utilization_distribution, home_range_mask,
                                               global_min_x, global_min_y):
    xs = [x for (x, y, *_rest) in bettong]
    ys = [y for (x, y, *_rest) in bettong]

    # background raster
    codes = sorted(WC_CLASSES.keys())
    boundaries = [codes[0] - 0.5] + [(a + b) / 2 for a, b in zip(codes[:-1], codes[1:])] + [codes[-1] + 0.5]
    cmap = ListedColormap([WC_CLASSES[c][1] for c in codes])
    norm = BoundaryNorm(boundaries, cmap.N)

    arr, extent = load_worldcover_raster_mga55(raster_tif_mga55)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(arr, extent=extent, origin="upper", cmap=cmap, norm=norm)
    
    

    # UD contour in UD coords (NOT raster extent)
    ud = np.asarray(utilization_distribution, dtype=float)
    H, W = ud.shape
    x = global_min_x + np.arange(W)
    y = global_min_y + np.arange(H)
    
    mask = home_range_mask.astype(float)   # shape (H, W), values 0/1

    H, W = mask.shape
    x = global_min_x + np.arange(W)
    y = global_min_y + np.arange(H)
    
    mask_f = gaussian_filter(home_range_mask.astype(float), sigma=1.0)
    # ax.contour(x, y, mask_f, levels=[0.5], colors="white", linewidths=2)
    ax.contour(x, y, mask_f, levels=[0.5], colors="purple", linewidths=3, alpha=0.9)
        
    
    level95 = ud_isopleth_level(ud, p=0.95)
    if np.isfinite(level95):
        ax.contour(x, y, ud / ud.sum(), levels=[level95],
               colors="orange", linewidths=2, alpha=0.9)

    mask = ud > 0
    if np.any(mask):
        vmin = ud[mask].min()
        vmax = ud.max()
        if vmax > 0 and vmin > 0:
            levels = np.logspace(np.log10(vmin), np.log10(vmax), 60)
            ax.contour(x, y, ud, levels=levels, colors="white", linewidths=1, alpha=0.9)

    ax.plot(xs, ys, linewidth=2, color="black")
    ax.scatter(xs, ys, s=12, color="black")

    ax.set_title("Trajectory over ESA WorldCover (MGA55) + UD contour")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    plt.tight_layout()
    plt.show()


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

        state = row["states"]
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
                # NOTE: your original logic uses i-2; keep it, but guard i>=2
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
# Utilization distribution computation
# -------------------------
def points_to_np_float64(points, n: int) -> np.ndarray:
    try:
        arr = np.fromiter(points, dtype=np.float64, count=n)
        if arr.size == n:
            return arr
    except TypeError:
        pass
    return np.fromiter((float(points[i]) for i in range(n)), dtype=np.float64, count=n)

def compute_segment_util(bettong, kernels, i: int, min_x: int, min_y: int, *, W: int, H: int, T: int):
    """
    Compute one segment UD on a LOCAL grid whose map origin is (min_x, min_y).

    Grid mapping:
        col = x - min_x
        row = y - min_y
    """
    x0, y0, t0, s0 = bettong[i]
    x1, y1, t1, s1 = bettong[i + 1]

    if (t1 - t0).total_seconds() / 60 != 15:
        print(f"Skipping segment {i} due to non-15 minute interval.")
        return None

    sx, sy = x0 - min_x, y0 - min_y
    tx, ty = x1 - min_x, y1 - min_y

    if not (0 <= sx < W and 0 <= sy < H and 0 <= tx < W and 0 <= ty < H):
        print(f"Skipping segment {i} because start/end outside local grid.")
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

# -------------------------
# Array placement helper
# -------------------------
def add_with_offset(total_arr, seg_arr, dx, dy):
    """
    Add seg_arr into total_arr at (dx, dy) top-left (col,row) with clipping.
    Arrays are indexed [row, col] == [y, x].
    """
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

def segment_worldcover_mass(seg, seg_min_x, seg_min_y, worldcover_tif, wc_classes=WC_CLASSES):
    """
    seg: 2D numpy array (H, W) utilization mass on a 1m grid
    seg_min_x, seg_min_y: map coords (EPSG:28355) of seg's top-left? (see below)
    worldcover_tif: path to WorldCover raster in EPSG:28355

    IMPORTANT: This assumes seg indices map as:
        x = seg_min_x + col
        y = seg_min_y + row
    i.e., row increases upward in map-y.
    If your y-axis is inverted, see the note after this function.
    """
    seg = np.asarray(seg, dtype=np.float64)
    H, W = seg.shape

    # Only process non-zero cells for speed
    rows, cols = np.nonzero(seg > 0)
    vals = seg[rows, cols]

    mass_by_code = {}

    with rasterio.open(worldcover_tif) as src:
        for r, c, v in zip(rows, cols, vals):
            x = seg_min_x + int(c)
            y = seg_min_y + int(r)

            # Skip if outside WorldCover extent
            if not (src.bounds.left <= x <= src.bounds.right and src.bounds.bottom <= y <= src.bounds.top):
                continue

            rr, cc = src.index(x, y)
            if rr < 0 or rr >= src.height or cc < 0 or cc >= src.width:
                continue

            code = int(src.read(1, window=((rr, rr+1), (cc, cc+1)))[0, 0])
            mass_by_code[code] = mass_by_code.get(code, 0.0) + float(v)

    total_mass = float(np.sum(seg))
    # Build a nice table
    rows_out = []
    for code, mass in sorted(mass_by_code.items(), key=lambda kv: kv[1], reverse=True):
        name = wc_classes.get(code, ("Unknown", None))[0]
        rows_out.append((code, name, mass, mass / total_mass if total_mass > 0 else np.nan))

    return pd.DataFrame(rows_out, columns=["wc_code", "wc_name", "mass", "proportion"]), total_mass

def ud_isopleth_mask(ud, p=0.95):
    """
    Returns:
      mask: boolean array where True cells form the p isopleth region
      level: the UD threshold used (on normalized UD)
    """
    ud = np.asarray(ud, dtype=float)
    ud = np.clip(ud, 0, None)
    s = ud.sum()
    if s <= 0:
        return np.zeros_like(ud, dtype=bool), np.nan

    udn = ud / s  # normalize to probability mass per cell
    flat = udn.ravel()
    idx = np.argsort(flat)[::-1]
    csum = np.cumsum(flat[idx])
    k = np.searchsorted(csum, p, side="left")
    level = flat[idx[k]]
    mask = (udn >= level)
    return mask, level


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    file_path = "/home/mart/Code/eco/Gardiner- Habitat.Perception.Bettongs.csv"
    bettongs = process_bettongs(file_path)

    # Select a short tomato chunk
    bettong = bettongs["tomato"][37:43]  # 3 points -> 2 segments

    # Global (track-level) grid definition
    xs, ys = zip(*[(x, y) for x, y, *_ in bettong])
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    padding = 100
    W = (max_x - min_x) + 2 * padding
    H = (max_y - min_y) + 2 * padding
    print(f"Global W: {W}, H: {H}")

    global_min_x = min_x - padding
    global_min_y = min_y - padding

    total_utilization = np.zeros((H, W), dtype=np.float64)

    # Fetch WorldCover for this track (for class sampling + plotting background)
    worldcover_mga55_tif = worldcover_trajectory_pipeline(bettong, pad_m=padding, year=2020)
    print_worldcover_per_step(bettong, worldcover_mga55_tif)

    # Fit step models and build kernels
    m_1, m_2, m_3 = pure_grouped(bettongs, 15)

    kernel_1 = correlated_kernels_from_matrix(m_1, reso, reso, 1)
    kernel_2 = correlated_kernels_from_matrix(m_2, reso, reso, 1)
    kernel_3 = correlated_kernels_from_matrix(m_3, reso, reso, 1)
    kernels = [kernel_1, kernel_2, kernel_3]

    # Segment UD parameters
    T = 15
    
    all_dfs = []

    home_range_mask = np.zeros((H, W), dtype=bool)

    # Compute and paste each segment UD into the global UD array
    for i in range(len(bettong) - 1):
        x0, y0, t0, s0 = bettong[i]
        x1, y1, t1, s1 = bettong[i + 1]

        # Segment-local grid definition (explicit padding)
        seg_min_x = min(x0, x1) - padding
        seg_min_y = min(y0, y1) - padding
        seg_max_x = max(x0, x1) + padding
        seg_max_y = max(y0, y1) + padding

        seg_W = seg_max_x - seg_min_x
        seg_H = seg_max_y - seg_min_y

        print(f"Segment {i}: seg_W={seg_W}, seg_H={seg_H}")

        seg = compute_segment_util(
            bettong,
            kernels,
            i,
            seg_min_x,
            seg_min_y,
            W=seg_W,
            H=seg_H,
            T=T
        )
        if seg is None:
            continue
        
        dx = seg_min_x - global_min_x
        dy = seg_min_y - global_min_y
        
        seg_home_range = np.zeros((H, W), dtype=np.float64)
        add_with_offset(seg_home_range, seg, dx, dy)
        seg_mask, _ = ud_isopleth_mask(seg_home_range, p=0.95)
        home_range_mask |= seg_mask

        add_with_offset(total_utilization, seg, dx, dy)
        
        
        df, total_mass = segment_worldcover_mass(seg, seg_min_x, seg_min_y, worldcover_mga55_tif)

        # Optional: drop No Data
        # df = df[df["wc_code"] != 0].copy()

        # ---- attach metadata ----
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
        
        print(df)

        all_dfs.append(df)


    total_sum = float(np.sum(total_utilization))
    print(f"Total sum of total_utilization: {total_sum}")

    out_csv = "segment_worldcover_mass_all_segments.csv"

    if all_dfs:
        big = pd.concat(all_dfs, ignore_index=True)
        big.to_csv(out_csv, index=False)
        print(f"Wrote {len(big)} rows to {out_csv}")
    else:
        print("No valid segments produced any results; nothing written.")

    plot_worldcover_with_trajectory_and_contour(bettong, worldcover_mga55_tif, total_utilization, home_range_mask, global_min_x, global_min_y)

    