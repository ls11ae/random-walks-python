import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from datetime import datetime
import matplotlib.patches as mpatches

from pystac_client import Client
import planetary_computer
import rioxarray

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.plot import show as rio_show

from . import bettong_test_parralel as par
# from .bettong_test_parralel import * # for processing CSV

from pyproj import Transformer

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
    """Convert a bbox from MGA55 meters to WGS84 lon/lat.
    Returns (minlon, minlat, maxlon, maxlat)."""
    corners = [(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]
    lons_lats = [to_wgs84.transform(x, y) for x, y in corners]
    lons, lats = zip(*lons_lats)
    return (min(lons), min(lats), max(lons), max(lats))


def compute_bbox_from_points_mga55(bettong, pad_m=100):
    xs = [x for (x, y, *_rest) in bettong]
    ys = [y for (x, y, *_rest) in bettong]
    return (min(xs) - pad_m, min(ys) - pad_m, max(xs) + pad_m, max(ys) + pad_m)


# -------------------------
# Raster reprojection
# -------------------------
def reproject_raster(in_tif, out_tif, dst_crs=CRS_MGA55):
    """Reproject a raster to dst_crs. Uses nearest resampling (categorical)."""
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
                    resampling=Resampling.nearest  # IMPORTANT for landcover classes
                )
    return out_tif


# -------------------------
# Fetch ESA WorldCover via Planetary Computer
# -------------------------
def fetch_landcover_data_worldcover(bbox_wgs84, year=2021, output_filename="worldcover_clip_4326.tif"):
    """
    Fetch ESA WorldCover "map" for a bbox in EPSG:4326 (lon/lat), clip, and save GeoTIFF.
    """
    print(f"Fetching ESA WorldCover for bbox (WGS84): {bbox_wgs84} and year {year}...")

    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    # Constrain to a year so you don't accidentally grab an arbitrary item.
    datetime = f"{year}-01-01/{year+1}-01-01"

    search = catalog.search(
        collections=["esa-worldcover"],
        bbox=list(bbox_wgs84),
        datetime=datetime,
    )

    items = search.item_collection()
    if not items:
        raise RuntimeError("No ESA WorldCover items found for the given AOI/year.")

    item = items[0]
    print(f"Found {len(items)} item(s). Using: {item.id}")

    asset_href = item.assets["map"].href  # classification map
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


# -------------------------
# Sampling
# -------------------------
# def sample_raster_at_points_mga55(points_df, raster_tif_mga55, x_col="x", y_col="y", out_col="worldcover"):
#     """Sample a 1-band raster at MGA55 point coordinates and add to DataFrame."""
#     coords = list(zip(points_df[x_col].astype(float), points_df[y_col].astype(float)))

#     with rasterio.open(raster_tif_mga55) as src:
#         vals = [int(v[0]) for v in src.sample(coords)]

#     points_df[out_col] = vals
#     return points_df

def load_worldcover_raster_mga55(raster_tif_mga55):
    with rasterio.open(raster_tif_mga55) as src:
        arr = src.read(1)
        extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
    
    return arr, extent

def plot_worldcover_with_trajectory(bettong, raster_tif_mga55):
    xs = [x for (x, y, *_rest) in bettong]
    ys = [y for (x, y, *_rest) in bettong]

    codes = sorted(WC_CLASSES.keys())
    colors = [WC_CLASSES[c][1] for c in codes]
    labels = [WC_CLASSES[c][0] for c in codes]

    # Discrete bins: put boundaries halfway between codes
    boundaries = [codes[0] - 0.5] + [(a + b) / 2 for a, b in zip(codes[:-1], codes[1:])] + [codes[-1] + 0.5]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, cmap.N)

    arr, extent = load_worldcover_raster_mga55(raster_tif_mga55)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(arr, extent=extent, origin="upper", cmap=cmap, norm=norm)

    ax.plot(xs, ys, linewidth=2, color="black", label="trajectory")
    ax.scatter(xs, ys, s=12, color="black")

    ax.set_title("Trajectory over ESA WorldCover (MGA55)")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    # Legend showing class colors
    patches = [mpatches.Patch(color=WC_CLASSES[c][1], label=f"{c}: {WC_CLASSES[c][0]}") for c in codes]
    ax.legend(handles=[*patches, mpatches.Patch(color="black", label="trajectory")],
              loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

    plt.tight_layout()
    plt.show()
    
    
def plot_worldcover_with_trajectory_and_contour(bettong, raster_tif_mga55, utilization_distribution):
    xs = [x for (x, y, *_rest) in bettong]
    ys = [y for (x, y, *_rest) in bettong]

    codes = sorted(WC_CLASSES.keys())
    colors = [WC_CLASSES[c][1] for c in codes]
    labels = [WC_CLASSES[c][0] for c in codes]

    # Discrete bins: put boundaries halfway between codes
    boundaries = [codes[0] - 0.5] + [(a + b) / 2 for a, b in zip(codes[:-1], codes[1:])] + [codes[-1] + 0.5]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, cmap.N)

    arr, extent = load_worldcover_raster_mga55(raster_tif_mga55)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(arr, extent=extent, origin="upper", cmap=cmap, norm=norm)
    
    vmin = utilization_distribution[utilization_distribution>0].min()
    vmax = utilization_distribution.max()
    levels = np.logspace(np.log10(vmin), np.log10(vmax), 100)
    ax.contour(utilization_distribution, extent=extent, levels=levels, colors='white', linewidths=1, alpha=0.7)

    ax.plot(xs, ys, linewidth=2, color="black", label="trajectory")
    ax.scatter(xs, ys, s=12, color="black")

    ax.set_title("Trajectory over ESA WorldCover (MGA55)")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    # Legend showing class colors
    patches = [mpatches.Patch(color=WC_CLASSES[c][1], label=f"{c}: {WC_CLASSES[c][0]}") for c in codes]
    ax.legend(handles=[*patches, mpatches.Patch(color="black", label="trajectory")],
              loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

    plt.tight_layout()
    plt.show()
    
def plot_worldcover_with_all_trajectories(bettongs_dict, raster_tif_mga55, label_last_point=True):
    # Discrete colormap for WorldCover classes
    codes = sorted(WC_CLASSES.keys())
    boundaries = [codes[0] - 0.5] + [(a + b) / 2 for a, b in zip(codes[:-1], codes[1:])] + [codes[-1] + 0.5]
    cmap = ListedColormap([WC_CLASSES[c][1] for c in codes])
    norm = BoundaryNorm(boundaries, cmap.N)

    with rasterio.open(raster_tif_mga55) as src:
        arr = src.read(1)
        extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(arr, extent=extent, origin="upper", cmap=cmap, norm=norm)

    # Plot each bettong trajectory
    for bettong_id, traj in bettongs_dict.items():
        if not traj:
            continue

        # Ensure sorted by time (you already sort in process_bettongs, but safe)
        traj = sorted(traj, key=lambda p: p[2])

        xs = [x for (x, y, *_rest) in traj]
        ys = [y for (x, y, *_rest) in traj]

        ax.plot(xs, ys, linewidth=1.5, label=bettong_id)
        ax.scatter(xs, ys, s=8)

        if label_last_point:
            ax.text(xs[-1], ys[-1], bettong_id, fontsize=8)

    ax.set_title("All bettong trajectories over ESA WorldCover (MGA55)")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    # Landcover legend (optional but useful)
    patches = [mpatches.Patch(color=WC_CLASSES[c][1], label=f"{c}: {WC_CLASSES[c][0]}") for c in codes]
    ax.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

    plt.tight_layout()
    plt.show()


# -------------------------
# One-call pipeline
# -------------------------
def worldcover_trajectory_pipeline(bettong, x_col="x", y_col="y",
                                  pad_m=100, year=2021,
                                  clip_4326="worldcover_clip_4326.tif",
                                  clip_28355="worldcover_clip_28355.tif"):
    """
    1) Compute bbox from MGA55 points
    2) Convert bbox to WGS84
    3) Fetch+clip WorldCover in WGS84
    4) Reproject raster to MGA55
    5) Sample class per point
    6) Return updated DataFrame and path to MGA55 raster
    """
    bbox_mga = compute_bbox_from_points_mga55(bettong, pad_m=pad_m)
    bbox_wgs = bbox_mga55_to_wgs84(*bbox_mga)

    tif_4326 = fetch_landcover_data_worldcover(bbox_wgs, year=year, output_filename=clip_4326)
    tif_28355 = reproject_raster(tif_4326, clip_28355, dst_crs=CRS_MGA55)

    # points_df = sample_raster_at_points_mga55(bettong, tif_28355, x_col=x_col, y_col=y_col, out_col="worldcover")
    return  tif_28355

# -------------------------------------------------------------------
# CSV processing
# -------------------------------------------------------------------
def process_bettongs(file_path: str):
    md = {
        "lagartha", "bjorn", "baldur", "sifa", "floki", "freya", "andive",
        "beetroot", "durian", "parsnip", "potato", "pumpkin", "raddish", "sprout",
        "swede", "turnip", "tomato"
    }
    dm = {"dot", "edwina", "egbert", "maud", "olga", "othello", "percy", "renet"}
    data = pd.read_csv(file_path)
    bettongs = {}

    for _, row in data.iterrows():
        bettong_id = row["ID"]

        if bettong_id in md:
            date_time_obj = datetime.strptime(row["datetime"], "%m/%d/%y %H:%M")
        elif bettong_id in dm:
            date_time_obj = datetime.strptime(row["datetime"], "%d/%m/%y %H:%M")
        else:
            raise ValueError(f"UNKNOWN ID: {bettong_id}")

        state = row["states"]

        if bettong_id not in bettongs:
            bettongs[bettong_id] = []
        bettongs[bettong_id].append((int(row["x"]), int(row["y"]), date_time_obj, state))

    for bettong_id in bettongs:
        bettongs[bettong_id].sort(key=lambda entry: entry[2])

    return bettongs


def print_worldcover_per_step(bettong_traj, raster_tif_mga55):
    """
    bettong_traj: list of tuples (x, y, datetime_obj, state)
    raster_tif_mga55: path to WorldCover raster in EPSG:28355
    """
    import rasterio

    with rasterio.open(raster_tif_mga55) as src:
        band1 = src.read(1)
        nodata = src.nodata

        for i, (x, y, dt, state) in enumerate(bettong_traj, start=1):
            # Check bounds first (avoids index errors)
            if not (src.bounds.left <= x <= src.bounds.right and src.bounds.bottom <= y <= src.bounds.top):
                print(f"{i:03d} | x={x}, y={y} | time={dt} | state={state} | OUTSIDE raster extent")
                continue

            # Convert map coords -> pixel row/col
            row, col = src.index(x, y)

            # Safety check (should be redundant if bounds check passed, but keeps it bulletproof)
            if row < 0 or row >= band1.shape[0] or col < 0 or col >= band1.shape[1]:
                print(f"{i:03d} | x={x}, y={y} | time={dt} | state={state} | OUTSIDE raster array")
                continue

            code = int(band1[row, col])

            # Handle nodata / unknown codes
            if nodata is not None and code == nodata:
                cls_name = "No Data"
            else:
                cls_name = WC_CLASSES.get(code, ("Unknown", None))[0]

            print(f"{i:03d} | x={x}, y={y} | time={dt} | state={state} | class={code} ({cls_name})")

# -------------------------
# Example usage
# -------------------------

if __name__ == "__main__":
    file_path = "/home/mart/Code/eco/Gardiner- Habitat.Perception.Bettongs.csv"
    bettongs = process_bettongs(file_path)
    bettong = bettongs["tomato"][35:45]

    padding = 100 
    worldcover_mga55_tif = worldcover_trajectory_pipeline(bettong, pad_m=padding, year=2020)

    print_worldcover_per_step(bettong, worldcover_mga55_tif)

    



    xs, ys = zip(*[(x, y) for x, y, *_ in bettong])
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    W = (max_x - min_x) + 2 * padding
    H = (max_y - min_y) + 2 * padding
    print(f"W: {W}, H: {H}")

    # Fit the step models and create kernels
    m_1, m_2, m_3 = par.pure_grouped(bettongs, 15)

    kernel_1 = par.correlated_kernels_from_matrix(m_1, par.reso, par.reso, 1)
    kernel_2 = par.correlated_kernels_from_matrix(m_2, par.reso, par.reso, 1)
    kernel_3 = par.correlated_kernels_from_matrix(m_3, par.reso, par.reso, 1)
    kernels = [kernel_1, kernel_2, kernel_3]

    # Parallel parameters
    T = 15
    total = len(bettong) - 1


    # Set worker globals (inherited by forked processes; avoids pickling kernels)
    par.G_BETTONG = bettong
    par.G_KERNELS = kernels
    par.G_W, par.G_H, par.G_T = W, H, T
    par.G_MIN_X, par.G_MIN_Y, par.G_PADDING = min_x, min_y, padding
    
    total_utilization = np.zeros((H, W), dtype=np.float64)

    # Use explicit fork context
    ctx = par.mp.get_context("fork")
    max_workers = min(par.os.cpu_count() or 1, total)

    used = 0
    done = 0

    with par.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        futures = [ex.submit(par.compute_segment_util, i) for i in range(total)]

        for fut in par.as_completed(futures):
            done += 1
            seg = fut.result()
            if seg is None:
                # print(f"  Skipped segment {done} due to invalid interval.")
                continue
            used += 1
            total_utilization += seg
            # print(f"  Finished {done}/{total} (accepted {used})")

    # Match your previous behavior: divide by total segments (even if some were skipped)
    # If you want to average only valid segments, replace `total` with `max(used, 1)`.
    total_utilization /= max(used, 1)
    
    total_sum = float(np.sum(total_utilization))
    print(f"Total sum of total_utilization: {total_sum}")

    # plot_single_utilisation_matrix(total_utilization, 1, W, H)
    
    plot_worldcover_with_trajectory_and_contour(bettong, worldcover_mga55_tif, total_utilization)
