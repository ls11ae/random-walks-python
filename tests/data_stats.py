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
def fetch_landcover_data_worldcover(bbox_wgs84, year=2020, output_filename="worldcover_clip_4326.tif"):
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


import numpy as np
import rioxarray
import stackstac
from pystac_client import Client
import planetary_computer

CRS_WGS84 = "EPSG:4326"

def fetch_ndvi_sentinel(bbox_wgs84, start_date, end_date,
                        max_cloud=20,
                        output_filename="ndvi_s2_clip_4326.tif"):

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

    items = list(search.get_items())
    if not items:
        raise RuntimeError("No Sentinel-2 items found for that AOI/date range.")

    # Stack red + nir bands (B04=red, B08=nir). Sentinel reflectance scale is 1e-4.
    stack = stackstac.stack(
        items,
        assets=["B04", "B08"],
        bounds_latlon=bbox_wgs84,
        epsg=4326,
        resolution=10,     # 10 m output in EPSG:4326 is approximate; OK for many uses.
        chunksize=2048,
    ).astype("float32") * 1e-4

    red = stack.sel(band="B04")
    nir = stack.sel(band="B08")

    # NDVI per scene, then take median over time for a composite
    ndvi = (nir - red) / (nir + red + 1e-6)
    ndvi_med = ndvi.median(dim="time", skipna=True)

    ndvi_med = ndvi_med.rio.write_crs(CRS_WGS84)
    ndvi_med.rio.to_raster(output_filename, compress="LZW", dtype="float32")

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
        ndvi = row.get("VegIndex")

        if bettong_id not in bettongs:
            bettongs[bettong_id] = []
        bettongs[bettong_id].append((int(row["x"]), int(row["y"]), date_time_obj, state, ndvi))

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

def fetch_ndvi_sentinel_mga55(
    bbox_wgs84, start_date, end_date,
    max_cloud=30,
    output_filename="ndvi_s2_clip_28355.tif"
):
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

    items = list(search.items())  # <- avoids get_items() warning
    
    if not items:
        raise RuntimeError("No Sentinel-2 items found for that AOI/date range.")

    stack = stackstac.stack(
        items,
        assets=["B04", "B08"],
        bounds_latlon=bbox_wgs84,
        epsg=28355,
        resolution=10,
        chunksize=2048,
        resampling=Resampling.nearest,   # NDVI = continuous
        # or Resampling.nearest for categorical
    ).astype("float32") * 1e-4

    red = stack.sel(band="B04")
    nir = stack.sel(band="B08")

    ndvi = (nir - red) / (nir + red + 1e-6)
    ndvi_med = ndvi.median(dim="time", skipna=True)

    # write CRS and save
    ndvi_med = ndvi_med.rio.write_crs(CRS_MGA55)
    ndvi_med.rio.to_raster(output_filename, compress="LZW", dtype="float32")

    return output_filename

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import matplotlib.pyplot as plt

CRS_MGA55 = "EPSG:28355"
CRS_WGS84 = "EPSG:4326"

# -------------------------
# Raster reprojection (UPDATED for NDVI)
# -------------------------
def reproject_raster(in_tif, out_tif, dst_crs=CRS_MGA55, resampling=Resampling.nearest):
    """
    Reproject a raster to dst_crs.
    For NDVI (continuous), use bilinear (default).
    For categorical rasters, use nearest.
    """
    with rasterio.open(in_tif) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height,
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
                    resampling=resampling
                )
    return out_tif


# -------------------------
# NDVI sampling at points (OPTIONAL but useful)
# # -------------------------
# def sample_ndvi_at_points_mga55(bettong, ndvi_tif_mga55, out_col="ndvi"):
#     """
#     bettong: list of tuples like (x, y, datetime_obj, state)
#     Returns a pandas DataFrame with NDVI values sampled at each (x,y).
#     """
#     import pandas as pd

#     rows = []
#     with rasterio.open(ndvi_tif_mga55) as src:
#         # rasterio.sample expects an iterable of (x, y)
#         coords = [(x, y) for (x, y, *_rest) in bettong]
#         samples = list(src.sample(coords))

#     for (x, y, dt, state), s in zip(bettong, samples):
#         v = float(s[0]) if s is not None and len(s) else np.nan
#         rows.append({"x": x, "y": y, "datetime": dt, "state": state, out_col: v})

#     df = pd.DataFrame(rows)

#     # Many NDVI pipelines can yield small out-of-range values due to clouds/nodata; clamp for sanity.
#     df[out_col] = df[out_col].clip(-1, 1)

#     return df


# -------------------------
# Load + plot NDVI raster (UPDATED)
# -------------------------
def load_ndvi_raster_mga55(raster_tif_mga55):
    with rasterio.open(raster_tif_mga55) as src:
        arr = src.read(1).astype("float32")
        extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
        nodata = src.nodata

    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)

    # NDVI should be [-1,1]; clamp for nicer plotting
    arr = np.clip(arr, -1, 1)
    return arr, extent


def plot_ndvi_with_trajectory(bettong, raster_tif_mga55, title="Trajectory over NDVI (MGA55)"):
    xs = [x for (x, y, *_rest) in bettong]
    ys = [y for (x, y, *_rest) in bettong]

    arr, extent = load_ndvi_raster_mga55(raster_tif_mga55)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(
        arr,
        extent=extent,
        origin="upper",
        vmin=-1,
        vmax=1,
        cmap="viridis"  # continuous; change if you want
    )

    ax.plot(xs, ys, linewidth=2, color="black", label="trajectory")
    ax.scatter(xs, ys, s=12, color="black")

    ax.set_title(title)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("NDVI")

    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


def ndvi_trajectory_pipeline(
    bettong,
    pad_m=100,
    start_date="2023-06-01",
    end_date="2023-06-31",
    out_tif="ndvi_clip_28355.tif",
):
    bbox_mga = compute_bbox_from_points_mga55(bettong, pad_m=pad_m)
    bbox_wgs = bbox_mga55_to_wgs84(*bbox_mga)

    ndvi_tif_mga55 = fetch_ndvi_sentinel_mga55(
        bbox_wgs, start_date=start_date, end_date=end_date,
        output_filename=out_tif
    )
    return ndvi_tif_mga55
# -------------------------
# NDVI sampling at points (OPTIONAL but useful)
# -------------------------
def sample_ndvi_at_points_mga55(bettong, ndvi_tif_mga55, out_col="ndvi"):
    """
    bettong: list of tuples like (x, y, datetime_obj, state)
    Returns a pandas DataFrame with NDVI values sampled at each (x,y).
    """
    import pandas as pd

    rows = []
    with rasterio.open(ndvi_tif_mga55) as src:
        # rasterio.sample expects an iterable of (x, y)
        coords = [(x, y) for (x, y, *_rest) in bettong]
        samples = list(src.sample(coords))

    for (x, y, dt, state, vegIndex), s in zip(bettong, samples):
        v = float(s[0]) if s is not None and len(s) else np.nan
        rows.append({"x": x, "y": y, "datetime": dt, "state": state, out_col: v})
        print(f"x: {x}, y: {y}, datetime: {dt}, state: {state}, vegIndex: {vegIndex}, sampled NDVI: {v}")

    df = pd.DataFrame(rows)

    # Many NDVI pipelines can yield small out-of-range values due to clouds/nodata; clamp for sanity.
    df[out_col] = df[out_col].clip(-1, 1)

    return df
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

def fetch_ndvi_sentinel_mga55(
    bbox_wgs84, start_date, end_date,
    max_cloud=30,
    output_filename="ndvi_s2_clip_28355.tif"
):
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

    items = list(search.items())  # <- avoids get_items() warning
    
    if not items:
        raise RuntimeError("No Sentinel-2 items found for that AOI/date range.")

    stack = stackstac.stack(
        items,
        assets=["B04", "B08"],
        bounds_latlon=bbox_wgs84,
        epsg=28355,
        resolution=10,
        chunksize=2048,
        resampling=Resampling.nearest,   # NDVI = continuous
        # or Resampling.nearest for categorical
    ).astype("float32") * 1e-4

    red = stack.sel(band="B04")
    nir = stack.sel(band="B08")

    ndvi = (nir - red) / (nir + red + 1e-6)
    ndvi_med = ndvi.median(dim="time", skipna=True)

    # write CRS and save
    ndvi_med = ndvi_med.rio.write_crs(CRS_MGA55)
    ndvi_med.rio.to_raster(output_filename, compress="LZW", dtype="float32")

    return output_filename

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import matplotlib.pyplot as plt

CRS_MGA55 = "EPSG:28355"
CRS_WGS84 = "EPSG:4326"

# -------------------------
# Raster reprojection (UPDATED for NDVI)
# -------------------------
def reproject_raster(in_tif, out_tif, dst_crs=CRS_MGA55, resampling=Resampling.nearest):
    """
    Reproject a raster to dst_crs.
    For NDVI (continuous), use bilinear (default).
    For categorical rasters, use nearest.
    """
    with rasterio.open(in_tif) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height,
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
                    resampling=resampling
                )
    return out_tif


# -------------------------
# NDVI sampling at points (OPTIONAL but useful)
# # -------------------------
# def sample_ndvi_at_points_mga55(bettong, ndvi_tif_mga55, out_col="ndvi"):
#     """
#     bettong: list of tuples like (x, y, datetime_obj, state)
#     Returns a pandas DataFrame with NDVI values sampled at each (x,y).
#     """
#     import pandas as pd

#     rows = []
#     with rasterio.open(ndvi_tif_mga55) as src:
#         # rasterio.sample expects an iterable of (x, y)
#         coords = [(x, y) for (x, y, *_rest) in bettong]
#         samples = list(src.sample(coords))

#     for (x, y, dt, state), s in zip(bettong, samples):
#         v = float(s[0]) if s is not None and len(s) else np.nan
#         rows.append({"x": x, "y": y, "datetime": dt, "state": state, out_col: v})

#     df = pd.DataFrame(rows)

#     # Many NDVI pipelines can yield small out-of-range values due to clouds/nodata; clamp for sanity.
#     df[out_col] = df[out_col].clip(-1, 1)

#     return df


# -------------------------
# Load + plot NDVI raster (UPDATED)
# -------------------------
def load_ndvi_raster_mga55(raster_tif_mga55):
    with rasterio.open(raster_tif_mga55) as src:
        arr = src.read(1).astype("float32")
        extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
        nodata = src.nodata

    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)

    # NDVI should be [-1,1]; clamp for nicer plotting
    arr = np.clip(arr, -1, 1)
    return arr, extent


def plot_ndvi_with_trajectory(bettong, raster_tif_mga55, title="Trajectory over NDVI (MGA55)"):
    xs = [x for (x, y, *_rest) in bettong]
    ys = [y for (x, y, *_rest) in bettong]

    arr, extent = load_ndvi_raster_mga55(raster_tif_mga55)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(
        arr,
        extent=extent,
        origin="upper",
        vmin=-1,
        vmax=1,
        cmap="viridis"  # continuous; change if you want
    )

    ax.plot(xs, ys, linewidth=2, color="black", label="trajectory")
    ax.scatter(xs, ys, s=12, color="black")

    ax.set_title(title)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("NDVI")

    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
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
        resampling=Resampling.nearest,   # important for SCL (categorical)
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



from sklearn.cluster import MiniBatchKMeans
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

def ndvi_trajectory_pipeline(
    bettong,
    pad_m=100,
    start_date="2023-06-01",
    end_date="2023-06-31",
    out_tif="ndvi_clip_28355.tif",
):
    bbox_mga = compute_bbox_from_points_mga55(bettong, pad_m=pad_m)
    bbox_wgs = bbox_mga55_to_wgs84(*bbox_mga)

    ndvi_tif_mga55 = fetch_ndvi_sentinel_mga55(
        bbox_wgs, start_date=start_date, end_date=end_date,
        output_filename=out_tif
    )
    return ndvi_tif_mga55

def sample_class_at_points_mga55(bettong, class_tif_mga55, out_col="ndvi_class"):
    """
    bettong: list of tuples (x, y, datetime_obj, state, vegIndex)
    class_tif_mga55: path to uint8 classified NDVI raster (e.g. temp.tif) in EPSG:28355
    Returns DataFrame with x,y,datetime,state,vegIndex,ndvi_class
    """
    rows = []
    with rasterio.open(class_tif_mga55) as src:
        coords = [(x, y) for (x, y, *_rest) in bettong]
        samples = list(src.sample(coords))
        nodata = src.nodata

    for (x, y, dt, state, vegIndex), s in zip(bettong, samples):
        cls = int(s[0]) if (s is not None and len(s)) else None
        if nodata is not None and cls == nodata:
            cls = None

        rows.append({
            "x": x,
            "y": y,
            "datetime": dt,
            "state": state,
            "VegIndex": vegIndex,
            out_col: cls,
        })

    df = pd.DataFrame(rows)
    # make VegIndex numeric (CSV sometimes stores it as str)
    df["VegIndex"] = pd.to_numeric(df["VegIndex"], errors="coerce")
    return df

def percent_same_vegindex_and_class(df, veg_col="VegIndex", class_col="ndvi_class"):
    """
    df must contain columns veg_col and class_col.
    Treats equality after converting both to integer codes (NaNs ignored).
    """
    d = df.copy()

    # coerce to numbers
    d[veg_col] = pd.to_numeric(d[veg_col], errors="coerce")
    d[class_col] = pd.to_numeric(d[class_col], errors="coerce")

    # drop missing
    d = d.dropna(subset=[veg_col, class_col])
    if len(d) == 0:
        return np.nan, 0, 0

    # compare as integer codes
    veg_int = d[veg_col].round().astype(int)     # change to .astype(int) if already int
    cls_int = d[class_col].astype(int)

    same = (veg_int == cls_int)
    pct = 100.0 * same.mean()
    return pct, int(same.sum()), int(len(same))

from scipy.stats import spearmanr


if __name__ == "__main__":
    file_path = "/home/mart/Code/eco/Gardiner- Habitat.Perception.Bettongs.csv"
    bettongs = process_bettongs(file_path)
    bettong = bettongs["lagartha"]

    padding = 150 

    xs, ys = zip(*[(x, y) for x, y, *_ in bettong])
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    W = (max_x - min_x) + 2 * padding
    H = (max_y - min_y) + 2 * padding
    print(f"W: {W}, H: {H}")
    start_date="2017-04-01"
    end_date="2017-08-30"
    
    K_CLASSES = 10

    # create figures bettong_tm
    # m_1, m_2, m_3 = par.pure_grouped(bettongs, 15)

    # ndvi_mga55_tif = ndvi_trajectory_pipeline(
    #     bettong,
    #     pad_m=padding,
    #     start_date=start_date,
    #     end_date=end_date
    # )
    
    
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
        class_tif_out="temp.tif",
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
    
    df_cmp = sample_class_at_points_mga55(bettong, "temp.tif", out_col="ndvi_class")
    # Example:
    
    d = df_cmp.dropna(subset=["VegIndex","ndvi_class"]).copy()
    d["VegIndex"] = pd.to_numeric(d["VegIndex"], errors="coerce")
    d = d.dropna(subset=["VegIndex"])
    veg_cls = d["VegIndex"].round().astype(int).to_numpy()   # only if VegIndex is class-like!
    ndv_cls = d["ndvi_class"].astype(int).to_numpy()

    best = None
    for s in range(-5, 6):
        acc = (veg_cls == (ndv_cls + s)).mean()
        if best is None or acc > best[0]:
            best = (acc, s)
    print("best exact-match after shift: accuracy=", best[0], "shift=", best[1])
    
    
    
    # build ndvi_center from your kmeans results
    center_map = {c: info["center"] for c, info in class_info.items()}
    df_cmp["ndvi_center"] = df_cmp["ndvi_class"].map(center_map)

    from scipy.stats import spearmanr
    d = df_cmp.dropna(subset=["VegIndex","ndvi_center"]).copy()
    d["VegIndex"] = pd.to_numeric(d["VegIndex"], errors="coerce")
    d = d.dropna(subset=["VegIndex"])

    rho, p = spearmanr(d["ndvi_center"], d["VegIndex"])
    print("Spearman rho (VegIndex vs ndvi_center) =", rho, "p =", p)

    
    # pct, n_same, n_total = percent_same_vegindex_and_class(df_cmp)
    # print(f"VegIndex == ndvi_class: {pct:.2f}% ({n_same}/{n_total})")
    
    # sample_ndvi_at_points_mga55(bettong, ndvi_mga55_tif)