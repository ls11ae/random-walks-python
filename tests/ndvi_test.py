
import numpy as np
import rioxarray
import stackstac
from pystac_client import Client
import planetary_computer

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


from pyproj import Transformer



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
        resampling=Resampling.bilinear,   # NDVI = continuous
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
def reproject_raster(in_tif, out_tif, dst_crs=CRS_MGA55, resampling=Resampling.bilinear):
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

if __name__ == "__main__":
    file_path = "/home/mart/Code/eco/Gardiner- Habitat.Perception.Bettongs.csv"
    bettongs = process_bettongs(file_path)
    bettong = bettongs["tomato"][150:450]

    padding = 100 
    ndvi_mga55_tif = ndvi_trajectory_pipeline(
        bettong,
        pad_m=padding,
        start_date="2017-06-01",
        end_date="2017-06-30"
    )

    plot_ndvi_with_trajectory(bettong, ndvi_mga55_tif)
    # Optional: attach NDVI values to each point
    # df = sample_ndvi_at_points_mga55(bettong, ndvi_mga55_tif)
    # print(df.head())
