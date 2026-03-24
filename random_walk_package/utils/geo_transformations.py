import numpy as np
from pyproj import Transformer

from random_walk_package.data_sources.geo_fetcher import utm_to_lonlat


def utm_zone_from_lon(lon):
    return int((lon + 180) // 6) + 1

def make_segment_transformer(min_lon, min_lat, max_lon, max_lat):
    center_lon = 0.5 * (min_lon + max_lon)
    center_lat = 0.5 * (min_lat + max_lat)

    zone = utm_zone_from_lon(center_lon)
    hemi = "N" if center_lat >= 0 else "S"
    epsg = 32600 + zone if hemi == "N" else 32700 + zone

    fwd = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    inv = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)

    return fwd, inv, zone, hemi, epsg

def padded_utm_bbox(min_lon, min_lat, max_lon, max_lat, padding, max_cell_size):
    fwd, inv, zone, hemi, epsg = make_segment_transformer(min_lon, min_lat, max_lon, max_lat)
    # all bbox corners in same crs
    corners_lonlat = [
        (min_lon, min_lat),
        (min_lon, max_lat),
        (max_lon, min_lat),
        (max_lon, max_lat),
    ]
    corners_utm = [fwd.transform(lon, lat) for lon, lat in corners_lonlat]

    xs = [p[0] for p in corners_utm]
    ys = [p[1] for p in corners_utm]

    min_utm_x = min(xs)
    max_utm_x = max(xs)
    min_utm_y = min(ys)
    max_utm_y = max(ys)
    pad_x = max((max_utm_x - min_utm_x) * padding, max_cell_size)
    pad_y = max((max_utm_y - min_utm_y) * padding, max_cell_size)

    utm_bbox = (
        min_utm_x - pad_x,
        min_utm_y - pad_y,
        max_utm_x + pad_x,
        max_utm_y + pad_y,
    )
    return utm_bbox, zone, hemi, epsg, fwd, inv

def grid_to_geo_walk(walk_segment, utm_bbox, width, height, inv_transformer):
    min_x, min_y, max_x, max_y = utm_bbox
    result = []

    for x, y in walk_segment:
        if width <= 1 or height <= 1:
            result.append((np.nan, np.nan))
            continue
        utm_x = min_x + x / (width - 1) * (max_x - min_x)
        utm_y = max_y - y / (height - 1) * (max_y - min_y)
        lon, lat = inv_transformer.transform(utm_x, utm_y)
        result.append((lon, lat))

    return result

def utm_to_grid(nx, ny, xmin, ymin, xmax, ymax, utm_x, utm_y):
    x = np.round((utm_x - xmin) / (xmax - xmin) * (nx - 1)).astype(int)
    y = np.round((ymax - utm_y) / (ymax - ymin) * (ny - 1)).astype(int)
    return x, y


def grid_shape_from_bbox(bbox_utm, resolution):
    """Compute regular grid shape (width, height) from utm bounding box and resolution."""
    xmin, ymin, xmax, ymax = bbox_utm
    width_m = xmax - xmin
    height_m = ymax - ymin

    if width_m >= height_m:
        nx = resolution
        ny = max(1, int(resolution * height_m / width_m))
    else:
        ny = resolution
        nx = max(1, int(resolution * width_m / height_m))

    return nx, ny

def grid_to_geo(x, y, utm_bbox, width, height, epsg):
    min_x, min_y, max_x, max_y = utm_bbox

    utm_x = min_x + x / (width - 1) * (max_x - min_x)
    utm_y = max_y - y / (height - 1) * (max_y - min_y)

    lon, lat = utm_to_lonlat(utm_x, utm_y, epsg)
    return lon, lat