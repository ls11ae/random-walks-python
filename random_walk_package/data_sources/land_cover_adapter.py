import os

import rasterio

from random_walk_package import TREE_COVER
from random_walk_package.bindings import MOSS_AND_LICHEN
from random_walk_package.data_sources.geo_fetcher import lonlat_bbox_to_utm, utm_bbox_to_lonlat

landcover_classes = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen"
}


def landcover_to_discrete_txt(file_path, res_x, res_y, min_lon, min_lat, max_lon, max_lat, output="terrain.txt") -> \
        tuple[int, tuple[float, float, float, float]] | None:
    if os.path.exists(output):
        print(f"Output file {output} already exists, overwriting...")
    try:
        # BBox in Lon/Lat
        bbox_lonlat = (min_lon, min_lat, max_lon, max_lat)

        with rasterio.open(file_path) as src:
            crs_epsg = src.crs.to_epsg()
            if crs_epsg is None:
                raise ValueError("Raster CRS has no valid EPSG code")

            # Transform BBox to CRS of grid (eg. UTM)
            min_x, min_y, max_x, max_y = lonlat_bbox_to_utm(
                *bbox_lonlat, crs_epsg
            )

            print(f"BBox transformed to UTM: {min_x}, {min_y}, {max_x}, {max_y}")

            landcover_array = src.read(1)
            array_height, array_width = landcover_array.shape

            # Calculate raster indices for the bounding box coordinates
            row_start, col_start = src.index(min_x, max_y)
            row_stop, col_stop = src.index(max_x, min_y)

            # Ensure start <= stop for rows and columns
            if row_start > row_stop:
                row_start, row_stop = row_stop, row_start
            if col_start > col_stop:
                col_start, col_stop = col_stop, col_start

            # Clamp the indices to valid ranges (handle bbox partially or fully outside raster)
            row_start = max(0, min(row_start, array_height - 1))
            row_stop = max(0, min(row_stop, array_height - 1))
            col_start = max(0, min(col_start, array_width - 1))
            col_stop = max(0, min(col_stop, array_width - 1))

            # Calculate the number of rows and columns in the ROI
            roi_rows = row_stop - row_start
            roi_cols = col_stop - col_start

            # If there is no overlap between bbox and raster, fail fast with a clear message
            if roi_rows < 0 or roi_cols < 0 or (row_start == row_stop and col_start == col_stop):
                raise ValueError(
                    "Requested bounding box does not overlap the landcover raster. "
                    f"Raster bounds (lon, lat): {src.bounds}. "
                    f"Requested bbox: ({min_lon}, {min_lat}, {max_lon}, {max_lat})."
                )

            # Avoid division by zero when resolution is 1
            step_y = roi_rows / (res_y - 1) if res_y > 1 else 0
            step_x = roi_cols / (res_x - 1) if res_x > 1 else 0

            # Open the output file for writing
            with open(output, 'w') as f:
                for y_idx in range(res_y):
                    # Calculate row index in the raster, clamped to the ROI and array bounds
                    r = row_start + int(y_idx * step_y)
                    r = max(row_start, min(r, row_stop))
                    r = min(r, array_height - 1)

                    row_values = []
                    for x_idx in range(res_x):
                        # Calculate column index in the raster, clamped to the ROI and array bounds
                        c = col_start + int(x_idx * step_x)
                        c = max(col_start, min(c, col_stop))
                        c = min(c, array_width - 1)

                        pixel_value = landcover_array[r, c]
                        row_values.append(str(pixel_value))

                    # Write the row as a space-separated string
                    f.write(' '.join(row_values) + '\n')

            print(f"Landcover grid written to {output}")
            print(f"UTM Bounds: {min_x}, {min_y}, {max_x}, {max_y}")  # Debug output
            # Innerhalb von landcover_to_discrete_txt, direkt nach dem Schreiben der TXT-Datei:
            lon_min, lat_min, lon_max, lat_max = utm_bbox_to_lonlat(min_x, min_y, max_x, max_y, crs_epsg)
            print(f"Debug: UTM Bounds zurück transformiert in Lon/Lat: ({lon_min}, {lat_min}, {lon_max}, {lat_max})")

            return crs_epsg, (min_x, min_y, max_x, max_y)
    except rasterio.RasterioIOError as e:
        print(f"Error opening the file: {e}")
