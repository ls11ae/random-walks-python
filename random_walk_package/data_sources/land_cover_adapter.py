import rasterio

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


def landcover_to_discrete_txt(file_path, res_x, res_y, min_lon, max_lat, max_lon, min_lat, output="terrain.txt") -> str:
    try:
        with rasterio.open(file_path) as src:
            landcover_array = src.read(1)
            array_height, array_width = landcover_array.shape

            # Calculate raster indices for the bounding box coordinates
            row_start, col_start = src.index(min_lon, max_lat)
            row_stop, col_stop = src.index(max_lon, min_lat)

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
            return output
    except rasterio.RasterioIOError as e:
        print(f"Error opening the file: {e}")
