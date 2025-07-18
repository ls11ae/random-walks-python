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


def landcover_to_discrete_txt(file_path, res_x, res_y, min_lon, max_lat, max_lon, min_lat, output="terrain.txt"):
    try:
        with rasterio.open(file_path) as src:
            landcover_array = src.read(1)

            # Calculate raster indices for the bounding box coordinates
            row_start, col_start = src.index(min_lon, max_lat)
            row_stop, col_stop = src.index(max_lon, min_lat)

            # Ensure start <= stop for rows and columns
            if row_start > row_stop:
                row_start, row_stop = row_stop, row_start
            if col_start > col_stop:
                col_start, col_stop = col_stop, col_start

            # Calculate the number of rows and columns in the ROI
            roi_rows = row_stop - row_start
            roi_cols = col_stop - col_start

            # Avoid division by zero when resolution is 1
            step_y = roi_rows / (res_y - 1) if res_y > 1 else 0
            step_x = roi_cols / (res_x - 1) if res_x > 1 else 0

            # Open the output file for writing
            with open(output, 'w') as f:
                for y_idx in range(res_y):
                    # Calculate row index in the raster, clamped to the ROI
                    r = row_start + int(y_idx * step_y)
                    r = max(row_start, min(r, row_stop))

                    row_values = []
                    for x_idx in range(res_x):
                        # Calculate column index in the raster, clamped to the ROI
                        c = col_start + int(x_idx * step_x)
                        c = max(col_start, min(c, col_stop))

                        pixel_value = landcover_array[r, c]
                        row_values.append(str(pixel_value))

                    # Write the row as a space-separated string
                    f.write(' '.join(row_values) + '\n')

            print(f"Landcover grid written to {output}")
            return output
    except rasterio.RasterioIOError as e:
        print(f"Error opening the file: {e}")