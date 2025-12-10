import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

land = gpd.read_file(
    "../resources/marine_cover/ne_10m_land.shp"
).to_crs("EPSG:4326")

# tiger shark bbox
minx, miny, maxx, maxy = -168.27, 16.97, -133.68, 26.46
# only hawaii bbox
# minx, miny, maxx, maxy = -161.0, 18.5, -154.5, 23.0
res = 0.01  # resolution in degrees

width = int((maxx - minx) / res)
height = int((maxy - miny) / res)

transform = rasterio.transform.from_bounds(
    minx, miny, maxx, maxy, width, height
)

# Clip geometries to area of interest
land = land.cx[minx:maxx, miny:maxy]

mask = rasterize(
    [(geom, 1) for geom in land.geometry],
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype="uint8",
    all_touched=True
)

with rasterio.open(
        "land_mask.tif",
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
        compress="LZW",
) as dst:
    dst.write(mask, 1)

print(mask, "land pixels")
print("non zero pixels" + str(np.count_nonzero(mask)))
import matplotlib.pyplot as plt

plt.imshow(mask, cmap="gray")
plt.title("Land mask")
plt.show()
