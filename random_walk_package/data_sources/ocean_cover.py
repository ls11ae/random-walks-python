import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

from random_walk_package.data_sources.land_cover_adapter import landcover_to_discrete_txt

GRID_RESOLUTION_LONGER_AXIS = 1000
OCEAN_COVER_SHAPE_PATH = "../resources/marine_cover/ne_10m_land.shp"
OCEAN_COVER_TIF_PATH = "../resources/marine_cover/ne_10m_ocean.tif"
OCEAN_COVER_TXT_PATH = "../resources/marine_cover/ne_10m_ocean.txt"

land = gpd.read_file(
    OCEAN_COVER_SHAPE_PATH
).to_crs("EPSG:4326")

# tiger shark bbox
minx, miny, maxx, maxy = -168.27, 16.97, -133.68, 26.46
# only hawaii bbox
# minx, miny, maxx, maxy = -161.0, 18.5, -154.5, 23.0

# calculate aspect ratio
aspect_ratio = (maxx - minx) / (maxy - miny)
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
        OCEAN_COVER_TIF_PATH,
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

g_x, g_y = 1000, int(1000 / aspect_ratio)
landcover_to_discrete_txt(OCEAN_COVER_TIF_PATH, g_x, g_y, minx, miny, maxx, maxy, output=OCEAN_COVER_TXT_PATH)
data = np.loadtxt(OCEAN_COVER_TXT_PATH, dtype=int)

plt.imshow(data, cmap="gray")
plt.colorbar()
plt.title("Ocean-Land mask from TXT with custom resolution")
plt.show()

# plot original mask
plt.imshow(mask, cmap="gray")
plt.title("Ocean-Land mask original resolution")
plt.show()
