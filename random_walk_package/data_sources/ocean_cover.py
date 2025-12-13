import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

from random_walk_package.data_sources.geo_fetcher import reproject_to_utm

def fetch_ocean_cover_tif(
    shapefile_path: str,
    bbox: tuple[float, float, float, float],
    output_tif_path: str,
    resolution_deg: float = 0.01,
    ocean_value: int = 0,
    land_value: int = 1,
) -> np.ndarray:
    """
    Standalone function to fetch ocean/land cover TIFF from shapefile.
    
    This is equivalent to fetch_landcover_data() but for ocean/land boundaries.
    
    Parameters
    ----------
    shapefile_path : str
        Path to land boundaries shapefile
    bbox : tuple of float
        Bounding box as (min_lon, min_lat, max_lon, max_lat)
    output_tif_path : str
        Path for output GeoTIFF file
    resolution_deg : float, optional
        Resolution in degrees (default: 0.01)
    ocean_value : int, optional
        Value for ocean pixels (default: 0)
    land_value : int, optional
        Value for land pixels (default: 1)
    
    Returns
    -------
    np.ndarray
        The rasterized ocean/land mask
    """
    print(f"\nFetching ocean cover data for BBOX: {bbox}...")
    min_lon, min_lat, max_lon, max_lat = bbox
    
    # Load and reproject shapefile
    land = gpd.read_file(shapefile_path).to_crs("EPSG:4326")
    print(f"Loaded {len(land)} land polygons")
    # Calculate raster dimensions
    width = int((max_lon - min_lon) / resolution_deg)
    height = int((max_lat - min_lat) / resolution_deg)
    aspect_ratio = (max_lon - min_lon) / (max_lat - min_lat)
    # Create geotransform
    transform = rasterio.transform.from_bounds(
        min_lon, min_lat, max_lon, max_lat, width, height
    )
    
    # Clip geometries to area of interest
    land_clipped = land.cx[min_lon:max_lon, min_lat:max_lat]
    print(f"Clipped to {len(land_clipped)} land polygons in bbox")
    if len(land_clipped) == 0:
            print("WARNING: No land geometries found in bbox - result will be all ocean!")
            
    # Rasterize
    mask = rasterize(
        [(geom, land_value) for geom in land_clipped.geometry],
        out_shape=(height, width),
        transform=transform,
        fill=ocean_value,
        dtype="uint8",
        all_touched=True
    )
    
    # Write GeoTIFF
    with rasterio.open(
        output_tif_path,
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
        
    print("Clipping to exact bbox...")
    import rioxarray    
    da = rioxarray.open_rasterio(output_tif_path)    
    
    clipped_xds = da.rio.clip_box(
            minx=bbox[0],
            miny=bbox[1],
            maxx=bbox[2],
            maxy=bbox[3],
            crs="EPSG:4326"  # Specify CRS of the bbox if not already aligned
        )

    
    clipped_xds.rio.to_raster(output_tif_path, compress='LZW', dtype='uint8')    

    print(f"Ocean cover data saved to {output_tif_path}")    
    print("Reprojecting to UTM zone...")
    reproject_to_utm(output_tif_path, output_tif_path)
    return output_tif_path    
    



'''
GRID_RESOLUTION_LONGER_AXIS = 1000
OCEAN_COVER_SHAPE_PATH = "../resources/marine_cover/ne_10m_land.shp"
OCEAN_COVER_TIF_PATH = "../resources/marine_cover/ne_10m_ocean.tif"
OCEAN_COVER_TXT_PATH = "../resources/marine_cover/ne_10m_ocean.txt"

OCEAN_VALUE = 0
LAND_VALUE = 1
OCEAN_VALUE_MAPPED = 80
LAND_VALUE_MAPPED = 10

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

g_x, g_y = GRID_RESOLUTION_LONGER_AXIS, int(GRID_RESOLUTION_LONGER_AXIS / aspect_ratio)
landcover_to_discrete_txt(OCEAN_COVER_TIF_PATH, g_x, g_y, minx, miny, maxx, maxy, output=OCEAN_COVER_TXT_PATH)

# dirty fix: remap 0/1 to out landcover classes 80/10. it would be cleaner to do that in landcover_to_discrete_txt, maybe with a 'is_marine:bool' flag
with open(OCEAN_COVER_TXT_PATH, 'r') as file:
    data = file.read()
    data = data.replace(str(OCEAN_VALUE), str(OCEAN_VALUE_MAPPED))
    data = data.replace(str(LAND_VALUE), str(LAND_VALUE_MAPPED))
with open(OCEAN_COVER_TXT_PATH, 'w') as file:
    file.write(data)

mask[mask == OCEAN_VALUE] = OCEAN_VALUE_MAPPED
mask[mask == LAND_VALUE] = LAND_VALUE_MAPPED

data = np.loadtxt(OCEAN_COVER_TXT_PATH, dtype=int)

plt.imshow(data, cmap="gray")
plt.colorbar()
plt.title("Ocean-Land mask from TXT with custom resolution")
plt.show()

# plot original mask
plt.imshow(mask, cmap="gray")
plt.title("Ocean-Land mask original resolution")
plt.show()
'''