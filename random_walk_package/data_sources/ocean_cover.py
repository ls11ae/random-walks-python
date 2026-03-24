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
    width = max(1, int(np.ceil((max_lon - min_lon) / resolution_deg)))
    height = max(1, int(np.ceil((max_lat - min_lat) / resolution_deg)))
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
    reproject_to_utm(output_tif_path, output_tif_path) # i think the issue happens somewhwere here
    return output_tif_path    
    
