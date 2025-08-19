import planetary_computer
from pystac_client import Client
import rioxarray

# --- 1. Fetch Landcover Data (ESA WorldCover via Planetary Computer) ---
def fetch_landcover_data(bbox, output_filename="landcover_aoi.tif"):
    """
    Fetches ESA WorldCover landcover data for a given bounding box
    using Microsoft Planetary Computer's STAC API.
    Saves the result as a GeoTIFF.
    """
    print(f"\nFetching landcover data for BBOX: {bbox}...")
    try:
        catalog = Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )

        search = catalog.search(
            collections=["esa-worldcover"],
            bbox=bbox,
        )

        items = search.item_collection()
        if not items:
            print("No ESA WorldCover items found for the given AOI.")
            return None

        print(f"Found {len(items)} STAC items. Using the first one: {items[0].id}")

        # Get the href for the data asset (usually 'map')
        # Asset keys can vary; 'map' is common for ESA WorldCover
        asset_href = items[0].assets["map"].href

        # Open the raster data, clip to BBOX, and set CRS
        # ESA WorldCover is typically in EPSG:4326
        xds = rioxarray.open_rasterio(asset_href).rio.write_crs("EPSG:4326")

        # Clip to the exact bounding box
        # Ensure your bbox is in the same CRS as the raster (EPSG:4326 for WorldCover)
        # The `rio.clip_box` expects minx, miny, maxx, maxy
        clipped_xds = xds.rio.clip_box(
            minx=bbox[0],
            miny=bbox[1],
            maxx=bbox[2],
            maxy=bbox[3],
            crs="EPSG:4326"  # Specify CRS of the bbox if not already aligned
        )

        # Save the clipped raster
        clipped_xds.rio.to_raster(output_filename, compress='LZW', dtype='uint8')
        print(f"Landcover data saved to {output_filename}")

        # For further processing in Python, you can return the xarray.DataArray
        # print("Landcover data snippet (shape):", clipped_xds.shape)
        # print(clipped_xds)
        return output_filename

    except Exception as e:
        print(f"Error fetching landcover data: {e}")
        return None
