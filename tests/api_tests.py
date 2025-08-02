from random_walk_package.core.MixedWalker import *
from random_walk_package.data_sources.geo_fetcher import *
from random_walk_package.data_sources.land_cover_adapter import landcover_to_discrete_txt
from random_walk_package.data_sources.movebank_adapter import *


def landcover_test():
    file = os.path.join(script_dir, 'resources',
                        'Baboon group movement, South Africa (data from Bonnell et al. 2016).csv')
    df = pd.read_csv(file)
    (min_lon, min_lat, max_lon, max_lat) = get_bounding_box(df)
    bb = (min_lon, min_lat, max_lon, max_lat)
    db = [a, b, res_x, res_y] = bbox_to_discrete_space((min_lon, min_lat, max_lon, max_lat), 200)
    print(db)
    print(map_lon_to_x(20.422875, min_lon, max_lon, res_x))
    print(map_lon_to_x(-34.467708, min_lat, max_lat, res_y))
    fetch_landcover_data(bb, "landcover_baboons.tif")
    landcover_to_discrete_txt("landcover_baboons.tif", res_x, res_y, min_lon, max_lat, max_lon, min_lat)


def temporal_data_test():
    file = os.path.join(script_dir, 'resources',
                        'Baboon group movement, South Africa (data from Bonnell et al. 2016).csv')
    df = pd.read_csv(file)
    AOI_BBOX = get_bounding_box(df)
    AOI_CENTER_LAT = (AOI_BBOX[1] + AOI_BBOX[3]) / 2
    AOI_CENTER_LON = (AOI_BBOX[0] + AOI_BBOX[2]) / 2

    OM_START_DATE, OM_END_DATE = get_start_end_dates(df)
    WEATHER_OUTPUT_FILE = "weather_baboons.csv"
    print(OM_START_DATE)
    print(OM_END_DATE)
    weather_file = fetch_weather_data(AOI_CENTER_LAT, AOI_CENTER_LON,
                                      OM_START_DATE, OM_END_DATE, WEATHER_OUTPUT_FILE)


def move_bank_test():
    file = os.path.join(script_dir, 'resources',
                        'Baboon group movement, South Africa (data from Bonnell et al. 2016).csv')
    df = pd.read_csv(file)
    ids = get_unique_animal_ids(df)
    print(ids)
    coords, timeline = get_animal_coordinates(df, ids[0], 100)
    fetch_weather_for_trajectory(coords, timestamps=timeline)
