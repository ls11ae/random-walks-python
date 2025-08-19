from random_walk_package.core.MixedWalker import *
from random_walk_package.data_sources.geo_fetcher import *
from random_walk_package.data_sources.land_cover_adapter import landcover_to_discrete_txt
from random_walk_package.data_sources.movebank_adapter import *


def landcover_test():
    file = os.path.join(script_dir, 'resources',
                        'Baboon group movement, South Africa (data from Bonnell et al. 2016).csv')
    df = pd.read_csv(file)
    (_min_lon, _min_lat, _max_lon, _max_lat) = get_bounding_box(df)
    bb = (_min_lon, _min_lat, _max_lon, _max_lat)
    db = [_, _, res_x, res_y] = bbox_to_discrete_space((_min_lon, _min_lat, _max_lon, _max_lat), 200)
    print(db)
    print(map_lon_to_x(20.422875, _min_lon, _max_lon, res_x))
    print(map_lon_to_x(-34.467708, _min_lat, _max_lat, res_y))
    fetch_landcover_data(bb, "landcover_baboons.tif")
    landcover_to_discrete_txt("landcover_baboons.tif", res_x, res_y, _min_lon, _max_lat, _max_lon, _min_lat)


def temporal_data_test():
    file = os.path.join(script_dir, 'resources',
                        'Baboon group movement, South Africa (data from Bonnell et al. 2016).csv')
    df = pd.read_csv(file)
    AOI_BBOX_ = get_bounding_box(df)
    AOI_CENTER_LAT_ = (AOI_BBOX_[1] + AOI_BBOX_[3]) / 2
    AOI_CENTER_LON_ = (AOI_BBOX_[0] + AOI_BBOX_[2]) / 2

    OM_START_DATE_, OM_END_DATE_ = get_start_end_dates(df)
    WEATHER_OUTPUT_FILE_ = "weather_baboons.csv"
    print(OM_START_DATE_)
    print(OM_END_DATE_)
    fetch_weather_data(AOI_CENTER_LAT_, AOI_CENTER_LON_,
                                      OM_START_DATE_, OM_END_DATE_, WEATHER_OUTPUT_FILE_)


def move_bank_test():
    file = os.path.join(script_dir, 'resources',
                        'Baboon group movement, South Africa (data from Bonnell et al. 2016).csv')
    df = pd.read_csv(file)
    ids = get_unique_animal_ids(df)
    print(ids)
    coords, timeline = get_animal_coordinates(df, ids[0], 100)
    fetch_weather_for_trajectory(coords, timestamps=timeline)
