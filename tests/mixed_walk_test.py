from memory_profiler import profile

from random_walk_package.core.MixedTimeWalker import MixedTimeWalker
from random_walk_package.core.MixedWalker import *


@profile
def mixed_walk():
    T = 100

    study = "baboon_SA_study/"
    walker = MixedWalker(T=T, resolution=200, study_folder=study)
    walker.generate_walk(serialized=True, steps=[(66, 66), (150, 100)])


def test_mixed_walk_time():
    animal_processor = AnimalMovementProcessor(
        'Baboon group movement, South Africa (data from Bonnell et al. 2016).csv')

    resolution = 200

    landcover = animal_processor.create_landcover_data(resolution, "landcover_baboons.tif")

    gridded_data: Point2DArrayGridPtr = animal_processor.fetch_gridded_weather_data(  # type: ignore
        output_filename="my_gridded_weather.csv",
        days_to_fetch=5,
        grid_points_per_edge=5,
        num_entries=60
    )

    kernels4d = tensor_map_terrain_bias_grid(landcover, gridded_data)

    walker = MixedWalker(
        T=60,
        width=resolution,
        spatial_map=landcover,
        movebank_study='Baboon group movement, South Africa (data from Bonnell et al. 2016).csv'
    )

    walker.set_kernels(terrain_only=False)

    print("Generating time-aware walk...")
    walk_np = walker.mixed_walk_time(kernels4d)

    plot_combined_terrain(
        landcover,
        walk_np,
        landcover.contents.width,
        landcover.contents.height,
        title="Time-Aware Mixed Walk"
    )

    return walk_np


# @profile
def test_time_walk():
    terrain = parse_terrain("time_walk_data/terrain_movebank.txt", " ")
    T = 100

    start = (100, 120)
    mid = (150, 120)
    end = (60, 20)

    steps = [start, mid, end]

    walk_points = time_walk_geo_multi(
        T=T,
        csv_path="time_walk_data/my_gridded_weather_grid_csvs",
        terrain_path="time_walk_data/terrain_movebank.txt",
        grid_x=5,
        grid_y=5,
        steps=steps
    )
    walknp = get_walk_points(walk_points)
    plot_combined_terrain(terrain, walk_points=walknp, terrain_height=terrain.height, terrain_width=terrain.width,
                          title="Time-Aware Mixed Walk")


@profile
def test_time_walker():
    start = (50, 70)
    end = (10, 10)

    walker = MixedTimeWalker(
        T=50,
        resolution=100,
        duration_in_days=3,
        study_folder="elephant_study/"
    )
    walker.preprocess()
    walker.generate_walk(start=start, end=end, output_file="time_walk3.json")


def test_time_walker_multi():
    start = (100, 120)
    mid = (60, 20)
    end = (150, 120)

    steps = [start, mid, end]

    walker = MixedTimeWalker(
        T=50,
        resolution=200,
        duration_in_days=5,
        study_folder="baboon_SA_study/"
    )
    walker.preprocess()
    walker.generate_walk_multi(steps=steps, output_file="time_walk3.json", serialized=True)
