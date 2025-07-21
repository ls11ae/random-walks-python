from memory_profiler import profile

from random_walk_package.core.AnimalMovement import *
from random_walk_package.core.BiasedWalker import BiasedWalker
from random_walk_package.core.BrownianWalker import *
from random_walk_package.core.CorrelatedWalker import *
from random_walk_package.core.MixedTimeWalker import MixedTimeWalker
from random_walk_package.core.MixedWalker import *
from random_walk_package.data_sources.geo_fetcher import *
from random_walk_package.data_sources.land_cover_adapter import landcover_to_discrete_txt
from random_walk_package.data_sources.movebank_adapter import *


# debugging: gdb --args python -m tests.test


def test_terrain():
    file = "landcover_142.txt"
    terrain = get_terrain_map(file, " ")

    num_steps = 200
    step_size = 7
    brownian_walker = BrownianWalker.from_terrain_map(terrain=terrain, T=num_steps, S=step_size, sigma=0.5, start_x=200,
                                                      start_y=200)
    walk = brownian_walker.generate_walk(375, 375)
    plot_walk_terrain(terrain=terrain, walk_points=walk, terrain_height=brownian_walker.H,
                      terrain_width=brownian_walker.W)


# @profile
def test_terrain_correlated():
    file = "landcover_142.txt"
    terrain = parse_terrain(file, " ")

    T = 150
    S = 7
    D = 8
    walker = CorrelatedWalker(D=D, S=S, W=401, H=401, T=T)
    walker.generate_kernel()
    walker.generate_from_terrain(terrain=terrain, start_x=200, start_y=200)
    walk = walker.backtrace(end_x=355, end_y=385, initial_direction=0)
    plot_walk_terrain(terrain=terrain, walk_points=walk, terrain_height=walker.H, terrain_width=walker.W)


def benchmark_brownian(num_steps=100):
    brownian_walker = BrownianWalker.from_parameters(T=num_steps, S=7, W=2 * num_steps + 1, H=2 * num_steps + 1)
    start = time.perf_counter()
    brownian_walker.generate_walk(175, 175)
    end = time.perf_counter()
    print(f"Generated walk in {end - start:.4f} seconds")


def test_brownian_wrapper():
    # Test 1: Initialize with parameters
    num_steps = 100
    step_size = 8
    brownian_walker = BrownianWalker.from_parameters(T=num_steps, S=step_size, sigma=0.5, scale=2)

    kernel_np = matrix_to_numpy(brownian_walker.kernel)
    print(kernel_np)

    # Test 2: Initialize with existing kernel/tensor
    brownian_walker2 = BrownianWalker.from_kernel_and_tensor(
        kernel=brownian_walker.kernel,
        dp_tensor=brownian_walker.dp_tensor,
        T=num_steps,
        W=brownian_walker.W,
        H=brownian_walker.H
    )

    # Test 3: Save/load functionality
    brownian_walker.save_tensor("test_tensor.bin")
    brownian_walker3 = BrownianWalker.load_tensor(
        filename="test_tensor.bin",
        kernel=brownian_walker.kernel,
        T=num_steps,
        W=brownian_walker.W,
        H=brownian_walker.H
    )

    # Test 4: Single walk generation
    walk = brownian_walker.generate_walk(end_x=num_steps // 3, end_y=num_steps // 4)
    plot_walk(walk_points=walk, terrain_height=brownian_walker.H, terrain_width=brownian_walker.W)

    walk2 = brownian_walker.generate_walk(end_x=num_steps // 2, end_y=num_steps)
    plot_walk(walk_points=walk2, terrain_height=brownian_walker2.H, terrain_width=brownian_walker2.W)

    walk3 = brownian_walker.generate_walk(end_x=num_steps // 5, end_y=num_steps // 2)
    plot_walk(walk_points=walk3, terrain_height=brownian_walker3.H, terrain_width=brownian_walker3.W)

    # Test 5: Multistep walk generation
    points = [(num_steps, num_steps), (num_steps // 2, num_steps // 2), (num_steps + 10, num_steps - 10)]
    multi_walk = brownian_walker.generate_multistep_walk(points)

    plot_walk_multistep(steps=points, walk_points=multi_walk, terrain_width=brownian_walker.W,
                        terrain_height=brownian_walker.H)


def movebank_test():
    csv_file = "1000 Cranes. Southern Kazakhstan.csv"
    [walk_csv, steps_csv] = BrownianWalker.generate_movebank_walk(csv_file, 10, 7, 100, 200, 200)
    walk_plot_array_csv = get_walk_points(walk_csv)
    plot_walk_multistep(steps_csv, walk_plot_array_csv, 200, 200)


def corr(S, D, T):
    W = 2 * T + 1
    H = 2 * T + 1

    walker = CorrelatedWalker(D, S, W, H, T)
    walker.generate_kernel()

    # Time the execution

    walker.generate(start_x=50, start_y=50)
    start = time.perf_counter()
    walk = walker.backtrace(180, 190, 0)
    end = time.perf_counter()
    print(f"Generated walk in {end - start:.4f} seconds")
    plot_walk(walk, W, H)
    return None


def corr_simple(D):
    S = 15
    T = 120
    W = 401
    H = 401

    walker = CorrelatedWalker(D, S, W, H, T)
    walker.generate(start_x=200, start_y=100)

    # Time the execution
    start = time.perf_counter()
    walk = walker.backtrace(330, 280, 0)
    end = time.perf_counter()

    print(f"Generated walk in {end - start:.4f} seconds")
    plot_walk(walk, 401, 401)


def corr_movebank_test():
    D = 16
    S = 10
    T = 50
    W = 2 * T + 1
    H = 2 * T + 1

    walker = CorrelatedWalker(D, S, W, H, T)
    walker.generate_kernel()

    # Time the execution
    start = time.perf_counter()
    csv_file = '1000 Cranes. Southern Kazakhstan.csv'
    [walk, steps] = walker.generate_movebank_walk(
        csv_file, 8)
    end = time.perf_counter()

    print(f"Generated walk in {end - start:.4f} seconds")
    plot_walk_multistep(steps, walk, terrain_width=W, terrain_height=H)


@profile
def mixed_walk():
    T = 100

    study = "baboon_SA_study/"
    walker = MixedWalker(T=T, resolution=200, study_folder=study)
    walker.generate_walk(serialized=True, steps=[(66, 66), (150, 100)])


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


def test_biased_walk():
    terrain = parse_terrain("terrain2.txt", " ")
    west_bias = [(-4, 0)] * 20
    no_bias = [(0, 0)] * 20
    east_bias = [(4, 0)] * 20
    biases = west_bias + no_bias + east_bias
    biased_walk = BiasedWalker(terrain=terrain, bias_array=biases)
    biased_walk.generate(start_x=100, start_y=150)
    walk = biased_walk.backtrace(end_x=100, end_y=45)
    plot_combined_terrain(
        terrain,
        walk,
        terrain.width,
        terrain.height,
        title="Time-Aware Mixed Walk"
    )


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
    start = (150, 170)
    end = (100, 100)

    walker = MixedTimeWalker(
        T=150,
        resolution=200,
        duration_in_days=7,
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
        T=100,
        resolution=200,
        duration_in_days=5,
        study_folder="baboon_SA_study/"
    )
    walker.preprocess()
    walker.generate_walk_multi(steps=steps, output_file="time_walk3.json")


if __name__ == "__main__":
    mixed_walk()
    # corr(7, 16, 300)
    # benchmark_brownian(200)
    # mixed_walk()
    # test_time_walk()
    # test_weather()
"""
    test_time_walker()
    test_time_walker_multi()
    test_biased_walk()
    test_brownian_wrapper()
    corr(16)
    test()
    test_mixed_walk_time()
    # move_bank_test()
    landcover_test()
    test_brownian_wrapper()
    movebank_test()
    test_terrain_correlated()
    test_mixed_walk_time()
    corr_simple(12)
    corr_simple(16)
    corr_simple(20)
    corr_simple(24)
    test_terrain()
    mixed_walk()
    test_terrain_correlated()
    corr()
    corr_movebank_test()
    movebank_test()
    test_brownian_wrapper()
    """
# benchmark_brownian(200)
