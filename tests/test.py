# debugging: gdb --args python -m tests.test
import gzip
import os.path
import pickle
import random
from datetime import datetime

import pandas as pd

from random_walk_package import set_forbidden_landmark, MixedTimeWalker
from random_walk_package.bindings.data_processing.environment_handling import parse_kernel_parameters, \
    get_kernels_environment_grid, free_environment_influence_grid, free_kernel_parameters_yxt
from random_walk_package.bindings.data_structures.kernel_terrain_mapping import kernel_mapping_free
from random_walk_package.core.MixedWalker import *
from random_walk_package.core.StateDependentWalker import StateDependentWalker
from random_walk_package.data_sources.walk_visualization import save_trajectory_collection_timed, \
    save_trajectory_coll_leaflet
from tests.mixed_walk_test import test_mixed_walk


def weather_terrain_params(landmark, row):
    is_brownian = True
    S = float(row["wind_speed_10m_max"]) / 2.0
    D = int(row["wind_direction_10m_dominant"] // 45)
    diffusity = float(row["cloud_cover_mean"]) / 100.0
    bias_x = int(row["precipitation_sum"] > 0.1)
    bias_y = int(landmark in (50, 80))
    return [is_brownian, S, D, diffusity, bias_x, bias_y]


def read_weather_csv(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df = df.loc[:, df.columns != "index"]

    return df


def environment_pipeline_test():
    # merging weather data to one dataset
    out_dir = "weather_data_full.csv"
    data_dir = "/home/omar/PycharmProjects/random-walks-python/random_walk_package/resources/leap_of_the_cat/weather_data/CAMILA"
    csv_files = [
        os.path.join(data_dir, file)
        for file in os.listdir(data_dir)
        if file.endswith(".csv") and not file.endswith(out_dir)
    ]
    print(len(csv_files))
    dfs = (read_weather_csv(f) for f in csv_files)

    # Dataframe for your environmental data, for weather i need to merge the CSVs, normally you wouldnt need this
    df_full = pd.concat(dfs).sort_values("timestamp").reset_index(drop=True)
    df_full.to_csv(os.path.join(data_dir, "weather_data_full.csv"))

    # path to the study containing the animal movement data
    study = 'leap_of_the_cat/The Leap of the Cat.csv'
    processor = AnimalMovementProcessor(study)
    # creates landcover grid txt files
    processor.create_landcover_data_txt(resolution=200, out_directory='leap_of_the_cat')

    # although filtering of dates also happens in C, it makes sense to set the dates of the interval of the study here
    start_date = datetime(1999, 8, 23)
    end_date = datetime(2030, 12, 6)

    # create kernel params csv files
    paths, times = processor.kernel_params_per_animal_csv(df=df_full,
                                                          kernel_resolver=weather_terrain_params,
                                                          start_date=start_date,
                                                          end_date=end_date,
                                                          time_stamp='timestamp',
                                                          lon='longitude',
                                                          lat='latitude')

    dimensions = processor.env_samples, processor.env_samples, times

    # C allocated, must be freed manually
    environment_parameters: EnvironmentInfluenceGrid = parse_kernel_parameters(paths['CAMILA'], start_date, end_date,
                                                                               dimensions)
    # some terrain txt, path starts from resources/ folder
    terrain = create_terrain_map("baboon_SA_study/landcover_baboons123_200.txt", " ")

    # mapping which defines kernel parameters based on landmark
    mapping = create_mixed_kernel_parameters(animal_type=HEAVY, base_step_size=7)
    set_forbidden_landmark(mapping, landmark=WATER)

    # C structure holding kernel params for each x,y in terrain grid and t in interval start_date, end_date
    kernel_environment: KernelParamsYXT = get_kernels_environment_grid(terrain, environment_parameters, mapping,
                                                                       environment_weight=0.5)
    # the actual walk: for a better idea, check out the MixedTimeWalk implementation (and how these details are abstracted/bundled for the user) in core/
    # for each 2 consecutive points, we set the number of steps we want in the RW, and the correct kernels are computed and used
    T = 80
    # the projected animal coords you iterate
    start_point = (50, 50)
    end_point = (150, 150)

    walk = environment_mixed_walk(T, mapping, terrain, kernel_environment, start_date, end_date, start_point, end_point)
    dll.point2d_array_print(walk)
    walknp = get_walk_points(walk)
    # print(terrain.contents.width, terrain.contents.height)
    plot_combined_terrain(terrain, walknp, steps=[(50, 50), (150, 150)], title="⋆༺︎⋆ my fancy walk ⋆༻⋆")

    # free C allocated memory (i will probably do that on the C side instead, unless we need these multiple times)
    point2d_arr_free(walk)
    kernel_mapping_free(mapping)
    free_kernel_parameters_yxt(kernel_environment)
    terrain_map_free(terrain)
    free_environment_influence_grid(environment_parameters)


def save_walks_pickle(filepath, traj_coll, pickle_name="walks.pickle"):
    pickle_path = os.path.join(filepath, pickle_name)
    with gzip.open(pickle_path, 'wb') as f:
        pickle.dump(traj_coll, f, protocol=pickle.HIGHEST_PROTOCOL)
        return pickle_path


def load_walks_pickle(filepath):
    with gzip.open(filepath, 'rb') as f:
        return pickle.load(f)


# map row of your csv to kernel params, terrain is always part of a row, so is x,y,t if needed
# keep in mind that NaN values can (and almost always) appear so must be handled here (unless you filled them earlier)
def marine_params(row):
    uo = row.get("uo")
    vo = row.get("vo")

    if pd.isna(uo) or pd.isna(vo):
        bias_x = 0
        bias_y = 0
        is_brownian = False
        diffusity = 1.0
    else:
        bias_x = int(np.round(float(uo) * 10))
        bias_y = int(np.round(float(vo) * 10))
        is_brownian = row.get("depth", 0) < 0.2
        diffusity = 0.9

    S = random.randint(3, 7)
    D = 8

    return [
        bool(is_brownian),
        float(S),
        int(D),
        float(diffusity),
        int(bias_x),
        int(bias_y),
    ]


if __name__ == "__main__":
    study_path = 'random_walk_package/resources/tiger_sharks/shark_13_filtered.csv'
    study_df = pd.read_csv(study_path)
    env_samples = 5
    T = 25
    # i took the original csv but this also works for your processed csv with additional data, just adapt the kernel resolver
    env_path = '/home/omar/Downloads/current_filename.csv'
    processor = AnimalMovementProcessor(study_df, env_samples=env_samples)
    processor.create_landcover_data_txt(is_marine=True, resolution=1000, out_directory=os.path.dirname(study_path))
    processor.kernel_params_per_animal_csv2(env_path=env_path,
                                            kernel_resolver=marine_params,
                                            time_stamp="time", lon="longitude", lat="latitude",
                                            out_directory=os.path.dirname(study_path))

    kernel_path = 'random_walk_package/resources/tiger_sharks/kernels/204413/204413_kernel_data_2024-09-12 12:00:00-2024-09-13 12:00:00.csv'
    kernel_df = pd.read_csv(kernel_path)
    kernel_df["time"] = pd.to_datetime(kernel_df["time"])
    kernel_df = kernel_df.sort_values(["y", "x", "time"])  # eig unnötig noch mal zu sortieren aber sicher ist sicher

    import struct

    DT_FMT = "<4i"  # 4 ints
    KP_FMT = "<?qqfqq"  # bool, 2 x size_t, float, 2 x size_t
    DIMS_FMT = "<qqq"  # 3 x size_t
    binary_dir = os.path.join(os.path.dirname(study_path), "env_2024-09-12_10_11.bin")
    with open(binary_dir, "wb") as f:
        # write dimensions
        f.write(struct.pack(
            DIMS_FMT,
            env_samples,
            env_samples,
            T
        ))

        for y in range(env_samples):
            for x in range(env_samples):
                cell = kernel_df[(kernel_df["y"] == y) & (kernel_df["x"] == x)]

                for _, row in cell.iterrows():
                    t = row["time"]

                    # DateTime
                    f.write(struct.pack(
                        DT_FMT,
                        t.year,
                        t.month,
                        t.day,
                        t.hour
                    ))

                    # KernelParameters
                    f.write(struct.pack(
                        KP_FMT,
                        bool(row["is_brownian"]),
                        int(row["S"]),
                        int(row["D"]),
                        float(row["diffusity"]),
                        int(row["bias_x"]),
                        int(row["bias_y"])
                    ))

                    # landmark
                    f.write(struct.pack("<i", int(row["terrain"])))
        print(
            struct.calcsize(DIMS_FMT) +
            env_samples * env_samples * T *
            (struct.calcsize(DT_FMT) + struct.calcsize(KP_FMT) + 4)
        )
        print(
            f"check if your {binary_dir} size matches the previous print in bytes, with stat -c {"%s"} <test.txt> in Linux")
    exit(0)
    study = "random_walk_package/resources/biology_birds/Biology of birds practical.csv"
    study_dir = os.path.dirname(study)

    df = pd.read_csv(study)
    from random_walk_package.bindings.data_structures.terrain import AIRBORNE, MANGROVES

    mapping = create_mixed_kernel_parameters(AIRBORNE, 7)
    # set_forbidden_landmark(mapping, MANGROVES)
    behavioral_walker = StateDependentWalker(data=df, mapping=mapping, out_directory=study_dir, resolution=600)
    trajectory_collection = behavioral_walker.generate_walks("")
    save_walks_pickle(study_dir, trajectory_collection, "state_walk.pickle")
    traj_coll = load_walks_pickle(os.path.join(study_dir, "state_walk.pickle"))
    save_trajectory_coll_leaflet(traj_coll, study_dir)
    exit(0)
    test_mixed_walk()
    for traj in proc.traj:
        df1 = traj.df[["location-long", "location-lat", "distance", "timedelta", "direction"]]
        print(df1)
    model = MarineMovement(data=data, age_class="pup")
    model.coordinates_to_xy()
    dx, dy, steps = model.compute_step_lengths()
    bear, turning = model.turning_angles()

    dt = model.compute_time_intervals()
    D = model.diffusivity()
    r, kappa = model.directional_persistance()
    print(len(steps))
    print(len(dt))
    print(len(turning))
    df = pd.DataFrame(
        {
            "distance": steps,
            "timedelta": dt,
            "direction": bear
        }
    )
    print(df)
    exit(0)

    print("steps", steps, "\nbear", bear, "\nturning", turning, "\ndt", dt)

    mean_v = model.behavioural_speed()

    print(data.head())
    resources_dir = os.path.dirname("random_walk_package/resources/leap_of_the_cat/walks")
    traj_coll = test_mixed_walk()
    pickle_path = save_walks_pickle(resources_dir, traj_coll)
    trj_coll = load_walks_pickle(pickle_path)
    leaflet_path = save_trajectory_collection_timed(traj_coll=trj_coll, save_path=str(os.path.dirname(pickle_path)))
    save_trajectory_coll_leaflet(trj_coll, save_path=str(os.path.dirname(pickle_path)))
    exit(0)
    # test_time_walker()
    # test_mixed_walk()
    study = 'random_walk_package/resources/leap_of_the_cat/The Leap of the Cat.csv'
    df = pd.read_csv(study)

    environment_csv = 'random_walk_package/resources/movebank_test/weather/weather_data_full.csv'
    df_env = pd.read_csv(environment_csv)

    out_dir = os.path.dirname(study)

    mapping = create_mixed_kernel_parameters(animal_type=HEAVY, base_step_size=7)
    walker = MixedTimeWalker(data=df,
                             env_data=df_env,
                             kernel_mapping=mapping,
                             resolution=400,
                             out_directory=out_dir,
                             env_samples=5,
                             kernel_resolver=weather_terrain_params,
                             time_col="timestamp",
                             lon_col="location-long",
                             lat_col="location-lat",
                             id_col="tag-local-identifier",
                             crs="EPSG:4326"
                             )

    processor = AnimalMovementProcessor(data=df,
                                        lat_col="location-lat",
                                        lon_col="location-long",
                                        time_col="timestamp",
                                        id_col="tag-local-identifier",
                                        crs="EPSG:4326")
    # creates landcover grid txt files
    processor.create_landcover_data_txt(resolution=500,
                                        out_directory='random_walk_package/resources/leap_of_the_cat/terrain')
    # processor.fetch_open_meteo_weather('random_walk_package/resources/leap_of_the_cat/weather', samples_per_dimension=2)
    movement_data = processor.create_movement_data_dict()

    start_date = datetime(2000, 8, 24)
    end_date = datetime(2001, 1, 15)

    # create kernel params csv files
    paths, _ = processor.kernel_params_per_animal_csv(df=df_env,
                                                      kernel_resolver=weather_terrain_params,
                                                      start_date=start_date,
                                                      end_date=end_date,
                                                      time_stamp='timestamp',
                                                      lon='longitude',
                                                      lat='latitude',
                                                      out_directory='random_walk_package/resources/leap_of_the_cat/kernel_data')
    print(paths)
