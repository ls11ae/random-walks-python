import os
import pickle
from pathlib import Path

import movingpandas as mpd
import numpy as np
from kernelcma import StateKernelFactory

from randomwalks import *
from randomwalks.bindings.walk_visualization import save_trajectory_collection_timed, save_trajectory_coll_leaflet

Feature = feature_enum()

OYSTER_CATCHER_KERNELS = (
        Path(__file__).resolve().parent
        / "oyster_catcher"
        / "kernels"
        / "criteria"
        / "kernels_p99"
        / "correlated__regularized__dt60s__p99.npz"
)

studies: dict[str, str] = {
    "buffalos": "/home/omar/PycharmProjects/RW-Python-gitlab/outputs/buffalo_state_ud/buffalo_movebank.csv",
    "turtles": "/home/omar/PycharmProjects/RW-Python-gitlab/tests/moveapps/turtles.pickle",
    "bears": "/home/omar/PycharmProjects/RW-Python-gitlab/tests/moveapps/slovenia_bears.pickle",
    "boars": "/home/omar/PycharmProjects/RW-Python-gitlab/tests/boars/boars2.pickle",
    "cattle": "/home/omar/PycharmProjects/RW-Python-gitlab/tests/cattle/cattle.pickle",
    "rubythroat": "/home/omar/PycharmProjects/RW-Python-gitlab/tests/rubythroat/siberian_rubythroat_qinghai.pickle",
    "shark": "/home/omar/PycharmProjects/RW-Python-gitlab/tests/shark/shark_13_filtered_full.csv",
    "oc": "/home/omar/PycharmProjects/RW-Python-gitlab/tests/oyster_catcher/best_criteria.pickle",
    # "oc": "/home/omar/PycharmProjects/RW-Python-gitlab/tests/oyster_catcher/baseline_annotation.pickle",
}

SHARK_EXCLUDED_SOURCE_ROWS = {
    449: "on land (Kauai)",
    834: "isolated southeast location outlier",
}


def save_walk_outputs(walker, result):
    walker.walks_directory.mkdir(parents=True, exist_ok=True)
    with (walker.walks_directory / "all_walks.pickle").open("wb") as output:
        pickle.dump(result, output)
    save_trajectory_collection_timed(result, walker.walks_directory / "all_walks.html")


def load_filtered_shark_data():
    import pandas as pd

    points = pd.read_csv(studies["shark"])
    source_row_col = "Unnamed: 0"
    excluded = points[source_row_col].isin(SHARK_EXCLUDED_SOURCE_ROWS)
    found = set(points.loc[excluded, source_row_col])
    missing = set(SHARK_EXCLUDED_SOURCE_ROWS) - found
    if missing:
        raise ValueError(f"Expected shark source rows were not found: {sorted(missing)}")

    for source_row in sorted(found):
        reason = SHARK_EXCLUDED_SOURCE_ROWS[source_row]
        print(f"Excluding shark source row {source_row}: {reason}")
    return points.loc[~excluded].copy()


def print_output():
    import pandas as pd

    points = pd.read_csv(studies["buffalos"], parse_dates=["timestamp"])
    print(points.head(20))
    print(points.columns)
    print(points.dtypes)


def buffalo_ud():
    buffalo_data_directory = os.path.dirname(studies["buffalos"])
    traj_coll = studies["buffalos"]

    # set how step sizes and number of steps are resolved. Here: 30 steps between each observed point
    movement_policy = FixedStepsPolicy(7)
    # set landmarks that act as barriers for animals, here: Buildings, roads, water (rivers, lakes, oceans etc)
    barriers = []
    # feature set for trajectory segmentation
    features = [Feature.TURN_ANGLE, Feature.SPEED]
    # initialize random walker for terrestrial animal with the movement policy, barriers and features
    with StateDependentWalker(data=traj_coll,
                              animal_type=Animal.TERRESTRIAL,
                              resolution=100,
                              out_directory=buffalo_data_directory,
                              movement_policy=movement_policy,
                              barriers=barriers) as walker:
        # annotate behavioral segments with method BCPA-HMM: Use HMM to determine state per behavioral change point
        walker.annotate_behavior(
            method=StateAnnotationMethod.HMM,
            features=features,
            num_states=3,
        )
        # Create movement kernels using Sum of Gaussians on segments
        walker.get_kernels(
            dt_tolerance=1.2,
            rnge=120,
            state_col="state",
            is_brownian=False,
            mass_percentile=0.99,
        )
        """neighborhoods = walker.save_kernel_neighborhoods(
            kernels,
            out_dir=walker.kernels_directory / "neighborhoods",
        )
        terrain_weights = walker.estimate_terrain_pair_weights(neighborhoods, lo=0.5, hi=1.5,
        print(terrain_weights)
        print(f"Saved {len(neighborhoods)} kernel terrain neighborhoods")
                                                               count_self_transitions=False)"""
        # interpolate random walks with these kernels and parameters (here: 2 interpolations per step)
        result = walker.generate_utilization_distribution(
            sample_walks=1,
            max_cell_size=5,
            save_plots=True,
            unmodelled_state_policy=UnmodelledStatePolicy.PREVIOUS,
            max_state_fill_gap="45D",
        )
        # save trajectory collection containing RW interpolations and segmentation info
        save_walk_outputs(walker, result)


def boars_ud():
    buffalo_data_directory = os.path.dirname(studies["boars"])
    traj_coll = pickle.load(open(studies["boars"], "rb"))

    # The adaptive policy reads the correlated kernel's inferred native
    # interval; it is not supplied or linearly rescaled by the caller.
    movement_policy = AdaptiveKernelMovementPolicy()
    # set landmarks that act as barriers for animals, here: Buildings, roads, water (rivers, lakes, oceans etc)
    barriers = [MesaLandcover.PERMANENT_WATER, MesaLandcover.BUILT_UP]
    # feature set for trajectory segmentation
    features = [Feature.TURN_ANGLE, Feature.SPEED]
    # initialize random walker for terrestrial animal with the movement policy, barriers and features
    with StateDependentWalker(data=traj_coll,
                              animal_type=Animal.TERRESTRIAL,
                              resolution=100,
                              out_directory=buffalo_data_directory,
                              movement_policy=movement_policy,
                              barriers=barriers) as walker:
        # annotate behavioral segments with method BCPA-HMM: Use HMM to determine state per behavioral change point
        walker.annotate_behavior(
            method=StateAnnotationMethod.HMM,
            features=features,
            num_states=3,
        )
        # Create movement kernels using Sum of Gaussians on segments
        walker.get_kernels(
            dt_tolerance=2.2,
            rnge=100,
            state_col="state",
            is_brownian=False,
            mass_percentile=0.95,
        )
        """neighborhoods = walker.save_kernel_neighborhoods(
            kernels,
            out_dir=walker.kernels_directory / "neighborhoods",
        )
        terrain_weights = walker.estimate_terrain_pair_weights(neighborhoods, lo=0.5, hi=1.5,
        print(terrain_weights)
        print(f"Saved {len(neighborhoods)} kernel terrain neighborhoods")
                                                               count_self_transitions=False)"""
        # interpolate random walks with these kernels and parameters (here: 2 interpolations per step)
        result = walker.generate_utilization_distribution(
            sample_walks=1,
            max_cell_size=10,
            save_plots=True,
            unmodelled_state_policy=UnmodelledStatePolicy.PREVIOUS,
            max_state_fill_gap="45D",
        )
        # save trajectory collection containing RW interpolations and segmentation info
        save_walk_outputs(walker, result)


def load_oyster_catcher_kernels(path=OYSTER_CATCHER_KERNELS):
    """Load state kernels and their physical radii from the baseline bundle."""
    with np.load(path, allow_pickle=False) as bundle:
        states = [int(state) for state in bundle["state_values"]]
        available = bundle["available"]
        ranges = bundle["kernel_range_m"]
        kernels = {
            state: np.asarray(bundle[f"kernel_state_{state}"], dtype=np.float64).copy()
            for state, is_available in zip(states, available)
            if is_available
        }
        kernel_ranges = {
            state: float(kernel_range)
            for state, is_available, kernel_range in zip(states, available, ranges)
            if is_available
        }

    if not kernels or kernels.keys() != kernel_ranges.keys():
        raise ValueError(f"Invalid or incomplete kernel bundle: {path}")
    return kernels, kernel_ranges


def load_oyster_catcher_trajectories(path=studies["oc"]):
    """Restore the timestamp metadata omitted by the baseline pickle."""
    with open(path, "rb") as source:
        trajectories = pickle.load(source)

    if trajectories.t is not None:
        return trajectories

    points = trajectories.to_point_gdf().copy()
    if "timestamp" not in points.columns:
        raise ValueError(f"Oystercatcher trajectories in {path} have no timestamp column.")
    return mpd.TrajectoryCollection(
        points,
        traj_id_col=trajectories.get_traj_id_col(),
        t="timestamp",
    )


def oyster_catcher():
    """Generate UDs from the saved baseline correlated kernels."""
    study_data_dir = Path(studies["oc"]).parent / "criteria_correlated_regularized_ud"
    traj_coll = load_oyster_catcher_trajectories()
    kernels, kernel_ranges = load_oyster_catcher_kernels()

    # This external NumPy bundle has no timestep metadata; 60 s is its native
    # training interval, not a request to rescale correlated displacements.
    movement_policy = AdaptiveKernelMovementPolicy(dt_model_s=60)
    with StateDependentWalker(data=traj_coll,
                              animal_type=Animal.AIRBORNE,
                              resolution=100,
                              out_directory=study_data_dir,
                              movement_policy=movement_policy,
                              barriers=[],
                              n=10) as walker:
        # The pickle is already annotated; keep those states aligned with the
        # state numbers in the baseline kernel bundle.
        walker._process_movebank_data(create_landcover=True)
        walker.kernel_state_col = "state"
        walker.is_brownian = False

        walker.generate_utilization_distribution(
            kernels=kernels,
            kernel_ranges=kernel_ranges,
            sample_walks=0,
            max_cell_size=10,
            save_plots=True,
            unmodelled_state_policy=UnmodelledStatePolicy.PREVIOUS,
            max_state_fill_gap="45D",
        )


def oyster_catcher_kernelcma():
    """Fit matching kernels with kernelcma, then generate the same UDs."""
    study_data_dir = Path(studies["oc"]).parent / "kernelcma_correlated_regularized_ud"
    traj_coll = load_oyster_catcher_trajectories()
    # Raw arrays below omit Kernel2D metadata, so retain their native 60 s
    # interval as the adaptive-policy fallback.
    movement_policy = AdaptiveKernelMovementPolicy(dt_model_s=60)

    with StateDependentWalker(data=traj_coll,
                              animal_type=Animal.AIRBORNE,
                              resolution=200,
                              out_directory=study_data_dir,
                              movement_policy=movement_policy,
                              barriers=[],
                              n=5) as walker:
        # Reuse the baseline states and fit only the movement kernels.
        walker._process_movebank_data(create_landcover=True)
        points = walker.animal_proc.traj.to_point_gdf()
        factory = StateKernelFactory(
            points,
            id_col=walker.animal_proc.traj.get_traj_id_col(),
            time_col="timestamp",
            state_col="state",
        )
        walker.kernels_directory.mkdir(parents=True, exist_ok=True)
        correlated, _ = factory.get_state_kernels(
            dt_tolerance=1.3,
            rnge=5000,
            reso=1001,
            out=walker.kernels_directory / "kernels.png",
            density_config="regularized",
            mass_percentile=99,
        )
        kernels = {
            int(kernel.state_value): np.asarray(kernel.Z, dtype=np.float64)
            for kernel in correlated
            if kernel.Z is not None
        }
        kernel_ranges = {
            int(kernel.state_value): float(kernel.rnge)
            for kernel in correlated
            if kernel.Z is not None
        }
        walker.kernel_state_col = "state"
        walker.is_brownian = False

        walker.generate_utilization_distribution(
            kernels=kernels,
            kernel_ranges=kernel_ranges,
            sample_walks=0,
            max_cell_size=10,
            save_plots=True,
            unmodelled_state_policy=UnmodelledStatePolicy.PREVIOUS,
            max_state_fill_gap="45D",
        )


def cattle_ud(n=1):
    buffalo_data_directory = os.path.dirname(studies["cattle"])
    traj_coll = pickle.load(open(studies["cattle"], "rb"))

    # Fit states and kernels to every one-minute fix. ``n`` is applied only to
    # the endpoints subsequently used for interpolation and UD generation.

    movement_policy = AdaptiveKernelMovementPolicy()
    # set landmarks that act as barriers for animals, here: Buildings, roads, water (rivers, lakes, oceans etc)
    barriers = []
    # feature set for trajectory segmentation
    features = [Feature.TURN_ANGLE, Feature.SPEED]
    # initialize random walker for terrestrial animal with the movement policy, barriers and features
    with StateDependentWalker(data=traj_coll,
                              animal_type=Animal.TERRESTRIAL,
                              resolution=100,
                              out_directory=buffalo_data_directory,
                              movement_policy=movement_policy,
                              barriers=barriers,
                              n=n) as walker:
        # annotate behavioral segments with method BCPA-HMM: Use HMM to determine state per behavioral change point
        walker.annotate_behavior(
            method=StateAnnotationMethod.HMM,
            features=features,
            num_states=3,
        )
        # Create movement kernels using Sum of Gaussians on segments
        walker.get_kernels(
            dt_tolerance=2.2,
            rnge=100,
            state_col="state",
            is_brownian=False,
            mass_percentile=0.99,
        )
        """neighborhoods = walker.save_kernel_neighborhoods(
            kernels,
            out_dir=walker.kernels_directory / "neighborhoods",
        )
        terrain_weights = walker.estimate_terrain_pair_weights(neighborhoods, lo=0.5, hi=1.5,
        print(terrain_weights)
        print(f"Saved {len(neighborhoods)} kernel terrain neighborhoods")
                                                               count_self_transitions=False)"""
        # Generate the UD and diagnostic random-walk plots at the requested stride.
        result = walker.generate_utilization_distribution(
            sample_walks=0,
            max_cell_size=10,
            save_plots=True,
            unmodelled_state_policy=UnmodelledStatePolicy.PREVIOUS,
            max_state_fill_gap="45D",
        )
        # save trajectory collection containing RW interpolations and segmentation info
        save_walk_outputs(walker, result)


def shark_ud():
    study_directory = os.path.dirname(studies["shark"])
    shark_data = load_filtered_shark_data()
    movement_policy = AdaptiveKernelMovementPolicy()
    features = [Feature.TURN_ANGLE, Feature.SPEED]

    with StateDependentWalker(
            data=shark_data,
            animal_type=Animal.MARINE,
            resolution=200,
            out_directory=study_directory,
            movement_policy=movement_policy,
            id_col="tag-local-identifier",
    ) as walker:
        walker.annotate_behavior(
            method=StateAnnotationMethod.HMM,
            features=features,
            num_states=2,
        )
        walker.get_kernels(
            dt_tolerance=4.2,
            rnge=2000,
            state_col="state",
            is_brownian=False,
            mass_percentile=0.98,
        )
        result = walker.generate_utilization_distribution(
            sample_walks=1,
            max_cell_size=100,
            save_plots=True,
            unmodelled_state_policy=UnmodelledStatePolicy.PREVIOUS,
            max_state_fill_gap="45D",
        )
        save_walk_outputs(walker, result)


def bears_ud(n=5):
    buffalo_data_directory = os.path.dirname(studies["bears"])
    traj_coll = pickle.load(open(studies["bears"], "rb"))

    movement_policy = AdaptiveKernelMovementPolicy()
    # set landmarks that act as barriers for animals, here: Buildings, roads, water (rivers, lakes, oceans etc)
    barriers = [MesaLandcover.PERMANENT_WATER, MesaLandcover.BUILT_UP]
    # feature set for trajectory segmentation
    features = [Feature.TURN_ANGLE, Feature.SPEED]
    # initialize random walker for terrestrial animal with the movement policy, barriers and features
    with StateDependentWalker(data=traj_coll,
                              animal_type=Animal.TERRESTRIAL,
                              resolution=100,
                              out_directory=buffalo_data_directory,
                              movement_policy=movement_policy,
                              barriers=barriers,
                              n=n) as walker:
        # annotate behavioral segments with method BCPA-HMM: Use HMM to determine state per behavioral change point
        walker.annotate_behavior(
            method=StateAnnotationMethod.HMM,
            features=features,
            num_states=3,
        )
        # Create movement kernels using Sum of Gaussians on segments
        walker.get_kernels(
            dt_tolerance=2.2,
            rnge=600,
            state_col="state",
            is_brownian=False,
            mass_percentile=0.99,
        )
        """neighborhoods = walker.save_kernel_neighborhoods(
            kernels,
            out_dir=walker.kernels_directory / "neighborhoods",
        )
        terrain_weights = walker.estimate_terrain_pair_weights(neighborhoods, lo=0.5, hi=1.5,
        print(terrain_weights)
        print(f"Saved {len(neighborhoods)} kernel terrain neighborhoods")
                                                               count_self_transitions=False)"""
        # interpolate random walks with these kernels and parameters (here: 2 interpolations per step)
        result = walker.generate_utilization_distribution(
            sample_walks=1,
            max_cell_size=40,
            save_plots=True,
            unmodelled_state_policy=UnmodelledStatePolicy.PREVIOUS,
            max_state_fill_gap="45D",
        )
        # save trajectory collection containing RW interpolations and segmentation info
        save_walk_outputs(walker, result)


def buffalo_ud():
    buffalo_data_directory = os.path.dirname(studies["buffalos"])
    traj_coll = studies["buffalos"]

    # set how step sizes and number of steps are resolved. Here: 30 steps between each observed point
    movement_policy = FixedStepsPolicy(7)
    # set landmarks that act as barriers for animals, here: Buildings, roads, water (rivers, lakes, oceans etc)
    barriers = []
    # feature set for trajectory segmentation
    features = [Feature.TURN_ANGLE, Feature.SPEED]
    # initialize random walker for terrestrial animal with the movement policy, barriers and features
    with StateDependentWalker(data=traj_coll,
                              animal_type=Animal.TERRESTRIAL,
                              resolution=100,
                              out_directory=buffalo_data_directory,
                              movement_policy=movement_policy,
                              barriers=barriers) as walker:
        # annotate behavioral segments with method BCPA-HMM: Use HMM to determine state per behavioral change point
        walker.annotate_behavior(
            method=StateAnnotationMethod.HMM,
            features=features,
            num_states=3,
        )
        # Create movement kernels using Sum of Gaussians on segments
        kernels = walker.get_kernels(
            dt_tolerance=1.2,
            rnge=120,
            state_col="state",
            is_brownian=False,
            mass_percentile=0.99,
        )
        """neighborhoods = walker.save_kernel_neighborhoods(
            kernels,
            out_dir=walker.kernels_directory / "neighborhoods",
        )
        terrain_weights = walker.estimate_terrain_pair_weights(neighborhoods, lo=0.5, hi=1.5,
        print(terrain_weights)
        print(f"Saved {len(neighborhoods)} kernel terrain neighborhoods")
                                                               count_self_transitions=False)"""
        # interpolate random walks with these kernels and parameters (here: 2 interpolations per step)
        result = walker.generate_utilization_distribution(sample_walks=1, max_cell_size=5, save_plots=True)
        # save trajectory collection containing RW interpolations and segmentation info
        save_walk_outputs(walker, result)


if __name__ == "__main__":
    oyster_catcher()
    # oyster_catcher_kernelcma()
    # bears_ud(n=2)
    # boars_ud()
    # shark_ud()
    # buffalo_ud()
