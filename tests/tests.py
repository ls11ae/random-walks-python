import os
import pickle

from randomwalks import *
from randomwalks.bindings.walk_visualization import save_trajectory_collection_timed, save_trajectory_coll_leaflet

Feature = feature_enum()


def print_output():
    pickle_path = "/home/omar/PycharmProjects/RW-Python-gitlab/tests/moveapps/RW_Test__Workflow_Instance_005__move2_loc_to_MovingPandas__2026-06-23_22-38-10.pickle"
    traj_coll = pickle.load(open(pickle_path, "rb"))
    print(traj_coll.to_point_gdf().head(20))
    print(traj_coll.to_point_gdf().columns)
    print(traj_coll.to_point_gdf().dtypes)


if __name__ == "__main__":
    # load trajectory collection from pickle
    working_directory = os.getcwd() + "/tests/moveapps/"
    with open(
            "/home/omar/PycharmProjects/hmmcma/tests/annotated.pickle",
            "rb") as f:
        traj_coll = pickle.load(f)

    print(traj_coll.to_point_gdf().head())

    # set how step sizes and number of steps are resolved. Here: 30 steps between each observed point
    movement_policy = FixedStepsPolicy(5)
    # set landmarks that act as barriers for animals, here: Buildings, roads, water (rivers, lakes, oceans etc)
    barriers = [MesaLandcover.PERMANENT_WATER, MesaLandcover.BUILT_UP]
    # feature set for trajectory segmentation
    features = [Feature.SPEED, Feature.ANGULAR_DIFFERENCE, Feature.PERSISTENCE_VELOCITY]
    # initialize random walker for terrestrial animal with the movement policy, barriers and features
    with StateDependentWalker(data=traj_coll,
                              animal_type=Animal.TERRESTRIAL,
                              resolution=300,
                              out_directory=working_directory + "data",
                              movement_policy=movement_policy,
                              barriers=barriers) as walker:
        # annotate behavioral segments with method BCPA-HMM: Use HMM to determine state per behavioral change point
        walker.annotate_behavior(
            method=StateAnnotationMethod.BCPAHMM,
            features=features,
            num_states=5,
            plot_path=working_directory + "BCPAHMM/states.png",
        )
        # Create movement kernels using Sum of Gaussians on segments
        kernels = walker.get_kernels(
            dt_tolerance=1.0,
            rnge=120,
            state_col="state",
            is_brownian=True,
            plot_dir=working_directory + "TERRAIN/kernels2.png",
            mass_percentile=0.95,
            )
        neighborhoods = walker.save_kernel_neighborhoods(
            kernels,
            out_dir=working_directory + "TERRAIN/neighborhoods",
        )
        print(f"Saved {len(neighborhoods)} kernel terrain neighborhoods")
        print(kernels)
        exit()
        # interpolate random walks with these kernels and parameters (here: 2 interpolations per step)
        result = walker.generate_walks(amount=2)
        # save trajectory collection containing RW interpolations and segmentation info
        pickle.dump(result, open(working_directory + "walks/all_walks.pickle", "wb"))
        # save leaflet animation of interpolated random walks
        save_trajectory_collection_timed(result, working_directory + "walks/all_walks.html")
