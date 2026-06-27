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
    working_directory = os.getcwd() + "/tests/moveapps/"
    with open(
            "/home/omar/PycharmProjects/hmmcma/tests/annotated.pickle",
            "rb") as f:
        traj_coll = pickle.load(f)
    print(traj_coll.to_point_gdf().columns)
    movement_policy = FixedStepsPolicy(20)
    barriers = [MesaLandcover.PERMANENT_WATER, MesaLandcover.BUILT_UP]
    features = [Feature.SPEED, Feature.ANGULAR_DIFFERENCE, Feature.PERSISTENCE_VELOCITY]
    with StateDependentWalker(data=traj_coll,
                              animal_type=Animal.TERRESTRIAL,
                              resolution=300,
                              out_directory=working_directory + "data",
                              movement_policy=movement_policy,
                              barriers=barriers) as walker:
        walker.annotate_behavior(
            method=StateAnnotationMethod.HMM,
            features=features,
            num_states=3,
            plot_path=working_directory + "/states.png",
        )
        walker.get_kernels(
            dt_tolerance=1.2,
            rnge=1000,
            state_col="state",
            is_brownian=True,
            plot_dir=working_directory + "/kernels.png")
        result = walker.generate_walks()
        pickle.dump(result, open(working_directory + "walks/all_walks.pickle", "wb"))
        save_trajectory_collection_timed(result, working_directory + "walks/all_walks.html")
