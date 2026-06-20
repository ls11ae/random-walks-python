import os
import pickle

from randomwalks import *
from randomwalks.bindings.walk_visualization import save_trajectory_collection_timed, save_trajectory_coll_leaflet


def main():
    print("Hello World!")


if __name__ == "__main__":
    """pickle_path = "/home/omar/PycharmProjects/RW-Python-gitlab/tests/moveapps/walks/all_walks.pickle"
    traj_coll = pickle.load(open(pickle_path, "rb"))
    print(traj_coll.to_point_gdf().head(20))

    save_trajectory_coll_leaflet(traj_coll,
                                 save_path="/home/omar/PycharmProjects/RW-Python-gitlab/tests/moveapps/walks/")"""

    working_directory = os.getcwd() + "/tests/moveapps/"
    with open(working_directory + "turtles.pickle", "rb") as f:
        traj_coll = pickle.load(f)
    print(traj_coll.to_point_gdf().columns)
    movement_policy = FixedStepsPolicy(20)
    barriers = [MesaLandcover.PERMANENT_WATER, MesaLandcover.BUILT_UP]
    with StateDependentWalker(data=traj_coll,
                              animal_type=Animal.TERRESTRIAL,
                              resolution=300,
                              out_directory=working_directory + "data",
                              movement_policy=movement_policy,
                              barriers=barriers) as walker:
        walker.get_kernels(n_hmm_states=2,
                           dt_tolerance=4,
                           rnge=100,
                           is_brownian=True)
        result = walker.generate_walks()
        pickle.dump(result, open(working_directory + "walks/all_walks.pickle", "wb"))
        save_trajectory_collection_timed(result, working_directory + "walks/all_walks.html")
