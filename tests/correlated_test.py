from random_walk_package.bindings.data_structures.kernel_terrain_mapping import set_landmark_mapping, \
    set_forbidden_landmark
from random_walk_package.core.CorrelatedWalker import *


def test_correlated_walk_terrain():
    terrain = create_terrain_map('terrain_baboons.txt', ' ')
    print(terrain.contents.width, terrain.contents.height)
    kernel_mapping = create_correlated_kernel_parameters(animal_type=MEDIUM, base_step_size=8)
    set_landmark_mapping(kernel_mapping, GRASSLAND, is_brownian=False, step_size=5, directions=8, diffusity=1)
    set_landmark_mapping(kernel_mapping, TREE_COVER, is_brownian=False, step_size=5, directions=6, diffusity=2.6)
    set_forbidden_landmark(kernel_mapping, WATER)

    with CorrelatedWalker(T=150, terrain=terrain, kernel_mapping=kernel_mapping) as walker:
        walker.generate_from_terrain(start_x=50, start_y=50)
        path3 = walker.backtrace_from_terrain(end_x=150, end_y=150, plot=False)
        assert len(path3) == 150
        assert tuple(path3[0]) == (50, 50)
        assert tuple(path3[-1]) == (150, 150)


def test_correlated_serialized():
    with CorrelatedWalker(T=150, W=201, H=201, D=16, S=7) as walker:
        dp_folder_path = walker.generate(start_x=100, start_y=100, use_serialization=True)
        walk = walker.backtrace(end_x=150, end_y=150, dp_folder=dp_folder_path, plot=False)
        assert len(walk) == 150
        assert tuple(walk[0]) == (100, 100)
        assert tuple(walk[-1]) == (150, 150)
        assert os.path.exists(dp_folder_path)


def test_correlated_single():
    with CorrelatedWalker(T=150, W=201, H=201, D=16, S=7) as walker:
        walker.generate(start_x=100, start_y=100)
        walk = walker.backtrace(end_x=150, end_y=150, plot=False)
        assert len(walk) == 150
        assert tuple(walk[0]) == (100, 100)
        assert tuple(walk[-1]) == (150, 150)


def test_correlated_multistep():
    with CorrelatedWalker(T=150, W=201, H=201, D=16, S=7) as walker:
        steps = np.array([[100, 100], [150, 150], [50, 50]], dtype=np.int32)
        full_path = walker.multistep_walk(steps=steps, use_serialization=True, plot=False)
        assert len(full_path) == 2 * 150
        assert tuple(full_path[0]) == (100, 100)
        assert tuple(full_path[-1]) == (50, 50)
