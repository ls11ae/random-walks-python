from random_walk_package.bindings import create_terrain_map, GRASSLAND, TREE_COVER, \
    WATER
from random_walk_package.bindings.data_structures.kernel_terrain_mapping import set_landmark_mapping, \
    set_forbidden_landmark
from random_walk_package.core.BrownianWalker import *


def test_brownian_walk():
    terrain = create_terrain_map('terrain_baboons.txt', ' ')
    print(terrain.contents.width, terrain.contents.height)
    kernel_mapping = create_brownian_kernel_parameters(animal_type=MEDIUM, base_step_size=8)
    set_landmark_mapping(kernel_mapping, GRASSLAND, is_brownian=True, step_size=5, directions=1, diffusity=1)
    set_landmark_mapping(kernel_mapping, TREE_COVER, is_brownian=True, step_size=5, directions=1, diffusity=2.6)
    set_forbidden_landmark(kernel_mapping, WATER)

    with BrownianWalker(T=150, terrain=terrain, k_mapping=kernel_mapping) as walker:
        walker.generate_with_terrain(start_x=50, start_y=50)
        path3 = walker.backtrace_terrain(end_x=150, end_y=150, plot=False)
        assert len(path3) == 150
        assert tuple(path3[0]) == (50, 50)
        assert tuple(path3[-1]) == (150, 150)


def test_brownian_multistep():
    #  Multistep Walk
    with BrownianWalker(T=20, W=30, H=30) as walker:
        walker.generate(start_x=15, start_y=15)

        # Mehrere Zielpunkte für multistep
        steps = np.array([[10, 10], [20, 20], [5, 25], [25, 5]], dtype=np.int32)
        full_path = walker.multistep_walk(steps)

        assert len(full_path) == 3 * 20
        assert tuple(full_path[0]) == (10, 10)
        assert tuple(full_path[20]) == (20, 20)
        assert tuple(full_path[40]) == (5, 25)
        assert tuple(full_path[-1]) == (25, 5)

    # 5. Verschiedene Kernel Parameter testen
    print("\n5. Verschiedene Kernel Parameter")


def test_brownian_kernels():
    kernels_to_test = [
        ("Kleiner Kernel (S=1)", None, 1.0, 1),
        ("Mittlerer Kernel (S=2)", None, 2.0, 2),
        ("Großer Kernel (S=3)", None, 3.0, 3),
    ]

    for name, kernel_np, sigma, S in kernels_to_test:
        with BrownianWalker(T=15, W=25, H=25) as walker:
            walker.set_kernel(kernel_np=kernel_np, sigma=sigma, S=S)
            walker.generate(start_x=12, start_y=12)
            path = walker.backtrace(end_x=18, end_y=18)
            assert len(path) == 15
            assert tuple(path[0]) == (12, 12)
            assert tuple(path[-1]) == (18, 18)


def test_brownian_errors():
    exc = False
    try:
        walker = BrownianWalker(W=10, H=10)
        walker.backtrace(5, 5)
    except ValueError as e:
        print(f"   Erwarteter Fehler: {e}")
        exc = True
    assert exc
    exc = False
    try:
        with BrownianWalker(W=10, H=10) as walker:
            walker.generate(start_x=5, start_y=5)
            walker.backtrace(15, 15)
    except ValueError as e:
        print(f"   Erwarteter Fehler: {e}")
        exc = True
    assert exc

    print("\n7. Komplexes Szenario: Terrain mit verschiedenen Einstellungen")


def test_brownian_complex_terrain():
    complex_terrain = create_terrain_map('terrain_baboons.txt', ' ')
    with BrownianWalker(T=60, S=7, terrain=complex_terrain) as walker:
        routes = [
            ((120, 20), (160, 160)),
            ((220, 210), (102, 240)),
            ((270, 130), (130, 270))
        ]

        for i, (start, end) in enumerate(routes):
            walker.generate_with_terrain(start_x=start[0], start_y=start[1])
            path = walker.backtrace_terrain(end_x=end[0], end_y=end[1])
            print(path)
            print(f"   Route {i + 1}: {start} -> {end}: {len(path)} Punkte")
            assert len(path) == 60
            assert tuple(path[0]) == start
            assert tuple(path[-1]) == end
