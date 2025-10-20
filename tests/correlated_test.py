from random_walk_package.bindings.data_structures.kernel_terrain_mapping import set_landmark_mapping, \
    set_forbidden_landmark
from random_walk_package.core.CorrelatedWalker import *


def test_correlated_walk():
    terrain = create_terrain_map('terrain_baboons.txt', ' ')
    print(terrain.contents.width, terrain.contents.height)
    kernel_mapping = create_correlated_kernel_parameters(animal_type=MEDIUM, base_step_size=8)
    set_landmark_mapping(kernel_mapping, GRASSLAND, is_brownian=False, step_size=5, directions=8, diffusity=1)
    set_landmark_mapping(kernel_mapping, TREE_COVER, is_brownian=False, step_size=5, directions=6, diffusity=2.6)
    set_forbidden_landmark(kernel_mapping, WATER)

    with CorrelatedWalker(T=150, terrain=terrain, kernel_mapping=kernel_mapping) as walker:
        walker.generate_from_terrain(start_x=50, start_y=50)
        path3 = walker.backtrace_from_terrain(end_x=150, end_y=150, plot=True)
        print(f"   Terrain Walk: {len(path3)} Punkte")
    # hier noch die dinger plotten also kernels und terrain


def test_correlated_serialized():
    with CorrelatedWalker(T=150, W=201, H=201, D=16, S=7) as walker:
        dp_folder_path = walker.generate(start_x=100, start_y=100, use_serialization=True)
        walker.backtrace(end_x=150, end_y=150, dp_folder=dp_folder_path, plot=True)


def test_correlated_multistep():
    with CorrelatedWalker(T=150, W=201, H=201, D=16, S=7) as walker:
        steps = np.array([[100, 100], [150, 150], [50, 50]], dtype=np.int32)
        full_path = walker.multistep_walk(steps=steps, use_serialization=True, plot=True)
        print(full_path)


def demonstrate_all_functionality():
    # Sample Terrain erstellen
    terrain = create_terrain_map('terrain_baboons.txt', ' ')
    kernel_mapping = create_correlated_kernel_parameters(animal_type=MEDIUM, base_step_size=7)
    set_landmark_mapping(kernel_mapping, GRASSLAND, is_brownian=False, step_size=5, directions=8, diffusity=1)
    set_landmark_mapping(kernel_mapping, TREE_COVER, is_brownian=False, step_size=5, directions=6, diffusity=2.6)
    set_forbidden_landmark(kernel_mapping, WATER)

    with CorrelatedWalker(T=150, terrain=terrain, kernel_mapping=kernel_mapping) as walker:
        walker.generate_from_terrain(start_x=50
                                     , start_y=50)
        path3 = walker.backtrace_from_terrain(end_x=150
                                              , end_y=150
                                              , plot=True)
        print(f"   Terrain Walk: {len(path3)} Punkte")

    # 4. Multistep Walk
    print("\n4. Multistep Walk Generierung")
    with CorrelatedWalker(T=100, S=7, D=8, W=200, H=200) as walker:
        # Mehrere Zielpunkte für multistep
        steps = np.array([[100, 100], [150, 150], [25, 25]], dtype=np.int32)
        full_path = walker.multistep_walk(steps)
        print(full_path)

    # 5. Verschiedene Kernel Parameter testen
    print("\n5. Verschiedene Kernel Parameter")

    kernels_to_test = [
        ("Kleiner Kernel (S=1)", None, 1.0, 1),
        ("Mittlerer Kernel (S=2)", None, 2.0, 2),
        ("Großer Kernel (S=3)", None, 3.0, 3),
    ]

    for name, kernel_np, sigma, S in kernels_to_test:
        with CorrelatedWalker(T=15, W=25, H=25) as walker2:
            walker2.set_kernel(kernel_np=kernel_np, d=8, S=S)
            walker2.generate(start_x=12, start_y=12)
            path = walker2.backtrace(end_x=18, end_y=18)

    # 6. Fehlerbehandlung demonstrieren
    print("\n6. Fehlerbehandlung")

    try:
        walker = CorrelatedWalker(W=10, H=10)
        walker.backtrace(5, 5)  # Sollte fehlschlagen (noch nicht generiert)
    except ValueError as e:
        print(f"   Erwarteter Fehler: {e}")

    try:
        with CorrelatedWalker(W=10, H=10) as walker:
            walker.generate(start_x=5, start_y=5)
            walker.backtrace(15, 15)  # Ungültige Koordinaten
    except ValueError as e:
        print(f"   Erwarteter Fehler: {e}")

    # 7. Komplexes Szenario: Terrain mit Custom Kernel
    print("\n7. Komplexes Szenario: Terrain mit verschiedenen Einstellungen")

    complex_terrain = create_terrain_map('terrain_baboons.txt', ' ')

    with CorrelatedWalker(T=100, terrain=complex_terrain) as walker:
        # Verschiedene Start/End-Paare testen
        routes = [
            ((220, 220), (260, 260)),
            ((240, 210), (210, 240)),
            ((70, 230), (230, 70))
        ]

        for i, (start, end) in enumerate(routes):
            walker.generate_from_terrain(start_x=start[0], start_y=start[1])
            path = walker.backtrace_from_terrain(end_x=end[0], end_y=end[1])
            print(f"   Route {i + 1}: {start} -> {end}: {len(path)} Punkte")

    test_correlated_walk()


def performance_test():
    """Performance-Test für große Walks."""
    print("\n=== Performance Test ===")

    import time

    # Großer Walk testen
    sizes = [(100, 100, 50), (200, 200, 100), (300, 300, 150)]

    for W, H, T in sizes:
        start_time = time.time()

        with CorrelatedWalker(W=W, H=H, T=T) as walker:
            walker.generate(start_x=W // 2, start_y=H // 2)
            path = walker.backtrace(end_x=W // 4, end_y=H // 4)

        duration = time.time() - start_time
        print(f"   Größe {W}x{W}, T={T}: {duration:.2f} Sekunden, {len(path)} Punkte")
