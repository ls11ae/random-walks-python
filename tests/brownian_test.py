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
        walker.generate_from_terrain(start_x=50, start_y=50)
        path3 = walker.backtrace_from_terrain(end_x=150, end_y=150, plot=True)
        print(f"   Terrain Walk: {len(path3)} Punkte")
    # hier noch die dinger plotten also kernels und terrain


def demonstrate_all_functionality():
    # Sample Terrain erstellen
    terrain = create_terrain_map('terrain_baboons.txt', ' ')
    kernel_mapping = create_brownian_kernel_parameters(animal_type=MEDIUM, base_step_size=4)
    set_landmark_mapping(kernel_mapping, GRASSLAND, is_brownian=True, step_size=5, directions=1, diffusity=1)
    set_landmark_mapping(kernel_mapping, TREE_COVER, is_brownian=True,
                         step_size=5,
                         directions=1,
                         diffusity=2.6)
    set_forbidden_landmark(kernel_mapping, WATER)

    with BrownianWalker(T=150, terrain=terrain, k_mapping=kernel_mapping) as walker:
        walker.generate_from_terrain(start_x=50, start_y=50)
        path3 = walker.backtrace_from_terrain(end_x=150, end_y=150, plot=True)
        print(f"   Terrain Walk: {len(path3)} Punkte")

    # 4. Multistep Walk
    print("\n4. Multistep Walk Generierung")
    with BrownianWalker(T=20, W=30, H=30) as walker:
        walker.generate(start_x=15, start_y=15)

        # Mehrere Zielpunkte für multistep
        steps = np.array([[10, 10], [20, 20], [5, 25], [25, 5]], dtype=np.int32)
        full_path = walker.generate_multistep_walk(steps)
        print(full_path)

    # 5. Verschiedene Kernel Parameter testen
    print("\n5. Verschiedene Kernel Parameter")

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

    # 6. Fehlerbehandlung demonstrieren
    print("\n6. Fehlerbehandlung")

    try:
        walker = BrownianWalker(W=10, H=10)
        walker.backtrace(5, 5)  # Sollte fehlschlagen (noch nicht generiert)
    except ValueError as e:
        print(f"   Erwarteter Fehler: {e}")

    try:
        with BrownianWalker(W=10, H=10) as walker:
            walker.generate(start_x=5, start_y=5)
            walker.backtrace(15, 15)  # Ungültige Koordinaten
    except ValueError as e:
        print(f"   Erwarteter Fehler: {e}")

    # 7. Komplexes Szenario: Terrain mit Custom Kernel
    print("\n7. Komplexes Szenario: Terrain mit verschiedenen Einstellungen")

    complex_terrain = create_terrain_map('terrain_baboons.txt', ' ')

    with BrownianWalker(T=50, terrain=complex_terrain) as walker:
        # Verschiedene Start/End-Paare testen
        routes = [
            ((20, 20), (60, 60)),
            ((40, 10), (10, 40)),
            ((70, 30), (30, 70))
        ]

        for i, (start, end) in enumerate(routes):
            walker.generate_from_terrain(start_x=start[0], start_y=start[1])
            path = walker.backtrace_from_terrain(end_x=end[0], end_y=end[1])
            print(f"   Route {i + 1}: {start} -> {end}: {len(path)} Punkte")

    test_brownian_walk()


def performance_test():
    """Performance-Test für große Walks."""
    print("\n=== Performance Test ===")

    import time

    # Großer Walk testen
    sizes = [(100, 100, 50), (200, 200, 100), (300, 300, 150)]

    for W, H, T in sizes:
        start_time = time.time()

        with BrownianWalker(W=W, H=H, T=T) as walker:
            walker.generate(start_x=W // 2, start_y=H // 2)
            path = walker.backtrace(end_x=W // 4, end_y=H // 4)

        duration = time.time() - start_time
        print(f"   Größe {W}x{W}, T={T}: {duration:.2f} Sekunden, {len(path)} Punkte")
