import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from randomwalks import (
    ComputationMode,
    CorrelatedWalker,
    MesaLandcover,
    MixedWalker,
    TerrainMapHandle,
    plot_walk_from_json,
    walk_from_json,
    walk_to_json,
)


def test_walk_json_round_trip_includes_terrain_steps_and_ud(tmp_path):
    walk = np.array([(0, 0), (1, 1), (2, 1)], dtype=np.int64)
    steps = [(0, 0), (2, 1)]
    terrain = np.array([
        [MesaLandcover.GRASSLAND, MesaLandcover.PERMANENT_WATER, MesaLandcover.GRASSLAND],
        [MesaLandcover.GRASSLAND, MesaLandcover.BUILT_UP, MesaLandcover.GRASSLAND],
    ])
    ud = np.ones((2, 3), dtype=float)

    json_path = walk_to_json(walk, tmp_path / "walk.json", steps=steps, terrain=terrain, ud=ud)
    loaded = walk_from_json(json_path)

    assert loaded.width == 3
    assert loaded.height == 2
    np.testing.assert_array_equal(loaded.walk, walk)
    np.testing.assert_array_equal(loaded.steps, np.asarray(steps, dtype=np.int64))
    np.testing.assert_array_equal(loaded.terrain, terrain.astype(int))
    np.testing.assert_array_equal(loaded.utilization_distribution, ud)
    assert loaded.start == (0, 0)
    assert loaded.end == (2, 1)


def test_plot_walk_from_json_only_lists_present_landcovers(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    terrain = np.array([
        [MesaLandcover.GRASSLAND, MesaLandcover.PERMANENT_WATER],
        [MesaLandcover.GRASSLAND, MesaLandcover.PERMANENT_WATER],
    ])
    json_path = walk_to_json([(0, 0), (1, 1)], tmp_path / "walk.json", terrain=terrain)

    ax = plot_walk_from_json(json_path, show=False)
    labels = [text.get_text() for text in ax.get_legend().get_texts()]

    assert "Grassland" in labels
    assert "Permanent water" in labels
    assert "Tree cover" not in labels


def test_correlated_dp_serialization_with_explicit_directory(tmp_path):
    start = (2, 2)
    end = (6, 6)
    dp_dir = tmp_path / "dp"

    with CorrelatedWalker(W=15, H=15, T=5, S=2, D=4) as walker:
        walker.generate(start, serialization_dir=dp_dir)
        walk = walker.backtrace(end)

    assert (dp_dir / "step_0").exists()
    assert walk.shape[1] == 2
    assert tuple(walk[0]) == start
    assert tuple(walk[-1]) == end


def test_correlated_dp_temp_serialization_directory_is_cleaned():
    walker = CorrelatedWalker(W=15, H=15, T=5, S=2, D=4)
    dp_folder = None
    try:
        walker.generate((2, 2), use_serialization=True)
        dp_folder = Path(walker.dp_folder)
        assert dp_folder.exists()
        walker.backtrace((6, 6))
    finally:
        walker.close()

    assert not dp_folder.exists()


def test_mixed_custom_walk_serialization_with_explicit_directory(tmp_path):
    terrain = TerrainMapHandle.single_value(MesaLandcover.GRASSLAND, 5, 5)
    context_dir = tmp_path / "mixed_context"

    try:
        walk = MixedWalker.generate_custom_walks(
            terrain,
            [(1, 1), (3, 3)],
            3,
            context=ComputationMode.SERIALIZATION,
            serialization_dir=context_dir,
        )
        assert (context_dir / "kernel_pool").exists()
        assert walk.shape[1] == 2
    finally:
        terrain.free()
