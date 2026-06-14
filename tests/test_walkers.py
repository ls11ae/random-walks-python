import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from randomwalks import (
    BrownianWalker,
    CorrelatedWalker,
    MesaLandcover,
    MixedWalker,
    TerrainMapHandle,
    plot_terrain_walk,
)


def _as_point_set(walk):
    return {tuple(map(int, point)) for point in np.asarray(walk)}


def test_brownian_walker_backtrace_contains_endpoints():
    start = (2, 2)
    end = (6, 6)

    with BrownianWalker(W=15, H=15, T=5, S=2) as walker:
        walk = walker.walk(start, end)

        assert walk.shape[1] == 2
        assert tuple(walk[0]) == start
        assert tuple(walk[-1]) == end
        assert {start, end}.issubset(_as_point_set(walk))


def test_correlated_walker_backtrace_contains_endpoints():
    start = (2, 2)
    end = (6, 6)

    with CorrelatedWalker(W=15, H=15, T=5, S=2, D=4) as walker:
        walk = walker.walk(start, end)

        assert walk.shape[1] == 2
        assert tuple(walk[0]) == start
        assert tuple(walk[-1]) == end
        assert {start, end}.issubset(_as_point_set(walk))


def test_mixed_walker_utilization_distribution_is_valid():
    steps = [(2, 2), (6, 6)]
    terrain = TerrainMapHandle.single_value(MesaLandcover.GRASSLAND, 20, 20)

    with MixedWalker(terrain, T=5) as walker:
        ud = walker.utilization_distribution(steps=steps, T=5)

        assert ud.shape == (20, 20)
        assert np.isfinite(ud).all()
        assert ud.sum() > 0
        assert ud[steps[0][1], steps[0][0]] > 0


def test_plot_terrain_walk_legend_only_contains_present_landcovers():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    terrain = TerrainMapHandle.single_value(MesaLandcover.GRASSLAND, 5, 5)
    try:
        ax = plot_terrain_walk(terrain=terrain, walk=[(0, 0), (2, 2)], show=False)
        legend = ax.get_legend()
        labels = [text.get_text() for text in legend.get_texts()]
    finally:
        terrain.free()

    assert "Grassland" in labels
    assert "Tree cover" not in labels
    assert "Permanent water" not in labels
