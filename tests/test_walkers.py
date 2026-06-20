import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from movingpandas import TrajectoryCollection

from randomwalks.bindings.data_structures import KernelMapping
from randomwalks.bindings.walk_visualization import (
    LeafletGridOverlay,
    LeafletTiles,
    save_trajectory_coll_leaflet,
    walk_to_osm,
)
from randomwalks.core import MovementPolicy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from randomwalks import (
    BrownianWalker,
    CorrelatedWalker,
    MesaLandcover,
    MixedWalker,
    TerrainMapHandle,
    plot_terrain_walk, Reachability, TimeStepPolicy,
)
from randomwalks.core.MixedWalker import _normalize_walk_segment


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

    try:
        ud = MixedWalker.generate_custom_utilization_distribution(terrain, steps=steps, T=5)
        assert ud.shape == (20, 20)
        assert np.isfinite(ud).all()
        assert ud.sum() > 0
        assert ud[steps[0][1], steps[0][0]] > 0
    finally:
        terrain.free()


def test_mixed_walker_failed_backtrace_falls_back_to_endpoints():
    start = (2, 2)
    end = (6, 6)

    assert _normalize_walk_segment(None, start, end) == [start, end]
    assert _normalize_walk_segment(np.array(end), start, end) == [start, end]
    assert _normalize_walk_segment(np.array([end]), start, end) == [start, end]


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


def test_plot_terrain_walk_can_save_without_showing(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    out_file = tmp_path / "walk_plot.png"
    ax = plot_terrain_walk(
        terrain=np.full((5, 5), MesaLandcover.GRASSLAND),
        walk=[[(0, 0), (2, 2)], [(0, 4), (4, 0)]],
        steps=[(0, 0), (4, 4)],
        ud=np.ones((5, 5)),
        show=False,
        save_path=out_file,
    )

    assert ax is not None
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_walk_to_osm_makes_walk_checkable_layer(tmp_path):
    out_file = walk_to_osm(
        [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)],
        animal_id="animal",
        walk_path=str(tmp_path),
        utilization_distribution_overlays={
            "animal": LeafletGridOverlay(
                grid=np.ones((3, 3)),
                bounds=(0.0, 0.0, 1.0, 1.0),
                name="animal UD",
            )
        },
    )
    html = Path(out_file).read_text()

    assert "animal walk" in html
    assert "animal UD" in html
    assert "layer_control" in html


def test_movebank_walks():
    data = pd.read_csv(ROOT / "tests" / "data" / "boar_study_austria.csv")
    movement_policy = TimeStepPolicy(timestep_s=3600 * 4)
    mapping = KernelMapping.mesa_default()
    mapping.set_barrier(MesaLandcover.BUILT_UP)
    mapping.set_barrier(MesaLandcover.PERMANENT_WATER)
    print(f"{mapping.weight(MesaLandcover.GRASSLAND, MesaLandcover.CROPLAND)}")
    with MixedWalker(data=data,
                     resolution=200,
                     reachability=Reachability.RELAXED,
                     out_directory=ROOT / "tests" / "data" / "movebank_output",
                     movement_policy=movement_policy) as walker:
        traj_col = walker.generate_utilization_distribution(sample_walks=3, save_plots=True)
        pickle_path = ROOT / "tests" / "data" / "movebank_output" / "trajectories.pickle"
        pickle.dump(traj_col, open(pickle_path, "wb"))
        save_trajectory_coll_leaflet(traj_col, save_path=ROOT / "tests" / "data" / "movebank_output")


def test_leaflet_map():
    traj_col: TrajectoryCollection = pickle.load(
        open(ROOT / "tests" / "data" / "movebank_output" / "trajectories.pickle", "rb"))
    print(traj_col.to_point_gdf().head(20))
    save_trajectory_coll_leaflet(traj_col, save_path=ROOT / "tests" / "data" / "movebank_output",
                                 terrain_overlays=True,
                                 terrain_opacity=0.4,
                                 tiles=LeafletTiles.CARTODB_POSITRON_NO_LABELS)


test_movebank_walks()
test_leaflet_map()
