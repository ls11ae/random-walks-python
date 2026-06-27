from enum import Enum
import geopandas as gpd
from kernelcma import StateKernelFactory


class StateAnnotationMethod(str, Enum):
    HMM = "hmm"
    BCPA = "bcpa"


def feature_enum():
    return _hmmcma()[2]


def annotate_states(
        trajectory_collection,
        method=StateAnnotationMethod.HMM,
        features=None,
        num_states=3,
        penalty=10,
        plot_path=None,
):
    HMM, BCPA, _ = _hmmcma()
    if method == StateAnnotationMethod.BCPA or method == StateAnnotationMethod.BCPA.value:
        annotator = BCPA(features=features, penalty=penalty, num_clusters=num_states)
    else:
        annotator = HMM(features=features, num_states=num_states)
    result = annotator.annotate(trajectory_collection)
    if plot_path:
        annotator.plot(plot_path)
    return result


def state_kernels(trajectory_collection, state_col="state", dt_tolerance=1.2, rnge=1000, out=None):
    gdf = trajectory_collection.to_point_gdf()
    time_col = trajectory_collection.t or gdf.index.name
    if {"utm_x", "utm_y"}.issubset(gdf.columns):
        crs = gdf["utm_crs"].dropna().iloc[0] if "utm_crs" in gdf and gdf["utm_crs"].notna().any() else None
        gdf = gpd.GeoDataFrame(gdf, geometry=gpd.points_from_xy(gdf["utm_x"], gdf["utm_y"]), crs=crs)

    factory = StateKernelFactory(
        gdf,
        id_col=trajectory_collection.get_traj_id_col(),
        time_col=time_col,
        state_col=state_col,
    )
    return factory.get_state_kernels(dt_tolerance, rnge, 2 * rnge + 1, out)


def _hmmcma():
    from hmmcma import BCPA, Feature, HMM

    return HMM, BCPA, Feature


__all__ = ["StateAnnotationMethod", "annotate_states", "feature_enum", "state_kernels"]
