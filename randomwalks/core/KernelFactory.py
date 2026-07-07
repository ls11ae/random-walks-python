from enum import Enum

import geopandas as gpd
from kernelcma import StateKernelFactory


class StateAnnotationMethod(str, Enum):
    HMM = "hmm"
    BCPA = "bcpa"
    BCPAHMM = "bcpahmm"


def feature_enum():
    return _hmmcma()[3]


def annotate_states(
        trajectory_collection,
        method=StateAnnotationMethod.HMM,
        features=None,
        num_states=3,
        penalty=10,
        plot_path=None,
):
    HMM, BCPA, BCPAHMM, _ = _hmmcma()
    if method == StateAnnotationMethod.BCPA or method == StateAnnotationMethod.BCPA.value:
        annotator = BCPA(features=features, penalty=penalty, num_clusters=num_states)
    elif method == StateAnnotationMethod.BCPAHMM or method == StateAnnotationMethod.BCPAHMM.value:
        annotator = BCPAHMM(features=features, penalty=penalty, num_states=num_states)
    else:
        annotator = HMM(features=features, num_states=num_states)
    result = annotator.annotate(trajectory_collection)
    result.trajectory_collection
    if plot_path:
        annotator.plot(plot_path)
    return result


def state_kernels(
        trajectory_collection,
        state_col="state",
        dt_tolerance=1.2,
        rnge=1000,
        out=None,
        mass_percentile=0.99,
        density_config=None,
        density_preset=None,
        density_method=None,
        density_model=None,
        n_components=None,
        covariance_type=None,
        reg_covar=None,
        reg_covariance=None,
):
    return _state_kernels(
        trajectory_collection,
        state_col=state_col,
        dt_tolerance=dt_tolerance,
        rnge=rnge,
        out=out,
        mass_percentile=mass_percentile,
        density_config=density_config,
        density_preset=density_preset,
        density_method=density_method,
        density_model=density_model,
        n_components=n_components,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        reg_covariance=reg_covariance,
    )


def _state_kernels(
        trajectory_collection,
        state_col="state",
        dt_tolerance=1.2,
        rnge=1000,
        out=None,
        mass_percentile=0.99,
        density_config=None,
        density_preset=None,
        density_method=None,
        density_model=None,
        n_components=None,
        covariance_type=None,
        reg_covar=None,
        reg_covariance=None,
):
    gdf = trajectory_collection.to_point_gdf().copy()
    time_col = trajectory_collection.t or gdf.index.name
    if time_col not in gdf.columns:
        gdf[time_col] = gdf.index
    gdf = gdf.reset_index(drop=True)
    if {"utm_x", "utm_y"}.issubset(gdf.columns):
        crs = gdf["utm_crs"].dropna().iloc[0] if "utm_crs" in gdf and gdf["utm_crs"].notna().any() else None
        gdf = gpd.GeoDataFrame(gdf, geometry=gpd.points_from_xy(gdf["utm_x"], gdf["utm_y"]), crs=crs)

    factory = StateKernelFactory(
        gdf,
        id_col=trajectory_collection.get_traj_id_col(),
        time_col=time_col,
        state_col=state_col,
    )
    density_options = _density_options(
        density_config=density_config,
        density_preset=density_preset,
        density_method=density_method,
        density_model=density_model,
        n_components=n_components,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        reg_covariance=reg_covariance,
    )
    return factory.get_state_kernels(
        dt_tolerance=dt_tolerance,
        rnge=rnge,
        reso=2 * rnge + 1,
        out=out,
        density_config=density_options,
        mass_percentile=mass_percentile,
    )


def _density_options(
        *,
        density_config=None,
        density_preset=None,
        density_method=None,
        density_model=None,
        n_components=None,
        covariance_type=None,
        reg_covar=None,
        reg_covariance=None,
):
    if density_config is None:
        options = {}
    elif isinstance(density_config, dict):
        options = dict(density_config)
    elif isinstance(density_config, str):
        options = {"preset": density_config}
    else:
        options = {"model": density_config}

    overrides = {
        "preset": density_preset,
        "method": density_method,
        "model": density_model,
        "n_components": n_components,
        "covariance_type": covariance_type,
        "reg_covar": reg_covar,
        "reg_covariance": reg_covariance,
    }
    options.update({key: value for key, value in overrides.items() if value is not None})
    return options or None


def _hmmcma():
    from hmmcma import BCPA, BCPAHMM, Feature, HMM

    return HMM, BCPA, BCPAHMM, Feature


__all__ = ["StateAnnotationMethod", "annotate_states", "feature_enum", "state_kernels"]
