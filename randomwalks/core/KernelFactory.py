from enum import Enum
from math import isfinite

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
    if hasattr(method, "create") and hasattr(method, "method"):
        annotator = method.create()
    elif method == StateAnnotationMethod.BCPA or method == StateAnnotationMethod.BCPA.value:
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
        reso=None,
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
        dt_model_s=None,
        time_factor=None,
        is_brownian=False,
):
    """Fit state kernels at an exact duration or a native-time divisor.

    ``dt_model_s`` specifies seconds directly. ``time_factor`` divides the
    inferred native sampling interval; for example, a factor of three maps a
    15-minute interval to a five-minute kernel. Both kernel families are
    returned for compatibility and ``is_brownian`` selects the downstream one.
    """
    return _state_kernels(
        trajectory_collection,
        state_col=state_col,
        dt_tolerance=dt_tolerance,
        rnge=rnge,
        reso=reso,
        out=out,
        dt_model_s=dt_model_s,
        mass_percentile=mass_percentile,
        density_config=density_config,
        density_preset=density_preset,
        density_method=density_method,
        density_model=density_model,
        n_components=n_components,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        reg_covariance=reg_covariance,
        time_factor=time_factor,
        is_brownian=is_brownian,
    )


def _state_kernels(
        trajectory_collection,
        state_col="state",
        dt_tolerance=1.2,
        rnge=1000,
        reso=None,
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
        dt_model_s=None,
        time_factor=None,
        is_brownian=False,
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
    factory.build_trajectories()
    native_dt_s = float(factory.dt_threshold) * 60.0
    if dt_model_s is not None and time_factor is not None:
        raise ValueError("Configure either dt_model_s or time_factor, not both.")
    if time_factor is not None:
        factor = float(time_factor)
        if not isfinite(factor) or factor <= 0:
            raise ValueError("time_factor must be positive.")
        model_dt_s = native_dt_s / factor
    elif dt_model_s is not None:
        model_dt_s = float(dt_model_s)
        if not isfinite(model_dt_s) or model_dt_s <= 0:
            raise ValueError("dt_model_s must be positive.")
    else:
        model_dt_s = native_dt_s
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
    correlated, brownian = factory.get_state_kernels(
        dt_tolerance=dt_tolerance,
        rnge=rnge,
        reso=int(2 * rnge + 1) if reso is None else int(reso),
        out=out,
        density_config=density_options,
        dt_model_s=model_dt_s,
        brownian_dt=model_dt_s / 60.0,
        mass_percentile=mass_percentile,
    )
    return correlated, brownian


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
