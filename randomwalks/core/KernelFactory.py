from enum import Enum
from math import isclose

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
        is_brownian=False,
):
    """Fit state kernels while keeping correlated kernels at native time.

    Both kernel families are returned for compatibility. ``is_brownian`` only
    permits ``dt_model_s`` to configure the Brownian family; correlated
    displacements are never time-rescaled and receive the inferred native
    sampling interval as metadata.
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
    # kernelcma's ``dt_model_s`` option linearly rescales correlated training
    # displacements.  That is not a valid operation for a CRW kernel: its
    # duration is the native interval represented by the observed turns.  Fit
    # without that option and retain the inferred interval only as runtime
    # metadata.  Brownian kernels may still be fitted at an explicit duration.
    factory.build_trajectories()
    native_dt_s = float(factory.dt_threshold) * 60.0
    if not is_brownian and dt_model_s is not None and not isclose(
            float(dt_model_s), native_dt_s, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError(
            "Correlated kernels must use the trajectory's native sampling interval "
            f"({native_dt_s:g} s); got dt_model_s={float(dt_model_s):g} s. "
            "Omit dt_model_s to use the inferred interval."
        )
    brownian_dt_s = (
        float(dt_model_s)
        if is_brownian and dt_model_s is not None
        else native_dt_s
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
    correlated, brownian = factory.get_state_kernels(
        dt_tolerance=dt_tolerance,
        rnge=rnge,
        reso=int(2 * rnge + 1) if reso is None else int(reso),
        out=out,
        density_config=density_options,
        # Never pass dt_model_s here: doing so also rescales correlated steps.
        dt_model_s=None,
        brownian_dt=brownian_dt_s / 60.0,
        mass_percentile=mass_percentile,
    )
    for kernel in correlated:
        kernel.dt_model_s = native_dt_s
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
