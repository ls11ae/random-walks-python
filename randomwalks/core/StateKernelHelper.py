import numpy as np
import pandas as pd

from randomwalks.bindings.data_structures.Kernels import kernel_array
from randomwalks.core.StateWalkerConfig import UnmodelledStatePolicy


def _is_unmodelled_state(state):
    if pd.isna(state):
        return True
    try:
        return float(state) < 0
    except (TypeError, ValueError):
        return False


def _state_for_step(steps, step_index, *, state_col, kernels, policy, max_fill_gap):
    start_state = steps[state_col].iloc[step_index]
    end_state = steps[state_col].iloc[step_index + 1]
    if _state_has_kernel(start_state, kernels) and _state_has_kernel(end_state, kernels):
        return start_state
    if policy is UnmodelledStatePolicy.SKIP:
        return None

    for previous_index in range(step_index, -1, -1):
        previous_state = steps[state_col].iloc[previous_index]
        if not _state_has_kernel(previous_state, kernels):
            continue
        elapsed = pd.Timestamp(steps.index[step_index + 1]) - pd.Timestamp(steps.index[previous_index])
        return previous_state if elapsed <= max_fill_gap else None
    return None


def _state_has_kernel(state, kernels):
    return not _is_unmodelled_state(state) and _kernel_for_state(kernels, state) is not None


def _kernel_for_state(kernels, state):
    if state in kernels:
        return kernels[state]
    for key, kernel in kernels.items():
        if _state_key_equal(key, state):
            return kernel
    return None


def _kernel_metadata_for_state(metadata_by_state, state):
    if not metadata_by_state:
        return None
    if state in metadata_by_state:
        return metadata_by_state[state]
    for key, metadata in metadata_by_state.items():
        if _state_key_equal(key, state):
            return metadata
    return None


def _state_key_equal(left, right):
    if left == right:
        return True
    try:
        return int(left) == int(right)
    except (TypeError, ValueError):
        return str(left) == str(right)


def _kernel_and_metadata(state_kernel, fallback_radius, mass_percentile):
    kernel = kernel_array(state_kernel)
    metadata = {
        "rnge": getattr(state_kernel, "rnge", None),
        "reso": getattr(state_kernel, "reso", None),
        "dx": getattr(state_kernel, "dx", None),
        "radius_cells": getattr(state_kernel, "radius_cells", None),
        "retained_mass": getattr(state_kernel, "retained_mass", None),
        "mass_percentile": getattr(state_kernel, "mass_percentile", None),
        "dt_model_s": getattr(state_kernel, "dt_model_s", None),
    }
    if _kernel_is_clipped_to_mass(kernel, metadata, mass_percentile):
        return kernel, metadata

    from kernelcma.postprocessing import clip_density_to_mass

    source_radius = _kernel_source_radius(kernel, metadata, fallback_radius)
    clipped = clip_density_to_mass(kernel, source_radius, mass_percentile=mass_percentile)
    return clipped.Z, {
        "rnge": clipped.rnge,
        "reso": clipped.reso,
        "dx": clipped.dx,
        "radius_cells": clipped.radius_cells,
        "retained_mass": clipped.retained_mass,
        "mass_percentile": clipped.mass_percentile,
        "dt_model_s": metadata["dt_model_s"],
    }


def _kernel_is_clipped_to_mass(kernel, metadata, mass_percentile):
    kernel = np.asarray(kernel) if kernel is not None else None
    if kernel is None or kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        return False
    radius_cells = metadata.get("radius_cells")
    actual_percentile = metadata.get("mass_percentile")
    if radius_cells is None or actual_percentile is None:
        return False
    try:
        same_percentile = np.isclose(float(actual_percentile), float(mass_percentile))
        expected_size = 2 * int(radius_cells) + 1
    except (TypeError, ValueError):
        return False
    return bool(same_percentile and kernel.shape == (expected_size, expected_size))


def _kernel_source_radius(kernel, metadata, fallback_radius):
    kernel = np.asarray(kernel)
    dx = metadata.get("dx")
    if dx is not None:
        try:
            dx = float(dx)
            if np.isfinite(dx) and dx > 0:
                return kernel.shape[0] * dx / 2.0
        except (TypeError, ValueError):
            pass
    return float(fallback_radius)


def _kernel_radius_m(kernel, *, metadata=None, fallback_radius=None, mass_percentile=0.99):
    if metadata and metadata.get("rnge") is not None:
        return float(metadata["rnge"])

    if fallback_radius is not None:
        try:
            from kernelcma.postprocessing import clip_density_to_mass

            return float(
                clip_density_to_mass(
                    kernel,
                    fallback_radius,
                    mass_percentile=mass_percentile,
                ).rnge
            )
        except Exception:
            pass

    kernel = np.asarray(kernel)
    if kernel.ndim >= 2 and min(kernel.shape[-2:]) > 1:
        return float(min(kernel.shape[-2], kernel.shape[-1]) // 2)
    return float(fallback_radius or 0)


def _terrain_values(matrix, nodata):
    values = np.unique(matrix)
    result = []
    for value in values:
        if nodata is not None and value == nodata:
            continue
        result.append(_json_scalar(value))
    return result


def _json_scalar(value):
    if hasattr(value, "item"):
        return value.item()
    return value
