from __future__ import annotations

import ctypes
import json
from pathlib import Path

import numpy as np

from randomwalks.bindings.data_structures.Kernels import normalize_kernel
from randomwalks.wrapper import dll


_SSF_BINDINGS_CONFIGURED = False


def segments_for_steps(
        steps,
        *,
        max_cell_size: int,
        resolution: int,
        criterion=None,
        segment_col="segment_id",
        merge_single_point_segments=True,
):
    from segmentationcma import UTMDistanceCriterion, annotate_segments_dataframe, make_overlapping, segment_dataframe

    if criterion is None:
        criterion = UTMDistanceCriterion.from_cell_grid(max_cell_size, resolution)

    base_segments = segment_dataframe(
        steps,
        criterion,
        merge_single_point_segments=merge_single_point_segments,
    )
    annotated = annotate_segments_dataframe(steps, segments=base_segments, segment_col=segment_col)
    steps[segment_col] = annotated[segment_col]
    return make_overlapping(base_segments)


def terrain_pair_weights_from_neighborhoods(
        neighborhoods,
        kernels,
        *,
        out_dir,
        terrain_values=None,
        exclude_terrain_values=(0,),
        state_values=None,
        lambda_=1.0,
        log_clip=None,
        lo=0.5,
        hi=1.5,
        count_self_transitions=True,
        save_heatmaps=True,
        verbose=True,
):
    """
    Estimate state-specific terrain-pair weights from saved terrain neighborhoods.

    ``neighborhoods`` is the metadata list returned by
    ``StateDependentWalker.save_kernel_neighborhoods``. Entries must contain a
    ``matrix_path`` and the observed step offset fields ``obs_dx``/``obs_dy``.
    ``kernels`` maps state values to 2D numpy kernels.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _ensure_ssf_bindings()
    if log_clip is not None:
        log_clip = abs(float(log_clip))
        lo = float(np.exp(-log_clip))
        hi = float(np.exp(log_clip))

    records = [_coerce_neighborhood_record(record) for record in neighborhoods]
    records = [
        record for record in records
        if record["matrix"] is not None and record["obs_dx"] is not None and record["obs_dy"] is not None
    ]
    if not records:
        raise ValueError("No neighborhoods with matrix data and observed offsets were provided.")

    if verbose:
        print(f"[SSF/Python] Preparing {len(records)} terrain neighborhoods.")

    excluded_terrain_values = {int(value) for value in (exclude_terrain_values or [])}
    terrain_values = _resolve_terrain_values(records, terrain_values, excluded_terrain_values)
    terrain_to_class = {int(value): index for index, value in enumerate(terrain_values)}
    L = len(terrain_values)
    if L == 0:
        raise ValueError("No terrain classes are available.")

    state_values = _resolve_state_values(records, kernels, state_values)
    state_to_index = {state: index for index, state in enumerate(state_values)}
    S = len(state_values)
    if S == 0:
        raise ValueError("No states are available.")

    used = np.zeros((S, L, L), dtype=np.float64)
    available = np.zeros((S, L, L), dtype=np.float64)

    processed = 0
    skipped = 0
    for state in state_values:
        state_records = [record for record in records if _state_key_equal(record["state"], state)]
        if not state_records:
            continue
        kernel = _kernel_for_state(kernels, state)
        if kernel is None:
            skipped += len(state_records)
            continue
        kernel = np.ascontiguousarray(normalize_kernel(kernel), dtype=np.float64)

        for radius, batch in _group_records_by_radius(state_records).items():
            packed = _pack_neighborhood_batch(batch, terrain_to_class)
            if packed is None:
                skipped += len(batch)
                continue
            terrain_stack, obs_dx, obs_dy, sample_weights = packed
            if terrain_stack.shape[0] == 0:
                skipped += len(batch)
                continue

            if verbose:
                print(
                    f"[SSF/Python] State {state}: sending {terrain_stack.shape[0]} "
                    f"neighborhoods (R={radius}) to C."
                )

            state_index = state_to_index[state]
            ok = dll.ssf_process_flat_neighborhoods(
                terrain_stack.shape[0],
                L,
                int(radius),
                obs_dx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                obs_dy.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                sample_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                terrain_stack.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                kernel.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                kernel.shape[1],
                kernel.shape[0],
                bool(count_self_transitions),
                used[state_index].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                available[state_index].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
            if not ok:
                raise RuntimeError("C SSF terrain-pair processing failed.")
            processed += terrain_stack.shape[0]

    weights = np.zeros((S, L, L), dtype=np.float64)
    dll.ssf_compute_weights(
        S,
        L,
        used.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        available.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        float(lambda_),
        float(lo),
        float(hi),
    )

    if verbose:
        print(f"[SSF/Python] Processed {processed} neighborhoods; skipped {skipped}.")

    used_csv = out_dir / "terrain_pair_used.csv"
    available_csv = out_dir / "terrain_pair_available.csv"
    weights_csv = out_dir / "terrain_pair_weights.csv"
    _write_pair_csv(used_csv, state_values, terrain_values, used, "used")
    _write_pair_csv(available_csv, state_values, terrain_values, available, "available")
    _write_pair_csv(weights_csv, state_values, terrain_values, weights, "weight")

    heatmap_paths = []
    if save_heatmaps:
        heatmap_paths = _save_weight_heatmaps(out_dir, state_values, terrain_values, weights)

    return {
        "used": used,
        "available": available,
        "weights": weights,
        "state_values": state_values,
        "terrain_values": terrain_values,
        "excluded_terrain_values": sorted(excluded_terrain_values),
        "used_csv": used_csv,
        "available_csv": available_csv,
        "weights_csv": weights_csv,
        "heatmaps": heatmap_paths,
        "processed": processed,
        "skipped": skipped,
    }


def _ensure_ssf_bindings():
    global _SSF_BINDINGS_CONFIGURED
    if _SSF_BINDINGS_CONFIGURED:
        return
    try:
        process = dll.ssf_process_flat_neighborhoods
        compute = dll.ssf_compute_weights
    except AttributeError as exc:
        raise RuntimeError(
            "The loaded librandom_walk.so does not expose the SSF terrain-pair "
            "adapter symbols. Rebuild/reinstall the package before calling "
            "terrain_pair_weights_from_neighborhoods()."
        ) from exc

    process.argtypes = [
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_bool,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    process.restype = ctypes.c_int

    compute.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
    ]
    compute.restype = None
    _SSF_BINDINGS_CONFIGURED = True


def _coerce_neighborhood_record(record):
    if isinstance(record, (str, Path)):
        path = Path(record)
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("matrix_path") is None:
            npy_path = path.with_suffix(".npy")
            if npy_path.exists():
                record["matrix_path"] = str(npy_path)
    else:
        record = dict(record)

    matrix = record.get("matrix")
    if matrix is None and record.get("matrix_path"):
        matrix_path = Path(record["matrix_path"])
        if matrix_path.exists():
            matrix = np.load(matrix_path)

    record["matrix"] = None if matrix is None else np.asarray(matrix)
    record["state"] = record.get("state")
    record["obs_dx"] = record.get("obs_dx")
    record["obs_dy"] = record.get("obs_dy")
    record["weight"] = float(record.get("weight", 1.0))
    return record


def _resolve_terrain_values(records, terrain_values, excluded_terrain_values):
    if terrain_values is not None:
        return [
            int(value)
            for value in terrain_values
            if int(value) not in excluded_terrain_values
        ]
    values = set()
    for record in records:
        matrix = record["matrix"]
        if matrix is not None:
            values.update(int(value) for value in np.unique(matrix))
    return sorted(value for value in values if value not in excluded_terrain_values)


def _resolve_state_values(records, kernels, state_values):
    if state_values is not None:
        return list(state_values)
    values = []
    for key in kernels.keys():
        if not any(_state_key_equal(key, value) for value in values):
            values.append(key)
    for record in records:
        state = record["state"]
        if not any(_state_key_equal(state, value) for value in values):
            values.append(state)
    return values


def _kernel_for_state(kernels, state):
    if state in kernels:
        return kernels[state]
    for key, kernel in kernels.items():
        if _state_key_equal(key, state):
            return kernel
    return None


def _state_key_equal(left, right):
    if left == right:
        return True
    try:
        return int(left) == int(right)
    except (TypeError, ValueError):
        return str(left) == str(right)


def _group_records_by_radius(records):
    grouped = {}
    for record in records:
        matrix = np.asarray(record["matrix"])
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] % 2 == 0:
            continue
        radius = matrix.shape[0] // 2
        grouped.setdefault(radius, []).append(record)
    return grouped


def _pack_neighborhood_batch(records, terrain_to_class):
    terrain_stack = []
    obs_dx = []
    obs_dy = []
    sample_weights = []
    for record in records:
        matrix = np.asarray(record["matrix"])
        mapped = np.full(matrix.shape, -1, dtype=np.int32)
        known_mask = np.zeros(matrix.shape, dtype=bool)
        for terrain_value, class_index in terrain_to_class.items():
            mask = matrix == terrain_value
            mapped[mask] = class_index
            known_mask |= mask

        present = set(int(value) for value in mapped[known_mask])
        if len(present) < 2:
            continue

        terrain_stack.append(mapped)
        obs_dx.append(int(record["obs_dx"]))
        obs_dy.append(int(record["obs_dy"]))
        sample_weights.append(float(record.get("weight", 1.0)))

    if not terrain_stack:
        return None

    return (
        np.ascontiguousarray(np.stack(terrain_stack), dtype=np.int32),
        np.ascontiguousarray(obs_dx, dtype=np.int32),
        np.ascontiguousarray(obs_dy, dtype=np.int32),
        np.ascontiguousarray(sample_weights, dtype=np.float64),
    )


def _write_pair_csv(path, state_values, terrain_values, values, value_col):
    import pandas as pd

    rows = []
    for state_index, state in enumerate(state_values):
        for a_index, terrain_a in enumerate(terrain_values):
            for b_index, terrain_b in enumerate(terrain_values):
                rows.append({
                    "state": state,
                    "from_terrain": terrain_a,
                    "to_terrain": terrain_b,
                    value_col: values[state_index, a_index, b_index],
                })
    pd.DataFrame(rows).to_csv(path, index=False)


def _save_weight_heatmaps(out_dir, state_values, terrain_values, weights):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    labels = [_terrain_label(value) for value in terrain_values]
    paths = []
    for state_index, state in enumerate(state_values):
        side = max(4, min(12, len(terrain_values) * 0.9))
        fig, ax = plt.subplots(figsize=(side, side))
        state_weights = weights[state_index]
        finite = state_weights[np.isfinite(state_weights)]
        if finite.size:
            vmin = float(np.min(finite))
            vmax = float(np.max(finite))
        else:
            vmin, vmax = 0.5, 1.5
        if not vmin < 1.0 < vmax:
            span = max(abs(vmin - 1.0), abs(vmax - 1.0), 0.01)
            vmin = 1.0 - span
            vmax = 1.0 + span
        norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
        im = ax.imshow(state_weights, cmap="RdBu_r", norm=norm)
        ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels=labels)
        ax.set_xlabel("To terrain")
        ax.set_ylabel("From terrain")
        ax.set_title(f"Terrain-pair weights, state {state}")
        for y in range(state_weights.shape[0]):
            for x in range(state_weights.shape[1]):
                value = state_weights[y, x]
                color = "white" if abs(value - 1.0) > 0.28 else "black"
                ax.text(x, y, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)
        colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("Weight (1 = neutral)")
        fig.tight_layout()
        path = out_dir / f"terrain_pair_weights_state_{_safe_filename(state)}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def _terrain_label(value):
    try:
        from randomwalks.bindings.data_structures.Terrain import MESA_LANDCOVER_LABELS

        return MESA_LANDCOVER_LABELS.get(int(value), str(value))
    except Exception:
        return str(value)


def _safe_filename(value):
    safe = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in str(value)
    ).strip("_")
    return safe or "value"
