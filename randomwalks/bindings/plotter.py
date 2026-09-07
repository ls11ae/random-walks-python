import numpy as np
from pathlib import Path

from randomwalks.bindings.data_structures.Terrain import (
    MESA_LANDCOVER_COLORS,
    MESA_LANDCOVER_LABELS,
    TerrainMapHandle,
)


def ud_isopleth_mask(ud, p=1):
    ud = np.asarray(ud, dtype=float)
    ud = np.clip(ud, 0, None)
    total = ud.sum()
    if total <= 0:
        return np.zeros_like(ud, dtype=bool), np.nan

    normalized = ud / total
    flat = normalized.ravel()
    order = np.argsort(flat)[::-1]
    csum = np.cumsum(flat[order])
    index = min(int(np.searchsorted(csum, p, side="left")), flat.size - 1)
    level = flat[order[index]]
    return normalized >= level, level


def ud_isopleth_band_map(ud, step=5, max_p=100):
    ud = np.asarray(ud, dtype=float)
    ud = np.clip(ud, 0, None)
    total = ud.sum()
    if total <= 0:
        return np.full_like(ud, np.nan, dtype=float)

    flat = (ud / total).ravel()
    order = np.argsort(flat)[::-1]
    csum = np.cumsum(flat[order]) * 100.0
    band_edges = np.arange(step, max_p + step, step)
    band_index = np.searchsorted(band_edges, csum, side="left")

    band_flat = np.full(flat.shape, np.nan, dtype=float)
    valid = band_index < len(band_edges)
    band_flat[order[valid]] = band_edges[band_index[valid]]
    return band_flat.reshape(ud.shape)


def plot_terrain_walk(
    *,
    terrain=None,
    walk=None,
    steps=None,
    ud=None,
    ud_alpha=0.45,
    width=None,
    height=None,
    title=None,
    show_legend=True,
    show=True,
    save_path=None,
    dpi=150,
    ax=None,
):
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    terrain_array = _terrain_array(terrain)
    if terrain_array is not None:
        height, width = terrain_array.shape
    elif width is None or height is None:
        raise ValueError("Either terrain or width/height is required")

    created_figure = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure

    legend_handles = []
    if terrain_array is not None:
        cmap, norm, terrain_handles = _terrain_cmap_and_legend(terrain_array, mcolors, Patch)
        ax.imshow(
            terrain_array,
            cmap=cmap,
            norm=norm,
            origin="lower",
            interpolation="nearest",
            extent=(-0.5, width - 0.5, -0.5, height - 0.5),
            zorder=0,
        )
        legend_handles.extend(terrain_handles)

    if ud is not None:
        band_map = ud_isopleth_band_map(ud)
        levels = np.arange(5, 100, 5)
        gradient = mcolors.LinearSegmentedColormap.from_list(
            "ud_isopleths",
            ["#FFFF99", "#FF0000", "#8B0000"],
        )
        colors = [mcolors.to_hex(gradient(1 - i / (len(levels) - 1))) for i in range(len(levels))]
        cmap = mcolors.ListedColormap(colors)
        cmap.set_bad((0, 0, 0, 0))
        norm = mcolors.BoundaryNorm(np.arange(2.5, 100, 5), cmap.N)
        ax.imshow(
            band_map,
            cmap=cmap,
            norm=norm,
            origin="lower",
            alpha=float(ud_alpha),
            interpolation="nearest",
            extent=(-0.5, width - 0.5, -0.5, height - 0.5),
            zorder=5,
        )

    walk_arrays = _walk_arrays(walk)
    if walk_arrays:
        for walk_array in walk_arrays:
            ax.plot(walk_array[:, 0], walk_array[:, 1], color="#B23A2E", linewidth=2.0, alpha=0.78, zorder=10)
            ax.scatter(walk_array[0, 0], walk_array[0, 1], color="blue", edgecolor="black", s=50, zorder=11)
            ax.scatter(walk_array[-1, 0], walk_array[-1, 1], color="black", edgecolor="black", s=50, zorder=11)
        legend_handles.extend([
            Line2D([0], [0], color="#B23A2E", linewidth=2, label="Path"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="blue",
                   markeredgecolor="black", markersize=8, label="Start"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
                   markeredgecolor="black", markersize=8, label="End"),
        ])

    if steps is not None:
        steps = np.asarray(steps, dtype=float)
        if steps.ndim == 1:
            steps = steps.reshape(1, 2)
        for index, (x, y) in enumerate(steps):
            ax.annotate(
                str(index),
                xy=(x, y),
                color="black",
                fontsize=9,
                ha="center",
                va="center",
                zorder=12,
                bbox={
                    "boxstyle": "square,pad=0.25",
                    "facecolor": "#D08A00",
                    "edgecolor": "black",
                    "linewidth": 1.0,
                },
            )

    ax.set_title("" if title is None else title)
    ax.set_xlim(-1, width)
    ax.set_ylim(height, -1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.35)
    ax.tick_params(axis="both", which="both", labelsize=7, colors="gray", length=2, width=0.5)
    for spine in ax.spines.values():
        spine.set_color("lightgray")
        spine.set_linewidth(0.5)

    if show_legend and legend_handles:
        labels_seen = set()
        unique_handles = []
        for handle in legend_handles:
            label = handle.get_label()
            if label not in labels_seen:
                unique_handles.append(handle)
                labels_seen.add(label)
        ax.legend(handles=unique_handles, loc="upper right", fontsize=9)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        fig.tight_layout()
        plt.show()
    elif save_path is not None and created_figure:
        plt.close(fig)
    return ax


def plot_walk_from_json(json_file, *, title=None, show=True, ax=None, show_legend=True):
    from randomwalks.serialization import plot_walk_from_json as _plot_walk_from_json

    return _plot_walk_from_json(
        json_file,
        title=title,
        show=show,
        ax=ax,
        show_legend=show_legend,
    )


def _terrain_array(terrain):
    if terrain is None:
        return None
    if isinstance(terrain, TerrainMapHandle):
        return terrain.to_numpy()
    if hasattr(terrain, "to_numpy"):
        return terrain.to_numpy()
    return np.asarray(terrain, dtype=int)


def _walk_arrays(walk):
    if walk is None:
        return []

    try:
        array = np.asarray(walk, dtype=float)
    except (TypeError, ValueError):
        arrays = []
        for item in walk:
            item_array = _single_walk_array(item)
            if item_array is not None:
                arrays.append(item_array)
        return arrays

    if array.dtype == object:
        arrays = []
        for item in walk:
            item_array = _single_walk_array(item)
            if item_array is not None:
                arrays.append(item_array)
        return arrays

    if array.ndim == 3:
        return [item for item in (_single_walk_array(path) for path in array) if item is not None]

    single = _single_walk_array(array)
    return [] if single is None else [single]


def _single_walk_array(walk):
    array = np.asarray(walk, dtype=float)
    if array.ndim == 1:
        if array.size != 2:
            return None
        array = array.reshape(1, 2)
    if array.ndim != 2 or array.shape[1] != 2 or array.size == 0:
        return None
    return array


def _terrain_cmap_and_legend(terrain_array, mcolors, Patch):
    present = sorted(int(value) for value in np.unique(terrain_array))
    colors = []
    handles = []
    for value in present:
        color = MESA_LANDCOVER_COLORS.get(value, (0.6, 0.6, 0.6, 0.75))
        label = MESA_LANDCOVER_LABELS.get(value, f"Class {value}")
        colors.append(color)
        handles.append(Patch(facecolor=color, edgecolor="black", label=label))

    if not present:
        return mcolors.ListedColormap([(0, 0, 0, 0)]), mcolors.BoundaryNorm([0, 1], 1), handles

    if len(present) == 1:
        bounds = [present[0] - 0.5, present[0] + 0.5]
    else:
        bounds = [present[0] - 0.5]
        bounds.extend((left + right) / 2.0 for left, right in zip(present, present[1:]))
        bounds.append(present[-1] + 0.5)

    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm, handles
