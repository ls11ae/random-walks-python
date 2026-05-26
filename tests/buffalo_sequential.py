import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time

from random_walk_package.bindings.data_structures.kernel_terrain_mapping import (
    create_brownian_kernel_parameters,
    set_landmark_mapping,
)
from random_walk_package.bindings.data_structures.terrain import *
from random_walk_package.core.MixedWalker import *

HAB_CSV = "tests/buf_data/buffalo_habitat.csv"
TRAJ_CSV = "tests/buf_data/buffalo_trajectory.csv"

# Your column names:
HAB_CLASS_COL = "habi.asc"
HAB_X_COL = "s1"
HAB_Y_COL = "s2"


def ud_isopleth_mask(ud, p=0.95):
    ud = np.asarray(ud, dtype=float)
    ud = np.clip(ud, 0, None)
    s = ud.sum()
    if s <= 0:
        return np.zeros_like(ud, dtype=bool), np.nan

    udn = ud / s
    flat = udn.ravel()
    idx = np.argsort(flat)[::-1]
    csum = np.cumsum(flat[idx])
    k = np.searchsorted(csum, p, side="left")
    k = min(int(k), flat.size - 1)
    level = flat[idx[k]]
    mask = (udn >= level)
    return mask, level


def add_with_offset(total_arr, seg_arr, dx, dy):
    """
    Add seg_arr into total_arr with top-left offset (dx, dy).
    Here dx,dy are offsets in (col,row) indexing for array slicing:
    total_arr[row, col]
    """
    Ht, Wt = total_arr.shape
    Hs, Ws = seg_arr.shape

    x0 = max(dx, 0)
    y0 = max(dy, 0)
    x1 = min(dx + Ws, Wt)
    y1 = min(dy + Hs, Ht)
    if x0 >= x1 or y0 >= y1:
        return

    sx0 = x0 - dx
    sy0 = y0 - dy
    sx1 = sx0 + (x1 - x0)
    sy1 = sy0 + (y1 - y0)

    total_arr[y0:y1, x0:x1] += seg_arr[sy0:sy1, sx0:sx1]


def points_to_np_float64(points, n: int) -> np.ndarray:
    """
    Convert binding points buffer/iterable into a float64 numpy array of length n.
    Works for both iterables and indexable ctypes-backed arrays.
    """
    try:
        arr = np.fromiter(points, dtype=np.float64, count=n)
        if arr.size == n:
            return arr
    except TypeError:
        pass
    return np.fromiter((float(points[i]) for i in range(n)), dtype=np.float64, count=n)


def load_terrain_grid(
    csv_path: str,
    value_col: str = "habi.asc",
    x_col: str = "s1",
    y_col: str = "s2",
    fill_value: int = -1,
):
    df = pd.read_csv(csv_path)

    # numeric + drop missing
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=[value_col, x_col, y_col])

    # unique sorted coordinates
    xs = np.sort(df[x_col].unique())
    ys = np.sort(df[y_col].unique())

    if len(xs) < 2 or len(ys) < 2:
        raise ValueError("Not enough unique x/y coordinates to form a grid.")

    # infer resolution (median spacing)
    dx = float(np.median(np.diff(xs)))
    dy = float(np.median(np.diff(ys)))

    # index maps (coord -> grid index)
    x_to_ix = {x: i for i, x in enumerate(xs)}
    y_to_iy = {y: i for i, y in enumerate(ys)}

    # choose dtype
    vals = df[value_col].to_numpy()
    is_intish = np.all(np.isclose(vals, np.round(vals)))
    if is_intish and fill_value is not np.nan:
        dtype = np.int32
        df["_val"] = np.round(df[value_col]).astype(dtype)
    else:
        dtype = np.float32
        df["_val"] = df[value_col].astype(dtype)

    # allocate (rows=y, cols=x)
    terrain = np.full((len(ys), len(xs)), fill_value, dtype=dtype)

    # vectorized fill
    ix = df[x_col].map(x_to_ix).to_numpy()
    iy = df[y_col].map(y_to_iy).to_numpy()
    terrain[iy, ix] = df["_val"].to_numpy()

    meta = {
        "xs": xs,
        "ys": ys,
        "dx": dx,
        "dy": dy,
        "x_min": float(xs.min()),
        "y_min": float(ys.min()),
        "shape": terrain.shape,
    }
    return terrain, meta


def world_to_grid_floor(x, y, x0, y0, dx, dy):
    """
    Map world coords (x,y) to grid cell (row,col) using cell edges.
    """
    col = np.floor((x - (x0 - dx / 2.0)) / dx).astype(int)
    row = np.floor((y - (y0 - dy / 2.0)) / dy).astype(int)
    return row, col


def export_ud_grid_csv(
    ud_grid: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    out_csv: str = "python_method_ud_grid.csv",
    include_mask95: bool = True,
):
    """
    Export a UD grid to CSV in a flat format with x/y coordinates.
    """
    ud = np.asarray(ud_grid, dtype=float)
    if ud.shape != (len(ys), len(xs)):
        raise ValueError(
            f"Shape mismatch: ud_grid.shape={ud.shape}, expected {(len(ys), len(xs))} from xs/ys."
        )

    ud = np.where(np.isfinite(ud), ud, 0.0)
    ud = np.clip(ud, 0.0, None)

    s = ud.sum()
    ud_norm = (ud / s) if s > 0 else ud.copy()

    XX, YY = np.meshgrid(xs, ys)

    df = pd.DataFrame({
        "x": XX.ravel(order="C"),
        "y": YY.ravel(order="C"),
        "ud": ud.ravel(order="C"),
        "ud_norm": ud_norm.ravel(order="C"),
    })

    if include_mask95 and s > 0:
        _, level95 = ud_isopleth_mask(ud_norm, p=0.95)
        if np.isfinite(level95):
            df["mask95"] = (df["ud_norm"] >= level95).astype(int)
        else:
            df["mask95"] = 0

    df.to_csv(out_csv, index=False)
    print(f"Saved UD CSV: {out_csv}")


def plot_terrain_and_traj(terrain, meta, steps, ud=None, p=0.95, outline_lw=1.1):
    fig, ax = plt.subplots(figsize=(10, 9))

    terrain_colors = [
        (0.0, 0.0, 0.0, 1),  # 0 UNKNOWN
        (1.0, 0.0, 1.0, 1),  # 1 ROCKY_GROUNDS
        (0.0, 1.0, 1.0, 1),  # 2 GALLERIES
        (0.0, 1.0, 0.0, 1),  # 3 ANNUALS
        (0.0, 0.5, 0.0, 1),  # 4 PERENNIALS
    ]
    cmap = ListedColormap(terrain_colors)

    ax.imshow(
        terrain,
        origin="lower",
        cmap=cmap,
        interpolation="nearest",
        aspect="equal",
        vmin=0,
        vmax=len(terrain_colors) - 1,
        alpha=0.6,
        zorder=1,
    )

    # steps are (row, col, dt)
    rows = np.array([r for r, c, *_ in steps], dtype=float)
    cols = np.array([c for r, c, *_ in steps], dtype=float)
    ax.plot(cols, rows, linewidth=1.8, label="trajectory (mapped)", color="red", zorder=5)

    if ud is not None:
        ud = np.asarray(ud, dtype=float)
        print(
            "UD stats:",
            "shape", ud.shape,
            "sum", float(np.nansum(ud)),
            "min", float(np.nanmin(ud)),
            "max", float(np.nanmax(ud)),
            "finite_frac", float(np.isfinite(ud).mean()),
        )
        print("UD nonzero cells:", int(np.count_nonzero(ud)))

        if np.isfinite(ud).all() and ud.sum() > 0:
            mask, level = ud_isopleth_mask(ud, p=p)
            if np.isfinite(level):
                udn = ud / ud.sum()
                ax.contour(
                    udn,
                    levels=[level],
                    origin="lower",
                    linewidths=outline_lw,
                    colors="black",
                    antialiased=False,
                    zorder=10,
                )

    ax.set_title("Mapped buffalo trajectory over mapped habitat grid")
    ax.set_xlabel("grid col (x index)")
    ax.set_ylabel("grid row (y index)")
    ax.legend(loc="upper right")

    plt.tight_layout()
    out = "mapped_traj_on_mapped_grid.png"
    plt.savefig(out, dpi=200)
    print(f"Saved {out}")
    plt.show()


def build_kernel_mapping(dx, step_size):
    kernel_mapping = create_brownian_kernel_parameters(
        animal_type=AMPHIBIAN,
        base_step_size=step_size,
    )

    # Use diffusities from your first script
    diffusities = {
        UNKNOWN: 440.0,
        ROCKY_GROUNDS: 350.0,
        GALLERIES: 440.0,
        ANNUALS: 320.0,
        PERENNIALS: 220.0,
    }

    for cls, diff in diffusities.items():
        set_landmark_mapping(
            kernel_mapping,
            cls,
            is_brownian=True,
            step_size=step_size,
            directions=1,
            diffusity=diff / (dx * dx),
        )

    return kernel_mapping


def compute_total_utilization_sequential(
    terrain: np.ndarray,
    steps_rcdt,
    dx: float,
    step_size: int,
    T: int,
    padding: int,
):
    """
    Non-parallel version of compute_total_utilization_parallel using the same logic.

    steps_rcdt contains tuples: (row, col, dt_minutes_ceil)
    """
    H, W = terrain.shape
    total = np.zeros((H, W), dtype=np.float64)

    kernel_mapping = build_kernel_mapping(dx=dx, step_size=step_size)

    jobs = [
        (
            steps_rcdt[i][0],          # r0
            steps_rcdt[i][1],          # c0
            steps_rcdt[i + 1][0],      # r1
            steps_rcdt[i + 1][1],      # c1
            steps_rcdt[i][2] + 24,     # jt (same as parallel code)
        )
        for i in range(len(steps_rcdt) - 1)
        if steps_rcdt[i][2] <= 35
    ]

    print(f"Running {len(jobs)} segments sequentially")

    gen_times = []

    for k, (r0, c0, r1, c1, jt) in enumerate(jobs, 1):
        print(f"Processing segment {k}/{len(jobs)} | jt={jt - 24}")

        # segment bounds in GLOBAL terrain
        seg_min_x = max(min(c0, c1) - padding, 0)
        seg_min_y = max(min(r0, r1) - padding, 0)
        seg_max_x = min(max(c0, c1) + padding, W - 1)
        seg_max_y = min(max(r0, r1) + padding, H - 1)

        seg_W = seg_max_x - seg_min_x + 1
        seg_H = seg_max_y - seg_min_y + 1
        if seg_W <= 0 or seg_H <= 0:
            continue

        # local coordinates inside segment (x=col, y=row in walker call)
        sx = c0 - seg_min_x
        sy = r0 - seg_min_y
        ex = c1 - seg_min_x
        ey = r1 - seg_min_y

        seg_terrain = terrain[seg_min_y:seg_max_y + 1, seg_min_x:seg_max_x + 1]
        seg_terrain_map = numpy_to_terrain_map(seg_terrain)
        
        t_gen0 = time.perf_counter()
        util = MixedWalker.generate_utilization_distribution(
            seg_terrain_map,
            sx, sy, ex, ey,
            T=jt,
            kernel_mapping=kernel_mapping
        )
        t_gen = time.perf_counter() - t_gen0
        gen_times.append(t_gen)

        n = seg_W * seg_H
        acc = np.zeros(n, dtype=np.float64)

        # same accumulation window as your parallel version
        for t in range(12, jt - 15):
            pts = util[t][0].data[0][0].data.points
            acc += points_to_np_float64(pts, n)

        acc2d = acc.reshape((seg_H, seg_W))

        if (~np.isfinite(acc2d)).any():
            print(
                f"[WARN] segment {k}/{len(jobs)} has non-finite acc2d: "
                f"offset=({seg_min_x},{seg_min_y}) jt={jt} acc2d_shape={acc2d.shape}"
            )
            continue

        # same merge scaling as parallel version
        add_with_offset(total, acc2d * jt, seg_min_x, seg_min_y)

        if k % 50 == 0:
            print(f"merged {k}/{len(jobs)}")

    if gen_times:
        print("generate_utilization_distribution timing:")
        print(f"  sum = {np.sum(gen_times)}")
        print(f"  mean   = {np.mean(gen_times)}s")

    return total


def median_seg_terrain_stats(terrain_shape, steps_rcdt, padding):
    H, W = terrain_shape

    heights = []
    widths = []
    areas = []

    for i in range(len(steps_rcdt) - 1):
        # same filter as your compute function
        if steps_rcdt[i][2] > 35:
            continue

        r0, c0, _ = steps_rcdt[i]
        r1, c1, _ = steps_rcdt[i + 1]

        seg_min_x = max(min(c0, c1) - padding, 0)
        seg_min_y = max(min(r0, r1) - padding, 0)
        seg_max_x = min(max(c0, c1) + padding, W - 1)
        seg_max_y = min(max(r0, r1) + padding, H - 1)

        seg_W = seg_max_x - seg_min_x + 1
        seg_H = seg_max_y - seg_min_y + 1

        if seg_W <= 0 or seg_H <= 0:
            continue

        widths.append(seg_W)
        heights.append(seg_H)
        areas.append(seg_W * seg_H)

    if not areas:
        return None

    return {
        "n_segments": len(areas),
        "median_area_cells": float(np.mean(areas)),
        "median_height": float(np.mean(heights)),
        "median_width": float(np.mean(widths)),
    }

def main():
    # ---- load and resample terrain ----
    terrain, meta = load_terrain_grid(
        HAB_CSV,
        value_col=HAB_CLASS_COL,
        x_col=HAB_X_COL,
        y_col=HAB_Y_COL,
    )
    print("terrain shape:", terrain.shape)
    print("dx, dy:", meta["dx"], meta["dy"])
    print("x range:", meta["xs"][0], "to", meta["xs"][-1])
    print("y range:", meta["ys"][0], "to", meta["ys"][-1])

    factor = 1  # 30m -> 10m
    terrain = np.repeat(np.repeat(terrain, factor, axis=0), factor, axis=1)
    
    print(terrain.shape)
    
    dx = dy = 30.0 / factor  # resampled resolution

    # ---- load trajectory and map to grid ----
    x0_world = meta["xs"][0]
    y0_world = meta["ys"][0]

    traj = pd.read_csv(TRAJ_CSV)
    traj["x"] = pd.to_numeric(traj["x"], errors="coerce")
    traj["y"] = pd.to_numeric(traj["y"], errors="coerce")
    traj["dt"] = pd.to_numeric(traj["dt"], errors="coerce")
    traj = traj.dropna(subset=["x", "y", "dt"]).copy()

    row, col = world_to_grid_floor(
        traj["x"].to_numpy(),
        traj["y"].to_numpy(),
        x0_world,
        y0_world,
        dx,
        dy,
    )
    traj["row"] = row
    traj["col"] = col

    # match logic from first file: dt in minutes, ceil
    dt = np.ceil(traj["dt"].to_numpy() / 60.0).astype(np.int32)
    print("dt (min, ceil):", dt)

    # steps are (row, col, dt)
    steps = list(zip(traj["row"].tolist(), traj["col"].tolist(), dt.tolist()))
    

    # optional test subset (same idea as your first script)
    steps = steps

    # ---- compute utilization (SEQUENTIAL, non-parallelized) ----
    step_size = 15
    T = 31
    G_PADDING = 30
    
    
    
    stats = median_seg_terrain_stats(
        terrain_shape=terrain.shape,
        steps_rcdt=steps,
        padding=G_PADDING,
    )

    print("seg_terrain size stats:", stats)
    # return 0
    
    
    start = time.perf_counter()

    total_utilization = compute_total_utilization_sequential(
        terrain=terrain,
        steps_rcdt=steps,
        dx=dx,
        step_size=step_size,
        T=T,
        padding=G_PADDING,
    )

    elapsed = time.perf_counter() - start
    print(f"compute_total_utilization_sequential took {elapsed:.6f} seconds")

    total_sum = total_utilization.sum()
    print("total utilization sum (pre-norm):", float(total_sum))
    if total_sum > 0:
        total_utilization /= total_sum

    # ---- export UD to CSV (optional, carried over from first file) ----
    ny, nx = terrain.shape
    xs_resampled = x0_world + np.arange(nx) * dx
    ys_resampled = y0_world + np.arange(ny) * dy

    export_ud_grid_csv(
        ud_grid=total_utilization,
        xs=xs_resampled,
        ys=ys_resampled,
        out_csv="python_method_ud_grid.csv",
        include_mask95=True,
    )

    # ---- plot ----
    plot_terrain_and_traj(terrain, meta, steps, ud=total_utilization, p=0.95)


if __name__ == "__main__":
    main()