import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from concurrent.futures import ProcessPoolExecutor, as_completed

import multiprocessing as mp

from random_walk_package.bindings.data_structures.kernel_terrain_mapping import (
    create_brownian_kernel_parameters,
    set_landmark_mapping,
)
from random_walk_package.bindings.data_structures.terrain import (
    kernels_map3d_free,
    numpy_to_terrain_map,
    UNKNOWN,
    ROCKY_GROUNDS,
    GALLERIES,
    ANNUALS,
    PERENNIALS,
    AMPHIBIAN,
)
from random_walk_package.core.MixedWalker import MixedWalker

HAB_CSV = "tests/buf_data/buffalo_habitat.csv"
TRAJ_CSV = "tests/buf_data/buffalo_trajectory.csv"


# ============================================================
# Helpers
# ============================================================
def load_terrain_grid(
    csv_path: str,
    value_col: str = "habi.asc",
    x_col: str = "s1",
    y_col: str = "s2",
    fill_value: int = -1,
):
    df = pd.read_csv(csv_path)

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=[value_col, x_col, y_col])

    xs = np.sort(df[x_col].unique())
    ys = np.sort(df[y_col].unique())

    if len(xs) < 2 or len(ys) < 2:
        raise ValueError("Not enough unique x/y coordinates to form a grid.")

    dx = float(np.median(np.diff(xs)))
    dy = float(np.median(np.diff(ys)))

    x_to_ix = {x: i for i, x in enumerate(xs)}
    y_to_iy = {y: i for i, y in enumerate(ys)}

    vals = df[value_col].to_numpy()
    is_intish = np.all(np.isclose(vals, np.round(vals)))
    if is_intish and fill_value is not np.nan:
        dtype = np.int32
        df["_val"] = np.round(df[value_col]).astype(dtype)
    else:
        dtype = np.float32
        df["_val"] = df[value_col].astype(dtype)

    terrain = np.full((len(ys), len(xs)), fill_value, dtype=dtype)

    ix = df[x_col].map(x_to_ix).to_numpy()
    iy = df[y_col].map(y_to_iy).to_numpy()
    terrain[iy, ix] = df["_val"].to_numpy()

    meta = {"xs": xs, "ys": ys, "dx": dx, "dy": dy, "shape": terrain.shape}
    return terrain, meta


def world_to_grid_floor(x, y, x0, y0, dx, dy):
    col = np.floor((x - (x0 - dx / 2.0)) / dx).astype(int)
    row = np.floor((y - (y0 - dy / 2.0)) / dy).astype(int)
    return row, col


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


def add_with_offset_inplace(total_arr: np.ndarray, seg_arr: np.ndarray, dx: int, dy: int) -> None:
    """
    Add seg_arr into total_arr with top-left offset (dx, dy).
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


def ud_isopleth_level(ud: np.ndarray, p: float = 0.95) -> float:
    """
    Compute density threshold level on normalized UD such that the
    highest-density cells sum to p.
    """
    ud = np.asarray(ud, dtype=float)
    ud = np.clip(ud, 0, None)
    s = ud.sum()
    if s <= 0:
        return np.nan

    udn = ud / s
    flat = udn.ravel()
    idx = np.argsort(flat)[::-1]
    csum = np.cumsum(flat[idx])
    k = np.searchsorted(csum, p, side="left")
    k = min(int(k), flat.size - 1)
    return float(flat[idx[k]])


# ============================================================
# Plotting
# ============================================================
def plot_terrain_and_traj(terrain: np.ndarray, steps_rc, ud: np.ndarray | None, p=0.95,
                         out="mapped_traj_on_mapped_grid.png"):
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
    )

    # steps are (row, col)
    rows = np.array([r for r, c, _ in steps_rc], dtype=float)
    cols = np.array([c for r, c, _ in steps_rc], dtype=float)
    ax.plot(cols, rows, linewidth=1.8, color="red", label="trajectory")
    
    
    
    print("UD stats:",
        "shape", ud.shape,
        "sum", float(np.nansum(ud)),
        "min", float(np.nanmin(ud)),
        "max", float(np.nanmax(ud)),
        "finite_frac", float(np.isfinite(ud).mean()))
    print("UD nonzero cells:", int(np.count_nonzero(ud)))

    if ud is not None and np.isfinite(ud).all() and ud.sum() > 0:
        level = ud_isopleth_level(ud, p=p)
        if np.isfinite(level):
            udn = ud / ud.sum()
            ax.contour(
                udn,
                levels=[level],
                origin="lower",
                linewidths=1.2,
                colors="black",
                antialiased=False,
                zorder=10,
            )

    ax.set_title("Trajectory + utilization isopleth")
    ax.set_xlabel("grid col")
    ax.set_ylabel("grid row")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(out, dpi=200)
    print(f"Saved {out}")
    plt.show()


# ============================================================
# Parallel segment processing (ProcessPool + initializer)
# ============================================================
_G_TERRAIN = None
_G_KERNEL_MAPPING = None
_G_T = None
_G_PADDING = None


def _init_worker(terrain: np.ndarray, dx: float, step_size: int, T: int, padding: int):
    """
    Runs once per worker process.
    Builds the non-picklable kernel mapping inside the process.
    """
    global _G_TERRAIN, _G_KERNEL_MAPPING, _G_T, _G_PADDING
    _G_TERRAIN = terrain
    _G_T = int(T)
    _G_PADDING = int(padding)

    km = create_brownian_kernel_parameters(animal_type=AMPHIBIAN, base_step_size=step_size)

    diffusities = {
        UNKNOWN: 440.0,
        ROCKY_GROUNDS: 350.0,
        GALLERIES: 440.0,
        ANNUALS: 320.0,
        PERENNIALS: 440.0,
    }
    for cls, diff in diffusities.items():
        set_landmark_mapping(
            km,
            cls,
            is_brownian=True,
            step_size=step_size,
            directions=1,
            diffusity=diff / (dx*dx),
        )

    _G_KERNEL_MAPPING = km


def _compute_segment_job(job):
    """
    job = (r0, c0, r1, c1) with steps in (row,col)
    returns (seg_min_x, seg_min_y, acc2d) or None
    """
    global _G_TERRAIN, _G_KERNEL_MAPPING, _G_PADDING

    H, W = _G_TERRAIN.shape
    r0, c0, r1, c1, jt = job

    seg_min_x = max(min(c0, c1) - _G_PADDING, 0)
    seg_min_y = max(min(r0, r1) - _G_PADDING, 0)
    seg_max_x = min(max(c0, c1) + _G_PADDING, W - 1)
    seg_max_y = min(max(r0, r1) + _G_PADDING, H - 1)

    seg_W = seg_max_x - seg_min_x + 1
    seg_H = seg_max_y - seg_min_y + 1
    if seg_W <= 0 or seg_H <= 0:
        return None

    # local coords inside segment
    sx = c0 - seg_min_x
    sy = r0 - seg_min_y
    ex = c1 - seg_min_x
    ey = r1 - seg_min_y

    seg_terrain = _G_TERRAIN[seg_min_y:seg_max_y + 1, seg_min_x:seg_max_x + 1]
    seg_terrain_map = numpy_to_terrain_map(seg_terrain)

    util = MixedWalker.generate_utilization_distribution(
        seg_terrain_map, sx, sy, ex, ey, T=jt, kernel_mapping=_G_KERNEL_MAPPING
    )

    n = seg_W * seg_H
    acc = np.zeros(n, dtype=np.float64)

    for t in range(12, jt-15):
        pts = util[t][0].data[0][0].data.points
        acc += points_to_np_float64(pts, n)

    # acc /= jt
    return (seg_min_x, seg_min_y, jt-3, acc.reshape((seg_H, seg_W)))


def compute_total_utilization_parallel(
    terrain: np.ndarray,
    steps_rc,
    dx: float,
    step_size: int,
    T: int,
    padding: int,
    max_workers: int | None = None,
):
    if max_workers is None:
        max_workers = os.cpu_count() or 1
        
    # max_workers = 28

    total = np.zeros(terrain.shape, dtype=np.float64)

    jobs = [
        (steps_rc[i][0], steps_rc[i][1], steps_rc[i + 1][0], steps_rc[i + 1][1], steps_rc[i][2] + 24)
        for i in range(len(steps_rc) - 1) if steps_rc[i][2] <= 35
    ]

    print(f"Running {len(jobs)} segments with process pool (spawn), workers={max_workers}")

    ctx = mp.get_context("spawn")  # <-- key line

    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,             # <-- key line
        initializer=_init_worker,
        initargs=(terrain, dx, step_size, T, padding),
    ) as ex:
        futures = [ex.submit(_compute_segment_job, job) for job in jobs]

        for k, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if res is None:
                continue

            seg_min_x, seg_min_y, jt, acc2d = res

            if (~np.isfinite(acc2d)).any():
                print(
                    f"[WARN] segment {k}/{len(jobs)} has non-finite acc2d: "
                    f"offset=({seg_min_x},{seg_min_y}) jt={jt!r} "
                    f"acc2d_shape={acc2d.shape}"
                )
                continue

            add_with_offset_inplace(total, acc2d * jt, seg_min_x, seg_min_y)

            if k % 50 == 0:
                print(f"merged {k}/{len(jobs)}")

    return total


def export_ud_grid_csv(
    ud_grid: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    out_csv: str = "python_method_ud_grid.csv",
    include_mask95: bool = True,
):
    """
    Export a UD grid to CSV in a format compatible with the BRB plotting script.

    Parameters
    ----------
    ud_grid : (ny, nx) array
        Utilization grid in habitat-grid alignment.
    xs : (nx,) array
        Real x coordinates for grid columns (cell centers).
    ys : (ny,) array
        Real y coordinates for grid rows (cell centers).
    out_csv : str
        Output CSV path.
    include_mask95 : bool
        If True, includes a binary mask95 column (1 inside 95% isopleth).
    """
    ud = np.asarray(ud_grid, dtype=float)
    if ud.shape != (len(ys), len(xs)):
        raise ValueError(
            f"Shape mismatch: ud_grid.shape={ud.shape}, expected {(len(ys), len(xs))} from xs/ys."
        )

    # keep nonnegative finite values
    ud = np.where(np.isfinite(ud), ud, 0.0)
    ud = np.clip(ud, 0.0, None)

    s = ud.sum()
    ud_norm = (ud / s) if s > 0 else ud.copy()

    # build flattened x/y grid matching row-major flatten of ud
    XX, YY = np.meshgrid(xs, ys)  # shapes (ny, nx)

    df = pd.DataFrame({
        "x": XX.ravel(order="C"),
        "y": YY.ravel(order="C"),
        "ud": ud.ravel(order="C"),
        "ud_norm": ud_norm.ravel(order="C"),
    })

    if include_mask95 and s > 0:
        level95 = ud_isopleth_level(ud_norm, p=0.95)
        if np.isfinite(level95):
            df["mask95"] = (df["ud_norm"] >= level95).astype(int)
        else:
            df["mask95"] = 0

    df.to_csv(out_csv, index=False)
    print(f"Saved UD CSV: {out_csv}")

# ============================================================
# Main
# ============================================================
def main():
    # ---- load and resample terrain ----
    terrain, meta = load_terrain_grid(HAB_CSV)

    factor = 3
    terrain = np.repeat(np.repeat(terrain, factor, axis=0), factor, axis=1)
    dx = dy = 30.0 / factor

    # ---- load trajectory and map to grid ----
    traj = pd.read_csv(TRAJ_CSV)
    traj["x"] = pd.to_numeric(traj["x"], errors="coerce")
    traj["y"] = pd.to_numeric(traj["y"], errors="coerce")
    traj = traj.dropna(subset=["x", "y"]).copy()

    x0_world = meta["xs"][0]
    y0_world = meta["ys"][0]

    row, col = world_to_grid_floor(traj["x"].to_numpy(), traj["y"].to_numpy(), x0_world, y0_world, dx, dy)
    
    dt = traj["dt"].to_numpy()
    
    dt = np.ceil(dt/60).astype(np.int32)
    
    print(dt)
    

    # steps are (row, col)
    steps = list(zip(row.tolist(), col.tolist(), dt.tolist()))
    
    steps = steps[:3]  # for testing, limit to first 100 steps
    print(steps)

    # ---- compute utilization in parallel ----
    step_size = 30
    T = 31
    G_PADDING = 60

    total_utilization = compute_total_utilization_parallel(
        terrain=terrain,
        steps_rc=steps,
        dx=dx,
        step_size=step_size,
        T=T,
        padding=G_PADDING,
        max_workers=1,  # or set e.g. 16
    )
    
    

    total_sum = total_utilization.sum()
    print(total_sum)
    if total_sum > 0:
        total_utilization /= total_sum

    # ---- export UD to CSV (compatible with BRB plotting script) ----
    # Reconstruct real-coordinate vectors for the RESAMPLED terrain grid
    # (same coordinates used for mapping/plotting)
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
    plot_terrain_and_traj(terrain, steps, ud=total_utilization, p=0.95)


if __name__ == "__main__":
    main()