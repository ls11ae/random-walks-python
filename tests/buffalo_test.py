import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


from random_walk_package.bindings.data_structures.kernel_terrain_mapping import create_brownian_kernel_parameters, set_landmark_mapping
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
    level = flat[idx[k]]
    mask = (udn >= level)
    return mask, level


def add_with_offset(total_arr, seg_arr, dx, dy):
    
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
    fill_value: int = -1,   # use -1 for missing cells; or np.nan with float dtype
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
    # If values are integer-coded habitats, store as int
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

    # vectorized fill using index arrays
    ix = df[x_col].map(x_to_ix).to_numpy()
    iy = df[y_col].map(y_to_iy).to_numpy()
    terrain[iy, ix] = df["_val"].to_numpy()

    meta = {
        "xs": xs,          # x coordinate of each column
        "ys": ys,          # y coordinate of each row
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
    This is safest for arbitrary GPS points.
    """
    col = np.floor((x - (x0 - dx / 2.0)) / dx).astype(int)
    row = np.floor((y - (y0 - dy / 2.0)) / dy).astype(int)
    return row, col


def plot_terrain_and_traj(terrain, meta, steps, ud=None, p=0.95, outline_lw=1.1):
    fig, ax = plt.subplots(figsize=(10, 9))



    mask, level = ud_isopleth_mask(ud, p=p)
    udn = ud / ud.sum()
    ax.contour(udn, levels=[level], origin="lower", linewidths=1.0, colors="black", antialiased=False, zorder=10)
    # im_acc = ax.imshow(
    #     ud,
    #     origin="lower",
    #     interpolation="nearest",
    #     aspect="equal",
    #     alpha=1.0,      # overlay strength
    # )

    terrain_colors = [
        (0.0, 0.0, 0.0, 1),  # 0 UNKNOWN
        (1.0, 0.0, 1.0, 1),  # 1 ROCKY_GROUNDS
        (0.0, 1.0, 1.0, 1),  # 2 GALLERIES
        (0.0, 1.0, 0.0, 1),  # 3 ANNUALS
        (0.0, 0.5, 0.0, 1),  # 4 PERENNIALS
    ]
    cmap = ListedColormap(terrain_colors)

    im = ax.imshow(
        terrain,
        origin="lower",
        cmap=cmap,
        interpolation="nearest",
        aspect="equal",
        vmin=0,
        vmax=len(terrain_colors)-1,
        alpha=0.5,      # overlay strength
    )

    
    col_ok = np.array([x for x, y, *_ in steps], dtype=np.float64)
    row_ok = np.array([y for x, y, *_ in steps], dtype=np.float64)
    
    ax.plot(col_ok, row_ok, linewidth=1.8, label="trajectory (mapped)", color="red")


    # Draw only the home-range outline if UD provided
    # if ud is not None:
    #     mask, level = ud_isopleth_mask(ud, p=p)
    #     if mask.shape != terrain.shape:
    #         raise ValueError(f"UD/mask shape {mask.shape} must match terrain shape {terrain.shape}")

    #     ax.contour(
    #         mask.astype(float),
    #         levels=[0.5],
    #         origin="lower",
    #         color="red",
    #         linewidths=outline_lw,
    #         label=f"{int(p*100)}% UD isopleth"
    #     )

    #     # Matplotlib contours don't integrate nicely with legend labels;
    #     # easiest: add a manual legend entry.
    #     from matplotlib.lines import Line2D
    #     handles = [
    #         Line2D([0], [0], color="red", lw=1.8, label="trajectory (mapped)"),
    #         Line2D([0], [0], color="black", lw=outline_lw, label=f"{int(p*100)}% UD isopleth"),
    #     ]
    #     ax.legend(handles=handles, loc="upper right")
    # else:
    #     ax.legend(loc="upper right")

    ax.set_title("Mapped buffalo trajectory over mapped habitat grid")
    ax.set_xlabel("grid col (x index)")
    ax.set_ylabel("grid row (y index)")

    # cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    # cbar.set_label("habitat class")

    plt.tight_layout()
    out = "mapped_traj_on_mapped_grid.png"
    plt.savefig(out, dpi=200)
    print(f"Saved {out}")
    plt.show()


def main():
    terrain, meta = load_terrain_grid(HAB_CSV)
    print("terrain shape:", terrain.shape)
    print("dx, dy:", meta["dx"], meta["dy"])
    print("x range:", meta["xs"][0], "to", meta["xs"][-1])
    print("y range:", meta["ys"][0], "to", meta["ys"][-1])
    
    factor = 3  # 30m -> 10m
    terrain = np.repeat(np.repeat(terrain, factor, axis=0), factor, axis=1)

    # update metadata
    dx = dy = 30.0 / factor  # 10.0
    

    # print unique values in the terrain
    uniques = np.unique(terrain)
    
    # dx = dy = 30
    x0 = meta["xs"][0]
    y0 = meta["ys"][0]
    
    traj = pd.read_csv(TRAJ_CSV)
    traj["x"] = pd.to_numeric(traj["x"], errors="coerce")
    traj["y"] = pd.to_numeric(traj["y"], errors="coerce")
    traj = traj.dropna(subset=["x", "y"]).copy()

    row, col = world_to_grid_floor(traj["x"].to_numpy(), traj["y"].to_numpy(), x0, y0, dx, dy)
    traj["row"] = row
    traj["col"] = col
    
    terrain_map = numpy_to_terrain_map(terrain)

    
    step_size = 15
    
    kernel_mapping = create_brownian_kernel_parameters(animal_type=AMPHIBIAN, base_step_size=step_size)
    
    diffusities = {
        UNKNOWN: 440.0,
        ROCKY_GROUNDS: 350.0,
        GALLERIES: 440.0,
        ANNUALS: 320.0,
        PERENNIALS: 220.0
    }
    
    G_PADDING = 50

    set_landmark_mapping(kernel_mapping, UNKNOWN, is_brownian=True, step_size=step_size, directions=1, diffusity=diffusities[UNKNOWN]/(dx * dx))
    set_landmark_mapping(kernel_mapping, ROCKY_GROUNDS, is_brownian=True, step_size=step_size, directions=1, diffusity=diffusities[ROCKY_GROUNDS]/(dx * dx))
    set_landmark_mapping(kernel_mapping, GALLERIES, is_brownian=True, step_size=step_size, directions=1, diffusity=diffusities[GALLERIES]/(dx * dx))
    set_landmark_mapping(kernel_mapping, ANNUALS, is_brownian=True, step_size=step_size, directions=1, diffusity=diffusities[ANNUALS]/(dx * dx))
    set_landmark_mapping(kernel_mapping, PERENNIALS, is_brownian=True, step_size=step_size, directions=1, diffusity=diffusities[PERENNIALS]/(dx * dx))


    test = 2
    steps = [(x, y) for x, y in zip(traj["row"], traj["col"], traj["dt"])] #[test:test+20]
    
    steps = steps[:4]
    
    H = terrain.shape[0]
    W = terrain.shape[1]
    T = 40
    
    total_utilization = np.zeros((H, W), dtype=np.float64)
    home_range_mask = np.zeros((H, W), dtype=bool)
    
    for i in range(len(steps) - 1):
        
        print(f"Processing segment {i} of {len(steps) - 2}", end="\n")
    
        x0, y0 = steps[i]
        x1, y1 = steps[i+1]
        
        seg_min_x = min(x0, x1) - G_PADDING
        seg_min_y = min(y0, y1) - G_PADDING
        seg_max_x = max(x0, x1) + G_PADDING
        seg_max_y = max(y0, y1) + G_PADDING
        
        seg_min_x = max(seg_min_x, 0)
        seg_min_y = max(seg_min_y, 0)
        seg_max_x = min(seg_max_x, W - 1)
        seg_max_y = min(seg_max_y, H - 1)

        seg_W = seg_max_x - seg_min_x + 1
        seg_H = seg_max_y - seg_min_y + 1
        
        sx = x0 - seg_min_x
        sy = y0 - seg_min_y
        ex = x1 - seg_min_x
        ey = y1 - seg_min_y


        dx = seg_min_x
        dy = seg_min_y
        

        # sx, sy = steps[0]
        # ex, ey = steps[1]

        # extract the rectangular segment from the full terrain
        seg_terrain = terrain[seg_min_y:seg_max_y + 1, seg_min_x:seg_max_x + 1]

        # create a terrain_map for the segment (bindings expect this form)
        seg_terrain_map = numpy_to_terrain_map(seg_terrain)
        
        util = MixedWalker.generate_utilization_distribution(seg_terrain_map, sx, sy, ex, ey, T=T, kernel_mapping=kernel_mapping)
        
        
        n = seg_W * seg_H
        
        acc = np.zeros(n, dtype=np.float64)
        
        for t in range(T):
            # print(util[t][0].data[0][0].len)
            pts = util[t][0].data[0][0].data.points
            acc += points_to_np_float64(pts, n)

        acc /= T

        # pts = util[2][0].data[0][0].data.points
        # acc += points_to_np_float64(pts, n)
        
        acc = acc.reshape((seg_H, seg_W))

        add_with_offset(total_utilization, acc, dx, dy)
    
    
    # plot_terrain_and_traj(seg_terrain, meta, [(sx, sy), (ex, ey)], ud=acc)

    # walk = MixedWalker.generate_custom_walks(
    #     terrain=seg_terrain_map,
    #     steps=[(sx, sy), (ex, ey)],
    #     T=30,
    #     kernel_mapping=kernel_mapping,
    #     plot=True,
    #     plot_title="Custom Terrain Walk"
    # )

    # print(type(terrain_map))

    plot_terrain_and_traj(terrain, meta, steps, ud=total_utilization, p=0.95)

if __name__ == "__main__":
    main()