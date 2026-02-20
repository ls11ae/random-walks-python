import os
import re
import math
import numpy as np
import pandas as pd
import rasterio
from datetime import datetime
from pyproj import Transformer
from collections import defaultdict

CRS_MGA55 = "EPSG:28355"

# -------------------------
# Your CSV processing
# -------------------------
def process_bettongs(file_path: str):
    md = {
        "lagartha", "bjorn", "baldur", "sifa", "floki", "freya", "andive",
        "beetroot", "durian", "parsnip", "potato", "pumpkin", "raddish", "sprout",
        "swede", "turnip", "tomato"
    }
    dm = {"dot", "edwina", "egbert", "maud", "olga", "othello", "percy", "renet"}

    data = pd.read_csv(file_path)
    bettongs = {}

    for _, row in data.iterrows():
        bettong_id = row["ID"]

        if bettong_id in md:
            dt = datetime.strptime(row["datetime"], "%m/%d/%y %H:%M")
        elif bettong_id in dm:
            dt = datetime.strptime(row["datetime"], "%d/%m/%y %H:%M")
        else:
            raise ValueError(f"UNKNOWN ID: {bettong_id}")

        state = row["states"]
        vegindex = row.get("VegIndex")  # unused, but kept
        cover = row.get("cover")  # unused, but kept

        bettongs.setdefault(bettong_id, []).append(
            (int(row["x"]), int(row["y"]), dt, state, vegindex, cover)
        )

    for bid in bettongs:
        bettongs[bid].sort(key=lambda t: t[2])

    return bettongs

def compare_cover_vs_pg_stats(df, cover_col="cover", pg_col="persistent_green_pct"):
    d = df[[cover_col, pg_col]].copy()
    d[cover_col] = pd.to_numeric(d[cover_col], errors="coerce")
    d[pg_col] = pd.to_numeric(d[pg_col], errors="coerce")
    d = d.dropna()

    if len(d) < 3:
        return {
            "n": int(len(d)),
            "pearson_r": np.nan, "pearson_p": np.nan,
            "spearman_rho": np.nan, "spearman_p": np.nan,
            "mae": np.nan, "rmse": np.nan, "bias": np.nan,
        }

    x = d[cover_col].to_numpy(float)
    y = d[pg_col].to_numpy(float)

    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)

    diff = y - x
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    bias = float(np.mean(diff))

    return {
        "n": int(len(d)),
        "pearson_r": float(pr), "pearson_p": float(pp),
        "spearman_rho": float(sr), "spearman_p": float(sp),
        "mae": mae, "rmse": rmse, "bias": bias,
    }

def within_tolerance_pct(df, tol=5, cover_col="cover", pg_col="persistent_green_pct"):
    d = df[[cover_col, pg_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) == 0:
        return np.nan
    return float((np.abs(d[pg_col] - d[cover_col]) <= tol).mean() * 100)

def compute_all_bettongs_cover_vs_pg(bettongs: dict, tif_dir: str,
                                    state_code="tas", nodata=255):
    """
    bettongs: dict[id] -> trajectory list of (x,y,dt,state,vegindex)
    Requires you already have `sample_persistent_green(bettong_traj, tif_dir, nodata)`
    which returns DF with persistent_green_pct.
    Assumes your CSV already contains 'cover' (you showed it does).
    If 'cover' is not in the sampled DF, merge it in from CSV before calling stats.
    """

    all_rows = []
    per_id_stats = []

    for bid, traj in bettongs.items():
        # traj must include cover somehow; if not, adapt sampler to carry it through.
        df_one = sample_persistent_green(traj, tif_dir=tif_dir, nodata=nodata)
        df_one["bettong_id"] = bid
        all_rows.append(df_one)

        st = compare_cover_vs_pg_stats(df_one)
        st["bettong_id"] = bid
        st["within5%"] = within_tolerance_pct(df_one, tol=5)
        st["within10%"] = within_tolerance_pct(df_one, tol=10)
        per_id_stats.append(st)

    df_all = pd.concat(all_rows, ignore_index=True)
    df_stats = pd.DataFrame(per_id_stats).sort_values("bettong_id")

    overall = compare_cover_vs_pg_stats(df_all)
    overall["within5%"] = within_tolerance_pct(df_all, tol=5)
    overall["within10%"] = within_tolerance_pct(df_all, tol=10)

    return df_all, df_stats, overall
# -------------------------
# Season logic matching your filenames
# -------------------------
def season_tag_for_date(dt: datetime) -> str:
    """
    Return the YYYYMMYYYYMM season tag used in filenames:
      DJF: Dec–Feb => m(prevDec)(Feb)
      MAM: Mar–May => m(Mar)(May)
      JJA: Jun–Aug => m(Jun)(Aug)
      SON: Sep–Nov => m(Sep)(Nov)
    """
    y, m = dt.year, dt.month

    if m in (12, 1, 2):  # DJF
        start_y = y if m == 12 else y - 1
        start = f"{start_y}12"
        end = f"{start_y + 1}02"
    elif m in (3, 4, 5):  # MAM
        start = f"{y}03"
        end = f"{y}05"
    elif m in (6, 7, 8):  # JJA
        start = f"{y}06"
        end = f"{y}08"
    else:  # SON
        start = f"{y}09"
        end = f"{y}11"

    return f"{start}{end}"

def build_filename_for_date(dt: datetime) -> str:
    return f"lztmre_tas_m{season_tag_for_date(dt)}_dpaa2.tif"

# -------------------------
# Index local TIFFs
# -------------------------
def index_local_tifs(tif_dir: str):
    """
    Returns dict: season_tag (YYYYMMYYYYMM) -> full path
    """
    out = {}
    for fn in os.listdir(tif_dir):
        if not fn.lower().endswith(".tif"):
            continue
        m = re.search(r"_m(\d{12})_", fn)
        if not m:
            continue
        out[m.group(1)] = os.path.join(tif_dir, fn)
    return out

# -------------------------
# Sample persistent green at points
# -------------------------
def sample_persistent_green(bettong_traj, tif_dir: str, nodata=255) -> pd.DataFrame:
    """
    Returns DF with persistent_green_pct sampled from the correct seasonal tif per point.
    """
    tif_index = index_local_tifs(tif_dir)

    # group points by needed tif file (reduces opens)
    groups = defaultdict(list)
    for (x, y, dt, state, veg, cover) in bettong_traj:
        tag = season_tag_for_date(dt)
        tif_path = tif_index.get(tag)
        # tif_path = f"{tif_dir}/lztmre_tas_m201603201605_djaa2.tif" 
        groups[tif_path].append((x, y, dt, state, veg, tag, cover))

    rows = []

    # process each tif group
    for tif_path, pts in groups.items():
        if tif_path is None:
            # you don't have that season downloaded
            for (x, y, dt, state, veg, tag, cover) in pts:
                rows.append({
                    "x": x, "y": y, "datetime": dt, "state": state,
                    "cover": cover, "tif": None, "persistent_green_pct": math.nan
                })
            continue

        with rasterio.open(tif_path) as src:
            src_crs = src.crs
            if src_crs is None:
                raise RuntimeError(f"TIFF has no CRS: {tif_path}")

            # transform MGA55 -> raster CRS only if needed
            if str(src_crs).upper() == CRS_MGA55:
                coords = [(float(x), float(y)) for (x, y, *_rest) in pts]
            else:
                tr = Transformer.from_crs(CRS_MGA55, src_crs, always_xy=True)
                coords = [tr.transform(float(x), float(y)) for (x, y, *_rest) in pts]

            samples = list(src.sample(coords))

            for (pt, s) in zip(pts, samples):
                x, y, dt, state, veg, tag, cover = pt
                v = float(s[0]) if s is not None and len(s) else math.nan
                if np.isfinite(v) and int(v) == int(nodata):
                    v = math.nan

                rows.append({
                    "x": x, "y": y, "datetime": dt, "state": state,
                    "season_tag": tag, "cover": cover,
                    "persistent_green_pct": v
                })
    df = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
    df["cover"] *= 100
    df["cover"] = df["cover"].astype(int)
    df["cover"] = df["cover"].astype(float)
    df["persistent_green_pct"] = pd.to_numeric(df["persistent_green_pct"], errors="coerce")
    mask = df["persistent_green_pct"] > 100
    if mask.any():
        df.loc[mask, "persistent_green_pct"] = df.loc[mask, "persistent_green_pct"] - 100
    # df["persistent_green_pct"] -= 100
    return df


def within_tolerance(df, tol=5, cover_col="cover", pg_col="persistent_green_pct"):
    d = df[[cover_col, pg_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) == 0:
        return {"tol": tol, "pct_within": np.nan, "n": 0}
    within = (np.abs(d[pg_col] - d[cover_col]) <= tol)
    return {"tol": tol, "pct_within": float(within.mean() * 100), "n": int(len(d))}



from scipy.stats import spearmanr, pearsonr

def compare_cover_vs_persistent_green(df,
                                     cover_col="cover",
                                     pg_col="persistent_green_pct"):
    d = df[[cover_col, pg_col]].copy()
    d[cover_col] = pd.to_numeric(d[cover_col], errors="coerce")
    d[pg_col] = pd.to_numeric(d[pg_col], errors="coerce")
    d = d.dropna()

    if len(d) < 3:
        return {"n": len(d), "pearson_r": np.nan, "pearson_p": np.nan,
                "spearman_rho": np.nan, "spearman_p": np.nan,
                "mae": np.nan, "rmse": np.nan, "bias": np.nan}

    x = d[cover_col].to_numpy(dtype=float)
    y = d[pg_col].to_numpy(dtype=float)

    # Correlations
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)

    # Agreement-ish metrics (treat persistent green as "estimate" of cover)
    diff = y - x
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    bias = float(np.mean(diff))          # positive => persistent_green higher on avg

    return {
        "n": int(len(d)),
        "pearson_r": float(pr), "pearson_p": float(pp),
        "spearman_rho": float(sr), "spearman_p": float(sp),
        "mae": mae, "rmse": rmse, "bias": bias,
    }


import os
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.stats import pearsonr, spearmanr

CRS_MGA55 = "EPSG:28355"

def _metrics(x, y):
    """x=cover, y=persistent_green sampled from tif"""
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 3:
        return {
            "n": n,
            "pearson_r": np.nan, "pearson_p": np.nan,
            "spearman_rho": np.nan, "spearman_p": np.nan,
            "mae": np.nan, "rmse": np.nan, "bias": np.nan,
            "within5%": np.nan, "within10%": np.nan,
        }

    xx = x[mask]
    yy = y[mask]
    pr, pp = pearsonr(xx, yy)
    sr, sp = spearmanr(xx, yy)

    diff = yy - xx
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    bias = float(np.mean(diff))
    within5 = float((np.abs(diff) <= 5).mean() * 100)
    within10 = float((np.abs(diff) <= 10).mean() * 100)

    return {
        "n": n,
        "pearson_r": float(pr), "pearson_p": float(pp),
        "spearman_rho": float(sr), "spearman_p": float(sp),
        "mae": mae, "rmse": rmse, "bias": bias,
        "within5%": within5, "within10%": within10,
    }

def sample_tif_at_points(tif_path, xs, ys, src_crs_points=CRS_MGA55, nodata=255):
    """
    Sample tif at MGA55 x/y arrays. Returns float array with NaN for nodata/outside.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    with rasterio.open(tif_path) as src:
        b = src.bounds
        crs = src.crs
        if crs is None:
            raise RuntimeError(f"No CRS in tif: {tif_path}")

        # transform points to tif CRS if needed
        if str(crs).upper() != src_crs_points:
            tr = Transformer.from_crs(src_crs_points, crs, always_xy=True)
            tx, ty = tr.transform(xs, ys)
        else:
            tx, ty = xs, ys

        # bounds mask (avoid sampling pointless outside)
        inside = (tx >= b.left) & (tx <= b.right) & (ty >= b.bottom) & (ty <= b.top)

        out = np.full(xs.shape[0], np.nan, dtype=float)
        if inside.any():
            coords = list(zip(np.asarray(tx)[inside], np.asarray(ty)[inside]))
            samp = np.array([s[0] for s in src.sample(coords)], dtype=float)

            # nodata handling
            if src.nodata is not None:
                samp[samp == float(src.nodata)] = np.nan
            # also apply known nodata=255 for this product
            samp[samp == float(nodata)] = np.nan
            
            print(samp)
            
            if samp.size > 0 and np.isfinite(samp[0]) and samp[0] > 100:
                samp = samp - 100

            out[inside] = samp

    return out

def find_best_tif_for_cover(df_all, tif_dir, cover_col="cover", x_col="x", y_col="y",
                            nodata=255, rank_by="mae"):
    """
    df_all must contain x,y,cover for ALL points.
    rank_by: "mae" (lower better) or "pearson_r" (higher better) or "within10%" (higher better)
    """
    d = df_all[[x_col, y_col, cover_col]].copy()
    d[cover_col] = pd.to_numeric(d[cover_col], errors="coerce")
    
    d = d.dropna(subset=[x_col, y_col, cover_col])

    xs = d[x_col].to_numpy()
    ys = d[y_col].to_numpy()
    cover = d[cover_col].to_numpy(dtype=float)

    rows = []
    tifs = sorted([f for f in os.listdir(tif_dir) if f.lower().endswith(".tif")])

    for fn in tifs:
        path = os.path.join(tif_dir, fn)
        sampled = sample_tif_at_points(path, xs, ys, src_crs_points=CRS_MGA55, nodata=nodata)
        m = _metrics(cover, sampled)
        m["tif"] = fn
        rows.append(m)

    res = pd.DataFrame(rows)

    # ranking
    if rank_by == "mae":
        res = res.sort_values(["mae", "rmse"], ascending=[True, True])
    elif rank_by == "pearson_r":
        res = res.sort_values(["pearson_r", "within10%"], ascending=[False, False])
    elif rank_by == "within10%":
        res = res.sort_values(["within10%", "mae"], ascending=[False, True])
    else:
        raise ValueError("rank_by must be one of: mae, pearson_r, within10%")

    return res


# -------------------------
# Example run
# -------------------------
if __name__ == "__main__":
    csv_path = "/home/mart/Code/eco/Gardiner- Habitat.Perception.Bettongs.csv"
    tif_dir = "csv_out/tif_in"

    bettongs = process_bettongs(csv_path)
    
    df_all, df_by_id, overall = compute_all_bettongs_cover_vs_pg(bettongs, tif_dir="csv_out/tif_in")
    print("OVERALL:", overall)
    print(df_by_id)
    df_all.to_csv("all_bettongs_persistent_green_samples.csv", index=False)
    df_by_id.to_csv("cover_vs_persistent_green_by_bettong.csv", index=False)
    
    by_season = (
    df_all.dropna(subset=["cover","persistent_green_pct"])
            .groupby("season_tag")
            .apply(lambda g: pd.Series({
                "n": len(g),
                "mae": np.mean(np.abs(g["persistent_green_pct"] - g["cover"])),
                "within5%": (np.abs(g["persistent_green_pct"] - g["cover"]) <= 5).mean()*100,
                "within10%": (np.abs(g["persistent_green_pct"] - g["cover"]) <= 10).mean()*100,
            }))
            .reset_index()
            .sort_values("n", ascending=False)
    )
    print(by_season)
    
    rank = find_best_tif_for_cover(df_all, "csv_out/tif_in", rank_by="mae")
    print(rank[["tif","n","mae","rmse","bias","pearson_r","spearman_rho","within5%","within10%"]])
    print("\nBEST:", rank.iloc[0].to_dict())
    bettong_id = "baldur"
    bettong = bettongs[bettong_id]

    df = sample_persistent_green(bettong, tif_dir=tif_dir, nodata=255)
    print(df.head(20))
    print("\nNaNs:", df["persistent_green_pct"].isna().sum(), "/", len(df))
    print(df["persistent_green_pct"].describe())

    df.to_csv(f"persistent_green_{bettong_id}.csv", index=False)
    print(f"\nWrote: persistent_green_{bettong_id}.csv")

    print(within_tolerance(df, tol=5))
    print(within_tolerance(df, tol=10))
    
    stats = compare_cover_vs_persistent_green(df)
    print(stats)