from movingpandas import Trajectory
from skimage.transform import resize

from environmentcma import padded_utm_bbox, grid_to_geo_walk, grid_shape_from_bbox, \
    utm_to_grid
from random_walk_package.utils.move_apps_patch import merge_traj_collections, apply_moveapps_id_dtype_patch, \
    debug_patch_state, force_tc_id_object_inplace
import geopandas as gpd
import movingpandas as mpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from random_walk_package import MixedWalker, get_walk_points, dll, tensor_free, tensor4D_free, \
    FixedStepsPolicy
from random_walk_package.bindings import kernels_map3d_free, Animal, create_mixed_kernel_parameters, \
    landcover_to_discrete_ptr, WaterMode, terrain_map_free
from random_walk_package.bindings.correlated_walk import correlated_walk_init, correlated_backtrace
from random_walk_package.bindings.data_structures.kernel_terrain_mapping import marine_kernels_baseline_crw
from random_walk_package.bindings.data_structures.kernels import normalize_kernel, clip_kernel, \
    correlated_kernels_from_matrix
from random_walk_package.bindings.mixed_walk import single_state_walk, kernels_map_single_kernel
from segmentationcma import (
    UTMDistanceCriterion,
    annotate_segments_dataframe,
    bbox_of_segment,
    make_overlapping,
    segment_dataframe,
)
from random_walk_package.utils.walker_utils import resample_kernel_to_grid, direction_from_points


class StateDependentWalker(MixedWalker):
    def __init__(self, data, animal_type, resolution, out_directory,
                 n_hmm_states=3,
                 time_col="timestamp",
                 lon_col="location-long",
                 lat_col="location-lat",
                 id_col="individual-local-identifier",
                 crs="EPSG:4326"):
        print("Version 0.1.7")
        apply_moveapps_id_dtype_patch()
        debug_patch_state()
        self.original_data = None
        if isinstance(data, mpd.TrajectoryCollection):
            force_tc_id_object_inplace(data)
            import copy
            data_copy = copy.deepcopy(data)
            self.original_data = data_copy
        self.animal = animal_type
        self.n_hmm_states = n_hmm_states
        is_marine = animal_type == Animal.MARINE or animal_type == Animal.AIRBORNE
        if animal_type is Animal.AIRBORNE:
            mapping = None
            self.is_marine = True
        elif animal_type is Animal.MARINE:
            mapping = marine_kernels_baseline_crw(5, 5, 1, 1)
        else:
            mapping = create_mixed_kernel_parameters(animal_type, 5)
        super().__init__(data, mapping, resolution, out_directory, time_col, lon_col, lat_col, id_col, crs, is_marine)

    def get_kernels(self, dt_tolerance, rnge, out_dir, is_brownian):
        super().process_movebank_data()

        if self.original_data is None:
            self.original_data = self.animal_proc.traj_coll

        [corZs, brwZs] = self.animal_proc.get_hmm_kernels(dt_tolerance=dt_tolerance,
                                                          rnge=rnge,
                                                          out_dir=out_dir,
                                                          num_states=self.n_hmm_states)
        selected_kernels = brwZs if is_brownian and self.animal is not Animal.AIRBORNE else corZs
        state_kernels = {
            Z.state_value: normalize_kernel(Z.Z)
            for Z in selected_kernels
            if Z.Z is not None and np.sum(Z) != 0
        }
        if not state_kernels:
            raise ValueError("No usable state kernels were generated.")
        return state_kernels

    def get_steps(self):
        return self.animal_proc.create_movement_data_dict(has_states=True)

    def generate_walks(self, out_dir=None, dt_tolerance=0.5, rnge=200, movement_policy=None, max_cell_size=10, water_mode:WaterMode=WaterMode.AVOID, is_brownian = False):
        t_col = self.original_data.t
        id_col = self.original_data.get_traj_id_col()

        py_kernels = self.get_kernels(dt_tolerance, rnge, out_dir, is_brownian)

        t_pol = FixedStepsPolicy(20) if movement_policy is None else movement_policy

        steps_dict = self.get_steps()
        per_animal_gdfs = []
        aid = 0
        for animal_id, trajectory in steps_dict.items():
            print(f"ANIMAL: {animal_id}\n")
            aid += 1
            steps: Trajectory = trajectory.df
            criterion = UTMDistanceCriterion.from_cell_grid(max_cell_size, self.resolution)
            base_segments = segment_dataframe(steps, criterion, merge_single_point_segments=True)
            steps = annotate_segments_dataframe(steps, segments=base_segments, segment_col="segment_id")
            # Overlapping segments are used for local terrain boxes, e.g. [(0, 3), (3, 5), (5, 9)].
            segments = make_overlapping(base_segments)

            animal_rows = []

            for segment in segments:
                min_lon, min_lat, max_lon, max_lat = bbox_of_segment(steps, segment)
                # convert bbox to UTM
                print(f"min_lon: {min_lon}, min_lat: {min_lat}, max_lon: {max_lon}, max_lat: {max_lat}")
                utm_bbox, zone, hemi, epsg_code, fwd, inv = padded_utm_bbox(
                    min_lon, min_lat, max_lon, max_lat,
                    padding=0.2,
                    max_cell_size=max_cell_size
                )
                print(f"utm_bbox: {utm_bbox}")
                min_utm_x, min_utm_y, max_utm_x, max_utm_y = utm_bbox
                # regular grid
                Nx, Ny = grid_shape_from_bbox(utm_bbox, self.resolution)
                # padded geo bbox
                min_lon, min_lat = inv.transform(min_utm_x, min_utm_y)
                max_lon, max_lat = inv.transform(max_utm_x, max_utm_y)
                # terrain for this segment
                terrain = None
                # sample landcover of new bounds
                if self.animal != Animal.AIRBORNE:
                    terrain = landcover_to_discrete_ptr(file_path=self.animal_proc.terrain_TIFFs[str(animal_id)],
                                                        res_x=Nx, res_y=Ny,
                                                        min_lon=min_lon, min_lat=min_lat,
                                                        max_lon=max_lon, max_lat=max_lat)
                cell_size = (max_utm_x - min_utm_x) / Nx
                cell_size = min(max(cell_size, 1.0), max_cell_size)

                # track segment boundaries so we can slice full_path per original segment
                seg_start, seg_end = segment
                for st_idx in range(seg_start, seg_end):
                    en_idx = st_idx + 1
                    print(f"[{aid-1} | {len(steps_dict)}] : ({st_idx} / {len(steps) - 1})\n")
                    start_lon = steps["geo_x"].iloc[st_idx]
                    start_lat = steps["geo_y"].iloc[st_idx]
                    end_lon = steps["geo_x"].iloc[en_idx]
                    end_lat = steps["geo_y"].iloc[en_idx]

                    start_time, end_time = steps["time"].iloc[st_idx], steps["time"].iloc[en_idx]
                    state = steps["state"].iloc[st_idx]

                    # start coordinates to UTM
                    st_utm_x, st_utm_y = fwd.transform(start_lon, start_lat)
                    en_utm_x, en_utm_y = fwd.transform(end_lon, end_lat)
                    # start, end GRID
                    start_x, start_y = utm_to_grid(
                        Nx, Ny, min_utm_x, min_utm_y, max_utm_x, max_utm_y,
                        st_utm_x, st_utm_y
                    )
                    end_x, end_y = utm_to_grid(
                        Nx, Ny, min_utm_x, min_utm_y, max_utm_x, max_utm_y,
                        en_utm_x, en_utm_y
                    )

                    assert start_x < Nx and start_y < Ny
                    assert end_x < Nx and end_y < Ny

                    if start_x == end_x and start_y == end_y:
                        animal_rows.append({
                            id_col: animal_id,
                            t_col: steps["time"].iloc[st_idx],
                            "geometry": Point(start_lon, start_lat),
                            "segment_id": int(steps["segment_id"].iloc[st_idx]),
                        })
                        continue

                    # Walker parameters
                    T, S = t_pol.resolve((start_x, start_y), (end_x, end_y), start_time, end_time)
                    T = int(np.ceil(T * 1.5))
                    D = 1 if is_brownian else 8

                    if self.animal == Animal.AIRBORNE or self.animal == Animal.MARINE:
                        target = 2 * S + 1
                        grid_kernel = resize(
                            py_kernels.get(state, next(iter(py_kernels.values()))),
                            (target, target),
                            order=1,
                            mode="reflect",
                            anti_aliasing=True,
                            preserve_range=True
                        )
                        grid_kernel = np.maximum(grid_kernel, 0)
                        grid_kernel /= grid_kernel.sum()
                    else:
                        # kernel parameters
                        kernel_radius = int(S * cell_size)
                        kernel_radius = min(rnge, kernel_radius)

                        clipped_kernel = normalize_kernel(
                            clip_kernel(py_kernels.get(state, next(iter(py_kernels.values()))), kernel_radius)
                        )
                        grid_kernel = resample_kernel_to_grid(clipped_kernel, cell_size, S)

                    h, w = grid_kernel.shape
                    print(f"W{w} : S: {S}\n")
                    assert w == 2 * S + 1 and h == 2 * S + 1
                    c_kernels = correlated_kernels_from_matrix(grid_kernel, w,h, directions=D)

                    print(f"[{start_time} - {end_time}]: {start_x}, {start_y} -> {end_x}, {end_y}: S - {S} T - {T} {Nx} x {Ny} - State {state}\n")

                    if self.animal is not Animal.AIRBORNE:
                        kmap = kernels_map_single_kernel(terrain, c_kernels, self.mapping, water_allowed=water_mode is not WaterMode.FORBID)
                        # Initialize DP matrix for the current start point
                        walk_ptr = single_state_walk(T,
                                                     kmap=kmap,
                                                     terrain=terrain,
                                                     start_x=start_x, start_y=start_y, end_x=end_x, end_y=end_y)
                        kernels_map3d_free(kmap)
                    else:
                        dp = correlated_walk_init(c_kernels, Nx, Ny,
                                                  T, start_x, start_y)
                        d = direction_from_points(start_x, start_y, end_x, end_y, D)

                        walk_ptr = correlated_backtrace(dp, T, c_kernels, end_x, end_y, d, out_ptr=True)
                        tensor4D_free(dp, T)
                        tensor_free(c_kernels)

                    if walk_ptr is not None:
                        walk_segment = get_walk_points(walk_ptr)
                        geo_walk = grid_to_geo_walk(walk_segment, utm_bbox, Nx, Ny, inv)
                        times = pd.date_range(
                            start=start_time,
                            end=end_time,
                            periods=len(geo_walk)
                        )
                        print("geo_walk sample:", geo_walk[:5])
                        for (lon, lat), t in zip(geo_walk, times):
                            animal_rows.append({
                                id_col: animal_id,
                                t_col: t,
                                "geometry": Point(lon, lat),
                                "segment_id": int(steps["segment_id"].iloc[st_idx]),
                            })
                        dll.point2d_array_free(walk_ptr)
                    else:
                        animal_rows.append({
                            id_col: animal_id,
                            t_col: steps["time"].iloc[st_idx],
                            "geometry": Point(start_lon, start_lat),
                            "segment_id": int(steps["segment_id"].iloc[st_idx]),
                        })

                terrain_map_free(terrain)

            animal_gdf = gpd.GeoDataFrame(animal_rows, geometry="geometry" ,crs="EPSG:4326")
            per_animal_gdfs.append(animal_gdf)

        # Combine all animals into a single GeoDataFrame and create one TrajectoryCollection
        combined_gdf = pd.concat(per_animal_gdfs, ignore_index=True)
        combined_gdf[t_col] = pd.to_datetime(combined_gdf[t_col])

        return merge_traj_collections(self.original_data, combined_gdf)
