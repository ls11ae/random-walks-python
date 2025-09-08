from pathlib import Path

import numpy as np

from random_walk_package import walk_to_json
from random_walk_package.bindings.data_structures.point2D import get_walk_points
from random_walk_package.bindings.mixed_walk import *
from random_walk_package.core.AnimalMovement import AnimalMovementProcessor
from random_walk_package.data_sources.walk_visualization import walk_to_osm


class MixedWalker:
    def __init__(self, T=30, S=9, animal_type=MEDIUM, resolution=100, kernel_mapping=None, study_folder=None):
        self.T = T
        self.resolution = resolution
        self.mapping = kernel_mapping if kernel_mapping is not None else create_mixed_kernel_parameters(animal_type, S)
        self.tensor_map = None
        self.movebank_processor = None

        self.script_dir = os.path.dirname(os.path.realpath(__file__))
        self.base_project_dir = os.path.abspath(os.path.join(self.script_dir, '..'))

        self.study_folder = os.path.join(self.base_project_dir, 'resources')
        if study_folder:
            self.study_folder = os.path.join(self.study_folder, study_folder)

        print(f"Using study folder: {self.study_folder}")

        # Find the only CSV file in the study_folder
        csv_files = [f for f in os.listdir(self.study_folder) if f.endswith('.csv')]
        if len(csv_files) != 1:
            raise FileNotFoundError("Expected exactly one CSV file in the study folder.")
        self.movebank_study = os.path.join(self.study_folder, csv_files[0])

        self.serialization_path = os.path.join(self.study_folder, 'serialization')
        if not os.path.exists(self.serialization_path):
            os.makedirs(self.serialization_path)

        # Create walks folder in study_folder and set as walks_path
        self.walks_path = os.path.join(self.study_folder, 'walks')
        if not os.path.exists(self.walks_path):
            os.makedirs(self.walks_path)

        self.aid_to_terrain_path = self.study_folder
        self._set_kernels()

    def _set_kernels(self):
        self.movebank_processor = AnimalMovementProcessor(self.movebank_study)
        self.aid_to_terrain_path = self.movebank_processor.create_landcover_data_txt(self.resolution,
                                                                                     out_directory=self.study_folder)

    def _as_terrain_ptr(self, t):
        if isinstance(t, TerrainMap):
            return pointer(t)
        return t

    def generate_walk(self, step_size_meters=1, serialized=False):
        steps_dict, steps_utm_dict = self.movebank_processor.create_movement_data(samples=-1)

        recmp: bool = True
        serialization_dir = Path(self.serialization_path) / 'tensors'

        if serialization_dir.exists() and any(serialization_dir.iterdir()):
            recmp = False

        for animal_id, steps in steps_dict.items():
            steps_utm = steps_utm_dict[animal_id]
            print(steps_utm)

            cell_size_in_meters = self.movebank_processor.cell_size_of(animal_id)
            print(f"Cell size: {cell_size_in_meters}")
            print(f"GRID: Starting to walk {animal_id} from {steps[0]} to {steps[-1]} with {len(steps)} steps.")

            # Parse terrain once
            spatial_map = parse_terrain(file=self.aid_to_terrain_path[animal_id][0], delim=' ')
            aspect_ratio = spatial_map.width / spatial_map.height
            utm_x1, utm_y1, utm_x2, utm_y2 = self.movebank_processor.bbox_utm_of(animal_id)

            # Tensor map vorbereiten (serialisiert oder im RAM)
            if serialized and recmp:
                tensor_map_terrain_serialize(spatial_map, self.mapping, self.serialization_path)
                print(f"Serialized terrain map to {self.serialization_path}")
            else:
                print("create kernels")
                self.tensor_map = get_tensor_map_terrain(spatial_map, self.mapping)
                print("create tensor map")

            full_path = []

            MAX_T = 600  # harte Obergrenze für Speicher
            MIN_T_FOR_UPSCALE = 3  # nur wenn Bewegung klein ist, aber nicht trivial

            for i in range(len(steps) - 1):
                start_x, start_y = steps[i]
                end_x, end_y = steps[i + 1]
                current_terrain = spatial_map
                current_tensor_map = self.tensor_map
                # Fix: use position-based indexing for pandas Series, fallback for list/array
                if hasattr(steps_utm, "iloc"):
                    start_x_utm, start_y_utm = steps_utm.iloc[i]
                    end_x_utm, end_y_utm = steps_utm.iloc[i + 1]
                else:
                    start_x_utm, start_y_utm = steps_utm[i]
                    end_x_utm, end_y_utm = steps_utm[i + 1]

                dist = np.sqrt((start_x_utm - end_x_utm) ** 2 + (start_y_utm - end_y_utm) ** 2)

                # Skip-Heuristik: kleine Distanzen < step_size_meters einfach übernehmen
                if dist < step_size_meters:
                    full_path.extend([(start_x, start_y)])
                    continue

                # T bestimmen
                n = int(np.ceil(dist / step_size_meters))
                print(f"n steps: {n}")
                if n > MAX_T:
                    print(f"T={n} capped to {MAX_T} (distance {dist:.1f} too large for fine walk)")
                    n = MAX_T
                    use_upscale = False
                elif MIN_T_FOR_UPSCALE <= n:
                    print(f"Extract terrain from {start_x_utm}x{start_y_utm} to {end_x_utm}x{end_y_utm}")
                    print(
                        f"Extract terrain args: file_path={self.aid_to_terrain_path[animal_id][1]}, x1={start_x_utm}, y1={start_y_utm}, x2={end_x_utm}, y2={end_y_utm}, res_x={n}, res_y={int(n / aspect_ratio)}, padding=0.1")
                    current_terrain, (min_x, min_y, max_x, max_y), start, end = extract_terrain_map(
                        file_path=self.aid_to_terrain_path[animal_id][1],
                        x1=start_x_utm, y1=start_y_utm, x2=end_x_utm,
                        y2=end_y_utm,
                        res_x=3 * n, res_y=int(3 * n / aspect_ratio),
                        padding=0.15)
                    start_x = start[0]
                    start_y = start[1]
                    end_x = end[0]
                    end_y = end[1]
                    print(f"Start: {start}, End: {end}")
                    print(f"Terrain upscaled to {current_terrain.contents.width}x{current_terrain.contents.height}")
                    current_tensor_map = get_tensor_map_terrain(current_terrain, self.mapping)
                    use_upscale = True
                else:
                    use_upscale = False
                    full_path.extend([(start_x, start_y)])
                    continue

                self.T = n
                print(f"UTM: Starting to walk {animal_id} from {start_x_utm}, {start_y_utm} "
                      f"to {end_x_utm}, {end_y_utm} with distance {dist:.2f} and time {self.T} "
                      f"(upscale={use_upscale})")

                ptr = self._as_terrain_ptr(current_terrain)
                W = ptr.contents.width
                H = ptr.contents.height

                dp_matrix_step = mix_walk(
                    W=W, H=H,
                    terrain_map=ptr,
                    kernels_map=current_tensor_map,
                    start_x=int(start_x), start_y=int(start_y),
                    T=self.T,
                    serialize=serialized, recompute=recmp,
                    serialize_path=self.serialization_path,
                    mapping=self.mapping
                )

                # Backtrace from the end point
                walk_ptr = mix_backtrace(
                    DP_Matrix=dp_matrix_step,
                    T=self.T,
                    tensor_map=current_tensor_map,
                    terrain=ptr,
                    end_x=int(end_x),
                    end_y=int(end_y),
                    directory=0,
                    serialize=serialized,
                    serialize_path=self.serialization_path,
                    dp_dir="dp_dir",
                    mapping=self.mapping
                )

                if walk_ptr.contents:
                    segment = get_walk_points(walk_ptr)
                else:
                    segment = [(start_x, start_y), (end_x, end_y)]

                # Cleanup C memory
                if not serialized:
                    dll.tensor4D_free(dp_matrix_step, self.T)
                dll.point2d_array_free(walk_ptr)
                if use_upscale:
                    segment = self.movebank_processor.grid_coordinates_to_utm(
                        bbox_utm=(min_x, min_y, max_x, max_y),
                        width=current_terrain.contents.width,
                        height=current_terrain.contents.height,
                        coord=segment, animal_id=animal_id)
                    terrain_map_free(current_terrain)
                    kernels_map3d_free(current_tensor_map)
                else:
                    segment = self.movebank_processor.grid_coordinates_to_utm(
                        bbox_utm=(utm_x1, utm_y1, utm_x2, utm_y2),
                        width=current_terrain.contents.width,
                        height=current_terrain.contents.height,
                        coord=segment, animal_id=animal_id)

                # Concatenate paths (skip duplicate point)
                full_path.extend(segment[:-1])

            print(f"Finished walking {animal_id} from {steps[0]} to {steps[-1]} with {len(steps)} steps.")
            # Add final point
            full_path.append(steps[-1])

            # Postprocessing
            geodetic_path = self.movebank_processor.grid_coordinates_to_geodetic(full_path, animal_id)
            walk_to_osm(geodetic_path, animal_id, self.walks_path)
            kernels_map3d_free(self.tensor_map)

            walk = np.array(full_path)
            steps_c = create_point2d_array(steps)
            walk_c = create_point2d_array(walk)
            walk_to_json(
                walk_c,
                json_file=os.path.join(self.walks_path, f"{animal_id}_{self.resolution}.json"),
                steps=steps_c,
                terrain_map=pointer(spatial_map)
            )
            # plot_combined_terrain(pointer(spatial_map), walk, steps=steps, title=self.movebank_study)
