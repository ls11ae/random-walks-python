import subprocess
from pathlib import Path

from random_walk_package.bindings.mixed_walk import *
from random_walk_package.bindings.plotter import plot_combined_terrain
from random_walk_package.core.AnimalMovement import AnimalMovementProcessor
from random_walk_package.core.WalkerHelper import WalkerHelper
from random_walk_package.data_sources.walk_visualization import walk_to_osm

try:
    from random_walk_package.bindings.cuda.mixed_gpu import preprocess_mixed_gpu, mixed_walk_gpu, free_kernel_pool

    CUDA_AVAILABLE = True
except (AttributeError, ImportError, OSError) as e:
    print(f"CUDA not available: {e}")
    CUDA_AVAILABLE = False


    def preprocess_mixed_gpu(*args, **kwargs):
        print("CUDA not available - using CPU fallback")
        return None


    def mixed_walk_gpu(*args, **kwargs):
        raise RuntimeError("CUDA not available on this system")


    def free_kernel_pool(*args, **kwargs):
        pass


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
        self._process_movebank_data()

    def _process_movebank_data(self):
        self.movebank_processor = AnimalMovementProcessor(self.movebank_study)
        self.aid_to_terrain_path = self.movebank_processor.create_landcover_data_txt(self.resolution,
                                                                                     out_directory=self.study_folder)

    @staticmethod
    def has_cuda():
        try:
            out = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
            return b"CUDA" in out or b"NVIDIA" in out
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def generate_movebank_walks(self, serialized=False):
        use_cuda = False  # self.has_cuda()
        grid_steps_dict, geo_steps_dict, times = self.movebank_processor.create_movement_data(samples=-1)

        recmp: bool = True
        serialization_dir = Path(self.base_project_dir) / 'resources' / self.serialization_path / 'tensors'

        geodetic_walks: dict[str, list[tuple[float, float]]] = {}

        if serialization_dir.exists() and any(serialization_dir.iterdir()):
            recmp = False
        for animal_id, steps in grid_steps_dict.items():
            spatial_map = parse_terrain(file=self.aid_to_terrain_path[animal_id], delim=' ')
            print(f"Loaded terrain from {self.aid_to_terrain_path[animal_id]}")
            if serialized and recmp:
                tensor_map_terrain_serialize(spatial_map, self.mapping, self.serialization_path)
                print(f"Serialized terrain map to {self.serialization_path}")
            else:
                print("create kernels")
                self.tensor_map = get_tensor_map_terrain(spatial_map, self.mapping)
                print("create tensor map")

            kernel_pool = preprocess_mixed_gpu(self.tensor_map, spatial_map) if use_cuda else None

            width = spatial_map.width
            height = spatial_map.height
            full_path = []
            for i in range(len(steps) - 1):
                start_x, start_y = steps[i]
                end_x, end_y = steps[i + 1]
                print("Start: " + str(start_x) + ", " + str(start_y))
                print(start_x, start_y, end_x, end_y)

                if start_x == end_x and start_y == end_y:
                    full_path.extend([(start_x, start_y)])
                    continue

                print(self.serialization_path)
                dp_dir = os.path.join(self.base_project_dir, 'resources', self.serialization_path,
                                      "DP_T" + str(self.T) + "_X" + str(start_x) + "_Y" + str(start_y))
                print(f"Recomputing {recmp}")
                # Initialize DP matrix for the current start point
                t = abs(start_x - end_x) + abs(start_y - end_y)
                self.T = 5 if t < 5 else t
                print(f"Setting T to {self.T}")
                if use_cuda:
                    walk_ptr = mixed_walk_gpu(self.T, width, height, start_x, start_y, end_x, end_y, self.tensor_map,
                                              self.mapping, spatial_map, False, "", kernel_pool)
                else:
                    dp_matrix_step = mix_walk(W=width, H=height, terrain_map=spatial_map, kernels_map=self.tensor_map,
                                              start_x=int(start_x), start_y=int(start_y), T=self.T,
                                              serialize=serialized,
                                              recompute=recmp,
                                              serialize_path=self.serialization_path, mapping=self.mapping)
                    if serialized:
                        print(dp_dir)
                    # Backtrace from the end point
                    walk_ptr = mix_backtrace_c(
                        DP_Matrix=dp_matrix_step,
                        T=self.T,
                        tensor_map=self.tensor_map,
                        terrain=spatial_map,
                        end_x=int(end_x),
                        end_y=int(end_y),
                        serialize=serialized,
                        serialize_path=self.serialization_path,
                        dp_dir=dp_dir,
                        mapping=self.mapping
                    )
                if walk_ptr is not None:
                    segment = get_walk_points(walk_ptr)
                else:
                    segment = [(start_x, start_y), (end_x, end_y)]
                # Cleanup C memory
                if not serialized and not use_cuda:
                    dll.tensor4D_free(dp_matrix_step, self.T)
                dll.point2d_array_free(walk_ptr)

                # Concatenate paths (skip duplicate point)
                full_path.extend(segment[:-1])

            print(
                f"Finished walking {animal_id} from {steps[0]} to {steps[-1]} with {len(steps)} steps."
            )
            free_kernel_pool(kernel_pool)
            # Add final point
            full_path.append(steps[-1])
            grid_steps_dict[animal_id] = self.movebank_processor.grid_coordinates_to_geodetic(steps, animal_id)
            geodetic_path = self.movebank_processor.grid_coordinates_to_geodetic(full_path, animal_id)
            geodetic_walks[animal_id] = geodetic_path
            walk_to_osm(walk_coords_or_dict=geodetic_path, original_coords=geo_steps_dict[animal_id],
                        step_annotations=grid_steps_dict,
                        animal_id=animal_id, walk_path=self.walks_path, annotated=True)
            kernels_map3d_free(self.tensor_map)
        map_path = os.path.join(self.walks_path, "entire_study.html")
        walk_to_osm(geodetic_walks, None, "entire study", self.walks_path, grid_steps_dict, map_path)
        return map_path

    @staticmethod
    def generate_custom_walks(terrain, steps, T, kernel_mapping, plot=False, plot_title="Mixed Walk"):
        tensor_map = get_tensor_map_terrain(terrain, kernel_mapping)
        walk = WalkerHelper.generate_multistep_walk(terrain, steps, T, kernel_mapping, tensor_map)
        kernels_map3d_free(tensor_map)
        if plot:
            plot_combined_terrain(terrain=terrain, walk_points=walk, steps=steps, title="Mixed Walk")
        return walk
