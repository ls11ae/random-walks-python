from importlib.resources import as_file, files

from randomwalks.bindings.data_structures.types import *
from randomwalks.wrapper import dll

KPM_KIND_PARAMETERS = int(KernelMapKind.PARAMETERS)
KPM_KIND_KERNELS = int(KernelMapKind.KERNELS)
MESA_DEFAULT_MAPPING_RESOURCE = "resources/kernel_mappings/mesa_mixed_terrestrial.csv"
RESOURCE_PACKAGE = "randomwalks"


class KernelMapping:
    dll.kernel_mapping_new.argtypes = [TerrainMapPtr, c_int]
    dll.kernel_mapping_new.restype = KernelParametersMappingPtr
    new = dll.kernel_mapping_new

    dll.kernel_mapping_load_csv.argtypes = [KernelParametersMappingPtr, c_char_p]
    dll.kernel_mapping_load_csv.restype = c_bool
    load_csv_ptr = dll.kernel_mapping_load_csv

    dll.set_terrain_params.argtypes = [KernelParametersMappingPtr, c_int, KernelParametersPtr]
    dll.set_terrain_params.restype = c_bool
    set_params_ptr = dll.set_terrain_params

    dll.set_terrain_kernel.argtypes = [KernelParametersMappingPtr, c_int, MatrixPtr, c_ssize_t]
    dll.set_terrain_kernel.restype = c_bool
    set_kernel_ptr = dll.set_terrain_kernel

    dll.set_terrain_barrier.argtypes = [KernelParametersMappingPtr, c_int, c_bool]
    dll.set_terrain_barrier.restype = c_bool
    set_barrier_ptr = dll.set_terrain_barrier

    dll.set_terrain_unmapped.argtypes = [KernelParametersMappingPtr, c_int, c_bool]
    dll.set_terrain_unmapped.restype = c_bool
    set_unmapped_ptr = dll.set_terrain_unmapped

    dll.set_terrain_weight.argtypes = [KernelParametersMappingPtr, c_int, c_int, c_double]
    dll.set_terrain_weight.restype = c_bool
    set_weight_ptr = dll.set_terrain_weight

    dll.terrain_weight.argtypes = [KernelParametersMappingPtr, c_int, c_int]
    dll.terrain_weight.restype = c_double
    weight_ptr = dll.terrain_weight

    dll.terrain_stay_weight.argtypes = [KernelParametersMappingPtr, c_int]
    dll.terrain_stay_weight.restype = c_double
    stay_weight_ptr = dll.terrain_stay_weight

    dll.terrain_to_mapping_index.argtypes = [KernelParametersMappingPtr, c_int]
    dll.terrain_to_mapping_index.restype = c_int
    terrain_to_index_ptr = dll.terrain_to_mapping_index

    dll.mapping_index_to_terrain.argtypes = [KernelParametersMappingPtr, c_size_t]
    dll.mapping_index_to_terrain.restype = c_int
    index_to_terrain_ptr = dll.mapping_index_to_terrain

    dll.is_barrier_terrain.argtypes = [c_int, KernelParametersMappingPtr]
    dll.is_barrier_terrain.restype = c_bool
    is_barrier_ptr = dll.is_barrier_terrain

    dll.is_unmapped_terrain.argtypes = [c_int, KernelParametersMappingPtr]
    dll.is_unmapped_terrain.restype = c_bool
    is_unmapped_ptr = dll.is_unmapped_terrain

    dll.terrain_params.argtypes = [KernelParametersMappingPtr, c_int]
    dll.terrain_params.restype = KernelParametersPtr
    params_ptr = dll.terrain_params

    dll.terrain_params_const.argtypes = [KernelParametersMappingPtr, c_int]
    dll.terrain_params_const.restype = KernelParametersPtr
    params_const_ptr = dll.terrain_params_const

    dll.kernel_mapping_free.argtypes = [KernelParametersMappingPtr]
    dll.kernel_mapping_free.restype = None
    free_ptr = dll.kernel_mapping_free

    def __init__(self, terrain=None, kind=KPM_KIND_PARAMETERS, *, ptr=None, owned=True):
        if ptr is None:
            if terrain is None:
                raise ValueError("terrain is required when ptr is not provided")
            ptr = self.new(_terrain_ptr(terrain), int(kind))
        if not ptr:
            raise RuntimeError("Failed to allocate KernelParametersMapping")
        self._ptr = ptr
        self._owned = owned
        self._terrain = terrain

    @classmethod
    def from_ptr(cls, ptr, *, owned=False, terrain=None):
        return cls(ptr=ptr, owned=owned, terrain=terrain)

    @classmethod
    def mesa_default(cls, terrain, resource=MESA_DEFAULT_MAPPING_RESOURCE):
        mapping = cls(terrain)
        if not mapping.load_resource(resource):
            mapping.free()
            raise ValueError(f"Failed to load kernel mapping resource '{resource}'")
        return mapping

    @classmethod
    def uniform(cls, terrain, *, is_brownian, step_size, directions,
                len_diffusity=1.0, angle_diffusity=0.3, max_bias_x=0, max_bias_y=0):
        mapping = cls(terrain)
        for terrain_value in mapping.terrain_values:
            mapping.set_parameters(
                terrain_value,
                is_brownian=is_brownian,
                step_size=step_size,
                directions=directions,
                len_diffusity=len_diffusity,
                angle_diffusity=angle_diffusity,
                max_bias_x=max_bias_x,
                max_bias_y=max_bias_y,
            )
        return mapping

    @property
    def ptr(self):
        return self._ptr

    @property
    def contents(self):
        return self._ptr.contents

    @property
    def terrain(self):
        return self._terrain

    @property
    def terrain_values(self):
        contents = self.contents
        return [int(contents.terrain_values[i]) for i in range(contents.terrain_count)]

    def load_csv(self, filename):
        return bool(self.load_csv_ptr(self._ptr, filename.encode("utf-8")))

    def load_resource(self, resource=MESA_DEFAULT_MAPPING_RESOURCE, package=RESOURCE_PACKAGE):
        resource_ref = files(package)
        for part in resource.split("/"):
            resource_ref = resource_ref.joinpath(part)
        with as_file(resource_ref) as path:
            return self.load_csv(str(path))

    def set_parameters(self, terrain, *, is_brownian, step_size, directions,
                       len_diffusity=1.0, angle_diffusity=0.3,
                       max_bias_x=0, max_bias_y=0):
        if is_brownian:
            directions = 1
        params = KernelParameters(
            bool(is_brownian),
            step_size,
            directions,
            len_diffusity,
            angle_diffusity,
            max_bias_x,
            max_bias_y,
        )
        return self.set_params(terrain, params)

    def set_params(self, terrain, params):
        if isinstance(params, KernelParameters):
            params = byref(params)
        return bool(self.set_params_ptr(self._ptr, int(terrain), params))

    def set_kernel(self, terrain, kernel, directions=1):
        return bool(self.set_kernel_ptr(self._ptr, int(terrain), _matrix_ptr(kernel), directions))

    def set_barrier(self, terrain, barrier=True):
        return bool(self.set_barrier_ptr(self._ptr, int(terrain), barrier))

    def is_barrier(self, terrain):
        return bool(self.is_barrier_ptr(int(terrain), self._ptr))

    def set_unmapped(self, terrain, unmapped=True):
        return bool(self.set_unmapped_ptr(self._ptr, int(terrain), unmapped))

    def is_unmapped(self, terrain):
        return bool(self.is_unmapped_ptr(int(terrain), self._ptr))

    def set_weight(self, from_terrain, to_terrain, weight):
        return bool(self.set_weight_ptr(self._ptr, int(from_terrain), int(to_terrain), weight))

    def weight(self, from_terrain, to_terrain):
        return float(self.weight_ptr(self._ptr, int(from_terrain), int(to_terrain)))

    def stay_weight(self, terrain):
        return float(self.stay_weight_ptr(self._ptr, int(terrain)))

    def terrain_to_index(self, terrain):
        return int(self.terrain_to_index_ptr(self._ptr, int(terrain)))

    def index_to_terrain(self, index):
        return int(self.index_to_terrain_ptr(self._ptr, index))

    def params(self, terrain):
        return self.params_ptr(self._ptr, int(terrain))

    def free(self):
        if self._owned and self._ptr:
            self.free_ptr(self._ptr)
        self._ptr = None

    def __bool__(self):
        return bool(self._ptr)

    def __del__(self):
        self.free()


def _terrain_ptr(terrain):
    return terrain.ptr if hasattr(terrain, "ptr") else terrain


def _mapping_ptr(mapping):
    return mapping.ptr if hasattr(mapping, "ptr") else mapping


def _matrix_ptr(matrix):
    return matrix.ptr if hasattr(matrix, "ptr") else matrix
