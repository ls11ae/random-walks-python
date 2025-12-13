import datetime

from random_walk_package.bindings.data_structures.types import *
from random_walk_package.wrapper import dll

dll.parse_kernel_params.argtypes = [c_char_p, DateTimeIntervalPtr, POINTER(Dimensions3D)]
dll.parse_kernel_params.restype = EnvironmentInfluenceGridPtr

dll.get_kernels_environment_grid.argtypes = [c_int, TerrainMapPtr, EnvironmentInfluenceGridPtr,
                                             KernelParametersMappingPtr,
                                             c_float]
dll.get_kernels_environment_grid.restype = KernelParamsXYTPtr

dll.free_environment_influence_grid.argtypes = [EnvironmentInfluenceGridPtr]
dll.free_environment_influence_grid.restype = None

dll.free_kernel_parameters_yxt.argtypes = [KernelParamsXYTPtr]
dll.free_kernel_parameters_yxt.restype = None


def parse_kernel_parameters(csv_path: str, start_date: datetime, end_date: datetime,
                            dimensions: tuple[int, int, int]) -> EnvironmentInfluenceGridPtr:
    """
    Parses the kernel parameters for a specified time interval and dimensions, using data from a CSV file,
    and returns a pointer to an EnvironmentInfluenceGrid object.

    Arguments:
        csv_path: str
            Path to the CSV file containing the kernel parameters.
        start_date:
            The start date and time for the data interval.
        end_date:
            The end date and time for the data interval.
        dimensions: tuple[int, int, int]
            A tuple specifying the dimensions (y, x, t) for the environment grid. (Must be in that order!)

    Returns:
        EnvironmentInfluenceGridPtr
            A pointer to the EnvironmentInfluenceGrid, containing the parsed parameters.
    """
    start_dt = DateTime(start_date.year, start_date.month, start_date.day, start_date.hour)
    end_dt = DateTime(end_date.year, end_date.month, end_date.day, end_date.hour)
    interval_ptr = pointer(DateTimeInterval(start_dt, end_dt))
    dimensions_ptr = pointer(Dimensions3D(*dimensions))
    return dll.parse_kernel_params(csv_path.encode('utf-8'), interval_ptr, dimensions_ptr)


def free_environment_influence_grid(influence_grid: EnvironmentInfluenceGridPtr):
    dll.free_environment_influence_grid(influence_grid)


def get_kernels_environment_grid(times: int, terrain_map: TerrainMapPtr, influence_grid: EnvironmentInfluenceGridPtr,
                                 mapping: KernelParametersMappingPtr, environment_weight: float) -> KernelParamsXYTPtr:
    return dll.get_kernels_environment_grid(times, terrain_map, influence_grid, mapping, environment_weight)


def free_kernel_parameters_yxt(kernel_params: KernelParamsXYTPtr):
    dll.free_kernel_parameters_yxt(kernel_params)
