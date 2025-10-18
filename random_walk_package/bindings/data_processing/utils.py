# Now use absolute imports
from ctypes import *

from random_walk_package.wrapper import dll

dll.calculate_ram_mib.argtypes = [
    c_ssize_t,  # D
    c_ssize_t,  # W
    c_ssize_t,  # H
    c_ssize_t,  # T
    c_bool  # is terrain walk?
]
dll.calculate_ram_mib.restype = c_double

dll.get_mem_available_mib.argtypes = []
dll.get_mem_available_mib.restype = c_double


def required_ram(D, W, H, T, is_terrain_walk) -> c_double:
    return dll.calculate_ram_mib(D, W, H, T, is_terrain_walk)


def available_ram() -> c_double:
    return dll.get_mem_available_mib()


def use_low_ram(D, W, H, T, is_terrain):
    return available_ram() < required_ram(D, W, H, T, is_terrain)
