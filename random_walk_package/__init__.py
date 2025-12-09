# random_walk_package/__init__.py
# --------------------------------------------------
# Load the shared library first
# --------------------------------------------------
# --------------------------------------------------
# Data processing helpers
# --------------------------------------------------
from .bindings.data_processing.movebank_parser import *
from .bindings.data_processing.walk_json import *
from .bindings.data_processing.weather_parser import *
# --------------------------------------------------
# Kernel / terrain helpers
# --------------------------------------------------
from .bindings.data_structures.kernel_terrain_mapping import (
    create_brownian_kernel_parameters,
    create_correlated_kernel_parameters,
    set_landmark_mapping,
    set_forbidden_landmark,
)
# --------------------------------------------------
# Matrix / tensor helpers
# --------------------------------------------------
from .bindings.data_structures.matrix import *
# --------------------------------------------------
# Point2D helpers
# --------------------------------------------------
from .bindings.data_structures.point2D import Point2DArrayPtr, point2d_arr_free
from .bindings.data_structures.tensor import *
from .wrapper import dll

# --------------------------------------------------
# Optional GPU helpers
# --------------------------------------------------
try:
    from .bindings.cuda.correlated_gpu import *
except ImportError:
    # GPU code not available; continue without raising
    pass

# --------------------------------------------------
# Re-export enums / constants for easier use
# --------------------------------------------------
from .bindings import MEDIUM, LIGHT, TREE_COVER, GRASSLAND, WATER, create_terrain_map

# --------------------------------------------------
# Core walkers
# --------------------------------------------------
from .core.BrownianWalker import BrownianWalker
from .core.MixedTimeWalker import MixedTimeWalker
from .core.MixedWalker import MixedWalker
from .core.BiasedWalker import BiasedWalker
from .core.AnimalMovementNew import AnimalMovementProcessor

# --------------------------------------------------
# Define __all__ for clean public API
# --------------------------------------------------
__all__ = [
    # DLL
    "dll",

    # Kernel / terrain helpers
    "create_terrain_map",
    "create_brownian_kernel_parameters",
    "create_correlated_kernel_parameters",
    "set_landmark_mapping",
    "set_forbidden_landmark",

    # Point2D helpers
    "Point2DArrayPtr",
    "point2d_arr_free",

    # Enums / constants
    "MEDIUM",
    "LIGHT",
    "GRASSLAND",
    "TREE_COVER",
    "WATER",

    # Walkers
    "BrownianWalker",
    "BiasedWalker",
    "MixedWalker",
    "MixedTimeWalker",
]
