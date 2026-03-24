# random_walk_package/__init__.py
# --------------------------------------------------
# Load the shared library first
# --------------------------------------------------
from .utils.move_apps_patch import debug_patch_state, apply_moveapps_id_dtype_patch
from .wrapper import dll
# --------------------------------------------------
# Data processing helpers
# --------------------------------------------------
from .bindings.data_processing.movebank_parser import *
from .bindings.data_processing.walk_json import *
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
from .bindings.data_structures.point2D import *
# --------------------------------------------------
# Point2D helpers
# --------------------------------------------------
from .bindings.data_structures.point2D import Point2DArrayPtr, point2d_arr_free
from .bindings.data_structures.point2D import get_walk_points
from .bindings.data_structures.tensor import *

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
from .bindings import (TREE_COVER, GRASSLAND, WATER, create_terrain_map, Animal, WaterMode, MovementPolicyCfg)

#---------------------------------------------------
# Movement Policies
#---------------------------------------------------
from random_walk_package.core.MovementPolicy import TimeStepPolicy, SpeedBasedPolicy, MovementPolicy, FixedStepsPolicy

#---------------------------------------------------
# Serialization
#---------------------------------------------------
from random_walk_package.data_sources.walk_visualization import save_trajectory_collection_timed


# --------------------------------------------------
# Core walkers
# --------------------------------------------------
from .core.AnimalMovement import AnimalMovementProcessor
from .core.BrownianWalker import BrownianWalker
from .core.MixedTimeWalker import MixedTimeWalker
from .core.MixedWalker import MixedWalker
from .core.BiasedWalker import BiasedWalker
from .core.StateDependentWalker import StateDependentWalker

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
    "get_walk_points",

    # Enums / constants
    WaterMode,
    Animal,
    MovementPolicyCfg,
    "GRASSLAND",
    "TREE_COVER",
    "WATER",

    # MovementPolicies
    "TimeStepPolicy",
    "FixedStepsPolicy",
    "SpeedBasedPolicy",
    "MovementPolicyCfg",
    "MovementPolicy",

    # Serialization
    "save_trajectory_collection_timed",

    # Walkers
    "BrownianWalker",
    "BiasedWalker",
    "MixedWalker",
    "MixedTimeWalker",
    "StateDependentWalker",

    "debug_patch_state",
    "apply_moveapps_id_dtype_patch"
]
