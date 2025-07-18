from .wrapper import dll  # Import dll first

# Import other modules after dll is available
from random_walk_package.bindings.brownian_walk import *
from random_walk_package.bindings.correlated_walk import *
from random_walk_package.bindings.data_structures.matrix import *
from random_walk_package.bindings.data_structures.point2D import *
from random_walk_package.bindings.data_structures.tensor import *
from random_walk_package.bindings.data_processing.weather_parser import *
from random_walk_package.bindings.data_processing.movebank_parser import *
from random_walk_package.bindings.data_processing.walk_json import *