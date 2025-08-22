from random_walk_package.bindings.data_structures.types import *
from ctypes import *
from random_walk_package.wrapper import dll

# steps, terrain
dll.save_walk_to_json.argtypes = [Point2DArrayPtr, Point2DArrayPtr, TerrainMapPtr, c_char_p]
dll.save_walk_to_json.restype = None

# no steps, terrain
dll.save_walk_to_json_nosteps.argtypes = [Point2DArrayPtr, TerrainMapPtr, c_char_p]
dll.save_walk_to_json_nosteps.restype = None

# steps, no terrain
dll.save_walk_to_json_noterrain.argtypes = [Point2DArrayPtr, Point2DArrayPtr, c_size_t, c_size_t,
                                            c_char_p]
dll.save_walk_to_json_noterrain.restype = None

# no steps, no terrain
dll.save_walk_to_json_onlywalk.argtypes = [Point2DArrayPtr, c_size_t, c_size_t, c_char_p]
dll.save_walk_to_json_onlywalk.restype = None

# load from file
dll.load_full_walk.argtypes = [c_char_p, Point2DArrayPtr, Point2DArrayPtr, TerrainMapPtr]
dll.load_full_walk.restype = None

dll.load_walk_with_terrain.argtypes = [c_char_p, Point2DArrayPtr, TerrainMapPtr]
dll.load_walk_with_terrain.restype = None

dll.load_walk_with_steps.argtypes = [c_char_p, Point2DArrayPtr, Point2DArrayPtr]
dll.load_walk_with_steps.restype = None

dll.load_walk_only.argtypes = [c_char_p, Point2DArrayPtr]
dll.load_walk_only.restype = None


def walk_to_json(walk, json_file: str, steps=None, terrain_map=None, W=None, H=None):
    file = c_char_p(json_file.encode('ascii'))

    if terrain_map is not None and steps is not None:
        dll.save_walk_to_json(steps, walk, terrain_map, file)
    elif terrain_map is not None and steps is None:
        dll.save_walk_to_json_nosteps(walk, terrain_map, file)
    elif terrain_map is None and steps is not None:
        dll.save_walk_to_json_noterrain(steps, walk, W, H, file)
    elif terrain_map is None and steps is None:
        dll.save_walk_to_json_onlywalk(walk, W, H, file)

    print("Walk saved to {}".format(json_file))


def walk_from_json(json_file, steps=None, walk=None, terrain_map=None):
    file = c_char_p(json_file.encode('ascii'))

    if terrain_map is not None and steps is not None:
        dll.load_full_walk(file, steps, steps, walk, terrain_map)
    elif terrain_map is not None and steps is None:
        dll.load_walk_with_terrain(file, walk, terrain_map)
    elif terrain_map is None and steps is not None:
        dll.load_walk_with_steps(file, steps, walk)
    elif terrain_map is None and steps is None:
        dll.load_walk_only(file, walk)

    print("Walk loaded from {}".format(json_file))
