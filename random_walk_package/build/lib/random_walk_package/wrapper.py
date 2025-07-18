import os
import ctypes
import sys
from pathlib import Path
from memory_profiler import profile  # Import memory profiler

# Get the absolute path to the current file (random_walk_package/random_walk_package/)
_base_dir = Path(__file__).resolve().parent

# Go one directory up to reach the folder where the .so file is copied
_lib_dir = _base_dir.parent.parent

# Build the full path to the shared library
_so_path = Path(__file__).resolve().parents[3] / "build/lib/librandom_walk.so"

# Load the shared library
dll = ctypes.CDLL(str(_so_path))

# Revert stdout to original (ensure normal text output)
sys.stdout = sys.__stdout__
