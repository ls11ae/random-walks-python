import ctypes

from random_walk_package import Point2DArrayPtr
from random_walk_package.wrapper import dll

"""dll.backtrace_correlated_gpu_wrapped.argtypes = [ctypes.c_char_p,  # dp_path
                                                 ctypes.c_int32,  # T
                                                 ctypes.c_int32,  # S
                                                 ctypes.c_uint32,  # W
                                                 ctypes.c_uint32,  # H
                                                 ctypes.POINTER(ctypes.c_float),  # kernel
                                                 ctypes.c_int32,  # end_x
                                                 ctypes.c_int32,  # end_y
                                                 ctypes.c_int32,  # dir
                                                 ctypes.c_int32]  # D
dll.backtrace_correlated_gpu_wrapped.restype = Point2DArrayPtr"""
