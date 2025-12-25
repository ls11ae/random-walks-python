from ctypes import *

from random_walk_package.bindings.data_structures.types import EnvWeightProfilePtr
from random_walk_package.bindings import EnvWeightProfile
from random_walk_package import dll

dll.env_weights_new.argtypes = [c_bool, c_float, c_float, c_float, c_float, c_float]
dll.env_weights_new.restype = EnvWeightProfilePtr
dll.env_weights_free.argtypes = [EnvWeightProfilePtr]
dll.env_weights_free.restype = None


class EnvWeights:
    def __init__(self, *, override=False,
                 S=0.0, D=0.0, diffusivity=0.0,
                 bias_x=0.0, bias_y=0.0):
        # Enforces valid weight range of zero to one
        if S > 1.0 or D > 1.0 or diffusivity > 1.0 or bias_x > 1.0 or bias_y > 1.0 \
                or S < 0.0 or D < 0.0 or diffusivity < 0.0 or bias_x < 0.0 or bias_y < 0.0:
            raise ValueError("Weights must be in range [0.0, 1.0]")
        self._profile = EnvWeightProfile(override, S, D, diffusivity, bias_x, bias_y)
        self._ptr = dll.env_weights_new(c_bool(override),
                                        c_float(S),
                                        c_float(D),
                                        c_float(diffusivity),
                                        c_float(bias_x),
                                        c_float(bias_y))
        if not self._ptr:
            raise RuntimeError("Failed to allocate EnvWeightProfile")

    @property
    def ptr(self):
        return self._ptr

    def free(self):
        if self._ptr:
            dll.env_weights_free(self._ptr)
            self._ptr = None

    def __del__(self):
        self.free()

    @classmethod
    def full(cls):
        return cls(override=True, S=1.0, D=1.0, diffusivity=1.0, bias_x=1.0, bias_y=1.0)

    @classmethod
    def bias_only(cls):
        return cls(
            override=False,
            S=0.0,
            D=0.0,
            diffusivity=0.0,
            bias_x=1.0,
            bias_y=1.0,
        )

    @classmethod
    def custom(cls, *, override, S, D, diffusivity, bias_x, bias_y):
        return cls(override=override,
                   S=S, D=D,
                   diffusivity=diffusivity,
                   bias_x=bias_x,
                   bias_y=bias_y)
