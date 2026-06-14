from randomwalks.bindings.data_structures.types import *
from randomwalks.wrapper import dll


class EnvWeights:
    dll.env_weights_new.argtypes = [c_bool, c_float, c_float, c_float, c_float, c_float, c_float]
    dll.env_weights_new.restype = EnvWeightProfilePtr
    new = dll.env_weights_new

    dll.env_weights_free.argtypes = [EnvWeightProfilePtr]
    dll.env_weights_free.restype = None
    free_ptr = dll.env_weights_free

    def __init__(
        self,
        *,
        override=False,
        S=0.0,
        D=0.0,
        len_diffusivity=0.0,
        angle_diffusivity=0.0,
        bias_x=0.0,
        bias_y=0.0,
    ):
        weights = (S, D, len_diffusivity, angle_diffusivity, bias_x, bias_y)
        if any(weight < 0.0 or weight > 1.0 for weight in weights):
            raise ValueError("Weights must be in range [0.0, 1.0]")

        self._ptr = self.new(
            c_bool(override),
            c_float(S),
            c_float(D),
            c_float(len_diffusivity),
            c_float(angle_diffusivity),
            c_float(bias_x),
            c_float(bias_y),
        )
        if not self._ptr:
            raise RuntimeError("Failed to allocate EnvWeightProfile")

    @property
    def ptr(self):
        return self._ptr

    def free(self):
        if self._ptr:
            self.free_ptr(self._ptr)
            self._ptr = None

    @classmethod
    def full(cls):
        return cls(
            override=True,
            S=1.0,
            D=1.0,
            len_diffusivity=0.0,
            angle_diffusivity=0.0,
            bias_x=1.0,
            bias_y=1.0,
        )

    @classmethod
    def bias_only(cls):
        return cls(
            override=False,
            S=0.0,
            D=0.0,
            len_diffusivity=0.0,
            angle_diffusivity=0.0,
            bias_x=1.0,
            bias_y=1.0,
        )

    @classmethod
    def custom(cls, *, override, S, D, length_diffusivity, angle_diffusivity, bias_x, bias_y):
        return cls(
            override=override,
            S=S,
            D=D,
            len_diffusivity=length_diffusivity,
            angle_diffusivity=angle_diffusivity,
            bias_x=bias_x,
            bias_y=bias_y,
        )

    def __del__(self):
        self.free()


__all__ = ["EnvWeights"]
