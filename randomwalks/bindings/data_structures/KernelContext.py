from pathlib import Path

from randomwalks.bindings.data_structures.KernelMapping import KernelMapping
from randomwalks.bindings.data_structures.types import *
from randomwalks.wrapper import dll


class KernelContextHandle:
    dll.kernel_context_on_fly.argtypes = [TerrainMapPtr, KernelParametersMappingPtr, c_int]
    dll.kernel_context_on_fly.restype = KernelContextPtr
    on_fly_ptr = dll.kernel_context_on_fly

    dll.kernel_context_pool.argtypes = [TerrainMapPtr, KernelParametersMappingPtr, c_int]
    dll.kernel_context_pool.restype = KernelContextPtr
    pool_ptr = dll.kernel_context_pool

    dll.kernel_context_serialization.argtypes = [TerrainMapPtr, KernelParametersMappingPtr, c_int, c_char_p]
    dll.kernel_context_serialization.restype = KernelContextPtr
    serialization_ptr = dll.kernel_context_serialization

    dll.kernel_context_free.argtypes = [KernelContextPtr]
    dll.kernel_context_free.restype = None
    free_ptr = dll.kernel_context_free

    def __init__(self, *, ptr, owned=True, mapping: KernelMapping | None = None):
        if not ptr:
            raise RuntimeError("Failed to allocate KernelContext")
        self._ptr = ptr
        self._owned = owned
        self._mapping = mapping
        self._terrain = mapping.terrain if mapping else None

    @classmethod
    def on_fly(cls, mapping, reachability):
        ptr = cls.on_fly_ptr(_terrain_ptr(mapping.terrain), _mapping_ptr(mapping), _reachability_value(reachability))
        return cls(ptr=ptr, owned=True, mapping=mapping)

    @classmethod
    def pool(cls, mapping, reachability):
        ptr = cls.pool_ptr(_terrain_ptr(mapping.terrain), _mapping_ptr(mapping), _reachability_value(reachability))
        return cls(ptr=ptr, owned=True, mapping=mapping)

    @classmethod
    def serialization(cls, mapping, reachability, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        ptr = cls.serialization_ptr(
            _terrain_ptr(mapping.terrain),
            _mapping_ptr(mapping),
            _reachability_value(reachability),
            str(directory).encode("utf-8"),
        )
        return cls(ptr=ptr, owned=True, mapping=mapping)

    @classmethod
    def from_ptr(cls, ptr, *, owned=False):
        return cls(ptr=ptr, owned=owned)

    @property
    def ptr(self):
        return self._ptr

    @property
    def contents(self):
        return self._ptr.contents

    def free(self):
        if self._owned and self._ptr:
            self.free_ptr(self._ptr)
        self._ptr = None

    def __bool__(self):
        return bool(self._ptr)

    def __del__(self):
        self.free()


def _terrain_ptr(terrain):
    return terrain.ptr if hasattr(terrain, "ptr") else terrain


def _mapping_ptr(mapping):
    return mapping.ptr if hasattr(mapping, "ptr") else mapping


def _reachability_value(reachability):
    return int(reachability.value) if hasattr(reachability, "value") else int(reachability)
