from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Optional, Tuple

import numpy as np


def manhattan(start_point, end_point):
    return max(abs(start_point[0] - end_point[0]), abs(start_point[1] - end_point[1]))


def chebyshev(start_point, end_point):
    return max(abs(start_point[0] - end_point[0]), abs(start_point[1] - end_point[1]))


def euclidean(start_point, end_point):
    return np.sqrt((start_point[0] - end_point[0]) ** 2 + (start_point[1] - end_point[1]) ** 2)


class MovementPolicy(ABC):
    def __init__(self, timestep_s):
        self.timestep_s = timestep_s

    @abstractmethod
    def resolve(
            self,
            start_point,
            end_point,
            start_time,
            end_time,
            reference_speed: Optional[float] = None,
            movement_diffusivity: Optional[float] = 1.5,
    ) -> Tuple[int, int]:
        """Return ``(T, S)`` for a segment."""
        pass

    def resolve_for_kernel(
            self,
            start_point,
            end_point,
            start_time,
            end_time,
            *,
            state=None,
            kernel=None,
            kernel_range_m=None,
            kernel_timestep_s=None,
            cell_size_m=None,
            reference_speed: Optional[float] = None,
            movement_diffusivity: Optional[float] = 1.5,
    ) -> Tuple[int, int]:
        """Resolve a segment after mapping a physical kernel to its RW grid.

        Existing policies keep their own time-step calculation, but the final
        spatial step radius is derived from the kernel's physical support. If
        that support cannot reach the endpoint in the requested number of
        transitions, ``T`` is increased without an artificial upper bound.

        Subclasses that model time from kernel metadata can override this
        method. The state and kernel arguments are supplied for those policies
        and are intentionally ignored by this compatibility implementation.
        """
        del state, kernel, kernel_timestep_s
        T, legacy_S = self.resolve(
            start_point=start_point,
            end_point=end_point,
            start_time=start_time,
            end_time=end_time,
            reference_speed=reference_speed,
            movement_diffusivity=movement_diffusivity,
        )
        if kernel_range_m is None or cell_size_m is None:
            return _positive_integer(T, "T"), _positive_integer(legacy_S, "S")

        S = _physical_radius_cells(kernel_range_m, cell_size_m)
        T = max(_positive_integer(T, "T"), _transitions_to_reach(start_point, end_point, S))
        return T, S


class AdaptiveKernelMovementPolicy(MovementPolicy):
    """Use a kernel's physical support and model duration on every interval.

    ``T`` starts at ``ceil(observed_duration / kernel_timestep)``. It is then
    increased only when required to make the observed endpoint geometrically
    reachable with the physical kernel radius. ``S`` is the physical kernel
    radius converted to cells on the current random-walk grid. Neither value
    is capped.

    ``dt_model_s`` is a fallback for kernels without timestep metadata. At
    runtime, state-specific ``dt_model_s`` metadata takes precedence.
    """

    def __init__(
            self,
            dt_model_s=None,
            *,
            grid_cell_m=None,
            ensure_reachable=True,
    ):
        if dt_model_s is not None:
            dt_model_s = _positive_float(dt_model_s, "dt_model_s")
        super().__init__(timestep_s=dt_model_s)
        self.dt_model_s = dt_model_s
        self.grid_cell_m = (
            None if grid_cell_m is None else _positive_float(grid_cell_m, "grid_cell_m")
        )
        self.ensure_reachable = bool(ensure_reachable)
        self.state_kernels = {}

    def bind_state_kernels(self, kernels):
        """Bind optional state-kernel objects for direct policy use."""
        if isinstance(kernels, Mapping):
            self.state_kernels = dict(kernels)
        else:
            self.state_kernels = {
                getattr(kernel, "state_value", index): kernel
                for index, kernel in enumerate(kernels)
            }
        return self

    def resolve(
            self,
            start_point,
            end_point,
            start_time,
            end_time,
            reference_speed: Optional[float] = None,
            movement_diffusivity: Optional[float] = 1.5,
    ) -> Tuple[int, int]:
        del reference_speed, movement_diffusivity
        if len(self.state_kernels) != 1:
            raise ValueError(
                "AdaptiveKernelMovementPolicy.resolve() requires exactly one bound kernel; "
                "use resolve_for_state() or let StateDependentWalker provide the state context."
            )
        state, kernel = next(iter(self.state_kernels.items()))
        return self.resolve_for_state(state, kernel, start_point, end_point, start_time, end_time)

    def resolve_for_state(
            self,
            state,
            kernel,
            start_point,
            end_point,
            start_time,
            end_time,
            *,
            cell_size_m=None,
            kernel_range_m=None,
            kernel_timestep_s=None,
    ) -> Tuple[int, int]:
        if kernel is None:
            kernel = self.state_kernels.get(state)
        kernel_range_m = _first_finite_positive(
            kernel_range_m,
            getattr(kernel, "rnge", None),
        )
        kernel_timestep_s = _first_finite_positive(
            kernel_timestep_s,
            getattr(kernel, "dt_model_s", None),
            self.dt_model_s,
        )
        cell_size_m = _first_finite_positive(
            cell_size_m,
            self.grid_cell_m,
            getattr(kernel, "dx", None),
        )
        return self._resolve_physical(
            start_point,
            end_point,
            start_time,
            end_time,
            kernel_range_m=kernel_range_m,
            kernel_timestep_s=kernel_timestep_s,
            cell_size_m=cell_size_m,
        )

    def resolve_for_kernel(
            self,
            start_point,
            end_point,
            start_time,
            end_time,
            *,
            state=None,
            kernel=None,
            kernel_range_m=None,
            kernel_timestep_s=None,
            cell_size_m=None,
            reference_speed: Optional[float] = None,
            movement_diffusivity: Optional[float] = 1.5,
    ) -> Tuple[int, int]:
        del reference_speed, movement_diffusivity
        if kernel is None and state in self.state_kernels:
            kernel = self.state_kernels[state]
        return self.resolve_for_state(
            state,
            kernel,
            start_point,
            end_point,
            start_time,
            end_time,
            cell_size_m=cell_size_m,
            kernel_range_m=kernel_range_m,
            kernel_timestep_s=kernel_timestep_s,
        )

    def _resolve_physical(
            self,
            start_point,
            end_point,
            start_time,
            end_time,
            *,
            kernel_range_m,
            kernel_timestep_s,
            cell_size_m,
    ) -> Tuple[int, int]:
        import pandas as pd

        if kernel_range_m is None:
            raise ValueError("A positive physical kernel range is required.")
        if kernel_timestep_s is None:
            raise ValueError("A positive kernel model timestep is required.")
        if cell_size_m is None:
            raise ValueError("A positive random-walk grid cell size is required.")

        dt_seconds = (
            pd.to_datetime(end_time) - pd.to_datetime(start_time)
        ).total_seconds()
        if not np.isfinite(dt_seconds) or dt_seconds <= 0:
            raise ValueError("Observed intervals must have a positive finite duration.")

        S = _physical_radius_cells(kernel_range_m, cell_size_m)
        time_T = max(1, int(np.ceil(dt_seconds / float(kernel_timestep_s))))
        if self.ensure_reachable:
            time_T = max(time_T, _transitions_to_reach(start_point, end_point, S))
        return time_T, S


class TimeStepPolicy(MovementPolicy):
    def resolve(
            self,
            start_point,
            end_point,
            start_time,
            end_time,
            reference_speed: Optional[float] = None,
            movement_diffusivity: Optional[float] = 1.5,
    ) -> Tuple[int, int]:
        import pandas as pd

        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)
        start_point = np.array(start_point)
        end_point = np.array(end_point)

        dt_seconds = (end_time - start_time).total_seconds()
        if dt_seconds <= 0:
            dt_seconds = 1e-4

        calculated_speed = np.linalg.norm(end_point - start_point) / dt_seconds
        if calculated_speed <= 0:
            calculated_speed = 1.0

        if reference_speed is not None:
            reference_speed /= self.timestep_s
            movement_diffusivity = max(1.2, reference_speed / calculated_speed)

        if movement_diffusivity is None or np.isnan(movement_diffusivity) or np.isinf(movement_diffusivity):
            movement_diffusivity = 1.5

        grid_dist = manhattan(start_point, end_point) * movement_diffusivity

        max_s = 30
        max_t = 2000
        min_s = 3
        min_t = 3

        T = max(min_t, int(np.ceil(dt_seconds / self.timestep_s)))
        S = max(1, int(np.ceil(grid_dist / T)))
        if S < min_s:
            S = min_s
            T = max(min_t, int(np.ceil(grid_dist / S))) if grid_dist > 0 else min_t

        if S > max_s:
            ratio = S / max_s
            T = int(np.ceil(T * ratio))
            S = max_s

        if T > max_t:
            ratio = T / max_t
            T = max_t
            S = int(np.ceil(S * ratio))

        T = max(min_t, min(T, max_t))
        S = max(min_s, min(S, max_s))

        return int(T), int(S)

    def resolve_for_kernel(
            self,
            start_point,
            end_point,
            start_time,
            end_time,
            *,
            kernel_range_m=None,
            cell_size_m=None,
            **context,
    ) -> Tuple[int, int]:
        if kernel_range_m is None or cell_size_m is None:
            return super().resolve_for_kernel(
                start_point,
                end_point,
                start_time,
                end_time,
                kernel_range_m=kernel_range_m,
                cell_size_m=cell_size_m,
                **context,
            )
        S = _physical_radius_cells(kernel_range_m, cell_size_m)
        T = max(1, int(np.ceil(_interval_seconds(start_time, end_time) / self.timestep_s)))
        return max(T, _transitions_to_reach(start_point, end_point, S)), S


class FixedStepsPolicy(MovementPolicy):
    def __init__(self, time_steps):
        super().__init__(time_steps)

    def resolve(
            self,
            start_point,
            end_point,
            start_time,
            end_time,
            reference_speed: Optional[float] = None,
            movement_diffusivity: Optional[float] = 1.5,
    ) -> Tuple[int, int]:
        import pandas as pd

        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)

        dt_seconds = (end_time - start_time).total_seconds()
        if dt_seconds <= 0:
            dt_seconds = 1e-4

        min_t = 3
        max_t = 2000
        fixed_t = max(min_t, min(int(np.ceil(self.timestep_s)), max_t))

        effective_timestep_s = dt_seconds / fixed_t
        if effective_timestep_s <= 0:
            effective_timestep_s = 1e-4

        return TimeStepPolicy(timestep_s=effective_timestep_s).resolve(
            start_point=start_point,
            end_point=end_point,
            start_time=start_time,
            end_time=end_time,
            reference_speed=reference_speed,
            movement_diffusivity=movement_diffusivity,
        )

    def resolve_for_kernel(
            self,
            start_point,
            end_point,
            start_time,
            end_time,
            *,
            kernel_range_m=None,
            cell_size_m=None,
            **context,
    ) -> Tuple[int, int]:
        if kernel_range_m is None or cell_size_m is None:
            return super().resolve_for_kernel(
                start_point,
                end_point,
                start_time,
                end_time,
                kernel_range_m=kernel_range_m,
                cell_size_m=cell_size_m,
                **context,
            )
        S = _physical_radius_cells(kernel_range_m, cell_size_m)
        requested_T = max(1, int(np.ceil(float(self.timestep_s))))
        return max(requested_T, _transitions_to_reach(start_point, end_point, S)), S


class SpeedBasedPolicy(MovementPolicy):
    def __init__(self, timestep_s, base_speed, grid_cell_m):
        super().__init__(timestep_s)
        self.base_speed = base_speed
        self.grid_cell_m = grid_cell_m

    def resolve(
            self,
            start_point,
            end_point,
            start_time=None,
            end_time=None,
            reference_speed: Optional[float] = None,
            movement_diffusivity: Optional[float] = 1.5,
    ) -> Tuple[int, int]:
        dist_m = euclidean(start_point, end_point)
        step_length_m = self.base_speed * self.timestep_s
        effective_dist = dist_m * movement_diffusivity
        S = max(1, int(np.round(step_length_m / self.grid_cell_m)))
        T = max(1, int(np.ceil(effective_dist / step_length_m)))
        return T, S


__all__ = [
    "AdaptiveKernelMovementPolicy",
    "MovementPolicy",
    "TimeStepPolicy",
    "FixedStepsPolicy",
    "SpeedBasedPolicy",
    "manhattan",
    "chebyshev",
    "euclidean",
]


def _physical_radius_cells(kernel_range_m, cell_size_m):
    kernel_range_m = _positive_float(kernel_range_m, "kernel_range_m")
    cell_size_m = _positive_float(cell_size_m, "cell_size_m")
    return max(1, int(np.ceil(kernel_range_m / cell_size_m)))


def _transitions_to_reach(start_point, end_point, step_radius_cells):
    grid_distance = float(chebyshev(np.asarray(start_point), np.asarray(end_point)))
    if not np.isfinite(grid_distance) or grid_distance < 0:
        raise ValueError("Grid endpoint distance must be finite and nonnegative.")
    return max(1, int(np.ceil(grid_distance / _positive_integer(step_radius_cells, "S"))))


def _positive_integer(value, name):
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < 1 or not numeric.is_integer():
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return int(numeric)


def _positive_float(value, name):
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}.")
    return numeric


def _first_finite_positive(*values):
    for value in values:
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric) and numeric > 0:
            return numeric
    return None


def _interval_seconds(start_time, end_time):
    import pandas as pd

    seconds = (pd.to_datetime(end_time) - pd.to_datetime(start_time)).total_seconds()
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("Observed intervals must have a positive finite duration.")
    return float(seconds)
