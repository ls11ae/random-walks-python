from abc import ABC, abstractmethod
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
        min_s = 2
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
    "MovementPolicy",
    "TimeStepPolicy",
    "FixedStepsPolicy",
    "SpeedBasedPolicy",
    "manhattan",
    "chebyshev",
    "euclidean",
]
