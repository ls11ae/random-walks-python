from abc import abstractmethod, ABC
from typing import Tuple, Optional

import numpy as np
import pandas as pd


def manhattan(start_point, end_point):
    return max(abs(start_point[0] - end_point[0]), abs(start_point[1] - end_point[1]))

def chebyshev(start_point, end_point):
    return max(abs(start_point[0] - end_point[0]),  abs(start_point[1] - end_point[1]))


def euclidean(start_point, end_point):
    return np.sqrt((start_point[0] - end_point[0]) ** 2 + (start_point[1] - end_point[1]) ** 2)


class MovementPolicy(ABC):
    def __init__(self, timestep_s):
        self.timestep_s = timestep_s

    @abstractmethod
    @abstractmethod
    def resolve(self,
                start_point,
                end_point,
                start_time,
                end_time,
                reference_speed:Optional[float]= None, movement_diffusivity:Optional[float] = 1.5
                ) -> Tuple[int, int]:
        """
        T = number of time steps
        S = kernel radius in grid cells per step (transition range = (2S+1)²)

        Args:
            start_point: start point
            end_point: end point
            start_time: start time
            end_time: end time
            reference_speed: reference speed in m/s for the animal
            movement_diffusivity: movement diffusivity >= 1, 1 straight line connection of two points
        """
        pass


class TimeStepPolicy(MovementPolicy):
    def __init__(self, timestep_s):
        super().__init__(timestep_s)

    def resolve(self, start_point, end_point, start_time, end_time, reference_speed:Optional[float]= None, movement_diffusivity:Optional[float] = 1.5):
        """
        Calculate T as number of time steps and S as step size in grid cells

        Parameters:
            start_point (tuple): Coordinates of the starting point as (x, y) in grid coordinates
            end_point (tuple): Coordinates of the ending point as (x, y) in grid coordinates
            start_time (str or pd.Timestamp): The start time of the traversal.
            end_time (str or pd.Timestamp): The end time of the traversal.
            reference_speed(float): literature-derived avg speed of the creature
            movement_diffusivity (float, optional): A factor influencing how much walk deviates from a straight line connection

        Returns:
            tuple: A tuple containing:
                - T (int): The number of time steps required.
                - S (int): The number of spatial steps per time step.

        Raises:
            None
        """
        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)
        start_point = np.array(start_point)
        end_point= np.array(end_point)

    
        dt_seconds = int((end_time - start_time).total_seconds())
        if dt_seconds == 0:
            dt_seconds = 0.0001
             
        calculated_speed = np.linalg.norm(end_point - start_point) / dt_seconds
        if calculated_speed == 0:
            calculated_speed = 1

        if reference_speed is not None :
            reference_speed /= self.timestep_s
            movement_diffusivity = reference_speed / calculated_speed
            movement_diffusivity = max(1.2, movement_diffusivity)
            
        if np.isnan(movement_diffusivity) or np.isinf(movement_diffusivity):
            movement_diffusivity = 1.5 
        else: 
            movement_diffusivity = movement_diffusivity if movement_diffusivity is not None else 1.5

        grid_dist = int(chebyshev(start_point, end_point) * movement_diffusivity)
        T = max(2, int(np.round(dt_seconds / self.timestep_s)))
        S = max(1, int(np.round(grid_dist / T)))

        if S > 50:
            ratio = S / 50
            T *= ratio
            S = 50

        return int(T), int(S)


class SpeedBasedPolicy(MovementPolicy):
    def __init__(self, timestep_s, base_speed, grid_cell_m):
        super().__init__(timestep_s)
        self.base_speed = base_speed
        self.grid_cell_m = grid_cell_m

    def resolve(self, start_point, end_point, start_time=None, end_time=None, reference_speed:Optional[float]= None, movement_diffusivity:Optional[float] = 1.5):
        """
        Calculate T as number of time steps and S as step size in grid cells

        Parameters:
        start_point : Any
            The starting point's coordinates in Euclidean space (UTM coordinates).
        end_point : Any
            The ending point's coordinates in Euclidean space (UTM coordinates).
        start_time : Optional[Any]
            The starting time for the movement. Defaults to None.
        end_time : Optional[Any]
            The ending time for the movement. Defaults to None.
        diffusity : float, optional
            A factor affecting the effective distance for movement, defaulting to 1.5. 1.0 means straight line interpolation

        Returns:
        tuple[int, int]
            A tuple containing:
            - T: The number of time steps needed for the movement.
            - S: The number of grid cells traversed per time step.
        """
        # in Euclidean space (L_2)
        dist_m = euclidean(start_point, end_point)
        step_length_m = self.base_speed * self.timestep_s
        effective_dist = dist_m * movement_diffusivity
        # on the grid (L_1)
        S = max(1, int(np.round(step_length_m / self.grid_cell_m)))
        T = max(1, int(np.ceil(effective_dist / step_length_m)))
        return T, S
