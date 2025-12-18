from dataclasses import dataclass

import geopandas as gpd
import numpy as np

from random_walk_package.core.hmm.kernels import pure_cor_grouped
from random_walk_package.core.hmm.models import apply_hmm
from random_walk_package.core.hmm.preprocessing import ColumnConfig, preprocess_hmm
from random_walk_package.core.hmm.utils import merge_states_to_gdf


@dataclass
class Kernel2D:
    Z: np.ndarray
    rnge: float
    reso: int
    dx: float


class KernelFactory:
    def __init__(self, gdf: gpd.GeoDataFrame,
                 id_cols='individual-local-identifier',
                 time_col='timestamp',
                 geom_col='geometry',
                 provided_dir_col='direction',  # degrees
                 feature_cols=('distance', 'angular_difference', 'speed'),  # additional data from the workflow
                 scale=True):
        self.columns = ColumnConfig(id_cols=id_cols,
                                    time_col=time_col,
                                    geom_col=geom_col,
                                    provided_dir_col=provided_dir_col,  # degrees
                                    feature_cols=feature_cols)

        self.gdf = gdf
        self.scale = scale
        self.__trajectories = None
        self.__threshold = None
        self.__state_mapping = None

    def __preprocess(self):
        return preprocess_hmm(self.gdf, self.columns, self.scale)

    def apply_hmm(self):
        arrays, scaler, seq_dfs = self.__preprocess()
        self.__trajectories, self.__threshold, self.__state_mapping = apply_hmm(arrays, seq_dfs)
        self.gdf = merge_states_to_gdf(self.gdf, seq_dfs, self.columns)
        return self.gdf

    def get_state_kernels(self, dt_tolerance, range, reso):
        dx = 2 * range / reso
        print(f"dx: {dx}\n")
        Za, Zb, Zc = pure_cor_grouped(self.__threshold, dt_tolerance, self.__trajectories, range, reso)
        kernelA = Kernel2D(Za, range, reso, dx)
        kernelB = Kernel2D(Zb, range, reso, dx)
        kernelC = Kernel2D(Zc, range, reso, dx)
        return kernelA, kernelB, kernelC
