import geopandas as gpd

from random_walk_package.core.hmm.kernels import pure_cor_grouped
from random_walk_package.core.hmm.models import apply_hmm
from random_walk_package.core.hmm.preprocessing import ColumnConfig, preprocess_hmm


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

        self.dt_threshold = None
        self.gdf = gdf
        self.scale = scale

    def __preprocess(self):
        return preprocess_hmm(self.gdf, self.columns, self.scale)

    def apply_hmm(self, dt_tolerance, range, reso):
        arrays, scaler, seq_dfs = self.__preprocess()
        trajectories, threshold = apply_hmm(arrays, seq_dfs)
        a, b, c = pure_cor_grouped(threshold, dt_tolerance, trajectories, range, reso)
