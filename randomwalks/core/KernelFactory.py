import geopandas as gpd

from hmmcma import HMMStateAnnotator
from hmmcma.preprocessing import ColumnConfig
from kernelcma import Kernel2D, StateKernelFactory


class KernelsFactory:
    def __init__(self, gdf: gpd.GeoDataFrame,
                 num_states,
                 id_cols='individual_local_identifier',
                 time_col='timestamp',
                 geom_col='geometry',
                 provided_dir_col='direction',  # degrees
                 feature_cols=('distance', 'angular_diffusivity', 'speed', 'terrain'),
                 # additional data from the workflow
                 state_col='state',
                 scale=True,
                 ):
        self.columns = ColumnConfig(id_cols=id_cols,
                                    time_col=time_col,
                                    geom_col=geom_col,
                                    provided_dir_col=provided_dir_col,  # degrees
                                    feature_cols=feature_cols)

        self.gdf = gdf
        self.state_col = state_col
        self.scale = scale
        self.__state_mapping = None
        self.__num_states = num_states
        self.__trajectories = None
        self.__threshold = None

    def apply_hmm(self):
        annotator = HMMStateAnnotator(
            columns=self.columns,
            scale=self.scale,
            num_states=self.__num_states,
        )
        self.gdf, self.__trajectories, self.__threshold, self.__state_mapping = annotator.annotate(self.gdf)
        return self.gdf

    def get_state_kernels(self, dt_tolerance, rnge, reso, out=None):
        factory = StateKernelFactory(
            self.gdf,
            id_col=self.columns.id_col,
            time_col=self.columns.time_col,
            geom_col=self.columns.geom_col,
            state_col=self.state_col,
        )
        return factory.get_state_kernels(dt_tolerance, rnge, reso, out)
