from random_walk_package.core.MixedWalker import MixedWalker


class MixedTimeWalker(MixedWalker):
    def __init__(self, data, env_data, kernel_mapping, resolution, out_directory, env_samples,
                 kernel_resolver=None,
                 time_col="timestamp",
                 lon_col="location-long",
                 lat_col="location-lat",
                 id_col="tag-local-identifier",
                 crs="EPSG:4326"
                 ):
        super().__init__(data, kernel_mapping, resolution, out_directory, time_col, lon_col, lat_col, id_col, crs)
        self.env_data = env_data
        self.env_paths: dict[str, str] = {}
        self.kernel_resolver = kernel_resolver
        self.env_samples = env_samples
        self._process_movebank_data()

    def _process_movebank_data(self):
        super()._process_movebank_data()
        self.movebank_processor.env_samples = self.env_samples
        self.env_paths, times = self.movebank_processor.kernel_params_per_animal_csv(df=self.env_data,
                                                                                     kernel_resolver=self.kernel_resolver,
                                                                                     time_stamp='timestamp',
                                                                                     lon='longitude',
                                                                                     lat='latitude',
                                                                                     out_directory='random_walk_package/resources/movebank_test/kernel_data')
