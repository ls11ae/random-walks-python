from random_walk_package.core.MixedWalker import MixedWalker


class MixedTimeWalker(MixedWalker):
    def __init__(self, data, env_data, kernel_mapping, resolution, out_directory):
        super().__init__(data, kernel_mapping, resolution, out_directory)
        self.env_data = env_data
        self.env_paths: dict[str, str] = {}
        self.kernel_resolver = None

    def _process_movebank_data(self):
        super().process_movebank_data()
        start_dt, end_dt = self.movebank_processor.time_period()
        self.env_paths = self.movebank_processor.kernel_params_per_animal_csv(df=self.env_data,
                                                                              kernel_resolver=self.kernel_resolver,
                                                                              start_date=start_dt,
                                                                              end_date=end_dt,
                                                                              time_stamp='timestamp',
                                                                              lon='longitude',
                                                                              lat='latitude',
                                                                              out_directory='random_walk_package/resources/movebank_test/kernel_data')
