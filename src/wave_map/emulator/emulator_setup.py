from wave_map.batch_runner.logger import Logger

class EmulatorSetup():
    def __init__(self):

        self.logger = Logger(self.base_output_dir/ "log.txt")
        pass

    def gather_data(self):
        self.logger.info("Gathering training/test data from simulations...")
        extractor = DataExtractor(
            run_family_name=self.run_family_name,
            test_hashes_file=self.test_hashes_file
        )
        X_train, Y_train, X_test, Y_test = extractor.extract()
    
        np.save(self.base_output_dir / "ppe_inputs_train.npy", X_train)
        np.save(self.base_output_dir / "ppe_outputs_train.npy", Y_train)
        np.save(self.base_output_dir / "ppe_inputs_test.npy", X_test)
        np.save(self.base_output_dir / "ppe_outputs_test.npy", Y_test)
    
        self.logger.info("Saved train/test datasets: X_train.npy, Y_train.npy, X_test.npy, Y_test.npy.")

