import pickle
import cupy as cp
import numpy as np
from pathlib import Path
from wave_map.loggers.logger import Logger
#from wave_map.inverse_models.PyRobustGaSP import PyRobustGaSP
from wave_map.inverse_models.parallel_partial_emulator import ParallelPartialEmulator 

# Constants
BATCH_DATA_DIR = "data/simulation_batch_data"
LOG_FILENAME = "model_tester_log.txt"
LOG_NAME = "modeltesterlog"


class ModelTester:
    """
    Sets up everything  needed to view and eventually validate an inverse model
    - Creates a logger
    - Loads a model
    - Creates images for the test set
    - Eventually (TODO) validate some metric for the images
    """

    def __init__(self, batch_name: str, model_name: str):  #, n_splits: int = 5, random_state: int = 42):
        #self.n_splits = n_splits
        #self.random_state = random_state

        self.batch_name = batch_name
        self.model_name = model_name

        # Directory where the trained models are saved
        self.batch_dir = Path(f"{BATCH_DATA_DIR}/{self.batch_name}/simulations")
        self.save_dir = Path(f"{BATCH_DATA_DIR}/{self.batch_name}/saved_inversion_models")
        self.predictions_output_path = Path(f"{BATCH_DATA_DIR}/{self.batch_name}/kspace_predictions")
        self.predictions_output_path.mkdir(exist_ok=True)

        self.logger = Logger(self.batch_dir / LOG_FILENAME, LOG_NAME)

        self._get_model()

    def _get_model(self):
        """Load the PPE model, type, and associated hashes."""
        self.model_path = self.save_dir / f"{self.model_name}.pkl"

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        with open(self.model_path, "rb") as f:
            data = pickle.load(f)

        # Store contents as instance attributes
        self.model = data.get("model")
        self.model_type = data.get("model_type", "unknown")
        self.train_hashes = data.get("train_hashes", [])
        self.test_hashes = data.get("test_hashes", [])
        self.mask = np.array(data.get("mask"), dtype=bool)

        self.logger.info(f"Loaded model '{self.model_name}' of type '{self.model_type}' from {self.model_path}")
        print(f"Loaded model '{self.model_name}' from {self.model_path}")

    def test_model(self):
        """Instantiate the ResultsValidator with the loaded data and logger."""
        if self.model_type == "parallel_partial_emulator":
            self._test_ppe_model()
        elif self.model_type == "neural_network":
            self._test_ppe_model()
        else:
            raise ValueError("not a valid model type")

    def _test_ppe_model(self):
        """
        Runs the loaded PPE model on all test simulations in self.test_hashes.
        Loads each simulation’s model_inputs.pkl, predicts k-space data, and saves it.
        """
        self.logger.info(f"Starting PPE model testing on {len(self.test_hashes)} test simulations")
        print(self.test_hashes)
        parallel_partial_emulator = ParallelPartialEmulator(self.batch_name)
        parallel_partial_emulator.load(self.model_path)

        for idx, sim_hash in enumerate(self.test_hashes, 1):
            sim_dir = self.batch_dir / sim_hash
            input_file = sim_dir / "model_input.pkl"

            if not input_file.exists():
                self.logger.warning(f"Missing model_input.pkl for test hash {sim_hash}, skipping")
                continue

            # --- Load model input ---
            with open(input_file, "rb") as f:
                X_test = pickle.load(f)

            # Ensure correct shape (single sample)
            if X_test.ndim == 1:
                X_test = X_test.reshape(1, -1)

            # Convert CuPy to NumPy if needed
            if isinstance(X_test, cp.ndarray):
                X_test = cp.asnumpy(X_test)

            # --- Predict with PPE model ---
            #prg = PyRobustGaSP()
            #prediction = prg.predict_ppgasp(self.model, X_test)["mean"]
            kspace_prediction = parallel_partial_emulator.predict(X_test)

            # --- Convert to k-space ---
            #kspace_pred = self._ppe_prediction_to_kspace(prediction)

            # --- Save the k-space output ---
            kspace_file = self.predictions_output_path / f"{sim_hash}.pkl"
            with open(kspace_file, "wb") as f:
                pickle.dump(kspace_prediction, f)

            self.logger.info(f"Saved k-space prediction for {sim_hash} → {kspace_file}")

        self.logger.info("Finished PPE model testing.")
