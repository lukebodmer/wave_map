from pathlib import Path
from wave_map.emulator.final_data_extractor import FinalDataExtractor
from wave_map.emulator.results_validator import ResultsValidator
from wave_map.loggers.logger import Logger

# Constants
BATCH_DATA_DIR = "data/simulation_batch_data"
LOG_FILENAME = "emulator_log.txt"
LOG_NAME = "emulatorlog"


class EmulatorSetup:
    """
    Sets up everything needed to run and validate an emulator batch:
    - Loads simulation data
    - Creates a logger
    - Instantiates the ResultsValidator
    """

    def __init__(self, batch_name: str, n_splits: int = 10, random_state: int = 42):
        self.batch_name = batch_name
        self.n_splits = n_splits
        self.random_state = random_state

        self.inputs = None
        self.outputs = None
        self.simulation_ids = None
        self.logger = None
        self.validator = None

        self.base_output_dir = Path(f"{BATCH_DATA_DIR}/{self.batch_name}")
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self._create_logger()

        self._load_data()
        self._create_results_validator()

    def _load_data(self):
        """Load simulation data using FinalDataExtractor."""
        extractor = FinalDataExtractor(batch_name=self.batch_name)
        self.inputs, self.outputs, self.simulation_ids = extractor.load()

    def _create_logger(self):
        """Create a logger for recording emulator validation results."""
        log_path = self.base_output_dir / LOG_FILENAME
        self.logger = Logger(log_path=log_path, name=LOG_NAME)

    def _create_results_validator(self):
        """Instantiate the ResultsValidator with the loaded data and logger."""
        self.validator = ResultsValidator(
            inputs=self.inputs,
            outputs=self.outputs,
            simulation_ids=self.simulation_ids,
            n_splits=self.n_splits,
            random_state=self.random_state
        )

    def run_validation(self):
        """Run k-fold cross-validation via the ResultsValidator."""
        self.validator.run_k_fold_validation()
