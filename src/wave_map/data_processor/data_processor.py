import pickle
import tomllib
import numpy as np

from pathlib import Path
from wave_map.loggers.logger import Logger

# Constants
BATCH_DATA_DIR = "data/simulation_batch_data"
LOG_FILENAME = "data_processor_log.txt"
LOG_NAME = "dataprocessorlog"


class DataProcessor:
    """
    Extracts and processes input and output data neede for training inverse model
    """
    def __init__(self, batch_name: str, downsample_factor: int = 2):
        self.batch_name = batch_name
        self.downsample_factor = downsample_factor
        self.base_dir = Path(f"{BATCH_DATA_DIR}/{self.batch_name}/simulations")
        self.logger = Logger(Path(f"{self.base_dir} / {LOG_FILENAME}"), LOG_NAME)

    def save_processed_training_files(self):
        """save missing input/output files for each simulation."""
        for sim_dir in self.base_dir.iterdir():
            if not sim_dir.is_dir():
                continue

            parameter_file = sim_dir / "parameters.toml"
            sensor_file = sim_dir / "final_sensor_data.pkl"
            model_output_file = sim_dir / "model_output.pkl"
            model_input_file = sim_dir / "model_input.pkl"

            # Skip if required raw files are missing
            if not (parameter_file.exists() and sensor_file.exists()):
                self.logger.info(f"Skipping {sim_dir.name}: missing input files.")
                continue

            # Generate model_output.pkl if missing
            if not model_output_file.exists():
                #try:
                self._generate_model_output_file(parameter_file, model_output_file)
                #self.logger.info(f"Created model_input.pkl for {sim_dir.name}")
                #except Exception as e:
                #    self.logger.info(f"Error creating model_output.pkl for {sim_dir.name}: {e}")

            # Generate model_input.pkl if missing
            if not model_input_file.exists():
                #try:
                self._generate_model_input_file(sensor_file, model_input_file)
                #except Exception as e:
                #self.logger.info(f"Error creating model_input.pkl for {sim_dir.name}: {e}")
        self.logger.info("model_input.pkl, and model_output.pkl files generated")

    def _generate_model_input_file(self, sensor_file, model_input_file):
        sensor_data = self._process_simulation_output(sensor_file)
        with open(model_input_file, "wb") as f:
            pickle.dump(sensor_data, f)

    def _generate_model_output_file(self, parameter_file, model_output_file):
        image_data = self._process_simulation_input(parameter_file)
        with open(model_output_file, "wb") as f:
            pickle.dump(image_data, f)

    def _load_sensor_data(self, sensor_file):
        # --- Load sensor data ---
        with open(sensor_file, "rb") as f:
            sensor_data = pickle.load(f)
        return sensor_data

    def _load_image_features(self, parameter_file):
        # Open and load it
        with parameter_file.open("rb") as f:  # must open in binary mode
            parameters = tomllib.load(f)

        input_features = [
            parameters["material"]["inclusion_density"],
            parameters["material"]["inclusion_wave_speed"],
            parameters["mesh"]["cube_centers"],
            parameters["mesh"]["cube_widths"],
        ]
        return input_features

    def _process_simulation_input(
        self,
        parameter_file,
        grid_size: int = 64,
       ) -> np.ndarray:
        """
        Convert inclusion parameters into a real-valued feature vector.

        Steps:
        1. Render cubes into a 3D voxel grid (unit cube domain).
        2. Compute FFT of voxel grid.
        3. Split into cos/sin coefficients.
        4. Flatten + concatenate into a single real-valued vector.
        """
        raw_image_data = self._load_image_features(parameter_file)

        density, wave_speed, cube_centers, cube_widths = raw_image_data

        # --- Step 1: voxel grid ---
        grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)

        # voxel coordinates from 0 → 1
        x = np.linspace(0, 1, grid_size, endpoint=False)
        y = np.linspace(0, 1, grid_size, endpoint=False)
        z = np.linspace(0, 1, grid_size, endpoint=False)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        for center, width in zip(cube_centers, cube_widths):
            cx, cy, cz = center
            hw = width / 2  # half-width
            mask = (
                (np.abs(X - cx) <= hw) &
                (np.abs(Y - cy) <= hw) &
                (np.abs(Z - cz) <= hw)
            )
            grid[mask] = 1.0

        # embed material properties
        grid *= density  # * wave_speed

        # --- Step 2: FFT ---
        kspace = np.fft.fftn(grid)
        kspace = np.fft.fftshift(kspace)  # shift zero-freq to center

        # --- Step 4: split cos/sin ---
        real_kspace = np.real(kspace).flatten()
        imaginary_kspace = np.imag(kspace).flatten()
        #magnitude = np.abs(kspace).flatten()
        #phase = np.angle(kspace).flatten()

        # --- Step 5: concatenate ---
        processed_image_data = np.concatenate([real_kspace, imaginary_kspace])
        #features = np.concatenate([magnitude, phase])

        return processed_image_data

    def _log_compress(self, data, eps=1e-6):
        # elementwise log compression
        return np.log1p(np.abs(data) / eps) * np.sign(data)

    def _power_compress(self, data, gamma=0.5):
        # gamma between 0 and 1, lower gamma = stronger compression
        return np.sign(data) * (np.abs(data) ** gamma)

    def _process_simulation_output(self, sensor_file: Path) -> np.ndarray:
        """
        Post-process sensor_data matrix.

        - Downsample each row (time series) by self.downsample_factor.
        - Flatten into a 1D array.
        """
        # get data
        processed_data = self._load_sensor_data(sensor_file)

        # cut off first timestep
        processed_data = processed_data[:, 1:]

        # trim more timesteps from start
        processed_data = processed_data[:, 50:]

        # downsample
        processed_data = processed_data[:, ::self.downsample_factor]

        # compress audio
        #processed_data = self._log_compress(processed_data)

        return processed_data.flatten()
