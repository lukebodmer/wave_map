import pickle
import pandas as pd
from pathlib import Path
import toml
import numpy as np
import cupy as cp


class FinalDataExtractor:
    """
    Extracts simulation data (parameters + sensor outputs) from a batch directory.

    - Inputs (X): parameters.toml values (e.g., density, wave speed, scaling, center)
    - Outputs (Y): processed + flattened final_sensor_data.pkl arrays
    - Simulation IDs: directory names for traceability
    """

    def __init__(self, batch_name: str, downsample_factor: int = 2):
        self.batch_name = batch_name
        self.downsample_factor = downsample_factor
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.batch_dir = self.project_root / f"data/simulation_batch_data/{batch_name}/simulations"

        if not self.batch_dir.exists():
            raise ValueError(f"Batch directory not found: {self.batch_dir}")

    def log_compress(self, data, eps=1e-6):
        # elementwise log compression
        return np.log1p(np.abs(data) / eps) * np.sign(data)

    def power_compress(self, data, gamma=0.5):
        # gamma between 0 and 1, lower gamma = stronger compression
        return np.sign(data) * (np.abs(data) ** gamma)

    def power_compress_rows(self, data, gamma=0.005):
        row_max = np.max(np.abs(data), axis=1, keepdims=True)
        #row_max[row_max == 0] = 1.0  # avoid division by zero
        normalized = data / row_max
        compressed = np.sign(normalized) * (np.abs(normalized) ** gamma)
        return compressed

    def post_process_output(self, sensor_data: np.ndarray) -> np.ndarray:
        """
        Post-process sensor_data matrix.

        - Downsample each row (time series) by self.downsample_factor.
        - Flatten into a 1D array.
        """
        processed_data = sensor_data
        # filter out small noise
        #processed[abs(processed) < 0.0000001] = 0

        # cut off first timestep
        processed_data = processed_data[:, 1:]

        # trim more timesteps from start
        #processed_data = processed_data[::6, :]
        #processed_data = np.delete(processed_data, slice(99, 125), axis=0)
        processed_data = processed_data[:, 200:]

        # downsample
        processed_data = processed_data[:, ::self.downsample_factor]
        #processed_data = self.log_compress(processed_data)
        processed_data = self.power_compress_rows(processed_data)

        # normalize each row by its maximum (avoid division by zero)
        #row_max = processed_data.max(axis=1, keepdims=True)
        #row_max[row_max == 0] = 1.0  # prevent divide-by-zero
        #processed_data = processed_data / row_max

        return processed_data.flatten()

    def load(self):
        """Load all simulations' parameters and sensor data."""
        inputs, outputs, simulation_ids = [], [], []

        for sim_dir in self.batch_dir.iterdir():
            if sim_dir.is_dir():
                param_file = sim_dir / "parameters.toml"
                sensor_file = sim_dir / "final_sensor_data.pkl"

                if not (param_file.exists() and sensor_file.exists()):
                    continue

                # --- Load parameter file ---
                params = toml.load(param_file)

                input_features = [
                    params["material"]["inclusion_density"],
                    params["material"]["inclusion_wave_speed"],
                    #params["material"]["inclusion_material_id"],
                    params["mesh"]["inclusion_scaling"][0],
                    params["mesh"]["inclusion_scaling"][1],
                    *params["mesh"]["inclusion_semi_major_axis_direction"],
                ]

                # --- Load sensor data ---
                with open(sensor_file, "rb") as f:
                    sensor_data = pickle.load(f)

                # convert from cupy to numpy
                sensor_data = cp.asnumpy(sensor_data)
                # --- Post-process output ---
                processed_output = self.post_process_output(sensor_data)

                inputs.append(input_features)
                outputs.append(processed_output)
                simulation_ids.append(sim_dir.name)

        X = np.array(inputs)
        Y = np.array(outputs)

        # --- Save to CSV ---
        #inputs_df = pd.DataFrame(X, index=simulation_ids)
        #outputs_df = pd.DataFrame(Y, index=simulation_ids)

        #inputs_df.to_csv(self.project_root / f"data/{self.batch_name}_inputs.csv")
        #outputs_df.to_csv(self.project_root / f"data/{self.batch_name}_outputs.csv")

        return X, Y, simulation_ids
