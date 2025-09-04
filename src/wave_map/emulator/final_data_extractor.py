import pickle
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

    def post_process_output(self, sensor_data: np.ndarray) -> np.ndarray:
        """
        Post-process sensor_data matrix.

        - Downsample each row (time series) by self.downsample_factor.
        - Flatten into a 1D array.
        """
       # cut off first timestep
        processed = sensor_data
        processed = processed[:, 1:]

        # trim more timesteps from start
        processed = processed[:, 650:]

        # downsample
        processed = processed[:, ::self.downsample_factor]

        return processed.flatten()

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

        return np.array(inputs), np.array(outputs), simulation_ids
