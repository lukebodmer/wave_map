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

    def post_process_input(
        self,
        input_features,
        grid_size: int = 64,
        trim_fraction: float = 0.5,
       ) -> np.ndarray:
        """
        Convert inclusion parameters into a real-valued feature vector.

        Steps:
        1. Render cubes into a 3D voxel grid (unit cube domain).
        2. Visualize voxel grid with PyVista.
        3. Compute FFT of voxel grid.
        4. Optionally trim high frequencies by keeping only a central cube.
        5. Split into cos/sin coefficients.
        6. Flatten + concatenate into a single real-valued vector.
        """
        density, wave_speed, cube_centers, cube_widths = input_features

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
        grid *= density# * wave_speed

        # --- Step 2: FFT ---
        kspace = np.fft.fftn(grid)
        kspace = np.fft.fftshift(kspace)  # shift zero-freq to center

        # --- Step 3: trim high frequencies ---
        if not (0 < trim_fraction <= 1.0):
            raise ValueError("trim_fraction must be in (0,1].")

        # --- Step 4: split cos/sin ---
        real_kspace = np.real(kspace).flatten()
        imaginary_kspace = np.imag(kspace).flatten()
        #magnitude = np.abs(kspace).flatten()
        #phase = np.angle(kspace).flatten()

        # --- Step 5: concatenate ---
        features = np.concatenate([real_kspace, imaginary_kspace])
        #features = np.concatenate([magnitude, phase])

        return features

    def post_process_output(self, sensor_data: np.ndarray) -> np.ndarray:
        """
        Post-process sensor_data matrix.

        - Downsample each row (time series) by self.downsample_factor.
        - Flatten into a 1D array.
        """
        # get data
        processed_data = sensor_data

        # cut off first timestep
        processed_data = processed_data[:, 1:]

        # trim more timesteps from start
        processed_data = processed_data[:, 50:]

        # downsample
        processed_data = processed_data[:, ::self.downsample_factor]

        # compress audio
        #processed_data = self.log_compress(processed_data)
        #processed_data = self.power_compress_rows(processed_data)

        return processed_data.flatten()

    def load(self):
        """Load all simulations' parameters and sensor data."""
        inputs, outputs, simulation_ids = [], [], []

        count = 0
        for sim_dir in self.batch_dir.iterdir():
            if count >= 10:
                break  # Stop after 100 files
            if sim_dir.is_dir():
                param_file = sim_dir / "parameters.toml"
                sensor_file = sim_dir / "final_sensor_data.pkl"

                if not (param_file.exists() and sensor_file.exists()):
                    continue

                # --- Load parameter file ---
                params = toml.load(param_file)

                #input_features = [
                #    params["material"]["inclusion_density"],
                #    params["material"]["inclusion_wave_speed"],
                #    #params["material"]["inclusion_material_id"],
                #    params["mesh"]["inclusion_scaling"][0],
                #    params["mesh"]["inclusion_scaling"][1],
                #    *params["mesh"]["inclusion_semi_major_axis_direction"],
                #]
                input_features = [
                    params["material"]["inclusion_density"],
                    params["material"]["inclusion_wave_speed"],
                    params["mesh"]["cube_centers"],
                    params["mesh"]["cube_widths"],
                ]

                # --- Load sensor data ---
                with open(sensor_file, "rb") as f:
                    sensor_data = pickle.load(f)

                # convert from cupy to numpy
                sensor_data = cp.asnumpy(sensor_data)
                # --- Post-process output ---
                processed_output = self.post_process_output(sensor_data)
                processed_input = self.post_process_input(input_features)

                #inputs.append(input_features)
                inputs.append(processed_input)
                outputs.append(processed_output)
                simulation_ids.append(sim_dir.name)

            #count += 1  # Increment counter

        X = np.array(inputs)
        Y = np.array(outputs)

        return X, Y, simulation_ids
