import pickle
import pyvista as pv
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

        # --- Step 2: visualize voxel grid ---
        #plotter = pv.Plotter()

        ## Instead of passing a raw ndarray to add_volume, create an ImageData
        #grid_data = pv.ImageData()
        #grid_data.dimensions = grid.shape  # (nx, ny, nz)
        #grid_data.spacing = (1/grid_size, 1/grid_size, 1/grid_size)  # so full domain is 0→1
        #grid_data.origin = (0, 0, 0)
        #grid_data["values"] = grid.flatten(order="F")  # column-major flatten

        #plotter.add_volume(grid_data, opacity="sigmoid", shade=True)
        #plotter.show_grid()
        #plotter.show()

        # --- Step 3: FFT ---
        kspace = np.fft.fftn(grid)
        kspace = np.fft.fftshift(kspace)  # shift zero-freq to center

        # --- Step 4: trim high frequencies ---
        if not (0 < trim_fraction <= 1.0):
            raise ValueError("trim_fraction must be in (0,1].")

        # --- Step 2b: visualize k-space magnitude ---
        #kspace_magnitude = np.abs(kspace)
        #kspace_grid = pv.ImageData()
        #kspace_grid.dimensions = kspace_magnitude.shape
        #kspace_grid.spacing = (1/grid_size, 1/grid_size, 1/grid_size)  # match voxel coordinates
        #kspace_grid.origin = (0, 0, 0)
        #kspace_grid["values"] = kspace_magnitude.flatten(order="F")
        #plotter = pv.Plotter()
        #plotter.add_volume(kspace_grid,
        #                   #opacity="sigmoid",
        #                   shade=True,
        #                   cmap="viridis")
        #plotter.show_grid()
        #plotter.show()

        # trim K space
        #keep = int(grid_size * trim_fraction)
        #start = (grid_size - keep) // 2
        #end = start + keep
        #kspace_trimmed = kspace[start:end, start:end, start:end]

        # --- Step 5: split cos/sin ---
        cos_coeffs = np.real(kspace).flatten()
        sin_coeffs = np.imag(kspace).flatten()

        # --- Step 6: concatenate ---
        features = np.concatenate([cos_coeffs, sin_coeffs])

        # Normalize
        #features /= np.max(np.abs(features)) + 1e-12

        return features

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
        processed_data = processed_data[:, 50:]

        # downsample
        processed_data = processed_data[:, ::self.downsample_factor]
        #processed_data = self.log_compress(processed_data)
        #processed_data = self.power_compress_rows(processed_data)

        # normalize each row by its maximum (avoid division by zero)
        #row_max = processed_data.max(axis=1, keepdims=True)
        #row_max[row_max == 0] = 1.0  # prevent divide-by-zero
        #processed_data = processed_data / row_max

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

        # --- Save to CSV ---
        #inputs_df = pd.DataFrame(X, index=simulation_ids)
        #outputs_df = pd.DataFrame(Y, index=simulation_ids)

        #inputs_df.to_csv(self.project_root / f"data/{self.batch_name}_inputs.csv")
        #outputs_df.to_csv(self.project_root / f"data/{self.batch_name}_outputs.csv")

        return X, Y, simulation_ids
