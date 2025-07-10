import tomli
import pickle
import numpy as np
from pathlib import Path


class DataExtractor:
    """Extracts parameters and final pressure data from simulation output folders."""

    def __init__(self,
                 run_family_name="default_family"):
        self.base_output_dir = Path(f"data/outputs/{run_family_name}")
        self.base_dir = Path(base_dir)

    def extract(self):
        param_list = []
        pressure_list = []

        for folder in sorted(self.base_dir.iterdir()):
            if not folder.is_dir():
                continue

            toml_file = folder / "parameters.toml"
            pkl_file = folder / "data" / "final_sensor_data.pkl"

            if not toml_file.exists() or not pkl_file.exists():
                continue

            params = self._extract_parameters(toml_file)
            pressure_data = self._load_numpy_array(pkl_file)

            if params is not None and pressure_data is not None:
                param_list.append(params)
                pressure_list.append(pressure_data)

        X = np.array(param_list)  # shape: (n_simulations, 2)
        Y = np.array(pressure_list)  # shape: (n_simulations, n_sensors, n_readings)
        return X, Y

    def _extract_parameters(self, toml_file):
        """Extracts inclusion_density and inclusion_wave_speed from a TOML file."""
        try:
            with open(toml_file, "rb") as f:
                toml_data = tomli.load(f)
            material = toml_data.get("material", {})
            return [
                material.get("inclusion_density"),
                material.get("inclusion_wave_speed")
            ]
        except Exception as e:
            print(f"Error reading parameters from {toml_file}: {e}")
            return None

    def _load_numpy_array(self, pkl_file):
        """Loads a numpy array from a .pkl file."""
        try:
            with open(pkl_file, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading numpy array from {pkl_file}: {e}")
            return None
