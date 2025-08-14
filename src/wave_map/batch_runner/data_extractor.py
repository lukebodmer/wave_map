import tomli
import pickle
import numpy as np
from pathlib import Path


class DataExtractor:
    """Extracts parameters and final pressure data from simulation output folders."""

    def __init__(self, run_family_name="default_family", test_hashes_file=None):
        self.base_dir = Path(f"data/outputs/{run_family_name}")
        self.test_hashes = set()
    
        if test_hashes_file and Path(test_hashes_file).exists():
            with open(test_hashes_file, "r") as f:
                self.test_hashes = {line.strip() for line in f if line.strip()}

    def extract(self):
        param_list = []
        pressure_list = []
        X_train, Y_train = [], []
        X_test, Y_test = [], []
        
        for folder in sorted(self.base_dir.iterdir()):
            if not folder.is_dir():
                continue
        
            hash_id = folder.name
            toml_file = folder / "parameters.toml"
            pkl_file = folder /  "final_sensor_data.pkl"
        
            if not toml_file.exists() or not pkl_file.exists():
                continue
        
            params = self._extract_parameters(toml_file)
            pressure_data = self._load_numpy_array(pkl_file)
        
            if params is not None and pressure_data is not None:
                if hash_id in self.test_hashes:
                    X_test.append(params)
                    Y_test.append(pressure_data)
                else:
                    X_train.append(params)
                    Y_train.append(pressure_data)
        
        X_train, Y_train = np.array(X_train), np.array(Y_train)
        X_test, Y_test = np.array(X_test), np.array(Y_test)
        
        return X_train, Y_train, X_test, Y_test


    def _extract_parameters(self, toml_file):
        """Extracts [density, wave_speed, cx, cy, cz, radius] from a TOML file."""
        try:
            with open(toml_file, "rb") as f:
                toml_data = tomli.load(f)
            material = toml_data.get("material", {})
            mesh = toml_data.get("mesh", {})
    
            density = material.get("inclusion_density")
            speed = material.get("inclusion_wave_speed")
            #center = mesh.get("inclusion_center", [None, None, None])
            rotation = mesh.get("inclusion_rotation", [None, None, None])
            scaling = mesh.get("inclusion_scaling", [None, None, None])
            
            if (
                density is None or
                speed is None or
                #any(c is None for c in center) or
                any(r is None for r in rotation) or
                any(s is None for s in scaling)
               ):
                raise ValueError("Missing parameter(s)")

            #return [density, speed, *center, radius]
            return [density, speed, *rotation, *scaling]
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
