import numpy as np
import tomli
from typing import Dict, Tuple, Union
from pathlib import Path
from scipy.stats import qmc

class ParameterSpaceSampler:
    def __init__(
        self,
        base_config_path: str,
        run_family_name: str,
        seed: int = 42
    ):
        self.base_config = self._load_base_config(base_config_path)
        self.output_dir = Path(f"data/inputs/{run_family_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(seed)

    def _load_base_config(self, path: str) -> dict:
        with open(path, "rb") as f:
            return tomli.load(f)

    def generate_lhs_samples(
        self,
        n_samples: int,
        density_range: Tuple[float, float],
        speed_range: Tuple[float, float]
       ) -> np.ndarray:
        """Generate LHS samples for density and speed ranges using SciPy"""
        sampler = qmc.LatinHypercube(d=2, rng=self.rng)
        unit_samples = sampler.random(n=n_samples)
        bounds = np.array([density_range, speed_range])
        samples = qmc.scale(unit_samples, bounds[:, 0], bounds[:, 1])
        return samples

    def create_parameter_files(self, n_samples: int = 50):
        """Generate multiple parameter files with LHS sampling"""
        samples = self.generate_lhs_samples(
            n_samples=n_samples,
            density_range=(0.1, 2.0),
            speed_range=(0.1, 2.0)
        )

        for i, (density, speed) in enumerate(samples):
            config = self.base_config.copy()
            config['material']['inclusion_density'] = float(density)
            config['material']['inclusion_wave_speed'] = float(speed)

            output_path = self.output_dir / f"config_{i:03d}.toml"
            with open(output_path, 'w') as f:
                toml.dump(config, f)

        print(f"Generated {n_samples} parameter files in {self.output_dir}")
