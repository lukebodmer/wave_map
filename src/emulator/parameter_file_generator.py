import numpy as np
import tomli
import toml
import hashlib
from typing import Tuple
from pathlib import Path
from scipy.stats import qmc

class ParameterFileGenerator:
    def __init__(
        self,
        base_config_path: str,
        run_family_name: str,
        inclusion_density_range: Tuple[float, float] = (0.1, 2.0),
        inclusion_speed_range: Tuple[float, float] = (0.1, 2.0),
        inclusion_radius_range: Tuple[float, float] = (0.05, 0.05),
        allow_inclusion_to_move: bool = False,
        boundary_buffer: float = 0.05,
        domain_size: float = 0.25,
        seed: int = 42
    ):
        self.base_config = self._load_base_config(base_config_path)
        self.output_dir = Path(f"data/emulator_data/{run_family_name}/parameter_files/")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(seed)

        self.inclusion_density_range = inclusion_density_range
        self.inclusion_speed_range = inclusion_speed_range
        self.inclusion_radius_range = inclusion_radius_range
        self.allow_inclusion_to_move = allow_inclusion_to_move
        self.boundary_buffer = boundary_buffer
        self.domain_size = domain_size

    def _load_base_config(self, path: str) -> dict:
        with open(path, "rb") as f:
            return tomli.load(f)

    def generate_lhs_samples(self, n_samples: int) -> np.ndarray:
        """Generate LHS samples for density, speed, radius, and (x, y, z) center."""
        dims = 6  # 3 material params + 3 center coords
        sampler = qmc.LatinHypercube(d=dims, rng=self.rng)
        unit_samples = sampler.random(n=n_samples)
    
        # Extract parameter bounds
        density_bounds = self.inclusion_density_range
        speed_bounds = self.inclusion_speed_range
        radius_bounds = self.inclusion_radius_range
    
        # We will scale spatial samples later, after getting each sample’s radius
        bounds = np.array([
            density_bounds,
            speed_bounds,
            radius_bounds,
            [0, 1], [0, 1], [0, 1]  # unit cube for x, y, z, scaled per-sample
        ])
    
        return qmc.scale(unit_samples, bounds[:, 0], bounds[:, 1])

    def _sample_inclusion_center(self, radius: float) -> Tuple[float, float, float]:
        """Sample a center for the inclusion that is safely away from domain boundaries."""
        min_val = self.boundary_buffer + radius
        max_val = self.domain_size - self.boundary_buffer - radius
        return tuple(self.rng.uniform(min_val, max_val, size=3))

    def create_parameter_files(self, n_samples: int = 50):
        samples = self.generate_lhs_samples(n_samples)
        hashes_seen = set()

        for i, sample in enumerate(samples):
            density, speed, radius, ux, uy, uz = sample
            config = self.base_config.copy()

            config['material']['inclusion_density'] = float(density)
            config['material']['inclusion_wave_speed'] = float(speed)
            config['mesh']['inclusion_radius'] = float(radius)

            if self.allow_inclusion_to_move:
                buffer = self.boundary_buffer + radius
                lower_bound = buffer
                upper_bound = self.domain_size - buffer

                cx = lower_bound + ux * (upper_bound - lower_bound)
                cy = lower_bound + uy * (upper_bound - lower_bound)
                cz = lower_bound + uz * (upper_bound - lower_bound)

                config['mesh']['inclusion_center'] = [float(cx), float(cy), float(cz)]

            # Convert config to string for hashing
            config_str = toml.dumps(config)
            hash_val = hashlib.sha1(config_str.encode("utf-8")).hexdigest()[:8]

            if hash_val in hashes_seen:
                continue  # skip duplicates (very rare)
            hashes_seen.add(hash_val)

            output_path = self.output_dir / f"{hash_val}.toml"
            with open(output_path, 'w') as f:
                f.write(config_str)

        print(f"Generated {len(hashes_seen)} unique parameter files in {self.output_dir}")
