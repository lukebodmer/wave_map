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
        inclusion_scaling_range: Tuple[Tuple[float, float], ...] = ((0.03, 0.07),) * 3,
        allow_inclusion_to_move: bool = False,
        allow_inclusion_to_rotate: bool = True,
        boundary_buffer: float = 0.05,
        domain_size: float = 1.00,
        seed: int = 42
    ):
        self.base_config = self._load_base_config(base_config_path)
        self.output_dir = Path(f"data/emulator_data/{run_family_name}/parameter_files/")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(seed)

        self.inclusion_density_range = inclusion_density_range
        self.inclusion_speed_range = inclusion_speed_range
        self.inclusion_scaling_range = inclusion_scaling_range
        self.allow_inclusion_to_move = allow_inclusion_to_move
        self.allow_inclusion_to_rotate = allow_inclusion_to_rotate
        self.boundary_buffer = boundary_buffer
        self.domain_size = domain_size

    def _load_base_config(self, path: str) -> dict:
        with open(path, "rb") as f:
            return tomli.load(f)

    def generate_lhs_samples(self, n_samples: int) -> np.ndarray:
        dims = 5  # density, speed, scaling (3)
        if self.allow_inclusion_to_rotate:
            dims += 3  # rotation: [angle, axis_azimuth, axis_cos_theta]
        if self.allow_inclusion_to_move:
            dims += 3  # center

        sampler = qmc.LatinHypercube(d=dims, rng=self.rng)
        unit_samples = sampler.random(n=n_samples)

        bounds = [
            self.inclusion_density_range,
            self.inclusion_speed_range,
            *self.inclusion_scaling_range,
        ]

        if self.allow_inclusion_to_rotate:
            bounds += [(0.0, 1.0)] * 3  # angle fraction, azimuthal angle, cos(theta)

        if self.allow_inclusion_to_move:
            bounds += [(0.0, 1.0)] * 3  # for x, y, z in unit cube

        bounds = np.array(bounds)
        return qmc.scale(unit_samples, bounds[:, 0], bounds[:, 1])

    def create_parameter_files(self, n_samples: int = 50):
        samples = self.generate_lhs_samples(n_samples)
        hashes_seen = set()

        for sample in samples:
            idx = 0
            density = sample[idx]
            speed = sample[idx + 1]
            # sorting the axes scaling parameters creates unique ellipsoid
            # orientations
            scaling = np.sort(sample[idx + 2:idx + 5])

            idx += 5

            # Sample rotation vector from angle and spherical axis coords
            if self.allow_inclusion_to_rotate:
                u_angle, u_phi, u_cos_theta = sample[idx:idx + 3]
                idx += 3

                angle = u_angle * np.pi  # θ ∈ [0, π]
                phi = 2 * np.pi * u_phi  # azimuthal angle ∈ [0, 2π]
                cos_theta = u_cos_theta  # ∈ [0, 1] for positive-z hemisphere
                sin_theta = np.sqrt(1 - cos_theta ** 2)

                x = sin_theta * np.cos(phi)
                y = sin_theta * np.sin(phi)
                z = cos_theta

                axis = np.array([x, y, z])
                rotation = axis * angle
            else:
                rotation = np.zeros(3)

            # Sample center position
            if self.allow_inclusion_to_move:
                ux, uy, uz = sample[idx:idx + 3]
                idx += 3

                radius_equiv = np.linalg.norm(scaling)
                buffer = self.boundary_buffer + radius_equiv
                lower_bound = buffer
                upper_bound = self.domain_size - buffer

                cx = lower_bound + ux * (upper_bound - lower_bound)
                cy = lower_bound + uy * (upper_bound - lower_bound)
                cz = lower_bound + uz * (upper_bound - lower_bound)
                center = [cx, cy, cz]
            else:
                c = self.domain_size / 2
                center = [c, c, c]

            # Prepare and write config
            config = self.base_config.copy()
            config['material']['inclusion_density'] = float(density)
            config['material']['inclusion_wave_speed'] = float(speed)
            config['mesh']['inclusion_scaling'] = [float(x) for x in scaling]
            config['mesh']['inclusion_rotation'] = [float(x) for x in rotation]
            config['mesh']['inclusion_center'] = [float(x) for x in center]

            config_str = toml.dumps(config)
            hash_val = hashlib.sha1(config_str.encode("utf-8")).hexdigest()[:8]
            if hash_val in hashes_seen:
                continue
            hashes_seen.add(hash_val)

            output_path = self.output_dir / f"{hash_val}.toml"
            with open(output_path, 'w') as f:
                f.write(config_str)

        print(f"Generated {len(hashes_seen)} unique parameter files in {self.output_dir}")
