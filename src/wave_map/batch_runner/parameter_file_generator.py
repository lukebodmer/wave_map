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
        batch_name: str,
        inclusion_density_range: Tuple[float, float] = (0.1, 2.0),
        inclusion_speed_range: Tuple[float, float] = (0.1, 2.0),
        inclusion_scaling_range: Tuple[Tuple[float, float], ...] = ((0.03, 0.07),) * 3,
        inclusion_is_sphere: bool = False,
        inclusion_is_ellipsoid_of_revolution: bool = False,
        allow_inclusion_to_move: bool = False,
        allow_inclusion_to_rotate: bool = True,   # kept for backwards compatibility
        boundary_buffer: float = 0.05,
        domain_size: float = 1.00,
        seed: int = 42
    ):
        self.base_config = self._load_base_config(base_config_path)
        self.output_dir = Path(f"data/simulation_batch_data/{batch_name}/parameter_files/")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(seed)

        self.inclusion_density_range = inclusion_density_range
        self.inclusion_speed_range = inclusion_speed_range
        self.inclusion_scaling_range = inclusion_scaling_range
        self.inclusion_is_sphere = inclusion_is_sphere
        self.inclusion_is_ellipsoid_of_revolution = inclusion_is_ellipsoid_of_revolution
        self.allow_inclusion_to_move = allow_inclusion_to_move
        self.allow_inclusion_to_rotate = allow_inclusion_to_rotate  # now means "sample orientation"
        self.boundary_buffer = boundary_buffer
        self.domain_size = domain_size

    def _load_base_config(self, path: str) -> dict:
        with open(path, "rb") as f: return tomli.load(f)

    def generate_lhs_samples(self, n_samples: int) -> np.ndarray:
        dims = 5  # density, speed, scaling (3)

        if self.inclusion_is_sphere:
            dims = 3  # density, speed, single scaling
        elif self.inclusion_is_ellipsoid_of_revolution:
            dims = 4  # density, speed, two scalings: [a, b] → [a, b, b]

        if self.allow_inclusion_to_rotate:
            dims += 3  # semi-major axis direction (positive octant)

        if self.allow_inclusion_to_move:
            dims += 3  # center

        # Build bounds
        bounds = [
            self.inclusion_density_range,
            self.inclusion_speed_range,
        ]

        if self.inclusion_is_sphere:
            bounds += [self.inclusion_scaling_range[0]]
        elif self.inclusion_is_ellipsoid_of_revolution:
            bounds += [self.inclusion_scaling_range[0], self.inclusion_scaling_range[1]]
        else:
            bounds += list(self.inclusion_scaling_range)

        if self.allow_inclusion_to_rotate:
            bounds += [(0.0, 1.0)] * 3

        if self.allow_inclusion_to_move:
            bounds += [(0.0, 1.0)] * 3

        bounds = np.array(bounds)

        # figure out which dimensions are fixed vs sampled
        fixed_values = []
        sample_bounds = []
        fixed_mask = []
        for low, high in bounds:
            if np.isclose(low, high):
                fixed_values.append(low)
                fixed_mask.append(True)
                sample_bounds.append((0, 0))  # placeholder
            else:
                fixed_values.append(None)
                fixed_mask.append(False)
                sample_bounds.append((low, high))

        sample_bounds = np.array([b for b, is_fixed in zip(bounds, fixed_mask) if not is_fixed])

        # generate samples only for free dimensions
        if len(sample_bounds) > 0:
            sampler = qmc.LatinHypercube(d=sample_bounds.shape[0], rng=self.rng)
            unit_samples = sampler.random(n=n_samples)
            scaled_samples = qmc.scale(unit_samples, sample_bounds[:, 0], sample_bounds[:, 1])
        else:
            scaled_samples = np.zeros((n_samples, 0))  # no free dimensions

        # rebuild full samples (fixed + sampled)
        all_samples = []
        for row in scaled_samples if len(sample_bounds) > 0 else [()]*n_samples:
            full = []
            ri = 0
            for is_fixed, val in zip(fixed_mask, fixed_values):
                if is_fixed:
                    full.append(val)
                else:
                    full.append(row[ri])
                    ri += 1
            all_samples.append(full)

        return np.array(all_samples)

    def create_parameter_files(self, n_samples: int = 50):
        samples = self.generate_lhs_samples(n_samples)
        hashes_seen = set()

        for sample in samples:
            idx = 0
            density = sample[idx]
            speed = sample[idx + 1]

            if self.inclusion_is_sphere:
                s = sample[idx + 2]
                scaling = np.array([s, s, s])
                idx += 3

            elif self.inclusion_is_ellipsoid_of_revolution:
                a, b = sample[idx + 2:idx + 4]
                # enforce ordering: a ≥ b
                if b > a:
                    a, b = b, a
                scaling = np.array([a, b, b])
                idx += 4

            else:  # general ellipsoid
                scaling = sample[idx + 2:idx + 5]
                # enforce ordering: a ≥ b ≥ c
                scaling = np.sort(scaling)[::-1]
                idx += 5

            # Orientation (positive octant unit vector)
            if self.allow_inclusion_to_rotate:
                vx, vy, vz = sample[idx:idx + 3]
                idx += 3
                vec = np.array([vx, vy, vz])
                vec = np.abs(vec)
                vec /= np.linalg.norm(vec)
                semi_major_axis_direction = vec.tolist()
            else:
                semi_major_axis_direction = [1.0, 0.0, 0.0]

            # Center
            if self.allow_inclusion_to_move:
                ux, uy, uz = sample[idx:idx + 3]
                idx += 3

                radius_equiv = np.max(scaling)
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

            config = self.base_config.copy()
            config['material']['inclusion_density'] = float(density)
            config['material']['inclusion_wave_speed'] = float(speed)
            config['mesh']['inclusion_scaling'] = [float(x) for x in scaling]
            config['mesh']['inclusion_semi_major_axis_direction'] = [float(x) for x in semi_major_axis_direction]
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
