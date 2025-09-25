import numpy as np
import tomli
import toml
import hashlib
from typing import Tuple
from pathlib import Path
from scipy.stats import qmc
from importlib import resources


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
        inclusions_are_multi_cubes: bool = False,
        allow_inclusion_to_move: bool = False,
        allow_inclusion_to_rotate: bool = True,
        boundary_buffer: float = 0.05,
        domain_size: float = 1.00,
        cube_quantity_range: Tuple[int, int] = (1, 3),
        cube_width_range: Tuple[float, float] = (0.05, 0.2),
        seed: int = 42
    ):
        self.material_list_file = resources.files("wave_map.config") / "material_properties.toml"
        self.materials = self._load_materials()
        self.base_config = self._load_base_config(base_config_path)
        self.output_dir = Path(f"data/simulation_batch_data/{batch_name}/parameter_files/")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(seed)

        self.inclusion_density_range = inclusion_density_range
        self.inclusion_speed_range = inclusion_speed_range
        self.inclusion_scaling_range = inclusion_scaling_range
        self.inclusion_is_sphere = inclusion_is_sphere
        self.inclusion_is_ellipsoid_of_revolution = inclusion_is_ellipsoid_of_revolution
        self.inclusions_are_multi_cubes = inclusions_are_multi_cubes
        self.allow_inclusion_to_move = allow_inclusion_to_move
        self.allow_inclusion_to_rotate = allow_inclusion_to_rotate
        self.boundary_buffer = boundary_buffer
        self.domain_size = domain_size
        self.cube_quantity_range = cube_quantity_range
        self.cube_width_range = cube_width_range

    # ------------------------------
    # Helpers
    # ------------------------------
    def _load_base_config(self, path: str) -> dict:
        with open(path, "rb") as f:
            return tomli.load(f)

    def _load_materials(self):
        with open(self.material_list_file, "rb") as f:
            mat_data = tomli.load(f)

        materials = []
        for name, props in mat_data.items():
            materials.append({
                "name": name,
                "density": props["density"],
                "wave_speed": props["wavespeed"],
            })
        return materials

    def _write_config(self, config: dict, hashes_seen: set):
        config_str = toml.dumps(config)
        hash_val = hashlib.sha1(config_str.encode("utf-8")).hexdigest()[:8]
        if hash_val in hashes_seen:
            return False
        hashes_seen.add(hash_val)

        output_path = self.output_dir / f"{hash_val}.toml"
        with open(output_path, 'w') as f:
            f.write(config_str)
        return True

    def _generate_lhs_samples(self, n_samples: int, bounds: list) -> np.ndarray:
        """Generic LHS sampling given bounds."""
        bounds = np.array(bounds)
        sampler = qmc.LatinHypercube(d=len(bounds), rng=self.rng)
        unit_samples = sampler.random(n=n_samples)
        return qmc.scale(unit_samples, bounds[:, 0], bounds[:, 1])

    # ------------------------------
    # Multi-cube workflow
    #------------------------------
    def _create_multi_cube_parameters(self, n_samples: int):
        """
        Sample cube widths first, then place centers so that:
          - each cube is fully inside the domain: center in [buffer + w/2, box_size - (buffer + w/2)]
          - cubes are separated by at least `buffer + 0.5*(w_i + w_j)` in Euclidean distance
        """
        hashes_seen = set()

        for _ in range(n_samples):
            # sample material properties
            inclusion_density = float(
                self.rng.uniform(self.inclusion_density_range[0], self.inclusion_density_range[1])
            )
            inclusion_wave_speed = float(
                self.rng.uniform(self.inclusion_speed_range[0], self.inclusion_speed_range[1])
            )

            # sample number of cubes and widths first
            n_cubes = int(self.rng.integers(self.cube_quantity_range[0], self.cube_quantity_range[1] + 1))
            widths = self.rng.uniform(self.cube_width_range[0], self.cube_width_range[1], size=n_cubes).astype(float)

            centers = []
            attempt_limit = 1000  # safeguard against infinite loop

            for i in range(n_cubes):
                w_i = float(widths[i])
                min_c = self.boundary_buffer + 0.5 * w_i
                max_c = self.domain_size - (self.boundary_buffer + 0.5 * w_i)

                # quick feasibility check
                if min_c > max_c:
                    raise RuntimeError(
                        f"Cube {i} width {w_i:.6g} too large to place inside domain "
                        f"with buffer {self.boundary_buffer:.6g} and box_size {self.domain_size:.6g}."
                    )

                placed = False
                for attempt in range(attempt_limit):
                    ux, uy, uz = self.rng.random(3)
                    cx = min_c + ux * (max_c - min_c)
                    cy = min_c + uy * (max_c - min_c)
                    cz = min_c + uz * (max_c - min_c)
                    candidate = np.array([cx, cy, cz], dtype=float)

                    # check pairwise separation using buffer + half-widths
                    ok = True
                    for j, c in enumerate(centers):
                        w_j = float(widths[j])
                        min_allowed = self.boundary_buffer + 0.5 * (w_i + w_j)
                        if np.linalg.norm(candidate - np.array(c)) < (min_allowed - 1e-12):
                            ok = False
                            break

                    if ok:
                        centers.append(candidate.tolist())
                        placed = True
                        break

                if not placed:
                    raise RuntimeError(
                        f"Could not place cube {i+1}/{n_cubes} with width {w_i:.6g} "
                        f"after {attempt_limit} attempts. Try reducing widths or cube count."
                    )

            # Build config and write
            config = self.base_config.copy()
            config['material']['inclusion_density'] = inclusion_density
            config['material']['inclusion_wave_speed'] = inclusion_wave_speed
            config['mesh']['number_of_cubes'] = n_cubes
            config['mesh']['cube_centers'] = [[float(x), float(y), float(z)] for x, y, z in centers]
            config['mesh']['cube_widths'] = [float(w) for w in widths]

            self._write_config(config, hashes_seen)

        print(f"Generated {len(hashes_seen)} unique multi-cube parameter files in {self.output_dir}")

    # ------------------------------
    # Sphere workflow
    # ------------------------------
    def _create_sphere_parameters(self, n_samples: int):
        bounds = [self.inclusion_scaling_range[0]]
        if self.allow_inclusion_to_rotate:
            bounds += [(0.0, 1.0)] * 3
        if self.allow_inclusion_to_move:
            bounds += [(0.0, 1.0)] * 3
        samples = self._generate_lhs_samples(n_samples, bounds)

        hashes_seen = set()
        for row in samples:
            mat = self.rng.choice(self.materials)
            idx = 0
            s = row[idx]; idx += 1
            scaling = [s, s, s]

            if self.allow_inclusion_to_rotate:
                vx, vy, vz = row[idx:idx+3]; idx += 3
                vec = np.abs([vx, vy, vz])
                vec /= np.linalg.norm(vec)
                direction = vec.tolist()
            else:
                direction = [1.0, 0.0, 0.0]

            if self.allow_inclusion_to_move:
                ux, uy, uz = row[idx:idx+3]; idx += 3
                radius_equiv = max(scaling)
                buffer = self.boundary_buffer + radius_equiv
                lower = buffer; upper = self.domain_size - buffer
                cx = lower + ux * (upper - lower)
                cy = lower + uy * (upper - lower)
                cz = lower + uz * (upper - lower)
                center = [cx, cy, cz]
            else:
                c = self.domain_size / 2
                center = [c, c, c]

            config = self.base_config.copy()
            config['material']['inclusion_density'] = float(mat["density"])
            config['material']['inclusion_wave_speed'] = float(mat["wave_speed"])
            config['mesh']['inclusion_scaling'] = [float(x) for x in scaling]
            config['mesh']['inclusion_semi_major_axis_direction'] = [float(x) for x in direction]
            config['mesh']['inclusion_center'] = [float(x) for x in center]

            self._write_config(config, hashes_seen)

        print(f"Generated {len(hashes_seen)} unique sphere parameter files in {self.output_dir}")

    # ------------------------------
    # Ellipsoid of revolution workflow
    # ------------------------------
    def _create_ellipsoid_of_revolution_parameters(self, n_samples: int):
        bounds = [self.inclusion_scaling_range[0], self.inclusion_scaling_range[1]]
        if self.allow_inclusion_to_rotate:
            bounds += [(0.0, 1.0)] * 3
        if self.allow_inclusion_to_move:
            bounds += [(0.0, 1.0)] * 3
        samples = self._generate_lhs_samples(n_samples, bounds)

        hashes_seen = set()
        for row in samples:
            mat = self.rng.choice(self.materials)
            idx = 0
            a, b = row[idx:idx+2]; idx += 2
            if b > a: a, b = b, a
            scaling = [a, b, b]

            if self.allow_inclusion_to_rotate:
                vx, vy, vz = row[idx:idx+3]; idx += 3
                vec = np.abs([vx, vy, vz])
                vec /= np.linalg.norm(vec)
                direction = vec.tolist()
            else:
                direction = [1.0, 0.0, 0.0]

            if self.allow_inclusion_to_move:
                ux, uy, uz = row[idx:idx+3]; idx += 3
                radius_equiv = max(scaling)
                buffer = self.boundary_buffer + radius_equiv
                lower = buffer; upper = self.domain_size - buffer
                cx = lower + ux * (upper - lower)
                cy = lower + uy * (upper - lower)
                cz = lower + uz * (upper - lower)
                center = [cx, cy, cz]
            else:
                c = self.domain_size / 2
                center = [c, c, c]

            config = self.base_config.copy()
            config['material']['inclusion_density'] = float(mat["density"])
            config['material']['inclusion_wave_speed'] = float(mat["wave_speed"])
            config['mesh']['inclusion_scaling'] = [float(x) for x in scaling]
            config['mesh']['inclusion_semi_major_axis_direction'] = [float(x) for x in direction]
            config['mesh']['inclusion_center'] = [float(x) for x in center]

            self._write_config(config, hashes_seen)

        print(f"Generated {len(hashes_seen)} unique ellipsoid-of-revolution parameter files in {self.output_dir}")

    # ------------------------------
    # General ellipsoid workflow
    # ------------------------------
    def _create_general_ellipsoid_parameters(self, n_samples: int):
        bounds = list(self.inclusion_scaling_range)
        if self.allow_inclusion_to_rotate:
            bounds += [(0.0, 1.0)] * 3
        if self.allow_inclusion_to_move:
            bounds += [(0.0, 1.0)] * 3
        samples = self._generate_lhs_samples(n_samples, bounds)

        hashes_seen = set()
        for row in samples:
            mat = self.rng.choice(self.materials)
            idx = 0
            scaling = np.sort(row[idx:idx+3])[::-1]; idx += 3

            if self.allow_inclusion_to_rotate:
                vx, vy, vz = row[idx:idx+3]; idx += 3
                vec = np.abs([vx, vy, vz])
                vec /= np.linalg.norm(vec)
                direction = vec.tolist()
            else:
                direction = [1.0, 0.0, 0.0]

            if self.allow_inclusion_to_move:
                ux, uy, uz = row[idx:idx+3]; idx += 3
                radius_equiv = max(scaling)
                buffer = self.boundary_buffer + radius_equiv
                lower = buffer; upper = self.domain_size - buffer
                cx = lower + ux * (upper - lower)
                cy = lower + uy * (upper - lower)
                cz = lower + uz * (upper - lower)
                center = [cx, cy, cz]
            else:
                c = self.domain_size / 2
                center = [c, c, c]

            config = self.base_config.copy()
            config['material']['inclusion_density'] = float(mat["density"])
            config['material']['inclusion_wave_speed'] = float(mat["wave_speed"])
            config['mesh']['inclusion_scaling'] = [float(x) for x in scaling]
            config['mesh']['inclusion_semi_major_axis_direction'] = [float(x) for x in direction]
            config['mesh']['inclusion_center'] = [float(x) for x in center]

            self._write_config(config, hashes_seen)

        print(f"Generated {len(hashes_seen)} unique general ellipsoid parameter files in {self.output_dir}")

    # ------------------------------
    # Public
    # ------------------------------
    def create_parameter_files(self, n_samples: int = 50):
        if self.inclusions_are_multi_cubes:
            self._create_multi_cube_parameters(n_samples)
        elif self.inclusion_is_sphere:
            self._create_sphere_parameters(n_samples)
        elif self.inclusion_is_ellipsoid_of_revolution:
            self._create_ellipsoid_of_revolution_parameters(n_samples)
        else:
            self._create_general_ellipsoid_parameters(n_samples)
