import sys
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GeneralConfig:
    batch_name: str
    number_initial_parameter_files_to_create: int
    base_config_path: str

    def __post_init__(self):
        if not isinstance(self.batch_name, str):
            raise TypeError("batch_name must be a string.")
        if not isinstance(self.base_config_path, str):
            raise TypeError("base_config_path must be a string.")
        if not isinstance(self.number_initial_parameter_files_to_create, int) or self.number_initial_parameter_files_to_create < 0:
            raise ValueError("number_initial_parameter_files_to_create must be a non-negative integer.")


@dataclass
class InclusionConfig:
    inclusion_wave_speed_range: List[float]
    inclusion_density_range: List[float]
    inclusion_scaling_range: List[List[float]]
    allow_inclusion_to_rotate: bool = False
    allow_inclusion_to_move: bool = False
    inclusion_is_sphere: bool = False
    inclusion_is_ellipsoid_of_revolution: bool = False
    inclusions_are_multi_cubes: bool = False
    cube_quantity_range: List[int] = None
    cube_width_range: List[float] = None

    def __post_init__(self):
        # Validate ranges
        for name, rng in [
            ("inclusion_wave_speed_range", self.inclusion_wave_speed_range),
            ("inclusion_density_range", self.inclusion_density_range),
        ]:
            if not (isinstance(rng, list) and len(rng) == 2 and all(isinstance(v, (int, float)) for v in rng)):
                raise ValueError(f"{name} must be a list of two numbers.")

        for name, nested_rng in [
            ("inclusion_scaling_range", self.inclusion_scaling_range),
        ]:
            if not (isinstance(nested_rng, list) and len(nested_rng) == 3):
                raise ValueError(f"{name} must have three sublists (for x/y/z axes).")
            for subrange in nested_rng:
                if not (isinstance(subrange, list) and len(subrange) == 2 and all(isinstance(v, (int, float)) for v in subrange)):
                    raise ValueError(f"Each sublist in {name} must contain two numbers.")

        # Validate booleans
        for name in ["allow_inclusion_to_move", "allow_inclusion_to_rotate",
                     "inclusion_is_sphere", "inclusion_is_ellipsoid_of_revolution",
                     "inclusions_are_multi_cubes"]:
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean.")

        # Conflict check: only one type of inclusion can be True
        inclusion_flags = [
            self.inclusion_is_sphere,
            self.inclusion_is_ellipsoid_of_revolution,
            self.inclusions_are_multi_cubes
        ]
        if sum(inclusion_flags) != 1:
            raise ValueError(
                "Exactly one of inclusion_is_sphere, inclusion_is_ellipsoid_of_revolution, "
                "or inclusions_are_multi_cubes must be True."
            )

        # If multi-cubes, validate cube ranges
        if self.inclusions_are_multi_cubes:
            if not (isinstance(self.cube_quantity_range, list) and len(self.cube_quantity_range) == 2 and
                    all(isinstance(v, int) for v in self.cube_quantity_range)):
                raise ValueError("cube_quantity_range must be a list of two integers.")
            if not (isinstance(self.cube_width_range, list) and len(self.cube_width_range) == 2 and
                    all(isinstance(v, (int, float)) for v in self.cube_width_range)):
                raise ValueError("cube_width_range must be a list of two numbers (floats).")


@dataclass
class GeometryConfig:
    boundary_buffer: float

    def __post_init__(self):
        if not isinstance(self.boundary_buffer, (int, float)) or self.boundary_buffer < 0:
            raise ValueError("boundary_buffer must be a non-negative number.")


@dataclass
class BatchInputParser:
    general: Optional[GeneralConfig] = None
    inclusion: Optional[InclusionConfig] = None
    geometry: Optional[GeometryConfig] = None

    def load_from_toml(self, toml_dict):
        try:
            self.general = GeneralConfig(**toml_dict["general"])
            self.inclusion = InclusionConfig(**toml_dict["inclusion"])
            self.geometry = GeometryConfig(**toml_dict["geometry"])
        except KeyError as e:
            print(f"Missing required section in parameters.toml: {e}")
            sys.exit(1)
        except (ValueError, TypeError) as e:
            print(f"Invalid value in parameters.toml: {e}")
            sys.exit(1)
