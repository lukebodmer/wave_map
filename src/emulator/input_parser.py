import sys
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GeneralConfig:
    run_family_name: str
    number_initial_parameter_files_to_create: int
    base_config_path: str

    def __post_init__(self):
        if not isinstance(self.run_family_name, str):
            raise TypeError("run_family_name must be a string.")
        if not isinstance(self.base_config_path, str):
            raise TypeError("base_config_path must be a string.")
        if not isinstance(self.number_initial_parameter_files_to_create, int) or self.number_initial_parameter_files_to_create < 0:
            raise ValueError("number_initial_parameter_files_to_create must be a non-negative integer.")


@dataclass
class InclusionConfig:
    inclusion_wave_speed_range: List[float]
    inclusion_density_range: List[float]
    inclusion_radius_range: List[float]
    allow_inclusion_to_move: bool = False

    def __post_init__(self):
        for name, rng in [
            ("inclusion_wave_speed_range", self.inclusion_wave_speed_range),
            ("inclusion_density_range", self.inclusion_density_range),
            ("inclusion_radius_range", self.inclusion_radius_range),
        ]:
            if not (isinstance(rng, list) and len(rng) == 2 and all(isinstance(v, (int, float)) for v in rng)):
                raise ValueError(f"{name} must be a list of two numbers.")
            if any(v < 0 for v in rng):
                raise ValueError(f"{name} values must be non-negative.")
        if not isinstance(self.allow_inclusion_to_move, bool):
            raise TypeError("allow_inclusion_to_move must be a boolean.")


@dataclass
class GeometryConfig:
    boundary_buffer: float

    def __post_init__(self):
        if not isinstance(self.boundary_buffer, (int, float)) or self.boundary_buffer < 0:
            raise ValueError("boundary_buffer must be a non-negative number.")


@dataclass
class InputParser:
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
