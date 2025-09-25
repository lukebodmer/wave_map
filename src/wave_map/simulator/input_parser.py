import sys
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SourcesConfig:
    number: int = 1
    centers: List[List[float]] = field(default_factory=lambda: [[0.5, 0.5, 0.0]])
    radii: List[float] = field(default_factory=lambda: [0.05])
    amplitudes: List[float] = field(default_factory=lambda: [0.1])
    frequencies: List[float] = field(default_factory=lambda: [30.0])

    def __post_init__(self):
        """Validate that all lists have the correct length matching the number"""
        if len(self.centers) != self.number:
            raise ValueError(f"Number of centers ({len(self.centers)}) must match number ({self.number})")
        if len(self.radii) != self.number:
            raise ValueError(f"Number of radii ({len(self.radii)}) must match number ({self.number})")
        if len(self.amplitudes) != self.number:
            raise ValueError(f"Number of amplitudes ({len(self.amplitudes)}) must match number ({self.number})")
        if len(self.frequencies) != self.number:
            raise ValueError(f"Number of frequencies ({len(self.frequencies)}) must match number ({self.number})")

        # Validate that each center has exactly 3 elements
        for i, center in enumerate(self.centers):
            if len(center) != 3:
                raise ValueError(f"Center {i} must have exactly 3 elements, got {len(center)}")


@dataclass
class MaterialConfig:
    inclusion_density: float = 8.0
    inclusion_wave_speed: float = 3.0
    inclusion_material_id: Optional[int] = None
    outer_density: float = 1.0
    outer_wave_speed: float = 1.5


@dataclass
class MeshConfig:
    number_of_cubes: int = 1
    cube_centers: List[List[float]] = field(default_factory=lambda: [[0.5, 0.5, 0.5]])
    cube_widths: List[float] = field(default_factory=lambda: [0.1])
    grid_size: float = 0.008
    box_size: float = 0.25
    inclusion_center: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    inclusion_scaling: List[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])
    inclusion_semi_major_axis_direction: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    msh_file: Optional[str] = None

    def __post_init__(self):
        """Validate that cube-related lists have the correct length matching number_of_cubes"""
        if len(self.cube_centers) != self.number_of_cubes:
            raise ValueError(f"Number of cube_centers ({len(self.cube_centers)}) must match number_of_cubes ({self.number_of_cubes})")
        if len(self.cube_widths) != self.number_of_cubes:
            raise ValueError(f"Number of cube_widths ({len(self.cube_widths)}) must match number_of_cubes ({self.number_of_cubes})")

        # Validate that each cube center has exactly 3 elements
        for i, center in enumerate(self.cube_centers):
            if len(center) != 3:
                raise ValueError(f"Cube center {i} must have exactly 3 elements, got {len(center)}")


@dataclass
class SolverConfig:
    total_time: Optional[float] = None
    number_of_timesteps: Optional[int] = None
    polynomial_order: int = 3

    def __post_init__(self):
        try:
            if (self.total_time is None) == (self.number_of_timesteps is None):
                raise ValueError("You must specify exactly one of 'total_time' or 'number_of_timesteps'.")
        except ValueError as e:
            print(f"Error in parameters.toml file: {e}")
            sys.exit(1)


@dataclass
class ReceiversConfig:
    pressure: List[List[float]] = field(default_factory=list)
    x_velocity: List[List[float]] = field(default_factory=list)
    y_velocity: List[List[float]] = field(default_factory=list)
    z_velocity: List[List[float]] = field(default_factory=list)
    top_sensors: Optional[int] = None
    side_sensors: Optional[int] = None
    sensors_per_face: Optional[int] = None
    additional_sensors: List[List[float]] = field(default_factory=list)

    def __post_init__(self):
        using_top_side = self.top_sensors is not None or self.side_sensors is not None
        using_sensors_per_face = self.sensors_per_face is not None

        if using_top_side and using_sensors_per_face:
            print("Error in parameters.toml file: Specify either 'top_sensors' and 'side_sensors' OR 'sensors_per_face', not both.")
            sys.exit(1)
        if not using_top_side and not using_sensors_per_face:
            print("Error in parameters.toml file: You must specify either 'top_sensors' and 'side_sensors' OR 'sensors_per_face'.")
            sys.exit(1)


@dataclass
class OutputIntervals:
    image: int = 10
    data: int = 100
    points: int = 10
    energy: int = 50


@dataclass
class SimulationInputParser:
    sources: SourcesConfig = field(default_factory=SourcesConfig)
    material: MaterialConfig = field(default_factory=MaterialConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    receivers: ReceiversConfig = field(default_factory=ReceiversConfig)
    output_intervals: OutputIntervals = field(default_factory=OutputIntervals)

    @classmethod
    def from_toml(cls, cfg: dict):
        return cls(
            sources=SourcesConfig(**cfg.get("sources", {})),
            material=MaterialConfig(**cfg.get("material", {})),
            mesh=MeshConfig(**cfg.get("mesh", {})),
            solver=SolverConfig(**cfg.get("solver", {})),
            receivers=ReceiversConfig(**cfg.get("receivers", {})),
            output_intervals=OutputIntervals(**cfg.get("output_intervals", {})),
        )
