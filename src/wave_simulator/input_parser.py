import sys
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SourceConfig:
    center: List[float] = field(default_factory=lambda: [0.125, 0.125, 0.0])
    radius: float = 0.02
    amplitude: float = 0.1
    frequency: float = 20.0


@dataclass
class MaterialConfig:
    inclusion_density: float = 8.0
    inclusion_wave_speed: float = 3.0
    inclusion_radius: float = 3.0
    outer_density: float = 1.0
    outer_wave_speed: float = 1.5


@dataclass
class MeshConfig:
    grid_size: float = 0.008
    box_size: float = 0.25
    inclusion_radius: float = 0.05
    inclusion_center: List[float] = field(default_factory=lambda: [0.125, 0.125, 0.125])


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
class InputParser:
    source: SourceConfig = field(default_factory=SourceConfig)
    material: MaterialConfig = field(default_factory=MaterialConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    receivers: ReceiversConfig = field(default_factory=ReceiversConfig)
    output_intervals: OutputIntervals = field(default_factory=OutputIntervals)
