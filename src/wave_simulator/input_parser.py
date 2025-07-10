import sys
from dataclasses import dataclass, field
from typing import List


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


@dataclass
class SolverConfig:
    total_time: float = None
    number_of_timesteps: int = None
    polynomial_order: int = 3

    def __post_init__(self):
        try:  
            if (self.total_time is None) == (self.number_of_timesteps is None):
                raise ValueError("You must specify exactly one of 'total_time' or 'number_of_timesteps'.")
        except ValueError as e:
            print(f"Error in parameters.toml file: {e}")
            sys.exit(1)  # or handle however you like


@dataclass
class ReceiversConfig:
    pressure: List[List[float]] = field(default_factory=list)
    x_velocity: List[List[float]] = field(default_factory=list)
    y_velocity: List[List[float]] = field(default_factory=list)
    z_velocity: List[List[float]] = field(default_factory=list)
    top_sensors: int = 15
    side_sensors: int = 10


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
