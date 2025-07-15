import tomli
import pickle
import json
import shutil
import hashlib
import sys
from pathlib import Path
from wave_simulator.simulator import Simulator
from wave_simulator.sensor_placer import SensorPlacer
from wave_simulator.finite_elements import LagrangeElement
from wave_simulator.mesh import Mesh3d
from wave_simulator.physics import LinearAcoustics
from wave_simulator.time_steppers import LowStorageRungeKutta
from wave_simulator.logger import Logger
from wave_simulator.input_parser import (
    InputParser,
    SourceConfig,
    MaterialConfig,
    MeshConfig,
    SolverConfig,
    ReceiversConfig,
    OutputIntervals,
)


class SimulationSetup:
    def __init__(self,
                 config_path: Path,
                 run_family_name="default_family"):
        self.config_path = Path(config_path)
        self.base_output_dir = Path(f"data/outputs/{run_family_name}")
        self.cfg = self._load_config()
        self.output_path = self._resolve_output_path()
        self.logger = Logger(self.output_path / "log.txt")
        self.prepare_output_dirs()

    def _load_config(self):
        with open(self.config_path, "rb") as f:
            raw = tomli.load(f)
        return InputParser(
            source=SourceConfig(**raw["source"]),
            material=MaterialConfig(**raw["material"]),
            mesh=MeshConfig(**raw["mesh"]),
            solver=SolverConfig(**raw["solver"]),
            receivers=ReceiversConfig(**raw["receivers"]),
            output_intervals=OutputIntervals(**raw["output_intervals"]),
        )

    def create_mesh(self):
        cfg = self.cfg
        # create a finite element
        finite_element = LagrangeElement(
            d=3,
            n=cfg.solver.polynomial_order
        )

        mesh = Mesh3d(
            finite_element=finite_element,
            grid_size=cfg.mesh.grid_size,
            box_size=cfg.mesh.box_size,
            inclusion_density=cfg.material.inclusion_density,
            inclusion_speed=cfg.material.inclusion_wave_speed,
            outer_density=cfg.material.outer_density,
            outer_speed=cfg.material.outer_wave_speed,
            source_center=cfg.source.center,
            source_radius=cfg.source.radius,
            source_amplitude=cfg.source.amplitude,
            source_frequency=cfg.source.frequency,
            inclusion_radius=cfg.mesh.inclusion_radius,
            inclusion_center=cfg.mesh.inclusion_center,
            msh_file=self.get_mesh_directory() / "mesh.msh",
        )

        # save mesh data needed for visualization
        mesh_path = self.get_mesh_directory() / "mesh.pkl"
        if not mesh_path.exists():
            self.save_mesh_visualization_data(mesh)

        return mesh

    def get_mesh_data(self, mesh):
        # Create minimal mesh data for visualization
        mesh_data = {
            'x': mesh.x,
            'y': mesh.y,
            'z': mesh.z,
            'vertex_coordinates': mesh.vertex_coordinates,
            'cell_to_vertices': mesh.cell_to_vertices,
            'nx': mesh.nx,
            'ny': mesh.ny,
            'nz': mesh.nz,
            'reference_element': mesh.reference_element,
            'initialize_gmsh': mesh.initialize_gmsh,
            'speed_per_cell': mesh.speed[0, :],  # First row only
            'density_per_cell': mesh.density[0, :],  # First row only
            'interior_face_node_indices': mesh.interior_face_node_indices,
            'boundary_node_indices': mesh.boundary_node_indices,
            'boundary_face_node_indices': mesh.boundary_face_node_indices,
            'cell_jacobians': mesh.jacobians[0, :],
            'num_cells': mesh.num_cells,
            'inclusion_center': mesh.inclusion_center,
            'inclusion_radius': mesh.inclusion_radius,
            }
        return mesh_data

    def save_mesh_visualization_data(self, mesh):
        mesh_data = self.get_mesh_data(mesh)
        mesh_path = self.get_mesh_directory() / "mesh.pkl"
        with open(mesh_path, 'wb') as f:
            pickle.dump(mesh_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def get_mesh_directory(self):
        mesh_hash = self._get_mesh_hash()
        return Path(f"data/inputs/meshes/{mesh_hash}")

    def _get_mesh_hash(self):
        """
        Create a short hash from mesh-related parameters.
        """
        params = {
            "grid_size": self.cfg.mesh.grid_size,
            "box_size": self.cfg.mesh.box_size,
            "inclusion_radius": self.cfg.mesh.inclusion_radius,
            "inclusion_center": self.cfg.mesh.inclusion_center,
            "source_center": self.cfg.source.center,
            "source_radius": self.cfg.source.radius,
            "polynomial_order": self.cfg.solver.polynomial_order,
        }
        encoded = json.dumps(params, sort_keys=True).encode()
        return hashlib.sha1(encoded).hexdigest()[:10]

    def _resolve_output_path(self):
        cfg = self.cfg
        name = (
            f"a{cfg.source.amplitude}_f{cfg.source.frequency}"
            f"_h{cfg.mesh.grid_size}_d{cfg.material.inclusion_density}"
            f"_c{cfg.material.inclusion_wave_speed}"
        )
        path = self.base_output_dir / name
        hash_suffix = self._hash_config()
        path = path.with_name(f"{hash_suffix}_{name}")
        # leave program if the simulation has already been run
        if path.exists():
            print(f"Simulation already exists at {path}. Exiting simulation.")
            sys.exit(0)
        return path

    def _hash_config(self):
        with open(self.config_path, "rb") as f:
            return hashlib.sha1(f.read()).hexdigest()[:8]

    def prepare_output_dirs(self):
        (self.output_path / "data").mkdir(parents=True, exist_ok=True)
        (self.output_path / "images").mkdir(parents=True, exist_ok=True)
        shutil.copy(self.config_path, self.output_path / "parameters.toml")

    def build_simulator(self):
        # get parameters from parameters.toml
        cfg = self.cfg

        # get mesh
        mesh = self.create_mesh()

        physics = LinearAcoustics(
            mesh=mesh,
            source_center=cfg.source.center,
            source_radius=cfg.source.radius,
            source_amplitude=cfg.source.amplitude,
            source_frequency=cfg.source.frequency,
        )

        # create timestepper from total time or number of timesteps
        if cfg.solver.total_time is not None:
            time_stepper = LowStorageRungeKutta(
                physics=physics,
                t_initial=0.0,
                t_final=cfg.solver.total_time,
            )
        elif cfg.solver.number_of_timesteps is not None:
            time_stepper = LowStorageRungeKutta(
                physics=physics,
                t_initial=0.0,
                number_of_timesteps=cfg.solver.number_of_timesteps,
            )

        sensor_placer = SensorPlacer(box_size=cfg.mesh.box_size,
                                     top_sensors=cfg.receivers.top_sensors,
                                     side_sensors=cfg.receivers.side_sensors,
                                     sensors_per_face=cfg.receivers.sensors_per_face,
                                     additional_sensors=cfg.receivers.additional_sensors,
                                     source_center=cfg.source.center,
                                     source_radius=cfg.source.radius,
                                     )

        sensor_coordinates = sensor_placer.get_sensor_coordinates()

        sim = Simulator(time_stepper,
                        output_path=self.output_path,
                        save_image_interval=cfg.output_intervals.image,
                        save_points_interval=cfg.output_intervals.points,
                        save_data_interval=cfg.output_intervals.data,
                        save_energy_interval=cfg.output_intervals.energy,
                        pressure_reciever_locations=sensor_coordinates,
                        u_velocity_reciever_locations=cfg.receivers.x_velocity,
                        v_velocity_reciever_locations=cfg.receivers.y_velocity,
                        w_velocity_reciever_locations=cfg.receivers.z_velocity,
                        mesh_directory=self.get_mesh_directory()
                        )

        return sim
