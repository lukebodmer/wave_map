import pickle
import tomli
import shutil
import sys
from pathlib import Path

from wave_map.hash_functions.parameter_hashing import ParameterHashFunctions
from wave_map.simulator.simulation_manager import SimulationManager
from wave_map.simulator.sensor_placer import SensorPlacer
from wave_map.simulator.finite_elements import LagrangeElement
from wave_map.simulator.mesh import Mesh3d
from wave_map.simulator.physics import LinearAcoustics
from wave_map.simulator.time_steppers import LowStorageRungeKutta
from wave_map.loggers.logger import Logger
from wave_map.simulator.input_parser import (
    SimulationInputParser,
    SourcesConfig,
    MaterialConfig,
    MeshConfig,
    SolverConfig,
    ReceiversConfig,
    OutputIntervals,
)

# Constants
BATCH_DATA_DIR = "data/simulation_batch_data"


class SimulationSetup:
    """
    Initializes the simulation by building the timestepper, mesh, ...

    """
    def __init__(self,
                 dt: float,
                 config_path: Path,
                 batch_name="default",
                 ):
        self.dt = dt
        self.config_path = Path(config_path)
        #self.base_output_dir = Path(f"data/simulation_batch_data/{batch_name}/simulations")

        self.base_output_path = Path(f"{BATCH_DATA_DIR}/{batch_name}")
        self.base_simulations_path = self.base_output_path / "simulations"
        self.mesh_base_output_path = self.base_output_path / "meshes"
        self.cfg = self._load_config()
        self.output_path = self._resolve_output_path()
        self.logger = Logger(log_path=self.output_path / "log.txt", name="simlog")
        self._prepare_output_dirs()

    def _load_config(self):
        with open(self.config_path, "rb") as f:
            raw = tomli.load(f)
        return SimulationInputParser(
            sources=SourcesConfig(**raw["sources"]),
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

        if cfg.mesh.msh_file is not None:
            # if the gmsh file name is specified, get the mesh from the
            # common mesh directory
            filename = Path(cfg.mesh.msh_file)
            self.mesh_directory = Path(f"data/common_meshes/{filename.stem}")
            msh_file = self.mesh_directory / filename
            mesh_path = self.mesh_directory / "mesh.pkl"
        else:
            self.mesh_directory = self._get_mesh_directory()
            msh_file = self.mesh_directory / "mesh.msh"
            mesh_path = self.mesh_directory / "mesh.pkl"

        mesh = Mesh3d(
            finite_element=finite_element,
            msh_file=msh_file,
            grid_size=cfg.mesh.grid_size,
            box_size=cfg.mesh.box_size,
            #source_centers=cfg.source.centers,
            #source_radii=cfg.source.radii,
            outer_density=cfg.material.outer_density,
            outer_speed=cfg.material.outer_wave_speed,
            inclusion_density=cfg.material.inclusion_density,
            inclusion_speed=cfg.material.inclusion_wave_speed,
            inclusion_center=cfg.mesh.inclusion_center,
            inclusion_scaling=cfg.mesh.inclusion_scaling,
            inclusion_semi_major_axis_direction=cfg.mesh.inclusion_semi_major_axis_direction,
        )

        # save mesh data needed for visualization
        if not mesh_path.exists():
            self.save_mesh_visualization_data(mesh, self.mesh_directory)
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
            'speed_per_cell': mesh.speed[0, :],  # First row only
            'density_per_cell': mesh.density[0, :],  # First row only
            'interior_face_node_indices': mesh.interior_face_node_indices,
            'boundary_node_indices': mesh.boundary_node_indices,
            'boundary_face_node_indices': mesh.boundary_face_node_indices,
            'cell_jacobians': mesh.jacobians[0, :],
            'num_cells': mesh.num_cells,
            'inclusion_center': mesh.inclusion_center,
            'inclusion_scaling': mesh.inclusion_scaling,
            'inclusion_semi_major_axis_direction': mesh.inclusion_semi_major_axis_direction,
            }
        return mesh_data

    def save_mesh_visualization_data(self, mesh, mesh_directory):
        mesh_data = self.get_mesh_data(mesh)
        mesh_path = mesh_directory / "mesh.pkl"
        with open(mesh_path, 'wb') as f:
            pickle.dump(mesh_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def get_mesh_hash(self):
        parameter_file = self.config_path
        with open(parameter_file, "rb") as f:
            simulation_parameters = tomli.load(f)

        parser = SimulationInputParser.from_toml(simulation_parameters)

        hash_functions = ParameterHashFunctions()
        mesh_hash = hash_functions.get_mesh_hash(parser)
        return mesh_hash

    def _get_mesh_directory(self):
        mesh_hash = self.get_mesh_hash()
        #return Path(f"data/inputs/meshes/{mesh_hash}")
        return self.mesh_base_output_path / mesh_hash

    def _resolve_output_path(self):
        config_hash = self._hash_config()
        path = self.base_simulations_path / config_hash
        # leave program if the simulation has already been run
        if path.exists():
            print(f"Simulation already exists at {path}. Exiting simulation.")
            sys.exit(0)
        return path

    def _hash_config(self):
        hash_functions = ParameterHashFunctions()
        simulation_hash = hash_functions.get_simulation_hash(self.config_path)
        return simulation_hash

    def _prepare_output_dirs(self):
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
            source_centers=cfg.sources.centers,
            source_radii=cfg.sources.radii,
            source_amplitudes=cfg.sources.amplitudes,
            source_frequencies=cfg.sources.frequencies,
        )

        # create timestepper from total time or number of timesteps
        if cfg.solver.total_time is not None:
            time_stepper = LowStorageRungeKutta(
                physics=physics,
                dt=self.dt,
                t_initial=0.0,
                t_final=cfg.solver.total_time,
            )
        elif cfg.solver.number_of_timesteps is not None:
            time_stepper = LowStorageRungeKutta(
                physics=physics,
                dt=self.dt,
                t_initial=0.0,
                number_of_timesteps=cfg.solver.number_of_timesteps,
            )

        sensor_placer = SensorPlacer(box_size=cfg.mesh.box_size,
                                     top_sensors=cfg.receivers.top_sensors,
                                     side_sensors=cfg.receivers.side_sensors,
                                     sensors_per_face=cfg.receivers.sensors_per_face,
                                     additional_sensors=cfg.receivers.additional_sensors,
                                     source_centers=cfg.sources.centers,
                                     source_radii=cfg.sources.radii,
                                     )

        sensor_coordinates = sensor_placer.get_sensor_coordinates()

        sim = SimulationManager(time_stepper,
                                output_path=self.output_path,
                                save_image_interval=cfg.output_intervals.image,
                                save_points_interval=cfg.output_intervals.points,
                                save_data_interval=cfg.output_intervals.data,
                                save_energy_interval=cfg.output_intervals.energy,
                                pressure_reciever_locations=sensor_coordinates,
                                u_velocity_reciever_locations=cfg.receivers.x_velocity,
                                v_velocity_reciever_locations=cfg.receivers.y_velocity,
                                w_velocity_reciever_locations=cfg.receivers.z_velocity,
                                mesh_directory=self.mesh_directory
                                )

        return sim
