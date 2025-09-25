import tomli
import toml
import shutil
from pathlib import Path
from importlib import resources
from typing import Dict, List, Optional, Any

from wave_map.batch_runner.input_parser import BatchInputParser
from wave_map.simulator.input_parser import SimulationInputParser
from wave_map.simulator.time_step_size_calculator import TimeStepSizeCalculator
from wave_map.batch_runner.parameter_file_generator import ParameterFileGenerator
from wave_map.loggers.logger import Logger
from wave_map.hash_functions.parameter_hashing import ParameterHashFunctions
from wave_map.simulator.gmsh_mesh_generator import GmshMeshGenerator
from wave_map.simulator.simulation_setup import SimulationSetup

# Constants
BATCH_DATA_DIR = "data/simulation_batch_data"
BASE_CONFIG_FILENAME = "base_parameters.toml"
LOG_FILENAME = "log.txt"
PARAMETER_FILES_SUBDIR = "parameter_files"
SIMULATIONS_SUBDIR = "simulations"
MESH_INFO_FILENAME = "mesh_info.toml"
BATCH_METADATA_FILENAME = "batch_metadata.toml"


class BatchRunnerSetup:
    """
    Manages the setup and execution of batch wave simulation runs.

    This class handles the complete workflow for running multiple wave simulations
    with different parameter sets, including:
    - Loading and parsing batch configuration files
    - Generating parameter files for individual simulations
    - Creating and managing mesh files with appropriate time step calculations
    - Tracking simulation completion status
    - Executing pending simulations

    The class automatically creates necessary directory structures, generates
    missing meshes, computes optimal time steps, and maintains metadata for
    the entire batch run.

    Attributes:
        config_path (Path): Path to the batch configuration file
        batch_name (str): Name identifier for this batch run
        base_output_dir (Path): Root directory for all batch outputs
        parameter_files_dir (Path): Directory containing individual parameter files
        mesh_output_dir (Path): Directory for generated mesh files
        min_dt (Optional[float]): Minimum time step across all meshes
        completed_all_training_simulations (bool): Whether all sims are complete
        unsimulated_hashes (List[str]): List of parameter hashes not yet simulated

    Example:
        >>> batch_setup = BatchRunnerSetup(Path("config/batch_params.toml"))
        >>> batch_setup.run()  # Execute all pending simulations
    """
    def __init__(self, config_path: Path) -> None:
        self.config_path = Path(config_path)
        self.params = self._load_batch_parameters()

        self.batch_name = self.params.general.batch_name
        self.base_output_path = Path(f"{BATCH_DATA_DIR}/{self.batch_name}")
        self.base_config_path = self._resolve_config_path(self.params.general.base_config_path)
        self.parameter_files_path = self.base_output_path / PARAMETER_FILES_SUBDIR
        self.mesh_base_output_path = self.base_output_path / "meshes"

        self.logger = Logger(log_path=self.base_output_path / LOG_FILENAME, name="batchlog")
        self.prepare_output_dirs()
        self.save_copy_of_config_files()

        self._generate_parameter_files_if_needed()
        self.min_dt: Optional[float] = None
        self.mesh_info = self._generate_gmsh_files_if_needed()

        self.completed_all_simulations = False
        self.unsimulated_hashes: List[str] = []
        self.check_completed_simulations()

    def save_copy_of_config_files(self) -> None:
        shutil.copy2(self.config_path, self.base_output_path / self.config_path.name)
        shutil.copy2(self.base_config_path, self.base_output_path / BASE_CONFIG_FILENAME)

    def _load_toml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load a TOML file and return its contents as a dictionary."""
        try:
            with open(file_path, "rb") as f:
                return tomli.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        except tomli.TOMLDecodeError as e:
            raise ValueError(f"Invalid TOML format in {file_path}: {e}")

    def _load_batch_parameters(self) -> BatchInputParser:
        """Load a toml file of the batch parameters"""
        config = self._load_toml_file(self.config_path)
        parser = BatchInputParser()
        parser.load_from_toml(config)
        return parser

    def _resolve_config_path(self, config_filename: str) -> Path:
        """Resolve a config file path using importlib.resources with fallbacks"""
        try:
            # First try to get from package resources
            with resources.as_file(resources.files("wave_map.config").joinpath(config_filename)) as path:
                if path.exists():
                    return path
        except (ImportError, TypeError):
            # Fallback if resources API isn't available or fails
            pass
        
        # Fallback 1: Same directory as the main config file
        same_dir_path = self.config_path.parent / config_filename
        if same_dir_path.exists():
            return same_dir_path
        
        # Fallback 2: Direct path (absolute or relative to CWD)
        direct_path = Path(config_filename)
        if direct_path.exists():
            return direct_path
        
        raise FileNotFoundError(
            f"Could not find {config_filename} in package config, "
            f"same directory as {self.config_path}, or as direct path"
        )

    def prepare_output_dirs(self) -> None:
        self.base_output_path.mkdir(parents=True, exist_ok=True)

    def _generate_parameter_files_if_needed(self) -> None:
        n_expected = self.params.general.number_initial_parameter_files_to_create

        if self.parameter_files_path.exists() and len(list(self.parameter_files_path.glob("*.toml"))) >= n_expected:
            self.logger.info("Parameter files already exist. Skipping generation.")
            return

        self.logger.info(f"Generating {n_expected} parameter files...")

        base_config = self._load_toml_file(self.base_config_path)
        domain_size = base_config["mesh"]["box_size"]

        parameter_file_generator = ParameterFileGenerator(
            base_config_path=str(self.base_config_path),  # Convert to string if needed
            batch_name=self.batch_name,
            inclusion_density_range=tuple(self.params.inclusion.inclusion_density_range),
            inclusion_speed_range=tuple(self.params.inclusion.inclusion_wave_speed_range),
            inclusion_scaling_range=tuple(self.params.inclusion.inclusion_scaling_range),
            inclusion_is_sphere=self.params.inclusion.inclusion_is_sphere,
            inclusion_is_ellipsoid_of_revolution=self.params.inclusion.inclusion_is_ellipsoid_of_revolution,
            inclusions_are_multi_cubes=self.params.inclusion.inclusions_are_multi_cubes,
            allow_inclusion_to_rotate=self.params.inclusion.allow_inclusion_to_rotate,
            allow_inclusion_to_move=self.params.inclusion.allow_inclusion_to_move,
            boundary_buffer=self.params.geometry.boundary_buffer,
            domain_size=domain_size,
            cube_quantity_range=self.params.inclusion.cube_quantity_range,
            cube_width_range=self.params.inclusion.cube_width_range
        )

        parameter_file_generator.create_parameter_files(n_samples=n_expected)

    def _collect_mesh_info_from_parameter_files(self) -> Dict[str, Dict[str, Any]]:
        """Collect mesh information from all parameter files."""
        mesh_info = {}
        hash_functions = ParameterHashFunctions()

        for toml_file in self.parameter_files_path.glob("*.toml"):
            config = self._load_toml_file(toml_file)
            parser = SimulationInputParser.from_toml(config)

            mesh_hash = hash_functions.get_mesh_hash(parser)
            polynomial_order = parser.solver.polynomial_order

            material = config["material"]
            max_wave_speed = max(material["inclusion_wave_speed"], material["outer_wave_speed"])

            if mesh_hash not in mesh_info:
                mesh_info[mesh_hash] = {
                    "max_wave_speed": max_wave_speed,
                    "polynomial_order": polynomial_order,
                    "smallest_radii": None,
                    "param_file": toml_file
                }

        return mesh_info

    def _save_mesh_info(self, mesh_hash: str, smallest_radii, simulation_hash: str) -> None:
        mesh_info_file = self._get_mesh_info_filename(mesh_hash)

        # Convert NumPy scalars/arrays to plain floats/lists of floats
        if isinstance(smallest_radii, (list, tuple)):
            cleaned_radii = [float(r) for r in smallest_radii]
        elif hasattr(smallest_radii, "__iter__"):  # e.g., NumPy array
            cleaned_radii = [float(r) for r in list(smallest_radii)]
        else:
            cleaned_radii = float(smallest_radii)

        mesh_data = {
            "smallest_radii": cleaned_radii,
            "simulation_hash": simulation_hash
        }

        try:
            with open(mesh_info_file, "w") as f:
                toml.dump(mesh_data, f)
        except IOError as e:
            self.logger.info(f"Warning: Could not save mesh info for {mesh_hash}: {e}")

    def _compute_global_min_dt(self, mesh_info: Dict[str, Dict[str, Any]]) -> float:
        """Compute the global minimum time step across all meshes, using batch metadata if available."""

        # Check if batch metadata exists and contains a min_dt
        meta_file = self.base_output_path / BATCH_METADATA_FILENAME
        if meta_file.exists():
            try:
                with open(meta_file, "r") as f:
                    metadata = toml.load(f)
                if "min_dt" in metadata:
                    min_dt = metadata["min_dt"]
                    self.logger.info(f"Using min_dt from batch metadata: {min_dt:.6e}")
                    return min_dt
            except (IOError, toml.TomlDecodeError) as e:
                self.logger.warning(f"Failed to read batch metadata from {meta_file}: {e}, computing min_dt manually")

        # If no min_dt in metadata, compute it
        min_dt = None
        for mesh_hash, info in mesh_info.items():
            smallest_radii = info.get("smallest_radii")
            max_wave_speed = info.get("max_wave_speed")
            polynomial_order = info.get("polynomial_order")

            if smallest_radii is None:
                self.logger.info(f"Warning: No smallest_radii found for mesh {mesh_hash}, skipping dt calculation")
                continue

            # Convert list to minimum value
            if isinstance(smallest_radii, list):
                min_radius = min(smallest_radii)
            else:
                min_radius = float(smallest_radii)

            calculator = TimeStepSizeCalculator(
                max_wave_speed=max_wave_speed,
                smallest_radii=min_radius,
                polynomial_order=polynomial_order
            )
            dt_mesh = calculator.calculate_cfl_dt()

            if min_dt is None or dt_mesh < min_dt:
                min_dt = dt_mesh

        return min_dt

    def _save_batch_metadata(self, mesh_info: Dict[str, Dict[str, Any]], min_dt: float) -> None:
        """Save batch metadata to TOML file."""
        meta_file = self.base_output_path / BATCH_METADATA_FILENAME

        metadata = {
            "min_dt": min_dt,
            "mesh": {
                mesh_hash: {
                    "simulation_hash": info["simulation_hash"],
                    "max_wave_speed": info["max_wave_speed"],
                    "polynomial_order": info["polynomial_order"],
                    **({"smallest_radii": info["smallest_radii"]} if info["smallest_radii"] else {})
                }
                for mesh_hash, info in mesh_info.items()
                if "simulation_hash" in info
            }
        }

        try:
            with open(meta_file, "w") as f:
                toml.dump(metadata, f)
            self.logger.info(f"Computed global minimum dt = {min_dt:.6e} and saved to {meta_file}")
        except IOError as e:
            raise IOError(f"Failed to save batch metadata to {meta_file}: {e}")

    def _get_mesh_directory(self, mesh_hash):
        return self.mesh_base_output_path / mesh_hash

    def _get_mesh_info_filename(self, mesh_hash):
        mesh_directory = self._get_mesh_directory(mesh_hash)
        return mesh_directory / MESH_INFO_FILENAME

    def _load_existing_mesh_metadata(self, mesh_hash: str, info: Dict[str, Any]) -> None:
        """Load metadata from existing mesh using mesh_info.toml."""

        hash_functions = ParameterHashFunctions()
        sim_hash = hash_functions.get_simulation_hash(info["param_file"])
        self.logger.info(f"Mesh {mesh_hash} for simulation {sim_hash} already exists.")

        mesh_info_file = self._get_mesh_info_filename(mesh_hash)

        if mesh_info_file.exists():
            try:
                mesh_data = self._load_toml_file(mesh_info_file)
                info["smallest_radii"] = mesh_data.get("smallest_radii")

                # Always recompute simulation_hash from the parameter file
                info["simulation_hash"] = sim_hash

                if info["smallest_radii"] is None:
                    self.logger.info(f"Warning: No smallest_radii found in mesh info for {mesh_hash}")
            except (FileNotFoundError, ValueError) as e:
                self.logger.info(f"Warning: Could not read mesh info for {mesh_hash}: {e}")
        else:
            self.logger.info(f"Warning: No mesh info file found for existing mesh {mesh_hash}")

    def _generate_single_mesh(self, mesh_hash: str, info: Dict[str, Any], hash_functions: ParameterHashFunctions) -> None:
        """Generate a single mesh and update info with metadata."""

        config = self._load_toml_file(info["param_file"])
        simulation_parameters = SimulationInputParser.from_toml(config)

        simulation_hash = hash_functions.get_simulation_hash(info["param_file"])

        self.logger.info(f"Generating mesh {mesh_hash} for simulation {simulation_hash}...")

        gmsh_generator = GmshMeshGenerator(simulation_parameters, mesh_hash, self.batch_name)
        #gmsh_generator.generate_ellipsoid_geometry()
        gmsh_generator.generate_multi_cube_geometry()

        smallest_radii = gmsh_generator.get_smallest_radii()

        info["smallest_radii"] = smallest_radii
        info["simulation_hash"] = simulation_hash 

        # Save mesh metadata to mesh directory
        self._save_mesh_info(mesh_hash, smallest_radii, simulation_hash)

    def _generate_missing_meshes(self, mesh_info: Dict[str, Dict[str, Any]]) -> None:
        """Generate missing meshes and update mesh_info with smallest_radii."""
        hash_functions = ParameterHashFunctions()
        
        for mesh_hash, info in mesh_info.items():
            mesh_dir = self._get_mesh_directory(mesh_hash)
            
            if not mesh_dir.exists():
                self._generate_single_mesh(mesh_hash, info, hash_functions)
            else:
                self._load_existing_mesh_metadata(mesh_hash, info)

    def _generate_gmsh_files_if_needed(self) -> Dict[str, Dict[str, Any]]:
        """Collect all unique mesh hashes and generate missing meshes."""
        mesh_info = self._collect_mesh_info_from_parameter_files()
        self._generate_missing_meshes(mesh_info)
        
        min_dt = self._compute_global_min_dt(mesh_info)
        if min_dt is None:
            raise ValueError("Could not compute minimum time step - no valid meshes found")
        
        self.min_dt = min_dt

        self._save_batch_metadata(mesh_info, min_dt)
        
        # Remove param_file references before returning
        for mesh_data in mesh_info.values():
            mesh_data.pop("param_file", None)
        
        return mesh_info

    def check_completed_simulations(self) -> None:
        outputs_dir = self.base_output_path / SIMULATIONS_SUBDIR

        if not self.base_output_path.exists():
            self.logger.info(f"Base output directory {self.base_output_path} does not exist.")
            return

        if not self.parameter_files_path.exists():
            self.logger.info(f"No parameter files found at {self.parameter_files_path}.")
            return

        parameter_hashes = {
            f.stem for f in self.parameter_files_path.glob("*.toml") if f.is_file()
        }
        simulated_hashes = {
            d.name for d in outputs_dir.iterdir() if d.is_dir()
        } if outputs_dir.exists() else set()

        missing_runs = parameter_hashes - simulated_hashes
        if missing_runs:
            self.completed_all_simulations = False
            self.unsimulated_hashes = sorted(missing_runs)
            self.logger.info("Found parameter files that have not been simulated.")
        else:
            self.completed_all_simulations = True
            self.logger.info("All simulations completed.")

    def run(self) -> None:
        if not self.unsimulated_hashes:
            self.logger.info("No simulations to run.")
            return

        self.logger.info(f"Running {len(self.unsimulated_hashes)} missing simulations...")

        for hash in self.unsimulated_hashes:
            parameter_file = self.parameter_files_path / f"{hash}.toml"
            if not parameter_file.exists():
                self.logger.info(f"Parameter file {parameter_file} not found. Skipping.")
                continue

            #try:
            self.logger.info(f"\n... Preparing simulation {hash}")
            setup = SimulationSetup(dt=self.min_dt, config_path=parameter_file, batch_name=self.batch_name)
            simulator = setup.build_simulator()
            simulator.run()
            self.logger.info(f"Completed simulation for: {hash}")
            #except FileNotFoundError as e:
            #    self.logger.info(f"Simulation failed for {hash} - file not found: {e}")
            #except ValueError as e:
            #    self.logger.info(f"Simulation failed for {hash} - invalid parameters: {e}")
            #except Exception as e:
            #    self.logger.info(f"Simulation failed for {hash} - unexpected error: {e}")
