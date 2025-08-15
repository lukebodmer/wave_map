import tomli
import toml
import shutil
import numpy as np
from pathlib import Path
from importlib import resources

from wave_map.batch_runner.input_parser import BatchInputParser
from wave_map.simulator.input_parser import SimulationInputParser
from wave_map.simulator.time_step_size_calculator import TimeStepSizeCalculator
from wave_map.batch_runner.parameter_file_generator import ParameterFileGenerator
from wave_map.batch_runner.logger import Logger
from wave_map.hash_functions.parameter_hashing import ParameterHashFunctions
from wave_map.simulator.gmsh_mesh_generator import GmshMeshGenerator  # assuming it lives here
from wave_map.simulator.simulation_setup import SimulationSetup


class BatchRunnerSetup:
    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.params = self._load_batch_parameters()

        self.batch_name = self.params.general.batch_name
        self.base_output_dir = Path(f"data/simulation_batch_data/{self.batch_name}")
        self.base_config_path = self._resolve_config_path(self.params.general.base_config_path)
        self.parameter_files_dir = self.base_output_dir / "parameter_files"
        self.mesh_output_dir = Path("data/inputs/meshes")

        self.logger = Logger(self.base_output_dir / "log.txt")
        self.prepare_output_dirs()
        self.save_copy_of_config_files()

        self._generate_parameter_files_if_needed()
        self.mesh_info = self._generate_gmsh_files_if_needed()

        self.completed_all_training_simulations = False
        self.unsimulated_hashes = []
        self.check_completed_simulations()

    def save_copy_of_config_files(self):
        # --- Save the original emulator_parameters.toml into the batch folder ---
        shutil.copy2(self.config_path, self.base_output_dir / self.config_path.name)
        shutil.copy2(self.base_config_path, self.base_output_dir / "base_parameters.toml")

    def _load_batch_parameters(self):
        with open(self.config_path, "rb") as f:
            config = tomli.load(f)

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

    def prepare_output_dirs(self):
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

    def _generate_parameter_files_if_needed(self):
        n_expected = self.params.general.number_initial_parameter_files_to_create

        if self.parameter_files_dir.exists() and len(list(self.parameter_files_dir.glob("*.toml"))) >= n_expected:
            self.logger.info("Parameter files already exist. Skipping generation.")
            return

        self.logger.info(f"Generating {n_expected} parameter files...")

        # Resolve the base config path using the new method
        
        with open(self.base_config_path, "rb") as f:
            base_config = tomli.load(f)
        domain_size = base_config["mesh"]["box_size"]

        parameter_file_generator = ParameterFileGenerator(
            base_config_path=str(self.base_config_path),  # Convert to string if needed
            batch_name=self.batch_name,
            inclusion_density_range=tuple(self.params.inclusion.inclusion_density_range),
            inclusion_speed_range=tuple(self.params.inclusion.inclusion_wave_speed_range),
            inclusion_scaling_range=tuple(self.params.inclusion.inclusion_scaling_range),
            allow_inclusion_to_rotate=self.params.inclusion.allow_inclusion_to_rotate,
            allow_inclusion_to_move=self.params.inclusion.allow_inclusion_to_move,
            boundary_buffer=self.params.geometry.boundary_buffer,
            domain_size=domain_size,
        )

        parameter_file_generator.create_parameter_files(n_samples=n_expected)

    def _generate_gmsh_files_if_needed(self):
        """
        Collect all unique mesh hashes and generate missing meshes.
        Returns:
            dict: {mesh_hash: {"smallest_radii": [...],
                               "max_wave_speed": float,
                               "polynomial_order": int}}
        Also computes the minimum dt across all meshes and saves it to batch_metadata.toml
        """
        mesh_info = {}
        parameter_files_dir = self.base_output_dir / "parameter_files"
    
        # Step 1: Open each parameter file, collect mesh hash + max wave speed
        for toml_file in parameter_files_dir.glob("*.toml"):
            with open(toml_file, "rb") as f:
                cfg = tomli.load(f)
    
            parser = SimulationInputParser.from_toml(cfg)
    
            hash_functions = ParameterHashFunctions()
            mesh_hash = hash_functions.get_mesh_hash(parser)

            poly_order = parser.solver.polynomial_order
    
            mat = cfg["material"]
            max_wave_speed = max(mat["inclusion_wave_speed"], mat["outer_wave_speed"])
    
            if mesh_hash not in mesh_info:
                mesh_info[mesh_hash] = {
                    "max_wave_speed": max_wave_speed,
                    "polynomial_order": poly_order,
                    "smallest_radii": None,
                    "param_file": toml_file
                }
    
        # Step 2: Generate missing meshes and record smallest radii
        hash_functions = ParameterHashFunctions()

        for mesh_hash, info in mesh_info.items():
            mesh_dir = self.mesh_output_dir / mesh_hash
            if not mesh_dir.exists():
                self.logger.info(f"Generating mesh for hash {mesh_hash}...")
                with open(info["param_file"], "rb") as f:
                    cfg = tomli.load(f)
    
                parser = SimulationInputParser.from_toml(cfg)
    
                gmsh_gen = GmshMeshGenerator(parser, mesh_hash)
                gmsh_gen.generate_ellipsoid_geometry()

                # get smalelst radii
                info["smallest_radii"] = gmsh_gen.get_smallest_radii()
                
                # get corresponding simulation hash
                sim_hash = hash_functions.get_simulation_hash(info["param_file"])
                info["simulation_hash"] = sim_hash
            else:
                self.logger.info(f"Mesh for hash {mesh_hash} already exists.")
                # Optionally read from mesh metadata if it exists
                mesh_meta_file = mesh_dir / "mesh_metadata.toml"
                if mesh_meta_file.exists():
                    with open(mesh_meta_file, "r") as f:
                        for line in f:
                            if line.startswith("smallest_radii"):
                                vals = line.split("=")[1].strip().strip("[]")
                                info["smallest_radii"] = [float(v) for v in vals.split(",") if v.strip()]
                                break
    
        # Step 3: Compute dt for each mesh and find global min
        min_dt = None
        for mesh_hash, info in mesh_info.items():
            r = info["smallest_radii"]
            c = info["max_wave_speed"]
            p = info["polynomial_order"]
    
            calc = TimeStepSizeCalculator(max_wave_speed=c, smallest_radii=r, polynomial_order=p)
            dt_mesh = calc.calculate_cfl_dt()
            if min_dt is None or dt_mesh < min_dt:
                min_dt = dt_mesh
    
        self.min_dt = min_dt

        # Step 4: Save batch metadata to TOML
        meta_file = self.base_output_dir / "batch_metadata.toml"
        
        # Prepare the data structure
        data = {
                  "min_dt": min_dt,
                  "mesh": {
                      mesh_hash: {
                          "simulation_hash": info["simulation_hash"],
                          "max_wave_speed": info["max_wave_speed"],
                          "polynomial_order": info["polynomial_order"],
                          **({"smallest_radii": info["smallest_radii"]} if info["smallest_radii"] else {})
                      }
                      for mesh_hash, info in mesh_info.items()
                  }
                     }
        
        # Write to file
        with open(meta_file, "w") as f:
                  toml.dump(data, f)

        self.logger.info(f"Computed global minimum dt = {min_dt:.6e} and saved to {meta_file}")
    
        # Remove param_file before returning
        for v in mesh_info.values():
            v.pop("param_file", None)
    
        return mesh_info

    def check_completed_simulations(self):
        outputs_dir = Path(f"data/simulation_batch_data/{self.batch_name}/simulations")

        if not self.base_output_dir.exists():
            self.logger.info(f"Base output directory {self.base_output_dir} does not exist.")
            return

        if not self.parameter_files_dir.exists():
            self.logger.info(f"No parameter files found at {self.parameter_files_dir}.")
            return

        parameter_hashes = {
            f.stem for f in self.parameter_files_dir.glob("*.toml") if f.is_file()
        }
        simulated_hashes = {
            d.name for d in outputs_dir.iterdir() if d.is_dir()
        } if outputs_dir.exists() else set()

        missing_runs = parameter_hashes - simulated_hashes
        if missing_runs:
            self.completed_all_training_simulations = False
            self.unsimulated_hashes = sorted(missing_runs)
            self.logger.info("Found training files that have not been simulated.")
        else:
            self.completed_all_training_simulations = True
            self.logger.info("All simulations completed.")

    def run(self):
        if not self.unsimulated_hashes:
            self.logger.info("No simulations to run.")
            return

        self.logger.info(f"Running {len(self.unsimulated_hashes)} missing simulations...")

        for hash in self.unsimulated_hashes:
            parameter_file = self.parameter_files_dir / f"{hash}.toml"
            if not parameter_file.exists():
                self.logger.info(f"Parameter file {parameter_file} not found. Skipping.")
                continue

            try:
                self.logger.info(f"Running simulation {hash}")
                setup = SimulationSetup(dt=self.min_dt, config_path=parameter_file, batch_name=self.batch_name)
                sim = setup.build_simulator()
                sim.run()
                self.logger.info(f"Completed simulation for: {hash}")
            except Exception as e:
                self.logger.info(f"Simulation failed for {hash}: {e}")
