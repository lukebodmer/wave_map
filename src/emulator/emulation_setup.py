import tomli
import numpy as np
from pathlib import Path
import random

from emulator.input_parser import InputParser
from emulator.data_extractor import DataExtractor  # Adjust if import path differs
from emulator.parameter_file_generator import ParameterFileGenerator
from emulator.logger import Logger  # Ensure you have a Logger class
from wave_simulator.simulation_setup import SimulationSetup

import os

class EmulationSetup:

    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.cfg = self._load_config()

        self.run_family_name = self.cfg.general.run_family_name
        self.base_output_dir = Path(f"data/emulator_data/{self.run_family_name}")

        self.test_set_percentage = 10  # can be parameterized if needed
        self.test_hashes_file = self.base_output_dir / "test_hashes.txt"

        self.logger = Logger(self.base_output_dir/ "log.txt")
        self.prepare_output_dirs()

        self._generate_parameter_files_if_needed()
        self.completed_all_training_simulations = False
        self.unsimulated_hashes = []

        self.check_completed_simulations()

    def _load_config(self):
        with open(self.config_path, "rb") as f:
            config = tomli.load(f)

        parser = InputParser()
        parser.load_from_toml(config)
        return parser

    def prepare_output_dirs(self):
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

    def create_test_set(self):
        parameter_files_dir = self.base_output_dir / "parameter_files"
        hashes = sorted(f.stem for f in parameter_files_dir.glob("*.toml"))
        n_total = len(hashes)
        n_test = max(1, round(n_total * self.test_set_percentage / 100))  # at least 1

        random.seed(42)  # For reproducibility
        test_hashes = sorted(random.sample(hashes, n_test))
        with open(self.test_hashes_file, "w") as f:
            for h in test_hashes:
                f.write(h + "\n")
    
        self.logger.info(f"Selected {n_test} test simulations out of {n_total} total.")

    def _generate_parameter_files_if_needed(self):
        parameter_files_dir = self.base_output_dir / "parameter_files"
        n_expected = self.cfg.general.number_initial_parameter_files_to_create

        if parameter_files_dir.exists() and len(list(parameter_files_dir.glob("*.toml"))) >= n_expected:
            self.logger.info("Parameter files already exist. Skipping generation.")
            return

        self.logger.info(f"Generating {n_expected} parameter files...")

        # Load the base_config to get the domain_size (mesh.box_size)
        base_config_path = self.cfg.general.base_config_path
        try:
            with open(base_config_path, "rb") as f:
                base_config = tomli.load(f)
            domain_size = base_config["mesh"]["box_size"]
        except Exception as e:
            self.logger.info(f"Failed to read domain size from base_config_path: {e}")
            domain_size = 0.25  # default fallback

        gen = ParameterFileGenerator(
            base_config_path=base_config_path,
            run_family_name=self.run_family_name,
            inclusion_density_range=tuple(self.cfg.inclusion.inclusion_density_range),
            inclusion_speed_range=tuple(self.cfg.inclusion.inclusion_wave_speed_range),
            inclusion_radius_range=tuple(self.cfg.inclusion.inclusion_radius_range),
            allow_inclusion_to_move=self.cfg.inclusion.allow_inclusion_to_move,
            boundary_buffer=self.cfg.geometry.boundary_buffer,
            domain_size=domain_size,
        )

        gen.create_parameter_files(n_samples=n_expected)
        self.create_test_set()

    def check_completed_simulations(self):
        parameter_files_dir = self.base_output_dir / "parameter_files"
        outputs_dir = Path(f"data/outputs/{self.run_family_name}")

        if not self.base_output_dir.exists():
            self.logger.info(f"Base output directory {self.base_output_dir} does not exist.")
            return

        if not parameter_files_dir.exists():
            self.logger.info(f"No parameter files found at {parameter_files_dir}.")
            return

        # List hashes from parameter file names (strip .toml extension)
        parameter_hashes = {
            f.stem for f in parameter_files_dir.glob("*.toml") if f.is_file()
        }

        # List directories in the output path (assumed to be run directories)
        simulated_hashes = {
            d.name for d in outputs_dir.iterdir() if d.is_dir()
        } if outputs_dir.exists() else set()

        # Compare
        missing_runs = parameter_hashes - simulated_hashes

        if missing_runs:
            self.completed_all_training_simulations = False
            self.unsimulated_hashes = sorted(missing_runs)
            self.logger.info("Found training files that have not been simulated:")
        else:
            self.completed_all_training_simulations = True
            self.logger.info("All simulations completed.")

    def gather_data(self):
        self.logger.info("Gathering training/test data from simulations...")
        extractor = DataExtractor(
            run_family_name=self.run_family_name,
            test_hashes_file=self.test_hashes_file
        )
        X_train, Y_train, X_test, Y_test = extractor.extract()
    
        np.save(self.base_output_dir / "ppe_inputs_train.npy", X_train)
        np.save(self.base_output_dir / "ppe_outputs_train.npy", Y_train)
        np.save(self.base_output_dir / "ppe_inputs_test.npy", X_test)
        np.save(self.base_output_dir / "ppe_outputs_test.npy", Y_test)
    
        self.logger.info("Saved train/test datasets: X_train.npy, Y_train.npy, X_test.npy, Y_test.npy.")

    def run(self):
        if not self.unsimulated_hashes:
            self.logger.info("No simulations to run.")
            return

        self.logger.info(f"Running {len(self.unsimulated_hashes)} missing simulations...")
        parameter_files_dir = self.base_output_dir / "parameter_files"

        for h in self.unsimulated_hashes:
            parameter_file = parameter_files_dir / f"{h}.toml"

            if not parameter_file.exists():
                self.logger.info(f"Parameter file {parameter_file} not found. Skipping.")
                continue

            try:
                self.logger.info(f"Running simulation {h}")
                setup = SimulationSetup(config_path=parameter_file, run_family_name=self.run_family_name)
                sim = setup.build_simulator()
                sim.run()
                self.logger.info(f"Completed simulation for: {h}")
            except Exception as e:
                self.logger.info(f"Simulation failed for {h}: {e}")

        self.gather_data()
