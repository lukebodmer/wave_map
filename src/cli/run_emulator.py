#!/usr/bin/env python3
import argparse
from pathlib import Path
import tomllib

from emulator.parameter_file_generator import ParameterFileGenerator
from emulator.emulation_setup import EmulationSetup

# Path to default config in the project root's config/ directory
DEFAULT_CONFIG = Path(__file__).parent.parent / "config" / "emulator_parameters.toml"

def main(parameter_file: Path):
    if not parameter_file.exists():
        raise FileNotFoundError(f"Config file not found: {parameter_file}")

    print(f"Using config: {parameter_file}")
    emulator = EmulationSetup(config_path=parameter_file)
    emulator.run()
    # TODO: test results, create PPE, test PPE

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the emulator with the specified parameter file."
    )
    parser.add_argument(
        "parameter_file",
        type=Path,
        nargs="?",
        default=DEFAULT_CONFIG,
        help=f"Path to the emulator parameter file (default: {DEFAULT_CONFIG})"
    )
    args = parser.parse_args()
    main(args.parameter_file)
