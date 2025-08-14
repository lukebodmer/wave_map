import argparse
from pathlib import Path
from importlib import resources

from wave_map.batch_runner.parameter_file_generator import ParameterFileGenerator
from wave_map.batch_runner.batch_runner_setup import BatchRunnerSetup


def get_default_config(filename="emulator_parameters.toml") -> Path:
    try:
        return resources.files("wave_map.config") / filename
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find {filename} in wave_map.config")


def main(parameter_file: Path):
    if not parameter_file.exists():
        raise FileNotFoundError(f"Config file not found: {parameter_file}")

    print(f"Using config: {parameter_file}")
    batch_runner = BatchRunnerSetup(config_path=str(parameter_file))
    batch_runner.run()
    # TODO: test results, create PPE, test PPE


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the emulator with the specified parameter file."
    )
    parser.add_argument(
        "parameter_file",
        type=Path,
        nargs="?",
        default=get_default_config(),
        help="Path to the emulator parameter file "
             "(default: bundled emulator_parameters.toml in wave_map.config)"
    )
    args = parser.parse_args()
    main(args.parameter_file)
