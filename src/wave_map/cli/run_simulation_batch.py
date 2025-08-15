import argparse
from pathlib import Path
from importlib import resources

from wave_map.batch_runner.batch_runner_setup import BatchRunnerSetup

# Constants
DEFAULT_CONFIG_FILENAME = "emulator_parameters.toml"

def get_default_config(filename=DEFAULT_CONFIG_FILENAME) -> Path:
    """Return the path to the bundled default configuration file."""
    try:
        return resources.files("wave_map.config") / filename
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find {filename} in wave_map.config")


def parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI arguments for running a simulation batch."""
    parser = argparse.ArgumentParser(
        description="Run the emulator with the specified parameter file."
    )
    parser.add_argument(
        "parameter_file",
        type=Path,
        nargs="?",
        default=get_default_config(),
        help=(
            "Path to the emulator parameter file "
            f"(default: bundled {DEFAULT_CONFIG_FILENAME} in wave_map.config)"
        ),
    )
    return parser.parse_args(argv)


def run_simulation_batch(parameter_file: Path):
    """Run a batch of simulations given a parameter file path."""
    if not parameter_file.exists():
        raise FileNotFoundError(f"Config file not found: {parameter_file}")

    print(f"Using config: {parameter_file}")
    batch_runner = BatchRunnerSetup(config_path=str(parameter_file))
    batch_runner.run()


def main(argv=None):
    args = parse_args(argv)
    run_simulation_batch(args.parameter_file)


if __name__ == "__main__":
    main()
